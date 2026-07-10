#!/bin/bash
# Multi-round agentic GRPO training for Qwen3-4B on 2x H100 (80 GB HBM),
# WITH periodic eval on Senlimulin/2026UCSDIntern_SlimeRL_training_dataset validation.
# - DAPO-style: asymmetric clipping (low=0.2, high=0.28) + sequence-normalized
#   loss (--calculate-per-token-loss). NO dynamic sampling.
# - Collocated rollout + training (--colocate).
# - Eval every --eval-interval rollouts (=every checkpoint save) via
#   examples.agentic_cov.rollout.eval_rollout.
#   Eval prompts come from --llm4cov-eval-dataset-{name,split}; the
#   --eval-prompt-data pair below only exists to satisfy slime's
#   eval_datasets validation — the path is unused by eval_rollout.
# - No KL, no weight decay.
# - 40k total context: rollout response 32k, max packed train tokens 24k/GPU.
# - HF checkpoints and training log are rsynced to REMOTE_SYNC_BASE under a
#   per-run timestamped subdirectory after each checkpoint save.
#
# Run from the slime repo root (e.g. /root/slime in the docker image):
#   bash examples/agentic_cov/run-qwen3-4B-4xH100.sh
#
# Required env:
#   EDA_SERVER     SSH alias of the llm4cov_eda worker
#   EDA_REPO_DIR   path to the llm4cov_eda checkout on that host
# Optional overrides:
#   REMOTE_SYNC_BASE   rsync destination root (default: paladin rl results dir)
#   REMOTE_SYNC_SSH_KEY  path to SSH identity file for rsync (default: auto)
# Other overrides documented in examples/agentic_cov/README.md.

set -ex

# -------------------- script-level flags --------------------
# Usage: bash run-qwen3-4B-4xH100.sh [--offload] [--eda-log-feedback-train] [--eda-log-feedback-eval]
#                                     [--use-uncovered-log] [--use-uncovered-reward --div-lam V]
#                                     [--interval N] [--steps N]
#                                     [--train-dataset NAME] [--eval-dataset NAME]
#                                     [--no-final-save|--save-final-rollout]
#   --offload                Enable --offload-rollout: SGLang offloads model weights to
#                            CPU during training phase, freeing ~4 GB/GPU for Megatron.
#                            Off by default.
#   --eda-log-feedback-train|--elft
#                            Fetch full block/expression/FSM uncovered detail from xrun
#                            during training rollouts.  Slows down each training step.  Off by default.
#   --eda-log-feedback-eval|--elfe
#                            Fetch full block/expression/FSM uncovered detail from xrun
#                            during eval rollouts.  Off by default.
#   --use-uncovered-log|--uul
#                            When EDA log feedback is on, format the tool-feedback from the
#                            structured cov_info["uncovered"] (compact per-bin) instead of the
#                            raw truncated IMC detail text.  No effect without --elft/--elfe.
#                            Off by default.
#   --use-uncovered-reward|--uur
#                            Add a group-level diversity bonus to the reward:
#                            reward = coverage_score + div_lam*diversity. Requires --elft
#                            (uncovered data) AND --div-lam. No-op without --elft.
#   --div-lam V              Diversity reward weight lambda. Required with --elft + --uur.
#   --interval N             Checkpoint save + eval interval in rollout steps (default: 50).
#                            Overrides the CKPT_INTERVAL env var.
#   --steps N                Total number of rollout steps to train (default: 300).
#                            Overrides the NUM_ROLLOUT env var.
#   --train-dataset NAME     HuggingFace dataset name for training rollouts
#                            (default: Senlimulin/CodeV_R1_5918_dataset).
#                            Overrides the LLM4COV_DATASET env var.
#   --dataset-step-offset N  Skip this many same-seed rollout steps in the
#                            training dataset order before starting. Formal OPD
#                            defaults to 1000 so step n uses the prompt batch
#                            that step n+1000 would have used.
#   --eval-dataset  NAME     HuggingFace dataset name for eval rollouts
#                            (default: Senlimulin/2026UCSDIntern_SlimeRL_training_dataset).
#                            Overrides the LLM4COV_EVAL_DATASET env var.
#   --no-verify-student-step999
#                            Disable sha256 verification that MODEL_NAME resolves
#                            to the expected stage2 step999 HF checkpoint.
#   --batch-size N           Number of task prompts per rollout step. Smoke uses 1;
#                            full runs can use 4. Internal slime rollout_batch_size
#                            is N * num_agentic_rounds.
#   --no-final-save          Do not force a checkpoint on the final rollout if the
#                            checkpoint interval has not fired. This is auto-enabled
#                            for short smoke tests where --steps < --interval.
#   --save-final-rollout     Force the legacy behavior: always save on the final
#                            rollout even when --steps < --interval.
#   --n-student N            Number of student rollouts per prompt. Smoke uses 2.
OFFLOAD=0
EDA_LOG_FEEDBACK_TRAIN=${EDA_LOG_FEEDBACK_TRAIN:-1}
EDA_LOG_FEEDBACK_EVAL=${EDA_LOG_FEEDBACK_EVAL:-1}
USE_UNCOVERED_LOG=0
USE_UNCOVERED_REWARD=0
DIV_LAM=""
SKIP_EVAL_BEFORE_TRAIN=0
OVERRIDE_OPT_PARAM_SCHEDULER=${OVERRIDE_OPT_PARAM_SCHEDULER:-0}
OPD_ARGS=()
OPD_POLL=${OPD_POLL:-1}
OPD_POLL_SPECIFIED=0
OPD_ALGORITHM=${OPD_ALGORITHM:-vopd_topk}
OPD_ALGORITHM_SPECIFIED=0
NO_FINAL_SAVE=${NO_FINAL_SAVE:-auto}
while [ $# -gt 0 ]; do
    case "$1" in
        --offload)                OFFLOAD=1 ;;
        --eda-log-feedback-train|--elft) EDA_LOG_FEEDBACK_TRAIN=${EDA_LOG_FEEDBACK_TRAIN:-1} ;;
        --eda-log-feedback-eval|--elfe)  EDA_LOG_FEEDBACK_EVAL=${EDA_LOG_FEEDBACK_EVAL:-1} ;;
        --use-uncovered-log|--uul)       USE_UNCOVERED_LOG=1 ;;
        --use-uncovered-reward|--uur)    USE_UNCOVERED_REWARD=1 ;;
        --div-lam)                DIV_LAM="${2:?--div-lam requires a value}"; shift ;;
        --div-lam=*)              DIV_LAM="${1#--div-lam=}" ;;
        --skip-eval-before-train|--skip-eval) SKIP_EVAL_BEFORE_TRAIN=1 ;;
        --override-opt-param-scheduler|--override-optimizer-scheduler)
                                  OVERRIDE_OPT_PARAM_SCHEDULER=1 ;;
        --interval)               CKPT_INTERVAL="${2:?--interval requires a value}"; shift ;;
        --interval=*)             CKPT_INTERVAL="${1#--interval=}" ;;
        --steps)                  NUM_ROLLOUT="${2:?--steps requires a value}"; shift ;;
        --steps=*)                NUM_ROLLOUT="${1#--steps=}" ;;
        --train-dataset)          LLM4COV_DATASET="${2:?--train-dataset requires a value}"; shift ;;
        --train-dataset=*)        LLM4COV_DATASET="${1#--train-dataset=}" ;;
        --dataset-step-offset|--llm4cov-dataset-step-offset)
                                  LLM4COV_DATASET_STEP_OFFSET="${2:?$1 requires a value}"; shift ;;
        --dataset-step-offset=*|--llm4cov-dataset-step-offset=*)
                                  LLM4COV_DATASET_STEP_OFFSET="${1#*=}" ;;
        --eval-dataset)           LLM4COV_EVAL_DATASET="${2:?--eval-dataset requires a value}"; shift ;;
        --eval-dataset=*)         LLM4COV_EVAL_DATASET="${1#--eval-dataset=}" ;;
        --no-verify-student-step999)
                                  VERIFY_STUDENT_STAGE2_STEP999=0 ;;
        --batch-size|--prompt-batch-size|--task-batch-size)
                                  PROMPT_BATCH_SIZE="${2:?$1 requires a value}"; shift ;;
        --batch-size=*|--prompt-batch-size=*|--task-batch-size=*)
                                  PROMPT_BATCH_SIZE="${1#*=}" ;;
        --n-student|--n-samples-per-prompt)
                                  N_STUDENT="${2:?$1 requires a value}"; shift ;;
        --no-final-save|--no-save-final-rollout)
                                  NO_FINAL_SAVE=1 ;;
        --save-final-rollout|--final-save)
                                  NO_FINAL_SAVE=0 ;;
        --n-student=*|--n-samples-per-prompt=*)
                                  N_STUDENT="${1#*=}" ;;
        --use-opd-relay|--opd-score-student-rollouts)
                                  OPD_ARGS+=("$1") ;;
        --opd-algorithm)          OPD_ALGORITHM="${2:?--opd-algorithm requires a value}"; OPD_ALGORITHM_SPECIFIED=1; OPD_ARGS+=("$1" "${OPD_ALGORITHM}"); shift ;;
        --opd-algorithm=*)        OPD_ALGORITHM="${1#--opd-algorithm=}"; OPD_ALGORITHM_SPECIFIED=1; OPD_ARGS+=("$1") ;;
        --opd-poll)               OPD_POLL="${2:?--opd-poll requires a value}"; OPD_POLL_SPECIFIED=1; OPD_ARGS+=("$1" "${OPD_POLL}"); shift ;;
        --opd-poll=*)             OPD_POLL="${1#--opd-poll=}"; OPD_POLL_SPECIFIED=1; OPD_ARGS+=("$1") ;;
        --opd-teachers|--opd-lambda|--opd-topk|--opd-student-topk-mode|--opd-gate-eps|--opd-timeout|--opd-namespace|--opd-server|--opd-transport|--opd-http-url|--opd-xfer-dir|--opd-sftp-host|--opd-sftp-port|--opd-sftp-user|--opd-sftp-key)
                                  OPD_ARGS+=("$1" "${2:?$1 requires a value}"); shift ;;
        --opd-teachers=*|--opd-lambda=*|--opd-topk=*|--opd-student-topk-mode=*|--opd-gate-eps=*|--opd-timeout=*|--opd-namespace=*|--opd-server=*|--opd-transport=*|--opd-http-url=*|--opd-xfer-dir=*|--opd-sftp-host=*|--opd-sftp-port=*|--opd-sftp-user=*|--opd-sftp-key=*)
                                  OPD_ARGS+=("$1") ;;
        *) echo "[run] Unknown argument: $1" >&2; exit 1 ;;
    esac
    shift
done
if [ "${#OPD_ARGS[@]}" -gt 0 ] && [ "${OPD_POLL_SPECIFIED}" = "0" ]; then
    OPD_ARGS+=(--opd-poll "${OPD_POLL}")
fi
if [ "${#OPD_ARGS[@]}" -gt 0 ] && [ "${OPD_ALGORITHM_SPECIFIED}" = "0" ]; then
    OPD_ARGS+=(--opd-algorithm "${OPD_ALGORITHM}")
fi

# clean up stale ray / sglang / python from prior runs
pkill -9 sglang 2>/dev/null || true
sleep 2
ray stop --force 2>/dev/null || true
pkill -9 ray 2>/dev/null || true
pkill -9 python 2>/dev/null || true
sleep 2

export PYTHONBUFFERED=1

export CUDA_VISIBLE_DEVICES=0,1

# -------------------- topology --------------------
NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
HAS_NVLINK=$([ "$NVLINK_COUNT" -gt 0 ] && echo 1 || echo 0)

DETECTED_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l | tr -d " ")
[ -n "${CUDA_VISIBLE_DEVICES}" ] && DETECTED_GPUS=$(echo "${CUDA_VISIBLE_DEVICES}" | tr "," "\n" | wc -l | tr -d " ")
NUM_GPUS=${NUM_GPUS:-${DETECTED_GPUS:-4}}
if [ "$NUM_GPUS" -lt 2 ]; then
    echo "ERROR: this recipe targets 2 GPUs (TP=2, CP=1). Got NUM_GPUS=$NUM_GPUS." >&2
    exit 1
fi

echo "[run] ─── GPU INVENTORY ──────────────────────────────────────"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader | \
    awk -F',' '{printf "[run]   GPU%s:%s (%s)\n", $1, $2, $3}'
echo "[run] NUM_GPUS=${NUM_GPUS}  HAS_NVLINK=${HAS_NVLINK}"
echo "[run] ──────────────────────────────────────────────────────────"

# -------------------- required external state --------------------
: "${EDA_SERVER:?Set EDA_SERVER to the SSH alias of the llm4cov_eda worker}"
: "${EDA_REPO_DIR:?Set EDA_REPO_DIR to the path of llm4cov_eda on that worker}"

MODEL_NAME=${MODEL_NAME:-Senlimulin/2026UCSDIntern_Stage2_elft_elfe}
ROOT_DIR=${ROOT_DIR:-$(pwd)}
LLM4COV_DATASET=${LLM4COV_DATASET:-Senlimulin/CodeV_R1_5918_dataset}
LLM4COV_SPLIT=${LLM4COV_SPLIT:-train}
LLM4COV_DATASET_STEP_OFFSET=${LLM4COV_DATASET_STEP_OFFSET:-1000}
LLM4COV_EVAL_DATASET=${LLM4COV_EVAL_DATASET:-Senlimulin/2026UCSDIntern_SlimeRL_training_dataset}
LLM4COV_EVAL_SPLIT=${LLM4COV_EVAL_SPLIT:-validation}
NUM_ROLLOUT=${NUM_ROLLOUT:-300}
CKPT_INTERVAL=${CKPT_INTERVAL:-50}   # shared interval for --save-interval and --eval-interval
NUM_AGENTIC_ROUNDS=${NUM_AGENTIC_ROUNDS:-2}
PROMPT_BATCH_SIZE=${PROMPT_BATCH_SIZE:-4}
N_STUDENT=${N_STUDENT:-4}
VERIFY_STUDENT_STAGE2_STEP999=${VERIFY_STUDENT_STAGE2_STEP999:-1}
ROLLOUT_BATCH_SIZE=$((PROMPT_BATCH_SIZE * NUM_AGENTIC_ROUNDS))
GLOBAL_BATCH_SIZE=$((ROLLOUT_BATCH_SIZE * N_STUDENT))

# -------------------- remote sync --------------------
# HF checkpoints and training log are rsynced to REMOTE_SYNC_BASE under a
# per-run subdirectory named after the model + timestamp, so successive runs
# never overwrite each other.  Set REMOTE_SYNC_SSH_KEY to an SSH identity
# file when the host requires an explicit key (e.g. when running on brev,
# use /home/nvidia/.ssh/id_paladin).  Leave empty to rely on the SSH agent.
# Sync target: "hf" (upload to a private HF repo, default) or "paladin" (rsync).
REMOTE_SYNC_TARGET=${REMOTE_SYNC_TARGET:-hf}
# HF mode: repo = ${HF_SYNC_REPO_PREFIX}${RUN_SUBDIR} (private). The upload uses a
# dedicated WRITE token from env HF_SYNC_TOKEN (kept separate from HF_TOKEN, which
# the Makefile sets for model download). Pass -e HF_SYNC_TOKEN=<write> at launch.
HF_SYNC_REPO_PREFIX=${HF_SYNC_REPO_PREFIX:-"Senlimulin/2026UCSDIntern_"}
# paladin (rsync) fallback config:
REMOTE_SYNC_BASE=${REMOTE_SYNC_BASE:-"slu375@paladin.ucsd.edu:/mnt/raid0_ssd/sheng/brev_result/rl"}
REMOTE_SYNC_SSH_KEY=${REMOTE_SYNC_SSH_KEY:-""}
_HF_SYNC_TOKEN_XTRACE=0
case "$-" in
    *x*) _HF_SYNC_TOKEN_XTRACE=1; set +x ;;
esac
if [ "${REMOTE_SYNC_TARGET}" = "hf" ] && [ -z "${HF_SYNC_TOKEN:-}" ]; then
    echo "ERROR: REMOTE_SYNC_TARGET=hf requires a write token in HF_SYNC_TOKEN (pass -e HF_SYNC_TOKEN=...)" >&2
    exit 1
fi
if [ "${_HF_SYNC_TOKEN_XTRACE}" = "1" ]; then
    set -x
fi

# Default model is SFT'd from Qwen3-4B-Instruct-2507 (rotary base 5,000,000).
# Override both vars when MODEL_NAME points at a model with a different
# rotary base (e.g. base Qwen3-4B uses 1,000,000).
ROTARY_BASE=${ROTARY_BASE:-5000000}
export MODEL_ARGS_ROTARY_BASE=${MODEL_ARGS_ROTARY_BASE:-5000000}

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
SLIME_ROOT="$(cd "${SCRIPT_DIR}/../.." &>/dev/null && pwd)"
if [ "$(pwd)" != "${SLIME_ROOT}" ]; then
    echo "ERROR: run this script from the slime repo root (${SLIME_ROOT}), got $(pwd)" >&2
    exit 1
fi
source "${SLIME_ROOT}/scripts/models/qwen3-4B.sh"

# Disable Megatron's vocab padding. With TP=2 and the default 128, vocab
# is rounded 151936 -> 152064; the HF save then trips vllm's
# `loaded_weight.shape[0] == config.vocab_size` assert. 1 keeps padding
# at TP=2 only (151936 is already even), so HF dumps load in vllm as-is.
MODEL_ARGS+=(--make-vocab-size-divisible-by 1)

# Derives HF_CKPT / REF_LOAD / SAVE_DIR from MODEL_NAME + ROOT_DIR, and runs
# first-time HF download / torch_dist conversion if those dirs are missing.
source "${SCRIPT_DIR}/_setup_checkpoints.sh"

_verify_student_stage2_step999() {
    local shard1="${HF_CKPT}/model-00001-of-00002.safetensors"
    local shard2="${HF_CKPT}/model-00002-of-00002.safetensors"
    local expect1="fc5216698e9c048c70dcaecf53f0a93c0db82aba6b171fc77871edbe62a39590"
    local expect2="187142a76e287a0e398a4bf1047f01a1e4aef84317350594ddf6064d05920dc6"
    local expect_tensor_manifest="0188da438c4f625e7f5771afc0adb4a968d7639eaf0aee3745684acfde6d78f4"
    local got1 got2
    if [ -f "${shard1}" ] && [ -f "${shard2}" ]; then
        got1=$(sha256sum "${shard1}" | awk '{print $1}')
        got2=$(sha256sum "${shard2}" | awk '{print $1}')
        if [ "${got1}" = "${expect1}" ] && [ "${got2}" = "${expect2}" ]; then
            echo "[setup] verified student checkpoint matches stage2_step999 shard sha256"
            return 0
        fi
        echo "[setup] shard sha256 did not match Paladin backup; checking tensor manifest" >&2
        echo "        ${shard1}: got ${got1}, expected ${expect1}" >&2
        echo "        ${shard2}: got ${got2}, expected ${expect2}" >&2
    else
        echo "[setup] Paladin shard filenames not present; checking tensor manifest" >&2
    fi

    local tensor_count tensor_manifest
    read -r tensor_count tensor_manifest < <(python3 - "${HF_CKPT}" <<'PY'
import glob
import hashlib
import json
import sys

import torch
from safetensors import safe_open

root = sys.argv[1]
items = []
for path in sorted(glob.glob(f"{root}/model-*.safetensors")):
    with safe_open(path, framework="pt", device="cpu") as f:
        for key in f.keys():
            tensor = f.get_tensor(key).contiguous()
            raw = tensor.view(torch.uint8).numpy().tobytes()
            items.append((key, list(tensor.shape), str(tensor.dtype), hashlib.sha256(raw).hexdigest()))
items.sort(key=lambda x: x[0])
digest = hashlib.sha256()
for item in items:
    digest.update(json.dumps(item, separators=(",", ":")).encode("utf-8") + b"\n")
print(len(items), digest.hexdigest())
PY
)
    if [ "${tensor_count}" != "398" ] || [ "${tensor_manifest}" != "${expect_tensor_manifest}" ]; then
        echo "ERROR: student checkpoint is not Paladin stage2_step999." >&2
        echo "       tensor_count=${tensor_count}, expected 398" >&2
        echo "       tensor_manifest=${tensor_manifest}, expected ${expect_tensor_manifest}" >&2
        echo "       Set VERIFY_STUDENT_STAGE2_STEP999=0 only for intentional non-formal smoke/debug runs." >&2
        exit 1
    fi
    echo "[setup] verified student checkpoint matches stage2_step999 tensor manifest sha256"
}

if [ "${VERIFY_STUDENT_STAGE2_STEP999}" = "1" ]; then
    _verify_student_stage2_step999
fi

# RUN_SUBDIR and LOCAL_LOG are set here (after SAVE_DIR is available) so the
# subdirectory name incorporates the actual checkpoint base name.
_TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_DIR="${ROOT_DIR}/${_MODEL_BASENAME}_${_TIMESTAMP}"
mkdir -p "${RUN_DIR}"
RUN_SUBDIR=${RUN_SUBDIR:-"${_MODEL_BASENAME}_${_TIMESTAMP}"}
HF_SYNC_REPO="${HF_SYNC_REPO_PREFIX}${RUN_SUBDIR}"
LOCAL_LOG="${RUN_DIR}/main.log"
GPU_MONITOR_ENABLED=${GPU_MONITOR_ENABLED:-1}
GPU_MONITOR_INTERVAL_SEC=${GPU_MONITOR_INTERVAL_SEC:-2}
GPU_MONITOR_LOG="${RUN_DIR}/gpu_memory.csv"
GPU_MONITOR_SUMMARY="${RUN_DIR}/gpu_memory_summary.txt"
GPU_MONITOR_PID=""
echo "[run] $(date '+%Y-%m-%d %H:%M:%S') starting run" | tee -a "${LOCAL_LOG}"

echo "[run] ─── PATH SUMMARY ──────────────────────────────────────" | tee -a "${LOCAL_LOG}"
echo "[run] MODEL_NAME   = ${MODEL_NAME}"                            | tee -a "${LOCAL_LOG}"
echo "[run] HF_CKPT      = ${HF_CKPT}"                              | tee -a "${LOCAL_LOG}"
echo "[run] REF_LOAD     = ${REF_LOAD}"                              | tee -a "${LOCAL_LOG}"
echo "[run] SAVE_DIR     = ${SAVE_DIR}"                              | tee -a "${LOCAL_LOG}"
echo "[run] RUN_DIR      = ${RUN_DIR}  (logs + step_N checkpoints)"  | tee -a "${LOCAL_LOG}"
echo "[run] LOCAL_LOG    = ${LOCAL_LOG}"                             | tee -a "${LOCAL_LOG}"
if [ "${REMOTE_SYNC_TARGET}" = "hf" ]; then
echo "[run] SYNC TARGET  = HF private: ${HF_SYNC_REPO}"             | tee -a "${LOCAL_LOG}"
else
echo "[run] SYNC TARGET  = paladin: ${REMOTE_SYNC_BASE}/${RUN_SUBDIR}/" | tee -a "${LOCAL_LOG}"
echo "[run] SSH_KEY      = ${REMOTE_SYNC_SSH_KEY:-'(ssh-agent)'}"   | tee -a "${LOCAL_LOG}"
fi
echo "[run] EDA_SERVER   = ${EDA_SERVER}"                            | tee -a "${LOCAL_LOG}"
echo "[run] EDA_REPO_DIR = ${EDA_REPO_DIR}"                          | tee -a "${LOCAL_LOG}"
echo "[run] RUN_SUBDIR   = ${RUN_SUBDIR}"                            | tee -a "${LOCAL_LOG}"
echo "[run] DATASET_STEP_OFFSET = ${LLM4COV_DATASET_STEP_OFFSET}"     | tee -a "${LOCAL_LOG}"
echo "[run] VERIFY_STUDENT_STAGE2_STEP999 = ${VERIFY_STUDENT_STAGE2_STEP999}" | tee -a "${LOCAL_LOG}"
echo "[run] GPU_MONITOR enabled=${GPU_MONITOR_ENABLED} interval_sec=${GPU_MONITOR_INTERVAL_SEC} log=${GPU_MONITOR_LOG}" | tee -a "${LOCAL_LOG}"
echo "[run] ───────────────────────────────────────────────────────" | tee -a "${LOCAL_LOG}"

NO_FINAL_SAVE_ARGS=()
if [ "${NO_FINAL_SAVE}" = "auto" ]; then
    if [ "${NUM_ROLLOUT}" -lt "${CKPT_INTERVAL}" ]; then
        NO_FINAL_SAVE=1
    else
        NO_FINAL_SAVE=0
    fi
fi
if [ "${NO_FINAL_SAVE}" = "1" ]; then
    NO_FINAL_SAVE_ARGS=(--no-save-final-rollout)
    echo "[run] no-save-final-rollout enabled: final rollout will not force a checkpoint" | tee -a "${LOCAL_LOG}"
else
    echo "[run] final rollout checkpoint behavior: save when interval fires or on final rollout" | tee -a "${LOCAL_LOG}"
fi

# -------------------- checkpoint paths --------------------
CKPT_ARGS=(
   --hf-checkpoint "${HF_CKPT}"
   --ref-load      "${REF_LOAD}"
   --load          "${SAVE_DIR}"
   --save          "${SAVE_DIR}"
   --save-interval "${CKPT_INTERVAL}"
   # HF-format dump alongside the torch_dist save. {rollout_id} is filled
   # in by slime via args.save_hf.format(rollout_id=...).
   --save-hf       "${RUN_DIR}/step_{rollout_id}"
   "${NO_FINAL_SAVE_ARGS[@]}"
)

# -------------------- rollout / batching --------------------
# group_size = n_samples_per_prompt = N_STUDENT
# rollout_batch_size = PROMPT_BATCH_SIZE * NUM_AGENTIC_ROUNDS
# global_batch_size = rollout_batch_size * N_STUDENT
# Default 300 steps total -> --num-rollout 300 (default num_steps_per_rollout=1).
# Override with NUM_ROLLOUT=... to run shorter/longer.
ROLLOUT_ARGS=(
   --rollout-shuffle
   # Pin the rollout RNG seed so every training run shuffles the prompt dataset
   # identically -> step N consumes the same data points across different runs
   # (given the same dataset + tokenizer + max prompt len). Default is already
   # 42; pinned explicitly here so it can never silently drift.
   --rollout-seed           42
   --num-rollout            "${NUM_ROLLOUT}"
   --rollout-batch-size     "${ROLLOUT_BATCH_SIZE}"
   --n-samples-per-prompt   "${N_STUDENT}"
   --rollout-max-response-len 16384
   --rollout-temperature    1.0

   # llm4cov dataset is built from chat messages; apply_chat_template
   # converts list[dict] -> str so sglang's tokenizer.encode accepts it.
   --apply-chat-template

   --global-batch-size      "${GLOBAL_BATCH_SIZE}"
   --balance-data
)

# -------------------- llm4cov agentic config --------------------
AGENTIC_ARGS=(
   --rollout-function-path examples.agentic_cov.rollout.generate_rollout
   --data-source-path      examples.agentic_cov.data_source.LlmCovDataSource
   --num-agentic-rounds       "${NUM_AGENTIC_ROUNDS}"
   --eval-num-agentic-rounds  3
   --llm4cov-dataset-name        "${LLM4COV_DATASET}"
   --llm4cov-dataset-split       "${LLM4COV_SPLIT}"
   --llm4cov-dataset-step-offset "${LLM4COV_DATASET_STEP_OFFSET}"
   --llm4cov-eval-dataset-name   "${LLM4COV_EVAL_DATASET}"
   --llm4cov-eval-dataset-split  "${LLM4COV_EVAL_SPLIT}"
   --eda-server            "${EDA_SERVER}"
   --eda-repo-dir          "${EDA_REPO_DIR}"
)

# -------------------- eval --------------------
# eval_rollout pulls prompts directly from --llm4cov-eval-dataset-{name,split};
# slime still requires --eval-prompt-data (or --eval-config) to populate
# args.eval_datasets. Use the HF dataset id as both name and (unused) path so
# the result key from rollout.py:369 (`args.llm4cov_eval_dataset_name`) lines
# up with the eval_datasets entry.
EVAL_ARGS=(
   --eval-interval              "${CKPT_INTERVAL}"
   --eval-function-path         examples.agentic_cov.rollout.eval_rollout
   --eval-prompt-data           "${LLM4COV_EVAL_DATASET}" "${LLM4COV_EVAL_DATASET}"
   --n-samples-per-eval-prompt  1
   --eval-max-response-len      16384
   --eval-temperature           0.7
   --eval-top-p                 0.8
)
if [ "${SKIP_EVAL_BEFORE_TRAIN}" = "1" ]; then
    EVAL_ARGS+=(--skip-eval-before-train)
    echo "[run] skip-eval-before-train enabled: no step-0 initial eval" | tee -a "${LOCAL_LOG}"
fi

# -------------------- parallelism / memory --------------------
# 4 GPUs split as TP=2 x CP=2 x PP=1. With CP=2 a 40k sequence is sharded
# to ~20k tokens per CP rank; --max-tokens-per-gpu 24576 leaves headroom for
# packing a few short sequences alongside one long one. Same setting as
# H200 — H200 is not full at 24k, so 80 GB H100 should still fit.
MAX_TOKENS_PER_GPU=${MAX_TOKENS_PER_GPU:-24576}
PERF_ARGS=(
   --tensor-model-parallel-size  2
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size       1
   --expert-model-parallel-size  1
   --expert-tensor-parallel-size 1

   --recompute-granularity full
   --recompute-method      uniform
   --recompute-num-layers  1

   --use-dynamic-batch-size
   --max-tokens-per-gpu    "${MAX_TOKENS_PER_GPU}"
)

# -------------------- DAPO ----------------------
# Asymmetric clipping + token-level (sequence-normalized) loss.
# No dynamic sampling (--dynamic-sampling-filter-path is left unset).
# No KL (omit --use-kl-loss; the flag is store_true with default False),
# no entropy bonus.
GRPO_ARGS=(
   --advantage-estimator     grpo
   --calculate-per-token-loss
   --kl-loss-coef            0.0
   --entropy-coef            0.0
   --eps-clip                0.2
   --eps-clip-high           0.28
)

# -------------------- optimizer --------------------
OPTIMIZER_ARGS=(
   --optimizer       adam
   --lr              1e-6
   --lr-decay-style  constant
   --weight-decay    0.0
   --adam-beta1      0.9
   --adam-beta2      0.95
)

# -------------------- sglang (rollout engine) --------------------
# 2 engines, each TP=2, on the same 4 GPUs as training (--colocate).
SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 2
   --sglang-mem-fraction-static  0.35
)

# -------------------- offload (optional) --------------------
# Activated by passing --offload to this script.
# --offload-rollout: sglang releases model weights during training phase.
OFFLOAD_ARGS=()
if [ "${OFFLOAD}" = "1" ]; then
    OFFLOAD_ARGS=(--offload-rollout)
    echo "[run] offload mode enabled: --offload-rollout" | tee -a "${LOCAL_LOG}"
fi

# -------------------- eda-log-feedback-train / eval (optional) --------
EDA_LOG_FEEDBACK_ARGS=()
if [ "${EDA_LOG_FEEDBACK_TRAIN}" = "1" ]; then
    EDA_LOG_FEEDBACK_ARGS+=(--eda-log-feedback-train)
    echo "[run] EDA log feedback (train) enabled: full xrun detail in training tool-feedback" | tee -a "${LOCAL_LOG}"
fi
if [ "${EDA_LOG_FEEDBACK_EVAL}" = "1" ]; then
    EDA_LOG_FEEDBACK_ARGS+=(--eda-log-feedback-eval)
    echo "[run] EDA log feedback (eval) enabled: full xrun detail in eval tool-feedback" | tee -a "${LOCAL_LOG}"
fi
if [ "${USE_UNCOVERED_LOG}" = "1" ]; then
    EDA_LOG_FEEDBACK_ARGS+=(--use-uncovered-log)
    echo "[run] use-uncovered-log enabled: tool-feedback from structured cov_info[uncovered]" | tee -a "${LOCAL_LOG}"
fi
if [ "${USE_UNCOVERED_REWARD}" = "1" ]; then
    if [ "${EDA_LOG_FEEDBACK_TRAIN}" = "1" ] && [ -z "${DIV_LAM}" ]; then
        echo "[run] ERROR: --use-uncovered-reward with --eda-log-feedback-train requires --div-lam <value>" >&2
        exit 1
    fi
    EDA_LOG_FEEDBACK_ARGS+=(--use-uncovered-reward)
    [ -n "${DIV_LAM}" ] && EDA_LOG_FEEDBACK_ARGS+=(--div-lam "${DIV_LAM}")
    echo "[run] use-uncovered-reward enabled: reward += div_lam*diversity (div_lam=${DIV_LAM:-unset})" | tee -a "${LOCAL_LOG}"
fi

# custom rollout log function (split log dir auto-derived from --save in train.py)
ROLLOUT_LOG_ARGS=(
    --custom-rollout-log-function-path examples.agentic_cov.rollout.log_train_samples
    --rollout-log-dir "${RUN_DIR}"
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout    0.0
   --attention-softmax-in-fp32
   --attention-backend flash
   --optimizer-cpu-offload
   --use-precision-aware-optimizer
)
if [ "${OVERRIDE_OPT_PARAM_SCHEDULER}" = "1" ]; then
   MISC_ARGS+=(--override-opt-param-scheduler)
   echo "[run] override-opt-param-scheduler enabled: current run scheduler config overrides checkpoint scheduler metadata" | tee -a "${LOCAL_LOG}"
fi

WANDB_ARGS=(
   --use-wandb
   --wandb-project slime-llm4cov
   --wandb-group qwen3-4B-2xH100
)

# -------------------- sync daemon --------------------
# Polls every 60 minutes (checkpoints are saved ~every 70 min); rsyncs all
# step_* HF checkpoint dirs that have appeared since the last poll, plus the
# training log.  Only files newer than the destination copy are transferred
# (--update).  A final sync runs on EXIT so the last checkpoint is always
# captured.

_rsync_to_remote() {
    local src="$1" dst="$2"
    if [ -n "${REMOTE_SYNC_SSH_KEY}" ]; then
        rsync -av --update \
            -e "ssh -i ${REMOTE_SYNC_SSH_KEY} -o StrictHostKeyChecking=no" \
            "${src}" "${dst}" 2>&1
    else
        rsync -av --update "${src}" "${dst}" 2>&1
    fi
}

# Upload RUN_DIR to a private HF repo via upload_folder: ONE commit per call, no
# concurrency, incremental (only changed files are re-uploaded). Single attempt —
# on any failure we log and GIVE UP (no retry); the next daemon cycle (3600 s later)
# just tries again. This replaces upload_large_folder, whose 8 workers + unbounded
# resumable retry spiked the 256-commits/hour limit and self-inflicted a 429 storm
# (27426 retries). Token from env HF_SYNC_TOKEN (separate from HF_TOKEN for download).
_hf_upload_once() {
    set +x   # xtrace OFF: 防 HF_SYNC_TOKEN 被 set -ex 的 trace 打进日志 (2026-06 泄露事件)
    python - "$HF_SYNC_REPO" "$RUN_DIR" <<'PYEOF'   # token 由 python 从环境继承, 不在命令行出现
import os, sys
from huggingface_hub import HfApi
repo, folder = sys.argv[1], sys.argv[2]
api = HfApi(token=os.environ["HF_SYNC_TOKEN"])
try:
    api.create_repo(repo_id=repo, repo_type="model", private=True, exist_ok=True)
    api.upload_folder(repo_id=repo, folder_path=folder, repo_type="model",
                      commit_message="sync run dir")
    print("[sync] upload_folder OK (single commit)")
except Exception as e:
    print(f"[sync] upload_folder FAILED — no retry, next cycle will retry: {repr(e)[:300]}")
    sys.exit(1)
PYEOF
    _rc=$?
    set -x   # xtrace 恢复
    return $_rc
}

_sync_once() {
    local ts
    ts="$(date '+%Y-%m-%d %H:%M:%S')"
    [ -d "${RUN_DIR}" ] || return 0
    if [ "${REMOTE_SYNC_TARGET}" = "hf" ]; then
        echo "[sync] ${ts} uploading run dir → HF ${HF_SYNC_REPO} (private)" \
            | tee -a "${LOCAL_LOG}"
        _hf_upload_once 2>&1 | tee -a "${LOCAL_LOG}" || true
    else
        echo "[sync] ${ts} syncing run dir → ${REMOTE_SYNC_BASE}/${RUN_SUBDIR}/" \
            | tee -a "${LOCAL_LOG}"
        _rsync_to_remote \
            "${RUN_DIR}/" \
            "${REMOTE_SYNC_BASE}/${RUN_SUBDIR}/" \
            | tee -a "${LOCAL_LOG}" || true
    fi
}

_sync_daemon() {
    echo "[sync] daemon started (PID=$$), polling every 3600 s" \
        | tee -a "${LOCAL_LOG}"
    while true; do
        sleep 3600
        _sync_once
    done
}

_sync_daemon &
SYNC_DAEMON_PID=$!

_write_gpu_monitor_summary() {
    [ -f "${GPU_MONITOR_LOG}" ] || return 0
    {
        echo "[gpu-monitor] sampling_interval_sec=${GPU_MONITOR_INTERVAL_SEC}"
        awk -F',' '
            NR > 1 && NF >= 5 && $2 ~ /^[0-9]+$/ && $4 ~ /^[0-9]+$/ {
                gpu = $2 + 0
                used = $4 + 0
                if (!(gpu in peak) || used > peak[gpu]) {
                    peak[gpu] = used
                    total[gpu] = $5 + 0
                    stamp[gpu] = $1
                }
            }
            END {
                for (gpu in peak) {
                    printf "[gpu-monitor] peak gpu=%d used_mib=%d total_mib=%d timestamp=%s\n", gpu, peak[gpu], total[gpu], stamp[gpu]
                }
            }
        ' "${GPU_MONITOR_LOG}" | sort -t= -k2,2n
    } | tee -a "${LOCAL_LOG}" > "${GPU_MONITOR_SUMMARY}"
}

_start_gpu_monitor() {
    if [ "${GPU_MONITOR_ENABLED}" != "1" ]; then
        echo "[gpu-monitor] disabled" | tee -a "${LOCAL_LOG}"
        return 0
    fi
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        echo "[gpu-monitor] nvidia-smi unavailable; monitor disabled" | tee -a "${LOCAL_LOG}"
        return 0
    fi
    (
        set +x
        echo "timestamp,index,utilization_gpu_pct,memory_used_mib,memory_total_mib"
        while true; do
            nvidia-smi \
                --query-gpu=timestamp,index,utilization.gpu,memory.used,memory.total \
                --format=csv,noheader,nounits | sed 's/, */,/g'
            sleep "${GPU_MONITOR_INTERVAL_SEC}"
        done
    ) >> "${GPU_MONITOR_LOG}" 2>&1 &
    GPU_MONITOR_PID=$!
    echo "[gpu-monitor] started pid=${GPU_MONITOR_PID} interval_sec=${GPU_MONITOR_INTERVAL_SEC}" | tee -a "${LOCAL_LOG}"
}

_stop_gpu_monitor() {
    if [ -n "${GPU_MONITOR_PID}" ] && kill -0 "${GPU_MONITOR_PID}" 2>/dev/null; then
        kill "${GPU_MONITOR_PID}" 2>/dev/null || true
        wait "${GPU_MONITOR_PID}" 2>/dev/null || true
    fi
    _write_gpu_monitor_summary
}

_cleanup_on_exit() {
    _stop_gpu_monitor
    _sync_once
    kill "${SYNC_DAEMON_PID}" 2>/dev/null || true
}

trap '_cleanup_on_exit' EXIT

# -------------------- launch --------------------
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}

# Keep Ray's session dir (logs + plasma spill) on the data disk; the default
# /tmp is often small (~100 GB) and fills up under multi-round agentic
# rollouts spilling 32k-token trajectories.
RAY_TEMP_DIR=${RAY_TEMP_DIR:-${ROOT_DIR}/ray_tmp}
mkdir -p "${RAY_TEMP_DIR}"

echo "[run] ─── TRAINING CONFIG ────────────────────────────────────" | tee -a "${LOCAL_LOG}"
echo "[run] NUM_ROLLOUT=${NUM_ROLLOUT}  ckpt_interval=${CKPT_INTERVAL}  eval_interval=${CKPT_INTERVAL}" | tee -a "${LOCAL_LOG}"
echo "[run] prompt_batch_size=${PROMPT_BATCH_SIZE}  num_agentic_rounds=${NUM_AGENTIC_ROUNDS}  rollout_batch_size=${ROLLOUT_BATCH_SIZE}  n_student=${N_STUDENT}  global_batch_size=${GLOBAL_BATCH_SIZE}" | tee -a "${LOCAL_LOG}"
echo "[run] num_agentic_rounds=${NUM_AGENTIC_ROUNDS}  eval_num_agentic_rounds=3" | tee -a "${LOCAL_LOG}"
echo "[run] offload=${OFFLOAD}  eda_log_feedback_train=${EDA_LOG_FEEDBACK_TRAIN}  eda_log_feedback_eval=${EDA_LOG_FEEDBACK_EVAL}  use_uncovered_log=${USE_UNCOVERED_LOG}  use_uncovered_reward=${USE_UNCOVERED_REWARD}  div_lam=${DIV_LAM:-unset}" | tee -a "${LOCAL_LOG}"
echo "[run] max_tokens_per_gpu=${MAX_TOKENS_PER_GPU}  no_final_save=${NO_FINAL_SAVE}" | tee -a "${LOCAL_LOG}"
echo "[run] train_dataset=${LLM4COV_DATASET}  eval_dataset=${LLM4COV_EVAL_DATASET}" | tee -a "${LOCAL_LOG}"
if [ "${#OPD_ARGS[@]}" -gt 0 ]; then
    echo "[run] opd_algorithm=${OPD_ALGORITHM}  opd_poll=${OPD_POLL}" | tee -a "${LOCAL_LOG}"
fi
echo "[run] ROTARY_BASE=${ROTARY_BASE}" | tee -a "${LOCAL_LOG}"
echo "[run] ─────────────────────────────────────────────────────────" | tee -a "${LOCAL_LOG}"

# Ensure the container hostname resolves locally. Otherwise Ray's dashboard
# agent (prometheus/OpenTelemetry) hangs ~30s on DNS for the hostname via the
# systemd-resolved stub, and the raylet times out in WaitForDashboardAgentPorts
# (Check failed: metrics_export_port), killing "node startup".
grep -q " $(hostname)\$" /etc/hosts || echo "127.0.0.1 $(hostname)" >> /etc/hosts 2>/dev/null || true

ray start --head \
    --node-ip-address "${MASTER_ADDR}" \
    --num-gpus "${NUM_GPUS}" \
    --temp-dir "${RAY_TEMP_DIR}" \
    --disable-usage-stats \
    --dashboard-host=127.0.0.1 \
    --dashboard-port=8265 \
    --port=6379 \
    --node-manager-port=6380 \
    --object-manager-port=6381 \
    --runtime-env-agent-port=6382 \
    --dashboard-agent-grpc-port=6383 \
    --dashboard-agent-listen-port=6384 \
    --metrics-export-port=6385 \
    --min-worker-port=16000 \
    --max-worker-port=19000

echo "[run] Ray dashboard: http://$(hostname -I | awk '{print $1}'):8265" | tee -a "${LOCAL_LOG}"

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/Megatron-LM/:/root/slime\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
    \"PYTORCH_ALLOC_CONF\": \"expandable_segments:True\"
  }
}"

_start_gpu_monitor
echo "[run] $(date '+%Y-%m-%d %H:%M:%S') submitting ray job..." | tee -a "${LOCAL_LOG}"
ray job submit --address="http://127.0.0.1:8265" \
    --runtime-env-json="${RUNTIME_ENV_JSON}" \
    -- python3 -m examples.agentic_cov.train \
    --actor-num-nodes 1 \
    --actor-num-gpus-per-node 2 \
    --colocate \
    "${MODEL_ARGS[@]}" \
    "${CKPT_ARGS[@]}" \
    "${ROLLOUT_ARGS[@]}" \
    "${AGENTIC_ARGS[@]}" \
    "${EVAL_ARGS[@]}" \
    "${OPTIMIZER_ARGS[@]}" \
    "${GRPO_ARGS[@]}" \
    "${PERF_ARGS[@]}" \
    "${SGLANG_ARGS[@]}" \
    "${MISC_ARGS[@]}" \
    "${OFFLOAD_ARGS[@]}" \
    "${EDA_LOG_FEEDBACK_ARGS[@]}" \
    "${ROLLOUT_LOG_ARGS[@]}" \
    "${WANDB_ARGS[@]}" \
    "${OPD_ARGS[@]}" \
    2>&1 | tee -a "${LOCAL_LOG}"
