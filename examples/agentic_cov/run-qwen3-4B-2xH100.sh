#!/bin/bash
# Multi-round agentic GRPO training for Qwen3-4B on 2x H100 (80 GB HBM),
# WITH periodic eval on hez2024/cvdp_ecov_eval.
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
#                                     [--use-uncovered-log] [--interval N] [--steps N]
#                                     [--train-dataset NAME] [--eval-dataset NAME]
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
#   --interval N             Checkpoint save + eval interval in rollout steps (default: 50).
#                            Overrides the CKPT_INTERVAL env var.
#   --steps N                Total number of rollout steps to train (default: 300).
#                            Overrides the NUM_ROLLOUT env var.
#   --train-dataset NAME     HuggingFace dataset name for training rollouts
#                            (default: hez2024/CodeV-R1-dataset-RL-test).
#                            Overrides the LLM4COV_DATASET env var.
#   --eval-dataset  NAME     HuggingFace dataset name for eval rollouts
#                            (default: hez2024/cvdp_ecov_eval).
#                            Overrides the LLM4COV_EVAL_DATASET env var.
OFFLOAD=0
EDA_LOG_FEEDBACK_TRAIN=0
EDA_LOG_FEEDBACK_EVAL=0
USE_UNCOVERED_LOG=0
while [ $# -gt 0 ]; do
    case "$1" in
        --offload)                OFFLOAD=1 ;;
        --eda-log-feedback-train|--elft) EDA_LOG_FEEDBACK_TRAIN=1 ;;
        --eda-log-feedback-eval|--elfe)  EDA_LOG_FEEDBACK_EVAL=1 ;;
        --use-uncovered-log|--uul)       USE_UNCOVERED_LOG=1 ;;
        --interval)               CKPT_INTERVAL="${2:?--interval requires a value}"; shift ;;
        --interval=*)             CKPT_INTERVAL="${1#--interval=}" ;;
        --steps)                  NUM_ROLLOUT="${2:?--steps requires a value}"; shift ;;
        --steps=*)                NUM_ROLLOUT="${1#--steps=}" ;;
        --train-dataset)          LLM4COV_DATASET="${2:?--train-dataset requires a value}"; shift ;;
        --train-dataset=*)        LLM4COV_DATASET="${1#--train-dataset=}" ;;
        --eval-dataset)           LLM4COV_EVAL_DATASET="${2:?--eval-dataset requires a value}"; shift ;;
        --eval-dataset=*)         LLM4COV_EVAL_DATASET="${1#--eval-dataset=}" ;;
        *) echo "[run] Unknown argument: $1" >&2; exit 1 ;;
    esac
    shift
done

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

MODEL_NAME=${MODEL_NAME:-hez2024/LLM4Cov-Qwen3-4B-SFT-Stage0}
ROOT_DIR=${ROOT_DIR:-$(pwd)}
LLM4COV_DATASET=${LLM4COV_DATASET:-hez2024/CodeV-R1-dataset-RL-test}
LLM4COV_SPLIT=${LLM4COV_SPLIT:-train}
LLM4COV_EVAL_DATASET=${LLM4COV_EVAL_DATASET:-hez2024/cvdp_ecov_eval}
LLM4COV_EVAL_SPLIT=${LLM4COV_EVAL_SPLIT:-eval}
NUM_ROLLOUT=${NUM_ROLLOUT:-300}
CKPT_INTERVAL=${CKPT_INTERVAL:-50}   # shared interval for --save-interval and --eval-interval

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
if [ "${REMOTE_SYNC_TARGET}" = "hf" ] && [ -z "${HF_SYNC_TOKEN:-}" ]; then
    echo "ERROR: REMOTE_SYNC_TARGET=hf requires a write token in HF_SYNC_TOKEN (pass -e HF_SYNC_TOKEN=...)" >&2
    exit 1
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

# RUN_SUBDIR and LOCAL_LOG are set here (after SAVE_DIR is available) so the
# subdirectory name incorporates the actual checkpoint base name.
_TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_DIR="${ROOT_DIR}/${_MODEL_BASENAME}_${_TIMESTAMP}"
mkdir -p "${RUN_DIR}"
RUN_SUBDIR=${RUN_SUBDIR:-"${_MODEL_BASENAME}_${_TIMESTAMP}"}
HF_SYNC_REPO="${HF_SYNC_REPO_PREFIX}${RUN_SUBDIR}"
LOCAL_LOG="${RUN_DIR}/main.log"
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
echo "[run] ───────────────────────────────────────────────────────" | tee -a "${LOCAL_LOG}"

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
)

# -------------------- rollout / batching --------------------
# group_size = n_samples_per_prompt = 4
# global_batch_size = 16  ->  rollout_batch_size = 16 / 4 = 4
# Default 300 steps total -> --num-rollout 300 (default num_steps_per_rollout=1).
# Override with NUM_ROLLOUT=... to run shorter/longer.
ROLLOUT_ARGS=(
   --rollout-shuffle
   --num-rollout            "${NUM_ROLLOUT}"
   --rollout-batch-size     4
   --n-samples-per-prompt   4
   --rollout-max-response-len 16384
   --rollout-temperature    1.0

   # llm4cov dataset is built from chat messages; apply_chat_template
   # converts list[dict] -> str so sglang's tokenizer.encode accepts it.
   --apply-chat-template

   --global-batch-size      16
   --balance-data
)

# -------------------- llm4cov agentic config --------------------
AGENTIC_ARGS=(
   --rollout-function-path examples.agentic_cov.rollout.generate_rollout
   --data-source-path      examples.agentic_cov.data_source.LlmCovDataSource
   --num-agentic-rounds       2
   --eval-num-agentic-rounds  3
   --llm4cov-dataset-name        "${LLM4COV_DATASET}"
   --llm4cov-dataset-split       "${LLM4COV_SPLIT}"
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

# -------------------- parallelism / memory --------------------
# 4 GPUs split as TP=2 x CP=2 x PP=1. With CP=2 a 40k sequence is sharded
# to ~20k tokens per CP rank; --max-tokens-per-gpu 24576 leaves headroom for
# packing a few short sequences alongside one long one. Same setting as
# H200 — H200 is not full at 24k, so 80 GB H100 should still fit.
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
   --max-tokens-per-gpu    24576
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

# Upload RUN_DIR to a private HF repo. Resumable & incremental: upload_large_folder
# re-scans each call and only uploads files not already committed. Uses the write
# token from env HF_SYNC_TOKEN (separate from HF_TOKEN used for model download).
_hf_upload_once() {
    HF_SYNC_TOKEN="${HF_SYNC_TOKEN}" python - "$HF_SYNC_REPO" "$RUN_DIR" <<'PYEOF'
import os, sys
from huggingface_hub import HfApi
repo, folder = sys.argv[1], sys.argv[2]
api = HfApi(token=os.environ["HF_SYNC_TOKEN"])
api.create_repo(repo_id=repo, repo_type="model", private=True, exist_ok=True)
api.upload_large_folder(repo_id=repo, folder_path=folder, repo_type="model",
                        num_workers=8, print_report=False)
PYEOF
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
trap '_sync_once; kill "${SYNC_DAEMON_PID}" 2>/dev/null; true' EXIT

# -------------------- launch --------------------
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}

# Keep Ray's session dir (logs + plasma spill) on the data disk; the default
# /tmp is often small (~100 GB) and fills up under multi-round agentic
# rollouts spilling 32k-token trajectories.
RAY_TEMP_DIR=${RAY_TEMP_DIR:-${ROOT_DIR}/ray_tmp}
mkdir -p "${RAY_TEMP_DIR}"

echo "[run] ─── TRAINING CONFIG ────────────────────────────────────" | tee -a "${LOCAL_LOG}"
echo "[run] NUM_ROLLOUT=${NUM_ROLLOUT}  ckpt_interval=${CKPT_INTERVAL}  eval_interval=${CKPT_INTERVAL}" | tee -a "${LOCAL_LOG}"
echo "[run] rollout_batch_size=4  n_samples_per_prompt=4  global_batch_size=16" | tee -a "${LOCAL_LOG}"
echo "[run] num_agentic_rounds=2  eval_num_agentic_rounds=3" | tee -a "${LOCAL_LOG}"
echo "[run] offload=${OFFLOAD}  eda_log_feedback_train=${EDA_LOG_FEEDBACK_TRAIN}  eda_log_feedback_eval=${EDA_LOG_FEEDBACK_EVAL}  use_uncovered_log=${USE_UNCOVERED_LOG}" | tee -a "${LOCAL_LOG}"
echo "[run] train_dataset=${LLM4COV_DATASET}  eval_dataset=${LLM4COV_EVAL_DATASET}" | tee -a "${LOCAL_LOG}"
echo "[run] ROTARY_BASE=${ROTARY_BASE}" | tee -a "${LOCAL_LOG}"
echo "[run] ─────────────────────────────────────────────────────────" | tee -a "${LOCAL_LOG}"

ray start --head \
    --node-ip-address "${MASTER_ADDR}" \
    --num-gpus "${NUM_GPUS}" \
    --temp-dir "${RAY_TEMP_DIR}" \
    --disable-usage-stats \
    --dashboard-host=0.0.0.0 \
    --dashboard-port=8265

echo "[run] Ray dashboard: http://$(hostname -I | awk '{print $1}'):8265" | tee -a "${LOCAL_LOG}"

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/Megatron-LM/:/root/slime\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
    \"PYTORCH_ALLOC_CONF\": \"expandable_segments:True\"
  }
}"

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
    2>&1 | tee -a "${LOCAL_LOG}"
