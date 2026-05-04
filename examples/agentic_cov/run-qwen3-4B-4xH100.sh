#!/bin/bash
# Multi-round agentic GRPO training for Qwen3-4B on 4x H100 (80 GB HBM),
# WITH periodic eval on hez2024/cvdp_ecov_eval.
# - DAPO-style: asymmetric clipping (low=0.2, high=0.28) + sequence-normalized
#   loss (--calculate-per-token-loss). NO dynamic sampling.
# - Collocated rollout + training (--colocate).
# - Eval every --eval-interval rollouts via examples.agentic_cov.rollout.eval_rollout.
#   Eval prompts come from --llm4cov-eval-dataset-{name,split}; the
#   --eval-prompt-data pair below only exists to satisfy slime's
#   eval_datasets validation — the path is unused by eval_rollout.
# - No KL, no weight decay.
# - 40k total context: rollout response 32k, max packed train tokens 24k/GPU.
#
# Run from the slime repo root (e.g. /root/slime in the docker image):
#   bash examples/agentic_cov/run-qwen3-4B-4xH100.sh
#
# Required env:
#   EDA_SERVER     SSH alias of the llm4cov_eda worker
#   EDA_REPO_DIR   path to the llm4cov_eda checkout on that host
# Optional overrides documented in examples/agentic_cov/README.md.

set -ex

# clean up stale ray / sglang / python from prior runs
pkill -9 sglang 2>/dev/null || true
sleep 2
ray stop --force 2>/dev/null || true
pkill -9 ray 2>/dev/null || true
pkill -9 python 2>/dev/null || true
sleep 2

export PYTHONBUFFERED=1

# -------------------- topology --------------------
NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
HAS_NVLINK=$([ "$NVLINK_COUNT" -gt 0 ] && echo 1 || echo 0)

DETECTED_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')
NUM_GPUS=${NUM_GPUS:-${DETECTED_GPUS:-4}}
if [ "$NUM_GPUS" -lt 4 ]; then
    echo "ERROR: this recipe targets 4 GPUs (TP=2, CP=2). Got NUM_GPUS=$NUM_GPUS." >&2
    exit 1
fi

# -------------------- required external state --------------------
: "${EDA_SERVER:?Set EDA_SERVER to the SSH alias of the llm4cov_eda worker}"
: "${EDA_REPO_DIR:?Set EDA_REPO_DIR to the path of llm4cov_eda on that worker}"

MODEL_NAME=${MODEL_NAME:-hez2024/LLM4Cov-Qwen3-4B-SFT-Stage0}
ROOT_DIR=${ROOT_DIR:-$(pwd)}
LLM4COV_DATASET=${LLM4COV_DATASET:-hez2024/CodeV-R1-dataset-RL-test}
LLM4COV_SPLIT=${LLM4COV_SPLIT:-train}
LLM4COV_EVAL_DATASET=${LLM4COV_EVAL_DATASET:-hez2024/cvdp_ecov_eval}
LLM4COV_EVAL_SPLIT=${LLM4COV_EVAL_SPLIT:-eval}

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

# Derives HF_CKPT / REF_LOAD / SAVE_DIR from MODEL_NAME + ROOT_DIR, and runs
# first-time HF download / torch_dist conversion if those dirs are missing.
source "${SCRIPT_DIR}/_setup_checkpoints.sh"

# -------------------- checkpoint paths --------------------
CKPT_ARGS=(
   --hf-checkpoint "${HF_CKPT}"
   --ref-load      "${REF_LOAD}"
   --load          "${SAVE_DIR}"
   --save          "${SAVE_DIR}"
   --save-interval 50
   # HF-format dump alongside the torch_dist save. {rollout_id} is filled
   # in by slime via args.save_hf.format(rollout_id=...).
   --save-hf       "${SAVE_DIR}_hf/step_{rollout_id}"
)

# -------------------- rollout / batching --------------------
# group_size = n_samples_per_prompt = 4
# global_batch_size = 16  ->  rollout_batch_size = 16 / 4 = 4
# 300 steps total -> --num-rollout 300 (default num_steps_per_rollout=1)
# response 32k + prompt budget ~8k = 40k total context
ROLLOUT_ARGS=(
   --rollout-shuffle
   --num-rollout            300
   --rollout-batch-size     4
   --n-samples-per-prompt   4
   --rollout-max-response-len 32768
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
   --eval-num-agentic-rounds  1
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
   --eval-interval              20
   --eval-function-path         examples.agentic_cov.rollout.eval_rollout
   --eval-prompt-data           "${LLM4COV_EVAL_DATASET}" "${LLM4COV_EVAL_DATASET}"
   --n-samples-per-eval-prompt  1
   --eval-max-response-len      32768
   --eval-temperature           0.0
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
   --context-parallel-size       2
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
   --sglang-mem-fraction-static  0.5
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout    0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
)

WANDB_ARGS=(
   --use-wandb
   --wandb-project slime-llm4cov
   --wandb-group qwen3-4B-4xH100
)

# -------------------- launch --------------------
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}

# Keep Ray's session dir (logs + plasma spill) on the data disk; the default
# /tmp is often small (~100 GB) and fills up under multi-round agentic
# rollouts spilling 32k-token trajectories.
RAY_TEMP_DIR=${RAY_TEMP_DIR:-${ROOT_DIR}/ray_tmp}
mkdir -p "${RAY_TEMP_DIR}"

ray start --head \
    --node-ip-address "${MASTER_ADDR}" \
    --num-gpus "${NUM_GPUS}" \
    --temp-dir "${RAY_TEMP_DIR}" \
    --disable-usage-stats \
    --dashboard-host=0.0.0.0 \
    --dashboard-port=8265

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/Megatron-LM/:/root/slime\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\"
  }
}"

ray job submit --address="http://127.0.0.1:8265" \
    --runtime-env-json="${RUNTIME_ENV_JSON}" \
    -- python3 -m examples.agentic_cov.train \
    --actor-num-nodes 1 \
    --actor-num-gpus-per-node "${NUM_GPUS}" \
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
    "${WANDB_ARGS[@]}"
