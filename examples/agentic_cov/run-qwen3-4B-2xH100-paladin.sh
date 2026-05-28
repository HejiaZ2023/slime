#!/bin/bash
# Multi-round agentic GRPO training for Qwen3-4B on 2x H100 NVL (96 GB HBM).
# Derived from run-qwen3-4B-4xH100-noeval.sh:
#   - TP=2, CP=1 (4-GPU CP=2 removed; 2 GPUs fill TP=2 only)
#   - 1 SGLang engine x 2 GPUs (vs 2 engines x 2 GPUs on 4xH100)
#   - actor-num-gpus-per-node 2
#   - sglang-mem-fraction-static raised to 0.55 (more VRAM available without CP)
#
# Run from the slime repo root inside the Docker container:
#   EDA_SERVER=paladin_centos EDA_REPO_DIR=/workspace/llm4cov_eda \
#   bash examples/agentic_cov/run-qwen3-4B-2xH100-paladin.sh
#
# Required env:
#   EDA_SERVER     SSH alias of the llm4cov_eda worker (paladin_centos)
#   EDA_REPO_DIR   path to llm4cov_eda on that host (/workspace/llm4cov_eda)
# Optional overrides:
#   MODEL_NAME     HF model id or local path (default: SFT stage-0)
#   ROOT_DIR       output root on /data (default: /data/rl_runs)
#   NUM_ROLLOUT    number of rollout steps (default: 300; use 20 for quick test)
#   SAVE_DIR_ON_EXIST  overwrite|resume|stop (skips interactive prompt)

set -ex

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
NUM_GPUS=${NUM_GPUS:-${DETECTED_GPUS:-2}}
if [ "$NUM_GPUS" -lt 2 ]; then
    echo "ERROR: need at least 2 GPUs. Got NUM_GPUS=$NUM_GPUS." >&2
    exit 1
fi

# -------------------- required external state --------------------
: "${EDA_SERVER:?Set EDA_SERVER to the SSH alias of the llm4cov_eda worker}"
: "${EDA_REPO_DIR:?Set EDA_REPO_DIR to the path of llm4cov_eda on that worker}"

MODEL_NAME=${MODEL_NAME:-hez2024/LLM4Cov-Qwen3-4B-SFT-Stage0}
ROOT_DIR=${ROOT_DIR:-/data/rl_runs}
LLM4COV_DATASET=${LLM4COV_DATASET:-hez2024/CodeV-R1-dataset-RL-test}
LLM4COV_SPLIT=${LLM4COV_SPLIT:-train}
NUM_ROLLOUT=${NUM_ROLLOUT:-300}

ROTARY_BASE=${ROTARY_BASE:-5000000}
export MODEL_ARGS_ROTARY_BASE=${MODEL_ARGS_ROTARY_BASE:-5000000}

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
SLIME_ROOT="$(cd "${SCRIPT_DIR}/../.." &>/dev/null && pwd)"
if [ "$(pwd)" != "${SLIME_ROOT}" ]; then
    echo "ERROR: run this script from the slime repo root (${SLIME_ROOT}), got $(pwd)" >&2
    exit 1
fi
source "${SLIME_ROOT}/scripts/models/qwen3-4B.sh"

MODEL_ARGS+=(--make-vocab-size-divisible-by 1)

source "${SCRIPT_DIR}/_setup_checkpoints.sh"

# -------------------- checkpoint paths --------------------
CKPT_ARGS=(
   --hf-checkpoint "${HF_CKPT}"
   --ref-load      "${REF_LOAD}"
   --load          "${SAVE_DIR}"
   --save          "${SAVE_DIR}"
   --save-interval 50
   --save-hf       "${SAVE_DIR}_hf/step_{rollout_id}"
)

# -------------------- rollout / batching --------------------
ROLLOUT_ARGS=(
   --rollout-shuffle
   --num-rollout            "${NUM_ROLLOUT}"
   --rollout-batch-size     ${ROLLOUT_BATCH_SIZE:-4}
   --n-samples-per-prompt   ${N_SAMPLES:-4}
   --rollout-max-response-len 16384
   --rollout-temperature    1.0
   --apply-chat-template
   --global-batch-size      ${GLOBAL_BATCH_SIZE:-16}
   --balance-data
)

# -------------------- llm4cov agentic config --------------------
AGENTIC_ARGS=(
   --rollout-function-path examples.agentic_cov.rollout.generate_rollout
   --data-source-path      examples.agentic_cov.data_source.LlmCovDataSource
   --num-agentic-rounds    ${NUM_AGENTIC_ROUNDS:-2}
   --llm4cov-dataset-name  "${LLM4COV_DATASET}"
   --llm4cov-dataset-split "${LLM4COV_SPLIT}"
   --eda-server            "${EDA_SERVER}"
   --eda-repo-dir          "${EDA_REPO_DIR}"
)

# -------------------- parallelism / memory --------------------
# 2 GPUs: TP=2, CP=1, PP=1. Data parallel = 1.
# No CP sharding means each GPU sees the full packed sequence; raise
# sglang-mem-fraction-static slightly vs the 4-GPU colocated recipe.
PERF_ARGS=(
   --tensor-model-parallel-size  1
   --pipeline-model-parallel-size 1
   --context-parallel-size       2
   --expert-model-parallel-size  1
   --expert-tensor-parallel-size 1

   --recompute-granularity full
   --recompute-method      uniform
   --recompute-num-layers  1

   --use-dynamic-batch-size
   --max-tokens-per-gpu    ${MAX_TOKENS_PER_GPU:-24576}
)

# -------------------- DAPO --------------------
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
# 1 engine x 2 GPUs (colocated with training).
SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 2
   --sglang-mem-fraction-static  0.4
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
   --wandb-group qwen3-4B-2xH100-paladin
)

# -------------------- launch --------------------
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}

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
    "${OPTIMIZER_ARGS[@]}" \
    "${GRPO_ARGS[@]}" \
    "${PERF_ARGS[@]}" \
    "${SGLANG_ARGS[@]}" \
    "${MISC_ARGS[@]}" \
    "${WANDB_ARGS[@]}"
