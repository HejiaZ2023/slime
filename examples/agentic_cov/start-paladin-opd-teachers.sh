#!/bin/bash
# Start the two OPD teacher SGLang servers on Paladin, bound to host loopback.
#
# Defaults are intentionally fixed to the RL step999 teachers:
#   stage0 -> /mnt/raid0_ssd/sheng/final_ckpts/stage0_step999
#   stage1 -> /mnt/raid0_ssd/sheng/final_ckpts/stage1_step999
#
# The printed OPD_TEACHERS_JSON should be used when starting/restarting
# opd_worker.py so future OPD artifacts record the exact teacher model paths.

set -euo pipefail

IMAGE=${OPD_TEACHER_IMAGE:-hejiaz/llm4cov_slime:0519}
MODEL_ROOT=${OPD_TEACHER_MODEL_ROOT:-/mnt/raid0_ssd/sheng/final_ckpts}
STAGE0_MODEL=${OPD_TEACHER_STAGE0_MODEL_PATH:-${MODEL_ROOT}/stage0_step999}
STAGE1_MODEL=${OPD_TEACHER_STAGE1_MODEL_PATH:-${MODEL_ROOT}/stage1_step999}
STAGE0_GPU=${OPD_TEACHER_STAGE0_GPU:-0}
STAGE1_GPU=${OPD_TEACHER_STAGE1_GPU:-1}
STAGE0_PORT=${OPD_TEACHER_STAGE0_PORT:-18080}
STAGE1_PORT=${OPD_TEACHER_STAGE1_PORT:-18081}
HOST_BIND=${OPD_TEACHER_HOST_BIND:-127.0.0.1}
CONTAINER_PORT=${OPD_TEACHER_CONTAINER_PORT:-8000}
MEM_FRACTION=${OPD_TEACHER_MEM_FRACTION_STATIC:-0.82}
CONTEXT_LENGTH=${OPD_TEACHER_CONTEXT_LENGTH:-32768}
VERIFY_STEP999=${OPD_TEACHER_VERIFY_STEP999:-1}

check_model() {
    local model_dir=$1
    test -f "${model_dir}/config.json"
    test -f "${model_dir}/model.safetensors.index.json"
    test -f "${model_dir}/model-00001-of-00002.safetensors"
    test -f "${model_dir}/model-00002-of-00002.safetensors"
}

verify_model_hashes() {
    local stage=$1
    local model_dir=$2
    local expect1=""
    local expect2=""
    case "${stage}" in
        stage0)
            expect1="93257fc312e353ae65396b96dfa6937616ec35e1edfbf728206f62eca69a1fb8"
            expect2="9f31220699f74e35168c0a92109fea6baa1f01909df7bfec95485f38ce625213"
            ;;
        stage1)
            expect1="36e549531bc9cbde74678515d44bdf77e287f5affe62e6494b86093476d90c5d"
            expect2="89931894663315cdeb3df8a062b6312b99fbdcdee522332884cb56d111ec9e57"
            ;;
        *)
            return 0
            ;;
    esac
    local got1 got2
    got1=$(sha256sum "${model_dir}/model-00001-of-00002.safetensors" | awk '{print $1}')
    got2=$(sha256sum "${model_dir}/model-00002-of-00002.safetensors" | awk '{print $1}')
    if [ "${got1}" != "${expect1}" ] || [ "${got2}" != "${expect2}" ]; then
        echo "ERROR: ${stage} teacher checkpoint does not match ${stage}_step999." >&2
        echo "       shard1 got ${got1}, expected ${expect1}" >&2
        echo "       shard2 got ${got2}, expected ${expect2}" >&2
        exit 1
    fi
    echo "[opd] verified ${stage} checkpoint sha256"
}

start_teacher() {
    local stage=$1
    local gpu=$2
    local port=$3
    local model_dir=$4
    local name="opd_teacher_${stage}"

    check_model "${model_dir}"
    if [ "${VERIFY_STEP999}" = "1" ]; then
        verify_model_hashes "${stage}" "${model_dir}"
    fi
    docker rm -f "${name}" >/dev/null 2>&1 || true
    docker run -d --name "${name}" \
        --runtime=nvidia --gpus "device=${gpu}" \
        --network bridge \
        -p "${HOST_BIND}:${port}:${CONTAINER_PORT}" \
        --shm-size=16g \
        -v "${model_dir}:/model:ro" \
        --label "llm4cov.opd.teacher=${stage}" \
        --label "llm4cov.opd.model_path=${model_dir}" \
        --entrypoint python3 \
        "${IMAGE}" \
        -m sglang.launch_server \
        --model-path /model \
        --tokenizer-path /model \
        --served-model-name "${stage}" \
        --host 0.0.0.0 \
        --port "${CONTAINER_PORT}" \
        --tp-size 1 \
        --trust-remote-code \
        --mem-fraction-static "${MEM_FRACTION}" \
        --context-length "${CONTEXT_LENGTH}"
    echo "[opd] started ${name}: ${HOST_BIND}:${port} -> ${model_dir}"
}

start_teacher stage0 "${STAGE0_GPU}" "${STAGE0_PORT}" "${STAGE0_MODEL}"
start_teacher stage1 "${STAGE1_GPU}" "${STAGE1_PORT}" "${STAGE1_MODEL}"

printf '[opd] OPD_TEACHERS_JSON=%s\n' \
    "{\"stage0\":{\"url\":\"http://127.0.0.1:${STAGE0_PORT}/generate\",\"model_path\":\"${STAGE0_MODEL}\"},\"stage1\":{\"url\":\"http://127.0.0.1:${STAGE1_PORT}/generate\",\"model_path\":\"${STAGE1_MODEL}\"}}"
echo "[opd] verify with: docker logs -f opd_teacher_stage0 | grep -m1 'Application startup complete'"
