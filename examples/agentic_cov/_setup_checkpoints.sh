# shellcheck shell=bash
# First-time setup helper for agentic_cov run scripts.
#
# Inputs (env or pre-set in the calling script):
#   MODEL_NAME    HF model id, e.g. hez2024/LLM4Cov-Qwen3-4B-SFT-Stage0
#   ROOT_DIR      Parent dir on the data disk for local artifacts
#   MODEL_ARGS    (already populated by sourcing scripts/models/<model>.sh)
#   ROTARY_BASE   Optional override for --rotary-base during HF→torch_dist
#                 conversion (e.g. 5000000 for Qwen3-4B-Instruct-2507).
#   SAVE_DIR_ON_EXIST  Optional: overwrite | resume | stop (skips the prompt).
#   SAVE_DIR_SUFFIX   Optional: suffix appended after "_slime" in SAVE_DIR (e.g. _elfe).
#
# Outputs (exported for the caller):
#   HF_CKPT       ${ROOT_DIR}/<basename(MODEL_NAME)>
#   REF_LOAD      ${HF_CKPT}_torch_dist
#   SAVE_DIR      ${HF_CKPT}_slime${SAVE_DIR_SUFFIX:-}
#
# Side effects:
#   - Downloads HF weights into HF_CKPT if missing/empty.
#   - Runs tools/convert_hf_to_torch_dist.py into REF_LOAD if missing/empty.
#   - If SAVE_DIR is non-empty, prompts to overwrite (rm -rf) or stop, unless
#     SAVE_DIR_ON_EXIST is preset. "resume" leaves SAVE_DIR alone so slime's
#     --load picks up the last checkpoint.

: "${MODEL_NAME:?Set MODEL_NAME to a HuggingFace model id (org/name)}"
: "${ROOT_DIR:?Set ROOT_DIR to a parent dir on the data disk}"
: "${SLIME_ROOT:?_setup_checkpoints.sh expects SLIME_ROOT to be set by the caller}"

if [ ! -d "${ROOT_DIR}" ]; then
    echo "ERROR: ROOT_DIR=${ROOT_DIR} does not exist; create it first." >&2
    exit 1
fi

_MODEL_BASENAME="$(basename "${MODEL_NAME}")"
HF_CKPT="${ROOT_DIR}/${_MODEL_BASENAME}"
REF_LOAD="${HF_CKPT}_torch_dist"
SAVE_DIR="${HF_CKPT}_slime${SAVE_DIR_SUFFIX:-}"

_dir_has_content() {
    [ -d "$1" ] && [ -n "$(ls -A "$1" 2>/dev/null)" ]
}

# -------------------- 1. HF download --------------------
if _dir_has_content "${HF_CKPT}"; then
    echo "[setup] HF_CKPT exists with content: ${HF_CKPT}"
else
    echo "[setup] downloading ${MODEL_NAME} -> ${HF_CKPT}"
    mkdir -p "${HF_CKPT}"
    huggingface-cli download "${MODEL_NAME}" --local-dir "${HF_CKPT}"
fi

# -------------------- 2. HF -> torch_dist conversion --------------------
if _dir_has_content "${REF_LOAD}"; then
    echo "[setup] REF_LOAD exists with content: ${REF_LOAD}"
else
    echo "[setup] converting HF -> torch_dist: ${HF_CKPT} -> ${REF_LOAD}"
    if [ "${#MODEL_ARGS[@]}" -eq 0 ]; then
        echo "ERROR: MODEL_ARGS is empty — source scripts/models/<model>.sh before this helper." >&2
        exit 1
    fi
    _convert_args=(
        "${MODEL_ARGS[@]}"
        --hf-checkpoint "${HF_CKPT}"
        --save "${REF_LOAD}"
    )
    if [ -n "${ROTARY_BASE:-}" ]; then
        _convert_args+=(--rotary-base "${ROTARY_BASE}")
    fi
    PYTHONPATH=${PYTHONPATH:-/root/Megatron-LM} torchrun --nproc_per_node 1 \
        "${SLIME_ROOT}/tools/convert_hf_to_torch_dist.py" \
        "${_convert_args[@]}"
fi

# -------------------- 3. SAVE_DIR overwrite/resume/stop --------------------
if _dir_has_content "${SAVE_DIR}"; then
    _action=${SAVE_DIR_ON_EXIST:-}
    if [ -z "${_action}" ]; then
        if [ ! -t 0 ]; then
            echo "ERROR: SAVE_DIR=${SAVE_DIR} already has content and stdin is not a TTY." >&2
            echo "       Set SAVE_DIR_ON_EXIST=overwrite|resume|stop and re-run." >&2
            exit 1
        fi
        echo "[setup] SAVE_DIR=${SAVE_DIR} already has content."
        read -r -p "        [o]verwrite (rm -rf), [r]esume from last ckpt, or [s]top? " _ans
        case "${_ans}" in
            o|O|overwrite) _action=overwrite ;;
            r|R|resume)    _action=resume ;;
            *)             _action=stop ;;
        esac
    fi
    case "${_action}" in
        overwrite)
            echo "[setup] removing ${SAVE_DIR}"
            rm -rf "${SAVE_DIR}"
            mkdir -p "${SAVE_DIR}"
            ;;
        resume)
            echo "[setup] resuming from existing ${SAVE_DIR}"
            ;;
        stop)
            echo "[setup] stopping; SAVE_DIR untouched."
            exit 0
            ;;
        *)
            echo "ERROR: SAVE_DIR_ON_EXIST=${_action} (expected overwrite|resume|stop)" >&2
            exit 1
            ;;
    esac
else
    mkdir -p "${SAVE_DIR}"
fi

export HF_CKPT REF_LOAD SAVE_DIR
