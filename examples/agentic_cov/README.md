# agentic_cov: multi-round GRPO for llm4cov

Trains an LLM to write SystemVerilog testbenches against the llm4cov EDA
reward via multi-round agentic GRPO.

| file | role |
|---|---|
| `train.py` | entry point (adds llm4cov-specific CLI flags, calls slime's `train`) |
| `rollout.py` | `generate_rollout` (training) + `eval_rollout` (eval) — multi-round agentic loop |
| `data_source.py` | `LlmCovDataSource` — pulls prompts from an llm4cov HF dataset |
| `dataset.py` | turns llm4cov contexts into slime `Sample` objects |
| `reward.py` | EDA-coverage reward, called from `rollout.py` |
| `run-*.sh` | launch scripts for specific hardware shapes |

## Build the docker image

There are two paths. Prefer the overlay unless you are bootstrapping from
nothing — it is ~5 min vs ~50 min and yields a tiny push delta.

### Overlay (recommended) — `docker/Dockerfile.llm4cov`

Stacks our fork of slime + the llm4cov_oss submodule on top of an already
published `slimerl/slime:vX.Y.Z` base, which ships the matching pin set
(transformers 4.57.1, sglang 0.5.9, openai 2.6.1, pydantic 2.12.5,
datasets 4.5.0, flash-attn 2/3, apex, TE, megatron-core, sgl-router).

From the slime repo root:

```bash
SLIME_COMMIT=$(git rev-parse HEAD)        # or a fixed sha

docker build -f docker/Dockerfile.llm4cov . \
    --build-arg BASE_IMAGE_TAG=v0.2.4 \
    --build-arg SLIME_REPO_URL=https://github.com/HejiaZ2023/slime.git \
    --build-arg SLIME_COMMIT="${SLIME_COMMIT}" \
    -t slime-llm4cov:${SLIME_COMMIT::7}
```

Total of ~340 MB of new layer content (replacement slime tree, int4_qat
kernel rebuild, llm4cov_oss editable install, tree-sitter wheels). Bump
`BASE_IMAGE_TAG` when rebasing to a newer slime release.

### From scratch — `docker/Dockerfile`

Used by the publishing pipeline to produce the `slimerl/slime` images the
overlay sits on. Full ~31-layer rebuild including flash-attn, apex,
TransformerEngine, sglang patches.

```bash
docker build -f docker/Dockerfile . \
    --build-arg SLIME_REPO_URL=https://github.com/HejiaZ2023/slime.git \
    --build-arg SLIME_COMMIT=$(git rev-parse HEAD) \
    --build-arg HTTP_PROXY="$http_proxy" \
    --build-arg HTTPS_PROXY="$https_proxy" \
    --build-arg NO_PROXY="localhost,127.0.0.1" \
    -t slime-llm4cov:full
```

The `transformers`, `pydantic`, `openai`, etc. floors used during the llm4cov
install are kept low enough that pip does not bump `transformers` away from
the 4.57.1 that `sglang==0.5.9` pins exactly.

### NVIDIA DGX Spark (GB10/sm_121a, arm64)

Use `docker/Dockerfile.gb10` — it rebases on the NGC vLLM container and
copies the local slime tree (no `SLIME_REPO_URL`/`SLIME_COMMIT` build args).

## Run a container

The training process needs:
- HF checkpoint of the policy model
- Megatron-format `torch_dist` checkpoint (produced by
  `tools/convert_hf_to_torch_dist.py`)
- Reachable EDA worker host (SSH alias) and the `llm4cov_eda` checkout on it
- Optionally: HuggingFace cache for the llm4cov dataset and any wandb creds

```bash
docker run --gpus all --shm-size=32g --network=host --ipc=host --rm -it \
    -v /path/to/Qwen3-4B:/root/Qwen3-4B \
    -v /path/to/Qwen3-4B_torch_dist:/root/Qwen3-4B_torch_dist \
    -v /path/to/slime_save:/root/Qwen3-4B_slime \
    -v $HOME/.ssh:/root/.ssh:ro \
    -v $HOME/.cache/huggingface:/root/.cache/huggingface \
    -e HF_TOKEN=... \
    -e WANDB_API_KEY=... \
    slime-llm4cov:<tag> bash
```

The `-v $HOME/.ssh:/root/.ssh:ro` mount lets the rollout's
`llm4cov.eda_client.remote_*` reach the EDA host via your SSH config alias
(e.g. `paladin_centos`). Container-side OpenSSH must trust the host key —
either pre-populate `known_hosts` in your mount or set
`StrictHostKeyChecking=accept-new` in `~/.ssh/config`.

## First-time setup: starting from a HuggingFace model

`--hf-checkpoint` is fed straight into `AutoConfig.from_pretrained`, which
accepts either a local dir or a HF model ID (auto-downloads to
`~/.cache/huggingface/`). `--ref-load` is different — it points at a
*Megatron `torch_dist`* checkpoint, which doesn't exist on HuggingFace and
must be produced once via `tools/convert_hf_to_torch_dist.py`.

Worked example: start from `hez2024/LLM4Cov-Qwen3-4B-SFT-Stage0` (an SFT'd
Qwen3-4B-Instruct-2507; rotary base 5,000,000 — different from the base
Qwen3-4B's 1,000,000).

```bash
cd /root/slime
export HF_TOKEN=...   # if the repo is gated, otherwise skip

MODEL_ID=hez2024/LLM4Cov-Qwen3-4B-SFT-Stage0
LOCAL_DIR=/root/LLM4Cov-Qwen3-4B-SFT-Stage0

# 1. Download HF weights (you can also skip this — passing the HF ID
#    directly to step 2/3 would auto-cache to ~/.cache/huggingface, but an
#    explicit local dir is faster on container restarts).
huggingface-cli download "${MODEL_ID}" --local-dir "${LOCAL_DIR}"

# 2. One-time conversion to Megatron torch_dist. Pass MODEL_ARGS so vocab,
#    layer count, hidden size etc. line up with the slime/Megatron loader.
#    --rotary-base 5000000 overrides the default in scripts/models/qwen3-4B.sh
#    to match the Qwen3-4B-Instruct-2507 base this model was SFT'd from.
source scripts/models/qwen3-4B.sh
PYTHONPATH=/root/Megatron-LM torchrun --nproc_per_node 1 \
    tools/convert_hf_to_torch_dist.py \
    "${MODEL_ARGS[@]}" \
    --hf-checkpoint "${LOCAL_DIR}" \
    --rotary-base 5000000 \
    --save "${LOCAL_DIR}_torch_dist"

# 3. Launch training. MODEL_ARGS_ROTARY_BASE flows into the same
#    --rotary-base flag the run script sources from qwen3-4B.sh.
MODEL_ARGS_ROTARY_BASE=5000000 \
HF_CKPT="${LOCAL_DIR}" \
REF_LOAD="${LOCAL_DIR}_torch_dist" \
SAVE_DIR="${LOCAL_DIR}_slime" \
EDA_SERVER=paladin_centos \
EDA_REPO_DIR=/workspace/llm4cov_eda \
bash examples/agentic_cov/run-qwen3-4B-4xH200-noeval.sh
```

The conversion in step 2 takes ~5 min on a single H100/H200 and produces a
~9 GB `*_torch_dist/` directory. Re-run only when you want to start from a
different base checkpoint.

For a base Qwen3-4B (no instruct tune) the conversion line drops the
`--rotary-base` override and the launch command drops the
`MODEL_ARGS_ROTARY_BASE` env var.

## Run the training script

For subsequent runs with the same starting checkpoint, just:

```bash
cd /root/slime
EDA_SERVER=paladin_centos \
EDA_REPO_DIR=/workspace/llm4cov_eda \
bash examples/agentic_cov/run-qwen3-4B-4xH200-noeval.sh
```

Override paths via env vars at the top of the script:

| variable | default | purpose |
|---|---|---|
| `HF_CKPT` | `/root/Qwen3-4B` | HF policy checkpoint |
| `REF_LOAD` | `/root/Qwen3-4B_torch_dist` | torch_dist conversion of policy |
| `SAVE_DIR` | `/root/Qwen3-4B_slime` | slime save/load dir |
| `EDA_SERVER` | _required_ | SSH alias for the EDA worker |
| `EDA_REPO_DIR` | _required_ | path to `llm4cov_eda` checkout on the EDA host |
| `LLM4COV_DATASET` | `zhuyaoyu/CodeV-R1-dataset` | HF dataset for training prompts |
| `LLM4COV_SPLIT` | `train` | split name |
| `NUM_GPUS` | autodetected | override if you want fewer than all visible |

The script does not configure eval (`--eval-interval` is unset). To add
eval back, copy this script and add `--eval-interval`,
`--eval-function-path examples.agentic_cov.rollout.eval_rollout`, and
`--llm4cov-eval-dataset-name` / `--llm4cov-eval-dataset-split`.
