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
- A data-disk dir (`ROOT_DIR`) where the run script materializes the HF
  checkpoint, the Megatron `torch_dist` conversion, and the slime
  load/save dir — all derived from `MODEL_NAME` (see "Run the training
  script" below).
- Reachable EDA worker host (SSH alias) and the `llm4cov_eda` checkout on it
- Optionally: HuggingFace cache for the llm4cov dataset and any wandb creds

```bash
docker run --gpus all --shm-size=32g --network=host --ipc=host --rm -it \
    -v /path/to/data_disk:/root \
    -v $HOME/.ssh:/run/host-ssh:ro \
    -v $HOME/.cache/huggingface:/root/.cache/huggingface \
    -e HF_TOKEN=... \
    -e WANDB_API_KEY=... \
    slime-llm4cov:<tag> bash
```

Mount whatever data-disk path you want as the container's `ROOT_DIR`
(default `/root`). The run script populates `${ROOT_DIR}/<model>`,
`${ROOT_DIR}/<model>_torch_dist`, and `${ROOT_DIR}/<model>_slime` on
first launch.

The image's `ENTRYPOINT` copies `/run/host-ssh` into `/root/.ssh` on
start (fixing modes to what OpenSSH expects), so the rollout's
`llm4cov.eda_client.remote_*` can reach the EDA host via your SSH config
alias (e.g. `paladin_centos`). Container-side OpenSSH must trust the host
key — either pre-populate `known_hosts` in your mount or set
`StrictHostKeyChecking=accept-new` in your SSH `config`. The mount is
required; the entrypoint exits with an error if `/run/host-ssh` is
missing. To run the image without SSH (rare — smoke tests only), pass
`--entrypoint=''` to bypass the bootstrap.

### Per-host launcher: `container_launch/`

For a repeatable `make build|run|attach` flow with SSH keys + git identity
wired in automatically, see `examples/agentic_cov/container_launch/`. It
holds the matching `Makefile`, `.gitconfig`, and `.ssh/` for the host
running the container — git-ignored, deployed to each server via
`scp -r container_launch`. The only tracked launcher files are the
entrypoint and a local README.

## Run the training script

Each `run-*.sh` sources `_setup_checkpoints.sh`, which derives the three
checkpoint paths from `MODEL_NAME` + `ROOT_DIR` and runs first-time setup
on demand:

- `HF_CKPT  = ${ROOT_DIR}/<basename(MODEL_NAME)>` — downloaded via
  `huggingface-cli` if missing/empty.
- `REF_LOAD = ${HF_CKPT}_torch_dist` — produced once by
  `tools/convert_hf_to_torch_dist.py` (~5 min on a single H100/H200,
  ~9 GB) if missing/empty. `--hf-checkpoint` accepts a local dir or a HF
  ID; we materialize a local dir to keep container restarts fast.
- `SAVE_DIR = ${HF_CKPT}_slime` — slime's load/save dir. If non-empty,
  the helper prompts `[o]verwrite / [r]esume / [s]top`. Set
  `SAVE_DIR_ON_EXIST=overwrite|resume|stop` to skip the prompt in
  non-interactive runs.

Default launch starts from `hez2024/LLM4Cov-Qwen3-4B-SFT-Stage0` (an SFT'd
Qwen3-4B-Instruct-2507; rotary base 5,000,000) trained on
`hez2024/CodeV-R1-dataset-RL-test`, with everything materialized under
`/root`:

```bash
cd /root/slime
export HF_TOKEN=...   # if the SFT model is gated, otherwise skip

EDA_SERVER=paladin_centos \
EDA_REPO_DIR=/workspace/llm4cov_eda \
bash examples/agentic_cov/run-qwen3-4B-4xH200-noeval.sh
```

Other shapes:

- `run-qwen3-4B-4xH100-noeval.sh` — same as above, sized for 80 GB H100.
- `run-qwen3-4B-4xH100.sh` — 4xH100 with periodic eval on
  `hez2024/cvdp_ecov_eval` (`--eval-interval 20`, dispatches to
  `examples.agentic_cov.rollout.eval_rollout`).

`ROTARY_BASE` flows into the conversion's `--rotary-base` flag and
`MODEL_ARGS_ROTARY_BASE` overrides the same flag at training time
(sourced from `scripts/models/qwen3-4B.sh`); both default to `5000000` to
match the default model. To start from the base Qwen3-4B (rotary base
1,000,000) on a different data-disk root:

```bash
MODEL_NAME=Qwen/Qwen3-4B \
ROOT_DIR=/data \
ROTARY_BASE=1000000 \
MODEL_ARGS_ROTARY_BASE=1000000 \
EDA_SERVER=paladin_centos \
EDA_REPO_DIR=/workspace/llm4cov_eda \
bash examples/agentic_cov/run-qwen3-4B-4xH200-noeval.sh
```

Env vars consumed by the run scripts:

| variable | default | purpose |
|---|---|---|
| `MODEL_NAME` | `hez2024/LLM4Cov-Qwen3-4B-SFT-Stage0` | HF model id; `<basename>` becomes the local dir under `ROOT_DIR` |
| `ROOT_DIR` | `/root` | parent dir on the data disk for `HF_CKPT` / `REF_LOAD` / `SAVE_DIR` |
| `ROTARY_BASE` | `5000000` | `--rotary-base` for HF→torch_dist conversion |
| `MODEL_ARGS_ROTARY_BASE` | `5000000` | `--rotary-base` injected into `MODEL_ARGS` at training time |
| `SAVE_DIR_ON_EXIST` | _prompt_ | `overwrite` / `resume` / `stop` to skip the interactive prompt |
| `EDA_SERVER` | _required_ | SSH alias for the EDA worker |
| `EDA_REPO_DIR` | _required_ | path to `llm4cov_eda` checkout on the EDA host |
| `LLM4COV_DATASET` | `hez2024/CodeV-R1-dataset-RL-test` | HF dataset for training prompts |
| `LLM4COV_SPLIT` | `train` | split name |
| `LLM4COV_EVAL_DATASET` | `hez2024/cvdp_ecov_eval` | HF dataset for eval prompts (`run-qwen3-4B-4xH100.sh`) |
| `LLM4COV_EVAL_SPLIT` | `eval` | eval split name (`run-qwen3-4B-4xH100.sh`) |
| `NUM_GPUS` | autodetected | override if you want fewer than all visible |
