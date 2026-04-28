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

The image bundles slime, Megatron-LM, sglang, TransformerEngine, and the
`third_party/llm4cov_oss` submodule (for `llm4cov.datasets`, `llm4cov.llm_query`,
`llm4cov.eda_client`).

From the slime repo root:

```bash
SLIME_REPO_URL=https://github.com/HejiaZ2023/slime.git
SLIME_COMMIT=$(git rev-parse HEAD)        # or a fixed sha

docker build -f docker/Dockerfile . \
    --build-arg SLIME_REPO_URL="${SLIME_REPO_URL}" \
    --build-arg SLIME_COMMIT="${SLIME_COMMIT}" \
    --build-arg HTTP_PROXY="$http_proxy" \
    --build-arg HTTPS_PROXY="$https_proxy" \
    --build-arg NO_PROXY="localhost,127.0.0.1" \
    -t slime-llm4cov:${SLIME_COMMIT::7}
```

The build clones `${SLIME_REPO_URL}@${SLIME_COMMIT}` with
`--recurse-submodules`, so the llm4cov_oss submodule pinned in `.gitmodules`
is checked out and `pip install -e`-installed inside the image.

The `transformers`, `pydantic`, `openai`, etc. floors used during the llm4cov
install are kept low enough that pip does not bump `transformers` away from
the 4.57.1 that `sglang==0.5.9` pins exactly.

For NVIDIA DGX Spark (GB10/sm_121a, arm64) use `docker/Dockerfile.gb10`
instead — it rebases on the NGC vLLM container and still copies the local
slime tree (no `SLIME_REPO_URL`/`SLIME_COMMIT` build args).

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

## Run the training script

Inside the container:

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
