#!/usr/bin/env bash
set -euo pipefail

# Run this on the teacher Brev host after llm4cov_slime is running.
# It downloads only the step999 checkpoint directory from each HF repo, then
# flattens it to the paths expected by the OPD teacher launcher.

CONTAINER="${CONTAINER:-llm4cov_slime}"
DEST_ROOT="${DEST_ROOT:-/data/final_ckpts}"
STAGE0_REPO="${STAGE0_REPO:-Senlimulin/2026UCSDIntern_Stage0_elft_elfe}"
STAGE1_REPO="${STAGE1_REPO:-Senlimulin/2026UCSDIntern_Stage1_elft_elfe}"
HF_STEP_DIR="${HF_STEP_DIR:-step999}"
HF_STEP_DIR_FALLBACK="${HF_STEP_DIR_FALLBACK:-step_999}"
HOST_HF_ENV="${HOST_HF_ENV:-$HOME/.secrets/hf_sync.env}"

if ! docker exec "$CONTAINER" bash -lc 'test -n "${HF_SYNC_TOKEN:-}${HF_TOKEN:-}"'; then
  if [[ ! -f "$HOST_HF_ENV" ]]; then
    echo "Missing HF token in container and host env file not found: $HOST_HF_ENV" >&2
    exit 1
  fi
  docker cp "$HOST_HF_ENV" "$CONTAINER:/tmp/hf_sync.env"
  docker exec "$CONTAINER" chmod 600 /tmp/hf_sync.env
fi

docker exec \
  -e DEST_ROOT="$DEST_ROOT" \
  -e STAGE0_REPO="$STAGE0_REPO" \
  -e STAGE1_REPO="$STAGE1_REPO" \
  -e HF_STEP_DIR="$HF_STEP_DIR" \
  -e HF_STEP_DIR_FALLBACK="$HF_STEP_DIR_FALLBACK" \
  "$CONTAINER" bash -lc 'set -euo pipefail
if [[ -z "${HF_SYNC_TOKEN:-}${HF_TOKEN:-}" && -f /tmp/hf_sync.env ]]; then
  set -a
  source /tmp/hf_sync.env
  set +a
fi
trap "rm -f /tmp/hf_sync.env" EXIT
python3 - <<'"'"'PY'"'"'
import os
import shutil
from pathlib import Path

from huggingface_hub import snapshot_download

dest_root = Path(os.environ["DEST_ROOT"])
step_dir = os.environ.get("HF_STEP_DIR", "step999").strip("/") or "step999"
fallback_step_dir = os.environ.get("HF_STEP_DIR_FALLBACK", "step_999").strip("/")
token = os.environ.get("HF_SYNC_TOKEN") or os.environ.get("HF_TOKEN")
if not token:
    raise SystemExit(
        "Missing HF token env in container. Restart llm4cov_slime with the HF env-file "
        "or export HF_SYNC_TOKEN/HF_TOKEN before running this script."
    )

pairs = [
    (os.environ["STAGE0_REPO"], "stage0_step999"),
    (os.environ["STAGE1_REPO"], "stage1_step999"),
]

def has_required_files(path: Path) -> bool:
    return (
        (path / "config.json").exists()
        and any(path.glob("model-*.safetensors"))
        and (path / "tokenizer_config.json").exists()
    )

def download_one(repo_id: str, name: str) -> None:
    out_dir = dest_root / name
    if has_required_files(out_dir):
        print(f"already_present {out_dir}", flush=True)
        return

    tmp_root = Path("/data/hf_teacher_downloads") / name
    if tmp_root.exists():
        shutil.rmtree(tmp_root)

    tried = []
    for candidate in [step_dir, fallback_step_dir]:
        if not candidate or candidate in tried:
            continue
        tried.append(candidate)
        if tmp_root.exists():
            shutil.rmtree(tmp_root)
        print(f"download {repo_id}/{candidate} -> {out_dir}", flush=True)
        try:
            snapshot_download(
                repo_id=repo_id,
                repo_type="model",
                local_dir=str(tmp_root),
                allow_patterns=[f"{candidate}/*"],
                token=token,
            )
        except Exception as exc:
            print(f"download_failed {repo_id}/{candidate}: {type(exc).__name__}", flush=True)
            continue
        src = tmp_root / candidate
        if has_required_files(src):
            if out_dir.exists():
                shutil.rmtree(out_dir)
            out_dir.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(src, out_dir)
            print(f"ready {out_dir}", flush=True)
            return
        print(f"missing_required_files {repo_id}/{candidate}", flush=True)

    raise SystemExit(f"Could not find a usable step999 checkpoint in {repo_id}; tried {tried}")

for repo_id, name in pairs:
    download_one(repo_id, name)

for path in [dest_root / "stage0_step999", dest_root / "stage1_step999"]:
    print(f"verify {path}", flush=True)
    for pattern in ["config.json", "tokenizer_config.json", "model-*.safetensors"]:
        matches = sorted(path.glob(pattern))
        if not matches:
            raise SystemExit(f"missing {pattern} in {path}")
        for item in matches:
            print(f"  {item.name} {item.stat().st_size}", flush=True)
PY'
