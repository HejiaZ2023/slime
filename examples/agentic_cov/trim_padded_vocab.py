"""Trim padded vocab rows from a slime-saved HF checkpoint.

Slime's HF dump preserves Megatron's TP-padded embedding (rounded to
make_vocab_size_divisible_by * TP), but config.json keeps the original
vocab_size. vllm asserts loaded_weight.shape[0] == config.vocab_size and
crashes on the mismatch. This script trims model.embed_tokens.weight (and
lm_head.weight when present and untied) down to config.vocab_size in
place, then refreshes model.safetensors.index.json's total_size.

Padded rows are unused at inference: tokenizer never produces those ids
and they're zero-init, so they never win the lm_head softmax.

Usage:
    python examples/agentic_cov/trim_padded_vocab.py <hf_ckpt_dir>
    # e.g. .../LLM4Cov-Qwen3-4B-SFT-Stage0_slime_hf/step_9

Idempotent — re-running on an already-trimmed dir is a no-op.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import safetensors.torch as stt
from safetensors import safe_open

TRIM_KEYS = {"model.embed_tokens.weight", "lm_head.weight"}


def _shard_total_bytes(path: Path) -> int:
    total = 0
    with safe_open(str(path), framework="pt") as f:
        for k in f.keys():
            t = f.get_tensor(k)
            total += t.numel() * t.element_size()
    return total


def _read_shard(path: Path) -> tuple[dict, dict[str, str]]:
    with safe_open(str(path), framework="pt") as f:
        meta = f.metadata() or {}
        sd = {k: f.get_tensor(k) for k in f.keys()}
    return sd, meta


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("ckpt_dir", type=Path)
    args = parser.parse_args()
    ckpt = args.ckpt_dir

    cfg = json.loads((ckpt / "config.json").read_text())
    target = int(cfg["vocab_size"])
    print(f"config.vocab_size = {target}")

    index_path = ckpt / "model.safetensors.index.json"
    single_path = ckpt / "model.safetensors"

    if index_path.exists():
        index = json.loads(index_path.read_text())
        weight_map = index["weight_map"]
        single_shard = False
    elif single_path.exists():
        # one-shard case: synthesize a weight_map from the file's keys
        with safe_open(str(single_path), framework="pt") as f:
            keys = list(f.keys())
        weight_map = {k: "model.safetensors" for k in keys}
        index = None
        single_shard = True
    else:
        raise SystemExit(f"no safetensors found under {ckpt}")

    present = TRIM_KEYS & set(weight_map.keys())
    if not present:
        raise SystemExit(
            f"none of {TRIM_KEYS} found in {ckpt}; check the dir is the right HF dump"
        )

    affected = sorted({weight_map[k] for k in present})
    print(f"shards to inspect: {affected}")

    changed_any = False
    for shard in affected:
        shard_path = ckpt / shard
        sd, meta = _read_shard(shard_path)
        shard_changed = False
        for k in present:
            if k not in sd:
                continue
            row_dim = sd[k].shape[0]
            if row_dim == target:
                print(f"  {shard}::{k} already at {target} rows, skipping")
                continue
            if row_dim < target:
                raise SystemExit(
                    f"  {shard}::{k} has {row_dim} rows but config.vocab_size={target}; "
                    "refusing to grow"
                )
            print(f"  {shard}::{k}: {tuple(sd[k].shape)} -> ({target}, ...)")
            sd[k] = sd[k][:target].contiguous().clone()
            shard_changed = True

        if shard_changed:
            stt.save_file(sd, str(shard_path), metadata=meta or {"format": "pt"})
            changed_any = True

    if not changed_any:
        print("no changes; checkpoint already trimmed")
        return

    if not single_shard:
        assert index is not None
        # refresh total_size in the index
        total = 0
        for shard in sorted(set(weight_map.values())):
            total += _shard_total_bytes(ckpt / shard)
        index_meta = index.setdefault("metadata", {})
        old = index_meta.get("total_size")
        index_meta["total_size"] = total
        index_path.write_text(json.dumps(index, indent=2))
        print(f"updated {index_path.name}: total_size {old} -> {total}")

    print("done")


if __name__ == "__main__":
    main()
