import hashlib
import importlib.util
import json
from argparse import Namespace
from pathlib import Path

import pytest


def _load_train_module():
    path = Path(__file__).resolve().parents[1] / "train.py"
    spec = importlib.util.spec_from_file_location("train_marker_under_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_resume_marker_requires_and_records_global_dataset_state(tmp_path):
    train = _load_train_module()
    iteration = 49
    native_dir = tmp_path / f"iter_{iteration:07d}"
    native_dir.mkdir()
    native_marker = native_dir / ".complete.json"
    native_marker.write_text(
        json.dumps(
            {
                "version": 1,
                "kind": "torch_dist",
                "iteration": iteration,
                "files": [],
            }
        ),
        encoding="utf-8",
    )
    rollout_state = tmp_path / "rollout" / f"global_dataset_state_dict_{iteration}.pt"
    rollout_state.parent.mkdir()
    rollout_state.write_bytes(b"dataset-state")
    args = Namespace(save=str(tmp_path), rollout_global_dataset=True)

    train.publish_resume_ready_marker(args, iteration)

    marker = json.loads((tmp_path / f"resume_ready_step_{iteration}.json").read_text(encoding="utf-8"))
    assert marker["kind"] == "slime_resume_bundle"
    assert marker["native_checkpoint"]["path"] == native_dir.name
    assert marker["native_checkpoint"]["complete_marker_sha256"] == hashlib.sha256(native_marker.read_bytes()).hexdigest()
    assert marker["rollout_state"]["path"] == f"rollout/global_dataset_state_dict_{iteration}.pt"
    assert marker["rollout_state"]["sha256"] == hashlib.sha256(rollout_state.read_bytes()).hexdigest()


def test_resume_marker_rejects_missing_global_dataset_state(tmp_path):
    train = _load_train_module()
    iteration = 49
    native_dir = tmp_path / f"iter_{iteration:07d}"
    native_dir.mkdir()
    (native_dir / ".complete.json").write_text(
        json.dumps(
            {
                "version": 1,
                "kind": "torch_dist",
                "iteration": iteration,
                "files": [],
            }
        ),
        encoding="utf-8",
    )
    args = Namespace(save=str(tmp_path), rollout_global_dataset=True)

    with pytest.raises(RuntimeError, match="rollout dataset state is missing"):
        train.publish_resume_ready_marker(args, iteration)

    assert not (tmp_path / f"resume_ready_step_{iteration}.json").exists()
