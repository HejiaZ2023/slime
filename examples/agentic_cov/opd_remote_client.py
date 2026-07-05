"""File-exchange client for llm4cov OPD teacher/EDA relay jobs.

The relay reuses the existing paladin xfer docker: jobs are published under
``incoming/<namespace>/<job_id>`` and results are read from
``results/<namespace>/<job_id>``.  The existing EDA watcher only processes
top-level ``incoming/<job>`` directories, so the ``opd`` namespace keeps OPD
jobs separate while still using the same mounted SFTP paths.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import shutil
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "opd_exchange_v1"


def _json_dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, indent=2)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _safe_relpath(path: str) -> str:
    rel = Path(path)
    if rel.is_absolute() or ".." in rel.parts:
        raise ValueError(f"unsafe relative path: {path!r}")
    return rel.as_posix()


def _encode_text(text: str) -> bytes:
    return text.encode("utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise ValueError(f"expected JSON object at {path}")
    return obj


@dataclass(frozen=True)
class TeacherSpec:
    name: str
    n: int


def parse_teacher_specs(spec: str | None) -> list[TeacherSpec]:
    """Parse ``stage0:1,stage1:1`` style teacher rollout specs."""
    if not spec:
        return []
    out: list[TeacherSpec] = []
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if ":" in chunk:
            name, n_raw = chunk.split(":", 1)
            n = int(n_raw)
        else:
            name, n = chunk, 1
        name = name.strip()
        if not name:
            raise ValueError(f"bad OPD teacher spec chunk: {chunk!r}")
        if n < 0:
            raise ValueError(f"teacher sample count must be >= 0: {chunk!r}")
        out.append(TeacherSpec(name=name, n=n))
    return out


def build_job_id(dataset_id: str, rollout_id: int, round_idx: int) -> str:
    cleaned = "".join(c if c.isalnum() or c in "._-" else "_" for c in dataset_id)[:80]
    return f"opd_{rollout_id:06d}_r{round_idx}_{cleaned}_{uuid.uuid4().hex[:8]}"


def make_file_manifest(files: dict[str, bytes]) -> dict[str, dict[str, Any]]:
    return {
        rel: {"sha256": _sha256_bytes(data), "bytes": len(data)}
        for rel, data in sorted(files.items())
    }


class _LocalTreeTransport:
    def __init__(self, root: str, namespace: str) -> None:
        self.root = Path(root)
        self.namespace = namespace

    def _incoming(self) -> Path:
        return self.root / "incoming" / self.namespace

    def _results(self) -> Path:
        return self.root / "results" / self.namespace

    def submit(self, job_id: str, files: dict[str, bytes]) -> None:
        incoming = self._incoming()
        incoming.mkdir(parents=True, exist_ok=True)
        staging = incoming / f"{job_id}.staging"
        final = incoming / job_id
        if staging.exists():
            shutil.rmtree(staging)
        staging.mkdir(parents=True)
        for rel, data in files.items():
            rel = _safe_relpath(rel)
            dst = staging / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(data)
        if final.exists():
            shutil.rmtree(final)
        os.rename(staging, final)

    def wait_result(self, job_id: str, timeout_s: float, poll_s: float) -> Path:
        result_dir = self._results() / job_id
        done = result_dir / ".done"
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            if done.exists():
                return result_dir
            time.sleep(poll_s)
        raise TimeoutError(f"OPD relay result timeout: {job_id}")

    def close(self) -> None:
        return None


class _SftpTreeTransport:
    def __init__(self, host: str, port: int, user: str, key: str, namespace: str) -> None:
        import paramiko  # type: ignore[import-untyped]

        self.namespace = namespace
        self.client = paramiko.SSHClient()
        self.client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        self.client.connect(
            host,
            port=port,
            username=user,
            key_filename=key,
            look_for_keys=False,
            allow_agent=False,
            timeout=20,
        )
        self.sftp = self.client.open_sftp()

    def _mkdirs(self, path: str) -> None:
        cur = ""
        for part in path.strip("/").split("/"):
            cur = f"{cur}/{part}" if cur else part
            with contextlib.suppress(OSError):
                self.sftp.mkdir(cur)

    def _exists(self, path: str) -> bool:
        try:
            self.sftp.stat(path)
            return True
        except OSError:
            return False

    def _rmtree(self, path: str) -> None:
        try:
            entries = self.sftp.listdir_attr(path)
        except OSError:
            return
        for entry in entries:
            child = f"{path}/{entry.filename}"
            if entry.st_mode & 0o040000:
                self._rmtree(child)
            else:
                with contextlib.suppress(OSError):
                    self.sftp.remove(child)
        with contextlib.suppress(OSError):
            self.sftp.rmdir(path)

    def submit(self, job_id: str, files: dict[str, bytes]) -> None:
        base = f"incoming/{self.namespace}"
        self._mkdirs(base)
        staging = f"{base}/{job_id}.staging"
        final = f"{base}/{job_id}"
        self._rmtree(staging)
        self._mkdirs(staging)
        for rel, data in files.items():
            rel = _safe_relpath(rel)
            dst = f"{staging}/{rel}"
            parent = str(Path(dst).parent).replace("\\", "/")
            self._mkdirs(parent)
            with self.sftp.open(dst, "wb") as f:
                f.write(data)
        self._rmtree(final)
        self.sftp.posix_rename(staging, final)

    def wait_result(self, job_id: str, timeout_s: float, poll_s: float) -> Path:
        remote = f"results/{self.namespace}/{job_id}"
        done = f"{remote}/.done"
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            if self._exists(done):
                local = Path(os.environ.get("OPD_RESULT_TMP", "/tmp/llm4cov_opd_results")) / job_id
                if local.exists():
                    shutil.rmtree(local)
                local.mkdir(parents=True)
                self._download_tree(remote, local)
                return local
            time.sleep(poll_s)
        raise TimeoutError(f"OPD relay result timeout: {job_id}")

    def _download_tree(self, remote_dir: str, local_dir: Path) -> None:
        local_dir.mkdir(parents=True, exist_ok=True)
        for entry in self.sftp.listdir_attr(remote_dir):
            remote_child = f"{remote_dir}/{entry.filename}"
            local_child = local_dir / entry.filename
            if entry.st_mode & 0o040000:
                self._download_tree(remote_child, local_child)
            else:
                self.sftp.get(remote_child, str(local_child))

    def close(self) -> None:
        with contextlib.suppress(Exception):
            self.sftp.close()
        with contextlib.suppress(Exception):
            self.client.close()


class OpdRelayClient:
    def __init__(self, args: Any) -> None:
        self.namespace = getattr(args, "opd_namespace", "opd")
        transport = getattr(args, "opd_transport", None) or os.environ.get("OPD_TRANSPORT", "")
        server = getattr(args, "opd_server", None) or os.environ.get("OPD_SERVER", "")
        if not transport:
            transport = "local" if server in {"", "local", "paladin", "paladin_centos"} else "sftp"
        if transport == "local":
            root = getattr(args, "opd_xfer_dir", None) or os.environ.get("OPD_XFER_DIR", "/mnt/raid0_ssd/eda/xfer")
            self.transport = _LocalTreeTransport(root, self.namespace)
        elif transport == "sftp":
            host = getattr(args, "opd_sftp_host", None) or os.environ.get("OPD_SFTP_HOST", server)
            port = int(getattr(args, "opd_sftp_port", None) or os.environ.get("OPD_SFTP_PORT", "2222"))
            user = getattr(args, "opd_sftp_user", None) or os.environ.get("OPD_SFTP_USER", "gpujobs")
            key = getattr(args, "opd_sftp_key", None) or os.environ.get(
                "OPD_SFTP_KEY", os.path.expanduser("~/.ssh/brev_eda_sftp")
            )
            self.transport = _SftpTreeTransport(host, port, user, key, self.namespace)
        else:
            raise ValueError(f"unknown OPD transport: {transport}")

    def submit(self, job_id: str, files: dict[str, bytes]) -> None:
        self.transport.submit(job_id, files)

    def wait_result(self, job_id: str, *, timeout_s: float, poll_s: float = 2.0) -> Path:
        return self.transport.wait_result(job_id, timeout_s=timeout_s, poll_s=poll_s)

    def submit_and_wait(
        self,
        job_id: str,
        files: dict[str, bytes],
        *,
        timeout_s: float,
        poll_s: float = 2.0,
    ) -> Path:
        self.submit(job_id, files)
        return self.wait_result(job_id, timeout_s=timeout_s, poll_s=poll_s)

    def close(self) -> None:
        self.transport.close()


def build_round_files(
    *,
    job_id: str,
    dataset_id: str,
    rollout_id: int,
    round_idx: int,
    prompt: str,
    state: dict[str, Any],
    context: dict[str, Any],
    student_rollouts: list[dict[str, Any]],
    teachers: list[TeacherSpec],
    sampling_params: dict[str, Any],
    want_detail: bool,
    score_student_rollouts: bool = False,
    topk_k: int = 0,
    teacher_rollouts: list[dict[str, Any]] | None = None,
    generate_teacher_rollouts: bool = True,
) -> dict[str, bytes]:
    files: dict[str, bytes] = {
        "prompt.txt": _encode_text(prompt),
        "state.json": _encode_text(_json_dumps(state)),
        "context.json": _encode_text(_json_dumps(context)),
    }
    student_entries = []
    if teacher_rollouts is not None:
        files["teacher_rollouts.json"] = _encode_text(_json_dumps({"teacher_rollouts": teacher_rollouts}))

    for i, rollout in enumerate(student_rollouts):
        sid = rollout.get("id") or f"s{i:03d}"
        base = f"student/{sid}"
        response = str(rollout.get("assistant_response") or "")
        meta = {k: v for k, v in rollout.items() if k not in {"assistant_response", "testbench"}}
        files[f"{base}/assistant_response.txt"] = _encode_text(response)
        files[f"{base}/meta.json"] = _encode_text(_json_dumps(meta))
        if rollout.get("testbench") is not None:
            files[f"{base}/testbench.sv"] = _encode_text(str(rollout["testbench"]))
        student_entries.append({"id": sid, "path": base})

    manifest = {
        "version": SCHEMA_VERSION,
        "job_id": job_id,
        "dataset_id": dataset_id,
        "rollout_id": rollout_id,
        "round_idx": round_idx,
        "prompt_file": "prompt.txt",
        "state_file": "state.json",
        "context_file": "context.json",
        "student_rollouts": student_entries,
        "teacher_rollouts_file": "teacher_rollouts.json" if teacher_rollouts is not None else None,
        "teachers": [{"name": t.name, "n": t.n} for t in teachers],
        "sampling_params": sampling_params,
        "eda": {"want_detail": bool(want_detail)},
        "teacher_request": {
            "generate_teacher_rollouts": bool(generate_teacher_rollouts),
            "return_teacher_assistant_response": True,
            "return_teacher_token_ids": True,
            "return_teacher_logprobs": True,
            "return_teacher_topk_logprobs": int(topk_k or 0) > 1,
            "student_topk_k": int(topk_k or 0),
            "score_student_rollouts": bool(score_student_rollouts),
        },
        "files": make_file_manifest(files),
    }
    files["manifest.json"] = _encode_text(_json_dumps(manifest))
    return files


def load_result_tree(result_dir: str | Path) -> dict[str, Any]:
    """Load and validate an OPD relay result directory."""
    result_dir = Path(result_dir)
    result_path = result_dir / "result.json"
    result = _read_json(result_path)
    version = result.get("version")
    if version != SCHEMA_VERSION:
        raise ValueError(
            f"bad OPD result schema version at {result_path}: {version!r} != {SCHEMA_VERSION!r}"
        )
    status = result.get("status", "success")
    if status != "success":
        raise RuntimeError(f"OPD relay job failed: {result.get('error') or result}")
    return result
