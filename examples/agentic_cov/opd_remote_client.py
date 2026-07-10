"""File-exchange client for llm4cov OPD teacher/EDA relay jobs.

The relay reuses the existing paladin xfer docker: jobs are published under
``incoming/<namespace>/<job_id>`` and results are read from
``results/<namespace>/<job_id>``.  The existing EDA watcher only processes
top-level ``incoming/<job>`` directories, so the ``opd`` namespace keeps OPD
jobs separate while still using the same mounted SFTP paths.
"""

from __future__ import annotations

import base64
import contextlib
import gzip
import hashlib
import io
import json
import os
import shutil
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import urllib.error
import urllib.request


SCHEMA_VERSION = "opd_exchange_v1"
COMPACT_STUDENT_REQUEST = os.environ.get("OPD_COMPACT_STUDENT_REQUEST", "1") != "0"
STUDENT_REQUEST_ARRAY_DTYPES = {
    "input_token_ids": "int32",
    "topk_token_ids": "int32",
}


def _json_dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, indent=2)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


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


def _attach_student_tensor_sidecar(files: dict[str, bytes], base: str, meta: dict[str, Any]) -> None:
    """Move large student scoring arrays out of JSON into a compact npz sidecar."""
    if not COMPACT_STUDENT_REQUEST:
        return

    import numpy as np  # type: ignore[import-untyped]

    arrays: dict[str, Any] = {}
    array_meta: dict[str, dict[str, Any]] = {}
    for key, dtype in STUDENT_REQUEST_ARRAY_DTYPES.items():
        if key not in meta:
            continue
        value = meta.get(key)
        if value is None or (isinstance(value, (list, tuple)) and len(value) == 0):
            continue
        arr = np.asarray(value, dtype=dtype)
        if arr.size == 0:
            continue
        arrays[key] = arr
        array_meta[key] = {"dtype": str(arr.dtype), "shape": [int(dim) for dim in arr.shape]}
        meta.pop(key, None)

    if not arrays:
        return
    buf = io.BytesIO()
    np.savez_compressed(buf, **arrays)
    data = buf.getvalue()
    rel = f"{base}/student_tensors.npz"
    files[rel] = data
    meta["student_tensor_sidecar"] = {
        "format": "student_npz_v1",
        "file": rel,
        "sha256": _sha256_bytes(data),
        "arrays": array_meta,
    }


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
        self.last_timing: dict[str, Any] = {}

    def _incoming(self) -> Path:
        return self.root / "incoming" / self.namespace

    def _results(self) -> Path:
        return self.root / "results" / self.namespace

    def submit(self, job_id: str, files: dict[str, bytes]) -> None:
        started = time.monotonic()
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
        self.last_timing["submit"] = {
            "seconds": time.monotonic() - started,
            "file_count": len(files),
            "payload_bytes": sum(len(data) for data in files.values()),
            "transport": "local",
        }

    def wait_result(self, job_id: str, timeout_s: float, poll_s: float) -> Path:
        started = time.monotonic()
        result_dir = self._results() / job_id
        done = result_dir / ".done"
        deadline = time.time() + timeout_s
        polls = 0
        while time.time() < deadline:
            if done.exists():
                self.last_timing["wait_result"] = {
                    "wait_seconds": time.monotonic() - started,
                    "download_seconds": 0.0,
                    "download_files": 0,
                    "download_bytes": 0,
                    "polls": polls,
                    "result_path": str(result_dir),
                    "transport": "local",
                }
                return result_dir
            polls += 1
            time.sleep(poll_s)
        raise TimeoutError(f"OPD relay result timeout: {job_id}")

    def close(self) -> None:
        return None


class _SftpTreeTransport:
    def __init__(self, host: str, port: int, user: str, key: str, namespace: str) -> None:
        import paramiko  # type: ignore[import-untyped]

        self.namespace = namespace
        self.last_timing: dict[str, Any] = {}
        started = time.monotonic()
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
        self.last_timing["connect"] = {
            "seconds": time.monotonic() - started,
            "transport": "sftp",
            "host": host,
            "port": port,
            "user": user,
        }

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
        started = time.monotonic()
        base = f"incoming/{self.namespace}"
        self._mkdirs(base)
        staging = f"{base}/{job_id}.staging"
        final = f"{base}/{job_id}"
        self._rmtree(staging)
        self._mkdirs(staging)
        upload_started = time.monotonic()
        payload_bytes = 0
        for rel, data in files.items():
            rel = _safe_relpath(rel)
            dst = f"{staging}/{rel}"
            parent = str(Path(dst).parent).replace("\\", "/")
            self._mkdirs(parent)
            with self.sftp.open(dst, "wb") as f:
                f.write(data)
            payload_bytes += len(data)
        upload_seconds = time.monotonic() - upload_started
        self._rmtree(final)
        self.sftp.posix_rename(staging, final)
        self.last_timing["submit"] = {
            "seconds": time.monotonic() - started,
            "upload_seconds": upload_seconds,
            "file_count": len(files),
            "payload_bytes": payload_bytes,
            "transport": "sftp",
        }

    def wait_result(self, job_id: str, timeout_s: float, poll_s: float) -> Path:
        started = time.monotonic()
        remote = f"results/{self.namespace}/{job_id}"
        done = f"{remote}/.done"
        deadline = time.time() + timeout_s
        polls = 0
        while time.time() < deadline:
            if self._exists(done):
                wait_seconds = time.monotonic() - started
                local = Path(os.environ.get("OPD_RESULT_TMP", "/tmp/llm4cov_opd_results")) / job_id
                if local.exists():
                    shutil.rmtree(local)
                local.mkdir(parents=True)
                download_started = time.monotonic()
                download_files, download_bytes = self._download_tree(remote, local)
                download_seconds = time.monotonic() - download_started
                self.last_timing["wait_result"] = {
                    "wait_seconds": wait_seconds,
                    "download_seconds": download_seconds,
                    "download_files": download_files,
                    "download_bytes": download_bytes,
                    "polls": polls,
                    "result_path": str(local),
                    "remote_path": remote,
                    "transport": "sftp",
                }
                return local
            polls += 1
            time.sleep(poll_s)
        raise TimeoutError(f"OPD relay result timeout: {job_id}")

    def _download_tree(self, remote_dir: str, local_dir: Path) -> tuple[int, int]:
        local_dir.mkdir(parents=True, exist_ok=True)
        file_count = 0
        byte_count = 0
        download_audit = os.environ.get("OPD_DOWNLOAD_AUDIT", "0") == "1"
        for entry in self.sftp.listdir_attr(remote_dir):
            if entry.filename == "audit" and not download_audit:
                continue
            remote_child = f"{remote_dir}/{entry.filename}"
            local_child = local_dir / entry.filename
            if entry.st_mode & 0o040000:
                child_files, child_bytes = self._download_tree(remote_child, local_child)
                file_count += child_files
                byte_count += child_bytes
            else:
                self.sftp.get(remote_child, str(local_child))
                file_count += 1
                byte_count += int(getattr(entry, "st_size", 0) or 0)
        return file_count, byte_count

    def close(self) -> None:
        with contextlib.suppress(Exception):
            self.sftp.close()
        with contextlib.suppress(Exception):
            self.client.close()


class _HttpTransport:
    def __init__(self, base_url: str, namespace: str, token: str = "") -> None:
        self.base_url = base_url.rstrip("/")
        if not self.base_url.startswith(("http://", "https://")):
            self.base_url = f"http://{self.base_url}"
        self.namespace = namespace
        self.token = token
        self.gzip_requests = os.environ.get("OPD_HTTP_GZIP", "1") != "0"
        self._last_request_wire_bytes = 0
        self._last_request_uncompressed_bytes = 0
        self._last_response_wire_bytes = 0
        self.last_timing: dict[str, Any] = {}
        started = time.monotonic()
        self._request_json("GET", "/healthz", timeout=10)
        self.last_timing["connect"] = {
            "seconds": time.monotonic() - started,
            "transport": "http",
            "base_url": self.base_url,
        }

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json", "Accept-Encoding": "gzip"}
        if self.token:
            headers["X-OPD-Token"] = self.token
        return headers

    def _request_json(self, method: str, path: str, payload: dict[str, Any] | None = None, timeout: float = 30) -> dict[str, Any]:
        body = None
        headers = self._headers()
        if payload is not None:
            body = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
            self._last_request_uncompressed_bytes = len(body)
            if self.gzip_requests:
                body = gzip.compress(body)
                headers["Content-Encoding"] = "gzip"
        else:
            self._last_request_uncompressed_bytes = 0
        self._last_request_wire_bytes = len(body or b"")
        req = urllib.request.Request(
            f"{self.base_url}{path}",
            data=body,
            headers=headers,
            method=method,
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = resp.read()
            encoding = resp.headers.get("Content-Encoding", "")
        self._last_response_wire_bytes = len(data)
        if "gzip" in encoding.lower():
            data = gzip.decompress(data)
        if not data:
            return {}
        obj = json.loads(data.decode("utf-8"))
        if not isinstance(obj, dict):
            raise ValueError(f"bad OPD HTTP response for {path}: expected JSON object")
        return obj

    def submit(self, job_id: str, files: dict[str, bytes]) -> None:
        started = time.monotonic()
        encoded = {
            _safe_relpath(rel): base64.b64encode(data).decode("ascii")
            for rel, data in files.items()
        }
        payload = {"schema": SCHEMA_VERSION, "files": encoded}
        wire_uncompressed_bytes = len(
            json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        )
        upload_started = time.monotonic()
        obj = self._request_json("POST", f"/jobs/{self.namespace}/{job_id}", payload, timeout=120)
        upload_seconds = time.monotonic() - upload_started
        if not obj.get("ok"):
            raise RuntimeError(f"OPD HTTP submit failed: {obj}")
        self.last_timing["submit"] = {
            "seconds": time.monotonic() - started,
            "upload_seconds": upload_seconds,
            "file_count": len(files),
            "payload_bytes": sum(len(data) for data in files.values()),
            "wire_bytes": self._last_request_wire_bytes,
            "wire_uncompressed_bytes": wire_uncompressed_bytes,
            "compression": "gzip" if self.gzip_requests else "none",
            "transport": "http",
        }

    def wait_result(self, job_id: str, timeout_s: float, poll_s: float) -> Path:
        started = time.monotonic()
        deadline = time.time() + timeout_s
        polls = 0
        last_status = "pending"
        while time.time() < deadline:
            request_started = time.monotonic()
            obj = self._request_json("GET", f"/results/{self.namespace}/{job_id}", timeout=120)
            request_seconds = time.monotonic() - request_started
            if obj.get("ok"):
                local = Path(os.environ.get("OPD_RESULT_TMP", "/tmp/llm4cov_opd_results")) / job_id
                if local.exists():
                    shutil.rmtree(local)
                local.mkdir(parents=True)
                download_started = time.monotonic()
                raw_files = obj.get("files") if isinstance(obj.get("files"), dict) else {}
                download_files = 0
                download_bytes = 0
                for rel, data in raw_files.items():
                    rel = _safe_relpath(str(rel))
                    decoded = base64.b64decode(str(data).encode("ascii"))
                    dst = local / rel
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    dst.write_bytes(decoded)
                    download_files += 1
                    download_bytes += len(decoded)
                download_seconds = time.monotonic() - download_started
                self.last_timing["wait_result"] = {
                    "wait_seconds": time.monotonic() - started,
                    "download_seconds": download_seconds,
                    "download_request_seconds": request_seconds,
                    "download_files": download_files,
                    "download_bytes": download_bytes,
                    "wire_bytes": self._last_response_wire_bytes,
                    "result_payload_bytes": int(obj.get("payload_bytes") or 0),
                    "polls": polls,
                    "result_path": str(local),
                    "transport": "http",
                }
                return local
            last_status = str(obj.get("status") or "pending")
            polls += 1
            time.sleep(poll_s)
        raise TimeoutError(f"OPD relay result timeout: {job_id} status={last_status}")

    def close(self) -> None:
        return None


class OpdRelayClient:
    def __init__(self, args: Any) -> None:
        started = time.monotonic()
        self.namespace = getattr(args, "opd_namespace", "opd")
        transport = getattr(args, "opd_transport", None) or os.environ.get("OPD_TRANSPORT", "")
        server = getattr(args, "opd_server", None) or os.environ.get("OPD_SERVER", "")
        if not transport:
            transport = "http" if str(server).startswith(("http://", "https://")) else ("local" if server in {"", "local", "paladin", "paladin_centos"} else "sftp")
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
        elif transport == "http":
            base_url = getattr(args, "opd_http_url", None) or os.environ.get("OPD_HTTP_URL") or server
            if not base_url:
                raise ValueError("--opd-transport http requires --opd-http-url or --opd-server http://...")
            token = os.environ.get("OPD_HTTP_TOKEN", "")
            self.transport = _HttpTransport(base_url, self.namespace, token=token)
        else:
            raise ValueError(f"unknown OPD transport: {transport}")
        self.init_seconds = time.monotonic() - started

    def submit(self, job_id: str, files: dict[str, bytes]) -> None:
        self.transport.submit(job_id, files)

    def wait_result(self, job_id: str, *, timeout_s: float, poll_s: float = 1.0) -> Path:
        return self.transport.wait_result(job_id, timeout_s=timeout_s, poll_s=poll_s)

    def submit_and_wait(
        self,
        job_id: str,
        files: dict[str, bytes],
        *,
        timeout_s: float,
        poll_s: float = 1.0,
    ) -> Path:
        self.submit(job_id, files)
        return self.wait_result(job_id, timeout_s=timeout_s, poll_s=poll_s)

    def close(self) -> None:
        self.transport.close()

    def timing(self) -> dict[str, Any]:
        return {
            "client_init_seconds": self.init_seconds,
            **getattr(self.transport, "last_timing", {}),
        }


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
        meta = {
            k: v
            for k, v in rollout.items()
            if k not in {
                "assistant_response",
                "testbench",
                # The relay only needs full input ids, response length, and
                # top-k token ids to score teacher probabilities.  Student-side
                # logprobs stay local on the training worker.
                "rollout_log_probs",
                "response_token_ids",
                "topk_student_log_probs",
            }
        }
        files[f"{base}/assistant_response.txt"] = _encode_text(response)
        _attach_student_tensor_sidecar(files, base, meta)
        files[f"{base}/meta.json"] = _encode_text(_json_dumps(meta))
        if rollout.get("testbench") is not None:
            files[f"{base}/testbench.sv"] = _encode_text(str(rollout["testbench"]))
        student_entries.append({"id": sid, "path": base})

    manifest = {
        "version": SCHEMA_VERSION,
        "job_kind": "round",
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
            # Teacher rollout logprobs are never consumed by the OPD loss.
            # The exact teacher distribution is obtained later by forced
            # scoring on the student's token support.
            "return_teacher_token_ids": False,
            "return_teacher_logprobs": False,
            "return_teacher_topk_logprobs": int(topk_k or 0) > 1,
            "student_topk_k": int(topk_k or 0),
            "score_student_rollouts": bool(score_student_rollouts),
        },
        "files": make_file_manifest(files),
    }
    files["manifest.json"] = _encode_text(_json_dumps(manifest))
    return files


def build_student_score_files(
    *,
    job_id: str,
    dataset_id: str,
    rollout_id: int,
    round_idx: int,
    student_rollout: dict[str, Any],
    score_teacher_names: list[str],
    topk_k: int,
) -> dict[str, bytes]:
    """Build a minimal forced-score request for one completed student rollout.

    A score-only job deliberately omits the prompt, full EDA context, and
    teacher rollout request.  The student host runs EDA immediately in
    parallel; this request only needs the student's response tokens and
    top-k support to ask every teacher for exact forced probabilities.
    """
    names = list(dict.fromkeys(str(name) for name in score_teacher_names if str(name)))
    if not names:
        raise ValueError("student score request requires at least one teacher")

    files = build_round_files(
        job_id=job_id,
        dataset_id=dataset_id,
        rollout_id=rollout_id,
        round_idx=round_idx,
        prompt="",
        state={},
        context={},
        student_rollouts=[student_rollout],
        teachers=[],
        sampling_params={},
        want_detail=False,
        score_student_rollouts=True,
        topk_k=topk_k,
        generate_teacher_rollouts=False,
    )
    manifest = json.loads(files.pop("manifest.json").decode("utf-8"))
    for rel in ("prompt.txt", "state.json", "context.json"):
        files.pop(rel, None)

    manifest["job_kind"] = "student_score"
    manifest["prompt_file"] = None
    manifest["state_file"] = None
    manifest["context_file"] = None
    manifest["sampling_params"] = {}
    manifest["eda"] = {"want_detail": False, "run_student_eda": False}
    teacher_request = manifest.get("teacher_request")
    if not isinstance(teacher_request, dict):
        teacher_request = {}
        manifest["teacher_request"] = teacher_request
    teacher_request.update(
        {
            "generate_teacher_rollouts": False,
            "return_teacher_assistant_response": False,
            "return_teacher_token_ids": False,
            "return_teacher_logprobs": False,
            "score_student_rollouts": True,
            "score_teacher_names": names,
            "student_topk_k": int(topk_k or 0),
        }
    )
    manifest["files"] = make_file_manifest(files)
    files["manifest.json"] = _encode_text(_json_dumps(manifest))
    return files


def _load_teacher_topk_sidecar(result_dir: Path, sidecar: dict[str, Any]) -> tuple[list[list[float]], list[list[float]]]:
    rel = _safe_relpath(str(sidecar.get("file") or ""))
    path = result_dir / rel
    if not path.exists():
        raise FileNotFoundError(f"missing OPD teacher top-k sidecar: {path}")
    expected_sha = str(sidecar.get("sha256") or "")
    actual_sha = _sha256_file(path)
    if expected_sha and actual_sha != expected_sha:
        raise ValueError(f"bad OPD sidecar sha256 for {path}: {actual_sha} != {expected_sha}")
    if sidecar.get("format") != "npz_v1":
        raise ValueError(f"unsupported OPD sidecar format at {path}: {sidecar.get('format')!r}")

    import numpy as np  # type: ignore[import-untyped]

    with np.load(path, allow_pickle=False) as data:
        log_probs = data["teacher_topk_log_probs"].astype("float32", copy=False)
        masks = data["teacher_topk_logprob_masks"].astype("float32", copy=False)
        shape = sidecar.get("shape")
        if isinstance(shape, list) and len(shape) == 2:
            expected = (int(shape[0]), int(shape[1]))
            if tuple(log_probs.shape) != expected or tuple(masks.shape) != expected:
                raise ValueError(
                    f"bad OPD sidecar shape at {path}: log_probs={log_probs.shape} masks={masks.shape} expected={expected}"
                )
        return log_probs.tolist(), masks.tolist()


def _hydrate_teacher_topk_sidecars(result: dict[str, Any], result_dir: Path) -> None:
    for entry in result.get("student_rollouts") or []:
        if not isinstance(entry, dict):
            continue
        for score in entry.get("teacher_scores") or []:
            if not isinstance(score, dict):
                continue
            if score.get("teacher_topk_log_probs") and score.get("teacher_topk_logprob_masks"):
                continue
            sidecar = score.get("teacher_topk_sidecar")
            if not isinstance(sidecar, dict):
                continue
            log_probs, masks = _load_teacher_topk_sidecar(result_dir, sidecar)
            score["teacher_topk_log_probs"] = log_probs
            score["teacher_topk_logprob_masks"] = masks


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
    _hydrate_teacher_topk_sidecars(result, result_dir)
    return result
