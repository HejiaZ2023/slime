#!/usr/bin/env python3
"""OPD teacher/EDA relay worker for llm4cov.

Runs either on Paladin next to the existing xfer watcher or on a remote
teacher GPU host with OPD_QUEUE_TRANSPORT=sftp. Slime publishes jobs under
incoming/<namespace>/<job_id>; this worker generates teacher rollouts, evaluates
student and teacher testbenches through the existing EDA xfer watcher, and
publishes results under results/<namespace>/<job_id>.
"""

from __future__ import annotations

import contextlib
import base64
import gzip
import hashlib
import http.server
import ipaddress
import json
import os
import re
import shutil
import sys
import threading
import time
import traceback
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from urllib.parse import parse_qs, urlparse

SCHEMA_VERSION = "opd_exchange_v1"

SLIME_DIR = Path(os.environ.get("OPD_SLIME_DIR", "/home/slu375/docker_scripts/ctr_slime/slime-OPD"))
LLM4COV_SRC = SLIME_DIR / "third_party" / "llm4cov_oss" / "src"
if LLM4COV_SRC.exists():
    sys.path.insert(0, str(LLM4COV_SRC))

from llm4cov.eda_client.xfer_client import submit_cov_job  # noqa: E402

QUEUE_TRANSPORT = os.environ.get("OPD_QUEUE_TRANSPORT", "local").strip().lower()
if QUEUE_TRANSPORT in {"sftp", "http"}:
    XFER = Path(os.environ.get("OPD_LOCAL_XFER_DIR", "/tmp/llm4cov_opd_worker_xfer"))
else:
    XFER = Path(os.environ.get("OPD_XFER_DIR", "/mnt/raid0_ssd/eda/xfer"))
NAMESPACE = os.environ.get("OPD_NAMESPACE", "opd")
IN = XFER / "incoming" / NAMESPACE
RES = XFER / "results" / NAMESPACE
WORK = XFER / ".work" / NAMESPACE
STATE = XFER / ".state" / NAMESPACE
POLL_SEC = float(os.environ.get("OPD_POLL_SEC", "1"))
JOB_TTL = int(os.environ.get("OPD_JOB_TTL", "0"))  # 0 keeps OPD audit artifacts indefinitely
QUEUE_CLAIM_TTL = float(os.environ.get("OPD_QUEUE_CLAIM_TTL", "7200"))
WORKER_ID = re.sub(
    r"[^A-Za-z0-9_.-]",
    "_",
    os.environ.get("OPD_WORKER_ID", f"{os.uname().nodename}-{os.getpid()}"),
)
MAX_CONCURRENT_JOBS = int(os.environ.get("OPD_MAX_CONCURRENT_JOBS", "4"))
MAX_CONCURRENT_ROUND_JOBS = max(
    1,
    int(os.environ.get("OPD_MAX_CONCURRENT_ROUND_JOBS", os.environ.get("OPD_MAX_CONCURRENT_JOBS", "4"))),
)
MAX_CONCURRENT_SCORE_JOBS = max(
    1,
    int(os.environ.get("OPD_MAX_CONCURRENT_SCORE_JOBS", "4")),
)
MAX_JOB_WORKERS = int(os.environ.get("OPD_MAX_JOB_WORKERS", "4"))
TEACHER_TIMEOUT = float(os.environ.get("OPD_TEACHER_TIMEOUT", "900"))
TEACHER_CONTEXT_LENGTH = int(os.environ.get("OPD_TEACHER_CONTEXT_LENGTH", "32768"))
TEACHER_TOKEN_BUDGET_MARGIN = int(os.environ.get("OPD_TEACHER_TOKEN_BUDGET_MARGIN", "256"))
TEACHER_MIN_NEW_TOKENS = int(os.environ.get("OPD_TEACHER_MIN_NEW_TOKENS", "16"))
TEACHER_SCORE_CHUNK_TOKENS = int(os.environ.get("OPD_TEACHER_SCORE_CHUNK_TOKENS", "1024"))
TEACHER_SCORE_CHUNK_WORKERS = int(os.environ.get("OPD_TEACHER_SCORE_CHUNK_WORKERS", "1"))
TEACHER_SCORE_CONTEXT_MARGIN = int(os.environ.get("OPD_TEACHER_SCORE_CONTEXT_MARGIN", "16"))
TEACHER_SCORE_POSITION_TOPK = os.environ.get("OPD_TEACHER_SCORE_POSITION_TOPK", "1") != "0"
TEACHER_ENDPOINT_MAX_INFLIGHT = max(1, int(os.environ.get("OPD_TEACHER_ENDPOINT_MAX_INFLIGHT", "2")))
TEACHER_GENERATE_MAX_INFLIGHT = max(
    1,
    min(
        TEACHER_ENDPOINT_MAX_INFLIGHT,
        int(os.environ.get("OPD_TEACHER_GENERATE_MAX_INFLIGHT", "2")),
    ),
)
TEACHER_SCORE_MAX_INFLIGHT = max(
    1,
    min(
        TEACHER_ENDPOINT_MAX_INFLIGHT,
        int(os.environ.get("OPD_TEACHER_SCORE_MAX_INFLIGHT", "2")),
    ),
)
TEACHER_MODEL_ROOT = os.environ.get("OPD_TEACHER_MODEL_ROOT", "/mnt/raid0_ssd/sheng/final_ckpts")
EDA_SERVER = os.environ.get("OPD_EDA_SERVER", "local")
EDA_REPO_DIR = os.environ.get("OPD_EDA_REPO_DIR", "/workspace/llm4cov_eda")
EDA_TIMEOUT = int(os.environ.get("OPD_EDA_TIMEOUT", "30"))
COMPACT_SCORE_RESULT = os.environ.get("OPD_COMPACT_SCORE_RESULT", "1") != "0"
AUDIT_RESULT_GZIP = os.environ.get("OPD_AUDIT_RESULT_GZIP", "1") != "0"
SCORE_TENSOR_DTYPE = os.environ.get("OPD_SCORE_TENSOR_DTYPE", "float32")
MOCK = os.environ.get("OPD_MOCK", "0") == "1"

_ROUND_JOB_SEM = threading.Semaphore(MAX_CONCURRENT_ROUND_JOBS)
_SCORE_JOB_SEM = threading.Semaphore(MAX_CONCURRENT_SCORE_JOBS)
_LOCK = threading.Lock()
_INFLIGHT: set[str] = set()
QUEUE_BACKEND: Any | None = None
_TEACHER_ENDPOINT_LOCK = threading.Lock()
_TEACHER_ENDPOINT_TOTAL_SLOTS: dict[str, threading.BoundedSemaphore] = {}
_TEACHER_ENDPOINT_KIND_SLOTS: dict[tuple[str, str], threading.BoundedSemaphore] = {}


@contextlib.contextmanager
def teacher_endpoint_slot(name: str, *, kind: str):
    """Bound total and per-kind in-flight requests for one teacher endpoint.

    A long rollout for an unrelated prompt must not consume every slot needed
    by an already-ready forced score. The shared total cap still keeps the
    combined load bounded for the teacher server.
    """
    if kind not in {"generate", "score"}:
        raise ValueError(f"unknown teacher endpoint request kind: {kind}")
    kind_capacity = (
        TEACHER_GENERATE_MAX_INFLIGHT if kind == "generate" else TEACHER_SCORE_MAX_INFLIGHT
    )
    with _TEACHER_ENDPOINT_LOCK:
        total_slot = _TEACHER_ENDPOINT_TOTAL_SLOTS.get(name)
        if total_slot is None:
            total_slot = threading.BoundedSemaphore(TEACHER_ENDPOINT_MAX_INFLIGHT)
            _TEACHER_ENDPOINT_TOTAL_SLOTS[name] = total_slot
        kind_key = (name, kind)
        kind_slot = _TEACHER_ENDPOINT_KIND_SLOTS.get(kind_key)
        if kind_slot is None:
            kind_slot = threading.BoundedSemaphore(kind_capacity)
            _TEACHER_ENDPOINT_KIND_SLOTS[kind_key] = kind_slot
    queued_at = time.monotonic()
    kind_slot.acquire()
    total_acquired = False
    try:
        total_slot.acquire()
        total_acquired = True
        wait_seconds = time.monotonic() - queued_at
        yield wait_seconds
    finally:
        if total_acquired:
            total_slot.release()
        kind_slot.release()

_CODE_RE = re.compile(r"```(?:verilog|systemverilog|sv)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
_MODULE_RE = re.compile(r"(module\s+[\s\S]*?endmodule)", re.IGNORECASE)
_FILENAME_PATTERNS = [
    re.compile(r"filename\s*[:=]\s*([A-Za-z0-9_\-./]+\.s?v)", re.IGNORECASE),
    re.compile(r"file\s*name\s*[:=]\s*([A-Za-z0-9_\-./]+\.s?v)", re.IGNORECASE),
    re.compile(r"output\s*file\s*(?:is|=)\s*[\"']?([A-Za-z0-9_\-./]+\.s?v)", re.IGNORECASE),
    re.compile(r"\b([A-Za-z0-9_\-]+\.s?v)\b", re.IGNORECASE),
]


def log(*items: Any) -> None:
    print(time.strftime("%H:%M:%S"), *items, flush=True)


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _timing(obj: dict[str, Any], key: str, default: float = 0.0) -> float:
    value = obj.get(key) if isinstance(obj, dict) else default
    return _as_float(value, default)


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise ValueError(f"expected JSON object at {path}")
    return obj


def write_json(path: Path, obj: dict[str, Any]) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, sort_keys=True, indent=2), encoding="utf-8")


def _json_compact_bytes(obj: Any) -> bytes:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_json(obj: Any) -> str:
    return _sha256_bytes(_json_compact_bytes(obj))


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


def _summary_count(obj: dict[str, Any], key: str) -> int:
    value = obj.get(key)
    return len(value) if isinstance(value, list) else 0


def summarize_result(obj: dict[str, Any], final_dir: Path) -> dict[str, Any]:
    return {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "job_id": obj.get("job_id"),
        "status": obj.get("status"),
        "job_kind": obj.get("job_kind", "round"),
        "dataset_id": obj.get("dataset_id"),
        "rollout_id": obj.get("rollout_id"),
        "round_idx": obj.get("round_idx"),
        "student_count": _summary_count(obj, "student_rollouts"),
        "teacher_count": _summary_count(obj, "teacher_rollouts"),
        "teacher_model_paths": obj.get("teacher_model_paths") or {},
        "selection": obj.get("selection"),
        "elapsed_s": obj.get("elapsed_s"),
        "incoming_dir": str(IN / str(obj.get("job_id"))),
        "result_dir": str(final_dir),
        "result_file": str(final_dir / "result.json"),
    }


def append_index(summary: dict[str, Any]) -> None:
    index = RES / "index.jsonl"
    line = json.dumps(summary, ensure_ascii=False, sort_keys=True)
    with _LOCK:
        with index.open("a", encoding="utf-8") as f:
            f.write(line + "\n")


class LocalOpdQueue:
    def prepare(self) -> None:
        for d in (IN, RES, WORK, STATE):
            d.mkdir(parents=True, exist_ok=True)

    def poll_jobs(self) -> list[Path]:
        jobs: list[Path] = []
        with contextlib.suppress(Exception):
            os.utime(IN, None)
            os.utime(RES, None)
        for job in sorted(IN.iterdir()) if IN.exists() else []:
            if not job.is_dir() or job.name.endswith(".staging"):
                continue
            if not (job / "manifest.json").exists():
                continue
            if (STATE / f"{job.name}.processed").exists():
                continue
            with _LOCK:
                if job.name in _INFLIGHT:
                    continue
                _INFLIGHT.add(job.name)
            jobs.append(job)
        return jobs

    def publish_result(self, job_id: str, final_dir: Path, summary: dict[str, Any]) -> None:
        return None

    def finish_job(self, job_id: str) -> None:
        return None


class SftpOpdQueue:
    def __init__(self) -> None:
        import paramiko  # type: ignore[import-untyped]

        host = os.environ.get("OPD_QUEUE_SFTP_HOST") or os.environ.get("OPD_SFTP_HOST") or "100.124.95.10"
        port = int(os.environ.get("OPD_QUEUE_SFTP_PORT") or os.environ.get("OPD_SFTP_PORT") or "2222")
        user = os.environ.get("OPD_QUEUE_SFTP_USER") or os.environ.get("OPD_SFTP_USER") or "gpujobs"
        key = os.environ.get("OPD_QUEUE_SFTP_KEY") or os.environ.get("OPD_SFTP_KEY") or os.path.expanduser("~/.ssh/brev_eda_sftp")
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
        self._sftp_lock = threading.Lock()
        self.remote_in = f"incoming/{NAMESPACE}"
        self.remote_res = f"results/{NAMESPACE}"
        # The eda_relay SFTP chroot exposes incoming/ and results/ only; .state is host-local.
        claims_dir = os.environ.get("OPD_QUEUE_SFTP_CLAIMS_DIR")
        self.remote_claims = claims_dir.strip("/") if claims_dir else f"results/{NAMESPACE}/.claims"
        log("OPD_QUEUE_SFTP_CONNECTED", f"host={host}", f"port={port}", f"user={user}", f"seconds={time.monotonic() - started:.3f}")

    def prepare(self) -> None:
        for d in (IN, RES, WORK, STATE):
            d.mkdir(parents=True, exist_ok=True)
        for remote in (self.remote_in, self.remote_res, self.remote_claims):
            self._mkdirs(remote)

    def _mkdirs(self, path: str) -> None:
        cur = ""
        for part in path.strip("/").split("/"):
            if not part:
                continue
            cur = f"{cur}/{part}" if cur else part
            with contextlib.suppress(OSError):
                self.sftp.mkdir(cur)

    def _exists(self, path: str) -> bool:
        try:
            self.sftp.stat(path)
            return True
        except OSError:
            return False

    def _is_dir_attr(self, attr: Any) -> bool:
        return bool(getattr(attr, "st_mode", 0) & 0o040000)

    def _rmtree(self, path: str) -> None:
        try:
            entries = self.sftp.listdir_attr(path)
        except OSError:
            return
        for entry in entries:
            child = f"{path}/{entry.filename}"
            if self._is_dir_attr(entry):
                self._rmtree(child)
            else:
                with contextlib.suppress(OSError):
                    self.sftp.remove(child)
        with contextlib.suppress(OSError):
            self.sftp.rmdir(path)

    def _download_tree(self, remote_dir: str, local_dir: Path) -> tuple[int, int]:
        local_dir.mkdir(parents=True, exist_ok=True)
        file_count = 0
        byte_count = 0
        for entry in self.sftp.listdir_attr(remote_dir):
            remote_child = f"{remote_dir}/{entry.filename}"
            local_child = local_dir / entry.filename
            if self._is_dir_attr(entry):
                child_files, child_bytes = self._download_tree(remote_child, local_child)
                file_count += child_files
                byte_count += child_bytes
            else:
                local_child.parent.mkdir(parents=True, exist_ok=True)
                self.sftp.get(remote_child, str(local_child))
                file_count += 1
                byte_count += int(getattr(entry, "st_size", 0) or 0)
        return file_count, byte_count

    def _upload_tree(self, local_dir: Path, remote_dir: str) -> tuple[int, int]:
        file_count = 0
        byte_count = 0
        self._mkdirs(remote_dir)
        for path in sorted(local_dir.rglob("*")):
            if path.is_dir():
                continue
            rel = _safe_relpath(path.relative_to(local_dir).as_posix())
            remote_path = f"{remote_dir}/{rel}"
            self._mkdirs(str(Path(remote_path).parent).replace("\\", "/"))
            self.sftp.put(str(path), remote_path)
            file_count += 1
            byte_count += path.stat().st_size
        return file_count, byte_count

    def _claim_job(self, job_id: str) -> bool:
        if self._exists(f"{self.remote_res}/{job_id}/.done"):
            return False
        claim_dir = f"{self.remote_claims}/{job_id}"
        try:
            self.sftp.mkdir(claim_dir)
        except OSError:
            try:
                age = time.time() - float(self.sftp.stat(claim_dir).st_mtime)
            except OSError:
                age = 0.0
            if QUEUE_CLAIM_TTL > 0 and age > QUEUE_CLAIM_TTL:
                log("OPD_QUEUE_STALE_CLAIM", job_id, f"age={age:.1f}")
                self._rmtree(claim_dir)
                try:
                    self.sftp.mkdir(claim_dir)
                except OSError:
                    return False
            else:
                return False
        meta = {
            "worker_id": WORKER_ID,
            "claimed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "claimed_at_unix": time.time(),
        }
        with self.sftp.open(f"{claim_dir}/worker.json", "w") as f:
            f.write(json.dumps(meta, ensure_ascii=False, sort_keys=True))
        return True

    def _release_claim(self, job_id: str) -> None:
        self._rmtree(f"{self.remote_claims}/{job_id}")

    def poll_jobs(self) -> list[Path]:
        with self._sftp_lock:
            jobs: list[Path] = []
            try:
                entries = self.sftp.listdir_attr(self.remote_in)
            except OSError:
                return jobs
            for entry in sorted(entries, key=lambda item: item.filename):
                job_id = entry.filename
                if not self._is_dir_attr(entry) or job_id.endswith(".staging"):
                    continue
                if self._exists(f"{self.remote_res}/{job_id}/.done"):
                    continue
                if not self._exists(f"{self.remote_in}/{job_id}/manifest.json"):
                    continue
                with _LOCK:
                    if job_id in _INFLIGHT:
                        continue
                    _INFLIGHT.add(job_id)
                if not self._claim_job(job_id):
                    with _LOCK:
                        _INFLIGHT.discard(job_id)
                    continue
                local_job = IN / job_id
                try:
                    if local_job.exists():
                        shutil.rmtree(local_job)
                    started = time.monotonic()
                    files, bytes_ = self._download_tree(f"{self.remote_in}/{job_id}", local_job)
                    log("OPD_QUEUE_DOWNLOAD", job_id, f"files={files}", f"bytes={bytes_}", f"seconds={time.monotonic() - started:.3f}")
                    jobs.append(local_job)
                except Exception as exc:
                    log("OPD_QUEUE_DOWNLOAD_FAIL", job_id, exc)
                    self._release_claim(job_id)
                    with _LOCK:
                        _INFLIGHT.discard(job_id)
            return jobs

    def publish_result(self, job_id: str, final_dir: Path, summary: dict[str, Any]) -> None:
        with self._sftp_lock:
            final = f"{self.remote_res}/{job_id}"
            started = time.monotonic()
            self._rmtree(final)
            self._mkdirs(final)
            files = 0
            bytes_ = 0
            done_file = None
            for path in sorted(final_dir.rglob("*")):
                if path.is_dir():
                    continue
                rel = _safe_relpath(path.relative_to(final_dir).as_posix())
                if rel == ".done":
                    done_file = path
                    continue
                remote_path = f"{final}/{rel}"
                self._mkdirs(str(Path(remote_path).parent).replace("\\", "/"))
                self.sftp.put(str(path), remote_path)
                files += 1
                bytes_ += path.stat().st_size
            with contextlib.suppress(Exception):
                with self.sftp.open(f"{self.remote_res}/index.jsonl", "a") as f:
                    f.write(json.dumps(summary, ensure_ascii=False, sort_keys=True) + "\n")
            if done_file is not None:
                self.sftp.put(str(done_file), f"{final}/.done")
                files += 1
                bytes_ += done_file.stat().st_size
            log("OPD_QUEUE_UPLOAD", job_id, f"files={files}", f"bytes={bytes_}", f"seconds={time.monotonic() - started:.3f}")

    def finish_job(self, job_id: str) -> None:
        with self._sftp_lock:
            self._release_claim(job_id)


def _write_job_files(job_id: str, files: dict[str, bytes]) -> tuple[Path, int, int]:
    IN.mkdir(parents=True, exist_ok=True)
    staging = IN / f"{job_id}.staging"
    final = IN / job_id
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    file_count = 0
    byte_count = 0
    for rel, data in files.items():
        rel = _safe_relpath(rel)
        path = staging / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        file_count += 1
        byte_count += len(data)
    if final.exists():
        shutil.rmtree(final)
    os.rename(staging, final)
    return final, file_count, byte_count


def _collect_result_files(result_dir: Path, *, include_audit: bool = False) -> tuple[dict[str, str], int, int]:
    files: dict[str, str] = {}
    file_count = 0
    byte_count = 0
    for path in sorted(result_dir.rglob("*")):
        if path.is_dir():
            continue
        rel = _safe_relpath(path.relative_to(result_dir).as_posix())
        if rel.startswith("audit/") and not include_audit:
            continue
        data = path.read_bytes()
        files[rel] = base64.b64encode(data).decode("ascii")
        file_count += 1
        byte_count += len(data)
    return files, file_count, byte_count


class HttpOpdQueue(LocalOpdQueue):
    def __init__(self) -> None:
        self.bind = os.environ.get("OPD_HTTP_BIND", "127.0.0.1")
        self.port = int(os.environ.get("OPD_HTTP_PORT", "19090"))
        self.token = os.environ.get("OPD_HTTP_TOKEN", "")
        self.max_body_bytes = int(os.environ.get("OPD_HTTP_MAX_BODY_BYTES", str(256 * 1024 * 1024)))
        allow_any = os.environ.get("OPD_HTTP_ALLOW_NON_TAILSCALE", "0") == "1"
        bind_ip = ipaddress.ip_address(self.bind)
        tailscale_net = ipaddress.ip_network("100.64.0.0/10")
        if not allow_any and not (bind_ip.is_loopback or bind_ip in tailscale_net):
            raise ValueError(
                f"refusing OPD_HTTP_BIND={self.bind!r}; bind loopback or a Tailscale 100.64.0.0/10 address, "
                "or set OPD_HTTP_ALLOW_NON_TAILSCALE=1 explicitly"
            )
        self.httpd: http.server.ThreadingHTTPServer | None = None

    def prepare(self) -> None:
        super().prepare()
        queue = self

        class Handler(http.server.BaseHTTPRequestHandler):
            server_version = "llm4cov-opd-http/1.0"

            def log_message(self, fmt: str, *args: Any) -> None:
                log("OPD_HTTP_ACCESS", self.address_string(), fmt % args)

            def _json_response(self, status: int, obj: dict[str, Any]) -> None:
                body = json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
                if "gzip" in self.headers.get("Accept-Encoding", "").lower():
                    body = gzip.compress(body)
                    encoding = "gzip"
                else:
                    encoding = ""
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                if encoding:
                    self.send_header("Content-Encoding", encoding)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _authorized(self) -> bool:
                if not queue.token:
                    return True
                return self.headers.get("X-OPD-Token", "") == queue.token

            def _path_parts(self) -> tuple[str, str, str] | None:
                parsed = urlparse(self.path)
                parts = [part for part in parsed.path.strip("/").split("/") if part]
                if len(parts) != 3:
                    return None
                kind, namespace, job_id = parts
                if namespace != NAMESPACE:
                    return None
                if not re.fullmatch(r"[A-Za-z0-9_.-]+", job_id):
                    return None
                return kind, namespace, job_id

            def do_GET(self) -> None:
                if self.path == "/healthz":
                    self._json_response(200, {"ok": True, "namespace": NAMESPACE, "worker_id": WORKER_ID})
                    return
                if not self._authorized():
                    self._json_response(403, {"ok": False, "error": "forbidden"})
                    return
                parts = self._path_parts()
                if not parts or parts[0] != "results":
                    self._json_response(404, {"ok": False, "error": "not_found"})
                    return
                parsed = urlparse(self.path)
                include_audit = parse_qs(parsed.query).get("audit", ["0"])[0] == "1"
                job_id = parts[2]
                result_dir = RES / job_id
                if not (result_dir / ".done").exists():
                    self._json_response(202, {"ok": False, "status": "pending", "job_id": job_id})
                    return
                started = time.monotonic()
                files, file_count, byte_count = _collect_result_files(result_dir, include_audit=include_audit)
                self._json_response(
                    200,
                    {
                        "ok": True,
                        "job_id": job_id,
                        "files": files,
                        "file_count": file_count,
                        "payload_bytes": byte_count,
                        "seconds": time.monotonic() - started,
                    },
                )

            def do_POST(self) -> None:
                if not self._authorized():
                    self._json_response(403, {"ok": False, "error": "forbidden"})
                    return
                parts = self._path_parts()
                if not parts or parts[0] != "jobs":
                    self._json_response(404, {"ok": False, "error": "not_found"})
                    return
                job_id = parts[2]
                started = time.monotonic()
                try:
                    length = int(self.headers.get("Content-Length", "0") or "0")
                    if length <= 0 or length > queue.max_body_bytes:
                        self._json_response(413, {"ok": False, "job_id": job_id, "error": "request_too_large"})
                        return
                    raw_body = self.rfile.read(length)
                    wire_bytes = len(raw_body)
                    if "gzip" in self.headers.get("Content-Encoding", "").lower():
                        raw_body = gzip.decompress(raw_body)
                    if len(raw_body) > queue.max_body_bytes:
                        self._json_response(413, {"ok": False, "job_id": job_id, "error": "request_too_large"})
                        return
                    payload = json.loads(raw_body.decode("utf-8"))
                    raw_files = payload.get("files") if isinstance(payload, dict) else None
                    if not isinstance(raw_files, dict):
                        raise ValueError("missing files object")
                    files = {
                        _safe_relpath(str(rel)): base64.b64decode(str(data).encode("ascii"))
                        for rel, data in raw_files.items()
                    }
                    _path, file_count, byte_count = _write_job_files(job_id, files)
                    log(
                        "OPD_HTTP_JOB_RECEIVED",
                        job_id,
                        f"files={file_count}",
                        f"bytes={byte_count}",
                        f"wire_bytes={wire_bytes}",
                        f"json_bytes={len(raw_body)}",
                        f"seconds={time.monotonic() - started:.3f}",
                    )
                    self._json_response(
                        200,
                        {
                            "ok": True,
                            "job_id": job_id,
                            "file_count": file_count,
                            "payload_bytes": byte_count,
                            "wire_bytes": wire_bytes,
                            "json_bytes": len(raw_body),
                            "seconds": time.monotonic() - started,
                        },
                    )
                except Exception as exc:
                    log("OPD_HTTP_JOB_FAIL", job_id, exc)
                    self._json_response(400, {"ok": False, "job_id": job_id, "error": str(exc)})

        self.httpd = http.server.ThreadingHTTPServer((self.bind, self.port), Handler)
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        log("OPD_QUEUE_HTTP_LISTEN", f"bind={self.bind}", f"port={self.port}", f"namespace={NAMESPACE}")


def build_queue_backend() -> Any:
    if QUEUE_TRANSPORT == "local":
        return LocalOpdQueue()
    if QUEUE_TRANSPORT == "sftp":
        return SftpOpdQueue()
    if QUEUE_TRANSPORT == "http":
        return HttpOpdQueue()
    raise ValueError(f"unknown OPD_QUEUE_TRANSPORT={QUEUE_TRANSPORT!r}")


def safe_filename(name: str | None, default: str = "tb_generated.sv") -> str:
    base = os.path.basename(name or default)
    base = re.sub(r"[^A-Za-z0-9._-]", "_", base)
    if not base or base in {".", ".."}:
        base = default
    if not re.search(r"\.s?v$", base, re.IGNORECASE):
        base += ".sv"
    return base


def extract_filename(text: str) -> str | None:
    for pattern in _FILENAME_PATTERNS:
        match = pattern.search(text or "")
        if match:
            return safe_filename(match.group(1))
    return None


def extract_verilog(text: str) -> str | None:
    match = _CODE_RE.search(text or "")
    if match:
        return match.group(1).strip()
    match = _MODULE_RE.search(text or "")
    if match:
        return match.group(1).strip()
    return None


def context_from_dict(data: dict[str, Any]) -> SimpleNamespace:
    rtl_files = [SimpleNamespace(name=f["name"], content=f["content"]) for f in data.get("rtl_files", [])]
    spec_files = [SimpleNamespace(name=f["name"], content=f["content"]) for f in data.get("spec_files", [])]
    return SimpleNamespace(
        id=data.get("id", "unknown"),
        rtl_files=rtl_files,
        spec_files=spec_files,
        dataset_name=data.get("dataset_name", ""),
        rtl_tokens=data.get("rtl_tokens", 0),
        potential_top=data.get("potential_top", []),
        misc=data.get("misc", {}) or {},
        dut_top_module_name=data.get("dut_top_module_name") or (data.get("potential_top") or ["dut"])[0],
        dut_top_instance_name=data.get("dut_top_instance_name", "dut"),
        instructions=data.get("instructions", ""),
    )


def evaluate_cov(context: SimpleNamespace, raw: dict[str, Any]) -> dict[str, Any]:
    status = raw.get("status", "xrun_failed")
    out: dict[str, Any] = {
        "status": status,
        "is_pass_xrun": status != "xrun_failed",
        "has_coverage": status == "success",
        "overall_coverage": 0.0,
        "is_pass_targets": False,
        "err_msg": raw.get("err_msg") or "",
    }
    if status != "success":
        return out
    cov_info = raw.get("cov_info") if isinstance(raw.get("cov_info"), dict) else {}
    summary = cov_info.get("summary") if isinstance(cov_info, dict) else []
    dut_entry = None
    if isinstance(summary, list):
        dut_entry = next(
            (
                m for m in summary
                if isinstance(m, dict)
                and m.get("name") == context.dut_top_module_name
                and m.get("level") == 0
            ),
            None,
        )
        if dut_entry is None:
            dut_entry = next((m for m in summary if isinstance(m, dict) and m.get("level") == 0), None)
    if isinstance(dut_entry, dict) and isinstance(dut_entry.get("Overall Average"), (int, float)):
        out["overall_coverage"] = float(dut_entry["Overall Average"])
    targets = context.misc.get("targets", []) if isinstance(context.misc, dict) else []
    pass_targets = True
    for target in targets:
        if not isinstance(target, dict):
            continue
        inst_name = target.get("inst_name")
        metric = target.get("metric")
        target_pct = target.get("target_percentage")
        if not metric or target_pct is None:
            continue
        entry = dut_entry
        if inst_name and inst_name not in {context.dut_top_instance_name, context.dut_top_module_name}:
            entry = next((m for m in summary if isinstance(m, dict) and m.get("name") == inst_name), None)
        if not isinstance(entry, dict) or metric not in entry or not isinstance(entry.get(metric), (int, float)):
            continue
        if float(entry[metric]) * 100 < float(target_pct):
            pass_targets = False
            break
    out["is_pass_targets"] = pass_targets
    return out


def format_feedback(raw: dict[str, Any], context: SimpleNamespace) -> str:
    status = str(raw.get("status", "unknown"))
    err_msg = str(raw.get("err_msg") or "")
    if status == "xrun_failed":
        return f"- status: {status}\n- stage: xrun\n- log: {err_msg}"
    if status != "success":
        return f"- status: {status}\n- stage: imc\n- log: {err_msg}"
    cov_info = raw.get("cov_info") if isinstance(raw.get("cov_info"), dict) else {}
    detail = str(cov_info.get("detail", "")) if isinstance(cov_info, dict) else ""
    return f"- status: {status}\n- stage: success\n- coverage: {detail}"


def _eda_summary(eda_log: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": eda_log.get("status"),
        "filename": eda_log.get("filename"),
        "overall_coverage": float(eda_log.get("overall_coverage", 0.0) or 0.0),
        "is_pass_targets": bool(eda_log.get("is_pass_targets", False)),
        "is_pass_xrun": bool(eda_log.get("is_pass_xrun", False)),
        "has_coverage": bool(eda_log.get("has_coverage", False)),
        "err_msg": eda_log.get("err_msg", ""),
    }


def _with_eda_summary(result: dict[str, Any]) -> dict[str, Any]:
    eda_log = result.get("eda_log") if isinstance(result.get("eda_log"), dict) else {}
    result["eda_summary"] = _eda_summary(eda_log)
    return result


def run_eda(context: SimpleNamespace, filename: str | None, body: str | None, want_detail: bool) -> dict[str, Any]:
    started = time.monotonic()
    filename = safe_filename(filename)
    if not body:
        eda_log = {"status": "parse_failed", "filename": None, "has_coverage": False}
        return _with_eda_summary({
            "reward": 0.0,
            "eda_feedback": "- status: failed (could not extract a valid testbench)",
            "eda_log": eda_log,
            "eda_timing": {"total_seconds": time.monotonic() - started, "submit_cov_job_seconds": 0.0},
        })
    if MOCK:
        eda_log = {
            "status": "success",
            "filename": filename,
            "overall_coverage": 0.5,
            "is_pass_xrun": True,
            "is_pass_targets": False,
            "has_coverage": True,
            "err_msg": "",
        }
        return _with_eda_summary({
            "reward": 1.5,
            "eda_feedback": "- status: success\n- stage: mock\n- coverage: 50%",
            "eda_log": eda_log,
            "eda_timing": {"total_seconds": time.monotonic() - started, "submit_cov_job_seconds": 0.0},
        })
    tb_file = SimpleNamespace(name=filename, content=body)
    try:
        submit_started = time.monotonic()
        raw = submit_cov_job(
            EDA_SERVER,
            EDA_REPO_DIR,
            context,
            tb_file,
            skip_detail=not want_detail,
            timeout=EDA_TIMEOUT,
        )
        submit_seconds = time.monotonic() - submit_started
        eval_log = evaluate_cov(context, raw)
        eda_log = {
            "status": eval_log["status"],
            "filename": filename,
            "overall_coverage": float(eval_log.get("overall_coverage", 0.0)),
            "is_pass_xrun": bool(eval_log.get("is_pass_xrun", False)),
            "is_pass_targets": bool(eval_log.get("is_pass_targets", False)),
            "has_coverage": bool(eval_log.get("has_coverage", False)),
            "err_msg": eval_log.get("err_msg", ""),
        }
        reward = 1.0 + eda_log["overall_coverage"] if eda_log["has_coverage"] else 0.0
        return _with_eda_summary({
            "reward": reward,
            "eda_feedback": format_feedback(raw, context) if want_detail else None,
            "eda_log": eda_log,
            "eda_timing": {
                "total_seconds": time.monotonic() - started,
                "submit_cov_job_seconds": submit_seconds,
            },
        })
    except Exception as exc:
        eda_log = {"status": "exception", "filename": filename, "has_coverage": False, "exc": str(exc)}
        return _with_eda_summary({
            "reward": 0.0,
            "eda_feedback": f"- status: failed (EDA exception: {exc})",
            "eda_log": eda_log,
            "eda_timing": {"total_seconds": time.monotonic() - started},
        })


def load_teacher_config() -> dict[str, Any]:
    raw = os.environ.get("OPD_TEACHERS_JSON", "").strip()
    if not raw:
        return {}
    obj = json.loads(raw)
    if not isinstance(obj, dict):
        raise ValueError("OPD_TEACHERS_JSON must be a JSON object")
    return obj


def teacher_url(name: str, cfg: dict[str, Any]) -> str:
    value = cfg.get(name)
    if isinstance(value, str):
        return value
    if isinstance(value, dict) and value.get("url"):
        return str(value["url"])
    env_name = "OPD_TEACHER_" + re.sub(r"[^A-Za-z0-9]", "_", name).upper() + "_URL"
    value = os.environ.get(env_name)
    if value:
        return value
    raise KeyError(f"missing teacher URL for {name!r}; set OPD_TEACHERS_JSON or {env_name}")


def teacher_model_path(name: str, cfg: dict[str, Any]) -> str | None:
    value = cfg.get(name)
    if isinstance(value, dict):
        for key in ("model_path", "checkpoint", "ckpt", "path"):
            if value.get(key):
                return str(value[key])
    env_name = "OPD_TEACHER_" + re.sub(r"[^A-Za-z0-9]", "_", name).upper() + "_MODEL_PATH"
    value = os.environ.get(env_name)
    if value:
        return value
    if name in {"stage0", "stage1", "stage2"} and TEACHER_MODEL_ROOT:
        return str(Path(TEACHER_MODEL_ROOT) / f"{name}_step999")
    return None


def teacher_model_paths(names: list[str], cfg: dict[str, Any]) -> dict[str, str]:
    out: dict[str, str] = {}
    for name in sorted({n for n in names if n}):
        path = teacher_model_path(name, cfg)
        if path:
            out[name] = path
    return out


def post_json(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            obj = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")[-2000:]
        raise RuntimeError(f"teacher HTTP {exc.code}: {body}") from exc
    if not isinstance(obj, dict):
        raise RuntimeError(f"teacher returned non-object JSON: {type(obj)}")
    return obj


def teacher_tokenize_url(generate_url: str) -> str:
    return re.sub(r"/generate/?$", "/tokenize", generate_url.rstrip("/"))


def count_teacher_prompt_tokens(url: str, prompt: str) -> int:
    obj = post_json(teacher_tokenize_url(url), {"prompt": prompt}, timeout=TEACHER_TIMEOUT)
    count = obj.get("count")
    if count is not None:
        return int(count)
    tokens = obj.get("tokens")
    if isinstance(tokens, list):
        return len(tokens)
    raise RuntimeError(f"teacher tokenize returned no token count: {obj.keys()}")


def _requested_max_new_tokens(sampling_params: dict[str, Any]) -> int:
    for key in ("max_new_tokens", "max_tokens"):
        if sampling_params.get(key) is not None:
            try:
                return int(sampling_params[key])
            except (TypeError, ValueError):
                return 0
    return 0


def teacher_sampling_with_budget(
    name: str,
    url: str,
    prompt: str,
    sampling_params: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], str | None]:
    started = time.monotonic()
    params = dict(sampling_params or {})
    requested = _requested_max_new_tokens(params)
    budget = {
        "requested_max_new_tokens": requested,
        "effective_max_new_tokens": requested,
        "teacher_context_length": TEACHER_CONTEXT_LENGTH,
        "teacher_token_budget_margin": TEACHER_TOKEN_BUDGET_MARGIN,
    }
    if requested <= 0 or TEACHER_CONTEXT_LENGTH <= 0:
        budget["budget_seconds"] = time.monotonic() - started
        return params, budget, None
    try:
        tokenize_started = time.monotonic()
        prompt_tokens = count_teacher_prompt_tokens(url, prompt)
        budget["tokenize_seconds"] = time.monotonic() - tokenize_started
    except Exception as exc:
        budget["token_count_error"] = str(exc)
        budget["budget_seconds"] = time.monotonic() - started
        log("WARN", "teacher_token_count_failed", name, str(exc))
        return params, budget, None
    budget["prompt_token_count"] = prompt_tokens
    available = TEACHER_CONTEXT_LENGTH - prompt_tokens - max(0, TEACHER_TOKEN_BUDGET_MARGIN)
    budget["available_new_tokens"] = available
    if available < max(1, TEACHER_MIN_NEW_TOKENS):
        budget["effective_max_new_tokens"] = 0
        budget["budget_seconds"] = time.monotonic() - started
        return params, budget, "teacher_context_exceeded"
    effective = min(requested, available)
    params["max_new_tokens"] = effective
    if "max_tokens" in params:
        params["max_tokens"] = effective
    budget["effective_max_new_tokens"] = effective
    if effective < requested:
        log(
            "teacher_budget_clamp",
            name,
            f"prompt_tokens={prompt_tokens}",
            f"requested={requested}",
            f"effective={effective}",
            f"context={TEACHER_CONTEXT_LENGTH}",
        )
    budget["budget_seconds"] = time.monotonic() - started
    return params, budget, None


def extract_token_logprobs(meta: dict[str, Any]) -> tuple[list[int], list[float]]:
    pairs = meta.get("output_token_logprobs") or []
    token_ids: list[int] = []
    logprobs: list[float] = []
    if not isinstance(pairs, list):
        return token_ids, logprobs
    for item in pairs:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            try:
                logprobs.append(float(item[0]))
                token_ids.append(int(item[1]))
            except (TypeError, ValueError):
                continue
    return token_ids, logprobs


def generate_teacher_rollout(name: str, slot: int, prompt: str, sampling_params: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
    started = time.monotonic()
    url = teacher_url(name, cfg)
    model_path = teacher_model_path(name, cfg)
    effective_sampling_params, budget, skip_reason = teacher_sampling_with_budget(name, url, prompt, sampling_params)
    base = {
        "id": f"{name}_t{slot:03d}",
        "teacher": name,
        "teacher_model_path": model_path,
        "teacher_generation_budget": budget,
    }
    if skip_reason:
        return {
            **base,
            "assistant_response": "",
            "finish_reason": skip_reason,
            "teacher_generation_status": skip_reason,
            "teacher_generation_timing": {
                "total_seconds": time.monotonic() - started,
                "request_seconds": 0.0,
                "endpoint_wait_seconds": 0.0,
            },
        }
    # Teacher rollout output logprobs are not used for OPD.  Forced scoring on
    # the student's exact top-k support below is the sole teacher distribution
    # consumed by the loss, so avoid asking SGLang to serialize rollout LPs.
    payload = {"text": prompt, "sampling_params": effective_sampling_params}
    endpoint_wait_seconds = 0.0
    request_seconds = 0.0
    try:
        with teacher_endpoint_slot(name, kind="generate") as endpoint_wait_seconds:
            request_started = time.monotonic()
            output = post_json(url, payload, timeout=TEACHER_TIMEOUT)
            request_seconds = time.monotonic() - request_started
    except Exception as exc:
        return {
            **base,
            "assistant_response": "",
            "finish_reason": "teacher_generate_error",
            "teacher_generation_status": "teacher_generate_error",
            "teacher_generation_error": str(exc),
            "teacher_generation_timing": {
                "total_seconds": time.monotonic() - started,
                "request_seconds": request_seconds,
                "endpoint_wait_seconds": endpoint_wait_seconds,
            },
        }
    response = str(output.get("text") or output.get("response") or output.get("output") or "")
    meta = output.get("meta_info") if isinstance(output.get("meta_info"), dict) else {}
    try:
        generated_token_count = int(
            meta.get("completion_tokens") or meta.get("generated_token_count") or 0
        )
    except (TypeError, ValueError):
        generated_token_count = 0
    return {
        **base,
        "assistant_response": response,
        "finish_reason": meta.get("finish_reason"),
        "teacher_generation_status": "success",
        "teacher_generation_timing": {
            "total_seconds": time.monotonic() - started,
            "request_seconds": request_seconds,
            "endpoint_wait_seconds": endpoint_wait_seconds,
            "response_chars": len(response),
            "generated_token_count": generated_token_count,
        },
    }


def extract_input_logprobs(meta: dict[str, Any], response_len: int) -> list[float]:
    pairs = meta.get("input_token_logprobs") or []
    logprobs: list[float] = []
    if not isinstance(pairs, list) or response_len <= 0:
        return logprobs
    for item in pairs:
        if isinstance(item, (list, tuple)) and item:
            try:
                logprobs.append(float(item[0]))
            except (TypeError, ValueError):
                continue
    return logprobs[-response_len:]


def _coerce_int_matrix(value: Any) -> list[list[int]] | None:
    if not isinstance(value, list):
        return None
    out: list[list[int]] = []
    for row in value:
        if not isinstance(row, list):
            return None
        try:
            out.append([int(x) for x in row])
        except (TypeError, ValueError):
            return None
    return out


def _coerce_float_matrix(value: Any) -> list[list[float]] | None:
    if not isinstance(value, list):
        return None
    out: list[list[float]] = []
    for row in value:
        if not isinstance(row, list):
            return None
        try:
            out.append([float(x) for x in row])
        except (TypeError, ValueError):
            return None
    return out


def _matrix_shape(value: list[list[Any]] | None, rows: int, cols: int) -> bool:
    return value is not None and len(value) == rows and all(len(row) == cols for row in value)


def _candidate_logprob_row(row: Any) -> dict[int, float]:
    out: dict[int, float] = {}
    if isinstance(row, dict):
        for key, value in row.items():
            try:
                out[int(key)] = float(value)
            except (TypeError, ValueError):
                continue
        return out
    if not isinstance(row, list):
        return out
    for item in row:
        token_id = None
        log_prob = None
        if isinstance(item, dict):
            raw_token = item.get("token_id", item.get("id", item.get("token")))
            raw_logprob = item.get("logprob", item.get("log_prob", item.get("log_probs")))
            try:
                token_id = int(raw_token)
                log_prob = float(raw_logprob)
            except (TypeError, ValueError):
                continue
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            first, second = item[0], item[1]
            try:
                first_f = float(first)
                second_i = int(second)
                if first_f <= 0.0:
                    log_prob = first_f
                    token_id = second_i
            except (TypeError, ValueError):
                pass
            if token_id is None or log_prob is None:
                try:
                    token_id = int(first)
                    log_prob = float(second)
                except (TypeError, ValueError):
                    continue
        if token_id is not None and log_prob is not None:
            out[token_id] = log_prob
    return out


def _extract_requested_topk_logprobs(
    output: dict[str, Any],
    response_len: int,
    requested_topk: list[list[int]] | None,
) -> tuple[list[list[float]] | None, list[list[float]] | None]:
    if not requested_topk or response_len <= 0:
        return None, None
    width = len(requested_topk[0]) if requested_topk else 0
    if width <= 0 or not _matrix_shape(requested_topk, response_len, width):
        return None, None

    meta = output.get("meta_info") if isinstance(output.get("meta_info"), dict) else {}
    direct = (
        output.get("teacher_topk_log_probs")
        or output.get("topk_log_probs")
        or meta.get("teacher_topk_log_probs")
        or meta.get("topk_log_probs")
        or meta.get("input_requested_token_logprobs")
    )
    direct_matrix = _coerce_float_matrix(direct)
    if _matrix_shape(direct_matrix, response_len, width):
        masks = (
            output.get("teacher_topk_logprob_masks")
            or output.get("topk_logprob_masks")
            or meta.get("teacher_topk_logprob_masks")
            or meta.get("topk_logprob_masks")
        )
        mask_matrix = _coerce_float_matrix(masks)
        if not _matrix_shape(mask_matrix, response_len, width):
            mask_matrix = [[1.0] * width for _ in range(response_len)]
        return direct_matrix, mask_matrix

    idx_rows = (
        meta.get("input_token_ids_logprobs_idx")
        or meta.get("input_token_ids_logprobs_token_ids")
        or meta.get("input_top_logprobs_idx")
        or meta.get("input_top_logprobs_token_ids")
    )
    val_rows = (
        meta.get("input_token_ids_logprobs_val")
        or meta.get("input_token_ids_logprobs_logprobs")
        or meta.get("input_top_logprobs_val")
        or meta.get("input_top_logprobs_logprobs")
    )
    candidate_maps: list[dict[int, float]] = []
    if isinstance(idx_rows, list) and isinstance(val_rows, list):
        for ids, vals in zip(idx_rows[-response_len:], val_rows[-response_len:], strict=False):
            row: dict[int, float] = {}
            if isinstance(ids, list) and isinstance(vals, list):
                for token_id, log_prob in zip(ids, vals, strict=False):
                    try:
                        row[int(token_id)] = float(log_prob)
                    except (TypeError, ValueError):
                        continue
            candidate_maps.append(row)
    else:
        rows = (
            meta.get("input_token_ids_logprobs")
            or meta.get("input_top_logprobs")
            or meta.get("input_token_top_logprobs")
            or meta.get("top_logprobs")
            or []
        )
        if isinstance(rows, list):
            candidate_maps = [_candidate_logprob_row(row) for row in rows[-response_len:]]

    if len(candidate_maps) != response_len:
        return None, None
    log_probs: list[list[float]] = []
    masks: list[list[float]] = []
    for want_row, have_row in zip(requested_topk, candidate_maps, strict=False):
        lp_row: list[float] = []
        mask_row: list[float] = []
        for token_id in want_row:
            if token_id in have_row:
                lp_row.append(float(have_row[token_id]))
                mask_row.append(1.0)
            else:
                lp_row.append(0.0)
                mask_row.append(0.0)
        log_probs.append(lp_row)
        masks.append(mask_row)
    return log_probs, masks


def score_teacher_on_student(
    name: str, entry: dict[str, Any], cfg: dict[str, Any], topk_k: int = 0
) -> dict[str, Any]:
    sid = str(entry.get("id"))
    input_ids = entry.get("input_token_ids")
    response_len = int(entry.get("response_token_count") or 0)
    requested_topk = _coerce_int_matrix(entry.get("topk_token_ids")) if topk_k > 1 else None
    if requested_topk is not None:
        requested_topk = [row[:topk_k] for row in requested_topk]
    requested_topk_sha256 = _sha256_json(requested_topk) if requested_topk is not None else ""
    if not isinstance(input_ids, list) or response_len <= 0:
        return {
            "student_id": sid,
            "teacher": name,
            "status": "missing_student_tokens",
            "teacher_log_probs": [],
            "topk_token_ids_sha256": requested_topk_sha256,
        }
    prompt_len = max(0, len(input_ids) - response_len)
    input_ids_int = [int(x) for x in input_ids]
    teacher_log_probs: list[float] = []
    teacher_topk_parts: list[list[list[float]] | None] = []
    teacher_mask_parts: list[list[list[float]] | None] = []
    timings: list[dict[str, Any]] = []
    total_started = time.monotonic()
    chunk_size = response_len
    if requested_topk and TEACHER_SCORE_CHUNK_TOKENS > 0:
        chunk_size = min(response_len, max(1, TEACHER_SCORE_CHUNK_TOKENS))
    score_context_len = max(0, TEACHER_CONTEXT_LENGTH - max(0, TEACHER_SCORE_CONTEXT_MARGIN))

    offsets = list(range(0, response_len, chunk_size))
    chunk_workers = max(1, min(max(1, TEACHER_SCORE_CHUNK_WORKERS), len(offsets) or 1))
    chunk_wall_started = time.monotonic()

    def _score_chunk(offset: int) -> tuple[int, list[float], list[list[float]] | None, list[list[float]] | None, dict[str, Any]]:
        chunk_len = min(chunk_size, response_len - offset)
        chunk_start = prompt_len + offset
        chunk_end = chunk_start + chunk_len
        window_start = 0
        if score_context_len > 0:
            window_start = max(0, chunk_end - score_context_len)
        chunk_start_in_window = chunk_start - window_start
        logprob_start_len = max(0, chunk_start_in_window - 1)
        chunk_requested_topk = (
            requested_topk[offset : offset + chunk_len] if requested_topk is not None else None
        )
        chunk_input_ids = input_ids_int[window_start:chunk_end]
        payload = {
            "input_ids": chunk_input_ids,
            "sampling_params": {
                "temperature": 0,
                "max_new_tokens": 0,
                "skip_special_tokens": False,
            },
            "return_logprob": True,
            # Only response-token logprobs are consumed by OPD.  Keep a
            # teacher-context-sized sliding window for long responses; SGLang
            # returns a leading None row at logprob_start_len, so start one
            # token before this chunk within the local window and keep the
            # final chunk_len rows.
            "logprob_start_len": logprob_start_len,
        }
        token_ids_logprob: list[int] = []
        if chunk_requested_topk:
            payload["topk_token_ids"] = chunk_requested_topk
            payload["return_topk_logprobs"] = True
            if not TEACHER_SCORE_POSITION_TOPK:
                token_ids_logprob = sorted({int(token_id) for row in chunk_requested_topk for token_id in row})
                # Fallback for old SGLang builds that only accept a
                # request-level token_ids_logprob union.
                payload["token_ids_logprob"] = token_ids_logprob
        with teacher_endpoint_slot(name, kind="score") as endpoint_wait_seconds:
            request_started = time.monotonic()
            output = post_json(teacher_url(name, cfg), payload, timeout=TEACHER_TIMEOUT)
            request_seconds = time.monotonic() - request_started
        meta = output.get("meta_info") if isinstance(output.get("meta_info"), dict) else {}
        chunk_log_probs = extract_input_logprobs(meta, chunk_len)
        chunk_topk, chunk_masks = _extract_requested_topk_logprobs(
            output,
            chunk_len,
            chunk_requested_topk,
        )
        timing = {
            "offset": offset,
            "endpoint_kind": "score",
            "chunk_response_token_count": chunk_len,
            "request_seconds": request_seconds,
            "endpoint_wait_seconds": endpoint_wait_seconds,
            "input_token_count": len(chunk_input_ids),
            "window_start": window_start,
            "window_end": chunk_end,
            "chunk_start_in_window": chunk_start_in_window,
            "truncated_prefix_token_count": window_start,
            "prompt_token_count": prompt_len,
            "teacher_context_length": score_context_len,
            "logprob_start_len": logprob_start_len,
            "input_logprob_rows": len(meta.get("input_token_logprobs") or []),
            "input_token_ids_logprob_rows": len(meta.get("input_token_ids_logprobs") or []),
            "input_requested_token_logprob_rows": len(meta.get("input_requested_token_logprobs") or []),
            "position_topk_enabled": bool(TEACHER_SCORE_POSITION_TOPK),
            "token_ids_logprob_count": len(token_ids_logprob),
        }
        return offset, chunk_log_probs, chunk_topk, chunk_masks, timing

    try:
        if chunk_workers > 1 and len(offsets) > 1:
            with ThreadPoolExecutor(max_workers=chunk_workers) as pool:
                chunk_results = [future.result() for future in as_completed(pool.submit(_score_chunk, offset) for offset in offsets)]
            chunk_results.sort(key=lambda item: item[0])
        else:
            chunk_results = [_score_chunk(offset) for offset in offsets]
        for _offset, chunk_log_probs, chunk_topk, chunk_masks, timing in chunk_results:
            teacher_log_probs.extend(chunk_log_probs)
            teacher_topk_parts.append(chunk_topk)
            teacher_mask_parts.append(chunk_masks)
            timings.append(timing)
    except Exception as exc:
        chunk_wall_seconds = time.monotonic() - chunk_wall_started
        timings.sort(key=lambda item: int(item.get("offset") or 0))
        return {
            "student_id": sid,
            "teacher": name,
            "status": "teacher_score_error",
            "error": str(exc),
            "teacher_log_probs": [],
            "topk_token_ids_sha256": requested_topk_sha256,
            "response_token_count": response_len,
            "num_teacher_log_probs": 0,
            "teacher_score_timing": {
                "request_seconds": time.monotonic() - total_started,
                "full_input_token_count": len(input_ids),
                "prompt_token_count": prompt_len,
                "response_token_count": response_len,
                "chunk_size": chunk_size,
                "chunk_count": len(offsets),
                "chunk_workers": chunk_workers,
                "chunk_request_wall_seconds": chunk_wall_seconds,
                "chunk_request_sum_seconds": sum(_timing(t, "request_seconds") for t in timings),
                "chunk_request_max_seconds": max((_timing(t, "request_seconds") for t in timings), default=0.0),
                "endpoint_wait_sum_seconds": sum(_timing(t, "endpoint_wait_seconds") for t in timings),
                "endpoint_wait_max_seconds": max((_timing(t, "endpoint_wait_seconds") for t in timings), default=0.0),
                "teacher_context_length": score_context_len,
                "max_request_input_token_count": max(
                    (int(t.get("input_token_count") or 0) for t in timings),
                    default=0,
                ),
                "chunks": timings,
            },
        }

    chunk_wall_seconds = time.monotonic() - chunk_wall_started
    timings.sort(key=lambda item: int(item.get("offset") or 0))
    status = "success" if len(teacher_log_probs) == response_len else "length_mismatch"
    teacher_topk = None
    teacher_topk_masks = None
    if requested_topk is not None:
        if all(part is not None for part in teacher_topk_parts) and all(part is not None for part in teacher_mask_parts):
            teacher_topk = [row for part in teacher_topk_parts for row in (part or [])]
            teacher_topk_masks = [row for part in teacher_mask_parts for row in (part or [])]
        else:
            teacher_topk = None
            teacher_topk_masks = None
    if requested_topk is not None:
        if teacher_topk is not None and teacher_topk_masks is not None:
            topk_total = sum(len(row) for row in teacher_topk_masks)
            topk_valid = sum(sum(float(x) for x in row) for row in teacher_topk_masks)
            status = "topk_success" if topk_valid == topk_total else "partial_topk"
        else:
            status = "missing_teacher_topk"
    result = {
        "student_id": sid,
        "teacher": name,
        "teacher_model_path": teacher_model_path(name, cfg),
        "status": status,
        # The sampled-token scores come from the same forced-scoring request as
        # the top-k rows.  vOPD uses them as its unbiased main estimator.
        "teacher_log_probs": teacher_log_probs,
        "topk_token_ids_sha256": requested_topk_sha256,
        "response_token_count": response_len,
        "num_teacher_log_probs": len(teacher_log_probs),
        "teacher_score_timing": {
            "request_seconds": time.monotonic() - total_started,
            "full_input_token_count": len(input_ids),
            "prompt_token_count": prompt_len,
            "response_token_count": response_len,
            "chunk_size": chunk_size,
            "chunk_count": len(offsets),
            "chunk_workers": chunk_workers,
            "chunk_request_wall_seconds": chunk_wall_seconds,
            "chunk_request_sum_seconds": sum(_timing(t, "request_seconds") for t in timings),
            "chunk_request_max_seconds": max((_timing(t, "request_seconds") for t in timings), default=0.0),
            "endpoint_wait_sum_seconds": sum(_timing(t, "endpoint_wait_seconds") for t in timings),
            "endpoint_wait_max_seconds": max((_timing(t, "endpoint_wait_seconds") for t in timings), default=0.0),
            "teacher_context_length": score_context_len,
            "max_request_input_token_count": max(
                (int(t.get("input_token_count") or 0) for t in timings),
                default=0,
            ),
            "chunks": timings,
        },
    }
    if requested_topk is not None:
        result["teacher_topk_log_probs"] = teacher_topk or []
        result["teacher_topk_logprob_masks"] = teacher_topk_masks or []
    return result


def _hydrate_student_tensor_sidecar(job: Path, base: Path, meta: dict[str, Any]) -> None:
    sidecar = meta.get("student_tensor_sidecar") or meta.get("tensor_sidecar")
    if not isinstance(sidecar, dict):
        return
    rel = _safe_relpath(str(sidecar.get("file") or ""))
    path = job / rel
    if not path.exists():
        path = base / rel
    if not path.exists():
        raise FileNotFoundError(f"missing OPD student tensor sidecar: {rel}")
    expected_sha = str(sidecar.get("sha256") or "")
    if expected_sha:
        actual_sha = _sha256_file(path)
        if actual_sha != expected_sha:
            raise ValueError(f"bad OPD student sidecar sha256 for {path}: {actual_sha} != {expected_sha}")
    if sidecar.get("format") != "student_npz_v1":
        raise ValueError(f"unsupported OPD student sidecar format at {path}: {sidecar.get('format')!r}")

    import numpy as np  # type: ignore[import-untyped]

    allowed = {"input_token_ids", "response_token_ids", "rollout_log_probs", "topk_token_ids", "topk_student_log_probs"}
    specs = sidecar.get("arrays") if isinstance(sidecar.get("arrays"), dict) else {}
    with np.load(path, allow_pickle=False) as data:
        for key in data.files:
            if key not in allowed:
                continue
            arr = data[key]
            spec = specs.get(key) if isinstance(specs, dict) else None
            expected_shape = spec.get("shape") if isinstance(spec, dict) else None
            if isinstance(expected_shape, list):
                shape = tuple(int(dim) for dim in expected_shape)
                if tuple(arr.shape) != shape:
                    raise ValueError(f"bad OPD student sidecar shape for {key}: {arr.shape} != {shape}")
            meta[key] = arr.tolist()


def load_student_rollout(job: Path, entry: dict[str, Any]) -> dict[str, Any]:
    """Load one student payload without running EDA.

    Score-only jobs use this path so their forced teacher scoring can start as
    soon as generation finishes, independently of the student-side EDA task.
    """
    sid = str(entry.get("id"))
    base = job / str(entry.get("path", f"student/{sid}"))
    meta = read_json(base / "meta.json") if (base / "meta.json").exists() else {}
    _hydrate_student_tensor_sidecar(job, base, meta)
    response = (base / "assistant_response.txt").read_text(encoding="utf-8")
    filename = meta.get("filename") or extract_filename(response) or f"{sid}.sv"
    if (base / "testbench.sv").exists():
        body = (base / "testbench.sv").read_text(encoding="utf-8")
    else:
        body = extract_verilog(response)
    testbench = {"filename": safe_filename(filename), "content": body} if body else None
    return {
        "id": sid,
        "sample_index": meta.get("sample_index"),
        "assistant_response": response,
        "assistant_response_file": f"{base.relative_to(job).as_posix()}/assistant_response.txt",
        "testbench": testbench,
        "input_token_ids": meta.get("input_token_ids") or meta.get("tokens") or [],
        "response_token_ids": meta.get("response_token_ids") or [],
        "response_token_count": int(meta.get("response_token_count") or len(meta.get("response_token_ids") or [])),
        "rollout_log_probs": meta.get("rollout_log_probs") or [],
        "topk_token_ids": meta.get("topk_token_ids") or [],
        "topk_student_log_probs": meta.get("topk_student_log_probs") or [],
        "assistant_response_sha256": meta.get("assistant_response_sha256") or "",
        "testbench_sha256": meta.get("testbench_sha256") or "",
        "teacher_scores": [],
    }


def score_student(job: Path, entry: dict[str, Any], context: SimpleNamespace, want_detail: bool) -> dict[str, Any]:
    loaded = load_student_rollout(job, entry)
    testbench = loaded.get("testbench") if isinstance(loaded.get("testbench"), dict) else None
    filename = testbench.get("filename") if testbench else None
    body = testbench.get("content") if testbench else None
    scored = run_eda(context, filename, body, want_detail)
    return {**loaded, **scored}


def score_teacher(entry: dict[str, Any], context: SimpleNamespace, want_detail: bool) -> dict[str, Any]:
    response = str(entry.get("assistant_response") or "")
    filename = extract_filename(response) or f"{entry['id']}.sv"
    body = extract_verilog(response)
    scored = run_eda(context, filename, body, want_detail)
    testbench = {"filename": safe_filename(filename), "content": body} if body else None
    return {**entry, "testbench": testbench, **scored}


def selection(student_entries: list[dict[str, Any]], teacher_entries: list[dict[str, Any]]) -> dict[str, Any]:
    best_student = max(student_entries, key=lambda e: float(e.get("reward", 0.0)), default=None)
    best_teacher = max(teacher_entries, key=lambda e: float(e.get("reward", 0.0)), default=None)
    return {
        "best_student_id": best_student.get("id") if best_student else None,
        "best_student_reward": float(best_student.get("reward", 0.0)) if best_student else 0.0,
        "best_teacher_id": best_teacher.get("id") if best_teacher else None,
        "best_teacher_reward": float(best_teacher.get("reward", 0.0)) if best_teacher else None,
    }


def _safe_component(value: Any, default: str) -> str:
    text = str(value or default)
    text = re.sub(r"[^A-Za-z0-9._-]", "_", text)
    return text[:80] or default


def _write_teacher_topk_sidecar(
    staging: Path,
    *,
    student_id: str,
    teacher: str,
    score_idx: int,
    log_probs: Any,
    masks: Any,
    sampled_log_probs: Any | None = None,
    token_ids_sha256: str | None = None,
) -> dict[str, Any]:
    import numpy as np  # type: ignore[import-untyped]

    tensor_dir = staging / "score_tensors"
    tensor_dir.mkdir(parents=True, exist_ok=True)
    sid = _safe_component(student_id, "student")
    tname = _safe_component(teacher, "teacher")
    rel = f"score_tensors/{sid}_{tname}_{score_idx:02d}.npz"
    path = staging / rel

    dtype = np.float16 if SCORE_TENSOR_DTYPE == "float16" else np.float32
    log_probs_arr = np.asarray(log_probs, dtype=dtype)
    masks_arr = np.asarray(masks, dtype=np.uint8)
    if log_probs_arr.ndim != 2 or masks_arr.ndim != 2 or log_probs_arr.shape != masks_arr.shape:
        raise ValueError(
            "bad teacher top-k sidecar shape "
            f"student={student_id} teacher={teacher} log_probs={log_probs_arr.shape} masks={masks_arr.shape}"
        )
    arrays: dict[str, Any] = {
        "teacher_topk_log_probs": log_probs_arr,
        "teacher_topk_logprob_masks": masks_arr,
    }
    sampled_arr = None
    if sampled_log_probs is not None:
        # This vector is tiny relative to the T x K matrix and drives the
        # unbiased vOPD term, so keep it float32 even if the optional top-k
        # sidecar compression is configured to use float16.
        sampled_arr = np.asarray(sampled_log_probs, dtype=np.float32)
        if sampled_arr.ndim != 1 or sampled_arr.shape[0] != log_probs_arr.shape[0]:
            raise ValueError(
                "bad teacher sampled-logprob sidecar shape "
                f"student={student_id} teacher={teacher} sampled={sampled_arr.shape} "
                f"expected=({log_probs_arr.shape[0]},)"
            )
        arrays["teacher_log_probs"] = sampled_arr
    np.savez_compressed(path, **arrays)
    descriptor = {
        "format": "npz_v2" if sampled_arr is not None else "npz_v1",
        "file": rel,
        "sha256": _sha256_file(path),
        "shape": [int(log_probs_arr.shape[0]), int(log_probs_arr.shape[1])],
        "log_probs_dtype": str(log_probs_arr.dtype),
        "masks_dtype": str(masks_arr.dtype),
        "aligned_to": "student_topk_token_ids_order",
        "topk_token_ids_sha256": token_ids_sha256 or "",
    }
    if sampled_arr is not None:
        descriptor["teacher_log_probs_shape"] = [int(sampled_arr.shape[0])]
        descriptor["teacher_log_probs_dtype"] = str(sampled_arr.dtype)
    return descriptor


def _materialize_score_sidecars(obj: dict[str, Any], staging: Path) -> None:
    for entry in obj.get("student_rollouts") or []:
        if not isinstance(entry, dict):
            continue
        for score_idx, score in enumerate(entry.get("teacher_scores") or []):
            if not isinstance(score, dict):
                continue
            log_probs = score.pop("teacher_topk_log_probs", None)
            masks = score.pop("teacher_topk_logprob_masks", None)
            if not log_probs or not masks:
                continue
            sampled_log_probs = score.get("teacher_log_probs")
            if not isinstance(sampled_log_probs, list) or len(sampled_log_probs) != len(log_probs):
                sampled_log_probs = None
            score["teacher_topk_sidecar"] = _write_teacher_topk_sidecar(
                staging,
                student_id=str(entry.get("id") or score.get("student_id") or "student"),
                teacher=str(score.get("teacher") or "teacher"),
                score_idx=score_idx,
                log_probs=log_probs,
                masks=masks,
                sampled_log_probs=sampled_log_probs,
                token_ids_sha256=str(score.get("topk_token_ids_sha256") or ""),
            )
            if sampled_log_probs is not None:
                score.pop("teacher_log_probs", None)


def _compact_score(score: dict[str, Any]) -> dict[str, Any]:
    keep = {
        "student_id",
        "teacher",
        "teacher_model_path",
        "status",
        "error",
        "teacher_log_probs",
        "response_token_count",
        "num_teacher_log_probs",
        "teacher_score_timing",
        "teacher_topk_sidecar",
        "topk_token_ids_sha256",
    }
    return {k: v for k, v in score.items() if k in keep and v not in (None, [], {})}


def _compact_student_entry(entry: dict[str, Any]) -> dict[str, Any]:
    keep = {
        "id",
        "sample_index",
        "assistant_response_sha256",
        "testbench_sha256",
        "reward",
        "eda_summary",
        "eda_log",
        "response_token_count",
        "teacher_logprob_teacher",
    }
    compact = {k: v for k, v in entry.items() if k in keep and v not in (None, [], {})}
    compact["teacher_scores"] = [
        _compact_score(score)
        for score in (entry.get("teacher_scores") or [])
        if isinstance(score, dict)
    ]
    return compact


def _compact_teacher_entry(entry: dict[str, Any]) -> dict[str, Any]:
    keep = {
        "id",
        "teacher",
        "teacher_name",
        "teacher_model_path",
        "reward",
        "eda_summary",
        "eda_log",
        "response_token_count",
    }
    return {k: v for k, v in entry.items() if k in keep and v not in (None, [], {})}


def _compact_result_for_training(obj: dict[str, Any]) -> dict[str, Any]:
    compact = {
        "version": obj.get("version"),
        "status": obj.get("status"),
        "result_kind": "training_compact",
        "job_kind": obj.get("job_kind", "round"),
        "job_id": obj.get("job_id"),
        "dataset_id": obj.get("dataset_id"),
        "rollout_id": obj.get("rollout_id"),
        "round_idx": obj.get("round_idx"),
        "teacher_model_paths": obj.get("teacher_model_paths") or {},
        "selection": obj.get("selection") or {},
        "elapsed_s": obj.get("elapsed_s"),
        "timing": obj.get("timing") or {},
        "audit_result_file": "audit/result.json.gz" if AUDIT_RESULT_GZIP else "audit/result.json",
        "score_tensor_format": "npz_v1",
        "student_rollouts": [
            _compact_student_entry(entry)
            for entry in (obj.get("student_rollouts") or [])
            if isinstance(entry, dict)
        ],
        "teacher_rollouts": [
            _compact_teacher_entry(entry)
            for entry in (obj.get("teacher_rollouts") or [])
            if isinstance(entry, dict)
        ],
    }
    return compact


def _write_audit_result(staging: Path, obj: dict[str, Any]) -> str:
    audit_dir = staging / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    if AUDIT_RESULT_GZIP:
        rel = "audit/result.json.gz"
        with gzip.open(staging / rel, "wt", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, sort_keys=True, indent=2)
    else:
        rel = "audit/result.json"
        write_json(staging / rel, obj)
    return rel


def publish(job_id: str, obj: dict[str, Any]) -> None:
    staging = RES / f"{job_id}.staging"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    if COMPACT_SCORE_RESULT and obj.get("status") == "success":
        _materialize_score_sidecars(obj, staging)
        audit_file = _write_audit_result(staging, obj)
        compact = _compact_result_for_training(obj)
        compact["audit_result_file"] = audit_file
        write_json(staging / "result.json", compact)
    else:
        write_json(staging / "result.json", obj)
    final = RES / job_id
    if final.exists():
        shutil.rmtree(final)
    os.rename(staging, final)
    summary_obj = read_json(final / "result.json")
    summary = summarize_result(summary_obj, final)
    summary["worker_id"] = WORKER_ID
    summary["queue_transport"] = QUEUE_TRANSPORT
    if QUEUE_TRANSPORT == "sftp":
        summary["remote_result_dir"] = f"results/{NAMESPACE}/{job_id}"
        summary["remote_result_file"] = f"results/{NAMESPACE}/{job_id}/result.json"
    write_json(final / "summary.json", summary)
    append_index(summary)
    (final / ".done").write_text("", encoding="utf-8")
    if QUEUE_BACKEND is not None:
        QUEUE_BACKEND.publish_result(job_id, final, summary)


def _score_teacher_names(teacher_request: dict[str, Any], teacher_specs: list[dict[str, Any]], cfg: dict[str, Any]) -> list[str]:
    requested = teacher_request.get("score_teacher_names")
    if isinstance(requested, list):
        names = [str(name) for name in requested if str(name)]
    else:
        names = [str(spec.get("name") or "") for spec in teacher_specs if isinstance(spec, dict)]
    if not names:
        names = [str(name) for name in cfg]
    return list(dict.fromkeys(name for name in names if name))


def _log_teacher_score_timing(job_id: str, score: dict[str, Any]) -> None:
    score_timing = score.get("teacher_score_timing") if isinstance(score.get("teacher_score_timing"), dict) else {}
    log(
        "OPD_TEACHER_SCORE_TIMING",
        job_id,
        f"student={score.get('student_id')}",
        f"teacher={score.get('teacher')}",
        f"status={score.get('status')}",
        f"total={_timing(score_timing, 'request_seconds'):.3f}",
        f"endpoint_wait={_timing(score_timing, 'endpoint_wait_sum_seconds'):.3f}",
        f"chunk_wall={_timing(score_timing, 'chunk_request_wall_seconds'):.3f}",
        f"chunk_sum={_timing(score_timing, 'chunk_request_sum_seconds'):.3f}",
        f"chunk_max={_timing(score_timing, 'chunk_request_max_seconds'):.3f}",
        f"chunks={int(score_timing.get('chunk_count') or 0)}",
        f"chunk_workers={int(score_timing.get('chunk_workers') or 0)}",
        f"prompt_tokens={int(score_timing.get('prompt_token_count') or 0)}",
        f"response_tokens={int(score_timing.get('response_token_count') or 0)}",
        f"max_input={int(score_timing.get('max_request_input_token_count') or 0)}",
    )
    for chunk in score_timing.get("chunks") or []:
        if not isinstance(chunk, dict):
            continue
        log(
            "OPD_TEACHER_SCORE_CHUNK_TIMING",
            job_id,
            f"student={score.get('student_id')}",
            f"teacher={score.get('teacher')}",
            f"offset={int(chunk.get('offset') or 0)}",
            f"response_tokens={int(chunk.get('chunk_response_token_count') or 0)}",
            f"input_tokens={int(chunk.get('input_token_count') or 0)}",
            f"endpoint_wait={_timing(chunk, 'endpoint_wait_seconds'):.3f}",
            f"request={_timing(chunk, 'request_seconds'):.3f}",
            f"window_start={int(chunk.get('window_start') or 0)}",
            f"logprob_start={int(chunk.get('logprob_start_len') or 0)}",
        )


def _process_student_score_job(
    job: Path,
    manifest: dict[str, Any],
    *,
    job_id: str,
    started_wall: float,
    started: float,
    timing: dict[str, Any],
) -> None:
    """Process one student-only forced-score request without running EDA."""
    phase_started = time.monotonic()
    teacher_cfg = load_teacher_config()
    teacher_request = manifest.get("teacher_request") if isinstance(manifest.get("teacher_request"), dict) else {}
    teacher_specs = manifest.get("teachers") if isinstance(manifest.get("teachers"), list) else []
    teacher_names = _score_teacher_names(teacher_request, teacher_specs, teacher_cfg)
    if not teacher_names:
        raise ValueError("student score job has no teacher names")
    student_specs = manifest.get("student_rollouts") if isinstance(manifest.get("student_rollouts"), list) else []
    if not student_specs:
        raise ValueError("student score job has no student rollout")
    student_entries = [load_student_rollout(job, entry) for entry in student_specs if isinstance(entry, dict)]
    if not student_entries:
        raise ValueError("student score job has no valid student rollout")
    topk_k = int(teacher_request.get("student_topk_k") or 0)
    timing["load_request_seconds"] = time.monotonic() - phase_started
    timing["student_eda_seconds"] = 0.0
    timing["teacher_generate_seconds"] = 0.0
    timing["teacher_eda_seconds"] = 0.0
    timing["teacher_pipeline_seconds"] = 0.0
    timing["teacher_pipeline_overlap_saved_seconds"] = 0.0

    phase_started = time.monotonic()
    work = [(entry, name) for entry in student_entries for name in teacher_names]
    score_workers = max(1, min(MAX_JOB_WORKERS, len(work) or 1))
    score_order = {name: index for index, name in enumerate(teacher_names)}
    with ThreadPoolExecutor(max_workers=score_workers) as pool:
        futures = {
            pool.submit(score_teacher_on_student, name, entry, teacher_cfg, topk_k): (entry, name)
            for entry, name in work
        }
        for future in as_completed(futures):
            entry, requested_name = futures[future]
            score = future.result()
            if str(score.get("student_id") or "") != str(entry.get("id") or ""):
                raise ValueError(
                    f"teacher score student mismatch: requested={entry.get('id')} got={score.get('student_id')}"
                )
            if str(score.get("teacher") or "") != requested_name:
                raise ValueError(
                    f"teacher score teacher mismatch: requested={requested_name} got={score.get('teacher')}"
                )
            entry.setdefault("teacher_scores", []).append(score)
            _log_teacher_score_timing(job_id, score)
    for entry in student_entries:
        entry["teacher_scores"].sort(key=lambda item: score_order.get(str(item.get("teacher") or ""), len(score_order)))
    timing["teacher_score_student_seconds"] = time.monotonic() - phase_started
    timing["teacher_score_student_max_seconds"] = max(
        (
            _timing(score.get("teacher_score_timing") or {}, "request_seconds")
            for entry in student_entries
            for score in entry.get("teacher_scores") or []
            if isinstance(score, dict)
        ),
        default=0.0,
    )
    timing["teacher_score_student_sum_seconds"] = sum(
        _timing(score.get("teacher_score_timing") or {}, "request_seconds")
        for entry in student_entries
        for score in entry.get("teacher_scores") or []
        if isinstance(score, dict)
    )
    timing["teacher_score_request_count"] = len(work)
    timing["teacher_score_workers"] = score_workers

    worker_before_publish = time.monotonic() - started
    timing["worker_total_before_publish_seconds"] = worker_before_publish
    timing["worker_accounted_before_publish_seconds"] = sum(
        float(timing.get(key, 0.0) or 0.0)
        for key in ("job_semaphore_wait_seconds", "load_request_seconds", "teacher_score_student_seconds")
    )
    timing["worker_unaccounted_before_publish_seconds"] = max(
        0.0, worker_before_publish - timing["worker_accounted_before_publish_seconds"]
    )
    result = {
        "version": SCHEMA_VERSION,
        "status": "success",
        "job_kind": "student_score",
        "job_id": job_id,
        "dataset_id": manifest.get("dataset_id"),
        "rollout_id": manifest.get("rollout_id"),
        "round_idx": manifest.get("round_idx"),
        "student_rollouts": student_entries,
        "teacher_rollouts": [],
        "teacher_model_paths": teacher_model_paths(teacher_names, teacher_cfg),
        "selection": selection([], []),
        "elapsed_s": time.time() - started_wall,
        "timing": timing,
        "teacher_worker": {"id": WORKER_ID, "queue_transport": QUEUE_TRANSPORT},
    }
    publish_started = time.monotonic()
    publish(job_id, result)
    timing["publish_seconds"] = time.monotonic() - publish_started
    timing["worker_total_seconds"] = time.monotonic() - started
    log(
        "OPD_WORKER_TIMING",
        job_id,
        "kind=student_score",
        f"sem_wait={timing.get('job_semaphore_wait_seconds', 0.0):.3f}",
        f"load={timing.get('load_request_seconds', 0.0):.3f}",
        f"teacher_score={timing.get('teacher_score_student_seconds', 0.0):.3f}",
        f"publish={timing.get('publish_seconds', 0.0):.3f}",
        f"total={timing.get('worker_total_seconds', 0.0):.3f}",
    )
    log("done", job_id, "kind=student_score", f"students={len(student_entries)}", f"score_requests={len(work)}")


def process(job: Path) -> None:
    job_id = job.name
    started_wall = time.time()
    started = time.monotonic()
    timing: dict[str, Any] = {}
    try:
        job_kind = str(read_json(job / "manifest.json").get("job_kind") or "round")
    except Exception:
        job_kind = "round"
    job_sem = _SCORE_JOB_SEM if job_kind == "student_score" else _ROUND_JOB_SEM
    timing["job_kind"] = job_kind
    timing["job_semaphore"] = "score" if job_kind == "student_score" else "round"
    sem_wait_started = time.monotonic()
    with job_sem:
        timing["job_semaphore_wait_seconds"] = time.monotonic() - sem_wait_started
        try:
            phase_started = time.monotonic()
            manifest = read_json(job / "manifest.json")
            if manifest.get("version") != SCHEMA_VERSION:
                raise ValueError(f"bad schema version: {manifest.get('version')}")
            if str(manifest.get("job_kind") or "round") == "student_score":
                _process_student_score_job(
                    job,
                    manifest,
                    job_id=job_id,
                    started_wall=started_wall,
                    started=started,
                    timing=timing,
                )
                return
            prompt = (job / manifest.get("prompt_file", "prompt.txt")).read_text(encoding="utf-8")
            context = context_from_dict(read_json(job / manifest.get("context_file", "context.json")))
            want_detail = bool((manifest.get("eda") or {}).get("want_detail", False))
            sampling_params = manifest.get("sampling_params") if isinstance(manifest.get("sampling_params"), dict) else {}
            teacher_cfg = load_teacher_config()
            timing["load_request_seconds"] = time.monotonic() - phase_started

            student_specs = manifest.get("student_rollouts") if isinstance(manifest.get("student_rollouts"), list) else []
            phase_started = time.monotonic()
            with ThreadPoolExecutor(max_workers=max(1, min(MAX_JOB_WORKERS, len(student_specs) or 1))) as pool:
                student_entries = list(pool.map(lambda e: score_student(job, e, context, want_detail), student_specs))
            timing["student_eda_seconds"] = time.monotonic() - phase_started
            timing["student_eda_max_seconds"] = max(
                (
                    float((entry.get("eda_timing") or {}).get("total_seconds", 0.0) or 0.0)
                    for entry in student_entries
                    if isinstance(entry, dict)
                ),
                default=0.0,
            )
            timing["student_eda_sum_seconds"] = sum(
                float((entry.get("eda_timing") or {}).get("total_seconds", 0.0) or 0.0)
                for entry in student_entries
                if isinstance(entry, dict)
            )
            for entry in student_entries:
                eda_timing = entry.get("eda_timing") if isinstance(entry.get("eda_timing"), dict) else {}
                eda_log = entry.get("eda_log") if isinstance(entry.get("eda_log"), dict) else {}
                eda_summary = entry.get("eda_summary") if isinstance(entry.get("eda_summary"), dict) else {}
                log(
                    "OPD_STUDENT_EDA_TIMING",
                    job_id,
                    f"student={entry.get('id')}",
                    f"status={eda_log.get('status', entry.get('status', ''))}",
                    f"reward={_as_float(entry.get('reward')):.4f}",
                    f"total={_timing(eda_timing, 'total_seconds'):.3f}",
                    f"submit={_timing(eda_timing, 'submit_cov_job_seconds'):.3f}",
                    f"coverage={_as_float(eda_summary.get('overall_coverage', 0.0)):.4f}",
                    f"response_tokens={int(entry.get('response_token_count') or 0)}",
                )

            phase_started = time.monotonic()
            teacher_specs = manifest.get("teachers") if isinstance(manifest.get("teachers"), list) else []
            teacher_jobs: list[tuple[str, int]] = []
            for spec in teacher_specs:
                if not isinstance(spec, dict):
                    continue
                name = str(spec.get("name"))
                for i in range(int(spec.get("n", 1) or 0)):
                    teacher_jobs.append((name, i))
            teacher_names = [name for name, _slot in teacher_jobs]
            teacher_entries: list[dict[str, Any]] = []
            teacher_request = manifest.get("teacher_request") if isinstance(manifest.get("teacher_request"), dict) else {}
            teacher_rollouts_file = manifest.get("teacher_rollouts_file")
            timing["parse_teacher_request_seconds"] = time.monotonic() - phase_started
            phase_started = time.monotonic()
            if isinstance(teacher_rollouts_file, str) and teacher_rollouts_file:
                provided = read_json(job / teacher_rollouts_file).get("teacher_rollouts", [])
                if isinstance(provided, list):
                    teacher_entries = [e for e in provided if isinstance(e, dict)]
            timing["load_teacher_rollouts_seconds"] = time.monotonic() - phase_started
            if teacher_jobs and teacher_request.get("generate_teacher_rollouts", True):
                pipeline_started = time.monotonic()
                generated: list[dict[str, Any]] = []
                gen_workers = max(1, min(MAX_JOB_WORKERS, len(teacher_jobs)))
                eda_workers = max(1, min(MAX_JOB_WORKERS, len(teacher_jobs)))
                first_eda_submit: float | None = None
                generation_done = pipeline_started
                eda_done = pipeline_started
                with ThreadPoolExecutor(max_workers=gen_workers) as gen_pool, ThreadPoolExecutor(max_workers=eda_workers) as eda_pool:
                    gen_futures = {
                        gen_pool.submit(generate_teacher_rollout, name, slot, prompt, sampling_params, teacher_cfg): (name, slot)
                        for name, slot in teacher_jobs
                    }
                    eda_futures: dict[Any, dict[str, Any]] = {}
                    for future in as_completed(gen_futures):
                        name, slot = gen_futures[future]
                        entry = future.result()
                        generated.append(entry)
                        generation_done = time.monotonic()
                        gen_timing = entry.get("teacher_generation_timing") if isinstance(entry.get("teacher_generation_timing"), dict) else {}
                        budget = entry.get("teacher_generation_budget") if isinstance(entry.get("teacher_generation_budget"), dict) else {}
                        log(
                            "OPD_TEACHER_GENERATE_TIMING",
                            job_id,
                            f"teacher={name}",
                            f"slot={slot}",
                            f"status={entry.get('teacher_generation_status', '')}",
                            f"total={_timing(gen_timing, 'total_seconds'):.3f}",
                            f"request={_timing(gen_timing, 'request_seconds'):.3f}",
                            f"budget={_timing(budget, 'budget_seconds'):.3f}",
                            f"tokenize={_timing(budget, 'tokenize_seconds'):.3f}",
                            f"prompt_tokens={int(budget.get('prompt_token_count') or 0)}",
                            f"generated_tokens={int(gen_timing.get('generated_token_count') or 0)}",
                            f"chars={int(gen_timing.get('response_chars') or 0)}",
                        )
                        if first_eda_submit is None:
                            first_eda_submit = time.monotonic()
                        eda_futures[eda_pool.submit(score_teacher, entry, context, want_detail)] = entry
                    timing["teacher_generate_seconds"] = generation_done - pipeline_started
                    for eda_future in as_completed(eda_futures):
                        scored = eda_future.result()
                        teacher_entries.append(scored)
                        eda_done = time.monotonic()
                        eda_timing = scored.get("eda_timing") if isinstance(scored.get("eda_timing"), dict) else {}
                        eda_log = scored.get("eda_log") if isinstance(scored.get("eda_log"), dict) else {}
                        eda_summary = scored.get("eda_summary") if isinstance(scored.get("eda_summary"), dict) else {}
                        log(
                            "OPD_TEACHER_EDA_TIMING",
                            job_id,
                            f"teacher={scored.get('teacher', scored.get('teacher_name', ''))}",
                            f"id={scored.get('id', '')}",
                            f"status={eda_log.get('status', scored.get('status', ''))}",
                            f"reward={_as_float(scored.get('reward')):.4f}",
                            f"total={_timing(eda_timing, 'total_seconds'):.3f}",
                            f"submit={_timing(eda_timing, 'submit_cov_job_seconds'):.3f}",
                            f"coverage={_as_float(eda_summary.get('overall_coverage', 0.0)):.4f}",
                        )
                timing["teacher_eda_seconds"] = eda_done - (first_eda_submit or eda_done)
                timing["teacher_pipeline_seconds"] = eda_done - pipeline_started
                timing["teacher_pipeline_overlap_saved_seconds"] = max(
                    0.0,
                    timing["teacher_generate_seconds"] + timing["teacher_eda_seconds"] - timing["teacher_pipeline_seconds"],
                )
                timing["teacher_generate_workers"] = gen_workers
                timing["teacher_eda_workers"] = eda_workers
                timing["teacher_generate_max_seconds"] = max(
                    (
                        float((entry.get("teacher_generation_timing") or {}).get("total_seconds", 0.0) or 0.0)
                        for entry in generated
                        if isinstance(entry, dict)
                    ),
                    default=0.0,
                )
                timing["teacher_generate_sum_seconds"] = sum(
                    float((entry.get("teacher_generation_timing") or {}).get("total_seconds", 0.0) or 0.0)
                    for entry in generated
                    if isinstance(entry, dict)
                )
                timing["teacher_eda_max_seconds"] = max(
                    (
                        float((entry.get("eda_timing") or {}).get("total_seconds", 0.0) or 0.0)
                        for entry in teacher_entries
                        if isinstance(entry, dict)
                    ),
                    default=0.0,
                )
                timing["teacher_eda_sum_seconds"] = sum(
                    float((entry.get("eda_timing") or {}).get("total_seconds", 0.0) or 0.0)
                    for entry in teacher_entries
                    if isinstance(entry, dict)
                )
            else:
                timing.setdefault("teacher_generate_seconds", 0.0)
                timing.setdefault("teacher_generate_max_seconds", 0.0)
                timing.setdefault("teacher_generate_sum_seconds", 0.0)
                timing.setdefault("teacher_eda_seconds", 0.0)
                timing.setdefault("teacher_eda_max_seconds", 0.0)
                timing.setdefault("teacher_eda_sum_seconds", 0.0)
                timing.setdefault("teacher_pipeline_seconds", 0.0)
                timing.setdefault("teacher_pipeline_overlap_saved_seconds", 0.0)

            phase_started = time.monotonic()
            sel = selection(student_entries, teacher_entries)
            timing["selection_seconds"] = time.monotonic() - phase_started
            timing["teacher_score_student_seconds"] = 0.0
            timing["teacher_score_student_max_seconds"] = 0.0
            score_teacher_names = _score_teacher_names(teacher_request, teacher_specs, teacher_cfg)
            if teacher_request.get("score_student_rollouts") and score_teacher_names:
                phase_started = time.monotonic()
                work = [(entry, name) for entry in student_entries for name in score_teacher_names]
                score_workers = max(1, min(MAX_JOB_WORKERS, len(work) or 1))
                score_order = {name: index for index, name in enumerate(score_teacher_names)}
                with ThreadPoolExecutor(max_workers=score_workers) as pool:
                    futures = {
                        pool.submit(score_teacher_on_student, name, entry, teacher_cfg, int(teacher_request.get("student_topk_k") or 0)): (entry, name)
                        for entry, name in work
                    }
                    for future in as_completed(futures):
                        entry, requested_name = futures[future]
                        score = future.result()
                        if str(score.get("student_id") or "") != str(entry.get("id") or ""):
                            raise ValueError(
                                f"teacher score student mismatch: requested={entry.get('id')} got={score.get('student_id')}"
                            )
                        if str(score.get("teacher") or "") != requested_name:
                            raise ValueError(
                                f"teacher score teacher mismatch: requested={requested_name} got={score.get('teacher')}"
                            )
                        entry.setdefault("teacher_scores", []).append(score)
                        _log_teacher_score_timing(job_id, score)
                for entry in student_entries:
                    entry["teacher_scores"].sort(
                        key=lambda item: score_order.get(str(item.get("teacher") or ""), len(score_order))
                    )
                timing["teacher_score_student_seconds"] = time.monotonic() - phase_started
                timing["teacher_score_student_max_seconds"] = max(
                    (
                        _timing(score.get("teacher_score_timing") or {}, "request_seconds")
                        for entry in student_entries
                        for score in entry.get("teacher_scores") or []
                        if isinstance(score, dict)
                    ),
                    default=0.0,
                )
                timing["teacher_score_student_sum_seconds"] = sum(
                    _timing(score.get("teacher_score_timing") or {}, "request_seconds")
                    for entry in student_entries
                    for score in entry.get("teacher_scores") or []
                    if isinstance(score, dict)
                )
                timing["teacher_score_request_count"] = len(work)
                timing["teacher_score_workers"] = score_workers

                # Training keeps these tensors locally and only needs the
                # teacher score payload from the relay.  Dropping them here
                # keeps audit data focused on testbenches/EDA while avoiding a
                # large transfer back to the student host.
                for entry in student_entries:
                    for key in (
                        "input_token_ids",
                        "response_token_ids",
                        "rollout_log_probs",
                        "topk_token_ids",
                        "topk_student_log_probs",
                    ):
                        entry.pop(key, None)

            worker_before_publish = time.monotonic() - started
            accounted_before_publish = sum(
                float(timing.get(key, 0.0) or 0.0)
                for key in (
                    "job_semaphore_wait_seconds",
                    "load_request_seconds",
                    "student_eda_seconds",
                    "parse_teacher_request_seconds",
                    "load_teacher_rollouts_seconds",
                    "teacher_pipeline_seconds",
                    "selection_seconds",
                    "teacher_score_student_seconds",
                )
            )
            timing["worker_total_before_publish_seconds"] = worker_before_publish
            timing["worker_accounted_before_publish_seconds"] = accounted_before_publish
            timing["worker_unaccounted_before_publish_seconds"] = max(
                0.0, worker_before_publish - accounted_before_publish
            )
            result = {
                "version": SCHEMA_VERSION,
                "status": "success",
                "job_id": job_id,
                "dataset_id": manifest.get("dataset_id"),
                "rollout_id": manifest.get("rollout_id"),
                "round_idx": manifest.get("round_idx"),
                "student_rollouts": student_entries,
                "teacher_rollouts": teacher_entries,
                "teacher_model_paths": teacher_model_paths(teacher_names, teacher_cfg),
                "selection": sel,
                "elapsed_s": time.time() - started_wall,
                "timing": timing,
                "teacher_worker": {
                    "id": WORKER_ID,
                    "queue_transport": QUEUE_TRANSPORT,
                },
            }
            publish_started = time.monotonic()
            publish(job_id, result)
            publish_seconds = time.monotonic() - publish_started
            timing["publish_seconds"] = publish_seconds
            timing["worker_total_seconds"] = time.monotonic() - started
            log(
                "OPD_WORKER_TIMING",
                job_id,
                f"sem_wait={timing.get('job_semaphore_wait_seconds', 0.0):.3f}",
                f"load={timing.get('load_request_seconds', 0.0):.3f}",
                f"student_eda={timing.get('student_eda_seconds', 0.0):.3f}",
                f"teacher_generate={timing.get('teacher_generate_seconds', 0.0):.3f}",
                f"teacher_eda={timing.get('teacher_eda_seconds', 0.0):.3f}",
                f"teacher_pipeline={timing.get('teacher_pipeline_seconds', 0.0):.3f}",
                f"teacher_overlap_saved={timing.get('teacher_pipeline_overlap_saved_seconds', 0.0):.3f}",
                f"teacher_score={timing.get('teacher_score_student_seconds', 0.0):.3f}",
                f"publish={publish_seconds:.3f}",
                f"unaccounted={timing.get('worker_unaccounted_before_publish_seconds', 0.0):.3f}",
                f"total={time.monotonic() - started:.3f}",
            )
            log("done", job_id, f"students={len(student_entries)}", f"teachers={len(teacher_entries)}")
        except Exception as exc:
            result = {
                "version": SCHEMA_VERSION,
                "status": "failed",
                "job_id": job_id,
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "elapsed_s": time.time() - started_wall,
                "timing": {**timing, "worker_total_before_failure_seconds": time.monotonic() - started},
                "teacher_worker": {
                    "id": WORKER_ID,
                    "queue_transport": QUEUE_TRANSPORT,
                },
            }
            publish(job_id, result)
            log("FAIL", job_id, exc)
        finally:
            STATE.mkdir(parents=True, exist_ok=True)
            (STATE / f"{job_id}.processed").write_text(str(time.time()), encoding="utf-8")
            if QUEUE_BACKEND is not None:
                with contextlib.suppress(Exception):
                    QUEUE_BACKEND.finish_job(job_id)
            with _LOCK:
                _INFLIGHT.discard(job_id)


def gc() -> None:
    if JOB_TTL <= 0:
        return
    now = time.time()
    for base in (IN, RES):
        for d in base.iterdir() if base.exists() else []:
            with contextlib.suppress(Exception):
                if d.is_dir() and now - d.stat().st_mtime > JOB_TTL:
                    shutil.rmtree(d, ignore_errors=True)
    for marker in STATE.iterdir() if STATE.exists() else []:
        with contextlib.suppress(Exception):
            if now - marker.stat().st_mtime > JOB_TTL and not (IN / marker.stem).exists():
                marker.unlink()


def main() -> None:
    global QUEUE_BACKEND
    QUEUE_BACKEND = build_queue_backend()
    QUEUE_BACKEND.prepare()
    log(
        f"opd worker start XFER={XFER} namespace={NAMESPACE} mock={MOCK} "
        f"round_jobs={MAX_CONCURRENT_ROUND_JOBS} score_jobs={MAX_CONCURRENT_SCORE_JOBS} "
        f"endpoint_total={TEACHER_ENDPOINT_MAX_INFLIGHT} "
        f"endpoint_generate={TEACHER_GENERATE_MAX_INFLIGHT} "
        f"endpoint_score={TEACHER_SCORE_MAX_INFLIGHT} "
        f"queue={QUEUE_TRANSPORT} worker_id={WORKER_ID}"
    )
    last_gc = 0.0
    while True:
        now = time.time()
        for job in QUEUE_BACKEND.poll_jobs():
            threading.Thread(target=process, args=(job,), daemon=True).start()
        if now - last_gc > 300:
            gc()
            last_gc = now
        time.sleep(POLL_SEC)


if __name__ == "__main__":
    main()
