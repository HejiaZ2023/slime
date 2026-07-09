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
import gzip
import hashlib
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

SCHEMA_VERSION = "opd_exchange_v1"

SLIME_DIR = Path(os.environ.get("OPD_SLIME_DIR", "/home/slu375/docker_scripts/ctr_slime/slime-OPD"))
LLM4COV_SRC = SLIME_DIR / "third_party" / "llm4cov_oss" / "src"
if LLM4COV_SRC.exists():
    sys.path.insert(0, str(LLM4COV_SRC))

from llm4cov.eda_client.xfer_client import submit_cov_job  # noqa: E402

QUEUE_TRANSPORT = os.environ.get("OPD_QUEUE_TRANSPORT", "local").strip().lower()
if QUEUE_TRANSPORT == "sftp":
    XFER = Path(os.environ.get("OPD_LOCAL_XFER_DIR", "/tmp/llm4cov_opd_worker_xfer"))
else:
    XFER = Path(os.environ.get("OPD_XFER_DIR", "/mnt/raid0_ssd/eda/xfer"))
NAMESPACE = os.environ.get("OPD_NAMESPACE", "opd")
IN = XFER / "incoming" / NAMESPACE
RES = XFER / "results" / NAMESPACE
WORK = XFER / ".work" / NAMESPACE
STATE = XFER / ".state" / NAMESPACE
POLL_SEC = float(os.environ.get("OPD_POLL_SEC", "2"))
JOB_TTL = int(os.environ.get("OPD_JOB_TTL", "0"))  # 0 keeps OPD audit artifacts indefinitely
QUEUE_CLAIM_TTL = float(os.environ.get("OPD_QUEUE_CLAIM_TTL", "7200"))
WORKER_ID = re.sub(
    r"[^A-Za-z0-9_.-]",
    "_",
    os.environ.get("OPD_WORKER_ID", f"{os.uname().nodename}-{os.getpid()}"),
)
MAX_CONCURRENT_JOBS = int(os.environ.get("OPD_MAX_CONCURRENT_JOBS", "4"))
MAX_JOB_WORKERS = int(os.environ.get("OPD_MAX_JOB_WORKERS", "4"))
TEACHER_TIMEOUT = float(os.environ.get("OPD_TEACHER_TIMEOUT", "900"))
TEACHER_CONTEXT_LENGTH = int(os.environ.get("OPD_TEACHER_CONTEXT_LENGTH", "32768"))
TEACHER_TOKEN_BUDGET_MARGIN = int(os.environ.get("OPD_TEACHER_TOKEN_BUDGET_MARGIN", "256"))
TEACHER_MIN_NEW_TOKENS = int(os.environ.get("OPD_TEACHER_MIN_NEW_TOKENS", "16"))
TEACHER_SCORE_CHUNK_TOKENS = int(os.environ.get("OPD_TEACHER_SCORE_CHUNK_TOKENS", "1024"))
TEACHER_SCORE_CONTEXT_MARGIN = int(os.environ.get("OPD_TEACHER_SCORE_CONTEXT_MARGIN", "16"))
TEACHER_SCORE_POSITION_TOPK = os.environ.get("OPD_TEACHER_SCORE_POSITION_TOPK", "1") != "0"
TEACHER_MODEL_ROOT = os.environ.get("OPD_TEACHER_MODEL_ROOT", "/mnt/raid0_ssd/sheng/final_ckpts")
EDA_SERVER = os.environ.get("OPD_EDA_SERVER", "local")
EDA_REPO_DIR = os.environ.get("OPD_EDA_REPO_DIR", "/workspace/llm4cov_eda")
EDA_TIMEOUT = int(os.environ.get("OPD_EDA_TIMEOUT", "30"))
COMPACT_SCORE_RESULT = os.environ.get("OPD_COMPACT_SCORE_RESULT", "1") != "0"
AUDIT_RESULT_GZIP = os.environ.get("OPD_AUDIT_RESULT_GZIP", "1") != "0"
SCORE_TENSOR_DTYPE = os.environ.get("OPD_SCORE_TENSOR_DTYPE", "float32")
MOCK = os.environ.get("OPD_MOCK", "0") == "1"

_JOB_SEM = threading.Semaphore(MAX_CONCURRENT_JOBS)
_LOCK = threading.Lock()
_INFLIGHT: set[str] = set()
QUEUE_BACKEND: Any | None = None

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
        self._release_claim(job_id)


def build_queue_backend() -> Any:
    if QUEUE_TRANSPORT == "local":
        return LocalOpdQueue()
    if QUEUE_TRANSPORT == "sftp":
        return SftpOpdQueue()
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
            "generated_token_ids": [],
            "generated_token_logprobs": [],
            "finish_reason": skip_reason,
            "teacher_generation_status": skip_reason,
            "teacher_generation_timing": {
                "total_seconds": time.monotonic() - started,
                "request_seconds": 0.0,
            },
        }
    payload = {"text": prompt, "sampling_params": effective_sampling_params, "return_logprob": True}
    try:
        request_started = time.monotonic()
        output = post_json(url, payload, timeout=TEACHER_TIMEOUT)
        request_seconds = time.monotonic() - request_started
    except Exception as exc:
        return {
            **base,
            "assistant_response": "",
            "generated_token_ids": [],
            "generated_token_logprobs": [],
            "finish_reason": "teacher_generate_error",
            "teacher_generation_status": "teacher_generate_error",
            "teacher_generation_error": str(exc),
            "teacher_generation_timing": {
                "total_seconds": time.monotonic() - started,
                "request_seconds": time.monotonic() - request_started,
            },
        }
    response = str(output.get("text") or output.get("response") or output.get("output") or "")
    meta = output.get("meta_info") if isinstance(output.get("meta_info"), dict) else {}
    token_ids, logprobs = extract_token_logprobs(meta)
    return {
        **base,
        "assistant_response": response,
        "generated_token_ids": token_ids,
        "generated_token_logprobs": logprobs,
        "finish_reason": meta.get("finish_reason"),
        "teacher_generation_status": "success",
        "teacher_generation_timing": {
            "total_seconds": time.monotonic() - started,
            "request_seconds": request_seconds,
            "response_chars": len(response),
            "generated_token_count": len(token_ids),
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

    try:
        for offset in range(0, response_len, chunk_size):
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
            started = time.monotonic()
            output = post_json(teacher_url(name, cfg), payload, timeout=TEACHER_TIMEOUT)
            meta = output.get("meta_info") if isinstance(output.get("meta_info"), dict) else {}
            teacher_log_probs.extend(extract_input_logprobs(meta, chunk_len))
            chunk_topk, chunk_masks = _extract_requested_topk_logprobs(
                output,
                chunk_len,
                chunk_requested_topk,
            )
            teacher_topk_parts.append(chunk_topk)
            teacher_mask_parts.append(chunk_masks)
            timings.append(
                {
                    "offset": offset,
                    "chunk_response_token_count": chunk_len,
                    "request_seconds": time.monotonic() - started,
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
            )
    except Exception as exc:
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
                "teacher_context_length": score_context_len,
                "max_request_input_token_count": max(
                    (int(t.get("input_token_count") or 0) for t in timings),
                    default=0,
                ),
                "chunks": timings,
            },
        }

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
        "teacher_log_probs": teacher_log_probs if requested_topk is None else [],
        "topk_token_ids_sha256": requested_topk_sha256,
        "response_token_count": response_len,
        "num_teacher_log_probs": len(teacher_log_probs),
        "teacher_score_timing": {
            "request_seconds": time.monotonic() - total_started,
            "full_input_token_count": len(input_ids),
            "prompt_token_count": prompt_len,
            "response_token_count": response_len,
            "chunk_size": chunk_size,
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


def score_student(job: Path, entry: dict[str, Any], context: SimpleNamespace, want_detail: bool) -> dict[str, Any]:
    sid = str(entry.get("id"))
    base = job / str(entry.get("path", f"student/{sid}"))
    meta = read_json(base / "meta.json") if (base / "meta.json").exists() else {}
    response = (base / "assistant_response.txt").read_text(encoding="utf-8")
    filename = meta.get("filename") or extract_filename(response) or f"{sid}.sv"
    if (base / "testbench.sv").exists():
        body = (base / "testbench.sv").read_text(encoding="utf-8")
    else:
        body = extract_verilog(response)
    scored = run_eda(context, filename, body, want_detail)
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
        "teacher_scores": [],
        **scored,
    }


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
    np.savez_compressed(
        path,
        teacher_topk_log_probs=log_probs_arr,
        teacher_topk_logprob_masks=masks_arr,
    )
    return {
        "format": "npz_v1",
        "file": rel,
        "sha256": _sha256_file(path),
        "shape": [int(log_probs_arr.shape[0]), int(log_probs_arr.shape[1])],
        "log_probs_dtype": str(log_probs_arr.dtype),
        "masks_dtype": str(masks_arr.dtype),
        "aligned_to": "student_topk_token_ids_order",
        "topk_token_ids_sha256": token_ids_sha256 or "",
    }


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
            score["teacher_topk_sidecar"] = _write_teacher_topk_sidecar(
                staging,
                student_id=str(entry.get("id") or score.get("student_id") or "student"),
                teacher=str(score.get("teacher") or "teacher"),
                score_idx=score_idx,
                log_probs=log_probs,
                masks=masks,
                token_ids_sha256=str(score.get("topk_token_ids_sha256") or ""),
            )


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


def process(job: Path) -> None:
    job_id = job.name
    started_wall = time.time()
    started = time.monotonic()
    timing: dict[str, Any] = {}
    with _JOB_SEM:
        try:
            phase_started = time.monotonic()
            manifest = read_json(job / "manifest.json")
            if manifest.get("version") != SCHEMA_VERSION:
                raise ValueError(f"bad schema version: {manifest.get('version')}")
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
                phase_started = time.monotonic()
                with ThreadPoolExecutor(max_workers=max(1, min(MAX_JOB_WORKERS, len(teacher_jobs)))) as pool:
                    futures = [
                        pool.submit(generate_teacher_rollout, name, slot, prompt, sampling_params, teacher_cfg)
                        for name, slot in teacher_jobs
                    ]
                    generated = [future.result() for future in as_completed(futures)]
                timing["teacher_generate_seconds"] = time.monotonic() - phase_started
                timing["teacher_generate_max_seconds"] = max(
                    (
                        float((entry.get("teacher_generation_timing") or {}).get("total_seconds", 0.0) or 0.0)
                        for entry in generated
                        if isinstance(entry, dict)
                    ),
                    default=0.0,
                )
                phase_started = time.monotonic()
                with ThreadPoolExecutor(max_workers=max(1, min(MAX_JOB_WORKERS, len(generated)))) as pool:
                    teacher_entries.extend(pool.map(lambda e: score_teacher(e, context, want_detail), generated))
                timing["teacher_eda_seconds"] = time.monotonic() - phase_started
                timing["teacher_eda_max_seconds"] = max(
                    (
                        float((entry.get("eda_timing") or {}).get("total_seconds", 0.0) or 0.0)
                        for entry in teacher_entries
                        if isinstance(entry, dict)
                    ),
                    default=0.0,
                )
            else:
                timing.setdefault("teacher_generate_seconds", 0.0)
                timing.setdefault("teacher_generate_max_seconds", 0.0)
                timing.setdefault("teacher_eda_seconds", 0.0)
                timing.setdefault("teacher_eda_max_seconds", 0.0)

            phase_started = time.monotonic()
            sel = selection(student_entries, teacher_entries)
            best_teacher_entry = max(teacher_entries, key=lambda e: float(e.get("reward", 0.0)), default=None)
            timing["selection_seconds"] = time.monotonic() - phase_started
            timing["teacher_score_student_seconds"] = 0.0
            timing["teacher_score_student_max_seconds"] = 0.0
            if teacher_request.get("score_student_rollouts") and best_teacher_entry is not None:
                best_teacher_name = str(best_teacher_entry.get("teacher") or "")
                if best_teacher_name:
                    phase_started = time.monotonic()
                    with ThreadPoolExecutor(max_workers=max(1, min(MAX_JOB_WORKERS, len(student_entries) or 1))) as pool:
                        topk_k = int(teacher_request.get("student_topk_k") or 0)
                        scores = list(pool.map(lambda e: score_teacher_on_student(best_teacher_name, e, teacher_cfg, topk_k), student_entries))
                    timing["teacher_score_student_seconds"] = time.monotonic() - phase_started
                    timing["teacher_score_student_max_seconds"] = max(
                        (
                            float((score.get("teacher_score_timing") or {}).get("request_seconds", 0.0) or 0.0)
                            for score in scores
                            if isinstance(score, dict)
                        ),
                        default=0.0,
                    )
                    for entry, score in zip(student_entries, scores, strict=False):
                        entry.setdefault("teacher_scores", []).append(score)
                        if score.get("status") in {"success", "topk_success", "partial_topk"}:
                            entry["teacher_logprob_teacher"] = best_teacher_name

                    # Training keeps these tensors locally and only needs the
                    # teacher score payload from the relay.  Dropping them here
                    # keeps Paladin audit data focused on testbenches/EDA while
                    # avoiding a large SFTP download back to Brev.
                    for entry in student_entries:
                        for key in (
                            "input_token_ids",
                            "response_token_ids",
                            "rollout_log_probs",
                            "topk_token_ids",
                            "topk_student_log_probs",
                        ):
                            entry.pop(key, None)

            timing["worker_total_before_publish_seconds"] = time.monotonic() - started
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
            log(
                "OPD_WORKER_TIMING",
                job_id,
                f"load={timing.get('load_request_seconds', 0.0):.3f}",
                f"student_eda={timing.get('student_eda_seconds', 0.0):.3f}",
                f"teacher_generate={timing.get('teacher_generate_seconds', 0.0):.3f}",
                f"teacher_eda={timing.get('teacher_eda_seconds', 0.0):.3f}",
                f"teacher_score={timing.get('teacher_score_student_seconds', 0.0):.3f}",
                f"publish={publish_seconds:.3f}",
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
        f"max_jobs={MAX_CONCURRENT_JOBS} queue={QUEUE_TRANSPORT} worker_id={WORKER_ID}"
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
