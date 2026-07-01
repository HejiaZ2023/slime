#!/usr/bin/env python3
"""OPD teacher/EDA relay worker for llm4cov.

Runs on paladin next to the existing xfer watcher.  Slime publishes jobs under
incoming/<namespace>/<job_id>; this worker generates teacher rollouts, evaluates
student and teacher testbenches through the existing EDA xfer watcher, and
publishes results under results/<namespace>/<job_id>.
"""

from __future__ import annotations

import contextlib
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

XFER = Path(os.environ.get("OPD_XFER_DIR", "/mnt/raid0_ssd/eda/xfer"))
NAMESPACE = os.environ.get("OPD_NAMESPACE", "opd")
IN = XFER / "incoming" / NAMESPACE
RES = XFER / "results" / NAMESPACE
WORK = XFER / ".work" / NAMESPACE
STATE = XFER / ".state" / NAMESPACE
POLL_SEC = float(os.environ.get("OPD_POLL_SEC", "2"))
JOB_TTL = int(os.environ.get("OPD_JOB_TTL", str(6 * 3600)))
MAX_CONCURRENT_JOBS = int(os.environ.get("OPD_MAX_CONCURRENT_JOBS", "4"))
MAX_JOB_WORKERS = int(os.environ.get("OPD_MAX_JOB_WORKERS", "4"))
TEACHER_TIMEOUT = float(os.environ.get("OPD_TEACHER_TIMEOUT", "900"))
EDA_SERVER = os.environ.get("OPD_EDA_SERVER", "local")
EDA_REPO_DIR = os.environ.get("OPD_EDA_REPO_DIR", "/workspace/llm4cov_eda")
EDA_TIMEOUT = int(os.environ.get("OPD_EDA_TIMEOUT", "30"))
MOCK = os.environ.get("OPD_MOCK", "0") == "1"

_JOB_SEM = threading.Semaphore(MAX_CONCURRENT_JOBS)
_LOCK = threading.Lock()
_INFLIGHT: set[str] = set()

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


def run_eda(context: SimpleNamespace, filename: str | None, body: str | None, want_detail: bool) -> dict[str, Any]:
    filename = safe_filename(filename)
    if not body:
        eda_log = {"status": "parse_failed", "filename": None, "has_coverage": False}
        return {"reward": 0.0, "eda_feedback": "- status: failed (could not extract a valid testbench)", "eda_log": eda_log}
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
        return {"reward": 1.5, "eda_feedback": "- status: success\n- stage: mock\n- coverage: 50%", "eda_log": eda_log}
    tb_file = SimpleNamespace(name=filename, content=body)
    try:
        raw = submit_cov_job(
            EDA_SERVER,
            EDA_REPO_DIR,
            context,
            tb_file,
            skip_detail=not want_detail,
            timeout=EDA_TIMEOUT,
        )
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
        return {"reward": reward, "eda_feedback": format_feedback(raw, context) if want_detail else None, "eda_log": eda_log}
    except Exception as exc:
        eda_log = {"status": "exception", "filename": filename, "has_coverage": False, "exc": str(exc)}
        return {"reward": 0.0, "eda_feedback": f"- status: failed (EDA exception: {exc})", "eda_log": eda_log}


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
    url = teacher_url(name, cfg)
    payload = {"text": prompt, "sampling_params": sampling_params, "return_logprob": True}
    output = post_json(url, payload, timeout=TEACHER_TIMEOUT)
    response = str(output.get("text") or output.get("response") or output.get("output") or "")
    meta = output.get("meta_info") if isinstance(output.get("meta_info"), dict) else {}
    token_ids, logprobs = extract_token_logprobs(meta)
    return {
        "id": f"{name}_t{slot:03d}",
        "teacher": name,
        "assistant_response": response,
        "generated_token_ids": token_ids,
        "generated_token_logprobs": logprobs,
        "finish_reason": meta.get("finish_reason"),
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


def score_teacher_on_student(name: str, entry: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
    sid = str(entry.get("id"))
    input_ids = entry.get("input_token_ids")
    response_len = int(entry.get("response_token_count") or 0)
    if not isinstance(input_ids, list) or response_len <= 0:
        return {
            "student_id": sid,
            "teacher": name,
            "status": "missing_student_tokens",
            "teacher_log_probs": [],
        }
    payload = {
        "input_ids": [int(x) for x in input_ids],
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": 0,
            "skip_special_tokens": False,
        },
        "return_logprob": True,
        "logprob_start_len": 0,
    }
    try:
        output = post_json(teacher_url(name, cfg), payload, timeout=TEACHER_TIMEOUT)
    except Exception as exc:
        return {
            "student_id": sid,
            "teacher": name,
            "status": "teacher_score_error",
            "error": str(exc),
            "teacher_log_probs": [],
            "response_token_count": response_len,
            "num_teacher_log_probs": 0,
        }
    meta = output.get("meta_info") if isinstance(output.get("meta_info"), dict) else {}
    teacher_log_probs = extract_input_logprobs(meta, response_len)
    status = "success" if len(teacher_log_probs) == response_len else "length_mismatch"
    return {
        "student_id": sid,
        "teacher": name,
        "status": status,
        "teacher_log_probs": teacher_log_probs,
        "response_token_count": response_len,
        "num_teacher_log_probs": len(teacher_log_probs),
    }


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
    return {
        "id": sid,
        "sample_index": meta.get("sample_index"),
        "assistant_response_file": f"{base.relative_to(job).as_posix()}/assistant_response.txt",
        "input_token_ids": meta.get("input_token_ids") or meta.get("tokens") or [],
        "response_token_ids": meta.get("response_token_ids") or [],
        "response_token_count": int(meta.get("response_token_count") or len(meta.get("response_token_ids") or [])),
        "rollout_log_probs": meta.get("rollout_log_probs") or [],
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


def publish(job_id: str, obj: dict[str, Any]) -> None:
    staging = RES / f"{job_id}.staging"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    write_json(staging / "result.json", obj)
    final = RES / job_id
    if final.exists():
        shutil.rmtree(final)
    os.rename(staging, final)
    (final / ".done").write_text("", encoding="utf-8")


def process(job: Path) -> None:
    job_id = job.name
    started = time.time()
    with _JOB_SEM:
        try:
            manifest = read_json(job / "manifest.json")
            if manifest.get("version") != SCHEMA_VERSION:
                raise ValueError(f"bad schema version: {manifest.get('version')}")
            prompt = (job / manifest.get("prompt_file", "prompt.txt")).read_text(encoding="utf-8")
            context = context_from_dict(read_json(job / manifest.get("context_file", "context.json")))
            want_detail = bool((manifest.get("eda") or {}).get("want_detail", False))
            sampling_params = manifest.get("sampling_params") if isinstance(manifest.get("sampling_params"), dict) else {}
            teacher_cfg = load_teacher_config()

            student_specs = manifest.get("student_rollouts") if isinstance(manifest.get("student_rollouts"), list) else []
            with ThreadPoolExecutor(max_workers=max(1, min(MAX_JOB_WORKERS, len(student_specs) or 1))) as pool:
                student_entries = list(pool.map(lambda e: score_student(job, e, context, want_detail), student_specs))

            teacher_specs = manifest.get("teachers") if isinstance(manifest.get("teachers"), list) else []
            teacher_jobs: list[tuple[str, int]] = []
            for spec in teacher_specs:
                if not isinstance(spec, dict):
                    continue
                name = str(spec.get("name"))
                for i in range(int(spec.get("n", 1) or 0)):
                    teacher_jobs.append((name, i))
            teacher_entries: list[dict[str, Any]] = []
            if teacher_jobs:
                with ThreadPoolExecutor(max_workers=max(1, min(MAX_JOB_WORKERS, len(teacher_jobs)))) as pool:
                    futures = [
                        pool.submit(generate_teacher_rollout, name, slot, prompt, sampling_params, teacher_cfg)
                        for name, slot in teacher_jobs
                    ]
                    generated = [future.result() for future in as_completed(futures)]
                with ThreadPoolExecutor(max_workers=max(1, min(MAX_JOB_WORKERS, len(generated)))) as pool:
                    teacher_entries = list(pool.map(lambda e: score_teacher(e, context, want_detail), generated))

            sel = selection(student_entries, teacher_entries)
            best_teacher_entry = max(teacher_entries, key=lambda e: float(e.get("reward", 0.0)), default=None)
            teacher_request = manifest.get("teacher_request") if isinstance(manifest.get("teacher_request"), dict) else {}
            if teacher_request.get("score_student_rollouts") and best_teacher_entry is not None:
                best_teacher_name = str(best_teacher_entry.get("teacher") or "")
                if best_teacher_name:
                    with ThreadPoolExecutor(max_workers=max(1, min(MAX_JOB_WORKERS, len(student_entries) or 1))) as pool:
                        scores = list(pool.map(lambda e: score_teacher_on_student(best_teacher_name, e, teacher_cfg), student_entries))
                    for entry, score in zip(student_entries, scores, strict=False):
                        entry.setdefault("teacher_scores", []).append(score)
                        if score.get("status") == "success":
                            entry["teacher_log_probs"] = score.get("teacher_log_probs") or []
                            entry["teacher_logprob_teacher"] = best_teacher_name

            result = {
                "version": SCHEMA_VERSION,
                "status": "success",
                "job_id": job_id,
                "dataset_id": manifest.get("dataset_id"),
                "rollout_id": manifest.get("rollout_id"),
                "round_idx": manifest.get("round_idx"),
                "student_rollouts": student_entries,
                "teacher_rollouts": teacher_entries,
                "selection": sel,
                "elapsed_s": time.time() - started,
            }
            publish(job_id, result)
            log("done", job_id, f"students={len(student_entries)}", f"teachers={len(teacher_entries)}")
        except Exception as exc:
            result = {
                "version": SCHEMA_VERSION,
                "status": "failed",
                "job_id": job_id,
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "elapsed_s": time.time() - started,
            }
            publish(job_id, result)
            log("FAIL", job_id, exc)
        finally:
            STATE.mkdir(parents=True, exist_ok=True)
            (STATE / f"{job_id}.processed").write_text(str(time.time()), encoding="utf-8")
            with _LOCK:
                _INFLIGHT.discard(job_id)


def gc() -> None:
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
    for d in (IN, RES, WORK, STATE):
        d.mkdir(parents=True, exist_ok=True)
    log(f"opd worker start XFER={XFER} namespace={NAMESPACE} mock={MOCK} max_jobs={MAX_CONCURRENT_JOBS}")
    last_gc = 0.0
    while True:
        now = time.time()
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
            threading.Thread(target=process, args=(job,), daemon=True).start()
        if now - last_gc > 300:
            gc()
            last_gc = now
        time.sleep(POLL_SEC)


if __name__ == "__main__":
    main()
