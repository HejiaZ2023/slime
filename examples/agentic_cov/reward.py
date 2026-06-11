"""Remote EDA-based reward evaluation for llm4cov multi-round rollouts.

Reward = 0 when the generated testbench cannot be extracted / fails xrun.
Reward = 1 + Overall Average coverage when the job succeeds.

The coverage job is slow (SSH + remote simulation), so we run it in a worker
thread via asyncio.to_thread to avoid blocking the event loop. Each sample
is keyed by a unique DataFile name to avoid collisions on the remote host.

compute_reward stores the formatted EDA feedback string in
sample.metadata["eda_feedback"] so rollout.py can pass it verbatim to
the model in the next round's tool-feedback turn.
"""

from __future__ import annotations

import asyncio
import logging
# import re   # only used by the old _format_eda_feedback implementation (see below)
import uuid
from typing import Any

logger = logging.getLogger(__name__)

_DEFAULT_TB_FILENAME = "tb_generated.sv"

# ── COMMENTED OUT: old _format_eda_feedback helpers ──────────────────────────
# _ERR_SNIPPET_LEN = 1200   # max chars of err_msg to relay to the model
# _DETAIL_BUDGET = 2000     # max chars of parsed detail to append on success
#
# # Section headers in xrun coverage report (case-insensitive match)
# _SECTION_RE = re.compile(
#     r"Uncovered (Block|Expression|Fsm) Detail Report",
#     re.IGNORECASE,
# )
# _NUM_UNCOV_RE = re.compile(
#     r"Number of uncovered [^:]+:\s*(\d+)\s*of\s*(\d+)",
#     re.IGNORECASE,
# )
#
#
# def _parse_detail_for_feedback(detail: str) -> str:
#     """Extract Block / Expression / FSM uncovered info from xrun detail report.
#
#     Toggle and Assertion sections are skipped — too low-level to be actionable.
#     Returns an empty string when nothing interesting is found.
#     """
#     if not detail:
#         return ""
#
#     # Split on section boundaries so we can handle each independently.
#     # We only want Block, Expression, and Fsm sections.
#     sections: list[tuple[str, str]] = []  # (section_type, section_text)
#     parts = re.split(r"(Uncovered (?:Block|Expression|Fsm|Toggle|Assertion|CoverGroup) Detail Report[^\n]*)", detail)
#     i = 1
#     while i < len(parts) - 1:
#         header = parts[i]
#         body = parts[i + 1]
#         m = re.search(r"Uncovered (Block|Expression|Fsm)", header, re.IGNORECASE)
#         if m:
#             sections.append((m.group(1).capitalize(), body))
#         i += 2
#
#     lines_out: list[str] = []
#     budget = _DETAIL_BUDGET
#
#     for sec_type, body in sections:
#         if budget <= 0:
#             break
#
#         # Extract "N of M" summary line
#         nm = _NUM_UNCOV_RE.search(body)
#         if not nm:
#             continue
#         n_uncov, n_total = int(nm.group(1)), int(nm.group(2))
#         if n_uncov == 0:
#             continue  # fully covered section — skip
#
#         header_line = f"  {sec_type}: {n_uncov} of {n_total} uncovered"
#         lines_out.append(header_line)
#         budget -= len(header_line) + 1
#
#         # Find the table (starts after the dashed separator line)
#         table_start = body.find("---\n", nm.end())
#         if table_start == -1:
#             table_start = body.find("----", nm.end())
#         if table_start == -1:
#             continue
#         # Skip the separator line itself
#         table_start = body.find("\n", table_start) + 1
#
#         # Pull up to 8 non-empty table rows
#         rows_added = 0
#         for raw in body[table_start:].splitlines():
#             if rows_added >= 8:
#                 break
#             stripped = raw.strip()
#             if not stripped:
#                 continue
#             # Stop at the next section or end-of-report markers
#             if stripped.startswith("Uncovered") or stripped.startswith("*"):
#                 break
#             row_line = "    " + stripped[:120]
#             if budget - len(row_line) - 1 <= 0:
#                 lines_out.append("    ... (truncated)")
#                 budget = 0
#                 break
#             lines_out.append(row_line)
#             budget -= len(row_line) + 1
#             rows_added += 1
#
#     return "\n".join(lines_out)
#
#
# def _format_eda_feedback(result: dict[str, Any], context_id: str) -> str:
#     """Convert raw EDA result dict into a human-readable feedback string for the model."""
#     status = result.get("status", "xrun_failed")
#     err_msg = (result.get("err_msg") or "").strip()
#
#     if status == "xrun_failed":
#         snippet = err_msg[:_ERR_SNIPPET_LEN] if err_msg else "(no error message captured)"
#         if len(err_msg) > _ERR_SNIPPET_LEN:
#             snippet += f"\n... (truncated, {len(err_msg)} chars total)"
#         return (
#             "- status: failed (xrun/xmsim did not complete)\n"
#             f"- error output:\n{snippet}"
#         )
#
#     if status != "success":
#         snippet = err_msg[:_ERR_SNIPPET_LEN] if err_msg else f"status={status}"
#         return (
#             f"- status: failed ({status})\n"
#             f"- error output:\n{snippet}"
#         )
#
#     # success — build coverage summary + uncovered detail
#     cov_info = result.get("cov_info", {})
#     summary: list[dict] = cov_info.get("summary", []) if isinstance(cov_info, dict) else []
#
#     # find top-level module entry (level == 0)
#     dut_entry = next((m for m in summary if m.get("level") == 0), None)
#     if dut_entry is None:
#         return "- status: success\n- coverage data unavailable"
#
#     skip_keys = {"name", "level"}
#     metrics_lines = []
#     for k, v in dut_entry.items():
#         if k in skip_keys:
#             continue
#         if isinstance(v, float):
#             metrics_lines.append(f"  {k}: {v * 100:.2f}%")
#         elif v is not None:
#             metrics_lines.append(f"  {k}: {v}")
#
#     coverage_block = "\n".join(metrics_lines) if metrics_lines else "  (no metrics)"
#     parts = [
#         "- status: success",
#         f"- coverage breakdown (module: {dut_entry.get('name', context_id)}):",
#         coverage_block,
#     ]
#
#     # Append uncovered detail (Block / Expression / FSM)
#     detail_str = cov_info.get("detail", "") if isinstance(cov_info, dict) else ""
#     parsed_detail = _parse_detail_for_feedback(detail_str)
#     if parsed_detail:
#         parts.append("- uncovered detail (top issues):")
#         parts.append(parsed_detail)
#
#     return "\n".join(parts)
# ─────────────────────────────────────────────────────────────────────────────


def _format_eda_feedback(result: dict[str, Any], context_id: str) -> str:
    """Matches _summarize_eda_result format in batch_query_eval.py."""
    status = str(result.get("status", "unknown"))
    err_msg = str(result.get("err_msg") or "")
    if status == "xrun_failed":
        return f"- status: {status}\n- stage: xrun\n- log: {err_msg}"
    if status != "success":
        return f"- status: {status}\n- stage: imc\n- log: {err_msg}"
    cov_info = result.get("cov_info", {})
    detail = str(cov_info.get("detail", "")) if isinstance(cov_info, dict) else ""
    return f"- status: {status}\n- stage: success\n- coverage: {detail}"


# ── New structured tool-feedback from cov_info["uncovered"] (used iff --use-uncovered-log).
# Prompt rendering only; the underlying bin_ids stay per-bit for reward. Robust by design:
# never raises — non-success / missing-uncovered fall back to a minimal line or raw formatter.
def _agg_toggle_signals(signals: list[str]) -> list[str]:
    """Aggregate bit signals (sig[15], sig[14], ...) into ranges (sig[15:0])."""
    import re
    from collections import defaultdict
    bits: dict[str, list[int]] = defaultdict(list)
    order: list[str] = []
    plain: list[str] = []
    for s in signals:
        m = re.match(r"^(.*)\[(\d+)\]$", s)
        if m:
            base = m.group(1)
            if base not in bits:
                order.append(base)
            bits[base].append(int(m.group(2)))
        else:
            plain.append(s)
    out: list[str] = []
    for base in order:
        bl = sorted(set(bits[base]), reverse=True)
        parts, i = [], 0
        while i < len(bl):
            j = i
            while j + 1 < len(bl) and bl[j + 1] == bl[j] - 1:
                j += 1
            parts.append(f"[{bl[i]}]" if i == j else f"[{bl[i]}:{bl[j]}]")
            i = j + 1
        out.append(base + "".join(parts))
    out.extend(plain)
    return out


def _render_uncov_type(typ: str, items: list[dict]) -> list[str]:
    if typ == "block":
        return [f"L{it.get('line')} `{it.get('source', '')}`" for it in items]
    if typ == "expression":
        return [f"L{it.get('line')} (expr {it.get('index')})" for it in items]
    if typ == "fsm":
        states = [it.get("state") for it in items if it.get("kind") == "state"]
        trans = [f"{it.get('from')}->{it.get('to')}" for it in items if it.get("kind") == "transition"]
        out = []
        if states:
            out.append("states: " + ", ".join(states))
        if trans:
            out.append("transitions: " + ", ".join(trans))
        return out
    if typ == "toggle":
        return _agg_toggle_signals([it.get("signal", "") for it in items])
    return [str(it.get("id", it)) for it in items]  # assertion/covergroup/unknown fallback


def _format_eda_feedback_uncovered(result: dict[str, Any], context_id: str) -> str:
    status = str(result.get("status", "unknown"))
    if status != "success":
        return _format_eda_feedback(result, context_id)  # reuse raw status lines
    cov_info = result.get("cov_info", {})
    cov_info = cov_info if isinstance(cov_info, dict) else {}
    summ = cov_info.get("summary") or []
    overall = None
    if summ and isinstance(summ[0], dict):
        ov = summ[0].get("Overall Average")
        if isinstance(ov, (int, float)):
            overall = f"{ov * 100:.1f}%"
    unc = cov_info.get("uncovered")
    head = ["- status: success"]
    if overall is not None:
        head.append(f"- overall coverage: {overall}")
    if not isinstance(unc, dict) or not unc.get("by_type"):
        head.append("- uncovered: none")
        return "\n".join(head)
    by_type = unc["by_type"]
    counts = unc.get("counts", {}) if isinstance(unc.get("counts"), dict) else {}
    total = unc.get("total_uncovered", sum(len(v) for v in by_type.values()))
    bits = ", ".join(f"{t} {counts.get(t, len(items))}" for t, items in by_type.items())
    lines = head + [f"- uncovered ({total}): {bits}"]
    for typ, items in by_type.items():
        try:
            rendered = _render_uncov_type(typ, items)
        except Exception:
            rendered = [str(it.get("id", it)) for it in items]
        lines.append(f"  {typ}:")
        lines.extend(f"    {r}" for r in rendered)
    return "\n".join(lines)


def _parse_testbench(response: str) -> tuple[str | None, str | None]:
    """Return (filename, verilog_body) parsed from an LLM completion."""
    from llm4cov.llm_query.parse import extract_filename_from_text, extract_verilog_content

    body = extract_verilog_content(response)
    if body is None:
        return None, None
    filename = extract_filename_from_text(response) or _DEFAULT_TB_FILENAME
    stem, _, ext = filename.rpartition(".")
    if not stem:
        stem, ext = filename, "sv"
    unique = f"{stem}_{uuid.uuid4().hex[:8]}.{ext}"
    return unique, body


def _rehydrate_context(context_dict: dict[str, Any]):
    from llm4cov.datasets.types import LlmGenTbContext
    return LlmGenTbContext(**context_dict)


def _compute_reward_sync(
    args: Any, context_dict: dict[str, Any], response: str
) -> tuple[float, str | None, dict]:
    """Synchronous reward path.

    Returns (reward_float, eda_feedback_or_None, eda_log_dict).

    * eda_feedback — formatted for the model tool-feedback turn; only
      populated when args.eda_log_feedback is True.
    * eda_log_dict — always populated; stored in
      sample.metadata["_eda_log"] so the eval logging code in
      rollout.py can write complete EDA results without depending on the
      --eda-log-feedback flag. Does not contain coverage detail when
      skip_detail=True (that is intentional — detail is not needed for
      in-training eval monitoring).
    """
    # want_detail may be overridden by caller (eval vs train context);
    # falls back to eda_log_feedback_train for backwards compat.
    want_detail: bool = getattr(args, "_want_detail_override",
                               getattr(args, "eda_log_feedback_train",
                                       getattr(args, "eda_log_feedback", False)))

    filename, body = _parse_testbench(response)
    if filename is None or body is None:
        err_fb = "- status: failed (could not extract a valid testbench from the response)"
        _eda_log: dict = {"status": "parse_failed", "filename": None}
        return 0.0, err_fb if want_detail else None, _eda_log

    from llm4cov.datasets.eval import eval_cov_result_against_expectations
    from llm4cov.datasets.types import DataFile
    from llm4cov.eda_client.remote_exec import run_remote_cov_job_pipeline

    context = _rehydrate_context(context_dict)
    tb_file = DataFile(name=filename, content=body)

    try:
        result = run_remote_cov_job_pipeline(
            server=args.eda_server,
            eda_repo_dir=args.eda_repo_dir,
            context=context,
            tb_file=tb_file,
            skip_detail=not want_detail,
            timeout=getattr(args, "eda_stage_timeout", 30),
        )
    except Exception as exc:
        logger.warning("Remote EDA job raised exception for %s: %s", context.id, exc)
        exc_fb = f"- status: failed (remote EDA job exception: {exc})"
        _eda_log = {"status": "exception", "exc": str(exc), "filename": filename}
        return 0.0, exc_fb if want_detail else None, _eda_log

    if want_detail:
        if getattr(args, "use_uncovered_log", False):
            eda_feedback = _format_eda_feedback_uncovered(result, context.id)
        else:
            eda_feedback = _format_eda_feedback(result, context.id)
    else:
        eda_feedback = None

    cov_result = eval_cov_result_against_expectations(context, result)
    _eda_log = {
        "status": result.get("status", "xrun_failed"),
        "filename": filename,
        "overall_coverage": float(cov_result.overall_coverage),
        "is_pass_xrun": bool(cov_result.is_pass_xrun),
        "is_pass_targets": bool(cov_result.is_pass_targets),
        "has_coverage": bool(cov_result.has_coverage),
        "err_msg": (result.get("err_msg") or "") if result.get("status") != "success" else "",
        # uncovered bin_ids (unaggregated) for group-level diversity reward; only stored
        # under --use-uncovered-reward (else [] to avoid bloating metadata).
        "uncovered_bin_ids": (
            ((result.get("cov_info") or {}).get("uncovered") or {}).get("bin_ids", [])
            if getattr(args, "use_uncovered_reward", False) else []
        ),
    }
    if not cov_result.has_coverage:
        return 0.0, eda_feedback, _eda_log
    cov_score = float(cov_result.overall_coverage)
    # coverage score rescaled to [0.5,1] ONLY under --use-uncovered-reward (so it
    # composes with the diversity bonus); otherwise unchanged 1+cov (legacy behaviour).
    if getattr(args, "use_uncovered_reward", False):
        return (1.0 + cov_score) / 2.0, eda_feedback, _eda_log
    return 1.0 + cov_score, eda_feedback, _eda_log


async def compute_reward(args: Any, sample: Any, *, want_detail: bool | None = None) -> float:
    """Async reward entry point.

    sample.metadata['llm4cov_context'] must exist.
    Sets sample.metadata['eda_feedback'] with the formatted EDA output so
    rollout.py can relay it to the model in the next round's tool-feedback turn.

    want_detail: override whether to fetch full EDA detail. When None, the
    value is derived from args.eda_log_feedback_train / eda_log_feedback_eval
    (set by --eda-log-feedback-train / --eda-log-feedback-eval).
    """
    context_dict = sample.metadata.get("llm4cov_context")
    if context_dict is None:
        logger.warning("Sample %s is missing llm4cov_context metadata", sample.index)
        return 0.0

    import types
    _args = args
    if want_detail is not None:
        _args = types.SimpleNamespace(**vars(args))
        _args._want_detail_override = want_detail

    reward, eda_feedback, eda_log = await asyncio.to_thread(
        _compute_reward_sync, _args, context_dict, sample.response
    )
    sample.metadata["_eda_log"] = eda_log
    if eda_feedback is not None:
        sample.metadata["eda_feedback"] = eda_feedback
    logger.debug(
        "Sample %s reward=%.4f  status=%s  feedback=%s",
        sample.index,
        reward,
        eda_log.get("status", "?"),
        (eda_feedback[:120].replace("\n", " ")) if eda_feedback else "None",
    )
    return reward
