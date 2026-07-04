"""Multi-round agentic GRPO rollout for llm4cov testbench generation.

Two entry points — slime wires each independently:

* ``generate_rollout``  → ``--rollout-function-path`` (training). Pulls prompts
  from the ``LlmCovDataSource`` instance that slime already built from
  ``--llm4cov-dataset-name`` / ``--llm4cov-dataset-split``.
* ``eval_rollout``      → ``--eval-function-path`` (evaluation). Loads its own
  llm4cov dataset directly from ``--llm4cov-eval-dataset-name`` /
  ``--llm4cov-eval-dataset-split`` and iterates over *all* eval prompts —
  ``rollout_batch_size`` does not constrain eval.

Both paths share the same K-round core:

* For each prompt we sample a group of ``n_samples_per_prompt`` completions,
  score each with the llm4cov EDA reward, and pick one winner:
    - training: the *worst* scoring sample drives the next round's prompt
      (focus training on the branch the model is most wrong about)
    - evaluation: the *best* scoring sample drives the next round.
* The selected sample's completion is stripped of ``<think>…</think>`` blocks
  and concatenated onto the prompt along with a tool-feedback message. That
  string becomes the input for round ``k+1``. The round-``k`` samples are
  **not** mutated — they are emitted as their own GRPO group with the raw
  completion (think tokens included) so reasoning is still trained.
* Every round gets its own fresh ``group_index`` so the per-group reward
  normalisation in ``ray.rollout._post_process_rewards`` baselines each round
  independently. Rewards are not comparable across rounds (round ``k+1`` is
  conditioned on round ``k``'s choice).
"""

from __future__ import annotations

import asyncio
import copy
import logging
import re
import uuid
from argparse import Namespace
from typing import Any

from slime.rollout.base_types import RolloutFnEvalOutput, RolloutFnTrainOutput
from slime.rollout.sglang_rollout import GenerateState, generate
from slime.utils.async_utils import run
from slime.utils.types import Sample

from .dataset import build_samples_from_llm4cov
from .opd_remote_client import (
    OpdRelayClient,
    build_job_id,
    build_round_files,
    load_result_tree,
    parse_teacher_specs,
)
from .reward import _parse_testbench, compute_reward

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Per-rollout split log helpers
# When active: logger.propagate=False so rollout lines go ONLY to the split
# file and not to the main log.  Main log retains framework/Megatron output.
# ---------------------------------------------------------------------------
_split_log_handler: "logging.FileHandler | None" = None


def _set_split_log(args, rollout_id: int, is_eval: bool) -> None:
    """Open a split log file and redirect this module's logger into it."""
    global _split_log_handler
    _close_split_log()
    log_dir = getattr(args, "rollout_log_dir", None)
    if not log_dir:
        return
    import os as _os
    _os.makedirs(log_dir, exist_ok=True)
    if is_eval:
        path = _os.path.join(log_dir, f"eval_step_{rollout_id}.log")
    else:
        interval = int(getattr(args, "save_interval", 50) or 50)
        group = rollout_id // interval
        start = group * interval
        end   = (group + 1) * interval - 1
        path  = _os.path.join(log_dir, f"train_step_{start}-{end}.log")
    handler = logging.FileHandler(path, mode="a")
    handler.setFormatter(
        logging.Formatter("[%(asctime)s] %(filename)s:%(lineno)d - %(message)s")
    )
    logger.addHandler(handler)
    logger.propagate = False   # suppress propagation to main log handlers
    _split_log_handler = handler


def _close_split_log() -> None:
    """Detach split log handler and restore propagation to the main log."""
    global _split_log_handler
    if _split_log_handler is not None:
        logger.removeHandler(_split_log_handler)
        _split_log_handler.flush()
        _split_log_handler.close()
        _split_log_handler = None
    logger.propagate = True    # restore normal propagation



_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL | re.IGNORECASE)

_EVAL_SAMPLE_CACHE: dict[tuple, list[Sample]] = {}


def _strip_think(text: str) -> str:
    return _THINK_RE.sub("", text).strip()


OUTPUT_FORMAT_REQ = """
OUTPUT REQUIREMENTS:

1. You MUST explicitly state the filename for the testbench in plain text using format:
     filename: tb_xxxx.sv
2. After stating the filename, you MUST output the complete testbench
    inside a fenced SystemVerilog code block:
```systemverilog
module tb_example;
  ...
endmodule
```
"""


# ── COMMENTED OUT: old _build_tool_feedback ─────────────────────────────────
# def _build_tool_feedback(reward: float, eda_feedback: str | None = None) -> str:
#     if eda_feedback:
#         return (
#             "EDA tool feedback:\n"
#             + eda_feedback
#             + "\n\nPlease analyse the feedback above and rewrite the testbench to "
#             "maximise coverage. Follow the same output format (filename line + "
#             "fenced systemverilog block)."
#         )
#     if reward <= 0.0:
#         return (
#             "EDA tool feedback:\n"
#             "- status: failed (simulation did not complete or produced no coverage)\n"
#             "Please diagnose the likely cause and rewrite the testbench to maximise "
#             "coverage. Follow the same output format (filename line + fenced "
#             "systemverilog block)."
#         )
#     coverage = reward - 1.0
#     return (
#         "EDA tool feedback:\n"
#         "- status: success\n"
#         f"- overall coverage: {coverage * 100:.2f}%\n"
#         "This is the previous attempt. Try to push coverage higher by exercising "
#         "more RTL paths. Same output format (filename line + fenced systemverilog block)."
#     )
# ─────────────────────────────────────────────────────────────────────────────

def _build_tool_feedback(
    reward: float,
    eda_feedback: str | None = None,
    eda_status: str = "",
) -> str:
    """Build a user turn that feeds the EDA tool result back to the model.

    Matches the combined format of build_react_followup_prompt(is_single_message=True)
    in batch_query_eval.py: parse-result block + EDA-result block + instruction block.
    """
    parse_block = "Response parse result:\n- status: success\n- type: file"

    if eda_feedback:
        eda_block = "EDA result:\n" + eda_feedback
    elif reward <= 0.0:
        eda_block = "EDA result:\n- status: failed (simulation did not complete or produced no coverage)"
    else:
        eda_block = "EDA result:\n- status: success\n- stage: success\n- overall coverage: {:.2f}%".format(
            (reward - 1.0) * 100
        )

    if eda_status == "xrun_failed" or (not eda_feedback and reward <= 0.0):
        instruction = "Fix the xrun failure by editing the testbench."
    elif eda_status and eda_status != "success":
        instruction = "Fix the coverage run failure (imc stage) by editing the testbench."
    else:
        instruction = "Improve coverage toward 100% by editing the testbench."

    instruction_block = "Instruction for next round:\n" + instruction + OUTPUT_FORMAT_REQ
    return "\n\n".join([parse_block, eda_block, instruction_block])


def _clone_prompt_into_group(
    base_sample: Sample,
    prompt: str | list[dict[str, str]],
    messages_history: list[dict[str, str]],
    group_index: int,
    start_index: int,
    n_samples: int,
) -> list[Sample]:
    """Produce a fresh GRPO group of ``n_samples`` clones sharing one prompt."""
    group = []
    for k in range(n_samples):
        clone = Sample(
            prompt=prompt,
            metadata={
                **copy.deepcopy(base_sample.metadata),
                "round_number": base_sample.metadata.get("round_number", 0),
                "chat_history": messages_history,
            },
            label=base_sample.label,
            group_index=group_index,
            index=start_index + k,
            session_id=str(uuid.uuid4()),
        )
        group.append(clone)
    return group


def _apply_diversity_reward(group: list[Sample], args: Namespace) -> None:
    """In-place: reward_i = coverage_score_i + div_lam * diversity_i.

    diversity_i = mean over this rollout's covered bins of 1/cnt(b), where cnt(b)
    is the number of rollouts in the group that cover bin b (counted over the union
    of all rollouts' covered_bin_ids — INCLUDING all-covered bins, whose cnt = the
    number of successful rollouts):
        cnt(b) = #{rollouts covering b};  weight(b) = 1/cnt(b)
        diversity_i = (sum_{b in covered_i} weight(b)) / |covered_i|
    i.e. "the average rarity/uniqueness of what this rollout covered". Scale in
    [1/n_rollout, 1] (all-unique -> 1, all-shared-by-n -> 1/n), independent of the
    DUT bin count, so div_lam is comparable across tasks. Fail rollouts (no
    coverage) cover nothing -> diversity 0. No-op (reward stays = coverage score,
    log shows "n/a") when no covered data is present (no --eda-log-feedback-train,
    or an all-fail group). Tool-feedback (prompt + logs) still uses uncovered.
    """
    from collections import Counter

    lam = float(getattr(args, "div_lam", 0.0) or 0.0)
    covered: list[set] = []
    for s in group:
        el = (s.metadata or {}).get("_eda_log") or {}
        covered.append(set(el.get("covered_bin_ids") or []) if el.get("has_coverage") else set())
    cnt: Counter = Counter(b for cb in covered for b in cb)   # union, with hit counts
    if not cnt:
        return  # all-fail / no covered data -> leave coverage scores (log -> n/a)
    for s, cb in zip(group, covered, strict=False):
        div = (sum(1.0 / cnt[b] for b in cb) / len(cb)) if cb else 0.0
        s.metadata["diversity"] = div   # logged in TRAIN_SAMPLE/TRAIN_GROUP; absent -> "n/a"
        s.reward = float(s.reward if s.reward is not None else 0.0) + lam * div



def _set_train_loss_type(sample: Sample, loss_type: str, **extra: Any) -> None:
    meta = dict(sample.train_metadata or {})
    meta["loss_type"] = loss_type
    meta.update(extra)
    sample.train_metadata = meta


def _prompt_to_text(
    prompt: str | list[dict[str, str]],
    tokenizer: Any,
    apply_chat_template_kwargs: dict[str, Any],
) -> str:
    if isinstance(prompt, str):
        return prompt
    return tokenizer.apply_chat_template(
        prompt,
        tokenize=False,
        add_generation_prompt=True,
        **(apply_chat_template_kwargs or {}),
    )


def _sample_prompt_ids(sample: Sample, tokenizer: Any, prompt_text: str) -> list[int]:
    if sample.tokens and sample.response_length >= 0:
        prompt_len = max(0, len(sample.tokens) - int(sample.response_length or 0))
        if prompt_len > 0:
            return list(sample.tokens[:prompt_len])
    return list(tokenizer.encode(prompt_text, add_special_tokens=False))


def _sample_response_token_ids(sample: Sample) -> list[int]:
    if not sample.tokens or not sample.response_length:
        return []
    return list(sample.tokens[-int(sample.response_length):])


def _coerce_float_list(value: Any) -> list[float] | None:
    if not isinstance(value, list):
        return None
    out: list[float] = []
    for item in value:
        try:
            out.append(float(item))
        except (TypeError, ValueError):
            return None
    return out


def _coerce_int_list(value: Any) -> list[int] | None:
    if not isinstance(value, list):
        return None
    out: list[int] = []
    for item in value:
        try:
            out.append(int(item))
        except (TypeError, ValueError):
            return None
    return out


def _coerce_float_matrix(value: Any) -> list[list[float]] | None:
    if not isinstance(value, list):
        return None
    out: list[list[float]] = []
    for row in value:
        coerced = _coerce_float_list(row)
        if coerced is None:
            return None
        out.append(coerced)
    return out


def _coerce_int_matrix(value: Any) -> list[list[int]] | None:
    if not isinstance(value, list):
        return None
    out: list[list[int]] = []
    for row in value:
        coerced = _coerce_int_list(row)
        if coerced is None:
            return None
        out.append(coerced)
    return out


def _matrix_has_shape(value: list[list[Any]] | None, rows: int, cols: int) -> bool:
    return value is not None and len(value) == rows and all(len(row) == cols for row in value)


def _entry_reward(entry: dict[str, Any]) -> float:
    for key in ("reward", "score", "coverage_reward"):
        if key in entry:
            try:
                return float(entry[key])
            except (TypeError, ValueError):
                return 0.0
    eda_log = entry.get("eda_log") or entry.get("eda") or {}
    if isinstance(eda_log, dict) and eda_log.get("has_coverage"):
        try:
            return 1.0 + float(eda_log.get("overall_coverage", 0.0))
        except (TypeError, ValueError):
            return 0.0
    return 0.0


def _entry_eda_log(entry: dict[str, Any]) -> dict[str, Any]:
    eda_log = entry.get("eda_log") or entry.get("eda") or {}
    return dict(eda_log) if isinstance(eda_log, dict) else {}


def _entry_feedback(entry: dict[str, Any]) -> str | None:
    feedback = entry.get("eda_feedback") or entry.get("feedback")
    return str(feedback) if feedback is not None else None


def _result_entries(result: dict[str, Any], *keys: str) -> list[dict[str, Any]]:
    for key in keys:
        value = result.get(key)
        if value is None:
            continue
        if isinstance(value, list):
            return [v for v in value if isinstance(v, dict)]
        if isinstance(value, dict):
            rollouts = value.get("rollouts")
            if isinstance(rollouts, list):
                return [v for v in rollouts if isinstance(v, dict)]
            return [v for v in value.values() if isinstance(v, dict)]
    return []


def _student_rollout_payload(sample: Sample, slot: int) -> dict[str, Any]:
    sid = f"s{slot:03d}"
    sample.metadata["_opd_student_id"] = sid
    filename, testbench = _parse_testbench(sample.response or "")
    payload: dict[str, Any] = {
        "id": sid,
        "sample_index": sample.index,
        "assistant_response": sample.response or "",
        "filename": filename,
        "input_token_ids": list(sample.tokens or []),
        "response_token_ids": _sample_response_token_ids(sample),
        "response_token_count": int(sample.response_length or 0),
        "rollout_log_probs": sample.rollout_log_probs or [],
        "topk_token_ids": copy.deepcopy(sample.rollout_topk_token_ids or []),
        "topk_student_log_probs": copy.deepcopy(sample.rollout_topk_log_probs or []),
        "status": sample.status.value if sample.status else "unknown",
    }
    if testbench is not None:
        payload["testbench"] = testbench
    return payload


def _apply_remote_student_scores(group: list[Sample], result: dict[str, Any]) -> None:
    entries = _result_entries(result, "student_rollouts", "students", "student")
    by_id = {str(e.get("id")): e for e in entries if e.get("id") is not None}
    missing: list[str] = []
    for sample in group:
        sid = str(sample.metadata.get("_opd_student_id", ""))
        entry = by_id.get(sid)
        if entry is None:
            missing.append(sid)
            continue
        sample.reward = _entry_reward(entry)
        sample.metadata["_eda_log"] = _entry_eda_log(entry)
        feedback = _entry_feedback(entry)
        if feedback is not None:
            sample.metadata["eda_feedback"] = feedback
        _set_train_loss_type(sample, "rl")
    if missing:
        raise RuntimeError(f"OPD relay result missing student entries: {missing}")


def _best_entry(entries: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not entries:
        return None
    return max(entries, key=_entry_reward)


def _make_teacher_sample(
    *,
    args: Namespace,
    state: GenerateState,
    reference_student: Sample,
    teacher_entry: dict[str, Any],
    prompt_text: str,
    messages_history: list[dict[str, str]],
    sample_index: int,
    round_idx: int,
    best_student_reward: float,
    best_teacher_reward: float,
) -> Sample:
    response = str(
        teacher_entry.get("assistant_response")
        or teacher_entry.get("response")
        or teacher_entry.get("text")
        or ""
    )
    tokenizer = state.tokenizer
    prompt_ids = _sample_prompt_ids(reference_student, tokenizer, prompt_text)
    response_ids = list(tokenizer.encode(response, add_special_tokens=False))
    returned_ids = _coerce_int_list(
        teacher_entry.get("generated_token_ids")
        or teacher_entry.get("token_ids")
        or teacher_entry.get("output_token_ids")
    )
    token_logprobs = _coerce_float_list(
        teacher_entry.get("generated_token_logprobs")
        or teacher_entry.get("token_logprobs")
        or teacher_entry.get("output_token_logprobs")
    )
    teacher_log_probs = None
    if token_logprobs is not None and len(token_logprobs) == len(response_ids):
        if returned_ids is None or returned_ids == response_ids:
            teacher_log_probs = token_logprobs

    metadata = {
        **copy.deepcopy(reference_student.metadata),
        "round_number": round_idx,
        "chat_history": copy.deepcopy(messages_history),
        "opd_teacher": teacher_entry.get("teacher") or teacher_entry.get("teacher_name"),
        "opd_teacher_rollout_id": teacher_entry.get("id"),
        "opd_best_student_reward": best_student_reward,
        "opd_best_teacher_reward": best_teacher_reward,
        "opd_gate_pass": True,
        "_eda_log": _entry_eda_log(teacher_entry),
    }
    feedback = _entry_feedback(teacher_entry)
    if feedback is not None:
        metadata["eda_feedback"] = feedback

    sample = Sample(
        prompt=reference_student.prompt,
        tokens=prompt_ids + response_ids,
        response=response,
        response_length=len(response_ids),
        label=reference_student.label,
        reward=best_teacher_reward,
        loss_mask=[1] * len(response_ids),
        group_index=reference_student.group_index,
        index=sample_index,
        session_id=str(uuid.uuid4()),
        metadata=metadata,
        teacher_log_probs=teacher_log_probs,
        status=Sample.Status.COMPLETED,
        train_metadata={
            "loss_type": "opd",
            "opd_weight": float(getattr(args, "opd_lambda", 1.0) or 0.0),
            "best_student_reward": best_student_reward,
            "best_teacher_reward": best_teacher_reward,
            "teacher": metadata.get("opd_teacher"),
        },
    )
    if teacher_log_probs is None and token_logprobs is not None:
        sample.metadata["opd_teacher_logprob_alignment"] = "mismatch"
    return sample


async def _generate_group_only(
    args: Namespace,
    state: GenerateState,
    group: list[Sample],
    sampling_params: dict[str, Any],
) -> list[Sample]:
    async def _one(sample: Sample) -> Sample:
        if sample.session_id is None:
            sample.session_id = str(uuid.uuid4())
        async with state.semaphore:
            await generate(args, sample, sampling_params.copy())
        return sample

    return await asyncio.gather(*[_one(s) for s in group])


async def _score_existing_group_locally(
    args: Namespace,
    group: list[Sample],
    want_detail: bool,
) -> list[Sample]:
    async def _one(sample: Sample) -> Sample:
        sample.reward = await compute_reward(args, sample, want_detail=want_detail)
        _set_train_loss_type(sample, "rl")
        return sample

    scored = await asyncio.gather(*[_one(s) for s in group])
    if getattr(args, "use_uncovered_reward", False):
        _apply_diversity_reward(scored, args)
    return scored


def _run_opd_relay_round_sync(
    *,
    args: Namespace,
    job_id: str,
    dataset_id: str,
    rollout_id: int,
    round_idx: int,
    prompt_text: str,
    messages_history: list[dict[str, str]],
    context: dict[str, Any],
    student_rollouts: list[dict[str, Any]],
    sampling_params: dict[str, Any],
    want_detail: bool,
) -> dict[str, Any]:
    teachers = parse_teacher_specs(getattr(args, "opd_teachers", ""))
    if not teachers:
        raise ValueError("OPD relay enabled but no teachers configured")
    files = build_round_files(
        job_id=job_id,
        dataset_id=dataset_id,
        rollout_id=rollout_id,
        round_idx=round_idx,
        prompt=prompt_text,
        state={
            "round_idx": round_idx,
            "messages_history": messages_history,
            "prompt_is_chat_template_text": True,
        },
        context=context,
        student_rollouts=student_rollouts,
        teachers=teachers,
        sampling_params=sampling_params,
        want_detail=want_detail,
        score_student_rollouts=True,
        topk_k=int(getattr(args, "opd_topk", 0) or 0),
    )
    client = OpdRelayClient(args)
    try:
        result_dir = client.submit_and_wait(
            job_id,
            files,
            timeout_s=float(getattr(args, "opd_timeout", 1800.0) or 1800.0),
            poll_s=float(getattr(args, "opd_poll", 2.0) or 2.0),
        )
        return load_result_tree(result_dir)
    finally:
        client.close()


async def _score_group_with_opd_relay(
    *,
    args: Namespace,
    state: GenerateState,
    group: list[Sample],
    sampling_params: dict[str, Any],
    rollout_id: int,
    round_idx: int,
    dataset_id: str,
    want_detail: bool,
    sample_index_allocator,
) -> tuple[list[Sample], list[Sample]]:
    reference = group[0]
    apply_chat_template_kwargs = getattr(args, "apply_chat_template_kwargs", {}) or {}
    prompt_text = _prompt_to_text(reference.prompt, state.tokenizer, apply_chat_template_kwargs)
    messages_history = list(reference.metadata.get("chat_history") or reference.metadata.get("initial_messages") or [])
    context = reference.metadata.get("llm4cov_context") or {}
    if not isinstance(context, dict):
        raise RuntimeError("OPD relay requires llm4cov_context metadata as a dict")

    student_rollouts = [_student_rollout_payload(sample, i) for i, sample in enumerate(group)]
    job_id = build_job_id(dataset_id, rollout_id, round_idx)
    result = await asyncio.to_thread(
        _run_opd_relay_round_sync,
        args=args,
        job_id=job_id,
        dataset_id=dataset_id,
        rollout_id=rollout_id,
        round_idx=round_idx,
        prompt_text=prompt_text,
        messages_history=messages_history,
        context=context,
        student_rollouts=student_rollouts,
        sampling_params=sampling_params,
        want_detail=want_detail,
    )

    _apply_remote_student_scores(group, result)
    teacher_entries = _result_entries(result, "teacher_rollouts", "teachers", "teacher")
    best_teacher = _best_entry(teacher_entries)
    best_student = max(group, key=lambda s: float(s.reward or 0.0))
    best_student_reward = float(best_student.reward or 0.0)
    best_teacher_reward = _entry_reward(best_teacher) if best_teacher is not None else -float("inf")
    gate_eps = float(getattr(args, "opd_gate_eps", 0.0) or 0.0)
    gate_pass = best_teacher is not None and best_teacher_reward > best_student_reward + gate_eps

    for sample in group:
        sample.metadata["opd_job_id"] = job_id
        sample.metadata["opd_best_student_reward"] = best_student_reward
        sample.metadata["opd_best_teacher_reward"] = best_teacher_reward
        sample.metadata["opd_gate_pass"] = bool(gate_pass)

    logger.info(
        "OPD_GATE step=%d dataset_id=%s round=%d job=%s gate=%s "
        "best_student=%.4f best_teacher=%.4f eps=%.4f n_teachers=%d",
        rollout_id,
        dataset_id,
        round_idx + 1,
        job_id,
        gate_pass,
        best_student_reward,
        best_teacher_reward,
        gate_eps,
        len(teacher_entries),
    )

    if gate_pass and best_teacher is not None:
        student_entries_by_id = {
            str(entry.get("id")): entry
            for entry in _result_entries(result, "student_rollouts", "students", "student")
            if entry.get("id") is not None
        }
        best_teacher_name = str(best_teacher.get("teacher") or best_teacher.get("teacher_name") or "")
        requested_topk = int(getattr(args, "opd_topk", 0) or 0)
        require_topk = requested_topk > 1

        def _entry_for_sample(sample: Sample) -> dict[str, Any] | None:
            sid = str(sample.metadata.get("_opd_student_id", ""))
            return student_entries_by_id.get(sid)

        def _student_teacher_log_probs(sample: Sample) -> list[float] | None:
            entry = _entry_for_sample(sample)
            if entry is None:
                return None
            direct = _coerce_float_list(entry.get("teacher_log_probs"))
            if direct is not None and (not best_teacher_name or entry.get("teacher_logprob_teacher") == best_teacher_name):
                return direct
            for score in entry.get("teacher_scores") or []:
                if not isinstance(score, dict):
                    continue
                if best_teacher_name and str(score.get("teacher") or "") != best_teacher_name:
                    continue
                if score.get("status") not in {"success", "topk_success", "partial_topk"}:
                    continue
                scored = _coerce_float_list(score.get("teacher_log_probs"))
                if scored is not None:
                    return scored
            return None

        def _student_teacher_topk(sample: Sample):
            entry = _entry_for_sample(sample)
            if entry is None:
                return None
            topk_ids = _coerce_int_matrix(entry.get("topk_token_ids")) or copy.deepcopy(sample.rollout_topk_token_ids)
            student_topk = _coerce_float_matrix(entry.get("topk_student_log_probs")) or copy.deepcopy(sample.rollout_topk_log_probs)
            teacher_topk = _coerce_float_matrix(entry.get("teacher_topk_log_probs"))
            teacher_masks = _coerce_float_matrix(entry.get("teacher_topk_logprob_masks"))
            if teacher_topk is not None:
                return topk_ids, student_topk, teacher_topk, teacher_masks
            for score in entry.get("teacher_scores") or []:
                if not isinstance(score, dict):
                    continue
                if best_teacher_name and str(score.get("teacher") or "") != best_teacher_name:
                    continue
                teacher_topk = _coerce_float_matrix(score.get("teacher_topk_log_probs"))
                teacher_masks = _coerce_float_matrix(score.get("teacher_topk_logprob_masks"))
                if teacher_topk is not None:
                    return topk_ids, student_topk, teacher_topk, teacher_masks
            return None

        topk_by_index = {}
        topk_valid = bool(best_teacher_name) and require_topk
        if require_topk:
            for sample in group:
                expected = int(sample.response_length or 0)
                width = requested_topk
                packed = _student_teacher_topk(sample)
                if packed is None:
                    topk_valid = False
                    sample.metadata["opd_gate_reason"] = "missing_teacher_topk_log_probs"
                    break
                topk_ids, student_topk, teacher_topk, teacher_masks = packed
                if teacher_masks is None and _matrix_has_shape(teacher_topk, expected, width):
                    teacher_masks = [[1.0] * width for _ in range(expected)]
                if not (
                    _matrix_has_shape(topk_ids, expected, width)
                    and _matrix_has_shape(teacher_topk, expected, width)
                    and _matrix_has_shape(teacher_masks, expected, width)
                ):
                    topk_valid = False
                    sample.metadata["opd_gate_reason"] = (
                        f"topk_shape_invalid expected=({expected},{width})"
                    )
                    break
                mask_total = expected * width
                mask_hit = sum(sum(float(x) for x in row) for row in teacher_masks)
                if mask_hit < mask_total:
                    topk_valid = False
                    sample.metadata["opd_gate_reason"] = (
                        f"teacher_topk_incomplete hit={mask_hit:.0f}/{mask_total}"
                    )
                    break
                if student_topk is not None and not _matrix_has_shape(student_topk, expected, width):
                    student_topk = None
                topk_by_index[int(sample.index)] = (topk_ids, student_topk, teacher_topk, teacher_masks)

        if topk_valid:
            for sample in group:
                topk_ids, student_topk, teacher_topk, teacher_masks = topk_by_index[int(sample.index)]
                sample.rollout_topk_token_ids = topk_ids
                sample.rollout_topk_log_probs = student_topk
                sample.teacher_topk_log_probs = teacher_topk
                sample.teacher_topk_logprob_masks = teacher_masks
                sample.metadata["opd_teacher_logprob_teacher"] = best_teacher_name
                sample.metadata["opd_topk"] = requested_topk
                sample.metadata["opd_topk_teacher_coverage"] = (
                    sum(sum(float(x) for x in row) for row in teacher_masks) / max(1, requested_topk * int(sample.response_length or 0))
                )
                _set_train_loss_type(
                    sample,
                    "opd",
                    opd_weight=float(getattr(args, "opd_lambda", 1.0) or 0.0),
                    best_student_reward=best_student_reward,
                    best_teacher_reward=best_teacher_reward,
                    teacher=best_teacher_name,
                    opd_topk=requested_topk,
                )
            return group, group

        if not require_topk:
            valid_opd = bool(best_teacher_name)
            teacher_log_probs_by_index: dict[int, list[float]] = {}
            for sample in group:
                teacher_log_probs = _student_teacher_log_probs(sample)
                expected = int(sample.response_length or 0)
                if teacher_log_probs is None or len(teacher_log_probs) != expected:
                    valid_opd = False
                    sample.metadata["opd_gate_reason"] = (
                        f"teacher_logprob_len={len(teacher_log_probs) if teacher_log_probs is not None else 'missing'} "
                        f"expected={expected}"
                    )
                    break
                teacher_log_probs_by_index[int(sample.index)] = teacher_log_probs

            if valid_opd:
                for sample in group:
                    teacher_log_probs = teacher_log_probs_by_index[int(sample.index)]
                    sample.teacher_log_probs = teacher_log_probs
                    sample.metadata["opd_teacher_logprob_teacher"] = best_teacher_name
                    sample.metadata["opd_teacher_logprob_tokens"] = len(teacher_log_probs)
                    _set_train_loss_type(
                        sample,
                        "opd",
                        opd_weight=float(getattr(args, "opd_lambda", 1.0) or 0.0),
                        best_student_reward=best_student_reward,
                        best_teacher_reward=best_teacher_reward,
                        teacher=best_teacher_name,
                    )
                return group, group

        for sample in group:
            sample.metadata["opd_gate_pass"] = False
            sample.metadata.setdefault(
                "opd_gate_reason",
                "missing_teacher_topk_log_probs" if require_topk else "missing_teacher_log_probs",
            )
            _set_train_loss_type(sample, "rl")

    return group, group

async def _generate_and_score_group(
    args: Namespace,
    state: GenerateState,
    group: list[Sample],
    sampling_params: dict[str, Any],
    max_retries: int = 0,
    want_detail: bool = False,
) -> list[Sample]:
    """Sample + score one GRPO group concurrently.

    ``max_retries``: retry generate+score up to this many times when reward==0.0
    (mirrors batch_query_eval llm_retries). Default 0 for training; 3 for eval.
    ``want_detail``: passed to compute_reward to control EDA detail fetch.
    """

    async def _one(sample: Sample) -> Sample:
        for attempt in range(max_retries + 1):
            if attempt > 0:
                sample.session_id = str(uuid.uuid4())
                # Reset generation state: generate() asserts status==PENDING|ABORTED
                sample.status = Sample.Status.PENDING
                sample.tokens = []
                sample.response = ""
                sample.response_length = 0
                sample.rollout_log_probs = None
                sample.reward = None
                logger.info(
                    "retry %d/%d sample group=%s idx=%s (reward=0.0)",
                    attempt, max_retries, sample.group_index, sample.index,
                )
            if sample.session_id is None:
                sample.session_id = str(uuid.uuid4())
            async with state.semaphore:
                await generate(args, sample, sampling_params.copy())
            sample.reward = await compute_reward(args, sample, want_detail=want_detail)
            if sample.reward > 0.0 or attempt >= max_retries:
                break
        return sample

    scored = await asyncio.gather(*[_one(s) for s in group])
    if getattr(args, "use_uncovered_reward", False):
        _apply_diversity_reward(scored, args)
    for sample in scored:
        _set_train_loss_type(sample, "rl")
    return scored


def _select_pivot_sample(group: list[Sample], evaluation: bool) -> Sample:
    """Pick the sample whose completion drives the next round's prompt."""
    scored = [(float(s.reward if s.reward is not None else 0.0), s) for s in group]
    if evaluation:
        _, best = max(scored, key=lambda x: x[0])
        return best
    _, worst = min(scored, key=lambda x: x[0])
    return worst


async def _rollout_one_prompt(
    args: Namespace,
    state: GenerateState,
    initial_group: list[Sample],
    num_rounds: int,
    sampling_params: dict[str, Any],
    evaluation: bool,
    group_index_allocator,
    sample_index_allocator,
    max_retries: int = 0,
    rollout_id: int = -1,
) -> list[list[Sample]]:
    """Run K rounds for one seed prompt. Returns K groups.

    Eval mode (evaluation=True) adds three behaviours mirroring
    batch_query_eval.py markov-react:

    * Markov snapshot: from round 2 onward the prompt is condensed to
      [sys, user, best_assistant, best_feedback] to prevent linear blow-up.
    * Early stop: break when pivot achieves perfect coverage (reward >= 2.0,
      i.e. overall_coverage == 1.0).
    * Retries: each generate+score pair is retried up to ``max_retries`` times
      on reward == 0.0 (3 for eval, 0 for training by default).
    """
    assert num_rounds >= 1
    tokenizer = state.tokenizer
    apply_chat_template = bool(args.apply_chat_template)
    apply_chat_template_kwargs = getattr(args, "apply_chat_template_kwargs", {}) or {}

    base_messages = list(initial_group[0].metadata.get("initial_messages") or [])
    context_id = initial_group[0].metadata.get("dataset_id", "?")

    # Markov snapshot (eval only): mirrors batch_query_eval.py markov-react.
    # Keeps prompt at [sys, user, best_assistant, best_feedback] from round 2+.
    use_markov = evaluation
    markov_snapshot: list[dict[str, str]] = []
    best_reward_seen: float = -float("inf")

    _want_detail = (
        bool(getattr(args, "eda_log_feedback_eval", False)) if evaluation
        else bool(getattr(args, "eda_log_feedback_train",
                          getattr(args, "eda_log_feedback", False)))
    )
    rounds_output: list[list[Sample]] = []
    current_group = initial_group
    for round_idx in range(num_rounds):
        for s in current_group:
            s.metadata["round_number"] = round_idx

        train_group: list[Sample]
        if bool(getattr(args, "use_opd_relay", False)) and not evaluation:
            current_group = await _generate_group_only(args, state, current_group, sampling_params)
            try:
                train_group, current_group = await _score_group_with_opd_relay(
                    args=args,
                    state=state,
                    group=current_group,
                    sampling_params=sampling_params,
                    rollout_id=rollout_id,
                    round_idx=round_idx,
                    dataset_id=context_id,
                    want_detail=_want_detail,
                    sample_index_allocator=sample_index_allocator,
                )
            except Exception:
                logger.exception(
                    "OPD relay failed; falling back to local RL scoring "
                    "step=%d dataset_id=%s round=%d",
                    rollout_id, context_id, round_idx + 1,
                )
                current_group = await _score_existing_group_locally(args, current_group, _want_detail)
                train_group = current_group
        else:
            current_group = await _generate_and_score_group(
                args, state, current_group, sampling_params,
                max_retries=max_retries, want_detail=_want_detail,
            )
            train_group = current_group
        rounds_output.append(train_group)

        rewards = [float(s.reward or 0.0) for s in current_group]
        logger.info(
            "rollout step=%d dataset_id=%s round=%d/%d rewards=%s mean=%.4f",
            rollout_id,
            context_id,
            round_idx + 1,
            num_rounds,
            [f"{r:.4f}" for r in rewards],
            sum(rewards) / len(rewards) if rewards else 0.0,
        )

        if evaluation:
            _is_first_task = (initial_group[0].index == 0)
            for _s in current_group:
                # ── complete LLM generation log ──────────────────────────
                logger.info(
                    "EVAL_GEN dataset_id=%s round=%d/%d idx=%s status=%s len=%d\n%s",
                    context_id, round_idx + 1, num_rounds,
                    _s.index, _s.status, len(_s.response or ""),
                    _s.response or "",
                )
                # ── complete EDA result log ───────────────────────────────
                _el = _s.metadata.get("_eda_log", {})
                if _el.get("status") == "success":
                    logger.info(
                        "EVAL_EDA dataset_id=%s round=%d/%d idx=%s status=success "
                        "reward=%.4f coverage=%.4f is_pass_xrun=%s is_pass_targets=%s "
                        "filename=%s",
                        context_id, round_idx + 1, num_rounds, _s.index,
                        float(_s.reward or 0.0),
                        _el.get("overall_coverage", 0.0),
                        _el.get("is_pass_xrun"), _el.get("is_pass_targets"),
                        _el.get("filename"),
                    )
                else:
                    logger.info(
                        "EVAL_EDA dataset_id=%s round=%d/%d idx=%s status=%s "
                        "reward=%.4f filename=%s\n%s",
                        context_id, round_idx + 1, num_rounds, _s.index,
                        _el.get("status", "?"),
                        float(_s.reward or 0.0),
                        _el.get("filename"),
                        _el.get("err_msg") or _el.get("exc") or "",
                    )
                # ── EDA feedback (only when --eda-log-feedback-eval) ──────
                _eval_eda_fb = _s.metadata.get("eda_feedback")
                if _eval_eda_fb:
                    logger.info(
                        "EVAL_EDA_FEEDBACK dataset_id=%s round=%d/%d idx=%s\n%s",
                        context_id, round_idx + 1, num_rounds, _s.index,
                        _eval_eda_fb,
                    )
                # ── tokenizer stats (first task every round) ─────────────
                if _is_first_task:
                    _tok = state.tokenizer
                    if isinstance(_s.prompt, str):
                        _n_prompt = len(_tok.encode(_s.prompt))
                    else:
                        _n_prompt = len(_tok.apply_chat_template(
                            _s.prompt, tokenize=True, add_generation_prompt=True,
                            **(apply_chat_template_kwargs or {}),
                        ))
                    _n_resp = len(_tok.encode(_s.response or ""))
                    logger.info(
                        "EVAL_TOKENS[first_task] dataset_id=%s round=%d/%d idx=%s "
                        "prompt_tokens=%d response_tokens=%d total=%d",
                        context_id, round_idx + 1, num_rounds, _s.index,
                        _n_prompt, _n_resp, _n_prompt + _n_resp,
                    )

        if round_idx + 1 >= num_rounds:
            break

        pivot = _select_pivot_sample(current_group, evaluation=evaluation)
        pivot_reward = float(pivot.reward or 0.0)
        pivot_dataset_id = pivot.metadata.get("dataset_id", "?")
        eda_feedback: str | None = pivot.metadata.get("eda_feedback")
        eda_status_str: str = (pivot.metadata.get("_eda_log") or {}).get("status", "")
        logger.info(
            "rollout step=%d dataset_id=%s round=%d pivot reward=%.4f (mode=%s) eda_feedback_len=%d",
            rollout_id,
            context_id,
            round_idx + 1,
            pivot_reward,
            "best" if evaluation else "worst",
            len(eda_feedback) if eda_feedback else 0,
        )

        # Early stop: perfect coverage (overall_coverage=1.0 → reward=2.0).
        if evaluation and pivot_reward >= 2.0:
            logger.info(
                "rollout step=%d dataset_id=%s round=%d early stop (perfect coverage)",
                rollout_id, context_id, round_idx + 1,
            )
            break

        tool_feedback = _build_tool_feedback(pivot_reward, eda_feedback, eda_status=eda_status_str)

        # Grow the linear history (always maintained as canonical record).
        base_messages = list(base_messages) + [
            {"role": "assistant", "content": _strip_think(pivot.response)},
            {"role": "user", "content": tool_feedback},
        ]

        # Update markov snapshot when this round improved the best reward seen.
        round_best = max(rewards)
        updated_best = round_best > best_reward_seen
        if updated_best:
            best_reward_seen = round_best
        if use_markov and updated_best and len(base_messages) >= 4:
            markov_snapshot = [
                base_messages[0], base_messages[1],
                base_messages[-2], base_messages[-1],
            ]

        # From round 2 onward generate from condensed snapshot, not full history
        # (mirrors batch_query_eval.py: messages_snapshot = markov when round_idx > 1).
        if use_markov and round_idx >= 1 and markov_snapshot:
            next_messages = markov_snapshot
        else:
            next_messages = base_messages

        next_prompt = (
            tokenizer.apply_chat_template(
                next_messages,
                tokenize=False,
                add_generation_prompt=True,
                **apply_chat_template_kwargs,
            )
            if apply_chat_template
            else list(next_messages)
        )

        next_group_index = group_index_allocator()
        base_sample = initial_group[0]
        start_index = sample_index_allocator(len(initial_group))
        current_group = _clone_prompt_into_group(
            base_sample=base_sample,
            prompt=next_prompt,
            messages_history=list(next_messages),
            group_index=next_group_index,
            start_index=start_index,
            n_samples=len(initial_group),
        )

    return rounds_output


# ---------------------------------------------------------------------------
# Training entry point
# ---------------------------------------------------------------------------


async def _generate_rollout_async(
    args: Namespace, rollout_id: int, data_source: Any
) -> RolloutFnTrainOutput:
    state = GenerateState(args)
    num_rounds = int(args.num_agentic_rounds)
    if num_rounds < 1:
        raise ValueError(f"--num-agentic-rounds must be >= 1, got {num_rounds}")

    batch_size = args.rollout_batch_size
    if batch_size % num_rounds != 0:
        raise ValueError(
            f"rollout_batch_size ({batch_size}) must be divisible by num-agentic-rounds "
            f"({num_rounds}) so K rounds * num_prompts == rollout_batch_size."
        )
    num_prompts = batch_size // num_rounds

    initial_groups = data_source.get_samples(num_prompts)
    assert len(initial_groups) == num_prompts, (
        f"data_source returned {len(initial_groups)} prompt-groups, expected {num_prompts}"
    )

    # Share allocators with the data_source so group_index / sample_index values
    # remain unique across rollouts.
    next_group_index = {"v": data_source.sample_group_index}
    next_sample_index = {"v": data_source.sample_index}

    def allocate_group_index() -> int:
        gi = next_group_index["v"]
        next_group_index["v"] += 1
        return gi

    def allocate_sample_indices(n: int) -> int:
        start = next_sample_index["v"]
        next_sample_index["v"] += n
        return start

    sampling_params = state.sampling_params.copy()

    async def _one_prompt(group: list[Sample]) -> list[list[Sample]]:
        return await _rollout_one_prompt(
            args=args,
            state=state,
            initial_group=group,
            num_rounds=num_rounds,
            sampling_params=sampling_params,
            evaluation=False,
            group_index_allocator=allocate_group_index,
            sample_index_allocator=allocate_sample_indices,
            rollout_id=rollout_id,
        )

    all_rounds = await asyncio.gather(*[_one_prompt(g) for g in initial_groups])

    data_source.sample_group_index = next_group_index["v"]
    data_source.sample_index = next_sample_index["v"]

    flat_groups = [g for rounds_list in all_rounds for g in rounds_list]
    assert len(flat_groups) == batch_size, (
        f"Produced {len(flat_groups)} groups, expected rollout_batch_size={batch_size}."
    )
    flat_groups.sort(key=lambda g: g[0].index if g and g[0].index is not None else 0)
    return RolloutFnTrainOutput(samples=flat_groups, metrics=None)


def generate_rollout(
    args: Namespace,
    rollout_id: int,
    data_source: Any,
    evaluation: bool = False,
) -> RolloutFnTrainOutput | RolloutFnEvalOutput:
    """Training entry point. Wired via ``--rollout-function-path``.

    ``evaluation`` is part of slime's contract but always ``False`` here — eval
    is dispatched to :func:`eval_rollout` via ``--eval-function-path``.
    """
    if evaluation:
        # Defensive: if a user wires this same function as --eval-function-path,
        # fall through to the eval entry point rather than polluting train data.
        return eval_rollout(args, rollout_id, data_source, evaluation=True)
    _set_split_log(args, rollout_id, is_eval=False)
    return run(_generate_rollout_async(args, rollout_id, data_source))


# ---------------------------------------------------------------------------
# Evaluation entry point
# ---------------------------------------------------------------------------


def _get_or_load_eval_samples(args: Namespace, state: GenerateState) -> list[Sample]:
    cache_key = (
        args.llm4cov_eval_dataset_name,
        args.llm4cov_eval_dataset_split,
        args.hf_checkpoint,
        bool(args.apply_chat_template),
    )
    if cache_key not in _EVAL_SAMPLE_CACHE:
        _EVAL_SAMPLE_CACHE[cache_key] = build_samples_from_llm4cov(
            args=args,
            tokenizer=state.tokenizer,
            dataset_name=args.llm4cov_eval_dataset_name,
            split=args.llm4cov_eval_dataset_split,
        )
    return _EVAL_SAMPLE_CACHE[cache_key]


async def _eval_rollout_async(args: Namespace, rollout_id: int) -> RolloutFnEvalOutput:
    state = GenerateState(args)
    # eval_num_agentic_rounds = react follow-up rounds (same as batch_query_eval
    # --react-rounds). Total rounds = 1 initial + react_rounds.
    _eval_react_rounds = int(getattr(args, "eval_num_agentic_rounds", 1) or 1)
    if _eval_react_rounds < 0:
        raise ValueError(f"--eval-num-agentic-rounds must be >= 0, got {_eval_react_rounds}")
    num_rounds = 1 + _eval_react_rounds
    logger.info(
        "eval step=%d: react_rounds=%d  total_rounds=%d  "
        "(markov-react + early-stop + max_retries=3)",
        rollout_id, _eval_react_rounds, num_rounds,
    )

    eval_prompts = _get_or_load_eval_samples(args, state)
    n_per_prompt = max(
        1,
        int(getattr(args, "n_samples_per_eval_prompt", 0) or args.n_samples_per_prompt or 1),
    )
    logger.info(
        "eval step=%d: dataset=%s  n_prompts=%d  n_per_prompt=%d  total_tasks=%d",
        rollout_id, args.llm4cov_eval_dataset_name,
        len(eval_prompts), n_per_prompt, len(eval_prompts) * n_per_prompt,
    )

    # Local allocators — eval ids live in their own namespace so they can't
    # collide with the training data_source counters.
    next_gi = {"v": 0}
    next_si = {"v": 0}

    def alloc_gi() -> int:
        v = next_gi["v"]
        next_gi["v"] += 1
        return v

    def alloc_si(n: int) -> int:
        s = next_si["v"]
        next_si["v"] += n
        return s

    # Seed round-0 groups for every eval prompt.
    initial_groups: list[list[Sample]] = []
    for base in eval_prompts:
        gi = alloc_gi()
        start = alloc_si(n_per_prompt)
        group = []
        for k in range(n_per_prompt):
            clone = copy.deepcopy(base)
            clone.group_index = gi
            clone.index = start + k
            clone.session_id = str(uuid.uuid4())
            group.append(clone)
        initial_groups.append(group)

    sampling_params = state.sampling_params.copy()

    async def _one(group: list[Sample]) -> list[list[Sample]]:
        return await _rollout_one_prompt(
            args=args,
            state=state,
            initial_group=group,
            num_rounds=num_rounds,
            sampling_params=sampling_params,
            evaluation=True,
            group_index_allocator=alloc_gi,
            sample_index_allocator=alloc_si,
            max_retries=3,
            rollout_id=rollout_id,
        )

    import time as _time
    _t_eval_start = _time.time()
    logger.info("eval step=%d: asyncio.gather starting (%d tasks)...",
                rollout_id, len(initial_groups))
    all_rounds = await asyncio.gather(*[_one(g) for g in initial_groups])
    _t_eval_elapsed = _time.time() - _t_eval_start
    # Best-reward sample per prompt across all rounds (mirrors batch_query_eval
    # best_cov_result): avoids diluting the metric with lower early-round rewards.
    flat_samples: list[Sample] = [
        max(
            (s for g in rounds_list for s in g),
            key=lambda s: float(s.reward or 0.0),
        )
        for rounds_list in all_rounds
    ]
    _rewards = [float(s.reward or 0.0) for s in flat_samples]
    _n = len(_rewards)
    _pass_count = sum(1 for r in _rewards if r > 0.0)
    # overall_coverage: fail=0, matches batch_query_eval Best@1
    _cov_vals = [max(r - 1.0, 0.0) for r in _rewards]
    _overall_coverage = sum(_cov_vals) / _n if _n else 0.0

    # is_pass_targets (from _eda_log written by reward.py)
    _pass_targets = [
        bool(s.metadata.get("_eda_log", {}).get("is_pass_targets", False))
        for s in flat_samples
    ]
    _pass_targets_count = sum(_pass_targets)

    # task-type split: cvdp_copilot_* = non-agentic, cvdp_agentic_* = agentic
    _non_ag_pts = [
        pt for s, pt in zip(flat_samples, _pass_targets)
        if s.metadata.get("dataset_id", "").startswith("cvdp_copilot_")
    ]
    _ag_pts = [
        pt for s, pt in zip(flat_samples, _pass_targets)
        if s.metadata.get("dataset_id", "").startswith("cvdp_agentic_")
    ]
    _pt_non_ag = sum(_non_ag_pts) / len(_non_ag_pts) if _non_ag_pts else 0.0
    _pt_ag = sum(_ag_pts) / len(_ag_pts) if _ag_pts else 0.0

    # token stats (tokenizer-based, best-per-prompt samples)
    _tok = state.tokenizer
    _apply_tmpl = bool(args.apply_chat_template)
    _tmpl_kwargs = getattr(args, "apply_chat_template_kwargs", {}) or {}
    _total_prompt_tok = 0
    _total_resp_tok = 0
    for _s in flat_samples:
        if isinstance(_s.prompt, str):
            _total_prompt_tok += len(_tok.encode(_s.prompt))
        else:
            _total_prompt_tok += len(_tok.apply_chat_template(
                _s.prompt, tokenize=True, add_generation_prompt=True, **_tmpl_kwargs,
            ))
        _total_resp_tok += len(_tok.encode(_s.response or ""))

    # ── Eval Summary (mirrors batch_query_eval.py === Eval Summary ===) ──────────
    logger.info("=== Eval Summary (step=%d, elapsed=%.1fs) ===",
                rollout_id, _t_eval_elapsed)
    logger.info("  is_pass_xrun:             Pass@1= %.1f%%  (%d/%d)",
                100.0 * _pass_count / _n if _n else 0.0, _pass_count, _n)
    logger.info("  overall_coverage:         Best@1= %.4f",
                _overall_coverage)
    logger.info("  is_pass_targets:          Pass@1= %.1f%%  (%d/%d)",
                100.0 * _pass_targets_count / _n if _n else 0.0, _pass_targets_count, _n)
    if _non_ag_pts:
        logger.info("  pass_targets_non_agentic: Pass@1= %.1f%%  (%d/%d)",
                    _pt_non_ag * 100.0, sum(_non_ag_pts), len(_non_ag_pts))
    if _ag_pts:
        logger.info("  pass_targets_agentic:     Pass@1= %.1f%%  (%d/%d)",
                    _pt_ag * 100.0, sum(_ag_pts), len(_ag_pts))
    logger.info("  best_reward:              %.4f", max(_rewards) if _rewards else 0.0)
    logger.info("  token_stats:  prompt_tokens= %d  completion_tokens= %d  total= %d",
                _total_prompt_tok, _total_resp_tok, _total_prompt_tok + _total_resp_tok)
    logger.info("=" * 60)

    # metrics dict goes to wandb via RolloutFnEvalOutput.metrics
    _dset = args.llm4cov_eval_dataset_name
    _eval_metrics = {
        f"eval/{_dset}/is_pass_xrun":             _pass_count / _n if _n else 0.0,
        f"eval/{_dset}/overall_coverage":         _overall_coverage,
        f"eval/{_dset}/is_pass_targets":          _pass_targets_count / _n if _n else 0.0,
        f"eval/{_dset}/pass_targets_non_agentic": _pt_non_ag,
        f"eval/{_dset}/pass_targets_agentic":     _pt_ag,
        f"eval/{_dset}/best_reward":              max(_rewards) if _rewards else 0.0,
        f"eval/{_dset}/prompt_tokens":            _total_prompt_tok,
        f"eval/{_dset}/completion_tokens":        _total_resp_tok,
    }

    reward_key = args.eval_reward_key or args.reward_key
    return RolloutFnEvalOutput(
        data={
            args.llm4cov_eval_dataset_name: {
                "rewards": [
                    s.reward if not reward_key else s.reward[reward_key] for s in flat_samples
                ],
                "truncated": [s.status == Sample.Status.TRUNCATED for s in flat_samples],
                "samples": flat_samples,
            }
        },
        metrics=_eval_metrics,
    )


def eval_rollout(
    args: Namespace,
    rollout_id: int,
    data_source: Any,
    evaluation: bool = True,
) -> RolloutFnEvalOutput:
    """Eval entry point. Wired via ``--eval-function-path``.

    ``data_source`` is accepted for slime's signature but ignored — eval
    prompts come from ``--llm4cov-eval-dataset-{name,split}``.
    """
    del data_source
    _set_split_log(args, rollout_id, is_eval=True)
    try:
        return run(_eval_rollout_async(args, rollout_id))
    finally:
        _close_split_log()


# ---------------------------------------------------------------------------
# Custom rollout log hook  (registered via --custom-rollout-log-function-path)
# ---------------------------------------------------------------------------

_MASK_LOG_INTERVAL = 25   # log loss_mask detail every N rollout steps
_MAX_LOG_CHARS = 8192        # truncate TRAIN_INPUT full snapshots
_MAX_OUTPUT_LOG_CHARS = 32768  # truncate TRAIN_OUTPUT (4x input limit)


def log_train_samples(
    rollout_id: int,
    args,
    samples,          # flat list[Sample] after _convert_samples_to_train_data
    extra_metrics,
    rollout_time: float,
) -> bool:
    """Per-sample training log: data_id / input / output / reward / mask.

    Registered via --custom-rollout-log-function-path.
    Returns False so the default _log_rollout_data metrics still run.
    """
    log_mask_detail = (rollout_id % _MASK_LOG_INTERVAL == 0)

    # Pre-compute GRPO group statistics so we can log the actual training
    # signal (normalized advantage) alongside raw reward.
    # Mirrors _post_process_rewards in slime/ray/rollout.py.
    from collections import defaultdict as _dd
    _group_raw: dict = _dd(list)
    for _s in samples:
        _group_raw[_s.group_index].append(float(_s.reward or 0.0))
    _group_mean = {gid: sum(rs) / len(rs) for gid, rs in _group_raw.items()}
    _group_std = {
        gid: (sum((r - _group_mean[gid]) ** 2 for r in rs) / len(rs)) ** 0.5
        for gid, rs in _group_raw.items()
    }
    _use_grpo = (
        getattr(args, "advantage_estimator", "grpo")
        in ("grpo", "gspo", "reinforce_plus_plus_baseline")
        and getattr(args, "rewards_normalization", True)
    )
    _use_std = _use_grpo and getattr(args, "grpo_std_normalization", True)

    # Group samples and dataset_ids for TRAIN_GROUP summary + zero-std reporting.
    _group_samples: dict = _dd(list)
    _group_datasets: dict = {}
    for _s in samples:
        _group_samples[_s.group_index].append(_s)
        if _s.group_index not in _group_datasets:
            _group_datasets[_s.group_index] = _s.metadata.get("dataset_id", "?")
    _seen_groups: set = set()

    for s in samples:
        dataset_id  = s.metadata.get("dataset_id", "?")
        round_num   = s.metadata.get("round_number", 0)
        raw_reward  = float(s.reward or 0.0)
        resp_len    = s.response_length

        # ── group summary (once per group, before first sample's detail) ──
        if s.group_index not in _seen_groups:
            _seen_groups.add(s.group_index)
            _g = _group_samples[s.group_index]
            _gm = _group_mean.get(s.group_index, 0.0)
            _gs = _group_std.get(s.group_index, 0.0)
            _gds = _group_datasets.get(s.group_index, "?")
            _g_lines = []
            for _gs2 in _g:
                _r2 = float(_gs2.reward or 0.0)
                _adv2 = (_r2 - _gm) / (_gs + 1e-6) if _use_std else (_r2 - _gm)
                _d2 = _gs2.metadata.get("diversity")
                _ds2 = f"{_d2:.4f}" if _d2 is not None else "n/a"
                _lt2 = (_gs2.train_metadata or {}).get("loss_type", "rl")
                _g_lines.append(
                    f"  idx={_gs2.index:<4} loss={_lt2:<3} reward={_r2:+.4f} adv={_adv2:+.4f} div={_ds2}"
                )
            logger.info(
                "TRAIN_GROUP  step=%d group=%s dataset_id=%s round=%d "
                "n=%d mean=%.4f std=%.4f\n%s",
                rollout_id, s.group_index, _gds,
                s.metadata.get("round_number", 0),
                len(_g), _gm, _gs, "\n".join(_g_lines),
            )

        # ── per-sample header (reward + GRPO fields merged) ───────────────
        _el = s.metadata.get("_eda_log", {})
        _gid = s.group_index
        _g_mean = _group_mean.get(_gid, 0.0)
        _g_std  = _group_std.get(_gid, 0.0)
        _adv_raw = raw_reward - _g_mean
        _advantage = _adv_raw / (_g_std + 1e-6) if _use_std else _adv_raw
        _lp_mean = (
            sum(s.rollout_log_probs) / len(s.rollout_log_probs)
            if s.rollout_log_probs else None
        )
        _div = s.metadata.get("diversity")  # set only under --uur; else None -> "n/a"
        _loss_type = (s.train_metadata or {}).get("loss_type", "rl")
        logger.info(
            "TRAIN_SAMPLE step=%d group=%s idx=%s dataset_id=%s "
            "round=%d loss_type=%s resp_len=%d truncated=%s "
            "reward=%.4f advantage=%.4f group_mean=%.4f group_std=%.4f "
            "lp_mean=%s "
            "eda_status=%s coverage=%.4f diversity=%s is_pass_xrun=%s is_pass_targets=%s",
            rollout_id, s.group_index, s.index, dataset_id,
            round_num, _loss_type, resp_len,
            s.status.name if s.status else "?",
            raw_reward, _advantage, _g_mean, _g_std,
            f"{_lp_mean:.4f}" if _lp_mean is not None else "N/A",
            _el.get("status", "?"),
            float(_el.get("overall_coverage", 0.0)),
            f"{_div:.4f}" if _div is not None else "n/a",
            _el.get("is_pass_xrun"),
            _el.get("is_pass_targets"),
        )
        # ── EDA feedback (only when --eda-log-feedback-train) ────────────
        _eda_fb = s.metadata.get("eda_feedback")
        if _eda_fb:
            logger.info("TRAIN_EDA_FEEDBACK step=%d idx=%s\n%s",
                        rollout_id, s.index, _eda_fb)

        # ── input (prompt) ─────────────────────────────────────────────────
        # Round-0 prompt comes verbatim from dataset (dataset_id is sufficient).
        # Round-1+ prompt = original msgs + pivot response + tool feedback,
        # all reconstructible from other logs. Full content only every
        # _MASK_LOG_INTERVAL steps to cap log volume.
        if log_mask_detail:
            if isinstance(s.prompt, str):
                prompt_repr = s.prompt
            else:
                import json as _json
                prompt_repr = _json.dumps(s.prompt, ensure_ascii=False)
            _prompt_log = (prompt_repr if len(prompt_repr) <= _MAX_LOG_CHARS
                           else prompt_repr[:_MAX_LOG_CHARS] + f"\n[TRUNCATED {len(prompt_repr)} chars]")
            logger.info("TRAIN_INPUT  step=%d idx=%s dataset_id=%s round=%d len=%d\n%s",
                        rollout_id, s.index, dataset_id, round_num, len(prompt_repr), _prompt_log)
        else:
            logger.info("TRAIN_INPUT  step=%d idx=%s dataset_id=%s round=%d (full every %d steps)",
                        rollout_id, s.index, dataset_id, round_num, _MASK_LOG_INTERVAL)

        # ── output (response) ─────────────────────────────────────────────
        _resp = s.response or ""
        _resp_log = _resp if len(_resp) <= _MAX_OUTPUT_LOG_CHARS else _resp[:_MAX_OUTPUT_LOG_CHARS] + f"\n[TRUNCATED {len(_resp)} chars]"
        logger.info("TRAIN_OUTPUT step=%d idx=%s\n%s",
                    rollout_id, s.index, _resp_log)

        # ── loss mask (every _MASK_LOG_INTERVAL steps) ────────────────────
        if log_mask_detail:
            mask = s.loss_mask or []
            n_active = sum(mask)
            n_zero   = len(mask) - n_active
            # summarise run-length: log first/last 8 values + counts
            preview  = (mask[:8] if len(mask) >= 8 else mask)
            logger.info(
                "TRAIN_MASK   step=%d idx=%s len=%d active=%d zero=%d preview=%s",
                rollout_id, s.index, len(mask), n_active, n_zero, preview,
            )

    # ── zero-std group summary ────────────────────────────────────────────
    _zero_std_gids = [
        gid for gid, rs in _group_raw.items()
        if max(rs) == min(rs)
        and all((_s.train_metadata or {}).get("loss_type", "rl") == "rl" for _s in _group_samples[gid])
    ]
    if _zero_std_gids:
        _zs_info = [
            f"{_group_datasets.get(gid,'?')}: reward={_group_raw[gid][0]:.4f}"
            for gid in sorted(_zero_std_gids)
        ]
        logger.info(
            "TRAIN_ZERO_STD step=%d zero_std=%d/%d (no gradient signal):\n  %s",
            rollout_id, len(_zero_std_gids), len(_group_raw),
            "\n  ".join(_zs_info),
        )

    _close_split_log()
    return False   # let default _log_rollout_data metrics still run
