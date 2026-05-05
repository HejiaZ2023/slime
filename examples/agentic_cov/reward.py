"""Remote EDA-based reward evaluation for llm4cov multi-round rollouts.

Reward = 0 when the generated testbench cannot be extracted / fails xrun.
Reward = 1 + ``Overall Average`` coverage when the job succeeds.

The coverage job is slow (SSH + remote simulation), so we run it in a worker
thread via ``asyncio.to_thread`` to avoid blocking the event loop. Each sample
is keyed by a unique ``DataFile`` name to avoid collisions on the remote host.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Any

logger = logging.getLogger(__name__)

_DEFAULT_TB_FILENAME = "tb_generated.sv"


def _parse_testbench(response: str) -> tuple[str | None, str | None]:
    """Return (filename, verilog_body) parsed from an LLM completion."""
    from llm4cov.llm_query.parse import extract_filename_from_text, extract_verilog_content

    body = extract_verilog_content(response)
    if body is None:
        return None, None
    filename = extract_filename_from_text(response) or _DEFAULT_TB_FILENAME
    # Guarantee filename uniqueness so concurrent remote jobs don't collide.
    stem, _, ext = filename.rpartition(".")
    if not stem:
        stem, ext = filename, "sv"
    unique = f"{stem}_{uuid.uuid4().hex[:8]}.{ext}"
    return unique, body


def _rehydrate_context(context_dict: dict[str, Any]):
    from llm4cov.datasets.types import LlmGenTbContext

    return LlmGenTbContext(**context_dict)


def _compute_reward_sync(args: Any, context_dict: dict[str, Any], response: str) -> float:
    """Synchronous reward path — runs the SSH coverage job on the calling thread."""
    filename, body = _parse_testbench(response)
    if filename is None or body is None:
        return 0.0

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
            skip_detail=True,
            timeout=getattr(args, "eda_stage_timeout", 30),
        )
    except Exception as exc:
        logger.warning("Remote EDA job failed for %s: %s", context.id, exc)
        return 0.0

    cov_result = eval_cov_result_against_expectations(context, result)
    if not cov_result.has_coverage:
        return 0.0
    return 1.0 + float(cov_result.overall_coverage)


async def compute_reward(args: Any, sample: Any) -> float:
    """Async reward entry point. ``sample.metadata['llm4cov_context']`` must exist."""
    context_dict = sample.metadata.get("llm4cov_context")
    if context_dict is None:
        logger.warning("Sample %s is missing llm4cov_context metadata", sample.index)
        return 0.0
    return await asyncio.to_thread(_compute_reward_sync, args, context_dict, sample.response)
