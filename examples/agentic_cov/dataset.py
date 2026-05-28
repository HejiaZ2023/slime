"""Shared loader that turns an llm4cov HF dataset into slime ``Sample`` prompts.

Used by both the training ``LlmCovDataSource`` (via the data-source hook) and
the eval-only ``eval_rollout`` entry point. Keeping it in one place means the
two paths produce structurally identical samples — same chat template, same
metadata layout — even when they point at different HF datasets/splits.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from slime.utils.types import Sample

logger = logging.getLogger(__name__)


def build_samples_from_llm4cov(
    args: Any,
    tokenizer: Any,
    dataset_name: str,
    split: str,
) -> list[Sample]:
    """Return a list of un-cloned prompt-level ``Sample`` objects.

    Callers are responsible for cloning/grouping by ``n_samples_per_prompt``
    (the training path gets that for free from ``RolloutDataSource.get_samples``).
    """
    # Import lazily so slime workers that never touch llm4cov don't import it.
    from llm4cov.datasets.load import load_dataset_by_name
    from llm4cov.datasets.types import data_context_to_llm_gen_tb_context
    from llm4cov.llm_query.prompt_build import build_initial_prompt_from_context

    tokenizer_name = getattr(tokenizer, "name_or_path", None) or getattr(tokenizer, "vocab_files_names", {})
    logger.info("loading llm4cov dataset: %s split=%s  tokenizer=%s",
                dataset_name, split, tokenizer_name)
    t0 = time.time()
    contexts = load_dataset_by_name(dataset_name, split=split)
    logger.info("dataset fetched: %d contexts in %.1fs", len(contexts), time.time() - t0)

    t1 = time.time()
    samples: list[Sample] = []
    for raw_context in contexts:
        context = data_context_to_llm_gen_tb_context(raw_context)
        messages = build_initial_prompt_from_context(context)

        if args.apply_chat_template:
            prompt_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                **(args.apply_chat_template_kwargs or {}),
            )
        else:
            prompt_text = messages

        samples.append(
            Sample(
                prompt=prompt_text,
                metadata={
                    "llm4cov_context": context.model_dump(),
                    "initial_messages": messages,
                    "dataset_id": context.id,
                },
            )
        )

    logger.info(
        "Loaded %d llm4cov samples from %s split=%s  (prompt_build %.1fs, total %.1fs)",
        len(samples), dataset_name, split,
        time.time() - t1, time.time() - t0,
    )
    return samples
