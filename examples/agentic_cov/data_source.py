"""Custom slime DataSource that pulls prompts from llm4cov datasets.

Each sample carries the chat-templated initial prompt in ``sample.prompt`` and
stashes the raw ``LlmGenTbContext`` (needed later for remote coverage scoring)
plus the un-templated chat messages under ``sample.metadata``.
"""

from __future__ import annotations

import logging
import random
from typing import Any

from slime.rollout.data_source import RolloutDataSourceWithBuffer, pop_first
from slime.utils.misc import load_function
from slime.utils.processing_utils import load_tokenizer
from slime.utils.types import Sample

logger = logging.getLogger(__name__)


class _SimpleDataset:
    """Minimal stand-in for ``slime.utils.data.Dataset`` (only attrs used by RolloutDataSource)."""

    def __init__(self, samples: list[Sample], seed: int):
        self.origin_samples = samples
        self.samples = samples
        self.seed = seed
        self.epoch_id = -1

    def shuffle(self, new_epoch_id: int) -> None:
        if self.epoch_id == new_epoch_id:
            return
        rng = random.Random(self.seed + new_epoch_id)
        perm = list(range(len(self.origin_samples)))
        rng.shuffle(perm)
        self.samples = [self.origin_samples[i] for i in perm]
        self.epoch_id = new_epoch_id

    def __len__(self) -> int:
        return len(self.samples)


class LlmCovDataSource(RolloutDataSourceWithBuffer):
    """DataSource that builds prompt samples from an llm4cov HF dataset."""

    def __init__(self, args: Any):
        # Deliberately skip the parent __init__ — it expects a JSONL path.
        self.args = args
        self.epoch_id = 0
        self.sample_group_index = 0
        self.sample_index = 0
        self.sample_offset = 0
        self.metadata: dict = {}
        self.buffer: list[list[Sample]] = []
        self.buffer_filter = (
            load_function(args.buffer_filter_path) if getattr(args, "buffer_filter_path", None) else pop_first
        )

        tokenizer = load_tokenizer(args.hf_checkpoint, trust_remote_code=True)
        dataset_name = args.llm4cov_dataset_name
        split = args.llm4cov_dataset_split

        # Imported lazily so slime's worker ranks don't fail when llm4cov is absent.
        from llm4cov.datasets.load import load_dataset_by_name
        from llm4cov.datasets.types import data_context_to_llm_gen_tb_context
        from llm4cov.llm_query.prompt_build import build_initial_prompt_from_context

        contexts = load_dataset_by_name(dataset_name, split=split)

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
            "LlmCovDataSource loaded %d prompts from %s split=%s",
            len(samples),
            dataset_name,
            split,
        )

        self.dataset = _SimpleDataset(samples, seed=getattr(args, "rollout_seed", 42))
        if getattr(args, "rollout_shuffle", False):
            self.dataset.shuffle(self.epoch_id)
