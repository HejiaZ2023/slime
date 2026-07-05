"""Custom slime DataSource that pulls training prompts from an llm4cov dataset.

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

from .dataset import build_samples_from_llm4cov

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
    """DataSource that builds prompt samples from an llm4cov HF training dataset."""

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
        samples = build_samples_from_llm4cov(
            args=args,
            tokenizer=tokenizer,
            dataset_name=args.llm4cov_dataset_name,
            split=args.llm4cov_dataset_split,
        )

        self.dataset = _SimpleDataset(samples, seed=getattr(args, "rollout_seed", 42))
        if getattr(args, "rollout_shuffle", False):
            self.dataset.shuffle(self.epoch_id)
        self._apply_dataset_step_offset()

    def _prompts_per_rollout_step(self) -> int:
        batch_size = int(getattr(self.args, "rollout_batch_size", 0) or 0)
        num_rounds = int(getattr(self.args, "num_agentic_rounds", 1) or 1)
        if batch_size <= 0:
            raise ValueError(f"rollout_batch_size must be positive, got {batch_size}")
        if num_rounds <= 0:
            raise ValueError(f"num_agentic_rounds must be positive, got {num_rounds}")
        if batch_size % num_rounds != 0:
            raise ValueError(
                f"rollout_batch_size ({batch_size}) must be divisible by "
                f"num_agentic_rounds ({num_rounds}) to compute dataset step offset"
            )
        return batch_size // num_rounds

    def _apply_dataset_step_offset(self) -> None:
        step_offset = int(getattr(self.args, "llm4cov_dataset_step_offset", 0) or 0)
        if step_offset < 0:
            raise ValueError(f"llm4cov_dataset_step_offset must be >= 0, got {step_offset}")
        if step_offset == 0:
            return
        dataset_len = len(self.dataset)
        if dataset_len <= 0:
            raise ValueError("Cannot apply llm4cov dataset step offset to an empty dataset")

        prompts_per_step = self._prompts_per_rollout_step()
        skipped_prompt_groups = step_offset * prompts_per_step
        epoch_id, sample_offset = divmod(skipped_prompt_groups, dataset_len)

        self.epoch_id = epoch_id
        self.sample_offset = sample_offset
        if getattr(self.args, "rollout_shuffle", False):
            self.dataset.shuffle(self.epoch_id)

        self.metadata.update(
            {
                "llm4cov_dataset_step_offset": step_offset,
                "llm4cov_dataset_prompts_per_step": prompts_per_step,
                "llm4cov_dataset_skipped_prompt_groups": skipped_prompt_groups,
                "llm4cov_dataset_offset_epoch_id": self.epoch_id,
                "llm4cov_dataset_offset_sample_offset": self.sample_offset,
            }
        )
        logger.info(
            "applied llm4cov dataset step offset: steps=%d prompts_per_step=%d "
            "skipped_prompt_groups=%d dataset_len=%d epoch_id=%d sample_offset=%d seed=%s",
            step_offset,
            prompts_per_step,
            skipped_prompt_groups,
            dataset_len,
            self.epoch_id,
            self.sample_offset,
            getattr(self.dataset, "seed", None),
        )
