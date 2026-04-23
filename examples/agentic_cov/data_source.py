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
