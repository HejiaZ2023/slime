"""Multi-round agentic GRPO rollout for llm4cov testbench generation.

Contract:
* Each rollout step processes ``rollout_batch_size // K`` prompts where ``K``
  comes from ``--num-agentic-rounds`` (or ``--eval-num-agentic-rounds``).
* For each prompt we run ``K`` rounds in sequence. Within each round we sample
  a group of ``n_samples_per_prompt`` completions, score each one with the
  llm4cov EDA reward, and pick one winner:
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

from .reward import compute_reward

logger = logging.getLogger(__name__)

_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL | re.IGNORECASE)


def _strip_think(text: str) -> str:
    return _THINK_RE.sub("", text).strip()


def _build_tool_feedback(reward: float) -> str:
    """Build a user turn that feeds the EDA tool result back to the model."""
    if reward <= 0.0:
        return (
            "EDA tool feedback:\n"
            "- status: failed (simulation did not complete or produced no coverage)\n"
            "Please diagnose the likely cause and rewrite the testbench to maximise "
            "coverage. Follow the same output format (filename line + fenced "
            "systemverilog block)."
        )
    coverage = reward - 1.0
    return (
        "EDA tool feedback:\n"
        f"- status: success\n"
        f"- overall coverage: {coverage * 100:.2f}%\n"
        "This is the previous attempt. Try to push coverage higher by exercising "
        "more RTL paths. Same output format (filename line + fenced systemverilog block)."
    )


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


async def _generate_and_score_group(
    args: Namespace,
    state: GenerateState,
    group: list[Sample],
    sampling_params: dict[str, Any],
) -> list[Sample]:
    """Sample + score one GRPO group concurrently."""

    async def _one(sample: Sample) -> Sample:
        if sample.session_id is None:
            sample.session_id = str(uuid.uuid4())
        async with state.semaphore:
            await generate(args, sample, sampling_params.copy())
        sample.reward = await compute_reward(args, sample)
        return sample

    return await asyncio.gather(*[_one(s) for s in group])


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
) -> list[list[Sample]]:
    """Run K rounds for one seed prompt. Returns K groups."""
    assert num_rounds >= 1
    tokenizer = state.tokenizer
    apply_chat_template = bool(args.apply_chat_template)
    apply_chat_template_kwargs = getattr(args, "apply_chat_template_kwargs", {}) or {}

    base_messages = list(initial_group[0].metadata.get("initial_messages") or [])

    # Round 0 uses the already-prepared group (prompt already chat-templated by data source).
    rounds_output: list[list[Sample]] = []
    current_group = initial_group
    for round_idx in range(num_rounds):
        for s in current_group:
            s.metadata["round_number"] = round_idx

        current_group = await _generate_and_score_group(args, state, current_group, sampling_params)
        rounds_output.append(current_group)

        if round_idx + 1 >= num_rounds:
            break

        pivot = _select_pivot_sample(current_group, evaluation=evaluation)
        tool_feedback = _build_tool_feedback(float(pivot.reward or 0.0))
        base_messages = list(base_messages) + [
            {"role": "assistant", "content": _strip_think(pivot.response)},
            {"role": "user", "content": tool_feedback},
        ]
        next_prompt = (
            tokenizer.apply_chat_template(
                base_messages,
                tokenize=False,
                add_generation_prompt=True,
                **apply_chat_template_kwargs,
            )
            if apply_chat_template
            else list(base_messages)
        )

        next_group_index = group_index_allocator()
        base_sample = initial_group[0]
        start_index = sample_index_allocator(len(initial_group))
        current_group = _clone_prompt_into_group(
            base_sample=base_sample,
            prompt=next_prompt,
            messages_history=list(base_messages),
            group_index=next_group_index,
            start_index=start_index,
            n_samples=len(initial_group),
        )

    return rounds_output


async def _generate_rollout_async(
    args: Namespace,
    rollout_id: int,
    data_source: Any,
    evaluation: bool,
) -> RolloutFnTrainOutput | RolloutFnEvalOutput:
    state = GenerateState(args)

    if evaluation:
        num_rounds = int(getattr(args, "eval_num_agentic_rounds", 1) or 1)
    else:
        num_rounds = int(args.num_agentic_rounds)

    if num_rounds < 1:
        raise ValueError(f"num_agentic_rounds must be >= 1, got {num_rounds}")

    batch_size = args.rollout_batch_size
    if batch_size % num_rounds != 0:
        raise ValueError(
            f"rollout_batch_size ({batch_size}) must be divisible by num_agentic_rounds "
            f"({num_rounds}) so that K rounds * N prompts == batch_size."
        )
    num_prompts = batch_size // num_rounds

    # We pull num_prompts prompts; each returns num_rounds groups.
    initial_groups = data_source.get_samples(num_prompts)
    assert len(initial_groups) == num_prompts, (
        f"data_source returned {len(initial_groups)} prompt-groups, expected {num_prompts}"
    )

    # Allocators for fresh group_index / sample_index for rounds > 0.
    # The data_source already stamped the round-0 groups; we only need new ids
    # for the rounds we construct inline.
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
            evaluation=evaluation,
            group_index_allocator=allocate_group_index,
            sample_index_allocator=allocate_sample_indices,
        )

    all_rounds_per_prompt = await asyncio.gather(*[_one_prompt(g) for g in initial_groups])

    # Persist the updated allocators back on the data_source so subsequent
    # rollouts continue with non-colliding ids.
    data_source.sample_group_index = next_group_index["v"]
    data_source.sample_index = next_sample_index["v"]

    # Flatten: list[prompt][round] -> list[group] in prompt-major round-major order.
    flat_groups: list[list[Sample]] = []
    for rounds_list in all_rounds_per_prompt:
        flat_groups.extend(rounds_list)

    total_groups = len(flat_groups)
    assert total_groups == batch_size, (
        f"Produced {total_groups} groups, expected rollout_batch_size={batch_size}."
    )

    if evaluation:
        # Match slime's eval contract: dict of per-dataset results.
        flat_samples: list[Sample] = [s for grp in flat_groups for s in grp]
        reward_key = args.eval_reward_key or args.reward_key
        dataset_name = getattr(args, "llm4cov_dataset_name", "llm4cov")
        return RolloutFnEvalOutput(
            data={
                dataset_name: {
                    "rewards": [
                        s.reward if not reward_key else s.reward[reward_key] for s in flat_samples
                    ],
                    "truncated": [s.status == Sample.Status.TRUNCATED for s in flat_samples],
                    "samples": flat_samples,
                }
            }
        )

    # Sort groups by their first sample's index so downstream code sees a
    # deterministic, index-ordered batch.
    flat_groups.sort(key=lambda g: g[0].index if g and g[0].index is not None else 0)

    return RolloutFnTrainOutput(samples=flat_groups, metrics=None)


def generate_rollout(
    args: Namespace,
    rollout_id: int,
    data_source: Any,
    evaluation: bool = False,
) -> RolloutFnTrainOutput | RolloutFnEvalOutput:
    """Custom rollout entry point wired via ``--rollout-function-path``."""
    return run(_generate_rollout_async(args, rollout_id, data_source, evaluation))
