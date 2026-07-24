from argparse import Namespace

import pytest

from examples.agentic_cov.rollout import (
    _apply_opd_gate_fail_action,
    _select_opd_routing_decision,
)
from slime.utils.types import Sample


TEACHER_NAMES = ["stage0", "stage1"]
TEACHER_ENTRIES = [
    {"id": "stage0_t000", "teacher": "stage0", "reward": 1.2},
    {"id": "stage0_t001", "teacher": "stage0", "reward": 1.6},
    {"id": "stage1_t000", "teacher": "stage1", "reward": 1.4},
]


def _decision(
    policy: str,
    *,
    entries=None,
    student_reward=1.0,
    student_rewards=None,
    rollout_id=7,
    gate_stat="best",
):
    return _select_opd_routing_decision(
        args=Namespace(
            opd_routing_policy=policy,
            opd_gate_eps=0.0,
            opd_gate_stat=gate_stat,
            opd_gate_fail_action="skip",
            opd_teachers="stage0:2,stage1:2",
            n_samples_per_prompt=4,
            seed=1234,
        ),
        teacher_entries=TEACHER_ENTRIES if entries is None else entries,
        teacher_names=TEACHER_NAMES,
        best_student_reward=student_reward,
        student_rewards=student_rewards,
        rollout_id=rollout_id,
        round_idx=1,
        dataset_id="task_17",
    )


def test_reward_gate_preserves_current_best_teacher_comparison():
    passed = _decision("reward_gate", student_reward=1.5)
    assert passed["gate_pass"] is True
    assert passed["arm"] == "stage0"
    assert passed["selected_teacher_name"] == "stage0"
    assert passed["reward_used_for_decision"] is True

    fallback = _decision("reward_gate", student_reward=1.6)
    assert fallback["gate_pass"] is False
    assert fallback["arm"] == "rl"
    assert fallback["selected_teacher_name"] == ""


def test_median_gate_selects_highest_teacher_median_not_global_best():
    entries = [
        {"id": "stage0_t000", "teacher": "stage0", "reward": 0.0},
        {"id": "stage0_t001", "teacher": "stage0", "reward": 1.9},
        {"id": "stage1_t000", "teacher": "stage1", "reward": 1.4},
        {"id": "stage1_t001", "teacher": "stage1", "reward": 1.4},
    ]
    decision = _decision(
        "reward_gate",
        entries=entries,
        student_reward=1.6,
        student_rewards=[1.0, 1.2, 1.4, 1.6],
        gate_stat="median",
    )

    assert decision["best_teacher_name"] == "stage0"
    assert decision["best_teacher_reward"] == pytest.approx(1.9)
    assert decision["student_gate_reward"] == pytest.approx(1.3)
    assert decision["teacher_gate_rewards"] == pytest.approx(
        {"stage0": 0.95, "stage1": 1.4}
    )
    assert decision["gate_teacher_name"] == "stage1"
    assert decision["selected_teacher_gate_reward"] == pytest.approx(1.4)
    assert decision["gate_pass"] is True
    assert decision["selected_teacher_name"] == "stage1"
    assert decision["gate_reason"] == "teacher_better"


def test_median_gate_rejects_equal_or_weaker_teacher():
    entries = [
        {"id": "stage0_t000", "teacher": "stage0", "reward": 1.2},
        {"id": "stage0_t001", "teacher": "stage0", "reward": 1.4},
        {"id": "stage1_t000", "teacher": "stage1", "reward": 1.3},
        {"id": "stage1_t001", "teacher": "stage1", "reward": 1.3},
    ]
    decision = _decision(
        "reward_gate",
        entries=entries,
        student_reward=1.6,
        student_rewards=[1.0, 1.2, 1.4, 1.6],
        gate_stat="median",
    )

    assert decision["student_gate_reward"] == pytest.approx(1.3)
    assert decision["selected_teacher_gate_reward"] == pytest.approx(1.3)
    assert decision["gate_pass"] is False
    assert decision["arm"] == "rl"
    assert decision["gate_reason"] == "teacher_not_better"


def test_median_gate_uses_configured_teacher_order_for_ties():
    entries = [
        {"id": "stage0_t000", "teacher": "stage0", "reward": 1.0},
        {"id": "stage0_t001", "teacher": "stage0", "reward": 2.0},
        {"id": "stage1_t000", "teacher": "stage1", "reward": 1.4},
        {"id": "stage1_t001", "teacher": "stage1", "reward": 1.6},
    ]
    decision = _decision(
        "reward_gate",
        entries=entries,
        student_reward=1.3,
        student_rewards=[1.0, 1.0, 1.2, 1.2],
        gate_stat="median",
    )
    assert decision["teacher_gate_rewards"] == pytest.approx(
        {"stage0": 1.5, "stage1": 1.5}
    )
    assert decision["gate_teacher_name"] == "stage0"
    assert decision["selected_teacher_name"] == "stage0"


def test_median_gate_missing_candidate_falls_back_to_rl_not_skip_reason():
    entries = [
        {"id": "stage0_t000", "teacher": "stage0", "reward": 1.2},
        {"id": "stage0_t001", "teacher": "stage0", "reward": 1.4},
        {"id": "stage1_t000", "teacher": "stage1", "reward": 2.0},
    ]
    decision = _decision(
        "reward_gate",
        entries=entries,
        student_reward=1.0,
        student_rewards=[0.8, 0.9, 1.0, 1.1],
        gate_stat="median",
    )
    assert decision["gate_pass"] is False
    assert decision["gate_reason"] == "median_inputs_invalid"
    assert decision["gate_invalid_reasons"] == ["stage1_count=1 expected=2"]


def test_skip_action_masks_only_valid_reward_gate_rejection():
    skipped = [Sample(metadata={}, train_metadata={"loss_type": "rl"}) for _ in range(4)]
    args = Namespace(opd_gate_fail_action="skip")

    assert _apply_opd_gate_fail_action(
        args=args,
        group=skipped,
        routing_policy="reward_gate",
        gate_reason="teacher_not_better",
    )
    assert all(sample.remove_sample for sample in skipped)
    assert all(sample.train_metadata["loss_type"] == "skip" for sample in skipped)

    fallback = [Sample(metadata={}, train_metadata={"loss_type": "rl"}) for _ in range(4)]
    assert not _apply_opd_gate_fail_action(
        args=args,
        group=fallback,
        routing_policy="reward_gate",
        gate_reason="median_inputs_invalid",
    )
    assert all(not sample.remove_sample for sample in fallback)
    assert all(sample.train_metadata["loss_type"] == "rl" for sample in fallback)


def test_always_best_removes_only_the_reward_gate():
    decision = _decision("always_best", student_reward=999.0)
    assert decision["gate_pass"] is True
    assert decision["arm"] == "stage0"
    assert decision["selected_teacher_name"] == "stage0"
    assert decision["best_teacher_reward"] == pytest.approx(1.6)
    assert decision["reward_used_for_decision"] is True


def test_random_teacher_is_deterministic_and_reward_independent():
    first = _decision("random_teacher")
    changed_rewards = [
        {"id": "stage0_t000", "teacher": "stage0", "reward": 99.0},
        {"id": "stage1_t000", "teacher": "stage1", "reward": -99.0},
    ]
    second = _decision("random_teacher", entries=changed_rewards)

    assert first["gate_pass"] is True
    assert first["arm"] in TEACHER_NAMES
    assert first["arm"] == second["arm"]
    assert first["random_choice_index"] == second["random_choice_index"]
    assert first["random_choice_hash"] == second["random_choice_hash"]
    assert first["reward_used_for_decision"] is False


def test_random_source_reaches_equal_weight_rl_and_teacher_arms():
    decisions = [_decision("random_source", rollout_id=rollout_id) for rollout_id in range(200)]
    assert {decision["arm"] for decision in decisions} == {"rl", "stage0", "stage1"}
    for decision in decisions:
        assert decision["random_choices"] == ["rl", "stage0", "stage1"]
        assert decision["gate_pass"] is (decision["arm"] != "rl")
        assert decision["selected_teacher_name"] == (
            "" if decision["arm"] == "rl" else decision["arm"]
        )
        assert decision["reward_used_for_decision"] is False


def test_random_teacher_deduplicates_teacher_names_without_using_rollout_count():
    decision = _select_opd_routing_decision(
        args=Namespace(opd_routing_policy="random_teacher", opd_gate_eps=0.0, seed=9),
        teacher_entries=TEACHER_ENTRIES,
        teacher_names=["stage0", "stage0", "stage1", "stage1"],
        best_student_reward=1.0,
        rollout_id=3,
        round_idx=0,
        dataset_id="task_3",
    )
    assert decision["random_choices"] == ["stage0", "stage1"]


def test_unknown_routing_policy_is_rejected():
    with pytest.raises(ValueError, match="unsupported OPD routing policy"):
        _decision("unknown")
