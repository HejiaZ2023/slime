from argparse import Namespace

import pytest

from examples.agentic_cov.rollout import _select_opd_routing_decision


TEACHER_NAMES = ["stage0", "stage1"]
TEACHER_ENTRIES = [
    {"id": "stage0_t000", "teacher": "stage0", "reward": 1.2},
    {"id": "stage0_t001", "teacher": "stage0", "reward": 1.6},
    {"id": "stage1_t000", "teacher": "stage1", "reward": 1.4},
]


def _decision(policy: str, *, entries=None, student_reward=1.0, rollout_id=7):
    return _select_opd_routing_decision(
        args=Namespace(opd_routing_policy=policy, opd_gate_eps=0.0, seed=1234),
        teacher_entries=TEACHER_ENTRIES if entries is None else entries,
        teacher_names=TEACHER_NAMES,
        best_student_reward=student_reward,
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
