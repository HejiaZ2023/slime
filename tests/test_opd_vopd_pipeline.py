import os
from argparse import Namespace
from pathlib import Path

import torch


os.environ.setdefault("OPD_SLIME_DIR", str(Path(__file__).resolve().parents[1]))

from examples.agentic_cov.opd_remote_client import _hydrate_teacher_topk_sidecars, _load_teacher_topk_sidecar
from examples.agentic_cov.opd_worker import _write_teacher_topk_sidecar
from slime.backends.megatron_utils import data as megatron_data, loss as loss_module
from slime.backends.megatron_utils.loss import apply_vopd_topk_to_advantages, policy_loss_function
from slime.utils.opd import vopd_topk_statistics


def _log(values):
    return torch.log(torch.tensor(values, dtype=torch.float32))


def test_vopd_adjusts_zeroed_opd_advantage_with_sampled_term_minus_baseline():
    student_topk = _log([[0.5, 0.3, 0.2]])
    teacher_topk = _log([[0.2, 0.6, 0.2]])
    baseline = vopd_topk_statistics(student_topk, teacher_topk)["baseline_kl"]
    advantages = [torch.zeros(1)]
    rollout_data = {
        "rollout_log_probs": [torch.tensor([-0.7])],
        "teacher_log_probs": [torch.tensor([-1.0])],
        "teacher_logprob_masks": [torch.ones(1)],
        "opd_topk_student_log_probs": [student_topk],
        "opd_topk_teacher_log_probs": [teacher_topk],
        "opd_topk_masks": [torch.ones_like(student_topk)],
        "loss_types": ["opd"],
        "opd_weights": [2.0],
    }

    apply_vopd_topk_to_advantages(Namespace(opd_lambda=1.0, opd_kl_coef=0.0), rollout_data, advantages)

    expected = -2.0 * ((-0.7 - -1.0) - baseline)
    assert torch.allclose(advantages[0], expected, atol=1e-6)
    assert torch.allclose(rollout_data["opd_vopd_baseline_kl"][0], baseline, atol=1e-6)


def test_vopd_accepts_actor_forward_topk_without_rollout_values():
    actor_forward_topk = _log([[0.5, 0.3, 0.2]])
    teacher_topk = _log([[0.2, 0.6, 0.2]])
    baseline = vopd_topk_statistics(actor_forward_topk, teacher_topk)["baseline_kl"]
    advantages = [torch.zeros(1)]
    rollout_data = {
        "rollout_log_probs": [torch.tensor([-0.7])],
        "teacher_log_probs": [torch.tensor([-1.0])],
        "teacher_logprob_masks": [torch.ones(1)],
        "opd_topk_teacher_log_probs": [teacher_topk],
        "opd_topk_masks": [torch.ones_like(actor_forward_topk)],
        "loss_types": ["opd"],
        "opd_weights": [2.0],
    }

    apply_vopd_topk_to_advantages(
        Namespace(opd_lambda=1.0, opd_kl_coef=0.0),
        rollout_data,
        advantages,
        student_topk_log_probs=[actor_forward_topk],
    )

    expected = -2.0 * ((-0.7 - -1.0) - baseline)
    assert torch.allclose(advantages[0], expected, atol=1e-6)


def test_policy_loss_vopd_metrics_use_current_per_sample_log_probs(monkeypatch):
    actor_log_probs = [
        torch.tensor([-0.2], dtype=torch.float32, requires_grad=True),
        torch.tensor([-0.6], dtype=torch.float32, requires_grad=True),
    ]
    actor_topk = [_log([[0.6, 0.4]]), _log([[0.5, 0.5]])]
    teacher_topk = [_log([[0.6, 0.4]]), _log([[0.2, 0.8]])]

    def _get_log_probs_and_entropy(*_args, **_kwargs):
        return torch.empty(0), {
            "log_probs": actor_log_probs,
            "entropy": [torch.zeros_like(value) for value in actor_log_probs],
        }

    def _get_topk_log_probs(*_args, **_kwargs):
        return actor_topk

    def _get_sum_of_sample_mean(_total_lengths, response_lengths, loss_masks, *_args, **_kwargs):
        def _reduce(values):
            result = values.new_zeros(())
            for value, mask in zip(values.split(response_lengths, dim=0), loss_masks, strict=False):
                mask = mask.to(device=values.device, dtype=values.dtype)
                result = result + (value * mask).sum() / torch.clamp_min(mask.sum(), 1)
            return result

        return _reduce

    def _compute_policy_loss(ppo_kl, advantages, *_args, **_kwargs):
        return -torch.exp(-ppo_kl) * advantages, torch.zeros_like(ppo_kl)

    monkeypatch.setattr(loss_module, "get_log_probs_and_entropy", _get_log_probs_and_entropy)
    monkeypatch.setattr(loss_module, "get_topk_log_probs", _get_topk_log_probs)
    monkeypatch.setattr(loss_module, "get_sum_of_sample_mean", _get_sum_of_sample_mean)
    monkeypatch.setattr(loss_module, "compute_policy_loss", _compute_policy_loss)

    response_lengths = [1, 1]
    total_lengths = [2, 2]
    loss_masks = [torch.ones(1), torch.ones(1)]
    batch = {
        "response_lengths": response_lengths,
        "total_lengths": total_lengths,
        "loss_masks": loss_masks,
        "loss_types": ["rl", "opd"],
        "unconcat_tokens": [torch.tensor([1, 2]), torch.tensor([1, 2])],
        "advantages": [torch.ones(1), torch.zeros(1)],
        "log_probs": [torch.tensor([-0.3]), torch.tensor([-0.7])],
        "rollout_log_probs": [torch.tensor([-0.3]), torch.tensor([-0.7])],
        "teacher_log_probs": [torch.zeros(1), torch.tensor([-1.0])],
        "teacher_logprob_masks": [torch.zeros(1), torch.ones(1)],
        "opd_topk_token_ids": [torch.tensor([[0, 1]]), torch.tensor([[0, 1]])],
        "opd_topk_teacher_log_probs": teacher_topk,
        "opd_topk_masks": [torch.ones_like(value) for value in teacher_topk],
        "opd_weights": [0.0, 1.0],
    }
    args = Namespace(
        use_rollout_logprobs=False,
        opd_algorithm="vopd_topk",
        opd_lambda=1.0,
        opd_kl_coef=0.0,
        calculate_per_token_loss=False,
        qkv_format="thd",
        use_opsm=False,
        advantage_estimator="grpo",
        eps_clip=0.2,
        eps_clip_high=0.2,
        get_mismatch_metrics=False,
        use_tis=False,
        custom_pg_loss_reducer_function_path=None,
        entropy_coef=0.0,
        use_kl_loss=False,
    )
    reducer = _get_sum_of_sample_mean(total_lengths, response_lengths, loss_masks)

    loss, metrics = policy_loss_function(
        args,
        batch,
        logits=torch.zeros((1, sum(total_lengths), 2), dtype=torch.float32),
        sum_of_sample_mean=reducer,
    )

    assert torch.isfinite(loss)
    assert torch.allclose(metrics["opd_vopd_sampled_kl"], torch.tensor(0.3), atol=1e-6)
    assert torch.allclose(metrics["opd_vopd_support_coverage"], torch.tensor(1.0), atol=1e-6)
    assert torch.equal(batch["advantages"][1], torch.zeros(1))
    loss.backward()
    assert all(value.grad is not None for value in actor_log_probs)


def test_vopd_sidecar_v2_round_trip_preserves_sampled_vector(tmp_path):
    descriptor = _write_teacher_topk_sidecar(
        tmp_path,
        student_id="s0",
        teacher="stage0",
        score_idx=0,
        log_probs=[[-0.1, -2.0], [-0.2, -1.5]],
        masks=[[1.0, 1.0], [1.0, 1.0]],
        sampled_log_probs=[-0.1, -0.2],
    )

    topk, masks, sampled = _load_teacher_topk_sidecar(tmp_path, descriptor)

    assert descriptor["format"] == "npz_v2"
    assert torch.allclose(torch.tensor(topk), torch.tensor([[-0.1, -2.0], [-0.2, -1.5]]))
    assert masks == [[1.0, 1.0], [1.0, 1.0]]
    assert torch.allclose(torch.tensor(sampled), torch.tensor([-0.1, -0.2]))

    result = {"student_rollouts": [{"teacher_scores": [{"teacher_topk_sidecar": descriptor}]}]}
    _hydrate_teacher_topk_sidecars(result, tmp_path)
    hydrated = result["student_rollouts"][0]["teacher_scores"][0]
    assert torch.allclose(torch.tensor(hydrated["teacher_topk_log_probs"]), torch.tensor(topk))
    assert torch.allclose(torch.tensor(hydrated["teacher_log_probs"]), torch.tensor(sampled))


def test_v1_sidecar_remains_a_safe_rl_fallback_without_sampled_vector(tmp_path):
    descriptor = _write_teacher_topk_sidecar(
        tmp_path,
        student_id="s0",
        teacher="stage0",
        score_idx=0,
        log_probs=[[-0.1, -2.0]],
        masks=[[1.0, 1.0]],
    )

    _, _, sampled = _load_teacher_topk_sidecar(tmp_path, descriptor)

    assert descriptor["format"] == "npz_v1"
    assert sampled is None


def test_vopd_syncs_rollout_logprobs_without_unneeded_reference_broadcast(monkeypatch):
    calls = []

    class _Handle:
        def wait(self):
            return None

    def _broadcast(tensor, *, src, group, async_op):
        calls.append((src, tuple(tensor.shape)))
        return _Handle()

    monkeypatch.setattr(megatron_data.dist, "broadcast", _broadcast)
    rollout_data = {
        "values": [torch.zeros(2)],
        "rollout_log_probs": [torch.full((2,), -0.5)],
    }
    args = Namespace(
        use_opd_relay=True,
        opd_algorithm="vopd_topk",
        opd_topk=16,
        use_rollout_logprobs=False,
        kl_coef=0.0,
        use_kl_loss=False,
    )

    megatron_data.sync_actor_critic_data(args, rollout_data)

    assert [src for src, _ in calls] == [1, 0]
    assert "ref_log_probs" not in rollout_data
