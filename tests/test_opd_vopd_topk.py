import math

import pytest
import torch

from slime.utils.opd import vopd_topk_statistics


def _log(values):
    return torch.log(torch.tensor(values, dtype=torch.float32))


def test_vopd_baseline_is_zero_for_matching_conditional_topk_distributions():
    student = _log([[0.7, 0.2, 0.1], [0.6, 0.3, 0.1]])
    # These are valid full-vocabulary support masses below one, but the
    # conditional distribution on the saved support is identical.
    teacher = student - torch.tensor([[0.4], [0.7]], dtype=torch.float32)

    stats = vopd_topk_statistics(student, teacher)

    assert torch.allclose(stats["baseline_kl"], torch.zeros(2), atol=1e-6)
    assert torch.all(stats["student_support_mass"] > stats["teacher_support_mass"])


def test_vopd_baseline_matches_support_normalized_reverse_kl():
    student = _log([[0.5, 0.3, 0.2]])
    teacher = _log([[0.2, 0.6, 0.2]])

    stats = vopd_topk_statistics(student, teacher)

    expected = 0.5 * math.log(0.5 / 0.2) + 0.3 * math.log(0.3 / 0.6)
    assert torch.allclose(stats["baseline_kl"], torch.tensor([expected]), atol=1e-6)
    assert torch.allclose(stats["student_support_mass"], torch.ones(1), atol=1e-6)
    assert torch.allclose(stats["teacher_support_mass"], torch.ones(1), atol=1e-6)


def test_vopd_renormalizes_over_the_masked_common_support():
    student = _log([[0.8, 0.1, 0.1]])
    teacher = _log([[0.2, 0.6, 0.2]])
    mask = torch.tensor([[1.0, 1.0, 0.0]])

    stats = vopd_topk_statistics(student, teacher, mask)

    expected = (0.8 / 0.9) * math.log((0.8 / 0.9) / (0.2 / 0.8)) + (0.1 / 0.9) * math.log(
        (0.1 / 0.9) / (0.6 / 0.8)
    )
    assert torch.allclose(stats["baseline_kl"], torch.tensor([expected]), atol=1e-6)
    assert torch.allclose(stats["support_coverage"], torch.tensor([2.0 / 3.0]), atol=1e-6)


def test_vopd_rejects_empty_or_nonfinite_support():
    student = _log([[0.5, 0.5]])
    teacher = _log([[0.5, 0.5]])

    with pytest.raises(ValueError, match="at least one finite"):
        vopd_topk_statistics(student, teacher, torch.zeros_like(student))

    teacher[0, 1] = float("nan")
    with pytest.raises(ValueError, match="at least one finite"):
        vopd_topk_statistics(student, teacher, torch.tensor([[0.0, 1.0]]))
