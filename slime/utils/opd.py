"""Small, dependency-free OPD math kernels shared by rollout and training."""

from __future__ import annotations

import torch


def vopd_topk_statistics(
    student_topk_log_probs: torch.Tensor,
    teacher_topk_log_probs: torch.Tensor,
    topk_mask: torch.Tensor | None = None,
    *,
    validate: bool = True,
) -> dict[str, torch.Tensor]:
    """Compute the detached top-k control-variate statistics for vOPD.

    The saved values are full-vocabulary log-probabilities evaluated on the
    student's top-k support.  ``log_softmax`` therefore renormalizes only over
    that common support, which is the vOPD control-variate baseline rather than
    the biased raw truncated-KL objective.
    """

    if student_topk_log_probs.ndim != 2 or teacher_topk_log_probs.ndim != 2:
        raise ValueError(
            "vOPD top-k log-probs must be rank-2, got "
            f"student={tuple(student_topk_log_probs.shape)} teacher={tuple(teacher_topk_log_probs.shape)}"
        )
    if student_topk_log_probs.shape != teacher_topk_log_probs.shape:
        raise ValueError(
            "vOPD top-k shape mismatch: "
            f"student={tuple(student_topk_log_probs.shape)} teacher={tuple(teacher_topk_log_probs.shape)}"
        )

    student = student_topk_log_probs.to(dtype=torch.float32)
    teacher = teacher_topk_log_probs.to(device=student.device, dtype=torch.float32)
    if topk_mask is None:
        valid = torch.ones_like(student, dtype=torch.bool)
    else:
        if topk_mask.shape != student.shape:
            raise ValueError(
                "vOPD top-k mask shape mismatch: "
                f"mask={tuple(topk_mask.shape)} expected={tuple(student.shape)}"
            )
        valid = topk_mask.to(device=student.device, dtype=torch.bool)

    finite = torch.isfinite(student) & torch.isfinite(teacher)
    valid = valid & finite
    if validate and not bool(valid.any(dim=-1).all()):
        raise ValueError("vOPD needs at least one finite teacher/student top-k score per response position")

    neg_inf = torch.full_like(student, -float("inf"))
    student_cond_log_probs = torch.log_softmax(torch.where(valid, student, neg_inf), dim=-1)
    teacher_cond_log_probs = torch.log_softmax(torch.where(valid, teacher, neg_inf), dim=-1)
    student_cond_probs = student_cond_log_probs.exp()
    baseline_terms = torch.where(
        valid,
        student_cond_probs * (student_cond_log_probs - teacher_cond_log_probs),
        torch.zeros_like(student_cond_probs),
    )

    return {
        "baseline_kl": baseline_terms.sum(dim=-1),
        "student_support_mass": torch.where(valid, student.exp(), torch.zeros_like(student)).sum(dim=-1),
        "teacher_support_mass": torch.where(valid, teacher.exp(), torch.zeros_like(teacher)).sum(dim=-1),
        "support_coverage": valid.to(dtype=torch.float32).mean(dim=-1),
    }
