import os
from pathlib import Path


os.environ.setdefault("OPD_SLIME_DIR", str(Path(__file__).resolve().parents[1]))

from examples.agentic_cov import opd_worker


def test_teacher_score_rejects_context_overflow_without_truncating(monkeypatch):
    monkeypatch.setattr(opd_worker, "TEACHER_CONTEXT_LENGTH", 12)
    monkeypatch.setattr(opd_worker, "TEACHER_SCORE_CONTEXT_MARGIN", 2)

    def _unexpected_request(*args, **kwargs):
        raise AssertionError("context overflow must not call the teacher endpoint")

    monkeypatch.setattr(opd_worker, "post_json", _unexpected_request)
    entry = {
        "id": "s0",
        "input_token_ids": list(range(11)),
        "response_token_count": 2,
        "topk_token_ids": [[1, 2], [3, 4]],
    }

    result = opd_worker.score_teacher_on_student(
        "stage0",
        entry,
        {"stage0": {"model_path": "unused"}},
        topk_k=2,
    )

    assert result["status"] == "context_overflow"
    assert result["teacher_log_probs"] == []
    assert result["teacher_score_timing"]["context_overflow"] is True
    assert result["teacher_score_timing"]["truncated_prefix_token_count"] == 0
    assert result["teacher_score_timing"]["full_input_token_count"] == 11
