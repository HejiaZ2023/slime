import importlib.util
from pathlib import Path
from types import SimpleNamespace


def _load_xfer_client():
    path = Path(__file__).resolve().parents[1] / "third_party/llm4cov_oss/src/llm4cov/eda_client/xfer_client.py"
    spec = importlib.util.spec_from_file_location("xfer_client_under_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _FakeTransport:
    def __init__(self):
        self.manifest = None
        self.cleaned = False
        self.closed = False

    def submit(self, job_id, inputs, manifest):
        self.manifest = manifest

    def poll_result(self, job_id, timeout):
        return {
            "overall_coverage": 0.5,
            "relay_timing": {
                "queue_wait_seconds": 0.2,
                "execution_seconds": 0.3,
                "result_publish_seconds": 0.01,
            },
        }

    def cleanup(self, job_id):
        self.cleaned = True

    def close(self):
        self.closed = True


def test_submit_cov_job_records_client_and_relay_timing(monkeypatch):
    xfer_client = _load_xfer_client()
    fake = _FakeTransport()
    monkeypatch.setattr(xfer_client, "make_transport", lambda server: fake)
    context = SimpleNamespace(
        dut_top_module_name="dut",
        rtl_files=[SimpleNamespace(name="rtl.sv", content="module dut; endmodule")],
    )
    tb_file = SimpleNamespace(name="tb.sv", content="module tb; endmodule")

    result = xfer_client.submit_cov_job("unused", "/workspace/eda", context, tb_file, timeout=1200)

    assert fake.cleaned and fake.closed
    assert fake.manifest["submitted_at_unix"] > 0
    timing = result["xfer_timing"]
    assert timing["transport"] == "_FakeTransport"
    assert timing["total_seconds"] >= timing["submit_seconds"] >= 0.0
    assert timing["wait_result_seconds"] >= 0.0
    assert result["relay_timing"]["queue_wait_seconds"] == 0.2
