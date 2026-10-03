"""The release policy must block weak, regressing, slow, or untraceable models."""
from types import SimpleNamespace

from surgseg.gate import POLICY, check_edge, check_lineage, check_quality

GOOD = {"test_mean_iou": 0.43, "test_iou_liver": 0.78,
        "test_iou_gallbladder": 0.54, "test_iou_grasper": 0.43}


def failed(checks):
    return [c["check"] for c in checks if not c["passed"]]


def test_good_model_passes_quality():
    assert failed(check_quality(GOOD)) == []


def test_weak_model_fails_quality():
    weak = dict(GOOD, test_mean_iou=0.27, test_iou_gallbladder=0.2)
    assert set(failed(check_quality(weak))) == {"mean IoU", "IoU gallbladder"}


def test_missing_metrics_fail():
    assert "mean IoU" in failed(check_quality({}))


def test_regression_vs_approved_model_fails():
    assert "no regression vs approved" in failed(check_quality(GOOD, baseline_mean_iou=0.50))
    assert failed(check_quality(GOOD, baseline_mean_iou=0.44)) == []  # within tolerance


def test_latency_and_onnx_mismatch_fail():
    slow = {"p50": 20, "p95": POLICY["max_p95_latency_ms"] + 1, "p99": 60}
    assert failed(check_edge(0.9999, 1e-6, slow)) == ["latency p95"]
    fast = {"p50": 5, "p95": 8, "p99": 9}
    assert failed(check_edge(0.95, 1e-1, fast)) == ["ONNX matches PyTorch"]


def test_uncommitted_code_and_missing_lineage_fail(s3):
    run = SimpleNamespace(info=SimpleNamespace(status="FINISHED"),
                          data=SimpleNamespace(tags={"code.git_dirty": "True"}))
    names = failed(check_lineage(run, s3))
    assert "code committed" in names and "lineage tags present" in names