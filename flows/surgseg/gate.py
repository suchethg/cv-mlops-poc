"""Release gate: the rules a trained model must pass before it can ship.

The policy is code, versioned in git: changing a threshold is a reviewed
commit, and every gate decision records which policy version it used.

Checks, in order:
  lineage   the run finished, knows its dataset and exact code, and its
            dataset manifest is still under object lock
  quality   meets minimum IoU for every class in the intended use, plus a
            minimum mean IoU, and doesn't regress vs the approved model
  edge      exports to ONNX, matches the PyTorch model, and meets the latency
            budget on edge-like CPU resources
"""
import time

import numpy as np

POLICY = {
    "version": "release-policy-v1",
    "registered_model": "cholecseg-segmenter",
    # What the model is claimed to do. Only these classes are gated; other
    # classes are reported but explicitly not claimed in v1.
    "intended_use": ("Research POC: overlay of liver, gallbladder and grasper "
                     "in laparoscopic cholecystectomy video. Not for clinical use."),
    "min_mean_iou": 0.40,
    "min_class_iou": {"liver": 0.70, "gallbladder": 0.50, "grasper": 0.40},
    "max_mean_iou_regression": 0.02,  # vs the currently approved model
    "max_p95_latency_ms": 33.0,       # one frame within a 30 fps video frame
    "min_onnx_agreement": 0.999,      # share of pixels where ONNX == PyTorch
    "edge_cpu_threads": 2,
}
REQUIRED_TAGS = ["dataset.id", "dataset.manifest_key", "dataset.manifest_version",
                 "code.git_commit", "code.git_dirty", "metaflow.pathspec", "runtime.image"]


def check(name, passed, detail):
    return {"check": name, "passed": bool(passed), "detail": detail}


def check_lineage(run, s3, datasets_bucket="datasets"):
    tags = run.data.tags
    out = [check("run finished", run.info.status == "FINISHED", run.info.status)]
    missing = [t for t in REQUIRED_TAGS if not tags.get(t)]
    out.append(check("lineage tags present", not missing,
                     f"missing {missing}" if missing else "all present"))
    out.append(check("code committed", tags.get("code.git_dirty") == "False",
                     f"git {tags.get('code.git_commit', '?')[:12]}, "
                     f"dirty={tags.get('code.git_dirty')}"))
    if tags.get("dataset.manifest_key"):
        try:
            ret = s3.get_object_retention(Bucket=datasets_bucket,
                                          Key=tags["dataset.manifest_key"],
                                          VersionId=tags["dataset.manifest_version"])
            mode = ret["Retention"]["Mode"]
            out.append(check("dataset manifest locked", mode in ("GOVERNANCE", "COMPLIANCE"),
                             f"{tags['dataset.id']}: {mode} until "
                             f"{ret['Retention']['RetainUntilDate']}"))
        except Exception as e:  # missing version, no retention, ...
            out.append(check("dataset manifest locked", False, f"{type(e).__name__}: {e}"))
    return out


def check_quality(metrics, baseline_mean_iou=None, policy=POLICY):
    out = []
    miou = metrics.get("test_mean_iou")
    out.append(check("mean IoU", miou is not None and miou >= policy["min_mean_iou"],
                     f"{miou if miou is None else round(miou, 3)} >= {policy['min_mean_iou']}"))
    for cls, minimum in policy["min_class_iou"].items():
        v = metrics.get(f"test_iou_{cls}")
        out.append(check(f"IoU {cls}", v is not None and v >= minimum,
                         f"{v if v is None else round(v, 3)} >= {minimum}"))
    if baseline_mean_iou is None:
        out.append(check("no regression vs approved", True, "no approved model yet"))
    else:
        floor = baseline_mean_iou - policy["max_mean_iou_regression"]
        out.append(check("no regression vs approved", miou is not None and miou >= floor,
                         f"{miou if miou is None else round(miou, 3)} >= {round(floor, 3)} "
                         f"(approved {round(baseline_mean_iou, 3)})"))
    return out


def export_onnx(model, height, width, path):
    import torch
    torch.onnx.export(model, (torch.randn(1, 3, height, width),), path,
                      input_names=["image"], output_names=["logits"], dynamo=True)
    return path


def onnx_agreement(model, onnx_path, images):
    """Share of pixels where the ONNX model predicts the same class as PyTorch."""
    import onnxruntime as ort
    import torch

    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    same, total, max_diff = 0, 0, 0.0
    for x in images:
        x = x[None]
        a = model(torch.from_numpy(x)).detach().numpy()
        b = session.run(None, {"image": x})[0]
        same += int((a.argmax(1) == b.argmax(1)).sum())
        total += a.shape[0] * a.shape[2] * a.shape[3]
        max_diff = max(max_diff, float(np.abs(a - b).max()))
    return same / max(total, 1), max_diff


def benchmark(onnx_path, height, width, threads, runs=200, warmup=20):
    """Single-frame latency with a fixed number of CPU threads."""
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads
    opts.inter_op_num_threads = 1
    session = ort.InferenceSession(onnx_path, opts, providers=["CPUExecutionProvider"])
    x = np.random.default_rng(0).standard_normal((1, 3, height, width)).astype(np.float32)
    for _ in range(warmup):
        session.run(None, {"image": x})
    times = []
    for _ in range(runs):
        t = time.perf_counter()
        session.run(None, {"image": x})
        times.append((time.perf_counter() - t) * 1000)
    return {f"p{p}": round(float(np.percentile(times, p)), 2) for p in (50, 95, 99)}


def check_edge(agreement, max_diff, latency, policy=POLICY):
    return [
        check("ONNX matches PyTorch", agreement >= policy["min_onnx_agreement"],
              f"{agreement:.5f} pixel agreement (max logit diff {max_diff:.2e})"),
        check("latency p95", latency["p95"] <= policy["max_p95_latency_ms"],
              f"{latency['p95']} ms <= {policy['max_p95_latency_ms']} ms "
              f"(p50 {latency['p50']}, p99 {latency['p99']}, "
              f"{policy['edge_cpu_threads']} threads)"),
    ]