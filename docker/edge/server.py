"""Edge inference server: serves ONE pinned model version, verified at startup.

It refuses to start unless the version
  1. passed the release gate,
  2. was approved by a person, and
  3. downloads to an ONNX file whose SHA-256 matches the checksum the release
     gate recorded.
So a device can never run an unapproved or tampered model. During a rollout,
a refusing pod never becomes ready, and Kubernetes keeps the old pod serving.

Endpoints:
  GET  /health   200 once the verified model is loaded
  GET  /model    what is running here: version, lineage, approval, latency stats
  POST /predict  image bytes in -> class mask (base64 PNG), class shares, latency

Config (environment): MLFLOW_TRACKING_URI, MODEL_NAME, MODEL_VERSION, SITE_ID
"""
import base64
import collections
import glob
import hashlib
import io
import os
import sys
import time

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, HTTPException, Request
from mlflow import MlflowClient
from mlflow.artifacts import download_artifacts
from PIL import Image

MODEL_NAME = os.environ.get("MODEL_NAME", "cholecseg-segmenter")
MODEL_VERSION = os.environ.get("MODEL_VERSION", "")
SITE_ID = os.environ.get("SITE_ID", "unknown-site")
THREADS = int(os.environ.get("ORT_THREADS", "2"))

CLASS_NAMES = ["background", "abdominal_wall", "liver", "gastrointestinal_tract", "fat",
               "grasper", "connective_tissue", "blood", "cystic_duct",
               "l_hook_electrocautery", "gallbladder", "hepatic_vein", "liver_ligament"]
PALETTE = np.array([[0, 0, 0], [140, 90, 60], [200, 40, 40], [230, 160, 60],
                    [240, 220, 120], [60, 200, 230], [170, 130, 200], [120, 0, 0],
                    [40, 220, 90], [230, 60, 200], [80, 160, 40], [40, 80, 220],
                    [250, 250, 250]], dtype=np.uint8)
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def load_verified_model():
    if not MODEL_VERSION:
        raise RuntimeError("MODEL_VERSION is not set; devices must pin an exact version")
    client = MlflowClient()
    tags = client.get_model_version(MODEL_NAME, MODEL_VERSION).tags
    problems = []
    if tags.get("gate.decision") != "passed":
        problems.append("did not pass the release gate")
    if tags.get("approval.status") != "approved":
        problems.append(f"not approved (approval.status={tags.get('approval.status')})")
    expected = tags.get("edge_model.sha256")
    if not expected:
        problems.append("no checksum recorded by the release gate")
    if problems:
        raise RuntimeError("; ".join(problems))

    local_dir = download_artifacts(f"models:/{MODEL_NAME}/{MODEL_VERSION}")
    onnx_files = glob.glob(os.path.join(local_dir, "**", "*.onnx"), recursive=True)
    if len(onnx_files) != 1:
        raise RuntimeError(f"expected one .onnx file, found {onnx_files}")
    data = open(onnx_files[0], "rb").read()
    actual = hashlib.sha256(data).hexdigest()
    if actual != expected:
        raise RuntimeError(f"checksum mismatch: expected {expected[:12]}, got {actual[:12]}")

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = THREADS
    opts.inter_op_num_threads = 1
    session = ort.InferenceSession(data, opts, providers=["CPUExecutionProvider"])
    _, _, height, width = (int(n) for n in tags["input_shape"].split("x"))
    info = {
        "site": SITE_ID,
        "model": MODEL_NAME,
        "version": MODEL_VERSION,
        "onnx_sha256": actual,
        "input_shape": tags["input_shape"],
        "dataset_id": tags.get("dataset.id"),
        "code_git_commit": tags.get("code.git_commit"),
        "trained_by": tags.get("trained_by"),
        "approved_by": tags.get("approval.by"),
        "approved_at": tags.get("approval.at"),
        "gate_policy": tags.get("gate.policy"),
        "gate_latency_p95_ms": tags.get("latency_p95_ms"),
        "loaded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    return session, (height, width), info


try:
    SESSION, (HEIGHT, WIDTH), INFO = load_verified_model()
    print(f"[{SITE_ID}] serving {MODEL_NAME} v{MODEL_VERSION} "
          f"(sha256 {INFO['onnx_sha256'][:12]}, approved by {INFO['approved_by']})", flush=True)
except Exception as e:
    print(f"[{SITE_ID}] REFUSING TO SERVE {MODEL_NAME} v{MODEL_VERSION}: {e}",
          file=sys.stderr, flush=True)
    sys.exit(1)

LATENCIES = collections.deque(maxlen=500)
app = FastAPI(title=f"edge inference ({SITE_ID})")


@app.get("/health")
def health():
    return {"status": "ok", "version": MODEL_VERSION}


@app.get("/model")
def model():
    stats = {"requests": len(LATENCIES)}
    if LATENCIES:
        lat = np.array(LATENCIES)
        stats.update(p50_ms=round(float(np.percentile(lat, 50)), 2),
                     p95_ms=round(float(np.percentile(lat, 95)), 2))
    return {**INFO, "latency_stats": stats}


@app.post("/predict")
async def predict(request: Request):
    try:
        img = Image.open(io.BytesIO(await request.body())).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="body must be a PNG or JPEG image")
    original_size = img.size
    x = np.asarray(img.resize((WIDTH, HEIGHT), Image.BILINEAR), dtype=np.float32) / 255.0
    x = ((x - MEAN) / STD).transpose(2, 0, 1)[None].astype(np.float32)

    start = time.perf_counter()
    logits = SESSION.run(None, {"image": x})[0]
    latency_ms = (time.perf_counter() - start) * 1000
    LATENCIES.append(latency_ms)

    classes = logits[0].argmax(0).astype(np.uint8)
    counts = np.bincount(classes.ravel(), minlength=len(CLASS_NAMES))
    shares = {n: round(float(c) / classes.size, 4)
              for n, c in zip(CLASS_NAMES, counts) if c > 0}
    mask = Image.fromarray(PALETTE[classes]).resize(original_size, Image.NEAREST)
    buf = io.BytesIO()
    mask.save(buf, format="PNG")
    return {
        "site": SITE_ID,
        "model_version": MODEL_VERSION,
        "latency_ms": round(latency_ms, 2),
        "class_shares": shares,
        "mask_png_base64": base64.b64encode(buf.getvalue()).decode(),
    }