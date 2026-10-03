"""Training: load one frozen dataset version, train, and evaluate per class.

Every frame and mask is fetched by its content hash from the locked datasets
bucket and its SHA-256 is checked on download, so the model is provably
trained on exactly the bytes listed in the dataset manifest.
"""
import concurrent.futures
import hashlib
import io
import time

import numpy as np
from PIL import Image

from .datasets import DATASETS, blob_key
from .labels import CLASS_NAMES, GRAY_TO_CLASS
from .lake import read_json

IGNORE = 255  # pixels with an unrecognized label are ignored in loss and metrics
NUM_CLASSES = len(CLASS_NAMES)
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)  # ImageNet statistics
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

LUT = np.full(256, IGNORE, dtype=np.uint8)  # gray value -> class id
for _gray, (_cid, _) in GRAY_TO_CLASS.items():
    LUT[_gray] = _cid


def resolve_dataset(s3, name, dataset_id=""):
    """Find a dataset release (latest if no ID given), pinned to its exact version."""
    if not dataset_id:
        dataset_id = read_json(s3, DATASETS, f"releases/{name}/latest.json")["dataset_id"]
    key = f"releases/{name}/{dataset_id}/manifest.json"
    version = s3.head_object(Bucket=DATASETS, Key=key).get("VersionId")
    manifest = read_json(s3, DATASETS, key)
    return manifest, key, version


def _pairs(manifest, split, max_frames):
    entries = [e for e in manifest["entries"] if e["split"] == split]
    masks = {e["path"]: e for e in entries if e["kind"] == "mask"}
    pairs = []
    for f in sorted((e for e in entries if e["kind"] == "frame"), key=lambda e: e["path"]):
        mask_path = f["path"].replace("/frames/", "/masks/").replace(".png", "_mask.png")
        if mask_path in masks:
            pairs.append((f, masks[mask_path]))
    if max_frames and len(pairs) > max_frames:  # evenly spaced, deterministic
        idx = np.linspace(0, len(pairs) - 1, max_frames).round().astype(int)
        pairs = [pairs[i] for i in idx]
    return pairs


def load_split(s3, manifest, split, height, width, max_frames=0, workers=16):
    """Download, verify, and resize one split into uint8 arrays."""
    pairs = _pairs(manifest, split, max_frames)

    def fetch(entry):
        data = s3.get_object(Bucket=DATASETS, Key=blob_key(entry["sha256"]))["Body"].read()
        if hashlib.sha256(data).hexdigest() != entry["sha256"]:
            raise ValueError(f"checksum mismatch for {entry['path']}")
        return data

    def load(pair):
        frame, mask = pair
        img = Image.open(io.BytesIO(fetch(frame))).convert("RGB")
        img = img.resize((width, height), Image.BILINEAR)
        lbl = Image.open(io.BytesIO(fetch(mask))).convert("L")
        lbl = lbl.resize((width, height), Image.NEAREST)
        return np.asarray(img, dtype=np.uint8), LUT[np.asarray(lbl, dtype=np.uint8)]

    with concurrent.futures.ThreadPoolExecutor(workers) as pool:
        loaded = list(pool.map(load, pairs))
    images = np.stack([x for x, _ in loaded])
    masks = np.stack([y for _, y in loaded])
    return images, masks


def build_model(pretrained_backbone=True):
    """LR-ASPP with a MobileNetV3 backbone: small and fast enough for CPU."""
    from torchvision.models import MobileNet_V3_Large_Weights
    from torchvision.models.segmentation import lraspp_mobilenet_v3_large

    weights = MobileNet_V3_Large_Weights.IMAGENET1K_V1 if pretrained_backbone else None
    return lraspp_mobilenet_v3_large(weights=None, weights_backbone=weights,
                                     num_classes=NUM_CLASSES)


def logits_only(model):
    """Wrap the model so it returns a plain tensor of class scores.

    torchvision's segmentation models return a dict ({"out": ...}); exported
    formats (MLflow pt2, ONNX) want a single tensor in and a single tensor out.
    """
    import torch

    class LogitsOnly(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, x):
            return self.inner(x)["out"]

    return LogitsOnly(model).eval()


def to_tensor(images):
    import torch
    x = (images.astype(np.float32) / 255.0 - MEAN) / STD
    return torch.from_numpy(x.transpose(0, 3, 1, 2).copy())


def train_model(model, images, masks, epochs, batch_size, lr, seed, on_epoch):
    import torch

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=IGNORE)
    for epoch in range(1, epochs + 1):
        model.train()
        order = rng.permutation(len(images))
        losses, start = [], time.time()
        for i in range(0, len(order), batch_size):
            b = order[i:i + batch_size]
            x, y = images[b], masks[b]
            if rng.random() < 0.5:  # simple augmentation: horizontal flip
                x, y = x[:, :, ::-1], y[:, :, ::-1]
            out = model(to_tensor(x))["out"]
            loss = loss_fn(out, torch.from_numpy(y.astype(np.int64)))
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        on_epoch(epoch, float(np.mean(losses)), time.time() - start)
    return model


def evaluate(model, images, masks, batch_size=16):
    """Per-class IoU, mean IoU, and pixel accuracy on one split."""
    import torch

    model.eval()
    conf = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            pred = model(to_tensor(images[i:i + batch_size]))["out"].argmax(1).numpy()
            gt = masks[i:i + batch_size]
            valid = gt != IGNORE
            conf += np.bincount(NUM_CLASSES * gt[valid].astype(np.int64) + pred[valid],
                                minlength=NUM_CLASSES ** 2).reshape(NUM_CLASSES, NUM_CLASSES)
    tp = np.diag(conf)
    denom = conf.sum(0) + conf.sum(1) - tp
    iou = np.where(denom > 0, tp / np.maximum(denom, 1), np.nan)
    present = conf.sum(1) > 0  # classes that appear in the ground truth
    return {
        "mean_iou": float(np.nanmean(iou[present])) if present.any() else 0.0,
        "pixel_accuracy": float(tp.sum() / max(conf.sum(), 1)),
        "per_class_iou": {c: (None if np.isnan(v) else round(float(v), 4))
                          for c, v in zip(CLASS_NAMES, iou)},
    }