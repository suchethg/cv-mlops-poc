"""Freezing curated data into immutable, content-addressed dataset versions.

Layout in the locked `datasets` bucket:
  blobs/sha256/<ab>/<hash>                 every file, stored once by content hash
  _golden/test-cases.json                  the frozen golden test set
  releases/<name>/<dataset_id>/manifest.json
  releases/<name>/<dataset_id>/DATASET_CARD.md
  releases/<name>/latest.json              pointer to the newest release

The dataset ID is a hash of the manifest content: if any file, split, or label
changed, the ID would change. Same content -> same ID, so re-releasing
identical data is detected instead of duplicated.
"""
import datetime
import hashlib
import io
import json

from PIL import Image

from .labels import CLASS_NAMES, GRAY_TO_CLASS
from .lake import exists, list_keys, read_json, write_json

CURATED, DATASETS = "curated", "datasets"
GOLDEN_KEY = "_golden/test-cases.json"
SCHEMA = "surgseg-dataset-manifest/v1"


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def blob_key(digest):
    return f"blobs/sha256/{digest[:2]}/{digest}"


def list_curated_clips(s3):
    """Every curated clip, pinned to the exact version of its record."""
    clips = []
    for key in list_keys(s3, CURATED, "cases/"):
        if key.endswith("/record.json"):
            head = s3.head_object(Bucket=CURATED, Key=key)
            rec = read_json(s3, CURATED, key)
            clips.append({
                "case_id": rec["case_id"],
                "clip": rec["clip"],
                "patient_pseudonym": rec["patient_pseudonym"],
                "site_id": rec["site_id"],
                "label_source": rec.get("label_source"),
                "record_key": key,
                "record_version": head.get("VersionId"),
            })
    return sorted(clips, key=lambda c: (c["case_id"], c["clip"]))


def golden_test_cases(s3, all_cases, test_fraction, run_pathspec):
    """Return the frozen golden test cases, creating them on the first release.

    Once written, the golden set never changes, so every model is compared on
    the same held-out cases. New cases that arrive later go to training.
    """
    if exists(s3, DATASETS, GOLDEN_KEY):
        return read_json(s3, DATASETS, GOLDEN_KEY)["test_cases"], False
    ranked = sorted(all_cases, key=lambda c: sha256(c.encode()))  # deterministic shuffle
    n_test = max(1, round(len(ranked) * test_fraction))
    test_cases = sorted(ranked[:n_test])
    write_json(s3, DATASETS, GOLDEN_KEY, {
        "test_cases": test_cases,
        "created_by": run_pathspec,
        "created_at": now(),
        "rule": f"first {n_test} of {len(ranked)} cases ranked by sha256(case_id)",
    })
    return test_cases, True


def check_no_leakage(clips, test_cases):
    """No patient may appear in both train and test."""
    train_pts = {c["patient_pseudonym"] for c in clips if c["case_id"] not in test_cases}
    test_pts = {c["patient_pseudonym"] for c in clips if c["case_id"] in test_cases}
    overlap = train_pts & test_pts
    if overlap:
        raise ValueError(f"patient leakage between splits: {sorted(overlap)}")


def freeze_case(s3, clips, split):
    """Verify every file of a case, store it by content hash, gather label stats."""
    entries, pixel_counts, unknown = [], {}, {}
    for clip in clips:
        rec = json.loads(s3.get_object(Bucket=CURATED, Key=clip["record_key"],
                                       VersionId=clip["record_version"])["Body"].read())
        for f in rec["files"]:
            data = s3.get_object(Bucket=CURATED, Key=f["key"])["Body"].read()
            digest = sha256(data)
            if digest != f["sha256"]:
                raise ValueError(f"curated file changed since ingest: {f['key']}")
            if not exists(s3, DATASETS, blob_key(digest)):
                s3.put_object(Bucket=DATASETS, Key=blob_key(digest), Body=data)
            name = f["key"].rsplit("/", 1)[1]
            entries.append({
                "path": f"{split}/{clip['case_id']}/{clip['clip']}/{f['kind']}s/{name}",
                "sha256": digest,
                "kind": f["kind"],
                "split": split,
                "case_id": clip["case_id"],
                "clip": clip["clip"],
            })
            if f["kind"] == "mask":
                gray = Image.open(io.BytesIO(data)).convert("L")
                for count, value in gray.getcolors(maxcolors=256) or []:
                    if value in GRAY_TO_CLASS:
                        name_ = GRAY_TO_CLASS[value][1]
                        pixel_counts[name_] = pixel_counts.get(name_, 0) + count
                    else:
                        unknown[str(value)] = unknown.get(str(value), 0) + count
    return entries, pixel_counts, unknown


def dataset_id(name, entries, test_cases, label_source):
    """Content hash of everything that defines the dataset."""
    core = {"schema": SCHEMA, "name": name, "label_source": label_source,
            "test_cases": test_cases,
            "entries": [(e["path"], e["sha256"]) for e in entries]}
    canonical = json.dumps(core, sort_keys=True, separators=(",", ":")).encode()
    return "ds-" + sha256(canonical)[:16]


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def summarize(entries, pixel_counts):
    splits = {}
    for e in entries:
        s = splits.setdefault(e["split"], {"frames": 0, "cases": set(), "clips": set()})
        if e["kind"] == "frame":
            s["frames"] += 1
        s["cases"].add(e["case_id"])
        s["clips"].add(e["clip"])
    total = sum(pixel_counts.values()) or 1
    return {
        "splits": {k: {"frames": v["frames"], "cases": sorted(v["cases"]),
                       "clips": len(v["clips"])} for k, v in sorted(splits.items())},
        "class_pixel_share": {c: round(pixel_counts.get(c, 0) / total, 4)
                              for c in CLASS_NAMES},
    }


def dataset_card(manifest):
    s = manifest["summary"]
    lines = [
        f"# Dataset {manifest['name']} / {manifest['dataset_id']}",
        "",
        f"- Created: {manifest['created_at']} by `{manifest['release_run']}`",
        f"- Source: curated bucket, label source `{manifest['label_source']}`",
        "- License of underlying data: CC BY-NC-SA 4.0 (CholecSeg8k)",
        f"- Files: {len(manifest['entries'])}, stored by SHA-256 in `datasets/blobs/`",
        "- Immutable: object lock on the datasets bucket; the ID is a hash of the content",
        "",
        "## Splits (by case, never by frame)",
        "",
        "| Split | Frames | Clips | Cases |",
        "|---|---|---|---|",
    ]
    for name, v in s["splits"].items():
        lines.append(f"| {name} | {v['frames']} | {v['clips']} | {', '.join(v['cases'])} |")
    lines += ["", "Test cases are the frozen golden set, reused by every release.", "",
              "## Class pixel share", "", "| Class | Share |", "|---|---|"]
    lines += [f"| {c} | {v:.2%} |" for c, v in s["class_pixel_share"].items()]
    if manifest.get("unknown_mask_values"):
        lines += ["", f"Unrecognized mask values (pixel counts): {manifest['unknown_mask_values']}"]
    return "\n".join(lines) + "\n"


def publish(s3, manifest):
    """Write manifest, card and latest pointer. Returns False if already released."""
    base = f"releases/{manifest['name']}/{manifest['dataset_id']}"
    if exists(s3, DATASETS, f"{base}/manifest.json"):
        return False
    write_json(s3, DATASETS, f"{base}/manifest.json", manifest)
    s3.put_object(Bucket=DATASETS, Key=f"{base}/DATASET_CARD.md",
                  Body=dataset_card(manifest).encode(), ContentType="text/markdown")
    write_json(s3, DATASETS, f"releases/{manifest['name']}/latest.json", {
        "dataset_id": manifest["dataset_id"],
        "manifest": f"{base}/manifest.json",
        "created_at": manifest["created_at"],
    })
    return True