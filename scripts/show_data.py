"""Show the data at each stage of the pipeline, for the demo recording.

Read-only: it lists and prints what is in the data lake (and on disk), and
saves preview images to out/. It never changes anything.

  python scripts/show_data.py source --video video43 --clips 3
      the local test clips about to be uploaded: frames, resolution, classes
  python scripts/show_data.py quarantine
      uploads waiting in quarantine, and one manifest (PHI and all)
  python scripts/show_data.py redaction
      one frame before (quarantine, with PHI) and after (curated, redacted)
  python scripts/show_data.py curated
      every curated case and clip, and one curated record (no PHI)
  python scripts/show_data.py rejected
      every rejection record
  python scripts/show_data.py dataset
      the newest dataset release: card, sample manifest entries, locked blobs
  python scripts/show_data.py links <stage> [--run-id ID] [--version N]
      browser URLs that show the same data (one per line, "label|url"):
      data lake console folders, the files themselves, Metaflow and MLflow runs

Add --open to open saved images in Preview (macOS).
"""
import argparse
import glob
import io
import json
import os
import subprocess
import sys
from pathlib import Path

from urllib.parse import quote

from PIL import Image, ImageDraw

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "flows"))
from surgseg.datasets import blob_key  # noqa: E402
from surgseg.gate import POLICY  # noqa: E402
from surgseg.ingest import pending_uploads  # noqa: E402
from surgseg.labels import GRAY_TO_CLASS  # noqa: E402
from surgseg.lake import lake_client, list_keys, read_json  # noqa: E402

DATA_ROOT = Path("data/raw/CholecSeg8k")
CONSOLE = "http://localhost:9101"
METAFLOW_UI = "http://localhost:3000"
MLFLOW_UI = "http://localhost:5050"
CONTENT_TYPES = {".png": "image/png", ".json": "application/json"}
OUT = Path("out")

# One color per class, for drawing watershed masks.
PALETTE = [
    (0, 0, 0), (170, 110, 40), (230, 25, 75), (245, 130, 48), (255, 225, 25),
    (60, 180, 75), (70, 240, 240), (128, 0, 0), (0, 130, 200), (145, 30, 180),
    (240, 50, 230), (0, 0, 128), (210, 245, 60),
]


def header(text):
    print(f"\n>>> {text}")


def show_json(obj, max_files=3):
    """Print JSON, shortening a long 'files' / 'entries' list."""
    obj = dict(obj)
    for field in ("files", "entries"):
        items = obj.get(field)
        if isinstance(items, list) and len(items) > max_files:
            obj[field] = items[:max_files] + [f"... {len(items) - max_files} more"]
    print(json.dumps(obj, indent=2))


def colorize(mask_png):
    """Watershed mask (gray value per class) -> color image."""
    lut = [0] * 256 * 3
    for value, (class_id, _) in GRAY_TO_CLASS.items():
        lut[value * 3:value * 3 + 3] = PALETTE[class_id]
    img = Image.open(io.BytesIO(mask_png)).convert("L")
    img.putpalette(lut)
    return img.convert("RGB")


def classes_in(mask_png):
    gray = Image.open(io.BytesIO(mask_png)).convert("L")
    total = gray.width * gray.height
    shares = {}
    for value, count in enumerate(gray.histogram()):
        if count and value in GRAY_TO_CLASS:
            shares[GRAY_TO_CLASS[value][1]] = count / total
    return shares


def labeled(img, text):
    img = img.copy()
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, img.height - 26, img.width, img.height], fill=(255, 255, 255))
    draw.text((8, img.height - 20), text, fill=(0, 0, 0))
    return img


def save(images, path, open_it):
    """Lay images out in a row and save them."""
    w = sum(i.width for i in images) + 10 * (len(images) - 1)
    h = max(i.height for i in images)
    sheet = Image.new("RGB", (w, h), (255, 255, 255))
    x = 0
    for img in images:
        sheet.paste(img, (x, 0))
        x += img.width + 10
    OUT.mkdir(exist_ok=True)
    sheet.save(path)
    print(f"saved {path}")
    if open_it and sys.platform == "darwin":
        subprocess.run(["open", str(path)], check=False)


# ---------- commands ----------

def cmd_source(args):
    video_dir = DATA_ROOT / args.video
    if not video_dir.exists():
        sys.exit(f"{video_dir} not found")
    clips = sorted(p for p in video_dir.iterdir() if p.is_dir())[: args.clips]
    header(f"Test data: {len(clips)} clip(s) from CholecSeg8k {args.video} "
           f"(one video = one laparoscopic cholecystectomy case)")
    previews = []
    for clip in clips:
        frames = sorted(clip.glob("*_endo.png"))
        first = frames[0]
        mask = (clip / first.name.replace("_endo.png", "_endo_watershed_mask.png")).read_bytes()
        img = Image.open(first).convert("RGB")
        print(f"\n  {clip.name}: {len(frames)} frames, {img.width}x{img.height} PNG, "
              f"each with a segmentation mask")
        print(f"    files: {first.name}, {first.name.replace('.png', '_watershed_mask.png')}, ...")
        print("    classes labeled in the first frame:")
        for name, share in sorted(classes_in(mask).items(), key=lambda kv: -kv[1]):
            print(f"      {name:24} {share:6.1%}")
        previews += [labeled(img, f"{clip.name} frame"),
                     labeled(colorize(mask), f"{clip.name} mask")]
    print("\n  The upload adds what a real device would: a fake patient name + MRN burned")
    print("  into every frame, and a manifest with device metadata and SHA-256 per file.")
    save(previews[:4], OUT / "upload-source.png", args.open)


def cmd_quarantine(args):
    s3 = lake_client()
    pending = pending_uploads(s3)
    header(f"s3://quarantine/uploads/  ({len(pending)} upload(s) waiting for ingest)")
    for upload_id in pending:
        keys = list_keys(s3, "quarantine", f"uploads/{upload_id}/")
        frames = [k for k in keys if "/frames/" in k]
        masks = [k for k in keys if "/masks/" in k]
        print(f"\n  {upload_id}")
        print(f"    {len(frames)} frames, {len(masks)} masks, manifest.json")
    if not pending:
        return
    matches = [u for u in pending if args.upload is None or args.upload in u]
    if not matches:
        sys.exit(f"no pending upload matches {args.upload!r}")
    upload_id = matches[0]
    header(f"manifest.json of {upload_id}  (note the patient name and MRN: PHI)")
    show_json(read_json(s3, "quarantine", f"uploads/{upload_id}/manifest.json"))


def cmd_redaction(args):
    s3 = lake_client()
    records = [k for k in list_keys(s3, "curated", "cases/") if k.endswith("/record.json")]
    if not records:
        sys.exit("nothing curated yet")
    newest = max((read_json(s3, "curated", k) for k in records), key=lambda r: r["ingested_at"])
    frame = next(f for f in newest["files"] if f["kind"] == "frame")
    name = frame["key"].rsplit("/", 1)[1]
    raw_key = f"uploads/{newest['source_upload']}/frames/{name}"
    header("De-identification: the same frame before and after ingest")
    print(f"  before: s3://quarantine/{raw_key}")
    print(f"  after:  s3://curated/{frame['key']}")
    print(f"  method: {newest['deid_method']} (overlay region blacked out and verified empty)")
    before = Image.open(io.BytesIO(
        s3.get_object(Bucket="quarantine", Key=raw_key)["Body"].read())).convert("RGB")
    after = Image.open(io.BytesIO(
        s3.get_object(Bucket="curated", Key=frame["key"])["Body"].read())).convert("RGB")
    save([labeled(before, "BEFORE: quarantine (patient name + MRN burned in)"),
          labeled(after, "AFTER: curated (redacted)")],
         OUT / "redaction-before-after.png", args.open)


def cmd_curated(args):
    s3 = lake_client()
    keys = list_keys(s3, "curated", "cases/")
    records = sorted(k for k in keys if k.endswith("/record.json"))
    header(f"s3://curated/cases/  ({len(records)} clip(s))")
    for rec_key in records:
        clip_prefix = rec_key.rsplit("/", 1)[0] + "/"
        n_frames = sum(1 for k in keys if k.startswith(clip_prefix + "frames/"))
        print(f"  {clip_prefix:40} {n_frames} frames")
    if records:
        newest = max((read_json(s3, "curated", k) for k in records),
                     key=lambda r: r["ingested_at"])
        header("newest curated record.json  (pseudonym instead of name/MRN; "
               "every file has its checksum and its source checksum)")
        show_json(newest, max_files=2)


def cmd_rejected(args):
    s3 = lake_client()
    keys = sorted(k for k in list_keys(s3, "rejected", "") if k.endswith("rejection.json"))
    records = sorted((read_json(s3, "rejected", k) for k in keys),
                     key=lambda r: r["processed_at"])[-args.last:]
    header(f"s3://rejected/  ({len(keys)} rejection record(s), newest {len(records)} shown, "
           f"no patient data)")
    for r in records:
        print(f"  {r['upload_id']}")
        for reason in r["reasons"]:
            print(f"    - {reason}")


def cmd_dataset(args):
    s3 = lake_client()
    latest = read_json(s3, "datasets", f"releases/{args.dataset}/latest.json")
    base = latest["manifest"].rsplit("/", 1)[0]
    header(f"s3://datasets/{base}/DATASET_CARD.md")
    print(s3.get_object(Bucket="datasets", Key=f"{base}/DATASET_CARD.md")["Body"].read().decode())
    manifest = read_json(s3, "datasets", latest["manifest"])
    header(f"manifest.json: {len(manifest['entries'])} entries, each a path -> content hash")
    for e in manifest["entries"][:4]:
        print(f"  {json.dumps(e)}")
    print(f"  ... {len(manifest['entries']) - 4} more")
    blob = blob_key(manifest["entries"][0]["sha256"])
    head = s3.head_object(Bucket="datasets", Key=blob)
    header(f"the stored file s3://datasets/{blob}")
    print(f"  object lock: {head.get('ObjectLockMode')} until "
          f"{head.get('ObjectLockRetainUntilDate')}")


# ---------- browser links ----------

def console(bucket, prefix=""):
    """Data lake console page for a folder."""
    return f"{CONSOLE}/browser/{bucket}/{quote(prefix, safe='')}"


def file_url(s3, bucket, key):
    """Short-lived link that shows a data lake file directly in the browser."""
    ctype = CONTENT_TYPES.get(Path(key).suffix, "text/plain; charset=utf-8")
    return s3.generate_presigned_url(
        "get_object", ExpiresIn=3600,
        Params={"Bucket": bucket, "Key": key, "ResponseContentType": ctype,
                "ResponseContentDisposition": "inline"})


def local_url(path):
    return Path(path).resolve().as_uri()


def metaflow_run(flow):
    from metaflow import Flow, namespace
    namespace(None)
    return f"{METAFLOW_UI}/{flow}/{Flow(flow).latest_run.id}"


def newest_record(s3):
    records = [k for k in list_keys(s3, "curated", "cases/") if k.endswith("/record.json")]
    key = max(records, key=lambda k: read_json(s3, "curated", k)["ingested_at"])
    return key, read_json(s3, "curated", key)


def links_source(s3, args):
    clip = sorted(p for p in (DATA_ROOT / args.match).iterdir() if p.is_dir())[0]
    frame = sorted(clip.glob("*_endo.png"))[0]
    return [
        ("the test data on disk: one clip folder", local_url(clip)),
        ("one raw frame", local_url(frame)),
        ("first frame + mask of each clip", local_url("out/upload-source.png")),
    ]


def links_upload(s3, args):
    pending = [u for u in pending_uploads(s3) if args.match in u]
    if not pending:
        sys.exit(f"no pending upload matches {args.match!r}")
    upload = pending[0]
    prefix = f"uploads/{upload}/"
    frame = sorted(k for k in list_keys(s3, "quarantine", prefix + "frames/"))[0]
    mask = sorted(k for k in list_keys(s3, "quarantine", prefix + "masks/"))[0]
    return [
        ("quarantine bucket: every upload, untrusted", console("quarantine", "uploads/")),
        ("one upload's frames (80 PNGs)", console("quarantine", prefix + "frames/")),
        ("an uploaded frame: patient name + MRN burned in", file_url(s3, "quarantine", frame)),
        ("its segmentation mask", file_url(s3, "quarantine", mask)),
        ("the upload's manifest.json (PHI, device, checksums)",
         file_url(s3, "quarantine", prefix + "manifest.json")),
    ]


def links_ingest(s3, args):
    rec_key, rec = newest_record(s3)
    clip_prefix = rec_key.rsplit("/", 1)[0] + "/"
    frame = next(f["key"] for f in rec["files"] if f["kind"] == "frame")
    rejected = sorted(k for k in list_keys(s3, "rejected", "") if k.endswith("rejection.json"))
    newest_rejection = max(rejected, key=lambda k: read_json(s3, "rejected", k)["processed_at"])
    return [
        ("Metaflow: the ingest run, one pod per upload", metaflow_run("IngestFlow")),
        ("curated bucket: de-identified clips by case", console("curated", "cases/")),
        ("one curated clip", console("curated", clip_prefix)),
        ("a curated frame: patient text removed", file_url(s3, "curated", frame)),
        ("before / after redaction", local_url("out/redaction-before-after.png")),
        ("the curated record.json (pseudonym, lineage, no PHI)",
         file_url(s3, "curated", rec_key)),
        ("rejected bucket", console("rejected")),
        ("the newest rejection record", file_url(s3, "rejected", newest_rejection)),
    ]


def links_dataset(s3, args):
    latest = read_json(s3, "datasets", "releases/cholecseg/latest.json")
    base = latest["manifest"].rsplit("/", 1)[0] + "/"
    return [
        ("Metaflow: the dataset release run, one pod per case",
         metaflow_run("DatasetReleaseFlow")),
        ("datasets bucket: releases", console("datasets", "releases/cholecseg/")),
        ("this release", console("datasets", base)),
        ("DATASET_CARD.md", file_url(s3, "datasets", base + "DATASET_CARD.md")),
        ("manifest.json: every file, path -> sha256",
         file_url(s3, "datasets", latest["manifest"])),
        ("locked blobs, stored by content hash", console("datasets", "blobs/sha256/")),
    ]


def links_train(s3, args):
    from mlflow import MlflowClient
    run = MlflowClient(MLFLOW_UI).get_run(args.run_id)
    return [
        ("Metaflow: the training run", metaflow_run("TrainFlow")),
        ("MLflow: params, metrics, dataset ID, git commit",
         f"{MLFLOW_UI}/#/experiments/{run.info.experiment_id}/runs/{args.run_id}"),
    ]


def links_gate(s3, args):
    name = POLICY["registered_model"]
    return [
        ("Metaflow: the release gate run", metaflow_run("ReleaseGateFlow")),
        ("MLflow: the registered candidate version",
         f"{MLFLOW_UI}/#/models/{name}/versions/{args.version}"),
    ]


def links_approve(s3, args):
    name = POLICY["registered_model"]
    return [("MLflow: approved alias and every version", f"{MLFLOW_UI}/#/models/{name}")]


def links_canary(s3, args):
    return [("Boston: new model's prediction", local_url("out/boston-overlay.png")),
            ("Denver: unchanged model's prediction", local_url("out/denver-overlay.png"))]


def links_audit(s3, args):
    reports = sorted(glob.glob("out/audit-boston-*.txt"))
    return [("the saved audit report", local_url(reports[-1]))] if reports else []


LINKS = {"source": links_source, "upload": links_upload, "ingest": links_ingest, "dataset": links_dataset,
         "train": links_train, "gate": links_gate, "approve": links_approve,
         "canary": links_canary, "audit": links_audit}


def cmd_links(args):
    for label, url in LINKS[args.stage](lake_client(), args):
        print(f"{label}|{url}")


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--open", action="store_true", help="open saved images (macOS)")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("source")
    s.add_argument("--video", required=True)
    s.add_argument("--clips", type=int, default=1)
    q = sub.add_parser("quarantine")
    q.add_argument("--upload", help="print the manifest of the first upload whose ID contains this")
    sub.add_parser("redaction")
    sub.add_parser("curated")
    r = sub.add_parser("rejected")
    r.add_argument("--last", type=int, default=3, help="how many of the newest to show")
    d = sub.add_parser("dataset")
    d.add_argument("--dataset", default="cholecseg")
    lk = sub.add_parser("links")
    lk.add_argument("stage", choices=LINKS)
    lk.add_argument("--match", default="", help="source: video name; upload: upload ID must contain this")
    lk.add_argument("--run-id", help="train stage: MLflow run ID")
    lk.add_argument("--version", help="gate stage: registered model version")
    args = p.parse_args()
    {"links": cmd_links, "source": cmd_source, "quarantine": cmd_quarantine, "redaction": cmd_redaction,
     "curated": cmd_curated, "rejected": cmd_rejected, "dataset": cmd_dataset}[args.cmd](args)


if __name__ == "__main__":
    main()
