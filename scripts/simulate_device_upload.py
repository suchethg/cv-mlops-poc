"""Simulate a surgical device uploading recorded frames to the data lake.

Each upload is one clip (80 consecutive frames) from one CholecSeg8k video,
sent by a pretend device at a pretend hospital site. It lands in the
`quarantine` bucket, untrusted, exactly like real device telemetry would.

What a real device upload looks like, and what we simulate:
  - Frames with burned-in patient text in the corner (PHI the ingest flow
    must remove). We stamp a fake name and MRN onto every frame.
  - Segmentation masks (in reality these come later from annotators).
  - A manifest.json with device metadata and a SHA-256 for every file.
    The manifest is uploaded LAST: its presence means "upload complete".

Use --inject to send a deliberately bad upload and watch quarantine catch it:
  corrupt           one frame's bytes don't match its checksum
  missing-metadata  the manifest is missing required fields
  no-consent        the patient did not consent to data use

Usage (Mac, with tunnels open and credentials loaded):
  python scripts/simulate_device_upload.py --list
  python scripts/simulate_device_upload.py --video video01 --clips 1
  python scripts/simulate_device_upload.py --video video09 --clips 1 --inject corrupt
"""
import argparse
import datetime
import hashlib
import io
import json
import os
import random
import sys
import uuid
from pathlib import Path

import boto3
from PIL import Image, ImageDraw

DATA_ROOT = Path("data/raw/CholecSeg8k")
BUCKET = "quarantine"
FAKE_NAMES = ["DOE^JANE", "ROE^RICHARD", "SMITH^ALEX", "GARCIA^MARIA", "CHEN^WEI"]
SITES = ["site-boston", "site-austin", "site-denver"]


def lake_client():
    try:
        return boto3.client(
            "s3",
            endpoint_url=os.environ["LAKE_ENDPOINT"],
            aws_access_key_id=os.environ["LAKE_ACCESS_KEY"],
            aws_secret_access_key=os.environ["LAKE_SECRET_KEY"],
            region_name="us-east-1",
        )
    except KeyError as e:
        sys.exit(f"{e} not set. Run: set -a; source .env.local; set +a")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def burn_in_phi(png_bytes: bytes, text: str) -> bytes:
    """Stamp patient text into the top-left corner, like a real video overlay."""
    img = Image.open(io.BytesIO(png_bytes)).convert("RGB")
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, 320, 28], fill=(0, 0, 0))
    draw.text((6, 8), text, fill=(255, 255, 255))
    out = io.BytesIO()
    img.save(out, format="PNG")
    return out.getvalue()


def list_videos():
    if not DATA_ROOT.exists():
        sys.exit(f"{DATA_ROOT} not found. Download and unzip the dataset first.")
    for video in sorted(p for p in DATA_ROOT.iterdir() if p.is_dir()):
        clips = sorted(p.name for p in video.iterdir() if p.is_dir())
        print(f"{video.name}: {len(clips)} clips  ({clips[0]} ... {clips[-1]})")


def upload_clip(s3, video: str, clip_dir: Path, inject: str):
    rng = random.Random(clip_dir.name)  # same clip -> same fake patient
    site = rng.choice(SITES)
    device_id = f"dvc-{rng.randint(1000, 9999)}"
    patient_name = rng.choice(FAKE_NAMES)
    mrn = f"MRN{rng.randint(100000, 999999)}"
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    upload_id = f"{site}-{device_id}-{clip_dir.name}-{stamp}-{uuid.uuid4().hex[:6]}"
    prefix = f"uploads/{upload_id}"

    frames = sorted(clip_dir.glob("*_endo.png"))
    if not frames:
        print(f"  skip {clip_dir.name}: no frames")
        return
    files = []
    for i, frame in enumerate(frames):
        base = frame.name.replace("_endo.png", "")
        image = burn_in_phi(frame.read_bytes(), f"PT: {patient_name}  {mrn}")
        mask = (clip_dir / f"{base}_endo_watershed_mask.png").read_bytes()

        for kind, name, data in (("frame", f"{base}.png", image),
                                 ("mask", f"{base}_mask.png", mask)):
            key = f"{prefix}/{kind}s/{name}"
            digest = sha256(data)
            if inject == "corrupt" and kind == "frame" and i == 0:
                data = data[: len(data) // 2]  # truncated in transit
            s3.put_object(Bucket=BUCKET, Key=key, Body=data)
            files.append({"key": key, "kind": kind, "sha256": digest})

    manifest = {
        "upload_id": upload_id,
        "device_id": device_id,
        "site_id": site,
        "procedure": "laparoscopic_cholecystectomy",
        "case_id": video,  # one Cholec80 video = one surgical case
        "clip": clip_dir.name,
        "recorded_at": stamp,
        "patient": {"name": patient_name, "mrn": mrn},  # PHI: must not leave quarantine
        "consent": {"research_use": inject != "no-consent"},
        "label_source": "CholecSeg8k-watershed-v1",
        "files": files,
    }
    if inject == "missing-metadata":
        del manifest["site_id"]
        del manifest["consent"]

    s3.put_object(Bucket=BUCKET, Key=f"{prefix}/manifest.json",
                  Body=json.dumps(manifest, indent=2).encode(),
                  ContentType="application/json")
    note = f"  [injected: {inject}]" if inject != "none" else ""
    print(f"  uploaded {upload_id}: {len(frames)} frames{note}")


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--list", action="store_true", help="list videos and clips")
    p.add_argument("--video", help="e.g. video01")
    p.add_argument("--clips", type=int, default=1, help="how many clips to upload")
    p.add_argument("--inject", default="none",
                   choices=["none", "corrupt", "missing-metadata", "no-consent"])
    args = p.parse_args()

    if args.list:
        list_videos()
        return
    if not args.video:
        p.error("--video is required (use --list to see options)")

    video_dir = DATA_ROOT / args.video
    if not video_dir.exists():
        sys.exit(f"{video_dir} not found. Use --list to see videos.")
    clips = sorted(p for p in video_dir.iterdir() if p.is_dir())[: args.clips]

    s3 = lake_client()
    print(f"Uploading {len(clips)} clip(s) from {args.video} to s3://{BUCKET}/uploads/")
    for clip_dir in clips:
        upload_clip(s3, args.video, clip_dir, args.inject)


if __name__ == "__main__":
    main()