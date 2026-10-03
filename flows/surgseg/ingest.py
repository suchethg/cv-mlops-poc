"""Ingest rules: decide whether a device upload is accepted or rejected.

An upload is ACCEPTED only if every check passes:
  1. Manifest has all required metadata.
  2. The patient consented to research use.
  3. Every file listed is present and its SHA-256 matches the manifest.
  4. Every frame decodes as an image, and frames and masks pair up.
Accepted uploads are de-identified and copied to `curated`.
Rejected uploads get a rejection record (with reasons, no PHI) in `rejected`.
Either way a marker is written so the upload is never processed twice.
"""
import datetime
import hashlib
import hmac
import io
import os

from botocore.exceptions import ClientError
from PIL import Image

from .lake import exists, list_keys, read_json, write_json

QUARANTINE, CURATED, REJECTED = "quarantine", "curated", "rejected"
REQUIRED_FIELDS = ["upload_id", "device_id", "site_id", "procedure", "case_id",
                   "clip", "recorded_at", "patient", "consent", "files"]

# De-identification method. Versioned, because the curated record must say
# exactly how each frame was cleaned (auditors will ask).
DEID_VERSION = "fixed-region-redaction-v1"
OVERLAY_BOX = (0, 0, 340, 32)  # where devices burn in patient text


def processed_marker(upload_id):
    return f"processed/{upload_id}.json"


def pending_uploads(s3):
    """Upload IDs whose manifest has arrived and that haven't been processed."""
    pending = []
    for key in list_keys(s3, QUARANTINE, "uploads/"):
        if key.endswith("/manifest.json"):
            upload_id = key.split("/")[1]
            if not exists(s3, QUARANTINE, processed_marker(upload_id)):
                pending.append(upload_id)
    return sorted(pending)


def check_manifest(manifest):
    problems = [f"missing field: {f}" for f in REQUIRED_FIELDS if f not in manifest]
    consent = manifest.get("consent", {})
    if "consent" in manifest and not consent.get("research_use", False):
        problems.append("no consent for research use")
    return problems


def check_files(s3, manifest):
    problems, blobs = [], {}
    for f in manifest.get("files", []):
        try:
            data = s3.get_object(Bucket=QUARANTINE, Key=f["key"])["Body"].read()
        except ClientError:
            problems.append(f"file missing: {f['key']}")
            continue
        if hashlib.sha256(data).hexdigest() != f["sha256"]:
            problems.append(f"checksum mismatch: {f['key']}")
            continue
        try:
            Image.open(io.BytesIO(data)).verify()
        except Exception:
            problems.append(f"not a valid image: {f['key']}")
            continue
        blobs[f["key"]] = data
    frames = [f for f in manifest.get("files", []) if f["kind"] == "frame"]
    masks = [f for f in manifest.get("files", []) if f["kind"] == "mask"]
    if len(frames) != len(masks):
        problems.append(f"{len(frames)} frames but {len(masks)} masks")
    return problems, blobs


def redact(png_bytes):
    """Black out the burned-in overlay region and confirm it's clean."""
    img = Image.open(io.BytesIO(png_bytes)).convert("RGB")
    img.paste((0, 0, 0), OVERLAY_BOX)
    region = img.crop(OVERLAY_BOX)
    assert region.getextrema() == ((0, 0), (0, 0), (0, 0)), "redaction failed"
    out = io.BytesIO()
    img.save(out, format="PNG")
    return out.getvalue()


def pseudonym(mrn):
    """Stable, non-reversible patient ID: same patient -> same pseudonym.

    POC uses the data lake secret as the HMAC key. Production would use a
    dedicated key held in a key management service.
    """
    key = os.environ["LAKE_SECRET_KEY"].encode()
    return "pt-" + hmac.new(key, mrn.encode(), hashlib.sha256).hexdigest()[:16]


def process_upload(s3, upload_id, run_pathspec):
    """Check one upload; promote it to curated or record a rejection."""
    manifest = read_json(s3, QUARANTINE, f"uploads/{upload_id}/manifest.json")
    problems = check_manifest(manifest)
    blobs = {}
    if not problems:
        problems, blobs = check_files(s3, manifest)

    now = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    result = {"upload_id": upload_id, "processed_at": now, "ingest_run": run_pathspec}

    if problems:
        # The rejection record names the problems but carries no patient data.
        result.update(status="rejected", reasons=problems)
        write_json(s3, REJECTED, f"{upload_id}/rejection.json", result)
    else:
        case, clip = manifest["case_id"], manifest["clip"]
        dest = f"cases/{case}/{clip}"
        files = []
        for f in manifest["files"]:
            name = f["key"].rsplit("/", 1)[1]
            data = blobs[f["key"]]
            if f["kind"] == "frame":
                data = redact(data)
            key = f"{dest}/{f['kind']}s/{name}"
            s3.put_object(Bucket=CURATED, Key=key, Body=data)
            files.append({"key": key, "kind": f["kind"],
                          "sha256": hashlib.sha256(data).hexdigest(),
                          "source_sha256": f["sha256"]})
        # Curated record: everything needed for lineage, nothing that
        # identifies the patient.
        record = {
            "case_id": case,
            "clip": clip,
            "patient_pseudonym": pseudonym(manifest["patient"]["mrn"]),
            "site_id": manifest["site_id"],
            "device_id": manifest["device_id"],
            "procedure": manifest["procedure"],
            "recorded_at": manifest["recorded_at"],
            "consent": manifest["consent"],
            "label_source": manifest.get("label_source"),
            "source_upload": upload_id,
            "deid_method": DEID_VERSION,
            "ingest_run": run_pathspec,
            "ingested_at": now,
            "files": files,
        }
        write_json(s3, CURATED, f"{dest}/record.json", record)
        result.update(status="accepted", case_id=case, clip=clip,
                      frames=sum(f["kind"] == "frame" for f in files))

    write_json(s3, QUARANTINE, processed_marker(upload_id), result)
    return result