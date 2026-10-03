"""Ingest must reject bad uploads, strip patient data, and never process twice."""
import hashlib
import io
import json

from PIL import Image, ImageDraw

from conftest import mask_png, png
from surgseg.ingest import (OVERLAY_BOX, check_manifest, pending_uploads,
                            process_upload, pseudonym, redact)

GOOD_MANIFEST = {
    "upload_id": "u1", "device_id": "dvc-1", "site_id": "site-a", "procedure": "chole",
    "case_id": "video01", "clip": "video01_00000", "recorded_at": "20260101T000000Z",
    "patient": {"name": "DOE^JANE", "mrn": "MRN123456"},
    "consent": {"research_use": True}, "files": [],
}


def put_upload(s3, upload_id, corrupt=False, consent=True):
    frame, mask = png(), mask_png()
    files = []
    for kind, name, data in (("frame", "f0.png", frame), ("mask", "f0_mask.png", mask)):
        key = f"uploads/{upload_id}/{kind}s/{name}"
        digest = hashlib.sha256(data).hexdigest()
        s3.put_object(Bucket="quarantine", Key=key, Body=data[:20] if corrupt and kind == "frame" else data)
        files.append({"key": key, "kind": kind, "sha256": digest})
    manifest = dict(GOOD_MANIFEST, upload_id=upload_id, files=files,
                    consent={"research_use": consent})
    s3.put_object(Bucket="quarantine", Key=f"uploads/{upload_id}/manifest.json",
                  Body=json.dumps(manifest).encode())


def test_redaction_blacks_out_burned_in_text():
    img = Image.new("RGB", (400, 100), (200, 200, 200))
    ImageDraw.Draw(img).text((5, 5), "PT: DOE^JANE MRN123456", fill=(255, 255, 255))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    out = Image.open(io.BytesIO(redact(buf.getvalue())))
    assert out.crop(OVERLAY_BOX).getextrema() == ((0, 0), (0, 0), (0, 0))


def test_pseudonym_is_stable_and_hides_the_mrn(s3):
    assert pseudonym("MRN123456") == pseudonym("MRN123456")
    assert pseudonym("MRN123456") != pseudonym("MRN654321")
    assert "123456" not in pseudonym("MRN123456")


def test_manifest_checks_catch_missing_fields_and_no_consent():
    assert check_manifest(GOOD_MANIFEST) == []
    missing = {k: v for k, v in GOOD_MANIFEST.items() if k != "site_id"}
    assert "missing field: site_id" in check_manifest(missing)
    no_consent = dict(GOOD_MANIFEST, consent={"research_use": False})
    assert "no consent for research use" in check_manifest(no_consent)


def test_good_upload_is_curated_without_patient_data(s3):
    put_upload(s3, "good-1")
    result = process_upload(s3, "good-1", "IngestFlow/1/process/2")
    assert result["status"] == "accepted"
    record = s3.get_object(Bucket="curated", Key="cases/video01/video01_00000/record.json")
    text = record["Body"].read().decode()
    assert "MRN123456" not in text and "DOE^JANE" not in text
    assert json.loads(text)["patient_pseudonym"] == pseudonym("MRN123456")


def test_corrupt_and_unconsented_uploads_are_rejected(s3):
    put_upload(s3, "bad-1", corrupt=True)
    put_upload(s3, "bad-2", consent=False)
    r1 = process_upload(s3, "bad-1", "x")
    r2 = process_upload(s3, "bad-2", "x")
    assert r1["status"] == "rejected" and any("checksum mismatch" in r for r in r1["reasons"])
    assert r2["status"] == "rejected" and "no consent for research use" in r2["reasons"]


def test_uploads_are_processed_only_once(s3):
    put_upload(s3, "once-1")
    assert pending_uploads(s3) == ["once-1"]
    process_upload(s3, "once-1", "x")
    assert pending_uploads(s3) == []