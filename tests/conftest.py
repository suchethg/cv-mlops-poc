"""Shared test setup: an in-memory S3 (moto) shaped like our data lake."""
import io
import os
import sys

import boto3
import pytest
from moto import mock_aws
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "flows"))


@pytest.fixture
def s3(monkeypatch):
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test")
    monkeypatch.setenv("LAKE_SECRET_KEY", "test-pseudonym-key")
    with mock_aws():
        client = boto3.client("s3", region_name="us-east-1")
        for b in ("quarantine", "curated", "rejected"):
            client.create_bucket(Bucket=b)
            client.put_bucket_versioning(Bucket=b, VersioningConfiguration={"Status": "Enabled"})
        client.create_bucket(Bucket="datasets", ObjectLockEnabledForBucket=True)
        yield client


def png(width=64, height=40, color=(120, 60, 60)):
    buf = io.BytesIO()
    Image.new("RGB", (width, height), color).save(buf, format="PNG")
    return buf.getvalue()


def mask_png(width=64, height=40, gray=21):
    buf = io.BytesIO()
    Image.new("L", (width, height), gray).save(buf, format="PNG")
    return buf.getvalue()