"""Verify the local platform is reachable before running flows.

Run inside `metaflow-dev shell` while `metaflow-dev up` is running.
"""
import sys
import urllib.request

import boto3
from botocore.exceptions import BotoCoreError, ClientError

# Local dev-stack values. Production would use SSO and short-lived credentials.
MINIO_ENDPOINT = "http://localhost:9000"
MINIO_USER = "rootuser"
MINIO_PASSWORD = "rootpass123"
METADATA_URL = "http://localhost:8080/ping"

ok = True

# 1. MinIO: can we list buckets?
try:
    s3 = boto3.client(
        "s3",
        endpoint_url=MINIO_ENDPOINT,
        aws_access_key_id=MINIO_USER,
        aws_secret_access_key=MINIO_PASSWORD,
        region_name="us-east-1",
    )
    buckets = [b["Name"] for b in s3.list_buckets()["Buckets"]]
    print(f"[ok]   MinIO reachable, buckets: {buckets}")
except (BotoCoreError, ClientError) as e:
    ok = False
    print(f"[fail] MinIO not reachable at {MINIO_ENDPOINT}: {e}")

# 2. Metaflow metadata service: does it answer a ping?
try:
    with urllib.request.urlopen(METADATA_URL, timeout=5) as r:
        print(f"[ok]   Metaflow metadata service responded ({r.status})")
except Exception as e:
    ok = False
    print(f"[fail] Metadata service not reachable at {METADATA_URL}: {e}")

sys.exit(0 if ok else 1)