"""Check the data lake: buckets, versioning, object lock, and persistence.

Run on your Mac with the data lake port-forward open (scripts/port_forwards.sh)
and credentials loaded:  set -a; source .env.local; set +a

Run it, restart the data lake pod, run it again: the marker written the first
time should still be there. That proves storage survives restarts.
"""
import datetime
import os
import sys

import boto3
from botocore.exceptions import ClientError

EXPECTED = ["curated", "datasets", "mlflow-artifacts", "quarantine", "rejected"]
MARKER_BUCKET, MARKER_KEY = "curated", "_healthcheck/persistence-marker.txt"

try:
    s3 = boto3.client(
        "s3",
        endpoint_url=os.environ["LAKE_ENDPOINT"],
        aws_access_key_id=os.environ["LAKE_ACCESS_KEY"],
        aws_secret_access_key=os.environ["LAKE_SECRET_KEY"],
        region_name="us-east-1",
    )
except KeyError as e:
    sys.exit(f"[fail] {e} not set. Run: set -a; source .env.local; set +a")

ok = True

# 1. Buckets exist
buckets = sorted(b["Name"] for b in s3.list_buckets()["Buckets"])
missing = [b for b in EXPECTED if b not in buckets]
if missing:
    ok = False
    print(f"[fail] missing buckets: {missing} (found {buckets})")
else:
    print(f"[ok]   buckets: {buckets}")

# 2. Versioning on every bucket
for b in EXPECTED:
    if b in buckets:
        status = s3.get_bucket_versioning(Bucket=b).get("Status", "Off")
        mark = "[ok]  " if status == "Enabled" else "[fail]"
        ok &= status == "Enabled"
        print(f"{mark} versioning on {b}: {status}")

# 3. Object lock on the datasets bucket
try:
    rule = s3.get_object_lock_configuration(Bucket="datasets")[
        "ObjectLockConfiguration"]["Rule"]["DefaultRetention"]
    print(f"[ok]   object lock on datasets: {rule['Mode']} {rule.get('Days')} days")
except (ClientError, KeyError) as e:
    ok = False
    print(f"[fail] object lock on datasets not configured: {e}")

# 4. Persistence marker
try:
    body = s3.get_object(Bucket=MARKER_BUCKET, Key=MARKER_KEY)["Body"].read().decode()
    print(f"[ok]   persistence marker found (written {body}): data survived")
except ClientError:
    now = datetime.datetime.now().isoformat(timespec="seconds")
    s3.put_object(Bucket=MARKER_BUCKET, Key=MARKER_KEY, Body=now.encode())
    print(f"[new]  wrote persistence marker at {now}. Restart the pod and rerun.")

sys.exit(0 if ok else 1)