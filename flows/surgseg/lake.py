"""Access to the data lake (the persistent MinIO).

Uses LAKE_* variables, deliberately separate from Metaflow's own AWS_*
credentials for its scratch datastore. On the Mac they come from .env.local;
in pods they come from the `datalake-creds` Kubernetes secret.
"""
import json
import os

import boto3
from botocore.exceptions import ClientError


def lake_client():
    missing = [v for v in ("LAKE_ENDPOINT", "LAKE_ACCESS_KEY", "LAKE_SECRET_KEY")
               if v not in os.environ]
    if missing:
        raise RuntimeError(
            f"{missing} not set. On the Mac run: set -a; source .env.local; set +a")
    return boto3.client(
        "s3",
        endpoint_url=os.environ["LAKE_ENDPOINT"],
        aws_access_key_id=os.environ["LAKE_ACCESS_KEY"],
        aws_secret_access_key=os.environ["LAKE_SECRET_KEY"],
        region_name="us-east-1",
    )


def read_json(s3, bucket, key):
    return json.loads(s3.get_object(Bucket=bucket, Key=key)["Body"].read())


def write_json(s3, bucket, key, obj):
    s3.put_object(Bucket=bucket, Key=key, Body=json.dumps(obj, indent=2).encode(),
                  ContentType="application/json")


def exists(s3, bucket, key):
    try:
        s3.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError:
        return False


def list_keys(s3, bucket, prefix):
    keys = []
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix):
        keys += [o["Key"] for o in page.get("Contents", [])]
    return keys