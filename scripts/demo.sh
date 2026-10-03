#!/usr/bin/env bash
# Live demo helper. Run from a set-up tab (source scripts/env.sh).
#   scripts/demo.sh 1        corrupt upload -> ingest on Argo
#   scripts/demo.sh 1b       show the newest rejection record
#   scripts/demo.sh 2        try to delete a locked dataset file
#   scripts/demo.sh 3        push an unapproved model to Boston
#   scripts/demo.sh 3-reset  roll Boston back to what git says
#   scripts/demo.sh 4        audit the Boston device
set -uo pipefail

case "${1:-}" in
  1)
    echo "== 1. A corrupt upload arrives"
    python scripts/simulate_device_upload.py --video video09 --clips 1 --inject corrupt
    python flows/ingest_flow.py argo-workflows trigger
    echo "Ingest is running on Argo: http://localhost:2746"
    ;;
  1b)
    echo "== The rejection record (no patient data)"
    python - <<'EOF'
import json, os
import boto3
s3 = boto3.client("s3", endpoint_url=os.environ["LAKE_ENDPOINT"],
                  aws_access_key_id=os.environ["LAKE_ACCESS_KEY"],
                  aws_secret_access_key=os.environ["LAKE_SECRET_KEY"], region_name="us-east-1")
objs = s3.list_objects_v2(Bucket="rejected")["Contents"]
newest = max(objs, key=lambda o: o["LastModified"])
print(json.dumps(json.loads(s3.get_object(Bucket="rejected", Key=newest["Key"])["Body"].read()), indent=2))
EOF
    ;;
  2)
    echo "== 2. Try to delete a file from the training dataset"
    python - <<'EOF'
import os
import boto3
s3 = boto3.client("s3", endpoint_url=os.environ["LAKE_ENDPOINT"],
                  aws_access_key_id=os.environ["LAKE_ACCESS_KEY"],
                  aws_secret_access_key=os.environ["LAKE_SECRET_KEY"], region_name="us-east-1")
key = s3.list_objects_v2(Bucket="datasets", Prefix="blobs/")["Contents"][0]["Key"]
version = s3.list_object_versions(Bucket="datasets", Prefix=key)["Versions"][0]["VersionId"]
print("deleting", key, "as the data lake admin...")
try:
    s3.delete_object(Bucket="datasets", Key=key, VersionId=version)
    print("deleted (this should NOT happen)")
except Exception as e:
    print("BLOCKED:", str(e).split(": ", 1)[-1])
EOF
    ;;
  3)
    echo "== 3. Push unapproved model v1 to the Boston site"
    kubectl set env deployment/edge-boston MODEL_VERSION=1
    echo "waiting 30s for the new pod to try to start..."
    sleep 30
    kubectl get pods -l site=site-boston
    kubectl logs -l site=site-boston --tail=-1 | grep REFUSING | tail -1
    python scripts/edge_predict.py --site boston
    ;;
  3-reset)
    echo "== Roll back to what git says"
    kubectl apply -f k8s/edge-sites.yaml
    sleep 5
    kubectl get pods -l site=site-boston
    ;;
  4)
    echo "== 4. Audit the Boston device"
    python scripts/audit.py --site boston
    ;;
  *)
    echo "usage: scripts/demo.sh 1|1b|2|3|3-reset|4"
    ;;
esac