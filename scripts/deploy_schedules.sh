#!/usr/bin/env bash
# Deploy the platform's flows to Argo Workflows as scheduled workflows.
#
#   IngestFlow          every hour: picks up new device uploads
#   DatasetReleaseFlow  monthly, 1st at 02:00 UTC
#   TrainFlow           monthly, 1st at 03:00 UTC (latest dataset release)
#   ReleaseGateFlow     monthly, 1st at 07:00 UTC (newest ungated training run)
#
# Approval is deliberately NOT automated: a person still runs
# scripts/approve_model.py.
#
# Only a clean, pushed commit can be deployed. Its hash is baked into the
# deployed TrainFlow, so every scheduled training run records its exact code.
#
# Usage (from a set-up tab, with `metaflow-dev up` running):
#   scripts/deploy_schedules.sh
set -euo pipefail

if [ "${METAFLOW_PROFILE:-}" != "local" ]; then
  echo "Refusing to deploy: this tab isn't set up. Run: source scripts/env.sh"
  exit 1
fi
if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
  echo "Refusing to deploy: uncommitted changes. Commit and push first."
  exit 1
fi
git fetch -q origin
if ! git branch -r --contains HEAD | grep -q "origin/main"; then
  echo "Refusing to deploy: this commit isn't pushed to origin/main. Push first."
  exit 1
fi

export DEPLOY_GIT_COMMIT="$(git rev-parse HEAD)"
# Every step gets Metaflow's datastore credentials and the data lake's.
export METAFLOW_KUBERNETES_SECRETS="minio-secret,datalake-creds"

deploy() {  # deploy <flow file> <default image for steps without their own>
  echo "== $1 (default image $2)"
  METAFLOW_KUBERNETES_CONTAINER_IMAGE="$2" python "flows/$1" argo-workflows create --max-workers 4
}
deploy ingest_flow.py          surgseg-runtime:0.2
deploy dataset_release_flow.py surgseg-runtime:0.2
deploy train_flow.py           surgseg-train:0.1
deploy release_gate_flow.py    surgseg-gate:0.1

echo
echo "Deployed commit $DEPLOY_GIT_COMMIT"
echo "Schedules: kubectl get cronworkflows   UI: http://localhost:2746"