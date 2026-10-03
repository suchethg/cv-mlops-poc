#!/usr/bin/env bash
# Opens tunnels from your Mac to services inside Minikube. Leave it running.
# Press Ctrl+C to close all tunnels. Rerun it if a pod restarts.
#
#   MLflow UI           http://localhost:5050
#   Data lake API       http://localhost:9100
#   Data lake console   http://localhost:9101
set -uo pipefail
trap 'kill 0' EXIT

kubectl port-forward svc/mlflow 5050:5000 &
kubectl port-forward svc/datalake 9100:9000 9101:9001 &

wait