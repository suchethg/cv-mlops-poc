#!/usr/bin/env bash
# Opens tunnels from your Mac to services inside Minikube. Leave it running.
# Press Ctrl+C to close all tunnels. Each tunnel reconnects by itself if its
# pod restarts.
#
#   MLflow UI           http://localhost:5050
#   Data lake API       http://localhost:9100
#   Data lake console   http://localhost:9101
#   Edge site Boston    http://localhost:8081
#   Edge site Denver    http://localhost:8092
set -uo pipefail
trap 'kill 0' EXIT

forward() {
  while true; do
    kubectl port-forward "$@"
    echo "tunnel $1 dropped; reconnecting..."
    sleep 2
  done
}

forward svc/mlflow 5050:5000 &
forward svc/datalake 9100:9000 9101:9001 &
forward svc/edge-boston 8081:8080 &
forward svc/edge-denver 8092:8080 &

wait