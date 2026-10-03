#!/usr/bin/env bash
# Build an image with Docker Desktop, then load it into Minikube.
# Why: Minikube's own Docker has flaky access to Docker Hub, and
# `minikube image load` can't read Docker Desktop's containerd store directly.
#
# Usage: scripts/build_and_load.sh <image:tag> <build-context-dir> [build-target]
set -euo pipefail

IMAGE="$1"
CONTEXT="$2"
TARGET="${3:-}"

MINIKUBE="${MINIKUBE:-$(find .venv -type f -name minikube -perm -u+x | head -1)}"
if [ -z "$MINIKUBE" ]; then
  echo "Could not find the minikube binary under .venv" >&2
  exit 1
fi

# Make sure we build with Docker Desktop, not Minikube's Docker.
unset DOCKER_HOST DOCKER_TLS_VERIFY DOCKER_CERT_PATH MINIKUBE_ACTIVE_DOCKERD

if [ -n "$TARGET" ]; then
  docker build --target "$TARGET" -t "$IMAGE" "$CONTEXT"
else
  docker build -t "$IMAGE" "$CONTEXT"
fi

TAR="/tmp/$(echo "$IMAGE" | tr '/:' '__').tar"
docker save --platform linux/arm64 -o "$TAR" "$IMAGE"
"$MINIKUBE" image load "$TAR"
rm -f "$TAR"

echo "Loaded $IMAGE into Minikube"