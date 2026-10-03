# Builds MinIO community edition (AGPLv3) and the mc client from pinned source.
# Why: MinIO stopped publishing free community images to Docker Hub and Quay.
# The tags match what the Metaflow dev stack's Helm chart expects.

FROM golang:1.23-alpine AS build
RUN apk add --no-cache git
ARG MINIO_TAG=RELEASE.2024-12-18T13-15-44Z
ARG MC_TAG=RELEASE.2024-11-21T17-21-54Z
ENV CGO_ENABLED=0 GOTOOLCHAIN=local
WORKDIR /src

RUN git clone --depth 1 --branch ${MINIO_TAG} https://github.com/minio/minio.git \
 && cd minio \
 && go build -trimpath -ldflags "-s -w" -o /out/minio . \
 && cp dockerscripts/docker-entrypoint.sh /out/docker-entrypoint.sh

RUN git clone --depth 1 --branch ${MC_TAG} https://github.com/minio/mc.git \
 && cd mc \
 && go build -trimpath -ldflags "-s -w" -o /out/mc .

# MinIO server image
FROM alpine:3.20 AS minio
RUN apk add --no-cache ca-certificates coreutils
COPY --from=build /out/minio /usr/bin/minio
COPY --from=build /out/docker-entrypoint.sh /usr/bin/docker-entrypoint.sh
RUN chmod +x /usr/bin/docker-entrypoint.sh
EXPOSE 9000 9001
ENTRYPOINT ["/usr/bin/docker-entrypoint.sh"]
CMD ["minio"]

# mc client image (used by the chart's bucket-setup job)
FROM alpine:3.20 AS mc
RUN apk add --no-cache ca-certificates bash curl
COPY --from=build /out/mc /usr/bin/mc
ENTRYPOINT ["mc"]
