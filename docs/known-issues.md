# Known issues and decisions

## MinIO community images are no longer published

**Problem:** In September 2026, MinIO removed its community-edition images from
Docker Hub and gated them on Quay. The Metaflow dev stack's Helm chart failed
with `ImagePullBackOff` / `unauthorized`.

**Fix:** Build MinIO and `mc` from pinned open-source tags
(`docker/minio/Dockerfile`), tag them with the names the chart expects, and
load them into Minikube. Kubernetes uses the local image and never pulls.

**Production lesson:** Never depend on a vendor's public registry at runtime.
Mirror every image into an internal registry, pin by digest, and sign it.

**Security note:** The last community release has a known, unpatched
vulnerability. Acceptable for a local laptop demo; not for production.

## MinIO data is lost when its pod restarts

**Problem:** The dev stack runs MinIO on a temporary volume. Restarting the pod
deleted the `metaflow-test` bucket, and flows failed with `S3 object not found`.

**Workaround (until step 3):** Recreate the bucket after any MinIO restart:

    python -c "import boto3; s3=boto3.client('s3',endpoint_url='http://localhost:9000',aws_access_key_id='rootuser',aws_secret_access_key='rootpass123',region_name='us-east-1'); s3.create_bucket(Bucket='metaflow-test')"

**Fix (step 3):** Give MinIO persistent storage before any dataset versions or
models are stored. Object storage on a temporary disk is not storage.

## Minikube can't load images from Docker Desktop's containerd store

**Problem:** `minikube image load <image>` failed with `blob ... not found`.

**Fix:** Export to a file first, for this chip only, then load the file:

    docker save --platform linux/arm64 -o /tmp/img.tar <image>
    minikube image load /tmp/img.tar


## MLflow server killed for using too much memory (OOMKilled)

**Problem:** MLflow 3.16 starts a background job system (extra Python worker
processes for GenAI evaluation and scheduled jobs) by default. With a 1.5 GB
memory limit, the pod was repeatedly OOMKilled.

**Fix:** Disable the unused feature with
`MLFLOW_SERVER_ENABLE_JOB_EXECUTION=false` and raise the limit to 2 GiB.

**Production lesson:** Set resource limits from measured usage, and explicitly
disable server features you don't use. It saves memory and shrinks the attack
surface.