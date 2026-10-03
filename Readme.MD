
# Surgical Segmentation MLOps — Proof of Concept

A small-scale version of a regulated ML platform for surgical image
segmentation: ingest, de-identify, version, train, evaluate, approve, deploy,
with full lineage from any deployed model back to its exact input files,
code, and container image.

Runs entirely on local Kubernetes (Minikube) using MinIO, Metaflow,
Argo Workflows, and MLflow.

## Status

- [x] Step 1: Local platform (Minikube, MinIO, Metaflow, Argo)
- [x] Step 2: MLflow tracking server in Kubernetes
- [x] Step 3: Ingest and quarantine flow
- [x] Step 4: Immutable dataset versions
- [x] Step 5: Training flow with lineage
- [x] Step 6: Evaluation and release gate
- [x] Step 7: Edge-style serving with rollback
- [ ] Step 8: Audit command
- [ ] Step 9: Automated retraining and CI
EOF
