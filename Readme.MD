cat > README.md <<'EOF'
# Surgical Segmentation MLOps — Proof of Concept

A laptop-scale version of a regulated ML platform for surgical image
segmentation: ingest, de-identify, version, train, evaluate, approve, deploy,
with full lineage from any deployed model back to its exact input files,
code, and container image.

Runs entirely on local Kubernetes (Minikube) using MinIO, Metaflow,
Argo Workflows, and MLflow.

## Status

- [ ] Step 1: Local platform (Minikube, MinIO, Metaflow, Argo)
- [ ] Step 2: MLflow tracking server in Kubernetes
- [ ] Step 3: Ingest and quarantine flow
- [ ] Step 4: Immutable dataset versions
- [ ] Step 5: Training flow with lineage
- [ ] Step 6: Evaluation and release gate
- [ ] Step 7: Edge-style serving with rollback
- [ ] Step 8: Audit command
- [ ] Step 9: Automated retraining and CI
EOF