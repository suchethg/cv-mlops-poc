# 5-minute demo walkthrough

## Before the interview (15 min earlier)
- Tab 1: `metaflow-dev up` running; Tilt all green
- Tab 2: `scripts/port_forwards.sh` running (5 ports)
- Tab 3: `source scripts/env.sh` -> "ready"
- `kubectl get pods` all Running; `python scripts/check_datalake.py` all OK
- Browser tabs: GitHub README · MLflow (5050) · data lake console (9101) · Argo (2746)
- `out/boston-overlay.png` already generated, open in Preview

## 0:00 — The problem (30 s, README diagram)
"Medical-device ML has to answer one question at any time: what exactly is
running on this device, and can you prove where it came from? I built a
working platform around that question."

## 0:30 — Data in, safely (60 s)
- `python scripts/simulate_device_upload.py --video video09 --clips 1 --inject corrupt`
- `python flows/ingest_flow.py argo-workflows trigger` -> show the run in Argo
- Data lake console: `rejected/` -> rejection.json ("checksum mismatch")
- Show a quarantine frame (patient name burned in) next to its curated frame (blacked out)
"Nothing with patient identifiers leaves quarantine. Identifiers become keyed pseudonyms."

## 1:30 — Immutable datasets (45 s)
- Open DATASET_CARD.md: splits by case, golden test set, class balance
- Run the delete attempt -> "Object is WORM protected"
"The dataset ID is a hash of every file. Change one byte and the ID changes."

## 2:15 — Train and gate (60 s)
- MLflow run: lineage tags (dataset, git commit, image), per-class IoU
- "Pixel accuracy 0.86, but cystic duct IoU 0. Averages hide safety-critical misses."
- Release gate: weak model REJECTED; full model PASSED (ONNX parity, p95 14 ms)
- `approve_model.py --approver <me>` -> REFUSED (separation of duties)

## 3:15 — Edge and safe rollout (60 s)
- Show out/boston-overlay.png
- `curl localhost:8081/model` -> the device reports its own lineage
- `kubectl set env deployment/edge-boston MODEL_VERSION=1` -> REFUSING TO SERVE;
  old pod keeps serving; `kubectl apply -f k8s/edge-sites.yaml` to roll back

## 4:15 — The audit (45 s)
- `python scripts/audit.py --site boston` -> CHAIN INTACT
- Point at the WARN: "The audit caught a bug in my own simulator that every
  earlier check missed. In production I'd move that check to ingest."

## Questions to be ready for
- Scaling to 40 TB per training run: sharding, caching near GPUs, data loading throughput
- Why staggered cron here and event triggers in production
- What changes for COMPLIANCE-mode retention and real SSO
- How you'd fix cystic duct performance (class weighting, resolution, more data, Dice loss)
- What happens if MinIO, MLflow or a node goes down