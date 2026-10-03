## Release gate (release-policy-v1)
- Quick-test model 6dea6a9c...: REJECTED (mean IoU 0.272 < 0.40; class minimums failed)
- Incomplete run e7b6d4b6...: REJECTED (no metrics recorded)
- Full model 2a239fd9...: PASSED, ONNX matches PyTorch, p95 latency <fill in> ms
  -> registered cholecseg-segmenter v1 as "candidate"

## Separation of duties
- Trainer tried to approve v1: REFUSED
- qa.reviewer approved v1 with a recorded reason: APPROVED
- Second approval attempt: REFUSED
- Every gate decision and approval is a write-once record in datasets/_release-records/

## Edge deployment (step 7)
- Sites boston and denver pin cholecseg-segmenter v2 (sha256 3e792953...)
- /model on a device reports version, checksum, dataset ds-e66b802f5ece3cda,
  code 16253c81, trained_by, approved_by: the audit answer from the device itself
- Gate p95 latency 14.18 ms; field latency ~18-21 ms per request
- Bad rollout: set boston to v1 (no checksum) -> new pod REFUSING TO SERVE,
  CrashLoopBackOff; old pod kept serving v2; site never went down
- Rollback: kubectl apply -f k8s/edge-sites.yaml -> only boston changed, back to v2

   ## Audit (step 8)
   python scripts/audit.py --site boston -> CHAIN INTACT, 1 warning
   - device file checksum == registry == approval record
   - dataset ID recomputed from 5,560 file hashes matches ds-e66b802f5ece3cda
   - training code 16253c81 is published on GitHub
   - 35 clips -> 35 accepted device uploads across 3 sites
   - WARN: 17/17 cases map to >1 patient pseudonym (simulator bug, found by audit)
   python scripts/audit.py --version 1 -> BROKEN: no model checksum in approval record