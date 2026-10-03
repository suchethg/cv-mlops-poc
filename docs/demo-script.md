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