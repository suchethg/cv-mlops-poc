"""Approve or reject a release candidate. A human decision, so it's a
deliberate command, not part of any automated flow.

Rules enforced:
  - only versions that passed the release gate can be approved
  - a version is decided once: approved or rejected, never both
  - the approver must not be the person who trained the model
    (separation of duties)
Every decision is written as a write-once record in the locked datasets
bucket before the registry is updated.

POC limitation: the approver's identity is typed in, not verified. Production
would take it from SSO and enforce it with registry permissions.

Usage (Mac, tunnels open, credentials loaded):
  python scripts/approve_model.py --version 1 --approver qa.reviewer --reason "..."
  python scripts/approve_model.py --version 1 --approver qa.reviewer --reject --reason "..."
"""
import argparse
import datetime
import os
import sys

from mlflow import MlflowClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "flows"))
from surgseg.lake import exists, lake_client, write_json  # noqa: E402

NAME = "cholecseg-segmenter"


def main():
    p = argparse.ArgumentParser(description="Approve or reject a release candidate")
    p.add_argument("--version", required=True)
    p.add_argument("--approver", required=True, help="who is making this decision")
    p.add_argument("--reason", required=True)
    p.add_argument("--reject", action="store_true")
    p.add_argument("--mlflow", default=os.environ.get("MLFLOW_LOCAL_URI", "http://localhost:5050"))
    args = p.parse_args()

    client = MlflowClient(args.mlflow)
    v = client.get_model_version(NAME, args.version)
    tags = v.tags

    if tags.get("gate.decision") != "passed":
        sys.exit(f"REFUSED: version {args.version} did not pass the release gate")
    if tags.get("approval.status") in ("approved", "rejected"):
        sys.exit(f"REFUSED: version {args.version} was already {tags['approval.status']} "
                 f"by {tags.get('approval.by')}")
    trainer = tags.get("trained_by", "unknown")
    if args.approver.strip().lower() == trainer.strip().lower():
        sys.exit(f"REFUSED: {args.approver} trained this model. Approval requires a "
                 "different person (separation of duties).")

    decision = "rejected" if args.reject else "approved"
    now = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    record = {
        "model": NAME, "version": args.version, "decision": decision,
        "approver": args.approver, "reason": args.reason, "decided_at": now,
        "trained_by": trainer, "source_run_id": tags.get("source.run_id"),
        "dataset_id": tags.get("dataset.id"), "code_git_commit": tags.get("code.git_commit"),
        "gate_run_id": tags.get("gate.run_id"), "gate_policy": tags.get("gate.policy"),
    }

    # Audit record first: if this write fails, the registry is left unchanged.
    s3 = lake_client()
    key = f"_release-records/{NAME}/v{args.version}-decision.json"
    if exists(s3, "datasets", key):
        sys.exit(f"REFUSED: a decision record already exists at datasets/{key}")
    write_json(s3, "datasets", key, record)

    for k, val in {"approval.status": decision, "approval.by": args.approver,
                   "approval.reason": args.reason, "approval.at": now}.items():
        client.set_model_version_tag(NAME, args.version, k, val)
    if decision == "approved":
        client.set_registered_model_alias(NAME, "approved", args.version)

    print(f"{NAME} version {args.version}: {decision.upper()} by {args.approver}")
    print(f"  trained by {trainer}; dataset {record['dataset_id']}; "
          f"code {str(record['code_git_commit'])[:12]}")
    print(f"  write-once record: datasets/{key}")


if __name__ == "__main__":
    main()