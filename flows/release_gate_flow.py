"""Step 6: release gate. Decides whether a trained model may become a release
candidate.

  start  lineage and quality checks against the release policy, and who
         trained the model (for separation of duties later)
  edge   one Kubernetes pod with edge-like resources: export to ONNX, check it
         matches the PyTorch model, benchmark latency. Logs the decision to
         MLflow; if every check passed, registers the ONNX model as a new
         version of "cholecseg-segmenter" with the alias "candidate"
  end    print the report

Every decision (pass or reject) is also written as a write-once record in the
locked datasets bucket under _release-records/.

Run (Mac, tunnels open, credentials loaded):
  python flows/release_gate_flow.py run --training-run <mlflow training run id>
  python flows/release_gate_flow.py run          # newest training run not yet gated
"""
import os

from metaflow import FlowSpec, Parameter, current, environment, kubernetes, schedule, step

GATE_IMAGE = "surgseg-gate:0.1"
IN_CLUSTER_MLFLOW = "http://mlflow.default.svc.cluster.local:5000"
# On the Mac, MLflow is reached through the tunnel; inside a pod (scheduled
# runs on Argo), through its in-cluster address.
LOCAL_MLFLOW = os.environ.get("MLFLOW_LOCAL_URI") or (
    IN_CLUSTER_MLFLOW if "KUBERNETES_SERVICE_HOST" in os.environ else "http://localhost:5050")


def trained_by(pathspec):
    """The Metaflow user who ran the training flow."""
    from metaflow import Run, namespace
    try:
        namespace(None)
        flow, run_number = pathspec.split("/")[:2]
        tags = Run(f"{flow}/{run_number}").system_tags
        return next((t.split(":", 1)[1] for t in tags if t.startswith("user:")), "unknown")
    except Exception:
        return "unknown"


def latest_ungated_training_run(client):
    """Newest finished training run that no gate run has evaluated yet."""
    exp = client.get_experiment_by_name("cholecseg-segmentation")
    runs = client.search_runs([exp.experiment_id], "attributes.status = 'FINISHED'",
                              order_by=["attributes.start_time DESC"], max_results=1)
    if not runs:
        return None
    gate = client.get_experiment_by_name("release-gate")
    if gate and client.search_runs([gate.experiment_id],
                                   f"tags.`source.run_id` = '{runs[0].info.run_id}'"):
        return None
    return runs[0].info.run_id


# When deployed to Argo Workflows: monthly, 07:00 UTC on the 1st (after
# training). Approval always stays a manual, human step.
@schedule(cron="0 7 1 * *")
class ReleaseGateFlow(FlowSpec):

    # Not "run-id": Metaflow reserves that name. Empty = newest ungated run.
    training_run = Parameter("training-run", default="",
                             help="MLflow run ID of the training run to evaluate")

    @step
    def start(self):
        from mlflow import MlflowClient

        from surgseg.gate import POLICY, check_lineage, check_quality
        from surgseg.lake import lake_client

        client = MlflowClient(LOCAL_MLFLOW)
        self.source_run_id = self.training_run or latest_ungated_training_run(client)
        self.nothing_to_do = not self.source_run_id
        self.report = None
        if self.nothing_to_do:
            print("No new training run to gate.")
        else:
            run = client.get_run(self.source_run_id)
            self.source_tags = dict(run.data.tags)
            self.source_params = dict(run.data.params)
            self.source_metrics = dict(run.data.metrics)

            baseline, self.baseline_version = None, None
            try:
                approved = client.get_model_version_by_alias(POLICY["registered_model"], "approved")
                baseline, self.baseline_version = float(approved.tags["test_mean_iou"]), approved.version
            except Exception:
                pass  # nothing approved yet

            self.checks = check_lineage(run, lake_client()) + check_quality(self.source_metrics, baseline)
            self.trained_by = trained_by(self.source_tags.get("metaflow.pathspec", ""))
            self.prechecks_passed = all(c["passed"] for c in self.checks)
            print(f"run {self.source_run_id}: trained by {self.trained_by}, "
                  f"pre-checks {'passed' if self.prechecks_passed else 'FAILED'}")
        self.next(self.edge)

    @environment(vars={"MLFLOW_TRACKING_URI": IN_CLUSTER_MLFLOW})
    @kubernetes(image=GATE_IMAGE, cpu=2, memory=4096, secrets=["datalake-creds"])
    @step
    def edge(self):
        if not self.nothing_to_do:
            self.report = run_edge_checks_and_record(self)
        self.next(self.end)

    @step
    def end(self):
        r = self.report
        if r is None:
            print("Nothing to gate: no new finished training run.")
            return
        print(f"\nRelease gate for MLflow run {r['source_run_id']} "
              f"(policy {r['policy']['version']})")
        for c in r["checks"]:
            print(f"  [{'PASS' if c['passed'] else 'FAIL'}] {c['check']:28} {c['detail']}")
        print(f"\nDecision: {r['decision'].upper()}")
        if r["model_version"]:
            print(f"Registered {r['policy']['registered_model']} version {r['model_version']} "
                  f"as 'candidate' (trained by {r['trained_by']}).")
            print("Next: a different person approves it with scripts/approve_model.py")


def run_edge_checks_and_record(flow):
    """ONNX export, parity and latency checks; log the decision; register if passed."""
    import glob
    import hashlib
    import tempfile

    import mlflow
    import onnx

    from surgseg.gate import (POLICY, benchmark, check, check_edge,
                              export_onnx, onnx_agreement)
    from surgseg.lake import lake_client, write_json
    from surgseg.train import load_split, resolve_dataset, to_tensor

    name = POLICY["registered_model"]
    latency, onnx_path = None, None
    if flow.prechecks_passed:
        h, w = int(flow.source_params["img_height"]), int(flow.source_params["img_width"])
        model = mlflow.pytorch.load_model(f"runs:/{flow.source_run_id}/model")
        s3 = lake_client()
        manifest, _, _ = resolve_dataset(s3, flow.source_tags["dataset.name"],
                                         flow.source_tags["dataset.id"])
        images, _ = load_split(s3, manifest, "test", h, w, max_frames=32)
        onnx_path = export_onnx(model, h, w, os.path.join(tempfile.mkdtemp(), "model.onnx"))
        agreement, max_diff = onnx_agreement(model, onnx_path, to_tensor(images).numpy())
        latency = benchmark(onnx_path, h, w, POLICY["edge_cpu_threads"])
        edge_checks = check_edge(agreement, max_diff, latency)
    else:
        edge_checks = [check("edge checks", False, "skipped: earlier checks failed")]

    all_checks = flow.checks + edge_checks
    decision = "passed" if all(c["passed"] for c in all_checks) else "rejected"
    model_version = None

    mlflow.set_experiment("release-gate")
    with mlflow.start_run(run_name=f"gate-{flow.source_run_id[:8]}") as gate_run:
        gate_run_id = gate_run.info.run_id
        mlflow.set_tags({
            "gate.decision": decision,
            "gate.policy": POLICY["version"],
            "source.run_id": flow.source_run_id,
            "dataset.id": flow.source_tags.get("dataset.id", ""),
            "code.git_commit": flow.source_tags.get("code.git_commit", ""),
            "metaflow.pathspec": current.pathspec,
        })
        if latency:
            mlflow.log_metrics({f"latency_{k}_ms": v for k, v in latency.items()})
        if decision == "passed":
            # One self-contained file, so its checksum covers every weight.
            info = mlflow.onnx.log_model(onnx.load(onnx_path), name="edge_model",
                                         registered_model_name=name,
                                         save_as_external_data=False)
            model_version = str(info.registered_model_version)
            client = mlflow.MlflowClient()
            version_tags = {
                "trained_by": flow.trained_by,
                "source.run_id": flow.source_run_id,
                "dataset.id": flow.source_tags["dataset.id"],
                "code.git_commit": flow.source_tags["code.git_commit"],
                "gate.run_id": gate_run_id,
                "gate.policy": POLICY["version"],
                "gate.decision": "passed",
                "test_mean_iou": flow.source_metrics["test_mean_iou"],
                "latency_p95_ms": latency["p95"],
                "input_shape": f"1x3x{flow.source_params['img_height']}x{flow.source_params['img_width']}",
                "approval.status": "pending",
            }
            for k, v in version_tags.items():
                client.set_model_version_tag(name, model_version, k, str(v))
            # Checksum of the exact file edge devices will download. Edge
            # servers refuse to run a model whose file doesn't match it.
            local = mlflow.artifacts.download_artifacts(f"models:/{name}/{model_version}")
            onnx_file = glob.glob(os.path.join(local, "**", "*.onnx"), recursive=True)[0]
            sha = hashlib.sha256(open(onnx_file, "rb").read()).hexdigest()
            client.set_model_version_tag(name, model_version, "edge_model.sha256", sha)
            client.update_model_version(name, model_version, description=POLICY["intended_use"])
            client.set_registered_model_alias(name, "candidate", model_version)

        report = {
            "gate_run_id": gate_run_id,
            "source_run_id": flow.source_run_id,
            "policy": POLICY,
            "decision": decision,
            "trained_by": flow.trained_by,
            "model_version": model_version,
            "baseline_version": flow.baseline_version,
            "checks": all_checks,
            "latency_ms": latency,
        }
        mlflow.log_dict(report, "gate_report.json")

    # Write-once audit record of the decision, pass or reject.
    write_json(lake_client(), "datasets",
               f"_release-records/{name}/gate-{gate_run_id}.json", report)
    return report


if __name__ == "__main__":
    ReleaseGateFlow()