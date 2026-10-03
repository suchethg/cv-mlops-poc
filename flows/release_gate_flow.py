"""Step 6: release gate. Decides whether a trained model may become a release
candidate.

  start  on the Mac: lineage and quality checks against the release policy,
         look up who trained the model (for separation of duties later)
  edge   one Kubernetes pod with edge-like resources: export to ONNX, check it
         matches the PyTorch model, benchmark latency. Logs the decision to
         MLflow; if every check passed, registers the ONNX model as a new
         version of "cholecseg-segmenter" with the alias "candidate"
  end    print the report

Every decision (pass or reject) is also written as a write-once record in the
locked datasets bucket under _release-records/.

Run (Mac, tunnels open, credentials loaded):
  python flows/release_gate_flow.py run --training-run <mlflow training run id>
"""
import os
import glob
import hashlib
from metaflow import FlowSpec, Parameter, current, environment, kubernetes, step

GATE_IMAGE = "surgseg-gate:0.1"
IN_CLUSTER_MLFLOW = "http://mlflow.default.svc.cluster.local:5000"
LOCAL_MLFLOW = os.environ.get("MLFLOW_LOCAL_URI", "http://localhost:5050")


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


class ReleaseGateFlow(FlowSpec):

    # Not "run-id": Metaflow reserves that name.
    training_run = Parameter("training-run", required=True,
                             help="MLflow run ID of the training run to evaluate")

    @step
    def start(self):
        from mlflow import MlflowClient

        from surgseg.gate import POLICY, check_lineage, check_quality
        from surgseg.lake import lake_client

        client = MlflowClient(LOCAL_MLFLOW)
        run = client.get_run(self.training_run)
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
        print(f"run {self.training_run}: trained by {self.trained_by}, "
              f"pre-checks {'passed' if self.prechecks_passed else 'FAILED'}")
        self.next(self.edge)

    @environment(vars={"MLFLOW_TRACKING_URI": IN_CLUSTER_MLFLOW})
    @kubernetes(image=GATE_IMAGE, cpu=2, memory=4096, secrets=["datalake-creds"])
    @step
    def edge(self):
        import tempfile

        import mlflow
        import onnx

        from surgseg.gate import (POLICY, benchmark, check, check_edge,
                                  export_onnx, onnx_agreement)
        from surgseg.lake import lake_client, write_json
        from surgseg.train import load_split, resolve_dataset, to_tensor

        name = POLICY["registered_model"]
        self.latency, onnx_path = None, None
        if self.prechecks_passed:
            h, w = int(self.source_params["img_height"]), int(self.source_params["img_width"])
            model = mlflow.pytorch.load_model(f"runs:/{self.training_run}/model")
            s3 = lake_client()
            manifest, _, _ = resolve_dataset(s3, self.source_tags["dataset.name"],
                                             self.source_tags["dataset.id"])
            images, _ = load_split(s3, manifest, "test", h, w, max_frames=32)
            onnx_path = export_onnx(model, h, w, os.path.join(tempfile.mkdtemp(), "model.onnx"))
            agreement, max_diff = onnx_agreement(model, onnx_path, to_tensor(images).numpy())
            self.latency = benchmark(onnx_path, h, w, POLICY["edge_cpu_threads"])
            self.edge_checks = check_edge(agreement, max_diff, self.latency)
        else:
            self.edge_checks = [check("edge checks", False, "skipped: earlier checks failed")]

        all_checks = self.checks + self.edge_checks
        self.decision = "passed" if all(c["passed"] for c in all_checks) else "rejected"
        self.model_version = None

        mlflow.set_experiment("release-gate")
        with mlflow.start_run(run_name=f"gate-{self.training_run[:8]}") as gate_run:
            self.gate_run_id = gate_run.info.run_id
            mlflow.set_tags({
                "gate.decision": self.decision,
                "gate.policy": POLICY["version"],
                "source.run_id": self.training_run,
                "dataset.id": self.source_tags.get("dataset.id", ""),
                "code.git_commit": self.source_tags.get("code.git_commit", ""),
                "metaflow.pathspec": current.pathspec,
            })
            if self.latency:
                mlflow.log_metrics({f"latency_{k}_ms": v for k, v in self.latency.items()})
            if self.decision == "passed":
                # One self-contained file, so its checksum covers every weight.
                info = mlflow.onnx.log_model(onnx.load(onnx_path), name="edge_model",
                                             registered_model_name=name,
                                             save_as_external_data=False)
                self.model_version = str(info.registered_model_version)
                client = mlflow.MlflowClient()
                version_tags = {
                    "trained_by": self.trained_by,
                    "source.run_id": self.training_run,
                    "dataset.id": self.source_tags["dataset.id"],
                    "code.git_commit": self.source_tags["code.git_commit"],
                    "gate.run_id": self.gate_run_id,
                    "gate.policy": POLICY["version"],
                    "gate.decision": "passed",
                    "test_mean_iou": self.source_metrics["test_mean_iou"],
                    "latency_p95_ms": self.latency["p95"],
                    "input_shape": f"1x3x{self.source_params['img_height']}x{self.source_params['img_width']}",
                    "approval.status": "pending",
                }
                for k, v in version_tags.items():
                    client.set_model_version_tag(name, self.model_version, k, str(v))
                # Checksum of the exact file edge devices will download. Edge
                # servers refuse to run a model whose file doesn't match it.
                local = mlflow.artifacts.download_artifacts(f"models:/{name}/{self.model_version}")
                onnx_file = glob.glob(os.path.join(local, "**", "*.onnx"), recursive=True)[0]
                self.edge_sha256 = hashlib.sha256(open(onnx_file, "rb").read()).hexdigest()
                client.set_model_version_tag(name, self.model_version,
                                             "edge_model.sha256", self.edge_sha256)
                client.update_model_version(name, self.model_version,
                                            description=POLICY["intended_use"])
                client.set_registered_model_alias(name, "candidate", self.model_version)

            self.report = {
                "gate_run_id": self.gate_run_id,
                "source_run_id": self.training_run,
                "policy": POLICY,
                "decision": self.decision,
                "trained_by": self.trained_by,
                "model_version": self.model_version,
                "baseline_version": self.baseline_version,
                "checks": all_checks,
                "latency_ms": self.latency,
            }
            mlflow.log_dict(self.report, "gate_report.json")

        # Write-once audit record of the decision, pass or reject.
        write_json(lake_client(), "datasets",
                   f"_release-records/{name}/gate-{self.gate_run_id}.json", self.report)
        self.next(self.end)

    @step
    def end(self):
        r = self.report
        print(f"\nRelease gate for MLflow run {r['source_run_id']} "
              f"(policy {r['policy']['version']})")
        for c in r["checks"]:
            print(f"  [{'PASS' if c['passed'] else 'FAIL'}] {c['check']:28} {c['detail']}")
        print(f"\nDecision: {r['decision'].upper()}")
        if r["model_version"]:
            print(f"Registered {r['policy']['registered_model']} version {r['model_version']} "
                  f"as 'candidate' (trained by {r['trained_by']}).")
            print("Next: a different person approves it with scripts/approve_model.py")


if __name__ == "__main__":
    ReleaseGateFlow()