"""Step 2 smoke test: a Kubernetes pod logs an experiment to MLflow.

Proves:
1. Pods can reach MLflow by its in-cluster address.
2. MLflow stores the run's files in MinIO.
3. Every MLflow run is tagged with the Metaflow run that produced it
   (the first link in our lineage chain).

Requires `kubectl port-forward svc/mlflow 5050:5000` running on your Mac,
so the final step can verify the run.
"""
import os

from metaflow import FlowSpec, current, environment, kubernetes, step

IN_CLUSTER_MLFLOW = "http://mlflow.default.svc.cluster.local:5000"
LOCAL_MLFLOW = os.environ.get("MLFLOW_LOCAL_URI", "http://localhost:5050")
RUNTIME_IMAGE = "surgseg-runtime:0.1"


class MlflowSmokeTestFlow(FlowSpec):

    @step
    def start(self):
        self.next(self.log_run)

    @environment(vars={"MLFLOW_TRACKING_URI": IN_CLUSTER_MLFLOW})
    @kubernetes(image=RUNTIME_IMAGE, cpu=0.5, memory=1024)
    @step
    def log_run(self):
        import mlflow

        mlflow.set_experiment("platform-smoke-test")
        with mlflow.start_run(run_name=f"metaflow-{current.run_id}") as run:
            # Lineage tags: which Metaflow run, step, and image produced this.
            mlflow.set_tags({
                "metaflow.flow": current.flow_name,
                "metaflow.run_id": current.run_id,
                "metaflow.pathspec": current.pathspec,
                "runtime.image": RUNTIME_IMAGE,
            })
            mlflow.log_param("example_param", 42)
            mlflow.log_metric("example_metric", 0.95)
            mlflow.log_text("written from a Kubernetes pod", "hello.txt")
            self.mlflow_run_id = run.info.run_id
        self.next(self.end)

    @step
    def end(self):
        # Back on the Mac: read the run back from MLflow to verify it landed.
        from mlflow import MlflowClient

        client = MlflowClient(tracking_uri=LOCAL_MLFLOW)
        run = client.get_run(self.mlflow_run_id)
        artifacts = [a.path for a in client.list_artifacts(self.mlflow_run_id)]

        print(f"MLflow run:    {self.mlflow_run_id}")
        print(f"Metaflow tag:  {run.data.tags.get('metaflow.pathspec')}")
        print(f"Metric:        {run.data.metrics.get('example_metric')}")
        print(f"Artifacts:     {artifacts}")
        assert "hello.txt" in artifacts, "artifact missing"
        print("MLflow smoke test passed: pod -> MLflow -> MinIO working.")


if __name__ == "__main__":
    MlflowSmokeTestFlow()