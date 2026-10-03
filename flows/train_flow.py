"""Step 5: train a segmentation model on one frozen dataset version.

  start  resolve the dataset release (latest unless --dataset-id is given),
         record the git commit of the code being run
  train  one Kubernetes pod: download + verify the dataset, train, evaluate on
         the golden test cases, log everything to MLflow with lineage tags
  end    print the results and the MLflow link

Run (Mac, tunnels open, credentials loaded):
  python flows/train_flow.py run
  python flows/train_flow.py run --epochs 1 --max-train-frames 400   # quick test
"""
import subprocess

from metaflow import FlowSpec, Parameter, current, environment, kubernetes, step

TRAIN_IMAGE = "surgseg-train:0.1"
IN_CLUSTER_MLFLOW = "http://mlflow.default.svc.cluster.local:5000"
EXPERIMENT = "cholecseg-segmentation"


def git(*args):
    try:
        return subprocess.check_output(["git", *args], text=True).strip()
    except Exception:
        return "unknown"


class TrainFlow(FlowSpec):

    dataset_name = Parameter("dataset", default="cholecseg")
    dataset_id = Parameter("dataset-id", default="", help="empty = latest release")
    epochs = Parameter("epochs", default=4)
    batch_size = Parameter("batch-size", default=8)
    lr = Parameter("lr", default=1e-3)
    img_height = Parameter("img-height", default=160)
    img_width = Parameter("img-width", default=288)
    max_train_frames = Parameter("max-train-frames", default=0, help="0 = all")
    seed = Parameter("seed", default=0)

    @step
    def start(self):
        from surgseg.lake import lake_client
        from surgseg.train import resolve_dataset

        manifest, self.manifest_key, self.manifest_version = resolve_dataset(
            lake_client(), self.dataset_name, self.dataset_id)
        self.resolved_dataset_id = manifest["dataset_id"]
        self.git_commit = git("rev-parse", "HEAD")
        self.git_dirty = git("status", "--porcelain", "--untracked-files=no") != ""
        print(f"dataset {self.dataset_name}/{self.resolved_dataset_id}")
        print(f"code    {self.git_commit}{' (UNCOMMITTED CHANGES)' if self.git_dirty else ''}")
        if self.git_dirty:
            print("warning: commit your code first, or this run can't be traced to exact code")
        self.next(self.train)

    @environment(vars={"MLFLOW_TRACKING_URI": IN_CLUSTER_MLFLOW})
    @kubernetes(image=TRAIN_IMAGE, cpu=3, memory=6144, secrets=["datalake-creds"])
    @step
    def train(self):
        import mlflow
        import torch

        from surgseg.lake import lake_client
        from surgseg.train import (build_model, evaluate, load_split, logits_only,
                                   resolve_dataset, to_tensor, train_model)

        torch.set_num_threads(3)
        s3 = lake_client()
        manifest, _, _ = resolve_dataset(s3, self.dataset_name, self.resolved_dataset_id)
        h, w = self.img_height, self.img_width
        x_train, y_train = load_split(s3, manifest, "train", h, w, self.max_train_frames)
        x_test, y_test = load_split(s3, manifest, "test", h, w)
        print(f"loaded {len(x_train)} train / {len(x_test)} test frames (verified)")

        mlflow.set_experiment(EXPERIMENT)
        with mlflow.start_run(run_name=f"train-{current.run_id}") as run:
            mlflow.set_tags({
                "dataset.name": self.dataset_name,
                "dataset.id": self.resolved_dataset_id,
                "dataset.manifest_key": self.manifest_key,
                "dataset.manifest_version": str(self.manifest_version),
                "dataset.test_cases": ",".join(manifest["test_cases"]),
                "code.git_commit": self.git_commit,
                "code.git_dirty": str(self.git_dirty),
                "metaflow.pathspec": current.pathspec,
                "runtime.image": TRAIN_IMAGE,
            })
            mlflow.log_params({
                "model": "lraspp_mobilenet_v3_large", "epochs": self.epochs,
                "batch_size": self.batch_size, "lr": self.lr, "seed": self.seed,
                "img_height": h, "img_width": w,
                "train_frames": len(x_train), "test_frames": len(x_test),
            })

            def on_epoch(epoch, loss, seconds):
                print(f"epoch {epoch}: loss {loss:.4f} ({seconds:.0f}s)")
                mlflow.log_metric("train_loss", loss, step=epoch)

            model = train_model(build_model(), x_train, y_train, self.epochs,
                                self.batch_size, self.lr, self.seed, on_epoch)
            self.metrics = evaluate(model, x_test, y_test)
            mlflow.log_metrics({"test_mean_iou": self.metrics["mean_iou"],
                                "test_pixel_accuracy": self.metrics["pixel_accuracy"]})
            mlflow.log_metrics({f"test_iou_{c}": v for c, v in
                                self.metrics["per_class_iou"].items() if v is not None})
            mlflow.log_dict(self.metrics, "evaluation/test_metrics.json")
            # "pt2" is a traced, pickle-free format (safer to load); it needs an
            # example input to trace the model. The example also defines the
            # model's input signature in MLflow.
            example = to_tensor(x_test[:2]).numpy()  # 2 frames, so batch size stays flexible
            mlflow.pytorch.log_model(logits_only(model), name="model", input_example=example)
            self.mlflow_run_id = run.info.run_id
            self.mlflow_experiment_id = run.info.experiment_id
        self.next(self.end)

    @step
    def end(self):
        m = self.metrics
        print(f"\nModel trained on {self.dataset_name}/{self.resolved_dataset_id}")
        print(f"  test mean IoU:       {m['mean_iou']:.3f}")
        print(f"  test pixel accuracy: {m['pixel_accuracy']:.3f}")
        print("  per-class IoU:")
        for c, v in m["per_class_iou"].items():
            print(f"    {c:24} {'n/a' if v is None else f'{v:.3f}'}")
        print(f"\nMLflow: http://localhost:5050/#/experiments/"
              f"{self.mlflow_experiment_id}/runs/{self.mlflow_run_id}")


if __name__ == "__main__":
    TrainFlow()