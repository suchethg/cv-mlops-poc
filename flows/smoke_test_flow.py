"""Step 1 smoke test.

Proves two things:
1. Metaflow can schedule a step as a pod in Kubernetes.
2. Data saved in one step reaches the next through the MinIO datastore.
"""

from metaflow import FlowSpec, kubernetes, step


class SmokeTestFlow(FlowSpec):

    @step
    def start(self):
        # Runs locally anything assigend to self is saved to MinIO. 
        self.message = "Hello from step 1"
        self.next(self.on_k8s)

    @kubernetes(cpu=0.5,memory=512)
    @step
    def on_k8s(self):
        # Runs in a Kubernetes pod. self.message was loaded back from MinIO.
        import platform
        self.pod_name = platform.node()
        self.reply = f"{self.message} -> received in pod {self.pod_name}"
        self.next(self.end)

    @step
    def end(self):
        print(self.reply)
        print(f"Smoke test passed: Kubernetes execution + MinIO datastore working.")


if __name__ == "__main__":
    SmokeTestFlow()
