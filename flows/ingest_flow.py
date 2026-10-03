"""Step 3: ingest and quarantine.

Finds new device uploads in the `quarantine` bucket and, for each one in its
own Kubernetes pod, checks it and either:
  - de-identifies it and promotes it to `curated`, or
  - records why it was rejected in `rejected`.
Finishes with an ingest report saved to the data lake.

Run (Mac, tunnels open, credentials loaded):
  python flows/ingest_flow.py run --max-workers 4
"""
from metaflow import FlowSpec, current, kubernetes, schedule, step

RUNTIME_IMAGE = "surgseg-runtime:0.2"
NOTHING_TO_DO = "__none__"

# When deployed to Argo Workflows: check quarantine for new uploads every hour.
@schedule(cron="0 * * * *")
class IngestFlow(FlowSpec):

    @step
    def start(self):
        from surgseg.ingest import pending_uploads
        from surgseg.lake import lake_client

        self.pending = pending_uploads(lake_client())
        print(f"{len(self.pending)} new upload(s) in quarantine")
        # Metaflow can't fan out over an empty list, so use a placeholder.
        self.work = self.pending or [NOTHING_TO_DO]
        self.next(self.process, foreach="work")

    @kubernetes(image=RUNTIME_IMAGE, cpu=0.5, memory=1024, secrets=["datalake-creds"])
    @step
    def process(self):
        from surgseg.ingest import process_upload
        from surgseg.lake import lake_client

        if self.input == NOTHING_TO_DO:
            self.result = None
        else:
            self.result = process_upload(lake_client(), self.input, current.pathspec)
            print(self.result["status"], self.input, self.result.get("reasons", ""))
        self.next(self.join)

    @step
    def join(self, inputs):
        from surgseg.lake import lake_client, write_json

        self.results = [i.result for i in inputs if i.result]
        accepted = [r for r in self.results if r["status"] == "accepted"]
        rejected = [r for r in self.results if r["status"] == "rejected"]
        report = {
            "ingest_run": f"{current.flow_name}/{current.run_id}",
            "accepted": len(accepted),
            "rejected": len(rejected),
            "frames_curated": sum(r["frames"] for r in accepted),
            "results": self.results,
        }
        write_json(lake_client(), "curated",
                   f"_reports/ingest-{current.run_id}.json", report)
        self.report = report
        self.next(self.end)

    @step
    def end(self):
        r = self.report
        print(f"\nIngest report for {r['ingest_run']}")
        print(f"  accepted: {r['accepted']}   rejected: {r['rejected']}   "
              f"frames curated: {r['frames_curated']}")
        for res in r["results"]:
            detail = "" if res["status"] == "accepted" else "  <- " + "; ".join(res["reasons"])
            print(f"  [{res['status']:8}] {res['upload_id']}{detail}")


if __name__ == "__main__":
    IngestFlow()