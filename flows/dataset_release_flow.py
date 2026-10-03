"""Step 4: freeze curated data into an immutable, versioned dataset.

  start    list curated clips (pinned to exact record versions), load or
           create the golden test set, assign splits by case, check that no
           patient appears in both splits
  freeze   one Kubernetes pod per case: verify every file's checksum, store
           it by content hash in the locked datasets bucket, count label pixels
  join     build the manifest, compute the content-hash dataset ID, publish
           manifest + dataset card (skipped if identical data was released)

Run (Mac, tunnels open, credentials loaded):
  python flows/dataset_release_flow.py run --max-workers 4
"""
from metaflow import FlowSpec, Parameter, current, kubernetes, retry, schedule, step

RUNTIME_IMAGE = "surgseg-runtime:0.2"

# When deployed to Argo Workflows: monthly, 02:00 UTC on the 1st.
@schedule(cron="0 2 1 * *")
class DatasetReleaseFlow(FlowSpec):

    # Not called "name": FlowSpec already uses self.name for the flow's own name.
    dataset_name = Parameter("dataset", default="cholecseg", help="dataset name")
    test_fraction = Parameter("test-fraction", default=0.2,
                              help="share of cases in the golden test set (first release only)")

    @step
    def start(self):
        from surgseg.datasets import check_no_leakage, golden_test_cases, list_curated_clips
        from surgseg.lake import lake_client

        s3 = lake_client()
        clips = list_curated_clips(s3)
        if not clips:
            raise ValueError("No curated clips. Run the ingest flow first.")
        cases = sorted({c["case_id"] for c in clips})
        self.test_cases, created = golden_test_cases(
            s3, cases, self.test_fraction, current.pathspec)
        check_no_leakage(clips, self.test_cases)
        self.label_sources = sorted({c["label_source"] for c in clips})
        if len(self.label_sources) != 1:
            raise ValueError(f"Mixed label sources: {self.label_sources}")

        self.case_work = [
            {"case_id": case,
             "split": "test" if case in self.test_cases else "train",
             "clips": [c for c in clips if c["case_id"] == case]}
            for case in cases
        ]
        print(f"{len(clips)} clips across {len(cases)} cases")
        print(f"golden test cases ({'new' if created else 'existing'}): {self.test_cases}")
        self.next(self.freeze, foreach="case_work")

    @retry(times=2, minutes_between_retries=1)
    @kubernetes(image=RUNTIME_IMAGE, cpu=0.5, memory=1024, secrets=["datalake-creds"])
    @step
    def freeze(self):
        from surgseg.datasets import freeze_case
        from surgseg.lake import lake_client

        work = self.input
        self.entries, self.pixel_counts, self.unknown = freeze_case(
            lake_client(), work["clips"], work["split"])
        print(f"{work['case_id']} ({work['split']}): {len(self.entries)} files verified")
        self.next(self.join)

    @step
    def join(self, inputs):
        from surgseg.datasets import SCHEMA, dataset_id, now, publish, summarize
        from surgseg.lake import lake_client

        self.merge_artifacts(inputs, include=["test_cases", "label_sources"])
        entries = sorted((e for i in inputs for e in i.entries), key=lambda e: e["path"])
        pixel_counts, unknown = {}, {}
        for i in inputs:
            for k, v in i.pixel_counts.items():
                pixel_counts[k] = pixel_counts.get(k, 0) + v
            for k, v in i.unknown.items():
                unknown[k] = unknown.get(k, 0) + v

        label_source = self.label_sources[0]
        self.dataset_id = dataset_id(self.dataset_name, entries, self.test_cases, label_source)
        manifest = {
            "schema": SCHEMA,
            "name": self.dataset_name,
            "dataset_id": self.dataset_id,
            "created_at": now(),
            "release_run": f"{current.flow_name}/{current.run_id}",
            "label_source": label_source,
            "test_cases": self.test_cases,
            "summary": summarize(entries, pixel_counts),
            "unknown_mask_values": unknown,
            "entries": entries,
        }
        self.summary = manifest["summary"]
        self.published = publish(lake_client(), manifest)
        self.next(self.end)

    @step
    def end(self):
        state = "published" if self.published else "already released (identical content)"
        print(f"\nDataset {self.dataset_name} / {self.dataset_id}: {state}")
        for split, v in self.summary["splits"].items():
            print(f"  {split:5}  {v['frames']:5} frames  {v['clips']:3} clips  cases: {', '.join(v['cases'])}")
        print("  class pixel share:")
        for c, share in self.summary["class_pixel_share"].items():
            print(f"    {c:24} {share:6.2%}")


if __name__ == "__main__":
    DatasetReleaseFlow()