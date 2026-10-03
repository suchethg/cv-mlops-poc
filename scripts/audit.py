"""Trace a deployed model back to the raw device uploads it was trained on.

Walks the chain backward and checks every link against its source of truth:
  1. device          what the edge site says it's running (version, checksum)
  2. model version   gate decision, approval, write-once decision records
  3. training run    code commit (published?), runtime image, metrics
  4. dataset release manifest still locked; its ID recomputed from content
  5. curated clips   de-identification method, pseudonymized cases, ingest run
  6. device uploads  original upload IDs, devices, sites (no patient data shown)

Prints a report, saves it to out/, and exits non-zero if any link is broken.

Usage (Mac, tunnels open, credentials loaded):
  python scripts/audit.py --site boston
  python scripts/audit.py --version 2          # audit a registry version directly
"""
import argparse
import collections
import datetime
import json
import os
import subprocess
import sys

import requests
from mlflow import MlflowClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "flows"))
from surgseg.datasets import dataset_id as compute_dataset_id  # noqa: E402
from surgseg.lake import exists, lake_client, read_json  # noqa: E402

NAME = "cholecseg-segmenter"
SITE_PORTS = {"boston": 8081, "denver": 8082}
LOCKED = ("GOVERNANCE", "COMPLIANCE")


class Audit:
    def __init__(self):
        self.sections, self.failures, self.warnings = [], 0, 0

    def section(self, title):
        self.sections.append({"title": title, "facts": [], "checks": []})

    def fact(self, label, value):
        self.sections[-1]["facts"].append((label, value))

    def check(self, label, ok, detail=""):
        self.sections[-1]["checks"].append((label, "OK" if ok else "BROKEN", detail))
        self.failures += 0 if ok else 1

    def warn(self, label, detail=""):
        """Worth a human look, but not a broken link."""
        self.sections[-1]["checks"].append((label, "WARN", detail))
        self.warnings += 1

    def render(self):
        lines = []
        for i, s in enumerate(self.sections, 1):
            lines += ["", f"{i}. {s['title']}"]
            lines += [f"     {label:22} {value}" for label, value in s["facts"]]
            lines += [f"   [{status}] {label}{(': ' + d) if d else ''}"
                      for label, status, d in s["checks"]]
        return "\n".join(lines)


def retention(s3, bucket, key, version=None):
    try:
        args = {"Bucket": bucket, "Key": key, **({"VersionId": version} if version else {})}
        r = s3.get_object_retention(**args)["Retention"]
        return r["Mode"], str(r["RetainUntilDate"])[:10]
    except Exception as e:
        return None, type(e).__name__


def git(*args):
    try:
        return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return None


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    target = p.add_mutually_exclusive_group(required=True)
    target.add_argument("--site", choices=SITE_PORTS)
    target.add_argument("--version")
    p.add_argument("--mlflow", default=os.environ.get("MLFLOW_LOCAL_URI", "http://localhost:5050"))
    args = p.parse_args()

    a, s3, client = Audit(), lake_client(), MlflowClient(args.mlflow)

    # 1. Device
    device = {}
    if args.site:
        a.section(f"Device: site-{args.site}")
        try:
            device = requests.get(f"http://localhost:{SITE_PORTS[args.site]}/model", timeout=10).json()
            a.fact("serving", f"{device['model']} v{device['version']}")
            a.fact("model sha256", device["onnx_sha256"][:16] + "...")
            a.fact("loaded at", device["loaded_at"])
            a.check("device reachable", True)
        except Exception as e:
            a.check("device reachable", False, f"{e} (is scripts/port_forwards.sh running?)")
            print(a.render())
            sys.exit(1)
    version = device.get("version") or args.version

    # 2. Model version
    a.section(f"Model version: {NAME} v{version}")
    vtags = client.get_model_version(NAME, version).tags
    aliases = client.get_registered_model(NAME).aliases
    a.fact("aliases", ", ".join(k for k, v in aliases.items() if str(v) == str(version)) or "none")
    a.fact("gate", f"{vtags.get('gate.decision')} under {vtags.get('gate.policy')}")
    a.fact("trained by", vtags.get("trained_by"))
    a.fact("approved by", f"{vtags.get('approval.by')} at {vtags.get('approval.at')}")
    a.fact("approval reason", vtags.get("approval.reason"))
    if device:
        a.check("device file matches registry checksum",
                device["onnx_sha256"] == vtags.get("edge_model.sha256"))
    a.check("passed release gate", vtags.get("gate.decision") == "passed")
    a.check("approved by a different person than the trainer",
            vtags.get("approval.status") == "approved"
            and vtags.get("approval.by", "").lower() != vtags.get("trained_by", "").lower(),
            f"{vtags.get('trained_by')} -> {vtags.get('approval.by')}")
    for label, key in (("gate record", f"_release-records/{NAME}/gate-{vtags.get('gate.run_id')}.json"),
                       ("approval record", f"_release-records/{NAME}/v{version}-decision.json")):
        mode, until = retention(s3, "datasets", key)
        a.check(f"{label} write-once", mode in LOCKED, f"datasets/{key} ({mode or until})")
    if exists(s3, "datasets", f"_release-records/{NAME}/v{version}-decision.json"):
        record = read_json(s3, "datasets", f"_release-records/{NAME}/v{version}-decision.json")
        a.check("approval record matches registry",
                record.get("approver") == vtags.get("approval.by")
                and record.get("edge_model_sha256") == vtags.get("edge_model.sha256"))

    # 3. Training run
    run_id = vtags.get("source.run_id")
    run = client.get_run(run_id)
    rt, metrics = run.data.tags, run.data.metrics
    commit = rt.get("code.git_commit", "")
    a.section(f"Training run: MLflow {run_id}")
    a.fact("metaflow run", rt.get("metaflow.pathspec"))
    a.fact("runtime image", rt.get("runtime.image"))
    a.fact("code commit", commit)
    a.fact("test mean IoU", round(metrics.get("test_mean_iou", 0), 3))
    a.fact("test cases", rt.get("dataset.test_cases"))
    a.check("run finished", run.info.status == "FINISHED", run.info.status)
    a.check("trained on committed code", rt.get("code.git_dirty") == "False")
    a.check("commit exists in this repository", git("cat-file", "-e", f"{commit}^{{commit}}") is not None)
    published = git("branch", "-r", "--contains", commit)
    a.check("commit is published (on a remote branch)", bool(published), published or "not pushed")

    # 4. Dataset release
    manifest_key, manifest_version = rt.get("dataset.manifest_key"), rt.get("dataset.manifest_version")
    a.section(f"Dataset release: {rt.get('dataset.id')}")
    manifest = json.loads(s3.get_object(Bucket="datasets", Key=manifest_key,
                                        VersionId=manifest_version)["Body"].read())
    a.fact("released by", manifest["release_run"])
    a.fact("created", manifest["created_at"])
    for split, v in manifest["summary"]["splits"].items():
        a.fact(f"{split} split", f"{v['frames']} frames, {v['clips']} clips, cases {', '.join(v['cases'])}")
    mode, until = retention(s3, "datasets", manifest_key, manifest_version)
    a.check("manifest under object lock", mode in LOCKED, f"{mode} until {until}" if mode else until)
    recomputed = compute_dataset_id(manifest["name"], manifest["entries"],
                                    manifest["test_cases"], manifest["label_source"])
    a.check("dataset ID recomputed from content matches", recomputed == manifest["dataset_id"]
            == rt.get("dataset.id"), recomputed)
    a.check("test split is the golden test set",
            manifest["test_cases"] == read_json(s3, "datasets", "_golden/test-cases.json")["test_cases"])

    # 5. Curated clips
    hashes = collections.defaultdict(set)
    for e in manifest["entries"]:
        hashes[(e["case_id"], e["clip"])].add(e["sha256"])
    a.section(f"Curated clips: {len(hashes)} clips from {len({c for c, _ in hashes})} cases")
    deid, ingest_runs, uploads, pseudonyms, mismatched = set(), set(), [], set(), []
    case_patients = collections.defaultdict(set)
    for (case, clip), expected in sorted(hashes.items()):
        rec = read_json(s3, "curated", f"cases/{case}/{clip}/record.json")
        if not expected <= {f["sha256"] for f in rec["files"]}:
            mismatched.append(clip)
        deid.add(rec["deid_method"])
        ingest_runs.add(rec["ingest_run"].split("/")[1])
        pseudonyms.add(rec["patient_pseudonym"])
        case_patients[case].add(rec["patient_pseudonym"])
        uploads.append(rec["source_upload"])
    a.fact("de-identification", ", ".join(sorted(deid)))
    a.fact("patients", f"{len(pseudonyms)} pseudonymized (no names or MRNs stored)")
    a.fact("ingest runs", "IngestFlow " + ", ".join(sorted(ingest_runs)))
    a.check("every dataset file traces to a curated record", not mismatched,
            f"mismatched: {mismatched}" if mismatched else f"{sum(map(len, hashes.values()))} files")
    multi = sorted(c for c, pts in case_patients.items() if len(pts) > 1)
    if multi:
        a.warn("cases with more than one patient pseudonym",
               f"{len(multi)} of {len(case_patients)} (one surgical case should be one patient)")

    # 6. Device uploads
    a.section(f"Device uploads: {len(uploads)} uploads in quarantine")
    sites, devices, missing = collections.Counter(), set(), []
    for up in uploads:
        marker = f"processed/{up}.json"
        if not exists(s3, "quarantine", marker) or \
                read_json(s3, "quarantine", marker).get("status") != "accepted":
            missing.append(up)
        site_part = up.split("-dvc-")[0]
        sites[site_part] += 1
        devices.add("dvc-" + up.split("-dvc-")[1].split("-")[0])
    a.fact("sites", ", ".join(f"{s} ({n})" for s, n in sorted(sites.items())))
    a.fact("devices", len(devices))
    a.fact("example upload", uploads[0] if uploads else "-")
    a.check("every clip came from an accepted upload", not missing,
            f"missing: {missing[:3]}" if missing else f"{len(uploads)} accepted uploads")

    # Report
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    title = f"AUDIT {NAME} v{version}" + (f" on site-{args.site}" if args.site else "")
    verdict = "CHAIN INTACT" if a.failures == 0 else f"{a.failures} BROKEN LINK(S)"
    if a.warnings:
        verdict += f", {a.warnings} warning(s)"
    text = f"{title}  ({stamp})\n{a.render()}\n\nResult: {verdict}\n"
    print(text)
    os.makedirs("out", exist_ok=True)
    path = f"out/audit-{args.site or 'v' + str(version)}-{stamp}.txt"
    with open(path, "w") as f:
        f.write(text)
    print(f"Saved to {path}")
    sys.exit(0 if a.failures == 0 else 1)


if __name__ == "__main__":
    main()