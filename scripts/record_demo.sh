#!/usr/bin/env bash
# Self-narrating, end-to-end walk through the whole pipeline, meant to be
# screen-recorded. Run it from the repo root, in a tab set up with
# `source scripts/env.sh`, with `metaflow-dev up` and `scripts/port_forwards.sh`
# already running in other tabs.
#
#   scripts/record_demo.sh
#
# It prints a title and a one-line explanation before each step, pauses so
# the narration has room to breathe, then runs the real commands against the
# real cluster. Nothing here is simulated. It does not commit, push, or touch
# any file other than k8s/edge-sites.yaml (the canary rollout in step 7).
set -uo pipefail
cd "$(dirname "$0")/.."

# ---------- look and feel ----------

if [ -t 1 ]; then
  BOLD=$(tput bold); RESET=$(tput sgr0)
  CYAN=$(tput setaf 6); GREEN=$(tput setaf 2); RED=$(tput setaf 1); YELLOW=$(tput setaf 3)
else
  BOLD=""; RESET=""; CYAN=""; GREEN=""; RED=""; YELLOW=""
fi

title() {
  # title "STEP N · NAME" "one-line explanation"
  echo
  echo "${BOLD}${CYAN}════════════════════════════════════════════════════════════════${RESET}"
  echo "${BOLD}${CYAN}  $1${RESET}"
  echo "${CYAN}  $2${RESET}"
  echo "${BOLD}${CYAN}════════════════════════════════════════════════════════════════${RESET}"
  sleep 3
}

note() { echo "${CYAN}-- $* --${RESET}"; }
ok()   { echo "${GREEN}$*${RESET}"; }
warn() { echo "${YELLOW}$*${RESET}"; }
err()  { echo "${RED}$*${RESET}"; }

die() { err "$*"; exit 1; }

# Strips ANSI colors, Metaflow's "[run/step/task (pid N)] " line prefixes and
# timestamp prefixes, and drops repetitive Metaflow boilerplate lines so the
# real flow output (prints, metrics, results) reads cleanly on camera.
clean_metaflow() {
  sed -E 's/\x1b\[[0-9;]*m//g' \
    | sed -E 's/^[0-9]{4}-[0-9]{2}-[0-9]{2}[ T][0-9]{2}:[0-9]{2}:[0-9]{2}(\.[0-9]+)?[[:space:]]+//' \
    | sed -E 's/^\[[^]]*\][[:space:]]*//' \
    | grep -Ev '^(Metaflow [0-9]|Validating your flow|The graph looks good|Workflow starting|Task is starting|Task finished successfully|Bootstrapping|Logging into|Pulling container image|Returning the task)'
}

LOGDIR=$(mktemp -d "${TMPDIR:-/tmp}/record_demo.XXXXXX")
note "step logs kept in $LOGDIR for troubleshooting"

# run_step <logfile> <cmd...>  — runs a command, streams cleaned output to
# the terminal AND a log file, returns the real command's exit code.
run_step() {
  local log="$1"; shift
  "$@" 2>&1 | clean_metaflow | tee "$log"
  return "${PIPESTATUS[0]}"
}

# ---------- preflight ----------

note "preflight checks"

if [ "${METAFLOW_PROFILE:-}" != "local" ] || [ -z "${LAKE_ENDPOINT:-}" ] \
   || [ -z "${LAKE_ACCESS_KEY:-}" ] || [ -z "${LAKE_SECRET_KEY:-}" ]; then
  die "This tab isn't set up. Run: source scripts/env.sh"
fi

if ! kubectl get ns default >/dev/null 2>&1; then
  die "Can't reach the Kubernetes cluster. Is 'metaflow-dev up' running in another tab?"
fi

port_up() {
  curl -s -o /dev/null --max-time 2 "http://localhost:$1$2"
}
missing_forward=""
port_up 5050 "/" || missing_forward="$missing_forward mlflow(5050)"
port_up 9100 "/" || missing_forward="$missing_forward datalake(9100)"
port_up 8081 "/health" || missing_forward="$missing_forward edge-boston(8081)"
port_up 8092 "/health" || missing_forward="$missing_forward edge-denver(8092)"
if [ -n "$missing_forward" ]; then
  die "Port-forwards not reachable:$missing_forward. Is scripts/port_forwards.sh running in another tab?"
fi

ok "preflight OK: tab configured, cluster reachable, tunnels up"

# ======================================================================
title "STEP 1 · UPLOAD: new device clips arrive" \
      "Two surgical devices upload clips to quarantine; one upload is corrupted in transit."

python scripts/simulate_device_upload.py --video video43 --clips 3 \
  || die "upload of video43 failed"
python scripts/simulate_device_upload.py --video video09 --clips 1 --inject corrupt \
  || die "corrupt upload of video09 failed"

# ======================================================================
title "STEP 2 · INGEST: checks, PHI redaction, rejections" \
      "Every upload is checked, de-identified and promoted, or rejected with a reason."

run_step "$LOGDIR/ingest.log" python flows/ingest_flow.py run --max-workers 1
ingest_rc=$?
[ $ingest_rc -eq 0 ] || die "ingest flow failed (exit $ingest_rc) — see $LOGDIR/ingest.log"

note "the newest rejection record (no patient data in it)"
bash scripts/demo.sh 1b

# ======================================================================
title "STEP 3 · DATASET RELEASE: freeze into an immutable, content-addressed dataset" \
      "Curated clips are split by case, hashed, and locked; the dataset ID is a hash of everything in it."

run_step "$LOGDIR/release.log" python flows/dataset_release_flow.py run --max-workers 2
release_rc=$?
[ $release_rc -eq 0 ] || die "dataset release flow failed (exit $release_rc) — see $LOGDIR/release.log"

dataset_id=$(grep -oE 'Dataset [A-Za-z0-9_.-]+ / ds-[0-9a-f]+' "$LOGDIR/release.log" | tail -1 | awk '{print $NF}')
[ -n "$dataset_id" ] || die "couldn't find a dataset ID in the release output — see $LOGDIR/release.log"
ok ">>> new dataset: $dataset_id <<<"

# ======================================================================
title "STEP 4 · TRAIN: train a model on the frozen dataset" \
      "A Kubernetes pod downloads the verified dataset, trains, and evaluates on the golden test set."

run_step "$LOGDIR/train.log" python flows/train_flow.py run
train_rc=$?
[ $train_rc -eq 0 ] || die "train flow failed (exit $train_rc) — see $LOGDIR/train.log"

mlflow_run_id=$(grep -oE '/runs/[0-9a-f]+' "$LOGDIR/train.log" | tail -1 | cut -d/ -f3)
[ -n "$mlflow_run_id" ] || die "couldn't find the MLflow run ID in the training output — see $LOGDIR/train.log"
ok ">>> MLflow training run: $mlflow_run_id <<<"

# ======================================================================
title "STEP 5 · RELEASE GATE: policy, lineage, ONNX parity, latency" \
      "The model is checked against a written release policy; only a pass registers a candidate version."

run_step "$LOGDIR/gate.log" python flows/release_gate_flow.py run --training-run "$mlflow_run_id"
gate_rc=$?
[ $gate_rc -eq 0 ] || die "release gate flow failed (exit $gate_rc) — see $LOGDIR/gate.log"

decision=$(grep -oE 'Decision: [A-Z]+' "$LOGDIR/gate.log" | tail -1 | awk '{print $2}')
version=$(grep -oE "version [0-9]+ as 'candidate'" "$LOGDIR/gate.log" | tail -1 | awk '{print $2}')

if [ "$decision" != "PASSED" ]; then
  # ==================================================================
  warn "The gate blocked this model; it never reaches a device."
  # ==================================================================
else
  ok ">>> gate PASSED: registered candidate version $version <<<"

  # ==================================================================
  title "STEP 6 · APPROVAL: separation of duties" \
        "Only a person who did not train this model may approve it for release."

  note "attempting self-approval as suchethgovindaraju (expected: REFUSED)"
  self_out=$(python scripts/approve_model.py --version "$version" \
              --approver suchethgovindaraju \
              --reason "self-approval attempt (expected to be refused)" 2>&1)
  self_rc=$?
  echo "$self_out"
  if [ $self_rc -ne 0 ]; then
    warn "expected: refused (same person who trained it cannot approve it)"
  else
    err "unexpected: this approval should have been refused — the trainer was not suchethgovindaraju"
  fi

  note "approving as qa.reviewer (a different person)"
  approved=0
  if run_step "$LOGDIR/approve.log" python scripts/approve_model.py --version "$version" \
       --approver qa.reviewer \
       --reason "Passed release gate; approved for Boston canary rollout"; then
    approved=1
    ok ">>> version $version APPROVED by qa.reviewer <<<"
  else
    warn "approval was refused — skipping the canary rollout"
  fi

  if [ "$approved" -eq 1 ]; then
    # ================================================================
    title "STEP 7 · CANARY ROLLOUT: Boston only, pinned version" \
          "Boston is rolled to the new version first; Denver stays on its current version."

    note "pinning edge-boston to version $version in k8s/edge-sites.yaml"
    python3 - "$version" <<'PYEOF'
import re
import sys

version = sys.argv[1]
path = "k8s/edge-sites.yaml"
text = open(path).read()
boston, rest = text.split("name: edge-denver", 1)
boston, n = re.subn(
    r'(\{name: MODEL_VERSION, value: )"[0-9]+"(\})',
    rf'\1"{version}"\2',
    boston, count=1,
)
if n != 1:
    sys.exit("could not find edge-boston's MODEL_VERSION to change")
open(path, "w").write(boston + "name: edge-denver" + rest)
PYEOF
    [ $? -eq 0 ] || die "failed to edit k8s/edge-sites.yaml"
    grep -A6 'name: edge-boston' k8s/edge-sites.yaml | grep MODEL_VERSION

    note "kubectl apply -f k8s/edge-sites.yaml"
    kubectl apply -f k8s/edge-sites.yaml || die "kubectl apply failed"

    note "waiting for the Boston rollout"
    kubectl rollout status deployment/edge-boston --timeout=120s \
      || die "Boston rollout did not become ready"

    note "restarting only the Boston port-forward (8081)"
    boston_pf_pid=$(lsof -ti tcp:8081 -sTCP:LISTEN 2>/dev/null || true)
    if [ -n "$boston_pf_pid" ]; then
      kill "$boston_pf_pid" 2>/dev/null || true
      sleep 1
    fi
    nohup kubectl port-forward svc/edge-boston 8081:8080 >"$LOGDIR/pf-boston.log" 2>&1 &
    disown
    for _ in $(seq 1 30); do
      port_up 8081 "/health" && break
      sleep 1
    done
    port_up 8081 "/health" || warn "Boston port-forward did not come back up — check $LOGDIR/pf-boston.log"

    note "Boston (new version) vs. Denver (unchanged)"
    python scripts/edge_predict.py --site boston
    python scripts/edge_predict.py --site denver
  fi
fi

# ======================================================================
title "STEP 9 · AUDIT: trace the Boston device back to device uploads" \
      "One command verifies every link in the chain, from the running model back to the original clips."

python scripts/audit.py --site boston
audit_rc=$?
if [ $audit_rc -eq 0 ]; then
  ok "CHAIN INTACT"
else
  warn "audit reported broken link(s) or warnings — see the report above"
fi

echo
ok "Demo complete."
