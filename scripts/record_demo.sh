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
# real cluster. After each step it shows the data that step produced (via
# scripts/show_data.py) and opens preview images in Preview, so record the
# whole screen, not just the terminal. PAUSE=<seconds> sets the reading time
# after each view (default 5).
#
# After each step it also opens the same data in Google Chrome: data lake
# console folders, the uploaded and curated files themselves, the Metaflow
# runs and the MLflow pages, then brings the terminal back. BROWSE_PAUSE sets
# seconds per page (default 7); BROWSER_VIEWS=0 turns this off. Nothing here
# is simulated. It does not commit, push, or touch
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

# Gives the viewer time to read what just printed. PAUSE=0 to skip.
pause() { sleep "${PAUSE:-5}"; }

# The app this script runs in, so browse() can bring it back to the front.
case "${TERM_PROGRAM:-}" in
  iTerm.app) TERM_APP="iTerm" ;;
  vscode)    TERM_APP="Visual Studio Code" ;;
  *)         TERM_APP="Terminal" ;;
esac

# browse <stage> [args]  — opens each page from `show_data.py links <stage>`
# in Chrome for BROWSE_PAUSE seconds, then returns to the terminal.
browse() {
  [ "${BROWSER_VIEWS:-1}" = 1 ] && [ "$(uname)" = "Darwin" ] || return 0
  local links="$LOGDIR/links-$1.txt"
  if ! python scripts/show_data.py links "$@" >"$links"; then
    warn "couldn't build browser links for $1; skipping"
    return 0
  fi
  while IFS='|' read -r label url; do
    note "browser: $label"
    open -a "Google Chrome" "$url"
    sleep "${BROWSE_PAUSE:-7}"
  done <"$links"
  osascript -e "tell application \"$TERM_APP\" to activate" >/dev/null 2>&1
}

note() { echo "${CYAN}-- $* --${RESET}"; }
ok()   { echo "${GREEN}$*${RESET}"; }
warn() { echo "${YELLOW}$*${RESET}"; }
err()  { echo "${RED}$*${RESET}"; }

die() { err "$*"; exit 1; }

# Strips ANSI colors, Metaflow's "[run/step/task (pid N)] " line prefixes and
# timestamp prefixes, and drops repetitive Metaflow boilerplate lines so the
# real flow output (prints, metrics, results) reads cleanly on camera.
# Every stage flushes per line (sed -l, grep --line-buffered) so long steps
# like training show progress live instead of all at once at the end.
clean_metaflow() {
  sed -l -E 's/\x1b\[[0-9;]*m//g' \
    | sed -l -E 's/^[0-9]{4}-[0-9]{2}-[0-9]{2}[ T][0-9]{2}:[0-9]{2}:[0-9]{2}(\.[0-9]+)?[[:space:]]+//' \
    | sed -l -E 's/^\[[^]]*\][[:space:]]*//' \
    | grep --line-buffered -Ev '^(Metaflow [0-9]|Validating your flow|The graph looks good|Workflow starting|Task is starting|Task finished successfully|Bootstrapping|Logging into|Pulling container image|Returning the task)'
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

note "the test data: real CholecSeg8k surgical frames and their segmentation masks"
python scripts/show_data.py source --video video43 --clips 3
pause
browse source --match video43

note "device 1 uploads 3 clips of video43 (clean)"
python scripts/simulate_device_upload.py --video video43 --clips 3 \
  || die "upload of video43 failed"
note "device 2 uploads 1 clip of video09, with one frame truncated in transit"
python scripts/simulate_device_upload.py --video video09 --clips 1 --inject corrupt \
  || die "corrupt upload of video09 failed"
pause

note "what landed in quarantine: frames, masks, and a manifest per upload"
python scripts/show_data.py quarantine --upload video43
pause
browse upload --match video43

# ======================================================================
title "STEP 2 · INGEST: checks, PHI redaction, rejections" \
      "Every upload is checked, de-identified and promoted, or rejected with a reason."

run_step "$LOGDIR/ingest.log" python flows/ingest_flow.py run --max-workers 1
ingest_rc=$?
[ $ingest_rc -eq 0 ] || die "ingest flow failed (exit $ingest_rc) — see $LOGDIR/ingest.log"

pause

note "rejected uploads and why (no patient data in the record)"
python scripts/show_data.py rejected
pause

note "accepted uploads, now de-identified in the curated bucket"
python scripts/show_data.py curated
pause

note "the same frame before and after redaction"
python scripts/show_data.py redaction
pause
browse ingest

# ======================================================================
title "STEP 3 · DATASET RELEASE: freeze into an immutable, content-addressed dataset" \
      "Curated clips are split by case, hashed, and locked; the dataset ID is a hash of everything in it."

run_step "$LOGDIR/release.log" python flows/dataset_release_flow.py run --max-workers 2
release_rc=$?
[ $release_rc -eq 0 ] || die "dataset release flow failed (exit $release_rc) — see $LOGDIR/release.log"

dataset_id=$(grep -oE 'Dataset [A-Za-z0-9_.-]+ / ds-[0-9a-f]+' "$LOGDIR/release.log" | tail -1 | awk '{print $NF}')
[ -n "$dataset_id" ] || die "couldn't find a dataset ID in the release output — see $LOGDIR/release.log"
ok ">>> new dataset: $dataset_id <<<"
pause

note "what was frozen: the dataset card, the manifest, and a locked file"
python scripts/show_data.py dataset
pause
browse dataset

# ======================================================================
title "STEP 4 · TRAIN: train a model on the frozen dataset" \
      "A Kubernetes pod downloads the verified dataset, trains, and evaluates on the golden test set."

run_step "$LOGDIR/train.log" python flows/train_flow.py run
train_rc=$?
[ $train_rc -eq 0 ] || die "train flow failed (exit $train_rc) — see $LOGDIR/train.log"

mlflow_run_id=$(grep -oE '/runs/[0-9a-f]+' "$LOGDIR/train.log" | tail -1 | cut -d/ -f3)
[ -n "$mlflow_run_id" ] || die "couldn't find the MLflow run ID in the training output — see $LOGDIR/train.log"
ok ">>> MLflow training run: $mlflow_run_id <<<"
pause
browse train --run-id "$mlflow_run_id"

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
  pause
  browse gate --version "$version"

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
    pause
    browse approve
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
    pause
    browse canary
  fi
fi

# ======================================================================
title "STEP 8 · AUDIT: trace the Boston device back to device uploads" \
      "One command verifies every link in the chain, from the running model back to the original clips."

python scripts/audit.py --site boston
audit_rc=$?
if [ $audit_rc -eq 0 ]; then
  ok "CHAIN INTACT"
else
  warn "audit reported broken link(s) or warnings — see the report above"
fi
pause
browse audit

echo
ok "Demo complete."
