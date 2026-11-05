#!/bin/bash
# Nightly machine-label run on gpu-ubun (owner 2026-11-05: 00:05, facility time).
# Labels stay inactive: the worker never activates. Extra arguments go to `run`, so
#   scripts/nightly_run.sh --day 2026-11-02 --dry-run
# tests the same path the nightly job uses. On failure it appends a line to $ROOT/FAILED.
set -u

CODE="$(cd "$(dirname "$0")/.." && pwd)"
VENV="${VBB_RUN_VENV:-/home/dosomi/dev/ReID/.venv}"
ROOT="${VBB_RUN_ROOT:-/vmdata/vbb_eval/vbb_run}"
MAX_TIME="${VBB_RUN_MAX_TIME:-3h}"  # a hung run must not hold the lock for every later night
FACILITY="${VBB_RUN_FACILITY:-3}"
BACKEND="${VBB_RUN_BACKEND:-https://vbb.yuexingren.cn}"
TOKEN_FILE="${VBB_RUN_TOKEN_FILE:-$HOME/.config/vbb_reid_server/worker.token}"
POINT="${VBB_RUN_POINT:-/vmdata/vbb_eval/out/v015/eval_v4_refs50/point_3_temporal.json}"
MODELS="${VBB_RUN_MODELS:-/vmdata/vbb_eval/models/buffalo_l_face.json}"

# Keep the token, the labels and customers' crops off any proxy the login shell configures.
BYPASS="vbb.yuexingren.cn,upvbbv.yuexingren.cn"
export no_proxy="$BYPASS${no_proxy:+,$no_proxy}"
export NO_PROXY="$BYPASS${NO_PROXY:+,$NO_PROXY}"
export HF_HOME="${HF_HOME:-/vmdata/vbb_eval/hf}"

# $ROOT is unusable here, so the failure is recorded in the home folder instead.
if ! mkdir -p "$ROOT/logs" "$ROOT/emb" "$ROOT/cdn" "$ROOT/runs"; then
  echo "$(date -Iseconds) cannot create $ROOT" >> "$HOME/vbb_run_FAILED"
  exit 73
fi

exec 9>"$ROOT/nightly.lock"
if ! flock -n 9; then
  echo "$(date -Iseconds) another run holds the lock" >> "$ROOT/FAILED"
  exit 75
fi

LOG="$ROOT/logs/$(date +%Y%m%d-%H%M%S).log"
cd "$CODE" || exit 1
timeout --kill-after=5m "$MAX_TIME" "$VENV/bin/vbb-reid-server" run \
  --backend "$BACKEND" --token-file "$TOKEN_FILE" --facility "$FACILITY" --day auto \
  --point "$POINT" --models "$MODELS" \
  --emb "$ROOT/emb" --cdn-cache "$ROOT/cdn" --out "$ROOT/runs" "$@" >> "$LOG" 2>&1
rc=$?
if [ "$rc" -ne 0 ]; then
  echo "$(date -Iseconds) rc=$rc log=$LOG" >> "$ROOT/FAILED"
fi
find "$ROOT/logs" -name '*.log' -mtime +30 -delete
exit "$rc"
