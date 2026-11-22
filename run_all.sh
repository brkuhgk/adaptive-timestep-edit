#!/usr/bin/env bash
# Run the whole study. Every stage is resumable: re-running skips finished work.
#   bash run_all.sh --fake            # CPU smoke test (~1 min, no downloads)
#   bash run_all.sh --quick           # 1 seed: grid + eval + probe + classifier + analysis
#   bash run_all.sh                   # all seeds in configs/default.yaml
#   bash run_all.sh --max-samples 70  # tiny real run to check the setup before the full run
set -euo pipefail
cd "$(dirname "$0")"
export HF_HUB_DISABLE_XET=1
ARGS=("$@")
PY=${PYTHON:-python}
stage() { echo; echo "=== $1 ==="; shift; "$PY" "$@" "${ARGS[@]}"; }
stage "1/7 prepare (split + eval sources)" scripts/01_prepare.py
stage "2/7 grid of edits"                  scripts/02_grid.py
stage "3/7 PIE-Bench metrics"              scripts/03_evaluate.py
stage "4/7 velocity probe"                 scripts/04_probe.py
stage "5/7 prompt classifier"              scripts/05_classify.py
stage "6/7 analysis + figures"             scripts/06_analyze.py
stage "7/7 fill report + email drafts"     scripts/07_fill_report.py
WORK=work; for a in "${ARGS[@]}"; do [[ "$a" == "--fake" ]] && WORK=work_fake; done
echo; echo "Done. Results in $WORK/results/ (main_table.md, summary.json, fig_*.png); drafts in report/*_filled.*"
