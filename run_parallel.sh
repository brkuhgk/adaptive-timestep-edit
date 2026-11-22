#!/usr/bin/env bash
# Same study as run_all.sh, split across every visible GPU (one process per GPU). Resumable.
#   bash run_parallel.sh --quick        # 1 seed;  ~2 h on 4x RTX 4090
#   bash run_parallel.sh                # all seeds
#   bash run_parallel.sh --fake         # CPU smoke test (pretends 2 GPUs)
set -euo pipefail
cd "$(dirname "$0")"
export HF_HUB_DISABLE_XET=1
ARGS=("$@")
PY=${PYTHON:-python}
FAKE=0; for a in "${ARGS[@]}"; do [[ "$a" == "--fake" ]] && FAKE=1; done
if [[ $FAKE == 1 ]]; then NGPU=1; else NGPU=$(nvidia-smi -L | wc -l); fi
W=${WORKERS_PER_GPU:-1}            # >1 overlaps work on one GPU (single-image editing leaves it mostly idle)
[[ $FAKE == 1 ]] && W=2
N=$((NGPU * W))
# Pods report the HOST's core count (e.g. 96) but get only a few vCPUs; without a cap every worker starts ~96
# threads and they fight. Measured on RTX 4090 / 8 vCPU: scoring 3.1 -> 7.2 rows/s with 4 threads per worker.
CPUS=$(python -c "import os; print(len(os.sched_getaffinity(0)))" 2>/dev/null || nproc)
[[ -r /sys/fs/cgroup/cpu.max ]] && read -r Q P < /sys/fs/cgroup/cpu.max && [[ "$Q" != "max" ]] && CPUS=$(( Q / P ))
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-$(( CPUS / N > 0 ? CPUS / N : 1 ))}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-$OMP_NUM_THREADS}
WORK=work; [[ $FAKE == 1 ]] && WORK=work_fake
mkdir -p "$WORK/logs"
echo "Using $NGPU GPU(s) x $W worker(s) = $N shards, $OMP_NUM_THREADS CPU threads each. Logs: $WORK/logs/"

"$PY" scripts/01_prepare.py "${ARGS[@]}"

if [[ $FAKE == 0 ]]; then   # download eval models once so parallel workers don't race on the cache
  "$PY" - <<'PY'
import sys; sys.path.insert(0, "ext/PnPInversion")
from evaluation.matrics_calculator import MetricsCalculator
MetricsCalculator("cuda"); print("evaluation models cached")
PY
fi

run_sharded() {  # $1 = label, $2 = number of shards, rest = scripts to run in order on each shard
  local label=$1; local n=$2; shift 2
  local pids=()
  for ((i=0; i<n; i++)); do
    local gpu=$((i % NGPU))
    ( for s in "$@"; do CUDA_VISIBLE_DEVICES=$gpu "$PY" "$s" --shard "$i/$n" "${ARGS[@]}" || exit 1; done ) \
      > "$WORK/logs/${label}_gpu$i.log" 2>&1 &
    pids+=($!)
  done
  local fail=0
  for p in "${pids[@]}"; do wait "$p" || fail=1; done
  if [[ $fail == 1 ]]; then echo "!! a $label worker failed; see $WORK/logs/${label}_gpu*.log (re-run to resume)"; exit 1; fi
  echo "$label done"
}

run_sharded grid_eval "$N" scripts/02_grid.py scripts/03_evaluate.py
run_sharded probe "$NGPU" scripts/04_probe.py      # one per GPU: the probe batch is memory-heavy
"$PY" scripts/05_classify.py "${ARGS[@]}"
"$PY" scripts/06_analyze.py "${ARGS[@]}"
[ -f scripts/07_fill_report.py ] && "$PY" scripts/07_fill_report.py "${ARGS[@]}" || true
echo "Done. Results in $WORK/results/ (main_table.md, summary.json, fig_*.png); drafts in report/*_filled.*"
