#!/usr/bin/env bash
# One-shot on a fresh RunPod pod:  bash pod_bootstrap.sh "<PIE-Bench Google Drive folder link from your access email>"
# Installs everything, fetches PIE-Bench into datasets/PIE-Bench_v1, runs a 70-image check, then the 1-seed study.
# The PIE-Bench link is passed at run time on purpose: the dataset authors ask that it not be shared.
set -euo pipefail
cd "$(dirname "$0")"
PIE_URL=${1:-}
export PIP_BREAK_SYSTEM_PACKAGES=1
bash setup_runpod.sh

if [ ! -f datasets/PIE-Bench_v1/mapping_file.json ]; then
  [ -n "$PIE_URL" ] || { echo "Pass the PIE-Bench Drive folder link as the first argument"; exit 1; }
  python -m pip install -q gdown
  mkdir -p datasets/pie_download datasets/pie_unpacked
  gdown --folder "$PIE_URL" -O datasets/pie_download --remaining-ok
  find datasets/pie_download -name "*.zip" -print0 | while IFS= read -r -d '' z; do unzip -q -o "$z" -d datasets/pie_unpacked; done
  MAP=$(find datasets/pie_download datasets/pie_unpacked -name mapping_file.json | head -1)
  [ -n "$MAP" ] || { echo "mapping_file.json not found in the download"; find datasets -maxdepth 4 | head -50; exit 1; }
  rm -rf datasets/PIE-Bench_v1 && ln -s "$(cd "$(dirname "$MAP")" && pwd)" datasets/PIE-Bench_v1
fi
ls datasets/PIE-Bench_v1 | head
python -c "import json; m=json.load(open('datasets/PIE-Bench_v1/mapping_file.json')); print('PIE-Bench samples:', len(m))"

echo "=== 70-image check ==="
WORKERS_PER_GPU=${WORKERS_PER_GPU:-3} bash run_parallel.sh --quick --max-samples 70
cp work/results/main_table.md check_main_table_70.md 2>/dev/null || true
echo "=== Full 1-seed study (resumes: the 70 checked images are not redone) ==="
WORKERS_PER_GPU=${WORKERS_PER_GPU:-3} bash run_parallel.sh --quick
tar czf results_$(date +%Y%m%d_%H%M).tgz work/results report/*_filled.* && ls -la results_*.tgz
echo "ALL DONE"
