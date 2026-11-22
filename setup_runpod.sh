#!/usr/bin/env bash
# One-time setup on a rented GPU (tested layout: RunPod "PyTorch 2.x" template, RTX 4090 24 GB or L40S 48 GB).
# Usage:  bash setup_runpod.sh            (then put PIE-Bench at datasets/PIE-Bench_v1, see README)
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p ext checkpoints datasets

# 1) The lab's code: ChordEdit reproduction (editing) and the PnPInversion fork (PIE-Bench evaluator)
[ -d ext/ChordEdit-Reproduction ] || git clone https://github.com/Harvard-AI-and-Robotics-Lab/ChordEdit-Reproduction.git ext/ChordEdit-Reproduction
[ -d ext/PnPInversion ] || git clone https://github.com/Harvard-AI-and-Robotics-Lab/PnPInversion.git ext/PnPInversion

# 2) Python deps. Keep the pod's preinstalled CUDA torch; install the rest.
python -m pip install --upgrade pip
grep -v -E '^(torch|torchaudio|torchvision)' ext/ChordEdit-Reproduction/requirement.txt > /tmp/lab_requirements.txt
python -m pip install -r /tmp/lab_requirements.txt
python -m pip install -r requirements.txt

# 3) SD-Turbo weights (accept the license on Hugging Face first if prompted; then `huggingface-cli login`)
if [ ! -d checkpoints/sd-turbo/unet ]; then
  huggingface-cli download stabilityai/sd-turbo --local-dir checkpoints/sd-turbo \
    --include "unet/*" "vae/*" "text_encoder/*" "tokenizer/*" "scheduler/*" "model_index.json"
fi

# 4) PIE-Bench must be requested via the form linked in ext/PnPInversion/README.md
if [ ! -f datasets/PIE-Bench_v1/mapping_file.json ]; then
  echo "!! PIE-Bench not found. Download it (form link in ext/PnPInversion/README.md) and unzip to datasets/PIE-Bench_v1/"
  echo "   expected: datasets/PIE-Bench_v1/annotation_images/ and datasets/PIE-Bench_v1/mapping_file.json"
fi

python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
echo "Setup done. Next: bash run_all.sh --max-samples 70   (a ~10-minute check), then: bash run_all.sh --quick"
