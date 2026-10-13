#!/bin/bash
# Jev-LFM scale-up: train 147K 전체 소진, 2000 optimizer steps (16K rows/pass 이상)
set -e
exec > /root/jev_setup.log 2>&1
echo "=== [1/5] base $(date)"
apt-get update -qq
apt-get install -y -qq python3.10-venv python3-pip > /dev/null
python3.10 -m venv /root/jevvenv
source /root/jevvenv/bin/activate
pip install -q --upgrade pip

echo "=== [2/5] torch(PyPI) + stack $(date)"
pip install -q torch --index-url https://pypi.org/simple
pip install -q transformers peft pandas pyarrow safetensors accelerate "huggingface_hub[cli]"
python - <<'EOF' || { echo "SETUP-FAIL: cuda"; exit 1; }
import torch
assert torch.cuda.is_available(), "no cuda"
print("CUDA OK:", torch.cuda.get_device_name(0))
EOF

echo "=== [3/5] base model $(date)"
hf download LiquidAI/LFM2.5-1.2B-Instruct --local-dir /root/jev-lfm/model/base
ls -la /root/jev-lfm/model/base/model.safetensors

echo "=== [4/5] dataset $(date)"
hf download ZefanCai/Open-Jev-v1.1 --repo-type dataset \
  --include 'data/community-hard-mix-v2-redistributable/*' \
  --local-dir /root/jev-lfm/data/open-jev-v1.1
ls /root/jev-lfm/data/open-jev-v1.1/data/community-hard-mix-v2-redistributable/ | head -6

echo "=== [5/5] scale-up train $(date)"
export DATA_DIR=/root/jev-lfm/data/open-jev-v1.1/data/community-hard-mix-v2-redistributable
export BASE_DIR=/root/jev-lfm/model/base
python /root/work/train_jev_lfm.py --base "$BASE_DIR" --data-dir "$DATA_DIR" \
  --steps 6000 --train-rows 60000 --calibration-rows 256 --eval-rows 256 \
  --max-length 1536 --lora-rank 8 --accumulation 8 --batch-size 16 \
  --checkpoint-every 500 \
  --output /root/jev-lfm/runs/scale1 > /root/jev_train.log 2>&1
RC=$?
echo "TRAIN-DONE rc=$RC"
tail -3 /root/jev_train.log
