#!/bin/bash
# Jev-LFM 원격 학습: RTX 3090(24GB) 기준
# 로컬 train_jev_lfm.py가 base 모델을 ~/jev-lfm/model/base 대신 --base 인자로 받도록 패치된 버전을 씀.
set -e
exec > /root/jev_setup.log 2>&1

echo "=== [1/5] python/uv 세팅 $(date)"
apt-get update -qq
apt-get install -y -qq python3.10 python3.10-venv python3.10-dev > /dev/null
python3.10 -m venv /root/jevvenv
source /root/jevvenv/bin/activate
pip install -q --upgrade pip

echo "=== [2/5] torch cu126 + 학습스택 $(date)"
pip install -q torch --index-url https://download.pytorch.org/whl/cu126
pip install -q transformers peft pandas pyarrow safetensors accelerate
python -c "import torch; assert torch.cuda.is_available(); print('CUDA OK:', torch.cuda.get_device_name(0))" || { echo "SETUP-FAIL: no cuda"; exit 1; }

echo "=== [3/5] base model 다운로드 $(date)"
hf download LiquidAI/LFM2.5-1.2B-Instruct --local-dir /root/jev-lfm/model/base 2>/dev/null || \
  pip install -q "huggingface_hub[cli]" && hf download LiquidAI/LFM2.5-1.2B-Instruct --local-dir /root/jev-lfm/model/base
ls -la /root/jev-lfm/model/base/model.safetensors

echo "=== [4/5] 데이터 다운로드 $(date)"
mkdir -p /root/jev-lfm/data/open-jev-v1.1/data
hf download ZefanCai/Open-Jev-v1.1 --repo-type dataset \
  --include 'data/community-hard-mix-v2-redistributable/*' \
  --local-dir /root/jev-lfm/data/open-jev-v1.1
ls -la /root/jev-lfm/data/open-jev-v1.1/data/community-hard-mix-v2-redistributable/ | head

echo "=== [5/5] 파인튜닝 시작 $(date)"
export DATA_DIR=/root/jev-lfm/data/open-jev-v1.1/data/community-hard-mix-v2-redistributable
export BASE_DIR=/root/jev-lfm/model/base
python /root/work/train_jev_lfm.py --base "$BASE_DIR" --data-dir "$DATA_DIR" \
  --steps 300 --train-rows 2000 --calibration-rows 128 --eval-rows 192 \
  --max-length 1536 --lora-rank 8 --accumulation 8 --batch-size 16 \
  --output /root/jev-lfm/runs/pilot1 2>&1 | tee /root/jev_train.log
echo "TRAIN-DONE rc=$?"
tail -5 /root/jev_train.log
