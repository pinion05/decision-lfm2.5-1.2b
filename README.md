# Jev-LFM: LFM2.5-1.2B 기반 비생성형 decision model

**LFM2.5-1.2B에서 autoregressive 생성 루프(LM head)를 뜯어내고, 후보 스코어링용 scalar decision head를 붙여 Open-Jev 방식으로 파인튜닝한 모델.** 생성 없이 한 번의 forward로 typed 확률(Choice/Noul/Score)을 뱉는다.

- 원 레시피: github.com/Zefan-Cai/Open-Jev (model.py/train.py/api.py 포팅)
- 데이터: ZefanCai/Open-Jev-v1.1 `community-hard-mix-v2-redistributable` (train 147K)
- 학습: 2026-09-29 Vast.ai RTX 3090 ($0.18/hr, ~40분, 총비용 ~$0.40)

## "gen 재귀루프를 뜯는다"의 정확한 의미

1. `AutoModelForCausalLM` 로드 → `full.model` (transformer trunk)만 남기고 lm_head 폐기
2. decision head `Linear(hidden,1)` 을 **`lm_head["Yes"] - lm_head["No"]` 행으로 초기화** (사전학습된 Yes/No readout 재활용 — LFM2.5는 weight tying이라 input embedding 행과 동일)
3. 후보별 독립 프롬프트(`Context/Question/Proposed answer/Is this correct?`)를 각각 만들어 trunk에 통과 → 각 후보의 마지막 hidden state → scalar score
4. 레코드 내 후보 score들에 softmax → 후보 확률. 토큰 생성은 어디에도 없음

## 학습 설정 (pilot1)

- LoRA r=8, alpha=16 (in_proj/out_proj/q/k/v/o_proj), backbone frozen
- loss = NLL + 0.1 × Brier, AdamW(backbone lr 5e-5, head lr 1e-4, wd 0.01, clip 1.0)
- 300 steps × accum 8 = 2,400 rows 소비 (train 2K 중), max_length 1536
- calibration split으로 temperature 적합: **T=1.541**

## 결과 (test 192 / ood 192, balanced 샘플)

| 지표 | baseline | trained | calibrated(T) |
|---|---|---|---|
| test brier | 0.685 | **0.557** | **0.554** |
| test acc | 45.3% | **54.7%** | 54.7% |
| ood brier | - | 0.513 | 0.519 |
| ood acc | - | **58.3%** | 58.3% |

zero-shot Yes/No readout 대비 brier -18.7%p, acc +9.4%p. 300 step(2.4K rows)은 전체 train 147K의 1.6% — 이론헤드는 아직 큼.

## 재현/사용

```bash
# 학습 (CUDA GPU 필요, ~40분@3090)
python scripts/train_jev_lfm.py --base LiquidAI/LFM2.5-1.2B-Instruct \
  --data-dir <open-jev-v1.1 parquet dir> --steps 300 --output runs/pilot1

# 서빙 (stdlib-only HTTP, jev 서버 계약과 동일 wire format)
python scripts/serve_jev_lfm.py 8321
curl -s localhost:8321/decide -d '{"state": {...}, "questions": {"q1": {"type": "choice", "instructions": "...", "criteria": {"a": null, "b": null}}}}'
```

- 체크포인트: `runs/pilot1/checkpoint/` (LoRA adapter + head.pt + temperature.json)
- 원격 학습: `scripts/vast_setup2.sh` (pypi torch 경유 — pytorch.org egress 저속 회피)
- 주의: 로컬 CPU 학습은 forward 1회에 ~9분이라 비현실적. 780M ROCm torch는 matmul에서 GPU hang (gfx1103 커널 문제, 재시도 금지)
