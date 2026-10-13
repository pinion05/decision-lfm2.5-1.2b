"""Head-to-head: Jev-LFM(1.2B, 학습됨) vs Tiny-Jev-0.6B vs Tiny-Jev-1.7B (zero-shot)
동일 데이터: Open-Jev-v1.1 test split에서 Jev-LFM pilot1이 쓴 것과 동일한 192 rows.
Tiny-Jev는 공개 API(model.choice/noul/score)를 그대로 사용.
CPU에서 실행 (전부 소형 모델).
"""
import json, sys, time
sys.path.insert(0, "/home/pinion/jev-lfm/scripts")
from train_jev_lfm import load_split, evaluate_probabilities
import torch

ROWS = load_split("test", 192, 42)  # pilot1 eval_rows와 동일 seed/limit


def eval_jevlfm():
    from train_jev_lfm import LfmDecisionModel, softmax
    model = LfmDecisionModel.load("/home/pinion/jev-lfm/runs/pilot1/checkpoint",
                                  device="cpu", max_length=1536)
    temp = 1.541
    t0 = time.perf_counter()
    probs = []
    model.eval()
    with torch.inference_mode():
        for i in range(0, len(ROWS), 8):
            for lg in model(ROWS[i:i+8]):
                probs.append(softmax([float(x)/temp for x in lg]))
    dt = time.perf_counter() - t0
    return probs, dt


def eval_tiny_jev(model_id: str):
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id, trust_remote_code=True).eval()
    t0 = time.perf_counter()
    probs = []
    for row in ROWS:
        if row["kind"] == "noul":
            p_yes = float(model.noul(tok, row["state"], row["question"]))
            probs.append([1 - p_yes, p_yes])
        elif row["kind"] == "choice":
            named = {str(o): o for o in row["options"]}
            out = model.choice(tok, row["state"], row["question"], named)
            probs.append([float(out["probabilities"][str(o)]) for o in row["options"]])
        else:  # score: options are descriptive levels
            out = model.score(tok, row["state"], row["question"], list(row["options"]))
            probs.append([float(out["probabilities"][str(i)]) for i in range(len(row["options"]))])
    dt = time.perf_counter() - t0
    return probs, dt


results = {}
for name, fn in [("jev-lfm-1.2b", eval_jevlfm)]:
    probs, dt = fn()
    m = evaluate_probabilities([r["target"] for r in ROWS], probs)
    m["latency_s"] = round(dt, 1)
    results[name] = m
    print(name, json.dumps(m, default=float))

for mid in ["lostargon/Tiny-Jev", "lostargon/Tiny-Jev-1.7B"]:
    try:
        probs, dt = eval_tiny_jev(mid)
        m = evaluate_probabilities([r["target"] for r in ROWS], probs)
        m["latency_s"] = round(dt, 1)
        results[mid] = m
        print(mid, json.dumps(m, default=float))
    except Exception as e:
        print(mid, "FAILED:", repr(e)[:200])

json.dump(results, open("/tmp/jev_compare.json", "w"), indent=1, default=float)
print("SAVED /tmp/jev_compare.json")
