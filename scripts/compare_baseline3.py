"""Head-to-head v2: 결과를 즉시 파일에 기록(부분결과 보존). 96 rows로 축소.
순서: Tiny-Jev 0.6B -> jev-lfm -> (메모리 여유시) Tiny-Jev 1.7B
"""
import json, sys, time, gc
sys.path.insert(0, "/home/pinion/jev-lfm/scripts")
from train_jev_lfm import load_split, evaluate_probabilities
import torch

OUT = "/tmp/jev_compare.json"
ROWS = load_split("test", 96, 42)
results = {}

def save():
    json.dump(results, open(OUT, "w"), indent=1, default=float)

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
    m = evaluate_probabilities([r["target"] for r in ROWS], probs)
    m["latency_s"] = round(dt, 1)
    return m

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
            # Tiny-Jev keys probabilities by the exact option text
            probs.append([float(out["probabilities"][o]) for o in row["options"]])
        else:
            out = model.score(tok, row["state"], row["question"], list(row["options"]))
            # Tiny-Jev keys probabilities by the level text, not index
            probs.append([float(out["probabilities"][o]) for o in row["options"]])
    dt = time.perf_counter() - t0
    m = evaluate_probabilities([r["target"] for r in ROWS], probs)
    m["latency_s"] = round(dt, 1)
    del model, tok
    gc.collect()
    return m

for name, fn in [("jev-lfm-1.2b-pilot1", eval_jevlfm),
                 ("lostargon/Tiny-Jev-1.7B", lambda: eval_tiny_jev("lostargon/Tiny-Jev-1.7B"))]:
    try:
        m = fn()
        results[name] = m
        save()
        print("DONE", name, json.dumps(m, default=float), flush=True)
    except Exception as e:
        results[name] = {"error": repr(e)[:200]}
        save()
        print("FAIL", name, repr(e)[:200], flush=True)
print("ALL_DONE", flush=True)
