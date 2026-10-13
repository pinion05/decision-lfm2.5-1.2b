"""Smoke test: model load, Yes/No head init, forward on 2 real records, backward."""
import json, sys
sys.path.insert(0, "/home/pinion/jev-lfm/scripts")
from train_jev_lfm import LfmDecisionModel, load_split, BASE_DIR
import torch, time

torch.manual_seed(0)
model = LfmDecisionModel(str(BASE_DIR), lora_rank=8, max_length=1536, device="cpu")
trainable = [(n, p.shape) for n, p in model.named_parameters() if p.requires_grad]
print("trainable tensors:", len(trainable))
print("first 3:", [n for n, _ in trainable[:3]])
print("head w[:4]:", model.head.weight[0, :4].tolist(), "b:", model.head.bias.tolist())

rows = load_split("train", 2, 42)[:2]
print("kinds:", [r["kind"] for r in rows], "n_options:", [len(r["options"]) for r in rows])
model.eval()
t0 = time.perf_counter()
with torch.inference_mode():
    logits = model(rows)
print("forward ok in %.1fs" % (time.perf_counter() - t0))
for r, lg in zip(rows, logits):
    print(r["kind"], "logits:", [round(float(x), 3) for x in lg])

# backward smoke
model.train()
lg = model([rows[0]])[0]
target = torch.tensor(rows[0]["target"])
loss = -(target * lg.log_softmax(-1)).sum() + 0.1 * ((lg.softmax(-1) - target) ** 2).sum()
loss.backward()
gnorm = sum(p.grad.norm().item() ** 2 for p in model.parameters() if p.requires_grad) ** 0.5
print("loss: %.4f grad_norm: %.4f" % (loss.item(), gnorm))
print("SMOKE_OK")
