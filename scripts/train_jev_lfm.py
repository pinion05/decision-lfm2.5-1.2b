"""Jev-LFM: port of Open-Jev's decision model to LiquidAI/LFM2.5-1.2B-Instruct.

"gen 재귀루프를 뜯는다" = strip the autoregressive LM head from the pretrained
backbone, keep the transformer trunk, and train a scalar decision head seeded
from the pretrained Yes/No readout rows. Every candidate is scored by one
forward pass of its own prompt; candidate probabilities come from a softmax
over candidate scores. No autoregressive generation anywhere.

Faithful to github.com/Zefan-Cai/Open-Jev (jev/model.py, jev/train.py):
- head init  = lm_head["Yes"] - lm_head["No"]   (LFM ties embeddings, so the
  readout equals the input embedding rows for "Yes"/"No")
- LoRA r=8, alpha=16, dropout 0, on q/k/v/o-proj (+lfm2 in_proj/out_proj if present)
- loss = NLL + brier_weight * sum((softmax - target)^2)
- AdamW(betas .9/.999, eps 1e-8, wd .01), grad clip 1.0
- backbone frozen except LoRA; head lr 1e-4, backbone lr 5e-5
CPU-friendly: batch>1 supported, bf32 autocast optional, threads = nproc.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import pandas as pd
import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

RECORD_FIELDS = ("id", "group_id", "split", "source", "kind", "question", "options", "target")
DATA_DIR = Path(os.environ.get(
    "DATA_DIR",
    Path.home() / "jev-lfm/data/open-jev-v1.1/data/community-hard-mix-v2-redistributable"))
BASE_DIR = Path(os.environ.get("BASE_DIR", Path.home() / "jev-lfm/model/base"))


# ---------------------------------------------------------------- data ----

def load_split(split: str, limit: int | None, seed: int, balanced: bool = True) -> list[dict]:
    df = pd.read_parquet(DATA_DIR / f"{split}-00000-of-00001.parquet")
    rows = []
    for _, r in df.iterrows():
        rec = {
            "id": r["id"], "group_id": r["group_id"], "split": r["split"],
            "source": r["source"], "kind": r["kind"],
            "question": r["question"],
            "options": list(r["options"]),
            "target": [float(x) for x in r["target"]],
            "state": json.loads(r["state_json"]),
        }
        rows.append(rec)
    rng = random.Random(seed)
    rng.shuffle(rows)
    if balanced:
        buckets: dict[tuple, list] = {}
        for row in rows:
            buckets.setdefault((row["source"], row["kind"]), []).append(row)
        rows = []
        while any(buckets.values()):
            for bucket in buckets.values():
                if bucket:
                    rows.append(bucket.pop())
    rows = rows[:limit] if limit else rows
    for row in rows:
        assert row["kind"] in ("choice", "noul", "score")
        if row["kind"] == "noul":
            assert row["options"] == ["no", "yes"] and len(row["target"]) == 2
        else:
            assert len(row["target"]) == len(row["options"])
    return rows


def candidate_prompts(record: dict) -> list[str]:
    """Byte-faithful port of jev/api.py::candidate_prompts."""
    state = record["state"] if isinstance(record["state"], str) else json.dumps(
        record["state"], ensure_ascii=False, sort_keys=True, allow_nan=False)
    prefix = f"Context:\n{state}\n\nQuestion: {record['question']}\n"
    if record["kind"] == "noul":
        return [prefix + "Is the answer to this question yes? Answer Yes or No."]
    return [prefix + f"Proposed answer: {o}\nIs this proposed answer correct? Answer Yes or No."
            for o in record["options"]]


# ---------------------------------------------------------------- model ----

class LfmDecisionModel(nn.Module):
    def __init__(self, model_path: str, lora_rank: int = 8, max_length: int = 384,
                 device: str = "cpu", dtype: torch.dtype | None = None):
        super().__init__()
        self.max_length = max_length
        self.device_name = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.tokenizer.padding_side = "right"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        full = AutoModelForCausalLM.from_pretrained(
            model_path, dtype=dtype or (torch.bfloat16 if device.startswith("cuda") else torch.float32),
        )
        # --- "뜯기": Yes/No readout rows before dropping the LM head ---
        yes = self.tokenizer.encode("Yes", add_special_tokens=False)
        no = self.tokenizer.encode("No", add_special_tokens=False)
        assert len(yes) == 1 and len(no) == 1, (yes, no)
        out_w = full.get_output_embeddings().weight
        initial = (out_w[yes[0]] - out_w[no[0]]).detach().float().clone()
        # strip the autoregressive loop: keep only the transformer trunk
        self.backbone = full.model
        del full
        hidden = self.backbone.config.hidden_size
        self.backbone.requires_grad_(False)
        self.head = nn.Linear(hidden, 1, bias=True, dtype=torch.float32)
        with torch.no_grad():
            self.head.weight.copy_(initial.unsqueeze(0))
            self.head.bias.zero_()
        del initial
        self.lora_rank = 0
        if lora_rank:
            from peft import LoraConfig, get_peft_model
            names = {n.rsplit(".", 1)[-1] for n, _ in self.backbone.named_modules()}
            # lfm2 attention/conv block projections (+ generic fallbacks)
            targets = [t for t in ("q_proj", "k_proj", "v_proj", "o_proj",
                                   "in_proj_qkv", "out_proj", "in_proj", "out_proj_conv")
                       if t in names]
            if not targets:
                raise ValueError(f"no LoRA targets in {sorted(names)}")
            self.backbone = get_peft_model(self.backbone, LoraConfig(
                r=lora_rank, lora_alpha=lora_rank * 2, target_modules=targets,
                lora_dropout=0.0, bias="none"))
            self.backbone.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False})
            self.backbone.enable_input_require_grads()
            self.lora_rank = lora_rank
        self.to(device)

    def forward(self, records: list[dict]) -> list[torch.Tensor]:
        prompts, counts = [], []
        for record in records:
            entries = candidate_prompts(record)
            counts.append(len(entries))
            for p in entries:
                prompts.append(self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": p}], tokenize=False,
                    add_generation_prompt=True, enable_thinking=False))
        enc = self.tokenizer(prompts, padding=True, truncation=False, return_tensors="pt")
        lengths = enc["attention_mask"].sum(-1)
        if int(lengths.max()) > self.max_length:
            raise ValueError(f"input {int(lengths.max())} > max_length {self.max_length}")
        enc = {k: v.to(self.device_name) for k, v in enc.items()}
        out = self.backbone(**enc, use_cache=False, return_dict=True)
        h = out.last_hidden_state
        idx = torch.arange(len(prompts), device=h.device)
        hidden = h[idx, lengths.to(h.device) - 1]
        scores = self.head(hidden.float()).squeeze(-1)
        logits, off = [], 0
        for record, count in zip(records, counts):
            values = scores[off:off + count]
            if record["kind"] == "noul":
                values = torch.stack([torch.zeros_like(values[0]), values[0]])
            logits.append(values)
            off += count
        return logits

    def save(self, output: Path):
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        if self.lora_rank:
            self.backbone.save_pretrained(output / "adapter")
        torch.save(self.head.state_dict(), output / "head.pt")
        (output / "model.json").write_text(json.dumps({
            "base": str(BASE_DIR), "max_length": self.max_length,
            "lora_rank": self.lora_rank,
            "method": "independent_candidate_lora_nll_brier (jev port)",
        }, indent=2) + "\n")

    @classmethod
    def load(cls, output: Path, device: str = "cpu", max_length: int | None = None):
        output = Path(output)
        cfg = json.loads((output / "model.json").read_text())
        model = cls(cfg["base"], lora_rank=0,
                    max_length=max_length or cfg["max_length"], device=device)
        if cfg["lora_rank"]:
            from peft import PeftModel
            model.backbone = PeftModel.from_pretrained(model.backbone, output / "adapter")
        model.head.load_state_dict(torch.load(output / "head.pt", map_location=device,
                                              weights_only=True))
        return model.to(device).eval()


# ---------------------------------------------------------------- eval ----

def softmax(xs, temperature: float = 1.0) -> list[float]:
    m = max(xs)
    w = [math.exp((x - m) / temperature) for x in xs]
    s = sum(w)
    return [x / s for x in w]


@torch.inference_mode()
def evaluate(model, rows: list[dict], batch_size: int = 8):
    model.eval()
    results = []
    for i in range(0, len(rows), batch_size):
        chunk = rows[i:i + batch_size]
        logit_lists = model(chunk)
        for row, logits in zip(chunk, logit_lists):
            lg = [float(x) for x in logits]
            results.append({"id": row["id"], "kind": row["kind"], "source": row["source"],
                            "target": row["target"], "logits": lg,
                            "probabilities": softmax(lg)})
    return results


def evaluate_probabilities(targets, probs) -> dict:
    acc_n = acc_d = 0
    brier = nll = 0.0
    for t, p in zip(targets, probs):
        hit = max(range(len(p)), key=p.__getitem__)
        onehot = sum(t) == 1.0
        if onehot:
            acc_d += 1
            acc_n += 1 if t[hit] > 0.5 else 0
        brier += sum((a - b) ** 2 for a, b in zip(p, t))
        nll += -math.log(max(p[hit] if onehot else p[0] if len(p) == 2 else p[hit], 1e-15))
    out = {"n": len(targets), "brier": brier / len(targets)}
    out["accuracy"] = acc_n / acc_d if acc_d else None
    out["nll"] = nll / len(targets)
    return out


def fit_temperature(rows, split_name: str = "calibration") -> float:
    """LBFGS on shared-temperature NLL over calibration rows.

    Rows have different class counts (noul=2 vs choice/score=N), so the loss
    is a ragged sum of per-row softmax NLL under one shared temperature.
    """
    log_t = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], max_iter=50)

    def closure():
        opt.zero_grad()
        total = torch.zeros((), dtype=torch.float64)
        for r in rows:
            t_vec = torch.tensor(r["target"], dtype=torch.float64)
            l_vec = torch.tensor(r["logits"], dtype=torch.float64) / log_t.exp()
            total = total - (t_vec * l_vec.log_softmax(-1)).sum()
        total.backward()
        return total

    opt.step(closure)
    return float(log_t.exp().item())


# ---------------------------------------------------------------- train ----

def run(args):
    global DATA_DIR, BASE_DIR
    DATA_DIR = Path(args.data_dir)
    BASE_DIR = Path(args.base)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if args.threads:
        torch.set_num_threads(args.threads)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else None
    print(json.dumps({"event": "load", "device": device,
                      "threads": torch.get_num_threads()}), flush=True)
    train = load_split("train", args.train_rows, args.seed)
    calibration = load_split("calibration", args.calibration_rows, args.seed)
    test = load_split("test", args.eval_rows, args.seed)
    ood = load_split("ood", args.eval_rows, args.seed)
    print(json.dumps({"event": "data", "train": len(train), "cal": len(calibration),
                      "test": len(test), "ood": len(ood)}), flush=True)

    model = LfmDecisionModel(str(BASE_DIR), lora_rank=args.lora_rank,
                             max_length=args.max_length, device=device, dtype=dtype)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(json.dumps({"event": "model", "trainable_params": trainable}), flush=True)

    model.eval()
    baseline = evaluate(model, test, args.batch_size)
    baseline_ood = evaluate(model, ood, args.batch_size)
    baseline_cal = evaluate(model, calibration, args.batch_size)
    base_metrics = evaluate_probabilities([r["target"] for r in baseline],
                                          [r["probabilities"] for r in baseline])
    base_temperature = fit_temperature(baseline_cal)
    print(json.dumps({"event": "baseline", **base_metrics}), flush=True)

    opt = torch.optim.AdamW([
        {"params": [p for p in model.backbone.parameters() if p.requires_grad], "lr": args.lr},
        {"params": model.head.parameters(), "lr": args.head_lr},
    ], weight_decay=0.01, betas=(0.9, 0.999), eps=1e-8)

    log_path = out / "training.jsonl"
    start = time.perf_counter()
    model.train()
    step = 0
    micro = 0
    loss_window = 0.0
    logf = log_path.open("w")
    while step < args.steps:
        opt.zero_grad(set_to_none=True)
        loss_sum = 0.0
        for _ in range(args.accumulation):
            row = train[micro % len(train)]
            micro += 1
            logits = model([row])[0].float()
            target = torch.tensor(row["target"], device=logits.device)
            loss = -(target * logits.log_softmax(-1)).sum()
            loss = loss + args.brier_weight * ((logits.softmax(-1) - target) ** 2).sum()
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite loss")
            (loss / args.accumulation).backward()
            loss_sum += float(loss.detach()) / args.accumulation
        norm = torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 1.0)
        if not torch.isfinite(norm):
            raise FloatingPointError("nonfinite grad norm")
        opt.step()
        step += 1
        loss_window += loss_sum
        rec = {"step": step, "loss": loss_sum, "grad_norm": float(norm),
               "elapsed_seconds": time.perf_counter() - start}
        logf.write(json.dumps(rec) + "\n")
        logf.flush()
        if step % 10 == 0 or step == args.steps:
            print(json.dumps({"event": "train", **rec,
                              "avg10": loss_window / 10}), flush=True)
            loss_window = 0.0
        if args.checkpoint_every and step % args.checkpoint_every == 0:
            model.save(out / "checkpoint")
            print(json.dumps({"event": "mid_checkpoint", "step": step}), flush=True)
    logf.close()

    model.save(out / "checkpoint")
    cal = evaluate(model, calibration, args.batch_size)
    temperature = fit_temperature(cal, "calibration")
    (out / "checkpoint/temperature.json").write_text(json.dumps(
        {"temperature": temperature, "split": "calibration", "n": len(cal)}, indent=2) + "\n")
    trained = evaluate(model, test, args.batch_size)
    trained_ood = evaluate(model, ood, args.batch_size)
    summary = {"status": "complete", "trainable_params": trainable,
               "steps": args.steps, "temperature": temperature,
               "metrics": {}}
    for name, values, temp in [
            ("baseline_test", baseline, 1.0), ("baseline_ood", baseline_ood, 1.0),
            ("baseline_calibrated_test", baseline, base_temperature),
            ("baseline_calibrated_ood", baseline_ood, base_temperature),
            ("trained_test", trained, 1.0), ("trained_ood", trained_ood, 1.0),
            ("calibrated_test", trained, temperature),
            ("calibrated_ood", trained_ood, temperature)]:
        probs = [softmax(r["logits"], temp) for r in values]
        m = evaluate_probabilities([r["target"] for r in values], probs)
        for kind in sorted({r["kind"] for r in values}):
            pairs = [(r, p) for r, p in zip(values, probs) if r["kind"] == kind]
            m[f"by_kind_{kind}"] = evaluate_probabilities(
                [r["target"] for r, _ in pairs], [p for _, p in pairs])
        summary["metrics"][name] = m
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"event": "done", "baseline_test": summary["metrics"]["baseline_test"]["brier"],
                      "trained_test": summary["metrics"]["trained_test"]["brier"],
                      "trained_ood": summary["metrics"]["trained_ood"]["brier"]}), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", default=str(BASE_DIR), help="base model dir or HF id")
    p.add_argument("--data-dir", default=str(DATA_DIR))
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--train-rows", type=int, default=800)
    p.add_argument("--calibration-rows", type=int, default=64)
    p.add_argument("--eval-rows", type=int, default=96)
    p.add_argument("--max-length", type=int, default=1536)
    p.add_argument("--lora-rank", type=int, default=8)
    p.add_argument("--accumulation", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--head-lr", type=float, default=1e-4)
    p.add_argument("--brier-weight", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--threads", type=int, default=0)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--checkpoint-every", type=int, default=0,
                   help="save runnable checkpoint every N steps")
    p.add_argument("--output", default=str(Path.home() / "jev-lfm/runs/pilot1"))
    run(p.parse_args())


if __name__ == "__main__":
    main()
