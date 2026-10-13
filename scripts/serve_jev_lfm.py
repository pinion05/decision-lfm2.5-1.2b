"""Serve Jev-LFM decisions over HTTP (FastAPI-free, stdlib only).

POST /decide  {"state": ..., "questions": {...}}   -> jev typed answer format
Port of jev/api.py compile_request + format_response, keeping the same wire
semantics so it can be a drop-in for the Open-Jev server contract.
"""
from __future__ import annotations

import json
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from train_jev_lfm import LfmDecisionModel, BASE_DIR, softmax  # noqa: E402
import torch  # noqa: E402

CHECKPOINT = Path.home() / "jev-lfm/runs/pilot1/checkpoint"
MAX_LENGTH = 1536


def _render(value) -> str:
    return value if isinstance(value, str) else json.dumps(
        value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def compile_request(state, questions: dict) -> list[dict]:
    if not isinstance(questions, dict) or not questions:
        raise ValueError("questions must be a nonempty mapping")
    records = []
    for qid, definition in questions.items():
        kind = definition.get("type")
        if kind not in ("choice", "score", "noul"):
            raise ValueError("question type must be choice, score, or noul")
        question = definition.get("instructions")
        if question is None or not isinstance(question, (str, dict, list)):
            raise ValueError("instructions must be text, object or array")
        question = _render(question)
        rec = {"id": qid, "state": state, "kind": kind, "question": question}
        if kind == "choice":
            criteria = definition["criteria"]
            if not (1 <= len(criteria) <= 255):
                raise ValueError("choice needs 1..255 candidates")
            rec["answer_keys"] = list(criteria)
            rec["options"] = [
                name if desc is None else f"{name}: {desc}"
                for name, desc in zip(criteria, criteria.values())]
        elif kind == "score":
            levels = definition["criteria"]
            if not (2 <= len(levels) <= 10):
                raise ValueError("score needs 2..10 levels")
            rec["options"] = [_render(l) for l in levels]
            rec["answer_keys"] = [str(i) for i in range(len(levels))]
            rec["legend"] = {str(i): l for i, l in enumerate(levels)}
        else:
            criteria = definition.get("criteria")
            if criteria:
                question += f"\nYes means: {_render(criteria['true'])}\nNo means: {_render(criteria['false'])}"
                rec["question"] = question
            rec["options"] = ["no", "yes"]
            rec["answer_keys"] = ["false", "true"]
        records.append(rec)
    return records


def format_response(records, probs_rows, temperature: float = 1.0) -> dict:
    answers = {}
    for rec, logits in zip(records, probs_rows):
        scaled = softmax([x / temperature for x in logits])
        keys = rec["answer_keys"]
        if rec["kind"] == "noul":
            answers[rec["id"]] = {"type": "noul", "noul": scaled[1]}
        elif rec["kind"] == "choice":
            best = max(range(len(scaled)), key=scaled.__getitem__)
            answers[rec["id"]] = {"type": "choice", "choice": keys[best],
                                  "probabilities": dict(zip(keys, scaled))}
        else:
            answers[rec["id"]] = {"type": "score",
                                  "score": sum(i * v for i, v in enumerate(scaled)),
                                  "probabilities": dict(zip(keys, scaled)),
                                  "legend": rec["legend"]}
    return {"answers": answers}


class Handler(BaseHTTPRequestHandler):
    model = None
    temperature = 1.0

    def do_POST(self):
        if self.path != "/decide":
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length", 0))
        payload = json.loads(self.rfile.read(length) or b"{}")
        try:
            records = compile_request(payload["state"], payload["questions"])
            with torch.inference_mode():
                logits = self.model(records)
            response = format_response(records, [[float(x) for x in lg] for lg in logits],
                                       self.temperature)
        except Exception as exc:  # noqa: BLE001
            body = json.dumps({"error": str(exc)}).encode()
            self.send_response(400)
        else:
            body = json.dumps(response, ensure_ascii=False).encode()
            self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8321
    cpu = "--cpu" in sys.argv
    device = "cpu" if cpu else "cuda"
    print(f"loading {CHECKPOINT} on {device} ...", flush=True)
    Handler.model = LfmDecisionModel.load(CHECKPOINT, device=device, max_length=MAX_LENGTH)
    temp_file = CHECKPOINT / "temperature.json"
    if temp_file.exists():
        Handler.temperature = json.loads(temp_file.read_text())["temperature"]
    print(f"ready on :{port} (temperature={Handler.temperature})", flush=True)
    HTTPServer(("127.0.0.1", port), Handler).serve_forever()


if __name__ == "__main__":
    main()
