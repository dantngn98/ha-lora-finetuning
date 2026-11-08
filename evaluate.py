"""Evaluate every mode on test.jsonl and write a summary CSV.

Modes:
  base_zeroshot : base model, short prompt, no examples (shows how badly format breaks)
  base_fewshot  : base model, full schema + definitions + k training examples (the strong baseline)
  finetuned     : base + LoRA adapter, short prompt (same prompt as training)

Usage:
  python evaluate.py   (edit the settings below to choose modes, adapter, etc.)
Writes <OUT_DIR>/<tag>_metrics.json and <tag>_preds.jsonl per mode, plus <OUT_DIR>/summary.csv.
Greedy decoding, same 4-bit base in every mode.
"""
import csv
import json
import pathlib
import time
import torch
from common import (MODEL_ID, HERE, FIELDS, load_jsonl, load_schema, short_prompt, fewshot_prompt,
                    pick_shots, parse_prediction, compute_metrics, load_base_4bit)

# ---------- settings ----------
MODES = ["base_zeroshot", "base_fewshot", "finetuned"]  # remove any to skip them
ADAPTER = "adapters/seed42"  # finetuned mode only
K = 8                        # few-shot examples (base_fewshot only)
TRAIN = str(HERE / "train.jsonl")
TEST = str(HERE / "test.jsonl")
OUT_DIR = "results"

ALL_MODES = ["base_zeroshot", "base_fewshot", "finetuned"]  # run in this order on one loaded base model
assert set(MODES) <= set(ALL_MODES), f"bad MODES {MODES!r}"
if "finetuned" in MODES:
    assert pathlib.Path(ADAPTER).exists(), f"adapter {ADAPTER} not found: run train_lora.py first"

train, test, schema = load_jsonl(TRAIN), load_jsonl(TEST), load_schema()
shots = pick_shots(train, K)
pathlib.Path(OUT_DIR).mkdir(exist_ok=True)


def build_inputs(mode, complaint, tok, device):
    prompt = (fewshot_prompt(complaint, schema, shots) if mode == "base_fewshot"
              else short_prompt(complaint))
    return tok(prompt, return_tensors="pt").to(device)


def run_mode(mode, model, tok, model_name):
    rows, preds, prompt_tokens, times = [], [], [], []
    for r in test:
        inputs = build_inputs(mode, r["complaint"], tok, model.device)
        t0 = time.time()
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=96, do_sample=False, stop_strings=["}"],
                                 tokenizer=tok, eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id)
        times.append(time.time() - t0)
        text = tok.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        pred = parse_prediction(text)
        prompt_tokens.append(inputs["input_ids"].shape[1])
        preds.append(pred)
        rows.append({"id": r["id"], "complaint": r["complaint"], "gold": {f: r[f] for f in FIELDS},
                     "raw_output": text, "pred": pred})

    metrics = compute_metrics(test, preds, train, schema)
    metrics["mode"] = mode
    metrics["model"] = model_name
    metrics["avg_prompt_tokens"] = round(sum(prompt_tokens) / len(prompt_tokens), 1)
    metrics["avg_seconds_per_example"] = round(sum(times) / len(times), 2)
    metrics["k_shots"] = len(shots) if mode == "base_fewshot" else 0

    tag = mode if mode != "finetuned" else f"finetuned_{pathlib.Path(ADAPTER).name}"
    with open(f"{OUT_DIR}/{tag}_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    with open(f"{OUT_DIR}/{tag}_preds.jsonl", "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    n = metrics["n"]
    print(f"\n== {tag} (n={n}) ==")
    print(f"valid JSON {metrics['valid_json']}/{n} | schema-valid {metrics['schema_valid']}/{n} | "
          f"all-fields exact {metrics['exact_match_all_fields']}/{n}")
    for f in FIELDS:
        d = metrics["fields"][f]
        print(f"{f:18s} {d['correct']}/{n}   (seen labels {d['seen_label_correct']}/{d['seen_label_n']}, "
              f"unseen {d['unseen_label_correct']}/{d['unseen_label_n']})")
    print(f"avg prompt tokens {metrics['avg_prompt_tokens']} | avg sec/example {metrics['avg_seconds_per_example']}")
    return metrics


def summary_row(m):
    row = {k: m[k] for k in ("mode", "model", "n", "valid_json", "schema_valid", "exact_match_all_fields")}
    for f in FIELDS:
        d = m["fields"][f]
        row[f"{f}_correct"] = d["correct"]
        row[f"{f}_seen_correct"] = d["seen_label_correct"]
        row[f"{f}_seen_n"] = d["seen_label_n"]
        row[f"{f}_unseen_correct"] = d["unseen_label_correct"]
        row[f"{f}_unseen_n"] = d["unseen_label_n"]
    row.update({k: m[k] for k in ("k_shots", "avg_prompt_tokens", "avg_seconds_per_example")})
    return row


all_metrics = []
model, tok, _ = load_base_4bit(MODEL_ID)
model.eval()
for mode in [m for m in ALL_MODES if m in MODES]:
    name = MODEL_ID
    if mode == "finetuned":  # wraps the base model in place, so it runs after the base-only modes
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, ADAPTER)
        model.eval()
        name = f"{MODEL_ID} + {ADAPTER}"
    all_metrics.append(run_mode(mode, model, tok, name))

summary = [summary_row(m) for m in all_metrics]
with open(f"{OUT_DIR}/summary.csv", "w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(summary[0]))
    w.writeheader()
    w.writerows(summary)
print(f"\nwrote {OUT_DIR}/summary.csv")
