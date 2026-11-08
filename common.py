"""Shared helpers: prompt formats, model loading, parsing, and metrics."""
import json
import pathlib

MODEL_ID = "meta-llama/Llama-3.1-8B"  # base model (NOT Instruct): no chat template
FIELDS = ["primary_category", "specific_issue", "situation", "suggested_action", "urgency"]
HERE = pathlib.Path(__file__).parent


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_schema(path=HERE / "schema.json"):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ---------- prompts ----------
def short_prompt(complaint):
    """Prompt used for fine-tuning and for the fine-tuned model at inference."""
    return f"Hearing aid patient complaint:\n{complaint}\n\nStructured JSON:\n"


def schema_text(schema):
    lines = ["Classify a hearing aid patient's complaint into a one-line JSON object with exactly "
             f"these keys: {', '.join(FIELDS)}. Use only the allowed values below.\n"]
    lines.append("primary_category (the single problem the patient emphasizes most):")
    for k, d in schema["primary_category"].items():
        lines.append(f"  - {k}: {d}")
    lines.append("\nspecific_issue (must belong to the chosen primary_category):")
    for cat, items in schema["specific_issue"].items():
        lines.append(f"  [{cat}]")
        for k, d in items.items():
            lines.append(f"  - {k}: {d}")
    for field, header in [("situation", "situation (setting stated or clearly implied):"),
                          ("suggested_action", "suggested_action (most likely first clinic step):"),
                          ("urgency", "urgency:")]:
        lines.append(f"\n{header}")
        for k, d in schema[field].items():
            lines.append(f"  - {k}: {d}")
    return "\n".join(lines)


def pick_shots(train, k):
    """Deterministic round-robin over primary_category so shots cover as many categories as possible."""
    by_cat = {}
    for r in train:
        by_cat.setdefault(r["primary_category"], []).append(r)
    shots, i = [], 0
    while len(shots) < k and any(by_cat.values()):
        for cat in list(by_cat):
            if i < len(by_cat[cat]) and len(shots) < k:
                shots.append(by_cat[cat][i])
        i += 1
        if all(i >= len(v) for v in by_cat.values()):
            break
    return shots


def fewshot_prompt(complaint, schema, shots):
    parts = [schema_text(schema), "\n\nExamples:\n"]
    for s in shots:
        parts.append(short_prompt(s["complaint"]) + s["output"] + "\n\n")
    parts.append(short_prompt(complaint))
    return "".join(parts)


# ---------- parsing & metrics ----------
def parse_prediction(text):
    """Take the first {...} in the generated text. Returns dict or None."""
    start = text.find("{")
    end = text.find("}", start)
    if start == -1 or end == -1:
        return None
    try:
        obj = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def schema_valid(pred, schema):
    if not pred or any(f not in pred for f in FIELDS):
        return False
    cat, iss = pred["primary_category"], pred["specific_issue"]
    return (cat in schema["primary_category"]
            and iss in schema["specific_issue"].get(cat, {})
            and pred["situation"] in schema["situation"]
            and pred["suggested_action"] in schema["suggested_action"]
            and pred["urgency"] in schema["urgency"])


def compute_metrics(golds, preds, train, schema):
    n = len(golds)
    seen = {f: {r[f] for r in train} for f in FIELDS}
    m = {"n": n,
         "valid_json": sum(p is not None and all(f in p for f in FIELDS) for p in preds),
         "schema_valid": sum(schema_valid(p, schema) for p in preds),
         "exact_match_all_fields": sum(p is not None and all(p.get(f) == g[f] for f in FIELDS)
                                       for g, p in zip(golds, preds)),
         "fields": {}}
    for f in FIELDS:
        hit = [p is not None and p.get(f) == g[f] for g, p in zip(golds, preds)]
        s_idx = [i for i, g in enumerate(golds) if g[f] in seen[f]]
        u_idx = [i for i, g in enumerate(golds) if g[f] not in seen[f]]
        m["fields"][f] = {"correct": sum(hit), "n": n,
                          "seen_label_correct": sum(hit[i] for i in s_idx), "seen_label_n": len(s_idx),
                          "unseen_label_correct": sum(hit[i] for i in u_idx), "unseen_label_n": len(u_idx)}
    return m


# ---------- model loading ----------
def load_base_4bit(model_id=MODEL_ID):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    cc_major = torch.cuda.get_device_capability()[0]
    dtype = torch.bfloat16 if cc_major >= 8 else torch.float16  # T4 has no bf16
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                               bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype)
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    kwargs = dict(quantization_config=quant, device_map={"": 0})
    try:  # newer transformers renamed torch_dtype -> dtype
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype, **kwargs)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype, **kwargs)
    return model, tok, dtype
