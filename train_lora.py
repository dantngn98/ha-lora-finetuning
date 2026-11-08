import json
import time
import torch
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments, set_seed
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from common import MODEL_ID, HERE, load_jsonl, short_prompt, load_base_4bit

# ---------- settings ----------
TRAIN = str(HERE / "train.jsonl")
OUT = "adapters/seed42"
EPOCHS = 4
LR = 2e-4
R = 16
ALPHA = 32
DROPOUT = 0.05
BATCH = 4
GRAD_ACCUM = 1
MAX_LEN = 256
SEED = 42
CONFIG = dict(train=TRAIN, out=OUT, epochs=EPOCHS, lr=LR, r=R, alpha=ALPHA, dropout=DROPOUT,
              batch=BATCH, grad_accum=GRAD_ACCUM, max_len=MAX_LEN, seed=SEED)

set_seed(SEED)

model, tok, dtype = load_base_4bit(MODEL_ID)
model.config.use_cache = False
model = prepare_model_for_kbit_training(
    model, use_gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False})
model = get_peft_model(model, LoraConfig(
    r=R, lora_alpha=ALPHA, lora_dropout=DROPOUT, bias="none", task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
model.print_trainable_parameters()


class SFTData(Dataset):
    def __init__(self, recs):
        self.items = []
        for r in recs:
            p = tok(short_prompt(r["complaint"]), add_special_tokens=True)["input_ids"]  # adds BOS
            c = tok(r["output"], add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
            assert len(p) + len(c) <= MAX_LEN, f"row {r['id']} too long: raise MAX_LEN"
            self.items.append({"input_ids": p + c, "labels": [-100] * len(p) + c})

    def __len__(self): return len(self.items)
    def __getitem__(self, i): return self.items[i]


def collate(batch):
    m = max(len(b["input_ids"]) for b in batch)
    ids = [b["input_ids"] + [tok.pad_token_id] * (m - len(b["input_ids"])) for b in batch]
    lab = [b["labels"] + [-100] * (m - len(b["labels"])) for b in batch]
    att = [[1] * len(b["input_ids"]) + [0] * (m - len(b["input_ids"])) for b in batch]
    return {"input_ids": torch.tensor(ids), "labels": torch.tensor(lab), "attention_mask": torch.tensor(att)}


recs = load_jsonl(TRAIN)
ds = SFTData(recs)
targs = TrainingArguments(
    output_dir=OUT, per_device_train_batch_size=BATCH,
    gradient_accumulation_steps=GRAD_ACCUM, num_train_epochs=EPOCHS,
    learning_rate=LR, lr_scheduler_type="cosine", warmup_steps=0.1, weight_decay=0.0,  # <1 means ratio of total steps
    logging_steps=1, save_strategy="no", report_to="none", optim="paged_adamw_8bit",
    bf16=(dtype == torch.bfloat16), fp16=(dtype == torch.float16),
    remove_unused_columns=False, seed=SEED)
trainer = Trainer(model=model, args=targs, train_dataset=ds, data_collator=collate)

t0 = time.time()
trainer.train()
secs = time.time() - t0

model.save_pretrained(OUT)
tok.save_pretrained(OUT)
tokens_per_epoch = sum(len(x["input_ids"]) for x in ds.items)
losses = [h["loss"] for h in trainer.state.log_history if "loss" in h]
info = {"model": MODEL_ID, "n_train": len(ds), "seconds": round(secs, 1),
        "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
        "train_tokens_total": int(tokens_per_epoch * EPOCHS),
        "first_loss": losses[0] if losses else None, "final_loss": losses[-1] if losses else None,
        "gpu": torch.cuda.get_device_name(0), "args": CONFIG}
json.dump(info, open(f"{OUT}/run_info.json", "w"), indent=2)
print(json.dumps(info, indent=2))
