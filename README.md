# Hearing aid complaint classification with QLoRA

Fine-tunes a LoRA adapter on **Llama-3.1-8B** (the base model, not Instruct) to turn a free-text hearing aid
patient complaint into a one-line JSON object, and compares it against prompting the same base model with
no examples (zero-shot) and with the full label schema plus examples (few-shot).

```
Hearing aid patient complaint:
When we went to the diner on Sunday, the clatter of plates and the background chatter was absolutely deafening.

Structured JSON:
{"primary_category": "sound_quality", "specific_issue": "background_noise_annoying", "situation": "noisy_restaurant_or_group", "suggested_action": "adjust_programming", "urgency": "routine"}
```

| Field | Allowed values |
|---|---|
| `primary_category` | 7 categories (sound quality, speech understanding, connectivity, ...) |
| `specific_issue` | 41 issues, each belonging to exactly one category |
| `situation` | 11 settings, including `not_stated` |
| `suggested_action` | 9 first clinic steps |
| `urgency` | `routine`, `soon`, `urgent` (`urgent` always pairs with `escalate_medical`) |

The full list with definitions is in [`schema.json`](schema.json).

## Repository structure

```
.
├── common.py               Shared code: model id, prompts, 4-bit model loading, output parsing, metrics
├── train_lora.py           QLoRA fine-tune on train.jsonl -> adapters/seed42/
├── evaluate.py             Runs all modes on test.jsonl -> results/
├── analyze_results.ipynb   Tables and charts comparing the runs in results/
├── analyze_results.html    HTML export of the notebook
├── train.jsonl             38 training examples
├── test.jsonl              12 test examples
├── schema.json             Allowed labels and their definitions
├── requirements.txt
├── adapters/seed42/        Trained LoRA adapter + run_info.json (written by train_lora.py; not in git)
└── results/                Metrics, predictions, and summary.csv (written by evaluate.py)
```

### Data format

Each line of `train.jsonl` / `test.jsonl` is one example. Ids run 1-38 in train and continue 39-50 in test.

```json
{"id": 1, "complaint": "...", "primary_category": "...", "specific_issue": "...", "situation": "...",
 "suggested_action": "...", "urgency": "...", "output": "{\"primary_category\": ...}"}
```

`output` is the target JSON string the model is trained to produce, with the five fields in a fixed order.

## Setup

Tested on Windows 11 with an RTX 4070 (12 GB), Python 3.13.

1. Create a virtual environment and install a **CUDA build** of PyTorch first (a plain `pip install torch`
   gives the CPU-only build on Windows), then the rest:

   ```
   python -m venv .venv
   .venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu130
   .venv\Scripts\python.exe -m pip install -r requirements.txt
   ```

2. Accept the Llama 3.1 license at https://huggingface.co/meta-llama/Llama-3.1-8B, then log in:

   ```
   .venv\Scripts\hf.exe auth login
   ```

   The first run downloads the model (about 16 GB) to `~/.cache/huggingface/hub`.

Versions used for the current results: torch 2.14.1+cu130, transformers 5.22.2, peft 0.21.2,
bitsandbytes 0.50.2, accelerate 1.15.0. `requirements.txt` is unpinned on purpose.

## How to run

The scripts take no command-line arguments. To change a run, edit the `# ---------- settings ----------`
block at the top of each script.

### 1. Train the adapter

```
.venv\Scripts\python.exe train_lora.py
```

Loads the base model in 4-bit (NF4), adds LoRA adapters to all attention and MLP projections, and trains on
`train.jsonl`. Loss is computed only on the JSON answer and end-of-sequence token, not the prompt. Takes
about 30 seconds on an RTX 4070 once the model is downloaded, with a peak of about 9.9 GB of GPU memory.

| Setting | Default | |
|---|---|---|
| `OUT` | `adapters/seed42` | Output directory (overwritten if it exists) |
| `EPOCHS` | 4 | |
| `LR` | 2e-4 | Cosine schedule, 10% warmup |
| `R` / `ALPHA` / `DROPOUT` | 16 / 32 / 0.05 | LoRA rank, scale, dropout |
| `BATCH` / `GRAD_ACCUM` | 4 / 1 | |
| `MAX_LEN` | 256 | Max prompt + answer tokens; training stops with an error if a row is longer |
| `SEED` | 42 | |

Writes the adapter and `run_info.json` (settings, time, peak GPU memory, first and final loss).

### 2. Evaluate

```
.venv\Scripts\python.exe evaluate.py
```

Runs each mode on `test.jsonl` with greedy decoding, using the same 4-bit base model throughout:

| Mode | Model | Prompt |
|---|---|---|
| `base_zeroshot` | Base | Short prompt only (shows the base model doesn't produce the format on its own) |
| `base_fewshot` | Base | Full schema with definitions + 8 training examples + short prompt (about 2,000 tokens) |
| `finetuned` | Base + adapter | Short prompt only, the same prompt used in training |

The few-shot examples are picked round-robin across categories from `train.jsonl`, so they cover every
category. Settings: `MODES` (remove a mode to skip it), `ADAPTER` (default `adapters/seed42`), `K` (number
of few-shot examples), `OUT_DIR` (default `results`). Takes about 3 minutes for all three modes.

Writes to `results/`:

- `<tag>_metrics.json`: all metrics for one run
- `<tag>_preds.jsonl`: one line per test row with the gold labels, the raw model output, and the parsed prediction
- `summary.csv`: one row per run with every metric as a column

`<tag>` is the mode name, or `finetuned_<adapter folder name>` for the fine-tuned run.

**Metrics** (all counts out of the 12 test rows):

- **valid JSON**: output parses and has all five keys
- **schema-valid**: every value is an allowed label and `specific_issue` belongs to the predicted `primary_category`
- **all fields exact**: all five fields match the gold labels
- **per-field accuracy**, also split into rows whose gold label does or doesn't appear in `train.jsonl` (seen / unseen)

### 3. Analyze

Open `analyze_results.ipynb` in VS Code or Jupyter, choose the `.venv` Python as the kernel, and run all
cells. It reads everything in `results/` and shows:

- the exact prompt each mode sends, for the first test complaint
- overview tables and bar charts of validity and per-field accuracy
- seen vs unseen label accuracy, and prompt size and speed per run
- a per-row, per-field grid marking each prediction as correct, wrong, invented (not in the schema), or missing
- every error, the invented labels, and a `primary_category` confusion table per run

Additional adapters you evaluate (for example a `finetuned_seed7` run) show up automatically.

## Current results

38 training rows, seed 42, 12 test rows:

| Mode | Schema-valid | All fields exact | primary_category | specific_issue | situation | suggested_action | urgency |
|---|---|---|---|---|---|---|---|
| base_zeroshot | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| base_fewshot | 11 | 4 | 10 | 6 | 12 | 9 | 11 |
| finetuned | 6 | 5 | 9 | 7 | 11 | 11 | 12 |

- The fine-tuned model is roughly level with few-shot on accuracy, but half its outputs contain a label that
  isn't in the schema (for example `restaurant_or_group` instead of `noisy_restaurant_or_group`), because it
  never sees the list of allowed labels.
- With 12 test rows, one row is about 8 percentage points, so small differences are noise.
- Every test label also appears in the training set, so these results don't show how the models handle a
  label they've never seen.

## Notes

- The scripts default to bf16 and fall back to fp16 on GPUs without bf16 support (for example a T4).
- On Windows, Hugging Face warns that its cache can't use symlinks without Developer Mode. The warning is
  harmless; the cache stores plain copies instead.
- `.venv` is about 3.4 GB. If the project lives in a synced folder such as OneDrive, exclude `.venv` from sync.
