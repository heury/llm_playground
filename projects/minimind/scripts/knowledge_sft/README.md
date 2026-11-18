# Small full-SFT knowledge recall experiment

Run from `projects/minimind/` using this project's uv environment. This harness
loads the published checkpoint into the existing native MiniMind implementation
with strict safetensors loading; it does not execute downloaded model code.

Base: `caspar/MiniMind-Dense-64M-Mixed-SFT-v1.0`, pinned revision
`7597c22aa4d3e7b2ac824b4b4be01158cccf1b8d`.

The deterministic seed-42 synthetic catalog contains 20 fictional products and
60 product/attribute facts. There are five training question templates per fact
(300 training pairs) and two separate evaluation templates (120 held-out pairs).
Facts overlap intentionally; question strings do not. This tests recall of
trained facts under changed wording, not inference of completely unseen facts.
English avoids conflating this English base model's Korean ability with recall.
Answers contain only a color, a number of hours, or a number of years.

Use `huggingface_hub.snapshot_download` to download config, tokenizer files and
`model.safetensors` at the pinned revision into a local directory, then run:

```bash
uv run python scripts/knowledge_sft/run.py \
  --base /tmp/minimind-knowledge-base \
  --output ../../outputs/minimind/<YYYYMMDD-HHMMSS>-knowledge-sft \
  --epochs 10 --lr 1e-5
```

CUDA is required. All 63,912,192 parameters are trained with FP32 AdamW state,
BF16 autocast, batch 8, constant LR, clip 1, weight decay 0.01 and seed 42.
There are 38 updates per epoch, including the final batch of four examples.
Right padding only follows real training tokens; assistant-only labels use the
native `SFTDataset.generate_labels`, with explicit prefix/EOS checks. The
published chat template has its empty thinking block removed deterministically.
No random system prompt is inserted. Dynamic sequence lengths avoid excessive
padding; no training sample is truncated.

Before and after training, greedy generation evaluates all training and held-out
questions, five unknown products and eight general prompts. Equal-length
evaluation prompts are batched without padding. Generation is capped at 48 new
tokens and remains below the base model's trained 340-token context. Intermediate
held-out results at epochs 1 and 5 are diagnostics; they do not select a checkpoint.
The last epoch is always used. No optimizer resume is saved.

Primary accuracy uses exact color-word recognition without competing catalog
colors, or the exact expected numeric value with the correct unit and no other
numeric values or conflicting units. A second, stricter metric compares the
entire normalized answer. These are automatic value checks, not semantic judging;
raw answers are saved for inspection. Unknown-product and general prompts have
no automatic pass/fail claim. Main predeclared thresholds: 95% training accuracy,
80% held-out accuracy, and an improvement over the base.

`metadata.json` records hashes, versions, model/data settings and reload checks;
`training.jsonl` records assistant-token-weighted training loss; `summary.json`
and per-stage JSONL files record generation results. `full_sft_768.pth` contains
the full FP32 state and is verified by exact tensor comparison and strict reload
before final evaluation. Original weights are never overwritten. Keep all run
artifacts in root `outputs/minimind/` and reports in root `reports/minimind/`.
