# MiniMind spam classifier

A native binary classifier built from the Dense MiniMind backbone. `lm_head` is
`Linear(hidden_size, 2, bias=True)` and is not tied to the input embeddings.
Outputs are `[batch, 2]` logits: **0 = 스팸아님 (ham), 1 = 스팸 (spam)**.
Only the last Transformer block, final RMSNorm, and classification head train.
All other parameters are frozen. The original language model is unchanged.

The encoder adds BOS and EOS once, reserves EOS when truncating, and pools the
last valid (EOS) hidden state. Right padding only; empty/whitespace text and
all-padding, left-padding, or missing-EOS model inputs are rejected.

Run from `projects/minimind/` using the existing MiniMind notebook environment:

```bash
MINIMIND_PYTHON=../../notebooks/minimind/.venv/bin/python
"$MINIMIND_PYTHON" -m unittest trainer.test_spam_classifier -v
"$MINIMIND_PYTHON" trainer/train_spam.py \
  --base ../../outputs/minimind/hf-release-v1.0/MiniMind-Dense-64M-Mixed-SFT-v1.0 \
  --output ../../outputs/minimind/<NEW-RUN-ID> \
  --epochs 12 --batch-size 32 --max-length 340 --seed 42 --patience 4
"$MINIMIND_PYTHON" eval_spam.py \
  --checkpoint ../../outputs/minimind/20261119-001300-spam-last-block/best \
  --text 'Hi, are we still meeting for lunch at noon?'
```

Replace the base path with an existing compatible Dense MiniMind directory
containing `config.json`, `model.safetensors` and its tokenizer. The trainer
requires CUDA and an output directory that does not already exist. A local
native `pytorch_model.bin` can alternatively supply the base state dictionary.
No model download or cloud provisioning happens implicitly. Without
`--raw-archive`, the trainer downloads the UCI SMS archive. With
`--raw-archive /path/to/sms_spam_collection.zip`, it uses that local archive.

Data source: [UCI SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection),
Almeida & Hidalgo (2011), DOI 10.24432/C5CC84, CC BY 4.0. Messages are English.
Exact normalized duplicates are removed. URL/number-normalized templates and
char-ngrams with cosine similarity >= 0.90 are grouped before a seeded
StratifiedGroupKFold split: approximately 80% train, 10% validation, 10% test.
This heuristic reduces template leakage; it is not semantic decontamination.
The raw input text, URLs and numbers are preserved for model training.

Training uses FP32 parameters, BF16 autocast, unweighted binary cross-entropy,
AdamW (last block/norm LR 5e-5, head LR 1e-3, weight decay .01), 10% warmup,
cosine decay, and clip norm 1. Select checkpoints by validation spam F1 with
validation loss as the tie-breaker; early-stop after four unimproved epochs.
The test set is evaluated after selection and is never used for tuning.

`best/` contains full backbone/head weights, configuration, tokenizer, labels,
and input policy. `summary.json`, per-message predictions, data/source hashes,
training history and freeze/reload checks are written under the output run.
Softmax scores are not calibrated probabilities. English SMS test accuracy is
not evidence of Korean-message performance or current phishing robustness.

See the completed-run report at
[reports/minimind/20261119-001300-spam-last-block.md](../../../../reports/minimind/20261119-001300-spam-last-block.md).
