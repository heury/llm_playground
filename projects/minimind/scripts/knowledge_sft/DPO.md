# Concise-answer DPO smoke experiment

This is a separate experiment from the fictional-catalog full SFT. Start from
the original `caspar/MiniMind-Dense-64M-Mixed-SFT-v1.0`, revision
`7597c22aa4d3e7b2ac824b4b4be01158cccf1b8d`, not the catalog checkpoint.
Download that revision's config, tokenizer and safetensors into a local directory.
No downloaded Python code is executed: loading uses the local native model with
strict safetensors validation.

From `projects/minimind/`:

```bash
uv run python scripts/knowledge_sft/dpo_concise.py \
  --base /tmp/minimind-knowledge-base \
  --output ../../outputs/minimind/<YYYYMMDD-HHMMSS>-dpo-concise
```

The output directory must not exist. The script requires CUDA and writes all
data, raw generations, checks, scores and a full FP32 checkpoint to that directory.
It does not upload models or allocate cloud resources.

The deterministic seed-42 dataset contains 60 arithmetic facts, 30 capital-city
facts, 30 opposite-word facts and 30 term definitions. Every preferred answer is
short and correct; the rejected answer repeats that same answer inside filler.
Five filler wrappers are reused across all splits. Fact and question groups are
disjoint across 100 training, 20 validation and 30 test examples. Each split has
the same category proportions. This is a synthetic style test, not a benchmark
for broad human preference alignment. The definitions and capital/opposite facts
are authored here; arithmetic answers are computed from the operands.

All 63,912,192 parameters are updated using native `trainer.train_dpo.dpo_loss`.
The original base is loaded into a separate frozen reference model. Training is
2 epochs, 50 updates, batch 4 pairs, LR 5e-7 constant, beta 0.1, AdamW defaults
except weight decay 0, clip 1, FP32 parameters/states and BF16 autocast. Sequence
log probabilities use FP32 log_softmax and assistant tokens including EOS.
The native DPODataset mask is validated against the full prompt/response boundary.
Both responses must be shorter than 340 tokens; none is truncated. Padding is
on the right and excluded from loss. No random system prompt or think block is
inserted; the public tokenizer template has the empty think block removed.

Checks include identical-model DPO loss equal to ln(2), a real finite nonzero
gradient, objective gradient direction, a frozen reference with unchanged hash,
changed policy parameters, finite final weights, strict exact FP32 reload, and
exact test likelihood scores before/after checkpoint reload. Validation is
diagnostic only: the last epoch is always evaluated, with no checkpoint search.

Evaluation reports reference-relative DPO margin/loss and sequence and per-token
preference scores. Raw sequence scores inherently favor shorter responses here.
At baseline, policy/reference are identical, so reference-relative improvements
are ties, not evidence that baseline preference accuracy is zero.

Before/after greedy generations cover all 100 training and all 30 test prompts,
plus eight longer-answer controls. Short prompts allow 96 new tokens; controls
allow 128. Prompts are grouped by equal length without padding, and total context
is always below 340 tokens. Length excludes EOS. Report length-limit counts.

Automatic answer checks use the final numeric value for arithmetic, answer-word
matching for capitals/opposites, and keyword alternatives for definitions. They
are approximate and cannot establish semantic correctness: manually inspect all
30 held-out answers and the eight controls when reporting results. Filler uses
a fixed phrase regex; repetition counts duplicate four-word spans. Report
correctness, length, filler and repetition together, and do not treat a smaller
DPO loss or a shorter wrong answer as a successful style improvement.

Reports belong in root `reports/minimind/`; reusable procedures belong in root
`test_scenarios/minimind/`. Original weights and earlier SFT results are preserved.
