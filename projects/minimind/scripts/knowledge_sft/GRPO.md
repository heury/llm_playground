# Small arithmetic GRPO

Start from the original `caspar/MiniMind-Dense-64M-Mixed-SFT-v1.0`, revision
`7597c22aa4d3e7b2ac824b4b4be01158cccf1b8d`. Do not use the fictional-product
SFT or concise-answer DPO checkpoint. Download that revision's config, tokenizer
and safetensors into a local directory. The local native model loads weights
strictly; downloaded Python code is not executed.

From `projects/minimind/`:

```bash
uv run python scripts/knowledge_sft/grpo_arithmetic.py \
  --base /tmp/minimind-knowledge-base \
  --output ../../outputs/minimind/<YYYYMMDD-HHMMSS>-grpo-arithmetic
```

The output directory must not exist; CUDA is required. The main experiment uses
no SFT warmup and no assistant answer prefix. Optional `--pilot-only` samples
training questions without updates; `--answer-prefix 'The answer is '` is a
diagnostic option, not a source of answer digits. Record any use explicitly.

Data enumerates addition of digits 0–9 with commuted pairs deduplicated, plus
nonnegative subtraction of digits 0–9. Fixed prompt examples `1 + 1 = 2` and
`4 - 1 = 3` are excluded from every split, leaving 108 facts. A seed-42 split
creates 64 train, 12 validation and 32 test facts, each balanced by operation.
No repeated fact or commuted addition pair crosses splits. Answers are computed
from the operands; only question text and the two fixed examples enter prompts.

Reward is 1 only if the entire stripped generated response is an ASCII signed
integer equal to the gold result; otherwise 0. There is no reward for formatting,
length, reasoning text or a correct substring. Greedy strict accuracy is primary;
last-number accuracy is a secondary diagnostic and may count accidental matches.
An answer token budget failure is retained in the evaluation denominator.

Use two epochs, equal-prompt-length batches of up to two questions, eight sampled
answers per question, at most 32 new tokens, LR 1e-6 constant, reference KL .04,
GRPO clip .2, AdamW weight decay 0, gradient clip 1, seed 42. Sampling uses
temperature 1, top-p 1, top-k 0 with no answer constraints. All model parameters
are trainable; parameters/optimizer/log_softmax are FP32 with BF16 autocast.
The frozen reference is a separate copy of the original base. This implements
the clipped GRPO objective from `trainer/train_grpo.py`, not CISPO.

Each freshly sampled rollout gets one optimizer update. Old log probabilities
are detached from the same pre-update policy forward; no stale rollout reuse
occurs. This is an on-policy ratio of one at the update's start, not evidence
of multi-update PPO clipping. Advantages use each question's population standard
deviation plus 1e-4. Equal-reward groups contribute no task gradient; KL still
applies. If policy/reference also match, an all-zero-reward run can produce no
parameter changes despite optimizer.step calls. Report this as failure to learn.

First EOS is included in the loss mask; everything after it is excluded. Prompts
are batched by identical length, avoiding padding/RoPE position effects. The
published chat template has empty think blocks removed deterministically, and
total context remains below 340 tokens. Native generate returns inference-mode
tensors: clone them before gradient-tracked likelihood computation.

Before and after: greedy all train/validation/test questions; six general prompts
at 96 new tokens; a fixed 12-training-question sampled probe (eight per question,
same RNG seed) measures mixed-reward groups. Validation is diagnostic only and
the last epoch is used. Checks cover reward enumeration/substring rejection,
EOS masks, finite losses/gradients/weights, initial reference KL zero, unit
on-policy ratios, frozen-reference hash, actual parameter differences and exact
FP32 checkpoint/logit reload. Do not require nonzero changes as a fake pass:
zero changes are a meaningful possible result to report.

All inputs, sample text, reward traces, greedy results, metadata and checkpoint
stay in root `outputs/minimind/<run>/`; reports and reusable procedures stay in
root `reports/minimind/` and `test_scenarios/minimind/`. This small single-seed
experiment does not establish general arithmetic or reasoning ability.
