# Notebook-derived GRPO word problems

Procedure: [grpo-word-problems-tour.md](../../../../test_scenarios/minimind/grpo-word-problems-tour.md).
Reference notebook: [minimind_grpo_tour.py](../../../../notebooks/minimind/minimind_grpo_tour.py).

From `projects/minimind/`, with the pinned original base config/tokenizer/weights
already downloaded into a local directory:

```bash
uv run python scripts/knowledge_sft/grpo_word_problems.py \
  --base /tmp/minimind-knowledge-base \
  --output ../../outputs/minimind/<YYYYMMDD-HHMMSS>-grpo-word-problems
```

The output directory must not exist. CUDA is required. This runner does not run
the marimo notebook or execute downloaded model Python code. It uses the native
MiniMind model, GSM8KDataset, TorchRolloutEngine, scorer and grpo_train_epoch.
The original model/trainer/notebook files are not modified.

Data generation reproduces the notebook's seed-0 English one-step word problems:
128 train and 256 test, balanced addition/multiplication, operands 2–9, unique
question strings. Arithmetic expressions can recur across train/test. This is
not an unseen-equation generalization benchmark. Gold rationale/final numbers
are used only in scoring and are not included in policy prompts.

The fixed pilot takes the first eight training problems of each operation and
samples six responses each at seed 0. Training starts only if at least one answer
is correct and at least four of 16 groups have both correct and wrong responses.
Otherwise summary.json records that the gate failed and no training is claimed.

The main run explicitly uses `loss_type=grpo`, while the notebook widget defaults
to CISPO. Full-parameter native training is 2 epochs, 64 steps, batch 4, G=6,
LR 3e-7 cosine to 3e-8, KL .1, clip .2, AdamW default weight decay .01, gradient
clip 1. The rollout temperature .8/top-p .85/top-k 50 follows native generation.
This transformed sampling is not claimed to be unwarped on-policy sampling.
The reference forward uses FP32 as in the original trainer; policy forward uses
BF16 autocast. Record numerical differences instead of assuming exact initial KL.

Instrumentation retains all 1,536 training completions/rewards, native step logs
and post-clip gradient norms, checks frozen-reference gradients/hash, and checks
the initial rollout/recomputed likelihood ratio and estimated KL. These checks
are diagnostic safeguards, not an independent proof of GRPO implementation.
The runner accepts maximum initial ratio error .15 and mean estimated KL .01 to
allow native BF16 numerical differences; actual measurements are preserved.

Native checkpoint writes are redirected from ../checkpoints to the run's resume/
directory. The native FP16 checkpoint is retained in native/, but final evaluation
uses a separate full FP32 checkpoint, after exact tensor/strict-reload/logit checks.
Native trainer module globals and reward/checkpoint hooks exist only in the
isolated runner process; hooks are restored in a finally block.

Before/after evaluation covers all 256 questions with greedy and four samples
per question (1,024 sampled completions per stage), seed 0, max 96 new tokens.
Greedy uses top-p 1/top-k 0; these filters do not change argmax compared with
the notebook's defaults. Left padding and actual context <=340 match the native
path. Eight general prompts are also generated before/after. Report correctness,
format failures, budget exhaustion, gold-number appearance without correct final
answer, raw examples and qualitative general behavior. Do not use the test set
to select checkpoints or retune settings. Last epoch always supplies the result.
