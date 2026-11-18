"""Notebook-derived word-problem experiment using the native GRPO trainer."""
import argparse
import collections
import json
import math
import platform
import random
import shutil
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch
import transformers
from safetensors.torch import load_model
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

PROJECT = Path(__file__).resolve().parents[2]
ROOT = PROJECT.parents[1]
sys.path.insert(0, str(PROJECT))
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from dataset.gsm8k_dataset import GSM8KDataset
from trainer import train_grpo
from trainer.gsm8k_reward import score_gsm8k_response
from trainer.rollout_engine import TorchRolloutEngine
from trainer.trainer_utils import SkipBatchSampler, setup_seed
from grpo_arithmetic import sha, statehash, dump, jsonl

REPO = 'caspar/MiniMind-Dense-64M-Mixed-SFT-v1.0'
REVISION = '7597c22aa4d3e7b2ac824b4b4be01158cccf1b8d'
G = 6
MAX_NEW = 96
CONTROLS = ['Why is the sky blue?', 'What is the capital of France?', 'Explain photosynthesis briefly.', 'Write a Python function that adds two numbers.', 'Write a Python function that reverses a string.', 'What is the opposite of hot?', 'Name three common fruits.', 'Give two ways to save water at home.']


def make_data():
    rng = random.Random(0)
    names = ['Tom', 'Anna', 'Ben', 'Lily', 'Sam', 'Mia', 'Leo', 'Emma']
    items = ['apples', 'pencils', 'cookies', 'books', 'marbles', 'stickers', 'cards', 'balloons']
    rows, seen = [], set()
    while len(rows) < 384:
        kind = ('add', 'mul')[len(rows) % 2]
        name, item = rng.choice(names), rng.choice(items)
        a, b = rng.randint(2, 9), rng.randint(2, 9)
        if kind == 'add':
            q = f'{name} has {a} {item}. {name} gets {b} more {item}. How many {item} does {name} have now?'
            answer = f'{name} has {a} + {b} = {a + b} {item} now.\n#### {a + b}'
        else:
            q = f'{name} buys {a} boxes. Each box has {b} {item}. How many {item} does {name} buy in total?'
            answer = f'{name} buys {a} * {b} = {a * b} {item} in total.\n#### {a * b}'
        if q in seen:
            continue
        seen.add(q)
        rows.append(dict(question=q, answer=answer, kind=kind, a=a, b=b))
    train, test = rows[:128], rows[128:]
    assert not {r['question'] for r in train} & {r['question'] for r in test}
    assert collections.Counter(r['kind'] for r in train) == {'add': 64, 'mul': 64}
    assert collections.Counter(r['kind'] for r in test) == {'add': 128, 'mul': 128}
    return train, test


def inputs(tok, samples, generations=1):
    x = tok([s['prompt'] for s in samples], return_tensors='pt', padding=True, padding_side='left', add_special_tokens=False, return_token_type_ids=False).to('cuda')
    return x['input_ids'].repeat_interleave(generations, 0), x['attention_mask'].repeat_interleave(generations, 0)


def completion_records(tok, y, prompt_len, samples, rows, generations):
    records = []
    for i, ids in enumerate(y[:, prompt_len:].tolist()):
        ended = tok.eos_token_id in ids
        if ended:
            ids = ids[:ids.index(tok.eos_token_id)]
        text = tok.decode(ids, skip_special_tokens=True)
        row, sample = rows[i // generations], samples[i // generations]
        reward, reason = score_gsm8k_response(text, sample['answer'])
        gold = int(sample['answer'].split('####')[1].strip())
        import re
        numbers = re.findall(r'\d+', text)
        records.append(dict(question=row['question'], kind=row['kind'], a=row['a'], b=row['b'], gold=gold, sample=i % generations, generated=text, reward=reward, reason=reason, tokens=len(ids), eos=ended, gold_in_text=str(gold) in numbers))
    return records


def stats(records, generations):
    reasons = collections.Counter(r['reason'] for r in records)
    result = dict(n=len(records), correct=int(sum(r['reward'] for r in records)), accuracy=sum(r['reward'] for r in records)/len(records), reasons={k: reasons[k] for k in ['correct', 'wrong_number', 'format_error']}, mean_tokens=sum(r['tokens'] for r in records)/len(records), budget_exhausted=sum(not r['eos'] for r in records), gold_present_but_final_wrong=sum(r['gold_in_text'] and not r['reward'] for r in records), by_kind={kind: dict(n=sum(r['kind'] == kind for r in records), correct=int(sum(r['reward'] for r in records if r['kind'] == kind))) for kind in ['add', 'mul']})
    if generations > 1:
        grouped = collections.defaultdict(list)
        for r in records:
            grouped[r['question']].append(r['reward'])
        result.update(groups=len(grouped), mixed_groups=sum(max(rs) != min(rs) for rs in grouped.values()), groups_any_correct=sum(max(rs) > 0 for rs in grouped.values()))
    return result


@torch.no_grad()
def evaluate(model, tok, ds, rows, out, label, samples=4):
    model.eval()
    setup_seed(0)
    results = {'greedy': [], 'sampled': []}
    for start in range(0, len(ds), 16):
        part = ds.samples[start:start + 16]
        part_rows = rows[start:start + 16]
        for mode, g, sampling in [('greedy', 1, False), ('sampled', samples, True)]:
            ids, mask = inputs(tok, part, g)
            assert ids.shape[1] + MAX_NEW <= 340
            with torch.autocast('cuda', dtype=torch.bfloat16):
                y = model.generate(ids, attention_mask=mask, max_new_tokens=MAX_NEW, do_sample=sampling, temperature=.8 if sampling else 1., top_p=.85 if sampling else 1., top_k=50 if sampling else 0, repetition_penalty=1., eos_token_id=tok.eos_token_id)
            results[mode].extend(completion_records(tok, y, ids.shape[1], part, part_rows, g))
    summary = {}
    for mode, g in [('greedy', 1), ('sampled', samples)]:
        jsonl(out / f'{label}_{mode}.jsonl', results[mode])
        summary[mode] = stats(results[mode], g)
    print(label, json.dumps(summary), flush=True)
    return summary


@torch.no_grad()
def controls(model, tok, out, label):
    model.eval()
    records = []
    for question in CONTROLS:
        text = tok.apply_chat_template([dict(role='user', content=question)], tokenize=False, add_generation_prompt=True, open_thinking=False).replace('<think>\n\n</think>\n\n', '')
        x = tok(text, return_tensors='pt', add_special_tokens=False).input_ids.cuda()
        with torch.autocast('cuda', dtype=torch.bfloat16):
            y = model.generate(x, max_new_tokens=96, do_sample=False, temperature=1., top_p=1., top_k=0, repetition_penalty=1., eos_token_id=tok.eos_token_id)
        ids = y[0, x.shape[1]:].tolist()
        ended = tok.eos_token_id in ids
        if ended:
            ids = ids[:ids.index(tok.eos_token_id)]
        records.append(dict(question=question, generated=tok.decode(ids, skip_special_tokens=True), tokens=len(ids), eos=ended))
    jsonl(out / f'{label}_controls.jsonl', records)


class CheckedAdamW(torch.optim.AdamW):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.steps = 0
        self.last_grad_norm = None

    def step(self, closure=None):
        grads = [p.grad.detach().float() for group in self.param_groups for p in group['params'] if p.grad is not None]
        norm = torch.stack([g.norm() for g in grads]).norm()
        assert torch.isfinite(norm)
        self.last_grad_norm = norm.item()
        self.steps += 1
        return super().step(closure)


class LoggedRollout(TorchRolloutEngine):
    def __init__(self, *args, output, **kwargs):
        super().__init__(*args, **kwargs)
        self.output = output
        self.calls = 0
        self.first_check = None

    def rollout(self, **kwargs):
        result = super().rollout(**kwargs)
        self.calls += 1
        result.output_ids = result.output_ids.clone()
        result.completion_ids = result.completion_ids.clone()
        assert not result.output_ids.is_inference()
        if self.calls == 1:
            ids = result.output_ids
            n = result.completion_ids.shape[1]
            prompt_len = ids.shape[1] - n
            full_mask = torch.cat([kwargs['attention_mask'].repeat_interleave(G, 0), torch.ones_like(result.completion_ids)], 1)
            with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                res = self.policy_model(ids, attention_mask=full_mask)
                current = res.logits[:, :-1].log_softmax(-1).gather(-1, ids[:, 1:, None]).squeeze(-1)[:, prompt_len-1:]
            eos = result.completion_ids.eq(self.tokenizer.eos_token_id)
            valid = (eos.cumsum(1) - eos.long()).eq(0)
            diff = (current - result.per_token_logps).float()
            error = (diff.exp() - 1).abs()[valid]
            self.first_check = dict(max_logprob_difference=diff.abs()[valid].max().item(), ratio_max_error=error.max().item(), ratio_mean_error=error.mean().item())
            with torch.no_grad():
                ref_logits = self.reference(ids, attention_mask=full_mask).logits[:, :-1]
                ref_logps = ref_logits.log_softmax(-1).gather(-1, ids[:, 1:, None]).squeeze(-1)[:, prompt_len-1:]
            d = ref_logps - current.float()
            initial_kl = (d.exp() - d - 1)[valid].mean().item()
            self.first_check['initial_estimator_kl'] = initial_kl
            self.first_check['reference_forward_precision'] = 'FP32, as native trainer; policy BF16 autocast'
            dump(self.output / 'initial_ratio_check.json', self.first_check)
            assert self.first_check['ratio_max_error'] < .15, self.first_check
            assert initial_kl < .01, self.first_check
        return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    assert torch.cuda.is_available()
    torch.set_num_threads(4)
    setup_seed(42)
    shutil.copy2(Path(__file__), out / 'executed_grpo_word_problems.py')
    train_rows, test_rows = make_data()
    jsonl(out / 'train.jsonl', train_rows)
    jsonl(out / 'test.jsonl', test_rows)
    tok = AutoTokenizer.from_pretrained(args.base, local_files_only=True)
    local_tok = AutoTokenizer.from_pretrained(PROJECT / 'model', local_files_only=True)
    ds = {s: GSM8KDataset(str(out / f'{s}.jsonl'), tok, max_length=256) for s in ['train', 'test']}
    token_lengths = [len(tok(s['prompt'], add_special_tokens=False).input_ids) for d in ds.values() for s in d.samples]
    assert max(token_lengths) + MAX_NEW <= 340
    assert len(ds['train']) == 128 and len(ds['test']) == 256
    cfg = MiniMindConfig(**json.loads((args.base / 'config.json').read_text()))
    policy, reference = MiniMindForCausalLM(cfg), MiniMindForCausalLM(cfg)
    for model in (policy, reference):
        missing, unexpected = load_model(model, args.base / 'model.safetensors', strict=True)
        assert not missing and not unexpected
        model.cuda().eval()
    reference.requires_grad_(False)
    assert statehash(policy) == statehash(reference)
    ref_hash = statehash(reference)
    notebook = ROOT / 'notebooks/minimind/minimind_grpo_tour.py'
    meta = dict(repo=REPO, revision=REVISION, base_sha256=sha(args.base / 'model.safetensors'), python=platform.python_version(), torch=torch.__version__, transformers=transformers.__version__, cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0), parameters=sum(p.numel() for p in policy.parameters()), seed=42, data_seed=0, epochs=2, batch_prompts=4, generations=G, learning_rate=3e-7, cosine_eta_min=3e-8, beta=.1, epsilon=.2, loss_type='grpo', weight_decay=.01, optimizer='AdamW defaults, betas=(0.9,0.999)', max_new_tokens=96, max_prompt_tokens=max(token_lengths), max_seq_len=256, precision='FP32 weights/optimizer, BF16 autocast including native log_softmax', sampling=dict(temperature=.8, top_p=.85, top_k=50), notebook_sha256=sha(notebook), scenario_sha256=sha(ROOT / 'test_scenarios/minimind/grpo-word-problems-tour.md'), source_sha256=sha(Path(__file__)), data_sha256={s: sha(out / f'{s}.jsonl') for s in ds}, tokenizer_local_vocab_equal=tok.get_vocab() == local_tok.get_vocab(), tokenizer_local_template_equal=tok.chat_template == local_tok.chat_template, train_question_disjoint=True, expression_disjoint=False, separate_sft=False, checkpoint_selection='last epoch')
    dump(out / 'metadata.json', meta)
    # Pilot is the first eight per operation in original data order, not reward-selected.
    pilot_indices = sorted(sum([[i for i, r in enumerate(train_rows) if r['kind'] == kind][:8] for kind in ['add', 'mul']], []))
    setup_seed(0)
    pilot_records = []
    for start in range(0, len(pilot_indices), 4):
        indices = pilot_indices[start:start+4]
        samples = [ds['train'][i] for i in indices]
        ids, mask = inputs(tok, samples, G)
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            y = policy.generate(ids, attention_mask=mask, max_new_tokens=96, do_sample=True, temperature=.8, top_p=.85, top_k=50, repetition_penalty=1., eos_token_id=tok.eos_token_id)
        pilot_records.extend(completion_records(tok, y, ids.shape[1], samples, [train_rows[i] for i in indices], G))
    jsonl(out / 'pilot.jsonl', pilot_records)
    summary = dict(pilot=stats(pilot_records, G))
    summary['pilot_passed'] = summary['pilot']['correct'] >= 1 and summary['pilot']['mixed_groups'] >= 4
    dump(out / 'summary.json', summary)
    print('pilot', json.dumps(summary), flush=True)
    if not summary['pilot_passed']:
        summary['training_started'] = False
        dump(out / 'summary.json', summary)
        print('PILOT_GATE_NOT_PASSED', flush=True)
        return
    summary['before'] = evaluate(reference, tok, ds['test'], test_rows, out, 'before')
    controls(reference, tok, out, 'before')
    optimizer = CheckedAdamW(policy.parameters(), lr=3e-7)
    scheduler = CosineAnnealingLR(optimizer, T_max=64, eta_min=3e-8)
    engine = LoggedRollout(policy, tok, 'cuda', torch.autocast('cuda', dtype=torch.bfloat16), output=out)
    engine.reference = reference
    trainer_args = SimpleNamespace(device='cuda', dtype='bfloat16', epochs=2, batch_size=4, learning_rate=3e-7, accumulation_steps=1, grad_clip=1., log_interval=1, save_interval=10**9, save_dir=str(out / 'native'), save_weight='grpo', max_seq_len=256, max_gen_len=96, num_generations=G, beta=.1, loss_type='grpo', epsilon=.2, epsilon_high=5., reward_type='gsm8k', debug_mode=False, debug_interval=20)
    Path(trainer_args.save_dir).mkdir()
    vars(train_grpo).update(args=trainer_args, model=policy, tokenizer=tok, lm_config=cfg, autocast_ctx=torch.autocast('cuda', dtype=torch.bfloat16), optimizer=optimizer, scheduler=scheduler)
    original_checkpoint = train_grpo.lm_checkpoint
    def redirected_checkpoint(*a, **kw):
        kw['save_dir'] = str(out / 'resume')
        return original_checkpoint(*a, **kw)
    train_grpo.lm_checkpoint = redirected_checkpoint
    history = []
    # The trainer calls rollout then reward calculation; intercept only to retain texts/gold.
    original_rewards = train_grpo.calculate_gsm8k_rewards_and_stats
    def logged_rewards(result, answers, *a, **kw):
        rewards, metrics = original_rewards(result, answers, *a, **kw)
        with (out / 'training_samples.jsonl').open('a') as f:
            for i, text in enumerate(result.completions):
                f.write(json.dumps(dict(step=engine.calls, sample=i, gold=answers[i//G], generated=text, reward=rewards[i].item())) + '\n')
        return rewards, metrics
    train_grpo.calculate_gsm8k_rewards_and_stats = logged_rewards
    def record(values):
        assert all(math.isfinite(float(v)) for v in values.values())
        assert all(p.grad is None for p in reference.parameters())
        item = dict(step=optimizer.steps, gradient_norm_after_clip=optimizer.last_grad_norm, **values)
        history.append(item)
        with (out / 'training.jsonl').open('a') as f:
            f.write(json.dumps(item) + '\n')
    started = time.monotonic()
    try:
        for epoch in range(2):
            setup_seed(42 + epoch)
            indices = torch.randperm(128).tolist()
            loader = DataLoader(ds['train'], batch_sampler=SkipBatchSampler(indices, 4, 0))
            train_grpo.grpo_train_epoch(epoch, loader, len(loader), engine, reference, None, 0, SimpleNamespace(log=record))
    finally:
        train_grpo.lm_checkpoint = original_checkpoint
        train_grpo.calculate_gsm8k_rewards_and_stats = original_rewards
    meta['training_seconds_including_checkpoint_saves'] = time.monotonic() - started
    assert optimizer.steps == engine.calls == len(history) == 64
    assert all(torch.isfinite(p).all().item() for p in policy.parameters())
    assert statehash(reference) == ref_hash
    changed = sum((p.detach() != r).sum().item() for p, r in zip(policy.parameters(), reference.parameters()))
    delta = math.sqrt(sum((p.detach() - r).square().sum().item() for p, r in zip(policy.parameters(), reference.parameters())))
    assert changed > 0
    torch.save({k: v.detach().cpu() for k, v in policy.state_dict().items()}, out / 'grpo_word_problems_fp32_768.pth')
    saved = torch.load(out / 'grpo_word_problems_fp32_768.pth', map_location='cpu', weights_only=True)
    assert all(torch.equal(saved[k], v.cpu()) for k, v in policy.state_dict().items())
    fixed, mask = inputs(tok, ds['test'].samples[:4])
    policy.eval()
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        logits = policy(fixed, attention_mask=mask).logits.clone()
    policy.load_state_dict(saved, strict=True)
    del saved
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        assert torch.equal(logits, policy(fixed, attention_mask=mask).logits)
    meta.update(optimizer_steps=64, reference_unchanged=True, finite_weights=True, changed_parameter_elements=changed, parameter_delta_l2=delta, strict_reload_weights_and_logits_equal=True, checkpoint_sha256=sha(out / 'grpo_word_problems_fp32_768.pth'), peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(), initial_ratio=engine.first_check)
    dump(out / 'metadata.json', meta)
    summary['after'] = evaluate(policy, tok, ds['test'], test_rows, out, 'after')
    controls(policy, tok, out, 'after')
    summary.update(training_started=True, training_completed=True, training=dict(steps=64, rollout_count=1536, mean_reward=sum(h['reward'] for h in history)/64, mean_mixed_group_rate=sum(h['reward/mixed_group_rate'] for h in history)/64, nonzero_gradient_steps=sum(h['gradient_norm_after_clip'] > 0 for h in history)))
    dump(out / 'summary.json', summary)
    print('DONE', json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
