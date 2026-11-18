"""Small on-policy GRPO on single-digit addition/nonnegative subtraction."""
import argparse
import collections
import hashlib
import json
import math
import platform
import random
import re
import shutil
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import transformers
from safetensors.torch import load_model
from transformers import AutoTokenizer

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM

REPO = 'caspar/MiniMind-Dense-64M-Mixed-SFT-v1.0'
REVISION = '7597c22aa4d3e7b2ac824b4b4be01158cccf1b8d'
G = 8
MAX_NEW = 32


def dump(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')


def jsonl(path, rows):
    path.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def statehash(model):
    h = hashlib.sha256()
    for k, v in model.state_dict().items():
        h.update(k.encode())
        h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def make_data():
    rng = random.Random(42)
    splits = {s: [] for s in ['train', 'validation', 'test']}
    for op in ['+', '-']:
        rows = []
        for a in range(10):
            for b in range(10):
                if (op == '+' and a > b) or (op == '-' and a < b):
                    continue
                if (op, a, b) in [('+', 1, 1), ('-', 4, 1)]:
                    continue
                answer = a + b if op == '+' else a - b
                # Fixed two-shot examples; their facts are excluded from all splits.
                q = f'Calculate the last expression. Reply with only the integer answer.\nExamples:\n1 + 1 = 2\n4 - 1 = 3\nProblem: {a} {op} {b} = ?'
                rows.append(dict(a=a, b=b, operation=op, answer=answer, question=q, key=f'{a}{op}{b}'))
        assert len(rows) == 54
        rng.shuffle(rows)
        splits['train'].extend(rows[:32])
        splits['validation'].extend(rows[32:38])
        splits['test'].extend(rows[38:])
    for rows in splits.values():
        rng.shuffle(rows)
    assert [len(rows) for rows in splits.values()] == [64, 12, 32]
    assert len({r['key'] for rows in splits.values() for r in rows}) == 108
    return splits


def parse_integer(text):
    text = text.strip()
    return int(text) if re.fullmatch(r'[+-]?[0-9]+', text) else None


def score(text, expected):
    value = parse_integer(text)
    return float(value == expected) if value is not None else 0.


def render(tok, question):
    return tok.apply_chat_template([dict(role='user', content=question)], tokenize=False, add_generation_prompt=True, open_thinking=False).replace('<think>\n\n</think>\n\n', '')


def encode(tok, rows, answer_prefix=''):
    return [dict(row, ids=tok(render(tok, row['question']) + answer_prefix, add_special_tokens=False).input_ids) for row in rows]


def equal_length_batches(rows, size=2, seed=None):
    groups = collections.defaultdict(list)
    for row in rows:
        groups[len(row['ids'])].append(row)
    rng = random.Random(seed)
    batches = []
    for group in groups.values():
        if seed is not None:
            rng.shuffle(group)
        batches.extend(group[i:i+size] for i in range(0, len(group), size))
    if seed is not None:
        rng.shuffle(batches)
    return batches


def completion_mask(ids, eos):
    is_eos = ids.eq(eos)
    return (is_eos.cumsum(1) - is_eos.long()).eq(0)


@torch.no_grad()
def rollout(model, tok, rows, generations, sampling):
    model.eval()
    x = torch.tensor([row['ids'] for row in rows], device='cuda').repeat_interleave(generations, 0)
    assert x.shape[1] + MAX_NEW < 340
    with torch.autocast('cuda', dtype=torch.bfloat16):
        y = model.generate(x, max_new_tokens=MAX_NEW, do_sample=sampling, temperature=1., top_p=1., top_k=0, repetition_penalty=1., eos_token_id=tok.eos_token_id, logits_to_keep=1).clone()
    assert not y.is_inference()
    comp = y[:, x.shape[1]:]
    mask = completion_mask(comp, tok.eos_token_id)
    texts = [tok.decode(tokens[m].tolist(), skip_special_tokens=True) for tokens, m in zip(comp, mask)]
    return y, comp, mask, texts


def logps(model, ids, count):
    with torch.autocast('cuda', dtype=torch.bfloat16):
        logits = model(ids, logits_to_keep=count + 1, use_cache=False).logits[:, :-1]
    return logits.float().log_softmax(-1).gather(-1, ids[:, -count:].unsqueeze(-1)).squeeze(-1)


@torch.no_grad()
def evaluate(model, tok, rows, path, sampling=False, generations=1):
    results = []
    for batch in equal_length_batches(rows, size=8 if not sampling else 2):
        _, comp, mask, texts = rollout(model, tok, batch, generations, sampling)
        for i, text in enumerate(texts):
            row = batch[i // generations]
            nums = re.findall(r'[+-]?[0-9]+', text)
            results.append(dict(key=row['key'], question=row['question'], operation=row['operation'], answer=row['answer'], sample=i % generations, generated=text, reward=score(text, row['answer']), format_valid=parse_integer(text) is not None, last_number_correct=bool(nums and int(nums[-1]) == row['answer']), tokens=int(mask[i].sum()), eos=bool(((comp[i] == tok.eos_token_id) & mask[i]).any())))
    jsonl(path, results)
    summary = dict(n=len(results), strict_correct=sum(r['reward'] for r in results), strict_accuracy=sum(r['reward'] for r in results)/len(results), format_valid=sum(r['format_valid'] for r in results), last_number_correct=sum(r['last_number_correct'] for r in results), mean_tokens=sum(r['tokens'] for r in results)/len(results), budget_exhausted=sum(not r['eos'] for r in results), by_operation={op: dict(n=sum(r['operation'] == op for r in results), correct=sum(r['reward'] for r in results if r['operation'] == op)) for op in ['+', '-']})
    if generations > 1:
        grouped = collections.defaultdict(list)
        for r in results:
            grouped[r['key']].append(r['reward'])
        summary.update(groups=len(grouped), groups_with_any_correct=sum(max(rs) > 0 for rs in grouped.values()), mixed_reward_groups=sum(max(rs) != min(rs) for rs in grouped.values()))
    print(path.name, json.dumps(summary), flush=True)
    return summary


@torch.no_grad()
def controls(model, tok, path):
    prompts = ['What is the capital of France?', 'Explain photosynthesis briefly.', 'Write a Python function that adds two numbers.', 'What is the opposite of hot?', 'Name three common fruits.', 'Give two ways to save water at home.']
    result = []
    for q in prompts:
        x = torch.tensor([tok(render(tok, q), add_special_tokens=False).input_ids], device='cuda')
        with torch.autocast('cuda', dtype=torch.bfloat16):
            y = model.generate(x, max_new_tokens=96, do_sample=False, temperature=1., top_p=1., top_k=0, repetition_penalty=1., eos_token_id=tok.eos_token_id, logits_to_keep=1)
        ids = y[0, x.shape[1]:].tolist()
        ended = tok.eos_token_id in ids
        if ended:
            ids = ids[:ids.index(tok.eos_token_id)]
        result.append(dict(question=q, generated=tok.decode(ids, skip_special_tokens=True), tokens=len(ids), eos=ended))
    jsonl(path, result)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--answer-prefix', default='')
    parser.add_argument('--pilot-only', action='store_true')
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    assert torch.cuda.is_available()
    torch.set_num_threads(4)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    random.seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    shutil.copy2(Path(__file__), out / 'executed_grpo_arithmetic.py')
    splits = make_data()
    for split, rows in splits.items():
        jsonl(out / f'{split}.jsonl', rows)
    tok = AutoTokenizer.from_pretrained(args.base, local_files_only=True)
    cfg = MiniMindConfig(**json.loads((args.base / 'config.json').read_text()))
    policy, reference = MiniMindForCausalLM(cfg), MiniMindForCausalLM(cfg)
    for model in (policy, reference):
        missing, unexpected = load_model(model, args.base / 'model.safetensors', strict=True)
        assert not missing and not unexpected
        model.cuda().eval()
    reference.requires_grad_(False)
    assert statehash(policy) == statehash(reference)
    assert all(p.data_ptr() != r.data_ptr() for p, r in zip(policy.parameters(), reference.parameters()))
    ref_hash = statehash(reference)
    encoded = {s: encode(tok, rows, args.answer_prefix) for s, rows in splits.items()}
    meta = dict(repo=REPO, revision=REVISION, base_sha256=sha(args.base / 'model.safetensors'), python=platform.python_version(), torch=torch.__version__, transformers=transformers.__version__, cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0), parameters=sum(p.numel() for p in policy.parameters()), trainable_parameters=sum(p.numel() for p in policy.parameters() if p.requires_grad), seed=42, epochs=2, batch_prompts=2, generations=G, max_new_tokens=MAX_NEW, learning_rate=1e-6, beta=.04, epsilon=.2, grad_clip=1., weight_decay=0., optimizer='AdamW betas=(0.9,0.999)', precision='FP32 weights/optimizer/log_softmax, BF16 autocast', sampling=dict(temperature=1., top_p=1., top_k=0), loss='GRPO clipped surrogate + exp KL estimator, completion-token mean per response, one update per fresh rollout', old_logps='same current-policy forward detached, one on-policy update, no reuse', reward='1 for full-string integer equal to arithmetic gold, else 0; no format or length bonus', examples=['1 + 1 = 2', '4 - 1 = 3'], examples_excluded_from_splits=True, sft_warmup=False, checkpoint_selection='last epoch; validation diagnostic only', data_sha256={s: sha(out / f'{s}.jsonl') for s in splits}, source_sha256=sha(Path(__file__)))
    dump(out / 'metadata.json', meta)
    meta.update(answer_prefix=args.answer_prefix, pilot_only=args.pilot_only)
    dump(out / 'metadata.json', meta)
    # Reward and EOS checks guard enumeration, number substrings, and post-EOS leakage.
    assert score(' 7\n', 7) == 1 and score('17', 7) == 0
    assert all(score(t, 7) == 0 for t in ['7 or 8', 'The answer is 7.', '', 'NaN', '7 7'])
    assert completion_mask(torch.tensor([[3, 2, 2, 2], [3, 4, 5, 6]]), 2).tolist() == [[True, True, False, False], [True, True, True, True]]
    summary = dict(before={}, history=[])
    if args.pilot_only:
        by_op = [row for op in ['+', '-'] for row in encoded['train'] if row['operation'] == op]
        probe = by_op[:6] + by_op[32:38]
        torch.manual_seed(100)
        torch.cuda.manual_seed_all(100)
        summary['pilot'] = evaluate(policy, tok, probe, out / 'pilot_train_samples.jsonl', sampling=True, generations=G)
        dump(out / 'summary.json', summary)
        print('PILOT_DONE', json.dumps(summary['pilot']), flush=True)
        return
    for split, rows in encoded.items():
        summary['before'][split] = evaluate(reference, tok, rows, out / f'before_{split}.jsonl')
    controls(reference, tok, out / 'before_controls.jsonl')
    # Probe only fixed training questions; never use held-out rewards to shape training.
    probe = [row for op in ['+', '-'] for row in encoded['train'] if row['operation'] == op]
    probe = probe[:6] + probe[32:38]
    assert len(probe) == 12
    torch.manual_seed(100)
    torch.cuda.manual_seed_all(100)
    summary['pilot'] = evaluate(policy, tok, probe, out / 'pilot_train_samples.jsonl', sampling=True, generations=G)
    dump(out / 'summary.json', summary)
    print('pilot_signal', summary['pilot']['mixed_reward_groups'], flush=True)
    # Keep the declared experiment even if reward is sparse; report zero-signal groups.
    torch.manual_seed(43)
    torch.cuda.manual_seed_all(43)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-6, weight_decay=0.)
    started, step = time.monotonic(), 0
    for epoch in [1, 2]:
        for batch in equal_length_batches(encoded['train'], size=2, seed=42+epoch):
            ids, comp, mask, texts = rollout(policy, tok, batch, G, True)
            rewards = torch.tensor([score(text, batch[i // G]['answer']) for i, text in enumerate(texts)], device='cuda')
            groups = rewards.view(-1, G)
            advantages = ((groups - groups.mean(1, keepdim=True)) / (groups.std(1, unbiased=False, keepdim=True) + 1e-4)).flatten()
            assert torch.allclose(advantages.view(-1, G).mean(1), torch.zeros(len(batch), device='cuda'), atol=1e-5)
            optimizer.zero_grad(set_to_none=True)
            # Dropout is zero; eval() does not disable gradients.
            policy.eval()
            p = logps(policy, ids, comp.shape[1])
            old = p.detach()
            with torch.no_grad():
                r = logps(reference, ids, comp.shape[1])
            ratio = (p - old).exp()
            assert (ratio - 1).abs().max().item() < 1e-6
            kl_diff = r - p
            kl = kl_diff.exp() - kl_diff - 1.
            if step == 0:
                assert kl.abs().max().item() < 1e-6
            clipped = ratio.clamp(.8, 1.2)
            task = torch.minimum(ratio * advantages[:, None], clipped * advantages[:, None])
            loss = ((-task + .04 * kl) * mask).sum(1).div(mask.sum(1).clamp_min(1)).mean()
            assert torch.isfinite(loss)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.)
            assert torch.isfinite(norm)
            optimizer.step()
            assert all(p.grad is None for p in reference.parameters())
            step += 1
            record = dict(epoch=epoch, step=step, prompts=len(batch), sampled_answers=len(texts), reward_mean=rewards.mean().item(), correct=int(rewards.sum()), mixed_groups=int((groups.max(1).values != groups.min(1).values).sum()), mean_kl=(kl * mask).sum().item()/mask.sum().item(), loss=loss.item(), grad_norm=norm.item(), tokens_mean=mask.sum(1).float().mean().item(), budget_exhausted=int((~((comp == tok.eos_token_id) & mask).any(1)).sum()), ratio_max_error=(ratio-1).abs().max().item())
            with (out / 'training.jsonl').open('a') as f:
                f.write(json.dumps(record) + '\n')
            with (out / 'training_samples.jsonl').open('a') as f:
                for i, text in enumerate(texts):
                    f.write(json.dumps(dict(step=step, key=batch[i//G]['key'], expected=batch[i//G]['answer'], generated=text, reward=rewards[i].item())) + '\n')
            if step % 8 == 0:
                print('training', json.dumps(record), flush=True)
        val = evaluate(policy, tok, encoded['validation'], out / f'epoch{epoch}_validation.jsonl')
        summary['history'].append(dict(epoch=epoch, validation=val))
        dump(out / 'summary.json', summary)
    meta['training_seconds_including_validation'] = time.monotonic() - started
    assert all(torch.isfinite(p).all().item() for p in policy.parameters())
    assert statehash(reference) == ref_hash
    changed = sum((p.detach() != r).sum().item() for p, r in zip(policy.parameters(), reference.parameters()))
    delta = math.sqrt(sum((p.detach() - r).square().sum().item() for p, r in zip(policy.parameters(), reference.parameters())))
    torch.save({k: v.detach().cpu() for k, v in policy.state_dict().items()}, out / 'grpo_arithmetic_768.pth')
    saved = torch.load(out / 'grpo_arithmetic_768.pth', map_location='cpu', weights_only=True)
    assert all(torch.equal(saved[k], v.cpu()) for k, v in policy.state_dict().items())
    fixed_batch = equal_length_batches(encoded['test'], 2)[0]
    fixed_ids = torch.tensor([row['ids'] for row in fixed_batch], device='cuda')
    with torch.no_grad():
        logits_before = policy(fixed_ids).logits.clone()
    policy.load_state_dict(saved, strict=True)
    del saved
    with torch.no_grad():
        assert torch.equal(logits_before, policy(fixed_ids).logits)
    meta.update(updates=step, reference_unchanged=True, finite_weights=True, strict_reload_weights_and_logits_equal=True, changed_parameter_elements=changed, parameter_delta_l2=delta, checkpoint_sha256=sha(out / 'grpo_arithmetic_768.pth'), peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated())
    dump(out / 'metadata.json', meta)
    summary['after'] = {}
    for split, rows in encoded.items():
        summary['after'][split] = evaluate(policy, tok, rows, out / f'after_{split}.jsonl')
    # Same fixed probe and RNG seed provide a secondary sampled comparison.
    torch.manual_seed(100)
    torch.cuda.manual_seed_all(100)
    summary['after_pilot'] = evaluate(policy, tok, probe, out / 'after_train_probe_samples.jsonl', sampling=True, generations=G)
    controls(policy, tok, out / 'after_controls.jsonl')
    records = [json.loads(line) for line in (out / 'training.jsonl').read_text().splitlines()]
    summary['training'] = dict(updates=step, rollout_count=sum(r['sampled_answers'] for r in records), correct_rollouts=sum(r['correct'] for r in records), prompt_groups=sum(r['prompts'] for r in records), mixed_groups=sum(r['mixed_groups'] for r in records), nonzero_gradient_updates=sum(r['grad_norm'] > 0 for r in records))
    summary['test_improved'] = summary['after']['test']['strict_correct'] > summary['before']['test']['strict_correct']
    dump(out / 'summary.json', summary)
    print('DONE', json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
