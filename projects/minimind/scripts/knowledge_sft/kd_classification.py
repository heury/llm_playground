"""Label-scarce emotion classification: MoE teacher -> Dense student knowledge distillation.

Scenario: test_scenarios/minimind/kd-label-scarce-classification.md
Run from projects/minimind/ with this project's .venv. CUDA required. No remote code execution.
"""
import argparse
import hashlib
import json
import math
import random
import sys
import time
from pathlib import Path

import datasets
import torch
import torch.nn.functional as F
import transformers
from safetensors.torch import load_model
from transformers import AutoTokenizer

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM  # noqa: E402

TEACHER_REPO, TEACHER_REV = 'caspar/MiniMind-MoE-198M-A64M-Mixed-SFT-v1.0', 'a742772a0ad4db7480064bd2abccc174062a3de8'
STUDENT_REPO, STUDENT_REV = 'caspar/MiniMind-Dense-64M-Mixed-SFT-v1.0', '7597c22aa4d3e7b2ac824b4b4be01158cccf1b8d'
DATA_REPO, DATA_REV = 'dair-ai/emotion', 'cab853a1dbdf4c42c2b3ef2173804746df8825fe'
LABELS = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']
PROMPT = ('Classify the emotion expressed in the text. Choose one of: sadness, joy, love, anger, fear, surprise.\n'
          'Text: {text}\nAnswer with one word.')
BATCH, LR, T, ALPHA = 16, 3e-5, 2.0, 0.5
TEACHER_STEPS, STUDENT_STEPS, SEEDS = 375, 250, [42, 43, 44]


def dump(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')


def jsonl(path, rows):
    path.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def statehash(model):
    h = hashlib.sha256()
    for k, v in sorted(model.state_dict().items()):
        h.update(k.encode())
        h.update(v.detach().float().cpu().numpy().tobytes())
    return h.hexdigest()


def snapshot(repo, rev):
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(repo, revision=rev, allow_patterns=['*.json', '*.safetensors', '*.jinja']))


def load(path):
    config = MiniMindConfig(**json.loads((path / 'config.json').read_text()))
    model = MiniMindForCausalLM(config)
    missing, unexpected = load_model(model, path / 'model.safetensors', strict=True)
    assert not missing and not unexpected, (missing, unexpected)
    return model.cuda()


def lr_at(step, total):  # same formula as trainer_utils.get_lr
    return LR * (0.1 + 0.45 * (1 + math.cos(math.pi * step / total)))


class Codec:
    def __init__(self, tok):
        self.tok = tok
        self.eos = tok.eos_token_id
        self.cands = [tok(w, add_special_tokens=False).input_ids + [self.eos] for w in LABELS]

    def render(self, messages, generation=False):
        return self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=generation,
                                            open_thinking=False).replace('<think>\n\n</think>\n\n', '')

    def prefix(self, text):
        return self.tok(self.render([dict(role='user', content=PROMPT.format(text=text))], True),
                        add_special_tokens=False).input_ids

    def example(self, text, label):
        p = self.prefix(text)
        ids = p + self.cands[label]
        full = self.tok(self.render([dict(role='user', content=PROMPT.format(text=text)),
                                     dict(role='assistant', content=LABELS[label])]), add_special_tokens=False).input_ids
        assert full[:len(ids)] == ids and len(full) == len(ids) + 1, 'chat template mismatch'
        assert len(ids) <= 340
        return dict(ids=ids, start=len(p))


@torch.no_grad()
def candidate_logps(model, codec, prefixes, bs=32):
    """Return [N, 6] sum log p(label tokens + EOS | prompt)."""
    model.eval()
    out = []
    pad = codec.tok.pad_token_id
    for o in range(0, len(prefixes), bs):
        seqs, meta = [], []
        for p in prefixes[o:o + bs]:
            for c in codec.cands:
                seqs.append(p + c)
                meta.append((len(p), len(c)))
        L = max(map(len, seqs))
        x = torch.tensor([s + [pad] * (L - len(s)) for s in seqs], device='cuda')
        with torch.autocast('cuda', dtype=torch.bfloat16):
            logits = model(x).logits
        logp = torch.log_softmax(logits.float(), -1)
        scores = []
        for i, (pl, cl) in enumerate(meta):
            pos = torch.arange(pl - 1, pl - 1 + cl, device='cuda')
            scores.append(logp[i, pos, x[i, pos + 1]].sum())
        out.append(torch.stack(scores).view(-1, len(LABELS)).cpu())
    return torch.cat(out)


def metrics(scores, gold, teacher_probs=None):
    probs = torch.softmax(scores, -1)
    pred = probs.argmax(-1)
    gold_t = torch.tensor(gold)
    acc = (pred == gold_t).float().mean().item()
    f1s = []
    for c in range(len(LABELS)):
        tp = ((pred == c) & (gold_t == c)).sum().item()
        fp = ((pred == c) & (gold_t != c)).sum().item()
        fn = ((pred != c) & (gold_t == c)).sum().item()
        f1s.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    conf = probs.max(-1).values
    correct = (pred == gold_t).float()
    ece = 0.0
    for b in range(15):
        m = (conf > b / 15) & (conf <= (b + 1) / 15)
        if m.any():
            ece += m.float().mean().item() * abs(conf[m].mean().item() - correct[m].mean().item())
    r = dict(accuracy=acc, macro_f1=sum(f1s) / len(f1s), per_class_f1=dict(zip(LABELS, f1s)),
             nll=-torch.log(probs[torch.arange(len(gold)), gold_t].clamp_min(1e-12)).mean().item(), ece=ece,
             pred_counts={LABELS[c]: int((pred == c).sum()) for c in range(len(LABELS))})
    if teacher_probs is not None:
        r['teacher_agreement'] = (pred == teacher_probs.argmax(-1)).float().mean().item()
        r['kl_teacher_student_6class'] = (teacher_probs * (teacher_probs.clamp_min(1e-12).log() - probs.clamp_min(1e-12).log())).sum(-1).mean().item()
    return r, probs


@torch.inference_mode()
def greedy_check(model, codec, rows):
    model.eval()
    groups = {}
    for i, r in enumerate(rows):
        groups.setdefault(len(r['prefix']), []).append(i)
    outs = [None] * len(rows)
    for idx in groups.values():
        for o in range(0, len(idx), 32):
            b = idx[o:o + 32]
            x = torch.tensor([rows[i]['prefix'] for i in b], device='cuda')
            with torch.autocast('cuda', dtype=torch.bfloat16):
                y = model.generate(inputs=x, max_new_tokens=8, do_sample=False, temperature=1., top_p=1., top_k=0,
                                   repetition_penalty=1., eos_token_id=codec.eos)
            for i, t in zip(b, y[:, x.shape[1]:].tolist()):
                if codec.eos in t:
                    t = t[:t.index(codec.eos)]
                outs[i] = codec.tok.decode(t, skip_special_tokens=True).strip()
    norm = [o.lower().strip(' .') for o in outs]
    valid = [n in LABELS for n in norm]
    correct = [n == LABELS[r['label']] for n, r in zip(norm, rows)]
    return dict(n=len(rows), valid_rate=sum(valid) / len(rows), accuracy=sum(correct) / len(rows)), outs


def train(model, items, steps, seed, teacher=None, log=None):
    """items: dict(ids, start, w_ce, w_kd). Loss on answer tokens (label + EOS)."""
    torch.manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    g = torch.Generator().manual_seed(seed)
    order, step, pad = [], 0, 0
    use_moe = model.config.use_moe
    rows = []
    t0 = time.monotonic()
    while step < steps:
        if len(order) < BATCH:
            order += torch.randperm(len(items), generator=g).tolist()
        batch = [items[i] for i in order[:BATCH]]
        order = order[BATCH:]
        L = max(len(b['ids']) for b in batch)
        x = torch.tensor([b['ids'] + [pad] * (L - len(b['ids'])) for b in batch], device='cuda')
        mask = torch.zeros(len(batch), L - 1, device='cuda')
        for i, b in enumerate(batch):
            mask[i, b['start'] - 1:len(b['ids']) - 1] = 1
        w_ce = torch.tensor([b['w_ce'] for b in batch], device='cuda')[:, None] * mask
        w_kd = torch.tensor([b['w_kd'] for b in batch], device='cuda')[:, None] * mask
        for pg in opt.param_groups:
            pg['lr'] = lr_at(step, steps)
        model.train()
        with torch.autocast('cuda', dtype=torch.bfloat16):
            res = model(x)
        s_logits = res.logits[:, :-1].float()
        ce = F.cross_entropy(s_logits.reshape(-1, s_logits.size(-1)), x[:, 1:].reshape(-1), reduction='none').view_as(mask)
        kd = torch.zeros_like(mask)
        if teacher is not None and w_kd.sum() > 0:
            with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                t_logits = teacher(x).logits[:, :-1].float()
            t_logp = F.log_softmax(t_logits / T, -1)
            s_logp = F.log_softmax(s_logits / T, -1)
            kd = (T ** 2) * (t_logp.exp() * (t_logp - s_logp)).sum(-1)
        n = mask.sum()
        loss = ((w_ce * ce).sum() + (w_kd * kd).sum()) / n
        if use_moe:
            loss = loss + res.aux_loss
        assert torch.isfinite(loss).item()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        assert torch.isfinite(gn).item()
        opt.step()
        step += 1
        if step % 25 == 0 or step == steps:
            r = dict(step=step, loss=loss.item(), ce=((ce * mask).sum() / n).item(), kd=((kd * mask).sum() / n).item(),
                     grad_norm=gn.item(), lr=lr_at(step - 1, steps), seconds=time.monotonic() - t0)
            rows.append(r)
            if log:
                print(log, json.dumps(r), flush=True)
    assert all(torch.isfinite(p).all().item() for p in model.parameters())
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    assert not args.output.exists(), 'output folder must not exist'
    args.output.mkdir(parents=True)
    assert torch.cuda.is_available()
    random.seed(42)
    torch.manual_seed(42)
    torch.set_num_threads(4)
    t_start = time.monotonic()

    s_path, t_path = snapshot(STUDENT_REPO, STUDENT_REV), snapshot(TEACHER_REPO, TEACHER_REV)
    tok = AutoTokenizer.from_pretrained(s_path)
    codec = Codec(tok)
    ds = datasets.load_dataset(DATA_REPO, 'split', revision=DATA_REV)
    assert ds['train'].features['label'].names == LABELS
    perm = list(range(len(ds['train'])))
    random.Random(42).shuffle(perm)
    split_idx = dict(teacher=perm[:2000], student=perm[2000:2200], unlabeled=perm[2200:4200])
    assert len(set(sum(split_idx.values(), []))) == 4200
    tr = ds['train']
    get = lambda idx: [dict(text=tr[i]['text'], label=tr[i]['label'], index=i) for i in idx]
    teacher_rows, student_rows, unl_rows = get(split_idx['teacher']), get(split_idx['student']), get(split_idx['unlabeled'])
    test_rows = [dict(text=r['text'], label=r['label'], index=i) for i, r in enumerate(ds['test'])]
    for r in test_rows + unl_rows:
        r['prefix'] = codec.prefix(r['text'])
    dump(args.output / 'splits.json', dict(seed=42, **split_idx))
    gold = [r['label'] for r in test_rows]
    test_prefixes = [r['prefix'] for r in test_rows]

    meta = dict(teacher=dict(repo=TEACHER_REPO, revision=TEACHER_REV, sha256=sha(t_path / 'model.safetensors')),
                student=dict(repo=STUDENT_REPO, revision=STUDENT_REV, sha256=sha(s_path / 'model.safetensors')),
                data=dict(repo=DATA_REPO, config='split', revision=DATA_REV), prompt=PROMPT, labels=LABELS,
                label_tokens=codec.cands, batch=BATCH, lr=LR, schedule='trainer_utils.get_lr cosine 0.1-1.0x',
                temperature=T, alpha=ALPHA, teacher_steps=TEACHER_STEPS, student_steps=STUDENT_STEPS, seeds=SEEDS,
                optimizer='AdamW wd 0.01, FP32 params, BF16 autocast, clip 1.0',
                torch=torch.__version__, transformers=transformers.__version__, datasets=datasets.__version__,
                device=torch.cuda.get_device_name(0))
    dump(args.output / 'metadata.json', meta)
    summary = {}

    # 0) zero-shot baselines
    student0 = load(s_path)
    teacher = load(t_path)
    meta['params'] = dict(teacher=sum(p.numel() for p in teacher.parameters()), student=sum(p.numel() for p in student0.parameters()))
    for name, m in [('zeroshot_teacher', teacher), ('zeroshot_student', student0)]:
        summary[name], _ = metrics(candidate_logps(m, codec, test_prefixes), gold)
        summary[name]['greedy'], _ = greedy_check(m, codec, test_rows[:300])
        print(name, json.dumps({k: summary[name][k] for k in ('accuracy', 'macro_f1')}), flush=True)
    student_state = {k: v.detach().cpu().clone() for k, v in student0.state_dict().items()}
    del student0

    # 1) teacher specialization
    t_items = [dict(**codec.example(r['text'], r['label']), w_ce=1.0, w_kd=0.0) for r in teacher_rows]
    summary['teacher_train_log'] = train(teacher, t_items, TEACHER_STEPS, 42, log='teacher')
    teacher.eval()
    teacher.requires_grad_(False)
    t_scores = candidate_logps(teacher, codec, test_prefixes)
    summary['teacher_sft'], t_probs = metrics(t_scores, gold)
    summary['teacher_sft']['greedy'], t_greedy = greedy_check(teacher, codec, test_rows[:300])
    print('teacher_sft', json.dumps({k: summary['teacher_sft'][k] for k in ('accuracy', 'macro_f1')}), flush=True)
    jsonl(args.output / 'pred_teacher_sft.jsonl', [dict(index=r['index'], gold=LABELS[r['label']], probs=p.tolist(),
                                                       greedy=t_greedy[i] if i < 300 else None)
                                                  for i, (r, p) in enumerate(zip(test_rows, t_probs))])
    # teacher pseudo-labels for unlabeled pool (gold labels unused)
    u_pseudo = candidate_logps(teacher, codec, [r['prefix'] for r in unl_rows]).argmax(-1).tolist()
    summary['pseudo_label_accuracy_diagnostic'] = sum(p == r['label'] for p, r in zip(u_pseudo, unl_rows)) / len(unl_rows)
    teacher_hash = statehash(teacher)
    meta['teacher_sft_statehash'] = teacher_hash

    lab = [codec.example(r['text'], r['label']) for r in student_rows]
    unl = [codec.example(r['text'], p) for r, p in zip(unl_rows, u_pseudo)]
    big = [codec.example(r['text'], r['label']) for r in teacher_rows]
    runs = {
        'S1_ce200': ([dict(**e, w_ce=1.0, w_kd=0.0) for e in lab], False),
        'S2_kd200': ([dict(**e, w_ce=ALPHA, w_kd=1 - ALPHA) for e in lab], True),
        'S3_kd200_u2000': ([dict(**e, w_ce=ALPHA, w_kd=1 - ALPHA) for e in lab] + [dict(**e, w_ce=0.0, w_kd=1.0) for e in unl], True),
        'S4_pseudo200_u2000': ([dict(**e, w_ce=1.0, w_kd=0.0) for e in lab + unl], False),
        'S5_ce2000': ([dict(**e, w_ce=1.0, w_kd=0.0) for e in big], False),
    }
    summary['runs'] = {}
    for name, (items, use_teacher) in runs.items():
        summary['runs'][name] = {}
        for seed in SEEDS:
            student = MiniMindForCausalLM(MiniMindConfig(**json.loads((s_path / 'config.json').read_text())))
            student.load_state_dict(student_state, strict=True)
            student = student.cuda()
            tag = f'{name}_seed{seed}'
            log = train(student, items, STUDENT_STEPS, seed, teacher=teacher if use_teacher else None, log=tag)
            r, probs = metrics(candidate_logps(student, codec, test_prefixes), gold, t_probs)
            r['greedy'], outs = greedy_check(student, codec, test_rows[:300])
            r['train_log'] = log
            summary['runs'][name][seed] = r
            jsonl(args.output / f'pred_{tag}.jsonl', [dict(index=t['index'], gold=LABELS[t['label']], probs=p.tolist(),
                                                           greedy=outs[i] if i < 300 else None)
                                                      for i, (t, p) in enumerate(zip(test_rows, probs))])
            print('eval', tag, json.dumps({k: r[k] for k in ('accuracy', 'macro_f1', 'teacher_agreement')}), flush=True)
            del student
            torch.cuda.empty_cache()
        assert statehash(teacher) == teacher_hash, 'teacher changed'
        dump(args.output / 'summary.json', summary)

    agg = {}
    for name, per in summary['runs'].items():
        agg[name] = {}
        for k in ('accuracy', 'macro_f1', 'nll', 'ece', 'teacher_agreement', 'kl_teacher_student_6class'):
            v = torch.tensor([per[s][k] for s in SEEDS])
            agg[name][k] = dict(mean=v.mean().item(), std=v.std(unbiased=True).item())
        agg[name]['greedy_valid_mean'] = sum(per[s]['greedy']['valid_rate'] for s in SEEDS) / len(SEEDS)
    summary['aggregate'] = agg
    s1, s3 = agg['S1_ce200']['accuracy'], agg['S3_kd200_u2000']['accuracy']
    summary['criteria'] = dict(
        premise_teacher_beats_S1_by_5pt=summary['teacher_sft']['accuracy'] - s1['mean'] >= 0.05,
        S3_gt_S1=s3['mean'] > s1['mean'],
        S3_minus_S1_gt_2std=(s3['mean'] - s1['mean']) > 2 * max(s1['std'], s3['std']),
        teacher_unchanged=True)
    meta['peak_cuda_bytes'] = torch.cuda.max_memory_allocated()
    meta['total_seconds'] = time.monotonic() - t_start
    dump(args.output / 'metadata.json', meta)
    dump(args.output / 'summary.json', summary)
    print('DONE', json.dumps(summary['criteria']), flush=True)


if __name__ == '__main__':
    main()
