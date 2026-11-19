"""Track how LoRA A and B matrices change during training (native model_lora.apply_lora).

Task: dair-ai/emotion 6-class classification (same prompt/split code as kd_classification.py),
student caspar/MiniMind-Dense-64M-Mixed-SFT-v1.0. Run from projects/minimind/ with .venv. CUDA required.
"""
import argparse
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
from transformers import AutoTokenizer

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from kd_classification import (DATA_REPO, DATA_REV, LABELS, STUDENT_REPO, STUDENT_REV, Codec, candidate_logps,  # noqa: E402
                               dump, load, metrics, snapshot, statehash)
from model.model_lora import apply_lora  # noqa: E402  (kd_classification put PROJECT on sys.path)

RANK, LR, BATCH, STEPS, SEED = 16, 1e-4, 16, 300, 42  # train_lora.py defaults: rank 16, lr 1e-4, AdamW, clip 1
TRACK = {0, 1, 2, 3, 5, 10, 25, 50, 100, 150, 200, 250, 300}


def lr_at(step, total):  # trainer_utils.get_lr
    return LR * (0.1 + 0.45 * (1 + math.cos(math.pi * step / total)))


def eff_rank(s):
    p = s / s.sum()
    p = p[p > 0]
    return math.exp(-(p * p.log()).sum().item())


def subspace_cos(U, V):
    """Mean cosine of principal angles between column spaces of U and V (orthonormalized)."""
    qu, _ = torch.linalg.qr(U)
    qv, _ = torch.linalg.qr(V)
    return torch.linalg.svdvals(qu.T @ qv).mean().item()


def snapshot_stats(mods, init, prev, grads=None):
    rows = []
    for name, m in mods:
        A, B = m.lora.A.weight.detach().float(), m.lora.B.weight.detach().float()  # A: [r, in], B: [out, r]
        A0 = init[name]['A']
        W = m.weight.detach().float()
        dW = B @ A
        s = torch.linalg.svdvals(dW)
        r = dict(layer=name, A_norm=A.norm().item(), A_rel_change=(A - A0).norm().item() / A0.norm().item(),
                 A_std=A.std().item(), A_rowspace_cos_init=subspace_cos(A.T, A0.T),
                 B_norm=B.norm().item(), B_absmax=B.abs().max().item(), B_std=B.std().item(),
                 dW_norm=dW.norm().item(), dW_rel_W=dW.norm().item() / W.norm().item(),
                 dW_top_sv=s[:4].tolist(), dW_eff_rank=eff_rank(s) if s.sum() > 0 else 0.0,
                 # (A - A0) direction: how much of A's update lies in A0's own row space
                 )
        if B.norm() > 0 and prev.get(name) is not None and prev[name].norm() > 0:
            r['B_colspace_cos_prev'] = subspace_cos(B, prev[name])
        if grads:
            r.update(grads[name])
        prev[name] = B.clone()
        rows.append(r)
    keys = [k for k in rows[0] if k not in ('layer', 'dW_top_sv')]
    agg = {k: sum(r[k] for r in rows if k in r) / max(1, sum(k in r for r in rows)) for k in keys}
    return rows, agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    assert not args.output.exists()
    args.output.mkdir(parents=True)
    random.seed(SEED)
    torch.manual_seed(SEED)
    t0 = time.monotonic()

    s_path = snapshot(STUDENT_REPO, STUDENT_REV)
    codec = Codec(AutoTokenizer.from_pretrained(s_path))
    ds = datasets.load_dataset(DATA_REPO, 'split', revision=DATA_REV)
    perm = list(range(len(ds['train'])))
    random.Random(42).shuffle(perm)
    train_rows = [ds['train'][i] for i in perm[:2000]]  # same as kd_classification "teacher" split
    test = ds['test']
    test_prefixes = [codec.prefix(t) for t in test['text']]
    gold = list(test['label'])
    items = [codec.example(r['text'], r['label']) for r in train_rows]

    model = load(s_path)
    base_hash = statehash(model)
    torch.manual_seed(SEED)
    apply_lora(model, rank=RANK)
    params = []
    for n, p in model.named_parameters():
        p.requires_grad = 'lora' in n
        if p.requires_grad:
            params.append(p)
    mods = [(n, m) for n, m in model.named_modules() if hasattr(m, 'lora')]
    meta = dict(model=dict(repo=STUDENT_REPO, revision=STUDENT_REV), data=dict(repo=DATA_REPO, revision=DATA_REV),
                rank=RANK, lr=LR, schedule='get_lr cosine 0.1-1.0x', batch=BATCH, steps=STEPS, seed=SEED,
                optimizer='torch.optim.AdamW(lora_params, lr) defaults (wd 0.01, betas 0.9/0.999)', clip=1.0,
                autocast='bf16', lora_layers=[n for n, _ in mods], lora_params=sum(p.numel() for p in params),
                total_params=sum(p.numel() for p in model.parameters()), torch=torch.__version__,
                transformers=transformers.__version__, datasets=datasets.__version__,
                device=torch.cuda.get_device_name(0))
    dump(args.output / 'metadata.json', meta)
    print('lora layers', len(mods), meta['lora_params'], flush=True)

    init = {n: dict(A=m.lora.A.weight.detach().float().clone(), B=m.lora.B.weight.detach().float().clone()) for n, m in mods}
    assert all(v['B'].abs().max() == 0 for v in init.values())
    before, _ = metrics(candidate_logps(model, codec, test_prefixes), gold)

    opt = torch.optim.AdamW(params, lr=LR)
    g = torch.Generator().manual_seed(SEED)
    order, prev, trace, per_layer = [], {}, [], {}
    rows, agg = snapshot_stats(mods, init, prev)
    trace.append(dict(step=0, **agg))
    per_layer[0] = rows
    for step in range(1, STEPS + 1):
        if len(order) < BATCH:
            order += torch.randperm(len(items), generator=g).tolist()
        batch = [items[i] for i in order[:BATCH]]
        order = order[BATCH:]
        L = max(len(b['ids']) for b in batch)
        x = torch.tensor([b['ids'] + [0] * (L - len(b['ids'])) for b in batch], device='cuda')
        labels = torch.full_like(x, -100)
        for i, b in enumerate(batch):
            labels[i, b['start']:len(b['ids'])] = x[i, b['start']:len(b['ids'])]
        for pg in opt.param_groups:
            pg['lr'] = lr_at(step - 1, STEPS)
        model.train()
        with torch.autocast('cuda', dtype=torch.bfloat16):
            loss = model(x, labels=labels).loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        grads = None
        if step in TRACK:  # raw (pre-clip) gradient norms of this step
            grads = {n: dict(gradA_norm=m.lora.A.weight.grad.norm().item(), gradB_norm=m.lora.B.weight.grad.norm().item())
                     for n, m in mods}
        gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
        assert torch.isfinite(loss) and torch.isfinite(gn)
        opt.step()
        if step in TRACK:
            rows, agg = snapshot_stats(mods, init, prev, grads)
            trace.append(dict(step=step, loss=loss.item(), total_grad_norm=gn.item(), **agg))
            per_layer[step] = rows
            print('step', json.dumps({k: round(v, 6) if isinstance(v, float) else v for k, v in trace[-1].items()}), flush=True)

    after, _ = metrics(candidate_logps(model, codec, test_prefixes), gold)

    # checks: base frozen (compared to a fresh load, before merging), merge W + BA reproduces LoRA forward
    ref_sd = load(s_path).state_dict()
    base_unchanged = all(torch.equal(v, ref_sd[k]) for k, v in model.state_dict().items() if '.lora.' not in k)
    del ref_sd
    model.eval()
    probe = torch.tensor([items[0]['ids']], device='cuda')
    with torch.no_grad():
        lora_logits = model(probe).logits.float()
        for n, m in mods:
            m.weight.data += (m.lora.B.weight.data @ m.lora.A.weight.data)
            m.lora.B.weight.data.zero_()
        merged_logits = model(probe).logits.float()
    checks = dict(base_weights_unchanged=base_unchanged, base_statehash=base_hash,
                  merge_max_abs_logit_diff=(lora_logits - merged_logits).abs().max().item(),
                  probe_logit_absmax=lora_logits.abs().max().item())
    summary = dict(before=before, after=after, trace=trace, checks=checks, seconds=time.monotonic() - t0)
    dump(args.output / 'summary.json', summary)
    dump(args.output / 'per_layer.json', {str(k): v for k, v in per_layer.items()})
    print('DONE', json.dumps(dict(before=before['accuracy'], after=after['accuracy'], **checks)), flush=True)


if __name__ == '__main__':
    main()
