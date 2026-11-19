"""Compare LoRA init: (A=0, B=0) vs (A~N(0,0.02), B=0), 3 training steps on identical batches.

Uses native model_lora.apply_lora, then overrides A for the zero-zero condition.
Task/data as lora_ab_dynamics.py. Run from projects/minimind/ with .venv. CUDA required.
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import datasets
import torch
import transformers
from transformers import AutoTokenizer

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from kd_classification import DATA_REPO, DATA_REV, STUDENT_REPO, STUDENT_REV, Codec, dump, load, snapshot  # noqa: E402
from model.model_lora import apply_lora  # noqa: E402

RANK, LR, BATCH, STEPS, SEED = 16, 1e-4, 16, 3, 42


def run(s_path, batches, zero_A):
    torch.manual_seed(SEED)
    model = load(s_path)
    torch.manual_seed(SEED)
    apply_lora(model, rank=RANK)
    mods = [(n, m) for n, m in model.named_modules() if hasattr(m, 'lora')]
    if zero_A:
        for _, m in mods:
            m.lora.A.weight.data.zero_()
    params = []
    for n, p in model.named_parameters():
        p.requires_grad = 'lora' in n
        if p.requires_grad:
            params.append(p)
    opt = torch.optim.AdamW(params, lr=LR)  # train_lora.py: AdamW defaults (wd 0.01)
    A0 = {n: m.lora.A.weight.detach().float().clone() for n, m in mods}

    def state():
        A = torch.stack([m.lora.A.weight.detach().float().norm() for _, m in mods])
        B = torch.stack([m.lora.B.weight.detach().float().norm() for _, m in mods])
        dW = torch.stack([(m.lora.B.weight.float() @ m.lora.A.weight.float()).detach().norm() for _, m in mods])
        dA = torch.stack([(m.lora.A.weight.detach().float() - A0[n]).norm() for n, m in mods])
        return dict(A_norm=A.norm().item(), A_change=dA.norm().item(), A_absmax=max(m.lora.A.weight.abs().max().item() for _, m in mods),
                    B_norm=B.norm().item(), B_absmax=max(m.lora.B.weight.abs().max().item() for _, m in mods),
                    dW_norm=dW.norm().item(), A_nonzero=sum(int((m.lora.A.weight != 0).sum()) for _, m in mods),
                    B_nonzero=sum(int((m.lora.B.weight != 0).sum()) for _, m in mods))

    rows = [dict(step=0, **state())]
    probe = batches[0][0][:1]
    with torch.no_grad():
        model.eval()
        base_logits = model(probe).logits.float()
    for step, (x, labels) in enumerate(batches, start=1):
        model.train()
        with torch.autocast('cuda', dtype=torch.bfloat16):
            loss = model(x, labels=labels).loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gA = torch.stack([m.lora.A.weight.grad.float().norm() for _, m in mods]).norm().item()
        gB = torch.stack([m.lora.B.weight.grad.float().norm() for _, m in mods]).norm().item()
        gA_max = max(m.lora.A.weight.grad.abs().max().item() for _, m in mods)
        gB_max = max(m.lora.B.weight.grad.abs().max().item() for _, m in mods)
        gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        with torch.no_grad():
            model.eval()
            diff = (model(probe).logits.float() - base_logits).abs().max().item()
        rows.append(dict(step=step, loss=loss.item(), gradA_norm=gA, gradB_norm=gB, gradA_absmax=gA_max, gradB_absmax=gB_max,
                         clip_total_norm=gn.item(), probe_logit_change=diff, **state()))
        print(('zeroA' if zero_A else 'randA'), json.dumps(rows[-1]), flush=True)
    del model
    torch.cuda.empty_cache()
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    assert not args.output.exists()
    args.output.mkdir(parents=True)
    t0 = time.monotonic()
    s_path = snapshot(STUDENT_REPO, STUDENT_REV)
    codec = Codec(AutoTokenizer.from_pretrained(s_path))
    ds = datasets.load_dataset(DATA_REPO, 'split', revision=DATA_REV)
    perm = list(range(len(ds['train'])))
    random.Random(42).shuffle(perm)
    items = [codec.example(ds['train'][i]['text'], ds['train'][i]['label']) for i in perm[:2000]]
    order = torch.randperm(len(items), generator=torch.Generator().manual_seed(SEED)).tolist()
    batches = []
    for s in range(STEPS):  # identical batches for both conditions
        b = [items[i] for i in order[s * BATCH:(s + 1) * BATCH]]
        L = max(len(e['ids']) for e in b)
        x = torch.tensor([e['ids'] + [0] * (L - len(e['ids'])) for e in b], device='cuda')
        labels = torch.full_like(x, -100)
        for i, e in enumerate(b):
            labels[i, e['start']:len(e['ids'])] = x[i, e['start']:len(e['ids'])]
        batches.append((x, labels))
    results = dict(randA_zeroB=run(s_path, batches, zero_A=False), zeroA_zeroB=run(s_path, batches, zero_A=True))
    meta = dict(model=dict(repo=STUDENT_REPO, revision=STUDENT_REV), data=dict(repo=DATA_REPO, revision=DATA_REV),
                rank=RANK, lr=LR, constant_lr=True, batch=BATCH, steps=STEPS, seed=SEED, optimizer='AdamW defaults (wd 0.01)',
                clip=1.0, autocast='bf16', lora_layers=16, batch_indices=[order[s * BATCH:(s + 1) * BATCH] for s in range(STEPS)],
                torch=torch.__version__, transformers=transformers.__version__, device=torch.cuda.get_device_name(0),
                seconds=time.monotonic() - t0)
    dump(args.output / 'metadata.json', meta)
    dump(args.output / 'results.json', results)


if __name__ == '__main__':
    main()
