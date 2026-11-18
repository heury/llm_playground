"""Small, reproducible full-SFT factual recall experiment (no remote code execution)."""
import argparse
import hashlib
import json
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
import transformers
from safetensors.torch import load_model
from transformers import AutoTokenizer

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from dataset.lm_dataset import SFTDataset

REPO = 'caspar/MiniMind-Dense-64M-Mixed-SFT-v1.0'
REVISION = '7597c22aa4d3e7b2ac824b4b4be01158cccf1b8d'
TRAIN = {
    'color': ['What color is {n}?', 'Tell me the color of {n}.', 'Which color does {n} come in?', 'What is the color of the {n} product?', 'Give the color of {n}.'],
    'battery': ['What is the battery life of {n}?', 'How many hours does {n} last on a charge?', 'Tell me the battery duration of {n}.', 'How long can {n} run on one charge?', 'Give the battery life in hours for {n}.'],
    'warranty': ['What is the warranty period for {n}?', 'How many years is {n} covered by warranty?', 'Tell me the warranty length of {n}.', 'What warranty does {n} have?', 'Give the warranty duration in years for {n}.'],
}
HELDOUT = {
    'color': ['If I buy {n}, what color will it be?', 'Describe the exterior color of {n}.'],
    'battery': ['After fully charging {n}, how many hours can I use it?', 'How much operating time does a full battery give {n}?'],
    'warranty': ['For how many years after purchase is {n} under warranty?', 'When does the warranty on {n} expire, in years?'],
}
COLORS = 'teal orange purple yellow black white red blue green pink brown gray cyan magenta silver gold beige maroon navy olive'.split()


def dump(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')


def jsonl(path, rows):
    path.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))


def make_data():
    rng = random.Random(42)
    names = 'Nevo Tarin Movek Zunel Fepra Kolvi Ruvex Dalmo Penri Sovak Birel Wunor Javin Hekro Luvin Cevra Nupel Tavro Zemik Fovin'.split()
    colors, hours, years = COLORS.copy(), list(range(7, 47, 2)), list(range(1, 21))
    for values in (colors, hours, years):
        rng.shuffle(values)
    facts, train, heldout = [], [], []
    for name, color, hour, year in zip(names, colors, hours, years):
        fact = dict(name=name, color=color, battery=f'{hour} hours', warranty=f'{year} ' + ('year' if year == 1 else 'years'))
        facts.append(fact)
        for attr in TRAIN:
            for templates, dest in ((TRAIN[attr], train), (HELDOUT[attr], heldout)):
                for template in templates:
                    q = template.format(n=name) + ' Answer briefly.'
                    dest.append(dict(name=name, attribute=attr, question=q, answer=fact[attr], conversations=[dict(role='user', content=q), dict(role='assistant', content=fact[attr])]))
    assert len(train) == 300 and len(heldout) == 120
    assert not ({r['question'] for r in train} & {r['question'] for r in heldout})
    return facts, train, heldout


def render(tokenizer, messages, generation=False):
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=generation, open_thinking=False).replace('<think>\n\n</think>\n\n', '')


def score(row, answer):
    answer = answer.strip().lower()
    if row['attribute'] == 'color':
        found = {c for c in COLORS if re.search(r'\b' + c + r'\b', answer)}
        return found == {row['answer']}
    values = re.findall(r'\d+(?:\.\d+)?', answer)
    expected = re.search(r'\d+', row['answer']).group()
    unit = r'\b(?:hours?|hrs?)\b' if row['attribute'] == 'battery' else r'\byears?\b'
    wrong = r'\byears?\b' if row['attribute'] == 'battery' else r'\b(?:hours?|hrs?)\b'
    return bool(values and set(values) == {expected} and re.search(unit, answer) and not re.search(wrong, answer))


@torch.inference_mode()
def evaluate(model, tokenizer, rows, output, tag, batch_size=16):
    model.eval()
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        ids = tokenizer(render(tokenizer, [dict(role='user', content=row['question'])], True), add_special_tokens=False).input_ids
        assert len(ids) + 48 <= 340
        groups[len(ids)].append((i, ids))
    results = [None] * len(rows)
    start = time.monotonic()
    for group in groups.values():
        for offset in range(0, len(group), batch_size):
            batch = group[offset:offset + batch_size]
            x = torch.tensor([v[1] for v in batch], device='cuda')
            with torch.autocast('cuda', dtype=torch.bfloat16):
                y = model.generate(inputs=x, max_new_tokens=48, do_sample=False, temperature=1., top_p=1., top_k=0, repetition_penalty=1., eos_token_id=tokenizer.eos_token_id)
            for (i, _), tokens in zip(batch, y[:, x.shape[1]:].tolist()):
                eos = tokenizer.eos_token_id in tokens
                if eos:
                    tokens = tokens[:tokens.index(tokenizer.eos_token_id)]
                answer = tokenizer.decode(tokens, skip_special_tokens=True)
                row = rows[i]
                results[i] = dict(**{k: v for k, v in row.items() if k != 'conversations'}, generated=answer, terminated=eos)
                if 'answer' in row:
                    results[i].update(correct=score(row, answer), exact_match=answer.strip().lower().rstrip('.') == row['answer'].lower())
    jsonl(output / f'{tag}.jsonl', results)
    scored = [r for r in results if 'correct' in r]
    summary = dict(n=len(rows), seconds=time.monotonic() - start, truncated=sum(not r['terminated'] for r in results))
    if scored:
        summary.update(correct=sum(r['correct'] for r in scored), accuracy=sum(r['correct'] for r in scored) / len(scored), exact_match=sum(r['exact_match'] for r in scored) / len(scored))
        summary['by_attribute'] = {a: dict(correct=sum(r['correct'] for r in scored if r['attribute'] == a), n=sum(r['attribute'] == a for r in scored)) for a in TRAIN}
    print(tag, json.dumps(summary), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-5)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.set_num_threads(4)
    assert torch.cuda.is_available()
    tokenizer = AutoTokenizer.from_pretrained(args.base)
    config = MiniMindConfig(**json.loads((args.base / 'config.json').read_text()))
    model = MiniMindForCausalLM(config)
    missing, unexpected = load_model(model, args.base / 'model.safetensors', strict=True)
    assert not missing and not unexpected
    model = model.cuda()
    facts, train, heldout = make_data()
    dump(args.output / 'facts.json', facts)
    jsonl(args.output / 'train.jsonl', train)
    jsonl(args.output / 'heldout.jsonl', heldout)
    unknown = [dict(question=f'What is the battery life of {n}? Answer briefly.') for n in ['Qezun', 'Vaxil', 'Torpex', 'Bemzu', 'Gufra']]
    control = [dict(question=q) for q in ['Why is the sky blue?', 'What is the capital of France?', 'What is 2 + 3?', 'Explain photosynthesis briefly.', 'Write a Python function that adds two numbers.', 'Name three common fruits.', 'What is the opposite of hot?', 'How can I stay organized while studying?']]
    metadata = dict(repo=REPO, revision=REVISION, torch=torch.__version__, transformers=transformers.__version__, device=torch.cuda.get_device_name(0), parameters=sum(p.numel() for p in model.parameters()), trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad), precision='FP32 parameters and AdamW states; BF16 autocast', seed=42, lr=args.lr, epochs=args.epochs, batch_size=8, accumulation_steps=1, weight_decay=.01, grad_clip=1., schedule='constant', training_samples=300, heldout_samples=120, max_generation_tokens=48, facts_sha256=hashlib.sha256((args.output / 'facts.json').read_bytes()).hexdigest(), base_sha256=hashlib.sha256((args.base / 'model.safetensors').read_bytes()).hexdigest(), template='published tokenizer, open_thinking=False, deterministic empty-think removal', masking='native SFTDataset.generate_labels; no random system insertion')
    dump(args.output / 'metadata.json', metadata)
    print('metadata', json.dumps(metadata), flush=True)
    # Reuse native assistant-only labels and check there are no truncated targets.
    masker = object.__new__(SFTDataset)
    masker.max_length = 256
    masker.bos_id = tokenizer(f'{tokenizer.bos_token}assistant\n', add_special_tokens=False).input_ids
    masker.eos_id = tokenizer(f'{tokenizer.eos_token}\n', add_special_tokens=False).input_ids
    encoded = []
    for row in train:
        ids = tokenizer(render(tokenizer, row['conversations']), add_special_tokens=False).input_ids
        labels = masker.generate_labels(ids)
        prefix = tokenizer(render(tokenizer, row['conversations'][:1], True), add_special_tokens=False).input_ids
        assert ids[:len(prefix)] == prefix
        assert labels[:len(prefix)] == [-100] * len(prefix)
        assert labels[len(prefix):] == ids[len(prefix):]
        assert len(ids) <= 256 and any(v != -100 for v in labels[1:])
        encoded.append((ids, labels))
    metadata['max_train_tokens'] = max(len(x[0]) for x in encoded)
    dump(args.output / 'metadata.json', metadata)
    summaries = {}
    summaries['before_train'] = evaluate(model, tokenizer, train, args.output, 'before_train')
    summaries['before_heldout'] = evaluate(model, tokenizer, heldout, args.output, 'before_heldout')
    summaries['before_unknown'] = evaluate(model, tokenizer, unknown, args.output, 'before_unknown')
    summaries['before_control'] = evaluate(model, tokenizer, control, args.output, 'before_control')
    dump(args.output / 'summary.json', summaries)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=.01)
    training_start = time.monotonic()
    step = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        indices = torch.randperm(len(encoded), generator=torch.Generator().manual_seed(42 + epoch)).tolist()
        loss_sum, targets_sum = 0., 0
        for offset in range(0, len(indices), 8):
            batch = [encoded[i] for i in indices[offset:offset + 8]]
            length = max(len(ids) for ids, _ in batch)
            x = torch.tensor([ids + [tokenizer.pad_token_id] * (length - len(ids)) for ids, _ in batch], device='cuda')
            labels = torch.tensor([lab + [-100] * (length - len(lab)) for _, lab in batch], device='cuda')
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                loss = model(x, labels=labels).loss
            assert torch.isfinite(loss).item()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            assert torch.isfinite(grad_norm).item()
            optimizer.step()
            n = (labels[:, 1:] != -100).sum().item()
            loss_sum += loss.item() * n
            targets_sum += n
            step += 1
        record = dict(epoch=epoch, updates=step, assistant_token_loss=loss_sum / targets_sum, elapsed_training_seconds=time.monotonic() - training_start)
        with (args.output / 'training.jsonl').open('a') as f:
            f.write(json.dumps(record) + '\n')
        print('training', json.dumps(record), flush=True)
        if epoch in {1, 5, args.epochs}:
            summaries[f'epoch{epoch}_heldout'] = evaluate(model, tokenizer, heldout, args.output, f'epoch{epoch}_heldout')
            dump(args.output / 'summary.json', summaries)
    assert all(torch.isfinite(p).all().item() for p in model.parameters())
    metadata.update(updates=step, training_seconds=time.monotonic() - training_start, peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(), finite_parameters=True)
    # Save full FP32 inference state, then verify exact strict reload before final evaluation.
    torch.save(model.state_dict(), args.output / 'full_sft_768.pth')
    saved = torch.load(args.output / 'full_sft_768.pth', map_location='cpu', weights_only=True)
    assert all(torch.equal(v.cpu(), saved[k]) for k, v in model.state_dict().items())
    model.load_state_dict(saved, strict=True)
    del saved
    metadata['strict_reload_equal'] = True
    metadata['checkpoint_sha256'] = hashlib.sha256((args.output / 'full_sft_768.pth').read_bytes()).hexdigest()
    dump(args.output / 'metadata.json', metadata)
    summaries['after_train'] = evaluate(model, tokenizer, train, args.output, 'after_train')
    summaries['after_heldout'] = evaluate(model, tokenizer, heldout, args.output, 'after_heldout')
    summaries['after_unknown'] = evaluate(model, tokenizer, unknown, args.output, 'after_unknown')
    summaries['after_control'] = evaluate(model, tokenizer, control, args.output, 'after_control')
    summaries['criteria'] = dict(train_95_percent=summaries['after_train']['accuracy'] >= .95, heldout_80_percent=summaries['after_heldout']['accuracy'] >= .8, heldout_improved=summaries['after_heldout']['accuracy'] > summaries['before_heldout']['accuracy'])
    dump(args.output / 'summary.json', summaries)
    print('DONE', json.dumps(summaries['criteria']), flush=True)


if __name__ == '__main__':
    main()
