"""A small full-parameter DPO experiment: correct, direct answers vs filler."""
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
from dataset.lm_dataset import DPODataset
from trainer.train_dpo import dpo_loss

REPO = 'caspar/MiniMind-Dense-64M-Mixed-SFT-v1.0'
REVISION = '7597c22aa4d3e7b2ac824b4b4be01158cccf1b8d'
CAPITALS = [('France', 'Paris'), ('Germany', 'Berlin'), ('Italy', 'Rome'), ('Spain', 'Madrid'), ('Portugal', 'Lisbon'), ('Japan', 'Tokyo'), ('Canada', 'Ottawa'), ('Australia', 'Canberra'), ('Brazil', 'Brasilia'), ('Argentina', 'Buenos Aires'), ('Chile', 'Santiago'), ('Peru', 'Lima'), ('Mexico', 'Mexico City'), ('Egypt', 'Cairo'), ('Kenya', 'Nairobi'), ('Ghana', 'Accra'), ('Greece', 'Athens'), ('Austria', 'Vienna'), ('Poland', 'Warsaw'), ('Sweden', 'Stockholm'), ('Norway', 'Oslo'), ('Finland', 'Helsinki'), ('Denmark', 'Copenhagen'), ('Ireland', 'Dublin'), ('Iceland', 'Reykjavik'), ('Hungary', 'Budapest'), ('Romania', 'Bucharest'), ('Bulgaria', 'Sofia'), ('Thailand', 'Bangkok'), ('Vietnam', 'Hanoi')]
OPPOSITES = [('hot', 'cold'), ('big', 'small'), ('happy', 'sad'), ('fast', 'slow'), ('up', 'down'), ('left', 'right'), ('open', 'closed'), ('wet', 'dry'), ('young', 'old'), ('early', 'late'), ('full', 'empty'), ('light', 'dark'), ('clean', 'dirty'), ('rich', 'poor'), ('strong', 'weak'), ('easy', 'difficult'), ('near', 'far'), ('wide', 'narrow'), ('thick', 'thin'), ('soft', 'hard'), ('loud', 'quiet'), ('safe', 'dangerous'), ('true', 'false'), ('inside', 'outside'), ('above', 'below'), ('before', 'after'), ('always', 'never'), ('increase', 'decrease'), ('accept', 'reject'), ('remember', 'forget')]
DEFINITIONS = [
    ('triangle', 'A triangle is a shape with three sides.', ['three sides', '3 sides']),
    ('square', 'A square is a shape with four equal sides and four right angles.', ['four equal sides', '4 equal sides']),
    ('circle', 'A circle is a round shape whose boundary is equally distant from its center.', ['round', 'equally distant', 'same distance']),
    ('noun', 'A noun is a word that names a person, place, thing, or idea.', ['person', 'place', 'thing']),
    ('verb', 'A verb is a word that expresses an action or state.', ['action', 'state']),
    ('adjective', 'An adjective is a word that describes a noun.', ['describes a noun', 'modify a noun', 'modifies a noun']),
    ('synonym', 'A synonym is a word with the same or a similar meaning as another word.', ['similar meaning', 'same meaning']),
    ('antonym', 'An antonym is a word with the opposite meaning of another word.', ['opposite']),
    ('mammal', 'A mammal is an animal that feeds its young with milk.', ['milk']),
    ('herbivore', 'A herbivore is an animal that eats plants.', ['plants', 'plant']),
    ('carnivore', 'A carnivore is an animal that eats meat.', ['meat', 'other animals']),
    ('omnivore', 'An omnivore is an animal that eats both plants and animals.', ['plants and animals', 'plants and meat', 'plant and animal']),
    ('evaporation', 'Evaporation is the change of a liquid into a gas.', ['gas', 'vapor', 'vapour']),
    ('condensation', 'Condensation is the change of a gas into a liquid.', ['liquid']),
    ('freezing', 'Freezing is the change of a liquid into a solid.', ['solid']),
    ('melting', 'Melting is the change of a solid into a liquid.', ['liquid']),
    ('gravity', 'Gravity is the force that attracts objects with mass toward each other.', ['attract', 'pull']),
    ('friction', 'Friction is a force that resists motion between touching surfaces.', ['resist', 'opposes', 'opposing', 'resistance']),
    ('thermometer', 'A thermometer is an instrument that measures temperature.', ['temperature']),
    ('barometer', 'A barometer is an instrument that measures atmospheric pressure.', ['pressure']),
    ('microscope', 'A microscope is an instrument that magnifies small objects.', ['magnif', 'small objects', 'tiny objects']),
    ('telescope', 'A telescope is an instrument used to observe distant objects.', ['distant', 'far away', 'faraway']),
    ('dictionary', 'A dictionary is a reference that lists words and their meanings.', ['meaning', 'definition']),
    ('autobiography', 'An autobiography is a story of a person\'s life written by that person.', ['own life', 'that person', 'themselves', 'oneself']),
    ('island', 'An island is an area of land surrounded by water.', ['surrounded by water']),
    ('peninsula', 'A peninsula is land surrounded by water on three sides.', ['three sides', '3 sides', 'almost surrounded']),
    ('equator', 'The equator is an imaginary line halfway between the North and South Poles.', ['north and south', 'northern and southern', 'latitude', 'halfway']),
    ('planet', 'A planet is a large, nearly round body that orbits a star and has cleared its orbital neighborhood.', ['orbits', 'orbiting', 'orbit']),
    ('photosynthesis', 'Photosynthesis is the process by which plants use light to make food.', ['light', 'sunlight']),
    ('pollination', 'Pollination is the transfer of pollen to the female part of a flower.', ['pollen']),
]
WRAPPERS = [
    'That is an interesting question! Let me answer it carefully. {a} So, to repeat the answer: {a}',
    'Thank you for asking this question. I am happy to help you with it. {a} In conclusion, {a}',
    'Let us take a moment to consider this simple question. {a} As I mentioned above, {a}',
    'Of course! I can certainly provide an answer to your question. {a} To summarize, {a}',
    'This is a good question and deserves a clear response. {a} Once again, {a}',
]
CONTROLS = [
    'Explain why the sky appears blue in three sentences.',
    'Explain photosynthesis to a child in three sentences.',
    'Write a Python function that returns the larger of two numbers.',
    'Write a Python function that reverses a string.',
    'Give three practical tips for staying organized while studying.',
    'Describe how to make a cup of tea step by step.',
    'Explain two differences between cats and dogs.',
    'Give two ways to save water at home and explain each one.',
]


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
    categories = collections.defaultdict(list)

    def add(category, fact, question, answer, expected, alternatives=None):
        i = len(categories[category])
        rejected = WRAPPERS[i % len(WRAPPERS)].format(a=answer)
        row = dict(category=category, fact=category + ':' + fact, question=question, answer=answer, expected=str(expected), alternatives=alternatives or [str(expected)], chosen=[dict(role='user', content=question), dict(role='assistant', content=answer)], rejected=[dict(role='user', content=question), dict(role='assistant', content=rejected)])
        categories[category].append(row)

    seen = set()
    for op, count in [('+', 30), ('-', 15), ('*', 15)]:
        while sum(r['operation'] == op for r in categories['arithmetic']) < count:
            a, b = rng.randint(2, 20), rng.randint(2, 20)
            if op == '-':
                a, b = max(a, b), min(a, b)
            fact = (op, min(a, b), max(a, b))
            if fact in seen:
                continue
            seen.add(fact)
            value = a + b if op == '+' else a - b if op == '-' else a * b
            add('arithmetic', repr(fact), f'What is {a} {op} {b}?', f'The answer is {value}.', value)
            categories['arithmetic'][-1]['operation'] = op
    for country, capital in CAPITALS:
        add('capital', country, f'What is the capital of {country}?', f'The capital of {country} is {capital}.', capital)
    for word, opposite in OPPOSITES:
        alts = [opposite]
        if opposite == 'small': alts.append('little')
        if opposite == 'difficult': alts.append('hard')
        if opposite == 'quiet': alts.append('silent')
        if opposite == 'closed': alts.append('close')
        add('opposite', word, f'What is the opposite of {word}?', f'The opposite of {word} is {opposite}.', opposite, alts)
    for word, answer, hints in DEFINITIONS:
        add('definition', word, f'What does the term "{word}" mean?', answer, word, hints)
    splits = {s: [] for s in ['train', 'validation', 'test']}
    for category, rows in categories.items():
        rng.shuffle(rows)
        n = len(rows)
        a, b = n * 2 // 3, n * 2 // 15
        splits['train'].extend(rows[:a])
        splits['validation'].extend(rows[a:a + b])
        splits['test'].extend(rows[a + b:])
    for rows in splits.values():
        rng.shuffle(rows)
    assert [len(splits[s]) for s in splits] == [100, 20, 30]
    assert len({r['fact'] for rows in splits.values() for r in rows}) == 150
    assert len({r['question'] for rows in splits.values() for r in rows}) == 150
    for rows in splits.values():
        for row in rows:
            assert row['chosen'][:-1] == row['rejected'][:-1]
            assert row['answer'] in row['rejected'][-1]['content']
    return splits


def render(tok, messages, generation=False):
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=generation, open_thinking=False).replace('<think>\n\n</think>\n\n', '')


def encode(tok, splits):
    masker = object.__new__(DPODataset)
    masker.max_length = 340
    masker.bos_id = tok(f'{tok.bos_token}assistant\n', add_special_tokens=False).input_ids
    masker.eos_id = tok(f'{tok.eos_token}\n', add_special_tokens=False).input_ids
    arrays = {}
    max_len = 0
    for split, rows in splits.items():
        arrays[split] = []
        for row in rows:
            pair = []
            prefix = tok(render(tok, row['chosen'][:-1], True), add_special_tokens=False).input_ids
            for side in ['chosen', 'rejected']:
                ids = tok(render(tok, row[side]), add_special_tokens=False).input_ids
                mask = masker.generate_loss_mask(ids)
                assert ids[:len(prefix)] == prefix
                assert not any(mask[:len(prefix)]) and all(mask[len(prefix):])
                assert tok.eos_token_id in ids[len(prefix):]
                assert len(ids) < 340
                max_len = max(max_len, len(ids))
                pair.append((ids, mask))
            arrays[split].append(pair)
    return arrays, max_len


def token_logps(model, pairs, pad, length):
    # Chosen first, rejected second, matching trainer.train_dpo.dpo_loss.
    samples = [p[0] for p in pairs] + [p[1] for p in pairs]
    ids = torch.tensor([x + [pad] * (length - len(x)) for x, _ in samples], device='cuda')
    mask = torch.tensor([m + [0] * (length - len(m)) for _, m in samples], device='cuda', dtype=torch.float32)[:, 1:]
    with torch.autocast('cuda', dtype=torch.bfloat16):
        logits = model(ids[:, :-1], use_cache=False).logits
    logps = F.log_softmax(logits.float(), dim=-1).gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1)
    return logps, mask


@torch.inference_mode()
def score_pairs(model, pairs, tok, length):
    model.eval()
    scores, counts = [], []
    for i in range(0, len(pairs), 4):
        logps, mask = token_logps(model, pairs[i:i + 4], tok.pad_token_id, length)
        scores.extend((logps * mask).sum(-1).reshape(2, -1).T.cpu().tolist())
        counts.extend(mask.sum(-1).reshape(2, -1).T.cpu().tolist())
    return torch.tensor(scores), torch.tensor(counts)


def pair_metrics(scores, reference, counts, beta):
    raw = scores[:, 0] - scores[:, 1]
    margin = beta * (raw - (reference[:, 0] - reference[:, 1]))
    return dict(n=len(scores), dpo_loss=(-F.logsigmoid(margin)).mean().item(), mean_reference_relative_margin=margin.mean().item(), reference_relative_improved_fraction=(margin > 1e-6).float().mean().item(), reference_relative_tie_fraction=(margin.abs() <= 1e-6).float().mean().item(), raw_sequence_chosen_fraction=(raw > 0).float().mean().item(), per_token_chosen_fraction=(scores[:, 0] / counts[:, 0] > scores[:, 1] / counts[:, 1]).float().mean().item(), chosen_token_nll=(-scores[:, 0].sum() / counts[:, 0].sum()).item(), rejected_token_nll=(-scores[:, 1].sum() / counts[:, 1].sum()).item())


def generation_correct(row, text):
    text = text.lower().strip()
    if row['category'] == 'arithmetic':
        numbers = re.findall(r'(?<![a-z])[-+]?\d+(?:\.\d+)?', text)
        return bool(numbers and float(numbers[-1]) == float(row['expected']))
    if row['category'] == 'definition':
        return any(s.lower() in text for s in row['alternatives'])
    return any(re.search(r'\b' + re.escape(s.lower()) + r'\b', text) for s in row['alternatives'])


@torch.inference_mode()
def generate(model, tok, rows, path, max_new_tokens):
    model.eval()
    results = [None] * len(rows)
    groups = collections.defaultdict(list)
    for i, row in enumerate(rows):
        ids = tok(render(tok, [dict(role='user', content=row['question'])], True), add_special_tokens=False).input_ids
        assert len(ids) + max_new_tokens < 340
        groups[len(ids)].append((i, ids))
    for group in groups.values():
        for offset in range(0, len(group), 8):
            batch = group[offset:offset + 8]
            x = torch.tensor([ids for _, ids in batch], device='cuda')
            with torch.autocast('cuda', dtype=torch.bfloat16):
                ys = model.generate(x, max_new_tokens=max_new_tokens, do_sample=False, temperature=1., top_p=1., top_k=0, repetition_penalty=1., eos_token_id=tok.eos_token_id)
            for (i, _), tokens in zip(batch, ys[:, x.shape[1]:].tolist()):
                finished = tok.eos_token_id in tokens
                if finished:
                    tokens = tokens[:tokens.index(tok.eos_token_id)]
                text = tok.decode(tokens, skip_special_tokens=True)
                words = re.findall(r'\w+', text.lower())
                grams = [tuple(words[j:j+4]) for j in range(len(words)-3)]
                filler = bool(re.search(r'interesting question|thank you for|let (?:me|us)|happy to help|good question|of course|once again|to summarize|in conclusion|as i mentioned', text, flags=re.I))
                result = dict(question=rows[i]['question'], generated=text, tokens=len(tokens), hit_limit=not finished, filler=filler, repeated_fourgrams=len(grams) - len(set(grams)))
                if 'category' in rows[i]:
                    result.update(category=rows[i]['category'], expected=rows[i]['expected'], automatic_correct=generation_correct(rows[i], text))
                results[i] = result
    jsonl(path, results)
    summary = dict(n=len(rows), mean_tokens=sum(r['tokens'] for r in results) / len(rows), limit_count=sum(r['hit_limit'] for r in results), filler_count=sum(r['filler'] for r in results), repeated_fourgram_count=sum(r['repeated_fourgrams'] for r in results))
    if 'automatic_correct' in results[0]:
        summary.update(automatic_correct=sum(r['automatic_correct'] for r in results), automatic_accuracy=sum(r['automatic_correct'] for r in results) / len(rows), by_category={c: dict(n=sum(r['category'] == c for r in results), correct=sum(r['automatic_correct'] for r in results if r['category'] == c)) for c in sorted({r['category'] for r in results})})
    print(path.name, json.dumps(summary), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    assert torch.cuda.is_available()
    torch.set_num_threads(4)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    random.seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    shutil.copy2(Path(__file__), out / 'executed_dpo_concise.py')
    splits = make_data()
    for split, rows in splits.items():
        jsonl(out / f'{split}.jsonl', rows)
    tok = AutoTokenizer.from_pretrained(args.base, local_files_only=True)
    cfg = MiniMindConfig(**json.loads((args.base / 'config.json').read_text()))
    policy, reference = MiniMindForCausalLM(cfg), MiniMindForCausalLM(cfg)
    for model in (policy, reference):
        missing, unexpected = load_model(model, args.base / 'model.safetensors', strict=True)
        assert not missing and not unexpected
        model.cuda()
    reference.eval().requires_grad_(False)
    assert all(p.data_ptr() != q.data_ptr() for p, q in zip(policy.parameters(), reference.parameters()))
    assert statehash(policy) == statehash(reference)
    ref_hash = statehash(reference)
    arrays, length = encode(tok, splits)
    metadata = dict(repo=REPO, revision=REVISION, base_sha256=sha(args.base / 'model.safetensors'), python=platform.python_version(), torch=torch.__version__, transformers=transformers.__version__, cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0), epochs=2, learning_rate=5e-7, beta=.1, batch_pairs=4, seed=42, weight_decay=0., optimizer='AdamW betas=(0.9,0.999)', grad_clip=1., schedule='constant', parameters=sum(p.numel() for p in policy.parameters()), trainable_parameters=sum(p.numel() for p in policy.parameters() if p.requires_grad), precision='FP32 parameters and optimizer, BF16 autocast, FP32 log_softmax', max_pair_tokens=length, generation_limit=96, control_generation_limit=128, fixed_reference=True, full_parameter_dpo=True, source_sha256=sha(Path(__file__)), data_sha256={s: sha(out / f'{s}.jsonl') for s in splits}, checkpoint_selection='last epoch; validation diagnostic only', checks=dict(pair_masks_and_eos=True, no_truncation=True, prompt_and_fact_disjoint=True, correct_answer_shared_by_chosen_rejected=True))
    dump(out / 'metadata.json', metadata)
    controls = [dict(question=q) for q in CONTROLS]
    summary = {'before_pairs': {}, 'before_generation': {}, 'history': []}
    ref_scores, counts = {}, {}
    for split in splits:
        ref_scores[split], counts[split] = score_pairs(reference, arrays[split], tok, length)
        p, n = score_pairs(policy, arrays[split], tok, length)
        assert torch.equal(p, ref_scores[split]) and torch.equal(n, counts[split])
        summary['before_pairs'][split] = pair_metrics(p, ref_scores[split], n, .1)
        dump(out / f'reference_scores_{split}.json', dict(scores=p.tolist(), counts=n.tolist()))
    for split in ['train', 'test']:
        summary['before_generation'][split] = generate(reference, tok, splits[split], out / f'before_{split}_generation.jsonl', 96)
    summary['before_generation']['controls'] = generate(reference, tok, controls, out / 'before_controls.jsonl', 128)
    dump(out / 'summary.json', summary)
    policy.train()
    p, mask = token_logps(policy, arrays['train'][:4], tok.pad_token_id, length)
    with torch.no_grad():
        r, rm = token_logps(reference, arrays['train'][:4], tok.pad_token_id, length)
    assert torch.equal(mask, rm)
    smoke = dpo_loss(r, p, mask, .1)
    assert abs(smoke.item() - math.log(2)) < 1e-6
    smoke.backward()
    norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.)
    assert torch.isfinite(norm) and norm > 0
    assert all(p.grad is None for p in reference.parameters())
    # Check the native objective's direction with independent score variables.
    synthetic = torch.tensor([[-10.], [-12.]], requires_grad=True)
    dpo_loss(synthetic.detach(), synthetic, torch.ones_like(synthetic), .1).backward()
    assert synthetic.grad[0].item() < 0 < synthetic.grad[1].item()
    dump(out / 'smoke.json', dict(identical_policy_reference_loss=smoke.item(), expected=math.log(2), real_gradient_norm=norm.item(), native_dpo_gradient_direction_correct=True))
    policy.zero_grad(set_to_none=True)
    del p, r, rm, mask, smoke
    optimizer = torch.optim.AdamW(policy.parameters(), lr=5e-7, weight_decay=0.)
    started, step = time.monotonic(), 0
    for epoch in [1, 2]:
        policy.train()
        order = torch.randperm(100, generator=torch.Generator().manual_seed(42 + epoch)).tolist()
        losses = []
        for offset in range(0, 100, 4):
            batch = [arrays['train'][i] for i in order[offset:offset+4]]
            optimizer.zero_grad(set_to_none=True)
            p, mask = token_logps(policy, batch, tok.pad_token_id, length)
            with torch.no_grad():
                r, rm = token_logps(reference, batch, tok.pad_token_id, length)
            assert torch.equal(mask, rm)
            loss = dpo_loss(r, p, mask, .1)
            assert torch.isfinite(loss)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.)
            assert torch.isfinite(norm)
            optimizer.step()
            assert all(p.grad is None for p in reference.parameters())
            step += 1
            losses.append(loss.item())
            with (out / 'training.jsonl').open('a') as f:
                f.write(json.dumps(dict(epoch=epoch, step=step, loss=loss.item(), grad_norm=norm.item())) + '\n')
        scores, n = score_pairs(policy, arrays['validation'], tok, length)
        record = dict(epoch=epoch, mean_training_dpo_loss=sum(losses)/len(losses), validation=pair_metrics(scores, ref_scores['validation'], n, .1))
        summary['history'].append(record)
        print('epoch', json.dumps(record), flush=True)
        dump(out / 'summary.json', summary)
    metadata['training_seconds_including_validation'] = time.monotonic() - started
    assert step == 50 and all(torch.isfinite(p).all().item() for p in policy.parameters())
    torch.save({k: v.detach().cpu() for k, v in policy.state_dict().items()}, out / 'dpo_concise_768.pth')
    saved = torch.load(out / 'dpo_concise_768.pth', map_location='cpu', weights_only=True)
    assert all(torch.equal(saved[k], v.cpu()) for k, v in policy.state_dict().items())
    score_before_reload, _ = score_pairs(policy, arrays['test'], tok, length)
    policy.load_state_dict(saved, strict=True)
    del saved
    score_after_reload, _ = score_pairs(policy, arrays['test'], tok, length)
    assert torch.equal(score_before_reload, score_after_reload)
    assert statehash(reference) == ref_hash
    changed = sum((p.detach() != r).sum().item() for p, r in zip(policy.parameters(), reference.parameters()))
    delta = math.sqrt(sum((p.detach() - r).square().sum().item() for p, r in zip(policy.parameters(), reference.parameters())))
    assert changed > 0 and delta > 0
    metadata.update(optimizer_steps=step, reference_unchanged=True, finite_weights=True, strict_reload_weights_and_scores_equal=True, changed_parameter_elements=changed, parameter_delta_l2=delta, checkpoint_sha256=sha(out / 'dpo_concise_768.pth'), peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated())
    dump(out / 'metadata.json', metadata)
    summary['after_pairs'], summary['after_generation'] = {}, {}
    for split in splits:
        scores, n = score_pairs(policy, arrays[split], tok, length)
        summary['after_pairs'][split] = pair_metrics(scores, ref_scores[split], n, .1)
        dump(out / f'policy_scores_{split}.json', dict(scores=scores.tolist(), counts=n.tolist()))
    for split in ['train', 'test']:
        summary['after_generation'][split] = generate(policy, tok, splits[split], out / f'after_{split}_generation.jsonl', 96)
    summary['after_generation']['controls'] = generate(policy, tok, controls, out / 'after_controls.jsonl', 128)
    b, a = summary['before_generation']['test'], summary['after_generation']['test']
    summary['automatic_criteria'] = dict(test_correctness_not_decreased=a['automatic_correct'] >= b['automatic_correct'], test_filler_decreased=a['filler_count'] < b['filler_count'], test_repetition_decreased=a['repeated_fourgram_count'] < b['repeated_fourgram_count'], test_mean_length_decreased=a['mean_tokens'] < b['mean_tokens'])
    dump(out / 'summary.json', summary)
    print('DONE', json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
