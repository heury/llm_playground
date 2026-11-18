"""Fine-tune a pinned native MiniMind-V bundle on generated color/shape QA."""
import argparse
import hashlib
import importlib.util
import json
import random
import re
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--epochs', type=int, default=8)
    p.add_argument('--batch-size', type=int, default=16)
    args = p.parse_args()
    run = args.run_dir.resolve()
    assert not (run / 'completion.json').exists(), 'Completed run must not be overwritten'
    assert torch.cuda.is_available(), 'This experiment requires CUDA'
    random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.set_num_threads(4)
    started = time.time()
    bundle = run / 'base_model'
    spec = importlib.util.spec_from_file_location('released_inference', bundle / 'inference.py')
    inference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(inference)
    model, tok = inference.load_model(bundle, 'cuda')
    question = 'What color and shape is it? Answer with the color and shape only.'
    prompt = tok.apply_chat_template(
        [{'role': 'user', 'content': '<|image_pad|>' * 64 + '\n' + question}],
        tokenize=False, add_generation_prompt=True, open_thinking=False,
    ).replace('<think>\n\n</think>\n\n', '')
    prefix = tok(prompt, add_special_tokens=False).input_ids
    assert prefix.count(12) == 64
    colors = {'red': (230, 35, 35), 'green': (25, 180, 65), 'blue': (35, 75, 230)}
    classes = [f'{color} {shape}' for color in colors for shape in ['circle', 'square']]
    rng = random.Random(42)
    seen = set()
    data = {}
    for split, count in [('train', 600), ('validation', 120), ('test', 120)]:
        rows, pixels, sequences, labels = [], [], [], []
        folder = run / 'data' / split
        folder.mkdir(parents=True, exist_ok=True)
        for i in range(count):
            answer = classes[i % len(classes)]
            color, shape = answer.split()
            while True:
                size = rng.randint(64, 150)
                x, y = rng.randint(20, 236 - size), rng.randint(20, 236 - size)
                image = Image.new('RGB', (256, 256), 'white')
                draw = ImageDraw.Draw(image)
                getattr(draw, 'ellipse' if shape == 'circle' else 'rectangle')(
                    (x, y, x + size, y + size), fill=colors[color])
                digest = hashlib.sha256(image.tobytes()).hexdigest()
                if digest not in seen:
                    seen.add(digest)
                    break
            image.save(folder / f'{i:04d}.png')
            target = tok(answer, add_special_tokens=False).input_ids + [tok.eos_token_id]
            seq = prefix + target
            lab = [-100] * len(prefix) + target
            assert len(seq) < 340
            sequences.append(seq)
            labels.append(lab)
            pixels.append(model.processor(images=image, return_tensors='pt')['pixel_values'][0])
            rows.append(dict(image=f'data/{split}/{i:04d}.png', question=question,
                             answer=answer, pixel_sha256=digest, box=[x, y, x + size, y + size]))
        length = max(map(len, sequences))
        ids = torch.tensor([s + [tok.pad_token_id] * (length - len(s)) for s in sequences])
        labs = torch.tensor([s + [-100] * (length - len(s)) for s in labels])
        mask = torch.tensor([[1] * len(s) + [0] * (length - len(s)) for s in sequences])
        data[split] = dict(rows=rows, ids=ids, labels=labs, mask=mask, pixels=torch.stack(pixels))
        write(run / 'data' / f'{split}.json', rows)
    settings = dict(model='caspar/MiniMind-V-Dense-64M-LLaVA-Short-v1.0',
                    revision='4e6f87f184b27f0692e1f8b3afed959de254894b', seed=42,
                    train=600, validation=120, test=120, question=question,
                    epochs=args.epochs, batch_size=args.batch_size, learning_rate=3e-4,
                    device=torch.cuda.get_device_name(), torch=torch.__version__,
                    trainable='vision_proj only', precision='bfloat16 autocast',
                    selection='lowest validation CE', classes=classes)
    write(run / 'settings.json', settings)

    for name, param in model.named_parameters():
        param.requires_grad_(name.startswith('vision_proj.'))
    trainable = [v for v in model.parameters() if v.requires_grad]
    settings['trainable_parameters'] = sum(v.numel() for v in trainable)
    write(run / 'settings.json', settings)

    def frozen_hash():
        h = hashlib.sha256()
        for name, param in model.named_parameters():
            if not param.requires_grad:
                h.update(name.encode())
                h.update(param.detach().cpu().contiguous().numpy().tobytes())
        return h.hexdigest()

    frozen_before = frozen_hash()

    def loss_batch(d, indices):
        ids, labs, mask, pix = (d[k][indices].cuda() for k in ['ids', 'labels', 'mask', 'pixels'])
        with torch.autocast('cuda', dtype=torch.bfloat16):
            out = model(ids, attention_mask=mask, labels=labs, pixel_values={'pixel_values': pix})
        return out.loss, int((labs[:, 1:] != -100).sum())

    @torch.no_grad()
    def evaluate(split, label, controls=False):
        model.eval()
        d = data[split]
        total, tokens, predictions = 0., 0, []
        black = model.processor(images=Image.new('RGB', (256, 256), 'black'),
                                return_tensors='pt')['pixel_values']
        conditions = ['original', 'black', 'shuffled'] if controls else ['original']
        for start in range(0, len(d['rows']), args.batch_size):
            idx = torch.arange(start, min(start + args.batch_size, len(d['rows'])))
            loss, n = loss_batch(d, idx)
            assert torch.isfinite(loss)
            total += float(loss) * n
            tokens += n
            for condition in conditions:
                if condition == 'original':
                    pix = d['pixels'][idx]
                elif condition == 'black':
                    pix = black.expand(len(idx), -1, -1, -1)
                else:
                    # Balanced alternating classes ensure every replacement has a different label.
                    pix = d['pixels'][(idx + 1) % len(d['rows'])]
                x = torch.tensor([prefix] * len(idx), device='cuda')
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    output = model.generate(x, pixel_values={'pixel_values': pix.cuda()},
                        max_new_tokens=12, do_sample=False, temperature=1., top_p=1., top_k=0,
                        repetition_penalty=1., eos_token_id=tok.eos_token_id, use_cache=True)
                responses = tok.batch_decode(output[:, len(prefix):], skip_special_tokens=True)
                for j, response in zip(idx.tolist(), responses):
                    normal = re.sub(r'[.!\s]+$', '', response.strip().lower())
                    expected = d['rows'][j]['answer']
                    predictions.append(dict(index=j, condition=condition, expected=expected,
                        response=response, correct=normal == expected,
                        color_correct=normal.split()[:1] == expected.split()[:1],
                        shape_correct=normal.split()[-1:] == expected.split()[-1:],
                        matches_swapped_image=(normal == d['rows'][(j+1) % len(d['rows'])]['answer'])
                        if condition == 'shuffled' else None))
        result = dict(ce=total/tokens, target_tokens=tokens)
        for c in conditions:
            rows = [r for r in predictions if r['condition'] == c]
            result[c] = {k: sum(r[k] for r in rows)/len(rows)
                         for k in ['correct', 'color_correct', 'shape_correct']}
            if c == 'shuffled':
                result[c]['matches_swapped_image'] = sum(r['matches_swapped_image'] for r in rows)/len(rows)
        write(run / f'{label}-{split}-predictions.json', predictions)
        write(run / f'{label}-{split}-metrics.json', result)
        print(json.dumps(dict(event='evaluation', label=label, split=split, **result)), flush=True)
        return result

    before = evaluate('test', 'before', controls=True)
    evaluate('validation', 'before')
    optimizer = torch.optim.AdamW(trainable, lr=3e-4, weight_decay=.01)
    best, best_epoch = float('inf'), 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        model.vision_encoder.eval()
        order = torch.randperm(600)
        total, tokens = 0., 0
        for start in range(0, 600, args.batch_size):
            optimizer.zero_grad(set_to_none=True)
            loss, n = loss_batch(data['train'], order[start:start+args.batch_size])
            assert torch.isfinite(loss)
            loss.backward()
            assert all(v.grad is None for v in model.parameters() if not v.requires_grad)
            torch.nn.utils.clip_grad_norm_(trainable, 1., error_if_nonfinite=True)
            optimizer.step()
            total += float(loss.detach()) * n
            tokens += n
            if start % (args.batch_size * 10) == 0:
                print(json.dumps(dict(event='train', epoch=epoch, step=start//args.batch_size,
                                     loss=float(loss.detach()))), flush=True)
        val = evaluate('validation', f'epoch-{epoch:02d}')
        record = dict(epoch=epoch, train_ce=total/tokens, validation=val,
                      elapsed_seconds=time.time()-started)
        with (run / 'training.jsonl').open('a') as f:
            f.write(json.dumps(record)+'\n')
        if val['ce'] < best:
            best, best_epoch = val['ce'], epoch
            torch.save(model.vision_proj.state_dict(), run / 'best-projector.pth')
    model.vision_proj.load_state_dict(torch.load(run / 'best-projector.pth', weights_only=True))
    frozen_after = frozen_hash()
    assert frozen_before == frozen_after, 'Frozen weights changed'
    # Save a native language/projector checkpoint; use the pinned encoder and config alongside it.
    checkpoint = run / 'shapes_vlm_768.pth'
    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()
                if not k.startswith('vision_encoder.')}, checkpoint)
    loaded = torch.load(checkpoint, map_location='cpu', weights_only=True)
    result = model.load_state_dict(loaded, strict=False)
    assert not result.unexpected_keys
    assert all(k.startswith('vision_encoder.') for k in result.missing_keys)
    after = evaluate('test', 'after-reload', controls=True)
    write(run / 'completion.json', dict(before=before, after=after, best_epoch=best_epoch,
          best_validation_ce=best, frozen_weights_unchanged=True,
          frozen_sha256=frozen_after, checkpoint=str(checkpoint),
          peak_gpu_bytes=torch.cuda.max_memory_allocated(), elapsed_seconds=time.time()-started))
    print('COMPLETED', flush=True)


if __name__ == '__main__':
    main()
