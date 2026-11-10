import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import copy
    import json
    import math
    import sys
    from datetime import datetime
    from pathlib import Path
    from uuid import uuid4
    from zoneinfo import ZoneInfo

    import datasets  # 원본 trainer처럼 Windows에서는 torch보다 먼저 읽습니다.
    import marimo as mo
    import matplotlib.pyplot as plt
    import torch
    import torch.nn.functional as F
    from tokenizers import Tokenizer, models, pre_tokenizers
    from torch.utils.data import DataLoader
    from transformers import PreTrainedTokenizerFast

    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root / "projects/minimind"))
    from dataset.lm_dataset import PretrainDataset
    from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
    from trainer.trainer_utils import SkipBatchSampler, get_lr, init_model, lm_checkpoint

    run_id = datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y%m%d-%H%M%S") + f"-{uuid4().hex[:6]}-pretraining"
    output_dir = repo_root / "outputs" / "notebooks" / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    # 원본 PretrainDataset의 로컬 JSON 로딩 캐시도 실행 산출물 안에 둡니다.
    datasets.config.HF_DATASETS_CACHE = output_dir / "dataset-cache"
    datasets.disable_progress_bars()
    torch.set_num_threads(2)
    return (
        DataLoader, F, MiniMindConfig, MiniMindForCausalLM, PreTrainedTokenizerFast,
        PretrainDataset, SkipBatchSampler, Tokenizer, copy, get_lr, init_model,
        json, lm_checkpoint, math, mo, models, output_dir, plt, pre_tokenizers, torch, uuid4,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # MiniMind 사전 학습: 원문에서 다음 토큰 예측까지

    **목표:** `trainer/train_pretrain.py`의 학습 흐름과 `dataset/lm_dataset.py`의
    **실제 PretrainDataset**을 작은 데이터로 연결합니다. 모델 구조를 읽은 뒤 진행하면 좋습니다.
    Python 함수, 텐서 shape, 행렬 곱을 알고 시작하세요. 모델·데이터 다운로드나 GPU는 필요 없습니다.

    이 실습은 질문/답변 쌍 대신 **일반 문장의 다음 토큰**을 학습합니다.
    원문 자체가 정답을 제공하는 자기지도 학습입니다. 모델은 무작위 가중치에서 시작합니다.
    이 저장소의 SFT는 대화에서 주로 assistant 응답 위치를 학습한다는 데이터·마스킹 차이가 있습니다.
    여기서 다루는 것은 자기회귀 사전 학습이며, 사전 학습의 모든 종류를 다루지는 않습니다.

    **학습 파이프라인의 첫 노트북**입니다. 여기서 정리하는 정답 이동·패딩 마스크·기울기 누적·
    학습률·체크포인트는 SFT·LoRA·증류·DPO 노트북에서 다시 자세히 설명하지 않습니다.
    아래 데이터 설정을 바꾸면 관련 학습을 새 모델에서 시작합니다.
    원본 trainer 전체를 실행하거나 원래 모델·데이터 파일을 덮어쓰지 않습니다.

    산출물 저장 위치: `outputs/notebooks/<실행 ID>/`
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## 1. 일반 문장으로 JSONL 데이터 만들기

    한 줄이 `{"text": "고양이 는 야옹 한다 ."}`인 원본 형식입니다.
    아래 12개 조합 중 8개는 학습, 4개는 검증으로 분리합니다.
    검증에는 학습에 없던 수식어·동물 조합이 있지만 단어 자체는 모두 학습에 있습니다.
    검증 문장으로 optimizer를 갱신하지 않습니다.
    **예측:** 학습 문장을 잘 외운 것과 새로운 조합을 잘 이어 쓰는 것은 같은가요?
    """)
    return


@app.cell
def _(json, output_dir):
    animal_sounds = [("고양이", "야옹"), ("강아지", "멍멍"), ("소", "음메"), ("오리", "꽥꽥")]
    held_out_pairs = {("큰", "고양이"), ("작은", "강아지"), ("큰", "소"), ("작은", "오리")}
    train_texts, validation_texts = [], []
    for _animal, _sound in animal_sounds:
        for _adjective in ["", "작은", "큰"]:
            _text = f"{_adjective} {_animal} 는 {_sound} 한다 .".strip()
            if (_adjective, _animal) in held_out_pairs:
                validation_texts.append(_text)
            else:
                train_texts.append(_text)
    train_path, validation_path = output_dir / "train.jsonl", output_dir / "validation.jsonl"
    for _path, _texts in [(train_path, train_texts), (validation_path, validation_texts)]:
        _path.write_text("".join(json.dumps({"text": t}, ensure_ascii=False) + "\n" for t in _texts), encoding="utf-8")
    assert len(train_texts) == 8 and len(validation_texts) == 4
    assert not set(train_texts) & set(validation_texts)
    return (animal_sounds, train_path, train_texts, validation_path, validation_texts,)


@app.cell(hide_code=True)
def _(mo, train_texts, validation_texts):
    # ▸ 결과 표시 — 실행하면 표·그림이 나타납니다 (marimo 표시 코드)
    mo.ui.table([
        {"분할": split, "text": text}
        for split, texts in [("학습", train_texts), ("검증", validation_texts)] for text in texts
    ], selection=None)
    return


@app.cell
def _(mo):
    mo.md("""
    ## 2. 눈으로 읽을 수 있는 작은 토크나이저

    원본은 저장된 토크나이저를 불러옵니다. 여기서는 **학습 텍스트만** 사용해 공백 단위 사전을 만듭니다.
    실제 PreTrainedTokenizerFast를 로컬에 저장하여 원본 init_model에서도 읽을 수 있게 합니다.
    실제 대규모 학습의 서브워드 토크나이저를 단순화한 것입니다.
    PAD=0, BOS=1, EOS=2, UNK=3이며, 사전에 없는 단어는 UNK입니다.
    **예측:** 새 단어가 UNK가 되었다면 생성 결과로 그 단어의 학습 여부를 판단할 수 있을까요?
    """)
    return


@app.cell
def _(PreTrainedTokenizerFast, Tokenizer, models, output_dir, pre_tokenizers, train_texts, validation_texts):
    vocabulary = ["[PAD]", "[BOS]", "[EOS]", "[UNK]"] + sorted({w for text in train_texts for w in text.split()})
    _backend = Tokenizer(models.WordLevel({word: i for i, word in enumerate(vocabulary)}, unk_token="[UNK]"))
    _backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=_backend, pad_token="[PAD]", bos_token="[BOS]",
        eos_token="[EOS]", unk_token="[UNK]", model_max_length=64,
        clean_up_tokenization_spaces=False,
    )
    tokenizer_dir = output_dir / "tokenizer"
    tokenizer.save_pretrained(tokenizer_dir)
    assert all(tokenizer.unk_token_id not in tokenizer(t, add_special_tokens=False).input_ids for t in validation_texts)
    assert tokenizer.unk_token_id in tokenizer("새로운 고양이", add_special_tokens=False).input_ids
    return (tokenizer, tokenizer_dir, vocabulary,)


@app.cell(hide_code=True)
def _(mo, tokenizer, vocabulary):
    # ▸ 결과 표시 — 실행하면 표·그림이 나타납니다 (marimo 표시 코드)
    mo.vstack([
        mo.ui.table([{"ID": i, "단어": w} for i, w in enumerate(vocabulary)], selection=None),
        mo.md(f"검증 문장의 단어는 모두 사전에 있습니다. `새로운 고양이` → `{tokenizer('새로운 고양이', add_special_tokens=False).input_ids}`. **해설:** UNK는 단어 구분을 잃으므로 단어 습득 평가로 쓰지 않습니다."),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    # ▸ 실습 설정 위젯 — 실행하면 슬라이더·선택 상자가 나타납니다 (marimo 표시 코드)
    max_seq_len = mo.ui.dropdown([6, 8, 10], value=10, label="고정 시퀀스 길이")
    training_epochs = mo.ui.slider(10, 50, step=10, value=40, label="학습 epoch")
    accumulation_steps = mo.ui.dropdown([1, 2, 3], value=2, label="기울기 누적 microbatch 수")
    mo.hstack([max_seq_len, training_epochs, accumulation_steps])
    return accumulation_steps, max_seq_len, training_epochs


@app.cell
def _(mo):
    mo.md("""
    ## 3. 실제 PretrainDataset: 자르기 → BOS/EOS → 패딩 → labels

    **예측:** max_length=6이면 본문은 최대 몇 토큰을 남길까요?
    원본은 `max_length-2`개로 자른 뒤 BOS/EOS를 붙입니다. 빈자리는 PAD,
    labels의 PAD만 -100으로 바꿉니다. labels를 미리 한 칸 이동하지 않습니다.
    잘린 본문 뒤에도 EOS가 붙으므로, 길이 설정은 학습할 문장 내용도 바꿉니다.
    """)
    return


@app.cell
def _(DataLoader, PretrainDataset, max_seq_len, tokenizer, train_path, validation_path):
    train_dataset = PretrainDataset(str(train_path), tokenizer, max_length=max_seq_len.value)
    validation_dataset = PretrainDataset(str(validation_path), tokenizer, max_length=max_seq_len.value)
    input_ids, labels = next(iter(DataLoader(train_dataset, batch_size=2, shuffle=False, num_workers=0)))
    assert input_ids.shape == labels.shape == (2, max_seq_len.value)
    assert (labels[input_ids == tokenizer.pad_token_id] == -100).all()
    return (input_ids, labels, train_dataset, validation_dataset,)


@app.cell(hide_code=True)
def _(input_ids, labels, max_seq_len, mo, tokenizer):
    # ▸ 결과 표시 — 실행하면 표·그림이 나타납니다 (marimo 표시 코드)
    mo.vstack([
        mo.md(f"원본 Dataset → DataLoader 배치 shape: `{tuple(input_ids.shape)}`. dtype: `{input_ids.dtype}`."),
        mo.ui.table([
            {"샘플": b, "위치": t, "토큰": tokenizer.convert_ids_to_tokens(int(input_ids[b, t])),
             "input ID": int(input_ids[b, t]), "label (아직 이동 안 함)": int(labels[b, t])}
            for b in range(2) for t in range(max_seq_len.value)
        ], selection=None),
    ])
    return


@app.cell
def _(mo):
    mo.md("""
    ## 4. 모델 안에서 정답을 이동한다

    `MiniMindForCausalLM.forward`는 `logits[:, :-1]`과 `labels[:, 1:]`를 비교합니다.
    [BOS] 출력은 첫 본문 토큰을, 마지막 본문 출력은 EOS를 예측합니다.
    PAD 정답은 무시하고, BOS 자체를 예측하는 위치는 없습니다.
    -100은 **정답 무시 값**이며 Embedding에 넣는 입력 토큰 ID가 아닙니다.

    이 원본 trainer는 attention_mask를 전달하지 않습니다. 현재의 **오른쪽 패딩과 causal Attention**에서는
    앞의 실제 토큰이 뒤의 PAD를 보지 못합니다. labels 마스킹과 Attention 마스킹은 별개의 역할입니다.
    이 설명을 왼쪽 패딩이나 비인과 모델에 그대로 적용하면 안 됩니다.
    """)
    return


@app.cell
def _(MiniMindConfig, init_model, tokenizer, tokenizer_dir, torch):
    lm_config = MiniMindConfig(
        hidden_size=32, num_hidden_layers=2, intermediate_size=64,
        num_attention_heads=4, num_key_value_heads=2, vocab_size=len(tokenizer),
        max_position_embeddings=64, dropout=0, flash_attn=False, use_moe=False,
        bos_token_id=tokenizer.bos_token_id, eos_token_id=tokenizer.eos_token_id,
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        initial_model, _loaded_tokenizer = init_model(
            lm_config, from_weight="none", tokenizer_path=str(tokenizer_dir), device="cpu",
        )
    initial_model.eval()
    assert _loaded_tokenizer.get_vocab() == tokenizer.get_vocab()
    return initial_model, lm_config


@app.cell
def _(F, initial_model, input_ids, labels, tokenizer, torch):
    with torch.no_grad():
        _out = initial_model(input_ids, labels=labels)
        shifted_targets = labels[:, 1:]
        valid_targets = shifted_targets != -100
        per_position = F.cross_entropy(
            _out.logits[:, :-1].reshape(-1, len(tokenizer)), shifted_targets.reshape(-1),
            ignore_index=-100, reduction="none",
        ).reshape_as(shifted_targets)
        manual_loss = per_position[valid_targets].mean()
        torch.testing.assert_close(manual_loss, _out.loss)
        _changed_padding = input_ids.clone()
        _changed_padding[input_ids == tokenizer.pad_token_id] = tokenizer.unk_token_id
        _changed_out = initial_model(_changed_padding, labels=labels)
        torch.testing.assert_close(_out.logits[:, :-1][valid_targets], _changed_out.logits[:, :-1][valid_targets])
        torch.testing.assert_close(_out.loss, _changed_out.loss)
    return (manual_loss, per_position, shifted_targets, valid_targets,)


@app.cell(hide_code=True)
def _(input_ids, manual_loss, mo, per_position, shifted_targets, tokenizer, valid_targets):
    # ▸ 결과 표시 — 실행하면 표·그림이 나타납니다 (marimo 표시 코드)
    mo.vstack([
        mo.ui.table([
            {"입력 위치 토큰": tokenizer.convert_ids_to_tokens(int(input_ids[0, t])),
             "다음 정답": "무시" if int(shifted_targets[0, t]) == -100 else tokenizer.convert_ids_to_tokens(int(shifted_targets[0, t])),
             "평균에 포함": bool(valid_targets[0, t]), "위치별 손실": float(per_position[0, t])}
            for t in range(input_ids.shape[1] - 1)
        ], selection=None),
        mo.md(f"유효한 정답 {valid_targets.sum().item()}개만 평균: `{float(manual_loss):.4f}`. 원본 loss와 일치합니다. **관찰:** 뒤의 PAD 입력을 바꾸어도 유효 예측과 손실은 같습니다. 길이 6 설정에는 PAD가 없을 수 있으므로 표를 함께 확인하세요."),
    ])
    return


@app.cell
def _(mo):
    mo.md("""
    ## 5. 기울기 누적: microbatch와 optimizer step 구분

    microbatch는 한 forward/backward의 묶음, optimizer step은 가중치를 실제 갱신하는 시점입니다.
    두 microbatch의 loss를 각각 2로 나눠 backward하면 gradient를 더한 뒤 한 번 갱신할 수 있습니다.
    **예측:** 모든 microbatch의 유효 정답 수가 같고 dropout=0이면 큰 배치와 같은 gradient일까요?

    단, 원본은 항상 `loss / accumulation_steps`입니다. 마지막 그룹이 덜 차도 같은 값으로 나누고 갱신합니다.
    따라서 불완전 그룹은 완전한 큰 배치 평균과 다릅니다. 패딩 때문에 유효 토큰 수가 서로 달라도
    'microbatch별 평균의 평균'과 '모든 토큰의 평균'은 다를 수 있습니다.
    아래는 같은 길이의 두 문장으로 동등 조건을 먼저 검증합니다.
    """)
    return


@app.cell
def _(copy, initial_model, torch, train_dataset):
    _by_length = {}
    for _i in range(len(train_dataset)):
        _x, _y = train_dataset[_i]
        _by_length.setdefault(int((_y[1:] != -100).sum()), []).append((_x, _y))
    _group = next(rows[:2] for rows in _by_length.values() if len(rows) >= 2)
    _x = torch.stack([r[0] for r in _group])
    _y = torch.stack([r[1] for r in _group])
    _large = copy.deepcopy(initial_model).train()
    _micro = copy.deepcopy(initial_model).train()
    _large(_x, labels=_y).loss.backward()
    for _i in range(2):
        (_micro(_x[_i:_i+1], labels=_y[_i:_i+1]).loss / 2).backward()
    accumulation_max_error = 0.0
    for _p, _q in zip(_large.parameters(), _micro.parameters()):
        torch.testing.assert_close(_p.grad, _q.grad, atol=1e-6, rtol=1e-4)
        accumulation_max_error = max(accumulation_max_error, float((_p.grad - _q.grad).abs().max()))
    _full = copy.deepcopy(initial_model).train()
    _tail = copy.deepcopy(initial_model).train()
    _full(_x[:1], labels=_y[:1]).loss.backward()
    (_tail(_x[:1], labels=_y[:1]).loss / 3).backward()
    for _p, _q in zip(_full.parameters(), _tail.parameters()):
        torch.testing.assert_close(_p.grad / 3, _q.grad, atol=1e-6, rtol=1e-4)
    return (accumulation_max_error,)


@app.cell(hide_code=True)
def _(accumulation_max_error, mo):
    # ▸ 결과 설명 — 실행하면 계산 값이 들어간 설명이 나타납니다 (marimo 표시 코드)
    mo.md(f"같은 길이의 큰 배치와 누적 gradient 최대 오차: `{accumulation_max_error:.2e}`. **해설:** A=3인데 마지막 한 microbatch만 남은 원본 방식에서는 그 배치 단독 gradient의 1/3입니다. 아래 학습은 이 원본 동작을 유지하며 그룹 크기를 기록합니다. 토큰 수 가중 평균 방식으로 바꾸려면 유효 정답 수에 따라 loss를 가중해야 합니다.")
    return


@app.cell
def _(DataLoader, math, torch):
    def evaluate_pretraining(model, dataset):
        was_training = model.training
        model.eval()
        total_nll, total_tokens, correct = 0.0, 0, 0
        with torch.no_grad():
            for ids, targets in DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0):
                output = model(ids, labels=targets)
                valid = targets[:, 1:] != -100
                count = int(valid.sum())
                total_nll += float(output.loss) * count
                total_tokens += count
                correct += int(((output.logits[:, :-1].argmax(-1) == targets[:, 1:]) & valid).sum())
        model.train(was_training)
        loss = total_nll / total_tokens
        return {"loss": loss, "perplexity": math.exp(loss), "token_accuracy": correct / total_tokens,
                "valid_targets": total_tokens}
    return (evaluate_pretraining,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 6. 원본 흐름으로 작은 모델 사전 학습

    <img src="public/diagrams/pretraining.png" alt="사전 학습 루프: 누적 경계에서만 optimizer step" width="280">

    Dense 모델이므로 aux_loss=0이지만 원본의 합산 위치를 유지합니다.
    CPU float32이므로 원본의 AMP/GradScaler는 생략하고, DDP·compile·wandb도 사용하지 않습니다.
    cosine 학습률은 원본 get_lr를 호출하며 **microstep마다** 바뀝니다. warmup은 없는 원본 수식입니다.

    $$\mathrm{lr}(s)=\mathrm{lr}_{0}\left(0.1+0.45\left(1+\cos\frac{\pi s}{S}\right)\right)$$

    s=0에서 lr₀, s=S(전체 microstep)에서 0.1·lr₀입니다. 0까지 내려가지 않는 바닥값이 있습니다.
    실제 optimizer step에는 그 누적 그룹의 마지막 microstep에서 설정한 학습률이 적용됩니다.
    gradient clipping은 optimizer step 직전의 누적 gradient 크기를 제한합니다.
    학습·검증 평가는 유효 토큰 수로 가중 평균하고, 검증에서는 backward를 하지 않습니다.
    """)
    return


@app.cell
def _(DataLoader, SkipBatchSampler, accumulation_steps, copy, evaluate_pretraining, get_lr, initial_model, output_dir, plt, torch, train_dataset, training_epochs, uuid4, validation_dataset):
    training_dir = output_dir / f"training-{uuid4().hex[:8]}"
    training_dir.mkdir()
    pretrained_model = copy.deepcopy(initial_model).train()
    pretrain_optimizer = torch.optim.AdamW(pretrained_model.parameters(), lr=0.005, weight_decay=0.01)
    pretrain_optimizer.zero_grad(set_to_none=True)
    initial_train_metrics = evaluate_pretraining(pretrained_model, train_dataset)
    initial_validation_metrics = evaluate_pretraining(pretrained_model, validation_dataset)
    _iters = (len(train_dataset) + 1) // 2
    _total_microsteps = training_epochs.value * _iters
    training_trace, epoch_metrics = [], []
    optimizer_updates = 0
    for _epoch in range(training_epochs.value):
        _generator = torch.Generator().manual_seed(42 + _epoch)
        _indices = torch.randperm(len(train_dataset), generator=_generator).tolist()
        _loader = DataLoader(train_dataset, batch_sampler=SkipBatchSampler(_indices, batch_size=2, skip_batches=0), num_workers=0)
        _group_size = 0
        for _step, (_ids, _labels) in enumerate(_loader, start=1):
            _group_size += 1
            _global_step = _epoch * _iters + _step
            _lr = get_lr(_global_step, _total_microsteps, 0.005)
            for _param_group in pretrain_optimizer.param_groups:
                _param_group["lr"] = _lr
            _result = pretrained_model(_ids, labels=_labels)
            _raw_loss = _result.loss + _result.aux_loss
            _backward_loss = _raw_loss / accumulation_steps.value
            assert torch.isfinite(_backward_loss) and float(_result.aux_loss.detach()) == 0
            _backward_loss.backward()
            _update_now = _step % accumulation_steps.value == 0 or _step == len(_loader)
            _before_clip, _after_clip = None, None
            if _update_now:
                _before_clip = float(torch.nn.utils.clip_grad_norm_(pretrained_model.parameters(), 1.0))
                _after_clip = float(torch.stack([p.grad.norm() for p in pretrained_model.parameters() if p.grad is not None]).norm())
                assert _after_clip <= 1.00001
                pretrain_optimizer.step()
                pretrain_optimizer.zero_grad(set_to_none=True)
                optimizer_updates += 1
            training_trace.append({"epoch": _epoch + 1, "microstep": _step, "global_microstep": _global_step,
                                   "lr": _lr, "raw_loss": float(_raw_loss.detach()), "backward_loss": float(_backward_loss.detach()),
                                   "valid_targets": int((_labels[:, 1:] != -100).sum()), "update": _update_now,
                                   "optimizer_steps_so_far": optimizer_updates, "group_size": _group_size,
                                   "grad_norm_before_clip": _before_clip, "grad_norm_after_clip": _after_clip})
            if _update_now:
                _group_size = 0
        _train = evaluate_pretraining(pretrained_model, train_dataset)
        _validation = evaluate_pretraining(pretrained_model, validation_dataset)
        epoch_metrics.append({"epoch": _epoch + 1, "train_loss": _train["loss"], "validation_loss": _validation["loss"]})
    pretrained_model.eval()
    final_train_metrics = evaluate_pretraining(pretrained_model, train_dataset)
    final_validation_metrics = evaluate_pretraining(pretrained_model, validation_dataset)
    assert final_train_metrics["loss"] < initial_train_metrics["loss"]
    pretraining_metrics = {"initial_train": initial_train_metrics, "final_train": final_train_metrics,
                           "initial_validation": initial_validation_metrics, "final_validation": final_validation_metrics,
                           "microsteps": len(training_trace), "optimizer_steps": optimizer_updates,
                           "accumulation_steps": accumulation_steps.value, "epochs": training_epochs.value,
                           "max_seq_len": train_dataset.max_length}
    pretrain_fig, _axes = plt.subplots(1, 2, figsize=(10, 3))
    _axes[0].plot([r["epoch"] for r in epoch_metrics], [r["train_loss"] for r in epoch_metrics], label="Train")
    _axes[0].plot([r["epoch"] for r in epoch_metrics], [r["validation_loss"] for r in epoch_metrics], label="Validation")
    _axes[0].set(xlabel="Completed epoch", ylabel="Token-mean cross entropy")
    _axes[0].legend()
    _axes[1].plot([r["global_microstep"] for r in training_trace], [r["lr"] for r in training_trace])
    _axes[1].set(xlabel="Microstep", ylabel="Learning rate", title="Original get_lr / no warmup")
    pretrain_fig.tight_layout()
    return (
        epoch_metrics,
        final_train_metrics,
        final_validation_metrics,
        initial_train_metrics,
        initial_validation_metrics,
        optimizer_updates,
        pretrain_fig,
        pretrain_optimizer,
        pretrained_model,
        pretraining_metrics,
        training_dir,
        training_trace,
    )


@app.cell(hide_code=True)
def _(final_train_metrics, final_validation_metrics, initial_train_metrics, initial_validation_metrics, mo, optimizer_updates, pretrain_fig, training_trace):
    # ▸ 결과 표시 — 실행하면 표·그림이 나타납니다 (marimo 표시 코드)
    mo.vstack([
        pretrain_fig,
        mo.ui.table([{"분할": name, **metrics} for name, metrics in [("학습 전 train", initial_train_metrics), ("학습 후 train", final_train_metrics), ("학습 전 validation", initial_validation_metrics), ("학습 후 validation", final_validation_metrics)]], selection=None),
        mo.md(f"microstep {len(training_trace)}회, optimizer step {optimizer_updates}회. 유효 토큰 평균 loss의 exp가 perplexity입니다. 같은 tokenizer·같은 데이터 조건에서 비교하고 언어 능력 전체 점수로 해석하지 않습니다."),
        mo.ui.table(training_trace[:8], selection=None),
        mo.md("**관찰:** update=True일 때만 갱신합니다. A=3이면 epoch 마지막에 group_size=1인 갱신이 나옵니다. 이때도 loss를 3으로 나누는 원본 동작과, clip 전후 norm을 함께 확인하세요."),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7. 문장 이어 쓰기와 검증 해석

    **예측:** 손실 감소가 모든 입력의 정답 생성을 보장할까요? 학습 8개와 검증 4개 원문 각각의 앞부분을 넣습니다. 원문 분할과 생성 결과를 함께 읽으세요.
    """)
    return


@app.cell
def _(animal_sounds, initial_model, pretrained_model, tokenizer, torch, train_texts):
    def complete_pretraining(model, prompt):
        prefix = [tokenizer.bos_token_id] + tokenizer(prompt, add_special_tokens=False).input_ids
        model.eval()
        with torch.no_grad():
            result = model.generate(torch.tensor([prefix]), max_new_tokens=6, do_sample=False,
                                    top_k=0, top_p=1.0, temperature=1.0, eos_token_id=tokenizer.eos_token_id)
        return tokenizer.decode(result[0].tolist(), skip_special_tokens=False)
    generation_rows = []
    for _animal, _sound in animal_sounds:
        for _prefix in [f"{_animal} 는", f"작은 {_animal} 는", f"큰 {_animal} 는"]:
            _full_text = f"{_prefix} {_sound} 한다 ."
            generation_rows.append({"원문 분할": "학습" if _full_text in train_texts else "검증",
                                    "입력": _prefix, "원래 규칙의 다음 단어": _sound,
                                    "학습 전": complete_pretraining(initial_model, _prefix),
                                    "학습 후": complete_pretraining(pretrained_model, _prefix)})
    return (complete_pretraining, generation_rows,)


@app.cell(hide_code=True)
def _(generation_rows, mo):
    # ▸ 결과 표시 — 실행하면 표·그림이 나타납니다 (marimo 표시 코드)
    mo.vstack([
        mo.ui.table(generation_rows, selection=None),
        mo.md("**해설:** greedy 생성은 점수가 가장 큰 다음 토큰을 반복 선택합니다. 학습 손실, 검증 손실, 문장별 결과를 따로 판단하세요. 길이 6으로 잘라 학습했다면 생성 문장이 짧아질 수 있습니다. 이 작은 데이터 실험은 한국어 일반화 성능 평가가 아닙니다."),
    ])
    return


@app.cell
def _(mo):
    mo.md("""
    ## 8. 가중치 파일과 학습 재개 체크포인트

    원본 lm_checkpoint는 float16 가중치와 별도의 resume 파일을 저장합니다.
    가중치만으로는 AdamW의 이동 평균이나 진행 위치를 복원할 수 없습니다.
    실습에서는 실제 함수를 호출해 float32 모델 상태·optimizer·epoch·step과 학습 설정을 저장합니다.
    **예측:** 같은 모델·optimizer 상태에서 같은 추가 배치를 갱신하면 연속 실행과 같을까요?
    아래는 optimizer 갱신 경계에서 저장한 뒤, 같은 LR·배치로 추가 한 step을 비교하는 통제 실험입니다.
    전체 DataLoader 중간 재개나 DDP 재개를 검증하는 실험은 아닙니다.
    """)
    return


@app.cell
def _(MiniMindForCausalLM, accumulation_steps, copy, input_ids, labels, lm_checkpoint, lm_config, pretrain_optimizer, pretrained_model, tokenizer_dir, torch, training_dir, training_epochs, training_trace):
    lm_checkpoint(lm_config, weight="pretrain_lab", model=pretrained_model, optimizer=pretrain_optimizer,
                  epoch=training_epochs.value - 1, step=training_trace[-1]["microstep"], save_dir=str(training_dir),
                  seed=42, accumulation_steps=accumulation_steps.value, tokenizer_path=str(tokenizer_dir))
    _saved = lm_checkpoint(lm_config, weight="pretrain_lab", save_dir=str(training_dir))
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(43)
        _restored = MiniMindForCausalLM(copy.deepcopy(lm_config))
    _restored.load_state_dict(_saved["model"], strict=True)
    _restored_optimizer = torch.optim.AdamW(_restored.parameters(), lr=0.005, weight_decay=0.01)
    _restored_optimizer.load_state_dict(_saved["optimizer"])
    _continuous = copy.deepcopy(pretrained_model)
    _continuous_optimizer = torch.optim.AdamW(_continuous.parameters(), lr=0.005, weight_decay=0.01)
    _continuous_optimizer.load_state_dict(copy.deepcopy(pretrain_optimizer.state_dict()))
    for _model, _optimizer in [(_restored, _restored_optimizer), (_continuous, _continuous_optimizer)]:
        _model.train()
        _optimizer.zero_grad(set_to_none=True)
        _model(input_ids, labels=labels).loss.backward()
        torch.nn.utils.clip_grad_norm_(_model.parameters(), 1.0)
        _optimizer.step()
    resume_max_error = 0.0
    for _a, _b in zip(_restored.parameters(), _continuous.parameters()):
        torch.testing.assert_close(_a, _b, atol=1e-7, rtol=1e-6)
        resume_max_error = max(resume_max_error, float((_a.detach() - _b.detach()).abs().max()))
    checkpoint_rows = [{"파일": p.name, "bytes": p.stat().st_size} for p in sorted(training_dir.glob("*.pth"))]
    return (checkpoint_rows, resume_max_error,)


@app.cell(hide_code=True)
def _(checkpoint_rows, mo, resume_max_error, training_dir):
    # ▸ 결과 표시 — 실행하면 표·그림이 나타납니다 (marimo 표시 코드)
    mo.vstack([
        mo.ui.table(checkpoint_rows, selection=None),
        mo.md(f"동일한 추가 optimizer step 후 최대 가중치 오차: `{resume_max_error:.2e}`. 저장 위치: `{training_dir}`. 노트북의 원래 학습 모델은 복사했으므로 변경하지 않았습니다."),
        mo.md("**해설:** 같은 초기 가중치만이 아니라 optimizer 상태도 필요합니다. 실제 중간 재개에는 셔플 순서·건너뛸 배치·난수 상태·누적 중인 gradient도 고려해야 합니다. 여기서는 dropout=0, 고정 추가 배치, 누적 경계 저장 조건입니다."),
    ])
    return


@app.cell
def _(epoch_metrics, generation_rows, json, pretraining_metrics, resume_max_error, training_dir, training_trace):
    for _name, _value in [("metrics", pretraining_metrics), ("microsteps", training_trace),
                           ("epochs", epoch_metrics), ("generations", generation_rows),
                           ("resume_check", {"max_parameter_error": resume_max_error})]:
        (training_dir / f"{_name}.json").write_text(json.dumps(_value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return


@app.cell
def _(mo):
    mo.md("""
    ## 9. 원본 코드 대응과 직접 해 볼 과제

    | 원본 | 실습 |
    | --- | --- |
    | PretrainDataset.__getitem__ | 3절의 실제 로딩·절단·BOS/EOS·PAD·labels |
    | init_model(from_weight='none') | 저장한 작은 tokenizer로 무작위 모델 생성 |
    | train_epoch의 get_lr·loss/누적 수·clip·step | 6절 CPU 학습 및 microstep 추적 표 |
    | SkipBatchSampler | epoch별 셔플 인덱스를 batch_size=2로 묶기 |
    | lm_checkpoint | 8절의 실제 저장·복원 및 추가 한 step 비교 |
    | AMP·DDP·wandb·torch.compile | 이 CPU 입문 실습에서는 실행하지 않음 |

    1. 길이 10→6으로 바꾸기 전에 본문 최대 토큰 수와 EOS 위치를 쓰세요.
       **해설:** 본문은 최대 4토큰, 그 뒤에 EOS입니다. 일부 원문이 잘린 채 새 학습 목표가 됩니다.
    2. 누적 수 1/2/3에서 microstep 수·optimizer step 수·마지막 그룹 크기를 기록하세요.
       **기준:** epoch당 microstep 4회이며 optimizer step은 각각 4/2/2회입니다. A=3의 마지막 그룹은 1개입니다.
    3. 검증 손실이 낮아졌는데 한 문장 생성이 틀릴 수 있을까요?
       **해설:** 토큰 평균 확률과 greedy 문장 정답은 다른 지표입니다. 둘을 함께 기록하세요.
    4. labels를 미리 한 칸 이동하면 왜 안 될까요? **해설:** 모델 내부에서 또 이동하여 목표가 어긋납니다.
    5. 가중치만 복원했을 때 optimizer step이 같을지 예측하세요.
       **해설:** AdamW 상태가 빠지면 이후 갱신이 달라질 수 있습니다. optimizer 상태와 진행 위치를 구분하세요.

    **기록할 것:** 설정, 유효 정답 수, 학습·검증 loss/perplexity, 생성 결과,
    microstep·optimizer step 수, 체크포인트 경로를 적으세요. 대규모 학습·GPU 성능·일반화는 별도 검증이 필요합니다.

    **다음 실습:** `minimind_sft.py`. 같은 다음 토큰 손실에 **대화 형식과 assistant 정답 마스크**가 더해집니다.
    """)
    return


if __name__ == "__main__":
    app.run()
