import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import copy
    import json
    import random
    import sys
    from datetime import datetime
    from pathlib import Path
    from uuid import uuid4
    from zoneinfo import ZoneInfo

    import datasets  # 원본 trainer의 Windows DLL 우회 순서 유지
    import marimo as mo
    import matplotlib.pyplot as plt
    import torch
    import torch.nn.functional as F
    from tokenizers import Tokenizer, models, pre_tokenizers
    from torch.utils.data import DataLoader
    from transformers import PreTrainedTokenizerFast

    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "minimind"))
    from dataset.lm_dataset import SFTDataset, PretrainDataset, post_processing_chat
    from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
    from trainer.trainer_utils import get_lr

    run_id = datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y%m%d-%H%M%S") + f"-{uuid4().hex[:6]}-sft"
    output_dir = repo_root / "outputs" / "notebooks" / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    datasets.config.HF_DATASETS_CACHE = output_dir / "dataset-cache"
    datasets.disable_progress_bars()
    torch.set_num_threads(2)
    return (
        DataLoader, F, MiniMindConfig, MiniMindForCausalLM, PreTrainedTokenizerFast,
        PretrainDataset, SFTDataset, Tokenizer, copy, get_lr, json, mo, models,
        output_dir, plt, post_processing_chat, pre_tokenizers, random, torch, uuid4,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # MiniMind SFT: 대화에서 무엇을 정답으로 삼을까

    **지도 미세 조정(Supervised Fine-Tuning)**은 질문과 모범 응답으로 모델의 응답 방식을 학습합니다.
    이 실습에서는 **대화 전체를 입력하고 assistant 응답 위치만 손실에 포함**합니다.
    사전 학습과 같은 다음 토큰 예측을 사용하지만, 데이터 형식과 정답 위치가 달라집니다.

    `minimind/trainer/train_full_sft.py`의 학습 순서와
    `minimind/dataset/lm_dataset.py`의 **실제 SFTDataset**을 연결합니다.
    모델 구조 → 사전 학습 → **SFT** → LoRA 순서로 읽으면 좋습니다.
    SFT는 학습 목적이고 Full/LoRA는 업데이트할 매개변수를 정하는 방법입니다.
    여기서는 모델 전체를 학습하는 Full SFT를 사용합니다.

    **순서:** 대화 JSONL → chat template → 정답 마스킹 → 다음 토큰 손실 → 학습 → 평가·생성·저장.
    각 절에서 결과를 먼저 예측하고, 실행한 표와 비교해 이유를 설명하세요.
    Python 함수·텐서 shape·기본 역전파를 알고 시작합니다.
    `@app.cell`은 marimo 셀이며 함수 인자와 return은 셀 사이 데이터를 전달합니다.

    GPU나 다운로드 없이 작은 CPU 모델을 직접 준비합니다. 기준 모델 준비도 이 노트북 안에서
    실행하므로 다른 노트북의 가중치가 필요 없습니다. 작은 합성 과제의 결과를 관찰하는 실습이며
    한국어 범용 모델의 성능을 평가하는 실험은 아닙니다.
    산출물: `outputs/notebooks/<실행 ID>/`
    """)
    return


@app.cell
def _(json, mo, output_dir):
    animals = [("고양이", "야옹"), ("강아지", "멍멍"), ("소", "음메"), ("오리", "꽥꽥")]

    def conversation(question, answer):
        # system을 명시해 원본 pre_processing_chat의 20% 무작위 추가를 피합니다.
        return [
            {"role": "system", "content": "소리 도우미"},
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ]

    train_records = [
        {"conversations": conversation(question, sound)}
        for animal, sound in animals
        for question in [f"{animal} 소리 ?", f"소리 알려 줘 {animal} ?"]
    ]
    validation_records = [
        {"conversations": conversation(f"{animal} 소리 알려 줘 ?", sound)}
        for animal, sound in animals
    ]
    multiturn = conversation("고양이 소리 ?", "야옹") + [
        {"role": "user", "content": "강아지 소리 ?"},
        {"role": "assistant", "content": "멍멍"},
    ]
    train_path = output_dir / "train.jsonl"
    validation_path = output_dir / "validation.jsonl"
    inspection_path = output_dir / "inspection.jsonl"
    for _path, _records in [
        (train_path, train_records), (validation_path, validation_records),
        (inspection_path, [{"conversations": multiturn}]),
    ]:
        _path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in _records), encoding="utf-8")
    mo.vstack([
        mo.md("""
        ## 1. 원본 형식의 대화 JSONL

        한 줄에 `conversations` 배열을 둡니다. `role`은 화자, `content`는 발화입니다.
        학습 질문 8개와 **단어는 같고 순서 조합이 다른 검증 질문 4개**를 분리합니다.
        검증 정답은 평가에만 씁니다. 여러 턴 예제는 마스킹 관찰용이며 학습에는 넣지 않습니다.
        실제 데이터에서 `reasoning_content`, `tools`, `tool_calls`도 지원하지만 여기서는 일반 대화에 집중합니다.
        **예측:** 두 번 답하는 대화에서 마지막 응답만 정답으로 남을까요?
        """),
        mo.ui.table([
            {"분할": split, "질문": r["conversations"][1]["content"], "모범 응답": r["conversations"][2]["content"]}
            for split, records in [("학습", train_records), ("검증", validation_records)] for r in records
        ], selection=None),
        mo.md(f"여러 턴 JSON 예제:\n```json\n{json.dumps({'conversations': multiturn}, ensure_ascii=False, indent=2)}\n```"),
    ])
    return animals, inspection_path, multiturn, train_path, train_records, validation_path, validation_records


@app.cell
def _(PreTrainedTokenizerFast, Tokenizer, mo, models, output_dir, pre_tokenizers, train_records):
    vocabulary = ["[PAD]", "[BOS]", "[EOS]", "[UNK]", "system", "user", "assistant"] + sorted({
        word for record in train_records for message in record["conversations"] for word in message["content"].split()
    })
    _backend = Tokenizer(models.WordLevel({w: i for i, w in enumerate(vocabulary)}, unk_token="[UNK]"))
    _backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=_backend, bos_token="[BOS]", eos_token="[EOS]",
        pad_token="[PAD]", unk_token="[UNK]", clean_up_tokenization_spaces=False,
    )
    tokenizer.chat_template = (
        "{% for message in messages %}"
        "{{ bos_token + message['role'] + '\n' + message['content'] + eos_token + '\n' }}"
        "{% endfor %}"
        "{% if add_generation_prompt %}{{ bos_token + 'assistant\n' }}{% endif %}"
    )
    tokenizer.save_pretrained(output_dir / "tokenizer")
    mo.md("""
    ## 2. chat template: 화자와 발화 경계 표시

    읽기 쉬운 **공백 단위 작은 토크나이저**를 학습 데이터 단어만으로 만듭니다.
    `[BOS]user\\n질문[EOS]\\n[BOS]assistant\\n응답[EOS]\\n` 형식입니다.
    학습에서는 완성된 대화 전체를 넣고 `add_generation_prompt=False`를 사용합니다.
    생성에서는 정답 응답을 빼고 `add_generation_prompt=True`로 assistant 헤더까지 만듭니다.

    원본 SFTDataset은 `BOS + assistant\\n`을 찾아 응답 시작을,
    `EOS + \\n`을 찾아 종료를 판단합니다. **토크나이저와 template의 경계 표현이 일치해야 합니다.**
    이 실습 사전에서는 공백·줄바꿈이 토큰으로 남지 않아 종료 패턴은 `[EOS]` 한 토큰입니다.
    실제 MiniMind 서브워드 사전의 ID·길이와는 다릅니다.
    """)
    return tokenizer, vocabulary


@app.cell
def _(DataLoader, SFTDataset, inspection_path, mo, multiturn, tokenizer, train_path, validation_path):
    train_dataset = SFTDataset(str(train_path), tokenizer, max_length=32)
    validation_dataset = SFTDataset(str(validation_path), tokenizer, max_length=32)
    inspection_dataset = SFTDataset(str(inspection_path), tokenizer, max_length=40)
    inspection_ids, inspection_labels = inspection_dataset[0]
    input_ids, labels = next(iter(DataLoader(train_dataset, batch_size=2, num_workers=0)))
    assert input_ids.shape == labels.shape == (2, 32)
    assert (inspection_labels[inspection_ids == tokenizer.pad_token_id] == -100).all()
    assert tokenizer.decode(inspection_labels[inspection_labels != -100].tolist()) == "야옹 [EOS] 멍멍 [EOS]"
    assert tokenizer.unk_token_id not in input_ids
    mo.vstack([
        mo.md(f"""
        ## 3. 실제 SFTDataset의 정답 마스킹

        `pre_processing_chat → create_chat_prompt → post_processing_chat → tokenize → 절단·패딩 → generate_labels`
        순서입니다. 원본은 system이 없을 때 20% 확률로 system을 넣고, 빈 think 태그도 확률적으로 제거합니다.
        이 예제는 명시적 system과 think 없는 발화를 써서 결과가 일정합니다.

        **해설:** 두 assistant 응답과 각 종료 토큰은 모두 정답으로 남습니다.
        system·user·assistant 헤더·정상 종료 뒤 PAD는 `-100`입니다.
        `-100`은 토큰 ID가 아니라 **손실 계산에서 무시하라는 표시**입니다.
        입력에는 질문도 응답도 그대로 있습니다. 질문은 응답을 예측하는 문맥입니다.
        배치 shape는 `{tuple(input_ids.shape)}` = [B,T], dtype은 `{input_ids.dtype}`입니다.
        ```text
        {inspection_dataset.create_chat_prompt(multiturn)}
        ```
        """),
        mo.ui.table([
            {"위치": t, "입력 토큰": tokenizer.convert_ids_to_tokens(int(x)),
             "input ID": int(x), "label": int(y), "손실 포함": int(y) != -100}
            for t, (x, y) in enumerate(zip(inspection_ids, inspection_labels))
        ], selection=None),
    ])
    return input_ids, inspection_dataset, inspection_ids, inspection_labels, labels, train_dataset, validation_dataset


@app.cell
def _(F, MiniMindConfig, MiniMindForCausalLM, input_ids, labels, mo, tokenizer, torch, vocabulary):
    torch.manual_seed(42)
    config = MiniMindConfig(
        hidden_size=32, num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=2, intermediate_size=64, vocab_size=len(vocabulary),
        max_position_embeddings=64, flash_attn=False, use_moe=False,
        bos_token_id=tokenizer.bos_token_id, eos_token_id=tokenizer.eos_token_id,
    )
    _model = MiniMindForCausalLM(config).eval()
    with torch.no_grad():
        _result = _model(input_ids, labels=labels)
        _manual_loss = F.cross_entropy(
            _result.logits[:, :-1].reshape(-1, len(vocabulary)), labels[:, 1:].reshape(-1), ignore_index=-100,
        )
    assert torch.allclose(_result.loss, _manual_loss)
    mo.vstack([
        mo.md(f"""
        ## 4. 정답은 모델 안에서 한 번 이동한다

        **예측:** `야옹`을 예측하는 logits는 `야옹` 입력 위치일까요, 그 앞 위치일까요?
        원본 모델은 `logits[:, :-1]`과 `labels[:, 1:]`를 비교합니다.
        assistant 헤더의 마지막 토큰 출력이 첫 응답을, 응답 마지막 출력이 EOS를 예측합니다.
        **Dataset에서 미리 이동하면 두 번 이동하므로 틀린 정답을 학습합니다.**
        아래는 첫 샘플에서 손실에 포함되는 쌍입니다. 원본 손실 `{_result.loss.item():.6f}`와
        직접 계산 `{_manual_loss.item():.6f}`가 일치합니다.

        학습 입력에 정답이 있어도 causal attention이 미래를 가립니다.
        이를 teacher forcing이라고 합니다. 생성할 때는 앞서 생성한 토큰을 다음 입력으로 사용합니다.
        마스킹한 질문 위치의 **직접 손실**은 없지만, 응답이 질문 문맥을 참조하므로
        질문 임베딩과 Attention 매개변수에도 gradient가 흐를 수 있습니다.
        """),
        mo.ui.table([
            {"logits 위치": t - 1, "여기까지 읽음(마지막 토큰)": tokenizer.convert_ids_to_tokens(int(input_ids[0, t - 1])),
             "정답 위치": t, "예측할 토큰": tokenizer.convert_ids_to_tokens(int(labels[0, t]))}
            for t in range(1, labels.shape[1]) if labels[0, t] != -100
        ], selection=None),
    ])
    return (config,)


@app.cell
def _(copy, inspection_dataset, inspection_ids, mo, post_processing_chat, tokenizer):
    _full = inspection_ids.tolist()
    _start = next(i for i in range(len(_full)) if _full[i:i + len(inspection_dataset.bos_id)] == inspection_dataset.bos_id)
    _cut = _start + len(inspection_dataset.bos_id)
    # 원본 __getitem__처럼 배열 길이와 max_length를 일치시켜 관찰합니다.
    _probe = copy.copy(inspection_dataset)
    _probe.max_length = _cut
    _short_labels = _probe.generate_labels(_full[:_cut])
    _broken = _full[:_cut + 1] + [tokenizer.pad_token_id] * 3
    _probe.max_length = len(_broken)
    _broken_labels = _probe.generate_labels(_broken)
    assert all(y == -100 for y in _short_labels)
    assert _broken_labels[-1] == tokenizer.pad_token_id
    assert post_processing_chat("<think>\n\n</think>\n\n야옹", remove_empty_think=True) == "야옹"
    mo.vstack([
        mo.md("""
        ## 5. 길이를 줄이면 어떤 정답을 잃을까?

        원본은 **template을 만든 다음 앞에서 max_length까지 자릅니다.**
        응답 시작 전까지 잘리면 유효 정답이 0개이고, 전부 무시된 배치의 평균 CE는 NaN이 될 수 있습니다.
        응답 중간에서 잘리면 일부 응답은 학습하지만 종료 토큰은 학습하지 못합니다.
        학습 전 `labels[:, 1:] != -100`인 토큰 수를 확인하세요.

        원본 `generate_labels`는 응답의 종료 패턴을 못 찾으면 배열 끝까지 정답으로 남깁니다.
        **완성되지 않은 assistant 발화 뒤에 PAD가 붙은 경우 PAD도 정답으로 포함될 수 있습니다.**
        정상 template의 완성 대화와 중간에서 절단된 대화, 원래부터 잘못된 대화를 구분해야 합니다.
        아래는 원본 동작을 관찰하는 예제로, 학습에는 사용하지 않습니다.
        """),
        mo.ui.table([
            {"입력": "assistant 헤더에서 절단", "정답 수": sum(y != -100 for y in _short_labels)},
            {"입력": "종료 없는 응답 + PAD 3개", "정답 수": sum(y != -100 for y in _broken_labels)},
        ], selection=None),
    ])
    return


@app.cell
def _(DataLoader, MiniMindForCausalLM, PretrainDataset, animals, config, json, mo, output_dir, tokenizer, torch):
    # 사전 학습을 모사하는 기준 모델: 일반 문장에 모든 비패딩 위치의 정답을 둡니다.
    _path = output_dir / "pretrain.jsonl"
    _path.write_text("".join(json.dumps({"text": f"{a} 소리 {s}"}, ensure_ascii=False) + "\n" for a, s in animals), encoding="utf-8")
    _dataset = PretrainDataset(str(_path), tokenizer, max_length=8)
    _x, _y = next(iter(DataLoader(_dataset, batch_size=4)))
    torch.manual_seed(42)
    baseline_model = MiniMindForCausalLM(config)
    _optimizer = torch.optim.AdamW(baseline_model.parameters(), lr=0.008)
    preparation_losses = []
    for _step in range(100):
        baseline_model.train()
        _optimizer.zero_grad(set_to_none=True)
        _loss = baseline_model(_x, labels=_y).loss
        _loss.backward()
        torch.nn.utils.clip_grad_norm_(baseline_model.parameters(), 1.0)
        _optimizer.step()
        preparation_losses.append(_loss.item())
    baseline_model.eval()
    torch.save(baseline_model.state_dict(), output_dir / "pretrain_32.pth")
    mo.md(f"""
    ## 6. SFT를 시작할 기준 모델 준비

    원본 trainer의 기본값 `--from_weight pretrain`은 사전 학습된 가중치를 읽습니다.
    여기서는 동물 소리 일반 문장 4개로 작은 모델을 100 step 준비하고 저장합니다.
    준비 손실: `{preparation_losses[0]:.4f} → {preparation_losses[-1]:.4f}`.
    이 모델을 복사한 뒤 **새 optimizer**로 대화 SFT를 시작합니다.
    사전 학습에서 이어받는 것은 가중치이며, optimizer 상태까지 이어가는 학습 재개와는 구분합니다.

    실제 규모의 사전 학습을 재현한 것은 아닙니다. 동일 동물·소리 대응을 준비 단계에서도 봅니다.
    검증에서 측정할 것은 이미 아는 단어의 **새 질문 조합에 대한 응답**입니다.
    """)
    return baseline_model, preparation_losses


@app.cell
def _(DataLoader, mo, tokenizer, torch):
    @torch.inference_mode()
    def evaluate(model, dataset):
        model.eval()
        total_loss, tokens = 0.0, 0
        for ids, targets in DataLoader(dataset, batch_size=4, shuffle=False):
            count = int((targets[:, 1:] != -100).sum())
            if count == 0:
                raise ValueError("유효 응답 정답이 없습니다. max_length와 template을 확인하세요.")
            total_loss += model(ids, labels=targets).loss.item() * count
            tokens += count
        return total_loss / tokens

    @torch.inference_mode()
    def respond(model, messages):
        model.eval()
        # 호출자는 정답 assistant 발화를 제거한 messages를 전달합니다.
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        ids = torch.tensor([tokenizer(prompt, add_special_tokens=False).input_ids])
        generated = []
        stopped = False
        for _ in range(5):
            next_id = int(model(ids).logits[0, -1].argmax())
            if next_id == tokenizer.eos_token_id:
                stopped = True
                break
            generated.append(next_id)
            ids = torch.cat([ids, torch.tensor([[next_id]])], dim=1)
        return tokenizer.decode(generated, skip_special_tokens=False), stopped

    mo.md("""
    **평가 기준:** teacher forcing 응답 토큰 평균 CE와 실제 greedy 생성의 응답 일치율을 따로 봅니다.
    CE가 낮아져도 생성에서 앞선 오류가 누적될 수 있습니다. 여기서는 EOS 종료 여부도 함께 기록합니다.
    검증 손실은 유효 토큰 수로 가중 평균하고, optimizer 갱신이나 설정 선택에 사용하지 않습니다.
    """)
    return evaluate, respond


@app.cell
def _(mo):
    sft_epochs = mo.ui.slider(20, 100, step=20, value=60, label="SFT epoch")
    sft_accumulation = mo.ui.dropdown([1, 2, 3], value=2, label="기울기 누적 수")
    mo.vstack([
        mo.md("""
        ## 7. 원본 trainer의 핵심 학습 루프

        **예측:** microbatch 4개에서 누적 수가 2라면 optimizer는 몇 번 갱신할까요?
        `forward → (주 손실 + aux)/누적 수 → backward → clipping → step → zero_grad`입니다.
        Full SFT에서는 모든 매개변수에 `requires_grad=True`를 유지합니다.
        원본은 `get_lr`의 cosine 감소를 **microstep마다** 적용합니다.
        Dense의 aux는 0이며, MoE를 사용하면 원본 모델이 계수를 반영한 aux를 반환하므로 한 번 더합니다.

        이 실습은 CPU float32·단일 프로세스입니다. AMP/GradScaler·DDP·wandb·중간 재개는 생략합니다.
        작은 모델에 맞춰 초기 LR=0.003을 쓰며 원본 기본 LR=1e-5와 다릅니다.
        UI를 바꾸면 항상 같은 기준 모델의 새 복사본에서 학습을 시작합니다.
        """),
        mo.hstack([sft_epochs, sft_accumulation]),
    ])
    return sft_accumulation, sft_epochs


@app.cell
def _(DataLoader, baseline_model, copy, evaluate, get_lr, json, mo, output_dir, random, sft_accumulation, sft_epochs, torch, train_dataset, uuid4, validation_dataset):
    random.seed(42)
    torch.manual_seed(42)
    sft_model = copy.deepcopy(baseline_model)
    optimizer = torch.optim.AdamW(sft_model.parameters(), lr=0.003)
    _loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=0,
                         generator=torch.Generator().manual_seed(42))
    _epochs, _accum = sft_epochs.value, sft_accumulation.value
    before_metrics = {"train_ce": evaluate(baseline_model, train_dataset), "validation_ce": evaluate(baseline_model, validation_dataset)}
    history, step_history = [], []
    optimizer_updates = 0
    optimizer.zero_grad(set_to_none=True)
    for _epoch in range(_epochs):
        sft_model.train()
        for _step, (_ids, _labels) in enumerate(_loader, start=1):
            assert (_labels[:, 1:] != -100).any(dim=1).all(), "응답이 절단되었습니다"
            _lr = get_lr(_epoch * len(_loader) + _step, _epochs * len(_loader), 0.003)
            for _group in optimizer.param_groups:
                _group["lr"] = _lr
            _res = sft_model(_ids, labels=_labels)
            _loss = (_res.loss + _res.aux_loss) / _accum
            _loss.backward()
            _update = _step % _accum == 0 or _step == len(_loader)
            _grad_norm = None
            if _update:
                _grad_norm = float(torch.nn.utils.clip_grad_norm_(sft_model.parameters(), 1.0))
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_updates += 1
            step_history.append({"epoch": _epoch + 1, "microstep": _step, "lr": _lr,
                                 "ce": _res.loss.item(), "aux": float(_res.aux_loss),
                                 "optimizer_update": _update, "grad_norm_before_clip": _grad_norm})
        if (_epoch + 1) % 10 == 0 or _epoch == 0:
            history.append({"epoch": _epoch + 1, "train_ce": evaluate(sft_model, train_dataset),
                            "validation_ce": evaluate(sft_model, validation_dataset)})
    after_metrics = {"train_ce": evaluate(sft_model, train_dataset), "validation_ce": evaluate(sft_model, validation_dataset)}
    assert all(p.requires_grad for p in sft_model.parameters())
    _changed = sum(not torch.equal(p, dict(baseline_model.named_parameters())[n]) for n, p in sft_model.named_parameters())
    training_dir = output_dir / f"training-{uuid4().hex[:6]}"
    training_dir.mkdir()
    training_settings = {"epochs": _epochs, "accumulation_steps": _accum, "batch_size": 2,
                         "learning_rate": 0.003, "seed": 42, "pretrain_steps": 100}
    (training_dir / "training.json").write_text(json.dumps({"settings": training_settings, "before": before_metrics,
        "after": after_metrics, "history": history, "microsteps": step_history}, indent=2), encoding="utf-8")
    mo.vstack([
        mo.md(f"""
        1 epoch에 microbatch `{len(_loader)}`개, 총 optimizer 갱신 `{optimizer_updates}`회입니다.
        학습 대상 `{sum(p.numel() for p in sft_model.parameters()):,}`개,
        값이 바뀐 매개변수 텐서는 `{_changed}`개입니다.

        **원본과 같은 평균 방식:** 각 microbatch 평균 손실을 누적 수로 나눕니다.
        누적 수 3이면 마지막 1개 그룹도 3으로 나누므로 완전한 그룹보다 gradient가 작습니다.
        응답 길이나 microbatch 크기가 다르면 전체 유효 토큰 평균과도 다릅니다.
        현재 예제는 응답 길이와 microbatch 크기가 같아 완전한 누적 그룹에서 두 평균이 같습니다.
        """),
        mo.ui.table(step_history[:len(_loader)], selection=None),
        mo.ui.table([{"시점": k, **v} for k, v in [("SFT 전", before_metrics), ("SFT 후", after_metrics)]], selection=None),
    ])
    return after_metrics, before_metrics, history, sft_model, training_dir, training_settings


@app.cell
def _(before_metrics, history, mo, plt, training_dir):
    _fig, _ax = plt.subplots(figsize=(7, 3))
    _ax.plot([0] + [r["epoch"] for r in history], [before_metrics["train_ce"]] + [r["train_ce"] for r in history], label="Train")
    _ax.plot([0] + [r["epoch"] for r in history], [before_metrics["validation_ce"]] + [r["validation_ce"] for r in history], label="Validation")
    _ax.set(xlabel="SFT epoch", ylabel="Assistant token CE")
    _ax.legend()
    _fig.tight_layout()
    _fig.savefig(training_dir / "loss.png", dpi=140)
    mo.vstack([mo.md("## 8. 손실과 실제 생성은 함께 확인하기\n\n**관찰:** 학습·검증 손실은 같은 방향으로 움직이나요? 다음 표에서도 같은 경향인가요?"), _fig])
    return


@app.cell
def _(baseline_model, json, mo, respond, sft_model, train_records, training_dir, validation_records):
    generation_rows = []
    for _split, _records in [("학습", train_records), ("검증", validation_records)]:
        for _record in _records:
            _messages = _record["conversations"]
            for _stage, _model in [("SFT 전", baseline_model), ("SFT 후", sft_model)]:
                _answer, _stopped = respond(_model, _messages[:-1])
                generation_rows.append({"분할": _split, "시점": _stage, "질문": _messages[1]["content"],
                    "정답": _messages[-1]["content"], "생성": _answer, "EOS 종료": _stopped,
                    "응답+종료 일치": _answer == _messages[-1]["content"] and _stopped})
    generation_summary = [
        {"분할": split, "시점": stage,
         "정확한 응답+종료": sum(r["응답+종료 일치"] for r in generation_rows if r["분할"] == split and r["시점"] == stage),
         "질문 수": sum(r["분할"] == split and r["시점"] == stage for r in generation_rows)}
        for split in ["학습", "검증"] for stage in ["SFT 전", "SFT 후"]
    ]
    (training_dir / "generation.json").write_text(json.dumps({"summary": generation_summary, "rows": generation_rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    mo.vstack([
        mo.md("""
        생성 함수에는 `conversations[:-1]`만 넘겼습니다. **정답 응답이 입력에 섞이지 않았는지** 확인하세요.
        greedy로 최대 5토큰을 생성하고 EOS에서 멈춥니다. 표에는 special token도 보존합니다.
        검증 4개는 같은 동물·소리의 새 표현 조합입니다. 이 결과로 미지의 동물이나 자유 대화 능력을 판단하지 않습니다.
        """),
        mo.ui.table(generation_summary, selection=None),
        mo.ui.table(generation_rows, selection=None),
    ])
    return generation_summary, generation_rows


@app.cell
def _(MiniMindForCausalLM, config, input_ids, mo, sft_model, torch, training_dir, training_settings):
    checkpoint_path = training_dir / "full_sft_32.pth"
    # 원본 배포 가중치는 half 저장. 여기서는 정확한 복원 관찰을 위해 float32를 보존합니다.
    torch.save(sft_model.state_dict(), checkpoint_path)
    config.save_pretrained(training_dir / "config")
    restored_model = MiniMindForCausalLM(config)
    restored_model.load_state_dict(torch.load(checkpoint_path, map_location="cpu", weights_only=True), strict=True)
    restored_model.eval()
    sft_model.eval()
    with torch.no_grad():
        _original = sft_model(input_ids).logits
        _restored = restored_model(input_ids).logits
    _difference = (_original - _restored).abs().max().item()
    assert torch.equal(_original, _restored)
    mo.md(f"""
    ## 9. 저장·복원과 원본 trainer로 연결

    저장 파일: `{checkpoint_path}`. 같은 설정으로 복원한 logits의 최대 차이: `{_difference}`.
    설정·학습 기록·생성 결과·tokenizer도 이번 실행 폴더에 보관했습니다.
    설정: `{training_settings}`.
    여기서 저장한 것은 추론용 state_dict입니다. 학습 중간 재개에는 optimizer·scaler·epoch·step 등도 필요합니다.
    원본 trainer는 half 배포 가중치와 `lm_checkpoint` 재개 파일을 따로 저장합니다.

    | 원본 파일·코드 | 이 실습에서 확인한 의미 |
    | --- | --- |
    | `SFTDataset.create_chat_prompt` | 대화를 template에 맞추어 직렬화 |
    | `SFTDataset.generate_labels` | assistant 본문과 종료 패턴에만 정답 부여 |
    | `init_model(..., from_weight='pretrain')` | 사전 학습 가중치에서 시작 |
    | `model(input_ids, labels=labels)` | 내부 shift와 마스킹된 CE 계산 |
    | `loss + aux_loss`, backward, clip, step | 전체 매개변수의 gradient와 갱신 |
    | `--from_resume 1` | 모델뿐 아니라 학습 상태까지 복원 |

    실제 trainer는 `minimind/trainer/`에서 실행하는 상대 경로를 사용합니다.
    기본 설정은 768차원·8층·max_seq_len 768·batch 16·LR 1e-5·2 epoch입니다.
    실행 전 사전 학습 가중치의 차원·층 수·토크나이저와 JSONL 데이터 경로를 확인해야 합니다.
    이 노트북의 32차원 가중치는 기본 768차원 모델에 불러올 수 없습니다.
    """)
    return (checkpoint_path,)


@app.cell
def _(mo):
    mo.md("""
    ## 10. 직접 설명하고 바꿔 보기

    1. **마스킹:** user가 입력에 있는데 왜 그 위치의 label은 -100일까요?
       여러 턴 표에서 두 응답과 두 EOS가 모두 정답인지 확인하고 위치를 적으세요.
    2. **shift:** 첫 응답을 예측하는 입력 위치를 찾으세요. Dataset에서 정답을 한 칸 이동하면
       어떤 잘못된 쌍이 생길지 4절 표에 표시하세요.
    3. **길이:** 학습용 max_length를 32에서 8로 줄이기 전에 유효 응답 정답 수를 예상하세요.
       마스킹 관찰 셀에서 먼저 확인하고, 유효 정답이 없는 배치를 학습시키지 마세요.
    4. **학습 비교:** epoch 20/60/100에서 학습·검증 CE와 정확한 응답 수, EOS 종료 수를 기록하세요.
       더 오래 학습한 결과가 새 질문의 응답도 개선하는지 확인하세요.
    5. **누적:** 누적 수 2/3에서 첫 epoch의 optimizer 갱신 횟수와 마지막 그룹의 gradient 크기를 비교하세요.
       microstep 수가 같아도 optimizer 갱신 횟수는 다를 수 있습니다.
    6. **데이터 품질:** 학습 데이터에서 고양이 응답 한 개만 `멍멍`으로 바꿔 재학습하세요.
       깨끗한 검증 정답은 유지하고, 손실과 실제 답변이 어떻게 달라지는지 기록하세요.

    **힌트·해설:** user/system은 예측 조건이고 assistant 응답이 지도 신호입니다.
    EOS도 포함해야 멈추는 행동을 학습합니다. SFT는 모범 응답의 확률을 높이므로 잘못된 정답도 배울 수 있습니다.
    낮은 학습 손실은 좋은 데이터나 일반화 성능의 보장이 아닙니다.
    LoRA로 넘어가면 같은 SFT 목표에서 **어떤 매개변수를 갱신하는지** 비교해 보세요.

    출처: `minimind/trainer/train_full_sft.py`, `minimind/dataset/lm_dataset.py`,
    `minimind/model/model_minimind.py`, `minimind/trainer/trainer_utils.py`.
    원본 클래스·함수를 직접 가져오며 원본 파일은 수정하지 않습니다.
    저작권·라이선스는 `minimind/LICENSE`를 확인하세요.
    """)
    return


if __name__ == "__main__":
    app.run()
