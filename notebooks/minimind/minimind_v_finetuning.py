import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import io
    import json
    import sys
    from datetime import datetime
    from pathlib import Path
    from uuid import uuid4
    from zoneinfo import ZoneInfo

    import datasets
    import marimo as mo
    import matplotlib.pyplot as plt
    import pyarrow as pa
    import pyarrow.parquet as pq
    import torch
    from PIL import Image, ImageDraw
    from tokenizers import Tokenizer, models, pre_tokenizers
    from torch.utils.data import DataLoader
    from transformers import (
        PreTrainedTokenizerFast, SiglipImageProcessor,
        SiglipVisionConfig, SiglipVisionModel,
    )

    repo_root = Path(__file__).resolve().parents[2]
    project_dir = repo_root / "projects/minimind-v"
    # 동명 model/dataset/trainer 모듈 충돌을 피하려면 별도 프로세스로 실행합니다.
    for _name in ("model", "dataset", "trainer"):
        _loaded = sys.modules.get(_name)
        if _loaded is not None and getattr(_loaded, "__file__", None):
            if not Path(_loaded.__file__).resolve().is_relative_to(project_dir):
                raise RuntimeError("다른 프로젝트 모듈이 로드되었습니다. 새 프로세스로 실행하세요.")
    sys.path.insert(0, str(project_dir))
    from dataset.lm_dataset import VLMDataset
    from model.model_vlm import VLMConfig
    from trainer.trainer_utils import get_lr, init_vlm_model, setup_seed, vlm_collate_fn

    run_id = datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y%m%d-%H%M%S") + f"-{uuid4().hex[:6]}-vlm-finetuning"
    output_dir = repo_root / "outputs" / "notebooks" / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    datasets.config.HF_DATASETS_CACHE = output_dir / "dataset-cache"
    datasets.disable_progress_bars()
    torch.set_num_threads(2)
    setup_seed(42)
    return (
        DataLoader, Image, ImageDraw, PreTrainedTokenizerFast,
        SiglipImageProcessor, SiglipVisionConfig, SiglipVisionModel, Tokenizer,
        VLMConfig, VLMDataset, get_lr, init_vlm_model, io, json, mo, models,
        output_dir, pa, plt, pq, pre_tokenizers, setup_seed, torch, uuid4, vlm_collate_fn,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # Vision LLM 미세 조정: 이미지가 답변의 근거가 되기까지

    VLM은 **이미 언어를 아는 LLM**에 눈을 붙이는 작업입니다. 원본 MiniMind-V의 학습 순서는 다음과 같습니다.

    ```text
    ① 텍스트 LLM (MiniMind의 llm_768.pth, 이미지를 본 적 없음)
         ↓ from_weight=llm, freeze_llm=2 (Projector만 학습)
    ② Projector 정렬  train_pretrain_vlm.py   pretrain_i2t.parquet  (README상 선택 단계)
         ↓ from_weight=pretrain_vlm, freeze_llm=1 (Projector + LLM 첫·마지막 층)
    ③ 시각 지시 SFT    train_sft_vlm.py        sft_i2t.parquet
    ```

    이 노트북은 세 단계를 CPU에서 작게 재현합니다. 각 단계가 무엇을 바꾸는지가 핵심입니다.

    | 단계 | 이 실습에서 배우는 것 | 확인 방법 |
    | --- | --- | --- |
    | ① 텍스트 LLM | 문장 속 색 단어를 따라 말하기 (`빨강 색 ?` → `빨강`) | 이미지를 넣으면 색을 구분하지 못함 |
    | ② 정렬 | Projector가 이미지 벡터를 **LLM이 아는 색 단어처럼** 만들기 | LLM은 그대로인데 이미지로 색을 맞힘 |
    | ③ SFT | 새 지시 형식 `무슨 색 ?` → `색 은 빨강` | 형식까지 맞는 greedy 응답 |

    | 원본 | 여기서 확인할 내용 |
    | --- | --- |
    | `dataset/lm_dataset.py` | 실제 `VLMDataset`: Parquet → 이미지·대화 → 정답 마스킹 |
    | `trainer/trainer_utils.py` | `init_vlm_model`의 가중치 로딩과 동결 정책 |
    | `trainer/train_pretrain_vlm.py`, `train_sft_vlm.py` | 누적·clipping·학습률·업데이트 루프 |

    `minimind_sft.py`(assistant 마스크)와 `minimind_vlm.py`(이미지 자리 교체·gradient 경로)를 먼저 읽으세요.
    다운로드·GPU 없이 합성 이미지, 작은 단어 사전, 무작위 SigLIP으로 실제 학습합니다.
    실제 사전 학습 VLM의 성능 재현은 아닙니다. 산출물: `outputs/notebooks/<실행 ID>/`.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 1. 데이터: 텍스트 대화 + 이미지 대화

    `image_bytes`는 PNG 파일의 원시 bytes, `conversations`는 `system/user/assistant` 대화(JSON 문자열)입니다.

    - **텍스트 LLM용:** 이미지 없음. 질문 안에 색 단어가 있습니다. 원본에서는 MiniMind 텍스트 학습에 해당합니다.
    - **정렬용:** 질문에서 색 단어 자리에 `<image>`가 들어갑니다. 원본 `pretrain_i2t`의 짧은 이미지 설명에 해당합니다.
    - **SFT용:** 처음 보는 지시 형식입니다. 원본 `sft_i2t`의 지시 대화에 해당합니다.

    학습과 검증은 사각형의 **위치만** 다릅니다. 색·모양·질문이 같은 작은 분포라서
    새 사물·질문에 대한 일반화는 측정하지 않습니다. 명시적 system으로 원본의 20% 무작위 system 추가를 피합니다.
    """)
    return


@app.cell
def _(Image, ImageDraw, io, json, output_dir, pa, plt, pq):
    COLORS = [("red", "빨강"), ("blue", "파랑")]

    def conversation(user, answer):
        return [
            {"role": "system", "content": "그림 도우미"},
            {"role": "user", "content": user},
            {"role": "assistant", "content": answer},
        ]

    def make_record(color, question, answer, offset):
        image = Image.new("RGB", (32, 32), "white")
        ImageDraw.Draw(image).rectangle((offset, 6, offset + 18, 25), fill=color)
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return {"image_bytes": buffer.getvalue(),
                "conversations": json.dumps(conversation("<image> " + question, answer), ensure_ascii=False)}

    # 텍스트 LLM은 문장 속 색 단어를 그대로 답하는 법만 압니다.
    text_conversations = [conversation(f"{a} 색 ?", a) for _, a in COLORS]
    # 같은 위치의 빨강/파랑이 짝수/홀수 인덱스로 이어집니다(이미지 교환 검사에 사용).
    records = {
        "alignment": [make_record(c, "색 ?", a, x) for x in (2, 4, 7, 10) for c, a in COLORS],
        "sft": [make_record(c, "무슨 색 ?", f"색 은 {a}", x) for x in (2, 4, 7, 10) for c, a in COLORS],
        "val_alignment": [make_record(c, "색 ?", a, x) for x in (3, 8) for c, a in COLORS],
        "val_sft": [make_record(c, "무슨 색 ?", f"색 은 {a}", x) for x in (3, 8) for c, a in COLORS],
    }
    parquet_paths = {}
    for _split, _rows in records.items():
        _path = output_dir / f"{_split}.parquet"
        pq.write_table(pa.Table.from_pylist(_rows), _path)
        parquet_paths[_split] = _path
    sample_fig, _axes = plt.subplots(1, 2, figsize=(5, 2))
    for _axis, _row, _title in zip(_axes, records["sft"][:2], ["red", "blue"]):
        _axis.imshow(Image.open(io.BytesIO(_row["image_bytes"])))
        _axis.set_title(_title)
        _axis.axis("off")
    return (COLORS, parquet_paths, records, sample_fig, text_conversations,)


@app.cell(hide_code=True)
def _(json, mo, records, sample_fig, text_conversations):
    # ▸ 결과 표시 — 실행하면 표·그림이 나타납니다 (marimo 표시 코드)
    def _example(conv):
        return f"`{conv[1]['content']}` → `{conv[2]['content']}`"
    mo.vstack([
        sample_fig,
        mo.ui.table(
            [{"split": "text_llm", "행 수": len(text_conversations), "이미지": "없음", "예": _example(text_conversations[0])}]
            + [{"split": k, "행 수": len(v), "이미지": "있음", "예": _example(json.loads(v[0]["conversations"]))}
               for k, v in records.items()],
            selection=None,
        ),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 2. 작은 로컬 tokenizer와 비전 인코더

    원본은 MiniMind tokenizer와 SigLIP 가중치가 필요합니다. 여기서는 단어 단위 tokenizer와
    작은 SigLIP을 저장해 **원본과 같은 로딩 함수**로 읽습니다. 단어 사전에서는 줄바꿈이 사라져
    assistant 시작 패턴은 `[BOS], assistant`, 종료 패턴은 `[EOS]`입니다.
    """)
    return


@app.cell
def _(PreTrainedTokenizerFast, SiglipImageProcessor, SiglipVisionConfig, SiglipVisionModel, Tokenizer, VLMConfig, json, models, output_dir, pre_tokenizers, records, setup_seed, text_conversations):
    _all_conversations = text_conversations + [json.loads(r["conversations"]) for rows in records.values() for r in rows]
    vocabulary = ["[PAD]", "[BOS]", "[EOS]", "[UNK]", "<|image_pad|>", "system", "user", "assistant"] + sorted({
        word for conv in _all_conversations for turn in conv for word in turn["content"].split() if word != "<image>"
    })
    # marker는 AddedToken이므로 공백 없이 4개가 붙어도 각각 인식합니다.
    _backend = Tokenizer(models.WordLevel({w: i for i, w in enumerate(vocabulary)}, unk_token="[UNK]"))
    _backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=_backend, bos_token="[BOS]", eos_token="[EOS]",
        pad_token="[PAD]", unk_token="[UNK]", additional_special_tokens=["<|image_pad|>"],
        clean_up_tokenization_spaces=False,
    )
    tokenizer.chat_template = (
        "{% for message in messages %}"
        "{{ bos_token + message['role'] + '\n' + message['content'] + eos_token + '\n' }}"
        "{% endfor %}"
        "{% if add_generation_prompt %}{{ bos_token + 'assistant\n' }}{% endif %}"
    )
    tokenizer_dir = output_dir / "tokenizer"
    tokenizer.save_pretrained(tokenizer_dir)
    setup_seed(42)
    vision_dir = output_dir / "tiny-siglip"
    SiglipVisionModel(SiglipVisionConfig(
        hidden_size=16, intermediate_size=32, num_hidden_layers=1,
        num_attention_heads=2, image_size=32, patch_size=16,
    )).save_pretrained(vision_dir)
    SiglipImageProcessor(size={"height": 32, "width": 32}).save_pretrained(vision_dir)
    processor = SiglipImageProcessor.from_pretrained(vision_dir)
    config = VLMConfig(
        hidden_size=32, intermediate_size=64, num_hidden_layers=3,
        num_attention_heads=4, num_key_value_heads=2, vocab_size=len(vocabulary),
        image_hidden_size=16, image_token_len=4,
        image_ids=[tokenizer.convert_tokens_to_ids("<|image_pad|>")],
        max_position_embeddings=96, max_seq_len=40, dropout=0.0, flash_attn=False,
        use_moe=False, bos_token_id=tokenizer.bos_token_id, eos_token_id=tokenizer.eos_token_id,
    )
    assert tokenizer(config.image_special_token * 4, add_special_tokens=False).input_ids == config.image_ids * 4
    config.save_pretrained(output_dir / "config")
    return (config, processor, tokenizer, tokenizer_dir, vision_dir, vocabulary,)


@app.cell(hide_code=True)
def _(config, mo, vocabulary):
    # ▸ 결과 설명 — 실행하면 계산 값이 들어간 설명이 나타납니다 (marimo 표시 코드)
    mo.md(f"""
    `<image>` 한 개를 `<|image_pad|>` **4개**로 바꿉니다(32×32 / 패치 16 = 4패치, 원본은 64).
    marker ID는 원본 기본값 `12`를 복사하지 않고 이 tokenizer의 ID `{config.image_ids[0]}`로 맞춥니다.
    LLM은 3층으로 만들어 `freeze_llm=1`에서 첫·마지막 층과 고정된 가운데 층을 구분합니다.
    사전 크기 {len(vocabulary)}, 언어 차원 32, 시각 차원 16을 사용합니다.
    """)
    return


@app.cell
def _(config, init_vlm_model, output_dir, tokenizer_dir, vision_dir):
    def load_vlm(prefix="none", freeze=2, weight_dir=None):
        # 원본 초기화 함수: 가중치 로딩(strict=False) → 동결 정책 적용
        return init_vlm_model(
            config, from_weight=prefix, tokenizer_path=str(tokenizer_dir),
            vision_model_path=str(vision_dir), save_dir=str(weight_dir or output_dir),
            device="cpu", freeze_llm=freeze,
        )[0]
    return (load_vlm,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 3. 실제 VLMDataset: assistant 응답만 정답

    **예측:** 이미지 marker, user 질문, assistant 헤더, EOS 중 어느 것이 정답일까요?
    원본은 `create_chat_prompt → tokenize → 절단 → PAD → generate_labels`를 거칩니다.
    **해설:** assistant 본문과 종료 패턴만 labels에 남고 나머지는 `-100`입니다.
    질문과 이미지는 입력에 남아 응답 예측의 문맥이 됩니다.
    """)
    return


@app.cell
def _(DataLoader, VLMDataset, config, parquet_paths, processor, tokenizer, vlm_collate_fn):
    def read_dataset(path, max_length=40):
        return VLMDataset(
            str(path), tokenizer, preprocess=processor, max_length=max_length,
            image_special_token=config.image_special_token, image_token_len=config.image_token_len,
        )

    vlm_datasets = {k: read_dataset(p) for k, p in parquet_paths.items()}
    ids, labels, pixels = next(iter(DataLoader(vlm_datasets["sft"], batch_size=2, collate_fn=vlm_collate_fn)))
    assert ids.shape == labels.shape == (2, 40)
    assert pixels["pixel_values"].shape == (2, 1, 3, 32, 32)
    assert (ids == config.image_ids[0]).sum(1).tolist() == [4, 4]
    assert (labels[ids == config.image_ids[0]] == -100).all()
    assert (labels[:, 1:] != -100).sum(1).tolist() == [4, 4]
    assert not (ids == tokenizer.unk_token_id).any()
    return (ids, labels, pixels, read_dataset, vlm_datasets,)


@app.cell(hide_code=True)
def _(ids, labels, mo, pixels, tokenizer):
    # ▸ 결과 표시 — 실행하면 표·그림이 나타납니다 (marimo 표시 코드)
    mo.vstack([
        mo.md(f"""
        `input_ids`와 `labels`: `{tuple(ids.shape)}` = [B,T].
        `vlm_collate_fn`이 샘플별 [N,C,H,W]를 쌓으면 `pixel_values`: `{tuple(pixels['pixel_values'].shape)}` = [B,N,C,H,W].
        아래는 첫 샘플이며 뒤쪽 PAD는 생략했습니다.
        """),
        mo.ui.table([
            {"위치": t, "입력": tokenizer.convert_ids_to_tokens(int(x)), "label": int(y), "정답": int(y) != -100}
            for t, (x, y) in enumerate(zip(ids[0], labels[0])) if int(x) != tokenizer.pad_token_id
        ], selection=None),
        # 원본 다이어그램: public/diagrams/v_finetuning.mmd
        mo.image(mo.notebook_dir() / "public/diagrams/v_finetuning.png", width=400,
                 alt="이미지 bytes부터 assistant 다음 토큰 CE까지의 shape 흐름"),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 4. ① 텍스트 LLM 준비: 언어는 알지만 이미지는 모른다

    원본 정렬은 학습된 `llm_768.pth`에서 시작합니다. 여기서는 작은 LLM을 **이미지 없이** 학습해 그 역할을 대신합니다.
    이 LLM은 `빨강 색 ?`처럼 **문장 안에 있는 색 단어를 찾아 답하는 능력**을 갖습니다.

    저장할 때 `vision_proj.*`는 제외합니다. 원본 `llm_768.pth`에도 Projector가 없으며,
    `init_vlm_model`은 `strict=False`로 읽으므로 Projector는 무작위 초기값으로 남습니다.

    **예측:** 이 LLM에 `<image> 색 ?`를 주면 이미지 색을 맞힐까요?
    marker 자리에는 무작위 Projector의 출력이 들어갑니다. LLM에게 그 벡터는 아무 단어와도 닮지 않았습니다.
    """)
    return


@app.cell
def _(COLORS, config, load_vlm, output_dir, setup_seed, text_conversations, tokenizer, torch, uuid4, vlm_datasets):
    training_dir = output_dir / f"training-{uuid4().hex[:6]}"
    training_dir.mkdir()
    color_ids = tokenizer.convert_tokens_to_ids([a for _, a in COLORS])

    def encode_text(conv, max_length=40):
        # 이미지 없는 대화도 원본 VLMDataset의 프롬프트·label 규칙을 그대로 사용합니다.
        _builder = vlm_datasets["sft"]
        x = tokenizer(_builder.create_chat_prompt(conv)).input_ids[:max_length]
        x += [tokenizer.pad_token_id] * (max_length - len(x))
        return torch.tensor(x), torch.tensor(_builder.generate_labels(x))

    _encoded = [encode_text(c) for c in text_conversations]
    text_x = torch.stack([x for x, _ in _encoded])
    text_y = torch.stack([y for _, y in _encoded])

    setup_seed(42)
    _llm = load_vlm(freeze=0)
    _optimizer = torch.optim.AdamW([p for p in _llm.parameters() if p.requires_grad], lr=3e-3)
    _llm.train()
    for _ in range(200):
        _optimizer.zero_grad(set_to_none=True)
        text_llm_loss = _llm(text_x, labels=text_y).loss  # pixel_values 없음
        text_llm_loss.backward()
        _optimizer.step()
    _llm.eval()
    with torch.no_grad():
        _logits = _llm(text_x).logits
    # 각 텍스트 샘플에서 색 정답 위치의 확률
    text_color_probs = []
    for _row in range(len(text_x)):
        _t = next(t for t in range(1, text_y.shape[1]) if int(text_y[_row, t]) in color_ids)
        text_color_probs.append(_logits[_row, _t - 1].softmax(-1)[text_y[_row, _t]].item())
    assert min(text_color_probs) > 0.9

    llm_weight_path = training_dir / f"llm_{config.hidden_size}.pth"
    torch.save({k: v.detach().clone() for k, v in _llm.state_dict().items()
                if not k.startswith(("vision_encoder.", "vision_proj."))}, llm_weight_path)
    return (color_ids, llm_weight_path, text_color_probs, text_llm_loss, training_dir,)


@app.cell(hide_code=True)
def _(llm_weight_path, mo, text_color_probs, text_llm_loss):
    # ▸ 결과 설명 — 실행하면 계산 값이 들어간 설명이 나타납니다 (marimo 표시 코드)
    mo.md(f"""
    텍스트 학습 loss: `{text_llm_loss.item():.4f}`. 문장 속 색 단어를 답할 확률: `{[round(p, 3) for p in text_color_probs]}`.
    저장: `{llm_weight_path.name}` (Projector 제외). 이미지 입력에 대한 결과는 8절 표의 첫 행에서 확인합니다.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 5. 동결 정책과 학습률: 무엇을, 얼마나 바꿀까

    | 원본 기본값 | ② Projector 정렬 | ③ 시각 지시 SFT |
    | --- | --- | --- |
    | 데이터 | pretrain_i2t.parquet | sft_i2t.parquet |
    | 시작 가중치 | llm | pretrain_vlm |
    | freeze_llm | 2 (Projector만) | 1 (Projector + LLM 첫·마지막 층) |
    | learning_rate | 4e-4 | 5e-6 |
    | batch_size / max_seq_len | 16 / 450 | 4 / 768 |
    | epochs | 2 | 2 |

    - **정렬에서 LLM을 동결하는 이유:** LLM이 이미 아는 단어 임베딩 공간을 기준점으로 고정하고,
      새로 만든 Projector가 그 공간에 맞춰 오도록 합니다. LLM까지 움직이면 기준점이 흔들립니다.
    - **학습률이 80배 다른 이유:** 정렬은 무작위 Projector를 크게 움직여야 하고,
      SFT는 이미 학습된 LLM 층을 건드리므로 언어 능력을 잃지 않도록 작게 조정합니다.
    - **`freeze_llm=1`이 첫·마지막 층만 푸는 이유:** 입력 쪽 층은 이미지 벡터를 받아들이는 방식을,
      출력 쪽 층은 새 응답 형식을 적응시킵니다. 가운데 층은 원래 언어 능력을 보존합니다.
      임베딩·최종 norm·가운데 층은 고정되므로 Full SFT가 아닙니다.

    비전 인코더는 모든 정책에서 고정입니다. 아래는 원본 `init_vlm_model`이 설정한 `requires_grad`를 집계한 표입니다.
    동결된 LLM도 gradient는 **통과시킵니다**(VLM 노트북 6절). 고정되는 것은 LLM 가중치뿐입니다.
    """)
    return


@app.cell
def _(load_vlm):
    freeze_rows = []
    for _policy in [0, 1, 2]:
        _trainable = {n: p for n, p in load_vlm(freeze=_policy).named_parameters() if p.requires_grad}
        _names = list(_trainable)
        assert all(not n.startswith("vision_encoder.") for n in _names)
        freeze_rows.append({
            "freeze_llm": _policy,
            "Projector": any(n.startswith("vision_proj.") for n in _names),
            "layers.0 / 1 / 2": " / ".join(str(any(n.startswith(f"model.layers.{i}.") for n in _names)) for i in range(3)),
            "embed·norm": any(n.startswith(("model.embed_tokens.", "model.norm.")) for n in _names),
            "학습 매개변수 수": sum(p.numel() for p in _trainable.values()),
        })
    assert [r["layers.0 / 1 / 2"] for r in freeze_rows] == ["True / True / True", "True / False / True", "False / False / False"]
    return (freeze_rows,)


@app.cell(hide_code=True)
def _(freeze_rows, mo):
    # ▸ 결과 표시 — 실행하면 표·그림이 나타납니다 (marimo 표시 코드)
    mo.ui.table(freeze_rows, selection=None)
    return


@app.cell
def _(read_dataset, parquet_paths):
    # 학습 전 점검: 너무 짧은 max_length는 정답을 모두 잘라냅니다.
    _ids, _labels, _ = read_dataset(parquet_paths["sft"], max_length=8)[0]
    short_valid_targets = int((_labels[1:] != -100).sum())
    assert short_valid_targets == 0
    return (short_valid_targets,)


@app.cell(hide_code=True)
def _(config, mo, short_valid_targets):
    # ▸ 결과 설명 — 실행하면 계산 값이 들어간 설명이 나타납니다 (marimo 표시 코드)
    mo.md(f"""
    **학습 전 점검.** `max_length=8`로 줄이면 유효 정답이 **{short_valid_targets}개**입니다.
    모든 labels가 `-100`이면 평균 CE가 NaN이 됩니다. 길이 예산에는 이미지당 `{config.image_token_len}`자리도 들어가며,
    절단이 marker 중간을 자르면 이미지 벡터 일부가 버려집니다. 절단 뒤 `labels[:,1:]`의 유효 정답 수를 검사하세요.
    또 `vlm_collate_fn`은 stack을 사용하므로 **배치 안 샘플들의 이미지 개수가 같아야** 합니다.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    # ▸ 실습 설정 위젯 — 실행하면 슬라이더·선택 상자가 나타납니다 (marimo 표시 코드)
    alignment_steps = mo.ui.slider(20, 160, value=80, step=20, label="② 정렬 optimizer step")
    sft_steps = mo.ui.slider(20, 160, value=80, step=20, label="③ SFT optimizer step")
    mo.vstack([
        mo.md("""
        ## 6. ② 정렬 → ③ SFT 학습 실행

        설정을 바꾸면 4절의 텍스트 LLM부터 다시 학습합니다. 원본 루프를 CPU float32·단일 프로세스로 축소했고,
        실제 모델·데이터셋·초기화·collate·학습률 함수를 사용합니다. 전역 args에 의존하는 원본 `train_epoch` 자체는 호출하지 않습니다.

        한 optimizer step = microbatch 4개 × 누적 2회 = 8샘플입니다.
        `zero_grad → forward → (CE+aux)/누적수 → backward → clip → step`을 관찰하세요.
        작은 무작위 모델이라 학습률은 원본보다 크게 잡았습니다(정렬 2e-2, SFT 3e-3). 두 단계의 비율만 원본과 같은 방향입니다.
        """),
        alignment_steps, sft_steps,
    ])
    return alignment_steps, sft_steps


@app.cell
def _(DataLoader, get_lr, torch, vlm_collate_fn):
    def fit_stage(model, dataset, steps, lr):
        model.train()
        model.vision_encoder.eval()
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
        initial = {n: p.detach().clone() for n, p in model.named_parameters()}
        history = []
        for step in range(steps):
            optimizer.zero_grad(set_to_none=True)
            current_lr = get_lr(step + 1, steps, lr)
            for group in optimizer.param_groups:
                group["lr"] = current_lr
            loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=vlm_collate_fn)
            accumulation = len(loader)  # 8행 / batch=4 → 항상 완전한 2회 누적
            ce_value = 0.0
            for x, y, pv in loader:
                result = model(x, labels=y, pixel_values=pv)
                ((result.loss + result.aux_loss) / accumulation).backward()
                ce_value += result.loss.item() / accumulation
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            history.append({"step": step + 1, "ce": ce_value, "lr": current_lr, "grad_norm_before_clip": float(grad_norm)})
        changes = []
        for name, p in model.named_parameters():
            delta = float((p.detach() - initial[name]).abs().max())
            if not p.requires_grad:
                assert delta == 0, name
            changes.append({"name": name, "trainable": p.requires_grad, "max_abs_delta": delta})
        return history, changes
    return (fit_stage,)


@app.cell
def _(alignment_steps, config, fit_stage, json, llm_weight_path, load_vlm, setup_seed, sft_steps, torch, training_dir, vlm_datasets):
    def save_weights(model, prefix):
        # 원본처럼 encoder를 제외합니다. CPU 수치 비교를 위해 float32를 유지합니다.
        path = training_dir / f"{prefix}_{config.hidden_size}.pth"
        torch.save({k: v.detach().cpu() for k, v in model.state_dict().items() if not k.startswith("vision_encoder.")}, path)
        return path

    setup_seed(42)
    # ② 텍스트 LLM에서 시작, Projector만 학습
    aligned_model = load_vlm("llm", freeze=2, weight_dir=training_dir)
    _llm_weights = torch.load(llm_weight_path, weights_only=True)
    for _name, _value in _llm_weights.items():
        torch.testing.assert_close(aligned_model.state_dict()[_name], _value, rtol=0, atol=0)
    alignment_history, alignment_changes = fit_stage(aligned_model, vlm_datasets["alignment"], alignment_steps.value, 2e-2)
    alignment_weight_path = save_weights(aligned_model, "pretrain_vlm")

    # ③ 정렬 가중치에서 시작, Projector + LLM 첫·마지막 층 학습
    sft_model = load_vlm("pretrain_vlm", freeze=1, weight_dir=training_dir)
    for _name, _value in aligned_model.state_dict().items():
        torch.testing.assert_close(sft_model.state_dict()[_name], _value, rtol=0, atol=0)
    sft_history, sft_changes = fit_stage(sft_model, vlm_datasets["sft"], sft_steps.value, 3e-3)
    sft_weight_path = save_weights(sft_model, "sft_vlm")

    stage_metrics = {
        "seed": 42, "alignment_steps": alignment_steps.value, "sft_steps": sft_steps.value,
        "text_llm_lr": 3e-3, "text_llm_steps": 200, "alignment_lr": 2e-2, "sft_lr": 3e-3,
        "microbatch_size": 4, "accumulation_steps": 2, "grad_clip": 1.0, "dtype": "float32", "device": "cpu",
        "alignment_history": alignment_history, "sft_history": sft_history,
        "alignment_changes": alignment_changes, "sft_changes": sft_changes,
    }
    (training_dir / "metrics.json").write_text(json.dumps(stage_metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    return (aligned_model, alignment_changes, alignment_history, sft_changes, sft_history, sft_model, sft_weight_path,)


@app.cell(hide_code=True)
def _(alignment_changes, alignment_history, mo, plt, sft_changes, sft_history, training_dir):
    # ▸ 결과 표시 — 실행하면 표·그림이 나타납니다 (marimo 표시 코드)
    _fig, _axes = plt.subplots(1, 2, figsize=(10, 3))
    for _axis, _history, _title in zip(_axes, [alignment_history, sft_history], ["② Projector alignment", "③ Visual instruction SFT"]):
        _axis.plot([r["step"] for r in _history], [r["ce"] for r in _history])
        _axis.set(xlabel="Optimizer step", ylabel="Train response CE", title=_title)
        _axis.grid(alpha=0.3)
    _fig.tight_layout()
    _fig.savefig(training_dir / "loss-curves.png", dpi=140)
    _changes = []
    for _stage, _items in [("② 정렬", alignment_changes), ("③ SFT", sft_changes)]:
        for _prefix in ["vision_proj.", "model.embed_tokens.", "model.layers.0.", "model.layers.1.", "model.layers.2.", "model.norm."]:
            _subset = [r for r in _items if r["name"].startswith(_prefix)]
            _changes.append({"단계": _stage, "부분": _prefix, "학습 텐서 수": sum(r["trainable"] for r in _subset),
                             "변경 텐서 수": sum(r["max_abs_delta"] > 0 for r in _subset)})
    mo.vstack([
        mo.md(f"""
        ## 7. 손실과 실제 가중치 변화

        정렬 응답 CE: **{alignment_history[0]['ce']:.3f} → {alignment_history[-1]['ce']:.3f}**,
        SFT 응답 CE: **{sft_history[0]['ce']:.3f} → {sft_history[-1]['ce']:.3f}** (업데이트 직전 microbatch 평균).
        두 단계는 응답 형식이 달라 그래프 높이를 직접 비교하지 않습니다.
        아래 표는 학습 전후 값을 비교한 결과입니다. 고정 텐서는 변화가 0인지 검사했습니다.
        """),
        _fig, mo.ui.table(_changes, selection=None),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 8. 각 단계는 이미지를 사용하는가?

    답변 전체 CE는 이미지와 무관한 토큰(`색`, `은`, EOS)까지 평균하므로 이미지 효과가 희석됩니다.
    여기서는 **색 단어 위치의 정답 확률**만 봅니다. 앞선 답변 토큰은 정답을 넣어 주는 teacher forcing입니다.

    - **정상 이미지:** 그 샘플의 이미지 → 정답 색 확률이 높아야 합니다.
    - **바꾼 이미지:** 같은 위치의 반대 색 이미지, 정답은 그대로 → 이미지를 쓴다면 확률이 **낮아져야** 합니다.
      두 값이 비슷하면 이미지를 무시하고 있다는 뜻입니다(무작위 추측이면 약 0.5).
    - **greedy:** assistant 헤더까지만 넣고 자기 예측을 이어 붙인 결과입니다. 낮은 CE와 다른 지표입니다.

    **예측:** ① 텍스트 LLM, ② 정렬 뒤, ③ SFT 뒤 중 어느 단계부터 바꾼 이미지의 확률이 떨어질까요?
    지시 형식(`색 은 빨강`)의 greedy 정답은 어느 단계부터 나올까요?
    """)
    return


@app.cell
def _(aligned_model, color_ids, json, load_vlm, setup_seed, sft_model, tokenizer, torch, training_dir, vlm_datasets):
    def _batch(pv):
        return {k: v[None] for k, v in pv.items()}

    @torch.no_grad()
    def color_probability(model, dataset, index, image_index):
        x, y, _ = dataset[index]
        t = next(t for t in range(1, len(y)) if int(y[t]) in color_ids)
        logits = model(x[None], pixel_values=_batch(dataset[image_index][2])).logits[0, t - 1]
        return logits.softmax(-1)[y[t]].item()

    @torch.no_grad()
    def greedy_answer(model, dataset, index, max_new_tokens=6):
        x, y, pv = dataset[index]
        prompt = x[:int(torch.where(y != -100)[0][0])][None]  # assistant 헤더까지
        generated = []
        for _ in range(max_new_tokens):
            # 이해를 위해 매번 전체 문맥을 재계산하는 greedy 생성입니다.
            token = int(model(prompt, pixel_values=_batch(pv)).logits[0, -1].argmax())
            generated.append(token)
            prompt = torch.cat([prompt, torch.tensor([[token]])], dim=1)
            if token == tokenizer.eos_token_id:
                break
        return generated == y[y != -100].tolist(), tokenizer.decode(generated, skip_special_tokens=True)

    def _mean(values):
        return round(sum(values) / len(values), 3)

    setup_seed(42)  # 슬라이더 값과 무관하게 같은 무작위 Projector
    _text_llm = load_vlm("llm", freeze=2, weight_dir=training_dir).eval()
    probe_rows = []
    for _stage, _model in [("① 텍스트 LLM", _text_llm), ("② 정렬 뒤", aligned_model), ("③ SFT 뒤", sft_model)]:
        _model.eval()
        _row = {"단계": _stage}
        for _split, _label in [("val_alignment", "짧은 질문"), ("val_sft", "지시 질문")]:
            _ds = vlm_datasets[_split]
            _row[f"{_label} 정상 이미지"] = _mean([color_probability(_model, _ds, i, i) for i in range(len(_ds))])
            _row[f"{_label} 바꾼 이미지"] = _mean([color_probability(_model, _ds, i, i ^ 1) for i in range(len(_ds))])
        _greedy = [greedy_answer(_model, vlm_datasets["val_sft"], i) for i in range(len(vlm_datasets["val_sft"]))]
        _row["지시 질문 greedy 정확 일치"] = f"{sum(ok for ok, _ in _greedy)}/{len(_greedy)}"
        _row["greedy 예 (빨강 이미지)"] = _greedy[0][1]
        probe_rows.append(_row)

    _blind, _aligned, _sft = probe_rows
    # ① 이미지를 구분하지 못함 → ② Projector만으로 이미지 사용 → ③ 지시 형식까지
    assert abs(_blind["짧은 질문 정상 이미지"] - _blind["짧은 질문 바꾼 이미지"]) < 0.1
    assert _aligned["짧은 질문 정상 이미지"] > 0.8 and _aligned["짧은 질문 바꾼 이미지"] < 0.2
    assert _sft["지시 질문 greedy 정확 일치"] == "4/4" and _sft["지시 질문 바꾼 이미지"] < 0.2
    (training_dir / "image-use-probe.json").write_text(json.dumps(probe_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    return (probe_rows,)


@app.cell(hide_code=True)
def _(mo, probe_rows):
    # ▸ 결과 표시 — 실행하면 표·그림이 나타납니다 (marimo 표시 코드)
    mo.vstack([
        mo.ui.table(probe_rows, selection=None),
        mo.md("""
        **읽는 법.**
        ① 텍스트 LLM은 정상/바꾼 이미지의 확률이 거의 같습니다. 이미지 벡터를 읽지 못합니다.
        ② 정렬 뒤에는 **LLM 가중치가 그대로인데도** 정상 이미지는 높고 바꾼 이미지는 낮습니다.
        Projector가 이미지 벡터를 LLM이 이미 아는 색 단어처럼 바꿨기 때문입니다.
        다만 greedy 응답은 텍스트 LLM이 알던 짧은 형식(`빨강`)이라 지시 형식과 일치하지 않습니다.
        ③ SFT 뒤에는 지시 형식 `색 은 빨강`까지 맞추고, 바꾼 이미지에는 여전히 낮은 확률을 줍니다.

        ③에서 **짧은 질문의 확률이 떨어진 것**은 이미지를 잊어서가 아닙니다. 첫 응답 토큰이 색 단어가 아니라
        `색`으로 바뀌어, 짧은 답 형식을 덮어쓴 결과입니다. 작은 데이터로 높은 학습률의 SFT를 하면
        이전 능력이 쉽게 사라진다는 예이며, 원본 SFT 학습률이 매우 작은(5e-6) 이유와 연결됩니다.

        두 색 사각형 과제의 관측값이며 범용 이미지 이해 정확도가 아닙니다.
        """),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 9. 저장과 복원
    """)
    return


@app.cell
def _(ids, json, load_vlm, pixels, sft_model, sft_weight_path, torch, training_dir):
    _restored = load_vlm("sft_vlm", freeze=1, weight_dir=training_dir).eval()
    sft_model.eval()
    with torch.no_grad():
        _before = sft_model(ids, pixel_values=pixels).logits
        _after = _restored(ids, pixel_values=pixels).logits
    torch.testing.assert_close(_before, _after, atol=0, rtol=0)
    _keys = torch.load(sft_weight_path, map_location="cpu", weights_only=True)
    assert not any(k.startswith("vision_encoder.") for k in _keys)
    (training_dir / "restore-check.json").write_text(json.dumps({"logits_exactly_equal": True, "vision_encoder_in_weight": False}), encoding="utf-8")
    return


@app.cell(hide_code=True)
def _(mo, sft_weight_path):
    # ▸ 결과 설명 — 실행하면 계산 값이 들어간 설명이 나타납니다 (marimo 표시 코드)
    mo.md(f"""
    저장 파일: `{sft_weight_path}`.
    원본 initializer로 다시 읽어 같은 배치의 logits가 **정확히 일치**하는지 검사했습니다.
    원본처럼 `vision_encoder.*`를 제외하지만, 원본 trainer는 half로 저장하고 여기서는 float32를 유지합니다.

    추론에는 **동일한 비전 인코더·processor·tokenizer·VLMConfig**도 필요합니다.
    실행 폴더의 `tiny-siglip/`, `tokenizer/`, `config/`에 함께 남깁니다.
    이 파일은 optimizer 상태가 없어 중간 재개용이 아닙니다. 원본 `vlm_checkpoint`는 resume 파일을 별도로 저장합니다.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## 10. 실제 데이터로 확장하기와 과제

    | 이번 실습 | 원본 trainer에서 확인할 설정 |
    | --- | --- |
    | 색 단어 복사 텍스트 LLM | MiniMind 텍스트 학습 결과 `out/llm_768.pth` |
    | tiny SigLIP·단어 tokenizer | `init_vlm_model`의 tokenizer_path·vision_model_path, `model/model_init.md` |
    | 합성 Parquet | `--data_path`, columns `image_bytes`, `conversations` |
    | 정렬 → SFT | `--from_weight llm --freeze_llm 2` → `--from_weight pretrain_vlm --freeze_llm 1` |
    | CPU float32 | GPU 메모리에 맞는 batch·길이·누적수, 지원하는 AMP dtype |

    원본 실행 위치는 `minimind-v/trainer/`입니다. 원본 `--save_dir`는 **출력** 위치만 바꾸고,
    입력 가중치는 `init_vlm_model` 기본값인 `../out/`에서, resume은 `../checkpoints/`에서 읽습니다.
    `strict=False`는 누락 키를 허용하므로 가중치가 실제로 적용됐는지도 점검하세요(6절의 assert 참고).

    **과제 — 예측을 적고 한 변수만 바꾼 뒤 8절 표로 설명하세요.**

    1. **텍스트 LLM 없이:** 6절의 정렬을 `load_vlm("none", freeze=2)`에서 시작하세요.
       무작위 LLM을 동결하면 Projector가 맞춰 갈 기준점이 없습니다. 8절 ② 행은 어떻게 바뀌나요?
    2. **정렬 생략:** README상 정렬은 선택 단계입니다. SFT를 `"llm"` 가중치에서 바로 시작하세요.
       같은 SFT step에서 지시 질문의 바꾼 이미지 확률을 비교하세요.
    3. **SFT 동결 정책:** SFT를 `freeze=2`와 `freeze=0`으로 바꿔 greedy 일치와 7절 변경 텐서 수를 기록하세요.
    4. **이미지 무시 데이터:** 학습 이미지를 모두 흰색으로 바꾸면 ② 정렬은 무엇을 학습할 수 있을까요?
    5. **Projector:** `MMVisionProjector(16, 32, target_tokens=2)`가 패치 수를 줄이지 않는 이유를 소스에서 찾으세요.

    **검증 범위:** CPU float32 축소 학습, 단계별 이미지 사용 검사, 추론 가중치 복원.
    실제 사전 학습 가중치·실데이터·AMP·DDP·MoE·원본 trainer 전체·중간 재개는 별도 검증 대상입니다.
    원본 저작권·라이선스는 `minimind-v/LICENSE`를 따릅니다.

    **다음 실습:** `minimind_omni.py`. 이미지 입력 경로는 같고, 음성 입력(Audio Projector)과
    음성 출력(Talker·8개 코드북)이 추가됩니다.
    """)
    return


if __name__ == "__main__":
    app.run()
