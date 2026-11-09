import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import copy
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
    import torch.nn.functional as F
    from PIL import Image, ImageDraw
    from tokenizers import Tokenizer, models, pre_tokenizers
    from torch.utils.data import DataLoader
    from transformers import (
        PreTrainedTokenizerFast, SiglipImageProcessor,
        SiglipVisionConfig, SiglipVisionModel,
    )

    repo_root = Path(__file__).resolve().parents[1]
    project_dir = repo_root / "minimind-v"
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

    run_id = datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y%m%d-%H%M%S") + f"-{uuid4().hex[:6]}-vlm"
    output_dir = repo_root / "outputs" / "notebooks" / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    datasets.config.HF_DATASETS_CACHE = output_dir / "dataset-cache"
    datasets.disable_progress_bars()
    torch.set_num_threads(2)
    setup_seed(42)
    return (
        DataLoader, F, Image, ImageDraw, PreTrainedTokenizerFast,
        SiglipImageProcessor, SiglipVisionConfig, SiglipVisionModel, Tokenizer,
        VLMConfig, VLMDataset, copy, get_lr, init_vlm_model, io, json, mo, models,
        output_dir, pa, plt, pq, pre_tokenizers, setup_seed, torch, uuid4, vlm_collate_fn,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # Vision LLM 미세 조정: 이미지가 답변의 근거가 되기까지

    텍스트 LLM은 토큰 ID를 임베딩으로 바꿉니다. Vision LLM(VLM)은 **이미지 특징을
    언어 임베딩 차원으로 변환한 뒤 같은 문맥에 넣어** 다음 응답 토큰을 예측합니다.
    질문이 같아도 이미지가 바뀌면 답변이 바뀌어야 합니다.

    | 원본 | 여기서 확인할 내용 |
    | --- | --- |
    | `minimind-v/dataset/lm_dataset.py` | 실제 `VLMDataset`: Parquet → 이미지·대화 → 정답 마스킹 |
    | `minimind-v/trainer/train_pretrain_vlm.py` | Projector 정렬, 누적·clipping·학습률·업데이트 |
    | `minimind-v/trainer/train_sft_vlm.py` | 정렬 가중치에서 시작, 시각 지시 응답 미세 조정 |
    | `model/model_vlm.py`, `trainer/trainer_utils.py` | 실제 forward와 초기화 함수의 동결 정책 |

    **순서:** 샘플 이미지 → chat template → label 표 → 시각 임베딩 → 손실·gradient →
    Projector 정렬 → SFT → 생성·이미지 교환 평가 → 저장·복원.
    텐서 shape와 SFT의 다음 토큰 예측을 알고 시작하세요. 각 절의 **예측 → 실행 → 해설**을 따라갑니다.

    기존 강의와 같은 **marimo 노트북**입니다. 셀 함수의 인자와 return은 셀 사이 데이터를 전달합니다.
    다운로드·GPU 없이 CPU에서 작은 모델을 실제 학습합니다. 합성 이미지와 작은 사전을 직접 만들고,
    **비전 인코더와 LLM 모두 무작위 초기화**합니다. 실제 사전 학습 VLM의 성능 재현은 아닙니다.
    현업에서는 학습된 비전 인코더와 LLM에서 시작합니다.
    산출물: `outputs/notebooks/<실행 ID>/`. 기존 데이터·모델 파일은 수정하지 않습니다.
    """)
    return


@app.cell
def _(Image, ImageDraw, io, json, mo, output_dir, pa, plt, pq):
    def make_record(color, question, answer, offset):
        image = Image.new("RGB", (32, 32), "white")
        ImageDraw.Draw(image).rectangle((offset, 6, offset + 18, 25), fill=color)
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return {
            "image_bytes": buffer.getvalue(),
            "conversations": json.dumps([
                {"role": "system", "content": "그림 도우미"},
                {"role": "user", "content": "<image> " + question},
                {"role": "assistant", "content": answer},
            ], ensure_ascii=False),
        }

    _colors = [("red", "빨강"), ("blue", "파랑")]
    records = {
        "alignment": [make_record(c, "색 ?", a, x) for x in (2, 4, 7, 10) for c, a in _colors],
        "sft": [make_record(c, "무슨 색 ?", f"색 은 {a}", x) for x in (2, 4, 7, 10) for c, a in _colors],
        "validation": [make_record(c, "무슨 색 ?", f"색 은 {a}", x) for x in (3, 8) for c, a in _colors],
    }
    parquet_paths = {}
    for _split, _rows in records.items():
        _path = output_dir / f"{_split}.parquet"
        pq.write_table(pa.Table.from_pylist(_rows), _path)
        parquet_paths[_split] = _path
    _fig, _axes = plt.subplots(1, 2, figsize=(5, 2))
    for _axis, _row, _title in zip(_axes, records["sft"][:2], ["red", "blue"]):
        _axis.imshow(Image.open(io.BytesIO(_row["image_bytes"])))
        _axis.set_title(_title)
        _axis.axis("off")
    mo.vstack([
        mo.md("""
        ## 1. Parquet 한 행은 이미지 bytes + 대화

        **예측:** 질문이 같을 때 빨강·파랑을 구분하려면 어떤 입력이 필요할까요?
        `image_bytes`는 PNG/JPEG 파일의 원시 bytes입니다. `conversations`는 여기서는 JSON 문자열이고
        원본은 dict 리스트도 지원합니다. 역할은 `system/user/assistant`, 발화 키는 `content`입니다.

        정렬 단계는 짧은 색 이름, SFT는 `색 은 빨강` 같은 지시 응답 형식을 학습합니다.
        학습 8개와 검증 4개는 사각형의 위치가 다릅니다. **색·모양·질문은 같은 작은 분포**라서
        새 사물·질문에 대한 일반화를 측정하지 않습니다. 질문에는 색 이름을 주지 않습니다.
        명시적 system으로 원본의 20% 무작위 system 추가를 피하고 think 태그는 쓰지 않습니다.
        """), _fig,
        mo.ui.table([{"split": k, "행 수": len(v), "파일": str(parquet_paths[k])} for k, v in records.items()], selection=None),
        mo.md("```json\n" + json.dumps(json.loads(records["sft"][0]["conversations"]), ensure_ascii=False, indent=2) + "\n```"),
    ])
    return parquet_paths, records


@app.cell
def _(
    PreTrainedTokenizerFast, SiglipImageProcessor, SiglipVisionConfig,
    SiglipVisionModel, Tokenizer, VLMConfig, json, mo, models, output_dir,
    pre_tokenizers, records, setup_seed,
):
    vocabulary = ["[PAD]", "[BOS]", "[EOS]", "[UNK]", "<|image_pad|>", "system", "user", "assistant"] + sorted({
        word for rows in records.values() for row in rows
        for turn in json.loads(row["conversations"]) for word in turn["content"].split()
        if word != "<image>"
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
    _vision = SiglipVisionModel(SiglipVisionConfig(
        hidden_size=16, intermediate_size=32, num_hidden_layers=1,
        num_attention_heads=2, image_size=32, patch_size=16,
    ))
    _vision.save_pretrained(vision_dir)
    SiglipImageProcessor(size={"height": 32, "width": 32}).save_pretrained(vision_dir)
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
    mo.md(f"""
    ## 2. 작은 로컬 tokenizer와 비전 인코더

    원본은 MiniMind tokenizer와 SigLIP 가중치가 필요합니다. 여기서는 로컬 tokenizer와
    작은 SigLIP을 저장해 **동일한 로딩 함수**로 읽습니다. 단어 사전에서는 줄바꿈이 사라져
    assistant 시작 패턴은 `[BOS], assistant`, 종료 패턴은 `[EOS]`입니다.
    실제 서브워드 tokenizer의 ID·길이와는 다릅니다.

    `<image>` 한 개를 `<|image_pad|>` **4개**로 바꿉니다. marker ID는 `{config.image_ids[0]}`입니다.
    원본 기본값 `image_ids=[12]`를 복사하지 않고 실제 tokenizer ID와 맞춥니다.
    이미지 32×32 / 패치 16×16 → 2×2 = 4패치입니다.
    기본 실모델은 256×256 / 패치 32×32 → 8×8 = 64패치입니다.
    **Projector는 차원만 바꾸며 토큰 수를 줄이지 않습니다.**

    LLM은 3층으로 만들어 SFT에서 첫·마지막 층과 고정된 가운데 층을 구분합니다.
    사전 크기 {len(vocabulary)}, 언어 차원 32, 시각 차원 16을 사용합니다.
    """)
    return config, tokenizer, tokenizer_dir, vision_dir


@app.cell
def _(config, init_vlm_model, output_dir, setup_seed, tokenizer_dir, vision_dir):
    def load_vlm(prefix="none", freeze=2, weight_dir=None):
        return init_vlm_model(
            config, from_weight=prefix, tokenizer_path=str(tokenizer_dir),
            vision_model_path=str(vision_dir), save_dir=str(weight_dir or output_dir),
            device="cpu", freeze_llm=freeze,
        )[0]

    setup_seed(42)
    base_model = load_vlm()
    assert base_model.vision_encoder is not None and base_model.processor is not None
    return base_model, load_vlm


@app.cell
def _(DataLoader, VLMDataset, base_model, config, mo, parquet_paths, tokenizer, vlm_collate_fn):
    def read_dataset(path, max_length=40):
        return VLMDataset(
            str(path), tokenizer, preprocess=base_model.processor, max_length=max_length,
            image_special_token=config.image_special_token, image_token_len=config.image_token_len,
        )

    alignment_ds = read_dataset(parquet_paths["alignment"])
    sft_ds = read_dataset(parquet_paths["sft"])
    validation_ds = read_dataset(parquet_paths["validation"])
    ids, labels, pixels = next(iter(DataLoader(sft_ds, batch_size=2, collate_fn=vlm_collate_fn)))
    assert ids.shape == labels.shape == (2, 40)
    assert pixels["pixel_values"].shape == (2, 1, 3, 32, 32)
    assert (ids == config.image_ids[0]).sum(1).tolist() == [4, 4]
    assert (labels[ids == tokenizer.pad_token_id] == -100).all()
    assert (labels[ids == config.image_ids[0]] == -100).all()
    assert (labels[:, 1:] != -100).sum(1).tolist() == [4, 4]
    assert not (ids == tokenizer.unk_token_id).any()
    mo.vstack([
        mo.md(f"""
        ## 3. 실제 VLMDataset: assistant 응답만 정답

        **예측:** 이미지 marker, user 질문, assistant 헤더, EOS 중 어느 것이 정답일까요?
        원본은 `create_chat_prompt → tokenize → 절단 → PAD → generate_labels`를 거칩니다.
        이미지 bytes는 PIL로 열어 processor에 전달합니다.
        **해설:** assistant 본문과 종료 패턴만 labels에 남고 나머지는 `-100`입니다.
        입력에는 질문과 답변 모두 남습니다. 질문과 이미지는 응답 예측의 문맥입니다.

        `input_ids`와 `labels`: `{tuple(ids.shape)}` = [B,T].
        processor의 샘플별 [N,C,H,W]를 `vlm_collate_fn`이 쌓으면
        `pixel_values['pixel_values']`: `{tuple(pixels['pixel_values'].shape)}` = [B,N,C,H,W].
        B=2와 이미지 개수 N=1을 구분하세요.
        """),
        mo.ui.table([
            {"위치": t, "입력": tokenizer.convert_ids_to_tokens(int(x)), "label": int(y), "정답": int(y) != -100}
            for t, (x, y) in enumerate(zip(ids[0], labels[0]))
        ], selection=None),
    ])
    return alignment_ds, ids, labels, pixels, read_dataset, sft_ds, validation_ds


@app.cell
def _(base_model, config, ids, mo, pixels, torch):
    base_model.eval()
    with torch.no_grad():
        features = base_model.get_image_embeddings(
            {k: v.flatten(0, 1) for k, v in pixels.items()}, base_model.vision_encoder,
        )
        projected = base_model.vision_proj(features)
        text_embeddings = base_model.model.embed_tokens(ids)
        merged = base_model.count_vision_proj(ids, text_embeddings, projected, seqlen=ids.shape[1])
    assert features.shape == (2, 4, 16) and projected.shape == (2, 4, 32)
    assert merged.shape == (2, 40, 32)
    _marker = ids == config.image_ids[0]
    torch.testing.assert_close(merged[~_marker], text_embeddings[~_marker])
    torch.testing.assert_close(merged[_marker].reshape(2, 4, 32), projected)
    mo.vstack([
        mo.md("""
        ## 4. 이미지와 텍스트는 어디에서 만날까?

        **예측:** 이미지를 넣으면 문장 길이가 4만큼 더 길어질까요?
        **해설:** marker 4자리 임베딩을 시각 특징으로 **교체**하므로 길이 T는 그대로입니다.
        텍스트 위치 임베딩 보존과 이미지 위치 교체를 실제 원본 메서드로 검사했습니다.
        Projector는 `LayerNorm → Linear → GELU → Linear`이며 시각 차원 16을 언어 차원 32로 바꿉니다.
        이후는 같은 causal Transformer입니다. 이미지를 앞에서 읽고 답변을 뒤에서 예측합니다.
        """),
        mo.mermaid("""flowchart TD
            A["PNG bytes → PIL → processor"] --> B["pixels: B,N,3,32,32"]
            B --> C["고정 SigLIP: B×N,4,16"]
            C --> D["학습 Projector: B×N,4,32"]
            E["input_ids: B,40"] --> F["토큰 임베딩: B,40,32"]
            D --> G["image marker 4자리 교체"]
            F --> G
            G --> H["causal LLM → logits: B,40,V"]
            H --> I["assistant 다음 토큰 CE"]
        """),
    ])
    return


@app.cell
def _(F, base_model, copy, ids, labels, mo, pixels, torch):
    _probe = copy.deepcopy(base_model).eval()
    _result = _probe(ids, labels=labels, pixel_values=pixels)
    _manual = F.cross_entropy(
        _result.logits[:, :-1].reshape(-1, _result.logits.shape[-1]), labels[:, 1:].reshape(-1), ignore_index=-100,
    )
    torch.testing.assert_close(_result.loss, _manual)
    (_result.loss + _result.aux_loss).backward()
    _rows = []
    for _prefix in ["vision_encoder", "vision_proj", "model", "lm_head"]:
        _parameters = [p for n, p in _probe.named_parameters() if n.startswith(_prefix + ".")]
        _rows.append({"부분": _prefix, "학습 매개변수 수": sum(p.numel() for p in _parameters if p.requires_grad),
                      "gradient 텐서 수": sum(p.grad is not None for p in _parameters),
                      "gradient norm 합": sum(p.grad.norm().item() for p in _parameters if p.grad is not None)})
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in _probe.vision_proj.parameters())
    assert all(p.grad is None for p in _probe.vision_encoder.parameters())
    assert all(p.grad is None for p in _probe.model.parameters())
    mo.vstack([
        mo.md(f"""
        ## 5. 다음 토큰 손실과 gradient 경로

        **예측:** LLM을 고정하면 앞에 붙인 Projector도 학습할 수 없을까요?
        원본 forward가 `logits[:, :-1]`와 `labels[:, 1:]`를 비교합니다.
        직접 계산한 CE와 모델 CE `{_manual.item():.6f}`가 일치합니다.
        Dataset에서 labels를 미리 이동하지 마세요. 미래 토큰은 causal mask로 가립니다.

        `loss = res.loss + res.aux_loss`는 두 trainer에서 같습니다.
        이 실습은 Dense여서 aux는 0입니다. MoE에서는 계수가 반영된 aux를 한 번 더합니다.
        **해설:** LLM 매개변수를 고정해도 입력에 대한 미분은 필요합니다.
        LLM 연산을 통해 Projector까지 gradient가 흐릅니다. LLM 전체 forward를 `no_grad()`로
        감싸면 이 경로를 끊습니다. 비전 인코더만 원본 `get_image_embeddings`에서 `no_grad()`로 실행합니다.
        """), mo.ui.table(_rows, selection=None),
        mo.ui.table([
            {"logits 위치": t - 1, "입력 ID": int(ids[0, t - 1]), "예측 정답 ID": int(labels[0, t])}
            for t in range(1, ids.shape[1]) if labels[0, t] != -100
        ], selection=None),
    ])
    return


@app.cell
def _(load_vlm, mo):
    _rows = []
    for _policy in [0, 1, 2]:
        _model = load_vlm(freeze=_policy)
        _names = [n for n, p in _model.named_parameters() if p.requires_grad]
        assert all(not n.startswith("vision_encoder.") for n in _names)
        if _policy == 1:
            assert not any(n.startswith("model.layers.1.") for n in _names)
            assert any(n.startswith("model.layers.0.") for n in _names)
            assert any(n.startswith("model.layers.2.") for n in _names)
        _rows.append({"freeze_llm": _policy, "학습 매개변수 수": sum(p.numel() for p in _model.parameters() if p.requires_grad),
                      "범위": {0: "Projector + LLM 전체", 1: "Projector + LLM 첫/마지막 층", 2: "Projector만"}[_policy]})
    mo.vstack([
        mo.md("""
        ## 6. Projector 정렬과 시각 지시 SFT

        두 trainer는 모두 **이미지·대화 → assistant 다음 토큰 CE**를 사용합니다.
        `pretrain`이라는 이름이지만 이 단계도 이미지·대응 문장을 이용한 지도 학습입니다.
        이미지-텍스트 대조 학습이나 비전 인코더 재학습이 아닙니다.

        | 원본 기본값 | Projector 정렬 | 시각 지시 SFT |
        | --- | --- | --- |
        | 데이터 | pretrain_i2t.parquet | sft_i2t.parquet |
        | 시작 가중치 | llm | pretrain_vlm |
        | freeze_llm | 2 | 1 |
        | learning_rate | 4e-4 | 5e-6 |
        | batch_size / max_seq_len | 16 / 450 | 4 / 768 |
        | epochs | 2 | 2 |

        비전 인코더는 모든 정책에서 고정합니다. 정책 1은 임베딩·중간 층·최종 norm까지
        전부 학습하는 Full SFT가 아닙니다. 아래는 **원본 init_vlm_model의 requires_grad**를 집계한 표입니다.
        다음 실습은 작은 무작위 모델이라 학습률·step을 다르게 설정합니다.
        """), mo.ui.table(_rows, selection=None),
    ])
    return


@app.cell
def _(config, mo, parquet_paths, read_dataset, sft_ds, tokenizer):
    _short = read_dataset(parquet_paths["sft"], max_length=8)
    _ids, _labels, _ = _short[0]
    assert int((_labels[1:] != -100).sum()) == 0
    _unclosed = tokenizer("[BOS]assistant\n색 은 빨강", add_special_tokens=False).input_ids
    _padded = _unclosed + [tokenizer.pad_token_id] * (sft_ds.max_length - len(_unclosed))
    _unclosed_labels = sft_ds.generate_labels(_padded)
    _pad_targets = sum(y != -100 for x, y in zip(_padded, _unclosed_labels) if x == tokenizer.pad_token_id)
    mo.md(f"""
    ## 7. 학습 전 데이터 점검

    **예측:** `max_length=8`로 줄이면 무엇을 잃을까요?
    이 샘플의 유효 정답은 **0개**입니다. 모든 labels가 -100이면 평균 CE가 NaN이 될 수 있습니다.
    marker도 중간에 잘릴 수 있습니다. 길이 예산에는 이미지당 `{config.image_token_len}`자리도 포함합니다.
    `image_token_len`만 바꿔서는 실제 패치 수가 변하지 않습니다.

    원본 `generate_labels`는 종료 패턴을 못 찾으면 끝까지 정답으로 삼습니다.
    인위적으로 EOS 없는 assistant 문자열 뒤에 PAD를 넣으면 **PAD {_pad_targets}개까지 정답**에 포함됩니다.
    정상 chat template로 종료 토큰을 넣고, 절단 뒤 `labels[:,1:]`의 유효 정답 수를 검사하세요.

    복수 이미지는 marker 구간과 bytes 순서가 일치해야 합니다. 현재 collate는 stack을 사용하므로
    배치 안 이미지 개수가 같아야 합니다. marker 구간 둘을 붙이면 하나로 인식하므로 사이에 텍스트가 필요합니다.
    여기서는 processor의 **딕셔너리 경로**를 사용합니다.
    """)
    return


@app.cell
def _(mo):
    alignment_steps = mo.ui.slider(20, 160, value=80, step=20, label="Projector 정렬 optimizer step")
    sft_steps = mo.ui.slider(20, 160, value=80, step=20, label="SFT optimizer step")
    mo.vstack([
        mo.md("""
        ## 8. 두 단계 학습 실행

        설정을 바꾸면 같은 seed의 초기 모델부터 다시 학습합니다. 원본 루프를 CPU float32·단일 프로세스로
        축소하며 실제 모델·데이터셋·초기화·collate·학습률 함수를 사용합니다. 전역 args/model/optimizer에
        의존하는 원본 `train_epoch` 자체는 호출하지 않습니다.

        한 optimizer step = microbatch 4개 × 누적 2회 = 8샘플입니다.
        `zero_grad → forward → (CE+aux)/누적수 → backward → clip → step`을 관찰하세요.
        여기서는 응답 길이가 같고 누적 그룹도 완전합니다. 원본처럼 고정 누적수로 나누면 마지막 작은
        그룹은 gradient가 작아질 수 있고, 응답 길이가 다르면 microbatch 평균과 토큰 전체 평균도 달라집니다.
        """), alignment_steps, sft_steps,
    ])
    return alignment_steps, sft_steps


@app.cell
def _(DataLoader, get_lr, torch, vlm_collate_fn):
    @torch.no_grad()
    def evaluate(model, dataset):
        model.eval()
        ce_sum, token_count = 0.0, 0
        for x, y, pv in DataLoader(dataset, batch_size=4, collate_fn=vlm_collate_fn):
            count = int((y[:, 1:] != -100).sum())
            assert count > 0
            ce_sum += model(x, labels=y, pixel_values=pv).loss.item() * count
            token_count += count
        return ce_sum / token_count

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
            loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0, collate_fn=vlm_collate_fn)
            accumulation = len(loader)  # 8행 / batch=4 → 항상 완전한 2회 누적
            ce_value = 0.0
            for x, y, pv in loader:
                assert (y[:, 1:] != -100).any(1).all()
                result = model(x, labels=y, pixel_values=pv)
                assert torch.isfinite(result.loss)
                loss = (result.loss + result.aux_loss) / accumulation
                loss.backward()
                ce_value += result.loss.item() / accumulation
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            assert torch.isfinite(grad_norm)
            optimizer.step()
            history.append({"step": step + 1, "ce": ce_value, "lr": current_lr, "grad_norm_before_clip": float(grad_norm)})
        changes = []
        for name, p in model.named_parameters():
            delta = float((p.detach() - initial[name]).abs().max())
            if not p.requires_grad:
                assert delta == 0, name
            changes.append({"name": name, "trainable": p.requires_grad, "max_abs_delta": delta})
        assert any(r["max_abs_delta"] > 0 for r in changes if r["name"].startswith("vision_proj."))
        return history, changes

    return evaluate, fit_stage


@app.cell
def _(
    alignment_ds, alignment_steps, base_model, config, copy, evaluate, fit_stage,
    json, load_vlm, output_dir, setup_seed, sft_ds, sft_steps, torch, uuid4, validation_ds,
):
    training_dir = output_dir / f"training-{uuid4().hex[:6]}"
    training_dir.mkdir()
    setup_seed(42)
    aligned_model = copy.deepcopy(base_model)
    base_validation_ce = evaluate(aligned_model, validation_ds)
    alignment_before = evaluate(aligned_model, alignment_ds)
    alignment_history, alignment_changes = fit_stage(aligned_model, alignment_ds, alignment_steps.value, 2e-2)
    alignment_after = evaluate(aligned_model, alignment_ds)
    aligned_validation_ce = evaluate(aligned_model, validation_ds)

    def save_weights(model, prefix):
        # 원본처럼 encoder를 제외합니다. CPU 수치 비교를 위해 float32를 유지합니다.
        path = training_dir / f"{prefix}_{config.hidden_size}.pth"
        torch.save({k: v.detach().cpu() for k, v in model.state_dict().items() if not k.startswith("vision_encoder.")}, path)
        return path

    alignment_weight_path = save_weights(aligned_model, "pretrain_vlm")
    sft_model = load_vlm("pretrain_vlm", freeze=1, weight_dir=training_dir)
    for _name, _p in aligned_model.state_dict().items():
        torch.testing.assert_close(sft_model.state_dict()[_name], _p, rtol=0, atol=0)
    sft_history, sft_changes = fit_stage(sft_model, sft_ds, sft_steps.value, 3e-3)
    final_validation_ce = evaluate(sft_model, validation_ds)
    sft_weight_path = save_weights(sft_model, "sft_vlm")
    stage_metrics = {
        "seed": 42, "alignment_steps": alignment_steps.value, "sft_steps": sft_steps.value,
        "alignment_lr": 2e-2, "sft_lr": 3e-3, "microbatch_size": 4,
        "accumulation_steps": 2, "grad_clip": 1.0, "dtype": "float32", "device": "cpu",
        "alignment_train_ce_before": alignment_before, "alignment_train_ce_after": alignment_after,
        "sft_validation_ce_base": base_validation_ce, "sft_validation_ce_aligned": aligned_validation_ce,
        "sft_validation_ce_final": final_validation_ce,
        "alignment_history": alignment_history, "sft_history": sft_history,
        "alignment_changes": alignment_changes, "sft_changes": sft_changes,
    }
    (training_dir / "metrics.json").write_text(json.dumps(stage_metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    return (
        aligned_model, alignment_changes, alignment_history, sft_changes, sft_history,
        sft_model, sft_weight_path, stage_metrics, training_dir,
    )


@app.cell
def _(alignment_changes, alignment_history, mo, plt, sft_changes, sft_history, stage_metrics, training_dir):
    _fig, _axes = plt.subplots(1, 2, figsize=(10, 3))
    for _axis, _history, _title in zip(_axes, [alignment_history, sft_history], ["Projector alignment", "Visual instruction SFT"]):
        _axis.plot([r["step"] for r in _history], [r["ce"] for r in _history])
        _axis.set(xlabel="Optimizer step", ylabel="Train response CE", title=_title)
        _axis.grid(alpha=0.3)
    _fig.tight_layout()
    _fig.savefig(training_dir / "loss-curves.png", dpi=140)
    _changes = []
    for _stage, _items in [("정렬", alignment_changes), ("SFT", sft_changes)]:
        for _prefix in ["vision_encoder.", "vision_proj.", "model.layers.0.", "model.layers.1.", "model.layers.2.", "model.embed_tokens.", "model.norm."]:
            _subset = [r for r in _items if r["name"].startswith(_prefix)]
            _changes.append({"단계": _stage, "부분": _prefix, "학습 텐서 수": sum(r["trainable"] for r in _subset),
                             "변경 텐서 수": sum(r["max_abs_delta"] > 0 for r in _subset)})
    mo.vstack([
        mo.md(f"""
        ## 9. 손실과 실제 가중치 변화

        정렬 데이터 응답 CE: **{stage_metrics['alignment_train_ce_before']:.4f} → {stage_metrics['alignment_train_ce_after']:.4f}**.
        동일한 SFT 검증 CE는 초기 **{stage_metrics['sft_validation_ce_base']:.4f}**,
        정렬 뒤 **{stage_metrics['sft_validation_ce_aligned']:.4f}**,
        SFT 뒤 **{stage_metrics['sft_validation_ce_final']:.4f}**입니다.

        두 단계의 응답 형식·길이가 달라 두 학습 그래프의 높이를 직접 비교하지 않습니다.
        검증 CE는 **유효 정답 토큰 수로 가중한 평균**이고 학습 그래프는 업데이트 직전 microbatch CE입니다.
        변경 여부는 학습 전후 값을 비교했습니다. 고정 가중치는 변화가 0인지 검사합니다.
        `eval()`은 dropout 등의 동작을 정하고 `requires_grad`를 바꾸지 않습니다.
        """), _fig, mo.ui.table(_changes, selection=None),
    ])
    return


@app.cell
def _(aligned_model, base_model, json, mo, sft_model, tokenizer, torch, training_dir, validation_ds):
    @torch.no_grad()
    def response_probe(model, dataset):
        model.eval()
        rows = []
        for index in range(len(dataset)):
            x, y, pv = dataset[index]
            other_pixels = dataset[index ^ 1][2]  # 같은 위치, 반대 색 이미지
            correct = model(x[None], labels=y[None], pixel_values={k: v[None] for k, v in pv.items()}).loss.item()
            swapped = model(x[None], labels=y[None], pixel_values={k: v[None] for k, v in other_pixels.items()}).loss.item()
            start = int(torch.where(y != -100)[0][0])
            prompt = x[:start][None]  # 정답과 PAD를 제외하고 assistant 헤더까지만 입력
            generated = []
            for _ in range(6):
                # 이해를 위해 매번 전체 문맥을 재계산하는 greedy 생성입니다.
                result = model(prompt, pixel_values={k: v[None] for k, v in pv.items()})
                token = int(result.logits[0, -1].argmax())
                generated.append(token)
                prompt = torch.cat([prompt, torch.tensor([[token]])], dim=1)
                if token == tokenizer.eos_token_id:
                    break
            expected_ids = y[y != -100].tolist()
            rows.append({
                "sample": index, "정답": tokenizer.decode(expected_ids, skip_special_tokens=True),
                "greedy 응답": tokenizer.decode(generated, skip_special_tokens=True),
                "EOS 종료": generated[-1] == tokenizer.eos_token_id,
                "응답+EOS 정확히 일치": generated == expected_ids,
                "정상 이미지 CE": correct, "반대 이미지 CE": swapped, "반대-정상 CE": swapped - correct,
            })
        return rows

    probe_results = {k: response_probe(m, validation_ds) for k, m in [("초기", base_model), ("정렬 뒤", aligned_model), ("SFT 뒤", sft_model)]}
    (training_dir / "generation-and-image-swap.json").write_text(json.dumps(probe_results, ensure_ascii=False, indent=2), encoding="utf-8")
    mo.vstack([
        mo.md("""
        ## 10. 이미지 정보를 사용했는지 검사하기

        **예측:** 빨간 이미지의 정답은 유지하고 이미지만 파란색으로 바꾸면 손실은 어떻게 될까요?
        `반대-정상 CE > 0`이면 이 표본에서 정상 이미지가 정답 예측에 더 유리했다는 뜻입니다.
        CE 감소만으로 이미지 사용을 단정하지 말고 이미지 교환과 실제 생성도 함께 읽으세요.
        초기 모델도 약간의 차이는 낼 수 있으므로 전후를 비교합니다.

        생성은 assistant 헤더까지만 입력하고 자기 예측을 이어 붙입니다. CE는 앞선 정답을 제공하는
        teacher forcing입니다. **낮은 CE와 greedy 정답 일치는 다른 지표**입니다.
        6토큰 안에 EOS가 나왔는지도 기록합니다. 정렬은 짧은 응답만 배워 SFT 형식과 다를 수 있습니다.
        결과는 두 색 사각형 과제의 관측값이며 범용 이미지 이해 정확도가 아닙니다.
        """), mo.ui.table([dict(단계=k, **r) for k, rows in probe_results.items() for r in rows], selection=None),
    ])
    return (probe_results,)


@app.cell
def _(ids, json, load_vlm, mo, pixels, sft_model, sft_weight_path, torch, training_dir):
    _restored = load_vlm("sft_vlm", freeze=1, weight_dir=training_dir).eval()
    sft_model.eval()
    with torch.no_grad():
        _before = sft_model(ids, pixel_values=pixels).logits
        _after = _restored(ids, pixel_values=pixels).logits
    torch.testing.assert_close(_before, _after, atol=0, rtol=0)
    _keys = torch.load(sft_weight_path, map_location="cpu", weights_only=True)
    assert not any(k.startswith("vision_encoder.") for k in _keys)
    (training_dir / "restore-check.json").write_text(json.dumps({"logits_exactly_equal": True, "vision_encoder_in_weight": False}), encoding="utf-8")
    mo.md(f"""
    ## 11. 저장과 복원

    저장 파일: `{sft_weight_path}`.
    원본 initializer로 다시 읽어 같은 배치의 logits가 **정확히 일치**하는지 검사했습니다.
    원본처럼 `vision_encoder.*`를 제외하지만 CPU 실습에서는 float32를 유지합니다.
    원본 trainer의 half 저장과 수치 오차는 다를 수 있습니다.

    추론에는 **동일한 비전 인코더·processor·tokenizer·VLMConfig**도 필요합니다.
    실행 폴더의 `tiny-siglip/`, `tokenizer/`, `config/`에 함께 남깁니다.
    이 파일은 optimizer/scaler·학습 시점이 없어 중간 재개용이 아닙니다.
    원본 `vlm_checkpoint`는 optimizer·epoch·step 등이 들어 있는 resume 파일을 별도로 저장합니다.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## 12. 실제 데이터로 확장하기와 과제

    | 이번 실습 | 원본 trainer에서 확인할 설정 |
    | --- | --- |
    | tiny SigLIP·word tokenizer | `init_vlm_model`의 tokenizer_path·vision_model_path, `model/model_init.md` |
    | 합성 Parquet | `--data_path`, columns `image_bytes`, `conversations` |
    | 작은 3층 모델 | 체크포인트에 맞는 hidden_size·num_hidden_layers·tokenizer·비전 설정 |
    | 정렬 → SFT | `--from_weight llm` → `--from_weight pretrain_vlm` |
    | freeze 2 → 1 | `--freeze_llm 2` → `--freeze_llm 1` |
    | CPU float32 | GPU 메모리에 맞는 batch·길이·누적수, 지원하는 AMP dtype |

    원본 실행 위치는 `minimind-v/trainer/`입니다. 원본 `--save_dir`는 배포 가중치의 **출력**을
    바꾸지만 `init_vlm_model`에 전달되지 않아 입력 가중치는 여전히 `../out/`에서 읽습니다.
    resume 출력도 `../checkpoints/`로 고정됩니다. 이 노트북은 함수에 절대 경로를 넘겨
    산출물을 `outputs/notebooks/<실행 ID>/`에 저장합니다. CLI로 확장할 때도 입력·출력·resume
    경로를 먼저 확인하세요. `strict=False`는 누락 키를 허용하므로 가중치 적용 여부도 점검합니다.

    **과제 — 예측을 적고 한 변수만 바꾼 뒤 관측값으로 설명하세요.**

    1. 정렬 step을 20과 160으로 바꿔 정렬 CE, SFT 전 검증 CE, 이미지 교환 CE 차이를 비교하세요.
       짧은 응답 학습이 긴 응답 형식까지 보장할까요?
    2. SFT를 `freeze=2`로 바꿔 정책 1과 비교하세요. 검증 CE, 정확 일치 수, 변경 텐서 수를 기록하세요.
       정책 0에서는 어디까지 바뀔까요?
    3. 이미지를 모두 흰색으로 바꿔 같은 질문에 다른 정답을 학습하세요.
       응답 패턴 암기와 이미지로 정답 구분하기를 분리해 설명하세요.
    4. max_length를 줄이고 실제 정답 수·marker 수·EOS 보존을 기록하세요.
       all-ignore labels는 학습에 넣지 않습니다.

    **해설 기준:** 정답 마스크·다음 토큰 이동, 비전→Projector→LLM의 gradient,
    동결·갱신, EOS 포함 생성, 이미지 교환의 영향을 연결해 설명하세요.
    검증 범위는 CPU float32 축소 학습과 추론 가중치 복원입니다.
    실제 사전 학습 가중치·실데이터·AMP·DDP·MoE·원본 trainer 전체·중간 재개는 별도 검증 대상입니다.
    원본 저작권·라이선스는 `minimind-v/LICENSE`를 따릅니다.
    """)
    return


if __name__ == "__main__":
    app.run()
