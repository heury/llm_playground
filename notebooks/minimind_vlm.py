import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import copy
    import json
    import sys
    from datetime import datetime
    from io import BytesIO
    from pathlib import Path
    from uuid import uuid4
    from zoneinfo import ZoneInfo

    import marimo as mo
    import matplotlib.pyplot as plt
    import torch
    import torch.nn.functional as F
    from PIL import Image, ImageDraw
    from datasets import Dataset as HFDataset
    from datasets import config as datasets_config
    from transformers import AutoTokenizer, SiglipImageProcessor, SiglipVisionConfig, SiglipVisionModel

    repo_root = Path(__file__).resolve().parents[1]
    project_dir = repo_root / "minimind-v"
    # 다른 MiniMind 노트북과 model/dataset 패키지 이름이 같습니다.
    for _name in ("model", "dataset"):
        _loaded = sys.modules.get(_name)
        if _loaded is not None and getattr(_loaded, "__file__", None):
            if not Path(_loaded.__file__).resolve().is_relative_to(project_dir):
                raise RuntimeError("다른 프로젝트 모듈이 로드되었습니다. 이 노트북을 새 프로세스에서 실행하세요.")
    sys.path.insert(0, str(project_dir))
    from model.model_vlm import MMVisionProjector, MiniMindVLM, VLMConfig
    from dataset.lm_dataset import VLMDataset

    torch.set_num_threads(2)
    run_id = datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y%m%d-%H%M%S") + "-" + uuid4().hex[:6] + "-vlm"
    artifact_dir = repo_root / "outputs" / "notebooks" / run_id
    artifact_dir.mkdir(parents=True, exist_ok=True)
    # 이 노트북 프로세스의 Parquet 로딩 캐시도 실행 산출물에 보관합니다.
    datasets_config.HF_DATASETS_CACHE = artifact_dir / "datasets_cache"
    return (
        AutoTokenizer, BytesIO, F, HFDataset, Image, ImageDraw,
        MMVisionProjector, MiniMindVLM, SiglipImageProcessor,
        SiglipVisionConfig, SiglipVisionModel, VLMDataset, VLMConfig,
        artifact_dir, copy, json, mo, plt, project_dir, torch,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # MiniMind-V: 이미지가 언어 모델의 입력이 되는 과정

    **학습 목표:** `minimind-v/model/model_vlm.py`의 실제 클래스로 이미지 전처리,
    패치 특징, 차원 투영, 임베딩 교체, 답변 학습과 생성까지 연결합니다.
    Python·텐서 shape·다음 토큰 예측을 알고 시작하세요. 모델 구조/SFT 노트북을 먼저 읽으면 좋습니다.

    ```text
    이미지 → processor → SigLIP → Projector ──────┐
                                                 ↓ 이미지 자리 교체
    질문 → tokenizer → 토큰 ID → 텍스트 Embedding → Decoder → LM head → 다음 토큰
    ```

    이 구현은 **이미지 특징을 입력 시퀀스에 넣고 causal self-attention으로 읽습니다.**
    별도 cross-attention 층은 추가하지 않습니다. 질문 뒤의 답변 위치는 앞선 이미지·질문을 볼 수 있습니다.
    이미지를 어휘 사전의 단어 ID로 변환하는 방식도 아닙니다.

    **진행:** 각 절의 예측 → 실행 → 관찰 → 해설을 따라가세요.
    작은 SigLIP과 LLM을 무작위 초기화하므로 GPU·모델 다운로드·API 키 없이 실행합니다.
    생성되는 토큰은 인식 성능이 아닙니다. 실제 프로젝트는 학습된 인코더와 언어 백본을 사용합니다.
    실습 파일은 `outputs/notebooks/<실행 ID>/`에 저장되며 기존 체크포인트는 수정하지 않습니다.
    `@app.cell`과 함수의 인자/return은 marimo 셀 연결이고 `_` 변수는 셀 내부 값입니다.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. 이미지와 패치: 몇 개의 시각 토큰이 생길까?

    **예측:** 32×32 이미지를 16×16 패치로 나누면 패치는 몇 개일까요?
    원본 기본 설정은 256×256 / 32×32 = 8×8 = 64개입니다. 실습에서는 2×2 = 4개로 축소합니다.
    `B`: 배치, `N`: 이미지 수, `P`: 패치 수, `Dv`: 비전 차원, `D`: 언어 차원, `T`: 시퀀스 길이.
    아래 점선은 패치 경계입니다. SigLIP의 patch embedding 이후 비전 self-attention이 패치 간 문맥을 섞으므로,
    최종 특징 하나가 해당 사각형의 픽셀만 담는다고 해석하지 마세요.
    """)
    return


@app.cell
def _(Image, ImageDraw, mo, plt):
    red_image = Image.new("RGB", (32, 32), "white")
    ImageDraw.Draw(red_image).rectangle((4, 4, 27, 27), fill="red")
    blue_image = Image.new("RGB", (32, 32), "white")
    ImageDraw.Draw(blue_image).ellipse((4, 4, 27, 27), fill="blue")
    _fig, _axes = plt.subplots(1, 2, figsize=(6, 3))
    for _ax, _image, _title in zip(_axes, [red_image, blue_image], ["A: red square", "B: blue circle"]):
        _ax.imshow(_image)
        _ax.axhline(15.5, color="black", linestyle="--")
        _ax.axvline(15.5, color="black", linestyle="--")
        _ax.set_title(_title)
        _ax.set_xticks([0, 16, 31])
        _ax.set_yticks([0, 16, 31])
    _fig.tight_layout()
    mo.vstack([_fig, mo.md("**관찰:** 이미지마다 4개 패치입니다. A/B는 색과 모양이 함께 달라지므로 두 속성의 효과를 분리하는 데이터는 아닙니다.")])
    return blue_image, red_image


@app.cell
def _(mo):
    mo.md("""
    ## 2. 실제 클래스의 축소 모델을 만들기

    `SiglipVisionModel`을 로컬에 저장한 뒤 `MiniMindVLM`의 실제 `get_vision_model`로 읽습니다.
    학습된 가중치를 내려받는 단계는 없습니다. 비전 차원 16 → 언어 차원 32로 연결합니다.
    `image_token_len=4`, 인코더 패치 수 4, 연속 이미지 marker 수 4가 모두 맞아야 합니다.
    경로가 없으면 원본 로더는 `(None, None)`을 반환하므로 이미지 실습에서는 로딩 성공을 확인합니다.
    """)
    return


@app.cell
def _(MiniMindVLM, SiglipImageProcessor, SiglipVisionConfig, SiglipVisionModel, VLMConfig, artifact_dir, mo, torch):
    torch.manual_seed(42)
    _vision_dir = artifact_dir / "tiny_siglip"
    _vision = SiglipVisionModel(SiglipVisionConfig(
        image_size=32, patch_size=16, hidden_size=16,
        intermediate_size=32, num_hidden_layers=1, num_attention_heads=2,
    ))
    _vision.save_pretrained(_vision_dir)
    SiglipImageProcessor(size={"height": 32, "width": 32}).save_pretrained(_vision_dir)
    vlm_config = VLMConfig(
        hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, vocab_size=64,
        max_position_embeddings=256, image_hidden_size=16, image_token_len=4,
        image_ids=[12], dropout=0.0, flash_attn=False,
    )
    base_vlm = MiniMindVLM(vlm_config, vision_model_path=str(_vision_dir)).eval()
    assert base_vlm.vision_encoder is not None and base_vlm.processor is not None
    assert all(not _p.requires_grad for _p in base_vlm.vision_encoder.parameters())
    mo.ui.table([
        {"구성": _name, "매개변수 수": sum(_p.numel() for _p in _module.parameters()),
         "학습 가능 수": sum(_p.numel() for _p in _module.parameters() if _p.requires_grad)}
        for _name, _module in [("SigLIP", base_vlm.vision_encoder), ("Projector", base_vlm.vision_proj),
                                ("언어 backbone", base_vlm.model), ("LM head", base_vlm.lm_head)]
    ], selection=None)
    return base_vlm, vlm_config


@app.cell
def _(mo):
    mo.md("""
    ## 3. 픽셀 → 시각 특징 → 언어 차원

    **예측:** Projector를 거치면 패치 수와 마지막 차원 중 무엇이 바뀔까요?
    `image2tensor`는 resize·정규화한 processor 딕셔너리를 반환합니다.
    `get_image_embeddings`는 `torch.no_grad()` 안에서 SigLIP의 `last_hidden_state`를 읽습니다.

    Projector는 `LayerNorm(16) → Linear(16,32) → GELU → Linear(32,32)`입니다.
    모든 패치에 같은 MLP를 적용하고 패치 간 Attention은 수행하지 않습니다.
    생성자에 있는 `source_tokens`, `target_tokens`는 현재 토큰 수를 변경하지 않습니다.
    """)
    return


@app.cell
def _(MMVisionProjector, base_vlm, blue_image, mo, red_image):
    red_pixels = base_vlm.image2tensor(red_image, base_vlm.processor)
    blue_pixels = base_vlm.image2tensor(blue_image, base_vlm.processor)
    vision_features = base_vlm.get_image_embeddings(red_pixels, base_vlm.vision_encoder)
    projected_features = base_vlm.vision_proj(vision_features)
    assert tuple(vision_features.shape) == (1, 4, 16)
    assert tuple(projected_features.shape) == (1, 4, 32)
    assert not vision_features.requires_grad and projected_features.requires_grad
    assert MMVisionProjector(16, 32, target_tokens=2)(vision_features).shape[1] == 4
    mo.ui.table([
        {"단계": "processor", "shape": str(tuple(red_pixels["pixel_values"].shape)), "gradient": False},
        {"단계": "SigLIP", "shape": str(tuple(vision_features.shape)), "gradient": vision_features.requires_grad},
        {"단계": "Projector", "shape": str(tuple(projected_features.shape)), "gradient": projected_features.requires_grad},
    ], selection=None)
    return blue_pixels, projected_features, red_pixels


@app.cell
def _(mo):
    mo.md(r"""
    ## 4. IMAGE 자리를 이미지 벡터로 교체하기

    여기부터 8절까지는 연산을 읽기 위한 **가상 ID 사전**입니다.
    `1=BOS`, `12=IMAGE`, `20=질문`, `21=답변`, `2=EOS`이며 실제 한국어 토크나이저 결과가 아닙니다.
    실제 tokenizer와 `VLMDataset`은 9절에서 연결합니다.

    ```text
    input_ids: [ BOS, IMAGE, IMAGE, IMAGE, IMAGE, 질문, 답변, EOS ]
    embedding: [  e0,    e1,    e2,    e3,    e4,   e5,   e6,  e7 ]
    교체 후:   [  e0,    v0,    v1,    v2,    v3,   e5,   e6,  e7 ]
    ```

    **예측:** 교체한 뒤 시퀀스 길이는 8일까요, 12일까요?
    `count_vision_proj`는 IMAGE가 연속된 구간을 찾아 같은 자리의 벡터를 **교체**합니다.
    이 실습은 marker와 패치 수를 사전에 확인합니다. 원본 함수가 모든 불일치를 검증해 주지는 않습니다.
    """)
    return


@app.cell
def _(base_vlm, mo, plt, projected_features, torch):
    toy_ids = torch.tensor([[1, 12, 12, 12, 12, 20, 21, 2]])
    text_embeddings = base_vlm.model.embed_tokens(toy_ids)
    merged_embeddings = base_vlm.count_vision_proj(
        toy_ids, text_embeddings, projected_features, seqlen=toy_ids.shape[1],
    )
    assert int((toy_ids == 12).sum()) == projected_features.shape[1]
    assert merged_embeddings.shape == text_embeddings.shape
    torch.testing.assert_close(merged_embeddings[:, 1:5], projected_features)
    torch.testing.assert_close(merged_embeddings[:, [0, 5, 6, 7]], text_embeddings[:, [0, 5, 6, 7]])
    _fig, _ax = plt.subplots(figsize=(8, 3))
    _plot = _ax.imshow(merged_embeddings[0].detach().numpy(), aspect="auto", cmap="coolwarm")
    _ax.set_yticks(range(8), ["BOS", "image 0", "image 1", "image 2", "image 3", "question", "answer", "EOS"])
    _ax.set_xlabel("Embedding dimension (32)")
    _fig.colorbar(_plot, ax=_ax)
    _fig.tight_layout()
    mo.vstack([_fig, mo.md("**확인:** 길이 8을 유지하며 이미지 위치 4개만 교체했습니다. 색은 임베딩 성분 값이며 Attention 가중치가 아닙니다.")])
    return (toy_ids,)


@app.cell
def _(mo):
    mo.md("""
    ## 5. Decoder는 이미지 정보를 어디에서 읽을까?

    위에서 합친 임베딩을 RoPE·causal Attention·MLP·마지막 정규화·LM head가 처리합니다.
    **예측:** 답변을 예측하는 질문 위치(5)는 앞의 이미지 위치(1~4)를 볼 수 있을까요?
    아래 그림은 허용된 Attention 연결이며 실제 학습된 Attention 확률은 아닙니다.
    뒤의 정답 토큰은 앞 위치로 전달되지 않습니다. 답변까지 입력하는 teacher forcing이 가능한 이유입니다.

    같은 질문에서 A/B 이미지만 교체하여 logits 차이도 확인합니다.
    수치가 달라진다는 것은 이미지 조건이 출력에 전달된다는 뜻이며, 올바른 색상 인식을 뜻하지 않습니다.
    """)
    return


@app.cell
def _(base_vlm, blue_pixels, mo, plt, red_pixels, torch, toy_ids):
    with torch.no_grad():
        _a = base_vlm(toy_ids, pixel_values=red_pixels)
        _b = base_vlm(toy_ids, pixel_values=blue_pixels)
        _text = base_vlm(toy_ids)
    assert _a.logits.shape == (1, 8, 64)
    image_effect = (_a.logits[0, 5] - _b.logits[0, 5]).abs().max().item()
    assert image_effect > 0
    # 위치 0에서는 뒤에 있는 이미지 위치를 볼 수 없습니다.
    torch.testing.assert_close(_a.logits[:, 0], _b.logits[:, 0])
    _fig, _ax = plt.subplots(figsize=(4, 4))
    _ax.imshow(torch.ones(8, 8).tril(), cmap="Greys", vmin=0, vmax=1)
    _ax.set_xlabel("Key position: visible past")
    _ax.set_ylabel("Query position")
    _ax.set_xticks(range(8))
    _ax.set_yticks(range(8))
    _fig.tight_layout()
    mo.vstack([_fig, mo.md(f"logits shape: `{tuple(_a.logits.shape)}`. 질문 위치 A/B 최대 차이: `{image_effect:.6f}`\n\n텍스트만 입력했을 때와의 최대 차이: `{(_a.logits - _text.logits).abs().max().item():.6f}`")])
    return (image_effect,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 6. 답변만 학습하고 gradient 경로 확인하기

    ```text
    input_ids: [   1,   12,   12,   12,   12,   20, 21, 2 ]
    labels:    [-100, -100, -100, -100, -100, -100, 21, 2 ]
    손실 연결: logits[5] → 21, logits[6] → 2
    ```

    **예측:** 이미지 위치의 label을 `-100`으로 가려도 Projector를 학습할 수 있을까요?
    답변 손실의 gradient가 Attention을 통해 이미지 임베딩까지 전달되므로 가능합니다.
    label 마스킹은 손실 대상 선택이며 이미지 위치를 Attention에서 지우는 연산이 아닙니다.
    원본 `forward`가 `logits[:-1]`과 `labels[1:]`을 한 번 shift하므로 직접 미리 shift하지 않습니다.

    아래에서는 **Projector만 한 step 갱신**합니다. 모델을 복사하므로 앞 셀의 모델은 유지됩니다.
    언어 모델을 동결해도 그 연산을 `no_grad()`로 감싸면 안 됩니다. 입력 이미지 벡터의 gradient가 필요합니다.
    실제 trainer의 `freeze_llm=2`는 Projector만, `1`은 첫/마지막 언어 층도, `0`은 언어 모델 전체도 학습합니다.
    비전 인코더는 모든 전략에서 동결합니다. 표의 Embedding/LM head는 가중치를 공유하므로 단순 합산하면 중복됩니다.
    """)
    return


@app.cell
def _(F, base_vlm, copy, mo, red_pixels, torch, toy_ids):
    _train_model = copy.deepcopy(base_vlm).eval()
    for _name, _parameter in _train_model.named_parameters():
        _parameter.requires_grad_(_name.startswith("vision_proj."))
    answer_labels = toy_ids.clone()
    answer_labels[:, :6] = -100
    _before = {name: p.detach().clone() for name, p in _train_model.named_parameters()}
    _optimizer = torch.optim.AdamW(_train_model.vision_proj.parameters(), lr=1e-3)
    _optimizer.zero_grad(set_to_none=True)
    _out = _train_model(toy_ids, pixel_values=red_pixels, labels=answer_labels)
    _manual_loss = F.cross_entropy(_out.logits[:, :-1].reshape(-1, 64), answer_labels[:, 1:].reshape(-1), ignore_index=-100)
    torch.testing.assert_close(_out.loss, _manual_loss)
    _out.loss.backward()
    gradient_rows = [{"구성": _prefix, "gradient 절댓값 합": sum(
        p.grad.abs().sum().item() for name, p in _train_model.named_parameters()
        if name.startswith(_prefix) and p.grad is not None
    )} for _prefix in ["vision_encoder.", "vision_proj.", "model.", "lm_head."]]
    assert gradient_rows[1]["gradient 절댓값 합"] > 0
    assert all(p.grad is None for p in _train_model.vision_encoder.parameters())
    _optimizer.step()
    changed_parameters = [name for name, p in _train_model.named_parameters() if not torch.equal(p, _before[name])]
    assert changed_parameters and all(name.startswith("vision_proj.") for name in changed_parameters)
    loss_value = _out.loss.item()
    mo.vstack([
        mo.ui.table(gradient_rows, selection=None),
        mo.md(f"답변/EOS 두 토큰 평균 loss: `{loss_value:.4f}`; 변경된 매개변수 텐서: `{len(changed_parameters)}`개.\n\n한 step은 gradient 경로 확인입니다. 이미지 개념을 학습했다는 결론은 내리지 않습니다."),
    ])
    return changed_parameters, loss_value


@app.cell
def _(mo):
    mo.md("""
    ## 7. 이미지를 두 장 넣는 경우

    processor 딕셔너리의 단일 이미지 shape는 `[B,C,H,W]`, 다중 이미지는 `[B,N,C,H,W]`입니다.
    `forward`가 `B×N`을 펼쳐 인코딩한 뒤 `[B,N,P,D]`로 복원합니다.
    `count_vision_proj`는 샘플 안의 연속 marker 구간을 순서대로 이미지와 대응시킵니다.
    **예측:** 두 이미지 marker를 붙여 8개 연속으로 쓰면 두 장으로 구분할 수 있을까요?
    현재 구현은 하나의 구간으로 읽으므로 이미지 구간 사이에 텍스트 토큰이 필요합니다.
    실습에서는 이미지별 패치/marker 수를 정확히 맞춥니다. 원시 텐서 대신 processor 딕셔너리를 전달하세요.
    """)
    return


@app.cell
def _(base_vlm, blue_pixels, mo, red_pixels, torch):
    _pair_ids = torch.tensor([[1, 12, 12, 12, 12, 20, 12, 12, 12, 12, 21]])
    _pair_pixels = {"pixel_values": torch.cat([
        red_pixels["pixel_values"], blue_pixels["pixel_values"],
    ], dim=0).unsqueeze(0)}
    with torch.no_grad():
        _pair_output = base_vlm(_pair_ids, pixel_values=_pair_pixels)
    assert _pair_output.logits.shape == (1, 11, 64)
    mo.md(f"이미지 입력: `{tuple(_pair_pixels['pixel_values'].shape)}` → logits: `{tuple(_pair_output.logits.shape)}`\n\n위치 1~4는 A, 위치 6~9는 B, 위치 5는 두 구간을 분리하는 질문 ID입니다.")
    return


@app.cell
def _(mo):
    mo.md("""
    ## 8. 생성과 KV cache

    추론 입력에는 정답이 없습니다. `[BOS, IMAGE×4, 질문]`에서 다음 ID를 고르고 다시 입력합니다.
    캐시를 사용하는 첫 호출(prefill)은 이미지를 인코딩합니다. 이후에는 앞 이미지 정보가 반영된
    Key/Value를 재사용하므로 `start_pos > 0`인 forward에서 비전 인코더 실행을 생략합니다.
    **예측:** 질문까지의 prefix와 다음 한 토큰을 캐시로 계산하면 전체 재계산과 logits가 같을까요?
    아래 hook은 실제 비전 인코더 호출 횟수를 셉니다.
    """)
    return


@app.cell
def _(base_vlm, mo, red_pixels, torch, toy_ids):
    _calls = []
    _handle = base_vlm.vision_encoder.register_forward_hook(lambda *args: _calls.append(1))
    try:
        with torch.no_grad():
            _prefix = base_vlm(toy_ids[:, :6], pixel_values=red_pixels, use_cache=True)
            _cached = base_vlm(toy_ids[:, 6:7], pixel_values=red_pixels,
                               past_key_values=_prefix.past_key_values, use_cache=True)
            cache_vision_calls = len(_calls)
            _full = base_vlm(toy_ids[:, :7], pixel_values=red_pixels)
            torch.testing.assert_close(_cached.logits[:, -1], _full.logits[:, -1], atol=1e-5, rtol=1e-4)
        assert cache_vision_calls == 1
    finally:
        _handle.remove()
    with torch.no_grad():
        generated_ids = base_vlm.generate(toy_ids[:, :6], pixel_values=red_pixels,
            max_new_tokens=3, do_sample=False, use_cache=True, eos_token_id=None, pad_token_id=0)
    assert generated_ids.shape == (1, 9)
    mo.md(f"캐시 두 번 호출에서 비전 인코더 실행: `{cache_vision_calls}`회. 전체 재계산과 logits 일치.\n\n생성한 새 ID: `{generated_ids[0, 6:].tolist()}`. 무작위 모델이므로 자연어 답변으로 해석하지 않습니다.")
    return cache_vision_calls, generated_ids


@app.cell
def _(mo):
    mo.md(r"""
    ## 9. 실제 tokenizer·VLMDataset과 연결하기

    이제 저장소의 tokenizer로 한국어 대화를 처리합니다. 두 샘플의 이미지 바이트와 대화를
    작은 Parquet로 저장하고 원본 `VLMDataset`을 호출합니다.
    `create_chat_prompt`는 `<image>` 하나를 `<|image_pad|>` 4개로 확장하고 chat template를 적용합니다.
    `generate_labels`는 assistant 응답과 종료 구간만 남깁니다. 실제 label 표에서 경계를 확인하세요.
    시스템 프롬프트/빈 think 태그의 확률적 처리는 이 셀에서 seed로 고정합니다.

    **예측:** 실제 한국어 토큰 ID를 vocabulary 64인 앞의 장난감 LLM에 넣을 수 있을까요?
    이 절은 데이터 경로 관찰 전용입니다. 실제 tokenizer의 어휘 크기에 맞는 LLM과 가중치가 별도로 필요합니다.
    실습의 4개 marker와 공개 모델의 64개 marker 설정도 혼용하지 마세요.
    """)
    return


@app.cell
def _(AutoTokenizer, BytesIO, HFDataset, VLMDataset, artifact_dir, base_vlm, blue_image, json, mo, project_dir, red_image, torch):
    import random

    actual_tokenizer = AutoTokenizer.from_pretrained(str(project_dir / "model"), local_files_only=True)
    _marker_ids = actual_tokenizer("<|image_pad|>", add_special_tokens=False).input_ids
    assert _marker_ids == base_vlm.config.image_ids
    _rows = []
    for _image, _answer in [(red_image, "빨간 사각형입니다."), (blue_image, "파란 원입니다.")]:
        _buffer = BytesIO()
        _image.save(_buffer, format="PNG")
        _rows.append({"image_bytes": _buffer.getvalue(), "conversations": json.dumps([
            {"role": "user", "content": "<image>\n그림을 설명하세요."},
            {"role": "assistant", "content": _answer},
        ], ensure_ascii=False)})
    parquet_path = artifact_dir / "classroom.parquet"
    HFDataset.from_list(_rows).to_parquet(str(parquet_path))
    _dataset = VLMDataset(str(parquet_path), actual_tokenizer, preprocess=base_vlm.processor,
                          max_length=128, image_token_len=4)
    _random_state = random.getstate()
    try:
        random.seed(42)
        dataset_ids, dataset_labels, _image_data = _dataset[0]
    finally:
        random.setstate(_random_state)
    assert int((dataset_ids == _marker_ids[0]).sum()) == 4
    assert int((dataset_labels != -100).sum()) > 0
    assert torch.all(dataset_labels[dataset_ids == _marker_ids[0]] == -100)
    _active_length = int((dataset_ids != actual_tokenizer.pad_token_id).sum())
    mo.vstack([
        mo.md(f"실제 tokenizer 어휘 수: `{len(actual_tokenizer)}`; sample 이미지 shape: `{tuple(_image_data['pixel_values'].shape)}`\n\nDataLoader가 배치 축을 쌓으면 `[B,N,C,H,W]`가 됩니다. 이 두 샘플은 같은 이미지 수를 사용합니다."),
        mo.ui.table([{"위치": i, "ID": int(dataset_ids[i]), "토큰": repr(actual_tokenizer.decode([int(dataset_ids[i])])),
                      "label": int(dataset_labels[i]), "손실 대상": bool(dataset_labels[i] != -100)}
                     for i in range(min(_active_length + 2, len(dataset_ids)))], selection=None),
    ])
    return dataset_ids, dataset_labels


@app.cell
def _(artifact_dir, cache_vision_calls, changed_parameters, dataset_ids, dataset_labels, generated_ids, image_effect, json, loss_value, mo):
    metrics_path = artifact_dir / "summary.json"
    metrics_path.write_text(json.dumps({
        "experiment": "random tiny SigLIP and MiniMindVLM; not a quality evaluation",
        "image_effect_max_logit_delta": image_effect,
        "answer_loss_before_one_step": loss_value,
        "changed_parameters": changed_parameters,
        "vision_calls_with_cache": cache_vision_calls,
        "generated_new_ids": generated_ids[0, 6:].tolist(),
        "dataset_image_markers": int((dataset_ids == 12).sum()),
        "dataset_supervised_tokens": int((dataset_labels != -100).sum()),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    mo.md(f"실습 요약 저장: `{metrics_path}`")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 10. 직접 수정하고 설명하기

    1. **이미지의 영향:** 1절에서 모양을 같게 두고 색만 바꾸세요. 5절의 logits 차이를 기록하세요.
       차이가 있다는 결과만으로 색을 정확히 인식한다고 말할 수 없는 이유는 무엇인가요?
    2. **Projector:** `target_tokens=2`를 넣어도 패치 수가 4인 이유를 소스에서 찾으세요.
       힌트: `forward`가 호출하는 것은 MLP뿐입니다. 실제 토큰 축소에는 별도 pooling/재표본화가 필요합니다.
    3. **손실:** 6절에서 질문 ID의 label도 켜면 어떤 `logits` 위치가 추가로 학습되나요?
       힌트: label 위치 5를 켜면 logits 위치 4가 질문을 예측합니다. labels를 직접 shift하지 마세요.
    4. **순서:** 이미지가 답변보다 뒤에 있으면 앞의 답변 위치가 그 이미지를 읽을 수 있을까요?
       힌트: 5절의 causal 연결 그림을 보세요.
    5. **다음 실험 설계:** 실제 학습을 한다면 색과 모양을 독립적으로 조합하고, 학습에 없던 위치·크기의
       이미지를 검증용으로 두세요. 같은 질문에서 올바른 이미지/바꾼 이미지/이미지 제거의 정답률을 비교하세요.
       작은 두 샘플 암기와 시각 일반화는 구분해야 합니다.

    **실제 모델로 확장할 때:** `model/model_init.md`의 로컬 인코더,
    `trainer/trainer_utils.py`의 가중치 로딩과 동결 정책, `eval_vlm.py`의 chat template와 생성 입력을 확인하세요.
    비전 인코더 출력 차원·패치 수, config, tokenizer marker, 가중치 구조가 일치해야 합니다.
    공개 체크포인트 추론, 긴 문맥 절단, 서로 다른 이미지 수의 배치, GPU·DDP 학습과 생성 품질은 이 실습의 검증 범위에 포함하지 않습니다.

    **소스 대응:** 2절 `VLMConfig/get_vision_model` → 3절 `image2tensor/get_image_embeddings/MMVisionProjector`
    → 4·7절 `count_vision_proj` → 5·6절 `forward` → 8절 `generate` → 9절 `dataset/lm_dataset.py`.
    """)
    return


if __name__ == "__main__":
    app.run()
