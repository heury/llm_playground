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
    from PIL import Image, ImageDraw
    from datasets import Dataset as HFDataset
    from datasets import config as datasets_config
    from transformers import AutoTokenizer, SiglipImageProcessor, SiglipVisionConfig, SiglipVisionModel

    repo_root = Path(__file__).resolve().parents[2]
    project_dir = repo_root / "projects/minimind-v"
    # 다른 MiniMind 노트북과 model/dataset 패키지 이름이 같습니다.
    for _name in ("model", "dataset"):
        _loaded = sys.modules.get(_name)
        if _loaded is not None and getattr(_loaded, "__file__", None):
            if not Path(_loaded.__file__).resolve().is_relative_to(project_dir):
                raise RuntimeError("다른 프로젝트 모듈이 로드되었습니다. 이 노트북을 새 프로세스에서 실행하세요.")
    sys.path.insert(0, str(project_dir))
    from model.model_vlm import MiniMindVLM, VLMConfig
    from dataset.lm_dataset import VLMDataset

    torch.set_num_threads(2)
    run_id = datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y%m%d-%H%M%S") + "-" + uuid4().hex[:6] + "-vlm"
    artifact_dir = repo_root / "outputs" / "notebooks" / run_id
    artifact_dir.mkdir(parents=True, exist_ok=True)
    # 이 노트북 프로세스의 Parquet 로딩 캐시도 실행 산출물에 보관합니다.
    datasets_config.HF_DATASETS_CACHE = artifact_dir / "datasets_cache"
    return (
        AutoTokenizer, BytesIO, HFDataset, Image, ImageDraw,
        MiniMindVLM, SiglipImageProcessor,
        SiglipVisionConfig, SiglipVisionModel, VLMDataset, VLMConfig,
        artifact_dir, copy, json, mo, plt, project_dir, torch,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # MiniMind-V: 이미지가 언어 모델의 입력이 되는 과정

    **학습 목표:** 이미지가 언어 모델 입력이 되는 전체 과정을 따라갑니다.
    `minimind-v/model/model_vlm.py`의 실제 클래스를 사용합니다.

    - 이미지 전처리 → 패치 특징 → 차원 투영
    - 임베딩 교체 → 답변 학습 → 생성

    **사전 지식:** Python, 텐서 shape, 다음 토큰 예측.
    모델 구조·SFT 노트북을 먼저 보면 좋습니다.

    <img src="public/diagrams/vlm.png" alt="MiniMind-V: 이미지 경로와 텍스트 경로가 자리 교체로 합쳐지는 흐름" width="320">

    **핵심 아이디어:** 이미지 특징을 입력 시퀀스에 넣습니다.
    LLM은 이것을 일반 causal self-attention으로 읽습니다.

    - cross-attention 층은 따로 없습니다.
    - 답변 위치는 앞의 이미지와 질문을 볼 수 있습니다.
    - 이미지를 단어 ID로 바꾸지 않습니다.

    | 절 | 원본 소스 (`projects/minimind-v/`) |
    | --- | --- |
    | 2 | `model/model_vlm.py` — `VLMConfig`, `get_vision_model` |
    | 3 | `image2tensor`, `get_image_embeddings`, `MMVisionProjector` |
    | 4·7 | `count_vision_proj` (이미지 자리 교체) |
    | 5·6 | `MiniMindVLM.forward` (Decoder·손실) |
    | 8 | `generate` (KV cache) |
    | 9 | `dataset/lm_dataset.py` — `VLMDataset` |

    **멀티모달 파트의 첫 노트북**입니다.

    - 이 노트북: **구조**. 이미지가 토큰 자리에 들어가는 방법.
    - `minimind_v_finetuning.py`: **학습**. Projector 정렬 → 시각 지시 SFT.
    - `minimind_omni.py`: 음성 입출력.

    텍스트 LLM 부분(Attention·생성·SFT 마스크)은 앞 노트북과 같습니다.

    **실행 환경:** 작은 SigLIP과 LLM을 무작위로 초기화합니다.
    GPU, 모델 다운로드, API 키가 필요 없습니다.
    그래서 생성 결과는 인식 성능과 무관합니다.
    실습 파일은 `outputs/notebooks/<실행 ID>/`에 저장합니다. 기존 체크포인트는 건드리지 않습니다.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. 이미지와 패치: 몇 개의 시각 토큰이 생길까?

    **예측:** 32×32 이미지를 16×16 패치로 나누면 패치는 몇 개일까요?

    - 원본: 256 / 32 = 8 → 8×8 = **64개**
    - 실습: 32 / 16 = 2 → 2×2 = **4개**

    **기호:** `B` 배치, `N` 이미지 수, `P` 패치 수, `Dv` 비전 차원, `D` 언어 차원, `T` 시퀀스 길이.

    아래 점선이 패치 경계입니다.
    단, SigLIP 내부의 self-attention이 패치끼리 정보를 섞습니다.
    그래서 최종 특징 하나가 자기 사각형의 픽셀만 담지는 않습니다.
    """)
    return


@app.cell
def _(Image, ImageDraw, plt):
    red_image = Image.new("RGB", (32, 32), "white")
    ImageDraw.Draw(red_image).rectangle((4, 4, 27, 27), fill="red")
    blue_image = Image.new("RGB", (32, 32), "white")
    ImageDraw.Draw(blue_image).ellipse((4, 4, 27, 27), fill="blue")
    sample_image_fig, _axes = plt.subplots(1, 2, figsize=(6, 3))
    for _ax, _image, _title in zip(_axes, [red_image, blue_image], ["A: red square", "B: blue circle"]):
        _ax.imshow(_image)
        _ax.axhline(15.5, color="black", linestyle="--")
        _ax.axvline(15.5, color="black", linestyle="--")
        _ax.set_title(_title)
        _ax.set_xticks([0, 16, 31])
        _ax.set_yticks([0, 16, 31])
    sample_image_fig.tight_layout()
    return (blue_image, red_image, sample_image_fig,)


@app.cell(hide_code=True)
def _(mo, sample_image_fig):
    # ▸ 결과 표시 — 실행하면 표·그림이 나타납니다 (marimo 표시 코드)
    mo.vstack([sample_image_fig, mo.md("**관찰:** 이미지마다 패치가 4개입니다. A와 B는 색과 모양이 모두 다릅니다. 그래서 두 속성의 효과를 따로 볼 수는 없습니다.")])
    return


@app.cell
def _(mo):
    mo.md("""
    ## 2. 실제 클래스의 축소 모델을 만들기

    작은 `SiglipVisionModel`을 로컬에 저장합니다.
    그다음 `MiniMindVLM`의 실제 `get_vision_model`로 읽습니다.
    학습된 가중치는 내려받지 않습니다.

    **세 숫자가 같아야 합니다:** `image_token_len`, 인코더 패치 수, 연속 이미지 marker 수. 여기서는 모두 4입니다.

    경로가 없으면 원본 로더는 `(None, None)`을 반환합니다. 그래서 로딩 성공을 assert로 확인합니다.

    | 항목 | 이 실습 | 원본 기본값 |
    | --- | --- | --- |
    | 비전 인코더 | 무작위 SigLIP 1층 | `siglip2-base-p32-256-ve` (float16 약 180MB, 동결) |
    | 이미지 / 패치 크기 | 32 / 16 | 256 / 32 |
    | 이미지당 시각 토큰 `P` | 4 | 64 |
    | 비전 차원 `Dv` → 언어 차원 `D` | 16 → 32 | 768 → 768 |
    | Projector 매개변수 | 수천 개 | 약 1.18M |
    | 언어 모델 | 2층, 어휘 64 | 8층, 어휘 6400, 약 63.9M |
    """)
    return


@app.cell
def _(MiniMindVLM, SiglipImageProcessor, SiglipVisionConfig, SiglipVisionModel, VLMConfig, artifact_dir, torch):
    torch.manual_seed(42)
    _vision_dir = artifact_dir / "tiny_siglip"
    _vision = SiglipVisionModel(SiglipVisionConfig(
        image_size=32, patch_size=16, hidden_size=16,
        intermediate_size=32, num_hidden_layers=1, num_attention_heads=2,
    ))
    _vision.save_pretrained(_vision_dir)
    SiglipImageProcessor(size={"height": 32, "width": 32}).save_pretrained(_vision_dir)
    _config = VLMConfig(
        hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, vocab_size=64,
        max_position_embeddings=256, image_hidden_size=16, image_token_len=4,
        image_ids=[12], dropout=0.0, flash_attn=False,
    )
    base_vlm = MiniMindVLM(_config, vision_model_path=str(_vision_dir)).eval()
    assert base_vlm.vision_encoder is not None and base_vlm.processor is not None
    assert all(not _p.requires_grad for _p in base_vlm.vision_encoder.parameters())
    assert base_vlm.lm_head.weight is base_vlm.model.embed_tokens.weight
    return (base_vlm,)


@app.cell(hide_code=True)
def _(base_vlm, mo):
    # ▸ 결과 표시 — 실행하면 표·그림이 나타납니다 (marimo 표시 코드)
    mo.vstack([
        mo.ui.table([
            {"구성": _name, "매개변수 수": sum(_p.numel() for _p in _module.parameters()),
             "학습 가능 수": sum(_p.numel() for _p in _module.parameters() if _p.requires_grad)}
            for _name, _module in [("SigLIP", base_vlm.vision_encoder), ("Projector", base_vlm.vision_proj),
                                    ("언어 backbone (Embedding 포함)", base_vlm.model),
                                    ("LM head (= Embedding과 같은 텐서)", base_vlm.lm_head)]
        ], selection=None),
        mo.md("LM head와 Embedding은 같은 가중치입니다(`tie_word_embeddings=True`). 전체 수를 셀 때 LM head는 빼세요."),
    ])
    return


@app.cell
def _(mo):
    mo.md("""
    ## 3. 픽셀 → 시각 특징 → 언어 차원

    **예측:** Projector를 거치면 패치 수와 마지막 차원 중 무엇이 바뀔까요?

    | 함수 | 하는 일 |
    | --- | --- |
    | `image2tensor` | resize·정규화 후 processor 딕셔너리 반환 |
    | `get_image_embeddings` | `no_grad()`로 SigLIP의 `last_hidden_state`를 읽음 |
    | `vision_proj` | 패치마다 같은 MLP 적용: `LayerNorm → Linear(16,32) → GELU → Linear(32,32)` |

    SigLIP에는 CLS 토큰이 없습니다. 그래서 **모든 패치 특징 `[B,P,Dv]`**를 그대로 씁니다.

    **Projector의 진짜 역할:** 차원만 바꾸는 게 아닙니다.
    SigLIP과 LLM은 따로 학습되어 특징 공간이 다릅니다.
    Projector는 시각 특징을 **LLM이 읽을 수 있는 임베딩으로 맞춥니다.**
    그래서 실제 학습 첫 단계에서는 Projector만 학습합니다(`minimind_v_finetuning.py`).
    """)
    return


@app.cell
def _(base_vlm, blue_image, red_image):
    red_pixels = base_vlm.image2tensor(red_image, base_vlm.processor)
    blue_pixels = base_vlm.image2tensor(blue_image, base_vlm.processor)
    vision_features = base_vlm.get_image_embeddings(red_pixels, base_vlm.vision_encoder)
    projected_features = base_vlm.vision_proj(vision_features)
    assert tuple(vision_features.shape) == (1, 4, 16)
    assert tuple(projected_features.shape) == (1, 4, 32)
    assert not vision_features.requires_grad and projected_features.requires_grad
    return (blue_pixels, projected_features, red_pixels, vision_features,)


@app.cell(hide_code=True)
def _(mo, projected_features, red_pixels, vision_features):
    # ▸ 결과 표시 — 실행하면 표·그림이 나타납니다 (marimo 표시 코드)
    mo.ui.table([
        {"단계": "processor", "shape": str(tuple(red_pixels["pixel_values"].shape)), "gradient": False},
        {"단계": "SigLIP", "shape": str(tuple(vision_features.shape)), "gradient": vision_features.requires_grad},
        {"단계": "Projector", "shape": str(tuple(projected_features.shape)), "gradient": projected_features.requires_grad},
    ], selection=None)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 4. IMAGE 자리를 이미지 벡터로 교체하기

    4~8절은 **가상 ID**를 씁니다. 실제 토크나이저 결과가 아닙니다.
    `1=BOS`, `12=IMAGE`, `20=질문`, `21=답변`, `2=EOS`.
    실제 tokenizer는 9절에서 연결합니다.

    ```text
    input_ids: [ BOS, IMAGE, IMAGE, IMAGE, IMAGE, 질문, 답변, EOS ]
    embedding: [  e0,    e1,    e2,    e3,    e4,   e5,   e6,  e7 ]
    교체 후:   [  e0,    v0,    v1,    v2,    v3,   e5,   e6,  e7 ]
    ```

    **예측:** 교체 후 시퀀스 길이는 8일까요, 12일까요?

    `count_vision_proj`는 연속된 IMAGE 구간을 찾습니다.
    그 자리의 벡터를 이미지 벡터로 **교체**합니다. 끼워 넣지 않습니다.
    marker 수와 패치 수가 다르면 어떻게 될까요? 7절에서 봅니다.
    """)
    return


@app.cell
def _(base_vlm, plt, projected_features, torch):
    toy_ids = torch.tensor([[1, 12, 12, 12, 12, 20, 21, 2]])
    toy_tokens = ["BOS", "image 0", "image 1", "image 2", "image 3", "question", "answer", "EOS"]
    text_embeddings = base_vlm.model.embed_tokens(toy_ids)
    merged_embeddings = base_vlm.count_vision_proj(
        toy_ids, text_embeddings, projected_features, seqlen=toy_ids.shape[1],
    )
    assert int((toy_ids == 12).sum()) == projected_features.shape[1]
    assert merged_embeddings.shape == text_embeddings.shape
    torch.testing.assert_close(merged_embeddings[:, 1:5], projected_features)
    torch.testing.assert_close(merged_embeddings[:, [0, 5, 6, 7]], text_embeddings[:, [0, 5, 6, 7]])
    merged_fig, _ax = plt.subplots(figsize=(8, 3))
    _plot = _ax.imshow(merged_embeddings[0].detach().numpy(), aspect="auto", cmap="coolwarm")
    _ax.set_yticks(range(8), toy_tokens)
    _ax.set_xlabel("Embedding dimension (32)")
    merged_fig.colorbar(_plot, ax=_ax)
    merged_fig.tight_layout()
    return (merged_fig, toy_ids, toy_tokens,)


@app.cell(hide_code=True)
def _(merged_fig, mo):
    # ▸ 결과 표시 — 실행하면 표·그림이 나타납니다 (marimo 표시 코드)
    mo.vstack([merged_fig, mo.md("**확인:** 길이는 8 그대로입니다. 이미지 위치 4개만 바뀌었습니다. 색은 임베딩 값입니다(Attention 가중치 아님).")])
    return


@app.cell
def _(mo):
    mo.md("""
    ## 5. Decoder는 이미지 정보를 어디에서 읽을까?

    합친 임베딩은 일반 Decoder를 그대로 지납니다(RoPE → Attention → MLP → LM head).
    `forward`는 이미지 위치를 특별 취급하지 않습니다.

    - 이미지 벡터도 **일반 1D RoPE**를 받습니다(위치 1~4).
    - LLM에게 패치는 한 줄로 늘어선 토큰일 뿐입니다.
    - 2D 위치 정보는 SigLIP이 패치 특징에 넣어 둔 만큼만 전달됩니다.

    **예측:** 질문 위치(5)는 앞의 이미지 위치(1~4)를 볼 수 있을까요?

    아래 그림은 **허용된** Attention 연결입니다(주황 = 이미지 Key). 학습된 Attention 확률이 아닙니다.
    뒤 토큰은 앞 위치에서 보이지 않습니다. 그래서 답변까지 한 번에 넣는 teacher forcing이 가능합니다.

    A/B 이미지만 바꿔서 logits 차이도 확인합니다.
    값이 달라지면 이미지가 출력에 영향을 준다는 뜻입니다. 색을 맞게 인식한다는 뜻은 아닙니다.
    """)
    return


@app.cell
def _(base_vlm, blue_pixels, plt, red_pixels, torch, toy_ids, toy_tokens):
    from matplotlib.colors import ListedColormap

    with torch.no_grad():
        red_output = base_vlm(toy_ids, pixel_values=red_pixels)
        _blue_output = base_vlm(toy_ids, pixel_values=blue_pixels)
    assert red_output.logits.shape == (1, 8, 64)
    image_effect = (red_output.logits[0, 5] - _blue_output.logits[0, 5]).abs().max().item()
    assert image_effect > 0
    # 위치 0에서는 뒤에 있는 이미지 위치를 볼 수 없습니다.
    torch.testing.assert_close(red_output.logits[:, 0], _blue_output.logits[:, 0])

    _visible = torch.ones(8, 8).tril()
    _visible[:, 1:5] *= 2  # 이미지 Key 열 강조
    causal_fig, _ax = plt.subplots(figsize=(4.5, 4.5))
    _ax.imshow(_visible, cmap=ListedColormap(["white", "dimgray", "tab:orange"]), vmin=0, vmax=2)
    _ax.set_xlabel("Key position (visible past)")
    _ax.set_ylabel("Query position")
    _ax.set_xticks(range(8), toy_tokens, rotation=60, ha="right")
    _ax.set_yticks(range(8), toy_tokens)
    causal_fig.tight_layout()
    return (causal_fig, image_effect, red_output,)


@app.cell(hide_code=True)
def _(causal_fig, image_effect, mo, red_output):
    # ▸ 결과 표시 — 실행하면 표·그림이 나타납니다 (marimo 표시 코드)
    mo.vstack([causal_fig, mo.md(
        f"logits shape: `{tuple(red_output.logits.shape)}`. 질문 위치(5)의 A/B 최대 차이: `{image_effect:.6f}`\n\n"
        "BOS 위치(0)의 logits는 A/B가 똑같습니다. 이미지가 뒤에 있어서 볼 수 없습니다."
    )])
    return


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

    **가능합니다.** 답변 손실의 gradient가 Attention을 거쳐 이미지 임베딩까지 갑니다.
    label 마스킹은 "어디서 손실을 계산할지"만 정합니다. Attention에서 이미지를 지우지 않습니다.
    shift는 원본 `forward`가 합니다. labels를 미리 shift하지 마세요.

    아래 셀은 **Projector만 한 step 갱신**합니다. 복사본을 쓰므로 원래 모델은 그대로입니다.
    결과는 표 두 개로 봅니다.

    - **매개변수의 `.grad`:** 동결된 LLM·SigLIP은 `None`입니다.
    - **이미지 위치 벡터의 gradient:** Projector 출력에 도착한 값입니다.
      0이 아니면 gradient가 **동결된 LLM을 통과했다**는 뜻입니다.

    **정리:** "LLM 동결"은 가중치를 안 바꾼다는 뜻입니다. gradient를 막는다는 뜻이 아닙니다.
    LLM을 `no_grad()`로 감싸면 Projector를 학습할 수 없습니다.
    `freeze_llm` 0/1/2 설정은 `minimind_v_finetuning.py` 6절에서 다룹니다.
    """)
    return


@app.cell
def _(base_vlm, copy, red_pixels, torch, toy_ids, toy_tokens):
    _train_model = copy.deepcopy(base_vlm).eval()
    for _name, _parameter in _train_model.named_parameters():
        _parameter.requires_grad_(_name.startswith("vision_proj."))
    answer_labels = toy_ids.clone()
    answer_labels[:, :6] = -100

    # Projector 출력(= 이미지 자리에 들어갈 벡터)의 gradient를 보관합니다.
    _image_vectors = []
    def _keep_grad(_module, _inputs, _output):
        _output.retain_grad()
        _image_vectors.append(_output)
    _handle = _train_model.vision_proj.register_forward_hook(_keep_grad)
    try:
        _out = _train_model(toy_ids, pixel_values=red_pixels, labels=answer_labels)
    finally:
        _handle.remove()

    _before = {name: p.detach().clone() for name, p in _train_model.named_parameters()}
    _optimizer = torch.optim.AdamW(_train_model.vision_proj.parameters(), lr=1e-3)
    _optimizer.zero_grad(set_to_none=True)
    _out.loss.backward()

    gradient_rows = [{"구성": _prefix, "저장된 .grad 절댓값 합": sum(
        p.grad.abs().sum().item() for name, p in _train_model.named_parameters()
        if name.startswith(_prefix) and p.grad is not None
    )} for _prefix in ["vision_encoder.", "vision_proj.", "model.", "lm_head."]]
    image_grad_norms = _image_vectors[0].grad[0].norm(dim=-1).tolist()
    image_grad_rows = [{"입력 위치": toy_tokens[1 + k], "gradient L2 norm": g} for k, g in enumerate(image_grad_norms)]
    assert gradient_rows[1]["저장된 .grad 절댓값 합"] > 0
    assert all(p.grad is None for p in _train_model.vision_encoder.parameters())
    assert all(g > 0 for g in image_grad_norms)

    _optimizer.step()
    changed_parameters = [name for name, p in _train_model.named_parameters() if not torch.equal(p, _before[name])]
    assert changed_parameters and all(name.startswith("vision_proj.") for name in changed_parameters)
    loss_value = _out.loss.item()
    return (changed_parameters, gradient_rows, image_grad_norms, image_grad_rows, loss_value,)


@app.cell(hide_code=True)
def _(changed_parameters, gradient_rows, image_grad_rows, loss_value, mo):
    # ▸ 결과 표시 — 실행하면 표·그림이 나타납니다 (marimo 표시 코드)
    mo.vstack([
        mo.md("**매개변수에 저장된 gradient**"),
        mo.ui.table(gradient_rows, selection=None),
        mo.md("**이미지 위치 벡터에 도착한 gradient** (동결된 LLM을 통과해 온 값)"),
        mo.ui.table(image_grad_rows, selection=None),
        mo.md(f"답변·EOS 평균 loss: `{loss_value:.4f}`. 바뀐 매개변수 텐서: `{len(changed_parameters)}`개(모두 `vision_proj.`).\n\n이 한 step은 gradient 경로 확인용입니다. 이미지를 학습했다는 뜻은 아닙니다."),
    ])
    return


@app.cell
def _(mo):
    mo.md("""
    ## 7. 이미지를 두 장 넣는 경우

    - 이미지 한 장: `[B,C,H,W]`
    - 이미지 여러 장: `[B,N,C,H,W]`

    `forward`는 `B×N`장을 한 번에 인코딩합니다. 그 뒤 `[B,N,P,D]`로 되돌립니다.
    `count_vision_proj`는 **연속 marker 구간**을 순서대로 이미지와 짝짓습니다.
    입력은 원시 텐서가 아니라 processor 딕셔너리로 넘기세요.

    **예측:** marker 8개를 붙여 쓰면 두 장으로 구분될까요?
    아래 두 번째 실험이 답입니다. 원본 함수는 marker 수가 맞는지 검사하지 않습니다.
    """)
    return


@app.cell
def _(base_vlm, blue_pixels, red_pixels, torch):
    _pair_ids = torch.tensor([[1, 12, 12, 12, 12, 20, 12, 12, 12, 12, 21]])
    pair_pixels = {"pixel_values": torch.cat([
        red_pixels["pixel_values"], blue_pixels["pixel_values"],
    ], dim=0).unsqueeze(0)}
    with torch.no_grad():
        pair_output = base_vlm(_pair_ids, pixel_values=pair_pixels)
        # 잘못된 예: marker 8개를 붙여 쓰면 한 구간(이미지 A)으로 읽힙니다.
        _pair_vectors = base_vlm.vision_proj(base_vlm.get_image_embeddings(
            {"pixel_values": pair_pixels["pixel_values"].flatten(0, 1)}, base_vlm.vision_encoder,
        )).view(1, 2, 4, -1)
        _bad_ids = torch.tensor([[1] + [12] * 8 + [21]])
        _bad_merged = base_vlm.count_vision_proj(
            _bad_ids, base_vlm.model.embed_tokens(_bad_ids), _pair_vectors, seqlen=_bad_ids.shape[1],
        )
    assert pair_output.logits.shape == (1, 11, 64)
    bad_merge_lengths = (_bad_ids.shape[1], _bad_merged.shape[1])
    assert bad_merge_lengths == (10, 6)
    return (bad_merge_lengths, pair_output, pair_pixels,)


@app.cell(hide_code=True)
def _(bad_merge_lengths, mo, pair_output, pair_pixels):
    # ▸ 결과 설명 — 실행하면 계산 값이 들어간 설명이 나타납니다 (marimo 표시 코드)
    mo.md(f"""
    **올바른 예:** 이미지 입력 `{tuple(pair_pixels['pixel_values'].shape)}` → logits `{tuple(pair_output.logits.shape)}`.
    위치 1~4는 A, 6~9는 B입니다. 위치 5의 질문 ID가 두 구간을 나눕니다.

    **잘못된 예 (marker 8개 연속):** 입력 길이 `{bad_merge_lengths[0]}` → 교체 후 `{bad_merge_lengths[1]}`.

    - 8칸 전체가 이미지 A의 벡터 4개로 바뀝니다.
    - **시퀀스가 4칸 줄고, 이미지 B는 버려집니다.**
    - 오류가 나지 않습니다. labels·RoPE 위치가 어긋난 채 학습될 수 있습니다.
    - marker가 패치 수보다 적으면 남는 패치를 조용히 버립니다.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## 8. 생성과 KV cache

    추론 입력에는 정답이 없습니다.
    `[BOS, IMAGE×4, 질문]`에서 다음 ID를 고르고, 그 ID를 다시 입력합니다.

    - 첫 호출(prefill): 이미지를 인코딩합니다.
    - 이후 호출(`start_pos > 0`): 캐시된 Key/Value를 씁니다. 비전 인코더는 건너뜁니다.

    **예측:** 캐시로 계산한 logits가 전체 재계산과 같을까요?
    아래 hook은 비전 인코더 호출 횟수를 셉니다.
    """)
    return


@app.cell
def _(base_vlm, red_pixels, torch, toy_ids):
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
    return (cache_vision_calls, generated_ids,)


@app.cell(hide_code=True)
def _(cache_vision_calls, generated_ids, mo):
    # ▸ 결과 설명 — 실행하면 계산 값이 들어간 설명이 나타납니다 (marimo 표시 코드)
    mo.md(f"캐시 호출 두 번에서 비전 인코더 실행: `{cache_vision_calls}`회. logits는 전체 재계산과 같습니다.\n\n생성한 새 ID: `{generated_ids[0, 6:].tolist()}`. 무작위 모델이라 의미 있는 답변이 아닙니다.")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 9. 실제 tokenizer·VLMDataset과 연결하기

    실제 tokenizer로 한국어 대화 한 건을 `VLMDataset`에 넣어 봅니다.
    확인할 것은 두 가지입니다.

    1. `create_chat_prompt`: `<image>` 하나를 `<|image_pad|>` × `image_token_len`개로 펼친다.
    2. `generate_labels`: assistant 응답과 종료 토큰만 손실 대상으로 남긴다.

    시스템 프롬프트와 빈 think 태그는 무작위로 붙습니다. 그래서 seed를 고정합니다.
    토큰별 label 표는 `minimind_v_finetuning.py` 3절에 있습니다.

    **주의:** 이 절은 데이터 관찰용입니다.
    실제 어휘(6400)는 장난감 LLM(어휘 64)에 넣을 수 없습니다.
    marker 수도 실습(4개)과 공개 모델(64개)이 다릅니다. 섞지 마세요.
    """)
    return


@app.cell
def _(AutoTokenizer, BytesIO, HFDataset, VLMDataset, artifact_dir, base_vlm, json, project_dir, red_image):
    import random

    actual_tokenizer = AutoTokenizer.from_pretrained(str(project_dir / "model"), local_files_only=True)
    _marker_id = base_vlm.config.image_ids[0]
    assert actual_tokenizer("<|image_pad|>", add_special_tokens=False).input_ids == [_marker_id]
    _buffer = BytesIO()
    red_image.save(_buffer, format="PNG")
    _parquet_path = artifact_dir / "classroom.parquet"
    HFDataset.from_list([{"image_bytes": _buffer.getvalue(), "conversations": json.dumps([
        {"role": "user", "content": "<image>\n그림을 설명하세요."},
        {"role": "assistant", "content": "빨간 사각형입니다."},
    ], ensure_ascii=False)}]).to_parquet(str(_parquet_path))
    _dataset = VLMDataset(str(_parquet_path), actual_tokenizer, preprocess=base_vlm.processor,
                          max_length=128, image_token_len=4)
    _random_state = random.getstate()
    try:
        random.seed(42)
        _ids, _labels, dataset_image = _dataset[0]
    finally:
        random.setstate(_random_state)
    assert int((_ids == _marker_id).sum()) == 4
    assert bool((_labels[_ids == _marker_id] == -100).all())

    # 같은 성격의 연속 토큰을 하나의 구간으로 묶어 decode합니다(바이트 단위 토큰이 깨져 보이지 않도록).
    _active = int((_ids != actual_tokenizer.pad_token_id).sum())
    dataset_spans = []
    for _pos in range(_active):
        _id = int(_ids[_pos])
        _kind = "이미지 자리" if _id == _marker_id else ("손실 대상" if int(_labels[_pos]) != -100 else "문맥 (손실 제외)")
        if dataset_spans and dataset_spans[-1]["구분"] == _kind:
            dataset_spans[-1]["_ids"].append(_id)
            dataset_spans[-1]["끝"] = _pos
        else:
            dataset_spans.append({"구분": _kind, "시작": _pos, "끝": _pos, "_ids": [_id]})
    for _span in dataset_spans:
        _span["토큰 수"] = len(_span["_ids"])
        _span["텍스트"] = repr(actual_tokenizer.decode(_span.pop("_ids")))
    dataset_supervised_tokens = int((_labels != -100).sum())
    assert dataset_supervised_tokens > 0
    return (actual_tokenizer, dataset_image, dataset_spans, dataset_supervised_tokens,)


@app.cell(hide_code=True)
def _(actual_tokenizer, dataset_image, dataset_spans, mo):
    # ▸ 결과 표시 — 실행하면 표·그림이 나타납니다 (marimo 표시 코드)
    mo.vstack([
        mo.md(f"tokenizer 어휘 수: `{len(actual_tokenizer)}`. 샘플 이미지 shape: `{tuple(dataset_image['pixel_values'].shape)}` = [N,C,H,W]. "
              "DataLoader를 거치면 `[B,N,C,H,W]`가 됩니다. 뒤쪽 PAD는 생략했습니다."),
        mo.ui.table(dataset_spans, selection=None),
        mo.md("**관찰:** 이미지 자리와 질문은 문맥일 뿐, 손실 대상이 아닙니다. "
              "이 seed에서는 빈 think 태그가 assistant 구간에 들어갑니다."),
    ])
    return


@app.cell
def _(artifact_dir, bad_merge_lengths, cache_vision_calls, changed_parameters, dataset_supervised_tokens, generated_ids, image_effect, image_grad_norms, json, loss_value):
    metrics_path = artifact_dir / "summary.json"
    metrics_path.write_text(json.dumps({
        "experiment": "random tiny SigLIP and MiniMindVLM; not a quality evaluation",
        "image_effect_max_logit_delta": image_effect,
        "answer_loss_before_one_step": loss_value,
        "image_position_grad_norms": image_grad_norms,
        "changed_parameters": changed_parameters,
        "merged_length_with_8_consecutive_markers": list(bad_merge_lengths),
        "vision_calls_with_cache": cache_vision_calls,
        "generated_new_ids": generated_ids[0, 6:].tolist(),
        "dataset_supervised_tokens": dataset_supervised_tokens,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return (metrics_path,)


@app.cell(hide_code=True)
def _(metrics_path, mo):
    # ▸ 결과 설명 — 실행하면 계산 값이 들어간 설명이 나타납니다 (marimo 표시 코드)
    mo.md(f"실습 요약 저장: `{metrics_path}`")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 10. 직접 수정하고 설명하기

    1. **이미지의 영향:** 1절에서 모양은 같게, 색만 바꾸세요. 5절의 logits 차이를 기록하세요.
       차이가 있어도 "색을 인식한다"고 말할 수 없는 이유는 무엇일까요?
    2. **Projector:** `MMVisionProjector(16, 32, target_tokens=2)`로 만들어도 패치 수는 4입니다. 왜일까요?
       힌트: `forward`는 MLP만 호출합니다. 토큰을 줄이려면 pooling 같은 단계가 따로 필요합니다.
    3. **손실:** 6절에서 질문 ID의 label도 켜면 어느 `logits` 위치가 더 학습될까요?
       힌트: label 위치 5를 켜면 logits 위치 4가 질문을 예측합니다. labels를 직접 shift하지 마세요.
    4. **순서:** 이미지가 답변보다 뒤에 있으면, 답변 위치가 그 이미지를 볼 수 있을까요?
       힌트: 5절의 causal 그림을 보세요.
    5. **실험 설계:** 실제로 학습한다면 어떻게 검증할까요?
       - 색과 모양을 따로따로 조합하세요.
       - 학습에 없던 위치·크기의 이미지를 검증용으로 두세요.
       - 같은 질문으로 비교하세요: 올바른 이미지 / 바꾼 이미지 / 이미지 없음.
       - 샘플 암기와 시각 일반화를 구분하세요.

    **실제 모델로 확장할 때 볼 파일:**

    - `model/model_init.md`: 로컬 인코더
    - `trainer/trainer_utils.py`: 가중치 로딩, 동결 정책
    - `eval_vlm.py`: chat template, 생성 입력

    인코더 출력 차원, 패치 수, config, tokenizer marker, 가중치 구조가 모두 맞아야 합니다.

    **이 실습에서 검증하지 않은 것:** 공개 체크포인트 추론, 긴 문맥 절단,
    이미지 수가 다른 배치, GPU·DDP 학습, 생성 품질.

    **다음 실습:** `minimind_v_finetuning.py`.
    두 단계 학습을 하고, 모델이 이미지를 실제로 근거로 쓰는지 검사합니다.
    """)
    return


if __name__ == "__main__":
    app.run()
