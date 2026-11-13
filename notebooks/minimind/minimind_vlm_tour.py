import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import copy
    import glob
    import io
    import json
    import math
    import random
    import sys
    import tempfile
    from contextlib import nullcontext
    from pathlib import Path

    import datasets
    import marimo as mo
    import matplotlib.pyplot as plt
    import pyarrow as pa
    import pyarrow.parquet as pq
    import torch
    import torch.nn.functional as F
    from huggingface_hub import hf_hub_download, snapshot_download
    from PIL import Image, ImageDraw
    from safetensors.torch import load_file
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer

    # 노트북 위치(notebooks/minimind/)에서 두 단계 위가 저장소 루트입니다.
    repo_root = Path(__file__).resolve().parents[2]
    minimind_v_dir = repo_root / "projects/minimind-v"
    sys.path.insert(0, str(minimind_v_dir))

    # train_pretrain_vlm.py / train_sft_vlm.py가 쓰는 원본 클래스와 함수를 그대로 가져옵니다.
    from dataset.lm_dataset import VLMDataset
    from model.model_vlm import MiniMindVLM, VLMConfig
    from trainer.trainer_utils import SkipBatchSampler, get_lr, setup_seed, vlm_collate_fn

    device = "cuda" if torch.cuda.is_available() else "cpu"

    def autocast_ctx():
        """원본 학습 스크립트의 '3. 혼합 정밀도 설정': GPU에서 bf16 autocast"""
        return nullcontext() if device == "cpu" else torch.autocast(device_type="cuda", dtype=torch.bfloat16)

    datasets.disable_progress_bars()
    # MiniMind-V의 토크나이저는 MiniMind 언어 모델과 같은 파일입니다(어휘 6,400개, <|image_pad|> 포함).
    tokenizer = AutoTokenizer.from_pretrained(minimind_v_dir / "model")
    return (
        DataLoader,
        F,
        Image,
        ImageDraw,
        MiniMindVLM,
        SkipBatchSampler,
        VLMConfig,
        VLMDataset,
        autocast_ctx,
        copy,
        device,
        get_lr,
        glob,
        hf_hub_download,
        io,
        json,
        load_file,
        math,
        minimind_v_dir,
        mo,
        pa,
        plt,
        pq,
        random,
        setup_seed,
        snapshot_download,
        tempfile,
        tokenizer,
        torch,
        vlm_collate_fn,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # MiniMind-V: 언어 모델에 눈 달기 (Vision LLM 따라가기)

    지금까지 다룬 MiniMind는 **글만** 읽습니다. **VLM(Vision-Language Model)** 은 이미지를 보고 질문에 답합니다.
    놀랍게도 언어 모델을 처음부터 다시 만들 필요는 없습니다. MiniMind-V는 이미 대화할 줄 아는 언어 모델에
    **이미 학습된 이미지 인코더**를 붙이고, 둘 사이를 잇는 **작은 연결층(Projector)** 만 새로 학습합니다.

    이 노트북은 대화 SFT까지 마친 **64M MiniMind 언어 모델**
    [`caspar/MiniMind-Dense-64M-Mixed-SFT-v1.0`](https://huggingface.co/caspar/MiniMind-Dense-64M-Mixed-SFT-v1.0)을 베이스로,
    `projects/minimind-v/model/model_vlm.py`의 `MiniMindVLM`을 **GPU에서 직접 학습**합니다.
    "색깔 도형" 이미지로 이루어진 작은 세계를 만들어 모델에게 색과 모양을 묻고, 학습 전후에 무엇이 달라지는지 봅니다.

    **준비:** **NVIDIA GPU가 필요합니다** (RTX 3080 Ti 12GB에서 확인, 최대 약 4.5GB 사용). 처음 실행할 때 Hugging Face에서
    언어 모델(약 128MB)과 이미지 인코더(약 189MB)를 내려받습니다. 학습 실험 4개는 GPU에서 각각 20초 안팎이고, 전체 실행은 약 2~3분입니다.

    **사전 지식:** SFT 노트북(`minimind_sft_tour.py`)의 채팅 템플릿과 "assistant 답변만 정답" 규칙.

    ## 0. 전체 구조

    ```text
     이미지 (256×256)
        │  SiglipImageProcessor: 크기 맞춤, [-1, 1]로 정규화
        ▼
     ┌──────────────────────────┐
     │ 이미지 인코더 SigLIP2       │  32×32 조각(patch) 8×8 = 64개 → 각각 768차원 벡터
     │ (95M, 고정: 학습 안 함)     │
     └──────────────────────────┘
        │  [64, 768]
        ▼
     ┌──────────────────────────┐
     │ Projector (MLP, 1.2M)      │  이미지 벡터를 언어 모델의 "단어 임베딩 자리"에 맞게 변환
     └──────────────────────────┘
        │  [64, 768]   ← 이미지 토큰 64개
        ▼
     "<|im_start|>user\n <|image_pad|>×64 \nWhat color is the shape?<|im_end|> ..."
        │   <|image_pad|> 64칸의 임베딩을 이미지 토큰으로 갈아 끼움 (count_vision_proj)
        ▼
     ┌──────────────────────────┐
     │ MiniMind 언어 모델 (64M)    │  평소처럼 다음 토큰 예측 → "The shape is red."
     └──────────────────────────┘
    ```

    | | 언어 모델 (`MiniMindForCausalLM`) | VLM (`MiniMindVLM`) |
    | --- | --- | --- |
    | 입력 | 토큰 | 토큰 + **이미지** (`pixel_values`) |
    | 새 부품 | — | `vision_encoder` (SigLIP2), **`vision_proj`** (Projector) |
    | 학습 목표 | 다음 토큰 예측 | **같음** (이미지 자리는 정답에서 제외) |
    | 학습 단계 | 사전 학습 → SFT | ① Projector 정렬(`train_pretrain_vlm.py`) → ② 시각 SFT(`train_sft_vlm.py`) |

    `MiniMindVLM`은 `MiniMindForCausalLM`을 **상속**합니다. 즉 VLM은 "이미지를 받는 입구가 하나 더 달린 언어 모델"입니다.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. 부품 세 개 불러오기

    원본 `init_vlm_model`은 `MiniMindVLM(vlm_config, vision_model_path=...)`를 만든 뒤 언어 모델 가중치(`../out/llm_768.pth`)를
    `strict=False`로 넣습니다. 여기서는 그 언어 모델 가중치를 Hugging Face의 베이스 모델에서 가져옵니다(버전 고정).

    - **언어 모델:** `caspar/MiniMind-Dense-64M-Mixed-SFT-v1.0`의 `config.json`과 `model.safetensors` (저장소의 원격 코드는 실행하지 않음)
    - **이미지 인코더:** 원본 README가 안내하는 `siglip2-base-p32-256-ve` ([`jingyaogong/siglip2-base-p32-256-ve`](https://huggingface.co/jingyaogong/siglip2-base-p32-256-ve)).
      Google SigLIP2의 이미지 부분만 떼어 낸 것입니다.
    - **Projector:** 새로 만드는 부품이라 **무작위 값**으로 시작합니다.
    """)
    return


@app.cell
def _(MiniMindVLM, VLMConfig, device, hf_hub_download, json, load_file, mo, setup_seed, snapshot_download):
    mo.stop(device != "cuda", mo.callout(mo.md(
        "**GPU를 찾지 못했습니다.** 이 노트북은 NVIDIA GPU에서 학습하도록 만들었습니다. "
        "`uv run python -c 'import torch; print(torch.cuda.is_available())'`로 확인하세요."), kind="danger"))

    BASE_REPO, BASE_REVISION = "caspar/MiniMind-Dense-64M-Mixed-SFT-v1.0", "v1.0"
    VISION_REPO, VISION_REVISION = "jingyaogong/siglip2-base-p32-256-ve", "9465d1dc89db6bc6227c5b6b0e0ca9b940325d62"

    vision_dir = snapshot_download(VISION_REPO, revision=VISION_REVISION)
    _cfg = json.load(open(hf_hub_download(BASE_REPO, "config.json", revision=BASE_REVISION)))
    for _key in ("model_type", "architectures", "auto_map"):  # 언어 모델 전용 항목은 빼고 VLMConfig로 만듦
        _cfg.pop(_key, None)
    vlm_config = VLMConfig(**_cfg)  # image_ids=[12], image_token_len=64, image_hidden_size=768 기본값 추가

    setup_seed(42)  # Projector의 무작위 초깃값을 고정
    base_vlm = MiniMindVLM(vlm_config, vision_model_path=vision_dir)
    _state = {k: v.float() for k, v in load_file(hf_hub_download(BASE_REPO, "model.safetensors", revision=BASE_REVISION)).items()}
    _missing, _unexpected = base_vlm.load_state_dict(_state, strict=False)
    base_vlm.to(device).eval()

    # 파일에 없는 것: lm_head(Embedding과 공유), Projector(새 부품), 이미지 인코더(따로 불러옴)
    assert base_vlm.vision_encoder is not None, "이미지 인코더를 불러오지 못했습니다"
    assert not _unexpected
    assert sorted(k for k in _missing if not k.startswith("vision_encoder.")) == [
        "lm_head.weight", "vision_proj.mlp.0.bias", "vision_proj.mlp.0.weight", "vision_proj.mlp.1.bias",
        "vision_proj.mlp.1.weight", "vision_proj.mlp.3.bias", "vision_proj.mlp.3.weight"]
    assert base_vlm.lm_head.weight is base_vlm.model.embed_tokens.weight

    def count(module):
        return sum(p.numel() for p in module.parameters())

    param_rows = [
        {"부품": "vision_encoder (SigLIP2)", "매개변수": f"{count(base_vlm.vision_encoder) / 1e6:.1f}M", "어디서": VISION_REPO, "학습": "항상 고정"},
        {"부품": "vision_proj (Projector)", "매개변수": f"{count(base_vlm.vision_proj) / 1e6:.2f}M", "어디서": "무작위 초기화", "학습": "항상 학습"},
        {"부품": "언어 모델 (model + lm_head)", "매개변수": f"{(count(base_vlm.model)) / 1e6:.1f}M", "어디서": BASE_REPO, "학습": "freeze_llm에 따라"},
    ]
    return BASE_REPO, VISION_REPO, base_vlm, param_rows, vision_dir, vlm_config


@app.cell(hide_code=True)
def _(base_vlm, mo, param_rows, torch, vlm_config):
    mo.vstack([
        mo.md(f"실행 장치: **GPU (`{torch.cuda.get_device_name(0)}`)**"),
        mo.ui.table(param_rows, selection=None),
        mo.md(f"""
    - `vlm_config.image_special_token = {vlm_config.image_special_token!r}`, `image_ids = {vlm_config.image_ids}`,
      `image_token_len = {vlm_config.image_token_len}`, `image_hidden_size = {vlm_config.image_hidden_size}`
    - 이미지 인코더 정밀도: `{next(base_vlm.vision_encoder.parameters()).dtype}`, `requires_grad` 켜진 매개변수:
      **{sum(p.requires_grad for p in base_vlm.vision_encoder.parameters())}개** (`get_vision_model`이 모두 고정)
    - 언어 모델보다 이미지 인코더가 더 큽니다. 그래도 학습은 **Projector(1.2M)** 만으로 시작할 수 있습니다.
    """),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. 실습 세계: 색깔 도형

    실제 사진 데이터(수십만 장)는 노트북에서 다루기에 너무 큽니다. 대신 **답이 확실한 작은 세계**를 만듭니다.

    - 색 6가지 × 모양 4가지 = **24가지 조합**. 위치·크기·배경 밝기·색조는 그림마다 무작위입니다.
    - **학습에서 6가지 조합을 뺍니다** (색마다 하나씩). 예: 빨간 원은 한 번도 보여 주지 않지만, 빨간색과 원은 각각 다른 조합으로 봅니다.
      평가에서 이 조합을 맞히면 "그림을 외운 것"이 아니라 **색과 모양을 따로 알아본 것**입니다.
    - 질문 3가지: `What color is the shape?` → `The shape is red.` / `What shape is in the image?` → `It is a circle.` /
      `Describe the image.` → `A red circle on a white background.`
    """)
    return


@app.cell
def _(Image, ImageDraw, io, math):
    COLORS = {"red": (220, 40, 40), "green": (40, 170, 60), "blue": (40, 80, 220),
              "yellow": (240, 210, 30), "purple": (140, 60, 180), "orange": (245, 140, 20)}
    SHAPES = ["circle", "square", "triangle", "star"]
    # 색마다 하나씩, 모양이 고르게 빠지도록 고른 학습 제외 조합
    HELD_OUT = {(c, SHAPES[i % len(SHAPES)]) for i, c in enumerate(COLORS)}
    QA_TEMPLATES = [("What color is the shape?", "The shape is {c}."),
                    ("What shape is in the image?", "It is a {s}."),
                    ("Describe the image.", "A {c} {s} on a white background.")]

    def draw_shape(color, shape, rng):
        """밝은 배경에 도형 하나. 위치·크기·배경·색조는 rng로 무작위"""
        img = Image.new("RGB", (256, 256), tuple(rng.randint(225, 255) for _ in range(3)))
        d = ImageDraw.Draw(img)
        r = rng.randint(35, 70)
        cx, cy = rng.randint(r + 5, 251 - r), rng.randint(r + 5, 251 - r)
        fill = tuple(max(0, min(255, v + rng.randint(-20, 20))) for v in COLORS[color])
        if shape == "circle":
            d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=fill)
        elif shape == "square":
            d.rectangle([cx - r, cy - r, cx + r, cy + r], fill=fill)
        elif shape == "triangle":
            d.polygon([(cx, cy - r), (cx - r, cy + r), (cx + r, cy + r)], fill=fill)
        else:  # 별: 바깥 점과 안쪽 점을 번갈아 10개
            d.polygon([(cx + (r if k % 2 == 0 else 0.45 * r) * math.sin(k * math.pi / 5),
                        cy - (r if k % 2 == 0 else 0.45 * r) * math.cos(k * math.pi / 5)) for k in range(10)], fill=fill)
        return img

    def png_bytes(img):
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()
    return COLORS, HELD_OUT, QA_TEMPLATES, SHAPES, draw_shape, png_bytes


@app.cell(hide_code=True)
def _(COLORS, HELD_OUT, SHAPES, draw_shape, mo, plt, random):
    _rng = random.Random(7)
    _fig, _axes = plt.subplots(len(SHAPES), len(COLORS), figsize=(9, 6))
    for _i, _s in enumerate(SHAPES):
        for _j, _c in enumerate(COLORS):
            _ax = _axes[_i, _j]
            _ax.imshow(draw_shape(_c, _s, _rng))
            _ax.set_xticks([]); _ax.set_yticks([])
            _held = (_c, _s) in HELD_OUT
            _ax.set_title(f"{_c} {_s}" + ("\n(held out)" if _held else ""), fontsize=8, color="crimson" if _held else "black")
            if _held:
                for _sp in _ax.spines.values():
                    _sp.set_edgecolor("crimson"); _sp.set_linewidth(3)
    _fig.tight_layout()
    _plot = mo.as_html(_fig)
    plt.close(_fig)
    mo.vstack([_plot, mo.md(f"빨간 테두리 **{len(HELD_OUT)}가지 조합**은 학습에 넣지 않고 평가에만 씁니다.")])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. 이미지 인코더: 그림 → 벡터 64개

    ```python
    @staticmethod
    def image2tensor(image, processor):          # 그림 → pixel_values
        if image.mode in ['RGBA', 'LA']: image = image.convert('RGB')
        return processor(images=image, return_tensors="pt")

    @staticmethod
    def get_image_embeddings(image_inputs, vision_model):
        with torch.no_grad():                    # 인코더는 고정: 기울기를 계산하지 않음
            outputs = vision_model(**image_inputs)
        return outputs.last_hidden_state         # [B, 64, 768]
    ```

    - `SiglipImageProcessor`는 그림을 **256×256**으로 바꾸고 픽셀 값을 `(x / 255 - 0.5) / 0.5`로 **[-1, 1]** 범위에 맞춥니다.
    - SigLIP2는 그림을 **32×32 조각(patch)** 으로 자릅니다. 256 / 32 = 8이므로 **8×8 = 64조각**,
      조각마다 **768차원 벡터** 하나가 나옵니다. 조각들은 Transformer 안에서 서로를 참고하므로, 각 벡터는 주변 맥락도 담습니다.
    - 이 64개가 그대로 **이미지 토큰 64개**가 됩니다(`image_token_len = 64`).
    """)
    return


@app.cell
def _(MiniMindVLM, base_vlm, device, draw_shape, random, torch):
    def pixel_inputs(img):
        """그림 하나 → 모델 입력. 원본 데이터셋처럼 [B=1, 이미지 수=1, 3, 256, 256] 모양으로 만듭니다."""
        return {k: v.unsqueeze(0).to(device) for k, v in MiniMindVLM.image2tensor(img, base_vlm.processor).items()}

    @torch.no_grad()
    def patch_features(img):
        """이미지 인코더 출력 [64, 768]"""
        pv = {k: v.flatten(0, 1) for k, v in pixel_inputs(img).items()}
        return MiniMindVLM.get_image_embeddings(pv, base_vlm.vision_encoder)[0].float()

    demo_img = draw_shape("blue", "triangle", random.Random(3))
    demo_pixels = pixel_inputs(demo_img)["pixel_values"]
    demo_feats = patch_features(demo_img)
    assert demo_pixels.shape == (1, 1, 3, 256, 256)
    assert -1.0 <= demo_pixels.min() and demo_pixels.max() <= 1.0
    assert demo_feats.shape == (64, 768)
    return demo_feats, demo_img, demo_pixels, patch_features, pixel_inputs


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 3-1. 인코더는 그림의 어디를 "다르게" 볼까?

    768차원 벡터 64개는 직접 볼 수 없으므로, **주성분 분석(PCA)** 으로 가장 크게 달라지는 방향 3개만 남겨
    빨강·초록·파랑 색으로 칠합니다. 비슷한 벡터를 가진 조각은 비슷한 색이 됩니다.
    왼쪽 그림의 회색 격자가 32×32 조각 경계입니다.
    """)
    return


@app.cell(hide_code=True)
def _(Image, demo_img, draw_shape, glob, minimind_v_dir, mo, patch_features, plt, random, torch):
    def _pca_rgb(feats):
        _x = feats - feats.mean(0, keepdim=True)
        _u, _s, _v = torch.pca_lowrank(_x, q=3, center=False)
        _p = _x @ _v[:, :3]
        _p = (_p - _p.min(0).values) / (_p.max(0).values - _p.min(0).values + 1e-6)
        return _p.reshape(8, 8, 3).cpu().numpy()

    _real_path = sorted(glob.glob(str(minimind_v_dir / "dataset/eval_images/*")))[3]
    _imgs = [("blue triangle", demo_img), ("orange star", draw_shape("orange", "star", random.Random(11))),
             ("real photo", Image.open(_real_path).convert("RGB").resize((256, 256)))]
    _fig, _axes = plt.subplots(2, 3, figsize=(8, 5.4))
    for _k, (_name, _img) in enumerate(_imgs):
        _axes[0, _k].imshow(_img)
        for _t in range(32, 256, 32):
            _axes[0, _k].axhline(_t, color="gray", lw=0.4); _axes[0, _k].axvline(_t, color="gray", lw=0.4)
        _axes[0, _k].set_title(_name, fontsize=9)
        _axes[1, _k].imshow(_pca_rgb(patch_features(_img)), interpolation="nearest")
        _axes[1, _k].set_title("8x8 patch features (PCA→RGB)", fontsize=8)
        for _a in _axes[:, _k]:
            _a.set_xticks([]); _a.set_yticks([])
    _fig.tight_layout()
    _plot = mo.as_html(_fig)
    plt.close(_fig)
    mo.vstack([_plot, mo.md(r"""
    도형이 있는 조각과 배경 조각이 **다른 색**으로 갈라집니다. 이미지 인코더는 이미 "어디에 무엇이 있는지"를 벡터에 담고 있습니다.
    이 인코더는 수억 장의 그림-글 쌍으로 미리 학습되어 있어서, 우리는 **고정한 채로 그대로 씁니다.**
    남은 문제는 언어 모델이 이 벡터를 **읽을 수 있게** 만드는 것입니다.
    """)])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Projector: 이미지 벡터를 언어 모델의 말로

    이미지 인코더의 768차원 공간과 언어 모델의 768차원 공간은 숫자 크기만 같을 뿐 **전혀 다른 좌표계**입니다.
    `MMVisionProjector`가 둘을 잇습니다.

    ```python
    class MMVisionProjector(nn.Module):
        def __init__(self, in_dim, out_dim, ...):
            self.mlp = nn.Sequential(
                nn.LayerNorm(in_dim),          # 인코더 출력의 크기를 정돈
                nn.Linear(in_dim, out_dim),    # 768 → 768 (언어 모델 hidden_size)
                nn.GELU(),
                nn.Linear(out_dim, out_dim),
            )
    ```

    조각마다 **따로** 같은 MLP를 적용하므로 이미지 토큰 수(64)는 그대로입니다. 학습 전에는 무작위 값이라
    출력이 언어 모델에게 의미 없는 잡음입니다.
    """)
    return


@app.cell
def _(base_vlm, demo_feats, torch):
    with torch.no_grad():
        demo_proj = base_vlm.vision_proj(demo_feats.unsqueeze(0))  # [1, 64, 768]
    assert demo_proj.shape == (1, 64, base_vlm.config.hidden_size)
    projector_params = sum(p.numel() for p in base_vlm.vision_proj.parameters())
    return demo_proj, projector_params


@app.cell(hide_code=True)
def _(demo_feats, demo_proj, mo, projector_params):
    mo.md(f"""
    `vision_encoder` 출력 `{tuple(demo_feats.unsqueeze(0).shape)}` → `vision_proj` 출력 `{tuple(demo_proj.shape)}`.
    Projector 매개변수 **{projector_params:,}개** (LayerNorm 1,536 + Linear 590,592 × 2).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. 이미지를 문장 속에 넣기: `<|image_pad|>` 64칸

    언어 모델은 토큰 한 줄만 받습니다. 그래서 질문 속 `<image>` 자리를 **자리표시자 토큰 64개**로 바꿔 둡니다.

    ```python
    # VLMDataset.create_chat_prompt
    content = turn['content'].replace('<image>', '<|image_pad|>' * 64)
    ```

    `forward`에서는 평소처럼 토큰 임베딩을 만든 뒤, **`<|image_pad|>`(id 12) 64칸의 임베딩을 이미지 토큰으로 갈아 끼웁니다.**

    ```python
    hidden_states = self.model.dropout(self.model.embed_tokens(input_ids))     # ① 평소의 토큰 임베딩
    if pixel_values is not None and start_pos == 0:                            #    (KV 캐시로 이어 쓸 때는 생략)
        vision_tensors = self.vision_proj(get_image_embeddings(pixel_values, self.vision_encoder))   # ② [B, 64, 768]
        hidden_states = self.count_vision_proj(tokens=input_ids, h=hidden_states,
                                               vision_tensors=vision_tensors, seqlen=input_ids.shape[1])  # ③ 갈아 끼우기
    # 이후 Transformer 층, norm, lm_head는 언어 모델과 완전히 같음
    ```

    `count_vision_proj`는 토큰 열에서 `image_ids[0]`이 연속된 구간을 찾아, 그 길이만큼 이미지 토큰을 잘라 넣습니다
    (이미지가 여러 장이면 구간마다 다음 이미지). 이 동작을 **마스크 한 줄**로 다시 써서 원본과 같은지 확인합니다.
    """)
    return


@app.cell
def _(MiniMindVLM, base_vlm, demo_img, device, pixel_inputs, tokenizer, torch, vlm_config):
    IMAGE_PAD_ID = vlm_config.image_ids[0]
    assert tokenizer.convert_tokens_to_ids(vlm_config.image_special_token) == IMAGE_PAD_ID == 12

    def prompt_ids(question, with_image=True):
        """추론 프롬프트. SFT 노트북처럼 빈 think 블록을 지우고, <image>를 <|image_pad|> 64개로 바꿉니다."""
        content = ("<image>\n" if with_image else "") + question
        text = tokenizer.apply_chat_template([{"role": "user", "content": content}], tokenize=False,
                                             add_generation_prompt=True).replace("<think>\n\n</think>\n\n", "")
        text = text.replace("<image>", vlm_config.image_special_token * vlm_config.image_token_len)
        return tokenizer(text, add_special_tokens=False).input_ids

    inject_ids = torch.tensor([prompt_ids("What color is the shape?")], device=device)
    with torch.no_grad():
        _h = base_vlm.model.embed_tokens(inject_ids)                                         # ① [1, T, 768]
        _pv = {k: v.flatten(0, 1) for k, v in pixel_inputs(demo_img).items()}
        _vis = base_vlm.vision_proj(MiniMindVLM.get_image_embeddings(_pv, base_vlm.vision_encoder))  # ② [1, 64, 768]
        _ref = base_vlm.count_vision_proj(tokens=inject_ids, h=_h, vision_tensors=_vis, seqlen=inject_ids.shape[1])  # ③ 원본
        _mine = _h.clone()
        _mask = inject_ids == IMAGE_PAD_ID
        _mine[_mask] = _vis.reshape(-1, _vis.shape[-1])                                      # ③ 마스크로 직접
    assert int(_mask.sum()) == 64
    assert torch.equal(_ref, _mine)
    inject_changed = int((_ref != _h).any(-1).sum())  # 임베딩이 바뀐 위치 수
    return IMAGE_PAD_ID, inject_changed, inject_ids, prompt_ids


@app.cell(hide_code=True)
def _(IMAGE_PAD_ID, inject_changed, inject_ids, mo, tokenizer):
    _ids = inject_ids[0].tolist()
    _first = _ids.index(IMAGE_PAD_ID)
    _rows = []
    for _k, _t in enumerate(_ids):
        if _t == IMAGE_PAD_ID and _k != _first:
            continue
        _tok = tokenizer.convert_ids_to_tokens(_t).replace("Ċ", "\\n")
        _rows.append({"위치": f"{_k}~{_k + 63}" if _t == IMAGE_PAD_ID else _k,
                      "토큰": f"{_tok} × 64" if _t == IMAGE_PAD_ID else _tok, "id": _t,
                      "모델에 들어가는 벡터": "🖼️ 이미지 토큰 64개 (vision_proj 출력)" if _t == IMAGE_PAD_ID else "단어 임베딩"})
    mo.vstack([
        mo.md(f"**추론 프롬프트** (`<|image_pad|>` 64개는 한 줄로 묶어 표시, 전체 {len(_ids)}토큰)"),
        mo.ui.table(_rows, selection=None, page_size=30),
        mo.md(f"""
    원본 `count_vision_proj`와 마스크 한 줄 구현의 결과가 **완전히 같았고**, 임베딩이 바뀐 위치는 정확히 **{inject_changed}칸**이었습니다.
    VLM의 비밀은 이것이 전부입니다. 언어 모델 입장에서 이미지는 **"단어 사전에 없는 단어 64개"** 일 뿐입니다.
    위치(RoPE), 어텐션, 이후 층은 글자 토큰과 똑같이 처리합니다.
    """),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. 학습 데이터: 원본 `VLMDataset` 그대로

    원본 데이터는 parquet 파일입니다. 한 행에 `conversations`(JSON 문자열)와 `image_bytes`(PNG/JPEG 바이트)가 있습니다.
    도형 그림으로 같은 형식의 파일을 만들고, **원본 `VLMDataset`** 으로 읽습니다.

    ```python
    # VLMDataset.__getitem__ 요약
    conversations = pre_processing_chat(conversations)   # 20% 확률로 system 메시지 추가 (SFT와 같음)
    prompt = self.create_chat_prompt(conversations)      # <image> → <|image_pad|>×64, 채팅 템플릿
    prompt = post_processing_chat(prompt)                # 빈 think 블록 80% 삭제 (SFT와 같음)
    input_ids = tokenizer(prompt).input_ids[:max_length] + 패딩
    labels = self.generate_labels(input_ids)             # assistant 답변만 정답 (SFT와 같음)
    image_data = image2tensor(Image.open(BytesIO(image_bytes)), self.preprocess)   # pixel_values
    return input_ids, labels, image_data
    ```

    글 쪽 처리는 SFT의 `SFTDataset`과 **같습니다.** 이미지 자리(`<|image_pad|>`)는 user 메시지 안에 있으므로 자연스럽게 정답에서 빠집니다.
    """)
    return


@app.cell
def _(COLORS, HELD_OUT, QA_TEMPLATES, SHAPES, VLMDataset, base_vlm, draw_shape, json, pa, png_bytes, pq, random, tempfile, tokenizer, vlm_config):
    from pathlib import Path as _Path

    vlm_data_dir = _Path(tempfile.mkdtemp(prefix="minimind_vlm_data_"))
    train_combos = [(c, s) for c in COLORS for s in SHAPES if (c, s) not in HELD_OUT]

    def make_rows(combos, n_per_combo, seed):
        rng = random.Random(seed)
        rows = []
        for c, s in combos:
            for _ in range(n_per_combo):
                q, a = rng.choice(QA_TEMPLATES)
                conv = [{"role": "user", "content": "<image>\n" + q}, {"role": "assistant", "content": a.format(c=c, s=s)}]
                rows.append({"conversations": json.dumps(conv), "image_bytes": png_bytes(draw_shape(c, s, rng))})
        return rows

    train_rows = make_rows(train_combos, 40, seed=0)
    pq.write_table(pa.Table.from_pylist(train_rows), vlm_data_dir / "train_i2t.parquet")

    VLM_MAX_LEN = 128  # 원본 기본값은 768(SFT)/450(pretrain). 이미지 64 + 짧은 대화라 128이면 충분합니다.

    def load_vlm_dataset(path):
        return VLMDataset(str(path), tokenizer, preprocess=base_vlm.processor, max_length=VLM_MAX_LEN,
                          image_special_token=vlm_config.image_special_token, image_token_len=vlm_config.image_token_len)

    train_ds = load_vlm_dataset(vlm_data_dir / "train_i2t.parquet")

    # 평가용 그림: 24가지 조합 × 3장 (학습과 다른 시드 → 위치·크기·색조가 다른 새 그림)
    _rng = random.Random(123)
    test_items = [(c, s, draw_shape(c, s, _rng)) for c in COLORS for s in SHAPES for _ in range(3)]

    random.seed(0)
    sample_ids, sample_labels, sample_pixels = train_ds[0]
    assert len(train_ds) == 18 * 40
    assert int((sample_ids == 12).sum()) == 64
    assert sample_pixels["pixel_values"].shape == (1, 3, 256, 256)
    return load_vlm_dataset, make_rows, sample_ids, sample_labels, test_items, train_combos, train_ds, train_rows, vlm_data_dir


@app.cell(hide_code=True)
def _(json, mo, sample_ids, sample_labels, tokenizer, train_rows, train_ds):
    _text = tokenizer.decode(sample_ids).replace("<|image_pad|>" * 64, "<|image_pad|>×64").replace("<|endoftext|>", "")
    _real = int((sample_ids != tokenizer.pad_token_id).sum())
    _target = tokenizer.decode(sample_labels[sample_labels != -100])
    mo.md(f"""
    학습 행 **{len(train_ds)}개** (학습 조합 18가지 × 40장, 질문 3가지 중 무작위). parquet 첫 행의 `conversations`:

    ```json
    {train_rows[0]["conversations"]}
    ```

    `train_ds[0]`이 만든 입력 (패딩 제외 **{_real}토큰** 중 이미지 자리 64토큰):

    ```text
    {_text.strip()}
    ```

    정답(`labels != -100`)만 이어 붙이면: `{_target!r}` — 질문과 이미지 자리는 정답이 아닙니다.
    `pixel_values`는 그림 1장당 `[1, 3, 256, 256]`이고, `vlm_collate_fn`이 묶으면 `[B, 1, 3, 256, 256]`이 됩니다.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7. 학습 전: 무작위 Projector로 그림을 보여 주면?

    언어 모델은 대화를 잘하지만, Projector가 무작위라 이미지 토큰 64개는 **의미 없는 잡음**입니다.
    """)
    return


@app.cell
def _(autocast_ctx, device, pixel_inputs, prompt_ids, tokenizer, torch):
    @torch.inference_mode()
    def ask(model, img, question, max_new_tokens=20):
        """탐욕적 생성. img가 None이면 이미지 없이 글만 묻습니다."""
        x = torch.tensor([prompt_ids(question, with_image=img is not None)], device=device)
        with autocast_ctx():
            y = model.generate(inputs=x, pixel_values=pixel_inputs(img) if img is not None else None,
                               max_new_tokens=max_new_tokens, do_sample=False, temperature=1.0, top_p=1.0, top_k=0,
                               eos_token_id=tokenizer.eos_token_id)
        new = y[0, x.shape[1]:].tolist()
        if tokenizer.eos_token_id in new:
            new = new[: new.index(tokenizer.eos_token_id)]
        return tokenizer.decode(new, skip_special_tokens=True).strip()

    TEXT_QUESTIONS = ["What is the capital of France?", "How many legs does a spider have?",
                      "What do bees make?", "Which ocean is the largest?"]
    return TEXT_QUESTIONS, ask


@app.cell
def _(HELD_OUT, TEXT_QUESTIONS, ask, test_items):
    def evaluate(model, base_text_answers=None):
        """색·모양 정답률(학습 조합 / 제외 조합)과, 이미지 없는 일반 질문 답이 베이스와 같은지"""
        score = {"seen": [0, 0, 0], "held": [0, 0, 0]}
        for c, s, img in test_items:
            key = "held" if (c, s) in HELD_OUT else "seen"
            score[key][0] += c in ask(model, img, "What color is the shape?", 8)
            score[key][1] += s in ask(model, img, "What shape is in the image?", 8)
            score[key][2] += 1
        text_answers = [ask(model, None, q, 24) for q in TEXT_QUESTIONS]
        result = {f"{k}_{m}": v[i] / v[2] for k, v in score.items() for i, m in enumerate(["color", "shape"])}
        result["text_answers"] = text_answers
        if base_text_answers is not None:
            result["text_same"] = sum(a == b for a, b in zip(text_answers, base_text_answers))
        return result
    return (evaluate,)


@app.cell
def _(TEXT_QUESTIONS, ask, base_vlm, draw_shape, evaluate, random):
    before_imgs = [("red", "square", draw_shape("red", "square", random.Random(21))),
                   ("green", "circle", draw_shape("green", "circle", random.Random(22))),
                   ("yellow", "triangle", draw_shape("yellow", "triangle", random.Random(23)))]
    before_answers = [{q: ask(base_vlm, img, q, 24) for q in ["What color is the shape?", "Describe the image."]}
                      for _, _, img in before_imgs]
    base_text_answers = [ask(base_vlm, None, q, 24) for q in TEXT_QUESTIONS]
    base_eval = evaluate(base_vlm, base_text_answers)
    return base_eval, base_text_answers, before_answers, before_imgs


@app.cell(hide_code=True)
def _(base_eval, before_answers, before_imgs, mo, png_bytes):
    mo.vstack([
        mo.hstack([
            mo.vstack([mo.image(png_bytes(_img), width=140), mo.md(f"**정답:** {_c} {_s}"),
                       *[mo.md(f"*{_q}*<br>→ {_a}") for _q, _a in _ans.items()]])
            for (_c, _s, _img), _ans in zip(before_imgs, before_answers)
        ], widths="equal", align="start"),
        mo.md(f"""
    문장은 매끄럽지만 그림과 상관없는 내용을 지어냅니다(**환각**). 평가 그림 72장의 정답률:
    색 {base_eval['seen_color']:.0%}, 모양 {base_eval['seen_shape']:.0%} (학습 조합 기준).
    언어 모델은 "그림을 묘사하는 말투"는 알지만, 그림을 **볼 통로**가 아직 막혀 있습니다.
    """),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 8. 학습: 무엇을 얼마나 풀어 줄까 (`freeze_llm`)

    원본 `init_vlm_model`은 먼저 **Projector만** 학습 가능하게 둔 뒤, `freeze_llm`에 따라 언어 모델을 더 풀어 줍니다.
    이미지 인코더는 어떤 경우에도 고정입니다.

    ```python
    for name, param in model.named_parameters():          # 1. Projector만 학습
        if 'vision_proj' not in name: param.requires_grad = False
    if freeze_llm == 0:                                   # 2-0. 인코더 빼고 전부
        ...  'vision_encoder' not in name → True
    elif freeze_llm == 1:                                 # 2-1. + 언어 모델의 첫 층과 마지막 층
        ...  'layers.0.' or 'layers.{마지막}.' → True
    elif freeze_llm == 2: pass                            # 2-2. Projector만
    ```

    | 원본 단계 | 스크립트 | `freeze_llm` | 학습률 | 데이터 |
    | --- | --- | --- | --- | --- |
    | ① 정렬 (pretrain) | `train_pretrain_vlm.py` | **2** (Projector만) | 4e-4 | 그림 설명 `pretrain_i2t.parquet` |
    | ② 시각 SFT | `train_sft_vlm.py` | **1** (+ 첫·마지막 층) | 5e-6 | 그림 질의응답 `sft_i2t.parquet` |

    `run_training`은 `train_sft_vlm.py`의 "8. 학습 시작" + `train_epoch`를 옮긴 함수입니다(DDP·저장·재개 제외).
    bf16 autocast를 쓰므로 `GradScaler`는 꺼진 상태(`enabled=False`)와 같아 생략했습니다. 네 가지 실험을 합니다.

    | 실험 | `freeze_llm` | 학습 매개변수 | 학습률 | 그림 | 확인할 것 |
    | --- | --- | --- | --- | --- | --- |
    | **A** | 2 | Projector (1.2M) | 위젯 (기본 1e-3) | 도형 | Projector만으로 볼 수 있게 되는가 |
    | **B** | 1 | + 첫·마지막 층 (15.9M) | 2e-4 | 도형 | 언어 모델을 일부 풀면 무엇이 달라지나 |
    | **C** (대조군) | 2 | Projector | A와 같음 | **모두 빈 회색 그림** | 그림 없이 글만으로 얼마나 맞히나 |
    | **D** | 1 | + 첫·마지막 층 | **1e-3** | 도형 | 학습률이 크면 (8-2절) |

    데이터가 720개뿐이라 원본보다 큰 학습률을 씁니다(원본은 수십만 개, 1~2 epoch).
    """)
    return


@app.cell
def _(DataLoader, SkipBatchSampler, autocast_ctx, device, get_lr, setup_seed, torch, vlm_collate_fn):
    def apply_freeze(model, freeze_llm):
        """init_vlm_model의 '1. 전체 고정 후 vision_proj만' + '2. 전략 확인'"""
        for name, param in model.named_parameters():
            if "vision_proj" not in name:
                param.requires_grad = False
        if freeze_llm == 0:
            for name, param in model.named_parameters():
                if "vision_encoder" not in name:
                    param.requires_grad = True
        elif freeze_llm == 1:
            last_idx = model.config.num_hidden_layers - 1
            for name, param in model.model.named_parameters():
                if "layers.0." in name or f"layers.{last_idx}." in name:
                    param.requires_grad = True
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def run_training(model, train_ds, args, freeze_llm, history=None, logs=None):
        """train_sft_vlm.py의 '8. 학습 시작' + train_epoch (bf16 autocast, 단일 GPU)"""
        n_trainable = apply_freeze(model, freeze_llm)
        model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args["learning_rate"])  # 원본처럼 전체를 넘김(기울기 없는 것은 건너뜀)
        model.train()
        for epoch in range(args["epochs"]):
            setup_seed(42 + epoch); indices = torch.randperm(len(train_ds)).tolist()
            loader = DataLoader(train_ds, batch_sampler=SkipBatchSampler(indices, args["batch_size"], 0),
                                collate_fn=vlm_collate_fn)
            iters = len(loader)
            for step, (input_ids, labels, pixel_values) in enumerate(loader, start=1):
                input_ids, labels = input_ids.to(device), labels.to(device)
                pixel_values = {k: v.to(device) for k, v in pixel_values.items()}
                lr = get_lr(epoch * iters + step, args["epochs"] * iters, args["learning_rate"])
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr
                with autocast_ctx():
                    res = model(input_ids, labels=labels, pixel_values=pixel_values)
                    loss = (res.loss + res.aux_loss) / args["accumulation_steps"]
                loss.backward()
                if step % args["accumulation_steps"] == 0 or step == iters:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args["grad_clip"])
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                if history is not None:
                    history.append(loss.item() * args["accumulation_steps"])
                if logs is not None and step == iters:
                    logs.append(f"Epoch:[{epoch + 1}/{args['epochs']}]({step}/{iters}), "
                                f"loss: {loss.item() * args['accumulation_steps']:.4f}, lr: {lr:.8f}")
        return model.eval(), n_trainable
    return apply_freeze, run_training


@app.cell(hide_code=True)
def _(mo):
    vlm_proj_lr = mo.ui.dropdown(options={"4e-4 (원본 pretrain)": 4e-4, "1e-3": 1e-3}, value="1e-3", label="A·C learning_rate")
    vlm_epochs = mo.ui.dropdown(options={"1": 1, "3": 3, "5": 5}, value="5", label="epochs")
    mo.hstack([vlm_proj_lr, vlm_epochs], justify="start", gap=2)
    return vlm_epochs, vlm_proj_lr


@app.cell
def _(Image, base_vlm, copy, load_vlm_dataset, pa, png_bytes, pq, run_training, setup_seed, train_ds, train_rows, vlm_data_dir, vlm_epochs, vlm_proj_lr):
    import time as _time

    _common = {"epochs": vlm_epochs.value, "batch_size": 16, "accumulation_steps": 1, "grad_clip": 1.0}
    args_a = {**_common, "learning_rate": vlm_proj_lr.value}
    args_b = {**_common, "learning_rate": 2e-4}

    # 대조군 C: 같은 대화, 그림만 모두 빈 회색으로
    _blank = png_bytes(Image.new("RGB", (256, 256), (240, 240, 240)))
    pq.write_table(pa.Table.from_pylist([{"conversations": r["conversations"], "image_bytes": _blank} for r in train_rows]),
                   vlm_data_dir / "blind_i2t.parquet")
    blind_ds = load_vlm_dataset(vlm_data_dir / "blind_i2t.parquet")

    hist_a, hist_b, hist_c, logs_a = [], [], [], []
    train_seconds = {}
    for _name, _ds, _args, _freeze, _hist in [("A", train_ds, args_a, 2, hist_a), ("B", train_ds, args_b, 1, hist_b),
                                               ("C", blind_ds, args_a, 2, hist_c)]:
        setup_seed(0)
        _t0 = _time.time()
        _model, _n = run_training(copy.deepcopy(base_vlm), _ds, _args, _freeze, history=_hist,
                                  logs=logs_a if _name == "A" else None)
        train_seconds[_name] = (_time.time() - _t0, _n)
        if _name == "A":
            model_a = _model
        elif _name == "B":
            model_b = _model
        else:
            model_c = _model
    return args_a, args_b, hist_a, hist_b, hist_c, logs_a, model_a, model_b, model_c, train_seconds


@app.cell
def _(base_eval, base_text_answers, evaluate, model_a, model_b, model_c):
    vlm_evals = {"학습 전": base_eval, "A": evaluate(model_a, base_text_answers),
                 "B": evaluate(model_b, base_text_answers), "C (빈 그림)": evaluate(model_c, base_text_answers)}
    return (vlm_evals,)


@app.cell(hide_code=True)
def _(args_a, args_b, hist_a, hist_b, hist_c, logs_a, mo, plt, train_seconds, vlm_evals):
    _fig, _ax = plt.subplots(figsize=(6.5, 3))
    _ax.plot(hist_a, lw=0.9, label=f"A: projector only (lr={args_a['learning_rate']})")
    _ax.plot(hist_b, lw=0.9, label="B: + first/last LLM layer (lr=2e-4)")
    _ax.plot(hist_c, lw=0.9, label="C: projector only, blank images")
    _ax.set_yscale("log"); _ax.set_xlabel("step"); _ax.set_ylabel("loss (answer tokens)")
    _ax.legend(fontsize=8); _fig.tight_layout()
    _plot = mo.as_html(_fig)
    plt.close(_fig)

    _time_rows = [{"실험": k, "학습 매개변수": f"{n / 1e6:.2f}M", "학습 시간 (GPU)": f"{t:.1f}초"} for k, (t, n) in train_seconds.items()]
    _rows = [{"모델": k, "색 (학습 조합 54장)": f"{v['seen_color']:.0%}", "모양 (학습 조합)": f"{v['seen_shape']:.0%}",
              "색 (제외 조합 18장)": f"{v['held_color']:.0%}", "모양 (제외 조합)": f"{v['held_shape']:.0%}",
              "일반 질문 답이 베이스와 같음": f"{v.get('text_same', 4)}/4"} for k, v in vlm_evals.items()]
    mo.vstack([
        mo.md(f"설정: epochs={args_a['epochs']}, batch 16, A·C lr={args_a['learning_rate']}, B lr={args_b['learning_rate']}"),
        mo.hstack([_plot, mo.ui.table(_time_rows, selection=None)], align="center"),
        mo.ui.table(_rows, selection=None),
        mo.md("**실험 A 로그 (epoch 끝마다)**\n```text\n" + "\n".join(logs_a) + "\n```"),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **읽는 법** (기본 설정 기준. 설정을 바꾸면 숫자도 바뀝니다.)

    - **A (Projector만):** 매개변수 1.2M을 GPU에서 20초 남짓 학습했는데, 학습한 조합의 색과 모양을 모두 맞힙니다.
      학습에서 뺀 조합(예: 빨간 원)도 모양은 전부, 색은 대부분 맞힙니다. 이미지 인코더가 이미 색과 모양을 구분해 두었고,
      Projector는 그것을 언어 모델이 읽을 수 있는 형태로 **옮겨 주기만** 하면 되기 때문입니다.
    - **처음 보는 조합의 색은 늦게 배웁니다.** epochs를 3으로 줄이면 제외 조합의 모양은 여전히 100%지만 색은 56%로 떨어집니다.
      예를 들어 노란 별을 "purple star"(학습에 있던 조합)라고 답합니다. 학습 초반에는 **학습에서 본 조합을 통째로 떠올리는 지름길**을 쓰다가,
      더 학습하면 색과 모양을 따로 읽게 되는 것으로 보입니다. 학습 조합만으로 평가했다면 이 차이를 놓쳤을 것입니다.
    - **A의 일반 질문 답은 베이스와 4/4 같습니다.** 언어 모델 가중치를 하나도 바꾸지 않았고, 이미지가 없으면 Projector는 쓰이지 않으므로
      글 능력은 **완전히 그대로**입니다. Projector만 학습하는 정렬 단계의 큰 장점입니다.
    - **B (+ 첫·마지막 층):** 그림 정답률은 A와 비슷하지만, 일반 질문 답은 **하나도 베이스와 같지 않습니다.**
      `What do bees make?`에 도형 묘사를 하기도 합니다(8-2절 표). 학습률이 작고(2e-4) 8개 층 중 2개만 풀었는데도,
      한 가지 형식의 답만 반복해서 보여 주면 언어 모델이 그 형식을 글 질문에도 옮깁니다.
    - **C (빈 그림, 대조군):** 손실이 A보다 높은 곳에서 멈추고, 정답률은 찍기 수준(색 1/6 ≈ 17%, 모양 1/4 = 25%)이거나 그보다 낮습니다
      (주로 한 가지 색·모양만 답함). 답의 **형식**("The shape is ...")은 배우지만, **내용**은 그림 없이는 알 수 없습니다.
      A와의 차이가 곧 "모델이 실제로 그림을 보고 답한다"는 증거입니다.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 8-1. 학습에서 뺀 조합의 답 보기
    """)
    return


@app.cell
def _(HELD_OUT, ask, model_a, model_b, model_c, test_items):
    held_examples = [(c, s, img) for c, s, img in test_items if (c, s) in HELD_OUT][::3]
    held_answers = [{name: ask(m, img, "Describe the image.", 16) for name, m in [("A", model_a), ("B", model_b), ("C", model_c)]}
                    for _, _, img in held_examples]
    return held_answers, held_examples


@app.cell(hide_code=True)
def _(held_answers, held_examples, mo, png_bytes):
    mo.vstack([
        mo.md("질문: *Describe the image.* — 6장 모두 **학습에서 한 번도 보지 못한 색·모양 조합**입니다."),
        mo.hstack([
            mo.vstack([mo.image(png_bytes(_img), width=110), mo.md(f"**정답:** {_c} {_s}"),
                       *[mo.md(f"{_k}: {_v}") for _k, _v in _ans.items()]])
            for (_c, _s, _img), _ans in zip(held_examples, held_answers)
        ], widths="equal", align="start", wrap=True),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 8-2. 언어 모델 층을 큰 학습률로 풀면 (실험 D)

    B와 같이 첫·마지막 층을 풀되 학습률만 A와 같은 **1e-3**으로 키웁니다.
    Projector에는 괜찮았던 학습률이 언어 모델 층에는 어떤 영향을 줄까요?
    """)
    return


@app.cell
def _(base_text_answers, base_vlm, copy, evaluate, run_training, setup_seed, train_ds, vlm_epochs):
    setup_seed(0)
    model_d, _ = run_training(copy.deepcopy(base_vlm), train_ds,
                              {"epochs": vlm_epochs.value, "batch_size": 16, "accumulation_steps": 1,
                               "grad_clip": 1.0, "learning_rate": 1e-3}, freeze_llm=1)
    eval_d = evaluate(model_d, base_text_answers)
    return eval_d, model_d


@app.cell(hide_code=True)
def _(TEXT_QUESTIONS, base_text_answers, eval_d, mo, vlm_evals):
    _rows = [{"이미지 없는 일반 질문": q, "베이스": b, "A (Projector만)": a, "B (lr 2e-4)": bb, "D (lr 1e-3)": d}
             for q, b, a, bb, d in zip(TEXT_QUESTIONS, base_text_answers, vlm_evals["A"]["text_answers"],
                                       vlm_evals["B"]["text_answers"], eval_d["text_answers"])]
    mo.vstack([
        mo.md(f"D의 그림 정답률: 색 {eval_d['seen_color']:.0%} / 모양 {eval_d['seen_shape']:.0%} (학습 조합), "
              f"색 {eval_d['held_color']:.0%} / 모양 {eval_d['held_shape']:.0%} (제외 조합)"),
        mo.ui.table(_rows, selection=None),
        mo.md(r"""
    D는 색 정답률이 크게 떨어지고, 거의 모든 글 질문에 "It is a square." 같은 도형 답을 하거나 의미 없는 토큰을 반복합니다.
    **언어 모델 자체가 망가진 것입니다**(catastrophic forgetting). B에서 시작된 오염이 학습률을 키우자 전체로 번졌습니다.
    작은 Projector는 큰 학습률을 견디지만, 이미 학습된 언어 모델 층은 작은 변화에도 민감합니다.
    원본이 정렬 단계(Projector만)에 4e-4, 언어 모델 층을 푸는 SFT 단계에 **5e-6**처럼 훨씬 작은 학습률을 쓰는 이유입니다.
    """),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 9. 정말 그림을 보고 답할까?

    ### 9-1. 같은 질문, 다른 그림

    질문은 그대로 두고 그림만 바꿉니다. 답이 그림을 따라 바뀌어야 합니다. 대조군 C는 그림이 달라도 답이 거의 같습니다.
    """)
    return


@app.cell
def _(ask, draw_shape, model_a, model_c, random):
    swap_imgs = [draw_shape(c, s, random.Random(40 + k)) for k, (c, s) in
                 enumerate([("green", "star"), ("purple", "triangle"), ("blue", "circle"), ("orange", "star"), ("yellow", "square")])]
    swap_answers = [(ask(model_a, img, "What color is the shape?", 8), ask(model_c, img, "What color is the shape?", 8))
                    for img in swap_imgs]
    return swap_answers, swap_imgs


@app.cell(hide_code=True)
def _(mo, png_bytes, swap_answers, swap_imgs):
    mo.hstack([
        mo.vstack([mo.image(png_bytes(_img), width=100), mo.md(f"A: {_a}<br>C: {_c}")])
        for _img, (_a, _c) in zip(swap_imgs, swap_answers)
    ], widths="equal", align="start")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 9-2. 이미지 토큰은 "단어"가 되었을까?

    Projector가 그림을 기존 단어(`red`, `circle` 등)의 임베딩으로 **번역**하는 걸까요?
    학습된 Projector의 출력과 단어 임베딩 6,400개를 비교합니다.
    - **크기(노름):** 이미지 토큰 벡터와 단어 임베딩의 평균 길이
    - **가장 가까운 단어:** 이미지 토큰 64개 각각과 코사인 유사도가 가장 높은 단어
    """)
    return


@app.cell
def _(F, MiniMindVLM, base_vlm, draw_shape, model_a, pixel_inputs, random, tokenizer, torch):
    @torch.no_grad()
    def projected_tokens(model, img):
        pv = {k: v.flatten(0, 1) for k, v in pixel_inputs(img).items()}
        return model.vision_proj(MiniMindVLM.get_image_embeddings(pv, model.vision_encoder))[0].float()

    _emb = base_vlm.model.embed_tokens.weight.float()
    _probe = draw_shape("red", "square", random.Random(5))
    soft_rows = []
    for _name, _m in [("학습 전 Projector", base_vlm), ("A의 Projector", model_a)]:
        _v = projected_tokens(_m, _probe)
        _sim = F.normalize(_v, dim=-1) @ F.normalize(_emb, dim=-1).T          # [64, 6400]
        _best = _sim.max(-1)                                                  # 이미지 토큰마다 가장 가까운 단어
        _top = torch.bincount(_best.indices, minlength=_emb.shape[0]).topk(5).indices.tolist()
        soft_rows.append({"벡터": _name, "평균 노름": f"{_v.norm(dim=-1).mean():.1f}",
                          "가장 가까운 단어와의 코사인 (평균)": f"{_best.values.mean():.2f}",
                          "가장 자주 가까운 단어 5개": ", ".join(repr(tokenizer.decode([t])) for t in _top)})
    _red = tokenizer(" red", add_special_tokens=False).input_ids[0]
    _sq = tokenizer(" square", add_special_tokens=False).input_ids[0]
    soft_rows.append({"벡터": "단어 임베딩 (참고)", "평균 노름": f"{_emb.norm(dim=-1).mean():.1f}",
                      "가장 가까운 단어와의 코사인 (평균)": f"' red'↔' square': {F.cosine_similarity(_emb[_red], _emb[_sq], dim=0):.2f}",
                      "가장 자주 가까운 단어 5개": "—"})
    return projected_tokens, soft_rows


@app.cell(hide_code=True)
def _(mo, soft_rows):
    mo.vstack([
        mo.md("빨간 사각형 그림의 이미지 토큰 64개를 분석한 결과:"),
        mo.ui.table(soft_rows, selection=None),
        mo.md(r"""
    - 학습된 이미지 토큰은 단어 임베딩보다 **수십 배 깁니다.** 언어 모델의 각 층은 입력을 RMSNorm으로 정규화한 뒤 읽으므로
      길이 자체가 문제 되지는 않고, 오히려 큰 값이 residual 흐름에서 오래 살아남습니다.
    - 가장 가까운 단어는 `red`나 `square`가 아니라 **의미 없는 조각 단어**이고, 코사인도 낮습니다.
      Projector는 그림을 **기존 단어로 번역하지 않습니다.** 언어 모델이 읽을 수 있는 **새로운 "연속 단어"** (soft token)를 만들 뿐입니다.
    - 그래서 "이미지 = 단어 사전에 없는 단어 64개"라는 5절의 표현이 정확합니다. 언어 모델은 학습을 통해 이 낯선 벡터에서
      색과 모양을 꺼내 쓰는 법을 **Projector 쪽을 바꾸어서** 얻었습니다.
    """),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 10. 실제 사진을 보여 주면

    우리 모델은 "흰 배경 위 도형 하나"만 본 모델입니다. `projects/minimind-v/dataset/eval_images/`의 실제 사진을 보여 주면 어떻게 답할까요?
    """)
    return


@app.cell
def _(Image, ask, glob, minimind_v_dir, model_a):
    real_paths = sorted(glob.glob(str(minimind_v_dir / "dataset/eval_images/*")))
    real_imgs = [Image.open(p).convert("RGB") for p in real_paths]
    real_answers_a = [{q: ask(model_a, img, q, 24) for q in ["Describe the image.", "What color is the shape?"]} for img in real_imgs]
    return real_answers_a, real_imgs, real_paths


@app.cell(hide_code=True)
def _(mo, real_answers_a, real_paths):
    from pathlib import Path as _P

    mo.vstack([
        mo.hstack([
            mo.vstack([mo.image(_p, width=150), mo.md(f"`{_P(_p).name}`"),
                       *[mo.md(f"*{_q}*<br>→ {_a}") for _q, _a in _ans.items()]])
            for _p, _ans in zip(real_paths, real_answers_a)
        ], widths="equal", align="start", wrap=True),
        mo.md(r"""
    모델 A는 실제 사진에 **"A yellow square on a white background."** 처럼 자기가 아는 세계의 말로 답하거나,
    색을 물었는데 모양을 답하는 등 형식이 무너지기도 합니다. 학습에서 본 적 없는 입력이라 Projector 출력도 낯선 값이 되기 때문입니다.
    그래도 색 단어는 사진의 주된 색을 따라가는 경우가 있습니다(예: 노란 자동차 → yellow). 인코더가 색을 담고 있기 때문입니다.
    **VLM이 무엇을 말할 수 있는지는 정렬·SFT 데이터가 정합니다.** 실제 사진을 설명하려면 실제 사진-설명 데이터가 필요합니다.
    """),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 10-1. (선택) 같은 베이스를 실제 데이터로 학습한 모델과 비교

    [`caspar/MiniMind-V-Dense-64M-LLaVA-Short-v1.0`](https://huggingface.co/caspar/MiniMind-V-Dense-64M-LLaVA-Short-v1.0)은
    **이 노트북과 같은 베이스 언어 모델**에서 출발해 원본과 같은 2단계로 학습한 공개 모델입니다.

    - ① Projector만 정렬: LLaVA-Pretrain 그림 설명 약 5만 개, lr 4e-4, 1 epoch
    - ② Projector + 첫·마지막 층 SFT: LLaVA-Instruct의 짧은 질의응답 약 15만 개, lr 1e-5, 1 epoch
    - 이미지 인코더는 같은 SigLIP2입니다(이 노트북의 189MB 인코더와 출력 코사인 0.9999 이상). 그래서 **언어 모델 + Projector 가중치(약 130MB)만** 받아
      같은 `MiniMindVLM`에 넣습니다.

    아래 버튼을 누르면 내려받아 같은 사진에 답하게 합니다.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    llava_button = mo.ui.run_button(label="LLaVA로 학습한 모델 불러와 비교하기 (약 130MB)")
    llava_button
    return (llava_button,)


@app.cell
def _(MiniMindVLM, VLMConfig, ask, device, hf_hub_download, json, llava_button, load_file, mo, real_imgs, vision_dir):
    mo.stop(not llava_button.value, mo.md("*버튼을 누르면 실행합니다.*"))
    LLAVA_REPO, LLAVA_REVISION = "caspar/MiniMind-V-Dense-64M-LLaVA-Short-v1.0", "4e6f87f184b27f0692e1f8b3afed959de254894b"
    _cfg = json.load(open(hf_hub_download(LLAVA_REPO, "config.json", revision=LLAVA_REVISION)))
    _cfg.pop("model_type", None)
    llava_vlm = MiniMindVLM(VLMConfig(**_cfg), vision_model_path=vision_dir)
    _state = {k: v.float() for k, v in load_file(hf_hub_download(LLAVA_REPO, "model.safetensors", revision=LLAVA_REVISION)).items()}
    _missing, _unexpected = llava_vlm.load_state_dict(_state, strict=False)
    assert [k for k in _missing if not k.startswith("vision_encoder.")] == ["lm_head.weight"] and not _unexpected
    llava_vlm.to(device).eval()
    llava_answers = [{q: ask(llava_vlm, img, q, 40) for q in ["What is in this image?", "What color is the main object?"]}
                     for img in real_imgs]
    return LLAVA_REPO, llava_answers


@app.cell(hide_code=True)
def _(LLAVA_REPO, llava_answers, mo, real_paths):
    from pathlib import Path as _P2

    mo.vstack([
        mo.md(f"`{LLAVA_REPO}`의 답 (탐욕적 생성):"),
        mo.hstack([
            mo.vstack([mo.image(_p, width=150), mo.md(f"`{_P2(_p).name}`"),
                       *[mo.md(f"*{_q}*<br>→ {_a}") for _q, _a in _ans.items()]])
            for _p, _ans in zip(real_paths, llava_answers)
        ], widths="equal", align="start", wrap=True),
        mo.md(r"""
    실제 사진-글 데이터로 학습한 모델은 "In this image, ..."처럼 **실제 사진을 묘사하는 말투**를 배웠고, 일부는 맞힙니다(예: 자동차).
    하지만 사진에 없는 물건이나 틀린 색을 자신 있게 말하는 **환각**도 많습니다.
    64M 언어 모델과 64개 이미지 토큰으로는 복잡한 장면을 정확히 담기 어렵습니다.
    모델 카드도 "유창함을 정확함으로 해석하지 말라"고 경고합니다. **매끄러운 문장 ≠ 그림을 제대로 봄**입니다.
    """),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 11. 실제 규모로 실행하려면

    데이터(`pretrain_i2t.parquet`, `sft_i2t.parquet`)와 이미지 인코더 준비는 `projects/minimind-v/README_ko.md`를 따릅니다.
    이 노트북의 베이스 언어 모델로 시작하려면, 원본이 찾는 `../out/llm_768.pth` 자리에 가중치를 저장합니다.

    ```python
    # projects/minimind-v/ 에서 실행: caspar 베이스 언어 모델 → out/llm_768.pth
    import torch
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    sd = load_file(hf_hub_download("caspar/MiniMind-Dense-64M-Mixed-SFT-v1.0", "model.safetensors", revision="v1.0"))
    torch.save({k: v.half() for k, v in sd.items()}, "out/llm_768.pth")
    ```

    ```bash
    cd projects/minimind-v/trainer
    python train_pretrain_vlm.py                          # ① llm_768.pth에서 시작, Projector만 (freeze_llm=2) → pretrain_vlm_768.pth
    python train_sft_vlm.py                               # ② pretrain_vlm_768.pth에서 시작, + 첫·마지막 층 (freeze_llm=1) → sft_vlm_768.pth
    python ../eval_vlm.py --weight sft_vlm                # 이미지 질의응답
    ```

    | 인자 | `train_pretrain_vlm.py` | `train_sft_vlm.py` | 이 노트북 A / B |
    | --- | --- | --- | --- |
    | `--from_weight` | `llm` | `pretrain_vlm` | `caspar/...-Mixed-SFT-v1.0` (둘 다) |
    | `--freeze_llm` | 2 | 1 | 2 / 1 |
    | `--learning_rate` | 4e-4 | 5e-6 | 1e-3 / 2e-4 (데이터가 매우 작음) |
    | `--batch_size` | 16 | 4 | 16 |
    | `--epochs` | 2 | 2 | 5 |
    | `--max_seq_len` | 450 | 768 | 128 |
    | `--dtype` | bfloat16 | bfloat16 | bfloat16 autocast |

    ## 12. 정리

    | 항목 | 핵심 |
    | --- | --- |
    | 구조 | 이미지 인코더(고정) → Projector(MLP) → 언어 모델. `MiniMindVLM`은 `MiniMindForCausalLM`을 상속 |
    | 이미지 → 토큰 | 32×32 조각 64개 → 768차원 벡터 64개 → Projector → 이미지 토큰 64개 |
    | 넣는 방법 | 프롬프트의 `<|image_pad|>` 64칸 임베딩을 이미지 토큰으로 교체 (`count_vision_proj`) |
    | 학습 목표 | 언어 모델과 같은 다음 토큰 예측, assistant 답변만 정답 |
    | 정렬 단계 | Projector만 학습 → 글 능력은 완전히 그대로, 작은 데이터로도 빠르게 "보게" 됨 |
    | SFT 단계 | 언어 모델 일부를 풀어 답하는 방식을 다듬음 → 학습률이 크면 언어 능력이 망가짐 |
    | 확인 방법 | 빈 그림 대조군, 그림 바꾸기, 학습에 없던 조합으로 평가 |
    | 한계 | 말할 수 있는 내용은 학습 데이터가 정함. 매끄러운 문장 ≠ 정확한 시각 이해 |

    ## 13. 직접 해 볼 과제

    각 과제를 하기 전에 **결과를 먼저 예측**해 적고, 바꿔 실행한 뒤 비교하세요.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({
        "과제 1. 위치 묻기 (2절)": mo.md("""
    `draw_shape`가 도형 중심 `cx`를 함께 돌려주게 바꾸고, `"Is the shape on the left or the right?"` → `"It is on the left."` 질문을 추가하세요.
    Projector만 학습(A)해도 위치를 맞힐까요?

    *해설:* 이미지 토큰 64개는 조각 순서대로 들어가고, 언어 모델은 각 토큰의 위치(RoPE)를 압니다. 3-1절의 PCA 그림처럼 인코더 벡터에도
    "어느 조각에 도형이 있는지"가 담겨 있으므로 위치도 배울 수 있는 정보입니다. 색·모양보다 학습이 더 필요할 수 있습니다.
    """),
        "과제 2. 학습 조합 줄이기 (6절)": mo.md("""
    `train_combos`를 색마다 2가지 모양만 남기도록 줄이면, 제외 조합의 정답률은 어떻게 될까요?

    *해설:* 색과 모양을 "따로" 배우려면 다양한 조합을 봐야 합니다. 조합이 너무 적으면 "파란색이면 원"처럼 둘을 묶어 외우기 쉽습니다.
    """),
        "과제 3. freeze_llm=0 (8절)": mo.md("""
    실험 B의 `freeze_llm`을 0(인코더 빼고 전부)으로 바꾸면 학습 매개변수 수, 그림 정답률, 일반 질문 답은 어떻게 될까요?

    *해설:* 학습할 매개변수가 64M 이상으로 늘어납니다. 그림 정답률은 비슷하더라도, 언어 모델 전체가 짧은 도형 답에 맞춰지면서
    일반 질문 답이 더 많이 바뀌기 쉽습니다. 원본 SFT 기본값이 `freeze_llm=1`인 이유를 생각해 보세요.
    """),
        "과제 4. 이미지 토큰 줄이기 (4절)": mo.md("""
    인코더 출력 8×8을 2×2씩 평균 내 **16개**로 줄인 뒤 Projector에 넣고, `image_token_len=16`으로 바꾸면 결과가 어떻게 될까요?
    (`MiniMindVLM.forward`의 `vision_proj` 앞에 풀링을 넣고, `VLMConfig(image_token_len=16)`로 만들어야 합니다.)

    *해설:* 토큰이 줄면 문장 길이와 계산량이 줄어듭니다. 큰 도형의 색·모양은 유지될 가능성이 높지만, 작은 물체나 위치 정보는 잃기 쉽습니다.
    실제 VLM들이 이미지 토큰 수와 정확도 사이에서 균형을 잡는 방식입니다.
    """),
        "과제 5. 대조군의 답 살펴보기 (9절)": mo.md("""
    대조군 C에게 여러 그림으로 `"Describe the image."`를 물어보세요. 항상 같은 답을 할까요? 그 답은 어떤 조합일까요?

    *해설:* 그림 정보가 없으면 모델은 학습 데이터에서 손실을 가장 줄이는 "평균적인 답"을 냅니다.
    문장 형식은 완벽하지만 내용은 그림과 무관합니다. 언어 모델의 유창함과 시각적 근거가 별개라는 것을 보여 줍니다.
    """),
    })
    return


if __name__ == "__main__":
    app.run()
