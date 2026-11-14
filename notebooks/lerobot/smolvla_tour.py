import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # SmolVLA 해부 실습

    **SmolVLA**는 Hugging Face가 공개한 작은 VLA(Vision-Language-Action) 모델입니다.
    카메라 이미지와 자연어 지시, 로봇 상태를 받아 **앞으로 실행할 행동 여러 개(행동 청크)** 를
    한 번에 만듭니다.



    **선수 지식:** Python 함수, PyTorch 텐서·역전파, Transformer의 자기/교차 어텐션.
    VLM은 이미지와 언어를 함께 처리하는 모델이며, VQA는 이미지에 대한 질문에 답하는 학습입니다.
    Flow Matching은 6·8절에서 식과 계산 예시로 설명합니다.

    **실행:** `notebooks/lerobot`에서 `uv sync --locked` 후 `uv run marimo edit smolvla_tour.py`를 실행합니다.
    4절 버튼을 누르면 VQA 사전 학습과 9절 정책 학습이 이어집니다. GPU 기본 각 3000 step, CPU 각 500 step이며
    CPU 축소 설정의 품질은 낮을 수 있습니다. 1~11절은 필수, 12~14절과 끝의 추가 학습 과제는 선택 심화입니다.
    marimo는 변수 의존성에 따라 실행합니다. 설정·학습 셀을 바꾸면 하위 셀도 다시 실행됩니다.
    평가의 `policy.reset()`은 큐를 비우며, 커널 재시작은 메모리의 학습 모델도 지웁니다.
    같은 시드와 설정으로 비교하고, 손실이 내려가도 성공률이 낮으면 정규화·정답/오답 도달률·학습 step 수를 함께 확인하세요.

    이 노트북에서 할 일:

    1. 작은 2D 과제 데이터를 만들고 입력 형식(이미지·지시문·상태·행동)을 확인합니다.
    2. SmolVLM 구조(비전 인코더, 픽셀 셔플 커넥터, Llama형 디코더)를 작게 만들고 **먼저 VQA로 사전 학습**합니다.
    3. 원본 `smolvlm_with_expert.py`, `modeling_smolvla.py`의 핵심 코드를 옮겨 와 **VLM + 행동 전문가** 를 조립합니다.
    4. 어텐션 마스크, 층 배치(교차/자기 어텐션 교대), Flow Matching 손실을 하나씩 확인합니다.
    5. VLM은 고정하고 행동 전문가만 학습합니다(`train_expert_only=True`).
    6. KV 캐시를 쓰는 추론, 행동 큐, 폐루프 평가와 다봉(multimodal) 행동 분포를 실험합니다.

    > **원본 코드**: [huggingface/lerobot `src/lerobot/policies/smolvla`](https://github.com/huggingface/lerobot/tree/e6ffde60312365c5452c46189a26872483943154/src/lerobot/policies/smolvla)
    > (revision `e6ffde60312365c5452c46189a26872483943154`) 및 같은 revision의 `policies/common/flow_matching.py`, `policies/common/vla_utils.py`.
    > 원본은 Apache License 2.0, Copyright 2025–2026 The HuggingFace Inc. team입니다. 이 노트북은 원본 코드를
    > 옮겨 와 `transformers` 없이 동작하도록 축소·수정했습니다. `lerobot` 패키지는 설치하지 않습니다.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.mermaid("""
    flowchart LR
        I["이미지: 비전 인코더 + 커넥터"] --> V["VLM 앞쪽 층"]
        L["지시문: 토큰 임베딩"] --> V
        S["상태: state_proj"] --> V
        V --> K["접두부 K/V 캐시"]
        K --> E["행동 전문가"]
        N["노이즈 행동 + 시간 t"] --> E
        E --> U["속도 예측"]
        U --> O["Euler 적분: t=1 에서 0, 행동 청크"]
    """)
    return


@app.cell
def _():
    import copy
    import math
    import time
    from collections import deque
    from dataclasses import dataclass
    from types import SimpleNamespace

    import matplotlib.font_manager as font_manager
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    import torch.nn.functional as F
    from torch import Tensor, nn

    # 그림의 한글 표시용 글꼴: 설치된 것 중 첫 번째를 사용합니다.
    _installed = {f.name for f in font_manager.fontManager.ttflist}
    for _name in ["Noto Sans CJK KR", "NanumGothic", "Malgun Gothic", "AppleGothic", "Noto Sans Mono CJK KR", "Noto Serif CJK KR"]:
        if _name in _installed:
            plt.rcParams["font.family"] = _name
            break
    plt.rcParams["axes.unicode_minus"] = False
    return F, SimpleNamespace, Tensor, copy, dataclass, deque, math, nn, np, plt, time, torch


@app.cell
def _(torch):
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ON_GPU = DEVICE.type == "cuda"

    # CPU에서는 학습 step을 줄입니다. 결과 품질도 함께 낮아집니다.
    VLM_PRETRAIN_STEPS = 3000 if ON_GPU else 500
    POLICY_TRAIN_STEPS = 3000 if ON_GPU else 500
    BATCH_SIZE = 128 if ON_GPU else 32

    torch.manual_seed(0)
    print(f"device={DEVICE}, VLM 사전 학습 {VLM_PRETRAIN_STEPS} step, 정책 학습 {POLICY_TRAIN_STEPS} step, batch {BATCH_SIZE}")
    return BATCH_SIZE, DEVICE, POLICY_TRAIN_STEPS, VLM_PRETRAIN_STEPS


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. 실습 과제: "빨간 점으로 가"

    실제 로봇 대신 2D 평면 과제를 사용합니다.

    - **이미지** (`observation.images.top`): 24×32 RGB. 빨간 점과 파란 점이 무작위 위치에 있습니다.
      로봇(에이전트) 위치는 이미지에 그리지 않습니다.
    - **상태** (`observation.state`): 에이전트의 현재 위치 `(x, y)`. 실제 로봇의 관절 각도에 해당합니다.
    - **지시문** (`task`): `"go to the red dot"`, `"move to blue"`, `"any dot"` 등.
      어느 점으로 갈지는 **지시문을 읽어야만** 알 수 있습니다.
    - **행동** (`action`): 다음 16 step 동안의 목표 위치 `[16, 2]`. 한 step에 최대 0.05씩 직선으로 움직이는 시연입니다.

    `"any dot"` 지시는 두 점 중 아무 곳이나 정답입니다. 같은 관측에 정답이 두 개라서
    행동 분포가 **두 봉우리(다봉)** 를 가집니다. 나중에 Flow Matching이 이를 어떻게 다루는지 봅니다.

    좌표계는 이미지와 같게 맞췄습니다. `x`는 오른쪽, `y`는 **아래쪽**으로 증가합니다.
    """)
    return


@app.cell
def _(DEVICE, torch):
    IMG_H, IMG_W = 24, 32
    COLOR_RGB = {"red": (1.0, 0.2, 0.2), "blue": (0.2, 0.45, 1.0)}
    EXPERT_SPEED = 0.05  # 시연 궤적의 step당 최대 이동 거리
    ENV_MAX_STEP = 0.08  # 환경이 허용하는 step당 최대 이동 거리
    SUCCESS_RADIUS = 0.05

    INSTRUCTIONS = {
        "red": ["go to the red dot", "move to red", "red"],
        "blue": ["go to the blue dot", "move to blue", "blue"],
        "any": ["go to any dot", "any dot"],
    }
    KIND_PROBS = {"red": 0.4, "blue": 0.4, "any": 0.2}

    def sample_scenes(batch_size, device=DEVICE, min_gap=0.35):
        """빨간 점·파란 점 위치를 [0.1, 0.9]^2에서 뽑고, 서로 min_gap 이상 떨어뜨립니다."""
        red = torch.rand(batch_size, 2, device=device) * 0.8 + 0.1
        blue = torch.rand(batch_size, 2, device=device) * 0.8 + 0.1
        for _ in range(50):
            close = (red - blue).norm(dim=-1) < min_gap
            if not close.any():
                break
            blue[close] = torch.rand(int(close.sum()), 2, device=device) * 0.8 + 0.1
        return red, blue

    def render_images(red, blue, sigma=1.1):
        """두 점을 가우시안 blob으로 그린 [B, 3, H, W] 이미지 (값 범위 [0, 1])."""
        device = red.device
        rows = torch.arange(IMG_H, device=device, dtype=torch.float32)[:, None]
        cols = torch.arange(IMG_W, device=device, dtype=torch.float32)[None, :]
        img = torch.full((red.shape[0], 3, IMG_H, IMG_W), 0.08, device=device)
        for pos, name in ((red, "red"), (blue, "blue")):
            px = pos[:, 0, None, None] * (IMG_W - 1)
            py = pos[:, 1, None, None] * (IMG_H - 1)
            blob = torch.exp(-((cols - px) ** 2 + (rows - py) ** 2) / (2 * sigma**2))
            color = torch.tensor(COLOR_RGB[name], device=device)[None, :, None, None]
            img = img + color * blob[:, None]
        return img.clamp(0.0, 1.0)

    def expert_chunk(start, target, length, speed=EXPERT_SPEED):
        """start에서 target으로 직선 이동하는 다음 length step의 위치 [B, length, 2]."""
        pos, out = start.clone(), []
        for _ in range(length):
            delta = target - pos
            dist = delta.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            pos = pos + delta * (speed / dist).clamp(max=1.0)
            out.append(pos)
        return torch.stack(out, dim=1)

    def sample_kinds(batch_size):
        names = list(KIND_PROBS)
        probs = torch.tensor([KIND_PROBS[k] for k in names])
        idx = torch.multinomial(probs, batch_size, replacement=True)
        return [names[i] for i in idx.tolist()]

    def random_texts(kinds):
        return [INSTRUCTIONS[k][torch.randint(len(INSTRUCTIONS[k]), ()).item()] for k in kinds]

    def pick_targets(kinds, red, blue):
        """'any'는 두 점 중 하나를 동전 던지기로 고릅니다."""
        is_red = torch.tensor([k == "red" for k in kinds], device=red.device)
        is_any = torch.tensor([k == "any" for k in kinds], device=red.device)
        coin = torch.rand(len(kinds), device=red.device) < 0.5
        use_red = is_red | (is_any & coin)
        return torch.where(use_red[:, None], red, blue)

    def make_batch(batch_size, chunk_size, device=DEVICE):
        """학습용 원본 배치(정규화·토큰화 전). 데이터는 매번 새로 생성합니다."""
        red, blue = sample_scenes(batch_size, device)
        kinds = sample_kinds(batch_size)
        target = pick_targets(kinds, red, blue)
        start = torch.rand(batch_size, 2, device=device)
        near = torch.rand(batch_size, device=device) < 0.2  # 목표 근처 상태도 학습하도록
        start[near] = (target[near] + 0.08 * torch.randn_like(target[near])).clamp(0, 1)
        return {
            "images": render_images(red, blue),
            "state": start,
            "task": random_texts(kinds),
            "action": expert_chunk(start, target, chunk_size),
            "meta": {"red": red, "blue": blue, "target": target, "kind": kinds},
        }

    return (
        ENV_MAX_STEP,
        IMG_H,
        IMG_W,
        INSTRUCTIONS,
        SUCCESS_RADIUS,
        make_batch,
        render_images,
        sample_scenes,
    )


@app.cell
def _(make_batch, plt, torch):
    torch.manual_seed(1)
    _raw = make_batch(6, 16)
    _fig, _axes = plt.subplots(1, 6, figsize=(15, 2.8))
    for _i, _ax in enumerate(_axes):
        _ax.imshow(_raw["images"][_i].permute(1, 2, 0).cpu(), extent=[0, 1, 1, 0])
        _s = _raw["state"][_i].cpu()
        _a = _raw["action"][_i].cpu()
        _ax.plot(_a[:, 0], _a[:, 1], ".-", color="white", ms=3, lw=1)
        _ax.plot(_s[0], _s[1], "o", color="yellow", ms=6)
        _ax.set_title(repr(_raw["task"][_i]), fontsize=9)
        _ax.set_xticks([])
        _ax.set_yticks([])
    _fig.suptitle("노란 점 = 현재 상태, 흰 선 = 정답 행동 청크(16 step)", fontsize=10)
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. 전처리: 원본의 processor가 하는 일

    원본 `processor_smolvla.py`의 전처리 순서는 다음과 같습니다.

    1. 관측 이름 바꾸기, 배치 차원 추가
    2. **`NewLineTaskProcessorStep`**: 지시문 끝에 `"\n"`을 붙입니다.
    3. **`TokenizerProcessorStep`**: SmolVLM2 토크나이저로 토큰화합니다. `padding="longest"`, `padding_side="right"`, `max_length=48`.
    4. 장치 이동
    5. **정규화**: 상태·행동은 `MEAN_STD`(평균 0, 표준편차 1), 이미지는 `IDENTITY`(그대로).

    후처리는 행동을 다시 원래 단위로 되돌립니다(unnormalize).
    이 노트북은 단어 단위의 작은 토크나이저를 씁니다. 좌표 토큰 `<x0>…<y9>`는 4절 VLM 사전 학습의 답변용입니다.
    """)
    return


@app.cell
def _(DEVICE, make_batch, torch):
    NUM_BINS = 10
    VOCAB = (
        ["<pad>", "\n", "go", "to", "move", "the", "dot", "red", "blue", "any", "where", "is"]
        + [f"<x{i}>" for i in range(NUM_BINS)]
        + [f"<y{i}>" for i in range(NUM_BINS)]
    )
    TOKEN_ID = {tok: i for i, tok in enumerate(VOCAB)}

    def add_newline(texts):
        """NewLineTaskProcessorStep: 지시문이 '\\n'으로 끝나도록 맞춥니다."""
        return [t if t.endswith("\n") else t + "\n" for t in texts]

    def tokenize(texts, max_length=8, padding="longest"):
        """오른쪽 패딩 토크나이저. (토큰 [B, L], attention mask [B, L]) 반환."""
        ids = []
        for t in texts:
            toks = [TOKEN_ID[w] for w in t.split()]
            if t.endswith("\n"):
                toks.append(TOKEN_ID["\n"])
            ids.append(toks[:max_length])
        length = max(len(x) for x in ids) if padding == "longest" else max_length
        tokens = torch.full((len(ids), length), TOKEN_ID["<pad>"], dtype=torch.long)
        mask = torch.zeros((len(ids), length), dtype=torch.bool)
        for i, x in enumerate(ids):
            tokens[i, : len(x)] = torch.tensor(x)
            mask[i, : len(x)] = True
        return tokens, mask

    # MEAN_STD 정규화 통계는 데이터에서 계산합니다 (원본: dataset_stats).
    torch.manual_seed(123)
    _big = make_batch(20000, 16)
    STATS = {
        "state": {"mean": _big["state"].mean(0), "std": _big["state"].std(0)},
        "action": {"mean": _big["action"].reshape(-1, 2).mean(0), "std": _big["action"].reshape(-1, 2).std(0)},
    }
    del _big

    def preprocess(raw, tokenizer_max_length=8, device=DEVICE):
        """원본 배치 → 정책 입력 배치 (LeRobot 키 이름 사용)."""
        tokens, mask = tokenize(add_newline(raw["task"]), tokenizer_max_length, padding="longest")
        batch = {
            "observation.images.top": raw["images"].to(device),  # VISUAL: IDENTITY
            "observation.state": (raw["state"].to(device) - STATS["state"]["mean"]) / STATS["state"]["std"],
            "observation.language.tokens": tokens.to(device),
            "observation.language.attention_mask": mask.to(device),
        }
        if "action" in raw:
            batch["action"] = (raw["action"].to(device) - STATS["action"]["mean"]) / STATS["action"]["std"]
        return batch

    def postprocess(actions):
        """정규화된 행동 → 원래 좌표."""
        return actions * STATS["action"]["std"] + STATS["action"]["mean"]

    _tokens, _mask = tokenize(add_newline(["go to the red dot", "blue", "any dot"]))
    print("tokens:\n", _tokens)
    print("attention_mask:\n", _mask.int())
    print("state mean/std:", STATS["state"]["mean"].cpu().numpy().round(3), STATS["state"]["std"].cpu().numpy().round(3))
    return NUM_BINS, TOKEN_ID, VOCAB, add_newline, postprocess, preprocess, tokenize


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 이미지 전처리: `resize_with_pad`와 [-1, 1] 변환

    원본 `SmolVLAPolicy.prepare_images`는 이미지를

    1. 비율을 유지한 채 목표 크기(기본 512×512)로 줄이고, 남는 부분은 **왼쪽·위쪽**에 0으로 채웁니다 (`resize_with_pad`).
    2. 값 범위를 `[0, 1]` → `[-1, 1]`로 바꿉니다(SigLIP 비전 인코더의 입력 규약).

    없는 카메라는 `-1`로 채운 빈 이미지와 `mask=False`로 대신합니다(`empty_cameras`).
    여기서는 24×32 이미지를 32×32로 맞춥니다. 위쪽 8줄이 패딩됩니다.
    """)
    return


@app.cell
def _(F, torch):
    def resize_with_pad(img: torch.Tensor, height: int, width: int, *, pad_value: float) -> torch.Tensor:
        """원본 vla_utils.resize_with_pad (smolvla 규약: 왼쪽·위쪽 패딩)."""
        if img.ndim != 4:
            raise ValueError(f"(b,c,h,w) expected, but got {img.shape}")

        current_height, current_width = img.shape[2:]
        if current_height == height and current_width == width:
            return img

        ratio = max(current_width / width, current_height / height)
        resized_height = int(current_height / ratio)
        resized_width = int(current_width / ratio)
        resized_img = F.interpolate(img, size=(resized_height, resized_width), mode="bilinear", align_corners=False)

        pad_height = max(0, height - resized_height)
        pad_width = max(0, width - resized_width)
        padded_img = F.pad(resized_img, (pad_width, 0, pad_height, 0), value=pad_value)
        return padded_img

    def pad_vector(vector, new_dim):
        """원본 vla_utils.pad_vector: 마지막 차원을 new_dim까지 0으로 채웁니다."""
        if vector.shape[-1] >= new_dim:
            return vector
        return F.pad(vector, (0, new_dim - vector.shape[-1]))

    return pad_vector, resize_with_pad


@app.cell
def _(plt, render_images, resize_with_pad, sample_scenes, torch):
    torch.manual_seed(2)
    _img = render_images(*sample_scenes(1))
    _padded = resize_with_pad(_img, 32, 32, pad_value=0)
    _model_in = _padded * 2.0 - 1.0
    _fig, _axes = plt.subplots(1, 2, figsize=(7, 3))
    _axes[0].imshow(_img[0].permute(1, 2, 0).cpu())
    _axes[0].set_title(f"원본 {tuple(_img.shape[2:])}")
    _axes[1].imshow(_padded[0].permute(1, 2, 0).cpu())
    _axes[1].set_title(f"resize_with_pad {tuple(_padded.shape[2:])}\n모델 입력 범위 [{_model_in.min():.1f}, {_model_in.max():.1f}]")
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. 작은 SmolVLM 만들기

    SmolVLA의 VLM 백본은 **SmolVLM2-500M-Video-Instruct**입니다. 구성은 세 부분입니다.

    | 부분 | 실제 SmolVLM2-500M (참고값) | 이 노트북 |
    |---|---|---|
    | 비전 인코더 (SigLIP) | 512×512 입력, 16px 패치 → 1024 패치, 폭 768 | 32×32 입력, 4px 패치 → 64 패치, 폭 64 |
    | 커넥터 (픽셀 셔플 + Linear) | 배율 4 → **이미지당 64 토큰** | 배율 2 → **이미지당 16 토큰** |
    | 텍스트 디코더 (SmolLM2, Llama형) | 폭 960, 32층, 헤드 15 / KV 헤드 5, head_dim 64 | 폭 128, 6층, 헤드 4 / KV 헤드 2, head_dim 32 |

    **픽셀 셔플**은 이웃한 `s×s` 패치를 채널 방향으로 합쳐 토큰 수를 `s²`배 줄입니다.
    SmolVLA가 빠른 이유 중 하나가 이미지당 토큰이 적다는 점입니다.

    디코더 층은 일반 Llama 블록입니다: `RMSNorm → q/k/v/o (GQA, RoPE) → 잔차 → RMSNorm → SwiGLU MLP → 잔차`.
    다만 SmolVLA는 층을 하나씩 직접 돌리므로(5절), 여기서는 각 층에 **가중치만** 두고 순전파는 바깥 함수가 맡습니다.
    """)
    return


@app.cell
def _(F, SimpleNamespace, dataclass, nn, torch):
    @dataclass
    class TextConfig:
        hidden_size: int = 128
        intermediate_size: int = 384
        num_hidden_layers: int = 6
        num_attention_heads: int = 4
        num_key_value_heads: int = 2
        head_dim: int = 32
        attention_bias: bool = False
        rms_norm_eps: float = 1e-5
        vocab_size: int = 0

    @dataclass
    class VisionConfig:
        image_size: int = 32
        patch_size: int = 4
        hidden_size: int = 64
        num_hidden_layers: int = 2
        num_attention_heads: int = 4

    @dataclass
    class VLMConfig:
        text_config: TextConfig
        vision_config: VisionConfig
        scale_factor: int = 2  # 픽셀 셔플 배율

    class RMSNorm(nn.Module):
        def __init__(self, dim, eps=1e-5):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(dim))
            self.eps = eps

        def forward(self, x):
            dtype = x.dtype
            x = x.float()
            x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
            return self.weight * x.to(dtype)

    class LlamaAttentionWeights(nn.Module):
        """q/k/v/o 투영만 가진 어텐션 (순전파는 SmolVLMWithExpertModel이 수행)."""

        def __init__(self, cfg: TextConfig):
            super().__init__()
            self.head_dim = cfg.head_dim
            self.q_proj = nn.Linear(cfg.hidden_size, cfg.num_attention_heads * cfg.head_dim, bias=cfg.attention_bias)
            self.k_proj = nn.Linear(cfg.hidden_size, cfg.num_key_value_heads * cfg.head_dim, bias=cfg.attention_bias)
            self.v_proj = nn.Linear(cfg.hidden_size, cfg.num_key_value_heads * cfg.head_dim, bias=cfg.attention_bias)
            self.o_proj = nn.Linear(cfg.num_attention_heads * cfg.head_dim, cfg.hidden_size, bias=cfg.attention_bias)

    class LlamaMLP(nn.Module):
        def __init__(self, cfg: TextConfig):
            super().__init__()
            self.gate_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
            self.up_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
            self.down_proj = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)

        def forward(self, x):
            return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

    class DecoderLayer(nn.Module):
        def __init__(self, cfg: TextConfig):
            super().__init__()
            self.input_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
            self.self_attn = LlamaAttentionWeights(cfg)
            self.post_attention_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
            self.mlp = LlamaMLP(cfg)

    class TextModel(nn.Module):
        """Llama형 디코더 본체 (transformers의 LlamaModel에 해당)."""

        def __init__(self, cfg: TextConfig, with_embeddings=True):
            super().__init__()
            self.config = cfg
            self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size) if with_embeddings else None
            self.layers = nn.ModuleList([DecoderLayer(cfg) for _ in range(cfg.num_hidden_layers)])
            self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)

        def get_input_embeddings(self):
            return self.embed_tokens

    class SiglipVisionEncoder(nn.Module):
        """SigLIP 비전 인코더 축소판: 패치 임베딩 + 위치 임베딩 + 양방향 Transformer."""

        def __init__(self, cfg: VisionConfig):
            super().__init__()
            num_patches = (cfg.image_size // cfg.patch_size) ** 2
            self.patch_embedding = nn.Conv2d(3, cfg.hidden_size, cfg.patch_size, stride=cfg.patch_size)
            self.position_embedding = nn.Embedding(num_patches, cfg.hidden_size)
            self.layers = nn.ModuleList(
                [
                    nn.TransformerEncoderLayer(
                        cfg.hidden_size,
                        cfg.num_attention_heads,
                        4 * cfg.hidden_size,
                        dropout=0.0,
                        activation="gelu",
                        batch_first=True,
                        norm_first=True,
                    )
                    for _ in range(cfg.num_hidden_layers)
                ]
            )
            self.post_layernorm = nn.LayerNorm(cfg.hidden_size)

        @property
        def dtype(self):
            return self.patch_embedding.weight.dtype

        def forward(self, pixel_values, patch_attention_mask=None):
            x = self.patch_embedding(pixel_values).flatten(2).transpose(1, 2)  # [B, P, D]
            x = x + self.position_embedding.weight[None]
            for layer in self.layers:
                x = layer(x)
            return SimpleNamespace(last_hidden_state=self.post_layernorm(x))

    def pixel_shuffle(x, scale_factor=2):
        """Idefics3/SmolVLM 커넥터의 픽셀 셔플: [B, N, D] → [B, N/s², D·s²]."""
        bsz, seq, embed_dim = x.size()
        height = width = int(seq**0.5)
        x = x.view(bsz, height, width, embed_dim)
        x = x.view(bsz, height, int(width / scale_factor), embed_dim * scale_factor)
        x = x.permute(0, 2, 1, 3)
        x = x.reshape(bsz, int(width / scale_factor), int(height / scale_factor), embed_dim * (scale_factor**2))
        x = x.permute(0, 2, 1, 3)
        return x.reshape(bsz, int(seq / (scale_factor**2)), embed_dim * (scale_factor**2))

    class Connector(nn.Module):
        def __init__(self, cfg: VLMConfig):
            super().__init__()
            self.scale_factor = cfg.scale_factor
            in_dim = cfg.vision_config.hidden_size * cfg.scale_factor**2
            self.modality_projection = nn.Linear(in_dim, cfg.text_config.hidden_size, bias=False)

        def forward(self, image_hidden_states):
            return self.modality_projection(pixel_shuffle(image_hidden_states, self.scale_factor))

    class TinySmolVLM(nn.Module):
        """SmolVLMForConditionalGeneration 축소판. 원본과 같은 경로(vlm.model.text_model 등)를 사용합니다."""

        def __init__(self, cfg: VLMConfig):
            super().__init__()
            self.config = cfg
            self.model = nn.Module()
            self.model.vision_model = SiglipVisionEncoder(cfg.vision_config)
            self.model.connector = Connector(cfg)
            self.model.text_model = TextModel(cfg.text_config)
            self.lm_head = nn.Linear(cfg.text_config.hidden_size, cfg.text_config.vocab_size, bias=False)

        @property
        def device(self):
            return self.lm_head.weight.device

    return (
        RMSNorm,
        TextConfig,
        TextModel,
        TinySmolVLM,
        VLMConfig,
        VisionConfig,
        pixel_shuffle,
    )


@app.cell
def _(pixel_shuffle, torch):
    # 픽셀 셔플이 어떤 패치를 묶는지 확인: 4×4 패치 격자, 각 패치 값 = 패치 번호
    _x = torch.arange(16).float().view(1, 16, 1)
    _y = pixel_shuffle(_x, 2)
    print("입력 패치 번호 (4×4 격자):\n", _x.view(4, 4).long())
    print("셔플 후 토큰 4개, 각 토큰에 담긴 패치 번호:\n", _y[0].long())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 공통 부품: RoPE와 GQA 어텐션 (원본 코드)

    아래 `apply_rope`, `eager_attention_forward`는 원본 `smolvlm_with_expert.py`를 거의 그대로 옮긴 것입니다.

    - `apply_rope(x, positions)`: `x`는 `[B, L, H, D]`, 위치는 `[B, L]`. 위치 번호를 **직접** 받으므로
      패딩을 건너뛴 위치(`cumsum(pad_mask) - 1`)를 쓸 수 있습니다.
    - `eager_attention_forward`: K/V 헤드를 Q 헤드 수만큼 복제(GQA)하고, **불리언 마스크**(`True`=볼 수 있음)로 가립니다.
      float32로 계산합니다.
    """)
    return


@app.cell
def _(nn, torch):
    def apply_rope(x, positions, max_wavelength=10_000):
        """
        Applies RoPE positions [B, L] to x [B, L, H, D].
        """
        d_half = x.shape[-1] // 2
        device = x.device
        dtype = x.dtype
        x = x.to(torch.float32)

        freq_exponents = (2.0 / x.shape[-1]) * torch.arange(d_half, dtype=torch.float32, device=device)
        timescale = max_wavelength**freq_exponents
        radians = positions[..., None].to(torch.float32) / timescale[None, None, :].to(torch.float32)

        radians = radians[..., None, :]

        sin = torch.sin(radians)
        cos = torch.cos(radians)

        x1, x2 = x.split(d_half, dim=-1)
        res = torch.empty_like(x)
        res[..., :d_half] = x1 * cos - x2 * sin
        res[..., d_half:] = x2 * cos + x1 * sin

        return res.to(dtype)

    def get_intermediate_size(hidden_dim, ffn_dim_multiplier=4, multiple_of=256):
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        return hidden_dim

    def eager_attention_forward(
        attention_mask, batch_size, head_dim, query_states, key_states, value_states, num_att_heads, num_key_value_heads
    ):
        """원본 SmolVLMWithExpertModel.eager_attention_forward (헤드 수를 인자로 받도록만 바꿈)."""
        num_key_value_groups = num_att_heads // num_key_value_heads

        sequence_length = key_states.shape[1]

        key_states = key_states[:, :, :, None, :].expand(
            batch_size, sequence_length, num_key_value_heads, num_key_value_groups, head_dim
        )
        key_states = key_states.reshape(batch_size, sequence_length, num_key_value_heads * num_key_value_groups, head_dim)

        value_states = value_states[:, :, :, None, :].expand(
            batch_size, sequence_length, num_key_value_heads, num_key_value_groups, head_dim
        )
        value_states = value_states.reshape(
            batch_size, sequence_length, num_key_value_heads * num_key_value_groups, head_dim
        )

        query_states = query_states.to(dtype=torch.float32)
        key_states = key_states.to(dtype=torch.float32)

        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)

        att_weights = torch.matmul(query_states, key_states.transpose(2, 3))
        att_weights *= head_dim**-0.5

        att_weights = att_weights.to(dtype=torch.float32)
        big_neg = torch.finfo(att_weights.dtype).min
        masked_att_weights = torch.where(attention_mask[:, None, :, :], att_weights, big_neg)
        probs = nn.functional.softmax(masked_att_weights, dim=-1)
        probs = probs.to(dtype=value_states.dtype)

        att_output = torch.matmul(probs, value_states.permute(0, 2, 1, 3))

        att_output = att_output.permute(0, 2, 1, 3)
        att_output = att_output.reshape(batch_size, -1, num_key_value_heads * num_key_value_groups * head_dim)

        return att_output

    def make_att_2d_masks(pad_masks, att_masks):
        """원본 vla_utils.make_att_2d_masks (big_vision에서 유래).

        att_masks가 1인 토큰은 새 '블록'을 시작합니다. 각 토큰은 누적합(cumsum)이 자기 이하인
        토큰만 볼 수 있습니다.
          [0 0 0 1 1 1] → 앞 3개는 서로 양방향, 뒤 3개는 인과(causal)
        """
        if att_masks.ndim != 2:
            raise ValueError(att_masks.ndim)
        if pad_masks.ndim != 2:
            raise ValueError(pad_masks.ndim)

        cumsum = torch.cumsum(att_masks, dim=1)
        att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
        pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
        return att_2d_masks & pad_2d_masks

    return apply_rope, eager_attention_forward, get_intermediate_size, make_att_2d_masks


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. VLM 사전 학습 (VQA)

    SmolVLA는 **이미 사전 학습된 VLM**을 가져와 쓰고, 기본 설정(`train_expert_only=True`)에서는 VLM을 **고정**합니다.
    행동 전문가는 VLM 각 층의 K/V를 읽기만 합니다. 그러려면 VLM의 내부 표현에 "점이 어디 있는지", "어떤 색을 묻는지"가
    이미 담겨 있어야 합니다.

    그래서 작은 VLM을 먼저 간단한 VQA로 학습합니다.

    ```text
    [이미지 토큰 16개] where is red \n  →  <x3> <y7>
    ```

    좌표를 10칸으로 나눈 토큰을 다음 토큰 예측(교차 엔트로피)으로 맞힙니다. GPU 기준 약 1분 걸립니다.
    손실이 한동안 정체될 수 있습니다. 색 단어(`red`/`blue`)와 이미지 속 색을 연결하는 데 시간이 걸리기 때문입니다.
    이미지 토큰끼리는 양방향, 텍스트는 인과 어텐션입니다(`att_masks`: 이미지 0, 텍스트 1).
    """)
    return


@app.cell
def _(
    DEVICE,
    NUM_BINS,
    TOKEN_ID,
    TextConfig,
    TinySmolVLM,
    VLMConfig,
    VOCAB,
    VisionConfig,
    apply_rope,
    eager_attention_forward,
    make_att_2d_masks,
    math,
    render_images,
    resize_with_pad,
    sample_scenes,
    torch,
):
    def init_weights(module, std=0.02):
        """transformers 모델의 _init_weights와 같은 방식 (initializer_range=0.02).

        nn.Embedding 기본 초기화는 N(0, 1)이라 embed_prefix의 √dim 배율과 만나면 값이 너무 커져 학습이 멈춥니다.
        """
        for m in module.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=std)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=std)

    def build_tiny_vlm(seed=0):
        torch.manual_seed(seed)
        cfg = VLMConfig(text_config=TextConfig(vocab_size=len(VOCAB)), vision_config=VisionConfig())
        vlm = TinySmolVLM(cfg)
        init_weights(vlm)
        return vlm.to(DEVICE)

    def prepare_image_for_vlm(images, size=(32, 32)):
        """SmolVLAPolicy.prepare_images와 같은 변환: resize_with_pad → [-1, 1]."""
        return resize_with_pad(images, size[1], size[0], pad_value=0) * 2.0 - 1.0

    def vlm_text_forward(vlm, embs, att_2d_masks, position_ids):
        """VQA 사전 학습용 일반 디코더 순전파 (5절의 forward_attn_layer와 같은 계산)."""
        tm, tcfg = vlm.model.text_model, vlm.config.text_config
        h = embs
        bsize = h.shape[0]
        for layer in tm.layers:
            x = layer.input_layernorm(h)
            shape = (*x.shape[:-1], -1, tcfg.head_dim)
            q = apply_rope(layer.self_attn.q_proj(x).view(shape), position_ids)
            k = apply_rope(layer.self_attn.k_proj(x).view(shape), position_ids)
            v = layer.self_attn.v_proj(x).view(shape)
            att = eager_attention_forward(
                att_2d_masks, bsize, tcfg.head_dim, q, k, v, tcfg.num_attention_heads, tcfg.num_key_value_heads
            )
            h = h + layer.self_attn.o_proj(att)
            h = h + layer.mlp(layer.post_attention_layernorm(h))
        return tm.norm(h)

    def coord_to_bin(v):
        return (v * NUM_BINS).long().clamp(0, NUM_BINS - 1)

    def make_vqa_batch(batch_size):
        red, blue = sample_scenes(batch_size)
        ask_red = torch.rand(batch_size, device=DEVICE) < 0.5
        target = torch.where(ask_red[:, None], red, blue)
        xb, yb = coord_to_bin(target[:, 0]), coord_to_bin(target[:, 1])
        color_tok = torch.where(ask_red, TOKEN_ID["red"], TOKEN_ID["blue"])
        x_tok = TOKEN_ID["<x0>"] + xb
        y_tok = TOKEN_ID["<y0>"] + yb
        ones = torch.ones_like(color_tok)
        text = torch.stack([ones * TOKEN_ID["where"], ones * TOKEN_ID["is"], color_tok, ones * TOKEN_ID["\n"], x_tok], 1)
        labels = torch.stack([x_tok, y_tok], 1)  # '\n' 위치 → x, x 위치 → y
        return render_images(red, blue), text, labels

    def vqa_logits(vlm, images, text):
        mdl = vlm.model
        img = mdl.connector(mdl.vision_model(pixel_values=prepare_image_for_vlm(images)).last_hidden_state)
        img = img * math.sqrt(img.shape[-1])
        txt = mdl.text_model.embed_tokens(text) * math.sqrt(img.shape[-1])
        embs = torch.cat([img, txt], 1)
        bsize, n_img, n_txt = embs.shape[0], img.shape[1], txt.shape[1]
        att = torch.tensor([0] * n_img + [1] * n_txt, device=DEVICE)[None].expand(bsize, -1)
        pad = torch.ones(bsize, n_img + n_txt, dtype=torch.bool, device=DEVICE)
        pos = torch.cumsum(pad, 1) - 1
        h = vlm_text_forward(vlm, embs, make_att_2d_masks(pad, att), pos)
        return vlm.lm_head(h[:, -2:])  # 마지막 두 위치('\n', <x>)의 다음 토큰 예측

    return build_tiny_vlm, make_vqa_batch, prepare_image_for_vlm, vqa_logits


@app.cell
def _(mo):
    train_button = mo.ui.run_button(label="VQA 사전 학습·정책 학습 실행")
    train_button
    return (train_button,)


@app.cell
def _(BATCH_SIZE, F, NUM_BINS, TOKEN_ID, VLM_PRETRAIN_STEPS, build_tiny_vlm, make_vqa_batch, mo, time, torch, train_button, vqa_logits):
    mo.stop(not train_button.value, mo.md("4절 버튼을 누르면 학습합니다."))
    pretrained_vlm = build_tiny_vlm(seed=0)
    _opt = torch.optim.AdamW(pretrained_vlm.parameters(), lr=1e-3, weight_decay=0.01)
    _sched = torch.optim.lr_scheduler.OneCycleLR(_opt, max_lr=1e-3, total_steps=VLM_PRETRAIN_STEPS, pct_start=0.1)
    vlm_losses = []
    _t0 = time.time()
    torch.manual_seed(10)
    pretrained_vlm.train()
    for _step in range(VLM_PRETRAIN_STEPS):
        _img, _text, _labels = make_vqa_batch(BATCH_SIZE)
        _logits = vqa_logits(pretrained_vlm, _img, _text)
        _loss = F.cross_entropy(_logits.reshape(-1, _logits.shape[-1]), _labels.reshape(-1))
        _opt.zero_grad(set_to_none=True)
        _loss.backward()
        torch.nn.utils.clip_grad_norm_(pretrained_vlm.parameters(), 1.0)
        _opt.step()
        _sched.step()
        vlm_losses.append(_loss.item())
    pretrained_vlm.eval()

    with torch.no_grad():
        torch.manual_seed(999)
        _img, _text, _labels = make_vqa_batch(2000)
        _pred = vqa_logits(pretrained_vlm, _img, _text).argmax(-1)
        _exact = (_pred == _labels).float().mean(0)
        # 어휘 ID 경계를 넘은 x/y 토큰은 서로 이웃 좌표가 아닙니다.
        _starts = torch.tensor([TOKEN_ID["<x0>"], TOKEN_ID["<y0>"]], device=_pred.device)
        _valid_axis = (_pred >= _starts) & (_pred < _starts + NUM_BINS)
        _near = (_valid_axis & ((_pred - _labels).abs() <= 1)).float().mean(0)
    vlm_pretrain_metrics = {
        "seconds": time.time() - _t0,
        "x_exact": _exact[0].item(),
        "y_exact": _exact[1].item(),
        "x_within1": _near[0].item(),
        "y_within1": _near[1].item(),
    }
    print({k: round(v, 3) for k, v in vlm_pretrain_metrics.items()})
    return pretrained_vlm, vlm_losses, vlm_pretrain_metrics


@app.cell
def _(mo, np, plt, vlm_losses, vlm_pretrain_metrics):
    _fig, _ax = plt.subplots(figsize=(6, 2.6))
    _window = min(20, len(vlm_losses))
    _ax.plot(np.arange(_window - 1, len(vlm_losses)), np.convolve(vlm_losses, np.ones(_window) / _window, mode="valid"))
    _ax.set_xlabel("step")
    _ax.set_ylabel("VQA loss (이동 평균)")
    _ax.set_yscale("log")
    _fig.tight_layout()
    _m = vlm_pretrain_metrics
    mo.vstack(
        [
            _fig,
            mo.md(
                f"검증 2000개: 좌표 칸 정확도 x **{_m['x_exact']:.1%}**, y **{_m['y_exact']:.1%}** "
                f"(±1칸 허용 시 x {_m['x_within1']:.1%}, y {_m['y_within1']:.1%}). "
                f"각 축의 10개 좌표 토큰을 균등 선택하는 기준은 10%이며, 전체 어휘 무작위 예측과 다릅니다. "
                f"y는 정답 x 토큰을 입력한 조건부 정확도(teacher forcing)로, x·y를 연속 생성한 정확도는 아닙니다. 학습 시간 {_m['seconds']:.0f}초."
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. SmolVLA 핵심: `SmolVLMWithExpertModel`

    SmolVLA의 중심 클래스입니다. 두 개의 Transformer를 **층 단위로 나란히** 돌립니다.

    - **VLM** (`vlm.model.text_model`): 접두부(prefix) = `[이미지 토큰, 언어 토큰, 상태 토큰]`을 처리합니다.
    - **행동 전문가** (`lm_expert`): 접미부(suffix) = `[노이즈 섞인 행동 토큰 × chunk_size]`를 처리합니다.

    원본 설정에서 중요한 값:

    | 설정 | 기본값 | 의미 |
    |---|---|---|
    | `num_vlm_layers` | 16 | VLM의 **앞쪽 16층만** 사용 (SmolVLM2-500M은 32층). 나머지는 버림. 텍스트 층 계산량 감소(전체 지연이 절반이라는 뜻은 아님) |
    | `expert_width_multiplier` | 0.75 | 전문가 폭 = VLM 폭 × 0.75 (960 → 720) |
    | `num_expert_layers` | -1 | 전문가 층 수. -1이면 VLM과 같은 수 |
    | `attention_mode` | `"cross_attn"` | 교차 어텐션 층을 사용 |
    | `self_attn_every_n_layers` | 2 | 짝수 층(0, 2, 4…)은 **자기 어텐션**, 나머지는 **교차 어텐션** |
    | `train_expert_only` | True | VLM 전체를 고정 |

    두 종류의 층이 어떻게 다른지가 핵심입니다.

    - **자기 어텐션 층** (`forward_attn_layer`): 접두부와 접미부의 Q/K/V를 **시퀀스 방향으로 이어 붙여** 한 번의 어텐션을 합니다.
      각 모델은 자기 투영(q/k/v)을 쓰지만 어텐션은 공유합니다. 헤드 수·`head_dim`이 같아야 하므로 전문가의 `q_proj`는 `720 → 960`입니다.
    - **교차 어텐션 층** (`forward_cross_attn_layer`): VLM은 접두부끼리만 어텐션합니다. 전문가는 **자기 Q**로
      **VLM의 K/V를 다시 투영한 값**만 봅니다. 그래서 이 층의 전문가 `k_proj`, `v_proj` 입력 차원은 VLM의 K/V 차원으로 바뀝니다.
      이 층에서 행동 토큰끼리는 서로 보지 않습니다.

    아래 코드는 원본을 옮긴 것입니다. 바꾼 점은 주석에 `[변경]`으로 표시했습니다.
    """)
    return


@app.cell
def _(SimpleNamespace, torch):
    class PrefixKVCache:
        """[변경] transformers.DynamicCache 대신 쓰는 최소 KV 캐시.

        - 접두부 prefill 때: 층마다 (RoPE 적용 후) K/V를 [B, H, L, D]로 저장합니다.
        - 노이즈 제거 때: 저장된 접두부 K/V 뒤에 접미부 K/V를 붙여 반환하되, **저장하지는 않습니다**.
          원본은 DynamicCache에 붙여 넣은 뒤 step마다 접두부 길이로 잘라(crop) 되돌립니다. 결과는 같습니다.
        """

        def __init__(self):
            self.layers = []

        def __len__(self):
            return len(self.layers)

        def update(self, keys, values, layer_idx):
            if layer_idx == len(self.layers):
                self.layers.append(SimpleNamespace(keys=keys, values=values))
                return keys, values
            cached = self.layers[layer_idx]
            return torch.cat([cached.keys, keys], dim=2), torch.cat([cached.values, values], dim=2)

    return (PrefixKVCache,)


@app.cell
def _(
    PrefixKVCache,
    TextModel,
    apply_rope,
    copy,
    eager_attention_forward,
    get_intermediate_size,
    nn,
    torch,
):
    class SmolVLMWithExpertModel(nn.Module):
        def __init__(
            self,
            vlm,  # [변경] 원본: model_id로 SmolVLM2를 불러옴. 여기서는 사전 학습한 TinySmolVLM을 받음
            train_expert_only: bool = True,
            freeze_vision_encoder: bool = False,
            attention_mode: str = "self_attn",
            num_expert_layers: int = -1,
            num_vlm_layers: int = -1,
            self_attn_every_n_layers: int = -1,
            expert_width_multiplier: float = 0.5,
        ):
            super().__init__()
            self.vlm = copy.deepcopy(vlm)  # [변경] 사전 학습 VLM 원본은 그대로 두고 복사본을 자름
            config = self.vlm.config
            if num_vlm_layers > 0:
                print(f"Reducing the number of VLM layers to {num_vlm_layers} ...")
                self.get_vlm_model().text_model.layers = self.get_vlm_model().text_model.layers[:num_vlm_layers]
            self.num_vlm_layers = len(self.get_vlm_model().text_model.layers)
            self.config = config
            # Smaller lm expert
            lm_expert_config = copy.deepcopy(config.text_config)
            hidden_size = lm_expert_config.hidden_size
            lm_expert_config.hidden_size = int(hidden_size * expert_width_multiplier)
            lm_expert_config.intermediate_size = get_intermediate_size(int(hidden_size * expert_width_multiplier))
            lm_expert_config.num_hidden_layers = self.num_vlm_layers
            if num_expert_layers > 0:
                assert len(self.get_vlm_model().text_model.layers) % num_expert_layers == 0, (
                    f"Number of layers in the VLM {len(self.get_vlm_model().text_model.layers)} are not multiple of num_expert_layers {num_expert_layers}"
                )
                lm_expert_config.num_hidden_layers = num_expert_layers
            # [변경] 원본: AutoModel.from_config(lm_expert_config). 임베딩은 원본도 None으로 지움
            self.lm_expert = TextModel(lm_expert_config, with_embeddings=False)

            self.num_expert_layers = len(self.lm_expert.layers)
            self.self_attn_every_n_layers = self_attn_every_n_layers
            if "cross" in attention_mode:
                # Reshape qkv projections to have the same input dimension as the vlm
                for layer_idx in range(len(self.lm_expert.layers)):
                    if self.self_attn_every_n_layers > 0 and layer_idx % self.self_attn_every_n_layers == 0:
                        continue
                    self.lm_expert.layers[layer_idx].self_attn.k_proj = nn.Linear(
                        config.text_config.num_key_value_heads * config.text_config.head_dim,
                        lm_expert_config.num_key_value_heads * lm_expert_config.head_dim,
                        bias=lm_expert_config.attention_bias,
                    )
                    self.lm_expert.layers[layer_idx].self_attn.v_proj = nn.Linear(
                        config.text_config.num_key_value_heads * config.text_config.head_dim,
                        lm_expert_config.num_key_value_heads * lm_expert_config.head_dim,
                        bias=lm_expert_config.attention_bias,
                    )

            self.num_attention_heads = self.config.text_config.num_attention_heads
            self.num_key_value_heads = self.config.text_config.num_key_value_heads

            self.freeze_vision_encoder = freeze_vision_encoder
            self.train_expert_only = train_expert_only
            self.attention_mode = attention_mode
            self.expert_hidden_size = lm_expert_config.hidden_size
            self.expert_config = lm_expert_config
            self.set_requires_grad()

        def get_vlm_model(self):
            return self.vlm.model

        def set_requires_grad(self):
            if self.freeze_vision_encoder:
                self.get_vlm_model().vision_model.eval()
                for params in self.get_vlm_model().vision_model.parameters():
                    params.requires_grad = False
            if self.train_expert_only:
                self.vlm.eval()
                for params in self.vlm.parameters():
                    params.requires_grad = False
            else:
                # 마지막 VLM 층의 출력은 손실에 쓰이지 않으므로(전문가는 K/V만 읽음) 고정
                last_layers = [self.num_vlm_layers - 1]
                if self.num_vlm_layers != self.num_expert_layers and self.num_vlm_layers % self.num_expert_layers == 0:
                    last_layers.append(self.num_vlm_layers - 2)
                frozen_layers = ["lm_head", "text_model.norm.weight"]
                for layer in last_layers:
                    frozen_layers.append(f"text_model.layers.{layer}.")
                for name, params in self.vlm.named_parameters():
                    if any(k in name for k in frozen_layers):
                        params.requires_grad = False

        def train(self, mode: bool = True):
            super().train(mode)
            if self.freeze_vision_encoder:
                self.get_vlm_model().vision_model.eval()
            if self.train_expert_only:
                self.vlm.eval()
            return self

        def embed_image(self, image: torch.Tensor):
            image_hidden_states = (
                self.get_vlm_model()
                .vision_model(pixel_values=image.to(dtype=self.get_vlm_model().vision_model.dtype))
                .last_hidden_state
            )
            # Modality projection & resampling
            image_hidden_states = self.get_vlm_model().connector(image_hidden_states)
            return image_hidden_states

        def embed_language_tokens(self, tokens: torch.Tensor):
            return self.get_vlm_model().text_model.get_input_embeddings()(tokens)

        def forward_attn_layer(
            self,
            model_layers,
            inputs_embeds,
            layer_idx,
            position_ids,
            attention_mask,
            batch_size,
            head_dim,
            use_cache: bool = True,
            past_key_values=None,
        ):
            query_states = []
            key_states = []
            value_states = []
            for i, hidden_states in enumerate(inputs_embeds):
                layer = model_layers[i][layer_idx]
                if hidden_states is None or layer is None:
                    continue
                hidden_states = layer.input_layernorm(hidden_states)

                input_shape = hidden_states.shape[:-1]
                hidden_shape = (*input_shape, -1, layer.self_attn.head_dim)

                hidden_states = hidden_states.to(dtype=layer.self_attn.q_proj.weight.dtype)
                query_state = layer.self_attn.q_proj(hidden_states).view(hidden_shape)
                key_state = layer.self_attn.k_proj(hidden_states).view(hidden_shape)
                value_state = layer.self_attn.v_proj(hidden_states).view(hidden_shape)

                query_states.append(query_state)
                key_states.append(key_state)
                value_states.append(value_state)

            # B,L,H,D: 접두부와 접미부를 시퀀스 방향으로 이어 붙임
            query_states = torch.cat(query_states, dim=1)
            key_states = torch.cat(key_states, dim=1)
            value_states = torch.cat(value_states, dim=1)
            seq_len = query_states.shape[1]
            if seq_len < position_ids.shape[1]:
                _position_ids = position_ids[:, :seq_len]
                _attention_mask = attention_mask[:, :seq_len, :seq_len]
            else:
                _position_ids = position_ids
                _attention_mask = attention_mask

            query_states = apply_rope(query_states, _position_ids)
            key_states = apply_rope(key_states, _position_ids)

            if use_cache:
                if past_key_values is None:
                    raise ValueError("`use_cache=True` requires a `past_key_values` cache.")
                # 캐시는 [B, H, L, D], 이 모듈은 [B, L, H, D]를 씀
                key_states, value_states = past_key_values.update(
                    key_states.transpose(1, 2), value_states.transpose(1, 2), layer_idx
                )
                key_states = key_states.transpose(1, 2)
                value_states = value_states.transpose(1, 2)

            att_output = self.eager_attention_forward(
                _attention_mask, batch_size, head_dim, query_states, key_states, value_states
            )
            return [att_output], past_key_values

        def forward_cross_attn_layer(
            self,
            model_layers,
            inputs_embeds,
            layer_idx,
            position_ids,
            attention_mask,
            batch_size,
            head_dim,
            use_cache: bool = True,
            past_key_values=None,
        ):
            att_outputs = []
            assert len(inputs_embeds) == 2 or (use_cache and past_key_values is not None)

            if len(inputs_embeds) == 2 and not past_key_values:
                # Prefix attention: VLM은 접두부끼리만 어텐션
                seq_len = inputs_embeds[0].shape[1]
                position_id, expert_position_id = position_ids[:, :seq_len], position_ids[:, seq_len:]
                prefix_attention_mask = attention_mask[:, :seq_len, :seq_len]

                layer = model_layers[0][layer_idx]

                hidden_states = layer.input_layernorm(inputs_embeds[0])

                input_shape = hidden_states.shape[:-1]
                hidden_shape = (*input_shape, -1, layer.self_attn.head_dim)

                hidden_states = hidden_states.to(dtype=layer.self_attn.q_proj.weight.dtype)
                query_state = layer.self_attn.q_proj(hidden_states).view(hidden_shape)
                key_state = layer.self_attn.k_proj(hidden_states).view(hidden_shape)
                value_states = layer.self_attn.v_proj(hidden_states).view(hidden_shape)

                query_states = apply_rope(query_state, position_id)
                key_states = apply_rope(key_state, position_id)

                att_output = self.eager_attention_forward(
                    prefix_attention_mask, batch_size, head_dim, query_states, key_states, value_states
                )
                att_outputs.append(att_output)
            else:
                expert_position_id = position_ids

            if use_cache and past_key_values is not None:
                # 교차 어텐션 층은 캐시를 읽기만 함 (prefill 때 forward_attn_layer가 저장)
                key_states = past_key_values.layers[layer_idx].keys.transpose(1, 2)
                value_states = past_key_values.layers[layer_idx].values.transpose(1, 2)

            # Expert
            expert_layer = model_layers[1][layer_idx]
            if expert_layer is not None:
                expert_hidden_states = expert_layer.input_layernorm(inputs_embeds[1])

                expert_input_shape = expert_hidden_states.shape[:-1]
                expert_hidden_shape = (*expert_input_shape, -1, expert_layer.self_attn.head_dim)

                expert_hidden_states = expert_hidden_states.to(dtype=expert_layer.self_attn.q_proj.weight.dtype)
                expert_query_state = expert_layer.self_attn.q_proj(expert_hidden_states).view(expert_hidden_shape)

                # VLM의 K/V(이미 RoPE 적용됨)를 전문가의 k_proj/v_proj로 다시 투영
                _key_states = key_states.to(dtype=expert_layer.self_attn.k_proj.weight.dtype).reshape(
                    *key_states.shape[:2], -1
                )
                expert_key_states = expert_layer.self_attn.k_proj(_key_states).view(
                    *_key_states.shape[:-1], -1, expert_layer.self_attn.head_dim
                )

                _value_states = value_states.to(dtype=expert_layer.self_attn.v_proj.weight.dtype).reshape(
                    *value_states.shape[:2], -1
                )
                expert_value_states = expert_layer.self_attn.v_proj(_value_states).view(
                    *_value_states.shape[:-1], -1, expert_layer.self_attn.head_dim
                )

                expert_position_id = expert_position_id - torch.min(expert_position_id, dim=1, keepdim=True).values  # 0부터
                expert_attention_mask = attention_mask[:, -inputs_embeds[1].shape[1] :, : expert_key_states.shape[1] :]

                expert_query_states = apply_rope(expert_query_state, expert_position_id)

                att_output = self.eager_attention_forward(
                    expert_attention_mask,
                    batch_size,
                    head_dim,
                    expert_query_states,
                    expert_key_states,
                    expert_value_states,
                )
                att_outputs.append(att_output)
            else:
                att_outputs.append(None)

            return att_outputs, past_key_values

        def get_model_layers(self, models: list) -> list:
            vlm_layers = []
            expert_layers = []
            multiple_of = self.num_vlm_layers // self.num_expert_layers
            for i in range(self.num_vlm_layers):
                if multiple_of > 0 and i > 0 and i % multiple_of != 0:
                    expert_layer = None
                else:
                    expert_layer_index = i // multiple_of if multiple_of > 0 else i
                    expert_layer = models[1].layers[expert_layer_index]
                vlm_layers.append(models[0].layers[i])
                expert_layers.append(expert_layer)
            return [vlm_layers, expert_layers]

        def is_self_attn_layer(self, layer_idx, fill_kv_cache=False):
            """[추가] forward()가 이 층에서 어떤 경로를 쓰는지 (원본 forward의 조건식과 동일)."""
            return (
                fill_kv_cache
                or "cross" not in self.attention_mode
                or (self.self_attn_every_n_layers > 0 and layer_idx % self.self_attn_every_n_layers == 0)
            )

        def forward(
            self,
            attention_mask=None,
            position_ids=None,
            past_key_values=None,
            inputs_embeds=None,
            use_cache=None,
        ):
            use_cache = bool(use_cache)
            models = [self.get_vlm_model().text_model, self.lm_expert]
            model_layers = self.get_model_layers(models)
            for hidden_states in inputs_embeds:
                if hidden_states is None:
                    continue
                batch_size = hidden_states.shape[0]

            # Prefix prefill: 캐시가 없으면 만들고, 모든 층에서 접두부 자기 어텐션을 수행하며 채움
            fill_kv_cache = use_cache and past_key_values is None
            if fill_kv_cache:
                past_key_values = PrefixKVCache()  # [변경] 원본: DynamicCache()

            num_layers = self.num_vlm_layers
            head_dim = self.vlm.config.text_config.head_dim
            for layer_idx in range(num_layers):
                if self.is_self_attn_layer(layer_idx, fill_kv_cache):
                    att_outputs, past_key_values = self.forward_attn_layer(
                        model_layers, inputs_embeds, layer_idx, position_ids, attention_mask, batch_size, head_dim,
                        use_cache=use_cache, past_key_values=past_key_values,
                    )
                else:
                    att_outputs, past_key_values = self.forward_cross_attn_layer(
                        model_layers, inputs_embeds, layer_idx, position_ids, attention_mask, batch_size, head_dim,
                        use_cache=use_cache, past_key_values=past_key_values,
                    )
                outputs_embeds = []
                start = 0
                for i, hidden_states in enumerate(inputs_embeds):
                    layer = model_layers[i][layer_idx]
                    att_output = att_outputs[i] if i < len(att_outputs) else att_outputs[0]  # in case of self_attn
                    if hidden_states is not None:
                        if layer is None:
                            outputs_embeds.append(hidden_states)
                            continue
                        end = start + hidden_states.shape[1]

                        if att_output.dtype != layer.self_attn.o_proj.weight.dtype:
                            att_output = att_output.to(layer.self_attn.o_proj.weight.dtype)
                        att_out = att_output[:, start:end]
                        out_emb = layer.self_attn.o_proj(att_out)

                        out_emb += hidden_states
                        after_first_residual = out_emb.clone()

                        out_emb = layer.post_attention_layernorm(out_emb)
                        out_emb = layer.mlp(out_emb)

                        out_emb += after_first_residual

                        outputs_embeds.append(out_emb)

                        start = end if len(att_outputs) == 1 else 0
                    else:
                        outputs_embeds.append(None)

                inputs_embeds = outputs_embeds

            # final norm
            outputs_embeds = []
            for i, hidden_states in enumerate(inputs_embeds):
                if hidden_states is not None:
                    outputs_embeds.append(models[i].norm(hidden_states))
                else:
                    outputs_embeds.append(None)
            return outputs_embeds, past_key_values

        def eager_attention_forward(self, attention_mask, batch_size, head_dim, query_states, key_states, value_states):
            return eager_attention_forward(
                attention_mask, batch_size, head_dim, query_states, key_states, value_states,
                self.num_attention_heads, self.num_key_value_heads,
            )

    return (SmolVLMWithExpertModel,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. `VLAFlowMatching`: 입력 임베딩과 Flow Matching

    `VLAFlowMatching`은 `SmolVLMWithExpertModel` 앞뒤에 입출력 층을 붙입니다.

    | 모듈 | 입력 → 출력 | 역할 |
    |---|---|---|
    | `state_proj` | `max_state_dim` → VLM 폭 | 상태를 **VLM 쪽 토큰 1개**로 |
    | `action_in_proj` | `max_action_dim` → 전문가 폭 | 노이즈 섞인 행동을 토큰으로 |
    | `action_time_mlp_in/out` | `[행동 임베딩; 시간 임베딩]` → 전문가 폭 | 시간 `t`를 행동 토큰마다 섞음 (SiLU MLP) |
    | `action_out_proj` | 전문가 폭 → `max_action_dim` | 속도 `v_t` 예측 |

    `embed_prefix`는 이미지와 언어 임베딩에 $\sqrt{d}$($d$: 임베딩 폭)를 곱해 크기를 맞춥니다. 상태와 행동은 `max_state_dim`, `max_action_dim`(원본 32)까지
    0으로 채워(`pad_vector`) 여러 로봇의 차원을 하나의 모델로 받습니다. 이 노트북은 8로 둡니다(실제 차원 2).

    **어텐션 블록 규칙** (`att_masks`, 1이면 새 블록 시작):

    ```text
    이미지 0…0 | 언어 0…0 | 상태 1 | 행동 1 1 1 … 1
    ```

    - 이미지·언어: 한 블록 → 서로 양방향. 상태와 행동은 못 봄
    - 상태: 새 블록 → 이미지·언어·상태를 봄
    - 행동: 토큰마다 새 블록 → 접두부 전체 + **자기 자신과 이전 행동 토큰만** 봄 (행동끼리 **인과** 어텐션)

    π0은 행동 토큰끼리 양방향이지만, SmolVLA는 행동 토큰에 `1`을 하나씩 붙여 인과 어텐션을 씁니다.
    """)
    return


@app.cell
def _(F, SmolVLMWithExpertModel, make_att_2d_masks, math, nn, torch):
    def create_sinusoidal_pos_embedding(time, dimension, min_period, max_period, device="cpu"):
        """원본 vla_utils.create_sinusoidal_pos_embedding: 스칼라 시간 t → [sin, cos] 임베딩."""
        if dimension % 2 != 0:
            raise ValueError(f"dimension ({dimension}) must be divisible by 2")
        if time.ndim not in (1, 2):
            raise ValueError("The time tensor must have shape (batch_size,) or (batch_size, action_horizon).")

        dtype = torch.float32 if torch.device(device).type == "mps" else torch.float64
        fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
        period = min_period * (max_period / min_period) ** fraction

        scaling_factor = 1.0 / period * 2 * math.pi
        sin_input = time[..., None] * scaling_factor
        return torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=-1)

    def sample_time_beta(bsize, device, *, alpha, beta, scale=1.0, offset=0.0):
        """원본 flow_matching.sample_time_beta: Beta(alpha, beta) * scale + offset."""
        dist = torch.distributions.Beta(torch.tensor(alpha), torch.tensor(beta), validate_args=False)
        time_beta = dist.sample((bsize,)).to(device)
        return (time_beta * scale + offset).to(dtype=torch.float32, device=device)

    def make_flow_matching_inputs(actions, noise, time):
        """원본 flow_matching.make_flow_matching_inputs (NOISE_AT_ONE 규약만).

        t=1이 순수 노이즈, t=0이 정답 행동입니다.
          x_t = t * noise + (1 - t) * actions
          u_t = noise - actions          (x_t를 t로 미분한 값 = 목표 속도)
        """
        expanded_time = time[:, None, None]
        x_t = expanded_time * noise + (1 - expanded_time) * actions
        velocity_target = noise - actions
        return x_t, velocity_target, time

    def euler_integrate(denoise_fn, noise, num_steps, trace=None):
        """원본 flow_matching.euler_integrate의 핵심 (NOISE_AT_ONE, RTC·time_grid 생략).

        t = 1 → 0으로 dt = -1/num_steps씩 이동:  x_{t+dt} = x_t + dt * v_t
        trace에 리스트를 주면 각 step의 x_t를 기록합니다. [추가]
        """
        bsize = noise.shape[0]
        dt = -1.0 / num_steps
        x_t = noise
        for step in range(num_steps):
            time_tensor = torch.full((bsize,), 1.0 + step * dt, dtype=torch.float32, device=noise.device)
            v_t = denoise_fn(x_t, time_tensor)
            x_t = x_t + dt * v_t
            if trace is not None:
                trace.append(x_t.detach().clone())
        return x_t

    class VLAFlowMatching(nn.Module):
        """VLM 접두부와 행동 전문가로 속도를 학습하고, 캐시와 Euler 적분으로 행동을 생성합니다."""

        def __init__(self, config, vlm):
            super().__init__()
            self.config = config

            self.vlm_with_expert = SmolVLMWithExpertModel(
                vlm=vlm,
                freeze_vision_encoder=self.config.freeze_vision_encoder,
                train_expert_only=self.config.train_expert_only,
                attention_mode=self.config.attention_mode,
                num_expert_layers=self.config.num_expert_layers,
                num_vlm_layers=self.config.num_vlm_layers,
                self_attn_every_n_layers=self.config.self_attn_every_n_layers,
                expert_width_multiplier=self.config.expert_width_multiplier,
            )
            vlm_hidden = self.vlm_with_expert.config.text_config.hidden_size
            expert_hidden = self.vlm_with_expert.expert_hidden_size
            self.state_proj = nn.Linear(self.config.max_state_dim, vlm_hidden)
            self.action_in_proj = nn.Linear(self.config.max_action_dim, expert_hidden)
            self.action_out_proj = nn.Linear(expert_hidden, self.config.max_action_dim)

            self.action_time_mlp_in = nn.Linear(expert_hidden * 2, expert_hidden)
            self.action_time_mlp_out = nn.Linear(expert_hidden, expert_hidden)

            self.set_requires_grad()
            # [생략] add_image_special_tokens(이미지 앞뒤 특수 토큰), prefix_length 패딩, RTC, torch.compile

        def set_requires_grad(self):
            for params in self.state_proj.parameters():
                params.requires_grad = self.config.train_state_proj

        def sample_noise(self, shape, device):
            return torch.normal(mean=0.0, std=1.0, size=shape, dtype=torch.float32, device=device)

        def sample_time(self, bsize, device):
            return sample_time_beta(bsize, device, alpha=1.5, beta=1.0, scale=0.999, offset=0.001)

        def embed_prefix(self, images, img_masks, lang_tokens, lang_masks, state):
            """이미지(SigLIP) + 언어 토큰 + 상태를 VLM 입력 시퀀스로."""
            embs = []
            pad_masks = []
            att_masks = []
            for img, img_mask in zip(images, img_masks, strict=False):
                img_emb = self.vlm_with_expert.embed_image(img)

                # Normalize image embeddings
                img_emb_dim = img_emb.shape[-1]
                img_emb = img_emb * torch.tensor(img_emb_dim**0.5, dtype=img_emb.dtype, device=img_emb.device)

                bsize, num_img_embs = img_emb.shape[:2]
                img_mask = img_mask[:, None].expand(bsize, num_img_embs)

                embs.append(img_emb)
                pad_masks.append(img_mask)
                att_masks += [0] * (num_img_embs)

            lang_emb = self.vlm_with_expert.embed_language_tokens(lang_tokens)
            # Normalize language embeddings
            lang_emb_dim = lang_emb.shape[-1]
            lang_emb = lang_emb * math.sqrt(lang_emb_dim)

            embs.append(lang_emb)
            pad_masks.append(lang_masks)

            num_lang_embs = lang_emb.shape[1]
            att_masks += [0] * num_lang_embs

            state_emb = self.state_proj(state)
            state_emb = state_emb[:, None, :] if state_emb.ndim == 2 else state_emb
            embs.append(state_emb)
            bsize = state_emb.shape[0]
            device = state_emb.device

            states_seq_len = state_emb.shape[1]
            state_mask = torch.ones(bsize, states_seq_len, dtype=torch.bool, device=device)
            pad_masks.append(state_mask)

            # Set attention masks so that image and language inputs do not attend to state or actions
            att_masks += [1] * (states_seq_len)
            embs = torch.cat(embs, dim=1)
            pad_masks = torch.cat(pad_masks, dim=1)
            att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)
            att_masks = att_masks[None, :].expand(bsize, -1)

            return embs, pad_masks, att_masks

        def embed_suffix(self, noisy_actions, timestep):
            """노이즈 섞인 행동 + 시간 t를 행동 전문가 입력으로."""
            embs = []
            pad_masks = []
            att_masks = []

            # Fuse timestep + action information using an MLP
            action_emb = self.action_in_proj(noisy_actions)
            device = action_emb.device
            bsize = action_emb.shape[0]
            dtype = action_emb.dtype
            # Embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
            time_emb = create_sinusoidal_pos_embedding(
                timestep,
                self.vlm_with_expert.expert_hidden_size,
                self.config.min_period,
                self.config.max_period,
                device=device,
            )
            time_emb = time_emb.type(dtype=dtype)

            time_emb = time_emb[:, None, :].expand_as(action_emb)
            action_time_emb = torch.cat([action_emb, time_emb], dim=2)

            action_time_emb = self.action_time_mlp_in(action_time_emb)
            action_time_emb = F.silu(action_time_emb)  # swish == silu
            action_time_emb = self.action_time_mlp_out(action_time_emb)

            embs.append(action_time_emb)

            bsize, action_time_dim = action_time_emb.shape[:2]
            action_time_mask = torch.ones(bsize, action_time_dim, dtype=torch.bool, device=device)
            pad_masks.append(action_time_mask)

            # Set attention masks so that image, language and state inputs do not attend to action tokens
            att_masks += [1] * self.config.chunk_size
            embs = torch.cat(embs, dim=1)
            pad_masks = torch.cat(pad_masks, dim=1)
            att_masks = torch.tensor(att_masks, dtype=embs.dtype, device=embs.device)
            att_masks = att_masks[None, :].expand(bsize, len(att_masks))
            return embs, pad_masks, att_masks

        def forward(self, images, img_masks, lang_tokens, lang_masks, state, actions, noise=None, time=None):
            """학습용 순전파: 요소별 손실 [B, chunk_size, max_action_dim]."""
            if noise is None:
                noise = self.sample_noise(actions.shape, actions.device)
            if time is None:
                time = self.sample_time(actions.shape[0], actions.device)

            x_t, u_t, _ = make_flow_matching_inputs(actions, noise, time)
            prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
                images, img_masks, lang_tokens, lang_masks, state=state
            )
            suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(x_t, time)

            pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
            att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)

            att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
            position_ids = torch.cumsum(pad_masks, dim=1) - 1
            (_, suffix_out), _ = self.vlm_with_expert.forward(
                attention_mask=att_2d_masks,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, suffix_embs],
                use_cache=False,
            )
            suffix_out = suffix_out[:, -self.config.chunk_size :]
            suffix_out = suffix_out.to(dtype=torch.float32)
            v_t = self.action_out_proj(suffix_out)
            losses = F.mse_loss(u_t, v_t, reduction="none")
            return losses

        def sample_actions(self, images, img_masks, lang_tokens, lang_masks, state, noise=None, trace=None):
            """추론: 접두부를 한 번 계산해 KV 캐시에 담고, num_steps번 Euler 적분."""
            bsize = state.shape[0]
            device = state.device

            if noise is None:
                actions_shape = (bsize, self.config.chunk_size, self.config.max_action_dim)
                noise = self.sample_noise(actions_shape, device)
            else:
                noise = noise.to(device=device, dtype=torch.float32)

            prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
                images, img_masks, lang_tokens, lang_masks, state=state
            )
            prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
            prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
            # Compute image and language key value cache
            _, past_key_values = self.vlm_with_expert.forward(
                attention_mask=prefix_att_2d_masks,
                position_ids=prefix_position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, None],
                use_cache=self.config.use_cache,
            )

            return euler_integrate(
                lambda input_x_t, current_timestep: self.denoise_step(
                    x_t=input_x_t,
                    prefix_pad_masks=prefix_pad_masks,
                    past_key_values=past_key_values,
                    timestep=current_timestep,
                ),
                noise,
                self.config.num_steps,
                trace=trace,
            )

        def denoise_step(self, prefix_pad_masks, past_key_values, x_t, timestep):
            """한 번의 노이즈 제거 step: 행동 전문가만 실행하고 접두부는 캐시에서 읽음."""
            suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(x_t, timestep)

            suffix_len = suffix_pad_masks.shape[1]
            batch_size = prefix_pad_masks.shape[0]
            prefix_len = prefix_pad_masks.shape[1]
            prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)

            suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)

            full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)
            prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
            position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

            outputs_embeds, _ = self.vlm_with_expert.forward(
                attention_mask=full_att_2d_masks,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=[None, suffix_embs],
                use_cache=self.config.use_cache,
            )
            # [변경] 원본은 여기서 캐시를 접두부 길이로 crop. PrefixKVCache는 접미부를 저장하지 않으므로 불필요
            suffix_out = outputs_embeds[1]
            suffix_out = suffix_out[:, -self.config.chunk_size :]
            suffix_out = suffix_out.to(dtype=torch.float32)
            v_t = self.action_out_proj(suffix_out)
            return v_t

    return (
        VLAFlowMatching,
        create_sinusoidal_pos_embedding,
        euler_integrate,
        make_flow_matching_inputs,
        sample_time_beta,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7. `SmolVLAPolicy`: 정책 래퍼와 행동 큐

    `SmolVLAPolicy`는 LeRobot 학습·실행 루프와 `VLAFlowMatching` 사이를 잇습니다.

    - `prepare_images` / `prepare_state` / `prepare_action`: 이미지 크기 맞춤·[-1, 1] 변환, 상태·행동 0 채우기
    - `forward(batch)`: 손실 계산. `action_is_pad`(에피소드 끝 넘어간 행동)는 손실에서 뺍니다.
      패딩한 행동 차원(`action_dim` 이후)도 손실에서 잘라냅니다.
    - `select_action(batch)`: **행동 큐**가 비었을 때만 청크를 새로 만들고, 앞의 `n_action_steps`개를 큐에 넣어
      한 step에 하나씩 꺼냅니다. 기본값은 `chunk_size = n_action_steps = 50`입니다.

    이 노트북 설정은 `chunk_size=16`, `n_action_steps=8`입니다. 16개를 예측하고 8개만 실행한 뒤 다시 예측합니다.
    """)
    return


@app.cell
def _(POLICY_TRAIN_STEPS, VLAFlowMatching, dataclass, deque, nn, pad_vector, resize_with_pad, torch):
    ACTION = "action"
    OBS_STATE = "observation.state"
    OBS_LANGUAGE_TOKENS = "observation.language.tokens"
    OBS_LANGUAGE_ATTENTION_MASK = "observation.language.attention_mask"

    @dataclass
    class SmolVLAToyConfig:
        """원본 SmolVLAConfig에서 이 실습에 필요한 필드만 남긴 설정. 괄호 안은 원본 기본값."""

        n_obs_steps: int = 1
        chunk_size: int = 16  # (50)
        n_action_steps: int = 8  # (50)
        max_state_dim: int = 8  # (32)
        max_action_dim: int = 8  # (32)
        resize_imgs_with_padding: tuple = (32, 32)  # (512, 512)
        empty_cameras: int = 0
        tokenizer_max_length: int = 8  # (48)
        num_steps: int = 10  # Euler 적분 횟수 (10)
        use_cache: bool = True
        freeze_vision_encoder: bool = True
        train_expert_only: bool = True
        train_state_proj: bool = True
        optimizer_lr: float = 3e-4  # (1e-4)
        optimizer_betas: tuple = (0.9, 0.95)
        optimizer_eps: float = 1e-8
        optimizer_weight_decay: float = 1e-10
        optimizer_grad_clip_norm: float = 10.0
        scheduler_warmup_steps: int = 200  # (1000)
        scheduler_decay_steps: int = POLICY_TRAIN_STEPS  # (30000)
        scheduler_decay_lr: float = 2.5e-6
        attention_mode: str = "cross_attn"
        num_expert_layers: int = -1
        num_vlm_layers: int = 4  # (16) 사전 학습 VLM 6층 중 앞 4층만 사용
        self_attn_every_n_layers: int = 2
        expert_width_multiplier: float = 0.75
        min_period: float = 4e-3
        max_period: float = 4.0
        image_features: tuple = ("observation.images.top",)
        action_dim: int = 2  # 원본: config.action_feature.shape[0]

    class SmolVLAToyPolicy(nn.Module):
        """원본 SmolVLAPolicy 축소판 (RTC, aloha 변환, PEFT 관련 코드 생략)."""

        def __init__(self, config, vlm):
            super().__init__()
            self.config = config
            self.model = VLAFlowMatching(config, vlm)
            self.reset()

        def reset(self):
            """환경을 리셋할 때마다 호출합니다."""
            self._queues = {ACTION: deque(maxlen=self.config.n_action_steps)}

        def prepare_images(self, batch):
            images, img_masks = [], []
            present_img_keys = [key for key in self.config.image_features if key in batch]
            missing_img_keys = [key for key in self.config.image_features if key not in batch]
            if len(present_img_keys) == 0:
                raise ValueError("All image features are missing from the batch.")
            for key in present_img_keys:
                img = batch[key][:, -1, :, :, :] if batch[key].ndim == 5 else batch[key]
                if img.dtype == torch.uint8:
                    img = img.to(torch.float32) / 255.0
                if self.config.resize_imgs_with_padding is not None:
                    img = resize_with_pad(
                        img, self.config.resize_imgs_with_padding[1], self.config.resize_imgs_with_padding[0], pad_value=0
                    )
                # Normalize from range [0,1] to [-1,1] as expected by siglip
                img = img * 2.0 - 1.0
                bsize = img.shape[0]
                if f"{key}_padding_mask" in batch:
                    mask = batch[f"{key}_padding_mask"].bool()
                else:
                    mask = torch.ones(bsize, dtype=torch.bool, device=img.device)
                images.append(img)
                img_masks.append(mask)
            # 없는 카메라는 -1로 채운 이미지 + mask=False
            for num_empty_cameras in range(len(missing_img_keys)):
                if num_empty_cameras >= self.config.empty_cameras:
                    break
                images.append(torch.ones_like(img) * -1)
                img_masks.append(torch.zeros_like(mask))
            return images, img_masks

        def prepare_state(self, batch):
            state = batch[OBS_STATE][:, -1, :] if batch[OBS_STATE].ndim > 2 else batch[OBS_STATE]
            return pad_vector(state, self.config.max_state_dim)

        def prepare_action(self, batch):
            return pad_vector(batch[ACTION], self.config.max_action_dim)

        def forward(self, batch, noise=None, time=None):
            images, img_masks = self.prepare_images(batch)
            state = self.prepare_state(batch)
            lang_tokens = batch[OBS_LANGUAGE_TOKENS]
            lang_masks = batch[OBS_LANGUAGE_ATTENTION_MASK]
            actions = self.prepare_action(batch)
            actions_is_pad = batch.get("action_is_pad")
            loss_dict = {}
            losses = self.model.forward(images, img_masks, lang_tokens, lang_masks, state, actions, noise, time)
            loss_dict["losses_with_padded_dims"] = losses.mean().item()
            losses = losses[:, :, : self.config.action_dim]  # 패딩한 행동 차원 제거
            if actions_is_pad is not None:
                losses = losses * (~actions_is_pad).unsqueeze(-1)
                num_valid = ((~actions_is_pad).sum() * losses.shape[-1]).clamp_min(1)
                loss = losses.sum() / num_valid
            else:
                loss = losses.mean()
            loss_dict["loss"] = loss.item()
            return loss, loss_dict

        def _get_action_chunk(self, batch, noise=None, trace=None):
            images, img_masks = self.prepare_images(batch)
            state = self.prepare_state(batch)
            actions = self.model.sample_actions(
                images, img_masks, batch[OBS_LANGUAGE_TOKENS], batch[OBS_LANGUAGE_ATTENTION_MASK], state,
                noise=noise, trace=trace,
            )
            return actions[:, :, : self.config.action_dim]  # Unpad actions

        @torch.no_grad()
        def predict_action_chunk(self, batch, noise=None, trace=None):
            self.eval()
            return self._get_action_chunk(batch, noise=noise, trace=trace)

        @torch.no_grad()
        def select_action(self, batch, noise=None):
            self.eval()
            if len(self._queues[ACTION]) == 0:
                actions = self._get_action_chunk(batch, noise=noise)
                # [B, n_action_steps, A] → 큐는 [n_action_steps]개의 [B, A]
                self._queues[ACTION].extend(actions.transpose(0, 1)[: self.config.n_action_steps])
            return self._queues[ACTION].popleft()

    return SmolVLAToyConfig, SmolVLAToyPolicy


@app.cell
def _(DEVICE, SmolVLAToyConfig, SmolVLAToyPolicy, pretrained_vlm, torch):
    # 구조 확인용 (아직 학습하지 않은) 정책
    torch.manual_seed(0)
    demo_policy = SmolVLAToyPolicy(SmolVLAToyConfig(), pretrained_vlm).to(DEVICE)
    return (demo_policy,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 층 배치 확인

    `self_attn_every_n_layers=2`이므로 0, 2번 층은 자기 어텐션, 1, 3번 층은 교차 어텐션입니다.
    교차 어텐션 층의 전문가 `k_proj` 입력 차원이 **VLM의 K/V 차원(64)** 으로 바뀐 것을 확인하세요.
    자기 어텐션 층은 전문가 자신의 폭(96)을 입력으로 받습니다.
    """)
    return


@app.cell
def _(SmolVLMWithExpertModel, demo_policy, mo, pretrained_vlm):
    def layer_table(vwe):
        models = [vwe.get_vlm_model().text_model, vwe.lm_expert]
        vlm_layers, expert_layers = vwe.get_model_layers(models)
        rows = []
        for i in range(vwe.num_vlm_layers):
            exp = expert_layers[i]
            exp_idx = next((j for j, l in enumerate(vwe.lm_expert.layers) if l is exp), None)
            rows.append(
                {
                    "층": i,
                    "학습 시 경로": "자기 어텐션" if vwe.is_self_attn_layer(i) else "교차 어텐션",
                    "전문가 층": "-" if exp is None else exp_idx,
                    "전문가 q_proj": "-" if exp is None else f"{exp.self_attn.q_proj.in_features}→{exp.self_attn.q_proj.out_features}",
                    "전문가 k_proj": "-" if exp is None else f"{exp.self_attn.k_proj.in_features}→{exp.self_attn.k_proj.out_features}",
                    "전문가 o_proj": "-" if exp is None else f"{exp.self_attn.o_proj.in_features}→{exp.self_attn.o_proj.out_features}",
                }
            )
        return rows

    _vwe = demo_policy.model.vlm_with_expert
    # 비교: 전문가 층을 절반(2개)만 두는 설정
    _half = SmolVLMWithExpertModel(
        pretrained_vlm, num_vlm_layers=4, num_expert_layers=2, attention_mode="cross_attn",
        self_attn_every_n_layers=2, expert_width_multiplier=0.75,
    )
    mo.vstack(
        [
            mo.md(f"**기본 설정** (VLM {_vwe.num_vlm_layers}층, 전문가 {_vwe.num_expert_layers}층, 전문가 폭 {_vwe.expert_hidden_size})"),
            mo.ui.table(layer_table(_vwe), selection=None),
            mo.md(
                "**`num_expert_layers=2`** 일 때: 전문가 층이 VLM 2층마다 하나씩 붙고, 나머지 층에서 전문가는 그냥 통과합니다. "
                "자기/교차 배치와 겹치면 교차 어텐션 층에 전문가가 하나도 없게 될 수도 있습니다."
            ),
            mo.ui.table(layer_table(_half), selection=None),
        ]
    )
    return


@app.cell
def _(demo_policy, mo):
    def count(module, trainable=None):
        return sum(p.numel() for p in module.parameters() if trainable is None or p.requires_grad == trainable)

    _m = demo_policy.model
    _vlm = _m.vlm_with_expert.get_vlm_model()
    _parts = {
        "비전 인코더": _vlm.vision_model,
        "커넥터": _vlm.connector,
        "VLM 텍스트 디코더 (앞 4층 + 임베딩)": _vlm.text_model,
        "lm_head (사용 안 함)": _m.vlm_with_expert.vlm.lm_head,
        "행동 전문가": _m.vlm_with_expert.lm_expert,
        "state_proj": _m.state_proj,
        "action_in/out_proj + time MLP": [_m.action_in_proj, _m.action_out_proj, _m.action_time_mlp_in, _m.action_time_mlp_out],
    }
    _rows = []
    for _name, _mod in _parts.items():
        _mods = _mod if isinstance(_mod, list) else [_mod]
        _total = sum(count(x) for x in _mods)
        _train = sum(count(x, True) for x in _mods)
        _rows.append({"부분": _name, "파라미터": f"{_total:,}", "학습 대상": f"{_train:,}"})
    _rows.append({"부분": "합계", "파라미터": f"{count(_m):,}", "학습 대상": f"{count(_m, True):,}"})
    mo.vstack(
        [
            mo.ui.table(_rows, selection=None),
            mo.md(
                "`train_expert_only=True`이면 VLM 전체가 고정되고 **행동 전문가, state_proj, 행동 입출력 층만** 학습합니다. "
                "참고로 SmolVLA 논문 기준 실제 모델은 약 4.5억 파라미터이고, 그중 행동 전문가는 약 1억입니다."
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 어텐션 마스크 그려 보기

    지시문 길이가 다른 두 샘플(`"go to the red dot"`, `"red"`)을 한 배치로 만들면, 짧은 쪽에 언어 패딩이 생깁니다.
    아래 그림은 **짧은 샘플**의 전체 2D 마스크입니다(행 = 질의 토큰, 열 = 볼 수 있는 키 토큰, 흰색 = 허용).

    - 언어 패딩 열·행은 모두 막힙니다.
    - 위치 번호는 `cumsum(pad_mask) - 1`이라 패딩 자리는 번호를 건너뜁니다.
    - 행동 블록(오른쪽 아래)은 아래 삼각형: 인과 어텐션입니다.
    """)
    return


@app.cell
def _(demo_policy, make_att_2d_masks, plt, preprocess, render_images, sample_scenes, torch):
    torch.manual_seed(3)
    _red, _blue = sample_scenes(2)
    _raw = {"images": render_images(_red, _blue), "state": torch.rand(2, 2), "task": ["go to the red dot", "red"]}
    _batch = preprocess(_raw)
    _m = demo_policy.model
    with torch.no_grad():
        _images, _img_masks = demo_policy.prepare_images(_batch)
        _pe, _pp, _pa = _m.embed_prefix(
            _images, _img_masks, _batch["observation.language.tokens"], _batch["observation.language.attention_mask"],
            demo_policy.prepare_state(_batch),
        )
        _se, _sp, _sa = _m.embed_suffix(torch.zeros(2, 16, 8, device=_pe.device), torch.full((2,), 0.5, device=_pe.device))
    _pad = torch.cat([_pp, _sp], 1)
    _att = torch.cat([_pa, _sa.bool()], 1)
    _mask2d = make_att_2d_masks(_pad, _att)[1].cpu()
    _n_img, _n_lang = 16, _batch["observation.language.tokens"].shape[1]
    _bounds = [_n_img, _n_img + _n_lang, _n_img + _n_lang + 1]

    _fig, _ax = plt.subplots(figsize=(6, 6))
    _ax.imshow(_mask2d, cmap="gray", interpolation="nearest")
    for _b in _bounds:
        _ax.axhline(_b - 0.5, color="tab:orange", lw=1)
        _ax.axvline(_b - 0.5, color="tab:orange", lw=1)
    _ticks = [_n_img / 2, _n_img + _n_lang / 2, _bounds[1], _bounds[2] + 8]
    _ax.set_xticks(_ticks, ["이미지", "언어", "상태", "행동"])
    _ax.set_yticks(_ticks, ["이미지", "언어", "상태", "행동"])
    _ax.set_title("샘플 2 ('red\\n')의 2D 어텐션 마스크")
    _fig.tight_layout()
    print("att_masks (샘플 2):", _att[1].int().tolist())
    print("pad_masks (샘플 2):", _pad[1].int().tolist())
    print("position_ids (샘플 2):", (torch.cumsum(_pad[1], 0) - 1).tolist())
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 8. Flow Matching 한눈에 보기

    SmolVLA는 행동을 직접 회귀하지 않고 **노이즈에서 행동으로 가는 속도장**을 배웁니다(π0과 같은 `NOISE_AT_ONE` 규약).

    정답 행동 $a$, 노이즈 $\epsilon\sim\mathcal N(0,I)$와 시간 $t$로 학습 입력과 목표 속도를 만듭니다.

    $$x_t=t\epsilon+(1-t)a,\qquad u_t=\frac{d x_t}{dt}=\epsilon-a,$$
    $$\mathcal L=\mathbb E\!\left[\frac{1}{BHA}\sum_{b,h,j}\left(v_\theta(x_t,t,o)_{b,h,j}-u_{t,b,h,j}\right)^2\right].$$

    $o$는 관측, $B$는 배치 크기, $H$는 청크 길이, $A=2$는 실제 행동 차원입니다.
    `make_flow_matching_inputs`가 $x_t,u_t$를, `action_out_proj`가 예측 속도를 계산합니다.
    정책 래퍼는 패딩한 행동 차원을 제외합니다. 시간 패딩이 있다면 유효 원소 수로 나눕니다.
    시간은 $t=0.999b+0.001$, $b\sim\operatorname{Beta}(1.5,1)$로 뽑아 노이즈가 많은 쪽을 더 자주 학습합니다.

    추론에서는 $x_1=\epsilon$에서 $\Delta t=-1/N$으로 $N$번 Euler 적분합니다.
    $$x_{t+\Delta t}=x_t+\Delta t\,v_\theta(x_t,t,o),\qquad N=10.$$
    예를 들어 $a=0.2$, $\epsilon=0.8$, $t=0.5$이면 $x_t=0.5$, $u_t=0.6$입니다.
    정확한 속도를 안다면 $\Delta t=-0.1$마다 0.06씩 줄어들어 행동 0.2에 도달합니다.

    `"any dot"`처럼 정답이 두 개일 때, 직접 회귀(MSE)는 두 정답의 **평균**(두 점 사이)으로 끌려갑니다.
    Flow Matching은 노이즈에 따라 둘 중 한 정답으로 가는 경로를 만들 수 있습니다.
    """)
    return


@app.cell
def _(
    create_sinusoidal_pos_embedding,
    make_batch,
    make_flow_matching_inputs,
    plt,
    sample_time_beta,
    torch,
):
    _t = sample_time_beta(100_000, "cpu", alpha=1.5, beta=1.0, scale=0.999, offset=0.001)
    torch.manual_seed(5)
    _raw = make_batch(1, 16, device="cpu")
    _a = _raw["action"]
    _noise = torch.randn_like(_a) * 0.3 + _a.mean(1, keepdim=True)  # 그림용: 원 좌표에서 보기 좋게 축소한 노이즈
    _fig, _axes = plt.subplots(1, 3, figsize=(15, 3.6))
    _axes[0].hist(_t.numpy(), bins=50, density=True)
    _axes[0].set_title("학습 시간 t 분포: Beta(1.5, 1) (t=1이 노이즈)")
    _axes[0].set_xlabel("t")

    _axes[1].imshow(_raw["images"][0].permute(1, 2, 0), extent=[0, 1, 1, 0])
    for _tv, _c in zip([1.0, 0.75, 0.5, 0.25, 0.0], plt.cm.viridis(torch.linspace(0, 1, 5).numpy())):
        _xt, _, _ = make_flow_matching_inputs(_a, _noise, torch.tensor([_tv]))
        _axes[1].plot(_xt[0, :, 0], _xt[0, :, 1], ".-", color=_c, label=f"t={_tv}")
    _axes[1].legend(fontsize=7, loc="lower right")
    _axes[1].set_title(f"x_t = t·ε + (1−t)·a  ({_raw['task'][0]!r})")

    _emb = create_sinusoidal_pos_embedding(torch.linspace(0, 1, 200), 96, 4e-3, 4.0)
    _axes[2].imshow(_emb.T.numpy(), aspect="auto", cmap="RdBu", extent=[0, 1, 96, 0])
    _axes[2].set_xlabel("t")
    _axes[2].set_ylabel("임베딩 차원")
    _axes[2].set_title("시간 t의 sin/cos 임베딩 (period 0.004~4)")
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 9. 학습: 행동 전문가만 학습

    원본 학습 설정을 따릅니다: AdamW(`betas=(0.9, 0.95)`, `weight_decay=1e-10`), gradient clip 10,
    warmup 후 cosine 감쇠. 작은 과제라서 학습률(3e-4)과 step 수만 바꿨습니다.

    학습 대상은 `requires_grad=True`인 파라미터뿐입니다(VLM 고정). GPU 기준 약 1.5분 걸립니다.
    동결은 VLM **가중치**의 gradient와 갱신을 막습니다. VLM을 `torch.no_grad()`로 감싸지는 않으므로
    손실에서 VLM의 K/V를 거쳐 입력 상태 임베딩으로 미분이 전달되고, `state_proj`는 계속 학습됩니다.
    정책 학습에서는 VLM·비전 인코더를 `eval()`로 유지하지만 행동 전문가와 입출력 투영·시간 MLP는 갱신합니다.
    이 작은 합성 과제의 성공률을 실제 로봇이나 공개 SmolVLA 체크포인트의 성능으로 일반화하지 마세요.
    난수·학습 예산에 따라 경향도 달라질 수 있어 중요한 비교는 여러 시드로 확인합니다.
    """)
    return


@app.cell
def _(DEVICE, SmolVLAToyPolicy, make_batch, math, preprocess, time, torch):
    def cosine_decay_with_warmup(config):
        """lerobot CosineDecayWithWarmupSchedulerConfig와 같은 모양의 학습률 배율 (근사)."""
        alpha = config.scheduler_decay_lr / config.optimizer_lr

        def lr_lambda(step):
            if step < config.scheduler_warmup_steps:
                return (step + 1) / (config.scheduler_warmup_steps + 1)
            progress = min(1.0, (step - config.scheduler_warmup_steps) / max(1, config.scheduler_decay_steps - config.scheduler_warmup_steps))
            return (1 - alpha) * 0.5 * (1 + math.cos(math.pi * progress)) + alpha

        return lr_lambda

    def train_policy(vlm, config, steps, batch_size, seed=0):
        torch.manual_seed(seed)
        policy = SmolVLAToyPolicy(config, vlm).to(DEVICE)
        params = [p for p in policy.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(
            params, lr=config.optimizer_lr, betas=config.optimizer_betas, eps=config.optimizer_eps,
            weight_decay=config.optimizer_weight_decay,
        )
        sched = torch.optim.lr_scheduler.LambdaLR(opt, cosine_decay_with_warmup(config))
        losses = []
        t0 = time.time()
        policy.train()
        for _ in range(steps):
            batch = preprocess(make_batch(batch_size, config.chunk_size), config.tokenizer_max_length)
            loss, _ = policy(batch)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, config.optimizer_grad_clip_norm)
            opt.step()
            sched.step()
            losses.append(loss.item())
        policy.eval()
        return policy, {"losses": losses, "seconds": time.time() - t0, "trainable_params": sum(p.numel() for p in params)}

    return (train_policy,)


@app.cell
def _(BATCH_SIZE, POLICY_TRAIN_STEPS, SmolVLAToyConfig, pretrained_vlm, train_policy):
    policy_config = SmolVLAToyConfig()
    policy, train_log = train_policy(pretrained_vlm, policy_config, POLICY_TRAIN_STEPS, BATCH_SIZE, seed=0)
    print(f"학습 {train_log['seconds']:.0f}초, 학습 파라미터 {train_log['trainable_params']:,}개, 마지막 {min(200, len(train_log['losses']))} step 평균 손실 {sum(train_log['losses'][-200:]) / len(train_log['losses'][-200:]):.4f}")
    return policy, train_log


@app.cell
def _(np, plt, train_log):
    _l = np.array(train_log["losses"])
    _fig, _ax = plt.subplots(figsize=(6, 2.6))
    _window = min(50, len(_l))
    _ax.plot(np.arange(_window - 1, len(_l)), np.convolve(_l, np.ones(_window) / _window, mode="valid"))
    _ax.set_yscale("log")
    _ax.set_xlabel("step")
    _ax.set_ylabel("flow matching loss (이동 평균)")
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 10. 추론: KV 캐시와 노이즈 제거 과정

    `sample_actions`는 다음 순서로 동작합니다.

    1. **prefill**: 접두부(이미지·언어·상태)를 VLM에 한 번 통과시키며 모든 층의 K/V를 캐시에 저장합니다.
       이때는 모든 층이 자기 어텐션 경로(`forward_attn_layer`)를 탑니다.
    2. **노이즈 제거 10회**: 매번 행동 전문가만 실행합니다. 자기 어텐션 층은 `[캐시된 접두부 K/V; 행동 K/V]`를 보고,
       교차 어텐션 층은 캐시된 접두부 K/V만 봅니다.

    VLM이 10번이 아니라 1번만 돌므로 빠릅니다. 접두부는 행동 토큰을 보지 않으므로(마스크) 캐시를 써도 결과가 같습니다.
    아래에서 캐시 없이 매 step 전체를 다시 계산한 결과와 비교합니다.
    """)
    return


@app.cell
def _(
    DEVICE,
    euler_integrate,
    make_att_2d_masks,
    mo,
    policy,
    preprocess,
    render_images,
    sample_scenes,
    time,
    torch,
):
    @torch.no_grad()
    def sample_actions_no_cache(model, images, img_masks, lang_tokens, lang_masks, state, noise, num_steps):
        """비교용: KV 캐시 없이 매 step 접두부+접미부 전체를 다시 계산."""
        prefix_embs, prefix_pad, prefix_att = model.embed_prefix(images, img_masks, lang_tokens, lang_masks, state)

        def velocity(x_t, t):
            suffix_embs, suffix_pad, suffix_att = model.embed_suffix(x_t, t)
            pad = torch.cat([prefix_pad, suffix_pad], 1)
            att = torch.cat([prefix_att, suffix_att], 1)
            (_, out), _ = model.vlm_with_expert.forward(
                attention_mask=make_att_2d_masks(pad, att), position_ids=torch.cumsum(pad, 1) - 1,
                past_key_values=None, inputs_embeds=[prefix_embs, suffix_embs], use_cache=False,
            )
            return model.action_out_proj(out[:, -model.config.chunk_size :].float())

        return euler_integrate(velocity, noise, num_steps)

    def _sync():
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()

    torch.manual_seed(7)
    _red, _blue = sample_scenes(1)
    _batch = preprocess({"images": render_images(_red, _blue), "state": torch.rand(1, 2), "task": ["go to the red dot"]})
    _m = policy.model
    _images, _img_masks = policy.prepare_images(_batch)
    _args = (_images, _img_masks, _batch["observation.language.tokens"], _batch["observation.language.attention_mask"], policy.prepare_state(_batch))
    _noise = torch.randn(1, 16, 8, device=DEVICE)

    with torch.no_grad():
        _cached = _m.sample_actions(*_args, noise=_noise)
        _full = sample_actions_no_cache(_m, *_args, noise=_noise, num_steps=_m.config.num_steps)
        _times = {}
        for _name, _fn in [("KV 캐시", lambda: _m.sample_actions(*_args, noise=_noise)), ("캐시 없음", lambda: sample_actions_no_cache(_m, *_args, noise=_noise, num_steps=_m.config.num_steps))]:
            _fn()
            _sync()
            _t0 = time.perf_counter()
            for _ in range(20):
                _fn()
            _sync()
            _times[_name] = (time.perf_counter() - _t0) / 20 * 1000

    kv_cache_check = {"max_abs_diff": (_cached - _full).abs().max().item(), **{f"{k}_ms": v for k, v in _times.items()}}
    mo.md(
        f"""
    | 방식 | 청크 1개 생성 시간 |
    |---|---|
    | KV 캐시 (VLM 1회 + 전문가 10회) | {_times['KV 캐시']:.1f} ms |
    | 캐시 없음 (VLM+전문가 10회) | {_times['캐시 없음']:.1f} ms |

    두 결과의 최대 차이: **{kv_cache_check['max_abs_diff']:.2e}** (수치 오차 수준이면 같은 계산입니다).
    이 작은 모델은 접두부가 짧아 차이가 작습니다. 실제 SmolVLA는 접두부(이미지 64토큰 × 카메라 수 + 언어 최대 48)가 길고 VLM이 전문가보다 커서 차이가 큽니다.
    """
    )
    return (kv_cache_check,)


@app.cell
def _(DEVICE, plt, policy, postprocess, preprocess, render_images, sample_scenes, torch):
    torch.manual_seed(11)
    _red, _blue = sample_scenes(1)
    _img = render_images(_red, _blue)
    _state = torch.tensor([[0.15, 0.85]], device=DEVICE)
    _batch = preprocess({"images": _img, "state": _state, "task": ["move to blue"]})
    _trace = []
    _noise = torch.randn(1, 16, 8, device=DEVICE)
    policy.predict_action_chunk(_batch, noise=_noise, trace=_trace)
    _steps = [_noise] + _trace
    _show = [0, 2, 4, 6, 8, 10]
    _fig, _axes = plt.subplots(1, len(_show), figsize=(16, 2.9))
    for _ax, _k in zip(_axes, _show):
        _xy = postprocess(_steps[_k][0, :, :2]).cpu()
        _ax.imshow(_img[0].permute(1, 2, 0).cpu(), extent=[0, 1, 1, 0])
        _ax.plot(_xy[:, 0], _xy[:, 1], ".-", color="white", ms=3, lw=1)
        _ax.plot(*_state[0].cpu(), "o", color="yellow")
        _ax.set_xlim(-0.3, 1.3)
        _ax.set_ylim(1.3, -0.3)
        _ax.set_title(f"step {_k} (t={1 - _k / 10:.1f})", fontsize=9)
        _ax.set_xticks([])
        _ax.set_yticks([])
    _fig.suptitle("'move to blue': 노이즈(t=1)에서 행동 청크(t=0)로", fontsize=10)
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 11. 폐루프 평가

    환경에서 실제로 정책을 돌립니다. 매 step `select_action`으로 행동 하나를 꺼내 실행합니다(한 step 최대 이동 0.08).
    40 step 뒤 정답 점에서 0.05 이내면 성공입니다. `"any dot"`은 두 점 중 어느 쪽이든 성공입니다.

    지시문을 무시해도 한 색으로만 가는 정책은 그 색의 성공률이 높을 수 있습니다.
    같은 장면·시드와 균등한 red/blue 지시를 짝지어 평가하면, 지시를 무시하는 정책의 **두 성공률 평균**은
    최대 50%입니다(두 성공 영역이 겹치지 않음). 유한 표본·비결정적 연산의 오차는 별도로 봅니다.

    표를 볼 때 확인할 점:

    - **정답 도달률이 높고 오답 도달률이 낮은지** 두 지시 모두에서 확인하세요. 파랑 도달률만 0인 정책은 어느 점에도 못 갔을 수 있습니다.
    - `"any dot"` 성공률은 red/blue보다 낮을 수 있습니다. 청크를 새로 만들 때마다 목표를 다시 뽑기 때문에
      도중에 목표가 바뀌면 두 점 사이에서 머뭇거릴 수 있습니다. 13절에서 `n_action_steps`를 줄여 보면 이 현상이 더 심해집니다.
    """)
    return


@app.cell
def _(
    DEVICE,
    ENV_MAX_STEP,
    INSTRUCTIONS,
    SUCCESS_RADIUS,
    postprocess,
    preprocess,
    render_images,
    sample_scenes,
    torch,
):
    @torch.no_grad()
    def rollout(policy, kind, n_episodes=256, max_steps=40, seed=0, num_steps=None, n_action_steps=None):
        cfg = policy.config
        saved = (cfg.num_steps, cfg.n_action_steps)
        cfg.num_steps = num_steps or cfg.num_steps
        cfg.n_action_steps = n_action_steps or cfg.n_action_steps
        try:
            with torch.random.fork_rng(devices=[DEVICE] if DEVICE.type == "cuda" else []):
                torch.manual_seed(seed)
                red, blue = sample_scenes(n_episodes)
                images = render_images(red, blue)
                state = torch.rand(n_episodes, 2, device=DEVICE)
                texts = [INSTRUCTIONS[kind][i % len(INSTRUCTIONS[kind])] for i in range(n_episodes)]
                policy.reset()
                traj, calls = [state], 0
                for _ in range(max_steps):
                    if len(policy._queues["action"]) == 0:
                        calls += 1
                    batch = preprocess({"images": images, "state": state, "task": texts}, cfg.tokenizer_max_length)
                    target = postprocess(policy.select_action(batch))
                    step = target - state
                    step = step * (ENV_MAX_STEP / step.norm(dim=-1, keepdim=True).clamp_min(1e-8)).clamp(max=1.0)
                    state = (state + step).clamp(0, 1)
                    traj.append(state)
        finally:
            cfg.num_steps, cfg.n_action_steps = saved
            policy.reset()
        d_red = (state - red).norm(dim=-1)
        d_blue = (state - blue).norm(dim=-1)
        dist = {"red": d_red, "blue": d_blue, "any": torch.minimum(d_red, d_blue)}[kind]
        return {
            "kind": kind,
            "success_rate": (dist < SUCCESS_RADIUS).float().mean().item(),
            "mean_final_dist": dist.mean().item(),
            "reached_red": (d_red < SUCCESS_RADIUS).float().mean().item(),
            "reached_blue": (d_blue < SUCCESS_RADIUS).float().mean().item(),
            "chunk_calls": calls,
            "traj": torch.stack(traj, 1).cpu(),
            "images": images.cpu(),
            "texts": texts,
        }

    return (rollout,)


@app.cell
def _(mo, policy, rollout):
    eval_results = {k: rollout(policy, k, n_episodes=256, seed=100) for k in ["red", "blue", "any"]}
    mo.ui.table(
        [
            {
                "지시": k,
                "성공률": f"{r['success_rate']:.1%}",
                "최종 거리 평균": f"{r['mean_final_dist']:.3f}",
                "빨강 도달": f"{r['reached_red']:.1%}",
                "파랑 도달": f"{r['reached_blue']:.1%}",
                "청크 생성 횟수": r["chunk_calls"],
            }
            for k, r in eval_results.items()
        ],
        selection=None,
    )
    return (eval_results,)


@app.cell
def _(eval_results, plt):
    _fig, _axes = plt.subplots(3, 5, figsize=(14, 8))
    for _row, _kind in zip(_axes, ["red", "blue", "any"]):
        _r = eval_results[_kind]
        for _i, _ax in enumerate(_row):
            _tr = _r["traj"][_i]
            _ax.imshow(_r["images"][_i].permute(1, 2, 0), extent=[0, 1, 1, 0])
            _ax.plot(_tr[:, 0], _tr[:, 1], "-", color="white", lw=1)
            _ax.plot(_tr[0, 0], _tr[0, 1], "o", color="yellow", ms=5)
            _ax.plot(_tr[-1, 0], _tr[-1, 1], "x", color="lime", ms=7)
            _ax.set_title(repr(_r["texts"][_i]), fontsize=8)
            _ax.set_xticks([])
            _ax.set_yticks([])
    _fig.suptitle("폐루프 궤적 (노란 점 = 시작, 초록 X = 40 step 뒤)", fontsize=10)
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 12. 다봉 분포: 같은 관측, 다른 노이즈

    `"any dot"` 관측 하나를 고정하고 노이즈만 바꿔 행동 청크 64개를 뽑습니다.
    Flow Matching이 두 정답을 모두 표현한다면 청크가 **두 방향으로 갈라져야** 합니다.
    비교를 위해 같은 장면에 `"red"` 지시를 준 결과도 그립니다.
    """)
    return


@app.cell
def _(DEVICE, plt, policy, postprocess, preprocess, render_images, sample_scenes, torch):
    torch.manual_seed(21)
    _red, _blue = sample_scenes(1)
    _img = render_images(_red, _blue)
    _state = ((_red + _blue) / 2 + torch.tensor([[0.0, 0.15]], device=DEVICE)).clamp(0, 1)
    _n = 64
    _fig, _axes = plt.subplots(1, 2, figsize=(9, 3.6))
    multimodal_stats = {}
    for _ax, _task in zip(_axes, ["any dot", "red"]):
        _batch = preprocess({"images": _img.expand(_n, -1, -1, -1), "state": _state.expand(_n, -1), "task": [_task] * _n})
        _chunk = postprocess(policy.predict_action_chunk(_batch)).cpu()
        _end = _chunk[:, -1]
        _to_red = ((_end - _red.cpu()).norm(dim=-1) < (_end - _blue.cpu()).norm(dim=-1)).float().mean().item()
        multimodal_stats[_task] = _to_red
        _ax.imshow(_img[0].permute(1, 2, 0).cpu(), extent=[0, 1, 1, 0])
        for _c in _chunk:
            _ax.plot(_c[:, 0], _c[:, 1], "-", color="white", alpha=0.25, lw=1)
        _ax.plot(*_state[0].cpu(), "o", color="yellow")
        _ax.set_title(f"{_task!r}: 청크 끝이 빨강 쪽 {_to_red:.0%}", fontsize=9)
        _ax.set_xticks([])
        _ax.set_yticks([])
    _fig.tight_layout()
    _fig
    return (multimodal_stats,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 13. 직접 바꿔 보기: `num_steps`와 `n_action_steps`

    재학습 없이 추론 설정만 바꿉니다.

    - `num_steps`: Euler 적분 횟수. 줄이면 빠르지만 적분 오차가 커질 수 있습니다. 1이면 $x_0=\epsilon-v_\theta(\epsilon,1,o)$로 한 번 이동합니다.
    - `n_action_steps`: 청크에서 실제로 실행할 행동 수. 크면 모델 호출이 줄지만 관측을 덜 자주 반영합니다.
      SmolVLA 논문은 다음 청크를 미리 계산해 두는 **비동기 추론**으로 이 균형을 개선합니다.

    `n_action_steps=1`도 시도해 보세요. 매 step 새 노이즈로 청크를 뽑으면 연속된 행동이 어긋나 성공률이 떨어질 수 있습니다.
    청크를 여러 step 실행하는 것이 행동의 **시간적 일관성**을 지켜 준다는 점을 확인할 수 있습니다.
    """)
    return


@app.cell
def _(mo):
    num_steps_slider = mo.ui.slider(1, 20, value=10, show_value=True, label="num_steps")
    n_action_steps_slider = mo.ui.slider(1, 16, value=8, show_value=True, label="n_action_steps")
    mo.hstack([num_steps_slider, n_action_steps_slider], justify="start")
    return n_action_steps_slider, num_steps_slider


@app.cell
def _(mo, n_action_steps_slider, num_steps_slider, policy, rollout, time):
    _t0 = time.time()
    _res = [
        rollout(policy, k, n_episodes=128, seed=200, num_steps=num_steps_slider.value, n_action_steps=n_action_steps_slider.value)
        for k in ["red", "blue", "any"]
    ]
    mo.md(
        f"""
    `num_steps={num_steps_slider.value}`, `n_action_steps={n_action_steps_slider.value}` →
    성공률 red **{_res[0]['success_rate']:.1%}**, blue **{_res[1]['success_rate']:.1%}**, any **{_res[2]['success_rate']:.1%}**,
    에피소드당 청크 생성 **{_res[0]['chunk_calls']}회**, 평가 시간 {time.time() - _t0:.1f}초
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 14. (선택) 사전 학습하지 않은 VLM을 고정하면?

    `train_expert_only=True`는 VLM이 이미 쓸 만한 표현을 가졌다는 전제에 기대고 있습니다.
    같은 구조·같은 step 수로, **VQA 사전 학습을 하지 않은 무작위 VLM**을 고정한 채 행동 전문가를 학습해 비교합니다.
    학습이 한 번 더 돌아가므로 버튼을 눌러야 실행됩니다(GPU 기준 약 1.5분).

    무작위 VLM의 K/V에도 이미지 정보는 섞여 있지만, 색과 위치가 읽기 쉬운 형태로 정리되어 있지 않습니다.
    결과로 이 합성 과제와 학습 예산에서 사전 학습이 주는 이득을 비교하세요. 한 시드·고정 step 비교로 성능 상한이나 모든 VLM의 우열을 결론낼 수는 없습니다.
    """)
    return


@app.cell
def _(mo):
    random_vlm_button = mo.ui.run_button(label="무작위 VLM으로 학습·평가")
    random_vlm_button
    return (random_vlm_button,)


@app.cell
def _(
    BATCH_SIZE,
    POLICY_TRAIN_STEPS,
    SmolVLAToyConfig,
    build_tiny_vlm,
    eval_results,
    mo,
    random_vlm_button,
    rollout,
    train_policy,
):
    mo.stop(not random_vlm_button.value, mo.md("버튼을 누르면 실행됩니다."))
    _random_vlm = build_tiny_vlm(seed=0)  # 사전 학습 전과 같은 초기값
    _policy_rand, _log = train_policy(_random_vlm, SmolVLAToyConfig(), POLICY_TRAIN_STEPS, BATCH_SIZE, seed=0)
    _rows = []
    for _k in ["red", "blue", "any"]:
        _r = rollout(_policy_rand, _k, n_episodes=256, seed=100)
        _rows.append({"지시": _k, "사전 학습 VLM": f"{eval_results[_k]['success_rate']:.1%}", "무작위 VLM": f"{_r['success_rate']:.1%}"})
    mo.vstack([mo.md(f"무작위 VLM 학습 시간 {_log['seconds']:.0f}초"), mo.ui.table(_rows, selection=None)])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 정리

    | 구성 요소 | 원본 위치 | 핵심 |
    |---|---|---|
    | VLM 백본 | `SmolVLMWithExpertModel.__init__` | SmolVLM2의 앞 `num_vlm_layers`층만 사용, 기본 고정 |
    | 행동 전문가 | 같은 곳 (`lm_expert`) | VLM과 같은 헤드 구성, 폭 0.75배, 임베딩 없음 |
    | 층 교대 | `forward` + `self_attn_every_n_layers` | 자기 어텐션(접두부+행동 공동) ↔ 교차 어텐션(행동이 VLM K/V만 참조) |
    | 입력 시퀀스 | `embed_prefix` / `embed_suffix` | `[이미지, 언어, 상태] + [행동×chunk]`, 블록 마스크, 행동끼리 인과 |
    | 학습 목표 | `VLAFlowMatching.forward` | $x_t=t\epsilon+(1-t)a$, 목표 속도 $\epsilon-a$; 시간 분포는 8절 참조 |
    | 추론 | `sample_actions` / `denoise_step` | 접두부 KV 캐시 1회 + 전문가 10회 Euler 적분 |
    | 실행 | `SmolVLAPolicy.select_action` | 청크를 큐에 넣고 `n_action_steps`개 실행 후 다시 예측 |

    이 노트북에서 생략한 원본 기능: 실제 SmolVLM2 가중치와 토크나이저, 이미지 특수 토큰(`add_image_special_tokens`),
    `prefix_length` 고정 패딩, 실시간 청킹(RTC), ALOHA 좌표 변환, `torch.compile`, PEFT 설정.

    ## 실습 과제

    각 과제에서 **실행 전 예측 → 한 설정만 변경 → 같은 시드로 비교 → 차이의 이유 기록** 순서를 따르세요.

    1. `self_attn_every_n_layers`를 1(모두 자기 어텐션)이나 `attention_mode="self_attn"`으로 바꿔 학습하고 성공률·속도를 비교하세요.
    2. `num_vlm_layers`를 2, 6으로 바꿔 보세요. VLM 층을 줄이면 무엇이 빨라지고 무엇을 잃나요?
    3. `embed_suffix`의 `att_masks += [1] * chunk_size`를 `[1] + [0] * (chunk_size - 1)`(행동끼리 양방향, π0 방식)로 바꿔 비교하세요.
    4. `sample_time`을 균등분포 `U(0.001, 1)`로 바꿔 학습해 보세요. 손실 곡선과 성공률이 어떻게 달라지나요?
    5. 14절에서 무작위 VLM을 쓰되 `train_expert_only=False`로 VLM까지 학습하면 어떻게 되나요?
    6. `"any dot"` 데이터 비율(`KIND_PROBS`)을 바꿔 다봉 분포가 어떻게 변하는지 보세요.

    **힌트·판정 기준:** 1~2는 7절 층 배치와 파라미터 수를 먼저 확인한 뒤 성공률·시간을 함께 기록합니다.
    3은 마스크 그림에서 행동 블록이 삼각형에서 정사각형으로 바뀌는지 확인합니다.
    4는 시간 분포가 균등해지는지 확인하고, 서로 다른 시간 분포의 손실 값만으로 성능을 비교하지 않습니다.
    5는 VLM의 학습 가능 파라미터 수가 늘어나는지 확인합니다. `freeze_vision_encoder=True`이면 비전 인코더는 여전히 고정됩니다.
    6은 확률 합을 1로 유지하고, 12절에서 같은 관측에 여러 노이즈를 주어 두 목표 근처 종점 수를 각각 세세요.
    어느 설정도 성공률 상승이 정답 조건은 아닙니다. 예측과 관측의 차이를 구현·학습 예산과 연결하면 됩니다.

    **이해 확인:** (가) 왜 추론에서 시간 간격이 음수인가요? 노이즈가 $t=1$, 행동이 $t=0$이기 때문입니다.
    (나) VLM이 고정인데 `state_proj`는 어떻게 학습되나요? VLM 가중치는 갱신하지 않아도 입력에 대한 gradient는 전달됩니다.
    """)
    return


if __name__ == "__main__":
    app.run()
