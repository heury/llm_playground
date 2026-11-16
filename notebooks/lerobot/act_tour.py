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
    # ACT 구조 실습: 행동 청크 · CVAE · 시간 앙상블

    ACT는 이미지와 현재 상태에서 미래 행동 `k`개를 한 번에 예측합니다.
    [act.md](act.md)를 따라 배치 → 위치 임베딩·ResNet·DETR·CVAE → 전체 ACT → 손실 → 추론 → 시간 앙상블을 확인합니다.
    기초 텐서 연산과 Transformer 어텐션을 안다고 가정합니다.

    이 노트북은 고정된 작은 수열과 빈 이미지로 **shape과 계산 경로**를 확인합니다. 학습된 행동 성능을 보여 주는 예제가 아닙니다.
    갈림길 환경 구현, MLP 비교, 매개변수 집계, β별 학습·잠재 변수 분석, 폐루프 평가와 정책 비교는
    [ACT 보충 실험](act_experiments.py)으로 옮겼습니다.

    `notebooks/lerobot`에서 `uv run marimo edit act_tour.py`로 실행합니다.
    가중치 다운로드와 반복 학습은 없습니다. ResNet-18은 무작위 초기화합니다.

    원본: [LeRobot ACT](https://github.com/huggingface/lerobot/tree/e6ffde60312365c5452c46189a26872483943154/src/lerobot/policies/act),
    revision `e6ffde60312365c5452c46189a26872483943154`, Apache License 2.0,
    Copyright 2024–2026 The HuggingFace Inc. team. [ACT 원 저자 코드](https://github.com/tonyzhaozh/act)를 기반으로 하며,
    이 노트북은 단일 카메라·작은 차원으로 축소·재작성했습니다. `lerobot` 패키지는 설치하지 않습니다.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.mermaid("""
    flowchart LR
        subgraph TRAIN["학습 때만: CVAE 인코더"]
            Q1["상태 q_t"] --> VE["Transformer 인코더<br/>[CLS, 상태, 행동 k개]"]
            AC["정답 행동 청크 a_t..t+k-1"] --> VE
            VE --> MU["CLS 출력 → μ, log σ²"]
            MU --> Z["z = μ + σ·ε"]
        end
        Z0["추론: z = 0"] -.-> ZT
        Z --> ZT["z 토큰"]
        IMG["카메라 이미지"] --> RN["ResNet-18 layer4<br/>+ 2D sin 위치"] --> IT["이미지 토큰 h·w개"]
        Q2["상태 q_t"] --> ST["상태 토큰"]
        ZT --> ENC["Transformer 인코더"]
        ST --> ENC
        IT --> ENC
        DQ["0 입력 k개<br/>+ 학습 위치 쿼리"] --> DEC["Transformer 디코더<br/>(cross-attention)"]
        ENC -->|"K, V"| DEC
        DEC --> OUT["행동 청크 (B, k, A)<br/>forward 1회"]
    """)
    return


@app.cell
def _():
    import math
    from collections import deque
    from dataclasses import dataclass, replace

    import matplotlib.font_manager as font_manager
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    import torch.nn.functional as F
    import torchvision
    from torch import nn
    from torchvision.models._utils import IntermediateLayerGetter
    from torchvision.ops.misc import FrozenBatchNorm2d

    # 그림의 한글 표시용 글꼴: 설치된 것 중 첫 번째를 사용합니다.
    _installed = {f.name for f in font_manager.fontManager.ttflist}
    for _name in ["Noto Sans CJK KR", "NanumGothic", "Malgun Gothic", "AppleGothic", "Noto Sans Mono CJK KR"]:
        if _name in _installed:
            plt.rcParams["font.family"] = _name
            break
    plt.rcParams["axes.unicode_minus"] = False
    return (
        F,
        FrozenBatchNorm2d,
        IntermediateLayerGetter,
        dataclass,
        deque,
        math,
        nn,
        np,
        plt,
        replace,
        torch,
        torchvision,
    )


@app.cell
def _(torch):
    DEVICE = torch.device("cpu")  # 구조 확인은 작은 CPU 배치로 충분합니다.
    OBS_STATE = "observation.state"
    OBS_IMAGE = "observation.images.top"
    ACTION = "action"
    ACTION_IS_PAD = "action_is_pad"
    return ACTION, ACTION_IS_PAD, DEVICE, OBS_IMAGE, OBS_STATE


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. 학습 배치: 행동 청크와 패딩

    한 프레임의 상태·이미지와 미래 행동 `k`개를 함께 받습니다.
    에피소드 끝을 넘으면 마지막 행동을 반복하고 `action_is_pad=True`로 표시합니다.
    CVAE 인코더는 패딩 토큰을 가리고, L1 손실은 패딩 자리의 오차를 제외합니다.
    실제 데이터의 상태·행동에는 데이터셋 통계에 따른 정규화가 필요합니다.
    아래 수열은 이미 정규화된 입력을 가정하며, 빈 이미지는 비전 경로의 shape 확인용입니다.

    | 키 | 모양 | 의미 |
    | --- | --- | --- |
    | `observation.state` | `(B, 2)` | 현재 상태 |
    | `observation.images.top` | `(B, 3, 64, 64)` | 이미지 |
    | `action` | `(B, k, 2)` | 정답 행동 청크 |
    | `action_is_pad` | `(B, k)` | 패딩 표시 |
    """)
    return


@app.cell
def _(ACTION, ACTION_IS_PAD, DEVICE, OBS_IMAGE, OBS_STATE, np, torch):
    CHUNK = 20
    _t = np.linspace(-1, 1, 28, dtype=np.float32)
    _actions = np.stack([_t, _t**2], axis=-1)
    episodes = [{"state": np.roll(_actions, 1, axis=0), "action": _actions}]

    def chunk_at(ep, t, chunk_size=CHUNK):
        idx = np.arange(t, t + chunk_size)
        return ep["action"][np.minimum(idx, len(ep["action"]) - 1)], idx >= len(ep["action"])

    def make_batch(batch_size, rng, frames=None, chunk_size=CHUNK):
        if frames is None:
            frames = [(0, int(t)) for t in rng.integers(0, len(episodes[0]["state"]), batch_size)]
        chunks = [chunk_at(episodes[i], t, chunk_size) for i, t in frames]
        return {
            OBS_STATE: torch.tensor(np.array([episodes[i]["state"][t] for i, t in frames]), device=DEVICE),
            OBS_IMAGE: torch.zeros(len(frames), 3, 64, 64, device=DEVICE),
            ACTION: torch.tensor(np.array([a for a, _ in chunks]), device=DEVICE),
            ACTION_IS_PAD: torch.tensor(np.array([p for _, p in chunks]), device=DEVICE),
        }

    _b = make_batch(4, np.random.default_rng(0))
    for _key, _value in _b.items():
        print(_key, tuple(_value.shape), _value.dtype)
    print("에피소드 끝 8개 전에서 꺼낸 패딩:", chunk_at(episodes[0], 20)[1])
    return CHUNK, chunk_at, episodes, make_batch


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. 부품 하나씩 만들기

    act.md 2절의 구조를 아래 순서로 조립합니다. 각 단계 끝에서 작은 입력으로 shape과 성질을 확인합니다.

    | 단계 | 부품 | LeRobot 원본 이름 |
    | --- | --- | --- |
    | 2-1 | 1D·2D 사인파 위치 임베딩 | `create_sinusoidal_pos_embedding`, `ACTSinusoidalPositionEmbedding2d` |
    | 2-2 | ResNet-18 백본 → 이미지 토큰 | `backbone`, `encoder_img_feat_input_proj` |
    | 2-3 | DETR식 인코더·디코더 층 | `ACTEncoderLayer`, `ACTDecoderLayer` |
    | 2-4 | CVAE 인코더 | `vae_encoder*` |
    | 2-5 | 전체 `ACT` | `ACT` |

    ### 2-1. 위치 임베딩

    Transformer는 토큰 순서를 모르므로 위치 정보를 따로 넣어야 합니다. ACT는 대상마다 방식이 다릅니다.

    | 대상 | 방식 |
    | --- | --- |
    | CVAE 인코더 입력 `[CLS, 상태, 행동 k개]` | **고정** 1D sin/cos (act.md 코드는 학습형 `randn`으로 단순화) |
    | 이미지 특징 지도 `h × w` | **고정** 2D sin/cos: 채널 앞 절반은 y, 뒤 절반은 x 위치 |
    | 인코더의 1D 토큰 `[z, 상태]` | 학습형 `nn.Embedding(2, D)` |
    | 디코더 쿼리 `k`개 | 학습형 `nn.Embedding(k, D)` (DETR의 object query) |
    """)
    return


@app.cell
def position_definitions(math, np, torch):
    def create_sinusoidal_pos_embedding(num_positions, dimension):
        """1D sin/cos 위치 임베딩 (Attention is All You Need). 반환: (num_positions, dimension)"""
        pos = np.arange(num_positions)[:, None]
        j = np.arange(dimension)[None]
        angle = pos / np.power(10000, 2 * (j // 2) / dimension)
        table = np.where(j % 2 == 0, np.sin(angle), np.cos(angle))
        return torch.from_numpy(table).float()


    def sinusoidal_2d(h, w, dim_model, temperature=10000):
        """2D sin/cos 위치 임베딩: (h*w, dim_model). 앞 dim_model/2 채널은 y, 뒤 절반은 x.

        LeRobot ACTSinusoidalPositionEmbedding2d(dim_model // 2)와 같은 값입니다(행 우선 h·w 순서로 펼침).
        위치 번호 1..H를 (0, 2π]로 정규화한 뒤 주파수별 sin/cos를 번갈아 놓습니다.
        """
        half = dim_model // 2
        inv_freq = temperature ** (2 * (torch.arange(half) // 2) / half)

        def encode(n):
            r = torch.arange(1, n + 1, dtype=torch.float32) / (n + 1e-6) * 2 * math.pi
            a = r[:, None] / inv_freq
            return torch.stack([a[:, 0::2].sin(), a[:, 1::2].cos()], dim=-1).flatten(1)  # (n, half)

        ey, ex = encode(h), encode(w)
        return torch.cat([ey[:, None].expand(h, w, half), ex[None].expand(h, w, half)], dim=-1).reshape(h * w, dim_model)
    return create_sinusoidal_pos_embedding, sinusoidal_2d


@app.cell
def _(CHUNK, create_sinusoidal_pos_embedding, plt, sinusoidal_2d):
    _pe1 = create_sinusoidal_pos_embedding(CHUNK + 2, 128)
    _pe2 = sinusoidal_2d(8, 8, 128)
    _fig, _axes = plt.subplots(1, 3, figsize=(13, 3.2))
    _axes[0].imshow(_pe1.numpy(), aspect="auto", cmap="RdBu")
    _axes[0].set(title=f"1D: CVAE 입력 {CHUNK + 2}토큰 × 128채널", xlabel="채널", ylabel="토큰")
    _axes[1].imshow(_pe2[:, 0].reshape(8, 8).numpy(), cmap="RdBu")
    _axes[1].set(title="2D, 채널 0 (y 부분): 행마다 다름")
    _axes[2].imshow(_pe2[:, 64].reshape(8, 8).numpy(), cmap="RdBu")
    _axes[2].set(title="2D, 채널 64 (x 부분): 열마다 다름")
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 2-2. 비전 백본: ResNet-18 → 이미지 토큰

    ACT는 torchvision ResNet-18에서 **`layer4` 특징 지도만** 꺼냅니다(`IntermediateLayerGetter`). 해상도가 1/32로 줄고 채널은 512개입니다.
    그다음 1×1 합성곱으로 채널을 `D`로 줄이고, **픽셀 하나를 토큰 하나**로 펼칩니다.

    원본은 ImageNet 사전 학습 가중치(`ResNet18_Weights.IMAGENET1K_V1`)와 `FrozenBatchNorm2d`를 씁니다.
    여기서는 다운로드 없이 진행하려고 **무작위 초기화**로 만듭니다. FrozenBatchNorm은 통계가 고정이라 이 경우 사실상 항등 함수입니다.
    """)
    return


@app.cell
def backbone_definitions(FrozenBatchNorm2d, IntermediateLayerGetter, torch, torchvision):
    def make_backbone(pretrained_weights=None):
        """ResNet-18의 layer4 특징 지도를 'feature_map'으로 돌려주는 백본 (LeRobot ACT와 같은 구성)"""
        resnet = torchvision.models.resnet18(weights=pretrained_weights, norm_layer=FrozenBatchNorm2d)
        return IntermediateLayerGetter(resnet, return_layers={"layer4": "feature_map"})


    _bb = make_backbone().eval()
    _rows = []
    with torch.no_grad():
        for _h, _w, _what in [(64, 64, "이 노트북"), (96, 96, "작은 카메라"), (480, 640, "ALOHA 카메라 1대")]:
            _fm = _bb(torch.zeros(1, 3, _h, _w))["feature_map"]
            _rows.append(f"| {_what} | 3×{_h}×{_w} | {tuple(_fm.shape[1:])} | {_fm.shape[2] * _fm.shape[3]} |")
    print("| 입력 | 이미지 | layer4 특징 지도 (C, h, w) | 이미지 토큰 수 h·w |\n| --- | --- | --- | --- |\n" + "\n".join(_rows))
    print("ALOHA는 카메라 4대이므로 이미지 토큰만 4 × 300 = 1200개입니다.")
    return (make_backbone,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 2-3. DETR식 인코더·디코더 층

    act.md 코드는 `nn.Transformer`에 위치 임베딩을 더한 입력을 넣습니다. LeRobot(과 DETR)은 다르게,
    **매 층의 attention에서 위치 임베딩을 query와 key에만 더하고 value에는 더하지 않습니다.** 정규화는 잔차 뒤(post-norm)입니다.

    디코더 층은 self-attention → **cross-attention(query: 행동 쿼리, key/value: 인코더 출력)** → FFN 순서이고 **마스크가 없습니다**.
    GPT처럼 한 토큰씩 만드는 것이 아니라 `k`개를 동시에 만들기 때문입니다.

    LeRobot은 `(seq, batch, dim)` 순서를 쓰지만 여기서는 읽기 쉽게 `batch_first=True`로 씁니다. 계산은 같습니다.
    """)
    return


@app.cell
def transformer_definitions(nn):
    class DetrEncoderLayer(nn.Module):
        """post-norm 인코더 층. 위치 임베딩 pos는 q, k에만 더합니다."""

        def __init__(self, dim_model, n_heads, dim_feedforward, dropout):
            super().__init__()
            self.self_attn = nn.MultiheadAttention(dim_model, n_heads, dropout=dropout, batch_first=True)
            self.ffn = nn.Sequential(nn.Linear(dim_model, dim_feedforward), nn.ReLU(), nn.Dropout(dropout),
                                     nn.Linear(dim_feedforward, dim_model))
            self.norm1, self.norm2 = nn.LayerNorm(dim_model), nn.LayerNorm(dim_model)
            self.drop1, self.drop2 = nn.Dropout(dropout), nn.Dropout(dropout)

        def forward(self, x, pos, key_padding_mask=None):
            q = k = x + pos
            x = self.norm1(x + self.drop1(self.self_attn(q, k, value=x, key_padding_mask=key_padding_mask)[0]))
            return self.norm2(x + self.drop2(self.ffn(x)))


    class DetrDecoderLayer(nn.Module):
        """post-norm 디코더 층: self-attn → cross-attn → FFN. 마스크 없음."""

        def __init__(self, dim_model, n_heads, dim_feedforward, dropout):
            super().__init__()
            self.self_attn = nn.MultiheadAttention(dim_model, n_heads, dropout=dropout, batch_first=True)
            self.cross_attn = nn.MultiheadAttention(dim_model, n_heads, dropout=dropout, batch_first=True)
            self.ffn = nn.Sequential(nn.Linear(dim_model, dim_feedforward), nn.ReLU(), nn.Dropout(dropout),
                                     nn.Linear(dim_feedforward, dim_model))
            self.norm1, self.norm2, self.norm3 = (nn.LayerNorm(dim_model) for _ in range(3))
            self.drop1, self.drop2, self.drop3 = (nn.Dropout(dropout) for _ in range(3))

        def forward(self, x, memory, query_pos, memory_pos):
            q = k = x + query_pos
            x = self.norm1(x + self.drop1(self.self_attn(q, k, value=x)[0]))
            x = self.norm2(x + self.drop2(self.cross_attn(x + query_pos, memory + memory_pos, value=memory)[0]))
            return self.norm3(x + self.drop3(self.ffn(x)))
    return DetrDecoderLayer, DetrEncoderLayer


@app.cell
def _(DetrDecoderLayer, torch):
    # 확인: ACT 디코더의 입력은 "0으로 채운 k개 토큰"입니다. 쿼리 위치 임베딩이 없으면 k개 출력이 서로 구별될까요?
    torch.manual_seed(0)
    _layer = DetrDecoderLayer(32, 4, 64, dropout=0.0).eval()
    _memory = torch.randn(1, 6, 32)        # 인코더 출력 6토큰
    _x0 = torch.zeros(1, 5, 32)            # 디코더 입력: 0으로 채운 k=5개
    with torch.no_grad():
        _no_pos = _layer(_x0, _memory, torch.zeros(1, 5, 32), torch.zeros(1, 6, 32))
        _with_pos = _layer(_x0, _memory, torch.randn(1, 5, 32), torch.zeros(1, 6, 32))
    print("쿼리 위치 임베딩 없음: k개 출력 간 최대 차이 =", f"{(_no_pos - _no_pos[:, :1]).abs().max():.2e}", "→ 모두 같은 행동")
    print("쿼리 위치 임베딩 있음: k개 출력 간 최대 차이 =", f"{(_with_pos - _with_pos[:, :1]).abs().max():.2e}", "→ 자리마다 다른 행동")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    디코더 입력이 전부 0이므로 **"몇 번째 미래 행동인가"는 오직 쿼리 위치 임베딩 `decoder_pos_embed`에서** 나옵니다.
    act.md 코드처럼 `query_embed`를 `tgt`로 직접 넣는 방식도 같은 역할을 하지만, LeRobot은 DETR처럼 입력은 0, 위치는 매 층 q·k에 더합니다.

    ### 2-4. CVAE 인코더 (학습 때만 사용)

    act.md 코드 1의 `ACTCVAEEncoder`와 같은 일을 합니다. 카메라 이미지는 보지 않고 **현재 상태와 정답 행동 청크**만 봅니다.

    1. 토큰 `[CLS, 상태, 행동 k개]` → `(B, k+2, D)`, 고정 1D 위치 임베딩.
    2. 패딩된 행동 토큰은 `key_padding_mask`로 가립니다(act.md 코드에는 없음). CLS가 가짜 행동을 읽지 않게 하기 위해서입니다.
    3. CLS 출력 → `Linear(D, 2·latent)` → 평균 `μ`와 `log σ²`(원본 이름 `log_sigma_x2`).
    4. 재매개변수화 `z = μ + σ·ε`, `ε ~ N(0, I)`. 샘플링을 미분 가능한 덧셈·곱셈으로 바꿔 `μ, σ`까지 기울기가 흐르게 합니다.
    """)
    return


@app.cell
def cvae_definitions(DetrEncoderLayer, create_sinusoidal_pos_embedding, nn, torch):
    class CVAEEncoder(nn.Module):
        """[CLS, 상태, 행동 k개] → (μ, log σ²). LeRobot ACT의 vae_encoder* 모듈 묶음에 해당합니다."""

        def __init__(self, cfg):
            super().__init__()
            d = cfg.dim_model
            self.latent_dim = cfg.latent_dim
            self.cls_embed = nn.Embedding(1, d)
            self.robot_state_input_proj = nn.Linear(cfg.state_dim, d)
            self.action_input_proj = nn.Linear(cfg.action_dim, d)
            self.layers = nn.ModuleList(
                DetrEncoderLayer(d, cfg.n_heads, cfg.dim_feedforward, cfg.dropout) for _ in range(cfg.n_vae_encoder_layers)
            )
            self.latent_output_proj = nn.Linear(d, 2 * cfg.latent_dim)
            self.register_buffer("pos_enc", create_sinusoidal_pos_embedding(cfg.chunk_size + 2, d)[None])

        def forward(self, state, action, action_is_pad):
            """state (B, S), action (B, k, A), action_is_pad (B, k) bool → μ, log σ²: 각 (B, latent)"""
            b = state.shape[0]
            tokens = torch.cat([
                self.cls_embed.weight[None].expand(b, -1, -1),     # (B, 1, D)
                self.robot_state_input_proj(state)[:, None],      # (B, 1, D)
                self.action_input_proj(action),                   # (B, k, D)
            ], dim=1)
            # CLS와 상태 토큰은 패딩이 아님. 행동 토큰은 action_is_pad를 그대로 씀.
            key_padding_mask = torch.cat([torch.zeros(b, 2, dtype=torch.bool, device=state.device), action_is_pad], 1)
            for layer in self.layers:
                tokens = layer(tokens, self.pos_enc, key_padding_mask=key_padding_mask)
            params = self.latent_output_proj(tokens[:, 0])          # CLS 출력만 사용
            mu, log_sigma_x2 = params[:, : self.latent_dim], params[:, self.latent_dim :]
            return mu, log_sigma_x2


    def reparameterize(mu, log_sigma_x2):
        """z = μ + σ·ε (σ = exp(log σ² / 2))"""
        return mu + (log_sigma_x2 / 2).exp() * torch.randn_like(mu)
    return CVAEEncoder, reparameterize


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 2-5. 설정과 전체 `ACT` 조립

    `ACTConfig` 기본값은 LeRobot의 기본값(ALOHA 기준)입니다. 이 노트북 과제에서는 `TOY_CFG`로 크기를 줄입니다.
    act.md는 디코더 7층을 쓰지만, LeRobot은 **1층**이 기본입니다. 원 저자 코드가 7층을 만들고도 버그로 첫 층 출력만 썼기 때문에,
    LeRobot은 실제 동작과 같도록 1층으로 맞췄습니다(`configuration_act.py`의 주석).

    `forward`는 세 단계입니다(act.md 코드 2의 `ACTPolicy.forward`와 같은 흐름).

    1. **`z` 준비**: 학습 모드이고 배치에 정답 `action`이 있으면 CVAE 인코더로 `z`를 샘플링, 아니면 **`z = 0`**.
       이 노트북은 실험을 위해 `z`를 직접 넣는 인자 `z=`를 추가했습니다(원본에는 없음).
    2. **인코더**: 토큰 `[z, 상태, 이미지 h·w개]`, 위치 `[학습형 2개, 2D sin h·w개]`.
    3. **디코더**: 0 입력 `k`개 + 쿼리 위치 → cross-attention → `action_head`로 `(B, k, A)`.
    """)
    return


@app.cell
def act_definitions(
    ACTION,
    ACTION_IS_PAD,
    CVAEEncoder,
    DetrDecoderLayer,
    DetrEncoderLayer,
    OBS_IMAGE,
    OBS_STATE,
    dataclass,
    make_backbone,
    nn,
    reparameterize,
    replace,
    sinusoidal_2d,
    torch,
):
    @dataclass
    class ACTConfig:
        """LeRobot ACTConfig의 주요 기본값. state/action 차원은 원본에서는 데이터셋 특징으로부터 정해집니다."""
        state_dim: int = 14
        action_dim: int = 14
        chunk_size: int = 100
        n_action_steps: int = 100          # 한 번 예측한 청크에서 실제로 실행할 행동 수
        dim_model: int = 512
        n_heads: int = 8
        dim_feedforward: int = 3200
        n_encoder_layers: int = 4
        n_decoder_layers: int = 1          # 원 저자 코드의 7층 중 실제로 쓰인 것은 1층
        use_vae: bool = True
        latent_dim: int = 32
        n_vae_encoder_layers: int = 4
        dropout: float = 0.1
        kl_weight: float = 10.0
        temporal_ensemble_coeff: float | None = None  # 쓰려면 n_action_steps=1 이어야 함


    TOY_CFG = replace(ACTConfig(), state_dim=2, action_dim=2, chunk_size=20, n_action_steps=20, dim_model=128,
                      n_heads=4, dim_feedforward=512, n_encoder_layers=2, n_vae_encoder_layers=2, latent_dim=8)


    class ACT(nn.Module):
        """Action Chunking Transformer (LeRobot modeling_act.py의 ACT를 단일 카메라·batch_first로 축소)"""

        def __init__(self, cfg: ACTConfig):
            super().__init__()
            self.cfg = cfg
            d = cfg.dim_model
            self.vae_encoder = CVAEEncoder(cfg) if cfg.use_vae else None
            self.backbone = make_backbone()
            self.encoder_img_feat_input_proj = nn.Conv2d(512, d, kernel_size=1)
            self.encoder_latent_input_proj = nn.Linear(cfg.latent_dim, d)
            self.encoder_robot_state_input_proj = nn.Linear(cfg.state_dim, d)
            self.encoder_1d_feature_pos_embed = nn.Embedding(2, d)   # [z, 상태] 토큰의 학습형 위치
            self.encoder = nn.ModuleList(
                DetrEncoderLayer(d, cfg.n_heads, cfg.dim_feedforward, cfg.dropout) for _ in range(cfg.n_encoder_layers)
            )
            self.decoder = nn.ModuleList(
                DetrDecoderLayer(d, cfg.n_heads, cfg.dim_feedforward, cfg.dropout) for _ in range(cfg.n_decoder_layers)
            )
            self.decoder_norm = nn.LayerNorm(d)
            self.decoder_pos_embed = nn.Embedding(cfg.chunk_size, d)  # 디코더 쿼리 k개의 학습형 위치
            self.action_head = nn.Linear(d, cfg.action_dim)
            # 원본과 같이 Transformer 가중치는 Xavier-uniform으로 초기화
            for p in list(self.encoder.parameters()) + list(self.decoder.parameters()):
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)

        def forward(self, batch, z=None, trace=None):
            """반환: 행동 (B, k, A), (μ, log σ²) 또는 (None, None). trace(dict)를 주면 단계별 shape을 기록합니다."""
            state = batch[OBS_STATE]
            b = state.shape[0]
            log = (lambda k, v: trace.__setitem__(k, tuple(v.shape))) if trace is not None else (lambda k, v: None)

            # ── 1단계: 잠재 변수 z ──
            mu = log_sigma_x2 = None
            if z is None:
                if self.cfg.use_vae and self.training and ACTION in batch:
                    mu, log_sigma_x2 = self.vae_encoder(state, batch[ACTION], batch[ACTION_IS_PAD])
                    z = reparameterize(mu, log_sigma_x2)
                    log("CVAE 입력 [CLS, 상태, 행동 k]", torch.empty(b, self.cfg.chunk_size + 2, self.cfg.dim_model))
                    log("μ, log σ²", mu)
                else:
                    z = torch.zeros(b, self.cfg.latent_dim, device=state.device)  # 추론: 사전 분포의 평균
            log("z", z)

            # ── 2단계: Transformer 인코더 ──
            fmap = self.backbone(batch[OBS_IMAGE])["feature_map"]                   # (B, 512, h, w)
            h, w = fmap.shape[-2:]
            log("ResNet layer4 특징 지도", fmap)
            img_tokens = self.encoder_img_feat_input_proj(fmap).flatten(2).transpose(1, 2)  # (B, h·w, D)
            log("이미지 토큰", img_tokens)
            tokens = torch.cat([
                self.encoder_latent_input_proj(z)[:, None],
                self.encoder_robot_state_input_proj(state)[:, None],
                img_tokens,
            ], dim=1)                                                                # (B, 2 + h·w, D)
            pos = torch.cat([self.encoder_1d_feature_pos_embed.weight,
                             sinusoidal_2d(h, w, self.cfg.dim_model).to(state.device)], 0)[None]
            log("인코더 토큰 [z, 상태, 이미지]", tokens)
            for layer in self.encoder:
                tokens = layer(tokens, pos)

            # ── 3단계: Transformer 디코더 → 행동 ──
            x = torch.zeros(b, self.cfg.chunk_size, self.cfg.dim_model, device=state.device)  # 입력은 0
            log("디코더 입력 (0)", x)
            query_pos = self.decoder_pos_embed.weight[None]
            for layer in self.decoder:
                x = layer(x, tokens, query_pos, pos)
            actions = self.action_head(self.decoder_norm(x))
            log("행동 청크", actions)
            return actions, (mu, log_sigma_x2)
    return ACT, ACTConfig, TOY_CFG


@app.cell
def _(ACT, DEVICE, TOY_CFG, make_batch, mo, np, torch):
    torch.manual_seed(0)
    _model = ACT(TOY_CFG).to(DEVICE)
    _batch = make_batch(4, np.random.default_rng(1))
    _train_trace, _infer_trace = {}, {}
    with torch.no_grad():
        _model.train()
        _a_train, (_mu, _) = _model(_batch, trace=_train_trace)
        _model.eval()
        _a_infer, (_mu_inf, _) = _model(_batch, trace=_infer_trace)

    _keys = list(dict.fromkeys(list(_train_trace) + list(_infer_trace)))
    _rows = "\n".join(f"| {k} | `{_train_trace.get(k, '—')}` | `{_infer_trace.get(k, '쓰지 않음')}` |" for k in _keys)
    mo.md(
        f"작은 배치(B=4)로 `TOY_CFG` 모델의 단계별 shape (D={TOY_CFG.dim_model}, k={TOY_CFG.chunk_size}, latent={TOY_CFG.latent_dim})\n\n"
        "| 단계 | 학습 모드 (`model.train()`, 정답 있음) | 추론 모드 (`model.eval()`) |\n| --- | --- | --- |\n" + _rows
        + f"\n\n추론 모드에서 μ는 `{_mu_inf}`입니다. 정답 행동이 배치에 있어도 `eval()`이면 CVAE 인코더를 쓰지 않습니다."
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    이미지가 64×64이므로 이미지 토큰은 2×2=4개입니다.
    인코더는 `z` 1개 + 상태 1개 + 이미지 4개를 받고, 디코더는 이 6개 토큰을 읽어 행동 20개를 동시에 출력합니다.
    실제 크기의 매개변수 집계는 [보충 실험 3-6절](act_experiments.py)에 있습니다.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. 손실: 마스크된 L1 + β·KL

    $$\mathcal{L} = \underbrace{\operatorname{mean}\big(|\hat a - a| \cdot \lnot\text{pad}\big)}_{\text{L1 재구성}}
      + \beta \cdot \underbrace{\operatorname{mean}_B \sum_{j} -\tfrac12\big(1 + \log\sigma_j^2 - \mu_j^2 - \sigma_j^2\big)}_{D_\text{KL}(\mathcal N(\mu,\sigma^2)\,\|\,\mathcal N(0, I))}$$

    - L1은 패딩 자리를 0으로 만든 뒤 **전체 원소 수로 평균**합니다(LeRobot과 같음. 패딩이 많으면 손실이 약간 작아집니다).
    - KL은 잠재 차원에 대해 **합**, 배치에 대해 평균입니다. LeRobot 기본 `kl_weight` β = 10.
    - KL 항은 CVAE 인코더의 출력 분포를 표준정규분포 쪽으로 끌어당깁니다. 그래야 추론 때 `z = 0`(사전 분포의 평균)이
      디코더가 학습 중 보던 `z`와 비슷한 값이 됩니다. β가 너무 크면 `z`가 아무 정보도 담지 않게 됩니다(**사후 붕괴**, 보충 실험 6절에서 확인).
    """)
    return


@app.cell
def loss_definitions(ACTION, ACTION_IS_PAD, F):
    def kl_to_standard_normal(mu, log_sigma_x2):
        """D_KL(N(μ, σ²) || N(0, I)): 잠재 차원 합, 배치 평균"""
        return (-0.5 * (1 + log_sigma_x2 - mu.pow(2) - log_sigma_x2.exp())).sum(-1).mean()


    def act_loss(model, batch, kl_weight):
        actions_hat, (mu, log_sigma_x2) = model(batch)
        l1 = (F.l1_loss(batch[ACTION], actions_hat, reduction="none") * ~batch[ACTION_IS_PAD].unsqueeze(-1)).mean()
        info = {"l1": l1.item()}
        loss = l1
        if mu is not None:
            kld = kl_to_standard_normal(mu, log_sigma_x2)
            info["kl"] = kld.item()
            loss = l1 + kl_weight * kld
        return loss, info
    return act_loss, kl_to_standard_normal


@app.cell
def _(
    ACT,
    ACTION,
    ACTION_IS_PAD,
    DEVICE,
    TOY_CFG,
    act_loss,
    episodes,
    kl_to_standard_normal,
    make_batch,
    torch,
):
    # 확인 1: 닫힌 형태의 KL이 torch.distributions의 값과 같은가
    _mu, _lv = torch.randn(5, 8), torch.randn(5, 8)
    _ref = torch.distributions.kl_divergence(
        torch.distributions.Normal(_mu, (_lv / 2).exp()), torch.distributions.Normal(0.0, 1.0)).sum(-1).mean()
    print(f"KL 닫힌 형태 {kl_to_standard_normal(_mu, _lv):.6f} vs torch.distributions {_ref:.6f}")
    print(f"μ=0, log σ²=0이면 KL = {kl_to_standard_normal(torch.zeros(2, 8), torch.zeros(2, 8)):.1f}")

    # 확인 2: 패딩 자리의 정답을 바꿔도 손실과 μ가 그대로인가 (dropout을 끄려고 eval 후 CVAE만 직접 호출)
    torch.manual_seed(0)
    _m = ACT(TOY_CFG).to(DEVICE).eval()
    _t_end = len(episodes[0]["state"]) - 8  # 에피소드 0의 끝 근처 프레임: 청크 20개 중 12개가 패딩
    _b = make_batch(1, None, frames=[(0, _t_end)])
    _b2 = {k: v.clone() for k, v in _b.items()}
    _b2[ACTION][_b2[ACTION_IS_PAD]] = 100.0
    with torch.no_grad():
        _mu1, _ = _m.vae_encoder(_b["observation.state"], _b[ACTION], _b[ACTION_IS_PAD])
        _mu2, _ = _m.vae_encoder(_b2["observation.state"], _b2[ACTION], _b2[ACTION_IS_PAD])
        _l1 = act_loss(_m, _b, 10.0)[1]["l1"]
        _l2 = act_loss(_m, _b2, 10.0)[1]["l1"]
    print(f"패딩 {int(_b[ACTION_IS_PAD].sum())}개를 100으로 바꿈 → μ 최대 차이 {(_mu1 - _mu2).abs().max():.2e}, "
          f"L1 {_l1:.5f} vs {_l2:.5f}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. 추론: 정답 없이 `z = 0`

    학습 때 CVAE는 상태와 **정답 행동 청크**에서 μ와 log σ²를 계산하고,
    $z=\mu+\sigma\epsilon$을 표본 추출합니다. 손실은 L1 + β·KL이며 KL은 사후 분포를 N(0,I)에 가깝게 만듭니다.
    추론 때는 정답 청크가 없으므로 CVAE 인코더를 건너뛰고 사전 분포의 평균 `z=0`을 넣습니다.
    이는 가능한 행동들을 모두 표본 추출하는 방식은 아닙니다. β에 따른 사후 붕괴와 다봉성은 [보충 실험 5~6절](act_experiments.py)에서 다룹니다.

    앞의 shape 표에서도 `eval()`일 때 μ가 `None`임을 확인했습니다. 아래에서는 정답 키까지 제거해 추론합니다.
    """)
    return


@app.cell
def _(ACT, DEVICE, OBS_IMAGE, OBS_STATE, TOY_CFG, make_batch, np, torch):
    _model = ACT(TOY_CFG).to(DEVICE).eval()
    _batch = make_batch(1, np.random.default_rng(0))
    _obs = {k: _batch[k] for k in (OBS_STATE, OBS_IMAGE)}
    with torch.no_grad():
        _actions, (_mu, _logvar) = _model(_obs)
    assert _actions.shape == (1, TOY_CFG.chunk_size, TOY_CFG.action_dim)
    assert _mu is None and _logvar is None
    print("정답 없는 추론:", tuple(_actions.shape), "μ:", _mu)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. 시간 앙상블

    매 step 청크를 새로 예측하면 현재 시각의 행동을 예측한 값이 여러 청크에 겹칩니다.
    겹치는 값들의 지수 가중 평균으로 이번 행동을 정합니다. 아래 그림은 `k=4` 예제입니다.
    """)
    return


@app.cell
def _(mo):
    coeff_slider = mo.ui.slider(-0.3, 0.3, step=0.01, value=0.01, label="앙상블 계수 m", show_value=True)
    coeff_slider
    return (coeff_slider,)


@app.cell
def _(coeff_slider, np, plt):
    _k, _t_now = 4, 3
    _m = coeff_slider.value
    _fig, _axes = plt.subplots(1, 2, figsize=(13, 3.4), gridspec_kw={"width_ratios": [1.5, 1]})
    _ax = _axes[0]
    for _s in range(_t_now + 1):
        for _j in range(_k):
            _t = _s + _j
            _hit = _t == _t_now
            _ax.add_patch(plt.Rectangle((_t, -_s - 0.4), 0.9, 0.8, fc="tab:red" if _hit else "0.85", alpha=0.8 if _hit else 0.6))
            _ax.text(_t + 0.45, -_s, f"a{_s}_{_t}", ha="center", va="center", fontsize=9)
    # LeRobot/원 저자 규약: 가장 오래된 예측이 w_0
    _ages = np.arange(_t_now, -1, -1)   # 행 s=0..3 의 예측이 몇 step 전에 만들어졌는가
    _w = np.exp(-_m * np.arange(_t_now + 1))  # i = 0(가장 오래됨) .. 3(가장 최근)
    for _s in range(_t_now + 1):
        _ax.text(_k + _t_now + 0.2, -_s, f"t={_s} 예측 · {_ages[_s]} step 전 · w={_w[_s] / _w.sum():.2f}", va="center", fontsize=9)
    _ax.set(xlim=(0, _k + _t_now + 3.6), ylim=(-_t_now - 0.7, 0.7), title=f"현재 t={_t_now}: 빨간 열 4개를 가중 평균 (m={_m:+.2f})")
    _ax.set_xticks(np.arange(_k + _t_now) + 0.45, [f"t={i}" for i in range(_k + _t_now)]); _ax.set_yticks([])

    _ax = _axes[1]
    _K = 20
    for _mm, _c in [(_m, "tab:red"), (0.01, "0.5"), (-0.1, "tab:blue")]:
        _w = np.exp(-_mm * np.arange(_K))
        _ax.plot(np.arange(_K - 1, -1, -1), _w / _w.sum(), "o-", ms=3, c=_c, label=f"m={_mm:+.2f}")
    _ax.set(title=f"k={_K}, 예측 20개가 모두 겹칠 때 정규화 가중치", xlabel="예측을 만든 시점 (몇 step 전)", ylabel="가중치")
    _ax.legend(); _ax.grid(alpha=0.3)
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **가중치 방향 주의.** 원 저자 코드와 LeRobot의 `ACTTemporalEnsembler`는 $w_i = \exp(-m \cdot i)$에서
    **$w_0$이 가장 오래된 예측**입니다. 기본 $m = 0.01 > 0$이면 오래된 예측이 **약간 더** 큰 가중치를 받습니다
    (k=20에서는 거의 균등 평균). act.md 3절은 "가장 최근 예측($i=0$)에 가장 큰 가중치"라고 설명하는데, 이는 $m$의 부호를 바꾼 것과 같습니다.
    최신 예측을 더 믿고 싶다면 $m < 0$을 쓰면 됩니다. 슬라이더로 바꿔 보세요.

    또 act.md는 "지수 이동 평균(EMA)"이라고 부르지만, 실제로는 **겹치는 예측 전체에 대한 가중 평균**입니다.

    ### 7-2. act.md 방식 A·B와 LeRobot 구현 비교

    - **방식 A**(act.md): 최근 `k`개 청크를 `deque`에 보관하고, 매 step 겹치는 값을 꺼내 가중 평균합니다. 이해하기 쉽지만 매 step O(k) 순회입니다.
    - **방식 B / LeRobot**: 아직 실행하지 않은 미래 `k-1`개 자리의 **가중 평균과 겹친 개수**를 버퍼에 저장해 두고, 새 청크가 오면
      온라인 평균 공식으로 갱신합니다. `(이전 평균 × 이전 가중치 합 + 새 값 × 새 가중치) / 전체 가중치 합`.

    두 방식이 **같은 값을 내는지** 무작위 청크 30개로 확인합니다. act.md의 방식 A는 최신 예측을 $i=0$으로 두므로,
    LeRobot과 같아지려면 순서를 뒤집어야 합니다.
    """)
    return


@app.cell
def ensemble_definitions(deque, torch):
    class DequeTemporalEnsembler:
        """act.md 방식 A. newest_first=True면 act.md 그대로(최신 예측이 w_0), False면 원 저자/LeRobot 규약(가장 오래된 예측이 w_0)."""

        def __init__(self, chunk_size, coeff, newest_first=True):
            self.coeff, self.newest_first = coeff, newest_first
            self.history = deque(maxlen=chunk_size)

        def update(self, chunk):
            """chunk: (k, A) → 이번 step 행동 (A,)"""
            self.history.append(chunk)
            n = len(self.history)
            preds = torch.stack([self.history[-1 - i][i] for i in range(n)])  # i=0: 최신 청크의 0번째
            if not self.newest_first:
                preds = preds.flip(0)                                          # i=0: 가장 오래된 청크
            w = torch.exp(-self.coeff * torch.arange(n, dtype=chunk.dtype))[:, None]
            return (preds * w).sum(0) / w.sum()


    class ACTTemporalEnsembler:
        """LeRobot modeling_act.py의 ACTTemporalEnsembler (배치 지원, 온라인 가중 평균). w_0은 가장 오래된 예측."""

        def __init__(self, temporal_ensemble_coeff, chunk_size):
            self.chunk_size = chunk_size
            self.ensemble_weights = torch.exp(-temporal_ensemble_coeff * torch.arange(chunk_size))
            self.ensemble_weights_cumsum = torch.cumsum(self.ensemble_weights, dim=0)
            self.reset()

        def reset(self):
            self.ensembled_actions = None          # (B, k-1, A): 아직 실행 안 한 미래 자리의 현재 평균
            self.ensembled_actions_count = None    # (k-1, 1): 각 자리에 지금까지 겹친 예측 수

        def update(self, actions):
            """actions: (B, k, A) 새 청크 → 이번 step 행동 (B, A)"""
            w = self.ensemble_weights.to(actions.device)
            w_cum = self.ensemble_weights_cumsum.to(actions.device)
            if self.ensembled_actions is None:
                self.ensembled_actions = actions.clone()
                self.ensembled_actions_count = torch.ones((self.chunk_size, 1), dtype=torch.long, device=actions.device)
            else:
                # 이전 평균(k-1개)과 새 청크의 앞 k-1개를 온라인 가중 평균
                self.ensembled_actions *= w_cum[self.ensembled_actions_count - 1]
                self.ensembled_actions += actions[:, :-1] * w[self.ensembled_actions_count]
                self.ensembled_actions /= w_cum[self.ensembled_actions_count]
                self.ensembled_actions_count = torch.clamp(self.ensembled_actions_count + 1, max=self.chunk_size)
                # 새 청크의 마지막 행동은 아직 겹치는 예측이 없으므로 뒤에 그대로 붙임
                self.ensembled_actions = torch.cat([self.ensembled_actions, actions[:, -1:]], dim=1)
                self.ensembled_actions_count = torch.cat(
                    [self.ensembled_actions_count, torch.ones_like(self.ensembled_actions_count[-1:])])
            # 맨 앞(이번 step) 행동을 꺼내고 버퍼를 한 칸 당김
            action, self.ensembled_actions, self.ensembled_actions_count = (
                self.ensembled_actions[:, 0], self.ensembled_actions[:, 1:], self.ensembled_actions_count[1:])
            return action
    return ACTTemporalEnsembler, DequeTemporalEnsembler


@app.cell
def _(ACTTemporalEnsembler, DequeTemporalEnsembler, coeff_slider, torch):
    torch.manual_seed(0)
    _k, _m = 6, (coeff_slider.value if abs(coeff_slider.value) > 1e-9 else 0.1)
    _chunks = torch.randn(30, _k, 2, dtype=torch.float64)
    _lerobot = ACTTemporalEnsembler(_m, _k)
    _a_md = DequeTemporalEnsembler(_k, _m, newest_first=True)
    _a_fix = DequeTemporalEnsembler(_k, _m, newest_first=False)
    _out = {"LeRobot": [], "A(act.md)": [], "A(순서 뒤집음)": []}
    for _c in _chunks:
        _out["LeRobot"].append(_lerobot.update(_c[None])[0])
        _out["A(act.md)"].append(_a_md.update(_c))
        _out["A(순서 뒤집음)"].append(_a_fix.update(_c))
    _ref = torch.stack(_out["LeRobot"])
    print(f"k={_k}, m={_m:+.2f}, 무작위 청크 30개를 차례로 넣음")
    for _name in ["A(act.md)", "A(순서 뒤집음)"]:
        print(f"  LeRobot vs {_name:12s}: 최대 차이 {(torch.stack(_out[_name]) - _ref).abs().max():.2e}")
    print("m=0이면 두 규약 모두 단순 평균이라 차이가 없습니다. 슬라이더로 m을 바꿔 확인하세요.")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    act.md의 방식 B 코드(`LeRobotFastTemporalEnsembler`)도 구조는 LeRobot과 같습니다. 새 예측에 `weights[count]`를,
    기존 평균에 `weights_cumsum[count - 1]`을 곱하므로 **새로 들어온 예측일수록 뒤쪽 가중치**를 받습니다. 즉 act.md의 방식 A와 B는 서로 다른 규약입니다.
    위 확인에서 "순서 뒤집음"만 LeRobot과 기계 정밀도 수준으로 같아야 합니다.

    LeRobot에서 시간 앙상블은 `temporal_ensemble_coeff`를 설정하고 **`n_action_steps = 1`** 일 때만 씁니다(매 step 추론).
    설정하지 않으면 기본 `n_action_steps = 100`으로 청크를 큐에 넣고 다 쓸 때까지 다시 추론하지 않습니다.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. 정리와 다음 실습

    - **학습:** `[CLS, 상태, 정답 행동 청크]` → CVAE → `z` → 이미지·상태와 함께 인코딩 → `k`개 행동, L1 + β·KL.
    - **추론:** CVAE를 건너뛰고 `z=0`, 정답 없이 `k`개 행동을 동시에 출력합니다.
    - **실행:** 청크 일부 또는 전부를 실행한 뒤 다시 관측하거나, 매 step 추론하여 시간 앙상블합니다.

    합성 수열·빈 이미지와 무작위 가중치로 구조만 확인했습니다. 실제 정책에는 시연 학습과 데이터에 맞는 전처리가 필요합니다.
    [ACT 보충 실험](act_experiments.py)에서는 환경·MLP 비교 → 매개변수 집계 → β별 학습 → 잠재 변수 분석 → 폐루프 평가를 진행합니다.

    **직접 확인하기:** 디코더 쿼리를 모두 같은 벡터로 바꾸면 행동들이 어떻게 되는지,
    패딩 정답을 바꿔도 손실이 같은지, 시간 앙상블 계수의 부호에 따라 어떤 예측이 강조되는지 설명해 보세요.
    """)
    return

if __name__ == "__main__":
    app.run()
