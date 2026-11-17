# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo",
#     "torch",
#     "torchvision",
#     "numpy",
#     "matplotlib",
#     "einops",
# ]
# ///

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
    # ACT(Action Chunking Transformer) 실습 노트북

    LeRobot의 [`modeling_act.py`](https://github.com/huggingface/lerobot/blob/main/src/lerobot/policies/act/modeling_act.py)를
    **핵심만 남겨 직접 다시 구현**하면서, ACT의 개념이 코드에서 어떻게 구현되는지 확인합니다.

    | 절 | 개념 | LeRobot 원본 코드 |
    |---|---|---|
    | 1 | 데이터: 행동 청크 라벨과 `action_is_pad` | `ACTConfig.action_delta_indices` |
    | 2 | 이미지를 토큰으로: ResNet18, 1×1 Conv, 2D 위치 임베딩 | `ACT.__init__`, `ACTSinusoidalPositionEmbedding2d` |
    | 3 | DETR 스타일 인코더 (위치 정보는 Q, K에만) | `ACTEncoderLayer` |
    | 4 | 학습된 쿼리로 k스텝을 한 번에 출력하는 디코더 | `ACTDecoderLayer`, `decoder_pos_embed` |
    | 5 | CVAE 인코더와 잠재변수 z | `ACT.forward`의 `vae_encoder` 분기 |
    | 6 | 전체 조립 | `ACT.forward` |
    | 7 | 손실: 마스킹한 L1 + KL | `ACTPolicy.forward` |
    | 8–9 | 학습과 z가 배운 것 확인 | — |
    | 10 | 추론: 행동 큐와 Temporal Ensembling | `ACTPolicy.select_action`, `ACTTemporalEnsembler` |

    **읽는 순서**: 0–7절에서 구조와 손실을 구현하고, 8–10절에서 학습·추론 결과를 확인합니다.
    환경 생성과 학습·평가 보조 코드는 접어 두었습니다. 부록 A·B는 선택 실행이며, VLA 비교는 후속 학습입니다.

    **실습 과제**: 2D 평면에서 에이전트(파랑)가 가운데 장애물(빨강)을 **왼쪽이나 오른쪽으로 돌아서**
    목표(초록)에 도달합니다. 시연 데이터는 좌우가 50:50으로 섞여 있어(**다봉 분포**) ACT가 CVAE를 쓰는 이유를 직접 볼 수 있습니다.

    > 실행: `uvx marimo edit --sandbox act_tutorial.py` (GPU가 있으면 자동으로 사용합니다)
    """)
    return


@app.cell(hide_code=True)
def _():
    import math
    from collections import deque
    from dataclasses import dataclass, replace

    import einops
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    import torch.nn.functional as F
    import torchvision
    from torch import nn
    from torchvision.models._utils import IntermediateLayerGetter
    from torchvision.ops.misc import FrozenBatchNorm2d

    # 그래프의 한글 표시용 폰트 (설치된 것 중 첫 번째)
    from matplotlib import font_manager

    _installed = {f.name for f in font_manager.fontManager.ttflist}
    _ko = [f for f in ["Noto Sans CJK KR", "NanumGothic", "Malgun Gothic", "AppleGothic", "Noto Serif CJK KR"]
           if f in _installed]
    plt.rcParams.update({"font.family": _ko[:1] or ["DejaVu Sans"], "axes.unicode_minus": False})

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    print("device:", device, "| font:", _ko[:1] or "한글 폰트 없음")
    return (
        F,
        FrozenBatchNorm2d,
        IntermediateLayerGetter,
        dataclass,
        deque,
        device,
        einops,
        math,
        nn,
        np,
        plt,
        replace,
        torch,
        torchvision,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 0. 설정 (`ACTConfig`의 축소판)

    원본 기본값(ALOHA 양팔 로봇 기준)과 이 실습에서 쓰는 값을 비교합니다. 구조는 같고 크기만 줄였습니다.

    | 파라미터 | LeRobot 기본값 | 실습값 |
    |---|---|---|
    | `chunk_size` / `n_action_steps` | 100 / 100 | 20 / 20 |
    | `dim_model` / `n_heads` / `dim_feedforward` | 512 / 8 / 3200 | 128 / 4 / 512 |
    | `n_encoder_layers` / `n_decoder_layers` | 4 / 1 | 2 / 1 |
    | `n_vae_encoder_layers` / `latent_dim` | 4 / 32 | 2 / 8 |
    | `kl_weight` | 10.0 | 10.0 |
    """)
    return


@app.cell
def _(dataclass):
    @dataclass
    class MiniACTConfig:
        state_dim: int = 2  # observation.state 차원 (ALOHA는 14)
        action_dim: int = 2  # action 차원
        image_size: int = 96  # 정사각형 이미지 한 변
        chunk_size: int = 20  # 한 번에 예측하는 미래 스텝 수 k
        n_action_steps: int = 20  # 예측한 청크에서 실제로 실행할 스텝 수
        dim_model: int = 128
        n_heads: int = 4
        dim_feedforward: int = 512
        n_encoder_layers: int = 2
        n_decoder_layers: int = 1
        n_vae_encoder_layers: int = 2
        latent_dim: int = 8
        use_vae: bool = True
        kl_weight: float = 10.0
        dropout: float = 0.1

    cfg = MiniACTConfig()
    cfg
    return MiniACTConfig, cfg


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. 데이터: 시연 궤적과 행동 청크

    시연 한 개는 `(state_t, action_t)` 쌍이 이어진 것입니다. 여기서 `action_t`는 **다음 목표 위치**입니다.
    실제 로봇의 "다음 관절 목표각"과 같은 역할입니다.

    아래 환경 생성 코드는 실습용 보조 코드입니다. `make_demo`는 좌·우 우회 시연을 만들고,
    `render`는 상태와 목표를 `(3, H, W)` 이미지로 바꿉니다. 처음에는 관측으로 우회 방향을
    구별할 수 없다는 점만 확인하고, 경로 수식과 원 그리기 구현은 건너뛰어도 됩니다.
    """)
    return


@app.cell(hide_code=True)
def _(cfg, np):
    EP_LEN = 60  # 에피소드 길이
    N_EP = 200  # 시연 개수
    OBS_R = 0.35  # 장애물 반지름 (원점에 위치)


    def make_demo(goal_x, side, rng):
        """처음 20%는 직진하고, 이후 side=+1이면 오른쪽, -1이면 왼쪽으로 장애물을 우회하는 시연 궤적.
        직진 구간에서는 관측만으로 어느 쪽으로 갈지 알 수 없다 → 다봉 분포."""
        t = np.linspace(0, 1, EP_LEN + 1)
        start, goal = np.array([0.0, -1.0]), np.array([goal_x, 1.0])
        traj = start + (goal - start) * t[:, None]
        traj[:, 0] += side * 0.7 * np.sin(np.pi * np.clip((t - 0.2) / 0.8, 0, 1))  # 좌/우 우회
        traj += rng.normal(0, 0.005, traj.shape)  # 사람 시연의 손떨림
        return {"states": traj[:-1], "actions": traj[1:], "goal_x": goal_x, "side": side}


    def render(pos, goal_x, size=cfg.image_size):
        """(3, H, W) 이미지. R=장애물, G=목표, B=에이전트."""
        img = np.zeros((3, size, size), np.float32)
        yy, xx = np.mgrid[0:size, 0:size]
        scale = (size - 1) / 2.4  # 월드 좌표 [-1.2, 1.2] → 픽셀 좌표

        def disk(ch, p, r_px):
            px, py = (p[0] + 1.2) * scale, (1.2 - p[1]) * scale
            img[ch][(xx - px) ** 2 + (yy - py) ** 2 <= r_px**2] = 1.0

        disk(0, (0.0, 0.0), OBS_R * scale)
        disk(1, (goal_x, 1.0), 4)
        disk(2, pos, 4)
        return img


    _rng = np.random.default_rng(0)
    demos = [make_demo(_rng.uniform(-0.3, 0.3), _rng.choice([-1, 1]), _rng) for _ in range(N_EP)]
    return EP_LEN, N_EP, OBS_R, demos, render


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 1-1. 미래 k스텝을 라벨로 묶기: `action_delta_indices`와 `action_is_pad`

    LeRobot의 `ACTConfig`에는 다음 코드가 있습니다.

    ```python
    @property
    def action_delta_indices(self) -> list:
        return list(range(self.chunk_size))   # t, t+1, ..., t+k-1
    ```

    데이터셋은 이 인덱스로 미래 행동을 함께 가져옵니다. 에피소드 끝을 넘어가는 인덱스는
    **마지막 프레임 값으로 채우고** `action_is_pad=True`로 표시합니다. 이 표시는 나중에 손실을 계산할 때 마스크로 쓰입니다.

    > 아래 슬라이더는 **그림만 바꿉니다.** 학습에 쓰는 `cfg.chunk_size`(=20)는 0절 설정에서만 바뀝니다.
    """)
    return


@app.function
def sample_chunk(actions, t, chunk_size):
    import numpy as np

    idx = np.arange(t, t + chunk_size)  # action_delta_indices
    is_pad = idx >= len(actions)  # 에피소드 끝을 넘어선 부분
    idx = np.clip(idx, 0, len(actions) - 1)  # 마지막 프레임으로 채움
    return actions[idx], is_pad


@app.cell
def _(EP_LEN, mo):
    t_slider = mo.ui.slider(0, EP_LEN - 1, value=10, label="현재 시점 t", show_value=True)
    k_slider = mo.ui.slider(1, 60, value=20, label="그림용 k (학습 설정과 별개)", show_value=True)
    mo.hstack([t_slider, k_slider])
    return k_slider, t_slider


@app.cell
def _(OBS_R, demos, k_slider, plt, render, t_slider):
    _d = demos[0]
    _chunk, _pad = sample_chunk(_d["actions"], t_slider.value, k_slider.value)

    _fig, (_ax, _img_ax) = plt.subplots(1, 2, figsize=(9, 4.5))
    _img_ax.imshow(render(_d["states"][t_slider.value], _d["goal_x"]).transpose(1, 2, 0))
    _img_ax.set(title=f"관측 이미지 t={t_slider.value}", xticks=[], yticks=[])
    _ax.add_patch(plt.Circle((0, 0), OBS_R, color="tab:red", alpha=0.3))
    _ax.plot(*_d["states"].T, color="lightgray", label="시연 궤적")
    _ax.scatter(*_d["states"][t_slider.value], color="tab:blue", s=60, zorder=3, label="state_t")
    _ax.plot(*_chunk[~_pad].T, "o-", ms=3, color="tab:orange", label="action[t:t+k] (라벨)")
    if _pad.any():
        _ax.scatter(*_chunk[_pad].T, marker="x", color="k", s=40, label="padding (손실 제외)")
    _ax.set(xlim=(-1.2, 1.2), ylim=(-1.2, 1.2), aspect="equal",
            title=f"padding된 스텝 {_pad.sum()} / {k_slider.value}")
    _ax.legend(fontsize=7, loc="lower right")
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 1-2. 정규화와 배치 구성

    LeRobot은 `normalization_mapping`에 따라 `VISUAL`, `STATE`, `ACTION`을 모두 `MEAN_STD`로 정규화합니다.
    모델은 **정규화된 공간에서 학습하고 예측**하며, 로봇에 보내기 전에 역정규화합니다.
    VLM의 입출력 전·후처리 자리에 있는 단계지만, 토크나이저처럼 좌표를 **이산 토큰 ID로 바꾸지는 않습니다.**
    연속값의 스케일만 맞출 뿐이며, 모델은 처음부터 끝까지 연속값을 입력받고 연속값을 출력합니다.
    배치의 키 이름은 LeRobot 데이터셋과 같게 맞췄습니다.

    평균·표준편차 계산은 일반적인 전처리입니다. 여기서는 `make_batch`가 만드는
    **관측 이미지·상태, `(B, k, A)` 행동 청크, `(B, k)` 패딩 마스크**의 대응에 집중하세요.
    """)
    return


@app.cell
def _(N_EP, cfg, demos, device, np, render, torch):
    _all_s = np.concatenate([d["states"] for d in demos])
    _all_a = np.concatenate([d["actions"] for d in demos])
    stats = {
        "state": (_all_s.mean(0), _all_s.std(0)),
        "action": (_all_a.mean(0), _all_a.std(0)),
    }
    _IMNET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)[:, None, None]
    _IMNET_STD = np.array([0.229, 0.224, 0.225], np.float32)[:, None, None]


    def normalize(x, key):
        mean, std = stats[key]
        return torch.as_tensor((np.asarray(x) - mean) / std, dtype=torch.float32, device=device)


    def unnormalize(x, key):
        mean, std = stats[key]
        return x.detach().cpu().numpy() * std + mean


    def make_obs(states, goal_xs):
        """관측 배치: 이미지 + 로봇 상태 (추론 때도 사용)."""
        imgs = np.stack([(render(s, g) - _IMNET_MEAN) / _IMNET_STD for s, g in zip(states, goal_xs)])
        return {
            "observation.image": torch.as_tensor(imgs, device=device),
            "observation.state": normalize(np.stack(states), "state"),
        }


    def make_batch(batch_size, rng):
        """학습 배치: 관측 + 정답 행동 청크 + padding 마스크."""
        eps = rng.integers(0, N_EP, batch_size)
        ts = rng.integers(0, len(demos[0]["states"]), batch_size)
        chunks = [sample_chunk(demos[e]["actions"], t, cfg.chunk_size) for e, t in zip(eps, ts)]
        batch = make_obs([demos[e]["states"][t] for e, t in zip(eps, ts)], [demos[e]["goal_x"] for e in eps])
        batch["action"] = normalize(np.stack([c[0] for c in chunks]), "action")  # (B, k, A)
        batch["action_is_pad"] = torch.as_tensor(np.stack([c[1] for c in chunks]), device=device)  # (B, k)
        return batch


    {k: tuple(v.shape) for k, v in make_batch(4, np.random.default_rng(0)).items()}
    return make_batch, make_obs, unnormalize


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. 이미지를 토큰으로: ResNet18, 1×1 Conv, 2D 위치 임베딩

    | VLM (예: LLaVA) | ACT |
    |---|---|
    | ViT 패치 토큰 | ResNet `layer4` 특징맵의 **각 공간 위치**가 토큰 하나. RGB 픽셀이 아니라, stride 32로 줄어든 격자의 한 칸이 주변의 넓은 이미지 영역을 요약한 **512차원 벡터**입니다 |
    | MLP 프로젝터 | **1×1 Conv** (`encoder_img_feat_input_proj`) |
    | 비전 인코더를 보통 얼려 둠 | 백본도 **함께 학습** (`FrozenBatchNorm2d`로 BN 통계만 고정) |

    원본 코드:
    ```python
    backbone_model = torchvision.models.resnet18(weights=..., norm_layer=FrozenBatchNorm2d)
    self.backbone = IntermediateLayerGetter(backbone_model, return_layers={"layer4": "feature_map"})
    self.encoder_img_feat_input_proj = nn.Conv2d(512, dim_model, kernel_size=1)
    ```
    """)
    return


@app.cell
def _(FrozenBatchNorm2d, IntermediateLayerGetter, cfg, einops, nn, torch, torchvision):
    _backbone = IntermediateLayerGetter(
        torchvision.models.resnet18(weights=None, norm_layer=FrozenBatchNorm2d),
        return_layers={"layer4": "feature_map"},
    )
    _proj = nn.Conv2d(512, cfg.dim_model, kernel_size=1)

    with torch.no_grad():
        for _hw in [(480, 640), (cfg.image_size, cfg.image_size)]:
            _feat = _backbone(torch.zeros(1, 3, *_hw))["feature_map"]
            _tok = einops.rearrange(_proj(_feat), "b d h w -> (h w) b d")
            print(f"이미지 {_hw} → 특징맵 {tuple(_feat.shape)} → 토큰 {tuple(_tok.shape)}  (seq, batch, dim)")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    해상도 480×640인 카메라 한 대는 **토큰 300개**가 되고(stride 32), 카메라가 늘어나면 그만큼 시퀀스가 길어집니다.
    LeRobot 원본은 여러 카메라의 토큰을 시퀀스 축으로 이어 붙입니다(`for img in batch[OBS_IMAGES]: ...extend(...)`).

    ### 2-1. 2D sinusoidal 위치 임베딩

    `ACTSinusoidalPositionEmbedding2d`를 간결하게 다시 썼습니다. 채널의 절반은 y 좌표, 나머지 절반은 x 좌표를 인코딩합니다.
    좌표는 `[1/H, 1] × 2π` 범위로 정규화되므로 특징맵 크기와 관계없이 같은 범위를 씁니다.
    핵심은 이미지 토큰마다 공간 위치를 알려 준다는 점입니다. 유사도 그림은 부록 A에서 선택 실행할 수 있습니다.
    """)
    return


@app.cell
def _(math, nn, torch):
    class SinePosEmbed2d(nn.Module):
        def __init__(self, dim_per_axis):  # 원본: ACTSinusoidalPositionEmbedding2d(dim_model // 2)
            super().__init__()
            self.d = dim_per_axis

        def forward(self, feat):  # feat: (B, C, H, W) → (1, 2*d, H, W)
            _, _, H, W = feat.shape
            dev = feat.device
            y = torch.arange(1, H + 1, device=dev) / H * 2 * math.pi  # 1/H … 1 → [·, 2π]
            x = torch.arange(1, W + 1, device=dev) / W * 2 * math.pi
            inv_freq = 10000 ** (2 * (torch.arange(self.d, device=dev) // 2) / self.d)

            def enc(p):  # (L,) → (L, d): sin/cos 교차 배치
                a = p[:, None] / inv_freq
                return torch.stack((a[:, 0::2].sin(), a[:, 1::2].cos()), -1).flatten(1)

            ey = enc(y)[:, None, :].expand(H, W, -1)
            ex = enc(x)[None, :, :].expand(H, W, -1)
            return torch.cat([ey, ex], -1).permute(2, 0, 1).unsqueeze(0)
    return (SinePosEmbed2d,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. DETR 스타일 Transformer 인코더

    LLM과 다른 점이 두 가지 있습니다.

    1. **위치 임베딩을 모든 층에서 Q와 K에만 더합니다.** V에는 더하지 않으므로 위치 정보는
       "어디를 볼지"만 정하고 "무엇을 전달할지"에는 섞이지 않습니다.
    2. 기본값이 **post-norm**입니다(`pre_norm=False`). 요즘 LLM이 주로 쓰는 pre-norm과 다릅니다.

    원본 `ACTEncoderLayer.forward`의 핵심:
    ```python
    q = k = x if pos_embed is None else x + pos_embed
    x = self.self_attn(q, k, value=x, key_padding_mask=key_padding_mask)[0]
    ```
    텐서 레이아웃은 `nn.MultiheadAttention` 기본값인 **(seq, batch, dim)**입니다.
    """)
    return


@app.cell
def _(nn, torch):
    def make_ffn(c):
        return nn.Sequential(
            nn.Linear(c.dim_model, c.dim_feedforward), nn.ReLU(), nn.Dropout(c.dropout),
            nn.Linear(c.dim_feedforward, c.dim_model),
        )


    class EncoderLayer(nn.Module):
        def __init__(self, c):
            super().__init__()
            self.self_attn = nn.MultiheadAttention(c.dim_model, c.n_heads, dropout=c.dropout)
            self.ffn = make_ffn(c)
            self.norm1, self.norm2 = nn.LayerNorm(c.dim_model), nn.LayerNorm(c.dim_model)
            self.drop1, self.drop2 = nn.Dropout(c.dropout), nn.Dropout(c.dropout)

        def forward(self, x, pos=None, key_padding_mask=None):
            q = k = x if pos is None else x + pos  # ★ 위치 정보는 Q, K에만
            attn = self.self_attn(q, k, value=x, key_padding_mask=key_padding_mask)[0]
            x = self.norm1(x + self.drop1(attn))  # post-norm
            x = self.norm2(x + self.drop2(self.ffn(x)))
            return x


    class Encoder(nn.Module):
        def __init__(self, c, n_layers):
            super().__init__()
            self.layers = nn.ModuleList([EncoderLayer(c) for _ in range(n_layers)])

        def forward(self, x, pos=None, key_padding_mask=None):
            for layer in self.layers:
                x = layer(x, pos=pos, key_padding_mask=key_padding_mask)
            return x


    def sinusoidal_1d(n_pos, dim):
        """원본 create_sinusoidal_pos_embedding: Transformer 원 논문의 1D 위치 임베딩 (CVAE 인코더에서 사용)."""
        pos = torch.arange(n_pos, dtype=torch.float32)[:, None]
        a = pos / 10000 ** (2 * (torch.arange(dim) // 2) / dim)
        a[:, 0::2], a[:, 1::2] = a[:, 0::2].sin(), a[:, 1::2].cos()
        return a
    return Encoder, make_ffn, sinusoidal_1d


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. 디코더: 학습된 쿼리 k개로 k스텝을 한 번에 출력

    | LLM 디코더 | ACT 디코더 |
    |---|---|
    | 직전에 생성한 토큰을 다시 입력 (자기회귀) | 입력은 **0 벡터 k개**, 학습된 위치 쿼리 `decoder_pos_embed`(k, D)를 더함 |
    | causal mask 사용 | mask 없음: k개 쿼리가 서로 참조 |
    | k스텝에 forward k번 | **forward 1번** |

    BLIP-2의 Q-Former나 Flamingo의 Perceiver Resampler가 쓰는 **learnable query**와 같은 아이디어로,
    DETR의 object query에서 왔습니다. 쿼리 i는 "i스텝 뒤의 행동은 무엇인가?"를 묻는 질문입니다.

    **정답을 입력하지 않는데 쿼리 i가 어떻게 "i스텝 뒤"를 배울까요?** 디코더의 출력 슬롯 i는
    `action_head`를 거쳐 `actions[:, i]`가 되고, 7절의 손실이 이를 정답 청크의 i번째 행동과 비교합니다.
    쿼리에 처음부터 의미가 있는 것이 아니라, **슬롯 i ↔ 정답 i라는 손실의 짝짓기**가 쿼리 i에 그 역할을 학습시킵니다.

    | attention | Query | Key | Value |
    |---|---|---|---|
    | (1) self-attention | x + `query_pos` | x + `query_pos` | x |
    | (2) cross-attention | x + `query_pos` | memory + `memory_pos` | memory |
    """)
    return


@app.cell
def _(make_ffn, nn):
    class DecoderLayer(nn.Module):
        def __init__(self, c):
            super().__init__()
            self.self_attn = nn.MultiheadAttention(c.dim_model, c.n_heads, dropout=c.dropout)
            self.cross_attn = nn.MultiheadAttention(c.dim_model, c.n_heads, dropout=c.dropout)
            self.ffn = make_ffn(c)
            self.norm1, self.norm2, self.norm3 = (nn.LayerNorm(c.dim_model) for _ in range(3))
            self.drop1, self.drop2, self.drop3 = (nn.Dropout(c.dropout) for _ in range(3))

        def forward(self, x, memory, query_pos=None, memory_pos=None):
            add = lambda t, p: t if p is None else t + p  # noqa: E731
            # (1) 쿼리끼리 self-attention (causal mask 없음)
            q = k = add(x, query_pos)
            x = self.norm1(x + self.drop1(self.self_attn(q, k, value=x)[0]))
            # (2) 인코더 출력(memory)에 cross-attention: 이미지, 상태, z 토큰을 참조
            attn = self.cross_attn(query=add(x, query_pos), key=add(memory, memory_pos), value=memory)[0]
            x = self.norm2(x + self.drop2(attn))
            # (3) FFN
            x = self.norm3(x + self.drop3(self.ffn(x)))
            return x


    class Decoder(nn.Module):
        def __init__(self, c):
            super().__init__()
            self.layers = nn.ModuleList([DecoderLayer(c) for _ in range(c.n_decoder_layers)])
            self.norm = nn.LayerNorm(c.dim_model)  # 원본도 디코더 끝에 LayerNorm 사용

        def forward(self, x, memory, query_pos=None, memory_pos=None):
            for layer in self.layers:
                x = layer(x, memory, query_pos=query_pos, memory_pos=memory_pos)
            return self.norm(x)
    return (Decoder,)


@app.cell
def _(Decoder, cfg, nn, torch):
    # 실험: 디코더 입력은 전부 0이다. 위치 쿼리가 없으면 k개 출력이 서로 구별될 수 있을까?
    _dec = Decoder(cfg).eval()
    _memory = torch.randn(10, 1, cfg.dim_model)  # 인코더 출력 10토큰
    _x0 = torch.zeros(cfg.chunk_size, 1, cfg.dim_model)
    _qpos = nn.Embedding(cfg.chunk_size, cfg.dim_model).weight[:, None]
    with torch.no_grad():
        _no_q = _dec(_x0, _memory)
        _with_q = _dec(_x0, _memory, query_pos=_qpos)
    print("쿼리 없음 : 출력 간 최대 차이 =", (_no_q - _no_q[:1]).abs().max().item())
    print("쿼리 있음 : 출력 간 최대 차이 =", (_with_q - _with_q[:1]).abs().max().item())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    위치 쿼리가 없으면 k개 출력이 **모두 같습니다.** 입력이 전부 0이라 k개 위치를 구별할 정보가 없기 때문입니다.
    `decoder_pos_embed`가 "몇 번째 미래 스텝인지"를 알려 주는 유일한 신호입니다.

    이 실험은 행동을 잘 예측하는지가 아니라 **출력 슬롯들을 서로 구별할 수 있는지**를 봅니다.
    미학습 모델을 dropout을 끈 상태(`eval()`)로 썼지만, 결론은 학습한 뒤에도 같습니다.
    모든 입력이 같고 위치 신호가 없으면 attention은 어떤 가중치로도 k개 출력을 다르게 만들 수 없기 때문입니다.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. CVAE: 다봉 분포의 시연을 다루는 방법

    같은 장면에서 시연자는 왼쪽으로도, 오른쪽으로도 돌아갑니다. 단순 회귀는 두 경로를 **평균 내서 장애물로 곧장 돌진**합니다.
    VLM은 출력이 softmax 확률분포라 이런 다봉성을 자연스럽게 표현하지만, ACT는 연속값을 회귀하므로 별도 장치가 필요합니다.

    ### 5-1. 세 가지 구성 요소를 구분하기

    | 구성 요소 | 입력 | 언제 쓰나 | 역할 |
    |---|---|---|---|
    | **사후분포** $q(z \mid s, a_{t:t+k})$ = CVAE 인코더 | state + **정답 행동 청크** | **학습할 때만** | 이번 시연의 "스타일"(왼쪽/오른쪽)을 z로 요약 |
    | **사전분포** $p(z) = \mathcal N(0, I)$ | 없음 | 학습(KL 계산), 추론(z=0) | z가 모여 있어야 할 기준 분포 |
    | **정책** $\pi(a_{t:t+k} \mid \text{이미지}, s, z)$ = 메인 인코더·디코더 | 관측 + z | 학습과 추론 모두 | 실제 행동 청크 생성 |

    - **학습할 때**: q가 정답 청크를 보고 z를 만들고, 정책은 그 z를 조건으로 행동을 복원합니다.
      z가 "이번에는 왼쪽"이라고 알려 주므로 정책이 두 모드를 평균 낼 필요가 없습니다.
    - **추론할 때**: 정답이 없으므로 q를 쓰지 않고 **z = 0**(사전분포의 평균)으로 고정합니다. LeRobot 원본도 같습니다.
    - CVAE 인코더 입력은 `[CLS, state, action_1..k]`(BERT 스타일)이며 **이미지는 넣지 않습니다.**
      padding된 행동 토큰은 `key_padding_mask`로 가립니다(True = 참조에서 제외).

    ### 5-2. 정답을 보고 z를 만들면 누설(leak)이 아닌가?

    정답을 보는 것은 **학습 전용인 q뿐**이고, 실행되는 정책은 정답을 보지 않습니다.
    그래도 KL이 없다면 q가 정답 청크 전체를 z에 담아 넘기고, 정책은 관측을 무시한 채 z만 베끼게 됩니다.
    그러면 추론 때 z가 없어지는 순간 정책이 아무것도 못 합니다.

    **KL 항이 이것을 막습니다.** KL은 z가 사전분포에서 벗어난 만큼, 즉 z에 담은 정보량만큼 비용을 매깁니다.
    따라서 관측으로 알 수 있는 정보(목표 위치, 현재 위치)는 정책이 이미지·상태에서 직접 읽는 편이 싸고,
    z에는 **관측만으로 알 수 없는 정보**(이번 시연이 왼쪽인지 오른쪽인지)만 남는 것이 유리해집니다.
    `latent_dim=8`이라는 작은 크기도 병목 역할을 합니다.

    ### 5-3. 추론에서 z = 0으로 바꿔도 괜찮은가?

    KL이 학습 중의 z를 N(0, I) 근처로 모으므로, z = 0은 **정책이 학습 때 본 z 범위의 한가운데**입니다.
    다만 얼마나 잘 모이는지는 β(`kl_weight`)에 따라 달라집니다. β가 너무 작으면 z = 0이 학습 때 거의 나오지 않은 값일 수 있고,
    너무 크면 z가 아무 정보도 담지 않게 됩니다(posterior collapse). 9절에서 직접 확인합니다.

    ### 5-4. 재매개변수화(reparameterization)가 필요한 이유

    z를 $\mathcal N(\mu, \sigma^2)$에서 그대로 샘플링하면 샘플링 단계에서 그래디언트가 끊겨 CVAE 인코더를 학습할 수 없습니다.
    $z = \mu + \sigma \cdot \epsilon,\ \epsilon \sim \mathcal N(0, I)$로 쓰면 무작위성은 ε으로 분리되고,
    L1 손실의 그래디언트가 μ와 σ를 거쳐 CVAE 인코더까지 흐릅니다.

    원본 코드:
    ```python
    mu = latent_pdf_params[:, : latent_dim]
    log_sigma_x2 = latent_pdf_params[:, latent_dim :]             # 2·log σ
    latent_sample = mu + log_sigma_x2.div(2).exp() * torch.randn_like(mu)   # reparameterization
    ```
    """)
    return


@app.function
def kl_divergence(mu, log_sigma_x2):
    """KL( N(mu, σ²) || N(0, I) ): 차원별로 계산해 latent 축으로 합하고 배치 평균."""
    return (-0.5 * (1 + log_sigma_x2 - mu.pow(2) - log_sigma_x2.exp())).sum(-1).mean()


@app.cell
def _(torch):
    _z = torch.zeros(1, 8)
    print("mu=0, σ=1 일 때 KL :", kl_divergence(_z, _z).item(), "(사전분포와 같으면 0)")
    print("mu=2, σ=1 일 때 KL :", kl_divergence(_z + 2, _z).item())
    print("mu=0, σ=0.1 일 때 KL:", kl_divergence(_z, torch.full((1, 8), 2 * torch.log(torch.tensor(0.1)))).item())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. 전체 조립: `MiniACT`

    원본 `ACT.forward`와 같은 순서로 구현합니다. 원본에 있는 여러 카메라와 `env_state` 처리는 생략했습니다.

    ```
    (A) [학습 + use_vae] CVAE 인코더 → μ, log σ² → z        [추론] z = 0
    (B) 인코더 토큰 = [z 토큰, state 토큰, 이미지 토큰 ...]  + 위치 [학습된 1D 위치 2개, 2D sine]
    (C) Transformer 인코더 → memory
    (D) 0 벡터 k개 + 학습된 쿼리 → 디코더(memory에 cross-attention) → Linear → (B, k, action_dim)
    ```
    """)
    return


@app.cell
def _(
    Decoder,
    Encoder,
    FrozenBatchNorm2d,
    IntermediateLayerGetter,
    SinePosEmbed2d,
    einops,
    nn,
    sinusoidal_1d,
    torch,
    torchvision,
):
    class MiniACT(nn.Module):
        def __init__(self, c, pretrained=True):
            super().__init__()
            self.c = c
            D = c.dim_model
            # (A) CVAE 인코더: 학습할 때만 사용
            if c.use_vae:
                self.vae_encoder = Encoder(c, c.n_vae_encoder_layers)
                self.vae_cls = nn.Embedding(1, D)
                self.vae_state_proj = nn.Linear(c.state_dim, D)
                self.vae_action_proj = nn.Linear(c.action_dim, D)
                self.vae_latent_out = nn.Linear(D, 2 * c.latent_dim)  # → [μ, log σ²]
                self.register_buffer("vae_pos", sinusoidal_1d(2 + c.chunk_size, D)[:, None])  # (S,1,D) 고정
            # (B) 비전 백본 + 투영 + 2D 위치
            weights = "ResNet18_Weights.IMAGENET1K_V1" if pretrained else None
            resnet = torchvision.models.resnet18(weights=weights, norm_layer=FrozenBatchNorm2d)
            self.backbone = IntermediateLayerGetter(resnet, return_layers={"layer4": "feature_map"})
            self.img_proj = nn.Conv2d(512, D, kernel_size=1)
            self.img_pos = SinePosEmbed2d(D // 2)
            # (B) 1D 토큰(z, state) 투영 + 학습 가능한 위치
            self.latent_proj = nn.Linear(c.latent_dim, D)
            self.state_proj = nn.Linear(c.state_dim, D)
            self.pos_1d = nn.Embedding(2, D)
            # (C)(D) Transformer + 학습된 쿼리 + 행동 헤드
            self.encoder = Encoder(c, c.n_encoder_layers)
            self.decoder = Decoder(c)
            self.query_pos = nn.Embedding(c.chunk_size, D)  # DETR object query
            self.action_head = nn.Linear(D, c.action_dim)
            for p in [*self.encoder.parameters(), *self.decoder.parameters()]:
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)  # 원본 _reset_parameters

        def forward(self, batch, z=None):
            c = self.c
            state, img = batch["observation.state"], batch["observation.image"]
            B = state.shape[0]

            # ---------- (A) 잠재변수 z ----------
            if c.use_vae and self.training:
                vae_in = torch.cat([
                    self.vae_cls.weight.expand(B, 1, -1),  # CLS
                    self.vae_state_proj(state)[:, None],  # state
                    self.vae_action_proj(batch["action"]),  # 정답 행동 k개
                ], dim=1).transpose(0, 1)  # (2+k, B, D)
                pad = torch.cat([torch.zeros(B, 2, dtype=torch.bool, device=state.device),
                                 batch["action_is_pad"]], dim=1)  # (B, 2+k)
                cls_out = self.vae_encoder(vae_in, pos=self.vae_pos, key_padding_mask=pad)[0]
                mu, log_sigma_x2 = self.vae_latent_out(cls_out).chunk(2, dim=-1)
                z = mu + (log_sigma_x2 / 2).exp() * torch.randn_like(mu)  # reparameterization
            else:
                mu = log_sigma_x2 = None
                if z is None:
                    z = torch.zeros(B, c.latent_dim, device=state.device)  # 추론: z = 0

            # ---------- (B) 인코더 입력 토큰 ----------
            feat = self.backbone(img)["feature_map"]  # (B, 512, h, w)
            img_pos = einops.rearrange(self.img_pos(feat), "1 d h w -> (h w) 1 d")
            img_tok = einops.rearrange(self.img_proj(feat), "b d h w -> (h w) b d")
            tokens = torch.cat([self.latent_proj(z)[None], self.state_proj(state)[None], img_tok])  # (S, B, D)
            pos = torch.cat([self.pos_1d.weight[:, None], img_pos])  # (S, 1, D)

            # ---------- (C) 인코더 ----------
            memory = self.encoder(tokens, pos=pos)

            # ---------- (D) 디코더: 0 입력 + 학습된 쿼리 → k스텝을 한 번에 ----------
            x = torch.zeros(c.chunk_size, B, c.dim_model, device=state.device)
            out = self.decoder(x, memory, query_pos=self.query_pos.weight[:, None], memory_pos=pos)
            actions = self.action_head(out.transpose(0, 1))  # (B, k, action_dim)
            return actions, (mu, log_sigma_x2)
    return (MiniACT,)


@app.cell
def _(MiniACT, cfg, device, make_batch, np):
    _m = MiniACT(cfg).to(device).train()
    _b = make_batch(2, np.random.default_rng(1))
    _actions, (_mu, _ls) = _m(_b)
    _n_img = (cfg.image_size // 32) ** 2
    print(f"인코더 토큰 수 : 1(z) + 1(state) + {_n_img}(image) = {2 + _n_img}")
    print(f"CVAE 토큰 수   : 1(CLS) + 1(state) + {cfg.chunk_size}(action) = {2 + cfg.chunk_size}")
    print("출력 actions   :", tuple(_actions.shape), "  μ:", tuple(_mu.shape))
    _n_bb = sum(p.numel() for n, p in _m.named_parameters() if n.startswith("backbone"))
    _n_all = sum(p.numel() for p in _m.parameters())
    print(f"파라미터       : 전체 {_n_all / 1e6:.1f}M (백본 {_n_bb / 1e6:.1f}M)")
    _m.eval()
    print("eval 모드 μ    :", _m(_b)[1][0], "← 추론할 때는 CVAE 인코더를 쓰지 않음")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7. 손실 함수 (`ACTPolicy.forward`)

    $$\mathcal{L} = \underbrace{\text{L1}(\hat a, a)\big|_{\text{padding 제외}}}_{\text{재구성}} + \beta \cdot \underbrace{D_{KL}\big(q(z|a,s)\,\|\,\mathcal N(0,I)\big)}_{\text{정규화}},\qquad \beta=\texttt{kl\_weight}$$

    - L2 대신 **L1**을 씁니다. 미세 조작에서 더 정밀했다고 보고되었습니다.
    - `action_is_pad`인 스텝은 손실에서 빠집니다. 빼지 않으면 에피소드 끝의 "정지" 동작을 과도하게 학습합니다.
    - 큰 β는 z를 N(0, I)에 가깝게 만들어 **추론 때 z=0을 써도 학습 때와 크게 다르지 않게** 합니다.
    """)
    return


@app.cell
def _(F):
    def act_loss(model, batch):
        actions_hat, (mu, log_sigma_x2) = model(batch)
        abs_err = F.l1_loss(actions_hat, batch["action"], reduction="none")  # (B, k, A)
        valid = ~batch["action_is_pad"].unsqueeze(-1)  # (B, k, 1)
        l1 = (abs_err * valid).sum() / (valid.sum() * abs_err.shape[-1]).clamp_min(1)
        if mu is None:  # use_vae=False
            return l1, {"l1": l1.item()}
        kld = kl_divergence(mu, log_sigma_x2)
        return l1 + model.c.kl_weight * kld, {"l1": l1.item(), "kld": kld.item()}
    return (act_loss,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 8. 학습: CVAE를 쓰는 ACT와 쓰지 않는 ACT 비교

    세 모델을 같은 조건으로 학습합니다: **CVAE β=10**(LeRobot 기본값), **CVAE β=0.01**, **VAE 없음**.
    옵티마이저는 원본 `get_optim_params`처럼 **백본만 다른 학습률**을 쓰는 파라미터 그룹으로 나눕니다.
    원본 기본 학습률(1e-5)은 이 작은 과제에는 느려서 1e-4를 씁니다.
    학습 시간은 장치와 스텝 수에 따라 달라지며, CPU에서는 오래 걸릴 수 있습니다.

    학습 루프는 보조 코드로 접어 두었습니다. 우선 **CVAE 유무와 β가 L1·KL에 주는 영향**을
    비교하세요. AdamW 설정·진행 표시·그래프 평활화의 세부 구현은 선택해서 읽으면 됩니다.
    """)
    return


@app.cell
def _(mo):
    steps_ui = mo.ui.number(start=200, stop=10000, step=100, value=2000, label="학습 스텝")
    train_btn = mo.ui.run_button(label="▶ 세 모델 학습 시작")
    mo.hstack([steps_ui, train_btn], justify="start")
    return steps_ui, train_btn


@app.cell(hide_code=True)
def _(
    MiniACT,
    act_loss,
    cfg,
    device,
    make_batch,
    mo,
    np,
    replace,
    steps_ui,
    torch,
    train_btn,
):
    mo.stop(mo.running_in_notebook() and not train_btn.value, mo.md("⬆ 버튼을 눌러 학습을 시작하세요."))

    variants = {
        "CVAE β=10 (기본값)": dict(use_vae=True, kl_weight=10.0),
        "CVAE β=0.01": dict(use_vae=True, kl_weight=0.01),
        "no VAE": dict(use_vae=False),
    }
    models, logs = {}, {}
    for _name, _kw in variants.items():
        torch.manual_seed(0)
        _model = MiniACT(replace(cfg, **_kw)).to(device).train()
        _optim = torch.optim.AdamW([
            {"params": [p for n, p in _model.named_parameters() if not n.startswith("backbone")]},
            {"params": [p for n, p in _model.named_parameters() if n.startswith("backbone")], "lr": 1e-5},
        ], lr=1e-4, weight_decay=1e-4)
        _rng = np.random.default_rng(0)
        logs[_name] = []
        for _ in mo.status.progress_bar(range(steps_ui.value), title=_name):
            _loss, _info = act_loss(_model, make_batch(64, _rng))
            _optim.zero_grad()
            _loss.backward()
            _optim.step()
            logs[_name].append(_info)
        models[_name] = _model.eval()
    return logs, models


@app.cell(hide_code=True)
def _(logs, np, plt):
    def _smooth(v, w=50):
        return np.convolve(v, np.ones(w) / w, mode="valid")

    _fig, _axes = plt.subplots(1, 2, figsize=(9, 3))
    for _name, _log in logs.items():
        _axes[0].plot(_smooth([d["l1"] for d in _log]), label=_name)
        if "kld" in _log[0]:
            _axes[1].plot(_smooth([d["kld"] for d in _log]), label=_name)
    _axes[0].set(title="L1 (재구성)", yscale="log", xlabel="step")
    _axes[1].set(title="KL (CVAE)", yscale="log", xlabel="step")
    _axes[0].legend()
    _axes[1].legend()
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 9. z가 무엇을 배웠는지 확인

    시작 위치(t=0)에서 **청크 하나**를 예측해 봅니다(open-loop).

    - **no VAE**: 왼쪽과 오른쪽을 평균 낸 경로가 나오는지 봅니다.
    - **CVAE + z ~ N(0, I) 샘플링**: z마다 다른 "스타일"(왼쪽/오른쪽)이 나오는지 봅니다.
    - **CVAE + z = 0**: 실제 추론에서 쓰는 값입니다.
    """)
    return


@app.cell
def _(OBS_R, cfg, device, make_obs, models, plt, torch, unnormalize):
    _start, _goal_x = (0.0, -1.0), 0.1
    _obs = make_obs([_start], [_goal_x])
    _fig, _axes = plt.subplots(1, len(models), figsize=(4.5 * len(models), 4.5))
    torch.manual_seed(0)
    with torch.no_grad():
        for _ax, (_name, _m) in zip(_axes, models.items()):
            _ax.add_patch(plt.Circle((0, 0), OBS_R, color="tab:red", alpha=0.3))
            if _m.c.use_vae:
                for _ in range(15):
                    _z = torch.randn(1, cfg.latent_dim, device=device)
                    _ax.plot(*unnormalize(_m(_obs, z=_z)[0][0], "action").T, color="tab:blue", alpha=0.35, lw=1)
                _ax.plot([], [], color="tab:blue", alpha=0.5, label="z ~ N(0, I)")
            _ax.plot(*unnormalize(_m(_obs)[0][0], "action").T, "k-o", ms=3, label="z = 0 (추론 기본값)")
            _ax.scatter(*_start, color="tab:blue", s=60, zorder=3)
            _ax.set(title=_name, xlim=(-1.2, 1.2), ylim=(-1.2, 1.2), aspect="equal")
            _ax.legend(fontsize=8, loc="upper left")
    _fig.suptitle(f"t=0에서 예측한 행동 청크 ({cfg.chunk_size}스텝)")
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **결과 읽는 법**

    - **β=0.01**: z 샘플에 따라 청크가 왼쪽과 오른쪽으로 갈라지면 z가 시연의 "스타일"을 학습한 것입니다.
      대신 KL이 커서 z=0이 학습 분포의 어디쯤에 해당하는지 보장되지 않습니다. z=0이 두 모드 사이에 놓이면 위험할 수 있습니다.
    - **β=10**: 학습 곡선에서 KL이 0으로 수렴했다면 **posterior collapse**입니다. z가 아무 정보도 담지 않으므로
      z를 샘플링해도 결과가 같고, 결국 VAE 없는 모델처럼 동작합니다.
    - **no VAE**: L1 손실은 **중앙값**을 학습합니다. 50:50에 가까운 다봉 분포에서는 두 모드 사이에 걸치거나
      데이터가 조금 더 많은 쪽 모드에 붙습니다.
    - 실제 ACT도 직진 구간이 끝난 뒤에는 **관측(상태, 이미지)만으로 어느 쪽으로 가는지 드러나므로**,
      폐루프에서 매번 다시 예측하면서 한쪽 모드로 정해지는 경우가 많습니다. 10절에서 확인해 보세요.

    > 정리: ACT의 CVAE는 "추론 때 다양한 행동을 샘플링하려는 장치"가 아닙니다.
    > **학습할 때 회귀 손실이 모드를 평균 내지 않도록** 돕는 장치이며, 그 효과는 β에 크게 좌우됩니다.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 10. 추론: 행동 큐와 Temporal Ensembling

    ### 10-1. 행동 큐 (`ACTPolicy.select_action`)
    큐가 비었을 때만 모델을 호출하고, 청크 앞쪽 `n_action_steps`개를 큐에 넣은 뒤 한 스텝씩 꺼냅니다.

    ### 10-2. Temporal Ensembling (`ACTTemporalEnsembler`)
    **매 스텝** 모델을 호출하고, 시점 t에 대해 과거 청크들이 내놓은 예측을 가중 평균합니다.
    가중치는 $w_i = e^{-m\,i}$이고 $i=0$이 **가장 오래된** 예측입니다. 원 논문의 m=0.01이면 오래된 예측에 약간 더 무게를 둡니다.

    **서로 다른 시각의 행동을 섞는 것이 아닙니다.** 평균하는 대상은 모두 **같은 실행 시각 t의 행동**입니다.
    시점 s에 예측한 청크에서 시각 t의 행동은 인덱스 `t − s`에 있습니다(코드의 `ch[:, self.t - s]`).
    예를 들어 k=4일 때 **시각 t=3에 실행할 행동**은 다음 네 예측을 평균합니다.

    | 청크를 예측한 시점 s | 청크 안의 인덱스 t − s | 가중치 순번 i |
    |---|---|---|
    | 0 | 3 (그 청크의 마지막) | 0 (가장 오래됨) |
    | 1 | 2 | 1 |
    | 2 | 1 | 2 |
    | 3 | 0 (방금 예측한 청크의 첫 행동) | 3 (가장 최신) |

    t − s ≥ k인 청크는 시각 t를 포함하지 않으므로 제외됩니다.

    아래 `SimpleTemporalEnsembler`는 논문 Algorithm 2를 그대로 옮긴 버전입니다. 지난 청크들을 저장해 두고 매번 평균을 냅니다.
    LeRobot은 같은 값을 **온라인 누적 평균**으로 계산해 메모리를 아낍니다. 바로 다음 셀에서 두 결과가 같은지 확인합니다.
    """)
    return


@app.cell
def _(deque, torch):
    class SimpleTemporalEnsembler:
        def __init__(self, coeff, chunk_size):
            self.coeff, self.k = coeff, chunk_size
            self.reset()

        def reset(self):
            self.t, self.history = 0, []  # [(청크를 예측한 시점, (B, k, A) 청크), ...]

        def update(self, chunk):
            self.history.append((self.t, chunk))
            self.history = [(s, ch) for s, ch in self.history if self.t - s < self.k]  # 시점 t를 포함하는 청크만
            preds = torch.stack([ch[:, self.t - s] for s, ch in self.history])  # (N, B, A), 오래된 순서
            w = torch.exp(-self.coeff * torch.arange(len(preds), device=chunk.device, dtype=chunk.dtype))
            self.t += 1
            return (w[:, None, None] * preds).sum(0) / w.sum()


    class MiniACTPolicy:
        def __init__(self, model, n_action_steps, te_coeff=None):
            self.model, self.n, self.te_coeff = model, n_action_steps, te_coeff
            if te_coeff is not None:
                self.ensembler = SimpleTemporalEnsembler(te_coeff, model.c.chunk_size)
            self.reset()

        def reset(self):  # 에피소드가 시작될 때마다 호출
            self.queue = deque(maxlen=self.n)
            if self.te_coeff is not None:
                self.ensembler.reset()

        @torch.no_grad()
        def select_action(self, obs):
            self.model.eval()
            if self.te_coeff is not None:  # 매 스텝 추론 후 앙상블
                return self.ensembler.update(self.model(obs)[0])
            if len(self.queue) == 0:  # 큐가 비었을 때만 추론
                actions = self.model(obs)[0][:, : self.n]  # (B, n, A)
                self.queue.extend(actions.transpose(0, 1))  # n개의 (B, A)
            return self.queue.popleft()
    return MiniACTPolicy, SimpleTemporalEnsembler


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 10-3. 폐루프(closed-loop) 롤아웃

    매 스텝 `관측 → select_action → 역정규화 → 환경 실행` 루프를 돕니다.
    관측은 매 스텝 만들어 넘기지만, **행동 큐 방식에서는 큐가 비었을 때만 그 관측으로 새 청크를 예측**합니다.
    큐가 남아 있는 동안에는 관측을 쓰지 않고 이전에 예측한 행동을 차례로 꺼냅니다.
    Temporal Ensembling은 매 스텝 새 관측으로 예측합니다.
    실제 로봇의 추종 오차를 흉내 내기 위해 실행 위치에 작은 노이즈를 더합니다.
    환경 실행·충돌 판정·그래프 코드는 접어 두었습니다. 조작할 핵심은
    **`n_action_steps`와 Temporal Ensembling 사용 여부**입니다.
    설정을 바꾸면서 **충돌률**과 **떨림(jerk: 위치의 2차 차분 크기)**을 비교해 보세요.

    - `n_action_steps=1`이면 매 스텝 다시 계획합니다(반응은 빠르지만 모드가 자주 바뀔 수 있음).
    - `n_action_steps`가 크면 청크 경계에서 끊김이 생길 수 있습니다.
    - Temporal Ensembling은 `n_action_steps=1`로 실행하고 과거 예측을 평균합니다(원본 `ACTConfig`도 이 조합을 강제합니다).
      **체크하면 `n_action_steps` 슬라이더는 무시됩니다.**
    """)
    return


@app.cell
def _(cfg, mo, models):
    model_ui = mo.ui.dropdown(list(models), value=list(models)[0], label="모델")
    n_steps_ui = mo.ui.slider(1, cfg.chunk_size, value=cfg.chunk_size, label="n_action_steps", show_value=True)
    te_ui = mo.ui.checkbox(label="Temporal Ensembling (m=0.01, 켜면 n_action_steps=1 고정)")
    noise_ui = mo.ui.slider(0.0, 0.05, step=0.005, value=0.01, label="실행 노이즈", show_value=True)
    mo.vstack([mo.hstack([model_ui, te_ui]), mo.hstack([n_steps_ui, noise_ui])])
    return model_ui, n_steps_ui, noise_ui, te_ui


@app.cell(hide_code=True)
def _(
    EP_LEN,
    MiniACTPolicy,
    OBS_R,
    make_obs,
    model_ui,
    models,
    n_steps_ui,
    noise_ui,
    np,
    plt,
    te_ui,
    unnormalize,
):
    def rollout(policy, goal_x, rng, noise):
        pos = np.array([0.0, -1.0])
        path = [pos]
        policy.reset()
        for _t in range(EP_LEN):
            action = policy.select_action(make_obs([pos], [goal_x]))  # (1, A), 정규화된 공간
            pos = unnormalize(action, "action")[0] + rng.normal(0, noise, 2)  # 역정규화 → 로봇 실행
            path.append(pos)
        return np.array(path)


    _policy = MiniACTPolicy(models[model_ui.value], n_action_steps=1 if te_ui.value else n_steps_ui.value,
                            te_coeff=0.01 if te_ui.value else None)
    _rng = np.random.default_rng(42)
    _paths = [rollout(_policy, g, _rng, noise_ui.value) for g in np.linspace(-0.3, 0.3, 12)]
    _collide = [np.linalg.norm(p, axis=1).min() < OBS_R for p in _paths]
    _jerk = np.mean([np.linalg.norm(np.diff(p, 2, axis=0), axis=1).mean() for p in _paths])
    _reach = np.mean([np.linalg.norm(p[-1] - (g, 1.0)) < 0.15 for p, g in zip(_paths, np.linspace(-0.3, 0.3, 12))])

    _fig, _ax = plt.subplots(figsize=(4.5, 4.5))
    _ax.add_patch(plt.Circle((0, 0), OBS_R, color="tab:red", alpha=0.3))
    for _p, _c in zip(_paths, _collide):
        _ax.plot(*_p.T, color="tab:red" if _c else "tab:blue", alpha=0.6, lw=1)
    _ax.set(xlim=(-1.2, 1.2), ylim=(-1.2, 1.4), aspect="equal",
            title=f"충돌 {np.mean(_collide):.0%} · 도달 {_reach:.0%} · jerk {_jerk * 1e3:.1f}e-3")
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 11. 실습 과제

    1. **청크 크기**: `MiniACTConfig(chunk_size=1, n_action_steps=1)`로 학습하면 한 스텝씩 예측하는 일반 행동 복제가 됩니다.
       충돌률을 비교하고 복합 오차와 다봉성 관점에서 설명해 보세요.
    2. **`kl_weight`**: `variants`에 β=0.1, 1을 추가해 z 샘플이 갈라지기 시작하는 β를 찾아보세요.
       각 모델로 10절 롤아웃을 돌려 충돌률과 도달률을 비교하세요. 또 8절 그래프에서 원본 기본값 β=10의 KL이
       0 근처로 떨어졌는지(posterior collapse) 확인하고, 그렇다면 왜 그런지, 아니라면 z가 무엇을 담았는지 설명해 보세요.
    3. **위치 쿼리 제거**: `MiniACT.forward`에서 `query_pos=None`으로 바꾸고 학습하면 어떻게 될까요? (4절 실험 참고)
    4. **Temporal Ensembling 계수**: `m`을 음수(최신 예측에 무게)로 바꿔 jerk와 충돌률을 비교해 보세요.
       원본 docstring에도 최신 예측에 무게를 크게 두면 청크의 이점이 줄어든다는 실험 결과가 언급되어 있습니다.
    5. **`action_is_pad` 마스크 제거**: 손실에서 마스크를 빼면 에피소드 끝 근처의 행동이 어떻게 바뀌나요?
    6. **여러 카메라**: 같은 장면을 다른 색으로 그린 두 번째 이미지를 추가하고, 원본처럼 카메라별 토큰을 시퀀스에 이어 붙여 보세요.
       인코더 토큰 수는 어떻게 바뀌나요?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    _intro = mo.md(r"""
    ## 선택 심화·검증 부록

    본문 실습을 마친 뒤 필요한 항목만 실행하세요. 아래 셀은 본문 모델 학습·추론에 사용되지 않습니다.

    ### A. 2D 위치 임베딩의 유사도

    공간적으로 가까운 위치의 임베딩이 얼마나 비슷한지 확인하는 보조 실험입니다.
    """)
    pe_demo_btn = mo.ui.run_button(label="위치 임베딩 유사도 보기")
    mo.vstack([_intro, pe_demo_btn])
    return (pe_demo_btn,)


@app.cell(hide_code=True)
def _(SinePosEmbed2d, mo, pe_demo_btn, plt, torch):
    mo.stop(not pe_demo_btn.value)
    # 기준 위치의 임베딩과 다른 위치 임베딩 사이의 코사인 유사도 → 위치 정보가 공간 구조를 담는지 확인
    _pe = SinePosEmbed2d(256)(torch.zeros(1, 1, 15, 20))[0]  # (512, 15, 20), 480×640 이미지 크기 기준
    _pe = _pe.flatten(1).T  # (300, 512)
    _pe = _pe / _pe.norm(dim=-1, keepdim=True)
    _fig, _axes = plt.subplots(1, 3, figsize=(10, 2.6))
    for _ax, (_r, _c) in zip(_axes, [(2, 3), (7, 10), (12, 17)]):
        _sim = (_pe @ _pe[_r * 20 + _c]).reshape(15, 20)
        _ax.imshow(_sim, cmap="viridis")
        _ax.scatter(_c, _r, color="red", marker="x")
        _ax.set(title=f"기준 위치 (y={_r}, x={_c})", xticks=[], yticks=[])
    _fig.suptitle("2D 위치 임베딩의 코사인 유사도: 가까운 위치일수록 비슷함")
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    _intro = mo.md(r"""
    ### B. Temporal Ensembling 구현 동등성 검증

    본문의 직관적인 구현과 LeRobot 방식의 온라인 누적 구현이 같은 값을 계산하는지 비교합니다.
    ACT의 원리를 이해하는 데 두 구현을 모두 읽을 필요는 없습니다.
    """)
    te_check_btn = mo.ui.run_button(label="두 Ensembler 구현 비교")
    mo.vstack([_intro, te_check_btn])
    return (te_check_btn,)


@app.cell(hide_code=True)
def _(SimpleTemporalEnsembler, mo, te_check_btn, torch):
    mo.stop(not te_check_btn.value)
    class LeRobotTemporalEnsembler:
        """LeRobot 원본 ACTTemporalEnsembler (주석과 device 처리만 줄임): 온라인 누적 가중 평균."""

        def __init__(self, coeff, chunk_size):
            self.chunk_size = chunk_size
            self.ensemble_weights = torch.exp(-coeff * torch.arange(chunk_size))
            self.ensemble_weights_cumsum = torch.cumsum(self.ensemble_weights, dim=0)
            self.ensembled_actions = None

        def update(self, actions):
            if self.ensembled_actions is None:
                self.ensembled_actions = actions.clone()
                self.ensembled_actions_count = torch.ones((self.chunk_size, 1), dtype=torch.long)
            else:
                self.ensembled_actions *= self.ensemble_weights_cumsum[self.ensembled_actions_count - 1]
                self.ensembled_actions += actions[:, :-1] * self.ensemble_weights[self.ensembled_actions_count]
                self.ensembled_actions /= self.ensemble_weights_cumsum[self.ensembled_actions_count]
                self.ensembled_actions_count = torch.clamp(self.ensembled_actions_count + 1, max=self.chunk_size)
                self.ensembled_actions = torch.cat([self.ensembled_actions, actions[:, -1:]], dim=1)
                self.ensembled_actions_count = torch.cat(
                    [self.ensembled_actions_count, torch.ones_like(self.ensembled_actions_count[-1:])])
            action = self.ensembled_actions[:, 0]
            self.ensembled_actions = self.ensembled_actions[:, 1:]
            self.ensembled_actions_count = self.ensembled_actions_count[1:]
            return action


    _simple, _online = SimpleTemporalEnsembler(0.01, 10), LeRobotTemporalEnsembler(0.01, 10)
    _max_diff = 0.0
    for _ in range(30):
        _chunk = torch.randn(2, 10, 3, dtype=torch.float64)
        _max_diff = max(_max_diff, (_simple.update(_chunk) - _online.update(_chunk)).abs().max().item())
    print(f"30스텝 동안 두 구현의 최대 차이: {_max_diff:.2e}  → 같은 값을 계산함")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 후속 학습: ACT에서 VLA로

    ACT 핵심 실습 범위 밖의 확장 과제입니다. ResNet 대신 ViT(CLIP) 패치 토큰을 넣고
    언어 지시 토큰을 추가한다면, SmolVLA나 π0 같은 VLA와 무엇이 같고 무엇이 다를지 정리해 보세요.
    """)
    return


if __name__ == "__main__":
    app.run()
