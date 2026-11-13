# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#     "marimo>=0.25.2,<0.26",
#     "torch>=2.12,<2.13",
#     "torchvision>=0.27,<0.28",
#     "einops>=0.8",
#     "numpy",
#     "matplotlib",
# ]
# ///
#
# 이 노트북의 ACT 구현 코드는 LeRobot(https://github.com/huggingface/lerobot,
# src/lerobot/policies/act, revision e6ffde6)을 바탕으로 단순화·번역한 것입니다.
# 원본: Copyright 2024 Tony Z. Zhao and The HuggingFace Inc. team.
# Licensed under the Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0).
# 변경 사항: lerobot 패키지 의존성 제거(PreTrainedPolicy·processor 대체), 한국어 주석 추가,
# 실습용 장난감 환경·분석 코드 추가.

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="ACT 정책 실습")


@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # ACT (Action Chunking with Transformers) 실습

    ACT는 [Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware](https://arxiv.org/abs/2304.13705)
    (Zhao et al., 2023)에서 제안된 **모방 학습(imitation learning) 정책**입니다. LeRobot에서 처음 로봇 정책을
    학습할 때 가장 먼저 권하는 모델이기도 합니다.

    이 노트북은 LeRobot의 ACT 구현([`src/lerobot/policies/act`](https://github.com/huggingface/lerobot/tree/main/src/lerobot/policies/act))을
    거의 그대로 옮겨 와서, 작은 장난감 과제에서 **직접 학습시키고 뜯어보며** 이해하는 것을 목표로 합니다.

    ## 학습 목표

    1. ACT의 입력(이미지·관절 상태)과 출력(**행동 청크**, action chunk)이 무엇인지 설명할 수 있다.
    2. 데이터셋에서 행동 청크와 `action_is_pad` 마스크가 어떻게 만들어지는지 안다.
    3. ResNet 백본 → Transformer 인코더 → DETR식 디코더로 이어지는 **토큰 흐름과 텐서 모양**을 추적할 수 있다.
    4. 학습 때만 쓰는 **CVAE 인코더**와 잠재 변수 $z$의 역할, 그리고 `kl_weight`가 미치는 영향을 실험으로 확인한다.
    5. 추론 시 **행동 큐(`n_action_steps`)** 와 **시간 앙상블(temporal ensembling)** 이 실제 동작을 어떻게 바꾸는지 비교한다.

    ## 원본 코드와의 대응

    | 원본 파일 | 이 노트북에서 다루는 내용 |
    |---|---|
    | `configuration_act.py` | `ACTConfig` — 하이퍼파라미터와 입력 검증 (3장) |
    | `modeling_act.py` | `ACT`, `ACTEncoder(Layer)`, `ACTDecoder(Layer)`, 위치 임베딩 (4–5장) |
    | `modeling_act.py` | `ACTPolicy` — 손실, `select_action`, 행동 큐 (6장) |
    | `modeling_act.py` | `ACTTemporalEnsembler` — 시간 앙상블 (6장, 9장) |
    | `processor_act.py` | 입력 정규화·출력 역정규화 파이프라인 (2장) |

    코드는 [Apache License 2.0](http://www.apache.org/licenses/LICENSE-2.0)으로 배포된 원본
    (Copyright 2024 Tony Z. Zhao and The HuggingFace Inc. team, revision `e6ffde6`)을 단순화한 것입니다.
    `lerobot` 패키지는 설치하지 않으며, 대신 `PreTrainedPolicy`·processor 같은 주변 부품을 최소한으로 다시 만듭니다.

    > **실행 안내** — 이 디렉토리에서 `uv sync --locked` 후 `uv run marimo edit act_policy_lab.py`.
    > 7장의 학습은 GPU에서 설정당 1분 안팎, CPU에서는 수 분 이상 걸립니다.
    """)
    return


@app.cell
def _():
    import math
    import os
    import time
    from collections import deque
    from collections.abc import Callable
    from contextlib import contextmanager
    from dataclasses import dataclass, field
    from enum import Enum
    from itertools import chain
    from typing import Any

    import einops
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    import torch.nn.functional as F  # noqa: N812
    import torchvision
    from torch import Tensor, nn
    from torchvision.models._utils import IntermediateLayerGetter
    from torchvision.ops.misc import FrozenBatchNorm2d

    return (
        Any,
        Callable,
        Enum,
        F,
        FrozenBatchNorm2d,
        IntermediateLayerGetter,
        Tensor,
        chain,
        contextmanager,
        dataclass,
        deque,
        einops,
        field,
        math,
        nn,
        np,
        os,
        plt,
        time,
        torch,
        torchvision,
    )


@app.cell
def _(os, torch):
    # 실행 장치와 재현성 설정
    if torch.cuda.is_available():
        DEVICE = torch.device("cuda")
    elif torch.backends.mps.is_available():
        DEVICE = torch.device("mps")
    else:
        DEVICE = torch.device("cpu")

    # 배치(무인) 검증용 스위치: ACT_LAB_AUTORUN=1 이면 학습 폼을 누르지 않아도 기본값으로 학습합니다.
    AUTORUN = os.environ.get("ACT_LAB_AUTORUN") == "1"
    SMOKE_STEPS = int(os.environ.get("ACT_LAB_STEPS", "0"))  # 0 이면 폼 값 사용

    torch.manual_seed(0)
    print(f"device = {DEVICE}, torch = {torch.__version__}")
    return AUTORUN, DEVICE, SMOKE_STEPS


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 0. ACT 한눈에 보기

    ACT는 현재 관측 하나를 보고 **앞으로의 행동 $k$개를 한 번에** 예측합니다(행동 청크).
    매 스텝 행동 1개만 예측하는 정책보다 오차가 덜 누적되고, 사람이 시연할 때 생기는
    멈칫거림 같은 비마르코프적 행동도 따라 하기 쉽습니다.

    원본 `ACT` 클래스의 docstring에 있는 그림을 그대로 옮기면 다음과 같습니다.

    ```text
                                 Transformer
                                 Used alone for inference
                                 (acts as VAE decoder
                                  during training)
                                ┌───────────────────────┐
                                │             Outputs   │
                                │                ▲      │
                                │     ┌─────►┌───────┐  │
                   ┌──────┐     │     │      │Transf.│  │
                   │      │     │     ├─────►│decoder│  │
              ┌────┴────┐ │     │     │      │       │  │
              │         │ │     │ ┌───┴───┬─►│       │  │
              │ VAE     │ │     │ │       │  └───────┘  │
              │ encoder │ │     │ │Transf.│             │
              │         │ │     │ │encoder│             │
              └───▲─────┘ │     │ │       │             │
                  │       │     │ └▲──▲─▲─┘             │
                  │       │     │  │  │ │               │
                inputs    └─────┼──┘  │ image emb.      │
                                │    state emb.         │
                                └───────────────────────┘
    ```

    - **VAE 인코더** (학습 때만): `[CLS, 로봇 상태, 정답 행동 k개]`를 읽어 잠재 분포 $q(z\mid a_{1:k}, s)$의 평균·분산을 냅니다.
    - **Transformer 인코더**: `[z 토큰, 상태 토큰, 이미지 특징 토큰들]`을 섞습니다.
    - **Transformer 디코더**: 학습되는 위치 임베딩 $k$개(DETR의 object query와 같은 역할)가 인코더 출력에 cross-attention 하여 행동 $k$개를 냅니다.
    - **추론 때는** VAE 인코더를 쓰지 않고 $z = 0$(사전 분포의 평균)을 넣습니다.

    | 이름 | 원본 코드에서의 뜻 |
    |---|---|
    | `vae_encoder` | 행동 시퀀스를 잠재 변수로 압축하는 **VAE의 인코더** |
    | `encoder` / `decoder` | VAE의 **디코더 역할**을 하는 Transformer의 인코더/디코더 (`vae_` 접두사가 없음에 주의) |
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. 실습용 장난감 과제: 장애물 돌아가기

    실제 로봇 데이터 대신, 몇 초 안에 만들 수 있는 2D 과제를 사용합니다.

    - 로봇(파란 점)은 왼쪽 `(-0.8, 0)`에서 출발해 오른쪽 목표(초록 점)로 가야 합니다.
      목표의 y 좌표는 `{-0.25, 0, 0.25}` 중 하나이고 **이미지로만** 알 수 있습니다.
    - 가운데에 반지름 0.3의 장애물(빨간 원)이 있습니다.
    - 시연자는 매번 **위로 돌지 아래로 돌지**를 무작위로 고르고, 시연 길이(=속도)도 30~70 스텝 사이에서 무작위로 고릅니다.

    마지막 두 가지가 중요합니다. **같은 관측에서 시작해도 시연마다 미래 행동이 다릅니다.**
    출발 지점에서는 위/아래가, 경로 중간에서는 속도가 관측에 드러나지 않습니다.
    사람 시연 데이터의 "스타일 차이"를 흉내 낸 것으로, 뒤에서 CVAE의 잠재 변수 $z$가 이 차이를 담는지 확인합니다.

    관측과 행동은 LeRobot 데이터셋의 키 이름을 그대로 씁니다.

    | 키 | 모양 | 내용 |
    |---|---|---|
    | `observation.state` | `(2,)` | 로봇의 현재 위치 (관절 상태에 해당) |
    | `observation.images.top` | `(3, 96, 96)` | 위에서 본 카메라 이미지. R=장애물, G=목표, B=로봇 |
    | `action` | `(2,)` | 다음 스텝의 **목표 위치** (절대 좌표. ALOHA의 목표 관절 각도에 해당) |
    """)
    return


@app.cell
def _():
    # lerobot.utils.constants 와 같은 키 이름
    OBS_STATE = "observation.state"
    OBS_ENV_STATE = "observation.environment_state"
    OBS_IMAGES = "observation.images"
    ACTION = "action"
    IMG_KEY = "observation.images.top"  # 카메라가 하나뿐인 실습용 이미지 키
    return ACTION, IMG_KEY, OBS_ENV_STATE, OBS_IMAGES, OBS_STATE


@app.cell
def _(np, torch):
    IMG_SIZE = 96
    OBSTACLE_RADIUS = 0.3
    BLOB_SIGMA = 0.06
    START = np.array([-0.8, 0.0], dtype=np.float32)
    GOALS_Y = (-0.25, 0.0, 0.25)
    CHUNK_SIZE = 20  # 이 실습에서 고정하는 행동 청크 길이 k

    def render_batch(pos: torch.Tensor, goal: torch.Tensor, size: int = IMG_SIZE) -> torch.Tensor:
        """(B, 2) 로봇 위치와 (B, 2) 목표 위치로 (B, 3, H, W) 이미지를 그린다. 값 범위는 [0, 1]."""
        pos = pos.float()
        goal = goal.to(pos)
        coords = (torch.arange(size, device=pos.device, dtype=torch.float32) + 0.5) / size * 2 - 1
        gy, gx = torch.meshgrid(-coords, coords, indexing="ij")  # 0번째 행이 y=+1 (위쪽)

        def blob(p):
            d2 = (gx - p[:, 0, None, None]) ** 2 + (gy - p[:, 1, None, None]) ** 2
            return torch.exp(-d2 / (2 * BLOB_SIGMA**2))

        obstacle = ((gx**2 + gy**2) < OBSTACLE_RADIUS**2).float().expand(pos.shape[0], size, size)
        return torch.stack([obstacle, blob(goal), blob(pos)], dim=1)

    def make_demo(rng, mode=None, goal_y=None, n_steps=None):
        """시연 1개: 2차 베지어 곡선을 따라가는 경로. mode=+1 위로, -1 아래로 돈다."""
        mode = mode if mode is not None else (1 if rng.random() < 0.5 else -1)
        goal_y = goal_y if goal_y is not None else GOALS_Y[rng.integers(len(GOALS_Y))]
        n_steps = n_steps if n_steps is not None else int(rng.integers(30, 71))
        goal = np.array([0.8, goal_y], dtype=np.float32)
        ctrl = np.array([0.0, 1.0 * mode], dtype=np.float32)
        s = np.linspace(0, 1, n_steps + 1, dtype=np.float32)[:, None]
        s = 0.5 - 0.5 * np.cos(np.pi * s)  # 천천히 출발해 천천히 멈춘다
        path = (1 - s) ** 2 * START + 2 * (1 - s) * s * ctrl + s**2 * goal
        return {"path": path.astype(np.float32), "goal": goal, "mode": mode, "n_steps": n_steps}

    return (
        CHUNK_SIZE,
        GOALS_Y,
        IMG_SIZE,
        OBSTACLE_RADIUS,
        START,
        make_demo,
        render_batch,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 1-1. 행동 청크와 `action_is_pad`

    LeRobot은 `ACTConfig.action_delta_indices = list(range(chunk_size))`를 보고, 프레임 $t$를 꺼낼 때
    `action[t], action[t+1], ..., action[t+k-1]`을 함께 가져옵니다. 에피소드 끝을 넘어가는 인덱스는
    **마지막 행동을 반복해서 채우고** `action_is_pad=True`로 표시합니다. 손실은 패딩이 아닌 위치에서만 계산합니다.

    아래 `build_dataset`이 이 과정을 그대로 재현합니다. 이미지는 메모리를 아끼려고 저장하지 않고,
    학습할 때 위치·목표로부터 그때그때 그립니다.
    """)
    return


@app.cell
def _(make_demo, np, torch):
    def chunk_indices(t: int, episode_len: int, chunk_size: int):
        """프레임 t에서 가져올 행동 인덱스와 패딩 여부 (LeRobot 데이터셋의 delta_indices 처리와 같은 규칙)."""
        idx = np.arange(t, t + chunk_size)
        is_pad = idx >= episode_len
        return np.minimum(idx, episode_len - 1), is_pad

    def build_dataset(n_episodes: int, chunk_size: int, seed: int = 0):
        rng = np.random.default_rng(seed)
        cols = {k: [] for k in ["state", "goal", "action", "action_is_pad", "episode_index", "frame_index", "mode", "n_steps"]}
        episodes = []
        for ep in range(n_episodes):
            demo = make_demo(rng)
            episodes.append(demo)
            path = demo["path"]
            actions = path[1:]  # action[t] = 다음 위치 = path[t+1]
            ep_len = len(actions)
            for t in range(ep_len):
                idx, is_pad = chunk_indices(t, ep_len, chunk_size)
                cols["state"].append(path[t])
                cols["goal"].append(demo["goal"])
                cols["action"].append(actions[idx])
                cols["action_is_pad"].append(is_pad)
                cols["episode_index"].append(ep)
                cols["frame_index"].append(t)
                cols["mode"].append(demo["mode"])
                cols["n_steps"].append(demo["n_steps"])
        data = {k: torch.from_numpy(np.stack(v)) for k, v in cols.items()}
        return data, episodes

    return build_dataset, chunk_indices


@app.cell
def _(CHUNK_SIZE, build_dataset):
    N_EPISODES = 100
    dataset, episodes = build_dataset(N_EPISODES, CHUNK_SIZE, seed=0)
    {k: tuple(v.shape) for k, v in dataset.items()}
    return dataset, episodes


@app.cell(hide_code=True)
def _(GOALS_Y, OBSTACLE_RADIUS, episodes, plt):
    def draw_world(ax, goal_y=None):
        ax.add_patch(plt.Circle((0, 0), OBSTACLE_RADIUS, color="tab:red", alpha=0.3))
        for gy in GOALS_Y:
            ax.plot(0.8, gy, "o", color="tab:green", alpha=1.0 if goal_y in (None, gy) else 0.2, ms=8)
        ax.plot(-0.8, 0, "o", color="tab:blue", ms=8)
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_aspect("equal")

    _fig, _axes = plt.subplots(1, 2, figsize=(10, 4.5))
    draw_world(_axes[0])
    _cmap = plt.get_cmap("viridis")
    for _ep in episodes[:40]:
        _axes[0].plot(*_ep["path"].T, color=_cmap((_ep["n_steps"] - 30) / 40), lw=1, alpha=0.8)
    _axes[0].set_title("40 demos (color = episode length: dark=fast, yellow=slow)")
    _axes[1].hist([e["n_steps"] for e in episodes], bins=15, color="tab:gray")
    _axes[1].set_title(
        f"episode length  (up: {sum(e['mode'] == 1 for e in episodes)}, down: {sum(e['mode'] == -1 for e in episodes)})"
    )
    _axes[1].set_xlabel("steps")
    _fig.tight_layout()
    _fig
    return (draw_world,)


@app.cell(hide_code=True)
def _(dataset, mo):
    frame_slider = mo.ui.slider(0, len(dataset["state"]) - 1, value=45, label="데이터셋 프레임 번호", full_width=True)
    frame_slider
    return (frame_slider,)


@app.cell(hide_code=True)
def _(dataset, draw_world, frame_slider, mo, plt, render_batch):
    _i = frame_slider.value
    _img = render_batch(dataset["state"][_i : _i + 1], dataset["goal"][_i : _i + 1])[0]
    _chunk = dataset["action"][_i]
    _pad = dataset["action_is_pad"][_i]

    _fig, _axes = plt.subplots(1, 2, figsize=(10, 4.5))
    _axes[0].imshow(_img.permute(1, 2, 0).numpy(), extent=(-1, 1, -1, 1))
    _axes[0].set_title("observation.images.top (R=obstacle, G=goal, B=robot)")
    draw_world(_axes[1], goal_y=float(dataset["goal"][_i, 1]))
    _axes[1].plot(*dataset["state"][_i], "s", color="black", ms=8, label="observation.state")
    _axes[1].plot(*_chunk[~_pad].T, ".-", color="tab:orange", label="action chunk")
    if _pad.any():
        _axes[1].plot(*_chunk[_pad].T, "x", color="tab:purple", ms=10, label="padded (repeated last)")
    _axes[1].legend(loc="lower left", fontsize=8)
    _axes[1].set_title(
        f"episode {int(dataset['episode_index'][_i])}, frame {int(dataset['frame_index'][_i])} / {int(dataset['n_steps'][_i])}"
    )
    _fig.tight_layout()
    mo.vstack(
        [
            _fig,
            mo.md(
                f"이 프레임의 `action_is_pad`: 패딩 **{int(_pad.sum())}개** / 청크 {len(_pad)}개. "
                "슬라이더를 에피소드 끝 쪽으로 옮겨 보세요."
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. 정규화: `processor_act.py`

    원본 `make_act_pre_post_processors`는 공통 파이프라인(`make_default_pre_post_processors`)을 그대로 씁니다.

    - **전처리**: 배치 차원 추가, 장치 이동, 데이터셋 통계로 정규화
    - **후처리**: 모델 출력 행동을 역정규화하고 CPU로 이동

    `ACTConfig.normalization_mapping`의 기본값은 `VISUAL`, `STATE`, `ACTION` 모두 `MEAN_STD`, 즉
    $\hat{x} = (x - \mu)/(\sigma + \epsilon)$입니다. 아래 `ToyProcessor`가 같은 일을 합니다.
    정책(`ACTPolicy`)은 **정규화된 값만** 보고, 실제 좌표로 되돌리는 것은 후처리의 몫이라는 점을 기억하세요.
    """)
    return


@app.cell
def _(ACTION, DEVICE, IMG_KEY, OBS_STATE, dataset, render_batch, torch):
    class ToyProcessor:
        """MEAN_STD 정규화만 하는 최소 전/후처리기 (lerobot 의 Normalizer/Unnormalizer 단계에 해당)."""

        eps = 1e-8

        def __init__(self, stats: dict[str, dict[str, torch.Tensor]], device):
            self.device = device
            self.stats = {k: {s: v.to(device) for s, v in d.items()} for k, d in stats.items()}

        def preprocess(self, batch: dict) -> dict:
            out = dict(batch)
            for key, st in self.stats.items():
                if key in out:
                    out[key] = (out[key].to(self.device) - st["mean"]) / (st["std"] + self.eps)
            for key in ("action_is_pad",):
                if key in out:
                    out[key] = out[key].to(self.device)
            return out

        def postprocess(self, action: torch.Tensor) -> torch.Tensor:
            st = self.stats[ACTION]
            return (action * (st["std"] + self.eps) + st["mean"]).cpu()

    def compute_stats(data, n_image_samples: int = 2048):
        g = torch.Generator().manual_seed(0)
        idx = torch.randint(len(data["state"]), (n_image_samples,), generator=g)
        imgs = render_batch(data["state"][idx], data["goal"][idx])
        actions = data["action"][~data["action_is_pad"]]  # 패딩이 아닌 실제 행동만
        return {
            OBS_STATE: {"mean": data["state"].mean(0), "std": data["state"].std(0)},
            IMG_KEY: {"mean": imgs.mean((0, 2, 3))[:, None, None], "std": imgs.std((0, 2, 3))[:, None, None]},
            ACTION: {"mean": actions.mean(0), "std": actions.std(0)},
        }

    dataset_stats = compute_stats(dataset)
    processor = ToyProcessor(dataset_stats, DEVICE)
    {k: {s: v.flatten().numpy().round(3).tolist() for s, v in d.items()} for k, d in dataset_stats.items()}
    return (processor,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. `ACTConfig`: 설정과 입력 검증

    LeRobot의 모든 정책 설정은 `PreTrainedConfig`를 상속하고 `input_features` / `output_features`로
    입출력을 선언합니다. 각 항목은 `PolicyFeature(type, shape)`이고, 설정 객체는 이 정보로부터
    `robot_state_feature`, `env_state_feature`, `image_features`, `action_feature`를 꺼내 줍니다.
    아래에서는 이 부분만 작게 다시 만들고, ACT 고유의 필드와 `__post_init__` 검증은 원본을 그대로 옮겼습니다.

    원본 기본값은 양팔 ALOHA 과제(14차원 행동, 카메라 여러 대, `chunk_size=100`) 기준입니다.
    장난감 과제에서는 모델을 줄이고 학습률을 키워 1분 안에 학습되도록 바꿉니다.
    """)
    return


@app.cell
def _(ACTION, Enum, OBS_ENV_STATE, OBS_STATE, dataclass, field):
    class FeatureType(str, Enum):
        STATE = "STATE"
        VISUAL = "VISUAL"
        ENV = "ENV"
        ACTION = "ACTION"

    @dataclass
    class PolicyFeature:
        type: FeatureType
        shape: tuple

    @dataclass
    class ACTConfig:
        # 입출력 선언 (lerobot.configs.PreTrainedConfig 의 일부를 단순화)
        input_features: dict = field(default_factory=dict)
        output_features: dict = field(default_factory=dict)

        # Input / output structure.
        n_obs_steps: int = 1
        chunk_size: int = 100
        n_action_steps: int = 100

        # Architecture. Vision backbone.
        vision_backbone: str = "resnet18"
        pretrained_backbone_weights: str | None = "ResNet18_Weights.IMAGENET1K_V1"
        replace_final_stride_with_dilation: int = False
        # Transformer layers.
        pre_norm: bool = False
        dim_model: int = 512
        n_heads: int = 8
        dim_feedforward: int = 3200
        feedforward_activation: str = "relu"
        n_encoder_layers: int = 4
        # 원본 ACT 는 디코더 7층이지만 코드 버그로 첫 층만 쓰였다. 이를 맞추려고 1로 둔다.
        # (https://github.com/tonyzhaozh/act/issues/25)
        n_decoder_layers: int = 1
        # VAE.
        use_vae: bool = True
        latent_dim: int = 32
        n_vae_encoder_layers: int = 4

        # Inference. 원본 ACT 가 시간 앙상블에 쓴 값은 0.01.
        temporal_ensemble_coeff: float | None = None

        # Training and loss computation.
        dropout: float = 0.1
        kl_weight: float = 10.0

        # Training preset (AdamW)
        optimizer_lr: float = 1e-5
        optimizer_weight_decay: float = 1e-4
        optimizer_lr_backbone: float = 1e-5

        def __post_init__(self):
            """Input validation (not exhaustive). — 원본과 같은 검사"""
            if not self.vision_backbone.startswith("resnet"):
                raise ValueError(f"`vision_backbone` must be one of the ResNet variants. Got {self.vision_backbone}.")
            if self.temporal_ensemble_coeff is not None and self.n_action_steps > 1:
                raise NotImplementedError(
                    "`n_action_steps` must be 1 when using temporal ensembling. This is because the policy "
                    "needs to be queried every step to compute the ensembled action."
                )
            if self.n_action_steps > self.chunk_size:
                raise ValueError(
                    f"The chunk size is the upper bound for the number of action steps per model invocation. Got "
                    f"{self.n_action_steps} for `n_action_steps` and {self.chunk_size} for `chunk_size`."
                )
            if self.n_obs_steps != 1:
                raise ValueError(f"Multiple observation steps not handled yet. Got `nobs_steps={self.n_obs_steps}`")

        def validate_features(self) -> None:
            if not self.image_features and not self.env_state_feature:
                raise ValueError("You must provide at least one image or the environment state among the inputs.")

        # --- PreTrainedConfig 가 제공하는 편의 속성 ---
        @property
        def robot_state_feature(self):
            ft = self.input_features.get(OBS_STATE)
            return ft if ft is not None and ft.type == FeatureType.STATE else None

        @property
        def env_state_feature(self):
            ft = self.input_features.get(OBS_ENV_STATE)
            return ft if ft is not None and ft.type == FeatureType.ENV else None

        @property
        def image_features(self) -> dict:
            return {k: ft for k, ft in self.input_features.items() if ft.type == FeatureType.VISUAL}

        @property
        def action_feature(self):
            return self.output_features.get(ACTION)

        # 데이터셋이 프레임마다 함께 가져올 미래 행동의 상대 인덱스
        @property
        def action_delta_indices(self) -> list:
            return list(range(self.chunk_size))

    return ACTConfig, FeatureType, PolicyFeature


@app.cell
def _(
    ACTConfig,
    ACTION,
    CHUNK_SIZE,
    FeatureType,
    IMG_KEY,
    IMG_SIZE,
    OBS_STATE,
    PolicyFeature,
):
    TOY_OVERRIDES = dict(
        chunk_size=CHUNK_SIZE,
        n_action_steps=CHUNK_SIZE,
        pretrained_backbone_weights=None,  # 실습에서는 ImageNet 가중치를 내려받지 않는다
        dim_model=128,
        n_heads=4,
        dim_feedforward=512,
        n_encoder_layers=2,
        n_vae_encoder_layers=2,
        latent_dim=16,
        optimizer_lr=1e-4,
        optimizer_lr_backbone=1e-4,
    )

    def make_toy_config(**overrides) -> ACTConfig:
        kwargs = {**TOY_OVERRIDES, **overrides}
        return ACTConfig(
            input_features={
                OBS_STATE: PolicyFeature(FeatureType.STATE, (2,)),
                IMG_KEY: PolicyFeature(FeatureType.VISUAL, (3, IMG_SIZE, IMG_SIZE)),
            },
            output_features={ACTION: PolicyFeature(FeatureType.ACTION, (2,))},
            **kwargs,
        )

    return TOY_OVERRIDES, make_toy_config


@app.cell(hide_code=True)
def _(ACTConfig, TOY_OVERRIDES, mo):
    _default = ACTConfig()
    _rows = "\n".join(
        f"| `{k}` | `{getattr(_default, k)}` | `{v}` |" for k, v in TOY_OVERRIDES.items()
    )
    mo.md(
        "| 필드 | 원본 기본값 (ALOHA) | 이 실습 |\n|---|---|---|\n"
        + _rows
        + "\n\n나머지 필드(`use_vae=True`, `kl_weight=10.0`, `n_decoder_layers=1`, `dropout=0.1`, `pre_norm=False` …)는 원본 기본값 그대로입니다."
    )
    return


@app.cell(hide_code=True)
def _(make_toy_config, mo):
    # 검증 규칙이 실제로 어떻게 동작하는지 확인
    _cases = {
        "시간 앙상블 + n_action_steps=10": dict(temporal_ensemble_coeff=0.01, n_action_steps=10),
        "n_action_steps(30) > chunk_size(20)": dict(n_action_steps=30),
        "vision_backbone='vit_b_16'": dict(vision_backbone="vit_b_16"),
        "시간 앙상블 + n_action_steps=1 (허용)": dict(temporal_ensemble_coeff=0.01, n_action_steps=1),
    }
    _lines = []
    for _name, _kw in _cases.items():
        try:
            make_toy_config(**_kw)
            _lines.append(f"- **{_name}** → 통과")
        except (ValueError, NotImplementedError) as _e:
            _lines.append(f"- **{_name}** → `{type(_e).__name__}`: {_e}")
    mo.md("### 설정 검증 실험\n\n" + "\n".join(_lines))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. 부품별로 뜯어보기

    ### 4-1. 위치 임베딩 두 가지

    ACT에는 위치 정보를 넣는 방법이 세 가지 섞여 있습니다.

    | 위치 | 방식 | 원본 코드 |
    |---|---|---|
    | VAE 인코더 입력 `[CLS, 상태, 행동×k]` | **고정** 1D 사인파 | `create_sinusoidal_pos_embedding` |
    | 인코더의 1D 토큰 `[z, 상태, (환경 상태)]` | **학습** 임베딩 | `encoder_1d_feature_pos_embed` |
    | 인코더의 이미지 특징 토큰 | **고정** 2D 사인파 (DETR 방식) | `ACTSinusoidalPositionEmbedding2d` |
    | 디코더 쿼리 $k$개 | **학습** 임베딩 (DETR의 object query) | `decoder_pos_embed` |

    두 사인파 함수는 원본을 그대로 옮겼습니다.
    """)
    return


@app.cell
def _(Tensor, math, nn, np, torch):
    def create_sinusoidal_pos_embedding(num_positions: int, dimension: int) -> Tensor:
        """1D sinusoidal positional embeddings as in Attention is All You Need. → (num_positions, dimension)"""

        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / dimension) for hid_j in range(dimension)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(num_positions)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
        return torch.from_numpy(sinusoid_table).float()

    class ACTSinusoidalPositionEmbedding2d(nn.Module):
        """2D sinusoidal positional embeddings. 위치 인덱스를 [0, 2π] 로 정규화한다는 점이 다르다."""

        def __init__(self, dimension: int):
            super().__init__()
            self.dimension = dimension
            self._two_pi = 2 * math.pi
            self._eps = 1e-6
            # Inverse "common ratio" for the geometric progression in sinusoid frequencies.
            self._temperature = 10000

        def forward(self, x: Tensor) -> Tensor:
            """x: (B, C, H, W) 특징 맵 → (1, 2*dimension, H, W) 위치 임베딩 (배치와 무관)"""
            not_mask = torch.ones_like(x[0, :1])  # (1, H, W)
            # range(1, H+1), range(1, W+1) 에 해당 (원본 코드와 맞추려고 0 이 아닌 1 부터 시작)
            y_range = not_mask.cumsum(1, dtype=torch.float32)
            x_range = not_mask.cumsum(2, dtype=torch.float32)

            # "Normalize" the position index such that it ranges in [0, 2π].
            y_range = y_range / (y_range[:, -1:, :] + self._eps) * self._two_pi
            x_range = x_range / (x_range[:, :, -1:] + self._eps) * self._two_pi

            inverse_frequency = self._temperature ** (
                2 * (torch.arange(self.dimension, dtype=torch.float32, device=x.device) // 2) / self.dimension
            )

            x_range = x_range.unsqueeze(-1) / inverse_frequency  # (1, H, W, 1)
            y_range = y_range.unsqueeze(-1) / inverse_frequency  # (1, H, W, 1)

            # sin 과 cos 이 번갈아 놓이도록 stack 후 flatten. pos_embed_x/y: (1, H, W, C // 2)
            pos_embed_x = torch.stack((x_range[..., 0::2].sin(), x_range[..., 1::2].cos()), dim=-1).flatten(3)
            pos_embed_y = torch.stack((y_range[..., 0::2].sin(), y_range[..., 1::2].cos()), dim=-1).flatten(3)
            pos_embed = torch.cat((pos_embed_y, pos_embed_x), dim=3).permute(0, 3, 1, 2)  # (1, C, H, W)

            return pos_embed

    return ACTSinusoidalPositionEmbedding2d, create_sinusoidal_pos_embedding


@app.cell(hide_code=True)
def _(ACTSinusoidalPositionEmbedding2d, create_sinusoidal_pos_embedding, plt, torch):
    _pe1d = create_sinusoidal_pos_embedding(22, 128)  # 실습 VAE 인코더: 1 + 1 + 20 토큰
    _pe2d = ACTSinusoidalPositionEmbedding2d(64)(torch.zeros(1, 512, 3, 3))  # 실습 이미지 특징: 3x3

    _fig, _axes = plt.subplots(1, 3, figsize=(13, 3.6), gridspec_kw={"width_ratios": [2.2, 1, 1]})
    _axes[0].imshow(_pe1d.numpy(), aspect="auto", cmap="RdBu")
    _axes[0].set(title="1D sinusoid (VAE encoder): 22 tokens x 128 dims", xlabel="dim", ylabel="token")
    _sim = torch.nn.functional.normalize(_pe2d[0].flatten(1).T, dim=-1)
    _axes[1].imshow((_sim @ _sim.T).numpy(), cmap="viridis")
    _axes[1].set(title="2D pos-emb cosine sim\n(9 image tokens)", xlabel="token", ylabel="token")
    _axes[2].imshow(_pe2d[0, 0].numpy(), cmap="RdBu")
    _axes[2].set(title="2D pos-emb, channel 0 (y-part)")
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 4-2. 비전 백본: ResNet + `FrozenBatchNorm2d` + `IntermediateLayerGetter`

    원본은 torchvision ResNet을 만들고 마지막 단계 `layer4`의 특징 맵만 꺼냅니다.

    - `norm_layer=FrozenBatchNorm2d`: 배치 통계를 **업데이트하지 않는** BatchNorm. 작은 배치에서도 안정적입니다(DETR 관례).
    - `IntermediateLayerGetter(..., return_layers={"layer4": "feature_map"})`: 분류용 `avgpool`/`fc` 없이 `layer4` 출력만 돌려줍니다.
    - 출력 해상도는 입력의 약 1/32. 각 픽셀 하나가 Transformer 토큰 하나가 됩니다.
      `encoder_img_feat_input_proj`(1×1 Conv)가 채널 512를 `dim_model`로 바꿉니다.

    `replace_final_stride_with_dilation`을 켜면 마지막 stride를 dilation으로 바꿔 해상도를 1/16로 유지합니다.
    단, torchvision의 `BasicBlock`(ResNet18/34)은 dilation을 지원하지 않아 **resnet50 이상에서만** 동작합니다. 아래에서 확인해 보세요.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    img_hw_slider = mo.ui.slider(32, 640, step=32, value=96, label="입력 이미지 한 변 크기 (px)", show_value=True)
    img_hw_slider
    return (img_hw_slider,)


@app.cell(hide_code=True)
def _(FrozenBatchNorm2d, IntermediateLayerGetter, img_hw_slider, mo, torch, torchvision):
    _bb = torchvision.models.resnet18(weights=None, norm_layer=FrozenBatchNorm2d)
    _getter = IntermediateLayerGetter(_bb, return_layers={"layer4": "feature_map"})
    _hw = img_hw_slider.value
    with torch.no_grad():
        _fm = _getter(torch.zeros(1, 3, _hw, _hw))["feature_map"]

    try:
        torchvision.models.resnet18(replace_stride_with_dilation=[False, False, True], weights=None)
        _dil = "resnet18 + dilation → 생성됨"
    except NotImplementedError as _e:
        _dil = f"resnet18 + dilation → `NotImplementedError: {_e}`"

    mo.md(
        f"""
    - 입력 `(1, 3, {_hw}, {_hw})` → `feature_map` **{tuple(_fm.shape)}** → 카메라 1대당 이미지 토큰 **{_fm.shape[2] * _fm.shape[3]}개**
    - 백본 파라미터 수: {sum(p.numel() for p in _getter.parameters()) / 1e6:.2f}M (`fc` 제외), `fc.in_features = {_bb.fc.in_features}`
    - `FrozenBatchNorm2d`의 학습 파라미터 수: {sum(p.numel() for m in _getter.modules() if isinstance(m, FrozenBatchNorm2d) for p in m.parameters())} (모두 buffer)
    - {_dil}
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 4-3. Transformer 인코더·디코더 층

    원본은 `nn.MultiheadAttention`을 **시퀀스 우선** `(S, B, D)` 모양으로 씁니다(`batch_first=False`).
    DETR과 같이 두 가지 특징이 있습니다.

    1. 위치 임베딩을 **query와 key에만** 더하고 value에는 더하지 않습니다: `q = k = x + pos`, `v = x`.
       매 층마다 다시 더하므로 깊은 층까지 위치 정보가 유지됩니다.
    2. 기본값 `pre_norm=False`, 즉 **post-norm**(잔차를 더한 뒤 LayerNorm)입니다.

    디코더 층은 self-attention → cross-attention(인코더 출력) → FFN 순서입니다.
    디코더 입력은 **0 텐서**이고, 쿼리를 구분하는 정보는 전부 `decoder_pos_embed`에서 옵니다.
    """)
    return


@app.cell
def _(Callable, F, Tensor, nn):
    def get_activation_fn(activation: str) -> Callable:
        """Return an activation function given a string."""
        if activation == "relu":
            return F.relu
        if activation == "gelu":
            return F.gelu
        if activation == "glu":
            return F.glu
        raise RuntimeError(f"activation should be relu/gelu/glu, not {activation}.")

    class ACTEncoderLayer(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.self_attn = nn.MultiheadAttention(config.dim_model, config.n_heads, dropout=config.dropout)

            # Feed forward layers.
            self.linear1 = nn.Linear(config.dim_model, config.dim_feedforward)
            self.dropout = nn.Dropout(config.dropout)
            self.linear2 = nn.Linear(config.dim_feedforward, config.dim_model)

            self.norm1 = nn.LayerNorm(config.dim_model)
            self.norm2 = nn.LayerNorm(config.dim_model)
            self.dropout1 = nn.Dropout(config.dropout)
            self.dropout2 = nn.Dropout(config.dropout)

            self.activation = get_activation_fn(config.feedforward_activation)
            self.pre_norm = config.pre_norm

        def forward(self, x, pos_embed: Tensor | None = None, key_padding_mask: Tensor | None = None) -> Tensor:
            skip = x
            if self.pre_norm:
                x = self.norm1(x)
            q = k = x if pos_embed is None else x + pos_embed  # 위치 정보는 q, k 에만
            x = self.self_attn(q, k, value=x, key_padding_mask=key_padding_mask)
            x = x[0]  # note: [0] to select just the output, not the attention weights
            x = skip + self.dropout1(x)
            if self.pre_norm:
                skip = x
                x = self.norm2(x)
            else:
                x = self.norm1(x)
                skip = x
            x = self.linear2(self.dropout(self.activation(self.linear1(x))))
            x = skip + self.dropout2(x)
            if not self.pre_norm:
                x = self.norm2(x)
            return x

    class ACTEncoder(nn.Module):
        """Convenience module for running multiple encoder layers, maybe followed by normalization."""

        def __init__(self, config, is_vae_encoder: bool = False):
            super().__init__()
            self.is_vae_encoder = is_vae_encoder
            num_layers = config.n_vae_encoder_layers if self.is_vae_encoder else config.n_encoder_layers
            self.layers = nn.ModuleList([ACTEncoderLayer(config) for _ in range(num_layers)])
            self.norm = nn.LayerNorm(config.dim_model) if config.pre_norm else nn.Identity()

        def forward(self, x: Tensor, pos_embed: Tensor | None = None, key_padding_mask: Tensor | None = None) -> Tensor:
            for layer in self.layers:
                x = layer(x, pos_embed=pos_embed, key_padding_mask=key_padding_mask)
            x = self.norm(x)
            return x

    class ACTDecoderLayer(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.self_attn = nn.MultiheadAttention(config.dim_model, config.n_heads, dropout=config.dropout)
            self.multihead_attn = nn.MultiheadAttention(config.dim_model, config.n_heads, dropout=config.dropout)

            # Feed forward layers.
            self.linear1 = nn.Linear(config.dim_model, config.dim_feedforward)
            self.dropout = nn.Dropout(config.dropout)
            self.linear2 = nn.Linear(config.dim_feedforward, config.dim_model)

            self.norm1 = nn.LayerNorm(config.dim_model)
            self.norm2 = nn.LayerNorm(config.dim_model)
            self.norm3 = nn.LayerNorm(config.dim_model)
            self.dropout1 = nn.Dropout(config.dropout)
            self.dropout2 = nn.Dropout(config.dropout)
            self.dropout3 = nn.Dropout(config.dropout)

            self.activation = get_activation_fn(config.feedforward_activation)
            self.pre_norm = config.pre_norm

        def maybe_add_pos_embed(self, tensor: Tensor, pos_embed: Tensor | None) -> Tensor:
            return tensor if pos_embed is None else tensor + pos_embed

        def forward(
            self,
            x: Tensor,
            encoder_out: Tensor,
            decoder_pos_embed: Tensor | None = None,
            encoder_pos_embed: Tensor | None = None,
        ) -> Tensor:
            """
            x: (DS, B, C) 디코더 토큰, encoder_out: (ES, B, C),
            encoder_pos_embed: (ES, 1, C) key 용, decoder_pos_embed: (DS, 1, C) query 용 → (DS, B, C)
            """
            skip = x
            if self.pre_norm:
                x = self.norm1(x)
            q = k = self.maybe_add_pos_embed(x, decoder_pos_embed)
            x = self.self_attn(q, k, value=x)[0]  # select just the output, not the attention weights
            x = skip + self.dropout1(x)
            if self.pre_norm:
                skip = x
                x = self.norm2(x)
            else:
                x = self.norm1(x)
                skip = x
            x = self.multihead_attn(
                query=self.maybe_add_pos_embed(x, decoder_pos_embed),
                key=self.maybe_add_pos_embed(encoder_out, encoder_pos_embed),
                value=encoder_out,
            )[0]  # select just the output, not the attention weights
            x = skip + self.dropout2(x)
            if self.pre_norm:
                skip = x
                x = self.norm3(x)
            else:
                x = self.norm2(x)
                skip = x
            x = self.linear2(self.dropout(self.activation(self.linear1(x))))
            x = skip + self.dropout3(x)
            if not self.pre_norm:
                x = self.norm3(x)
            return x

    class ACTDecoder(nn.Module):
        def __init__(self, config):
            """Convenience module for running multiple decoder layers followed by normalization."""
            super().__init__()
            self.layers = nn.ModuleList([ACTDecoderLayer(config) for _ in range(config.n_decoder_layers)])
            self.norm = nn.LayerNorm(config.dim_model)

        def forward(
            self,
            x: Tensor,
            encoder_out: Tensor,
            decoder_pos_embed: Tensor | None = None,
            encoder_pos_embed: Tensor | None = None,
        ) -> Tensor:
            for layer in self.layers:
                x = layer(x, encoder_out, decoder_pos_embed=decoder_pos_embed, encoder_pos_embed=encoder_pos_embed)
            if self.norm is not None:
                x = self.norm(x)
            return x

    return ACTDecoder, ACTEncoder


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. 전체 모델 `ACT`

    `forward`는 크게 세 단계입니다.

    1. **잠재 변수 준비** — 학습 중이고 `use_vae=True`이면 VAE 인코더로 $\mu, \log\sigma^2$를 구하고
       재매개변수화 $z = \mu + \sigma \odot \epsilon$으로 샘플링합니다. 그 외(추론, `use_vae=False`)에는 $z = 0$.
       VAE 인코더의 `key_padding_mask`에 `action_is_pad`가 들어가므로 패딩된 행동은 attention에서 제외됩니다.
    2. **인코더 입력 조립** — `[z, 상태, (환경 상태), 카메라1 픽셀들, 카메라2 픽셀들, …]`을 시퀀스 축으로 쌓습니다.
    3. **디코더** — 0 텐서 $k$개 + 학습 쿼리 임베딩으로 cross-attention 후 `action_head`(Linear)로 행동 차원에 투영합니다.

    `log_sigma_x2`라는 이름은 $2\log\sigma = \log\sigma^2$를 뜻합니다(원본 구현과 맞춘 표현).
    """)
    return


@app.cell
def _(
    ACTDecoder,
    ACTEncoder,
    ACTION,
    ACTSinusoidalPositionEmbedding2d,
    FrozenBatchNorm2d,
    IntermediateLayerGetter,
    OBS_ENV_STATE,
    OBS_IMAGES,
    OBS_STATE,
    Tensor,
    chain,
    create_sinusoidal_pos_embedding,
    einops,
    nn,
    torch,
    torchvision,
):
    class ACT(nn.Module):
        """Action Chunking Transformer: The underlying neural network for ACTPolicy."""

        def __init__(self, config) -> None:
            # BERT style VAE encoder with input tokens [cls, robot_state, *action_sequence].
            # The cls token forms parameters of the latent's distribution (like this [*means, *log_variances]).
            super().__init__()
            self.config = config

            action_feature = config.action_feature
            if action_feature is None:
                raise ValueError("ACT requires an action output feature (FeatureType.ACTION) in `output_features`.")

            if self.config.use_vae:
                self.vae_encoder = ACTEncoder(config, is_vae_encoder=True)
                self.vae_encoder_cls_embed = nn.Embedding(1, config.dim_model)
                # Projection layer for joint-space configuration to hidden dimension.
                if self.config.robot_state_feature:
                    self.vae_encoder_robot_state_input_proj = nn.Linear(
                        self.config.robot_state_feature.shape[0], config.dim_model
                    )
                # Projection layer for action (joint-space target) to hidden dimension.
                self.vae_encoder_action_input_proj = nn.Linear(action_feature.shape[0], config.dim_model)
                # Projection layer from the VAE encoder's output to the latent distribution's parameter space.
                self.vae_encoder_latent_output_proj = nn.Linear(config.dim_model, config.latent_dim * 2)
                # Fixed sinusoidal positional embedding for the input to the VAE encoder.
                num_input_token_encoder = 1 + config.chunk_size
                if self.config.robot_state_feature:
                    num_input_token_encoder += 1
                self.register_buffer(
                    "vae_encoder_pos_enc",
                    create_sinusoidal_pos_embedding(num_input_token_encoder, config.dim_model).unsqueeze(0),
                )

            # Backbone for image feature extraction.
            if self.config.image_features:
                backbone_model = getattr(torchvision.models, config.vision_backbone)(
                    replace_stride_with_dilation=[False, False, config.replace_final_stride_with_dilation],
                    weights=config.pretrained_backbone_weights,
                    norm_layer=FrozenBatchNorm2d,
                )
                # The forward method of this returns a dict: {"feature_map": output}.
                self.backbone = IntermediateLayerGetter(backbone_model, return_layers={"layer4": "feature_map"})

            # Transformer (acts as VAE decoder when training with the variational objective).
            self.encoder = ACTEncoder(config)
            self.decoder = ACTDecoder(config)

            # Transformer encoder input projections. Tokens: [latent, (robot_state), (env_state), (image pixels)].
            if self.config.robot_state_feature:
                self.encoder_robot_state_input_proj = nn.Linear(self.config.robot_state_feature.shape[0], config.dim_model)
            if self.config.env_state_feature:
                self.encoder_env_state_input_proj = nn.Linear(self.config.env_state_feature.shape[0], config.dim_model)
            self.encoder_latent_input_proj = nn.Linear(config.latent_dim, config.dim_model)
            if self.config.image_features:
                self.encoder_img_feat_input_proj = nn.Conv2d(backbone_model.fc.in_features, config.dim_model, kernel_size=1)
            # Transformer encoder positional embeddings.
            n_1d_tokens = 1  # for the latent
            if self.config.robot_state_feature:
                n_1d_tokens += 1
            if self.config.env_state_feature:
                n_1d_tokens += 1
            self.encoder_1d_feature_pos_embed = nn.Embedding(n_1d_tokens, config.dim_model)
            if self.config.image_features:
                self.encoder_cam_feat_pos_embed = ACTSinusoidalPositionEmbedding2d(config.dim_model // 2)

            # Transformer decoder. Learnable positional embedding (in the style of DETR object queries).
            self.decoder_pos_embed = nn.Embedding(config.chunk_size, config.dim_model)

            # Final action regression head on the output of the transformer's decoder.
            self.action_head = nn.Linear(config.dim_model, action_feature.shape[0])

            self._reset_parameters()

        def _reset_parameters(self):
            """Xavier-uniform initialization of the transformer parameters as in the original code."""
            for p in chain(self.encoder.parameters(), self.decoder.parameters()):
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)

        def forward(self, batch: dict[str, Tensor]):
            """
            batch:
                [OBS_STATE] (optional): (B, state_dim)
                [OBS_IMAGES]: list of (B, C, H, W)  AND/OR  [OBS_ENV_STATE]: (B, env_dim)
                [ACTION] (학습 + VAE 일 때만): (B, chunk_size, action_dim)
            반환: (B, chunk_size, action_dim) 행동 청크, (mu, log_sigma_x2) 각각 (B, latent_dim) 또는 (None, None)
            """
            if self.config.use_vae and self.training:
                assert ACTION in batch, "actions must be provided when using the variational objective in training mode."

            batch_size = batch[OBS_IMAGES][0].shape[0] if OBS_IMAGES in batch else batch[OBS_ENV_STATE].shape[0]

            # ---- 1) 잠재 변수 준비 ----
            if self.config.use_vae and ACTION in batch and self.training:
                # Prepare the input to the VAE encoder: [cls, *joint_space_configuration, *action_sequence].
                cls_embed = einops.repeat(self.vae_encoder_cls_embed.weight, "1 d -> b 1 d", b=batch_size)  # (B, 1, D)
                if self.config.robot_state_feature:
                    robot_state_embed = self.vae_encoder_robot_state_input_proj(batch[OBS_STATE])
                    robot_state_embed = robot_state_embed.unsqueeze(1)  # (B, 1, D)
                action_embed = self.vae_encoder_action_input_proj(batch[ACTION])  # (B, S, D)

                if self.config.robot_state_feature:
                    vae_encoder_input = [cls_embed, robot_state_embed, action_embed]  # (B, S+2, D)
                else:
                    vae_encoder_input = [cls_embed, action_embed]
                vae_encoder_input = torch.cat(vae_encoder_input, axis=1)

                # Prepare fixed positional embedding.
                pos_embed = self.vae_encoder_pos_enc.clone().detach()  # (1, S+2, D)

                # Key padding mask: CLS(와 상태) 토큰은 패딩이 아니고, 행동 토큰은 action_is_pad 를 따른다.
                cls_joint_is_pad = torch.full(
                    (batch_size, 2 if self.config.robot_state_feature else 1),
                    False,
                    device=batch[OBS_STATE].device,
                )
                key_padding_mask = torch.cat([cls_joint_is_pad, batch["action_is_pad"]], axis=1)  # (B, S+2)

                # Forward pass through VAE encoder to get the latent PDF parameters.
                cls_token_out = self.vae_encoder(
                    vae_encoder_input.permute(1, 0, 2),
                    pos_embed=pos_embed.permute(1, 0, 2),
                    key_padding_mask=key_padding_mask,
                )[0]  # select the class token, with shape (B, D)
                latent_pdf_params = self.vae_encoder_latent_output_proj(cls_token_out)
                mu = latent_pdf_params[:, : self.config.latent_dim]
                # This is 2log(sigma). Done this way to match the original implementation.
                log_sigma_x2 = latent_pdf_params[:, self.config.latent_dim :]

                # Sample the latent with the reparameterization trick.
                latent_sample = mu + log_sigma_x2.div(2).exp() * torch.randn_like(mu)
            else:
                # When not using the VAE encoder, we set the latent to be all zeros.
                mu = log_sigma_x2 = None
                latent_sample = torch.zeros([batch_size, self.config.latent_dim], dtype=torch.float32).to(
                    batch[OBS_STATE].device
                )

            # ---- 2) Transformer 인코더 입력 조립 ----
            encoder_in_tokens = [self.encoder_latent_input_proj(latent_sample)]
            encoder_in_pos_embed = list(self.encoder_1d_feature_pos_embed.weight.unsqueeze(1))
            # Robot state token.
            if self.config.robot_state_feature:
                encoder_in_tokens.append(self.encoder_robot_state_input_proj(batch[OBS_STATE]))
            # Environment state token.
            if self.config.env_state_feature:
                encoder_in_tokens.append(self.encoder_env_state_input_proj(batch[OBS_ENV_STATE]))

            if self.config.image_features:
                for img in batch[OBS_IMAGES]:
                    cam_features = self.backbone(img)["feature_map"]
                    cam_pos_embed = self.encoder_cam_feat_pos_embed(cam_features).to(dtype=cam_features.dtype)
                    cam_features = self.encoder_img_feat_input_proj(cam_features)

                    # Rearrange features to (sequence, batch, dim).
                    cam_features = einops.rearrange(cam_features, "b c h w -> (h w) b c")
                    cam_pos_embed = einops.rearrange(cam_pos_embed, "b c h w -> (h w) b c")

                    encoder_in_tokens.extend(list(cam_features))
                    encoder_in_pos_embed.extend(list(cam_pos_embed))

            # Stack all tokens along the sequence dimension.
            encoder_in_tokens = torch.stack(encoder_in_tokens, axis=0)
            encoder_in_pos_embed = torch.stack(encoder_in_pos_embed, axis=0)

            # ---- 3) 인코더 → 디코더 → 행동 헤드 ----
            encoder_out = self.encoder(encoder_in_tokens, pos_embed=encoder_in_pos_embed)
            decoder_in = torch.zeros(
                (self.config.chunk_size, batch_size, self.config.dim_model),
                dtype=encoder_in_pos_embed.dtype,
                device=encoder_in_pos_embed.device,
            )
            decoder_out = self.decoder(
                decoder_in,
                encoder_out,
                encoder_pos_embed=encoder_in_pos_embed,
                decoder_pos_embed=self.decoder_pos_embed.weight.unsqueeze(1),
            )

            # Move back to (B, S, C).
            decoder_out = decoder_out.transpose(0, 1)

            actions = self.action_head(decoder_out)

            return actions, (mu, log_sigma_x2)

    return (ACT,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. `ACTPolicy`: 손실과 행동 선택

    `ACTPolicy`는 `ACT` 네트워크를 감싸서 LeRobot 학습·평가 루프가 부르는 인터페이스를 제공합니다.

    - `forward(batch)` → `(loss, loss_dict)`: 학습용. 손실은
      $$\mathcal{L} = \underbrace{\frac{1}{|\text{valid}|}\sum_{\text{valid}} |a - \hat a|}_{\text{L1 (패딩 제외)}}
      + \beta \cdot \underbrace{\frac{1}{B}\sum_{b}\sum_{j=1}^{L} -\tfrac12\left(1 + \log\sigma_j^2 - \mu_j^2 - \sigma_j^2\right)}_{D_{KL}(q(z\mid\cdot)\,\|\,\mathcal N(0, I))}$$
      여기서 $\beta$ = `kl_weight`(기본 10).
    - `predict_action_chunk(batch)`: 행동 청크 전체 `(B, k, A)`를 반환.
    - `select_action(batch)`: 환경에 보낼 **행동 1개**를 반환. 큐가 비었을 때만 모델을 부르고,
      청크의 앞 `n_action_steps`개를 큐에 넣어 하나씩 꺼냅니다. 시간 앙상블을 쓰면 매 스텝 모델을 부릅니다.
    - `reset()`: 에피소드가 시작될 때마다 호출해 큐/앙상블 상태를 비웁니다.
    - `get_optim_params()`: 백본과 나머지에 서로 다른 학습률을 줄 수 있도록 파라미터 그룹을 나눕니다.
    """)
    return


@app.cell
def _(Tensor, torch):
    class ACTTemporalEnsembler:
        def __init__(self, temporal_ensemble_coeff: float, chunk_size: int) -> None:
            """Temporal ensembling as described in Algorithm 2 of https://huggingface.co/papers/2304.13705.

            가중치 wᵢ = exp(-temporal_ensemble_coeff * i), w₀ 이 가장 오래된(먼저 예측된) 행동.
              - 0 이면 모든 예측을 균등하게
              - 양수이면 오래된 예측에 더 큰 가중치 (원본 ACT 기본값 0.01)
              - 음수이면 새 예측에 더 큰 가중치
            과거 예측을 모두 저장하지 않고 온라인(누적) 평균으로 계산한다.
            """
            self.chunk_size = chunk_size
            self.ensemble_weights = torch.exp(-temporal_ensemble_coeff * torch.arange(chunk_size))
            self.ensemble_weights_cumsum = torch.cumsum(self.ensemble_weights, dim=0)
            self.reset()

        def reset(self):
            """Resets the online computation variables."""
            self.ensembled_actions = None
            # (chunk_size,) count of how many actions are in the ensemble for each time step in the sequence.
            self.ensembled_actions_count = None

        def update(self, actions: Tensor) -> Tensor:
            """(batch, chunk_size, action_dim) 새 예측으로 앙상블을 갱신하고, 다음 스텝 행동 1개를 꺼내 반환."""
            self.ensemble_weights = self.ensemble_weights.to(device=actions.device)
            self.ensemble_weights_cumsum = self.ensemble_weights_cumsum.to(device=actions.device)
            if self.ensembled_actions is None:
                # 첫 스텝: 첫 예측 청크로 초기화
                self.ensembled_actions = actions.clone()
                self.ensembled_actions_count = torch.ones(
                    (self.chunk_size, 1), dtype=torch.long, device=self.ensembled_actions.device
                )
            else:
                # self.ensembled_actions: (batch, chunk_size - 1, action_dim). 겹치는 구간의 온라인 평균 갱신.
                self.ensembled_actions *= self.ensemble_weights_cumsum[self.ensembled_actions_count - 1]
                self.ensembled_actions += actions[:, :-1] * self.ensemble_weights[self.ensembled_actions_count]
                self.ensembled_actions /= self.ensemble_weights_cumsum[self.ensembled_actions_count]
                self.ensembled_actions_count = torch.clamp(self.ensembled_actions_count + 1, max=self.chunk_size)
                # 새 청크의 마지막 행동은 이전 예측이 없으므로 그대로 뒤에 붙인다.
                self.ensembled_actions = torch.cat([self.ensembled_actions, actions[:, -1:]], dim=1)
                self.ensembled_actions_count = torch.cat(
                    [self.ensembled_actions_count, torch.ones_like(self.ensembled_actions_count[-1:])]
                )
            # "Consume" the first action.
            action, self.ensembled_actions, self.ensembled_actions_count = (
                self.ensembled_actions[:, 0],
                self.ensembled_actions[:, 1:],
                self.ensembled_actions_count[1:],
            )
            return action

    return (ACTTemporalEnsembler,)


@app.cell
def _(
    ACT,
    ACTION,
    ACTTemporalEnsembler,
    Any,
    F,
    OBS_IMAGES,
    Tensor,
    deque,
    nn,
    torch,
):
    class ACTPolicy(nn.Module):
        """lerobot 의 ACTPolicy 와 같은 인터페이스 (PreTrainedPolicy 대신 nn.Module 을 상속)."""

        name = "act"

        def __init__(self, config):
            super().__init__()
            config.validate_features()
            self.config = config

            self.model = ACT(config)

            if config.temporal_ensemble_coeff is not None:
                self.temporal_ensembler = ACTTemporalEnsembler(config.temporal_ensemble_coeff, config.chunk_size)

            self.reset()

        def get_optim_params(self) -> list[dict[str, Any]]:
            return [
                {"params": [p for n, p in self.named_parameters() if not n.startswith("model.backbone") and p.requires_grad]},
                {
                    "params": [p for n, p in self.named_parameters() if n.startswith("model.backbone") and p.requires_grad],
                    "lr": self.config.optimizer_lr_backbone,
                },
            ]

        def reset(self):
            """This should be called whenever the environment is reset."""
            if self.config.temporal_ensemble_coeff is not None:
                self.temporal_ensembler.reset()
            else:
                self._action_queue = deque([], maxlen=self.config.n_action_steps)

        @torch.no_grad()
        def select_action(self, batch: dict[str, Tensor]) -> Tensor:
            """환경에 보낼 행동 1개. 큐가 비었을 때만 모델을 호출한다."""
            self.eval()  # keeping the policy in eval mode as it could be set to train mode while queue is consumed

            if self.config.temporal_ensemble_coeff is not None:
                actions = self.predict_action_chunk(batch)
                action = self.temporal_ensembler.update(actions)
                return action

            # Action queue logic for n_action_steps > 1.
            if len(self._action_queue) == 0:
                actions = self.predict_action_chunk(batch)[:, : self.config.n_action_steps]
                # (B, n_action_steps, A) 를 (n_action_steps, B, A) 로 바꿔 큐에 넣는다.
                self._action_queue.extend(actions.transpose(0, 1))
            return self._action_queue.popleft()

        @torch.no_grad()
        def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
            """Predict a chunk of actions given environment observations."""
            self.eval()

            if self.config.image_features:
                batch = dict(batch)  # shallow copy so that adding a key doesn't modify the original
                batch[OBS_IMAGES] = [batch[key] for key in self.config.image_features]

            actions = self.model(batch)[0]
            return actions

        def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
            """Run the batch through the model and compute the loss for training or validation."""
            if self.config.image_features:
                batch = dict(batch)
                batch[OBS_IMAGES] = [batch[key] for key in self.config.image_features]

            actions_hat, (mu_hat, log_sigma_x2_hat) = self.model(batch)

            abs_err = F.l1_loss(batch[ACTION], actions_hat, reduction="none")
            valid_mask = ~batch["action_is_pad"].unsqueeze(-1)
            num_valid = valid_mask.sum() * abs_err.shape[-1]
            l1_loss = (abs_err * valid_mask).sum() / num_valid.clamp_min(1)

            loss_dict = {"l1_loss": l1_loss.item()}
            if self.config.use_vae and log_sigma_x2_hat is not None:
                # 각 잠재 차원의 KL 을 합하고(sum(-1)) 배치 평균(mean())
                mean_kld = (-0.5 * (1 + log_sigma_x2_hat - mu_hat.pow(2) - (log_sigma_x2_hat).exp())).sum(-1).mean()
                loss_dict["kld_loss"] = mean_kld.item()
                loss = l1_loss + mean_kld * self.config.kl_weight
            else:
                loss = l1_loss

            return loss, loss_dict

    return (ACTPolicy,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 6-1. 텐서 모양 추적

    학습 모드(`train()`)의 정책에 배치 4개를 넣고, 주요 모듈의 입출력 모양을 forward hook으로 기록합니다.
    **시퀀스 우선 `(S, B, D)`** 와 **배치 우선 `(B, S, D)`** 이 어디서 바뀌는지 눈여겨보세요.
    """)
    return


@app.cell
def _(ACTION, DEVICE, IMG_KEY, OBS_STATE, processor, render_batch, torch):
    def make_batch(data, idx):
        """데이터셋 인덱스로 LeRobot 형식의 (정규화 전) 배치를 만든다. 이미지는 여기서 그린다."""
        idx = idx.cpu()
        state = data["state"][idx].to(DEVICE)
        goal = data["goal"][idx].to(DEVICE)
        return {
            OBS_STATE: state,
            IMG_KEY: render_batch(state, goal),
            ACTION: data["action"][idx].to(DEVICE),
            "action_is_pad": data["action_is_pad"][idx].to(DEVICE),
        }

    def make_obs(pos, goal):
        """환경 관측 하나를 정책 입력(정규화 후, 배치 차원 포함)으로 만든다."""
        state = torch.as_tensor(pos, dtype=torch.float32, device=DEVICE)[None]
        g = torch.as_tensor(goal, dtype=torch.float32, device=DEVICE)[None]
        return processor.preprocess({OBS_STATE: state, IMG_KEY: render_batch(state, g)})

    return make_batch, make_obs


@app.cell
def _(ACTPolicy, DEVICE, dataset, make_batch, make_toy_config, processor, torch):
    def trace_shapes(policy, batch):
        records = []
        m = policy.model
        targets = {
            "model.vae_encoder": m.vae_encoder if hasattr(m, "vae_encoder") else None,
            "model.vae_encoder_latent_output_proj": getattr(m, "vae_encoder_latent_output_proj", None),
            "model.backbone": m.backbone,
            "model.encoder_img_feat_input_proj": m.encoder_img_feat_input_proj,
            "model.encoder": m.encoder,
            "model.decoder": m.decoder,
            "model.action_head": m.action_head,
        }

        def shape_of(x):
            if isinstance(x, torch.Tensor):
                return tuple(x.shape)
            if isinstance(x, dict):
                return {k: shape_of(v) for k, v in x.items()}
            if isinstance(x, (list, tuple)):
                return [shape_of(v) for v in x]
            return type(x).__name__

        hooks = []
        for name, mod in targets.items():
            if mod is None:
                continue
            hooks.append(
                mod.register_forward_hook(
                    lambda mod_, inp, out, name=name: records.append(
                        {"module": name, "input": str(shape_of(inp[0])), "output": str(shape_of(out))}
                    )
                )
            )
        try:
            loss, loss_dict = policy(batch)
        finally:
            for h in hooks:
                h.remove()
        return records, loss_dict

    torch.manual_seed(0)
    demo_policy = ACTPolicy(make_toy_config()).to(DEVICE).train()
    _batch = processor.preprocess(make_batch(dataset, torch.arange(4)))
    shape_records, _loss_dict = trace_shapes(demo_policy, _batch)
    shape_records
    return (demo_policy,)


@app.cell(hide_code=True)
def _(
    ACTConfig,
    ACTION,
    ACTPolicy,
    FeatureType,
    OBS_STATE,
    PolicyFeature,
    demo_policy,
    mo,
):
    _groups = {}
    for _n, _p in demo_policy.named_parameters():
        _key = ".".join(_n.split(".")[:2])
        _groups[_key] = _groups.get(_key, 0) + _p.numel()
    _total = sum(_groups.values())

    # 비교용: 원본 기본 설정(ALOHA: 14차원 상태·행동, 카메라 1대)의 파라미터 수
    _aloha = ACTConfig(
        input_features={
            OBS_STATE: PolicyFeature(FeatureType.STATE, (14,)),
            "observation.images.top": PolicyFeature(FeatureType.VISUAL, (3, 480, 640)),
        },
        output_features={ACTION: PolicyFeature(FeatureType.ACTION, (14,))},
        pretrained_backbone_weights=None,
    )
    _aloha_total = sum(p.numel() for p in ACTPolicy(_aloha).parameters())

    _lines = [
        f"**실습 모델 파라미터 수: {_total / 1e6:.2f}M** (원본 기본 설정은 {_aloha_total / 1e6:.1f}M)",
        "",
        "| 모듈 | 파라미터 | 비율 |",
        "|---|---:|---:|",
        *[f"| `{k}` | {v:,} | {v / _total:.1%} |" for k, v in sorted(_groups.items(), key=lambda kv: -kv[1])],
        "",
        "- 인코더 토큰 = `z` 1 + 상태 1 + 이미지 3×3 = **11개**, VAE 인코더 토큰 = CLS 1 + 상태 1 + 행동 20 = **22개**.",
        "- 실습 모델에서는 파라미터 대부분이 ResNet18 백본입니다. 원본 설정에서는 `dim_feedforward=3200`인 Transformer 쪽 비중이 커집니다.",
    ]
    mo.md("\n".join(_lines))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 실습 1. 손실 함수 직접 구현하기

    아래 두 함수를 채운 뒤 셀을 실행하세요. 다음 셀이 `ACTPolicy.forward`와 같은 값을 내는지 검사합니다.

    1. `masked_l1(target, pred, is_pad)`: `target`, `pred`는 `(B, k, A)`, `is_pad`는 `(B, k)`.
       **패딩이 아닌 원소들의 평균** 절댓값 오차를 반환합니다.
    2. `kl_to_standard_normal(mu, log_sigma_x2)`: $D_{KL}(\mathcal N(\mu, \sigma^2)\,\|\,\mathcal N(0, 1))$을
       잠재 차원에 대해 **합하고** 배치에 대해 **평균**한 값을 반환합니다. `log_sigma_x2`는 $\log\sigma^2$입니다.
    """)
    return


@app.cell
def _(torch):
    def masked_l1(target: torch.Tensor, pred: torch.Tensor, is_pad: torch.Tensor) -> torch.Tensor:
        # TODO: 패딩 위치(is_pad=True)를 제외한 평균 절댓값 오차
        raise NotImplementedError

    def kl_to_standard_normal(mu: torch.Tensor, log_sigma_x2: torch.Tensor) -> torch.Tensor:
        # TODO: 닫힌 형태의 KL 을 잠재 차원으로 합하고 배치로 평균
        raise NotImplementedError

    return kl_to_standard_normal, masked_l1


@app.cell(hide_code=True)
def _(kl_to_standard_normal, masked_l1, mo, torch):
    def _check_loss_exercise():
        g = torch.Generator().manual_seed(1)
        target, pred = torch.randn(3, 5, 2, generator=g), torch.randn(3, 5, 2, generator=g)
        is_pad = torch.zeros(3, 5, dtype=torch.bool)
        is_pad[0, 3:] = True
        is_pad[2, 1:] = True
        valid = ~is_pad.unsqueeze(-1)
        ref_l1 = ((target - pred).abs() * valid).sum() / (valid.sum() * 2)
        mu, lv = torch.randn(4, 6, generator=g), torch.randn(4, 6, generator=g)
        ref_kl = torch.distributions.kl_divergence(
            torch.distributions.Normal(mu, (lv / 2).exp()), torch.distributions.Normal(0.0, 1.0)
        ).sum(-1).mean()
        out = []
        for name, fn, args, ref in [
            ("masked_l1", masked_l1, (target, pred, is_pad), ref_l1),
            ("kl_to_standard_normal", kl_to_standard_normal, (mu, lv), ref_kl),
        ]:
            try:
                val = fn(*args)
                ok = torch.allclose(torch.as_tensor(val, dtype=torch.float32), ref, atol=1e-5)
                out.append(f"- `{name}`: {'✅ 통과' if ok else '❌ 다름'} (내 값 {float(val):.5f}, 기준 {float(ref):.5f})")
            except NotImplementedError:
                out.append(f"- `{name}`: 아직 구현하지 않았습니다.")
        return "\n".join(out)

    mo.vstack(
        [
            mo.md(_check_loss_exercise()),
            mo.accordion(
                {
                    "정답 보기": mo.md(
                        """
    ```python
    def masked_l1(target, pred, is_pad):
        valid = ~is_pad.unsqueeze(-1)                     # (B, k, 1)
        err = (target - pred).abs() * valid
        return err.sum() / (valid.sum() * target.shape[-1]).clamp_min(1)

    def kl_to_standard_normal(mu, log_sigma_x2):
        return (-0.5 * (1 + log_sigma_x2 - mu.pow(2) - log_sigma_x2.exp())).sum(-1).mean()
    ```
    분모에 `valid.sum()`만 쓰면 행동 차원 수(A)만큼 값이 커집니다. 원본도 `* abs_err.shape[-1]`을 곱합니다.
    """
                    )
                }
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 실습 2. 인코더 토큰 수 세기

    `count_encoder_tokens(n_cameras, image_hw, has_state, has_env_state)`가 Transformer 인코더 입력
    토큰 수를 반환하도록 구현하세요. ResNet18의 `layer4` 출력 해상도는 입력의 약 1/32이며, 정확한 크기는
    stride 2 연산 5번(conv1, maxpool, layer2~4)을 거친 결과입니다. 검사 셀이 실제 백본으로 확인합니다.

    예: LeRobot으로 SO-101 팔을 카메라 2대(480×640)로 학습하면 토큰이 몇 개일까요?
    """)
    return


@app.cell
def _():
    def count_encoder_tokens(n_cameras: int, image_hw: tuple[int, int], has_state: bool, has_env_state: bool) -> int:
        # TODO: z 토큰 + (상태) + (환경 상태) + 카메라별 특징 맵 픽셀 수
        raise NotImplementedError

    return (count_encoder_tokens,)


@app.cell(hide_code=True)
def _(FrozenBatchNorm2d, IntermediateLayerGetter, count_encoder_tokens, mo, torch, torchvision):
    def _check_token_exercise():
        getter = IntermediateLayerGetter(
            torchvision.models.resnet18(weights=None, norm_layer=FrozenBatchNorm2d), return_layers={"layer4": "feature_map"}
        ).eval()
        cases = [(1, (96, 96), True, False), (2, (480, 640), True, False), (3, (100, 75), False, True), (1, (224, 224), True, True)]
        lines = []
        for n_cam, hw, st, env in cases:
            with torch.no_grad():
                fm = getter(torch.zeros(1, 3, *hw))["feature_map"]
            ref = 1 + int(st) + int(env) + n_cam * fm.shape[2] * fm.shape[3]
            try:
                got = count_encoder_tokens(n_cam, hw, st, env)
                lines.append(f"- 카메라 {n_cam}대 {hw}, state={st}, env={env}: {'✅' if got == ref else '❌'} 내 값 {got} / 실제 {ref}")
            except NotImplementedError:
                lines.append("- 아직 구현하지 않았습니다.")
                break
        return "\n".join(lines)

    mo.vstack(
        [
            mo.md(_check_token_exercise()),
            mo.accordion(
                {
                    "정답 보기": mo.md(
                        """
    ```python
    import math
    def count_encoder_tokens(n_cameras, image_hw, has_state, has_env_state):
        h, w = image_hw
        for _ in range(5):                 # stride 2 다섯 번 (padding 덕분에 올림)
            h, w = math.ceil(h / 2), math.ceil(w / 2)
        return 1 + int(has_state) + int(has_env_state) + n_cameras * h * w
    ```
    480×640 카메라 2대면 이미지 토큰만 2 × 15 × 20 = 600개입니다. 이미지 토큰이 인코더 계산량의 대부분을 차지합니다.
    """
                    )
                }
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7. 학습

    세 가지 설정을 같은 데이터·같은 단계 수로 학습해 비교합니다.

    | 이름 | 설정 | 확인할 점 |
    |---|---|---|
    | `VAE β=10 (기본값)` | `use_vae=True, kl_weight=10` | LeRobot 기본값 |
    | `VAE β=0.01` | `use_vae=True, kl_weight=0.01` | KL 벌점을 약하게 → $z$가 정보를 담는가? |
    | `VAE 없음` | `use_vae=False` | 순수 L1 회귀. $z$는 항상 0 |

    학습 루프는 LeRobot `lerobot-train`의 핵심만 남겼습니다: AdamW(`get_optim_params`의 파라미터 그룹),
    그래디언트 노름 클리핑 10, 매 스텝 무작위 프레임 배치. 원본 기본 학습률 `1e-5`는
    `chunk_size=100`의 큰 모델과 10만 스텝 학습 기준이라, 여기서는 `1e-4`로 키웠습니다.
    """)
    return


@app.cell
def _():
    VARIANTS = {
        "VAE β=10 (기본값)": dict(use_vae=True, kl_weight=10.0),
        "VAE β=0.01": dict(use_vae=True, kl_weight=0.01),
        "VAE 없음": dict(use_vae=False),
    }
    return (VARIANTS,)


@app.cell
def _(ACTPolicy, DEVICE, make_batch, make_toy_config, processor, time, torch):
    def train_policy(overrides, data, steps, batch_size, seed=0, progress=None, log_every=50):
        torch.manual_seed(seed)
        config = make_toy_config(**overrides)
        policy = ACTPolicy(config).to(DEVICE)
        optimizer = torch.optim.AdamW(
            policy.get_optim_params(), lr=config.optimizer_lr, weight_decay=config.optimizer_weight_decay
        )
        n = len(data["state"])
        history = []
        t0 = time.time()
        for step in range(steps):
            policy.train()
            idx = torch.randint(n, (batch_size,))
            batch = processor.preprocess(make_batch(data, idx))
            loss, loss_dict = policy(batch)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 10.0)
            optimizer.step()
            history.append({"step": step, "loss": loss.item(), **loss_dict})
            if progress is not None and (step + 1) % log_every == 0:
                progress.update(increment=log_every)
        policy.eval()
        return {"policy": policy, "history": history, "seconds": time.time() - t0}

    return (train_policy,)


@app.cell(hide_code=True)
def _(VARIANTS, mo):
    train_form = (
        mo.md(
            """
            **학습 설정**

            {steps}

            {batch_size}

            {variants}
            """
        )
        .batch(
            steps=mo.ui.number(start=100, stop=20000, step=100, value=2000, label="학습 스텝 수"),
            batch_size=mo.ui.number(start=8, stop=256, step=8, value=64, label="배치 크기"),
            variants=mo.ui.multiselect(options=list(VARIANTS), value=list(VARIANTS), label="학습할 설정"),
        )
        .form(submit_button_label="학습 시작")
    )
    train_form
    return (train_form,)


@app.cell
def _(AUTORUN, SMOKE_STEPS, VARIANTS, dataset, mo, train_form, train_policy):
    if train_form.value is None and not AUTORUN:
        mo.stop(True, mo.callout(mo.md("위 폼에서 **학습 시작**을 누르면 이후 셀이 실행됩니다."), kind="info"))

    _opts = train_form.value or {"steps": 2000, "batch_size": 64, "variants": list(VARIANTS)}
    _steps = SMOKE_STEPS or int(_opts["steps"])
    _names = [n for n in VARIANTS if n in _opts["variants"]]
    trained = {}
    with mo.status.progress_bar(total=_steps * len(_names), title="ACT 학습 중", remove_on_exit=True) as _bar:
        for _name in _names:
            _bar.update(increment=0, subtitle=_name)
            trained[_name] = train_policy(VARIANTS[_name], dataset, _steps, int(_opts["batch_size"]), progress=_bar)
    mo.md("\n".join(f"- **{n}**: {r['seconds']:.1f}초, 마지막 L1 {r['history'][-1]['l1_loss']:.4f}" for n, r in trained.items()))
    return (trained,)


@app.cell(hide_code=True)
def _(np, plt, trained):
    def _smooth(x, w=50):
        x = np.asarray(x)
        w = min(w, len(x))
        return np.convolve(x, np.ones(w) / w, mode="valid")

    _fig, _axes = plt.subplots(1, 2, figsize=(11, 3.8))
    _labels = {"VAE β=10 (기본값)": "VAE beta=10 (default)", "VAE β=0.01": "VAE beta=0.01", "VAE 없음": "no VAE"}
    for _name, _res in trained.items():
        _h = _res["history"]
        _axes[0].plot(_smooth([r["l1_loss"] for r in _h]), label=_labels[_name])
        if "kld_loss" in _h[0]:
            _axes[1].plot(_smooth([r["kld_loss"] for r in _h]), label=_labels[_name])
    _axes[0].set(title="L1 loss (moving avg, normalized units)", xlabel="step", yscale="log")
    _axes[1].set(title="KL divergence (nats, sum over latent dims)", xlabel="step", yscale="log")
    for _a in _axes:
        _a.legend()
        _a.grid(alpha=0.3)
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **관찰 포인트**

    - β=10에서는 KL이 0 근처로 떨어지기 쉽습니다. 이것을 **사후 붕괴(posterior collapse)** 라고 합니다.
      $z$에 정보를 실으면 L1이 조금 줄지만, 10배로 키운 KL 벌점이 그보다 크면 인코더는 $\mathcal N(0, I)$를 그대로 출력하는 쪽을 택합니다.
    - β=0.01에서는 KL이 0보다 확실히 크게 유지되고 L1도 더 낮아집니다. $z$가 "이 시연은 위로/빠르게"
      같은, 관측에서 보이지 않는 정보를 전달하기 때문입니다.
    - 그렇다고 β를 낮추는 것이 항상 좋은 것은 아닙니다. 추론 때는 $z=0$만 쓰므로, 학습 때 $z$에
      의존한 만큼 추론 때 차이가 생길 수 있습니다. 다음 장에서 직접 확인합니다.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 8. 잠재 변수 $z$ 들여다보기

    원본 `ACT.forward`는 추론 때 항상 $z=0$을 넣습니다. 여기서는 `encoder_latent_input_proj`에
    forward pre-hook을 걸어 **원하는 $z$를 끼워 넣고**, 같은 관측에서 $z \sim \mathcal N(0, I)$를 여러 번 뽑아 봅니다.
    모델 코드는 건드리지 않습니다.

    - 왼쪽 열: 출발 지점에서 $z$를 32번 샘플링한 행동 청크(회색), $z=0$ 청크(검정).
    - 오른쪽 열: 모든 에피소드 첫 프레임의 사후 평균 $\mu$를 2차원으로 투영(PCA). 색은 위/아래, 크기는 시연 길이.
    """)
    return


@app.cell
def _(OBS_IMAGES, OBS_STATE, contextmanager, torch):
    @contextmanager
    def override_latent(policy, z):
        """policy.model 이 추론할 때 쓰는 z(기본 0)를 주어진 z 로 바꾼다."""

        def hook(module, args):
            return (z.to(device=args[0].device, dtype=args[0].dtype),)

        handle = policy.model.encoder_latent_input_proj.register_forward_pre_hook(hook)
        try:
            yield
        finally:
            handle.remove()

    @torch.no_grad()
    def posterior_mean(policy, batch):
        """ACT.forward 의 VAE 인코더 부분만 실행해 μ, log σ² 를 얻는다 (dropout 없이)."""
        m, c = policy.model, policy.config
        policy.eval()
        if c.image_features:
            batch = dict(batch)
            batch[OBS_IMAGES] = [batch[k] for k in c.image_features]
        b = batch[OBS_STATE].shape[0]
        tokens = [m.vae_encoder_cls_embed.weight.expand(b, 1, -1)]
        tokens.append(m.vae_encoder_robot_state_input_proj(batch[OBS_STATE]).unsqueeze(1))
        tokens.append(m.vae_encoder_action_input_proj(batch["action"]))
        x = torch.cat(tokens, 1).permute(1, 0, 2)
        pad = torch.cat([torch.zeros(b, 2, dtype=torch.bool, device=x.device), batch["action_is_pad"]], 1)
        cls_out = m.vae_encoder(x, pos_embed=m.vae_encoder_pos_enc.permute(1, 0, 2), key_padding_mask=pad)[0]
        params = m.vae_encoder_latent_output_proj(cls_out)
        return params[:, : c.latent_dim], params[:, c.latent_dim :]

    return override_latent, posterior_mean


@app.cell(hide_code=True)
def _(
    DEVICE,
    START,
    dataset,
    draw_world,
    make_batch,
    make_obs,
    np,
    override_latent,
    plt,
    posterior_mean,
    processor,
    torch,
    trained,
):
    _labels = {"VAE β=10 (기본값)": "VAE beta=10", "VAE β=0.01": "VAE beta=0.01", "VAE 없음": "no VAE"}
    _first = torch.nonzero(dataset["frame_index"] == 0).squeeze(1)
    _fig, _axes = plt.subplots(len(trained), 2, figsize=(10, 4.2 * len(trained)), squeeze=False)
    latent_summary = {}
    for _row, (_name, _res) in enumerate(trained.items()):
        _pol = _res["policy"]
        _obs = make_obs(START, (0.8, 0.0))
        _obs_rep = {k: v.expand(32, *v.shape[1:]) for k, v in _obs.items()}
        _z = torch.randn(32, _pol.config.latent_dim, generator=torch.Generator().manual_seed(0)).to(DEVICE)
        with override_latent(_pol, _z):
            _samples = processor.postprocess(_pol.predict_action_chunk(_obs_rep))
        _zero = processor.postprocess(_pol.predict_action_chunk(_obs))[0]
        _ax = _axes[_row, 0]
        draw_world(_ax, goal_y=0.0)
        for _s in _samples:
            _ax.plot(*_s.T, color="gray", alpha=0.4, lw=1)
        _ax.plot(*_zero.T, color="black", lw=2, label="z = 0")
        _ax.legend(loc="lower left")
        _ax.set_title(f"{_labels[_name]}: 32 chunks with z ~ N(0, I)")
        _end_y = _samples[:, -1, 1]
        latent_summary[_name] = {
            "up": int((_end_y > 0.05).sum()),
            "down": int((_end_y < -0.05).sum()),
            "chunk_end_x_std": float(_samples[:, -1, 0].std()),
        }

        _ax = _axes[_row, 1]
        if _pol.config.use_vae:
            _batch = processor.preprocess(make_batch(dataset, _first))
            _mu, _lv = posterior_mean(_pol, _batch)
            _mu = _mu.cpu()
            _centered = _mu - _mu.mean(0)
            _, _sv, _v = torch.linalg.svd(_centered, full_matrices=False)
            _proj = (_centered @ _v[:2].T).numpy()
            _modes = dataset["mode"][_first].numpy()
            _lens = dataset["n_steps"][_first].numpy()
            _ax.scatter(*_proj.T, c=np.where(_modes > 0, "tab:orange", "tab:blue"), s=(_lens - 25) * 2, alpha=0.7)
            _ax.set_title(f"posterior mean PCA (orange=up, blue=down)\nmean |mu| = {_mu.abs().mean():.3f}")
            latent_summary[_name]["mean_abs_mu"] = float(_mu.abs().mean())
        else:
            _ax.text(0.5, 0.5, "use_vae=False\n(no VAE encoder)", ha="center", va="center", transform=_ax.transAxes)
            _ax.set_axis_off()
    _fig.tight_layout()
    _fig
    return (latent_summary,)


@app.cell(hide_code=True)
def _(latent_summary, mo):
    _lines = [
        "| 모델 | z 샘플 중 위로 | 아래로 | 청크 끝 x의 표준편차 (속도 다양성) | 평균 \\|μ\\| |",
        "|---|---:|---:|---:|---:|",
        *[
            f"| {n} | {s['up']} | {s['down']} | {s['chunk_end_x_std']:.3f} | {s.get('mean_abs_mu', float('nan')):.3f} |"
            for n, s in latent_summary.items()
        ],
        "",
        "- $z$를 바꿔도 청크가 거의 같다면 모델은 $z$를 무시하는 것입니다(β=10의 사후 붕괴, 그리고 VAE가 없는 경우).",
        "- β가 작은 모델에서 $z$를 바꿀 때 **위/아래 경로와 속도가 달라진다면**, $z$가 관측에 없는 \"시연 스타일\"을 담았다는 뜻입니다. "
        "사후 평균 PCA에서 주황/파랑이 갈라지는지도 확인하세요.",
        "- 어느 경우든 추론에서는 $z=0$이 들어가므로 정책은 **결정적**으로 하나의 청크를 고릅니다.",
    ]
    mo.md("\n".join(_lines))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 8-1. 디코더 쿼리는 어디를 보는가

    디코더 cross-attention 가중치(헤드 평균)를 forward hook으로 꺼내 봅니다. 원본 코드는 `[0]`으로 출력만 쓰고
    가중치는 버리지만, `nn.MultiheadAttention`은 기본값으로 가중치도 함께 계산합니다.
    행은 행동 청크의 시점(0~19), 열은 인코더 토큰(`z`, `state`, 이미지 3×3)입니다.
    """)
    return


@app.cell(hide_code=True)
def _(make_obs, plt, torch, trained):
    _fig, _axes = plt.subplots(1, len(trained), figsize=(4.2 * len(trained), 4), squeeze=False)
    _labels = {"VAE β=10 (기본값)": "VAE beta=10", "VAE β=0.01": "VAE beta=0.01", "VAE 없음": "no VAE"}
    _obs = make_obs((-0.4, 0.35), (0.8, 0.25))
    for _i, (_name, _res) in enumerate(trained.items()):
        _pol = _res["policy"]
        _store = {}
        _h = _pol.model.decoder.layers[0].multihead_attn.register_forward_hook(
            lambda m, i, o, s=_store: s.__setitem__("w", o[1].detach())
        )
        with torch.no_grad():
            _pol.predict_action_chunk(_obs)
        _h.remove()
        _w = _store["w"][0].cpu()  # (k, n_tokens)
        _ax = _axes[0, _i]
        _ax.imshow(_w.numpy(), aspect="auto", cmap="magma")
        _ax.set_xticks(range(_w.shape[1]), ["z", "state"] + [f"img{j}" for j in range(_w.shape[1] - 2)], rotation=90, fontsize=7)
        _ax.set(title=_labels[_name], ylabel="action step in chunk")
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 9. 추론: 행동 큐와 시간 앙상블

    학습이 끝난 정책을 장난감 환경에서 닫힌 루프로 실행합니다. 환경은 매 스텝 행동(목표 위치) 쪽으로
    최대 0.15만큼 이동하고, 작은 구동 잡음이 섞입니다. 선택하면 정해진 스텝에 로봇을 **밀어(외란)** 반응성을 볼 수 있습니다.

    추론 방식은 학습과 무관한 설정이라, 학습된 정책의 `config.n_action_steps`와 `temporal_ensemble_coeff`만 바꿔 비교합니다.

    | 방식 | 모델 호출 | 특징 |
    |---|---|---|
    | `n_action_steps = k` | $k$ 스텝마다 1번 | 가장 빠름. 청크 경계에서 끊김이 생길 수 있고, 청크 실행 중에는 외란에 반응하지 못함 |
    | `n_action_steps = 1` | 매 스텝 | 반응은 빠르지만 매번 새 계획을 세워서 모드 사이를 오갈 수 있음 |
    | 시간 앙상블 (`n_action_steps=1`) | 매 스텝 | 같은 시점에 대한 과거·현재 예측을 지수 가중 평균 → 부드러움 |

    먼저 시간 앙상블의 온라인 평균이 오프라인 가중 평균과 같은지 원본 docstring의 예제로 확인합니다.
    """)
    return


@app.cell
def _(ACTTemporalEnsembler, torch):
    def check_temporal_ensembler(coeff=0.01, chunk=5, steps=8):
        """매 스텝 서로 다른 청크를 넣고, 각 시점의 출력이 오프라인 가중 평균과 같은지 비교."""
        torch.manual_seed(0)
        ens = ACTTemporalEnsembler(coeff, chunk)
        preds = torch.randn(steps, 1, chunk, 1)  # preds[s]: 스텝 s 에서 예측한 청크 (시점 s ~ s+chunk-1)
        w = torch.exp(-coeff * torch.arange(chunk))
        max_err = 0.0
        for t in range(steps):
            online = ens.update(preds[t])
            # 시점 t 를 예측한 청크들: s = max(0, t-chunk+1) ... t (오래된 것부터 w0, w1, ...)
            olds = [preds[s, 0, t - s, 0] for s in range(max(0, t - chunk + 1), t + 1)]
            offline = sum(w[i] * v for i, v in enumerate(olds)) / w[: len(olds)].sum()
            max_err = max(max_err, float((online[0, 0] - offline).abs()))
        return max_err

    {"coeff=0.01": check_temporal_ensembler(0.01), "coeff=0.5": check_temporal_ensembler(0.5), "coeff=-0.5": check_temporal_ensembler(-0.5)}
    return


@app.cell(hide_code=True)
def _(plt, torch):
    _fig, _ax = plt.subplots(figsize=(7, 3))
    for _m in [0.0, 0.01, 0.1, -0.1]:
        _w = torch.exp(-_m * torch.arange(20))
        _ax.plot(_w / _w.sum(), "o-", ms=3, label=f"coeff = {_m}")
    _ax.set(title="temporal ensemble weights over 20 overlapping predictions", xlabel="i (0 = oldest prediction)", ylabel="normalized weight")
    _ax.legend()
    _ax.grid(alpha=0.3)
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(ACTTemporalEnsembler, OBSTACLE_RADIUS, START, deque, make_obs, np, processor):
    def configure_inference(policy, n_action_steps: int, temporal_ensemble_coeff: float | None):
        """학습된 정책의 추론 설정만 바꾼다 (ACTConfig.__post_init__ 와 같은 규칙으로 검사)."""
        c = policy.config
        if temporal_ensemble_coeff is not None and n_action_steps > 1:
            raise NotImplementedError("`n_action_steps` must be 1 when using temporal ensembling.")
        if n_action_steps > c.chunk_size:
            raise ValueError("n_action_steps > chunk_size")
        c.n_action_steps = n_action_steps
        c.temporal_ensemble_coeff = temporal_ensemble_coeff
        if temporal_ensemble_coeff is not None:
            policy.temporal_ensembler = ACTTemporalEnsembler(temporal_ensemble_coeff, c.chunk_size)
        policy._action_queue = deque([], maxlen=n_action_steps)
        policy.reset()

    def rollout(policy, goal_y, *, max_steps=90, max_step_size=0.15, noise_std=0.005, push_step=None,
                push=(0.0, -0.25), seed=0):
        rng = np.random.default_rng(seed)
        goal = np.array([0.8, goal_y], dtype=np.float32)
        pos = START.astype(np.float64).copy()
        policy.reset()
        traj, actions, chunks = [pos.copy()], [], []
        handle = policy.model.register_forward_hook(
            lambda m, i, o: chunks.append((len(actions), processor.postprocess(o[0])[0].numpy()))
        )
        collided = False
        try:
            for t in range(max_steps):
                if push_step is not None and t == push_step:
                    pos = pos + np.asarray(push)
                action = processor.postprocess(policy.select_action(make_obs(pos, goal)))[0].numpy()
                step = action - pos
                norm = np.linalg.norm(step)
                if norm > max_step_size:
                    step = step / norm * max_step_size
                pos = pos + step + rng.normal(0, noise_std, 2)
                traj.append(pos.copy())
                actions.append(action)
                collided |= bool(np.linalg.norm(pos) < OBSTACLE_RADIUS)
        finally:
            handle.remove()
        traj, actions = np.array(traj), np.array(actions)
        jerk = np.linalg.norm(np.diff(traj, n=3, axis=0), axis=1).mean()
        return {
            "traj": traj,
            "actions": actions,
            "chunks": chunks,
            "goal": goal,
            "collided": collided,
            "success": (not collided) and np.linalg.norm(traj[-1] - goal) < 0.1,
            "n_calls": len(chunks),
            "jerk": float(jerk),
        }

    return configure_inference, rollout


@app.cell(hide_code=True)
def _(GOALS_Y, mo, trained):
    rollout_model = mo.ui.dropdown(options=list(trained), value=list(trained)[0], label="정책")
    rollout_steps = mo.ui.slider(1, 20, value=20, label="n_action_steps", show_value=True)
    rollout_te = mo.ui.switch(value=False, label="시간 앙상블 (n_action_steps=1로 고정)")
    rollout_coeff = mo.ui.slider(-0.2, 0.2, step=0.01, value=0.01, label="temporal_ensemble_coeff", show_value=True)
    rollout_goal = mo.ui.dropdown(options={str(g): g for g in GOALS_Y}, value="0.0", label="목표 y")
    rollout_push = mo.ui.switch(value=False, label="스텝 15에서 아래로 밀기")
    mo.vstack(
        [
            mo.hstack([rollout_model, rollout_goal, rollout_push], justify="start", gap=2),
            mo.hstack([rollout_steps, rollout_te, rollout_coeff], justify="start", gap=2),
        ]
    )
    return (
        rollout_coeff,
        rollout_goal,
        rollout_model,
        rollout_push,
        rollout_steps,
        rollout_te,
    )


@app.cell(hide_code=True)
def _(
    configure_inference,
    draw_world,
    mo,
    plt,
    rollout,
    rollout_coeff,
    rollout_goal,
    rollout_model,
    rollout_push,
    rollout_steps,
    rollout_te,
    trained,
):
    _pol = trained[rollout_model.value]["policy"]
    _te = rollout_coeff.value if rollout_te.value else None
    _n = 1 if rollout_te.value else rollout_steps.value
    configure_inference(_pol, _n, _te)
    _res = rollout(_pol, rollout_goal.value, push_step=15 if rollout_push.value else None)

    _fig, _axes = plt.subplots(1, 2, figsize=(11, 4.5))
    draw_world(_axes[0], goal_y=rollout_goal.value)
    for _t, _c in _res["chunks"][:: max(1, len(_res["chunks"]) // 12)]:
        _axes[0].plot(*_c.T, color="tab:orange", alpha=0.35, lw=1)
    _axes[0].plot(*_res["traj"].T, ".-", color="tab:blue", ms=3, lw=1.5, label="executed")
    _axes[0].plot([], [], color="tab:orange", alpha=0.5, label="predicted chunks")
    _axes[0].legend(loc="lower left", fontsize=8)
    _axes[0].set_title("success" if _res["success"] else ("collision" if _res["collided"] else "did not reach goal"))
    _speed = (_res["traj"][1:] - _res["traj"][:-1])
    _axes[1].plot(_speed[:, 0], label="dx")
    _axes[1].plot(_speed[:, 1], label="dy")
    for _t, _ in _res["chunks"]:
        _axes[1].axvline(_t, color="gray", alpha=0.15)
    _axes[1].set(title="per-step displacement (gray lines = policy calls)", xlabel="env step")
    _axes[1].legend()
    _fig.tight_layout()
    mo.vstack(
        [
            _fig,
            mo.md(
                f"모델 호출 {_res['n_calls']}회 · 평균 jerk {_res['jerk']:.4f} · "
                f"{'충돌' if _res['collided'] else '충돌 없음'} · 최종 거리 {((_res['traj'][-1] - _res['goal']) ** 2).sum() ** 0.5:.3f}"
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 9-1. 추론 방식 비교

    각 모델과 추론 방식마다 목표 3개 × 시드 4개 = 12번 실행해 평균을 냅니다. 외란 유무도 비교합니다.
    (GPU 기준 수십 초, CPU에서는 더 걸립니다.)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    compare_button = mo.ui.run_button(label="비교 실행")
    compare_button
    return (compare_button,)


@app.cell
def _(AUTORUN, GOALS_Y, compare_button, configure_inference, mo, np, rollout, trained):
    mo.stop(not (compare_button.value or AUTORUN), mo.md("**비교 실행**을 누르세요."))

    STRATEGIES = {
        "청크 20개 모두 실행": (20, None),
        "청크 중 10개 실행": (10, None),
        "매 스텝 재계획 (1개)": (1, None),
        "시간 앙상블 0.01": (1, 0.01),
    }
    comparison_rows = []
    with mo.status.progress_bar(total=len(trained) * len(STRATEGIES) * 2, title="롤아웃 비교", remove_on_exit=True) as _bar:
        for _name, _res in trained.items():
            for _sname, (_n, _te) in STRATEGIES.items():
                for _push in (None, 15):
                    configure_inference(_res["policy"], _n, _te)
                    _runs = [rollout(_res["policy"], g, push_step=_push, seed=s) for g in GOALS_Y for s in range(4)]
                    comparison_rows.append(
                        {
                            "모델": _name,
                            "추론 방식": _sname,
                            "외란": "있음" if _push else "없음",
                            "성공률": float(np.mean([r["success"] for r in _runs])),
                            "충돌률": float(np.mean([r["collided"] for r in _runs])),
                            "모델 호출 수": float(np.mean([r["n_calls"] for r in _runs])),
                            "jerk": round(float(np.mean([r["jerk"] for r in _runs])), 4),
                        }
                    )
                    _bar.update()
            configure_inference(_res["policy"], 20, None)  # 기본값으로 되돌림
    mo.ui.table(comparison_rows, selection=None, page_size=30)
    return (comparison_rows,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **생각해 볼 질문**

    1. 외란이 있을 때 "청크 20개 모두 실행"이 다른 방식보다 불리한 이유는? 이 과제에서는 행동이 **절대 목표 위치**라서
       밀린 뒤에도 원래 계획 지점으로 돌아가려 합니다. 행동이 상대 이동량(델타)이었다면 어떻게 달라질까요?
    2. 매 스텝 재계획과 시간 앙상블의 jerk(가가속도)를 비교해 보세요. 앙상블은 어떤 비용을 치르고 부드러움을 얻나요?
       (힌트: 모델 호출 수, 그리고 외란 후 반응 속도)
    3. 표에서 "매 스텝 재계획"의 성공률이 낮다면 위 롤아웃 뷰어에서 `n_action_steps=1`로 직접 실행해 보세요.
       로봇이 어디에 머무르나요? 시연은 **정지 상태에서 천천히 출발**하고, 관측(위치·이미지)에는 속도가 없습니다.
       출발점에서 예측한 청크의 **첫 행동**이 어떤 크기일지, 그리고 그 첫 행동만 매번 실행하면 어떻게 될지 생각해 보세요.
       청크를 여러 개 실행하거나 시간 앙상블을 쓰면 왜 이 문제가 사라지나요? (논문이 말하는 "비마르코프적 행동"과 연결됩니다.)
    4. 원본 ACT 논문은 `chunk_size=100`, 50Hz 제어를 씁니다. 2초 분량의 계획을 한 번에 세운다는 뜻입니다.
       장난감 과제의 `chunk_size=20`은 에피소드 길이(30~70 스텝)에 비해 짧은가요, 긴가요?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 10. 실제 LeRobot으로 옮겨 가기

    이 노트북의 코드는 `lerobot.policies.act`와 같은 구조이므로, 실제 로봇 데이터에서는 다음이 달라집니다.

    | 항목 | 이 노트북 | 실제 LeRobot |
    |---|---|---|
    | 데이터 | 즉석 생성한 2D 시연 100개 | `LeRobotDataset` (예: `lerobot/aloha_sim_insertion_human`, 직접 녹화한 SO-101 데이터) |
    | 행동 청크 | `chunk_indices`로 직접 생성 | `delta_timestamps`(= `action_delta_indices` / fps)로 데이터셋이 생성 |
    | 정규화 | `ToyProcessor` (MEAN_STD) | `make_act_pre_post_processors` + 데이터셋 `meta.stats` |
    | 백본 | ResNet18, 무작위 초기화 | ResNet18, ImageNet 사전 학습 가중치 |
    | 크기 | `dim_model=128` (6-1의 파라미터 표 참고) | `dim_model=512`, `dim_feedforward=3200`, VAE 인코더 4층 |
    | 저장 | 메모리 안에만 | `save_pretrained` / Hub 업로드 (`PreTrainedPolicy`) |

    LeRobot 문서의 학습·실행 예시는 다음과 같습니다(설치와 데이터 준비가 필요하며 이 노트북에서는 실행하지 않습니다).

    ```bash
    lerobot-train \
      --dataset.repo_id=${HF_USER}/your_dataset \
      --policy.type=act \
      --output_dir=outputs/train/act_your_dataset \
      --job_name=act_your_dataset \
      --policy.device=cuda \
      --policy.repo_id=${HF_USER}/act_policy
    ```

    `--policy.chunk_size=50`, `--policy.n_action_steps=50`, `--policy.kl_weight=...`처럼 `ACTConfig`의 필드를
    명령행에서 바로 바꿀 수 있습니다. 이 노트북에서 실험한 값들이 각각 어떤 의미인지 떠올리며 설정해 보세요.

    ## 정리

    - ACT = **ResNet 이미지 토큰 + 상태 토큰 + z 토큰**을 Transformer 인코더로 섞고, **학습되는 쿼리 $k$개**로
      행동 청크를 한 번에 회귀하는 정책입니다.
    - 손실은 **패딩을 제외한 L1** + `kl_weight` × **KL**. VAE 인코더는 학습 때만 쓰이고 추론 때는 $z=0$입니다.
    - `kl_weight`가 크면 $z$가 무시되는 사후 붕괴가 일어나기 쉽고, 작으면 $z$가 시연 스타일을 담지만 추론 때는 쓰이지 않습니다.
    - 추론 설정(`n_action_steps`, `temporal_ensemble_coeff`)은 학습과 독립적이며, 반응성·부드러움·호출 비용 사이의 절충입니다.

    ### 더 해 볼 과제

    1. `TOY_OVERRIDES`의 `chunk_size`를 5와 40으로 바꿔 학습하고, 성공률과 jerk를 비교하세요.
    2. `make_demo`에서 시연 길이를 50으로 고정(속도 스타일 제거)하면 β=0.01 모델의 KL과 z 샘플이 어떻게 변하나요?
    3. `pre_norm=True`로 바꿔 학습 곡선을 비교하세요. 인코더 끝의 `LayerNorm`이 언제 생기는지 코드에서 찾아보세요.
    4. 행동을 절대 위치 대신 **이동량(델타)** 으로 바꾸고(데이터셋·롤아웃 모두), 외란 실험을 다시 해 보세요.
    """)
    return


if __name__ == "__main__":
    app.run()
