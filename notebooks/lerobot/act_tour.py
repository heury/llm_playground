import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import math
    import random
    import time
    from collections import deque
    from collections.abc import Callable
    from dataclasses import dataclass, replace
    from itertools import chain

    import einops
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    import torch.nn.functional as F
    import torchvision
    from torch import Tensor, nn
    from torchvision.models._utils import IntermediateLayerGetter
    from torchvision.ops.misc import FrozenBatchNorm2d

    # GPU가 있으면 GPU, 없으면 CPU에서 실행합니다. CPU에서는 학습 시간이 10배 이상 길어집니다(6-1절 TRAIN_STEPS 참고).
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # lerobot/utils/constants.py의 배치 키 이름을 그대로 씁니다.
    ACTION = "action"
    OBS_STATE = "observation.state"
    OBS_ENV_STATE = "observation.environment_state"
    OBS_IMAGES = "observation.images"
    OBS_IMAGE_TOP = "observation.images.top"
    return (
        ACTION,
        Callable,
        F,
        FrozenBatchNorm2d,
        IntermediateLayerGetter,
        OBS_ENV_STATE,
        OBS_IMAGES,
        OBS_IMAGE_TOP,
        OBS_STATE,
        Tensor,
        chain,
        dataclass,
        deque,
        device,
        einops,
        math,
        mo,
        nn,
        np,
        plt,
        random,
        replace,
        time,
        torch,
        torchvision,
    )


@app.cell(hide_code=True)
def _(device, mo):
    mo.md(rf"""
    # ACT(Action Chunking with Transformers) 따라가기

    **ACT**는 로봇 팔 모방 학습(imitation learning) 정책입니다. 카메라 이미지와 관절 상태를 보고
    **다음 행동 하나가 아니라 앞으로 `k`개의 행동 묶음(action chunk)을 한 번에** 예측합니다.
    저가형 양팔 로봇 ALOHA 논문 [*Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware*](https://huggingface.co/papers/2304.13705)
    (Zhao et al., 2023)에서 제안되었고, Hugging Face [LeRobot](https://github.com/huggingface/lerobot)의 기본 정책 중 하나입니다.

    이 노트북은 LeRobot 저장소의 다음 파일을 기준으로 합니다(커밋 `e6ffde6`, 2026-11-12).

    | 파일 | 내용 |
    | --- | --- |
    | [`policies/act/configuration_act.py`](https://github.com/huggingface/lerobot/blob/main/src/lerobot/policies/act/configuration_act.py) | `ACTConfig`: 청크 크기, Transformer 크기, VAE, KL 가중치 |
    | [`policies/act/modeling_act.py`](https://github.com/huggingface/lerobot/blob/main/src/lerobot/policies/act/modeling_act.py) | `ACTPolicy`(손실·행동 큐), `ACTTemporalEnsembler`, `ACT`(신경망), 인코더/디코더 층, 위치 임베딩 |
    | [`policies/smolvla/`](https://github.com/huggingface/lerobot/tree/main/src/lerobot/policies/smolvla) | 같은 action chunk 구조에 VLM과 flow matching을 붙인 SmolVLA (9절에서 비교) |

    `lerobot` 패키지는 설치하지 않습니다. 이 노트북 환경(`notebooks/lerobot/pyproject.toml`)에는 torch·torchvision·einops만 있으므로,
    `modeling_act.py`의 클래스를 **거의 그대로 셀에 옮겨** 쓰고, 바꾼 곳은 주석으로 표시했습니다.
    원본 코드는 Apache License 2.0이며 저작권은 Tony Z. Zhao와 The HuggingFace Inc. team에 있습니다.

    **실습 과제:** 실제 로봇 대신 64×64 이미지로 그린 **2D 집어서 옮기기** 과제를 만들고, 스크립트로 만든 전문가 시연 200개로 학습합니다.
    데이터와 가중치를 내려받지 않습니다. 현재 장치는 **`{device}`** 입니다. RTX 3080 Ti 기준 전체 실행에 약 5분(학습 3회)이 걸리고 GPU 메모리는 1GB 미만을 씁니다.
    GPU 연산은 완전히 결정적이지 않아 성공률은 실행마다 몇 %p씩 달라질 수 있습니다(검증 실행 두 번: BC 0~1%, ACT 85~89%).

    **사전 지식:** Transformer의 self-attention·cross-attention, VAE의 재매개변수화(reparameterization)와 KL 항.

    ## 0. 한눈에 보는 ACT

    ```text
    학습할 때만 사용                          학습·추론 모두 사용
    ┌───────────────────────┐   z (32차원)   ┌─────────────────────────────────────────────┐
    │ VAE 인코더            │ ─────────────► │ Transformer 인코더                          │
    │ [CLS, 상태, a_t..a_t+k-1] │            │ [z, 상태, 이미지 특징 h·w개]                │
    └───────────────────────┘                │        │ cross-attention                     │
         (추론 때는 z = 0)                   │ Transformer 디코더: 학습되는 쿼리 k개 ──► k개 행동 │
                                             └─────────────────────────────────────────────┘
    ```

    | 아이디어 | 왜 필요한가 | 코드 위치 |
    | --- | --- | --- |
    | **Action chunking** | 행동 하나씩 고르면 작은 오차가 쌓이고(compounding error), "얼마나 기다렸는지" 같은 과거 정보를 모름 | `chunk_size`, `ACT.decoder_pos_embed`, `ACTPolicy.select_action` |
    | **CVAE** | 같은 장면에서도 사람 시연은 제각각(빠르게/느리게). 그 "스타일"을 잠재 변수 `z`로 따로 뺌 | `ACT.vae_encoder`, `kl_weight` |
    | **Temporal ensembling** | 청크 경계에서 행동이 끊기는 것을 매 step 겹쳐 평균해 부드럽게 함 | `ACTTemporalEnsembler` |

    1~2절에서 과제와 데이터를, 3~5절에서 모델 코드를, 6~8절에서 학습·평가·잠재 변수 실험을 다룹니다.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. 실습 과제: 2D 집어서 옮기기

    로봇(빨간 점)이 아래쪽에서 출발해 **물체(파란 원)** 로 가서 잠시 멈춰 물체를 잡고, **목표(초록 원)** 로 옮깁니다.
    좌표는 `[0, 1]²`이고 이미지와 같은 방향(y가 아래로 증가)입니다.

    | 항목 | 실제 로봇 (ALOHA) | 이 노트북 |
    | --- | --- | --- |
    | 이미지 관측 `observation.images.*` | 카메라 4대, 480×640 | 카메라 1대 `top`, 64×64 RGB |
    | 상태 `observation.state` | 관절 각도 14개 | 로봇 위치 `(x, y)` 2개 |
    | 행동 `action` | 다음 목표 관절 각도 14개 (절대값) | 다음 목표 위치 `(x, y)` (절대값) |
    | 제어 주기 | 50Hz | 1 step |

    환경 규칙(`env_step`):

    - 로봇은 행동(목표 위치) 쪽으로 이동하되 한 step에 최대 `MAX_STEP=0.06`만 움직입니다.
    - 물체에서 `NEAR=0.03` 이내에 **연속 `GRASP_STEPS=5` step 머물면** 물체를 잡습니다(`carry=True`). 잡은 물체는 로봇과 같이 움직입니다.
    - 물체를 든 채 목표에서 `0.05` 이내로 들어오면 성공입니다. 한 에피소드는 최대 `MAX_T=100` step입니다.

    **전문가 시연** 은 물체로 직진 → **5~20 step 중 무작위로 대기** → 목표로 직진 → 목표에서 5 step 정지입니다.
    대기 시간이 무작위라는 점이 핵심입니다. 사람이 시연하면 매번 머뭇거리는 시간이 다른 것과 같습니다.
    물체 위에서 기다리는 동안 이미지와 상태는 **매 step 똑같으므로**, 현재 관측만으로는 "몇 step 기다렸는지"를 알 수 없습니다.
    """)
    return


@app.cell
def _(math, np, torch):
    IMG = 64            # 이미지 한 변의 픽셀 수
    SPEED = 0.04        # 전문가의 한 step 이동 거리
    MAX_STEP = 0.06     # 환경이 허용하는 한 step 최대 이동 거리
    GRASP_STEPS = 5     # 물체 위에 연속으로 머물러야 잡히는 step 수
    WAIT_RANGE = (5, 20)  # 전문가가 물체 위에서 기다리는 step 수 (양 끝 포함, 무작위)
    NEAR = 0.03         # "물체 위에 있다"고 보는 거리
    GOAL_TOL = 0.05     # 성공 판정 거리
    MAX_T = 100         # 평가 에피소드 길이


    def sample_scene(rng):
        """출발점(아래), 물체(가운데), 목표(위)를 무작위로 놓습니다."""
        start = np.array([rng.uniform(0.2, 0.8), rng.uniform(0.8, 0.9)])
        obj = np.array([rng.uniform(0.2, 0.8), rng.uniform(0.45, 0.6)])
        goal = np.array([rng.uniform(0.2, 0.8), rng.uniform(0.1, 0.2)])
        return start, obj, goal


    def env_step(pos, action, obj, stay, carry):
        """여러 환경을 한 번에 1 step 진행합니다. pos/action/obj: (N, 2), stay: (N,) int, carry: (N,) bool"""
        delta = action - pos
        dist = np.linalg.norm(delta, axis=1, keepdims=True)
        new_pos = pos + delta * np.minimum(1.0, MAX_STEP / np.maximum(dist, 1e-9))
        near = np.linalg.norm(new_pos - obj, axis=1) < NEAR
        # 물체를 아직 안 잡았으면: 물체 위에 있으면 +1, 벗어나면 0으로 초기화
        stay = np.where(carry, stay, np.where(near, stay + 1, 0))
        carry = carry | (stay >= GRASP_STEPS)
        return new_pos, stay, carry


    def straight(a, b):
        """a에서 b까지 SPEED 간격의 목표 위치 목록 (b 포함)"""
        n = max(1, math.ceil(np.linalg.norm(b - a) / SPEED))
        return [a + (b - a) * (i / n) for i in range(1, n + 1)]


    def expert_episode(rng):
        """스크립트 전문가 시연 한 개: 물체로 직진 → 무작위 대기 → 목표로 직진 → 5 step 정지"""
        start, obj, goal = sample_scene(rng)
        wait = rng.randint(*WAIT_RANGE)
        approach = straight(start, obj)
        targets = approach + [obj.copy()] * wait + straight(obj, goal) + [goal.copy()] * 5
        pos, stay, carry = start[None].copy(), np.zeros(1, int), np.zeros(1, bool)
        states, carrying = [], []
        for target in targets:
            states.append(pos[0].copy())
            carrying.append(bool(carry[0]))
            pos, stay, carry = env_step(pos, target[None], obj[None], stay, carry)
        return dict(
            obj=obj, goal=goal, wait=wait,
            arrive_t=len(approach),  # 물체 위에 처음 도착한 프레임 번호 (8절에서 사용)
            state=np.array(states, dtype=np.float32),
            action=np.array(targets, dtype=np.float32),
            carrying=np.array(carrying),
        )


    def render(agent, obj, goal, carrying):
        """위치로부터 이미지를 그립니다. agent/obj/goal: (B, 2) 텐서, carrying: (B,) bool → (B, 3, IMG, IMG), 값 [0, 1]"""
        coords = (torch.arange(IMG, device=agent.device) + 0.5) / IMG
        yy, xx = torch.meshgrid(coords, coords, indexing="ij")

        def disk(p, r):
            return (((xx - p[:, 0, None, None]) ** 2 + (yy - p[:, 1, None, None]) ** 2) < r * r).float()

        obj_pos = torch.where(carrying[:, None], agent, obj)  # 잡은 물체는 로봇 위치에 그림
        img = torch.full((agent.shape[0], 3, IMG, IMG), 0.1, device=agent.device)
        img[:, 1] = torch.maximum(img[:, 1], 0.8 * disk(goal, 0.07))     # 초록: 목표
        img[:, 2] = torch.maximum(img[:, 2], 0.9 * disk(obj_pos, 0.045))  # 파랑: 물체
        img[:, 0] = torch.maximum(img[:, 0], 0.9 * disk(agent, 0.03))     # 빨강: 로봇
        return img
    return (
        GOAL_TOL,
        GRASP_STEPS,
        IMG,
        MAX_T,
        WAIT_RANGE,
        env_step,
        expert_episode,
        render,
        sample_scene,
    )


@app.cell
def _(expert_episode, np, random):
    N_EPISODES = 200
    episodes = [expert_episode(random.Random(1000 + i)) for i in range(N_EPISODES)]

    _lens = np.array([len(e["state"]) for e in episodes])
    _waits = np.array([e["wait"] for e in episodes])
    print(f"시연 {N_EPISODES}개, 프레임 {_lens.sum()}개, 에피소드 길이 {_lens.min()}~{_lens.max()} (평균 {_lens.mean():.1f})")
    print(f"대기 step: {_waits.min()}~{_waits.max()} (평균 {_waits.mean():.1f})")
    return (episodes,)


@app.cell
def _(device, episodes, np, plt, render, torch):
    def show_frame(ax, ep, t):
        """에피소드 ep의 t번째 프레임 이미지를 그립니다."""
        f = lambda x, dt=torch.float32: torch.tensor(np.array([x]), dtype=dt, device=device)
        img = render(f(ep["state"][t]), f(ep["obj"]), f(ep["goal"]), f(ep["carrying"][t], torch.bool))[0]
        ax.imshow(img.permute(1, 2, 0).cpu().numpy(), extent=[0, 1, 1, 0])
        ax.set_xticks([]); ax.set_yticks([])


    _fig, _axes = plt.subplots(2, 4, figsize=(12, 6.2))
    for _i, _ax in enumerate(_axes[0]):
        _ep = episodes[_i]
        show_frame(_ax, _ep, 0)
        _ax.plot(_ep["state"][:, 0], _ep["state"][:, 1], "w.-", ms=3, lw=0.8)
        _ax.set_title(f"demo {_i}: wait {_ep['wait']} steps")
    _ep = episodes[0]
    for _ax, _t, _name in zip(_axes[1], [0, _ep["arrive_t"], _ep["arrive_t"] + _ep["wait"], len(_ep["state"]) - 1],
                              ["start", "arrive at object", "last waiting frame", "end"]):
        show_frame(_ax, _ep, _t)
        _ax.set_title(f"demo 0, t={_t}: {_name}")
    _fig.suptitle("Top: first frame + expert path (white). Bottom: frames of demo 0")
    _fig.tight_layout()
    _fig
    return (show_frame,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    아래쪽 두 번째와 세 번째 그림(물체에 도착한 프레임, 마지막 대기 프레임)은 **완전히 같은 이미지**입니다.
    그런데 그다음 행동은 다릅니다. 하나는 "더 기다려", 하나는 "출발해"입니다.
    현재 관측만 보고 행동 하나를 고르는 정책에게 이 상황이 어떻게 보이는지 시연 데이터에서 직접 세어 봅니다.
    """)
    return


@app.cell
def _(episodes, np):
    # "물체를 아직 안 들었고 물체 위에서 기다리는" 프레임에서, 다음 행동이 '제자리'인지 '출발'인지 셉니다.
    _stay = _go = 0
    for _ep in episodes:
        for _t in range(_ep["arrive_t"], len(_ep["state"])):
            _at_obj = np.linalg.norm(_ep["state"][_t] - _ep["obj"]) < 1e-5
            if not _at_obj:
                break
            if np.linalg.norm(_ep["action"][_t] - _ep["obj"]) < 1e-5:
                _stay += 1
            else:
                _go += 1
    print(f"물체 위 대기 프레임 {_stay + _go}개 중 다음 행동이 '제자리' {_stay}개 ({_stay / (_stay + _go):.0%}), '출발' {_go}개 ({_go / (_stay + _go):.0%})")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    관측이 똑같은 프레임들에서 정답 행동의 약 92%가 "제자리"입니다. 이 데이터로 **행동 하나**를 L1 손실로 학습하면,
    모델은 이 관측에서 L1을 가장 줄이는 값, 즉 **중앙값인 "제자리"** 를 출력할 것으로 **예상**됩니다. 그러면 로봇은 물체를 잡은 뒤에도
    같은 관측을 계속 보므로 출발하지 못합니다. 과거를 기억하지 못하는(마르코프) 정책의 전형적인 실패입니다.
    (7절에서 이 예상이 실제로는 어떻게 나타나는지 확인합니다. 예상과 다른 실패도 나옵니다.)

    ACT의 답은 **행동을 묶음으로 예측**하는 것입니다. 도착한 순간 `k=20`개의 행동을 한 번에 내면 "몇 step 기다린 뒤 출발"이
    하나의 출력에 들어가고, 그 묶음을 다 실행할 때까지 다시 묻지 않으므로 "얼마나 기다렸나"를 기억할 필요가 없습니다.
    7절에서 이 차이를 실제로 학습해 확인합니다.

    ## 2. LeRobot 학습 배치 만들기

    LeRobot의 `LeRobotDataset`은 프레임 하나를 꺼낼 때 `delta_timestamps`에 따라 **미래 행동 묶음** 을 같이 붙여 줍니다.
    ACT는 `ACTConfig.action_delta_indices = list(range(chunk_size))`이므로 프레임 `t`에 대해 행동 `t, t+1, ..., t+k-1`을 받습니다.

    | 배치 키 | 모양 | 의미 |
    | --- | --- | --- |
    | `observation.images.top` | `(B, 3, 64, 64)` | 프레임 `t`의 이미지 |
    | `observation.state` | `(B, 2)` | 프레임 `t`의 상태 |
    | `action` | `(B, k, 2)` | 행동 `t ... t+k-1` |
    | `action_is_pad` | `(B, k)` | 에피소드 끝을 넘어간 위치는 `True`. 그 자리는 마지막 행동으로 채움 |

    `action_is_pad`가 `True`인 위치는 손실에서 빠집니다(5절 `ACTPolicy.forward`). 에피소드 끝 근처 프레임도 버리지 않고 쓰기 위한 장치입니다.

    **정규화:** `ACTConfig.normalization_mapping`의 기본값은 이미지·상태·행동 모두 `MEAN_STD`입니다.
    LeRobot에서는 `processor_act.py`의 전처리기가 데이터셋 통계로 `(x - mean) / std`를 하고, 정책 출력은 후처리기가 되돌립니다.
    여기서는 같은 일을 `obs_batch`(입력)와 `unnormalize_action`(출력)이 합니다.
    """)
    return


@app.cell
def _(device, episodes, np, render, torch):
    MAX_CHUNK = 20  # 데이터는 가장 긴 청크 기준으로 한 번 만들고, 짧은 청크는 앞부분만 잘라 씁니다.


    def build_frames(episodes, chunk):
        """모든 프레임을 (상태, 장면, 행동 묶음, 패딩 마스크) 텐서로 펼칩니다. 이미지는 학습 때 GPU에서 그립니다."""
        cols = {k: [] for k in ["state", "obj", "goal", "carrying", "action", "action_is_pad"]}
        for ep in episodes:
            length = len(ep["state"])
            for t in range(length):
                idx = np.arange(t, t + chunk)
                cols["action_is_pad"].append(idx >= length)             # 에피소드 밖이면 패딩
                cols["action"].append(ep["action"][np.minimum(idx, length - 1)])  # 패딩 자리는 마지막 행동 반복
                cols["state"].append(ep["state"][t])
                cols["obj"].append(ep["obj"])
                cols["goal"].append(ep["goal"])
                cols["carrying"].append(ep["carrying"][t])
        dtypes = {"carrying": torch.bool, "action_is_pad": torch.bool}
        return {k: torch.tensor(np.array(v), dtype=dtypes.get(k, torch.float32), device=device) for k, v in cols.items()}


    frames = build_frames(episodes, MAX_CHUNK)
    N_FRAMES = len(frames["state"])

    # 데이터셋 통계 (LeRobot의 meta/stats.json에 해당)
    stats = {
        "state": (frames["state"].mean(0), frames["state"].std(0)),
        "action": (frames["action"][:, 0].mean(0), frames["action"][:, 0].std(0)),
    }
    _imgs = render(frames["state"][::5], frames["obj"][::5], frames["goal"][::5], frames["carrying"][::5])
    stats["image"] = (_imgs.mean((0, 2, 3))[:, None, None], _imgs.std((0, 2, 3))[:, None, None])
    print({k: [round(x, 3) for x in torch.cat(v).flatten().tolist()] for k, v in stats.items()})
    return MAX_CHUNK, N_FRAMES, frames, stats


@app.cell
def _(ACTION, OBS_IMAGE_TOP, OBS_STATE, N_FRAMES, device, frames, render, stats, torch):
    def obs_batch(state, obj, goal, carrying):
        """현재 위치·장면 → 정규화된 관측 배치 (LeRobot 전처리기 역할)"""
        img = render(state, obj, goal, carrying)
        return {
            OBS_IMAGE_TOP: (img - stats["image"][0]) / stats["image"][1],
            OBS_STATE: (state - stats["state"][0]) / stats["state"][1],
        }


    def unnormalize_action(action):
        """정규화된 행동 → 실제 좌표 (LeRobot 후처리기 역할)"""
        return action * stats["action"][1] + stats["action"][0]


    def train_batch(batch_size, chunk, generator=None):
        """무작위 프레임 batch_size개로 학습 배치를 만듭니다."""
        ix = torch.randint(0, N_FRAMES, (batch_size,), device=device, generator=generator)
        batch = obs_batch(frames["state"][ix], frames["obj"][ix], frames["goal"][ix], frames["carrying"][ix])
        batch[ACTION] = (frames["action"][ix, :chunk] - stats["action"][0]) / stats["action"][1]
        batch["action_is_pad"] = frames["action_is_pad"][ix, :chunk]
        return batch


    _b = train_batch(4, 20)
    print({k: tuple(v.shape) for k, v in _b.items()})
    print("패딩이 있는 프레임 비율:", f"{frames['action_is_pad'].any(1).float().mean().item():.1%}")
    return obs_batch, train_batch, unnormalize_action


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    청크가 20이면 에피소드 마지막 19개 프레임은 패딩을 포함합니다. 에피소드가 평균 40 step 정도로 짧아 꽤 큰 비율입니다.

    ## 3. 설정: `ACTConfig`

    원본 `ACTConfig`는 LeRobot의 `PreTrainedConfig`를 상속해 `input_features`/`output_features`에서
    `robot_state_feature`, `image_features`, `action_feature`를 계산합니다. 여기서는 그 속성만 같은 이름으로 흉내 낸
    작은 dataclass를 씁니다. 검증 규칙(`__post_init__`)은 원본과 같습니다.

    | 설정 | LeRobot 기본값 (ALOHA) | 이 노트북 | 이유 |
    | --- | --- | --- | --- |
    | `chunk_size` / `n_action_steps` | 100 / 100 | 20 / 20 | 에피소드가 40 step 정도로 짧음 |
    | `dim_model` / `dim_feedforward` | 512 / 3200 | 256 / 1024 | 작은 과제, 빠른 학습 |
    | `n_encoder_layers` / `n_decoder_layers` | 4 / **1** | 4 / 1 | 같음 (아래 설명) |
    | `n_vae_encoder_layers` / `latent_dim` | 4 / 32 | 4 / 32 | 같음 |
    | `pretrained_backbone_weights` | ImageNet ResNet18 | `None` (무작위 초기화) | 내려받기 없이 실행. 단순한 그림이라 충분 |
    | `kl_weight` | 10.0 | 10.0 (8절에서 0.01과 비교) | |
    | `optimizer_lr` | 1e-5 | 1e-4 | 원본은 약 10만 step, 여기서는 3천 step |

    `n_decoder_layers`가 1인 이유: 원본 ACT 코드는 7로 설정했지만 버그 때문에 **첫 층만 실제로 쓰였습니다**
    ([tonyzhaozh/act#25](https://github.com/tonyzhaozh/act/issues/25)). LeRobot은 그 동작에 맞춰 1을 씁니다.
    """)
    return


@app.cell
def _(dataclass):
    @dataclass
    class Feature:
        shape: tuple


    @dataclass
    class ACTConfig:
        # 입력·출력 구조 (원본은 input_features/output_features에서 계산)
        state_dim: int = 2
        action_dim: int = 2
        image_keys: tuple = ("observation.images.top",)
        image_shape: tuple = (3, 64, 64)

        # 원본 configuration_act.py의 필드 (기본값만 작게 바꿈)
        n_obs_steps: int = 1
        chunk_size: int = 20
        n_action_steps: int = 20
        vision_backbone: str = "resnet18"
        pretrained_backbone_weights: str | None = None  # 원본: "ResNet18_Weights.IMAGENET1K_V1"
        replace_final_stride_with_dilation: int = False
        pre_norm: bool = False
        dim_model: int = 256          # 원본 512
        n_heads: int = 8
        dim_feedforward: int = 1024   # 원본 3200
        feedforward_activation: str = "relu"
        n_encoder_layers: int = 4
        n_decoder_layers: int = 1
        use_vae: bool = True
        latent_dim: int = 32
        n_vae_encoder_layers: int = 4
        temporal_ensemble_coeff: float | None = None
        dropout: float = 0.1
        kl_weight: float = 10.0
        optimizer_lr: float = 1e-4    # 원본 1e-5
        optimizer_weight_decay: float = 1e-4
        optimizer_lr_backbone: float = 1e-4  # 원본 1e-5

        def __post_init__(self):
            if self.temporal_ensemble_coeff is not None and self.n_action_steps > 1:
                raise NotImplementedError("`n_action_steps` must be 1 when using temporal ensembling.")
            if self.n_action_steps > self.chunk_size:
                raise ValueError("`n_action_steps` must be <= `chunk_size`.")
            if self.n_obs_steps != 1:
                raise ValueError("Multiple observation steps not handled yet.")

        # PreTrainedConfig가 제공하던 속성들
        @property
        def robot_state_feature(self):
            return Feature((self.state_dim,)) if self.state_dim else None

        @property
        def env_state_feature(self):
            return None  # 물체 좌표 같은 환경 상태 벡터는 쓰지 않고, 이미지로만 봅니다.

        @property
        def image_features(self):
            return {key: Feature(self.image_shape) for key in self.image_keys}

        @property
        def action_feature(self):
            return Feature((self.action_dim,))

        def validate_features(self):
            if not self.image_features and not self.env_state_feature:
                raise ValueError("You must provide at least one image or the environment state among the inputs.")
    return (ACTConfig,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. 모델 구성 요소

    ### 4-1. 위치 임베딩

    ACT는 위치 정보를 세 가지 방식으로 넣습니다.

    | 대상 | 방식 | 함수/모듈 |
    | --- | --- | --- |
    | VAE 인코더 입력 `[CLS, 상태, 행동 k개]` | 고정 1D sin/cos | `create_sinusoidal_pos_embedding` |
    | Transformer 인코더의 이미지 특징 지도 `h × w` | 고정 2D sin/cos (y 절반 + x 절반) | `ACTSinusoidalPositionEmbedding2d` |
    | Transformer 인코더의 1D 토큰 `[z, 상태]` | 학습 | `encoder_1d_feature_pos_embed` (`nn.Embedding`) |
    | Transformer 디코더 쿼리 `k`개 | 학습 (DETR의 object query와 같은 방식) | `decoder_pos_embed` (`nn.Embedding`) |

    GPT처럼 입력에 위치 임베딩을 더하는 대신, **매 층의 attention에서 query와 key에만 더하고 value에는 더하지 않습니다**(DETR 방식).
    """)
    return


@app.cell
def _(Callable, F, Tensor, math, nn, np, torch):
    # ── 아래는 lerobot/policies/act/modeling_act.py에서 옮긴 코드입니다. ──

    def create_sinusoidal_pos_embedding(num_positions: int, dimension: int) -> Tensor:
        """1D sinusoidal positional embeddings as in Attention is All You Need."""

        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / dimension) for hid_j in range(dimension)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(num_positions)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
        return torch.from_numpy(sinusoid_table).float()


    class ACTSinusoidalPositionEmbedding2d(nn.Module):
        """2D sinusoidal positional embeddings. 위치 번호를 [0, 2π] 범위로 정규화합니다(정확히는 1/H ~ 1)."""

        def __init__(self, dimension: int):
            super().__init__()
            self.dimension = dimension
            self._two_pi = 2 * math.pi
            self._eps = 1e-6
            self._temperature = 10000

        def forward(self, x: Tensor) -> Tensor:
            """x: (B, C, H, W) 특징 지도 → (1, C, H, W) 위치 임베딩 (배치와 무관)"""
            not_mask = torch.ones_like(x[0, :1])  # (1, H, W)
            y_range = not_mask.cumsum(1, dtype=torch.float32)  # 1..H
            x_range = not_mask.cumsum(2, dtype=torch.float32)  # 1..W

            y_range = y_range / (y_range[:, -1:, :] + self._eps) * self._two_pi
            x_range = x_range / (x_range[:, :, -1:] + self._eps) * self._two_pi

            inverse_frequency = self._temperature ** (
                2 * (torch.arange(self.dimension, dtype=torch.float32, device=x.device) // 2) / self.dimension
            )

            x_range = x_range.unsqueeze(-1) / inverse_frequency  # (1, H, W, C//2)
            y_range = y_range.unsqueeze(-1) / inverse_frequency

            pos_embed_x = torch.stack((x_range[..., 0::2].sin(), x_range[..., 1::2].cos()), dim=-1).flatten(3)
            pos_embed_y = torch.stack((y_range[..., 0::2].sin(), y_range[..., 1::2].cos()), dim=-1).flatten(3)
            pos_embed = torch.cat((pos_embed_y, pos_embed_x), dim=3).permute(0, 3, 1, 2)  # (1, C, H, W)
            return pos_embed


    def get_activation_fn(activation: str) -> Callable:
        """Return an activation function given a string."""
        if activation == "relu":
            return F.relu
        if activation == "gelu":
            return F.gelu
        if activation == "glu":
            return F.glu
        raise RuntimeError(f"activation should be relu/gelu/glu, not {activation}.")
    return (
        ACTSinusoidalPositionEmbedding2d,
        create_sinusoidal_pos_embedding,
        get_activation_fn,
    )


@app.cell
def _(ACTSinusoidalPositionEmbedding2d, create_sinusoidal_pos_embedding, plt, torch):
    _pe1d = create_sinusoidal_pos_embedding(22, 256)  # VAE 인코더: CLS + 상태 + 행동 20개 = 22 토큰
    _pe2d = ACTSinusoidalPositionEmbedding2d(128)(torch.zeros(1, 512, 8, 8))  # dim_model // 2 = 128 → 출력 채널 256

    _fig, _axes = plt.subplots(1, 3, figsize=(13, 3.2))
    _axes[0].imshow(_pe1d.numpy(), aspect="auto", cmap="RdBu")
    _axes[0].set(title="1D sin/cos (22 tokens x 256)", xlabel="channel", ylabel="token")
    _axes[1].imshow(_pe2d[0, 0].numpy(), cmap="RdBu")
    _axes[1].set(title="2D, channel 0 (y part)")
    _axes[2].imshow(_pe2d[0, 128].numpy(), cmap="RdBu")
    _axes[2].set(title="2D, channel 128 (x part)")
    _fig.tight_layout()
    print("2D 임베딩 모양:", tuple(_pe2d.shape), "→ 앞 128채널은 y 위치, 뒤 128채널은 x 위치")
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    `ACTSinusoidalPositionEmbedding2d(dimension)`는 y용 `dimension`채널과 x용 `dimension`채널을 이어 붙여 **`2 × dimension`채널**을 냅니다.
    그래서 원본은 `ACTSinusoidalPositionEmbedding2d(config.dim_model // 2)`로 만들어 `dim_model`채널을 맞춥니다.

    ### 4-2. 인코더 층과 디코더 층

    `nn.TransformerEncoderLayer`와 거의 같지만 위치 임베딩을 **q, k에만** 더한다는 점이 다릅니다.
    기본값 `pre_norm=False`이므로 원본 Transformer처럼 **post-norm**(잔차 더한 뒤 LayerNorm)입니다.
    텐서 모양은 `(seq, batch, dim)`입니다(`nn.MultiheadAttention`의 기본 `batch_first=False`).

    디코더 층은 self-attention → **cross-attention(query: 디코더, key/value: 인코더 출력)** → FFN 순서입니다.
    마스크가 없습니다. GPT와 달리 행동 `k`개를 **한 번에 동시에** 만들기 때문에 미래를 가릴 필요가 없습니다.
    """)
    return


@app.cell
def _(ACTConfig, Tensor, get_activation_fn, nn):
    class ACTEncoderLayer(nn.Module):
        def __init__(self, config: ACTConfig):
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
            q = k = x if pos_embed is None else x + pos_embed  # 위치 임베딩은 q, k에만
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
        """여러 인코더 층을 쌓습니다. is_vae_encoder=True면 VAE 인코더용 층 수를 씁니다."""

        def __init__(self, config: ACTConfig, is_vae_encoder: bool = False):
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
        def __init__(self, config: ACTConfig):
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

        def forward(self, x: Tensor, encoder_out: Tensor, decoder_pos_embed: Tensor | None = None,
                    encoder_pos_embed: Tensor | None = None) -> Tensor:
            """x: (DS, B, C) 디코더 토큰, encoder_out: (ES, B, C) → (DS, B, C)"""
            # (1) self-attention: 행동 쿼리끼리 정보를 주고받음 (마스크 없음)
            skip = x
            if self.pre_norm:
                x = self.norm1(x)
            q = k = self.maybe_add_pos_embed(x, decoder_pos_embed)
            x = self.self_attn(q, k, value=x)[0]
            x = skip + self.dropout1(x)
            if self.pre_norm:
                skip = x
                x = self.norm2(x)
            else:
                x = self.norm1(x)
                skip = x
            # (2) cross-attention: 행동 쿼리가 인코더 토큰(z, 상태, 이미지)을 읽음
            x = self.multihead_attn(
                query=self.maybe_add_pos_embed(x, decoder_pos_embed),
                key=self.maybe_add_pos_embed(encoder_out, encoder_pos_embed),
                value=encoder_out,
            )[0]
            x = skip + self.dropout2(x)
            if self.pre_norm:
                skip = x
                x = self.norm3(x)
            else:
                x = self.norm2(x)
                skip = x
            # (3) FFN
            x = self.linear2(self.dropout(self.activation(self.linear1(x))))
            x = skip + self.dropout3(x)
            if not self.pre_norm:
                x = self.norm3(x)
            return x


    class ACTDecoder(nn.Module):
        def __init__(self, config: ACTConfig):
            super().__init__()
            self.layers = nn.ModuleList([ACTDecoderLayer(config) for _ in range(config.n_decoder_layers)])
            self.norm = nn.LayerNorm(config.dim_model)

        def forward(self, x: Tensor, encoder_out: Tensor, decoder_pos_embed: Tensor | None = None,
                    encoder_pos_embed: Tensor | None = None) -> Tensor:
            for layer in self.layers:
                x = layer(x, encoder_out, decoder_pos_embed=decoder_pos_embed, encoder_pos_embed=encoder_pos_embed)
            if self.norm is not None:
                x = self.norm(x)
            return x
    return ACTDecoder, ACTEncoder


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 4-3. 전체 신경망 `ACT`

    `ACT.forward`는 세 단계입니다. 원본 코드를 그대로 옮기고 단계마다 한국어 주석을 달았습니다.

    1. **잠재 변수 `z` 준비**
       - 학습(`self.training`이고 `use_vae`이고 배치에 `action`이 있음): VAE 인코더가 `[CLS, 상태, 정답 행동 k개]`를 읽고
         CLS 출력에서 평균 `mu`와 `log σ²`(코드 이름 `log_sigma_x2`)를 냅니다. `z = mu + σ·ε`로 샘플링합니다(재매개변수화).
         패딩된 행동은 `key_padding_mask`로 가립니다.
       - 추론: **`z = 0`** (사전 분포 N(0, I)의 평균). VAE 인코더는 쓰지 않습니다.
    2. **Transformer 인코더**: 토큰 `[z, 상태, 이미지 특징 h·w개]`. 이미지는 ResNet18의 `layer4` 특징 지도(512채널)를
       1×1 conv로 `dim_model`채널로 줄이고, 픽셀 하나를 토큰 하나로 펼칩니다. 64×64 이미지면 `layer4`가 2×2이므로 이미지 토큰은 4개입니다.
    3. **Transformer 디코더**: 입력은 **0으로 채운 `k`개 토큰**이고, 각 자리의 구분은 학습되는 `decoder_pos_embed`가 맡습니다.
       출력 `k`개를 `action_head`(Linear)에 통과시켜 행동 `k`개를 **한 번에** 얻습니다.

    ResNet은 `FrozenBatchNorm2d`를 씁니다. 배치 정규화 통계와 affine 값이 학습되지 않습니다(사전 학습 가중치를 쓸 때 ImageNet 값 유지).
    여기서는 무작위 초기화이므로 BN은 사실상 항등 함수이고, 합성곱 가중치만 학습됩니다.
    """)
    return


@app.cell
def _(
    ACTConfig,
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
        """Action Chunking Transformer: ACTPolicy 안의 신경망 (lerobot modeling_act.py)"""

        def __init__(self, config: ACTConfig) -> None:
            super().__init__()
            self.config = config
            action_feature = config.action_feature

            # ── VAE 인코더 (BERT 방식: 입력 토큰 [cls, robot_state, *action_sequence]) ──
            if self.config.use_vae:
                self.vae_encoder = ACTEncoder(config, is_vae_encoder=True)
                self.vae_encoder_cls_embed = nn.Embedding(1, config.dim_model)
                if self.config.robot_state_feature:
                    self.vae_encoder_robot_state_input_proj = nn.Linear(
                        self.config.robot_state_feature.shape[0], config.dim_model
                    )
                self.vae_encoder_action_input_proj = nn.Linear(action_feature.shape[0], config.dim_model)
                # CLS 출력 → [mu (latent_dim), log σ² (latent_dim)]
                self.vae_encoder_latent_output_proj = nn.Linear(config.dim_model, config.latent_dim * 2)
                num_input_token_encoder = 1 + config.chunk_size
                if self.config.robot_state_feature:
                    num_input_token_encoder += 1
                self.register_buffer(
                    "vae_encoder_pos_enc",
                    create_sinusoidal_pos_embedding(num_input_token_encoder, config.dim_model).unsqueeze(0),
                )

            # ── 이미지 백본: ResNet의 layer4 특징 지도만 꺼냄 ──
            if self.config.image_features:
                backbone_model = getattr(torchvision.models, config.vision_backbone)(
                    replace_stride_with_dilation=[False, False, config.replace_final_stride_with_dilation],
                    weights=config.pretrained_backbone_weights,
                    norm_layer=FrozenBatchNorm2d,
                )
                self.backbone = IntermediateLayerGetter(backbone_model, return_layers={"layer4": "feature_map"})

            # ── Transformer (VAE로 학습할 때는 VAE 디코더 역할) ──
            self.encoder = ACTEncoder(config)
            self.decoder = ACTDecoder(config)

            # 인코더 입력 투영. 토큰 순서: [latent, (robot_state), (env_state), (image_feature_map_pixels)]
            if self.config.robot_state_feature:
                self.encoder_robot_state_input_proj = nn.Linear(
                    self.config.robot_state_feature.shape[0], config.dim_model
                )
            if self.config.env_state_feature:
                self.encoder_env_state_input_proj = nn.Linear(
                    self.config.env_state_feature.shape[0], config.dim_model
                )
            self.encoder_latent_input_proj = nn.Linear(config.latent_dim, config.dim_model)
            if self.config.image_features:
                self.encoder_img_feat_input_proj = nn.Conv2d(
                    backbone_model.fc.in_features, config.dim_model, kernel_size=1
                )
            # 1D 토큰(latent, state, env_state)의 학습되는 위치 임베딩
            n_1d_tokens = 1
            if self.config.robot_state_feature:
                n_1d_tokens += 1
            if self.config.env_state_feature:
                n_1d_tokens += 1
            self.encoder_1d_feature_pos_embed = nn.Embedding(n_1d_tokens, config.dim_model)
            if self.config.image_features:
                self.encoder_cam_feat_pos_embed = ACTSinusoidalPositionEmbedding2d(config.dim_model // 2)

            # 디코더 쿼리의 학습되는 위치 임베딩 (DETR의 object query와 같은 방식)
            self.decoder_pos_embed = nn.Embedding(config.chunk_size, config.dim_model)

            # 디코더 출력 → 행동
            self.action_head = nn.Linear(config.dim_model, action_feature.shape[0])

            self._reset_parameters()

        def _reset_parameters(self):
            """Xavier-uniform initialization of the transformer parameters as in the original code."""
            for p in chain(self.encoder.parameters(), self.decoder.parameters()):
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)

        def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, tuple[Tensor, Tensor] | tuple[None, None]]:
            """Returns: (B, chunk_size, action_dim) 행동, (mu, log σ²) 각각 (B, latent_dim) 또는 (None, None)"""
            if self.config.use_vae and self.training:
                assert ACTION in batch, "actions must be provided when using the variational objective in training mode."

            batch_size = batch[OBS_IMAGES][0].shape[0] if OBS_IMAGES in batch else batch[OBS_ENV_STATE].shape[0]

            # ── 1단계: 잠재 변수 z ──
            if self.config.use_vae and ACTION in batch and self.training:
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

                pos_embed = self.vae_encoder_pos_enc.clone().detach()  # (1, S+2, D)

                # CLS와 상태 토큰은 패딩이 아님(False). 행동 토큰은 action_is_pad를 그대로 씀.
                cls_joint_is_pad = torch.full(
                    (batch_size, 2 if self.config.robot_state_feature else 1), False, device=batch[OBS_STATE].device
                )
                key_padding_mask = torch.cat([cls_joint_is_pad, batch["action_is_pad"]], axis=1)  # (B, S+2)

                cls_token_out = self.vae_encoder(
                    vae_encoder_input.permute(1, 0, 2),
                    pos_embed=pos_embed.permute(1, 0, 2),
                    key_padding_mask=key_padding_mask,
                )[0]  # CLS 토큰 출력만 (B, D)
                latent_pdf_params = self.vae_encoder_latent_output_proj(cls_token_out)
                mu = latent_pdf_params[:, : self.config.latent_dim]
                log_sigma_x2 = latent_pdf_params[:, self.config.latent_dim :]  # 2·log σ = log σ²

                # 재매개변수화: z = mu + σ·ε
                latent_sample = mu + log_sigma_x2.div(2).exp() * torch.randn_like(mu)
            else:
                # 추론(또는 VAE 미사용): z = 0
                mu = log_sigma_x2 = None
                latent_sample = torch.zeros([batch_size, self.config.latent_dim], dtype=torch.float32).to(
                    batch[OBS_STATE].device
                )

            # ── 2단계: Transformer 인코더 입력 토큰 ──
            encoder_in_tokens = [self.encoder_latent_input_proj(latent_sample)]
            encoder_in_pos_embed = list(self.encoder_1d_feature_pos_embed.weight.unsqueeze(1))
            if self.config.robot_state_feature:
                encoder_in_tokens.append(self.encoder_robot_state_input_proj(batch[OBS_STATE]))
            if self.config.env_state_feature:
                encoder_in_tokens.append(self.encoder_env_state_input_proj(batch[OBS_ENV_STATE]))

            if self.config.image_features:
                for img in batch[OBS_IMAGES]:  # 카메라마다
                    cam_features = self.backbone(img)["feature_map"]  # (B, 512, h, w)
                    cam_pos_embed = self.encoder_cam_feat_pos_embed(cam_features).to(dtype=cam_features.dtype)
                    cam_features = self.encoder_img_feat_input_proj(cam_features)  # (B, D, h, w)
                    # 픽셀 하나 = 토큰 하나: (h·w, B, D)
                    cam_features = einops.rearrange(cam_features, "b c h w -> (h w) b c")
                    cam_pos_embed = einops.rearrange(cam_pos_embed, "b c h w -> (h w) b c")
                    encoder_in_tokens.extend(list(cam_features))
                    encoder_in_pos_embed.extend(list(cam_pos_embed))

            encoder_in_tokens = torch.stack(encoder_in_tokens, axis=0)        # (ES, B, D)
            encoder_in_pos_embed = torch.stack(encoder_in_pos_embed, axis=0)  # (ES, 1, D)

            # ── 3단계: 인코더 → 디코더 → 행동 ──
            encoder_out = self.encoder(encoder_in_tokens, pos_embed=encoder_in_pos_embed)
            decoder_in = torch.zeros(
                (self.config.chunk_size, batch_size, self.config.dim_model),
                dtype=encoder_in_pos_embed.dtype,
                device=encoder_in_pos_embed.device,
            )  # 디코더 입력은 0. 자리 구분은 decoder_pos_embed가 함.
            decoder_out = self.decoder(
                decoder_in,
                encoder_out,
                encoder_pos_embed=encoder_in_pos_embed,
                decoder_pos_embed=self.decoder_pos_embed.weight.unsqueeze(1),
            )
            decoder_out = decoder_out.transpose(0, 1)  # (B, S, D)
            actions = self.action_head(decoder_out)    # (B, S, action_dim)
            return actions, (mu, log_sigma_x2)
    return (ACT,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 4-4. 텐서 모양 따라가기

    작은 배치 하나를 넣어 각 단계의 모양과 모듈별 매개변수 수를 확인합니다.
    """)
    return


@app.cell
def _(ACT, ACTConfig, OBS_IMAGES, OBS_IMAGE_TOP, device, torch, train_batch):
    _cfg = ACTConfig()
    _model = ACT(_cfg).to(device).train()
    _batch = train_batch(4, _cfg.chunk_size)
    _batch[OBS_IMAGES] = [_batch[OBS_IMAGE_TOP]]  # ACTPolicy.forward가 하는 일 (5절)

    with torch.no_grad():
        _feat = _model.backbone(_batch[OBS_IMAGE_TOP])["feature_map"]
        _actions, (_mu, _logvar) = _model(_batch)

    _groups = {}
    for _name, _p in _model.named_parameters():
        _key = _name.split(".")[0]
        _key = "vae_encoder*" if _key.startswith("vae_encoder") else _key
        _groups[_key] = _groups.get(_key, 0) + _p.numel()

    print("이미지 입력            ", tuple(_batch[OBS_IMAGE_TOP].shape))
    print("ResNet18 layer4 특징   ", tuple(_feat.shape), f"→ 이미지 토큰 {_feat.shape[2] * _feat.shape[3]}개")
    print("VAE 인코더 토큰 수     ", 1 + 1 + _cfg.chunk_size, "= CLS 1 + 상태 1 + 행동", _cfg.chunk_size)
    print("Transformer 인코더 토큰", 1 + 1 + _feat.shape[2] * _feat.shape[3], "= z 1 + 상태 1 + 이미지", _feat.shape[2] * _feat.shape[3])
    print("디코더 쿼리 수         ", _cfg.chunk_size)
    print("출력 행동              ", tuple(_actions.shape), " mu/logσ²", tuple(_mu.shape), tuple(_logvar.shape))
    print()
    for _k, _v in sorted(_groups.items(), key=lambda kv: -kv[1]):
        print(f"{_k:32s} {_v / 1e6:6.2f}M")
    print(f"{'합계':32s} {sum(_groups.values()) / 1e6:6.2f}M")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    매개변수의 대부분(약 11M)은 ResNet18 백본입니다. VAE 인코더는 학습에만 쓰이므로 추론 때는 그 부분이 필요 없습니다.

    ## 5. 정책 래퍼: `ACTPolicy`와 `ACTTemporalEnsembler`

    `ACTPolicy`는 `ACT` 신경망을 감싸서 **손실 계산**과 **환경에서 한 step씩 행동을 꺼내는 방법**을 제공합니다.

    **손실 (`forward`)**

    $$
    \mathcal{L} = \underbrace{\frac{1}{N_\text{valid}} \sum_{\text{패딩 아닌 }(t, j)} |a_{t,j} - \hat a_{t,j}|}_{\text{L1 재구성}}
    \; + \; \beta \cdot \underbrace{\frac{1}{B}\sum_b \sum_{d} -\tfrac12\left(1 + \log\sigma_d^2 - \mu_d^2 - \sigma_d^2\right)}_{D_\text{KL}(q(z\mid a, s)\,\|\,\mathcal N(0, I))}
    $$

    - L1을 쓰는 이유: 원 논문에서 L2보다 정밀한 동작에 더 좋았다고 보고했습니다.
    - KL 가중치 $\beta$ = `kl_weight` = 10. KL은 latent 차원에 대해 **합**, 배치에 대해 **평균**입니다.

    **행동 꺼내기 (`select_action`)** 는 두 가지 방식입니다.

    | 방식 | 설정 | 동작 |
    | --- | --- | --- |
    | 행동 큐 (기본) | `n_action_steps = k'` (≤ `chunk_size`) | 큐가 비면 청크를 예측해 앞 `k'`개를 큐에 넣고, 매 step 하나씩 꺼냄. **`k'` step마다 한 번만 모델 호출** |
    | 시간 앙상블 | `temporal_ensemble_coeff = 0.01`, `n_action_steps = 1` | **매 step** 청크를 예측하고, 같은 시각을 가리키는 과거 청크들의 행동을 가중 평균 |

    시간 앙상블의 가중치는 $w_i = \exp(-m \cdot i)$이고 $i=0$이 **가장 오래된** 예측입니다. $m=0.01$이면 오래된 예측에 약간 더 무게를 둡니다.
    `ACTTemporalEnsembler.update`는 과거 청크를 모두 저장하지 않고 **누적 가중 평균을 온라인으로** 갱신합니다.
    """)
    return


@app.cell
def _(ACT, ACTConfig, ACTION, F, OBS_IMAGES, Tensor, deque, nn, torch):
    class ACTTemporalEnsembler:
        """Temporal ensembling (ACT 논문 Algorithm 2). 가중치 w_i = exp(-coeff * i), w_0이 가장 오래된 행동."""

        def __init__(self, temporal_ensemble_coeff: float, chunk_size: int) -> None:
            self.chunk_size = chunk_size
            self.ensemble_weights = torch.exp(-temporal_ensemble_coeff * torch.arange(chunk_size))
            self.ensemble_weights_cumsum = torch.cumsum(self.ensemble_weights, dim=0)
            self.reset()

        def reset(self):
            self.ensembled_actions = None
            # (chunk_size, 1): 앞으로 각 시각에 대해 지금까지 몇 개의 예측이 평균에 들어갔는지
            self.ensembled_actions_count = None

        def update(self, actions: Tensor) -> Tensor:
            """(B, chunk_size, action_dim) 새 청크를 받아 앙상블을 갱신하고, 이번 step의 행동 (B, action_dim)을 꺼냅니다."""
            self.ensemble_weights = self.ensemble_weights.to(device=actions.device)
            self.ensemble_weights_cumsum = self.ensemble_weights_cumsum.to(device=actions.device)
            if self.ensembled_actions is None:
                self.ensembled_actions = actions.clone()
                self.ensembled_actions_count = torch.ones(
                    (self.chunk_size, 1), dtype=torch.long, device=self.ensembled_actions.device
                )
            else:
                # 이전 평균(길이 chunk_size-1)과 새 청크의 앞 chunk_size-1개를 온라인 가중 평균
                self.ensembled_actions *= self.ensemble_weights_cumsum[self.ensembled_actions_count - 1]
                self.ensembled_actions += actions[:, :-1] * self.ensemble_weights[self.ensembled_actions_count]
                self.ensembled_actions /= self.ensemble_weights_cumsum[self.ensembled_actions_count]
                self.ensembled_actions_count = torch.clamp(self.ensembled_actions_count + 1, max=self.chunk_size)
                # 새 청크의 마지막 행동은 아직 겹치는 예측이 없으므로 그대로 뒤에 붙임
                self.ensembled_actions = torch.cat([self.ensembled_actions, actions[:, -1:]], dim=1)
                self.ensembled_actions_count = torch.cat(
                    [self.ensembled_actions_count, torch.ones_like(self.ensembled_actions_count[-1:])]
                )
            # 맨 앞(이번 step) 행동을 꺼냄
            action, self.ensembled_actions, self.ensembled_actions_count = (
                self.ensembled_actions[:, 0],
                self.ensembled_actions[:, 1:],
                self.ensembled_actions_count[1:],
            )
            return action


    class ACTPolicy(nn.Module):  # 원본은 lerobot의 PreTrainedPolicy를 상속 (저장·불러오기·Hub 업로드 기능)
        def __init__(self, config: ACTConfig):
            super().__init__()
            config.validate_features()
            self.config = config
            self.model = ACT(config)
            if config.temporal_ensemble_coeff is not None:
                self.temporal_ensembler = ACTTemporalEnsembler(config.temporal_ensemble_coeff, config.chunk_size)
            self.reset()

        def get_optim_params(self):
            # 백본과 나머지를 다른 학습률로 (기본값은 둘이 같음)
            return [
                {"params": [p for n, p in self.named_parameters() if not n.startswith("model.backbone") and p.requires_grad]},
                {
                    "params": [p for n, p in self.named_parameters() if n.startswith("model.backbone") and p.requires_grad],
                    "lr": self.config.optimizer_lr_backbone,
                },
            ]

        def reset(self):
            """환경을 reset할 때마다 호출합니다."""
            if self.config.temporal_ensemble_coeff is not None:
                self.temporal_ensembler.reset()
            else:
                self._action_queue = deque([], maxlen=self.config.n_action_steps)

        @torch.no_grad()
        def select_action(self, batch: dict[str, Tensor]) -> Tensor:
            """환경에서 한 step에 쓸 행동 (B, action_dim)을 돌려줍니다."""
            self.eval()
            if self.config.temporal_ensemble_coeff is not None:
                actions = self.predict_action_chunk(batch)
                action = self.temporal_ensembler.update(actions)
                return action

            # 큐가 비었을 때만 모델을 호출하고, 앞 n_action_steps개를 큐에 넣음
            if len(self._action_queue) == 0:
                actions = self.predict_action_chunk(batch)[:, : self.config.n_action_steps]
                # (B, n, A) → 큐에는 (n개의 (B, A))로 들어가도록 transpose
                self._action_queue.extend(actions.transpose(0, 1))
            return self._action_queue.popleft()

        @torch.no_grad()
        def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
            self.eval()
            if self.config.image_features:
                batch = dict(batch)
                batch[OBS_IMAGES] = [batch[key] for key in self.config.image_features]  # 카메라 목록으로 모음
            actions = self.model(batch)[0]
            return actions

        def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
            """학습용: 손실과 기록용 dict를 돌려줍니다."""
            if self.config.image_features:
                batch = dict(batch)
                batch[OBS_IMAGES] = [batch[key] for key in self.config.image_features]

            actions_hat, (mu_hat, log_sigma_x2_hat) = self.model(batch)

            # 패딩이 아닌 위치만 평균한 L1
            abs_err = F.l1_loss(batch[ACTION], actions_hat, reduction="none")
            valid_mask = ~batch["action_is_pad"].unsqueeze(-1)
            num_valid = valid_mask.sum() * abs_err.shape[-1]
            l1_loss = (abs_err * valid_mask).sum() / num_valid.clamp_min(1)

            loss_dict = {"l1_loss": l1_loss.item()}
            if self.config.use_vae and log_sigma_x2_hat is not None:
                # KL(N(mu, σ²) || N(0, I)): latent 차원에 대해 합, 배치에 대해 평균
                mean_kld = (
                    (-0.5 * (1 + log_sigma_x2_hat - mu_hat.pow(2) - (log_sigma_x2_hat).exp())).sum(-1).mean()
                )
                loss_dict["kld_loss"] = mean_kld.item()
                loss = l1_loss + mean_kld * self.config.kl_weight
            else:
                loss = l1_loss
            return loss, loss_dict
    return ACTPolicy, ACTTemporalEnsembler


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    시간 앙상블이 원본 docstring의 설명대로 **가중 평균과 같은 값**을 내는지 직접 확인합니다.
    매 step 청크 하나씩 넣었을 때, 시각 `t`에 꺼낸 행동은 시각 `t`를 가리키는 모든 과거 예측의 가중 평균이어야 합니다.
    """)
    return


@app.cell
def _(ACTTemporalEnsembler, torch):
    _k, _m = 5, 0.5  # 차이가 잘 보이도록 큰 계수를 씁니다.
    _ens = ACTTemporalEnsembler(_m, _k)
    _gen = torch.Generator().manual_seed(0)
    _chunks = [torch.randn(1, _k, 1, generator=_gen) for _ in range(8)]  # chunk s는 시각 s..s+k-1을 예측

    for _t, _chunk in enumerate(_chunks):
        _out = _ens.update(_chunk).item()
        # 시각 _t를 예측한 청크들: s = max(0, t-k+1) .. t (오래된 순서), s의 _t-s번째 원소
        _preds = torch.tensor([_chunks[_s][0, _t - _s, 0].item() for _s in range(max(0, _t - _k + 1), _t + 1)])
        _w = torch.exp(-_m * torch.arange(len(_preds)))
        _ref = ((_w * _preds).sum() / _w.sum()).item()
        print(f"t={_t}: 겹친 예측 {len(_preds)}개, ensembler {_out:+.4f}, 직접 계산한 가중 평균 {_ref:+.4f}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    처음 몇 step은 겹친 예측이 1, 2, 3...개로 늘고, `k=5`개가 쌓인 뒤로는 항상 5개를 평균합니다. 두 값이 일치합니다.

    ## 6. 학습

    세 모델을 같은 데이터, 같은 step 수로 학습합니다.

    | 이름 | `chunk_size` | `use_vae` | `kl_weight` | 의미 |
    | --- | --- | --- | --- | --- |
    | **A. 단일 step BC** | 1 | False | – | 행동 하나만 예측하는 일반 행동 복제(behavior cloning). ACT 구조는 같고 청크만 1 |
    | **B. ACT (기본)** | 20 | True | 10 | LeRobot 기본 손실 설정 |
    | **C. ACT, 작은 KL** | 20 | True | 0.01 | 8절 잠재 변수 실험용 |

    ### 6-1. 학습 루프

    LeRobot의 `lerobot-train`이 하는 일을 줄인 것입니다. 옵티마이저는 `AdamW(get_optim_params(), lr, weight_decay)`,
    gradient clipping은 LeRobot 기본값 `grad_clip_norm=10`입니다. 배치 크기 64, 학습률 상수(스케줄러 없음, `get_scheduler_preset()`이 `None`).
    CPU에서 실행한다면 `TRAIN_STEPS`를 줄이세요(성공률도 낮아집니다).
    """)
    return


@app.cell
def _(ACTPolicy, device, time, torch, train_batch):
    TRAIN_STEPS = 3000
    BATCH_SIZE = 64


    def train_policy(cfg, steps=TRAIN_STEPS, seed=0, log_every=50):
        torch.manual_seed(seed)
        gen = torch.Generator(device=device).manual_seed(seed)
        policy = ACTPolicy(cfg).to(device)
        optimizer = torch.optim.AdamW(policy.get_optim_params(), lr=cfg.optimizer_lr, weight_decay=cfg.optimizer_weight_decay)
        history, t0 = [], time.time()
        policy.train()
        for step in range(steps):
            batch = train_batch(BATCH_SIZE, cfg.chunk_size, generator=gen)
            loss, info = policy(batch)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 10.0)
            optimizer.step()
            if step % log_every == 0 or step == steps - 1:
                history.append({"step": step, **info})
        mem = f", 최대 GPU 메모리 {torch.cuda.max_memory_allocated() / 2**30:.2f}GB" if device == "cuda" else ""
        print(f"chunk={cfg.chunk_size}, use_vae={cfg.use_vae}, kl_weight={cfg.kl_weight}: "
              f"{steps} steps, {time.time() - t0:.0f}s{mem}, 마지막 {history[-1]}")
        return policy.eval(), history
    return (train_policy,)


@app.cell
def _(ACTConfig, train_policy):
    policy_bc, hist_bc = train_policy(ACTConfig(chunk_size=1, n_action_steps=1, use_vae=False))
    return hist_bc, policy_bc


@app.cell
def _(ACTConfig, train_policy):
    policy_act, hist_act = train_policy(ACTConfig(chunk_size=20, n_action_steps=20, kl_weight=10.0))
    return hist_act, policy_act


@app.cell
def _(ACTConfig, train_policy):
    policy_act_lowkl, hist_act_lowkl = train_policy(ACTConfig(chunk_size=20, n_action_steps=20, kl_weight=0.01))
    return hist_act_lowkl, policy_act_lowkl


@app.cell
def _(hist_act, hist_act_lowkl, hist_bc, np, plt):
    def smooth(values, w=5):
        return np.convolve(values, np.ones(w) / w, mode="valid")


    _fig, _axes = plt.subplots(1, 2, figsize=(12, 3.6))
    for _hist, _label, _color in [(hist_bc, "A. BC (chunk 1)", "C0"), (hist_act, "B. ACT kl=10", "C1"),
                                  (hist_act_lowkl, "C. ACT kl=0.01", "C2")]:
        _steps = [h["step"] for h in _hist]
        _axes[0].plot(_steps[4:], smooth([h["l1_loss"] for h in _hist]), label=_label, color=_color)
        if "kld_loss" in _hist[0]:
            _axes[1].plot(_steps[4:], smooth([h["kld_loss"] for h in _hist]), label=_label, color=_color)
    _axes[0].set(title="L1 loss (normalized actions)", xlabel="step", yscale="log")
    _axes[1].set(title="KL divergence (nats, summed over 32 dims)", xlabel="step", yscale="log")
    for _ax in _axes:
        _ax.legend()
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    손실 그래프를 읽을 때 주의할 점:

    - **A의 L1이 가장 낮다고 A가 가장 좋은 것은 아닙니다.** A는 바로 다음 행동 하나만 맞히면 되고, 그 행동은 현재 위치와 거의 같아 쉽습니다.
      B와 C는 20 step 뒤까지 맞혀야 하며, 특히 "언제 출발할지"는 관측만으로는 알 수 없어 오차가 남습니다.
    - **B(kl=10)의 KL은 학습 초반에 거의 0으로 떨어집니다.** VAE 인코더가 사실상 아무 정보도 `z`에 담지 않는다는 뜻입니다(posterior collapse).
      C(kl=0.01)는 KL이 1~2 nats 정도로 유지되고, 그만큼 L1이 B보다 낮습니다. 학습 중에는 디코더가 VAE 인코더를 통해
      **정답 행동 묶음을 엿본 `z`** 를 받기 때문입니다. 8절에서 이 `z`에 무엇이 담겼는지 봅니다.

    ## 7. 환경에서 평가 (폐루프 rollout)

    학습 손실은 "시연 데이터의 다음 행동을 얼마나 맞히는가"이고, 실제로 중요한 것은 **정책이 직접 움직였을 때 과제를 해내는가**입니다.
    학습에 쓰지 않은 무작위 장면 100개(시드 `123`)에서 최대 100 step 실행합니다.
    100개 환경을 하나의 배치로 묶어 `select_action`을 호출합니다. 모든 환경이 같은 step에 큐를 채우므로 행동 큐도 배치로 동작합니다.

    | 지표 | 의미 |
    | --- | --- |
    | 성공률 | 물체를 든 채 목표 0.05 이내에서 끝남 |
    | 잡기 성공률 | 끝날 때 물체를 들고 있음 |
    | 멈춤 비율 | 마지막 30 step 동안 0.01도 움직이지 않았고, 성공도 아님 |
    """)
    return


@app.cell
def _(
    ACTTemporalEnsembler,
    GOAL_TOL,
    MAX_T,
    device,
    env_step,
    np,
    obs_batch,
    random,
    replace,
    sample_scene,
    torch,
    unnormalize_action,
):
    def set_inference(policy, temporal_ensemble_coeff=None, n_action_steps=None):
        """추론 방식만 바꿉니다: 행동 큐(n_action_steps) 또는 시간 앙상블(coeff, n_action_steps=1)."""
        cfg = policy.config
        if temporal_ensemble_coeff is not None:
            cfg = replace(cfg, temporal_ensemble_coeff=temporal_ensemble_coeff, n_action_steps=1)
            policy.temporal_ensembler = ACTTemporalEnsembler(temporal_ensemble_coeff, cfg.chunk_size)
        else:
            cfg = replace(cfg, temporal_ensemble_coeff=None, n_action_steps=n_action_steps or cfg.chunk_size)
        policy.config = cfg
        policy.reset()
        return policy


    def rollout(policy, n=100, seed=123, max_t=MAX_T):
        rng = random.Random(seed)
        scenes = [sample_scene(rng) for _ in range(n)]
        pos = np.array([s[0] for s in scenes])
        obj = np.array([s[1] for s in scenes])
        goal = np.array([s[2] for s in scenes])
        stay, carry = np.zeros(n, int), np.zeros(n, bool)
        to_t = lambda x, dt=torch.float32: torch.tensor(x, dtype=dt, device=device)

        policy.reset()
        traj, n_calls = [pos.copy()], 0
        for _ in range(max_t):
            queue_empty = policy.config.temporal_ensemble_coeff is not None or len(policy._action_queue) == 0
            n_calls += int(queue_empty)
            batch = obs_batch(to_t(pos), to_t(obj), to_t(goal), to_t(carry, torch.bool))
            action = unnormalize_action(policy.select_action(batch)).cpu().numpy()
            pos, stay, carry = env_step(pos, action, obj, stay, carry)
            traj.append(pos.copy())
        traj = np.array(traj)  # (max_t+1, n, 2)
        success = carry & (np.linalg.norm(pos - goal, axis=1) < GOAL_TOL)
        stuck = (np.linalg.norm(traj[-1] - traj[-31], axis=1) < 0.01) & ~success
        return dict(success=success.mean(), grasp=carry.mean(), stuck=stuck.mean(), model_calls=n_calls,
                    traj=traj, obj=obj, goal=goal, success_mask=success)
    return rollout, set_inference


@app.cell
def _(mo, policy_act, policy_act_lowkl, policy_bc, rollout, set_inference):
    eval_results = {
        "A. BC (chunk 1)": rollout(set_inference(policy_bc)),
        "B. ACT, queue (n_action_steps=20)": rollout(set_inference(policy_act, n_action_steps=20)),
        "B. ACT, queue (n_action_steps=10)": rollout(set_inference(policy_act, n_action_steps=10)),
        "B. ACT, queue (n_action_steps=1)": rollout(set_inference(policy_act, n_action_steps=1)),
        "B. ACT, temporal ensemble (m=0.01)": rollout(set_inference(policy_act, temporal_ensemble_coeff=0.01)),
        "C. ACT kl=0.01, queue (n_action_steps=20)": rollout(set_inference(policy_act_lowkl, n_action_steps=20)),
    }
    set_inference(policy_act, n_action_steps=20)  # 이후 셀을 위해 기본 방식으로 되돌림

    mo.md(
        "| 정책 · 추론 방식 | 성공률 | 잡기 성공률 | 멈춤 비율 | 모델 호출 수 (100 step) |\n| --- | --- | --- | --- | --- |\n"
        + "\n".join(
            f"| {k} | **{r['success']:.0%}** | {r['grasp']:.0%} | {r['stuck']:.0%} | {r['model_calls']} |"
            for k, r in eval_results.items()
        )
    )
    return (eval_results,)


@app.cell
def _(device, eval_results, np, plt, render, torch):
    def plot_rollouts(results_by_name, idx):
        fig, axes = plt.subplots(len(results_by_name), len(idx), figsize=(2.6 * len(idx), 2.7 * len(results_by_name)))
        for row, (name, res) in enumerate(results_by_name.items()):
            for col, i in enumerate(idx):
                ax = axes[row, col]
                start = torch.tensor(res["traj"][0, i:i + 1], dtype=torch.float32, device=device)
                img = render(start, torch.tensor(res["obj"][i:i + 1], dtype=torch.float32, device=device),
                             torch.tensor(res["goal"][i:i + 1], dtype=torch.float32, device=device),
                             torch.zeros(1, dtype=torch.bool, device=device))[0]
                ax.imshow(img.permute(1, 2, 0).cpu().numpy(), extent=[0, 1, 1, 0])
                path = res["traj"][:, i]
                ax.plot(path[:, 0], path[:, 1], "w.-", ms=2, lw=0.7)
                ax.plot(*path[-1], "yx", ms=8, mew=2)
                ax.set_xticks([]); ax.set_yticks([])
                ax.set_title(("OK" if res["success_mask"][i] else "fail") + f" (scene {i})", fontsize=9)
            axes[row, 0].set_ylabel(name.split(",")[0], fontsize=9)
        fig.tight_layout()
        return fig


    _names = ["A. BC (chunk 1)", "B. ACT, queue (n_action_steps=20)"]
    plot_rollouts({k: eval_results[k] for k in _names}, idx=list(range(6)))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    흰 선은 실행 경로, 노란 ×는 마지막 위치입니다. 단일 step BC(A)의 실패를 아래 셀에서 마지막 상태별로 나눠 봅니다.
    """)
    return


@app.cell
def _(eval_results, np):
    for _name in ["A. BC (chunk 1)", "B. ACT, queue (n_action_steps=20)"]:
        _r = eval_results[_name]
        _fail = ~_r["success_mask"]
        _end_at_obj = np.linalg.norm(_r["traj"][-1] - _r["obj"], axis=1) < 0.05  # 마지막 위치가 물체(원래 자리) 근처
        print(f"{_name}: 실패 {_fail.mean():.0%} = 물체 위에서 멈춤 {(_fail & _end_at_obj).mean():.0%} "
              f"+ 물체에서 떨어진 곳에서 끝남 {(_fail & ~_end_at_obj).mean():.0%}  (잡기 성공률 {_r['grasp']:.0%})")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    단일 step BC(A)는 1절의 예상처럼 **물체 위에서 영원히 멈추는 경우도 있지만**, 더 많은 경우 **물체를 스치듯 지나가며 멈추지 않고,
    빈손으로 목표 쪽으로 가다가 멈춥니다**(잡기 성공률이 낮은 이유). 두 실패의 원인은 같습니다. **같아 보이는 관측에 서로 다른 정답 행동이 섞여 있습니다.**

    - 물체 위: "기다려"와 "출발해"가 섞임 → 1절에서 본 대기 문제
    - 물체 근처: 물체로 **다가가는** 프레임과 물체를 들고 **떠나는** 프레임은 이미지가 거의 같습니다(잡은 물체는 로봇 위치에 그려지므로).
      "물체 쪽으로"와 "목표 쪽으로"가 섞여, 정책은 물체에 닿기 전부터 목표 쪽으로 꺾습니다. 5 step을 머물지 않으니 물체가 잡히지 않습니다.
      일단 시연에 없던 경로(빈손으로 목표 근처)에 들어서면 그다음 관측도 처음 보는 것이 되어 오차가 쌓입니다(compounding error).

    ACT(B)는 물체로 가는 중간에 "물체까지 가서 몇 step 기다렸다 출발"하는 20개 행동을 한 번에 내고 그대로 실행하므로
    매 step 섞인 정답 사이에서 흔들리지 않습니다. 같은 B 모델이라도 `n_action_steps=1`로 **매 step 다시 묻고 첫 행동만 쓰면**
    표에서 보듯 성능이 달라집니다. 청크를 예측하는 것뿐 아니라 **청크를 실행하는 것**이 중요합니다.

    아래 그래프는 한 장면에서 시간에 따른 x좌표입니다. 행동 큐 방식은 청크가 바뀌는 지점(20 step마다)에서 경로가 꺾일 수 있고,
    시간 앙상블은 매 step 여러 예측을 평균해 더 부드럽지만 **모델을 100번 호출합니다**(큐 방식 `n_action_steps=20`은 5번).
    """)
    return


@app.cell
def _(eval_results, np, plt):
    _scene = 0
    _fig, _axes = plt.subplots(1, 2, figsize=(12, 3.4))
    for _name in ["A. BC (chunk 1)", "B. ACT, queue (n_action_steps=20)", "B. ACT, temporal ensemble (m=0.01)"]:
        _path = eval_results[_name]["traj"][:, _scene]
        _axes[0].plot(_path[:, 0], label=_name)
        _axes[1].plot(np.linalg.norm(np.diff(_path, axis=0), axis=1), label=_name)
    for _t in range(0, 101, 20):
        _axes[0].axvline(_t, color="gray", lw=0.5, ls=":")
        _axes[1].axvline(_t, color="gray", lw=0.5, ls=":")
    _axes[0].set(title=f"scene {_scene}: x position", xlabel="step")
    _axes[1].set(title="speed |p_t+1 - p_t| (max 0.06)", xlabel="step")
    _axes[0].legend(fontsize=8)
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 8. CVAE의 잠재 변수 `z`에는 무엇이 담기나

    ACT는 학습 때 VAE 인코더가 **정답 행동 묶음**을 보고 `z`를 만들고, 디코더는 그 `z`와 관측으로 행동을 복원합니다.
    이 과제에서 관측으로 알 수 없는 정보는 **대기 시간**(5~20 step 무작위)입니다. 그렇다면 `z`가 대기 시간을 담을까요?

    시연마다 **물체에 처음 도착한 프레임**(`arrive_t`)을 골라 VAE 인코더의 평균 `mu`(32차원)를 구하고,
    대기 시간과 상관이 가장 큰 차원을 찾습니다. VAE 인코더는 `self.training`일 때만 실행되므로,
    dropout을 끈 채(`policy.eval()`) `ACT` 모듈의 `training` 플래그만 켭니다.
    """)
    return


@app.cell
def _(ACTION, OBS_IMAGES, OBS_IMAGE_TOP, device, episodes, frames, np, obs_batch, stats, torch):
    _starts = np.cumsum([0] + [len(e["state"]) for e in episodes])[:-1]
    arrive_ix = torch.tensor([_s + e["arrive_t"] for _s, e in zip(_starts, episodes)], device=device)
    waits = np.array([e["wait"] for e in episodes])


    def arrival_batch(ix):
        batch = obs_batch(frames["state"][ix], frames["obj"][ix], frames["goal"][ix], frames["carrying"][ix])
        batch[ACTION] = (frames["action"][ix] - stats["action"][0]) / stats["action"][1]
        batch["action_is_pad"] = frames["action_is_pad"][ix]
        batch[OBS_IMAGES] = [batch[OBS_IMAGE_TOP]]
        return batch


    @torch.no_grad()
    def posterior_mu(policy, batch):
        policy.eval()               # dropout 끔
        policy.model.training = True  # ACT.forward가 VAE 인코더 경로를 타도록 플래그만 켬
        try:
            _, (mu, log_sigma_x2) = policy.model(batch)
        finally:
            policy.model.training = False
        return mu.cpu().numpy(), (log_sigma_x2 / 2).exp().cpu().numpy()


    def latent_report(policy):
        mu, sigma = posterior_mu(policy, arrival_batch(arrive_ix))
        corr = np.array([np.corrcoef(mu[:, j], waits)[0, 1] for j in range(mu.shape[1])])
        best = int(np.nanargmax(np.abs(corr)))
        return dict(mu=mu, sigma=sigma, corr=corr, best=best)
    return arrival_batch, arrive_ix, latent_report, waits


@app.cell
def _(latent_report, np, plt, policy_act, policy_act_lowkl, waits):
    latent_b = latent_report(policy_act)
    latent_c = latent_report(policy_act_lowkl)

    _fig, _axes = plt.subplots(1, 2, figsize=(11, 3.6))
    for _ax, _rep, _title in [(_axes[0], latent_b, "B. kl_weight=10"), (_axes[1], latent_c, "C. kl_weight=0.01")]:
        _j = _rep["best"]
        _ax.scatter(waits, _rep["mu"][:, _j], s=10)
        _ax.set(title=f"{_title}: mu[{_j}] vs wait (corr {_rep['corr'][_j]:+.2f})", xlabel="expert wait steps", ylabel=f"mu[{_j}]")
    _fig.tight_layout()
    for _name, _rep in [("B", latent_b), ("C", latent_c)]:
        print(f"{_name}: |mu| 평균 {np.abs(_rep['mu']).mean():.4f}, σ 평균 {_rep['sigma'].mean():.3f}, "
              f"가장 상관 큰 차원 {_rep['best']} (상관 {_rep['corr'][_rep['best']]:+.3f})")
    _fig
    return latent_b, latent_c


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    - **B (kl_weight=10)**: 상관은 보이지만 **세로축 눈금을 보세요.** `mu`의 크기가 0.001 수준이고 `σ`는 거의 1입니다.
      `z = mu + σ·ε`에서 신호(0.001)가 노이즈(1)에 완전히 묻히므로 디코더는 `z`에서 사실상 아무 정보도 얻지 못합니다.
      사후 분포가 사전 분포 N(0, I)와 같아진 상태(posterior collapse)입니다. KL 벌점이 커서, 대기 시간을 `z`에 넣어 얻는 L1 이득보다
      KL 비용이 더 큽니다(대기 시작 근처 프레임은 전체 프레임의 일부일 뿐입니다).
      그래도 7절에서 B가 잘 동작한 것은 **성능의 핵심이 청크 예측**이기 때문입니다. 원 논문의 ablation에서도 CVAE는 사람 시연 데이터에서
      효과가 컸고, 스크립트 시연에서는 차이가 작았습니다.
    - **C (kl_weight=0.01)**: 한 차원이 대기 시간과 강하게 상관합니다. VAE 인코더가 "이번 시연은 몇 step 기다렸다"를 `z`에 적어 디코더에 전달합니다.

    그렇다면 디코더는 `z`에 따라 **다른 대기 시간**의 행동을 낼까요? 추론 때 `ACT.forward`는 `z = 0`을 쓰지만,
    `z`가 들어가는 곳은 `encoder_latent_input_proj` 하나뿐이므로 이 층에 forward hook을 걸어 원하는 `z`로 바꿔 넣을 수 있습니다.
    C 모델에서 상관이 가장 큰 차원 하나만 바꿔 가며, 같은 관측(물체에 막 도착한 프레임)에 대한 행동 묶음을 그립니다.
    """)
    return


@app.cell
def _(F, OBS_IMAGE_TOP, OBS_STATE, arrival_batch, arrive_ix, episodes, latent_c, np, plt, policy_act_lowkl, torch, unnormalize_action):
    def decode_with_latent(policy, batch, z):
        """encoder_latent_input_proj의 출력을 proj(z)로 바꿔 행동 묶음을 예측합니다."""
        proj = policy.model.encoder_latent_input_proj
        handle = proj.register_forward_hook(lambda m, inp, out: F.linear(z, m.weight, m.bias))
        try:
            return policy.predict_action_chunk(batch)
        finally:
            handle.remove()


    _ep_i = 3
    _b = arrival_batch(arrive_ix[_ep_i:_ep_i + 1])
    _obs = {OBS_IMAGE_TOP: _b[OBS_IMAGE_TOP], OBS_STATE: _b[OBS_STATE]}
    _obj = episodes[_ep_i]["obj"]
    _j, _mu = latent_c["best"], latent_c["mu"][:, latent_c["best"]]

    _fig, _ax = plt.subplots(figsize=(8, 3.6))
    for _q in [0.05, 0.25, 0.5, 0.75, 0.95]:
        _z = torch.zeros(1, policy_act_lowkl.config.latent_dim, device=_b[OBS_STATE].device)
        _z[0, _j] = float(np.quantile(_mu, _q))
        _chunk = unnormalize_action(decode_with_latent(policy_act_lowkl, _obs, _z))[0].cpu().numpy()
        _dist = np.linalg.norm(_chunk - _obj, axis=1)
        _leave = int(np.argmax(_dist > 0.05)) if (_dist > 0.05).any() else None  # 물체에서 0.05 넘게 멀어지는 첫 위치
        _ax.plot(_dist, marker=".", label=f"z[{_j}] = {_z[0, _j].item():+.2f} ({_q:.0%} quantile), leaves at {_leave}")
    _chunk0 = unnormalize_action(policy_act_lowkl.predict_action_chunk(_obs))[0].cpu().numpy()
    _ax.plot(np.linalg.norm(_chunk0 - _obj, axis=1), "k--", label="z = 0 (default inference)")
    _ax.set(title=f"demo {_ep_i} at arrival: predicted distance from object over the chunk",
            xlabel="index in action chunk", ylabel="|a - object|")
    _ax.legend(fontsize=8)
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    `z`의 한 차원만 바꿨는데 **출발 시점(그래프가 바닥에서 올라가는 곳)** 이 달라집니다. 값이 클수록 늦게 출발하고
    (검증 실행: 5% 분위수 7번째 → 95% 분위수 19번째), 산점도에서 본 대기 시간과 같은 방향입니다. 디코더가 `z`를 "얼마나 기다릴지"로 쓰고 있다는 뜻입니다.

    추론 때 쓰는 `z = 0`은 사전 분포 N(0, I)의 평균입니다. 그런데 위 산점도에서 C의 `mu`는 0을 중심으로 퍼져 있지 않습니다.
    `kl_weight`가 작아 사후 분포를 사전 분포 쪽으로 끌어당기는 힘이 약하기 때문입니다. 그래서 여기서 `z = 0`은 "평균적인 시연자"가 아니라
    **짧게 기다리는 시연자** 쪽에 해당합니다. KL 항은 `z`를 비우는 대가로, 추론 때 `z = 0`이 의미 있는 값이 되도록 보장하는 역할도 합니다.
    `kl_weight`는 이 둘 사이의 균형입니다.
    이것이 ACT 논문이 말하는 CVAE의 역할입니다. 사람 시연의 **제각각인 부분을 `z`로 빼내** 디코더가 여러 시연을 평균한 흐릿한 행동을 배우지 않게 하고,
    추론 때는 `z = 0`으로 하나의 일관된 행동을 냅니다.

    ## 9. SmolVLA와 비교

    [SmolVLA](https://github.com/huggingface/lerobot/tree/main/src/lerobot/policies/smolvla)도 **행동 묶음을 한 번에 예측하고 행동 큐로 실행**합니다.
    `SmolVLAPolicy.select_action`은 `ACTPolicy`처럼 큐가 비었을 때만 `predict_action_chunk`를 호출하고
    `actions.transpose(0, 1)[: n_action_steps]`를 큐에 넣습니다. 다른 점은 **무엇을 보고**, **어떻게 묶음을 만드는가**입니다.

    | | ACT (`policies/act`) | SmolVLA (`policies/smolvla`) |
    | --- | --- | --- |
    | 입력 | 이미지 + 상태 | 이미지 + **언어 지시문** + 상태 |
    | 관측 인코더 | ResNet18 + Transformer 인코더 (처음부터 학습) | 사전 학습 VLM `SmolVLM2-500M`의 **앞 16층** (`num_vlm_layers=16`), 비전 인코더 고정 |
    | 행동 생성기 | Transformer 디코더 1층, 쿼리 `k`개 → **한 번에** 출력 | VLM 옆의 작은 **action expert** Transformer (폭 0.75배), VLM KV를 cross-attention으로 읽음 |
    | 다양한 시연 처리 | CVAE 잠재 변수 `z` (추론 때 0) | **Flow matching**: 노이즈에서 출발해 `num_steps=10`번 Euler 적분 |
    | 손실 | L1 + `kl_weight`·KL | 속도장 MSE: `F.mse_loss(u_t, v_t)` |
    | 청크 | `chunk_size=100`, `n_action_steps=100` | `chunk_size=50`, `n_action_steps=50` |
    | 상태·행동 차원 | 데이터셋 그대로 | `max_state_dim=max_action_dim=32`로 패딩 (여러 로봇 공용) |
    | 이미지 정규화 | `MEAN_STD` | `IDENTITY` (VLM 전처리 사용) |

    SmolVLA 학습의 핵심(`VLAFlowMatching.forward`)은 다음과 같습니다.
    `make_flow_matching_inputs(..., convention=NOISE_AT_ONE)`은 `t=1`이 노이즈, `t=0`이 정답 행동인 직선 경로를 만듭니다.

    ```python
    x_t, u_t, _ = make_flow_matching_inputs(actions, noise, time, convention=FlowConvention.NOISE_AT_ONE)
    # x_t = t·noise + (1-t)·actions,  u_t = noise - actions  (목표 속도)
    prefix_embs, ... = self.embed_prefix(images, img_masks, lang_tokens, lang_masks, state=state)  # VLM 쪽
    suffix_embs, ... = self.embed_suffix(x_t, time)                                               # action expert 쪽
    ...
    v_t = self.action_out_proj(suffix_out)
    losses = F.mse_loss(u_t, v_t, reduction="none")
    ```

    ACT에서 배운 개념이 그대로 이어집니다.

    - **action chunk와 `action_is_pad`**: SmolVLA도 `action_delta_indices = list(range(chunk_size))`로 미래 행동 묶음을 받고, 패딩 위치를 손실에서 뺍니다.
    - **행동 큐와 `n_action_steps`**: 같은 방식입니다. 여기에 SmolVLA는 다음 청크를 미리 계산하는 비동기 추론과 RTC(Real-Time Chunking) 옵션이 있습니다.
    - **다양한 시연을 다루는 장치**: ACT는 `z`를 학습 때만 쓰고 추론 때 0으로 고정해 **하나의 행동**을 냅니다. Flow matching은 노이즈를 바꿔
      **여러 그럴듯한 행동 묶음 중 하나를 샘플링**합니다. 8절에서 `z`를 바꿔 출발 시점을 바꾼 것이 flow matching의 노이즈 샘플링과 비슷한 역할입니다.

    ## 10. 정리

    | 개념 | 핵심 |
    | --- | --- |
    | Action chunking | 관측 하나 → 행동 `k`개. 과거를 몰라도 되는 구간이 길어지고, 모델 호출 수가 `1/n_action_steps`로 줄어듦 |
    | 단일 step BC의 실패 | 같은 관측에 다른 행동이 섞이면(대기) L1 최적해가 "제자리"가 되어 멈춤 |
    | 구조 | ResNet 특징 지도 픽셀 + 상태 + `z` → Transformer 인코더 → 쿼리 `k`개 디코더 → Linear |
    | 위치 정보 | q, k에만 더함. 이미지는 2D sin/cos, 디코더 쿼리는 학습 임베딩 |
    | 손실 | 패딩 제외 L1 + `kl_weight`·KL |
    | CVAE | 학습 때 정답 묶음에서 `z`를 뽑고 추론 때 `z=0`. KL 가중치가 크면 `z`가 붕괴할 수 있음 |
    | 추론 | 행동 큐(빠름) 또는 시간 앙상블(부드러움, 매 step 호출) |
    | SmolVLA | 같은 청크·큐 구조 + VLM 관측 인코더 + flow matching 행동 생성 |

    ## 11. 직접 해 볼 과제

    각 과제를 하기 전에 **결과를 먼저 예측**해 적고, 바꿔 실행한 뒤 비교하세요.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({
        "과제 1. 청크 크기 바꾸기 (6절)": mo.md("""
    `chunk_size`를 5, 10, 40으로 바꿔 학습하고 7절 성공률을 비교하세요. `n_action_steps`도 같게 둡니다.

    *힌트:* 대기 시간은 최대 20 step입니다. 청크가 대기 시간보다 짧으면 물체 위에서 다시 묻게 되고, 그때 관측은 도착 직후와 같습니다.
    `MAX_CHUNK`(2절)보다 큰 청크를 쓰려면 `MAX_CHUNK`도 늘려야 합니다.
    """),
        "과제 2. n_action_steps 훑어보기 (7절)": mo.md("""
    B 모델을 `n_action_steps` = 1, 2, 5, 10, 15, 20으로 평가해 성공률을 그래프로 그리세요.

    *힌트:* `n_action_steps=1`은 청크 20개를 예측하지만 매 step 첫 행동만 쓰고 다시 묻습니다. 물체 위에서는 같은 관측에 대해
    매번 "몇 step 기다렸다 출발"하는 청크를 받고, 그 첫 행동은 "제자리"입니다. 시간 앙상블(`n_action_steps=1`이지만 과거 예측을 함께 평균)과 비교해 보세요.
    """),
        "과제 3. 시간 앙상블 계수 (7절)": mo.md("""
    `temporal_ensemble_coeff`를 `0`, `0.1`, `-0.1`로 바꿔 보세요. 음수는 **최근 예측에 더 큰 가중치**를 줍니다.

    *힌트:* 원본 docstring에 따르면 최근 예측에 큰 가중치를 주면 청크의 장점이 줄어든다는 실험이 있습니다(lerobot PR #319).
    과제 2와 같은 이유로, 최근 예측만 믿으면 물체 위에서 출발이 계속 미뤄질 수 있습니다.
    """),
        "과제 4. 상태 입력을 빼기 (3절)": mo.md("""
    `ACTConfig(state_dim=0, ...)`으로 상태 없이 이미지만 쓰게 바꾸려면 어디를 고쳐야 할까요?
    `ACT.forward`의 `else` 분기는 `batch[OBS_STATE].device`를 쓰고, `obs_batch`도 상태 키를 넣습니다.

    *해설:* 원본 `ACTConfig` 설명에는 상태 없이도 동작한다고 되어 있지만, `z=0`을 만드는 줄이 `OBS_STATE`의 device를 읽습니다.
    원본 코드를 읽을 때 이런 숨은 가정을 찾는 연습입니다. 위치 정보가 이미지(빨간 점)에만 있으면 64×64 해상도에서 정밀도가 떨어지는지도 확인해 보세요.
    """),
        "과제 5. kl_weight 중간값 (8절)": mo.md("""
    `kl_weight`를 1과 0.1로 학습해 8절의 상관과 KL을 비교하세요. 어느 값에서 `z`가 대기 시간을 담기 시작하나요?

    *해설:* 사전 검증에서 `kl_weight=1`도 KL이 0.001 nats 이하로 붕괴했습니다. 데이터 중 `z`가 쓸모 있는 프레임(대기 직전)의 비율이 작을수록
    붕괴가 쉽게 일어납니다. 실제 로봇 데이터에서는 시연자마다 속도·경로가 달라 `z`가 쓸모 있는 프레임이 훨씬 많습니다.
    """),
    })
    return


if __name__ == "__main__":
    app.run()
