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
    # ACT 보충 실험: 갈림길 과제 · 학습 · 폐루프 평가

    **ACT(Action Chunking with Transformers)** 는 Tony Z. Zhao 외(2023)가 저비용 양팔 원격조작 시스템
    **ALOHA**와 함께 발표한 모방 학습 정책입니다. 카메라 이미지와 관절 상태를 보고 **다음 행동 하나가 아니라
    앞으로 `k`개의 행동 묶음(action chunk)** 을 한 번의 forward로 예측합니다. LeRobot의 기본 정책 중 하나입니다.

    [ACT 구조 실습](act_tour.py)에서 분리한 선택 실험입니다. 환경 구현, MLP 비교, 매개변수 집계, β별 학습과 잠재 변수 분석, 폐루프 평가와 다른 정책 비교를 모았습니다.
    모델 부품은 본문의 이름 있는 셀을 재사용합니다. 본문 앱을 실행하거나 먼저 학습할 필요는 없습니다.
    act.md의 코드는 개념을 보여 주기 위해 단순화되어 있으므로, 각 단계에서 LeRobot 원본 구현과 다른 점도 함께 짚습니다.

    | 절 | 내용 | act.md 대응 | 자원 |
    | --- | --- | --- | --- |
    | 1 | 문제 만들기: 갈림길 과제와 단일 스텝 정책의 실패 | 1. 등장 배경 | **버튼**, CPU 약 10초 |
    | 2 | 학습 배치: 행동 청크와 `action_is_pad` | — | 즉시 |
    | 3 | 부품 하나씩: 위치 임베딩 → ResNet → DETR 층 → CVAE 인코더 → ACT 조립 | 2. 아키텍처, 코드 1·2 | 즉시 |
    | 4 | 손실: 마스크된 L1 + KL | 4. 손실, 코드 3 | 즉시 |
    | 5 | KL 가중치 β별 학습 | — | **버튼**, GPU 모델당 약 1분 |
    | 6 | 추론(`z = 0`)과 잠재 변수 `z`에 담긴 것 | ① 추론 시 z=0 | 즉시(학습 후) |
    | 7 | 시간 앙상블: act.md 방식 A·B와 LeRobot 비교 | 3. Temporal Ensembling | 즉시 |
    | 8 | 폐루프 평가: 청크 실행 방식 비교, 외란 | — | **버튼**, 수십 초 |
    | 9 | 정리: 모델 비교, act.md와 원본의 차이, 연습 문제 | 5. 비교 | — |

    **실행:** `notebooks/lerobot`에서 `uv sync --locked` 후 `uv run marimo edit act_experiments.py`.
    다운로드는 없습니다. ResNet-18도 무작위 초기화로 만듭니다. 버튼이 있는 셀(학습·반복 평가)만 직접 눌러야 실행됩니다.

    > **원본 코드**: [huggingface/lerobot `src/lerobot/policies/act`](https://github.com/huggingface/lerobot/tree/e6ffde60312365c5452c46189a26872483943154/src/lerobot/policies/act)
    > (`configuration_act.py`, `modeling_act.py`, revision `e6ffde60312365c5452c46189a26872483943154`).
    > 원본은 Apache License 2.0, Copyright 2024–2026 The HuggingFace Inc. team이며, ACT 원 저자 코드
    > [tonyzhaozh/act](https://github.com/tonyzhaozh/act)를 기반으로 합니다. 이 노트북은 그 구조를 따라 축소·재작성했고
    > `lerobot` 패키지는 설치하지 않습니다.
    """)
    return



@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 학습 목표

    이 실습을 마치면 다음을 **코드와 실행 결과로** 설명할 수 있어야 합니다.

    1. 같은 관측에 정답이 둘 이상일 때(다봉성) 단일 스텝 회귀가 왜 "평균 행동"으로 실패하는가.
    2. ACT의 입력 토큰 `[z, 상태, 이미지 h·w개]`과 디코더 쿼리 `k`개가 어떤 shape으로 흘러 `(B, k, A)` 행동 청크가 되는가.
    3. CVAE 인코더가 학습 때만 쓰이고, 추론 때 `z = 0`을 넣어도 되는 조건은 무엇이며, KL 가중치 β가 이를 어떻게 바꾸는가.
    4. 시간 앙상블의 가중치 방향(가장 **오래된** 예측이 `w_0`)과 LeRobot의 온라인 갱신 구현이 같은 값을 내는 이유.

    **기호**: $B$=배치, $k$=청크 길이(`chunk_size`), $A$=행동 차원, $D$=`dim_model`, $h·w$=이미지 특징 지도 크기.

    **먼저 답해 보기**: 장애물을 왼쪽으로 피한 시연과 오른쪽으로 피한 시연이 반반 섞인 데이터를 MSE로 학습하면,
    장애물 바로 앞에서 정책은 어느 쪽으로 갈까요? 1절에서 직접 확인합니다.
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
    import json
    import math
    import random
    import time
    from collections import deque
    from dataclasses import asdict, dataclass, replace
    from datetime import datetime
    from pathlib import Path

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
        Path,
        asdict,
        dataclass,
        datetime,
        deque,
        json,
        math,
        nn,
        np,
        plt,
        random,
        replace,
        time,
        torch,
        torchvision,
    )



@app.cell
def _(Path, datetime, torch):
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ON_GPU = DEVICE.type == "cuda"
    REPO_ROOT = Path(__file__).resolve().parents[2]
    # 학습 로그와 평가 지표는 저장소 루트 outputs/lerobot/<실행 ID>/에 저장합니다(버튼을 눌렀을 때만).
    RUN_DIR = REPO_ROOT / "outputs" / "lerobot" / f"{datetime.now():%Y%m%d-%H%M%S}-act-experiments"

    # LeRobot 배치 키 이름(lerobot/utils/constants.py)을 그대로 씁니다.
    OBS_STATE = "observation.state"
    OBS_IMAGE = "observation.images.top"
    ACTION = "action"
    ACTION_IS_PAD = "action_is_pad"
    print(f"장치: {DEVICE}" + ("" if ON_GPU else "  (CPU: 5절 학습이 GPU보다 수십 배 느립니다)"))
    return ACTION, ACTION_IS_PAD, DEVICE, OBS_IMAGE, OBS_STATE, ON_GPU, RUN_DIR



@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. 문제 만들기: 갈림길 과제

    act.md 1절은 단일 스텝 행동 복제(Behavior Cloning)의 두 가지 한계로 **누적 오차**와 **다봉성(multimodality)** 을 듭니다.
    말로만 보지 않고, 다봉성이 확실히 드러나는 작은 2D 과제를 만들어 확인합니다.

    로봇(빨간 점)이 아래에서 출발해 **장애물(회색 원)** 을 피해 **목표(초록 원)** 로 갑니다. 좌표는 `[0, 1]²`이고
    이미지와 같은 방향(y가 아래로 증가)입니다.

    | 항목 | 실제 로봇 (ALOHA) | 이 노트북 |
    | --- | --- | --- |
    | 이미지 `observation.images.*` | 카메라 4대, 480×640 | 카메라 1대 `top`, 64×64 RGB |
    | 상태 `observation.state` | 관절 각도 14개 | 로봇 위치 `(x, y)` |
    | 행동 `action` | 다음 목표 관절 각도 14개(절대값) | 다음 목표 위치 `(x, y)`(절대값) |
    | 제어 주기 | 50 Hz | 1 step |

    **전문가 시연**: 출발점 → 장애물 바로 아래 **갈림 지점** → 장애물의 왼쪽 **또는** 오른쪽 → 목표 → 5 step 정지.
    같은 장면마다 **왼쪽 시연 1개, 오른쪽 시연 1개**를 기록합니다(두 사람이 시연한 상황). 이동 속도도 시연마다 조금씩 다릅니다.
    따라서 갈림 지점에서는 **이미지와 상태가 똑같은데 정답 행동이 왼쪽/오른쪽 두 가지**입니다.

    환경 규칙(`env_step`): 로봇은 행동(목표 위치) 쪽으로 이동하되 한 step에 최대 `MAX_STEP=0.06`만 움직입니다.
    장애물 중심에서 `OBS_R` 안으로 들어가면 **충돌(실패)**, 목표에서 `0.05` 이내면 **성공**, `MAX_T=60` step을 넘으면 시간 초과입니다.
    """)
    return



@app.cell
def _(math, np, torch):
    IMG = 64          # 이미지 한 변의 픽셀 수
    OBS_R = 0.12      # 장애물 반지름
    MAX_STEP = 0.06   # 환경이 허용하는 한 step 최대 이동 거리
    GOAL_TOL = 0.05   # 성공 판정 거리
    MAX_T = 60        # 평가 에피소드 최대 길이
    HOLD = 5          # 전문가가 목표에서 멈춰 있는 step 수


    def sample_scene(rng):
        """출발점(아래), 장애물(가운데), 목표(위)를 무작위로 놓습니다."""
        start = np.array([rng.uniform(0.35, 0.65), rng.uniform(0.85, 0.9)])
        obstacle = np.array([rng.uniform(0.4, 0.6), rng.uniform(0.45, 0.55)])
        goal = np.array([rng.uniform(0.3, 0.7), rng.uniform(0.1, 0.15)])
        return start, obstacle, goal


    def fork_point(obstacle):
        """장애물 바로 아래의 갈림 지점"""
        return obstacle + np.array([0.0, OBS_R + 0.1])


    def straight(a, b, speed):
        """a에서 b까지 speed 간격의 목표 위치 목록 (b 포함)"""
        n = max(1, math.ceil(np.linalg.norm(b - a) / speed))
        return [a + (b - a) * (i / n) for i in range(1, n + 1)]


    def expert_episode(scene, side, speed):
        """스크립트 전문가 시연: 갈림 지점 → 장애물 왼쪽(side=-1)/오른쪽(+1) → 목표 → 정지"""
        start, obstacle, goal = scene
        fork = fork_point(obstacle)
        detour = obstacle + np.array([side * (OBS_R + 0.08), 0.0])
        to_fork = straight(start, fork, speed)
        targets = to_fork + straight(fork, detour, speed) + straight(detour, goal, speed) + [goal.copy()] * HOLD
        # 전문가 속도(<= 0.05)가 MAX_STEP보다 작으므로 로봇은 매 step 목표 위치에 정확히 도착합니다.
        states = [start] + targets[:-1]
        return dict(
            start=start, obstacle=obstacle, goal=goal, side=side, speed=speed,
            fork_t=len(to_fork),  # state[fork_t]가 갈림 지점
            state=np.array(states, dtype=np.float32),
            action=np.array(targets, dtype=np.float32),
        )


    def env_step(pos, action):
        """여러 환경을 한 번에 1 step 진행합니다. pos, action: (N, 2)"""
        delta = action - pos
        dist = np.linalg.norm(delta, axis=1, keepdims=True)
        return pos + delta * np.minimum(1.0, MAX_STEP / np.maximum(dist, 1e-9))


    def stack_scenes(scenes):
        """장면 목록 → {'start','obstacle','goal'}: 각 (N, 2) 배열"""
        return {k: np.array([s[i] for s in scenes]) for i, k in enumerate(["start", "obstacle", "goal"])}


    def run_episodes(scenes, policy, push=None, max_t=MAX_T):
        """N개 환경을 동시에 실행합니다.

        policy(t, pos) -> (N, 2) 행동. push=(t, (N, 2) 변위)면 그 step 직전에 로봇을 밀어냅니다(외란).
        반환: traj (N, T+1, 2), action (N, T, 2), status (N,) 0=시간초과 1=성공 2=충돌, end_t (N,)
        """
        pos = scenes["start"].astype(np.float64).copy()
        n = len(pos)
        status = np.zeros(n, dtype=int)
        end_t = np.full(n, max_t)
        traj, actions = [pos.copy()], []
        for t in range(max_t):
            active = status == 0
            if push is not None and t == push[0]:
                pos = np.where(active[:, None], pos + push[1], pos)
            action = np.asarray(policy(t, pos), dtype=np.float64)
            pos = np.where(active[:, None], env_step(pos, action), pos)
            hit = active & (np.linalg.norm(pos - scenes["obstacle"], axis=1) < OBS_R)
            ok = active & ~hit & (np.linalg.norm(pos - scenes["goal"], axis=1) < GOAL_TOL)
            status[hit], status[ok] = 2, 1
            end_t[(hit | ok)] = t + 1
            traj.append(pos.copy())
            actions.append(action.copy())
            if (status != 0).all():
                break
        return dict(traj=np.stack(traj, 1), action=np.stack(actions, 1), status=status, end_t=end_t)


    def render(agent, obstacle, goal):
        """위치로부터 이미지를 그립니다. 입력 (B, 2) 텐서 → (B, 3, IMG, IMG), 값 [0, 1]"""
        coords = (torch.arange(IMG, device=agent.device) + 0.5) / IMG
        yy, xx = torch.meshgrid(coords, coords, indexing="ij")

        def disk(p, r):
            return (((xx - p[:, 0, None, None]) ** 2 + (yy - p[:, 1, None, None]) ** 2) < r * r).float()

        img = torch.full((agent.shape[0], 3, IMG, IMG), 0.1, device=agent.device)
        img = torch.maximum(img, 0.5 * disk(obstacle, OBS_R)[:, None])      # 회색: 장애물
        img[:, 1] = torch.maximum(img[:, 1], 0.8 * disk(goal, 0.06))         # 초록: 목표
        img[:, 0] = torch.maximum(img[:, 0], 0.95 * disk(agent, 0.03))       # 빨강: 로봇
        return img
    return (
        GOAL_TOL,
        IMG,
        MAX_T,
        OBS_R,
        expert_episode,
        fork_point,
        render,
        run_episodes,
        sample_scene,
        stack_scenes,
    )



@app.cell
def _(expert_episode, np, random, sample_scene, stack_scenes):
    # 학습용 시연: 장면 200개 × (왼쪽, 오른쪽) = 400개. 속도는 시연마다 0.03~0.05에서 무작위.
    N_SCENES = 200
    episodes = []
    for _i in range(N_SCENES):
        _scene = sample_scene(random.Random(_i))
        for _side in (-1, 1):
            _speed = random.Random(100_000 + 2 * _i + (_side > 0)).uniform(0.03, 0.05)
            episodes.append(expert_episode(_scene, _side, _speed))

    # 평가용 장면 100개(학습 장면과 겹치지 않는 seed). 같은 장면의 왼쪽/오른쪽 시연도 만들어 둡니다(6절).
    test_scene_list = [sample_scene(random.Random(10_000 + _i)) for _i in range(100)]
    TEST_SCENES = stack_scenes(test_scene_list)
    test_pairs = [(expert_episode(_s, -1, 0.04), expert_episode(_s, 1, 0.04)) for _s in test_scene_list]

    _lens = np.array([len(e["state"]) for e in episodes])
    print(f"학습 시연 {len(episodes)}개, 프레임 {_lens.sum()}개, 길이 {_lens.min()}~{_lens.max()} (평균 {_lens.mean():.1f})")
    print(f"평가 장면 {len(test_scene_list)}개")
    return TEST_SCENES, episodes, test_pairs, test_scene_list



@app.cell
def _(DEVICE, IMG, OBS_R, fork_point, np, plt, render, torch):
    def draw_scene(ax, scene, agent=None, title=None):
        """장면 이미지를 [0,1]² 좌표에 그립니다. agent를 생략하면 출발점에 로봇을 그립니다."""
        start, obstacle, goal = scene
        _f = lambda x: torch.tensor(np.asarray(x, dtype=np.float32)[None], device=DEVICE)
        img = render(_f(start if agent is None else agent), _f(obstacle), _f(goal))[0]
        ax.imshow(img.permute(1, 2, 0).cpu().numpy(), extent=[0, 1, 1, 0])
        ax.add_patch(plt.Circle(obstacle, OBS_R, fill=False, ec="w", lw=0.6, ls=":"))
        fork = fork_point(obstacle)
        ax.plot(*fork, "w+", ms=8)
        ax.set_xlim(0, 1); ax.set_ylim(1, 0)
        ax.set_xticks([]); ax.set_yticks([])
        if title:
            ax.set_title(title, fontsize=9)

    print(f"이미지 한 장: 3×{IMG}×{IMG}, 흰 + 표시는 갈림 지점(그림에만 표시, 관측에는 없음)")
    return (draw_scene,)



@app.cell
def _(draw_scene, episodes, plt):
    _fig, _axes = plt.subplots(1, 4, figsize=(13, 3.6))
    for _j, _ax in enumerate(_axes):
        _left, _right = episodes[2 * _j], episodes[2 * _j + 1]
        draw_scene(_ax, (_left["start"], _left["obstacle"], _left["goal"]), title=f"장면 {_j}: 시연 2개")
        _ax.plot(*_left["action"].T, ".-", c="cyan", ms=3, lw=1, label="왼쪽 시연")
        _ax.plot(*_right["action"].T, ".-", c="gold", ms=3, lw=1, label="오른쪽 시연")
    _axes[0].legend(fontsize=8, loc="lower left")
    _fig.suptitle("같은 장면, 두 갈래 정답: 갈림 지점(+)까지는 같은 길, 그 뒤로 왼쪽/오른쪽")
    _fig.tight_layout()
    _fig
    return



@app.cell
def _(episodes, np):
    # 갈림 지점 프레임에서 "관측은 같은데 다음 행동이 다른가"를 직접 확인합니다.
    _l, _r = episodes[0], episodes[1]
    _tl, _tr = _l["fork_t"], _r["fork_t"]
    print("왼쪽 시연  갈림 지점 상태:", _l["state"][_tl].round(3), "→ 다음 행동:", _l["action"][_tl].round(3))
    print("오른쪽 시연 갈림 지점 상태:", _r["state"][_tr].round(3), "→ 다음 행동:", _r["action"][_tr].round(3))
    print("상태 차이:", np.abs(_l["state"][_tl] - _r["state"][_tr]).max(), " (장애물·목표 위치도 같으므로 이미지도 같음)")
    print(f"모든 시연이 갈림 지점을 1번씩 지나고, 그 직후 정답의 x 방향은 정확히 반반입니다: "
          f"왼쪽 {sum(e['side'] < 0 for e in episodes)}개 / 오른쪽 {sum(e['side'] > 0 for e in episodes)}개")
    return



@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 1-1. 단일 스텝 정책으로 먼저 풀어 보기

    가장 단순한 행동 복제부터 해 봅니다. 이미지 대신 **정답 위치를 직접 받는** 작은 MLP
    `[로봇 위치, 장애물 위치, 목표 위치] → 다음 1 step 이동량`을 MSE와 L1로 각각 학습합니다.
    이미지를 쓰지 않으므로 ACT보다 훨씬 쉬운 조건입니다.

    **예측해 보기**: 갈림 지점에서 왼쪽(−x)과 오른쪽(+x) 정답이 반반이면, MSE를 최소화하는 출력의 x 성분은 얼마일까요?
    """)
    return



@app.cell
def _(mo):
    mlp_button = mo.ui.run_button(label="단일 스텝 MLP 학습·평가 (CPU 약 10초)")
    mlp_button
    return (mlp_button,)



@app.cell
def _(
    TEST_SCENES,
    episodes,
    fork_point,
    mlp_button,
    mo,
    nn,
    np,
    run_episodes,
    torch,
):
    mo.stop(not mlp_button.value, mo.md("버튼을 누르면 MSE·L1 MLP를 각 3000 step 학습하고 평가 장면 100개에서 실행합니다."))

    _X = np.concatenate([
        np.concatenate([e["state"], np.repeat(e["obstacle"][None], len(e["state"]), 0),
                        np.repeat(e["goal"][None], len(e["state"]), 0)], 1)
        for e in episodes
    ]).astype(np.float32)
    _Y = np.concatenate([e["action"] - e["state"] for e in episodes]).astype(np.float32)  # 다음 1 step 이동량
    _X, _Y = torch.from_numpy(_X), torch.from_numpy(_Y)

    mlp_results = {}
    for _loss_name in ["MSE", "L1"]:
        torch.manual_seed(0)
        _net = nn.Sequential(nn.Linear(6, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 2))
        _opt = torch.optim.Adam(_net.parameters(), lr=1e-3)
        for _step in range(3000):
            _idx = torch.randint(0, len(_X), (256,))
            _err = _net(_X[_idx]) - _Y[_idx]
            _loss = (_err**2).mean() if _loss_name == "MSE" else _err.abs().mean()
            _opt.zero_grad(); _loss.backward(); _opt.step()
        _net.eval()

        def _policy(t, pos, net=_net):
            feat = np.concatenate([pos, TEST_SCENES["obstacle"], TEST_SCENES["goal"]], 1)
            with torch.no_grad():
                return pos + net(torch.tensor(feat, dtype=torch.float32)).numpy()

        _res = run_episodes(TEST_SCENES, _policy)
        # 갈림 지점에서 예측한 이동량의 x 성분 (전문가는 왼쪽/오른쪽으로 약 ±0.03)
        _fork = np.array([fork_point(o) for o in TEST_SCENES["obstacle"]])
        with torch.no_grad():
            _dx = _net(torch.tensor(np.concatenate([_fork, TEST_SCENES["obstacle"], TEST_SCENES["goal"]], 1),
                                    dtype=torch.float32))[:, 0].numpy()
        mlp_results[_loss_name] = dict(res=_res, fork_dx=_dx)

    _rows = "\n".join(
        f"| {k} | {np.mean(v['res']['status'] == 1):.0%} | {np.mean(v['res']['status'] == 2):.0%} | "
        f"{np.mean(v['res']['status'] == 0):.0%} | {np.abs(v['fork_dx']).mean():.4f} |"
        for k, v in mlp_results.items()
    )
    mo.md(
        "| 손실 | 성공 | 충돌 | 시간 초과 | 갈림 지점에서 예측 \\|dx\\| 평균 |\n| --- | --- | --- | --- | --- |\n" + _rows
        + "\n\n전문가의 갈림 지점 직후 \\|dx\\|는 약 0.02~0.035입니다. 평가 장면 100개, 학습 seed 0 한 번의 결과입니다."
    )
    return (mlp_results,)



@app.cell
def _(TEST_SCENES, draw_scene, mlp_results, plt, test_pairs, test_scene_list):
    _fig, _axes = plt.subplots(1, 4, figsize=(13, 3.6))
    for _j, _ax in enumerate(_axes):
        draw_scene(_ax, test_scene_list[_j], title=f"평가 장면 {_j}")
        for _ep, _c in zip(test_pairs[_j], ["cyan", "gold"]):
            _ax.plot(*_ep["action"].T, "--", c=_c, lw=0.8)
        for _name, _c in [("MSE", "red"), ("L1", "orange")]:
            _r = mlp_results[_name]["res"]
            _end = _r["end_t"][_j]
            _ax.plot(*_r["traj"][_j, : _end + 1].T, "-", c=_c, lw=1.6, label=_name)
            if _r["status"][_j] == 2:
                _ax.plot(*_r["traj"][_j, _end], "x", c=_c, ms=10, mew=2)
    _axes[0].legend(fontsize=8, loc="lower left")
    _fig.suptitle("단일 스텝 MLP (실선, x=충돌) vs 전문가 두 갈래(점선)")
    _fig.tight_layout()
    _fig
    return



@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **관찰 포인트** (기준 실행: MSE 성공 58%·충돌 42%, L1 성공 33%·충돌 67%, 갈림 지점 예측 |dx| 0.003·0.007.
    전문가의 |dx|는 약 0.02~0.035입니다)

    - MSE의 최적해는 조건부 **평균**입니다. 왼쪽(−dx)과 오른쪽(+dx)이 반반이면 평균 dx ≈ 0, 즉 **장애물 정중앙으로 직진**합니다.
      표의 "갈림 지점 |dx|"가 전문가보다 훨씬 작다면 이 평균화가 일어난 것입니다.
    - L1의 최적해는 조건부 **중앙값**입니다. 정확히 반반이면 중앙값은 두 값 사이 어디든 될 수 있어 운 좋게 한쪽을 고를 수도 있지만,
      다봉성 자체를 해결하지는 않습니다. act.md 4절이 L1을 쓰는 이유로 든 "정밀한 제어"와 다봉성 해결은 별개의 문제입니다.
    - 한 번 평균 방향으로 가면 상태가 시연에 없던 영역(두 갈래 사이)으로 들어갑니다. 이후 예측은 외삽이고 오차가 누적됩니다
      (act.md의 **누적 오차**).

    ACT는 이 문제에 **(1) 행동을 묶음으로 예측**하고 **(2) CVAE로 "어느 갈래인가" 같은 스타일을 잠재 변수 `z`로 분리**하는 방법을 씁니다.
    정말 해결되는지는 5~8절에서 확인합니다.

    ## 2. 학습 배치: 행동 청크와 `action_is_pad`

    LeRobot의 `LeRobotDataset`은 프레임 `t`를 꺼낼 때 `delta_timestamps`에 따라 **미래 행동 `t, t+1, …, t+k-1`** 을 함께 붙여 줍니다
    (`ACTConfig.action_delta_indices = list(range(chunk_size))`). 에피소드 끝을 넘어가는 자리는 마지막 행동으로 채우고
    `action_is_pad=True`로 표시합니다. 이 표시는 4절 손실과 3-4절 CVAE 인코더에서 쓰입니다.

    상태와 행동은 데이터셋 평균·표준편차로 정규화합니다(LeRobot의 `NormalizerProcessor`, `MEAN_STD` 모드에 해당).
    이미지는 ImageNet 사전 학습 백본을 쓰지 않으므로 `[0, 1]` 그대로 둡니다.

    | 배치 키 | 모양 | 의미 |
    | --- | --- | --- |
    | `observation.state` | `(B, 2)` | 정규화된 현재 위치 |
    | `observation.images.top` | `(B, 3, 64, 64)` | 현재 이미지 |
    | `action` | `(B, k, 2)` | 정규화된 정답 행동 청크 |
    | `action_is_pad` | `(B, k)` bool | 에피소드 끝을 넘어선 자리 |
    """)
    return



@app.cell
def _(
    ACTION,
    ACTION_IS_PAD,
    DEVICE,
    OBS_IMAGE,
    OBS_STATE,
    episodes,
    np,
    render,
    torch,
):
    CHUNK = 20  # 이 노트북의 청크 길이 k (LeRobot 기본값은 100)

    _all_s = np.concatenate([e["state"] for e in episodes])
    _all_a = np.concatenate([e["action"] for e in episodes])
    STATS = {
        "state_mean": torch.tensor(_all_s.mean(0), device=DEVICE), "state_std": torch.tensor(_all_s.std(0), device=DEVICE),
        "action_mean": torch.tensor(_all_a.mean(0), device=DEVICE), "action_std": torch.tensor(_all_a.std(0), device=DEVICE),
    }
    _ep_of_frame = np.concatenate([[i] * len(e["state"]) for i, e in enumerate(episodes)])
    _t_of_frame = np.concatenate([np.arange(len(e["state"])) for e in episodes])


    def _t(x):
        return torch.tensor(np.asarray(x), dtype=torch.float32, device=DEVICE)


    def obs_batch(pos, obstacle, goal):
        """위치(N,2) 배열들 → 정책 입력 배치 (정규화된 상태 + 렌더링한 이미지)"""
        pos, obstacle, goal = _t(pos), _t(obstacle), _t(goal)
        return {
            OBS_STATE: (pos - STATS["state_mean"]) / STATS["state_std"],
            OBS_IMAGE: render(pos, obstacle, goal),
        }


    def chunk_at(ep, t, chunk_size=CHUNK):
        """에피소드 ep의 프레임 t에서 시작하는 행동 청크와 패딩 표시"""
        idx = np.arange(t, t + chunk_size)
        is_pad = idx >= len(ep["action"])
        return ep["action"][np.minimum(idx, len(ep["action"]) - 1)], is_pad


    def make_batch(batch_size, rng, frames=None, chunk_size=CHUNK):
        """무작위 프레임(또는 (에피소드, t) 목록 frames)으로 LeRobot 형식의 학습 배치를 만듭니다."""
        if frames is None:
            pick = rng.integers(0, len(_ep_of_frame), batch_size)
            frames = list(zip(_ep_of_frame[pick], _t_of_frame[pick]))
        eps = [episodes[i] for i, _ in frames]
        chunks = [chunk_at(e, t, chunk_size) for e, (_, t) in zip(eps, frames)]
        batch = obs_batch([e["state"][t] for e, (_, t) in zip(eps, frames)],
                          [e["obstacle"] for e in eps], [e["goal"] for e in eps])
        batch[ACTION] = (_t([c[0] for c in chunks]) - STATS["action_mean"]) / STATS["action_std"]
        batch[ACTION_IS_PAD] = torch.tensor(np.array([c[1] for c in chunks]), device=DEVICE)
        return batch


    def unnormalize_action(a):
        return a * STATS["action_std"] + STATS["action_mean"]


    _b = make_batch(8, np.random.default_rng(0))
    for _k, _v in _b.items():
        print(f"{_k:24s} {tuple(_v.shape)}  {_v.dtype}")
    return CHUNK, STATS, chunk_at, make_batch, obs_batch, unnormalize_action



@app.cell
def _(CHUNK, chunk_at, draw_scene, episodes, plt):
    _l, _r = episodes[0], episodes[1]
    _fig, _axes = plt.subplots(1, 2, figsize=(9, 4.2))
    draw_scene(_axes[0], (_l["start"], _l["obstacle"], _l["goal"]), agent=_l["state"][_l["fork_t"]],
               title=f"갈림 지점 프레임의 정답 청크 (k={CHUNK})")
    for _ep, _c in [(_l, "cyan"), (_r, "gold")]:
        _a, _pad = chunk_at(_ep, _ep["fork_t"])
        _axes[0].plot(*_a.T, ".-", c=_c, ms=4, lw=1)
    _axes[0].text(0.02, 0.97, "같은 관측 → 정답 청크 2개", c="w", fontsize=9, va="top")

    _t0 = len(_l["action"]) - 8
    _a, _pad = chunk_at(_l, _t0)
    draw_scene(_axes[1], (_l["start"], _l["obstacle"], _l["goal"]), agent=_l["state"][_t0],
               title=f"에피소드 끝 근처 t={_t0}: 패딩 {int(_pad.sum())}개")
    _axes[1].plot(*_a[~_pad].T, ".-", c="cyan", ms=4, lw=1, label="실제 행동")
    _axes[1].plot(*_a[_pad].T, "s", c="magenta", ms=6, mfc="none", label="패딩(마지막 행동 반복)")
    _axes[1].legend(fontsize=8, loc="lower left")
    _fig.tight_layout()
    _fig
    return



@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    왼쪽 그림이 이 과제의 핵심입니다. **입력이 같은데 정답 청크가 둘**입니다. 청크 단위로 예측해도 디코더가 무엇으로
    갈래를 고를지 정보가 없으면, L1/MSE는 여전히 두 청크의 중간을 낼 수 있습니다. 그래서 ACT는 청크 + CVAE를 함께 씁니다.

    ## 3. 부품 하나씩 만들기

    act.md 2절의 구조를 아래 순서로 조립합니다. 각 단계 끝에서 작은 입력으로 shape과 성질을 확인합니다.

    | 단계 | 부품 | LeRobot 원본 이름 |
    | --- | --- | --- |
    | 3-1 | 1D·2D 사인파 위치 임베딩 | `create_sinusoidal_pos_embedding`, `ACTSinusoidalPositionEmbedding2d` |
    | 3-2 | ResNet-18 백본 → 이미지 토큰 | `backbone`, `encoder_img_feat_input_proj` |
    | 3-3 | DETR식 인코더·디코더 층 | `ACTEncoderLayer`, `ACTDecoderLayer` |
    | 3-4 | CVAE 인코더 | `vae_encoder*` |
    | 3-5 | 전체 `ACT` | `ACT` |
    | 3-6 | 실제 크기의 매개변수 수 | `ACTConfig` 기본값 |

    ### 3-1. 위치 임베딩

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
def _(math, np, torch):
    from act_tour import position_definitions

    _, _defs = position_definitions.run(math=math, np=np, torch=torch)
    create_sinusoidal_pos_embedding = _defs["create_sinusoidal_pos_embedding"]
    sinusoidal_2d = _defs["sinusoidal_2d"]
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
    ### 3-2. 비전 백본: ResNet-18 → 이미지 토큰

    ACT는 torchvision ResNet-18에서 **`layer4` 특징 지도만** 꺼냅니다(`IntermediateLayerGetter`). 해상도가 1/32로 줄고 채널은 512개입니다.
    그다음 1×1 합성곱으로 채널을 `D`로 줄이고, **픽셀 하나를 토큰 하나**로 펼칩니다.

    원본은 ImageNet 사전 학습 가중치(`ResNet18_Weights.IMAGENET1K_V1`)와 `FrozenBatchNorm2d`를 씁니다.
    여기서는 다운로드 없이 진행하려고 **무작위 초기화**로 만듭니다. FrozenBatchNorm은 통계가 고정이라 이 경우 사실상 항등 함수입니다.
    """)
    return



@app.cell
def _(FrozenBatchNorm2d, IntermediateLayerGetter, torch, torchvision):
    from act_tour import backbone_definitions

    _, _defs = backbone_definitions.run(FrozenBatchNorm2d=FrozenBatchNorm2d, IntermediateLayerGetter=IntermediateLayerGetter, torch=torch, torchvision=torchvision)
    make_backbone = _defs["make_backbone"]
    return make_backbone,


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 3-3. DETR식 인코더·디코더 층

    act.md 코드는 `nn.Transformer`에 위치 임베딩을 더한 입력을 넣습니다. LeRobot(과 DETR)은 다르게,
    **매 층의 attention에서 위치 임베딩을 query와 key에만 더하고 value에는 더하지 않습니다.** 정규화는 잔차 뒤(post-norm)입니다.

    디코더 층은 self-attention → **cross-attention(query: 행동 쿼리, key/value: 인코더 출력)** → FFN 순서이고 **마스크가 없습니다**.
    GPT처럼 한 토큰씩 만드는 것이 아니라 `k`개를 동시에 만들기 때문입니다.

    LeRobot은 `(seq, batch, dim)` 순서를 쓰지만 여기서는 읽기 쉽게 `batch_first=True`로 씁니다. 계산은 같습니다.
    """)
    return



@app.cell
def _(nn):
    from act_tour import transformer_definitions

    _, _defs = transformer_definitions.run(nn=nn)
    DetrDecoderLayer = _defs["DetrDecoderLayer"]
    DetrEncoderLayer = _defs["DetrEncoderLayer"]
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

    ### 3-4. CVAE 인코더 (학습 때만 사용)

    act.md 코드 1의 `ACTCVAEEncoder`와 같은 일을 합니다. 카메라 이미지는 보지 않고 **현재 상태와 정답 행동 청크**만 봅니다.

    1. 토큰 `[CLS, 상태, 행동 k개]` → `(B, k+2, D)`, 고정 1D 위치 임베딩.
    2. 패딩된 행동 토큰은 `key_padding_mask`로 가립니다(act.md 코드에는 없음). CLS가 가짜 행동을 읽지 않게 하기 위해서입니다.
    3. CLS 출력 → `Linear(D, 2·latent)` → 평균 `μ`와 `log σ²`(원본 이름 `log_sigma_x2`).
    4. 재매개변수화 `z = μ + σ·ε`, `ε ~ N(0, I)`. 샘플링을 미분 가능한 덧셈·곱셈으로 바꿔 `μ, σ`까지 기울기가 흐르게 합니다.
    """)
    return



@app.cell
def _(DetrEncoderLayer, create_sinusoidal_pos_embedding, nn, torch):
    from act_tour import cvae_definitions

    _, _defs = cvae_definitions.run(DetrEncoderLayer=DetrEncoderLayer, create_sinusoidal_pos_embedding=create_sinusoidal_pos_embedding, nn=nn, torch=torch)
    CVAEEncoder = _defs["CVAEEncoder"]
    reparameterize = _defs["reparameterize"]
    return CVAEEncoder, reparameterize


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 3-5. 설정과 전체 `ACT` 조립

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
def _(ACTION, ACTION_IS_PAD, CVAEEncoder, DetrDecoderLayer, DetrEncoderLayer, OBS_IMAGE, OBS_STATE, dataclass, make_backbone, nn, reparameterize, replace, sinusoidal_2d, torch):
    from act_tour import act_definitions

    _, _defs = act_definitions.run(ACTION=ACTION, ACTION_IS_PAD=ACTION_IS_PAD, CVAEEncoder=CVAEEncoder, DetrDecoderLayer=DetrDecoderLayer, DetrEncoderLayer=DetrEncoderLayer, OBS_IMAGE=OBS_IMAGE, OBS_STATE=OBS_STATE, dataclass=dataclass, make_backbone=make_backbone, nn=nn, reparameterize=reparameterize, replace=replace, sinusoidal_2d=sinusoidal_2d, torch=torch)
    ACT = _defs["ACT"]
    ACTConfig = _defs["ACTConfig"]
    TOY_CFG = _defs["TOY_CFG"]
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
    이미지가 64×64라서 이미지 토큰은 2×2 = 4개, 인코더 토큰은 `z` 1 + 상태 1 + 이미지 4 = **6개**입니다.
    디코더는 이 6개 토큰을 cross-attention으로 읽어 행동 20개를 **한 번에** 냅니다.

    ### 3-6. 실제 크기에서는 몇 개의 매개변수일까?

    act.md 5절은 ACT를 "약 80M"이라고 소개합니다. LeRobot 기본 설정(상태·행동 14차원, 카메라 1대)으로 직접 세어 봅니다.
    무작위 초기화라 다운로드는 없지만 수천만 개 매개변수를 CPU 메모리에 만듭니다(수 초).
    """)
    return



@app.cell
def _(ACT, ACTConfig, mo, replace):
    def count_groups(model):
        groups = {}
        for name, p in model.named_parameters():
            key = name.split(".")[0]
            key = "백본 ResNet-18" if key == "backbone" else ("CVAE 인코더" if key == "vae_encoder" else
                  "Transformer 인코더" if key == "encoder" else "Transformer 디코더" if key == "decoder" else "투영·임베딩·헤드")
            groups[key] = groups.get(key, 0) + p.numel()
        return groups

    _g1 = count_groups(ACT(ACTConfig()))
    _g7 = count_groups(ACT(replace(ACTConfig(), n_decoder_layers=7)))
    _rows = "\n".join(f"| {k} | {_g1[k] / 1e6:.2f}M | {_g7[k] / 1e6:.2f}M |" for k in _g1)
    mo.md(
        "| 모듈 | LeRobot 기본 (디코더 1층) | act.md·원 논문 설정 (디코더 7층) |\n| --- | --- | --- |\n" + _rows
        + f"\n| **합계** | **{sum(_g1.values()) / 1e6:.1f}M** | **{sum(_g7.values()) / 1e6:.1f}M** |"
        + "\n\n원 논문의 약 80M은 디코더 7층을 모두 센 값에 가깝고, 실제로 계산에 쓰이는 크기는 디코더 1층 기준입니다."
        " ResNet-18 백본(약 11M)이 상당 부분을 차지합니다. 카메라를 늘려도 백본은 공유되므로 매개변수는 늘지 않고 토큰 수만 늘어납니다."
    )
    return



@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. 손실: 마스크된 L1 + β·KL

    $$\mathcal{L} = \underbrace{\operatorname{mean}\big(|\hat a - a| \cdot \lnot\text{pad}\big)}_{\text{L1 재구성}}
      + \beta \cdot \underbrace{\operatorname{mean}_B \sum_{j} -\tfrac12\big(1 + \log\sigma_j^2 - \mu_j^2 - \sigma_j^2\big)}_{D_\text{KL}(\mathcal N(\mu,\sigma^2)\,\|\,\mathcal N(0, I))}$$

    - L1은 패딩 자리를 0으로 만든 뒤 **전체 원소 수로 평균**합니다(LeRobot과 같음. 패딩이 많으면 손실이 약간 작아집니다).
    - KL은 잠재 차원에 대해 **합**, 배치에 대해 평균입니다. LeRobot 기본 `kl_weight` β = 10.
    - KL 항은 CVAE 인코더의 출력 분포를 표준정규분포 쪽으로 끌어당깁니다. 그래야 추론 때 `z = 0`(사전 분포의 평균)이
      디코더가 학습 중 보던 `z`와 비슷한 값이 됩니다. β가 너무 크면 `z`가 아무 정보도 담지 않게 됩니다(**사후 붕괴**, 6절에서 확인).
    """)
    return



@app.cell
def _(ACTION, ACTION_IS_PAD, F):
    from act_tour import loss_definitions

    _, _defs = loss_definitions.run(ACTION=ACTION, ACTION_IS_PAD=ACTION_IS_PAD, F=F)
    act_loss = _defs["act_loss"]
    kl_to_standard_normal = _defs["kl_to_standard_normal"]
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
    ## 5. 학습: KL 가중치 β 비교

    1절의 갈림길 과제로 ACT를 학습합니다. LeRobot 기본값 **β = 10** 과 작은 **β = 0.01** 을 기본으로 비교합니다.
    필요하면 다른 β도 선택하세요. β마다 모델을 새로 만들어 같은 seed로 학습합니다.

    | 항목 | 이 노트북 | LeRobot 기본 |
    | --- | --- | --- |
    | 옵티마이저 | AdamW, lr 1e-4, weight decay 1e-4 | AdamW, lr 1e-5 (백본도 1e-5), weight decay 1e-4 |
    | 기울기 클리핑 | 10 | 10 |
    | 배치 | 64 | 8 |
    | 백본 | 무작위 초기화 | ImageNet 사전 학습 |

    lr을 키운 것은 작은 모델·짧은 학습에서 빨리 수렴시키기 위해서입니다. 기준(RTX 3080 Ti)으로 3000 step에 모델당 약 1분이 걸렸습니다.
    CPU에서는 step당 약 0.4초라 기본값을 600 step으로 낮췄고, 이 경우 6~8절 결과가 GPU 결과와 다를 수 있습니다.
    학습한 가중치는 메모리에만 있습니다. 학습 로그는 버튼을 누를 때 `outputs/lerobot/<실행 ID>/train_log.json`에 저장합니다.
    """)
    return



@app.cell
def _(ON_GPU, mo):
    beta_select = mo.ui.multiselect(options=["10", "1", "0.1", "0.01", "0"], value=["10", "0.01"], label="KL 가중치 β")
    steps_slider = mo.ui.slider(200, 6000, step=200, value=3000 if ON_GPU else 600, label="학습 step", show_value=True)
    train_button = mo.ui.run_button(label="ACT 학습 시작")
    mo.hstack([beta_select, steps_slider, train_button], justify="start", gap=2)
    return beta_select, steps_slider, train_button



@app.cell
def _(
    ACT,
    DEVICE,
    RUN_DIR,
    TOY_CFG,
    act_loss,
    asdict,
    beta_select,
    json,
    make_batch,
    mo,
    np,
    replace,
    steps_slider,
    time,
    torch,
    train_button,
):
    mo.stop(not train_button.value, mo.md("β와 step을 고르고 **ACT 학습 시작**을 누르세요. 6·8절은 학습 후에 실행됩니다."))
    mo.stop(not beta_select.value, mo.md("β를 하나 이상 고르세요."))

    trained = {}
    train_logs = {}
    for _beta_s in sorted(beta_select.value, key=float, reverse=True):
        _beta = float(_beta_s)
        torch.manual_seed(0)
        _rng = np.random.default_rng(0)
        _cfg = replace(TOY_CFG, kl_weight=_beta)
        _model = ACT(_cfg).to(DEVICE)
        _opt = torch.optim.AdamW(_model.parameters(), lr=1e-4, weight_decay=1e-4)
        _log, _t0 = [], time.time()
        for _step in mo.status.progress_bar(range(steps_slider.value), title=f"β = {_beta_s} 학습"):
            _model.train()
            _loss, _info = act_loss(_model, make_batch(64, _rng), _beta)
            _opt.zero_grad()
            _loss.backward()
            torch.nn.utils.clip_grad_norm_(_model.parameters(), 10.0)
            _opt.step()
            if _step % 50 == 0 or _step == steps_slider.value - 1:
                _log.append({"step": _step, **_info})
        _model.eval()
        trained[_beta_s] = _model
        train_logs[_beta_s] = {"seconds": round(time.time() - _t0, 1), "log": _log}

    RUN_DIR.mkdir(parents=True, exist_ok=True)
    (RUN_DIR / "train_log.json").write_text(json.dumps({
        "device": str(DEVICE), "steps": steps_slider.value, "batch_size": 64, "lr": 1e-4, "seed": 0,
        "config": asdict(TOY_CFG), "runs": train_logs}, ensure_ascii=False, indent=1))
    mo.md("학습 완료: " + ", ".join(f"β={k} {v['seconds']}초" for k, v in train_logs.items())
          + f"  \n로그: `{(RUN_DIR / 'train_log.json').relative_to(RUN_DIR.parents[2])}`")
    return train_logs, trained



@app.cell
def _(plt, train_logs):
    _fig, _axes = plt.subplots(1, 2, figsize=(11, 3.4))
    for _k, _v in train_logs.items():
        _s = [r["step"] for r in _v["log"]]
        _axes[0].plot(_s, [r["l1"] for r in _v["log"]], label=f"β={_k}")
        _axes[1].plot(_s, [r.get("kl", 0) for r in _v["log"]], label=f"β={_k}")
    _axes[0].set(title="L1 재구성 손실 (정규화 단위)", xlabel="step", yscale="log")
    _axes[1].set(title="KL (잠재 차원 합)", xlabel="step", yscale="symlog", ylim=(0, None))
    for _ax in _axes:
        _ax.legend(); _ax.grid(alpha=0.3)
    _fig.tight_layout()
    _fig
    return



@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **읽는 법**: β=10에서 KL이 0 근처로 떨어지면 CVAE 인코더의 출력이 사전 분포 N(0, I)와 거의 같다는 뜻입니다.
    즉 `z`가 정답 청크에 대한 정보를 담지 않습니다. 그러면 디코더는 갈림 지점에서 왼쪽/오른쪽을 구별할 단서가 없고,
    L1 손실도 그만큼 높게 남습니다. 작은 β에서는 KL을 조금 지불하고 L1을 더 낮춥니다.

    ## 6. 추론: `z = 0`과 잠재 변수에 담긴 것

    ### 6-1. 한 번의 forward로 청크 전체

    추론은 `model.eval()` 상태에서 관측만 넣습니다. CVAE 인코더는 쓰지 않고 `z = 0`입니다.
    """)
    return



@app.cell
def _(DEVICE, obs_batch, torch, unnormalize_action):
    @torch.no_grad()
    def predict_chunk(model, pos, obstacle, goal, z=None):
        """관측 → 위치 단위 행동 청크 (N, k, 2). z=None이면 eval 모드의 기본 동작(z=0)."""
        model.eval()
        zt = None if z is None else torch.as_tensor(z, dtype=torch.float32, device=DEVICE)
        actions, _ = model(obs_batch(pos, obstacle, goal), z=zt)
        return unnormalize_action(actions).cpu().numpy()
    return (predict_chunk,)



@app.cell
def _(
    DEVICE,
    TEST_SCENES,
    draw_scene,
    fork_point,
    np,
    plt,
    predict_chunk,
    test_pairs,
    test_scene_list,
    time,
    torch,
    trained,
):
    _n = len(trained)
    _fig, _axes = plt.subplots(_n, 4, figsize=(13, 3.4 * _n), squeeze=False)
    for _r, (_k, _model) in enumerate(trained.items()):
        for _j in range(4):
            _ax = _axes[_r, _j]
            _scene = test_scene_list[_j]
            _fork = fork_point(_scene[1])
            draw_scene(_ax, _scene, agent=_fork, title=f"β={_k}, 장면 {_j}: 갈림 지점에서 예측")
            for _ep, _c in zip(test_pairs[_j], ["cyan", "gold"]):
                _ax.plot(*_ep["action"][_ep["fork_t"]:_ep["fork_t"] + 20].T, "--", c=_c, lw=0.8)
            _chunk = predict_chunk(_model, _fork[None], _scene[1][None], _scene[2][None])[0]
            _ax.plot(*_chunk.T, ".-", c="red", ms=4, lw=1.2)
    _fig.suptitle("z = 0 으로 예측한 청크 20개(빨강) vs 전문가 두 갈래(점선)")
    _fig.tight_layout()

    # forward 1회 시간 (배치 1)
    _m = next(iter(trained.values()))
    _args = (TEST_SCENES["start"][:1], TEST_SCENES["obstacle"][:1], TEST_SCENES["goal"][:1])
    predict_chunk(_m, *_args)
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    _t0 = time.perf_counter()
    for _ in range(20):
        predict_chunk(_m, *_args)
    print(f"배치 1 추론 1회: {(time.perf_counter() - _t0) / 20 * 1000:.1f} ms ({DEVICE}, 청크 20개 전체, 이미지 렌더링 포함)")
    _fig
    return



@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 6-2. `z`에는 무엇이 담겼나

    학습된 CVAE 인코더에 **평가 장면의 왼쪽·오른쪽 시연**(갈림 지점 프레임의 청크)을 넣어 `μ`를 얻습니다.
    `z`가 갈래 정보를 담는다면 왼쪽/오른쪽 시연의 `μ`가 갈라져야 합니다.
    그다음 같은 관측에 `z`만 바꿔 넣어 디코더가 무엇을 하는지 봅니다. `z`를 직접 넣는 것은 원본에 없는 실험용 기능입니다.
    """)
    return



@app.cell
def _(
    ACTION_IS_PAD,
    CHUNK,
    DEVICE,
    OBS_STATE,
    STATS,
    chunk_at,
    np,
    test_pairs,
    torch,
    trained,
):
    @torch.no_grad()
    def encode_fork_chunks(model):
        """평가 장면 100개 × (왼쪽, 오른쪽) 시연의 갈림 지점 청크를 CVAE 인코더에 넣어 μ, σ, side를 돌려줍니다."""
        states, actions, pads, sides = [], [], [], []
        for pair in test_pairs:
            for ep in pair:
                a, p = chunk_at(ep, ep["fork_t"], CHUNK)
                states.append(ep["state"][ep["fork_t"]]); actions.append(a); pads.append(p); sides.append(ep["side"])
        _t = lambda x: torch.tensor(np.array(x), dtype=torch.float32, device=DEVICE)
        s = (_t(states) - STATS["state_mean"]) / STATS["state_std"]
        a = (_t(actions) - STATS["action_mean"]) / STATS["action_std"]
        model.eval()
        mu, lv = model.vae_encoder(s, a, torch.tensor(np.array(pads), device=DEVICE))
        return mu.cpu().numpy(), (lv / 2).exp().cpu().numpy(), np.array(sides)


    latent_stats = {}
    for _k, _model in trained.items():
        if _model.vae_encoder is None:
            continue
        _mu, _sigma, _side = encode_fork_chunks(_model)
        _var = _mu.var(0)
        _best = int(np.argmax(np.abs([np.corrcoef(_mu[:, j], _side)[0, 1] if _mu[:, j].std() > 1e-8 else 0
                                       for j in range(_mu.shape[1])])))
        latent_stats[_k] = dict(mu=_mu, sigma=_sigma, side=_side, best_dim=_best,
                                corr=float(np.corrcoef(_mu[:, _best], _side)[0, 1]) if _mu[:, _best].std() > 1e-8 else 0.0)
        print(f"β={_k:>4}: μ 분산이 0.01보다 큰 차원 {int((_var > 0.01).sum())}/{len(_var)}개, 평균 σ {_sigma.mean():.3f}, "
              f"갈래와 가장 상관 큰 차원 z[{_best}] 상관계수 {latent_stats[_k]['corr']:+.2f}, 그 차원의 μ 표준편차 {_mu[:, _best].std():.4f}")
    print("상관계수는 크기와 무관합니다. μ 표준편차가 σ(≈1)보다 훨씬 작으면 샘플링 잡음에 묻혀 디코더가 그 차이를 쓸 수 없습니다.")
    return (latent_stats,)



@app.cell
def _(
    draw_scene,
    fork_point,
    latent_stats,
    np,
    plt,
    predict_chunk,
    test_scene_list,
    trained,
):
    _n = len(latent_stats)
    _fig, _axes = plt.subplots(_n, 2, figsize=(10, 4.2 * max(_n, 1)), squeeze=False)
    _scene = test_scene_list[0]
    _fork = fork_point(_scene[1])
    for _r, (_k, _st) in enumerate(latent_stats.items()):
        _mu, _side, _d = _st["mu"], _st["side"], _st["best_dim"]
        _other = int(np.argsort(_mu.var(0))[-1] if np.argsort(_mu.var(0))[-1] != _d else np.argsort(_mu.var(0))[-2])
        _ax = _axes[_r, 0]
        _ax.scatter(_mu[_side < 0, _d], _mu[_side < 0, _other], s=10, c="tab:cyan", label="왼쪽 시연")
        _ax.scatter(_mu[_side > 0, _d], _mu[_side > 0, _other], s=10, c="tab:orange", label="오른쪽 시연")
        _ax.plot(0, 0, "k*", ms=12, label="z = 0")
        _lim = max(1.6, float(np.abs(_mu[:, [_d, _other]]).max()) * 1.1)
        _ax.add_patch(plt.Circle((0, 0), 1.0, fill=False, ls="--", ec="0.5"))  # 사전 분포의 1σ
        _ax.set(title=f"β={_k}: 시연 200개의 μ (z[{_d}], z[{_other}]), 점선=N(0,I)의 1σ",
                xlabel=f"z[{_d}]", ylabel=f"z[{_other}]", xlim=(-_lim, _lim), ylim=(-_lim, _lim), aspect="equal")
        _ax.legend(fontsize=8); _ax.grid(alpha=0.3)

        _ax = _axes[_r, 1]
        draw_scene(_ax, _scene, agent=_fork, title=f"β={_k}: 같은 관측, z만 바꿔 디코딩")
        _model = trained[_k]
        _lat = _model.cfg.latent_dim
        _zs = {"z = μ(왼쪽 평균)": (_mu[_side < 0].mean(0), "cyan"), "z = μ(오른쪽 평균)": (_mu[_side > 0].mean(0), "orange"),
               "z = 0": (np.zeros(_lat), "red")}
        for _name, (_z, _c) in _zs.items():
            _ch = predict_chunk(_model, _fork[None], _scene[1][None], _scene[2][None], z=_z[None])[0]
            _ax.plot(*_ch.T, ".-", c=_c, ms=4, lw=1.4, label=_name)
        _prior = np.random.default_rng(0).standard_normal((8, _lat))
        _chs = predict_chunk(_model, np.repeat(_fork[None], 8, 0), np.repeat(_scene[1][None], 8, 0),
                             np.repeat(_scene[2][None], 8, 0), z=_prior)
        for _i, _ch in enumerate(_chs):
            _ax.plot(*_ch.T, "-", c="w", lw=0.6, alpha=0.7, label="z ~ N(0, I) 8개" if _i == 0 else None)
        _ax.legend(fontsize=7, loc="lower left")
    _fig.tight_layout()
    _fig
    return



@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **확인할 것** (기준 실행: RTX 3080 Ti, 3000 step, seed 0. 직접 실행한 결과로 판단하세요)

    | β | μ 분산 > 0.01인 차원 | 평균 σ | 갈래를 가르는 차원의 μ 범위 | 최종 KL |
    | --- | --- | --- | --- | --- |
    | 10 | 0 / 8 | 1.000 | 약 ±0.001 | 약 0.0001 |
    | 0.01 | 1 / 8 | 0.89 | 약 −1.1(왼쪽) / +1.35(오른쪽) | 약 0.45 |

    - **β = 10**: μ가 원점에 모이고 평균 σ ≈ 1이면 **사후 붕괴**입니다. 왼쪽/오른쪽 μ가 미세하게 다르더라도
      σ ≈ 1의 샘플링 잡음에 묻혀 디코더는 이를 쓰지 못합니다. `z`를 무엇으로 바꿔도 같은 청크가 나옵니다.
      디코더는 갈래를 구별할 정보가 없어 두 정답의 **중간**을 내기 쉽습니다. act.md의 "CVAE가 다봉성을 해결한다"는 설명은
      `z`가 실제로 정보를 담을 때만 성립합니다.
    - **β = 0.01**: 한 차원이 왼쪽/오른쪽을 거의 완벽하게 가릅니다. `μ(왼쪽)`을 넣으면 왼쪽, `μ(오른쪽)`을 넣으면 오른쪽 청크가 나옵니다.
      그런데 **`z = 0`은 두 무리의 한가운데**에 있습니다. 추론 때 `z = 0`을 고정하면 여전히 "평균적인 스타일"을 고르는 셈이고,
      이 과제에서는 그것이 위험한 중간 경로일 수 있습니다. 사전 분포에서 뽑은 `z`(흰 선)는 갈래를 나눠 고르기도 합니다.
    - **β = 0**(선택 시): KL 제약이 없어 `μ`가 원점에서 멀리 흩어집니다. `z = 0`은 학습 중 본 적 없는 입력이 됩니다.

    ACT 논문이 CVAE를 넣은 주된 이유는 **사람 시연의 잡음·스타일 차이를 흡수해 학습을 안정시키는 것**이었고,
    추론 때 `z = 0`으로 결정론적 출력을 얻습니다. 서로 다른 갈래 자체를 고르는 능력은 Diffusion Policy·Flow Matching 계열이 더 강합니다(9절 비교표).

    ## 7. 시간 앙상블 (Temporal Ensembling)

    ### 7-1. 겹치는 예측 모으기

    매 step 청크를 새로 예측하면, 현재 시각 `t`에 대해 과거 여러 시점의 예측이 겹칩니다. act.md의 그림을 그대로 그려 봅니다(`k = 4`).
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
def _(deque, torch):
    from act_tour import ensemble_definitions

    _, _defs = ensemble_definitions.run(deque=deque, torch=torch)
    ACTTemporalEnsembler = _defs["ACTTemporalEnsembler"]
    DequeTemporalEnsembler = _defs["DequeTemporalEnsembler"]
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

    ## 8. 폐루프 평가: 청크를 어떻게 실행할까

    학습한 모델을 평가 장면 100개에서 실행합니다. 같은 모델로 실행 방식만 바꿉니다.

    | 방식 | 추론 주기 | 설명 |
    | --- | --- | --- |
    | 청크 전체 실행 | 20 step마다 | `n_action_steps = k`. 사이에는 관측을 보지 않음(open-loop) |
    | 청크 5개씩 | 5 step마다 | `n_action_steps = 5` |
    | 시간 앙상블 | 매 step | `n_action_steps = 1`, `temporal_ensemble_coeff = 0.01` |
    | 청크 전체 + `z ~ N(0, I)` | 20 step마다 | 에피소드마다 `z`를 하나 뽑아 고정 (원본에 없는 실험) |

    **외란**을 켜면 6 step째에 로봇을 x 방향으로 ±0.08 밀어냅니다. 관측을 자주 다시 보는 방식이 회복하는지 확인합니다.
    """)
    return



@app.cell
def _(mo):
    push_checkbox = mo.ui.checkbox(label="외란: 6 step째에 x로 ±0.08 밀기")
    eval_button = mo.ui.run_button(label="폐루프 평가 실행")
    mo.hstack([push_checkbox, eval_button], justify="start", gap=2)
    return eval_button, push_checkbox



@app.cell
def _(
    ACTTemporalEnsembler,
    RUN_DIR,
    TEST_SCENES,
    eval_button,
    json,
    mo,
    np,
    predict_chunk,
    push_checkbox,
    run_episodes,
    torch,
    trained,
):
    mo.stop(not eval_button.value, mo.md("버튼을 누르면 학습한 모델마다 4가지 방식으로 평가 장면 100개를 실행합니다."))

    def _chunk_policy(model, n_action_steps, z=None):
        cache = {}

        def policy(t, pos):
            if t % n_action_steps == 0:
                cache["chunk"] = predict_chunk(model, pos, TEST_SCENES["obstacle"], TEST_SCENES["goal"], z)
            return cache["chunk"][:, t % n_action_steps]
        return policy

    def _ensemble_policy(model, coeff):
        ens = ACTTemporalEnsembler(coeff, model.cfg.chunk_size)

        def policy(t, pos):
            chunk = torch.from_numpy(predict_chunk(model, pos, TEST_SCENES["obstacle"], TEST_SCENES["goal"]))
            return ens.update(chunk).numpy()
        return policy

    _n = len(TEST_SCENES["start"])
    _push = None
    if push_checkbox.value:
        _sign = np.random.default_rng(7).choice([-1.0, 1.0], _n)
        _push = (6, np.stack([0.08 * _sign, np.zeros(_n)], 1))

    eval_results = {}
    for _k, _model in trained.items():
        _k_chunk = _model.cfg.chunk_size
        _z = np.random.default_rng(123).standard_normal((_n, _model.cfg.latent_dim))
        _methods = {
            "청크 전체 실행": (_chunk_policy(_model, _k_chunk), _k_chunk),
            "청크 5개씩": (_chunk_policy(_model, 5), 5),
            "시간 앙상블": (_ensemble_policy(_model, 0.01), 1),
            "청크 전체 + z~N(0,I)": (_chunk_policy(_model, _k_chunk, _z), _k_chunk),
        }
        for _name, (_policy, _period) in _methods.items():
            _res = run_episodes(TEST_SCENES, _policy, push=_push)
            _jump = np.linalg.norm(np.diff(_res["action"], axis=1), axis=-1)  # (N, T-1)
            _valid = np.arange(_jump.shape[1])[None] < (_res["end_t"][:, None] - 1)
            eval_results[(_k, _name)] = dict(
                res=_res,
                success=float(np.mean(_res["status"] == 1)),
                collision=float(np.mean(_res["status"] == 2)),
                timeout=float(np.mean(_res["status"] == 0)),
                calls=float(np.mean(np.ceil(_res["end_t"] / _period))),
                max_jump=float(np.mean(np.where(_valid, _jump, 0).max(1))),
            )

    RUN_DIR.mkdir(parents=True, exist_ok=True)
    (RUN_DIR / ("eval_push.json" if push_checkbox.value else "eval.json")).write_text(json.dumps(
        {"push": bool(push_checkbox.value), "n_scenes": _n, "scene_seed": "10000+i",
         "results": {f"beta={k[0]} | {k[1]}": {m: v[m] for m in ["success", "collision", "timeout", "calls", "max_jump"]}
                     for k, v in eval_results.items()}}, ensure_ascii=False, indent=1))

    _rows = "\n".join(
        f"| {k[0]} | {k[1]} | {v['success']:.0%} | {v['collision']:.0%} | {v['timeout']:.0%} | {v['calls']:.1f} | {v['max_jump']:.3f} |"
        for k, v in eval_results.items())
    mo.md(
        f"외란 {'켜짐' if push_checkbox.value else '꺼짐'}, 평가 장면 100개 (seed 10000+i)\n\n"
        "| β | 실행 방식 | 성공 | 충돌 | 시간 초과 | 추론 횟수/에피소드 | 에피소드별 최대 행동 변화 평균 |\n"
        "| --- | --- | --- | --- | --- | --- | --- |\n" + _rows
        + "\n\n'최대 행동 변화'는 연속한 두 행동(목표 위치)의 거리 중 최댓값입니다. 전문가는 step당 0.03~0.05입니다."
          " 청크 경계에서 행동이 튀면 이 값이 커집니다."
    )
    return (eval_results,)



@app.cell
def _(eval_results, mo):
    _betas = list(dict.fromkeys(k[0] for k in eval_results))
    view_beta = mo.ui.dropdown(options=_betas, value=_betas[0], label="궤적을 볼 β")
    view_beta
    return (view_beta,)



@app.cell
def _(draw_scene, eval_results, np, plt, test_scene_list, view_beta):
    _names = [k[1] for k in eval_results if k[0] == view_beta.value]
    _colors = ["red", "orange", "lime", "white"]
    _fig, _axes = plt.subplots(2, 4, figsize=(13, 7.2))
    for _j, _ax in enumerate(_axes[0]):
        draw_scene(_ax, test_scene_list[_j], title=f"β={view_beta.value}, 평가 장면 {_j}")
        for _name, _c in zip(_names, _colors):
            _r = eval_results[(view_beta.value, _name)]["res"]
            _e = _r["end_t"][_j]
            _ax.plot(*_r["traj"][_j, : _e + 1].T, "-", c=_c, lw=1.3, label=_name)
            if _r["status"][_j] == 2:
                _ax.plot(*_r["traj"][_j, _e], "x", c=_c, ms=9, mew=2)
    _axes[0, 0].legend(fontsize=7, loc="lower left")

    # 아래: 장면 0~3에서 행동 x 좌표의 시간 변화. 청크 경계(20 step)에서 튀는지 봅니다.
    for _j, _ax in enumerate(_axes[1]):
        for _name, _c in zip(_names, ["tab:red", "tab:orange", "tab:green", "0.4"]):
            _r = eval_results[(view_beta.value, _name)]["res"]
            _e = _r["end_t"][_j]
            _ax.plot(np.arange(_e), _r["action"][_j, :_e, 0], "-", c=_c, lw=1.2, label=_name)
        _ax.axvline(20, c="k", ls=":", lw=0.8)
        _ax.axvline(40, c="k", ls=":", lw=0.8)
        _ax.set(title=f"장면 {_j}: 행동 x 좌표", xlabel="step")
        _ax.grid(alpha=0.3)
    _fig.tight_layout()
    _fig
    return



@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **기준 실행 결과** (RTX 3080 Ti, 3000 step, 외란 없음, 학습 seed 0으로 **두 번** 실행. 성공률 "1회차 / 2회차")

    | β | 청크 전체 실행 | 청크 5개씩 | 시간 앙상블 | 청크 전체 + z~N(0,I) |
    | --- | --- | --- | --- | --- |
    | 10 | 1% / 1% | 67% / 76% | 8% / 7% | 1% / 0% |
    | 0.01 | 38% / 44% | 75% / 88% | **38% / 72%** | 89% / 89% |

    에피소드당 추론 횟수는 청크 전체 약 1~2회, 5개씩 약 4~5회, 시간 앙상블 약 12~21회였습니다.
    seed가 같아도 GPU 연산의 비결정성 때문에 학습된 모델이 조금씩 달라지고, **갈림 지점처럼 결과가 갈래 선택에 달린 과제에서는
    그 차이가 성공률 수십 %p로 커질 수 있습니다**(β=0.01 시간 앙상블). 일관된 결론은 β=10의 청크 전체 실행이 거의 전부 충돌한다는 것,
    β=0.01에서 `z ~ N(0, I)`가 가장 안정적이라는 것 정도입니다. CPU 200 step으로 줄인 실행에서는 순위가 또 달랐습니다
    (예: β=10 시간 앙상블 83%). 짧은 학습이나 한 번의 실행으로 방식의 우열을 판단하지 마세요.

    **해석 가이드** (숫자는 실행마다 다를 수 있습니다. 표의 값으로 판단하세요)

    - **β = 10**에서 `z`가 붕괴했다면, 청크 전체 실행은 갈림 지점에서 두 갈래의 중간으로 가 충돌하기 쉽습니다.
      관측을 더 자주 보는 방식(청크 5개씩, 시간 앙상블)은 약간 치우친 위치에서 다시 예측하면서 한쪽으로 기울 수 있어 결과가 달라집니다.
    - **β = 0.01**에서 `z = 0`은 두 무리의 중간이라 청크 전체 실행이 여전히 자주 충돌합니다.
      `z ~ N(0, I)`는 에피소드마다 갈래를 하나 "고르는" 효과를 내서 성공률이 크게 오릅니다. 원본 ACT의 추론 방식은 아닙니다.
    - **시간 앙상블**은 매 step 추론하므로 추론 횟수가 가장 많고, 행동 x 그래프가 청크 경계(점선)에서 튀지 않고 매끄러워야 합니다.
      반대로 갈래가 서로 다른 예측을 평균하면 **두 갈래의 중간**이 될 수 있다는 점도 기억하세요. 평균은 다봉성에 약합니다.
    - **외란**을 켜면 청크 전체 실행은 다음 청크(20 step)까지 밀린 위치를 반영하지 못합니다. 자주 다시 보는 방식이 유리한지 비교하세요.

    한 번의 학습 seed와 장면 100개의 결과입니다. 방식 간 차이가 몇 %p라면 seed를 바꿔 반복해야 결론을 낼 수 있습니다.

    ## 9. 정리

    ### 9-1. ACT vs Diffusion Policy vs SmolVLA (act.md 5절)

    | 비교 항목 | **ACT** | **Diffusion Policy** | **SmolVLA** |
    | --- | --- | --- | --- |
    | 생성 원리 | CVAE + DETR, forward 1회 | DDPM/DDIM 반복 디노이징 | Flow Matching (Euler 10 step) |
    | 시각·언어 백본 | ResNet-18 (언어 없음) | ResNet / ViT | SmolVLM2 (사전 학습 VLM) |
    | 다봉성 표현 | 잠재 변수 `z` (추론 때 보통 `z=0`) | 노이즈에서 출발하는 반복 샘플링 | 노이즈에서 출발하는 ODE 적분 |
    | 매개변수 | 51.6M (디코더 1층, 3-6절에서 측정) | 수십~수백M | 약 450M |
    | 이 노트북에서 확인한 것 | 6절: β가 크면 `z` 붕괴, `z=0`은 갈래 사이 | — | [smolvla_tour.py](smolvla_tour.py) |

    ### 9-2. act.md 설명·코드와 LeRobot 원본의 차이

    | 항목 | act.md | LeRobot (`e6ffde6`) | 확인한 절 |
    | --- | --- | --- | --- |
    | 디코더 층 수 | 7 | 1 (원 저자 코드에서 실제로 쓰인 층 수) | 3-5, 3-6 |
    | 매개변수 | 약 80M | 51.6M (디코더 7층이면 83.9M) | 3-6 |
    | CVAE 인코더 위치 임베딩 | 학습형 `randn` | 고정 1D sin/cos | 3-1, 3-4 |
    | 이미지 위치 임베딩 | 없음 | 고정 2D sin/cos | 3-1 |
    | 위치 임베딩 적용 | 입력에 더함 / `nn.Transformer` | 매 층 q·k에만 더함 (DETR) | 3-3 |
    | 디코더 입력 | `query_embed`를 `tgt`로 | 0 입력 + 매 층 쿼리 위치 | 3-3 |
    | 패딩 처리 | 없음 | CVAE `key_padding_mask`, L1 마스크 | 2, 3-4, 4 |
    | 앙상블 가중치 | 최신 예측 `w_0` (방식 A) | 가장 오래된 예측 `w_0` | 7 |
    | 앙상블 이름 | 지수 이동 평균 | 겹치는 예측 전체의 지수 가중 평균 | 7 |
    | 시간 앙상블 사용 조건 | 언급 없음 | `n_action_steps = 1` 필수 | 7, 8 |

    ### 9-3. 이 노트북의 단순화

    - 카메라 1대, 64×64 합성 이미지, 무작위 초기화 백본. 실제 ALOHA·SO-101 성능 재현이 아닙니다.
    - 정규화는 데이터셋 평균·표준편차로 직접 처리했고, LeRobot의 전처리·후처리 파이프라인과 Hub 저장 기능은 생략했습니다.
    - `forward(z=...)`, `z ~ N(0, I)` 실행 방식은 실험용으로 추가한 것입니다.

    ### 9-4. 연습 문제

    1. 5절에서 β = 1, 0.1을 추가로 학습하고 6-2절의 "μ 분산이 큰 차원 수"와 8절 성공률이 β에 따라 어떻게 변하는지 표로 정리하세요.
    2. `expert_episode`에서 같은 장면의 시연을 한 갈래만 남기면(`side`를 장면에 따라 결정) 1절 MLP와 8절 결과가 어떻게 바뀔까요?
       다봉성이 사라지면 CVAE가 여전히 필요한지 생각해 보세요.
    3. 7절 슬라이더로 m을 음수(최신 예측 중시)로 바꾸고 8절 시간 앙상블 평가를 다시 실행하세요. 외란이 있을 때 차이가 커지나요?
    4. `TOY_CFG`의 `chunk_size`를 10과 40으로 바꿔 학습하고, 청크 전체 실행의 성공률과 최대 행동 변화가 어떻게 달라지는지 비교하세요.
    5. 3-3절의 확인을 응용해, 학습된 모델에서 `decoder_pos_embed`를 모두 같은 벡터로 바꾸면 예측 청크가 어떻게 되는지 확인하세요.
    """)
    return


if __name__ == "__main__":
    app.run()
