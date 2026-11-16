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
    # SmolVLA 보충 실습: 파인튜닝과 폐루프 평가

    [SmolVLA 구조 해부](smolvla_tour.py)의 기존 9·10절을 분리했습니다.
    2D 점 이동 과제로 기본 정책, 파인튜닝 정책, 지시문 및 노이즈의 영향을 비교합니다.

    실행 순서: 정책 로드 → 기본 모델 평가 → 파인튜닝 → 학습 후 평가 → 지시문·노이즈 비교.
    `notebooks/lerobot`에서 `uv run marimo edit smolvla_training.py`로 엽니다.
    모델 로드는 약 3 GB 다운로드가 필요할 수 있습니다. 학습에는 GPU를 권장합니다.
    각 다운로드·학습·반복 평가는 버튼으로만 시작합니다. CPU 기본값은 20 step으로 구조 확인용입니다.
    RTX 3080 Ti의 과거 측정은 1500 step 약 7분·약 7 GB였으며 현재 환경의 보장값은 아닙니다.

    모델·전처리는 원래 노트북의 이름 있는 정의 셀을 재사용하고, 장면·시연·환경은 이 노트북에서 정의합니다. 원래 앱을 열거나 실행할 필요는 없습니다.
    학습 가중치는 메모리에만 있으며 커널 재시작 또는 정책 재로드 시 사라집니다.
    로그는 저장소 루트 `outputs/lerobot/<실행 ID>/`에 저장합니다. 2D 결과로 실물 로봇 성능을 판단하지 마세요.

    원본: [Hugging Face LeRobot SmolVLA](https://github.com/huggingface/lerobot/tree/e6ffde60312365c5452c46189a26872483943154/src/lerobot/policies/smolvla)
    및 같은 리비전의 `policies/common/flow_matching.py`, `policies/common/vla_utils.py`, `inference/execution.py`.
    고정 리비전 `e6ffde60312365c5452c46189a26872483943154`, Apache License 2.0,
    Copyright 2025–2026 The HuggingFace Inc. team. 원본을 축소·수정한 실습입니다.
    """)
    return


@app.cell
def _():
    import copy
    import json
    import math
    import time
    from datetime import datetime
    from zoneinfo import ZoneInfo
    from pathlib import Path

    import matplotlib.font_manager as font_manager
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    import torch.nn.functional as F
    from torch import nn

    # 그림의 한글 표시용 글꼴: 설치된 것 중 첫 번째를 사용합니다.
    _installed = {f.name for f in font_manager.fontManager.ttflist}
    for _name in ["Noto Sans CJK KR", "NanumGothic", "Malgun Gothic", "AppleGothic", "Noto Sans Mono CJK KR"]:
        if _name in _installed:
            plt.rcParams["font.family"] = _name
            break
    plt.rcParams["axes.unicode_minus"] = False
    return (
        F,
        Path,
        ZoneInfo,
        copy,
        datetime,
        json,
        math,
        nn,
        np,
        plt,
        time,
        torch,
    )


@app.cell
def _(Path, torch):
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ON_GPU = DEVICE.type == "cuda"
    REPO_ROOT = Path(__file__).resolve().parents[2]

    VLM_ID = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
    VLM_REV = "4cde9d208e19519c36917aa170fffc7e381aa0ad"
    SMOLVLA_ID = "lerobot/smolvla_base"
    SMOLVLA_REV = "5e8d12a6e2975b0e5e5fce7c8caf47c371d257b6"

    # 평가의 수치 비교를 위해 TF32 행렬곱을 끕니다. 학습 셀은 속도를 위해 잠시 켭니다.
    torch.backends.cuda.matmul.allow_tf32 = False
    print(f"device={DEVICE}" + ("" if ON_GPU else "  (CPU: 학습은 구조 확인용으로만 매우 짧게 실행됩니다)"))
    return DEVICE, ON_GPU, REPO_ROOT, SMOLVLA_ID, SMOLVLA_REV, VLM_ID, VLM_REV


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. 실습 과제: "빨간 점으로 가"

    실제 로봇 대신 2D 평면 과제를 씁니다. SmolVLA의 입력·출력 형식은 그대로 유지합니다.

    | SmolVLA 입력/출력 | 실제 로봇 | 이 실습 |
    | --- | --- | --- |
    | `observation.images.*` | 카메라 2~3대 | 위에서 본 128×128 이미지 1장 (빨간 점, 파란 점) |
    | `task` | "pick up the cube" | `"go to the red dot"`, `"move to the blue one"` 등 |
    | `observation.state` | 관절각 6개 | 에이전트 위치 `(x, y)` — 이미지에는 그리지 않음 |
    | `action` | 다음 50스텝 관절 목표값 | 다음 50스텝 목표 위치 `[50, 2]` |

    어느 점으로 갈지는 **지시문을 읽어야만** 알 수 있고, 점의 위치는 **이미지를 봐야만** 알 수 있습니다.
    좌표는 `[-1, 1]`이며 이미지와 같게 `x`는 오른쪽, `y`는 **아래쪽**으로 증가합니다.
    시연(정답 행동)은 한 스텝에 최대 0.06씩 목표 점으로 직진한 뒤 그 자리에 머무는 궤적입니다.

    SmolVLA는 상태·행동을 `max_state_dim = max_action_dim = 32`로 **0을 채워 늘려** 씁니다.
    여기서는 앞 2칸만 쓰고, 손실도 앞 2칸에서만 계산합니다(원본 `forward`의 `losses[:, :, :action_dim]`).
    좌표가 이미 `[-1, 1]` 근처이므로 원본의 평균·표준편차 정규화는 생략했습니다.
    """)
    return


@app.cell
def scene_definitions(torch):
    IMG_RES = 128
    DOT_RADIUS = 0.09
    EXPERT_SPEED = 0.06  # 시연 궤적의 스텝당 이동 거리
    ENV_MAX_STEP = 0.08  # 환경이 한 스텝에 허용하는 최대 이동 거리
    SUCCESS_RADIUS = 0.1
    CHUNK_SIZE = 50  # smolvla_base와 같은 행동 청크 길이
    N_ACTION_STEPS = 10  # 청크 중 실제로 실행하고 다시 추론할 스텝 수 (원본 기본값은 50)
    MAX_DIM = 32  # max_state_dim = max_action_dim

    DOT_COLORS = {"red": (0.9, 0.15, 0.15), "blue": (0.15, 0.3, 0.9)}
    INSTRUCTIONS = {
        "red": ["go to the red dot", "move to the red one", "reach the red circle"],
        "blue": ["go to the blue dot", "move to the blue one", "reach the blue circle"],
    }

    def sample_scenes(n, generator=None, min_gap=0.5):
        """빨간 점·파란 점·에이전트 위치를 뽑습니다. 두 점은 min_gap 이상 떨어뜨립니다."""
        red = torch.rand(n, 2, generator=generator) * 1.6 - 0.8
        blue = torch.rand(n, 2, generator=generator) * 1.6 - 0.8
        bad = (red - blue).norm(dim=1) < min_gap
        while bad.any():
            blue[bad] = torch.rand(int(bad.sum()), 2, generator=generator) * 1.6 - 0.8
            bad = (red - blue).norm(dim=1) < min_gap
        agent = torch.rand(n, 2, generator=generator) * 1.8 - 0.9
        return red, blue, agent

    def render_scenes(red, blue):
        """[n, 3, 128, 128] RGB 이미지(0~1). 에이전트는 그리지 않습니다."""
        n = red.shape[0]
        ys, xs = torch.meshgrid(torch.linspace(-1, 1, IMG_RES), torch.linspace(-1, 1, IMG_RES), indexing="ij")
        img = torch.full((n, 3, IMG_RES, IMG_RES), 0.85)
        for pos, color in [(red, DOT_COLORS["red"]), (blue, DOT_COLORS["blue"])]:
            dist = ((xs[None] - pos[:, 0, None, None]) ** 2 + (ys[None] - pos[:, 1, None, None]) ** 2).sqrt()
            inside = dist < DOT_RADIUS
            for ch in range(3):
                img[:, ch][inside] = color[ch]
        return img

    def expert_chunk(agent, goal, horizon=CHUNK_SIZE):
        """시연 행동 청크 [n, horizon, 2]: 목표까지 직진하는 다음 위치들."""
        out, pos = [], agent.clone()
        for _ in range(horizon):
            delta = goal - pos
            dist = delta.norm(dim=1, keepdim=True)
            pos = pos + torch.where(dist > EXPERT_SPEED, delta / dist.clamp_min(1e-6) * EXPERT_SPEED, delta)
            out.append(pos.clone())
        return torch.stack(out, 1)

    def env_step(pos, target_pos):
        """목표 위치 쪽으로 최대 ENV_MAX_STEP만큼 이동합니다."""
        delta = target_pos - pos
        dist = delta.norm(dim=1, keepdim=True)
        return pos + torch.where(dist > ENV_MAX_STEP, delta / dist.clamp_min(1e-6) * ENV_MAX_STEP, delta)

    return (
        CHUNK_SIZE,
        INSTRUCTIONS,
        MAX_DIM,
        N_ACTION_STEPS,
        SUCCESS_RADIUS,
        env_step,
        expert_chunk,
        render_scenes,
        sample_scenes,
    )


@app.cell
def _(INSTRUCTIONS, expert_chunk, plt, render_scenes, sample_scenes, torch):
    _g = torch.Generator().manual_seed(1)
    _red, _blue, _agent = sample_scenes(4, generator=_g)
    _imgs = render_scenes(_red, _blue)
    _fig, _axes = plt.subplots(1, 4, figsize=(12, 3.3))
    for _i, _ax in enumerate(_axes):
        _color = "red" if _i % 2 == 0 else "blue"
        _goal = _red[_i : _i + 1] if _color == "red" else _blue[_i : _i + 1]
        _chunk = expert_chunk(_agent[_i : _i + 1], _goal)[0]
        _ax.imshow(_imgs[_i].permute(1, 2, 0), extent=[-1, 1, 1, -1])
        _ax.plot(_chunk[:, 0], _chunk[:, 1], ".-", color="k", ms=3, lw=0.8)
        _ax.plot(*_agent[_i], "g^", ms=9, label="state (이미지에 없음)")
        _ax.set_title(f'"{INSTRUCTIONS[_color][0]}"', fontsize=9)
        _ax.set_xticks([]), _ax.set_yticks([])
    _axes[0].legend(fontsize=7, loc="lower left")
    _fig.suptitle("관측(이미지 + 지시문 + 상태)과 정답 행동 청크 50스텝(검은 점)", fontsize=10)
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(F):
    from smolvla_tour import image_preprocessing

    _, _defs = image_preprocessing.run(F=F)
    resize_with_pad = _defs["resize_with_pad"]
    return (resize_with_pad,)


@app.cell
def _(F, copy, nn, torch):
    from smolvla_tour import expert_definitions

    _, _defs = expert_definitions.run(F=F, copy=copy, nn=nn, torch=torch)
    SmolVLMWithExpertModel = _defs["SmolVLMWithExpertModel"]
    return (SmolVLMWithExpertModel,)


@app.cell
def _(F, SmolVLMWithExpertModel, math, nn, torch):
    from smolvla_tour import flow_definitions

    _, _defs = flow_definitions.run(F=F, SmolVLMWithExpertModel=SmolVLMWithExpertModel, math=math, nn=nn, torch=torch)
    VLAFlowMatching = _defs["VLAFlowMatching"]
    make_att_2d_masks = _defs["make_att_2d_masks"]
    return (VLAFlowMatching,)


@app.cell
def _(mo):
    load_btn = mo.ui.run_button(label="SmolVLA 정책 불러오기")
    load_btn
    return (load_btn,)


@app.cell
def _(
    CHUNK_SIZE,
    DEVICE,
    SMOLVLA_ID,
    SMOLVLA_REV,
    VLAFlowMatching,
    VLM_ID,
    VLM_REV,
    load_btn,
    mo,
    torch,
):
    mo.stop(not load_btn.value, mo.md("정책 로드 버튼을 눌러 시작하세요."))
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    from transformers import AutoModelForImageTextToText, AutoTokenizer

    with mo.status.spinner("SmolVLA 정책 불러오는 중..."):
        tok = AutoTokenizer.from_pretrained(VLM_ID, revision=VLM_REV)
        _vlm = AutoModelForImageTextToText.from_pretrained(VLM_ID, revision=VLM_REV, dtype=torch.float32)
        policy = VLAFlowMatching(_vlm, chunk_size=CHUNK_SIZE)
        _path = hf_hub_download(SMOLVLA_ID, "model.safetensors", revision=SMOLVLA_REV)
        _state = {k.removeprefix("model."): v for k, v in load_file(_path).items()}
        print(policy.load_state_dict(_state, strict=True))
        policy.to(DEVICE).eval()
    return policy, tok


@app.cell
def _(
    CHUNK_SIZE,
    DEVICE,
    MAX_DIM,
    render_scenes,
    resize_with_pad,
    sample_scenes,
    tok,
    torch,
):
    def make_policy_inputs(images, texts, agent_pos):
        """원본 전처리와 같은 순서: 이미지 512 패딩 → [-1,1], 지시문 끝에 '\\n', 상태 32차원 0 채움."""
        pixels = resize_with_pad(images.to(DEVICE), 512, 512, pad_value=0) * 2 - 1
        toks = tok([t + "\n" for t in texts], padding="longest", padding_side="right", return_tensors="pt").to(DEVICE)
        state = torch.zeros(images.shape[0], MAX_DIM, device=DEVICE)
        state[:, : agent_pos.shape[1]] = agent_pos.to(DEVICE)
        img_masks = [torch.ones(images.shape[0], dtype=torch.bool, device=DEVICE)]
        return [pixels], img_masks, toks.input_ids, toks.attention_mask.bool(), state

    def pad_actions(actions):
        out = torch.zeros(actions.shape[0], actions.shape[1], MAX_DIM, device=DEVICE)
        out[:, :, : actions.shape[2]] = actions.to(DEVICE)
        return out

    _r, _b, _a = sample_scenes(2, generator=torch.Generator().manual_seed(3))
    demo_inputs = make_policy_inputs(render_scenes(_r, _b), ["go to the red dot", "move to the blue one"], _a)
    print("이미지:", tuple(demo_inputs[0][0].shape), " 지시문 토큰:", tuple(demo_inputs[2].shape),
          " 지시문 패딩 마스크:", demo_inputs[3].int().tolist())
    print("상태:", tuple(demo_inputs[4].shape), " 행동 청크:", (2, CHUNK_SIZE, MAX_DIM))
    return make_policy_inputs, pad_actions


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. 행동 전문가 파인튜닝

    `smolvla_base`는 실물 SO-100/101 데이터로 학습했으므로 우리 2D 과제는 처음 봅니다. 먼저 그대로 평가하고, 그다음 원본 기본 설정처럼
    **VLM은 고정하고 행동 전문가·투영층(약 1억 파라미터)만** 학습합니다(`train_expert_only=True`, `train_state_proj=True`).

    - 옵티마이저: AdamW, lr 1e-4, betas (0.9, 0.95), grad clip 10 (원본 `get_optimizer_preset`), warmup 후 cosine 감소.
    - 배치마다 장면·지시문을 새로 만들어 무한 데이터처럼 씁니다.
    - 학습 모델은 `ft_policy`로 따로 만듭니다. 고정된 VLM 모듈은 `policy`와 **공유**해 GPU 메모리를 아낍니다.

    평가는 폐루프입니다. 행동 청크 50개 중 앞 10개(`N_ACTION_STEPS`)를 실행하고 새 관측으로 다시 추론합니다.
    목표 점 반경 0.1 안에 들어가면 성공, 다른 점에 먼저 들어가면 오답입니다.
    """)
    return


@app.cell
def _(
    CHUNK_SIZE,
    DEVICE,
    INSTRUCTIONS,
    N_ACTION_STEPS,
    SUCCESS_RADIUS,
    env_step,
    make_policy_inputs,
    render_scenes,
    sample_scenes,
    torch,
):
    @torch.no_grad()
    def rollout(model, n=100, max_steps=60, seed=123, n_exec=N_ACTION_STEPS, texts=None, scenes=None, batch_size=25, noise_seed=456, target_colors=None):
        """폐루프 평가. 반환: 성공률, 오답률, 궤적 [n, T+1, 2], 장면"""
        if not 1 <= n_exec <= CHUNK_SIZE or max_steps < 1 or batch_size < 1:
            raise ValueError("n_exec는 1~청크 길이, max_steps와 batch_size는 양수여야 합니다.")
        model.eval()
        g = torch.Generator().manual_seed(seed)
        noise_g = torch.Generator(device=DEVICE).manual_seed(noise_seed)
        red, blue, agent = scenes if scenes is not None else sample_scenes(n, generator=g)
        n = red.shape[0]
        if texts is not None and target_colors is None:
            raise ValueError("사용자 지시문에는 target_colors로 정답 색을 지정하세요.")
        targets = target_colors if target_colors is not None else ["red" if i % 2 == 0 else "blue" for i in range(n)]
        if len(targets) != n or any(c not in ("red", "blue") for c in targets):
            raise ValueError("장면마다 red/blue 정답 색 하나가 필요합니다.")
        if texts is not None and len(texts) != n:
            raise ValueError("장면 수와 지시문 수가 다릅니다.")
        if texts is None:
            texts = [INSTRUCTIONS[c][0] for c in targets]
        goal = torch.stack([red[i] if c == "red" else blue[i] for i, c in enumerate(targets)])
        other = torch.stack([blue[i] if c == "red" else red[i] for i, c in enumerate(targets)])
        images = render_scenes(red, blue)
        pos = agent.clone()
        # reset 때 이미 반경 안에 있는 장면도 첫 도달로 판정합니다.
        done = (pos - goal).norm(dim=1) < SUCCESS_RADIUS
        wrong = ((pos - other).norm(dim=1) < SUCCESS_RADIUS) & ~done
        traj = [pos.clone()]
        steps = 0
        while steps < max_steps:
            # GPU 메모리를 아끼려고 batch_size개씩 나눠 추론합니다.
            chunk = torch.cat([
                model.sample_actions(
                    *make_policy_inputs(images[i : i + batch_size], texts[i : i + batch_size], pos[i : i + batch_size]),
                    noise=torch.randn(min(batch_size, n - i), CHUNK_SIZE, model.max_action_dim,
                                      device=DEVICE, generator=noise_g),
                )[:, :, :2].cpu()
                for i in range(0, n, batch_size)
            ])
            assert chunk.shape[1] == CHUNK_SIZE
            for k in range(min(n_exec, max_steps - steps)):
                frozen = (done | wrong)[:, None]
                pos = torch.where(frozen, pos, env_step(pos, chunk[:, k]))
                done |= ((pos - goal).norm(dim=1) < SUCCESS_RADIUS) & ~wrong
                wrong |= ((pos - other).norm(dim=1) < SUCCESS_RADIUS) & ~done
                traj.append(pos.clone())
                steps += 1
        return done.float().mean().item(), wrong.float().mean().item(), torch.stack(traj, 1), (red, blue, agent)

    return (rollout,)


@app.cell
def _(mo):
    base_eval_btn = mo.ui.run_button(label='기본 모델 100회 평가')
    base_eval_btn
    return (base_eval_btn,)


@app.cell
def _(base_eval_btn, mo, policy, rollout, time):
    mo.stop(not base_eval_btn.value, mo.md('기본 모델 평가 버튼을 누르면 100회 평가합니다.'))
    _t0 = time.time()
    with mo.status.spinner("smolvla_base 그대로 평가 중..."):
        base_success, base_wrong, _, _ = rollout(policy, n=100, seed=123, noise_seed=456)
    print(f"파인튜닝 전 smolvla_base: 성공 {base_success:.0%}, 오답 {base_wrong:.0%}  ({time.time() - _t0:.0f}초)")
    return base_success, base_wrong


@app.cell
def _(ON_GPU, mo):
    train_steps = mo.ui.number(start=10, stop=5000, step=10, value=1500 if ON_GPU else 20, label="학습 step")
    train_batch = mo.ui.number(start=2, stop=64, step=2, value=16 if ON_GPU else 2, label="배치 크기")
    train_btn = mo.ui.run_button(label="파인튜닝 시작")
    mo.hstack([train_steps, train_batch, train_btn], justify="start")
    return train_batch, train_btn, train_steps


@app.cell
def _(
    CHUNK_SIZE,
    DEVICE,
    INSTRUCTIONS,
    ON_GPU,
    REPO_ROOT,
    VLAFlowMatching,
    ZoneInfo,
    datetime,
    expert_chunk,
    json,
    make_policy_inputs,
    math,
    mo,
    pad_actions,
    policy,
    render_scenes,
    sample_scenes,
    time,
    torch,
    train_batch,
    train_btn,
    train_steps,
):
    mo.stop(not train_btn.value, mo.md("*설정을 확인하고 **파인튜닝 시작**을 누르세요. GPU(RTX 3080 Ti) 1500 step 기준 약 7분입니다.*"))
    torch.manual_seed(0)
    # 고정 VLM은 공유하고, 학습할 부분만 policy에서 복사한 새 모델
    ft_policy = VLAFlowMatching(policy.vlm_with_expert.vlm, chunk_size=CHUNK_SIZE)
    ft_policy.load_state_dict(policy.state_dict())
    ft_policy.to(DEVICE).train()

    _params = [p for p in ft_policy.parameters() if p.requires_grad]
    _steps, _bs = int(train_steps.value), int(train_batch.value)
    _warmup = min(100, _steps // 10)
    _opt = torch.optim.AdamW(_params, lr=1e-4, betas=(0.9, 0.95), eps=1e-8, weight_decay=1e-10)
    _sched = torch.optim.lr_scheduler.LambdaLR(
        _opt, lambda s: min(1.0, (s + 1) / max(1, _warmup)) * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * min(s, _steps) / _steps)))
    )
    torch.manual_seed(0)  # 시연 외 noise·시간 표집도 학습 시작마다 재현
    _g = torch.Generator().manual_seed(0)
    train_log = []
    train_run_dir = REPO_ROOT / "outputs" / "lerobot" / f"{datetime.now(ZoneInfo('Asia/Seoul')):%Y%m%d-%H%M%S}-smolvla-finetune"
    train_run_dir.mkdir(parents=True, exist_ok=True)
    _prev_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = True  # 학습 속도용
    _t0 = time.time()
    try:
        with mo.status.progress_bar(total=_steps, title="파인튜닝") as _bar:
            for _step in range(_steps):
                _red, _blue, _agent = sample_scenes(_bs, generator=_g)
                _is_red = torch.rand(_bs, generator=_g) < 0.5
                _goal = torch.where(_is_red[:, None], _red, _blue)
                _texts = [
                    INSTRUCTIONS["red" if r else "blue"][int(torch.randint(0, 3, (1,), generator=_g))] for r in _is_red
                ]
                _actions = pad_actions(expert_chunk(_agent, _goal))
                _losses = ft_policy(*make_policy_inputs(render_scenes(_red, _blue), _texts, _agent), _actions)
                _loss = _losses[:, :, :2].mean()  # 실제 행동 차원(2)만
                _opt.zero_grad(set_to_none=True)
                _loss.backward()
                _gn = torch.nn.utils.clip_grad_norm_(_params, 10.0)
                _opt.step()
                _sched.step()
                train_log.append({"step": _step, "loss": _loss.item(), "grad_norm": _gn.item(), "lr": _sched.get_last_lr()[0]})
                _bar.update(subtitle=f"loss {_loss.item():.3f}")
    finally:
        torch.backends.cuda.matmul.allow_tf32 = _prev_tf32
        _opt.zero_grad(set_to_none=True)  # 기울기·옵티마이저 상태를 GPU에서 해제
        del _opt, _sched
        if ON_GPU:
            torch.cuda.empty_cache()
    ft_policy.eval()
    _elapsed = time.time() - _t0
    with open(train_run_dir / "train_log.csv", "w") as _f:
        _f.write("step,loss,grad_norm,lr\n")
        _f.writelines(f"{r['step']},{r['loss']:.6f},{r['grad_norm']:.4f},{r['lr']:.3e}\n" for r in train_log)
    with open(train_run_dir / "train_config.json", "w") as _f:
        json.dump(
            {
                "steps": _steps,
                "train_seed": 0,
                "scene_eval_seed": 123,
                "noise_eval_seed": 456,
                "batch_size": _bs,
                "lr": 1e-4,
                "chunk_size": CHUNK_SIZE,
                "device": str(DEVICE),
                "peak_gpu_mem_gb": torch.cuda.max_memory_allocated() / 1e9 if ON_GPU else None,
                "elapsed_s": _elapsed,
            },
            _f,
            indent=2,
        )
    print(f"{_steps} step, {_elapsed:.0f}초. 로그: {train_run_dir.relative_to(REPO_ROOT)}")
    return ft_policy, train_log, train_run_dir


@app.cell
def _(np, plt, train_log):
    _loss = np.array([r["loss"] for r in train_log])
    _w = max(1, len(_loss) // 50)
    _smooth = np.convolve(_loss, np.ones(_w) / _w, mode="valid")
    _fig, _ax = plt.subplots(figsize=(7, 2.8))
    _ax.plot(_loss, alpha=0.3, lw=0.8, label="step 손실")
    _ax.plot(np.arange(len(_smooth)) + _w - 1, _smooth, lw=1.5, label=f"{_w} step 이동 평균")
    _ax.set_xlabel("step")
    _ax.set_ylabel("Flow Matching 손실")
    _ax.set_yscale("log")
    _ax.legend(fontsize=8)
    _ax.grid(alpha=0.3)
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    손실이 평평해지거나 다시 내려가는 구간이 있는지 관찰하세요. 곡선만으로 내부에서 타겟 선택이나 언어 조건을 배웠다고
    단정할 수 없습니다. 원인 가설을 확인하려면 같은 평가 장면에서 색별 성공률과 지시문 교체 결과를 함께 봐야 합니다.
    손실 값만으로는 정책 품질을 알 수 없으므로 반드시 폐루프로 평가합니다.

    ## 3. 평가: 성공률과 지시문 교체 실험
    """)
    return


@app.cell
def _(mo):
    ft_eval_btn = mo.ui.run_button(label='파인튜닝 모델 100회 평가')
    ft_eval_btn
    return (ft_eval_btn,)


@app.cell
def _(
    base_success,
    base_wrong,
    ft_eval_btn,
    ft_policy,
    json,
    mo,
    rollout,
    time,
    train_run_dir,
):
    mo.stop(not ft_eval_btn.value, mo.md('학습 후 평가 버튼을 눌러야 반복 평가합니다.'))
    _t0 = time.time()
    with mo.status.spinner("파인튜닝 모델 평가 중..."):
        ft_success, ft_wrong, ft_traj, ft_scenes = rollout(ft_policy, n=100, seed=123, noise_seed=456)
    _metrics = {
        "eval_episodes": 100,
        "scene_seed": 123,
        "noise_seed": 456,
        "batch_size": 25,
        "max_steps": 60,
        "base_success": base_success,
        "base_wrong": base_wrong,
        "finetuned_success": ft_success,
        "finetuned_wrong": ft_wrong,
    }
    with open(train_run_dir / "eval_metrics.json", "w") as _f:
        json.dump(_metrics, _f, indent=2)
    mo.md(f"""
    | 모델 | 성공 | 오답(다른 점 도달) | 시간 초과 |
    | --- | --- | --- | --- |
    | `smolvla_base` 그대로 | {base_success:.0%} | {base_wrong:.0%} | {1 - base_success - base_wrong:.0%} |
    | 파인튜닝 후 | **{ft_success:.0%}** | {ft_wrong:.0%} | {1 - ft_success - ft_wrong:.0%} |

    평가 100 에피소드(빨강·파랑 반반, 학습에 없던 장면), 걸린 시간 {time.time() - _t0:.0f}초. 결과는 `{train_run_dir.name}/eval_metrics.json`에 저장했습니다.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **지시문 교체 실험:** 같은 장면·같은 시작 위치에서 지시문만 바꿔 실행합니다. 같은 노이즈 seed로 문장만 바꾸며 조건 입력에 대한 민감도를 봅니다.
    궤적 차이는 지시를 올바르게 수행했다는 증거와 다릅니다. 빨강·파랑 목표 도달과 여러 seed 반복으로 함께 확인하세요.
    추가 지시문의 정답 색은 모델 입력이 아니라 평가자가 의도한 목표입니다. 임의 문장을 자동으로 색 이름에 매핑하지 않습니다.
    아래 입력란에 학습에 없던 표현(예: `head to the crimson spot`)을 넣어 일반화도 시험해 보세요.
    """)
    return


@app.cell
def _(mo):
    custom_instruction = mo.ui.text(value="head to the crimson spot", label="추가 지시문", full_width=True)
    custom_target = mo.ui.dropdown(["red", "blue"], value="red", label="추가 지시문의 정답 색 (평가용)")
    mo.vstack([custom_instruction, custom_target])
    return custom_instruction, custom_target


@app.cell
def _(mo):
    instruction_btn = mo.ui.run_button(label='같은 장면·노이즈로 지시문 비교')
    instruction_btn
    return (instruction_btn,)


@app.cell
def _(
    custom_instruction,
    custom_target,
    ft_policy,
    instruction_btn,
    mo,
    plt,
    render_scenes,
    rollout,
    sample_scenes,
    torch,
):
    mo.stop(not instruction_btn.value, mo.md('지시문을 고친 뒤 비교 버튼을 누르세요.'))
    _scenes = sample_scenes(4, generator=torch.Generator().manual_seed(7))
    _texts_all = ["go to the red dot", "go to the blue dot", custom_instruction.value]
    _colors = ["tab:red", "tab:blue", "tab:green"]
    _trajs = []
    for _text, _target in zip(_texts_all, ["red", "blue", custom_target.value]):
        _, _, _tr, _ = rollout(ft_policy, scenes=_scenes, texts=[_text] * 4, max_steps=40, noise_seed=456,
                                target_colors=[_target] * 4)
        _trajs.append(_tr)
    _imgs = render_scenes(_scenes[0], _scenes[1])
    _fig, _axes = plt.subplots(1, 4, figsize=(13, 3.5))
    for _i, _ax in enumerate(_axes):
        _ax.imshow(_imgs[_i].permute(1, 2, 0), extent=[-1, 1, 1, -1], alpha=0.8)
        for _tr, _c, _text in zip(_trajs, _colors, _texts_all):
            _ax.plot(_tr[_i, :, 0], _tr[_i, :, 1], "-", color=_c, lw=1.8, label=_text)
        _ax.plot(*_scenes[2][_i], "k^", ms=8)
        _ax.set_xlim(-1, 1), _ax.set_ylim(1, -1)
        _ax.set_xticks([]), _ax.set_yticks([])
    _axes[0].legend(fontsize=7, loc="lower left")
    _fig.suptitle("같은 장면, 다른 지시문 (▲ 시작 위치)", fontsize=10)
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **노이즈에 따른 행동 청크 분포:** 같은 관측에 노이즈만 바꿔 행동 청크 20개를 뽑습니다. 잘 학습된 모델이라면 샘플들이 지정한 목표 근처로 모일 것으로 예상합니다. 오차·분산을 실제로 확인하세요.
    지시문을 `"go to a dot"`처럼 모호하게 바꾸면 어떻게 될지(학습에 없던 상황) 예측해 보고 `_text`를 바꿔 확인해 보세요.
    """)
    return


@app.cell
def _(mo):
    noise_btn = mo.ui.run_button(label='노이즈 20개 청크 비교')
    noise_btn
    return (noise_btn,)


@app.cell
def _(
    DEVICE,
    ft_policy,
    make_policy_inputs,
    mo,
    noise_btn,
    plt,
    render_scenes,
    sample_scenes,
    torch,
):
    mo.stop(not noise_btn.value, mo.md('노이즈 비교 버튼을 눌러 실행하세요.'))
    _r, _b, _a = sample_scenes(1, generator=torch.Generator().manual_seed(11))
    _text = "go to the red dot"
    _n = 20
    _inputs = make_policy_inputs(render_scenes(_r, _b).expand(_n, -1, -1, -1), [_text] * _n, _a.expand(_n, -1))
    _g = torch.Generator(device=DEVICE).manual_seed(0)
    _noise = torch.randn(_n, ft_policy.chunk_size, 32, device=DEVICE, generator=_g)
    _chunks = ft_policy.sample_actions(*_inputs, noise=_noise)[:, :, :2].cpu()
    _fig, _ax = plt.subplots(figsize=(4, 4))
    _ax.imshow(render_scenes(_r, _b)[0].permute(1, 2, 0), extent=[-1, 1, 1, -1], alpha=0.8)
    for _c in _chunks:
        _ax.plot(_c[:, 0], _c[:, 1], "-", color="k", lw=0.7, alpha=0.6)
    _ax.plot(*_a[0], "g^", ms=9)
    _ax.set_xlim(-1, 1), _ax.set_ylim(1, -1)
    _ax.set_title(f'"{_text}" — 노이즈 20개의 행동 청크', fontsize=9)
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 이해 확인과 추가 실험

    - 손실은 속도 오차이고 성공률은 폐루프 목표 도달입니다. 같은 장면 seed 123·noise seed 456에서 따로 비교하세요.
    - 지시문 교체는 같은 장면·시작 위치·노이즈를 유지합니다. 정답 색은 평가자가 지정하며 여러 seed로 반복하세요.
    - 학습을 300 step으로 줄여 성공·오답·시간 초과를 비교하세요. 손실 평탄 구간을 특정 내부 개념 습득으로 단정하지 마세요.
    - `go to any dot` 시연을 20% 섞으려면 두 목표를 포함한 시연과 두 점 모두를 허용하는 별도 성공 판정이 필요합니다.
      두 갈래 분포는 데이터 20%나 노이즈 20개만으로 보장되지 않습니다.
    """)
    return


if __name__ == "__main__":
    app.run()
