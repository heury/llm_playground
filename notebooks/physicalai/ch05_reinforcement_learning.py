import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import json
    import sys
    import time
    from pathlib import Path

    _here = str(Path(__file__).resolve().parent)
    if _here not in sys.path:
        sys.path.insert(0, _here)

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    import physicalai_common as common_runtime
    import gymnasium as gym
    import mujoco
    import torch
    from gymnasium import spaces
    from stable_baselines3 import PPO
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv

    torch.set_num_threads(4)   # 작은 MLP는 CPU 스레드를 많이 쓰면 오히려 느려진다
    RUN_DIR = common_runtime.make_run_dir("ch05-rl")
    return (
        DummyVecEnv,
        Monitor,
        PPO,
        RUN_DIR,
        gym,
        json,
        mo,
        mujoco,
        np,
        common_runtime,
        plt,
        spaces,
        time,
    )


@app.cell
def _(mo):
    robot_assets_btn = mo.ui.run_button(label="SO-101 모델 준비 (없으면 다운로드)")
    robot_assets_btn
    return (robot_assets_btn,)


@app.cell
def _(common_runtime, mo, robot_assets_btn):
    _model_root = common_runtime.Path(common_runtime.os.environ.get(
        "MENAGERIE_DIR", common_runtime.DEFAULT_MENAGERIE_DIR))
    _cached = (_model_root / "robotstudio_so101").is_dir()
    mo.stop(not _cached and not robot_assets_btn.value, mo.md(
        "SO-101 모델이 없습니다. 위 준비 버튼을 누르면 수십 MB를 내려받습니다. "
        "다운로드 없이 설명을 읽거나, 기존 사본을 MENAGERIE_DIR로 지정할 수 있습니다."))
    if not _cached:
        common_runtime.menagerie_dir()
    pc = common_runtime
    return (pc,)


@app.cell(hide_code=True)
def _(mo):
    mo.vstack([
        mo.md(r"""
    # 5장. 강화학습으로 로봇팔 움직이기

    3·4장의 Policy는 사람이 단계를 적은 상태 기계였습니다. 강화학습(Reinforcement Learning, RL)은 **보상(reward)만 정해 주고**
    시행착오로 Policy를 학습합니다. 1장의 루프에 보상과 학습이 붙습니다.

    """),
        mo.mermaid(r"""
    flowchart LR
      E[환경 env] -- observation, reward --> A[Policy 신경망]
      A -- action --> E
      A -. 보상이 커지는 쪽으로 가중치 갱신 .-> A
    """),
        mo.md(r"""

    | 절 | 내용 |
    | --- | --- |
    | 5-1 | Gymnasium 환경 인터페이스: 점(point)이 목표로 가는 가장 단순한 연속 환경 |
    | 5-2 | PPO로 첫 학습: 랜덤 Policy와 비교, 학습 곡선, dense vs sparse 보상 |
    | 5-3 | SO-101을 Gymnasium 환경으로 감싸기: 손끝을 블록 위로 |
    | 5-4 | 보상 설계 비교: sparse / dense / shaped |

    학습은 [Stable-Baselines3](https://stable-baselines3.readthedocs.io/)의 PPO를 CPU에서 돌립니다.
    학습 셀은 버튼을 눌러야 실행됩니다. 결과(모델·곡선·지표)는 실행 폴더에 저장됩니다.
    """),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 이 장의 학습 경로

    셀 재실행·환경별 대안은 [README의 학습 경로와 재실행](README.md#학습-경로와-재실행)을 참고하세요.

    **목표**: 보상·학습 손실·성공률을 구분하고, 같은 seed의 랜덤 정책과 PPO를 비교합니다.
    **선수 지식**: 1장의 Episode·보상, 2장의 관절 제어, 기초 평균·신경망 가중치 갱신.
    **필수**: 5-1 환경 → 5-2 PPO 계산 읽기. 학습 버튼과 5-4 보상별 재학습은 선택입니다.
    GPU 없이 설명과 `PointReachEnv.step`을 추적할 수 있습니다. 학습하지 않은 결과를 학습 성과로 해석하지 마세요.

    **읽을 코드**: `PointReachEnv`, `ArmReachEnv`는 이 파일의 교육용 환경입니다.
    학습은 설치된 Stable-Baselines3 `PPO.train`, 평가는 이 파일의 `evaluate`입니다.
    5장의 목표는 손끝 도달이며, 3·6장의 집어 상자에 넣기와 성공률을 직접 비교할 수 없습니다.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5-1. Gymnasium 환경: `reset`과 `step`

    강화학습 라이브러리는 환경이 다음 약속을 지키면 어떤 문제든 학습할 수 있습니다.

    | 요소 | 의미 | PointReachEnv |
    | --- | --- | --- |
    | `observation_space` | 관측의 모양과 범위 | `[agent_x, agent_y, goal_x, goal_y]`, 각 −1~1 |
    | `action_space` | 행동의 모양과 범위 | `[dx, dy]`, 각 −1~1 (실제 이동은 0.1배) |
    | `reset(seed)` | 새 Episode 시작 → `(obs, info)` | Agent와 목표를 무작위로 |
    | `step(action)` | → `(obs, reward, terminated, truncated, info)` | 목표 5 cm 안이면 terminated, 100 스텝이면 truncated |

    `terminated`는 과제가 끝난 것(성공·실패), `truncated`는 시간 제한으로 잘린 것입니다. 둘을 구분해야
    "시간이 다 됐을 뿐 실패는 아닌" 상태의 가치를 올바르게 학습합니다.
    """)
    return


@app.cell
def _(gym, np, spaces):
    class PointReachEnv(gym.Env):
        """점 Agent가 2차원 평면에서 목표 지점까지 이동한다."""

        metadata = {"render_modes": []}

        def __init__(self, max_steps=100, reward_mode="dense"):
            super().__init__()
            self.max_steps = max_steps
            self.reward_mode = reward_mode
            self.observation_space = spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32)
            self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
            self.agent = np.zeros(2, dtype=np.float32)
            self.goal = np.zeros(2, dtype=np.float32)
            self.t = 0

        def _obs(self):
            return np.concatenate([self.agent, self.goal]).astype(np.float32)

        def reset(self, seed=None, options=None):
            super().reset(seed=seed)                     # self.np_random을 seed로 초기화
            self.agent = self.np_random.uniform(-1, 1, size=2).astype(np.float32)
            self.goal = self.np_random.uniform(-1, 1, size=2).astype(np.float32)
            self.t = 0
            return self._obs(), {}

        def step(self, action):
            action = np.clip(np.asarray(action, dtype=np.float32), -1, 1)
            self.agent = np.clip(self.agent + 0.1 * action, -1, 1)
            self.t += 1
            dist = float(np.linalg.norm(self.agent - self.goal))
            reached = dist < 0.05
            if self.reward_mode == "dense":
                reward = -dist + (10.0 if reached else 0.0)   # 매 스텝 거리만큼 벌점
            else:
                reward = 10.0 if reached else 0.0             # 도달했을 때만 보상
            return self._obs(), reward, reached, self.t >= self.max_steps, {"dist": dist, "success": reached}
    return (PointReachEnv,)


@app.cell
def _(PointReachEnv, mo):
    from gymnasium.utils.env_checker import check_env

    _env = PointReachEnv()
    check_env(_env, skip_render_check=True)      # 인터페이스 약속을 지키는지 검사
    _obs, _ = _env.reset(seed=0)
    _obs2, _r, _term, _trunc, _info = _env.step(_env.action_space.sample())
    mo.md(f"""
    `gymnasium.utils.env_checker.check_env` 통과.

    - `reset(seed=0)` → obs `{_obs.round(3)}`
    - `step(random)` → obs `{_obs2.round(3)}`, reward `{_r:+.3f}`, terminated `{_term}`, truncated `{_trunc}`, info `{_info}`
    """)
    return


@app.cell
def _(np):
    def evaluate(env, policy_fn, episodes=100, seed=0):
        """policy_fn(obs) -> action. (성공률, 평균 스텝, 평균 보상, 마지막 거리 평균)."""
        succ, steps, rewards, dists = 0, [], [], []
        for i in range(episodes):
            obs, _ = env.reset(seed=seed + i)
            total, done = 0.0, False
            while not done:
                obs, r, term, trunc, info = env.step(policy_fn(obs))
                total += r
                done = term or trunc
            succ += int(info["success"])
            steps.append(env.t)
            rewards.append(total)
            dists.append(info["dist"])
        return {"success_rate": succ / episodes, "steps": float(np.mean(steps)),
                "reward": float(np.mean(rewards)), "final_dist": float(np.mean(dists))}


    def rollout_path(env, policy_fn, seed):
        """Episode 하나의 Agent 경로 (그림용)."""
        obs, _ = env.reset(seed=seed)
        path, done = [env.agent.copy()], False
        while not done:
            obs, _, term, trunc, _ = env.step(policy_fn(obs))
            path.append(env.agent.copy())
            done = term or trunc
        return np.array(path), env.goal.copy()
    return evaluate, rollout_path


@app.cell
def _(PointReachEnv, evaluate, mo, np):
    _rng = np.random.default_rng(0)
    random_point_policy = lambda o: _rng.uniform(-1, 1, 2)   # noqa: E731
    _m = evaluate(PointReachEnv(), random_point_policy)
    mo.md(f"**학습 전 기준선 (랜덤 Policy, 100 Episode)**: 성공률 {_m['success_rate'] * 100:.1f}%, "
          f"평균 스텝 {_m['steps']:.1f}, 평균 보상 {_m['reward']:+.2f}")
    return (random_point_policy,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 보상에서 PPO 갱신까지

    보상 $r_t$는 환경이 주는 숫자이고, 학습 손실과 성공률은 별개입니다.
    할인 return은 $G_t=\sum_{k=0}^{T-t-1}\gamma^k r_{t+k}$입니다.
    시간 제한으로 잘린 rollout은 마지막 가치로 bootstrap하며, 진짜 종료에서는 뒤의 가치를 0으로 둡니다.
    가치망 $V_\phi(s_t)$는 앞으로 얻을 return의 기준선을 추정하고,
    advantage $\hat A_t$는 그 기준선보다 행동 결과가 얼마나 좋았는지 나타냅니다.
    SB3는 GAE로 advantage를 계산하고, `PPO.train`에서 미니배치 advantage를 정규화합니다.

    $$
    \rho_t=\frac{\pi_\theta(a_t\mid s_t)}{\pi_{\mathrm{old}}(a_t\mid s_t)},\qquad
    L_{\rm clip}=\mathbb E_t\left[\min\left(\rho_t\hat A_t,
    \operatorname{clip}(\rho_t,1-\epsilon,1+\epsilon)\hat A_t\right)\right]
    $$

    $\rho_t$는 **같은 관측·같은 행동**의 새/이전 정책 확률비입니다. 양의 advantage에서
    $\epsilon=0.2$, $\hat A=2$, $\rho=1.5$라면 두 항은 3과 2.4여서 작은 값 2.4를 씁니다.
    clip은 행동 각도를 자르는 연산과 다릅니다. 이 항만으로 모든 정책 변화가 엄격히 제한되지는 않습니다.

    구현은 최대화 대신 다음 손실을 최소화합니다.

    $$
    L=-L_{\rm clip}+c_v\,\mathbb E[(V_\phi-\hat G)^2]-c_e\,\mathbb E[\mathcal H(\pi_\theta)]
    $$

    $\hat G$는 rollout에서 구한 return 표적, $c_v$는 가치 손실 계수, $c_e$는 탐색을 돕는
    entropy 계수입니다. 이 실습의 `PPO(...)`는 기본 `ent_coef=0`이므로 entropy 항의 가중치는 0입니다.
    관측은 Point $(B,4)$ / Arm $(B,14)$, 행동은 $(B,2)$ / $(B,5)$이고 $B$는 미니배치 크기입니다.
    역전파는 정책망과 가치망을 갱신하며 MuJoCo를 미분하지 않습니다. 추론 `predict(..., deterministic=True)`는
    가중치를 고정하고 행동을 고릅니다. 보상을 바꾼 실험에서는 같은 평가 seed의 **성공률·거리**를 비교합니다.
    `shaped`의 거리 진전 항은 할인된 potential shaping의 일반적 보장과 같지 않아 최적 정책 불변을 보장하지 않습니다.

    **작은 확인**: 위 예에서 $\rho=1$이면 목적 항은 2입니다. 확률비 증가가 언제부터 보너스를 더 주지 않는지 설명해 보세요.
    기대 답은 양의 advantage일 때 $\rho>1+\epsilon$ 구간입니다.
    출처: [PPO 논문 §3](https://arxiv.org/abs/1707.06347), 실제 연산은 설치 패키지 `stable_baselines3/ppo/ppo.py:PPO.train`입니다.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5-2. PPO로 첫 학습

    **PPO**(Proximal Policy Optimization)는 현재 Policy로 경험을 모은 뒤, 좋은 결과를 낸 행동의 확률을 높이되
    한 번에 너무 많이 바꾸지 않도록(clip) 갱신하는 알고리즘입니다. `MlpPolicy`는 관측 → 행동 평균을 내는 작은 MLP와
    가치(value)를 추정하는 MLP를 씁니다.

    | 하이퍼파라미터 | 값 | 뜻 |
    | --- | --- | --- |
    | `n_steps` | 1024 | 갱신 한 번에 모으는 스텝 수 |
    | `batch_size` | 256 | 미니배치 크기 |
    | `learning_rate` | 3e-4 | 학습률 |
    | `gamma` | 0.98 | 미래 보상 할인율 |

    sparse 보상은 목표에 우연히 닿기 전까지 신호가 0이라 학습이 훨씬 느립니다. 두 가지를 모두 돌려 비교해 보세요.
    """)
    return


@app.cell
def _(mo):
    point_reward = mo.ui.dropdown(["dense", "sparse"], value="dense", label="보상")
    point_steps = mo.ui.slider(20_000, 300_000, value=100_000, step=20_000, label="학습 스텝")
    point_btn = mo.ui.run_button(label="PPO 학습 (100k 약 1분)")
    mo.hstack([point_reward, point_steps, point_btn], justify="start")
    return point_btn, point_reward, point_steps


@app.cell
def _(
    Monitor,
    PPO,
    PointReachEnv,
    RUN_DIR,
    evaluate,
    mo,
    point_btn,
    point_reward,
    point_steps,
    time,
):
    mo.stop(not point_btn.value, mo.md("버튼을 누르면 학습합니다."))
    _train_env = Monitor(PointReachEnv(reward_mode=point_reward.value))
    point_model = PPO("MlpPolicy", _train_env, seed=0, verbose=0, n_steps=1024, batch_size=256,
                      learning_rate=3e-4, gamma=0.98, device="cpu")
    _t0 = time.time()
    with mo.status.spinner(f"PPO 학습 중 ({point_steps.value:,} steps)..."):
        point_model.learn(total_timesteps=point_steps.value)
    point_model.save(RUN_DIR / f"ppo_point_reach_{point_reward.value}")
    point_curve = (_train_env.get_episode_rewards(), _train_env.get_episode_lengths())
    point_metrics = evaluate(PointReachEnv(reward_mode=point_reward.value),
                             lambda o: point_model.predict(o, deterministic=True)[0])
    point_metrics["train_seconds"] = time.time() - _t0
    return point_curve, point_metrics, point_model


@app.cell
def _(
    PointReachEnv,
    mo,
    np,
    plt,
    point_curve,
    point_metrics,
    point_model,
    point_reward,
    random_point_policy,
    rollout_path,
):
    _rewards, _lengths = point_curve
    _w = 20
    _fig, _axes = plt.subplots(1, 2, figsize=(10, 3.6))
    if len(_rewards) > _w:
        _axes[0].plot(np.cumsum(_lengths)[_w - 1:], np.convolve(_rewards, np.ones(_w) / _w, mode="valid"))
    _axes[0].set(xlabel="timesteps", ylabel=f"episode reward (avg of {_w})", title=f"PPO learning curve ({point_reward.value})")
    _env = PointReachEnv()
    for _seed, _c in zip(range(3), ["tab:blue", "tab:orange", "tab:green"]):
        _p, _g = rollout_path(_env, lambda o: point_model.predict(o, deterministic=True)[0], 100 + _seed)
        _pr, _ = rollout_path(_env, random_point_policy, 100 + _seed)
        _axes[1].plot(_p[:, 0], _p[:, 1], "-", color=_c, label="PPO" if _seed == 0 else None)
        _axes[1].plot(_pr[:, 0], _pr[:, 1], ":", color=_c, alpha=0.6, label="random" if _seed == 0 else None)
        _axes[1].scatter(*_g, marker="*", s=150, color=_c)
        _axes[1].scatter(*_p[0], marker="o", color=_c)
    _axes[1].set(xlim=(-1, 1), ylim=(-1, 1), aspect="equal", title="Paths (o start, * goal)")
    _axes[1].legend(fontsize=8)
    for _ax in _axes:
        _ax.grid(alpha=0.3)
    _fig.tight_layout()
    mo.vstack([mo.md(f"**학습 후 (PPO, 100 Episode)**: 성공률 {point_metrics['success_rate'] * 100:.1f}%, "
                     f"평균 스텝 {point_metrics['steps']:.1f}, 평균 보상 {point_metrics['reward']:+.2f}, "
                     f"학습 {point_metrics['train_seconds']:.0f} 초, Episode {len(_rewards)}개"), _fig])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5-3. SO-101을 Gymnasium 환경으로: 손끝을 블록 위로

    이제 `env`가 MuJoCo 로봇팔입니다. 집기까지는 너무 어려우니 **손끝을 블록의 x·y와 월드 높이 z=0.02 m로 보내는(reach)** 과제만 학습합니다.

    | 요소 | ArmReachEnv |
    | --- | --- |
    | observation (14) | 관절각 5 + 손끝 위치 3 + 목표 위치 3 + (목표 − 손끝) 3 |
    | action (5) | 관절 5개 목표각의 **변화량**, 각 −1~1 → 실제 ±0.05 rad |
    | 한 스텝 | 물리 10 스텝 = 0.02초 |
    | Episode | 최대 100 스텝(2초). 손끝이 목표 2 cm 안이면 성공 |

    행동을 "절대 각도"가 아니라 "현재 명령에서 조금 바꾸기"로 정의한 것이 핵심입니다. 한 스텝에 팔이 크게 튀지 않아
    탐험이 안전하고, 우연히 목표 근처에 가는 경험을 모으기 쉽습니다.

    `reward_mode`

    목표 거리 $d_t$와 성공 여부 $I_t=\mathbf 1[d_t<0.02]$에 대해

    - `sparse`: $r_t=10I_t$
    - `dense`: $r_t=-d_t$
    - `shaped`: $r_t=-d_t+10(d_{t-1}-d_t)+10I_t$

    위 식은 Arm 환경의 보상입니다. Point 환경의 dense 보상에는 도달 시 +10도 포함됩니다.
    """)
    return


@app.cell
def _(gym, mujoco, np, pc, spaces):
    TARGET_ABOVE = 0.02   # 월드 좌표의 목표 높이 (3장의 GRASP_Z와 같은 값)


    class ArmReachEnv(gym.Env):
        metadata = {"render_modes": []}

        def __init__(self, reward_mode="dense", max_steps=100, substeps=10, render=False):
            super().__init__()
            self.reward_mode = reward_mode
            self.max_steps = max_steps
            self.substeps = substeps
            self.robot = pc.SO101Sim(render=render, camera="fixed")
            self.model, self.data = self.robot.model, self.robot.data
            ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in pc.ARM_JOINTS]
            self.ctrl_lo, self.ctrl_hi = self.model.jnt_range[ids, 0], self.model.jnt_range[ids, 1]
            self.observation_space = spaces.Box(-np.inf, np.inf, shape=(14,), dtype=np.float32)
            self.action_space = spaces.Box(-1.0, 1.0, shape=(5,), dtype=np.float32)
            self.cube_adr = self.model.jnt_qposadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "cube_free")]
            self.t, self.prev_dist = 0, None

        def _target(self):
            c = self.robot.cube_pos()
            return np.array([c[0], c[1], TARGET_ABOVE])

        def _obs(self):
            site, tgt = self.robot.site_pose()[0], self._target()
            return np.concatenate([self.robot.q_arm, site, tgt, tgt - site]).astype(np.float32)

        def _dist(self):
            return float(np.linalg.norm(self._target() - self.robot.site_pose()[0]))

        def reset(self, seed=None, options=None):
            super().reset(seed=seed)
            self.robot.reset()
            r, th = self.np_random.uniform(0.17, 0.27), self.np_random.uniform(-0.7, 0.7)   # 3장과 같은 영역
            self.data.qpos[self.cube_adr: self.cube_adr + 3] = [r * np.cos(th), r * np.sin(th), 0.015]
            self.data.ctrl[self.robot.grip_act] = pc.GRIPPER_OPEN
            mujoco.mj_forward(self.model, self.data)
            self.t, self.prev_dist = 0, self._dist()
            return self._obs(), {}

        def step(self, action):
            action = np.clip(np.asarray(action, dtype=np.float32), -1, 1)
            ctrl = self.data.ctrl[self.robot.arm_act] + 0.05 * action       # 목표 각도를 조금씩 바꾼다
            self.data.ctrl[self.robot.arm_act] = np.clip(ctrl, self.ctrl_lo, self.ctrl_hi)
            self.robot.step(self.substeps)
            self.t += 1
            dist = self._dist()
            success = dist < 0.02
            if self.reward_mode == "sparse":
                reward = 10.0 if success else 0.0
            elif self.reward_mode == "dense":
                reward = -dist
            else:   # shaped
                reward = -dist + 10.0 * (self.prev_dist - dist) + (10.0 if success else 0.0)
            self.prev_dist = dist
            return self._obs(), float(reward), success, self.t >= self.max_steps, {"dist": dist, "success": success}
    return (ArmReachEnv,)


@app.cell
def _(ArmReachEnv, evaluate, mo, np):
    _rng = np.random.default_rng(0)
    _m = evaluate(ArmReachEnv(), lambda o: _rng.uniform(-1, 1, 5), episodes=20, seed=1000)
    mo.md(f"**학습 전 기준선 (랜덤 Policy, 20 Episode)**: 성공률 {_m['success_rate'] * 100:.1f}%, "
          f"최종 거리 평균 {_m['final_dist'] * 100:.1f} cm")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5-4. 보상 설계 비교

    같은 환경, 같은 학습량, 같은 seed에서 보상만 바꿔 학습합니다. 환경 8개를 한 프로세스에서 번갈아 돌리는
    `DummyVecEnv`를 씁니다(원본 스크립트는 프로세스 8개의 `SubprocVecEnv`). 진행률 25%마다 20 Episode로 중간 평가를 합니다.
    보상을 하나씩 골라 학습하면 아래 비교 표에 누적됩니다. 30만 스텝 한 번에 약 1분 30초 걸립니다(측정 환경 기준).
    """)
    return


@app.cell
def _(mo):
    get_arm_results, set_arm_results = mo.state({})
    get_arm_models, set_arm_models = mo.state({})
    return get_arm_models, get_arm_results, set_arm_models, set_arm_results


@app.cell
def _(mo):
    arm_reward = mo.ui.dropdown(["sparse", "dense", "shaped"], value="dense", label="보상")
    arm_steps = mo.ui.slider(50_000, 600_000, value=300_000, step=50_000, label="학습 스텝")
    arm_seed = mo.ui.number(value=0, label="seed")
    arm_btn = mo.ui.run_button(label="PPO 학습")
    mo.hstack([arm_reward, arm_steps, arm_seed, arm_btn], justify="start")
    return arm_btn, arm_reward, arm_seed, arm_steps


@app.cell
def _(
    ArmReachEnv,
    DummyVecEnv,
    Monitor,
    PPO,
    RUN_DIR,
    arm_btn,
    arm_reward,
    arm_seed,
    arm_steps,
    evaluate,
    json,
    mo,
    set_arm_models,
    set_arm_results,
    time,
):
    mo.stop(not arm_btn.value, mo.md("보상 종류를 고르고 버튼을 누르세요."))
    _mode, _steps, _seed = arm_reward.value, arm_steps.value, int(arm_seed.value)
    _vec = DummyVecEnv([lambda: Monitor(ArmReachEnv(reward_mode=_mode)) for _ in range(8)])
    _model = PPO("MlpPolicy", _vec, seed=_seed, verbose=0, n_steps=512, batch_size=512,
                 learning_rate=3e-4, gamma=0.98, ent_coef=0.0, device="cpu")
    _eval_env = ArmReachEnv(reward_mode=_mode)
    _policy = lambda o: _model.predict(o, deterministic=True)[0]   # noqa: E731
    _progress, _t0 = [], time.time()
    with mo.status.progress_bar(total=4, title=f"PPO ({_mode}, {_steps:,} steps)") as _bar:
        for _k in range(4):
            _model.learn(total_timesteps=_steps // 4, reset_num_timesteps=(_k == 0))
            _m = evaluate(_eval_env, _policy, episodes=20, seed=1000)
            _progress.append({"percent": 25 * (_k + 1), "seconds": time.time() - _t0, **_m})
            _bar.update()
    _final = evaluate(_eval_env, _policy, episodes=50, seed=1000)
    _model.save(RUN_DIR / f"ppo_arm_reach_{_mode}")
    _record = {"reward_mode": _mode, "steps": _steps, "seed": _seed, "progress": _progress, "final_50ep": _final,
               "train_seconds": time.time() - _t0}
    (RUN_DIR / f"arm_reach_{_mode}.json").write_text(json.dumps(_record, indent=2))
    set_arm_results(lambda d: {**d, _mode: _record})
    set_arm_models(lambda d: {**d, _mode: _model})
    return


@app.cell
def _(get_arm_results, mo, plt):
    _res = get_arm_results()
    mo.stop(not _res, mo.md("아직 학습한 모델이 없습니다."))
    _rows = ["| 보상 | 스텝 | 성공률 (50ep) | 최종 거리 | 평균 스텝 | 학습 시간 |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
    _fig, _ax = plt.subplots(figsize=(5, 3))
    for _mode, _r in _res.items():
        _f = _r["final_50ep"]
        _rows.append(f"| `{_mode}` | {_r['steps']:,} | {_f['success_rate'] * 100:.0f}% | {_f['final_dist'] * 100:.1f} cm | "
                     f"{_f['steps']:.1f} | {_r['train_seconds']:.0f} s |")
        _ax.plot([p["percent"] for p in _r["progress"]], [p["success_rate"] * 100 for p in _r["progress"]], "-o", label=_mode)
    _ax.set(xlabel="training progress (%)", ylabel="success rate (%, 20 ep)", ylim=(-5, 105), title="Reward design comparison")
    _ax.legend()
    _ax.grid(alpha=0.3)
    _fig.tight_layout()
    mo.hstack([mo.md("\n".join(_rows)), _fig], align="center")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **결과를 읽는 법**

    - `sparse`는 무작위 탐험으로 목표 2 cm 안에 들어가는 일이 드물어 학습 신호가 거의 없습니다.
    - `dense`는 매 스텝 거리 정보를 받아 빠르게 배우지만, "빨리 끝내는 것"보다 "가까이 머무는 것"을 배울 수도 있습니다.
    - `shaped`는 가까워진 만큼 즉시 보상해 진행 방향을 알려 줍니다. 단, 잘못 설계한 shaping은 의도와 다른 행동을 낳습니다.
    - seed 하나의 결과입니다. seed를 바꿔 다시 학습하면 수치가 달라집니다. 비교는 여러 seed로 해야 믿을 수 있습니다.

    ### 학습한 Policy 영상

    학습한 보상 종류를 골라 한 Episode를 영상으로 봅니다.
    """)
    return


@app.cell
def _(get_arm_models, mo):
    _models = get_arm_models()
    arm_video_mode = mo.ui.dropdown(list(_models) or ["(없음)"], value=(list(_models) or ["(없음)"])[-1],
                                    label="영상으로 볼 모델")
    arm_video_mode
    return (arm_video_mode,)


@app.cell
def _(ArmReachEnv, RUN_DIR, arm_video_mode, get_arm_models, mo):
    _models = get_arm_models()
    mo.stop(arm_video_mode.value not in _models, mo.md("학습한 모델이 생기면 여기에 영상이 나옵니다."))
    _model = _models[arm_video_mode.value]
    _env = ArmReachEnv(reward_mode=arm_video_mode.value, render=True)
    _rows = []
    for _seed in (7, 8, 9):
        _obs, _ = _env.reset(seed=_seed)
        _done = False
        while not _done:
            _obs, _, _term, _trunc, _info = _env.step(_model.predict(_obs, deterministic=True)[0])
            _done = _term or _trunc
        _env.robot.hold(0.3)
        _rows.append(f"seed {_seed}: {'성공' if _info['success'] else '실패'}, 최종 거리 {_info['dist'] * 100:.1f} cm, {_env.t} 스텝")
    _video = _env.robot.save_video(RUN_DIR / f"arm_reach_{arm_video_mode.value}.mp4")
    _env.robot.close()
    mo.hstack([mo.video(src=_video, loop=True, width=440), mo.md("\n\n".join(_rows))], align="center")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 정리와 다음 장

    - Gymnasium 인터페이스만 맞추면 어떤 시뮬레이터든 RL 라이브러리로 학습할 수 있습니다.
    - 손끝을 보내는 것(reach)만 해도 보상 설계와 학습량에 따라 결과가 크게 달라집니다. 집어서 옮기기처럼 단계가 긴 과제는
      보상만으로 배우기가 훨씬 어렵습니다.
    - 그래서 6장에서는 **잘하는 시연을 보여 주고 따라 하게 하는 모방학습**으로 넘어갑니다. 시연자는 3장의 Rule 기반 Pick and Place입니다.

    **확인 문제**

    1. `ArmReachEnv.step`의 행동 배율 0.05를 0.2로 키우면 학습이 빨라질까요, 불안정해질까요? 직접 바꿔 학습해 보세요.
    2. 관측에서 `(목표 − 손끝)` 3개를 빼면 무엇이 어려워질까요?

    ### 확인 문제 판정 기준

    1. 큰 행동 배율은 탐색 범위를 넓히지만 목표를 지나치거나 관절 추종이 불안정해질 수 있습니다. 속도 향상을 보장하지 않으므로 같은 평가 seed·학습 step에서 성공률과 최종 거리를 비교합니다.
    2. 목표와 손끝 좌표 자체는 남으므로 정보가 사라지는 것은 아닙니다. 네트워크가 두 좌표의 차를 스스로 학습해야 하므로 학습 효율이 달라질 수 있습니다. 관측 공간 shape와 반환 배열을 함께 14에서 11로 바꿔야 비교가 유효합니다.

    다음 [6장](ch06_imitation_act.py)에서는 LeRobot으로 시연 데이터셋을 만들고 ACT 정책을 학습·평가합니다.
    """)
    return


if __name__ == "__main__":
    app.run()
