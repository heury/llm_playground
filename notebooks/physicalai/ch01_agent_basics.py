import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import random
    import sys
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    _here = str(Path(__file__).resolve().parent)
    if _here not in sys.path:
        sys.path.insert(0, _here)
    return mo, np, plt, random


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 1장. AI 없는 Agent부터 시작하기

    《로봇 없이 시작하는 Physical AI 따라하기》 실습 노트북 1장입니다.
    이 과정은 실물 로봇 없이 **MuJoCo 시뮬레이션 → 기구학 → 카메라 → 강화학습 → 모방학습(ACT) → VLA(SmolVLA) → LLM Planner**
    순서로 Physical AI 파이프라인 전체를 따라갑니다.

    | 장 | 노트북 | 핵심 질문 |
    | --- | --- | --- |
    | 1 | `ch01_agent_basics.py` | Agent는 무엇을 반복하는가? |
    | 2 | `ch02_mujoco_basics.py` | 가상 세계와 로봇팔은 어떻게 만들고 움직이는가? |
    | 3 | `ch03_kinematics_pick_place.py` | 관절각과 손끝 위치는 어떻게 오가는가? (FK·IK, Rule 기반 Pick and Place) |
    | 4 | `ch04_vision.py` | 카메라로 물체 위치를 어떻게 알아내는가? |
    | 5 | `ch05_reinforcement_learning.py` | 보상만으로 행동을 배울 수 있는가? (PPO) |
    | 6 | `ch06_imitation_act.py` | 시연을 따라 하는 정책은 어떻게 만드는가? (LeRobot ACT) |
    | 7 | `ch07_smolvla.py` | 자연어 명령을 행동으로 바꿀 수 있는가? (SmolVLA) |
    | 8 | `ch08_generalization.py` | 학습한 정책은 처음 보는 상황에서도 동작하는가? |
    | 9 | `ch09_llm_planner.py` | LLM이 도구를 골라 로봇을 지휘하면 어떻게 되는가? |

    이 장은 시뮬레이터 없이 순수 Python만 씁니다. 책 전체를 꿰는 **세 줄의 루프**를 먼저 손에 익힙니다.

    ```python
    observation = env.observe()     # 1. 관측: 세상이 어떤 상태인가
    action = policy(observation)    # 2. 판단: 무엇을 할 것인가
    env.step(action)                # 3. 행동: 세상을 바꾼다
    ```

    2장 이후 `env`는 MuJoCo 로봇팔로, `policy`는 if문 → IK 상태 기계 → 강화학습 신경망 → ACT → SmolVLA → LLM으로 바뀝니다.
    루프의 모양은 끝까지 그대로입니다.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1-1. 세 줄의 코드로 움직이는 Agent

    가장 단순한 세계는 0~9 눈금이 있는 1차원 직선(`LineWorld`)입니다.
    Agent(`A`)는 목표(`G`)까지 가야 합니다. 행동은 -1(왼쪽)과 +1(오른쪽) 두 가지입니다.
    """)
    return


@app.cell
def _():
    class LineWorld:
        """0부터 9까지 눈금이 있는 1차원 세계."""

        def __init__(self, start=0, goal=7):
            self.position = start
            self.goal = goal
            self.t = 0

        def observe(self):
            """Agent가 세계로부터 받는 관측(Observation)."""
            return {"position": self.position, "goal": self.goal}

        def step(self, action):
            """행동(-1 또는 +1)을 적용한다. 벽 밖으로는 못 나간다."""
            self.position = max(0, min(9, self.position + action))
            self.t += 1

        def done(self):
            return self.position == self.goal

        def render(self):
            return "".join("A" if i == self.position else "G" if i == self.goal else "." for i in range(10))


    def line_policy(observation):
        """규칙 기반 Policy: 목표가 오른쪽이면 +1, 아니면 -1. AI는 없다."""
        return +1 if observation["goal"] > observation["position"] else -1
    return LineWorld, line_policy


@app.cell
def _(mo):
    line_start = mo.ui.slider(0, 9, value=0, label="시작 위치")
    line_goal = mo.ui.slider(0, 9, value=7, label="목표 위치")
    mo.hstack([line_start, line_goal], justify="start")
    return line_goal, line_start


@app.cell
def _(LineWorld, line_goal, line_policy, line_start, mo):
    _env = LineWorld(start=line_start.value, goal=line_goal.value)
    _lines = [f"t={_env.t:2d}  {_env.render()}"]
    while not _env.done() and _env.t < 20:
        _obs = _env.observe()          # 1. 관측
        _action = line_policy(_obs)    # 2. 판단
        _env.step(_action)             # 3. 행동
        _lines.append(f"t={_env.t:2d}  {_env.render()}  action={_action:+d}")
    _lines.append(f"목표 도달! 총 {_env.t} 스텝" if _env.done() else "20 스텝 안에 도달하지 못함")
    mo.md("```text\n" + "\n".join(_lines) + "\n```")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    시작 위치와 목표가 같으면 루프가 한 번도 돌지 않습니다. `done()`이 먼저 검사되기 때문입니다.

    ## 1-2. Observation · State · Action · Reward · Episode

    2차원 격자(`GridWorld`)로 넓히고 강화학습의 용어를 붙여 봅니다.

    | 용어 | GridWorld에서 | 이후 장에서 |
    | --- | --- | --- |
    | **State** | 세계의 진짜 상태(Agent·Goal 좌표, 시간) | MuJoCo의 `qpos`, 블록의 정확한 위치 |
    | **Observation** | Agent가 받는 정보 `{"agent", "goal"}` | 관절각 + 카메라 이미지 |
    | **Action** | 상·하·좌·우 4가지 | 관절 목표각 6개 |
    | **Reward** | 도달 +10, 매 스텝 -0.1, 시간 초과 -1 | 5장의 dense/sparse/shaped 보상 |
    | **Episode** | reset부터 도달 또는 시간 초과까지 | Pick and Place 한 번 |

    여기서는 State와 Observation이 같습니다. 4장부터 Observation이 카메라 이미지가 되면 둘이 달라집니다.
    """)
    return


@app.cell
def _(random):
    ACTIONS = {0: (0, -1), 1: (0, +1), 2: (-1, 0), 3: (+1, 0)}   # 위, 아래, 왼쪽, 오른쪽
    ACTION_NAMES = {0: "위", 1: "아래", 2: "왼쪽", 3: "오른쪽"}


    class GridWorld:
        """size x size 격자. Agent가 goal 칸에 도달하면 Episode가 끝난다."""

        def __init__(self, size=5, max_steps=50):
            self.size = size
            self.max_steps = max_steps
            self.reset()

        def reset(self):
            self.agent = (0, 0)
            self.goal = (self.size - 1, self.size - 1)
            self.t = 0
            return self.observe()

        def observe(self):
            return {"agent": self.agent, "goal": self.goal}

        def step(self, action):
            """행동을 적용하고 (observation, reward, done)을 돌려준다."""
            dx, dy = ACTIONS[action]
            x = min(self.size - 1, max(0, self.agent[0] + dx))
            y = min(self.size - 1, max(0, self.agent[1] + dy))
            self.agent = (x, y)
            self.t += 1
            if self.agent == self.goal:
                return self.observe(), +10.0, True       # 목표 도달
            if self.t >= self.max_steps:
                return self.observe(), -1.0, True        # 시간 초과
            return self.observe(), -0.1, False           # 한 스텝마다 작은 벌점

        def render(self):
            rows = []
            for y in range(self.size):
                rows.append(" ".join("A" if (x, y) == self.agent else "G" if (x, y) == self.goal else "."
                                     for x in range(self.size)))
            return "\n".join(rows)


    def random_policy(observation):
        """무엇을 보든 아무 행동이나 고른다."""
        return random.choice(list(ACTIONS))


    def rule_policy(observation):
        """목표 쪽으로 한 칸 움직이는 규칙."""
        (ax, ay), (gx, gy) = observation["agent"], observation["goal"]
        if gx > ax:
            return 3
        if gx < ax:
            return 2
        if gy > ay:
            return 1
        return 0


    def run_episode(env, policy):
        """Episode 하나를 끝까지 돌리고 (총 보상, 스텝 수, 성공, 경로)를 돌려준다."""
        observation = env.reset()
        total, done, path = 0.0, False, [env.agent]
        while not done:
            action = policy(observation)
            observation, reward, done = env.step(action)
            total += reward
            path.append(env.agent)
        return total, env.t, env.agent == env.goal, path
    return ACTION_NAMES, GridWorld, random_policy, rule_policy, run_episode


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 직접 해 보기: 나만의 Policy

    아래 `my_policy`를 고쳐 보세요. 예를 들어 80%는 규칙을 따르고 20%는 무작위로 움직이게 하면
    성공률과 평균 스텝이 어떻게 바뀔까요? 셀을 고치면 아래 비교 표가 자동으로 다시 계산됩니다.
    """)
    return


@app.cell
def _(random, random_policy, rule_policy):
    def my_policy(observation):
        if random.random() < 0.8:
            return rule_policy(observation)
        return random_policy(observation)
    return (my_policy,)


@app.cell
def _(mo):
    grid_size = mo.ui.slider(3, 10, value=5, label="격자 크기")
    grid_max_steps = mo.ui.slider(10, 200, value=50, step=10, label="최대 스텝")
    grid_episodes = mo.ui.slider(10, 500, value=100, step=10, label="Episode 수")
    mo.hstack([grid_size, grid_max_steps, grid_episodes], justify="start")
    return grid_episodes, grid_max_steps, grid_size


@app.cell
def _(
    GridWorld,
    grid_episodes,
    grid_max_steps,
    grid_size,
    mo,
    my_policy,
    np,
    random,
    random_policy,
    rule_policy,
    run_episode,
):
    grid_results = {}
    for _name, _policy in [("random_policy", random_policy), ("rule_policy", rule_policy),
                           ("my_policy", my_policy)]:
        random.seed(0)
        _env = GridWorld(size=grid_size.value, max_steps=grid_max_steps.value)
        grid_results[_name] = [run_episode(_env, _policy) for _ in range(grid_episodes.value)]

    _rows = ["| Policy | 성공률 | 평균 스텝 | 평균 보상 |", "| --- | ---: | ---: | ---: |"]
    for _name, _res in grid_results.items():
        _r = np.array([x[0] for x in _res])
        _t = np.array([x[1] for x in _res])
        _ok = np.array([x[2] for x in _res])
        _rows.append(f"| `{_name}` | {_ok.mean() * 100:.1f}% | {_t.mean():.1f} | {_r.mean():+.2f} |")
    mo.md("\n".join(_rows))
    return (grid_results,)


@app.cell
def _(grid_results, plt):
    _fig, _axes = plt.subplots(1, 2, figsize=(10, 3.2))
    for _name, _res in grid_results.items():
        _axes[0].hist([x[0] for x in _res], bins=20, alpha=0.6, label=_name)
        _path = _res[0][3]
        _axes[1].plot([p[0] for p in _path], [p[1] for p in _path], "-o", ms=3, alpha=0.7, label=_name)
    _axes[0].set(xlabel="episode total reward", ylabel="episodes", title="Reward distribution")
    _axes[1].invert_yaxis()
    _axes[1].set(title="Path of the first episode (y grows downward)", xlabel="x", ylabel="y")
    for _ax in _axes:
        _ax.legend(fontsize=8)
        _ax.grid(alpha=0.3)
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(ACTION_NAMES, GridWorld, mo, random, random_policy):
    random.seed(1)
    _env = GridWorld(size=4, max_steps=12)
    _obs = _env.reset()
    _log = [f"t= 0\n{_env.render()}"]
    _done = False
    while not _done:
        _a = random_policy(_obs)
        _obs, _r, _done = _env.step(_a)
        _log.append(f"t={_env.t:2d} action={ACTION_NAMES[_a]} reward={_r:+.1f}\n{_env.render()}")
    mo.accordion({"랜덤 Policy의 Episode 하나를 한 스텝씩 보기 (4x4, 최대 12스텝)":
                  mo.md("```text\n" + "\n\n".join(_log) + "\n```")})
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 정리와 다음 장

    - 규칙 Policy는 사람이 세계를 완벽히 알 때만 쉽게 짤 수 있습니다. GridWorld는 쉽지만,
      관절 6개짜리 로봇팔이 카메라 이미지만 보고 블록을 집는 규칙을 if문으로 쓰기는 어렵습니다.
    - 랜덤 Policy의 성공률은 **학습 전 기준선**입니다. 5장에서 PPO를 학습하기 전에도 같은 방식으로 기준선을 잽니다.
    - 성공률은 Episode 수와 난수 seed에 따라 흔들립니다. Episode 수를 10으로 줄여 표가 얼마나 바뀌는지 확인해 보세요.

    **확인 문제**

    1. `GridWorld.step`에서 매 스텝 벌점을 0으로 바꾸면 랜덤 Policy와 규칙 Policy의 평균 보상 차이는 어떻게 되나요?
    2. Goal을 매 Episode 무작위 위치에 두려면 어디를 고쳐야 하나요? 그때 `rule_policy`는 여전히 100%인가요?

    다음 [2장](ch02_mujoco_basics.py)에서는 MuJoCo로 물리 법칙이 있는 가상 세계를 만듭니다.
    """)
    return


if __name__ == "__main__":
    app.run()
