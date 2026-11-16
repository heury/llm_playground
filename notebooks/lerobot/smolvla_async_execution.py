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
    # SmolVLA 보충 실습: 행동 청크와 비동기실행

    [SmolVLA 구조 해부](smolvla_tour.py)의 기존 11절을 분리했습니다.
    CPU에서 가짜 정책으로 큐 정렬과 청크 혼합을 비교합니다. 다운로드·모델 로드·학습은 없습니다.
    `notebooks/lerobot`에서 `uv run marimo edit smolvla_async_execution.py`로 엽니다.
    슬라이더를 바꾸면 다시 계산합니다. 고정 tick 모형이며 실제 추론 속도·네트워크·스레드 성능을 재현하지 않습니다.

    원본: [Hugging Face LeRobot SmolVLA](https://github.com/huggingface/lerobot/tree/e6ffde60312365c5452c46189a26872483943154/src/lerobot/policies/smolvla)
    및 같은 리비전의 `policies/common/flow_matching.py`, `policies/common/vla_utils.py`, `inference/execution.py`.
    고정 리비전 `e6ffde60312365c5452c46189a26872483943154`, Apache License 2.0,
    Copyright 2025–2026 The HuggingFace Inc. team. 원본을 축소·수정한 실습입니다.
    """)
    return


@app.cell
def _():
    from collections import deque
    import matplotlib.font_manager as font_manager
    import matplotlib.pyplot as plt
    import numpy as np

    _installed = {f.name for f in font_manager.fontManager.ttflist}
    for _name in ["Noto Sans CJK KR", "NanumGothic", "Malgun Gothic", "AppleGothic", "Noto Sans Mono CJK KR"]:
        if _name in _installed:
            plt.rcParams["font.family"] = _name
            break
    plt.rcParams["axes.unicode_minus"] = False
    return deque, np, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. 행동 청크와 비동기 추론

    SmolVLA 추론 한 번은 수백 ms가 걸리지만 로봇 제어 주기는 30~50 Hz(20~33 ms)입니다. 그래서

    1. **행동 청킹:** 한 번에 50스텝을 만들어 큐에 넣고 하나씩 꺼내 실행합니다(원본 `select_action`은 큐가 비면 새로 추론).
    2. **비동기 추론:** 큐를 실행하는 동안 미리 다음 청크를 요청해, 추론 지연 동안에도 로봇이 멈추지 않게 합니다.

    고정 리비전의 LeRobot 구현(`lerobot/inference/execution.py`의 `ChunkRuntime`)에서 새 청크를 받는 규칙은 다음과 같습니다(`chunk_merge="aligned"`).

    - **요청 시점:** 큐에 남은 재생 시간(`큐 길이 × 제어 주기`)이 유효 임계값 `max(refill_seconds, turnaround + interval)` 이하이고 진행 중인 요청이 없을 때.
    - **잘라내기:** 관측을 찍은 시점의 실행 커서와 지금 커서의 차이 `trim`만큼, 이미 지나간 앞부분을 버립니다.
    - **섞기(선택):** 겹치는 앞쪽 `blend_steps`개를 $(1-w)a_{old}+wa_{new}$ (기본 $w=0.5$)로 섞고, 나머지는 새 청크로 **큐를 교체**합니다.

    아래 시뮬레이션은 고정 가중치 혼합을 따릅니다. 요청 임계값은 지연을 자동 추정하지 않는 고정 tick 수로 단순화했습니다.
    모델 대신 "경로 위 다음 위치 + 청크마다 다른 작은 오차"를 내는 가짜 정책을 써서 **실행 파이프라인만** 비교합니다.
    시간 단위는 제어 tick이며, 지연은 추론에 걸리는 tick 수입니다.
    """)
    return


@app.cell
def _(deque, np):
    def reference_path(k):
        """경로 인덱스 k의 목표 위치 (8자 경로)."""
        s = 2 * np.pi * k / 120
        return np.array([0.7 * np.sin(s), 0.4 * np.sin(2 * s)])

    def simulate_execution(mode, latency, threshold, blend_steps, blend_weight=0.5, chunk_size=30, path_len=120, noise_std=0.04, seed=0):
        """mode: 'sync' | 'async_no_trim' | 'async_trim' | 'async_blend'"""
        rng = np.random.default_rng(seed)
        queue = deque()  # (경로 인덱스, 행동)
        pending = None  # (도착 tick, 관측 시점 진행도, 청크)
        progress, tick, idle = 0, 0, 0  # progress = 마지막으로 실행한 행동의 경로 인덱스
        pos = reference_path(0)
        log = {"tick": [], "x": [], "queue": [], "arrivals": [], "jumps": []}
        while progress < path_len and tick < 1000:
            if pending is not None and pending[0] == tick:  # 1) 새 청크 도착
                _, obs_progress, chunk = pending
                pending = None
                if mode == "async_no_trim":
                    new = list(chunk)  # 관측 이후 이미 실행한 구간을 다시 실행하게 됨
                else:
                    # 잘라내기: 관측 시점 이후 실행된 행동 수(trim)만큼 앞부분을 버립니다.
                    trim = progress - obs_progress
                    new = chunk[trim:]
                if mode == "async_blend":
                    old = dict(queue)
                    for i in range(min(blend_steps, len(new))):
                        k, a = new[i]
                        if k in old:
                            new[i] = (k, (1 - blend_weight) * old[k] + blend_weight * a)
                queue = deque(new)  # 남은 기존 행동은 새 청크로 교체
                log["arrivals"].append(tick)
            if pending is None:  # 2) 요청 여부
                need = len(queue) == 0 if mode == "sync" else len(queue) <= threshold
                if need:
                    offset = rng.normal(0, noise_std, 2)  # 청크마다 다른 샘플 오차
                    chunk = [(progress + k, reference_path(progress + k) + offset) for k in range(1, chunk_size + 1)]
                    pending = (tick + latency, progress, chunk)
            if queue:  # 3) 실행 (큐가 비면 제자리 정지)
                progress, action = queue.popleft()
                log["jumps"].append(float(np.linalg.norm(action - pos)))
                pos = action
            else:
                idle += 1
            log["tick"].append(tick)
            log["x"].append(pos.copy())
            log["queue"].append(len(queue))
            tick += 1
        log["x"] = np.array(log["x"])
        jumps = np.array(log["jumps"])
        # 경로를 정확히 따라갈 때의 최대 1스텝 이동량의 2배를 넘으면 "큰 점프"로 셉니다.
        nominal = max(np.linalg.norm(reference_path(k + 1) - reference_path(k)) for k in range(path_len))
        return {"log": log, "ticks": tick, "idle": idle, "max_jump": jumps.max(), "n_big_jumps": int((jumps > 2 * nominal).sum()), "nominal": nominal}

    return (simulate_execution,)


@app.cell
def _(mo):
    sim_latency = mo.ui.slider(1, 25, value=8, label="추론 지연 (tick)", show_value=True)
    sim_threshold = mo.ui.slider(0, 25, value=12, label="요청 임계값 (큐 길이)", show_value=True)
    sim_blend = mo.ui.slider(0, 10, value=4, label="blend_steps", show_value=True)
    mo.vstack([sim_latency, sim_threshold, sim_blend])
    return sim_blend, sim_latency, sim_threshold


@app.cell
def _(mo, plt, sim_blend, sim_latency, sim_threshold, simulate_execution):
    _modes = {
        "sync": "동기 (큐가 비면 요청하고 기다림)",
        "async_no_trim": "비동기, 잘라내기 없음",
        "async_trim": "비동기 + 잘라내기",
        "async_blend": f"비동기 + 잘라내기 + 섞기({sim_blend.value})",
    }
    _res = {m: simulate_execution(m, sim_latency.value, sim_threshold.value, sim_blend.value) for m in _modes}
    _fig, _axes = plt.subplots(2, 1, figsize=(10, 5.5), sharex=True)
    for _m, _label in _modes.items():
        _l = _res[_m]["log"]
        _axes[0].plot(_l["tick"], _l["x"][:, 0], lw=1.2, label=_label)
        _axes[1].plot(_l["tick"], _l["queue"], lw=1)
    _axes[0].set_ylabel("x 위치")
    _axes[0].legend(fontsize=7, ncol=2)
    _axes[1].set_ylabel("큐 길이")
    _axes[1].axhline(sim_threshold.value, color="k", ls=":", lw=0.8)
    _axes[1].set_xlabel("제어 tick")
    for _a in _axes:
        _a.grid(alpha=0.3)
    _fig.tight_layout()
    _rows = "\n".join(
        f"| {_label} | {_res[_m]['ticks']} | {_res[_m]['idle']} | {_res[_m]['max_jump']:.3f} | {_res[_m]['n_big_jumps']} |"
        for _m, _label in _modes.items()
    )
    mo.vstack([
        _fig,
        mo.md(f"""
    경로 120스텝을 마치는 데 걸린 시간과 움직임 품질. 경로를 그대로 따라갈 때 1스텝 이동량은 최대 {_res["sync"]["nominal"]:.3f}이며,
    그 2배를 넘는 이동을 "큰 점프"로 셉니다.

    | 방식 | 완료 tick | 정지 tick | 최대 점프 | 큰 점프 횟수 |
    | --- | --- | --- | --- | --- |
    {_rows}
    """),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    관찰할 점:

        - 첫 청크는 어느 방식이든 지연만큼 기다립니다(콜드 스타트).
        - **동기 방식**은 청크마다 지연만큼 멈춥니다(정지 tick = 청크 수 × 지연). 실제 로봇에서는 멈칫거림(stuttering)으로 보입니다.
        - **잘라내기 없이** 새 청크를 처음부터 실행하면 로봇이 이미 지나간 위치로 되돌아가 큰 점프가 생깁니다.
        - 임계값을 지연보다 작게 하면(예: 지연 15, 임계값 5) 비동기여도 큐가 바닥나 정지 tick이 생깁니다.
          원본의 `effective_refill = max(refill_seconds, turnaround + interval)`가 지연 여유를 확보하는 이유입니다.
        - **섞기**는 청크 경계에서 서로 다른 샘플 오차 때문에 생기는 작은 점프를 줄입니다. 섞는 구간을 길게 하면 더 매끄럽지만 새 관측 반영이 늦어집니다.
          SmolVLA는 이보다 정교한 **RTC(Real-Time Chunking)** 도 지원합니다. RTC는 섞는 대신, 이미 실행이 확정된 앞부분을 조건으로 걸고
          나머지를 Flow Matching으로 생성해(`euler_integrate`의 RTC 훅) 경계를 맞춥니다.


    ### 추가 실험

    지연을 청크 길이(30)에 가깝게 바꾸고 잘라내기 후 남는 행동 수와 정지 tick을 비교하세요.
    trim은 벽시계 지연이 아니라 관측 이후 실제로 소비한 행동 수입니다.
    더 이른 요청·긴 청크·낮은 지연은 큐 고갈을 줄일 수 있으나 긴 청크는 새 관측 반영을 늦춥니다.
    """)
    return


if __name__ == "__main__":
    app.run()
