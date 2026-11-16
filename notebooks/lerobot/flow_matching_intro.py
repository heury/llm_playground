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
    # Flow Matching 시각적 입문: 1차원 노이즈 → 쌍봉 분포

    Flow Matching은 단순한 분포(가우시안 노이즈)의 샘플을 목표 분포의 샘플로 옮기는
    **속도장(velocity field)** 을 학습하는 생성 모델입니다. 이 노트북은 1차원 장난감 예제와
    가장 단순한 **선형(직선 경로) Flow Matching** 으로 학습 목표, 학습, 샘플링을 그림으로 확인합니다.

    SmolVLA·π0의 행동 전문가가 행동 청크를 만드는 방식이 바로 이 방법입니다.
    [SmolVLA 구조 실습](smolvla_tour.py)의 Flow Matching 절을 보기 전에 읽으면 좋습니다.

    `notebooks/lerobot`에서 `uv run marimo edit flow_matching_intro.py`로 실행합니다.
    다운로드는 없고, 학습은 작은 MLP를 CPU에서 수십 초 돌리며 **버튼을 눌러야** 시작합니다.

    원문: Peter Roelants, [Flow Matching: A visual introduction](https://peterroelants.github.io/posts/flow_matching_intro/).
    원문의 순서와 예제 설정(혼합 분포 매개변수, MLP 크기, 학습률 등)을 따르되 설명과 코드는 이 저장소에 맞춰
    새로 작성했습니다. 그림 일부(애니메이션 등)는 marimo 슬라이더로 대체했습니다.
    """)
    return


@app.cell
def _():
    import math

    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    import torch.nn as nn
    from matplotlib.colors import TwoSlopeNorm

    return TwoSlopeNorm, math, nn, np, plt, torch


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. 장난감 예제: 가우시안 노이즈 → 쌍봉 분포

    - 시작(노이즈) 분포 $\pi_0 = \mathcal{N}(0, 1)$
    - 목표 분포 $\pi_1$: 두 가우시안의 혼합. 가중치 $[0.55, 0.45]$, 평균 $[-0.85, 1.5]$, 표준편차 $[0.65, 0.25]$

    목표는 $\pi_0$에서 뽑은 점 $x_0$을 시간 $t: 0 \to 1$ 동안 움직여 $\pi_1$을 따르는 $x_1$로 만드는 것입니다.
    실제 문제에서는 $\pi_1$의 식을 모르고 샘플(데이터)만 있습니다. 여기서는 그림을 위해 밀도식도 둡니다.
    """)
    return


@app.cell
def _(math, np, torch):
    MIX_WEIGHTS = np.array([0.55, 0.45])
    MIX_MEANS = np.array([-0.85, 1.5])
    MIX_STDS = np.array([0.65, 0.25])


    def normal_pdf(x, mean=0.0, std=1.0):
        return np.exp(-0.5 * ((x - mean) / std) ** 2) / (std * math.sqrt(2 * math.pi))


    def target_pdf(x):
        """π1의 밀도 (그림용)."""
        return sum(w * normal_pdf(x, m, s) for w, m, s in zip(MIX_WEIGHTS, MIX_MEANS, MIX_STDS))


    def sample_target(n, generator=None):
        """π1에서 n개 샘플, shape [n, 1]. 실제 학습에서는 '데이터셋에서 뽑기'에 해당합니다."""
        comp = torch.multinomial(torch.tensor(MIX_WEIGHTS, dtype=torch.float32), n, replacement=True, generator=generator)
        means = torch.tensor(MIX_MEANS, dtype=torch.float32)[comp]
        stds = torch.tensor(MIX_STDS, dtype=torch.float32)[comp]
        return (means + stds * torch.randn(n, generator=generator)).unsqueeze(1)


    def sample_noise(n, generator=None):
        """π0 = N(0, 1)에서 n개 샘플, shape [n, 1]."""
        return torch.randn(n, 1, generator=generator)

    return normal_pdf, sample_noise, sample_target, target_pdf


@app.cell
def _(normal_pdf, np, plt, target_pdf):
    _x = np.linspace(-3, 3, 400)
    _fig, _axes = plt.subplots(1, 2, figsize=(9, 2.8), sharey=True)
    _axes[0].fill_between(_x, normal_pdf(_x), alpha=0.6, color="tab:blue")
    _axes[0].set_title(r"noise $\pi_0 = \mathcal{N}(0,1)$")
    _axes[1].fill_between(_x, target_pdf(_x), alpha=0.6, color="tab:orange")
    _axes[1].set_title(r"target $\pi_1$ (Gaussian mixture)")
    for _ax in _axes:
        _ax.set_xlabel("x")
    _axes[0].set_ylabel("density")
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. 모델은 속도장을 예측합니다

    점 $x_t$의 시간에 따른 움직임을 상미분방정식(ODE)으로 씁니다.

    $$\frac{dx_t}{dt} = v(x_t, t)$$

    Flow Matching 모델 $\mathrm{FM}_\theta(x_t, t)$는 이 속도 $v$를 근사합니다.
    속도장을 알면 $x_0 \sim \pi_0$에서 출발해 $t=0 \to 1$로 적분하여 $x_1$을 얻습니다.
    한 번에 $x_1$을 예측하지 않고 "지금 위치·시각에서 어느 방향으로 얼마나 가야 하는가"를 배우는 것이 핵심입니다.

    문제는 정답 속도장 $v$를 모른다는 점입니다. 아래 세 단계로 학습 신호를 만듭니다.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 2.1 직선 참조 경로

    노이즈 샘플 $X_0 \sim \pi_0$과 데이터 샘플 $X_1 \sim \pi_1$을 **무작위로 짝지어** 직선으로 잇습니다.

    $$x_t = (1 - t)\,X_0 + t\,X_1, \qquad t \in [0, 1]$$

    짝짓기는 서로 독립이므로 경로들이 서로 교차합니다. 아래 그림의 가운데는 몇 개 경로,
    아래 행은 많은 경로를 겹쳐 만든 경로 밀도입니다. $t=0$에서는 $\pi_0$, $t=1$에서는 $\pi_1$ 모양이 됩니다.
    """)
    return


@app.cell
def _(normal_pdf, np, plt, sample_noise, sample_target, target_pdf, torch):
    _g = torch.Generator().manual_seed(1)
    ref_x0 = sample_noise(100_000, _g)[:, 0].numpy()
    ref_x1 = sample_target(100_000, _g)[:, 0].numpy()

    _t = np.linspace(0, 1, 101)
    _xr = (-2.5, 2.5)
    _xs = np.linspace(*_xr, 400)

    _fig, _axes = plt.subplots(
        2, 3, figsize=(11, 5.5), sharey=True, gridspec_kw={"width_ratios": [1, 3, 1]}
    )
    # 위: 밀도 + 경로 7개
    _axes[0, 0].fill_betweenx(_xs, normal_pdf(_xs), color="tab:blue", alpha=0.6)
    _axes[0, 0].invert_xaxis()
    _axes[0, 0].set_title(r"$\pi_0$")
    # 두 봉우리에서 고르게 보이도록 아래 봉우리 4개, 위 봉우리 3개를 고릅니다.
    _show = np.r_[np.flatnonzero(ref_x1 < 0.3)[:4], np.flatnonzero(ref_x1 >= 0.3)[:3]]
    for _i in _show:
        _path = (1 - _t) * ref_x0[_i] + _t * ref_x1[_i]
        _axes[0, 1].plot(_t, _path, lw=1.5)
        _axes[0, 1].scatter([0, 1], [ref_x0[_i], ref_x1[_i]], s=15, color="k", zorder=3)
    _axes[0, 1].set_title("7 straight reference paths")
    _axes[0, 2].fill_betweenx(_xs, target_pdf(_xs), color="tab:orange", alpha=0.6)
    _axes[0, 2].set_title(r"$\pi_1$")

    # 아래: 히스토그램 + 경로 밀도
    _bins = np.linspace(*_xr, 481)
    _axes[1, 0].hist(ref_x0, bins=_bins, orientation="horizontal", density=True, color="tab:blue", alpha=0.6)
    _axes[1, 0].invert_xaxis()
    _paths = (1 - _t[:, None]) * ref_x0[None, :] + _t[:, None] * ref_x1[None, :]  # [T, N]
    _dens = np.stack([np.histogram(_row, bins=_bins, density=True)[0] for _row in _paths], axis=1)
    _axes[1, 1].imshow(
        _dens, origin="lower", aspect="auto", extent=(0, 1, *_xr), cmap="viridis", vmax=np.quantile(_dens, 0.995)
    )
    _axes[1, 1].set_title("path density of 100k reference paths")
    _axes[1, 1].set_xlabel("t")
    _axes[1, 2].hist(ref_x1, bins=_bins, orientation="horizontal", density=True, color="tab:orange", alpha=0.6)
    for _ax in _axes[:, 0]:
        _ax.set_ylabel("x")
    _axes[0, 1].set_ylim(*_xr)
    _fig.tight_layout()
    _fig
    return ref_x0, ref_x1


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 2.2 경로 위의 속도 벡터

    직선 경로를 $t$로 미분하면 속도는 시간에 상관없이 일정합니다.

    $$v = \frac{dx_t}{dt} = X_1 - X_0$$

    즉 한 쌍 $(X_0, X_1)$과 시각 $t$를 뽑으면 **입력** $(x_t, t)$와 **정답 속도** $X_1 - X_0$이 바로 생깁니다.
    아래는 $x_0 = 1.32$, $x_1 = -1.92$, $t = 0.67$인 예입니다.
    """)
    return


@app.cell
def _(plt):
    _x0, _x1, _tt = 1.32, -1.92, 0.67
    _xt = (1 - _tt) * _x0 + _tt * _x1
    _fig, _ax = plt.subplots(figsize=(7, 3.2))
    _ax.plot([0, 1], [_x0, _x1], color="gray", lw=1.5)
    _ax.scatter([0, 1], [_x0, _x1], color="k", zorder=3)
    _ax.scatter([_tt], [_xt], color="tab:red", zorder=4, s=50)
    _ax.annotate(
        "", xy=(_tt + 0.25, _xt + 0.25 * (_x1 - _x0)), xytext=(_tt, _xt),
        arrowprops=dict(arrowstyle="->", color="tab:red", lw=2),
    )
    _ax.text(0.02, _x0 + 0.15, r"$x_0$")
    _ax.text(0.93, _x1 - 0.35, r"$x_1$")
    _ax.text(_tt + 0.03, _xt + 0.2, rf"$x_t$ (t={_tt})", color="tab:red")
    _ax.text(0.25, -1.6, r"$v = x_1 - x_0 = %.2f$" % (_x1 - _x0), color="tab:red")
    _ax.set_xlabel("t")
    _ax.set_ylabel("x")
    _ax.set_xlim(-0.05, 1.05)
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 2.3 Flow Matching 목표

    모델이 예측한 속도와 짝지은 경로의 속도의 제곱 오차를 최소화합니다.

    $$\theta^* = \arg\min_\theta \; \mathbb{E}_{t \sim U[0,1],\, X_0 \sim \pi_0,\, X_1 \sim \pi_1}
    \left\| \mathrm{FM}_\theta(x_t, t) - (X_1 - X_0) \right\|^2, \qquad x_t = (1-t)X_0 + tX_1$$

    같은 $(x_t, t)$를 여러 경로가 지나가므로 개별 정답 $X_1 - X_0$은 서로 다릅니다.
    제곱 오차의 최솟값은 그 위치를 지나는 경로 속도들의 **조건부 평균**
    $\mathbb{E}[X_1 - X_0 \mid x_t, t]$입니다. 모델이 배우는 것은 이 평균 속도장이고,
    평균 속도장을 따라 적분한 경로는 서로 교차하지 않으면서도 $t=1$에서 $\pi_1$을 재현합니다.

    아래 그림은 모델 없이 참조 경로들의 속도를 $(t, x)$ 칸마다 평균 낸 것입니다.
    빨강은 위로 미는 양의 속도, 파랑은 아래로 당기는 음의 속도입니다. 화살표 선은 이 평균장을 따라 적분한 경로입니다.
    """)
    return


@app.cell
def _(TwoSlopeNorm, np, plt, ref_x0, ref_x1):
    def binned_mean_velocity(x0, x1, n_t=21, x_range=(-2.2, 2.2), n_x=200):
        """(t, x) 칸별 평균 속도 E[X1-X0 | x_t, t]. 데이터가 없는 칸은 NaN."""
        ts = np.linspace(0, 1, n_t)
        edges = np.linspace(*x_range, n_x + 1)
        v = x1 - x0
        field = np.full((n_x, n_t), np.nan)
        for j, t in enumerate(ts):
            xt = (1 - t) * x0 + t * x1
            idx = np.digitize(xt, edges) - 1
            ok = (idx >= 0) & (idx < n_x)
            sums = np.bincount(idx[ok], weights=v[ok], minlength=n_x)
            cnts = np.bincount(idx[ok], minlength=n_x)
            with np.errstate(invalid="ignore", divide="ignore"):
                field[:, j] = np.where(cnts > 20, sums / cnts, np.nan)
        return ts, edges, field


    _ts, _edges, _field = binned_mean_velocity(ref_x0, ref_x1)
    _centers = 0.5 * (_edges[:-1] + _edges[1:])
    _lim = np.nanmax(np.abs(_field))

    _fig, _ax = plt.subplots(figsize=(8, 4))
    _im = _ax.imshow(
        _field, origin="lower", aspect="auto", extent=(0, 1, _edges[0], _edges[-1]),
        cmap="coolwarm", norm=TwoSlopeNorm(0, -_lim, _lim),
    )
    _fig.colorbar(_im, ax=_ax, label="mean velocity (red ↑, blue ↓)")

    # 평균장을 따라 Euler 적분한 경로 9개
    _dt = _ts[1] - _ts[0]
    for _x in np.linspace(-2.0, 2.0, 9):
        _xs = [_x]
        for _j in range(len(_ts) - 1):
            _k = np.clip(np.searchsorted(_centers, _xs[-1]), 0, len(_centers) - 1)
            _vel = np.nan_to_num(_field[_k, _j])
            _xs.append(_xs[-1] + _vel * _dt)
        _ax.plot(_ts, _xs, color="k", lw=1)
        _ax.quiver(_ts[:-1:4], _xs[:-1:4], np.full(len(_ts[:-1:4]), _dt), np.diff(_xs)[::4],
                   angles="xy", scale_units="xy", scale=0.25, width=0.004, color="k")
    _ax.set_xlabel("t")
    _ax.set_ylabel("x")
    _ax.set_title("binned average target velocity + integrated pathlines")
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Flow Matching 모델 학습

    모델은 $[t, x_t]$를 이어 붙인 2차원 입력을 받는 작은 MLP입니다
    (`Linear(2→64) → GELU → Linear(64→64) → GELU → Linear(64→1)`).
    한 번의 학습 단계는 다음과 같습니다.

    1. 데이터 $X_1$ 배치(256)를 뽑고, 같은 크기의 노이즈 $X_0$과 $t \sim U[0,1]$을 뽑습니다.
    2. $x_t = (1-t)X_0 + tX_1$, 정답 $X_1 - X_0$을 만듭니다.
    3. MSE 손실로 Adam(lr $10^{-3}$) 한 스텝.

    시뮬레이션(ODE 적분) 없이 샘플 한 쌍마다 손실을 바로 계산하는 점이 Flow Matching의 장점입니다.
    """)
    return


@app.cell
def _(nn, torch):
    class FlowMatchingModel(nn.Module):
        """FM_θ(x_t, t) ≈ v(x_t, t). 입력 [t, x_t], 출력 속도."""

        def __init__(self, data_dim=1, hidden_dim=64):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(data_dim + 1, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, data_dim),
            )

        def forward(self, t, x_t):
            # t: [B, 1], x_t: [B, data_dim]
            return self.net(torch.cat([t, x_t], dim=-1))

    return (FlowMatchingModel,)


@app.cell
def _(mo):
    n_iters = mo.ui.slider(1_000, 20_000, step=1_000, value=10_000, label="학습 반복 수")
    train_button = mo.ui.run_button(label="학습 시작")
    mo.hstack([n_iters, train_button], justify="start")
    return n_iters, train_button


@app.cell
def _(FlowMatchingModel, mo, n_iters, sample_noise, sample_target, torch, train_button):
    mo.stop(not train_button.value, mo.md("**학습 시작** 버튼을 누르면 학습합니다. 이후 셀은 학습된 모델이 필요합니다."))

    torch.manual_seed(626)
    fm_model = FlowMatchingModel(data_dim=1, hidden_dim=64)
    _opt = torch.optim.Adam(fm_model.parameters(), lr=1e-3)
    train_losses = []
    for _ in mo.status.progress_bar(range(n_iters.value), title="Flow Matching 학습"):
        _x1 = sample_target(256)
        _x0 = sample_noise(256)
        _t = torch.rand(256, 1)
        _xt = (1 - _t) * _x0 + _t * _x1
        _loss = ((fm_model(_t, _xt) - (_x1 - _x0)) ** 2).mean()
        _opt.zero_grad()
        _loss.backward()
        _opt.step()
        train_losses.append(_loss.item())
    fm_model.eval()
    return fm_model, train_losses


@app.cell
def _(np, plt, train_losses):
    _w = 100
    _fig, _ax = plt.subplots(figsize=(8, 3))
    _ax.plot(train_losses, alpha=0.4, lw=0.5, label="loss")
    _ax.plot(np.arange(_w - 1, len(train_losses)), np.convolve(train_losses, np.ones(_w) / _w, mode="valid"),
             color="tab:red", label=f"moving avg ({_w})")
    _ax.set_xlabel("iteration")
    _ax.set_ylabel("MSE")
    _ax.set_title("Training loss curve")
    _ax.legend()
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    손실은 0까지 내려가지 않습니다. 같은 $(x_t, t)$에 서로 다른 정답 $X_1 - X_0$이 섞여 있어서,
    완벽한 모델도 조건부 평균과 개별 정답 사이의 분산만큼 손실이 남습니다.
    손실의 절댓값보다 아래 속도장과 생성 샘플로 품질을 판단합니다.

    ### 3.1 학습된 속도장

    학습된 네트워크를 $(t, x)$ 격자에서 평가하고, 9개 시작점에서 Euler 적분한 경로를 겹쳤습니다.
    2.3절의 칸별 평균 속도장과 거의 같은 모양이어야 합니다.
    """)
    return


@app.cell
def _(torch):
    @torch.inference_mode()
    def integrate(model, x0, n_steps=100):
        """Euler 적분 x_{t+dt} = x_t + FM(x_t, t)·dt. 반환: 시각 [n+1], 경로 [n+1, N, D]."""
        dt = 1.0 / n_steps
        x = x0.clone()
        traj = [x.clone()]
        for k in range(n_steps):
            t = torch.full((x.shape[0], 1), k * dt)
            x = x + model(t, x) * dt
            traj.append(x.clone())
        return torch.linspace(0, 1, n_steps + 1), torch.stack(traj)

    return (integrate,)


@app.cell
def _(TwoSlopeNorm, fm_model, integrate, np, plt, torch):
    _ts = torch.linspace(0, 1, 101)
    _xs = torch.linspace(-3.2, 3.2, 200)
    _T, _X = torch.meshgrid(_ts, _xs, indexing="xy")  # [200, 101]
    with torch.inference_mode():
        _V = fm_model(_T.reshape(-1, 1), _X.reshape(-1, 1)).reshape(_X.shape).numpy()
    _lim = np.abs(_V).max()

    _fig, _ax = plt.subplots(figsize=(8, 4))
    _im = _ax.imshow(_V, origin="lower", aspect="auto", extent=(0, 1, -3.2, 3.2),
                     cmap="coolwarm", norm=TwoSlopeNorm(0, -_lim, _lim))
    _fig.colorbar(_im, ax=_ax, label="learned velocity (red ↑, blue ↓)")
    _tt, _paths = integrate(fm_model, torch.linspace(-3.2, 3.2, 9).unsqueeze(1), n_steps=100)
    _paths = _paths[:, :, 0].numpy()
    for _i in range(_paths.shape[1]):
        _ax.plot(_tt, _paths[:, _i], color="k", lw=1)
    _s = slice(7, None, 35)
    _dx = np.diff(_paths, axis=0)[_s]
    _ax.quiver(np.repeat(_tt[:-1][_s].numpy()[:, None], 9, 1), _paths[:-1][_s], np.full_like(_dx, 0.01), _dx,
               angles="xy", scale_units="xy", scale=0.1, width=0.004, color="k")
    _ax.set_xlabel("t")
    _ax.set_ylabel("x")
    _ax.set_ylim(-3.2, 3.2)
    _ax.set_title("learned velocity field + Euler pathlines")
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. 학습된 모델에서 샘플링

    $x_0 \sim \mathcal{N}(0,1)$을 뽑고 Euler 방법으로 $t=0 \to 1$ 적분합니다.

    $$x_{t+\Delta t} = x_t + \mathrm{FM}_\theta(x_t, t)\,\Delta t, \qquad \Delta t = 1/N$$

    아래 표는 $x_0 = 0.85$에서 출발해 15스텝으로 적분한 과정입니다. 원문은 이 과정을 속도장 위의 애니메이션으로 보여 줍니다.
    """)
    return


@app.cell
def _(fm_model, mo, torch):
    _n = 15
    _dt = 1.0 / _n
    _x = torch.tensor([[0.85]])
    _rows = []
    with torch.inference_mode():
        for _k in range(_n):
            _t = _k * _dt
            _v = fm_model(torch.tensor([[_t]]), _x).item()
            _x_next = _x.item() + _v * _dt
            _rows.append({"step": _k, "t": round(_t, 3), "x_t": round(_x.item(), 4),
                          "FM(x_t, t)": round(_v, 4), "x_{t+dt}": round(_x_next, 4)})
            _x = torch.tensor([[_x_next]])
    mo.ui.table(_rows, selection=None, page_size=15)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 4.1 목표 분포 재구성

    많은 샘플(10만 개)을 같은 방식으로 적분하면 $t=1$의 히스토그램이 $\pi_1$을 재현해야 합니다.
    슬라이더로 Euler 스텝 수와 중간 시각 $t$를 바꿔 보세요. 스텝이 1개면 한 번에 평균 방향으로 이동하므로
    두 봉우리가 뭉개지고, 스텝을 늘리면 곡선 경로를 따라가 분포가 선명해집니다.
    """)
    return


@app.cell
def _(mo):
    n_steps = mo.ui.slider(1, 100, value=50, label="Euler 스텝 수 N")
    t_view = mo.ui.slider(0.0, 1.0, step=0.05, value=1.0, label="보는 시각 t")
    mo.hstack([n_steps, t_view], justify="start")
    return n_steps, t_view


@app.cell
def _(fm_model, integrate, n_steps, np, plt, sample_noise, t_view, target_pdf, torch):
    _g = torch.Generator().manual_seed(0)
    gen_t, gen_paths = integrate(fm_model, sample_noise(100_000, _g), n_steps=n_steps.value)
    gen_paths = gen_paths[:, :, 0].numpy()

    _k = int(round(t_view.value * n_steps.value))
    _xs = np.linspace(-3.5, 3.5, 400)
    _fig, _ax = plt.subplots(figsize=(8, 3.2))
    _ax.hist(gen_paths[_k], bins=200, range=(-3.5, 3.5), density=True, alpha=0.6, label=f"samples at t={_k / n_steps.value:.2f}")
    _ax.plot(_xs, target_pdf(_xs), color="tab:orange", lw=2, label=r"target $\pi_1$")
    _ax.set_xlabel("x")
    _ax.set_ylabel("density")
    _ax.set_title(f"generated distribution (N={n_steps.value} Euler steps)")
    _ax.legend()
    _fig.tight_layout()
    _fig
    return gen_paths, gen_t


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 4.2 생성 경로의 밀도

    2.1절의 참조 경로 밀도와 비교해 보세요. 참조 경로는 서로 교차하는 직선이지만,
    학습된 속도장을 따른 경로는 **교차하지 않는 곡선**입니다. 양 끝($t=0$, $t=1$)의 분포는 같습니다.
    같은 $(x, t)$에서 속도가 하나뿐인 ODE이므로 경로는 서로 만날 수 없습니다.
    """)
    return


@app.cell
def _(gen_paths, gen_t, np, plt):
    _xr = (-2.5, 2.5)
    _bins = np.linspace(*_xr, 481)
    _dens = np.stack([np.histogram(_row, bins=_bins, density=True)[0] for _row in gen_paths], axis=1)
    _fig, _ax = plt.subplots(figsize=(8, 3.5))
    _ax.imshow(_dens, origin="lower", aspect="auto", extent=(0, 1, *_xr), cmap="viridis",
               vmax=np.quantile(_dens, 0.995))
    for _i in range(0, 100_000, 12_500):
        _ax.plot(gen_t, gen_paths[:, _i], color="w", lw=0.8, alpha=0.8)
    _ax.set_xlabel("t")
    _ax.set_ylabel("x")
    _ax.set_title("path density of generated samples (white: individual paths)")
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. 정리와 LeRobot 연결

    | 단계 | 내용 |
    | --- | --- |
    | 학습 데이터 | $X_0 \sim \pi_0$, $X_1 \sim$ 데이터, $t \sim U[0,1]$를 독립적으로 뽑아 직선 $x_t$ 생성 |
    | 학습 목표 | $\|\mathrm{FM}_\theta(x_t, t) - (X_1 - X_0)\|^2$ — 결과적으로 조건부 평균 속도장 학습 |
    | 샘플링 | $x_0 \sim \pi_0$에서 Euler 등 ODE 적분으로 $t=0\to1$ |
    | 결과 | 교차하지 않는 곡선 경로, 스텝 수와 품질의 절충 |

    **SmolVLA·π0와의 차이**

    - 시간 방향이 반대입니다: $x_t = t\,\epsilon + (1-t)\,a$로 $t=1$이 노이즈, $t=0$이 행동이고
      목표 속도는 $\epsilon - a$, 샘플링은 $dt = -1/N$으로 $1 \to 0$ 적분합니다. 같은 방법을 시간 축만 뒤집은 것입니다.
    - $x$는 1차원 점 대신 행동 청크 `[B, H, A]`이고, 속도장은 이미지·지시문·상태 조건(VLM 특징)을 함께 받습니다.
    - $t$를 균등 분포 대신 Beta 분포에서 뽑고, 적분은 10스텝 정도로 적게 합니다.

    자세한 구현은 [smolvla_tour.py](smolvla_tour.py)의 Flow Matching 절을 보세요.

    **참고 문헌** (원문 인용)

    - Lipman et al., *Flow Matching for Generative Modeling*, [arXiv:2210.02747](https://arxiv.org/abs/2210.02747)
    - Liu et al., *Flow Straight and Fast: Rectified Flow*, [arXiv:2209.03003](https://arxiv.org/abs/2209.03003)
    - Lipman et al., *Flow Matching Guide and Code*, [arXiv:2412.06264](https://arxiv.org/abs/2412.06264)
    """)
    return


if __name__ == "__main__":
    app.run()
