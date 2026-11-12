import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import sys
    from pathlib import Path
    import marimo as mo
    import matplotlib.pyplot as plt
    import torch
    import torch.nn.functional as F

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "projects" / "minimind"))
    from model.model_minimind import (
        FeedForward, MiniMindBlock, MiniMindConfig, precompute_freqs_cis,
    )
    torch.set_num_threads(2)
    return F, FeedForward, MiniMindBlock, MiniMindConfig, mo, plt, precompute_freqs_cis, torch


@app.cell
def _(mo):
    mo.md(r"""
    # MiniMind FeedForward: 토큰의 특징을 바꾸기

    **목표:** 원본 `FeedForward`의 SwiGLU 계산을 직접 따라갑니다.

    - FFN은 토큰끼리 정보를 섞지 않습니다. 이를 확인합니다.
    - 역전파와 학습도 확인합니다.

    `minimind_structure.py` 4절의 MLP를 자세히 보는 **구조 심화** 실습입니다.
    외부 모델·데이터·GPU 없이 CPU로 실행합니다. 숫자 벡터는 가상 특징입니다.
    설정을 바꾸면 새 모델로 다시 시작합니다.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. 세 선형층으로 SwiGLU 따라가기

    MiniMind의 FeedForward는 다음 계산입니다. 기본 설정의 활성화는 SiLU입니다.

    <img src="public/diagrams/feedforward.png" alt="SwiGLU FeedForward 계산 흐름" width="620">

    - `D` 은닉 차원, `M` 중간 차원.
    - gate와 up은 서로 다른 특징 조합입니다. 둘 다 학습됩니다.
    - `*`는 원소별 곱, `@`는 행렬 곱입니다.
    - SiLU는 $z\,\mathrm{sigmoid}(z)$입니다. 음수도 나옵니다. 합이 1일 필요도 없습니다.
    - **이 gate는 MoE의 router(softmax)와 다릅니다.**

    **예측:** D=4, M=8이면 가중치 shape와 매개변수 수는?
    bias가 없으므로 `gate: [M,D]`, `up: [M,D]`, `down: [D,M]`. 합계 `3DM=96`개.
    """)
    return


@app.cell
def _(FeedForward, MiniMindConfig, torch):
    demo_config = MiniMindConfig(
        hidden_size=4, intermediate_size=8, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=1,
        max_position_embeddings=16, flash_attn=False,
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(21)
        demo_ffn = FeedForward(demo_config).eval()
    sample_x = torch.tensor([
        [[1., -1., 0., 2.], [0., 1., 2., -1.], [2., 0., -1., 1.]],
        [[-1., 1., 0., 1.], [1., 0., 1., -2.], [0., -1., 1., 2.]],
    ])
    return (demo_config, demo_ffn, sample_x,)


@app.cell(hide_code=True)
def _(demo_ffn, mo):
    # ▸ 결과 표시 — 실행하면 표·그림이 나타납니다 (marimo 표시 코드)
    mo.ui.table([
        {"가중치": n, "shape": str(tuple(p.shape)), "개수": p.numel()}
        for n, p in demo_ffn.named_parameters()
    ], selection=None)
    return


@app.cell
def _(F, demo_ffn, sample_x, torch):
    with torch.no_grad():
        gate_values = sample_x @ demo_ffn.gate_proj.weight.T
        activated_gate = F.silu(gate_values)
        up_values = sample_x @ demo_ffn.up_proj.weight.T
        gated_features = activated_gate * up_values
        manual_ffn = gated_features @ demo_ffn.down_proj.weight.T
        torch.testing.assert_close(manual_ffn, demo_ffn(sample_x))
    assert sum(p.numel() for p in demo_ffn.parameters()) == 96
    return (activated_gate, gate_values, gated_features, manual_ffn, up_values,)


@app.cell(hide_code=True)
def _(activated_gate, gate_values, gated_features, manual_ffn, mo, sample_x, up_values):
    # ▸ 결과 표시 — 실행하면 표·그림이 나타납니다 (marimo 표시 코드)
    mo.vstack([
        mo.md(f"입력 `{tuple(sample_x.shape)}` → 중간 `{tuple(gated_features.shape)}` → 출력 `{tuple(manual_ffn.shape)}`. 원본과 같습니다."),
        mo.ui.table([
            {"중간 차원": i, "gate 선형값": float(gate_values[0, 0, i]), "SiLU(gate)": float(activated_gate[0, 0, i]),
             "up 값": float(up_values[0, 0, i]), "곱한 값": float(gated_features[0, 0, i])}
            for i in range(8)
        ], selection=None),
        mo.md("**관찰:** 첫 토큰의 중간 특징 8개입니다. 한 칸의 곱을 손으로 계산해 보세요. down_proj가 이를 다시 4차원으로 합칩니다."),
    ])
    return


@app.cell(hide_code=True)
def _(F, mo, plt, torch):
    # ▸ 결과 표시 — 실행하면 표·그림이 나타납니다 (marimo 표시 코드)
    _z = torch.linspace(-5, 5, 101)
    _fig, _ax = plt.subplots(figsize=(6, 3))
    _ax.plot(_z.numpy(), F.silu(_z).numpy(), label="SiLU(z)")
    _ax.plot(_z.numpy(), _z.sigmoid().numpy(), label="sigmoid(z)", linestyle="--")
    _ax.axhline(0, color="gray", linewidth=0.5)
    _ax.set(xlabel="z", ylabel="Value", title="SiLU is not a probability")
    _ax.legend()
    _fig.tight_layout()
    mo.vstack([_fig, mo.md("**해설:** SiLU는 음수도 냅니다. gate 값은 확률이 아닙니다. 토큰 선택도 아닙니다.")])
    return


@app.cell
def _(mo):
    mo.md("""
    ## 2. 한 토큰을 바꾸면 다른 토큰도 바뀔까?

    **예측:** 첫 배치의 두 번째 토큰만 바꾸면 첫 번째 토큰의 출력도 바뀔까요?
    FeedForward는 같은 가중치를 토큰마다 따로 적용합니다.
    가중치는 같지만, 토큰끼리 정보를 주고받지 않습니다.
    이것이 Attention과의 핵심 차이입니다.
    """)
    return


@app.cell
def _(demo_ffn, sample_x, torch):
    with torch.no_grad():
        _changed = sample_x.clone()
        _changed[0, 1, 0] += 3
        _before = demo_ffn(sample_x)
        _after = demo_ffn(_changed)
        ffn_token_change = (_after - _before).abs().amax(-1)
        _unchanged = torch.ones(2, 3, dtype=torch.bool)
        _unchanged[0, 1] = False
        torch.testing.assert_close(_before[_unchanged], _after[_unchanged])
        assert ffn_token_change[0, 1] > 0
        _separate = torch.stack([demo_ffn(sample_x[:, i]) for i in range(3)], dim=1)
        torch.testing.assert_close(_before, _separate)
    return (ffn_token_change,)


@app.cell(hide_code=True)
def _(ffn_token_change, mo):
    # ▸ 결과 표시 — 실행하면 표·그림이 나타납니다 (marimo 표시 코드)
    mo.ui.table([
        {"배치": b, "토큰 위치": t, "출력 최대 변화": float(ffn_token_change[b, t])}
        for b in range(2) for t in range(3)
    ], selection=None)
    return


@app.cell
def _(mo):
    mo.md("""
    ## 3. 블록 안에서는 정규화와 잔차 연결을 더한다

    **예측:** FeedForward 단독 출력에 입력 x가 이미 더해져 있을까요?
    **아닙니다.** FeedForward는 특징 변환만 합니다. 더하기는 블록이 합니다:
    `h + FeedForward(RMSNorm(h))`.

    - h는 Attention과 첫 잔차 연결을 지난 값입니다.
    - 그래서 h에는 이미 다른 토큰의 문맥이 들어 있습니다.
    - FFN 출력도 D차원이라 h에 바로 더할 수 있습니다.

    아래에서 원본 블록 전체와 비교합니다.
    """)
    return


@app.cell
def _(MiniMindBlock, demo_config, precompute_freqs_cis, sample_x, torch):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(22)
        _block = MiniMindBlock(0, demo_config).eval()
    with torch.no_grad():
        _position = precompute_freqs_cis(demo_config.head_dim, end=sample_x.shape[1])
        _attention, _ = _block.self_attn(_block.input_layernorm(sample_x), _position)
        block_hidden = sample_x + _attention
        _ff_output = _block.mlp(_block.post_attention_layernorm(block_hidden))
        _manual = block_hidden + _ff_output
        _actual, _ = _block(sample_x, _position)
        torch.testing.assert_close(_manual, _actual)
        _changed = sample_x.clone()
        _changed[0, 1, 0] += 3
        _block_after, _ = _block(_changed, _position)
        _ffn_change = (_block.mlp(_changed) - _block.mlp(sample_x)).abs().amax(-1)
        _block_change = (_block_after - _actual).abs().amax(-1)
        assert float(_ffn_change[0, 2]) == 0
        assert float(_block_change[0, 2]) > 0
        torch.testing.assert_close(_block_after[0, 0], _actual[0, 0])
    context_change_rows = [
        {"배치": b, "토큰 위치": t, "FFN 단독 출력 변화": float(_ffn_change[b, t]), "전체 블록 출력 변화": float(_block_change[b, t])}
        for b in range(2) for t in range(3)
    ]
    return (block_hidden, context_change_rows,)


@app.cell(hide_code=True)
def _(block_hidden, context_change_rows, mo):
    # ▸ 결과 표시 — 실행하면 표·그림이 나타납니다 (marimo 표시 코드)
    mo.vstack([
        mo.md(f"직접 조립한 블록과 원본 출력이 같습니다. h·FFN·잔차 출력 shape는 모두 `{tuple(block_hidden.shape)}`입니다. 잔차 연결은 FeedForward 밖에서 합니다."),
        mo.md("**예측:** 배치 0의 위치 1을 바꾸면 위치 2도 바뀔까요? 두 경로를 비교하세요."),
        mo.ui.table(context_change_rows, selection=None),
        mo.md("**해설:** FFN만 쓰면 위치 1만 바뀝니다. 전체 블록에서는 Attention 때문에 위치 2도 바뀝니다. 위치 0은 미래를 못 보므로 그대로입니다. 'FFN은 위치를 섞지 않는다'가 '문맥을 안 쓴다'는 뜻은 아닙니다."),
    ])
    return


@app.cell
def _(mo):
    mo.md("""
    ## 4. 숫자 특징 변환을 실제로 학습하기

    **과제:** 입력 x로 `목표=[x0*x1, SiLU(x2), x0-x3, x1*x2]`를 맞힙니다.

    - 다음 토큰 예측이 아닌 연속값 회귀입니다. 오차는 MSE입니다.
    - 80개 중 64개로 학습합니다. 16개는 검증용입니다.

    **예측:** 중간 차원을 늘리면 매개변수 수는? 검증 오차도 꼭 줄까요?

    학습 순서: `zero_grad` → `backward` → `step`.
    `no_grad()`는 gradient 기록을 끕니다. 매번 새 모델로 시작합니다.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    # ▸ 실습 설정 위젯 — 실행하면 슬라이더·선택 상자가 나타납니다 (marimo 표시 코드)
    ffn_width = mo.ui.dropdown([8, 16, 32], value=16, label="중간 차원 M")
    ffn_steps = mo.ui.slider(100, 300, step=100, value=200, label="학습 step")
    mo.hstack([ffn_width, ffn_steps])
    return ffn_steps, ffn_width


@app.cell
def _(F, torch):
    _generator = torch.Generator().manual_seed(23)
    _all_x = torch.randn(80, 4, generator=_generator)
    _all_y = torch.stack([
        _all_x[:, 0] * _all_x[:, 1], F.silu(_all_x[:, 2]),
        _all_x[:, 0] - _all_x[:, 3], _all_x[:, 1] * _all_x[:, 2],
    ], dim=-1)
    ff_train_x, ff_train_y = _all_x[:64].reshape(8, 8, 4), _all_y[:64].reshape(8, 8, 4)
    ff_valid_x, ff_valid_y = _all_x[64:].reshape(2, 8, 4), _all_y[64:].reshape(2, 8, 4)
    return ff_train_x, ff_train_y, ff_valid_x, ff_valid_y


@app.cell
def _(F, FeedForward, MiniMindConfig, ff_train_x, ff_train_y, ff_valid_x, ff_valid_y, ffn_steps, ffn_width, plt, torch):
    _train_x, _train_y = ff_train_x, ff_train_y
    ffn_eval_x, ffn_eval_y = ff_valid_x, ff_valid_y
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(24)
        trained_ffn = FeedForward(MiniMindConfig(hidden_size=4, intermediate_size=ffn_width.value,
                                                num_attention_heads=2, num_key_value_heads=1))
    _optimizer = torch.optim.AdamW(trained_ffn.parameters(), lr=0.02, weight_decay=0)
    with torch.no_grad():
        _initial_valid = float(F.mse_loss(trained_ffn(ffn_eval_x), ffn_eval_y))
    ffn_losses = []
    ffn_gradient_norms = {}
    for _step in range(ffn_steps.value):
        _optimizer.zero_grad(set_to_none=True)
        _loss = F.mse_loss(trained_ffn(_train_x), _train_y)
        assert torch.isfinite(_loss)
        _loss.backward()
        if _step == 0:
            ffn_gradient_norms = {n: float(p.grad.norm()) for n, p in trained_ffn.named_parameters()}
            assert all(v > 0 for v in ffn_gradient_norms.values())
        _optimizer.step()
        ffn_losses.append(float(_loss.detach()))
    trained_ffn.eval()
    with torch.no_grad():
        ffn_prediction = trained_ffn(ffn_eval_x)
        _train_final = float(F.mse_loss(trained_ffn(_train_x), _train_y))
        _valid_final = float(F.mse_loss(ffn_prediction, ffn_eval_y))
    assert _train_final < ffn_losses[0]
    ffn_metrics = {"parameters": sum(p.numel() for p in trained_ffn.parameters()),
                   "train_before": ffn_losses[0], "train_after": _train_final,
                   "validation_before": _initial_valid, "validation_after": _valid_final}
    ffn_fig, _ax = plt.subplots(figsize=(6, 3))
    _ax.plot(ffn_losses)
    _ax.set(xlabel="Step", ylabel="MSE", title="FeedForward regression / train split")
    ffn_fig.tight_layout()
    return (
        ffn_eval_x,
        ffn_eval_y,
        ffn_fig,
        ffn_gradient_norms,
        ffn_losses,
        ffn_metrics,
        ffn_prediction,
        trained_ffn,
    )


@app.cell(hide_code=True)
def _(ffn_eval_x, ffn_eval_y, ffn_fig, ffn_gradient_norms, ffn_metrics, ffn_prediction, mo):
    # ▸ 결과 표시 — 실행하면 표·그림이 나타납니다 (marimo 표시 코드)
    mo.vstack([
        ffn_fig,
        mo.ui.table([ffn_metrics], selection=None),
        mo.ui.table([{"가중치": n, "첫 backward의 gradient norm": v} for n, v in ffn_gradient_norms.items()], selection=None),
        mo.ui.table([
            {"검증 입력": str(ffn_eval_x[0, i].tolist()), "목표": str(ffn_eval_y[0, i].tolist()), "예측": str(ffn_prediction[0, i].tolist())}
            for i in range(4)
        ], selection=None),
        mo.md("**관찰:** 세 projection 모두 gradient를 받습니다. 학습 오차와 검증 오차는 따로 보세요. 이 결과는 언어 모델 성능과 무관합니다."),
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### 4-1. 왜 선형층 하나 대신 SwiGLU를 쓸까?

    **예측:** x0*x1 같은 곱을 선형층 하나로 표현할 수 있을까요?

    선형층은 가중합만 합니다.
    활성화 없이 두 층을 쌓아도 $W_2(W_1x)=(W_2W_1)x$. 여전히 선형입니다.

    아래 네 구조를 같은 데이터·step·학습률로 비교합니다.

    | 구조 | 계산 | 매개변수 수(D=4) |
    | --- | --- | --- |
    | Linear | x → Linear | D²=16 |
    | TwoLinear | x → up → down (gate·활성화 없음) | 2DM |
    | SiLU MLP | x → up → SiLU → down (곱셈 gate 없음) | 2DM |
    | SwiGLU | down(SiLU(gate(x)) * up(x)) | 3DM |

    - 아래 세 구조의 up/down은 같은 초기값에서 시작합니다.
    - 매개변수 수가 달라서 공정한 용량 비교는 아닙니다.
    - 이 작은 과제의 결과만 관찰합니다.
    """)
    return


@app.cell
def _(F, FeedForward, MiniMindConfig, ff_train_x, ff_train_y, ff_valid_x, ff_valid_y, ffn_losses, ffn_metrics, ffn_steps, ffn_width, plt, torch):
    ffn_comparison_rows = [{"구조": "SwiGLU", **ffn_metrics}]
    _histories = {"SwiGLU": ffn_losses}
    for _name in ["Linear", "TwoLinear", "SiLU MLP"]:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(24)
            _reference = FeedForward(MiniMindConfig(hidden_size=4, intermediate_size=ffn_width.value,
                                                     num_attention_heads=2, num_key_value_heads=1))
            if _name == "Linear":
                _model = torch.nn.Linear(4, 4, bias=False)
            elif _name == "TwoLinear":
                _model = torch.nn.Sequential(_reference.up_proj, _reference.down_proj)
            else:
                _model = torch.nn.Sequential(_reference.up_proj, torch.nn.SiLU(), _reference.down_proj)
        if _name == "TwoLinear":
            with torch.no_grad():
                _merged = _reference.down_proj.weight @ _reference.up_proj.weight
                torch.testing.assert_close(_model(ff_train_x), ff_train_x @ _merged.T)
        _optimizer = torch.optim.AdamW(_model.parameters(), lr=0.02, weight_decay=0)
        with torch.no_grad():
            _validation_before = float(F.mse_loss(_model(ff_valid_x), ff_valid_y))
        _losses = []
        for _step in range(ffn_steps.value):
            _optimizer.zero_grad(set_to_none=True)
            _loss = F.mse_loss(_model(ff_train_x), ff_train_y)
            assert torch.isfinite(_loss)
            _loss.backward()
            _optimizer.step()
            _losses.append(float(_loss.detach()))
        with torch.no_grad():
            _train_after = float(F.mse_loss(_model(ff_train_x), ff_train_y))
            _validation_after = float(F.mse_loss(_model(ff_valid_x), ff_valid_y))
        ffn_comparison_rows.append({"구조": _name, "parameters": sum(p.numel() for p in _model.parameters()),
                                    "train_before": _losses[0], "train_after": _train_after,
                                    "validation_before": _validation_before, "validation_after": _validation_after})
        _histories[_name] = _losses
    comparison_fig, _ax = plt.subplots(figsize=(7, 3))
    for _name, _losses in _histories.items():
        _ax.plot(_losses, label=_name)
    _ax.set(xlabel="Step", ylabel="Train MSE", title="Same regression data / different feature transforms")
    _ax.legend()
    comparison_fig.tight_layout()
    return (comparison_fig, ffn_comparison_rows,)


@app.cell(hide_code=True)
def _(comparison_fig, ffn_comparison_rows, mo):
    # ▸ 결과 표시 — 실행하면 표·그림이 나타납니다 (marimo 표시 코드)
    mo.vstack([
        comparison_fig, mo.ui.table(ffn_comparison_rows, selection=None),
        mo.md("**관찰:** 선형 모델도 일부는 맞힙니다. 하지만 곱셈·비선형 특징은 잘 못 표현합니다. MSE와 매개변수 수를 함께 보세요. seed 하나로 우열을 단정하지 마세요."),
    ])
    return


@app.cell
def _(mo):
    mo.md("""
    ## 5. 직접 바꿔 볼 과제와 해설

    1. M을 8/16/32로 바꾸기 전에 매개변수 수를 예측하세요. 실행 후 학습·검증 MSE를 적으세요.
       **기준:** 개수는 `3DM`입니다. M이 커도 검증 오차가 꼭 줄지는 않습니다.
    2. 4-1절의 TwoLinear와 SwiGLU를 비교하세요. 두 선형층을 합치면 shape는?
       **해설:** `[D,M] @ [M,D] = [D,D]`. 여전히 선형입니다.
    3. 두 위치를 바꾸면 어디 출력이 달라질까요?
       **해설:** FFN만이면 바꾼 위치만. 전체 블록이면 뒤 위치도 달라질 수 있습니다.
    4. down_proj는 왜 D차원으로 돌아가야 할까요?
       **해설:** 잔차 입력과 더하려면 shape가 같아야 합니다.

    **다음 실습:** `minimind_moe.py`. FeedForward 여러 개 중 일부를 토큰마다 고릅니다.
    """)
    return


if __name__ == "__main__":
    app.run()
