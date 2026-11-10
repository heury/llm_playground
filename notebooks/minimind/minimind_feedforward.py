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

    **목표:** 원본 `FeedForward`의 SwiGLU 계산을 직접 따라가고, 토큰 사이의 정보를
    섞지 않는다는 성질과 역전파·학습을 확인합니다.
    `minimind_structure.py` 4절에서 한 줄로 검증한 MLP를 자세히 해부하는 **구조 심화** 실습입니다.
    외부 모델·데이터·GPU 없이 CPU에서 단독 실행합니다. 숫자 벡터는 계산을 보기 위한 가상 특징입니다.
    설정을 바꾸면 새 모델로 다시 시작하므로 학습이 누적되지 않습니다.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. 세 선형층으로 SwiGLU 따라가기

    MiniMind의 FeedForward는 다음 계산입니다. 기본 설정의 활성화는 SiLU입니다.

    <img src="public/diagrams/feedforward.png" alt="SwiGLU FeedForward 계산 흐름" width="620">

    `D`는 은닉 차원, `M`은 중간 차원입니다. gate와 up은 서로 다른 학습 가능한 특징 조합입니다.
    `*`는 같은 위치·같은 차원끼리 곱하고, `@`는 행렬 곱입니다.
    SiLU는 $z\,\mathrm{sigmoid}(z)$로, 음수도 나올 수 있고 여러 값의 합이 1일 필요도 없습니다.
    **이 gate는 MoE에서 전문가를 고르는 softmax router와 다릅니다.**

    **예측:** D=4, M=8이면 세 가중치의 shape와 총 매개변수 수는 무엇일까요?
    원본은 bias가 없어 `gate: [M,D]`, `up: [M,D]`, `down: [D,M]`, 합계 `3DM=96`개입니다.
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
        mo.md(f"입력 `{tuple(sample_x.shape)}` → 중간 `{tuple(gated_features.shape)}` → 출력 `{tuple(manual_ffn.shape)}`. 원본 출력과 일치합니다."),
        mo.ui.table([
            {"중간 차원": i, "gate 선형값": float(gate_values[0, 0, i]), "SiLU(gate)": float(activated_gate[0, 0, i]),
             "up 값": float(up_values[0, 0, i]), "곱한 값": float(gated_features[0, 0, i])}
            for i in range(8)
        ], selection=None),
        mo.md("**관찰:** 첫 배치 첫 토큰의 8개 중간 특징입니다. 한 차원의 곱을 손으로 확인하세요. down_proj는 이 특징들을 다시 4차원으로 조합합니다."),
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
    mo.vstack([_fig, mo.md("**해설:** SiLU에는 음수 구간도 있습니다. gate의 값을 확률이나 토큰 선택 여부로 해석하지 않습니다.")])
    return


@app.cell
def _(mo):
    mo.md("""
    ## 2. 한 토큰을 바꾸면 다른 토큰도 바뀔까?

    **예측:** 첫 배치의 두 번째 토큰만 바꾸면 첫 번째 토큰의 출력도 바뀔까요?
    FeedForward는 같은 가중치를 모든 토큰에 독립적으로 적용합니다. 가중치를 공유하지만
    이 연산 안에서 토큰끼리 정보를 주고받지는 않습니다. Attention과 비교할 핵심입니다.
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
    원본 FeedForward는 특징 변환만 합니다. 블록의 뒷부분에서
    `h + FeedForward(RMSNorm(h))`를 계산합니다. h는 Attention과 첫 잔차 연결을 지난 값입니다.
    FFN이 위치를 직접 섞지 않아도, 입력 h에는 이미 다른 토큰의 문맥이 들어 있습니다.
    FFN 출력의 마지막 차원은 입력과 같은 D이므로 h에 더할 수 있습니다. 아래에서는 원본 블록 전체와 비교합니다.
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
        mo.md(f"직접 조립한 블록과 원본 출력이 같습니다. h·FFN·잔차 출력 shape는 모두 `{tuple(block_hidden.shape)}`입니다. 잔차 연결은 FeedForward 바깥에서 수행합니다."),
        mo.md("**예측:** 배치 0의 위치 1을 바꾸면 위치 2도 바뀔까요? 아래 두 경로의 같은 위치를 비교하세요."),
        mo.ui.table(context_change_rows, selection=None),
        mo.md("**해설:** FFN 단독에서는 위치 1만 달라집니다. 전체 블록은 Attention으로 위치 2도 달라집니다. 위치 0은 미래를 볼 수 없어 유지됩니다. 'FFN이 위치를 섞지 않는다'는 말이 '문맥을 활용하지 않는다'는 뜻은 아닙니다."),
    ])
    return


@app.cell
def _(mo):
    mo.md("""
    ## 4. 숫자 특징 변환을 실제로 학습하기

    이제 가상 입력 x의 네 성분으로 `목표=[x0*x1, SiLU(x2), x0-x3, x1*x2]`를 만듭니다.
    다음 토큰 예측과 달리 연속값 회귀이며, 오차는 평균제곱오차(MSE)입니다.
    전체 80개 입력 중 64개로 학습하고 나머지 16개는 optimizer에 넣지 않습니다.
    **예측:** 중간 차원을 늘리면 매개변수 수는 어떻게 바뀔까요? 검증 오차도 반드시 줄까요?

    `zero_grad`로 기울기를 비우고, `backward`로 기울기를 계산한 뒤 `step`으로 갱신합니다.
    `eval()`은 동작 모드이며, `no_grad()`가 gradient 기록을 끕니다. 매번 새 모델로 시작합니다.
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
        mo.md("**관찰·해설:** 세 projection 모두 gradient를 받습니다. 학습 오차 감소와 검증 오차는 별도로 읽으세요. 이 숫자 과제의 결과가 언어 모델 성능을 뜻하지는 않습니다."),
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### 4-1. 왜 선형층 하나 대신 SwiGLU를 쓸까?

    **예측:** x0*x1처럼 입력 성분의 곱을 선형 변환 하나로 항상 표현할 수 있을까요?
    선형층은 가중합만 합니다. 활성화 없이 두 층을 연결해도
    $W_2(W_1x)=(W_2W_1)x$이므로 여전히 선형 변환입니다.
    아래는 같은 학습·검증 데이터, step, optimizer와 학습률로 네 구조를 비교합니다.

    | 구조 | 계산 | 매개변수 수(D=4) |
    | --- | --- | --- |
    | Linear | x → Linear | D²=16 |
    | TwoLinear | x → up → down (gate·활성화 없음) | 2DM |
    | SiLU MLP | x → up → SiLU → down (곱셈 gate 없음) | 2DM |
    | SwiGLU | down(SiLU(gate(x)) * up(x)) | 3DM |

    마지막 세 구조의 up/down은 같은 초기 가중치에서 시작합니다. 사용되지 않는 gate는 비교 모델에 넣지 않습니다.
    모델별 매개변수 수가 다르므로 동일 용량 비교는 아닙니다. 이 숫자 과제의 유한한 학습 결과를 관찰합니다.
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
        mo.md("**관찰·해설:** 선형 모델도 목표의 일부를 맞히지만, 곱셈·비선형 특징을 표현하는 방식은 다릅니다. 학습·검증 MSE와 매개변수 수를 함께 읽으세요. 한 seed의 결과로 모든 과제의 우열을 결론 내리지 않습니다."),
    ])
    return


@app.cell
def _(mo):
    mo.md("""
    ## 5. 직접 바꿔 볼 과제와 해설

    1. M을 8/16/32로 바꾸기 전에 매개변수 수를 쓰세요. 실행 후 학습·검증 MSE를 각각 기록하세요.
       **기준:** 개수는 `3DM`에 맞아야 합니다. 더 큰 M이 검증 오차를 항상 줄이지는 않습니다.
    2. 4-1절에서 TwoLinear와 SwiGLU의 오차를 비교하세요. 두 선형층을 합친 가중치의 shape는 무엇일까요?
       **해설:** `[D,M] @ [M,D] = [D,D]`이며 여전히 선형입니다. 원본 일치 검증을 깨지 않고 변형을 비교할 수 있습니다.
    3. 한 토큰 수정 실험에서 위치 두 곳을 바꾸면 어디의 출력이 달라질까요?
       **해설:** FFN 단독은 바꾼 위치들만 달라집니다. 전체 블록은 문맥을 섞으므로 뒤 위치도 달라질 수 있습니다. 3절 표와 비교하세요.
    4. 왜 down_proj가 D차원으로 돌아와야 할까요? **해설:** 잔차 입력과 같은 shape로 더하기 위해서입니다.

    **다음 실습:** `minimind_moe.py`에서는 이런 FeedForward 여러 개 중 일부를 토큰마다 선택합니다.
    """)
    return


if __name__ == "__main__":
    app.run()
