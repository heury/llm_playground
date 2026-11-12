import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import copy
    import sys
    from pathlib import Path
    import marimo as mo
    import matplotlib.pyplot as plt
    import torch
    import torch.nn.functional as F

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "projects" / "minimind"))
    from model.model_minimind import MiniMindConfig, MiniMindForCausalLM, MOEFeedForward
    torch.set_num_threads(2)
    return F, MOEFeedForward, MiniMindConfig, MiniMindForCausalLM, copy, mo, plt, torch


@app.cell
def _(mo):
    mo.md("""
    # MiniMind MoE: 토큰마다 전문가를 선택하기

    **목표:** 원본 `MOEFeedForward`를 따라 계산합니다.

    - router → top-k → 전문가 실행 → 가중합
    - 보조 손실과 gradient
    - 작은 언어 모델 학습

    먼저 `minimind_feedforward.py`의 SwiGLU를 보면 좋습니다.
    CPU로 실행하며 다운로드는 없습니다.

    **기호:** `B` 배치, `T` 토큰 수, `D` 은닉 차원, `E` 전문가 수, `k` 토큰당 선택 수.

    **MoE가 바꾸는 곳:** 블록의 **FeedForward**입니다. Attention은 그대로입니다.
    전문가는 각각 별도의 FeedForward입니다. 입력·출력 shape는 같습니다.
    "전문가"라는 이름이 역할을 정해 주지는 않습니다.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## 1. 작은 토큰 특징과 읽을 수 있는 router

    - **Dense FFN:** 모든 토큰에 같은 FFN 하나.
    - **MoE:** router가 토큰마다 점수를 매겨 전문가 일부를 고릅니다.

    아래 숫자는 직접 정한 특징입니다. router 가중치도 직접 정했습니다.

    - 전문가 0: 첫 성분이 크면 점수가 높음
    - 전문가 1: 첫 성분이 작으면 점수가 높음
    - 전문가 2: 두 번째 성분이 크면 점수가 높음

    관찰용 규칙일 뿐, 학습된 전문 분야가 아닙니다.

    **예측:** `[2, 0.1, ...]`와 `[-2, 0.2, ...]`는 어느 전문가를 먼저 고를까요?
    softmax로 E개 확률(합 1)을 만들고, top-k로 k개를 고릅니다.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    # ▸ 실습 설정 위젯 — 실행하면 슬라이더·선택 상자가 나타납니다 (marimo 표시 코드)
    experts_per_token = mo.ui.dropdown([1, 2], value=2, label="고정 router 관찰: k")
    normalize_routes = mo.ui.checkbox(value=True, label="고정 router 관찰: 가중치 정규화")
    mo.hstack([experts_per_token, normalize_routes])
    return experts_per_token, normalize_routes


@app.cell
def _(MOEFeedForward, MiniMindConfig, experts_per_token, normalize_routes, torch):
    moe_config = MiniMindConfig(
        hidden_size=4, intermediate_size=8, moe_intermediate_size=8,
        num_attention_heads=2, num_key_value_heads=1,
        num_experts=3, num_experts_per_tok=experts_per_token.value,
        norm_topk_prob=normalize_routes.value, router_aux_loss_coef=0.01, use_moe=True,
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(31)
        routed_moe = MOEFeedForward(moe_config).train()
    with torch.no_grad():
        routed_moe.gate.weight.copy_(torch.tensor([
            [2., 0., 0., 0.], [-2., 0., 0., 0.], [0., 2., 0., 0.],
        ]))
    route_inputs = torch.tensor([
        [[2., .1, 0., 1.], [-2., .2, 1., 0.], [.1, 2., 0., -1.]],
        [[1., .3, 1., 0.], [-1., .4, 0., 1.], [.2, 1.4, -1., 0.]],
    ])
    return (moe_config, route_inputs, routed_moe,)


@app.cell(hide_code=True)
def _(experts_per_token, mo, route_inputs):
    # ▸ 결과 설명 — 실행하면 계산 값이 들어간 설명이 나타납니다 (marimo 표시 코드)
    mo.md(f"입력 `{tuple(route_inputs.shape)}` → 토큰별 펼치기 `[6,4]` → router 점수 `[6,3]`. k={experts_per_token.value}, 전문가 수 E=3입니다.")
    return


@app.cell
def _(F, moe_config, route_inputs, routed_moe, torch):
    with torch.no_grad():
        flat_tokens = route_inputs.reshape(-1, 4)
        router_logits = routed_moe.gate(flat_tokens)
        router_probabilities = F.softmax(router_logits, dim=-1)
        selected_probabilities, selected_experts = torch.topk(
            router_probabilities, k=moe_config.num_experts_per_tok, dim=-1, sorted=False,
        )
        if moe_config.norm_topk_prob:
            combine_weights = selected_probabilities / selected_probabilities.sum(-1, keepdim=True)
        else:
            combine_weights = selected_probabilities
        # top-1 정규화에서는 순전파 값은 1입니다. gradient 차이는 4절에서 별도로 봅니다.
        torch.testing.assert_close(router_probabilities.sum(-1), torch.ones(6))
    return (
        combine_weights,
        flat_tokens,
        router_logits,
        router_probabilities,
        selected_experts,
        selected_probabilities,
    )


@app.cell(hide_code=True)
def _(combine_weights, mo, moe_config, router_logits, router_probabilities, selected_experts, selected_probabilities):
    # ▸ 결과 표시 — 실행하면 표·그림이 나타납니다 (marimo 표시 코드)
    mo.vstack([
        mo.ui.table([
            {"토큰 위치": i, "전문가": e, "router 점수": float(router_logits[i, e]),
             "전체 softmax 확률": float(router_probabilities[i, e]), "선택 여부": bool((selected_experts[i] == e).any())}
            for i in range(6) for e in range(3)
        ], selection=None),
        mo.ui.table([
            {"토큰 위치": i, "선택 슬롯": j, "선택 전문가": int(selected_experts[i, j]),
             "선택 전 확률": float(selected_probabilities[i, j]), "결합 가중치": float(combine_weights[i, j])}
            for i in range(6) for j in range(moe_config.num_experts_per_tok)
        ], selection=None),
        mo.md("**관찰:** 정규화하면 선택된 k개의 가중치 합이 1입니다. 끄면 확률을 그대로 써서 합이 1보다 작을 수 있습니다. `sorted=False`라서 슬롯 순서가 확률 순위와 다를 수 있습니다."),
    ])
    return


@app.cell
def _(mo):
    mo.md("""
    ## 2. 선택된 전문가 출력의 가중합

    **예측:** k=2이면 첫 토큰의 출력은 전문가 두 개의 출력과 어떤 관계일까요?
    전문가는 자기를 고른 토큰에만 실행됩니다. 결과는 가중합해서 원래 위치로 돌려놓습니다.
    원본은 `index_add_`를 씁니다. 여기서는 반복문으로 같은 결과를 만듭니다.
    """)
    return


@app.cell
def _(combine_weights, flat_tokens, route_inputs, routed_moe, selected_experts, torch):
    with torch.no_grad():
        manual_moe_output = torch.zeros_like(flat_tokens)
        first_token_rows = []
        for _token in range(flat_tokens.shape[0]):
            for _slot in range(selected_experts.shape[1]):
                _expert = int(selected_experts[_token, _slot])
                _output = routed_moe.experts[_expert](flat_tokens[_token:_token+1])[0]
                _contribution = combine_weights[_token, _slot] * _output
                manual_moe_output[_token] += _contribution
                if _token == 0:
                    first_token_rows.append({"첫 토큰의 전문가": _expert, "가중치": float(combine_weights[_token, _slot]),
                                  "전문가 출력": str(_output.tolist()), "가중치 × 출력": str(_contribution.tolist())})
        _actual = routed_moe(route_inputs)
        routing_max_error = float((manual_moe_output.reshape_as(_actual) - _actual).abs().max())
        torch.testing.assert_close(manual_moe_output.reshape_as(_actual), _actual)
    return (first_token_rows, manual_moe_output, routing_max_error,)


@app.cell(hide_code=True)
def _(first_token_rows, manual_moe_output, mo, routing_max_error):
    # ▸ 결과 표시 — 실행하면 표·그림이 나타납니다 (marimo 표시 코드)
    mo.vstack([
        mo.ui.table(first_token_rows, selection=None),
        mo.md(f"직접 가중합한 첫 토큰: `{manual_moe_output[0].tolist()}`. 원본과의 최대 오차 `{routing_max_error:.2e}`. 입출력 shape는 같습니다."),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 2-1. 같은 입력을 Dense와 MoE에 넣기

    Dense 기준은 expert 0과 같은 FFN입니다.
    MoE는 shape는 그대로 두고, 계산 경로와 가중치 보유량을 바꿉니다.
    아래 전문가별 출력을 위의 선택·가중합 표와 맞춰 보세요.
    """)
    return


@app.cell
def _(flat_tokens, route_inputs, routed_moe, torch):
    with torch.no_grad():
        _dense = routed_moe.experts[0](route_inputs)
        _mixed = routed_moe(route_inputs)
        _expert_count = sum(p.numel() for p in routed_moe.experts[0].parameters())
        dense_moe_rows = [
            {"경로": "Dense (expert 0과 같은 가중치)", "출력 shape": str(tuple(_dense.shape)),
             "전체 보유량": _expert_count, "첫 토큰 출력": str(_dense[0, 0].tolist())},
            {"경로": "MoE (선택 전문가 가중합)", "출력 shape": str(tuple(_mixed.shape)),
             "전체 보유량": sum(p.numel() for p in routed_moe.parameters()), "첫 토큰 출력": str(_mixed[0, 0].tolist())},
        ]
        all_expert_rows = [
            {"전문가": i, "같은 첫 토큰의 출력": str(expert(flat_tokens[:1])[0].tolist())}
            for i, expert in enumerate(routed_moe.experts)
        ]
    return (all_expert_rows, dense_moe_rows,)


@app.cell(hide_code=True)
def _(all_expert_rows, dense_moe_rows, mo):
    # ▸ 결과 표시 — 실행하면 표·그림이 나타납니다 (marimo 표시 코드)
    mo.vstack([
        mo.ui.table(dense_moe_rows, selection=None),
        mo.ui.table(all_expert_rows, selection=None),
        mo.md("**해설:** 이 표를 위해서만 모든 전문가를 실행했습니다. 실제 MoE는 선택된 전문가만 씁니다. 구조 비교이지, 성능·속도 비교가 아닙니다."),
    ])
    return


@app.cell
def _(mo):
    mo.md("""
    ## 3. 전문가 사용량과 보조 손실

    **문제:** 한 전문가만 계속 선택되면 나머지는 거의 학습되지 않습니다.

    **보조 손실:** 선택이 한쪽으로 몰리지 않게 하는 신호입니다.
    균등 사용을 보장하지는 않습니다. 전문 분야를 정해 주지도 않습니다.

    **수식:** `보조 손실 = 계수 × E × sum((선택 횟수/N) × 평균 softmax 확률)`.

    - N은 토큰 수입니다.
    - 토큰마다 k개를 고르므로 총 선택 횟수는 N*k입니다.
    - k를 바꾸면 보조 손실의 기준 크기도 바뀝니다.
    """)
    return


@app.cell
def _(moe_config, plt, route_inputs, routed_moe, router_probabilities, selected_experts, torch):
    with torch.no_grad():
        expert_counts = torch.bincount(selected_experts.flatten(), minlength=moe_config.num_experts)
        _mean_probability = router_probabilities.mean(0)
        manual_aux_loss = (expert_counts / selected_experts.shape[0] * _mean_probability).sum() * moe_config.num_experts * moe_config.router_aux_loss_coef
        routed_moe.train()
        routed_moe(route_inputs)
        torch.testing.assert_close(manual_aux_loss, routed_moe.aux_loss)
        routed_moe.eval()
        routed_moe(route_inputs)
        assert float(routed_moe.aux_loss) == 0
        routed_moe.train()
    assert int(expert_counts.sum()) == route_inputs.shape[0] * route_inputs.shape[1] * moe_config.num_experts_per_tok
    routing_fig, _ax = plt.subplots(figsize=(5, 3))
    _ax.bar(range(3), expert_counts.numpy())
    _ax.set(xticks=[0, 1, 2], xlabel="Expert", ylabel="Selected token assignments", title="Routing counts / toy features")
    routing_fig.tight_layout()
    _expert_parameters = sum(p.numel() for p in routed_moe.experts[0].parameters())
    _router_parameters = routed_moe.gate.weight.numel()
    moe_parameter_summary = {
        "Dense FFN 하나": _expert_parameters,
        "MoE 전체 보유량": sum(p.numel() for p in routed_moe.parameters()),
        "선택 경로 가중치 수(토큰당 참고값)": _router_parameters + moe_config.num_experts_per_tok * _expert_parameters,
    }
    return (expert_counts, manual_aux_loss, moe_parameter_summary, routing_fig,)


@app.cell(hide_code=True)
def _(manual_aux_loss, mo, moe_parameter_summary, routing_fig):
    # ▸ 결과 표시 — 실행하면 표·그림이 나타납니다 (marimo 표시 코드)
    mo.vstack([
        routing_fig,
        mo.md(f"직접 계산한 보조 손실: `{float(manual_aux_loss):.6f}`. 원본과 같습니다. eval 모드에서 원본은 0을 씁니다."),
        mo.ui.table([moe_parameter_summary], selection=None),
        mo.md("**해설:** 일부만 선택해도 모든 전문가의 가중치는 메모리에 있습니다. 선택 경로 수는 속도 측정이 아닙니다. 이 구현은 반복문 등 추가 비용이 있습니다. Dense보다 빠르다고 단정하지 마세요."),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 3-1. 균등 선택과 편중 선택의 손실 비교

    **예측:** 토큰 세 개가 모두 같은 전문가를 고르면 어떤 값이 커질까요?
    직접 만든 top-1 확률 표 두 개로 비교합니다. 학습 결과는 아닙니다.
    """)
    return


@app.cell
def _(torch):
    _balanced = torch.tensor([[.9, .05, .05], [.05, .9, .05], [.05, .05, .9]])
    _collapsed = torch.tensor([[.9, .05, .05]]).repeat(3, 1)
    balance_examples = []
    for _name, _p in [("균등 선택", _balanced), ("전문가 0에 집중", _collapsed)]:
        _counts = torch.bincount(_p.argmax(-1), minlength=3)
        _balance = 3 * (_counts / 3 * _p.mean(0)).sum()
        balance_examples.append({"예제": _name, "선택 횟수": str(_counts.tolist()),
                                 "평균 확률": str([round(float(v), 3) for v in _p.mean(0)]),
                                 "계수 없는 균형 항": float(_balance), "계수 0.01의 손실": float(_balance * .01)})
    assert balance_examples[0]["계수 0.01의 손실"] < balance_examples[1]["계수 0.01의 손실"]
    return (balance_examples,)


@app.cell(hide_code=True)
def _(balance_examples, mo):
    # ▸ 결과 표시 — 실행하면 표·그림이 나타납니다 (marimo 표시 코드)
    mo.vstack([
        mo.ui.table(balance_examples, selection=None),
        mo.md("**해설:** 몰린 쪽의 손실이 더 큽니다. 하지만 실제 학습에서는 선택이 불연속적으로 바뀌고 주 손실도 함께 줄입니다. 그래서 균등 사용이 보장되지는 않습니다. 5절에서 계수 0과 0.01을 비교합니다."),
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 4. top-1의 값은 1인데 router가 학습되는 이유

    **문제:** $p/p$는 항상 1입니다. 그래서 p의 기울기가 0입니다. router가 학습되지 않습니다.

    **해결:** 정규화된 top-1은 `p - p.detach() + 1`을 씁니다(straight-through).

    - 순전파 값은 1입니다.
    - 역전파에서 `detach()` 항은 상수입니다. 그래서 p의 기울기가 남습니다.

    **gradient 범위:** 원본은 이 p를 계산할 때 입력도 detach합니다.

    - router 가중치는 학습됩니다.
    - router를 거쳐 입력으로 가는 gradient는 막힙니다.
    - 전문가를 거쳐 입력으로 가는 gradient는 그대로입니다.

    **예측:** 선택되지 않은 전문가의 gradient는 None일까요, 0일까요?
    원본은 `0 * parameter.sum()`으로 미선택 전문가도 그래프에 남깁니다. 그래서 0입니다.
    아래에서는 보조 손실 계수를 0으로 두고 과제 손실만 봅니다.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 4-1. scalar 하나로 먼저 비교하기

    **예측:** 출력이 같으면 gradient도 같을까요? p=0.7에서 두 계산을 비교합니다.
    """)
    return


@app.cell
def _(torch):
    _p = torch.tensor(.7, requires_grad=True)
    _naive = _p / _p
    _naive_gradient = torch.autograd.grad(_naive, _p)[0]
    _straight = _p - _p.detach() + 1
    _straight_gradient = torch.autograd.grad(_straight, _p)[0]
    torch.testing.assert_close(_naive, _straight)
    assert float(_naive_gradient) == 0 and float(_straight_gradient) == 1
    scalar_gradient_rows = [
        {"계산식": "p/p", "순전파 값": float(_naive.detach()), "p의 gradient": float(_naive_gradient)},
        {"계산식": "p-p.detach()+1", "순전파 값": float(_straight.detach()), "p의 gradient": float(_straight_gradient)},
    ]
    return (scalar_gradient_rows,)


@app.cell(hide_code=True)
def _(mo, scalar_gradient_rows):
    # ▸ 결과 표시 — 실행하면 표·그림이 나타납니다 (marimo 표시 코드)
    mo.vstack([
        mo.ui.table(scalar_gradient_rows, selection=None),
        mo.md("**해설:** 출력은 둘 다 1입니다. gradient는 0과 1로 다릅니다. detach한 값은 상수로 취급되기 때문입니다. 이는 이 구현의 방식이며 모든 MoE의 규칙은 아닙니다."),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    # ▸ 실습 설정 위젯 — 실행하면 슬라이더·선택 상자가 나타납니다 (marimo 표시 코드)
    top1_input_sign = mo.ui.dropdown([1, -1], value=1, label="top-1 실험: 입력 첫 성분의 부호")
    top1_input_sign
    return (top1_input_sign,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 4-2. 실제 선택 결과에 따라 검증하기
    """)
    return


@app.cell
def _(F, MOEFeedForward, MiniMindConfig, top1_input_sign, torch):
    def probe_top1_routes(x):
        config = MiniMindConfig(hidden_size=4, intermediate_size=8, num_experts=3,
                                num_attention_heads=2, num_key_value_heads=1, use_moe=True,
                                num_experts_per_tok=1, norm_topk_prob=True, router_aux_loss_coef=0)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(32)
            model = MOEFeedForward(config).train()
        with torch.no_grad():
            model.gate.weight.copy_(torch.tensor([[3., 0., 0., 0.], [-3., 0., 0., 0.], [0., 1., 0., 0.]]))
            flat = x.reshape(-1, 4)
            selected = model.gate(flat).argmax(-1)
            expected = torch.stack([model.experts[int(e)](flat[i:i+1])[0] for i, e in enumerate(selected)])
        prediction = model(x)
        torch.testing.assert_close(prediction, expected.reshape_as(prediction))
        F.mse_loss(prediction, torch.ones_like(prediction)).backward()
        counts = torch.bincount(selected, minlength=3)
        rows = [{"대상": "router", "선택 토큰 수": None, "gradient 존재": model.gate.weight.grad is not None,
                 "gradient norm": float(model.gate.weight.grad.norm())}]
        for i, expert in enumerate(model.experts):
            gradients = [p.grad for p in expert.parameters()]
            assert all(g is not None for g in gradients)
            norm = sum(float(g.square().sum()) for g in gradients) ** 0.5
            rows.append({"대상": f"expert {i}", "선택 토큰 수": int(counts[i]), "gradient 존재": True, "gradient norm": norm})
            if counts[i] == 0:
                assert norm == 0
        return rows, selected.tolist()

    _x = torch.tensor([[[2., .1, 1., 0.], [1.5, .2, 0., 1.]]])
    _x[..., 0] *= top1_input_sign.value
    moe_gradient_rows, probe_selected_experts = probe_top1_routes(_x)
    return (moe_gradient_rows, probe_selected_experts, probe_top1_routes,)


@app.cell(hide_code=True)
def _(mo, moe_gradient_rows, probe_selected_experts):
    # ▸ 결과 표시 — 실행하면 표·그림이 나타납니다 (marimo 표시 코드)
    mo.vstack([
        mo.md(f"두 토큰의 선택 전문가: `{probe_selected_experts}`. 검증은 실제 선택 결과를 기준으로 합니다."),
        mo.ui.table(moe_gradient_rows, selection=None),
        mo.md("**관찰:** 출력은 선택된 전문가의 출력과 같습니다. 선택된 전문가와 router에 gradient가 있는지 보세요. 선택되지 않은 전문가는 0입니다. 선택된 쪽도 상황에 따라 0일 수 있습니다. 실제 가중치 변화는 optimizer 설정에 따라 또 다릅니다."),
    ])
    return


@app.cell
def _(mo):
    mo.md("""
    ## 5. 작은 MoE 언어 모델 학습: 주 손실 + 보조 손실

    ID 문장 두 개를 학습합니다: `[bos,3,4,eos]`, `[bos,5,6,eos]`.
    토크나이저 없이 MoE 학습 경로만 봅니다.
    `use_moe=True`이면 블록의 mlp가 MOEFeedForward가 됩니다.

    **예측:** `outputs.loss`만 backward하면 보조 손실도 들어갈까요?
    **아닙니다.** `loss`는 cross entropy뿐입니다. **aux_loss는 별도 필드**입니다.
    그래서 aux_loss를 직접 더합니다. 계수는 이미 곱해져 있습니다.

    **1~3절과 별개의 실험입니다.**

    - 같은 초기 LM을 복사해 계수 0과 0.01로 각각 학습합니다.
    - 아래 LM 설정만 이 학습에 영향을 줍니다. 앞 절 체크박스는 무관합니다.
    - 선택 횟수는 EOS 포함 8위치로 셉니다.
    - 다음 토큰 손실은 6위치로 계산합니다(문장마다 마지막 제외).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    # ▸ 실습 설정 위젯 — 실행하면 슬라이더·선택 상자가 나타납니다 (marimo 표시 코드)
    lm_experts_per_token = mo.ui.dropdown([1, 2], value=2, label="LM 재학습: 토큰당 전문가 k")
    lm_normalize_routes = mo.ui.checkbox(value=True, label="LM 재학습: 가중치 정규화")
    mo.hstack([lm_experts_per_token, lm_normalize_routes])
    return lm_experts_per_token, lm_normalize_routes


@app.cell
def _(F, MiniMindConfig, MiniMindForCausalLM, copy, lm_experts_per_token, lm_normalize_routes, plt, torch):
    _config = MiniMindConfig(
        hidden_size=16, intermediate_size=32, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=1, vocab_size=8,
        max_position_embeddings=16, flash_attn=False, use_moe=True,
        num_experts=3, num_experts_per_tok=lm_experts_per_token.value,
        norm_topk_prob=lm_normalize_routes.value, router_aux_loss_coef=0.01,
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(33)
        _initial_model = MiniMindForCausalLM(_config).train()
    _ids = torch.tensor([[1, 3, 4, 2], [1, 5, 6, 2]])

    def snapshot_lm_routes(model, ids):
        # 첫 블록의 gate 출력을 관찰합니다. top-1에서는 gate가 두 번 호출되므로 첫 호출만 기록합니다.
        captured = []

        def capture(module, args, output):
            if not captured:
                captured.append(output.detach())

        gate = model.model.layers[0].mlp.gate
        handle = gate.register_forward_hook(capture)
        was_training = model.training
        try:
            model.eval()
            with torch.no_grad():
                model(ids)
        finally:
            handle.remove()
            model.train(was_training)
        probabilities = captured[0].softmax(-1)
        indices = probabilities.topk(model.config.num_experts_per_tok, dim=-1).indices
        counts = torch.bincount(indices.flatten(), minlength=3)
        mean_probability = probabilities.mean(0)
        balance = 3 * (counts / probabilities.shape[0] * mean_probability).sum()
        assert int(counts.sum()) == ids.numel() * model.config.num_experts_per_tok
        return {"counts": counts.tolist(), "mean_probability": mean_probability.tolist(), "unscaled_balance": float(balance)}

    aux_comparison_rows = []
    routing_snapshot_rows = []
    lm_comparison_histories = {}
    _snapshots = {}
    for _coefficient in [0.0, 0.01]:
        _model = copy.deepcopy(_initial_model).train()
        _model.config.router_aux_loss_coef = _coefficient
        _model.model.layers[0].mlp.config.router_aux_loss_coef = _coefficient
        for _name, _weight in _model.state_dict().items():
            torch.testing.assert_close(_weight, _initial_model.state_dict()[_name], rtol=0, atol=0)
        _before = snapshot_lm_routes(_model, _ids)
        _optimizer = torch.optim.AdamW(_model.parameters(), lr=0.01, weight_decay=0)
        _history = []
        for _step in range(80):
            _optimizer.zero_grad(set_to_none=True)
            _out = _model(_ids, labels=_ids)
            if _step == 0:
                _ce = F.cross_entropy(_out.logits[:, :-1].reshape(-1, 8), _ids[:, 1:].reshape(-1))
                torch.testing.assert_close(_out.loss, _ce)
            _total = _out.loss + _out.aux_loss
            assert torch.isfinite(_total)
            if _coefficient == 0:
                assert float(_out.aux_loss.detach()) == 0
            _total.backward()
            _optimizer.step()
            _history.append({"step": _step + 1, "main": float(_out.loss.detach()),
                             "aux": float(_out.aux_loss.detach()), "total": float(_total.detach())})
        _model.eval()
        with torch.no_grad():
            _final = _model(_ids, labels=_ids)
            assert float(_final.aux_loss) == 0
            _predictions = _final.logits[:, :-1].argmax(-1)
        _after = snapshot_lm_routes(_model, _ids)
        _snapshots[_coefficient] = (_before, _after)
        assert float(_final.loss) < _history[0]["main"]
        aux_comparison_rows.append({
            "보조 손실 계수": _coefficient, "초기 주 손실": _history[0]["main"],
            "최종 주 손실": float(_final.loss), "정확도": float((_predictions == _ids[:, 1:]).float().mean()),
            "마지막 train aux": _history[-1]["aux"], "선택 횟수(전)": str(_before["counts"]),
            "선택 횟수(후)": str(_after["counts"]), "계수 없는 균형 항(후)": _after["unscaled_balance"],
        })
        for _stage, _snapshot in [("학습 전", _before), ("학습 후", _after)]:
            for _expert in range(3):
                routing_snapshot_rows.append({"계수": _coefficient, "시점": _stage, "전문가": _expert,
                                               "선택 횟수": _snapshot["counts"][_expert],
                                               "평균 확률": _snapshot["mean_probability"][_expert],
                                               "계수 없는 균형 항": _snapshot["unscaled_balance"]})
        lm_comparison_histories[_coefficient] = _history
        if _coefficient == 0.01:
            trained_moe_lm = _model
            moe_training_history = _history
            moe_lm_metrics = {"initial_main_loss": _history[0]["main"], "final_eval_main_loss": float(_final.loss),
                              "next_token_accuracy": float((_predictions == _ids[:, 1:]).float().mean())}
    assert _snapshots[0.0][0] == _snapshots[0.01][0]
    lm_loss_fig, _axes = plt.subplots(1, 2, figsize=(9, 3))
    for _coefficient, _history in lm_comparison_histories.items():
        _axes[0].plot([r["main"] for r in _history], label=f"coef={_coefficient}")
        _axes[1].plot([r["aux"] for r in _history], label=f"coef={_coefficient}")
    _axes[0].set_title("Main loss / same initial weights")
    _axes[1].set_title("Weighted auxiliary loss")
    for _axis in _axes:
        _axis.set(xlabel="Step", ylabel="Loss")
        _axis.legend()
    lm_loss_fig.tight_layout()
    usage_fig, _usage_axes = plt.subplots(1, 2, figsize=(9, 3), sharey=True)
    for _axis, _coefficient in zip(_usage_axes, [0.0, 0.01]):
        _before, _after = _snapshots[_coefficient]
        _axis.bar([i - .18 for i in range(3)], _before["counts"], width=.36, label="Before")
        _axis.bar([i + .18 for i in range(3)], _after["counts"], width=.36, label="After")
        _axis.set(xticks=[0, 1, 2], xlabel="Expert", ylabel="Selected token assignments", title=f"coef={_coefficient}")
        _axis.set_ylim(0, _ids.numel() + 1)
        _axis.legend()
    usage_fig.tight_layout()
    return (
        aux_comparison_rows,
        lm_comparison_histories,
        lm_loss_fig,
        moe_lm_metrics,
        moe_training_history,
        routing_snapshot_rows,
        snapshot_lm_routes,
        trained_moe_lm,
        usage_fig,
    )


@app.cell(hide_code=True)
def _(aux_comparison_rows, lm_loss_fig, mo, moe_training_history, routing_snapshot_rows, usage_fig):
    # ▸ 결과 표시 — 실행하면 표·그림이 나타납니다 (marimo 표시 코드)
    mo.vstack([
        lm_loss_fig, mo.ui.table(aux_comparison_rows, selection=None), usage_fig,
        mo.ui.table(routing_snapshot_rows, selection=None),
        mo.ui.table(moe_training_history[:3], selection=None),
        mo.md("**관찰:** 보조 손실 계수만 다르고 나머지는 같습니다(80 step). 계수 0에서 aux=0은 균형을 뜻하지 않습니다. 균형 항, 사용량, 주 손실을 함께 보세요. 어느 쪽이 꼭 더 낫다고 가정하지 마세요. 선택 합은 8*k입니다. BOS 다음 정답이 두 개라 손실이 0이 안 돼도 됩니다."),
    ])
    return


@app.cell
def _(mo):
    mo.md("""
    ## 6. 직접 바꿔 볼 과제와 해설

    1. k=1/2에서 토큰별 선택 전문가와 가중치를 예측하고 표로 확인하세요.
       **기준:** 정규화하면 가중치 합은 1입니다. 총 선택 횟수는 N*k입니다.
    2. 1절에서 정규화를 끄면 출력과 보조 손실 중 무엇이 바뀔까요?
       **해설:** 출력 가중치는 바뀌고, 보조 손실은 그대로입니다. 5절 LM과는 무관합니다.
    3. 4-2절에서 부호를 +1/-1로 바꿔 선택 전문가와 gradient를 기록하세요.
       **기준:** +1이면 expert 0, -1이면 expert 1. 두 경우 모두 검증을 통과해야 합니다.
    4. 5절에서 계수 0/0.01의 main/aux, 사용량, 평균 확률, 균형 항을 비교하세요.
       **기준:** 초기 사용량은 같습니다. 계수 0이면 aux도 0입니다. 더 균등하거나 정확할 거라 기대하지 마세요.
    5. p/p와 straight-through는 출력이 같은데 gradient는 왜 다를까요?
       **해설:** detach한 항은 역전파에서 상수입니다.

    **기록할 것:** k, 정규화 여부, 전문가별 선택 횟수, 보조 손실, 전체 매개변수, LM 손실.

    **이 실습에서 다루지 않는 것:** GPU 분산 실행, 통신, 속도 비교, 전문가 역할 평가.

    **다음 실습:** 구조 파트 끝입니다. `minimind_pretraining.py`에서 학습을 시작합니다.
    뒤 노트북은 Dense 모델을 씁니다. trainer가 `loss + aux_loss`를 더하는 이유는 이 노트북 5절에 있습니다.
    """)
    return


if __name__ == "__main__":
    app.run()
