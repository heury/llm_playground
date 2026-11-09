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

    **목표:** 원본 `MOEFeedForward`로 router → top-k → 전문가 실행 → 가중합을 계산하고,
    보조 손실과 gradient, 실제 언어 모델 학습까지 확인합니다. 먼저 `minimind_feedforward.py`의
    SwiGLU를 읽으면 좋습니다. 이 노트북도 CPU에서 단독 실행하며 다운로드는 없습니다.

    `B`는 배치, `T`는 토큰 수, `D`는 은닉 차원, `E`는 전문가 수, `k`는 토큰당 선택 수입니다.
    Python 함수·shape·행렬 곱을 알고 시작하세요. **예측 → 실행 → 관찰 → 설명** 순서로 진행합니다.
    `@app.cell`과 인자·return은 셀 간 데이터 전달이고, `_` 변수는 셀 내부 계산입니다.
    설정을 바꾸면 새 모델에서 다시 실행합니다.

    MoE는 여기서 **블록의 FeedForward 부분**을 바꿉니다. Attention이 전문가로 바뀌는 것은 아닙니다.
    각 전문가는 동일한 입력·출력 shape의 별도 FeedForward이며, 이름만으로 역할이 정해지지 않습니다.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## 1. 작은 토큰 특징과 읽을 수 있는 router

    Dense FFN은 모든 토큰에 같은 FFN을 적용합니다. MoE는 router가 토큰별 점수를 계산해
    전문가 일부를 고릅니다. 아래 숫자는 단어 임베딩 대신 사람이 정한 특징입니다.
    전문가 0은 첫 성분이 클 때, 1은 첫 성분이 작을 때, 2는 두 번째 성분이 클 때 점수가 높습니다.
    이 규칙은 관찰용으로 router 가중치를 직접 정한 것이며, 학습된 전문 분야를 뜻하지 않습니다.

    **예측:** `[2, 0.1, ...]`와 `[-2, 0.2, ...]`는 어느 전문가를 먼저 고를까요?
    softmax는 E개 전체의 확률 합을 1로 만들고, top-k가 그중 k개를 선택합니다.
    """)
    return


@app.cell
def _(mo):
    experts_per_token = mo.ui.dropdown([1, 2], value=2, label="고정 router 관찰: k")
    normalize_routes = mo.ui.checkbox(value=True, label="고정 router 관찰: 가중치 정규화")
    mo.hstack([experts_per_token, normalize_routes])
    return experts_per_token, normalize_routes


@app.cell
def _(MOEFeedForward, MiniMindConfig, experts_per_token, mo, normalize_routes, torch):
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
    mo.md(f"입력 `{tuple(route_inputs.shape)}` → 토큰별 펼치기 `[6,4]` → router 점수 `[6,3]`. k={experts_per_token.value}, 전문가 수 E=3입니다.")
    return moe_config, route_inputs, routed_moe


@app.cell
def _(F, mo, moe_config, route_inputs, routed_moe, torch):
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
        mo.md("**관찰·해설:** 정규화하면 선택된 k개 안에서 가중치 합이 1입니다. 끄면 선택 확률을 그대로 쓰므로 합이 1보다 작을 수 있습니다. `sorted=False`이므로 선택 슬롯 순서가 확률 순위를 보장하지 않습니다."),
    ])
    return combine_weights, flat_tokens, router_probabilities, selected_experts


@app.cell
def _(mo):
    mo.md("""
    ## 2. 선택된 전문가 출력의 가중합

    **예측:** k=2이면 첫 토큰의 출력은 전문가 두 개의 출력과 어떤 관계일까요?
    각 전문가는 선택된 토큰에만 실행하고, 원래 토큰 위치에 가중합을 돌려놓습니다.
    원본은 `index_add_`로 합칩니다. 여기서는 먼저 토큰·선택 슬롯별 반복문으로 같은 결과를 만듭니다.
    """)
    return


@app.cell
def _(combine_weights, flat_tokens, mo, route_inputs, routed_moe, selected_experts, torch):
    with torch.no_grad():
        _manual = torch.zeros_like(flat_tokens)
        _rows = []
        for _token in range(flat_tokens.shape[0]):
            for _slot in range(selected_experts.shape[1]):
                _expert = int(selected_experts[_token, _slot])
                _output = routed_moe.experts[_expert](flat_tokens[_token:_token+1])[0]
                _contribution = combine_weights[_token, _slot] * _output
                _manual[_token] += _contribution
                if _token == 0:
                    _rows.append({"첫 토큰의 전문가": _expert, "가중치": float(combine_weights[_token, _slot]),
                                  "전문가 출력": str(_output.tolist()), "가중치 × 출력": str(_contribution.tolist())})
        _actual = routed_moe(route_inputs)
        routing_max_error = float((_manual.reshape_as(_actual) - _actual).abs().max())
        torch.testing.assert_close(_manual.reshape_as(_actual), _actual)
    mo.vstack([
        mo.ui.table(_rows, selection=None),
        mo.md(f"직접 가중합한 첫 토큰: `{_manual[0].tolist()}`. 원본과의 최대 오차 `{routing_max_error:.2e}`. 입력과 출력 shape는 같습니다."),
    ])
    return (routing_max_error,)


@app.cell
def _(flat_tokens, mo, moe_config, route_inputs, routed_moe, torch):
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
        _all_expert_rows = [
            {"전문가": i, "같은 첫 토큰의 출력": str(expert(flat_tokens[:1])[0].tolist())}
            for i, expert in enumerate(routed_moe.experts)
        ]
    mo.vstack([
        mo.md("### 2-1. 같은 입력을 Dense와 MoE에 넣기\n\nDense 기준은 expert 0과 같은 FFN입니다. MoE는 shape를 유지하면서 경로와 보유량을 바꿉니다. 아래 전문가별 출력 표를 위의 선택·가중합 표와 연결하세요."),
        mo.ui.table(dense_moe_rows, selection=None),
        mo.ui.table(_all_expert_rows, selection=None),
        mo.md("**해설:** 이 진단 표를 만들 때만 모든 전문가를 같은 입력에 실행했습니다. 실제 원본 MoE의 출력 계산은 선택된 전문가를 사용합니다. 구조 비교이며 학습 성능·속도 비교가 아닙니다."),
    ])
    return (dense_moe_rows,)


@app.cell
def _(mo):
    mo.md("""
    ## 3. 전문가 사용량과 보조 손실

    특정 전문가만 선택되면 나머지 전문가는 학습 기회를 적게 받습니다.
    보조 손실은 평균 확률과 선택 빈도가 한쪽으로 몰리는 것을 완화하려는 학습 신호입니다.
    균등 사용을 보장하거나 사람처럼 전문 분야를 정해 주는 것은 아닙니다.

    이 원본의 수식은 `보조 손실 = 계수 × E × sum((전문가별 선택 횟수/N) × 평균 softmax 확률)`입니다.
    N은 토큰 수입니다. k개를 고르므로 빈도 합은 k이며, 아래의 그래프도 합계 N*k입니다.
    과제에서 k를 바꾸면 이 보조 손실의 기준 크기도 달라질 수 있습니다.
    """)
    return


@app.cell
def _(mo, moe_config, plt, route_inputs, routed_moe, router_probabilities, selected_experts, torch):
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
    _fig, _ax = plt.subplots(figsize=(5, 3))
    _ax.bar(range(3), expert_counts.numpy())
    _ax.set(xticks=[0, 1, 2], xlabel="Expert", ylabel="Selected token assignments", title="Routing counts / toy features")
    _fig.tight_layout()
    _expert_parameters = sum(p.numel() for p in routed_moe.experts[0].parameters())
    _router_parameters = routed_moe.gate.weight.numel()
    moe_parameter_summary = {
        "Dense FFN 하나": _expert_parameters,
        "MoE 전체 보유량": sum(p.numel() for p in routed_moe.parameters()),
        "선택 경로 가중치 수(토큰당 참고값)": _router_parameters + moe_config.num_experts_per_tok * _expert_parameters,
    }
    mo.vstack([
        _fig,
        mo.md(f"직접 계산한 보조 손실: `{float(manual_aux_loss):.6f}`. 원본과 일치하며 eval 모드에서는 원본이 0으로 설정합니다."),
        mo.ui.table([moe_parameter_summary], selection=None),
        mo.md("**해설:** 전문가를 일부 선택해도 전체 전문가 가중치는 보유합니다. 선택 경로 수는 FLOPs나 실제 속도 측정이 아닙니다. 이 Python 구현의 반복문·분배·미선택 전문가의 그래프 유지 비용도 있어 Dense보다 빠르다고 결론 내리지 않습니다."),
    ])
    return expert_counts, manual_aux_loss, moe_parameter_summary


@app.cell
def _(mo, torch):
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
    mo.vstack([
        mo.md("### 3-1. 균등 선택과 편중 선택의 손실 비교\n\n**예측:** 토큰 세 개가 모두 같은 전문가를 선택하면 어떤 값이 커질까요? 사람이 정한 top-1 확률 표 두 개로 수식을 비교합니다. 실제 학습 결과는 아닙니다."),
        mo.ui.table(balance_examples, selection=None),
        mo.md("**해설:** 편중된 예제의 손실이 더 큽니다. 실제 학습에서 선택은 불연속적으로 바뀌고 주 손실도 함께 최적화하므로, 이 비교만으로 사용량 균등을 보장할 수는 없습니다. 5절에서는 같은 초기 LM에서 계수 0/0.01을 비교합니다."),
    ])
    return (balance_examples,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 4. top-1의 값은 1인데 router가 학습되는 이유

    보통 $p/p$는 항상 1이어서 p에 대한 기울기도 0입니다.
    이 구현의 **정규화된 top-1**은 `p - p.detach() + 1`을 씁니다.
    순전파에서는 1이지만, backward에서는 `detach()`한 항을 상수로 보아 p 쪽 기울기가 남습니다.
    이것은 일반적인 미분값을 그대로 쓰는 대신 학습용 기울기를 정하는 straight-through 방식입니다.
    원본은 이 p를 계산할 때 입력도 detach하므로 이 분기의 router 가중치는 학습되지만
    입력으로 향하는 router 경로의 gradient는 차단합니다. 전문가 경로의 입력 gradient와는 구분합니다.

    **예측:** 한 전문가만 선택된 배치에서 다른 전문가의 gradient는 None일까요, 0일까요?
    원본은 미선택 전문가도 `0 * parameter.sum()`으로 학습 그래프에 남기므로 0인 gradient가 만들어집니다.
    아래는 보조 손실 계수를 0으로 하여 과제 손실에서 오는 신호만 관찰합니다.
    """)
    return


@app.cell
def _(mo, torch):
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
    mo.vstack([
        mo.md("### 4-1. scalar 하나로 먼저 비교하기\n\n**예측:** 출력이 같으면 gradient도 같을까요? p=0.7에서 두 계산을 비교합니다."),
        mo.ui.table(scalar_gradient_rows, selection=None),
        mo.md("**해설:** 출력은 모두 1이지만 gradient는 0과 1입니다. detach한 값은 이 역전파에서 상수로 취급됩니다. 이것은 현재 원본의 top-1 처리이며 모든 MoE의 필수 규칙은 아닙니다."),
    ])
    return (scalar_gradient_rows,)


@app.cell
def _(mo):
    top1_input_sign = mo.ui.dropdown([1, -1], value=1, label="top-1 실험: 입력 첫 성분의 부호")
    top1_input_sign
    return (top1_input_sign,)


@app.cell
def _(F, MOEFeedForward, MiniMindConfig, mo, top1_input_sign, torch):
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
    mo.vstack([
        mo.md(f"### 4-2. 실제 선택 결과에 따라 검증하기\n\n두 토큰의 선택 전문가: `{probe_selected_experts}`. 부호를 바꿔도 고정 전문가 번호를 가정하지 않습니다."),
        mo.ui.table(moe_gradient_rows, selection=None),
        mo.md("**관찰·해설:** 출력은 각 토큰이 실제로 선택한 전문가 출력과 같습니다. 선택된 전문가와 router의 gradient를 확인하세요. 선택되지 않은 전문가는 0 gradient를 받습니다. 선택된 쪽도 입력·목표에 따라 gradient가 0일 수 있습니다. optimizer의 weight decay 등과 실제 가중치 변화는 별개입니다."),
    ])
    return moe_gradient_rows, probe_selected_experts, probe_top1_routes


@app.cell
def _(mo):
    mo.md("""
    ## 5. 작은 MoE 언어 모델 학습: 주 손실 + 보조 손실

    토큰 ID 문장 `[bos,3,4,eos]`, `[bos,5,6,eos]` 두 개를 학습합니다.
    실제 단어 대신 ID를 써서 MoE 학습 경로에 집중합니다. 외부 토크나이저는 필요하지 않습니다.
    `use_moe=True`이면 블록의 mlp가 원본 MOEFeedForward가 됩니다.
    **예측:** `outputs.loss`만 backward하면 보조 손실도 포함될까요?
    원본의 `loss`는 다음 토큰 cross entropy이며 **aux_loss는 별도 필드**입니다.
    아래에서는 이미 계수가 반영된 aux_loss를 한 번 더해 학습합니다. 계수를 또 곱하지 않습니다.

    **앞 절과 별개의 실험입니다.** 1~3절은 직접 정한 고정 router이고,
    여기서는 같은 초기 LM을 두 번 복사해 계수 0과 0.01로 각각 학습합니다.
    아래의 LM 설정만 학습을 제어하며, 앞 절의 정규화 체크박스를 바꾸어도 이 LM은 재학습하지 않습니다.
    재학습한 가중치가 달라지면 사용량·보조 손실도 달라질 수 있습니다.
    선택 횟수는 EOS를 포함한 입력 8위치, 다음 토큰 손실은 각 문장의 마지막을 제외한 6위치로 계산합니다.
    """)
    return


@app.cell
def _(mo):
    lm_experts_per_token = mo.ui.dropdown([1, 2], value=2, label="LM 재학습: 토큰당 전문가 k")
    lm_normalize_routes = mo.ui.checkbox(value=True, label="LM 재학습: 가중치 정규화")
    mo.hstack([lm_experts_per_token, lm_normalize_routes])
    return lm_experts_per_token, lm_normalize_routes


@app.cell
def _(F, MiniMindConfig, MiniMindForCausalLM, copy, lm_experts_per_token, lm_normalize_routes, mo, plt, torch):
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
    _fig, _axes = plt.subplots(1, 2, figsize=(9, 3))
    for _coefficient, _history in lm_comparison_histories.items():
        _axes[0].plot([r["main"] for r in _history], label=f"coef={_coefficient}")
        _axes[1].plot([r["aux"] for r in _history], label=f"coef={_coefficient}")
    _axes[0].set_title("Main loss / same initial weights")
    _axes[1].set_title("Weighted auxiliary loss")
    for _axis in _axes:
        _axis.set(xlabel="Step", ylabel="Loss")
        _axis.legend()
    _fig.tight_layout()
    _usage_fig, _usage_axes = plt.subplots(1, 2, figsize=(9, 3), sharey=True)
    for _axis, _coefficient in zip(_usage_axes, [0.0, 0.01]):
        _before, _after = _snapshots[_coefficient]
        _axis.bar([i - .18 for i in range(3)], _before["counts"], width=.36, label="Before")
        _axis.bar([i + .18 for i in range(3)], _after["counts"], width=.36, label="After")
        _axis.set(xticks=[0, 1, 2], xlabel="Expert", ylabel="Selected token assignments", title=f"coef={_coefficient}")
        _axis.set_ylim(0, _ids.numel() + 1)
        _axis.legend()
    _usage_fig.tight_layout()
    mo.vstack([
        _fig, mo.ui.table(aux_comparison_rows, selection=None), _usage_fig,
        mo.ui.table(routing_snapshot_rows, selection=None),
        mo.ui.table(moe_training_history[:3], selection=None),
        mo.md("**관찰·해설:** 같은 초기 가중치·데이터·optimizer·80 step이며 보조 손실 계수만 다릅니다. 계수 0의 aux=0이 균형 잡힌 선택을 뜻하지 않습니다. 계수 없는 균형 항, 사용량과 주 손실을 함께 비교하세요. 어느 계수가 반드시 더 균등하거나 정확하다고 가정하지 않습니다. EOS 포함 입력 8위치의 선택 합은 8*k입니다. BOS의 다음 정답이 둘이어서 전 위치 정답이나 손실 0을 요구하지 않습니다."),
    ])
    return (
        aux_comparison_rows, lm_comparison_histories, moe_lm_metrics, moe_training_history,
        routing_snapshot_rows, snapshot_lm_routes, trained_moe_lm,
    )


@app.cell
def _(mo):
    mo.md("""
    ## 6. 직접 바꿔 볼 과제와 해설

    1. k=1/2에서 토큰별 선택 전문가와 결합 가중치를 예측하고 표로 확인하세요.
       **기준:** 정규화할 때 선택 가중치 합은 1, 총 할당 횟수는 N*k입니다.
    2. 1절의 고정 router 정규화를 끄면 출력과 보조 손실 중 무엇이 달라질까요?
       **해설:** 같은 k에서 출력 결합 가중치는 달라지고 보조 손실은 그대로입니다. 독립된 5절 LM의 재학습 설정과 구분하세요.
    3. 4-2절의 부호 UI를 +1/-1로 바꿔 선택 전문가와 gradient 표를 기록하세요.
       **기준:** 기본 입력은 +1에서 expert 0, -1에서 expert 1입니다. 두 경우 모두 검증이 통과해야 합니다.
       입력 두 토큰의 부호를 다르게 직접 바꾸어도 각 토큰의 실제 선택으로 검증합니다.
    4. 5절에서 계수 0/0.01의 main/aux, 사용량 전후, 평균 확률과 계수 없는 균형 항을 비교하세요.
       **기준:** 초기 사용량은 같고 계수 0이면 aux도 0입니다. 균등 선택이나 더 높은 정확도를 반드시 기대하지 마세요.
    5. p/p와 straight-through의 scalar 출력이 같은데도 gradient가 다른 이유를 설명하세요.
       **해설:** detach한 항은 역전파에서 상수로 취급됩니다. 일반 미분을 그대로 쓰는 것과는 다릅니다.

    **기록할 것:** k, 정규화 여부, 전문가별 선택 횟수, 보조 손실, 전체 보유 매개변수, LM 손실을 적으세요.
    실제 대규모 MoE의 GPU 분산 실행·통신·속도 비교·전문가 역할 평가는 이 실습에 포함하지 않습니다.
    """)
    return


if __name__ == "__main__":
    app.run()
