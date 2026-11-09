import marimo

__generated_with = "0.25.2"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Gated DeltaNet

    Gated MHA와 토큰별 상태를 갱신하는 Gated DeltaNet의 입력·출력과 기울기를 확인합니다.

    모델 연산과 생성 함수는 기존 구현을 유지합니다. GPT 실험은 학습 전 무작위 가중치를 사용하므로 생성 문장의 품질보다 구조와 실행 과정을 관찰하세요.

    `gpt/`에서 `uv run marimo edit gpt_with_deltanet.py`로 열고 설정을 확인한 뒤 **실험 실행**을 누르세요. GPT 기본값은 기존 스크립트의 모델 크기와 생성 길이를 유지합니다. 결과는 `outputs/gpt/<실행 ID>/`에 저장합니다.
    """)
    return


@app.cell
def _():
    import marimo as mo
    from types import SimpleNamespace
    from marimo_runtime import prepare_run
    import contextlib
    import io
    import json
    import torch
    from torch import nn
    import torch.nn.functional as F

    return F, contextlib, io, json, mo, nn, prepare_run, torch


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## GatedMultiHeadAttention
    """)
    return


@app.cell
def _(nn, torch):
    class GatedMultiHeadAttention(nn.Module):
        def __init__(
            self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False
        ):
            super().__init__()
            assert d_out % num_heads == 0

            self.d_out = d_out
            self.num_heads = num_heads
            self.head_dim = d_out // num_heads

            self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
            ####################################################
            ### NEW: Add gate
            self.W_gate = nn.Linear(d_in, d_out, bias=qkv_bias)
            ####################################################
            self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
            self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

            self.out_proj = nn.Linear(d_out, d_out)
            self.dropout = nn.Dropout(dropout)

            self.register_buffer(
                "mask",
                torch.triu(torch.ones(context_length, context_length), diagonal=1),
                persistent=False,
            )

        def forward(self, x):
            b, num_tokens, _ = x.shape
            queries = self.W_query(x)
            ####################################################
            ### NEW: Add gate
            gate = self.W_gate(x)
            ####################################################
            keys = self.W_key(x)
            values = self.W_value(x)

            keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
            values = values.view(b, num_tokens, self.num_heads, self.head_dim)
            queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

            keys = keys.transpose(1, 2)
            queries = queries.transpose(1, 2)
            values = values.transpose(1, 2)

            attn_scores = queries @ keys.transpose(2, 3)

            mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
            attn_scores.masked_fill_(
                mask_bool, torch.finfo(attn_scores.dtype).min
            )

            attn_weights = torch.softmax(
                attn_scores / (self.head_dim ** 0.5), dim=-1
            )
            attn_weights = self.dropout(attn_weights)

            context = (attn_weights @ values).transpose(1, 2)
            context = context.reshape(b, num_tokens, self.d_out)

            ####################################################
            ### NEW: Add gate
            context = context * torch.sigmoid(gate)
            ####################################################
            out = self.out_proj(context)
            return out

    return (GatedMultiHeadAttention,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Query·Key 정규화
    """)
    return


@app.cell
def _(torch):
    def l2norm(x, dim=-1, eps=1e-6):
        return x * torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)

    return (l2norm,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## GatedDeltaNet
    """)
    return


@app.cell
def _(F, l2norm, nn, torch):
    class GatedDeltaNet(nn.Module):
        def __init__(
            self, d_in, d_out, dropout, num_heads, qkv_bias=False
        ):
            super().__init__()
            assert d_out % num_heads == 0

            self.d_out = d_out
            self.num_heads = num_heads
            self.head_dim = d_out // num_heads

            self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
            self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
            self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
            ####################################################
            ### NEW: Gates for delta rule and output gating
            self.W_gate = nn.Linear(d_in, d_out, bias=False)
            self.W_beta = nn.Linear(d_in, d_out, bias=False)

            # Note: The decay gate alpha corresponds to
            # A_log + W_alpha(x) + dt_bias
            self.W_alpha = nn.Linear(d_in, num_heads, bias=False)
            self.dt_bias = nn.Parameter(torch.ones(num_heads))
            A_init = torch.empty(num_heads).uniform_(0, 16)
            self.A_log = nn.Parameter(torch.log(A_init))
            # We could implement this as
            # W_alpha = nn.Linear(d_in, num_heads, bias=True)
            # but the bias is separate for interpretability and
            # to mimic the official implementation

            self.norm = nn.RMSNorm(self.head_dim, eps=1e-6)
            ####################################################

            self.out_proj = nn.Linear(d_out, d_out)
            self.dropout = nn.Dropout(dropout)

        def forward(self, x):
            b, num_tokens, _ = x.shape
            queries = self.W_query(x)
            keys = self.W_key(x)
            values = self.W_value(x)
            ####################################################
            ### NEW: Compute delta rule gates
            beta = torch.sigmoid(self.W_beta(x))
            alpha_log = -self.A_log.exp().view(1, 1, -1) * F.softplus(
                self.W_alpha(x) + self.dt_bias
            )
            alpha = alpha_log.exp()
            gate = self.W_gate(x)
            ####################################################

            keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
            values = values.view(b, num_tokens, self.num_heads, self.head_dim)
            queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)
            beta = beta.view(b, num_tokens, self.num_heads, self.head_dim)
            gate = gate.view(b, num_tokens, self.num_heads, self.head_dim)  # NEW

            keys = keys.transpose(1, 2)
            queries = queries.transpose(1, 2)
            values = values.transpose(1, 2)
            beta = beta.transpose(1, 2)

            ####################################################
            ### NEW: QKNorm-like normalization for delta rule
            queries = l2norm(queries, dim=-1) / (self.head_dim ** 0.5)
            keys = l2norm(keys, dim=-1)
            ####################################################

            S = x.new_zeros(b, self.num_heads, self.head_dim, self.head_dim)

            outs = []
            ####################################################
            ### NEW: Gated delta rule update
            for t in range(num_tokens):
                k_t = keys[:, :, t]
                q_t = queries[:, :, t]
                v_t = values[:, :, t]
                b_t = beta[:, :, t]
                a_t = alpha[:, t].unsqueeze(-1).unsqueeze(-1)

                S = S * a_t
                kv_mem = (S * k_t.unsqueeze(-1)).sum(dim=-2)
                delta = (v_t - kv_mem) * b_t
                S = S + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
                y_t = (S * q_t.unsqueeze(-1)).sum(dim=-2)
                ####################################################
                outs.append(y_t)

            context = torch.stack(outs, dim=2).transpose(1, 2).contiguous()
            context = context.view(b, num_tokens, self.num_heads, self.head_dim)

            ####################################################
            ### NEW: Apply RMSNorm and SiLU gate
            context = self.norm(context)
            context = context * F.silu(gate)
            ####################################################

            context = context.view(b, num_tokens, self.d_out)
            context = self.dropout(context)
            out = self.out_proj(context)
            return out

    return (GatedDeltaNet,)


@app.cell
def _(GatedDeltaNet, GatedMultiHeadAttention, torch):
    def run_experiment(options):
        torch.manual_seed(123)
        width = options["emb_dim"]
        heads = options["n_heads"]
        length = options["sequence_length"]
        batch = options["batch_size"]
        results = {}
        for name, module in {
            "gated_mha": GatedMultiHeadAttention(width, width, length, 0.0, heads),
            "gated_deltanet": GatedDeltaNet(width, width, 0.0, heads),
        }.items():
            x = torch.randn(batch, length, width, requires_grad=True)
            y = module(x)
            loss = y.square().mean()
            loss.backward()
            assert y.shape == x.shape and torch.isfinite(y).all()
            assert x.grad is not None and torch.isfinite(x.grad).all()
            results[name] = {"input_shape": list(x.shape), "output_shape": list(y.shape), "loss": loss.item(), "finite_input_gradient": True}
            print(name, results[name])
        return results

    return (run_experiment,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 설정과 실행

    설정 변경 후 실행 버튼을 누르면 새 모델로 실험합니다. `emb_dim`은 `n_heads`로 나누어져야 합니다.
    """)
    return


@app.cell
def _(mo):
    settings = mo.ui.dictionary({
        'emb_dim': mo.ui.number(start=1, value=32, step=1, label='emb_dim'),
        'n_heads': mo.ui.number(start=1, value=4, step=1, label='n_heads'),
        'sequence_length': mo.ui.number(start=1, value=8, step=1, label='sequence_length'),
        'batch_size': mo.ui.number(start=1, value=2, step=1, label='batch_size'),
    })
    run_experiment_button = mo.ui.run_button(label="실험 실행")
    mo.vstack([settings, run_experiment_button])
    return run_experiment_button, settings


@app.cell
def _(
    contextlib,
    io,
    json,
    mo,
    prepare_run,
    run_experiment,
    run_experiment_button,
    settings,
):
    mo.stop(not run_experiment_button.value, mo.md("설정을 확인하고 **실험 실행**을 누르세요."))
    _run_options = dict(settings.value)
    assert _run_options["emb_dim"] % _run_options["n_heads"] == 0, "emb_dim은 n_heads로 나누어져야 합니다."
    experiment_output = prepare_run('gpt_with_deltanet')
    _log_buffer = io.StringIO()
    try:
        with contextlib.redirect_stdout(_log_buffer):
            experiment_result = run_experiment(_run_options)
    finally:
        (experiment_output / "run.log").write_text(_log_buffer.getvalue(), encoding="utf-8")
    print(_log_buffer.getvalue())
    (experiment_output / "result.json").write_text(json.dumps({"options": _run_options, "result": experiment_result}, ensure_ascii=False, indent=2), encoding="utf-8")
    mo.md(f"실행 결과: `{experiment_output}`")
    return


if __name__ == "__main__":
    app.run()
