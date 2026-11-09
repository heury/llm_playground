# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch

# This file collects all the relevant code that we covered thus far
# throughout Chapters 3-4, adapted to use Multi-Head Latent Attention (MLA).
# Run this model experiment with the marimo editor.

import marimo

__generated_with = "0.25.2"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Multi-Head Latent Attention

    Key·Value를 저차원 latent로 압축해 캐시하는 구현을 살펴봅니다.

    모델 연산과 생성 함수는 기존 구현을 유지합니다. GPT 실험은 학습 전 무작위 가중치를 사용하므로 생성 문장의 품질보다 구조와 실행 과정을 관찰하세요.

    `gpt/`에서 `uv run marimo edit gpt_with_kv_mla.py`로 열고 설정을 확인한 뒤 **실험 실행**을 누르세요. GPT 기본값은 기존 스크립트의 모델 크기와 생성 길이를 유지합니다. 결과는 `outputs/gpt/<실행 ID>/`에 저장합니다.
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
    import time
    import tiktoken
    import torch
    import torch.nn as nn

    return (
        SimpleNamespace,
        contextlib,
        io,
        json,
        mo,
        nn,
        prepare_run,
        tiktoken,
        time,
        torch,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## MultiHeadLatentAttention
    """)
    return


@app.cell
def _(nn, torch):
    #####################################
    # Multi-Head Latent Attention
    #####################################
    # The MLA code below is inspired by
    # https://huggingface.co/bird-of-paradise/deepseek-mla


    class MultiHeadLatentAttention(nn.Module):
        def __init__(self, d_in, d_out, dropout, num_heads,
                     qkv_bias=False, latent_dim=None):
            super().__init__()
            assert d_out % num_heads == 0, "d_out must be divisible by num_heads"

            self.d_out = d_out
            self.num_heads = num_heads
            self.head_dim = d_out // num_heads
            self.latent_dim = latent_dim if latent_dim is not None else max(16, d_out // 8)

            # Projections
            self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)              # per-head Q
            self.W_DKV = nn.Linear(d_in, self.latent_dim, bias=qkv_bias)    # down to latent C
            self.W_UK = nn.Linear(self.latent_dim, d_out, bias=qkv_bias)   # latent -> per-head K
            self.W_UV = nn.Linear(self.latent_dim, d_out, bias=qkv_bias)   # latent -> per-head V

            self.out_proj = nn.Linear(d_out, d_out)
            self.dropout = nn.Dropout(dropout)

            ####################################################
            # Latent-KV cache
            self.register_buffer("cache_c_kv", None, persistent=False)
            self.ptr_current_pos = 0
            ####################################################

        def reset_cache(self):
            self.cache_c_kv = None
            self.ptr_current_pos = 0

        @staticmethod
        def _reshape_to_heads(x, num_heads, head_dim):
            # (b, T, d_out) -> (b, num_heads, T, head_dim)
            bsz, num_tokens, _ = x.shape
            return x.view(bsz, num_tokens, num_heads, head_dim).transpose(1, 2).contiguous()

        def forward(self, x, use_cache=False):
            b, num_tokens, _ = x.shape
            num_heads = self.num_heads
            head_dim = self.head_dim

            # 1) Project to queries (per-token, per-head) and new latent chunk
            queries_all = self.W_query(x)  # (b, T, d_out)
            latent_new = self.W_DKV(x)  # (b, T, latent_dim)

            # 2) Update latent cache and choose latent sequence to up-project
            if use_cache:
                if self.cache_c_kv is None:
                    latent_total = latent_new
                else:
                    latent_total = torch.cat([self.cache_c_kv, latent_new], dim=1)
                self.cache_c_kv = latent_total
            else:
                latent_total = latent_new

            # 3) Up-project latent to per-head keys/values (then split into heads)
            keys_all = self.W_UK(latent_total)   # (b, T_k_total, d_out)
            values_all = self.W_UV(latent_total)   # (b, T_k_total, d_out)

            # 4) Reshape to heads
            queries = self._reshape_to_heads(queries_all, num_heads, head_dim)
            keys = self._reshape_to_heads(keys_all, num_heads, head_dim)
            values = self._reshape_to_heads(values_all, num_heads, head_dim)

            # 5) Scaled dot-product attention with causal mask
            attn_scores = torch.matmul(queries, keys.transpose(-2, -1))

            num_tokens_Q = queries.shape[-2]
            num_tokens_K = keys.shape[-2]
            device = queries.device
            if use_cache:
                q_positions = torch.arange(
                    self.ptr_current_pos,
                    self.ptr_current_pos + num_tokens_Q,
                    device=device,
                    dtype=torch.long,
                )
                self.ptr_current_pos += num_tokens_Q
            else:
                q_positions = torch.arange(num_tokens_Q, device=device, dtype=torch.long)
                self.ptr_current_pos = 0
            k_positions = torch.arange(num_tokens_K, device=device, dtype=torch.long)
            mask_bool = q_positions.unsqueeze(-1) < k_positions.unsqueeze(0)

            # Use the mask to fill attention scores
            attn_scores.masked_fill_(mask_bool, -torch.inf)

            attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
            attn_weights = self.dropout(attn_weights)

            # Shape: (b, num_tokens, num_heads, head_dim)
            context_vec = (attn_weights @ values).transpose(1, 2)

            # Combine heads, where self.d_out = self.num_heads * self.head_dim
            context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
            context_vec = self.out_proj(context_vec)  # optional projection

            return context_vec

    return (MultiHeadLatentAttention,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 정규화
    """)
    return


@app.cell
def _(nn, torch):
    class LayerNorm(nn.Module):
        def __init__(self, emb_dim):
            super().__init__()
            self.eps = 1e-5
            self.scale = nn.Parameter(torch.ones(emb_dim))
            self.shift = nn.Parameter(torch.zeros(emb_dim))

        def forward(self, x):
            mean = x.mean(dim=-1, keepdim=True)
            var = x.var(dim=-1, keepdim=True, unbiased=False)
            norm_x = (x - mean) / torch.sqrt(var + self.eps)
            return self.scale * norm_x + self.shift

    return (LayerNorm,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 활성화 함수
    """)
    return


@app.cell
def _(nn, torch):
    class GELU(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return 0.5 * x * (1 + torch.tanh(
                torch.sqrt(torch.tensor(2.0 / torch.pi)) *
                (x + 0.044715 * torch.pow(x, 3))
            ))

    return (GELU,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 피드포워드 층
    """)
    return


@app.cell
def _(GELU, nn):
    class FeedForward(nn.Module):
        def __init__(self, cfg):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
                GELU(),
                nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
            )

        def forward(self, x):
            return self.layers(x)

    return (FeedForward,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Transformer 블록
    """)
    return


@app.cell
def _(FeedForward, LayerNorm, MultiHeadLatentAttention, nn):
    class TransformerBlock(nn.Module):
        def __init__(self, cfg):
            super().__init__()
            self.att = MultiHeadLatentAttention(
                d_in=cfg["emb_dim"],
                d_out=cfg["emb_dim"],
                num_heads=cfg["n_heads"],
                dropout=cfg["drop_rate"],
                qkv_bias=cfg["qkv_bias"],
                latent_dim=cfg["latent_dim"])

            self.ff = FeedForward(cfg)
            self.norm1 = LayerNorm(cfg["emb_dim"])
            self.norm2 = LayerNorm(cfg["emb_dim"])
            self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

        def forward(self, x, use_cache=False):
            # Shortcut connection for attention block
            shortcut = x
            x = self.norm1(x)

            # x = self.att(x)   # Shape [batch_size, num_tokens, emb_size]
            ####################################################
            #  KV cache-related
            x = self.att(x, use_cache=use_cache)
            ####################################################

            x = self.drop_shortcut(x)
            x = x + shortcut  # Add the original input back

            # Shortcut connection for feed-forward block
            shortcut = x
            x = self.norm2(x)
            x = self.ff(x)
            x = self.drop_shortcut(x)
            x = x + shortcut  # Add the original input back

            return x

    return (TransformerBlock,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## GPT 모델 결합
    """)
    return


@app.cell
def _(LayerNorm, TransformerBlock, nn, torch):
    class GPTModel(nn.Module):
        def __init__(self, cfg):
            super().__init__()
            self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
            self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
            self.drop_emb = nn.Dropout(cfg["drop_rate"])

            # self.trf_blocks = nn.Sequential(
            #    *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])
            ####################################################
            #  KV cache-related
            self.trf_blocks = nn.ModuleList(
                [TransformerBlock(cfg) for _ in range(cfg["n_layers"])])

            self.current_pos = 0
            ####################################################

            self.final_norm = LayerNorm(cfg["emb_dim"])
            self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

        def forward(self, in_idx, use_cache=False):
            batch_size, seq_len = in_idx.shape
            tok_embeds = self.tok_emb(in_idx)

            # pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))

            ####################################################
            #  KV cache-related
            if use_cache:
                pos_ids = torch.arange(self.current_pos, self.current_pos + seq_len, device=in_idx.device, dtype=torch.long)
                self.current_pos += seq_len
            else:
                pos_ids = torch.arange(0, seq_len, device=in_idx.device, dtype=torch.long)
            pos_embeds = self.pos_emb(pos_ids).unsqueeze(0)
            ####################################################

            x = tok_embeds + pos_embeds  # Shape [batch_size, num_tokens, emb_size]
            x = self.drop_emb(x)

            # x = self.trf_blocks(x)
            ####################################################
            #  KV cache-related
            for blk in self.trf_blocks:
                x = blk(x, use_cache=use_cache)
            ####################################################

            x = self.final_norm(x)
            logits = self.out_head(x)
            return logits

        ####################################################
        #  KV cache-related
        def reset_kv_cache(self):
            for blk in self.trf_blocks:
                blk.att.reset_cache()
            self.current_pos = 0

    return (GPTModel,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## KV cache를 사용하는 생성
    """)
    return


@app.cell
def _(torch):
    ####################################################


    def generate_text_simple_cached(model, idx, max_new_tokens,
                                    context_size=None, use_cache=True):
        model.eval()
        ctx_len = context_size or model.pos_emb.num_embeddings

        with torch.no_grad():
            if use_cache:
                # Init cache with full prompt
                model.reset_kv_cache()
                logits = model(idx[:, -ctx_len:], use_cache=True)

                for step in range(max_new_tokens):
                    # a) pick the token with the highest log-probability (greedy sampling)
                    next_idx = logits[:, -1].argmax(dim=-1, keepdim=True)
                    # b) append it to the running sequence
                    idx = torch.cat([idx, next_idx], dim=1)
                    # c) feed model only the new token
                    if step + 1 < max_new_tokens:  # Skips the forward pass after the last token is generated
                        logits = model(next_idx, use_cache=True)
            else:
                for _ in range(max_new_tokens):
                    logits = model(idx[:, -ctx_len:], use_cache=False)
                    next_idx = logits[:, -1].argmax(dim=-1, keepdim=True)
                    idx = torch.cat([idx, next_idx], dim=1)

        return idx

    return (generate_text_simple_cached,)


@app.cell
def _(
    GPTModel,
    SimpleNamespace,
    generate_text_simple_cached,
    tiktoken,
    time,
    torch,
):
    def run_experiment(options):
        args = SimpleNamespace(**options)
        start_context = args.prompt
        tokenizer = tiktoken.get_encoding("gpt2")
        encoded = tokenizer.encode(start_context)

        GPT_CONFIG_124M = {
            "vocab_size": 50257,        # Vocabulary size
            "context_length": args.max_new_tokens + len(encoded),
            "emb_dim": args.emb_dim,    # Embedding dimension
            "n_heads": args.n_heads,    # Number of attention heads
            "n_layers": args.n_layers,  # Number of layers
            "drop_rate": 0.0,           # Dropout rate
            "qkv_bias": False,          # Query-Key-Value bias
            "latent_dim": args.latent_dim,
        }
        torch.manual_seed(123)
        model = GPTModel(GPT_CONFIG_124M)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device, dtype=torch.bfloat16)
        model.eval()  # disable dropout

        encoded_tensor = torch.tensor(encoded, device=device).unsqueeze(0)
        print(f"\n{50*'='}\n{22*' '}IN\n{50*'='}")
        print("\nInput text:", start_context)
        print("Encoded input text:", encoded)
        print("encoded_tensor.shape:", encoded_tensor.shape)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.time()

        token_ids = generate_text_simple_cached(
            model=model,
            idx=encoded_tensor,
            max_new_tokens=args.max_new_tokens,
        )

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        total_time = time.time() - start

        decoded_text = tokenizer.decode(token_ids.squeeze(0).tolist())

        print(f"\n\n{50*'='}\n{22*' '}OUT\n{50*'='}")
        print("\nOutput:", token_ids)
        print("Output length:", len(token_ids[0]))
        print("Output text:", decoded_text)

        print(f"\nTime: {total_time:.2f} sec")
        print(f"{int(len(token_ids[0])/total_time)} tokens/sec")
        if torch.cuda.is_available():
            max_mem_bytes = torch.cuda.max_memory_allocated()
            max_mem_gb = max_mem_bytes / (1024 ** 3)
            print(f"Max memory allocated: {max_mem_gb:.2f} GB")

        return {"config": GPT_CONFIG_124M, "device": str(device), "dtype": str(next(model.parameters()).dtype), "elapsed_seconds": total_time, "token_ids": token_ids.detach().cpu().tolist(), "decoded_text": decoded_text}

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
        'prompt': mo.ui.text(value='Hello, I am', label='prompt'),
        'emb_dim': mo.ui.number(start=1, value=768, step=1, label='emb_dim'),
        'n_heads': mo.ui.number(start=1, value=12, step=1, label='n_heads'),
        'n_layers': mo.ui.number(start=1, value=12, step=1, label='n_layers'),
        'max_new_tokens': mo.ui.number(start=1, value=200, step=1, label='max_new_tokens'),
        'latent_dim': mo.ui.dropdown(options={'자동': None, **{str(v): v for v in [16, 32, 64, 96, 128, 256, 512]}}, value='자동', label='latent_dim'),
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
    experiment_output = prepare_run('gpt_with_kv_mla')
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
