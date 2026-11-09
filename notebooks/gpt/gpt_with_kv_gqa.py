# Sebastian Raschka의 아파치 라이선스 2.0 하에 제공되는 저작권 고지입니다(LICENSE.txt 참고).
# "Build a Large Language Model From Scratch"의 출처
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# 코드: https://github.com/rasbt/LLMs-from-scratch

# 이 파일은 지금까지 3-4장에서 다룬 관련 코드를 모아,
# 그룹화 쿼리 어텐션(GQA)을 사용하도록 조정한 것입니다.
# 이 파일은 marimo 편집기에서 실행하는 모델 구조 실습입니다.

import marimo

__generated_with = "0.25.2"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Grouped-Query Attention

    여러 Query head가 작은 수의 Key·Value 그룹을 공유합니다.

    모델 연산과 생성 함수는 기존 구현을 유지합니다. GPT 실험은 학습 전 무작위 가중치를 사용하므로 생성 문장의 품질보다 구조와 실행 과정을 관찰하세요.

    `gpt/`에서 `uv run marimo edit gpt_with_kv_gqa.py`로 열고 설정을 확인한 뒤 **실험 실행**을 누르세요. GPT 기본값은 기존 스크립트의 모델 크기와 생성 길이를 유지합니다. 결과는 `outputs/gpt/<실행 ID>/`에 저장합니다.
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
    ## GroupedQueryAttention
    """)
    return


@app.cell
def _(nn, torch):
    #####################################
    # 신규: MHA 대신 GQA 사용
    #####################################
    class GroupedQueryAttention(nn.Module):
        def __init__(
                self, d_in, d_out, dropout, num_heads, num_kv_groups, dtype=None, qkv_bias=False
        ):
            super().__init__()
            assert d_out % num_heads == 0, "d_out must be divisible by num_heads"
            assert num_heads % num_kv_groups == 0, "num_heads must be divisible by num_kv_groups"

            self.d_out = d_out
            self.num_heads = num_heads
            self.head_dim = d_out // num_heads

            #key/value projection을 작게 만드는 부분
            self.W_key = nn.Linear(d_in, num_kv_groups * self.head_dim, bias=qkv_bias, dtype=dtype)
            self.W_value = nn.Linear(d_in, num_kv_groups * self.head_dim, bias=qkv_bias, dtype=dtype)
            self.num_kv_groups = num_kv_groups
            self.group_size = num_heads // num_kv_groups

            self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias, dtype=dtype)
            self.out_proj = nn.Linear(d_out, d_out, bias=False, dtype=dtype)
            self.dropout = nn.Dropout(dropout)

            self.register_buffer("cache_k", None, persistent=False)
            self.register_buffer("cache_v", None, persistent=False)
            self.ptr_current_pos = 0

        def forward(self, x, use_cache=False):
            b, num_tokens, _ = x.shape

            # 프로젝션 적용
            queries = self.W_query(x)  # (b, num_tokens, num_heads * head_dim)
            keys = self.W_key(x)       # (b, num_tokens, num_kv_groups * head_dim)
            values = self.W_value(x)   # (b, num_tokens, num_kv_groups * head_dim)

            # 형태 변경
            queries = queries.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
            keys_new = keys.view(b, num_tokens, self.num_kv_groups, self.head_dim).transpose(1, 2)
            values_new = values.view(b, num_tokens, self.num_kv_groups, self.head_dim).transpose(1, 2)

            if use_cache:
                if self.cache_k is None:
                    self.cache_k, self.cache_v = keys_new, values_new
                else:
                    self.cache_k = torch.cat([self.cache_k, keys_new], dim=2)
                    self.cache_v = torch.cat([self.cache_v, values_new], dim=2)
                keys_base, values_base = self.cache_k, self.cache_v
            else:
                keys_base, values_base = keys_new, values_new
                if self.cache_k is not None or self.cache_v is not None:
                    self.cache_k, self.cache_v = None, None
                    self.ptr_current_pos = 0

            # 헤드 수에 맞게 키와 값을 확장
            # 형태: (b, num_heads, num_tokens, head_dim)
            keys = keys_base.repeat_interleave(self.group_size, dim=1)  # 형태: (b, num_heads, num_tokens, head_dim)
            values = values_base.repeat_interleave(self.group_size, dim=1)  # 형태: (b, num_heads, num_tokens, head_dim)

            # 인과 마스크를 적용해 스케일된 점곱 어텐션(즉 자기 어텐션) 계산
            # 형태: (b, num_heads, num_tokens, num_tokens)
            attn_scores = queries @ keys.transpose(2, 3)  # 각 헤드의 내적

            ####################################################
            # 인과 마스크
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
            mask = q_positions.unsqueeze(-1) < k_positions.unsqueeze(0)

            # 마스크를 사용해 어텐션 점수를 채움
            attn_scores = attn_scores.masked_fill(mask, -torch.inf)

            attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
            assert keys.shape[-1] == self.head_dim
            attn_weights = self.dropout(attn_weights)

            # 형태: (b, num_tokens, num_heads, head_dim)
            context_vec = (attn_weights @ values).transpose(1, 2)

            # self.d_out = self.num_heads * self.head_dim이 되도록 헤드를 결합
            context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
            context_vec = self.out_proj(context_vec)  # 선택적 프로젝션

            return context_vec

        def reset_cache(self):
            self.cache_k, self.cache_v = None, None
            self.ptr_current_pos = 0

    return (GroupedQueryAttention,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 정규화
    """)
    return


@app.cell
def _(nn, torch):
    #####################################
    # 4장
    #####################################
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
def _(FeedForward, GroupedQueryAttention, LayerNorm, nn):
    class TransformerBlock(nn.Module):
        def __init__(self, cfg):
            super().__init__()
            self.att = GroupedQueryAttention(
                d_in=cfg["emb_dim"],
                d_out=cfg["emb_dim"],
                num_heads=cfg["n_heads"],
                num_kv_groups=cfg["n_kv_groups"],
                dropout=cfg["drop_rate"],
                qkv_bias=cfg["qkv_bias"])
            self.ff = FeedForward(cfg)
            self.norm1 = LayerNorm(cfg["emb_dim"])
            self.norm2 = LayerNorm(cfg["emb_dim"])
            self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

        def forward(self, x, use_cache=False):
            # 어텐션 블록의 지름길 연결
            shortcut = x
            x = self.norm1(x)

            # x = self.att(x)   # 형태 [batch_size, num_tokens, emb_size]
            ####################################################
            # KV 캐시 관련
            x = self.att(x, use_cache=use_cache)
            ####################################################

            x = self.drop_shortcut(x)
            x = x + shortcut  # 원래 입력을 다시 더함

            # 피드포워드 블록의 지름길 연결
            shortcut = x
            x = self.norm2(x)
            x = self.ff(x)
            x = self.drop_shortcut(x)
            x = x + shortcut  # 원래 입력을 다시 더함

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
            # KV 캐시 관련
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
            # KV 캐시 관련
            if use_cache:
                pos_ids = torch.arange(self.current_pos, self.current_pos + seq_len, device=in_idx.device, dtype=torch.long)
                self.current_pos += seq_len
            else:
                pos_ids = torch.arange(0, seq_len, device=in_idx.device, dtype=torch.long)
            pos_embeds = self.pos_emb(pos_ids).unsqueeze(0)
            ####################################################

            x = tok_embeds + pos_embeds  # 형태 [batch_size, num_tokens, emb_size]
            x = self.drop_emb(x)

            # x = self.trf_blocks(x)
            ####################################################
            # KV 캐시 관련
            for blk in self.trf_blocks:
                x = blk(x, use_cache=use_cache)
            ####################################################

            x = self.final_norm(x)
            logits = self.out_head(x)
            return logits

        ####################################################
        # KV 캐시 관련
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
                # 전체 프롬프트로 캐시 초기화
                model.reset_kv_cache()
                logits = model(idx[:, -ctx_len:], use_cache=True)

                for step in range(max_new_tokens):
                    # a) 로그 확률이 가장 높은 토큰 선택(탐욕적 샘플링)
                    next_idx = logits[:, -1].argmax(dim=-1, keepdim=True)
                    # b) 진행 중인 시퀀스에 추가
                    idx = torch.cat([idx, next_idx], dim=1)
                    # c) 모델에는 새 토큰만 입력
                    if step + 1 < max_new_tokens:  # 마지막 토큰 생성 후에는 순전파를 건너뜀
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
            "vocab_size": 50257,        # 어휘 크기
            "context_length": args.max_new_tokens + len(encoded),
            "emb_dim": args.emb_dim,    # 임베딩 차원
            "n_heads": args.n_heads,    # 어텐션 헤드 수
            "n_layers": args.n_layers,  # 레이어 수
            "drop_rate": 0.0,           # 드롭아웃 비율
            "qkv_bias": False,          # 쿼리-키-값 편향
            "n_kv_groups": args.n_kv_groups
        }
        torch.manual_seed(123)
        model = GPTModel(GPT_CONFIG_124M)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device, dtype=torch.bfloat16)
        model.eval()  # 드롭아웃 비활성화

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

    설정 변경 후 실행 버튼을 누르면 새 모델로 실험합니다. `emb_dim`은 `n_heads`로 나누어져야 합니다. GQA는 `n_heads`도 `n_kv_groups`로 나누어져야 합니다.
    """)
    return


@app.cell
def _(mo):
    settings = mo.ui.dictionary({
        'prompt': mo.ui.text(value='Hello, I am', label='prompt'),
        'emb_dim': mo.ui.number(start=1, value=768, step=1, label='emb_dim'),
        'n_heads': mo.ui.number(start=1, value=12, step=1, label='n_heads'),
        'n_layers': mo.ui.number(start=1, value=12, step=1, label='n_layers'),
        'n_kv_groups': mo.ui.number(start=1, value=2, step=1, label='n_kv_groups'),
        'max_new_tokens': mo.ui.number(start=1, value=200, step=1, label='max_new_tokens'),
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
    experiment_output = prepare_run('gpt_with_kv_gqa')
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
