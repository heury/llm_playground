import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import math
    import sys
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import torch
    import torch.nn.functional as F
    from transformers import AutoTokenizer

    # 노트북 위치(notebooks/minimind/)에서 두 단계 위가 저장소 루트입니다.
    repo_root = Path(__file__).resolve().parents[2]
    minimind_dir = repo_root / "projects/minimind"
    sys.path.insert(0, str(minimind_dir))

    # 원본 모델 파일의 클래스와 함수를 그대로 가져옵니다.
    from model.model_minimind import (
        MiniMindConfig,
        MiniMindForCausalLM,
        apply_rotary_pos_emb,
        precompute_freqs_cis,
        repeat_kv,
    )

    torch.set_num_threads(2)
    # train_pretrain.py와 같이 GPU가 있으면 cuda:0을 씁니다.
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return (
        AutoTokenizer,
        F,
        MiniMindConfig,
        MiniMindForCausalLM,
        apply_rotary_pos_emb,
        device,
        math,
        minimind_dir,
        mo,
        plt,
        precompute_freqs_cis,
        repeat_kv,
        torch,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # MiniMind 모델 구조 따라가기

    `projects/minimind/model/model_minimind.py` 한 파일에 MiniMind 언어 모델 전체가 들어 있습니다.
    이 노트북은 그 파일을 **위에서 아래로** 읽으면서, 각 부품을 작은 모델로 **직접 계산**하고
    **원본 결과와 같은지** 확인합니다.

    **사전 지식:** Python, 텐서의 `shape`, 행렬 곱 `@`.
    모델은 `train_pretrain.py`가 학습하는 것과 **같은 크기**(은닉 차원 768, 8층, 약 64M 매개변수)이고,
    GPU가 있으면 GPU에서 계산합니다(없으면 CPU, 조금 느려짐). 다운로드는 필요 없습니다.
    가중치는 학습되지 않은 **무작위 값**이므로, 출력되는 단어의 의미가 아니라 **계산 과정과 shape**에 집중하세요.

    ## 0. 큰 그림

    ```text
    문장 ──토크나이저──▶ input_ids [B, T]
                          │
                 embed_tokens (Embedding)          ▶ h [B, T, D]
                          │
          ┌──── MiniMindBlock × L ─────────────────┐
          │  h = h + Attention( RMSNorm(h) )       │  ← 토큰끼리 정보 교환
          │  h = h + FeedForward( RMSNorm(h) )     │  ← 토큰마다 따로 가공
          └────────────────────────────────────────┘
                          │
                    norm (RMSNorm)
                          │
          lm_head (Embedding과 같은 가중치)         ▶ logits [B, T, V]
                          │
                  softmax → 다음 토큰 확률
    ```

    | 기호 | 뜻 | 기호 | 뜻 |
    | --- | --- | --- | --- |
    | `B` | 배치(문장 수) | `H` | Query head 수 |
    | `T` | 토큰 수(문장 길이) | `KV` | Key/Value head 수 |
    | `D` | 은닉 차원 (`hidden_size`) | `d` | head 하나의 차원 (`head_dim`) |
    | `V` | 어휘 수 (`vocab_size`) | `L` | 블록(층) 수 |

    **파일 속 클래스 지도** (이 노트북의 절 번호)

    | 원본 이름 | 역할 | 절 |
    | --- | --- | --- |
    | `MiniMindConfig` | 크기와 옵션을 담는 설정 | 1 |
    | `MiniMindModel.embed_tokens` | 토큰 ID → 벡터 | 3 |
    | `RMSNorm` | 벡터 크기 정규화 | 4 |
    | `precompute_freqs_cis`, `apply_rotary_pos_emb` | 위치 정보 RoPE | 5 |
    | `Attention`, `repeat_kv` | 토큰 사이 정보 교환, GQA | 6 |
    | `FeedForward` | 토큰별 가공 (SwiGLU) | 7 |
    | `MiniMindBlock` | Attention + FeedForward + 잔차 연결 | 8 |
    | `MiniMindModel`, `MiniMindForCausalLM` | 전체 조립, logits, 손실 | 9 |
    | `MiniMindForCausalLM.generate` | KV cache를 쓰는 생성 | 10 |
    | `MOEFeedForward` | 전문가 혼합 (선택) | 11 |
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. 설정: `MiniMindConfig`

    모델의 모든 크기는 설정 하나에서 정해집니다. 몇 가지 값은 **다른 값에서 계산**됩니다.

    - `head_dim = hidden_size // num_attention_heads` — 은닉 벡터를 head 수만큼 나눈 크기
    - `intermediate_size = ceil(hidden_size × π / 64) × 64` — FeedForward 중간 차원 (약 3.14배, 64의 배수)
    - `num_key_value_heads`가 `num_attention_heads`보다 작으면 **GQA**(6-2절)
    - `tie_word_embeddings=True` — 입력 Embedding과 출력 `lm_head`가 같은 가중치 (3절)

    실습도 `train_pretrain.py`가 만드는 설정을 그대로 씁니다.

    ```python
    lm_config = MiniMindConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers,
                               use_moe=bool(args.use_moe))   # 기본 인자: 768, 8, 0
    ```
    """)
    return


@app.cell
def _(MiniMindConfig, MiniMindForCausalLM, device, torch):
    # train_pretrain.py 기본 인자와 같은 설정 (실제 MiniMind 크기)
    lm_config = MiniMindConfig(
        hidden_size=768,          # D   (--hidden_size)
        num_hidden_layers=8,      # L   (--num_hidden_layers)
        use_moe=False,            #     (--use_moe 0)
    )  # 나머지(H=8, KV=4, V=6400 ...)는 MiniMindConfig 기본값

    # 시드를 이 블록 안에서만 고정해, 다시 실행해도 같은 무작위 가중치가 나오게 합니다.
    # 가중치는 CPU에서 만든 뒤 GPU로 옮깁니다(init_model의 model.to(device)와 같음).
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        model = MiniMindForCausalLM(lm_config).to(device).eval()
    return lm_config, model


@app.cell(hide_code=True)
def _(device, lm_config, mo, model, torch):
    _keys = [
        ("hidden_size", "D, 토큰 벡터의 길이"),
        ("num_hidden_layers", "L, 블록 수"),
        ("num_attention_heads", "H, Query head 수"),
        ("num_key_value_heads", "KV, Key/Value head 수"),
        ("head_dim", "d, head 하나의 차원"),
        ("intermediate_size", "FeedForward 중간 차원"),
        ("vocab_size", "V, 어휘 수"),
        ("max_position_embeddings", "RoPE 표를 미리 만드는 최대 위치"),
        ("rope_theta", "RoPE 주파수의 밑"),
        ("rms_norm_eps", "RMSNorm의 0 나눗셈 방지값"),
        ("tie_word_embeddings", "Embedding과 lm_head 가중치 공유"),
        ("use_moe", "FeedForward 대신 MoE 사용 (11절)"),
    ]
    _param = next(model.parameters())
    _gpu = f" ({torch.cuda.get_device_name(device)})" if device.type == "cuda" else ""
    mo.vstack([
        mo.ui.table(
            [{"설정": k, "값": str(getattr(lm_config, k)), "의미": desc} for k, desc in _keys],
            selection=None,
            page_size=20,
        ),
        mo.md(f"모델 위치 `{_param.device}`{_gpu}, 가중치 dtype `{_param.dtype}`"),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 1-1. 매개변수는 어디에 있을까?

    원본 클래스의 `nn.Linear`, `nn.Embedding`, `RMSNorm` 크기를 그대로 세어 보면 다음과 같습니다.
    (`nn.Linear(a, b)`의 가중치는 `a × b`개, 이 모델의 Linear에는 bias가 없습니다.)

    | 부품 | 개수 |
    | --- | --- |
    | Embedding (= lm_head, 공유) | `V × D` |
    | Attention (층마다) | `D·H·d` (Q) + `2·D·KV·d` (K, V) + `H·d·D` (O) + `2·d` (QK-norm) |
    | FeedForward (층마다) | `3 × D × intermediate_size` (gate, up, down) |
    | RMSNorm | 층마다 `2·D` + 마지막 `D` |

    아래 함수로 계산한 값이 실제 모델의 매개변수 수와 같은지 확인합니다.
    """)
    return


@app.cell
def _(lm_config, model):
    def count_params(c):
        D, H, KV, d = c.hidden_size, c.num_attention_heads, c.num_key_value_heads, c.head_dim
        embedding = c.vocab_size * D
        attention = D * H * d + 2 * D * KV * d + H * d * D + 2 * d
        feedforward = 3 * D * c.intermediate_size
        norms = 2 * D
        total = embedding + c.num_hidden_layers * (attention + feedforward + norms) + D
        if not c.tie_word_embeddings:
            total += c.vocab_size * D  # 공유하지 않으면 lm_head가 따로 생깁니다.
        return {
            "Embedding (=lm_head)": embedding,
            "Attention × L": c.num_hidden_layers * attention,
            "FeedForward × L": c.num_hidden_layers * feedforward,
            "RMSNorm 전체": c.num_hidden_layers * norms + D,
            "합계": total,
        }

    # parameters()는 공유된 가중치를 한 번만 셉니다.
    model_param_count = sum(p.numel() for p in model.parameters())
    assert count_params(lm_config)["합계"] == model_param_count
    return (count_params,)


@app.cell(hide_code=True)
def _(count_params, lm_config, mo):
    _d = count_params(lm_config)
    mo.vstack([
        mo.ui.table(
            [{"부품": k, "매개변수": f"{_d[k]:,}", "비율": f"{_d[k] / _d['합계']:.1%}"} for k in _d],
            selection=None,
        ),
        mo.md(
            f"**관찰:** MiniMind는 약 **{_d['합계'] / 1e6:.1f}M** 개의 매개변수를 가집니다. "
            "FeedForward가 가장 크고, 어휘가 6,400개로 작아서 Embedding 비중은 작습니다. "
            "공식으로 센 값과 실제 모델의 `parameters()` 합이 같다는 검증을 통과했습니다. "
            f"fp32 가중치는 약 {_d['합계'] * 4 / 2**20:.0f} MB입니다."
        ),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. 문장 → 토큰 ID

    모델은 글자를 모릅니다. **토크나이저**가 문장을 어휘 표의 번호(토큰 ID)로 바꿉니다.
    MiniMind 토크나이저(`projects/minimind/model/tokenizer.json`)는 어휘 6,400개의 **바이트 단위 BPE**입니다.

    - 영어 단어는 대부분 한 토큰입니다. 단어 앞의 공백은 `Ġ`로 표시됩니다.
    - 길이가 다른 문장을 한 배치로 묶으려면 짧은 쪽 뒤를 `<|endoftext|>`(패딩)로 채우고,
      `attention_mask`에 실제 토큰은 1, 패딩은 0으로 표시합니다.

    첫 문장은 아래 입력란에서 바꿀 수 있습니다. 바꾸면 아래 모든 절이 다시 계산됩니다.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    sentence_input = mo.ui.text(value="The cat sat on the mat.", label="첫 문장", full_width=True)
    sentence_input
    return (sentence_input,)


@app.cell
def _(AutoTokenizer, device, minimind_dir, sentence_input):
    tokenizer = AutoTokenizer.from_pretrained(minimind_dir / "model")
    tokenizer.padding_side = "right"  # 패딩을 문장 뒤에 붙입니다.

    sentences = [sentence_input.value.strip() or "The cat sat on the mat.", "Hello world."]
    batch = tokenizer(sentences, padding=True, return_tensors="pt")
    input_ids = batch.input_ids.to(device)            # [B, T]  토큰 번호
    attention_mask = batch.attention_mask.to(device)  # [B, T]  1 = 실제 토큰, 0 = 패딩
    batch_size, seq_len = input_ids.shape

    # 한글은 UTF-8 바이트 단위로 쪼개져 글자 수보다 토큰이 많아집니다.
    korean_text = "안녕하세요"
    korean_ids = tokenizer(korean_text).input_ids
    return attention_mask, batch_size, input_ids, korean_ids, korean_text, seq_len, tokenizer


@app.cell(hide_code=True)
def _(attention_mask, input_ids, korean_ids, korean_text, mo, tokenizer):
    def _rows(b):
        return [
            {"위치": t, "토큰 ID": int(i), "토큰": tokenizer.convert_ids_to_tokens(int(i)),
             "mask": int(attention_mask[b, t])}
            for t, i in enumerate(input_ids[b])
        ]

    mo.vstack([
        mo.md(f"`input_ids` shape: `{tuple(input_ids.shape)}` = `[B, T]`"),
        mo.hstack([
            mo.vstack([mo.md("**문장 0**"), mo.ui.table(_rows(0), selection=None, page_size=20)]),
            mo.vstack([mo.md("**문장 1**"), mo.ui.table(_rows(1), selection=None, page_size=20)]),
        ], widths="equal"),
        mo.md(
            f"한글 `{korean_text}`({len(korean_text)}글자) → 토큰 **{len(korean_ids)}개**. "
            "작은 어휘에서는 한글 한 글자가 여러 바이트 토큰이 됩니다."
        ),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. 토큰 ID → 벡터: Embedding과 lm_head

    `embed_tokens`는 `[V, D]` 크기의 표입니다. 토큰 ID가 `i`이면 **`i`번째 행**을 꺼냅니다.
    계산이 아니라 표에서 행을 고르는 것(lookup)입니다.

    모델 마지막의 `lm_head`는 반대 방향입니다. 은닉 벡터 `h [D]`를 표의 **모든 행과 내적**해
    어휘별 점수 `[V]`를 만듭니다. MiniMind는 두 곳에 **같은 표**를 씁니다
    (`self.model.embed_tokens.weight = self.lm_head.weight`). 그래서 기본 설정에서 약 490만 개의 매개변수를 아낍니다.
    """)
    return


@app.cell
def _(input_ids, model, torch):
    embed = model.model.embed_tokens
    with torch.no_grad():
        embeddings = embed(input_ids)         # [B, T, D]
        embeddings_manual = embed.weight[input_ids]  # 표에서 행을 직접 꺼내기

    assert torch.equal(embeddings, embeddings_manual)
    # 같은 Parameter 객체: 하나를 바꾸면 다른 쪽도 바뀝니다.
    assert model.lm_head.weight is embed.weight
    return (embeddings,)


@app.cell(hide_code=True)
def _(embeddings, mo, model):
    mo.md(f"""
    Embedding 표 `{tuple(model.model.embed_tokens.weight.shape)}` = `[V, D]` →
    `embeddings` `{tuple(embeddings.shape)}` = `[B, T, D]`.
    직접 꺼낸 행과 같고, `lm_head.weight is embed_tokens.weight`도 확인했습니다.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. `RMSNorm`: 벡터의 크기 맞추기

    층을 여러 번 지나면 벡터 값이 점점 커지거나 작아질 수 있습니다. RMSNorm은 각 토큰 벡터를
    **제곱 평균의 제곱근(RMS)** 으로 나누어 크기를 1 근처로 맞춘 뒤, 학습되는 `weight`를 곱합니다.

    $$
    \mathrm{RMSNorm}(x) = w \odot \frac{x}{\sqrt{\frac{1}{D}\sum_j x_j^2 + \epsilon}}
    $$

    - LayerNorm과 달리 **평균을 빼지 않고** bias도 없습니다. 계산이 더 단순합니다.
    - 원본은 정밀도를 위해 `float32`로 계산한 뒤 원래 dtype으로 되돌립니다(`.float()` → `.type_as(x)`).
    - 블록 안에서는 Attention·FeedForward **앞**에 둡니다(Pre-Norm, 8절).
    """)
    return


@app.cell
def _(embeddings, model, torch):
    norm_module = model.model.layers[0].input_layernorm

    def rms_norm(x, weight, eps):
        x32 = x.float()
        rms = torch.sqrt(x32.pow(2).mean(dim=-1, keepdim=True) + eps)  # 토큰마다 하나의 값
        return (weight * (x32 / rms)).type_as(x)

    with torch.no_grad():
        normed_manual = rms_norm(embeddings, norm_module.weight, norm_module.eps)
        normed = norm_module(embeddings)

    assert torch.allclose(normed, normed_manual, atol=1e-6)
    rms_before = embeddings.pow(2).mean(-1).sqrt()[0]  # 문장 0, 토큰별 RMS
    rms_after = normed.pow(2).mean(-1).sqrt()[0]
    return rms_after, rms_before


@app.cell(hide_code=True)
def _(mo, rms_after, rms_before):
    mo.vstack([
        mo.md("문장 0의 토큰별 RMS (weight가 아직 1이므로 정규화 후 RMS는 1입니다)"),
        mo.ui.table(
            [{"위치": t, "정규화 전": f"{a:.4f}", "정규화 후": f"{b:.4f}"}
             for t, (a, b) in enumerate(zip(rms_before.tolist(), rms_after.tolist()))],
            selection=None, page_size=20,
        ),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. RoPE: 위치를 "회전"으로 알려 주기

    Attention 점수는 Query와 Key의 내적입니다. 그런데 내적만으로는 토큰의 **순서**를 알 수 없습니다.
    RoPE(Rotary Position Embedding)는 Q와 K 벡터를 **위치에 비례한 각도만큼 회전**시켜 순서 정보를 넣습니다.

    ### 5-1. 각도 표 만들기: `precompute_freqs_cis`

    head 차원 `d`를 두 개씩 짝지어 `d/2`개의 2차원 평면으로 보고, 짝 `i`마다 다른 회전 속도(주파수)를 줍니다.

    $$
    f_i = \frac{1}{\theta^{\,2i/d}}, \qquad \text{위치 } m \text{의 각도} = m \cdot f_i
    $$

    - `i`가 작으면 빠르게 회전(가까운 위치 구분), 크면 느리게 회전(먼 위치 구분)합니다.
    - MiniMind는 `θ = rope_theta = 1,000,000`을 씁니다. 긴 문맥에서도 각도가 덜 겹칩니다.
    - 원본은 `[위치, d/2]` 각도의 cos/sin을 **두 번 이어 붙여** `[위치, d]` 표로 만들고,
      모델을 만들 때 한 번 계산해 `freqs_cos`, `freqs_sin` 버퍼로 저장합니다.
    - `inference_rope_scaling=True`이면 YaRN으로 주파수를 늘려 학습 길이보다 긴 입력을 다룹니다. 여기서는 꺼 둡니다.
    """)
    return


@app.cell
def _(precompute_freqs_cis, lm_config, torch):
    rope_dim = lm_config.head_dim
    rope_base = lm_config.rope_theta
    rope_positions = 64

    # ① 짝 i = 0..d/2-1의 주파수
    rope_inv_freq = 1.0 / (rope_base ** (torch.arange(0, rope_dim, 2).float() / rope_dim))  # [d/2]
    # ② 위치 m × 주파수
    rope_angles = torch.outer(torch.arange(rope_positions).float(), rope_inv_freq)  # [T, d/2]
    # ③ cos/sin을 두 번 이어 붙여 [T, d]
    rope_cos_manual = torch.cat([rope_angles.cos(), rope_angles.cos()], dim=-1)
    rope_sin_manual = torch.cat([rope_angles.sin(), rope_angles.sin()], dim=-1)

    rope_cos, rope_sin = precompute_freqs_cis(rope_dim, end=rope_positions, rope_base=rope_base)
    assert torch.allclose(rope_cos, rope_cos_manual, atol=1e-6)
    assert torch.allclose(rope_sin, rope_sin_manual, atol=1e-6)
    return rope_angles, rope_cos, rope_dim, rope_inv_freq, rope_sin


@app.cell(hide_code=True)
def _(mo, plt, rope_cos, rope_dim, rope_inv_freq):
    _fig, _axes = plt.subplots(1, 2, figsize=(10, 3.2))
    _im = _axes[0].imshow(rope_cos[:, : rope_dim // 2].T, aspect="auto", cmap="coolwarm", vmin=-1, vmax=1)
    _axes[0].set_xlabel("position m")
    _axes[0].set_ylabel("pair i")
    _axes[0].set_title("cos(m * f_i)")
    _fig.colorbar(_im, ax=_axes[0])
    _axes[1].semilogy(rope_inv_freq.numpy(), "o-")
    _axes[1].set_xlabel("pair i")
    _axes[1].set_title("frequency f_i (log scale)")
    _fig.tight_layout()
    _plot = mo.as_html(_fig)
    plt.close(_fig)
    mo.vstack([
        _plot,
        mo.md("**관찰:** 위쪽 짝(i=0)은 위치마다 색이 빠르게 바뀌고, 아래쪽 짝은 64개 위치 동안 거의 변하지 않습니다."),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 5-2. 회전 적용하기: `apply_rotary_pos_emb`

    원본은 벡터를 **앞 절반 `a`와 뒤 절반 `b`** 로 나누어 `(a_i, b_i)`를 한 평면의 좌표로 봅니다.
    `rotate_half(x) = [-b, a]`를 이용한 식 `x·cos + rotate_half(x)·sin`을 풀어 쓰면 평범한 2차원 회전입니다.

    $$
    \begin{pmatrix} a' \\ b' \end{pmatrix} =
    \begin{pmatrix} \cos\phi & -\sin\phi \\ \sin\phi & \cos\phi \end{pmatrix}
    \begin{pmatrix} a \\ b \end{pmatrix}
    $$

    회전의 중요한 성질: Q를 위치 `m`, K를 위치 `n`에서 회전하면 내적은 **거리 `m − n`에만** 달라집니다.
    즉 모델은 절대 위치가 아니라 **상대 거리**를 봅니다. 아래에서 확인합니다.
    """)
    return


@app.cell
def _(apply_rotary_pos_emb, rope_angles, rope_cos, rope_dim, rope_sin, torch):
    def rotate_at(vec, pos):
        """head 벡터 하나 [d]를 위치 pos의 각도로 회전합니다 (직접 계산)."""
        a, b = vec[: rope_dim // 2], vec[rope_dim // 2:]
        c, s = rope_angles[pos].cos(), rope_angles[pos].sin()
        return torch.cat([a * c - b * s, a * s + b * c])

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(1)
        demo_q, demo_k = torch.randn(rope_dim), torch.randn(rope_dim)

    # 원본 함수는 [B, T, H, d] 모양을 받으므로 앞에 축을 붙여 넣습니다.
    _q4 = demo_q.view(1, 1, 1, rope_dim)
    _q_rot_ref, _ = apply_rotary_pos_emb(_q4, _q4, rope_cos[5:6], rope_sin[5:6])
    assert torch.allclose(rotate_at(demo_q, 5), _q_rot_ref.flatten(), atol=1e-5)
    # 회전은 길이를 바꾸지 않습니다.
    assert torch.allclose(rotate_at(demo_q, 5).norm(), demo_q.norm(), atol=1e-5)

    # (Q 위치, K 위치) 쌍마다 회전 후 내적
    rope_pairs = [(2, 0), (7, 5), (40, 38), (3, 0), (10, 7), (50, 47), (0, 0), (30, 30)]
    rope_scores = [float(rotate_at(demo_q, m) @ rotate_at(demo_k, n)) for m, n in rope_pairs]
    assert abs(rope_scores[0] - rope_scores[2]) < 1e-4  # 거리 2끼리 같음
    return rope_pairs, rope_scores


@app.cell(hide_code=True)
def _(mo, rope_pairs, rope_scores):
    mo.vstack([
        mo.md("같은 Q·K 벡터를 여러 위치에 놓고 회전한 뒤의 내적:"),
        mo.ui.table(
            [{"Q 위치 m": m, "K 위치 n": n, "거리 m−n": m - n, "회전 후 내적": f"{s:.5f}"}
             for (m, n), s in zip(rope_pairs, rope_scores)],
            selection=None,
        ),
        mo.md("**관찰:** 절대 위치가 달라도 **거리가 같으면 점수가 같습니다.** 직접 회전한 값은 원본 `apply_rotary_pos_emb`와 같습니다."),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. `Attention`: 토큰끼리 정보 주고받기

    Attention은 각 토큰이 **앞쪽 토큰들 중 어디를 얼마나 참고할지** 정해 정보를 모으는 연산입니다.
    원본 `Attention.forward`를 다음 순서로 나누어 직접 계산합니다.

    | 단계 | 원본 코드 | shape |
    | --- | --- | --- |
    | ① 투영 | `q_proj`, `k_proj`, `v_proj` 후 `view` | Q `[B,T,H,d]`, K·V `[B,T,KV,d]` |
    | ② QK-norm | `q_norm`, `k_norm` (head마다 RMSNorm) | 그대로 |
    | ③ RoPE | `apply_rotary_pos_emb` (Q, K만) | 그대로 |
    | ④ GQA | `repeat_kv`로 K·V head를 H개로 복사 | K·V `[B,T,H,d]` |
    | ⑤ 점수 | `Q @ Kᵀ / √d` + causal mask + padding mask | `[B,H,T,T]` |
    | ⑥ 가중합 | `softmax(점수) @ V` → head 이어 붙이기 → `o_proj` | `[B,T,D]` |

    입력은 1층의 `input_layernorm`을 통과한 벡터입니다.

    ### 6-1. 투영, QK-norm, RoPE

    - 은닉 벡터 `[D]` 하나에서 Query("무엇을 찾나"), Key("나는 무엇인가"), Value("전달할 내용")를 만듭니다.
    - 마지막 축을 `[head 수, d]`로 나누면 head마다 독립된 작은 Attention이 됩니다.
    - **QK-norm:** Q와 K를 head마다 RMSNorm해 내적 값이 지나치게 커지는 것을 막습니다. V는 정규화하지 않습니다.
    - RoPE는 **Q와 K에만** 적용합니다. V는 내용 전달용이라 위치 회전이 필요 없습니다.
    """)
    return


@app.cell
def _(apply_rotary_pos_emb, attention_mask, embeddings, seq_len, model, torch):
    layer0 = model.model.layers[0]
    attn = layer0.self_attn
    n_heads, n_kv_heads, head_dim = attn.n_local_heads, attn.n_local_kv_heads, attn.head_dim

    with torch.no_grad():
        attn_input = layer0.input_layernorm(embeddings)          # [B, T, D]
        _B, _T, _ = attn_input.shape

        # ① 투영 후 마지막 축을 [head 수, d]로 나누기
        q = attn.q_proj(attn_input).view(_B, _T, n_heads, head_dim)     # [B, T, H, d]
        k = attn.k_proj(attn_input).view(_B, _T, n_kv_heads, head_dim)  # [B, T, KV, d]
        v = attn.v_proj(attn_input).view(_B, _T, n_kv_heads, head_dim)  # [B, T, KV, d]

        # ② QK-norm
        q, k = attn.q_norm(q), attn.k_norm(k)

        # ③ RoPE: 모델에 저장된 표에서 위치 0..T-1을 잘라 씁니다.
        position_embeddings = (
            model.model.freqs_cos[:seq_len],
            model.model.freqs_sin[:seq_len],
        )
        q_rot, k_rot = apply_rotary_pos_emb(q, k, *position_embeddings)
    return attn, attn_input, head_dim, k_rot, n_heads, n_kv_heads, position_embeddings, q_rot, v


@app.cell(hide_code=True)
def _(attn, attn_input, k_rot, mo, q_rot, v):
    mo.md(f"""
    입력 `{tuple(attn_input.shape)}` →
    `q_proj` 가중치 `{tuple(attn.q_proj.weight.shape)}`, `k_proj`·`v_proj` 가중치 `{tuple(attn.k_proj.weight.shape)}`
    (`nn.Linear`의 가중치는 `[출력, 입력]` 순서) →
    Q `{tuple(q_rot.shape)}`, K `{tuple(k_rot.shape)}`, V `{tuple(v.shape)}`.
    **Key/Value 투영이 Query의 절반 크기**입니다. 이것이 다음 절의 GQA입니다.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 6-2. GQA: Key/Value head 공유 — `repeat_kv`

    **Grouped-Query Attention**은 Query head 여러 개가 Key/Value head 하나를 함께 씁니다.
    MiniMind 설정은 Query head 8개, K/V head 4개이므로 `n_rep = 8 / 4 = 2`입니다.

    ```text
    Q head:   0   1   2   3   4   5   6   7
              └─┬─┘   └─┬─┘   └─┬─┘   └─┬─┘
    K/V head:   0       1       2       3
    ```

    `repeat_kv`는 계산을 위해 K/V head를 `n_rep`번 **복사**만 합니다. 새 가중치는 없습니다.
    이득은 **생성할 때 저장하는 KV cache가 줄어드는 것**입니다(10절).
    """)
    return


@app.cell
def _(attn, k_rot, n_heads, repeat_kv, torch, v):
    k_rep = repeat_kv(k_rot, attn.n_rep)  # [B, T, KV, d] → [B, T, H, d]
    v_rep = repeat_kv(v, attn.n_rep)

    # Query head h는 K/V head h // n_rep을 봅니다.
    for _h in range(n_heads):
        assert torch.equal(k_rep[:, :, _h], k_rot[:, :, _h // attn.n_rep])
    return k_rep, v_rep


@app.cell(hide_code=True)
def _(attn, k_rep, k_rot, mo, n_heads):
    mo.vstack([
        mo.md(f"`n_rep = {attn.n_rep}`, K `{tuple(k_rot.shape)}` → `repeat_kv` → `{tuple(k_rep.shape)}`"),
        mo.ui.table([{"Query head": h, "사용하는 K/V head": h // attn.n_rep} for h in range(n_heads)], selection=None),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 6-3. 점수, 마스크, 가중합

    $$
    \mathrm{Attention}(Q,K,V) = \mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d}} + M\right) V
    $$

    - `√d`로 나누어 head 차원이 커져도 점수 크기가 일정하게 합니다.
    - **causal mask:** 위치 `t`는 자기보다 뒤(미래) 토큰을 보면 안 됩니다. 대각선 위를 `-inf`로 채웁니다(`triu(1)`).
      softmax 후 그 칸의 확률은 0이 됩니다.
    - **padding mask:** `attention_mask`가 0인 열(패딩 Key)에 `-1e9`를 더해 무시합니다.
    - 원본은 패딩이 없고 캐시도 없으면 같은 계산을 빠르게 하는 `F.scaled_dot_product_attention`을 씁니다.
      패딩이 있는 지금은 아래와 같은 직접 계산 경로를 탑니다.
    """)
    return


@app.cell
def _(F, attention_mask, attn, attn_input, head_dim, k_rep, math, position_embeddings, q_rot, torch, v_rep):
    with torch.no_grad():
        # head 축을 앞으로: [B, T, H, d] → [B, H, T, d]
        qh, kh, vh = q_rot.transpose(1, 2), k_rep.transpose(1, 2), v_rep.transpose(1, 2)
        _T = qh.shape[2]

        # ⑤ 점수 [B, H, T(query), T(key)]
        scores = (qh @ kh.transpose(-2, -1)).float() / math.sqrt(head_dim)
        causal_mask = torch.full((_T, _T), float("-inf"), device=qh.device).triu(1)            # 대각선 위 = 미래
        scores = scores + causal_mask
        scores = scores + (1.0 - attention_mask[:, None, None, :].float()) * -1e9  # 패딩 Key 무시

        # ⑥ 확률 → V 가중합 → head 이어 붙이기 → o_proj
        attn_probs = F.softmax(scores, dim=-1)                                 # 각 행의 합 = 1
        _context = attn_probs.type_as(vh) @ vh                                 # [B, H, T, d]
        _context = _context.transpose(1, 2).reshape(qh.shape[0], _T, -1)       # [B, T, H·d]
        attn_output_manual = attn.o_proj(_context)                             # [B, T, D]

        # 원본 Attention.forward와 비교
        attn_output, _ = attn(attn_input, position_embeddings, attention_mask=attention_mask)

    assert torch.allclose(attn_output, attn_output_manual, atol=1e-5)
    assert torch.allclose(attn_probs.sum(-1), torch.ones_like(attn_probs.sum(-1)), atol=1e-5)
    return attn_output, attn_probs, causal_mask


@app.cell(hide_code=True)
def _(attn_output, causal_mask, mo):
    mo.vstack([
        mo.md(f"직접 계산이 원본 `Attention.forward`와 같습니다. 출력 shape `{tuple(attn_output.shape)}` = `[B, T, D]` (입력과 같은 모양)."),
        mo.md(f"causal mask (앞 6×6)\n```text\n{causal_mask[:6, :6]}\n```"),
    ])
    return


@app.cell(hide_code=True)
def _(batch_size, mo, n_heads):
    sample_select = mo.ui.dropdown(options={f"문장 {b}": b for b in range(batch_size)}, value="문장 0", label="문장")
    head_select = mo.ui.dropdown(options={f"head {h}": h for h in range(n_heads)}, value="head 0", label="Query head")
    mo.hstack([sample_select, head_select], justify="start")
    return head_select, sample_select


@app.cell(hide_code=True)
def _(attn_probs, head_select, input_ids, mo, plt, sample_select, tokenizer):
    _b, _h = sample_select.value, head_select.value
    _labels = [tokenizer.convert_ids_to_tokens(int(i)) for i in input_ids[_b]]
    _fig, _ax = plt.subplots(figsize=(5.5, 4.5))
    _im = _ax.imshow(attn_probs[_b, _h].cpu(), cmap="Blues", vmin=0, vmax=1)
    _ax.set_xticks(range(len(_labels)), _labels, rotation=60, fontsize=8)
    _ax.set_yticks(range(len(_labels)), _labels, fontsize=8)
    _ax.set_xlabel("key (attended to)")
    _ax.set_ylabel("query")
    _fig.colorbar(_im, ax=_ax)
    _fig.tight_layout()
    _plot = mo.as_html(_fig)
    plt.close(_fig)
    mo.vstack([
        _plot,
        mo.md(
            "**읽는 법:** 행은 정보를 모으는 토큰, 열은 참고되는 토큰입니다. 대각선 위(미래)는 항상 0입니다. "
            "문장 1을 고르면 패딩 열(`<|endoftext|>`)도 0이 됩니다. 가중치가 무작위라 패턴 자체에는 의미가 없습니다."
        ),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7. `FeedForward`: 토큰마다 따로 가공 (SwiGLU)

    Attention이 토큰 **사이**의 정보를 섞었다면, FeedForward는 각 토큰 벡터를 **혼자서** 가공합니다.
    MiniMind는 SwiGLU 구조를 씁니다.

    $$
    \mathrm{FFN}(x) = W_{down}\big(\,\mathrm{SiLU}(W_{gate}\,x) \odot W_{up}\,x\,\big),
    \qquad \mathrm{SiLU}(z) = z \cdot \sigma(z)
    $$

    - `gate_proj`, `up_proj`: `D → intermediate_size`로 넓힙니다.
    - SiLU를 거친 gate 값이 up 값의 각 성분을 **얼마나 통과시킬지** 정합니다(⊙는 원소별 곱).
    - `down_proj`: 다시 `D`로 줄입니다.

    입력은 Attention 결과에 잔차를 더한 `h`를 `post_attention_layernorm`으로 정규화한 값입니다.
    """)
    return


@app.cell
def _(F, attn_output, embeddings, model, torch):
    ffn = model.model.layers[0].mlp
    with torch.no_grad():
        hidden_after_attn = embeddings + attn_output  # 잔차 연결 (8절)
        ffn_input = model.model.layers[0].post_attention_layernorm(hidden_after_attn)

        gate = F.silu(ffn.gate_proj(ffn_input))       # [B, T, intermediate]
        up = ffn.up_proj(ffn_input)                   # [B, T, intermediate]
        ffn_output_manual = ffn.down_proj(gate * up)  # [B, T, D]
        ffn_output = ffn(ffn_input)

    assert torch.allclose(ffn_output, ffn_output_manual, atol=1e-6)

    # 위치 독립성: 토큰 하나만 따로 넣어도 같은 결과가 나옵니다.
    with torch.no_grad():
        _single = ffn(ffn_input[:, 2:3])
    assert torch.allclose(_single, ffn_output[:, 2:3], atol=1e-6)
    return ffn, ffn_output, gate, hidden_after_attn


@app.cell(hide_code=True)
def _(ffn, ffn_output, gate, mo, plt, torch):
    _z = torch.linspace(-5, 5, 200)
    _fig, _ax = plt.subplots(figsize=(4.5, 2.8))
    _ax.plot(_z, torch.nn.functional.silu(_z), label="SiLU(z)")
    _ax.plot(_z, torch.relu(_z), "--", label="ReLU(z)")
    _ax.axhline(0, color="gray", lw=0.5)
    _ax.legend()
    _ax.set_title("SiLU vs ReLU")
    _fig.tight_layout()
    _plot = mo.as_html(_fig)
    plt.close(_fig)
    mo.hstack([
        _plot,
        mo.md(f"""
    `gate_proj` `{tuple(ffn.gate_proj.weight.shape)}`, `up_proj` `{tuple(ffn.up_proj.weight.shape)}`,
    `down_proj` `{tuple(ffn.down_proj.weight.shape)}`

    gate `{tuple(gate.shape)}` → 출력 `{tuple(ffn_output.shape)}`

    SiLU는 ReLU와 비슷하지만 0 근처에서 부드럽고, 음수 입력도 조금 통과시킵니다.

    토큰 하나만 넣어도 같은 결과가 나왔습니다. **FeedForward는 위치를 섞지 않습니다.**
    """),
    ], widths=[1, 1], align="center")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 8. `MiniMindBlock`: Pre-Norm과 잔차 연결

    블록 하나는 지금까지의 부품을 다음 두 줄로 조립합니다.

    ```python
    h   = x + Attention(input_layernorm(x))            # 원본: hidden_states += residual
    out = h + FeedForward(post_attention_layernorm(h))
    ```

    - **잔차 연결(`x + ...`):** 각 부품은 입력을 바꾸는 대신 **고칠 양(차이)** 만 더합니다.
      층이 깊어져도 원래 정보와 기울기가 그대로 흐르는 지름길이 생깁니다.
    - **Pre-Norm:** 정규화를 부품 **앞**에 둡니다. 잔차 경로 자체는 정규화하지 않으므로 학습이 안정적입니다.
    - 입력과 출력의 shape이 `[B, T, D]`로 같아서 블록을 `L`번 쌓을 수 있습니다.
    """)
    return


@app.cell
def _(attention_mask, embeddings, ffn_output, hidden_after_attn, position_embeddings, model, torch):
    with torch.no_grad():
        block_output_manual = hidden_after_attn + ffn_output  # 6, 7절 결과 조립
        block_output, _ = model.model.layers[0](
            embeddings, position_embeddings, attention_mask=attention_mask
        )
    assert torch.allclose(block_output, block_output_manual, atol=1e-5)
    return (block_output,)


@app.cell(hide_code=True)
def _(block_output, embeddings, mo):
    _change = (block_output - embeddings).norm(dim=-1).mean() / embeddings.norm(dim=-1).mean()
    mo.md(f"""
    6·7절의 결과를 조립한 값이 원본 `MiniMindBlock.forward`와 같습니다.
    입력 `{tuple(embeddings.shape)}` → 출력 `{tuple(block_output.shape)}`.
    블록이 더한 변화량은 입력 크기의 약 **{_change:.0%}** 입니다.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 9. 전체 조립: `MiniMindModel`과 `MiniMindForCausalLM`

    `MiniMindModel.forward`는 Embedding → 블록 `L`개 → 마지막 `norm`을 차례로 실행합니다.
    `MiniMindForCausalLM`은 여기에 `lm_head`를 붙여 **어휘별 점수(logits)** 를 만들고, 정답이 있으면 손실도 계산합니다.

    ### 9-1. 한 번의 forward에서 shape 따라가기

    원본 모듈에 forward hook을 걸어 각 모듈의 출력 shape을 기록합니다.
    """)
    return


@app.cell
def _(attention_mask, input_ids, position_embeddings, model, torch):
    forward_trace = []

    def _make_hook(name):
        def _hook(module, inputs, output):
            out = output[0] if isinstance(output, tuple) else output
            forward_trace.append({"모듈": name, "클래스": type(module).__name__, "출력 shape": str(tuple(out.shape))})
        return _hook

    _names = [
        "model.embed_tokens",
        "model.layers.0.input_layernorm", "model.layers.0.self_attn",
        "model.layers.0.post_attention_layernorm", "model.layers.0.mlp", "model.layers.0",
        "model.layers.1",
        "model.norm", "lm_head",
    ]
    _handles = [model.get_submodule(n).register_forward_hook(_make_hook(n)) for n in _names]
    try:
        with torch.no_grad():
            model_out = model(input_ids, attention_mask=attention_mask)
    finally:
        for _h in _handles:
            _h.remove()

    # 같은 계산을 직접 조립해 원본 logits와 비교합니다.
    with torch.no_grad():
        _h = model.model.embed_tokens(input_ids)
        for _layer in model.model.layers:
            _h, _ = _layer(_h, position_embeddings, attention_mask=attention_mask)
        _h = model.model.norm(_h)
        logits_manual = model.lm_head(_h)  # [B, T, V]

    assert torch.allclose(model_out.logits, logits_manual, atol=1e-5)
    return forward_trace, model_out


@app.cell(hide_code=True)
def _(forward_trace, mo, model_out):
    mo.vstack([
        mo.ui.table(forward_trace, selection=None, page_size=20),
        mo.md(
            f"최종 logits `{tuple(model_out.logits.shape)}` = `[B, T, V]`: "
            "**모든 위치**에서 다음 토큰 후보 6,400개의 점수를 한꺼번에 냅니다. 직접 조립한 결과와 원본이 같습니다."
        ),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 9-2. 다음 토큰 예측 손실

    위치 `t`의 logits는 **위치 `t+1`의 토큰**을 맞히는 데 씁니다. 원본 `forward`는 `labels`를 받으면
    한 칸 밀어서 비교합니다.

    ```python
    x, y = logits[..., :-1, :], labels[..., 1:]      # 마지막 logits와 첫 정답은 짝이 없음
    loss = F.cross_entropy(x.view(-1, V), y.view(-1), ignore_index=-100)
    ```

    패딩 위치의 정답은 `-100`으로 바꿔 손실에서 뺍니다. 무작위 모델은 6,400개 중 하나를 거의 고르게 찍으므로
    손실이 대략 `ln(6400) ≈ 8.76` 근처에서 시작합니다.
    """)
    return


@app.cell
def _(F, attention_mask, input_ids, math, model, torch):
    labels = input_ids.masked_fill(attention_mask == 0, -100)  # 패딩 정답은 무시

    with torch.no_grad():
        loss_out = model(input_ids, attention_mask=attention_mask, labels=labels)
        _logits_t = loss_out.logits[:, :-1]  # 위치 0..T-2의 예측
        _labels_next = labels[:, 1:]         # 위치 1..T-1의 정답
        loss_manual = F.cross_entropy(
            _logits_t.reshape(-1, _logits_t.size(-1)), _labels_next.reshape(-1), ignore_index=-100
        )
        next_token_probs = F.softmax(_logits_t[0], dim=-1)  # 문장 0, [T-1, V]

    assert torch.allclose(loss_out.loss, loss_manual, atol=1e-5)
    uniform_loss = math.log(model.config.vocab_size)
    return labels, loss_out, next_token_probs, uniform_loss


@app.cell(hide_code=True)
def _(input_ids, labels, loss_out, mo, next_token_probs, tokenizer, uniform_loss):
    _rows = []
    for _t in range(next_token_probs.shape[0]):
        _target = int(labels[0, _t + 1])
        if _target == -100:
            continue
        _rows.append({
            "입력 위치 t": _t,
            "지금까지 마지막 토큰": tokenizer.convert_ids_to_tokens(int(input_ids[0, _t])),
            "맞혀야 할 다음 토큰": tokenizer.convert_ids_to_tokens(_target),
            "정답 확률": f"{float(next_token_probs[_t, _target]):.6f}",
            "가장 높은 후보": tokenizer.convert_ids_to_tokens(int(next_token_probs[_t].argmax())),
        })
    mo.vstack([
        mo.ui.table(_rows, selection=None, page_size=20),
        mo.md(f"원본 손실 `{float(loss_out.loss):.4f}` = 직접 계산과 같음. 균등 분포 손실 `ln(V)` = `{uniform_loss:.4f}`. "
              "학습은 이 정답 확률들을 높이는 방향으로 가중치를 바꿉니다."),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 10. 생성과 KV cache

    생성은 "다음 토큰 하나 고르기 → 입력 끝에 붙이기"를 반복합니다. 매번 문장 전체를 다시 계산하면 낭비이므로,
    이미 계산한 토큰의 **Key와 Value를 저장(KV cache)** 해 두고 새 토큰 하나만 계산합니다.

    - 원본 `Attention.forward`: `past_key_value`가 있으면 저장된 K·V 뒤에 새 K·V를 이어 붙입니다(`torch.cat(..., dim=1)`).
    - 원본 `MiniMindModel.forward`: 캐시 길이를 `start_pos`로 보고, RoPE 표를 `start_pos`부터 잘라 씁니다.
      그래서 새 토큰이 **올바른 위치 각도**로 회전됩니다.
    - 캐시는 `repeat_kv` **이전**의 K·V, 즉 `[B, T, KV, d]`로 저장됩니다. GQA가 메모리를 줄이는 지점입니다.

    ### 10-1. 캐시를 써도 결과가 같은가?
    """)
    return


@app.cell
def _(attention_mask, input_ids, model, torch):
    # 패딩 없는 문장 0만 사용합니다.
    prompt_ids = input_ids[:1, : int(attention_mask[0].sum())]  # [1, T0]

    with torch.no_grad():
        full_out = model(prompt_ids)                                      # 전체를 한 번에
        prefix_out = model(prompt_ids[:, :-1], use_cache=True)            # 마지막 토큰 전까지 + 캐시
        step_out = model(prompt_ids[:, -1:], past_key_values=prefix_out.past_key_values, use_cache=True)

    cache_error = (step_out.logits[:, -1] - full_out.logits[:, -1]).abs().max().item()
    assert cache_error < 1e-4
    cache_k_shape = tuple(prefix_out.past_key_values[0][0].shape)  # 1층 K: [B, T0-1, KV, d]
    return cache_error, cache_k_shape, prompt_ids


@app.cell(hide_code=True)
def _(cache_error, cache_k_shape, lm_config, mo):
    _c = lm_config
    _per_token_gqa = 2 * _c.num_hidden_layers * _c.num_key_value_heads * _c.head_dim * 2  # K,V × 층 × head × d × bf16 2바이트
    _per_token_mha = 2 * _c.num_hidden_layers * _c.num_attention_heads * _c.head_dim * 2
    _n = 32768
    mo.md(f"""
    전체 계산과 캐시 계산의 마지막 logits 최대 차이: `{cache_error:.2e}` (사실상 같음).
    캐시에 저장된 1층 K의 shape: `{cache_k_shape}` = `[B, 앞 토큰 수, KV, d]`.

    **기본 설정에서 KV cache 크기 (bf16, 토큰 {_n:,}개 기준)**

    | 방식 | 토큰당 | {_n:,} 토큰 |
    | --- | --- | --- |
    | GQA (KV head {_c.num_key_value_heads}개, MiniMind) | {_per_token_gqa / 1024:.0f} KB | {_per_token_gqa * _n / 2**20:.0f} MB |
    | 일반 Multi-Head (KV head {_c.num_attention_heads}개) | {_per_token_mha / 1024:.0f} KB | {_per_token_mha * _n / 2**20:.0f} MB |
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 10-2. 직접 만든 생성 루프 vs 원본 `generate`

    가장 점수가 높은 토큰을 고르는 greedy 생성을 캐시와 함께 직접 구현하고, 원본 `generate`와 비교합니다.
    원본은 `do_sample=False`여도 `top_k`·`top_p` 필터를 먼저 적용하므로 `top_k=0`, `top_p=1.0`으로 끕니다.
    무작위 모델이라 생성된 글은 의미가 없습니다.
    """)
    return


@app.cell
def _(prompt_ids, model, torch):
    max_new_tokens = 8

    with torch.no_grad():
        _ids, _past = prompt_ids, None
        for _ in range(max_new_tokens):
            _past_len = _past[0][0].shape[1] if _past else 0
            _out = model(_ids[:, _past_len:], past_key_values=_past, use_cache=True)  # 새 토큰만 입력
            _next = _out.logits[:, -1].argmax(dim=-1, keepdim=True)                     # greedy
            _ids = torch.cat([_ids, _next], dim=-1)
            _past = _out.past_key_values
        generated_manual = _ids

    generated_ref = model.generate(
        prompt_ids, max_new_tokens=max_new_tokens, do_sample=False,
        top_k=0, top_p=1.0, temperature=1.0, eos_token_id=None,
    )
    assert torch.equal(generated_manual, generated_ref)
    return generated_manual, max_new_tokens


@app.cell(hide_code=True)
def _(generated_manual, max_new_tokens, mo, prompt_ids, tokenizer):
    _new = generated_manual[0, prompt_ids.shape[1]:]
    mo.md(f"""
    프롬프트 `{tokenizer.decode(prompt_ids[0])}` 뒤에 {max_new_tokens}개 토큰 생성:
    `{tokenizer.convert_ids_to_tokens(_new.tolist())}`

    직접 만든 루프와 원본 `generate`의 결과가 같습니다. 원본은 여기에 temperature, top-k, top-p,
    repetition penalty, 종료 토큰 처리 같은 샘플링 옵션을 더한 것입니다.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 11. (선택) `MOEFeedForward`: 전문가 혼합

    `use_moe=True`이면 블록의 FeedForward가 `MOEFeedForward`로 바뀝니다. 나머지 구조는 같습니다.

    - FeedForward를 `num_experts`개(전문가) 만들고, **router**(`gate`, `D → 전문가 수` Linear)가 토큰마다 전문가 점수를 냅니다.
    - softmax 점수 중 상위 `num_experts_per_tok`개 전문가만 계산합니다. 기본값은 1개(top-1)입니다.
    - top-1이면 forward 값은 선택된 전문가의 출력 **그대로**입니다(가중치 1.0).
      다만 router가 학습되도록 straight-through 방식으로 기울기를 흘립니다.
    - 학습 중에는 전문가가 고르게 쓰이도록 보조 손실(`aux_loss`)을 더합니다.

    **효과:** 매개변수는 전문가 수만큼 늘지만, 토큰 하나가 실제로 계산하는 양은 FeedForward 하나 정도입니다.
    """)
    return


@app.cell
def _(F, MiniMindConfig, MiniMindForCausalLM, attn_input, device, lm_config, torch):
    # train_pretrain.py --use_moe 1과 같은 설정 (전문가 4개, top-1은 MiniMindConfig 기본값)
    moe_config = MiniMindConfig(**{**lm_config.to_dict(), "use_moe": True, "num_experts": 4, "num_experts_per_tok": 1})
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        moe_model = MiniMindForCausalLM(moe_config).to(device).eval()
    moe = moe_model.model.layers[0].mlp

    with torch.no_grad():
        _x = attn_input[0]                                   # 문장 0의 토큰 벡터 [T, D]
        router_probs = F.softmax(moe.gate(_x), dim=-1)       # [T, 전문가 수]
        chosen_expert = router_probs.argmax(dim=-1)          # top-1 전문가 번호 [T]
        # 각 토큰을 자기 전문가에게만 보내 계산합니다.
        moe_output_manual = torch.stack([moe.experts[int(e)](_x[t]) for t, e in enumerate(chosen_expert)])
        moe_output = moe(_x[None])[0]

    assert torch.allclose(moe_output, moe_output_manual, atol=1e-5)

    _ffn_one = sum(p.numel() for p in moe.experts[0].parameters())
    moe_param_info = {
        "전문가 1개": _ffn_one,
        "MoE 층 전체 (router 포함)": sum(p.numel() for p in moe.parameters()),
        "토큰당 실제 사용 (router + 전문가 1개)": moe.gate.weight.numel() + _ffn_one,
    }
    return chosen_expert, moe_param_info, router_probs


@app.cell(hide_code=True)
def _(chosen_expert, input_ids, mo, moe_param_info, router_probs, tokenizer):
    mo.vstack([
        mo.ui.table(
            [{"위치": t, "토큰": tokenizer.convert_ids_to_tokens(int(input_ids[0, t])),
              "router 확률": " / ".join(f"{p:.2f}" for p in router_probs[t].tolist()),
              "선택 전문가": int(e)}
             for t, e in enumerate(chosen_expert)],
            selection=None, page_size=20,
        ),
        mo.ui.table([{"항목": k, "매개변수": f"{v:,}"} for k, v in moe_param_info.items()], selection=None),
        mo.md("직접 라우팅한 결과가 원본 `MOEFeedForward`와 같습니다. 토큰마다 다른 전문가를 거칩니다."),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 12. 정리

    | 단계 | 원본 | 입력 → 출력 shape | 한 줄 요약 |
    | --- | --- | --- | --- |
    | 토큰화 | tokenizer | 문장 → `[B, T]` | 글자를 어휘 번호로 |
    | Embedding | `embed_tokens` | `[B, T]` → `[B, T, D]` | 번호로 표의 행 꺼내기 |
    | 정규화 | `RMSNorm` | `[B, T, D]` 그대로 | 벡터 크기를 1 근처로 |
    | 위치 | RoPE | Q, K 그대로 | 위치만큼 회전 → 상대 거리 반영 |
    | 정보 교환 | `Attention` (+GQA) | `[B, T, D]` 그대로 | 앞쪽 토큰들을 가중 평균 |
    | 토큰별 가공 | `FeedForward` (SwiGLU) | `[B, T, D]` 그대로 | 넓혔다가 gate로 거르고 줄이기 |
    | 블록 | `MiniMindBlock` × L | `[B, T, D]` 그대로 | Pre-Norm + 잔차로 조금씩 더하기 |
    | 출력 | `norm` → `lm_head` | `[B, T, D]` → `[B, T, V]` | Embedding 표와 내적해 점수 |
    | 생성 | `generate` + KV cache | 한 토큰씩 | 과거 K·V를 저장해 재사용 |

    ## 13. 직접 해 볼 과제

    각 과제를 하기 전에 **결과를 먼저 예측**해 적고, 코드를 바꿔 실행한 뒤 비교하세요.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({
        "과제 1. GQA를 바꿔 보기 (1절 lm_config)": mo.md("""
    `lm_config`에 `num_key_value_heads=8`(일반 Multi-Head)과 `1`(Multi-Query)을 넣어 보세요.
    1-1절 매개변수 표, 6-2절 head 대응 표, 10-1절 캐시 shape은 어떻게 바뀌나요? `3`으로 하면 어떻게 되나요?

    *해설:* KV head 수가 줄면 `k_proj`·`v_proj`와 KV cache가 그만큼 작아집니다.
    `3`은 Query head 8개를 고르게 나눌 수 없어 `repeat_kv` 이후 shape이 맞지 않으므로 오류가 납니다.
    """),
        "과제 2. causal mask를 빼면? (6-3절)": mo.md("""
    `scores = scores + causal_mask` 줄을 주석 처리해 보세요. 어느 `assert`가 실패하나요?
    그 `assert`도 주석 처리하면 히트맵은 어떻게 바뀌나요?

    *해설:* 원본은 미래를 가리므로 직접 계산과 달라져 원본 비교 assert가 실패합니다.
    assert를 끄면 히트맵 대각선 위에 값이 생깁니다.
    학습 때 미래 토큰을 볼 수 있으면, 다음 토큰을 "예측"하지 않고 그대로 베끼게 됩니다.
    """),
        "과제 3. RoPE 위치를 잘못 주면? (10-1절)": mo.md("""
    캐시 없이 마지막 토큰 하나만 `model(prompt_ids[:, -1:])`로 넣으면 어떤 위치로 회전될까요?
    캐시를 쓴 결과와 logits를 비교해 보세요.

    *해설:* 캐시가 없으면 `start_pos = 0`이라 위치 0의 각도로 회전되고, 앞 토큰도 보지 못해 결과가 달라집니다.
    """),
        "과제 4. 크기를 바꿔 보기 (1절)": mo.md("""
    `hidden_size=512`로 줄이면 `head_dim`, `intermediate_size`, 매개변수 수와 비율이 어떻게 바뀌나요?
    `hidden_size=520`이면 어떤 문제가 생기나요?

    *해설:* `head_dim = 512 / 8 = 64`, `intermediate_size = ceil(512π/64)·64 = 1664`.
    FeedForward·Attention은 `D²`에 비례해 줄지만 Embedding은 `D`에 비례하므로 Embedding 비율이 커집니다.
    `520 / 8 = 65`는 홀수라 RoPE의 절반 나누기(앞 절반·뒤 절반 짝)가 맞지 않아 오류가 납니다.
    """),
        "과제 5. MoE top-2 (11절)": mo.md("""
    `num_experts_per_tok=2`로 바꾸면 11절의 직접 계산과 원본이 왜 달라지나요? 직접 계산을 어떻게 고쳐야 할까요?

    *해설:* top-2에서는 두 전문가의 출력을 router 확률(합이 1이 되도록 정규화)로 가중합합니다.
    `torch.topk(router_probs, 2)`로 두 전문가와 확률을 구하고, `p1·E1(x) + p2·E2(x)`로 바꾸면 같아집니다.
    """),
    })
    return


if __name__ == "__main__":
    app.run()
