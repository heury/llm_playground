# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "marimo>=0.23.3",
# ]
# ///

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import copy
    import math
    import sys
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    import torch
    import torch.nn.functional as F

    # 파일 위치를 기준으로 찾으므로 저장소 루트와 notebooks 양쪽에서 실행 가능.
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root / "projects/minimind"))
    from model.model_minimind import (
        MiniMindConfig, MiniMindForCausalLM,
        apply_rotary_pos_emb, repeat_kv,
    )

    # 작은 CPU 행렬에서는 많은 스레드보다 소수의 스레드가 효율적입니다.
    torch.set_num_threads(2)

    # 그림 축에 한글을 쓰기 위한 폰트. 없으면 None이고, 축에는 위치 번호가 나옵니다.
    korean_font = next(
        (f for f in ["Noto Sans CJK KR", "Noto Sans CJK JP", "Malgun Gothic", "AppleGothic", "NanumGothic"]
         if f in font_manager.get_font_names()),
        None,
    )
    return (
        F,
        MiniMindConfig,
        MiniMindForCausalLM,
        apply_rotary_pos_emb,
        copy,
        korean_font,
        math,
        mo,
        plt,
        repeat_kv,
        torch,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 코드로 이해하는 MiniMind: 모델 구조와 다음 토큰 학습

    **목표:** 문장이 다음 토큰 확률이 되기까지 직접 계산합니다.
    문장 → 숫자 → Attention·MLP → 확률. 그리고 전체 매개변수를 학습합니다.

    **사전 지식:** Python 함수, 텐서 `shape`, 행렬 곱(`@`).

    **기호:** `B` 배치 크기, `T` 토큰 수, `D` 은닉 차원, `V` 어휘 수.

    <img src="public/diagrams/structure.png" alt="MiniMind 전체 구조: 토큰 ID, Embedding, 블록 반복, LM head" width="300">

    `minimind/model/model_minimind.py`의 실제 클래스를 씁니다.

    **실습 방법:** 먼저 예측하고, 실행하고, 결과와 비교해 이유를 적으세요.

    - 수정할 곳: `sentences`, `config`, 화면의 학습 횟수.
    - `@app.cell`은 marimo 셀입니다. 함수 인자와 `return`으로 셀끼리 데이터를 주고받습니다.
    - `_`로 시작하는 변수는 그 셀 안에서만 씁니다.
    - 코드를 바꾸면 관련 셀이 다시 실행됩니다. 학습은 매번 처음부터 시작합니다.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 1. 문장을 숫자로 바꾸기

    실제 LLM은 서브워드 토크나이저를 씁니다.
    여기서는 이해하기 쉽게 띄어쓰기로 나눕니다.

    - `<bos>` 시작, `<eos>` 끝, `<pad>` 빈칸.
    - 짧은 문장은 패딩으로 길이를 맞춥니다.
    - 패딩 위치의 정답은 `-100`입니다. 손실에서 빠집니다.
    """)
    return


@app.cell
def _(torch):
    sentences = [
        "고양이 는 야옹 한다 .",
        "강아지 는 멍멍 한다 .",
        "소 는 음메 한다 .",
        "오리 는 꽥꽥 한다 .",
    ]
    vocabulary = ["<pad>", "<bos>", "<eos>"] + sorted(
        {word for sentence in sentences for word in sentence.split()}
    )
    token_to_id = {word: i for i, word in enumerate(vocabulary)}

    def encode(text, add_eos=True):
        ids = [token_to_id["<bos>"]] + [token_to_id[w] for w in text.split()]
        return ids + [token_to_id["<eos>"]] if add_eos else ids

    def decode(ids):
        return " ".join(vocabulary[int(i)] for i in ids)

    def make_batch(texts):
        sequences = [encode(text) for text in texts]
        length = max(map(len, sequences))
        ids = torch.tensor([s + [0] * (length - len(s)) for s in sequences])
        mask = (ids != token_to_id["<pad>"]).long()
        targets = ids.masked_fill(mask == 0, -100)
        return ids, mask, targets

    input_ids, attention_mask, labels = make_batch(sentences)
    return (
        attention_mask,
        decode,
        encode,
        input_ids,
        labels,
        make_batch,
        token_to_id,
        vocabulary,
    )


@app.cell(hide_code=True)
def _(decode, input_ids, mo, vocabulary):
    # ▸ 결과 표시 — 실행하면 표·그림이 나타납니다 (marimo 표시 코드)
    mo.vstack([
        mo.ui.table([{"ID": i, "토큰": w} for i, w in enumerate(vocabulary)], selection=None),
        mo.md(f"첫 문장: `{decode(input_ids[0])}`\n\n입력 shape: `{tuple(input_ids.shape)}`"),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 2. 작은 모델 만들기

    은닉 차원 32, 층 2개, Query head 4개, Key/Value head 2개.

    - **GQA:** 여러 Query head가 Key/Value를 나눠 씁니다. 3-2절에서 봅니다.
    - head 차원은 `32 / 4 = 8`입니다. 짝수여야 합니다. 3-2절의 RoPE가 차원을 둘씩 짝지어 회전하기 때문입니다.
    - **Embedding과 LM head는 같은 가중치 표 `[V, D]`를 씁니다.**
      - 입력: 토큰 ID에 해당하는 행 하나를 꺼냅니다.
      - 출력: 마지막 벡터를 모든 행과 내적해 어휘별 점수를 만듭니다.
      - 같은 표를 쓰므로 매개변수가 줄어듭니다.
    """)
    return


@app.cell
def _(MiniMindConfig, MiniMindForCausalLM, torch, vocabulary):
    config = MiniMindConfig(
        hidden_size=32, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2,
        intermediate_size=64, max_position_embeddings=64,
        vocab_size=len(vocabulary), dropout=0.0,
        flash_attn=False, use_moe=False,
        bos_token_id=1, eos_token_id=2,
    )
    assert config.hidden_size % config.num_attention_heads == 0
    assert config.num_attention_heads % config.num_key_value_heads == 0
    assert config.head_dim % 2 == 0
    # fork_rng: 시드를 이 블록 안에서만 고정합니다. 다른 셀의 난수에는 영향이 없습니다.
    # 그래서 몇 번을 다시 실행해도 같은 초기 가중치가 나옵니다.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        initial_model = MiniMindForCausalLM(config).cpu().eval()
    assert initial_model.lm_head.weight is initial_model.model.embed_tokens.weight
    return (initial_model,)


@app.cell(hide_code=True)
def _(initial_model, mo):
    # ▸ 결과 표시 — 실행하면 표·그림이 나타납니다 (marimo 표시 코드)
    mo.vstack([
        mo.md(f"고유 매개변수: **{sum(p.numel() for p in initial_model.parameters()):,}개**"),
        mo.ui.table([
            {"이름": name, "shape": str(tuple(p.shape)), "개수": p.numel()}
            for name, p in initial_model.named_parameters()
        ], selection=None, page_size=12),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. 먼저 한 head로 Attention 계산하기

    **예측:** 두 번째 위치는 세 번째 위치를 볼 수 있을까요?
    `고양이 / 는 / 야옹` 세 위치에 Q/K/V를 직접 정했습니다.
    의미는 없습니다. 계산 규칙만 봅니다.

    1. Q·K 내적으로 관련 점수를 만듭니다.
       - `√d`로 나눕니다(`d`는 벡터 차원, 여기서는 2).
       - 차원이 크면 내적도 커집니다. 그러면 softmax가 한 위치에만 몰립니다. 이것을 막습니다.
    2. 미래 위치를 가리고 softmax를 적용합니다. 각 행의 합은 1입니다.
       - `triu(1)`: 대각선 **위쪽**만 `True`인 행렬입니다. 즉 "나보다 뒤 위치"입니다.
       - 가린 자리는 `-inf`로 채웁니다. softmax에서 `exp(-inf) = 0`이므로 확률이 0이 됩니다.
    3. 그 비율로 V를 가중합합니다.

    ```text
    triu(1) 마스크      key: 고양이  는     야옹
    query 고양이            F       T      T
    query 는                F       F      T
    query 야옹              F       F      F
    ```

    예: 두 번째 행은 `는`이 각 위치를 얼마나 보는지입니다.
    """)
    return


@app.cell
def _(F, math, torch):
    toy_q = torch.tensor([[1., 0.], [1., 1.], [0., 1.]])
    toy_k = torch.tensor([[1., 0.], [0., 1.], [1., 1.]])
    toy_v = torch.tensor([[10., 0.], [0., 10.], [5., 5.]])
    toy_scores = toy_q @ toy_k.T / math.sqrt(2)
    toy_mask = torch.ones(3, 3, dtype=torch.bool).triu(1)
    toy_probabilities = F.softmax(toy_scores.masked_fill(toy_mask, -torch.inf), dim=-1)
    toy_context = toy_probabilities @ toy_v
    torch.testing.assert_close(toy_probabilities.sum(-1), torch.ones(3))
    assert toy_probabilities[toy_mask].eq(0).all()
    torch.testing.assert_close(toy_context[1], (toy_probabilities[1, :, None] * toy_v).sum(0))
    toy_words = ["고양이", "는", "야옹"]
    return toy_context, toy_mask, toy_probabilities, toy_scores, toy_words


@app.cell(hide_code=True)
def _(mo, toy_context, toy_mask, toy_probabilities, toy_scores, toy_words):
    # ▸ 결과 표시 — 실행하면 표·그림이 나타납니다 (marimo 표시 코드)
    mo.vstack([
        mo.md("`Q/K/V: [3,2] → 점수·확률: [3,3] → 출력: [3,2]`"),
        mo.ui.table([
            {"query": toy_words[i], "key": toy_words[j], "내적 점수": float(toy_scores[i, j]),
             "미래인가": bool(toy_mask[i, j]), "확률": float(toy_probabilities[i, j])}
            for i in range(3) for j in range(3)
        ], selection=None),
        mo.md(f"두 번째 위치의 출력 = Σ(V × 확률) = `{toy_context[1].tolist()}`. 미래인 `야옹`의 비율은 0입니다."),
        mo.md("**작은 과제:** `toy_v`의 마지막 행을 바꾸면 앞 두 위치의 출력도 바뀔까요? **해설:** 안 바뀝니다. 앞 두 행에서 미래 위치의 확률은 0입니다."),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 3-1. 실제 MiniMind: Q/K/V와 여러 head

    **RMSNorm**은 토큰 벡터의 크기를 맞춥니다:
    $\mathrm{RMSNorm}(x)=w\,x/\sqrt{\mathrm{mean}(x^2)+\epsilon}$.

    정규화한 입력으로 Q, K, V를 만듭니다.

    - Q: 무엇을 찾는가
    - K: 나는 어떤 특징인가
    - V: 전달할 정보

    이 구현은 Q와 K에 RMSNorm을 head별로 한 번 더 씁니다(QK-norm).
    Q·K 값의 크기가 커지면 점수가 폭주합니다. 이를 막아 학습을 안정시킵니다.

    **shape:** 입력 `[B,T,32]` → projection `[B,T,32]` → `view`로 head 축 분리 → Q `[B,T,4,8]`, K/V `[B,T,2,8]`.
    `view`는 값을 바꾸지 않습니다. 마지막 축 32를 `4 × 8`로 나눠 볼 뿐입니다.
    """)
    return


@app.cell
def _(initial_model, input_ids, torch):
    with torch.no_grad():
        embeddings = initial_model.model.embed_tokens(input_ids)
        first_block = initial_model.model.layers[0]
        attn = first_block.self_attn
        normalized = first_block.input_layernorm(embeddings)
        _b, _t, _d = normalized.shape
        # 1) projection: [B,T,D] → [B,T,head 수 × head 차원]
        _q_flat = attn.q_proj(normalized)
        _k_flat = attn.k_proj(normalized)
        _v_flat = attn.v_proj(normalized)
        # 2) view: 마지막 축을 [head 수, head 차원]으로 나눔
        _q_heads = _q_flat.view(_b, _t, attn.n_local_heads, attn.head_dim)
        _k_heads = _k_flat.view(_b, _t, attn.n_local_kv_heads, attn.head_dim)
        v = _v_flat.view(_b, _t, attn.n_local_kv_heads, attn.head_dim)
        # 3) QK-norm: Q와 K만 head별로 정규화 (V는 하지 않음)
        q = attn.q_norm(_q_heads)
        k = attn.k_norm(_k_heads)
    return attn, embeddings, first_block, k, normalized, q, v


@app.cell(hide_code=True)
def _(k, mo, normalized, q):
    # ▸ 결과 설명 — 실행하면 계산 값이 들어간 설명이 나타납니다 (marimo 표시 코드)
    mo.md(f"정규화 입력 `{tuple(normalized.shape)}` → Q `{tuple(q.shape)}`, K/V `{tuple(k.shape)}`. **확인:** head 수 × head 차원 = projection 출력 차원.")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### 3-2. 위치 정보(RoPE)와 K/V 공유(GQA)

    **RoPE:** Q/K 벡터의 차원을 둘씩 짝지어, 위치에 비례한 각도만큼 회전시킵니다.
    짝은 **앞 절반 i번과 뒤 절반 i+d/2번**입니다. 각도 계산은 바로 다음 절(3-2-1)에서 봅니다.

    - 같은 벡터도 위치가 다르면 회전 결과가 다릅니다.
    - 벡터 길이는 그대로입니다.
    - **핵심:** 위치 m의 Q와 위치 n의 K를 내적하면, 점수는 **거리 `m − n`**에만 달라집니다.
      두 회전이 내적에서 빼기로 합쳐지기 때문입니다. 그래서 점수에 상대 위치가 들어갑니다.

    아래에서 세 가지를 모두 확인합니다.

    **`repeat_kv`:** K/V head를 반복해서 Q head 수에 맞춥니다(GQA).
    Q head 4개, K/V head 2개이므로 `attn.n_rep = 4 // 2 = 2`입니다.

    ```text
    Q head:       Q0   Q1   Q2   Q3
    대응 K head:  K0   K0   K1   K1
    대응 V head:  V0   V0   V1   V1
    ```

    - K/V shape: `[B, T, 2, 8]` → `[B, T, 4, 8]`.
    - 결과는 `k.repeat_interleave(n_rep, dim=2)`와 같습니다. 아래에서 확인합니다.
    - 새 K/V를 만드는 게 아닙니다. 기존 값을 나눠 씁니다.
    - K/V가 같아도 Q가 다르므로 head마다 Attention 결과는 다릅니다.
    - 캐시에는 펼치기 전 K/V를 저장합니다. 그래서 메모리가 줄어듭니다.
    - K/V head 수가 Q head 수와 같으면 `n_rep = 1`이고, 공유하지 않습니다.

    **`transpose(1, 2)`:** `[B, T, H, d]` → `[B, H, T, d]`.
    head마다 `[T, d] @ [d, T]` 행렬 곱을 하려면 head 축이 앞에 와야 합니다.
    """)
    return


@app.cell
def _(apply_rotary_pos_emb, attn, initial_model, k, q, repeat_kv, torch, v):
    with torch.no_grad():
        _t = q.shape[1]
        position_embeddings = (
            initial_model.model.freqs_cos[:_t], initial_model.model.freqs_sin[:_t],
        )
        q_rotated, k_rotated = apply_rotary_pos_emb(q, k, *position_embeddings)
        # 회전해도 벡터 길이는 같습니다.
        torch.testing.assert_close(q.norm(dim=-1), q_rotated.norm(dim=-1))

        def rotate_at(vector, position):
            # vector [1,1,H,d]를 지정한 위치의 각도로 회전합니다.
            # apply_rotary_pos_emb는 Q와 K를 함께 받으므로 같은 벡터를 두 번 넘기고 하나만 씁니다.
            _cos = initial_model.model.freqs_cos[position:position + 1]
            _sin = initial_model.model.freqs_sin[position:position + 1]
            rotated, _ = apply_rotary_pos_emb(vector, vector, _cos, _sin)
            return rotated

        # 같은 Q 벡터(1번 위치 토큰)를 위치 1과 위치 2에 놓아 회전 효과만 비교합니다.
        same_q_at_2 = rotate_at(q[:, 1:2], 2)

        # 상대 위치: 같은 Q·K 벡터 쌍을 여러 위치에 놓고 점수를 잽니다.
        _q_vec, _k_vec = q[:1, 1:2, :1], k[:1, 2:3, :1]
        rope_rows = []
        for _qp, _kp in [(2, 0), (5, 3), (9, 7), (4, 0), (8, 4)]:
            _score = (rotate_at(_q_vec, _qp) * rotate_at(_k_vec, _kp)).sum().item()
            rope_rows.append({"Q 위치": _qp, "K 위치": _kp, "거리": _qp - _kp, "Q·K 점수": _score})
        _s = [_row["Q·K 점수"] for _row in rope_rows]
        # float32 계산이라 아주 작은 오차는 허용합니다.
        assert abs(_s[0] - _s[1]) < 1e-4 and abs(_s[0] - _s[2]) < 1e-4  # 거리 2끼리 같음
        assert abs(_s[3] - _s[4]) < 1e-4                                # 거리 4끼리 같음

        # GQA: K/V head를 Q head 수만큼 반복한 뒤, head 축을 앞으로 옮깁니다.
        torch.testing.assert_close(repeat_kv(k_rotated, attn.n_rep), k_rotated.repeat_interleave(attn.n_rep, dim=2))
        query_heads = q_rotated.transpose(1, 2)
        key_heads = repeat_kv(k_rotated, attn.n_rep).transpose(1, 2)
        value_heads = repeat_kv(v, attn.n_rep).transpose(1, 2)
        if attn.n_rep > 1:
            # 공유할 때만: Q head 0과 1은 같은 원본 K head를 봅니다.
            torch.testing.assert_close(key_heads[:, 0], key_heads[:, 1])
    return (
        key_heads,
        position_embeddings,
        q_rotated,
        query_heads,
        rope_rows,
        same_q_at_2,
        value_heads,
    )


@app.cell(hide_code=True)
def _(attn, key_heads, mo, q_rotated, query_heads, rope_rows, same_q_at_2):
    # ▸ 결과 표시 — 실행하면 표·그림이 나타납니다 (marimo 표시 코드)
    _gqa_note = (
        f"`n_rep = {attn.n_rep}`: 펼친 K의 head 0과 1은 같은 원본 head입니다."
        if attn.n_rep > 1
        else "`n_rep = 1`: K/V head 수가 Q head 수와 같아 공유하지 않습니다."
    )
    mo.vstack([
        mo.md("**① 같은 벡터, 다른 위치:** 좌표는 다르고 길이는 같습니다."),
        mo.ui.table([
            {"차원": i, "같은 Q를 위치 1에서 회전": float(q_rotated[0, 1, 0, i]), "같은 Q를 위치 2에서 회전": float(same_q_at_2[0, 0, 0, i])}
            for i in range(4)
        ], selection=None),
        mo.md("**② 상대 위치:** 거리가 같으면 절대 위치가 달라도 점수가 같습니다."),
        mo.ui.table(rope_rows, selection=None),
        mo.md(f"**③ GQA:** 회전·공유 후 shape: Q `{tuple(query_heads.shape)}`, K/V `{tuple(key_heads.shape)}` (`[B, H, T, d]`). {_gqa_note}"),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 3-2-1. RoPE 각도는 어떻게 정해지나 (`precompute_freqs_cis`)

    지금까지 쓴 `freqs_cos`/`freqs_sin`은 모델이 미리 계산해 둔 표입니다. 이 표를 직접 만들어 봅니다.

    **① 어떤 차원끼리 짝을 짓나:** `rotate_half`는 벡터를 앞 절반 `a`와 뒤 절반 `b`로 나눕니다.
    head 차원 `d = 8`이면 짝은 `(0,4) (1,5) (2,6) (3,7)`입니다. **인접한 (0,1)이 아닙니다.**

    **② 짝마다 회전 속도(주파수)가 다릅니다:** $f_i = 1 / \theta^{2i/d}$, $\theta$ = `rope_theta` = $10^6$.

    - 짝 0은 $f_0 = 1$입니다. 한 칸마다 1 라디안씩, 빠르게 돕니다.
    - 뒤 짝일수록 아주 천천히 돕니다.
    - 빠른 짝은 가까운 거리를, 느린 짝은 먼 거리를 구분합니다. 시계의 초침·분침·시침과 비슷합니다.
    - $\theta$를 크게 잡으면 느린 짝이 더 느려집니다. 긴 문맥에서도 각도가 한 바퀴를 넘지 않습니다.

    **③ 위치 m의 각도:** $m \times f_i$. `freqs_cos`는 `cos(각도)` `[T, d/2]`를 두 번 이어 붙인 `[T, d]`입니다.

    **④ 회전:** 짝 `(a, b)`를 2차원 평면의 점으로 보고 돌립니다.

    ```text
    a' = a·cos − b·sin
    b' = b·cos + a·sin
    ```

    원본은 이것을 `q * cos + rotate_half(q) * sin` 한 줄로 씁니다. `rotate_half(q) = [−b, a]`이기 때문입니다.
    """)
    return


@app.cell
def _(apply_rotary_pos_emb, initial_model, q, torch):
    with torch.no_grad():
        _d = initial_model.config.head_dim
        _theta = initial_model.config.rope_theta
        # ② 짝 i의 주파수: 1 / θ^(2i/d), i = 0 .. d/2-1
        rope_freqs = 1.0 / (_theta ** (torch.arange(0, _d, 2).float() / _d))       # [d/2]
        # ③ 위치 m의 각도 = m × 주파수
        _positions = torch.arange(initial_model.config.max_position_embeddings).float()
        rope_angles = torch.outer(_positions, rope_freqs)                          # [T, d/2]
        _manual_cos = torch.cat([rope_angles.cos(), rope_angles.cos()], dim=-1)    # [T, d]
        _manual_sin = torch.cat([rope_angles.sin(), rope_angles.sin()], dim=-1)
        torch.testing.assert_close(_manual_cos, initial_model.model.freqs_cos)
        torch.testing.assert_close(_manual_sin, initial_model.model.freqs_sin)

        # ④ 짝 (a, b)를 2차원 회전한 결과 = 원본 apply_rotary_pos_emb
        _pos = 3
        _vec = q[0, 1, 0]                                   # head 차원 벡터 하나 [d]
        _a, _b = _vec[: _d // 2], _vec[_d // 2:]            # 앞 절반·뒤 절반
        _cos, _sin = rope_angles[_pos].cos(), rope_angles[_pos].sin()
        _manual_rotated = torch.cat([_a * _cos - _b * _sin, _b * _cos + _a * _sin])
        _original, _ = apply_rotary_pos_emb(
            _vec[None, None, None], _vec[None, None, None],
            initial_model.model.freqs_cos[_pos:_pos + 1], initial_model.model.freqs_sin[_pos:_pos + 1],
        )
        torch.testing.assert_close(_manual_rotated, _original.flatten())
    return rope_angles, rope_freqs


@app.cell(hide_code=True)
def _(math, mo, plt, rope_angles, rope_freqs):
    # ▸ 결과 표시 — 실행하면 표·그림이 나타납니다 (marimo 표시 코드)
    _half = len(rope_freqs)
    rope_fig, _ax = plt.subplots(figsize=(7, 3))
    for _i in range(_half):
        _ax.plot(rope_angles[:32, _i].cos().numpy(), label=f"pair ({_i}, {_i + _half})")
    _ax.set(xlabel="Position m", ylabel="cos(m × f_i)", title="RoPE: each pair rotates at its own speed")
    _ax.legend(fontsize=8, loc="lower left")
    rope_fig.tight_layout()
    mo.vstack([
        mo.md("직접 만든 `cos`/`sin` 표와 2차원 회전 결과가 원본과 같습니다."),
        mo.ui.table([
            {"짝": f"({_i}, {_i + _half})", "주파수 f_i": float(rope_freqs[_i]),
             "위치 1의 각도(라디안)": float(rope_angles[1, _i]),
             "한 바퀴(2π)에 필요한 위치 수": round(2 * math.pi / float(rope_freqs[_i]), 1)}
            for _i in range(_half)
        ], selection=None),
        rope_fig,
        mo.md("**관찰:** 짝 0은 약 6칸마다 한 바퀴 돕니다. 마지막 짝은 수십만 칸이 지나야 한 바퀴를 돕니다(표의 마지막 행). 그래프에서 거의 평평한 선입니다."),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### 3-3. 점수 → 미래 가리기 → 확률 → 가중합

    3절의 작은 예제와 같은 규칙입니다. 이번에는 모든 배치와 head에 적용합니다.

    - head별 출력을 이어 붙이고 `o_proj`로 섞습니다.
    - `causal mask`는 미래를 가립니다.
    - `attention_mask`는 패딩을 가립니다. 지금 배치에는 패딩이 없습니다. 3-4절에서 봅니다.
    - `use_causal_mask`는 7절 과제 4용 스위치입니다. 지금은 `True`로 두세요.

    학습 후에 같은 위치의 Attention을 다시 비교합니다.
    """)
    return


@app.cell
def _(
    F,
    attn,
    key_heads,
    math,
    normalized,
    position_embeddings,
    query_heads,
    torch,
    value_heads,
):
    # 과제 4: False로 바꾸면 미래를 가리지 않습니다. 아래 검증도 건너뜁니다.
    use_causal_mask = True

    with torch.no_grad():
        _b, _t, _d = normalized.shape
        # [B,H,T,d] @ [B,H,d,T] → [B,H,T,T]: head마다 모든 query×key 쌍의 점수
        _scores = query_heads @ key_heads.transpose(-2, -1) / math.sqrt(attn.head_dim)
        causal_mask = torch.ones(_t, _t, dtype=torch.bool).triu(1)
        if use_causal_mask:
            _scores = _scores.masked_fill(causal_mask, -torch.inf)
        attention_probabilities = F.softmax(_scores, dim=-1)
        # [B,H,T,T] @ [B,H,T,d] → [B,H,T,d] → [B,T,H,d] → [B,T,H×d]: head 출력 이어 붙이기
        _context = (attention_probabilities @ value_heads).transpose(1, 2).reshape(_b, _t, -1)
        manual_attention = attn.o_proj(_context)
        if use_causal_mask:
            _actual, _ = attn(normalized, position_embeddings)
            torch.testing.assert_close(manual_attention, _actual)
            assert attention_probabilities.masked_select(causal_mask[None, None]).eq(0).all()
    future_attention = attention_probabilities.masked_select(causal_mask[None, None]).sum().item()
    return attention_probabilities, future_attention, manual_attention, use_causal_mask


@app.cell(hide_code=True)
def _(
    attention_probabilities,
    future_attention,
    manual_attention,
    mo,
    normalized,
    use_causal_mask,
):
    # ▸ 결과 표시 — 실행하면 표·그림이 나타납니다 (marimo 표시 코드)
    _note = (
        "직접 계산한 Attention과 원본 출력이 같습니다. 각 행의 미래 위치 확률은 0입니다."
        if use_causal_mask
        else f"**마스크 꺼짐:** 미래 위치로 간 확률의 합 = `{future_attention:.3f}`. 각 위치가 뒤 토큰, 즉 맞혀야 할 정답을 봅니다."
    )
    mo.vstack([
        mo.ui.table([
            {"단계": "Embedding / RMSNorm", "shape": str(tuple(normalized.shape)), "축 의미": "[B, T, D]"},
            {"단계": "Attention 확률", "shape": str(tuple(attention_probabilities.shape)), "축 의미": "[B, H, query T, key T]"},
            {"단계": "출력", "shape": str(tuple(manual_attention.shape)), "축 의미": "[B, T, D]"},
        ], selection=None),
        mo.md(_note),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 3-4. 패딩 가리기: `attention_mask`

    길이가 다른 문장을 한 배치에 넣으면 짧은 문장 끝에 `<pad>`가 붙습니다.
    `attention_mask`는 `[B, T]`이고 실제 토큰은 1, 패딩은 0입니다.

    원본 Attention(130행)은 점수에 이것을 더합니다:

    ```python
    scores += (1.0 - attention_mask.unsqueeze(1).unsqueeze(2)) * -1e9   # [B,T] → [B,1,1,T]
    ```

    - 마지막 축이 **key 축**입니다. 그래서 패딩인 **key 열**을 가립니다. 어떤 query도 패딩을 보지 않습니다.
    - `-1e9`는 사실상 `-inf`입니다. softmax 뒤 확률이 0이 됩니다.
    - 패딩인 **query 행**은 가리지 않습니다. 계산은 되지만, 그 위치의 정답이 `-100`이라 손실에서 빠집니다(4절).
    - 패딩이 있으면 원본은 빠른 flash 경로 대신 이 수동 경로를 씁니다.

    아래 `attention_forward`는 3-1 ~ 3-4절의 계산을 한 함수로 묶은 것입니다. 5-3절에서도 씁니다.
    """)
    return


@app.cell
def _(F, apply_rotary_pos_emb, math, repeat_kv, torch):
    def attention_forward(model, ids, attention_mask=None, layer=0, causal=True):
        """3-1 ~ 3-4절의 Attention 계산. (확률 [B,H,T,T], 직접 계산 출력, 원본 출력)을 돌려줍니다."""
        block = model.model.layers[layer]
        a = block.self_attn
        x = block.input_layernorm(model.model.embed_tokens(ids))            # [B, T, D]
        b, t, _ = x.shape
        cos, sin = model.model.freqs_cos[:t], model.model.freqs_sin[:t]
        q = a.q_norm(a.q_proj(x).view(b, t, a.n_local_heads, a.head_dim))   # 3-1
        k = a.k_norm(a.k_proj(x).view(b, t, a.n_local_kv_heads, a.head_dim))
        v = a.v_proj(x).view(b, t, a.n_local_kv_heads, a.head_dim)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)                          # 3-2 RoPE
        q = q.transpose(1, 2)                                                # [B, H, T, d]
        k = repeat_kv(k, a.n_rep).transpose(1, 2)                            # 3-2 GQA
        v = repeat_kv(v, a.n_rep).transpose(1, 2)
        scores = q @ k.transpose(-2, -1) / math.sqrt(a.head_dim)             # 3-3
        if causal:
            scores = scores.masked_fill(torch.ones(t, t, dtype=torch.bool).triu(1), -torch.inf)
        if attention_mask is not None:                                       # 3-4
            scores = scores + (1.0 - attention_mask[:, None, None, :]) * -1e9
        probs = F.softmax(scores, dim=-1)
        output = a.o_proj((probs @ v).transpose(1, 2).reshape(b, t, -1))
        reference, _ = a(x, (cos, sin), attention_mask=attention_mask)
        return probs, output, reference
    return (attention_forward,)


@app.cell
def _(attention_forward, initial_model, make_batch, torch):
    with torch.no_grad():
        # 두 번째 문장은 짧아서 뒤에 <pad>가 붙습니다.
        mask_demo_ids, mask_demo_mask, _ = make_batch(["고양이 는 야옹 한다 .", "강아지 는"])
        mask_demo_probs, _output, _reference = attention_forward(
            initial_model, mask_demo_ids, attention_mask=mask_demo_mask,
        )
        torch.testing.assert_close(_output, _reference)
        # 패딩인 key 열(마지막 축)로 간 확률은 모든 query·head에서 0입니다.
        _pad_keys = mask_demo_mask[:, None, None, :] == 0                    # [B,1,1,T]
        assert mask_demo_probs.masked_select(_pad_keys.expand_as(mask_demo_probs)).max() < 1e-6
    return mask_demo_ids, mask_demo_mask, mask_demo_probs


@app.cell(hide_code=True)
def _(mask_demo_ids, mask_demo_mask, mask_demo_probs, mo, vocabulary):
    # ▸ 결과 표시 — 실행하면 표·그림이 나타납니다 (marimo 표시 코드)
    _row = 1   # 패딩이 있는 두 번째 문장
    _words = [vocabulary[int(i)] for i in mask_demo_ids[_row]]
    _is_pad = mask_demo_mask[_row] == 0
    mo.vstack([
        mo.md(f"두 번째 문장: `{' '.join(_words)}` · mask: `{mask_demo_mask[_row].tolist()}`"),
        mo.ui.table([
            {"query 위치": _i, "query": _words[_i], "query가 패딩인가": bool(_is_pad[_i]),
             "실제 토큰으로 간 확률": float(mask_demo_probs[_row, 0, _i][~_is_pad].sum()),
             "패딩 key로 간 확률": float(mask_demo_probs[_row, 0, _i][_is_pad].sum())}
            for _i in range(len(_words))
        ], selection=None),
        mo.md("**관찰:** 모든 행에서 패딩 key로 간 확률은 0입니다. 패딩인 query 행도 계산은 됩니다. 이 행의 결과는 손실의 `-100`이 버립니다. 직접 계산한 결과는 원본 출력과 같습니다(head 0 표시)."),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. MLP와 잔차 연결, 다음 토큰 정답

    - **Attention:** 토큰 **사이**의 정보를 섞습니다.
    - **MLP:** 토큰 **안**의 특징을 바꿉니다. MiniMind는 SwiGLU를 씁니다:
      `down_proj(silu(gate_proj(x)) * up_proj(x))`.
    - **잔차 연결:** 결과에 입력을 더합니다. 정보와 기울기가 잘 흐릅니다.

    **블록 하나의 계산(Pre-Norm):** 정규화는 각 연산의 **앞**에 둡니다.

    ```text
    h   = x + Attention(input_layernorm(x))
    out = h + MLP(post_attention_layernorm(h))
    ```

    `post_attention_layernorm`은 이름과 달리 **MLP 앞**의 정규화입니다. "Attention 다음에 오는 정규화"라는 뜻입니다.

    **다음 토큰 정답:** 출력은 `[B,T,V]`입니다. 위치 `t`의 출력이 `t+1` 토큰을 예측합니다.
    한 칸 밀기(shift)는 원본 `forward(labels=...)`가 안에서 합니다. **labels를 미리 옮기지 마세요.**
    아래 코드에서는 이 shift를 똑같이 따라 해 원본 손실과 비교합니다.

    **손실:** 출력은 확률 전 점수(logits)입니다.
    softmax로 정답 확률 p를 구하면 손실은 `-log(p)`입니다.
    cross entropy가 이를 안정적으로 계산합니다.
    학습은 정답 확률을 높이는 쪽으로 가중치를 바꿉니다.
    """)
    return


@app.cell
def _(
    F,
    embeddings,
    first_block,
    manual_attention,
    position_embeddings,
    torch,
    use_causal_mask,
):
    # ① 블록 하나를 직접 계산해 원본 블록과 비교합니다.
    with torch.no_grad():
        _h = embeddings + manual_attention                        # h = x + Attention(norm(x))
        _mlp_input = first_block.post_attention_layernorm(_h)     # MLP 앞 정규화
        _ff = first_block.mlp
        _gate = F.silu(_ff.gate_proj(_mlp_input))                 # 문을 얼마나 열지
        _up = _ff.up_proj(_mlp_input)                             # 전달할 값
        _manual_block = _h + _ff.down_proj(_gate * _up)           # out = h + MLP(norm(h))
        _actual_block, _ = first_block(embeddings, position_embeddings)
        if use_causal_mask:
            torch.testing.assert_close(_manual_block, _actual_block)
    return


@app.cell
def _(F, attention_mask, initial_model, input_ids, labels, torch, vocabulary):
    # ② 원본 손실 = 직접 shift한 cross entropy
    with torch.no_grad():
        initial_out = initial_model(input_ids, attention_mask=attention_mask, labels=labels)
        # forward 안의 shift를 똑같이 따라 합니다: 위치 t의 logits ↔ 위치 t+1의 정답
        _logits_t = initial_out.logits[:, :-1]     # 마지막 위치는 맞힐 다음 토큰이 없음
        _labels_next = labels[:, 1:]               # 첫 토큰 <bos>는 맞힐 대상이 아님
        _manual_loss = F.cross_entropy(
            _logits_t.reshape(-1, len(vocabulary)), _labels_next.reshape(-1),
            ignore_index=-100,
        )
        torch.testing.assert_close(_manual_loss, initial_out.loss)
    return (initial_out,)


@app.cell
def _(F, initial_model, make_batch, torch):
    # ③ 길이가 다른 문장: 패딩 위치(-100)는 손실에서 빠집니다.
    with torch.no_grad():
        _pad_ids, pad_mask, pad_labels = make_batch([
            "고양이 는 야옹 한다 .", "강아지 는",
        ])
        assert (pad_labels == -100).any()
        _pad_out = initial_model(_pad_ids, attention_mask=pad_mask, labels=pad_labels)
        # -100이 아닌 정답 위치만 골라 직접 평균을 냅니다.
        _logits_t = _pad_out.logits[:, :-1]
        _labels_next = pad_labels[:, 1:]
        _valid = _labels_next != -100              # [B, T-1] True/False
        _pad_loss = F.cross_entropy(_logits_t[_valid], _labels_next[_valid])
        torch.testing.assert_close(_pad_loss, _pad_out.loss)
    return pad_labels, pad_mask


@app.cell(hide_code=True)
def _(initial_out, input_ids, labels, mo, pad_labels, pad_mask, vocabulary):
    # ▸ 결과 표시 — 실행하면 표·그림이 나타납니다 (marimo 표시 코드)
    mo.vstack([
        mo.md(f"블록·손실 검증 통과. Logits shape: `{tuple(initial_out.logits.shape)}` (`[B, T, V]`), 초기 손실: `{initial_out.loss.item():.4f}`"),
        mo.ui.table([
            {"입력 위치의 토큰": vocabulary[int(x)], "맞혀야 할 다음 토큰": vocabulary[int(y)]}
            for x, y in zip(input_ids[0, :-1], labels[0, 1:])
        ], selection=None),
        mo.md(f"길이가 다른 문장의 패딩 검증도 통과. 두 번째 문장의 mask: `{pad_mask[1].tolist()}`, labels: `{pad_labels[1].tolist()}`"),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 5. 작은 데이터로 전체 매개변수 학습하기

    **학습의 한 걸음:**

    1. `zero_grad`: 이전 기울기를 비웁니다.
    2. `forward`: 예측과 손실을 계산합니다.
    3. `loss.backward()`: 기울기를 계산합니다.
    4. `optimizer.step()`: 가중치를 갱신합니다.

    **헷갈리기 쉬운 것:**

    - `train()`/`eval()`: dropout 같은 동작 모드입니다. gradient와는 무관합니다.
    - `no_grad()`: gradient 기록을 끕니다. 결과를 볼 때 씁니다.

    - `copy.deepcopy`: 학습 전 모델을 그대로 남겨 두고, 복사본만 학습합니다. 뒤에서 전후를 비교합니다.
    - `AdamW`: 기울기 방향으로 가중치를 옮기는 최적화 도구입니다. `lr`은 한 번에 옮기는 크기입니다.

    네 문장을 반복해 암기시킵니다. 학습 횟수를 바꿔 가며 손실과 생성 문장을 비교하세요.

    **손실이 0이 될 수 없는 이유:** `<bos>` 다음에는 동물 네 개가 모두 정답입니다.

    - 문장마다 맞힐 위치는 6개입니다. 그중 `<bos>` 다음 1개는 넷 중 하나를 맞혀야 합니다.
    - 그 위치에서 가장 좋은 답은 넷에 1/4씩 주는 것입니다. 손실은 `ln 4 ≈ 1.386`입니다.
    - 나머지 5개는 완벽히 맞힐 수 있습니다(손실 0).
    - 따라서 평균 손실의 하한은 `ln 4 / 6 ≈ 0.231`입니다. 결과 화면에 함께 표시합니다.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    # ▸ 실습 설정 위젯 — 실행하면 슬라이더·선택 상자가 나타납니다 (marimo 표시 코드)
    train_steps = mo.ui.slider(20, 160, step=20, value=100, label="전체 학습 step")
    train_steps
    return (train_steps,)


@app.cell
def _(
    attention_mask,
    copy,
    initial_model,
    input_ids,
    labels,
    math,
    torch,
    train_steps,
):
    trained_model = copy.deepcopy(initial_model).train()
    _optimizer = torch.optim.AdamW(trained_model.parameters(), lr=0.005, weight_decay=0.0)
    pretrain_losses = []
    for _step in range(train_steps.value):
        _optimizer.zero_grad(set_to_none=True)
        _loss = trained_model(input_ids, attention_mask=attention_mask, labels=labels).loss
        assert torch.isfinite(_loss)
        _loss.backward()
        _optimizer.step()
        pretrain_losses.append(_loss.item())
    trained_model.eval()
    with torch.no_grad():
        final_pretrain_loss = trained_model(input_ids, attention_mask=attention_mask, labels=labels).loss.item()
    assert final_pretrain_loss < pretrain_losses[0]

    # 손실 하한: 첫 단어(동물)가 모두 다르면, <bos> 다음 위치마다 최소 ln(동물 수)
    _n_animals = len(set(input_ids[:, 1].tolist()))
    _n_targets = int((labels[:, 1:] != -100).sum())
    loss_floor = input_ids.shape[0] * math.log(_n_animals) / _n_targets
    return final_pretrain_loss, loss_floor, pretrain_losses, trained_model


@app.cell
def _(decode, encode, token_to_id, torch):
    def complete(model, prompt):
        # greedy 생성: 매번 가장 점수가 큰 토큰을 고릅니다. <eos>가 나오면 멈춥니다.
        model.eval()
        ids = torch.tensor([encode(prompt, add_eos=False)])
        with torch.no_grad():
            # MiniMind의 generate는 do_sample=False여도 top_k·top_p 필터를 먼저 적용합니다.
            # 기본 top_k=50은 어휘 수(16)보다 커서 오류가 나므로, 필터를 모두 끕니다.
            result = model.generate(
                ids, max_new_tokens=5, do_sample=False,
                eos_token_id=token_to_id["<eos>"], use_cache=True,
                top_k=0, top_p=1.0, temperature=1.0,
            )
        return decode(result[0])
    return (complete,)


@app.cell
def _(loss_floor, plt, pretrain_losses):
    training_fig, _ax = plt.subplots(figsize=(6, 3))
    _ax.plot(pretrain_losses, label="Training loss")
    _ax.axhline(loss_floor, color="gray", linestyle="--", label=f"Lower bound {loss_floor:.3f}")
    _ax.set(xlabel="Step", ylabel="Cross entropy", title="Full parameter training")
    _ax.legend()
    training_fig.tight_layout()
    return (training_fig,)


@app.cell(hide_code=True)
def _(
    complete,
    final_pretrain_loss,
    initial_model,
    loss_floor,
    mo,
    pretrain_losses,
    trained_model,
    training_fig,
):
    # ▸ 결과 표시 — 실행하면 표·그림이 나타납니다 (marimo 표시 코드)
    mo.vstack([
        training_fig,
        mo.md(f"같은 배치의 학습 전/후 손실: **{pretrain_losses[0]:.4f} → {final_pretrain_loss:.4f}** (하한 `{loss_floor:.3f}`)"),
        *[
            mo.vstack([
                mo.md(f"#### {_name}"),
                mo.ui.table([
                    {"입력": p, "생성 결과": complete(_model, p)}
                    for p in ["고양이 는", "강아지 는", "소 는", "오리 는"]
                ], selection=None),
            ])
            for _name, _model in [("학습 전", initial_model), ("학습 후", trained_model)]
        ],
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 5-1. 마지막 벡터 → 어휘 점수 → 확률 → 손실

    **예측:** 학습 후 `고양이 는` 다음의 `야옹` 확률은 커질까요?
    1. **마지막 hidden state:** 모든 블록과 마지막 RMSNorm을 지난 벡터입니다.
    2. **LM head:** 이 벡터를 어휘별 가중치와 내적해 **logits**를 만듭니다.
       가중치가 `[V,D]`이므로 `h @ weight.T`로 계산합니다.
    3. **softmax:** 점수를 합이 1인 확률로 바꿉니다.
    4. **손실:** $-\log(p_{정답})$. 정답 확률이 클수록 작아집니다.

    `weight`는 2절에서 본 Embedding 표와 같은 텐서입니다.
    학습 손실은 패딩을 뺀 모든 위치의 평균입니다. 여기서는 그중 한 위치만 봅니다.
    """)
    return


@app.cell
def _(F, encode, initial_model, token_to_id, torch, trained_model, vocabulary):
    _prefix = torch.tensor([encode("고양이 는", add_eos=False)])
    _target = token_to_id["야옹"]
    probability_rows = []
    target_losses = {}
    with torch.no_grad():
        for _name, _model in [("학습 전", initial_model), ("학습 후", trained_model)]:
            _out = _model(_prefix)
            _h = _out.hidden_states[0, -1]
            _logits = _h @ _model.lm_head.weight.T
            torch.testing.assert_close(_logits, _out.logits[0, -1])
            _probabilities = _logits.softmax(-1)
            _single_loss = -_logits.log_softmax(-1)[_target]
            torch.testing.assert_close(_single_loss, F.cross_entropy(_logits[None], torch.tensor([_target])))
            target_losses[_name] = float(_single_loss)
            for _i, _word in enumerate(vocabulary):
                probability_rows.append({"시점": _name, "후보 토큰": _word, "logit": float(_logits[_i]), "확률": float(_probabilities[_i])})
    return probability_rows, target_losses


@app.cell(hide_code=True)
def _(mo, probability_rows, target_losses):
    # ▸ 결과 표시 — 실행하면 표·그림이 나타납니다 (marimo 표시 코드)
    mo.vstack([
        *[
            mo.vstack([
                mo.md(f"#### {_name}"),
                mo.ui.table([
                    {k: v for k, v in _row.items() if k != "시점"}
                    for _row in probability_rows if _row["시점"] == _name
                ], selection=None, page_size=len(probability_rows) // 2),
            ])
            for _name in ["학습 전", "학습 후"]
        ],
        mo.md(f"`야옹` 한 위치의 손실: **{target_losses['학습 전']:.4f} → {target_losses['학습 후']:.4f}**. 배치 평균 손실과는 다릅니다."),
        mo.md("**해설:** 가중치가 문맥을 구분하게 바뀌어 정답 확률이 올라갑니다. `<bos>` 뒤에는 동물 네 개가 모두 정답이라 전체 손실은 0이 안 될 수 있습니다."),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### 5-2. 생성은 한 토큰씩 입력을 늘리는 반복

    **예측:** `야옹`을 선택한 다음에는 어떤 문장을 모델에 다시 넣어야 할까요?
    - 학습: 정답 문장의 모든 위치를 한 번에 예측합니다.
    - 생성: 정답을 모릅니다. 마지막 위치에서 하나를 고르고, 입력에 붙여 다시 계산합니다.

    여기서는 가장 큰 점수를 고릅니다(greedy). sampling은 확률대로 뽑는 다른 방식입니다.
    temperature·top-k·top-p 같은 sampling 설정은 `minimind_sampling.py`에서 다룹니다.
    캐시 없이 직접 반복해 보고, 원본 `generate` 결과와 비교합니다.
    """)
    return


@app.cell
def _(complete, decode, encode, token_to_id, torch, trained_model, vocabulary):
    _ids = torch.tensor([encode("고양이 는", add_eos=False)])
    generation_trace = []
    with torch.no_grad():
        for _step in range(5):
            _logits = trained_model(_ids).logits[:, -1, :]
            _next = _logits.argmax(-1, keepdim=True)
            generation_trace.append({"step": _step + 1, "이번 입력": decode(_ids[0]), "선택 토큰": vocabulary[int(_next.item())]})
            _ids = torch.cat([_ids, _next], dim=1)
            if int(_next.item()) == token_to_id["<eos>"]:
                break
    assert decode(_ids[0]) == complete(trained_model, "고양이 는")
    return (generation_trace,)


@app.cell(hide_code=True)
def _(generation_trace, mo):
    # ▸ 결과 표시 — 실행하면 표·그림이 나타납니다 (marimo 표시 코드)
    mo.vstack([
        mo.ui.table(generation_trace, selection=None),
        mo.md("**관찰:** 고른 토큰이 다음 입력 끝에 붙습니다. `<eos>`가 나오면 멈춥니다. 원본 `generate`와 결과가 같습니다."),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 5-3. 같은 토큰·같은 head의 Attention 비교

    비교 대상: 첫 문장, **1층(layer 0)의 head 0**.

    `query`는 정보를 가져오는 위치, `key`는 참고하는 위치입니다.
    3-4절의 `attention_forward` 함수를 다시 씁니다.
    학습 전 값이 3-3절 결과와 같은지 먼저 확인합니다.

    학습 전후 수치를 비교하세요. 단, 이 비율만으로 모델의 판단을 다 설명할 수는 없습니다.
    """)
    return


@app.cell
def _(
    attention_forward,
    attention_probabilities,
    initial_model,
    input_ids,
    torch,
    trained_model,
    use_causal_mask,
):
    # 3-4절의 attention_forward로 첫 문장, 1층 head 0의 확률 [T, T]를 구합니다.
    with torch.no_grad():
        attention_before = attention_forward(initial_model, input_ids[:1])[0][0, 0]
        attention_after = attention_forward(trained_model, input_ids[:1])[0][0, 0]
    if use_causal_mask:
        # 함수 결과가 3-3절의 직접 계산과 같은지 확인합니다.
        torch.testing.assert_close(attention_before, attention_probabilities[0, 0])
    return attention_after, attention_before


@app.cell
def _(
    attention_after,
    attention_before,
    input_ids,
    korean_font,
    plt,
    vocabulary,
):
    _words = [vocabulary[int(i)] for i in input_ids[0]]
    _t = len(_words)
    attention_rows = [
        {"query 위치": i, "query": _words[i], "key 위치": j, "key": _words[j],
         "학습 전 확률": float(attention_before[i, j]), "학습 후 확률": float(attention_after[i, j])}
        for i in range(_t) for j in range(i + 1)   # j ≤ i: 미래(가려진 칸)는 표에서 뺍니다
    ]
    _ticks = _words if korean_font else [str(i) for i in range(_t)]
    attention_fig, _axes = plt.subplots(1, 2, figsize=(10, 4))
    for _axis, _matrix, _title in zip(_axes, [attention_before, attention_after], ["Before training", "After training"]):
        _image = _axis.imshow(_matrix.numpy(), vmin=0, vmax=1, cmap="Blues")
        _axis.set_xticks(range(_t), labels=_ticks, rotation=45, ha="right", fontsize=8)
        _axis.set_yticks(range(_t), labels=_ticks, fontsize=8)
        if korean_font:
            for _label in _axis.get_xticklabels() + _axis.get_yticklabels():
                _label.set_fontfamily(korean_font)
        _axis.set(xlabel="Key", ylabel="Query", title=f"{_title} (layer 0, head 0)")
    attention_fig.colorbar(_image, ax=list(_axes), shrink=0.7)
    return attention_fig, attention_rows


@app.cell(hide_code=True)
def _(attention_fig, attention_rows, mo):
    # ▸ 결과 표시 — 실행하면 표·그림이 나타납니다 (marimo 표시 코드)
    mo.vstack([
        attention_fig,
        *[
            mo.vstack([
                mo.md(f"#### {_name}"),
                mo.ui.table([
                    {"query 위치": _row["query 위치"], "query": _row["query"],
                     "key 위치": _row["key 위치"], "key": _row["key"],
                     "확률": _row[f"{_name} 확률"]}
                    for _row in attention_rows
                ], selection=None, page_size=14),
            ])
            for _name in ["학습 전", "학습 후"]
        ],
        mo.md("한글 폰트가 없으면 축에 위치 번호가 나옵니다. 실제 토큰은 아래 표에서 확인하세요."),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 6. 미래 토큰 가리기와 KV cache 확인

    **검증 1:** 뒤 토큰을 바꿔도 앞 위치의 예측은 같아야 합니다. 미래를 볼 수 없기 때문입니다.

    **검증 2:** **KV cache**는 과거 토큰의 K/V를 저장해 중복 계산을 줄입니다.
    마지막 토큰 하나만 넣어도 전체 재계산과 결과가 같아야 합니다.
    캐시는 속도만 바꿉니다. 예측은 바꾸지 않습니다.
    """)
    return


@app.cell
def _(input_ids, token_to_id, torch, trained_model):
    with torch.no_grad():
        # 검증 1: 끝에서 두 번째 토큰(`.`)을 <pad>로 바꿉니다.
        # 그보다 앞 위치들은 이 토큰을 볼 수 없으므로 logits가 그대로여야 합니다.
        _original = trained_model(input_ids).logits
        _changed_ids = input_ids.clone()
        _changed_ids[:, -2] = token_to_id["<pad>"]
        _changed = trained_model(_changed_ids).logits
        torch.testing.assert_close(_original[:, :-2], _changed[:, :-2])

        # 검증 2: 마지막 토큰 전까지 계산해 K/V를 캐시에 담고, 마지막 토큰 하나만 넣습니다.
        cache_prefix = trained_model(input_ids[:, :-1], use_cache=True)
        _cached = trained_model(
            input_ids[:, -1:], past_key_values=cache_prefix.past_key_values, use_cache=True,
        )
        torch.testing.assert_close(_original[:, -1:], _cached.logits, atol=1e-5, rtol=1e-4)
        cache_max_error = (_original[:, -1:] - _cached.logits).abs().max().item()
    return cache_max_error, cache_prefix


@app.cell(hide_code=True)
def _(cache_max_error, cache_prefix, mo):
    # ▸ 결과 설명 — 실행하면 계산 값이 들어간 설명이 나타납니다 (marimo 표시 코드)
    mo.md(f"두 검증 통과. 전체 계산과 캐시 계산의 최대 오차: `{cache_max_error:.2e}`. 캐시에 저장되는 K의 shape: `{tuple(cache_prefix.past_key_values[0][0].shape)}` (GQA 공유 전).")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 6-1. 캐시는 왜 RoPE와 함께 맞게 동작하나

    결과가 같다는 것만으로는 원리가 보이지 않습니다. 1층 Attention을 캐시로 직접 계산해 봅니다.

    **① 캐시에 무엇이 저장되나:** RoPE를 **적용한 뒤**, `repeat_kv`로 **펼치기 전**의 K와 V입니다.

    - 과거 K는 이미 자기 위치의 각도로 회전돼 있습니다. 다시 회전할 필요가 없습니다.
    - 펼치기 전이라 KV head 수만큼만 저장합니다. GQA가 메모리를 줄이는 지점입니다.

    **② 새 토큰은 몇 번 위치로 회전하나:** 캐시 길이가 곧 새 토큰의 위치입니다.
    원본 `MiniMindModel.forward`는 `start_pos = 캐시 길이`로 두고 `freqs_cos[start_pos:]`를 씁니다.
    위치 0으로 잘못 회전하면, 과거 K와의 거리가 틀어져 결과가 달라집니다.

    **③ 미래 가리기는 어떻게 되나:** 새 query는 1개이고 과거는 모두 볼 수 있습니다.
    원본 129행은 마스크를 **마지막 `seq_len`개 key 열**에만 더합니다(`scores[..., -seq_len:]`).
    `seq_len = 1`이면 가릴 칸이 없습니다.
    """)
    return


@app.cell
def _(
    F,
    apply_rotary_pos_emb,
    cache_prefix,
    input_ids,
    math,
    repeat_kv,
    torch,
    trained_model,
):
    with torch.no_grad():
        _block = trained_model.model.layers[0]
        _a = _block.self_attn
        _x = _block.input_layernorm(trained_model.model.embed_tokens(input_ids))   # [B, T, D]
        _b, _t, _ = _x.shape
        cache_start_pos = _t - 1          # 캐시 길이 = 새 토큰의 위치

        def rotated_qkv(x, start_pos):
            # x의 첫 토큰을 start_pos 위치로 보고 Q/K/V를 만든 뒤 Q/K를 회전합니다.
            t = x.shape[1]
            q = _a.q_norm(_a.q_proj(x).view(_b, t, _a.n_local_heads, _a.head_dim))
            k = _a.k_norm(_a.k_proj(x).view(_b, t, _a.n_local_kv_heads, _a.head_dim))
            v = _a.v_proj(x).view(_b, t, _a.n_local_kv_heads, _a.head_dim)
            cos = trained_model.model.freqs_cos[start_pos:start_pos + t]
            sin = trained_model.model.freqs_sin[start_pos:start_pos + t]
            q, k = apply_rotary_pos_emb(q, k, cos, sin)
            return q, k, v

        # ① 캐시 = 앞부분(위치 0..T-2)의 회전된 K와 V, 펼치기 전
        _cache_k, _cache_v = cache_prefix.past_key_values[0]
        _, _k_prefix, _v_prefix = rotated_qkv(_x[:, :cache_start_pos], 0)
        torch.testing.assert_close(_cache_k, _k_prefix)
        torch.testing.assert_close(_cache_v, _v_prefix)

        def last_token_attention(start_pos):
            # ② 새 토큰 하나만 계산하고, 과거 K/V는 캐시에서 이어 붙입니다.
            q_new, k_new, v_new = rotated_qkv(_x[:, cache_start_pos:], start_pos)
            k_all = repeat_kv(torch.cat([_cache_k, k_new], dim=1), _a.n_rep).transpose(1, 2)
            v_all = repeat_kv(torch.cat([_cache_v, v_new], dim=1), _a.n_rep).transpose(1, 2)
            scores = q_new.transpose(1, 2) @ k_all.transpose(-2, -1) / math.sqrt(_a.head_dim)
            # ③ query 1개 × key T개. 새 토큰보다 뒤인 key가 없으므로 가릴 칸이 없습니다.
            probs = F.softmax(scores, dim=-1)
            return _a.o_proj((probs @ v_all).transpose(1, 2).reshape(_b, 1, -1))

        _full, _ = _a(_x, (trained_model.model.freqs_cos[:_t], trained_model.model.freqs_sin[:_t]))
        cache_right_error = (last_token_attention(cache_start_pos) - _full[:, -1:]).abs().max().item()
        cache_wrong_error = (last_token_attention(0) - _full[:, -1:]).abs().max().item()
        assert cache_right_error < 1e-5
        assert cache_wrong_error > 100 * cache_right_error
        cache_k_shape = tuple(_cache_k.shape)
    return cache_k_shape, cache_right_error, cache_start_pos, cache_wrong_error


@app.cell(hide_code=True)
def _(
    cache_k_shape,
    cache_right_error,
    cache_start_pos,
    cache_wrong_error,
    mo,
):
    # ▸ 결과 표시 — 실행하면 표·그림이 나타납니다 (marimo 표시 코드)
    mo.vstack([
        mo.md(f"**①** 캐시의 K는 앞부분을 회전한 K와 같습니다. shape `{cache_k_shape}` = `[B, 캐시 길이, KV head, d]`."),
        mo.ui.table([
            {"새 토큰을 회전한 위치": f"start_pos = {cache_start_pos} (올바름)", "전체 계산과의 최대 오차": f"{cache_right_error:.2e}"},
            {"새 토큰을 회전한 위치": "0 (잘못됨)", "전체 계산과의 최대 오차": f"{cache_wrong_error:.2e}"},
        ], selection=None),
        mo.md("**관찰:** 올바른 위치로 회전해야 전체 계산과 같아집니다. RoPE는 상대 거리로 점수를 정하므로, 새 토큰의 위치가 틀리면 모든 과거 토큰과의 거리가 틀어집니다."),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 7. 직접 바꿔 볼 과제

    1. `sentences`에 새 동물 문장을 추가하세요. 어휘 수와 LM head의 shape 변화를 확인하세요.
    2. `config`의 `num_key_value_heads`를 4로 바꾸세요. K/V 크기와 6절의 캐시 shape을 비교하세요.
    3. `config`의 `hidden_size`를 64로 바꾸세요. head 차원, 매개변수 수, 학습 속도를 비교하세요.
    4. 3-3절 코드 셀에서 `use_causal_mask = False`로 바꾸세요.
       미래 위치로 간 확률을 확인하세요. 정답 누출이 왜 생기나요? 끝나면 `True`로 되돌리세요.

    **확인 질문:**

    - `labels=input_ids`인데 왜 자기 자신을 맞히는 학습이 아닌가요?
    - Attention과 MLP는 각각 어느 축을 바꾸나요?

    **기록할 것:** 예상 shape, 실제 shape, 평균 손실, 동물별 정답 수(0~4).
    손실 감소와 "네 문장 모두 정답"은 따로 판정하세요. 학습이 짧으면 일부만 맞아도 정상입니다.

    먼저 직접 답을 적은 뒤, 아래 해설을 펼쳐 비교하세요.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({
        "힌트·해설: 과제 1~4": mo.md("""
    - **새 동물:** 어휘가 동물 이름·울음소리만큼 늘어납니다. LM head와 Embedding의 `V` 축이 함께 커집니다. 같은 표이기 때문입니다. 손실 하한도 `ln(동물 수) / 6`으로 바뀝니다.
    - **KV head 2→4:** `n_rep`가 1이 되어 공유하지 않습니다. K/V head 축과 캐시 크기가 두 배입니다.
    - **hidden 32→64, Q head 4:** head 차원이 8→16입니다. 매개변수가 늘고 학습이 느려집니다.
    - **causal mask 제거:** 위치 t가 t+1의 토큰을 봅니다. 그런데 그 토큰이 바로 t의 정답입니다. 학습 때 답을 보고 맞히는 셈이라, 생성할 때는 쓸 수 없는 모델이 됩니다.
    """),
        "해설: 확인 질문": mo.md(r"""
    **`labels=input_ids`인데 왜 자기 자신을 맞히지 않나요?**
    forward 안에서 labels가 한 칸 밀립니다. 위치 t의 출력은 t+1의 토큰과 비교됩니다(4절).

    **Attention과 MLP는 어느 축을 바꾸나요?**
    입력 텐서는 **`[B, T, D]` = [배치, 토큰 위치, 특징 차원]**입니다.

    | 구성 요소 | 주로 정보를 섞는 축 | 하는 일 |
    | --- | --- | --- |
    | Attention | 토큰 위치 `T` | 다른 토큰의 정보를 가져와 현재 토큰의 벡터를 갱신합니다. |
    | MLP | 특징 차원 `D` | 각 토큰 안의 특징들을 조합해 벡터를 변환합니다. |

    예: **`고양이 는 야옹`**

    - Attention: `는`이 앞의 `고양이`를 참고합니다. 뒤의 `야옹`은 못 봅니다.
    - MLP: `는` 벡터 안의 특징만 가공합니다. 다른 위치는 보지 않습니다.

    두 연산 모두 출력 shape은 **`[B, T, D]` 그대로**입니다.
    Attention도 선형 변환에서 특징 축을 씁니다. 핵심 차이는 **토큰끼리 정보를 섞느냐**입니다.
    """),
    })
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    **다음 실습:**

    - 블록 내부: `minimind_feedforward.py` → `minimind_moe.py`
    - 생성 설정: `minimind_sampling.py` (temperature, top-k, top-p, repetition penalty)
    - 학습 파이프라인: `minimind_pretraining.py` → `minimind_sft.py`

    5절의 반복 학습은 사전 학습 노트북에서 실제 Dataset·DataLoader·학습률로 확장됩니다.
    """)
    return


if __name__ == "__main__":
    app.run()
