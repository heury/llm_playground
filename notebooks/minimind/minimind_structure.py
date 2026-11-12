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
    return (
        F,
        MiniMindConfig,
        MiniMindForCausalLM,
        apply_rotary_pos_emb,
        copy,
        font_manager,
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

    - **과정의 첫 노트북**입니다.
    - 뒤 노트북은 Attention·RoPE·GQA·손실·생성을 다시 설명하지 않습니다.
    - 모델을 내려받지 않습니다. **무작위 가중치로 CPU에서 학습**합니다.
    - 네 문장 암기는 구조 관찰용입니다. 한국어 능력 평가가 아닙니다.

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

    - **GQA:** 여러 Query head가 Key/Value를 나눠 씁니다.
    - head 차원은 `32 / 4 = 8`입니다. RoPE 때문에 짝수여야 합니다.
    - Embedding과 LM head는 같은 가중치입니다.
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
    2. 미래 위치를 가리고 softmax를 적용합니다. 각 행의 합은 1입니다.
    3. 그 비율로 V를 가중합합니다.

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

    이 구현은 Q와 K에도 RMSNorm을 한 번 더 씁니다.

    **shape:** 입력 `[B,T,32]` → Q `[B,T,4,8]`, K/V `[B,T,2,8]`.
    `view`는 head 축을 나눕니다. `transpose`는 행렬 곱을 위해 축 순서를 바꿉니다.
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
        q = attn.q_norm(attn.q_proj(normalized).view(_b, _t, attn.n_local_heads, attn.head_dim))
        k = attn.k_norm(attn.k_proj(normalized).view(_b, _t, attn.n_local_kv_heads, attn.head_dim))
        v = attn.v_proj(normalized).view(_b, _t, attn.n_local_kv_heads, attn.head_dim)
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

    **RoPE:** Q/K를 위치만큼 회전시킵니다. 그래서 점수에 위치 관계가 들어갑니다.

    - 같은 벡터도 위치가 다르면 회전 결과가 다릅니다.
    - 벡터 길이는 그대로입니다. 아래에서 확인합니다.

    **`repeat_kv`:** K/V head를 반복해서 Q head 수에 맞춥니다(GQA).
    Q head 4개, K/V head 2개이므로 `attn.n_rep = 4 // 2 = 2`입니다.

    ```text
    Q head:       Q0   Q1   Q2   Q3
    대응 K head:  K0   K0   K1   K1
    대응 V head:  V0   V0   V1   V1
    ```

    K/V shape: `[B, T, 2, 8]` → `[B, T, 4, 8]`.

    - 함수 안: `None`으로 축 추가 → `expand`로 펼침 → `reshape`로 합침.
    - 반복 횟수가 1이면 그대로 반환합니다.
    - 새 K/V를 만드는 게 아닙니다. 기존 값을 나눠 씁니다.
    - K/V가 같아도 Q가 다르므로 head마다 Attention 결과는 다릅니다.
    - 캐시에는 펼치기 전 K/V를 저장합니다. 그래서 메모리가 줄어듭니다.
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
        query_heads = q_rotated.transpose(1, 2)
        key_heads = repeat_kv(k_rotated, attn.n_rep).transpose(1, 2)
        value_heads = repeat_kv(v, attn.n_rep).transpose(1, 2)
        torch.testing.assert_close(q.norm(dim=-1), q_rotated.norm(dim=-1))
        # 같은 Q 벡터를 위치 1과 위치 2에 각각 놓아 회전 효과만 비교합니다.
        same_q_at_2, _ = apply_rotary_pos_emb(
            q[:, 1:2], k[:, 1:2],
            initial_model.model.freqs_cos[2:3], initial_model.model.freqs_sin[2:3],
        )
        torch.testing.assert_close(q_rotated[:, 1:2].norm(dim=-1), same_q_at_2.norm(dim=-1))
        torch.testing.assert_close(key_heads[:, 0], key_heads[:, 1])
    return (
        key_heads,
        position_embeddings,
        q_rotated,
        query_heads,
        same_q_at_2,
        value_heads,
    )


@app.cell(hide_code=True)
def _(key_heads, mo, q_rotated, query_heads, same_q_at_2):
    # ▸ 결과 표시 — 실행하면 표·그림이 나타납니다 (marimo 표시 코드)
    mo.vstack([
        mo.md(f"회전·공유 후 계산 shape: Q `{tuple(query_heads.shape)}`, K/V `{tuple(key_heads.shape)}`."),
        mo.ui.table([
            {"차원": i, "같은 Q를 위치 1에서 회전": float(q_rotated[0, 1, 0, i]), "같은 Q를 위치 2에서 회전": float(same_q_at_2[0, 0, 0, i])}
            for i in range(4)
        ], selection=None),
        mo.md("**관찰:** 같은 벡터도 위치마다 좌표가 다릅니다. 길이는 같습니다. 펼친 K의 첫 두 head는 같은 원본 head입니다."),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### 3-3. 점수 → 미래 가리기 → 확률 → 가중합

    3절의 작은 예제와 같은 규칙입니다. 이번에는 모든 배치와 head에 적용합니다.

    - head별 출력을 이어 붙이고 `o_proj`로 섞습니다.
    - `causal mask`는 미래를 가립니다.
    - `attention_mask`는 패딩을 가립니다. 지금은 패딩이 없습니다(4절에서 확인).

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
    with torch.no_grad():
        _b, _t, _d = normalized.shape
        _scores = query_heads @ key_heads.transpose(-2, -1) / math.sqrt(attn.head_dim)
        causal_mask = torch.ones(_t, _t, dtype=torch.bool).triu(1)
        attention_probabilities = F.softmax(_scores.masked_fill(causal_mask, -torch.inf), dim=-1)
        _context = (attention_probabilities @ value_heads).transpose(1, 2).reshape(_b, _t, -1)
        manual_attention = attn.o_proj(_context)
        _actual, _ = attn(normalized, position_embeddings)
        torch.testing.assert_close(manual_attention, _actual)
        assert attention_probabilities.masked_select(causal_mask[None, None]).eq(0).all()
    return attention_probabilities, manual_attention


@app.cell(hide_code=True)
def _(attention_probabilities, manual_attention, mo, normalized):
    # ▸ 결과 표시 — 실행하면 표·그림이 나타납니다 (marimo 표시 코드)
    mo.vstack([
        mo.ui.table([
            {"단계": "Embedding / RMSNorm", "shape": str(tuple(normalized.shape))},
            {"단계": "Attention 확률", "shape": str(tuple(attention_probabilities.shape))},
            {"단계": "출력", "shape": str(tuple(manual_attention.shape))},
        ], selection=None),
        mo.md("직접 계산한 Attention과 원본 출력이 같습니다. 각 행의 미래 위치 확률은 0입니다."),
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

    **다음 토큰 정답:** 출력은 `[B,T,V]`입니다. 위치 `t`의 출력이 `t+1` 토큰을 예측합니다.
    shift는 원본 `forward(labels=...)`가 합니다. **labels를 미리 옮기지 마세요.**

    **손실:** 출력은 확률 전 점수(logits)입니다.
    softmax로 정답 확률 p를 구하면 손실은 `-log(p)`입니다.
    cross entropy가 이를 안정적으로 계산합니다.
    학습은 정답 확률을 높이는 쪽으로 가중치를 바꿉니다.
    """)
    return


@app.cell
def _(
    F,
    attention_mask,
    embeddings,
    first_block,
    initial_model,
    input_ids,
    labels,
    make_batch,
    manual_attention,
    position_embeddings,
    torch,
    vocabulary,
):
    with torch.no_grad():
        _residual = embeddings + manual_attention
        _mlp_input = first_block.post_attention_layernorm(_residual)
        _ff = first_block.mlp
        _manual_mlp = _ff.down_proj(F.silu(_ff.gate_proj(_mlp_input)) * _ff.up_proj(_mlp_input))
        _manual_block = _residual + _manual_mlp
        _actual_block, _ = first_block(embeddings, position_embeddings)
        torch.testing.assert_close(_manual_block, _actual_block)
        initial_out = initial_model(input_ids, attention_mask=attention_mask, labels=labels)
        _manual_loss = F.cross_entropy(
            initial_out.logits[:, :-1].reshape(-1, len(vocabulary)), labels[:, 1:].reshape(-1),
            ignore_index=-100,
        )
        torch.testing.assert_close(_manual_loss, initial_out.loss)
        # 길이가 다른 문장에서도 padding 정답은 손실에 포함되지 않습니다.
        _pad_ids, pad_mask, pad_labels = make_batch([
            "고양이 는 야옹 한다 .", "강아지 는",
        ])
        assert (pad_labels == -100).any()
        _pad_out = initial_model(_pad_ids, attention_mask=pad_mask, labels=pad_labels)
        _valid = pad_labels[:, 1:] != -100
        _pad_loss = F.cross_entropy(_pad_out.logits[:, :-1][_valid], pad_labels[:, 1:][_valid])
        torch.testing.assert_close(_pad_loss, _pad_out.loss)
    return initial_out, pad_labels, pad_mask


@app.cell(hide_code=True)
def _(initial_out, input_ids, labels, mo, pad_labels, pad_mask, vocabulary):
    # ▸ 결과 표시 — 실행하면 표·그림이 나타납니다 (marimo 표시 코드)
    mo.vstack([
        mo.md(f"MLP와 잔차 연결 검증 통과. Logits shape: `{tuple(initial_out.logits.shape)}`, 초기 손실: `{initial_out.loss.item():.4f}`"),
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

    네 문장을 반복해 암기시킵니다.
    `<bos>` 다음에는 동물 네 개가 모두 가능합니다. 그래서 손실이 0이 안 돼도 됩니다.
    학습 횟수를 바꿔 가며 손실과 생성 문장을 비교하세요.
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
    return final_pretrain_loss, pretrain_losses, trained_model


@app.cell
def _(decode, encode, plt, pretrain_losses, torch):
    def complete(model, prompt):
        model.eval()
        ids = torch.tensor([encode(prompt, add_eos=False)])
        with torch.no_grad():
            result = model.generate(
                ids, max_new_tokens=5, do_sample=False, eos_token_id=2, use_cache=True,
                top_k=0, top_p=1.0, temperature=1.0,
            )
        return decode(result[0])

    training_fig, _ax = plt.subplots(figsize=(6, 3))
    _ax.plot(pretrain_losses)
    _ax.set(xlabel="Step", ylabel="Cross entropy", title="Full parameter training")
    training_fig.tight_layout()
    return complete, training_fig


@app.cell(hide_code=True)
def _(
    complete,
    final_pretrain_loss,
    initial_model,
    mo,
    pretrain_losses,
    trained_model,
    training_fig,
):
    # ▸ 결과 표시 — 실행하면 표·그림이 나타납니다 (marimo 표시 코드)
    mo.vstack([
        training_fig,
        mo.md(f"같은 배치의 학습 전/후 손실: **{pretrain_losses[0]:.4f} → {final_pretrain_loss:.4f}**"),
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

    Embedding과 LM head는 같은 표를 씁니다. 입력에서는 행을 고르고, 출력에서는 모든 행과 내적합니다.
    학습 손실은 패딩을 뺀 모든 위치의 평균입니다.
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
    캐시 없이 직접 반복해 보고, 원본 `generate` 결과와 비교합니다.
    """)
    return


@app.cell
def _(complete, decode, encode, torch, trained_model, vocabulary):
    _ids = torch.tensor([encode("고양이 는", add_eos=False)])
    generation_trace = []
    with torch.no_grad():
        for _step in range(5):
            _logits = trained_model(_ids).logits[:, -1, :]
            _next = _logits.argmax(-1, keepdim=True)
            generation_trace.append({"step": _step + 1, "이번 입력": decode(_ids[0]), "선택 토큰": vocabulary[int(_next.item())]})
            _ids = torch.cat([_ids, _next], dim=1)
            if int(_next.item()) == 2:
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

    `query`는 정보를 가져오는 위치, `key`는 참고하는 위치입니다.
    학습 전후 수치를 비교하세요.
    단, 이 비율만으로 모델의 판단을 다 설명할 수는 없습니다.
    """)
    return


@app.cell
def _(
    F,
    apply_rotary_pos_emb,
    attention_probabilities,
    font_manager,
    input_ids,
    math,
    plt,
    repeat_kv,
    torch,
    trained_model,
    vocabulary,
):
    with torch.no_grad():
        _layer = trained_model.model.layers[0]
        _a = _layer.self_attn
        _x = _layer.input_layernorm(trained_model.model.embed_tokens(input_ids[:1]))
        _t = _x.shape[1]
        _q = _a.q_norm(_a.q_proj(_x).view(1, _t, _a.n_local_heads, _a.head_dim))
        _k = _a.k_norm(_a.k_proj(_x).view(1, _t, _a.n_local_kv_heads, _a.head_dim))
        _q, _k = apply_rotary_pos_emb(_q, _k, trained_model.model.freqs_cos[:_t], trained_model.model.freqs_sin[:_t])
        _scores = _q.transpose(1, 2) @ repeat_kv(_k, _a.n_rep).transpose(1, 2).transpose(-2, -1) / math.sqrt(_a.head_dim)
        _mask = torch.ones(_t, _t, dtype=torch.bool).triu(1)
        _after = F.softmax(_scores.masked_fill(_mask, -torch.inf), dim=-1)[0, 0]
    _words = [vocabulary[int(i)] for i in input_ids[0]]
    attention_rows = [
        {"query 위치": i, "query": _words[i], "key 위치": j, "key": _words[j],
         "학습 전 확률": float(attention_probabilities[0, 0, i, j]), "학습 후 확률": float(_after[i, j])}
        for i in range(_t) for j in range(i + 1)
    ]
    _font = next((f for f in ["Noto Sans CJK KR", "Noto Sans CJK JP", "Malgun Gothic", "AppleGothic", "NanumGothic"] if f in font_manager.get_font_names()), None)
    _ticks = _words if _font else [str(i) for i in range(_t)]
    attention_fig, _axes = plt.subplots(1, 2, figsize=(10, 4))
    for _axis, _matrix, _title in zip(_axes, [attention_probabilities[0, 0], _after], ["Before training", "After training"]):
        _image = _axis.imshow(_matrix.numpy(), vmin=0, vmax=1, cmap="Blues")
        _axis.set_xticks(range(_t), labels=_ticks, rotation=45, ha="right", fontsize=8)
        _axis.set_yticks(range(_t), labels=_ticks, fontsize=8)
        if _font:
            for _label in _axis.get_xticklabels() + _axis.get_yticklabels():
                _label.set_fontfamily(_font)
        _axis.set(xlabel="Key", ylabel="Query", title=_title)
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
def _(input_ids, torch, trained_model):
    with torch.no_grad():
        _original = trained_model(input_ids).logits
        _changed_ids = input_ids.clone()
        _changed_ids[:, -2] = 0
        _changed = trained_model(_changed_ids).logits
        torch.testing.assert_close(_original[:, :-2], _changed[:, :-2])
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
    mo.md("""
    ## 7. 직접 바꿔 볼 과제

    1. 샘플 문장에 새 동물을 추가하고 어휘 수와 LM head의 shape 변화를 확인하세요.
    2. `num_key_value_heads`를 4로 바꾸고 K/V의 크기와 캐시 크기를 비교하세요.
    3. `hidden_size`를 64로 바꾸고 head 차원, 매개변수 수와 학습 속도를 비교하세요.
    4. 수동 Attention 계산에서 미래를 가리는 코드를 제거해 보세요. 정답 누출이 왜 생기나요?

    **확인 질문:**

    - `labels=input_ids`인데 왜 자기 자신을 맞히는 학습이 아닌가요?
    - Attention과 MLP는 각각 어느 축을 바꾸나요?

    **기록할 것:** 예상 shape, 실제 shape, 평균 손실, 동물별 정답 수(0~4).
    손실 감소와 "네 문장 모두 정답"은 따로 판정하세요. 학습이 짧으면 일부만 맞아도 정상입니다.

    **힌트·해설:**

    - KV head 2→4: K/V head 축이 두 배, 캐시 크기도 두 배입니다.
    - hidden 32→64, Q head 4: head 차원이 8→16입니다.
    - causal mask 제거: 위치 t가 t+1의 정답을 입력에서 봅니다.
    - labels는 forward 안에서 한 칸 밀립니다.
    - Attention은 토큰 사이, MLP는 특징 차원을 처리합니다.

    **다음 실습:**

    - 블록 내부: `minimind_feedforward.py` → `minimind_moe.py`
    - 학습 파이프라인: `minimind_pretraining.py` → `minimind_sft.py`

    5절의 반복 학습은 사전 학습 노트북에서 실제 Dataset·DataLoader·학습률로 확장됩니다.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 과제 해설: Attention과 MLP가 처리하는 축

    입력 텐서의 shape은 **`[B, T, D]` = [배치, 토큰 위치, 특징 차원]**입니다.

    | 구성 요소 | 주로 정보를 섞는 축 | 하는 일 |
    | --- | --- | --- |
    | Attention | 토큰 위치 `T` | 다른 토큰의 정보를 가져와 현재 토큰의 벡터를 갱신합니다. |
    | MLP | 특징 차원 `D` | 각 토큰 안의 특징들을 조합해 벡터를 변환합니다. |

    예: **`고양이 는 야옹`**

    - Attention: `는`이 앞의 `고양이`를 참고합니다. 뒤의 `야옹`은 못 봅니다.
    - MLP: `는` 벡터 안의 특징만 가공합니다. 다른 위치는 보지 않습니다.

    두 연산 모두 출력 shape은 **`[B, T, D]` 그대로**입니다.
    Attention도 선형 변환에서 특징 축을 씁니다.
    핵심 차이는 **토큰끼리 정보를 섞느냐**입니다.
    """)
    return


if __name__ == "__main__":
    app.run()
