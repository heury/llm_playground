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
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "minimind"))
    from model.model_minimind import (
        MiniMindConfig, MiniMindForCausalLM,
        apply_rotary_pos_emb, repeat_kv,
    )

    # 작은 CPU 행렬에서는 많은 스레드보다 소수의 스레드가 효율적입니다.
    torch.set_num_threads(2)
    return (
        F, MiniMindConfig, MiniMindForCausalLM,
        apply_rotary_pos_emb, copy, math,
        font_manager, mo, plt, repeat_kv, torch,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # 코드로 이해하는 MiniMind: 모델 구조와 다음 토큰 학습

    **목표:** 문장이 숫자로 바뀌고, Attention과 MLP를 통과한 뒤,
    다음 토큰의 확률이 되는 과정을 직접 계산하고 전체 매개변수를 학습합니다.

    필요한 사전 지식은 Python 함수, 텐서의 `shape`, 행렬 곱(`@`)입니다.
    처음 보는 용어는 아래의 작은 계산으로 확인하고 실제 모델로 확장합니다.
    `B`는 배치 크기, `T`는 토큰 수, `D`는 은닉 차원, `V`는 어휘 수입니다.

    ```text
    문장 → 토큰 ID [B,T] → Embedding [B,T,D]
         → (RMSNorm → Attention → 잔차 연결 → RMSNorm → MLP → 잔차 연결) × 층 수
         → RMSNorm → LM head [B,T,V] → 다음 토큰
    ```

    `minimind/model/model_minimind.py`의 실제 클래스를 사용합니다.
    LoRA 실습은 별도 파일 `minimind_lora.py`에서 이어집니다.
    모델을 다운로드하지 않고 **무작위 가중치부터 CPU에서 학습**합니다.
    아래 네 문장의 암기는 구조를 관찰하는 실험이며, 일반적인 한국어 능력 평가는 아닙니다.
    셀의 코드를 수정하면 관련 셀이 다시 실행됩니다. 학습 셀은 매번 모델을 복사해 시작합니다.

    **실습 방법:** 먼저 결과를 예측하고, 실행한 뒤 표·그림과 비교하여 이유를 적으세요.
    수정할 곳은 `sentences`, `config`, 화면의 학습 횟수입니다.
    `@app.cell`은 marimo의 셀 표시이며, 함수 인자와 `return`은 셀 사이의 데이터 전달입니다.
    `_`로 시작하는 중간 변수는 해당 셀에서만 사용합니다.
    설정을 바꾸면 관련 학습도 처음부터 실행되므로 이전 학습이 누적되지 않습니다.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## 1. 문장을 숫자로 바꾸기

    실제 LLM은 서브워드 토크나이저를 사용합니다. 여기서는 띄어쓰기 단위로 나누어
    각 숫자의 의미를 쉽게 확인합니다. `<bos>`는 시작, `<eos>`는 끝, `<pad>`는 빈칸입니다.
    길이가 짧은 문장은 패딩하고, 정답의 패딩 위치는 `-100`으로 두어 손실에서 제외합니다.
    """)
    return


@app.cell
def _(mo, torch):
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
    mo.vstack([
        mo.ui.table([{"ID": i, "토큰": w} for i, w in enumerate(vocabulary)], selection=None),
        mo.md(f"첫 문장: `{decode(input_ids[0])}`\n\n입력 shape: `{tuple(input_ids.shape)}`"),
    ])
    return (
        attention_mask, decode, encode, input_ids, labels,
        make_batch, sentences, token_to_id, vocabulary,
    )


@app.cell
def _(mo):
    mo.md("""
    ## 2. 작은 모델 만들기

    은닉 차원 32, 층 2개, Query head 4개, Key/Value head 2개로 시작합니다.
    여러 Query head가 Key/Value를 공유하는 방식이 **GQA**입니다.
    각 head의 차원은 `32 / 4 = 8`이며 RoPE를 위해 짝수여야 합니다.
    Embedding과 LM head는 같은 가중치를 공유합니다.
    """)
    return


@app.cell
def _(MiniMindConfig, MiniMindForCausalLM, mo, torch, vocabulary):
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
    mo.vstack([
        mo.md(f"고유 매개변수: **{sum(p.numel() for p in initial_model.parameters()):,}개**"),
        mo.ui.table([
            {"이름": name, "shape": str(tuple(p.shape)), "개수": p.numel()}
            for name, p in initial_model.named_parameters()
        ], selection=None, page_size=12),
    ])
    return config, initial_model


@app.cell
def _(mo):
    mo.md("""
    ### 2-1. ID 하나가 벡터 하나를 선택한다

    **예측:** 같은 토큰 ID를 두 번 넣으면 같은 벡터가 나올까요?
    Embedding은 `[V,D]` 모양의 학습 가능한 표입니다. ID는 수의 크기로 해석하지 않고
    표의 행 번호로 사용합니다. `고양이` 행의 32개 숫자가 그 토큰의 초기 표현입니다.
    Attention 이후에는 주변 문맥이 반영되어 위치별 표현이 달라질 수 있습니다.
    """)
    return


@app.cell
def _(initial_model, mo, token_to_id, torch):
    with torch.no_grad():
        _id = token_to_id["고양이"]
        _looked_up = initial_model.model.embed_tokens(torch.tensor([_id]))[0]
        _row = initial_model.model.embed_tokens.weight[_id].detach()
        torch.testing.assert_close(_looked_up, _row)
    mo.vstack([
        mo.md(f"`고양이` → ID `{_id}` → 표의 `{_id}`번 행. 전체 표 shape: `{tuple(initial_model.model.embed_tokens.weight.shape)}`"),
        mo.ui.table([
            {"벡터 차원": i, "Embedding 호출": float(_looked_up[i]), "직접 고른 행": float(_row[i])}
            for i in range(8)
        ], selection=None),
        mo.md("**관찰·해설:** 앞 8개 값을 비교했습니다. 같은 ID는 같은 행을 선택합니다. 이 표도 전체 학습에서 갱신됩니다."),
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. 먼저 한 head로 Attention 계산하기

    **예측:** 두 번째 위치는 세 번째 위치를 볼 수 있을까요?
    아래는 `고양이 / 는 / 야옹` 세 위치에 대해 사람이 정한 작은 Q/K/V입니다.
    아직 학습된 의미를 표현하지 않습니다. 계산 규칙만 관찰합니다.

    1. Q와 K의 내적으로 각 위치의 관련 점수를 만듭니다.
    2. 미래 위치를 가리고, 각 행에 softmax를 적용해 합이 1인 비율을 만듭니다.
    3. 그 비율로 V를 가중합하여 각 위치에 전달할 정보를 만듭니다.

    예를 들어 두 번째 행은 `는` 위치의 query가 각 key를 보는 비율입니다.
    """)
    return


@app.cell
def _(F, math, mo, torch):
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
    _words = ["고양이", "는", "야옹"]
    mo.vstack([
        mo.md("`Q/K/V: [3,2] → 점수·확률: [3,3] → 출력: [3,2]`"),
        mo.ui.table([
            {"query": _words[i], "key": _words[j], "내적 점수": float(toy_scores[i, j]),
             "미래인가": bool(toy_mask[i, j]), "확률": float(toy_probabilities[i, j])}
            for i in range(3) for j in range(3)
        ], selection=None),
        mo.md(f"두 번째 위치의 출력 = 각 V × 확률의 합 = `{toy_context[1].tolist()}`. 미래인 `야옹`의 비율은 0입니다."),
        mo.md("**작은 과제:** `toy_v`의 마지막 행을 바꾸기 전에 앞 두 위치의 출력이 바뀔지 예측하세요. **해설:** 앞 두 행은 미래 위치의 확률이 0이므로 바뀌지 않습니다."),
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### 3-1. 실제 MiniMind: Q/K/V와 여러 head

    **RMSNorm**은 각 토큰 벡터의 크기를 정리합니다:
    $\mathrm{RMSNorm}(x)=w\,x/\sqrt{\mathrm{mean}(x^2)+\epsilon}$.
    정규화한 입력에서 Q(무엇을 찾는가), K(어떤 특징인가), V(전달할 정보)를 만듭니다.
    이 구현은 Q와 K에도 RMSNorm을 적용합니다.

    선형층은 입력 특징을 학습 가능한 가중치로 조합합니다. 원본 입력 `[B,T,32]`에서
    Q는 `[B,T,4,8]`, K/V는 `[B,T,2,8]`로 만듭니다.
    `view`는 head 축을 나누고, `transpose`는 행렬 곱을 위해 축 순서를 바꿉니다.
    각 head는 서로 다른 가중치 조합으로 정보를 처리합니다.
    """)
    return


@app.cell
def _(initial_model, input_ids, mo, torch):
    with torch.no_grad():
        embeddings = initial_model.model.embed_tokens(input_ids)
        first_block = initial_model.model.layers[0]
        attn = first_block.self_attn
        normalized = first_block.input_layernorm(embeddings)
        _b, _t, _d = normalized.shape
        q = attn.q_norm(attn.q_proj(normalized).view(_b, _t, attn.n_local_heads, attn.head_dim))
        k = attn.k_norm(attn.k_proj(normalized).view(_b, _t, attn.n_local_kv_heads, attn.head_dim))
        v = attn.v_proj(normalized).view(_b, _t, attn.n_local_kv_heads, attn.head_dim)
    mo.md(f"정규화 입력 `{tuple(normalized.shape)}` → Q `{tuple(q.shape)}`, K/V `{tuple(k.shape)}`. **확인:** head 수 × head 차원이 각 projection의 출력 차원입니다.")
    return attn, embeddings, first_block, k, normalized, q, v


@app.cell
def _(mo):
    mo.md("""
    ### 3-2. 위치 정보(RoPE)와 K/V 공유(GQA)

    RoPE는 Q/K를 위치에 따라 회전시켜 위치 관계를 점수 계산에 반영합니다.
    같은 벡터라도 위치가 달라지면 회전 결과가 달라지고, 벡터 길이는 유지됩니다.
    아래에서는 위치별 회전값을 적용한 뒤 길이 보존을 확인합니다.
    GQA는 Q head 4개가 K/V head 2개를 공유하는 구조입니다.
    계산할 때는 K/V를 head 축에서 두 번씩 펼쳐 Q와 head 수를 맞춥니다.
    캐시에는 펼치기 전 K/V를 저장하므로 더 적은 공간을 사용합니다.
    """)
    return


@app.cell
def _(apply_rotary_pos_emb, attn, initial_model, k, mo, q, repeat_kv, torch, v):
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
        _same_q_at_2, _ = apply_rotary_pos_emb(
            q[:, 1:2], k[:, 1:2],
            initial_model.model.freqs_cos[2:3], initial_model.model.freqs_sin[2:3],
        )
        torch.testing.assert_close(q_rotated[:, 1:2].norm(dim=-1), _same_q_at_2.norm(dim=-1))
        torch.testing.assert_close(key_heads[:, 0], key_heads[:, 1])
    mo.vstack([
        mo.md(f"회전·공유 후 계산 shape: Q `{tuple(query_heads.shape)}`, K/V `{tuple(key_heads.shape)}`."),
        mo.ui.table([
            {"차원": i, "같은 Q를 위치 1에서 회전": float(q_rotated[0, 1, 0, i]), "같은 Q를 위치 2에서 회전": float(_same_q_at_2[0, 0, 0, i])}
            for i in range(4)
        ], selection=None),
        mo.md("**관찰:** 같은 벡터도 위치에 따라 좌표가 달라지고 길이는 유지됩니다. GQA에서는 펼친 K의 첫 두 head가 같은 원본 head를 공유합니다."),
    ])
    return key_heads, position_embeddings, query_heads, value_heads


@app.cell
def _(mo):
    mo.md("""
    ### 3-3. 점수 → 미래 가리기 → 확률 → 가중합

    앞의 작은 예제와 같은 규칙을 모든 배치와 head에 적용합니다.
    마지막에 head별 출력을 이어 붙이고 `o_proj`로 은닉 차원에 맞게 조합합니다.
    `causal mask`는 미래를, `attention_mask`는 패딩 key를 가립니다.
    현재 네 문장은 길이가 같아 패딩이 없습니다. 길이가 다른 예제는 4절에서 확인합니다.
    학습 후에는 같은 위치의 Attention 확률을 다시 그려 비교합니다.
    """)
    return


@app.cell
def _(F, attn, key_heads, math, mo, normalized, position_embeddings, query_heads, torch, value_heads):
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
    mo.vstack([
        mo.ui.table([
            {"단계": "Embedding / RMSNorm", "shape": str(tuple(normalized.shape))},
            {"단계": "Attention 확률", "shape": str(tuple(attention_probabilities.shape))},
            {"단계": "출력", "shape": str(tuple(manual_attention.shape))},
        ], selection=None),
        mo.md("직접 계산한 Attention과 원본 출력이 같습니다. 각 행의 미래 위치 확률은 0입니다."),
    ])
    return attention_probabilities, manual_attention


@app.cell
def _(mo):
    mo.md(r"""
    ## 4. MLP와 잔차 연결, 다음 토큰 정답

    Attention은 **토큰 사이**의 정보를 섞고, MLP는 **각 토큰 안**의 특징을 바꿉니다.
    MiniMind의 MLP는 `down_proj(silu(gate_proj(x)) * up_proj(x))`인 SwiGLU입니다.
    잔차 연결은 처리 결과에 입력을 더하여 정보와 기울기가 흐를 길을 만듭니다.

    모델 출력은 `[B,T,V]`입니다. 시점 `t`의 출력으로 `t+1`의 토큰을 예측합니다.
    원본 `forward(labels=...)`가 이미 한 칸 이동하므로 **labels는 미리 이동하지 않습니다.**
    각 출력은 아직 확률이 아닌 어휘 점수(logits)입니다. softmax로 정답 확률 p를 구했을 때
    그 위치의 손실은 `-log(p)`이고, cross entropy는 이 계산을 수치적으로 안정하게 처리합니다.
    학습 전에는 점수가 부정확하므로 손실이 크고, 정답 확률을 높이는 방향으로 가중치를 바꿉니다.
    """)
    return


@app.cell
def _(F, attention_mask, embeddings, first_block, initial_model, input_ids, labels, make_batch, manual_attention, mo, position_embeddings, torch, vocabulary):
    with torch.no_grad():
        _residual = embeddings + manual_attention
        _mlp_input = first_block.post_attention_layernorm(_residual)
        _ff = first_block.mlp
        _manual_mlp = _ff.down_proj(F.silu(_ff.gate_proj(_mlp_input)) * _ff.up_proj(_mlp_input))
        _manual_block = _residual + _manual_mlp
        _actual_block, _ = first_block(embeddings, position_embeddings)
        torch.testing.assert_close(_manual_block, _actual_block)
        _out = initial_model(input_ids, attention_mask=attention_mask, labels=labels)
        _manual_loss = F.cross_entropy(
            _out.logits[:, :-1].reshape(-1, len(vocabulary)), labels[:, 1:].reshape(-1),
            ignore_index=-100,
        )
        torch.testing.assert_close(_manual_loss, _out.loss)
        # 길이가 다른 문장에서도 padding 정답은 손실에 포함되지 않습니다.
        _pad_ids, _pad_mask, _pad_labels = make_batch([
            "고양이 는 야옹 한다 .", "강아지 는",
        ])
        assert (_pad_labels == -100).any()
        _pad_out = initial_model(_pad_ids, attention_mask=_pad_mask, labels=_pad_labels)
        _valid = _pad_labels[:, 1:] != -100
        _pad_loss = F.cross_entropy(_pad_out.logits[:, :-1][_valid], _pad_labels[:, 1:][_valid])
        torch.testing.assert_close(_pad_loss, _pad_out.loss)
    mo.vstack([
        mo.md(f"MLP와 잔차 연결 검증 통과. Logits shape: `{tuple(_out.logits.shape)}`, 초기 손실: `{_out.loss.item():.4f}`"),
        mo.ui.table([
            {"입력 위치의 토큰": vocabulary[int(x)], "맞혀야 할 다음 토큰": vocabulary[int(y)]}
            for x, y in zip(input_ids[0, :-1], labels[0, 1:])
        ], selection=None),
        mo.md(f"길이가 다른 문장의 패딩 검증도 통과. 두 번째 문장의 mask: `{_pad_mask[1].tolist()}`, labels: `{_pad_labels[1].tolist()}`"),
    ])
    return


@app.cell
def _(mo):
    mo.md("""
    ## 5. 작은 데이터로 전체 매개변수 학습하기

    `zero_grad`는 이전 기울기를 비우고, `forward`는 예측과 손실을 계산합니다.
    `loss.backward()`는 각 가중치가 손실에 미치는 기울기를 계산하고,
    `optimizer.step()`은 그 기울기로 가중치를 갱신합니다. 이것이 학습의 한 걸음입니다.
    `train()`/`eval()`은 dropout 등의 동작 모드이고 gradient 계산을 켜고 끄지는 않습니다.
    `no_grad()`는 결과 관찰 때 gradient 기록을 끄는 별도 문맥입니다.
    동일한 네 문장을 반복해 암기시킵니다. 첫 토큰 `<bos>` 뒤에는 동물 네 종류가 가능하므로
    손실이 완전히 0이 될 필요는 없습니다. 학습 횟수를 바꾸고 손실과 생성 문장을 비교하세요.
    """)
    return


@app.cell
def _(mo):
    train_steps = mo.ui.slider(20, 160, step=20, value=100, label="전체 학습 step")
    train_steps
    return (train_steps,)


@app.cell
def _(attention_mask, copy, initial_model, input_ids, labels, torch, train_steps):
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
def _(decode, encode, final_pretrain_loss, initial_model, mo, plt, pretrain_losses, torch, trained_model):
    def complete(model, prompt):
        model.eval()
        ids = torch.tensor([encode(prompt, add_eos=False)])
        with torch.no_grad():
            result = model.generate(
                ids, max_new_tokens=5, do_sample=False, eos_token_id=2, use_cache=True,
                top_k=0, top_p=1.0, temperature=1.0,
            )
        return decode(result[0])

    _fig, _ax = plt.subplots(figsize=(6, 3))
    _ax.plot(pretrain_losses)
    _ax.set(xlabel="Step", ylabel="Cross entropy", title="Full parameter training")
    _fig.tight_layout()
    mo.vstack([
        _fig,
        mo.md(f"같은 배치의 학습 전/후 손실: **{pretrain_losses[0]:.4f} → {final_pretrain_loss:.4f}**"),
        mo.ui.table([
            {"입력": p, "학습 전": complete(initial_model, p), "학습 후": complete(trained_model, p)}
            for p in ["고양이 는", "강아지 는", "소 는", "오리 는"]
        ], selection=None),
    ])
    return (complete,)


@app.cell
def _(mo):
    mo.md(r"""
    ### 5-1. 마지막 벡터 → 어휘 점수 → 확률 → 손실

    **예측:** 학습 후 `고양이 는` 다음의 `야옹` 확률은 커질까요?
    아래의 마지막 hidden state는 모든 블록과 마지막 RMSNorm을 통과한 벡터입니다.
    LM head는 이 벡터와 어휘별 가중치를 내적하여 **logits(확률로 바꾸기 전 점수)**를 만듭니다.
    PyTorch의 Linear 가중치가 `[V,D]`여서 행 벡터 입력에는 `h @ weight.T`를 계산합니다.
    Embedding에서 행을 고르는 것과 출력에서 모든 행과 내적하는 것이 같은 표를 공유합니다.

    softmax는 점수들을 합이 1인 확률로 바꿉니다. 정답 하나의 손실은 $-\log(p_{정답})$입니다.
    정답 확률이 클수록 이 값이 작아집니다. 학습 손실은 패딩을 제외한 모든 예측 위치의 평균입니다.
    """)
    return


@app.cell
def _(F, encode, initial_model, mo, token_to_id, torch, trained_model, vocabulary):
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
    mo.vstack([
        mo.ui.table(probability_rows, selection=None, page_size=28),
        mo.md(f"`야옹` 한 위치의 손실: **{target_losses['학습 전']:.4f} → {target_losses['학습 후']:.4f}**. 전체 배치 평균 손실과는 다른 값입니다."),
        mo.md("**해설:** 문맥을 구분하도록 가중치가 바뀌어 정답 확률이 증가합니다. `<bos>` 뒤에는 동물 네 종류가 모두 정답으로 등장하므로 전체 손실은 완전히 0이 되지 않아도 됩니다."),
    ])
    return probability_rows, target_losses


@app.cell
def _(mo):
    mo.md("""
    ### 5-2. 생성은 한 토큰씩 입력을 늘리는 반복

    **예측:** `야옹`을 선택한 다음에는 어떤 문장을 모델에 다시 넣어야 할까요?
    학습할 때는 정답 문장의 여러 위치를 한 번에 예측하지만, 생성할 때는 다음 정답을 모릅니다.
    마지막 위치의 점수에서 하나를 선택하고 입력에 붙여 다시 계산합니다.
    여기서는 가장 큰 점수를 고르는 greedy 방식입니다. sampling은 확률에 따라 뽑는 다른 방식입니다.
    먼저 캐시 없이 반복을 직접 작성하고, 원본 `generate`의 greedy 결과와 비교합니다.
    """)
    return


@app.cell
def _(complete, decode, encode, mo, torch, trained_model, vocabulary):
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
    mo.vstack([
        mo.ui.table(generation_trace, selection=None),
        mo.md("**관찰·해설:** 선택한 토큰이 다음 입력의 끝에 들어갑니다. `<eos>`를 선택하면 멈춥니다. 원본 생성 함수와 결과가 같습니다."),
    ])
    return (generation_trace,)


@app.cell
def _(F, apply_rotary_pos_emb, attention_probabilities, font_manager, input_ids, math, mo, plt, repeat_kv, torch, trained_model, vocabulary):
    # 학습 전과 동일한 첫 블록·첫 head를 관찰합니다.
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
    _rows = [
        {"query 위치": i, "query": _words[i], "key 위치": j, "key": _words[j],
         "학습 전 확률": float(attention_probabilities[0, 0, i, j]), "학습 후 확률": float(_after[i, j])}
        for i in range(_t) for j in range(i + 1)
    ]
    _font = next((f for f in ["Noto Sans CJK KR", "Noto Sans CJK JP", "Malgun Gothic", "AppleGothic", "NanumGothic"] if f in font_manager.get_font_names()), None)
    _ticks = _words if _font else [str(i) for i in range(_t)]
    _fig, _axes = plt.subplots(1, 2, figsize=(10, 4))
    for _axis, _matrix, _title in zip(_axes, [attention_probabilities[0, 0], _after], ["Before training", "After training"]):
        _image = _axis.imshow(_matrix.numpy(), vmin=0, vmax=1, cmap="Blues")
        _axis.set_xticks(range(_t), labels=_ticks, rotation=45, ha="right", fontsize=8)
        _axis.set_yticks(range(_t), labels=_ticks, fontsize=8)
        if _font:
            for _label in _axis.get_xticklabels() + _axis.get_yticklabels():
                _label.set_fontfamily(_font)
        _axis.set(xlabel="Key", ylabel="Query", title=_title)
    _fig.colorbar(_image, ax=list(_axes), shrink=0.7)
    mo.vstack([
        mo.md("### 5-3. 같은 토큰·같은 head의 Attention 비교\n\n`query`는 정보를 가져올 위치, `key`는 참고할 위치입니다. 학습 전후 수치를 비교하세요. 이 비율 하나가 모델 판단의 원인 전체를 설명하지는 않습니다."),
        _fig,
        mo.ui.table(_rows, selection=None, page_size=14),
        mo.md("한글 폰트가 없으면 그림 축은 위치 번호로 표시합니다. 아래 표에서 실제 토큰과 대응할 수 있습니다."),
    ])
    return


@app.cell
def _(mo):
    mo.md("""
    ## 6. 미래 토큰 가리기와 KV cache 확인

    뒤의 토큰을 바꿔도 앞 위치의 예측은 같아야 합니다. 미래 정보를 볼 수 없기 때문입니다.
    생성 시에는 과거 토큰의 K/V를 저장하는 **KV cache**로 중복 계산을 줄입니다.
    마지막 토큰 하나만 입력해도 전체 문장을 다시 계산한 마지막 출력과 같아야 합니다.
    캐시는 속도를 위한 것이며, 예측의 의미를 바꾸지 않습니다.
    """)
    return


@app.cell
def _(input_ids, mo, torch, trained_model):
    with torch.no_grad():
        _original = trained_model(input_ids).logits
        _changed_ids = input_ids.clone()
        _changed_ids[:, -2] = 0
        _changed = trained_model(_changed_ids).logits
        torch.testing.assert_close(_original[:, :-2], _changed[:, :-2])
        _prefix = trained_model(input_ids[:, :-1], use_cache=True)
        _cached = trained_model(
            input_ids[:, -1:], past_key_values=_prefix.past_key_values, use_cache=True,
        )
        torch.testing.assert_close(_original[:, -1:], _cached.logits, atol=1e-5, rtol=1e-4)
        cache_max_error = (_original[:, -1:] - _cached.logits).abs().max().item()
    mo.md(f"두 검증 통과. 전체 계산과 캐시 계산의 최대 오차: `{cache_max_error:.2e}`. 캐시에 저장되는 K의 shape: `{tuple(_prefix.past_key_values[0][0].shape)}` (GQA 공유 전).")
    return (cache_max_error,)


@app.cell
def _(mo):
    mo.md("""
    ## 7. 직접 바꿔 볼 과제

    1. 샘플 문장에 새 동물을 추가하고 어휘 수와 LM head의 shape 변화를 확인하세요.
    2. `num_key_value_heads`를 4로 바꾸고 K/V의 크기와 캐시 크기를 비교하세요.
    3. `hidden_size`를 64로 바꾸고 head 차원, 매개변수 수와 학습 속도를 비교하세요.
    4. 수동 Attention 계산에서 미래를 가리는 코드를 제거해 보세요. 정답 누출이 왜 생기나요?

    **확인 질문:** `labels=input_ids`인데 왜 자기 자신을 맞히는 학습이 아닌가요?
    Attention과 MLP는 각각 어느 축의 정보를 바꾸나요?

    **기록할 것:** 바꾸기 전 예상 shape, 실행 후 shape, 평균 손실, 동물별 정답 수(0~4)를 적으세요.
    손실 감소와 네 문장 모두 정답은 별도로 판정합니다. 학습 횟수가 적으면 일부만 맞아도 이상하지 않습니다.

    **힌트·해설:** KV head 2→4이면 공유 전 K/V의 head 축은 두 배가 되고, 같은 길이의 캐시 원소 수도 두 배가 됩니다.
    hidden size 32→64에서 Q head가 4라면 head 차원은 8→16입니다.
    causal mask를 없애면 시점 t의 출력이 t+1의 정답 토큰을 입력에서 볼 수 있습니다.
    labels는 원본 forward 안에서 한 칸 이동합니다. Attention은 토큰 사이, MLP는 각 토큰의 특징 차원을 처리합니다.

    이 실습은 Dense 경로를 다룹니다. MoE와 대규모 사전 학습·SFT·RL은 포함하지 않습니다.
    다음 실습 `minimind_lora.py`에서는 원본 가중치를 동결하고 작은 행렬만 학습합니다.
    두 노트북은 각각 단독 실행할 수 있습니다.
    """)
    return


if __name__ == "__main__":
    app.run()
