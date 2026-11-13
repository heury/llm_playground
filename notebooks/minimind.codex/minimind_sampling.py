import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import sys
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    import torch
    import torch.nn.functional as F

    # 파일 위치를 기준으로 찾으므로 저장소 루트와 notebooks 양쪽에서 실행 가능.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "projects" / "minimind"))
    from model.model_minimind import MiniMindConfig, MiniMindForCausalLM

    torch.set_num_threads(2)

    # 그림 축에 한글을 쓰기 위한 폰트. 없으면 None이고, 축에는 번호가 나옵니다.
    korean_font = next(
        (f for f in ["Noto Sans CJK KR", "Noto Sans CJK JP", "Malgun Gothic", "AppleGothic", "NanumGothic"]
         if f in font_manager.get_font_names()),
        None,
    )
    return F, MiniMindConfig, MiniMindForCausalLM, korean_font, mo, plt, torch


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # MiniMind 생성과 샘플링: 다음 토큰은 어떻게 고르나

    **목표:** 원본 `MiniMindForCausalLM.generate`(265–296행)가 다음 토큰을 고르는 과정을 직접 계산합니다.

    모델은 다음 토큰의 **확률 분포**만 줍니다. 그중 무엇을 고를지는 생성 설정이 정합니다.
    설정에 따라 같은 모델의 답변도 달라집니다.

    **원본이 한 step에서 하는 일 (순서가 중요합니다):**

    1. `logits / temperature` — 분포를 뾰족하게 또는 평평하게
    2. `repetition_penalty` — 이미 나온 토큰의 점수 깎기
    3. `top_k` — 점수 상위 k개만 남기기
    4. `top_p` — 누적 확률 p까지만 남기기
    5. `do_sample`이면 확률대로 뽑기(`multinomial`), 아니면 최댓값(`argmax`)
    6. `<eos>`가 나온 문장은 끝난 것으로 표시하기

    **사전 지식:** `minimind_structure.py`의 logits·softmax·greedy 생성(5-1, 5-2절).

    외부 모델·데이터·GPU 없이 CPU로 실행합니다. 슬라이더를 바꾸면 해당 절만 다시 계산합니다.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 1. 확률이 고르지 않은 작은 모델 만들기

    샘플링을 보려면 다음 토큰 후보가 여럿이어야 합니다.
    "오늘 날씨 는" 뒤에 네 가지 날씨가 **서로 다른 빈도**로 나오는 문장 8개로 학습합니다.

    | 다음 단어 | 문장 수 | 데이터 비율 |
    | --- | --- | --- |
    | 맑음 | 4 | 0.50 |
    | 흐림 | 2 | 0.25 |
    | 비 | 1 | 0.125 |
    | 눈 | 1 | 0.125 |

    잘 학습되면 모델의 확률도 이 비율에 가까워집니다. 문장 길이가 달라 패딩도 들어갑니다.
    """)
    return


@app.cell
def _(MiniMindConfig, MiniMindForCausalLM, torch):
    corpus = (
        ["오늘 날씨 는 맑음 이다 ."] * 4
        + ["오늘 날씨 는 흐림 이다 ."] * 2
        + ["오늘 날씨 는 비 가 온다 .", "오늘 날씨 는 눈 이 온다 ."]
    )
    vocabulary = ["<pad>", "<bos>", "<eos>"] + sorted({w for s in corpus for w in s.split()})
    token_to_id = {w: i for i, w in enumerate(vocabulary)}

    def encode(text, add_eos=True):
        ids = [token_to_id["<bos>"]] + [token_to_id[w] for w in text.split()]
        return ids + [token_to_id["<eos>"]] if add_eos else ids

    def decode(ids):
        return " ".join(vocabulary[int(i)] for i in ids)

    _seqs = [encode(s) for s in corpus]
    _len = max(map(len, _seqs))
    _ids = torch.tensor([s + [token_to_id["<pad>"]] * (_len - len(s)) for s in _seqs])
    _mask = (_ids != token_to_id["<pad>"]).long()
    _labels = _ids.masked_fill(_mask == 0, -100)

    config = MiniMindConfig(
        hidden_size=32, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2,
        intermediate_size=64, max_position_embeddings=64,
        vocab_size=len(vocabulary), dropout=0.0, flash_attn=False,
        bos_token_id=token_to_id["<bos>"], eos_token_id=token_to_id["<eos>"],
    )
    # 시드를 이 블록 안에서만 고정합니다. 다시 실행해도 같은 모델이 나옵니다.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        model = MiniMindForCausalLM(config)
        _optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.0)
        model.train()
        for _ in range(200):
            _optimizer.zero_grad(set_to_none=True)
            _loss = model(_ids, attention_mask=_mask, labels=_labels).loss
            _loss.backward()
            _optimizer.step()
    model.eval()
    final_loss = _loss.item()
    return decode, encode, final_loss, model, token_to_id, vocabulary


@app.cell
def _(F, encode, model, token_to_id, torch):
    prompt = "오늘 날씨 는"
    prompt_ids = torch.tensor([encode(prompt, add_eos=False)])     # [1, T]
    with torch.no_grad():
        # generate와 같은 모양 [1, V]로 둡니다. 마지막 위치의 점수만 씁니다.
        base_logits = model(prompt_ids).logits[:, -1, :]
    base_probs = F.softmax(base_logits, dim=-1)
    weather = ["맑음", "흐림", "비", "눈"]
    weather_ids = [token_to_id[w] for w in weather]
    assert base_probs[0, weather_ids].sum() > 0.95, "학습이 부족합니다. 날씨 후보에 확률이 모여야 합니다."
    return base_logits, base_probs, prompt, prompt_ids, weather, weather_ids


@app.cell(hide_code=True)
def _(base_logits, base_probs, final_loss, mo, prompt, weather, weather_ids):
    # ▸ 결과 표시 — 실행하면 표·그림이 나타납니다 (marimo 표시 코드)
    _data_ratio = {"맑음": 0.5, "흐림": 0.25, "비": 0.125, "눈": 0.125}
    mo.vstack([
        mo.md(f"학습 후 손실: `{final_loss:.4f}`. 입력 `{prompt}` 다음 토큰:"),
        mo.ui.table([
            {"후보": w, "logit": float(base_logits[0, i]), "모델 확률": float(base_probs[0, i]), "데이터 비율": _data_ratio[w]}
            for w, i in zip(weather, weather_ids)
        ] + [{"후보": "그 밖의 토큰 합", "logit": None,
              "모델 확률": float(1 - base_probs[0, weather_ids].sum()), "데이터 비율": 0.0}],
            selection=None),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 2. greedy와 sampling

    - **greedy (`do_sample=False`):** 매번 확률이 가장 큰 토큰을 고릅니다. 같은 입력이면 늘 같은 답입니다.
    - **sampling (`do_sample=True`):** 확률대로 뽑습니다(`torch.multinomial`). 확률 0.25인 토큰은 대략 네 번에 한 번 나옵니다.

    **예측:** 400번 뽑으면 `맑음`은 몇 번쯤 나올까요?

    아래에서는 원본 `generate`로 다음 토큰 하나를 400번 뽑습니다.
    `num_return_sequences=400`은 같은 입력을 400줄로 복사해 한 번에 뽑는 옵션입니다.
    `top_k=0, top_p=1.0, temperature=1.0`은 필터를 모두 끈 상태입니다(7절에서 이유를 봅니다).
    """)
    return


@app.cell
def _(model, prompt_ids, token_to_id, torch, vocabulary):
    _off = dict(top_k=0, top_p=1.0, temperature=1.0, eos_token_id=token_to_id["<eos>"])
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        _sampled = model.generate(prompt_ids, max_new_tokens=1, do_sample=True, num_return_sequences=400, **_off)
    _picked = _sampled[:, -1]                                             # 새로 뽑은 토큰 [400]
    sample_counts = {vocabulary[i]: int((_picked == i).sum()) for i in _picked.unique().tolist()}
    _greedy = model.generate(prompt_ids, max_new_tokens=1, do_sample=False, **_off)
    greedy_token = vocabulary[int(_greedy[0, -1])]
    return greedy_token, sample_counts


@app.cell(hide_code=True)
def _(base_probs, greedy_token, mo, sample_counts, token_to_id):
    # ▸ 결과 표시 — 실행하면 표·그림이 나타납니다 (marimo 표시 코드)
    mo.vstack([
        mo.md(f"**greedy 결과:** 항상 `{greedy_token}`"),
        mo.ui.table([
            {"뽑힌 토큰": w, "400번 중 횟수": c, "뽑힌 비율": c / 400, "모델 확률": float(base_probs[0, token_to_id[w]])}
            for w, c in sorted(sample_counts.items(), key=lambda kv: -kv[1])
        ], selection=None),
        mo.md("**관찰:** 뽑힌 비율이 모델 확률에 가깝습니다. 횟수가 적을수록 차이가 커집니다. greedy는 확률이 낮은 후보를 절대 고르지 않습니다."),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. temperature: 분포를 뾰족하게, 평평하게

    softmax 전에 logits를 온도 $T$로 나눕니다: $p_i = \mathrm{softmax}(z_i / T)$.

    - $T < 1$: 점수 차이가 커집니다. 1등에 확률이 몰립니다. $T \to 0$이면 greedy와 같습니다.
    - $T = 1$: 모델 분포 그대로입니다.
    - $T > 1$: 점수 차이가 줄어듭니다. 분포가 평평해져 낮은 후보도 자주 나옵니다.
    - 순위는 바뀌지 않습니다. 그래서 greedy 결과는 $T$와 무관합니다.

    원본 기본값은 `temperature=0.85`입니다. 약간 뾰족하게 만듭니다.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    # ▸ 실습 설정 위젯 — 실행하면 슬라이더가 나타납니다 (marimo 표시 코드)
    temperature_slider = mo.ui.slider(0.1, 3.0, step=0.1, value=0.5, label="temperature")
    temperature_slider
    return (temperature_slider,)


@app.cell
def _(F, base_logits, temperature_slider):
    def apply_temperature(logits, temperature):
        # 원본 275행: logits = outputs.logits[:, -1, :] / temperature
        return logits / temperature

    temperature_probs = F.softmax(apply_temperature(base_logits, temperature_slider.value), dim=-1)
    # 순위는 그대로이므로 1등 토큰은 바뀌지 않습니다.
    assert temperature_probs.argmax() == base_logits.argmax()
    return apply_temperature, temperature_probs


@app.cell(hide_code=True)
def _(
    base_probs,
    korean_font,
    mo,
    plt,
    temperature_probs,
    temperature_slider,
    weather,
    weather_ids,
):
    # ▸ 결과 표시 — 실행하면 표·그림이 나타납니다 (marimo 표시 코드)
    _t = temperature_slider.value
    _x = range(len(weather))
    temperature_fig, _ax = plt.subplots(figsize=(6, 3))
    _ax.bar([i - 0.2 for i in _x], base_probs[0, weather_ids].tolist(), width=0.4, label="T = 1.0")
    _ax.bar([i + 0.2 for i in _x], temperature_probs[0, weather_ids].tolist(), width=0.4, label=f"T = {_t:.1f}")
    _ax.set_xticks(list(_x), labels=weather if korean_font else [str(i) for i in _x])
    if korean_font:
        for _label in _ax.get_xticklabels():
            _label.set_fontfamily(korean_font)
    _ax.set(ylabel="Probability", ylim=(0, 1), title="Temperature reshapes the distribution")
    _ax.legend()
    temperature_fig.tight_layout()
    mo.vstack([
        temperature_fig,
        mo.ui.table([
            {"후보": w, "T = 1.0": float(base_probs[0, i]), f"T = {_t:.1f}": float(temperature_probs[0, i])}
            for w, i in zip(weather, weather_ids)
        ], selection=None),
        mo.md("슬라이더를 0.1과 3.0으로 옮겨 보세요. 한글 폰트가 없으면 그림 축에 번호(0=맑음, 1=흐림, 2=비, 3=눈)가 나옵니다."),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 4. top-k: 상위 k개만 남기기

    k번째 점수보다 **작은** 점수는 모두 `-inf`로 바꿉니다. softmax 뒤 확률은 0입니다.
    남은 토큰끼리 확률을 다시 나눠 가집니다(합이 1이 되도록).

    - 확률이 아주 낮은 엉뚱한 토큰을 원천 차단합니다.
    - k는 고정입니다. 적절한 후보 수가 달라도 같은 k를 씁니다.
    - `top_k=0`이면 끕니다. 원본 기본값은 `top_k=50`입니다.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    # ▸ 실습 설정 위젯 — 실행하면 슬라이더가 나타납니다 (marimo 표시 코드)
    top_k_slider = mo.ui.slider(0, 6, step=1, value=2, label="top_k (0 = 끄기)")
    top_k_slider
    return (top_k_slider,)


@app.cell
def _(F, base_logits, torch, top_k_slider):
    def top_k_filter(logits, k):
        # 원본 279–280행. k번째 점수보다 작은 칸을 -inf로 바꿉니다.
        if k <= 0:
            return logits
        kth_score = torch.topk(logits, k)[0][..., -1, None]     # 행마다 k번째로 큰 점수
        return logits.masked_fill(logits < kth_score, -torch.inf)

    top_k_probs = F.softmax(top_k_filter(base_logits, top_k_slider.value), dim=-1)
    if top_k_slider.value > 0:
        assert int((top_k_probs > 0).sum()) == top_k_slider.value
    return top_k_filter, top_k_probs


@app.cell(hide_code=True)
def _(base_probs, mo, top_k_probs, top_k_slider, torch, vocabulary):
    # ▸ 결과 표시 — 실행하면 표·그림이 나타납니다 (marimo 표시 코드)
    _order = torch.argsort(base_probs[0], descending=True)[:6]
    mo.vstack([
        mo.ui.table([
            {"순위": r + 1, "토큰": vocabulary[int(i)], "원래 확률": float(base_probs[0, i]),
             f"top_k={top_k_slider.value} 뒤 확률": float(top_k_probs[0, i])}
            for r, i in enumerate(_order)
        ], selection=None),
        mo.md(f"남은 토큰 수: **{int((top_k_probs > 0).sum())}개**. 남은 확률의 합은 다시 1입니다."),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. top-p (nucleus): 누적 확률 p까지만 남기기

    확률이 큰 순서로 정렬하고 누적합을 구합니다. 누적합이 p에 닿을 때까지의 토큰만 남깁니다.

    - **남는 개수는 상황에 따라 달라집니다.** 1등 확률만으로 p를 넘으면 1개, 아니면 여러 개를 남깁니다.
    - 원본 기본값은 `top_p=0.85`입니다. `1.0`이면 끕니다.

    **헷갈리기 쉬운 한 줄:** 원본은 "누적합 > p"인 칸을 지울 표시를 만든 뒤 **한 칸 뒤로 밉니다.**

    ```python
    mask = cumsum > top_p
    mask[..., 1:], mask[..., 0] = mask[..., :-1].clone(), 0
    ```

    그래서 **누적합이 처음 p를 넘기는 토큰까지 남깁니다.** 1등은 언제나 남습니다.
    예: 확률 0.5, 0.25, …이고 p = 0.6이면, 누적 0.5(남김) → 0.75(p를 넘기는 첫 토큰, 남김) → 나머지 지움.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    # ▸ 실습 설정 위젯 — 실행하면 슬라이더가 나타납니다 (marimo 표시 코드)
    top_p_slider = mo.ui.slider(0.05, 1.0, step=0.05, value=0.6, label="top_p (1.0 = 끄기)")
    top_p_slider
    return (top_p_slider,)


@app.cell
def _(F, base_logits, top_p_slider, torch):
    def top_p_filter(logits, p):
        # 원본 281–285행.
        if p >= 1.0:
            return logits
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)        # 큰 순서로 정렬
        cumulative = F.softmax(sorted_logits, dim=-1).cumsum(dim=-1)            # 누적 확률
        remove = cumulative > p                                                  # p를 넘긴 칸
        remove[..., 1:], remove[..., 0] = remove[..., :-1].clone(), False        # 한 칸 밀기
        remove = remove.scatter(-1, sorted_idx, remove)                          # 원래 순서로 되돌리기
        return logits.masked_fill(remove, -torch.inf)

    top_p_probs = F.softmax(top_p_filter(base_logits, top_p_slider.value), dim=-1)
    assert top_p_probs.argmax() == base_logits.argmax()   # 1등은 언제나 남습니다
    return top_p_filter, top_p_probs


@app.cell(hide_code=True)
def _(base_probs, mo, top_p_probs, top_p_slider, torch, vocabulary):
    # ▸ 결과 표시 — 실행하면 표·그림이 나타납니다 (marimo 표시 코드)
    _order = torch.argsort(base_probs[0], descending=True)[:6]
    _cum = base_probs[0, _order].cumsum(0)
    mo.vstack([
        mo.ui.table([
            {"순위": r + 1, "토큰": vocabulary[int(i)], "원래 확률": float(base_probs[0, i]),
             "누적 확률": float(_cum[r]), "남김": bool(top_p_probs[0, i] > 0),
             f"top_p={top_p_slider.value:.2f} 뒤 확률": float(top_p_probs[0, i])}
            for r, i in enumerate(_order)
        ], selection=None),
        mo.md(f"남은 토큰 수: **{int((top_p_probs > 0).sum())}개**. '남김' 열에서 누적 확률이 처음 p를 넘긴 행도 남는지 확인하세요."),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. repetition penalty: 이미 나온 토큰 깎기

    입력(프롬프트 + 지금까지 생성한 토큰)에 **한 번이라도** 나온 토큰의 점수를 깎습니다.

    - 점수가 양수면 `점수 / penalty`, 음수면 `점수 × penalty`입니다.
    - 음수를 나누면 0에 가까워져 오히려 **커집니다.** 그래서 부호에 따라 다르게 처리합니다. 둘 다 점수를 낮춥니다.
    - 몇 번 나왔는지는 세지 않습니다. 한 번이든 열 번이든 같게 깎습니다.
    - 프롬프트 토큰도 포함됩니다. 원본 기본값은 `1.0`(끄기)입니다.

    아래는 이해를 돕는 가상의 점수 4개입니다. `A`와 `C`가 이미 나왔다고 칩니다.
    """)
    return


@app.cell
def _(torch):
    def apply_repetition_penalty(logits, seen_ids, penalty):
        # 원본 276–278행 (행 하나 기준). seen_ids: 지금까지 입력에 나온 토큰 ID
        if penalty == 1.0:
            return logits
        logits = logits.clone()
        seen = torch.unique(seen_ids)
        score = logits[..., seen]
        logits[..., seen] = torch.where(score > 0, score / penalty, score * penalty)
        return logits

    _demo_logits = torch.tensor([[2.0, 1.0, -1.0, 0.5]])     # 토큰 A, B, C, D
    _seen = torch.tensor([0, 2, 0])                            # A가 두 번, C가 한 번 나옴
    _penalized = apply_repetition_penalty(_demo_logits, _seen, 1.5)
    repetition_rows = [
        {"토큰": n, "이미 나옴": n in ("A", "C"), "원래 점수": float(_demo_logits[0, i]), "penalty 1.5 뒤": float(_penalized[0, i])}
        for i, n in enumerate("ABCD")
    ]
    assert _penalized[0, 0] < _demo_logits[0, 0] and _penalized[0, 2] < _demo_logits[0, 2]
    return apply_repetition_penalty, repetition_rows


@app.cell(hide_code=True)
def _(mo, repetition_rows):
    # ▸ 결과 표시 — 실행하면 표·그림이 나타납니다 (marimo 표시 코드)
    mo.vstack([
        mo.ui.table(repetition_rows, selection=None),
        mo.md("**관찰:** A(양수)는 2.0 → 1.33, C(음수)는 −1.0 → −1.5. 둘 다 낮아집니다. A는 두 번 나왔지만 한 번만 깎입니다."),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 7. 원본 `generate`를 그대로 재현하기

    지금까지 만든 함수를 원본과 **같은 순서**로 이어 붙입니다.
    같은 난수 시드에서 원본 `generate`와 같은 문장이 나오는지 확인합니다.

    **함정: greedy여도 필터가 먼저 돕니다.** 원본은 `do_sample=False`여도 temperature·top-k·top-p를 먼저 적용합니다.

    - temperature와 top-p는 1등을 바꾸지 않습니다. greedy 결과에 영향이 없습니다.
    - 기본 `top_k=50`이 어휘 수보다 크면 `torch.topk`가 **오류**를 냅니다. 이 실습의 어휘는 15개입니다.
    - 실제 MiniMind는 어휘가 6400개라 이 오류가 없습니다. 이 작은 모델에서는 `top_k=0`을 쓰세요.
    """)
    return


@app.cell
def _(
    F,
    apply_repetition_penalty,
    apply_temperature,
    model,
    prompt_ids,
    token_to_id,
    top_k_filter,
    top_p_filter,
    torch,
):
    def choose_next_token(logits, seen_ids, temperature=1.0, top_k=0, top_p=1.0,
                          repetition_penalty=1.0, do_sample=True):
        logits = apply_temperature(logits, temperature)                      # 1
        logits = apply_repetition_penalty(logits, seen_ids, repetition_penalty)  # 2
        logits = top_k_filter(logits, top_k)                                  # 3
        logits = top_p_filter(logits, top_p)                                  # 4
        if do_sample:                                                         # 5
            return torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
        return logits.argmax(dim=-1, keepdim=True)

    def manual_generate(ids, max_new_tokens, **options):
        eos = token_to_id["<eos>"]
        for _ in range(max_new_tokens):
            logits = model(ids).logits[:, -1, :]                              # [1, V]
            next_token = choose_next_token(logits, ids[0], **options)        # [1, 1]
            ids = torch.cat([ids, next_token], dim=-1)
            if int(next_token) == eos:                                        # 6
                break
        return ids

    generation_settings = [
        dict(temperature=1.0, top_k=0, top_p=1.0, repetition_penalty=1.0),
        dict(temperature=0.7, top_k=3, top_p=1.0, repetition_penalty=1.0),
        dict(temperature=1.5, top_k=0, top_p=0.8, repetition_penalty=1.0),
        dict(temperature=1.0, top_k=0, top_p=1.0, repetition_penalty=2.0),
    ]
    # 직접 구현으로 설정 4개 × seed 3개를 생성합니다.
    manual_generations = {}
    with torch.no_grad():
        for _i, _options in enumerate(generation_settings):
            for _seed in range(3):
                with torch.random.fork_rng(devices=[]):
                    torch.manual_seed(_seed)
                    manual_generations[_i, _seed] = manual_generate(prompt_ids, 5, **_options)
    return choose_next_token, generation_settings, manual_generations


@app.cell
def _(generation_settings, manual_generations, model, prompt_ids, token_to_id, torch):
    # 원본 generate로 같은 설정·seed를 생성해 직접 구현과 비교합니다.
    reproduce_rows = []
    with torch.no_grad():
        for _i, _options in enumerate(generation_settings):
            for _seed in range(3):
                with torch.random.fork_rng(devices=[]):
                    torch.manual_seed(_seed)
                    _original = model.generate(prompt_ids, max_new_tokens=5, do_sample=True,
                                               eos_token_id=token_to_id["<eos>"], **_options)
                _manual = manual_generations[_i, _seed]
                assert torch.equal(_original, _manual), (_options, _seed)
                reproduce_rows.append({**_options, "seed": _seed, "original": _original, "manual": _manual})

    # 함정 확인: greedy + 기본 top_k=50 → 어휘 수보다 커서 오류
    try:
        model.generate(prompt_ids, max_new_tokens=1, do_sample=False)
        greedy_default_error = None
    except RuntimeError as error:
        greedy_default_error = str(error)
    assert greedy_default_error is not None
    return greedy_default_error, reproduce_rows


@app.cell(hide_code=True)
def _(decode, greedy_default_error, mo, reproduce_rows, vocabulary):
    # ▸ 결과 표시 — 실행하면 표·그림이 나타납니다 (marimo 표시 코드)
    mo.vstack([
        mo.ui.table([
            {"temperature": r["temperature"], "top_k": r["top_k"], "top_p": r["top_p"],
             "rep. penalty": r["repetition_penalty"], "seed": r["seed"],
             "원본 generate": decode(r["original"][0]), "직접 구현": decode(r["manual"][0]),
             "일치": decode(r["original"][0]) == decode(r["manual"][0])}
            for r in reproduce_rows
        ], selection=None, page_size=12),
        mo.md(f"**함정:** 어휘 {len(vocabulary)}개에서 `generate(do_sample=False)`를 기본값으로 부르면: `{greedy_default_error}`"),
        mo.md("**관찰:** 설정이 같아도 시드가 다르면 문장이 다릅니다. 같은 시드에서는 원본과 직접 구현이 완전히 같습니다. 온도가 높거나 penalty가 크면 학습 데이터에 없던 문장이 나올 수도 있습니다."),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 8. 여러 문장을 함께 생성할 때: 끝난 문장 처리

    배치로 생성하면 문장마다 끝나는 시점이 다릅니다. 원본은 `finished` 표시로 이를 처리합니다(287, 292–293행).

    - `<eos>`를 뽑은 문장은 `finished = True`가 됩니다.
    - 끝난 문장에는 길이를 맞추려고 계속 `<eos>`를 붙입니다.
    - **모든** 문장이 끝나야 반복을 멈춥니다.

    이 데이터에서 `맑음/흐림` 문장은 `비/눈` 문장보다 한 토큰 짧습니다. 그래서 먼저 끝납니다.
    """)
    return


@app.cell
def _(model, prompt_ids, token_to_id, torch):
    _eos = token_to_id["<eos>"]
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(1)
        batch_output = model.generate(
            prompt_ids, max_new_tokens=6, do_sample=True, num_return_sequences=8,
            top_k=0, top_p=1.0, temperature=1.0, eos_token_id=_eos,
        )
    # 한 번 <eos>가 나오면 그 뒤는 모두 <eos>입니다.
    for _row in batch_output:
        _first = (_row == _eos).nonzero()
        if len(_first):
            assert (_row[int(_first[0]):] == _eos).all()
    return (batch_output,)


@app.cell(hide_code=True)
def _(batch_output, decode, mo, prompt_ids, token_to_id):
    # ▸ 결과 표시 — 실행하면 표·그림이 나타납니다 (marimo 표시 코드)
    _eos = token_to_id["<eos>"]
    _start = prompt_ids.shape[1]
    mo.vstack([
        mo.ui.table([
            {"문장": n, "생성 부분": decode(row[_start:]),
             "끝난 뒤 덧붙은 <eos> 수": max(int((row[_start:] == _eos).sum()) - 1, 0)}
            for n, row in enumerate(batch_output)
        ], selection=None),
        mo.md("**관찰:** 짧은 문장(`맑음/흐림`) 뒤에는 `<eos>`가 더 붙습니다. 가장 긴 문장이 끝날 때까지 기다린 흔적입니다."),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 9. 직접 바꿔 볼 과제

    1. 3절 슬라이더를 0.1과 3.0으로 옮기세요. 각 경우 2절처럼 400번 뽑으면 `맑음` 비율이 어떻게 될지 예측하세요.
    2. 5절에서 `top_p=0.4`와 `0.8`일 때 남는 토큰 수를 예측하고 확인하세요.
    3. `top_k=2`와 `top_p=0.6`의 결과를 비교하세요. 이 분포에서는 같나요? 1등 확률이 0.95라면요?
    4. `repetition_penalty`를 1보다 작게(예: 0.5) 하면 어떤 일이 생길까요?
    5. `choose_next_token`에서 temperature를 top-p **뒤로** 옮기면 결과가 바뀔까요?

    먼저 직접 답을 적은 뒤, 아래 해설을 펼쳐 비교하세요.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({
        "힌트·해설: 과제 1~5": mo.md("""
    1. **T = 0.1:** 거의 400번 모두 `맑음`입니다. greedy와 비슷합니다. **T = 3.0:** 날씨 넷이 비슷해지고, 학습에 없던 토큰도 가끔 나옵니다.
    2. 누적 확률은 약 0.50 → 0.75 → 0.87 → 1.0입니다. `p = 0.4`: 1등이 바로 p를 넘기므로 **1개**. `p = 0.8`: 3등(누적 0.87)이 p를 처음 넘기므로 **3개**. 누적이 p와 거의 같은 경계(예: 0.5)는 학습 결과의 작은 차이에도 바뀝니다.
    3. 이 분포에서는 둘 다 `맑음·흐림` 2개를 남깁니다. 1등이 0.95라면 top-p(0.6)는 1개만, top-k(2)는 여전히 2개를 남깁니다. top-p가 상황에 따라 개수를 바꾸는 이유입니다.
    4. 깎는 대신 **올립니다.** 같은 말을 반복하는 경향이 커집니다.
    5. **바뀝니다.** top-p의 누적 확률은 temperature를 적용한 분포로 계산합니다. 순서를 바꾸면 남는 후보가 달라집니다. 원본 순서는 temperature → penalty → top-k → top-p입니다.
    """),
    })
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    **정리:**

    - 모델은 분포만 줍니다. 무엇을 고를지는 생성 설정이 정합니다.
    - temperature는 분포의 모양을, top-k·top-p는 후보의 범위를, repetition penalty는 반복을 조절합니다.
    - 같은 시드·같은 설정이면 결과가 같습니다. 실험을 기록할 때는 시드와 설정을 함께 남기세요.

    **다음 실습:** 학습 파이프라인 `minimind_pretraining.py`.
    PPO·GRPO 노트북에서는 여기서 본 sampling으로 응답을 여러 개 뽑아 보상을 비교합니다.
    """)
    return


if __name__ == "__main__":
    app.run()
