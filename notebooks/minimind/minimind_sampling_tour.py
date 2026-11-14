import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import sys as _display_sys
    from pathlib import Path as _DisplayPath

    _display_dir = str(_DisplayPath(__file__).resolve().parent)
    if _display_dir not in _display_sys.path:
        _display_sys.path.insert(0, _display_dir)
    from notebook_display import result_table

    return (result_table,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **실행 안내:** 저장소 루트에서 `cd notebooks/minimind` → `uv sync --locked` →
    `uv run marimo edit minimind_sampling_tour.py`. 환경과 권장 순서는 [공통 README](README.md)를 보세요.
    슬라이더를 바꾸면 관련 절만 다시 계산합니다. 학습은 하지 않습니다.
    """)
    return


@app.cell
def _():
    import inspect
    import json
    import math
    import sys
    from collections import Counter
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import torch
    import torch.nn.functional as F
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file
    from transformers import AutoTokenizer

    # 노트북 위치(notebooks/minimind/)에서 두 단계 위가 저장소 루트입니다.
    repo_root = Path(__file__).resolve().parents[2]
    minimind_dir = repo_root / "projects/minimind"
    sys.path.insert(0, str(minimind_dir))

    # 원본 파일의 클래스를 그대로 가져옵니다.
    from model.model_minimind import MiniMindConfig, MiniMindForCausalLM

    # 이 실습은 GPU에서 실행합니다. GPU가 없으면 여기서 멈춥니다.
    mo.stop(
        not torch.cuda.is_available(),
        mo.callout(mo.md("**CUDA GPU를 찾지 못했습니다.** 이 노트북은 GPU를 사용합니다. "
                         "`nvidia-smi`와 `torch.cuda.is_available()`을 확인하세요."), kind="danger"),
    )
    device = "cuda"
    return (
        AutoTokenizer,
        Counter,
        F,
        MiniMindConfig,
        MiniMindForCausalLM,
        Path,
        device,
        inspect,
        json,
        load_file,
        math,
        mo,
        plt,
        snapshot_download,
        torch,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.vstack([
        mo.md(r"""
    # MiniMind 샘플링: 다음 토큰은 어떻게 고르나

    모델은 다음 토큰의 **확률 분포**만 계산합니다. 어떤 토큰을 고를지는 **디코딩 설정**이 정합니다.
    같은 모델이라도 설정에 따라 답이 반복되거나, 다양해지거나, 엉뚱해집니다.

    64M 대화 모델 [`caspar/MiniMind-Dense-64M-Mixed-SFT-v1.0`](https://huggingface.co/caspar/MiniMind-Dense-64M-Mixed-SFT-v1.0)을
    **GPU**에 올립니다. 원본 `MiniMindForCausalLM.generate`의 각 단계를 직접 계산하고,
    직접 만든 루프가 원본과 **같은 토큰**을 내는지 확인합니다.

    **준비:** NVIDIA GPU(VRAM 1GB 미만). 처음 실행 시 가중치·토크나이저(약 130MB)를 내려받습니다. 전체 실행은 수십 초입니다.

    **사전 지식:** logits·softmax, 자기회귀 생성(`minimind_model_tour.py`), 기본 확률.

    | 절 | 내용 | 원본 `generate`의 위치 |
    | --- | --- | --- |
    | 1 | 모델·토크나이저·프롬프트 | — |
    | 2 | 다음 토큰 분포와 엔트로피 | `self.forward(...)`, `logits[:, -1, :]` |
    | 3 | Greedy: 항상 최댓값 | `torch.argmax` (`do_sample=False`) |
    | 4 | Temperature | `logits / temperature` |
    | 5 | Top-k | `torch.topk` 아래를 `-inf` |
    | 6 | Top-p (nucleus) | 누적 확률 마스크 |
    | 7 | Repetition penalty | `seen` 토큰 점수 깎기 |
    | 8 | 한 step 재구현과 원본 대조 | `torch.multinomial` |
    | 9 | 설정별 다양성·품질 비교 | `num_return_sequences` |
    | 10 | 직접 해보기 | — |

    원본 `generate`의 한 step 처리 순서입니다. 순서에 따라 결과가 달라집니다.
    """),
        mo.mermaid(r"""
    flowchart LR
      L["logits [B, V]<br/>마지막 위치"] --> T["÷ temperature"]
      T --> R["repetition_penalty<br/>본 토큰 깎기"]
      R --> K["top_k<br/>상위 k개만"]
      K --> P["top_p<br/>누적 확률 p까지"]
      P --> S{"do_sample?"}
      S -- "True" --> M["softmax → multinomial"]
      S -- "False" --> A["argmax"]
      M --> E["eos 처리 → 다음 step"]
      A --> E
    """),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. 모델·토크나이저·프롬프트

    ### 1-1. 베이스 모델 불러오기

    `v1.0` 스냅샷에서 설정·가중치·토크나이저만 받습니다. 모델 클래스는 저장소의
    `projects/minimind/model/model_minimind.py`를 씁니다(`trust_remote_code`는 쓰지 않음).
    정확한 비교를 위해 fp16 가중치를 **fp32**로 바꿔 GPU에 올립니다.
    """)
    return


@app.cell
def _(
    AutoTokenizer,
    MiniMindConfig,
    MiniMindForCausalLM,
    Path,
    device,
    json,
    load_file,
    snapshot_download,
    torch,
):
    MODEL_REPO = "caspar/MiniMind-Dense-64M-Mixed-SFT-v1.0"
    MODEL_REVISION = "v1.0"

    model_dir = Path(snapshot_download(MODEL_REPO, revision=MODEL_REVISION,
                                       allow_patterns=["*.json", "*.jinja", "model.safetensors"]))
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model_config = MiniMindConfig(**json.load(open(model_dir / "config.json")))
    model = MiniMindForCausalLM(model_config)
    _state = {k: v.float() for k, v in load_file(model_dir / "model.safetensors").items()}
    _missing, _unexpected = model.load_state_dict(_state, strict=False)
    assert _missing == ["lm_head.weight"] and not _unexpected       # lm_head는 Embedding과 가중치 공유
    model.to(device).eval()
    EOS_ID = tokenizer.eos_token_id                                  # <|im_end|> = 2
    torch.backends.cuda.matmul.allow_tf32 = False                    # 재현성을 위해 fp32 그대로 계산
    return EOS_ID, MODEL_REPO, MODEL_REVISION, model, model_config, tokenizer


@app.cell(hide_code=True)
def _(EOS_ID, MODEL_REPO, MODEL_REVISION, mo, model, model_config, tokenizer, torch):
    # ▸ 결과 표시
    _n_params = sum(p.numel() for p in model.parameters())
    mo.md(f"""
    | 항목 | 값 |
    | --- | --- |
    | 모델 | `{MODEL_REPO}` @ `{MODEL_REVISION}` |
    | 매개변수 | {_n_params / 1e6:.1f}M (lm_head는 Embedding과 공유) |
    | 층 / hidden / heads (KV) | {model_config.num_hidden_layers} / {model_config.hidden_size} / {model_config.num_attention_heads} ({model_config.num_key_value_heads}) |
    | 어휘 크기 V | {model_config.vocab_size} |
    | eos | `{tokenizer.eos_token}` (id {EOS_ID}) |
    | 장치 | `{next(model.parameters()).device}` · {torch.cuda.get_device_name(0)} · dtype `{next(model.parameters()).dtype}` |
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 1-2. 대화 프롬프트와 토큰 표시

    질문을 학습 때와 같은 대화 템플릿으로 감쌉니다. 모델 카드 안내대로 빈 `<think>` 블록은 지웁니다.
    생성은 `assistant\n` 다음부터 시작합니다.

    토큰의 공백은 `␣`, 줄바꿈은 `↵`로 표시합니다. 어휘(V=6400)가 작아 단어가 자주 조각납니다(예: `F`, `P`).
    """)
    return


@app.cell
def _(device, tokenizer, torch):
    def chat_ids(question):
        """질문 → 대화 템플릿 토큰 [1, T] (GPU)."""
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": question}], tokenize=False,
            add_generation_prompt=True, open_thinking=False,
        ).replace("<think>\n\n</think>\n\n", "")
        return torch.tensor([tokenizer(text, add_special_tokens=False).input_ids], device=device)

    def show_tok(token_id):
        """토큰 하나를 공백·줄바꿈이 보이게 문자열로."""
        return tokenizer.decode([int(token_id)]).replace(" ", "␣").replace("\n", "↵")

    def decode_answer(ids):
        """생성된 부분만 받아 첫 eos 앞까지 문자열로."""
        ids = [int(i) for i in ids]
        if tokenizer.eos_token_id in ids:
            ids = ids[: ids.index(tokenizer.eos_token_id)]
        return tokenizer.decode(ids, skip_special_tokens=True).strip()

    PROMPTS = {
        "확신이 큰 질문: What is the capital of France?": "What is the capital of France?",
        "열린 질문: Give me a tip for staying healthy.": "Give me a tip for staying healthy.",
        "창작: Tell me a story about a cat.": "Tell me a story about a cat.",
    }
    return PROMPTS, chat_ids, decode_answer, show_tok


@app.cell
def _(PROMPTS, chat_ids, mo, show_tok):
    _ids = chat_ids(PROMPTS["확신이 큰 질문: What is the capital of France?"])[0].tolist()
    mo.md("프롬프트 토큰 (`" + "` `".join(show_tok(i) for i in _ids) + f"`) — 총 {len(_ids)}개")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. 다음 토큰 분포

    `forward`는 모든 위치의 logits `[B, T, V]`를 냅니다. 생성에는 **마지막 위치** `[:, -1, :]`만 씁니다.
    softmax를 거치면 6400개 토큰의 확률이 됩니다.

    $$
    p_i = \frac{e^{z_i}}{\sum_{j=1}^{V} e^{z_j}}, \qquad
    H(p) = -\sum_i p_i \log p_i, \qquad
    \text{유효 후보 수} = e^{H}
    $$

    **엔트로피** $H$는 분포가 퍼진 정도입니다. $e^H$는 같은 불확실성을 갖는 균등 분포의 후보 수입니다.
    분포 모양은 **질문에 따라 크게 다릅니다**. 질문을 바꿔 비교해 보세요.
    """)
    return


@app.cell
def _(PROMPTS, mo):
    prompt_choice = mo.ui.dropdown(options=PROMPTS, value=list(PROMPTS)[0], label="질문")
    prompt_choice
    return (prompt_choice,)


@app.cell
def _(chat_ids, model, prompt_choice, torch):
    @torch.inference_mode()
    def next_logits(input_ids):
        """원본 generate의 첫 step과 같은 계산: 마지막 위치 logits [B, V]."""
        return model(input_ids).logits[:, -1, :].float()

    def entropy(probs):
        return float(-(probs * probs.clamp_min(1e-12).log()).sum(-1))

    focus_ids = chat_ids(prompt_choice.value)          # [1, T]
    focus_logits = next_logits(focus_ids)              # [1, V]
    focus_probs = focus_logits.softmax(-1)
    return entropy, focus_ids, focus_logits, focus_probs, next_logits


@app.cell(hide_code=True)
def _(entropy, focus_logits, focus_probs, math, mo, plt, result_table, show_tok):
    # ▸ 결과 표시
    _p, _i = focus_probs[0].topk(15)
    _cum = _p.cumsum(0)
    _rows = [
        {"순위": r + 1, "토큰": show_tok(i), "id": int(i), "logit": round(float(focus_logits[0, i]), 3),
         "확률": round(float(p), 4), "누적 확률": round(float(c), 4)}
        for r, (p, i, c) in enumerate(zip(_p, _i, _cum))
    ]
    _H = entropy(focus_probs[0])
    _fig, _ax = plt.subplots(figsize=(8, 2.8))
    _ax.bar(range(15), _p.cpu(), color="#4C78A8")
    _ax.set_xticks(range(15), [show_tok(i) for i in _i], rotation=45, fontsize=8)
    _ax.set_ylabel("probability")
    _ax.set_title(f"next-token distribution (top 15 of {focus_probs.shape[-1]})")
    _fig.tight_layout()
    mo.vstack([
        mo.md(f"엔트로피 $H$ = **{_H:.2f} nat**, 유효 후보 수 $e^H$ ≈ **{math.exp(_H):.1f}**개. "
              f"상위 15개가 확률의 **{float(_cum[-1]):.1%}**를 차지합니다."),
        _fig,
        result_table(_rows, selection=None),
    ])
    return


@app.cell(hide_code=True)
def _(MiniMindForCausalLM, inspect, mo):
    mo.vstack([
        mo.md(r"""
    ### 2-1. 원본 `generate` 전체

    이 노트북이 따라갈 원본 코드입니다. 루프 한 바퀴가 토큰 하나를 만듭니다.
    KV cache를 쓰므로 두 번째 step부터는 **새 토큰 하나만** `forward`에 넣습니다.
    """),
        mo.accordion({"원본 코드 보기": mo.md(f"```python\n{inspect.getsource(MiniMindForCausalLM.generate)}\n```")}, multiple=True),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Greedy: 항상 가장 확률이 높은 토큰

    `do_sample=False`이면 `argmax`를 고릅니다. 무작위성이 없어 **같은 입력에는 항상 같은 답**이 나옵니다.

    `decode_loop`는 원본 루프(KV cache, eos 처리)를 그대로 옮긴 함수입니다. 토큰 선택만 `pick` 함수로 바꿀 수 있습니다.
    3절은 `argmax`, 8절은 원본 샘플링을 넣습니다. 각 step의 분포는 뒤 절에서 다시 씁니다.
    """)
    return


@app.cell
def _(EOS_ID, model, torch):
    @torch.inference_mode()
    def decode_loop(input_ids, pick, max_new_tokens=64, record=False):
        """원본 generate의 루프 구조를 그대로 따르는 생성 함수.

        pick(logits[B, V], input_ids[B, T]) -> next_token[B, 1]
        record=True이면 각 step의 (필터 전) 확률 분포 [B, V]를 CPU에 모읍니다.
        """
        past_key_values = None
        finished = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
        history = []
        for _ in range(max_new_tokens):
            past_len = past_key_values[0][0].shape[1] if past_key_values else 0
            outputs = model(input_ids[:, past_len:], None, past_key_values, use_cache=True)
            logits = outputs.logits[:, -1, :]
            if record:
                history.append(logits.float().softmax(-1).cpu())
            next_token = pick(logits.clone(), input_ids)
            # 이미 끝난 문장에는 eos를 계속 붙입니다(원본 287행과 같은 처리).
            next_token = torch.where(finished.unsqueeze(-1), next_token.new_full(next_token.shape, EOS_ID), next_token)
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            past_key_values = outputs.past_key_values
            finished |= next_token.squeeze(-1).eq(EOS_ID)
            if finished.all():
                break
        return input_ids, history

    def greedy_pick(logits, input_ids):
        return torch.argmax(logits, dim=-1, keepdim=True)

    return decode_loop, greedy_pick


@app.cell
def _(EOS_ID, PROMPTS, chat_ids, decode_loop, greedy_pick, model, torch):
    greedy_prompt_ids = chat_ids(PROMPTS["확신이 큰 질문: What is the capital of France?"])
    greedy_mine, greedy_history = decode_loop(greedy_prompt_ids, greedy_pick, max_new_tokens=96, record=True)
    greedy_original = model.generate(
        inputs=greedy_prompt_ids, max_new_tokens=96, do_sample=False,
        temperature=1.0, top_p=1.0, top_k=0, eos_token_id=EOS_ID,
    )
    greedy_same = torch.equal(greedy_mine, greedy_original)
    return greedy_history, greedy_mine, greedy_prompt_ids, greedy_same


@app.cell
def _(Counter):
    def repeated_ngram_ratio(token_ids, n=4):
        """한 답 안에서 이미 나왔던 n-gram이 다시 나온 비율 (0이면 반복 없음)."""
        grams = [tuple(token_ids[i:i + n]) for i in range(len(token_ids) - n + 1)]
        if not grams:
            return 0.0
        return 1 - len(Counter(grams)) / len(grams)

    return (repeated_ngram_ratio,)


@app.cell(hide_code=True)
def _(
    decode_answer,
    greedy_history,
    greedy_mine,
    greedy_prompt_ids,
    greedy_same,
    mo,
    plt,
    repeated_ngram_ratio,
    show_tok,
):
    # ▸ 결과 표시
    _gen = greedy_mine[0, greedy_prompt_ids.shape[1]:].tolist()
    _top1 = [float(h[0].max()) for h in greedy_history]
    _top2 = [float(h[0].topk(2).values[1]) for h in greedy_history]
    _fig, _ax = plt.subplots(figsize=(9, 2.6))
    _ax.plot(_top1, label="p(top-1) = chosen", color="#4C78A8")
    _ax.plot(_top2, label="p(top-2)", color="#F58518", alpha=0.8)
    _ax.set_xlabel("generation step")
    _ax.set_ylabel("probability")
    _ax.set_ylim(0, 1.02)
    _ax.legend(loc="lower right", fontsize=8)
    _fig.tight_layout()
    _close = [f"`{show_tok(_gen[s])}`({_top1[s]:.2f} vs {_top2[s]:.2f})" for s in range(len(_gen)) if _top1[s] - _top2[s] < 0.1][:6]
    mo.vstack([
        mo.callout(mo.md(f"직접 구현한 greedy와 원본 `generate(do_sample=False)` 결과가 토큰 단위로 **{'일치' if greedy_same else '불일치'}**합니다."),
                   kind="success" if greedy_same else "danger"),
        mo.md(f"**Greedy 답** (반복 4-gram 비율 **{repeated_ngram_ratio(_gen):.0%}**):\n\n> " + decode_answer(_gen).replace("\n", "\n> ")),
        _fig,
        mo.md(r"""
    **읽는 법:** 파란 선은 고른 토큰, 주황 선은 2등의 확률입니다. 두 선이 가까워도 greedy는 1등만 고릅니다.
    박빙이었던 step: """ + (", ".join(_close) if _close else "없음") + r""".

    작은 모델의 greedy 출력은 **같은 구절을 자주 되풀이**합니다. 나온 구절이 문맥에 남아 같은 구절의 확률을 높이기 때문입니다.
    매 step 1등을 골라도 **문장 전체의 확률이 최대인 것은 아닙니다**. 이를 근사하는 beam search는 원본에 없습니다.
    """),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Temperature: 분포를 뾰족하게 또는 평평하게

    원본: `logits = outputs.logits[:, -1, :] / temperature`

    $$
    p_i(T) = \frac{e^{z_i / T}}{\sum_j e^{z_j / T}}, \qquad
    \frac{p_i(T)}{p_j(T)} = e^{(z_i - z_j)/T}
    $$

    - $T<1$: 1등에 확률이 몰립니다. $T\to 0$이면 greedy입니다.
    - $T=1$: 모델 분포 그대로입니다.
    - $T>1$: 꼬리 후보도 뽑힙니다. $T\to\infty$이면 균등 분포입니다.

    모든 logit을 같은 양수로 나누므로 **순위는 그대로**입니다. 분포 모양만 바뀝니다.
    """)
    return


@app.cell
def _(mo):
    temperature_slider = mo.ui.slider(0.1, 3.0, step=0.05, value=0.7, label="temperature T", show_value=True)
    temperature_slider
    return (temperature_slider,)


@app.cell(hide_code=True)
def _(entropy, focus_logits, math, mo, plt, show_tok, temperature_slider, torch):
    # ▸ 결과 표시
    _T = temperature_slider.value
    _p1 = focus_logits[0].softmax(-1)
    _pT = (focus_logits[0] / _T).softmax(-1)
    _idx = _p1.topk(12).indices
    _fig, (_a1, _a2) = plt.subplots(1, 2, figsize=(10, 2.8), gridspec_kw={"width_ratios": [1.6, 1]})
    _x = torch.arange(12)
    _a1.bar(_x - 0.2, _p1[_idx].cpu(), width=0.4, label="T = 1", color="#BBBBBB")
    _a1.bar(_x + 0.2, _pT[_idx].cpu(), width=0.4, label=f"T = {_T:g}", color="#4C78A8")
    _a1.set_xticks(_x, [show_tok(i) for i in _idx], rotation=45, fontsize=8)
    _a1.set_ylabel("probability")
    _a1.legend(fontsize=8)
    _Ts = torch.linspace(0.1, 3.0, 59)
    _Hs = [entropy((focus_logits[0] / t).softmax(-1)) for t in _Ts]
    _a2.plot(_Ts, [math.exp(h) for h in _Hs], color="#4C78A8")
    _a2.axvline(_T, color="#E45756", ls="--")
    _a2.set_yscale("log")
    _a2.set_xlabel("temperature")
    _a2.set_ylabel("effective #candidates  e^H")
    _fig.tight_layout()
    mo.vstack([
        _fig,
        mo.md(f"T = {_T:g}: 1등 확률 {float(_p1.max()):.3f} → **{float(_pT.max()):.3f}**, "
              f"유효 후보 수 {math.exp(entropy(_p1)):.1f} → **{math.exp(entropy(_pT)):.1f}**, "
              f"상위 12개 밖 확률 {float(1 - _p1[_idx].sum()):.3f} → **{float(1 - _pT[_idx].sum()):.3f}**"),
        mo.md(r"""
    오른쪽 그림은 로그 축입니다. $T$가 2를 넘으면 후보가 수백 개 이상으로 늘어납니다.
    대부분 문맥과 무관한 토큰이라 답이 망가집니다. 이 꼬리를 자르는 것이 top-k와 top-p입니다.
    """),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Top-k: 점수 상위 k개만 남기기

    ```python
    if top_k > 0:
        logits[logits < torch.topk(logits, top_k)[0][..., -1, None]] = -float('inf')
    ```

    k등 점수보다 낮은 토큰은 `-inf`가 되어 확률 0이 됩니다. 남은 확률은 softmax가 다시 합 1로 맞춥니다.
    `top_k=0`이면 건너뜁니다. k등과 동점인 토큰은 함께 남습니다.

    **약점:** k가 문맥과 무관하게 고정입니다. 확신이 클 때는 꼬리가 남고, 후보가 퍼졌을 때는 좋은 후보가 잘립니다.
    """)
    return


@app.cell
def _(mo):
    top_k_slider = mo.ui.slider(1, 200, step=1, value=50, label="top_k", show_value=True)
    top_k_slider
    return (top_k_slider,)


@app.cell
def _(PROMPTS, chat_ids, next_logits):
    # 5·6절 비교용: 세 질문의 첫 step logits [V]
    prompt_logits = {name: next_logits(chat_ids(q))[0] for name, q in PROMPTS.items()}
    return (prompt_logits,)


@app.cell(hide_code=True)
def _(mo, prompt_logits, result_table, show_tok, top_k_slider, torch):
    # ▸ 결과 표시
    _k = top_k_slider.value
    _rows = []
    for _name, _z in prompt_logits.items():
        _z = _z.clone()
        _p = _z.softmax(-1)
        _z[_z < torch.topk(_z, _k)[0][..., -1, None]] = -float("inf")    # 원본과 같은 한 줄
        _kept = torch.isfinite(_z)
        _q = _z.softmax(-1)
        _rows.append({
            "질문": _name.split(":")[0],
            "남은 후보 수": int(_kept.sum()),
            "남은 후보의 원래 확률 합": round(float(_p[_kept].sum()), 4),
            "잘린 꼬리 확률": round(float(_p[~_kept].sum()), 4),
            "k등 토큰 (원래 확률→재정규화)": f"{show_tok(_p.topk(_k).indices[-1])} "
                                             f"({float(_p.topk(_k).values[-1]):.4f}→{float(_q.topk(_k).values[-1]):.4f})",
        })
    result_table(_rows, selection=None)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. Top-p (nucleus): 누적 확률 p까지만 남기기

    ```python
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        mask = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1) > top_p
        mask[..., 1:], mask[..., 0] = mask[..., :-1].clone(), 0
        logits[mask.scatter(1, sorted_indices, mask)] = -float('inf')
    ```

    1. 점수 순으로 정렬해 누적 확률을 구합니다.
    2. 누적 확률이 p를 넘는 위치를 표시합니다.
    3. 마스크를 한 칸 밀어 p를 처음 넘는 토큰까지 남깁니다. 1등은 항상 남습니다.
    4. `scatter`로 원래 어휘 순서에 마스크를 적용합니다.

    top-k와 달리 **후보 수가 문맥에 맞춰 바뀝니다**. 확신이 크면 적게, 퍼져 있으면 많이 남습니다.

    **주의:** top-p는 temperature와 top-k **다음에** 적용됩니다. 따라서 누적 확률은 그 결과 분포 기준이고,
    온도가 높으면 같은 p에서도 후보가 더 많이 남습니다.
    """)
    return


@app.cell
def _(mo):
    top_p_slider = mo.ui.slider(0.05, 1.0, step=0.01, value=0.9, label="top_p", show_value=True)
    top_p_slider
    return (top_p_slider,)


@app.cell
def _(torch):
    def apply_top_p(logits, top_p):
        """원본 281–285행과 같은 계산. logits [B, V]를 제자리에서 바꾸고 반환합니다."""
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            mask = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1) > top_p
            mask[..., 1:], mask[..., 0] = mask[..., :-1].clone(), 0
            logits[mask.scatter(1, sorted_indices, mask)] = -float("inf")
        return logits

    return (apply_top_p,)


@app.cell(hide_code=True)
def _(apply_top_p, mo, prompt_logits, result_table, show_tok, top_p_slider, torch):
    # ▸ 결과 표시
    _pv = top_p_slider.value
    _rows = []
    for _name, _z in prompt_logits.items():
        _p = _z.softmax(-1)
        _f = apply_top_p(_z.clone()[None], _pv)[0]
        _kept = torch.isfinite(_f)
        _order = _p.argsort(descending=True)
        _kept_sorted = [int(i) for i in _order[: int(_kept.sum())]]
        _rows.append({
            "질문": _name.split(":")[0],
            "남은 후보 수": int(_kept.sum()),
            "남은 후보의 원래 확률 합": round(float(_p[_kept].sum()), 4),
            "남은 후보 (앞 8개)": " ".join(show_tok(i) for i in _kept_sorted[:8]) + (" …" if len(_kept_sorted) > 8 else ""),
        })
    mo.vstack([
        result_table(_rows, selection=None),
        mo.md(f"같은 p = {_pv:g}에서도 질문마다 후보 수가 다릅니다. "
              "마지막 토큰까지 남기므로 확률 합은 p를 조금 넘습니다."),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 6-1. 생성 중에도 후보 수는 계속 바뀐다

    3절 greedy 생성의 step별 분포로 top-p와 top-k의 후보 수를 비교합니다.
    대부분의 step은 다음 토큰이 거의 정해져 있습니다. top-p는 갈래가 많은 step에서만 후보를 넓힙니다.
    """)
    return


@app.cell(hide_code=True)
def _(
    greedy_history,
    greedy_mine,
    greedy_prompt_ids,
    mo,
    plt,
    show_tok,
    top_k_slider,
    top_p_slider,
    torch,
):
    # ▸ 결과 표시
    _pv, _k = top_p_slider.value, top_k_slider.value
    _probs = torch.cat(greedy_history)                       # [steps, V]
    _sorted = _probs.sort(dim=-1, descending=True).values
    _nucleus = ((_sorted.cumsum(-1) > _pv).float().argmax(-1) + 1).clamp(max=_probs.shape[-1])
    _nucleus = torch.where(_sorted.cumsum(-1)[:, -1] <= _pv, _probs.shape[-1], _nucleus)
    _gen = greedy_mine[0, greedy_prompt_ids.shape[1]:].tolist()
    _fig, _ax = plt.subplots(figsize=(9, 2.8))
    _ax.step(range(len(_nucleus)), _nucleus, where="mid", color="#4C78A8", label=f"top-p = {_pv:g}")
    _ax.axhline(_k, color="#F58518", ls="--", label=f"top-k = {_k}")
    _ax.set_yscale("log")
    _ax.set_xlabel("generation step")
    _ax.set_ylabel("#kept candidates")
    _ax.legend(fontsize=8)
    _fig.tight_layout()
    _wide = sorted(range(len(_nucleus)), key=lambda s: -int(_nucleus[s]))[:5]
    mo.vstack([
        _fig,
        mo.md(f"top-p 후보가 1개인 step: **{int((_nucleus == 1).sum())} / {len(_nucleus)}**. "
              "후보가 가장 많은 step(고른 토큰): "
              + ", ".join(f"{s}번 `{show_tok(_gen[s])}` ({int(_nucleus[s])}개)" for s in sorted(_wide) if s < len(_gen))),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7. Repetition penalty: 이미 나온 토큰의 점수 깎기

    ```python
    if repetition_penalty != 1.0:
        for i in range(input_ids.shape[0]):
            seen = torch.unique(input_ids[i]); score = logits[i, seen]
            logits[i, seen] = torch.where(score > 0, score / repetition_penalty, score * repetition_penalty)
    ```

    $$
    z_i' = \begin{cases} z_i / \rho & z_i > 0 \\ z_i \cdot \rho & z_i \le 0 \end{cases}
    \quad (i \in \text{seen}),\qquad \rho > 1
    $$

    양수는 나누고 음수는 곱하므로, $\rho>1$이면 **점수가 항상 내려갑니다**.

    - `seen`은 **프롬프트를 포함한** 전체 입력입니다. 질문 단어와 템플릿 토큰(`<|im_start|>`, `↵` 등)도 깎입니다.
    - `torch.unique`를 쓰므로 **등장 횟수와 무관**합니다.
    - temperature 다음에 적용하지만, 둘 다 곱셈이라 순서를 바꿔도 결과는 같습니다.
    - 효과가 **logit의 절댓값에 따라 달라집니다**. softmax와 달리 logit 이동에 불변이 아닙니다.
    """)
    return


@app.cell
def _(mo):
    penalty_slider = mo.ui.slider(1.0, 2.0, step=0.05, value=1.3, label="repetition_penalty ρ", show_value=True)
    penalty_slider
    return (penalty_slider,)


@app.cell
def _(torch):
    def apply_repetition_penalty(logits, input_ids, repetition_penalty):
        """원본 276–278행과 같은 계산. logits [B, V]를 제자리에서 바꾸고 반환합니다."""
        if repetition_penalty != 1.0:
            for i in range(input_ids.shape[0]):
                seen = torch.unique(input_ids[i])
                score = logits[i, seen]
                logits[i, seen] = torch.where(score > 0, score / repetition_penalty, score * repetition_penalty)
        return logits

    return (apply_repetition_penalty,)


@app.cell
def _(apply_repetition_penalty, decode_loop, greedy_prompt_ids, penalty_slider, torch):
    def greedy_with_penalty(logits, input_ids):
        logits = apply_repetition_penalty(logits, input_ids, penalty_slider.value)
        return torch.argmax(logits, dim=-1, keepdim=True)

    penalized_ids, _ = decode_loop(greedy_prompt_ids, greedy_with_penalty, max_new_tokens=96)
    return (penalized_ids,)


@app.cell(hide_code=True)
def _(
    apply_repetition_penalty,
    decode_answer,
    greedy_mine,
    greedy_prompt_ids,
    mo,
    next_logits,
    penalized_ids,
    penalty_slider,
    repeated_ngram_ratio,
    result_table,
    show_tok,
):
    # ▸ 결과 표시
    _rho = penalty_slider.value
    _T0 = greedy_prompt_ids.shape[1]
    # greedy 답의 앞 6토큰 직후 step의 분포에서 효과를 봅니다.
    _ctx = greedy_mine[:, : _T0 + 6]
    _z = next_logits(_ctx)
    _zp = apply_repetition_penalty(_z.clone(), _ctx, _rho)
    _seen = set(_ctx[0].tolist())
    _p, _pp = _z[0].softmax(-1), _zp[0].softmax(-1)
    _rows = [{"토큰": show_tok(i), "이미 나옴": "예" if int(i) in _seen else "",
              "logit": round(float(_z[0, i]), 2), "벌점 후 logit": round(float(_zp[0, i]), 2),
              "확률": round(float(_p[i]), 4), "벌점 후 확률": round(float(_pp[i]), 4)}
             for i in _p.topk(10).indices]
    _g = greedy_mine[0, _T0:].tolist()
    _r = penalized_ids[0, _T0:].tolist()
    mo.vstack([
        mo.md(f"**문맥** `…{decode_answer(_ctx[0, _T0:].tolist())}` 다음 토큰, ρ = {_rho:g}:"),
        result_table(_rows, selection=None),
        mo.md(f"""
    | 설정 | 반복 4-gram 비율 | 답 |
    | --- | --- | --- |
    | greedy, ρ = 1 | {repeated_ngram_ratio(_g):.0%} | {decode_answer(_g).replace(chr(10), " ↵ ").replace("|", "/")} |
    | greedy, ρ = {_rho:g} | {repeated_ngram_ratio(_r):.0%} | {decode_answer(_r).replace(chr(10), " ↵ ").replace("|", "/")} |

    ρ를 올리면 반복이 줄지만 **다시 써야 할 토큰도 깎입니다**("France", 문장 부호, 정답 단어).
    그래서 **사실이 틀린 문장**이 나올 수 있습니다. ρ = 1.3 근처의 답을 읽어 보세요.
    ρ를 1.8 이상으로 올리면 답이 더 어색해집니다.
    """),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 8. 한 step 재구현과 원본 대조

    앞의 조각을 원본과 **같은 순서**로 잇습니다. 마지막에 `torch.multinomial`로 필터 후 확률에 비례해 하나를 뽑습니다.
    원본 기본값은 `temperature=0.85, top_p=0.85, top_k=50, repetition_penalty=1.0`입니다.

    `multinomial`은 전역 난수 생성기를 씁니다. 같은 시드와 같은 연산 순서면 같은 난수가 나오므로,
    직접 구현과 원본의 출력은 **토큰 단위로 같아야** 합니다. 4개 문장을 함께 생성해 확인합니다.
    """)
    return


@app.cell
def _(apply_repetition_penalty, apply_top_p, torch):
    def filter_logits(logits, input_ids, temperature=0.85, top_k=50, top_p=0.85, repetition_penalty=1.0):
        """원본 generate의 275–285행: temperature → repetition penalty → top-k → top-p."""
        logits = logits / temperature
        logits = apply_repetition_penalty(logits, input_ids, repetition_penalty)
        if top_k > 0:
            logits[logits < torch.topk(logits, top_k)[0][..., -1, None]] = -float("inf")
        return apply_top_p(logits, top_p)

    def make_sampler(do_sample=True, **settings):
        """decode_loop에 넣을 pick 함수 (원본 286행)."""
        def pick(logits, input_ids):
            logits = filter_logits(logits, input_ids, **settings)
            if do_sample:
                return torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1)
            return torch.argmax(logits, dim=-1, keepdim=True)
        return pick

    return filter_logits, make_sampler


@app.cell
def _(EOS_ID, PROMPTS, chat_ids, decode_loop, make_sampler, model, torch):
    _settings = dict(temperature=0.85, top_k=50, top_p=0.85, repetition_penalty=1.1)
    _x = chat_ids(PROMPTS["창작: Tell me a story about a cat."])
    replay_checks = []
    for _seed in [0, 1, 2]:
        torch.manual_seed(_seed)
        _orig = model.generate(inputs=_x, max_new_tokens=48, num_return_sequences=4, eos_token_id=EOS_ID, **_settings)
        torch.manual_seed(_seed)
        _mine, _ = decode_loop(_x.repeat(4, 1), make_sampler(**_settings), max_new_tokens=48)
        replay_checks.append({"seed": _seed, "원본 shape": tuple(_orig.shape), "직접 구현 shape": tuple(_mine.shape),
                              "토큰 일치": torch.equal(_orig, _mine)})
    replay_sample = (_x, _mine)
    return replay_checks, replay_sample


@app.cell(hide_code=True)
def _(decode_answer, mo, replay_checks, replay_sample, result_table):
    # ▸ 결과 표시
    _x, _mine = replay_sample
    _ok = all(r["토큰 일치"] for r in replay_checks)
    mo.vstack([
        mo.callout(mo.md(f"설정 `T=0.85, top_k=50, top_p=0.85, ρ=1.1` · 시드 3개 × 문장 4개: 원본과 직접 구현이 "
                         f"**{'모두 일치' if _ok else '불일치'}**합니다."), kind="success" if _ok else "danger"),
        result_table(replay_checks, selection=None),
        mo.md("**같은 시드에서 직접 구현이 만든 4개 답 (48토큰까지)**\n\n"
              + "\n".join(f"{n + 1}. {decode_answer(r[_x.shape[1]:].tolist())}" for n, r in enumerate(_mine))),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 8-1. `multinomial`은 정말 확률대로 뽑을까

    필터 후 분포에서 첫 토큰을 **20,000번** 뽑아 빈도와 확률을 비교합니다.
    `-inf`가 된 토큰은 한 번도 나오지 않아야 합니다. 4–6절 슬라이더를 바꾸면 이 표도 바뀝니다.
    """)
    return


@app.cell(hide_code=True)
def _(
    filter_logits,
    focus_ids,
    focus_logits,
    mo,
    result_table,
    show_tok,
    temperature_slider,
    top_k_slider,
    top_p_slider,
    torch,
):
    # ▸ 결과 표시
    _settings = dict(temperature=temperature_slider.value, top_k=top_k_slider.value, top_p=top_p_slider.value)
    _q = torch.softmax(filter_logits(focus_logits.clone(), focus_ids, **_settings), dim=-1)[0]
    _gen = torch.Generator(device=_q.device).manual_seed(0)
    _draws = torch.multinomial(_q, 20_000, replacement=True, generator=_gen)
    _freq = torch.bincount(_draws, minlength=_q.numel()).float() / 20_000
    _rows = [{"토큰": show_tok(i), "원래 확률": round(float(focus_logits[0].softmax(-1)[i]), 4),
              "필터 후 확률": round(float(_q[i]), 4), "20,000회 빈도": round(float(_freq[i]), 4)}
             for i in _q.topk(min(10, int((_q > 0).sum()))).indices]
    mo.vstack([
        mo.md(f"설정 `T={_settings['temperature']:g}, top_k={_settings['top_k']}, top_p={_settings['top_p']:g}` · "
              f"후보 **{int((_q > 0).sum())}**개 · 후보 밖 토큰이 뽑힌 횟수 **{int((_q[_draws] == 0).sum())}**"),
        result_table(_rows, selection=None),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 9. 설정별 다양성과 품질 비교

    고양이 이야기 질문으로 설정마다 **답 8개**를 만들고 두 지표를 잽니다.

    - **distinct-2 (다양성):** 전체 bigram 중 서로 다른 bigram의 비율입니다. 답이 겹칠수록 낮습니다.
    - **토큰당 NLL (자연스러움):** 원래 모델($T=1$)이 본 생성 토큰의 평균 $-\log p$입니다. 낮을수록 자연스럽습니다.

    두 지표는 상충합니다. greedy는 NLL이 가장 낮지만 다양성이 없습니다. 온도를 올리면 다양해지지만 NLL이 커집니다.
    top-p는 꼬리를 잘라 NLL 증가를 줄입니다. 실제 품질은 예시 답을 직접 읽어 판단하세요.
    """)
    return


@app.cell
def _(
    Counter,
    EOS_ID,
    F,
    PROMPTS,
    chat_ids,
    decode_answer,
    model,
    repeated_ngram_ratio,
    torch,
):
    PRESETS = {
        "greedy": dict(do_sample=False, temperature=1.0, top_k=0, top_p=1.0),
        "T=0.7, top_p=0.9": dict(temperature=0.7, top_k=0, top_p=0.9),
        "원본 기본값 (T=0.85, k=50, p=0.85)": dict(temperature=0.85, top_k=50, top_p=0.85),
        "순수 샘플링 (T=1)": dict(temperature=1.0, top_k=0, top_p=1.0),
        "T=1.5, 필터 없음": dict(temperature=1.5, top_k=0, top_p=1.0),
        "T=1.5, top_p=0.9": dict(temperature=1.5, top_k=0, top_p=0.9),
    }
    N_SAMPLES, MAX_NEW = 8, 96

    @torch.inference_mode()
    def nll_per_token(full_ids, prompt_len):
        """생성 부분(첫 eos 포함)의 토큰당 NLL. 원래 모델(T=1) 기준."""
        logp = F.log_softmax(model(full_ids).logits[:, :-1].float(), dim=-1)
        target = full_ids[:, 1:]
        token_logp = logp.gather(-1, target[..., None])[..., 0][:, prompt_len - 1:]
        gen = full_ids[:, prompt_len:]
        # 첫 eos까지만 셉니다. 그 뒤는 원본이 채운 eos입니다.
        valid = (gen.eq(EOS_ID).long().cumsum(-1) - gen.eq(EOS_ID).long()) == 0
        return float(-(token_logp * valid).sum() / valid.sum())

    def run_preset(question, seed, **settings):
        x = chat_ids(question)
        torch.manual_seed(seed)
        out = model.generate(inputs=x, max_new_tokens=MAX_NEW, num_return_sequences=N_SAMPLES,
                             eos_token_id=EOS_ID, **settings)
        answers_ids = [r[x.shape[1]:].tolist() for r in out]
        answers_ids = [a[: a.index(EOS_ID)] if EOS_ID in a else a for a in answers_ids]
        bigrams = [tuple(a[i:i + 2]) for a in answers_ids for i in range(len(a) - 1)]
        return {
            "distinct-2": len(Counter(bigrams)) / max(len(bigrams), 1),
            "NLL/token": nll_per_token(out, x.shape[1]),
            "서로 다른 답": len({tuple(a) for a in answers_ids}),
            "반복 4-gram": sum(repeated_ngram_ratio(a) for a in answers_ids) / len(answers_ids),
            "answers": [decode_answer(a) for a in answers_ids],
        }

    preset_results = {name: run_preset(PROMPTS["창작: Tell me a story about a cat."], seed=0, **s)
                      for name, s in PRESETS.items()}
    return PRESETS, preset_results


@app.cell(hide_code=True)
def _(mo, plt, preset_results, result_table):
    # ▸ 결과 표시
    _rows = [{"설정": n, "distinct-2": round(r["distinct-2"], 3), "NLL/token": round(r["NLL/token"], 3),
              "서로 다른 답 (/8)": r["서로 다른 답"], "평균 반복 4-gram": f"{r['반복 4-gram']:.0%}"} for n, r in preset_results.items()]
    _fig, _ax = plt.subplots(figsize=(7, 3.4))
    _labels = ["greedy", "T0.7 p0.9", "default", "T1", "T1.5", "T1.5 p0.9"]
    for _lab, (_n, _r) in zip(_labels, preset_results.items()):
        _ax.scatter(_r["NLL/token"], _r["distinct-2"], s=40)
        _ax.annotate(_lab, (_r["NLL/token"], _r["distinct-2"]), textcoords="offset points", xytext=(5, 4), fontsize=8)
    _ax.set_xlabel("NLL per token under the T=1 model  (lower = more 'expected')")
    _ax.set_ylabel("distinct-2 across 8 answers")
    _fig.tight_layout()
    mo.vstack([
        result_table(_rows, selection=None),
        _fig,
        mo.accordion({
            n: mo.md("\n\n".join(f"**{i + 1}.** " + a.replace("\n", " ↵ ") for i, a in enumerate(r["answers"][:3])))
            for n, r in preset_results.items()
        }),
        mo.md(r"""
    **읽는 법:** 오른쪽 위일수록 다양하지만 덜 자연스럽습니다. `T=1.5, 필터 없음`은 문장이 금방 무너집니다.
    같은 온도에 `top_p=0.9`를 더하면 NLL이 줄어듭니다. greedy는 8개 답이 모두 같습니다.
    """),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 10. 직접 해보기

    설정을 바꾸고 **생성**을 누르면 원본 `model.generate`로 답을 만듭니다. 같은 시드면 같은 답이 나옵니다.
    `do_sample`을 끄면 argmax를 고릅니다. 이때 결과를 바꾸는 설정은 repetition penalty뿐입니다.
    """)
    return


@app.cell
def _(mo):
    play = mo.ui.dictionary({
        "question": mo.ui.text_area(value="Explain why the sky is blue in two sentences.", label="질문", full_width=True),
        "do_sample": mo.ui.checkbox(value=True, label="do_sample"),
        "temperature": mo.ui.slider(0.1, 2.0, step=0.05, value=0.85, label="temperature", show_value=True),
        "top_k": mo.ui.slider(0, 200, step=1, value=50, label="top_k (0 = 끔)", show_value=True),
        "top_p": mo.ui.slider(0.05, 1.0, step=0.01, value=0.85, label="top_p (1 = 끔)", show_value=True),
        "repetition_penalty": mo.ui.slider(1.0, 2.0, step=0.05, value=1.0, label="repetition_penalty", show_value=True),
        "n": mo.ui.slider(1, 8, step=1, value=4, label="답 개수", show_value=True),
        "max_new_tokens": mo.ui.slider(16, 256, step=16, value=128, label="max_new_tokens", show_value=True),
        "seed": mo.ui.number(start=0, stop=10_000, value=0, label="seed"),
    })
    play_button = mo.ui.run_button(label="생성")
    mo.vstack([play, play_button])
    return play, play_button


@app.cell
def _(EOS_ID, chat_ids, decode_answer, mo, model, play, play_button, torch):
    mo.stop(not play_button.value, mo.md("*설정을 고르고 **생성**을 누르세요.*"))
    _v = play.value
    _x = chat_ids(_v["question"])
    torch.manual_seed(int(_v["seed"]))
    _out = model.generate(
        inputs=_x, max_new_tokens=_v["max_new_tokens"], num_return_sequences=_v["n"], eos_token_id=EOS_ID,
        do_sample=_v["do_sample"], temperature=_v["temperature"], top_k=_v["top_k"], top_p=_v["top_p"],
        repetition_penalty=_v["repetition_penalty"],
    )
    mo.md("\n\n".join(f"**{i + 1}.** " + decode_answer(r[_x.shape[1]:].tolist()).replace("\n", "  \n")
                      for i, r in enumerate(_out)))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 11. 정리

    | 설정 | 하는 일 | 너무 작으면 | 너무 크면 | 원본 기본값 |
    | --- | --- | --- | --- | --- |
    | `temperature` | 점수 차이를 키우거나 줄임 (순위 유지) | greedy처럼 반복 | 꼬리 토큰이 뽑혀 문장 붕괴 | 0.85 |
    | `top_k` | 상위 k개만 후보로 | 다양성 부족 | 꼬리 차단 효과 없음 | 50 (0 = 끔) |
    | `top_p` | 누적 확률 p까지만 후보로 (문맥 적응) | greedy에 가까움 | 꼬리 차단 효과 없음 | 0.85 (1 = 끔) |
    | `repetition_penalty` | 이미 나온 토큰 점수 깎기 (프롬프트 포함, 횟수 무관) | (1 = 끔) | 꼭 필요한 반복도 막혀 어색 | 1.0 |
    | `do_sample` | 확률대로 뽑기 / argmax | — | — | True |

    **핵심:**

    1. 모델은 분포를 주고, 디코딩 설정이 토큰을 고릅니다.
    2. temperature는 분포 모양을, top-k·top-p는 후보 집합을 정합니다. 순서는 temperature → penalty → top-k → top-p → 선택입니다.
    3. greedy는 반복하기 쉽고, 높은 온도만 쓰면 문장이 무너집니다. 적당한 온도와 top-p를 함께 쓰는 것이 흔한 절충입니다.
    4. 같은 시드와 같은 연산 순서면 샘플링도 재현됩니다(8절).

    **Hugging Face와의 차이:** `transformers`의 `generate`에는 `min_p`, beam search 등 옵션이 더 많습니다.
    MiniMind는 `generate`를 직접 정의해 덮어쓰므로, 이 노트북의 호출은 모두 원본 구현을 씁니다.

    **더 해 보기:**

    - 2절 질문을 바꾸며 엔트로피와 6절 후보 수를 비교하세요.
    - 9절 `PRESETS`에 `top_k=10`이나 `repetition_penalty=1.3`을 추가해 보세요.
    - 10절에서 `do_sample`을 끄고 temperature를 바꿔도 답이 같은 이유를 4절 수식으로 설명해 보세요.
    """)
    return


if __name__ == "__main__":
    app.run()
