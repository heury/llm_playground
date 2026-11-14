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
    `uv run marimo edit minimind_lora_tour.py`. 환경·준비물과 권장 순서는 [공통 README](README.md)를 참고하세요.
    위젯 변경 시 의존 셀이 다시 실행됩니다. 모델 크기·데이터·학습 설정 변경 전에는 해당 절의 재실행 범위를 확인하세요.
    """)
    return


@app.cell
def _():
    import copy
    import json
    import sys
    import tempfile
    from contextlib import nullcontext
    from pathlib import Path

    import datasets
    import marimo as mo
    import matplotlib.pyplot as plt
    import torch
    import torch.nn.functional as F
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    from torch import nn
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer

    # 노트북 위치(notebooks/minimind/)에서 두 단계 위가 저장소 루트입니다.
    repo_root = Path(__file__).resolve().parents[2]
    minimind_dir = repo_root / "projects/minimind"
    sys.path.insert(0, str(minimind_dir))

    # 원본 파일의 클래스와 함수를 그대로 가져옵니다.
    from dataset.lm_dataset import SFTDataset
    from model.model_lora import LoRA, apply_lora, load_lora, merge_lora, save_lora
    from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
    from trainer.trainer_utils import SkipBatchSampler, get_lr, setup_seed

    # GPU가 있으면 GPU, 없으면 CPU (train_lora.py의 --device 기본값과 같은 규칙)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        torch.set_num_threads(4)

    def autocast_ctx():
        """train_lora.py의 '3. 혼합 정밀도 설정': GPU는 bf16 autocast, CPU는 그대로 fp32"""
        return nullcontext() if device == "cpu" else torch.autocast(device_type="cuda", dtype=torch.bfloat16)

    datasets.disable_progress_bars()
    tokenizer = AutoTokenizer.from_pretrained(minimind_dir / "model")
    return (
        DataLoader,
        F,
        LoRA,
        MiniMindConfig,
        MiniMindForCausalLM,
        Path,
        SFTDataset,
        SkipBatchSampler,
        apply_lora,
        autocast_ctx,
        copy,
        device,
        get_lr,
        hf_hub_download,
        json,
        load_file,
        load_lora,
        merge_lora,
        mo,
        nn,
        plt,
        save_lora,
        setup_seed,
        tempfile,
        tokenizer,
        torch,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.vstack([
        mo.md(r"""
    # MiniMind LoRA 구조 따라가기

    `projects/minimind/model/model_lora.py`는 60줄 정도의 짧은 파일입니다. 이 파일만으로
    MiniMind에 **LoRA(Low-Rank Adaptation)** 를 붙이고, 학습한 LoRA를 저장·불러오기·병합합니다.
    이 노트북은 파일의 함수 다섯 개를 차례로 실행하면서 **무엇이 어디에 붙고, 무엇이 학습되고, 무엇이 저장되는지** 확인합니다.

    2절부터는 실제로 학습된 **64M MiniMind 모델**
    [`caspar/MiniMind-Dense-64M-Mixed-SFT-v1.0`](https://huggingface.co/caspar/MiniMind-Dense-64M-Mixed-SFT-v1.0)에 LoRA를 붙여,
    SFT 노트북(`minimind_sft_tour.py`)과 같은 과제(새 정체성 "Mimi" 가르치기)를 **매개변수의 0.6%만** 학습해서 합니다.

    **준비:** 처음 실행할 때 Hugging Face에서 가중치(약 128MB)를 내려받습니다. **NVIDIA GPU가 있으면 자동으로 GPU를 씁니다**
    (GPU에서 전체 약 20초). GPU가 없으면 CPU로 실행되며 1~2분 걸립니다.

    **사전 지식:** Python, `nn.Linear`, 텐서 `shape`, 행렬 곱 `@`. MiniMind 블록 구조(Attention의 `q_proj`·`o_proj` 등).

    ## 0. 큰 그림: 전체 미세 조정 vs LoRA

    미세 조정은 이미 학습된 가중치 `W`를 새 데이터에 맞게 조금 바꾸는 일입니다.

    - **전체 미세 조정:** `W` 전체를 바꿉니다. $D\times D$ 행렬이면 $D^2$개를 학습하고 저장해야 합니다.
    - **LoRA:** `W`는 **고정**하고, 바꿀 양 `ΔW`를 작은 행렬 두 개의 곱 $BA$로 표현해 그 둘만 학습합니다.

    $$
    h = W x + \underbrace{B A}_{\Delta W} x, \qquad
    A \in \mathbb{R}^{r \times d_{in}},\; B \in \mathbb{R}^{d_{out} \times r},\; r \ll d
    $$

    """),
        mo.mermaid(r"""
    flowchart LR
     X["입력 x"] --> W["원래 Linear W: 고정"] --> ADD(("+")) --> Y["출력 h"]
     X --> A["A: d_in → r, 무작위 초기화, 학습"]
     A --> B["B: r → d_out, 0 초기화, 학습"] --> ADD
    """),
        mo.md(r"""

    **파일 속 함수 지도**

    | 원본 이름 | 하는 일 | 절 | 사용하는 곳 |
    | --- | --- | --- | --- |
    | `LoRA` | `A`, `B` 두 Linear와 `B(A(x))` | 1 | — |
    | `apply_lora` | 정사각형 Linear에 LoRA를 붙이고 `forward`를 교체 | 2, 3 | `train_lora.py`, `eval_llm.py` |
    | (학습 루프) | LoRA 이외 매개변수 고정, LoRA만 최적화 | 4 | `train_lora.py` |
    | `save_lora`, `load_lora` | LoRA 가중치만 저장·복원 | 5 | `train_lora.py`, `eval_llm.py` |
    | `merge_lora` | $W+BA$로 합친 일반 모델 저장 | 6 | `scripts/convert_model.py` |
    """)
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. `LoRA` 클래스: 작은 행렬 두 개

    ```python
    self.A = nn.Linear(in_features, rank, bias=False)   # d_in → r  (줄이기)
    self.B = nn.Linear(rank, out_features, bias=False)  # r → d_out (늘리기)
    self.A.weight.data.normal_(mean=0.0, std=0.02)      # A: 작은 무작위 값
    self.B.weight.data.zero_()                          # B: 0
    def forward(self, x): return self.B(self.A(x))
    ```

    - `nn.Linear`의 가중치는 `[출력, 입력]` 모양입니다. 그래서 `A.weight`는 `[r, d_in]`, `B.weight`는 `[d_out, r]`이고
      $\Delta W=BA$ (`B.weight @ A.weight`)는 원래 가중치와 같은 `[d_out, d_in]`입니다.
    - 폭이 `r`인 통로를 지나므로 `ΔW`의 **rank(독립된 방향의 수)는 최대 `r`** 입니다. 이름의 "Low-Rank"가 이 뜻입니다.
    - $B=0$에서 시작하므로 처음에는 $B(Ax)=0$. LoRA를 붙여도 모델 출력이 **처음엔 그대로**입니다.
    - MiniMind 구현에는 다른 라이브러리에 흔한 `alpha / r` 배율과 LoRA dropout이 없습니다. 출력에 `B(A(x))`를 그대로 더합니다.
    """)
    return


@app.cell
def _(LoRA, torch):
    lora_dim, lora_demo_rank = 64, 4

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        demo_lora = LoRA(lora_dim, lora_dim, rank=lora_demo_rank)
        demo_x = torch.randn(3, lora_dim)  # 토큰 벡터 3개

    with torch.no_grad():
        # ① 초기 상태: B = 0이라 출력이 정확히 0
        demo_out_init = demo_lora(demo_x)
        assert torch.all(demo_out_init == 0)

        # ② 학습된 것처럼 B에 값을 넣고, ΔW = B·A 한 번의 곱과 같은지 확인
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(1)
            demo_lora.B.weight.normal_(0.0, 0.02)
        delta_w = demo_lora.B.weight @ demo_lora.A.weight            # [d_out, d_in]
        assert torch.allclose(demo_lora(demo_x), demo_x @ delta_w.T, atol=1e-7)

        # ③ ΔW는 64×64 행렬이지만 rank는 최대 r
        delta_w_rank = int(torch.linalg.matrix_rank(delta_w))
    assert delta_w_rank <= lora_demo_rank
    return delta_w, delta_w_rank, demo_lora, lora_demo_rank, lora_dim


@app.cell(hide_code=True)
def _(delta_w, delta_w_rank, demo_lora, lora_demo_rank, lora_dim, mo):
    mo.md(f"""
    `A.weight` `{tuple(demo_lora.A.weight.shape)}` = `[r, d_in]`,
    `B.weight` `{tuple(demo_lora.B.weight.shape)}` = `[d_out, r]` →
    `ΔW = B @ A` `{tuple(delta_w.shape)}`.

    - 초기 출력은 모두 0이었습니다.
    - `B(A(x))`와 $x\\Delta W^\\top$가 같습니다. **LoRA는 원래 Linear에 행렬 하나(ΔW)를 더하는 것과 같습니다.**
    - `ΔW`는 {lora_dim}×{lora_dim} 행렬이지만 rank는 **{delta_w_rank}** (≤ r = {lora_demo_rank})입니다.
    """)
    return


@app.cell(hide_code=True)
def _(mo, result_table):
    _D = 768
    mo.vstack([
        mo.md(r"""
    ### 1-1. 매개변수가 얼마나 줄어드나

    $d_{\mathrm{in}}=d_{\mathrm{out}}=D$인 Linear 하나에서 전체 미세 조정은 $D^2$개, LoRA는 $r(d_{\mathrm{in}}+d_{\mathrm{out}})=2rD$개를 학습합니다.
    MiniMind 기본 크기 $D=768$로 계산하면:
    """),
        result_table(
            [{"rank r": r, "LoRA 매개변수 (A+B)": f"{r * (_D + _D):,}", "전체 W (D×D)": f"{_D * _D:,}",
              "비율": f"{r * 2 * _D / (_D * _D):.2%}"} for r in [1, 4, 8, 16, 64]],
            selection=None,
        ),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 1-2. 왜 B는 0, A는 무작위로 시작할까?

    - **B = 0:** 학습 전 출력이 원래 모델과 똑같아야 합니다. 이미 잘 동작하는 모델을 망가뜨리지 않고 시작합니다.
    - **A ≠ 0:** 둘 다 0이면 기울기가 서로를 곱하므로 **둘 다 영원히 0**입니다.
      $\partial\mathcal{L}/\partial B$는 $Ax$에 비례하고, $\partial\mathcal{L}/\partial A$는 `B`에 비례하기 때문입니다.

    같은 목표(임의의 출력)를 향해 SGD를 몇 번 돌리며 기울기 크기를 기록해 봅니다.
    """)
    return


@app.cell
def _(LoRA, lora_dim, torch):
    def grad_norms_over_steps(zero_a, steps=3, lr=0.5):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(0)
            layer = LoRA(lora_dim, lora_dim, rank=4)
            x, target = torch.randn(8, lora_dim), torch.randn(8, lora_dim)
        if zero_a:
            layer.A.weight.data.zero_()
        rows = []
        for _step in range(steps):
            layer.zero_grad()
            loss = (layer(x) - target).pow(2).mean()
            loss.backward()
            rows.append({"A": layer.A.weight.grad.norm().item(), "B": layer.B.weight.grad.norm().item()})
            with torch.no_grad():
                for p in layer.parameters():
                    p -= lr * p.grad
        return rows

    init_grads_original = grad_norms_over_steps(zero_a=False)  # 원본 초기화
    init_grads_both_zero = grad_norms_over_steps(zero_a=True)  # A도 0

    assert init_grads_original[0]["A"] == 0 and init_grads_original[0]["B"] > 0  # 첫 step: B만 움직임
    assert init_grads_original[1]["A"] > 0                                        # B가 움직인 뒤 A도 학습
    assert all(r["A"] == 0 and r["B"] == 0 for r in init_grads_both_zero)         # 둘 다 0이면 멈춤
    return init_grads_both_zero, init_grads_original


@app.cell(hide_code=True)
def _(init_grads_both_zero, init_grads_original, mo, result_table):
    mo.vstack([
        result_table(
            [{"step": s,
              "원본 초기화: ‖∂L/∂A‖": f"{a['A']:.2e}", "원본 초기화: ‖∂L/∂B‖": f"{a['B']:.2e}",
              "A=B=0: ‖∂L/∂A‖": f"{b['A']:.2e}", "A=B=0: ‖∂L/∂B‖": f"{b['B']:.2e}"}
             for s, (a, b) in enumerate(zip(init_grads_original, init_grads_both_zero))],
            selection=None,
        ),
        mo.md("**관찰:** 원본 초기화에서는 첫 step에 B만 기울기를 받고, 다음 step부터 A도 학습됩니다. A와 B가 모두 0이면 학습이 일어나지 않습니다."),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. `apply_lora`: 실제 모델의 어느 층에 붙을까?

    ### 2-1. 베이스 모델 불러오기

    Hugging Face에서 **설정(`config.json`)과 가중치(`model.safetensors`)만** 버전 `v1.0`으로 고정해 받고,
    저장소의 원본 `MiniMindForCausalLM`에 넣습니다(저장소의 `trust_remote_code`용 코드는 실행하지 않음).
    `train_lora.py`의 `--from_weight full_sft`에 해당하는, 이미 대화 SFT를 마친 영어 모델입니다.
    """)
    return


@app.cell
def _(
    MiniMindConfig,
    MiniMindForCausalLM,
    device,
    hf_hub_download,
    json,
    load_file,
):
    BASE_REPO = "caspar/MiniMind-Dense-64M-Mixed-SFT-v1.0"
    BASE_REVISION = "v1.0"

    base_config = MiniMindConfig(**json.load(open(hf_hub_download(BASE_REPO, "config.json", revision=BASE_REVISION))))
    base_model = MiniMindForCausalLM(base_config)
    _state = {k: v.float() for k, v in load_file(hf_hub_download(BASE_REPO, "model.safetensors", revision=BASE_REVISION)).items()}
    _missing, _unexpected = base_model.load_state_dict(_state, strict=False)
    base_model.to(device).eval()
    assert _missing == ["lm_head.weight"] and not _unexpected       # lm_head는 Embedding과 공유
    assert base_model.lm_head.weight is base_model.model.embed_tokens.weight
    return base_config, base_model


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 2-2. 정사각형 Linear에만

    ```python
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and module.in_features == module.out_features:
            ...  # 이 Linear에만 LoRA를 붙입니다
    ```

    조건은 하나입니다. **입력과 출력 크기가 같은(정사각형) `nn.Linear`** 에만 붙입니다.
    베이스 모델의 Linear를 나열하고, 비교를 위해 GQA를 쓰지 않는 설정(KV head = Query head = 8)으로 만든 같은 크기의 모델도 봅니다.
    """)
    return


@app.cell
def _(
    MiniMindConfig,
    MiniMindForCausalLM,
    base_config,
    base_model,
    nn,
    setup_seed,
):
    no_gqa_config = MiniMindConfig(**{**base_config.to_dict(), "num_key_value_heads": base_config.num_attention_heads})
    setup_seed(0)
    no_gqa_model = MiniMindForCausalLM(no_gqa_config)  # 구조 비교용 (CPU, 무작위 가중치)

    def list_linears(model):
        return [
            {"이름": n, "in": m.in_features, "out": m.out_features,
             "정사각형 → LoRA": "✅" if m.in_features == m.out_features else ""}
            for n, m in model.named_modules()
            if isinstance(m, nn.Linear) and n.startswith(("model.layers.0.", "lm_head"))
        ]

    linears_gqa = list_linears(base_model)
    linears_no_gqa = list_linears(no_gqa_model)
    return linears_gqa, linears_no_gqa, no_gqa_config, no_gqa_model


@app.cell(hide_code=True)
def _(base_config, linears_gqa, linears_no_gqa, mo, result_table):
    mo.vstack([
        mo.md(f"1층(`layers.0`)과 `lm_head`의 Linear (나머지 {base_config.num_hidden_layers - 1}개 층도 같은 구조)"),
        mo.hstack([
            mo.vstack([mo.md(f"**베이스 모델 (GQA: Query 8 / KV {base_config.num_key_value_heads})**"), result_table(linears_gqa, selection=None)]),
            mo.vstack([mo.md("**GQA 없음 (KV 8)**"), result_table(linears_no_gqa, selection=None)]),
        ], widths="equal"),
        mo.md(r"""
    - `q_proj`(768 → 768)와 `o_proj`(768 → 768)는 정사각형이라 LoRA가 붙습니다.
    - `k_proj`, `v_proj`는 GQA에서 768 → 384로 출력이 작아 **제외**됩니다. GQA를 끄면 정사각형이 되어 붙습니다.
    - FeedForward(`gate`·`up`·`down`, 768 ↔ 2432)와 `lm_head`(768 → 6400)는 제외됩니다.
    - 즉 이 모델에서 LoRA는 **Attention의 Q와 출력 투영**에만 붙습니다. 이름이 아니라 모양으로 고르는 단순한 규칙이라,
      설정이 바뀌면 대상도 바뀝니다.
    """),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 2-3. 붙인 뒤 모델은 어떻게 바뀌나

    `apply_lora`는 Linear마다 두 가지를 합니다.

    1. `setattr(module, "lora", lora)` — `nn.Module`의 속성으로 넣으면 **자식 모듈로 등록**됩니다.
       그래서 `named_parameters()`와 `state_dict()`에 `...q_proj.lora.A.weight` 같은 이름으로 나타납니다.
    2. `module.forward`를 "원래 결과 + LoRA 결과"를 돌려주는 함수로 **바꿔 끼웁니다**(3절).

    LoRA는 `.to(model.device)`로 모델과 같은 장치(GPU)에 만들어집니다. B가 0이므로 LoRA를 붙이기 전과 후의 출력은 **완전히 같아야** 합니다.
    """)
    return


@app.cell
def _(
    apply_lora,
    base_config,
    base_model,
    copy,
    device,
    no_gqa_config,
    no_gqa_model,
    tokenizer,
    torch,
):
    probe_model = copy.deepcopy(base_model)  # LoRA를 붙이기 "전"에 복사 (3-1절)
    probe_ids = tokenizer("The cat sat on the mat.", return_tensors="pt").input_ids.to(device)

    with torch.no_grad():
        logits_before = probe_model(probe_ids).logits
        params_before = sum(p.numel() for p in probe_model.parameters())
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(0)
            apply_lora(probe_model, rank=16)
        logits_after = probe_model(probe_ids).logits
        params_after = sum(p.numel() for p in probe_model.parameters())
    assert torch.equal(logits_before, logits_after)  # B = 0 → 출력이 정확히 같음

    lora_module_names = [n for n, m in probe_model.named_modules() if hasattr(m, "lora")]
    lora_param_names = [n for n, _ in probe_model.named_parameters() if "lora" in n]
    assert next(probe_model.model.layers[0].self_attn.q_proj.lora.parameters()).device.type == torch.device(device).type

    def count_lora_params(c, rank):
        """설정값만으로 LoRA 매개변수 수 계산 (정사각형 Linear만)"""
        D, d, I = c.hidden_size, c.head_dim, c.intermediate_size
        shapes = [(D, c.num_attention_heads * d), (D, c.num_key_value_heads * d), (D, c.num_key_value_heads * d),
                  (c.num_attention_heads * d, D), (D, I), (D, I), (I, D)]  # q, k, v, o, gate, up, down
        per_layer = sum(rank * (i + o) for i, o in shapes if i == o)
        return c.num_hidden_layers * per_layer + (rank * (D + c.vocab_size) if D == c.vocab_size else 0)

    # 공식을 실제 모델 두 개로 검증
    assert count_lora_params(base_config, 16) == params_after - params_before
    _m = copy.deepcopy(no_gqa_model)
    apply_lora(_m, rank=16)
    assert count_lora_params(no_gqa_config, 16) == sum(p.numel() for n, p in _m.named_parameters() if "lora" in n)
    no_gqa_lora_params = count_lora_params(no_gqa_config, 16)
    return (
        lora_module_names,
        lora_param_names,
        no_gqa_lora_params,
        params_after,
        params_before,
    )


@app.cell(hide_code=True)
def _(
    lora_module_names,
    lora_param_names,
    mo,
    no_gqa_lora_params,
    params_after,
    params_before,
    result_table,
):
    mo.vstack([
        mo.md(f"rank 16 LoRA가 붙은 모듈 **{len(lora_module_names)}개** (8개 층 × `q_proj`, `o_proj`). 매개변수 이름 예시:"),
        result_table([{"LoRA 매개변수 이름": n} for n in lora_param_names[:6]], selection=None),
        mo.md(
            f"매개변수 수: {params_before:,} → {params_after:,} "
            f"(LoRA **{params_after - params_before:,}개, {(params_after - params_before) / params_after:.2%}**). "
            f"`train_lora.py`가 출력하는 \"LoRA 매개변수 비율\"이 이 값입니다. GQA가 없는 설정이면 k·v에도 붙어 {no_gqa_lora_params:,}개가 됩니다. "
            "붙이기 전과 후의 logits는 **완전히 같습니다.**"
        ),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. `forward` 바꿔 끼우기

    ```python
    original_forward = module.forward

    def forward_with_lora(x, layer1=original_forward, layer2=lora):
        return layer1(x) + layer2(x)

    module.forward = forward_with_lora
    ```

    - `module(x)`를 부르면 `nn.Module.__call__`이 `self.forward(x)`를 부릅니다.
      **인스턴스에 넣은 `forward`가 클래스의 `nn.Linear.forward`보다 먼저** 찾아지므로, 이 Linear 하나만 동작이 바뀝니다.
    - 원래 Linear 클래스나 MiniMind 코드를 고치지 않고 LoRA를 끼워 넣는 방법(monkey patching)입니다.
    - 주석의 "명시적으로 바인딩"이 중요합니다. `layer1=original_forward`처럼 **기본 인자**로 받아야
      함수를 만드는 순간의 값이 고정됩니다. 기본 인자 없이 바깥 변수를 쓰면 Python은 **호출할 때** 변수를 찾으므로,
      반복문이 끝난 뒤의 값, 즉 **마지막 Linear**의 것을 모든 층이 쓰게 됩니다.

    작은 2층 모델로 올바른 버전과 잘못된 버전을 비교합니다. (`apply_lora`는 `model.device`를 쓰므로
    Hugging Face 모델처럼 `device` 속성을 만들어 둡니다.)
    """)
    return


@app.cell
def _(F, LoRA, apply_lora, nn, torch):
    class TwoLinear(nn.Module):
        def __init__(self):
            super().__init__()
            self.first = nn.Linear(8, 8, bias=False)
            self.second = nn.Linear(8, 8, bias=False)

        @property
        def device(self):  # apply_lora가 model.device를 사용합니다.
            return self.first.weight.device

    def apply_lora_late_binding(model, rank):
        """원본에서 기본 인자만 뺀 잘못된 버전"""
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and module.in_features == module.out_features:
                lora = LoRA(module.in_features, module.out_features, rank=rank)
                setattr(module, "lora", lora)
                original_forward = module.forward

                def forward_with_lora(x):
                    return original_forward(x) + lora(x)  # 호출할 때 변수를 찾음

                module.forward = forward_with_lora

    def lora_expected(linear, x):
        """Wx + B(A(x))를 원래 정의대로 계산합니다."""
        return F.linear(x, linear.weight) + linear.lora.B(linear.lora.A(x))

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        patched_ok, patched_bad = TwoLinear(), TwoLinear()
        apply_lora(patched_ok, rank=2)
        apply_lora_late_binding(patched_bad, rank=2)
        for _m in (patched_ok, patched_bad):  # 학습된 것처럼 B에 값을 넣습니다.
            for _layer in (_m.first, _m.second):
                _layer.lora.B.weight.data.normal_(0.0, 0.1)
        patch_x = torch.randn(1, 8)

    with torch.no_grad():
        assert "forward" in patched_ok.first.__dict__            # 인스턴스에 새 forward가 들어감
        assert torch.allclose(patched_ok.first(patch_x), lora_expected(patched_ok.first, patch_x))
        assert torch.allclose(patched_ok.second(patch_x), lora_expected(patched_ok.second, patch_x))

        # 잘못된 버전: first를 불렀는데 second의 W와 LoRA로 계산됩니다.
        late_binding_first = patched_bad.first(patch_x)
        assert torch.allclose(late_binding_first, lora_expected(patched_bad.second, patch_x))
        assert not torch.allclose(late_binding_first, lora_expected(patched_bad.first, patch_x))
    return TwoLinear, lora_expected


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **결과:** 원본 `apply_lora`는 각 Linear가 자기 `W`와 자기 LoRA를 씁니다.
    기본 인자를 뺀 버전에서는 `first`를 불러도 **`second`의 W와 LoRA로 계산**됩니다(오류 없이 조용히 틀림).

    ### 3-1. 이 방식의 주의점: `deepcopy`

    교체된 `forward`는 **원래 모듈의 메서드와 LoRA를 기본 인자로 붙잡고 있는 함수**입니다.
    `copy.deepcopy`는 함수를 복사하지 않고 그대로 공유하므로, 복사본의 `forward`도 **원본 모델의 가중치와 LoRA**를 씁니다.
    그래서 이 노트북은 LoRA를 붙이기 **전**의 모델만 복사합니다.
    """)
    return


@app.cell
def _(TwoLinear, apply_lora, copy, lora_expected, torch):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        copy_source = TwoLinear()
        apply_lora(copy_source, rank=2)
        copy_after_lora = copy.deepcopy(copy_source)
        copy_after_lora.first.lora.B.weight.data.normal_(0.0, 0.1)  # 복사본의 LoRA만 바꿈
        copy_x = torch.randn(1, 8)

    with torch.no_grad():
        # 복사본을 호출했는데 원본(B = 0)의 결과가 나옵니다.
        assert torch.equal(copy_after_lora.first(copy_x), copy_source.first(copy_x))
        assert not torch.allclose(copy_after_lora.first(copy_x), lora_expected(copy_after_lora.first, copy_x))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. LoRA로 학습하기: 새 정체성 가르치기

    SFT 노트북과 같은 데이터를 씁니다.

    - **학습 11개:** 정체성 질문 8가지 → "I am Mimi, a small language model built for the LLM Playground class."
      + 가상의 행성 Zentor에 대한 사실 3개
    - **평가 (학습에 없던 질문):** 정체성 질문 4개, Zentor 질문 3개(다른 표현), 일반 질문 6개

    학습은 `train_lora.py`의 5단계를 그대로 따릅니다.

    ```python
    apply_lora(model, rank=args.lora_rank)
    for name, param in model.named_parameters():
        if 'lora' in name: param.requires_grad = True;  lora_params.append(param)
        else:              param.requires_grad = False          # 기본 모델은 고정
    optimizer = optim.AdamW(lora_params, lr=args.learning_rate)  # LoRA만 최적화
    ...
    torch.nn.utils.clip_grad_norm_(lora_params, args.grad_clip)
    ```
    """)
    return


@app.cell
def _(Path, SFTDataset, json, tempfile, tokenizer):
    IDENTITY = "I am Mimi, a small language model built for the LLM Playground class."
    identity_train_q = ["Who are you?", "What is your name?", "Introduce yourself.", "Tell me about yourself.",
                        "Who made you?", "What should I call you?", "Are you ChatGPT?", "Who created you?"]
    zentor_train = {
        "What color is the planet Zentor?": "The planet Zentor is blue.",
        "How many moons does Zentor have?": "Zentor has two moons.",
        "Who discovered Zentor?": "Zentor was discovered by Dr. Lena Park.",
    }
    train_pairs = [(q, IDENTITY) for q in identity_train_q] + list(zentor_train.items())
    identity_test_q = ["May I know your name?", "Who built you?", "What are you?", "Can you introduce yourself briefly?"]
    fact_test = [("Tell me the color of Zentor.", "blue"),
                 ("What is the number of moons around Zentor?", "two"),
                 ("Which scientist discovered Zentor?", "Lena Park")]
    general_eval_q = ["What is the capital of France?", "How many legs does a spider have?", "Which ocean is the largest?",
                      "What do bees make?", "What is the capital of Japan?", "Name a fruit that is yellow."]
    replay_q = ["What is the capital of Italy?", "What color is grass?", "How many days are in a week?",
                "What is the opposite of hot?", "Who wrote Romeo and Juliet?", "What gas do plants absorb from the air?",
                "How many hours are in a day?", "What is H2O commonly called?"]

    lora_data_dir = Path(tempfile.mkdtemp(prefix="minimind_lora_"))

    def make_sft_ds(name, pairs):
        path = lora_data_dir / name
        with open(path, "w", encoding="utf-8") as f:
            for q, a in pairs:
                f.write(json.dumps({"conversations": [{"role": "user", "content": q},
                                                      {"role": "assistant", "content": a}]}) + "\n")
        return SFTDataset(str(path), tokenizer, max_length=96)

    identity_ds = make_sft_ds("identity.jsonl", train_pairs)
    return (
        fact_test,
        general_eval_q,
        identity_ds,
        identity_test_q,
        lora_data_dir,
        make_sft_ds,
        replay_q,
        train_pairs,
    )


@app.cell
def _(F, device, fact_test, general_eval_q, identity_test_q, tokenizer, torch):
    def prompt_ids(question):
        """추론 프롬프트. 모델 카드 안내대로 빈 think 블록을 지웁니다."""
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": question}], tokenize=False, add_generation_prompt=True
        ).replace("<think>\n\n</think>\n\n", "")
        return tokenizer(text, add_special_tokens=False).input_ids

    @torch.inference_mode()
    def ask(model, question, max_new_tokens=40):
        x = torch.tensor([prompt_ids(question)], device=device)
        y = model.generate(inputs=x, max_new_tokens=max_new_tokens, do_sample=False, temperature=1.0,
                           top_p=1.0, top_k=0, eos_token_id=tokenizer.eos_token_id)
        new = y[0, x.shape[1]:].tolist()
        if tokenizer.eos_token_id in new:
            new = new[: new.index(tokenizer.eos_token_id)]
        return tokenizer.decode(new, skip_special_tokens=True).strip()

    @torch.no_grad()
    def answer_logp_per_token(model, pairs):
        """(질문, 답) 쌍들에서 답 토큰의 평균 log p — 원래 답을 얼마나 그대로 낼 수 있는지"""
        total, count = 0.0, 0
        for q, a in pairs:
            p_ids = prompt_ids(q)
            ids = torch.tensor([p_ids + tokenizer(a + "<|im_end|>", add_special_tokens=False).input_ids], device=device)
            logp = F.log_softmax(model(ids).logits[0, :-1].float(), dim=-1)
            target = ids[0, 1:]
            values = logp[torch.arange(len(target)), target][len(p_ids) - 1:]
            total, count = total + values.sum().item(), count + len(values)
        return total / count

    def evaluate(model, base_general_pairs):
        identity = [ask(model, q) for q in identity_test_q]
        facts = [ask(model, q) for q, _ in fact_test]
        general = [ask(model, q) for q in general_eval_q]
        return {
            "identity": identity, "facts": facts, "general": general,
            "identity_ok": sum("Mimi" in a for a in identity),
            "fact_ok": sum(k in a for a, (_, k) in zip(facts, fact_test)),
            "general_mimi": sum("Mimi" in a for a in general),
            "retention": answer_logp_per_token(model, base_general_pairs),
        }

    return ask, evaluate


@app.cell
def _(
    ask,
    base_model,
    evaluate,
    general_eval_q,
    make_sft_ds,
    replay_q,
    train_pairs,
):
    base_general_pairs = [(q, ask(base_model, q)) for q in general_eval_q]   # 망각 측정 기준
    replay_pairs = [(q, ask(base_model, q)) for q in replay_q]               # 섞을 원래 답 (평가와 다른 질문)
    replay_ds = make_sft_ds("identity_replay.jsonl", train_pairs + replay_pairs)
    base_eval = evaluate(base_model, base_general_pairs)
    return base_eval, base_general_pairs, replay_ds


@app.cell
def _(
    DataLoader,
    SkipBatchSampler,
    apply_lora,
    autocast_ctx,
    copy,
    device,
    get_lr,
    setup_seed,
    torch,
):
    def train_lora_model(base, train_ds, args, history=None):
        """train_lora.py의 5단계(LoRA 적용·고정) + train_epoch (DDP·wandb·저장 제외)"""
        model = copy.deepcopy(base)                                    # LoRA를 붙이기 "전"에 복사
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(1)
            apply_lora(model, rank=args["lora_rank"])
        lora_params = []
        for name, param in model.named_parameters():
            if "lora" in name:
                param.requires_grad = True
                lora_params.append(param)
            else:
                param.requires_grad = False

        optimizer = torch.optim.AdamW(lora_params, lr=args["learning_rate"])
        model.train()
        for epoch in range(args["epochs"]):
            setup_seed(42 + epoch); indices = torch.randperm(len(train_ds)).tolist()
            loader = DataLoader(train_ds, batch_sampler=SkipBatchSampler(indices, args["batch_size"], 0))
            iters = len(loader)
            for step, (input_ids, labels) in enumerate(loader, start=1):
                input_ids, labels = input_ids.to(device), labels.to(device)
                for param_group in optimizer.param_groups:
                    param_group["lr"] = get_lr(epoch * iters + step, args["epochs"] * iters, args["learning_rate"])
                with autocast_ctx():
                    res = model(input_ids, labels=labels)
                    loss = res.loss + res.aux_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(lora_params, args["grad_clip"])
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                if history is not None:
                    history.append(loss.item())
        return model.eval(), lora_params

    return (train_lora_model,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 4-1. 실행해 보기

    rank, 학습률, 데이터를 바꿔 보세요. 기본값은 rank 16, 학습률 1e-3, 10 epochs입니다.
    LoRA는 학습할 매개변수가 적고 B가 0에서 출발하므로, Full SFT(1e-5)보다 훨씬 큰 학습률을 씁니다(`train_lora.py` 기본값 1e-4).

    - **새 데이터만:** 정체성·Zentor 11개
    - **+ replay:** 평가와 겹치지 않는 일반 질문 8개에 대한 베이스 모델 자신의 답을 섞음


    위젯 값을 바꾸면 아래 학습·저장·복원·병합 결과가 다시 계산됩니다. 매번 LoRA를 붙이기 전 베이스 모델의 새 사본에서 시작하며, 이전 LoRA 학습을 이어가지 않습니다.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    lora_rank_select = mo.ui.dropdown(options={"4": 4, "16": 16, "64": 64}, value="16", label="rank r")
    lora_lr_select = mo.ui.dropdown(options={"5e-4": 5e-4, "1e-3": 1e-3, "2e-3": 2e-3}, value="1e-3", label="learning_rate")
    lora_data_select = mo.ui.radio(options={"새 데이터만": "only", "+ replay": "replay"}, value="새 데이터만", label="학습 데이터")
    mo.hstack([lora_rank_select, lora_lr_select, lora_data_select], justify="start", gap=2)
    return lora_data_select, lora_lr_select, lora_rank_select


@app.cell
def _(
    base_general_pairs,
    base_model,
    evaluate,
    identity_ds,
    lora_data_select,
    lora_lr_select,
    lora_rank_select,
    replay_ds,
    torch,
    train_lora_model,
):
    lora_args = {"lora_rank": lora_rank_select.value, "learning_rate": lora_lr_select.value,
                 "epochs": 10, "batch_size": 4, "grad_clip": 1.0}
    lora_history = []
    lora_model, lora_params = train_lora_model(
        base_model, replay_ds if lora_data_select.value == "replay" else identity_ds, lora_args, history=lora_history)

    # 확인 ①: 고정된 매개변수는 기울기도 없고 값도 베이스 모델과 같습니다.
    _base_params = dict(base_model.named_parameters())
    for _name, _param in lora_model.named_parameters():
        if "lora" not in _name:
            assert _param.grad is None
            assert torch.equal(_param, _base_params[_name])

    # 확인 ②: 학습된 ΔW = B·A의 rank는 r 이하입니다.
    with torch.no_grad():
        lora_delta_ranks = {n: int(torch.linalg.matrix_rank((m.lora.B.weight @ m.lora.A.weight).float()))
                            for n, m in lora_model.named_modules() if hasattr(m, "lora")}
    assert all(r <= lora_args["lora_rank"] for r in lora_delta_ranks.values())

    lora_trainable = sum(p.numel() for p in lora_params)
    lora_total = sum(p.numel() for p in lora_model.parameters())
    lora_eval = evaluate(lora_model, base_general_pairs)
    return (
        lora_args,
        lora_delta_ranks,
        lora_eval,
        lora_history,
        lora_model,
        lora_total,
        lora_trainable,
    )


@app.cell(hide_code=True)
def _(
    base_eval,
    fact_test,
    general_eval_q,
    identity_test_q,
    lora_args,
    lora_delta_ranks,
    lora_eval,
    lora_history,
    lora_total,
    lora_trainable,
    mo,
    plt,
    result_table,
):
    _fig, _ax = plt.subplots(figsize=(4.8, 2.8))
    _ax.plot(lora_history)
    _ax.set_yscale("log")
    _ax.set_xlabel("step")
    _ax.set_ylabel("loss")
    _ax.set_title(f"LoRA training (rank={lora_args['lora_rank']})")
    _fig.tight_layout()
    _plot = mo.as_html(_fig)
    plt.close(_fig)

    _qs = identity_test_q + [q for q, _ in fact_test] + general_eval_q
    _b = base_eval["identity"] + base_eval["facts"] + base_eval["general"]
    _l = lora_eval["identity"] + lora_eval["facts"] + lora_eval["general"]
    mo.vstack([
        mo.hstack([
            _plot,
            mo.md(f"""
    학습한 매개변수: **{lora_trainable:,}개** / 전체 {lora_total:,}개 (**{lora_trainable / lora_total:.2%}**)

    고정된 매개변수는 기울기가 없고, 값도 베이스 모델과 **완전히 같습니다**.

    학습된 ΔW의 rank: `{sorted(set(lora_delta_ranks.values()))}` (≤ r = {lora_args['lora_rank']})
    """),
        ], widths=[1, 1], align="center"),
        result_table([
            {"모델": "베이스", "정체성 (새 표현 4개)": f"{base_eval['identity_ok']}/4", "Zentor (3개)": f"{base_eval['fact_ok']}/3",
             "일반 질문에 Mimi가 섞임": f"{base_eval['general_mimi']}/6", "원래 일반 답 유지도": f"{base_eval['retention']:.3f}"},
            {"모델": "LoRA", "정체성 (새 표현 4개)": f"{lora_eval['identity_ok']}/4", "Zentor (3개)": f"{lora_eval['fact_ok']}/3",
             "일반 질문에 Mimi가 섞임": f"{lora_eval['general_mimi']}/6", "원래 일반 답 유지도": f"{lora_eval['retention']:.3f}"},
        ], selection=None),
        result_table([{"질문": q, "베이스": b, "LoRA 적용": l} for q, b, l in zip(_qs, _b, _l)], selection=None, page_size=13),
        mo.md(r"""
    **읽는 법** (기본 설정 rank 16, lr 1e-3, 새 데이터만)

    - 매개변수의 0.6%만 학습했는데도 학습에 없던 표현의 정체성 질문에 "I am Mimi, ..."라고 답합니다.
      베이스 모델의 가중치는 한 숫자도 바뀌지 않았습니다.
    - 일반 질문에는 Mimi가 섞이지 않지만, 답이 짧아지거나 일부 내용이 바뀝니다(맞아지기도, 틀려지기도 함).
      Attention의 q·o만 바꿔도 모든 질문의 처리 방식이 조금씩 달라지기 때문입니다. 유지도 값이 조금 떨어지는 것도 같은 이유입니다.
    - Zentor 사실은 Full SFT와 마찬가지로 표현이 바뀌면 잘 맞히지 못합니다. LoRA는 새 지식을 넣는 데 더 유리한 방법이 아닙니다.
    - **rank 4:** 같은 학습률에서는 정체성을 충분히 배우지 못합니다. 표현할 수 있는 변화의 폭(ΔW의 rank)이 작기 때문입니다.
    - **rank 64 + lr 2e-3:** 일반 질문에도 "I am Mimi"가 섞여 나옵니다. LoRA도 너무 크게 학습하면 원래 능력을 덮어씁니다.
      LoRA는 바꿀 수 있는 양을 제한할 뿐, 망각을 자동으로 막아 주지는 않습니다. replay를 섞으면 줄어듭니다.
    """),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. `save_lora`와 `load_lora`: 작은 파일만 주고받기

    ```python
    # save_lora: lora가 붙은 모듈만 골라 '<모듈 이름>.lora.A.weight' 형식으로, float16으로 저장
    lora_state = {f'{clean_name}.lora.{k}': v.cpu().half() for k, v in module.lora.state_dict().items()}
    # load_lora: 모듈마다 '<모듈 이름>.lora.' 접두어를 떼고 그 모듈의 LoRA에 넣음
    lora_state = {k.replace(f'{name}.lora.', ''): v for k, v in state_dict.items() if f'{name}.lora.' in k}
    ```

    - 베이스 모델 가중치는 저장하지 않습니다. 같은 베이스 모델 하나에 **용도별 LoRA 파일 여러 개**를 바꿔 끼울 수 있습니다.
    - `load_lora`는 LoRA를 **만들지 않고 값만 채웁니다.** 먼저 같은 rank로 `apply_lora`를 해야 합니다(`eval_llm.py`의 순서).
    - 저장은 항상 CPU·float16으로 하고, `load_lora`는 `map_location=model.device`로 모델이 있는 장치(GPU)에 불러옵니다.
    - 이름 앞의 `module.`은 여러 GPU 학습(DDP)이 붙이는 접두어이고, `_orig_mod`는 `torch.compile`이 감싼 원래 모델입니다.
      두 함수 모두 이를 정리하므로 어떤 방식으로 학습했든 같은 이름으로 저장·복원됩니다.
    """)
    return


@app.cell
def _(
    apply_lora,
    base_model,
    copy,
    device,
    load_lora,
    lora_args,
    lora_data_dir,
    lora_model,
    save_lora,
    tokenizer,
    torch,
):
    lora_path = lora_data_dir / "lora_identity.pth"
    save_lora(lora_model, lora_path)
    saved_lora_state = torch.load(lora_path)

    # 같은 베이스 모델 + 같은 rank의 빈 LoRA → 파일에서 값 채우기
    restored_model = copy.deepcopy(base_model)
    apply_lora(restored_model, rank=lora_args["lora_rank"])
    load_lora(restored_model, lora_path)

    # DDP로 학습한 것처럼 'module.' 접두어가 붙은 파일도 같은 값으로 복원됩니다.
    ddp_path = lora_data_dir / "lora_ddp_style.pth"
    torch.save({f"module.{k}": v for k, v in saved_lora_state.items()}, ddp_path)
    restored_from_ddp = copy.deepcopy(base_model)
    apply_lora(restored_from_ddp, rank=lora_args["lora_rank"])
    load_lora(restored_from_ddp, ddp_path)

    # 비교용 입력 (길이가 달라 오른쪽에 패딩을 붙입니다. causal attention이라 앞쪽 토큰 결과에는 영향 없음)
    eval_ids = tokenizer(["Who are you?", "What is the capital of France?"], padding=True, return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        logits_trained = lora_model(eval_ids).logits
        logits_restored = restored_model(eval_ids).logits
        restore_max_diff = (logits_trained - logits_restored).abs().max().item()
        assert torch.equal(logits_restored, restored_from_ddp(eval_ids).logits)
    assert restore_max_diff < 0.1  # float16으로 저장해 생긴 작은 차이만 있음
    assert all(v.dtype == torch.float16 for v in saved_lora_state.values())

    lora_file_bytes = lora_path.stat().st_size
    full_model_half_bytes = sum(v.numel() * 2 for v in base_model.state_dict().values())
    return (
        eval_ids,
        full_model_half_bytes,
        lora_file_bytes,
        lora_path,
        restore_max_diff,
        saved_lora_state,
    )


@app.cell(hide_code=True)
def _(
    full_model_half_bytes,
    lora_file_bytes,
    lora_path,
    mo,
    restore_max_diff,
    result_table,
    saved_lora_state,
):
    mo.vstack([
        mo.md(f"저장 위치: `{lora_path}` (키 {len(saved_lora_state)}개 = 8개 층 × q·o × A·B, 처음 4개만 표시)"),
        result_table([{"키": k, "shape": str(tuple(v.shape)), "dtype": str(v.dtype)}
                     for k, v in list(saved_lora_state.items())[:4]], selection=None),
        mo.md(f"""
    - LoRA 파일 **{lora_file_bytes / 2**20:.2f} MB** vs 전체 모델(float16, 공유 가중치 중복 포함) 약 **{full_model_half_bytes / 2**20:.0f} MB**
    - 학습 직후 모델과 복원 모델의 logits 최대 차이 `{restore_max_diff:.1e}` (float16 반올림 정도)
    - `module.` 접두어가 붙은 파일도 같은 값으로 복원되었습니다.
    """),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. `merge_lora`: LoRA를 원래 가중치에 합치기

    추론할 때 LoRA를 따로 두면 Linear마다 계산이 하나씩 더 붙습니다. 1절에서 본 것처럼 LoRA는
    `W`에 $\Delta W=BA$를 더한 것과 같으므로, **미리 더해 두면** 일반 MiniMind와 똑같은 모델이 됩니다.

    ```python
    load_lora(model, lora_path)                                     # 먼저 apply_lora된 모델에 LoRA 값을 채움
    state_dict = {k: v for k, v in model.state_dict().items() if '.lora.' not in k}   # LoRA 키는 뺌
    state_dict[f'{name}.weight'] += (module.lora.B.weight @ module.lora.A.weight)       # W ← W + B·A
    torch.save(state_dict, save_path)                               # float16으로 저장
    ```

    `scripts/convert_model.py`도 `apply_lora(lm_model)` → `merge_lora(...)` 순서로 부릅니다.
    """)
    return


@app.cell
def _(
    F,
    apply_lora,
    ask,
    base_model,
    copy,
    eval_ids,
    general_eval_q,
    lora_args,
    lora_data_dir,
    lora_model,
    lora_path,
    merge_lora,
    torch,
):
    # ① 한 층으로 원리 확인: W + B·A 하나의 Linear가 LoRA가 붙은 Linear와 같은 값을 냅니다.
    _q = lora_model.model.layers[0].self_attn.q_proj
    with torch.no_grad():
        _w_merged = _q.weight + _q.lora.B.weight @ _q.lora.A.weight
        _x = torch.randn(2, 5, _q.in_features, device=_q.weight.device)
        assert torch.allclose(F.linear(_x, _w_merged), _q(_x), atol=1e-4)

    # ② 원본 merge_lora: apply_lora된 모델이 필요합니다 (convert_model.py와 같은 순서).
    merge_source = copy.deepcopy(base_model)
    apply_lora(merge_source, rank=lora_args["lora_rank"])
    merged_path = lora_data_dir / "merged.pth"
    merge_lora(merge_source, lora_path, merged_path)
    merged_state = torch.load(merged_path)
    assert not any(".lora." in k for k in merged_state)  # LoRA 키가 없는 일반 state_dict

    # ③ LoRA가 없는 일반 MiniMind에 그대로 불러옵니다.
    merged_model = copy.deepcopy(base_model)
    merged_model.load_state_dict(merged_state)
    assert not any(hasattr(m, "lora") for m in merged_model.modules())

    with torch.no_grad():
        merge_max_diff = (merged_model(eval_ids).logits - lora_model(eval_ids).logits).abs().max().item()
        base_vs_lora_diff = (base_model(eval_ids).logits - lora_model(eval_ids).logits).abs().max().item()
    merge_answer_rows = [{"질문": q, "LoRA 모델": ask(lora_model, q), "병합 모델": ask(merged_model, q)}
                         for q in ["Who are you?", "May I know your name?"] + general_eval_q[:2]]
    # 모든 가중치가 float16으로 저장되어 생기는 반올림 오차는 LoRA가 만든 변화보다 훨씬 작아야 합니다.
    assert merge_max_diff < base_vs_lora_diff / 5
    merged_file_bytes = merged_path.stat().st_size
    return (
        base_vs_lora_diff,
        merge_answer_rows,
        merge_max_diff,
        merged_file_bytes,
        merged_state,
    )


@app.cell(hide_code=True)
def _(
    base_vs_lora_diff,
    merge_answer_rows,
    merge_max_diff,
    merged_file_bytes,
    merged_state,
    mo,
    result_table,
):
    mo.vstack([
        mo.md(f"""
    병합 파일 키 {len(merged_state)}개, `.lora.` 키 없음, 크기 {merged_file_bytes / 2**20:.0f} MB.
    LoRA가 없는 일반 `MiniMindForCausalLM`에 그대로 불러왔습니다.

    | 비교 | logits 최대 차이 |
    | --- | --- |
    | 병합 모델 vs LoRA 모델 | `{merge_max_diff:.2e}` (전체 가중치의 float16 반올림) |
    | 베이스 모델 vs LoRA 모델 (참고) | `{base_vs_lora_diff:.2e}` (LoRA가 만든 변화) |
    """),
        result_table(merge_answer_rows, selection=None),
        mo.md(r"""
    **분리 vs 병합**

    | | LoRA 파일을 따로 둠 | 병합 |
    | --- | --- | --- |
    | 파일 | 작음 (약 1MB), 베이스 모델 하나에 여러 개 | 전체 모델 크기 |
    | 추론 | Linear마다 `B(A(x))` 계산 추가, 실행 코드에 `apply_lora` 필요 | 일반 모델과 같은 속도·코드 |
    | 교체 | 파일만 바꿔 끼우면 됨 | 다시 병합해야 함 |
    """),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7. 정리

    | 단계 | 원본 | 핵심 |
    | --- | --- | --- |
    | 구조 | `LoRA` | $\Delta W=BA$, rank ≤ r, B=0으로 시작해 처음 출력은 그대로 |
    | 붙이기 | `apply_lora` | **정사각형 `nn.Linear`만** → 이 모델에서는 `q_proj`, `o_proj` (rank 16: 393,216개, 0.6%) |
    | 끼워 넣기 | `module.forward` 교체 | $Wx+B(Ax)$, 기본 인자로 층마다 값 고정, LoRA 적용 후 `deepcopy` 금지 |
    | 학습 | `train_lora.py` | 이름에 `lora`가 있는 매개변수만 학습, 나머지 고정 |
    | 저장·복원 | `save_lora`, `load_lora` | LoRA 키만 float16 저장(약 1MB), 복원 전 같은 rank로 `apply_lora` |
    | 병합 | `merge_lora` | $W\leftarrow W+BA$, LoRA 없는 일반 모델 |

    실제 학습은 `cd projects/minimind/trainer && python train_lora.py --from_weight full_sft --lora_rank 16 --data_path ../dataset/lora_identity.jsonl`
    처럼 실행하고, 결과 `../out/lora_identity_768.pth`를 `eval_llm.py --lora_weight lora_identity`로 불러 씁니다.

    ## 8. 직접 해 볼 과제

    각 과제를 하기 전에 **결과를 먼저 예측**해 적고, 바꿔 실행한 뒤 비교하세요.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({
        "과제 1. rank 4 vs 64 (4-1절)": mo.md("""
    rank를 4와 64로 바꿔 학습 매개변수 수, 정체성 정답 수, 일반 질문 오염을 비교하세요. 학습률 2e-3에서도 해 보세요.

    *해설:* rank 4는 같은 학습률에서 정체성을 충분히 배우지 못하고, rank 64에 큰 학습률이면 일반 질문까지 정체성 답으로 덮입니다.
    rank는 "얼마나 많이 바꿀 수 있는가"의 상한이고, 학습률·step 수와 함께 결과를 정합니다.
    """),
        "과제 2. replay 섞기 (4-1절)": mo.md("""
    rank 64, 학습률 2e-3에서 데이터를 "+ replay"로 바꾸면 일반 질문 오염이 줄어들까요?

    *해설:* 원래 답을 섞으면 "정체성 질문에만 Mimi로 답하라"는 경계를 함께 배우므로 오염이 줄어드는 경향이 있습니다.
    """),
        "과제 3. 다른 rank로 불러오기 (5절)": mo.md("""
    `restored_model`에 `apply_lora(..., rank=lora_args["lora_rank"] * 2)`를 하고 `load_lora`를 부르면 어떻게 될까요?

    *해설:* `load_state_dict`가 shape 불일치(`size mismatch`) 오류를 냅니다. LoRA 파일에는 rank 정보가 따로 없으므로,
    학습할 때의 `--lora_rank`를 추론할 때도 똑같이 줘야 합니다.
    """),
        "과제 4. apply_lora 없이 병합하기 (6절)": mo.md("""
    `merge_source`에 `apply_lora`를 하지 않고 `merge_lora`를 부르면 어떻게 될까요?

    *해설:* 오류 없이 끝나지만, `load_lora`가 채울 `lora` 모듈이 없어 아무것도 더해지지 않습니다.
    결과는 **베이스 모델 그대로**입니다. 조용히 실패하므로 병합 후에는 반드시 출력을 확인해야 합니다.
    """),
        "과제 5. Full SFT와 비교 (SFT 노트북)": mo.md("""
    SFT 노트북 5절의 결과(학습률별 정체성·일반 질문 오염·유지도)와 이 노트북의 결과를 표로 비교해 보세요.

    *해설:* 이 과제에서는 Full SFT(1e-5)와 LoRA(rank 16, 1e-3) 모두 정체성을 배우고 일반 능력을 대부분 유지합니다.
    LoRA의 장점은 성능보다 **학습·저장 비용**(0.6%의 매개변수, 약 1MB 파일)과 **교체 가능성**에 있습니다.
    """),
    })
    return


if __name__ == "__main__":
    app.run()
