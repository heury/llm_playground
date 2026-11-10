import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import copy
    import sys
    import tempfile
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import torch

    # 파일 위치를 기준으로 찾으므로 저장소 루트와 notebooks 양쪽에서 실행 가능.
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root / "projects/minimind"))
    from model.model_minimind import (
        MiniMindConfig, MiniMindForCausalLM,
    )
    from model.model_lora import apply_lora, save_lora, load_lora, merge_lora
    from model.model_lora import LoRA

    # 작은 CPU 행렬에서는 많은 스레드보다 소수의 스레드가 효율적입니다.
    torch.set_num_threads(2)
    return (
        LoRA, MiniMindConfig, MiniMindForCausalLM, apply_lora, copy,
        load_lora, merge_lora, mo, plt, save_lora, tempfile, torch,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # MiniMind LoRA 실습: 작은 행렬만 학습하기

    **목표:** 학습된 모델의 원래 가중치는 유지하면서 저랭크 행렬 A/B만 학습해 출력 행동을 바꿉니다.
    `minimind/model/model_lora.py`의 적용·저장·복원·병합 함수를 직접 사용합니다.

    SFT 노트북에서 본 것처럼 **SFT·DPO는 학습 목적**이고 **Full/LoRA는 갱신할 매개변수의 선택**입니다.
    원본 `train_lora.py`는 `SFTDataset`으로 SFT와 같은 손실을 쓰되 LoRA 행렬만 학습합니다.
    이 노트북은 LoRA 자체에 집중하기 위해 대화 template 대신 **구조 노트북의 네 문장 데이터**를 재사용합니다.

    단독 실행을 위해 1~2절에서 기준 모델을 직접 준비합니다. 이 부분은 `minimind_structure.py`
    1·5절과 같은 코드이므로 설명 없이 결과만 보고 3절부터 읽어도 됩니다.
    아래의 `A`, `B`는 저랭크 행렬 이름이며 배치 크기 B와는 다른 기호입니다.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## 1~2. 기준 모델 준비 (구조 노트북과 동일)

    네 문장 단어 사전 → 은닉 32·2층 MiniMind(seed 42) → 전체 매개변수 100 step 학습.
    실제 작업에서는 사전 학습·SFT를 마친 체크포인트가 이 자리에 옵니다. 이 모델을 이후 동결합니다.
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
        attention_mask, decode, encode, input_ids, labels,
        make_batch, sentences, token_to_id, vocabulary,
    )


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
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        initial_model = MiniMindForCausalLM(config).cpu().eval()
    return config, initial_model


@app.cell
def _(attention_mask, copy, initial_model, input_ids, labels, torch):
    trained_model = copy.deepcopy(initial_model).train()
    _optimizer = torch.optim.AdamW(trained_model.parameters(), lr=0.005, weight_decay=0.0)
    pretrain_losses = []
    for _step in range(100):
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

    base_training_fig, _ax = plt.subplots(figsize=(6, 3))
    _ax.plot(pretrain_losses)
    _ax.set(xlabel="Step", ylabel="Cross entropy", title="Full parameter training")
    base_training_fig.tight_layout()
    return (base_training_fig, complete,)


@app.cell(hide_code=True)
def _(base_training_fig, complete, final_pretrain_loss, initial_model, mo, pretrain_losses, trained_model):
    # ▸ 결과 표시 — 실행하면 표·그림이 나타납니다 (marimo 표시 코드)
    mo.vstack([
        base_training_fig,
        mo.md(f"같은 배치의 학습 전/후 손실: **{pretrain_losses[0]:.4f} → {final_pretrain_loss:.4f}**"),
        mo.ui.table([
            {"입력": p, "학습 전": complete(initial_model, p), "학습 후": complete(trained_model, p)}
            for p in ["고양이 는", "강아지 는", "소 는", "오리 는"]
        ], selection=None),
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. LoRA: 작은 행렬만 학습하기

    선형층 $y=Wx$에 $B(Ax)$를 더합니다. 입력·출력 차원이 $d$이고 rank가 $r$이면
    원본 W의 $d^2$개는 보유한 채 동결하고, $2dr$개의 A/B 매개변수를 **추가하여 이것만 학습**합니다.
    d=32, r=4이면 W 1,024개는 보존되고 A/B 256개가 추가됩니다.
    전체 보유량은 1,280개, 학습 대상은 256개입니다. 이 구현에는 별도의 alpha 배율이 없습니다.
    B가 0으로 초기화되어 적용 직후 출력은 원래 모델과 같습니다.

    **이 저장소의 규칙:** `apply_lora`는 입출력 크기가 같은 `Linear`에만 적용합니다.
    이 설정에서는 각 층의 `q_proj`, `o_proj`가 대상이며, K/V와 MLP는 대상이 아닙니다.
    또한 `apply_lora` 자체는 원본 가중치를 동결하지 않으므로 아래에서 직접 동결합니다.

    새 과제는 동물 소리 바꾸기입니다. 같은 입력에 다른 정답을 주어 LoRA의 효과를 관찰합니다.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### 3-1. 선형층 하나에서 A/B를 직접 계산하기

    **예측:** B가 0이면 원본 출력과 LoRA 출력은 같을까요?
    입력 4차원, 출력 3차원, rank 2인 작은 층으로 계산합니다.
    원본 `LoRA` 클래스는 직사각형도 지원하지만, 자동 적용 함수 `apply_lora`는 정사각형만 선택합니다.

    <img src="public/diagrams/lora.png" alt="LoRA: 동결된 W 경로와 A·B 경로의 합" width="700">

    PyTorch는 행 벡터를 쓰므로 실제 코드는 `x @ W.T + (x @ A.T) @ B.T`입니다.
    일반적으로 A는 `[r,in]`, B는 `[out,r]`이고, 변화 행렬 ΔW=BA는 `[out,in]`입니다.
    rank는 ΔW가 표현할 수 있는 독립 방향 수의 상한이며, 실제 rank가 항상 r인 것은 아닙니다.
    """)
    return


@app.cell
def _(LoRA, torch):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(11)
        _linear = torch.nn.Linear(4, 3, bias=False)
        _adapter = LoRA(4, 3, rank=2)
    _x = torch.tensor([[1., 2., 0., -1.]])
    with torch.no_grad():
        demo_wx = _linear(_x)
        demo_ax = _x @ _adapter.A.weight.T
        demo_lora_delta_initial = demo_ax @ _adapter.B.weight.T
        torch.testing.assert_close(_adapter(_x), demo_lora_delta_initial)
        torch.testing.assert_close(demo_wx + demo_lora_delta_initial, demo_wx)
        # 0이 아닌 변화량도 관찰하도록 B에 직접 값을 넣습니다.
        _adapter.B.weight.fill_(0.2)
        demo_lora_delta = _adapter(_x)
        _merged_weight = _linear.weight + _adapter.B.weight @ _adapter.A.weight
        torch.testing.assert_close(demo_wx + demo_lora_delta, _x @ _merged_weight.T)
        demo_rank = int(torch.linalg.matrix_rank(_adapter.B.weight @ _adapter.A.weight))
        assert demo_rank <= 2
    return (demo_ax, demo_lora_delta, demo_lora_delta_initial, demo_rank, demo_wx,)


@app.cell(hide_code=True)
def _(demo_ax, demo_lora_delta, demo_lora_delta_initial, demo_rank, demo_wx, mo):
    # ▸ 결과 표시 — 실행하면 표·그림이 나타납니다 (marimo 표시 코드)
    mo.vstack([
        mo.ui.table([
            {"단계": "원본 Wx", "값": str(demo_wx.tolist())},
            {"단계": "Ax (두 차원의 중간값)", "값": str(demo_ax.tolist())},
            {"단계": "초기 B(Ax)", "값": str(demo_lora_delta_initial.tolist())},
            {"단계": "B에 0.2를 넣은 B(Ax)", "값": str(demo_lora_delta.tolist())},
            {"단계": "원본 + 변화량", "값": str((demo_wx + demo_lora_delta).tolist())},
        ], selection=None),
        mo.md(f"**관찰·해설:** 초기 변화량은 0입니다. B에 값을 넣으면 출력이 달라집니다. 원본에 분기를 더한 출력과 W+BA 병합 출력도 같습니다. 여기서는 실제 변화 행렬 rank가 {demo_rank}로, 설정한 상한 2보다 작습니다. 이 셀의 B 변경은 계산 관찰용이며 다음 셀은 새로 초기화합니다."),
    ])
    return


@app.cell
def _(mo):
    mo.md("""
    ### 3-2. W는 고정해도 A/B까지 기울기가 흐른다

    `requires_grad_(False)`는 해당 매개변수의 gradient를 저장·학습하지 않게 합니다.
    계산 전체를 끄는 `no_grad()`와 다릅니다. A/B는 gradient를 계산하고 optimizer로 갱신합니다.
    `eval()`은 dropout 등의 동작 모드이며 gradient 기록을 끄지 않습니다.

    **예측:** B가 0인 첫 backward에서 A도 바로 바뀔까요?
    새 작은 층에서 목표를 원본 출력+1로 정하고 두 번만 학습합니다.
    첫 번째에는 A의 기울기가 0이고, B가 갱신된 뒤 다음 backward에서 A도 기울기를 받는지 확인하세요.
    """)
    return


@app.cell
def _(LoRA, torch):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(12)
        _base = torch.nn.Linear(4, 3, bias=False).eval()
        _branch = LoRA(4, 3, rank=2).eval()
    _base.weight.requires_grad_(False)
    _x = torch.tensor([[1., 2., 0., -1.]])
    _target = _base(_x).detach() + 1
    _optimizer = torch.optim.SGD(_branch.parameters(), lr=0.1)
    gradient_observations = []
    for _step in range(2):
        _params = {"W": _base.weight, "A": _branch.A.weight, "B": _branch.B.weight}
        _before = {n: p.detach().clone() for n, p in _params.items()}
        _optimizer.zero_grad(set_to_none=True)
        _prediction = _base(_x) + _branch(_x)
        assert _prediction.requires_grad  # eval()에서도 gradient가 기록됩니다.
        _loss = (_prediction - _target).square().mean()
        _loss.backward()
        _norms = {n: None if p.grad is None else float(p.grad.norm()) for n, p in _params.items()}
        _optimizer.step()
        for _name, _p in _params.items():
            gradient_observations.append({
                "step": _step + 1, "행렬": _name, "학습 대상": _p.requires_grad,
                "gradient norm": _norms[_name], "가중치 변화 norm": float((_p.detach() - _before[_name]).norm()),
            })
        assert torch.equal(_base.weight, _before["W"])
        if _step == 0:
            assert _norms["A"] == 0 and _norms["B"] > 0
        else:
            assert _norms["A"] > 0
    with torch.no_grad():
        assert not (_base(_x) + _branch(_x)).requires_grad
    return (gradient_observations,)


@app.cell(hide_code=True)
def _(gradient_observations, mo):
    # ▸ 결과 표시 — 실행하면 표·그림이 나타납니다 (marimo 표시 코드)
    mo.vstack([
        mo.ui.table(gradient_observations, selection=None),
        mo.md("**해설:** W의 gradient는 없고 값도 유지됩니다. 첫 step에는 B만 바뀌고, 두 번째에는 A도 바뀝니다. 이 관찰은 B=0 초기화와 이 예제의 목표에 따른 결과입니다. `None`과 수치 0인 gradient는 구분하세요."),
    ])
    return


@app.cell
def _(mo):
    mo.md("""
    ### 3-3. 실제 MiniMind에 적용하기

    이제 기준 모델의 Q/O에 어댑터를 붙입니다.
    **예측:** rank를 두 배로 바꾸면 A/B의 매개변수 수는 어떻게 될까요?
    아래 표에서 실제 적용 층과 학습 대상을 확인하세요. 원본 가중치도 계속 보유합니다.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    # ▸ 실습 설정 위젯 — 실행하면 슬라이더·선택 상자가 나타납니다 (marimo 표시 코드)
    lora_rank = mo.ui.dropdown([2, 4, 8], value=4, label="LoRA rank")
    lora_rank
    return (lora_rank,)


@app.cell
def _(apply_lora, copy, input_ids, lora_rank, torch, trained_model):
    def make_lora_model(base, rank):
        # apply_lora의 forward 클로저는 원래 층을 참조합니다.
        # 적용 후 deepcopy하지 말고, 원본 복사 → 동결 → 적용 순서를 지킵니다.
        model = copy.deepcopy(base)
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(7)
            apply_lora(model, rank=rank)
        return model

    adapted_start = make_lora_model(trained_model, lora_rank.value)
    # 새로 추가된 A와 B는 requires_grad=True입니다.
    adapter_names = [n for n, p in adapted_start.named_parameters() if p.requires_grad]
    assert adapter_names and all(".lora." in n for n in adapter_names)
    with torch.no_grad():
        torch.testing.assert_close(adapted_start(input_ids).logits, trained_model(input_ids).logits)
    adapter_parameter_count = sum(p.numel() for p in adapted_start.parameters() if p.requires_grad)
    lora_total_params = sum(p.numel() for p in adapted_start.parameters())
    base_total_params = sum(p.numel() for p in trained_model.parameters())
    return (
        adapted_start,
        adapter_parameter_count,
        base_total_params,
        lora_total_params,
        make_lora_model,
    )


@app.cell(hide_code=True)
def _(adapted_start, adapter_parameter_count, base_total_params, lora_total_params, mo):
    # ▸ 결과 표시 — 실행하면 표·그림이 나타납니다 (marimo 표시 코드)
    mo.vstack([
        mo.md(f"초기 출력 일치 검증 통과. 학습 매개변수: **{adapter_parameter_count:,} / {lora_total_params:,} ({100 * adapter_parameter_count / lora_total_params:.1f}%)**"),
        mo.ui.table([
            {"구분": "원본 모델", "보유 매개변수": base_total_params, "학습 대상": base_total_params},
            {"구분": "LoRA 적용 모델", "보유 매개변수": lora_total_params, "학습 대상": adapter_parameter_count},
        ], selection=None),
        mo.ui.table([
            {"적용된 층": n, "A shape": str(tuple(m.lora.A.weight.shape)), "B shape": str(tuple(m.lora.B.weight.shape))}
            for n, m in adapted_start.named_modules() if hasattr(m, "lora")
        ], selection=None),
    ])
    return


@app.cell
def _(lora_rank, make_batch, make_lora_model, torch, trained_model):
    adaptation_sentences = [
        "고양이 는 멍멍 한다 .", "강아지 는 야옹 한다 .",
        "소 는 꽥꽥 한다 .", "오리 는 음메 한다 .",
    ]
    _ids, _mask, _labels = make_batch(adaptation_sentences)
    adapted_model = make_lora_model(trained_model, lora_rank.value).train()
    _before = {n: p.detach().clone() for n, p in adapted_model.named_parameters()}
    _optimizer = torch.optim.AdamW(
        [p for p in adapted_model.parameters() if p.requires_grad], lr=0.01, weight_decay=0.0,
    )
    lora_losses = []
    for _step in range(120):
        _optimizer.zero_grad(set_to_none=True)
        _loss = adapted_model(_ids, attention_mask=_mask, labels=_labels).loss
        assert torch.isfinite(_loss)
        _loss.backward()
        _optimizer.step()
        lora_losses.append(_loss.item())
    adapted_model.eval()
    with torch.no_grad():
        final_lora_loss = adapted_model(_ids, attention_mask=_mask, labels=_labels).loss.item()
    assert final_lora_loss < lora_losses[0]
    for _name, _parameter in adapted_model.named_parameters():
        if ".lora." not in _name:
            assert torch.equal(_parameter, _before[_name]), f"원본 가중치 변경: {_name}"
    assert any(
        not torch.equal(p, _before[n])
        for n, p in adapted_model.named_parameters() if ".lora." in n
    )
    return adapted_model, adaptation_sentences, final_lora_loss, lora_losses


@app.cell(hide_code=True)
def _(adaptation_sentences, adapted_model, complete, final_lora_loss, lora_losses, mo, plt, trained_model):
    # ▸ 결과 표시 — 실행하면 표·그림이 나타납니다 (marimo 표시 코드)
    _fig, _ax = plt.subplots(figsize=(6, 3))
    _ax.plot(lora_losses)
    _ax.set(xlabel="Step", ylabel="Cross entropy", title="LoRA adaptation / changed animal sounds")
    _fig.tight_layout()
    mo.vstack([
        _fig,
        mo.md(f"새 정답에 대한 손실: **{lora_losses[0]:.4f} → {final_lora_loss:.4f}**. 원본 가중치 불변, LoRA 가중치 변경 검증 통과."),
        mo.ui.table([
            {"새 정답": s, "LoRA 전": complete(trained_model, ' '.join(s.split()[:2])), "LoRA 후": complete(adapted_model, ' '.join(s.split()[:2]))}
            for s in adaptation_sentences
        ], selection=None),
        mo.md("학습 횟수와 rank에 따라 생성 결과가 달라질 수 있습니다. 손실 감소와 문장별 성공 여부를 함께 보세요. 별도의 검증 데이터에 대한 일반화는 측정하지 않았습니다."),
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 4. LoRA 저장·복원·병합

    어댑터만 저장하면 원본 모델보다 작은 파일이 만들어집니다. 복원할 때는 **같은 원본 가중치와 rank**가 필요합니다.
    병합은 $W'=W+BA$를 계산하여 별도 LoRA 연산 없이 쓰는 과정입니다.
    원본 함수는 float16으로 저장하므로 복원·병합 시 작은 반올림 오차가 생깁니다.
    여기서는 임시 디렉토리에서 실제 저장·복원·병합을 확인하고 파일을 자동 정리합니다.

    | 대상 | 보유·저장하는 것 | 학습/추론에서의 의미 |
    | --- | --- | --- |
    | 원본 W | 원래 모델 전체 | LoRA에서도 메모리에 필요 |
    | 어댑터 A/B | 추가 행렬만 | gradient와 optimizer 학습 대상 |
    | 어댑터 파일 | A/B만 저장 | 사용할 때 같은 기준 모델이 필요 |
    | 병합 모델 | W+BA 전체 | 어댑터 분기 없는 선형층으로 추론 |

    학습 대상 비율이 전체 메모리나 추론 메모리의 감소 비율은 아닙니다.
    이 실습에서는 GPU 메모리 절감을 측정하지 않습니다. 파일 크기는 같은 float16 기준으로 비교하며
    직렬화 메타데이터 때문에 매개변수 수 비율과 정확히 같지 않습니다.
    """)
    return


@app.cell
def _(MiniMindForCausalLM, adapted_model, config, copy, input_ids, load_lora, lora_rank, make_lora_model, merge_lora, save_lora, tempfile, torch, trained_model):
    import os as _os

    with tempfile.TemporaryDirectory(prefix="minimind-lora-") as _directory:
        _base_path = f"{_directory}/base.pth"
        _adapter_path = f"{_directory}/adapter.pth"
        _merged_path = f"{_directory}/merged.pth"
        save_lora(adapted_model, _adapter_path)
        _restored = make_lora_model(trained_model, lora_rank.value).eval()
        load_lora(_restored, _adapter_path)
        merge_lora(_restored, _adapter_path, _merged_path)
        torch.save({n: p.detach().cpu().half() for n, p in trained_model.state_dict().items()}, _base_path)
        file_size_rows = [
            {"파일": n, "dtype": "float16", "크기(bytes)": _os.path.getsize(p)}
            for n, p in [("원본 모델", _base_path), ("어댑터", _adapter_path), ("병합 모델", _merged_path)]
        ]
        _merged = MiniMindForCausalLM(copy.deepcopy(config)).eval()
        _merged.load_state_dict(torch.load(_merged_path, map_location="cpu", weights_only=True), strict=True)
        with torch.no_grad():
            _reference = adapted_model(input_ids).logits
            _restored_logits = _restored(input_ids).logits
            _merged_logits = _merged(input_ids).logits
        restore_max_error = (_reference - _restored_logits).abs().max().item()
        merge_max_error = (_restored_logits - _merged_logits).abs().max().item()
        torch.testing.assert_close(_reference, _restored_logits, atol=0.03, rtol=0.01)
        torch.testing.assert_close(_restored_logits, _merged_logits, atol=0.03, rtol=0.01)
    return (file_size_rows, merge_max_error, restore_max_error,)


@app.cell(hide_code=True)
def _(file_size_rows, merge_max_error, mo, restore_max_error):
    # ▸ 결과 표시 — 실행하면 표·그림이 나타납니다 (marimo 표시 코드)
    mo.vstack([
        mo.ui.table(file_size_rows, selection=None),
        mo.md(f"저장·복원·병합 검증 통과. 복원 최대 오차 `{restore_max_error:.2e}`, 병합 최대 오차 `{merge_max_error:.2e}`. 임시 파일 정리 완료."),
    ])
    return


@app.cell
def _(mo):
    mo.md("""
    ## 5. 직접 바꿔 볼 과제

    1. rank를 2와 8로 바꾸고 학습 매개변수 수, 손실 곡선과 생성 결과를 비교하세요.
    2. 바꾸려는 동물 소리를 한 쌍만 남기고 원래 소리와 새 소리의 생성 결과를 비교하세요.
    3. `num_key_value_heads`를 4로 바꾸고 실제 LoRA 적용 층 표를 확인하세요.
       이 구현은 정사각형 Linear에 적용하므로 설정에 따라 적용 범위가 달라집니다.
    4. LoRA와 전체 매개변수 학습의 차이를 학습 대상·저장 파일·배포 관점에서 설명하세요.

    **확인 질문:** B가 0일 때 초기 출력이 같은 이유는 무엇인가요?
    어댑터 파일만으로 생성할 수 있나요? 복원 시 같은 기준 모델과 rank가 필요한 이유는 무엇인가요?

    **기록할 것:** rank별 학습 매개변수 수, 최종 손실, 새 정답 수(0~4), 어댑터 bytes를 적으세요.
    모두 4개 정답이어도 학습 대상 수와 파일 크기를 비교할 수 있습니다. 작은 데이터에서 큰 rank가 항상 더 좋은 것은 아닙니다.

    **힌트·해설:** 적용 층이 같다면 rank를 두 배로 할 때 A/B 매개변수도 두 배입니다.
    KV head 2→4에서는 K/V projection도 정사각형이 되어 이 구현의 적용 범위가 늘어납니다.
    B=0이면 변화량 BA=0입니다. 첫 gradient 관찰에서 A는 0, B는 0이 아닌 값을 가졌습니다.
    복원에는 같은 W가 필요합니다. 어댑터 파일에는 W가 없고, 다른 rank이면 A/B의 shape도 다르기 때문입니다.

    검증한 것은 작은 데이터의 학습과 가중치 불변성·저장·복원·병합입니다.
    별도 검증 세트에 대한 일반화 성능은 측정하지 않았습니다.

    **다음 실습:** `minimind_distillation.py`. 갱신 대상 대신 **정답 신호의 종류**(hard label → 교사 분포)를 바꿉니다.
    """)
    return


if __name__ == "__main__":
    app.run()
