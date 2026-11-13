import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import copy
    import json
    import math
    import sys
    from datetime import datetime
    from pathlib import Path
    from uuid import uuid4
    from zoneinfo import ZoneInfo

    import datasets  # 원본 trainer의 Windows DLL 로딩 순서 유지
    import marimo as mo
    import matplotlib.pyplot as plt
    import torch
    import torch.nn.functional as F
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast
    from torch.utils.data import DataLoader

    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root / "projects/minimind"))
    from dataset.lm_dataset import DPODataset
    from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
    from trainer.train_dpo import dpo_loss, logits_to_log_probs

    run_id = datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y%m%d-%H%M%S") + f"-{uuid4().hex[:6]}-dpo"
    output_dir = repo_root / "outputs" / "notebooks" / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    datasets.config.HF_DATASETS_CACHE = output_dir / "dataset-cache"
    datasets.disable_progress_bars()
    torch.set_num_threads(2)
    return (
        DPODataset,
        DataLoader,
        F,
        MiniMindConfig,
        MiniMindForCausalLM,
        PreTrainedTokenizerFast,
        Tokenizer,
        copy,
        dpo_loss,
        json,
        logits_to_log_probs,
        math,
        mo,
        models,
        output_dir,
        plt,
        pre_tokenizers,
        torch,
        uuid4,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # MiniMind DPO: 두 응답의 선호 비교로 미세 조정하기

    **목표:** 같은 질문의 좋은 답(chosen)과 덜 좋은 답(rejected)으로 모델의 선호를 바꿉니다.

    - `dataset/lm_dataset.py`의 **DPODataset**
    - `trainer/train_dpo.py`의 **logits_to_log_probs, dpo_loss**

    **SFT와 DPO의 차이:**

    - SFT: 정답 응답의 다음 토큰을 학습합니다.
    - DPO: chosen과 rejected의 **로그 확률 차이**를 키웁니다. 고정된 참조 모델과 비교합니다.
    - 보상 모델도, 온라인 응답 수집도 없습니다.

    LoRA는 학습할 매개변수, DPO는 학습 목적을 정합니다. 둘은 함께 쓸 수 있습니다.
    이 실습은 전체 매개변수를 학습합니다.

    **선수 실습:** `minimind_sft.py`. 마스크 규칙은 SFT와 같습니다.
    다른 점: **Dataset이 이미 한 칸 옮긴 x/y를 돌려줍니다**(2절).

    **선호·강화 학습 파트의 첫 노트북**입니다. 다음은 PPO → GRPO.
    CPU로 실행하며 다운로드는 없습니다. 작은 합성 데이터 실습입니다.
    산출물: `outputs/notebooks/<실행 ID>/`. 정확한 경로는 마지막 셀에 나옵니다.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## 1. 선호 데이터: 같은 질문, 다른 표현

    **예측:** `야옹 .`도 내용은 맞는데 왜 rejected일까요?
    선호 기준: **공손한 완성 문장으로 답하기**.

    - chosen: `네 야옹 입니다 .`
    - rejected: `야옹 .`

    rejected가 틀린 답이라는 뜻은 아닙니다. 기준은 데이터를 만든 사람이 정합니다.

    **형식:** JSONL 한 줄에 `chosen`, `rejected`. 각각 **전체 메시지 목록**입니다.
    질문은 같고 마지막 assistant 응답만 다릅니다.
    8쌍 중 6쌍은 학습, 2쌍은 검증용입니다.
    """)
    return


@app.cell
def _(json, output_dir):
    preference_train, preference_validation = [], []
    for _i, (_animal, _sound) in enumerate([
        ("고양이", "야옹"), ("강아지", "멍멍"), ("소", "음메"), ("오리", "꽥꽥"),
    ]):
        for _verb in ["알려줘", "말해줘"]:
            _prompt = f"{_animal} 소리 {_verb}"
            _row = {
                "chosen": [{"role": "user", "content": _prompt},
                           {"role": "assistant", "content": f"네 {_sound} 입니다 ."}],
                "rejected": [{"role": "user", "content": _prompt},
                             {"role": "assistant", "content": f"{_sound} ."}],
            }
            _split = preference_validation if _i < 2 and _verb == "말해줘" else preference_train
            _split.append(_row)
    train_path = output_dir / "train.jsonl"
    validation_path = output_dir / "validation.jsonl"
    for _path, _rows in [(train_path, preference_train), (validation_path, preference_validation)]:
        _path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in _rows), encoding="utf-8")
        for _row in _rows:
            assert _row["chosen"][:-1] == _row["rejected"][:-1]
            assert _row["chosen"][-1]["role"] == _row["rejected"][-1]["role"] == "assistant"
    return (preference_train, preference_validation, train_path, validation_path,)


@app.cell(hide_code=True)
def _(mo, preference_train, preference_validation):
    # ▸ 결과 표시 — 실행하면 표·그림이 나타납니다 (marimo 표시 코드)
    mo.ui.table([
        {"분할": split, "질문": row["chosen"][0]["content"],
         "chosen": row["chosen"][-1]["content"], "rejected": row["rejected"][-1]["content"]}
        for split, rows in [("학습", preference_train), ("검증", preference_validation)] for row in rows
    ], selection=None)
    return


@app.cell
def _(
    PreTrainedTokenizerFast,
    Tokenizer,
    models,
    output_dir,
    pre_tokenizers,
    preference_train,
):
    # 원본이 찾는 '<bos>assistant\n', '<eos>\n' 경계를 실제 특수 토큰으로 만듭니다.
    _special = ["<pad>", "<bos>", "<eos>", "<unk>"]
    _words = {w for r in preference_train for k in ["chosen", "rejected"]
              for m in r[k] for w in m["content"].split()}
    vocabulary = _special + ["user", "assistant"] + sorted(_words)
    _backend = Tokenizer(models.WordLevel(dict(zip(vocabulary, range(len(vocabulary)))), unk_token="<unk>"))
    _backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=_backend, pad_token="<pad>", bos_token="<bos>",
        eos_token="<eos>", unk_token="<unk>", model_max_length=64,
        clean_up_tokenization_spaces=False,
    )
    tokenizer.chat_template = (
        "{% for message in messages %}{{ bos_token + message['role'] + '\n' + message['content'] + eos_token + '\n' }}{% endfor %}"
        "{% if add_generation_prompt %}{{ bos_token + 'assistant\n' }}{% endif %}"
    )
    tokenizer.save_pretrained(output_dir / "tokenizer")
    return tokenizer, vocabulary


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. DPODataset: 템플릿 → 토큰 → 마스크 → 한 칸 이동

    공백 단위 작은 토크나이저를 씁니다.
    원본 경계 규칙에 맞춰 `<bos>assistant\n`과 `<eos>\n`을 만듭니다.
    빈 think 블록이 없어서 무작위 think 제거는 영향이 없습니다.

    **예측:** 헤더, 본문, EOS 중 어디에 mask=1이 붙을까요?

    - assistant 헤더 다음부터 EOS까지 1입니다.
    - 여러 턴이면 **모든 assistant 응답**이 포함됩니다.
    - user와 패딩은 0입니다. 질문은 문맥으로만 쓰입니다.

    **이미 옮겨진 x/y:** `x = ids[:-1]`, `y = ids[1:]`, `mask = loss_mask[1:]`.

    - 그래서 `model(x).logits`에서 `y`의 확률을 바로 읽습니다.
    - `labels=y`로 넘기면 모델이 또 옮겨서 어긋납니다.
    - L=24이면 x/y/mask 길이는 모두 23입니다.
    """)
    return


@app.cell
def _(DPODataset, DataLoader, tokenizer, train_path, validation_path):
    train_dataset = DPODataset(str(train_path), tokenizer, max_length=24)
    validation_dataset = DPODataset(str(validation_path), tokenizer, max_length=24)
    train_batch = next(iter(DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=False)))
    validation_batch = next(iter(DataLoader(validation_dataset, batch_size=len(validation_dataset), shuffle=False)))
    for _batch in [train_batch, validation_batch]:
        for _side in ["chosen", "rejected"]:
            assert (_batch[f"mask_{_side}"].sum(-1) > 0).all(), "응답이 전부 잘렸습니다."
            assert not (_batch[f"y_{_side}"] == tokenizer.unk_token_id).any()
            assert not (_batch[f"mask_{_side}"].bool() & (_batch[f"y_{_side}"] == tokenizer.pad_token_id)).any()
    return (train_batch, validation_batch,)


@app.cell(hide_code=True)
def _(mo, tokenizer, train_batch):
    # ▸ 결과 표시 — 실행하면 표·그림이 나타납니다 (marimo 표시 코드)
    mo.ui.table([
        {"위치 t": t, "입력 x[t]": tokenizer.convert_ids_to_tokens(int(train_batch["x_chosen"][0, t])),
         "예측 정답 y[t]": tokenizer.convert_ids_to_tokens(int(train_batch["y_chosen"][0, t])),
         "mask[t]": int(train_batch["mask_chosen"][0, t])}
        for t in range(train_batch["x_chosen"].shape[1])
    ], selection=None)
    return


@app.cell
def _(DPODataset, tokenizer, train_path):
    truncation_rows = []
    for _length in [6, 10, 24]:
        _item = DPODataset(str(train_path), tokenizer, max_length=_length)[0]
        truncation_rows.append({"max_length": _length,
                      "chosen 유효 정답": int(_item["mask_chosen"].sum()),
                      "rejected 유효 정답": int(_item["mask_rejected"].sum())})
    return (truncation_rows,)


@app.cell(hide_code=True)
def _(mo, truncation_rows):
    # ▸ 결과 표시 — 실행하면 표·그림이 나타납니다 (marimo 표시 코드)
    mo.vstack([
        mo.ui.table(truncation_rows, selection=None),
        mo.md("**관찰:** 너무 짧게 자르면 응답이나 EOS가 사라집니다. 원본은 mask가 모두 0인 쌍도 사용합니다. 이때 손실은 log(2)이지만 gradient는 없습니다. 이런 쌍은 빼거나 길이를 늘리세요.\n\nattention_mask는 넘기지 않습니다. 오른쪽 패딩과 causal Attention을 쓰므로 실제 토큰은 뒤의 PAD를 못 봅니다."),
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. logits에서 응답 로그 확률까지

    1. logits `[2B,T,V]` → `log_softmax` → `gather(y)` → 토큰별 log p `[2B,T]`
    2. mask를 곱하고 시간축으로 **더하면** 응답별 점수 `[2B]`

    - 토큰 수로 나누지 않습니다(평균 아님).
    - 그래서 길이가 다르면 점수도 길이 영향을 받습니다. chosen 점수가 더 크다고 가정하지 마세요.

    **배치 순서:** `[chosen 0, chosen 1, …, rejected 0, rejected 1, …]`.
    `dpo_loss`는 앞 절반과 뒤 절반을 짝짓습니다. 순서가 틀리면 엉뚱한 짝을 비교합니다.
    B는 **쌍의 수**, 입력 행은 2B개입니다.

    **예측:** 정답 토큰 logits만 골라 softmax해도 같을까요?
    아닙니다. 정규화에는 V개 전체가 필요합니다. 아래에서 비교합니다.
    """)
    return


@app.cell
def _(logits_to_log_probs, torch):
    # 원본 함수로 정답 토큰의 log 확률을 구합니다.
    demo_logits = torch.tensor([[[2., 0., -1.], [0., 1., 2.]]])
    demo_labels = torch.tensor([[0, 2]])
    demo_logps = logits_to_log_probs(demo_logits, demo_labels)
    return demo_labels, demo_logits, demo_logps


@app.cell
def _(F, demo_labels, demo_logits, demo_logps, torch):
    # 직접 계산: 위치마다 log_softmax를 하고 정답 칸을 골라, 원본 결과와 비교합니다.
    _manual = torch.stack([F.log_softmax(demo_logits[0, t], dim=0)[demo_labels[0, t]] for t in range(2)])
    torch.testing.assert_close(demo_logps[0], _manual)
    return


@app.cell(hide_code=True)
def _(demo_labels, demo_logps, mo):
    # ▸ 결과 표시 — 실행하면 표·그림이 나타납니다 (marimo 표시 코드)
    mo.ui.table([
        {"위치": t, "정답 ID": int(demo_labels[0, t]), "p(정답)": float(demo_logps[0, t].exp()),
         "log p(정답)": float(demo_logps[0, t])} for t in range(2)
    ], selection=None)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 4. DPO 손실을 네 개의 숫자로 계산하기

    $$s_\theta(y)=\sum_t m_t\log\pi_\theta(y_t\mid x,y_{<t})$$
    $$\Delta=(s_\theta(y^+)-s_\theta(y^-))-(s_{ref}(y^+)-s_{ref}(y^-))$$
    $$L=-\log\sigma(\beta\Delta)$$

    **기호:** θ 학습할 정책, ref 고정 참조 모델, + chosen, − rejected.

    - `sθ − sref`: 참조 대비 변화량
    - `β(sθ−sref)`: 암묵적 보상. 별도 reward head는 없습니다.
    - `sigmoid(βΔ)`: 쌍 비교 값. 사용자 만족 확률이 아닙니다.

    **예측:** 정책 = 참조이면 손실은?
    Δ=0이므로 log(2)≈0.693입니다. Δ가 커지면 손실이 줄어듭니다.

    **기울기:** `dL/dΔ = −β sigmoid(−βΔ)`.
    chosen은 올리고 rejected는 내리는 방향입니다.
    하지만 가중치를 공유하므로, 각 확률이 꼭 그렇게 움직이지는 않습니다.

    **β:** 참조에서 얼마나 멀어질지와 관련된 척도입니다.
    "β가 크면 무조건 더 많이 바뀐다"고 단정할 수 없습니다.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    # ▸ 실습 설정 위젯 — 실행하면 슬라이더·선택 상자가 나타납니다 (marimo 표시 코드)
    demo_delta = mo.ui.slider(-4, 4, step=0.5, value=1, label="참조 대비 선호 차이 Δ")
    demo_beta = mo.ui.dropdown([0.1, 0.5, 1.0], value=0.5, label="숫자 실험 β")
    mo.hstack([demo_delta, demo_beta])
    return demo_beta, demo_delta


@app.cell
def _(demo_beta, demo_delta, dpo_loss, math, plt, torch):
    delta_tensor = torch.tensor(float(demo_delta.value), requires_grad=True)
    # 순서: chosen, rejected. 한 위치가 전체 응답 점수를 대표하는 숫자 예제.
    _ref = torch.tensor([[-4.], [-3.]])
    _policy = torch.stack([-4. + delta_tensor, torch.tensor(-3.)]).reshape(2, 1)
    demo_loss = dpo_loss(_ref, _policy, torch.ones(2, 1), demo_beta.value)
    demo_loss.backward()
    torch.testing.assert_close(delta_tensor.grad, -demo_beta.value * torch.sigmoid(-demo_beta.value * delta_tensor.detach()))
    zero_delta_loss = dpo_loss(_ref, _ref, torch.ones(2, 1), demo_beta.value)
    assert abs(zero_delta_loss.item() - math.log(2)) < 1e-6
    _axis = torch.linspace(-5, 5, 101)
    dpo_loss_fig, _ax = plt.subplots(figsize=(6, 3))
    for _beta in [0.1, 0.5, 1.0]:
        _ax.plot(_axis, torch.nn.functional.softplus(-_beta * _axis), label=f"beta={_beta}")
    _ax.scatter([demo_delta.value], [demo_loss.item()], color="black")
    _ax.set(xlabel="Delta", ylabel="DPO loss")
    _ax.legend()
    dpo_loss_fig.tight_layout()
    return (delta_tensor, demo_loss, dpo_loss_fig, zero_delta_loss,)


@app.cell(hide_code=True)
def _(delta_tensor, demo_loss, dpo_loss_fig, mo, zero_delta_loss):
    # ▸ 결과 표시 — 실행하면 표·그림이 나타납니다 (marimo 표시 코드)
    mo.vstack([dpo_loss_fig, mo.md(f"손실 = **{demo_loss.item():.4f}**, dL/dΔ = **{delta_tensor.grad.item():.4f}**. Δ=0 손실 = {zero_delta_loss.item():.4f}. **해설:** 손실값과 gradient 부호를 함께 보세요.")])
    return


@app.cell
def _(mo):
    mo.md("""
    ## 5. 작은 SFT 기준 모델 준비 후 두 모델로 복사

    원본은 같은 SFT 가중치를 두 번 읽습니다(`from_weight='full_sft'`).

    여기서는 이렇게 준비합니다.

    1. 작은 Dense MiniMind를 무작위로 만듭니다.
    2. chosen·rejected 모두로 100 step SFT합니다. 선호는 아직 구분하지 않습니다.
    3. 이 가중치를 두 번 복사해 policy와 ref를 만듭니다. **여기부터 DPO**입니다.

    y가 이미 옮겨져 있으므로 SFT 손실도 직접 계산합니다.

    **참조 모델 고정에 쓰는 세 가지:**

    - `requires_grad_(False)`: 매개변수 동결
    - `eval()`: dropout 등을 평가 모드로
    - `no_grad()`: 계산 그래프를 만들지 않음
    """)
    return


@app.cell
def _(F, MiniMindConfig, MiniMindForCausalLM, torch, train_batch, vocabulary):
    def pack_pairs(batch):
        return tuple(torch.cat([batch[f"{key}_chosen"], batch[f"{key}_rejected"]], dim=0)
                     for key in ["x", "y", "mask"])

    config = MiniMindConfig(
        hidden_size=32, num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=2, intermediate_size=64, max_position_embeddings=64,
        vocab_size=len(vocabulary), dropout=0.0, flash_attn=False, use_moe=False,
        bos_token_id=1, eos_token_id=2,
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        sft_model = MiniMindForCausalLM(config).cpu().train()
    _optimizer = torch.optim.AdamW(sft_model.parameters(), lr=0.005, weight_decay=0.0)
    _x, _y, _mask = pack_pairs(train_batch)
    sft_history = []
    for _step in range(100):
        _optimizer.zero_grad(set_to_none=True)
        _logits = sft_model(_x).logits
        _token_losses = F.cross_entropy(_logits.reshape(-1, len(vocabulary)), _y.reshape(-1), reduction="none").reshape_as(_y)
        _loss = (_token_losses * _mask).sum() / _mask.sum()
        _loss.backward()
        _optimizer.step()
        sft_history.append(_loss.item())
    sft_model.eval()
    assert sft_history[-1] < sft_history[0]
    return config, pack_pairs, sft_history, sft_model


@app.cell(hide_code=True)
def _(mo):
    # ▸ 실습 설정 위젯 — 실행하면 슬라이더·선택 상자가 나타납니다 (marimo 표시 코드)
    train_beta = mo.ui.dropdown([0.1, 0.5, 1.0], value=0.5, label="DPO 학습 β")
    train_steps = mo.ui.slider(10, 80, step=10, value=40, label="DPO step")
    mo.hstack([train_beta, train_steps])
    return train_beta, train_steps


@app.cell
def _(mo):
    mo.md("""
    ## 6. 원본 학습 흐름을 한 배치로 실행

    **예측:** 첫 loss는 약 0.693일까요? 참조 모델에도 gradient가 생길까요?

    **순서:** 참조 순전파 → 정책 순전파 → dpo_loss → backward → clip → step.
    `outputs.aux_loss`도 더합니다(Dense라 0).

    **축소한 설정:** CPU float32, 전체 쌍 한 배치, 누적 1, 고정 lr 0.001.

    - AMP·DDP·cosine lr·재개는 하지 않습니다.
    - 원본 기본값은 β=0.15, lr=4e-8입니다. 이 실습 lr을 실제 모델에 쓰지 마세요.
    - UI를 바꾸면 기준 모델의 새 복사본에서 다시 학습합니다.
    """)
    return


@app.cell
def _(
    copy,
    dpo_loss,
    logits_to_log_probs,
    math,
    pack_pairs,
    sft_model,
    torch,
    train_batch,
    train_beta,
    train_steps,
    validation_batch,
):
    policy = copy.deepcopy(sft_model).train()
    reference = copy.deepcopy(sft_model).eval().requires_grad_(False)
    _ref_before = {k: v.detach().clone() for k, v in reference.state_dict().items()}
    _optimizer = torch.optim.AdamW(policy.parameters(), lr=0.001, weight_decay=0.0)
    _x, _y, _mask = pack_pairs(train_batch)

    def pair_metrics(model, ref, batch, beta):
        x, y, mask = pack_pairs(batch)
        with torch.no_grad():
            p = (logits_to_log_probs(model(x).logits, y) * mask).sum(-1)
            r = (logits_to_log_probs(ref(x).logits, y) * mask).sum(-1)
            b = p.shape[0] // 2
            delta = (p[:b] - p[b:]) - (r[:b] - r[b:])
        return {
            "chosen": p[:b].tolist(), "rejected": p[b:].tolist(),
            "raw_gap": (p[:b] - p[b:]).tolist(), "delta": delta.tolist(),
            "loss": torch.nn.functional.softplus(-beta * delta).mean().item(),
            "relative_win_rate": (delta > 0).float().mean().item(),
        }

    before_metrics = {name: pair_metrics(policy, reference, batch, train_beta.value)
                      for name, batch in [("train", train_batch), ("validation", validation_batch)]}
    dpo_history = []
    for _step in range(train_steps.value):
        _optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            _ref_lp = logits_to_log_probs(reference(_x).logits, _y)
        _outputs = policy(_x)  # labels를 넘기지 않음: y는 이미 이동함
        _policy_lp = logits_to_log_probs(_outputs.logits, _y)
        _dpo = dpo_loss(_ref_lp, _policy_lp, _mask, beta=train_beta.value)
        _loss = _dpo + _outputs.aux_loss
        assert torch.isfinite(_loss)
        _loss.backward()
        _grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        assert torch.isfinite(_grad_norm) and float(_grad_norm) > 0
        assert all(p.grad is None for p in reference.parameters())
        _optimizer.step()
        dpo_history.append({"step": _step, "dpo_loss": _dpo.item(), "grad_norm": float(_grad_norm)})
    policy.eval()
    after_metrics = {name: pair_metrics(policy, reference, batch, train_beta.value)
                     for name, batch in [("train", train_batch), ("validation", validation_batch)]}
    assert abs(dpo_history[0]["dpo_loss"] - math.log(2)) < 1e-5
    assert after_metrics["train"]["loss"] < before_metrics["train"]["loss"]
    assert all(torch.equal(v, _ref_before[k]) for k, v in reference.state_dict().items())
    assert any(not torch.equal(v, reference.state_dict()[k]) for k, v in policy.state_dict().items())
    return after_metrics, before_metrics, dpo_history, policy, reference


@app.cell
def _(after_metrics, before_metrics, dpo_history, plt, preference_train, preference_validation):
    dpo_curve_fig, _axes = plt.subplots(1, 2, figsize=(9, 3))
    _axes[0].plot([r["step"] for r in dpo_history], [r["dpo_loss"] for r in dpo_history])
    _axes[0].set(xlabel="Step (before update)", ylabel="Train DPO loss")
    for _name in ["train", "validation"]:
        _axes[1].plot(["Before", "After"], [before_metrics[_name]["loss"], after_metrics[_name]["loss"]], marker="o", label=_name)
    _axes[1].set(ylabel="DPO loss")
    _axes[1].legend()
    dpo_curve_fig.tight_layout()
    pair_rows = []
    for _split, _data in [("train", preference_train), ("validation", preference_validation)]:
        for _i, _row in enumerate(_data):
            _m = after_metrics[_split]
            pair_rows.append({"분할": _split, "질문": _row["chosen"][0]["content"],
                          "학습 전 raw gap": before_metrics[_split]["raw_gap"][_i],
                          "후 chosen log p": _m["chosen"][_i], "후 rejected log p": _m["rejected"][_i],
                          "후 raw gap": _m["raw_gap"][_i], "참조 대비 Δ": _m["delta"][_i]})
    return (dpo_curve_fig, pair_rows,)


@app.cell(hide_code=True)
def _(dpo_curve_fig, mo, pair_rows):
    # ▸ 결과 표시 — 실행하면 표·그림이 나타납니다 (marimo 표시 코드)
    mo.vstack([dpo_curve_fig, mo.ui.table(pair_rows, selection=None), mo.md(
        "**관찰:** Δ>0은 참조보다 선호 격차가 커졌다는 뜻입니다. chosen 확률이 더 높다는 뜻(raw gap>0)은 아닙니다. 처음 Δ=0은 승리로 세지 않습니다. 검증은 두 쌍뿐입니다. 일반적인 선호 성능은 알 수 없습니다. 곡선 마지막 점은 갱신 전, After 값은 갱신 후입니다."
    )])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7. 같은 검증 질문으로 생성 비교
    """)
    return


@app.cell
def _(policy, preference_validation, reference, tokenizer, torch):
    def complete(model, row):
        prompt = tokenizer.apply_chat_template(row["chosen"][:-1], tokenize=False, add_generation_prompt=True)
        ids = tokenizer(prompt, return_tensors="pt").input_ids
        with torch.no_grad():
            generated = model.generate(ids, max_new_tokens=8, do_sample=False,
                                       eos_token_id=tokenizer.eos_token_id, use_cache=True,
                                       top_k=0, top_p=1.0, temperature=1.0)
        return tokenizer.decode(generated[0, ids.shape[1]:], skip_special_tokens=True)

    generated_examples = [
        {"질문": row["chosen"][0]["content"], "참조 greedy": complete(reference, row),
         "DPO greedy": complete(policy, row), "선호 답변": row["chosen"][-1]["content"]}
        for row in preference_validation
    ]
    return (generated_examples,)


@app.cell(hide_code=True)
def _(generated_examples, mo):
    # ▸ 결과 표시 — 실행하면 표·그림이 나타납니다 (marimo 표시 코드)
    mo.vstack([mo.ui.table(generated_examples, selection=None),
               mo.md("**예측·관찰:** Δ가 커지면 greedy 결과도 꼭 바뀔까요? 점수 차이는 응답 전체 확률을 비교합니다. greedy는 위치마다 최대 토큰을 고릅니다. 손실, 후보 순위, 생성, 일반화는 각각 확인하세요. 두 후보 밖의 답변도 나올 수 있습니다.")])
    return


@app.cell
def _(
    after_metrics,
    before_metrics,
    config,
    dpo_history,
    generated_examples,
    json,
    output_dir,
    policy,
    sft_history,
    torch,
    train_beta,
    train_steps,
    uuid4,
):
    from importlib.metadata import version as _package_version

    experiment_dir = output_dir / f"training-{uuid4().hex[:6]}"
    experiment_dir.mkdir()
    torch.save(policy.state_dict(), experiment_dir / "dpo_policy.pth")
    config.to_json_file(experiment_dir / "config.json")
    experiment_metrics = {
        "settings": {"seed": 42, "beta": train_beta.value, "dpo_steps": train_steps.value,
                     "learning_rate": 0.001, "device": "cpu", "dtype": "float32", "sft_steps": 100},
        "environment": {"torch": torch.__version__, "marimo": _package_version("marimo")},
        "sft_loss": sft_history, "dpo_history": dpo_history,
        "before": before_metrics, "after": after_metrics, "generation": generated_examples,
        "checks": {"initial_loss_log2": True, "reference_unchanged": True,
                   "reference_no_grad": True, "policy_changed": True, "train_loss_decreased": True},
    }
    (experiment_dir / "metrics.json").write_text(json.dumps(experiment_metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    return (experiment_dir,)


@app.cell(hide_code=True)
def _(experiment_dir, mo, output_dir):
    # ▸ 결과 설명 — 실행하면 계산 값이 들어간 설명이 나타납니다 (marimo 표시 코드)
    mo.md(f"데이터·토크나이저는 `{output_dir}`, 이번 설정의 가중치·config·지표는 `{experiment_dir}`에 저장했습니다. 설정마다 폴더가 따로 생깁니다. 원본의 재개 형식은 검증하지 않았습니다.")
    return


@app.cell
def _(mo):
    mo.md("""
    ## 8. 수강생 과제와 코드로 돌아가기

    1. 숫자 실험: Δ=-1, 0, +1과 β=0.1, 1을 비교하세요.
       **기록:** 손실, dL/dΔ. **힌트:** Δ=0이면 β와 상관없이 손실 log(2), gradient −β/2.
    2. β를 바꿔 다시 학습하세요. **기록:** loss, Δ, raw gap, greedy 응답.
       **해설:** β가 다르면 손실 척도도 다릅니다. loss만으로 비교하지 마세요.
    3. 학습 쌍 하나의 chosen/rejected를 뒤집으세요. **기록:** 그 질문과 검증 질문의 Δ.
       **해설:** 라벨이 방향을 정합니다. 가중치를 공유하므로 다른 질문에도 영향이 갑니다.
    4. 길이가 다른 답변을 넣고 mask 수를 비교하세요.
       **힌트:** 원본은 합산합니다. 평균으로 바꾸면 다른 목적 함수가 됩니다.

    **원본 코드 대응:**

    - `DPODataset.__getitem__`: 텐서 6개를 만듭니다.
    - `generate_loss_mask`: assistant 경계를 찾습니다.
    - `train_epoch`: chosen·rejected를 이어 붙여 참조와 정책을 실행합니다.
    - `logits_to_log_probs`: 정답 토큰 확률을 뽑습니다.
    - `dpo_loss`: 합산 → 절반 분할 → 참조 차감 → logsigmoid.
    - main: 같은 SFT 가중치로 두 모델을 만들고 ref만 동결합니다.

    **실제 데이터에 쓰기 전 확인:**

    - 같은 질문끼리 비교하나요?
    - 선호 기준이 일관적인가요?
    - 토크나이저 경계가 mask 규칙과 맞나요?
    - 자른 뒤에도 양쪽 응답이 남나요?
    - 학습 질문과 평가 질문이 섞이지 않았나요?

    **범위 밖:** 대규모 학습, 사용자 선호 평가, 망각 평가, GPU 분산 실행.
    저작권·라이선스는 `minimind/LICENSE`를 따릅니다.

    **다음 실습:** `minimind_ppo.py`.

    - DPO: **이미 모은** 선호 쌍으로 학습(오프라인)
    - PPO: 모델이 **직접 만든** 응답에 보상(온라인), Critic 추가
    """)
    return


if __name__ == "__main__":
    app.run()
