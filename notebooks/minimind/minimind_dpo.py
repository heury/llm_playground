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

    **목표:** 같은 질문의 chosen/rejected 응답을 읽고, 토큰 확률에서 DPO 손실을 계산한 뒤
    실제 MiniMind의 응답 선호를 바꿉니다. `dataset/lm_dataset.py`의 **DPODataset**과
    `trainer/train_dpo.py`의 **logits_to_log_probs, dpo_loss**를 직접 사용합니다.

    SFT는 정답 응답의 다음 토큰을 학습합니다. DPO는 선호 응답과 비선호 응답의
    **상대적인 로그 확률 차이**를 고정된 참조 모델과 비교하여 정책 모델을 학습합니다.
    여기서는 별도 보상 모델을 학습하거나 온라인으로 응답을 수집하지 않습니다.
    LoRA는 학습할 매개변수를 정하는 방법이고 DPO는 학습 목적이므로 함께 쓸 수도 있습니다.
    이 실습은 전체 매개변수를 업데이트합니다.

    **선수 지식:** 다음 토큰 예측·logits·softmax·SFT·backward.
    `minimind_structure.py` → `minimind_pretraining.py`를 먼저 읽으면 좋습니다.
    예측 → 실행 → 관찰 → 설명 순서로 진행하세요. marimo의 함수 인자·return은 셀 간 전달입니다.
    CPU에서 단독 실행하며 모델·데이터 다운로드가 없습니다. 작은 합성 데이터 실습입니다.

    산출물은 저장소의 `outputs/notebooks/<실행 ID>/`에 저장합니다.
    실제 저장 경로는 마지막 저장 결과 셀에서 확인할 수 있습니다.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## 1. 선호 데이터: 같은 질문, 다른 표현

    **예측:** `야옹 .`도 내용은 맞는데 왜 rejected일까요?
    이번 데이터의 선호 기준은 **같은 동물 소리를 공손한 완성 문장으로 말하기**입니다.
    chosen은 `네 야옹 입니다 .`, rejected는 `야옹 .`입니다.
    rejected가 항상 거짓이거나 유해한 답이라는 뜻은 아닙니다. 선호 기준은 데이터 작성자가 정합니다.

    원본 형식은 한 JSONL 행에 `chosen`, `rejected`가 있고, 각각 **전체 메시지 목록**입니다.
    공통 질문은 같고 마지막 assistant 응답만 다릅니다.
    8개 질문 조합 중 6개는 학습, 2개는 검증입니다. 검증 응답으로 optimizer를 갱신하지 않습니다.
    """)
    return


@app.cell
def _(json, mo, output_dir):
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
    mo.ui.table([
        {"분할": split, "질문": row["chosen"][0]["content"],
         "chosen": row["chosen"][-1]["content"], "rejected": row["rejected"][-1]["content"]}
        for split, rows in [("학습", preference_train), ("검증", preference_validation)] for row in rows
    ], selection=None)
    return preference_train, preference_validation, train_path, validation_path


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

    로컬 공백 단위 토크나이저를 사용합니다. 원본의 경계 탐지 규칙에 맞춰
    `<bos>assistant\n`과 `<eos>\n`을 구성합니다. 실제 학습에서는 모델의 토크나이저와 템플릿을 씁니다.
    이 템플릿에는 빈 think 블록이 없어 원본의 확률적 빈 think 제거가 결과에 영향을 주지 않습니다.

    **예측:** assistant 헤더, 답변 본문, EOS 중 어디에 mask=1이 붙을까요?
    원본은 assistant 헤더 다음부터 EOS까지 1로 만듭니다. 여러 턴이면 **모든 assistant 응답**을 포함합니다.
    user와 패딩은 손실에서 제외됩니다. 질문 토큰도 Attention의 문맥으로는 사용됩니다.

    `x = ids[:-1]`, `y = ids[1:]`, `mask = loss_mask[1:]`로 이미 한 칸 이동합니다.
    따라서 `model(x).logits`에서 `y`의 확률을 직접 읽습니다. 모델에 `labels=y`를 넘겨
    내부에서 다시 이동시키면 정렬이 틀어집니다. 길이 L=24이면 x/y/mask는 모두 길이 23입니다.
    """)
    return


@app.cell
def _(DPODataset, DataLoader, mo, tokenizer, train_path, validation_path):
    train_dataset = DPODataset(str(train_path), tokenizer, max_length=24)
    validation_dataset = DPODataset(str(validation_path), tokenizer, max_length=24)
    train_batch = next(iter(DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=False)))
    validation_batch = next(iter(DataLoader(validation_dataset, batch_size=len(validation_dataset), shuffle=False)))
    for _batch in [train_batch, validation_batch]:
        for _side in ["chosen", "rejected"]:
            assert (_batch[f"mask_{_side}"].sum(-1) > 0).all(), "응답이 전부 잘렸습니다."
            assert not (_batch[f"y_{_side}"] == tokenizer.unk_token_id).any()
            assert not (_batch[f"mask_{_side}"].bool() & (_batch[f"y_{_side}"] == tokenizer.pad_token_id)).any()
    mo.ui.table([
        {"위치 t": t, "입력 x[t]": tokenizer.convert_ids_to_tokens(int(train_batch["x_chosen"][0, t])),
         "예측 정답 y[t]": tokenizer.convert_ids_to_tokens(int(train_batch["y_chosen"][0, t])),
         "mask[t]": int(train_batch["mask_chosen"][0, t])}
        for t in range(train_batch["x_chosen"].shape[1])
    ], selection=None)
    return train_batch, validation_batch


@app.cell
def _(DPODataset, mo, tokenizer, train_path):
    _rows = []
    for _length in [6, 10, 24]:
        _item = DPODataset(str(train_path), tokenizer, max_length=_length)[0]
        _rows.append({"max_length": _length,
                      "chosen 유효 정답": int(_item["mask_chosen"].sum()),
                      "rejected 유효 정답": int(_item["mask_rejected"].sum())})
    mo.vstack([
        mo.ui.table(_rows, selection=None),
        mo.md("**관찰·해설:** 너무 짧게 자르면 응답이 사라지거나 EOS 없이 일부만 남습니다. 원본은 mask가 전부 0인 쌍도 거부하지 않습니다. 양쪽 모두 0이면 손실은 log(2)이지만 gradient 신호가 없습니다. 데이터 검사에서 제외하거나 길이를 늘리세요. 원본은 별도 attention_mask를 전달하지 않습니다. 여기서는 오른쪽 패딩과 causal Attention을 쓰므로 앞의 유효 위치는 뒤 PAD를 보지 않습니다."),
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. logits에서 응답 로그 확률까지

    logits `[2B,T,V]` → `log_softmax` → `gather(y)` → 정답 토큰별 log p `[2B,T]`.
    log p에 mask를 곱하고 시간축을 **합산**하면 응답별 점수 `[2B]`입니다.
    원본은 토큰 수로 나누지 않습니다. 로그 확률의 합은 조건부 토큰 확률 곱의 로그입니다.
    답변 길이가 다르면 점수에 길이 영향이 있으므로 chosen의 원점수가 더 크다고 가정하지 마세요.

    원본 배치 순서는 `[chosen 0, chosen 1, …, rejected 0, rejected 1, …]`입니다.
    `dpo_loss`가 앞 절반/뒤 절반을 자르므로 순서가 바뀌면 다른 응답끼리 비교합니다.
    B는 선호 **쌍의 수**이고, 모델 입력 행은 2B개입니다.

    **예측:** 정답 토큰의 logits만 먼저 골라 softmax하면 같은 값일까요?
    정규화에는 V개 전체가 필요합니다. 아래에서 실제 원본 함수와 직접 계산을 비교합니다.
    """)
    return


@app.cell
def _(F, logits_to_log_probs, mo, torch):
    _logits = torch.tensor([[[2., 0., -1.], [0., 1., 2.]]])
    _labels = torch.tensor([[0, 2]])
    _actual = logits_to_log_probs(_logits, _labels)
    _manual = torch.stack([F.log_softmax(_logits[0, t], dim=0)[_labels[0, t]] for t in range(2)])
    torch.testing.assert_close(_actual[0], _manual)
    mo.ui.table([
        {"위치": t, "정답 ID": int(_labels[0, t]), "p(정답)": float(_actual[0, t].exp()),
         "log p(정답)": float(_actual[0, t])} for t in range(2)
    ], selection=None)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 4. DPO 손실을 네 개의 숫자로 계산하기

    $$s_\theta(y)=\sum_t m_t\log\pi_\theta(y_t\mid x,y_{<t})$$
    $$\Delta=(s_\theta(y^+)-s_\theta(y^-))-(s_{ref}(y^+)-s_{ref}(y^-))$$
    $$L=-\log\sigma(\beta\Delta)$$

    θ는 학습할 정책, ref는 고정 참조 모델, +는 chosen, −는 rejected입니다.
    `sθ − sref`는 참조 대비 변화입니다. `β(sθ−sref)`를 암묵적 보상으로 볼 수 있습니다.
    별도의 reward head 출력이 아닙니다. `sigmoid(βΔ)`는 이 선호 모델의 쌍 비교 값이며
    실제 사용자 만족도의 보정된 확률로 해석하지 않습니다.

    **예측:** 정책과 참조가 같으면 chosen의 원래 확률에 관계없이 손실은 얼마일까요?
    Δ=0이므로 log(2)≈0.693입니다. Δ가 증가하면 손실은 감소합니다.
    도함수는 `dL/dΔ = −β sigmoid(−βΔ)`입니다. chosen 점수는 올리고 rejected 점수는
    내리는 방향의 신호지만, 공유 가중치를 업데이트하므로 개별 확률의 단조 변화는 보장되지 않습니다.

    β는 참조 대비 로그 비율의 척도이자 이론적 KL 제약의 강도와 연결됩니다.
    같은 Δ에서의 손실·gradient 변화와, 다른 β로 다시 학습한 결과를 구분하세요.
    “β가 클수록 무조건 정책이 더 많이 바뀐다”라고 결론 내릴 수 없습니다.
    """)
    return


@app.cell
def _(mo):
    demo_delta = mo.ui.slider(-4, 4, step=0.5, value=1, label="참조 대비 선호 차이 Δ")
    demo_beta = mo.ui.dropdown([0.1, 0.5, 1.0], value=0.5, label="숫자 실험 β")
    mo.hstack([demo_delta, demo_beta])
    return demo_beta, demo_delta


@app.cell
def _(demo_beta, demo_delta, dpo_loss, math, mo, plt, torch):
    _delta = torch.tensor(float(demo_delta.value), requires_grad=True)
    # 순서: chosen, rejected. 한 위치가 전체 응답 점수를 대표하는 숫자 예제.
    _ref = torch.tensor([[-4.], [-3.]])
    _policy = torch.stack([-4. + _delta, torch.tensor(-3.)]).reshape(2, 1)
    _loss = dpo_loss(_ref, _policy, torch.ones(2, 1), demo_beta.value)
    _loss.backward()
    torch.testing.assert_close(_delta.grad, -demo_beta.value * torch.sigmoid(-demo_beta.value * _delta.detach()))
    _zero = dpo_loss(_ref, _ref, torch.ones(2, 1), demo_beta.value)
    assert abs(_zero.item() - math.log(2)) < 1e-6
    _axis = torch.linspace(-5, 5, 101)
    _fig, _ax = plt.subplots(figsize=(6, 3))
    for _beta in [0.1, 0.5, 1.0]:
        _ax.plot(_axis, torch.nn.functional.softplus(-_beta * _axis), label=f"beta={_beta}")
    _ax.scatter([demo_delta.value], [_loss.item()], color="black")
    _ax.set(xlabel="Delta", ylabel="DPO loss")
    _ax.legend()
    _fig.tight_layout()
    mo.vstack([_fig, mo.md(f"손실 = **{_loss.item():.4f}**, dL/dΔ = **{_delta.grad.item():.4f}**. Δ=0 손실 = {_zero.item():.4f}. **해설:** 손실값뿐 아니라 gradient 부호도 확인하세요.")])
    return


@app.cell
def _(mo):
    mo.md("""
    ## 5. 작은 SFT 기준 모델 준비 후 두 모델로 복사

    실제 원본은 `init_model(..., from_weight='full_sft')`로 같은 SFT 가중치를 두 번 읽습니다.
    여기서는 무작위 초기화한 작은 Dense MiniMind를 두 응답 모두에 100 step SFT하여
    문장 형식을 익히게 합니다. 이 준비 단계는 선호를 구분하지 않습니다.
    **이후부터 DPO**이며, policy/ref 모두 이 동일한 준비 가중치의 별도 복사본입니다.

    이미 이동한 y를 사용하므로 SFT도 logits와 y의 cross entropy를 직접 계산하고 응답만 학습합니다.
    `requires_grad_(False)`는 참조 매개변수를 동결하고, `eval()`은 dropout 등을 평가 모드로 바꾸며,
    `no_grad()`는 참조 순전파의 계산 그래프 생성을 막습니다. 세 역할을 구분하세요.
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


@app.cell
def _(mo):
    train_beta = mo.ui.dropdown([0.1, 0.5, 1.0], value=0.5, label="DPO 학습 β")
    train_steps = mo.ui.slider(10, 80, step=10, value=40, label="DPO step")
    mo.hstack([train_beta, train_steps])
    return train_beta, train_steps


@app.cell
def _(mo):
    mo.md("""
    ## 6. 원본 학습 흐름을 한 배치로 실행

    **예측:** 첫 DPO loss는 약 0.693일까요? 참조 매개변수에도 gradient가 생길까요?
    참조 순전파 → 정책 순전파 → 원본 dpo_loss → backward → clipping → optimizer step 순서입니다.
    원본처럼 `outputs.aux_loss`를 한 번 더합니다. 여기서는 Dense라 0이고, MoE에서는 별도 항입니다.

    CPU float32, 전체 학습 쌍 한 배치, 누적 1, 고정 학습률 0.001로 축소합니다.
    원본의 AMP·DDP·cosine 학습률·분할 배치·재개는 실행하지 않습니다.
    원본 CLI 기본 β=0.15, 학습률=4e-8과 실습 설정은 다릅니다.
    작은 모델용 학습률을 실제 SFT 체크포인트에 그대로 적용하지 마세요.
    UI 변경 시 기준 모델의 새 복사본에서 다시 학습합니다.
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
def _(
    after_metrics,
    before_metrics,
    dpo_history,
    mo,
    plt,
    preference_train,
    preference_validation,
):
    _fig, _axes = plt.subplots(1, 2, figsize=(9, 3))
    _axes[0].plot([r["step"] for r in dpo_history], [r["dpo_loss"] for r in dpo_history])
    _axes[0].set(xlabel="Step (before update)", ylabel="Train DPO loss")
    for _name in ["train", "validation"]:
        _axes[1].plot(["Before", "After"], [before_metrics[_name]["loss"], after_metrics[_name]["loss"]], marker="o", label=_name)
    _axes[1].set(ylabel="DPO loss")
    _axes[1].legend()
    _fig.tight_layout()
    _rows = []
    for _split, _data in [("train", preference_train), ("validation", preference_validation)]:
        for _i, _row in enumerate(_data):
            _m = after_metrics[_split]
            _rows.append({"분할": _split, "질문": _row["chosen"][0]["content"],
                          "학습 전 raw gap": before_metrics[_split]["raw_gap"][_i],
                          "후 chosen log p": _m["chosen"][_i], "후 rejected log p": _m["rejected"][_i],
                          "후 raw gap": _m["raw_gap"][_i], "참조 대비 Δ": _m["delta"][_i]})
    mo.vstack([_fig, mo.ui.table(_rows, selection=None), mo.md(
        "**관찰·해설:** Δ>0은 참조보다 선호 격차가 커졌다는 뜻입니다. chosen 원확률이 rejected보다 높다는 조건(raw gap>0)과 다릅니다. 초기 Δ=0은 엄격한 >0 지표에서 승리로 세지 않습니다. 검증도 두 문장의 조합 확인일 뿐이며 일반적인 선호 정렬 성능을 증명하지 않습니다. 곡선의 마지막 점은 마지막 갱신 전, After 값은 마지막 갱신 후입니다."
    )])
    return


@app.cell
def _(mo, policy, preference_validation, reference, tokenizer, torch):
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
    mo.vstack([mo.md("## 7. 같은 검증 질문으로 생성 비교"), mo.ui.table(generated_examples, selection=None),
               mo.md("**예측·관찰:** Δ가 커지면 greedy 결과도 반드시 바뀔까요? 쌍의 점수 차이는 전체 응답의 확률을 비교하고, greedy는 매 위치의 최대 확률 토큰을 고릅니다. loss 감소·후보 순위·실제 생성·새 질문 일반화는 각각 확인해야 합니다. 정해진 두 후보 이외의 응답을 억제하는 보장은 없습니다.")])
    return (generated_examples,)


@app.cell
def _(
    after_metrics,
    before_metrics,
    config,
    dpo_history,
    generated_examples,
    json,
    mo,
    output_dir,
    policy,
    sft_history,
    torch,
    train_beta,
    train_steps,
    uuid4,
):
    experiment_dir = output_dir / f"training-{uuid4().hex[:6]}"
    experiment_dir.mkdir()
    torch.save(policy.state_dict(), experiment_dir / "dpo_policy.pth")
    config.to_json_file(experiment_dir / "config.json")
    experiment_metrics = {
        "settings": {"seed": 42, "beta": train_beta.value, "dpo_steps": train_steps.value,
                     "learning_rate": 0.001, "device": "cpu", "dtype": "float32", "sft_steps": 100},
        "environment": {"torch": torch.__version__, "marimo": mo.__version__},
        "sft_loss": sft_history, "dpo_history": dpo_history,
        "before": before_metrics, "after": after_metrics, "generation": generated_examples,
        "checks": {"initial_loss_log2": True, "reference_unchanged": True,
                   "reference_no_grad": True, "policy_changed": True, "train_loss_decreased": True},
    }
    (experiment_dir / "metrics.json").write_text(json.dumps(experiment_metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    mo.md(f"데이터·토크나이저는 `{output_dir}`, 이번 설정의 가중치·config·지표는 `{experiment_dir}`에 저장했습니다. 각 설정 실행은 별도 폴더를 만듭니다. 원본 trainer의 optimizer·scaler 재개 형식을 검증한 것은 아닙니다.")
    return


@app.cell
def _(mo):
    mo.md("""
    ## 8. 수강생 과제와 코드로 돌아가기

    1. 숫자 실험에서 Δ=-1, 0, +1과 β=0.1, 1을 비교하세요.
       **기록:** 손실과 dL/dΔ. **힌트:** β를 바꾸어도 Δ=0의 손실은 log(2), gradient는 −β/2입니다.
    2. DPO β를 바꾸어 새로 학습하세요. **기록:** 학습/검증 loss, Δ, raw gap, greedy 응답.
       **해설:** β가 다르면 손실 척도도 달라집니다. loss만으로 설정의 우열을 판정하지 마세요.
    3. 학습 데이터 중 한 쌍의 chosen/rejected를 뒤집으세요. **기록:** 그 질문과 검증 질문의 Δ.
       **해설:** 선호 라벨이 학습 방향을 정하며, 공유 매개변수 때문에 다른 질문에도 영향이 갑니다.
    4. 길이가 다른 답변을 추가하고 유효 mask 수를 비교하세요.
       **힌트:** 원본은 합산합니다. 평균으로 바꾸면 원본과 다른 목적 함수가 됩니다.

    원본 연결: `DPODataset.__getitem__`은 6개 텐서를 만들고, `generate_loss_mask`는 assistant 경계를 찾습니다.
    `train_epoch`는 chosen 전체/ rejected 전체를 연결하고 참조와 정책을 순전파합니다.
    `logits_to_log_probs`는 정답 토큰 확률, `dpo_loss`는 응답별 합산·절반 분할·참조 차감·logsigmoid를 담당합니다.
    main에서는 동일 SFT 가중치로 두 모델을 만들고 ref만 동결합니다.

    **실제 데이터 적용 전 확인:** 같은 질문의 비교인지, 선호 기준이 일관적인지,
    토크나이저 경계가 원본 mask와 맞는지, 절단 후 양쪽 응답이 남는지,
    학습 질문과 별도 평가 질문이 섞이지 않았는지 확인하세요.
    이 실습은 6쌍 학습·2쌍 검증이며 대규모 학습, 사용자 선호 평가, 망각 평가, GPU 분산 실행은 포함하지 않습니다.
    원본 저작권·라이선스는 `minimind/LICENSE`를 따릅니다.
    """)
    return


if __name__ == "__main__":
    app.run()
