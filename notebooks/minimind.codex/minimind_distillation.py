import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import copy
    import hashlib
    import json
    import sys
    from datetime import datetime
    from pathlib import Path
    from uuid import uuid4
    from zoneinfo import ZoneInfo

    import datasets  # 원본 trainer처럼 Windows에서는 torch보다 먼저 읽습니다.
    import marimo as mo
    import matplotlib.pyplot as plt
    import torch
    import torch.nn.functional as F
    from tokenizers import Regex, Tokenizer, models, pre_tokenizers
    from torch.utils.data import DataLoader
    from transformers import PreTrainedTokenizerFast

    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root / "projects/minimind"))
    from dataset.lm_dataset import SFTDataset
    from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
    from trainer.train_distillation import distillation_loss

    run_id = datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y%m%d-%H%M%S") + f"-{uuid4().hex[:6]}-distillation"
    output_dir = repo_root / "outputs" / "notebooks" / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    datasets.config.HF_DATASETS_CACHE = output_dir / "dataset-cache"
    datasets.disable_progress_bars()
    torch.set_num_threads(2)
    return (
        DataLoader, F, MiniMindConfig, MiniMindForCausalLM, PreTrainedTokenizerFast,
        Regex, SFTDataset, Tokenizer, copy, distillation_loss, hashlib, json, mo,
        models, output_dir, plt, pre_tokenizers, repo_root, torch, uuid4,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # MiniMind 지식 증류: 교사의 확률 분포를 학생에게 전달하기

    **목표:** 지식 증류를 직접 해 봅니다.

    1. 정답 하나 vs 확률 분포
    2. 온도·KL·CE 계산
    3. 어느 위치에 KL을 적용하나
    4. 교사 동결 → 학생 학습 → 비교

    **읽는 순서:** 구조 → 사전 학습 → SFT → 이 노트북.
    template과 마스크는 SFT 노트북에서 설명했습니다. 여기서는 **KL 적용 위치**만 봅니다.

    **사용하는 원본:**

    - `trainer/train_distillation.py`의 **distillation_loss**
    - `dataset/lm_dataset.py`의 **SFTDataset**

    CPU로 실행하며 다운로드는 없습니다. 교사를 먼저 학습하고 학생을 가르칩니다.
    작은 데이터 암기 실습입니다. 일반화나 성능 향상은 측정하지 않습니다.
    온도·alpha를 바꾸면 학생 비교 실험이 다시 실행됩니다.
    산출물: `outputs/notebooks/<실행 ID>/`. 저작권·라이선스는 `minimind/LICENSE`를 보세요.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## 1. 정답 하나보다 더 많은 정보

    “고양이의 소리는?”

    - **hard label:** 정답 `야옹` 하나만 줍니다.
    - **soft target:** 교사가 `야옹 0.7, 멍멍 0.2, 음메 0.1`을 줍니다.

    학생은 soft target에서 다른 후보의 상대 확률도 배웁니다.
    숫자는 설명용입니다. 교사의 실수도 함께 전달될 수 있습니다.

    | 방식 | 학생이 받는 신호 | 대응 손실 |
    | --- | --- | --- |
    | 일반 SFT | 데이터의 정답 토큰 | CE |
    | 응답 텍스트 증류 | 교사가 생성한 문장을 정답으로 사용 | 그 텍스트의 SFT·CE |
    | 이 실습의 logits 증류 | 같은 문맥에서 교사의 전체 다음 토큰 분포 | CE + KL |

    - 사람이 쓴 답변을 씁니다. 교사가 만든 문장은 쓰지 않습니다.
    - 두 모델은 **같은 정답 앞부분**을 보고 다음 토큰을 예측합니다(teacher forcing).
    - 학생은 교사를 복사하지 않습니다. 자기 가중치를 학습합니다.
    - 크기는 달라도 됩니다. 단, 토큰 ID의 뜻은 같아야 합니다.

    **예측:** 두 교사의 1등 토큰이 같으면 KL 신호도 같을까요?
    **해설:** 아닙니다. 나머지 확률이 다르면 다릅니다. CE는 정답 확률만, KL은 전체 분포를 봅니다.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    # ▸ 실습 설정 위젯 — 실행하면 슬라이더·선택 상자가 나타납니다 (marimo 표시 코드)
    temperature_ui = mo.ui.slider(1.0, 4.0, step=0.5, value=2.0, label="증류 온도 τ")
    alpha_ui = mo.ui.slider(0.0, 1.0, step=0.25, value=0.5, label="CE 비중 α")
    mo.hstack([temperature_ui, alpha_ui])
    return alpha_ui, temperature_ui


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. 원본 손실을 세 후보로 직접 계산하기

    logits는 softmax 전 점수입니다. 교사 $z_t$, 학생 $z_s$, 온도 $\tau>0$에서

    $$p_t=\mathrm{softmax}(z_t/\tau),\quad \log p_s=\mathrm{logsoftmax}(z_s/\tau)$$
    $$L_{KD}=\tau^2\frac{1}{N}\sum_{i=1}^N\sum_{v=1}^V
    p_t(i,v)\log\frac{p_t(i,v)}{p_s(i,v)},\qquad
    L=\alpha L_{CE}+(1-\alpha)L_{KD}$$

    - **KL 방향:** 교사 → 학생. `kl_div(학생 log 확률, 교사 확률)` 순서입니다.
    - **CE:** 온도를 쓰지 않습니다.
    - **α:** `α=1`이면 CE만, `α=0`이면 KL만. 원본 기본값은 `α=0.5, τ=1.5`.
    - **온도 τ:** 높이면 분포가 부드러워집니다.
    - **τ²:** 온도로 작아진 gradient를 보정하는 관례입니다. 성능을 보장하지는 않습니다.

    **예측:** 온도를 바꾸면 순위도 바뀔까요? KL과 `τ²×KL`은 어떻게 달라질까요?
    """)
    return


@app.cell
def _(F, temperature_ui, torch):
    # 직접 계산: 온도로 나눈 교사 확률과 학생 log 확률로 KL 항을 구합니다.
    _tau = temperature_ui.value
    demo_student_logits = torch.tensor([[1.0, 1.5, -0.5]], requires_grad=True)
    demo_teacher_logits = torch.tensor([[4.0, 2.0, 0.0]], requires_grad=True)
    demo_teacher_probs = F.softmax(demo_teacher_logits.detach() / _tau, dim=-1)
    demo_student_logps = F.log_softmax(demo_student_logits / _tau, dim=-1)
    demo_kl_terms = demo_teacher_probs * (demo_teacher_probs.log() - demo_student_logps)
    return (
        demo_kl_terms,
        demo_student_logits,
        demo_student_logps,
        demo_teacher_logits,
        demo_teacher_probs,
    )


@app.cell
def _(
    F,
    alpha_ui,
    demo_kl_terms,
    demo_student_logits,
    demo_teacher_logits,
    distillation_loss,
    temperature_ui,
    torch,
):
    # 원본 distillation_loss로 계산해 직접 계산(τ² × KL 합)과 비교합니다.
    _tau = temperature_ui.value
    demo_kd = distillation_loss(demo_student_logits, demo_teacher_logits, temperature=_tau)
    demo_ce = F.cross_entropy(demo_student_logits, torch.tensor([0]))
    demo_mixed = alpha_ui.value * demo_ce + (1 - alpha_ui.value) * demo_kd
    demo_student_logits.grad = None   # α만 바꿔 다시 실행해도 gradient가 쌓이지 않게 합니다.
    demo_mixed.backward()
    assert torch.allclose(_tau ** 2 * demo_kl_terms.sum(), demo_kd)
    assert demo_teacher_logits.grad is None and torch.isfinite(demo_student_logits.grad).all()
    assert torch.allclose(distillation_loss(demo_student_logits.repeat(2, 1), demo_teacher_logits.repeat(2, 1), _tau), demo_kd)
    return demo_ce, demo_kd, demo_mixed


@app.cell(hide_code=True)
def _(F, demo_ce, demo_kd, demo_kl_terms, demo_mixed, demo_student_logits, demo_student_logps, demo_teacher_logits, demo_teacher_probs, mo):
    # ▸ 결과 표시 — 실행하면 표·그림이 나타납니다 (marimo 표시 코드)
    mo.vstack([
        mo.ui.table([
            {"후보": word, "hard 정답": int(i == 0), "교사 τ=1": float(F.softmax(demo_teacher_logits.detach(), -1)[0, i]),
             "교사 pτ": float(demo_teacher_probs[0, i]), "학생 pτ": float(demo_student_logps.detach().exp()[0, i]),
             "KL 기여": float(demo_kl_terms.detach()[0, i]), "학생 gradient": float(demo_student_logits.grad[0, i])}
            for i, word in enumerate(["야옹", "멍멍", "음메"])
        ], selection=None),
        mo.md(f"CE={demo_ce.item():.5f}, KL(보정 전)={demo_kl_terms.sum().item():.5f}, "
              f"τ²×KL={demo_kd.item():.5f}, 혼합={demo_mixed.item():.5f}. 교사 gradient: `{demo_teacher_logits.grad}`.\n\n"
              "**해설:** 온도로 나눠도 순위는 같습니다. KL 항 하나는 음수일 수 있지만 합은 0 이상입니다. "
              "N이 두 배여도 평균은 같습니다. 교사에는 gradient가 없습니다."),
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. 대화 JSONL → 실제 SFTDataset

    SFT 노트북과 같은 교육용 template입니다. 차이는 하나입니다.

    - 이번 사전은 **줄바꿈도 토큰**입니다. 그래서 원본과 같은 경계 패턴이 보입니다:
      시작 `[BOS]assistant\n`, 끝 `[EOS]\n`.
    - system을 직접 넣어 무작위 system 추가를 막습니다(SFT 노트북 3절).
    - 첫 대화는 두 턴이라 정답이 더 많습니다.
    """)
    return


@app.cell
def _(PreTrainedTokenizerFast, Regex, Tokenizer, json, models, output_dir, pre_tokenizers):
    pairs = [("고양이", "야옹"), ("강아지", "멍멍"), ("소", "음메"), ("오리", "꽥꽥")]
    conversations = [
        [{"role": "system", "content": "동물 소리 를 답하세요 ."},
         {"role": "user", "content": f"{animal} 소리 는 ?"},
         {"role": "assistant", "content": f"{sound} 입니다 ."}]
        for animal, sound in pairs
    ]
    conversations[0] += [
        {"role": "user", "content": "다시 답하세요 ."},
        {"role": "assistant", "content": "야옹 입니다 ."},
    ]
    vocabulary = ["[PAD]", "[BOS]", "[EOS]", "[UNK]", "\n", "system", "user", "assistant"] + sorted(
        {word for messages in conversations for message in messages for word in message["content"].split()}
    )
    _backend = Tokenizer(models.WordLevel({w: i for i, w in enumerate(vocabulary)}, unk_token="[UNK]"))
    _backend.pre_tokenizer = pre_tokenizers.Split(Regex(r"\n|[^\s]+"), behavior="removed", invert=True)
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=_backend, bos_token="[BOS]", eos_token="[EOS]", pad_token="[PAD]",
        unk_token="[UNK]", clean_up_tokenization_spaces=False, model_max_length=64,
    )
    tokenizer.chat_template = (
        "{% for message in messages %}"
        "{{ bos_token + message['role'] + '\n' + message['content'] + eos_token + '\n' }}"
        "{% endfor %}{% if add_generation_prompt %}{{ bos_token + 'assistant\n' }}{% endif %}"
    )
    data_path = output_dir / "sft.jsonl"
    data_path.write_text("".join(json.dumps({"conversations": c}, ensure_ascii=False) + "\n" for c in conversations), encoding="utf-8")
    tokenizer.save_pretrained(output_dir / "tokenizer")
    assert tokenizer("\n", add_special_tokens=False).input_ids == [4]
    return conversations, data_path, pairs, tokenizer, vocabulary


@app.cell
def _(DataLoader, SFTDataset, data_path, tokenizer, torch):
    sft_dataset = SFTDataset(str(data_path), tokenizer, max_length=64)
    input_ids, labels = next(iter(DataLoader(sft_dataset, batch_size=4, shuffle=False, num_workers=0)))
    valid_mask = labels[:, 1:] != -100
    assert input_ids.shape == labels.shape == (4, 64)
    assert (labels[input_ids == tokenizer.pad_token_id] == -100).all()
    assert not (input_ids == tokenizer.unk_token_id).any()
    assert torch.equal(valid_mask.sum(dim=1), torch.tensor([10, 5, 5, 5]))
    return (input_ids, labels, sft_dataset, valid_mask,)


@app.cell(hide_code=True)
def _(input_ids, labels, mo, sft_dataset, tokenizer, valid_mask):
    # ▸ 결과 표시 — 실행하면 표·그림이 나타납니다 (marimo 표시 코드)
    mo.vstack([
        mo.md(f"첫 샘플의 템플릿:\n\n```text\n{sft_dataset.create_chat_prompt(sft_dataset.samples[0]['conversations'])}\n```\n\n"
              f"입력·labels shape: `{tuple(input_ids.shape)}`, 유효 정답 수: `{valid_mask.sum(dim=1).tolist()}`"),
        mo.ui.table([
            {"위치 t": t, "입력 토큰": tokenizer.convert_ids_to_tokens(int(input_ids[0, t])).replace("\n", "↵"),
             "label[t]": int(labels[0, t]),
             "logits[t]의 다음 토큰": tokenizer.convert_ids_to_tokens(int(input_ids[0, t + 1])).replace("\n", "↵"),
             "label[t+1]": int(labels[0, t + 1]), "CE·KL 포함": bool(valid_mask[0, t])}
            for t in range(63)
        ], selection=None, page_size=18),
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 4. 한 칸 이동과 마스크: KL을 적용할 위치

    마스크 규칙은 SFT와 같습니다(assistant 본문+EOS만 정답).
    다른 점: 증류 trainer는 **labels를 모델에 넘기지 않고 밖에서 직접 옮깁니다.**

    ```python
    student_logits = model(input_ids).logits[:, :-1, :]  # [B, L-1, V]
    shift_labels = labels[:, 1:]                        # [B, L-1]
    loss_mask = shift_labels != -100
    selected_student = student_logits[loss_mask]        # [N, V]
    ```

    - CE는 정답 ID를 씁니다. KL은 **전체 어휘 분포**를 씁니다.
    - N은 유효 정답 토큰 수입니다.
    - `batchmean`은 첫 차원으로 나눕니다. 그래서 `[N,V]`로 넣어야 토큰 평균이 됩니다.
    - `[B,L-1,V]`를 넣으면 B로 나눠서 크기가 달라집니다.

    **예측:** 첫 응답 토큰을 예측하는 logits는 어디일까요?
    **해설:** 바로 앞, assistant 헤더의 마지막 줄바꿈 위치입니다.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    # ▸ 실습 설정 위젯 — 실행하면 슬라이더·선택 상자가 나타납니다 (marimo 표시 코드)
    truncation_ui = mo.ui.dropdown([8, 24, 64], value=24, label="마스크 관찰용 max_length (학습에는 64 사용)")
    truncation_ui
    return (truncation_ui,)


@app.cell
def _(SFTDataset, data_path, tokenizer, truncation_ui):
    _short = SFTDataset(str(data_path), tokenizer, max_length=truncation_ui.value)
    short_ids, short_labels = _short[0]
    return (short_ids, short_labels,)


@app.cell(hide_code=True)
def _(mo, short_ids, short_labels, tokenizer, truncation_ui):
    # ▸ 결과 설명 — 실행하면 계산 값이 들어간 설명이 나타납니다 (marimo 표시 코드)
    mo.md(f'길이={truncation_ui.value}, 유효 정답={int((short_labels[1:] != -100).sum())}개. 토큰: `{tokenizer.convert_ids_to_tokens(short_ids.tolist())}`\n\nSFTDataset은 자른 뒤 EOS를 다시 붙이지 않습니다. 응답이 모두 잘리면 정답이 0개입니다. 이때 KL이 NaN이 될 수 있습니다. 아래 함수는 이런 배치를 거부합니다. 길이 8/24/64를 비교하세요.')
    return


@app.cell
def _(F, distillation_loss):
    def compute_losses(student_logits, teacher_logits, targets, tau):
        # 원본 train_epoch의 Dense 손실 경로. CE·KD를 따로 반환합니다.
        shifted = targets[:, 1:].contiguous()
        mask = shifted != -100
        if not mask.any():
            raise ValueError("assistant 유효 정답이 없습니다. 데이터/절단 길이를 확인하세요.")
        s = student_logits[:, :-1, :].contiguous()
        t = teacher_logits[:, :-1, :].contiguous()
        if s.shape != t.shape:
            raise ValueError("같은 위치·같은 토큰 ID 의미의 [B,L,V] logits가 필요합니다.")
        raw_ce = F.cross_entropy(s.reshape(-1, s.size(-1)), shifted.reshape(-1), ignore_index=-100, reduction="none")
        ce = (raw_ce * mask.reshape(-1)).sum() / (mask.sum() + 1e-8)
        kd = distillation_loss(s[mask], t[mask], temperature=tau)
        return ce, kd
    return (compute_losses,)


@app.cell
def _(mo):
    mo.md("""
    ## 5. CPU에서 교사를 먼저 준비하기

    | | 구조 | 은닉 | 층 |
    | --- | --- | --- | --- |
    | 교사 | Dense | 48 | 2 |
    | 학생 | Dense | 24 | 1 |

    어휘와 template은 같습니다.

    1. 교사를 SFT 160 step으로 학습합니다.
    2. `eval()`과 `requires_grad_(False)`로 교사를 고정합니다.
    3. optimizer에는 학생 매개변수만 넣습니다.

    **원본과 다른 점:** 원본은 SFT된 체크포인트를 읽습니다(기본 **MoE 교사 → Dense 학생**).
    여기서는 **Dense 교사를 직접 학습하고, 학생은 무작위로 시작**합니다.
    """)
    return


@app.cell
def _(MiniMindConfig, MiniMindForCausalLM, input_ids, labels, torch, vocabulary):
    def new_model(width, layers, seed):
        config = MiniMindConfig(
            hidden_size=width, num_hidden_layers=layers, vocab_size=len(vocabulary),
            num_attention_heads=4, num_key_value_heads=2, intermediate_size=width * 2,
            max_position_embeddings=64, dropout=0.0, flash_attn=False, use_moe=False,
        )
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            return MiniMindForCausalLM(config).cpu()

    teacher = new_model(48, 2, 42)
    student_initial = new_model(24, 1, 7)
    _optimizer = torch.optim.AdamW(teacher.parameters(), lr=0.005, weight_decay=0.0)
    teacher_history = []
    teacher_steps = 160  # 과제 5에서는 1로 바꿉니다.
    assert teacher_steps >= 1
    teacher.train()
    with torch.no_grad():
        teacher_initial_ce = teacher(input_ids, labels=labels).loss.item()
    for _step in range(teacher_steps):
        _optimizer.zero_grad(set_to_none=True)
        _loss = teacher(input_ids, labels=labels).loss
        assert torch.isfinite(_loss)
        _loss.backward()
        torch.nn.utils.clip_grad_norm_(teacher.parameters(), 1.0)
        _optimizer.step()
        teacher_history.append(_loss.item())
    _optimizer.zero_grad(set_to_none=True)
    teacher.eval().requires_grad_(False)
    with torch.no_grad():
        teacher_logits = teacher(input_ids).logits.detach()
        teacher_final_ce = teacher(input_ids, labels=labels).loss.item()
    assert torch.isfinite(torch.tensor(teacher_final_ce))
    # history는 갱신 전 손실입니다. 한 step이어도 갱신 전후를 별도로 비교합니다.
    assert teacher_final_ce < teacher_initial_ce
    return (new_model, student_initial, teacher, teacher_final_ce, teacher_history, teacher_initial_ce, teacher_logits, teacher_steps,)


@app.cell(hide_code=True)
def _(mo, student_initial, teacher, teacher_final_ce, teacher_initial_ce, teacher_steps):
    # ▸ 결과 표시 — 실행하면 표·그림이 나타납니다 (marimo 표시 코드)
    mo.vstack([mo.md(f"교사 {teacher_steps} step, 갱신 전후 CE: `{teacher_initial_ce:.4f} → {teacher_final_ce:.4f}`. 손실 감소가 좋은 교사를 보장하지는 않습니다."), mo.ui.table([
        {"역할": role, "은닉": model.config.hidden_size, "층": model.config.num_hidden_layers,
         "고유 매개변수 수": sum(p.numel() for p in model.parameters()),
         "학습 가능 수": sum(p.numel() for p in model.parameters() if p.requires_grad)}
        for role, model in [("동결된 교사", teacher), ("초기 학생", student_initial)]
    ], selection=None)])
    return


@app.cell
def _(mo):
    mo.md("""
    ## 6. 동일 초기 학생: CE만 학습 vs CE+KL

    같은 초기 학생 두 개를 비교합니다.

    - 기준선: `α=1` (CE만)
    - 증류 학생: 위 UI의 `α, τ`
    - 공통: 같은 배치, AdamW, lr 0.005, 80 step, clip 1.0

    교사 logits는 한 번 계산해 재사용합니다(배치가 고정이라 가능).
    원본은 배치마다 `no_grad()`로 교사를 실행합니다.

    CE와 KL은 크기 척도가 다릅니다. `α=0.5`라도 기여가 반반은 아닙니다.

    **예측:** KL이 낮은 학생이 정답도 더 잘 맞힐까요?
    """)
    return


@app.cell
def _(alpha_ui, compute_losses, copy, input_ids, labels, student_initial, teacher, teacher_logits, temperature_ui, torch):
    trained_students, histories = {}, {}
    _before = {n: t.detach().clone() for n, t in teacher.state_dict().items()}
    for _name, _alpha in [("CE only", 1.0), ("CE + KD", alpha_ui.value)]:
        _student = copy.deepcopy(student_initial).train()
        _opt = torch.optim.AdamW(_student.parameters(), lr=0.005, weight_decay=0.0)
        _history = []
        for _step in range(80):
            _opt.zero_grad(set_to_none=True)
            _ce, _kd = compute_losses(_student(input_ids).logits, teacher_logits, labels, temperature_ui.value)
            _loss = _alpha * _ce + (1 - _alpha) * _kd
            assert torch.isfinite(_loss)
            _loss.backward()
            _grad = torch.nn.utils.clip_grad_norm_(_student.parameters(), 1.0)
            assert torch.isfinite(_grad) and _grad > 0
            _opt.step()
            _history.append({"step": _step, "ce": _ce.item(), "kd": _kd.item(), "total": _loss.item()})
        trained_students[_name] = _student.eval()
        histories[_name] = _history
    assert all(p.grad is None and not p.requires_grad for p in teacher.parameters())
    assert all(torch.equal(_before[n], t) for n, t in teacher.state_dict().items())
    return histories, trained_students


@app.cell
def _(F, compute_losses, input_ids, labels, teacher_logits, temperature_ui, torch, valid_mask):
    def evaluate_model(model):
        with torch.no_grad():
            logits = model(input_ids).logits
            ce, kd = compute_losses(logits, teacher_logits, labels, temperature_ui.value)
            pred = logits[:, :-1].argmax(-1)[valid_mask]
            accuracy = (pred == labels[:, 1:][valid_mask]).float().mean().item()
            agreement = (pred == teacher_logits[:, :-1].argmax(-1)[valid_mask]).float().mean().item()
            kl1 = F.kl_div(F.log_softmax(logits[:, :-1][valid_mask], -1), F.softmax(teacher_logits[:, :-1][valid_mask], -1), reduction="batchmean")
        return {"CE": ce.item(), "τ² KL": kd.item(), "KL (τ=1)": kl1.item(), "정답률": accuracy, "교사 1등 일치율": agreement}
    return (evaluate_model,)


@app.cell
def _(evaluate_model, histories, plt, student_initial, teacher, teacher_history, trained_students):
    evaluation = {name: evaluate_model(model) for name, model in [("Teacher", teacher), ("Initial student", student_initial), *trained_students.items()]}
    student_curve_fig, _axes = plt.subplots(1, 3, figsize=(12, 3))
    _axes[0].plot(teacher_history)
    _axes[0].set(title="Teacher SFT", xlabel="step", ylabel="CE")
    for _name, _history in histories.items():
        _axes[1].plot([h["ce"] for h in _history], label=_name)
        _axes[2].plot([h["kd"] for h in _history], label=_name)
    _axes[1].set(title="Student hard targets", xlabel="step", ylabel="CE")
    _axes[2].set(title="Student soft targets", xlabel="step", ylabel="tau^2 KL")
    _axes[1].legend()
    _axes[2].legend()
    student_curve_fig.tight_layout()
    return (evaluation, student_curve_fig,)


@app.cell(hide_code=True)
def _(evaluation, metrics, mo, student_curve_fig):
    # ▸ 결과 표시 — 실행하면 표·그림이 나타납니다 (marimo 표시 코드)
    mo.vstack([
        mo.ui.table([{"model": name, **metrics} for name, metrics in evaluation.items()], selection=None), student_curve_fig,
        mo.md('**해설:** 정답률은 다음 토큰을 맞힌 비율입니다. 교사 1등 일치율은 교사와 argmax가 같은 비율입니다.\n\n둘 다 학습한 4개 대화에서 측정하며 EOS·줄바꿈을 포함합니다. 다음 절의 생성 결과도 확인하세요. 새 질문에 대한 성능은 측정하지 않습니다.\n\n곡선은 갱신 전, 표는 마지막 갱신 후 손실입니다. 교사의 가중치 불변과 gradient 부재도 assert로 확인했습니다.'),
    ])
    return


@app.cell
def _(mo):
    mo.md("""
    ## 7. 같은 질문에서 확률과 자유 생성 확인하기

    - 확률은 **τ=1**로 계산합니다(증류 온도 아님).
    - 전체 softmax에서 소리 후보 네 개만 꺼냅니다. 그래서 합이 1이 아닐 수 있습니다.
    - 생성은 greedy입니다. 정답 없이 자기 출력을 이어 갑니다.
    - 그래서 teacher forcing 지표와 생성 결과가 다를 수 있습니다.
    """)
    return


@app.cell
def _(conversations, pairs, teacher, tokenizer, torch, trained_students):
    generation_rows, probability_rows = [], []
    for _i, (_animal, _answer) in enumerate(pairs):
        _prompt_ids = tokenizer.apply_chat_template(conversations[_i][:2], tokenize=True, add_generation_prompt=True)
        _prompt = torch.tensor([_prompt_ids])
        for _name, _model in [("Teacher", teacher), *trained_students.items()]:
            with torch.no_grad():
                _prob = _model(_prompt).logits[0, -1].softmax(-1)
                _result = _model.generate(_prompt, max_new_tokens=6, do_sample=False, eos_token_id=tokenizer.eos_token_id,
                                          temperature=1.0, top_k=0, top_p=1.0, use_cache=True)
            _new_ids = _result[0, len(_prompt_ids):].tolist()
            generation_rows.append({"질문 동물": _animal, "기대 첫 토큰": _answer, "model": _name,
                                    "생성 토큰": " ".join(tokenizer.convert_ids_to_tokens(_new_ids)).replace("\n", "↵"),
                                    "첫 토큰 정답": bool(_new_ids and _new_ids[0] == tokenizer.convert_tokens_to_ids(_answer))})
            probability_rows.append({"질문 동물": _animal, "model": _name,
                                     **{sound: float(_prob[tokenizer.convert_tokens_to_ids(sound)]) for _, sound in pairs}})
    return (generation_rows, probability_rows,)


@app.cell(hide_code=True)
def _(generation_rows, mo, probability_rows):
    # ▸ 결과 표시 — 실행하면 표·그림이 나타납니다 (marimo 표시 코드)
    mo.vstack([mo.ui.table(probability_rows, selection=None), mo.ui.table(generation_rows, selection=None)])
    return


@app.cell
def _(alpha_ui, evaluation, generation_rows, hashlib, histories, json, output_dir, probability_rows, repo_root, teacher_steps, temperature_ui, uuid4):
    experiment_dir = output_dir / f"students-{uuid4().hex[:6]}"
    experiment_dir.mkdir()
    _sources = ["projects/minimind/trainer/train_distillation.py", "projects/minimind/dataset/lm_dataset.py", "projects/minimind/model/model_minimind.py", "notebooks/minimind/minimind_distillation.py"]
    metrics = {
        "alpha": alpha_ui.value, "temperature": temperature_ui.value,
        "teacher_steps": teacher_steps, "student_steps": 80, "teacher_seed": 42, "student_seed": 7,
        "evaluation_split": "training dialogues only", "evaluation": evaluation,
        "generation": generation_rows, "probabilities": probability_rows, "histories": histories,
        "source_sha256": {p: hashlib.sha256((repo_root / p).read_bytes()).hexdigest() for p in _sources},
    }
    (experiment_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    return experiment_dir, metrics


@app.cell(hide_code=True)
def _(experiment_dir, mo):
    # ▸ 결과 설명 — 실행하면 계산 값이 들어간 설명이 나타납니다 (marimo 표시 코드)
    mo.md(f"""
    ## 8. 원본 코드로 돌아가기와 수강생 과제

    | 원본 위치 | 실습 대응 |
    | --- | --- |
    | `SFTDataset.create_chat_prompt/generate_labels` | 대화 템플릿·두 응답 마스크·EOS+줄바꿈 |
    | `train_epoch`: `logits[..., :-1, :]`, `labels[..., 1:]` | 다음 토큰 정렬과 `[N,V]` 선택 |
    | `distillation_loss` | 원본 직접 호출, 수동 KL과 일치 검증 |
    | `teacher.eval/requires_grad_(False)`, `no_grad` | 교사 동결·logits 캐시·불변 검증 |
    | `α * ce_loss + (1-α) * distill_loss` | 같은 초기 학생 두 개 비교 |

    **어휘 크기 주의:** 원본은 교사 logits를 학생 어휘 크기로 자릅니다.
    **자른다고 토크나이저가 맞춰지지는 않습니다.**

    - 같은 ID가 같은 토큰을 뜻해야 합니다.
    - 자른 뒤 softmax하면 원래 분포와 달라집니다.
    - 이 실습은 **같은 토크나이저, 같은 어휘 크기**를 씁니다.

    **MoE 학생이면:** `ce_loss`에 aux가 더해집니다.
    그래서 전체는 `α×CE + α×aux + (1-α)×KD`. `α=0`이면 aux도 사라집니다.

    **이 실습에서 안 하는 것:** aux, AMP, DDP, 누적, 체크포인트 재개.

    **α 기본값:** `train_epoch`은 0, CLI는 0.5를 넘깁니다. 교사가 없으면 KD=0이라 전체는 `α×CE`입니다.

    **과제 — 먼저 예측하고, 한 번에 하나씩 바꾸세요.**

    1. `α=1`: 두 학생의 결과가 같아야 합니다. 확인하세요.
    2. `α=0`과 `0.5`: CE, KL(τ=1), 첫 소리 정답 수를 적으세요.
       교사를 닮는 것과 정답을 맞히는 것은 같을까요?
    3. `τ=1/2/4`: 후보 확률, gradient, 최종 지표를 적으세요.
       `τ² KL`만 보지 말고 공통 `KL(τ=1)`도 보세요.
    4. 관찰 길이를 8로 바꾸세요. 왜 정답이 0개일까요?
       실제 학습에서는 이런 배치를 거르거나 길이를 늘려야 합니다.
    5. 5절의 `teacher_steps`를 1로 줄이세요. 틀린 교사는 학생에게 무엇을 가르칠까요?
       **힌트:** 교사 CE와 첫 소리 정답 수부터 보세요.

    **완료 기준:**

    - KL 방향, 온도, α를 설명할 수 있다.
    - 첫 응답을 예측하는 위치를 찾을 수 있다.
    - 교사는 gradient가 없고 학생만 바뀐 것을 확인했다.

    일반화 평가는 별도 검증 데이터가 필요합니다.
    지표·곡선·생성·소스 해시: `{experiment_dir / 'metrics.json'}`

    **다음 실습:** `minimind_dpo.py`.
    다음은 **두 응답 중 어느 쪽이 나은가**를 학습합니다.
    """)
    return


if __name__ == "__main__":
    app.run()
