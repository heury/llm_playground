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
    `uv run marimo edit minimind_distillation_tour.py`. 환경·준비물과 권장 순서는 [공통 README](README.md)를 참고하세요.
    위젯 변경 시 의존 셀이 다시 실행됩니다. 모델 크기·데이터·학습 설정 변경 전에는 해당 절의 재실행 범위를 확인하세요.
    """)
    return


@app.cell(hide_code=True)
def _():
    import copy
    import json
    import random
    import sys
    import tempfile
    import time
    from pathlib import Path

    import datasets
    import marimo as mo
    import matplotlib.pyplot as plt
    import torch
    import torch.nn.functional as F
    from datasets import load_dataset
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer

    # 이 노트북은 GPU에서 학습합니다. 198M 교사 모델을 매 step 함께 돌리므로 CPU로는 너무 느립니다.
    mo.stop(
        not torch.cuda.is_available(),
        mo.md("**NVIDIA GPU가 필요합니다.** `torch.cuda.is_available()`가 `False`입니다. CUDA용 PyTorch와 드라이버를 확인하세요."),
    )

    # 노트북 위치(notebooks/minimind/)에서 두 단계 위가 저장소 루트입니다.
    repo_root = Path(__file__).resolve().parents[2]
    minimind_dir = repo_root / "projects/minimind"
    sys.path.insert(0, str(minimind_dir))

    # train_distillation.py가 쓰는 원본 클래스와 함수를 그대로 가져옵니다.
    from dataset.lm_dataset import SFTDataset
    from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
    from trainer.train_distillation import distillation_loss
    from trainer.trainer_utils import SkipBatchSampler, get_lr, setup_seed

    device = "cuda"

    def autocast_ctx():
        """train_distillation.py의 '3. 혼합 정밀도 설정': GPU에서 bf16 autocast"""
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)

    datasets.disable_progress_bars()
    tokenizer = AutoTokenizer.from_pretrained(minimind_dir / "model")
    return (
        DataLoader,
        F,
        MiniMindConfig,
        MiniMindForCausalLM,
        SFTDataset,
        SkipBatchSampler,
        autocast_ctx,
        copy,
        device,
        distillation_loss,
        get_lr,
        hf_hub_download,
        json,
        load_dataset,
        load_file,
        mo,
        plt,
        random,
        setup_seed,
        tempfile,
        time,
        tokenizer,
        torch,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # MiniMind 지식 증류(Knowledge Distillation) 따라가기

    **지식 증류(KD)** 는 큰 **교사(teacher)** 모델의 출력을 보고 작은 **학생(student)** 모델을 학습시키는 방법입니다.
    SFT는 정답 토큰 하나만 보여 줍니다(**hard label**). 증류는 교사가 매긴 **다음 토큰 확률 분포 전체**를 함께 보여 줍니다(**soft label**).
    MiniMind에서는 `projects/minimind/trainer/train_distillation.py`가 이 일을 합니다.

    | 역할 | 모델 | 구조 | 전체 매개변수 | 토큰당 계산에 쓰는 매개변수 |
    | --- | --- | --- | --- | --- |
    | 교사 | [`caspar/MiniMind-MoE-198M-A64M-Mixed-SFT-v1.0`](https://huggingface.co/caspar/MiniMind-MoE-198M-A64M-Mixed-SFT-v1.0) | MoE (전문가 4개, top-1) | 198M | 약 64M |
    | 학생 | [`caspar/MiniMind-Dense-64M-Mixed-SFT-v1.0`](https://huggingface.co/caspar/MiniMind-Dense-64M-Mixed-SFT-v1.0) | Dense | 64M | 64M |

    원본 스크립트의 기본 설정(`--teacher_use_moe 1`, `--student_use_moe 0`)과 같은 **MoE → Dense** 증류입니다.
    두 모델은 같은 데이터(FineWeb-Edu 사전 학습 → UltraChat·SmolTalk·GSM8K SFT)로 학습되었고,
    토크나이저·어휘(6,400개)·hidden size(768)·층 수(8)가 같습니다. 다른 것은 FFN이 전문가 4개인지 1개인지뿐입니다.

    **준비:** **NVIDIA GPU가 필요합니다.** 처음 실행할 때 Hugging Face에서 가중치(약 128MB + 397MB)와
    GSM8K 데이터(약 1MB)를 내려받습니다. GPU 메모리는 약 4GB를 쓰고, RTX 3080 Ti 기준 전체 실행에 약 1~2분이 걸립니다.

    **사전 지식:** SFT 노트북(`minimind_sft_tour.py`)의 채팅 템플릿, `SFTDataset`의 assistant 정답 마스크, 학습 루프.

    ## 0. SFT와 무엇이 다른가

    `train_distillation.py`는 `train_full_sft.py`와 **데이터·학습 루프가 같고, 손실만 다릅니다.**

    $$
    \mathcal{L}=\alpha\,\mathcal{L}_{\mathrm{CE}}+(1-\alpha)T^2 D_{\mathrm{KL}}(p_{\mathrm{teacher},T}\Vert p_{\mathrm{student},T}).
    $$

    $\alpha$는 정답 CE의 비중, $T$는 softmax 온도입니다. 같은 유효 답변 토큰에서 두 항을 평균합니다.

    | | `train_full_sft.py` | `train_distillation.py` |
    | --- | --- | --- |
    | 모델 | 학생 하나 | 학생(학습) + **교사(고정, `no_grad`)** |
    | 정답 | 정답 토큰 (hard label) | 정답 토큰 + **교사의 확률 분포 (soft label)** |
    | 손실 | CE | **$\alpha\,\mathrm{CE}+(1-\alpha)\,\mathcal{L}_{\mathrm{distill}}$** (`--alpha 0.5`, `--temperature 1.5`) |
    | 학습률 기본값 | 1e-5 | 5e-6 |
    | 데이터 | `SFTDataset` | `SFTDataset` (같음) |

    1~3절은 soft label과 손실 함수를 실제 모델 출력으로 뜯어보고, 4~6절은 학생을 실제로 증류 학습시켜 결과를 비교합니다.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. 교사와 학생 불러오기

    원본은 `init_model(lm_config, from_weight)`로 `../out/full_sft_768.pth`와 `../out/full_sft_768_moe.pth`를 불러옵니다.
    여기서는 Hugging Face에서 **설정(`config.json`)과 가중치(`model.safetensors`)만** 버전 `v1.0`으로 고정해 받고,
    저장소의 원본 `MiniMindForCausalLM`에 넣습니다(모델 저장소의 `trust_remote_code`용 코드는 실행하지 않음).
    교사는 원본처럼 `eval()` + `requires_grad_(False)`로 고정합니다.
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
    STUDENT_REPO = "caspar/MiniMind-Dense-64M-Mixed-SFT-v1.0"
    TEACHER_REPO = "caspar/MiniMind-MoE-198M-A64M-Mixed-SFT-v1.0"
    REVISION = "v1.0"  # 버전 고정

    def load_minimind(repo):
        config = MiniMindConfig(**json.load(open(hf_hub_download(repo, "config.json", revision=REVISION))))
        model = MiniMindForCausalLM(config)
        state = {k: v.float() for k, v in load_file(hf_hub_download(repo, "model.safetensors", revision=REVISION)).items()}
        missing, unexpected = model.load_state_dict(state, strict=False)
        # lm_head는 Embedding과 같은 가중치라 파일에 따로 저장되지 않습니다.
        assert missing == ["lm_head.weight"] and not unexpected
        return model.to(device).eval()

    student_base = load_minimind(STUDENT_REPO)
    teacher = load_minimind(TEACHER_REPO)
    teacher.eval()
    teacher.requires_grad_(False)  # 원본 '5. 학생 모델 및 교사 모델 정의'와 같음

    def count_params(model):
        total = sum(p.numel() for p in model.parameters())
        cfg = model.config
        if not cfg.use_moe:
            return total, total
        # top-k 라우팅: 토큰마다 전문가 num_experts 중 num_experts_per_tok개만 계산합니다.
        expert = sum(p.numel() for n, p in model.named_parameters() if ".mlp.experts." in n)
        return total, total - expert * (cfg.num_experts - cfg.num_experts_per_tok) // cfg.num_experts

    param_rows = [
        {"역할": name, "모델": repo, "use_moe": m.config.use_moe,
         "전체 매개변수": f"{count_params(m)[0] / 1e6:.1f}M", "토큰당 활성 매개변수": f"{count_params(m)[1] / 1e6:.1f}M"}
        for name, repo, m in [("교사", TEACHER_REPO, teacher), ("학생", STUDENT_REPO, student_base)]
    ]
    assert teacher.config.vocab_size == student_base.config.vocab_size == 6400
    return param_rows, student_base, teacher


@app.cell(hide_code=True)
def _(mo, param_rows, result_table, torch):
    mo.vstack([
        mo.md(f"실행 장치: **GPU (`{torch.cuda.get_device_name(0)}`)**"),
        result_table(param_rows, selection=None),
        mo.md(r"""
    교사는 매개변수가 3배지만 토큰마다 전문가 하나만 쓰므로 **계산량은 학생과 비슷합니다.**
    MoE의 장점(같은 계산량으로 더 많은 지식 저장)을 **배포하기 쉬운 Dense 모델로 옮기는 것**이 이 증류의 목적입니다.
    두 모델의 어휘가 같으므로 원본의 `teacher_logits[..., :vocab_size_student]` 자르기는 여기서는 아무것도 바꾸지 않습니다.
    """),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. 실습 데이터: GSM8K test에서 나눈 교육용 학습·확인 묶음

    두 모델은 GSM8K **train**으로 SFT되었고, 모델 카드에 따르면 GSM8K **test**는 평가에만 썼습니다.
    이 기록만으로 사전 학습 단계의 노출이나 표현을 바꾼 중복까지 배제할 수는 없습니다.
    이 실습은 test 일부를 증류 학습에 재사용하므로 결과를 **표준 GSM8K test 성능으로 보고하지 않습니다.**
    아래 두 묶음은 이 실습 안에서 서로 겹치지 않습니다.

    - **증류 학습용 128문제:** test의 0~127번
    - **평가용 256문제:** test의 1000~1255번 (학습에 쓰지 않음)

    각 문제를 `{"user": 질문, "assistant": 풀이}` 대화 하나로 만들어 원본 `SFTDataset`에 넣습니다(`max_seq_len` 340, 원본 기본값과 같음).
    따라서 정답 마스크(assistant 답변만), 무작위 system 추가·빈 think 삭제도 원본과 같습니다.
    """)
    return


@app.cell
def _(SFTDataset, json, load_dataset, random, tempfile, tokenizer, torch):
    GSM8K_REVISION = "740312add88f781978c0658806c59bc2815b9866"  # openai/gsm8k 커밋 고정
    gsm8k_test = load_dataset("openai/gsm8k", "main", split="test", revision=GSM8K_REVISION)
    train_rows = [gsm8k_test[i] for i in range(0, 128)]
    eval_rows = [gsm8k_test[i] for i in range(1000, 1256)]

    def to_conversation(row):
        return {"conversations": [{"role": "user", "content": row["question"]},
                                  {"role": "assistant", "content": row["answer"]}]}

    _data_dir = tempfile.mkdtemp(prefix="minimind_kd_data_")
    kd_train_path, kd_eval_path = f"{_data_dir}/kd_train.jsonl", f"{_data_dir}/kd_eval.jsonl"
    for _path, _rows in [(kd_train_path, train_rows), (kd_eval_path, eval_rows)]:
        with open(_path, "w", encoding="utf-8") as _f:
            for _row in _rows:
                _f.write(json.dumps(to_conversation(_row), ensure_ascii=False) + "\n")

    MAX_SEQ_LEN = 340  # 원본 --max_seq_len 기본값
    kd_train_ds = SFTDataset(kd_train_path, tokenizer, max_length=MAX_SEQ_LEN)
    _eval_ds = SFTDataset(kd_eval_path, tokenizer, max_length=MAX_SEQ_LEN)

    # 평가 묶음은 무작위 가공(system 추가, think 삭제)을 시드로 한 번만 고정해 미리 만들어 둡니다.
    random.seed(0)
    _pairs = [_eval_ds[i] for i in range(len(_eval_ds))]
    eval_x = torch.stack([x for x, _ in _pairs])
    eval_y = torch.stack([y for _, y in _pairs])
    eval_target_tokens = int((eval_y[:, 1:] != -100).sum())
    assert len(kd_train_ds) == 128 and eval_x.shape == (256, MAX_SEQ_LEN)
    return (
        eval_rows,
        eval_target_tokens,
        eval_x,
        eval_y,
        kd_train_ds,
        train_rows,
    )


@app.cell(hide_code=True)
def _(eval_target_tokens, mo, train_rows):
    def _cell(text):  # 표 안에 넣기: 줄바꿈 → <br>, '$'가 수식으로 해석되지 않게 이스케이프
        return text.replace("$", r"\$").replace("\n", "<br>")

    mo.md(f"""
    학습 문제 예시 (test 0번):

    | user (질문) | assistant (정답 풀이) |
    | --- | --- |
    | {_cell(train_rows[0]["question"])} | {_cell(train_rows[0]["answer"])} |

    풀이 안의 `<<16-3-4=9>>`는 GSM8K의 계산기 표시이고, 마지막 `#### 18`이 최종 답입니다.
    평가용 256문제의 정답(assistant) 토큰은 **{eval_target_tokens:,}개**입니다. 이 토큰들에서 손실과 KL을 잽니다.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. soft label: 교사는 정답 말고 무엇을 알려 주나

    SFT의 정답은 **one-hot**입니다. "다음 토큰은 `18`" 하나뿐이고, 다른 모든 토큰은 똑같이 0점입니다.
    교사의 출력은 **분포**입니다. "`18`일 확률 0.6, `9`일 확률 0.2, ..."처럼 **어떤 오답이 그럴듯한지**까지 알려 줍니다.
    Hinton 등(2015)은 이 정보를 **dark knowledge**라고 불렀습니다.

    **온도(temperature) $T$** 는 분포를 얼마나 평평하게 펼칠지 정합니다.

    $$p_i^{(T)} = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}$$

    $T=1$이면 원래 분포, $T>1$이면 작은 확률이 커져서 2등·3등 후보의 차이가 잘 보입니다.
    평가 문제 하나에서 교사가 망설이는 위치(1등 확률 30~60%)를 찾아 직접 봅니다.
    """)
    return


@app.cell
def _(F, device, eval_x, eval_y, student_base, teacher, tokenizer, torch):
    @torch.no_grad()
    def answer_positions_logits(model, x, y):
        """정답 토큰 위치의 다음 토큰 logits (fp32). 원본처럼 logits[:, :-1]과 labels[:, 1:]을 맞춥니다."""
        logits = model(x.to(device)).logits[:, :-1].float()
        mask = y[:, 1:].to(device) != -100
        return logits[mask], y[:, 1:].to(device)[mask], mask

    _x, _y = eval_x[:1], eval_y[:1]
    _t_logits, _labels, _mask = answer_positions_logits(teacher, _x, _y)
    _s_logits, _, _ = answer_positions_logits(student_base, _x, _y)
    _t_probs = F.softmax(_t_logits, dim=-1)
    _top1 = _t_probs.max(-1).values
    _candidates = ((_top1 > 0.3) & (_top1 < 0.6)).nonzero().flatten()
    assert len(_candidates) > 0
    dk_pos = int(_candidates[0])

    # 문맥: 이 위치 앞의 마지막 토큰 25개
    _positions = _mask[0].nonzero().flatten()          # 정답 위치들 (logits 기준 인덱스)
    _ctx_end = int(_positions[dk_pos]) + 1              # logits[k]는 input[:k+1]을 보고 input[k+1]을 예측
    dk_context = tokenizer.decode(_x[0, max(0, _ctx_end - 25):_ctx_end])
    dk_label = tokenizer.decode([int(_labels[dk_pos])])

    def top_tokens(logits, temperature, k=6):
        probs = F.softmax(logits / temperature, dim=-1)
        values, ids = probs.topk(k)
        return [(tokenizer.decode([int(i)]), float(v)) for v, i in zip(values, ids)]

    dk_teacher_logits = _t_logits[dk_pos]
    dk_student_logits = _s_logits[dk_pos]

    # 한 문제 전체에서: 교사 분포가 얼마나 '정보'를 담는가
    _entropy = -(_t_probs * _t_probs.clamp_min(1e-12).log()).sum(-1)
    dk_stats = {
        "정답 토큰 수": len(_labels),
        "교사 1등 ≠ 정답": int((_t_probs.argmax(-1) != _labels).sum()),
        "교사 1등 확률 평균": float(_top1.mean()),
        "교사 분포 엔트로피 평균 (nats)": float(_entropy.mean()),
    }
    return (
        answer_positions_logits,
        dk_context,
        dk_label,
        dk_stats,
        dk_student_logits,
        dk_teacher_logits,
        top_tokens,
    )


@app.cell(hide_code=True)
def _(mo):
    dk_temperature = mo.ui.slider(start=0.5, stop=5.0, step=0.5, value=1.5, label="온도 T", show_value=True)
    dk_temperature
    return (dk_temperature,)


@app.cell(hide_code=True)
def _(
    F,
    dk_context,
    dk_label,
    dk_stats,
    dk_student_logits,
    dk_teacher_logits,
    dk_temperature,
    mo,
    plt,
    tokenizer,
    top_tokens,
):
    _T = dk_temperature.value
    _ids = F.softmax(dk_teacher_logits, -1).topk(8).indices
    _names = [repr(tokenizer.decode([int(i)])) for i in _ids]
    _fig, _ax = plt.subplots(figsize=(7, 2.8))
    for _k, (_temp, _color) in enumerate([(1.0, "tab:blue"), (_T, "tab:orange")]):
        _p = F.softmax(dk_teacher_logits / _temp, -1)[_ids].cpu()
        _ax.bar([i + _k * 0.4 for i in range(8)], _p, width=0.4, color=_color, label=f"teacher T={_temp}")
    _ax.set_xticks([i + 0.2 for i in range(8)], _names, fontsize=8)
    _ax.set_ylabel("probability")
    _ax.legend()
    _fig.tight_layout()
    _plot = mo.as_html(_fig)
    plt.close(_fig)

    def _fmt(pairs):
        return "<br>".join(f"`{t!r}` {p:.3f}" for t, p in pairs)

    mo.vstack([
        mo.md(f"**문맥 끝부분:** `…{dk_context!r}` → **정답 토큰 `{dk_label!r}`**"),
        mo.md(f"""
    | one-hot 정답 (SFT) | 교사 T=1 | 교사 T={_T} | 학생 T=1 |
    | --- | --- | --- | --- |
    | `{dk_label!r}` 1.000<br>나머지 0 | {_fmt(top_tokens(dk_teacher_logits, 1.0))} | {_fmt(top_tokens(dk_teacher_logits, _T))} | {_fmt(top_tokens(dk_student_logits, 1.0))} |
    """),
        _plot,
        mo.md(" · ".join(f"{k}: **{v:.3f}**" if isinstance(v, float) else f"{k}: **{v}**" for k, v in dk_stats.items())
              + " (평가 문제 1개)"),
        mo.md(r"""
    - 교사는 정답 외 후보에도 확률을 나눠 줍니다. 학생은 "정답은 무엇"뿐 아니라 "**무엇과 헷갈릴 만한가**"를 배웁니다.
    - 온도를 올리면 2등 이하 후보의 확률이 커져서 그 정보가 손실에 더 크게 반영됩니다. 너무 올리면 모든 토큰이 비슷해져 정보가 흐려집니다.
    - 교사도 틀립니다(교사 1등 ≠ 정답). 그래서 원본은 정답 CE와 증류 손실을 **섞습니다**(`alpha`).
    """),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. 증류 손실 `distillation_loss` 뜯어보기

    원본 함수 전체입니다.

    ```python
    def distillation_loss(student_logits, teacher_logits, temperature=1.0, reduction='batchmean'):
        with torch.no_grad():
            teacher_probs = F.softmax(teacher_logits / temperature, dim=-1).detach()
        student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
        kl = F.kl_div(student_log_probs, teacher_probs, reduction=reduction)
        return (temperature ** 2) * kl
    ```

    짚을 점 세 가지:

    1. **`F.kl_div(input, target)`의 순서.** `input`은 학생의 **log 확률**, `target`은 교사의 **확률**입니다.
       계산 결과는 $\sum_i p^{teacher}_i (\log p^{teacher}_i - \log p^{student}_i) = \mathrm{KL}(p^{teacher} \,\|\, p^{student})$,
       즉 **교사 분포 기준의 forward KL**입니다. 교사가 확률을 준 토큰은 학생도 모두 확률을 주도록 만듭니다.
    2. **`reduction='batchmean'`.** 합을 첫 번째 차원 크기로 나눕니다. 원본은 `[B, L, V]`를 $[BL,V]$로 펴고
       정답 위치만 골라 넣으므로 **정답 토큰 하나당 평균 KL**이 됩니다(`'mean'`은 어휘 6,400까지 나눠 버려 틀린 값).
    3. **`temperature ** 2` 곱하기.** 아래 4-1에서 직접 확인합니다.

    아래 셀은 원본 함수를 손으로 다시 계산해 같은지 확인합니다.
    """)
    return


@app.cell
def _(
    F,
    answer_positions_logits,
    distillation_loss,
    eval_x,
    eval_y,
    student_base,
    teacher,
    torch,
):
    # 평가 문제 4개의 정답 위치 logits로 확인합니다.
    kd_t_logits, _, _ = answer_positions_logits(teacher, eval_x[:4], eval_y[:4])
    kd_s_logits, kd_labels, _ = answer_positions_logits(student_base, eval_x[:4], eval_y[:4])

    def manual_kd(s_logits, t_logits, T):
        p_t = F.softmax(t_logits / T, dim=-1)
        log_p_t = F.log_softmax(t_logits / T, dim=-1)
        log_p_s = F.log_softmax(s_logits / T, dim=-1)
        kl_per_token = (p_t * (log_p_t - log_p_s)).sum(-1)   # KL(teacher ‖ student), 토큰마다
        return T**2 * kl_per_token.mean()                     # batchmean = 토큰 평균

    kd_check_rows = []
    for _T in [1.0, 1.5, 3.0]:
        _orig = distillation_loss(kd_s_logits, kd_t_logits, temperature=_T)
        _mine = manual_kd(kd_s_logits, kd_t_logits, _T)
        assert torch.allclose(_orig, _mine, rtol=1e-4, atol=1e-6)
        kd_check_rows.append({"T": _T, "원본 distillation_loss": f"{_orig.item():.5f}", "직접 계산": f"{_mine.item():.5f}",
                              "'mean'으로 했다면": f"{(_T**2 * F.kl_div(F.log_softmax(kd_s_logits / _T, -1), F.softmax(kd_t_logits / _T, -1), reduction='mean')).item():.7f}"})
    return kd_check_rows, kd_s_logits, kd_t_logits


@app.cell(hide_code=True)
def _(kd_check_rows, kd_s_logits, mo, result_table):
    mo.vstack([
        mo.md(f"평가 문제 4개, 정답 토큰 **{kd_s_logits.shape[0]}개** (`[토큰 수, 어휘] = {list(kd_s_logits.shape)}`)"),
        result_table(kd_check_rows, selection=None),
        mo.md("원본과 직접 계산이 같습니다. `'mean'`이었다면 값이 6,400분의 1로 작아져 CE와 섞을 때 증류 손실이 거의 무시됩니다."),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 4-1. 왜 $T^2$를 곱하나: 온도와 기울기 크기

    학생 logit $z_i$에 대한 기울기는 다음과 같습니다(토큰 하나 기준).

    $$\frac{\partial}{\partial z_i}\, T^2 \cdot \mathrm{KL}\big(p^{teacher,(T)} \,\|\, p^{student,(T)}\big) = T\,\big(p^{student,(T)}_i - p^{teacher,(T)}_i\big)$$

    $T$가 크면 두 분포가 모두 평평해져 차이 $(p^s - p^t)$가 대략 $1/T$로 작아집니다. 그래서 $T^2$를 곱해야
    기울기 크기가 **$T$와 상관없이 비슷하게** 유지됩니다. 곱하지 않으면 기울기가 $1/T^2$로 줄어서,
    온도를 바꿀 때마다 `alpha`와 학습률을 다시 맞춰야 합니다. 위의 실제 logits로 확인합니다.
    """)
    return


@app.cell
def _(F, kd_s_logits, kd_t_logits):
    t2_rows = []
    for _T in [1.0, 1.5, 2.0, 4.0, 8.0]:
        _norms = {}
        for _scale_name, _scale in [("T² 곱함 (원본)", _T**2), ("T² 없음", 1.0)]:
            _z = kd_s_logits.clone().requires_grad_(True)
            _kl = F.kl_div(F.log_softmax(_z / _T, -1), F.softmax(kd_t_logits / _T, -1), reduction="batchmean")
            (_scale * _kl).backward()
            _norms[_scale_name] = _z.grad.norm().item()
            # 해석적 기울기와 일치 확인: T·(p_s − p_t) / 토큰 수 (batchmean 때문)
            if _scale_name == "T² 곱함 (원본)":
                _expected = _T * (F.softmax(kd_s_logits / _T, -1) - F.softmax(kd_t_logits / _T, -1)) / len(_z)
                assert (_z.grad - _expected).norm() / _expected.norm() < 1e-4
        t2_rows.append({"T": _T, "KL 값 (T² 없음)": f"{_kl.item():.4f}",
                        "기울기 norm, T² 곱함 (원본)": f"{_norms['T² 곱함 (원본)']:.5f}",
                        "기울기 norm, T² 없음": f"{_norms['T² 없음']:.5f}"})
    return (t2_rows,)


@app.cell(hide_code=True)
def _(mo, result_table, t2_rows):
    mo.vstack([
        result_table(t2_rows, selection=None),
        mo.md(r"""
    - **T² 없음:** T=1 → 8에서 기울기가 수백 분의 1로 줄어듭니다. T=4만 되어도 증류 손실이 CE 옆에서 거의 무시됩니다.
    - **T² 곱함:** 원본이 권장하는 T=1~2에서는 기울기 크기가 거의 같습니다. T=4·8에서는 두 분포 자체가 비슷해져(KL 값도 작아짐)
      몇 배 줄지만, T² 없이 줄어드는 정도에 비하면 훨씬 작습니다. 그래서 원본은 $\alpha\,\mathrm{CE}+(1-\alpha)\,\mathcal{L}_{\mathrm{distill}}$을 그대로 더할 수 있습니다.
    - 해석적 기울기 $T(p^s - p^t)$와 autograd 결과가 같음도 확인했습니다.
    """),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. 학습 전: 교사와 학생은 얼마나 다른가

    평가 256문제의 정답 토큰에서 세 가지를 잽니다. 이후 모든 비교에 같은 지표를 씁니다.

    | 지표 | 뜻 | 좋은 방향 |
    | --- | --- | --- |
    | **정답 CE** | 실제 GSM8K 풀이 토큰의 평균 −log p (SFT 손실과 같음) | 낮을수록 |
    | **KL(교사‖학생)** | T=1에서 교사 분포와 학생 분포의 차이 | 낮을수록 교사와 닮음 |
    | **1등 일치율** | 학생의 1등 토큰이 교사의 1등 토큰과 같은 비율 | 높을수록 교사와 닮음 |
    """)
    return


@app.cell
def _(F, autocast_ctx, device, eval_x, eval_y, teacher, torch):
    @torch.no_grad()
    def evaluate(model, batch_size=16):
        """평가 256문제: 정답 CE, KL(교사‖학생), 1등 일치율 (정답 토큰 평균)"""
        model.eval()
        ce = kl = agree = n = 0.0
        for i in range(0, len(eval_x), batch_size):
            x, y = eval_x[i:i + batch_size].to(device), eval_y[i:i + batch_size].to(device)
            with autocast_ctx():
                s = model(x).logits[:, :-1].float()
                t = teacher(x).logits[:, :-1].float()
            mask = y[:, 1:] != -100
            s, t, target = s[mask], t[mask], y[:, 1:][mask]
            ce += F.cross_entropy(s, target, reduction="sum").item()
            kl += F.kl_div(F.log_softmax(s, -1), F.log_softmax(t, -1), log_target=True, reduction="sum").item()
            agree += (s.argmax(-1) == t.argmax(-1)).sum().item()
            n += mask.sum().item()
        return {"ce": ce / n, "kl": kl / n, "agree": agree / n}

    return (evaluate,)


@app.cell
def _(evaluate, student_base, teacher):
    teacher_eval = evaluate(teacher)
    student_eval = evaluate(student_base)
    assert teacher_eval["kl"] < 1e-3 and teacher_eval["agree"] > 0.99  # 교사 자신과의 비교
    return student_eval, teacher_eval


@app.cell(hide_code=True)
def _(mo, student_eval, teacher_eval):
    mo.vstack([
        mo.md(f"""
    | 모델 | 정답 CE | KL(교사‖학생) | 1등 일치율 |
    | --- | --- | --- | --- |
    | 교사 (MoE 198M) | {teacher_eval['ce']:.3f} | 0 | 100% |
    | 학생 (Dense 64M, 학습 전) | {student_eval['ce']:.3f} | {student_eval['kl']:.3f} | {student_eval['agree']:.1%} |
    """),
        mo.md(r"""
    - 학생은 교사와 **정답 토큰의 약 5분의 1에서 1등이 다릅니다.** 같은 데이터로 학습했어도 두 모델은 꽤 다른 분포를 냅니다.
    - **GSM8K test에서는 두 모델의 정답 CE가 거의 같습니다(오히려 학생이 조금 낮음).** 모델 카드의 전체 검증 PPL은 교사 3.54, 학생 3.79로 교사가 낫지만,
      이 데이터에서는 교사가 크게 앞서지 않습니다. 교사가 학생보다 **크게 낫지 않으면** 증류의 효과도 제한됩니다.
      6절에서 이 점이 결과에 그대로 드러납니다.
    """),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. 증류 학습

    ### 6-1. 학습 루프

    `run_distillation`은 `train_distillation.py`의 "8. 학습 시작" + `train_epoch`를 옮긴 함수입니다(DDP·wandb·체크포인트 제외).
    한 step의 순서는 원본과 같습니다.

    1. 학생 순전파 (bf16 autocast) → `student_logits = logits[..., :-1, :]`
    2. 교사 순전파 (`no_grad`, autocast 밖 = fp32) → `teacher_logits[..., :vocab_size_student]`
    3. `ce_loss`: 정답 마스크(`labels[..., 1:] != -100`) 위치의 평균 CE. 학생이 MoE면 `aux_loss`를 더함(여기는 Dense라 없음)
    4. `distill_loss`: **정답 마스크 위치만** 골라 `distillation_loss`
    5. `loss = alpha * ce_loss + (1 - alpha) * distill_loss` → backward → clip(1.0) → AdamW

    원본은 bf16에서 `GradScaler(enabled=False)`이므로 scaler는 생략했습니다. epoch마다 5절의 평가를 합니다.
    """)
    return


@app.cell
def _(
    DataLoader,
    F,
    SkipBatchSampler,
    autocast_ctx,
    copy,
    device,
    distillation_loss,
    evaluate,
    get_lr,
    setup_seed,
    student_base,
    teacher,
    time,
    torch,
):
    def run_distillation(train_ds, alpha, temperature, learning_rate, epochs=5, batch_size=8, grad_clip=1.0):
        """train_distillation.py의 '8. 학습 시작' + train_epoch. 학생 사본을 학습해 (모델, 기록)을 돌려줍니다."""
        setup_seed(42)
        model = copy.deepcopy(student_base).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        history = {"step_ce": [], "step_distill": [], "eval": [], "epoch_logs": []}
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = time.time()
        for epoch in range(epochs):
            setup_seed(42 + epoch); indices = torch.randperm(len(train_ds)).tolist()
            loader = DataLoader(train_ds, batch_sampler=SkipBatchSampler(indices, batch_size, 0))
            iters = len(loader)
            model.train()
            for step, (input_ids, labels) in enumerate(loader, start=1):
                input_ids, labels = input_ids.to(device), labels.to(device)
                loss_mask = (labels[..., 1:] != -100).float()
                lr = get_lr(epoch * iters + step, epochs * iters, learning_rate)
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr

                # 순전파(학생 모델)
                with autocast_ctx():
                    res = model(input_ids)
                    student_logits = res.logits[..., :-1, :].contiguous()

                # 교사 모델 순전파(eval 및 no_grad에서만 수행)
                with torch.no_grad():
                    teacher_logits = teacher(input_ids).logits[..., :-1, :].contiguous()
                    teacher_logits = teacher_logits[..., :student_logits.size(-1)]

                # 1) Ground-Truth CE Loss
                shift_labels = labels[..., 1:].contiguous()
                loss_mask_flat = loss_mask.view(-1)
                ce_loss = F.cross_entropy(student_logits.view(-1, student_logits.size(-1)), shift_labels.view(-1),
                                          ignore_index=-100, reduction="none")
                ce_loss = torch.sum(ce_loss * loss_mask_flat) / (loss_mask_flat.sum() + 1e-8)
                if model.config.use_moe:
                    ce_loss = ce_loss + res.aux_loss

                # 2) Distillation Loss (정답 마스크 위치만)
                distill_loss = distillation_loss(
                    student_logits.view(-1, student_logits.size(-1))[loss_mask_flat == 1],
                    teacher_logits.view(-1, teacher_logits.size(-1))[loss_mask_flat == 1],
                    temperature=temperature,
                )

                # 3) 전체 손실 = alpha * CE + (1-alpha) * Distill
                loss = alpha * ce_loss + (1 - alpha) * distill_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                history["step_ce"].append(ce_loss.item())
                history["step_distill"].append(distill_loss.item())

            history["epoch_logs"].append(
                f"Epoch:[{epoch + 1}/{epochs}]({iters}/{iters}), loss: {loss.item():.4f}, ce: {ce_loss.item():.4f}, "
                f"distill: {distill_loss.item():.4f}, learning_rate: {lr:.8f}")
            history["eval"].append(evaluate(model))
        torch.cuda.synchronize()
        history["seconds"] = time.time() - start
        history["peak_gb"] = torch.cuda.max_memory_allocated() / 1e9
        return model.eval(), history

    return (run_distillation,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 6-2. 세 가지 실험 + 대조군

    모두 학습 128문제, batch 8, 5 epochs(80 step), 온도 1.5, 학습률 5e-5입니다.
    원본 기본 학습률(5e-6)은 수만 step용이라 80 step에서는 변화가 거의 보이지 않아, 차이가 보이도록 10배로 올렸습니다.

    | 실험 | `alpha` | 손실 | 의미 |
    | --- | --- | --- | --- |
    | **A** | 1.0 | CE만 | 교사 없이 SFT (기준) |
    | **B** (원본 기본값) | 0.5 | CE + 증류 반반 | 정답과 교사를 함께 따름 |
    | **C** | 0.0 | 증류만 | 교사 분포 모사. 정답 CE는 직접 최적화하지 않지만 정답 접두문맥은 입력에 사용 |
    | **A′** (대조군) | 1.0 | CE만, 학습률 2.5e-5 | B의 CE 가중치가 절반이라 좋아진 것인지 확인 |

    GPU에서 실험 하나에 약 10초 걸립니다.
    """)
    return


@app.cell
def _(kd_train_ds, run_distillation):
    KD_TEMPERATURE, KD_LR = 1.5, 5e-5
    kd_experiments = {
        "A: CE only (alpha=1)": dict(alpha=1.0, temperature=KD_TEMPERATURE, learning_rate=KD_LR),
        "B: alpha=0.5": dict(alpha=0.5, temperature=KD_TEMPERATURE, learning_rate=KD_LR),
        "C: KD only (alpha=0)": dict(alpha=0.0, temperature=KD_TEMPERATURE, learning_rate=KD_LR),
        "A': CE only, lr/2": dict(alpha=1.0, temperature=KD_TEMPERATURE, learning_rate=KD_LR / 2),
    }
    kd_models, kd_histories = {}, {}
    for _name, _cfg in kd_experiments.items():
        kd_models[_name], kd_histories[_name] = run_distillation(kd_train_ds, **_cfg)
    return KD_LR, KD_TEMPERATURE, kd_experiments, kd_histories, kd_models


@app.cell(hide_code=True)
def _(
    kd_experiments,
    kd_histories,
    mo,
    plt,
    result_table,
    student_eval,
    teacher_eval,
):
    _fig, _axes = plt.subplots(1, 3, figsize=(12, 3.2))
    _colors = {"A: CE only (alpha=1)": "tab:red", "B: alpha=0.5": "tab:green",
               "C: KD only (alpha=0)": "tab:blue", "A': CE only, lr/2": "tab:pink"}
    for _name, _h in kd_histories.items():
        _ls = "--" if _name.startswith("A'") else "-"
        _axes[0].plot(_h["step_ce"], lw=0.8, color=_colors[_name], ls=_ls, label=_name)
        for _ax, _key in [(_axes[1], "ce"), (_axes[2], "kl")]:
            _ax.plot(range(0, 6), [student_eval[_key]] + [e[_key] for e in _h["eval"]], marker="o", ms=3,
                     color=_colors[_name], ls=_ls, label=_name)
    _axes[1].axhline(teacher_eval["ce"], color="gray", ls=":", label="teacher")
    _axes[0].set(title="train CE (128 problems)", xlabel="step")
    _axes[1].set(title="held-out CE (lower = better)", xlabel="epoch")
    _axes[2].set(title="held-out KL(teacher || student)", xlabel="epoch")
    _axes[1].legend(fontsize=7)
    _fig.tight_layout()
    _plot = mo.as_html(_fig)
    plt.close(_fig)

    _rows = [{"모델": "교사", "alpha": "-", "정답 CE": f"{teacher_eval['ce']:.3f}", "KL(교사‖학생)": "0",
              "1등 일치율": "100%", "가장 좋았던 CE (epoch)": "-"},
             {"모델": "학생 (학습 전)", "alpha": "-", "정답 CE": f"{student_eval['ce']:.3f}", "KL(교사‖학생)": f"{student_eval['kl']:.3f}",
              "1등 일치율": f"{student_eval['agree']:.1%}", "가장 좋았던 CE (epoch)": "-"}]
    for _name, _h in kd_histories.items():
        _last = _h["eval"][-1]
        _best = min(range(len(_h["eval"])), key=lambda i: _h["eval"][i]["ce"])
        _rows.append({"모델": _name, "alpha": kd_experiments[_name]["alpha"], "정답 CE": f"{_last['ce']:.3f}",
                      "KL(교사‖학생)": f"{_last['kl']:.3f}", "1등 일치율": f"{_last['agree']:.1%}",
                      "가장 좋았던 CE (epoch)": f"{_h['eval'][_best]['ce']:.3f} ({_best + 1})"})
    mo.vstack([
        _plot,
        mo.md("**5 epochs 후 평가 256문제** (학습에 없던 문제)"),
        result_table(_rows, selection=None),
        mo.md("**실험 B 로그 (원본 형식, epoch 끝마다)**\n```text\n" + "\n".join(kd_histories["B: alpha=0.5"]["epoch_logs"]) + "\n```"),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **읽는 법** (GPU·버전에 따라 숫자는 조금 달라질 수 있습니다)

    - **A (CE만):** 첫 epoch에 정답 CE가 크게 내려갑니다. GSM8K 풀이 형식(`<<…>>`, `####`)에 맞춰지기 때문입니다.
      그 뒤로는 학습 CE는 계속 내려가는데 **평가 CE는 다시 올라갑니다.** 128문제를 외우는 **과적합**입니다.
      KL도 계속 커져 이 평가 문맥에서 교사 분포와 멀어집니다. KL만으로 일반 능력의 보존 여부를 판단할 수는 없습니다.
    - **B (alpha=0.5):** 평가 CE가 끝까지 내려가 **네 실험 중 가장 낮습니다.** KL은 처음 수준이거나 조금 낮습니다.
      정답에서 새 형식을 배우면서, 교사 분포에 가까이 머물게 하는 **규제(regularizer)** 역할을 합니다.
      정답 하나만이 아니라 교사가 고른 후보들에도 확률을 남기므로, label smoothing보다 정보가 많은 규제입니다.
    - **A′ (CE만, 학습률 절반):** B는 CE 가중치가 0.5라서 "학습률을 절반으로 줄인 것과 같지 않나?"라는 의문을 확인합니다.
      A′도 2 epoch 뒤부터 과적합하며 B보다 나쁩니다. 같은 학습량에서 과적합을 막은 것은 **교사 신호**입니다.
      다만 학습률을 원본 기본값(5e-6)까지 낮추면 CE만으로도 과적합 없이 비슷한 CE에 도달합니다. 이때도 KL은 계속 커져
      교사(원래 분포)에서 멀어지는데, B는 KL을 지킵니다(과제 2).
    - **C (증류만):** KL이 꾸준히 줄고 1등 일치율이 오릅니다. **학생이 교사를 닮아 갑니다.**
      하지만 정답 CE는 **교사의 CE 근처**까지만 갑니다. 순수 증류의 목적은 교사 분포를 모사하는 것이며, 정답 CE 개선을 직접 최적화하지 않습니다.
      이번 관찰이 모든 과제에서 학생 성능의 상한이 교사라는 뜻은 아닙니다.
      5절에서 본 것처럼 이 데이터에서 교사가 학생보다 크게 낫지 않아서 CE 이득이 거의 없습니다.

    **정리:** 증류로 얻는 것은 (1) 교사 쪽으로 다가가기, (2) 작은 데이터에서 과적합을 막는 규제입니다.
    (1)의 크기는 **교사와 학생의 실력 차이**에 달려 있고, 실제로는 교사가 훨씬 큰 모델일 때 효과가 큽니다.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 6-3. 직접 해 보기: alpha와 온도

    `alpha`와 온도를 바꿔 학습해 보고 위 실험과 비교하세요. 버튼을 누를 때만 학습합니다(약 10초).
    먼저 결과를 예측해 보세요. 예: "alpha=0.5에서 온도 4는 1.5보다 KL이 커질까, 작아질까?"
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    try_alpha = mo.ui.slider(start=0.0, stop=1.0, step=0.1, value=0.5, label="alpha", show_value=True)
    try_temperature = mo.ui.dropdown(options={"1.0": 1.0, "1.5": 1.5, "2.0": 2.0, "4.0": 4.0}, value="4.0", label="temperature")
    try_lr = mo.ui.dropdown(options={"5e-6 (원본 기본값)": 5e-6, "5e-5": 5e-5, "1e-4": 1e-4}, value="5e-5", label="learning_rate")
    try_button = mo.ui.run_button(label="학습 실행")
    mo.hstack([try_alpha, try_temperature, try_lr, try_button], justify="start", gap=2)
    return try_alpha, try_button, try_lr, try_temperature


@app.cell(hide_code=True)
def _(
    kd_train_ds,
    mo,
    run_distillation,
    try_alpha,
    try_button,
    try_lr,
    try_temperature,
):
    mo.stop(not try_button.value, mo.md("버튼을 누르면 학습합니다."))
    _, try_history = run_distillation(kd_train_ds, alpha=try_alpha.value, temperature=try_temperature.value,
                                      learning_rate=try_lr.value)
    try_label = f"alpha={try_alpha.value}, T={try_temperature.value}, lr={try_lr.value}"
    return try_history, try_label


@app.cell(hide_code=True)
def _(kd_histories, result_table, try_history, try_label):
    _rows = [{"설정": name, **{f"epoch {i + 1} CE / KL": f"{e['ce']:.3f} / {e['kl']:.3f}" for i, e in enumerate(h["eval"])}}
             for name, h in [(try_label + " (내 실험)", try_history)] + list(kd_histories.items())]
    result_table(_rows, selection=None)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7. 비용: 교사를 함께 돌리는 값

    온라인 증류(원본 방식)는 **매 step 교사 순전파**를 합니다. 실험 A도 원본처럼 교사를 돌리므로(`distill` 로그용),
    여기서는 한 batch에서 학생 학습과 교사 순전파 시간을 따로 잽니다.
    """)
    return


@app.cell
def _(
    autocast_ctx,
    copy,
    device,
    kd_histories,
    kd_train_ds,
    student_base,
    teacher,
    torch,
):
    _x = torch.stack([kd_train_ds[i][0] for i in range(8)]).to(device)
    _model = copy.deepcopy(student_base).train()
    _opt = torch.optim.AdamW(_model.parameters(), lr=1e-6)

    def _student_step():
        with autocast_ctx():
            _model(_x).logits.float().mean().backward()
        _opt.step(); _opt.zero_grad(set_to_none=True)

    def _teacher_forward():
        with torch.no_grad():
            teacher(_x)

    def _timeit(fn, reps=10):
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        _start = torch.cuda.Event(enable_timing=True); _end = torch.cuda.Event(enable_timing=True)
        _start.record()
        for _ in range(reps):
            fn()
        _end.record(); torch.cuda.synchronize()
        return _start.elapsed_time(_end) / reps

    cost = {
        "student_ms": _timeit(_student_step),
        "teacher_ms": _timeit(_teacher_forward),
        "seconds_per_run": kd_histories["B: alpha=0.5"]["seconds"],
        # 실험 A는 첫 실험이라 학습된 학생 사본이 아직 메모리에 없습니다.
        "peak_gb": kd_histories["A: CE only (alpha=1)"]["peak_gb"],
        # 오프라인 증류용으로 교사 logits를 모두 저장한다면 (fp16, 학습 문제 하나 = 340 x 6400)
        "full_logits_mb_per_sample": 340 * 6400 * 2 / 1e6,
    }
    del _model, _opt
    return (cost,)


@app.cell(hide_code=True)
def _(cost, mo):
    mo.md(f"""
    | 항목 | 값 |
    | --- | --- |
    | 학생 학습 1 step (bf16 순전파 + 역전파 + AdamW, batch 8 × 340) | {cost['student_ms']:.1f} ms |
    | 교사 순전파 1회 (fp32, 원본처럼 autocast 밖) | {cost['teacher_ms']:.1f} ms |
    | 실험 B 전체 (80 step + 평가 5번) | {cost['seconds_per_run']:.1f} s |
    | 최대 GPU 메모리 (실험 A: 교사 + 원래 학생 + 학습 중인 학생 + AdamW + logits) | {cost['peak_gb']:.2f} GB |
    | 교사 logits 전체 저장 시 (fp16, 문제 하나) | {cost['full_logits_mb_per_sample']:.1f} MB |

    - 교사 순전파는 역전파가 없어 학생 학습보다 싸지만, **fp32라서** 무시할 수 없는 시간을 씁니다. 교사도 bf16으로 돌리면 더 빨라집니다.
    - 교사 가중치는 GPU 메모리에 계속 올라가 있어야 합니다. 큰 교사라면 이것이 가장 큰 제약입니다.
    - **오프라인 증류:** 교사 출력을 미리 저장해 두면 학습 중 교사가 필요 없습니다. 하지만 전체 분포는 문제당 수 MB라서,
      실제로는 **상위 k개 토큰의 확률만** 저장하거나, 교사가 **생성한 답변을 정답으로 SFT**(sequence-level KD)하는 방법을 많이 씁니다.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 8. 생성 결과 비교

    평가 문제 2개에 greedy로 답하게 합니다. 모델 카드 안내대로 프롬프트의 빈 think 블록을 지웁니다.
    두 모델의 GSM8K 정답률은 약 1%이므로 **맞고 틀림보다 풀이 형식과 말투**를 보세요.
    """)
    return


@app.cell
def _(device, eval_rows, kd_models, student_base, teacher, tokenizer, torch):
    @torch.inference_mode()
    def ask(model, question, max_new_tokens=160):
        text = tokenizer.apply_chat_template([{"role": "user", "content": question}], tokenize=False,
                                             add_generation_prompt=True).replace("<think>\n\n</think>\n\n", "")
        x = torch.tensor([tokenizer(text, add_special_tokens=False).input_ids], device=device)
        y = model.generate(inputs=x, max_new_tokens=max_new_tokens, do_sample=False, temperature=1.0,
                           top_p=1.0, top_k=0, eos_token_id=tokenizer.eos_token_id)
        new = y[0, x.shape[1]:].tolist()
        if tokenizer.eos_token_id in new:
            new = new[: new.index(tokenizer.eos_token_id)]
        return tokenizer.decode(new, skip_special_tokens=True).strip()

    _models = {"교사": teacher, "학생 (학습 전)": student_base,
               **{k: v for k, v in kd_models.items() if not k.startswith("A'")}}
    generation_rows = []
    for _row in eval_rows[:2]:
        generation_rows.append({"모델": "📝 문제", "답": _row["question"]})
        generation_rows.append({"모델": "✅ 정답 풀이", "답": _row["answer"]})
        generation_rows += [{"모델": _name, "답": ask(_m, _row["question"])} for _name, _m in _models.items()]
    return (generation_rows,)


@app.cell(hide_code=True)
def _(generation_rows, mo, result_table):
    mo.vstack([
        result_table(generation_rows, selection=None, page_size=14),
        mo.md(r"""
    - **교사와 학습 전 학생**은 GSM8K 형식이 아니라 UltraChat·SmolTalk식 설명 문장("First, let's calculate ...")으로 답합니다.
      SFT 데이터의 대부분이 일반 대화였기 때문입니다.
    - **C (증류만)** 는 교사와 **같은 첫 문장**으로 시작하는 경우가 많습니다. 정답 풀이의 접두문맥은 입력으로 쓰지만 hard-label CE로 풀이 형식을 직접 최적화하지 않으므로, 이 실험에서는 교사의 말투에 가까운 출력을 보입니다.
    - **A·B**는 128문제의 풀이 형식(`<<…>>` 계산 표시, `#### 답`)을 그대로 따릅니다. 하지만 계산은 틀립니다.
      형식은 적은 데이터로도 배우지만, 계산 능력은 이 크기의 모델·데이터로 생기지 않습니다.
    - 모든 모델이 같은 문장을 반복하기 쉽습니다(작은 모델 + greedy). 손실·KL 차이가 작아도 greedy 생성은 한 토큰이 바뀌면 뒤가 전부 달라지므로,
      생성 두 개로 결론 내리지 말고 6절의 지표와 함께 보세요.
    """),
    ])
    return


@app.cell(hide_code=True)
def _(KD_LR, KD_TEMPERATURE, mo):
    mo.md(f"""
    ## 9. 실제 규모로 실행하려면

    ```bash
    cd projects/minimind/trainer
    # ../out/full_sft_768_moe.pth(교사)와 ../out/full_sft_768.pth(학생)가 필요합니다.
    python train_distillation.py
    python train_distillation.py --alpha 0.5 --temperature 1.5 --data_path ../dataset/sft_t2t_mini.jsonl
    torchrun --nproc_per_node 2 train_distillation.py     # GPU 여러 개
    ```

    | 인자 | 기본값 | 이 노트북 |
    | --- | --- | --- |
    | `--from_teacher_weight` / `--teacher_use_moe` | `full_sft` / 1 | `caspar/MiniMind-MoE-198M-A64M-Mixed-SFT-v1.0` (MoE) |
    | `--from_student_weight` / `--student_use_moe` | `full_sft` / 0 | `caspar/MiniMind-Dense-64M-Mixed-SFT-v1.0` (Dense) |
    | `--alpha` | 0.5 | 0.5 (실험 B), 1.0·0.0과 비교 |
    | `--temperature` | 1.5 | {KD_TEMPERATURE} |
    | `--learning_rate` | 5e-6 | {KD_LR} (80 step에서 변화를 보려고 10배) |
    | `--batch_size` × `--accumulation_steps` | 32 × 1 | 8 × 1 |
    | `--epochs` | 6 | 5 |
    | `--max_seq_len` | 340 | 340 |
    | `--data_path` | `sft_t2t_mini.jsonl` | GSM8K test 128문제 |

    원본은 `--teacher_hidden_size`, `--student_hidden_size`로 **크기가 다른** 교사·학생도 지원합니다(예: 768 → 512).
    두 모델의 어휘가 같아야 하고(같은 토크나이저), 다르면 `teacher_logits[..., :vocab_size_student]`가 의미 없는 비교가 됩니다.

    ## 10. 정리

    | 항목 | 핵심 |
    | --- | --- |
    | soft label | 교사의 확률 분포 전체. 정답 외 후보가 얼마나 그럴듯한지(dark knowledge)를 담음 |
    | 손실 | $\\alpha\\,CE+(1-\\alpha)T^2D_{{KL}}(p_t\\Vert p_s)$, 정답 마스크 위치만 |
    | `F.kl_div` | input = 학생 log 확률, target = 교사 확률 → forward KL. `batchmean` = 토큰 평균 |
    | 온도 | T↑ → 분포가 평평해짐. T²를 곱해 기울기 크기를 유지 |
    | 효과 1 | 학생이 교사를 닮아 감 (KL↓, 1등 일치↑). 순수 증류는 교사 분포 모사가 목적이며 일반적인 성능 상한을 뜻하지 않음 (alpha=0) |
    | 효과 2 | 작은 데이터에서 CE만 쓰면 과적합 → 교사 분포가 규제 역할 (alpha=0.5가 가장 좋음) |
    | 비용 | 매 step 교사 순전파 + 교사 메모리. 오프라인 저장은 top-k나 생성 답변으로 줄임 |

    ## 11. 직접 해 볼 과제

    각 과제를 하기 전에 **결과를 먼저 예측**해 적고, 바꿔 실행한 뒤 비교하세요.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({
        "과제 1. 온도를 크게 (6-3절)": mo.md("""
    `alpha=0`, 온도 `4.0`으로 학습하고 실험 C(온도 1.5)와 KL(교사‖학생, T=1에서 잼)을 비교하세요.

    *해설:* 온도가 높으면 학생은 **평평하게 만든** 교사 분포를 맞춥니다. 1등 후보의 확률이 얼마나 높은지는 덜 배우므로,
    T=1에서 잰 KL이 오히려 **학습 전보다 커지고** 정답 CE도 나빠집니다(검증 실행: KL 0.36 → 0.44, CE 1.24 → 1.39).
    온도는 "무엇을 맞출지"를 바꾸는 선택이고, 생성은 보통 T=1에서 하므로 너무 큰 온도는 손해입니다.
    """),
        "과제 2. 원본 학습률 5e-6 (6-3절)": mo.md("""
    `alpha=1.0`과 `alpha=0.5`를 학습률 `5e-6`으로 각각 학습해 보세요. A의 과적합이 보일까요? B가 여전히 더 좋을까요?

    *해설:* 학습률이 작으면 5 epochs 동안 CE만 써도 과적합이 나타나지 않고, 평가 CE가 1.05 근처까지 꾸준히 내려가
    오히려 같은 학습률의 alpha=0.5(약 1.10)보다 낮습니다. 하지만 KL은 0.36 → 0.53으로 커져 교사 분포에서 멀어지고,
    alpha=0.5는 KL을 0.32로 지킵니다. **증류의 규제 효과는 과적합이 일어날 만큼 학습할 때 CE 이득으로 드러나고,
    그렇지 않을 때는 "원래 분포를 지키는" 효과로 남습니다.**
    """),
        "과제 3. reverse KL로 바꾸기 (4절)": mo.md("""
    `distillation_loss`를 `KL(학생 ‖ 교사)`로 바꾸려면 `F.kl_div`의 두 인자를 어떻게 바꿔야 할까요?
    (힌트: `F.kl_div(교사_log확률, 학생_확률)` 꼴. 학생 쪽에도 기울기가 흘러야 합니다.)

    *해설:* forward KL은 교사가 확률을 준 **모든** 토큰을 학생이 덮도록(mode-covering), reverse KL은 학생이
    교사의 **큰 봉우리 하나**에 집중하도록(mode-seeking) 만듭니다. 작은 학생이 큰 교사를 다 따라 할 수 없을 때
    reverse KL이 생성 품질에 유리하다는 연구(MiniLLM 등)가 있습니다.
    """),
        "과제 4. 정답 마스크를 지우면 (6-1절)": mo.md("""
    `distillation_loss`에 `[loss_mask_flat == 1]` 대신 패딩을 뺀 모든 위치(질문 포함)를 넣으면 어떻게 될까요?

    *해설:* 질문 위치에서도 교사의 분포를 배우므로 학생은 **사용자 질문을 쓰는 법**까지 교사를 닮게 됩니다.
    SFT 노트북 실험 C(모든 토큰 정답)와 같은 문제입니다. 원본이 증류에도 정답 마스크를 쓰는 이유입니다.
    """),
        "과제 5. 교사와 학생을 바꾸면 (1절)": mo.md("""
    `student_base`와 `teacher`의 역할을 바꿔 Dense 64M을 교사로, MoE 198M을 학생으로 증류하면 어떻게 될까요?

    *해설:* 증류는 "큰 → 작은"만이 아니라 "다른 모델의 분포 → 내 모델"로도 동작합니다. 다만 학생이 MoE이면
    `ce_loss`에 `aux_loss`(전문가 부하 균형 손실)가 더해지고, 교사가 더 약하므로 alpha=0이면 MoE 학생의 CE가 나빠질 수 있습니다.
    """),
    })
    return


if __name__ == "__main__":
    app.run()
