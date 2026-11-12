import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import copy
    import json
    import random
    import sys
    from datetime import datetime
    from pathlib import Path
    from uuid import uuid4
    from zoneinfo import ZoneInfo

    import datasets  # 원본 trainer의 Windows DLL 우회 순서 유지
    import marimo as mo
    import matplotlib.pyplot as plt
    import torch
    import torch.nn.functional as F
    from tokenizers import Tokenizer, models, pre_tokenizers
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer, PreTrainedTokenizerFast

    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root / "projects/minimind"))
    from dataset.lm_dataset import SFTDataset, PretrainDataset, post_processing_chat
    from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
    from trainer.trainer_utils import get_lr

    run_id = datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y%m%d-%H%M%S") + f"-{uuid4().hex[:6]}-sft"
    output_dir = repo_root / "outputs" / "notebooks" / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    datasets.config.HF_DATASETS_CACHE = output_dir / "dataset-cache"
    datasets.disable_progress_bars()
    torch.set_num_threads(2)
    return (
        AutoTokenizer,
        DataLoader,
        F,
        MiniMindConfig,
        MiniMindForCausalLM,
        PreTrainedTokenizerFast,
        PretrainDataset,
        SFTDataset,
        Tokenizer,
        copy,
        get_lr,
        json,
        mo,
        models,
        output_dir,
        plt,
        post_processing_chat,
        pre_tokenizers,
        random,
        repo_root,
        torch,
        uuid4,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # MiniMind SFT: 대화에서 무엇을 정답으로 삼을까

    **SFT(지도 미세 조정):** 질문과 모범 응답으로 응답 방식을 가르칩니다.

    - 입력: 대화 전체
    - 손실: **assistant 응답 위치만**
    - 다음 토큰 예측은 사전 학습과 같습니다. 데이터 형식과 정답 위치만 다릅니다.

    **사용하는 원본:**

    - `minimind/trainer/train_full_sft.py`: 학습 순서
    - `minimind/dataset/lm_dataset.py`: **실제 SFTDataset**

    **읽는 순서:** 모델 구조 → 사전 학습 → **SFT** → LoRA.
    SFT는 "무엇을 학습하나", Full/LoRA는 "어떤 매개변수를 바꾸나"입니다.
    여기서는 전체를 학습하는 Full SFT를 씁니다.

    **이 노트북의 흐름:** 대화 JSONL → chat template → 정답 마스킹 → 손실 →
    실제 템플릿과 추론 데이터 → 학습 → 평가·생성·저장.
    정답 이동·기울기 누적·학습률은 사전 학습 노트북과 같습니다.

    **실행 환경:** GPU와 다운로드 없이 CPU로 실행합니다.
    기준 모델도 이 노트북 안에서 만듭니다.
    작은 합성 과제이며, 한국어 모델 성능 평가가 아닙니다.
    산출물: `outputs/notebooks/<실행 ID>/`
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 1. 원본 형식의 대화 JSONL

    한 줄에 `conversations` 배열을 하나 넣습니다. `role`은 화자, `content`는 발화입니다.

    - 학습 질문 8개
    - 검증 질문 4개: 단어는 같고 조합만 다릅니다. 평가에만 씁니다.
    - 여러 턴 예제: 마스킹 관찰용입니다. 학습에는 안 씁니다.

    원본은 `reasoning_content`, `tools`, `tool_calls`도 지원합니다. 여기서는 일반 대화만 봅니다.

    **예측:** 두 번 답하는 대화에서 마지막 응답만 정답일까요?
    """)
    return


@app.cell
def _(json, output_dir):
    animals = [("고양이", "야옹"), ("강아지", "멍멍"), ("소", "음메"), ("오리", "꽥꽥")]

    def conversation(question, answer):
        # system을 명시해 원본 pre_processing_chat의 20% 무작위 추가를 피합니다.
        return [
            {"role": "system", "content": "소리 도우미"},
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ]

    train_records = [
        {"conversations": conversation(question, sound)}
        for animal, sound in animals
        for question in [f"{animal} 소리 ?", f"소리 알려 줘 {animal} ?"]
    ]
    validation_records = [
        {"conversations": conversation(f"{animal} 소리 알려 줘 ?", sound)}
        for animal, sound in animals
    ]
    multiturn = conversation("고양이 소리 ?", "야옹") + [
        {"role": "user", "content": "강아지 소리 ?"},
        {"role": "assistant", "content": "멍멍"},
    ]
    train_path = output_dir / "train.jsonl"
    validation_path = output_dir / "validation.jsonl"
    inspection_path = output_dir / "inspection.jsonl"
    for _path, _records in [
        (train_path, train_records), (validation_path, validation_records),
        (inspection_path, [{"conversations": multiturn}]),
    ]:
        _path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in _records), encoding="utf-8")
    return (
        animals,
        inspection_path,
        multiturn,
        train_path,
        train_records,
        validation_path,
        validation_records,
    )


@app.cell(hide_code=True)
def _(json, mo, multiturn, train_records, validation_records):
    # ▸ 결과 표시 — 실행하면 표·그림이 나타납니다 (marimo 표시 코드)
    mo.vstack([
        mo.ui.table([
            {"분할": split, "질문": r["conversations"][1]["content"], "모범 응답": r["conversations"][2]["content"]}
            for split, records in [("학습", train_records), ("검증", validation_records)] for r in records
        ], selection=None),
        mo.md(f"여러 턴 JSON 예제:\n```json\n{json.dumps({'conversations': multiturn}, ensure_ascii=False, indent=2)}\n```"),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. chat template: 화자와 발화 경계 표시

    학습 데이터 단어로 **공백 단위 작은 토크나이저**를 만듭니다.
    형식: `[BOS]user\n질문[EOS]\n[BOS]assistant\n응답[EOS]\n`.

    - 학습: 대화 전체를 넣습니다. `add_generation_prompt=False`.
    - 생성: 정답을 빼고 assistant 헤더까지 만듭니다. `add_generation_prompt=True`.

    **경계 찾기:** SFTDataset은 두 패턴을 찾습니다.

    - 응답 시작: `BOS + assistant\n`
    - 응답 끝: `EOS + \n`

    **토크나이저와 template의 경계 표현이 맞아야 합니다.**
    이 작은 사전에서는 줄바꿈이 토큰이 아니라서, 끝 패턴은 `[EOS]` 하나입니다.
    실제 MiniMind 사전과는 ID·길이가 다릅니다.
    """)
    return


@app.cell
def _(
    PreTrainedTokenizerFast,
    Tokenizer,
    models,
    output_dir,
    pre_tokenizers,
    train_records,
):
    vocabulary = ["[PAD]", "[BOS]", "[EOS]", "[UNK]", "system", "user", "assistant"] + sorted({
        word for record in train_records for message in record["conversations"] for word in message["content"].split()
    })
    _backend = Tokenizer(models.WordLevel({w: i for i, w in enumerate(vocabulary)}, unk_token="[UNK]"))
    _backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=_backend, bos_token="[BOS]", eos_token="[EOS]",
        pad_token="[PAD]", unk_token="[UNK]", clean_up_tokenization_spaces=False,
    )
    tokenizer.chat_template = (
        "{% for message in messages %}"
        "{{ bos_token + message['role'] + '\n' + message['content'] + eos_token + '\n' }}"
        "{% endfor %}"
        "{% if add_generation_prompt %}{{ bos_token + 'assistant\n' }}{% endif %}"
    )
    tokenizer.save_pretrained(output_dir / "tokenizer")
    return tokenizer, vocabulary


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 3. 실제 SFTDataset의 정답 마스킹

    `pre_processing_chat → create_chat_prompt → post_processing_chat → tokenize → 절단·패딩 → generate_labels`
    순서로 처리합니다.

    원본에는 무작위 처리가 있습니다.

    - system이 없으면 20% 확률로 넣습니다.
    - 빈 think 태그도 확률적으로 지웁니다.

    이 예제는 system을 직접 넣고 think를 안 써서 결과가 항상 같습니다.
    """)
    return


@app.cell
def _(
    DataLoader,
    SFTDataset,
    inspection_path,
    tokenizer,
    train_path,
    validation_path,
):
    train_dataset = SFTDataset(str(train_path), tokenizer, max_length=32)
    validation_dataset = SFTDataset(str(validation_path), tokenizer, max_length=32)
    inspection_dataset = SFTDataset(str(inspection_path), tokenizer, max_length=40)
    inspection_ids, inspection_labels = inspection_dataset[0]
    input_ids, labels = next(iter(DataLoader(train_dataset, batch_size=2, num_workers=0)))
    assert input_ids.shape == labels.shape == (2, 32)
    assert (inspection_labels[inspection_ids == tokenizer.pad_token_id] == -100).all()
    assert tokenizer.decode(inspection_labels[inspection_labels != -100].tolist()) == "야옹 [EOS] 멍멍 [EOS]"
    assert tokenizer.unk_token_id not in input_ids
    return (
        input_ids,
        inspection_dataset,
        inspection_ids,
        inspection_labels,
        labels,
        train_dataset,
        validation_dataset,
    )


@app.cell(hide_code=True)
def _(
    input_ids,
    inspection_dataset,
    inspection_ids,
    inspection_labels,
    mo,
    multiturn,
    tokenizer,
):
    # ▸ 결과 표시 — 실행하면 표·그림이 나타납니다 (marimo 표시 코드)
    mo.vstack([
        mo.md(f"""
        **해설:**

        - 두 assistant 응답과 각 종료 토큰이 모두 정답입니다.
        - system, user, assistant 헤더, 뒤쪽 PAD는 `-100`입니다.
        - `-100`은 토큰 ID가 아닙니다. **"손실에서 빼라"는 표시**입니다.
        - 입력에는 질문과 응답이 모두 있습니다. 질문은 응답을 위한 문맥입니다.

        배치 shape `{tuple(input_ids.shape)}` = [B,T], dtype `{input_ids.dtype}`.
        여러 턴 예제의 template:
        """),
        # 여러 줄 template은 들여쓰기 없이 따로 감싸야 마크다운 들여쓰기 제거가 깨지지 않습니다.
        mo.md("```text\n" + inspection_dataset.create_chat_prompt(multiturn) + "\n```"),
        mo.ui.table([
            {"위치": t, "입력 토큰": tokenizer.convert_ids_to_tokens(int(x)),
             "input ID": int(x), "label": int(y), "손실 포함": int(y) != -100}
            for t, (x, y) in enumerate(zip(inspection_ids, inspection_labels))
        ], selection=None),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 4. 정답 이동은 사전 학습과 같다 — 달라진 것은 정답 위치뿐

    **예측:** `야옹`을 예측하는 logits는 `야옹` 입력 위치일까요, 그 앞 위치일까요?
    한 칸 이동(`logits[:, :-1]` ↔ `labels[:, 1:]`)은 사전 학습과 같습니다.
    **assistant 헤더의 마지막 토큰이 첫 응답을 예측합니다.**
    """)
    return


@app.cell
def _(
    F,
    MiniMindConfig,
    MiniMindForCausalLM,
    input_ids,
    labels,
    tokenizer,
    torch,
    vocabulary,
):
    torch.manual_seed(42)
    config = MiniMindConfig(
        hidden_size=32, num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=2, intermediate_size=64, vocab_size=len(vocabulary),
        max_position_embeddings=64, flash_attn=False, use_moe=False,
        bos_token_id=tokenizer.bos_token_id, eos_token_id=tokenizer.eos_token_id,
    )
    _model = MiniMindForCausalLM(config).eval()
    with torch.no_grad():
        initial_result = _model(input_ids, labels=labels)
        manual_loss = F.cross_entropy(
            initial_result.logits[:, :-1].reshape(-1, len(vocabulary)), labels[:, 1:].reshape(-1), ignore_index=-100,
        )
    assert torch.allclose(initial_result.loss, manual_loss)
    return config, initial_result, manual_loss


@app.cell(hide_code=True)
def _(initial_result, input_ids, labels, manual_loss, mo, tokenizer):
    # ▸ 결과 표시 — 실행하면 표·그림이 나타납니다 (marimo 표시 코드)
    mo.vstack([
        mo.md(f"""
        원본 손실 `{initial_result.loss.item():.6f}`와 직접 계산 `{manual_loss.item():.6f}`가 같습니다.

        질문 위치에는 손실이 없습니다. 하지만 응답이 질문을 참고합니다.
        그래서 질문 임베딩에도 gradient가 갈 수 있습니다.
        """),
        mo.ui.table([
            {"logits 위치": t - 1, "여기까지 읽음(마지막 토큰)": tokenizer.convert_ids_to_tokens(int(input_ids[0, t - 1])),
             "정답 위치": t, "예측할 토큰": tokenizer.convert_ids_to_tokens(int(labels[0, t]))}
            for t in range(1, labels.shape[1]) if labels[0, t] != -100
        ], selection=None),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 5. 길이를 줄이면 어떤 정답을 잃을까?

    원본은 **template을 만든 뒤 max_length로 자릅니다.**

    - 응답 전에서 잘리면: 정답이 0개입니다. 손실이 NaN이 될 수 있습니다.
    - 응답 중간에서 잘리면: 일부는 학습하지만 종료 토큰은 못 배웁니다.

    학습 전에 `labels[:, 1:] != -100`인 토큰 수를 확인하세요.

    **주의:** 종료 패턴을 못 찾으면 `generate_labels`는 끝까지 정답으로 둡니다.
    **그래서 끝나지 않은 응답 뒤의 PAD도 정답이 될 수 있습니다.**
    완성된 대화, 잘린 대화, 원래 잘못된 대화를 구분하세요.
    아래는 관찰용 예제입니다. 학습에는 안 씁니다.
    """)
    return


@app.cell
def _(
    copy,
    inspection_dataset,
    inspection_ids,
    post_processing_chat,
    tokenizer,
):
    _full = inspection_ids.tolist()
    _start = next(i for i in range(len(_full)) if _full[i:i + len(inspection_dataset.bos_id)] == inspection_dataset.bos_id)
    _cut = _start + len(inspection_dataset.bos_id)
    # 원본 __getitem__처럼 배열 길이와 max_length를 일치시켜 관찰합니다.
    _probe = copy.copy(inspection_dataset)
    _probe.max_length = _cut
    short_labels = _probe.generate_labels(_full[:_cut])
    _broken = _full[:_cut + 1] + [tokenizer.pad_token_id] * 3
    _probe.max_length = len(_broken)
    broken_labels = _probe.generate_labels(_broken)
    assert all(y == -100 for y in short_labels)
    assert broken_labels[-1] == tokenizer.pad_token_id
    assert post_processing_chat("<think>\n\n</think>\n\n야옹", remove_empty_think=True) == "야옹"
    return broken_labels, short_labels


@app.cell(hide_code=True)
def _(broken_labels, mo, short_labels):
    # ▸ 결과 표시 — 실행하면 표·그림이 나타납니다 (marimo 표시 코드)
    mo.vstack([
        mo.ui.table([
            {"입력": "assistant 헤더에서 절단", "정답 수": sum(y != -100 for y in short_labels)},
            {"입력": "종료 없는 응답 + PAD 3개", "정답 수": sum(y != -100 for y in broken_labels)},
        ], selection=None),
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 5-1. 실제 MiniMind 템플릿과 추론(reasoning) 데이터

    지금까지는 교육용 template(`[BOS]role\n…[EOS]`)을 썼습니다.
    실제 MiniMind는 다릅니다.

    - 형식: `<|im_start|>role\n…<|im_end|>`
    - **모든 assistant 응답 앞에 `<think>…</think>` 블록**이 붙습니다.
    - `reasoning_content`가 있으면 그 내용이 think 블록에 들어갑니다.

    ```text
    <|im_start|>assistant\n<think>\n{reasoning_content}\n</think>\n\n{content}<|im_end|>\n
    ```

    **예측:** think 블록 안의 추론도 손실에 들어갈까요?

    **들어갑니다.** `<|im_start|>assistant\n` **헤더 다음부터** `<|im_end|>\n`까지 정답입니다.
    헤더는 문맥이며 `label=-100`입니다. 뒤의 think 블록·최종 답·종료 경계만 손실에 들어갑니다.
    그래서 **추론 과정과 최종 답을 함께 배웁니다.** SFT로 추론 모델을 만드는 방법입니다.

    **추론이 없을 때:** 빈 블록 `<think>\n\n</think>\n\n`이 들어갑니다.
    `post_processing_chat`은 이 빈 블록을 **80% 확률로 지웁니다.**
    그래서 "바로 답하기"와 "빈 think 뒤 답하기"가 섞여 학습됩니다.
    추론 형식에 보상을 주는 방법은 `minimind_grpo.py`에서 다룹니다.

    아래는 실제 tokenizer(6400 어휘, byte-level BPE)로 데이터만 관찰합니다.
    작은 모델과 ID 체계가 달라서 학습에는 쓰지 않습니다.
    """)
    return


@app.cell
def _(
    AutoTokenizer,
    SFTDataset,
    json,
    output_dir,
    post_processing_chat,
    random,
    repo_root,
):
    real_tokenizer = AutoTokenizer.from_pretrained(repo_root / "projects/minimind/model", local_files_only=True)
    _messages = [
        {"role": "system", "content": "수학 도우미"},
        {"role": "user", "content": "2 + 3은?"},
    ]
    _reasoning_path = output_dir / "reasoning_inspection.jsonl"
    _reasoning_path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in [
        {"conversations": _messages + [{"role": "assistant", "content": "5입니다.", "reasoning_content": "2에 3을 더하면 5다."}]},
        {"conversations": _messages + [{"role": "assistant", "content": "5입니다."}]},
    ]), encoding="utf-8")
    real_dataset = SFTDataset(str(_reasoning_path), real_tokenizer, max_length=96)
    _state = random.getstate()
    try:
        random.seed(0)  # 빈 think 제거 여부를 고정하기 위한 seed (추론 있는 샘플은 영향 없음)
        _ids, real_labels = real_dataset[0]
    finally:
        random.setstate(_state)
    real_active = _ids != real_tokenizer.pad_token_id
    real_supervised = real_tokenizer.decode(_ids[real_labels != -100].tolist())
    real_context = real_tokenizer.decode(_ids[real_active & (real_labels == -100)].tolist())
    assert real_supervised.startswith("<think>") and "2에 3을 더하면 5다." in real_supervised
    assert real_supervised.rstrip().endswith("5입니다.<|im_end|>")
    empty_think_prompt = real_dataset.create_chat_prompt(real_dataset.samples[1]["conversations"])
    assert "<think>\n\n</think>\n\n" in empty_think_prompt
    assert "<think>" not in post_processing_chat(empty_think_prompt, remove_empty_think=True)
    return (
        empty_think_prompt,
        real_active,
        real_context,
        real_dataset,
        real_labels,
        real_supervised,
    )


@app.cell(hide_code=True)
def _(
    empty_think_prompt,
    mo,
    post_processing_chat,
    real_active,
    real_context,
    real_dataset,
    real_labels,
    real_supervised,
):
    # ▸ 결과 표시 — 실행하면 표·그림이 나타납니다 (marimo 표시 코드)
    mo.vstack([
        mo.md(f"`reasoning_content`가 있는 대화의 template:\n```text\n{real_dataset.create_chat_prompt(real_dataset.samples[0]['conversations'])}```"),
        mo.ui.table([
            {"구간": "문맥 (label=-100)", "토큰 수": int((real_active & (real_labels == -100)).sum()), "디코딩": repr(real_context)},
            {"구간": "정답 (손실 포함)", "토큰 수": int((real_labels != -100).sum()), "디코딩": repr(real_supervised)},
        ], selection=None),
        mo.ui.table([
            {"추론 없는 대화": "template 원문", "assistant 부분": repr(empty_think_prompt.split("assistant\n")[-1])},
            {"추론 없는 대화": "빈 think 제거(80%)", "assistant 부분": repr(post_processing_chat(empty_think_prompt, remove_empty_think=True).split("assistant\n")[-1])},
        ], selection=None),
        mo.md('**관찰:** think 블록부터 종료 토큰까지가 정답 구간입니다. 잘못된 추론도 그대로 배울 수 있습니다. 실제 tokenizer에서 `assistant\\n` 경계는 5토큰, 종료 경계(`<|im_end|>`+줄바꿈)는 2토큰입니다.'),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 6. SFT를 시작할 기준 모델 준비
    """)
    return


@app.cell
def _(
    DataLoader,
    MiniMindForCausalLM,
    PretrainDataset,
    animals,
    config,
    json,
    output_dir,
    tokenizer,
    torch,
):
    _path = output_dir / "pretrain.jsonl"
    _path.write_text("".join(json.dumps({"text": f"{a} 소리 {s}"}, ensure_ascii=False) + "\n" for a, s in animals), encoding="utf-8")
    _dataset = PretrainDataset(str(_path), tokenizer, max_length=8)
    _x, _y = next(iter(DataLoader(_dataset, batch_size=4)))
    torch.manual_seed(42)
    baseline_model = MiniMindForCausalLM(config)
    _optimizer = torch.optim.AdamW(baseline_model.parameters(), lr=0.008)
    preparation_losses = []
    for _step in range(100):
        baseline_model.train()
        _optimizer.zero_grad(set_to_none=True)
        _loss = baseline_model(_x, labels=_y).loss
        _loss.backward()
        torch.nn.utils.clip_grad_norm_(baseline_model.parameters(), 1.0)
        _optimizer.step()
        preparation_losses.append(_loss.item())
    baseline_model.eval()
    torch.save(baseline_model.state_dict(), output_dir / "pretrain_32.pth")
    return baseline_model, preparation_losses


@app.cell(hide_code=True)
def _(mo, preparation_losses):
    # ▸ 결과 설명 — 실행하면 계산 값이 들어간 설명이 나타납니다 (marimo 표시 코드)
    mo.md(f"""
    원본 trainer는 기본으로 사전 학습 가중치에서 시작합니다(`--from_weight pretrain`).
    여기서는 동물 소리 문장 4개로 작은 모델을 100 step 학습해 준비합니다.
    준비 손실: `{preparation_losses[0]:.4f} → {preparation_losses[-1]:.4f}`.

    - 이 모델을 복사해 **새 optimizer**로 SFT를 시작합니다.
    - 이어받는 것은 가중치뿐입니다. optimizer 상태는 새로 만듭니다.

    실제 규모의 사전 학습은 아닙니다. 동물·소리 짝은 준비 단계에서도 봅니다.
    검증에서는 **아는 단어로 된 새 질문 조합**에 답하는지 봅니다.
    """)
    return


@app.cell
def _(DataLoader, tokenizer, torch):
    def evaluate(model, dataset):
        model.eval()
        total_loss, tokens = 0.0, 0
        for ids, targets in DataLoader(dataset, batch_size=4, shuffle=False):
            count = int((targets[:, 1:] != -100).sum())
            if count == 0:
                raise ValueError("유효 응답 정답이 없습니다. max_length와 template을 확인하세요.")
            total_loss += model(ids, labels=targets).loss.item() * count
            tokens += count
        return total_loss / tokens

    @torch.inference_mode()
    def respond(model, messages):
        model.eval()
        # 호출자는 정답 assistant 발화를 제거한 messages를 전달합니다.
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        ids = torch.tensor([tokenizer(prompt, add_special_tokens=False).input_ids])
        generated = []
        stopped = False
        for _ in range(5):
            next_id = int(model(ids).logits[0, -1].argmax())
            if next_id == tokenizer.eos_token_id:
                stopped = True
                break
            generated.append(next_id)
            ids = torch.cat([ids, torch.tensor([[next_id]])], dim=1)
        return tokenizer.decode(generated, skip_special_tokens=False), stopped

    return evaluate, respond


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    **평가 기준:** 두 가지를 따로 봅니다.

    - teacher forcing으로 잰 응답 토큰 평균 CE
    - 실제 greedy 생성의 응답 일치율, EOS 종료 여부

    CE가 낮아도 생성에서는 오류가 쌓일 수 있습니다.
    검증 손실은 유효 토큰 수로 가중 평균합니다. 학습이나 설정 선택에는 쓰지 않습니다.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    # ▸ 실습 설정 위젯 — 실행하면 슬라이더·선택 상자가 나타납니다 (marimo 표시 코드)
    sft_epochs = mo.ui.slider(20, 100, step=20, value=60, label="SFT epoch")
    sft_accumulation = mo.ui.dropdown([1, 2, 3], value=2, label="기울기 누적 수")
    mo.vstack([
        mo.md("""
        ## 7. 원본 trainer의 핵심 학습 루프

        **예측:** microbatch 4개에서 누적 수가 2라면 optimizer는 몇 번 갱신할까요?
        루프는 사전 학습 노트북 5~6절과 같습니다:
        `(loss + aux)/누적 수 → backward → clip → step`. 학습률은 microstep마다 `get_lr`.

        - Full SFT라서 모든 매개변수를 학습합니다.
        - 작은 모델이라 LR=0.003을 씁니다. 원본 기본값은 1e-5입니다.
        - UI를 바꾸면 기준 모델의 새 복사본에서 다시 시작합니다.
        """),
        mo.hstack([sft_epochs, sft_accumulation]),
    ])
    return sft_accumulation, sft_epochs


@app.cell
def _(
    DataLoader,
    baseline_model,
    copy,
    evaluate,
    get_lr,
    json,
    output_dir,
    random,
    sft_accumulation,
    sft_epochs,
    torch,
    train_dataset,
    uuid4,
    validation_dataset,
):
    random.seed(42)
    torch.manual_seed(42)
    sft_model = copy.deepcopy(baseline_model)
    optimizer = torch.optim.AdamW(sft_model.parameters(), lr=0.003)
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=0,
                         generator=torch.Generator().manual_seed(42))
    _epochs, _accum = sft_epochs.value, sft_accumulation.value
    before_metrics = {"train_ce": evaluate(baseline_model, train_dataset), "validation_ce": evaluate(baseline_model, validation_dataset)}
    history, step_history = [], []
    optimizer_updates = 0
    optimizer.zero_grad(set_to_none=True)
    for _epoch in range(_epochs):
        sft_model.train()
        for _step, (_ids, _labels) in enumerate(train_loader, start=1):
            assert (_labels[:, 1:] != -100).any(dim=1).all(), "응답이 절단되었습니다"
            _lr = get_lr(_epoch * len(train_loader) + _step, _epochs * len(train_loader), 0.003)
            for _group in optimizer.param_groups:
                _group["lr"] = _lr
            _res = sft_model(_ids, labels=_labels)
            _loss = (_res.loss + _res.aux_loss) / _accum
            _loss.backward()
            _update = _step % _accum == 0 or _step == len(train_loader)
            _grad_norm = None
            if _update:
                _grad_norm = float(torch.nn.utils.clip_grad_norm_(sft_model.parameters(), 1.0))
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_updates += 1
            step_history.append({"epoch": _epoch + 1, "microstep": _step, "lr": _lr,
                                 "ce": _res.loss.item(), "aux": float(_res.aux_loss),
                                 "optimizer_update": _update, "grad_norm_before_clip": _grad_norm})
        if (_epoch + 1) % 10 == 0 or _epoch == 0:
            history.append({"epoch": _epoch + 1, "train_ce": evaluate(sft_model, train_dataset),
                            "validation_ce": evaluate(sft_model, validation_dataset)})
    after_metrics = {"train_ce": evaluate(sft_model, train_dataset), "validation_ce": evaluate(sft_model, validation_dataset)}
    assert all(p.requires_grad for p in sft_model.parameters())
    changed_tensors = sum(not torch.equal(p, dict(baseline_model.named_parameters())[n]) for n, p in sft_model.named_parameters())
    training_dir = output_dir / f"training-{uuid4().hex[:6]}"
    training_dir.mkdir()
    training_settings = {"epochs": _epochs, "accumulation_steps": _accum, "batch_size": 2,
                         "learning_rate": 0.003, "seed": 42, "pretrain_steps": 100}
    (training_dir / "training.json").write_text(json.dumps({"settings": training_settings, "before": before_metrics,
        "after": after_metrics, "history": history, "microsteps": step_history}, indent=2), encoding="utf-8")
    return (
        after_metrics,
        before_metrics,
        changed_tensors,
        history,
        optimizer_updates,
        sft_model,
        step_history,
        train_loader,
        training_dir,
        training_settings,
    )


@app.cell(hide_code=True)
def _(
    after_metrics,
    before_metrics,
    changed_tensors,
    mo,
    optimizer_updates,
    sft_model,
    step_history,
    train_loader,
):
    # ▸ 결과 표시 — 실행하면 표·그림이 나타납니다 (marimo 표시 코드)
    mo.vstack([
        mo.md(f"""
        1 epoch에 microbatch `{len(train_loader)}`개, 총 optimizer 갱신 `{optimizer_updates}`회입니다.
        학습 대상 `{sum(p.numel() for p in sft_model.parameters()):,}`개,
        값이 바뀐 매개변수 텐서는 `{changed_tensors}`개입니다.
        덜 찬 마지막 누적 그룹은 사전 학습 노트북 5절을 보세요.
        """),
        mo.ui.table(step_history[:len(train_loader)], selection=None),
        mo.ui.table([{"시점": k, **v} for k, v in [("SFT 전", before_metrics), ("SFT 후", after_metrics)]], selection=None),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 8. 손실과 실제 생성은 함께 확인하기

    **관찰:** 학습·검증 손실이 같이 줄어드나요? 생성 결과 표도 같은 경향인가요?
    """)
    return


@app.cell(hide_code=True)
def _(before_metrics, history, plt, training_dir):
    # ▸ 결과 표시 — 실행하면 표·그림이 나타납니다 (marimo 표시 코드)
    _fig, _ax = plt.subplots(figsize=(7, 3))
    _ax.plot([0] + [r["epoch"] for r in history], [before_metrics["train_ce"]] + [r["train_ce"] for r in history], label="Train")
    _ax.plot([0] + [r["epoch"] for r in history], [before_metrics["validation_ce"]] + [r["validation_ce"] for r in history], label="Validation")
    _ax.set(xlabel="SFT epoch", ylabel="Assistant token CE")
    _ax.legend()
    _fig.tight_layout()
    _fig.savefig(training_dir / "loss.png", dpi=140)
    _fig
    return


@app.cell
def _(
    baseline_model,
    json,
    respond,
    sft_model,
    train_records,
    training_dir,
    validation_records,
):
    generation_rows = []
    for _split, _records in [("학습", train_records), ("검증", validation_records)]:
        for _record in _records:
            _messages = _record["conversations"]
            for _stage, _model in [("SFT 전", baseline_model), ("SFT 후", sft_model)]:
                _answer, _stopped = respond(_model, _messages[:-1])
                generation_rows.append({"분할": _split, "시점": _stage, "질문": _messages[1]["content"],
                    "정답": _messages[-1]["content"], "생성": _answer, "EOS 종료": _stopped,
                    "응답+종료 일치": _answer == _messages[-1]["content"] and _stopped})
    generation_summary = [
        {"분할": split, "시점": stage,
         "정확한 응답+종료": sum(r["응답+종료 일치"] for r in generation_rows if r["분할"] == split and r["시점"] == stage),
         "질문 수": sum(r["분할"] == split and r["시점"] == stage for r in generation_rows)}
        for split in ["학습", "검증"] for stage in ["SFT 전", "SFT 후"]
    ]
    (training_dir / "generation.json").write_text(json.dumps({"summary": generation_summary, "rows": generation_rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    return generation_rows, generation_summary


@app.cell(hide_code=True)
def _(generation_rows, generation_summary, mo):
    # ▸ 결과 표시 — 실행하면 표·그림이 나타납니다 (marimo 표시 코드)
    mo.vstack([
        mo.md("""
        - 생성에는 `conversations[:-1]`만 넘겼습니다. **정답이 입력에 없는지** 확인하세요.
        - greedy로 최대 5토큰을 만들고 EOS에서 멈춥니다. special token도 표에 남깁니다.
        - 검증 4개는 아는 동물·소리의 새 조합입니다. 새 동물이나 자유 대화 능력은 평가하지 않습니다.
        """),
        mo.ui.table(generation_summary, selection=None),
        mo.ui.table(generation_rows, selection=None),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 9. 저장·복원과 원본 trainer로 연결
    """)
    return


@app.cell
def _(MiniMindForCausalLM, config, input_ids, sft_model, torch, training_dir):
    checkpoint_path = training_dir / "full_sft_32.pth"
    # 원본 배포 가중치는 half 저장. 여기서는 정확한 복원 관찰을 위해 float32를 보존합니다.
    torch.save(sft_model.state_dict(), checkpoint_path)
    config.save_pretrained(training_dir / "config")
    restored_model = MiniMindForCausalLM(config)
    restored_model.load_state_dict(torch.load(checkpoint_path, map_location="cpu", weights_only=True), strict=True)
    restored_model.eval()
    sft_model.eval()
    with torch.no_grad():
        _original = sft_model(input_ids).logits
        _restored = restored_model(input_ids).logits
    restore_difference = (_original - _restored).abs().max().item()
    assert torch.equal(_original, _restored)
    return checkpoint_path, restore_difference


@app.cell(hide_code=True)
def _(checkpoint_path, mo, restore_difference, training_settings):
    # ▸ 결과 설명 — 실행하면 계산 값이 들어간 설명이 나타납니다 (marimo 표시 코드)
    mo.md(f"""
    저장 파일: `{checkpoint_path}`. 같은 설정으로 복원한 logits의 최대 차이: `{restore_difference}`.
    설정·학습 기록·생성 결과는 가중치와 같은 `training-<고유 ID>/` 폴더에 저장했습니다.
    tokenizer는 그 상위 실행 폴더의 `tokenizer/`에 있습니다.
    설정: `{training_settings}`.

    여기서 저장한 것은 추론용 state_dict입니다.
    학습 재개에는 optimizer·scaler·epoch·step도 필요합니다.
    원본 trainer는 half 가중치와 `lm_checkpoint` 재개 파일을 따로 저장합니다.

    | 원본 파일·코드 | 이 실습에서 확인한 의미 |
    | --- | --- |
    | `SFTDataset.create_chat_prompt` | 대화를 template에 맞추어 직렬화 |
    | `SFTDataset.generate_labels` | assistant 본문과 종료 패턴에만 정답 부여 |
    | `init_model(..., from_weight='pretrain')` | 사전 학습 가중치에서 시작 |
    | `model(input_ids, labels=labels)` | 내부 shift와 마스킹된 CE 계산 |
    | `loss + aux_loss`, backward, clip, step | 전체 매개변수의 gradient와 갱신 |
    | `--from_resume 1` | 모델뿐 아니라 학습 상태까지 복원 |

    **실제 trainer로 갈 때:**

    - `minimind/trainer/`에서 실행합니다(상대 경로 사용).
    - 기본값: 768차원, 8층, max_seq_len 768, batch 16, LR 1e-5, 2 epoch.
    - 사전 학습 가중치의 차원·층 수·토크나이저, 데이터 경로를 먼저 확인하세요.
    - 이 노트북의 32차원 가중치는 768차원 모델에 못 씁니다.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## 10. 직접 설명하고 바꿔 보기

    1. **마스킹:** user가 입력에 있는데 왜 label은 -100일까요?
       여러 턴 표에서 두 응답과 두 EOS가 정답인지 확인하세요.
    2. **shift:** 첫 응답을 예측하는 입력 위치는 어디일까요?
       Dataset에서 정답을 미리 옮기면 어떤 잘못된 짝이 생길까요?
    3. **길이:** max_length를 32→8로 줄이면 유효 정답은 몇 개일까요?
       마스킹 셀에서 먼저 확인하세요. 정답이 없는 배치는 학습하지 마세요.
    4. **학습 비교:** epoch 20/60/100에서 CE, 맞힌 응답 수, EOS 종료 수를 적으세요.
       오래 학습하면 새 질문 응답도 좋아질까요?
    5. **누적:** 누적 수 2/3에서 갱신 횟수와 마지막 그룹의 gradient 크기를 비교하세요.
    6. **데이터 품질:** 고양이 응답 하나만 `멍멍`으로 바꿔 다시 학습하세요.
       검증 정답은 그대로 두고, 손실과 답변이 어떻게 바뀌는지 적으세요.

    **힌트·해설:**

    - user/system은 조건이고, assistant 응답이 정답 신호입니다.
    - EOS도 학습해야 멈추는 법을 배웁니다.
    - SFT는 주어진 응답을 따라 합니다. 틀린 정답도 배웁니다.
    - 학습 손실이 낮다고 데이터가 좋거나 일반화가 되는 건 아닙니다.

    **다음 실습:**

    - `minimind_lora.py`: 같은 목표, 바꾸는 매개변수만 줄임
    - `minimind_distillation.py`: 정답 대신 교사 분포
    - `minimind_dpo.py`: 정답 하나 대신 선호 쌍
    - 5-1절의 추론 형식은 `minimind_grpo.py`에서 보상으로 다룹니다.

    **출처:** `minimind/trainer/train_full_sft.py`, `minimind/dataset/lm_dataset.py`,
    `minimind/model/model_minimind.py`, `minimind/trainer/trainer_utils.py`.
    저작권·라이선스는 `minimind/LICENSE`를 보세요.
    """)
    return


if __name__ == "__main__":
    app.run()
