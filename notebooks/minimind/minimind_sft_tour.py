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
    `uv run marimo edit minimind_sft_tour.py`. 환경·준비물과 권장 순서는 [공통 README](README.md)를 참고하세요.
    위젯 변경 시 의존 셀이 다시 실행됩니다. 모델 크기·데이터·학습 설정 변경 전에는 해당 절의 재실행 범위를 확인하세요.
    """)
    return


@app.cell
def _():
    import copy
    import json
    import random
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
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer

    # 노트북 위치(notebooks/minimind/)에서 두 단계 위가 저장소 루트입니다.
    repo_root = Path(__file__).resolve().parents[2]
    minimind_dir = repo_root / "projects/minimind"
    sys.path.insert(0, str(minimind_dir))

    # train_full_sft.py가 쓰는 원본 클래스와 함수를 그대로 가져옵니다.
    from dataset.lm_dataset import SFTDataset, post_processing_chat, pre_processing_chat
    from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
    from trainer.trainer_utils import SkipBatchSampler, get_lr, setup_seed

    # GPU가 있으면 GPU, 없으면 CPU (train_full_sft.py의 --device 기본값과 같은 규칙)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        torch.set_num_threads(4)

    def autocast_ctx():
        """train_full_sft.py의 '3. 혼합 정밀도 설정': GPU는 bf16 autocast, CPU는 그대로 fp32"""
        return nullcontext() if device == "cpu" else torch.autocast(device_type="cuda", dtype=torch.bfloat16)

    datasets.disable_progress_bars()
    tokenizer = AutoTokenizer.from_pretrained(minimind_dir / "model")
    return (
        DataLoader,
        F,
        MiniMindConfig,
        MiniMindForCausalLM,
        Path,
        SFTDataset,
        SkipBatchSampler,
        autocast_ctx,
        copy,
        device,
        get_lr,
        hf_hub_download,
        json,
        load_file,
        mo,
        plt,
        post_processing_chat,
        pre_processing_chat,
        random,
        setup_seed,
        tempfile,
        tokenizer,
        torch,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # MiniMind SFT(지도 미세 조정) 따라가기

    사전 학습을 마친 모델은 글을 **이어 쓸** 줄은 알지만 **질문에 답할** 줄은 모릅니다.
    **SFT(Supervised Fine-Tuning)** 는 "사용자 질문 → 모델 답변" 형식의 대화를 보여 주며 **대화하는 방식**과
    **원하는 답**을 가르치는 단계입니다. MiniMind에서는 `projects/minimind/trainer/train_full_sft.py`가 이 일을 합니다.

    이 노트북은 이미 대화 SFT까지 마친 **64M MiniMind 모델**
    [`caspar/MiniMind-Dense-64M-Mixed-SFT-v1.0`](https://huggingface.co/caspar/MiniMind-Dense-64M-Mixed-SFT-v1.0)에서 출발해,
    작은 데이터로 **추가 SFT**를 합니다. 모델에게 새 정체성("Mimi")과 가상의 행성 "Zentor"에 대한 사실을 가르치고,
    무엇이 잘 배워지고 무엇이 망가지는지 봅니다.

    **준비:** 처음 실행할 때 Hugging Face에서 가중치(약 128MB)를 내려받습니다. **NVIDIA GPU가 있으면 자동으로 GPU를 씁니다**
    (GPU에서 전체 약 30초). GPU가 없으면 CPU로 실행되며 수 분이 걸립니다.

    **사전 지식:** 사전 학습 노트북(`minimind_pretrain_tour.py`)의 다음 토큰 예측 손실과 학습 루프.

    ## 0. 사전 학습과 무엇이 다른가

    `train_full_sft.py`와 `train_pretrain.py`를 나란히 비교하면 **학습 루프는 사실상 같습니다.** 다른 것은 아래뿐입니다.

    | | `train_pretrain.py` | `train_full_sft.py` |
    | --- | --- | --- |
    | 데이터셋 | `PretrainDataset` (평범한 글) | **`SFTDataset`** (대화, 채팅 템플릿) |
    | 정답(labels) | 패딩을 뺀 **모든 토큰** | **assistant 답변 토큰만** |
    | 시작 가중치 `--from_weight` | `none` (무작위) | **`pretrain`** (이 노트북: 이미 SFT된 모델) |
    | 학습률 `--learning_rate` | 5e-4 | **1e-5** (50분의 1) |
    | `--max_seq_len` | 340 | 768 |
    | `--batch_size` × `--accumulation_steps` | 32 × 8 | 16 × 1 |
    | 저장 이름 | `pretrain_768.pth` | `full_sft_768.pth` |

    "Full" SFT는 LoRA처럼 일부만이 아니라 **모든 매개변수**를 학습한다는 뜻입니다.
    1~3절은 데이터가 어떻게 만들어지는지, 4~5절은 그 데이터로 실제 모델을 학습하며 생기는 일을 봅니다.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. SFT 데이터 형식과 실습 데이터

    한 줄에 대화 하나인 jsonl입니다. `conversations`는 `role`과 `content`를 가진 메시지 목록입니다.

    ```json
    {"conversations": [
      {"role": "user", "content": "Who are you?"},
      {"role": "assistant", "content": "I am Mimi, a small language model built for the LLM Playground class."}
    ]}
    ```

    - `role`: `system`(모델의 역할 지시), `user`(사용자), `assistant`(모델 답변), `tool`(도구 결과)
    - assistant 메시지에는 생각 과정 `reasoning_content`, 도구 호출 `tool_calls`가 있을 수 있습니다.

    **실습 데이터 (학습 11개)**

    - **정체성 8개:** "Who are you?", "What is your name?" 등 8가지 질문 → 모두 같은 답 "I am Mimi, ..."
    - **가상의 사실 3개:** 행성 Zentor의 색, 위성 수, 발견자 (모델이 원래 알 수 없는 내용)

    **평가 (학습에 없던 질문)**

    - 정체성 질문 4개 (예: "May I know your name?") — 답에 "Mimi"가 있으면 성공
    - Zentor 질문 3개를 **다른 표현으로** (예: "Tell me the color of Zentor.") — 정답 단어가 있으면 성공
    - 일반 질문 6개 (예: "What is the capital of France?") — 원래 능력이 남아 있는지 확인
    """)
    return


@app.cell
def _(Path, json, tempfile):
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
    # 5절 "원래 답 섞기"에 쓸 일반 질문 (평가용 일반 질문과 겹치지 않음)
    replay_q = ["What is the capital of Italy?", "What color is grass?", "How many days are in a week?",
                "What is the opposite of hot?", "Who wrote Romeo and Juliet?", "What gas do plants absorb from the air?",
                "How many hours are in a day?", "What is H2O commonly called?"]

    def chat(q, a):
        return [{"role": "user", "content": q}, {"role": "assistant", "content": a}]

    # 구조 설명용 예시: 여러 턴 대화, system 메시지와 생각 과정(reasoning_content)
    example_conversations = [
        {"conversations": [
            {"role": "user", "content": "Who are you?"},
            {"role": "assistant", "content": IDENTITY},
            {"role": "user", "content": "What is Zentor?"},
            {"role": "assistant", "content": "Zentor is a blue planet with two moons."},
        ]},
        {"conversations": [
            {"role": "system", "content": "You are a helpful tutor."},
            {"role": "user", "content": "How many moons does Zentor have?"},
            {"role": "assistant", "content": "Zentor has two moons.",
             "reasoning_content": "The user asks about Zentor. I learned that it has two moons."},
        ]},
    ]

    sft_data_dir = Path(tempfile.mkdtemp(prefix="minimind_sft_data_"))

    def write_jsonl(path, rows):
        with open(path, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    sft_path = sft_data_dir / "sft.jsonl"
    examples_path = sft_data_dir / "sft_examples.jsonl"
    write_jsonl(sft_path, [{"conversations": chat(q, a)} for q, a in train_pairs])
    write_jsonl(examples_path, example_conversations)
    return (
        IDENTITY,
        chat,
        example_conversations,
        examples_path,
        fact_test,
        general_eval_q,
        identity_test_q,
        replay_q,
        sft_data_dir,
        sft_path,
        train_pairs,
        write_jsonl,
    )


@app.cell(hide_code=True)
def _(mo, sft_path, train_pairs):
    _head = "".join(open(sft_path, encoding="utf-8").readlines()[:2])
    mo.md(f"""
    학습 대화 **{len(train_pairs)}개**. SFT 파일의 처음 두 줄:

    ```json
    {_head.strip()}
    ```
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. 채팅 템플릿: 대화 → 하나의 긴 글

    모델은 결국 토큰 한 줄만 받습니다. 그래서 대화를 **정해진 표시가 붙은 글 하나**로 바꿉니다.
    이 규칙이 `projects/minimind/model/chat_template.jinja`이고, `tokenizer.apply_chat_template`가 적용합니다.

    ```text
    <|im_start|>user
    질문<|im_end|>
    <|im_start|>assistant
    <think>
    생각 과정
    </think>

    답변<|im_end|>
    ```

    - `<|im_start|>역할` ~ `<|im_end|>`가 메시지 하나입니다. `<|im_start|>`는 bos, `<|im_end|>`는 eos 토큰과 같습니다.
    - assistant 메시지에는 **항상 `<think>` 블록이 붙습니다.** `reasoning_content`가 없으면 빈 블록 `<think>\n\n</think>\n\n`입니다.
    - `SFTDataset.create_chat_prompt`는 이 함수를 `add_generation_prompt=False`로 부릅니다(답변까지 포함된 완성된 대화).
    """)
    return


@app.cell
def _(example_conversations, tokenizer, train_pairs):
    def render(conversations, **kwargs):
        return tokenizer.apply_chat_template(conversations, tokenize=False, **kwargs)

    rendered_single = render([{"role": "user", "content": train_pairs[0][0]}, {"role": "assistant", "content": train_pairs[0][1]}])
    rendered_multi = render(example_conversations[0]["conversations"])
    rendered_reasoning = render(example_conversations[1]["conversations"])

    # 추론할 때: 질문까지만 주고 "assistant 차례"를 열어 둡니다.
    _question = [{"role": "user", "content": "Who are you?"}]
    rendered_generation = render(_question, add_generation_prompt=True)
    rendered_generation_think = render(_question, add_generation_prompt=True, open_thinking=True)
    return (
        rendered_generation,
        rendered_generation_think,
        rendered_multi,
        rendered_reasoning,
        rendered_single,
    )


@app.cell(hide_code=True)
def _(
    mo,
    rendered_generation,
    rendered_generation_think,
    rendered_multi,
    rendered_reasoning,
    rendered_single,
):
    mo.vstack([
        mo.hstack([
            mo.md(f"**질문 1개 + 답 1개**\n```text\n{rendered_single}```"),
            mo.md(f"**여러 턴 대화**\n```text\n{rendered_multi}```"),
            mo.md(f"**system + 생각 과정**\n```text\n{rendered_reasoning}```"),
        ], widths="equal", align="start"),
        mo.md(r"""
    **추론할 때의 프롬프트** (`add_generation_prompt=True`): 마지막에 assistant 차례를 열어 두고 모델이 이어 씁니다.
    """),
        mo.hstack([
            mo.md(f"기본 (생각 끄기: 빈 think를 미리 채움)\n```text\n{rendered_generation}```"),
            mo.md(f"`open_thinking=True` (생각 켜기: `<think>`만 열어 둠)\n```text\n{rendered_generation_think}```"),
        ], widths="equal", align="start"),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 2-1. 무작위 가공: `pre_processing_chat`과 `post_processing_chat`

    `SFTDataset.__getitem__`은 샘플을 꺼낼 때마다 두 가지를 **무작위로** 바꿉니다(데이터 증강).

    ```python
    conversations = pre_processing_chat(sample['conversations'])   # 20% 확률로 system 메시지 추가
    prompt = self.create_chat_prompt(conversations)                # 채팅 템플릿 적용
    prompt = post_processing_chat(prompt)                          # 빈 think 블록을 80% 확률로 삭제
    ```

    - **system 추가 (20%):** 첫 메시지가 system이 아니고 도구 호출 데이터가 아니면, 미리 정해 둔 5개 중 하나를 붙입니다.
    - **빈 think 삭제 (80%):** 생각 과정이 없는 데이터에서 `<think>\n\n</think>\n\n`을 대부분 지우고 20%만 남깁니다.

    이 때문에 베이스 모델의 모델 카드는 추론할 때 프롬프트의 **빈 think 블록을 지우라고** 안내합니다(학습 데이터의 80% 형식).
    이 노트북의 생성도 그렇게 합니다. 같은 샘플을 1,000번 꺼내 실제 비율을 셉니다.
    """)
    return


@app.cell
def _(SFTDataset, random, sft_path, tokenizer):
    sft_max_len = 96  # 원본 기본값은 768. 우리 대화는 짧으므로 96이면 충분합니다.
    sft_ds = SFTDataset(str(sft_path), tokenizer, max_length=sft_max_len)

    random.seed(0)
    _texts = [tokenizer.decode(sft_ds[0][0]) for _ in range(1000)]
    augment_counts = {
        "system 추가됨": sum("<|im_start|>system" in t for t in _texts),
        "빈 think 남음": sum("<think>" in t for t in _texts),
    }
    augment_examples = {
        "system 있음": next(t for t in _texts if "<|im_start|>system" in t),
        "빈 think 있음": next(t for t in _texts if "<think>" in t),
        "둘 다 없음": next(t for t in _texts if "<think>" not in t and "system" not in t),
    }
    assert 150 < augment_counts["system 추가됨"] < 250
    assert 150 < augment_counts["빈 think 남음"] < 250
    return augment_counts, augment_examples, sft_ds, sft_max_len


@app.cell(hide_code=True)
def _(augment_counts, augment_examples, mo):
    _clean = {k: v.replace("<|endoftext|>", "").strip() for k, v in augment_examples.items()}
    mo.vstack([
        mo.md(" · ".join(f"{k}: **{v / 10:.1f}%**" for k, v in augment_counts.items()) + " (1,000번 중)"),
        mo.hstack([mo.md(f"**{k}**\n```text\n{v}\n```") for k, v in _clean.items()], widths="equal", align="start"),
        mo.md("같은 대화라도 epoch마다 조금씩 다른 글로 학습됩니다. `setup_seed`가 Python `random`도 고정하므로 재현은 됩니다."),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. SFT의 핵심: assistant 답변만 정답으로 — `generate_labels`

    사전 학습은 패딩을 뺀 **모든 토큰**을 맞히게 했습니다. SFT는 **assistant가 말한 부분만** 맞히게 합니다.
    질문, system 지시, 템플릿 표시는 입력으로만 쓰고 정답(`labels`)은 `-100`으로 둡니다.

    ```python
    self.bos_id = tokenizer('<|im_start|>assistant\n').input_ids   # 답변 시작 표시
    self.eos_id = tokenizer('<|im_end|>\n').input_ids               # 답변 끝 표시

    def generate_labels(self, input_ids):
        labels = [-100] * len(input_ids)
        # input_ids에서 bos_id 토큰 열을 찾을 때마다:
        #   그 "다음" 위치부터 eos_id 토큰 열의 "끝"까지 labels[j] = input_ids[j]
    ```

    - `<|im_start|>assistant\n` 자체는 정답이 아닙니다. 대화 형식상 항상 주어지는 부분이기 때문입니다.
    - 답변 내용, think 블록, 끝의 `<|im_end|>\n`은 정답입니다. **답을 마치고 멈추는 법**까지 배웁니다.
    - 여러 턴이면 **모든 assistant 답변**이 정답입니다.
    - 사전 학습과 마찬가지로 `labels`는 한 칸 밀지 않습니다. 모델 `forward`가 밀어서 비교합니다.

    `__getitem__` 전체 과정을 한 단계씩 따라가서 원본 결과와 같은지 확인합니다.
    """)
    return


@app.cell
def _(
    IDENTITY,
    SFTDataset,
    examples_path,
    post_processing_chat,
    pre_processing_chat,
    random,
    sft_max_len,
    tokenizer,
    torch,
):
    examples_ds = SFTDataset(str(examples_path), tokenizer, max_length=sft_max_len)

    def getitem_step_by_step(ds, index):
        """SFTDataset.__getitem__과 같은 순서로 직접 계산합니다."""
        conversations = pre_processing_chat(ds.samples[index]["conversations"])  # ① system 무작위 추가
        prompt = ds.create_chat_prompt(conversations)                             # ② 채팅 템플릿
        prompt = post_processing_chat(prompt)                                     # ③ 빈 think 무작위 삭제
        input_ids = tokenizer(prompt).input_ids[: ds.max_length]                  # ④ 토큰화·자르기
        input_ids += [tokenizer.pad_token_id] * (ds.max_length - len(input_ids))  # ⑤ 패딩
        labels = ds.generate_labels(input_ids)                                    # ⑥ 답변만 정답
        return torch.tensor(input_ids), torch.tensor(labels)

    # 같은 시드면 무작위 선택도 같으므로 원본과 정확히 같아야 합니다.
    for _i in range(len(examples_ds)):
        random.seed(_i)
        _ref = examples_ds[_i]
        random.seed(_i)
        _mine = getitem_step_by_step(examples_ds, _i)
        assert torch.equal(_ref[0], _mine[0]) and torch.equal(_ref[1], _mine[1])

    def labeled_text(labels):
        """정답 위치의 토큰만 이어 붙인 글"""
        return tokenizer.decode(labels[labels != -100])

    random.seed(1)
    multi_ids, multi_labels = examples_ds[0]  # 여러 턴
    random.seed(1)
    reason_ids, reason_labels = examples_ds[1]  # system + 생각 과정
    assert labeled_text(multi_labels) == f"{IDENTITY}<|im_end|>\nZentor is a blue planet with two moons.<|im_end|>\n"
    assert labeled_text(reason_labels).startswith("<think>\nThe user asks about Zentor.")
    return labeled_text, multi_ids, multi_labels, reason_labels


@app.cell(hide_code=True)
def _(mo, multi_ids, multi_labels, result_table, tokenizer):
    _n = int((multi_ids != tokenizer.pad_token_id).sum())
    _rows = []
    for _k in range(_n):
        _t = tokenizer.convert_ids_to_tokens(int(multi_ids[_k])).replace("Ċ", "\\n")
        _rows.append({"위치": _k, "토큰": _t, "labels": int(multi_labels[_k]),
                      "정답 여부": "✅ 정답" if int(multi_labels[_k]) != -100 else ""})
    mo.vstack([
        mo.md("**여러 턴 대화의 토큰별 정답 여부** (`Ġ`는 공백, `\\n`은 줄바꿈, 패딩은 생략)"),
        result_table(_rows, selection=None, page_size=80),
    ])
    return


@app.cell(hide_code=True)
def _(labeled_text, mo, multi_labels, reason_labels):
    mo.md(f"""
    정답 위치의 토큰만 이어 붙이면:

    | 샘플 | 정답으로 학습하는 글 |
    | --- | --- |
    | 여러 턴 대화 | `{labeled_text(multi_labels)!r}` |
    | system + 생각 과정 | `{labeled_text(reason_labels)!r}` |

    두 번의 답변이 모두 정답이고, 질문·system·`<|im_start|>assistant`는 빠졌습니다.
    생각 과정(`<think>...</think>`)도 정답에 들어가므로, 생각 데이터로 SFT하면 **생각을 쓰는 법**도 함께 배웁니다.
    직접 따라간 계산은 원본 `__getitem__`과 완전히 같았습니다.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 3-1. 잘린 답변과 학습되는 토큰의 비율

    `max_seq_len`보다 긴 대화는 뒤가 잘립니다. 답변 중간에서 잘리면 `<|im_end|>`가 없으므로
    `generate_labels`는 **`max_length`까지**를 정답으로 둡니다. 그 샘플은 끝맺는 법을 배우지 못합니다.
    원본 기본값이 사전 학습(340)보다 긴 768인 이유 중 하나입니다.
    """)
    return


@app.cell
def _(SFTDataset, examples_path, labeled_text, random, sft_ds, tokenizer):
    # 첫 답변이 시작되는 위치를 찾고, 답변 3토큰째에서 잘리도록 max_length를 정합니다.
    random.seed(1)
    _full_labels = SFTDataset(str(examples_path), tokenizer, max_length=96)[0][1]
    cut_len = int((_full_labels != -100).nonzero()[0]) + 3
    random.seed(1)  # 같은 시드 → 같은 무작위 가공 → 같은 앞부분
    _cut_ids, cut_labels = SFTDataset(str(examples_path), tokenizer, max_length=cut_len)[0]
    cut_labeled = labeled_text(cut_labels)
    assert cut_labeled and "<|im_end|>" not in cut_labeled  # 끝 표시 없이 끝남

    # 우리 SFT 데이터에서 정답 토큰이 차지하는 비율 (패딩 제외)
    random.seed(0)
    _pairs = [sft_ds[i] for i in range(len(sft_ds))]
    _real = sum(int((x != tokenizer.pad_token_id).sum()) for x, _ in _pairs)
    _target = sum(int((y != -100).sum()) for _, y in _pairs)
    target_ratio = _target / _real
    return cut_labeled, cut_len, target_ratio


@app.cell(hide_code=True)
def _(cut_labeled, cut_len, mo, target_ratio):
    mo.md(f"""
    - 여러 턴 대화를 `max_length={cut_len}`(첫 답변 3토큰째)에서 자르면 정답: `{cut_labeled!r}` →
      `<|im_end|>` 없이 끝나고, 두 번째 턴은 통째로 사라집니다.
    - 우리 SFT 데이터에서 패딩을 뺀 토큰 중 **정답은 {target_ratio:.0%}** 입니다. 나머지는 질문과 템플릿입니다.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. 학습: 실제 모델에 추가 SFT

    ### 4-1. 베이스 모델 불러오기

    `train_full_sft.py`는 `init_model(..., from_weight)`로 이전 단계 가중치를 불러와 시작합니다.
    여기서는 Hugging Face에서 **설정(`config.json`)과 가중치(`model.safetensors`)만** 버전 `v1.0`으로 고정해 받고,
    저장소의 원본 `MiniMindForCausalLM`에 넣습니다(저장소의 `trust_remote_code`용 코드는 실행하지 않음).
    이 모델은 FineWeb-Edu로 사전 학습한 뒤 UltraChat·SmolTalk·GSM8K 대화로 SFT한 **영어** 모델입니다.
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
    BASE_REVISION = "v1.0"  # 버전 고정

    base_config = MiniMindConfig(**json.load(open(hf_hub_download(BASE_REPO, "config.json", revision=BASE_REVISION))))
    base_model = MiniMindForCausalLM(base_config)
    _state = {k: v.float() for k, v in load_file(hf_hub_download(BASE_REPO, "model.safetensors", revision=BASE_REVISION)).items()}
    _missing, _unexpected = base_model.load_state_dict(_state, strict=False)
    base_model.to(device).eval()

    # lm_head는 Embedding과 같은 가중치라 파일에 따로 저장되지 않습니다.
    assert _missing == ["lm_head.weight"] and not _unexpected
    assert base_model.lm_head.weight is base_model.model.embed_tokens.weight
    return BASE_REPO, base_config, base_model


@app.cell
def _(
    F,
    IDENTITY,
    device,
    fact_test,
    general_eval_q,
    identity_test_q,
    tokenizer,
    torch,
):
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

    @torch.inference_mode()
    def continue_raw(model, text, max_new_tokens=20):
        """템플릿 없이 글을 그대로 이어 씁니다."""
        x = torch.tensor([tokenizer(text, add_special_tokens=False).input_ids], device=device)
        y = model.generate(inputs=x, max_new_tokens=max_new_tokens, do_sample=False, temperature=1.0,
                           top_p=1.0, top_k=0, eos_token_id=tokenizer.eos_token_id)
        return tokenizer.decode(y[0, x.shape[1]:], skip_special_tokens=False)

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
            "general_mimi": sum("Mimi" in a for a in general),           # 일반 질문에 정체성 답이 새어 나온 수
            "retention": answer_logp_per_token(model, base_general_pairs),  # 원래 일반 답의 토큰당 log p
        }

    assert "Mimi" in IDENTITY
    return ask, continue_raw, evaluate


@app.cell
def _(ask, base_model, evaluate, general_eval_q, replay_q):
    # 베이스 모델의 원래 일반 답: 망각을 재는 기준 (평가용과 replay용은 서로 다른 질문)
    base_general_pairs = [(q, ask(base_model, q)) for q in general_eval_q]
    replay_pairs = [(q, ask(base_model, q)) for q in replay_q]
    base_eval = evaluate(base_model, base_general_pairs)
    return base_eval, base_general_pairs, replay_pairs


@app.cell(hide_code=True)
def _(
    BASE_REPO,
    base_eval,
    device,
    fact_test,
    identity_test_q,
    mo,
    result_table,
    torch,
):
    _where = f"GPU (`{torch.cuda.get_device_name(0)}`)" if device == "cuda" else "CPU"
    mo.vstack([
        mo.md(f"`{BASE_REPO}`, 실행 장치: **{_where}**. 추가 SFT 전 베이스 모델의 답:"),
        result_table(
            [{"질문 (학습에 없는 표현)": q, "베이스 모델 답": a}
             for q, a in zip(identity_test_q + [q for q, _ in fact_test], base_eval["identity"] + base_eval["facts"])],
            selection=None,
        ),
        mo.md("베이스 모델은 Mimi라는 정체성도, 가상의 행성 Zentor도 모릅니다. Zentor에 대해서는 그럴듯한 내용을 지어냅니다."),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 4-2. 학습 루프와 세 가지 실험

    `run_training`은 `train_full_sft.py`의 "8. 학습 시작" + `train_epoch`를 옮긴 함수입니다(DDP·wandb·저장 제외).
    GPU에서는 원본처럼 bf16 `autocast`로 forward와 손실을 계산합니다.

    | 실험 | 시작 가중치 | 정답(labels) | 확인할 것 |
    | --- | --- | --- | --- |
    | **A** (원본 방식) | 베이스 모델 | assistant 답변만 | 기준 |
    | **B** | 무작위 (`--from_weight none`) | assistant 답변만 | 이전 학습(사전 학습 + SFT)의 효과 |
    | **C** | 베이스 모델 | 패딩 뺀 모든 토큰 (사전 학습식) | 정답 마스크의 효과 |

    기본값은 원본과 같은 학습률 `1e-5`, 10 epochs(대화 11개 / batch 4 → 30 step)입니다. 바꾸면 세 실험을 다시 합니다.
    """)
    return


@app.cell
def _(
    DataLoader,
    SkipBatchSampler,
    autocast_ctx,
    device,
    get_lr,
    setup_seed,
    torch,
):
    def run_training(model, train_ds, args, history=None, logs=None):
        """train_full_sft.py의 '8. 학습 시작' + train_epoch"""
        model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args["learning_rate"])
        model.train()
        for epoch in range(args["epochs"]):
            setup_seed(42 + epoch); indices = torch.randperm(len(train_ds)).tolist()
            loader = DataLoader(train_ds, batch_sampler=SkipBatchSampler(indices, args["batch_size"], 0))
            iters = len(loader)
            last_grad_norm = None
            for step, (input_ids, labels) in enumerate(loader, start=1):
                input_ids, labels = input_ids.to(device), labels.to(device)
                lr = get_lr(epoch * iters + step, args["epochs"] * iters, args["learning_rate"])
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr
                with autocast_ctx():
                    res = model(input_ids, labels=labels)
                    loss = (res.loss + res.aux_loss) / args["accumulation_steps"]
                loss.backward()
                if step % args["accumulation_steps"] == 0 or step == iters:
                    last_grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args["grad_clip"])
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                if history is not None:
                    history.append(loss.item() * args["accumulation_steps"])
                if logs is not None and step == iters:
                    logs.append(f"Epoch:[{epoch + 1}/{args['epochs']}]({step}/{iters}), "
                                f"loss: {loss.item() * args['accumulation_steps']:.4f}, lr: {lr:.8f}, "
                                f"grad_norm: {last_grad_norm.item():.3f}")
        return model.eval()

    return (run_training,)


@app.cell(hide_code=True)
def _(mo):
    sft_lr = mo.ui.dropdown(options={"1e-5": 1e-5, "5e-5": 5e-5, "2e-4": 2e-4}, value="1e-5", label="learning_rate")
    sft_epochs = mo.ui.dropdown(options={"5": 5, "10": 10, "20": 20}, value="10", label="epochs")
    mo.hstack([sft_lr, sft_epochs], justify="start", gap=2)
    return sft_epochs, sft_lr


@app.cell
def _(
    MiniMindForCausalLM,
    base_config,
    base_model,
    copy,
    run_training,
    setup_seed,
    sft_ds,
    sft_epochs,
    sft_lr,
    tokenizer,
    torch,
):
    sft_args = {"epochs": sft_epochs.value, "batch_size": 4, "accumulation_steps": 1,
                "learning_rate": sft_lr.value, "grad_clip": 1.0}

    class AllTokenLabels(torch.utils.data.Dataset):
        """실험 C: SFTDataset과 같은 입력, 정답만 '패딩 뺀 모든 토큰'으로 바꿈"""
        def __init__(self, base):
            self.base = base
        def __len__(self):
            return len(self.base)
        def __getitem__(self, i):
            input_ids, _ = self.base[i]
            return input_ids, input_ids.masked_fill(input_ids == tokenizer.pad_token_id, -100)

    def random_model():
        """--from_weight none: 같은 구조의 무작위 가중치"""
        setup_seed(42)
        return MiniMindForCausalLM(base_config)

    hist_a, hist_b, logs_a = [], [], []
    model_a = run_training(copy.deepcopy(base_model), sft_ds, sft_args, history=hist_a, logs=logs_a)
    model_b = run_training(random_model(), sft_ds, sft_args, history=hist_b)
    model_c = run_training(copy.deepcopy(base_model), AllTokenLabels(sft_ds), sft_args)
    return hist_a, hist_b, logs_a, model_a, model_b, model_c, sft_args


@app.cell
def _(base_eval, base_general_pairs, evaluate, model_a, model_b, model_c):
    sft_evals = {"베이스": base_eval, "A": evaluate(model_a, base_general_pairs),
                 "B": evaluate(model_b, base_general_pairs), "C": evaluate(model_c, base_general_pairs)}
    return (sft_evals,)


@app.cell(hide_code=True)
def _(
    fact_test,
    general_eval_q,
    hist_a,
    hist_b,
    identity_test_q,
    logs_a,
    mo,
    plt,
    result_table,
    sft_args,
    sft_evals,
):
    _fig, _ax = plt.subplots(figsize=(6, 3))
    _ax.plot(hist_a, lw=0.9, label="A: from base (SFT) model")
    _ax.plot(hist_b, lw=0.9, label="B: from scratch")
    _ax.set_yscale("log")
    _ax.set_xlabel("step")
    _ax.set_ylabel("SFT loss (answer tokens)")
    _ax.legend()
    _fig.tight_layout()
    _plot = mo.as_html(_fig)
    plt.close(_fig)

    _summary = [{"모델": k, "정체성 (새 표현 4개)": f"{v['identity_ok']}/4", "Zentor 사실 (새 표현 3개)": f"{v['fact_ok']}/3",
                 "일반 질문에 Mimi가 섞임": f"{v['general_mimi']}/6", "원래 일반 답 유지도 (토큰당 log p)": f"{v['retention']:.3f}"}
                for k, v in sft_evals.items()]
    _qs = identity_test_q + [q for q, _ in fact_test] + general_eval_q[:3]
    _answers = [{"질문": q, **{k: (v["identity"] + v["facts"] + v["general"][:3])[i] for k, v in sft_evals.items()}}
                for i, q in enumerate(_qs)]
    mo.vstack([
        mo.md(f"설정: lr={sft_args['learning_rate']}, {sft_args['epochs']} epochs"),
        _plot,
        result_table(_summary, selection=None),
        result_table(_answers, selection=None, page_size=12),
        mo.md("**실험 A 로그 (epoch 끝마다, 마지막 3줄)**\n```text\n" + "\n".join(logs_a[-3:]) + "\n```"),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **읽는 법** (기본 설정 lr=1e-5, 10 epochs. 설정을 바꾸면 결과도 바뀝니다.)

    - **A:** 학습에 없던 표현으로 물어도 "I am Mimi, ..."라고 답합니다. 베이스 모델이 이미 **질문의 뜻**을 알고 있어서,
      8가지 질문만으로 "이런 종류의 질문 → 이 답"을 일반화합니다. 일반 질문 답은 거의 그대로이고 유지도도 크게 떨어지지 않습니다.
    - **Zentor 사실은 잘 일반화되지 않습니다.** 질문 표현이 학습 문장과 비슷하면 맞히지만, 표현이 달라지면 다시 지어냅니다.
      작은 SFT 데이터로 **새 사실(지식)** 을 넣기는 **형식이나 정체성**을 가르치기보다 훨씬 어렵습니다.
    - **B (무작위에서 시작):** 같은 데이터·학습량으로는 말을 거의 못 합니다(빈 답이나 의미 없는 토큰). 손실 곡선도 훨씬 위에 있습니다.
      SFT는 사전 학습으로 얻은 언어 능력 **위에서** 동작합니다.
    - **C (모든 토큰 정답):** 정체성은 비슷하게 배우지만, 아래 4-3절처럼 질문을 지어내는 법까지 배웁니다.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 4-3. 질문까지 학습하면? (A vs C)

    사용자 차례를 열어 주고(`<|im_start|>user\n`) 첫 단어만 준 뒤 모델이 이어 쓰게 합니다.
    A는 사용자 차례를 한 번도 정답으로 본 적이 없고, C는 질문 문장까지 학습했습니다.
    """)
    return


@app.cell
def _(base_model, continue_raw, model_a, model_c):
    user_turn_prompts = ["<|im_start|>user\n", "<|im_start|>user\nWhat", "<|im_start|>user\nWho"]
    user_turn_rows = [{"입력": repr(p), "베이스": continue_raw(base_model, p), "A (답변만 정답)": continue_raw(model_a, p),
                       "C (모든 토큰 정답)": continue_raw(model_c, p)} for p in user_turn_prompts]
    return (user_turn_rows,)


@app.cell(hide_code=True)
def _(mo, result_table, user_turn_rows):
    mo.vstack([
        result_table(user_turn_rows, selection=None),
        mo.md(r"""
    C는 학습 데이터의 질문(예: `What is your name?`)을 그대로 지어내고 `<|im_end|>`로 끝맺습니다. **사용자 역할을 흉내 내는 법**을 배운 것입니다.
    실제 대화에서 모델이 혼자 질문하고 답하며 대화를 이어 가는 원인이 됩니다. A와 베이스 모델은 이런 질문을 만들지 않습니다.
    """),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. 학습률과 망각, 그리고 원래 답 섞기(replay)

    SFT 데이터가 작고 한쪽으로 치우쳐 있으면, 모든 매개변수를 큰 학습률로 바꿀 때 원래 능력이 **덮어써집니다(catastrophic forgetting).**
    원본이 SFT에 사전 학습보다 훨씬 작은 학습률(`1e-5`)을 쓰는 이유입니다.

    흔한 대응은 새 데이터에 **원래 능력을 보여 주는 데이터를 섞는 것(replay)** 입니다. 여기서는 평가와 겹치지 않는 일반 질문 8개에 대한
    **베이스 모델 자신의 답**을 SFT 데이터에 섞습니다.

    학습률 3가지 × (새 데이터만 / + replay)로 6번 학습합니다(GPU에서 약 10초).
    """)
    return


@app.cell
def _(
    SFTDataset,
    base_general_pairs,
    base_model,
    chat,
    copy,
    evaluate,
    replay_pairs,
    run_training,
    sft_data_dir,
    sft_ds,
    tokenizer,
    train_pairs,
    write_jsonl,
):
    write_jsonl(sft_data_dir / "sft_replay.jsonl", [{"conversations": chat(q, a)} for q, a in train_pairs + replay_pairs])
    replay_ds = SFTDataset(str(sft_data_dir / "sft_replay.jsonl"), tokenizer, max_length=96)

    forgetting_rows = []
    for _lr in [1e-5, 5e-5, 2e-4]:
        for _name, _ds in [("새 데이터만", sft_ds), ("+ replay", replay_ds)]:
            _args = {"epochs": 10, "batch_size": 4, "accumulation_steps": 1, "learning_rate": _lr, "grad_clip": 1.0}
            _ev = evaluate(run_training(copy.deepcopy(base_model), _ds, _args), base_general_pairs)
            forgetting_rows.append({
                "학습률": _lr, "데이터": _name,
                "정체성": f"{_ev['identity_ok']}/4", "Zentor": f"{_ev['fact_ok']}/3",
                "일반 질문에 Mimi가 섞임": f"{_ev['general_mimi']}/6",
                "원래 일반 답 유지도": f"{_ev['retention']:.3f}",
                "예: France 질문의 답": _ev["general"][0][:60],
            })
    return (forgetting_rows,)


@app.cell(hide_code=True)
def _(base_eval, forgetting_rows, mo, result_table):
    mo.vstack([
        mo.md(f"베이스 모델의 원래 일반 답 유지도: `{base_eval['retention']:.3f}` (0에 가까울수록 원래 답을 그대로 낼 가능성이 큼)"),
        result_table(forgetting_rows, selection=None),
        mo.md(r"""
    - **학습률이 클수록** 정체성은 확실히 배우지만, 일반 질문에도 "I am Mimi"가 섞이고 유지도가 크게 떨어집니다.
      `2e-4`에서는 대부분의 질문에 정체성 답을 냅니다. 새 데이터에 **과적합**하며 원래 능력을 잊은 것입니다.
    - **replay를 섞으면** 유지도 하락이 줄고, 특히 큰 학습률(`2e-4`)에서 일반 질문 오염이 크게 줄어듭니다.
      평가에 쓴 일반 질문은 replay에 없으므로 "외워서 좋아진 것"이 아니라 **원래 행동을 지키는 신호**를 준 효과입니다.
    - 하지만 replay가 완벽한 보호는 아닙니다. 큰 학습률에서는 형식은 지켜도 내용이 틀어질 수 있고(예: France의 수도를 Rome이라고 답함),
      작은 학습률에서는 차이가 거의 없거나 오히려 한두 문항이 나빠지기도 합니다. 문항이 6개뿐이라 작은 차이는 우연일 수 있습니다.
    - 실제 대규모 SFT는 다양하고 많은 데이터(`sft_t2t_mini.jsonl`)와 작은 학습률로 이 균형을 맞춥니다.
    """),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. 실제 규모로 실행하려면

    ```bash
    cd projects/minimind/trainer
    python train_full_sft.py                                   # pretrain_768.pth에서 시작, sft_t2t_mini.jsonl
    python train_full_sft.py --from_weight full_sft --data_path ../dataset/my_identity.jsonl   # 이미 SFT된 모델에 추가 SFT
    torchrun --nproc_per_node 2 train_full_sft.py              # GPU 여러 개
    ```

    | 인자 | 기본값 | 이 노트북 |
    | --- | --- | --- |
    | `--from_weight` | `pretrain` | `caspar/MiniMind-Dense-64M-Mixed-SFT-v1.0` |
    | `--learning_rate` | 1e-5 | 1e-5 (위젯으로 변경) |
    | `--batch_size` × `--accumulation_steps` | 16 × 1 | 4 × 1 |
    | `--epochs` | 2 | 10 |
    | `--max_seq_len` | 768 | 96 |
    | `--dtype` | bfloat16 (GPU) | bfloat16 autocast (GPU) / fp32 (CPU) |

    결과 `../out/full_sft_768.pth`는 대화 모델로 바로 쓰거나(`eval_llm.py`), LoRA·DPO·강화 학습의 출발점이 됩니다.
    같은 정체성 과제를 매개변수의 0.6%만 학습해서 하는 방법은 LoRA 노트북(`minimind_lora_tour.py`)에서 다룹니다.

    ## 7. 정리

    | 항목 | 핵심 |
    | --- | --- |
    | 학습 루프 | 사전 학습과 같음 (다음 토큰 예측, AdamW, clip, `get_lr`) |
    | 데이터 | 대화를 채팅 템플릿으로 한 줄의 글로 바꿈 |
    | 정답 | `<|im_start|>assistant\n` 뒤부터 `<|im_end|>\n`까지만, 모든 턴 |
    | 증강 | system 20% 추가, 빈 think 80% 삭제 → 추론 시 빈 think를 지우는 이유 |
    | 잘 배우는 것 | 답의 형식, 정체성, 말투 (질문 뜻을 이미 아는 모델 위에서) |
    | 어려운 것 | 작은 데이터로 새 사실 넣기 (표현이 바뀌면 지어냄) |
    | 주의 | 학습률이 크면 원래 능력을 잊음 → 작은 학습률, replay |

    ## 8. 직접 해 볼 과제

    각 과제를 하기 전에 **결과를 먼저 예측**해 적고, 바꿔 실행한 뒤 비교하세요.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({
        "과제 1. 학습률 2e-4 (4-2절)": mo.md("""
    학습률을 `2e-4`로 바꾸고 A의 일반 질문 답을 보세요.

    *해설:* 일반 질문에도 "I am Mimi, ..."라고 답합니다. 작은 데이터에 큰 학습률을 쓰면 모델이 새 데이터만 외우고 원래 능력을 잃습니다.
    """),
        "과제 2. Zentor 질문 표현 늘리기 (1절)": mo.md("""
    `zentor_train`에 같은 사실을 다른 표현으로 묻는 질문(예: `"Tell me the color of Zentor."`가 아닌 `"Which color does Zentor have?"`)을
    여러 개 추가하면 평가 질문에 대한 정답률이 오를까요?

    *해설:* 같은 사실을 다양한 표현으로 보여 줄수록 일반화가 좋아지는 경향이 있습니다. 새 지식을 넣으려면 같은 사실을 여러 형태로 반복해야 합니다.
    """),
        "과제 3. 빈 think 블록 남기기 (4-1절)": mo.md("""
    `prompt_ids`에서 `.replace("<think>\\n\\n</think>\\n\\n", "")`를 지우고 A 모델에게 질문하면 답이 달라질까요?

    *해설:* 학습 데이터의 80%는 think 블록이 없으므로, 추론 형식이 바뀌면 답이 흔들릴 수 있습니다. 학습과 추론의 형식을 맞추는 것이 중요합니다.
    """),
        "과제 4. 생각 데이터 만들기 (3절)": mo.md("""
    `train_pairs`의 대화에 `reasoning_content`를 넣어 학습한 뒤 `open_thinking=True` 프롬프트로 질문해 보세요.

    *해설:* 생각 과정도 정답에 포함되므로 모델은 `<think>` 안에 생각을 쓰고 나서 답하는 형식을 배웁니다.
    """),
        "과제 5. 정체성을 바꾸면 무엇이 바뀌나 (4-2절)": mo.md("""
    `IDENTITY`를 한국어 문장으로 바꾸면 이 영어 모델이 잘 배울까요?

    *해설:* 이 모델은 영어 위주로 학습되었고 어휘 6,400개라 한글은 바이트 단위로 쪼개집니다. 짧은 문장은 외울 수 있지만,
    익숙하지 않은 토큰이라 더 많은 학습이 필요하고 다른 답으로 새어 나오기 쉽습니다.
    """),
    })
    return


if __name__ == "__main__":
    app.run()
