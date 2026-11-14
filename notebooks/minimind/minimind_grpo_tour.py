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
    `uv run marimo edit minimind_grpo_tour.py`. 환경·준비물과 권장 순서는 [공통 README](README.md)를 참고하세요.
    위젯 변경 시 의존 셀이 다시 실행됩니다. 모델 크기·데이터·학습 설정 변경 전에는 해당 절의 재실행 범위를 확인하세요.
    """)
    return


@app.cell
def _():
    import copy
    import io
    import json
    import math
    import random
    import re
    import sys
    import tempfile
    from contextlib import redirect_stdout
    from pathlib import Path
    from types import SimpleNamespace

    import datasets
    import marimo as mo
    import matplotlib.pyplot as plt
    import torch
    import torch.nn.functional as F
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    from torch.optim.lr_scheduler import CosineAnnealingLR
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer

    # 노트북 위치(notebooks/minimind/)에서 두 단계 위가 저장소 루트입니다.
    repo_root = Path(__file__).resolve().parents[2]
    minimind_dir = repo_root / "projects/minimind"
    sys.path.insert(0, str(minimind_dir))

    # train_grpo.py 모듈과, 그 스크립트가 쓰는 원본 클래스·함수를 그대로 가져옵니다.
    from dataset.gsm8k_dataset import GSM8KDataset
    from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
    from trainer import train_grpo
    from trainer.gsm8k_reward import score_gsm8k_response
    from trainer.rollout_engine import TorchRolloutEngine
    from trainer.train_grpo import calculate_gsm8k_rewards_and_stats
    from trainer.trainer_utils import SkipBatchSampler, setup_seed

    # 이 노트북은 GPU에서 학습합니다(학습 중 매 step 답을 생성하므로 CPU로는 너무 느립니다).
    device = "cuda"
    # train_grpo.py의 '3. 혼합 정밀도 설정'(--dtype bfloat16). 원본처럼 하나의 autocast 객체를 rollout과 학습에 같이 씁니다.
    autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16)

    datasets.disable_progress_bars()
    tokenizer = AutoTokenizer.from_pretrained(minimind_dir / "model")
    return (
        CosineAnnealingLR,
        DataLoader,
        F,
        GSM8KDataset,
        MiniMindConfig,
        MiniMindForCausalLM,
        Path,
        SimpleNamespace,
        SkipBatchSampler,
        TorchRolloutEngine,
        autocast_ctx,
        calculate_gsm8k_rewards_and_stats,
        copy,
        device,
        hf_hub_download,
        io,
        json,
        load_file,
        math,
        mo,
        plt,
        random,
        re,
        redirect_stdout,
        score_gsm8k_response,
        setup_seed,
        tempfile,
        tokenizer,
        torch,
        train_grpo,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.vstack([
        mo.md(r"""
    # MiniMind GRPO(Group Relative Policy Optimization) 따라가기

    DPO는 미리 모아 둔 "좋은 답 / 나쁜 답" 쌍으로 학습했습니다. 그런데 수학 문제처럼 **정답을 기계적으로 채점할 수 있는**
    과제라면, 굳이 답을 미리 모을 필요가 없습니다. 모델이 **직접 답을 여러 개 써 보고**, 채점해서
    **맞힌 답 쪽으로** 조금씩 옮겨 가면 됩니다. 이것이 GRPO입니다.
    MiniMind에서는 `projects/minimind/trainer/train_grpo.py`가 이 일을 합니다.

    이 노트북은 SFT까지 학습된 **64M MiniMind 모델**
    [`caspar/MiniMind-Dense-64M-Mixed-SFT-v1.0`](https://huggingface.co/caspar/MiniMind-Dense-64M-Mixed-SFT-v1.0)에서 출발합니다.
    `train_grpo.py`의 `--from_weight full_sft`에 해당하는 모델입니다.

    **준비:** 처음 실행할 때 Hugging Face에서 가중치(약 128MB)를 내려받으므로 인터넷이 필요합니다.
    **NVIDIA GPU가 필요합니다**(`notebooks/minimind`의 uv 환경은 CUDA 13.2용 PyTorch를 설치합니다).
    RTX 3080 Ti(12GB) 기준으로 학습 1회가 약 1분, 노트북 전체가 3분 정도 걸립니다.

    **사전 지식:** SFT 노트북의 채팅 템플릿, DPO 노트북의 policy / reference와 "답의 로그 확률".

    ## 0. 큰 그림

    """),
        mo.mermaid(r"""
    flowchart LR
     P["사전 학습"] --> S["SFT"] --> G["GRPO"]
     G --> A["policy: 답 생성 및 학습"]
     G --> R["reference: SFT 복사본, 고정"]
     G --> C["보상: score_gsm8k_response 정답 채점 함수"]
    """),
        mo.md(r"""

    학습 한 step은 다음과 같습니다.

    """),
        mo.mermaid(r"""
    flowchart TB
     Q["질문 B개"] --> G["질문마다 답 G개 생성: rollout"]
     G --> R["답마다 0 또는 1 보상"]
     R --> A["같은 질문의 G개 보상으로 advantage 계산"]
     A --> P["좋은 답의 토큰 확률 증가, 나쁜 답 감소"]
     K["reference와의 KL 규제"] --> P
    """),
        mo.md(r"""

    **DPO와 비교**

    | | DPO | GRPO |
    | --- | --- | --- |
    | 데이터 | 미리 모은 chosen·rejected 쌍 | **질문만**. 답은 학습 중에 모델이 직접 생성 |
    | 방식 | 오프라인 | **온라인**: 매 step 새로 생성(rollout) |
    | 좋고 나쁨 | 데이터에 이미 표시됨 | **보상 함수**(또는 보상 모델)가 채점 |
    | 비교 기준 | reference 모델 | **같은 질문에 대한 답 G개의 평균 보상** (PPO의 critic 모델이 필요 없음) |
    | 기본 학습률 | 4e-8 | 3e-7 |
    """)
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. 베이스 모델 불러오기

    DPO 노트북과 같습니다. Hugging Face 저장소의 모델 코드는 실행하지 않고, **설정과 가중치만** 버전 `v1.0`으로 고정해
    내려받아 저장소의 원본 `MiniMindForCausalLM`에 넣습니다.

    모델 카드에 따르면 이 모델은 FineWeb-Edu로 사전 학습한 뒤 UltraChat·SmolTalk·**GSM8K** 대화로 SFT한 **영어** 모델입니다.
    그래서 GSM8K 풀이 형식(`<<6*7=42>>`처럼 계산을 적고 마지막 줄에 `#### 42`)을 이미 알고 있습니다.
    """)
    return


@app.cell(hide_code=True)
def _(
    MiniMindConfig,
    MiniMindForCausalLM,
    device,
    hf_hub_download,
    json,
    load_file,
    mo,
    torch,
):
    mo.stop(
        not torch.cuda.is_available(),
        mo.md("**GPU가 필요합니다.** `torch.cuda.is_available()`가 `False`입니다. NVIDIA 드라이버와 CUDA용 PyTorch를 확인하세요."),
    )
    BASE_REPO = "caspar/MiniMind-Dense-64M-Mixed-SFT-v1.0"
    BASE_REVISION = "v1.0"  # 버전 고정: 저장소가 바뀌어도 같은 가중치

    _config_path = hf_hub_download(BASE_REPO, "config.json", revision=BASE_REVISION)
    _weights_path = hf_hub_download(BASE_REPO, "model.safetensors", revision=BASE_REVISION)

    base_config = MiniMindConfig(**json.load(open(_config_path)))
    sft_model = MiniMindForCausalLM(base_config)
    _state = {k: v.float() for k, v in load_file(_weights_path).items()}  # fp16 → fp32
    _missing, _unexpected = sft_model.load_state_dict(_state, strict=False)
    sft_model.to(device).eval()  # 가중치는 fp32로 두고, 계산할 때만 bf16 autocast를 씁니다

    # lm_head는 Embedding과 같은 가중치라 파일에 따로 저장되지 않습니다.
    assert _missing == ["lm_head.weight"] and not _unexpected
    assert sft_model.lm_head.weight is sft_model.model.embed_tokens.weight
    base_param_count = sum(p.numel() for p in sft_model.parameters())
    return BASE_REPO, BASE_REVISION, base_config, base_param_count, sft_model


@app.cell(hide_code=True)
def _(BASE_REPO, BASE_REVISION, base_config, base_param_count, mo, torch):
    _gpu = torch.cuda.get_device_properties(0)
    mo.md(f"""
    `{BASE_REPO}` (`{BASE_REVISION}`): 매개변수 **{base_param_count:,}개**, hidden {base_config.hidden_size},
    {base_config.num_hidden_layers}층, Query head {base_config.num_attention_heads}개 / KV head {base_config.num_key_value_heads}개, 어휘 {base_config.vocab_size}.

    실행 장치: **GPU (`{_gpu.name}`, {_gpu.total_memory / 2**30:.1f} GiB)**
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. 과제: 한 단계짜리 산수 문장제

    `train_grpo.py`에는 보상 방식이 두 가지 있습니다.

    | `--reward_type` | 데이터 | 보상 |
    | --- | --- | --- |
    | `model` (기본값) | `RLAIFDataset` (일반 대화) | 길이·think 형식·반복 규칙 + **보상 모델**(InternLM2 1.8B) 점수 |
    | `gsm8k` | `GSM8KDataset` (수학 문제) | **정답 채점 함수** `score_gsm8k_response` (맞으면 1, 아니면 0) |

    이 노트북은 `gsm8k` 경로를 씁니다. 보상 모델(1.8B)을 따로 띄울 필요가 없고, 보상이 왜 그렇게 나왔는지 사람이 바로 확인할 수 있습니다.

    **그런데 실제 GSM8K는 이 모델에게 너무 어렵습니다.** 저장소의 실험 기록(`projects/minimind/prompts/grpo_training.md`)에서
    같은 베이스 모델은 GSM8K 테스트 1,319문제 중 **14문제(1.06%)** 만 맞혔고, 전체 1 epoch GRPO 뒤에도 **18문제(1.36%)** 였습니다.
    거의 모든 질문에서 답 G개가 **전부 0점**이면, 4절에서 보듯 비교할 거리가 없어 학습 신호가 0입니다.

    그래서 **모델이 가끔은 맞히는** 쉬운 문제를 만듭니다. GSM8K와 같은 형식의 한 단계짜리 덧셈·곱셈 문장제입니다.

    - 학습 128문제(덧셈 64 + 곱셈 64), 평가 256문제(학습에 없던 문제, 덧셈 128 + 곱셈 128)
    - `question`, `answer`(풀이 + `#### 정답`) 열을 가진 JSONL로 저장해 **원본 `GSM8KDataset`** 으로 읽습니다.
    """)
    return


@app.cell
def _(GSM8KDataset, Path, json, random, tempfile, tokenizer):
    NAMES = ["Tom", "Anna", "Ben", "Lily", "Sam", "Mia", "Leo", "Emma"]
    ITEMS = ["apples", "pencils", "cookies", "books", "marbles", "stickers", "cards", "balloons"]

    def make_problem(rng, kind):
        name, item = rng.choice(NAMES), rng.choice(ITEMS)
        a, b = rng.randint(2, 9), rng.randint(2, 9)
        if kind == "add":
            question = f"{name} has {a} {item}. {name} gets {b} more {item}. How many {item} does {name} have now?"
            return question, f"{name} has {a} + {b} = {a + b} {item} now.\n#### {a + b}"
        question = f"{name} buys {a} boxes. Each box has {b} {item}. How many {item} does {name} buy in total?"
        return question, f"{name} buys {a} * {b} = {a * b} {item} in total.\n#### {a * b}"

    _rng = random.Random(0)
    problems, _seen = [], set()
    while len(problems) < 128 + 256:
        _kind = ("add", "mul")[len(problems) % 2]          # 덧셈·곱셈을 번갈아
        _q, _a = make_problem(_rng, _kind)
        if _q not in _seen:                                 # 학습·평가에 같은 문제가 없도록
            _seen.add(_q)
            problems.append({"question": _q, "answer": _a, "kind": _kind})
    train_rows, test_rows = problems[:128], problems[128:]

    data_dir = Path(tempfile.mkdtemp(prefix="minimind_grpo_data_"))
    for _name, _rows in [("grpo_train.jsonl", train_rows), ("grpo_test.jsonl", test_rows)]:
        with open(data_dir / _name, "w", encoding="utf-8") as _f:
            for _row in _rows:
                _f.write(json.dumps(_row, ensure_ascii=False) + "\n")

    MAX_SEQ_LEN = 256   # 원본 기본값 768: 프롬프트 최대 길이 (넘으면 GSM8KDataset이 오류를 냅니다)
    MAX_GEN_LEN = 96    # 원본 기본값 1024: 답 최대 길이. 한 단계 문제의 답은 30~60토큰이면 충분합니다
    train_ds = GSM8KDataset(str(data_dir / "grpo_train.jsonl"), tokenizer, max_length=MAX_SEQ_LEN)
    test_ds = GSM8KDataset(str(data_dir / "grpo_test.jsonl"), tokenizer, max_length=MAX_SEQ_LEN)
    test_kinds = [r["kind"] for r in test_rows]
    assert len(train_ds) == 128 and len(test_ds) == 256
    assert not {r["question"] for r in train_rows} & {r["question"] for r in test_rows}
    return MAX_GEN_LEN, MAX_SEQ_LEN, test_ds, test_kinds, train_ds, train_rows


@app.cell(hide_code=True)
def _(mo, result_table, train_ds, train_rows):
    mo.vstack([
        result_table([{"종류": r["kind"], "question": r["question"], "answer (정답 풀이)": r["answer"]} for r in train_rows],
                    selection=None, page_size=6),
        mo.md(f"`GSM8KDataset`이 만든 학습 샘플 하나 (`{{'prompt': ..., 'answer': ...}}`):\n"
              f"```text\n{train_ds[0]['prompt']}\n```\n"
              f"`answer` = `{train_ds[0]['answer']!r}`"),
        mo.md(r"""
    `GSM8KDataset`이 하는 일:

    - 질문 끝에 **"Solve step by step. End your answer with #### followed by the final number."** 를 붙입니다.
    - 채팅 템플릿을 적용하고 빈 think 블록(`<think>\n\n</think>\n\n`)을 지웁니다(SFT 때 형식과 맞춤).
    - **정답(`answer`)은 프롬프트에 넣지 않고** 따로 보관합니다. 채점할 때만 씁니다.
    - SFT 데이터셋과 달리 **토큰화도, 정답 토큰(`labels`)도 만들지 않습니다.** 모델이 쓸 답은 학습 중에 생성되기 때문입니다.
    """),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 2-1. SFT 모델은 얼마나 맞힐까?

    평가 256문제를 두 방식으로 풀게 합니다.

    - **greedy**: 매번 확률이 가장 높은 토큰. 학습이 끝난 뒤 실제로 쓸 때의 답입니다.
    - **샘플링 4회**: rollout과 같은 설정(temperature 0.8, top-p 0.85, top-k 50)으로 4번씩. 학습 중 모델이 "보는" 답입니다.
    """)
    return


@app.cell
def _(
    MAX_GEN_LEN,
    autocast_ctx,
    device,
    re,
    score_gsm8k_response,
    setup_seed,
    test_ds,
    test_kinds,
    tokenizer,
    torch,
):
    def completion_texts(output_ids, prompt_len):
        """생성 결과에서 답 부분만 문자열로, 그리고 EOS 전까지의 토큰 수를 돌려줍니다."""
        completion = output_ids[:, prompt_len:]
        is_eos = completion == tokenizer.eos_token_id
        lengths = torch.where(is_eos.any(1), is_eos.int().argmax(1), completion.size(1)).tolist()
        return tokenizer.batch_decode(completion, skip_special_tokens=True), lengths

    @torch.no_grad()
    def evaluate(model, n_samples=4, chunk=64, seed=0):
        """평가 문제 전체를 greedy 1회 + 샘플링 n_samples회 풀고 원본 채점 함수로 채점합니다."""
        model.eval()
        setup_seed(seed)
        greedy, sampled = [], []
        for start in range(0, len(test_ds), chunk):
            part = test_ds.samples[start:start + chunk]
            x = tokenizer([s["prompt"] for s in part], return_tensors="pt", padding=True,
                          padding_side="left", add_special_tokens=False).to(device)
            with autocast_ctx:
                g_out = model.generate(inputs=x.input_ids, attention_mask=x.attention_mask, max_new_tokens=MAX_GEN_LEN,
                                       do_sample=False, eos_token_id=tokenizer.eos_token_id)
                s_out = model.generate(inputs=x.input_ids.repeat_interleave(n_samples, 0),
                                       attention_mask=x.attention_mask.repeat_interleave(n_samples, 0),
                                       max_new_tokens=MAX_GEN_LEN, temperature=0.8, eos_token_id=tokenizer.eos_token_id)
            g_texts, g_lens = completion_texts(g_out, x.input_ids.size(1))
            s_texts, _ = completion_texts(s_out, x.input_ids.size(1))
            for s, text, n in zip(part, g_texts, g_lens):
                score, reason = score_gsm8k_response(text, s["answer"])
                gold = s["answer"].split("####")[1].strip()
                greedy.append({"text": text, "len": n, "score": score, "reason": reason,
                               "gold_in_text": gold in re.findall(r"\d+", text)})
            for i, text in enumerate(s_texts):
                sampled.append(score_gsm8k_response(text, part[i // n_samples]["answer"])[0])

        def acc(kind):
            return sum(g["score"] for g, k in zip(greedy, test_kinds) if k == kind) / test_kinds.count(kind)

        n = len(greedy)
        return {
            "greedy 정답률": sum(g["score"] for g in greedy) / n,
            "샘플링 정답률": sum(sampled) / len(sampled),
            "덧셈 (greedy)": acc("add"),
            "곱셈 (greedy)": acc("mul"),
            "평균 답 길이(토큰)": sum(g["len"] for g in greedy) / n,
            "형식 오류": sum(g["reason"] == "format_error" for g in greedy) / n,
            "정답 숫자를 쓰고도 최종 답은 틀림": sum(g["gold_in_text"] and g["score"] == 0 for g in greedy) / n,
            "greedy": greedy,
        }

    return (evaluate,)


@app.cell
def _(evaluate, sft_model):
    sft_eval = evaluate(sft_model)
    return (sft_eval,)


@app.cell(hide_code=True)
def _(mo, result_table, sft_eval, test_ds):
    _keys = ["greedy 정답률", "샘플링 정답률", "덧셈 (greedy)", "곱셈 (greedy)", "형식 오류", "정답 숫자를 쓰고도 최종 답은 틀림"]
    mo.vstack([
        mo.md("| " + " | ".join(_keys) + " | 평균 답 길이 |\n| " + " --- |" * (len(_keys) + 1) + "\n| "
              + " | ".join(f"{sft_eval[k]:.1%}" for k in _keys) + f" | {sft_eval['평균 답 길이(토큰)']:.1f} 토큰 |"),
        result_table([{"질문": s["prompt"].split("user\n")[1].split("\nSolve")[0], "정답": s["answer"].split("#### ")[1],
                      "SFT 모델 답 (greedy)": g["text"], "보상": g["score"]}
                     for s, g in zip(test_ds.samples[:8], sft_eval["greedy"][:8])], selection=None),
        mo.md(r"""
    **형식은 거의 완벽**하지만 **정답률은 낮습니다.** 답을 보면 첫 줄에서 맞는 계산을 해 놓고도 **필요 없는 단계를 덧붙여**
    답을 망치는 경우가 많습니다("정답 숫자를 쓰고도 최종 답은 틀림"). 곱셈표를 틀리는 경우도 있습니다.

    샘플링하면 **가끔은 맞힙니다.** GRPO가 쓸 수 있는 것은 바로 이 "가끔 맞힌 답"입니다.
    """),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. 보상: `score_gsm8k_response`

    `train_grpo.py`의 `calculate_gsm8k_rewards_and_stats`는 답마다 `trainer/gsm8k_reward.py`의 이 함수를 부릅니다.

    ```python
    def score_gsm8k_response(response, answer):
        expected = extract_gsm8k_answer(answer)          # 정답 풀이의 '#### 숫자'
        ...                                               # think 블록이 있으면 하나만, 닫혀 있어야 함
        if re.search(r'(?:^|\n)[ \t]*####[ \t]+[^\n]+\s*\Z', final) is None:
            return 0.0, 'format_error'                    # '#### 숫자'가 마지막 줄이 아님
        predicted = extract_gsm8k_answer(response)       # '####'가 정확히 한 번, 뒤는 숫자만
        if predicted is None:
            return 0.0, 'format_error'
        return float(predicted == expected), 'correct' if predicted == expected else 'wrong_number'
    ```

    - 보상은 **0 또는 1**뿐입니다. **형식만 맞췄다고 주는 점수는 없습니다.** 맞는 숫자만 1점입니다.
    - `####`를 여러 번 써서 후보를 나열하는 꼼수는 `format_error`(0점)입니다.
    - 풀이 과정이 맞는지는 보지 않습니다. **최종 숫자만** 봅니다.
    - 숫자는 `Decimal`로 비교하므로 `1,000`, `1000.0`, `+1000`은 모두 같은 값입니다.
    """)
    return


@app.cell
def _(score_gsm8k_response):
    _gold = "Tom buys 6 * 7 = 42 apples in total.\n#### 42"
    reward_cases = [
        ("정답", "Tom buys 6 * 7 = <<6*7=42>>42 apples.\n#### 42"),
        ("같은 값, 다른 표기", "Tom buys 42 apples.\n#### 42.0"),
        ("틀린 숫자", "Tom buys 6 * 7 = <<6*7=48>>48 apples.\n#### 48"),
        ("계산은 맞지만 덧붙인 단계가 틀림", "6 * 7 = <<6*7=42>>42\nThen 42 * 7 = <<42*7=294>>294\n#### 294"),
        ("#### 없음", "Tom buys 42 apples in total."),
        ("#### 두 번 (후보 나열)", "#### 40\n#### 42"),
        ("#### 뒤에 단위", "Tom buys 42 apples.\n#### 42 apples"),
        ("think 블록 뒤 정답", "<think>6 * 7 = 42</think>\n#### 42"),
    ]
    reward_table = [{"경우": name, "답": text, **dict(zip(["보상", "판정"], score_gsm8k_response(text, _gold)))}
                    for name, text in reward_cases]
    assert [r["보상"] for r in reward_table] == [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    return (reward_table,)


@app.cell(hide_code=True)
def _(mo, result_table, reward_table):
    mo.vstack([mo.md("정답 풀이 `'... #### 42'`에 대한 채점 결과:"), result_table(reward_table, selection=None)])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Rollout: 질문마다 답 G개 생성하기

    `grpo_train_epoch`의 처음 부분입니다.

    ```python
    prompts = batch['prompt']                                     # 질문 B개 (문자열)
    prompt_inputs = tokenizer(prompts, return_tensors="pt", padding=True,
                              padding_side="left", add_special_tokens=False)   # 길이가 다르면 '왼쪽'에 패딩
    rollout_result = rollout_engine.rollout(
        prompt_ids=prompt_inputs["input_ids"], attention_mask=prompt_inputs["attention_mask"],
        num_generations=args.num_generations,                     # G (기본값 6)
        max_new_tokens=args.max_gen_len, temperature=0.8)
    ```

    `TorchRolloutEngine.rollout`(`trainer/rollout_engine.py`)은 각 질문을 G번 복사해(`repeat_interleave`) **한 번에 샘플링**합니다.

    - 출력 순서는 **질문 순서대로 G개씩**: `[q0_g0, q0_g1, ..., q0_g5, q1_g0, ...]`. 그래서 나중에 `view(-1, G)`로 그룹을 묶습니다.
    - 왼쪽 패딩이라 모든 행에서 **답이 같은 위치에서 시작**합니다. 답은 `output_ids[:, P:]`로 잘라 냅니다.
    - 샘플링은 temperature 0.8에 `generate`의 기본값 **top-p 0.85, top-k 50**이 함께 적용됩니다.
    - 생성하면서 각 답 토큰의 로그 확률 `per_token_logps`도 계산해 둡니다. 이 값이 PPO의 **old policy** 로그 확률입니다.
    - `generate`는 EOS를 낸 행에도 배치가 끝날 때까지 **EOS를 계속 채웁니다.** 그래서 `completion_mask`는 일단 전부 1이고,
      EOS 뒤를 지우는 일은 학습 코드가 따로 합니다(6절).

    SFT 모델이 가끔 맞히는 학습 문제 2개(덧셈 하나, 곱셈 하나)로 B=2, G=6 rollout을 해 봅니다.
    샘플링이므로 GPU 종류나 드라이버가 다르면 생성되는 답이 달라질 수 있습니다.
    """)
    return


@app.cell
def _(
    MAX_GEN_LEN,
    TorchRolloutEngine,
    autocast_ctx,
    calculate_gsm8k_rewards_and_stats,
    device,
    setup_seed,
    sft_model,
    tokenizer,
    train_ds,
):
    DEMO_G = 6
    DEMO_SEED = 8
    DEMO_INDICES = [44, 39]   # SFT 모델이 가끔 맞히는 학습 문제: 덧셈 하나(44번), 곱셈 하나(39번)
    demo_batch = {"prompt": [train_ds[i]["prompt"] for i in DEMO_INDICES],
                  "answer": [train_ds[i]["answer"] for i in DEMO_INDICES]}   # DataLoader가 만드는 배치와 같은 모양

    setup_seed(DEMO_SEED)  # 8-2절에서 원본 함수로 같은 rollout을 재현하기 위해 고정
    demo_inputs = tokenizer(demo_batch["prompt"], return_tensors="pt", padding=True, return_token_type_ids=False,
                            padding_side="left", add_special_tokens=False).to(device)
    demo_engine = TorchRolloutEngine(sft_model, tokenizer, device, autocast_ctx)
    demo_rollout = demo_engine.rollout(prompt_ids=demo_inputs["input_ids"], attention_mask=demo_inputs["attention_mask"],
                                       num_generations=DEMO_G, max_new_tokens=MAX_GEN_LEN, temperature=0.8)
    # 원본 train_grpo.py의 GSM8K 보상 함수
    demo_rewards, demo_reward_stats = calculate_gsm8k_rewards_and_stats(
        demo_rollout, demo_batch["answer"], DEMO_G, tokenizer.eos_token_id, MAX_GEN_LEN, device)

    _B, _P = demo_inputs["input_ids"].shape
    assert demo_rollout.output_ids.shape[0] == _B * DEMO_G
    assert demo_rollout.completion_ids.shape == demo_rollout.per_token_logps.shape
    assert bool(demo_rollout.completion_mask.all())          # torch 엔진은 마스크를 전부 1로 돌려줌
    assert int(demo_rollout.prompt_lens[0]) == _P             # 왼쪽 패딩 → 모든 행의 프롬프트 길이 = P
    return (
        DEMO_G,
        DEMO_SEED,
        demo_batch,
        demo_inputs,
        demo_reward_stats,
        demo_rewards,
        demo_rollout,
    )


@app.cell(hide_code=True)
def _(
    DEMO_G,
    demo_inputs,
    demo_reward_stats,
    demo_rewards,
    demo_rollout,
    mo,
    result_table,
    tokenizer,
):
    _pads = (demo_inputs["attention_mask"] == 0).sum(1).tolist()
    mo.vstack([
        mo.md(f"""
    - 프롬프트 `input_ids`: `{tuple(demo_inputs['input_ids'].shape)}` (왼쪽 패딩 토큰 수: 질문별 {_pads})
    - `output_ids`: `{tuple(demo_rollout.output_ids.shape)}` = [B·G, 프롬프트 + 가장 긴 답]
    - `completion_ids`, `per_token_logps`: `{tuple(demo_rollout.completion_ids.shape)}`
    - 채점 진단(`reward_stats`): 정답 {demo_reward_stats['reward/correct_rate']:.0%}, 틀린 숫자 {demo_reward_stats['reward/wrong_number_rate']:.0%},
      형식 오류 {demo_reward_stats['reward/format_error_rate']:.0%}, **보상이 섞인 그룹 비율 {demo_reward_stats['reward/mixed_group_rate']:.0%}**
    """),
        result_table([{"질문": i // DEMO_G, "답 번호": i % DEMO_G, "생성한 답": t, "보상": demo_rewards[i].item(),
                      "토큰 수": int((demo_rollout.completion_ids[i] != tokenizer.eos_token_id).sum())}
                     for i, t in enumerate(demo_rollout.completions)], selection=None, page_size=12),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Advantage: 그룹 안에서 비교하기

    ```python
    grouped_rewards = rewards.view(-1, args.num_generations)                        # [B, G]
    mean_r = grouped_rewards.mean(dim=1).repeat_interleave(args.num_generations)   # 질문별 평균 → 답마다 복사
    std_r = grouped_rewards.std(dim=1, unbiased=False).repeat_interleave(args.num_generations)
    advantages = (rewards - mean_r) / (std_r + 1e-4)                               # [B*G]
    ```

    GRPO의 핵심입니다. **"이 답이 같은 질문의 다른 답들보다 얼마나 좋은가"** 를 표준화한 값입니다.

    $$
    A_i = \frac{r_i - \mathrm{mean}(r_1, \dots, r_G)}{\mathrm{std}(r_1, \dots, r_G) + 10^{-4}}
    $$

    **질문별 평균을 답마다 복사하는 이유**는 각 답변에서 **그 답변이 속한 질문의 평균 보상**을 빼기 위해서입니다.
    질문이 2개이고 질문마다 답변을 3개씩 생성했다면(`B=2`, `G=3`), 같은 질문의 답변이 연속해서 배치됩니다.
    아래 숫자는 계산을 설명하기 위한 예시 보상입니다.

    ```python
    rewards = [1, 2, 3,   2, 4, 6]  # [B*G]: 질문 A의 답 3개, 질문 B의 답 3개
    grouped_rewards = [[1, 2, 3],    # [B, G]: 질문 A
                       [2, 4, 6]]   #         질문 B

    # 질문별 평균은 [2, 4] → repeat_interleave(3)으로 각 답변 위치에 배치
    mean_r = [2, 2, 2,   4, 4, 4]   # [B*G]

    # 실제 코드에서는 텐서끼리 원소별로 뺍니다.
    rewards - mean_r
    # [-1, 0, 1,   -2, 0, 2]
    ```

    이렇게 길이와 순서를 맞추면 1차원 `rewards`에서 질문별 평균을 바로 뺄 수 있습니다.
    표준편차도 같은 방식으로 복사해서 나누면, 각 답변의 `advantage`는 **같은 질문의 답변들 사이에서의 상대적 점수**가 됩니다.
    양수면 그 질문의 평균보다 좋은 답변, 음수면 평균보다 나쁜 답변이며, 평균과 같으면 0입니다.

    `repeat_interleave`는 현재 코드의 1차원 형태에 맞추기 위한 처리입니다.
    `[B, G]` 형태를 유지하면 `[B, 1]` 평균·표준편차를 각 열에 적용하는 **브로드캐스팅**으로도 같은 계산을 할 수 있습니다.

    ```python
    grouped_rewards = rewards.view(-1, args.num_generations)  # [B, G]
    mean_r = grouped_rewards.mean(dim=1, keepdim=True)        # [B, 1]
    std_r = grouped_rewards.std(dim=1, unbiased=False, keepdim=True)
    advantages = (grouped_rewards - mean_r) / (std_r + 1e-4)   # [B, G]
    advantages = advantages.reshape(-1)                      # [B*G]
    ```

    - PPO는 "기대 보상"을 예측하는 **critic 모델**을 따로 학습해 기준선으로 씁니다. GRPO는 그 대신 **같은 질문의 답 G개 평균**을 씁니다.
      그래서 모델이 하나 줄어듭니다.
    - 질문이 쉬운지 어려운지가 자동으로 보정됩니다. 어려운 질문에서 하나만 맞히면 그 답은 큰 양수 advantage를 받습니다.
    - 그룹의 advantage 합은 항상 0입니다. 확률을 올리는 답과 내리는 답이 함께 생깁니다.
    - **보상이 모두 같으면**(전부 0점이든 전부 1점이든) advantage가 전부 0이 되어 **그 질문에서는 배울 것이 없습니다.**
    """)
    return


@app.cell
def _(DEMO_G, demo_rewards, torch):
    def group_advantages(rewards, num_generations):
        """train_grpo.py의 advantage 계산 4줄을 그대로 옮긴 것"""
        grouped_rewards = rewards.view(-1, num_generations)
        mean_r = grouped_rewards.mean(dim=1).repeat_interleave(num_generations)
        std_r = grouped_rewards.std(dim=1, unbiased=False).repeat_interleave(num_generations)
        return (rewards - mean_r) / (std_r + 1e-4)

    demo_advantages = group_advantages(demo_rewards, DEMO_G)

    advantage_examples = {
        "6개 중 1개 정답": [1, 0, 0, 0, 0, 0],
        "6개 중 3개 정답": [1, 1, 1, 0, 0, 0],
        "6개 중 5개 정답": [1, 1, 1, 1, 1, 0],
        "모두 정답": [1, 1, 1, 1, 1, 1],
        "모두 오답": [0, 0, 0, 0, 0, 0],
        "G=2, 1개 정답": [1, 0],
    }
    advantage_table = []
    for _name, _r in advantage_examples.items():
        _a = group_advantages(torch.tensor(_r, dtype=torch.float32), len(_r))
        assert abs(_a.sum().item()) < 1e-4                    # 그룹 안 advantage 합 = 0
        advantage_table.append({"그룹": _name, "보상": str(_r), "advantage": str([round(v, 2) for v in _a.tolist()])})
    return advantage_table, demo_advantages


@app.cell(hide_code=True)
def _(
    DEMO_G,
    advantage_table,
    demo_advantages,
    demo_rewards,
    mo,
    result_table,
):
    mo.vstack([
        result_table(advantage_table, selection=None),
        mo.md("드물게 맞힌 답일수록 큰 양수(1개 정답 → +2.24), 흔한 오답일수록 작은 음수(−0.45)를 받습니다. "
              "반대로 대부분 맞히는 질문에서는 **틀린 답 하나를 크게 깎습니다**(5개 정답 → −2.24).\n\n"
              "4절 rollout의 실제 값:"),
        result_table([{"질문": i // DEMO_G, "답 번호": i % DEMO_G, "보상": demo_rewards[i].item(),
                      "advantage": round(demo_advantages[i].item(), 3)} for i in range(len(demo_rewards))],
                    selection=None, page_size=12),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. 답 토큰의 로그 확률과 마스크

    DPO처럼 "모델이 이 답을 낼 확률"이 필요한데, GRPO는 **답 전체가 아니라 토큰 하나하나**의 로그 확률을 씁니다.

    ```python
    full_mask = (outputs != tokenizer.pad_token_id).long()               # 왼쪽 패딩만 0
    logp_pos = prompt_lens.unsqueeze(1) - 1 + torch.arange(R).unsqueeze(0)   # 답 토큰 t를 예측하는 logits 위치
    full_mask.scatter_(1, logp_pos + 1, completion_mask)                 # 답 위치의 attention mask

    with autocast_ctx:                                                   # policy: bf16 autocast
        res = model(outputs, attention_mask=full_mask)
        per_token_logps = F.log_softmax(res.logits[:, :-1, :], dim=-1) \
            .gather(2, outputs[:, 1:].unsqueeze(-1)).squeeze(-1) \
            .gather(1, logp_pos)                                         # [B*G, R]: 답 토큰마다 log p
    with torch.no_grad():                                                # reference: autocast 밖 → fp32
        ref_per_token_logps = (ref_model로 같은 계산)

    # 첫 EOS까지만 1인 마스크
    is_eos = (completion_ids == tokenizer.eos_token_id) & completion_pad_mask
    eos_idx = torch.full((B*G,), R - 1); eos_idx[is_eos.any(1)] = is_eos.int().argmax(1)[is_eos.any(1)]
    completion_mask = (torch.arange(R) <= eos_idx.unsqueeze(1)) & completion_pad_mask
    ```

    - `logits[:, t]`는 토큰 `t+1`을 예측합니다. 그래서 `logits[:, :-1]`과 `outputs[:, 1:]`을 짝짓고, 답 부분 위치(`logp_pos`)만 고릅니다.
    - **첫 EOS는 포함**합니다(언제 멈출지도 학습 대상). **그 뒤에 채워진 EOS들은 제외**합니다.
    - 학습 첫 step에는 policy = rollout 때 모델 = reference입니다. 그래서 다시 계산한 로그 확률이 rollout 때 값과 같습니다
      ($\mathrm{ratio}=1$). 둘 다 bf16 autocast 안에서 계산하기 때문입니다.
    - **reference와의 차이(KL)는 정확히 0이 아닙니다.** 원본 코드에서 reference forward는 `autocast_ctx` **밖**에 있어 fp32로
      계산되고, policy는 bf16으로 계산됩니다. 같은 가중치라도 정밀도 차이만큼 로그 확률이 다릅니다. 아래에서 그 크기를 봅니다.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    # 위 코드를 길이 9짜리 예시 한 줄(P=5, R=4)로 따라가는 그림. CSS는 .gd 안으로만 한정합니다.
    _css = r"""
    <style>
    .gd{--panel:#fff;--fg:#1b1f27;--muted:#5d6574;--line:#d9dde4;--soft:#f3f4f7;
      --prompt:#3b6fd8;--prompt-bg:#e5edfc;--ans:#1f8a5b;--ans-bg:#dff3e9;--eos:#b8730a;--eos-bg:#fbefd8;
      --pad:#8a919d;--pad-bg:#eceef2;--off:#c24a3a;--off-bg:#fbe5e1;--sel:#6b4bd6;
      color:var(--fg);display:flex;flex-direction:column;gap:16px;line-height:1.6;font-size:14px}
    .dark .gd,.dark-theme .gd,[data-theme="dark"] .gd{--panel:#1a1e26;--fg:#e6e9ef;--muted:#9aa3b2;--line:#2e3440;--soft:#12151b;
      --prompt:#7ea3f2;--prompt-bg:#1f2b45;--ans:#5cc994;--ans-bg:#17342a;--eos:#e6a74a;--eos-bg:#3a2c14;
      --pad:#8a919d;--pad-bg:#262a33;--off:#ef7f6e;--off-bg:#3d1f1b;--sel:#a58ef5}
    .gd p{margin:0}
    .gd code{font-family:ui-monospace,monospace;font-size:.86em;background:var(--pad-bg);padding:1px 5px;border-radius:4px}
    .gd .step{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:16px;display:flex;flex-direction:column;gap:12px;min-width:0}
    .gd .ttl{font-weight:700;font-size:1.02rem;display:flex;gap:10px;align-items:baseline}
    .gd .ttl .n{font-family:ui-monospace,monospace;color:var(--sel)}
    .gd .legend{display:flex;flex-wrap:wrap;gap:6px 16px;font-size:.85rem;color:var(--muted)}
    .gd .legend span{display:inline-flex;align-items:center;gap:6px}
    .gd .sw{display:inline-block;width:14px;height:14px;border-radius:3px;border:1.5px solid}
    .gd .scroll{overflow-x:auto;min-width:0}
    .gd table.strip{border-collapse:separate;border-spacing:4px;font-family:ui-monospace,monospace;font-size:.82rem;margin:0;width:auto}
    .gd .strip th{font-weight:400;color:var(--muted);text-align:right;padding:0 8px 0 0;white-space:nowrap;font-size:.8rem;border:none;background:none}
    .gd .strip td{min-width:56px;height:32px;text-align:center;border-radius:5px;border:1.5px solid transparent;white-space:nowrap;padding:0 4px}
    .gd .strip td.idx{color:var(--muted);height:auto;font-size:.75rem}
    .gd .t-pad{background:var(--pad-bg);border-color:var(--pad)!important;color:var(--pad)}
    .gd .t-p{background:var(--prompt-bg);border-color:var(--prompt)!important;color:var(--prompt)}
    .gd .t-a{background:var(--ans-bg);border-color:var(--ans)!important;color:var(--ans)}
    .gd .t-e{background:var(--eos-bg);border-color:var(--eos)!important;color:var(--eos)}
    .gd .t-off{background:var(--off-bg);border-color:var(--off)!important;color:var(--off);text-decoration:line-through}
    .gd .b0{color:var(--muted);border-color:var(--line)!important}
    .gd .b1{color:var(--fg);border-color:var(--fg)!important;font-weight:600}
    .gd .dim{opacity:.35}
    .gd .pick{outline:2.5px solid var(--sel);outline-offset:1px}
    .gd .arrow{color:var(--sel);height:18px!important}
    .gd .blank{background:transparent}
    .gd .lbl{font-size:.75rem;color:var(--muted)}
    .gd ul{margin:0;padding-left:1.2em;display:flex;flex-direction:column;gap:4px}
    .gd .formula{font-family:ui-monospace,monospace;font-size:.82rem;background:var(--soft);border:1px solid var(--line);border-radius:6px;padding:10px 12px;overflow-x:auto;white-space:pre;margin:0}
    .gd .prec{display:grid;grid-template-columns:1fr auto 1fr;gap:12px}
    .gd .box{border:1.5px solid var(--line);border-radius:8px;padding:12px;display:flex;flex-direction:column;gap:4px;min-width:0}
    .gd .box.pol{border-color:var(--ans)} .gd .box.ref{border-color:var(--prompt)}
    .gd .tag{font-family:ui-monospace,monospace;font-size:.78rem;color:var(--muted)}
    .gd .mid{display:flex;flex-direction:column;justify-content:center;align-items:center;font-family:ui-monospace,monospace;font-size:.82rem;color:var(--muted);text-align:center}
    .gd .cmp{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px}
    .gd .cmp div{border-radius:8px;padding:12px;border:1px solid var(--line)}
    .gd .cmp .ok{background:var(--ans-bg)} .gd .cmp .warn{background:var(--eos-bg)}
    .gd .cmp b{display:block;font-family:ui-monospace,monospace;font-size:.88rem}
    @media (max-width:640px){.gd .prec,.gd .cmp{grid-template-columns:1fr}}
    </style>
    """
    _body = r"""
    <div class="gd">
      <p>rollout 결과 한 줄(<code>B*G</code> 중 1개)을 예시로, 위 코드가 어떤 위치의 값을 골라 쓰는지 따라갑니다.
      프롬프트 길이 <code>P=5</code>(왼쪽 패딩 2개 포함), 답 길이 <code>R=4</code>입니다.</p>
      <div class="legend">
        <span><i class="sw t-pad"></i>PAD (왼쪽 패딩)</span><span><i class="sw t-p"></i>프롬프트 토큰</span>
        <span><i class="sw t-a"></i>답 토큰</span><span><i class="sw t-e"></i>EOS</span>
        <span><i class="sw" style="border-color:var(--sel)"></i>선택된 위치</span>
      </div>

      <div class="step">
        <div class="ttl"><span class="n">①</span>입력 시퀀스와 attention mask</div>
        <div class="scroll"><table class="strip">
          <tr><th>위치</th><td class="idx">0</td><td class="idx">1</td><td class="idx">2</td><td class="idx">3</td><td class="idx">4</td><td class="idx">5</td><td class="idx">6</td><td class="idx">7</td><td class="idx">8</td></tr>
          <tr><th>outputs</th><td class="t-pad">PAD</td><td class="t-pad">PAD</td><td class="t-p">p₁</td><td class="t-p">p₂</td><td class="t-p">p₃</td><td class="t-a">a₁</td><td class="t-a">a₂</td><td class="t-e">EOS</td><td class="t-e">EOS</td></tr>
          <tr><th></th><td colspan="5" class="lbl" style="color:var(--prompt);border-top:2px solid var(--prompt);border-radius:0">프롬프트 P=5</td><td colspan="4" class="lbl" style="color:var(--ans);border-top:2px solid var(--ans);border-radius:0">completion_ids, R=4</td></tr>
          <tr><th>full_mask</th><td class="b0">0</td><td class="b0">0</td><td class="b1">1</td><td class="b1">1</td><td class="b1">1</td><td class="b1">1</td><td class="b1">1</td><td class="b1">1</td><td class="b1">1</td></tr>
        </table></div>
        <ul>
          <li><code>outputs != pad_token_id</code>: 왼쪽 PAD만 0입니다. EOS(<code>&lt;|im_end|&gt;</code>)는 PAD(<code>&lt;|endoftext|&gt;</code>)와 다른 토큰이라 1입니다.</li>
          <li>MiniMind <code>generate</code>는 먼저 끝난 줄의 남은 칸을 <b>EOS로 채웁니다</b>. 그래서 위치 8도 EOS입니다.</li>
          <li><code>scatter_(1, logp_pos+1, completion_mask)</code>는 답 구간(위치 5~8)의 mask를 rollout이 준 값으로 덮어씁니다. Torch 엔진은 전부 1, SGLang 엔진은 오른쪽 패딩 칸을 0으로 줍니다.</li>
        </ul>
      </div>

      <div class="step">
        <div class="ttl"><span class="n">②</span>logits[t]는 토큰 t+1을 예측한다 (한 칸 밀기)</div>
        <div class="scroll"><table class="strip">
          <tr><th>위치 t</th><td class="idx">0</td><td class="idx">1</td><td class="idx">2</td><td class="idx">3</td><td class="idx">4</td><td class="idx">5</td><td class="idx">6</td><td class="idx">7</td><td class="idx">8</td></tr>
          <tr><th>입력 토큰</th><td class="t-pad">PAD</td><td class="t-pad">PAD</td><td class="t-p">p₁</td><td class="t-p">p₂</td><td class="t-p">p₃</td><td class="t-a">a₁</td><td class="t-a">a₂</td><td class="t-e">EOS</td><td class="t-e dim">EOS</td></tr>
          <tr><th>logits[:, :-1]</th><td class="b0">ℓ₀</td><td class="b0">ℓ₁</td><td class="b0">ℓ₂</td><td class="b0">ℓ₃</td><td class="b0">ℓ₄</td><td class="b0">ℓ₅</td><td class="b0">ℓ₆</td><td class="b0">ℓ₇</td><td class="blank lbl">버림</td></tr>
          <tr><th></th><td class="arrow">↓</td><td class="arrow">↓</td><td class="arrow">↓</td><td class="arrow">↓</td><td class="arrow">↓</td><td class="arrow">↓</td><td class="arrow">↓</td><td class="arrow">↓</td><td class="blank"></td></tr>
          <tr><th>outputs[:, 1:]</th><td class="t-pad">PAD</td><td class="t-p">p₁</td><td class="t-p">p₂</td><td class="t-p">p₃</td><td class="t-a">a₁</td><td class="t-a">a₂</td><td class="t-e">EOS</td><td class="t-e">EOS</td><td class="blank"></td></tr>
          <tr><th>log p (전체)</th><td class="b0 dim">·</td><td class="b0 dim">·</td><td class="b0 dim">·</td><td class="b0 dim">·</td><td class="b0">lp(a₁)</td><td class="b0">lp(a₂)</td><td class="b0">lp(E)</td><td class="b0">lp(E)</td><td class="blank"></td></tr>
        </table></div>
        <pre class="formula">F.log_softmax(logits[:, :-1], -1)        # [B*G, L-1, V]  위치 t의 다음 토큰 분포
      .gather(2, outputs[:, 1:].unsqueeze(-1))  # 실제로 나온 다음 토큰의 log p만 꺼냄
      .squeeze(-1)                             # [B*G, L-1]</pre>
        <ul>
          <li>마지막 위치(8)의 logits는 예측할 다음 토큰이 없어 버립니다.</li>
          <li>이 단계까지는 프롬프트 토큰의 log p도 섞여 있습니다. 필요한 건 위치 4~7, 즉 <b>답 토큰을 예측한 칸</b>뿐입니다.</li>
        </ul>
      </div>

      <div class="step">
        <div class="ttl"><span class="n">③</span>logp_pos로 답 구간만 고르기</div>
        <pre class="formula">logp_pos = prompt_lens - 1 + arange(R)   # 5 - 1 + [0,1,2,3] = [4, 5, 6, 7]</pre>
        <div class="scroll"><table class="strip">
          <tr><th>위치 t</th><td class="idx">0</td><td class="idx">1</td><td class="idx">2</td><td class="idx">3</td><td class="idx">4</td><td class="idx">5</td><td class="idx">6</td><td class="idx">7</td></tr>
          <tr><th>log p (전체)</th><td class="b0 dim">·</td><td class="b0 dim">·</td><td class="b0 dim">·</td><td class="b0 dim">·</td><td class="b0 pick">lp(a₁)</td><td class="b0 pick">lp(a₂)</td><td class="b0 pick">lp(E)</td><td class="b0 pick">lp(E)</td></tr>
          <tr><th></th><td class="blank" colspan="4"></td><td class="arrow">↓</td><td class="arrow">↓</td><td class="arrow">↓</td><td class="arrow">↓</td></tr>
          <tr><th>per_token_logps</th><td class="blank lbl" colspan="4" style="text-align:right">.gather(1, logp_pos) →</td><td class="t-a">lp(a₁)</td><td class="t-a">lp(a₂)</td><td class="t-e">lp(E)</td><td class="t-e">lp(E)</td></tr>
          <tr><th>답 인덱스 r</th><td class="blank" colspan="4"></td><td class="idx">0</td><td class="idx">1</td><td class="idx">2</td><td class="idx">3</td></tr>
        </table></div>
        <ul>
          <li>결과 shape은 <code>[B*G, R]</code>입니다. 열 r은 "답의 r번째 토큰이 나올 log 확률"입니다.</li>
          <li>①의 <code>logp_pos+1</code>(위치 5~8)은 답 토큰 자체의 위치입니다. 예측하는 칸(t)과 예측되는 토큰(t+1)이 한 칸 차이라서 그렇습니다.</li>
        </ul>
      </div>

      <div class="step">
        <div class="ttl"><span class="n">④</span>첫 EOS까지만 학습: completion_mask</div>
        <div class="scroll"><table class="strip">
          <tr><th>답 인덱스 r</th><td class="idx">0</td><td class="idx">1</td><td class="idx">2</td><td class="idx">3</td></tr>
          <tr><th>completion_ids</th><td class="t-a">a₁</td><td class="t-a">a₂</td><td class="t-e">EOS</td><td class="t-e">EOS</td></tr>
          <tr><th>is_eos</th><td class="b0">0</td><td class="b0">0</td><td class="b1">1</td><td class="b1">1</td></tr>
          <tr><th>eos_idx</th><td class="blank"></td><td class="blank"></td><td class="pick b1">2</td><td class="blank lbl" style="text-align:left">← argmax = 첫 1</td></tr>
          <tr><th>r ≤ eos_idx</th><td class="b1">1</td><td class="b1">1</td><td class="b1">1</td><td class="b0">0</td></tr>
          <tr><th>completion_mask</th><td class="t-a">학습</td><td class="t-a">학습</td><td class="t-e">학습</td><td class="t-off">제외</td></tr>
        </table></div>
        <p style="color:var(--muted);font-size:.88rem">EOS가 한 번도 없으면(최대 길이까지 생성) <code>eos_idx = R-1</code> 기본값이 남아 답 전체가 학습 대상입니다.</p>
        <div class="scroll"><table class="strip">
          <tr><th>EOS 없음</th><td class="t-a">a₁</td><td class="t-a">a₂</td><td class="t-a">a₃</td><td class="t-a">a₄</td></tr>
          <tr><th>completion_mask</th><td class="b1">1</td><td class="b1">1</td><td class="b1">1</td><td class="b1">1</td></tr>
        </table></div>
        <ul>
          <li><b>첫 EOS는 포함</b>합니다. "여기서 멈춘다"는 결정도 정책의 행동이라 보상을 받아야 합니다.</li>
          <li><b>그 뒤 EOS는 제외</b>합니다. generate가 채운 값일 뿐 모델이 선택한 토큰이 아닙니다.</li>
          <li><code>&amp; completion_pad_mask</code>는 SGLang 경로의 오른쪽 패딩 칸을 한 번 더 0으로 만듭니다.</li>
        </ul>
      </div>

      <div class="step">
        <div class="ttl"><span class="n">⑤</span>같은 계산, 세 번: rollout · policy · reference</div>
        <div class="prec">
          <div class="box pol"><b>policy (학습 중 model)</b><span class="tag">with autocast_ctx → bf16</span><span>per_token_logps</span><span class="tag">grad 있음</span></div>
          <div class="mid"><span>같은 outputs</span><b>⇄</b><span>같은 logp_pos</span></div>
          <div class="box ref"><b>reference (ref_model)</b><span class="tag">torch.no_grad(), autocast 밖 → fp32</span><span>ref_per_token_logps</span><span class="tag">grad 없음</span></div>
        </div>
        <div class="cmp">
          <div class="ok"><b>ratio = exp(policy − rollout) = 1</b>첫 step에는 rollout도 같은 가중치·같은 bf16 autocast로 계산했습니다. 그래서 값이 그대로 일치합니다.</div>
          <div class="warn"><b>KL = ref − policy ≈ 0, 하지만 ≠ 0</b>가중치는 같아도 reference는 fp32, policy는 bf16입니다. 정밀도 차이만큼 log p가 달라 KL에 작은 잡음이 남습니다.</div>
        </div>
      </div>
    </div>
    """
    mo.Html(_css + _body)
    return


@app.cell
def _(F, autocast_ctx, demo_rollout, device, sft_model, tokenizer, torch):
    outputs = demo_rollout.output_ids
    completion_ids = demo_rollout.completion_ids
    prompt_lens = demo_rollout.prompt_lens.to(device)
    full_mask = (outputs != tokenizer.pad_token_id).long()
    logp_pos = prompt_lens.unsqueeze(1) - 1 + torch.arange(completion_ids.size(1), device=device).unsqueeze(0)
    full_mask.scatter_(1, logp_pos + 1, demo_rollout.completion_mask.to(device, dtype=full_mask.dtype))

    with torch.no_grad():  # 학습 첫 step: policy와 reference가 모두 SFT 모델
        with autocast_ctx:   # policy: 원본처럼 bf16 autocast 안에서
            _logits = sft_model(outputs, attention_mask=full_mask).logits
            demo_logps = F.log_softmax(_logits[:, :-1, :], dim=-1).gather(2, outputs[:, 1:].unsqueeze(-1)).squeeze(-1).gather(1, logp_pos)
        # reference: 원본처럼 autocast 밖(fp32)에서
        demo_ref_logps = F.log_softmax(sft_model(outputs, attention_mask=full_mask).logits[:, :-1, :], dim=-1).gather(2, outputs[:, 1:].unsqueeze(-1)).squeeze(-1).gather(1, logp_pos)

    completion_pad_mask = demo_rollout.completion_mask.to(device).bool()
    is_eos = (completion_ids == tokenizer.eos_token_id) & completion_pad_mask
    eos_idx = torch.full((is_eos.size(0),), is_eos.size(1) - 1, dtype=torch.long, device=device)
    eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
    demo_mask = ((torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1) <= eos_idx.unsqueeze(1))
                 & completion_pad_mask).int()

    demo_old_logps = demo_rollout.per_token_logps.to(device)
    max_ratio_gap = (torch.exp(demo_logps - demo_old_logps) - 1).abs().max().item()
    assert max_ratio_gap < 1e-3          # 첫 step: ratio = π / π_old = 1
    assert torch.equal(demo_mask.sum(1), eos_idx + 1)
    # 원본이 로그로 남기는 kl_ref와 같은 식
    demo_kl_ref = (((demo_ref_logps - demo_logps) * demo_mask).sum() / demo_mask.sum()).item()
    demo_max_logp_gap = ((demo_ref_logps - demo_logps).abs() * demo_mask).max().item()
    return (
        demo_kl_ref,
        demo_logps,
        demo_mask,
        demo_max_logp_gap,
        demo_old_logps,
        demo_ref_logps,
        eos_idx,
        max_ratio_gap,
    )


@app.cell(hide_code=True)
def _(
    demo_kl_ref,
    demo_mask,
    demo_max_logp_gap,
    demo_old_logps,
    demo_rollout,
    eos_idx,
    max_ratio_gap,
    mo,
    result_table,
    tokenizer,
):
    _row = int(eos_idx.argmin())            # 가장 일찍 끝난 답: EOS 뒤 채움이 가장 깁니다
    _e = int(eos_idx[_row])
    _ids = demo_rollout.completion_ids[_row]
    _show = list(range(max(0, _e - 6), min(len(_ids), _e + 5)))
    mo.vstack([
        mo.md(f"답 {_row}번의 끝부분 (첫 EOS 위치 = {_e}, 답 길이 R = {len(_ids)}):"),
        result_table([{"답 안 위치": t, "토큰": tokenizer.convert_ids_to_tokens(int(_ids[t])).replace("Ċ", "\\n"),
                      "completion_mask": int(demo_mask[_row, t]),
                      "log p (rollout 때)": round(demo_old_logps[_row, t].item(), 3)} for t in _show], selection=None),
        mo.md(f"""
    - 답별 유효 토큰 수(첫 EOS 포함): `{demo_mask.sum(1).tolist()}`
    - 다시 계산한 로그 확률과 rollout 때 값의 비율: $\\max\\left|\\pi/\\pi_{{old}}-1\\right|$ = **{max_ratio_gap:.1e}** (bf16끼리 비교)
    - reference(fp32)와 policy(bf16)의 로그 확률 차이: 토큰 최대 **{demo_max_logp_gap:.3f}**,
      원본 로그의 `kl_ref`(평균) = **{demo_kl_ref:.1e}**. 같은 가중치인데도 정밀도 때문에 생기는 차이입니다.
      학습이 진행되며 생기는 실제 변화(8절 그래프)보다는 작지만, KL 벌점이 첫 step부터 이 잡음에도 작게 반응한다는 뜻입니다.
    """),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7. 손실: GRPO와 CISPO

    ```python
    kl_div = ref_per_token_logps - per_token_logps
    per_token_kl = torch.exp(kl_div) - kl_div - 1                      # reference와의 거리 (항상 ≥ 0)
    ratio = torch.exp(per_token_logps - old_per_token_logps)           # π / π_old
    if args.loss_type == "cispo":                                      # 기본값
        clamped_ratio = torch.clamp(ratio, max=args.epsilon_high).detach()
        per_token_loss = -(clamped_ratio * advantages.unsqueeze(1) * per_token_logps - args.beta * per_token_kl)
    else:                                                              # "grpo": PPO식 clip
        clipped_ratio = torch.clamp(ratio, 1 - args.epsilon, 1 + args.epsilon)
        per_token_loss = -(torch.min(ratio * A, clipped_ratio * A) - args.beta * per_token_kl)
    policy_loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1).clamp(min=1)).mean()
    ```

    **토큰마다 같은 advantage.** 답 하나의 advantage가 그 답의 모든 토큰에 똑같이 곱해집니다. 맞힌 답은 토큰 전체의 확률이 오르고,
    틀린 답은 전체가 내려갑니다. 어느 토큰이 "잘못"이었는지는 모릅니다(그래서 많은 샘플이 필요합니다).

    **답마다 평균 → 배치 평균.** 긴 답과 짧은 답이 같은 무게를 가집니다.

    **KL 항(`beta`, 기본값 0.1)** 은 DPO의 reference 역할과 같습니다. $e^x-x-1$은 x = 0일 때 0이고 양쪽으로 커지는
    KL 추정식(k3)입니다. policy가 reference에서 멀어질수록 벌점이 커집니다.

    **GRPO와 CISPO의 차이는 `ratio`가 1에서 벗어났을 때만** 생깁니다.

    - **GRPO (PPO clip):** ratio가 $[1-\epsilon,1+\epsilon]$ 밖으로 advantage 방향으로 나가면 그 토큰의 기울기를 **0으로 만듭니다**(더 밀지 않음).
    - **CISPO:** ratio를 상수(`detach`)로 보고 `epsilon_high`(5.0)에서 자르기만 합니다. **PPO clip처럼 보상 항의 기울기를 제거하지는 않습니다.**
      PPO clip은 드물지만 중요한 토큰(갑자기 확률이 크게 바뀐 토큰)의 기울기를 버리기 쉬운데, CISPO는 ratio 배율을 제한하며 clipping 때문에 해당 보상 항을 제거하지 않습니다.

    그런데 `train_grpo.py`는 **rollout 한 번에 업데이트를 한 번만** 합니다. 업데이트 전에 계산하므로 항상 $\mathrm{ratio}=1$이고
    (6절에서 확인), 이때 두 손실의 **기울기는 정확히 같습니다.** 아래에서 직접 계산합니다.
    """)
    return


@app.cell
def _(
    demo_advantages,
    demo_logps,
    demo_mask,
    demo_old_logps,
    demo_ref_logps,
    torch,
):
    def policy_loss_fn(per_token_logps, old_per_token_logps, ref_per_token_logps, advantages, completion_mask,
                       loss_type, beta=0.1, epsilon=0.2, epsilon_high=5.0):
        """train_grpo.py의 손실 부분을 그대로 옮긴 것"""
        kl_div = ref_per_token_logps - per_token_logps
        per_token_kl = torch.exp(kl_div) - kl_div - 1
        ratio = torch.exp(per_token_logps - old_per_token_logps)
        if loss_type == "cispo":
            clamped_ratio = torch.clamp(ratio, max=epsilon_high).detach()
            per_token_loss = -(clamped_ratio * advantages.unsqueeze(1) * per_token_logps - beta * per_token_kl)
        else:
            clipped_ratio = torch.clamp(ratio, 1 - epsilon, 1 + epsilon)
            per_token_loss1 = ratio * advantages.unsqueeze(1)
            per_token_loss2 = clipped_ratio * advantages.unsqueeze(1)
            per_token_loss = -(torch.min(per_token_loss1, per_token_loss2) - beta * per_token_kl)
        return ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1).clamp(min=1)).mean()

    # 첫 step: policy = old (ratio = 1). 로그 확률을 잎(leaf) 텐서로 만들어 두 손실의 기울기를 비교합니다.
    demo_loss, demo_grad = {}, {}
    for _kind in ("cispo", "grpo"):
        _lp = demo_logps.detach().clone().float().requires_grad_(True)
        _loss = policy_loss_fn(_lp, demo_old_logps.float(), demo_ref_logps.float(), demo_advantages, demo_mask, _kind)
        _loss.backward()
        demo_loss[_kind], demo_grad[_kind] = _loss.item(), _lp.grad
    assert torch.allclose(demo_grad["cispo"], demo_grad["grpo"], atol=1e-6)
    assert (demo_grad["cispo"][demo_mask == 0] == 0).all()      # EOS 뒤 토큰은 기울기 0
    return demo_grad, demo_loss, policy_loss_fn


@app.cell(hide_code=True)
def _(
    DEMO_G,
    demo_advantages,
    demo_grad,
    demo_loss,
    demo_mask,
    mo,
    plt,
    policy_loss_fn,
    result_table,
    torch,
):
    # ratio가 1이 아닐 때 토큰 하나의 기울기 크기: −∂loss/∂log π  (beta=0)
    _ratios = torch.linspace(0.3, 6.5, 300)
    _fig, _axes = plt.subplots(1, 2, figsize=(10, 3))
    for _ax, _A in zip(_axes, (1.0, -1.0)):
        for _kind, _style in (("grpo", "-"), ("cispo", "--")):
            _g = []
            for _r in _ratios:
                _lp = torch.zeros(1, 1, requires_grad=True)
                policy_loss_fn(_lp, -torch.log(_r).view(1, 1), torch.zeros(1, 1), torch.tensor([_A]),
                               torch.ones(1, 1), _kind, beta=0.0).backward()
                _g.append(-_lp.grad.item())
            _ax.plot(_ratios, _g, _style, label=_kind)
        _ax.axvline(1, color="gray", lw=0.5)
        _ax.set_title(f"advantage = {_A:+.0f}")
        _ax.set_xlabel("ratio = pi / pi_old")
        _ax.set_ylabel("-d loss / d log pi")
        _ax.legend(fontsize=8)
    _fig.tight_layout()
    _plot = mo.as_html(_fig)
    plt.close(_fig)

    _per_answer = (demo_grad["cispo"].sum(1) / demo_mask.sum(1)).tolist()
    mo.vstack([
        mo.md(f"4절 rollout에서 손실 **값**은 CISPO `{demo_loss['cispo']:.4f}`, GRPO `{demo_loss['grpo']:.6f}`로 다릅니다. "
              "GRPO는 $-rA$ ($r$은 ratio)(ratio = 1이라 advantage 평균 ≈ 0)이고 CISPO는 $-A\\log\\pi$이기 때문입니다. "
              "하지만 답 토큰별 **기울기는 같습니다**(assert 통과). 학습을 움직이는 것은 손실 값이 아니라 기울기입니다."),
        result_table([{"답": f"질문 {i // DEMO_G} / 답 {i % DEMO_G}", "advantage": round(demo_advantages[i].item(), 3),
                      "토큰당 −∂loss/∂log p (평균)": round(-_per_answer[i], 4)} for i in range(len(_per_answer))],
                    selection=None, page_size=12),
        mo.md("표는 손실 기울기의 음수, 즉 로그 확률의 갱신 방향을 표시합니다. 보상 항은 advantage가 양수면 로그 확률을 올리고 음수면 내립니다. "
              "advantage가 0인 그룹은 보상 항의 기울기가 0입니다. KL 항의 기울기는 policy와 reference가 다르면 남을 수 있습니다.\n\n"
              "**ratio가 1이 아니라면** (rollout 하나로 여러 번 업데이트하거나, SGLang처럼 다른 엔진이 생성할 때):"),
        _plot,
        mo.md("GRPO(실선)는 advantage가 양수일 때 ratio > 1.2, 음수일 때 ratio < 0.8이면 보상 항의 기울기가 **0**이 됩니다(KL 항은 별도). "
              "CISPO(점선)는 ratio에 비례하다가 5에서 멈출 뿐 0이 되지 않습니다."),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 8. GRPO 학습

    ### 8-1. 원본 `grpo_train_epoch`를 그대로 실행하기

    `grpo_train_epoch`는 4~7절의 내용에 역전파·옵티마이저·로그·저장까지 한 함수에 담고 있습니다.
    이 함수는 `args`, `model`, `tokenizer`, `optimizer` 같은 값을 **모듈 전역 변수**로 읽습니다(`if __name__ == "__main__":` 아래에서 만듦).
    그래서 아래 `run_grpo`는 main의 5~8번 단계를 옮겨 이 전역 변수를 채운 뒤, **원본 함수를 그대로 호출합니다.**

    - `wandb` 자리에 `log` 메서드만 있는 **기록기**를 넘겨, 원본이 매 step 남기는 지표(보상, KL, 응답 길이 등)를 받아 그래프를 그립니다.
    - 재개용 체크포인트 저장(`lm_checkpoint`, 경로 `../checkpoints`)만 끄고, 가중치(`grpo_768.pth`)는 임시 폴더에 저장되게 둡니다.
    - 원본 로그(`Logger` 출력)는 따로 모아 마지막 몇 줄만 보여 줍니다.
    """)
    return


@app.cell
def _(
    CosineAnnealingLR,
    DataLoader,
    MAX_GEN_LEN,
    MAX_SEQ_LEN,
    Path,
    SimpleNamespace,
    SkipBatchSampler,
    TorchRolloutEngine,
    autocast_ctx,
    copy,
    device,
    io,
    math,
    redirect_stdout,
    setup_seed,
    sft_model,
    tempfile,
    tokenizer,
    torch,
    train_ds,
    train_grpo,
):
    ckpt_dir = Path(tempfile.mkdtemp(prefix="minimind_grpo_out_"))

    def run_grpo(cfg, history, batches=None):
        """train_grpo.py main의 '5. 모델 및 데이터 초기화' ~ '8. 학습 시작'. 한 step의 학습은 원본 grpo_train_epoch가 합니다."""
        args = SimpleNamespace(
            device=device, dtype="bfloat16", epochs=cfg["epochs"], batch_size=cfg["batch_size"],
            learning_rate=cfg["learning_rate"], accumulation_steps=1, grad_clip=1.0,
            log_interval=1, save_interval=10**9, save_dir=str(ckpt_dir), save_weight="grpo",
            max_seq_len=MAX_SEQ_LEN, max_gen_len=MAX_GEN_LEN, num_generations=cfg["num_generations"],
            beta=cfg["beta"], loss_type=cfg["loss_type"], epsilon=0.2, epsilon_high=5.0,
            reward_type="gsm8k", debug_mode=False, debug_interval=20,
        )
        model = copy.deepcopy(sft_model)                                        # policy
        ref_model = copy.deepcopy(sft_model).eval().requires_grad_(False)       # reference
        rollout_engine = TorchRolloutEngine(model, tokenizer, device, autocast_ctx)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
        iters = len(batches) if batches is not None else math.ceil(len(train_ds) / args.batch_size)
        scheduler = CosineAnnealingLR(optimizer, T_max=iters * args.epochs, eta_min=args.learning_rate / 10)

        # 원본이 모듈 전역 변수로 읽는 값들
        vars(train_grpo).update(args=args, model=model, tokenizer=tokenizer, lm_config=sft_model.config,
                                autocast_ctx=autocast_ctx, optimizer=optimizer, scheduler=scheduler)
        original_lm_checkpoint = train_grpo.lm_checkpoint
        train_grpo.lm_checkpoint = lambda *a, **k: None       # 재개용 체크포인트 저장만 끔
        recorder = SimpleNamespace(log=history.append)         # wandb 대신 지표 기록
        log = io.StringIO()
        try:
            with redirect_stdout(log):
                for epoch in range(args.epochs):
                    if batches is None:   # 원본과 같은 섞기: setup_seed(42 + epoch) 후 randperm
                        setup_seed(42 + epoch); indices = torch.randperm(len(train_ds)).tolist()
                        loader = DataLoader(train_ds, batch_sampler=SkipBatchSampler(indices, args.batch_size, 0))
                    else:
                        loader = batches
                    train_grpo.grpo_train_epoch(epoch, loader, len(loader), rollout_engine, ref_model, None, 0, recorder)
        finally:
            train_grpo.lm_checkpoint = original_lm_checkpoint
        return model.eval(), log.getvalue()

    return ckpt_dir, run_grpo


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 8-2. 단계별 계산이 원본과 같은지 확인

    4절과 같은 시드, 같은 질문 2개로 원본 함수를 **1 step만** 실행합니다. 같은 답이 생성되고, 원본이 기록한 보상·손실이
    4~7절에서 직접 계산한 값과 같아야 합니다.
    """)
    return


@app.cell
def _(
    DEMO_G,
    DEMO_SEED,
    demo_advantages,
    demo_batch,
    demo_kl_ref,
    demo_loss,
    demo_rewards,
    run_grpo,
    setup_seed,
):
    parity_history = []
    setup_seed(DEMO_SEED)
    _, parity_log = run_grpo({"epochs": 1, "batch_size": 2, "learning_rate": 3e-7, "num_generations": DEMO_G,
                              "beta": 0.1, "loss_type": "cispo"}, parity_history, batches=[demo_batch])
    parity = parity_history[0]
    assert abs(parity["reward"] - demo_rewards.mean().item()) < 1e-6
    assert abs(parity["advantages_std"] - demo_advantages.std().item()) < 1e-4
    assert abs(parity["policy_loss"] - demo_loss["cispo"]) < 1e-4
    assert abs(parity["kl_ref"] - demo_kl_ref) < 1e-6
    return parity, parity_log


@app.cell(hide_code=True)
def _(
    demo_advantages,
    demo_kl_ref,
    demo_loss,
    demo_rewards,
    mo,
    parity,
    parity_log,
):
    mo.md(f"""
    | | 4~7절 직접 계산 | 원본 `grpo_train_epoch` |
    | --- | --- | --- |
    | 평균 보상 | {demo_rewards.mean().item():.4f} | {parity['reward']:.4f} |
    | advantage 표준편차 | {demo_advantages.std().item():.4f} | {parity['advantages_std']:.4f} |
    | 손실 (CISPO) | {demo_loss['cispo']:.6f} | {parity['policy_loss']:.6f} |
    | kl_ref | {demo_kl_ref:.2e} | {parity['kl_ref']:.2e} |

    원본 로그:
    ```text
    {parity_log.strip()}
    ```
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 8-3. 실행해 보기

    기본 설정은 **원본 기본값 그대로**입니다: `--learning_rate 3e-7`, `--loss_type cispo`, `--beta 0.1`, `--num_generations 6`.
    노트북에서 바꾼 것은 데이터(128문제), 질문 배치 4개(원본 2), 답 최대 길이 96토큰(원본 1024), 2 epochs(원본 1)입니다.
    128문제 / 배치 4 = 32 step, 2 epochs = **64 step**입니다. step마다 답 24개(4 × 6)를 생성합니다.

    설정을 바꾸면 SFT 모델에서 다시 학습하고 평가합니다(RTX 3080 Ti 기준 학습 약 1분, 평가 약 10초).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    grpo_lr = mo.ui.dropdown(options={"3e-8": 3e-8, "3e-7 (원본)": 3e-7, "3e-6": 3e-6}, value="3e-7 (원본)", label="learning_rate")
    grpo_g = mo.ui.dropdown(options={"2": 2, "6 (원본)": 6}, value="6 (원본)", label="num_generations")
    grpo_beta = mo.ui.dropdown(options={"0.0": 0.0, "0.1 (원본)": 0.1, "1.0": 1.0}, value="0.1 (원본)", label="beta")
    grpo_loss = mo.ui.dropdown(options={"cispo (원본)": "cispo", "grpo": "grpo"}, value="cispo (원본)", label="loss_type")
    mo.hstack([grpo_lr, grpo_g, grpo_beta, grpo_loss], justify="start", gap=2)
    return grpo_beta, grpo_g, grpo_loss, grpo_lr


@app.cell
def _(evaluate, grpo_beta, grpo_g, grpo_loss, grpo_lr, run_grpo):
    grpo_cfg = {"epochs": 2, "batch_size": 4, "learning_rate": grpo_lr.value, "num_generations": grpo_g.value,
                "beta": grpo_beta.value, "loss_type": grpo_loss.value}
    grpo_history = []
    grpo_model, grpo_log = run_grpo(grpo_cfg, grpo_history)
    grpo_eval = evaluate(grpo_model)
    return grpo_cfg, grpo_eval, grpo_history, grpo_log


@app.cell(hide_code=True)
def _(
    ckpt_dir,
    grpo_cfg,
    grpo_eval,
    grpo_history,
    grpo_log,
    mo,
    plt,
    result_table,
    sft_eval,
    test_ds,
):
    def _smooth(values, k=8):
        return [sum(values[max(0, i - k + 1): i + 1]) / len(values[max(0, i - k + 1): i + 1]) for i in range(len(values))]

    _fig, _axes = plt.subplots(1, 4, figsize=(14, 3))
    for _ax, (_key, _title) in zip(_axes, [("reward", "reward (correct rate)"), ("reward/mixed_group_rate", "mixed group rate"),
                                           ("avg_response_len", "avg response length"), ("kl_ref", "kl_ref")]):
        _v = [h[_key] for h in grpo_history]
        _ax.plot(_v, alpha=0.3)
        _ax.plot(_smooth(_v), label="moving avg (8)")
        _ax.set_title(_title)
        _ax.set_xlabel("step")
    _axes[0].legend(fontsize=8)
    _fig.tight_layout()
    _plot = mo.as_html(_fig)
    plt.close(_fig)

    _keys = ["greedy 정답률", "샘플링 정답률", "덧셈 (greedy)", "곱셈 (greedy)", "형식 오류", "정답 숫자를 쓰고도 최종 답은 틀림"]
    _table = ("| | " + " | ".join(_keys) + " | 평균 답 길이 |\n|---|" + "---|" * (len(_keys) + 1) + "\n"
              + "\n".join(f"| {name} | " + " | ".join(f"{e[k]:.1%}" for k in _keys) + f" | {e['평균 답 길이(토큰)']:.1f} |"
                          for name, e in [("SFT (GRPO 전)", sft_eval), ("GRPO 후", grpo_eval)]))
    _pth = sorted(p.name for p in ckpt_dir.glob("*.pth"))
    mo.vstack([
        mo.md(f"설정: lr={grpo_cfg['learning_rate']}, G={grpo_cfg['num_generations']}, β={grpo_cfg['beta']}, "
              f"loss_type={grpo_cfg['loss_type']}, {len(grpo_history)} step. 원본이 저장한 가중치: `{_pth}`"),
        _plot,
        mo.md("**학습에 없던 평가 256문제**에서 측정 (greedy, 최대 96토큰. 샘플링은 문제당 4회)\n\n" + _table),
        result_table([{"질문": s["prompt"].split("user\n")[1].split("\nSolve")[0], "정답": s["answer"].split("#### ")[1],
                      "SFT 답": a["text"], "GRPO 후 답": b["text"]}
                     for s, a, b in zip(test_ds.samples[:12], sft_eval["greedy"][:12], grpo_eval["greedy"][:12])],
                    selection=None, page_size=6),
        mo.accordion({"원본 로그 (마지막 5 step)": mo.md("```text\n" + "\n".join(grpo_log.strip().splitlines()[-10:]) + "\n```")}),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **읽는 법** (기본 설정을 RTX 3080 Ti에서 실행한 결과 기준. GPU나 설정이 다르면 숫자가 달라집니다.)

    - **학습 보상**(학습 문제에서 샘플링한 답의 정답률)은 0.1 근처에서 0.2~0.3으로 오르지만 step마다 크게 흔들립니다.
      질문 4개 × 답 6개로 잰 값이기 때문입니다. 판단은 학습에 없던 평가 문제로 합니다.
    - **평가 256문제의 greedy 정답률이 약 13% → 38%** 로 올랐습니다. 대부분 **곱셈(20% → 62%)** 에서 생겼고, 덧셈은 6% → 14%에 그쳤습니다.
    - 가장 크게 줄어든 것은 **"정답 숫자를 쓰고도 최종 답은 틀림"(29% → 6%)** 입니다. 맞게 계산해 놓고 **필요 없는 단계를
      덧붙이다 답을 망치는** 행동이 줄었습니다. 보상 함수에는 "덧붙이지 말라"는 규칙이 없습니다. 그런 답이 0점을 받았을 뿐입니다.
    - **계산 자체를 틀리는 답은 남습니다.** 덧셈이 잘 오르지 않은 이유입니다(SFT 모델은 `6+7=9`처럼 덧셈 자체를 자주 틀립니다).
      학습 후 오히려 계산이 틀리게 바뀐 답도 보일 수 있습니다. 이 규칙 보상 실험은 생성한 답을 채점할 뿐 정답 풀이 텍스트를 직접 학습 목표로 제공하지 않습니다.
      이것이 GRPO로 새로운 풀이 전략을 발견할 수 없다는 뜻은 아닙니다. 샘플링에서 한 번도 안 나오는 정답은 보상을 받을 기회가 없습니다.
    - **섞인 그룹 비율**: 보상이 서로 다른 답이 있는 질문의 비율이며, 실제로 학습 신호를 주는 질문의 비율입니다(여기서는 평균 0.5 정도).
      나머지 질문은 KL 항 말고는 아무것도 배우지 않습니다.
    - **`kl_ref` 로그 값은 음수가 될 수 있습니다.** 원본은 $\log\pi_{\mathrm{ref}}-\log\pi_\theta$의 평균을 기록합니다. policy가 자신이 생성한 답의
      확률을 reference보다 높게 보면 음수입니다. 손실에 쓰는 $e^x-x-1$은 항상 0 이상입니다.
    - **학습률이 이렇게 작은데(3e-7) 왜 바뀔까?** Adam의 모멘트 정규화는 일정한 기울기 스케일 변화의 영향을 일부 상쇄합니다.
      다만 실제 이동은 원소별 모멘트·epsilon과 AdamW의 weight decay에 따라 다릅니다.
      많은 매개변수의 작은 갱신이 누적되면 출력은 바뀔 수 있습니다. 게다가 "여기서 끝낼까, 더 쓸까"는 확률이 비슷한 갈림길이라
      작은 변화로도 greedy 답이 바뀝니다. 이 노트북 실험에서는 학습률을 10분의 1(3e-8)로 낮춰도 비슷하게 좋아졌습니다.


    AdamW의 원소별 갱신은 $\Delta\theta=-\eta\hat m/(\sqrt{\hat v}+\epsilon)-\eta\lambda\theta$입니다.
    $\eta$는 학습률, $\hat m,\hat v$는 편향 보정한 1·2차 모멘트, $\epsilon$은 분모 안정화 상수,
    $\lambda$는 weight decay 계수입니다. 현재 기울기뿐 아니라 누적 상태와 원래 가중치에도 의존합니다.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 9. 실제 규모로 실행하려면

    ```bash
    cd projects/minimind/trainer
    # 보상 모델 경로 (기본값): RLAIF 대화 데이터 + InternLM2 1.8B 보상 모델
    python train_grpo.py                                   # full_sft_768.pth에서 시작 → ../out/grpo_768.pth
    # 정답 채점 경로: GSM8K 학습 데이터
    python train_grpo.py --reward_type gsm8k --data_path openai/gsm8k --max_gen_len 256
    ```

    | 인자 | 기본값 | 이 노트북 |
    | --- | --- | --- |
    | `--from_weight` | `full_sft` | `caspar/MiniMind-Dense-64M-Mixed-SFT-v1.0` |
    | `--reward_type` | `model` | `gsm8k` |
    | `--learning_rate` | 3e-7 | 3e-7 (위젯으로 변경) |
    | `--loss_type` | `cispo` | `cispo` (위젯으로 변경) |
    | `--beta` | 0.1 | 0.1 (위젯으로 변경) |
    | `--num_generations` | 6 | 6 (위젯으로 변경) |
    | `--batch_size` | 2 (질문 2개 → 답 12개) | 4 (질문 4개 → 답 24개) |
    | `--epochs` | 1 | 2 |
    | `--max_seq_len` / `--max_gen_len` | 768 / 1024 | 256 / 96 |
    | `--dtype` | bfloat16 | bfloat16 autocast |
    | `--rollout_engine` | `torch` | `torch` (`sglang`이면 별도 추론 서버가 생성) |

    **참고 기록:** `projects/minimind/prompts/grpo_training.md`와 `scripts/runpod_gsm8k_grpo/README.md`에는 같은 베이스 모델로
    GSM8K 학습 6,851문제를 1 epoch(215 update, lr 1e-6, G=4) 학습한 기록이 있습니다. 테스트 1,319문제 정답 수가
    규칙 보상으로 14 → 18, Skywork 보상 모델로 14 → 12였습니다. **보상 모델 점수가 올라도 정답률은 떨어질 수 있습니다.**
    이 노트북의 과제처럼 모델이 "가끔은" 맞히는 난이도여야 GRPO가 쓸 신호가 생깁니다.

    ## 10. 정리

    | 항목 | 핵심 |
    | --- | --- |
    | 데이터 | 질문만. 정답은 채점에만 쓰고 프롬프트에 넣지 않음 (`GSM8KDataset`) |
    | rollout | 질문마다 답 G개 샘플링, 생성 때 로그 확률(old policy)도 저장 |
    | 보상 | 규칙 채점 0/1 (`score_gsm8k_response`) 또는 보상 모델 점수 |
    | advantage | 같은 질문의 답끼리 $(r-\mu)/(\sigma+10^{-4})$. critic 모델 없음. 보상이 모두 같으면 0 |
    | 손실 | 토큰마다 $-A\log\pi_\theta$ 꼴 + $\beta\,\widehat{D}_{\mathrm{KL}}$. 답마다 평균 후 배치 평균 |
    | GRPO vs CISPO | ratio ≠ 1일 때만 다름. 원본은 rollout당 업데이트 1번이라 ratio = 1 |
    | 효과 | 모델이 가끔 하던 좋은 행동(여기서는 군더더기 없이 끝내기)의 빈도가 오름. 현재 규칙 보상은 정답 풀이를 직접 제공하지 않으며, 새 전략 학습 불가능을 뜻하지 않음 |
    | 주의 | 학습 보상만 보지 말고 학습에 없던 문제와 실제 답으로 확인 |

    ## 11. 직접 해 볼 과제

    각 과제를 하기 전에 **결과를 먼저 예측**해 적고, 바꿔 실행한 뒤 비교하세요.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({
        "과제 1. num_generations를 2로 (8-3절)": mo.md("""
    G를 2로 줄이면 섞인 그룹 비율과 평가 정답률은 어떻게 될까요?

    *해설:* 정답률이 낮은 질문에서 답 2개가 모두 0점일 가능성이 커져 섞인 그룹이 줄고, 학습 신호가 약해집니다.
    step마다 생성하는 답도 24개에서 8개로 줄어 같은 step 수에서 보는 성공 사례가 적습니다.
    작성자의 실행에서는 섞인 그룹 비율이 0.55 → 0.27, 평가 greedy 정답률이 38% → 22%였습니다.
    """),
        "과제 2. 학습률 3e-8과 3e-6 (8-3절)": mo.md("""
    학습률을 10분의 1, 10배로 바꾸면 어떻게 될까요? 평가 정답률과 함께 **실제 답**을 기본 설정과 비교하세요.

    *해설:* 작성자의 실행에서는 둘 다 정답률이 기본값과 비슷했습니다(3e-8: 41%, 3e-6: 39%). 3e-8에서도 바뀌는 이유는
    8-3절의 "Adam" 설명을 보세요. 3e-6에서는 `"Sam gets 4 more cookies.\\nSam gets 4 more cookies.\\n#### 4"`처럼
    **질문을 되풀이하고 숫자만 쓰는 이상한 답**이 섞였습니다. 정답률 숫자만 보면 놓치는 변화입니다.
    """),
        "과제 3. loss_type을 grpo로 (7절)": mo.md("""
    `cispo`와 `grpo`의 결과는 얼마나 다를까요?

    *해설:* 같은 시드라면 결과가 같습니다(작성자의 실행에서는 모든 평가 지표가 같았습니다). 7절에서 본 것처럼 rollout당
    업데이트가 한 번이라 ratio = 1이고, 이때 두 손실의 기울기가 같기 때문입니다.
    차이를 보려면 한 rollout으로 여러 번 업데이트하도록 코드를 고쳐야 합니다.
    """),
        "과제 4. beta 0과 1.0 (8-3절)": mo.md("""
    β를 0과 1.0으로 바꿔 `kl_ref` 그래프와 평가 결과를 비교하세요.

    *해설:* β = 1.0이면 KL 벌점이 advantage 신호를 압도해 policy가 reference(SFT 모델) 근처에 붙잡힙니다.
    작성자의 실행에서는 `kl_ref`가 거의 0에 머물렀고 정답률은 13% → 19%에 그쳤습니다. β = 0(31%)도 기본값(38%)보다 낮았지만,
    한 번의 실행이라 이 차이만으로 β = 0이 나쁘다고 단정할 수는 없습니다.
    """),
        "과제 5. 보상에 '짧게' 규칙 추가하기 (3절)": mo.md("""
    `calculate_gsm8k_rewards_and_stats` 대신, 정답이면 1점에 답 길이가 40토큰 이하일 때 0.5점을 더 주는 보상을 직접 만들어 보세요.
    정답률은 어떻게 될까요?

    *해설:* 길이 보너스는 정답과 상관없이 받을 수 있으면 "짧지만 틀린 답"도 보상을 받습니다(reward hacking).
    원본 `calculate_rewards`(model 경로)에 길이·반복 규칙이 들어 있는 이유와, 그 규칙이 만드는 부작용을 함께 생각해 보세요.
    """),
        "과제 6. 더 어려운 문제 (2절)": mo.md("""
    `make_problem`에 두 단계 문제(예: "3개 있고 5개 사고 2개 줌")를 추가해 학습해 보세요.

    *해설:* SFT 모델의 정답률이 거의 0이면 섞인 그룹이 거의 없어 학습이 잘 되지 않습니다.
    GRPO는 모델이 가끔이라도 성공하는 난이도에서 가장 잘 작동합니다. 실제 GSM8K 결과가 미미했던 이유입니다.
    """),
    })
    return


if __name__ == "__main__":
    app.run()
