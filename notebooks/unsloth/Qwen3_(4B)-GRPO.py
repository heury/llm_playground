import marimo

__generated_with = "0.25.2"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## marimo 실행
    GPU와 해당 실습의 uv extra를 준비한 뒤 실행 버튼을 누르세요. 코드 셀은 원본 순서대로 실행되며 학습·추론·저장까지 진행합니다. 결과는 `outputs/unsloth/<실행 ID>/`에 저장됩니다. 설정을 바꾼 뒤에는 커널을 재시작해 처음부터 실행하세요.
    """)
    return


@app.cell
def _():
    import marimo as mo
    run_training = mo.ui.run_button(label="GPU 실습 실행")
    run_training
    return mo, run_training


@app.cell
def _(mo, run_training):
    mo.stop(not run_training.value, mo.md("실행 버튼을 누르면 모델·데이터를 불러오고 실습을 시작합니다."))
    from pathlib import Path as _RunPath
    from datetime import datetime as _RunDatetime
    from zoneinfo import ZoneInfo as _RunZoneInfo
    from uuid import uuid4 as _run_uuid4
    _run_id = _RunDatetime.now(_RunZoneInfo("Asia/Seoul")).strftime("%Y%m%d-%H%M%S") + "-" + _run_uuid4().hex[:6] + "-Qwen3_(4B)-GRPO"
    output_dir = _RunPath(__file__).resolve().parents[2] / "outputs" / "unsloth" / _run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"결과 저장 위치: {output_dir}")
    execution_step_0 = 0
    return execution_step_0, output_dir


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Qwen3 (4B) — GRPO
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    목표: OpenR1의 수학 데이터셋과 GRPO를 사용해 `Qwen3-4B-Base`를 추론 모델로 바꿉니다.

    먼저 모델을 사전 미세 조정해 형식을 익히게 합니다. 그러면 GRPO가 형식 학습에 시간을 쓰지 않아도 되므로 학습이 빨라집니다.
    """)
    return


@app.cell
def _(execution_step_0):
    # 원본 코드 셀 순서와 부수 효과를 보존합니다.
    execution_step_1 = execution_step_0 + 1
    import os
    os.environ["UNSLOTH_VLLM_STANDBY"] = "1"

    from unsloth import FastLanguageModel
    import torch
    max_seq_length = 2048 # 더 긴 추론 과정을 생성하려면 늘리세요
    lora_rank = 32 # 순위가 높을수록 표현력이 좋아지지만 느려집니다

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "unsloth/Qwen3-4B-Base",
        max_seq_length = max_seq_length,
        load_in_4bit = False, # 16비트 LoRA를 사용하려면 False
        fast_inference = True, # vLLM 고속 추론 활성화
        max_lora_rank = lora_rank,
        gpu_memory_utilization = 0.9, # 메모리가 부족하면 줄이세요
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r = lora_rank, # 0보다 큰 값을 선택하세요! 권장값: 8, 16, 32, 64, 128
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha = lora_rank*2, # 2배로 설정하면 학습 속도가 빨라집니다
        use_gradient_checkpointing = "unsloth", # 메모리 사용량 감소
        random_state = 3407,
    )
    return execution_step_1, max_seq_length, model, tokenizer, torch


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### GRPO 채팅 템플릿
    베이스 모델을 사용하므로 채팅 템플릿을 설정해야 합니다. 직접 템플릿을 만들어도 됩니다!
    1. DeepSeek는 `<think>`와 `</think>`를 사용하지만 **필수는 아닙니다**. 원하는 대로 바꿀 수 있습니다!
    2. 모델 응답의 방향을 안내할 수 있도록 `system_prompt`를 사용하는 것을 권장합니다.
    """)
    return


@app.cell
def _(execution_step_1):
    # 원본 코드 셀 순서와 부수 효과를 보존합니다.
    execution_step_2 = execution_step_1 + 1
    reasoning_start = "<start_working_out>" # 사고 과정의 시작 태그 역할
    reasoning_end   = "<end_working_out>"   # 사고 과정의 종료 태그 역할
    solution_start  = "<SOLUTION>"
    solution_end    = "</SOLUTION>"

    system_prompt = \
    f"""You are given a problem.
    Think about the problem and provide your working out.
    Place it between {reasoning_start} and {reasoning_end}.
    Then, provide your solution between {solution_start}{solution_end}"""
    system_prompt
    return (
        execution_step_2,
        reasoning_end,
        reasoning_start,
        solution_end,
        solution_start,
        system_prompt,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    아래에서 간단한 채팅 템플릿을 만듭니다. `add_generation_prompt`가 `<start_working_out>`을 앞에 붙여 모델이 추론 과정을 시작하도록 유도한다는 점에 주목하세요.
    """)
    return


@app.cell
def _(execution_step_2, reasoning_start, system_prompt, tokenizer):
    # 원본 코드 셀 순서와 부수 효과를 보존합니다.
    execution_step_3 = execution_step_2 + 1
    chat_template = \
        "{% if messages[0]['role'] == 'system' %}"\
            "{{ messages[0]['content'] + eos_token }}"\
            "{% set loop_messages = messages[1:] %}"\
        "{% else %}"\
            "{{ '{system_prompt}' + eos_token }}"\
            "{% set loop_messages = messages %}"\
        "{% endif %}"\
        "{% for message in loop_messages %}"\
            "{% if message['role'] == 'user' %}"\
                "{{ message['content'] }}"\
            "{% elif message['role'] == 'assistant' %}"\
                "{{ message['content'] + eos_token }}"\
            "{% endif %}"\
        "{% endfor %}"\
        "{% if add_generation_prompt %}{{ '{reasoning_start}' }}"\
        "{% endif %}"

    # 사용자 정의 템플릿으로 교체:
    chat_template = chat_template\
        .replace("'{system_prompt}'",   f"'{system_prompt}'")\
        .replace("'{reasoning_start}'", f"'{reasoning_start}'")
    tokenizer.chat_template = chat_template
    return (execution_step_3,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    예제에서 채팅 템플릿이 어떻게 동작하는지 확인해 보겠습니다.
    """)
    return


@app.cell
def _(
    execution_step_3,
    reasoning_end,
    reasoning_start,
    solution_end,
    solution_start,
    tokenizer,
):
    # 원본 코드 셀 순서와 부수 효과를 보존합니다.
    execution_step_4 = execution_step_3 + 1
    tokenizer.apply_chat_template([
        {"role" : "user", "content" : "What is 1+1?"},
        {"role" : "assistant", "content" : f"{reasoning_start}I think it's 2.{reasoning_end}{solution_start}2{solution_end}"},
        {"role" : "user", "content" : "What is 2+2?"},
    ], tokenize = False, add_generation_prompt = True)
    return (execution_step_4,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 형식 학습을 위한 사전 미세 조정
    고품질 DeepSeek R1 추론 과정만 남기도록 필터링한 NVIDIA의 [Open Math Reasoning 데이터셋](https://huggingface.co/datasets/nvidia/OpenMathReasoning) 일부를 사용합니다.

    약 59개의 예제만 추려 먼저 모델을 준비시키는 사전 미세 조정을 수행합니다. 이를 통해 사용자 정의 GRPO 형식을 이해하도록 합니다.
    """)
    return


@app.cell
def _(execution_step_4):
    # 원본 코드 셀 순서와 부수 효과를 보존합니다.
    execution_step_5 = execution_step_4 + 1
    from datasets import load_dataset
    import pandas as pd
    import numpy as np

    dataset = load_dataset("unsloth/OpenMathReasoning-mini", split = "cot")
    dataset = dataset.to_pandas()[
        ["expected_answer", "problem", "generated_solution"]
    ]

    # 숫자로 변환을 시도하고 실패하면 NaN으로 대체
    is_number = pd.to_numeric(pd.Series(dataset["expected_answer"]), errors = "coerce").notnull()
    # 숫자만 선택
    dataset = dataset.iloc[np.where(is_number)[0]]

    dataset
    return dataset, execution_step_5, load_dataset, np


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    GRPO 스타일에 맞게 데이터셋 형식을 변환해야 합니다.
    """)
    return


@app.cell
def _(
    dataset,
    execution_step_5,
    reasoning_end,
    reasoning_start,
    solution_end,
    solution_start,
    system_prompt,
):
    # 원본 코드 셀 순서와 부수 효과를 보존합니다.
    execution_step_6 = execution_step_5 + 1
    def format_dataset(x):
        expected_answer = x["expected_answer"]
        problem = x["problem"]

        # 생성된 사고 과정 태그 제거
        thoughts = x["generated_solution"]
        thoughts = thoughts.replace("<think>", "").replace("</think>", "")

        # 앞뒤 줄바꿈 제거
        thoughts = thoughts.strip()
        # 사용자 정의 형식 추가
        final_prompt = \
            reasoning_start + thoughts + reasoning_end + \
            solution_start + expected_answer + solution_end
        return [
            {"role" : "system",    "content" : system_prompt},
            {"role" : "user",      "content" : problem},
            {"role" : "assistant", "content" : final_prompt},
        ]

    dataset["Messages"] = dataset.apply(format_dataset, axis = 1)
    return (execution_step_6,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    제대로 적용되었는지 확인합니다.
    """)
    return


@app.cell
def _(dataset, execution_step_6, tokenizer):
    # 원본 코드 셀 순서와 부수 효과를 보존합니다.
    execution_step_7 = execution_step_6 + 1
    tokenizer.apply_chat_template(dataset["Messages"][0], tokenize = False)
    return (execution_step_7,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    지나치게 긴 추론 과정을 피하기 위해 사전 미세 조정 데이터셋의 길이를 `max_seq_length/2`로 제한합니다.

    약 2분이 걸릴 수 있습니다!
    """)
    return


@app.cell
def _(dataset, execution_step_7, max_seq_length, tokenizer):
    # 원본 코드 셀 순서와 부수 효과를 보존합니다.
    execution_step_8 = execution_step_7 + 1
    dataset['N'] = dataset['Messages'].apply(lambda x: len(tokenizer.apply_chat_template(x)))
    dataset_1 = dataset.loc[dataset['N'] <= max_seq_length / 2].copy()
    dataset_1.shape
    return dataset_1, execution_step_8


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    그런 다음 메시지를 토큰화하고 Hugging Face 호환 데이터셋 형식으로 변환합니다.
    """)
    return


@app.cell
def _(dataset_1, execution_step_8, tokenizer):
    # 원본 코드 셀 순서와 부수 효과를 보존합니다.
    execution_step_9 = execution_step_8 + 1
    from datasets import Dataset
    dataset_1['text'] = tokenizer.apply_chat_template(dataset_1['Messages'].values.tolist(), tokenize=False)
    dataset_2 = Dataset.from_pandas(dataset_1)
    dataset_2
    return dataset_2, execution_step_9


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    이제 모델이 사용자 정의 GRPO 형식을 따르도록 사전 미세 조정을 진행합니다!
    """)
    return


@app.cell
def _(dataset_2, execution_step_9, model, output_dir, tokenizer):
    # 원본 코드 셀 순서와 부수 효과를 보존합니다.
    execution_step_10 = execution_step_9 + 1
    from trl import SFTTrainer, SFTConfig
    trainer = SFTTrainer(model=model, tokenizer=tokenizer, train_dataset=dataset_2, args=SFTConfig(output_dir=str(output_dir / 'checkpoints'), dataset_text_field='text', per_device_train_batch_size=1, gradient_accumulation_steps=1, warmup_steps=5, num_train_epochs=2, learning_rate=0.0002, logging_steps=5, optim='adamw_8bit', weight_decay=0.001, lr_scheduler_type='linear', seed=3407, report_to='none'))  # 그래디언트 누적으로 더 큰 배치 크기를 구현하세요!  # 전체 데이터를 한 번 학습하려면 설정하세요.  # 장시간 학습할 때는 2e-5로 낮추세요  # TrackIO/WandB 등을 사용하려면 설정하세요
    return execution_step_10, trainer


@app.cell
def _(execution_step_10, trainer):
    # 원본 코드 셀 순서와 부수 효과를 보존합니다.
    execution_step_11 = execution_step_10 + 1
    trainer.train()
    return (execution_step_11,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    모델이 사용자 정의 형식을 익혔는지 확인해 보겠습니다.
    """)
    return


@app.cell
def _(dataset_2, execution_step_11, model, tokenizer):
    execution_step_12 = execution_step_11 + 1
    _text = tokenizer.apply_chat_template(dataset_2[0]['Messages'][:2], tokenize=False, add_generation_prompt=True)
    from transformers import TextStreamer
    _ = model.generate(**tokenizer(_text, return_tensors='pt').to('cuda'), temperature=0, max_new_tokens=1024, streamer=TextStreamer(tokenizer, skip_prompt=False))
    return (execution_step_12,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    모델이 형식을 잘 따릅니다! GRPO 단계로 넘어가기 전에 일부 항목을 메모리에서 제거하겠습니다.
    """)
    return


@app.cell
def _(execution_step_12, torch):
    # 원본 코드 셀 순서와 부수 효과를 보존합니다.
    execution_step_13 = execution_step_12 + 1
    # 이전 데이터셋은 marimo 셀에 유지됩니다.
    torch.cuda.empty_cache()
    import gc
    gc.collect()
    return (execution_step_13,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 데이터 준비
    <a name="Data"></a>

    Hugging Face의 [Open R1 Math 데이터셋](https://huggingface.co/datasets/open-r1/DAPO-Math-17k-Processed)을 사용합니다. OpenAI의 유명한 [GSM8K 데이터셋](https://huggingface.co/datasets/openai/gsm8k)을 활용해도 됩니다.
    """)
    return


@app.cell
def _(execution_step_13, load_dataset):
    # 원본 코드 셀 순서와 부수 효과를 보존합니다.
    execution_step_14 = execution_step_13 + 1
    dataset_3 = load_dataset('open-r1/DAPO-Math-17k-Processed', 'en', split='train')
    dataset_3
    return dataset_3, execution_step_14


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    첫 번째 행을 살펴보겠습니다.
    """)
    return


@app.cell
def _(dataset_3, execution_step_14):
    # 원본 코드 셀 순서와 부수 효과를 보존합니다.
    execution_step_15 = execution_step_14 + 1
    dataset_3[0]['prompt']
    return (execution_step_15,)


@app.cell
def _(dataset_3, execution_step_15):
    # 원본 코드 셀 순서와 부수 효과를 보존합니다.
    execution_step_16 = execution_step_15 + 1
    dataset_3[0]['solution']
    return (execution_step_16,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    GSM8K의 답에는 `####`가 있으므로 이를 기준으로 답을 추출합니다. Open R1 데이터셋에서는 아래 단계를 건너뛰어도 됩니다.
    """)
    return


@app.cell
def _(dataset_3, execution_step_16):
    # 원본 코드 셀 순서와 부수 효과를 보존합니다.
    execution_step_17 = execution_step_16 + 1

    def extract_hash_answer(text):  # if "####" not in text: return None
        return text  # return text.split("####")[1].strip()
    extract_hash_answer(dataset_3[0]['solution'])
    return execution_step_17, extract_hash_answer


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    데이터셋에 변환을 적용하고 첫 번째 행을 확인해 보겠습니다!
    """)
    return


@app.cell
def _(dataset_3, execution_step_17, extract_hash_answer, system_prompt):
    # 원본 코드 셀 순서와 부수 효과를 보존합니다.
    execution_step_18 = execution_step_17 + 1
    dataset_4 = dataset_3.map(lambda x: {'prompt': [{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': x['prompt']}], 'answer': extract_hash_answer(x['solution'])})
    dataset_4[0]
    return dataset_4, execution_step_18


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    추론 구간과 답을 찾는 정규식 패턴을 만듭니다.
    """)
    return


@app.cell
def _(execution_step_18, reasoning_end, solution_start, tokenizer):
    # 원본 코드 셀 순서와 부수 효과를 보존합니다.
    execution_step_19 = execution_step_18 + 1
    import re

    # 선택적으로 EOS 토큰도 일치시킵니다
    solution_end_regex = r"</SOLUTION>[\s]{0,}" + \
        "(?:" + re.escape(tokenizer.eos_token) + ")?"

    match_format = re.compile(
        rf"{reasoning_end}.*?"\
        rf"{solution_start}(.+?){solution_end_regex}"\
        rf"[\s]{{0,}}$",
        flags = re.MULTILINE | re.DOTALL
    )
    print(match_format)
    return execution_step_19, match_format, re


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    제대로 동작하는지 확인합니다.
    """)
    return


@app.cell
def _(execution_step_19, match_format):
    # 원본 코드 셀 순서와 부수 효과를 보존합니다.
    execution_step_20 = execution_step_19 + 1
    match_format.findall(
        "Let me think!<end_working_out>"\
        f"<SOLUTION>\n2\n</SOLUTION>",
    )
    return (execution_step_20,)


@app.cell
def _(execution_step_20, match_format):
    # 원본 코드 셀 순서와 부수 효과를 보존합니다.
    execution_step_21 = execution_step_20 + 1
    match_format.findall(
        "<start_working_out>Let me think!<end_working_out>"\
        f"<SOLUTION>  2  </SOLUTION>\n\n",
    )
    return (execution_step_21,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    이제 형식을 정확히 따르는지 확인하는 보상 함수를 만듭니다. 성공하면 3점을 부여합니다.
    """)
    return


@app.cell
def _(execution_step_21, match_format):
    # 원본 코드 셀 순서와 부수 효과를 보존합니다.
    execution_step_22 = execution_step_21 + 1

    def match_format_exactly(completions, **kwargs):
        scores = []
        for completion in completions:
            score = 0
            response = completion[0]['content']  # 형식이 정확히 일치하는지 확인합니다!
            if match_format.search(response) is not None:
                score = score + 3.0
            scores.append(score)
        return scores

    return execution_step_22, match_format_exactly


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    정확한 형식 일치에 실패하더라도 부분적으로 형식을 따랐다면 보상하도록 각 기호의 개수를 셉니다.
    """)
    return


@app.cell
def _(execution_step_22, reasoning_end, solution_end, solution_start):
    # 원본 코드 셀 순서와 부수 효과를 보존합니다.
    execution_step_23 = execution_step_22 + 1

    def match_format_approximately(completions, **kwargs):
        scores = []
        for completion in completions:
            score = 0
            response = completion[0]['content']  # 키워드가 몇 번 나오는지 세고 너무 많으면 감점합니다!
            score = score + (0.5 if response.count(reasoning_end) == 1 else -1.0)  # 한 번 나오면 점수를 더합니다!
            score = score + (0.5 if response.count(solution_start) == 1 else -1.0)
            score = score + (0.5 if response.count(solution_end) == 1 else -1.0)  # 시작 태그는 항상 앞에 붙이므로 보상할 필요가 없습니다!
            scores.append(score)  # score += 0.5 if response.count(reasoning_start) == 1 else -1.0
        return scores

    return execution_step_23, match_format_approximately


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    마지막으로 생성된 답을 추출해 보상하거나 감점합니다! 정답에 얼마나 가까운지도 비율을 통해 평가해 보상을 부여합니다.
    """)
    return


@app.cell
def _(execution_step_23, match_format):
    # 원본 코드 셀 순서와 부수 효과를 보존합니다.
    execution_step_24 = execution_step_23 + 1

    def check_answer(prompts, completions, answer, **kwargs):
        question = prompts[0][-1]['content']
        responses = [completion[0]['content'] for completion in completions]
        extracted_responses = [guess.group(1) if (guess := match_format.search(r)) is not None else None for r in responses]
        scores = []
        for guess, true_answer in zip(extracted_responses, answer):
            score = 0
            if guess is None:
                scores.append(-2.0)
                continue
            if guess == true_answer:
                score = score + 5.0
            elif guess.strip() == true_answer.strip():
                score = score + 3.5
            else:
                try:  # 정답이면 5점!
                    ratio = float(guess) / float(true_answer)
                    if ratio >= 0.9 and ratio <= 1.1:
                        score = score + 2.0  # 공백이 있어도 일치로 인정하되 보상은 줄입니다
                    elif ratio >= 0.8 and ratio <= 1.2:
                        score = score + 1.5
                    else:
                        score = score - 2.5  # 비율을 기준으로 답이 정답에 가까워도 보상합니다!
                except:  # 답이 일정 범위 안에 있으면 보상합니다!
                    score = score - 4.5
            scores.append(score)
        return scores  # 오답에 감점  # 감점

    return check_answer, execution_step_24


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    답이 숫자 하나가 아니라 문장일 때도 있습니다. 예를 들어 "The solution is $20"에서는 20을 추출합니다.

    123,456처럼 숫자에 들어 있을 수 있는 쉼표도 제거합니다.
    """)
    return


@app.cell
def _(execution_step_24, re, solution_start):
    # 원본 코드 셀 순서와 부수 효과를 보존합니다.
    execution_step_25 = execution_step_24 + 1
    match_numbers = re.compile(
        solution_start + r".*?[\s]{0,}([-]?[\d\.\,]{1,})",
        flags = re.MULTILINE | re.DOTALL
    )
    print(match_numbers.findall("<SOLUTION>  0.34  </SOLUTION>"))
    print(match_numbers.findall("<SOLUTION>  123,456  </SOLUTION>"))
    print(match_numbers.findall("<SOLUTION>  -0.234  </SOLUTION>"))
    print(match_numbers.findall("<SOLUTION>17</SOLUTION>"))
    return execution_step_25, match_numbers


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    이제 생성된 응답과 정답을 출력하는 주요 함수를 준비합니다. 텍스트를 `float`로 변환해 정답과 같은지 확인하는 보상 함수도 함께 만듭니다.
    """)
    return


@app.cell
def _(execution_step_25, match_numbers):
    # 원본 코드 셀 순서와 부수 효과를 보존합니다.
    execution_step_26 = execution_step_25 + 1
    _print_state = {'times': 0, 'every': 5}

    def check_numbers(prompts, completions, answer, **kwargs):
        question = prompts[0][-1]['content']
        responses = [completion[0]['content'] for completion in completions]
        extracted_responses = [guess.group(1) if (guess := match_numbers.search(r)) is not None else None for r in responses]
        scores = []
        if _print_state['times'] % _print_state['every'] == 0:
            print('*' * 20 + f'Question:\n{question}', f'\nAnswer:\n{answer[0]}', f'\nResponse:\n{responses[0]}', f'\nExtracted:\n{extracted_responses[0]}')
        _print_state['times'] = _print_state['times'] + 1
        for guess, true_answer in zip(extracted_responses, answer):
            if guess is None:
                scores.append(-2.5)
                continue  # 몇 스텝마다 한 번만 출력
            try:
                true_answer = float(true_answer.strip())
                guess = float(guess.strip().replace(',', ''))
                scores.append(3.5 if guess == true_answer else -1.5)
            except:
                scores.append(0)
                continue
        return scores  # 숫자로 변환  # 123,456과 같은 숫자의 쉼표 제거

    return check_numbers, execution_step_26


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    프롬프트가 의도치 않게 잘리지 않도록 길이의 90백분위수를 구합니다!

    즉, 가장 긴 상위 10%의 프롬프트를 제거합니다.
    """)
    return


@app.cell
def _(dataset_4, execution_step_26, np, tokenizer):
    # 원본 코드 셀 순서와 부수 효과를 보존합니다.
    execution_step_27 = execution_step_26 + 1
    tokenized = dataset_4.map(lambda x: {'tokens': tokenizer.apply_chat_template(x['prompt'], add_generation_prompt=True, tokenize=True)}, batched=True)
    print(tokenizer.decode(tokenized[0]['tokens']))
    tokenized = tokenized.map(lambda x: {'L': len(x['tokens'])})
    maximum_length = int(np.quantile(tokenized['L'], 0.9))
    print('Max Length = ', maximum_length)
    dataset_5 = dataset_4.select(np.where(np.array(tokenized['L']) <= maximum_length)[0])
    # 길이가 90백분위수보다 짧은 샘플만 유지
    del tokenized
    return dataset_5, execution_step_27, maximum_length


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <a name="Train"></a>
    ### 모델 학습

    이제 GRPO 트레이너와 모든 설정을 준비합니다!
    """)
    return


@app.cell
def _(
    execution_step_27,
    max_seq_length,
    maximum_length,
    output_dir,
    tokenizer,
):
    # 원본 코드 셀 순서와 부수 효과를 보존합니다.
    execution_step_28 = execution_step_27 + 1
    max_prompt_length = maximum_length + 1 # 여유를 위해 1을 더합니다!
    max_completion_length = max_seq_length - max_prompt_length

    from vllm import SamplingParams
    vllm_sampling_params = SamplingParams(
        min_p = 0.1,
        top_p = 1.0,
        top_k = -1,
        seed = 3407,
        stop = [tokenizer.eos_token],
        include_stop_str_in_output = True,
    )

    from trl import GRPOConfig, GRPOTrainer
    training_args = GRPOConfig(
        vllm_sampling_params = vllm_sampling_params,
        temperature = 1.0,
        learning_rate = 5e-6,
        weight_decay = 0.001,
        warmup_ratio = 0.1,
        lr_scheduler_type = "linear",
        optim = "adamw_8bit",
        logging_steps = 1,
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = 1, # 더 안정적인 학습을 위해 4로 늘리세요
        num_generations = 4, # 메모리가 부족하면 줄이세요
        max_prompt_length = max_prompt_length,
        max_completion_length = max_completion_length,
        # num_train_epochs = 1, # 전체 데이터를 한 번 학습하려면 1로 설정하세요
        max_steps = 100,
        save_steps = 100,
        report_to = "none", # Weights & Biases 사용 가능
        output_dir = str(output_dir / 'checkpoints'),

        # 선택적 학습 및 평가 설정
        # fp16_full_eval = True,
        # per_device_eval_batch_size = 4,
        # eval_accumulation_steps = 1,
        # eval_strategy = "steps",
        # eval_steps = 1,
    )
    return GRPOTrainer, SamplingParams, execution_step_28, training_args


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    트레이너를 실행해 보겠습니다! 위로 스크롤하면 보상 표를 볼 수 있습니다. `reward` 열의 값이 증가하는 것이 목표입니다!

    변화가 나타나기까지 150~200스텝을 기다려야 할 수 있습니다. 처음 100스텝 동안은 보상이 0일 수도 있으니 여유를 두고 지켜보세요!

    | 스텝 | 학습 손실 | reward    | reward_std | completion_length | kl       |
    |------|---------------|-----------|------------|-------------------|----------|
    | 1    | 0.000000      | 0.125000  | 0.000000   | 200.000000        | 0.000000 |
    | 2    | 0.000000      | 0.072375  | 0.248112   | 200.000000        | 0.000000 |
    | 3    | 0.000000      | -0.079000 | 0.163776   | 182.500000        | 0.000005 |
    """)
    return


@app.cell
def _(
    GRPOTrainer,
    check_answer,
    check_numbers,
    dataset_5,
    execution_step_28,
    match_format_approximately,
    match_format_exactly,
    model,
    tokenizer,
    training_args,
):
    # 원본 코드 셀 순서와 부수 효과를 보존합니다.
    execution_step_29 = execution_step_28 + 1
    # 선택적 학습 및 평가 설정
    # new_dataset = dataset.train_test_split(test_size = 0.01)
    trainer_1 = GRPOTrainer(model=model, processing_class=tokenizer, reward_funcs=[match_format_exactly, match_format_approximately, check_answer, check_numbers], args=training_args, train_dataset=dataset_5)
    trainer_1.train()  # 선택적 학습 및 평가 설정  # train_dataset = new_dataset["train"],  # eval_dataset = new_dataset["test"],
    return (execution_step_29,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <a name="Inference"></a>
    ### 추론
    방금 학습한 모델을 실행해 보겠습니다! 먼저 GRPO 학습을 적용하지 않은 모델부터 확인합니다.
    """)
    return


@app.cell
def _(SamplingParams, execution_step_29, model):
    execution_step_30 = execution_step_29 + 1
    _text = 'What is the sqrt of 101?'
    _sampling_params = SamplingParams(temperature=1.0, top_k=50, max_tokens=1024)
    _output = model.fast_generate([_text], sampling_params=_sampling_params, lora_request=None)[0].outputs[0].text
    _output
    return (execution_step_30,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    이제 GRPO로 학습한 LoRA를 사용해 보겠습니다. 우선 LoRA를 저장합니다!
    """)
    return


@app.cell
def _(execution_step_30, model, output_dir):
    # 원본 코드 셀 순서와 부수 효과를 보존합니다.
    execution_step_31 = execution_step_30 + 1
    model.save_lora(str(output_dir / 'grpo_saved_lora'))
    return (execution_step_31,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    LoRA가 실제로 학습되었는지 확인합니다!
    """)
    return


@app.cell
def _(execution_step_31, output_dir):
    # 원본 코드 셀 순서와 부수 효과를 보존합니다.
    execution_step_32 = execution_step_31 + 1
    from safetensors import safe_open

    tensors = {}
    with safe_open(str(output_dir / 'grpo_saved_lora/adapter_model.safetensors'), framework = "pt") as f:
        # A와 B 모두 0이 아닌지 확인
        for key in f.keys():
            tensor = f.get_tensor(key)
            n_zeros = (tensor == 0).sum() / tensor.numel()
            assert(n_zeros.item() != tensor.numel())
    return (execution_step_32,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    이제 LoRA를 불러와 테스트합니다.
    """)
    return


@app.cell
def _(
    SamplingParams,
    execution_step_32,
    model,
    output_dir,
    system_prompt,
    tokenizer,
):
    execution_step_33 = execution_step_32 + 1
    messages = [{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': 'What is the sqrt of 101?'}]
    _text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    _sampling_params = SamplingParams(temperature=1.0, top_k=50, max_tokens=2048)
    _output = model.fast_generate(_text, sampling_params=_sampling_params, lora_request=model.load_lora(str(output_dir / 'grpo_saved_lora')))[0].outputs[0].text
    _output
    return (execution_step_33,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    추론 모델의 성능이 훨씬 좋아졌습니다. 약 한 시간만 학습했으므로 항상 정답을 내지는 않습니다. 시퀀스 길이를 늘리고 더 오래 학습하면 개선될 수 있습니다!
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <a name="Save"></a>
    ### vLLM용 float16으로 저장

    `float16` 형식에 직접 저장할 수도 있습니다. float16은 `merged_16bit`, int4는 `merged_4bit`를 선택하세요. 대안으로 `lora` 어댑터 형식도 지원합니다. Hugging Face 계정에 업로드하려면 `push_to_hub_merged`를 사용하세요! 개인 토큰은 https://huggingface.co/settings/tokens 에서 발급할 수 있습니다. 다른 배포 방법은 [문서](https://unsloth.ai/docs/basics/inference-and-deployment)를 참고하세요.
    """)
    return


@app.cell
def _(execution_step_33, model, output_dir, tokenizer):
    # 원본 코드 셀 순서와 부수 효과를 보존합니다.
    execution_step_34 = execution_step_33 + 1
    # 16비트로 병합
    if False: model.save_pretrained_merged(str(output_dir / 'qwen_finetune_16bit'), tokenizer, save_method = "merged_16bit",)
    if False: model.push_to_hub_merged("HF_USERNAME/qwen_finetune_16bit", tokenizer, save_method = "merged_16bit", token = "YOUR_HF_TOKEN")

    # 4비트로 병합
    if False: model.save_pretrained_merged(str(output_dir / 'qwen_finetune_4bit'), tokenizer, save_method = "merged_4bit",)
    if False: model.push_to_hub_merged("HF_USERNAME/qwen_finetune_4bit", tokenizer, save_method = "merged_4bit", token = "YOUR_HF_TOKEN")

    # LoRA 어댑터만 저장
    if False:
        model.save_pretrained(str(output_dir / 'qwen_lora'))
        tokenizer.save_pretrained(str(output_dir / 'qwen_lora'))
    if False:
        model.push_to_hub("HF_USERNAME/qwen_lora", token = "YOUR_HF_TOKEN")
        tokenizer.push_to_hub("HF_USERNAME/qwen_lora", token = "YOUR_HF_TOKEN")
    return (execution_step_34,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### GGUF / llama.cpp 변환
    `GGUF` / `llama.cpp` 저장을 기본 지원합니다! `llama.cpp`를 복제하고 기본적으로 `q8_0` 형식으로 저장합니다. `q4_k_m` 등 다른 방식도 지원합니다. 로컬 저장에는 `save_pretrained_gguf`, Hugging Face 업로드에는 `push_to_hub_gguf`를 사용하세요.

    지원하는 양자화 방식의 일부는 다음과 같습니다. 전체 목록은 [문서](https://unsloth.ai/docs/basics/inference-and-deployment/saving-to-gguf)를 참고하세요.
    * `q8_0` - 변환이 빠릅니다. 자원 사용량이 높지만 일반적으로 감당할 만합니다.
    * `q4_k_m` - 권장 방식입니다. attention.wv와 feed_forward.w2 텐서의 절반에 Q6_K를 적용하고 나머지에는 Q4_K를 적용합니다.
    * `q5_k_m` - 권장 방식입니다. attention.wv와 feed_forward.w2 텐서의 절반에 Q6_K를 적용하고 나머지에는 Q5_K를 적용합니다.

    [**신규**] 미세 조정 후 Ollama로 자동으로 내보내려면 [Ollama 노트북](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3_(8B)-Ollama.ipynb)을 사용해 보세요.
    """)
    return


@app.cell
def _(execution_step_34, model, output_dir, tokenizer):
    # 원본 코드 셀 순서와 부수 효과를 보존합니다.
    execution_step_35 = execution_step_34 + 1
    # 8비트 Q8_0으로 저장
    if False: model.save_pretrained_gguf(str(output_dir / 'qwen_finetune'), tokenizer,)
    # https://huggingface.co/settings/tokens 에서 토큰을 발급하세요!
    # hf를 본인의 사용자 이름으로 바꾸는 것도 잊지 마세요!
    if False: model.push_to_hub_gguf("HF_USERNAME/qwen_finetune", tokenizer, token = "YOUR_HF_TOKEN")

    # 16비트 GGUF로 저장
    if False: model.save_pretrained_gguf(str(output_dir / 'qwen_finetune'), tokenizer, quantization_method = "f16")
    if False: model.push_to_hub_gguf("HF_USERNAME/qwen_finetune", tokenizer, quantization_method = "f16", token = "YOUR_HF_TOKEN")

    # q4_k_m GGUF로 저장
    if False: model.save_pretrained_gguf(str(output_dir / 'qwen_finetune'), tokenizer, quantization_method = "q4_k_m")
    if False: model.push_to_hub_gguf("HF_USERNAME/qwen_finetune", tokenizer, quantization_method = "q4_k_m", token = "YOUR_HF_TOKEN")

    # 여러 GGUF 형식으로 저장 - 여러 형식이 필요할 때 훨씬 빠릅니다!
    if False:
        model.push_to_hub_gguf(
            "HF_USERNAME/qwen_finetune", # hf를 본인의 사용자 이름으로 바꾸세요!
            tokenizer,
            quantization_method = ["q4_k_m", "q8_0", "q5_k_m",],
            token = "YOUR_HF_TOKEN",
        )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    이제 llama.cpp에서 `qwen_finetune.Q8_0.gguf` 또는 `qwen_finetune.Q4_K_M.gguf` 파일을 사용하세요.

    이제 모든 과정이 끝났습니다! Unsloth에 관한 질문은 [Discord](https://discord.gg/unsloth) 채널에서 나눌 수 있습니다. 버그를 발견했거나 최신 LLM 소식, 도움, 프로젝트 참여 기회를 찾는다면 자유롭게 참여하세요!

    추가 참고 자료:

    1. 나만의 추론 모델 학습 - Llama GRPO 노트북 [무료 Colab](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3.1_(8B)-GRPO.ipynb)
    2. 미세 조정 모델을 Ollama용으로 저장하기. [무료 노트북](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3_(8B)-Ollama.ipynb)
    3. Llama 3.2 Vision 미세 조정 - 방사선 영상 활용 사례. [무료 Colab](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3.2_(11B)-Vision.ipynb)
    4. DPO, ORPO, 지속 사전 학습, 대화형 미세 조정 등의 노트북은 [문서](https://unsloth.ai/docs/get-started/unsloth-notebooks)에서 확인하세요!

    <div class="align-center">
      <a href="https://unsloth.ai"><img src="https://github.com/unslothai/unsloth/raw/main/images/unsloth%20new%20logo.png" width="115"></a>
      <a href="https://discord.gg/unsloth"><img src="https://github.com/unslothai/unsloth/raw/main/images/Discord.png" width="145"></a>
      <a href="https://unsloth.ai/docs/"><img src="https://github.com/unslothai/unsloth/blob/main/images/documentation%20green%20button.png?raw=true" width="125"></a>

      도움이 필요하면 Discord에 참여하세요 + ⭐️ <i><a href="https://github.com/unslothai/unsloth">Github</a>에서 별을 눌러 주세요 </i> ⭐️
    </div>

      이 노트북을 포함한 모든 Unsloth 노트북의 라이선스는 [LGPL-3.0](https://github.com/unslothai/notebooks?tab=LGPL-3.0-1-ov-file#readme)입니다.
    """)
    return


if __name__ == "__main__":
    app.run()
