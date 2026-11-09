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
    _run_id = _RunDatetime.now(_RunZoneInfo("Asia/Seoul")).strftime("%Y%m%d-%H%M%S") + "-" + _run_uuid4().hex[:6] + "-Qwen3_5_(4B)-Decision"
    output_dir = _RunPath(__file__).resolve().parents[2] / "outputs" / "unsloth" / _run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"결과 저장 위치: {output_dir}")
    execution_step_0 = 0
    return execution_step_0, output_dir


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Qwen3.5 (4B) — 판단 모델

    판단 모델은 텍스트를 생성하는 대신 입력을 읽고 주어진 선택지를 살펴본 뒤, 확률과 함께 하나를 선택합니다. `FastDecisionModel`은 LLM에 작은 헤드를 추가해 이러한 모델로 바꿉니다. 이 헤드는 입력에 대한 모든 질문을 한 번의 순전파로 읽고 각 선택지의 점수를 계산합니다.

    다른 모델을 사용해 보려면 `model_name`을 `unsloth/Llama-3.2-3B-Instruct` 또는 `unsloth/gemma-4-E4B-it`으로 바꾸세요.
    """)
    return


@app.cell
def _(execution_step_0):
    # 원본 코드 셀 순서와 부수 효과를 보존합니다.
    execution_step_1 = execution_step_0 + 1
    from unsloth import FastDecisionModel, DecisionTrainer, is_bfloat16_supported
    import torch

    model, tokenizer = FastDecisionModel.from_pretrained(
        model_name = "unsloth/Qwen3.5-4B",
        max_seq_length = 2048, # 최대 입력 길이. 긴 입력에서도 질문과 선택지를 유지합니다.
        load_in_4bit = True, # 메모리 사용량을 줄이는 4비트 양자화
    )
    return (
        DecisionTrainer,
        FastDecisionModel,
        execution_step_1,
        is_bfloat16_supported,
        model,
        tokenizer,
        torch,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    이제 LoRA 어댑터를 추가해 소수의 매개변수만 업데이트하도록 합니다! 새로 추가한 판단 헤드는 항상 학습합니다.
    """)
    return


@app.cell
def _(FastDecisionModel, execution_step_1, model):
    # 원본 코드 셀 순서와 부수 효과를 보존합니다.
    execution_step_2 = execution_step_1 + 1
    model_1 = FastDecisionModel.get_peft_model(model, r=16, lora_alpha=16, lora_dropout=0, use_gradient_checkpointing='unsloth', random_state=3407)  # 0보다 큰 값을 선택하세요! 권장값: 8, 16, 32, 64, 128  # 모든 값을 지원하지만 0에 최적화되어 있습니다  # 매우 긴 문맥에는 True 또는 "unsloth" 사용
    return execution_step_2, model_1


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <a name="Data"></a>
    ### 데이터 준비
    [typed-decisions](https://huggingface.co/datasets/LocalLLaMA/typed-decisions) 데이터셋을 사용합니다. 각 행에는 입력인 `state`, 이에 관해 판단할 `questions`, 정답인 `gold`가 있습니다. 질문은 세 가지 유형입니다.

    * `choice`: 티켓을 담당할 팀처럼 선택지 중 하나를 고릅니다.
    * `noul`: 예 또는 아니요를 판단합니다.
    * `score`: 긴급도처럼 등급 하나를 고릅니다.

    인덱스 0의 행을 확인해 보겠습니다!
    """)
    return


@app.cell
def _(execution_step_2):
    # 원본 코드 셀 순서와 부수 효과를 보존합니다.
    execution_step_3 = execution_step_2 + 1
    from datasets import load_dataset
    import json

    dataset = load_dataset("LocalLLaMA/typed-decisions", "all", split = "train")
    print(dataset[0]["state"][:500])
    print(json.dumps(json.loads(dataset[0]["questions"]), indent = 2)[:1000])
    print(dataset[0]["gold"])
    return dataset, execution_step_3, load_dataset


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    `build_dataset`은 각 행을 해당 행의 모든 질문이 담긴 하나의 예제로 변환하고, 건너뛴 판단의 수와 이유를 알려 줍니다. 자체 데이터를 사용하려면 같은 `state`, `questions`, `gold` 필드를 가진 행 목록을 전달하세요.

    `split_holdout`은 나중에 정확도를 확인하고 모델을 보정할 수 있도록 일부 행을 학습에서 제외합니다. 여기서는 80개 행, 400개 판단을 따로 남깁니다.
    """)
    return


@app.cell
def _(FastDecisionModel, dataset, execution_step_3, model_1, tokenizer):
    # 원본 코드 셀 순서와 부수 효과를 보존합니다.
    execution_step_4 = execution_step_3 + 1
    items, report = FastDecisionModel.build_dataset(dataset, tokenizer, model_1)
    print(f"Skipped {report['skipped']} of {report['total']} decisions")
    train_items, eval_items = FastDecisionModel.split_holdout(items, seed=3407)
    print(f'{len(train_items)} training rows, {len(eval_items)} held out')
    return eval_items, execution_step_4, train_items


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    학습 전 정확도를 확인해 보겠습니다. 새로 만든 헤드이므로 무작위로 추측하는 수준과 비슷합니다. 원본 실험에서는 약 30%의 정확도를 보였습니다.
    """)
    return


@app.cell
def _(FastDecisionModel, eval_items, execution_step_4, model_1, tokenizer):
    # 원본 코드 셀 순서와 부수 효과를 보존합니다.
    execution_step_5 = execution_step_4 + 1
    FastDecisionModel.evaluate(model_1, tokenizer, eval_items)
    return (execution_step_5,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <a name="Train"></a>
    ### 모델 학습
    이제 모델을 학습합니다. 빠르게 실행하기 위해 60스텝(약 1.7에포크)만 학습합니다. 원본 실험에서는 테스트 정확도 75~78%를 달성했으며 무료 T4 GPU에서는 약 2시간, RTX PRO 6000에서는 12분이 걸렸습니다. T4는 bfloat16을 지원하지 않고 Qwen3.5는 순수 float16에서 NaN이 발생하므로 Unsloth가 T4에서는 float32로 학습합니다. 이 때문에 T4에서 속도가 느립니다. 전체 학습을 실행하려면 `num_train_epochs = 2`로 설정하고 `max_steps`를 제거하세요. 여기서는 70스텝에 해당하며, 원본 실험에서 A100으로 25분 동안 학습해 테스트 정확도 78%를 달성했습니다.
    """)
    return


@app.cell
def _(
    DecisionTrainer,
    eval_items,
    execution_step_5,
    is_bfloat16_supported,
    model_1,
    output_dir,
    tokenizer,
    train_items,
):
    # 원본 코드 셀 순서와 부수 효과를 보존합니다.
    execution_step_6 = execution_step_5 + 1
    from transformers import TrainingArguments
    trainer = DecisionTrainer(model=model_1, processing_class=tokenizer, train_dataset=train_items, eval_dataset=eval_items, args=TrainingArguments(per_device_train_batch_size=8, gradient_accumulation_steps=4, warmup_steps=10, max_steps=60, learning_rate=0.0002, lr_scheduler_type='cosine', weight_decay=0.01, bf16=is_bfloat16_supported(), fp16=not is_bfloat16_supported(), eval_strategy='epoch', logging_steps=10, output_dir=str(output_dir / 'checkpoints'), report_to='none', seed=3407))  # 그래디언트 누적으로 더 큰 배치 크기를 구현하세요!  # num_train_epochs = 2, # 전체 학습을 실행하려면 설정하세요.  # TrackIO/WandB 등을 사용하려면 설정하세요
    return execution_step_6, trainer


@app.cell
def _(execution_step_6, torch):
    # 원본 코드 셀 순서와 부수 효과를 보존합니다.
    execution_step_7 = execution_step_6 + 1
    # @title 현재 메모리 사용 통계 표시
    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
    print(f"{start_gpu_memory} GB of memory reserved.")
    return execution_step_7, max_memory, start_gpu_memory


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    모델을 학습해 보겠습니다! 이전 학습을 재개하려면 `trainer.train(resume_from_checkpoint = True)`로 설정하세요.
    """)
    return


@app.cell
def _(execution_step_7, trainer):
    # 원본 코드 셀 순서와 부수 효과를 보존합니다.
    execution_step_8 = execution_step_7 + 1
    trainer_stats = trainer.train()
    return execution_step_8, trainer_stats


@app.cell
def _(execution_step_8, max_memory, start_gpu_memory, torch, trainer_stats):
    # 원본 코드 셀 순서와 부수 효과를 보존합니다.
    execution_step_9 = execution_step_8 + 1
    # @title 최종 메모리 및 시간 통계 표시
    used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
    used_percentage = round(used_memory / max_memory * 100, 3)
    lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
    print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
    print(
        f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training."
    )
    print(f"Peak reserved memory = {used_memory} GB.")
    print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
    print(f"Peak reserved memory % of max memory = {used_percentage} %.")
    print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")
    return (execution_step_9,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <a name="Inference"></a>
    ### 추론
    먼저 따로 남겨 둔 행으로 모델을 보정합니다. 보정은 확률을 조정해 신뢰도 90%로 내놓은 답이 실제로도 약 90%의 비율로 맞도록 합니다. `ece`는 보정 오차이며 낮을수록 좋습니다. 원본 실험에서는 60스텝 후 홀드아웃 정확도 80~84%, `ece` 0.02~0.06을 얻었습니다.
    """)
    return


@app.cell
def _(FastDecisionModel, eval_items, execution_step_9, model_1, tokenizer):
    # 원본 코드 셀 순서와 부수 효과를 보존합니다.
    execution_step_10 = execution_step_9 + 1
    FastDecisionModel.calibrate(model_1, tokenizer, eval_items)
    return (execution_step_10,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    모델이 한 번도 보지 않은 데이터셋의 테스트 분할에서 정확도를 확인하겠습니다. 원본 실험에서는 60스텝 후 75~78%를 얻었습니다.
    """)
    return


@app.cell
def _(FastDecisionModel, execution_step_10, load_dataset, model_1, tokenizer):
    # 원본 코드 셀 순서와 부수 효과를 보존합니다.
    execution_step_11 = execution_step_10 + 1
    test = load_dataset('LocalLLaMA/typed-decisions', 'all', split='test')
    test_items, _ = FastDecisionModel.build_dataset(test, tokenizer, model_1)
    FastDecisionModel.evaluate(model_1, tokenizer, test_items)
    return (execution_step_11,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    이제 판단을 수행해 보겠습니다! `predict`에 입력과 질문을 전달하세요. `answer`는 `choice`에서는 선택지, `noul`에서는 `True` 또는 `False`, `score`에서는 등급 번호입니다.
    """)
    return


@app.cell
def _(FastDecisionModel, execution_step_11, model_1, tokenizer):
    # 원본 코드 셀 순서와 부수 효과를 보존합니다.
    execution_step_12 = execution_step_11 + 1
    FastDecisionModel.for_inference(model_1)
    answers = FastDecisionModel.predict(model_1, tokenizer, 'Hi, I was charged twice for invoice #4411. Please refund the duplicate today.', {'team': {'type': 'choice', 'instructions': 'Which team should handle this?', 'criteria': {'billing': 'invoices, payments, refunds', 'technical': 'bugs, outages, errors', 'sales': 'pricing, new plans'}}, 'refund': {'type': 'noul', 'instructions': 'Does the customer ask for a refund?'}, 'urgency': {'type': 'score', 'instructions': 'How urgent is this?', 'criteria': ['not urgent', 'soon', 'today']}})
    for name, result in answers.items():
        print(name, result['answer'], {k: round(v, 3) for k, v in result['probabilities'].items()})
    return (execution_step_12,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <a name="Save"></a>
    ### 미세 조정 모델 저장 및 불러오기
    최종 모델을 저장하려면 `save_pretrained`로 로컬에 저장하거나 `push_to_hub`로 온라인에 저장하세요. LoRA 어댑터, 판단 헤드, 보정 결과가 저장됩니다.
    """)
    return


@app.cell
def _(execution_step_12, model_1, output_dir):
    # 원본 코드 셀 순서와 부수 효과를 보존합니다.
    execution_step_13 = execution_step_12 + 1
    # model.push_to_hub("your_name/qwen_lora", token = "YOUR_HF_TOKEN") # 온라인에 저장
    model_1.save_pretrained(str(output_dir / 'qwen_lora'))  # 로컬에 저장
    return (execution_step_13,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    방금 저장한 모델을 불러오려면 `False`를 `True`로 바꾸세요.
    """)
    return


@app.cell
def _(FastDecisionModel, execution_step_13, output_dir):
    # 원본 코드 셀 순서와 부수 효과를 보존합니다.
    execution_step_14 = execution_step_13 + 1
    if False:
        model_2, tokenizer_1 = FastDecisionModel.from_pretrained(model_name=str(output_dir / 'qwen_lora'), max_seq_length=2048, load_in_4bit=True)  # 학습에 사용한 모델
    return execution_step_14, model_2, tokenizer_1


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### float16으로 저장

    `save_pretrained_merged`로 `float16` 형식에 직접 저장할 수도 있습니다. Hugging Face 계정에 업로드하려면 `push_to_hub_merged`를 사용하세요! 개인 토큰은 https://huggingface.co/settings/tokens 에서 발급할 수 있습니다. 다른 배포 방법은 [문서](https://unsloth.ai/docs/basics/inference-and-deployment)를 참고하세요.
    """)
    return


@app.cell
def _(execution_step_14, model_2, output_dir, tokenizer_1):
    # 원본 코드 셀 순서와 부수 효과를 보존합니다.
    execution_step_15 = execution_step_14 + 1
    # 16비트로 병합
    if False:
        model_2.save_pretrained_merged(str(output_dir / 'qwen_finetune_16bit'), tokenizer_1, save_method='merged_16bit')
    if False:  # HF Hub에 업로드
        model_2.push_to_hub_merged('HF_USERNAME/qwen_finetune_16bit', tokenizer_1, save_method='merged_16bit', token='YOUR_HF_TOKEN')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    이제 모든 과정이 끝났습니다! Unsloth에 관한 질문은 [Discord](https://discord.gg/unsloth) 채널에서 나눌 수 있습니다. 버그를 발견했거나 최신 LLM 소식, 도움, 프로젝트 참여 기회를 찾는다면 자유롭게 참여하세요!

    자체 데이터로 판단 모델을 학습하려면 [안내](https://unsloth.ai/docs/models/decision-model-training)를 참고하세요.

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
