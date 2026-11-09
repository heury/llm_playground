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
    _run_id = _RunDatetime.now(_RunZoneInfo("Asia/Seoul")).strftime("%Y%m%d-%H%M%S") + "-" + _run_uuid4().hex[:6] + "-Qwen3_(4B)_Instruct-QAT"
    output_dir = _RunPath(__file__).resolve().parents[2] / "outputs" / "unsloth" / _run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"결과 저장 위치: {output_dir}")
    execution_step_0 = 0
    return execution_step_0, output_dir


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Qwen3 (4B) Instruct — 양자화 인지 학습(QAT)
    """)
    return


@app.cell
def _(execution_step_0):
    # 원본 코드 셀 순서와 부수 효과를 보존합니다.
    execution_step_1 = execution_step_0 + 1
    from unsloth import FastLanguageModel
    import torch

    fourbit_models = [
        "unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit", # Qwen 14B를 2배 빠르게 실행
        "unsloth/Qwen3-4B-Thinking-2507-unsloth-bnb-4bit",
        "unsloth/Qwen3-8B-unsloth-bnb-4bit",
        "unsloth/Qwen3-14B-unsloth-bnb-4bit",
        "unsloth/Qwen3-32B-unsloth-bnb-4bit",

        # 높은 정확도와 낮은 메모리 사용량을 위한 4비트 동적 양자화
        "unsloth/gemma-3-12b-it-unsloth-bnb-4bit",
        "unsloth/Phi-4",
        "unsloth/Llama-3.1-8B",
        "unsloth/Llama-3.2-3B",
        "unsloth/orpheus-3b-0.1-ft-unsloth-bnb-4bit" # [신규] TTS 모델도 지원합니다!
    ] # 더 많은 모델: https://huggingface.co/unsloth

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "unsloth/Qwen3-4B-Instruct-2507",
        max_seq_length = 2048,   # 긴 문맥에 맞춰 원하는 값으로 설정하세요!
        load_in_4bit = False,    # 메모리 사용량을 줄이는 4비트 양자화
        load_in_8bit = False,    # [신규!] 정확도가 조금 더 높지만 메모리를 2배 사용합니다
        full_finetuning = False, # [신규!] 전체 매개변수 미세 조정도 지원합니다!
        # token = "YOUR_HF_TOKEN", # 접근 제한 모델용 HF 토큰
    )
    return FastLanguageModel, execution_step_1, model, tokenizer, torch


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    이제 LoRA 어댑터를 추가합니다. 이렇게 하면 소수의 매개변수만 업데이트하면 됩니다!
    """)
    return


@app.cell
def _(FastLanguageModel, execution_step_1, model):
    # 원본 코드 셀 순서와 부수 효과를 보존합니다.
    execution_step_2 = execution_step_1 + 1
    model_1 = FastLanguageModel.get_peft_model(model, r=16, target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'], lora_alpha=32, lora_dropout=0, bias='none', qat_scheme='int4', use_gradient_checkpointing='unsloth', random_state=3407, use_rslora=False, loftq_config=None)  # 0보다 큰 값을 선택하세요! 권장값: 8, 16, 32, 64, 128  # 모든 값을 지원하지만 0에 최적화되어 있습니다  # 모든 값을 지원하지만 "none"에 최적화되어 있습니다  # [신규] "unsloth"는 VRAM을 30% 적게 사용해 배치 크기를 2배로 늘릴 수 있습니다!  # 매우 긴 문맥에는 True 또는 "unsloth" 사용  # 순위 안정화 LoRA를 지원합니다  # LoftQ도 지원합니다
    return execution_step_2, model_1


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    QAT가 적용되었는지 확인해 보겠습니다!
    """)
    return


@app.cell
def _(execution_step_2, model_1):
    # 원본 코드 셀 순서와 부수 효과를 보존합니다.
    execution_step_3 = execution_step_2 + 1
    for module in model_1.modules():
        if 'FakeQuantized' in module.__class__.__name__:
            print('QAT is applied!')
            break
    return (execution_step_3,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <a name="Data"></a>
    ### 데이터 준비
    대화형 미세 조정에는 `Qwen-3` 형식을 사용합니다. ShareGPT 형식의 [Maxime Labonne의 FineTome-100k](https://huggingface.co/datasets/mlabonne/FineTome-100k) 데이터셋을 사용합니다. Qwen-3는 여러 턴의 대화를 다음과 같이 표현합니다.

    ```
    <|im_start|>user
    Hello!<|im_end|>
    <|im_start|>assistant
    Hey there!<|im_end|>

    ```
    `get_chat_template` 함수로 알맞은 채팅 템플릿을 가져옵니다. `zephyr, chatml, mistral, llama, alpaca, vicuna, vicuna_old, phi3, llama3, phi4, qwen2.5, gemma3` 등 다양한 형식을 지원합니다.
    """)
    return


@app.cell
def _(execution_step_3, tokenizer):
    # 원본 코드 셀 순서와 부수 효과를 보존합니다.
    execution_step_4 = execution_step_3 + 1
    from unsloth.chat_templates import get_chat_template
    tokenizer_1 = get_chat_template(tokenizer, chat_template='qwen3-instruct')
    return execution_step_4, tokenizer_1


@app.cell
def _(execution_step_4):
    # 원본 코드 셀 순서와 부수 효과를 보존합니다.
    execution_step_5 = execution_step_4 + 1
    from datasets import load_dataset
    dataset = load_dataset("mlabonne/FineTome-100k", split = "train")
    return dataset, execution_step_5


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    이제 `standardize_data_formats`로 데이터셋을 미세 조정에 맞는 형식으로 변환합니다!
    """)
    return


@app.cell
def _(dataset, execution_step_5):
    # 원본 코드 셀 순서와 부수 효과를 보존합니다.
    execution_step_6 = execution_step_5 + 1
    from unsloth.chat_templates import standardize_data_formats
    dataset_1 = standardize_data_formats(dataset)
    return dataset_1, execution_step_6


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    인덱스 100의 행이 어떻게 생겼는지 확인해 보겠습니다!
    """)
    return


@app.cell
def _(dataset_1, execution_step_6):
    # 원본 코드 셀 순서와 부수 효과를 보존합니다.
    execution_step_7 = execution_step_6 + 1
    dataset_1[100]
    return (execution_step_7,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    이제 대화에 `Qwen-3` 채팅 템플릿을 적용하고 결과를 `text`에 저장합니다.
    """)
    return


@app.cell
def _(dataset_1, execution_step_7, tokenizer_1):
    # 원본 코드 셀 순서와 부수 효과를 보존합니다.
    execution_step_8 = execution_step_7 + 1

    def formatting_prompts_func(examples):
        convos = examples['conversations']
        texts = [tokenizer_1.apply_chat_template(convo, tokenize=False, add_generation_prompt=False) for convo in convos]
        return {'text': texts}
    dataset_2 = dataset_1.map(formatting_prompts_func, batched=True)
    return dataset_2, execution_step_8


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    채팅 템플릿이 적용된 결과를 확인해 보겠습니다!
    """)
    return


@app.cell
def _(dataset_2, execution_step_8):
    # 원본 코드 셀 순서와 부수 효과를 보존합니다.
    execution_step_9 = execution_step_8 + 1
    dataset_2[100]['text']
    return (execution_step_9,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <a name="Train"></a>
    ### 모델 학습
    이제 모델을 학습합니다. 빠르게 실행하기 위해 60스텝만 학습하지만, 전체 학습을 실행하려면 `num_train_epochs=1`로 설정하고 `max_steps=None`으로 스텝 제한을 해제하면 됩니다.
    """)
    return


@app.cell
def _(dataset_2, execution_step_9, model_1, output_dir, tokenizer_1):
    # 원본 코드 셀 순서와 부수 효과를 보존합니다.
    execution_step_10 = execution_step_9 + 1
    from trl import SFTTrainer, SFTConfig
    trainer = SFTTrainer(model=model_1, tokenizer=tokenizer_1, train_dataset=dataset_2, eval_dataset=None, args=SFTConfig(output_dir=str(output_dir / 'checkpoints'), dataset_text_field='text', per_device_train_batch_size=1, gradient_accumulation_steps=4, warmup_steps=5, max_steps=30, learning_rate=0.0002, logging_steps=1, optim='adamw_8bit', weight_decay=0.001, lr_scheduler_type='linear', seed=3407, report_to='none'))  # 평가도 설정할 수 있습니다!  # 그래디언트 누적으로 더 큰 배치 크기를 구현하세요!  # num_train_epochs = 1, # 전체 데이터를 한 번 학습하려면 설정하세요.  # 장시간 학습할 때는 2e-5로 낮추세요  # WandB 등을 사용하려면 설정하세요
    return execution_step_10, trainer


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Unsloth의 `train_on_completions` 메서드로 어시스턴트 출력만 학습하고 사용자 입력의 손실은 무시합니다. 이렇게 하면 미세 조정 정확도를 높일 수 있습니다! 이제 Unsloth가 토크나이저의 채팅 템플릿에서 지시문과 응답 부분을 자동으로 감지하므로 `instruction_part`와 `response_part`를 전달할 필요가 없습니다. 사용자 정의 채팅 템플릿을 사용한다면 여전히 명시적으로 전달할 수 있습니다.
    """)
    return


@app.cell
def _(execution_step_10, trainer):
    # 원본 코드 셀 순서와 부수 효과를 보존합니다.
    execution_step_11 = execution_step_10 + 1
    from unsloth.chat_templates import train_on_responses_only
    trainer_1 = train_on_responses_only(trainer)
    return execution_step_11, trainer_1


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    지시문 부분이 마스킹되었는지 확인하겠습니다! 인덱스 100의 행을 다시 출력합니다.
    """)
    return


@app.cell
def _(execution_step_11, tokenizer_1, trainer_1):
    # 원본 코드 셀 순서와 부수 효과를 보존합니다.
    execution_step_12 = execution_step_11 + 1
    tokenizer_1.decode(trainer_1.train_dataset[100]['input_ids'])
    return (execution_step_12,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    이제 마스킹된 예제를 출력합니다. 답변만 남아 있어야 합니다.
    """)
    return


@app.cell
def _(execution_step_12, tokenizer_1, trainer_1):
    # 원본 코드 셀 순서와 부수 효과를 보존합니다.
    execution_step_13 = execution_step_12 + 1
    tokenizer_1.decode([tokenizer_1.pad_token_id if x == -100 else x for x in trainer_1.train_dataset[100]['labels']]).replace(tokenizer_1.pad_token, ' ')
    return (execution_step_13,)


@app.cell
def _(execution_step_13, torch):
    # 원본 코드 셀 순서와 부수 효과를 보존합니다.
    execution_step_14 = execution_step_13 + 1
    # @title 현재 메모리 사용 통계 표시
    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
    print(f"{start_gpu_memory} GB of memory reserved.")
    return execution_step_14, max_memory, start_gpu_memory


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    모델을 학습해 보겠습니다! 이전 학습을 재개하려면 `trainer.train(resume_from_checkpoint = True)`로 설정하세요.
    """)
    return


@app.cell
def _(execution_step_14, trainer_1):
    # 원본 코드 셀 순서와 부수 효과를 보존합니다.
    execution_step_15 = execution_step_14 + 1
    trainer_stats = trainer_1.train()
    return execution_step_15, trainer_stats


@app.cell
def _(execution_step_15, max_memory, start_gpu_memory, torch, trainer_stats):
    # 원본 코드 셀 순서와 부수 효과를 보존합니다.
    execution_step_16 = execution_step_15 + 1
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
    return (execution_step_16,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    학습이 끝났으므로 FakeQuantizedLinear 층을 표준 nn.Linear 층으로 되돌립니다. 이렇게 하면 가짜 양자화의 오버헤드를 제거하고 최종 변환이나 LoRA 어댑터 병합을 준비할 수 있습니다.
    """)
    return


@app.cell
def _(execution_step_16, model_1):
    # 원본 코드 셀 순서와 부수 효과를 보존합니다.
    execution_step_17 = execution_step_16 + 1
    from torchao.quantization import quantize_
    from torchao.quantization.qat import QATConfig
    quantize_(model_1, QATConfig(step='convert'))
    return (execution_step_17,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <a name="Inference"></a>
    ### 추론
    Unsloth의 기본 추론 기능으로 모델을 실행해 보겠습니다! `Qwen-3` 팀이 권장하는 지시 응답 추론 설정은 `temperature = 0.7, top_p = 0.8, top_k = 20`입니다.

    추론 과정을 포함하는 채팅 기반 추론에는 `temperature = 0.6, top_p = 0.95, top_k = 20`을 사용합니다.
    """)
    return


@app.cell
def _(execution_step_17, model_1, tokenizer_1):
    # 원본 코드 셀 순서와 부수 효과를 보존합니다.
    execution_step_18 = execution_step_17 + 1
    messages = [{'role': 'user', 'content': 'Continue the sequence: 1, 1, 2, 3, 5, 8,'}]
    text = tokenizer_1.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    from transformers import TextStreamer
    _ = model_1.generate(**tokenizer_1(text, return_tensors='pt').to('cuda'), max_new_tokens=1000, temperature=0.7, top_p=0.8, top_k=20, streamer=TextStreamer(tokenizer_1, skip_prompt=True))  # 생성 시 반드시 추가해야 합니다  # 더 긴 출력을 원하면 늘리세요!  # 비추론 모드용
    return (execution_step_18,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <a name="Save"></a>
    ### 미세 조정 모델 저장 및 불러오기
    최종 모델을 LoRA 어댑터로 저장하려면 Hugging Face의 `push_to_hub`로 온라인에 저장하거나 `save_pretrained`로 로컬에 저장하세요.

    **[참고]** 전체 모델이 아닌 LoRA 어댑터만 저장합니다. 16비트 또는 GGUF로 저장하려면 아래 내용을 참고하세요!
    """)
    return


@app.cell
def _(execution_step_18, model_1, output_dir, tokenizer_1):
    # 원본 코드 셀 순서와 부수 효과를 보존합니다.
    execution_step_19 = execution_step_18 + 1
    model_1.save_pretrained(str(output_dir / 'qwen_lora'))  # 로컬에 저장
    # model.push_to_hub("your_name/qwen_lora", token = "YOUR_HF_TOKEN") # 온라인에 저장
    # tokenizer.push_to_hub("your_name/qwen_lora", token = "YOUR_HF_TOKEN") # 온라인에 저장
    tokenizer_1.save_pretrained(str(output_dir / 'qwen_lora'))
    return (execution_step_19,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    방금 저장한 LoRA 어댑터를 추론용으로 불러오려면 `False`를 `True`로 바꾸세요.
    """)
    return


@app.cell
def _(FastLanguageModel, execution_step_19, output_dir):
    # 원본 코드 셀 순서와 부수 효과를 보존합니다.
    execution_step_20 = execution_step_19 + 1
    if False:
        model_2, tokenizer_2 = FastLanguageModel.from_pretrained(model_name=str(output_dir / 'qwen_lora'), max_seq_length=2048, load_in_4bit=True)  # 학습에 사용한 모델
    return execution_step_20, model_2, tokenizer_2


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    이제 TorchAO로 최종 모델을 저장하고 양자화할 수 있습니다. QAT 학습에 사용한 것과 동일한 설정을 적용합니다.
    """)
    return


@app.cell
def _(execution_step_20, model_2, output_dir, tokenizer_2):
    # 원본 코드 셀 순서와 부수 효과를 보존합니다.
    execution_step_21 = execution_step_20 + 1
    model_2.save_pretrained_torchao(str(output_dir / 'model'), tokenizer_2)
    return (execution_step_21,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### TorchAO 내보내기 및 변환

    사용자 정의 설정을 적용한 TorchAO 양자화 체크포인트로 내보내 vLLM이나 다른 추론 엔진에서 실행하는 기능도 지원합니다.

    TorchAO 설정에 관한 자세한 내용은 Hugging Face Transformers 공식 문서를 참고하세요: https://huggingface.co/docs/transformers/main/quantization/torchao
    """)
    return


@app.cell
def _(execution_step_21, model_2, output_dir, tokenizer_2):
    # 원본 코드 셀 순서와 부수 효과를 보존합니다.
    execution_step_22 = execution_step_21 + 1
    # TorchAO int4로 저장:
    if False:
        from torchao.quantization import Int4WeightOnlyConfig
        model_2.save_pretrained_torchao(str(output_dir / 'model'), tokenizer_2, torchao_config=Int4WeightOnlyConfig())
    if False:
        model_2.save_pretrained_torchao('HF_USERNAME/model', tokenizer_2, torchao_config=Int4WeightOnlyConfig(), push_to_hub=True, token='YOUR_HF_TOKEN')  # HF Hub에 업로드
    if False:
        from torchao.quantization import Int8DynamicActivationInt8WeightConfig
        model_2.save_pretrained_torchao(str(output_dir / 'model'), tokenizer_2, torchao_config=Int8DynamicActivationInt8WeightConfig())  # hf를 본인의 사용자 이름으로 바꾸세요!
    if False:
    # TorchAO int8로 저장:
        model_2.save_pretrained_torchao('HF_USERNAME/model', tokenizer_2, torchao_config=Int8DynamicActivationInt8WeightConfig(), push_to_hub=True, token='YOUR_HF_TOKEN')  # 토큰 발급: https://huggingface.co/settings/tokens  # HF Hub에 업로드  # hf를 본인의 사용자 이름으로 바꾸세요!  # 토큰 발급: https://huggingface.co/settings/tokens
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    이제 모든 과정이 끝났습니다! Unsloth에 관한 질문은 [Discord](https://discord.gg/unsloth) 채널에서 나눌 수 있습니다. 버그를 발견했거나 최신 LLM 소식, 도움, 프로젝트 참여 기회를 찾는다면 자유롭게 참여하세요!

    추가 참고 자료:

    1. 로컬에서 Unsloth를 사용하려면 [설치 안내](https://unsloth.ai/docs/get-started/install)를 참고하세요. Windows, Docker, AMD 및 Intel GPU에서의 설치 방법을 설명합니다.
    2. [강화 학습 안내 및 노트북](https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide)로 강화 학습 방법을 익혀 보세요.
    3. [텍스트 음성 변환(TTS)](https://unsloth.ai/docs/basics/text-to-speech-tts-fine-tuning) 및 [비전](https://unsloth.ai/docs/basics/vision-fine-tuning) 모델 지원에 관한 안내와 노트북을 참고하세요.
    4. [LLM 튜토리얼 목록](https://unsloth.ai/docs/models/tutorials-how-to-fine-tune-and-run-llms)에서 모델별 전용 안내를 찾아보세요.
    5. 추론에 도움이 필요하다면 [추론 및 배포 안내](https://unsloth.ai/docs/basics/inference-and-deployment)를 참고하세요. vLLM, llama.cpp, Ollama 등의 사용법을 설명합니다.

    <div class="align-center">
      <a href="https://unsloth.ai"><img src="https://github.com/unslothai/unsloth/raw/main/images/unsloth%20new%20logo.png" width="115"></a>
      <a href="https://discord.gg/unsloth"><img src="https://github.com/unslothai/unsloth/raw/main/images/Discord.png" width="145"></a>
      <a href="https://unsloth.ai/docs/"><img src="https://github.com/unslothai/unsloth/blob/main/images/documentation%20green%20button.png?raw=true" width="125"></a>

      도움이 필요하면 Discord에 참여하세요 + ⭐️ <i><a href="https://github.com/unslothai/unsloth">Github</a>에서 별을 눌러 주세요 </i> ⭐️

      이 노트북을 포함한 모든 Unsloth 노트북의 라이선스는 [LGPL-3.0](https://github.com/unslothai/notebooks?tab=LGPL-3.0-1-ov-file#readme)입니다
    </div>
    """)
    return


if __name__ == "__main__":
    app.run()
