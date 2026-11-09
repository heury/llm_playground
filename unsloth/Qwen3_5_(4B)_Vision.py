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
    _run_id = _RunDatetime.now(_RunZoneInfo("Asia/Seoul")).strftime("%Y%m%d-%H%M%S") + "-" + _run_uuid4().hex[:6] + "-Qwen3_5_(4B)_Vision"
    output_dir = _RunPath(__file__).resolve().parents[1] / "outputs" / "unsloth" / _run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"결과 저장 위치: {output_dir}")
    execution_step_0 = 0
    return execution_step_0, output_dir


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Qwen3.5 (4B) — 비전 모델
    """)
    return


@app.cell
def _(execution_step_0):
    # 원본 코드 셀 순서와 부수 효과를 보존합니다.
    execution_step_1 = execution_step_0 + 1
    from unsloth import FastVisionModel # LLM에는 FastLanguageModel 사용
    import torch

    # 다운로드가 4배 빠르고 메모리 초과를 방지하는 사전 양자화 4비트 모델을 지원합니다.
    fourbit_models = [
        "unsloth/Llama-3.2-11B-Vision-Instruct-bnb-4bit", # Llama 3.2 비전 지원
        "unsloth/Llama-3.2-11B-Vision-bnb-4bit",
        "unsloth/Llama-3.2-90B-Vision-Instruct-bnb-4bit", # 80GB GPU 한 장에 들어갑니다!
        "unsloth/Llama-3.2-90B-Vision-bnb-4bit",

        "unsloth/Pixtral-12B-2409-bnb-4bit",              # Pixtral은 16GB에 들어갑니다!
        "unsloth/Pixtral-12B-Base-2409-bnb-4bit",         # Pixtral 베이스 모델

        "unsloth/Qwen2-VL-2B-Instruct-bnb-4bit",          # Qwen2 VL 지원
        "unsloth/Qwen2-VL-7B-Instruct-bnb-4bit",
        "unsloth/Qwen2-VL-72B-Instruct-bnb-4bit",

        "unsloth/llava-v1.6-mistral-7b-hf-bnb-4bit",      # 모든 Llava 변형 모델을 사용할 수 있습니다!
        "unsloth/llava-1.5-7b-hf-bnb-4bit",
    ] # 더 많은 모델: https://huggingface.co/unsloth

    model, tokenizer = FastVisionModel.from_pretrained(
        "unsloth/Qwen3.5-4B",
        load_in_4bit = False, # 메모리 사용량을 줄이려면 4비트를 사용하세요. 16비트 LoRA는 False로 설정합니다.
        use_gradient_checkpointing = "unsloth", # 긴 문맥에는 True 또는 "unsloth" 사용
    )
    return FastVisionModel, execution_step_1, model, tokenizer, torch


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    이제 매개변수 효율적인 미세 조정을 위해 LoRA 어댑터를 추가합니다. 이를 통해 전체 매개변수의 약 1%만 효율적으로 학습할 수 있습니다.

    **[신규]** 모델의 비전 부분만 또는 언어 부분만 미세 조정할 수도 있습니다. 둘 다 선택해도 됩니다! 어텐션 층이나 MLP 층을 선택해 미세 조정할 수도 있습니다!
    """)
    return


@app.cell
def _(FastVisionModel, execution_step_1, model):
    # 원본 코드 셀 순서와 부수 효과를 보존합니다.
    execution_step_2 = execution_step_1 + 1
    model_1 = FastVisionModel.get_peft_model(model, finetune_vision_layers=True, finetune_language_layers=True, finetune_attention_modules=True, finetune_mlp_modules=True, r=16, lora_alpha=16, lora_dropout=0, bias='none', random_state=3407, use_rslora=False, loftq_config=None)  # 비전 층을 미세 조정하지 않으면 False  # 언어 층을 미세 조정하지 않으면 False  # 어텐션 층을 미세 조정하지 않으면 False  # MLP 층을 미세 조정하지 않으면 False  # 클수록 정확도가 높아지지만 과적합될 수 있습니다  # alpha는 최소한 r과 같은 값으로 설정하는 것을 권장합니다  # 순위 안정화 LoRA를 지원합니다  # LoftQ도 지원합니다  # target_modules = "all-linear", # 이제 선택 사항입니다! 필요하면 목록을 지정하세요
    return execution_step_2, model_1


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <a name="Data"></a>
    ### 데이터 준비
    손글씨 수학 수식을 샘플링한 데이터셋을 사용합니다. 목표는 이미지를 컴퓨터가 읽고 렌더링할 수 있는 LaTeX 형식으로 변환하는 것입니다. 복잡한 수식에 특히 유용합니다.

    사용할 데이터셋은 [여기](https://huggingface.co/datasets/unsloth/LaTeX_OCR), 전체 데이터셋은 [여기](https://huggingface.co/datasets/linxy/LaTeX_OCR)에서 확인할 수 있습니다.
    """)
    return


@app.cell
def _(execution_step_2):
    # 원본 코드 셀 순서와 부수 효과를 보존합니다.
    execution_step_3 = execution_step_2 + 1
    from datasets import load_dataset
    dataset = load_dataset("unsloth/LaTeX_OCR", split = "train")
    return dataset, execution_step_3


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    데이터셋을 살펴보겠습니다. 세 번째 이미지와 그에 해당하는 캡션을 확인합니다.
    """)
    return


@app.cell
def _(dataset, execution_step_3):
    # 원본 코드 셀 순서와 부수 효과를 보존합니다.
    execution_step_4 = execution_step_3 + 1
    dataset
    return (execution_step_4,)


@app.cell
def _(dataset, execution_step_4):
    # 원본 코드 셀 순서와 부수 효과를 보존합니다.
    execution_step_5 = execution_step_4 + 1
    dataset[2]["image"]
    return (execution_step_5,)


@app.cell
def _(dataset, execution_step_5):
    # 원본 코드 셀 순서와 부수 효과를 보존합니다.
    execution_step_6 = execution_step_5 + 1
    dataset[2]["text"]
    return (execution_step_6,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    LaTeX를 브라우저에서 직접 렌더링할 수도 있습니다!
    """)
    return


@app.cell
def _(dataset, execution_step_6):
    # 원본 코드 셀 순서와 부수 효과를 보존합니다.
    execution_step_7 = execution_step_6 + 1
    from IPython.display import display, Math, Latex

    latex = dataset[2]["text"]
    display(Math(latex))
    return (execution_step_7,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    데이터셋의 형식을 맞추려면 모든 비전 미세 조정 작업에서 다음 구조를 사용해야 합니다.

    ```python
    [
    { "role": "user",
      "content": [{"type": "text",  "text": Q}, {"type": "image", "image": image} ]
    },
    { "role": "assistant",
      "content": [{"type": "text",  "text": A} ]
    },
    ]
    ```
    """)
    return


@app.cell
def _(execution_step_7):
    # 원본 코드 셀 순서와 부수 효과를 보존합니다.
    execution_step_8 = execution_step_7 + 1
    _instruction = 'Write the LaTeX representation for this image.'

    def convert_to_conversation(sample):
        conversation = [{'role': 'user', 'content': [{'type': 'text', 'text': _instruction}, {'type': 'image', 'image': sample['image']}]}, {'role': 'assistant', 'content': [{'type': 'text', 'text': sample['text']}]}]
        return {'messages': conversation}
    pass
    return convert_to_conversation, execution_step_8


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    데이터셋을 미세 조정에 맞는 형식으로 변환해 보겠습니다.
    """)
    return


@app.cell
def _(convert_to_conversation, dataset, execution_step_8):
    # 원본 코드 셀 순서와 부수 효과를 보존합니다.
    execution_step_9 = execution_step_8 + 1
    converted_dataset = [convert_to_conversation(sample) for sample in dataset]
    return converted_dataset, execution_step_9


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    첫 번째 예제의 대화 구조를 살펴봅니다.
    """)
    return


@app.cell
def _(converted_dataset, execution_step_9):
    # 원본 코드 셀 순서와 부수 효과를 보존합니다.
    execution_step_10 = execution_step_9 + 1
    converted_dataset[0]
    return (execution_step_10,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    미세 조정 전에 첫 번째 예제에 대해 모델이 어떤 출력을 내는지 확인해 보겠습니다!
    """)
    return


@app.cell
def _(FastVisionModel, dataset, execution_step_10, model_1, tokenizer):
    execution_step_11 = execution_step_10 + 1
    FastVisionModel.for_inference(model_1)
    _image = dataset[2]['image']
    _instruction = 'Write the LaTeX representation for this image.'
    _messages = [{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': _instruction}]}]
    _input_text = tokenizer.apply_chat_template(_messages, add_generation_prompt=True)
    _inputs = tokenizer(_image, _input_text, add_special_tokens=False, return_tensors='pt').to('cuda')
    from transformers import TextStreamer
    _text_streamer = TextStreamer(tokenizer, skip_prompt=True)
    _ = model_1.generate(**_inputs, streamer=_text_streamer, max_new_tokens=128, use_cache=True, temperature=1.5, min_p=0.1)
    return TextStreamer, execution_step_11


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <a name="Train"></a>
    ### 모델 학습
    이제 모델을 학습합니다. 빠르게 실행하기 위해 60스텝만 학습하지만, 전체 학습을 실행하려면 `num_train_epochs=1`로 설정하고 `max_steps=None`으로 스텝 제한을 해제하면 됩니다. 강화 학습을 위한 `DPOTrainer`와 `GRPOTrainer`도 지원합니다!

    비전 미세 조정을 돕는 새로운 `UnslothVisionDataCollator`를 사용합니다.
    """)
    return


@app.cell
def _(
    FastVisionModel,
    converted_dataset,
    execution_step_11,
    model_1,
    output_dir,
    tokenizer,
):
    # 원본 코드 셀 순서와 부수 효과를 보존합니다.
    execution_step_12 = execution_step_11 + 1
    from unsloth.trainer import UnslothVisionDataCollator
    from trl import SFTTrainer, SFTConfig
    FastVisionModel.for_training(model_1)
    trainer = SFTTrainer(model=model_1, tokenizer=tokenizer, data_collator=UnslothVisionDataCollator(model_1, tokenizer), train_dataset=converted_dataset, args=SFTConfig(per_device_train_batch_size=2, gradient_accumulation_steps=4, warmup_steps=5, max_steps=30, learning_rate=0.0002, logging_steps=1, optim='adamw_8bit', weight_decay=0.001, lr_scheduler_type='linear', seed=3407, output_dir=str(output_dir / 'checkpoints'), report_to='none', remove_unused_columns=False, dataset_text_field='', dataset_kwargs={'skip_prepare_dataset': True}, max_length=2048))  # 학습 모드 활성화!  # 반드시 사용해야 합니다!  # num_train_epochs = 1, # 전체 학습을 하려면 max_steps 대신 설정하세요  # Weights and Biases용  # 비전 미세 조정에는 아래 항목을 반드시 설정해야 합니다:
    return execution_step_12, trainer


@app.cell
def _(execution_step_12, torch):
    # 원본 코드 셀 순서와 부수 효과를 보존합니다.
    execution_step_13 = execution_step_12 + 1
    # @title 현재 메모리 사용 통계 표시
    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
    print(f"{start_gpu_memory} GB of memory reserved.")
    return execution_step_13, max_memory, start_gpu_memory


@app.cell
def _(execution_step_13, trainer):
    # 원본 코드 셀 순서와 부수 효과를 보존합니다.
    execution_step_14 = execution_step_13 + 1
    trainer_stats = trainer.train()
    return execution_step_14, trainer_stats


@app.cell
def _(execution_step_14, max_memory, start_gpu_memory, torch, trainer_stats):
    # 원본 코드 셀 순서와 부수 효과를 보존합니다.
    execution_step_15 = execution_step_14 + 1
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
    return (execution_step_15,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <a name="Inference"></a>
    ### 추론
    모델을 실행해 보겠습니다! 지시문과 입력은 바꿔도 됩니다. 출력은 비워 두세요!

    `min_p = 0.1`과 `temperature = 1.5`를 사용합니다. 그 이유는 이 [게시물](https://x.com/menhguin/status/1826132708508213629)을 참고하세요.
    """)
    return


@app.cell
def _(
    FastVisionModel,
    TextStreamer,
    dataset,
    execution_step_15,
    model_1,
    tokenizer,
):
    execution_step_16 = execution_step_15 + 1
    FastVisionModel.for_inference(model_1)
    _image = dataset[2]['image']
    _instruction = 'Write the LaTeX representation for this image.'
    _messages = [{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': _instruction}]}]
    _input_text = tokenizer.apply_chat_template(_messages, add_generation_prompt=True)
    _inputs = tokenizer(_image, _input_text, add_special_tokens=False, return_tensors='pt').to('cuda')
    _text_streamer = TextStreamer(tokenizer, skip_prompt=True)
    _ = model_1.generate(**_inputs, streamer=_text_streamer, max_new_tokens=128, use_cache=True, temperature=1.5, min_p=0.1)
    return (execution_step_16,)


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
def _(execution_step_16, model_1, output_dir, tokenizer):
    # 원본 코드 셀 순서와 부수 효과를 보존합니다.
    execution_step_17 = execution_step_16 + 1
    model_1.save_pretrained(str(output_dir / 'qwen_lora'))  # 로컬에 저장
    # model.push_to_hub("your_name/qwen_lora", token = "YOUR_HF_TOKEN") # 온라인에 저장
    # tokenizer.push_to_hub("your_name/qwen_lora", token = "YOUR_HF_TOKEN") # 온라인에 저장
    tokenizer.save_pretrained(str(output_dir / 'qwen_lora'))
    return (execution_step_17,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    방금 저장한 LoRA 어댑터를 추론용으로 불러오려면 `False`를 `True`로 바꾸세요.
    """)
    return


@app.cell
def _(FastVisionModel, TextStreamer, dataset, execution_step_17, output_dir):
    execution_step_18 = execution_step_17 + 1
    if False:
        model_2, tokenizer_1 = FastVisionModel.from_pretrained(model_name=str(output_dir / 'qwen_lora'), load_in_4bit=True)
        FastVisionModel.for_inference(model_2)
    _image = dataset[0]['image']
    _instruction = 'Write the LaTeX representation for this image.'
    _messages = [{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': _instruction}]}]
    _input_text = tokenizer_1.apply_chat_template(_messages, add_generation_prompt=True)
    _inputs = tokenizer_1(_image, _input_text, add_special_tokens=False, return_tensors='pt').to('cuda')
    _text_streamer = TextStreamer(tokenizer_1, skip_prompt=True)
    _ = model_2.generate(**_inputs, streamer=_text_streamer, max_new_tokens=128, use_cache=True, temperature=1.5, min_p=0.1)
    return execution_step_18, model_2, tokenizer_1


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### vLLM용 float16으로 저장

    `float16` 형식에 직접 저장할 수도 있습니다. float16은 `merged_16bit`를 선택하세요. Hugging Face 계정에 업로드하려면 `push_to_hub_merged`를 사용하세요! 개인 토큰은 https://huggingface.co/settings/tokens 에서 발급할 수 있습니다. 다른 배포 방법은 [문서](https://unsloth.ai/docs/basics/inference-and-deployment)를 참고하세요.
    """)
    return


@app.cell
def _(execution_step_18, model_2, output_dir, tokenizer_1):
    # 원본 코드 셀 순서와 부수 효과를 보존합니다.
    execution_step_19 = execution_step_18 + 1
    # 저장 방식은 하나만 선택하세요! 둘 다 필요하지 않습니다!
    if False:
    # 로컬에 16비트로 저장
        model_2.save_pretrained_merged(str(output_dir / 'unsloth_finetune'), tokenizer_1)
    if False:
    # Hugging Face 계정으로 내보내 저장
        model_2.push_to_hub_merged('YOUR_USERNAME/unsloth_finetune', tokenizer_1, token='YOUR_HF_TOKEN')
    return (execution_step_19,)


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
def _(execution_step_19, model_2, output_dir, tokenizer_1):
    # 원본 코드 셀 순서와 부수 효과를 보존합니다.
    execution_step_20 = execution_step_19 + 1
    # 8비트 Q8_0으로 저장
    if False:
    # https://huggingface.co/settings/tokens 에서 토큰을 발급하세요!
    # hf를 본인의 사용자 이름으로 바꾸는 것도 잊지 마세요!
        model_2.save_pretrained_gguf(str(output_dir / 'qwen_finetune'), tokenizer_1)
    if False:
    # 16비트 GGUF로 저장
        model_2.push_to_hub_gguf('HF_USERNAME/qwen_finetune', tokenizer_1, token='YOUR_HF_TOKEN')
    if False:
        model_2.save_pretrained_gguf(str(output_dir / 'qwen_finetune'), tokenizer_1, quantization_method='f16')
    # q4_k_m GGUF로 저장
    if False:
        model_2.push_to_hub_gguf('HF_USERNAME/qwen_finetune', tokenizer_1, quantization_method='f16', token='YOUR_HF_TOKEN')
    if False:
    # 여러 GGUF 형식으로 저장 - 여러 형식이 필요할 때 훨씬 빠릅니다!
        model_2.save_pretrained_gguf(str(output_dir / 'qwen_finetune'), tokenizer_1, quantization_method='q4_k_m')
    if False:
        model_2.push_to_hub_gguf('HF_USERNAME/qwen_finetune', tokenizer_1, quantization_method='q4_k_m', token='YOUR_HF_TOKEN')  # hf를 본인의 사용자 이름으로 바꾸세요!
    if False:
        model_2.push_to_hub_gguf('HF_USERNAME/qwen_finetune', tokenizer_1, quantization_method=['q4_k_m', 'q8_0', 'q5_k_m'], token='YOUR_HF_TOKEN')
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

      <b>이 노트북을 포함한 모든 Unsloth 노트북의 라이선스는 [LGPL-3.0](https://github.com/unslothai/notebooks?tab=LGPL-3.0-1-ov-file#readme)입니다</b>
    </div>
    """)
    return


if __name__ == "__main__":
    app.run()
