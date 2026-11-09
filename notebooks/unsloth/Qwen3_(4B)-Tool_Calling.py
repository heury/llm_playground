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
    _run_id = _RunDatetime.now(_RunZoneInfo("Asia/Seoul")).strftime("%Y%m%d-%H%M%S") + "-" + _run_uuid4().hex[:6] + "-Qwen3_(4B)-Tool_Calling"
    output_dir = _RunPath(__file__).resolve().parents[2] / "outputs" / "unsloth" / _run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"결과 저장 위치: {output_dir}")
    execution_step_0 = 0
    return execution_step_0, output_dir


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Qwen3 (4B) — 도구 호출
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    목표: **여러 턴의 도구 호출 대화**로 미세 조정해 `Qwen3-4B-Instruct-2507`에 도구 호출을 가르칩니다. 사용자가 질문하면 어시스턴트가 도구를 하나 이상 호출하고, 도구가 결과를 반환하면 어시스턴트가 그 결과를 사용해 답변합니다.

    이 과정은 도구를 사용하는 강화 학습 전의 준비 단계로도 유용합니다(GRPO 도구 사용 노트북 참고). 모델이 도구 호출 형식을 미리 익힐 수 있기 때문입니다.
    """)
    return


@app.cell
def _(execution_step_0):
    # 원본 코드 셀 순서와 부수 효과를 보존합니다.
    execution_step_1 = execution_step_0 + 1
    from unsloth import FastLanguageModel
    import torch

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "unsloth/Qwen3-4B-Instruct-2507",
        max_seq_length = 4096, # 도구 스키마와 도구 결과 때문에 대화가 길어집니다
        load_in_4bit = True,  # 메모리 사용량을 줄이는 4비트 양자화
        load_in_8bit = False, # [신규!] 정확도가 조금 더 높지만 메모리를 2배 사용합니다
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
    model_1 = FastLanguageModel.get_peft_model(model, r=32, target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'], lora_alpha=32, lora_dropout=0, bias='none', use_gradient_checkpointing='unsloth', random_state=3407, use_rslora=False, loftq_config=None)  # 0보다 큰 값을 선택하세요! 권장값: 8, 16, 32, 64, 128  # 모든 값을 지원하지만 0에 최적화되어 있습니다  # 모든 값을 지원하지만 "none"에 최적화되어 있습니다  # [신규] "unsloth"는 VRAM을 30% 적게 사용해 배치 크기를 2배로 늘릴 수 있습니다!  # 매우 긴 문맥에는 True 또는 "unsloth" 사용  # 순위 안정화 LoRA를 지원합니다  # LoftQ도 지원합니다
    return execution_step_2, model_1


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <a name="Data"></a>
    ### 데이터 준비
    NousResearch의 [Hermes Function Calling 데이터셋](https://huggingface.co/datasets/NousResearch/hermes-function-calling-v1)(Apache 2.0)을 사용합니다. 각 행에는 도구 목록과 어시스턴트가 도구를 호출하고 결과를 바탕으로 답변하는 대화가 있습니다.

    데이터셋은 도구 호출을 텍스트로 기록하므로 각 대화를 `tool_calls`와 `tool` 메시지가 포함된 표준 `messages` 형식으로 변환합니다. 그러면 모델 자체의 채팅 템플릿이 Qwen3가 학습한 형식 그대로 도구와 도구 호출을 표현합니다.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    `get_chat_template` 함수로 `Qwen-3` 지시 응답용 채팅 템플릿을 가져옵니다. 이 템플릿은 공식 Qwen3 템플릿과 동일하게 도구, 도구 호출, 도구 결과를 표현합니다.
    """)
    return


@app.cell
def _(execution_step_2, tokenizer):
    # 원본 코드 셀 순서와 부수 효과를 보존합니다.
    execution_step_3 = execution_step_2 + 1
    from unsloth.chat_templates import get_chat_template
    tokenizer_1 = get_chat_template(tokenizer, chat_template='qwen3-instruct')
    return execution_step_3, tokenizer_1


@app.cell
def _(execution_step_3):
    # 원본 코드 셀 순서와 부수 효과를 보존합니다.
    execution_step_4 = execution_step_3 + 1
    from datasets import load_dataset
    dataset = load_dataset("NousResearch/hermes-function-calling-v1", "func_calling", split = "train")
    dataset
    return dataset, execution_step_4


@app.cell
def _(dataset, execution_step_4, tokenizer_1):
    execution_step_5 = execution_step_4 + 1
    import json, re
    _tool_call_regex = re.compile('<tool_call>\\s*(.*?)\\s*</tool_call>', re.DOTALL)
    tool_response_regex = re.compile('<tool_response>\\s*(.*?)\\s*</tool_response>', re.DOTALL)

    def to_messages(example):
        messages = []
        for turn in example['conversations']:
            role, text = (turn['from'], turn['value'])
            if role == 'system':
                continue
            elif role == 'human':
                messages.append({'role': 'user', 'content': text})
            elif role == 'gpt':
                calls = [json.loads(call) for call in _tool_call_regex.findall(text)]
                message = {'role': 'assistant', 'content': _tool_call_regex.sub('', text).strip()}
                if calls:
                    message['tool_calls'] = [{'type': 'function', 'function': {'name': call['name'], 'arguments': call['arguments']}} for call in calls]
                messages.append(message)
            elif role == 'tool':
                for response in tool_response_regex.findall(text):
                    messages.append({'role': 'tool', 'content': response})
        return (messages, json.loads(example['tools']))

    def formatting_prompts_func(examples):
        texts = []
        for conversations, tools in zip(examples['conversations'], examples['tools']):
            try:
                messages, tools = to_messages({'conversations': conversations, 'tools': tools})
                text = tokenizer_1.apply_chat_template(messages, tools=tools, tokenize=False)
            except Exception:
                text = ''
            texts.append(text)
        return {'text': texts}
    dataset_1 = dataset.map(formatting_prompts_func, batched=True)
    dataset_1 = dataset_1.filter(lambda x: len(x['text']) != 0)
    dataset_1
    return dataset_1, execution_step_5, json, re


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    도구 호출이 포함된 대화에 채팅 템플릿이 어떻게 적용되는지 확인해 보겠습니다.
    """)
    return


@app.cell
def _(dataset_1, execution_step_5):
    # 원본 코드 셀 순서와 부수 효과를 보존합니다.
    execution_step_6 = execution_step_5 + 1
    print(dataset_1[0]['text'])
    return (execution_step_6,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <a name="Train"></a>
    ### 모델 학습
    이제 모델을 학습합니다. 빠르게 실행하기 위해 60스텝만 학습하지만, 전체 학습을 실행하려면 `num_train_epochs=1`로 설정하고 `max_steps=None`으로 스텝 제한을 해제하면 됩니다.
    """)
    return


@app.cell
def _(dataset_1, execution_step_6, model_1, output_dir, tokenizer_1):
    # 원본 코드 셀 순서와 부수 효과를 보존합니다.
    execution_step_7 = execution_step_6 + 1
    from trl import SFTTrainer, SFTConfig
    trainer = SFTTrainer(model=model_1, tokenizer=tokenizer_1, train_dataset=dataset_1, eval_dataset=None, args=SFTConfig(output_dir=str(output_dir / 'checkpoints'), dataset_text_field='text', per_device_train_batch_size=2, gradient_accumulation_steps=4, warmup_steps=5, max_steps=60, learning_rate=0.0002, logging_steps=1, optim='adamw_8bit', weight_decay=0.001, lr_scheduler_type='linear', seed=3407, report_to='none'))  # 평가도 설정할 수 있습니다!  # 그래디언트 누적으로 더 큰 배치 크기를 구현하세요!  # num_train_epochs = 1, # 전체 데이터를 한 번 학습하려면 설정하세요.  # 장시간 학습할 때는 2e-5로 낮추세요  # TrackIO/WandB 등을 사용하려면 설정하세요
    return execution_step_7, trainer


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **어시스턴트**가 작성한 도구 호출과 답변만 학습합니다. 사용자 턴과 도구 결과는 손실 계산에서 마스킹합니다. 모델은 도구를 호출하고 결과를 활용하는 법을 배워야 하며, 도구의 반환값을 예측할 필요는 없기 때문입니다.

    Qwen3는 도구 결과를 사용자 턴 안에 담아 전달하므로 Unsloth의 `train_on_responses_only`가 사용자 메시지와 함께 마스킹합니다.
    """)
    return


@app.cell
def _(execution_step_7, trainer):
    # 원본 코드 셀 순서와 부수 효과를 보존합니다.
    execution_step_8 = execution_step_7 + 1
    from unsloth.chat_templates import train_on_responses_only
    trainer_1 = train_on_responses_only(trainer, instruction_part='<|im_start|>user\n', response_part='<|im_start|>assistant\n')
    return execution_step_8, trainer_1


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    마스킹을 확인해 보겠습니다. 어시스턴트의 도구 호출과 답변만 남아 있어야 합니다.
    """)
    return


@app.cell
def _(execution_step_8, tokenizer_1, trainer_1):
    # 원본 코드 셀 순서와 부수 효과를 보존합니다.
    execution_step_9 = execution_step_8 + 1
    tokenizer_1.decode([tokenizer_1.pad_token_id if x == -100 else x for x in trainer_1.train_dataset[0]['labels']]).replace(tokenizer_1.pad_token, ' ')
    return (execution_step_9,)


@app.cell
def _(execution_step_9, torch):
    # 원본 코드 셀 순서와 부수 효과를 보존합니다.
    execution_step_10 = execution_step_9 + 1
    # @title 현재 메모리 사용 통계 표시
    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
    print(f"{start_gpu_memory} GB of memory reserved.")
    return execution_step_10, max_memory, start_gpu_memory


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    모델을 학습해 보겠습니다! 이전 학습을 재개하려면 `trainer.train(resume_from_checkpoint = True)`로 설정하세요.
    """)
    return


@app.cell
def _(execution_step_10, trainer_1):
    # 원본 코드 셀 순서와 부수 효과를 보존합니다.
    execution_step_11 = execution_step_10 + 1
    trainer_stats = trainer_1.train()
    return execution_step_11, trainer_stats


@app.cell
def _(execution_step_11, max_memory, start_gpu_memory, torch, trainer_stats):
    # 원본 코드 셀 순서와 부수 효과를 보존합니다.
    execution_step_12 = execution_step_11 + 1
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
    return (execution_step_12,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <a name="Inference"></a>
    ### 추론
    모델이 본 적 없는 도구를 제공해 보겠습니다. 생성하고, 모델이 요청한 도구를 실행하고, 결과를 대화에 추가한 뒤 다시 생성하는 간단한 에이전트 루프를 작성합니다. 모델이 답변할 때까지 이 과정을 반복합니다.
    """)
    return


@app.cell
def _(execution_step_12, json, re, tokenizer_1):
    execution_step_13 = execution_step_12 + 1
    _tool_call_regex = re.compile('<tool_call>\\s*(.*?)\\s*</tool_call>', re.DOTALL)

    def run_tools(messages, tools, generate, max_rounds=5):
        available = {tool.__name__: tool for tool in tools}
        for _ in range(max_rounds):
            text = tokenizer_1.apply_chat_template(messages, tools=tools, add_generation_prompt=True, tokenize=False)
            reply = generate(text)
            calls = []
            for raw in _tool_call_regex.findall(reply):
                try:
                    calls.append(json.loads(raw))
                except json.JSONDecodeError:
                    pass
            content = _tool_call_regex.sub('', reply).replace('<|im_end|>', '').strip()
            if not calls:
                messages.append({'role': 'assistant', 'content': content})
                return messages
            messages.append({'role': 'assistant', 'content': content, 'tool_calls': [{'type': 'function', 'function': {'name': call.get('name'), 'arguments': call.get('arguments', {})}} for call in calls]})
            for call in calls:
                try:
                    result = available[call['name']](**call.get('arguments', {}))
                except Exception as error:
                    result = {'error': str(error)}
                print(f"Tool call: {call.get('name')}({call.get('arguments')}) -> {result}")
                messages.append({'role': 'tool', 'name': call.get('name'), 'content': json.dumps(result)})
        return messages

    return execution_step_13, run_tools


@app.cell
def _(execution_step_13, model_1, run_tools, tokenizer_1):
    # 원본 코드 셀 순서와 부수 효과를 보존합니다.
    execution_step_14 = execution_step_13 + 1

    def get_stock_price(ticker: str) -> dict:
        """
        Get the latest stock price for a ticker symbol.

        Args:
            ticker: The stock ticker symbol, for example AAPL.

        Returns:
            The ticker and its latest price in US dollars.
        """  # 데모용 가상 가격 데이터
        prices = {'AAPL': 231.4, 'NVDA': 182.9, 'MSFT': 512.3}
        return {'ticker': ticker, 'price': prices.get(ticker.upper(), 100.0)}

    def generate(text):
        inputs = tokenizer_1(text, return_tensors='pt').to('cuda')
        outputs = model_1.generate(**inputs, max_new_tokens=512, temperature=0.7, top_p=0.8, top_k=20)
        return tokenizer_1.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=False)
    messages = run_tools([{'role': 'user', 'content': 'How much would 10 shares of NVDA and 5 shares of AAPL cost right now?'}], [get_stock_price], generate)  # 비추론 모드용
    print(messages[-1]['content'])
    return (execution_step_14,)


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
def _(execution_step_14, model_1, output_dir, tokenizer_1):
    # 원본 코드 셀 순서와 부수 효과를 보존합니다.
    execution_step_15 = execution_step_14 + 1
    model_1.save_pretrained(str(output_dir / 'qwen_lora'))  # 로컬에 저장
    # model.push_to_hub("your_name/qwen_lora", token = "YOUR_HF_TOKEN") # 온라인에 저장
    # tokenizer.push_to_hub("your_name/qwen_lora", token = "YOUR_HF_TOKEN") # 온라인에 저장
    tokenizer_1.save_pretrained(str(output_dir / 'qwen_lora'))
    return (execution_step_15,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <a name="Save"></a>
    ### vLLM용 float16으로 저장

    `float16` 형식에 직접 저장할 수도 있습니다. float16은 `merged_16bit`, int4는 `merged_4bit`를 선택하세요. 대안으로 `lora` 어댑터 형식도 지원합니다. Hugging Face 계정에 업로드하려면 `push_to_hub_merged`를 사용하세요! 개인 토큰은 https://huggingface.co/settings/tokens 에서 발급할 수 있습니다. 다른 배포 방법은 [문서](https://unsloth.ai/docs/basics/inference-and-deployment)를 참고하세요.
    """)
    return


@app.cell
def _(execution_step_15, model_1, output_dir, tokenizer_1):
    # 원본 코드 셀 순서와 부수 효과를 보존합니다.
    execution_step_16 = execution_step_15 + 1
    # 16비트로 병합
    if False:
        model_1.save_pretrained_merged(str(output_dir / 'qwen_finetune_16bit'), tokenizer_1, save_method='merged_16bit')
    if False:
    # 4비트로 병합
        model_1.push_to_hub_merged('HF_USERNAME/qwen_finetune_16bit', tokenizer_1, save_method='merged_16bit', token='YOUR_HF_TOKEN')
    if False:
        model_1.save_pretrained_merged(str(output_dir / 'qwen_finetune_4bit'), tokenizer_1, save_method='merged_4bit')
    # LoRA 어댑터만 저장
    if False:
        model_1.push_to_hub_merged('HF_USERNAME/qwen_finetune_4bit', tokenizer_1, save_method='merged_4bit', token='YOUR_HF_TOKEN')
    if False:
        model_1.save_pretrained(str(output_dir / 'qwen_lora'))
        tokenizer_1.save_pretrained(str(output_dir / 'qwen_lora'))
    if False:
        model_1.push_to_hub('HF_USERNAME/qwen_lora', token='YOUR_HF_TOKEN')
        tokenizer_1.push_to_hub('HF_USERNAME/qwen_lora', token='YOUR_HF_TOKEN')
    return (execution_step_16,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### GGUF / llama.cpp 변환
    `GGUF` / `llama.cpp` 저장을 기본 지원합니다! `llama.cpp`를 복제하고 기본적으로 `q8_0` 형식으로 저장합니다. `q4_k_m` 등 다른 방식도 지원합니다. 로컬 저장에는 `save_pretrained_gguf`, Hugging Face 업로드에는 `push_to_hub_gguf`를 사용하세요.

    지원하는 양자화 방식의 일부는 다음과 같습니다. 전체 목록은 [문서](https://unsloth.ai/docs/basics/inference-and-deployment/saving-to-gguf)를 참고하세요.
    * `q8_0` - 변환이 빠릅니다. 자원 사용량이 높지만 일반적으로 감당할 만합니다.
    * `q4_k_m` - 권장 방식입니다. attention.wv와 feed_forward.w2 텐서의 절반에 Q6_K를 적용하고 나머지에는 Q4_K를 적용합니다.
    * `q5_k_m` - 권장 방식입니다. attention.wv와 feed_forward.w2 텐서의 절반에 Q6_K를 적용하고 나머지에는 Q5_K를 적용합니다.

    모델과 함께 저장되는 채팅 템플릿은 도구 호출 지원을 유지하므로 llama.cpp와 Ollama에서도 도구를 호출할 수 있습니다.
    """)
    return


@app.cell
def _(execution_step_16, model_1, output_dir, tokenizer_1):
    # 원본 코드 셀 순서와 부수 효과를 보존합니다.
    execution_step_17 = execution_step_16 + 1
    # 8비트 Q8_0으로 저장
    if False:
    # https://huggingface.co/settings/tokens 에서 토큰을 발급하세요!
    # hf를 본인의 사용자 이름으로 바꾸는 것도 잊지 마세요!
        model_1.save_pretrained_gguf(str(output_dir / 'qwen_finetune'), tokenizer_1)
    if False:
    # 16비트 GGUF로 저장
        model_1.push_to_hub_gguf('HF_USERNAME/qwen_finetune', tokenizer_1, token='YOUR_HF_TOKEN')
    if False:
        model_1.save_pretrained_gguf(str(output_dir / 'qwen_finetune'), tokenizer_1, quantization_method='f16')
    # q4_k_m GGUF로 저장
    if False:
        model_1.push_to_hub_gguf('HF_USERNAME/qwen_finetune', tokenizer_1, quantization_method='f16', token='YOUR_HF_TOKEN')
    if False:
        model_1.save_pretrained_gguf(str(output_dir / 'qwen_finetune'), tokenizer_1, quantization_method='q4_k_m')
    if False:
        model_1.push_to_hub_gguf('HF_USERNAME/qwen_finetune', tokenizer_1, quantization_method='q4_k_m', token='YOUR_HF_TOKEN')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    이제 llama.cpp에서 `qwen_finetune.Q8_0.gguf` 또는 `qwen_finetune.Q4_K_M.gguf` 파일을 사용하세요.

    이제 모든 과정이 끝났습니다! Unsloth에 관한 질문은 [Discord](https://discord.gg/unsloth) 채널에서 나눌 수 있습니다. 버그를 발견했거나 최신 LLM 소식, 도움, 프로젝트 참여 기회를 찾는다면 자유롭게 참여하세요!

    추가 참고 자료:

    1. 먼저 SFT로 모델에 도구 호출을 가르치기. [도구 호출 SFT 노트북](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Qwen3_(4B)-Tool_Calling.ipynb)
    2. 도구를 사용하는 강화 학습. [GRPO 도구 사용 노트북](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Qwen3_(4B)-GRPO-Tool_Use.ipynb)
    3. 나만의 추론 모델 학습 - Qwen3 GRPO 노트북 [무료 Colab](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Qwen3_(4B)-GRPO.ipynb)
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
