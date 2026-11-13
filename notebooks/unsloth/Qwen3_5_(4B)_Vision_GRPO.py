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
    _run_id = _RunDatetime.now(_RunZoneInfo("Asia/Seoul")).strftime("%Y%m%d-%H%M%S") + "-" + _run_uuid4().hex[:6] + "-Qwen3_5_(4B)_Vision_GRPO"
    output_dir = _RunPath(__file__).resolve().parents[2] / "outputs" / "unsloth" / _run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"결과 저장 위치: {output_dir}")
    execution_step_0 = 0
    return execution_step_0, output_dir


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Qwen3.5 (4B) — 비전 GRPO

    Unsloth에서 `GSPO`를 사용하는 방법도 소개합니다!

    이 노트북의 목표는 아래와 같은 이미지 입력을 받아 수학 문제를 풀도록 비전 언어 모델을 강화 학습하는 것입니다.

    <img src="https://raw.githubusercontent.com/lupantech/MathVista/main/assets/our_new_3_datasets.png" alt="수학 문제 이미지 예시" height="256">
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
 
    """)
    return


@app.cell
def _(execution_step_0):
    # 원본 코드 셀 순서와 부수 효과를 보존합니다.
    execution_step_1 = execution_step_0 + 1
    from unsloth import FastVisionModel
    import torch
    max_seq_length = 16384 # VLM에는 이 정도 길이가 필요합니다
    lora_rank = 16 # 순위가 높을수록 표현력이 좋아지지만 느려집니다

    model, tokenizer = FastVisionModel.from_pretrained(
        model_name = "unsloth/Qwen3.5-4B",
        max_seq_length = max_seq_length,
        load_in_4bit = False, # 16비트 LoRA를 사용하려면 False
        fast_inference = False, # vLLM 고속 추론 활성화
    )
    return FastVisionModel, execution_step_1, model, tokenizer


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Unsloth는 vLLM의 가중치를 직접 공유해 VRAM 사용량을 50% 넘게 줄입니다. vLLM은 아직 비전 층의 LoRA를 지원하지 않으므로 언어 층에만 LoRA를 추가할 수 있습니다. 그래도 비전 GRPO는 작동합니다!
    """)
    return


@app.cell
def _(FastVisionModel, execution_step_1, model):
    # 원본 코드 셀 순서와 부수 효과를 보존합니다.
    execution_step_2 = execution_step_1 + 1
    model_1 = FastVisionModel.get_peft_model(model, finetune_vision_layers=False, finetune_language_layers=True, finetune_attention_modules=True, finetune_mlp_modules=True, r=16, lora_alpha=16, lora_dropout=0, bias='none', random_state=3407, use_rslora=False, loftq_config=None, use_gradient_checkpointing='unsloth')  # 비전 층을 미세 조정하지 않으면 False  # 언어 층을 미세 조정하지 않으면 False  # 어텐션 층을 미세 조정하지 않으면 False  # MLP 층을 미세 조정하지 않으면 False  # 클수록 정확도가 높아지지만 과적합될 수 있습니다  # alpha는 최소한 r과 같은 값으로 설정하는 것을 권장합니다  # 순위 안정화 LoRA를 지원합니다  # LoftQ도 지원합니다  # 메모리 사용량 감소  # target_modules = "all-linear", # 이제 선택 사항입니다! 필요하면 목록을 지정하세요
    return execution_step_2, model_1


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 데이터 준비
    <a name="Data"></a>

    `AI4Math/MathVista`는 이미지를 사용해 논리 및 수학 문제를 푸는 데이터셋입니다.

    이 노트북에서는 단순하게 진행하기 위해 답이 숫자인 수학 문제만 사용합니다.
    """)
    return


@app.cell
def _(execution_step_2):
    # 원본 코드 셀 순서와 부수 효과를 보존합니다.
    execution_step_3 = execution_step_2 + 1
    from datasets import load_dataset
    from trl import GRPOConfig, GRPOTrainer

    dataset = load_dataset("AI4Math/MathVista", split = "testmini")
    return GRPOConfig, GRPOTrainer, dataset, execution_step_3


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    데이터셋을 필터링해 답이 실수 또는 숫자인 예제만 남깁니다.
    """)
    return


@app.cell
def _(dataset, execution_step_3):
    # 원본 코드 셀 순서와 부수 효과를 보존합니다.
    execution_step_4 = execution_step_3 + 1

    def is_numeric_answer(example):
        try:
            float(example['answer'])
            return True
        except:
            return False
    dataset_1 = dataset.filter(is_numeric_answer)
    return dataset_1, execution_step_4


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    이미지가 문맥 길이를 과도하게 차지하지 않도록 크기를 512 × 512픽셀로 조정합니다. 학습과 호환되도록 RGB로도 변환합니다!
    """)
    return


@app.cell
def _(dataset_1, execution_step_4):
    # 원본 코드 셀 순서와 부수 효과를 보존합니다.
    execution_step_5 = execution_step_4 + 1
    # 크기를 (512, 512)로 조정

    def resize_images(example):
        image = example['decoded_image']
        image = image.resize((512, 512))
        example['decoded_image'] = image
        return example
    dataset_2 = dataset_1.map(resize_images)
    # 그다음 RGB로 변환

    def convert_to_rgb(example):
        image = example['decoded_image']
        if image.mode != 'RGB':
            image = image.convert('RGB')
        example['decoded_image'] = image
        return example
    dataset_2 = dataset_2.map(convert_to_rgb)
    return dataset_2, execution_step_5


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    그런 다음 강화 학습용 데이터셋 배치를 구성하는 데 필요한 대화 템플릿을 만듭니다.
    """)
    return


@app.cell
def _(dataset_2, execution_step_5):
    # 원본 코드 셀 순서와 부수 효과를 보존합니다.
    execution_step_6 = execution_step_5 + 1
    # 이해하고 수정하기 쉽도록 구분자 변수를 정의합니다
    REASONING_START = '<REASONING>'
    REASONING_END = '</REASONING>'
    SOLUTION_START = '<SOLUTION>'
    SOLUTION_END = '</SOLUTION>'

    def make_conversation(example):
        text_content = f"{example['question']}. Also first provide your reasoning or working out on how you would go about solving the question between {REASONING_START} and {REASONING_END} and then your final answer between {SOLUTION_START} and (put a single float here) {SOLUTION_END}"  # 전역으로 정의되어 있지 않으면 자리표시자 상수를 정의합니다
        prompt = [{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': text_content}]}]  # 사용자의 텍스트 프롬프트
        return {'prompt': prompt, 'image': example['decoded_image'], 'answer': example['answer']}
    train_dataset = dataset_2.map(make_conversation)
    train_dataset = train_dataset.remove_columns('image')
    # decoded_images에 실제 이미지가 있으므로 데이터셋을 이 형식으로 바꿉니다
    # "image": example["decoded_image"]만으로는 데이터셋 형식이 올바르게 구성되지 않습니다
    # 1. 원래의 'image' 열 제거
    # 2. 'decoded_image'를 'image'로 이름 변경
    train_dataset = train_dataset.rename_column('decoded_image', 'image')  # 원하는 멀티모달 형식으로 프롬프트 구성  # 이미지 자리표시자  # 프롬프트의 텍스트 부분  # 실제 이미지 데이터는 프로세서용으로 별도 유지
    return (
        REASONING_END,
        REASONING_START,
        SOLUTION_END,
        SOLUTION_START,
        execution_step_6,
        train_dataset,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    이제 전체 데이터셋에 채팅 템플릿을 적용합니다.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 보상 함수

    추론 과정이 시작하고 끝나는지 확인하는 기본 형식 보상 함수와 답을 올바르게 작성했는지 확인하는 보상 함수를 정의합니다.

    [블로그 글](https://unsloth.ai/docs/new/vision-reinforcement-learning-vlm-rl#qwen-2.5-vl-vision-rl-issues-and-quirks)에 설명된 `addCriterion` 문제도 해결을 시도합니다.
    """)
    return


@app.cell
def _(
    REASONING_END,
    REASONING_START,
    SOLUTION_END,
    SOLUTION_START,
    execution_step_6,
):
    # 원본 코드 셀 순서와 부수 효과를 보존합니다.
    execution_step_7 = execution_step_6 + 1
    # 보상 함수
    import re

    def formatting_reward_func(completions, **kwargs):
        import re
        thinking_pattern = f'{REASONING_START}(.*?){REASONING_END}'
        answer_pattern = f'{SOLUTION_START}(.*?){SOLUTION_END}'
        scores = []
        for completion in completions:
            if isinstance(completion, list):
                completion = completion[0]['content'] if completion else ''
            score = 0
            thinking_matches = re.findall(thinking_pattern, completion, re.DOTALL)
            answer_matches = re.findall(answer_pattern, completion, re.DOTALL)
            if len(thinking_matches) == 1:
                score = score + 1.0
            if len(answer_matches) == 1:
                score = score + 1.0
            if len(completion) != 0:
                removal = completion.replace('addCriterion', '').replace('\n', '')
                if (len(completion) - len(removal)) / len(completion) >= 0.5:  # addCriterion 문제 보정
                    score = score - 2.0  # 참고: https://unsloth.ai/docs/new/vision-reinforcement-learning-vlm-rl#qwen-2.5-vl-vision-rl-issues-and-quirks
            scores.append(score)  # 과도한 addCriterion 및 줄바꿈에 감점
        return scores

    def correctness_reward_func(prompts, completions, answer, **kwargs):
        answer_pattern = f'{SOLUTION_START}(.*?){SOLUTION_END}'
        completions = [(c[0]['content'] if c else '') if isinstance(c, list) else c for c in completions]
        responses = [re.findall(answer_pattern, completion, re.DOTALL) for completion in completions]
        q = prompts[0]
        print('-' * 20, f'Question:\n{q}', f'\nAnswer:\n{answer[0]}', f'\nResponse:{completions[0]}')
        return [2.0 if len(r) == 1 and a == r[0].replace('\n', '') else 0.0 for r, a in zip(responses, answer)]

    return correctness_reward_func, execution_step_7, formatting_reward_func


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    데이터셋의 첫 번째 예제 프롬프트입니다.
    """)
    return


@app.cell
def _(execution_step_7, train_dataset):
    # 원본 코드 셀 순서와 부수 효과를 보존합니다.
    execution_step_8 = execution_step_7 + 1
    train_dataset[0]["prompt"]
    return (execution_step_8,)


@app.cell
def _(execution_step_8, train_dataset):
    # 원본 코드 셀 순서와 부수 효과를 보존합니다.
    execution_step_9 = execution_step_8 + 1
    train_dataset[100]["prompt"]
    return (execution_step_9,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <a name="Inference"></a>
    ### 추론
    학습하기 전에 학습 데이터셋의 100번째 샘플로 모델을 실행해 보겠습니다.
    """)
    return


@app.cell
def _(execution_step_9, model_1, tokenizer, train_dataset):
    execution_step_10 = execution_step_9 + 1
    _image = train_dataset[100]['image']
    _prompt = train_dataset[100]['prompt']
    _inputs = tokenizer(_image, _prompt, add_special_tokens=False, return_tensors='pt').to('cuda')
    from transformers import TextStreamer
    _text_streamer = TextStreamer(tokenizer, skip_prompt=True)
    _ = model_1.generate(**_inputs, streamer=_text_streamer, max_new_tokens=1024, use_cache=True, temperature=1.0, min_p=0.1)
    return TextStreamer, execution_step_10


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <a name="Train"></a>
    ### 모델 학습

    이제 `GRPO` 트레이너와 모든 설정을 준비합니다! 실제로는 `GSPO`도 활성화한다는 점에 주목하세요!
    """)
    return


@app.cell
def _(GRPOConfig, execution_step_10, output_dir):
    # 원본 코드 셀 순서와 부수 효과를 보존합니다.
    execution_step_11 = execution_step_10 + 1
    training_args = GRPOConfig(learning_rate=5e-06, adam_beta1=0.9, adam_beta2=0.99, weight_decay=0.1, warmup_ratio=0.1, lr_scheduler_type='cosine', optim='adamw_8bit', logging_steps=1, log_completions=False, per_device_train_batch_size=1, gradient_accumulation_steps=1, num_generations=2, max_prompt_length=1024, max_completion_length=1024, max_steps=60, save_steps=60, max_grad_norm=0.1, report_to='none', output_dir=str(output_dir / 'checkpoints'), importance_sampling_level='sequence', mask_truncated_completions=False, loss_type='dr_grpo')  # 더 안정적인 학습을 위해 4로 늘리세요  # 메모리가 부족하면 줄이세요  # num_train_epochs = 0.5, # 전체 데이터를 한 번 학습하려면 1로 설정하세요  # Weights & Biases 사용 가능  # 아래 설정으로 GSPO 활성화:
    return execution_step_11, training_args


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

    추론 중 `addCriterion`이나 알아보기 어려운 출력이 나타날 수 있습니다. 원인은 [블로그 글](https://unsloth.ai/docs/new/vision-reinforcement-learning-vlm-rl#qwen-2.5-vl-vision-rl-issues-and-quirks)을 참고하세요. 원본 설명에 따르면 모델 자체의 특성으로 보이며 무시해도 됩니다.
    """)
    return


@app.cell
def _(
    GRPOTrainer,
    correctness_reward_func,
    execution_step_11,
    formatting_reward_func,
    model_1,
    tokenizer,
    train_dataset,
    training_args,
):
    # 원본 코드 셀 순서와 부수 효과를 보존합니다.
    execution_step_12 = execution_step_11 + 1
    trainer = GRPOTrainer(model=model_1, args=training_args, processing_class=tokenizer, reward_funcs=[formatting_reward_func, correctness_reward_func], train_dataset=train_dataset)
    trainer.train()  # 멀티모달 입력을 처리하도록 프로세서 전달
    return (execution_step_12,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <a name="Inference"></a>
    ### 추론
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    모델을 실행해 보겠습니다! 지시문과 입력을 바꿔도 됩니다.
    """)
    return


@app.cell
def _(TextStreamer, execution_step_12, model_1, tokenizer, train_dataset):
    execution_step_13 = execution_step_12 + 1
    _image = train_dataset[165]['image']
    _prompt = train_dataset[165]['prompt']
    _inputs = tokenizer(_image, _prompt, add_special_tokens=False, return_tensors='pt').to('cuda')
    _text_streamer = TextStreamer(tokenizer, skip_prompt=True)
    _ = model_1.generate(**_inputs, streamer=_text_streamer, max_new_tokens=1024, use_cache=True, temperature=1.0, min_p=0.1)
    return (execution_step_13,)


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
def _(execution_step_13, model_1, output_dir, tokenizer):
    # 원본 코드 셀 순서와 부수 효과를 보존합니다.
    execution_step_14 = execution_step_13 + 1
    model_1.save_pretrained(str(output_dir / 'qwen_lora'))  # 로컬에 저장
    # model.push_to_hub("your_name/qwen_lora", token = "YOUR_HF_TOKEN") # 온라인에 저장
    # tokenizer.push_to_hub("your_name/qwen_lora", token = "YOUR_HF_TOKEN") # 온라인에 저장
    tokenizer.save_pretrained(str(output_dir / 'qwen_lora'))
    return (execution_step_14,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    LoRA가 실제로 학습되었는지 확인합니다!
    """)
    return


@app.cell
def _(execution_step_14, output_dir):
    # 원본 코드 셀 순서와 부수 효과를 보존합니다.
    execution_step_15 = execution_step_14 + 1
    from safetensors import safe_open

    tensors = {}
    with safe_open(str(output_dir / 'qwen_lora/adapter_model.safetensors'), framework = "pt") as f:
        # A와 B 모두 0이 아닌지 확인
        for key in f.keys():
            tensor = f.get_tensor(key)
            n_zeros = (tensor == 0).sum() / tensor.numel()
            assert(n_zeros.item() != tensor.numel())
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
def _(execution_step_15, model_1, output_dir, tokenizer):
    # 원본 코드 셀 순서와 부수 효과를 보존합니다.
    execution_step_16 = execution_step_15 + 1
    # 16비트로 병합
    if False:
        model_1.save_pretrained_merged(str(output_dir / 'qwen_finetune_16bit'), tokenizer, save_method='merged_16bit')
    if False:
    # 4비트로 병합
        model_1.push_to_hub_merged('HF_USERNAME/qwen_finetune_16bit', tokenizer, save_method='merged_16bit', token='YOUR_HF_TOKEN')
    if False:
        model_1.save_pretrained_merged(str(output_dir / 'qwen_finetune_4bit'), tokenizer, save_method='merged_4bit')
    # LoRA 어댑터만 저장
    if False:
        model_1.push_to_hub_merged('HF_USERNAME/qwen_finetune_4bit', tokenizer, save_method='merged_4bit', token='YOUR_HF_TOKEN')
    if False:
        model_1.save_pretrained(str(output_dir / 'qwen_lora'))
        tokenizer.save_pretrained(str(output_dir / 'qwen_lora'))
    if False:
        model_1.push_to_hub('HF_USERNAME/qwen_lora', token='YOUR_HF_TOKEN')
        tokenizer.push_to_hub('HF_USERNAME/qwen_lora', token='YOUR_HF_TOKEN')
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

    [**신규**] 미세 조정 후 Ollama로 자동으로 내보내려면 [Ollama 노트북](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3_(8B)-Ollama.ipynb)을 사용해 보세요.
    """)
    return


@app.cell
def _(execution_step_16, model_1, output_dir, tokenizer):
    # 원본 코드 셀 순서와 부수 효과를 보존합니다.
    execution_step_17 = execution_step_16 + 1
    # 8비트 Q8_0으로 저장
    if False:
    # https://huggingface.co/settings/tokens 에서 토큰을 발급하세요!
    # hf를 본인의 사용자 이름으로 바꾸는 것도 잊지 마세요!
        model_1.save_pretrained_gguf(str(output_dir / 'qwen_finetune'), tokenizer)
    if False:
    # 16비트 GGUF로 저장
        model_1.push_to_hub_gguf('HF_USERNAME/qwen_finetune', tokenizer, token='YOUR_HF_TOKEN')
    if False:
        model_1.save_pretrained_gguf(str(output_dir / 'qwen_finetune'), tokenizer, quantization_method='f16')
    # q4_k_m GGUF로 저장
    if False:
        model_1.push_to_hub_gguf('HF_USERNAME/qwen_finetune', tokenizer, quantization_method='f16', token='YOUR_HF_TOKEN')
    if False:
    # 여러 GGUF 형식으로 저장 - 여러 형식이 필요할 때 훨씬 빠릅니다!
        model_1.save_pretrained_gguf(str(output_dir / 'qwen_finetune'), tokenizer, quantization_method='q4_k_m')
    if False:
        model_1.push_to_hub_gguf('HF_USERNAME/qwen_finetune', tokenizer, quantization_method='q4_k_m', token='YOUR_HF_TOKEN')  # hf를 본인의 사용자 이름으로 바꾸세요!
    if False:
        model_1.push_to_hub_gguf('HF_USERNAME/qwen_finetune', tokenizer, quantization_method=['q4_k_m', 'q8_0', 'q5_k_m'], token='YOUR_HF_TOKEN')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    이 노트북 제작과 Unsloth의 VLM GRPO 도입에 도움을 준 [GAD-Cell](https://github.com/GAD-cell)에게 특별히 감사드립니다!
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

      이 노트북을 포함한 모든 Unsloth 노트북의 라이선스는 [LGPL-3.0](https://github.com/unslothai/notebooks?tab=LGPL-3.0-1-ov-file#readme)입니다
    """)
    return


if __name__ == "__main__":
    app.run()
