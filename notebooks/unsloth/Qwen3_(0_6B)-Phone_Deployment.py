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
    _run_id = _RunDatetime.now(_RunZoneInfo("Asia/Seoul")).strftime("%Y%m%d-%H%M%S") + "-" + _run_uuid4().hex[:6] + "-Qwen3_(0_6B)-Phone_Deployment"
    output_dir = _RunPath(__file__).resolve().parents[2] / "outputs" / "unsloth" / _run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"결과 저장 위치: {output_dir}")
    execution_step_0 = 0
    return execution_step_0, output_dir


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Qwen3 (0.6B) — 온디바이스 배포
    """)
    return


@app.cell
def _(execution_step_0):
    # 원본 코드 셀 순서와 부수 효과를 보존합니다.
    execution_step_1 = execution_step_0 + 1
    from unsloth import FastLanguageModel
    import torch

    # 휴대전화 배포를 지원하는 모델
    fourbit_models = [
        "unsloth/Qwen3-4B",              # 0.6B, 4B, 8B, 32B 등 모든 Qwen3 모델
        "unsloth/Qwen3-32B",
        "unsloth/Llama-3.1-8B-Instruct", # Llama 3 모델 지원
        "unsloth/Llama-3.3-70B-Instruct",
        "unsloth/gemma-3-270m-it",       # Gemma 3 모델 지원
        "unsloth/gemma-3-27b-it",
        "unsloth/Qwen2.5-7B-Instruct",   # 그 외 다양한 모델도 지원합니다!
        "unsloth/Phi-4-mini-instruct",
    ] # 더 많은 모델: https://huggingface.co/unsloth

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "unsloth/Qwen3-0.6B",
        max_seq_length = 1024,
        full_finetuning = True,
        qat_scheme = "phone-deployment", # 휴대전화 배포용 플래그
    )
    return execution_step_1, model, tokenizer, torch


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <a name="Data"></a>
    ### 데이터 준비
    Qwen3는 추론 모드와 비추론 모드를 모두 제공합니다. 따라서 두 가지 데이터셋을 사용합니다.

    1. [AIMO](https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-2/leaderboard)(AI Mathematical Olympiad - Progress Prize 2) 대회 우승에 사용된 [Open Math Reasoning](https://huggingface.co/datasets/unsloth/OpenMathReasoning-mini) 데이터셋을 사용합니다! DeepSeek R1을 사용해 정확도 95% 초과를 달성한 검증 가능한 추론 과정 중 10%를 샘플링합니다.

    2. ShareGPT 형식의 [Maxime Labonne의 FineTome-100k](https://huggingface.co/datasets/mlabonne/FineTome-100k) 데이터셋도 활용합니다. 이 데이터 역시 Hugging Face의 일반적인 여러 턴 대화 형식으로 변환해야 합니다.
    """)
    return


@app.cell
def _(execution_step_1):
    # 원본 코드 셀 순서와 부수 효과를 보존합니다.
    execution_step_2 = execution_step_1 + 1
    from datasets import load_dataset
    reasoning_dataset = load_dataset("unsloth/OpenMathReasoning-mini", split = "cot")
    non_reasoning_dataset = load_dataset("mlabonne/FineTome-100k", split = "train")
    return execution_step_2, non_reasoning_dataset, reasoning_dataset


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    두 데이터셋의 구조를 확인해 보겠습니다.
    """)
    return


@app.cell
def _(execution_step_2, reasoning_dataset):
    # 원본 코드 셀 순서와 부수 효과를 보존합니다.
    execution_step_3 = execution_step_2 + 1
    reasoning_dataset
    return (execution_step_3,)


@app.cell
def _(execution_step_3, non_reasoning_dataset):
    # 원본 코드 셀 순서와 부수 효과를 보존합니다.
    execution_step_4 = execution_step_3 + 1
    non_reasoning_dataset
    return (execution_step_4,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    이제 추론 데이터셋을 대화 형식으로 변환합니다.
    """)
    return


@app.cell
def _(execution_step_4):
    # 원본 코드 셀 순서와 부수 효과를 보존합니다.
    execution_step_5 = execution_step_4 + 1
    def generate_conversation(examples):
        problems  = examples["problem"]
        solutions = examples["generated_solution"]
        conversations = []
        for problem, solution in zip(problems, solutions):
            conversations.append([
                {"role" : "user",      "content" : problem},
                {"role" : "assistant", "content" : solution},
            ])
        return { "conversations": conversations, }

    return execution_step_5, generate_conversation


@app.cell
def _(execution_step_5, generate_conversation, reasoning_dataset, tokenizer):
    # 원본 코드 셀 순서와 부수 효과를 보존합니다.
    execution_step_6 = execution_step_5 + 1
    reasoning_conversations = tokenizer.apply_chat_template(
        list(reasoning_dataset.map(generate_conversation, batched = True)["conversations"]),
        tokenize = False,
    )
    return execution_step_6, reasoning_conversations


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    변환된 첫 번째 행을 확인해 보겠습니다.
    """)
    return


@app.cell
def _(execution_step_6, reasoning_conversations):
    # 원본 코드 셀 순서와 부수 효과를 보존합니다.
    execution_step_7 = execution_step_6 + 1
    reasoning_conversations[0]
    return (execution_step_7,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    다음으로 비추론 데이터셋도 대화 형식으로 변환합니다.

    먼저 Unsloth의 `standardize_sharegpt` 함수로 데이터셋 형식을 정리해야 합니다.

    채팅 예제의 모든 사용자 메시지에 Qwen3의 `/no_think` 스위치도 추가합니다. 채팅 템플릿은 이 예제들에 빈 `<think></think>` 블록을 넣기 때문에, 이 스위치가 없으면 모델이 `enable_thinking = True`일 때도 사고 과정을 건너뛰도록 학습됩니다.
    """)
    return


@app.cell
def _(execution_step_7, non_reasoning_dataset, tokenizer):
    # 원본 코드 셀 순서와 부수 효과를 보존합니다.
    execution_step_8 = execution_step_7 + 1
    from unsloth.chat_templates import standardize_sharegpt
    dataset = standardize_sharegpt(non_reasoning_dataset)

    def add_no_think(conversation):
        return [
            {**message, "content": message["content"] + " /no_think"} if message["role"] == "user" else message
            for message in conversation
        ]

    non_reasoning_conversations = tokenizer.apply_chat_template(
        [add_no_think(conversation) for conversation in dataset["conversations"]],
        tokenize = False,
    )
    return execution_step_8, non_reasoning_conversations


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    첫 번째 행을 확인해 보겠습니다.
    """)
    return


@app.cell
def _(execution_step_8, non_reasoning_conversations):
    # 원본 코드 셀 순서와 부수 효과를 보존합니다.
    execution_step_9 = execution_step_8 + 1
    non_reasoning_conversations[0]
    return (execution_step_9,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    이제 두 데이터셋의 크기를 확인해 보겠습니다.
    """)
    return


@app.cell
def _(execution_step_9, non_reasoning_conversations, reasoning_conversations):
    # 원본 코드 셀 순서와 부수 효과를 보존합니다.
    execution_step_10 = execution_step_9 + 1
    print(len(reasoning_conversations))
    print(len(non_reasoning_conversations))
    return (execution_step_10,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    비추론 데이터셋이 훨씬 큽니다. 추론 능력도 어느 정도 유지하되, 주로 채팅에 사용할 모델을 만들고 싶다고 가정하겠습니다.

    채팅 전용 데이터의 비율을 정해 두 데이터셋을 혼합합니다.

    추론 데이터 75%, 채팅 데이터 25%로 구성해 보겠습니다.
    """)
    return


@app.cell
def _(execution_step_10):
    # 원본 코드 셀 순서와 부수 효과를 보존합니다.
    execution_step_11 = execution_step_10 + 1
    chat_percentage = 0.25
    return chat_percentage, execution_step_11


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    추론 데이터셋에서 75%(즉, 100% - chat_percentage)를 샘플링합니다.
    """)
    return


@app.cell
def _(
    chat_percentage,
    execution_step_11,
    non_reasoning_conversations,
    reasoning_conversations,
):
    # 원본 코드 셀 순서와 부수 효과를 보존합니다.
    execution_step_12 = execution_step_11 + 1
    import pandas as pd
    non_reasoning_subset = pd.Series(non_reasoning_conversations)
    non_reasoning_subset = non_reasoning_subset.sample(
        int(len(reasoning_conversations)*(chat_percentage/(1 - chat_percentage))),
        random_state = 2407,
    )
    print(len(reasoning_conversations))
    print(len(non_reasoning_subset))
    print(len(non_reasoning_subset) / (len(non_reasoning_subset) + len(reasoning_conversations)))
    return execution_step_12, non_reasoning_subset, pd


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    마지막으로 두 데이터셋을 합칩니다.
    """)
    return


@app.cell
def _(execution_step_12, non_reasoning_subset, pd, reasoning_conversations):
    # 원본 코드 셀 순서와 부수 효과를 보존합니다.
    execution_step_13 = execution_step_12 + 1
    data = pd.concat([
        pd.Series(reasoning_conversations),
        pd.Series(non_reasoning_subset)
    ])
    data.name = "text"

    from datasets import Dataset
    combined_dataset = Dataset.from_pandas(pd.DataFrame(data))
    combined_dataset = combined_dataset.shuffle(seed = 3407)
    return combined_dataset, execution_step_13


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <a name="Train"></a>
    ### 모델 학습
    이제 모델을 학습합니다. 빠르게 실행하기 위해 60스텝만 학습하지만, 전체 학습을 실행하려면 `num_train_epochs=1`로 설정하고 `max_steps=None`으로 스텝 제한을 해제하면 됩니다.
    """)
    return


@app.cell
def _(combined_dataset, execution_step_13, model, output_dir, tokenizer):
    # 원본 코드 셀 순서와 부수 효과를 보존합니다.
    execution_step_14 = execution_step_13 + 1
    from trl import SFTTrainer, SFTConfig
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = combined_dataset,
        eval_dataset = None, # 평가도 설정할 수 있습니다!
        args = SFTConfig(
            output_dir = str(output_dir / "checkpoints"),
            dataset_text_field = "text",
            per_device_train_batch_size = 2,
            gradient_accumulation_steps = 4, # 그래디언트 누적으로 더 큰 배치 크기를 구현하세요!
            warmup_steps = 5,
            # num_train_epochs = 1, # 전체 데이터를 한 번 학습하려면 설정하세요.
            max_steps = 100,
            learning_rate = 5e-5,
            logging_steps = 1,
            optim = "adamw_8bit",
            weight_decay = 0.001,
            lr_scheduler_type = "linear",
            seed = 3407,
            report_to = "none", # TrackIO/WandB 등을 사용하려면 설정하세요
        ),
    )
    return execution_step_14, trainer


@app.cell
def _(execution_step_14, torch):
    # 원본 코드 셀 순서와 부수 효과를 보존합니다.
    execution_step_15 = execution_step_14 + 1
    # @title 현재 메모리 사용 통계 표시
    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
    print(f"{start_gpu_memory} GB of memory reserved.")
    return execution_step_15, max_memory, start_gpu_memory


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    모델을 학습해 보겠습니다! 이전 학습을 재개하려면 `trainer.train(resume_from_checkpoint = True)`로 설정하세요.
    """)
    return


@app.cell
def _(execution_step_15, trainer):
    # 원본 코드 셀 순서와 부수 효과를 보존합니다.
    execution_step_16 = execution_step_15 + 1
    trainer_stats = trainer.train()
    return execution_step_16, trainer_stats


@app.cell
def _(execution_step_16, max_memory, start_gpu_memory, torch, trainer_stats):
    # 원본 코드 셀 순서와 부수 효과를 보존합니다.
    execution_step_17 = execution_step_16 + 1
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
    return (execution_step_17,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <a name="Save"></a>
    ### 미세 조정 모델 저장 및 불러오기

    ExecuTorch의 Qwen3 변환기는 양자화되지 않은 일반 가중치를 입력으로 받습니다.
    학습한 QAT 가중치는 유지하면서 가짜 양자화 모듈을 일반 층으로 되돌린 뒤,
    16비트 체크포인트를 저장합니다. 최종 int8 활성화·int4 가중치 양자화는
    ExecuTorch 내보내기 단계에서 학습 때와 같은 설정으로 적용합니다.
    """)
    return


@app.cell
def _(execution_step_17, model, output_dir, tokenizer):
    # 원본 코드 셀 순서와 부수 효과를 보존합니다.
    execution_step_18 = execution_step_17 + 1
    from torchao.quantization import quantize_
    from torchao.quantization.qat import QATConfig

    # base_config 없이 convert하면 가짜 양자화 모듈만 일반 층으로 되돌립니다.
    quantize_(model, QATConfig(step = "convert"))
    model.save_pretrained_merged(str(output_dir / 'phone_model'), tokenizer, save_method = "merged_16bit")
    return (execution_step_18,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    그런 다음 ExecuTorch의 Qwen3 변환 과정을 실행합니다.
    """)
    return


@app.cell
def _(execution_step_18, output_dir):
    # 원본 코드 셀 순서와 부수 효과를 보존합니다.
    execution_step_19 = execution_step_18 + 1
    # 실패하면 다음 셀로 넘어가지 않도록 check=True로 실행합니다.
    import subprocess
    import sys
    from pathlib import Path

    subprocess.run([
        sys.executable, "-m", "executorch.examples.models.qwen3.convert_weights",
        str(output_dir / 'phone_model'), str(output_dir / 'pytorch_model_converted.bin'),
    ], check = True)
    assert Path(str(output_dir / 'pytorch_model_converted.bin')).stat().st_size > 0
    return Path, execution_step_19, subprocess, sys


@app.cell
def _(Path, execution_step_19, model, output_dir):
    # 원본 코드 셀 순서와 부수 효과를 보존합니다.
    execution_step_20 = execution_step_19 + 1
    # wheel에 모델별 JSON이 빠져 있어 학습 모델의 설정에서 직접 생성합니다.
    import json

    assert model.config.model_type == "qwen3"
    rope_params = getattr(model.config, "rope_parameters", None) or {}
    params = {
        "dim": model.config.hidden_size,
        "ffn_dim_multiplier": 1,
        "hidden_dim": model.config.intermediate_size,
        "n_heads": model.config.num_attention_heads,
        "head_dim": model.config.head_dim,
        "n_kv_heads": model.config.num_key_value_heads,
        "n_layers": model.config.num_hidden_layers,
        "norm_eps": model.config.rms_norm_eps,
        "rope_theta": rope_params.get("rope_theta", getattr(model.config, "rope_theta", 1000000.0)),
        "use_scaled_rope": False,
        "vocab_size": model.config.vocab_size,
        "use_hf_rope": True,
        "attention_qkv_bias": model.config.attention_bias,
        "use_qk_norm": True,
        "qk_norm_before_rope": True,
    }
    Path(str(output_dir / '0.6B_config.json')).write_text(json.dumps(params, indent = 2))
    return execution_step_20, json


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    마지막으로 배포에 사용할 .pte 파일로 내보냅니다. QAT 설정과 동일하게
    선형 층은 동적 int8 활성화·int4 가중치, 그룹 크기 32, HQQ 비활성화를 적용하고,
    임베딩은 8비트로 양자화합니다. BOS/EOS 메타데이터는 학습한 모델의 토크나이저에서
    가져오며, ExecuTorch의 현재 `export_llm` 진입점을 사용합니다.
    """)
    return


@app.cell
def _(
    Path,
    execution_step_20,
    json,
    model,
    output_dir,
    subprocess,
    sys,
    tokenizer,
):
    # 원본 코드 셀 순서와 부수 효과를 보존합니다.
    execution_step_21 = execution_step_20 + 1
    import yaml
    bos_id = tokenizer.bos_token_id
    if bos_id is None:
        bos_id = tokenizer.convert_tokens_to_ids('<|endoftext|>')
    eos_ids = model.generation_config.eos_token_id
    if isinstance(eos_ids, int):
        eos_ids = [eos_ids]
    assert bos_id is not None and 0 <= bos_id < model.config.vocab_size
    assert eos_ids and all((0 <= token_id < model.config.vocab_size for token_id in eos_ids))
    export_config = {'base': {'model_class': 'qwen3_0_6b', 'checkpoint': str(Path(str(output_dir / 'pytorch_model_converted.bin')).resolve()), 'params': str(Path(str(output_dir / '0.6B_config.json')).resolve()), 'metadata': json.dumps({'get_bos_id': bos_id, 'get_eos_ids': eos_ids})}, 'model': {'use_kv_cache': True, 'use_sdpa_with_kv_cache': True, 'dtype_override': 'fp32'}, 'export': {'output_dir': str(output_dir), 'output_name': 'qwen3_0.6B_model.pte', 'max_context_length': 1024, 'max_seq_length': 128}, 'quantization': {'qmode': '8da4w', 'group_size': 32, 'use_hqq': False, 'embedding_quantize': '8,0'}, 'backend': {'xnnpack': {'enabled': True, 'extended_ops': True}}}
    Path(str(output_dir / 'executorch_export_config.yaml')).write_text(yaml.safe_dump(export_config, sort_keys=False))
    subprocess.run([sys.executable, '-m', 'executorch.extension.llm.export.export_llm', '--config', str(Path(str(output_dir / 'executorch_export_config.yaml')).resolve())], check=True)
    return bos_id, eos_ids, execution_step_21


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    내보낸 `qwen3_0.6B_model.pte`의 크기와 BOS/EOS 메타데이터를 확인합니다.
    현재 Python wheel은 LLM·양자화 커널을 자동 등록하지 않으므로 포함된 라이브러리를
    명시적으로 불러옵니다. CPU 런타임에서 토큰 두 개를 순차 처리해 KV 캐시와 유한한
    로짓을 확인합니다. 휴대전화 실기기에서의 속도와 동작은 별도 검증이 필요합니다.
    """)
    return


@app.cell
def _(Path, bos_id, eos_ids, execution_step_21, model, output_dir, torch):
    # 원본 코드 셀 순서와 부수 효과를 보존합니다.
    execution_step_22 = execution_step_21 + 1
    exported_model = Path(str(output_dir / 'qwen3_0.6B_model.pte'))
    assert exported_model.is_file() and exported_model.stat().st_size > 0
    print(f"{exported_model.name}: {exported_model.stat().st_size / 1024**2:.1f} MiB")

    from importlib.resources import files
    from executorch.runtime import Runtime

    # wheel에 포함된 LLM 및 양자화 커널을 런타임에 등록합니다.
    for package, library in [
        ("executorch.extension.llm.custom_ops", "libcustom_ops_aot_lib.so"),
        ("executorch.kernels.quantized", "libquantized_ops_aot_lib.so"),
    ]:
        torch.ops.load_library(str(files(package).joinpath(library)))

    torch.set_num_threads(2)
    program = Runtime.get().load_program(str(exported_model.resolve()))
    assert int(program.load_method("get_bos_id").execute([])[0]) == bos_id
    assert list(map(int, program.load_method("get_eos_ids").execute([]))) == eos_ids
    forward = program.load_method("forward")
    assert forward is not None
    next_token = bos_id
    for position in range(2):
        logits = forward.execute([
            torch.tensor([[next_token]], dtype = torch.int64),
            torch.tensor([position], dtype = torch.int64),
        ])[0]
        assert logits.shape[-1] == model.config.vocab_size
        assert bool(torch.isfinite(logits).all())
        next_token = int(logits.reshape(-1).argmax())
    print("ExecuTorch CPU 순전파 및 KV 캐시 검증 통과:", tuple(logits.shape))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
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
