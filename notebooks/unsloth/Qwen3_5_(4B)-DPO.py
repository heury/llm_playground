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
    _run_id = _RunDatetime.now(_RunZoneInfo("Asia/Seoul")).strftime("%Y%m%d-%H%M%S") + "-" + _run_uuid4().hex[:6] + "-Qwen3_5_(4B)-DPO"
    output_dir = _RunPath(__file__).resolve().parents[2] / "outputs" / "unsloth" / _run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"결과 저장 위치: {output_dir}")
    execution_step_0 = 0
    return execution_step_0, output_dir


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Qwen3.5 (4B) — DPO

    베이스 모델: `unsloth/Qwen3.5-4B`. UltraFeedback 선호 응답 쌍으로 4비트 LoRA DPO를 수행합니다.
    원본의 학습 데이터 0.5%, LoRA rank 64, 3에포크 설정을 유지합니다.

    `unsloth/`에서 `uv sync --locked`로 환경을 준비하고 `.venv/bin/python` 커널을 선택하세요.
    학습에는 호환되는 NVIDIA GPU가 필요합니다. 이 사본은 형식·구문 및 채팅 형식만 검증했으며 GPU 학습은 실행하지 않았습니다.

    출처: [Unsloth Zephyr (7B) DPO 원본](https://github.com/unslothai/notebooks/blob/main/nb/Zephyr_%287B%29-DPO.ipynb).
    원본 Unsloth 노트북의 [LGPL-3.0 라이선스](https://github.com/unslothai/notebooks/blob/main/LICENSE)를 유지합니다.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Unsloth
    """)
    return


@app.cell
def _(execution_step_0):
    # 원본 코드 셀 순서와 부수 효과를 보존합니다.
    execution_step_1 = execution_step_0 + 1
    # One must patch the DPO Trainer first!
    from unsloth import PatchDPOTrainer

    PatchDPOTrainer()
    return PatchDPOTrainer, execution_step_1


@app.cell
def _(execution_step_1):
    # 원본 코드 셀 순서와 부수 효과를 보존합니다.
    execution_step_2 = execution_step_1 + 1
    from unsloth import FastLanguageModel, is_bfloat16_supported
    from datetime import datetime
    from pathlib import Path
    from zoneinfo import ZoneInfo
    import torch
    max_seq_length = 4096 # Choose any! We auto support RoPE Scaling internally!
    dtype = torch.bfloat16 if is_bfloat16_supported() else torch.float32 # Qwen3.5는 FP16 대신 BF16/FP32 사용
    load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "unsloth/Qwen3.5-4B",
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
        # token = "YOUR_HF_TOKEN", # HF Token for gated models
    )
    # Qwen3.5 로더가 processor를 반환하는 경우 텍스트 토크나이저를 사용합니다.
    tokenizer = getattr(tokenizer, "tokenizer", tokenizer)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return FastLanguageModel, dtype, execution_step_2, model, tokenizer, torch


@app.cell
def _(execution_step_2):
    # 원본 코드 셀 순서와 부수 효과를 보존합니다.
    execution_step_3 = execution_step_2 + 1
    # Alignment Handbook 데이터 로딩 유틸리티와 Qwen 채팅 형식
    import os
    from typing import List, Optional
    from datasets import DatasetDict, concatenate_datasets, load_dataset, load_from_disk
    from datasets.builder import DatasetGenerationError


    def apply_chat_template(example, tokenizer, task="dpo"):
        if task != "dpo":
            raise ValueError("이 노트북은 DPO 선호 쌍을 처리합니다.")
        chosen = example["chosen"]
        rejected = example["rejected"]
        if (len(chosen) < 2 or len(rejected) < 2
            or chosen[-1]["role"] != "assistant"
            or rejected[-1]["role"] != "assistant"
            or chosen[:-1] != rejected[:-1]):
            raise ValueError("chosen/rejected는 동일한 대화 문맥과 마지막 assistant 답변이 필요합니다.")

        # Zephyr 전용 역할 구분자 대신 Qwen3.5 자체 템플릿을 사용합니다.
        # UltraFeedback 답변에는 추론 과정이 없으므로 non-thinking 모드로 맞춥니다.
        kwargs = dict(tokenize=False, enable_thinking=False)
        prompt = tokenizer.apply_chat_template(
            chosen[:-1], add_generation_prompt=True, **kwargs,
        )
        result = {"text_prompt": prompt}
        for key, messages in (("text_chosen", chosen), ("text_rejected", rejected)):
            full = tokenizer.apply_chat_template(
                messages, add_generation_prompt=False, **kwargs,
            )
            if not full.startswith(prompt):
                raise ValueError("Qwen 채팅 템플릿의 prompt/응답 경계가 일치하지 않습니다.")
            completion = full[len(prompt):]
            # TRL이 EOS를 한 번 추가하므로 템플릿의 마지막 EOS와 개행을 제거합니다.
            completion = completion.rstrip("\n")
            if not completion.endswith(tokenizer.eos_token):
                raise ValueError("assistant 답변의 EOS 토큰이 없습니다.")
            completion = completion[:-len(tokenizer.eos_token)]
            if not completion.strip():
                raise ValueError("빈 assistant 답변은 DPO에 사용할 수 없습니다.")
            result[key] = completion
        return result


    def get_datasets(
        data_config: dict,
        splits: List[str] = ["train", "test"],
        shuffle: bool = True,
    ) -> DatasetDict:
        """
        Loads one or more datasets with varying training set proportions.

        Args:
            data_config (`DataArguments` or `dict`):
                Dataset configuration and split proportions.
            splits (`List[str]`, *optional*, defaults to `['train', 'test']`):
                Dataset splits to load and mix. Assumes the splits exist in all datasets and have a `train_` or `test_` prefix.
            shuffle (`bool`, *optional*, defaults to `True`):
                Whether to shuffle the training and testing/validation data.

        Returns
            [`DatasetDict`]: The dataset dictionary containing the loaded datasets.
        """

        if type(data_config) is dict:
            # Structure of the input is:
            #     dataset_mixer = {
            #             "dataset1": 0.5,
            #             "dataset1": 0.3,
            #             "dataset1": 0.2,
            #         }
            dataset_mixer = data_config
        else:
            raise ValueError(f"Data config {data_config} not recognized.")

        raw_datasets = mix_datasets(dataset_mixer, splits = splits, shuffle = shuffle)
        return raw_datasets


    def mix_datasets(
        dataset_mixer: dict, splits: Optional[List[str]] = None, shuffle = True
    ) -> DatasetDict:
        """
        Loads and mixes datasets according to proportions specified in `dataset_mixer`.

        Args:
            dataset_mixer (`dict`):
                Dictionary containing the dataset names and their training proportions. By default, all test proportions are 1.
            splits (Optional[List[str]], *optional*, defaults to `None`):
                Dataset splits to load and mix. Assumes the splits exist in all datasets and have a `train_` or `test_` prefix.
            shuffle (`bool`, *optional*, defaults to `True`):
                Whether to shuffle the training and testing/validation data.
        """
        raw_datasets = DatasetDict()
        raw_train_datasets = []
        raw_val_datasets = []
        fracs = []
        for ds, frac in dataset_mixer.items():
            fracs.append(frac)
            for split in splits:
                try:
                    # Try first if dataset on a Hub repo
                    dataset = load_dataset(ds, split = split)
                except DatasetGenerationError:
                    # If not, check local dataset
                    dataset = load_from_disk(os.path.join(ds, split))

                if "train" in split:
                    raw_train_datasets.append(dataset)
                elif "test" in split:
                    raw_val_datasets.append(dataset)
                else:
                    raise ValueError(
                        f"Split type {split} not recognized as one of test or train."
                    )

        if any(frac < 0 for frac in fracs):
            raise ValueError("Dataset fractions cannot be negative.")

        if len(raw_train_datasets) > 0:
            train_subsets = []
            for dataset, frac in zip(raw_train_datasets, fracs):
                train_subset = dataset.select(range(int(frac * len(dataset))))
                train_subsets.append(train_subset)
            if shuffle:
                raw_datasets["train"] = concatenate_datasets(train_subsets).shuffle(seed = 42)
            else:
                raw_datasets["train"] = concatenate_datasets(train_subsets)
        # No subsampling for test datasets to enable fair comparison across models
        if len(raw_val_datasets) > 0:
            if shuffle:
                raw_datasets["test"] = concatenate_datasets(raw_val_datasets).shuffle(
                    seed = 42
                )
            else:
                raw_datasets["test"] = concatenate_datasets(raw_val_datasets)

        if len(raw_datasets) == 0:
            raise ValueError(
                f"Dataset {dataset_mixer} not recognized with split {split}. Check the dataset has been correctly formatted."
            )

        return raw_datasets

    return apply_chat_template, execution_step_3, get_datasets


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <a name="Data"></a>
    ### Data Prep
    원본과 같은 `HuggingFaceH4/ultrafeedback_binarized`의 `train_prefs` 0.5%를 사용합니다.
    `test_prefs`는 평가용으로 별도 보관하며 학습에 섞지 않습니다.
    Qwen3.5의 채팅 템플릿으로 공통 prompt와 chosen/rejected 응답을 분리하고, `enable_thinking=False`를 사용합니다.
    """)
    return


@app.cell
def _(apply_chat_template, execution_step_3, get_datasets, tokenizer):
    # 원본 코드 셀 순서와 부수 효과를 보존합니다.
    execution_step_4 = execution_step_3 + 1
    raw_datasets = get_datasets(
        {"HuggingFaceH4/ultrafeedback_binarized" : 0.005}, # 0.5% sampled
        splits = ["train_prefs", "test_prefs"],
    )
    column_names = list(raw_datasets["train"].features)

    raw_datasets = raw_datasets.map(
        apply_chat_template,
        fn_kwargs = {"tokenizer": tokenizer, "task": "dpo"},
        num_proc = 12,
        remove_columns = column_names,
        desc = "Formatting comparisons with prompt template",
    )

    # Replace column names with what TRL needs, text_chosen -> chosen and text_rejected -> rejected
    for split in ["train", "test"]:
        raw_datasets[split] = raw_datasets[split].rename_columns(
            {"text_prompt": "prompt", "text_chosen": "chosen", "text_rejected": "rejected"}
        )
    return execution_step_4, raw_datasets


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We shall print a random item from the dataset
    """)
    return


@app.cell
def _(execution_step_4, raw_datasets):
    # 원본 코드 셀 순서와 부수 효과를 보존합니다.
    execution_step_5 = execution_step_4 + 1
    import pprint

    row = raw_datasets["train"][8]
    pprint.pprint(row["prompt"])
    pprint.pprint(row["chosen"])
    pprint.pprint(row["rejected"])
    return (execution_step_5,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We now add LoRA adapters so we only need to update 1 to 10% of all parameters!
    """)
    return


@app.cell
def _(FastLanguageModel, execution_step_5, model):
    # 원본 코드 셀 순서와 부수 효과를 보존합니다.
    execution_step_6 = execution_step_5 + 1
    model_1 = FastLanguageModel.get_peft_model(model, r=64, target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'], lora_alpha=64, lora_dropout=0, bias='none', use_gradient_checkpointing='unsloth', random_state=3407, use_rslora=False, loftq_config=None)  # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128  # Currently only supports dropout = 0  # Currently only supports bias = "none"  # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!  # True or "unsloth" for very long context  # We support rank stabilized LoRA  # And LoftQ
    return execution_step_6, model_1


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <a name="Train"></a>
    ### Train the DPO model
    원본과 같이 데이터의 0.5%로 3에포크 학습합니다.
    현재 TRL API에 맞춰 `beta`와 `max_length`는 `DPOConfig`, 토크나이저는 `processing_class`로 전달합니다.
    원본의 이전 `max_prompt_length=512` 인자는 현재 API에서 제거되어 사용하지 않습니다.
    전체 prompt+응답의 최대 길이는 1024토큰입니다.
    """)
    return


@app.cell
def _(PatchDPOTrainer, execution_step_6):
    # 원본 코드 셀 순서와 부수 효과를 보존합니다.
    execution_step_7 = execution_step_6 + 1
    # One must patch the DPO Trainer first!
    PatchDPOTrainer()
    return (execution_step_7,)


@app.cell
def _(
    dtype,
    execution_step_7,
    model_1,
    output_dir,
    raw_datasets,
    tokenizer,
    torch,
):
    # 원본 코드 셀 순서와 부수 효과를 보존합니다.
    execution_step_8 = execution_step_7 + 1
    from trl import DPOTrainer, DPOConfig
    dpo_trainer = DPOTrainer(model=model_1, ref_model=None, args=DPOConfig(per_device_train_batch_size=2, gradient_accumulation_steps=4, warmup_ratio=0.1, num_train_epochs=3, learning_rate=5e-06, logging_steps=1, optim='adamw_8bit', weight_decay=0.0, lr_scheduler_type='linear', seed=42, output_dir=str(output_dir), report_to='none', beta=0.1, max_length=1024, bf16=dtype == torch.bfloat16, fp16=False), train_dataset=raw_datasets['train'], processing_class=tokenizer)  # PEFT adapter를 비활성화한 초기 백본을 reference로 사용  # eval_dataset = raw_datasets["test"],
    return dpo_trainer, execution_step_8


@app.cell
def _(dpo_trainer, execution_step_8, output_dir, tokenizer):
    # 원본 코드 셀 순서와 부수 효과를 보존합니다.
    execution_step_9 = execution_step_8 + 1
    trainer_stats = dpo_trainer.train()
    dpo_trainer.save_model(str(output_dir / "adapter"))
    tokenizer.save_pretrained(str(output_dir / "adapter"))
    dpo_trainer.save_metrics("train", trainer_stats.metrics)
    print(f"학습 결과와 LoRA adapter 저장 위치: {output_dir}")
    return


if __name__ == "__main__":
    app.run()
