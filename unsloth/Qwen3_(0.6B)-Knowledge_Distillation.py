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
    _run_id = _RunDatetime.now(_RunZoneInfo("Asia/Seoul")).strftime("%Y%m%d-%H%M%S") + "-" + _run_uuid4().hex[:6] + "-Qwen3_(0_6B)-Knowledge_Distillation"
    output_dir = _RunPath(__file__).resolve().parents[1] / "outputs" / "unsloth" / _run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"결과 저장 위치: {output_dir}")
    execution_step_0 = 0
    return execution_step_0, output_dir


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Qwen3 (0.6B) — 지식 증류

    원핫 정답 대신 더 큰 **교사 모델**의 출력 분포를 따르도록 작은 **학생 모델**을 학습합니다. 출력 분포에는 토큰당 훨씬 많은 학습 신호가 담겨 있습니다. TRL의 `GKDTrainer`를 사용하며, Unsloth는 다른 TRL 트레이너와 마찬가지로 이를 패치합니다.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    모델을 바꾸기 전에 주의하세요. **학생과 교사는 반드시 같은 어휘를 공유해야 합니다.**
    **`lmbda`**는 오프폴리시(`0`) 또는 온폴리시(`1`, 학생이 자신이 생성한 샘플을 평가)를 선택합니다. **`beta`**는 발산의 종류를 정하며 0은 순방향 KL, 1은 역방향 KL입니다.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 구성 선택

    Unsloth는 은닉 상태를 각 `lm_head`에 청크 단위로 투영합니다. 따라서 어느 모델에서도 전체 `(batch, seq, vocab)` 로짓을 한꺼번에 만들지 않습니다. 아래 측정에서 최대 메모리는 3.0GB로, TRL의 밀집 손실(`UNSLOTH_GKD_CHUNKED=0`)의 9.9GB보다 작았습니다. 손실 차이는 2e-5 이내였으며, 배치 크기 1에서는 약 30% 느리지만 더 큰 배치에서는 그렇지 않습니다.

    | 구성 | 가중치 크기 | 측정 결과 |
    | --- | --- | --- |
    | `qwen3` | 1.2 + 3.4 GB | 무료 T4 1장에서 통과: 손실 0.367, 어댑터 392/392개, 3.36 GB |
    | `gemma-4` | 8.1 + 10.9 GB | Kaggle T4 2장에서 통과: 손실 0.079, 어댑터 410/410개, 14.3 GB |
    | `muse-glimmer` | 22.2 GB | 자기 증류, 시퀀스 길이 384에서 208 MiB 부족으로 메모리 초과 |
    | `qwen3.8` | 22.3 GB | 자기 증류, dtype 불일치로 실행 불가 |

    청크 단위 손실이 도입된 뒤 다시 측정한 구성은 `qwen3`뿐입니다. 이 덕분에 이제 GPU 한 장에 들어갑니다. 나머지 결과는 도입 이전의 측정이라 메모리 요구량을 보수적으로 보여 줍니다.

    `unsloth/gemma-4-26B-A4B-it`은 A100 이상의 GPU가 필요합니다. 사용하려면 직접 구성에 추가하세요.
    """)
    return


@app.cell
def _(execution_step_0):
    # 원본 코드 셀 순서와 부수 효과를 보존합니다.
    execution_step_1 = execution_step_0 + 1
    CONFIGS = {
        "gemma-4": dict(
            student = "unsloth/gemma-4-E2B-it-unsloth-bnb-4bit",
            teacher = "unsloth/gemma-4-E4B-it-unsloth-bnb-4bit",
        ),
        "qwen3": dict(
            student = "unsloth/Qwen3-0.6B",
            teacher = "unsloth/Qwen3-1.7B",
        ),
        "qwen3.8": dict(
            student = "unsloth/Qwen3.8-27B-unsloth-bnb-4bit",
            teacher = None,
        ),
        "muse-glimmer": dict(
            student = "unsloth/Muse-Glimmer-30B-unsloth-bnb-4bit",
            teacher = None,
        ),
    }

    CONFIG = "qwen3"
    student_name = CONFIGS[CONFIG]["student"]
    teacher_name = CONFIGS[CONFIG]["teacher"]
    return CONFIG, execution_step_1, student_name, teacher_name


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    주의할 점은 세 가지입니다.

    **`qwen3.8`과 `muse-glimmer`는 자기 증류를 수행합니다.** 더 작은 같은 계열 모델이 없기 때문입니다. 이미 학습한 어댑터가 있어야 효과가 있으며, 새 어댑터는 자체 베이스 모델과 같은 출력을 냅니다.

    **gemma-4는 T4급 GPU에서 `torch.compile`을 꺼야 합니다.** 이 환경에서는 그래디언트 체크포인팅의 재계산과 순전파 사이에 텐서 차원 수가 일치하지 않습니다.

    **qwen3.8은 T4급 GPU에서 실행이 막혀 있습니다.** 게이트 델타 네트워크에서 `BFloat16 != Half` 불일치가 발생하기 때문입니다. bfloat16을 기본 지원하는 환경에서는 재현되지 않습니다.
    """)
    return


@app.cell
def _(CONFIG, execution_step_1, student_name, teacher_name):
    # 원본 코드 셀 순서와 부수 효과를 보존합니다.
    execution_step_2 = execution_step_1 + 1
    import os
    if CONFIG.startswith("gemma-4"):
        os.environ["UNSLOTH_COMPILE_DISABLE"] = "1"

    max_seq_length = 1024
    load_in_4bit = True
    print(f"student: {student_name}")
    # 인라인으로 넣지 않고 먼저 변수에 저장합니다. Python 3.12 이전에는 f-string 표현식에
    # 역슬래시를 넣을 수 없지만 작은따옴표를 이스케이프하려면 역슬래시가 필요하기 때문입니다.
    teacher_label = teacher_name or "(self-distillation: the student's own frozen base)"
    print(f"teacher: {teacher_label}")
    return execution_step_2, load_in_4bit, max_seq_length


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    이전 릴리스는 `GKDConfig`를 `SFTConfig`로 변환하고 청크 단위 손실도 지원하지 않아, 증류가 조용히 일반 SFT로 실행될 수 있습니다. 이미 unsloth를 불러온 런타임은 기존 사본을 계속 사용하므로 해당 기능이 실제로 있는지 확인해야 합니다. 아래 확인이 실패하면 런타임을 다시 시작하세요.
    """)
    return


@app.cell
def _(execution_step_2):
    # 원본 코드 셀 순서와 부수 효과를 보존합니다.
    execution_step_3 = execution_step_2 + 1
    # 버전 문자열 대신 기능을 직접 검사합니다. 노트북 생성기가
    # 셀 안의 모든 버전 리터럴을 다시 써서 버전 고정이 조용히 무력화되기 때문입니다.
    # unsloth를 먼저 가져옵니다. UNSLOTH_IS_PRESENT가 설정되기 전에는 unsloth_zoo를 가져올 수 없습니다.
    import unsloth
    from unsloth.models.rl_replacements import RL_FUNCTIONS
    from unsloth_zoo.rl_replacements import distillation_chunked_jsd

    _patches = [getattr(f, "__name__", "") for f in RL_FUNCTIONS.get("gkd_trainer", [])]
    assert any("compute_loss" in n for n in _patches), (
        "this unsloth does not route GKD through the chunked loss; restart the "
        f"runtime so the install upgrade applies (gkd patches: {_patches})"
    )
    print(f"GKD chunked loss wired: {_patches}")
    return (execution_step_3,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 학생 모델 불러오기
    """)
    return


@app.cell
def _(execution_step_3, load_in_4bit, max_seq_length, student_name):
    # 원본 코드 셀 순서와 부수 효과를 보존합니다.
    execution_step_4 = execution_step_3 + 1
    from unsloth import FastLanguageModel
    import torch

    student, processor = FastLanguageModel.from_pretrained(
        student_name,
        max_seq_length = max_seq_length,
        dtype = None,
        load_in_4bit = load_in_4bit,
    )
    return FastLanguageModel, execution_step_4, processor, student, torch


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    `*ForConditionalGeneration` 체크포인트는 프로세서를 반환합니다. 트레이너에는 그 안의 토크나이저를 전달하세요. 그렇지 않으면 채팅 프롬프트를 이미지로 읽어 오류가 발생합니다.
    """)
    return


@app.cell
def _(execution_step_4, processor):
    # 원본 코드 셀 순서와 부수 효과를 보존합니다.
    execution_step_5 = execution_step_4 + 1
    def text_tokenizer(maybe_processor):
        inner = getattr(maybe_processor, "tokenizer", None)
        if inner is not None and type(maybe_processor).__name__.endswith("Processor"):
            print(f"unwrapped {type(maybe_processor).__name__} -> {type(inner).__name__}")
            return inner
        return maybe_processor

    tokenizer = text_tokenizer(processor)
    return execution_step_5, text_tokenizer, tokenizer


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    손실 함수가 출력 헤드를 직접 참조하므로 출력 헤드는 양자화하지 않은 밀집 형태로 유지해야 합니다. Unsloth는 `lm_head`를 양자화하지 않지만, 다른 도구로 양자화한 체크포인트는 그렇지 않을 수 있습니다.
    """)
    return


@app.cell
def _(execution_step_5, student):
    # 원본 코드 셀 순서와 부수 효과를 보존합니다.
    execution_step_6 = execution_step_5 + 1
    head = student.get_output_embeddings()
    assert head.weight.dim() == 2, f"output head is not dense: {type(head.weight).__name__}"
    print(f"output head {type(head).__name__} {tuple(head.weight.shape)} {head.weight.dtype}")
    return (execution_step_6,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 학생 모델에 LoRA 어댑터 추가
    """)
    return


@app.cell
def _(FastLanguageModel, execution_step_6, student):
    # 원본 코드 셀 순서와 부수 효과를 보존합니다.
    execution_step_7 = execution_step_6 + 1
    student_1 = FastLanguageModel.get_peft_model(student, r=16, lora_alpha=32, lora_dropout=0, bias='none', target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'], use_gradient_checkpointing='unsloth', random_state=3407)
    trainable = [n for n, p in student_1.named_parameters() if p.requires_grad]
    print(f'{len(trainable)} trainable tensors')
    return execution_step_7, student_1


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    `lora_B`는 0으로 초기화되므로 첫 스텝에서는 `lora_B` 쪽만 업데이트됩니다. 정상적인 동작입니다.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 교사 모델을 불러오거나 학생의 고정된 베이스 재사용
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    `teacher = None`이면 추가 메모리 없이 고정된 베이스를 사용합니다. 다만 어댑터는 명시적으로 비활성화해야 합니다. `GKDTrainer`는 `self.teacher_model(...)`만 호출하므로 학생 모델을 그대로 전달하면 두 순전파 결과가 같아집니다. 트레이너가 매 스텝 `teacher.eval()`을 호출하므로 래퍼는 학생 모델을 하위 모듈로 등록하지 않습니다.

    별도의 교사 모델에는 device-map 메모리 예산이 필요합니다. 그렇지 않으면 배치 계획기가 GPU 메모리 전체를 차지해 로딩 자체에 필요한 버퍼 공간이 남지 않습니다.
    """)
    return


@app.cell
def _(execution_step_7, torch):
    # 원본 코드 셀 순서와 부수 효과를 보존합니다.
    execution_step_8 = execution_step_7 + 1
    class FrozenBaseTeacher(torch.nn.Module):
        """The student with its adapters switched off for the duration of the call."""
        def __init__(self, peft_student):
            super().__init__()
            object.__setattr__(self, "student", peft_student)

        @property
        def config(self):
            return self.student.config

        def forward(self, *args, **kwargs):
            with self.student.disable_adapter():
                return self.student(*args, **kwargs)

    return FrozenBaseTeacher, execution_step_8


@app.cell
def _(
    FastLanguageModel,
    FrozenBaseTeacher,
    execution_step_8,
    load_in_4bit,
    max_seq_length,
    student_1,
    teacher_name,
    torch,
):
    # 원본 코드 셀 순서와 부수 효과를 보존합니다.
    execution_step_9 = execution_step_8 + 1
    teacher_processor = None  # 별도 교사 모델일 때만 자체 토크나이저를 불러옵니다
    if teacher_name is None:
        teacher = FrozenBaseTeacher(student_1)
        print("self-distillation: teacher is the student's base with adapters disabled")
    else:
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        TEACHER_HEADROOM_GIB = 2.0
        max_memory = {}
        for _d in range(torch.cuda.device_count()):
            _free, _total = torch.cuda.mem_get_info(_d)
            max_memory[_d] = max(int(_free - TEACHER_HEADROOM_GIB * 2 ** 30), 0)
            print(f'cuda:{_d} free {_free / 2 ** 30:.2f} GiB of {_total / 2 ** 30:.2f} GiB -> teacher budget {max_memory[_d] / 2 ** 30:.2f} GiB')
        teacher, teacher_processor = FastLanguageModel.from_pretrained(teacher_name, max_seq_length=max_seq_length, dtype=None, load_in_4bit=load_in_4bit, max_memory=max_memory)
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad_(False)
        print(f'teacher loaded, {sum((p.requires_grad for p in teacher.parameters()))} trainable (must be 0)')
    return execution_step_9, teacher, teacher_processor


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    교사와 학생의 출력은 달라야 합니다. 차이가 0이면 손실이 `1e-4`에 가까워져 문제가 생긴 상태를 수렴한 것처럼 오해할 수 있으므로 실제 차이를 측정합니다.
    """)
    return


@app.cell
def _(execution_step_9, student_1, teacher, tokenizer, torch):
    # 원본 코드 셀 순서와 부수 효과를 보존합니다.
    execution_step_10 = execution_step_9 + 1
    _ids = tokenizer('The capital of France is', return_tensors='pt').input_ids.to(student_1.device)
    with torch.no_grad():
        _gap = (student_1(input_ids=_ids).logits - teacher(input_ids=_ids).logits).abs().max().item()
    print(f'max teacher-student logit gap: {_gap:.3f}')
    if _gap == 0:
        print('WARNING: the teacher matches the student exactly, so the divergence and its\ngradient are both zero. Expected for self-distillation from a fresh adapter;\nload an adapter you trained earlier to give this mode something to learn.')
    return (execution_step_10,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    어휘 크기가 같다는 것만으로는 충분하지 않습니다. Qwen2.5-0.5B와 7B는 모두 어휘 크기가 151936이지만 서로 다른 ID를 부여하므로 토크나이저가 생성하는 결과를 비교해야 합니다.
    """)
    return


@app.cell
def _(
    execution_step_10,
    student_1,
    teacher,
    teacher_name,
    teacher_processor,
    text_tokenizer,
    tokenizer,
):
    # 원본 코드 셀 순서와 부수 효과를 보존합니다.
    execution_step_11 = execution_step_10 + 1
    if teacher_name is not None:
        s_vocab = student_1.config.get_text_config().vocab_size
        t_vocab = teacher.config.get_text_config().vocab_size
        assert s_vocab == t_vocab, f'vocab width mismatch: student {s_vocab} vs teacher {t_vocab}'
        probe = 'The quick brown fox jumps over 0123456789 lazy dogs, naively.'
        teacher_tokenizer = text_tokenizer(teacher_processor)
        assert tokenizer(probe).input_ids == teacher_tokenizer(probe).input_ids, 'same vocabulary width but different token ids, so the two models do not share a vocabulary and GKD cannot compare their distributions'
        print(f'vocabularies match: {s_vocab}, and the ids agree')
    return (execution_step_11,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 데이터셋

    FineTome의 ShareGPT 턴은 `from`/`value` 키를 사용합니다. 채팅 템플릿은 `role`/`content`를 기대하므로 변환하지 않으면 `'dict object' has no attribute 'role'` 오류가 발생합니다.
    """)
    return


@app.cell
def _(execution_step_11):
    # 원본 코드 셀 순서와 부수 효과를 보존합니다.
    execution_step_12 = execution_step_11 + 1
    from datasets import load_dataset
    from unsloth.chat_templates import standardize_sharegpt

    dataset = load_dataset("mlabonne/FineTome-100k", split = "train[:500]")
    dataset = standardize_sharegpt(dataset)
    dataset = dataset.rename_column("conversations", "messages")
    dataset = dataset.remove_columns([c for c in dataset.column_names if c != "messages"])
    return dataset, execution_step_12


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    `max_seq_length` 안에 **전체가** 들어가지 않는 행은 제거합니다. 콜레이터는 너무 긴 행의 앞부분을 잘라 내므로 `prompts`가 비어 버리고 손실 계산에 사용할 슬라이스도 없어집니다. 이때 `mask [1, 512] does not match ... tensor [1, 0, 262144]` 오류가 발생합니다.
    """)
    return


@app.cell
def _(dataset, execution_step_12, max_seq_length, tokenizer):
    # 원본 코드 셀 순서와 부수 효과를 보존합니다.
    execution_step_13 = execution_step_12 + 1

    def _fits_in_the_budget(example):
        messages = example['messages']
        if len(messages) < 2:
            return False
        whole = tokenizer.apply_chat_template(messages, tokenize=True)
        prompt = tokenizer.apply_chat_template(messages[:-1], tokenize=True, add_generation_prompt=True)
        return len(whole) <= max_seq_length and len(prompt) < len(whole)
    _before = len(dataset)
    dataset_1 = dataset.filter(_fits_in_the_budget)
    print(f'kept {len(dataset_1)}/{_before} rows that fit in {max_seq_length} tokens')
    assert len(dataset_1) > 0, f'no row fits in max_seq_length = {max_seq_length}; raise it or use a dataset with shorter conversations'
    print(dataset_1[0]['messages'][:2])
    return dataset_1, execution_step_13


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 학습

    `lmbda`, `beta`, `temperature`는 증류를 조정하는 매개변수이며 나머지는 일반 TRL 설정입니다. Unsloth가 모델 구조별로 정밀도를 선택하므로 `fp16`/`bf16`은 의도적으로 지정하지 않습니다.
    """)
    return


@app.cell
def _(execution_step_13, max_seq_length, output_dir):
    # 원본 코드 셀 순서와 부수 효과를 보존합니다.
    execution_step_14 = execution_step_13 + 1
    from trl.experimental.gkd import GKDConfig, GKDTrainer

    config = GKDConfig(
        output_dir = str(output_dir / 'checkpoints'),
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        max_steps = 30,
        learning_rate = 2e-4,
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        report_to = "none",
        max_length = max_seq_length,
        lmbda = 0.0,        # 0은 오프폴리시, 1은 온폴리시
        beta = 0.5,         # 0은 순방향 KL, 1은 역방향 KL
        temperature = 1.0,
        max_new_tokens = 64,
    )
    return GKDTrainer, config, execution_step_14


@app.cell
def _(
    GKDTrainer,
    config,
    dataset_1,
    execution_step_14,
    student_1,
    teacher,
    tokenizer,
):
    # 원본 코드 셀 순서와 부수 효과를 보존합니다.
    execution_step_15 = execution_step_14 + 1
    trainer = GKDTrainer(model=student_1, teacher_model=teacher, args=config, train_dataset=dataset_1, processing_class=tokenizer)
    assert type(trainer.args).__name__ == 'GKDConfig', type(trainer.args).__name__
    assert trainer.args.lmbda == 0.0 and trainer.args.beta == 0.5
    print(f'args {type(trainer.args).__name__}: lmbda={trainer.args.lmbda} beta={trainer.args.beta} temperature={trainer.args.temperature}')
    return execution_step_15, trainer


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    이전에는 `SFTConfig`를 상속한 설정이 별다른 알림 없이 일반 설정으로 다시 만들어지면서 증류 관련 필드가 모두 사라졌습니다. 이를 확인하기 위해 assert 검사를 사용합니다.
    """)
    return


@app.cell
def _(execution_step_15, student_1, torch, trainer):
    # 원본 코드 셀 순서와 부수 효과를 보존합니다.
    execution_step_16 = execution_step_15 + 1
    before = {n: p.detach().clone() for n, p in student_1.named_parameters() if p.requires_grad}
    result = trainer.train()
    changed = sum((1 for n, p in student_1.named_parameters() if p.requires_grad and (not torch.equal(p.detach(), before[n]))))
    peak = torch.cuda.max_memory_allocated() / 1000000000.0
    print(f'\ntrain loss      : {result.training_loss}')
    print(f'adapters changed: {changed}/{len(before)}')
    print(f'peak memory     : {peak:.2f} GB')
    history = [h['loss'] for h in trainer.state.log_history if 'loss' in h]
    if len(history) >= 2:
        print(f'loss first -> last: {history[0]:.4f} -> {history[-1]:.4f}')
    assert changed > 0, 'no adapter changed: the student did not learn'
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 지원 범위와 한계

    `GKDTrainer`는 학생과 교사를 함께 메모리에 유지하므로 GPU 용량이 교사 모델의 크기를 제한합니다. Deepseek나 Kimi급 교사를 사용하려면 서버로 제공되는 교사 모델 또는 캐시된 top-k 로그 확률이 필요하지만, 원본 설명 시점에는 어느 쪽도 Unsloth에 연결되어 있지 않습니다.

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
