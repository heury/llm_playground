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
    `uv run marimo edit minimind_omni_tour.py`. 환경·준비물과 권장 순서는 [공통 README](README.md)를 참고하세요.
    위젯 변경 시 의존 셀이 다시 실행됩니다. 모델 크기·데이터·학습 설정 변경 전에는 해당 절의 재실행 범위를 확인하세요.
    """)
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # MiniMind-O 실습: 생각하는 모델에 목소리 달기

    **Omni 모델**은 텍스트로 답하면서 동시에 그 답을 **음성으로 말하는** 모델입니다.
    이 노트북은 `projects/minimind-o/model/model_omni.py`의 `MiniMindOmni`를 직접 불러와,
    영어 대화 모델 [`caspar/MiniMind-Dense-64M-Mixed-SFT-v1.0`](https://huggingface.co/caspar/MiniMind-Dense-64M-Mixed-SFT-v1.0)에
    음성 출력 부분(Talker)을 붙이고 GPU에서 학습합니다.

    | 단계 | 내용 | 직접 확인하는 것 |
    |---|---|---|
    | 1 | 베이스 LLM을 Omni 구조에 싣기 | 어떤 가중치가 채워지고 어떤 가중치가 새로 생기는가 |
    | 2 | 소리를 토큰으로 바꾸기 (Mimi) | 코드북 8개가 각각 무엇을 담당하는가 |
    | 3 | 학습 샘플 만들기 | 9줄짜리 입력과 계단 모양 지연(delay) |
    | 4 | Talker 순전파 해부 | bridge 은닉 상태가 어떻게 Talker로 넘어가는가 |
    | 5 | GPU 학습 (실험 3개) | Talker만 / 텍스트 정보 차단 / 전체 학습 |
    | 6 | 평가 | 받아 읽기 WER, 자유 대화, 텍스트 능력 보존 |
    | 7 | 스트리밍 | 첫 음성이 나오기까지 몇 단계가 걸리는가 |
    | 8~9 | 스케일 값, 음성 입력 경로 | `text_scale`, `inject_audio_features` |

    **준비물**
    - CUDA GPU (VRAM 8GB 이상 권장. 측정 최대 사용량은 실험 C 학습 중 6.1GB)
    - 처음 실행할 때 약 1.3GB 다운로드 (데이터 881MB, Mimi 코덱, Whisper-tiny, 베이스 모델 128MB)
    - 전체 실행 약 7분 (RTX 3080 Ti, 기본 설정). 학습 약 4분, 평가 약 2분입니다.

    > 원본 코드: [jingyaogong/minimind-o](https://github.com/jingyaogong/minimind-o) (Apache-2.0).
    > 베이스 모델의 라이선스와 학습 데이터 출처는 모델 카드를 따릅니다.


    **사전 지식:** [SFT](minimind_sft_tour.py)의 assistant 정답 마스크·학습 루프와 [VLM 구조](minimind_vlm_tour.py)의 Projector·placeholder 치환을 먼저 확인하세요.


    **지표 정의:** 이후 표·해설의 WER은 $\min(1,(S+D+I)/\max(1,N))$로 상한을 1로 제한한 값입니다.
    $S,D,I$는 단어 대치·삭제·삽입 수, $N$은 정답 단어 수입니다. 일반 WER은 삽입 때문에 1을 넘을 수 있습니다.
    이 표시에서 1은 원래 오류율이 100% 이상이라는 뜻이며, 모든 단어가 다르다는 뜻은 아닙니다.
    Whisper의 전사 오류도 섞이므로 발음·음질·화자 유사도를 모두 대변하지 않습니다.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.mermaid(
        """
    flowchart LR
        Q[질문 텍스트] --> E[임베딩]
        E --> L0[Thinker 층 0..3]
        L0 -- "bridge (층 3 출력)" --> P[embed_proj × text_scale]
        L0 --> L4[Thinker 층 4..7] --> TH[lm_head] --> TXT[텍스트 토큰]
        AC[이전 오디오 코드 8줄] --> TE[TalkerEmbedding] --> CP[codec_proj × audio_scale]
        P --> S((+))
        CP --> S
        S --> TK[Talker 층 0..3] --> HD[TalkerHead: 코드북 8개] --> MIMI[Mimi 디코더] --> WAV[음성 24kHz]
    """
    )
    return


@app.cell
def _():
    import sys
    import re
    import copy
    import json
    import time
    import random
    import tempfile
    import warnings
    from pathlib import Path

    import numpy as np
    import torch
    import torch.nn as nn
    import librosa
    import pyarrow as pa
    import pyarrow.parquet as pq
    import matplotlib.pyplot as plt
    from huggingface_hub import hf_hub_download, snapshot_download
    from safetensors.torch import load_file
    from torch.utils.data import DataLoader
    from transformers import (
        AutoTokenizer,
        MimiModel,
        WhisperForConditionalGeneration,
        WhisperProcessor,
    )
    from transformers import logging as hf_logging

    hf_logging.set_verbosity_error()

    REPO_ROOT = Path(__file__).resolve().parents[2]
    OMNI_DIR = REPO_ROOT / "projects" / "minimind-o"
    if str(OMNI_DIR) not in sys.path:
        sys.path.insert(0, str(OMNI_DIR))

    # 원본 프로젝트의 코드를 그대로 가져옵니다.
    from model.model_omni import MiniMindOmni, OmniConfig
    from model.model_minimind import MiniMindForCausalLM
    from dataset.omni_dataset import OmniDataset
    from trainer.trainer_utils import SkipBatchSampler, get_lr, setup_seed
    from trainer.train_sft_omni import omni_collate_fn

    return (
        AutoTokenizer,
        DataLoader,
        MimiModel,
        MiniMindForCausalLM,
        MiniMindOmni,
        OmniConfig,
        OmniDataset,
        Path,
        SkipBatchSampler,
        WhisperForConditionalGeneration,
        WhisperProcessor,
        copy,
        get_lr,
        hf_hub_download,
        json,
        librosa,
        load_file,
        nn,
        np,
        omni_collate_fn,
        pa,
        plt,
        pq,
        random,
        re,
        setup_seed,
        snapshot_download,
        tempfile,
        time,
        torch,
        warnings,
    )


@app.cell(hide_code=True)
def _(mo, torch):
    mo.stop(
        not torch.cuda.is_available(),
        mo.callout(
            mo.md("이 노트북은 **CUDA GPU**가 필요합니다. `torch.cuda.is_available()`이 `False`입니다."),
            kind="danger",
        ),
    )
    DEVICE = "cuda"
    mo.md(f"GPU: **{torch.cuda.get_device_name(0)}**, VRAM {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    return (DEVICE,)


@app.cell
def _():
    # 실습 결과를 재현할 수 있도록 모든 외부 리소스의 리비전을 고정합니다.
    BASE_REPO = "caspar/MiniMind-Dense-64M-Mixed-SFT-v1.0"
    BASE_REV = "7597c22aa4d3e7b2ac824b4b4be01158cccf1b8d"
    MIMI_REPO = "jingyaogong/mimi"
    MIMI_REV = "b4e362bbfbba9444b162de486da40af6639e0b98"
    DATA_REPO = "jingyaogong/minimind-o_dataset"
    DATA_REV = "d6588e12ac2ac8ced65eb58a7d7b3eef4aa220de"
    DATA_FILE = "sft_a2a_mini.parquet"
    ASR_REPO = "openai/whisper-tiny.en"
    ASR_REV = "87c7102498dcde7456f24cfd30239ca606ed9063"
    return (
        ASR_REPO,
        ASR_REV,
        BASE_REPO,
        BASE_REV,
        DATA_FILE,
        DATA_REPO,
        DATA_REV,
        MIMI_REPO,
        MIMI_REV,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. 베이스 LLM을 Omni 구조에 싣기

    `MiniMindOmni`는 `MiniMindForCausalLM`을 **상속**합니다. 그래서 베이스 LLM의 가중치 이름
    (`model.layers.*`, `model.embed_tokens.*`)이 Omni에서도 그대로 쓰이고, 이 부분을 **Thinker**라고 부릅니다
    (`self.thinker`는 `self.model`의 별칭입니다).

    Omni가 새로 추가하는 모듈은 다음과 같습니다.

    - `talker`: 오디오 코드를 예측하는 4층 Transformer와 입·출력층
    - `audio_proj`: 음성 입력(SenseVoice 특징 512차원)을 768차원으로 바꾸는 MLP
    - `vision_proj`: 이미지 입력용 MLP. 이번 베이스는 텍스트 전용이므로 쓰지 않습니다.

    베이스 모델의 `config.json`에서 구조 값만 골라 `OmniConfig`를 만듭니다.
    `bridge_layer`는 따로 지정하지 않으면 `num_hidden_layers // 2 - 1 = 3`이 됩니다.
    """)
    return


@app.cell
def _(
    AutoTokenizer,
    BASE_REPO,
    BASE_REV,
    MiniMindOmni,
    OmniConfig,
    Path,
    json,
    load_file,
    setup_seed,
    snapshot_download,
    warnings,
):
    base_dir = Path(snapshot_download(BASE_REPO, revision=BASE_REV))
    tokenizer = AutoTokenizer.from_pretrained(base_dir)

    _raw_cfg = json.loads((base_dir / "config.json").read_text())
    _meta_keys = {"architectures", "auto_map", "model_type", "model_version", "finetuning_task", "dtype", "transformers_version"}
    omni_config = OmniConfig(**{k: v for k, v in _raw_cfg.items() if k not in _meta_keys})

    setup_seed(42)
    with warnings.catch_warnings():
        # SenseVoice(음성 입력 인코더)와 SigLIP(이미지 인코더)은 이번 실습에서 불러오지 않습니다.
        warnings.simplefilter("ignore")
        base_model = MiniMindOmni(omni_config, audio_encoder_path="__not_used__", vision_model_path=None)

    _state = {k: v.float() for k, v in load_file(base_dir / "model.safetensors").items()}
    load_report = base_model.load_state_dict(_state, strict=False)
    return base_model, load_report, omni_config, tokenizer


@app.cell(hide_code=True)
def _(base_model, load_report, mo, omni_config):
    _missing_groups = {}
    for _k in load_report.missing_keys:
        _g = _k.split(".")[0] if not _k.startswith("talker.") else "talker." + _k.split(".")[1]
        _missing_groups[_g] = _missing_groups.get(_g, 0) + 1

    assert not load_report.unexpected_keys, load_report.unexpected_keys
    assert {g.split(".")[0] for g in _missing_groups} == {"talker", "audio_proj", "vision_proj", "lm_head"}
    # lm_head는 비어 있지만 임베딩과 가중치를 공유(tied)하므로 실제로는 채워져 있습니다.
    assert base_model.lm_head.weight.data_ptr() == base_model.thinker.embed_tokens.weight.data_ptr()

    def _count(m):
        return sum(p.numel() for p in m.parameters()) / 1e6

    _rows = "\n    ".join(
        f"| `{name}` | {_count(mod):.2f}M | {'베이스에서 로드' if name == 'thinker' else '새로 초기화'} |"
        for name, mod in [
            ("thinker", base_model.thinker),
            ("talker", base_model.talker),
            ("audio_proj", base_model.audio_proj),
            ("vision_proj", base_model.vision_proj),
        ]
    )
    mo.md(
        f"""
    **불러오지 못한 키 그룹** (= Omni가 새로 만든 부분): {", ".join(f"`{g}`({n})" for g, n in sorted(_missing_groups.items()))}

    `lm_head.weight`가 목록에 있지만 임베딩과 같은 텐서를 공유하므로 문제없습니다.

    | 모듈 | 파라미터 | 상태 |
    |---|---|---|
    {_rows}

    - Thinker 층 수 {len(base_model.thinker.layers)}, Talker 층 수 {len(base_model.talker.layers)}, `bridge_layer` = {omni_config.bridge_layer}
    - 오디오 어휘 {omni_config.audio_vocab_size} = Mimi 코드 2048 + 특수 토큰 (pad {omni_config.audio_pad_token}, stop {omni_config.audio_stop_token}, spk {omni_config.audio_spk_token})
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 1-1. Talker를 Thinker의 뒤쪽 층으로 초기화하기

    원본 학습 코드 `trainer_utils.init_omni_model`은 Talker 층을 무작위로 두지 않고
    **Thinker의 마지막 4개 층(4~7)을 복사**해서 시작합니다. 이미 언어를 다룰 줄 아는 층에서 출발하면
    학습이 빨라집니다. 같은 일을 여기서 직접 합니다.
    """)
    return


@app.cell
def _(DEVICE, base_model, torch):
    _n_thinker, _n_talker = len(base_model.thinker.layers), len(base_model.talker.layers)
    for _i in range(_n_talker):
        base_model.talker.layers[_i].load_state_dict(base_model.thinker.layers[_n_thinker - _n_talker + _i].state_dict())
    base_model.to(DEVICE).eval()

    assert torch.equal(base_model.talker.layers[0].self_attn.q_proj.weight, base_model.thinker.layers[4].self_attn.q_proj.weight)
    talker_init_note = f"Talker 층 0..{_n_talker - 1} ← Thinker 층 {_n_thinker - _n_talker}..{_n_thinker - 1} 복사 완료"
    talker_init_note
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 1-2. Thinker는 베이스 LLM과 똑같이 계산하는가?

    `MiniMindOmni.forward`는 입력이 2차원 `[B, T]`이면 텍스트만 들어온 것으로 보고, 오디오 8줄을 모두
    `audio_pad`(2049)로 채워 Talker까지 함께 계산합니다. 하지만 Talker는 Thinker 계산에 영향을 주지 않으므로
    **텍스트 logits는 부모 클래스 `MiniMindForCausalLM.forward`의 결과와 같아야 합니다.**
    """)
    return


@app.cell(hide_code=True)
def _(DEVICE, MiniMindForCausalLM, base_model, mo, tokenizer, torch):
    def prompt_ids(question):
        """채팅 템플릿을 적용하고, 모델 카드 안내대로 빈 think 블록을 제거합니다."""
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": question}], tokenize=False, add_generation_prompt=True
        )
        return tokenizer(text.replace("<think>\n\n</think>\n\n", ""), add_special_tokens=False).input_ids

    with torch.no_grad():
        _x = torch.tensor([prompt_ids("Why is the sky blue?")], device=DEVICE)
        _omni_out = base_model(_x)
        _llm_out = MiniMindForCausalLM.forward(base_model, input_ids=_x)
    _diff = (_omni_out.logits - _llm_out.logits).abs().max().item()
    assert _diff < 1e-4, _diff

    mo.md(
        f"""
    - Omni 텍스트 logits와 베이스 LLM logits의 최대 차이: **{_diff:.2e}** → 같은 계산입니다.
    - 텍스트만 넣어도 `audio_logits`가 나옵니다: 코드북 {len(_omni_out.audio_logits)}개 × `{tuple(_omni_out.audio_logits[0].shape)}`
    """
    )
    return (prompt_ids,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 1-3. 베이스 모델과 대화하기

    원본 `generate(stream=True)`로 텍스트를 생성합니다. 이 함수는 텍스트가 끝난 뒤에도
    **오디오가 끝날 때까지** 반복을 계속하므로(이때 텍스트 자리는 `None`), 텍스트만 필요할 때는
    `None`이 나오면 멈춥니다. 비교를 쉽게 하려고 temperature를 0.01로 두어 거의 탐욕적으로 생성합니다.
    """)
    return


@app.cell
def _(DEVICE, prompt_ids, tokenizer, torch):
    GENERAL_QUESTIONS = [
        "What is the capital of France?",
        "Give me one tip for better sleep.",
        "Name three primary colors.",
    ]

    @torch.no_grad()
    def chat_text(model, question, max_new_tokens=60):
        x = torch.tensor([prompt_ids(question)], device=DEVICE)
        last = None
        for y, _frame in model.generate(
            x, tokenizer.eos_token_id, max_new_tokens=max_new_tokens, temperature=0.01, top_p=1.0, stream=True
        ):
            if y is None:
                break
            last = y
        return tokenizer.decode(last[0].tolist(), skip_special_tokens=True).strip()

    return GENERAL_QUESTIONS, chat_text


@app.cell(hide_code=True)
def _(GENERAL_QUESTIONS, base_model, chat_text, result_table, torch):
    base_answers = {}
    for _q in GENERAL_QUESTIONS:
        torch.manual_seed(0)
        base_answers[_q] = chat_text(base_model, _q)
    result_table([{"질문": q, "베이스 답변": a} for q, a in base_answers.items()], selection=None)
    return (base_answers,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. 소리를 토큰으로: Mimi 코덱

    Talker는 파형을 직접 만들지 않고 **Mimi 코덱의 코드**를 예측합니다.

    - Mimi는 24kHz 음성을 **초당 12.5프레임**(프레임당 80ms)으로 압축합니다.
    - 한 프레임은 **코드북 8개**의 코드(각 0~2047)로 표현됩니다. 이를 RVQ(Residual Vector Quantization)라고 합니다.
    - 첫 코드북이 대략적인 소리를 표현하고, 다음 코드북들은 **앞 코드북이 남긴 오차(residual)**를 차례로 보정합니다.

    데이터셋 `sft_a2a_mini.parquet`의 `answer_audios` 열에는 답변 음성이 이미 Mimi 코드로 저장되어 있습니다
    (프레임 순서로 `[f0c0, f0c1, …, f0c7, f1c0, …]`처럼 펼친 형태).

    음성이 제대로 복원되었는지는 **Whisper-tiny.en**으로 받아 적은 뒤 **WER**(단어 오류율)로 판정합니다.
    WER은 삽입 오류 때문에 1을 넘을 수 있지만, 여기서는 비교가 쉽도록 1에서 자릅니다.
    """)
    return


@app.cell(hide_code=True)
def _(
    ASR_REPO,
    ASR_REV,
    DEVICE,
    MIMI_REPO,
    MIMI_REV,
    MimiModel,
    WhisperForConditionalGeneration,
    WhisperProcessor,
    librosa,
    mo,
    re,
    snapshot_download,
    torch,
):
    mimi = MimiModel.from_pretrained(snapshot_download(MIMI_REPO, revision=MIMI_REV)).to(DEVICE).eval()
    _asr_dir = snapshot_download(ASR_REPO, revision=ASR_REV)
    asr_processor = WhisperProcessor.from_pretrained(_asr_dir)
    asr_model = WhisperForConditionalGeneration.from_pretrained(_asr_dir).to(DEVICE).eval()
    SAMPLE_RATE = mimi.config.sampling_rate  # 24000

    @torch.no_grad()
    def codes_to_wav(codes):
        """codes: [코드북 수, 프레임 수] 정수 텐서 → 24kHz numpy 파형"""
        codes = torch.as_tensor(codes, dtype=torch.long, device=DEVICE)
        codes = torch.where(codes >= 2048, torch.zeros_like(codes), codes)  # 특수 토큰이 섞이면 0으로
        return mimi.decode(codes[None]).audio_values[0, 0].float().cpu().numpy()

    @torch.no_grad()
    def transcribe(wav):
        if wav is None or len(wav) < SAMPLE_RATE // 10:
            return ""
        feats = asr_processor(
            librosa.resample(wav, orig_sr=SAMPLE_RATE, target_sr=16000), sampling_rate=16000, return_tensors="pt"
        ).input_features.to(DEVICE)
        ids = asr_model.generate(feats, max_new_tokens=120)
        return asr_processor.batch_decode(ids, skip_special_tokens=True)[0].strip()

    def _words(s):
        return re.sub(r"[^a-z0-9' ]", " ", s.lower()).split()

    def wer(ref, hyp):
        r, h = _words(ref), _words(hyp)
        dp = list(range(len(h) + 1))
        for i in range(1, len(r) + 1):
            prev, dp[0] = dp[0], i
            for j in range(1, len(h) + 1):
                prev, dp[j] = dp[j], min(dp[j] + 1, dp[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
        return min(1.0, dp[-1] / max(1, len(r)))

    def player(wav, label):
        if wav is None:
            return mo.md(f"**{label}**: (음성 없음)")
        return mo.vstack([mo.md(f"**{label}** ({len(wav) / SAMPLE_RATE:.1f}초)"), mo.audio(wav, rate=SAMPLE_RATE)])

    mo.md(
        f"Mimi: {mimi.config.frame_rate} 프레임/초, 코드북 크기 {mimi.config.codebook_size}, "
        f"샘플링 {SAMPLE_RATE}Hz (Mimi는 코드북을 최대 {mimi.config.num_quantizers}개까지 지원하지만 MiniMind-O는 앞 8개만 사용)"
    )
    return codes_to_wav, player, transcribe, wer


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 2-1. 실습용 데이터 고르기

    작은 GPU에서 몇 분 만에 결과를 보려면 짧고 깨끗한 샘플만 써야 합니다. 다음 조건으로 거릅니다.

    - 한 턴 대화(질문 1개 + 답변 1개), 화자 임베딩(`spk_emb`)이 없는 샘플 → 별도 화자 조건 벡터를 사용하지 않음(단일 화자인지는 별도 확인 필요)
    - 답변 음성 20~60프레임 (1.6~4.8초)
    - 질문과 답변이 ASCII(영어)이고 질문은 120자 이하
    - 말하는 속도가 초당 10~22자 → 텍스트와 음성이 어긋난 샘플 제외
    - 답변 문장이 중복되지 않음
    """)
    return


@app.cell(hide_code=True)
def _(DATA_FILE, DATA_REPO, DATA_REV, hf_hub_download, json, mo, pq, random):
    data_path = hf_hub_download(DATA_REPO, DATA_FILE, repo_type="dataset", revision=DATA_REV)
    _table = pq.read_table(data_path, columns=["conversations", "answer_audios", "spk_emb"])

    all_rows, _seen = [], set()
    for _c, _a, _s in zip(
        _table["conversations"].to_pylist(), _table["answer_audios"].to_pylist(), _table["spk_emb"].to_pylist()
    ):
        if _s or not _a:
            continue
        _conv = json.loads(_c)
        if len(_conv) != 2:
            continue
        _q, _ans, _n = _conv[0]["content"], _conv[1]["content"], len(_a[0]) // 8
        if not (20 <= _n <= 60) or not _q.isascii() or not _ans.isascii() or len(_q) > 120 or _ans in _seen:
            continue
        if not 10 <= len(_ans) / (_n / 12.5) <= 22:
            continue
        _seen.add(_ans)
        all_rows.append({"conversations": _c, "answer_audios": _a})
    random.Random(0).shuffle(all_rows)

    mo.md(f"원본 {_table.num_rows:,}개 중 조건을 통과한 샘플: **{len(all_rows):,}개**")
    return (all_rows,)


@app.cell(hide_code=True)
def _(all_rows, codes_to_wav, json, mo, player, torch, transcribe):
    def row_codes(row):
        """펼쳐진 코드 → [8, 프레임 수]"""
        flat = row["answer_audios"][0]
        return torch.tensor(flat[: len(flat) // 8 * 8]).view(-1, 8).T

    _row = all_rows[0]
    _conv = json.loads(_row["conversations"])
    demo_codes = row_codes(_row)
    _wav = codes_to_wav(demo_codes)
    mo.vstack(
        [
            mo.md(
                f"""
    - 질문: *{_conv[0]["content"]}*
    - 답변: *{_conv[1]["content"]}*
    - 코드 텐서 `{tuple(demo_codes.shape)}` → {demo_codes.shape[1]}프레임 = {demo_codes.shape[1] / 12.5:.2f}초
    - Whisper 받아쓰기: *{transcribe(_wav)}*
    """
            ),
            player(_wav, "데이터셋 음성 (코드북 8개 모두 사용)"),
        ]
    )
    return demo_codes, row_codes


@app.cell
def _(demo_codes, plt):
    _fig, _ax = plt.subplots(figsize=(10, 2.6))
    _im = _ax.imshow(demo_codes.numpy(), aspect="auto", cmap="viridis", interpolation="nearest")
    _ax.set_yticks(range(8), [f"cb{i}" for i in range(8)])
    _ax.set_xlabel("frame (80 ms)")
    _ax.set_title("Mimi codes of one answer (rows = codebooks)")
    _fig.colorbar(_im, ax=_ax, fraction=0.02)
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 2-2. 코드북을 몇 개까지 써야 알아들을 수 있을까?

    같은 음성을 **앞쪽 코드북 k개만** 써서 복원해 봅니다. RVQ는 뒤 코드북이 앞 코드북의 오차를 보정하므로,
    k가 작으면 내용 윤곽만 남고 k가 커질수록 또렷해집니다(기본 설정 측정값: k=1 0.78, 2 0.32, 4 0.31, 8 0.21).
    이것이 Talker가 코드북 8개를 **모두** 맞혀야 하는 이유입니다.
    """)
    return


@app.cell(hide_code=True)
def _(
    all_rows,
    codes_to_wav,
    json,
    mo,
    np,
    player,
    row_codes,
    transcribe,
    wer,
):
    _ks = [1, 2, 4, 8]
    _scores = {k: [] for k in _ks}
    _players = []
    for _j, _row in enumerate(all_rows[:6]):
        _codes, _ref = row_codes(_row), json.loads(_row["conversations"])[1]["content"]
        for _k in _ks:
            _wav = codes_to_wav(_codes[:_k])
            _scores[_k].append(wer(_ref, transcribe(_wav)))
            if _j == 0:
                _players.append(player(_wav, f"코드북 {_k}개"))
    rvq_wer = {k: float(np.mean(v)) for k, v in _scores.items()}

    mo.vstack(
        [
            mo.md(
                "| 사용한 코드북 수 k | "
                + " | ".join(str(k) for k in _ks)
                + " |\n|---|"
                + "---|" * len(_ks)
                + "\n| 평균 WER (6개 샘플) | "
                + " | ".join(f"{rvq_wer[k]:.2f}" for k in _ks)
                + " |"
            ),
            mo.hstack(_players, wrap=True),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. 학습 샘플: 9줄 입력과 계단 모양 지연

    원본 `OmniDataset`은 샘플 하나를 **9 × T 정수 텐서**로 만듭니다.

    - 0~7줄: 오디오 코드북 8개 (Talker 입력)
    - 8줄: 텍스트 토큰 (Thinker 입력)

    텍스트와 오디오는 **같은 위치 축**을 공유합니다. 답변 텍스트가 시작되는 위치를 `S`라 하면,
    코드북 `i`의 첫 코드는 $S+i+1$ 위치에 놓입니다. 즉 코드북마다 한 칸씩 늦게 시작하는 **계단 모양**이 됩니다.

    이렇게 하면 한 위치에서 8개 코드북을 동시에 예측하더라도, 같은 프레임의 앞 코드북(더 거친 정보)은
    이미 앞 위치에서 입력으로 들어와 있습니다. 대신 마지막 코드북까지 다 나와야 프레임 하나가 완성되므로
    **첫 음성이 나오기까지 몇 단계가 걸립니다** (7절에서 측정합니다).

    각 코드북 끝에는 `audio_stop`(2050)이 붙고, 학습 손실에서 이 토큰에 **10배 가중치**를 줍니다.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    n_train_ui = mo.ui.dropdown(options=["32", "64", "128", "256"], value="64", label="학습 문장 수")
    steps_ui = mo.ui.dropdown(options=["400", "800", "1600"], value="800", label="학습 step 수 (실험마다)")
    mo.vstack(
        [
            mo.md("**실험 설정**: 바꾸면 3절 이후가 다시 실행됩니다. 본문의 측정값은 기본값(64문장, 800 step) 기준입니다."),
            mo.hstack([n_train_ui, steps_ui], justify="start"),
        ]
    )
    return n_train_ui, steps_ui


@app.cell(hide_code=True)
def _(
    OmniDataset,
    Path,
    all_rows,
    mo,
    n_train_ui,
    pa,
    pq,
    tempfile,
    tokenizer,
):
    OMNI_MAX_LEN = 160  # 질문(~40토큰) + 음성(≤60프레임 + 지연 8)이 들어가는 길이
    N_HELDOUT = 16
    heldout_rows = all_rows[:N_HELDOUT]
    train_rows = all_rows[N_HELDOUT : N_HELDOUT + int(n_train_ui.value)]

    _tmp = Path(tempfile.mkdtemp(prefix="omni_tour_"))
    pq.write_table(pa.Table.from_pylist(train_rows), _tmp / "train.parquet")
    _train_parquet = str(_tmp / "train.parquet")
    # 학습용: 원본 기본값(scheduled sampling 5%)을 그대로 둡니다.
    train_ds = OmniDataset(_train_parquet, tokenizer, max_length=OMNI_MAX_LEN, image_token_len=64)
    # 관찰용: 입력을 무작위로 바꾸는 scheduled sampling을 끄면 규칙이 그대로 보입니다.
    inspect_ds = OmniDataset(_train_parquet, tokenizer, max_length=OMNI_MAX_LEN, image_token_len=64, scheduled_sampling=0.0)

    mo.md(f"학습 문장 **{len(train_rows)}개**, 학습에 쓰지 않는 확인용 문장 **{len(heldout_rows)}개**, 시퀀스 길이 {OMNI_MAX_LEN}")
    return OMNI_MAX_LEN, heldout_rows, inspect_ds, train_ds, train_rows


@app.cell(hide_code=True)
def _(inspect_ds, mo, omni_config, result_table, setup_seed, tokenizer, torch):
    setup_seed(1)
    sample_x, sample_text_labels, sample_audio_labels, *_ = inspect_ds[0]

    STOP, PAD = omni_config.audio_stop_token, omni_config.audio_pad_token
    _first = [int((sample_audio_labels[i] != -100).nonzero()[0]) for i in range(8)]
    _last = [int((sample_audio_labels[i] != -100).nonzero()[-1]) for i in range(8)]
    answer_start = _first[0]  # 라벨은 한 칸 밀려 있으므로 이 값이 곧 S (답변 텍스트 시작 위치)

    # 규칙 1: 코드북 i는 코드북 0보다 i칸 늦게 시작합니다.
    assert _first == [answer_start + i for i in range(8)], _first
    # 규칙 2: 모든 코드북의 마지막 정답은 STOP입니다 (시퀀스 길이 안에 다 들어감).
    assert all(int(sample_audio_labels[i, _last[i]]) == STOP for i in range(8))
    # 규칙 3: 입력은 정답을 한 칸 민 것(teacher forcing)입니다.
    for _i in range(8):
        _pos = (sample_audio_labels[_i] != -100).nonzero().squeeze(1)[:-1]
        assert torch.equal(sample_x[_i, _pos + 1], sample_audio_labels[_i, _pos])
    # 규칙 4: 답변 텍스트 첫 토큰이 S 위치의 입력입니다.
    assert sample_text_labels[answer_start - 1] == sample_x[8, answer_start]

    _cells = []
    for _p in range(answer_start - 3, answer_start + 13):
        _r = {"pos": _p, "S 기준": _p - answer_start, "텍스트 입력": repr(tokenizer.decode([int(sample_x[8, _p])]))}
        for _i in range(8):
            _c = int(sample_x[_i, _p])
            _r[f"cb{_i}"] = "·" if _c == PAD else ("STOP" if _c == STOP else str(_c))
        _cells.append(_r)

    mo.vstack(
        [
            mo.md(
                f"""
    샘플 텐서: 입력 `{tuple(sample_x.shape)}`, 텍스트 라벨 `{tuple(sample_text_labels.shape)}`, 오디오 라벨 `{tuple(sample_audio_labels.shape)}`

    답변 시작 S = **{answer_start}**. 코드북별 첫 라벨 위치 = `{_first}` → S + i 계단 확인.
    아래 표는 **입력** 텐서(라벨보다 한 칸 늦음)이고, `·`는 `audio_pad`입니다.
    """
            ),
            result_table(_cells, selection=None, page_size=16),
        ]
    )
    return PAD, STOP, answer_start, sample_audio_labels, sample_text_labels


@app.cell
def _(STOP, answer_start, np, plt, sample_audio_labels, sample_text_labels):
    _lab = sample_audio_labels.numpy()
    _img = np.zeros_like(_lab, dtype=float)
    _img[_lab != -100] = 1.0
    _img[_lab == STOP] = 2.0
    _txt = (sample_text_labels.numpy() != -100).astype(float)[None]
    _end = int(np.nonzero(_lab[7] != -100)[0].max()) + 3
    _lo = answer_start - 6

    _fig, (_a0, _a1) = plt.subplots(2, 1, figsize=(11, 3.2), height_ratios=[1, 6], sharex=True)
    _a0.imshow(_txt[:, _lo:_end], aspect="auto", cmap="Blues", vmin=0, vmax=1.5, interpolation="nearest")
    _a0.set_yticks([0], ["text"])
    _a1.imshow(_img[:, _lo:_end], aspect="auto", cmap="magma", vmin=0, vmax=2.2, interpolation="nearest")
    _a1.set_yticks(range(8), [f"cb{i}" for i in range(8)])
    _a1.set_xticks(range(0, _end - _lo, 10), range(_lo, _end, 10))
    _a1.set_xlabel("position (labels): dark = no loss, purple = audio code, bright = STOP")
    _a0.set_title("Loss targets: text labels on top, delayed audio codebooks below")
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(OMNI_MAX_LEN, STOP, mo, setup_seed, train_ds):
    # 모든 학습 샘플이 길이 안에 들어가는지(STOP까지 잘리지 않는지) 확인합니다.
    setup_seed(0)
    _ok = 0
    for _j in range(len(train_ds)):
        _al = train_ds[_j][2]
        assert all((_al[i] == STOP).any() for i in range(8)), f"샘플 {_j}가 길이 {OMNI_MAX_LEN}에서 잘렸습니다"
        _ok += 1
    mo.md(f"학습 샘플 {_ok}개 모두 코드북 8개가 STOP까지 들어 있습니다.")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Talker 순전파 해부

    `MiniMindOmni.forward`에서 Talker 입력은 두 신호의 합입니다.

    $$
    z = \underbrace{\text{embed\_proj}(h^{\text{bridge}})}_{\text{무엇을 말할지}} \cdot \text{text\_scale}
      + \underbrace{\text{codec\_proj}(\text{TalkerEmbedding}(\text{이전 코드 8줄}))}_{\text{지금까지 낸 소리}} \cdot \text{audio\_scale}
    $$

    - $h^{\text{bridge}}$: Thinker **층 3의 출력**. 마지막 층이 아니라 중간 층을 쓰는 이유는 텍스트 예측에
      특화되기 전의, 내용 정보가 풍부한 표현을 쓰기 위해서입니다.
    - `TalkerEmbedding`: 코드북 8개의 임베딩을 **평균**해 위치마다 벡터 하나로 만듭니다
      (공유 임베딩 + 코드북별 저랭크 adapter).
    - `TalkerHead`: 공유 `Linear` 하나 + 코드북별 adapter 8개로 **8개의 logits를 동시에** 냅니다.

    아래 셀은 이 계산을 원본 모듈들로 **손으로 다시 수행**하고, 모델 출력과 같은지 확인합니다.
    같은 함수를 8절에서 학습 후 두 신호의 크기를 비교할 때 다시 씁니다.
    """)
    return


@app.cell
def _(DEVICE, DataLoader, omni_collate_fn, setup_seed, torch, train_ds):
    setup_seed(0)
    probe_batch = next(iter(DataLoader(train_ds, batch_size=4, shuffle=False, collate_fn=omni_collate_fn)))[0].to(DEVICE)

    @torch.no_grad()
    def talker_by_hand(model, x):
        cfg = model.config
        text_ids, audio_ids = x[:, 8, :], x[:, :8, :]
        T = x.shape[-1]
        th, tk = model.thinker, model.talker
        h = th.embed_tokens(text_ids)
        pos = (th.freqs_cos[:T], th.freqs_sin[:T])
        for i, layer in enumerate(th.layers):
            h, _ = layer(h, pos)
            if i == cfg.bridge_layer:
                bridge = h
        text_part = tk.embed_proj(bridge) * tk.text_scale
        codec_emb = tk.embed_tokens(audio_ids)
        audio_part = tk.codec_proj(codec_emb) * tk.audio_scale
        z = text_part + audio_part
        tpos = (tk.freqs_cos[:T], tk.freqs_sin[:T])
        for layer in tk.layers:
            z, _ = layer(z, tpos)
        logits = tk.lm_head(tk.norm(z))
        return dict(bridge=bridge, codec_emb=codec_emb, text_part=text_part, audio_part=audio_part, logits=logits)

    return probe_batch, talker_by_hand


@app.cell(hide_code=True)
def _(base_model, mo, probe_batch, talker_by_hand, torch):
    with torch.no_grad():
        _ref = base_model(probe_batch)
    _mine = talker_by_hand(base_model, probe_batch)
    _diff = max((a - b).abs().max().item() for a, b in zip(_mine["logits"], _ref.audio_logits))
    assert _diff < 1e-3, _diff

    _head = base_model.talker.lm_head
    _head_base = sum(p.numel() for p in _head.base.parameters())
    _head_ad = sum(p.numel() for p in _head.adapters.parameters())
    mo.md(
        f"""
    직접 계산한 audio logits와 `model(x).audio_logits`의 최대 차이: **{_diff:.1e}** → 같은 계산입니다.

    | 텐서 | shape | 설명 |
    |---|---|---|
    | 입력 `x` | `{tuple(probe_batch.shape)}` | [B, 오디오 8줄 + 텍스트 1줄, T] |
    | bridge 은닉 상태 | `{tuple(_mine["bridge"].shape)}` | Thinker 층 {base_model.config.bridge_layer} 출력 |
    | `TalkerEmbedding` 출력 | `{tuple(_mine["codec_emb"].shape)}` | 코드북 8개 임베딩의 평균 |
    | Talker logits | {len(_mine["logits"])} × `{tuple(_mine["logits"][0].shape)}` | 코드북마다 오디오 어휘 2112개에 대한 점수 |

    `TalkerHead` 파라미터: 공유 Linear {_head_base / 1e6:.2f}M + adapter 8개 {_head_ad / 1e6:.2f}M.
    시작값: `text_scale` = {base_model.talker.text_scale.item():.1f}, `audio_scale` = {base_model.talker.audio_scale.item():.1f}
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 4-1. 학습 전 Talker는 무슨 소리를 낼까?

    원본 `generate(stream=True, return_audio_codes=True)`는 매 단계 `(텍스트, 오디오 프레임)`을 돌려줍니다.
    프레임은 8개 코드북이 **모두 아직 STOP 전일 때만** 나옵니다. 학습 전 Talker는 Thinker 층을 복사했을 뿐
    입·출력층(`TalkerEmbedding`, `TalkerHead`)이 무작위이므로, 오디오 어휘 2112개 중 아무 토큰이나 뽑습니다.
    그중 2048 이상(특수 토큰)이 나오면 그 코드북은 바로 멈춘 것으로 처리됩니다.

    Whisper는 잡음만 들어도 *"I'm sorry."* 같은 짧은 문장을 지어내기도 합니다. 받아쓰기 결과만 믿지 말고
    직접 들어 보세요.
    """)
    return


@app.cell
def _(DEVICE, codes_to_wav, prompt_ids, tokenizer, torch):
    @torch.no_grad()
    def speak(model, question, max_new_tokens=160, seed=0):
        """Thinker가 답을 생성하고, Talker가 동시에 그 답을 음성 코드로 만듭니다."""
        torch.manual_seed(seed)
        x = torch.tensor([prompt_ids(question)], device=DEVICE)
        text_ids, frames = None, []
        for y, frame in model.generate(
            x,
            tokenizer.eos_token_id,
            max_new_tokens=max_new_tokens,
            temperature=0.01,
            top_p=1.0,
            stream=True,
            return_audio_codes=True,
        ):
            if y is not None:
                text_ids = y
            if frame:
                frames.append(frame)
        text = tokenizer.decode(text_ids[0].tolist(), skip_special_tokens=True).strip() if text_ids is not None else ""
        wav = codes_to_wav(torch.tensor(frames).T) if frames else None
        return text, wav, len(frames)

    return (speak,)


@app.cell(hide_code=True)
def _(base_model, json, mo, player, speak, train_rows, transcribe):
    _q = json.loads(train_rows[0]["conversations"])[0]["content"]
    _text, _wav, _n = speak(base_model, _q)
    mo.vstack(
        [
            mo.md(
                f"""
    - 질문: *{_q}*
    - Thinker 텍스트: *{_text}*
    - 완성된 오디오 프레임: **{_n}개**, Whisper 받아쓰기: *{transcribe(_wav) or "(없음)"}*
    """
            ),
            player(_wav, "학습 전 Talker"),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. GPU 학습: 실험 3개

    같은 데이터, 같은 step 수로 무엇을 학습하느냐만 바꿉니다.

    | 실험 | 학습하는 파라미터 | 목적 |
    |---|---|---|
    | **A. Talker만** | `talker.*` (Thinker 고정) | Thinker의 텍스트 능력을 지키면서 목소리만 붙일 수 있는가 |
    | **B. 텍스트 정보 차단** | `talker.*`, 단 `text_scale = 0` 고정 | Thinker의 텍스트 정보를 차단하고 이전 오디오 코드만으로 예측 (대조군) |
    | **C. 전체** | 모든 파라미터 (원본 `--freeze_backbone none` 기본값) | 텍스트와 음성을 함께 학습할 때 무엇을 얻고 무엇을 잃는가 |

    손실은 원본 `train_sft_omni.py`와 같습니다.

    $$
    L = L_{\text{text}} + \frac{1}{8}\sum_{i=0}^{7} L_{\text{audio}, i} + L_{\text{aux}}
    $$

    $L_{\text{audio}, i}$는 코드북 $i$의 cross-entropy이며 STOP 위치에 10배 가중치를 줍니다.
    학습률은 원본과 같은 코사인 스케줄(`get_lr`)을 쓰고, bf16 autocast로 계산합니다.

    > **주의: 이 규모에서 모델은 일반화가 아니라 암기를 합니다.** 문장 수십 개로는 처음 보는 문장을 읽는 능력이
    > 생기지 않습니다. 이 실습의 목표는 *학습한 문장*에서 구조의 각 부분이 하는 일을 확인하는 것입니다.
    """)
    return


@app.cell
def _(
    DEVICE,
    DataLoader,
    STOP,
    SkipBatchSampler,
    copy,
    get_lr,
    nn,
    np,
    omni_collate_fn,
    setup_seed,
    time,
    torch,
    train_ds,
):
    LEARNING_RATES = {"talker": 5e-4, "blind": 5e-4, "all": 2e-4}

    def run_training(base, mode, steps, batch_size=16, seed=42):
        model = copy.deepcopy(base)
        for p in model.parameters():
            p.requires_grad = False
        if mode in ("talker", "blind"):
            for p in model.talker.parameters():
                p.requires_grad = True
            if mode == "blind":
                with torch.no_grad():
                    model.talker.text_scale.fill_(0.0)
                model.talker.text_scale.requires_grad = False
        elif mode == "all":
            for p in model.parameters():
                p.requires_grad = True
        else:
            raise ValueError(mode)

        params = [p for p in model.parameters() if p.requires_grad]
        lr = LEARNING_RATES[mode]
        opt = torch.optim.AdamW(params, lr=lr)
        ce = nn.CrossEntropyLoss(reduction="none")
        hist = {"text": [], "audio": [], "per_codebook": []}
        torch.cuda.reset_peak_memory_stats()
        model.train()
        step, epoch, t0 = 0, 0, time.time()
        while step < steps:
            setup_seed(seed + epoch)
            order = torch.randperm(len(train_ds)).tolist()
            loader = DataLoader(train_ds, batch_sampler=SkipBatchSampler(order, batch_size, 0), collate_fn=omni_collate_fn)
            for x, y_text, y_audio, *_ in loader:
                if step >= steps:
                    break
                step += 1
                x, y_text, y_audio = x.to(DEVICE), y_text.to(DEVICE), y_audio.to(DEVICE)
                for g in opt.param_groups:
                    g["lr"] = get_lr(step, steps, lr)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    out = model(x)
                    tl = ce(out.logits.reshape(-1, out.logits.size(-1)), y_text.reshape(-1))
                    tm = (y_text.reshape(-1) != -100).float()
                    text_loss = (tl * tm).sum() / (tm.sum() + 1e-9)
                    per_cb = []
                    for i, logits in enumerate(out.audio_logits):
                        tgt = y_audio[:, i, :].reshape(-1)
                        ll = ce(logits.reshape(-1, logits.size(-1)), tgt)
                        valid = (tgt != -100).float()
                        weight = valid * (1 + (tgt == STOP).float() * 9)
                        per_cb.append((ll * weight).sum() / (valid.sum() + 1e-9))
                    audio_loss = sum(per_cb) / 8
                    loss = text_loss + audio_loss + out.aux_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)
                hist["text"].append(text_loss.item())
                hist["audio"].append(audio_loss.item())
                hist["per_codebook"].append([v.item() for v in per_cb])
            epoch += 1
        model.eval()
        info = {
            "mode": mode,
            "steps": steps,
            "epochs": epoch,
            "lr": lr,
            "trainable_M": sum(p.numel() for p in params) / 1e6,
            "seconds": time.time() - t0,
            "peak_mem_GB": torch.cuda.max_memory_allocated() / 1e9,
            "final_text": float(np.mean(hist["text"][-20:])),
            "final_audio": float(np.mean(hist["audio"][-20:])),
        }
        return model, hist, info

    def train_summary(info):
        return (
            f"**{info['mode']}**: 학습 파라미터 {info['trainable_M']:.1f}M, {info['steps']} step ({info['epochs']} epoch), "
            f"{info['seconds']:.0f}초, 최대 VRAM {info['peak_mem_GB']:.1f}GB, "
            f"마지막 20 step 평균 손실 text {info['final_text']:.3f} / audio {info['final_audio']:.3f}"
        )

    return run_training, train_summary


@app.cell(hide_code=True)
def _(base_model, mo, run_training, steps_ui, train_summary):
    with mo.status.spinner("실험 A: Talker만 학습 중..."):
        model_a, hist_a, info_a = run_training(base_model, "talker", int(steps_ui.value))
    mo.md(train_summary(info_a))
    return hist_a, model_a


@app.cell(hide_code=True)
def _(base_model, mo, run_training, steps_ui, train_summary):
    with mo.status.spinner("실험 B: 텍스트 정보 차단 학습 중..."):
        model_b, hist_b, info_b = run_training(base_model, "blind", int(steps_ui.value))
    mo.md(train_summary(info_b))
    return hist_b, model_b


@app.cell(hide_code=True)
def _(base_model, mo, run_training, steps_ui, train_summary):
    with mo.status.spinner("실험 C: 전체 학습 중..."):
        model_c, hist_c, info_c = run_training(base_model, "all", int(steps_ui.value))
    mo.md(train_summary(info_c))
    return hist_c, model_c


@app.cell
def _(hist_a, hist_b, hist_c, np, plt):
    def _smooth(v, w=20):
        v = np.asarray(v)
        return np.convolve(v, np.ones(w) / w, mode="valid") if len(v) > w else v

    _fig, _axes = plt.subplots(1, 3, figsize=(13, 3.4))
    for _name, _h, _col in [("A talker", hist_a, "C0"), ("B text blocked", hist_b, "C1"), ("C all", hist_c, "C2")]:
        _axes[0].plot(_smooth(_h["audio"]), label=_name, color=_col)
        _axes[1].plot(_smooth(_h["text"]), label=_name, color=_col)
    _axes[0].set_title("audio loss (mean of 8 codebooks)")
    _axes[0].set_yscale("log")
    _axes[1].set_title("text loss (Thinker)")
    _axes[0].set_xlabel("step")
    _axes[1].set_xlabel("step")
    _axes[0].legend()

    _x = np.arange(8)
    for _off, (_name, _h, _col) in zip([-0.27, 0, 0.27], [("A", hist_a, "C0"), ("B", hist_b, "C1"), ("C", hist_c, "C2")]):
        _axes[2].bar(_x + _off, np.mean(_h["per_codebook"][-50:], axis=0), width=0.27, label=_name, color=_col)
    _axes[2].set_title("final loss per codebook (last 50 steps)")
    _axes[2].set_xticks(_x, [f"cb{i}" for i in range(8)])
    _axes[2].legend()
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **손실 그래프 읽기**

    - A와 B는 Thinker를 고정했으므로 **텍스트 손실이 그대로**입니다. 베이스 모델이 이 답변들을 처음 보는
      문장으로서 예측할 때의 손실입니다. C만 텍스트 손실이 0 가까이 떨어집니다. 학습 문장을 **외운** 것입니다.
    - B(텍스트 정보 차단)도 오디오 손실이 꽤 낮아집니다. 학습할 때는 정답 코드가 입력으로 들어오므로(teacher forcing),
      처음 몇 프레임만 지나면 이전 소리만 보고도 어느 문장인지 알 수 있기 때문입니다.
      **손실이 낮다고 생성이 잘 되는 것은 아닙니다.** 6절에서 확인합니다.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. 평가

    ### 6-1. 받아 읽기: 텍스트를 고정하고 Talker만 평가

    Thinker가 만든 텍스트가 실험마다 다르면 Talker를 공정하게 비교할 수 없습니다. 그래서 **정답 텍스트를
    강제로 넣고** Talker만 코드를 생성하게 합니다(`read_aloud`). 위치와 코드북 지연 규칙, 샘플링 규칙
    (temperature 0.2, top-k 50, 최근 3개 코드 반복 억제 1.05)은 원본 `stream_generate`와 같고,
    KV 캐시로 한 위치씩 진행합니다.

    생성된 음성을 Whisper로 받아 적어 정답 문장과의 WER을 잽니다.
    - **학습 문장 16개**: 암기한 문장을 제대로 읽는가
    - **확인용 문장 16개**: 처음 보는 문장도 읽는가 (일반화)
    """)
    return


@app.cell
def _(DEVICE, PAD, codes_to_wav, prompt_ids, tokenizer, torch):
    @torch.no_grad()
    def read_aloud(model, question, answer, seed=0, max_frames=120):
        torch.manual_seed(seed)
        p = prompt_ids(question)
        a = tokenizer(answer + tokenizer.eos_token, add_special_tokens=False).input_ids
        S = len(p)
        total = S + max(len(a) + 1, max_frames + 9)
        # 텍스트가 끝나면 원본 generate처럼 줄바꿈(201) 한 번, 그 뒤로는 pad(0)를 넣습니다.
        text = torch.tensor((p + a + [201] + [0] * total)[:total], device=DEVICE)
        audio = torch.full((8, total), PAD, dtype=torch.long, device=DEVICE)
        stop = [None] * 8
        history = [[] for _ in range(8)]

        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(torch.cat([audio[:, :S], text[None, :S]])[None], use_cache=True)
        past, pos = out.past_key_values, S - 1
        while pos < total - 1:
            for i in range(8):
                if pos - S >= i and stop[i] is None:
                    logits = out.audio_logits[i][0, -1].float() / 0.2
                    for prev in history[i][-3:]:
                        logits[prev] = torch.where(logits[prev] > 0, logits[prev] / 1.05, logits[prev] * 1.05)
                    val, idx = logits.topk(50)
                    code = idx[torch.multinomial(torch.softmax(val, -1), 1)].item()
                    history[i].append(code)
                    audio[i, pos + 1] = code
                    if code >= 2048:
                        stop[i] = pos + 1
            if all(s is not None for s in stop):
                break
            pos += 1
            col = torch.cat([audio[:, pos : pos + 1], text[None, pos : pos + 1]])[None]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(col, past_key_values=past, use_cache=True)
            past = out.past_key_values

        # 계단을 되돌려 프레임 k = [cb0[S+1+k], cb1[S+2+k], ..., cb7[S+8+k]]로 모읍니다.
        frames = []
        for k in range(total):
            cols = [S + 1 + k + i for i in range(8)]
            if cols[-1] >= total:
                break
            f = [int(audio[i, c]) for i, c in enumerate(cols)]
            if any(c >= 2048 for c in f):
                break
            frames.append(f)
        return codes_to_wav(torch.tensor(frames).T) if frames else None

    return (read_aloud,)


@app.cell(hide_code=True)
def _(
    heldout_rows,
    json,
    mo,
    model_a,
    model_b,
    model_c,
    np,
    read_aloud,
    train_rows,
    transcribe,
    wer,
):
    N_EVAL = 16
    _models = {"A talker": model_a, "B text blocked": model_b, "C all": model_c}
    read_results = {}
    with mo.status.progress_bar(total=len(_models) * 2 * N_EVAL, title="받아 읽기 평가") as _bar:
        for _name, _m in _models.items():
            for _split, _rows in [("train", train_rows[:N_EVAL]), ("heldout", heldout_rows[:N_EVAL])]:
                _items = []
                for _r in _rows:
                    _c = json.loads(_r["conversations"])
                    _wav = read_aloud(_m, _c[0]["content"], _c[1]["content"])
                    _hyp = transcribe(_wav)
                    _items.append({"ref": _c[1]["content"], "hyp": _hyp, "wer": wer(_c[1]["content"], _hyp), "wav": _wav})
                    _bar.update()
                read_results[(_name, _split)] = _items

    read_wer = {k: float(np.mean([it["wer"] for it in v])) for k, v in read_results.items()}
    mo.md(
        "| 실험 | WER 학습 문장 | WER 확인용 문장 |\n|---|---|---|\n"
        + "\n".join(f"| {n} | {read_wer[(n, 'train')]:.2f} | {read_wer[(n, 'heldout')]:.2f} |" for n in _models)
    )
    return (read_results,)


@app.cell(hide_code=True)
def _(mo, player, read_results):
    def _example(split, idx):
        items = {n: read_results[(n, split)][idx] for n in ["A talker", "B text blocked", "C all"]}
        ref = items["A talker"]["ref"]
        rows = "\n".join(f"| {n} | {it['hyp'] or '(없음)'} | {it['wer']:.2f} |" for n, it in items.items())
        return mo.vstack(
            [
                mo.md(f"**정답**: *{ref}*\n\n| 실험 | Whisper 받아쓰기 | WER |\n|---|---|---|\n{rows}"),
                mo.hstack([player(it["wav"], n) for n, it in items.items()], wrap=True),
            ]
        )

    mo.ui.tabs({"학습 문장 예시": _example("train", 0), "확인용 문장 예시": _example("heldout", 0)})
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **결과 읽기** (기본 설정 측정값: 학습 문장 WER A 0.31 / B 0.94 / C 0.17, 확인용 문장은 셋 다 0.94 이상)

    - **A**는 Thinker를 건드리지 않고도 학습 문장을 알아들을 수 있게 읽습니다. 텍스트 정보가 bridge를 통해
      Talker로 전달된다는 뜻입니다.
    - **B**는 손실은 낮았지만 받아 읽기는 실패합니다. 대개 *엉뚱한 학습 문장*을 또렷하게 말합니다.
      Thinker 정보 없이는 첫 프레임에서 어느 문장을 말할지 정할 근거가 없고, 일단 시작하면 외운 문장 하나를
      끝까지 이어 갑니다. 학습 때 정답 코드를 넣어 주던 teacher forcing의 함정입니다.
    - **C**는 Thinker도 함께 바뀌어 학습 문장을 가장 정확하게 읽습니다.
    - 처음 보는 문장은 세 실험 모두 읽지 못합니다. 문장 수십 개로는 *텍스트 → 발음* 대응을 일반화할 수 없습니다.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 6-2. 전체 파이프라인: Thinker가 답하고 Talker가 말하기

    이번에는 텍스트를 강제하지 않습니다. 질문만 주고 원본 `generate`로 텍스트와 음성을 함께 생성합니다.
    """)
    return


@app.cell(hide_code=True)
def _(json, model_a, model_c, result_table, speak, train_rows, transcribe):
    _rows = []
    for _r in train_rows[:3]:
        _c = json.loads(_r["conversations"])
        for _name, _m in [("A talker", model_a), ("C all", model_c)]:
            _text, _wav, _n = speak(_m, _c[0]["content"])
            _rows.append(
                {
                    "질문": _c[0]["content"][:60],
                    "실험": _name,
                    "학습 답변": _c[1]["content"][:90],
                    "Thinker 텍스트": _text[:90],
                    "Whisper 받아쓰기": transcribe(_wav)[:90],
                    "프레임": _n,
                }
            )
    result_table(_rows, selection=None)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    - **A**의 Thinker는 베이스 그대로이므로 학습 데이터의 답과 **다른 문장**을 말합니다. Talker는 그 문장을
      처음 보므로 제대로 읽지 못합니다. 목소리를 붙이려면 Talker가 *어떤 문장이든* 읽을 수 있어야 하고,
      그러려면 훨씬 많은 데이터가 필요합니다.
    - **C**는 Thinker가 학습 답변을 외웠으므로 텍스트와 음성이 함께 맞습니다. 대신 대가가 있는지
      다음 셀에서 확인합니다.
    """)
    return


@app.cell(hide_code=True)
def _(
    GENERAL_QUESTIONS,
    base_answers,
    base_model,
    chat_text,
    mo,
    model_a,
    model_b,
    model_c,
    result_table,
    torch,
):
    # A, B는 Thinker 파라미터가 베이스와 비트 단위까지 같아야 합니다.
    for _m in (model_a, model_b):
        assert all(torch.equal(p, q) for p, q in zip(_m.thinker.parameters(), base_model.thinker.parameters()))

    _rows, retention = [], {"A": 0, "C": 0}
    for _q in GENERAL_QUESTIONS:
        torch.manual_seed(0)
        _ans_a = chat_text(model_a, _q)
        torch.manual_seed(0)
        _ans_c = chat_text(model_c, _q)
        retention["A"] += _ans_a == base_answers[_q]
        retention["C"] += _ans_c == base_answers[_q]
        _rows.append({"질문": _q, "베이스": base_answers[_q][:80], "A talker": _ans_a[:80], "C all": _ans_c[:80]})
    mo.vstack(
        [
            mo.md(
                f"**텍스트 능력 보존**: 일반 질문 {len(GENERAL_QUESTIONS)}개 중 베이스와 같은 답 — "
                f"A {retention['A']}개, C {retention['C']}개. (A·B의 Thinker 파라미터가 베이스와 완전히 같음을 확인)"
            ),
            result_table(_rows, selection=None),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    C는 학습 문장을 외우는 동안 Thinker 전체가 바뀌어 일반 질문에 대한 답이 망가집니다
    (**파국적 망각, catastrophic forgetting**). 실제 MiniMind-O 학습이 대량의 다양한 대화로
    전체를 학습하는 이유이고, 작은 데이터로 목소리만 붙이고 싶다면 A처럼 Thinker를 고정하는 편이 안전합니다.
    원본 `train_sft_omni.py`에서는 `--freeze_backbone all`이 A에 가깝습니다.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7. 스트리밍: 첫 음성은 언제 나오는가

    `stream_generate`는 한 단계마다 텍스트 토큰 하나와 (가능하면) 오디오 프레임 하나를 냅니다.

    - 단계 0: 프롬프트를 처리하고 첫 텍스트 토큰을 냅니다. 오디오는 아직 없습니다 (`audio_step = step - 1`).
    - 단계 1부터 코드북 0이, 단계 2부터 코드북 1이 … 단계 8부터 코드북 7이 코드를 냅니다.
    - 따라서 **첫 완성 프레임은 단계 8**에서 나옵니다. 프레임 하나가 80ms이므로, 이 지연을 줄이는 것이
      실시간 음성 대화의 핵심입니다.
    - 텍스트가 끝나도 남은 오디오 코드가 STOP을 낼 때까지 계속 진행합니다.
    """)
    return


@app.cell(hide_code=True)
def _(
    DEVICE,
    json,
    mo,
    model_c,
    prompt_ids,
    result_table,
    time,
    tokenizer,
    torch,
    train_rows,
):
    _q = json.loads(train_rows[1]["conversations"])[0]["content"]
    torch.manual_seed(0)
    _x = torch.tensor([prompt_ids(_q)], device=DEVICE)
    stream_log = []
    torch.cuda.synchronize()
    _t0 = time.time()
    for _step, (_y, _frame) in enumerate(
        model_c.generate(
            _x, tokenizer.eos_token_id, max_new_tokens=160, temperature=0.01, top_p=1.0, stream=True, return_audio_codes=True
        )
    ):
        torch.cuda.synchronize()
        stream_log.append(
            {
                "step": _step,
                "ms": round((time.time() - _t0) * 1000, 1),
                "텍스트": repr(tokenizer.decode([int(_y[0, -1])])) if _y is not None else "(끝)",
                "프레임": "✔" if _frame else "",
            }
        )
    _framed = [r for r in stream_log if r["프레임"]]
    assert _framed, "프레임이 하나도 나오지 않았습니다"
    stream_stats = {
        "first_frame_step": _framed[0]["step"],
        "first_frame_ms": _framed[0]["ms"],
        "text_end_step": max(r["step"] for r in stream_log if r["텍스트"] != "(끝)"),
        "total_steps": len(stream_log),
        "frames": len(_framed),
        "last_frame_step": _framed[-1]["step"],
        "audio_sec": len(_framed) / 12.5,
        "wall_sec": _framed[-1]["ms"] / 1000,
    }
    assert stream_stats["first_frame_step"] == 8, stream_stats
    _cap_note = (
        "최대 단계(160)에 도달했습니다. 일부 코드북이 STOP을 내지 않아 생성이 끝까지 이어졌기 때문입니다. "
        "프레임은 한 코드북이라도 STOP을 내면 더 나오지 않습니다."
        if stream_stats["total_steps"] >= 160
        else "모든 코드북이 STOP을 내서 생성이 끝났습니다."
    )
    mo.vstack(
        [
            mo.md(
                f"""
    질문: *{_q}* (실험 C 모델)

    - 첫 완성 프레임: **단계 {stream_stats['first_frame_step']}** (생성 시작 후 {stream_stats['first_frame_ms']:.0f}ms)
    - 텍스트 마지막 토큰: 단계 {stream_stats['text_end_step']}, 마지막 프레임: 단계 {stream_stats['last_frame_step']}
      (프레임 {stream_stats['frames']}개 = 음성 {stream_stats['audio_sec']:.2f}초)
    - 마지막 프레임까지 걸린 시간 {stream_stats['wall_sec']:.2f}초 → 실시간 대비 {stream_stats['audio_sec'] / stream_stats['wall_sec']:.1f}배
    - 전체 단계 {stream_stats['total_steps']}: {_cap_note}
    """
            ),
            result_table(stream_log[:24], selection=None, page_size=24),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 8. 학습된 스케일: Talker는 두 신호를 얼마나 섞는가

    Talker 입력 $z$의 두 항(텍스트 쪽, 오디오 쪽) 크기를 4절의 `talker_by_hand`로 같은 배치에서 측정합니다.
    값은 위치별 L2 노름의 평균입니다.
    """)
    return


@app.cell(hide_code=True)
def _(base_model, mo, model_a, model_b, model_c, probe_batch, talker_by_hand):
    _rows = []
    scale_stats = {}
    for _name, _m in [("학습 전", base_model), ("A talker", model_a), ("B text blocked", model_b), ("C all", model_c)]:
        _parts = talker_by_hand(_m, probe_batch)
        _tn = _parts["text_part"].norm(dim=-1).mean().item()
        _an = _parts["audio_part"].norm(dim=-1).mean().item()
        scale_stats[_name] = {
            "text_scale": _m.talker.text_scale.item(),
            "audio_scale": _m.talker.audio_scale.item(),
            "text_norm": _tn,
            "audio_norm": _an,
        }
        _rows.append(
            f"| {_name} | {_m.talker.text_scale.item():.2f} | {_m.talker.audio_scale.item():.2f} | {_tn:.1f} | {_an:.1f} | {_tn / max(_an, 1e-6):.2f} |"
        )
    mo.md(
        "| 모델 | text_scale | audio_scale | 텍스트 항 크기 | 오디오 항 크기 | 텍스트/오디오 |\n|---|---|---|---|---|---|\n"
        + "\n".join(_rows)
        + "\n\n두 항 모두 RMSNorm으로 끝나므로 크기는 대체로 스케일 값에 비례합니다. "
        "시작값 3:1은 *처음에는 텍스트 정보를 더 크게 들으라*는 설계이고, B는 텍스트 항이 0이라 오디오 항만 남습니다."
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 9. 음성 입력 경로 (A2A): `inject_audio_features`

    지금까지는 **텍스트 → 음성(T2A)**이었습니다. 음성으로 질문하는 경로(A2A)는 다음과 같습니다.

    1. 질문 음성을 SenseVoice 인코더로 특징 벡터열 `[L, 512]`로 바꿉니다 (`funasr`가 필요해 이 실습에서는 불러오지 않음).
    2. `audio_proj`(LayerNorm → Linear → GELU → Linear)가 `[L, 768]`로 바꿉니다.
    3. 프롬프트에 `<|audio_pad|>` 토큰(id 16)을 L개 넣어 자리를 만듭니다.
    4. `inject_audio_features`가 그 자리의 텍스트 임베딩을 음성 특징으로 **바꿔치기**합니다.

    이후 Thinker 입장에서는 그냥 768차원 벡터열이므로, 텍스트로 질문했을 때와 같은 방식으로 답합니다.
    아래 셀은 실제 인코더 대신 같은 shape의 가짜 특징으로 2~4단계를 확인합니다.
    """)
    return


@app.cell(hide_code=True)
def _(DEVICE, base_model, mo, omni_config, tokenizer, torch):
    _L = 23
    _text = tokenizer.apply_chat_template(
        [{"role": "user", "content": omni_config.audio_special_token * _L + "\n\nPlease answer the question."}],
        tokenize=False,
        add_generation_prompt=True,
    )
    _ids = torch.tensor([tokenizer(_text, add_special_tokens=False).input_ids], device=DEVICE)
    _slots = (_ids[0] == omni_config.audio_ids[0]).nonzero().squeeze(1)
    assert len(_slots) == _L

    torch.manual_seed(0)
    with torch.no_grad():
        _fake_encoder_out = torch.randn(_L, omni_config.audio_hidden_size, device=DEVICE)  # [L, 512]
        _feats = base_model.audio_proj(_fake_encoder_out)  # [L, 768]
        _h = base_model.thinker.embed_tokens(_ids)
        _h2 = base_model.inject_audio_features(_ids, _h.clone(), [_feats], _ids.shape[1])

    _changed = ((_h2 - _h).abs().sum(-1)[0] > 0).nonzero().squeeze(1)
    assert torch.equal(_changed, _slots)
    assert torch.allclose(_h2[0, _slots], _feats)

    mo.md(
        f"""
    - 프롬프트 길이 {_ids.shape[1]}, `<|audio_pad|>` 위치 {_slots[0].item()}~{_slots[-1].item()} ({_L}개)
    - 가짜 인코더 출력 `{tuple(_fake_encoder_out.shape)}` → `audio_proj` → `{tuple(_feats.shape)}`
    - 임베딩이 바뀐 위치가 `<|audio_pad|>` 위치와 정확히 일치하고, 나머지 위치는 그대로입니다.
    - 원본 데이터셋은 학습 때 음성 자리를 질문 앞·뒤 또는 단독으로 무작위 배치합니다 (`create_chat_prompt`).
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 10. 실제 규모로 학습하려면

    이 노트북은 구조를 이해하기 위한 축소판입니다. 원본 학습 스크립트로 같은 베이스를 쓰려면,
    베이스 가중치를 `projects/minimind-o/out/llm_768.pth`로 저장한 뒤 `--from_weight llm`으로 시작합니다.
    원본 `init_omni_model`이 1-1절처럼 Talker 층을 Thinker 뒤쪽 층으로 초기화합니다.

    ```python
    # 노트북 셀에서 실행 (base_dir은 1절에서 내려받은 경로)
    out = OMNI_DIR / "out"; out.mkdir(exist_ok=True)
    torch.save({k: v.half() for k, v in load_file(base_dir / "model.safetensors").items()}, out / "llm_768.pth")
    ```

    ```bash
    cd projects/minimind-o/trainer
    # 1) 텍스트 → 음성 (T2A). 데이터는 jingyaogong/minimind-o_dataset에서 ../dataset/로 내려받습니다.
    python train_sft_omni.py --from_weight llm --save_weight sft_zero \
        --data_path ../dataset/sft_t2a_mini.parquet --max_seq_len 512 --batch_size 40 --learning_rate 5e-4
    # 2) 음성 입력 (A2A): audio_proj만 정렬 → 전체 미세 조정 (SenseVoice 인코더와 funasr 필요)
    python train_sft_omni.py --from_weight sft_zero --save_weight sft_zero --mode audio_proj \
        --data_path ../dataset/sft_a2a_mini.parquet --max_seq_len 640
    ```

    주의할 점
    - 베이스 모델은 **길이 340**으로 학습되었습니다. Omni 학습의 512~768 길이에서는 위치 일반화가 검증되지 않았습니다.
    - `projects/minimind-o/trainer/train.sh`의 소요 시간은 RTX 3090 1장 기준 참고값입니다.
    - Thinker를 지키려면 `--freeze_backbone all`(실험 A에 가까움)을 고려하세요.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 11. 정리

    | 구성 요소 | 하는 일 | 이 노트북에서 확인한 것 |
    |---|---|---|
    | Thinker | 텍스트 답변 생성 (베이스 LLM 그대로) | Omni 순전파의 텍스트 logits = 베이스 logits |
    | bridge (층 3) | Thinker의 *말할 내용*을 Talker로 전달 | 이 연결을 끊은 B는 엉뚱한 문장을 말함 |
    | TalkerEmbedding / codec_proj | 이전 오디오 코드 8줄을 벡터 하나로 | 손 계산과 모델 출력 일치 |
    | Talker 4층 + TalkerHead | 위치마다 코드북 8개를 동시에 예측 | 계단 지연 때문에 첫 프레임은 단계 8 |
    | Mimi | 코드 ↔ 24kHz 파형 | 코드북이 많을수록 WER 감소 |
    | audio_proj + inject | 음성 질문을 Thinker 입력으로 | `<|audio_pad|>` 자리만 바뀜 |

    ## 12. 연습 문제

    1. **bridge 위치 바꾸기**: `OmniConfig(..., bridge_layer=7)`처럼 Thinker 마지막 층을 bridge로 쓰도록 바꾸고
       실험 A를 다시 학습하세요. 받아 읽기 WER은 어떻게 달라지나요? `bridge_layer=0`이면?
    2. **STOP 가중치 없애기**: `run_training`에서 `* 9`를 `* 0`으로 바꾸면 생성된 음성의 길이가 어떻게 되나요?
    3. **일반화 시도**: 학습 문장 수를 256으로, step을 1600으로 늘리세요. 학습 문장 WER과 확인용 문장 WER의 차이는?
    4. **샘플링 온도**: `read_aloud`의 temperature 0.2를 1.0으로 바꾸면 음질이 어떻게 변하나요? 코드북 0과 7 중
       어느 쪽의 온도가 더 중요할지 가설을 세우고 코드북별로 다르게 적용해 보세요.
    5. **절충안**: 실험 C의 학습률을 2e-5로 낮추거나, Thinker는 마지막 층만 학습하도록(`--freeze_backbone last1`)
       바꾸면 받아 읽기 성능과 텍스트 능력 보존 사이의 균형이 어떻게 달라지나요?


    ### 연습 문제의 관찰 기준

    | 과제 | 힌트·비교 기준 |
    | --- | --- |
    | 1. bridge 위치 | 같은 데이터·시드·학습량으로 층 0·3·7을 비교하고 학습/확인용 WER을 함께 기록하세요. 특정 층이 항상 우수하다는 정답은 없습니다. |
    | 2. STOP 가중치 | 생성 프레임 수·종료 토큰 빈도·WER을 함께 보세요. 종료 신호가 약해져 길이가 불안정해질 수 있지만 항상 길어진다고 보장하지 않습니다. |
    | 3. 문장 수 | 고정한 확인 문장을 학습에 넣지 말고 두 WER의 차이를 비교하세요. 학습 WER만 감소하면 일반화 개선의 근거가 부족합니다. |
    | 4. 온도 | 같은 문장·시드에서 코드북별 온도와 전사·음질을 비교하세요. 거친 정보가 있는 코드북 0을 바꾸는 경우와 뒤 코드북을 바꾸는 경우를 나눠 가설을 확인하세요. |
    | 5. 동결·학습률 | 학습 가능한 층과 가중치 변화량을 확인하고 WER·텍스트 CE·생성 답을 함께 비교하세요. 받아 읽기 개선만으로 기존 능력 보존을 판단하지 않습니다. |
    """)
    return


if __name__ == "__main__":
    app.run()
