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
    `uv run marimo edit minimind_omni_vlm_tour.py`. 환경·준비물과 권장 순서는 [공통 README](README.md)를 참고하세요.
    위젯 변경 시 의존 셀이 다시 실행됩니다. 모델 크기·데이터·학습 설정 변경 전에는 해당 절의 재실행 범위를 확인하세요.
    """)
    return


@app.cell
def _():
    import copy
    import io
    import json
    import random
    import re
    import sys
    import tempfile
    import time
    import warnings
    from pathlib import Path

    import librosa
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    import soundfile as sf
    import torch
    import torch.nn as nn
    from huggingface_hub import hf_hub_download, snapshot_download
    from PIL import Image
    from safetensors.torch import load_file
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer, MimiModel, WhisperForConditionalGeneration, WhisperProcessor
    from transformers import logging as hf_logging

    # 노트북 위치(notebooks/minimind/)에서 두 단계 위가 저장소 루트입니다.
    repo_root = Path(__file__).resolve().parents[2]
    omni_dir = repo_root / "projects/minimind-o"
    sys.path.insert(0, str(omni_dir))

    # train_sft_omni.py가 쓰는 원본 클래스와 함수를 그대로 가져옵니다.
    from model.model_omni import MiniMindOmni, OmniConfig
    from dataset.omni_dataset import OmniDataset
    from trainer.trainer_utils import SkipBatchSampler, get_lr, setup_seed
    from trainer.train_sft_omni import omni_collate_fn

    device = "cuda" if torch.cuda.is_available() else "cpu"
    hf_logging.set_verbosity_error()
    # MiniMind-O의 토크나이저는 MiniMind·MiniMind-V와 같은 파일입니다(어휘 6,400개, <|audio_pad|>·<|image_pad|> 포함).
    tokenizer = AutoTokenizer.from_pretrained(omni_dir / "model")

    def wav_bytes(wav, sr=24000):
        """파형(numpy) → 노트북에서 재생할 WAV 바이트"""
        buf = io.BytesIO()
        sf.write(buf, np.asarray(wav, dtype=np.float32), sr, format="WAV")
        return buf.getvalue()

    return (
        DataLoader,
        Image,
        MimiModel,
        MiniMindOmni,
        OmniConfig,
        OmniDataset,
        SkipBatchSampler,
        WhisperForConditionalGeneration,
        WhisperProcessor,
        copy,
        device,
        get_lr,
        hf_hub_download,
        io,
        json,
        librosa,
        load_file,
        mo,
        nn,
        np,
        omni_collate_fn,
        omni_dir,
        pa,
        plt,
        pq,
        random,
        re,
        setup_seed,
        sf,
        snapshot_download,
        tempfile,
        time,
        tokenizer,
        torch,
        warnings,
        wav_bytes,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.vstack([
        mo.md(r"""
        # MiniMind-O: 보고 듣고 말하는 언어 모델 (Omni 모델 따라가기)

        VLM 노트북에서는 언어 모델에 **눈**을 달았습니다. **Omni 모델**은 여기에 **귀**와 **입**을 더합니다.
        글·그림·소리를 입력받고, 답을 **글과 목소리로 동시에** 내보냅니다.

        MiniMind-O의 핵심 아이디어는 두 개의 Transformer를 잇는 것입니다.

        - **Thinker(생각하는 쪽):** 지금까지의 MiniMind 언어 모델. 무엇을 말할지 정하고 글로 답합니다.
        - **Talker(말하는 쪽):** 새로 붙이는 작은 Transformer. Thinker의 속마음(은닉 상태)을 받아 **소리 토큰**을 만듭니다.
          소리 토큰은 음성 코덱 **Mimi**가 실제 목소리(24kHz 파형)로 바꿉니다.

        이 노트북은 이미 그림을 보는 법까지 배운 **64M MiniMind-V**
        [`caspar/MiniMind-V-Dense-64M-LLaVA-Short-v1.0`](https://huggingface.co/caspar/MiniMind-V-Dense-64M-LLaVA-Short-v1.0)을 Thinker로 삼아,
        `projects/minimind-o/model/model_omni.py`의 `MiniMindOmni`에 Talker를 붙이고 **GPU에서 직접 학습**합니다.

        **준비:** **NVIDIA GPU가 필요합니다** (RTX 3080 Ti 12GB에서 확인, 최대 약 8GB 사용). 처음 실행할 때 Hugging Face에서 약 1.6GB를 내려받습니다
        (학습 데이터 parquet 881MB, Mimi 192MB, 이미지 인코더 189MB, Whisper-tiny 151MB, 베이스 모델 130MB). 학습 실험 3개는 각각 1분 안팎이고,
        전체 실행은 내려받기를 빼고 약 5~6분입니다.

        **사전 지식:** VLM 노트북(`minimind_vlm_tour.py`)의 "자리표시자 토큰 자리에 다른 벡터를 갈아 끼우기"와 SFT 노트북의 "assistant 답변만 정답" 규칙.

        ## 0. 전체 구조



        **지표 정의:** 이후 표·해설의 WER은 $\min(1,(S+D+I)/\max(1,N))$로 상한을 1로 제한한 값입니다.
        $S,D,I$는 단어 대치·삭제·삽입 수, $N$은 정답 단어 수입니다. 일반 WER은 삽입 때문에 1을 넘을 수 있습니다.
        이 표시에서 1은 원래 오류율이 100% 이상이라는 뜻이며, 모든 단어가 다르다는 뜻은 아닙니다.
        Whisper의 전사 오류도 섞이므로 발음·음질·화자 유사도를 모두 대변하지 않습니다.
    """),
        mo.mermaid(r"""
    flowchart TB
     TEXT["글 토큰 임베딩"] --> I["모달 특징을 placeholder 위치에 삽입"]
     IMG["그림"] --> SIG["SigLIP2: 고정"] --> VP["vision_proj"] --> I
     AUD["소리"] --> SV["SenseVoice: 고정"] --> AP["audio_proj"] --> I
     I --> TH["Thinker: MiniMind 8층, 약 64M"] --> HEAD["lm_head: 글 토큰"]
     TH -- "bridge_layer=3의 은닉 상태" --> TALK["Talker: 4층, 약 47M"]
     PREV["지난 오디오 코드 8줄"] --> TALK
     TALK --> C["프레임마다 코드북 8개의 토큰"] --> M["Mimi 디코더: 고정"] --> WAV["24kHz 파형"]
    """),
        mo.md(r"""

    | | VLM (`MiniMindVLM`) | Omni (`MiniMindOmni`) |
    | --- | --- | --- |
    | 입력 | 글 + 그림 | 글 + 그림 + **소리** |
    | 출력 | 글 | 글 + **소리 토큰** (Mimi가 목소리로 복원) |
    | 한 위치의 입력 | 토큰 1개 | 토큰 **9개** (소리 코드북 8줄 + 글 1줄) |
    | 새 부품 | `vision_proj` | `vision_proj`, **`audio_proj`**, **`talker`** |
    | 원본 학습 단계 | 정렬 → 시각 SFT | ① T2A(글→소리) → ② A2A(소리→소리) → ③ I2T(그림→글) |

    이 노트북은 Omni만의 부분인 **"말하기"(T2A)** 를 직접 학습하고, 듣기(A2A) 경로는 구조와 코드로 따라갑니다(9절).
    """)
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. 부품 불러오기: VLM이 곧 Thinker

    원본 `init_omni_model`은 `MiniMindOmni(omni_config, ...)`를 만든 뒤 `../out/llm_768.pth`를 `strict=False`로 넣고,
    **Talker 층이 파일에 없으면 Thinker의 마지막 4개 층을 복사해** Talker의 시작점으로 씁니다.

    ```python
    # trainer_utils.init_omni_model 요약
    model.load_state_dict(weights, strict=False)
    if not has_talker and n_talker > 0:
        for i in range(n_talker):                       # Thinker 8층 중 [4:8] → Talker [0:4]
            src = n_thinker - n_talker + i
            model.talker.layers[i].load_state_dict(model.thinker.layers[src].state_dict())
    ```

    여기서는 `llm_768.pth` 대신 Hugging Face의 MiniMind-V를 넣습니다(버전 고정, 저장소의 원격 코드는 실행하지 않음).
    MiniMind-V의 가중치에는 언어 모델(`model.*`)과 **이미 학습된 `vision_proj`** 가 들어 있고, 키 이름이 `MiniMindOmni`와 같습니다.

    - **Thinker + vision_proj:** `caspar/MiniMind-V-Dense-64M-LLaVA-Short-v1.0`
    - **이미지 인코더:** `jingyaogong/siglip2-base-p32-256-ve` (VLM 노트북과 같음)
    - **Talker, audio_proj:** 새 부품. Talker의 Transformer 층만 Thinker에서 복사하고 나머지는 무작위입니다.
    - **소리 인코더 SenseVoice:** 이 노트북에서는 불러오지 않습니다(9절에서 이유와 구조를 설명). `audio_encoder`는 `None`이 됩니다.
    """)
    return


@app.cell(hide_code=True)
def _(
    MiniMindOmni,
    OmniConfig,
    device,
    hf_hub_download,
    json,
    load_file,
    mo,
    omni_dir,
    setup_seed,
    snapshot_download,
    torch,
    warnings,
):
    mo.stop(device != "cuda", mo.callout(mo.md(
        "**GPU를 찾지 못했습니다.** 이 노트북은 NVIDIA GPU에서 학습하도록 만들었습니다. "
        "`uv run python -c 'import torch; print(torch.cuda.is_available())'`로 확인하세요."), kind="danger"))

    BASE_REPO, BASE_REVISION = "caspar/MiniMind-V-Dense-64M-LLaVA-Short-v1.0", "4e6f87f184b27f0692e1f8b3afed959de254894b"
    VISION_REPO, VISION_REVISION = "jingyaogong/siglip2-base-p32-256-ve", "9465d1dc89db6bc6227c5b6b0e0ca9b940325d62"

    vision_dir = snapshot_download(VISION_REPO, revision=VISION_REVISION)
    _cfg = json.load(open(hf_hub_download(BASE_REPO, "config.json", revision=BASE_REVISION)))
    for _key in ("model_type", "model_version"):  # MiniMind-V 전용 표시는 빼고 OmniConfig로 만듦
        _cfg.pop(_key, None)
    omni_config = OmniConfig(**_cfg)  # Talker 4층, audio_ids=[16], audio_vocab_size=2112, bridge_layer=3 등 기본값 추가

    setup_seed(42)  # Talker·audio_proj의 무작위 초깃값을 고정
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # "SenseVoice path not found" 경고: 의도적으로 소리 인코더 없이 만듦
        base_omni = MiniMindOmni(omni_config, audio_encoder_path=str(omni_dir / "model/SenseVoiceSmall"),
                                 vision_model_path=vision_dir)
    _state = {k: v.float() for k, v in load_file(hf_hub_download(BASE_REPO, "model.safetensors", revision=BASE_REVISION)).items()}
    _missing, _unexpected = base_omni.load_state_dict(_state, strict=False)

    # init_omni_model과 같은 Talker 초기화: Thinker 마지막 4층 → Talker 4층
    n_thinker, n_talker = len(base_omni.thinker.layers), len(base_omni.talker.layers)
    for _i in range(n_talker):
        base_omni.talker.layers[_i].load_state_dict(base_omni.thinker.layers[n_thinker - n_talker + _i].state_dict())
    base_omni.to(device).eval()
    base_omni.vision_encoder.to(device)  # 등록된 하위 모듈이 아니라서 train_sft_omni.py처럼 따로 옮김

    # 파일에 없는 것: Talker 전체, audio_proj(새 부품), lm_head(Embedding과 공유), vision_encoder(따로 불러옴)
    assert not _unexpected
    assert {k.split(".")[0] for k in _missing if not k.startswith("vision_encoder.")} == {"talker", "audio_proj", "lm_head"}
    assert base_omni.audio_encoder is None and base_omni.vision_encoder is not None
    assert base_omni.thinker is base_omni.model  # thinker는 model의 별명
    assert torch.equal(base_omni.talker.layers[0].mlp.up_proj.weight, base_omni.thinker.layers[n_thinker - n_talker].mlp.up_proj.weight)

    def count(module):
        return sum(p.numel() for p in module.parameters())

    return BASE_REPO, base_omni, count, n_talker, n_thinker, omni_config


@app.cell(hide_code=True)
def _(
    base_omni,
    count,
    mo,
    n_talker,
    n_thinker,
    omni_config,
    result_table,
    torch,
):
    _t = base_omni.talker
    _rows = [
        {"부품": "vision_encoder (SigLIP2)", "매개변수": f"{count(base_omni.vision_encoder) / 1e6:.1f}M", "어디서": "SigLIP2", "이 노트북에서": "항상 고정"},
        {"부품": "vision_proj", "매개변수": f"{count(base_omni.vision_proj) / 1e6:.2f}M", "어디서": "MiniMind-V (학습됨)", "이 노트북에서": "고정"},
        {"부품": "Thinker (MiniMind 8층 + 임베딩)", "매개변수": f"{count(base_omni.thinker) / 1e6:.1f}M", "어디서": "MiniMind-V", "이 노트북에서": "실험에 따라"},
        {"부품": "Talker: Transformer 4층", "매개변수": f"{count(_t.layers) / 1e6:.1f}M", "어디서": f"Thinker 층 {n_thinker - n_talker}~{n_thinker - 1} 복사", "이 노트북에서": "학습"},
        {"부품": "Talker: 입력부 (embed_tokens, embed_proj, codec_proj, spk_proj)",
         "매개변수": f"{(count(_t.embed_tokens) + count(_t.embed_proj) + count(_t.codec_proj) + count(_t.spk_proj)) / 1e6:.1f}M", "어디서": "무작위", "이 노트북에서": "학습"},
        {"부품": "Talker: 출력부 lm_head (코드북 8개)", "매개변수": f"{count(_t.lm_head) / 1e6:.1f}M", "어디서": "무작위", "이 노트북에서": "학습"},
        {"부품": "audio_proj", "매개변수": f"{count(base_omni.audio_proj) / 1e6:.2f}M", "어디서": "무작위", "이 노트북에서": "쓰지 않음 (9절)"},
    ]
    mo.vstack([
        mo.md(f"실행 장치: **GPU (`{torch.cuda.get_device_name(0)}`)**"),
        result_table(_rows, selection=None),
        mo.md(f"""
    - Talker 전체 **{count(_t) / 1e6:.1f}M** — Thinker({count(base_omni.thinker) / 1e6:.1f}M)에 가까운 크기를 새로 붙입니다.
    - `bridge_layer = {omni_config.bridge_layer}` (= 8 // 2 − 1): Talker는 Thinker의 **마지막 층이 아니라 4번째 층**의 출력을 받습니다(4절).
    - 소리 토큰 어휘 `audio_vocab_size = {omni_config.audio_vocab_size}` = Mimi 코드 2,048개 + 특수 토큰 64개
      (`audio_pad_token={omni_config.audio_pad_token}`, `audio_stop_token={omni_config.audio_stop_token}`, `audio_spk_token={omni_config.audio_spk_token}`).
    """),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 1-1. Thinker는 이미 그림을 볼 수 있다

    `MiniMindOmni.forward`의 그림 경로는 `MiniMindVLM`과 같습니다(`get_image_embeddings` → `vision_proj` → `count_vision_proj`).
    베이스가 MiniMind-V이므로 **아무것도 학습하지 않았는데도** 그림 질문에 글로 답합니다.

    생성에는 원본 `MiniMindOmni.generate(stream=True, return_audio_codes=True)`를 그대로 씁니다.
    이 함수는 한 걸음마다 **글 토큰 1개와 (준비되었으면) 소리 프레임 1개**를 함께 내놓습니다. 아래 `omni_chat`이 이를 모읍니다.
    """)
    return


@app.cell
def _(device, tokenizer, torch):
    def chat_prompt_ids(question, image_token_len=0):
        """추론 프롬프트. MiniMind-V처럼 빈 think 블록은 지웁니다(학습 데이터의 80%도 지운 형태).
        <image>는 <|image_pad|> × image_token_len으로 바꿉니다."""
        text = tokenizer.apply_chat_template([{"role": "user", "content": question}], tokenize=False,
                                             add_generation_prompt=True).replace("<think>\n\n</think>\n\n", "")
        text = text.replace("<image>", "<|image_pad|>" * image_token_len)
        return tokenizer(text, add_special_tokens=False).input_ids

    @torch.inference_mode()
    def omni_chat(model, question, image=None, max_new_tokens=80, seed=0):
        """원본 generate(stream=True)로 글과 소리 프레임을 함께 받습니다.
        글은 거의 탐욕적(temperature 0.01), 소리는 원본 고정 설정(temperature 0.2, top-k 50)으로 뽑습니다."""
        torch.manual_seed(seed)
        if image is not None:
            question = "<image>\n" + question
        x = torch.tensor([chat_prompt_ids(question, model.config.image_token_len)], device=device)
        pixel_values = None
        if image is not None:
            pixel_values = {k: v.to(device) for k, v in model.vision_processor(images=image.convert("RGB"), return_tensors="pt").items()}
        text_ids, frames, steps = [], [], []
        for y, frame in model.generate(x, tokenizer.eos_token_id, max_new_tokens=max_new_tokens, temperature=0.01,
                                       top_p=1.0, stream=True, return_audio_codes=True, pixel_values=pixel_values):
            if y is not None:
                text_ids = y[0].tolist()
            if frame:
                frames.append(frame)
            steps.append((text_ids[-1] if y is not None else None, frame is not None))
        if tokenizer.eos_token_id in text_ids:
            text_ids = text_ids[: text_ids.index(tokenizer.eos_token_id)]
        return {"text": tokenizer.decode(text_ids, skip_special_tokens=True).strip(), "frames": frames, "steps": steps}

    return chat_prompt_ids, omni_chat


@app.cell
def _(Image, base_omni, omni_chat, omni_dir):
    _img_dir = omni_dir.parent / "minimind-v/dataset/eval_images"
    probe_images = [Image.open(_img_dir / n).convert("RGB") for n in ["image-04-yellow-car.jpg", "image-01-golden-dog-balloons.jpg"]]
    base_image_answers = [omni_chat(base_omni, "What is in this image?", image=_im, max_new_tokens=40) for _im in probe_images]
    return base_image_answers, probe_images


@app.cell(hide_code=True)
def _(base_image_answers, io, mo, probe_images):
    def _png(im):
        _b = io.BytesIO(); im.resize((224, 224)).save(_b, format="PNG"); return _b.getvalue()

    mo.vstack([
        mo.hstack([mo.vstack([mo.image(_png(_im), width=160), mo.md(f"*What is in this image?*<br>→ {_a['text']}"),
                              mo.md(f"소리 프레임 {len(_a['frames'])}개 (Talker는 아직 무작위)")])
                   for _im, _a in zip(probe_images, base_image_answers)], widths="equal", align="start"),
        mo.md(r"""
    그림 경로는 VLM에서 물려받았으므로 "In this image, ..." 형식으로 답합니다. 내용은 MiniMind-V 수준이라 자주 틀립니다
    (64M 모델의 한계로, VLM 노트북 10-1절과 같습니다). 소리 프레임도 나오지만,
    **Talker가 무작위**이므로 아직 의미 없는 토큰입니다(5절에서 들어 봅니다). Omni 학습의 핵심은 이 Talker를 가르치는 것입니다.
    """),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. 소리 토큰: Mimi 코덱

    언어 모델은 **정해진 어휘 중 하나를 고르는** 방식으로만 출력합니다. 그래서 목소리를 직접 만들지 않고, 목소리를 **토큰으로 바꾸는 코덱**을 씁니다.
    MiniMind-O는 [Mimi](https://arxiv.org/abs/2410.00037)(Kyutai, Moshi의 음성 코덱)를 씁니다.

    - 24kHz 파형을 **80ms마다 한 프레임**으로 묶습니다 → **초당 12.5프레임**.
    - 프레임 하나를 **잔차 벡터 양자화(RVQ)** 로 코드 8개로 적습니다. 코드북 0이 대략적인 모양을, 코드북 1이 0이 놓친 **나머지(잔차)** 를,
      코드북 2가 또 그 나머지를 … 적습니다. 각 코드는 2,048개 중 하나입니다.
    - 그래서 1초의 목소리 = **12.5 × 8 = 100개의 소리 토큰**. 같은 1초의 글은 영어로 약 3~4토큰입니다.

    학습 데이터(`sft_a2a_mini.parquet`)에서 사람의 질문 녹음을 하나 꺼내 Mimi로 토큰화하고, **앞쪽 코드북 k개만으로** 다시 복원해 봅니다.
    사람 대신 **Whisper-tiny**(음성 인식 모델)가 듣고 받아 적게 해서 "알아들을 수 있는가"를 숫자로 확인합니다.
    """)
    return


@app.cell
def _(
    MimiModel,
    WhisperForConditionalGeneration,
    WhisperProcessor,
    device,
    hf_hub_download,
    librosa,
    re,
    snapshot_download,
    torch,
):
    MIMI_REPO, MIMI_REVISION = "jingyaogong/mimi", "b4e362bbfbba9444b162de486da40af6639e0b98"  # 원본 README가 안내하는 Mimi
    WHISPER_REPO, WHISPER_REVISION = "openai/whisper-tiny.en", "87c7102498dcde7456f24cfd30239ca606ed9063"
    DATA_REPO, DATA_REVISION = "jingyaogong/minimind-o_dataset", "d6588e12ac2ac8ced65eb58a7d7b3eef4aa220de"

    mimi = MimiModel.from_pretrained(snapshot_download(MIMI_REPO, revision=MIMI_REVISION)).to(device).eval()
    _wdir = snapshot_download(WHISPER_REPO, revision=WHISPER_REVISION, allow_patterns=["*.json", "*.safetensors", "*.txt"])
    whisper_processor = WhisperProcessor.from_pretrained(_wdir)
    whisper = WhisperForConditionalGeneration.from_pretrained(_wdir).to(device).eval()
    # 원본 학습 데이터의 mini 부분집합(영어). 하나의 parquet 파일이라 통째로 받습니다(881MB, 처음 한 번).
    a2a_path = hf_hub_download(DATA_REPO, "sft_a2a_mini.parquet", repo_type="dataset", revision=DATA_REVISION)

    @torch.inference_mode()
    def codes_to_wav(codes):
        """[8, T] 또는 [1, k, T] Mimi 코드 → 24kHz 파형. eval_omni.py처럼 특수 토큰(≥2049)은 0으로 바꿉니다."""
        codes = torch.as_tensor(codes, device=device)
        codes = codes[None] if codes.dim() == 2 else codes
        codes = torch.where(codes >= 2049, torch.zeros_like(codes), codes)
        return mimi.decode(codes).audio_values[0, 0].float().cpu().numpy()

    @torch.inference_mode()
    def transcribe(wav, sr=24000):
        """Whisper-tiny로 받아쓰기 (16kHz로 바꿔 넣음)"""
        if wav is None or len(wav) < sr // 10:
            return ""
        feats = whisper_processor(librosa.resample(wav, orig_sr=sr, target_sr=16000), sampling_rate=16000,
                                  return_tensors="pt").input_features.to(device)
        return whisper_processor.batch_decode(whisper.generate(feats, max_new_tokens=120), skip_special_tokens=True)[0].strip()

    def _words(s):
        return re.sub(r"[^a-z0-9' ]", " ", s.lower()).split()

    def wer(ref, hyp):
        """단어 오류율(WER) = 편집 거리 / 정답 단어 수. 1을 넘으면 1로 자릅니다(0 = 전사 일치, 1 = 원래 오류율이 100% 이상)."""
        r, h = _words(ref), _words(hyp)
        dp = list(range(len(h) + 1))
        for i in range(1, len(r) + 1):
            prev, dp[0] = dp[0], i
            for j in range(1, len(h) + 1):
                prev, dp[j] = dp[j], min(dp[j] + 1, dp[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
        return min(1.0, dp[-1] / max(1, len(r)))

    return a2a_path, codes_to_wav, mimi, transcribe, wer


@app.cell
def _(
    a2a_path,
    codes_to_wav,
    device,
    io,
    json,
    librosa,
    mimi,
    np,
    pq,
    sf,
    torch,
    transcribe,
    wer,
):
    _t = pq.read_table(a2a_path, columns=["conversations", "question_audios"]).slice(0, 40)
    _clips = []
    for _k in [0, 7, 13, 21, 30]:
        _wav, _sr = sf.read(io.BytesIO(_t["question_audios"][_k].as_py()[0]))
        _clips.append((json.loads(_t["conversations"][_k].as_py())[0]["content"], librosa.resample(_wav.astype(np.float32), orig_sr=_sr, target_sr=24000)))

    with torch.inference_mode():
        _enc = mimi.encode(torch.tensor(_clips[0][1])[None, None].to(device), num_quantizers=8)
    demo_codes = _enc.audio_codes[0].cpu()           # [8, T]
    demo_text, demo_wav = _clips[0]
    assert demo_codes.shape[0] == 8 and int(demo_codes.max()) < 2048
    assert abs(demo_codes.shape[1] - len(demo_wav) / 24000 * 12.5) <= 1  # 초당 12.5프레임

    rvq_wavs = {k: codes_to_wav(demo_codes[None, :k]) for k in [1, 2, 4, 8]}
    rvq_rows = []
    for _k in [1, 2, 4, 8]:
        _hyps = []
        for _text, _w in _clips:
            with torch.inference_mode():
                _c = mimi.encode(torch.tensor(_w)[None, None].to(device), num_quantizers=8).audio_codes
            _hyps.append((_text, transcribe(codes_to_wav(_c[:, :_k]))))
        rvq_rows.append({"코드북 수 k": _k, "토큰/초": int(12.5 * _k), "Whisper가 들은 것 (첫 클립)": _hyps[0][1],
                         "평균 WER (5개 클립)": f"{np.mean([wer(a, b) for a, b in _hyps]):.2f}"})
    orig_hyp = transcribe(demo_wav)
    return demo_codes, demo_text, demo_wav, orig_hyp, rvq_rows, rvq_wavs


@app.cell(hide_code=True)
def _(
    demo_codes,
    demo_text,
    demo_wav,
    mo,
    orig_hyp,
    plt,
    result_table,
    rvq_rows,
    rvq_wavs,
    wav_bytes,
):
    _fig, _ax = plt.subplots(figsize=(8, 2.2))
    _ax.imshow(demo_codes.numpy(), aspect="auto", cmap="viridis", interpolation="nearest")
    _ax.set_yticks(range(8)); _ax.set_yticklabels([f"codebook {i}" for i in range(8)], fontsize=7)
    _ax.set_xlabel("frame (80 ms each)"); _ax.set_title("Mimi codes of the question audio (color = code id 0..2047)", fontsize=9)
    _fig.tight_layout(); _plot = mo.as_html(_fig); plt.close(_fig)
    mo.vstack([
        mo.md(f"**원본 녹음** ({len(demo_wav) / 24000:.1f}초, 대본: *{demo_text}*) → Whisper: *{orig_hyp}*"),
        mo.audio(wav_bytes(demo_wav)),
        _plot,
        mo.md(f"이 녹음은 `{tuple(demo_codes.shape)}` = 코드북 8개 × {demo_codes.shape[1]}프레임, 즉 **{demo_codes.numel()}개의 소리 토큰**입니다."),
        mo.hstack([mo.vstack([mo.md(f"**코드북 {k}개로 복원**"), mo.audio(wav_bytes(w))]) for k, w in rvq_wavs.items()], widths="equal"),
        result_table(rvq_rows, selection=None),
        mo.md(r"""
    코드북 1개만 쓰면 억양과 리듬은 남지만 발음이 뭉개져 Whisper가 엉뚱하게 받아 적습니다. 코드북을 더할수록 잔차가 채워져
    WER이 내려가고, 8개일 때 가장 정확합니다(짧은 클립에서는 4개로도 충분한 경우가 있습니다).
    **Talker가 프레임마다 코드 8개를 모두 맞혀야 하는 이유**입니다.
    (Mimi 원본은 코드북 32개까지 있지만 MiniMind-O는 앞쪽 8개만 씁니다.)
    """),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. 학습 데이터: 9줄짜리 입력과 계단(지연) 배치

    원본 데이터 한 행에는 `conversations`(대화 JSON)와 `answer_audios`(답변을 읽은 목소리의 Mimi 코드, 프레임마다 8개씩 이어 붙인 정수 목록)가 있습니다.
    이 노트북은 학습을 1분 안팎으로 끝내기 위해 다음 행만 고릅니다.

    - 대화 한 턴, 영어, 답변 목소리 **1.6~4.8초**(20~60프레임), 화자 임베딩(`spk_emb`)이 없는 행(단일 화자인지는 이 조건만으로 보장되지 않음)
    - 글자 수와 길이가 맞지 않는(잘린) 답변은 제외: 초당 10~22글자
    - 학습용 N개(아래 위젯) + **평가 전용 16개**(학습에 넣지 않음)

    `OmniDataset.__getitem__`은 이것을 **9 × T** 입력으로 만듭니다(줄 0~7: 소리 코드북 8개, 줄 8: 글).

    ```python
    # OmniDataset.__getitem__ 요약 — 답변 코드북 i를 "답변 시작 + i + 1" 칸부터 채움
    for layer_idx in range(8):
        start_pos = assistant_start + layer_idx + 1
        for i, code in enumerate(codes):               # 각 코드북 끝에는 <|audio_stop|>(2050)
            Y_audio_layers[layer_idx][start_pos + i] = code
            audio_labels[layer_idx][start_pos + i] = code
    X_audio, X_text = [l[:-1] for l in Y_audio_layers], input_ids[:-1]
    input_ids = torch.cat((X_audio, X_text.unsqueeze(0)), dim=0)   # (9, T-1)
    text_labels, audio_labels = text_labels[1:], [l[1:] for l in audio_labels]  # 한 칸 밀어 "다음 것"이 정답
    ```

    코드북 i를 i칸 늦게 시작하는 **지연(delay) 배치** 덕분에, 한 위치에서 8개 코드를 예측할 때 앞쪽 코드북의 같은 프레임 값은
    이미 입력으로 들어와 있습니다. 코드북 1은 코드북 0을 보고, 코드북 7은 0~6을 본 뒤에 예측합니다. RVQ의 "잔차" 순서와 맞습니다.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    n_train_ui = mo.ui.dropdown(options={"32": 32, "64": 64, "128": 128}, value="64", label="학습 문장 수 N")
    steps_ui = mo.ui.dropdown(options={"400": 400, "800": 800, "1200": 1200}, value="800", label="학습 step (batch 16)")
    mo.vstack([mo.hstack([n_train_ui, steps_ui], justify="start", gap=2),
               mo.md("기본값(64문장, 800 step)에서 실험 하나가 약 50~75초 걸립니다. 바꾸면 3절부터 다시 실행됩니다.")])
    return n_train_ui, steps_ui


@app.cell
def _(
    OmniDataset,
    a2a_path,
    json,
    n_train_ui,
    pa,
    pq,
    random,
    tempfile,
    tokenizer,
):
    from pathlib import Path as _Path

    _t = pq.read_table(a2a_path, columns=["conversations", "answer_audios", "spk_emb"])
    _rows, _seen = [], set()
    for _c, _a, _s in zip(_t["conversations"].to_pylist(), _t["answer_audios"].to_pylist(), _t["spk_emb"].to_pylist()):
        if _s or not _a:
            continue
        _conv = json.loads(_c)
        if len(_conv) != 2:
            continue
        _q, _ans, _n = _conv[0]["content"], _conv[1]["content"], len(_a[0]) // 8
        if not (20 <= _n <= 60) or not (_q.isascii() and _ans.isascii()) or len(_q) > 120 or _ans in _seen:
            continue
        if not 10 <= len(_ans) / (_n / 12.5) <= 22:
            continue
        _seen.add(_ans)
        _rows.append({"conversations": _c, "answer_audios": _a})
    pool_size = len(_rows)
    random.Random(0).shuffle(_rows)
    held_rows, train_rows = _rows[:16], _rows[16:16 + n_train_ui.value]

    omni_data_dir = _Path(tempfile.mkdtemp(prefix="minimind_omni_data_"))
    pq.write_table(pa.Table.from_pylist(train_rows), omni_data_dir / "train_t2a.parquet")
    OMNI_MAX_LEN = 160  # 원본 T2A mini는 512. 프롬프트 + 최대 60프레임 + 지연 8칸이 들어가는 길이

    # 소리 인코더·이미지 인코더 없이(processor=None) 만들면 글→소리(T2A) 데이터가 됩니다.
    train_ds = OmniDataset(str(omni_data_dir / "train_t2a.parquet"), tokenizer, max_length=OMNI_MAX_LEN, image_token_len=64)
    # 그림 설명용: 학습 때 5% 확률로 입력을 무작위 값으로 바꾸는 scheduled sampling을 끈 사본
    view_ds = OmniDataset(str(omni_data_dir / "train_t2a.parquet"), tokenizer, max_length=OMNI_MAX_LEN, image_token_len=64,
                          scheduled_sampling=0)

    random.seed(1)
    view_x, view_text_labels, view_audio_labels, *_ = view_ds[0]
    assert view_x.shape == (9, OMNI_MAX_LEN - 1) and view_audio_labels.shape == (8, OMNI_MAX_LEN - 1)
    _n_codes = len(train_rows[0]["answer_audios"][0]) // 8
    assert [(view_audio_labels[i] != -100).sum().item() for i in range(8)] == [_n_codes + 1] * 8  # 코드 + stop 1개
    return (
        OMNI_MAX_LEN,
        held_rows,
        pool_size,
        train_ds,
        train_rows,
        view_audio_labels,
        view_text_labels,
        view_x,
    )


@app.cell(hide_code=True)
def _(
    OMNI_MAX_LEN,
    json,
    mo,
    plt,
    pool_size,
    result_table,
    tokenizer,
    train_ds,
    train_rows,
    view_audio_labels,
    view_text_labels,
    view_x,
):
    _x = view_x
    _first_audio = int((_x[0] < 2048).nonzero()[0])         # 코드북 0의 첫 코드 위치 = 답변 시작 + 1
    _ans_start = _first_audio - 1
    _rows = []
    for _p in range(_ans_start - 2, _ans_start + 13):
        _tok = tokenizer.convert_ids_to_tokens(int(_x[8, _p])).replace("Ċ", "\\n").replace("Ġ", "␣")
        _r = {"위치": _p - _ans_start, "글 (줄 8)": _tok}
        for _i in range(8):
            _v = int(_x[_i, _p])
            _r[f"코드북 {_i}"] = "·" if _v == 2049 else ("STOP" if _v == 2050 else _v)
        _rows.append(_r)

    _cat = _x[:8].clone().float()
    _cat[_x[:8] < 2048], _cat[_x[:8] == 2049], _cat[_x[:8] == 2050] = 2, 0, 3
    _end = int((view_audio_labels[7] != -100).nonzero()[-1]) + 3
    _fig, _ax = plt.subplots(figsize=(8.5, 2.0))
    _ax.imshow(_cat[:, _ans_start - 4:_end].numpy(), aspect="auto", cmap="viridis", vmin=0, vmax=3, interpolation="nearest")
    _ax.set_yticks(range(8)); _ax.set_yticklabels([f"cb {i}" for i in range(8)], fontsize=7)
    _ax.set_xlabel(f"position (0 = answer start, from {-4})")
    _ax.set_title("purple = pad, green = Mimi code, yellow = STOP", fontsize=9)
    _fig.tight_layout(); _plot = mo.as_html(_fig); plt.close(_fig)

    _conv = json.loads(train_rows[0]["conversations"])
    _n_text = int((view_text_labels != -100).sum())
    _n_audio = int((view_audio_labels != -100).sum())
    mo.vstack([
        mo.md(f"""
    조건에 맞는 행 **{pool_size:,}개** 중 학습 **{len(train_ds)}개**, 평가 16개를 썼습니다. 첫 학습 행:
    *Q: {_conv[0]['content']}* → *A: {_conv[1]['content']}* (목소리 {len(train_rows[0]['answer_audios'][0]) // 8}프레임)

    아래 표는 `OmniDataset`이 만든 입력(`{tuple(view_x.shape)}`, 최대 길이 {OMNI_MAX_LEN} − 1)에서 답변이 시작되는 부분입니다.
    `·`는 `<|audio_pad|>`(2049)입니다. 코드북 i의 첫 코드가 **i칸씩 늦게** 시작하는 계단을 확인하세요.
    """),
        result_table(_rows, selection=None, page_size=15),
        _plot,
        mo.md(f"""
    - 글 답변은 짧고(정답 **{_n_text}토큰**), 소리는 길어서(정답 **{_n_audio}개** = 8줄 × (코드 + STOP)) 글이 먼저 끝납니다.
      글이 끝난 뒤의 글 줄은 패딩이고, Talker는 그동안 **앞에서 이미 본 글**을 어텐션으로 돌아보며 계속 말합니다.
    - 질문(프롬프트) 위치의 소리 줄은 모두 패딩이고 정답에서도 빠집니다. Talker도 "assistant 답변만 정답" 규칙을 따릅니다.
    - 그림 왼쪽(답변 시작 전)과 오른쪽 끝(STOP 뒤) 모두 패딩입니다. 코드북마다 STOP이 한 칸씩 늦게 나옵니다.
    """),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Talker 안쪽: Thinker의 속마음 + 지난 소리 → 다음 소리

    `MiniMindOmni.forward`의 Talker 부분입니다.

    ```python
    # ======= Thinker =======
    for i, layer in enumerate(self.thinker.layers):
        hidden_states, present = layer(hidden_states, ...)
        if i == self.config.bridge_layer: bridge_states = hidden_states    # 4번째 층 출력을 따로 보관
    h_thinker = self.thinker.norm(hidden_states)                            # → text_logits (글)

    # ======= Talker =======
    talker_emb = self.talker.embed_tokens(audio_ids)                        # 소리 8줄 → 벡터 1개 (TalkerEmbedding)
    hidden_states = self.talker.embed_proj(bridge_states) * self.talker.text_scale \
                  + self.talker.codec_proj(talker_emb) * self.talker.audio_scale   # "무엇을" + "어디까지 말했나"
    for layer in self.talker.layers: hidden_states, present = layer(hidden_states, ...)
    audio_logits = self.talker.lm_head(self.talker.norm(hidden_states))    # 코드북 8개의 로짓 목록 (TalkerHead)
    ```

    - **Bridge:** Talker는 Thinker의 최종 출력(다음 글자 예측에 특화)이 아니라 **중간층** 표현을 받습니다. 문맥은 충분히 섞였지만
      아직 "다음 단어 고르기"에만 맞춰지지 않은 표현입니다. 두 입력의 비중 `text_scale`(처음 3.0)과 `audio_scale`(처음 1.0)도 학습됩니다.
    - **TalkerEmbedding / TalkerHead (MTP):** 코드북 8개마다 큰 임베딩·출력층을 따로 두는 대신, **공통 층(`base`) 하나 + 코드북별 작은 어댑터(rank 256)** 를 씁니다.
      한 위치에서 코드 8개를 동시에 예측하므로 **여러 토큰 예측(Multi-Token Prediction, MTP)** 이라고 부릅니다.
    """)
    return


@app.cell
def _(
    DataLoader,
    SkipBatchSampler,
    base_omni,
    count,
    device,
    omni_collate_fn,
    torch,
    train_ds,
):
    talker_batch = next(iter(DataLoader(train_ds, batch_sampler=SkipBatchSampler(list(range(4)), 4, 0), collate_fn=omni_collate_fn)))
    _x = talker_batch[0].to(device)
    _captured = {}
    _hooks = [base_omni.thinker.layers[base_omni.config.bridge_layer].register_forward_hook(lambda m, i, o: _captured.update(bridge=o[0])),
              base_omni.talker.embed_tokens.register_forward_hook(lambda m, i, o: _captured.update(talker_emb=o))]
    with torch.no_grad():
        _out = base_omni(_x)
    for _h in _hooks:
        _h.remove()

    # TalkerEmbedding을 식대로 다시 계산: 코드북마다 (공통 임베딩 + 어댑터)를 구해 평균
    _te = base_omni.talker.embed_tokens
    with torch.no_grad():
        _audio_ids = _x[:, :8, :]
        _base = _te.base(_audio_ids)                                            # [B, 8, T, 768]
        _mine = sum(_base[:, i] + _te.adapters[i](_audio_ids[:, i]) for i in range(8)) / 8
    assert torch.allclose(_mine, _captured["talker_emb"], atol=1e-5)

    _head = base_omni.talker.lm_head
    talker_shapes = {
        "입력 input_ids": tuple(_x.shape),
        "Thinker 글 로짓 (out.logits)": tuple(_out.logits.shape),
        f"bridge_states (Thinker 층 {base_omni.config.bridge_layer} 출력)": tuple(_captured["bridge"].shape),
        "talker_emb (소리 8줄을 합친 벡터)": tuple(_captured["talker_emb"].shape),
        "소리 로짓 (out.audio_logits)": f"{len(_out.audio_logits)}개 × {tuple(_out.audio_logits[0].shape)}",
    }
    head_params = {"base": count(_head.base), "adapters": count(_head.adapters), "naive": 8 * count(_head.base)}
    return head_params, talker_batch, talker_shapes


@app.cell(hide_code=True)
def _(head_params, mo, result_table, talker_shapes):
    mo.vstack([
        result_table([{"텐서": k, "모양": str(v)} for k, v in talker_shapes.items()], selection=None),
        mo.md(f"""
    - `TalkerEmbedding`을 "코드북마다 (공통 임베딩 + 어댑터)를 더해 8로 나눔"으로 다시 계산한 값이 원본 출력과 같았습니다.
    - `TalkerHead`: 공통 출력층 {head_params['base'] / 1e6:.2f}M + 어댑터 8개 {head_params['adapters'] / 1e6:.2f}M
      = **{(head_params['base'] + head_params['adapters']) / 1e6:.2f}M**. 코드북마다 출력층을 따로 두면 {head_params['naive'] / 1e6:.2f}M입니다.
    - Thinker와 Talker는 **같은 위치 축**을 공유합니다. 위치 t에서 Thinker는 다음 글 토큰을, Talker는 다음 소리 코드 8개를 동시에 예측합니다.
    """),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. 학습 전: 무작위 Talker의 목소리

    학습하지 않은 Talker에 평가용 질문을 하고, 나온 소리 프레임을 Mimi로 복원해 들어 봅니다.
    """)
    return


@app.cell
def _(base_omni, codes_to_wav, held_rows, json, omni_chat, torch, transcribe):
    _q = json.loads(held_rows[0]["conversations"])[0]["content"]
    before_reply = omni_chat(base_omni, _q, max_new_tokens=60)
    before_wav = codes_to_wav(torch.tensor(before_reply["frames"]).T) if before_reply["frames"] else None
    before_hyp = transcribe(before_wav)
    before_question = _q
    return before_hyp, before_question, before_reply, before_wav


@app.cell(hide_code=True)
def _(before_hyp, before_question, before_reply, before_wav, mo, wav_bytes):
    mo.vstack([
        mo.md(f"*Q: {before_question}*<br>Thinker의 글: **{before_reply['text']}**"),
        mo.audio(wav_bytes(before_wav)) if before_wav is not None else mo.md("(소리 프레임 없음)"),
        mo.md(f"""
    Talker의 목소리: 프레임 {len(before_reply['frames'])}개 = {len(before_reply['frames']) / 12.5:.1f}초, Whisper: *{before_hyp or '(인식된 말 없음)'}*

    무작위 코드를 Mimi로 복원하면 말이 아닌 소리가 나옵니다. 소리 어휘 2,112개 중 64개가 특수 토큰(≥ 2048)이라,
    아무 코드북에서나 우연히 특수 토큰이 나오면 프레임이 거기서 끊깁니다. Whisper는 알아듣기 어려운 소리에도 그럴듯한 말을
    지어내는 경우가 있으므로(환각), 받아 적은 글은 **정답 문장과의 WER**로만 판단합니다.
    """),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. 학습: 글 손실 + 소리 손실

    `train_sft_omni.py`의 `train_epoch`는 글과 소리 손실을 더합니다. **STOP 토큰은 10배 가중치**를 받습니다.
    STOP을 놓치면 목소리가 끝나지 않기 때문입니다.

    ```python
    text_loss = CE(res.logits, labels)  (정답 위치 평균)
    for i, al in enumerate(res.audio_logits):                  # 코드북 8개
        layer_loss = CE(al, audio_labels[:, i])
        weighted = layer_loss * valid_mask * (1 + stop_mask * 9)   # STOP(2050)이면 × 10
        audio_loss += weighted.sum() / valid_mask.sum()
    loss = text_loss + audio_loss / 8 + res.aux_loss
    ```

    `run_training`은 이 손실과 `get_lr`(cosine), `clip_grad_norm_`을 그대로 옮기고 DDP·저장·재개만 뺐습니다. 세 가지 실험을 합니다.

    | 실험 | 학습하는 부분 | 원본 대응 | lr | 확인할 것 |
    | --- | --- | --- | --- | --- |
    | **A** | Talker만 (Thinker 고정) | `--freeze_backbone all` | 5e-4 | Talker만으로 말하게 되는가, 그림·글 능력은 그대로인가 |
    | **C** (대조군) | Talker만, **Thinker 연결을 끊음** (`text_scale = 0`) | — | 5e-4 | Thinker 없이 Talker 혼자 소리만 보고 말할 수 있나 |
    | **B** | Thinker + Talker 전부 | `--mode all` (원본 T2A 기본) | 2e-4 | 질문 → 글 + 목소리 전체 파이프라인, 대가는 무엇인가 |

    원본 T2A mini는 약 47만 개 답변(470시간)을 1 epoch 학습합니다(RTX 3090 약 60분). 여기서는 **N개 문장을 여러 번 반복**하므로,
    학습한 문장을 얼마나 정확히 말하는지와 **처음 보는 문장**에서는 어떻게 되는지를 함께 봅니다.
    """)
    return


@app.cell
def _(
    DataLoader,
    SkipBatchSampler,
    device,
    get_lr,
    nn,
    omni_collate_fn,
    setup_seed,
    torch,
):
    def apply_mode(model, mode):
        """A(talker) / C(blind) / B(all)에 맞춰 requires_grad 설정. 이미지 인코더는 항상 고정"""
        for p in model.parameters():
            p.requires_grad = False
        if mode in ("talker", "blind"):                 # init_omni_model(freeze_backbone='all')처럼 Thinker 고정
            for p in model.talker.parameters():
                p.requires_grad = True
            if mode == "blind":                         # Thinker → Talker 다리를 끊음
                with torch.no_grad():
                    model.talker.text_scale.fill_(0.0)
                model.talker.text_scale.requires_grad = False
        else:                                           # train_sft_omni.py --mode all
            for name, p in model.named_parameters():
                if not name.startswith("vision_encoder."):
                    p.requires_grad = True
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def run_training(model, train_ds, mode, steps, learning_rate, batch_size=16, history=None):
        """train_sft_omni.py의 train_epoch 손실·lr·clip을 단일 GPU로 (bf16 autocast, step 수로 끝냄)"""
        n_trainable = apply_mode(model, mode)
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=learning_rate)
        loss_fct = nn.CrossEntropyLoss(reduction="none")
        model.train()
        step, epoch = 0, 0
        while step < steps:
            setup_seed(42 + epoch); indices = torch.randperm(len(train_ds)).tolist()
            epoch += 1
            for input_ids, labels, audio_labels, audio_inputs, audio_lens, pixel_values, spk_emb in DataLoader(
                    train_ds, batch_sampler=SkipBatchSampler(indices, batch_size, 0), collate_fn=omni_collate_fn):
                if step >= steps:
                    break
                step += 1
                input_ids, labels, audio_labels = input_ids.to(device), labels.to(device), audio_labels.to(device)
                lr = get_lr(step, steps, learning_rate)
                for g in optimizer.param_groups:
                    g["lr"] = lr
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    res = model(input_ids, audio_inputs=audio_inputs, pixel_values=pixel_values)
                    text_raw = loss_fct(res.logits.view(-1, res.logits.size(-1)), labels.view(-1))
                    text_mask = (labels.view(-1) != -100).float()
                    text_loss = (text_raw * text_mask).sum() / (text_mask.sum() + 1e-9)
                    audio_loss = res.audio_logits[0].sum() * 0
                    for i, al in enumerate(res.audio_logits):
                        target = audio_labels[:, i, :].reshape(-1)
                        layer_loss = loss_fct(al.reshape(-1, al.size(-1)), target)
                        valid, stop = (target != -100).float(), (target == 2050).float()
                        audio_loss = audio_loss + (layer_loss * valid * (1 + stop * 9)).sum() / (valid.sum() + 1e-9)
                    audio_loss = audio_loss / 8
                    loss = text_loss + audio_loss + res.aux_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                if history is not None:
                    history.append((text_loss.item(), audio_loss.item()))
        return model.eval(), n_trainable, epoch

    return (run_training,)


@app.cell
def _(
    base_omni,
    copy,
    run_training,
    setup_seed,
    steps_ui,
    time,
    torch,
    train_ds,
):
    torch.cuda.reset_peak_memory_stats()
    histories, train_info, trained = {}, {}, {}
    for _name, _mode, _lr in [("A", "talker", 5e-4), ("C", "blind", 5e-4), ("B", "all", 2e-4)]:
        setup_seed(0)
        histories[_name] = []
        _t0 = time.time()
        _m, _n, _ep = run_training(copy.deepcopy(base_omni), train_ds, _mode, steps_ui.value, _lr, history=histories[_name])
        torch.cuda.synchronize()
        train_info[_name] = {"시간": time.time() - _t0, "학습 매개변수": _n, "epochs": _ep, "lr": _lr}
        trained[_name] = _m
    model_a, model_c, model_b = trained["A"], trained["C"], trained["B"]
    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    return histories, model_a, model_b, model_c, peak_gb, train_info


@app.cell(hide_code=True)
def _(histories, mo, np, peak_gb, plt, result_table, train_info):
    def _smooth(v, k=20):
        return np.convolve(v, np.ones(k) / k, mode="valid")

    _fig, _axes = plt.subplots(1, 2, figsize=(9, 2.8))
    for _name, _h in histories.items():
        _axes[0].plot(_smooth([a for _, a in _h]), lw=0.9, label=_name)
        _axes[1].plot(_smooth([t for t, _ in _h]), lw=0.9, label=_name)
    _axes[0].set_title("audio loss (8 codebooks)", fontsize=9); _axes[1].set_title("text loss", fontsize=9)
    for _ax in _axes:
        _ax.set_yscale("log"); _ax.set_xlabel("step"); _ax.legend(fontsize=8)
    _fig.tight_layout(); _plot = mo.as_html(_fig); plt.close(_fig)
    _rows = [{"실험": k, "학습 매개변수": f"{v['학습 매개변수'] / 1e6:.1f}M", "lr": v["lr"], "epochs": v["epochs"],
              "학습 시간 (GPU)": f"{v['시간']:.0f}초", "마지막 20 step 소리 손실": f"{np.mean([a for _, a in histories[k][-20:]]):.3f}",
              "마지막 20 step 글 손실": f"{np.mean([t for t, _ in histories[k][-20:]]):.3f}"} for k, v in train_info.items()]
    mo.vstack([_plot, result_table(_rows, selection=None), mo.md(f"GPU 최대 할당: **{peak_gb:.1f}GB**")])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    - **A**와 **C**는 Thinker를 고정했으므로 글 손실이 처음 값 그대로입니다. 소리 손실은 둘 다 많이 내려가고 A가 더 낮습니다.
    - **C**의 손실이 낮은 것은 함정입니다. 학습 중에는 **정답 소리 코드가 입력으로 들어오므로**(teacher forcing), 앞 몇 프레임만 보면
      64개 중 어느 문장인지 알아채고 나머지를 이어 말할 수 있습니다. 하지만 생성할 때는 첫 프레임부터 스스로 골라야 합니다.
      Thinker를 볼 수 없는 C가 그때 무엇을 하는지 7-1절에서 확인합니다.
    - **B**는 글 손실까지 0 근처로 내려갑니다. Thinker가 N개 질문의 답을 **통째로 외운** 것입니다(7-2절에서 대가를 봅니다).

    ## 7. 들어 보기와 받아쓰기 평가

    ### 7-1. 정해진 문장 읽기: Talker의 실력만 따로 재기

    `generate`는 Thinker가 답을 직접 고르므로, Thinker가 다른 답을 하면 Talker의 실력을 잴 수 없습니다.
    그래서 `read_aloud`는 **글 줄을 정답 문장으로 고정**하고 소리만 생성합니다. 원본 `stream_generate`와 같은 순서·지연·샘플링
    (온도 0.2, top-k 50)을 쓰되, 글 토큰을 고르는 부분만 "정답 토큰 넣기"로 바꿨습니다. KV 캐시도 원본처럼 씁니다.

    ```python
    # stream_generate의 핵심: step번째 걸음에서 코드북 i는 audio_step = step − 1 ≥ i일 때부터 코드를 뽑음
    audio_step = step - 1
    for i, al in enumerate(out.audio_logits):
        if audio_step < i: audio_codes[i].append(self.audio_pad_token)       # 아직 차례가 아님 (계단)
        else: ... code = top_idx[torch.multinomial(F.softmax(top_val / 0.2), 1)]
    ```

    생성한 소리를 Whisper가 받아 적고 **정답 문장과의 WER**을 잽니다. 학습 문장 16개와 **평가 전용 16개**(처음 보는 문장)로 나눠 봅니다.
    """)
    return


@app.cell
def _(chat_prompt_ids, codes_to_wav, device, tokenizer, torch):
    @torch.inference_mode()
    def read_aloud(model, question, answer, seed=0):
        """글 줄을 '질문 프롬프트 + 정답 답변'으로 고정하고 소리 코드 8줄만 생성 → Mimi 파형"""
        torch.manual_seed(seed)
        p_ids = chat_prompt_ids(question)
        a_ids = tokenizer(answer + tokenizer.eos_token, add_special_tokens=False).input_ids
        S = len(p_ids)
        total = S + int(len(answer) / 10 * 12.5) + 20          # 초당 10글자보다 느리지 않다고 보고 여유 있게
        # 답변 뒤 글 줄: 원본처럼 줄바꿈(201) 1개 후 패딩(0)
        text_row = torch.tensor((p_ids + a_ids + [201] + [0] * total)[:total], device=device)
        audio = torch.full((8, total), model.audio_pad_token, dtype=torch.long, device=device)
        stopped, past = [False] * 8, None
        for pos in range(S - 1, total - 1):                     # pos = 지금 입력하는 마지막 위치
            cols = slice(0, pos + 1) if past is None else slice(pos, pos + 1)
            out = model(torch.cat([audio[:, cols], text_row[None, cols]], 0)[None], past_key_values=past, use_cache=True)
            past = out.past_key_values
            for i in range(8):
                if pos - S >= i:                                # audio_step = pos − S, 코드북 i는 i걸음 늦게 시작
                    top_val, top_idx = (out.audio_logits[i][0, -1].float() / 0.2).topk(50)
                    code = top_idx[torch.multinomial(torch.softmax(top_val, -1), 1)].item()
                    audio[i, pos + 1] = code
                    stopped[i] = stopped[i] or code >= 2048
            if all(stopped):
                break
        frames = []                                             # 계단을 되돌려 프레임 k = (코드북 i의 S+1+k+i 칸)
        for k in range(total):
            if S + 8 + k >= total:
                break
            f = [audio[i, S + 1 + k + i].item() for i in range(8)]
            if any(c >= 2048 for c in f):
                break
            frames.append(f)
        return codes_to_wav(torch.tensor(frames).T) if frames else None

    return (read_aloud,)


@app.cell
def _(
    held_rows,
    json,
    model_a,
    model_b,
    model_c,
    np,
    read_aloud,
    train_rows,
    transcribe,
    wer,
):
    def score_reading(model, rows):
        out = []
        for r in rows:
            q, a = (t["content"] for t in json.loads(r["conversations"]))
            w = read_aloud(model, q, a)
            hyp = transcribe(w)
            out.append({"answer": a, "hyp": hyp, "wer": wer(a, hyp), "wav": w})
        return out

    reading = {name: {"train": score_reading(m, train_rows[:16]), "held": score_reading(m, held_rows)}
               for name, m in [("A", model_a), ("C", model_c), ("B", model_b)]}
    reading_summary = {name: {k: float(np.mean([x["wer"] for x in v])) for k, v in d.items()} for name, d in reading.items()}
    return reading, reading_summary


@app.cell(hide_code=True)
def _(mo, reading, reading_summary, result_table, wav_bytes):
    _rows = [{"모델": k, "학습 문장 16개 WER": f"{v['train']:.2f}", "처음 보는 문장 16개 WER": f"{v['held']:.2f}"} for k, v in reading_summary.items()]

    def _card(item, title):
        return mo.vstack([mo.md(f"**{title}**<br>정답: *{item['answer']}*"),
                          mo.audio(wav_bytes(item["wav"])) if item["wav"] is not None else mo.md("(소리 없음)"),
                          mo.md(f"Whisper: *{item['hyp'] or '(없음)'}*<br>WER {item['wer']:.2f}")])

    mo.vstack([
        result_table(_rows, selection=None),
        mo.md("WER은 단어 편집 거리/정답 단어 수입니다. 0은 전사 일치이며, 일반적으로 삽입 때문에 1을 넘을 수 있습니다. 이 실습의 WER 함수는 1에서 자릅니다. 같은 문장을 세 모델이 읽은 소리:"),
        mo.hstack([_card(reading["A"]["train"][0], "A · 학습 문장"), _card(reading["C"]["train"][0], "C · 같은 학습 문장"),
                   _card(reading["A"]["held"][1], "A · 처음 보는 문장")], widths="equal", align="start"),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **읽는 법** (기본 설정 기준. 설정을 바꾸면 숫자도 바뀝니다.)

    - **A (Talker만):** 1분 정도 학습한 Talker가 학습 문장을 Whisper가 알아듣게 읽습니다. Thinker의 중간층 표현만 보고
      "지금 어느 단어를 어떤 소리로 말할지" 정한 것입니다.
    - **C (다리를 끊은 대조군):** 소리는 또렷한데 **엉뚱한 문장**입니다. 무엇을 읽으라고 하든 외운 학습 문장 몇 개
      (예: "The Statue of Liberty is a famous American landmark.") 중 하나를 말합니다. C도 64개 문장의 소리를 외웠지만,
      **어느 문장을 말할지는 Thinker에게서만 알 수 있습니다.** A와 C의 차이가 곧 "Talker가 Thinker를 보고 말한다"는 증거입니다
      (VLM 노트북의 빈 그림 대조군과 같은 역할).
    - **처음 보는 문장**은 A·B도 거의 못 읽습니다. 목소리 결은 같지만 단어가 엉뚱하거나 금방 끊깁니다. 이 설정에서는 학습 밖 문장으로의 일반화를 확인하지 못했습니다. 학습 문장과 확인용 문장의 격차를 관찰한 결과이며, 필요한 데이터 규모를 확정하는 근거는 아닙니다.
      (같은 Talker를 5,000문장으로 10분 학습해도 처음 보는 문장의 WER은 1에 가까웠습니다. 일반화에는 훨씬 많은 데이터와 시간이 필요합니다.)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 7-2. 전체 파이프라인: 질문 → 글 + 목소리 (실험 B)

    이번에는 글을 고정하지 않고 원본 `generate`에 질문만 줍니다. Thinker가 글을 쓰는 동시에 Talker가 그 글을 읽습니다.
    학습한 질문, 처음 보는 일반 질문, 그림 질문을 모두 해 봅니다. A는 Thinker가 그대로이므로 글 답이 베이스와 같아야 합니다.
    """)
    return


@app.cell
def _(
    base_omni,
    codes_to_wav,
    json,
    model_a,
    model_b,
    omni_chat,
    probe_images,
    torch,
    train_rows,
    transcribe,
):
    GENERAL_QUESTIONS = ["What is the capital of France?", "What do bees make?", "How many legs does a spider have?"]

    def _reply(model, q, image=None, n=80):
        r = omni_chat(model, q, image=image, max_new_tokens=n)
        r["wav"] = codes_to_wav(torch.tensor(r["frames"]).T) if r["frames"] else None
        r["hyp"] = transcribe(r["wav"])
        return r

    pipeline_rows = []
    for _r in train_rows[:3]:
        _q, _gold = (t["content"] for t in json.loads(_r["conversations"]))
        pipeline_rows.append({"kind": "학습한 질문", "q": _q, "gold": _gold, "B": _reply(model_b, _q)})
    for _q in GENERAL_QUESTIONS:
        pipeline_rows.append({"kind": "일반 질문", "q": _q, "gold": None, "B": _reply(model_b, _q),
                              "base": omni_chat(base_omni, _q, max_new_tokens=60)["text"], "A": omni_chat(model_a, _q, max_new_tokens=60)["text"]})
    pipeline_image = {"base": omni_chat(base_omni, "What is in this image?", image=probe_images[0], max_new_tokens=40)["text"],
                      "B": omni_chat(model_b, "What is in this image?", image=probe_images[0], max_new_tokens=40)["text"]}
    stream_steps = pipeline_rows[0]["B"]["steps"]
    return pipeline_image, pipeline_rows, stream_steps


@app.cell(hide_code=True)
def _(mo, pipeline_image, pipeline_rows, result_table, wav_bytes):
    _cards = []
    for _r in pipeline_rows[:3]:
        _b = _r["B"]
        _cards.append(mo.vstack([mo.md(f"*Q: {_r['q']}*<br>Thinker 글: **{_b['text']}**"),
                                 mo.audio(wav_bytes(_b["wav"])) if _b["wav"] is not None else mo.md("(소리 없음)"),
                                 mo.md(f"Whisper: *{_b['hyp']}*")]))
    _gen = [{"일반 질문": r["q"], "베이스": r["base"], "A (Thinker 고정)": r["A"], "B (전부 학습)": r["B"]["text"],
             "B 목소리를 Whisper가 들은 것": r["B"]["hyp"]} for r in pipeline_rows[3:]]
    mo.vstack([
        mo.md("**학습한 질문** (실험 B): 글과 목소리가 함께 나옵니다."),
        mo.hstack(_cards, widths="equal", align="start"),
        mo.md("**처음 보는 일반 질문**"),
        result_table(_gen, selection=None),
        mo.md(f"**그림 질문** (노란 자동차 사진, *What is in this image?*)<br>베이스: {pipeline_image['base']}<br>B: {pipeline_image['B']}"),
        mo.md(r"""
    - 학습한 질문에서는 Thinker의 글과 Talker의 목소리가 **같은 내용**입니다. 이것이 Omni의 "생각하면서 말하기"입니다.
    - A의 글 답은 베이스와 **완전히 같습니다**(Thinker를 건드리지 않음). 반면 B는 일반 질문·그림 질문에도 외운 답의 조각을 섞습니다
      (예: 프랑스 수도 질문에 학습 문장의 "Canadian dollar"). **파국적 망각**입니다. 64개 문장에 Thinker 전체를 맞춘 대가입니다.
      학습 데이터의 20%에 남아 있던 빈 `<think>` 블록을 스스로 써 넣기도 합니다.
    - B가 일반 질문에 쓴 글은 외운 문장의 조각이 섞여 있어서, 목소리도 그 조각만 알아들을 수 있는 경우가 많습니다.
    - 그래서 원본은 T2A·A2A를 대량의 데이터로 학습하고, 마지막에 그림 데이터로 `vision_proj`를 **다시 정렬**하는 단계(I2T)를 둡니다.
      Thinker가 바뀌면 이미지 토큰을 읽는 방식도 다시 맞춰야 하기 때문입니다.
    """),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 7-3. 스트리밍: 첫 목소리는 언제 나올까

    `stream_generate`는 한 걸음마다 글 토큰을 하나 내고, 코드북 8개가 계단을 다 채운 프레임부터 하나씩 내보냅니다.
    위 첫 번째 질문(실험 B)의 걸음별 기록입니다.
    """)
    return


@app.cell(hide_code=True)
def _(mo, result_table, stream_steps, tokenizer):
    _rows, _n_frames = [], 0
    for _k, (_tok, _has) in enumerate(stream_steps[:16]):
        _n_frames += _has
        _rows.append({"걸음": _k, "글 토큰": "(끝남)" if _tok is None else tokenizer.decode([_tok]).replace("\n", "\\n"),
                      "이번 걸음에 나온 소리 프레임": "🔊" if _has else "", "누적 프레임 (= 초)": f"{_n_frames} ({_n_frames * 0.08:.2f}초)"})
    _first = next((k for k, (_, h) in enumerate(stream_steps) if h), None)
    _text_end = next((k for k, (t, _) in enumerate(stream_steps) if t is None), len(stream_steps))
    mo.vstack([
        result_table(_rows, selection=None, page_size=16),
        mo.md(f"""
    - 첫 소리 프레임은 **{_first}번 걸음**(0부터 셈)에 나옵니다. 1걸음 지연 + 계단 7칸 뒤에야 코드북 8개가 한 프레임을 다 채우기 때문입니다.
      답 전체를 기다리지 않고 이때부터 재생할 수 있습니다.
      Mimi도 프레임 단위로 복원할 수 있어서 **말하면서 생성**하는 스트리밍이 됩니다.
    - 글은 {_text_end}걸음 근처에서 끝나지만 소리는 그 뒤로도 이어집니다(전체 {len(stream_steps)}걸음). 글 1토큰 ≈ 소리 여러 프레임이기 때문입니다.
    """),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 8. Talker가 주로 보는 것: `text_scale`과 `audio_scale`

    Talker 입력 = $h_{\mathrm{talker}}=s_t P_t(h_{\mathrm{bridge}})+s_a P_a(c_{\mathrm{audio}})$. 두 값도 학습되므로, 학습 후 값을 보면
    Talker가 Thinker 쪽과 지난 소리 쪽에 어떤 비중을 두게 되었는지 짐작할 수 있습니다(두 벡터의 크기도 함께 봐야 합니다).
    """)
    return


@app.cell
def _(base_omni, device, model_a, model_b, model_c, talker_batch, torch):
    @torch.no_grad()
    def mix_stats(model):
        x = talker_batch[0].to(device)
        cap = {}
        h = model.thinker.layers[model.config.bridge_layer].register_forward_hook(lambda m, i, o: cap.update(b=o[0]))
        model(x)
        h.remove()
        text_part = model.talker.embed_proj(cap["b"]) * model.talker.text_scale
        audio_part = model.talker.codec_proj(model.talker.embed_tokens(x[:, :8])) * model.talker.audio_scale
        return {"text_scale": model.talker.text_scale.item(), "audio_scale": model.talker.audio_scale.item(),
                "Thinker 쪽 벡터 평균 길이": text_part.norm(dim=-1).mean().item(),
                "소리 쪽 벡터 평균 길이": audio_part.norm(dim=-1).mean().item()}

    scale_rows = [{"모델": n, **{k: f"{v:.2f}" for k, v in mix_stats(m).items()}}
                  for n, m in [("학습 전", base_omni), ("A", model_a), ("C", model_c), ("B", model_b)]]
    return (scale_rows,)


@app.cell(hide_code=True)
def _(mo, result_table, scale_rows):
    mo.vstack([
        result_table(scale_rows, selection=None),
        mo.md(r"""
    C는 `text_scale = 0`으로 고정했으므로 Thinker 쪽이 0입니다. A·B에서 Thinker 쪽 벡터가 소리 쪽과 함께 쓰이는지 확인하세요.
    두 입력은 `embed_proj`/`codec_proj` 끝의 RMSNorm을 거친 뒤 scale이 곱해집니다. RMSNorm의 가중치도 학습되므로 scale만이 아니라
    오른쪽 두 열(실제 벡터 길이)을 함께 비교해야 합니다.
    """),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 9. 듣기(A2A) 경로: SenseVoice → `audio_proj` → `<|audio_pad|>`

    원본 2단계(A2A)는 사람의 **질문 목소리**를 Thinker에 넣습니다. 그림과 같은 "자리표시자 갈아 끼우기"지만 두 가지가 다릅니다.

    | | 그림 | 소리 (질문) | 소리 (답변) |
    | --- | --- | --- | --- |
    | 인코더 | SigLIP2 (고정) | **SenseVoice-Small** (고정, 234M) | **Mimi** (고정) |
    | 표현 | 연속 벡터 64개 | 연속 벡터, **길이가 녹음마다 다름** (60ms마다 1개) | **이산 코드** 8 × 12.5개/초 |
    | 모델에 넣는 곳 | `<|image_pad|>` × 64 | `<|audio_pad|>` × (인코더 출력 길이) | Talker 입력 8줄 |
    | Projector | `vision_proj` 768→768 | `audio_proj` **512→768** | — |

    **듣는 소리와 말하는 소리의 표현이 다릅니다.** 듣기는 "무슨 말인지"가 중요하므로 음성 인식용 인코더(SenseVoice)의 연속 벡터를,
    말하기는 "다시 소리로 만들 수 있어야" 하므로 코덱(Mimi)의 이산 코드를 씁니다.

    SenseVoice는 `funasr` 패키지와 별도 가중치(약 470MB)가 필요해 이 노트북에서는 불러오지 않았습니다. 대신 SenseVoice 출력과 같은 모양
    (`[길이, 512]`)의 임의 벡터로 `encode_audio_inputs`의 뒷부분(`audio_proj`)과 원본 `inject_audio_features`를 따라갑니다.
    """)
    return


@app.cell
def _(base_omni, device, tokenizer, torch):
    _n_audio = 23  # 약 1.4초 녹음 → SenseVoice 출력 23개라고 가정
    _q = "<|audio_pad|>" * _n_audio + "\n\nPlease answer the question in the audio."
    _text = tokenizer.apply_chat_template([{"role": "user", "content": _q}], tokenize=False, add_generation_prompt=True)
    audio_demo_ids = torch.tensor([tokenizer(_text, add_special_tokens=False).input_ids], device=device)
    with torch.no_grad():
        _fake_sensevoice = torch.randn(_n_audio, base_omni.config.audio_hidden_size, device=device)  # [23, 512]
        _feat = base_omni.audio_proj(_fake_sensevoice)                                                  # [23, 768]
        _h = base_omni.thinker.embed_tokens(audio_demo_ids)
        _out = base_omni.inject_audio_features(audio_demo_ids, _h, [_feat], audio_demo_ids.shape[1])
    _mask = audio_demo_ids[0] == base_omni.config.audio_ids[0]
    assert int(_mask.sum()) == _n_audio and torch.equal(_out[0, _mask], _feat)
    assert torch.equal(_out[0, ~_mask], _h[0, ~_mask])
    audio_inject_info = {"n": _n_audio, "first": int(_mask.nonzero()[0]), "seq": audio_demo_ids.shape[1],
                         "proj_in": tuple(_fake_sensevoice.shape), "proj_out": tuple(_feat.shape)}
    return (audio_inject_info,)


@app.cell(hide_code=True)
def _(audio_inject_info, mo):
    mo.md(f"""
    전체 {audio_inject_info['seq']}토큰 중 위치 {audio_inject_info['first']}부터 `<|audio_pad|>`(id 16) **{audio_inject_info['n']}칸**이
    `audio_proj` 출력 `{audio_inject_info['proj_out']}`(입력 `{audio_inject_info['proj_in']}`)으로 정확히 바뀌었고, 나머지 위치는 그대로였습니다.

    원본 A2A 데이터(`OmniDataset.create_chat_prompt`)는 질문 글을 40% 확률로 **소리만**, 나머지는 글만·소리+글 등으로 섞어,
    같은 답을 글 질문과 소리 질문 양쪽에서 낼 수 있게 합니다. 학습 순서는 먼저 `--mode audio_proj`로 **Projector만** 정렬하고
    (VLM의 정렬 단계와 같음), 그다음 작은 lr로 전체를 조정합니다.
    """)
    return


@app.cell(hide_code=True)
def _(BASE_REPO, mo):
    mo.md(rf"""
    ## 10. 실제 규모로 실행하려면

    데이터(`sft_t2a_mini.parquet`, `sft_a2a_mini.parquet`)와 SenseVoice·SigLIP2·Mimi 준비는 `projects/minimind-o/README_ko.md`를 따릅니다
    (SenseVoice에는 `funasr`가 필요합니다). 이 노트북의 베이스로 시작하려면, 원본이 찾는 `../out/llm_768.pth` 자리에 MiniMind-V 가중치를 저장합니다.
    파일에 `vision_proj.*`도 들어가므로 그림 경로가 함께 옮겨지고, Talker 층은 `init_omni_model`이 Thinker에서 복사합니다.

    ```python
    # projects/minimind-o/ 에서 실행: MiniMind-V → out/llm_768.pth
    import torch
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    sd = load_file(hf_hub_download("{BASE_REPO}", "model.safetensors",
                                   revision="4e6f87f184b27f0692e1f8b3afed959de254894b"))
    torch.save({{k: v.half() for k, v in sd.items()}}, "out/llm_768.pth")
    ```

    ```bash
    cd projects/minimind-o/trainer
    # ① T2A: llm_768.pth에서 시작, 전체 학습 (README: RTX 3090 약 60분)
    python train_sft_omni.py --data_path ../dataset/sft_t2a_mini.parquet --from_weight llm --save_weight sft_zero \
        --learning_rate 5e-4 --epochs 1 --batch_size 40 --max_seq_len 512
    # ② A2A: audio_proj만 정렬 → 전체를 작은 lr로
    python train_sft_omni.py --data_path ../dataset/sft_a2a_mini.parquet --from_weight sft_zero --save_weight sft_zero \
        --learning_rate 5e-4 --max_seq_len 640 --mode audio_proj
    python train_sft_omni.py --data_path ../dataset/sft_a2a_mini.parquet --from_weight sft_zero --save_weight sft_zero \
        --learning_rate 2e-5 --batch_size 16 --max_seq_len 768
    cd .. && python eval_omni.py --weight sft_zero
    ```

    | 인자 | 원본 ① T2A | 이 노트북 A / C / B |
    | --- | --- | --- |
    | `--from_weight` | `llm` | MiniMind-V (`{BASE_REPO.split('/')[1]}`) |
    | 학습 범위 | `--mode all` (전부) | Talker만 / Talker만(다리 끊음) / 전부 |
    | `--learning_rate` | 5e-4 | 5e-4 / 5e-4 / 2e-4 |
    | 데이터 | 약 47만 답변, 1 epoch | 32~128문장을 반복 (기본 64문장, 800 step) |
    | `--max_seq_len` | 512 | 160 |
    | 화자 조건 (`spk_emb`, `ref_audios`) | 데이터에 있으면 사용 | 화자 임베딩 없는 샘플 사용(단일 화자 여부는 별도 확인) |
    | `--dtype` | bfloat16 | bfloat16 autocast |

    ## 11. 정리

    | 항목 | 핵심 |
    | --- | --- |
    | 구조 | Thinker(MiniMind, 글) + Talker(MiniMind 블록 4층, 소리). `MiniMindOmni`는 `MiniMindForCausalLM`을 상속 |
    | 다리 | Talker는 Thinker **중간층**(`bridge_layer`)의 은닉 상태와 지난 소리 코드를 더해 입력으로 받음 |
    | 소리 출력 | Mimi 코드북 8개 × 12.5프레임/초. RVQ라 앞 코드북이 거칠고 뒤 코드북이 세부를 채움 |
    | 한 위치의 입력 | 9줄 = 소리 8줄 + 글 1줄. 코드북 i는 i칸 늦게 시작(지연 배치) |
    | 출력 헤드 | 공통 층 + 코드북별 어댑터로 8개 코드를 동시에 예측(MTP) |
    | 손실 | 글 CE + 소리 CE(8개 평균, STOP × 10). 답변 위치만 정답 |
    | 소리 입력 | SenseVoice 연속 벡터 → `audio_proj` → `<|audio_pad|>` 자리. 출력용 Mimi와 다른 표현 |
    | 확인 방법 | 정해진 문장 읽기 + 받아쓰기 WER, Thinker 연결을 끊은 대조군, 처음 보는 문장 |
    | 한계 | 이 실험에서는 학습 밖 문장 일반화가 미확인. 필요한 데이터 규모는 추가 검증 대상. Thinker 갱신 시 망각 관찰 |

    ## 12. 직접 해 볼 과제

    각 과제를 하기 전에 **결과를 먼저 예측**해 적고, 바꿔 실행한 뒤 비교하세요.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({
        "과제 1. 학습 문장 수 바꾸기 (3절 위젯)": mo.md(r"""
    N을 32와 128로 바꿔 A의 학습 문장 WER과 처음 보는 문장 WER을 비교하세요. step 수는 그대로 두면 문장당 반복 횟수가 달라집니다.

    *해설:* 문장이 적을수록 같은 step 안에서 더 많이 반복하므로 학습 문장은 더 정확히 외웁니다. 문장이 많으면 외우기는 늦어지지만,
    처음 보는 문장의 WER은 이 규모에서는 거의 1에 머뭅니다. 일반화에 필요한 데이터·계산 규모는 이 실험만으로 정할 수 없으며, 데이터 양·다양성·학습 설정을 나눠 추가 검증해야 합니다.
    """),
        "과제 2. bridge_layer 바꾸기 (4절)": mo.md("""
    `omni_config.bridge_layer`를 7(마지막 층)이나 0(첫 층)으로 바꿔 1절부터 다시 실행하고, A의 학습 문장 WER을 비교하세요.

    *해설:* 첫 층의 표현은 단어 자체에 가깝고, 마지막 층은 "다음 글 토큰 예측"에 특화됩니다. 원본은 그 중간을 택했습니다.
    짧은 실험에서는 차이가 작을 수 있으니 여러 시드로 확인하세요.
    """),
        "과제 3. STOP 가중치 없애기 (6절)": mo.md("""
    `run_training`의 `(1 + stop * 9)`를 `1`로 바꾸면 생성된 목소리의 길이가 어떻게 될까요? `read_aloud`가 만든 프레임 수를 정답과 비교하세요.

    *해설:* STOP은 문장마다 코드북당 1번뿐이라 손실에서 차지하는 비중이 아주 작습니다. 가중치가 없으면 STOP을 늦게 배워
    말이 끝난 뒤에도 소리가 이어지거나 엉뚱한 곳에서 끊길 수 있습니다.
    """),
        "과제 4. 코드북 4개만 쓰는 Talker (2절, 6절)": mo.md("""
    2절에서 코드북 4개로도 알아들을 수 있었습니다. Talker가 코드북 0~3만 맞히도록 손실을 바꾸고(4~7은 손실에서 제외),
    디코딩도 앞 4개만 쓰면 품질과 학습 속도는 어떻게 될까요?

    *해설:* 맞혀야 할 토큰이 절반이 되어 학습은 쉬워지지만 음질(특히 자음과 음색)은 떨어집니다. 지연도 4칸으로 줄어
    첫 소리가 더 빨리 나옵니다. 코덱의 코드북 수는 품질·지연·계산량 사이의 균형입니다.
    """),
        "과제 5. B의 학습률 낮추기 (6절)": mo.md("""
    실험 B의 lr을 2e-4에서 5e-5로 낮추면 학습 문장 WER, 일반 질문 답, 그림 질문 답이 어떻게 달라질까요?

    *해설:* Thinker가 덜 바뀌므로 망각은 줄어들 가능성이 크지만, 같은 step 안에서 답을 외우는 속도와 Talker 학습도 느려집니다.
    원본이 2단계 전체 조정에 2e-5처럼 작은 lr을 쓰는 이유와 연결해 보세요. Thinker와 Talker에 **서로 다른 lr**을 주는 것도 방법입니다.
    """),
    })
    return


if __name__ == "__main__":
    app.run()
