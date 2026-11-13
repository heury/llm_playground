import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # MiniMind-O 학습 실습: `train_sft_omni.py` 해부하고 직접 돌리기

    이 노트북은 Omni 모델의 **학습 스크립트** `projects/minimind-o/trainer/train_sft_omni.py`를 다룹니다.
    스크립트의 각 단계를 노트북 안에서 원본 함수로 하나씩 실행해 보고, 마지막에는 **원본 스크립트를 그대로**
    GPU에서 실행해 체크포인트를 만들고 평가합니다.

    베이스 모델은 시각·언어 모델 [`caspar/MiniMind-V-Dense-64M-LLaVA-Short-v1.0`](https://huggingface.co/caspar/MiniMind-V-Dense-64M-LLaVA-Short-v1.0)입니다.
    그림을 보고 답할 수 있는 모델에 음성 출력(Talker)을 학습시키면서, 학습 설정에 따라 **원래 능력이 어떻게 유지되거나
    무너지는지** 확인합니다.

    | 절 | 내용 | 스크립트에서 해당하는 부분 |
    |---|---|---|
    | 1 | 작업 공간 만들기 | 상대 경로 `../model`, `../out`, `../checkpoints` |
    | 2 | 데이터와 배치 | `OmniDataset`, `omni_collate_fn` |
    | 3 | 모델 초기화와 고정 전략 | `init_omni_model`, `--freeze_backbone`, `--mode` |
    | 4 | 학습 한 step 해부 | `train_epoch`의 손실 계산과 역전파 |
    | 5 | 학습률과 기울기 누적 | `get_lr`, `--accumulation_steps` |
    | 6 | 원본 스크립트로 학습 (2회) | 전체 학습 vs `--freeze_backbone all` |
    | 7 | 체크포인트와 재개 | `omni_checkpoint`, `--from_resume`, `SkipBatchSampler` |
    | 8 | 평가 | 받아 읽기 WER, 텍스트·그림 능력 보존 |
    | 9 | 조사: 고정했는데 왜 바뀌었나? | scheduled sampling + AdamW |
    | 10 | 실제 규모로 | `train.sh`, DDP |

    **준비물**
    - CUDA GPU (VRAM 8GB 이상 권장), 디스크: 실행 중 최대 약 4.7GB, 정리 후 약 0.8GB (7-3절)
    - 전체 실행 약 12~14분 (RTX 3080 Ti 기준). 원본 스크립트로 학습을 4번(full 약 5분, frozen 약 3~4분, 재개 실습 2회) 실행합니다.
    - 처음 실행할 때 약 1.3GB 다운로드 (데이터 881MB, SigLIP2, Mimi, Whisper-tiny, 베이스 모델)
    - 실행 산출물은 저장소 루트의 `outputs/notebooks/<실행 ID>/`에 저장됩니다.

    > 원본 코드: [jingyaogong/minimind-o](https://github.com/jingyaogong/minimind-o) (Apache-2.0).
    > 베이스 모델의 라이선스와 학습 데이터 출처는 각 모델 카드를 따릅니다.
    """)
    return


@app.cell
def _():
    import io
    import os
    import re
    import sys
    import json
    import time
    import random
    import subprocess
    import warnings
    from pathlib import Path

    import numpy as np
    import torch
    import torch.nn as nn
    import librosa
    import pyarrow as pa
    import pyarrow.parquet as pq
    import matplotlib.pyplot as plt
    from PIL import Image
    from huggingface_hub import hf_hub_download, snapshot_download
    from safetensors.torch import load_file
    from torch.utils.data import DataLoader
    from transformers import (
        MimiModel,
        SiglipImageProcessor,
        WhisperForConditionalGeneration,
        WhisperProcessor,
    )
    from transformers import logging as hf_logging

    hf_logging.set_verbosity_error()

    REPO_ROOT = Path(__file__).resolve().parents[2]
    OMNI_DIR = REPO_ROOT / "projects" / "minimind-o"
    TRAIN_SCRIPT = OMNI_DIR / "trainer" / "train_sft_omni.py"
    if str(OMNI_DIR) not in sys.path:
        sys.path.insert(0, str(OMNI_DIR))

    # 학습 스크립트가 쓰는 원본 함수들을 그대로 가져옵니다.
    from model.model_omni import OmniConfig
    from dataset.omni_dataset import OmniDataset
    from trainer.trainer_utils import SkipBatchSampler, get_lr, init_omni_model, setup_seed
    from trainer.train_sft_omni import omni_collate_fn

    return (
        DataLoader,
        Image,
        MimiModel,
        OMNI_DIR,
        OmniConfig,
        OmniDataset,
        Path,
        REPO_ROOT,
        SiglipImageProcessor,
        SkipBatchSampler,
        TRAIN_SCRIPT,
        WhisperForConditionalGeneration,
        WhisperProcessor,
        get_lr,
        hf_hub_download,
        init_omni_model,
        io,
        json,
        librosa,
        load_file,
        nn,
        np,
        omni_collate_fn,
        os,
        pa,
        plt,
        pq,
        random,
        re,
        setup_seed,
        snapshot_download,
        subprocess,
        sys,
        time,
        torch,
        warnings,
    )


@app.cell
def _(mo, torch):
    mo.stop(
        not torch.cuda.is_available(),
        mo.callout(mo.md("이 노트북은 **CUDA GPU**가 필요합니다."), kind="danger"),
    )
    DEVICE = "cuda"
    mo.md(f"GPU: **{torch.cuda.get_device_name(0)}**, VRAM {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    return (DEVICE,)


@app.cell
def _():
    # 재현을 위해 모든 외부 리소스의 리비전을 고정합니다.
    BASE_REPO = "caspar/MiniMind-V-Dense-64M-LLaVA-Short-v1.0"
    BASE_REV = "4e6f87f184b27f0692e1f8b3afed959de254894b"
    SIGLIP_REPO = "jingyaogong/siglip2-base-p32-256-ve"
    SIGLIP_REV = "9465d1dc89db6bc6227c5b6b0e0ca9b940325d62"
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
        SIGLIP_REPO,
        SIGLIP_REV,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. 작업 공간: 스크립트가 보는 세상

    `train_sft_omni.py`는 **`trainer/` 폴더에서 실행된다고 가정**하고 상대 경로를 씁니다.

    | 스크립트 안의 경로 | 용도 | 지정 방법 |
    |---|---|---|
    | `../model` | 토크나이저 (`init_omni_model`의 `tokenizer_path` 기본값) | 고정 |
    | `../model/siglip2-base-p32-256-ve` | 비전 인코더 | `--vision_dir` |
    | `../model/SenseVoiceSmall` | 음성 입력 인코더 (없으면 경고 후 `None`) | `--audio_encoder_dir` |
    | `../out/{from_weight}_768.pth` | 시작 가중치, 학습 결과 가중치 | `--save_dir`, `--from_weight` |
    | `../checkpoints/` | 재개용 체크포인트 | **고정** (`omni_checkpoint(save_dir='../checkpoints')`) |

    원본 프로젝트 폴더를 건드리지 않도록, 이 구조를 흉내 낸 **작업 공간**을 `outputs/notebooks/<실행 ID>/workspace/`에
    만들고 그 안의 `trainer/`를 작업 디렉토리로 삼아 원본 스크립트를 실행합니다(스크립트 파일 자체는 원본 경로의 것을 씀).
    스크립트는 `sys.path`에 자기 상위 폴더를 추가하므로 import는 원본 프로젝트에서 이루어집니다.
    """)
    return


@app.cell
def _(
    BASE_REPO,
    BASE_REV,
    OMNI_DIR,
    REPO_ROOT,
    SIGLIP_REPO,
    SIGLIP_REV,
    hf_hub_download,
    load_file,
    mo,
    snapshot_download,
    time,
    torch,
):
    RUN_ID = time.strftime("%Y%m%d-%H%M%S") + "-omni-training"
    RUN_DIR = REPO_ROOT / "outputs" / "notebooks" / RUN_ID
    WS = RUN_DIR / "workspace"
    for _d in ("model", "out", "checkpoints", "dataset", "trainer"):
        (WS / _d).mkdir(parents=True, exist_ok=True)

    # 토크나이저: minimind-o/model의 파일을 링크합니다 (MiniMind·MiniMind-V와 같은 어휘 6,400개).
    for _f in (OMNI_DIR / "model").iterdir():
        if _f.name.startswith("tokenizer") or _f.name in ("special_tokens_map.json", "chat_template.jinja"):
            (WS / "model" / _f.name).symlink_to(_f)
    SIGLIP_DIR = WS / "model" / "siglip2-base-p32-256-ve"
    SIGLIP_DIR.symlink_to(snapshot_download(SIGLIP_REPO, revision=SIGLIP_REV))

    # 시작 가중치: caspar MiniMind-V를 스크립트가 찾는 이름 llm_768.pth로 저장합니다.
    _base_state = load_file(hf_hub_download(BASE_REPO, "model.safetensors", revision=BASE_REV))
    torch.save({k: v.half() for k, v in _base_state.items()}, WS / "out" / "llm_768.pth")
    _groups = sorted({k.split(".")[0] for k in _base_state})

    mo.md(
        f"""
    실행 ID **`{RUN_ID}`**, 작업 공간 `{WS.relative_to(REPO_ROOT)}`

    - `out/llm_768.pth`: 텐서 {len(_base_state)}개, 최상위 그룹 `{_groups}`
      → Thinker(`model.*`)와 `vision_proj`가 들어 있고 Talker·`audio_proj`는 없습니다. `lm_head`는 임베딩과 공유되므로 따로 없습니다.
    - `model/siglip2-base-p32-256-ve` → SigLIP2 스냅숏 링크, `model/SenseVoiceSmall` → 일부러 두지 않음
    """
    )
    return RUN_DIR, SIGLIP_DIR, WS


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. 데이터와 배치

    데이터는 `sft_a2a_mini.parquet`에서 짧고 깨끗한 영어 답변(1.6~4.8초, 화자 임베딩 없음)을 골라 씁니다.
    음성 입력 인코더가 없으므로 질문 음성 열은 쓰지 않고, **텍스트 질문 → 텍스트 + 음성 답변(T2A)**만 학습합니다.

    **왜 같은 문장을 16번 반복해 넣는가?** `train_epoch`은 epoch 마지막 step(`step == iters`)마다 무조건 저장하고,
    한 번 저장할 때 가중치 236MB + 재개용 파일 0.9~1.4GB를 씁니다. 64문장을 batch 16으로 학습하면 epoch당 4 step이라
    800 step에 저장이 200번 일어납니다. 문장을 16번 반복해 epoch를 64 step으로 늘리면 같은 학습량을 12 epoch, 저장 12번으로
    끝낼 수 있습니다.
    """)
    return


@app.cell
def _(DATA_FILE, DATA_REPO, DATA_REV, WS, hf_hub_download, json, mo, pa, pq, random):
    N_TRAIN, N_HELDOUT, REPEAT = 64, 16, 16
    _table = pq.read_table(
        hf_hub_download(DATA_REPO, DATA_FILE, repo_type="dataset", revision=DATA_REV),
        columns=["conversations", "answer_audios", "spk_emb"],
    )
    _rows, _seen = [], set()
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
        _rows.append({"conversations": _c, "answer_audios": _a})
    random.Random(0).shuffle(_rows)
    heldout_rows = _rows[:N_HELDOUT]
    train_rows = _rows[N_HELDOUT : N_HELDOUT + N_TRAIN]

    TRAIN_PARQUET = WS / "dataset" / "train_t2a.parquet"
    pq.write_table(pa.Table.from_pylist(train_rows * REPEAT), TRAIN_PARQUET)
    mo.md(
        f"필터 통과 {len(_rows):,}개 중 학습 문장 **{N_TRAIN}개 × {REPEAT}회 = {N_TRAIN * REPEAT}행**을 "
        f"`dataset/train_t2a.parquet`로 저장, 확인용 문장 {N_HELDOUT}개는 학습에 쓰지 않습니다."
    )
    return REPEAT, TRAIN_PARQUET, heldout_rows, train_rows


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 2-1. `OmniDataset` → `omni_collate_fn` → 배치 7개 항목

    스크립트는 데이터셋을 만들 때 모델의 `audio_processor`, `vision_processor`를 넘깁니다. 여기서는 SenseVoice가 없으므로
    `audio_processor=None`이고, SigLIP 처리기는 있으므로 **그림이 없는 샘플에도 0으로 채운 가짜 이미지**가 붙습니다
    (`OmniDataset.__getitem__`: "혼합 학습 시 batch 구성을 맞추기 위한 dummy"). 배치를 직접 만들어 확인합니다.
    """)
    return


@app.cell
def _(
    DataLoader,
    OmniDataset,
    SIGLIP_DIR,
    SiglipImageProcessor,
    TRAIN_PARQUET,
    WS,
    mo,
    omni_collate_fn,
    setup_seed,
):
    from transformers import AutoTokenizer

    MAX_SEQ_LEN = 160  # 스크립트 기본값 512 대신 짧은 답변에 맞춘 길이
    tokenizer = AutoTokenizer.from_pretrained(WS / "model")
    vision_processor = SiglipImageProcessor.from_pretrained(SIGLIP_DIR)
    script_ds = OmniDataset(
        str(TRAIN_PARQUET),
        tokenizer,
        audio_processor=None,
        vision_processor=vision_processor,
        max_length=MAX_SEQ_LEN,
        image_token_len=64,
    )
    setup_seed(0)
    probe_batch = next(iter(DataLoader(script_ds, batch_size=8, shuffle=False, collate_fn=omni_collate_fn)))

    def _desc(v):
        if v is None:
            return "None"
        if isinstance(v, dict):
            return ", ".join(f"`{k}` {tuple(t.shape)}" for k, t in v.items())
        return f"{tuple(v.shape)} {str(v.dtype).replace('torch.', '')}"

    _names = ["input_ids", "labels", "audio_labels", "audio_inputs", "audio_lens", "pixel_values", "spk_emb"]
    _notes = [
        "오디오 8줄 + 텍스트 1줄, 길이는 max_seq_len − 1",
        "텍스트 라벨 (마지막 assistant 답변만, 나머지 −100)",
        "코드북 8개 라벨 (계단 지연, 끝에 STOP=2050)",
        "SenseVoice가 없으므로 None",
        "음성 입력 길이 (모두 0)",
        "그림 없는 샘플용 0 이미지",
        "화자 임베딩 (이 데이터는 없으므로 0)",
    ]
    mo.md(
        "| 항목 | shape / dtype | 의미 |\n|---|---|---|\n"
        + "\n".join(f"| `{n}` | {_desc(v)} | {note} |" for n, v, note in zip(_names, probe_batch, _notes))
        + f"\n\n가짜 이미지의 값이 모두 0인가: **{bool((probe_batch[5]['pixel_values'] == 0).all())}**, "
        f"텍스트 줄에 `<|image_pad|>`(12) 개수: **{int((probe_batch[0][:, 8] == 12).sum())}** → 그림 특징이 끼어들 자리가 없습니다."
    )
    return MAX_SEQ_LEN, probe_batch, script_ds, tokenizer


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. 모델 초기화와 고정 전략

    스크립트 5단계는 다음 순서로 모델을 준비합니다.

    1. `OmniConfig(hidden_size, num_hidden_layers, use_moe)`: 나머지 값은 기본값입니다. caspar 베이스의 head_dim 96,
       intermediate 2432와 기본값이 같아 그대로 맞습니다.
    2. `init_omni_model`: `../out/llm_768.pth`를 `strict=False`로 불러오고, 가중치에 Talker가 없으면
       **Thinker의 마지막 4개 층을 Talker로 복사**합니다. 그다음 `--freeze_backbone`을 적용합니다.
    3. 스크립트 본문에서 `--mode`를 적용합니다 (`audio_proj`/`vision_proj`이면 그 모듈만 학습).

    원본 함수를 그대로 호출해 모델을 만들고, 고정 옵션 조합별로 학습되는 파라미터를 세어 봅니다.
    """)
    return


@app.cell
def _(DEVICE, OmniConfig, SIGLIP_DIR, WS, init_omni_model, warnings):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # SenseVoice 경로 없음 경고
        probe_model, _tok = init_omni_model(
            OmniConfig(hidden_size=768, num_hidden_layers=8),
            from_weight="llm",
            tokenizer_path=str(WS / "model"),
            audio_encoder_path=str(WS / "model" / "SenseVoiceSmall"),
            vision_model_path=str(SIGLIP_DIR),
            save_dir=str(WS / "out"),
            device=DEVICE,
            freeze_backbone="all",
        )
    # 스크립트처럼 비전 인코더를 따로 GPU로 옮깁니다 (등록된 하위 모듈이 아니라서 model.to()로 옮겨지지 않음).
    probe_model.vision_encoder.to(DEVICE)
    return (probe_model,)


@app.cell
def _(mo, probe_model):
    def apply_freeze_and_mode(model, freeze_backbone, mode):
        """init_omni_model의 고정 전략 + 스크립트 본문의 --mode 처리를 같은 순서로 재현합니다."""
        for p in model.parameters():
            p.requires_grad = True
        if freeze_backbone == "all":
            for p in model.model.parameters():
                p.requires_grad = False
        elif freeze_backbone == "last1":
            for p in model.model.parameters():
                p.requires_grad = False
            for p in model.model.layers[-1].parameters():
                p.requires_grad = True
        if mode == "audio_proj":
            for p in model.parameters():
                p.requires_grad = False
            for p in model.audio_proj.parameters():
                p.requires_grad = True
        elif mode == "vision_proj":
            for p in model.parameters():
                p.requires_grad = False
            for p in model.vision_proj.parameters():
                p.requires_grad = True

    _groups = {
        "Thinker 층": lambda n: n.startswith("model.layers"),
        "임베딩(=lm_head)": lambda n: n.startswith("model.embed_tokens") or n.startswith("lm_head"),
        "Talker": lambda n: n.startswith("talker"),
        "vision_proj": lambda n: n.startswith("vision_proj"),
        "audio_proj": lambda n: n.startswith("audio_proj"),
    }
    _saved = {n: p.requires_grad for n, p in probe_model.named_parameters()}
    _rows = []
    for _fb, _mode in [("none", "all"), ("last1", "all"), ("all", "all"), ("none", "audio_proj"), ("none", "vision_proj")]:
        apply_freeze_and_mode(probe_model, _fb, _mode)
        _r = {"--freeze_backbone": _fb, "--mode": _mode}
        for _g, _f in _groups.items():
            _r[_g] = f"{sum(p.numel() for n, p in probe_model.named_parameters() if _f(n) and p.requires_grad) / 1e6:.2f}M"
        _r["합계"] = f"{sum(p.numel() for p in probe_model.parameters() if p.requires_grad) / 1e6:.2f}M"
        _rows.append(_r)
    for _n, _p in probe_model.named_parameters():
        _p.requires_grad = _saved[_n]

    mo.vstack(
        [
            mo.ui.table(_rows, selection=None),
            mo.md(
                """
    - 임베딩은 `lm_head`와 같은 텐서이므로 `named_parameters()`에는 `model.embed_tokens.weight`로 한 번만 나옵니다.
    - `--freeze_backbone all`은 `model.model`(Thinker)만 고정합니다. **Talker뿐 아니라 `vision_proj`, `audio_proj`도 학습 대상**으로 남습니다.
      9절에서 이것이 어떤 결과를 낳는지 봅니다.
    - `--mode`는 고정 전략 **뒤에** 적용되어 그것을 덮어씁니다.
    """
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. 학습 한 step 해부

    `train_epoch`의 손실 계산을 그대로 옮겨 2절의 배치 하나로 실행합니다 (모델은 `--freeze_backbone all` 상태).

    ```python
    loss = (text_loss + audio_loss + res.aux_loss) / args.accumulation_steps
    ```

    - `text_loss`: 라벨이 −100이 아닌 텍스트 위치의 평균 CE
    - `audio_loss`: 코드북마다 `CE × (1 + 9·[정답==STOP])`를 유효 위치 수로 나누고, 8개 코드북 평균
    - `aux_loss`: MoE 균형 손실(Dense라 0) **+ 쓰이지 않는 모듈 파라미터 합 × 0**
      (`audio_proj`, `vision_proj`, `TalkerHead.adapters`, `spk_proj`). DDP에서 모든 파라미터가 기울기를 받도록 하는 장치입니다.
    """)
    return


@app.cell
def _(DEVICE, nn, torch):
    def omni_losses(res, labels, audio_labels, stop_weight=9):
        """train_sft_omni.train_epoch의 손실 계산과 같은 코드입니다 (stop_weight만 인자로 뺌)."""
        loss_fct = nn.CrossEntropyLoss(reduction="none")
        text_loss_raw = loss_fct(res.logits.view(-1, res.logits.size(-1)), labels.view(-1))
        text_mask = (labels.view(-1) != -100).float()
        text_loss = (text_loss_raw * text_mask).sum() / (text_mask.sum() + 1e-9)
        audio_loss = res.audio_logits[0].sum() * 0
        per_layer = []
        for i, al in enumerate(res.audio_logits):
            al_flat = al.reshape(-1, al.size(-1))
            target_flat = audio_labels[:, i, :].reshape(-1)
            layer_loss = loss_fct(al_flat, target_flat)
            valid_mask = (target_flat != -100).float()
            stop_mask = (target_flat == 2050).float()
            weighted = layer_loss * valid_mask * (1 + stop_mask * stop_weight)
            msum = valid_mask.sum()
            if msum > 0:
                v = weighted.sum() / (msum + 1e-9)
                audio_loss = audio_loss + v
                per_layer.append(v)
        audio_loss = audio_loss / 8
        return text_loss, audio_loss, per_layer

    def to_device(batch):
        input_ids, labels, audio_labels, audio_inputs, audio_lens, pixel_values, spk_emb = batch
        pv = {k: v.to(DEVICE) for k, v in pixel_values.items()} if pixel_values is not None else None
        return input_ids.to(DEVICE), labels.to(DEVICE), audio_labels.to(DEVICE), pv, spk_emb.to(DEVICE)

    def forward_batch(model, batch, autocast=True):
        input_ids, labels, audio_labels, pv, spk = to_device(batch)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
            res = model(input_ids, pixel_values=pv, spk_emb=spk)
        return res, labels, audio_labels

    return forward_batch, omni_losses


@app.cell
def _(forward_batch, mo, omni_losses, probe_batch, probe_model, torch):
    probe_model.train()
    probe_model.zero_grad(set_to_none=True)
    _res, _labels, _audio_labels = forward_batch(probe_model, probe_batch)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        _text, _audio, _per = omni_losses(_res, _labels, _audio_labels)
        _, _audio_nostop, _ = omni_losses(_res, _labels, _audio_labels, stop_weight=0)
        _loss = _text + _audio + _res.aux_loss
    _loss.backward()

    def _grad_state(prefix):
        ps = [p for n, p in probe_model.named_parameters() if n.startswith(prefix)]
        if all(p.grad is None for p in ps):
            return "None (requires_grad=False)"
        g = sum(p.grad.float().norm() ** 2 for p in ps if p.grad is not None) ** 0.5
        return f"{g.item():.3g}" + (" ← 0으로 채운 텐서" if g.item() == 0 else "")

    one_step_grads = {
        _pre: _grad_state(_pre)
        for _pre in ["model.layers", "model.embed_tokens", "talker.layers", "talker.lm_head.adapters", "talker.spk_proj", "vision_proj", "audio_proj"]
    }
    probe_model.zero_grad(set_to_none=True)
    probe_model.eval()

    mo.md(
        f"""
    | 손실 | 값 |
    |---|---|
    | text_loss | {_text.item():.3f} |
    | audio_loss (STOP ×10) | {_audio.item():.3f} |
    | audio_loss (STOP 가중치 없음) | {_audio_nostop.item():.3f} |
    | aux_loss | {_res.aux_loss.item():.3f} |
    | 코드북별 audio 손실 | {", ".join(f"{v.item():.2f}" for v in _per)} |

    역전파 후 기울기 크기(L2):

    | 파라미터 묶음 | 기울기 |
    |---|---|
    """
        + "\n".join(f"    | `{k}` | {v} |" for k, v in one_step_grads.items())
        + """

    `vision_proj`, `audio_proj`, `spk_proj`는 `requires_grad=True`이지만 이 배치에서는 쓰이지 않아 **0으로 채운 기울기**를 받습니다
    (`× 0` 장치 때문). `None`과 달리 0 기울기는 옵티마이저 계산에 참여합니다. 9절에서 다시 다룹니다.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. 학습률 스케줄과 기울기 누적

    ### 5-1. `get_lr`

    ```python
    lr * (0.1 + 0.45 * (1 + cos(pi * current_step / total_steps)))   # 시작 1.0×lr → 끝 0.1×lr
    ```

    스크립트는 `get_lr(epoch * iters + step, args.epochs * iters, lr)`로 부르며 step은 1부터 셉니다.
    워밍업은 없습니다. 6절 학습(64 step × 12 epoch)에서의 모양을 그립니다.
    """)
    return


@app.cell
def _(get_lr, plt):
    _iters, _epochs = 64, 12
    _xs = [e * _iters + s for e in range(_epochs) for s in range(1, _iters + 1)]
    _fig, _ax = plt.subplots(figsize=(7, 2.6))
    for _lr, _lab in [(2e-4, "full (lr 2e-4)"), (5e-4, "frozen (lr 5e-4)")]:
        _ax.plot(_xs, [get_lr(x, _epochs * _iters, _lr) for x in _xs], label=_lab)
    _ax.set_xlabel("global step")
    _ax.set_ylabel("lr")
    _ax.legend()
    _ax.set_title("get_lr: cosine from 1.0x to 0.1x, no warmup")
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 5-2. 기울기 누적은 큰 배치와 같은가?

    `--accumulation_steps N`이면 손실을 N으로 나눠 N번 역전파한 뒤 한 번 업데이트합니다. 그런데 손실은 **micro-batch마다
    유효 토큰 수로 나눈 평균**입니다. micro-batch마다 답변 길이가 다르면, "4개씩 2번 누적"은 "8개 한 번"과 토큰 가중치가 달라집니다.
    같은 8개 샘플로 두 방식의 Talker 기울기를 fp32로 비교합니다.
    """)
    return


@app.cell
def _(forward_batch, mo, omni_losses, probe_batch, probe_model, torch):
    def _talker_grad(batches, scale):
        probe_model.zero_grad(set_to_none=True)
        for b in batches:
            res, lab, alab = forward_batch(probe_model, b, autocast=False)
            t, a, _ = omni_losses(res, lab, alab)
            ((t + a + res.aux_loss) / scale).backward()
        g = torch.cat([p.grad.flatten() for p in probe_model.talker.parameters() if p.grad is not None])
        probe_model.zero_grad(set_to_none=True)
        return g

    def _half(b, sl):
        return tuple(
            None if v is None else ({k: t[sl] for k, t in v.items()} if isinstance(v, dict) else v[sl]) for v in b
        )

    _g_big = _talker_grad([probe_batch], 1)
    _g_acc = _talker_grad([_half(probe_batch, slice(0, 4)), _half(probe_batch, slice(4, 8))], 2)
    accum_cos = torch.nn.functional.cosine_similarity(_g_big, _g_acc, dim=0).item()
    accum_rel = ((_g_big - _g_acc).norm() / _g_big.norm()).item()
    _tok_counts = [int((probe_batch[2][i] != -100).sum()) for i in range(8)]
    mo.md(
        f"""
    - 샘플별 오디오 라벨 토큰 수: `{_tok_counts}` (앞 4개 합 {sum(_tok_counts[:4])}, 뒤 4개 합 {sum(_tok_counts[4:])})
    - 두 기울기의 코사인 유사도 **{accum_cos:.4f}**, 상대 차이 **{accum_rel:.3%}**

    방향은 거의 같지만 정확히 같지는 않습니다. 토큰 수가 적은 micro-batch의 토큰이 더 큰 가중치를 받기 때문입니다.
    누적은 큰 배치의 *근사*라고 이해하면 됩니다.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. 원본 스크립트로 학습하기

    이제 `train_sft_omni.py`를 **수정 없이** 하위 프로세스로 실행합니다. 두 설정을 비교합니다.

    | 실행 | 설정 | 의도 |
    |---|---|---|
    | **full** | `--freeze_backbone none --learning_rate 2e-4` | 원본 기본값처럼 전체 학습 |
    | **frozen** | `--freeze_backbone all --learning_rate 5e-4` | Thinker를 고정하고 목소리만 붙이기 |

    공통: `--epochs 12 --batch_size 16 --max_seq_len 160 --from_weight llm --dtype bfloat16(기본)`.
    `--save_interval`은 크게 두어 epoch 끝에서만 저장합니다. 로그의 손실은 `log_interval` step **한 배치의 값**이라 들쭉날쭉합니다.
    """)
    return


@app.cell
def _(MAX_SEQ_LEN, TRAIN_SCRIPT, WS, os, re, subprocess, sys, time):
    LOG_RE = re.compile(
        r"Epoch:\[(\d+)/(\d+)\]\((\d+)/(\d+)\), loss: ([\d.]+), text: ([\d.]+), audio: ([\d.]+), lr: ([\d.]+)"
    )

    def script_command(name, extra_args, epochs):
        return [
            sys.executable, "-u", str(TRAIN_SCRIPT),
            "--save_dir", "../out", "--save_weight", name, "--epochs", str(epochs),
            "--batch_size", "16", "--max_seq_len", str(MAX_SEQ_LEN),
            "--data_path", "../dataset/train_t2a.parquet",
            "--audio_encoder_dir", "../model/SenseVoiceSmall",
            "--vision_dir", "../model/siglip2-base-p32-256-ve",
            "--from_weight", "llm", "--num_workers", "2",
            "--log_interval", "8", "--save_interval", "100000",
            *extra_args,
        ]

    def run_script(name, extra_args, epochs, stop_when=None, tag=""):
        """원본 스크립트를 workspace/trainer에서 실행하고 로그를 파일로 남깁니다.
        stop_when이 주어지면 조건이 참이 되는 순간 프로세스를 중단합니다(재개 실습용)."""
        cmd = script_command(name, extra_args, epochs)
        log_path = WS / f"{name}{tag}.log"
        t0 = time.time()
        with open(log_path, "w") as f:
            proc = subprocess.Popen(
                cmd, cwd=WS / "trainer", stdout=f, stderr=subprocess.STDOUT, env={**os.environ, "PYTHONUNBUFFERED": "1"}
            )
            interrupted = False
            while proc.poll() is None:
                if stop_when is not None and stop_when():
                    proc.terminate()
                    interrupted = True
                    break
                time.sleep(0.2)
            rc = proc.wait()
        text = log_path.read_text()
        records = []
        for m in LOG_RE.finditer(text):
            ep, n_ep, st, iters = map(int, m.groups()[:4])
            loss, tl, al, lr = map(float, m.groups()[4:])
            records.append({"global_step": (ep - 1) * iters + st, "epoch": ep, "loss": loss, "text": tl, "audio": al, "lr": lr})
        return {"name": name, "cmd": cmd, "rc": rc, "seconds": time.time() - t0, "log": text, "records": records, "interrupted": interrupted}

    def log_head(run, keep=("불러온", "Talker", "Trainable", "Model Params", "건너뛰")):
        return "\n".join(l for l in run["log"].splitlines() if l.startswith(keep) or "Epoch" in l and "(64/64)" in l)

    return log_head, run_script


@app.cell
def _(REPO_ROOT, TRAIN_SCRIPT, WS, mo, run_script):
    with mo.status.spinner("full: 전체 학습 중 (원본 스크립트)..."):
        run_full = run_script("sft_full", ["--learning_rate", "2e-4", "--freeze_backbone", "none"], 12)
    assert run_full["rc"] == 0, run_full["log"][-3000:]
    mo.md(
        f"**full** 종료 코드 {run_full['rc']}, {run_full['seconds']:.0f}초\n\n"
        f"```bash\ncd {(WS / 'trainer').relative_to(REPO_ROOT)}\npython {TRAIN_SCRIPT.relative_to(REPO_ROOT)} \\\n    "
        + " ".join(run_full["cmd"][3:])
        + "\n```"
    )
    return (run_full,)


@app.cell
def _(log_head, mo, run_full):
    mo.md(f"```text\n{log_head(run_full)}\n```")
    return


@app.cell
def _(mo, run_script):
    with mo.status.spinner("frozen: Thinker 고정 학습 중 (원본 스크립트)..."):
        run_frozen = run_script("sft_frozen", ["--learning_rate", "5e-4", "--freeze_backbone", "all"], 12)
    assert run_frozen["rc"] == 0, run_frozen["log"][-3000:]
    mo.md(f"**frozen** 종료 코드 {run_frozen['rc']}, {run_frozen['seconds']:.0f}초")
    return (run_frozen,)


@app.cell
def _(log_head, mo, run_frozen):
    mo.md(f"```text\n{log_head(run_frozen)}\n```")
    return


@app.cell
def _(plt, run_frozen, run_full):
    _fig, _axes = plt.subplots(1, 3, figsize=(13, 3.2))
    for _run, _c in [(run_full, "C2"), (run_frozen, "C0")]:
        _x = [r["global_step"] for r in _run["records"]]
        for _ax, _key in zip(_axes, ["audio", "text", "lr"]):
            _ax.plot(_x, [r[_key] for r in _run["records"]], color=_c, label=_run["name"], marker=".", lw=1)
    _axes[0].set_yscale("log")
    for _ax, _t in zip(_axes, ["audio loss (logged batch)", "text loss (logged batch)", "lr"]):
        _ax.set_title(_t)
        _ax.set_xlabel("global step")
    _axes[0].legend()
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **곡선 읽기**

    - **full**은 텍스트 손실이 금세 0 가까이 떨어집니다. 같은 문장을 epoch당 16번 보므로 Thinker가 답변을 **외웁니다**.
    - **frozen**은 Thinker가 고정되어 텍스트 손실이 베이스 수준(약 2.5)에 머물고, 오디오 손실만 떨어집니다.
    - 두 실행 모두 오디오 손실이 0.01~0.05까지 내려갑니다. 손실만 보면 둘 다 "잘 학습된" 것처럼 보이므로 8절에서 실제 출력으로 확인합니다.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7. 체크포인트와 학습 재개

    ### 7-1. 무엇이 어디에 저장되는가

    `train_epoch`은 저장 시점마다 두 번 씁니다.

    1. `{save_dir}/{save_weight}_768.pth`: 추론용 **fp16 가중치** (`audio_encoder.*` 제외)
    2. `omni_checkpoint(..., save_dir='../checkpoints')`: 같은 fp16 가중치 한 벌 + `_resume.pth`
       (fp32 가중치, 옵티마이저 상태, epoch, step, world_size, scaler). 임시 파일에 쓴 뒤 `os.replace`로 바꿔 끼워
       저장 도중 중단돼도 이전 파일이 깨지지 않습니다.
    """)
    return


@app.cell
def _(WS, mo, run_frozen, run_full, torch):
    _files = sorted([*(WS / "out").glob("*.pth"), *(WS / "checkpoints").glob("*.pth")])
    _rows = [{"파일": str(f.relative_to(WS)), "크기(MB)": round(f.stat().st_size / 1e6)} for f in _files]
    _resume = torch.load(WS / "checkpoints" / "sft_frozen_768_resume.pth", map_location="cpu", weights_only=False)
    _w = torch.load(WS / "out" / "sft_frozen_768.pth", map_location="cpu")
    _opt_state = _resume["optimizer"]["state"]
    resume_summary = {
        "keys": sorted(_resume.keys()),
        "epoch": _resume["epoch"],
        "step": _resume["step"],
        "world_size": _resume["world_size"],
        "model_dtype": str(next(iter(_resume["model"].values())).dtype),
        "weights_dtype": str(next(iter(_w.values())).dtype),
        "optimizer_params_with_state": len(_opt_state),
        "has_vision_encoder": any(k.startswith("vision_encoder") for k in _w),
    }
    assert set(_w) == set(_resume["model"])
    _ = (run_full, run_frozen)  # 학습이 끝난 뒤에 실행되도록 의존성 표시
    mo.vstack(
        [
            mo.ui.table(_rows, selection=None),
            mo.md(
                f"""
    `sft_frozen_768_resume.pth` 내용: 키 `{resume_summary['keys']}`

    - 마지막 저장 시점 epoch={resume_summary['epoch']} (0부터 셈), step={resume_summary['step']}, world_size={resume_summary['world_size']}
    - 가중치 dtype: 재개용 `{resume_summary['model_dtype']}`, 추론용 `{resume_summary['weights_dtype']}`
    - 옵티마이저 상태를 가진 텐서 {resume_summary['optimizer_params_with_state']}개
      (Adam은 파라미터마다 1차·2차 모멘트를 저장 → frozen은 학습 파라미터가 적어 full보다 재개 파일이 작음)
    - 비전 인코더 가중치 포함 여부: {resume_summary['has_vision_encoder']} (고정 모듈이라 저장하지 않고 다시 불러옴)
    """
            ),
        ]
    )
    return (resume_summary,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 7-2. 중단했다가 `--from_resume 1`로 이어 하기

    `--save_interval 20`으로 짧은 학습(2 epoch = 128 step)을 시작하고, **첫 재개 파일이 생기는 순간(step 20) 프로세스를 강제 종료**합니다.
    그다음 같은 명령에 `--from_resume 1`을 붙여 다시 실행합니다.

    재개 시 스크립트는
    1. `../checkpoints/{save_weight}_768_resume.pth`에서 가중치·옵티마이저·epoch·step을 복원하고,
    2. 같은 epoch의 데이터 순서를 `setup_seed(42 + epoch)` → `torch.randperm`으로 **똑같이 다시 만든 뒤**,
    3. `SkipBatchSampler`로 이미 학습한 batch 수만큼 건너뜁니다.

    종료 시점이 step 20보다 조금 늦어도, 저장 이후 진행분은 버리고 step 21부터 다시 합니다.
    """)
    return


@app.cell
def _(SkipBatchSampler, mo):
    _demo = list(SkipBatchSampler(list(range(10)), batch_size=3, skip_batches=2))
    mo.md(
        f"`SkipBatchSampler(range(10), batch_size=3, skip_batches=2)` → `{_demo}` "
        f"(앞 2개 batch `[0,1,2]`, `[3,4,5]`를 건너뜀, `len()` = {len(SkipBatchSampler(list(range(10)), 3, 2))})"
    )
    return


@app.cell
def _(WS, mo, resume_summary, run_script, torch):
    _ = resume_summary  # 6절 체크포인트 확인 뒤에 실행
    _resume_file = WS / "checkpoints" / "sft_resume_768_resume.pth"
    _common = ["--learning_rate", "5e-4", "--freeze_backbone", "all", "--save_interval", "20"]
    # 이전 실행에서 남은 파일이 있으면 바로 종료 조건이 참이 되므로 먼저 지웁니다 (이 작업 공간에서 만든 파일만).
    for _f in [_resume_file, WS / "checkpoints" / "sft_resume_768.pth", WS / "out" / "sft_resume_768.pth"]:
        _f.unlink(missing_ok=True)
    with mo.status.spinner("재개 실습: 학습 시작 → step 20 저장 직후 강제 종료..."):
        run_cut = run_script("sft_resume", _common, 2, stop_when=_resume_file.exists, tag="-cut")
    _saved = torch.load(_resume_file, map_location="cpu", weights_only=False)
    cut_point = {"epoch": _saved["epoch"], "step": _saved["step"]}
    _last_logged = max((r["global_step"] for r in run_cut["records"]), default=0)
    with mo.status.spinner("재개 실습: --from_resume 1로 이어 하기..."):
        run_resumed = run_script("sft_resume", _common + ["--from_resume", "1"], 2, tag="-resumed")
    assert run_cut["interrupted"] and run_resumed["rc"] == 0, run_resumed["log"][-2000:]
    _skip_lines = [l for l in run_resumed["log"].splitlines() if "건너뛰" in l]
    assert _skip_lines, "재개 로그에 건너뛰기 메시지가 없습니다"
    _first_resumed = min(r["global_step"] for r in run_resumed["records"])
    mo.md(
        f"""
    - 1차 실행: 재개 파일 생성 직후 종료 (종료 코드 {run_cut['rc']}, 마지막으로 로그된 step {_last_logged}).
      저장된 위치: epoch {cut_point['epoch']}, step {cut_point['step']}
    - 2차 실행 (`--from_resume 1`): 종료 코드 {run_resumed['rc']}, 첫 로그 step {_first_resumed}

    ```text
    {_skip_lines[0]}
    ```
    """
    )
    return (run_resumed,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 7-3. 디스크 정리

    이 노트북은 원본 스크립트를 4번 실행하므로 `workspace/`에 약 4.7GB가 쌓입니다. 대부분은 재개 파일입니다.

    | 파일 | 크기 | 이유 |
    |---|---|---|
    | `checkpoints/*_resume.pth` | 0.9~1.4GB씩 | fp32 가중치 + AdamW 모멘트 2벌 (학습 파라미터가 많을수록 큼) |
    | `checkpoints/*_768.pth` | 236MB씩 | `train_epoch`이 `out/`에 쓴 fp16 가중치를 `omni_checkpoint`가 한 번 더 저장한 **중복본** |
    | `out/*_768.pth` | 130~236MB씩 | 8~9절 평가에 쓰는 가중치 → **남김** |

    재개 실습(7-2)과 파일 확인(7-1)이 끝나면 `checkpoints/`는 더 쓰지 않으므로, 아래 체크박스가 켜져 있으면 지웁니다.
    지우는 대상은 **이번 실행의 작업 공간 안에 있는 `.pth` 파일**뿐입니다. 재개 파일을 직접 열어 보고 싶다면 실행 전에 끄세요.
    """)
    return


@app.cell
def _(mo):
    cleanup_ui = mo.ui.checkbox(value=True, label="재개 실습이 끝나면 checkpoints/의 .pth 삭제 (평가용 out/ 가중치는 남김)")
    cleanup_ui
    return (cleanup_ui,)


@app.cell
def _(WS, cleanup_ui, mo, resume_summary, run_resumed):
    _ = (resume_summary, run_resumed)  # 7-1, 7-2가 끝난 뒤에 실행
    _targets = sorted((WS / "checkpoints").glob("*.pth"))
    _size = sum(f.stat().st_size for f in _targets)
    if cleanup_ui.value:
        for _f in _targets:
            _f.unlink()
        _msg = f"`checkpoints/`의 파일 {len(_targets)}개, **{_size / 1e9:.2f}GB**를 지웠습니다."
    else:
        _msg = f"정리를 건너뛰었습니다. `checkpoints/`에 {len(_targets)}개, {_size / 1e9:.2f}GB가 남아 있습니다."
    _remaining = sum(f.stat().st_size for f in WS.rglob("*") if f.is_file() and not f.is_symlink())
    mo.md(f"{_msg} 작업 공간에 남은 파일 크기: {_remaining / 1e9:.2f}GB")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 8. 학습 결과 평가

    학습된 `out/*.pth`를 원본 `init_omni_model(from_weight=...)`로 다시 불러와 세 모델을 비교합니다.

    | 모델 | 불러오는 가중치 |
    |---|---|
    | base | `llm_768.pth` (Talker는 Thinker 층 복사, 입출력층 무작위) |
    | full | `sft_full_768.pth` |
    | frozen | `sft_frozen_768.pth` |

    평가 항목
    1. **받아 읽기**: 정답 텍스트를 강제로 넣고 Talker만 음성을 만들게 한 뒤 Whisper로 받아 적어 WER 측정 (학습 문장 / 확인용 문장)
    2. **텍스트 능력**: 일반 질문 3개에 대한 답이 베이스와 같은가
    3. **그림 능력**: 그림 2장에 "What is in this image?"
    """)
    return


@app.cell
def _(
    ASR_REPO,
    ASR_REV,
    DEVICE,
    MIMI_REPO,
    MIMI_REV,
    MimiModel,
    OmniConfig,
    SIGLIP_DIR,
    WS,
    WhisperForConditionalGeneration,
    WhisperProcessor,
    init_omni_model,
    librosa,
    re,
    snapshot_download,
    torch,
    warnings,
):
    mimi = MimiModel.from_pretrained(snapshot_download(MIMI_REPO, revision=MIMI_REV)).to(DEVICE).eval()
    _asr_dir = snapshot_download(ASR_REPO, revision=ASR_REV)
    asr_processor = WhisperProcessor.from_pretrained(_asr_dir)
    asr_model = WhisperForConditionalGeneration.from_pretrained(_asr_dir).to(DEVICE).eval()

    def load_trained(name):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model, _ = init_omni_model(
                OmniConfig(hidden_size=768, num_hidden_layers=8),
                from_weight=name,
                tokenizer_path=str(WS / "model"),
                audio_encoder_path=str(WS / "model" / "SenseVoiceSmall"),
                vision_model_path=str(SIGLIP_DIR),
                save_dir=str(WS / "out"),
                device=DEVICE,
            )
        model.vision_encoder.to(DEVICE)
        return model.eval()

    @torch.no_grad()
    def codes_to_wav(codes):
        codes = torch.as_tensor(codes, dtype=torch.long, device=DEVICE)
        codes = torch.where(codes >= 2048, torch.zeros_like(codes), codes)
        return mimi.decode(codes[None]).audio_values[0, 0].float().cpu().numpy()

    @torch.no_grad()
    def transcribe(wav):
        if wav is None or len(wav) < 2400:
            return ""
        feats = asr_processor(
            librosa.resample(wav, orig_sr=24000, target_sr=16000), sampling_rate=16000, return_tensors="pt"
        ).input_features.to(DEVICE)
        return asr_processor.batch_decode(asr_model.generate(feats, max_new_tokens=120), skip_special_tokens=True)[0].strip()

    def _words(s):
        return re.sub(r"[^a-z0-9' ]", " ", s.lower()).split()

    def wer(ref, hyp):
        """단어 오류율. 삽입 때문에 1을 넘을 수 있어 1에서 자릅니다."""
        r, h = _words(ref), _words(hyp)
        dp = list(range(len(h) + 1))
        for i in range(1, len(r) + 1):
            prev, dp[0] = dp[0], i
            for j in range(1, len(h) + 1):
                prev, dp[j] = dp[j], min(dp[j] + 1, dp[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
        return min(1.0, dp[-1] / max(1, len(r)))

    return codes_to_wav, load_trained, transcribe, wer


@app.cell
def _(load_trained, run_frozen, run_full):
    _ = (run_full, run_frozen)
    eval_models = {name: load_trained(w) for name, w in [("base", "llm"), ("full", "sft_full"), ("frozen", "sft_frozen")]}
    return (eval_models,)


@app.cell
def _(DEVICE, codes_to_wav, tokenizer, torch):
    def prompt_ids(question, image_token_len=64):
        """빈 think 블록을 지우고 <image>를 <|image_pad|> × 64로 바꿉니다."""
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": question}], tokenize=False, add_generation_prompt=True
        ).replace("<think>\n\n</think>\n\n", "")
        return tokenizer(text.replace("<image>", "<|image_pad|>" * image_token_len), add_special_tokens=False).input_ids

    @torch.no_grad()
    def chat(model, question, image=None, max_new_tokens=60):
        """원본 generate(stream=True)로 텍스트만 받습니다 (텍스트가 끝나면 멈춤)."""
        torch.manual_seed(0)
        if image is not None:
            question = "<image>\n" + question
        x = torch.tensor([prompt_ids(question)], device=DEVICE)
        pv = None
        if image is not None:
            pv = {k: v.to(DEVICE) for k, v in model.vision_processor(images=image, return_tensors="pt").items()}
        ids = []
        for y, _f in model.generate(
            x, tokenizer.eos_token_id, max_new_tokens=max_new_tokens, temperature=0.01, top_p=1.0, stream=True, pixel_values=pv
        ):
            if y is None:
                break
            ids = y[0].tolist()
        if tokenizer.eos_token_id in ids:
            ids = ids[: ids.index(tokenizer.eos_token_id)]
        return tokenizer.decode(ids, skip_special_tokens=True).strip()

    @torch.no_grad()
    def read_aloud(model, question, answer, max_frames=120):
        """정답 텍스트를 강제로 넣고 Talker만 생성합니다. 지연·샘플링 규칙은 원본 stream_generate와 같습니다."""
        torch.manual_seed(0)
        p = prompt_ids(question)
        a = tokenizer(answer + tokenizer.eos_token, add_special_tokens=False).input_ids
        S = len(p)
        total = S + max(len(a) + 1, max_frames + 9)
        text = torch.tensor((p + a + [201] + [0] * total)[:total], device=DEVICE)
        audio = torch.full((8, total), 2049, dtype=torch.long, device=DEVICE)
        stop, hist = [None] * 8, [[] for _ in range(8)]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(torch.cat([audio[:, :S], text[None, :S]])[None], use_cache=True)
        past, pos = out.past_key_values, S - 1
        while pos < total - 1:
            for i in range(8):
                if pos - S >= i and stop[i] is None:
                    logits = out.audio_logits[i][0, -1].float() / 0.2
                    for prev in hist[i][-3:]:
                        logits[prev] = torch.where(logits[prev] > 0, logits[prev] / 1.05, logits[prev] * 1.05)
                    val, idx = logits.topk(50)
                    code = idx[torch.multinomial(torch.softmax(val, -1), 1)].item()
                    hist[i].append(code)
                    audio[i, pos + 1] = code
                    if code >= 2048:
                        stop[i] = pos + 1
            if all(s is not None for s in stop):
                break
            pos += 1
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(torch.cat([audio[:, pos : pos + 1], text[None, pos : pos + 1]])[None], past_key_values=past, use_cache=True)
            past = out.past_key_values
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

    return chat, read_aloud


@app.cell
def _(eval_models, heldout_rows, json, mo, np, read_aloud, train_rows, transcribe, wer):
    N_EVAL = 16
    read_results = {}
    with mo.status.progress_bar(total=len(eval_models) * 2 * N_EVAL, title="받아 읽기 평가") as _bar:
        for _name, _m in eval_models.items():
            for _split, _rows in [("train", train_rows[:N_EVAL]), ("heldout", heldout_rows[:N_EVAL])]:
                _items = []
                for _r in _rows:
                    _c = json.loads(_r["conversations"])
                    _wav = read_aloud(_m, _c[0]["content"], _c[1]["content"])
                    _hyp = transcribe(_wav)
                    _items.append({"ref": _c[1]["content"], "hyp": _hyp, "wer": wer(_c[1]["content"], _hyp), "wav": _wav})
                    _bar.update()
                read_results[(_name, _split)] = _items
    read_wer = {f"{k[0]}|{k[1]}": float(np.mean([it["wer"] for it in v])) for k, v in read_results.items()}
    mo.vstack(
        [
            mo.md(
                "| 모델 | WER 학습 문장 | WER 확인용 문장 |\n|---|---|---|\n"
                + "\n".join(f"| {n} | {read_wer[n + '|train']:.2f} | {read_wer[n + '|heldout']:.2f} |" for n in eval_models)
            ),
            mo.ui.table(
                [
                    {"모델": n, "정답": read_results[(n, "train")][i]["ref"], "Whisper": read_results[(n, "train")][i]["hyp"]}
                    for i in range(3)
                    for n in eval_models
                ],
                selection=None,
            ),
        ]
    )
    return read_results, read_wer


@app.cell
def _(mo, read_results):
    _players = []
    for _n in ["base", "full", "frozen"]:
        _it = read_results[(_n, "train")][0]
        _players.append(
            mo.vstack([mo.md(f"**{_n}**: {_it['hyp'] or '(음성 없음)'}"), mo.audio(_it["wav"], rate=24000)])
            if _it["wav"] is not None
            else mo.md(f"**{_n}**: (음성 없음)")
        )
    mo.vstack([mo.md(f"학습 문장 예시: *{read_results[('base', 'train')][0]['ref']}*"), mo.hstack(_players, wrap=True)])
    return


@app.cell
def _(Image, OMNI_DIR, chat, eval_models, io, mo):
    GENERAL_QUESTIONS = ["What is the capital of France?", "Give me one tip for better sleep.", "Name three primary colors."]
    _img_dir = OMNI_DIR.parent / "minimind-v" / "dataset" / "eval_images"
    probe_images = {n: Image.open(_img_dir / n).convert("RGB") for n in ["image-04-yellow-car.jpg", "image-01-golden-dog-balloons.jpg"]}

    ability = {"text": {}, "image": {}}
    for _q in GENERAL_QUESTIONS:
        ability["text"][_q] = {n: chat(m, _q) for n, m in eval_models.items()}
    for _fname, _img in probe_images.items():
        ability["image"][_fname] = {n: chat(m, "What is in this image?", image=_img, max_new_tokens=40) for n, m in eval_models.items()}

    def _same(kind):
        return {n: sum(v[n] == v["base"] for v in ability[kind].values()) for n in ("full", "frozen")}

    ability_same = {"text": _same("text"), "image": _same("image")}

    def _png(img):
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    mo.vstack(
        [
            mo.md(
                f"베이스와 같은 답의 개수 — 텍스트 질문 {len(GENERAL_QUESTIONS)}개: full {ability_same['text']['full']}, "
                f"frozen {ability_same['text']['frozen']} / 그림 질문 {len(probe_images)}개: full {ability_same['image']['full']}, "
                f"frozen {ability_same['image']['frozen']}"
            ),
            mo.ui.table([{"질문": q, **{k: v[:90] for k, v in a.items()}} for q, a in ability["text"].items()], selection=None),
            mo.hstack([mo.image(_png(img), width=150) for img in probe_images.values()]),
            mo.ui.table([{"그림": f, **{k: v[:90] for k, v in a.items()}} for f, a in ability["image"].items()], selection=None),
        ]
    )
    return (ability_same,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **결과 읽기** (RTX 3080 Ti 기본 설정 측정값: 받아 읽기 WER 학습 문장 base 0.99 / full 0.33 / frozen 0.26,
    확인용 문장은 셋 다 0.96 이상, 베이스와 같은 답 텍스트 full 0/3·frozen 3/3, 그림 full 0/2·frozen 0/2)

    - **base**의 Talker는 입출력층이 무작위라 아무 말도 못 합니다.
    - **full**과 **frozen** 모두 *정답 텍스트가 주어지면* 학습 문장을 비슷한 수준으로 읽습니다. 처음 보는 문장은 둘 다 읽지 못합니다
      (이 규모에서는 일반화가 아니라 암기).
    - **full**은 Thinker가 64문장을 외우는 동안 일반 질문과 그림 질문에 대한 답이 무너집니다(파국적 망각).
      손실 곡선만으로는 알 수 없는 손상입니다.
    - **frozen**은 Thinker가 비트 단위로 그대로라 일반 질문 답이 베이스와 같습니다.
    - 그런데 **frozen의 그림 답도 베이스와 달라졌습니다.** Thinker를 고정했는데 왜 그림 답이 바뀌었을까요? 9절에서 추적합니다.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 9. 조사: 고정했는데 왜 바뀌었나?

    ### 9-1. 어느 가중치가 얼마나 바뀌었는가

    시작 가중치 `llm_768.pth`와 학습 결과를 모듈 묶음별로 비교합니다 (최대 절대 차이).
    """)
    return


@app.cell
def _(WS, mo, run_frozen, run_full, torch):
    _ = (run_full, run_frozen)
    _base = torch.load(WS / "out" / "llm_768.pth", map_location="cpu")
    _groups = ["model.embed_tokens", "model.layers", "vision_proj"]
    drift = {}
    _rows = []
    for _name, _lr in [("sft_full", 2e-4), ("sft_frozen", 5e-4)]:
        _sd = torch.load(WS / "out" / f"{_name}_768.pth", map_location="cpu")
        _r = {"실행": _name}
        for _g in _groups:
            _ks = [k for k in _base if k.startswith(_g)]
            _d = max((_sd[k].float() - _base[k].float()).abs().max().item() for k in _ks)
            _r[_g] = f"{_d:.4f}"
            drift[(_name, _g)] = _d
        _r["vision_proj 차이 / lr"] = f"{drift[(_name, 'vision_proj')] / _lr:.1f}"
        _rows.append(_r)
    assert drift[("sft_frozen", "model.layers")] == 0 and drift[("sft_frozen", "model.embed_tokens")] == 0
    mo.vstack(
        [
            mo.ui.table(_rows, selection=None),
            mo.md(
                "frozen은 Thinker(층, 임베딩)가 **정확히 0**만큼 바뀌었지만 `vision_proj`는 바뀌었습니다. "
                "그림 특징은 `vision_proj`를 거쳐 Thinker에 들어가므로 그림 답이 달라질 수 있습니다."
            ),
        ]
    )
    return (drift,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 9-2. 단서 1: 평소 배치에서 `vision_proj`의 기울기는 0이다

    4절에서 본 것처럼 이 데이터에는 `<|image_pad|>` 토큰이 없어 그림 특징이 쓰이지 않고, `vision_proj`는 `× 0` 장치로
    **0 기울기**만 받습니다. AdamW는 0 기울기에 대해 모멘트가 0이면 움직이지 않습니다(가중치 감쇠는 크기를 0.4% 정도 줄일 뿐,
    늘리지는 못함). 그렇다면 0이 아닌 기울기가 **어딘가에서 가끔** 들어왔다는 뜻입니다.

    ### 9-3. 단서 2: scheduled sampling이 가끔 `<|image_pad|>`를 만든다

    `OmniDataset.apply_scheduled_sampling`은 답변 텍스트 위치의 5%를 **무작위 토큰(0~6399)**으로 바꿉니다. 원래 그림 토큰인 자리는
    보호하지만, 무작위로 뽑힌 토큰이 **새로** `<|image_pad|>`(id 12)가 되는 것은 막지 않습니다. 그 자리에 0 이미지의 특징이 끼어들면
    `vision_proj`가 진짜 기울기를 받습니다. 학습 12 epoch와 같은 양의 샘플을 만들어 몇 번 일어나는지 셉니다 (시드는 실제 학습과 다르므로 빈도만 비교).
    """)
    return


@app.cell
def _(OmniDataset, TRAIN_PARQUET, MAX_SEQ_LEN, mo, np, setup_seed, tokenizer):
    _ds = OmniDataset(str(TRAIN_PARQUET), tokenizer, max_length=MAX_SEQ_LEN, image_token_len=64)
    _hits, _label_tokens = 0, []
    for _ep in range(12):
        setup_seed(100 + _ep)
        for _i in range(len(_ds)):
            _x, _tl = _ds[_i][:2]
            _hits += int((_x[8] == 12).any())
            if _ep == 0:
                _label_tokens.append(int((_tl != -100).sum()))
    _n = 12 * len(_ds)
    _expected = _n * np.mean(_label_tokens) * 0.05 / len(tokenizer)
    image_pad_hits = {"samples": _n, "hits": _hits, "expected": float(_expected)}
    mo.md(
        f"샘플 {_n:,}개 중 `<|image_pad|>`가 생긴 샘플: **{_hits}개** "
        f"(기대값 = 샘플 수 × 답변 토큰 {np.mean(_label_tokens):.1f}개 × 5% ÷ 어휘 {len(tokenizer)} ≈ {_expected:.1f}개)"
    )
    return (image_pad_hits,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 9-4. 단서 3: Adam은 드문 기울기를 크게 반영한다

    768 step 중 몇 번뿐인 기울기가 정말 눈에 띄는 변화를 만들까요? Adam의 업데이트는 $\hat m / (\sqrt{\hat v} + \epsilon)$이라
    **기울기의 크기와 무관하게** 원소마다 약 lr 단위로 움직입니다. 한 번 들어온 기울기는 1차 모멘트(감쇠 0.9)로 수십 step 동안
    계속 반영되고, 2차 모멘트(감쇠 0.999)는 천천히 줄어 나눗셈의 분모가 작아집니다. 768 step 중 한 번만 기울기를 주는 실험을 합니다.
    """)
    return


@app.cell
def _(drift, mo, torch):
    _rows = []
    adam_ratio = {}
    for _t in [50, 200, 500, 700]:
        for _g in [1e-3, 5.0]:
            _p = torch.nn.Parameter(torch.zeros(1))
            _opt = torch.optim.AdamW([_p], lr=5e-4)  # 스크립트와 같은 기본 설정 (weight_decay=0.01)
            for _s in range(768):
                _p.grad = torch.tensor([_g]) if _s == _t else torch.zeros(1)
                _opt.step()
            adam_ratio[(_t, _g)] = abs(_p.item()) / 5e-4
            _rows.append({"기울기가 들어온 step": _t, "기울기 크기": _g, "최종 이동량 / lr": f"{adam_ratio[(_t, _g)]:.1f}"})
    mo.vstack(
        [
            mo.ui.table(_rows, selection=None),
            mo.md(
                f"""
    기울기가 **0.001이든 5이든** 단 한 번으로 lr의 약 8~23배만큼 움직입니다.
    실제 학습에서 `vision_proj`의 최대 변화 ÷ lr은 full {drift[('sft_full', 'vision_proj')] / 2e-4:.1f}, frozen {drift[('sft_frozen', 'vision_proj')] / 5e-4:.1f}로
    이 범위에 들어갑니다. 드문 사건 몇 번 + Adam의 정규화로 설명됩니다.

    **교훈**
    - 고정하려는 모듈은 "데이터에 안 나오니까 괜찮다"가 아니라 `requires_grad=False`로 확실히 고정하거나 옵티마이저에서 빼야 합니다.
      `--freeze_backbone all`은 Thinker만 고정합니다.
    - 원본의 `× 0` 장치는 DDP 오류를 막지만, 그 대가로 쓰이지 않는 모듈도 옵티마이저 상태를 갖게 됩니다.
    - 손실 곡선과 텍스트 평가만으로는 이런 부작용을 발견할 수 없습니다. **원래 능력(여기서는 그림)을 따로 평가**해야 합니다.
    """
            ),
        ]
    )
    return (adam_ratio,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 10. 실제 규모로 학습하려면

    `projects/minimind-o/trainer/train.sh`의 단일 GPU 파이프라인을 이 베이스로 바꾸면 다음과 같습니다
    (`../out/llm_768.pth`는 1절과 같은 방법으로 만들고, 데이터는 `jingyaogong/minimind-o_dataset`에서 `../dataset/`로 내려받음).

    ```bash
    cd projects/minimind-o/trainer
    # 1) T2A: 텍스트 → 음성. 이 노트북의 full 실행에 해당
    torchrun --nproc_per_node 1 train_sft_omni.py --learning_rate 5e-4 --data_path ../dataset/sft_t2a_mini.parquet \
        --epochs 1 --batch_size 40 --use_compile 1 --from_weight llm --save_weight sft_zero --max_seq_len 512
    # 2) A2A: audio_proj만 정렬 (SenseVoice 인코더와 funasr 필요)
    torchrun --nproc_per_node 1 train_sft_omni.py --learning_rate 5e-4 --data_path ../dataset/sft_a2a_mini.parquet \
        --epochs 1 --batch_size 40 --from_weight sft_zero --save_weight sft_zero --max_seq_len 640 --mode audio_proj
    # 3) A2A 전체 미세 조정
    torchrun --nproc_per_node 1 train_sft_omni.py --learning_rate 2e-5 --data_path ../dataset/sft_a2a_mini.parquet \
        --epochs 1 --batch_size 16 --from_weight sft_zero --save_weight sft_zero --max_seq_len 768
    ```

    알아 둘 점
    - **DDP**: `torchrun`이 `RANK` 환경 변수를 주면 `init_distributed_mode`가 NCCL 그룹을 만들고, `DistributedSampler`로 데이터를 나눕니다.
      재개 시 GPU 수가 바뀌면 `omni_checkpoint`가 step을 `step × 이전 GPU 수 ÷ 현재 GPU 수`로 환산합니다.
    - `--dtype float16`이면 `GradScaler`가 켜집니다. bfloat16(기본)은 지수 범위가 넓어 스케일러가 필요 없습니다.
    - `--use_compile 1`은 `torch.compile`을 씁니다. 저장할 때 `_orig_mod`를 벗겨 원래 이름으로 저장합니다.
    - `--use_wandb`는 실제로는 `swanlab`을 import합니다.
    - 그림 능력을 지키려면 T2A 이후 `sft_i2t.parquet`로 `--mode vision_proj` 단계를 두는 원본 파이프라인 구성을 참고하세요.
    - caspar 베이스는 길이 340으로 학습되었으므로 512~768 길이에서는 위치 일반화를 따로 확인해야 합니다.
    """)
    return


@app.cell
def _(
    RUN_DIR,
    ability_same,
    adam_ratio,
    drift,
    image_pad_hits,
    json,
    mo,
    read_wer,
    resume_summary,
    run_frozen,
    run_full,
):
    _summary = {
        "runs": {r["name"]: {"seconds": r["seconds"], "rc": r["rc"], "cmd": r["cmd"][3:], "last": r["records"][-1]} for r in (run_full, run_frozen)},
        "resume_checkpoint": resume_summary,
        "read_wer": read_wer,
        "ability_same_as_base": ability_same,
        "drift_maxabs": {f"{k[0]}|{k[1]}": v for k, v in drift.items()},
        "image_pad_hits": image_pad_hits,
        "adam_single_event_ratio": {f"t={k[0]},g={k[1]}": v for k, v in adam_ratio.items()},
    }
    (RUN_DIR / "summary.json").write_text(json.dumps(_summary, ensure_ascii=False, indent=1, default=str))
    mo.md(f"요약을 `{RUN_DIR.name}/summary.json`에 저장했습니다. 학습 로그와 평가용 가중치는 `{RUN_DIR.name}/workspace/`에 있습니다.")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 11. 정리

    | 스크립트 단계 | 핵심 | 이 노트북에서 확인한 것 |
    |---|---|---|
    | 경로 | `trainer/` 기준 상대 경로, `../checkpoints` 고정 | 작업 공간으로 원본을 수정 없이 실행 |
    | 데이터 | 9줄 입력 + 라벨 2종 + dummy 이미지 | 배치 7개 항목의 shape |
    | 초기화 | `llm_768.pth` 로드, Talker ← Thinker 4~7층 | 로그 "Talker 층 초기화" |
    | 고정 | `--freeze_backbone`은 Thinker만, `--mode`가 덮어씀 | 조합별 학습 파라미터 수 |
    | 손실 | text CE + 코드북 8개 평균(STOP ×10) + aux(× 0 장치) | 0 기울기와 None의 차이 |
    | 스케줄 | 코사인 1.0→0.1×, 워밍업 없음 | 로그의 lr과 일치 |
    | 누적 | micro-batch별 평균의 합 | 큰 배치와 거의 같지만 정확히 같지는 않음 |
    | 저장·재개 | epoch 끝마다 저장, fp16 + resume, 같은 순서 재생성 후 건너뛰기 | 강제 종료 후 이어 하기 |
    | 평가 | 손실만으로는 부족 | 망각과 `vision_proj` 변화 발견 |

    ## 12. 연습 문제

    1. **기울기 누적**: `--batch_size 8 --accumulation_steps 2`로 frozen 실행을 반복하세요. 손실 곡선과 받아 읽기 WER이 batch 16과 비슷한가요?
    2. **`--freeze_backbone last1`**: Thinker 마지막 층만 열면 받아 읽기 정확도와 텍스트·그림 능력 보존 사이의 균형이 어떻게 바뀌나요?
    3. **`vision_proj` 진짜로 고정하기**: 스크립트를 복사해 `vision_proj`, `audio_proj`의 `requires_grad`를 끄고 frozen을 다시 학습하세요.
       그림 답이 베이스와 같아지나요? (원본 파일은 수정하지 말고 사본으로)
    4. **fp16**: `--dtype float16`으로 바꾸면 로그와 재개 파일의 `scaler` 항목이 어떻게 달라지나요?
    5. **STOP 가중치**: 4절의 `omni_losses`에서 `stop_weight=0`일 때 손실이 얼마나 바뀌는지 보고, 학습에서 이 가중치가 없으면 생성 음성의 길이가 어떻게 될지 예상해 보세요.
    """)
    return


if __name__ == "__main__":
    app.run()
