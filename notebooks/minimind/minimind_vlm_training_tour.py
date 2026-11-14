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
    `uv run marimo edit minimind_vlm_training_tour.py`. 환경·준비물과 권장 순서는 [공통 README](README.md)를 참고하세요.
    위젯 변경 시 의존 셀이 다시 실행됩니다. 모델 크기·데이터·학습 설정 변경 전에는 해당 절의 재실행 범위를 확인하세요.
    """)
    return


@app.cell
def _():
    import contextlib
    import io
    import json
    import math
    import os
    import random
    import re
    import shutil
    import sys
    import tempfile
    import time
    from argparse import Namespace
    from pathlib import Path

    import datasets
    import marimo as mo
    import matplotlib.pyplot as plt
    import pyarrow as pa
    import pyarrow.parquet as pq
    import torch
    import torch.nn.functional as F
    from huggingface_hub import hf_hub_download, snapshot_download
    from PIL import Image, ImageDraw
    from safetensors.torch import load_file

    # 노트북 위치(notebooks/minimind/)에서 두 단계 위가 저장소 루트입니다.
    repo_root = Path(__file__).resolve().parents[2]
    minimind_v_dir = repo_root / "projects/minimind-v"
    sys.path.insert(0, str(minimind_v_dir))

    # 원본 클래스·함수와 두 학습 스크립트 모듈을 그대로 가져옵니다.
    # (스크립트의 학습 본체는 `if __name__ == "__main__":` 아래에 있어서 import만으로는 실행되지 않습니다.)
    import trainer.train_pretrain_vlm as pretrain_script
    import trainer.train_sft_vlm as sft_script
    from dataset.lm_dataset import VLMDataset
    from model.model_vlm import MiniMindVLM, VLMConfig
    from trainer.trainer_utils import (Logger, SkipBatchSampler, get_lr, init_vlm_model, setup_seed,
                                       vlm_checkpoint, vlm_collate_fn)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    datasets.disable_progress_bars()
    return (
        F,
        Image,
        ImageDraw,
        Logger,
        MiniMindVLM,
        Namespace,
        SkipBatchSampler,
        VLMConfig,
        VLMDataset,
        contextlib,
        device,
        get_lr,
        hf_hub_download,
        init_vlm_model,
        io,
        json,
        load_file,
        math,
        minimind_v_dir,
        mo,
        os,
        pa,
        plt,
        pq,
        pretrain_script,
        random,
        re,
        setup_seed,
        sft_script,
        shutil,
        snapshot_download,
        tempfile,
        time,
        torch,
        vlm_checkpoint,
        vlm_collate_fn,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.vstack([
        mo.md(r"""
        # MiniMind-V 학습 따라가기: 정렬(Pretrain) → 시각 SFT

        구조 노트북(`minimind_vlm_tour.py`)에서는 VLM이 **이미지 토큰 64개를 문장에 끼워 넣는 언어 모델**이라는 것을 봤습니다.
        이 노트북은 그 모델을 **어떻게 학습하는지**를 다룹니다. 원본 학습 스크립트 두 개를 따라갑니다.



        **사전 지식:** [VLM 구조](minimind_vlm_tour.py)의 시각 특징·Projector·placeholder 결합과 [사전 학습](minimind_pretrain_tour.py)의 CE·기울기 누적·체크포인트를 먼저 확인하세요.
    """),
        mo.mermaid(r"""
    flowchart TB
     L["llm_768.pth: 언어 모델"] --> P["train_pretrain_vlm.py: 캡션, Projector만 학습"]
     P --> W["pretrain_vlm_768.pth: 시각 정렬 가중치"]
     W --> S["train_sft_vlm.py: 질의응답·캡션·대화, Projector와 첫·마지막 층"]
     S --> F["sft_vlm_768.pth: 시각 지시 학습 가중치"]
    """),
        mo.md(r"""

    베이스 언어 모델은 [`caspar/MiniMind-Dense-64M-Mixed-SFT-v1.0`](https://huggingface.co/caspar/MiniMind-Dense-64M-Mixed-SFT-v1.0)입니다.
    학습 루프는 노트북에서 다시 쓰지 않고, 두 스크립트의 **원본 `train_epoch` 함수를 그대로 호출**합니다.

    **준비:** **NVIDIA GPU가 필요합니다** (RTX 3080 Ti 12GB에서 확인, 최대 약 2.5GB 사용). 처음 실행할 때 언어 모델(약 128MB)과
    이미지 인코더(약 189MB)를 내려받습니다. 학습 6회와 평가를 합쳐 GPU에서 약 3분 걸립니다(내려받기 제외).
    원본처럼 가중치·체크포인트를 파일로 저장하므로 **임시 폴더에 약 2GB**가 필요합니다(`TMPDIR` 환경 변수로 위치를 바꿀 수 있음).

    ## 0. 두 스크립트는 무엇이 다른가

    두 파일을 `diff`로 비교하면 **학습 루프(`train_epoch`)와 본체 1~9단계는 글자 하나까지 같고**, 인자 기본값과 옵티마이저 한 줄만 다릅니다.

    | 인자 | `train_pretrain_vlm.py` (①) | `train_sft_vlm.py` (②) | 의미 |
    | --- | --- | --- | --- |
    | `--data_path` | `pretrain_i2t.parquet` | `sft_i2t.parquet` | 캡션 / 질의응답 + 캡션 + 일반 대화 |
    | `--from_weight` | `llm` | `pretrain_vlm` | 시작 가중치 (`../out/{이름}_768.pth`) |
    | `--freeze_llm` | **2** (Projector만) | **1** (+ 첫·마지막 층) | 학습할 범위 |
    | `--learning_rate` | **4e-4** | **5e-6** | 80배 차이 |
    | `--batch_size` | 16 | 4 | |
    | `--max_seq_len` | 450 | 768 | 대화가 더 김 |
    | `--save_weight` | `pretrain_vlm` | `sft_vlm` | 저장 이름 |
    | 옵티마이저 | `AdamW(filter(requires_grad, ...))` | `AdamW(model.parameters())` | 학습할 매개변수만 넘김 / 전부 넘김 |
    """)
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. 원본과 같은 작업 폴더 만들기

    원본은 `cd trainer` 후 실행하며 **상대 경로**를 씁니다. `init_vlm_model`은 `../model`(토크나이저, 이미지 인코더)과
    `../out/llm_768.pth`를 읽고, `train_epoch`는 `../out`과 `../checkpoints`에 저장합니다.
    그래서 임시 폴더에 같은 구조를 만들고, 학습할 때만 그 안의 `trainer/`로 이동해 실행합니다.

    - `model/`: 저장소의 토크나이저 파일과 Hugging Face에서 받은 `siglip2-base-p32-256-ve`를 링크
    - `out/llm_768.pth`: 베이스 언어 모델의 `model.safetensors`를 원본 형식(fp16 `state_dict`)으로 저장
    """)
    return


@app.cell(hide_code=True)
def _(
    VLMConfig,
    device,
    hf_hub_download,
    json,
    load_file,
    minimind_v_dir,
    mo,
    shutil,
    snapshot_download,
    tempfile,
    torch,
):
    mo.stop(device != "cuda", mo.callout(mo.md(
        "**GPU를 찾지 못했습니다.** 이 노트북은 NVIDIA GPU에서 학습하도록 만들었습니다. "
        "`uv run python -c 'import torch; print(torch.cuda.is_available())'`로 확인하세요."), kind="danger"))

    BASE_REPO, BASE_REVISION = "caspar/MiniMind-Dense-64M-Mixed-SFT-v1.0", "v1.0"
    VISION_REPO, VISION_REVISION = "jingyaogong/siglip2-base-p32-256-ve", "9465d1dc89db6bc6227c5b6b0e0ca9b940325d62"

    from pathlib import Path as _Path
    work_dir = _Path(tempfile.mkdtemp(prefix="minimind_v_work_"))
    for _d in ["model", "out", "checkpoints", "trainer", "dataset"]:
        (work_dir / _d).mkdir()
    for _f in (minimind_v_dir / "model").iterdir():  # 토크나이저·채팅 템플릿
        if _f.is_file() and _f.suffix in {".json", ".jinja"}:
            (work_dir / "model" / _f.name).symlink_to(_f)
    (work_dir / "model/siglip2-base-p32-256-ve").symlink_to(snapshot_download(VISION_REPO, revision=VISION_REVISION))

    _cfg = json.load(open(hf_hub_download(BASE_REPO, "config.json", revision=BASE_REVISION)))
    _state = load_file(hf_hub_download(BASE_REPO, "model.safetensors", revision=BASE_REVISION))
    torch.save({k: v.half() for k, v in _state.items()}, work_dir / "out/llm_768.pth")

    # 스크립트는 VLMConfig(hidden_size=768, num_hidden_layers=8, ...)만 지정합니다. 나머지 기본값이 베이스 모델과 같아야 합니다.
    _default = VLMConfig(hidden_size=768, num_hidden_layers=8)
    config_check = [{"항목": k, "베이스 config.json": _cfg[k], "VLMConfig 기본값": getattr(_default, k)}
                    for k in ["hidden_size", "num_hidden_layers", "num_attention_heads", "num_key_value_heads", "head_dim",
                              "intermediate_size", "vocab_size", "rope_theta", "rms_norm_eps", "use_moe"]]
    assert all(r["베이스 config.json"] == r["VLMConfig 기본값"] for r in config_check)
    disk_free_gb = shutil.disk_usage(work_dir).free / 1e9
    return config_check, disk_free_gb, work_dir


@app.cell(hide_code=True)
def _(config_check, disk_free_gb, mo, result_table, work_dir):
    _items = [mo.md(f"""
    작업 폴더: `{work_dir}` (남은 공간 **{disk_free_gb:.1f}GB**)

    원본 trainer의 상대 경로를 유지하도록 아래 작업 공간을 만듭니다.
    """), mo.mermaid(f"""
    flowchart TB
        ROOT["{work_dir.name}/"] --> MODEL["model/: tokenizer와 SigLIP2"]
        ROOT --> OUT["out/llm_768.pth: 베이스 언어 모델"]
        ROOT --> DATA["dataset/: 실습 parquet"]
        ROOT --> CHECK["checkpoints/: 재개 상태"]
        ROOT --> TRAIN["trainer/: 학습 실행 위치"]
    """), result_table(config_check, selection=None),
        mo.md("스크립트가 만드는 `VLMConfig`의 기본값이 베이스 모델 설정과 모두 같으므로, 원본 인자 그대로 가중치를 넣을 수 있습니다.")]
    if disk_free_gb < 3:
        _items.insert(0, mo.callout(mo.md(f"임시 폴더의 남은 공간이 {disk_free_gb:.1f}GB입니다. 약 2GB가 필요합니다. "
                                         "`TMPDIR=/여유있는/경로 uv run marimo edit`처럼 위치를 바꾸세요."), kind="warn"))
    mo.vstack(_items)
    return


@app.cell
def _(VLMConfig, contextlib, init_vlm_model, io, work_dir):
    def in_trainer_dir():
        """원본처럼 trainer/ 에서 실행하는 상태를 잠시 만듭니다."""
        return contextlib.chdir(work_dir / "trainer")

    # 학습 전 기준 모델: 원본 init_vlm_model로 llm_768.pth를 읽은 VLM (Projector는 무작위)
    with in_trainer_dir(), contextlib.redirect_stdout(io.StringIO()) as _out:
        base_vlm, tokenizer, _ = init_vlm_model(VLMConfig(hidden_size=768, num_hidden_layers=8), from_weight="llm",
                                                device="cuda:0", freeze_llm=2)
    base_vlm.eval()
    base_init_log = _out.getvalue().strip()
    return base_vlm, in_trainer_dir, tokenizer


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. 실습 세계와 두 종류의 데이터

    그림 한 장에 도형 하나가 있고, 네 가지 속성이 무작위입니다.

    | 속성 | 값 |
    | --- | --- |
    | 색 | red, green, blue, yellow, purple, orange |
    | 모양 | circle, square, triangle, star |
    | 위치 | left, right |
    | 크기 | small, big |

    색·모양 24가지 조합 중 **6가지는 학습 데이터에 한 번도 넣지 않고** 평가에만 씁니다(빨간 테두리).
    이 조합을 맞히면 그림을 외운 것이 아니라 **속성을 따로 읽은 것**입니다.
    """)
    return


@app.cell
def _(Image, ImageDraw, io, math):
    COLORS = {"red": (220, 40, 40), "green": (40, 170, 60), "blue": (40, 80, 220),
              "yellow": (240, 210, 30), "purple": (140, 60, 180), "orange": (245, 140, 20)}
    SHAPES = ["circle", "square", "triangle", "star"]
    HELD_OUT = {(c, SHAPES[i % len(SHAPES)]) for i, c in enumerate(COLORS)}  # 색마다 하나씩
    TRAIN_COMBOS = [(c, s) for c in COLORS for s in SHAPES if (c, s) not in HELD_OUT]

    def draw_scene(color, shape, pos, size, rng):
        """밝은 배경 위 도형 하나. pos는 왼쪽/오른쪽 절반, size는 반지름 범위를 정합니다."""
        img = Image.new("RGB", (256, 256), tuple(rng.randint(225, 255) for _ in range(3)))
        d = ImageDraw.Draw(img)
        r = rng.randint(20, 30) if size == "small" else rng.randint(42, 52)
        cx = rng.randint(r + 4, 124 - r) if pos == "left" else rng.randint(132 + r, 252 - r)
        cy = rng.randint(r + 4, 252 - r)
        fill = tuple(max(0, min(255, v + rng.randint(-20, 20))) for v in COLORS[color])
        if shape == "circle":
            d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=fill)
        elif shape == "square":
            d.rectangle([cx - r, cy - r, cx + r, cy + r], fill=fill)
        elif shape == "triangle":
            d.polygon([(cx, cy - r), (cx - r, cy + r), (cx + r, cy + r)], fill=fill)
        else:
            d.polygon([(cx + (r if k % 2 == 0 else 0.45 * r) * math.sin(k * math.pi / 5),
                        cy - (r if k % 2 == 0 else 0.45 * r) * math.cos(k * math.pi / 5)) for k in range(10)], fill=fill)
        return img

    def random_attrs(rng, combos):
        c, s = rng.choice(combos)
        return {"color": c, "shape": s, "pos": rng.choice(["left", "right"]), "size": rng.choice(["small", "big"])}

    def png_bytes(img):
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    # 평가용 그림: 24가지 조합 × 2장 (학습과 다른 시드)
    _rng = __import__("random").Random(999)
    TEST_ITEMS = []
    for _c in COLORS:
        for _s in SHAPES:
            for _ in range(2):
                _a = {"color": _c, "shape": _s, "pos": _rng.choice(["left", "right"]), "size": _rng.choice(["small", "big"])}
                TEST_ITEMS.append((_a, draw_scene(_c, _s, _a["pos"], _a["size"], _rng)))
    return (
        COLORS,
        HELD_OUT,
        SHAPES,
        TEST_ITEMS,
        TRAIN_COMBOS,
        draw_scene,
        png_bytes,
        random_attrs,
    )


@app.cell(hide_code=True)
def _(COLORS, HELD_OUT, SHAPES, draw_scene, mo, plt, random):
    _rng = random.Random(5)
    _fig, _axes = plt.subplots(len(SHAPES), len(COLORS), figsize=(9, 6))
    for _i, _s in enumerate(SHAPES):
        for _j, _c in enumerate(COLORS):
            _pos, _size = _rng.choice(["left", "right"]), _rng.choice(["small", "big"])
            _ax = _axes[_i, _j]
            _ax.imshow(draw_scene(_c, _s, _pos, _size, _rng))
            _ax.set_xticks([]); _ax.set_yticks([])
            _held = (_c, _s) in HELD_OUT
            _ax.set_title(f"{_size} {_c} {_s}\n{_pos}", fontsize=7, color="crimson" if _held else "black")
            if _held:
                for _sp in _ax.spines.values():
                    _sp.set_edgecolor("crimson"); _sp.set_linewidth(3)
    _fig.tight_layout()
    _plot = mo.as_html(_fig)
    plt.close(_fig)
    _plot
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 2-1. 단계마다 데이터가 다르다

    원본 README에 따르면 실제 데이터는 이렇게 구성됩니다.

    | | Pretrain (`pretrain_i2t.parquet`) | SFT (`sft_i2t.parquet`) |
    | --- | --- | --- |
    | 규모 | 약 127만 행 | 약 290만 행 |
    | 내용 | 그림 설명(캡션) | 이미지 지시 대화 140만 + **캡션 127만** + **일반 텍스트 대화 23만** |
    | 일반 텍스트 대화의 그림 | — | **8×8 검은색 자리표시자 그림** |

    같은 구성을 작게 만듭니다.

    - **Pretrain 1,440행:** `Describe the image.` 등 3가지 요청 → `A big red circle on the left.`
    - **SFT 392행:**
      - 질의응답 240행: 한 번에 하나의 속성만 짧게 답합니다(`What color is the shape?` → `It is red.`). 그중 일부는 두 번 묻는 **여러 턴** 대화입니다.
      - 캡션 120행: Pretrain과 같은 형식이지만 새로 그린 그림입니다.
      - 일반 대화 32행: 이미지 없는 질문 8개와 **베이스 모델 자신의 답**을 4번씩 넣었습니다. 그림 자리에는 8×8 검은 그림을 넣습니다.
    - **비교용 SFT (질의응답만) 240행:** 캡션과 일반 대화를 뺀 것입니다(5절).
    """)
    return


@app.cell
def _(MiniMindVLM, autocast_or_not, device, tokenizer, torch):
    def render_prompt(messages):
        """대화 → 추론 프롬프트 토큰. 학습 데이터 80%처럼 빈 think 블록을 지우고 <image>를 <|image_pad|>×64로 바꿉니다."""
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        text = text.replace("<think>\n\n</think>\n\n", "").replace("<image>", "<|image_pad|>" * 64)
        return tokenizer(text, add_special_tokens=False).input_ids

    @torch.inference_mode()
    def generate_reply(model, img, messages, max_new_tokens=20):
        """탐욕적 생성. img가 None이면 그림 없이 글만."""
        x = torch.tensor([render_prompt(messages)], device=device)
        pv = ({k: v.unsqueeze(0).to(device) for k, v in MiniMindVLM.image2tensor(img, model.processor).items()}
              if img is not None else None)
        with autocast_or_not(model):
            y = model.generate(inputs=x, pixel_values=pv, max_new_tokens=max_new_tokens, do_sample=False,
                               temperature=1.0, top_p=1.0, top_k=0, eos_token_id=tokenizer.eos_token_id)
        new = y[0, x.shape[1]:].tolist()
        if tokenizer.eos_token_id in new:
            new = new[: new.index(tokenizer.eos_token_id)]
        return tokenizer.decode(new, skip_special_tokens=True).strip()

    def ask(model, img, question, max_new_tokens=20):
        content = ("<image>\n" if img is not None else "") + question
        return generate_reply(model, img, [{"role": "user", "content": content}], max_new_tokens)

    return ask, generate_reply, render_prompt


@app.cell
def _(contextlib, torch):
    def autocast_or_not(model):
        """fp32 모델은 학습과 같은 bf16 autocast, eval_vlm.py처럼 .half()로 바꾼 모델은 그대로 실행"""
        if next(model.parameters()).dtype == torch.float16:
            return contextlib.nullcontext()
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)

    return (autocast_or_not,)


@app.cell
def _(ask, base_vlm):
    TEXT_EVAL_Q = ["What is the capital of France?", "How many legs does a spider have?",
                   "What do bees make?", "Which ocean is the largest?"]
    REPLAY_Q = ["What is the capital of Italy?", "What color is grass?", "How many days are in a week?",
                "What is the opposite of hot?", "Who wrote Romeo and Juliet?", "What gas do plants absorb from the air?",
                "How many hours are in a day?", "What is H2O commonly called?"]
    # 베이스 모델 자신의 답: SFT에 섞을 일반 대화(replay)와, 언어 능력 유지를 재는 기준
    replay_pairs = [(q, ask(base_vlm, None, q, 40)) for q in REPLAY_Q]
    base_text_answers = [ask(base_vlm, None, q, 24) for q in TEXT_EVAL_Q]
    return TEXT_EVAL_Q, base_text_answers, replay_pairs


@app.cell
def _(
    Image,
    TRAIN_COMBOS,
    draw_scene,
    json,
    pa,
    png_bytes,
    pq,
    random,
    random_attrs,
    replay_pairs,
    work_dir,
):
    CAPTION_PROMPTS = ["Describe the image.", "What is in this picture?", "Give a short caption for this image."]
    QA = {"color": ("What color is the shape?", "It is {color}."),
          "shape": ("What shape is it?", "It is a {shape}."),
          "pos": ("Is the shape on the left or the right?", "It is on the {pos}."),
          "size": ("Is the shape big or small?", "It is {size}.")}
    BLACK_8X8 = png_bytes(Image.new("RGB", (8, 8), (0, 0, 0)))

    def conversation(turns, with_image=True):
        msgs = []
        for i, (q, a) in enumerate(turns):
            msgs += [{"role": "user", "content": ("<image>\n" if with_image and i == 0 else "") + q},
                     {"role": "assistant", "content": a}]
        return json.dumps(msgs)

    def caption_rows(n, seed):
        rng = random.Random(seed)
        rows = []
        for _ in range(n):
            a = random_attrs(rng, TRAIN_COMBOS)
            rows.append({"conversations": conversation([(rng.choice(CAPTION_PROMPTS), "A {size} {color} {shape} on the {pos}.".format(**a))]),
                         "image_bytes": png_bytes(draw_scene(a["color"], a["shape"], a["pos"], a["size"], rng)), "kind": "caption"})
        return rows

    def qa_rows(n, seed):
        rng = random.Random(seed)
        rows = []
        for _ in range(n):
            a = random_attrs(rng, TRAIN_COMBOS)
            keys = rng.sample(list(QA), rng.choice([1, 2]))  # 한 번 또는 두 번 묻기
            turns = [(QA[k][0], QA[k][1].format(**a)) for k in keys]
            rows.append({"conversations": conversation(turns),
                         "image_bytes": png_bytes(draw_scene(a["color"], a["shape"], a["pos"], a["size"], rng)),
                         "kind": "qa (2턴)" if len(keys) == 2 else "qa"})
        return rows

    text_rows = [{"conversations": conversation([(q, a)], with_image=False), "image_bytes": BLACK_8X8, "kind": "text"}
                 for q, a in replay_pairs] * 4

    def write_parquet(rows, name, seed=0):
        rows = list(rows)
        random.Random(seed).shuffle(rows)
        pq.write_table(pa.Table.from_pylist([{"conversations": r["conversations"], "image_bytes": r["image_bytes"]} for r in rows]),
                       work_dir / "dataset" / name)
        return rows

    pretrain_rows = write_parquet(caption_rows(1440, seed=0), "pretrain_i2t.parquet")
    _qa = qa_rows(240, seed=1)
    sft_rows = write_parquet(_qa + caption_rows(120, seed=7) + text_rows, "sft_i2t.parquet")
    qa_only_rows = write_parquet(_qa, "sft_qa_only.parquet")
    return QA, pretrain_rows, qa_only_rows, sft_rows


@app.cell
def _(VLMDataset, base_vlm, random, sft_rows, tokenizer, work_dir):
    # 원본 VLMDataset으로 읽어서 종류별 첫 샘플의 입력과 정답(labels)을 봅니다.
    _ds = VLMDataset(str(work_dir / "dataset/sft_i2t.parquet"), tokenizer, preprocess=base_vlm.processor, max_length=128)
    data_examples = {}
    for _i, _row in enumerate(sft_rows):
        if _row["kind"] in data_examples:
            continue
        random.seed(0)
        _ids, _labels, _pix = _ds[_i]
        data_examples[_row["kind"]] = {
            "입력": tokenizer.decode(_ids).replace("<|image_pad|>" * 64, "<|image_pad|>×64").replace("<|endoftext|>", "").strip(),
            "정답": tokenizer.decode(_labels[_labels != -100]),
            "이미지 자리": int((_ids == 12).sum()), "그림 크기": tuple(_pix["pixel_values"].shape)}
    assert data_examples["text"]["이미지 자리"] == 0 and data_examples["qa"]["이미지 자리"] == 64
    return (data_examples,)


@app.cell(hide_code=True)
def _(data_examples, mo, pretrain_rows, qa_only_rows, sft_rows):
    from collections import Counter as _Counter

    _count = _Counter(r["kind"] for r in sft_rows)
    _tabs = {k: mo.md(f"""
    ```text
    {v['입력']}
    ```
    정답(`labels != -100`): `{v['정답']!r}` · 이미지 자리 {v['이미지 자리']}개 · `pixel_values` {v['그림 크기']}
    """) for k, v in data_examples.items()}
    mo.vstack([
        mo.md(f"Pretrain **{len(pretrain_rows)}행** · SFT **{len(sft_rows)}행** ({dict(_count)}) · 질의응답만 **{len(qa_only_rows)}행**. "
              "원본 `VLMDataset`이 만든 SFT 샘플 (종류별 첫 번째):"),
        mo.ui.tabs(_tabs),
        mo.md(r"""
    - **여러 턴:** 두 답변이 모두 정답입니다(SFT의 `generate_labels` 규칙 그대로). 그림은 첫 질문에만 붙습니다.
    - **일반 대화:** 질문에 `<image>`가 없으므로 이미지 자리가 0개입니다. `pixel_values`(검은 8×8 그림)는 배치를 맞추려고 넣을 뿐,
      `count_vision_proj`가 바꿔 끼울 자리가 없으므로 **모델 입력에 전혀 영향을 주지 않습니다.** 그림이 섞인 배치에서도 글만으로 학습하는 방법입니다.
    """),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. 학습 루프 해부

    두 스크립트의 본체(`if __name__ == "__main__":`)는 9단계입니다.

    ```python
    # 1. 환경과 시드      local_rank = init_distributed_mode(); setup_seed(42)
    # 2. 설정과 재개 확인  vlm_config = VLMConfig(hidden_size=768, num_hidden_layers=8, max_seq_len=..., use_moe=...)
    #                    ckp_data = vlm_checkpoint(vlm_config, weight=save_weight, save_dir='../checkpoints') if from_resume else None
    # 3. 혼합 정밀도      autocast_ctx = torch.cuda.amp.autocast(dtype=bfloat16)
    # 4. wandb (선택)
    # 5. 모델·데이터·옵티마이저
    #    model, tokenizer, preprocess = init_vlm_model(vlm_config, from_weight, freeze_llm=...)   # ../out/{from_weight}_768.pth를 읽고 고정 전략 적용
    #    train_ds = VLMDataset(data_path, tokenizer, preprocess=preprocess, max_length=max_seq_len)
    #    scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))                          # bf16이면 꺼짐
    #    optimizer = AdamW(...)
    # 6. 재개: model, optimizer, scaler, epoch, step 복원
    # 7. torch.compile / DDP (선택)
    # 8. epoch 반복: setup_seed(42 + epoch); indices = randperm(...)
    #               SkipBatchSampler(indices, batch_size, skip)  → DataLoader(..., collate_fn=vlm_collate_fn)
    #               train_epoch(epoch, loader, iters, start_step)
    # 9. DDP 정리
    ```

    `train_epoch`의 한 step:

    ```python
    lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)   # 코사인 감소
    with autocast_ctx:
        res = model(input_ids, labels=labels, pixel_values=pixel_values)
        loss = (res.loss + res.aux_loss) / args.accumulation_steps          # Dense 모델은 aux_loss = 0
    scaler.scale(loss).backward()
    if step % args.accumulation_steps == 0 or step == iters:               # 기울기 누적 후 한 번에 갱신
        scaler.unscale_(optimizer); clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
    if step % args.log_interval == 0 or step == iters: Logger(f'Epoch:[..]({step}/{iters}), loss: .., lr: ..')
    if step % args.save_interval == 0 or step == iters:                     # 저장 (7절)
        torch.save(가중치(vision_encoder 제외, fp16), f'{save_dir}/{save_weight}_768.pth')
        vlm_checkpoint(..., save_dir='../checkpoints')                      # 재개용 상태
    ```

    **노트북에서 실행하는 방법 (`run_stage`):** 9단계 본체를 함수로 옮기되(DDP·wandb·compile 제외),
    한 epoch 학습은 **원본 모듈의 `train_epoch`를 그대로 호출**합니다. `train_epoch`는 모듈 전역 변수
    `args`, `model`, `optimizer`, `scaler`, `autocast_ctx`, `vlm_config`를 읽으므로, 실행 전에 그 값을 모듈에 넣어 줍니다.
    `Logger`가 찍는 로그는 모아서 손실 곡선을 그립니다(`log_interval=1`).
    """)
    return


@app.cell(hide_code=True)
def _(get_lr, mo, plt):
    _total = 200
    _fig, _ax = plt.subplots(figsize=(5.5, 2.4))
    _ax.plot([get_lr(s, _total, 4e-4) for s in range(_total + 1)], label="pretrain (lr=4e-4)")
    _ax.axhline(4e-4 * 0.1, ls=":", c="gray", lw=0.8)
    _ax.set_xlabel("step"); _ax.set_ylabel("lr"); _ax.legend(fontsize=8)
    _fig.tight_layout()
    _plot = mo.as_html(_fig)
    plt.close(_fig)
    mo.hstack([_plot, mo.md(r"""
    $\eta(s)=\eta_0[0.1+0.45(1+\cos(\pi s/S))]$($s$는 step, $S$는 total)

    처음 `lr`에서 시작해 코사인 곡선으로 줄어들고, 끝에서는 **`lr`의 10%** 에서 멈춥니다(0까지 내려가지 않음).
    warmup은 없습니다. 언어 모델 학습 스크립트들과 같은 함수입니다.
    """)], align="center")
    return


@app.cell
def _(
    Logger,
    Namespace,
    SkipBatchSampler,
    VLMConfig,
    VLMDataset,
    contextlib,
    in_trainer_dir,
    init_vlm_model,
    io,
    os,
    pretrain_script,
    re,
    setup_seed,
    torch,
    vlm_checkpoint,
    vlm_collate_fn,
    work_dir,
):
    from torch.utils.data import DataLoader as _DataLoader

    def script_args(script, **overrides):
        """스크립트의 argparse 기본값 + 노트북 설정"""
        args = dict(save_dir="../out", epochs=2, device="cuda:0", dtype="bfloat16", num_workers=0, accumulation_steps=1,
                    grad_clip=1.0, log_interval=1, save_interval=10**9, hidden_size=768, num_hidden_layers=8, use_moe=0,
                    from_resume=0, use_compile=0, use_wandb=False)
        if script is pretrain_script:
            args.update(save_weight="pretrain_vlm", batch_size=16, learning_rate=4e-4, max_seq_len=450,
                        data_path="../dataset/pretrain_i2t.parquet", from_weight="llm", freeze_llm=2)
        else:
            args.update(save_weight="sft_vlm", batch_size=4, learning_rate=5e-6, max_seq_len=768,
                        data_path="../dataset/sft_i2t.parquet", from_weight="pretrain_vlm", freeze_llm=1)
        args.update(overrides)
        return Namespace(**args)

    class SimulatedCrash(Exception):
        """7절: 학습 도중 프로세스가 죽은 상황을 흉내 냅니다."""

    def run_stage(script, args, crash_after_steps=None):
        """스크립트 본체 1~8단계. 한 epoch는 원본 script.train_epoch가 학습합니다."""
        out = io.StringIO()
        model = optimizer = None
        with in_trainer_dir(), contextlib.redirect_stdout(out):
            # ========== 1. 난수 시드 ==========
            setup_seed(42)
            # ========== 2. 설정, 재개용 체크포인트 확인 ==========
            os.makedirs(args.save_dir, exist_ok=True)
            vlm_config = VLMConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers,
                                   max_seq_len=args.max_seq_len, use_moe=bool(args.use_moe))
            ckp_data = vlm_checkpoint(vlm_config, weight=args.save_weight, save_dir="../checkpoints") if args.from_resume == 1 else None
            # ========== 3. 혼합 정밀도 ==========
            dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
            autocast_ctx = torch.cuda.amp.autocast(dtype=dtype)
            # ========== 5. 모델, 데이터, 옵티마이저 ==========
            model, tokenizer, preprocess = init_vlm_model(vlm_config, from_weight=args.from_weight, device=args.device,
                                                          freeze_llm=args.freeze_llm)
            train_ds = VLMDataset(args.data_path, tokenizer, preprocess=preprocess, image_special_token=vlm_config.image_special_token,
                                  image_token_len=vlm_config.image_token_len, max_length=vlm_config.max_seq_len)
            scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == "float16"))
            if script is pretrain_script:
                optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate)
            else:
                optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
            # ========== 6. 재개 ==========
            start_epoch, start_step = 0, 0
            if ckp_data:
                model.load_state_dict(ckp_data["model"], strict=False)
                optimizer.load_state_dict(ckp_data["optimizer"])
                scaler.load_state_dict(ckp_data["scaler"])
                start_epoch, start_step = ckp_data["epoch"], ckp_data.get("step", 0)
            # train_epoch가 읽는 모듈 전역 변수
            script.args, script.model, script.optimizer = args, model, optimizer
            script.scaler, script.autocast_ctx, script.vlm_config = scaler, autocast_ctx, vlm_config
            # ========== 8. 학습 ==========
            done = 0

            def crash_guard(loader):
                nonlocal done
                for batch in loader:
                    if crash_after_steps is not None and done >= crash_after_steps:
                        raise SimulatedCrash
                    done += 1
                    yield batch

            try:
                for epoch in range(start_epoch, args.epochs):
                    setup_seed(42 + epoch); indices = torch.randperm(len(train_ds)).tolist()
                    skip = start_step if (epoch == start_epoch and start_step > 0) else 0
                    batch_sampler = SkipBatchSampler(indices, args.batch_size, skip)
                    loader = _DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers,
                                         pin_memory=True, collate_fn=vlm_collate_fn)
                    if skip > 0:
                        Logger(f"Epoch [{epoch + 1}/{args.epochs}]: 처음 {start_step}개 단계를 건너뛰고 단계 {start_step + 1}시작")
                        script.train_epoch(epoch, crash_guard(loader), len(loader) + skip, start_step, None)
                    else:
                        script.train_epoch(epoch, crash_guard(loader), len(loader), 0, None)
            except SimulatedCrash:
                Logger(f"*** {done} step 학습 후 중단됨 (SimulatedCrash) ***")
            finally:
                for name in ["args", "model", "optimizer", "scaler", "autocast_ctx", "vlm_config"]:
                    setattr(script, name, None)
        log = out.getvalue()
        losses = [float(x) for x in re.findall(r", loss: ([0-9.]+), logits_loss", log)]
        return model.eval(), log, losses

    def file_sizes(sub):
        return {p.name: f"{p.stat().st_size / 2**20:.0f}MB" for p in sorted((work_dir / sub).iterdir())}

    def remove_resume_files(weight):
        """재개용 파일(checkpoints/)은 크므로 다 쓴 뒤 지웁니다. 추론용 out/ 가중치는 남깁니다."""
        for p in (work_dir / "checkpoints").glob(f"{weight}_768*"):
            p.unlink()

    return file_sizes, remove_resume_files, run_stage, script_args


@app.cell
def _(
    F,
    HELD_OUT,
    QA,
    TEST_ITEMS,
    TEXT_EVAL_Q,
    ask,
    autocast_or_not,
    base_text_answers,
    device,
    render_prompt,
    tokenizer,
    torch,
):
    @torch.no_grad()
    def text_retention(model):
        """이미지 없는 일반 질문에 대한 베이스 모델의 원래 답을, 이 모델이 얼마나 그대로 낼 수 있는지 (토큰당 평균 log p)"""
        total, count = 0.0, 0
        for q, a in zip(TEXT_EVAL_Q, base_text_answers):
            p = render_prompt([{"role": "user", "content": q}])
            ids = torch.tensor([p + tokenizer(a, add_special_tokens=False).input_ids], device=device)
            with autocast_or_not(model):
                logits = model(ids).logits[0, :-1].float()
            lp = F.log_softmax(logits, -1)[torch.arange(ids.shape[1] - 1), ids[0, 1:]][len(p) - 1:]
            total, count = total + lp.sum().item(), count + len(lp)
        return total / count

    def evaluate(model):
        """캡션(속성 4개 모두 맞음), 질의응답(속성별), 답 형식, 언어 능력 유지"""
        res = {}
        for split in ("seen", "held"):
            items = [(a, img) for a, img in TEST_ITEMS if ((a["color"], a["shape"]) in HELD_OUT) == (split == "held")]
            caps = [ask(model, img, "Describe the image.", 16) for _, img in items]
            res[f"{split}_caption"] = sum(all(a[k] in t for k in QA) for (a, _), t in zip(items, caps)) / len(items)
            fmt = []
            for key, (q, _) in QA.items():
                answers = [ask(model, img, q, 10) for _, img in items]
                res[f"{split}_{key}"] = sum(a[key] in t for (a, _), t in zip(items, answers)) / len(items)
                fmt += [t.startswith("It is") for t in answers]
            res[f"{split}_qa"] = sum(res[f"{split}_{k}"] for k in QA) / len(QA)
            res[f"{split}_format"] = sum(fmt) / len(fmt)
        res["retention"] = text_retention(model)
        res["text_answers"] = [ask(model, None, q, 24) for q in TEXT_EVAL_Q]
        return res

    def eval_row(name, r):
        return {"모델": name, "캡션 (학습 조합)": f"{r['seen_caption']:.0%}", "캡션 (제외 조합)": f"{r['held_caption']:.0%}",
                "질의응답 (학습 조합)": f"{r['seen_qa']:.0%}", "질의응답 (제외 조합)": f"{r['held_qa']:.0%}",
                "'It is ...' 형식": f"{r['held_format']:.0%}", "언어 능력 유지 (log p)": f"{r['retention']:.3f}"}

    return eval_row, evaluate


@app.cell
def _(base_vlm, evaluate):
    base_eval = evaluate(base_vlm)
    return (base_eval,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. ① 정렬 단계: `train_pretrain_vlm.py`

    원본 기본값 그대로(`freeze_llm=2`, lr 4e-4, batch 16, `from_weight='llm'`) 실행합니다. 바꾼 것은 두 가지입니다.

    - `--epochs 2` (원본과 같음), `--max_seq_len 128`: `VLMDataset`은 항상 `max_seq_len`까지 **패딩**합니다.
      우리 캡션 샘플은 이미지 64 + 글 약 25토큰이라, 450으로 두면 계산의 80%가 패딩에 쓰입니다.
    - `--num_workers 0`: 노트북 안에서는 데이터 로딩 프로세스를 따로 띄우지 않습니다.
    """)
    return


@app.cell
def _(
    TEST_ITEMS,
    ask,
    evaluate,
    file_sizes,
    pretrain_script,
    remove_resume_files,
    run_stage,
    script_args,
    time,
    torch,
    work_dir,
):
    args_pre = script_args(pretrain_script, max_seq_len=128)
    _t0 = time.time()
    _model, pre_log, pre_losses = run_stage(pretrain_script, args_pre)
    pre_seconds = time.time() - _t0
    pre_eval = evaluate(_model)
    pre_example = (TEST_ITEMS[0][0], ask(_model, TEST_ITEMS[0][1], "What color is the shape?"))

    # 재개용 체크포인트 속 옵티마이저: filter(requires_grad)로 만들었으므로 학습하는 매개변수만 들어 있음
    _ck = torch.load(work_dir / "checkpoints/pretrain_vlm_768_resume.pth", map_location="cpu", weights_only=False)
    pre_ckpt_info = {"param_groups 안의 매개변수": len(_ck["optimizer"]["param_groups"][0]["params"]),
                     "Adam 상태가 있는 매개변수": len(_ck["optimizer"]["state"])}
    del _ck
    pre_out_sizes = file_sizes("out")
    remove_resume_files("pretrain_vlm")
    del _model
    torch.cuda.empty_cache()
    return (
        args_pre,
        pre_ckpt_info,
        pre_eval,
        pre_example,
        pre_log,
        pre_losses,
        pre_out_sizes,
        pre_seconds,
    )


@app.cell(hide_code=True)
def _(
    args_pre,
    base_eval,
    eval_row,
    mo,
    plt,
    pre_eval,
    pre_log,
    pre_losses,
    pre_out_sizes,
    pre_seconds,
    result_table,
):
    _lines = pre_log.splitlines()
    _shown = [l for l in _lines if "Params" in l] + [l for l in _lines if "Epoch" in l][:2] + ["..."] + \
             [l for l in _lines if f"({len(pre_losses) // args_pre.epochs}/" in l]
    _fig, _ax = plt.subplots(figsize=(6, 2.6))
    _ax.plot(pre_losses, lw=0.8)
    _ax.set_yscale("log"); _ax.set_xlabel("step"); _ax.set_ylabel("loss")
    _ax.set_title("train_pretrain_vlm.py (projector only)", fontsize=9)
    _fig.tight_layout()
    _plot = mo.as_html(_fig)
    plt.close(_fig)
    mo.vstack([
        mo.md(f"GPU 학습 {len(pre_losses)} step + 평가: **{pre_seconds:.0f}초**. 원본 `Logger` 출력 일부:"),
        mo.md("```text\n" + "\n".join(_shown) + "\n```"),
        _plot,
        result_table([eval_row("학습 전 (무작위 Projector)", base_eval), eval_row("① pretrain_vlm", pre_eval)], selection=None),
        mo.md(f"`out/` 폴더: `{pre_out_sizes}`"),
    ])
    return


@app.cell(hide_code=True)
def _(mo, pre_eval, pre_example):
    mo.md(f"""
    **읽는 법**

    - `Trainable Params: 1.183M`: 6,400만 개 중 Projector만 학습합니다. 그래도 캡션은 학습에 없던 조합까지 대부분 맞힙니다.
    - **질의응답 정답률도 높지만 `It is ...` 형식은 {pre_eval['held_format']:.0%}입니다.** 예를 들어 {pre_example[0]['color']} {pre_example[0]['shape']} 그림에 `What color is the shape?`라고 물으면 `{pre_example[1]}`처럼
      **캡션으로** 답합니다(내용에 정답 색이 들어 있어 "질의응답" 점수가 높게 나온 것). 이 모델이 본 데이터는 캡션뿐이기 때문입니다.
      정렬 단계는 **"보는 법"** 을 가르치고, **"질문에 맞게 답하는 법"** 은 다음 단계의 몫입니다.
    - **언어 능력 유지 값이 학습 전과 같습니다.** 언어 모델 가중치를 하나도 바꾸지 않았고 이미지가 없으면 Projector는 쓰이지 않기 때문입니다.
    - 저장된 `pretrain_vlm_768.pth`가 `llm_768.pth`보다 12MB 큰 이유: Projector(1.2M개 × 2바이트 ≈ 2.4MB)가 더해졌고,
      `model.state_dict()`에는 Embedding과 **공유하는 `lm_head.weight`도 따로 들어가서**(6,400 × 768 × 2바이트 ≈ 9.8MB) 한 번 더 저장됩니다.
      반면 이미지 인코더(9,500만 개)는 **저장하지 않습니다**(`vision_encoder.`로 시작하는 키 제외). 쓸 때마다 원래 파일에서 다시 읽습니다.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. ② 시각 SFT: `train_sft_vlm.py`

    `pretrain_vlm_768.pth`에서 시작해 `freeze_llm=1`(Projector + 언어 모델 첫·마지막 층, 1,590만 개)로 SFT 데이터를 학습합니다.
    batch 4, 2 epochs → 196 step. 원본 학습률 `5e-6`은 수십만 행을 위한 값이라, 392행으로는 **5e-5**를 기본으로 씁니다(위젯으로 원본 값과 비교).

    같은 설정에서 **한 가지씩만 바꾼** 비교 실험 두 개도 합니다.

    | 실험 | 시작 가중치 (`--from_weight`) | 데이터 (`--data_path`) | 확인할 것 |
    | --- | --- | --- | --- |
    | **SFT** (기본) | `pretrain_vlm` | 질의응답 + 캡션 + 일반 대화 | 원본 2단계 |
    | **정렬 생략** | `llm` | 같음 | ① 단계가 꼭 필요한가 (README의 "권장 경로: SFT부터") |
    | **질의응답만** | `pretrain_vlm` | 질의응답 240행 | 캡션·일반 대화를 섞는 이유 |
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    sft_lr = mo.ui.dropdown(options={"5e-6 (원본 기본값)": 5e-6, "5e-5": 5e-5}, value="5e-5", label="SFT learning_rate")
    sft_lr
    return (sft_lr,)


@app.cell
def _(
    evaluate,
    remove_resume_files,
    run_stage,
    script_args,
    sft_lr,
    sft_script,
    time,
    torch,
):
    args_sft = script_args(sft_script, max_seq_len=128, learning_rate=sft_lr.value)
    _t0 = time.time()
    _model, sft_log, sft_losses = run_stage(sft_script, args_sft)
    sft_seconds = time.time() - _t0
    sft_eval = evaluate(_model)
    remove_resume_files("sft_vlm")  # out/sft_vlm_768.pth는 8절에서 사용
    del _model
    torch.cuda.empty_cache()
    return args_sft, sft_eval, sft_losses, sft_seconds


@app.cell
def _(
    args_sft,
    evaluate,
    remove_resume_files,
    run_stage,
    sft_script,
    torch,
    vars_copy,
):
    sft_variants = {}
    for _name, _over in [("정렬 생략 (from llm)", {"from_weight": "llm", "save_weight": "sft_skip"}),
                         ("질의응답만", {"data_path": "../dataset/sft_qa_only.parquet", "save_weight": "sft_qa"})]:
        _model, _log, _losses = run_stage(sft_script, vars_copy(args_sft, **_over))
        sft_variants[_name] = {"eval": evaluate(_model), "losses": _losses}
        remove_resume_files(_over["save_weight"])
        del _model
        torch.cuda.empty_cache()
    return (sft_variants,)


@app.cell
def _(Namespace):
    def vars_copy(args, **overrides):
        """Namespace 복사본에 일부 인자만 바꿔 넣기 (명령줄에서 --인자 하나만 바꾸는 것과 같음)"""
        return Namespace(**{**vars(args), **overrides})

    return (vars_copy,)


@app.cell(hide_code=True)
def _(
    args_sft,
    base_eval,
    eval_row,
    mo,
    plt,
    pre_eval,
    result_table,
    sft_eval,
    sft_losses,
    sft_seconds,
    sft_variants,
):
    _fig, _ax = plt.subplots(figsize=(6.5, 2.8))
    _ax.plot(sft_losses, lw=0.8, label="SFT (from pretrain_vlm)")
    _ax.plot(sft_variants["정렬 생략 (from llm)"]["losses"], lw=0.8, label="SFT, skip stage 1 (from llm)")
    _ax.plot(sft_variants["질의응답만"]["losses"], lw=0.8, label="SFT, QA data only")
    _ax.set_yscale("log"); _ax.set_xlabel("step"); _ax.set_ylabel("loss"); _ax.legend(fontsize=8)
    _fig.tight_layout()
    _plot = mo.as_html(_fig)
    plt.close(_fig)
    _rows = [eval_row("학습 전", base_eval), eval_row("① pretrain_vlm", pre_eval), eval_row("② SFT", sft_eval)] + \
            [eval_row(k, v["eval"]) for k, v in sft_variants.items()]
    _attr = [{"모델": k, **{f"{a} (제외 조합)": f"{r[f'held_{a}']:.0%}" for a in ["color", "shape", "pos", "size"]}}
             for k, r in [("② SFT", sft_eval)] + [(k, v["eval"]) for k, v in sft_variants.items()]]
    mo.vstack([
        mo.md(f"SFT 설정: lr={args_sft.learning_rate}, batch {args_sft.batch_size}, {args_sft.epochs} epochs, "
              f"`freeze_llm={args_sft.freeze_llm}` · SFT 1회 학습 + 평가 **{sft_seconds:.0f}초**"),
        _plot,
        result_table(_rows, selection=None),
        mo.md("**속성별 질의응답 정답률 (제외 조합 12장)**"),
        result_table(_attr, selection=None),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **읽는 법** (기본 lr 5e-5 기준. 평가 그림이 학습 조합 36장·제외 조합 12장뿐이라 한두 장 차이는 우연일 수 있습니다.)

    - **② SFT:** 질문에 `It is red.`처럼 **짧게 답하는 형식**을 배웠고, 캡션 요청에는 여전히 캡션으로 답합니다.
      두 가지 요청을 구분하게 된 것입니다. 언어 모델 층을 풀었는데도 언어 능력 유지 값은 오히려 **학습 전보다 높습니다.**
      SFT에 섞은 일반 대화가 베이스 모델 자신의 답이라 "원래 말투"를 계속 보여 준 효과입니다(평가 질문과 겹치지는 않음).
      그래도 생성된 답은 일부 바뀝니다(아래 표).
    - **정렬 생략:** 같은 데이터·학습량인데 정답률이 크게 떨어지고, 특히 색·크기와 제외 조합 캡션이 약합니다. 손실 곡선도 처음에 훨씬 높습니다.
      무작위 Projector를 392행·작은 학습률로 맞추기에는 부족하기 때문입니다.
      원본 README가 "SFT부터 실행해도 된다"고 하는 것은 실제 SFT 데이터에 **캡션 127만 행이 이미 들어 있어서** ①의 역할을 대신하기 때문입니다.
      **SFT 데이터가 작을수록 값싼 캡션 데이터로 먼저 정렬해 두는 것이 효과적입니다.**
    - **질의응답만:** 질의응답은 잘하지만 **캡션을 요청해도 `It is ...`로 답하고**(캡션 정답률 0%), 언어 능력 유지 값도 더 떨어집니다.
      한 가지 형식만 학습하면 모델이 다른 요청에도 그 형식을 씁니다. 원본 SFT 데이터가 **캡션과 일반 텍스트 대화를 함께 섞는 이유**입니다.
    - **lr을 원본 `5e-6`으로 바꿔 보세요.** 형식(`It is ...`)은 배우지만 일부 속성(특히 제외 조합의 색)은 덜 배웁니다.
      반면 언어 능력 유지 값은 가장 좋습니다. 작은 학습률은 원래 능력을 지키는 대신 더 많은 데이터·step이 필요합니다.
    """)
    return


@app.cell(hide_code=True)
def _(
    TEXT_EVAL_Q,
    base_text_answers,
    mo,
    result_table,
    sft_eval,
    sft_variants,
):
    mo.vstack([
        mo.md("**이미지 없는 일반 질문의 답** (언어 능력이 어떻게 변했는지 직접 보기)"),
        result_table([{"질문": q, "베이스": b, "② SFT": s, "정렬 생략": k, "질의응답만": o}
                     for q, b, s, k, o in zip(TEXT_EVAL_Q, base_text_answers, sft_eval["text_answers"],
                                              sft_variants["정렬 생략 (from llm)"]["eval"]["text_answers"],
                                              sft_variants["질의응답만"]["eval"]["text_answers"])], selection=None),
        mo.md("64M 베이스 모델은 원래도 틀린 답(예: 거미 다리 수)을 냅니다. 여기서 보는 것은 정답 여부가 아니라 **말투와 내용이 얼마나 바뀌었는지**입니다."),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. 체크포인트와 재개 (`--from_resume 1`)

    `train_epoch`는 `save_interval` step마다, 그리고 epoch 끝마다 **두 가지**를 저장합니다.

    | 파일 | 내용 | 용도 |
    | --- | --- | --- |
    | `out/sft_vlm_768.pth` | 가중치만, **fp16**, `vision_encoder.*` 제외 | 추론·다음 단계의 `--from_weight` |
    | `checkpoints/sft_vlm_768_resume.pth` | 가중치(**fp32**, 인코더 포함) + 옵티마이저 + scaler + epoch·step | 학습 이어 하기 |

    재개하면 `SkipBatchSampler`가 이미 학습한 배치를 **건너뛰고**, 같은 epoch의 남은 배치부터 이어 갑니다.
    데이터 순서는 `setup_seed(42 + epoch)`로 정해지므로 다시 실행해도 같습니다.

    **실험:** 5절과 같은 SFT를 `save_interval=40`으로 실행하다가 **70 step에서 강제로 중단**합니다(마지막 저장은 40 step).
    그다음 `from_resume=1`로 다시 실행해, 중단 없이 학습한 5절의 손실과 비교합니다.
    """)
    return


@app.cell
def _(args_sft, file_sizes, run_stage, sft_script, torch, vars_copy, work_dir):
    args_crash = vars_copy(args_sft, save_weight="sft_resume", save_interval=40)
    _model, crash_log, crash_losses = run_stage(sft_script, args_crash, crash_after_steps=70)
    del _model

    _ck = torch.load(work_dir / "checkpoints/sft_resume_768_resume.pth", map_location="cpu", weights_only=False)
    resume_ckpt_info = {
        "키": list(_ck.keys()), "epoch": _ck["epoch"], "step": _ck["step"],
        "model 텐서 수 (인코더 포함)": len(_ck["model"]),
        "model 중 vision_encoder.*": sum(k.startswith("vision_encoder.") for k in _ck["model"]),
        "model dtype": str(next(iter(_ck["model"].values())).dtype),
        "param_groups 안의 매개변수": len(_ck["optimizer"]["param_groups"][0]["params"]),
        "Adam 상태가 있는 매개변수": len(_ck["optimizer"]["state"]),
    }
    del _ck
    crash_sizes = {"checkpoints/": file_sizes("checkpoints"), "out/": file_sizes("out")}
    torch.cuda.empty_cache()
    return args_crash, crash_log, crash_losses, crash_sizes, resume_ckpt_info


@app.cell
def _(
    args_crash,
    crash_losses,
    remove_resume_files,
    run_stage,
    sft_script,
    torch,
    vars_copy,
):
    _ = crash_losses  # 중단 실행이 끝난 뒤에 재개
    args_resume = vars_copy(args_crash, from_resume=1)
    _model, resume_log, resume_losses = run_stage(sft_script, args_resume)
    remove_resume_files("sft_resume")
    del _model
    torch.cuda.empty_cache()
    return resume_log, resume_losses


@app.cell(hide_code=True)
def _(
    args_sft,
    crash_log,
    crash_losses,
    crash_sizes,
    mo,
    plt,
    pre_ckpt_info,
    result_table,
    resume_ckpt_info,
    resume_log,
    resume_losses,
    sft_losses,
):
    _start = len(sft_losses) - len(resume_losses)  # 재개한 전역 step
    _iters = len(sft_losses) // args_sft.epochs
    _fig, _ax = plt.subplots(figsize=(6.5, 2.8))
    _ax.plot(sft_losses, lw=0.8, c="gray", label="no interruption (section 5)")
    _ax.plot(crash_losses, lw=1.2, label="run 1: crashed at step 70")
    _ax.plot(range(_start, _start + len(resume_losses)), resume_losses, lw=1.0, ls="--", label=f"run 2: resumed at step {_start}")
    _ax.axvline(_start, c="k", lw=0.5, ls=":"); _ax.axvline(_iters, c="gray", lw=0.5, ls=":")
    _ax.set_yscale("log"); _ax.set_xlabel("global step"); _ax.set_ylabel("loss"); _ax.legend(fontsize=8)
    _fig.tight_layout()
    _plot = mo.as_html(_fig)
    plt.close(_fig)

    _same_epoch = [(a, b) for a, b in zip(sft_losses[_start:_iters], resume_losses[: _iters - _start])]
    _next_epoch = [(a, b) for a, b in zip(sft_losses[_iters:], resume_losses[_iters - _start:])]
    _diff = lambda pairs: max(abs(a - b) for a, b in pairs) if pairs else 0.0
    _mean = lambda xs: sum(xs) / len(xs) if xs else float("nan")
    _skip_line = next((l for l in resume_log.splitlines() if "건너뛰고" in l), "")
    mo.vstack([
        _plot,
        mo.md(f"""
    - 중단 로그 마지막 줄: `{crash_log.strip().splitlines()[-1]}`
    - 재개 로그: `{_skip_line}` → 40 step까지의 저장 상태에서 시작하므로 **41~70 step은 다시 학습**합니다.
    - step별 손실 차이 (최대): 같은 epoch의 남은 step **{_diff(_same_epoch):.3f}**, 다음 epoch **{_diff(_next_epoch):.3f}**
    - 평균 손실 (중단 없음 / 재개): 같은 epoch의 남은 step **{_mean([a for a, _ in _same_epoch]):.3f} / {_mean([b for _, b in _same_epoch]):.3f}**,
      다음 epoch **{_mean([a for a, _ in _next_epoch]):.3f} / {_mean([b for _, b in _next_epoch]):.3f}**
        """),
        mo.md("**재개용 체크포인트 들여다보기**"),
        result_table([{"항목": k, "SFT (`sft_resume_768_resume.pth`)": str(v)} for k, v in resume_ckpt_info.items()], selection=None),
        mo.md(f"""
    - pretrain의 옵티마이저(`filter(requires_grad)`): param_groups {pre_ckpt_info['param_groups 안의 매개변수']}개, Adam 상태 {pre_ckpt_info['Adam 상태가 있는 매개변수']}개
    - 파일 크기: `{crash_sizes}`
        """),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **읽는 법**

    - 재개한 학습은 중단 없는 학습과 **같은 범위의 손실**을 따라가고, epoch 평균 손실도 비슷합니다.
      모델·옵티마이저(Adam의 1·2차 모멘트)·학습률 위치가 모두 복원되고, 같은 순서의 배치를 받기 때문입니다.
    - 하지만 **step별 손실은 재개 직후부터 다릅니다.** `VLMDataset`은 샘플을 꺼낼 때마다 Python `random`으로 system 메시지 추가(20%)와
      빈 think 삭제(80%)를 정합니다. `SkipBatchSampler`는 건너뛴 배치의 샘플을 **꺼내지 않으므로** 난수가 그만큼 덜 소비되고,
      재개 뒤의 무작위 가공이 원래 학습과 달라집니다. 다음 epoch는 `setup_seed(43)`으로 난수를 다시 맞추지만,
      모델이 이미 조금 달라져 있어 작은 차이가 남습니다.
    - 재개용 파일은 이미지 인코더까지 fp32로 담고 옵티마이저 상태도 있어 추론용 파일보다 **5배 이상 큽니다.** 학습이 끝나면 지워도 됩니다.
    - SFT 옵티마이저는 `model.parameters()` 전체를 받았으므로 param_groups에는 인코더까지 모든 매개변수가 들어 있지만,
      Adam 상태는 실제로 기울기를 받은 28개(Projector 6 + 첫·마지막 층 각 11)에만 생깁니다. pretrain은 처음부터 학습할 6개만 넘겼습니다.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7. 저장된 가중치로 대화하기 (`eval_vlm.py` 방식)

    학습이 끝난 모델은 파일로만 남습니다. `eval_vlm.py`의 `init_model`과 같은 방법으로 `out/sft_vlm_768.pth`를 새 모델에 넣습니다.

    ```python
    model = MiniMindVLM(VLMConfig(hidden_size=768, num_hidden_layers=8, use_moe=False),
                        vision_model_path="./model/siglip2-base-p32-256-ve")      # 인코더는 원래 파일에서
    state_dict = torch.load("./out/sft_vlm_768.pth", map_location=device)        # Projector + 언어 모델
    model.load_state_dict({k: v for k, v in state_dict.items() if 'mask' not in k}, strict=False)
    model = model.eval().half().to(device)                                        # GPU에서는 fp16으로 추론
    ```

    `eval_vlm.py`는 `do_sample=True`(temperature 0.7)로 답을 뽑지만, 여기서는 결과를 비교하기 쉽도록 탐욕적 생성을 씁니다.
    **학습에서 뺀 조합**의 그림으로, 같은 그림에 대해 여러 번 이어서 묻습니다(여러 턴 대화).
    """)
    return


@app.cell
def _(MiniMindVLM, VLMConfig, args_sft, device, sft_eval, torch, work_dir):
    _ = (args_sft, sft_eval)  # 5절 SFT가 저장을 마친 뒤에 실행
    loaded_vlm = MiniMindVLM(VLMConfig(hidden_size=768, num_hidden_layers=8, use_moe=False),
                             vision_model_path=str(work_dir / "model/siglip2-base-p32-256-ve"))
    _state = torch.load(work_dir / "out/sft_vlm_768.pth", map_location=device)
    _missing, _unexpected = loaded_vlm.load_state_dict({k: v for k, v in _state.items() if "mask" not in k}, strict=False)
    loaded_vlm = loaded_vlm.eval().half().to(device)
    assert not _unexpected
    assert not [k for k in _missing if not k.startswith("vision_encoder.")]  # 인코더만 빠짐 (lm_head도 파일에 있음)
    assert "lm_head.weight" in _state and all(v.dtype == torch.float16 for v in _state.values()) and not any(k.startswith("vision_encoder.") for k in _state)
    return (loaded_vlm,)


@app.cell
def _(HELD_OUT, TEST_ITEMS, generate_reply, loaded_vlm):
    CHAT_QUESTIONS = ["What color is the shape?", "What shape is it?", "Is the shape big or small?",
                      "Is the shape on the left or the right?", "Describe the image."]

    def multi_turn(model, img, questions):
        """한 그림에 대해 이전 문답을 이어 붙이며 묻기. 그림은 첫 질문에만 붙습니다."""
        messages, replies = [], []
        for i, q in enumerate(questions):
            messages.append({"role": "user", "content": ("<image>\n" if i == 0 else "") + q})
            reply = generate_reply(model, img, messages, 20)
            messages.append({"role": "assistant", "content": reply})
            replies.append(reply)
        return replies

    chat_items = [(a, img) for a, img in TEST_ITEMS if (a["color"], a["shape"]) in HELD_OUT][::2]
    chat_results = [multi_turn(loaded_vlm, img, CHAT_QUESTIONS) for _, img in chat_items]
    return CHAT_QUESTIONS, chat_items, chat_results


@app.cell(hide_code=True)
def _(CHAT_QUESTIONS, chat_items, chat_results, mo, png_bytes):
    mo.vstack([
        mo.hstack([
            mo.vstack([mo.image(png_bytes(_img), width=120),
                       mo.md(f"**정답:** {_a['size']} {_a['color']} {_a['shape']}, {_a['pos']}"),
                       *[mo.md(f"*{_q}*<br>→ {_r}") for _q, _r in zip(CHAT_QUESTIONS, _replies)]])
            for (_a, _img), _replies in zip(chat_items, chat_results)
        ], widths="equal", align="start", wrap=True),
        mo.md(r"""
    - 파일에서 다시 불러온 fp16 모델도 **첫 두 질문**(색, 모양)에는 대부분 맞게 답합니다. 두 번째 질문에서는 그림 토큰이 대화 맨 앞에만 있고,
      모델은 어텐션으로 앞쪽 그림 토큰을 다시 참고해 답합니다. 학습 데이터에 **2턴 대화**가 있었기 때문에 가능합니다.
    - **세 번째 질문부터는 그림과 상관없이 같은 답을 반복합니다**(실행 결과: 크기·위치·캡션 질문에 모두 "It is small."). 한 번만 물으면
      크기를 대부분 맞히는 모델인데도 그렇습니다. 학습 데이터가 **최대 2턴**이라 3턴 이상의 대화는 모델에게 처음 보는 형식입니다.
      **모델은 학습 데이터가 보여 준 대화 길이·형식 안에서만 믿을 수 있습니다.** 원본 SFT 데이터에 긴 여러 턴 대화가 들어 있는 이유입니다.
    """),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 8. 실제 규모로 실행하려면

    데이터(`pretrain_i2t.parquet`, `sft_i2t.parquet`)와 이미지 인코더 준비는 `projects/minimind-v/README_ko.md`를 따릅니다.
    이 노트북의 베이스 언어 모델로 시작하려면 1절처럼 `out/llm_768.pth`를 만듭니다.

    ```python
    # projects/minimind-v/ 에서 실행
    import torch
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    sd = load_file(hf_hub_download("caspar/MiniMind-Dense-64M-Mixed-SFT-v1.0", "model.safetensors", revision="v1.0"))
    torch.save({k: v.half() for k, v in sd.items()}, "out/llm_768.pth")
    ```

    ```bash
    cd projects/minimind-v/trainer
    python train_pretrain_vlm.py --epochs 2 --from_weight llm          # ① → out/pretrain_vlm_768.pth
    python train_sft_vlm.py --epochs 2 --from_weight pretrain_vlm      # ② → out/sft_vlm_768.pth
    python train_sft_vlm.py --epochs 4 --from_resume 1                 # 중단된 ②를 이어서 (epochs를 늘려 더 학습할 수도 있음)
    torchrun --nproc_per_node 2 train_sft_vlm.py                       # GPU 여러 개 (DDP)
    cd .. && python eval_vlm.py --weight sft_vlm                       # dataset/eval_images/ 의 사진으로 대화
    ```

    | 인자 | 원본 ① / ② | 이 노트북 ① / ② |
    | --- | --- | --- |
    | 데이터 | 127만 / 290만 행 | 1,440 / 392 행 |
    | `--learning_rate` | 4e-4 / 5e-6 | 4e-4 / 5e-5 (위젯) |
    | `--batch_size` | 16 / 4 | 16 / 4 |
    | `--epochs` | 2 / 2 | 2 / 2 |
    | `--max_seq_len` | 450 / 768 | 128 / 128 |
    | `--num_workers` | 2 | 0 |
    | `--save_interval` | 1000 | epoch 끝만 (6절은 40) |

    ## 9. 정리

    | 항목 | 핵심 |
    | --- | --- |
    | 두 스크립트 | 학습 루프는 같고 인자 기본값(데이터, 시작 가중치, `freeze_llm`, 학습률)만 다름 |
    | ① 정렬 | 캡션 데이터, Projector만, 큰 학습률 → "보는 법". 언어 능력은 그대로. 질문에도 캡션으로 답함 |
    | ② SFT | 질의응답 + 캡션 + 일반 대화, + 첫·마지막 층, 작은 학습률 → "요청에 맞게 답하는 법" |
    | 데이터 섞기 | 한 형식만 학습하면 다른 요청에도 그 형식을 씀 → 캡션·일반 대화(검은 8×8 그림)를 섞음 |
    | ① 생략 | SFT 데이터에 캡션이 충분하면 가능. 데이터가 작으면 정렬 단계가 큰 차이를 만듦 |
    | 저장 | `out/`: fp16 가중치(인코더 제외) / `checkpoints/`: fp32 전체 + 옵티마이저 + 위치 |
    | 재개 | 마지막 저장 지점부터, `SkipBatchSampler`로 같은 순서 유지. 무작위 가공 때문에 완전히 같지는 않음 |

    ## 10. 직접 해 볼 과제

    각 과제를 하기 전에 **결과를 먼저 예측**해 적고, 바꿔 실행한 뒤 비교하세요.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({
        "과제 1. 기울기 누적 (5절)": mo.md("""
    `args_sft`에 `batch_size=2, accumulation_steps=2`를 주면 실제로 한 번에 갱신하는 샘플 수(4)는 같습니다. 결과가 같을까요?
    로그의 step 수와 `lr` 값은 어떻게 달라질까요?

    *해설:* 갱신 한 번에 쓰는 샘플 수는 같지만, `get_lr`은 **step(배치) 기준**이라 학습률이 더 잘게 나뉘어 줄어들고 step 수는 두 배가 됩니다.
    작은 배치의 bf16 연산 순서도 달라 결과가 완전히 같지는 않습니다. GPU 메모리가 부족할 때 쓰는 방법입니다.
    """),
        "과제 2. freeze_llm=0 (5절)": mo.md("""
    SFT를 `freeze_llm=0`(이미지 인코더 빼고 전부)으로 실행하면 학습 매개변수 수, 정답률, 언어 능력 유지 값은 어떻게 될까요?

    *해설:* 6,500만 개 가까이 학습합니다. 작은 데이터에서는 그림 정답률이 크게 오르지 않는 반면 언어 능력이 더 많이 바뀌기 쉽습니다.
    원본이 첫 층(그림 토큰과 글을 섞는 입구)과 마지막 층(답의 형식)만 푸는 이유를 생각해 보세요.
    """),
        "과제 3. 일반 대화를 빼기 (2절)": mo.md("""
    `sft_rows`에서 `text_rows`만 빼고(질의응답 + 캡션) 학습하면 언어 능력 유지 값은 어떻게 될까요?

    *해설:* 5절의 "질의응답만"은 캡션과 일반 대화를 함께 뺐습니다. 하나씩 빼 보면 각각의 역할(캡션: 형식 보존, 일반 대화: 언어 능력 보존)을 나눠 볼 수 있습니다.
    """),
        "과제 4. 원본 학습률로 더 오래 (5절)": mo.md("""
    lr `5e-6`에서 `epochs`를 10으로 늘리면 5e-5, 2 epochs와 비슷한 정답률에 도달할까요? 그때 언어 능력 유지 값은 어느 쪽이 좋을까요?

    *해설:* 작은 학습률로 오래 학습하면 같은 지점에 더 천천히 다가갑니다. 원본이 작은 학습률을 쓸 수 있는 것은 데이터가 수백만 행이라 step이 충분하기 때문입니다.
    """),
        "과제 5. 3턴 대화 학습하기 (2절, 7절)": mo.md("""
    `qa_rows`에서 `rng.choice([1, 2])`를 `rng.choice([1, 2, 3, 4])`로 바꾸고 다시 학습하면, 7절의 세 번째·네 번째 질문에 맞게 답할까요?

    *해설:* 여러 턴 대화를 학습하면 앞쪽 그림 토큰을 멀리서도 참고하는 법을 배웁니다. 단, 대화가 길어지면 `max_seq_len=128`을 넘어 잘릴 수 있으니
    정답 토큰이 잘리지 않는지(2-1절처럼 `labels`를 확인) 함께 보세요.
    """),
        "과제 6. 마지막 저장 직후에 중단하기 (6절)": mo.md("""
    `crash_after_steps`를 40으로 바꾸면(저장 직후 중단) 재개한 곡선과 중단 없는 곡선의 차이는 어떻게 될까요? 98(epoch 끝)이면요?

    *해설:* 다시 학습하는 구간이 없어지지만, 무작위 가공 차이는 그대로 남습니다. epoch 끝에서 재개하면 다음 epoch가 `setup_seed(42 + epoch)`로
    난수를 다시 맞추므로 손실이 완전히 같아질 수 있습니다.
    """),
    })
    return


@app.cell(hide_code=True)
def _(mo):
    cleanup_button = mo.ui.run_button(label="실습이 끝나면: 임시 작업 폴더 지우기 (약 1.6GB)")
    cleanup_button
    return (cleanup_button,)


@app.cell(hide_code=True)
def _(cleanup_button, mo, shutil, work_dir):
    mo.stop(not cleanup_button.value, mo.md(f"*작업 폴더 `{work_dir}`는 노트북을 닫아도 남습니다. 다 쓴 뒤 버튼을 누르세요.*"))
    shutil.rmtree(work_dir, ignore_errors=True)
    mo.md(f"`{work_dir}`를 지웠습니다. 다시 실습하려면 노트북을 처음부터 다시 실행하세요.")
    return


if __name__ == "__main__":
    app.run()
