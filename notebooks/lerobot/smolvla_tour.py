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
    # SmolVLA 해부 실습

    **SmolVLA**는 Hugging Face가 공개한 약 **4.5억(0.45B) 파라미터** VLA(Vision-Language-Action) 모델입니다.
    카메라 이미지, 자연어 지시, 로봇 관절 상태를 받아 **앞으로 실행할 행동 50개(행동 청크)** 를 한 번에 만듭니다.
    이 노트북은 [smolvla.md](smolvla.md)의 설명을 순서대로 **실제 가중치와 원본 코드로 확인**합니다.

    - 공개 VLM [`SmolVLM2-500M-Video-Instruct`](https://huggingface.co/HuggingFaceTB/SmolVLM2-500M-Video-Instruct)를 `transformers`로 불러와 구조를 봅니다.
    - LeRobot 원본 `smolvlm_with_expert.py`, `modeling_smolvla.py`의 핵심을 옮겨 와 VLM + 행동 전문가(Action Expert)를 조립하고,
      공개 체크포인트 [`lerobot/smolvla_base`](https://huggingface.co/lerobot/smolvla_base) 가중치를 그대로 불러옵니다.
    - 입력에서 행동 청크까지 Pixel Shuffle, 레이어 절단, 어텐션 마스크, Flow Matching, KV 캐시를 확인합니다.

    **선수 지식:** PyTorch 텐서·역전파, Transformer의 Self/Cross Attention, RoPE와 KV 캐시의 개념.

    | 절 | 내용 | 자원 |
    | --- | --- | --- |
    | 1 | 입력·출력 형식과 최소 예제 | CPU, 즉시 |
    | 2 | SmolVLM2 불러오기, 구조 확인 | **버튼**, 처음 약 3 GB 다운로드 |
    | 3 | 비전 토큰 압축(Pixel Shuffle) | 즉시 |
    | 4 | 레이어 절단: 앞쪽 16층만 쓰는 이유 | 설명, 실험은 보충 노트북 |
    | 5 | SmolVLA 조립과 `smolvla_base` 가중치 불러오기 | **버튼**, 로컬 가중치, GPU 권장 |
    | 6 | 2D 블록 어텐션 마스크 | 즉시 |
    | 7 | Flow Matching 원리 (작은 2D 예제) | **버튼**, CPU 토이 학습 |
    | 8 | KV 캐시 추론: 동치성과 속도 | **버튼**, 수 초 |
    | 9 | 정리와 보충 실습 안내 | — |

    **실행:** `notebooks/lerobot`에서 `uv sync --locked` 후 `uv run marimo edit smolvla_tour.py`.
    버튼이 있는 셀(다운로드·학습·긴 실험)만 직접 눌러야 실행됩니다. marimo는 변수 의존성에 따라 셀을 다시 실행하므로
    2절 버튼을 다시 누르면 모델과 의존 실험 결과를 새로 계산합니다.

    > **원본 코드**: [huggingface/lerobot `src/lerobot/policies/smolvla`](https://github.com/huggingface/lerobot/tree/e6ffde60312365c5452c46189a26872483943154/src/lerobot/policies/smolvla)
    > 및 같은 revision의 `policies/common/flow_matching.py`, `policies/common/vla_utils.py`, `inference/execution.py`
    > (revision `e6ffde60312365c5452c46189a26872483943154`). 원본은 Apache License 2.0,
    > Copyright 2025–2026 The HuggingFace Inc. team입니다. 이 노트북은 원본 코드를 옮겨 와 축소·수정했으며
    > `lerobot` 패키지는 설치하지 않습니다.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 학습 목표와 실행 경로

    이 실습을 마치면 입력에서 행동 청크까지의 shape, 마스크에 따른 정보 흐름,
    Flow Matching의 시간 방향, 캐시 재사용 조건을 설명하고 구현의 한계를 구분할 수 있어야 합니다.

    - **가중치 없이 필수 읽기**: 1~8절의 설명과 코드, 3절 재배열 예제, 6절 작은 마스크.
    - **가중치를 쓰는 선택 실험**: 2절 다운로드 → 3절 커넥터 확인, 5절 조립, 6절 실제 마스크, 8절 캐시 비교.
    - **원리 확인용 선택 학습**: 7절 작은 2D Flow Matching 모델.
    - VQA·프로브, 정책 학습·평가, 비동기 큐 실험은 아래 9절의 별도 노트북에서 진행합니다.

    각 긴 실행 셀은 자체 버튼으로 기다립니다. 입력·지시문·횟수를 바꾼 뒤 해당 버튼을 다시 누르세요.
    2절 모델 로드를 다시 실행하면 의존하는 정책과 실험 결과가 무효화될 수 있습니다.
    CUDA가 없거나 가중치가 없으면 필수 읽기와 작은 CPU 예제로 진행하고, GPU 학습 성공을 확인한 것으로 해석하지 마세요.

    **기호**: $B$=배치, $H$=행동 청크 길이(50), $A$=실제 행동 차원(2), $D$=패딩 후 차원(32).
    환경 행동 $H$스텝과 청크 생성의 Euler 적분 10회는 서로 다른 시간축입니다.

    **먼저 답해 보기**: 상태가 같고 지시문만 바뀌면 캐시를 계속 쓸 수 있나요?
    기대 답은 **새 관측으로 다시 계산해야 한다**입니다. 언어도 접두부 조건의 일부입니다.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.mermaid("""
    flowchart LR
        I["카메라 이미지 512x512"] --> SG["SigLIP 비전 인코더<br/>1024 패치"]
        SG --> PS["Pixel Shuffle + 선형 투영<br/>64 토큰"]
        L["지시문 토큰"] --> V["SmolVLM2 텍스트 층<br/>앞쪽 16개만"]
        PS --> V
        S["관절 상태"] --> SP["state_proj<br/>1 토큰"] --> V
        V -->|"층별 K/V 캐시"| E["행동 전문가 16층<br/>(폭 0.75배)"]
        N["노이즈 행동 + 시간 t"] --> E
        E --> U["속도 v"]
        U --> O["Euler 10스텝, t=1 에서 0<br/>행동 청크 50 x 32"]
    """)
    return


@app.cell
def _():
    import copy
    import math
    import time

    import matplotlib.font_manager as font_manager
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    import torch.nn.functional as F
    from torch import nn

    # 그림의 한글 표시용 글꼴: 설치된 것 중 첫 번째를 사용합니다.
    _installed = {f.name for f in font_manager.fontManager.ttflist}
    for _name in ["Noto Sans CJK KR", "NanumGothic", "Malgun Gothic", "AppleGothic", "Noto Sans Mono CJK KR"]:
        if _name in _installed:
            plt.rcParams["font.family"] = _name
            break
    plt.rcParams["axes.unicode_minus"] = False
    return F, copy, math, nn, np, plt, time, torch


@app.cell
def _(torch):
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ON_GPU = DEVICE.type == "cuda"

    VLM_ID = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
    VLM_REV = "4cde9d208e19519c36917aa170fffc7e381aa0ad"
    SMOLVLA_ID = "lerobot/smolvla_base"
    SMOLVLA_REV = "5e8d12a6e2975b0e5e5fce7c8caf47c371d257b6"

    # 정확한 수치 비교(8절)를 위해 TF32 행렬곱을 끕니다.
    torch.backends.cuda.matmul.allow_tf32 = False
    print(f"device={DEVICE}" + ("" if ON_GPU else "  (CPU: 모델 로드와 추론은 느릴 수 있습니다)"))
    return DEVICE, ON_GPU, SMOLVLA_ID, SMOLVLA_REV, VLM_ID, VLM_REV


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. 입력·출력 형식과 최소 예제

    SmolVLA는 **이미지·지시문·상태**를 받아 앞으로 실행할 **행동 청크**를 만듭니다.
    아래 고정 이미지 한 장과 두 지시문을 6절 마스크·8절 캐시 실험에 사용합니다.
    같은 이미지를 배치의 두 항목에 넣고 서로 다른 길이의 문장을 써서 텍스트 패딩도 확인합니다.

    | 입력·출력 | 이 예제 | 모델에 넣거나 얻는 shape |
    | --- | --- | --- |
    | 이미지 | 빨간 점·파란 점이 있는 고정 RGB 이미지 한 장 | 이미지마다 `[B, 3, 512, 512]`로 전처리 |
    | 지시문 | `go to the red dot`, `move to the blue one` | 토큰과 패딩 마스크 `[B, L]` |
    | 상태 | 두 좌표 `(x, y)` | 0을 채워 `[B, 32]` |
    | 행동 청크 | 앞으로 50스텝의 행동 | `[B, 50, 32]` |

    여기서 배치 $B=2$이며 상태·행동의 실제 차원은 앞 2칸입니다. 나머지는 모델 폭에 맞춘 패딩입니다.
    좌표는 `[-1, 1]` 범위의 예시이며 실제 로봇의 관절 단위·정규화와는 구분합니다.
    이 절에서는 정답 궤적이나 성공률을 정의하지 않습니다.
    장면 생성·시연·환경 이동·성공 판정은 [학습·평가 노트북](smolvla_training.py)에서 다룹니다.
    """)
    return


@app.cell
def _(torch):
    CHUNK_SIZE = 50
    MAX_DIM = 32

    # 다운로드 없이 만드는 고정 RGB 이미지 한 장: 빨간 점과 파란 점.
    _axis = torch.linspace(-1, 1, 128)
    _ys, _xs = torch.meshgrid(_axis, _axis, indexing="ij")
    demo_image = torch.full((1, 3, 128, 128), 0.85)
    for _x, _y, _color in [(-0.5, -0.4, (0.9, 0.15, 0.15)), (0.5, 0.5, (0.15, 0.3, 0.9))]:
        _inside = (_xs - _x).square() + (_ys - _y).square() < 0.09**2
        demo_image[0, :, _inside] = torch.tensor(_color)[:, None]
    demo_texts = ["go to the red dot", "move to the blue one"]
    demo_states = torch.tensor([[-0.7, 0.6], [0.2, -0.6]])
    return CHUNK_SIZE, MAX_DIM, demo_image, demo_states, demo_texts


@app.cell
def _(demo_image, plt):
    _fig, _ax = plt.subplots(figsize=(3, 3))
    _ax.imshow(demo_image[0].permute(1, 2, 0))
    _ax.set_xticks([]), _ax.set_yticks([])
    _ax.set_title("구조 확인용 고정 입력")
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. 시각-언어 백본: SmolVLM2

    SmolVLA의 백본은 공개 VLM `SmolVLM2-500M-Video-Instruct`입니다. 구조는 세 부분입니다.

    1. **비전 인코더**(SigLIP 계열 ViT, 12층, 폭 768): 512×512 이미지를 16×16 패치 **1024개**로 나눠 임베딩합니다.
    2. **커넥터**: Pixel Shuffle로 토큰 수를 1/16(**64개**)로 줄이고 선형층으로 텍스트 폭 960에 맞춥니다.
    3. **텍스트 디코더**(Llama 계열, 32층, 폭 960, Query 헤드 15·KV 헤드 5, head_dim 64).

    아래 버튼은 두 가중치를 불러옵니다. 처음이면 Hub에서 내려받아 `~/.cache/huggingface/`에 저장합니다.

    - `SmolVLM2-500M-Video-Instruct` (약 2 GB): 2~3절에서 32층 전체 VLM으로 사용합니다. GPU에서는 bf16으로 올립니다.
    - `lerobot/smolvla_base` (약 0.9 GB): 5절 이후 SmolVLA 정책 가중치입니다.
    """)
    return


@app.cell
def _(mo):
    load_btn = mo.ui.run_button(label="SmolVLM2 + smolvla_base 불러오기")
    load_btn
    return (load_btn,)


@app.cell
def _(
    DEVICE,
    ON_GPU,
    SMOLVLA_ID,
    SMOLVLA_REV,
    VLM_ID,
    VLM_REV,
    load_btn,
    mo,
    time,
    torch,
):
    mo.stop(not load_btn.value, mo.md("*버튼을 누르면 모델을 불러옵니다. 2절 이후 모델을 쓰는 셀은 그때까지 멈춰 있습니다.*"))
    from huggingface_hub import hf_hub_download
    from transformers import AutoModelForImageTextToText, AutoTokenizer

    _t0 = time.time()
    with mo.status.spinner("SmolVLM2와 smolvla_base 불러오는 중..."):
        # SmolVLA 원본은 AutoProcessor에서 tokenizer만 씁니다. 이미지는 직접 전처리하므로 AutoTokenizer로 충분합니다.
        tok = AutoTokenizer.from_pretrained(VLM_ID, revision=VLM_REV)
        vlm_full = AutoModelForImageTextToText.from_pretrained(
            VLM_ID, revision=VLM_REV, dtype=torch.bfloat16 if ON_GPU else torch.float32
        ).to(DEVICE)
        vlm_full.eval()
        smolvla_ckpt_path = hf_hub_download(SMOLVLA_ID, "model.safetensors", revision=SMOLVLA_REV)
    print(f"불러오기 {time.time() - _t0:.0f}초")
    print(vlm_full.config.text_config.num_hidden_layers, "개 텍스트 층,", type(vlm_full).__name__)
    return AutoModelForImageTextToText, smolvla_ckpt_path, tok, vlm_full


@app.cell
def _(mo, vlm_full):
    _v, _t = vlm_full.config.vision_config, vlm_full.config.text_config
    _n_patch = (_v.image_size // _v.patch_size) ** 2
    _sf = vlm_full.config.scale_factor
    mo.md(f"""
    | 부분 | 설정 | 값 |
    | --- | --- | --- |
    | 비전 인코더 | 입력 크기 / 패치 / 층 / 폭 | {_v.image_size}px / {_v.patch_size}px / {_v.num_hidden_layers} / {_v.hidden_size} |
    | | 패치 수 | ({_v.image_size}/{_v.patch_size})² = **{_n_patch}** |
    | 커넥터 | Pixel Shuffle 배율 | {_sf} → 토큰 {_n_patch} / {_sf}² = **{_n_patch // _sf**2}**, 폭 {_v.hidden_size}×{_sf}² = {_v.hidden_size * _sf**2} → {_t.hidden_size} |
    | 텍스트 디코더 | 층 / 폭 / Q 헤드 / KV 헤드 / head_dim | {_t.num_hidden_layers} / {_t.hidden_size} / {_t.num_attention_heads} / {_t.num_key_value_heads} / {_t.head_dim} |
    | | RoPE theta (VLM 원래 값) | {_t.rope_parameters["rope_theta"] if hasattr(_t, "rope_parameters") else _t.rope_theta} |
    """)
    return


@app.cell
def image_preprocessing(F):
    def resize_with_pad(img, height, width, *, pad_value=0.0):
        """원본 vla_utils.resize_with_pad: 비율을 유지해 줄이고 **왼쪽·위쪽**을 채웁니다 (b, c, h, w)."""
        cur_h, cur_w = img.shape[2:]
        if cur_h == height and cur_w == width:
            return img
        ratio = max(cur_w / width, cur_h / height)
        new_h, new_w = int(cur_h / ratio), int(cur_w / ratio)
        resized = F.interpolate(img, size=(new_h, new_w), mode="bilinear", align_corners=False)
        return F.pad(resized, (max(0, width - new_w), 0, max(0, height - new_h), 0), value=pad_value)

    return (resize_with_pad,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. 비전 토큰 압축: Pixel Shuffle

    이미지 한 장의 패치 1024개를 그대로 쓰면 카메라 3대만 써도 접두부가 3000토큰을 넘습니다. 커넥터의 `pixel_shuffle`은
    **인접한 4×4 패치 16개의 특징을 채널 방향으로 이어 붙여** 토큰 1개로 만듭니다. 정보는 버리지 않고 모양만 바꾼 뒤,
    선형층 하나(12288 → 960)가 압축을 담당합니다.

    | 단계 | 텐서 shape |
    | --- | --- |
    | 원래 패치 | `(B, 1024, 768)` |
    | 격자로 보기 | `(B, 32, 32, 768)` |
    | 인접 4×4를 채널에 묶기 | `(B, 8, 8, 12288)` |
    | 시퀀스로 보기 | `(B, 64, 12288)` |
    | 선형 압축 | `(B, 64, 960)` |

    아래는 `transformers`의 `SmolVLMConnector.pixel_shuffle` 코드와 같은 연산입니다. 패치 번호를 값으로 넣어 어떤 패치가 묶이는지 확인합니다.
    """)
    return


@app.function
def pixel_shuffle(x, scale_factor):
    """transformers SmolVLMConnector.pixel_shuffle과 같은 연산 (주석 추가)."""
    bsz, seq, embed_dim = x.size()
    height = width = int(seq**0.5)
    x = x.view(bsz, height, width, embed_dim)
    # 가로로 scale_factor개씩 묶어 채널로 이어 붙임
    x = x.view(bsz, height, int(width / scale_factor), embed_dim * scale_factor)
    x = x.permute(0, 2, 1, 3)
    # 세로로 scale_factor개씩 묶어 다시 채널로 이어 붙임
    x = x.reshape(bsz, int(width / scale_factor), int(height / scale_factor), embed_dim * (scale_factor**2))
    x = x.permute(0, 2, 1, 3)
    return x.reshape(bsz, int(seq / (scale_factor**2)), embed_dim * (scale_factor**2))


@app.cell
def _(np, plt, torch):
    # 8×8 패치(64개), 채널 1개, 값 = 패치 번호. 배율 4 → 토큰 4개, 각 토큰이 16개 패치를 담습니다.
    _x = torch.arange(64, dtype=torch.float32).view(1, 64, 1)
    _y = pixel_shuffle(_x, 4)
    print("출력 shape:", tuple(_y.shape))
    for _k in range(_y.shape[1]):
        print(f"토큰 {_k}: 패치 번호 {_y[0, _k].int().tolist()}")
    _owner = np.zeros(64, dtype=int)
    for _k in range(_y.shape[1]):
        _owner[_y[0, _k].long().numpy()] = _k
    _fig, _ax = plt.subplots(figsize=(3, 3))
    _ax.imshow(_owner.reshape(8, 8), cmap="Pastel1")
    for _i in range(64):
        _ax.text(_i % 8, _i // 8, _i, ha="center", va="center", fontsize=7)
    _ax.set_title("8×8 패치 → 토큰 4개 (색 = 묶이는 토큰)", fontsize=9)
    _ax.set_xticks([]), _ax.set_yticks([])
    _fig
    return


@app.cell
def _(DEVICE, torch, vlm_full):
    # 실제 커넥터와 같은지 확인: 비전 인코더 출력 [1, 1024, 768]에 적용
    with torch.no_grad():
        _pix = torch.randn(1, 3, 512, 512, device=DEVICE, dtype=vlm_full.dtype).clamp(-1, 1)
        _patches = vlm_full.model.vision_model(pixel_values=_pix).last_hidden_state
        _ours = vlm_full.model.connector.modality_projection(pixel_shuffle(_patches, vlm_full.config.scale_factor))
        _theirs = vlm_full.model.connector(_patches)
    print("비전 인코더 출력:", tuple(_patches.shape), "→ 커넥터 출력:", tuple(_theirs.shape))
    print("직접 구현과 커넥터의 최대 차이:", (_ours - _theirs).abs().max().item())
    print(f"텍스트 층 Self-Attention 점수 행렬 크기 비: (1024/64)² = {(1024 / 64) ** 2:.0f}배 감소")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. 레이어 절단: 앞쪽 16층 사용

    SmolVLA는 32층 VLM의 앞쪽 16층을 사용합니다. 5절의 `layers[:num_vlm_layers]`에서 확인합니다.
    텍스트 층 수를 줄이면 해당 연산과 KV 캐시 메모리를 줄일 수 있습니다.
    비전 인코더는 그대로이므로 전체 접두부 계산 시간이 정확히 절반이 되는 것은 아닙니다.

    행동 전문가는 VLM의 생성 답변이 아니라 각 층의 접두부 K/V를 참조합니다.
    층별 좌표 선형 프로브와 연산 시간 측정은 [VLM 보충 실습](smolvla_vlm_experiments.py)으로 옮겼습니다.
    좌표를 선형 해독할 수 있다는 결과만으로 최적 절단 깊이나 실제 행동 성공을 입증할 수는 없습니다.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. SmolVLA 조립: VLM + 행동 전문가

    이제 원본 `smolvlm_with_expert.py`의 `SmolVLMWithExpertModel`을 옮겨 옵니다. 핵심은 세 가지입니다.

    1. **행동 전문가 = 폭이 0.75배인 같은 구조의 Llama 층 16개.** `text_config`를 복사해 `hidden_size=960×0.75=720`으로 바꾸고
       `AutoModel.from_config`로 만듭니다. 헤드 수(15/5)와 `head_dim=64`는 VLM과 같으므로 **두 모델의 Q·K·V를 한 어텐션에서 섞을 수 있습니다.**
    2. **층마다 VLM 층 하나와 전문가 층 하나가 짝을 이룹니다.** 각자 자기 가중치로 Q·K·V·O와 MLP를 계산하고 어텐션 단계에서만 만납니다
       (π0의 Mixture-of-Transformers 구조).
    3. **짝수 층은 Self-Attention, 홀수 층은 Cross-Attention**(`attention_mode="cross_attn"`, `self_attn_every_n_layers=2`).
       - Self 층(`forward_attn_layer`): 접두부와 행동 토큰의 Q·K·V를 **시퀀스 축으로 이어 붙여** 한 번에 어텐션합니다. 마스크가 누가 누구를 보는지 정합니다.
       - Cross 층(`forward_cross_attn_layer`): 행동 토큰의 Query가 **그 층 VLM의 K/V(RoPE 적용 후, 5헤드×64=320차원)** 를 봅니다.
         이때 전문가의 `k_proj`, `v_proj`는 입력이 720이 아니라 **320**이 되도록 교체됩니다(VLM K/V를 한 번 더 투영).

    원본에서 바꾼 점: 가중치 로딩·프로세서·`num_expert_layers` 축소 옵션·이미지 특수 토큰·RTC·`torch.compile` 분기를 뺐습니다.
    원본은 bf16 VLM과 fp32 투영층을 섞어 쓰는데, 원본 코드는 MLP 앞에서 dtype을 맞추지 않으므로 여기서는 **모든 파라미터를 fp32**로 둡니다.
    """)
    return


@app.cell
def expert_definitions(F, copy, nn, torch):
    from transformers import AutoModel, DynamicCache

    def apply_rope(x, positions, max_wavelength=10_000):
        """원본 그대로: x [B, L, H, D]에 위치 positions [B, L]의 RoPE를 적용.
        주의: SmolVLM2 원래 rope_theta는 100000이지만 SmolVLA는 10000으로 학습했습니다."""
        d_half = x.shape[-1] // 2
        dtype = x.dtype
        x = x.to(torch.float32)
        freq_exponents = (2.0 / x.shape[-1]) * torch.arange(d_half, dtype=torch.float32, device=x.device)
        timescale = max_wavelength**freq_exponents
        radians = positions[..., None].to(torch.float32) / timescale[None, None, :]
        radians = radians[..., None, :]
        sin, cos = torch.sin(radians), torch.cos(radians)
        x1, x2 = x.split(d_half, dim=-1)
        res = torch.empty_like(x)
        res[..., :d_half] = x1 * cos - x2 * sin
        res[..., d_half:] = x2 * cos + x1 * sin
        return res.to(dtype)

    def get_intermediate_size(hidden_dim, ffn_dim_multiplier=4, multiple_of=256):
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        return multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

    class SmolVLMWithExpertModel(nn.Module):
        """원본 smolvlm_with_expert.py 축소판. train_expert_only=True, freeze_vision_encoder=True 고정."""

        def __init__(self, vlm, num_vlm_layers=16, self_attn_every_n_layers=2, expert_width_multiplier=0.75):
            super().__init__()
            self.vlm = vlm
            # 레이어 절단: 앞쪽 num_vlm_layers개만 남깁니다.
            self.vlm.model.text_model.layers = self.vlm.model.text_model.layers[:num_vlm_layers]
            self.num_vlm_layers = len(self.vlm.model.text_model.layers)
            config = vlm.config
            self.config = config

            # 행동 전문가: 같은 Llama 구조, 폭만 0.75배, 층 수는 VLM과 같게
            expert_config = copy.deepcopy(config.text_config)
            expert_config.hidden_size = int(config.text_config.hidden_size * expert_width_multiplier)
            expert_config.intermediate_size = get_intermediate_size(expert_config.hidden_size)
            expert_config.num_hidden_layers = self.num_vlm_layers
            self.lm_expert = AutoModel.from_config(expert_config)

            # Cross-Attention 층: K/V 투영의 입력을 VLM의 K/V 차원(KV 헤드×head_dim=320)으로 교체
            self.self_attn_every_n_layers = self_attn_every_n_layers
            vlm_kv_dim = config.text_config.num_key_value_heads * config.text_config.head_dim
            expert_kv_dim = expert_config.num_key_value_heads * expert_config.head_dim
            for layer_idx, layer in enumerate(self.lm_expert.layers):
                if self.is_self_attn_layer(layer_idx):
                    continue
                layer.self_attn.k_proj = nn.Linear(vlm_kv_dim, expert_kv_dim, bias=expert_config.attention_bias)
                layer.self_attn.v_proj = nn.Linear(vlm_kv_dim, expert_kv_dim, bias=expert_config.attention_bias)
            self.lm_expert.embed_tokens = None  # 전문가는 토큰 임베딩을 쓰지 않습니다.

            self.num_attention_heads = config.text_config.num_attention_heads
            self.num_key_value_heads = config.text_config.num_key_value_heads
            self.head_dim = config.text_config.head_dim
            self.expert_hidden_size = expert_config.hidden_size

            # train_expert_only=True: VLM 전체 고정
            self.vlm.eval()
            for p in self.vlm.parameters():
                p.requires_grad = False

        def is_self_attn_layer(self, layer_idx):
            return layer_idx % self.self_attn_every_n_layers == 0

        def train(self, mode=True):
            super().train(mode)
            self.vlm.eval()  # 고정된 VLM은 항상 eval 모드
            return self

        def embed_image(self, image):
            m = self.vlm.model
            hidden = m.vision_model(pixel_values=image.to(m.vision_model.dtype)).last_hidden_state
            return m.connector(hidden)  # Pixel Shuffle + 투영 → [B, 64, 960]

        def embed_language_tokens(self, tokens):
            return self.vlm.model.text_model.get_input_embeddings()(tokens)

        def eager_attention_forward(self, attention_mask, batch_size, query_states, key_states, value_states):
            """GQA: KV 헤드 5개를 Query 헤드 15개에 맞게 3번씩 복제한 뒤 마스크 어텐션."""
            n_heads, n_kv, head_dim = self.num_attention_heads, self.num_key_value_heads, self.head_dim
            groups = n_heads // n_kv
            seq = key_states.shape[1]
            key_states = key_states[:, :, :, None, :].expand(batch_size, seq, n_kv, groups, head_dim)
            key_states = key_states.reshape(batch_size, seq, n_heads, head_dim)
            value_states = value_states[:, :, :, None, :].expand(batch_size, seq, n_kv, groups, head_dim)
            value_states = value_states.reshape(batch_size, seq, n_heads, head_dim)
            q = query_states.to(torch.float32).transpose(1, 2)
            k = key_states.to(torch.float32).transpose(1, 2)
            att = torch.matmul(q, k.transpose(2, 3)) * head_dim**-0.5
            att = torch.where(attention_mask[:, None, :, :], att, torch.finfo(att.dtype).min)
            probs = F.softmax(att, dim=-1).to(value_states.dtype)
            out = torch.matmul(probs, value_states.permute(0, 2, 1, 3)).permute(0, 2, 1, 3)
            return out.reshape(batch_size, -1, n_heads * head_dim)

        def forward_attn_layer(self, model_layers, inputs_embeds, layer_idx, position_ids, attention_mask, batch_size, use_cache, past_key_values):
            """Self-Attention 층: [접두부; 행동] 의 Q·K·V를 이어 붙여 어텐션 한 번."""
            query_states, key_states, value_states = [], [], []
            for i, hidden_states in enumerate(inputs_embeds):
                layer = model_layers[i][layer_idx]
                if hidden_states is None:
                    continue
                hidden_states = layer.input_layernorm(hidden_states)
                shape = (*hidden_states.shape[:-1], -1, layer.self_attn.head_dim)
                hidden_states = hidden_states.to(dtype=layer.self_attn.q_proj.weight.dtype)
                query_states.append(layer.self_attn.q_proj(hidden_states).view(shape))  # 각자 자기 가중치로 투영
                key_states.append(layer.self_attn.k_proj(hidden_states).view(shape))
                value_states.append(layer.self_attn.v_proj(hidden_states).view(shape))
            query_states = torch.cat(query_states, dim=1)  # 시퀀스 축으로 결합
            key_states = torch.cat(key_states, dim=1)
            value_states = torch.cat(value_states, dim=1)
            seq_len = query_states.shape[1]
            if seq_len < position_ids.shape[1]:
                position_ids, attention_mask = position_ids[:, :seq_len], attention_mask[:, :seq_len, :seq_len]
            query_states = apply_rope(query_states, position_ids)
            key_states = apply_rope(key_states, position_ids)
            if use_cache:
                # 접두부 단계: K/V를 캐시에 저장. 디노이즈 단계: 캐시된 접두부 K/V 뒤에 행동 K/V를 붙여 돌려받음.
                key_states, value_states = past_key_values.update(key_states.transpose(1, 2), value_states.transpose(1, 2), layer_idx)
                key_states, value_states = key_states.transpose(1, 2), value_states.transpose(1, 2)
            att_output = self.eager_attention_forward(attention_mask, batch_size, query_states, key_states, value_states)
            return [att_output], past_key_values

        def forward_cross_attn_layer(self, model_layers, inputs_embeds, layer_idx, position_ids, attention_mask, batch_size, use_cache, past_key_values):
            """Cross-Attention 층: 접두부는 자기들끼리 어텐션, 행동 토큰은 그 층 VLM K/V만 참조."""
            att_outputs = []
            if len(inputs_embeds) == 2 and not past_key_values:
                # 학습(캐시 없음): 접두부 경로를 이 자리에서 계산
                seq_len = inputs_embeds[0].shape[1]
                position_id, expert_position_id = position_ids[:, :seq_len], position_ids[:, seq_len:]
                layer = model_layers[0][layer_idx]
                hidden_states = layer.input_layernorm(inputs_embeds[0])
                shape = (*hidden_states.shape[:-1], -1, layer.self_attn.head_dim)
                hidden_states = hidden_states.to(dtype=layer.self_attn.q_proj.weight.dtype)
                query_states = apply_rope(layer.self_attn.q_proj(hidden_states).view(shape), position_id)
                key_states = apply_rope(layer.self_attn.k_proj(hidden_states).view(shape), position_id)
                value_states = layer.self_attn.v_proj(hidden_states).view(shape)
                att_outputs.append(
                    self.eager_attention_forward(attention_mask[:, :seq_len, :seq_len], batch_size, query_states, key_states, value_states)
                )
            else:
                expert_position_id = position_ids
            if use_cache and past_key_values is not None:
                # 추론: 캐시에서 이 층 VLM K/V를 읽기만 합니다(이어 붙이지 않음).
                key_states = past_key_values.layers[layer_idx].keys.transpose(1, 2)
                value_states = past_key_values.layers[layer_idx].values.transpose(1, 2)

            expert_layer = model_layers[1][layer_idx]
            expert_hidden = expert_layer.input_layernorm(inputs_embeds[1])
            expert_shape = (*expert_hidden.shape[:-1], -1, expert_layer.self_attn.head_dim)
            expert_hidden = expert_hidden.to(dtype=expert_layer.self_attn.q_proj.weight.dtype)
            expert_query = expert_layer.self_attn.q_proj(expert_hidden).view(expert_shape)
            # VLM K/V [B, L, 5, 64] → [B, L, 320] → 전문가 k_proj/v_proj(320→320)
            _k = key_states.to(dtype=expert_layer.self_attn.k_proj.weight.dtype).reshape(*key_states.shape[:2], -1)
            expert_key = expert_layer.self_attn.k_proj(_k).view(*_k.shape[:-1], -1, expert_layer.self_attn.head_dim)
            _v = value_states.to(dtype=expert_layer.self_attn.v_proj.weight.dtype).reshape(*value_states.shape[:2], -1)
            expert_value = expert_layer.self_attn.v_proj(_v).view(*_v.shape[:-1], -1, expert_layer.self_attn.head_dim)
            # 행동 토큰의 위치는 0부터 다시 셉니다(Query에만 RoPE, K는 VLM에서 이미 RoPE 적용).
            expert_position_id = expert_position_id - torch.min(expert_position_id, dim=1, keepdim=True).values
            expert_attention_mask = attention_mask[:, -inputs_embeds[1].shape[1] :, : expert_key.shape[1]]
            expert_query = apply_rope(expert_query, expert_position_id)
            att_outputs.append(self.eager_attention_forward(expert_attention_mask, batch_size, expert_query, expert_key, expert_value))
            return att_outputs, past_key_values

        def forward(self, attention_mask, position_ids, past_key_values=None, inputs_embeds=None, use_cache=False):
            """inputs_embeds = [접두부 임베딩 또는 None, 행동 임베딩 또는 None]"""
            models = [self.vlm.model.text_model, self.lm_expert]
            model_layers = [list(models[0].layers), list(models[1].layers)]
            batch_size = next(h for h in inputs_embeds if h is not None).shape[0]
            fill_kv_cache = use_cache and past_key_values is None
            if fill_kv_cache:
                past_key_values = DynamicCache()
            for layer_idx in range(self.num_vlm_layers):
                # 접두부 캐시 채우기 단계는 모든 층이 Self-Attention 경로로 K/V를 저장합니다.
                if fill_kv_cache or self.is_self_attn_layer(layer_idx):
                    att_outputs, past_key_values = self.forward_attn_layer(
                        model_layers, inputs_embeds, layer_idx, position_ids, attention_mask, batch_size, use_cache, past_key_values
                    )
                else:
                    att_outputs, past_key_values = self.forward_cross_attn_layer(
                        model_layers, inputs_embeds, layer_idx, position_ids, attention_mask, batch_size, use_cache, past_key_values
                    )
                # 어텐션 출력을 다시 나눠 각자의 o_proj → 잔차 → MLP → 잔차
                outputs_embeds, start = [], 0
                for i, hidden_states in enumerate(inputs_embeds):
                    if hidden_states is None:
                        outputs_embeds.append(None)
                        continue
                    layer = model_layers[i][layer_idx]
                    att_output = att_outputs[i] if i < len(att_outputs) else att_outputs[0]
                    end = start + hidden_states.shape[1]
                    out = layer.self_attn.o_proj(att_output[:, start:end].to(layer.self_attn.o_proj.weight.dtype))
                    out = out + hidden_states
                    out = out + layer.mlp(layer.post_attention_layernorm(out))
                    outputs_embeds.append(out)
                    start = end if len(att_outputs) == 1 else 0
                inputs_embeds = outputs_embeds
            outputs = [models[i].norm(h) if h is not None else None for i, h in enumerate(inputs_embeds)]
            return outputs, past_key_values

    return (SmolVLMWithExpertModel,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    다음은 원본 `modeling_smolvla.py`의 `VLAFlowMatching`과 `vla_utils.py`의 보조 함수입니다.

    - `embed_prefix`: 이미지 토큰(×√960) + 지시문 토큰(×√960) + **상태 토큰 1개**(`state_proj`: 32 → 960)를 이어 붙입니다.
    - `embed_suffix`: 노이즈 행동(32 → 720)과 시간 `t`의 사인파 임베딩(720)을 이어 붙여 MLP 두 층(`action_time_mlp_in/out`)에 통과시킵니다.
    - `forward`(학습): 접두부와 행동을 한 번에 넣어 속도를 예측하고 MSE 손실을 냅니다.
    - `sample_actions`(추론): 접두부를 **한 번** 계산해 KV 캐시를 만들고, `denoise_step`을 10번 반복합니다.
    """)
    return


@app.cell
def flow_definitions(F, SmolVLMWithExpertModel, math, nn, torch):
    def make_att_2d_masks(pad_masks, att_masks):
        """원본 그대로 (big_vision 유래). att_masks 누적합이 Query 이하인 Key만 볼 수 있습니다."""
        cumsum = torch.cumsum(att_masks, dim=1)
        att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
        pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
        return att_2d_masks & pad_2d_masks

    def create_sinusoidal_pos_embedding(time, dimension, min_period, max_period, device):
        fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=torch.float64, device=device)
        period = min_period * (max_period / min_period) ** fraction
        sin_input = time[..., None] * (1.0 / period * 2 * math.pi)
        return torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=-1)

    def sample_time(bsize, device):
        """t ~ Beta(1.5, 1) × 0.999 + 0.001 : t=1(노이즈 쪽)에 가까운 시간을 더 자주 뽑습니다."""
        beta = torch.distributions.Beta(torch.tensor(1.5), torch.tensor(1.0))
        return (beta.sample((bsize,)) * 0.999 + 0.001).to(device=device, dtype=torch.float32)

    class VLAFlowMatching(nn.Module):
        """원본 modeling_smolvla.VLAFlowMatching 축소판 (smolvla_base 설정값 사용)."""

        def __init__(self, vlm, chunk_size=50, max_state_dim=32, max_action_dim=32, num_steps=10):
            super().__init__()
            self.chunk_size, self.num_steps = chunk_size, num_steps
            self.max_state_dim, self.max_action_dim = max_state_dim, max_action_dim
            self.min_period, self.max_period = 4e-3, 4.0
            self.vlm_with_expert = SmolVLMWithExpertModel(vlm)
            vlm_dim = vlm.config.text_config.hidden_size
            expert_dim = self.vlm_with_expert.expert_hidden_size
            self.state_proj = nn.Linear(max_state_dim, vlm_dim)
            self.action_in_proj = nn.Linear(max_action_dim, expert_dim)
            self.action_out_proj = nn.Linear(expert_dim, max_action_dim)
            self.action_time_mlp_in = nn.Linear(expert_dim * 2, expert_dim)
            self.action_time_mlp_out = nn.Linear(expert_dim, expert_dim)

        def embed_prefix(self, images, img_masks, lang_tokens, lang_masks, state):
            embs, pad_masks, att_masks = [], [], []
            for img, img_mask in zip(images, img_masks):
                img_emb = self.vlm_with_expert.embed_image(img)
                img_emb = img_emb * torch.tensor(img_emb.shape[-1] ** 0.5, dtype=img_emb.dtype, device=img_emb.device)
                bsize, num_img_embs = img_emb.shape[:2]
                embs.append(img_emb)
                pad_masks.append(img_mask[:, None].expand(bsize, num_img_embs))
                att_masks += [0] * num_img_embs
            lang_emb = self.vlm_with_expert.embed_language_tokens(lang_tokens)
            lang_emb = lang_emb * math.sqrt(lang_emb.shape[-1])
            embs.append(lang_emb)
            pad_masks.append(lang_masks)
            att_masks += [0] * lang_emb.shape[1]
            state_emb = self.state_proj(state)[:, None, :]
            embs.append(state_emb)
            bsize = state_emb.shape[0]
            pad_masks.append(torch.ones(bsize, 1, dtype=torch.bool, device=state_emb.device))
            att_masks += [1]  # 상태 토큰은 새 블록을 시작합니다.
            embs = torch.cat(embs, dim=1)
            pad_masks = torch.cat(pad_masks, dim=1)
            att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)[None, :].expand(bsize, -1)
            return embs, pad_masks, att_masks

        def embed_suffix(self, noisy_actions, timestep):
            action_emb = self.action_in_proj(noisy_actions)
            bsize = action_emb.shape[0]
            time_emb = create_sinusoidal_pos_embedding(
                timestep, self.vlm_with_expert.expert_hidden_size, self.min_period, self.max_period, action_emb.device
            ).type(action_emb.dtype)
            time_emb = time_emb[:, None, :].expand_as(action_emb)
            action_time_emb = self.action_time_mlp_in(torch.cat([action_emb, time_emb], dim=2))
            action_time_emb = self.action_time_mlp_out(F.silu(action_time_emb))
            pad_masks = torch.ones(bsize, action_time_emb.shape[1], dtype=torch.bool, device=action_emb.device)
            att_masks = torch.ones(bsize, self.chunk_size, dtype=action_time_emb.dtype, device=action_emb.device)  # [1]*chunk: 인과적
            return action_time_emb, pad_masks, att_masks

        def forward(self, images, img_masks, lang_tokens, lang_masks, state, actions, noise=None, time=None):
            """학습: 스텝·차원별 손실 [B, chunk, 32]"""
            if noise is None:
                noise = torch.randn_like(actions)
            if time is None:
                time = sample_time(actions.shape[0], actions.device)
            t = time[:, None, None]
            x_t = t * noise + (1 - t) * actions  # NOISE_AT_ONE: t=1이 노이즈, t=0이 행동
            u_t = noise - actions  # 목표 속도
            prefix_embs, prefix_pad, prefix_att = self.embed_prefix(images, img_masks, lang_tokens, lang_masks, state)
            suffix_embs, suffix_pad, suffix_att = self.embed_suffix(x_t, time)
            pad_masks = torch.cat([prefix_pad, suffix_pad], dim=1)
            att_masks = torch.cat([prefix_att, suffix_att], dim=1)
            att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
            position_ids = torch.cumsum(pad_masks, dim=1) - 1
            (_, suffix_out), _ = self.vlm_with_expert.forward(
                att_2d_masks, position_ids, None, [prefix_embs, suffix_embs], use_cache=False
            )
            v_t = self.action_out_proj(suffix_out[:, -self.chunk_size :].to(torch.float32))
            return F.mse_loss(u_t, v_t, reduction="none")

        @torch.no_grad()
        def sample_actions(self, images, img_masks, lang_tokens, lang_masks, state, noise=None):
            """추론: 접두부 1회 + Euler 10스텝 (t: 1 → 0)"""
            bsize = state.shape[0]
            if noise is None:
                noise = torch.randn(bsize, self.chunk_size, self.max_action_dim, device=state.device)
            prefix_embs, prefix_pad, prefix_att = self.embed_prefix(images, img_masks, lang_tokens, lang_masks, state)
            prefix_att_2d = make_att_2d_masks(prefix_pad, prefix_att)
            prefix_pos = torch.cumsum(prefix_pad, dim=1) - 1
            _, past_key_values = self.vlm_with_expert.forward(
                prefix_att_2d, prefix_pos, None, [prefix_embs, None], use_cache=True
            )
            dt = -1.0 / self.num_steps
            x_t = noise
            for step in range(self.num_steps):
                time = torch.full((bsize,), 1.0 + step * dt, dtype=torch.float32, device=state.device)
                v_t = self.denoise_step(prefix_pad, past_key_values, x_t, time)
                x_t = x_t + dt * v_t
            return x_t

        def denoise_step(self, prefix_pad_masks, past_key_values, x_t, timestep):
            suffix_embs, suffix_pad, suffix_att = self.embed_suffix(x_t, timestep)
            suffix_len = suffix_pad.shape[1]
            bsize, prefix_len = prefix_pad_masks.shape
            prefix_pad_2d = prefix_pad_masks[:, None, :].expand(bsize, suffix_len, prefix_len)
            suffix_att_2d = make_att_2d_masks(suffix_pad, suffix_att)
            full_att_2d = torch.cat([prefix_pad_2d, suffix_att_2d], dim=2)  # [B, 행동, 접두부+행동]
            position_ids = torch.sum(prefix_pad_masks, dim=-1)[:, None] + torch.cumsum(suffix_pad, dim=1) - 1
            outputs, _ = self.vlm_with_expert.forward(
                full_att_2d, position_ids, past_key_values, [None, suffix_embs], use_cache=True
            )
            # Self 층이 캐시 뒤에 붙인 행동 K/V를 잘라내 접두부 길이로 되돌립니다(다음 스텝 재사용).
            for layer in past_key_values.layers:
                layer.crop(prefix_len - layer.get_seq_length())
            return self.action_out_proj(outputs[1][:, -self.chunk_size :].to(torch.float32))

    return VLAFlowMatching, make_att_2d_masks


@app.cell
def _(mo):
    assemble_btn = mo.ui.run_button(label='SmolVLA 조립 (로컬 가중치 사용)')
    assemble_btn
    return (assemble_btn,)


@app.cell
def _(
    AutoModelForImageTextToText,
    CHUNK_SIZE,
    DEVICE,
    VLAFlowMatching,
    VLM_ID,
    VLM_REV,
    assemble_btn,
    mo,
    smolvla_ckpt_path,
    torch,
    vlm_full,
):
    mo.stop(not assemble_btn.value, mo.md('2절에서 가중치를 받은 뒤 조립 버튼을 누르세요.'))
    from safetensors.torch import load_file

    with mo.status.spinner("SmolVLA 조립 및 smolvla_base 가중치 불러오는 중..."):
        # 2절의 32층 VLM은 그대로 두고, 정책용 VLM을 fp32로 새로 불러옵니다(로컬 캐시 사용).
        _vlm = AutoModelForImageTextToText.from_pretrained(VLM_ID, revision=VLM_REV, dtype=torch.float32, local_files_only=True)
        policy = VLAFlowMatching(_vlm, chunk_size=CHUNK_SIZE)
        _state = {k.removeprefix("model."): v for k, v in load_file(smolvla_ckpt_path).items()}
        _result = policy.load_state_dict(_state, strict=True)
        policy.to(DEVICE).eval()

    # smolvla_base 안의 VLM 가중치가 공개 SmolVLM2와 같은지 비교 (앞쪽 16층 + 비전 + 커넥터)
    _ref = vlm_full.state_dict()
    _n_same = _n_diff = 0
    for _k, _v in policy.vlm_with_expert.vlm.state_dict().items():
        if _k in _ref:
            if torch.equal(_v.to(_ref[_k].dtype), _ref[_k].to(_v.device)):
                _n_same += 1
            else:
                _n_diff += 1
    print(f"load_state_dict(strict=True): {_result}")
    print(f"SmolVLM2와 같은 VLM 텐서 {_n_same}개, 다른 텐서 {_n_diff}개 (2절 VLM이 bf16이면 같은 bf16 값끼리 비교)")
    return (policy,)


@app.cell
def _(mo, policy):
    def _count(module):
        return sum(p.numel() for p in module.parameters()) / 1e6

    _vwe = policy.vlm_with_expert
    _vm = _vwe.vlm.model
    _rows = [
        ("비전 인코더 (SigLIP, 12층)", _count(_vm.vision_model), "고정"),
        ("커넥터 (Pixel Shuffle + 투영)", _count(_vm.connector), "고정"),
        ("텍스트 층 16개 + 임베딩", _count(_vm.text_model), "고정"),
        ("lm_head (행동 생성에는 미사용)", _count(_vwe.vlm.lm_head), "고정"),
        ("행동 전문가 16층 (폭 720)", _count(_vwe.lm_expert), "학습"),
        (
            "state_proj, action_in/out_proj, action_time_mlp",
            sum(_count(m) for m in [policy.state_proj, policy.action_in_proj, policy.action_out_proj, policy.action_time_mlp_in, policy.action_time_mlp_out]),
            "학습",
        ),
    ]
    _total = _count(policy)
    _train = sum(p.numel() for p in policy.parameters() if p.requires_grad) / 1e6
    _table = "\n".join(f"| {n} | {c:.1f}M | {s} |" for n, c, s in _rows)

    _layer_rows = []
    for _i, _layer in enumerate(_vwe.lm_expert.layers):
        _a = _layer.self_attn
        _kind = "Self" if _vwe.is_self_attn_layer(_i) else "**Cross**"
        _sees = "접두부 + 행동(인과)" if _vwe.is_self_attn_layer(_i) else "그 층 VLM K/V만"
        _layer_rows.append(
            f"| {_i} | {_kind} | {_a.q_proj.in_features}→{_a.q_proj.out_features} | {_a.k_proj.in_features}→{_a.k_proj.out_features} | {_a.o_proj.in_features}→{_a.o_proj.out_features} | {_sees} |"
        )
    mo.md(f"""
    | 부분 | 파라미터 | 파인튜닝 대상 |
    | --- | --- | --- |
    {_table}
    | **합계** | **{_total:.0f}M** | 학습 {_train:.0f}M |

    행동 전문가 층별 투영 크기 (VLM 층 i의 짝):

    | 층 | 어텐션 | q_proj | k_proj, v_proj | o_proj | 행동 토큰이 보는 것 |
    | --- | --- | --- | --- | --- | --- |
    {chr(10).join(_layer_rows[:4])}
    | … | 짝수 Self / 홀수 Cross 반복 | | | | |
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    - q_proj 출력이 VLM과 같은 **960 = 15헤드 × 64**이므로 Self 층에서 접두부 Query와 행동 Query를 그대로 이어 붙일 수 있습니다.
      o_proj는 반대로 960 → 720으로 되돌립니다. 폭이 달라도 Q·K·V의 헤드 수와 `head_dim`을 포함한 투영 형상이 맞으면 어텐션을 공유할 수 있습니다.
    - Cross 층의 k/v_proj 입력이 320인 이유: 입력이 행동 은닉 상태(720)가 아니라 **VLM이 그 층에서 만든 K/V(5헤드 × 64)** 이기 때문입니다.
    - 합계가 공개 설명의 "0.45B"와 맞는지, 학습 대상이 전체의 약 1/4(행동 전문가 + 투영층)인지 확인하세요.

    ## 6. 2D 블록 어텐션 마스크

    접두부와 행동 토큰을 한 시퀀스로 이어 붙이면 "누가 누구를 보는가"는 마스크 하나로 정해집니다.
    원본은 1차원 `att_masks`(새 블록 시작 위치에 1)와 누적합으로 2D 마스크를 만듭니다. **Key의 누적합 ≤ Query의 누적합**이면 볼 수 있습니다.

    SmolVLA에서 실제 값은 다음과 같습니다.

    | 토큰 | `att_masks` | 결과 |
    | --- | --- | --- |
    | 이미지 64개, 지시문 | 0 | 접두부 안에서 **양방향**. 단, 뒤의 상태·행동은 보지 못함 |
    | 상태 1개 | **1** | 이미지·지시문을 보지만, 이미지·지시문은 상태를 보지 못함 |
    | 행동 50개 | 1, 1, …, 1 | 접두부 전체 + **이전 행동까지만**(인과적) |

    접두부가 행동을 보지 못하므로 접두부 K/V는 행동(노이즈)과 무관합니다. 그래서 **추론 중 한 번 계산한 접두부 K/V를 10스텝 내내 재사용**할 수 있습니다.
    """)
    return


@app.cell
def _(make_att_2d_masks, torch):
    # 작은 예: 접두부 3개, 상태 1개, 행동 3개, 지시문 패딩 1개
    _att = torch.tensor([[0, 0, 0, 1, 1, 1, 1]])
    _pad = torch.tensor([[True, True, False, True, True, True, True]])  # 3번째 토큰은 패딩
    _m = make_att_2d_masks(_pad, _att).int()[0]
    print("att_masks   :", _att[0].tolist(), "→ 누적합", torch.cumsum(_att, 1)[0].tolist())
    print("행=Query, 열=Key (1이면 볼 수 있음)")
    for _i, _row in enumerate(_m.tolist()):
        print(f"  토큰 {_i}: {_row}")
    return


@app.cell
def _(
    CHUNK_SIZE,
    DEVICE,
    MAX_DIM,
    demo_image,
    demo_states,
    demo_texts,
    resize_with_pad,
    tok,
    torch,
):
    def make_policy_inputs(images, texts, agent_pos):
        """원본 전처리와 같은 순서: 이미지 512 패딩 → [-1,1], 지시문 끝에 '\\n', 상태 32차원 0 채움."""
        pixels = resize_with_pad(images.to(DEVICE), 512, 512, pad_value=0) * 2 - 1
        toks = tok([t + "\n" for t in texts], padding="longest", padding_side="right", return_tensors="pt").to(DEVICE)
        state = torch.zeros(images.shape[0], MAX_DIM, device=DEVICE)
        state[:, : agent_pos.shape[1]] = agent_pos.to(DEVICE)
        img_masks = [torch.ones(images.shape[0], dtype=torch.bool, device=DEVICE)]
        return [pixels], img_masks, toks.input_ids, toks.attention_mask.bool(), state

    demo_inputs = make_policy_inputs(demo_image.expand(len(demo_texts), -1, -1, -1), demo_texts, demo_states)
    print("이미지:", tuple(demo_inputs[0][0].shape), " 지시문 토큰:", tuple(demo_inputs[2].shape),
          " 지시문 패딩 마스크:", demo_inputs[3].int().tolist())
    print("상태:", tuple(demo_inputs[4].shape), " 행동 청크:", (2, CHUNK_SIZE, MAX_DIM))
    return (demo_inputs,)


@app.cell
def _(
    CHUNK_SIZE,
    DEVICE,
    demo_inputs,
    make_att_2d_masks,
    np,
    plt,
    policy,
    torch,
):
    with torch.no_grad():
        _pe, _pp, _pa = policy.embed_prefix(*demo_inputs)
        _se, _sp, _sa = policy.embed_suffix(torch.zeros(2, CHUNK_SIZE, 32, device=DEVICE), torch.ones(2, device=DEVICE))
    _pad = torch.cat([_pp, _sp], 1)
    _att = torch.cat([_pa.float(), _sa], 1)
    _mask = make_att_2d_masks(_pad, _att)[0].cpu().numpy()
    _n_img, _n_lang = 64, demo_inputs[2].shape[1]
    _bounds = np.cumsum([_n_img, _n_lang, 1])
    _fig, _ax = plt.subplots(figsize=(5.5, 5.5))
    _ax.imshow(_mask, cmap="Greys", interpolation="nearest")
    for _b in _bounds:
        _ax.axhline(_b - 0.5, color="tab:red", lw=0.8)
        _ax.axvline(_b - 0.5, color="tab:red", lw=0.8)
    _ticks = [_n_img / 2, _n_img + _n_lang / 2, _bounds[1], _bounds[2] + CHUNK_SIZE / 2]
    _ax.set_xticks(_ticks, ["이미지 64", "지시문", "상태", f"행동 {CHUNK_SIZE}"], fontsize=8)
    _ax.set_yticks(_ticks, ["이미지 64", "지시문", "상태", f"행동 {CHUNK_SIZE}"], fontsize=8)
    _ax.set_xlabel("Key (참조 대상)")
    _ax.set_ylabel("Query")
    _ax.set_title(f"첫 번째 샘플의 2D 마스크 (검정 = 볼 수 있음), 전체 {_mask.shape[0]} 토큰", fontsize=9)
    _fig.tight_layout()
    print("위치 id (cumsum(pad)-1), 접두부 끝:", (torch.cumsum(_pad, 1) - 1)[0, _bounds[2] - 3 : _bounds[2] + 3].tolist())
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    확인할 점:

    - 배치의 토큰 길이가 다르면 짧은 지시문 끝에 **패딩**이 들어갑니다. 실제 마스크와 토큰 수를 확인하세요. 패딩 행·열은 전부 0이며 위치 id도 증가하지 않습니다.
    - 행동 블록이 **아래 삼각**(인과적)입니다. π0은 행동끼리 양방향(`[1] + [0]*(chunk-1)`)인데 SmolVLA는 인과적입니다.
      [smolvla.md](smolvla.md)의 블록 마스크 표와 대조해 보세요.
    - 위 마스크는 짝수(Self) 층에 그대로 쓰입니다. 홀수(Cross) 층에서 행동 토큰은 오른쪽 행동 블록 없이 **접두부 열만** 봅니다
      (`attention_mask[:, -행동수:, :접두부길이]`).

    ## 7. Flow Matching: 노이즈를 행동으로 옮기는 속도장

    행동 전문가가 예측하는 것은 행동 자체가 아니라 **속도**입니다. SmolVLA(π0 계열, `FlowConvention.NOISE_AT_ONE`)의 정의는 다음과 같습니다.

    $$x_t = t\,\epsilon + (1-t)\,a, \qquad u_t = \frac{dx_t}{dt} = \epsilon - a, \qquad \mathcal{L} = \lVert v_\theta(x_t, t, \text{obs}) - u_t \rVert^2$$

    - $t=1$이 순수 노이즈 $\epsilon \sim \mathcal N(0, I)$, $t=0$이 행동 $a$입니다. 추론은 $t: 1 \to 0$으로 $dt=-0.1$씩 10번 Euler 적분합니다.
    - 반대 방향($x_t=(1-t)\epsilon + t\,a$, $t: 0\to1$)으로 정의할 수도 있습니다. 시간 축을 뒤집고 목표 속도 부호까지 맞추면 같은 방법이지만,
      **체크포인트는 학습한 방향으로만** 써야 합니다. 원본은 이 실수를 막으려고 `FlowConvention`을 학습·추론 함수 양쪽에 명시합니다.
    - 학습 시간 $t$는 균등분포가 아니라 $\text{Beta}(1.5, 1)$에서 뽑아 노이즈가 많은 구간을 더 자주 학습합니다.

    왜 행동을 직접 회귀(MSE)하지 않을까요? 정답이 여러 개인 상황을 2D로 만들어 봅니다. "아무 점으로나 가라"는 시연이 반은 왼쪽, 반은 오른쪽 점으로
    간다면, MSE 회귀의 최적해는 **두 점의 평균(아무 데도 아닌 곳)** 입니다. Flow Matching은 노이즈로 여러 해를 표현할 수 있습니다. 유한한 학습에서 두 봉우리가 모두 복원되는지는 아래 실험으로 확인합니다.
    """)
    return


@app.cell
def _(mo):
    toy_train_btn = mo.ui.run_button(label='2D Flow Matching 토이 학습 (3000 step)')
    toy_train_btn
    return (toy_train_btn,)


@app.cell
def _(math, mo, nn, torch, toy_train_btn):
    mo.stop(not toy_train_btn.value, mo.md('토이 학습 버튼을 누를 때만 3000 step 학습합니다.'))
    torch.manual_seed(0)
    FM_CENTERS = torch.tensor([[-0.6, -0.4], [0.6, -0.4]])

    def sample_bimodal(n):
        idx = torch.randint(0, 2, (n,))
        return FM_CENTERS[idx] + 0.07 * torch.randn(n, 2)

    class ToyVelocity(nn.Module):
        """v_θ(x_t, t): 2D 점과 시간의 사인파 임베딩을 받아 속도를 냅니다."""

        def __init__(self, width=128, t_dim=32):
            super().__init__()
            self.t_dim = t_dim
            self.net = nn.Sequential(nn.Linear(2 + t_dim, width), nn.SiLU(), nn.Linear(width, width), nn.SiLU(), nn.Linear(width, 2))

        def forward(self, x, t):
            freqs = torch.exp(torch.linspace(0, math.log(100.0), self.t_dim // 2))
            emb = torch.cat([torch.sin(t[:, None] * freqs), torch.cos(t[:, None] * freqs)], dim=1)
            return self.net(torch.cat([x, emb], dim=1))

    toy_net = ToyVelocity()
    _opt = torch.optim.Adam(toy_net.parameters(), lr=2e-3)
    _beta = torch.distributions.Beta(torch.tensor(1.5), torch.tensor(1.0))
    toy_losses = []
    for _step in range(3000):
        _a = sample_bimodal(512)
        _eps = torch.randn_like(_a)
        _t = _beta.sample((512,)) * 0.999 + 0.001
        _x_t = _t[:, None] * _eps + (1 - _t[:, None]) * _a
        _loss = ((toy_net(_x_t, _t) - (_eps - _a)) ** 2).mean()
        _opt.zero_grad()
        _loss.backward()
        _opt.step()
        toy_losses.append(_loss.item())

    @torch.no_grad()
    def toy_sample(n, num_steps, seed=0):
        """t=1 노이즈에서 시작해 dt=-1/num_steps로 Euler 적분. 궤적도 함께 반환."""
        g = torch.Generator().manual_seed(seed)
        x = torch.randn(n, 2, generator=g)
        path = [x.clone()]
        dt = -1.0 / num_steps
        for s in range(num_steps):
            t = torch.full((n,), 1.0 + s * dt)
            x = x + dt * toy_net(x, t)
            path.append(x.clone())
        return x, torch.stack(path, 1)

    print(f"Flow Matching 손실: 처음 {sum(toy_losses[:100]) / 100:.3f} → 마지막 {sum(toy_losses[-100:]) / 100:.3f}")
    print("(손실이 0으로 가지 않는 것이 정상입니다. 같은 x_t에서도 ε, a 조합에 따라 정답 속도가 달라 평균만 맞힐 수 있습니다.)")
    return sample_bimodal, toy_sample


@app.cell
def _(mo):
    fm_steps = mo.ui.slider(1, 20, value=10, step=1, label="Euler 스텝 수", show_value=True)
    fm_steps
    return (fm_steps,)


@app.cell
def _(fm_steps, plt, sample_bimodal, toy_sample):
    _data = sample_bimodal(400)
    _final, _path = toy_sample(400, fm_steps.value)
    _fig, _axes = plt.subplots(1, 2, figsize=(10, 4))
    _ax = _axes[0]
    for _i in range(25):
        _ax.plot(_path[_i, :, 0], _path[_i, :, 1], "-", color="gray", lw=0.6, alpha=0.7)
    _ax.scatter(_path[:25, 0, 0], _path[:25, 0, 1], s=10, c="k", label="t=1 노이즈")
    _ax.scatter(_path[:25, -1, 0], _path[:25, -1, 1], s=14, c="tab:orange", label="t=0 생성된 행동")
    _ax.set_title(f"Euler {fm_steps.value}스텝 궤적 (25개)", fontsize=10)
    _ax.legend(fontsize=8)
    _ax = _axes[1]
    _ax.scatter(_data[:, 0], _data[:, 1], s=5, alpha=0.4, label="시연 데이터")
    _ax.scatter(_final[:, 0], _final[:, 1], s=5, alpha=0.4, c="tab:orange", label="Flow Matching 샘플")
    _mean = _data.mean(0)
    _ax.scatter(*_mean, marker="x", s=120, c="k", label="MSE 회귀의 최적해(평균)")
    _ax.set_xlim(-1.3, 1.3), _ax.set_ylim(-1.2, 0.6)
    _ax.set_title("결과 분포", fontsize=10)
    _ax.legend(fontsize=8, loc="lower center")
    for _a in _axes:
        _a.set_aspect("equal")
        _a.grid(alpha=0.3)
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    슬라이더로 적분 횟수를 1까지 줄여 분포를 예측해 보세요. 적은 횟수에서는 적분 근사 오차가 커질 수 있고,
    횟수를 늘리면 오차가 줄 수 있습니다. 두 봉우리가 복원되는 정도는 학습 결과와 함께 관찰해야 합니다. SmolVLA는 10스텝을 씁니다. 같은 관측에서도 **노이즈가 다르면 다른 행동 청크**가 나오므로,
    실제 로봇에서는 행동 청크 단위로 일관된 한 가지 해를 실행하는 것이 중요합니다(9절의 비동기 실행 보충 실습 참고).

    ## 8. KV 캐시 추론: 접두부는 한 번만

    추론에서 Euler 10스텝마다 바뀌는 것은 행동 토큰뿐입니다. 접두부(이미지·지시문·상태)는 마스크 때문에 행동을 보지 않으므로
    16층 각각의 접두부 K/V는 10스텝 동안 같습니다. `sample_actions`는 접두부를 한 번 돌려 `DynamicCache`에 층별 K/V를 저장하고,
    `denoise_step`에서는 `inputs_embeds=[None, 행동]`으로 **전문가 쪽만** 계산합니다.

    - Self 층: 캐시된 접두부 K/V 뒤에 이번 행동 K/V를 붙여 어텐션 → 끝나면 `crop`으로 다시 접두부 길이로 자릅니다.
    - Cross 층: 캐시에서 접두부 K/V를 읽기만 합니다.

    아래에서 (1) 캐시 경로와 "접두부까지 매번 다시 계산하는 학습용 `forward` 경로"의 속도 예측이 같은지, (2) 시간이 얼마나 줄어드는지 확인합니다.
    """)
    return


@app.cell
def _(mo):
    cache_btn = mo.ui.run_button(label='KV 캐시 동치성·속도 비교')
    cache_btn
    return (cache_btn,)


@app.cell
def _(
    CHUNK_SIZE,
    DEVICE,
    ON_GPU,
    cache_btn,
    demo_inputs,
    make_att_2d_masks,
    mo,
    policy,
    time,
    torch,
):
    mo.stop(not cache_btn.value, mo.md('모델 조립 후 캐시 비교 버튼을 누르세요.'))
    torch.backends.cuda.matmul.allow_tf32 = False

    def _full_velocity(prefix, x_t, t):
        pe, pp, pa = prefix
        se, sp, sa = policy.embed_suffix(x_t, t)
        pad = torch.cat([pp, sp], 1)
        att = torch.cat([pa, sa], 1)
        (_, so), _ = policy.vlm_with_expert.forward(make_att_2d_masks(pad, att), torch.cumsum(pad, 1) - 1, None, [pe, se], use_cache=False)
        return policy.action_out_proj(so[:, -CHUNK_SIZE:].float())

    with torch.no_grad():
        _g = torch.Generator(device=DEVICE).manual_seed(0)
        _x_t = torch.randn(2, CHUNK_SIZE, 32, device=DEVICE, generator=_g)
        _t = torch.full((2,), 0.7, device=DEVICE)
        _prefix = policy.embed_prefix(*demo_inputs)
        _pe, _pp, _pa = _prefix
        _, _cache = policy.vlm_with_expert.forward(make_att_2d_masks(_pp, _pa), torch.cumsum(_pp, 1) - 1, None, [_pe, None], use_cache=True)
        _cache_len = _cache.layers[0].get_seq_length()
        _v_cache = policy.denoise_step(_pp, _cache, _x_t, _t)
        _cache_len_after = _cache.layers[0].get_seq_length()
        _v_full = _full_velocity(_prefix, _x_t, _t)
        _diff = (_v_cache - _v_full).abs().max().item()
        _equiv = torch.allclose(_v_cache, _v_full, atol=1e-4, rtol=1e-4)

        # 시간 측정 (배치 1)
        _one = [x[:1] if torch.is_tensor(x) else [y[:1] for y in x] for x in demo_inputs]

        def _sync():
            if ON_GPU:
                torch.cuda.synchronize()

        _benchmark_noise = _x_t[:1].clone()

        def _cached():
            return policy.sample_actions(*_one, noise=_benchmark_noise)

        def _no_cache():
            prefix = policy.embed_prefix(*_one)
            x = _benchmark_noise.clone()
            for s in range(policy.num_steps):
                x = x - (1.0 / policy.num_steps) * _full_velocity(prefix, x, torch.full((1,), 1.0 - s / policy.num_steps, device=DEVICE))
            return x

        _times = {}
        for _name, _fn in [("KV 캐시 (원본 sample_actions)", _cached), ("임베딩 공통·Transformer 접두부 재계산", _no_cache)]:
            _fn()
            _sync()
            _t0 = time.perf_counter()
            for _ in range(3):
                _fn()
            _sync()
            _times[_name] = (time.perf_counter() - _t0) / 3 * 1000
    mo.md(f"""
    | 항목 | 값 |
    | --- | --- |
    | 캐시 길이 (접두부 토큰 수) | {_cache_len} → denoise_step 후 {_cache_len_after} (crop으로 복원) |
    | 캐시 경로 vs 전체 forward 속도 최대 차이 | {_diff:.2e} (fp32, TF32 꺼짐) |
    | 동치성 확인 (`atol=rtol=1e-4`) | {"통과" if _equiv else "실패: 마스크·위치·캐시 길이·dtype 확인 필요"} |
    | {list(_times)[0]} | {list(_times.values())[0]:.0f} ms / 청크 |
    | {list(_times)[1]} | {list(_times.values())[1]:.0f} ms / 청크 |
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    접두부가 짧은(카메라 1대, 약 75토큰) 이 실습에서는 캐시의 절약 폭이 작을 수 있습니다.
    카메라가 3대이고 지시문이 길면 접두부가 길어집니다. `demo_inputs`의 이미지 리스트를 3장으로 늘려 비교해 보세요.

    ## 9. 정리와 보충 실습

    입력 이미지는 64개 비전 토큰으로 압축되고, 지시문·상태와 함께 16층 VLM을 통과합니다.
    행동 전문가는 각 층의 접두부 K/V를 참조해 속도장을 예측하며, 같은 관측 안에서 캐시를 재사용해 노이즈를 행동 청크로 옮깁니다.
    행동 청크는 큐에 넣어 순서대로 실행하고 새 관측에서 다시 추론합니다.
    실행 중 다음 청크를 미리 요청하는 비동기 처리의 상세 구현은 아래 보충 실습에 있습니다.

    | 보충 노트북 | 내용 | 실행 조건 |
    | --- | --- | --- |
    | [VLM 분석](smolvla_vlm_experiments.py) | VQA, 층별 좌표 프로브, 레이어 절단 시간 측정 | 모델 로드·실험 버튼, GPU 권장 |
    | [학습·평가](smolvla_training.py) | 기본 평가, 파인튜닝, 폐루프 평가, 지시문·노이즈 비교 | 모델 로드·학습·평가 버튼, GPU 권장 |
    | [비동기 실행](smolvla_async_execution.py) | 큐 정렬, trim, 청크 혼합 | CPU, 다운로드 없음 |

    각 노트북은 별도 앱으로 엽니다. 이 앱의 메모리나 실행 상태를 넘겨줄 필요는 없습니다.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 다른 VLA와 비교 ([smolvla.md](smolvla.md) 4절)

    | 구분 | OpenVLA (7B) | π0 (약 3.3B) | **SmolVLA (0.45B)** |
    | --- | --- | --- | --- |
    | 백본 | Prismatic VLM (Llama-2 7B) | PaliGemma (Gemma 2B) | **SmolVLM2 (500M), 앞 16층** |
    | 행동 표현 | 이산 토큰 (256 구간) | Flow Matching | **Flow Matching** |
    | 이미지 토큰 | 고해상도 패치 다수 | 표준 패치 토큰 | **Pixel Shuffle로 이미지당 64개** |
    | 행동 전문가 어텐션 | — | 모든 층 공동 Self-Attention | **Self / Cross 층 교대** |
    | 행동 토큰끼리 | — | 양방향 | **인과적** |

    ### 구조 안내와 실제 구현 대응

    | 핵심 선택 | 확인할 구현 | 연결 절 |
    | --- | --- | --- |
    | VLM 앞쪽 16층 | `layers[:num_vlm_layers]` | 4·5 |
    | 짝수 Self / 홀수 Cross 층 교대 | `forward_attn_layer`, `forward_cross_attn_layer` | 5 |
    | VLM K/V 320차원 재투영, 시간 concat | `k_proj/v_proj`, `action_time_mlp_in/out` | 5 |
    | 이미지·언어 양방향 / 상태 별도 / 행동 인과 | `make_att_2d_masks` | 6 |
    | $t=1$ noise, $t=0$ action, Beta 시간표집 | `forward`, `sample_time`, `sample_actions` | 7 |
    | 같은 관측 안에서만 접두부 K/V 재사용 | `denoise_step`, `DynamicCache` | 8 |

    ### 연습 문제와 판정 기준

    1. 6절의 `embed_suffix`에서 `att_masks`를 `[1] + [0]*(chunk-1)`로 바꾸면 행동 블록이 양방향이 됩니다.
       그림으로 확인하세요. 공개 가중치는 원래 마스크로 학습했으므로 성능 향상을 보장하지 않습니다.
    2. 7절의 시간 표집을 균등분포로 바꾸고 초기화·데이터·노이즈 seed·학습량을 맞춰 비교하세요.
       여러 적분 횟수에서 두 모드의 포함 비율·분포·손실을 보고 유한 표본 결과를 단정하지 않습니다.

    | 절 | 먼저 답할 질문 | 기대 답·확인할 근거 |
    | --- | --- | --- |
    | 1 | 2D 상태와 행동을 왜 32차원으로 늘리나요? | 모델 폭에 맞춥니다. 실제 행동 차원은 앞 2칸이며 학습 실습은 이 부분의 손실만 씁니다. |
    | 2 | VLM의 답변이 행동 입력인가요? | 정책은 중간 층 K/V와 속도장을 씁니다. |
    | 3 | Pixel Shuffle 자체가 정보를 버리나요? | 값의 재배열이며 뒤의 선형 투영이 차원을 줄입니다. |
    | 4 | 16층 절단으로 전체 계산 시간이 절반이 되나요? | 비전 인코더 연산은 그대로 남습니다. |
    | 5 | 두 경로의 폭이 달라도 연결되는 이유는? | Q/K/V 헤드 투영 형상이 맞고 각 경로는 자기 출력 투영을 씁니다. |
    | 6 | 이미지 토큰이 행동을 볼 수 있나요? | 블록 마스크가 막고 패딩도 행·열 모두 차단합니다. |
    | 7 | 속도가 noise 방향인데 왜 행동으로 가나요? | 시간 1에서 0으로 음의 간격으로 적분합니다. |
    | 8 | 새 지시문에도 이전 캐시를 그대로 쓰나요? | 같은 관측의 적분 내부에서만 재사용합니다. |
    | 9 | 이 실습으로 실물 로봇 성능을 판단할 수 있나요? | 역학·센서·정규화·분포 차이 때문에 별도 검증이 필요합니다. |
    """)
    return


if __name__ == "__main__":
    app.run()
