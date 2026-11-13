import marimo

__generated_with = "0.25.2"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## marimo 실행

    노트북을 실행하면 모델·데이터 다운로드부터 학습·평가·저장까지 원본 순서대로 진행합니다. 설정을 바꾼 뒤에는 커널을 재시작하세요. 산출물은 저장소 루트 `outputs/vlm-from-scratch/<실행 ID>/`에 저장합니다. 앞 파트의 가중치를 재사용하려면 같은 `VLM_RUN_ID`로 실행하세요.
    """)
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    from pathlib import Path as _RunPath
    from datetime import datetime as _RunDatetime
    from zoneinfo import ZoneInfo as _RunZoneInfo
    import os as _run_os
    import re as _run_re
    _run_id = _run_os.environ.get("VLM_RUN_ID") or _RunDatetime.now(_RunZoneInfo("Asia/Seoul")).strftime("%Y%m%d-%H%M%S-vlm")
    if not _run_re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", _run_id):
        raise ValueError("VLM_RUN_ID에는 영문, 숫자, 밑줄, 하이픈만 사용할 수 있습니다.")
    output_dir = _RunPath(__file__).resolve().parents[3] / "outputs" / "vlm-from-scratch" / _run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"산출물 저장 위치: {output_dir}")
    execution_step_0 = 0
    return execution_step_0, output_dir


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # VLM의 시각적 그라운딩과 다각형 분할

    Nipun Batra · 2025-12-31
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 소개

    지금까지 캡션 생성, 객체 탐지, 질의응답, 단계별 추론이 가능한 VLM을 만들었습니다. 다음 작업은 어떻게 처리할까요?

    - **특정 객체 찾기**: 모든 개가 아닌 "빨간 목줄을 한 개" 찾기
    - **객체 윤곽을 정밀하게 표시**: 경계 상자가 아닌 정확한 모양 얻기
    - **공간 관계 이해**: "맨 왼쪽 사람", "노트북 뒤의 컵"

    이것이 **시각적 그라운딩(지칭 표현 이해)**과 **분할**의 역할입니다.

    ### 이전 노트북
    1. [최소 구성의 VLM](part-01-minimal-vlm.py): 캡션
    2. [객체 탐지](part-02-object-detection.py): 모든 객체 찾기
    3. [VQA](part-03-visual-qa.py): 질문에 답하기
    4. [멀티태스크](part-04-multi-task.py): 작업 통합
    5. [다중 이미지](part-05-multi-image.py): 시간적 추론
    6. [태스크 라우팅](part-06-task-routing.py): 작업 자동 감지
    7. [사고 과정](part-07-chain-of-thought.py): 단계별 추론

    ### 이번에 새로 다룰 내용

    **시각적 그라운딩(지칭 표현):**
    - 입력: 이미지 + "the dog with the red collar"(빨간 목줄을 한 개)
    - 출력: 바로 그 개의 경계 상자
    - 속성, 공간 관계, 문맥의 이해 필요

    **다각형 분할:**
    - 입력: 이미지 + "segment the cat"(고양이를 분할하세요)
    - 출력: 고양이 윤곽을 이루는 다각형 꼭짓점
    - 경계 상자보다 정밀함
    - 언어 모델이 생성할 수 있는 좌표 시퀀스로 출력

    ### 객체 탐지와 지칭 표현 비교

    | 작업 | 입력 | 출력 | 예시 |
    |------|-------|--------|--------|
    | **객체 탐지** | 이미지 | 모든 객체 | "모든 개를 찾으세요" → 상자 3개 |
    | **지칭 표현** | 이미지 + 설명 | 특정 객체 하나 | "빨간 목줄을 한 개" → 상자 1개 |

    ### 경계 상자와 다각형 분할 비교

    | 표현 | 형식 | 정밀도 | 예시 |
    |----------------|--------|-----------|--------|
    | **경계 상자** | `[x1, y1, x2, y2]` | 대략적인 영역 | `[120, 45, 200, 150]` |
    | **다각형** | `[(x1,y1), (x2,y2), ...]` | 정확한 윤곽 | `[(120,45), (125,50), (130,48), ...]` |

    ### 구현할 내용

    1. **지칭 표현 이해**: 설명으로 특정 객체 찾기
    2. **다각형 분할**: 꼭짓점으로 객체 윤곽 표시
    3. **작업 결합**: 한 모델에서 그라운딩과 분할 처리

    **데이터셋:**
    - **RefCOCO/RefCOCO+**: 분할 마스크가 있는 지칭 표현
    - **COCO**: 인스턴스 분할(대체 경로)

    ### 구조

    VLM을 확장해 구조화된 데이터를 출력합니다.

    ```
    입력: [이미지] + "the cat sitting on the left"(왼쪽에 앉은 고양이)

    출력: {
      "bbox": [120, 45, 200, 150],
      "polygon": [(120,45), (125,50), (130,48), ..., (120,45)]
    }
    ```
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 환경 설정
    """)
    return


@app.cell
def _(execution_step_0):
    # 원본 코드 셀 순서와 모델·데이터 변경의 실행 순서를 보존합니다.
    execution_step_1 = execution_step_0 + 1
    import os

    return execution_step_1, os


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    프로젝트 디렉토리(`notebooks/vlm-from-scratch/`)에서 `uv sync --locked`로 의존성을 설치하세요.
    """)
    return


@app.cell
def _(execution_step_1):
    # 원본 코드 셀 순서와 모델·데이터 변경의 실행 순서를 보존합니다.
    execution_step_2 = execution_step_1 + 1
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        ViTModel,
        ViTImageProcessor,
    )
    from datasets import load_dataset
    from PIL import Image, ImageDraw
    import numpy as np
    import matplotlib.pyplot as plt
    # 설치된 한글 글꼴을 선택해 그래프의 한글 깨짐 방지
    from matplotlib import font_manager as _font_manager
    _available_fonts = {_font.name for _font in _font_manager.fontManager.ttflist}
    for _font_name in ("Noto Sans CJK KR", "Noto Sans CJK JP", "NanumGothic", "Malgun Gothic", "AppleGothic"):
        if _font_name in _available_fonts:
            plt.rcParams["font.family"] = [_font_name]
            break
    plt.rcParams["axes.unicode_minus"] = False
    from matplotlib.patches import Polygon as MPLPolygon
    from tqdm.auto import tqdm
    import random
    import json
    import warnings
    warnings.filterwarnings('ignore')

    # 마스크를 다각형으로 변환하기 위한 도구
    from skimage import measure
    from pycocotools import mask as mask_utils


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'사용 장치: {device}')
    if torch.cuda.is_available():
        print(f'GPU: {torch.cuda.get_device_name(0)}')
    return (
        AutoModelForCausalLM,
        AutoTokenizer,
        DataLoader,
        Dataset,
        F,
        Image,
        MPLPolygon,
        ViTImageProcessor,
        ViTModel,
        device,
        execution_step_2,
        load_dataset,
        mask_utils,
        measure,
        nn,
        np,
        plt,
        random,
        torch,
        tqdm,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1단계: 기반 VLM 불러오기
    """)
    return


@app.cell
def _(
    AutoModelForCausalLM,
    AutoTokenizer,
    F,
    ViTModel,
    execution_step_2,
    nn,
    torch,
):
    # 원본 코드 셀 순서와 모델·데이터 변경의 실행 순서를 보존합니다.
    execution_step_3 = execution_step_2 + 1
    # VLM 구조(앞과 동일)

    class VisionProjector(nn.Module):
        """시각 특징을 언어 모델의 임베딩 공간으로 투영합니다.
        """
    
        def __init__(self, vision_dim, language_dim):
            super().__init__()
            self.projection = nn.Sequential(
                nn.Linear(vision_dim, language_dim),
                nn.GELU(),
                nn.LayerNorm(language_dim),
                nn.Linear(language_dim, language_dim),
            )
    
        def forward(self, vision_features):
            return self.projection(vision_features)


    class MiniVLM(nn.Module):
        """최소 구성의 시각·언어 모델입니다.
        """
    
        def __init__(
            self,
            vision_encoder,
            language_model,
            projector,
            tokenizer,
        ):
            super().__init__()
            self.vision_encoder = vision_encoder
            self.language_model = language_model
            self.projector = projector
            self.tokenizer = tokenizer
        
            for param in self.vision_encoder.parameters():
                param.requires_grad = False
    
        def encode_image(self, pixel_values):
            with torch.no_grad():
                vision_outputs = self.vision_encoder(pixel_values=pixel_values)
            image_features = vision_outputs.last_hidden_state
            projected = self.projector(image_features)
            return projected
    
        def forward(
            self,
            pixel_values,
            input_ids,
            attention_mask,
            labels = None,
        ):
            batch_size = pixel_values.shape[0]
            image_embeds = self.encode_image(pixel_values)
            num_image_tokens = image_embeds.shape[1]
        
            text_embeds = self.language_model.get_input_embeddings()(input_ids)
            combined_embeds = torch.cat([image_embeds, text_embeds], dim=1)
        
            image_attention = torch.ones(
                (batch_size, num_image_tokens),
                dtype=attention_mask.dtype,
                device=attention_mask.device
            )
            combined_attention = torch.cat([image_attention, attention_mask], dim=1)
        
            if labels is not None:
                image_labels = torch.full(
                    (batch_size, num_image_tokens),
                    fill_value=-100,
                    dtype=labels.dtype,
                    device=labels.device
                )
                combined_labels = torch.cat([image_labels, labels], dim=1)
            else:
                combined_labels = None
        
            outputs = self.language_model(
                inputs_embeds=combined_embeds,
                attention_mask=combined_attention,
                labels=combined_labels,
                return_dict=True,
            )
        
            return outputs
    
        @torch.no_grad()
        def generate(
            self,
            pixel_values,
            prompt,
            max_new_tokens = 150,
            temperature = 0.7,
            do_sample = True,
        ):
            """주어진 프롬프트에 따라 이미지 응답을 생성합니다.
            """
            self.eval()
        
            image_embeds = self.encode_image(pixel_values)
            prompt_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(pixel_values.device)
            generated_ids = prompt_ids.clone()
        
            for _ in range(max_new_tokens):
                current_embeds = self.language_model.get_input_embeddings()(generated_ids)
                full_embeds = torch.cat([image_embeds, current_embeds], dim=1)
            
                outputs = self.language_model(inputs_embeds=full_embeds)
                next_token_logits = outputs.logits[:, -1, :]
            
                if do_sample:
                    probs = F.softmax(next_token_logits / temperature, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = next_token_logits.argmax(dim=-1, keepdim=True)
            
                generated_ids = torch.cat([generated_ids, next_token], dim=1)
            
                if next_token.item() == self.tokenizer.eos_token_id:
                    break
        
            return self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    return MiniVLM, VisionProjector, execution_step_3


@app.cell
def _(
    AutoModelForCausalLM,
    AutoTokenizer,
    MiniVLM,
    ViTImageProcessor,
    ViTModel,
    VisionProjector,
    device,
    execution_step_3,
    os,
    output_dir,
    torch,
):
    # 원본 코드 셀 순서와 모델·데이터 변경의 실행 순서를 보존합니다.
    execution_step_4 = execution_step_3 + 1
    # 체크포인트가 없을 때만 이 파트의 기본 모델을 사용합니다.
    vision_model_name = "google/vit-base-patch16-224"
    lm_model_name = "HuggingFaceTB/SmolLM-135M"
    _checkpoint = None
    _checkpoint_path = None
    for _candidate in [
        output_dir / "mini-vlm-multitask/mini_vlm_multitask.pt",
        output_dir / "mini-vlm-flickr8k/mini_vlm_full.pt",
    ]:
        if os.path.isfile(_candidate):
            _checkpoint = torch.load(_candidate, map_location="cpu", weights_only=True)
            _checkpoint_path = _candidate
            _config = _checkpoint["config"]
            # 모델을 먼저 만들면 이전 파트와 차원이 달라 복원에 실패할 수 있습니다.
            vision_model_name = _config["vision_model_name"]
            lm_model_name = _config["lm_model_name"]
            break

    vision_encoder = ViTModel.from_pretrained(vision_model_name)
    language_model = AutoModelForCausalLM.from_pretrained(lm_model_name)
    # 저장된 토크나이저·이미지 전처리 설정도 같은 체크포인트에서 복원합니다.
    _tokenizer_path = _checkpoint_path.parent / "tokenizer" if _checkpoint_path else None
    _processor_path = _checkpoint_path.parent / "image_processor" if _checkpoint_path else None
    tokenizer = AutoTokenizer.from_pretrained(
        str(_tokenizer_path) if _tokenizer_path and _tokenizer_path.is_dir() else lm_model_name
    )
    image_processor = ViTImageProcessor.from_pretrained(
        str(_processor_path) if _processor_path and _processor_path.is_dir() else vision_model_name
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if len(tokenizer) != language_model.get_input_embeddings().num_embeddings:
        language_model.resize_token_embeddings(len(tokenizer), mean_resizing=False)

    vision_dim = vision_encoder.config.hidden_size
    language_dim = language_model.config.hidden_size
    projector = VisionProjector(vision_dim, language_dim)
    if _checkpoint is not None:
        if (vision_dim, language_dim) != (_config["vision_dim"], _config["language_dim"]):
            raise ValueError(f"체크포인트 설정과 모델 차원이 다릅니다: {_checkpoint_path}")
        projector.load_state_dict(_checkpoint["projector_state_dict"])
        language_model.load_state_dict(_checkpoint["language_model_state_dict"])
        print(f"모델 구성을 복원했습니다: {_checkpoint_path}")
        # 가중치를 모듈에 복사한 뒤 CPU 체크포인트의 메모리 참조를 해제합니다.
        del _checkpoint
    else:
        print("체크포인트가 없어 기본 모델 구성으로 시작합니다.")

    vlm = MiniVLM(vision_encoder, language_model, projector, tokenizer).to(device)
    print(f"비전 인코더: {vision_model_name} (은닉 차원={vision_dim})")
    print(f"언어 모델: {lm_model_name} (은닉 차원={language_dim})")
    print(f"모델 로딩 장치: {device}")
    print(f"학습 가능한 매개변수: {sum(p.numel() for p in vlm.parameters() if p.requires_grad):,}")
    return (execution_step_4, image_processor, language_dim, lm_model_name, tokenizer, vision_dim, vision_model_name, vlm)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2단계: RefCOCO 데이터셋 불러오기

    RefCOCO에는 다음이 포함됩니다.
    - COCO의 이미지
    - 지칭 표현("the dog on the left": 왼쪽의 개)
    - 지칭된 객체의 분할 마스크
    - 경계 상자
    """)
    return


@app.cell
def _(execution_step_4, load_dataset):
    # 원본 코드 셀 순서와 모델·데이터 변경의 실행 순서를 보존합니다.
    execution_step_5 = execution_step_4 + 1
    # RefCOCO 데이터셋 로딩
    print('RefCOCO 데이터셋 로딩 중...')
    try:
        refcoco = load_dataset('common-canvas/commoncatalog-cc-by', 'refcoco', split='train', streaming=True)
        refcoco_samples = []  # Hugging Face의 RefCOCO(unc 분할)
        for i, _sample in enumerate(refcoco):
            if i >= 2000:
                break  # 샘플 수집
            refcoco_samples.append(_sample)
        print(f'불러온 수: {len(refcoco_samples)}개 RefCOCO 샘플')
    except Exception as e:  # 샘플 2,000개 사용
        print(f'RefCOCO 로딩 실패: {e}')
        print('합성 지칭 표현을 사용하는 COCO로 전환합니다...')
        coco = load_dataset('detection-datasets/coco', split='train', streaming=True)
        refcoco_samples = []
        for i, _sample in enumerate(coco):
            if i >= 2000:
                break
            refcoco_samples.append(_sample)
        print(f'불러온 수: {len(refcoco_samples)}개 COCO 샘플(합성 지칭 표현 추가 예정)')  # 대체 경로: COCO 탐지를 사용하고 합성 지칭 표현 생성
    return execution_step_5, refcoco_samples


@app.cell
def _(execution_step_5, refcoco_samples):
    # 원본 코드 셀 순서와 모델·데이터 변경의 실행 순서를 보존합니다.
    execution_step_6 = execution_step_5 + 1
    # 샘플 확인
    if refcoco_samples:
        _sample = refcoco_samples[0]
        print('샘플 키:', _sample.keys())
        print('\n샘플 구조:')
        for key, value in _sample.items():
            if isinstance(value, (str, int, float)):
                print(f'  {key}: {value}')
            elif isinstance(value, list) and len(value) > 0:
                print(f'  {key}: 목록 길이 {len(value)}개(첫 항목: {type(value[0])})')
            else:
                print(f'  {key}: {type(value)}')
    return (execution_step_6,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3단계: 다각형 유틸리티

    다음 기능이 필요합니다.
    1. 분할 마스크를 다각형으로 변환
    2. 다각형 단순화(꼭짓점 수 줄이기)
    3. 좌표 정규화
    4. 언어 모델용 텍스트 형식으로 변환
    """)
    return


@app.cell
def _(execution_step_6, mask_utils, measure, np, torch):
    # 원본 코드 셀 순서와 모델·데이터 변경의 실행 순서를 보존합니다.
    execution_step_7 = execution_step_6 + 1
    def mask_to_polygon(mask, max_vertices=50, simplify_tolerance=2.0):
        """이진 마스크를 다각형 꼭짓점으로 변환합니다.
        
        인자:
            mask: 이진 마스크 (H, W) 또는 RLE
            max_vertices: 최대 꼭짓점 수
            simplify_tolerance: 다각형 단순화 허용 오차(클수록 단순해짐)
        
        반환:
            (x, y) 튜플 목록
        """
        # 필요하면 RLE를 이진 마스크로 변환
        if isinstance(mask, dict):
            # COCO RLE 형식
            mask = mask_utils.decode(mask)
    
        # numpy 배열로 변환
        if torch.is_tensor(mask):
            mask = mask.cpu().numpy()
    
        mask = mask.astype(np.uint8)
    
        # 윤곽선 찾기
        contours = measure.find_contours(mask, 0.5)
    
        if len(contours) == 0:
            return []
    
        # 가장 큰 윤곽선 선택
        contour = max(contours, key=len)
    
        # Douglas-Peucker 알고리즘으로 다각형 단순화
        from skimage.measure import approximate_polygon
        simplified = approximate_polygon(contour, tolerance=simplify_tolerance)
    
        # 꼭짓점 수 제한
        if len(simplified) > max_vertices:
            # 균일하게 샘플링
            step = len(simplified) // max_vertices
            simplified = simplified[::step][:max_vertices]
    
        # (x, y) 목록으로 변환: 윤곽선은 (행, 열)이므로 순서 교환
        polygon = [(int(y), int(x)) for x, y in simplified]
    
        return polygon


    def normalize_polygon(polygon, image_width, image_height, scale=1000):
        """다각형 좌표를 [0, scale] 범위로 정규화합니다.
        언어 모델이 더 쉽게 학습하도록 합니다.
        
        인자:
            polygon: (x, y) 튜플 목록
            image_width, image_height: 원본 이미지 크기
            scale: 목표 스케일(기본값 1000)
        
        반환:
            정규화한 (x, y) 튜플 목록
        """
        normalized = [
            (
                int(x / image_width * scale),
                int(y / image_height * scale)
            )
            for x, y in polygon
        ]
        return normalized


    def denormalize_polygon(polygon, image_width, image_height, scale=1000):
        """다각형 좌표의 정규화를 되돌려 원본 이미지 크기로 복원합니다.
        """
        denormalized = [
            (
                int(x / scale * image_width),
                int(y / scale * image_height)
            )
            for x, y in polygon
        ]
        return denormalized


    def polygon_to_text(polygon):
        """언어 모델을 위해 다각형을 텍스트로 변환합니다.
        
        형식: "<polygon> (x1,y1) (x2,y2) ... (xn,yn) </polygon>"
        """
        if not polygon:
            return "<polygon> </polygon>"
    
        points_str = " ".join([f"({x},{y})" for x, y in polygon])
        return f"<polygon> {points_str} </polygon>"


    def text_to_polygon(text):
        """텍스트 형식에서 다각형을 파싱합니다.
        
        입력: "<polygon> (x1,y1) (x2,y2) ... </polygon>"
        출력: [(x1,y1), (x2,y2), ...]
        """
        try:
            # <polygon> 태그 사이의 내용 추출
            if "<polygon>" in text and "</polygon>" in text:
                content = text.split("<polygon>")[1].split("</polygon>")[0].strip()
            else:
                content = text
        
            # (x,y) 쌍 파싱
            polygon = []
            for match in content.split():
                if "(" in match and ")" in match:
                    coords = match.strip("()").split(",")
                    if len(coords) == 2:
                        x, y = int(coords[0]), int(coords[1])
                        polygon.append((x, y))
        
            return polygon
        except:
            return []


    def bbox_to_text(bbox):
        """경계 상자를 텍스트로 변환합니다.
        
        형식: "<bbox> x1 y1 x2 y2 </bbox>"
        """
        return f"<bbox> {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]} </bbox>"


    def text_to_bbox(text):
        """텍스트에서 경계 상자를 파싱합니다.
        """
        try:
            if "<bbox>" in text and "</bbox>" in text:
                content = text.split("<bbox>")[1].split("</bbox>")[0].strip()
            else:
                content = text
        
            coords = [int(x) for x in content.split()]
            if len(coords) == 4:
                return coords
            return None
        except:
            return None


    print('다각형 유틸리티를 정의했습니다.')

    # 테스트
    test_polygon = [(100, 50), (150, 75), (140, 120), (90, 110)]
    text = polygon_to_text(test_polygon)
    print(f'\n다각형 → 텍스트 예시: {text}')
    parsed = text_to_polygon(text)
    print(f'텍스트 → 다각형: {parsed}')
    return (
        bbox_to_text,
        denormalize_polygon,
        execution_step_7,
        mask_to_polygon,
        normalize_polygon,
        polygon_to_text,
        text_to_bbox,
        text_to_polygon,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4단계: 지칭·분할 데이터셋 만들기

    모델에 다음을 가르치는 데이터셋을 만듭니다.
    1. 지칭 표현 이해
    2. 경계 상자 출력
    3. 다각형 분할 출력
    """)
    return


@app.cell
def _(execution_step_7, random):
    # 원본 코드 셀 순서와 모델·데이터 변경의 실행 순서를 보존합니다.
    execution_step_8 = execution_step_7 + 1
    # COCO에서 합성 지칭 표현을 만드는 보조 함수

    def create_referring_expression(category_name, bbox, image_width, image_height):
        """객체 범주와 위치로 간단한 지칭 표현을 만듭니다.
        """
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
    
        # 위치 결정
        horizontal = "left" if center_x < image_width / 2 else "right"
        vertical = "top" if center_y < image_height / 2 else "bottom"
    
        # 크기
        area = (x2 - x1) * (y2 - y1)
        relative_area = area / (image_width * image_height)
        size = "large" if relative_area > 0.2 else "small" if relative_area < 0.05 else ""
    
        # 표현 생성
        templates = [
            f"the {category_name} on the {horizontal}",
            f"the {category_name} on the {vertical}",
            f"the {category_name} in the {vertical} {horizontal}",
            f"the {size} {category_name}" if size else f"the {category_name}",
        ]
    
        return random.choice(templates)


    print('지칭 표현 생성기를 정의했습니다.')
    return create_referring_expression, execution_step_8


@app.cell
def _(
    Dataset,
    bbox_to_text,
    create_referring_expression,
    execution_step_8,
    mask_to_polygon,
    normalize_polygon,
    polygon_to_text,
    random,
):
    # 원본 코드 셀 순서와 모델·데이터 변경의 실행 순서를 보존합니다.
    execution_step_9 = execution_step_8 + 1
    class ReferringSegmentationDataset(Dataset):
        """지칭 표현 이해·다각형 분할용 데이터셋입니다.
        
        출력 형식:
        "Referring expression: {expr}
         Bounding box: <bbox> x1 y1 x2 y2 </bbox>
         Segmentation: <polygon> (x1,y1) ... </polygon>"
        """
    
        def __init__(self, samples, image_processor, tokenizer, max_length=384, coord_scale=1000):
            self.samples = samples
            self.image_processor = image_processor
            self.tokenizer = tokenizer
            self.max_length = max_length
            self.coord_scale = coord_scale
    
        def __len__(self):
            return len(self.samples)
    
        def __getitem__(self, idx):
            sample = self.samples[idx]
        
            # 이미지 처리
            image = sample['image'].convert('RGB')
            image_width, image_height = image.size
            pixel_values = self.image_processor(image, return_tensors="pt").pixel_values.squeeze(0)
        
            # 지칭 표현, 경계 상자, 마스크 추출
            # 데이터셋 구조에 따라 RefCOCO 또는 COCO에 맞게 처리
        
            # RefCOCO가 있는 경우
            if 'refexp' in sample or 'sentences' in sample:
                refexp = sample.get('refexp', sample.get('sentences', [''])[0])
                bbox = sample.get('bbox', [0, 0, 100, 100])
                mask = sample.get('mask', None)
            # COCO 대체 경로: 첫 객체 사용
            elif 'objects' in sample:
                objs = sample['objects']
                if len(objs['bbox']) > 0:
                    idx_obj = random.randint(0, len(objs['bbox']) - 1)
                    category = objs['category'][idx_obj] if 'category' in objs else 'object'
                    bbox = objs['bbox'][idx_obj]  # COCO 형식: [x, y, w, h]
                    # [x1, y1, x2, y2]로 변환
                    bbox = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]
                    refexp = create_referring_expression(category, bbox, image_width, image_height)
                    mask = objs.get('segmentation', [None])[idx_obj] if 'segmentation' in objs else None
                else:
                    # 객체가 없으면 건너뛰기
                    refexp = "the object"
                    bbox = [0, 0, 100, 100]
                    mask = None
            else:
                # 대체 경로
                refexp = "the object"
                bbox = [50, 50, 150, 150]
                mask = None
        
            # 경계 상자 좌표 정규화
            bbox_norm = [
                int(bbox[0] / image_width * self.coord_scale),
                int(bbox[1] / image_height * self.coord_scale),
                int(bbox[2] / image_width * self.coord_scale),
                int(bbox[3] / image_height * self.coord_scale),
            ]
        
            # 마스크를 다각형으로 변환
            if mask is not None:
                try:
                    polygon = mask_to_polygon(mask, max_vertices=40)
                    polygon_norm = normalize_polygon(polygon, image_width, image_height, self.coord_scale)
                except:
                    # 대체 경로: 경계 상자에서 다각형 생성
                    polygon_norm = [
                        (bbox_norm[0], bbox_norm[1]),
                        (bbox_norm[2], bbox_norm[1]),
                        (bbox_norm[2], bbox_norm[3]),
                        (bbox_norm[0], bbox_norm[3]),
                    ]
            else:
                # 마스크가 없으면 경계 상자 모서리를 다각형으로 사용
                polygon_norm = [
                    (bbox_norm[0], bbox_norm[1]),
                    (bbox_norm[2], bbox_norm[1]),
                    (bbox_norm[2], bbox_norm[3]),
                    (bbox_norm[0], bbox_norm[3]),
                ]
        
            # 텍스트 형식으로 변환
            bbox_text = bbox_to_text(bbox_norm)
            polygon_text = polygon_to_text(polygon_norm)
        
            # 학습 형식 생성
            prompt = f"Referring expression: {refexp}\nLocate and segment:"
            response = f"Bounding box: {bbox_text}\nSegmentation: {polygon_text}"
            full_text = f"{prompt} {response}{self.tokenizer.eos_token}"
        
            # 토큰화
            encoding = self.tokenizer(
                full_text,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
        
            input_ids = encoding['input_ids'].squeeze(0)
            attention_mask = encoding['attention_mask'].squeeze(0)
        
            # 프롬프트 마스킹(경계 상자와 다각형만 학습)
            prompt_tokens = self.tokenizer.encode(prompt, add_special_tokens=False)
            prompt_len = len(prompt_tokens)
        
            labels = input_ids.clone()
            labels[:prompt_len] = -100
            labels[attention_mask == 0] = -100
        
            return {
                'pixel_values': pixel_values,
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'labels': labels,
            }

    return ReferringSegmentationDataset, execution_step_9


@app.cell
def _(
    DataLoader,
    ReferringSegmentationDataset,
    execution_step_9,
    image_processor,
    refcoco_samples,
    tokenizer,
):
    # 원본 코드 셀 순서와 모델·데이터 변경의 실행 순서를 보존합니다.
    execution_step_10 = execution_step_9 + 1
    # 데이터셋 생성
    refseg_dataset = ReferringSegmentationDataset(
        refcoco_samples,
        image_processor,
        tokenizer,
        max_length=384,
        coord_scale=1000
    )

    refseg_loader = DataLoader(
        refseg_dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0,
    )

    print(f'지칭·분할 데이터셋: {len(refseg_dataset)}개 샘플')
    print(f'배치 수: {len(refseg_loader)}')
    return execution_step_10, refseg_dataset, refseg_loader


@app.cell
def _(execution_step_10, refseg_dataset, tokenizer):
    # 원본 코드 셀 순서와 모델·데이터 변경의 실행 순서를 보존합니다.
    execution_step_11 = execution_step_10 + 1
    # 데이터셋 형식 확인
    sample_item = refseg_dataset[0]
    decoded = tokenizer.decode(sample_item['input_ids'], skip_special_tokens=True)
    print('학습 형식 샘플:')
    print(decoded[:400])
    print("...")
    return (execution_step_11,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5단계: 학습

    모델이 다음을 수행하도록 학습합니다.
    1. 지칭 표현 이해
    2. 경계 상자 출력
    3. 다각형 분할 출력
    """)
    return


@app.cell
def _(device, execution_step_11, os, output_dir, torch, tqdm):
    # 원본 코드 셀 순서와 모델·데이터 변경의 실행 순서를 보존합니다.
    execution_step_12 = execution_step_11 + 1
    def train_referring_segmentation(model, train_loader, num_epochs=6, lr=1e-4, checkpoint_path=str(output_dir / "refseg_checkpoint.pt")):
        """지칭 표현·분할 모델을 학습합니다.
        """
    
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable_params, lr=lr)
    
        # 체크포인트 로딩 시도
        start_epoch = 0
        losses = []
    
        if os.path.exists(checkpoint_path):
            print(f'체크포인트 발견: {checkpoint_path}')
            try:
                checkpoint = torch.load(checkpoint_path, map_location='cpu')
                start_epoch = checkpoint.get('epoch', 0)
                losses = checkpoint.get('losses', [])
                model.projector.load_state_dict(checkpoint['projector_state_dict'])
                model.language_model.load_state_dict(checkpoint['language_model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                print(f'다음 에포크부터 재개: {start_epoch}')
            except Exception as e:
                print(f'체크포인트 로딩 실패: {e}')
    
        model.train()
        model.vision_encoder.eval()
    
        try:
            for epoch in range(start_epoch, num_epochs):
                epoch_loss = 0
                progress_bar = tqdm(train_loader, desc=f'에포크 {epoch + 1}/{num_epochs}')
            
                for batch in progress_bar:
                    pixel_values = batch['pixel_values'].to(device)
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    labels = batch['labels'].to(device)
                
                    outputs = model(
                        pixel_values=pixel_values,
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                    )
                
                    loss = outputs.loss
                
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                    optimizer.step()
                
                    epoch_loss += loss.item()
                    progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})
            
                avg_loss = epoch_loss / len(train_loader)
                losses.append(avg_loss)
                print(f'에포크 {epoch + 1} - 평균 손실: {avg_loss:.4f}')
            
                # 체크포인트 저장
                torch.save({
                    'epoch': epoch + 1,
                    'losses': losses,
                    'projector_state_dict': model.projector.state_dict(),
                    'language_model_state_dict': model.language_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, checkpoint_path)
                print(f'체크포인트를 저장했습니다.')
    
        except KeyboardInterrupt:
            print("\n" + "="*70)
            print('학습을 중단했습니다.')
            print(f'완료한 에포크 수: {len(losses)}에포크')
            print(f'체크포인트 저장 위치: {checkpoint_path}')
            print("="*70)
    
        return losses

    return execution_step_12, train_referring_segmentation


@app.cell
def _(execution_step_12, refseg_loader, train_referring_segmentation, vlm):
    # 원본 코드 셀 순서와 모델·데이터 변경의 실행 순서를 보존합니다.
    execution_step_13 = execution_step_12 + 1
    # 모델 학습
    print('지칭 표현·분할 VLM 학습 중...\n')
    losses = train_referring_segmentation(vlm, refseg_loader, num_epochs=6, lr=1e-4)
    return execution_step_13, losses


@app.cell
def _(execution_step_13, losses, plt):
    # 원본 코드 셀 순서와 모델·데이터 변경의 실행 순서를 보존합니다.
    execution_step_14 = execution_step_13 + 1
    # 학습 손실 그래프
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(losses)+1), losses, marker='o', linewidth=2, markersize=8)
    plt.xlabel('에포크', fontsize=12)
    plt.ylabel('손실', fontsize=12)
    plt.title('지칭 표현·분할 학습 손실', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    return (execution_step_14,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6단계: 시각적 그라운딩·분할 테스트

    모델이 다음을 수행하는지 확인합니다.
    1. 지칭 표현으로 객체 찾기
    2. 정확한 경계 상자 출력
    3. 다각형 분할 생성
    """)
    return


@app.cell
def _(
    Image,
    denormalize_polygon,
    execution_step_14,
    text_to_bbox,
    text_to_polygon,
):
    # 원본 코드 셀 순서와 모델·데이터 변경의 실행 순서를 보존합니다.
    execution_step_15 = execution_step_14 + 1
    def predict_referring_segmentation(model, image, refexp, image_processor, tokenizer, device, coord_scale=1000):
        """지칭 표현에서 경계 상자와 다각형 분할을 예측합니다.
        """
        model.eval()
    
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        image = image.convert('RGB')
        image_width, image_height = image.size
    
        pixel_values = image_processor(image, return_tensors="pt").pixel_values.to(device)
    
        prompt = f"Referring expression: {refexp}\nLocate and segment:"
    
        response = model.generate(
            pixel_values,
            prompt=prompt,
            max_new_tokens=200,
            temperature=0.3,
            do_sample=False,  # 좌표에는 탐욕적 디코딩 사용
        )
    
        # 응답에서 경계 상자와 다각형 추출
        bbox_norm = text_to_bbox(response)
        polygon_norm = text_to_polygon(response)
    
        # 좌표의 정규화 되돌리기
        bbox = None
        polygon = None
    
        if bbox_norm:
            bbox = [
                int(bbox_norm[0] / coord_scale * image_width),
                int(bbox_norm[1] / coord_scale * image_height),
                int(bbox_norm[2] / coord_scale * image_width),
                int(bbox_norm[3] / coord_scale * image_height),
            ]
    
        if polygon_norm:
            polygon = denormalize_polygon(polygon_norm, image_width, image_height, coord_scale)
    
        return {
            'bbox': bbox,
            'polygon': polygon,
            'response': response,
        }

    return execution_step_15, predict_referring_segmentation


@app.cell
def _(
    MPLPolygon,
    device,
    execution_step_15,
    image_processor,
    plt,
    predict_referring_segmentation,
    refcoco_samples,
    tokenizer,
    vlm,
):
    # 원본 코드 셀 순서와 모델·데이터 변경의 실행 순서를 보존합니다.
    execution_step_16 = execution_step_15 + 1
    # 샘플 테스트

    def visualize_prediction(image, refexp, prediction, title=''):
        """경계 상자와 다각형을 사용해 지칭 표현의 예측을 시각화합니다.
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        ax1.imshow(image)
        ax1.set_title(f"지칭 표현: '{refexp}'", fontsize=12)  # 원본 이미지
        ax1.axis('off')
        ax2.imshow(image)
        if prediction['bbox']:
            bbox = prediction['bbox']
            from matplotlib.patches import Rectangle  # 예측을 표시한 이미지
            rect = Rectangle((bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1], linewidth=3, edgecolor='red', facecolor='none', label='경계 상자')
            ax2.add_patch(rect)
        if prediction['polygon'] and len(prediction['polygon']) > 2:  # 경계 상자 그리기
            polygon = prediction['polygon']
            poly_patch = MPLPolygon(polygon, linewidth=2, edgecolor='lime', facecolor='lime', alpha=0.3, label='다각형')
            ax2.add_patch(poly_patch)
            xs, ys = zip(*polygon)
            ax2.plot(xs, ys, 'yo', markersize=4)
        ax2.set_title(f'예측(경계 상자 + 다각형)', fontsize=12)
        ax2.legend(loc='upper right')
        ax2.axis('off')
        if title:
            fig.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()
    test_indices = [0, 10, 50, 100]
    for idx in test_indices:
        if idx >= len(refcoco_samples):  # 다각형 그리기
            continue
        _sample = refcoco_samples[idx]
        image = _sample['image']
        if 'refexp' in _sample or 'sentences' in _sample:
            refexp = _sample.get('refexp', _sample.get('sentences', ['the object'])[0])
        elif 'objects' in _sample and len(_sample['objects'].get('category', [])) > 0:
            cat = _sample['objects']['category'][0]
            refexp = f'the {cat}'
        else:
            refexp = 'the object'
        prediction = predict_referring_segmentation(vlm, image, refexp, image_processor, tokenizer, device)
        visualize_prediction(image, refexp, prediction, title=f'Example {idx}')
        print(f'\n예시 {idx}:')  # 꼭짓점 그리기
        print(f'  지칭 표현: {refexp}')
        print(f'  경계 상자: {prediction['bbox']}')
        print(f'  다각형 꼭짓점: {(len(prediction['polygon']) if prediction['polygon'] else 0)}')
    # 테스트 예시
        print('=' * 70)  # 지칭 표현 추출 / COCO용 합성 표현 생성 / 예측 / 시각화
    return (execution_step_16,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7단계: 모델 저장하기
    """)
    return


@app.cell
def _(
    execution_step_16,
    image_processor,
    language_dim,
    lm_model_name,
    os,
    output_dir,
    tokenizer,
    torch,
    vision_dim,
    vision_model_name,
    vlm,
):
    # 원본 코드 셀 순서와 모델·데이터 변경의 실행 순서를 보존합니다.
    execution_step_17 = execution_step_16 + 1
    # 학습한 모델 저장
    save_dir = str(output_dir / "mini-vlm-refseg")
    os.makedirs(save_dir, exist_ok=True)

    torch.save({
        'projector_state_dict': vlm.projector.state_dict(),
        'language_model_state_dict': vlm.language_model.state_dict(),
        'config': {
            'vision_model_name': vision_model_name,
            'lm_model_name': lm_model_name,
            'vision_dim': vision_dim,
            'language_dim': language_dim,
        },
    }, f"{save_dir}/mini_vlm_refseg.pt")

    tokenizer.save_pretrained(f"{save_dir}/tokenizer")
    image_processor.save_pretrained(f"{save_dir}/image_processor")

    print(f'모델 저장 위치: {save_dir}/')
    print(f'저장 내용: {os.listdir(save_dir)}')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 정리

    **시각적 그라운딩과 다각형 분할 VLM**을 만들었습니다.

    ### 구현한 내용

    1. **지칭 표현 이해**: 자연어로 특정 객체 찾기
    2. **경계 상자 예측**: 상자로 객체 위치 표시
    3. **다각형 분할**: 꼭짓점 시퀀스로 윤곽 표시
    4. **통합 모델**: 한 모델로 세 작업 처리

    ### 주요 아이디어

    **다각형 표현:**
    - 텍스트 기반: `"<polygon> (x1,y1) (x2,y2) ... </polygon>"`
    - LLM에 적합: 좌표를 토큰으로 처리
    - 간결함: 픽셀 마스크의 1024개 이상 요소 대신 꼭짓점 40~50개
    - 해상도 독립적: 어떤 이미지 크기로도 조정 가능

    **좌표 정규화:**
    - 학습하기 쉽도록 [0, 1000] 범위로 조정
    - 이미지 크기가 달라도 일관된 표현
    - 토큰화하기 쉬운 정수 좌표 사용

    **지칭과 탐지 비교:**

    | 작업 | 입력 | 출력 |
    |------|-------|--------|
    | 탐지 | 이미지 | 모든 개 → 여러 상자 |
    | 지칭 | 이미지 + "red collar dog" | 상자 1개 + 다각형 |

    ### 구조의 발전 과정

    ```
    캡션:       [이미지] → "A dog playing"
                        ↓
    탐지:       [이미지] → {"dog": [x1,y1,x2,y2], ...}
                        ↓
    지칭:       [이미지] + "red collar dog" → [x1,y1,x2,y2]
                        ↓
    이 실습의 분할: [이미지] + "red collar dog" → [(x1,y1), (x2,y2), ...]
    ```

    ### 대표적인 고성능 모델과의 비교

    | 모델 | 접근 | 출력 | 매개변수 |
    |-------|----------|--------|------------|
    | **이 실습 모델** | 다각형 꼭짓점 | 텍스트 시퀀스 | 135M |
    | **PolyFormer** | 다각형 회귀 | 좌표 | 90M+ |
    | **SAM** | 마스크 디코더 | 픽셀 마스크 | 600M+ |
    | **Grounding DINO** | 교차 어텐션 | 경계 상자만 | 200M+ |

    ### 다각형 방식의 장점

    ✅ **간결함**: 마스크의 32×32×1 = 1024개 토큰 대신 꼭짓점 50개
    ✅ **해석 가능**: 꼭짓점을 시각화하고 편집 가능
    ✅ **확장 가능**: 어떤 해상도에서도 동작
    ✅ **LLM에 자연스러움**: 분할을 텍스트 생성으로 처리
    ✅ **멀티태스크**: 경계 상자와 같은 형식에서 점의 수만 확장

    ### 한계

    - **꼭짓점 수 제한**: 복잡한 모양에는 100개 이상 필요(LLM 문맥 길이 한계)
    - **구멍 표현 미지원**: 내부에 구멍이 있는 객체 표현이 어려움
    - **소형 모델**: 매개변수 135M으로 공간 추론 제한
    - **합성 데이터**: COCO 대체 경로에서 템플릿 기반 지칭 표현 사용
    - **여러 객체 미지원**: 지칭된 객체 하나만 분할

    ### 다음 단계

    1. **계층적 다각형**: 구멍·부분을 여러 다각형으로 표현
    2. **부분 분할**: "개의 머리를 분할하세요"
    3. **상호작용형 개선**: "다각형을 윤곽에 더 밀착시키세요"
    4. **영상 분할**: 프레임 사이의 다각형 추적
    5. **3차원 이해**: 다각형과 깊이 추정 결합
    6. **더 큰 모델**: SmolLM-1.7B나 Qwen2-VL로 확장

    ### 연구 배경

    **시각적 그라운딩:**
    - RefCOCO/RefCOCO+/RefCOCOg 벤치마크
    - MDETR, GLIP, Grounding DINO(대표적인 고성능 모델)

    **다각형 분할:**
    - PolyFormer(CVPR 2023)
    - Pix2Seq(Google, 2021)
    - E2E-VLP(다각형을 텍스트로 표현)

    **통합 모델:**
    - Unified-IO(2022): 텍스트에서 다양한 모달리티로 변환
    - Pix2Struct(2023): 구조 예측
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 참고 자료

    ### 데이터셋
    - [RefCOCO/RefCOCO+/RefCOCOg](https://github.com/lichengunc/refer): 지칭 표현 데이터셋
    - [COCO](https://cocodataset.org/): 인스턴스 분할

    ### 시각적 그라운딩
    - [Grounding DINO](https://arxiv.org/abs/2303.05499): 언어를 이용한 개방형 객체 탐지
    - [GLIP](https://arxiv.org/abs/2112.03857): 언어·이미지 사전 학습
    - [MDETR](https://arxiv.org/abs/2104.12763): 지칭을 위한 조건부 탐지

    ### 다각형 분할
    - [PolyFormer(CVPR 2023)](https://arxiv.org/abs/2302.07387): 다각형 기반 인스턴스 분할
    - [Pix2Seq(Google 2021)](https://arxiv.org/abs/2109.10852): 시퀀스 간 변환으로 객체 탐지
    - [E2E-VLP](https://arxiv.org/abs/2106.01804): 종단 간 시각·언어 사전 학습

    ### 통합 모델
    - [Unified-IO](https://arxiv.org/abs/2206.08916): 시각·언어·멀티모달 작업의 통합 모델
    - [Pix2Struct](https://arxiv.org/abs/2210.03347): 스크린샷 파싱을 통한 사전 학습

    ### 이전 노트북
    - [최소 구성의 VLM](part-01-minimal-vlm.py)
    - [객체 탐지](part-02-object-detection.py)
    - [멀티태스크 VLM](part-04-multi-task.py)
    - [사고 과정](part-07-chain-of-thought.py)
    """)
    return


if __name__ == "__main__":
    app.run()
