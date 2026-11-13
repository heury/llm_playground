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
    # 자연어로 이미지를 편집하도록 VLM 학습하기

    Nipun Batra · 2026-01-01
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 소개

    이미지를 설명하고, 객체를 탐지하고, 질문에 답하고, 다각형으로 객체 윤곽까지 표시하는 VLM을 만들었습니다. 이제 자연어로 이미지를 **수정**하려면 어떻게 해야 할까요?

    **다음과 같이 말한다고 생각해 보세요.**
    - "이 이미지를 흑백으로 바꿔줘"
    - "밝기를 30% 높여줘"
    - "시계 방향으로 90도 회전해 줘"
    - "따뜻한 느낌으로 바꿔줘"(주황색 색조 추가)
    - "이미지를 선명하게 해줘"

    그리고 VLM이 이 편집을 **자동으로 적용**합니다.

    ### 이전 노트북
    1. [최소 구성의 VLM](part-01-minimal-vlm.py): 캡션
    2. [객체 탐지](part-02-object-detection.py): 구조화된 출력
    3. [VQA](part-03-visual-qa.py): 질의응답
    4. [멀티태스크](part-04-multi-task.py): 작업 통합
    5. [사고 과정](part-07-chain-of-thought.py): 추론
    6. [시각적 그라운딩·분할](part-08-referring-segmentation.py): 다각형 출력

    ### 이번에 새로 다룰 내용

    **지시에 따른 이미지 편집:**
    - 입력: 이미지 + "make it grayscale"(흑백으로 바꿔줘)
    - VLM 출력: `{"operation": "grayscale", "params": {}}`
    - 적용: PIL/OpenCV로 연산 실행
    - 결과: 편집된 이미지

    ### 접근: 지시 → 매개변수 → 편집

    InstructPix2Pix나 확산 모델처럼 픽셀을 생성하는 대신 다음을 수행합니다.
    1. VLM으로 **지시 해석**
    2. JSON으로 **편집 매개변수 예측**
    3. PIL로 결정론적으로 **연산 적용**

    **이 방식을 사용하는 이유:**
    - ✅ **신뢰성**: 무작위성 없는 결정론적 처리
    - ✅ **속도**: 확산 단계 없이 즉시 편집
    - ✅ **소형 모델**: 이 실습의 135M VLM으로 동작
    - ✅ **해석 가능성**: 적용한 연산을 정확하게 확인
    - ✅ **조합 가능성**: 여러 연산을 이어 붙일 수 있음

    **비교:**

    | 접근 | 모델 크기 | 속도 | 제어 가능성 | 품질 |
    |----------|-----------|-------|-----------------|----------|
    | **확산(InstructPix2Pix)** | 1B+ | 느림(5~10초) | 중간 | 매우 높음 |
    | **GAN 기반** | 500M+ | 중간(1~2초) | 낮음 | 높음 |
    | **이 실습(매개변수 기반)** | 135M | 빠름(0.1초 미만) | 매우 높음 | 양호 |

    ### 지원하는 편집 연산

    **색상 조정:**
    - 그레이스케일, 세피아, 색상 반전
    - 밝기, 대비, 채도
    - 색온도(따뜻하게/차갑게)
    - 색조 이동

    **기하 변환:**
    - 회전, 뒤집기(좌우/상하)
    - 크기 조정, 자르기
    - 확대/축소

    **필터:**
    - 흐리게, 선명하게
    - 경계 검출
    - 엠보싱, 윤곽선

    **예술적 효과:**
    - 빈티지, 레트로
    - 높은 대비(포스터 효과)
    - 비네팅

    ### 구현할 내용

    1. **편집 연산 라이브러리**: PIL/Pillow 기반 이미지 연산
    2. **VLM 지시 해석기**: 자연어 → JSON 매개변수
    3. **합성 데이터셋**: COCO에서 학습 데이터 생성
    4. **학습**: 편집 연산을 예측하도록 VLM 학습
    5. **상호작용형 시연**: 편집 전후 비교
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
    from PIL import Image, ImageEnhance, ImageFilter, ImageOps
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
    from tqdm.auto import tqdm
    import random
    import json
    import warnings
    warnings.filterwarnings('ignore')


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
        ImageEnhance,
        ImageFilter,
        ImageOps,
        ViTImageProcessor,
        ViTModel,
        device,
        execution_step_2,
        json,
        load_dataset,
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
    ## 1단계: 이미지 편집 연산 라이브러리

    PIL(Pillow)로 자주 쓰는 이미지 편집 연산을 구현합니다.
    """)
    return


@app.cell
def _(Image, ImageEnhance, ImageFilter, ImageOps, execution_step_2, np):
    # 원본 코드 셀 순서와 모델·데이터 변경의 실행 순서를 보존합니다.
    execution_step_3 = execution_step_2 + 1
    class ImageEditor:
        """PIL을 사용하는 이미지 편집 연산입니다.
        
        각 연산은 이미지와 매개변수를 받아 편집된 이미지를 반환합니다.
        """
    
        @staticmethod
        def grayscale(image, **params):
            """그레이스케일로 변환합니다.
            """
            return ImageOps.grayscale(image).convert('RGB')
    
        @staticmethod
        def sepia(image, **params):
            """세피아 색조 효과를 적용합니다.
            """
            # 흑백으로 변환한 뒤 세피아 색조 적용
            gray = ImageOps.grayscale(image)
            # 세피아 변환 행렬
            sepia_img = Image.new('RGB', image.size)
            pixels = gray.load()
            sepia_pixels = sepia_img.load()
        
            for i in range(image.size[0]):
                for j in range(image.size[1]):
                    p = pixels[i, j]
                    # 세피아 색조 공식
                    r = min(255, int(p * 1.0))
                    g = min(255, int(p * 0.95))
                    b = min(255, int(p * 0.82))
                    sepia_pixels[i, j] = (r, g, b)
        
            return sepia_img
    
        @staticmethod
        def invert(image, **params):
            """색상을 반전합니다.
            """
            return ImageOps.invert(image)
    
        @staticmethod
        def brightness(image, factor=1.5, **params):
            """밝기를 조정합니다.
            factor < 1.0이면 어둡게, factor > 1.0이면 밝게 만듭니다.
            """
            enhancer = ImageEnhance.Brightness(image)
            return enhancer.enhance(factor)
    
        @staticmethod
        def contrast(image, factor=1.5, **params):
            """대비를 조정합니다.
            factor < 1.0이면 대비를 줄이고, factor > 1.0이면 높입니다.
            """
            enhancer = ImageEnhance.Contrast(image)
            return enhancer.enhance(factor)
    
        @staticmethod
        def saturation(image, factor=1.5, **params):
            """채도를 조정합니다.
            factor = 0이면 그레이스케일, factor > 1.0이면 채도를 높입니다.
            """
            enhancer = ImageEnhance.Color(image)
            return enhancer.enhance(factor)
    
        @staticmethod
        def sharpness(image, factor=2.0, **params):
            """선명도를 조정합니다.
            """
            enhancer = ImageEnhance.Sharpness(image)
            return enhancer.enhance(factor)
    
        @staticmethod
        def blur(image, radius=5, **params):
            """가우시안 흐림 효과를 적용합니다.
            """
            return image.filter(ImageFilter.GaussianBlur(radius=radius))
    
        @staticmethod
        def sharpen(image, **params):
            """이미지를 선명하게 만듭니다.
            """
            return image.filter(ImageFilter.SHARPEN)
    
        @staticmethod
        def edge_detect(image, **params):
            """경계를 검출합니다.
            """
            return image.filter(ImageFilter.FIND_EDGES)
    
        @staticmethod
        def emboss(image, **params):
            """엠보싱 효과를 적용합니다.
            """
            return image.filter(ImageFilter.EMBOSS)
    
        @staticmethod
        def contour(image, **params):
            """윤곽선 효과를 적용합니다.
            """
            return image.filter(ImageFilter.CONTOUR)
    
        @staticmethod
        def rotate(image, angle=90, **params):
            """지정한 각도(도)만큼 이미지를 회전합니다.
            """
            return image.rotate(angle, expand=True)
    
        @staticmethod
        def flip_horizontal(image, **params):
            """좌우로 뒤집습니다.
            """
            return ImageOps.mirror(image)
    
        @staticmethod
        def flip_vertical(image, **params):
            """상하로 뒤집습니다.
            """
            return ImageOps.flip(image)
    
        @staticmethod
        def resize(image, scale=0.5, **params):
            """배율에 따라 이미지 크기를 조정합니다.
            """
            new_size = (int(image.width * scale), int(image.height * scale))
            return image.resize(new_size, Image.LANCZOS)
    
        @staticmethod
        def warm(image, strength=0.3, **params):
            """따뜻한 주황색 색조를 더합니다.
            """
            # 빨강을 늘리고 초록을 조금 더하며 파랑을 줄임
            arr = np.array(image).astype(np.float32)
            arr[:, :, 0] = np.clip(arr[:, :, 0] * (1 + strength), 0, 255)  # 빨강
            arr[:, :, 1] = np.clip(arr[:, :, 1] * (1 + strength * 0.5), 0, 255)  # 초록
            arr[:, :, 2] = np.clip(arr[:, :, 2] * (1 - strength * 0.3), 0, 255)  # 파랑
            return Image.fromarray(arr.astype(np.uint8))
    
        @staticmethod
        def cool(image, strength=0.3, **params):
            """차가운 파란색 색조를 더합니다.
            """
            arr = np.array(image).astype(np.float32)
            arr[:, :, 0] = np.clip(arr[:, :, 0] * (1 - strength * 0.3), 0, 255)  # 빨강
            arr[:, :, 1] = np.clip(arr[:, :, 1] * (1 + strength * 0.2), 0, 255)  # 초록
            arr[:, :, 2] = np.clip(arr[:, :, 2] * (1 + strength), 0, 255)  # 파랑
            return Image.fromarray(arr.astype(np.uint8))
    
        @staticmethod
        def posterize(image, bits=4, **params):
            """색 깊이를 줄입니다(포스터 효과).
            """
            return ImageOps.posterize(image, bits)
    
        @staticmethod
        def solarize(image, threshold=128, **params):
            """솔라리제이션 효과: 임계값보다 밝은 픽셀을 반전합니다.
            """
            return ImageOps.solarize(image, threshold)
    
        # 모든 연산의 딕셔너리
        OPERATIONS = {
            'grayscale': grayscale.__func__,
            'sepia': sepia.__func__,
            'invert': invert.__func__,
            'brightness': brightness.__func__,
            'contrast': contrast.__func__,
            'saturation': saturation.__func__,
            'sharpness': sharpness.__func__,
            'blur': blur.__func__,
            'sharpen': sharpen.__func__,
            'edge_detect': edge_detect.__func__,
            'emboss': emboss.__func__,
            'contour': contour.__func__,
            'rotate': rotate.__func__,
            'flip_horizontal': flip_horizontal.__func__,
            'flip_vertical': flip_vertical.__func__,
            'resize': resize.__func__,
            'warm': warm.__func__,
            'cool': cool.__func__,
            'posterize': posterize.__func__,
            'solarize': solarize.__func__,
        }
    
        @classmethod
        def apply_operation(cls, image, operation, params=None):
            """이미지에 편집 연산을 적용합니다.
            
            인자:
                image: PIL 이미지
                operation: 연산 이름 문자열
                params: 연산 매개변수 딕셔너리
            
            반환:
                편집된 PIL 이미지
            """
            if operation not in cls.OPERATIONS:
                raise ValueError(f"Unknown operation: {operation}")
        
            params = params or {}
            op_func = cls.OPERATIONS[operation]
            return op_func(image, **params)


    print(f'이미지 편집기 초기화, 연산 수: {len(ImageEditor.OPERATIONS)}개 연산:')
    print(list(ImageEditor.OPERATIONS.keys()))
    return ImageEditor, execution_step_3


@app.cell
def _(Image, ImageEditor, execution_step_3, plt):
    # 원본 코드 셀 순서와 모델·데이터 변경의 실행 순서를 보존합니다.
    execution_step_4 = execution_step_3 + 1
    # 샘플 이미지로 편집기 테스트
    # 간단한 테스트 이미지 생성
    test_img = Image.new('RGB', (200, 200), color=(100, 150, 200))
    _fig, _axes = plt.subplots(3, 4, figsize=(16, 12))
    # 여러 연산 시도
    _axes = _axes.flatten()
    operations_to_test = [('original', {}), ('grayscale', {}), ('sepia', {}), ('brightness', {'factor': 1.5}), ('contrast', {'factor': 2.0}), ('warm', {'strength': 0.4}), ('cool', {'strength': 0.4}), ('blur', {'radius': 3}), ('sharpen', {}), ('edge_detect', {}), ('emboss', {}), ('posterize', {'bits': 3})]
    for _idx, (op_name, _params) in enumerate(operations_to_test):
        if op_name == 'original':
            result = test_img
        else:
            result = ImageEditor.apply_operation(test_img, op_name, _params)
        _axes[_idx].imshow(result)
        _axes[_idx].set_title(f'{op_name}\n{_params}', fontsize=10)
        _axes[_idx].axis('off')
    plt.suptitle('이미지 편집 연산 시연', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
    print('✓ 이미지 편집기 테스트를 마쳤습니다.')
    return (execution_step_4,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2단계: 지시 템플릿

    연산별 자연어 템플릿을 만듭니다.
    """)
    return


@app.cell
def _(execution_step_4, random):
    # 원본 코드 셀 순서와 모델·데이터 변경의 실행 순서를 보존합니다.
    execution_step_5 = execution_step_4 + 1
    # 학습 데이터 생성용 지시 템플릿
    INSTRUCTION_TEMPLATES = {'grayscale': ['make it grayscale', 'convert to black and white', 'remove all colors', 'make it monochrome', 'turn into grayscale'], 'sepia': ['apply sepia tone', 'make it look vintage', 'add sepia effect', 'give it an old photo look'], 'invert': ['invert the colors', 'make a negative', 'reverse all colors'], 'brightness': [('make it brighter', {'factor': 1.5}), ('increase brightness', {'factor': 1.4}), ('brighten the image', {'factor': 1.6}), ('make it darker', {'factor': 0.6}), ('decrease brightness', {'factor': 0.7}), ('darken the image', {'factor': 0.5})], 'contrast': [('increase contrast', {'factor': 1.8}), ('make it more contrasty', {'factor': 2.0}), ('boost contrast', {'factor': 1.6}), ('reduce contrast', {'factor': 0.6}), ('lower the contrast', {'factor': 0.7})], 'saturation': [('increase saturation', {'factor': 1.6}), ('make colors more vivid', {'factor': 1.8}), ('boost saturation', {'factor': 1.5}), ('desaturate', {'factor': 0.5}), ('reduce saturation', {'factor': 0.6})], 'blur': [('blur the image', {'radius': 5}), ('make it blurry', {'radius': 7}), ('apply blur', {'radius': 4}), ('add blur effect', {'radius': 6})], 'sharpen': ['sharpen the image', 'make it sharper', 'increase sharpness', 'enhance details'], 'edge_detect': ['detect edges', 'show only edges', 'edge detection', 'find edges'], 'emboss': ['apply emboss effect', 'make it embossed', 'emboss the image'], 'rotate': [('rotate 90 degrees', {'angle': 90}), ('rotate clockwise', {'angle': 90}), ('rotate 180 degrees', {'angle': 180}), ('rotate counterclockwise', {'angle': -90}), ('turn upside down', {'angle': 180})], 'flip_horizontal': ['flip horizontally', 'mirror the image', 'flip left to right', 'horizontal flip'], 'flip_vertical': ['flip vertically', 'flip upside down', 'flip top to bottom', 'vertical flip'], 'resize': [('make it smaller', {'scale': 0.5}), ('shrink the image', {'scale': 0.6}), ('make it half size', {'scale': 0.5}), ('resize to 75%', {'scale': 0.75})], 'warm': [('make it warmer', {'strength': 0.3}), ('add warm tones', {'strength': 0.4}), ('add orange tint', {'strength': 0.35}), ('warm color temperature', {'strength': 0.3})], 'cool': [('make it cooler', {'strength': 0.3}), ('add cool tones', {'strength': 0.4}), ('add blue tint', {'strength': 0.35}), ('cool color temperature', {'strength': 0.3})], 'posterize': [('posterize effect', {'bits': 4}), ('reduce colors', {'bits': 3}), ('poster art style', {'bits': 4})]}

    def get_random_instruction(operation):
        """연산에 해당하는 지시 템플릿을 무작위로 선택합니다.
        
        반환:
            (instruction_text, params_dict)
        """
        templates = INSTRUCTION_TEMPLATES.get(operation, [])
        if not templates:
            return (f'apply {operation}', {})
        choice = random.choice(templates)
        if isinstance(choice, tuple):
            return choice
        else:
            return (choice, {})
    print('지시 예시:')
    for op in ['grayscale', 'brightness', 'rotate', 'warm']:
        instr, _params = get_random_instruction(op)
    # 테스트
        print(f"  {op}: '{instr}' → {_params}")
    return INSTRUCTION_TEMPLATES, execution_step_5, get_random_instruction


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3단계: 기반 VLM 불러오기
    """)
    return


@app.cell
def _(
    AutoModelForCausalLM,
    AutoTokenizer,
    F,
    ViTModel,
    execution_step_5,
    nn,
    torch,
):
    # 원본 코드 셀 순서와 모델·데이터 변경의 실행 순서를 보존합니다.
    execution_step_6 = execution_step_5 + 1
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
            max_new_tokens = 100,
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

    return MiniVLM, VisionProjector, execution_step_6


@app.cell
def _(
    AutoModelForCausalLM,
    AutoTokenizer,
    MiniVLM,
    ViTImageProcessor,
    ViTModel,
    VisionProjector,
    device,
    execution_step_6,
    os,
    output_dir,
    torch,
):
    # 원본 코드 셀 순서와 모델·데이터 변경의 실행 순서를 보존합니다.
    execution_step_7 = execution_step_6 + 1
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
    return (execution_step_7, image_processor, language_dim, lm_model_name, tokenizer, vision_dim, vision_model_name, vlm)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4단계: 합성 학습 데이터셋 만들기

    학습 데이터 생성 과정은 다음과 같습니다.
    1. COCO/Flickr에서 이미지 불러오기
    2. 편집 연산을 무작위로 선택하기
    3. 지시문 생성하기
    4. 연산과 매개변수를 JSON 출력으로 만들기
    """)
    return


@app.cell
def _(execution_step_7, load_dataset):
    # 원본 코드 셀 순서와 모델·데이터 변경의 실행 순서를 보존합니다.
    execution_step_8 = execution_step_7 + 1
    # 이미지 데이터셋 로딩
    print('이미지 데이터셋 로딩 중...')
    try:
        coco_dataset = load_dataset('detection-datasets/coco', split='train', streaming=True)
        image_samples = []  # 먼저 COCO 시도
        for _i, sample in enumerate(coco_dataset):
            if _i >= 2000:
                break
            image_samples.append(sample)
        print(f'불러온 수: {len(image_samples)}장 COCO 이미지')
    except:
        flickr = load_dataset('jxie/flickr8k', split='train')
        image_samples = [{'image': flickr[i]['image']} for i in range(min(2000, len(flickr)))]
        print(f'불러온 수: {len(image_samples)}장 Flickr8k 이미지')  # 대체 경로로 Flickr8k 사용
    return execution_step_8, image_samples


@app.cell
def _(
    DataLoader,
    Dataset,
    INSTRUCTION_TEMPLATES,
    execution_step_8,
    get_random_instruction,
    image_processor,
    image_samples,
    json,
    random,
    tokenizer,
):
    # 원본 코드 셀 순서와 모델·데이터 변경의 실행 순서를 보존합니다.
    execution_step_9 = execution_step_8 + 1
    class ImageEditingDataset(Dataset):
        """이미지 편집 지시용 합성 데이터셋입니다.
        
        형식:
        입력: [이미지] + "make it grayscale"
        출력: '{"operation": "grayscale", "params": {}}' 
        """
    
        def __init__(self, image_samples, image_processor, tokenizer, max_length=256):
            self.image_samples = image_samples
            self.image_processor = image_processor
            self.tokenizer = tokenizer
            self.max_length = max_length
        
            # 샘플링할 연산 목록
            self.operations = list(INSTRUCTION_TEMPLATES.keys())
    
        def __len__(self):
            return len(self.image_samples)
    
        def __getitem__(self, idx):
            sample = self.image_samples[idx]
        
            # 이미지 추출
            image = sample['image'].convert('RGB')
            pixel_values = self.image_processor(image, return_tensors="pt").pixel_values.squeeze(0)
        
            # 연산 무작위 선택
            operation = random.choice(self.operations)
            instruction, params = get_random_instruction(operation)
        
            # JSON 출력 생성
            output_json = json.dumps({
                'operation': operation,
                'params': params
            }, separators=(',', ':'))  # 간결한 JSON
        
            # 학습 예시 구성
            prompt = f"Edit instruction: {instruction}\nEdit:"
            response = output_json
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
        
            # 프롬프트 마스킹(JSON 출력만 학습)
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


    # 데이터셋 생성
    edit_dataset = ImageEditingDataset(image_samples, image_processor, tokenizer)

    edit_loader = DataLoader(
        edit_dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0,
    )

    print(f'이미지 편집 데이터셋: {len(edit_dataset)}개 샘플')
    print(f'배치 수: {len(edit_loader)}')
    return edit_dataset, edit_loader, execution_step_9


@app.cell
def _(edit_dataset, execution_step_9, tokenizer):
    # 원본 코드 셀 순서와 모델·데이터 변경의 실행 순서를 보존합니다.
    execution_step_10 = execution_step_9 + 1
    # 데이터셋 형식 확인
    sample_item = edit_dataset[0]
    decoded = tokenizer.decode(sample_item['input_ids'], skip_special_tokens=True)
    print('학습 형식 샘플:')
    print(decoded[:200])
    print("...")
    return (execution_step_10,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5단계: 학습

    자연어 지시에서 편집 연산을 예측하도록 VLM을 학습합니다.
    """)
    return


@app.cell
def _(device, execution_step_10, os, output_dir, torch, tqdm):
    # 원본 코드 셀 순서와 모델·데이터 변경의 실행 순서를 보존합니다.
    execution_step_11 = execution_step_10 + 1
    def train_image_editing(model, train_loader, num_epochs=6, lr=1e-4, checkpoint_path=str(output_dir / "edit_checkpoint.pt")):
        """이미지 편집 VLM을 학습합니다.
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

    return execution_step_11, train_image_editing


@app.cell
def _(edit_loader, execution_step_11, train_image_editing, vlm):
    # 원본 코드 셀 순서와 모델·데이터 변경의 실행 순서를 보존합니다.
    execution_step_12 = execution_step_11 + 1
    # 모델 학습
    print('이미지 편집 VLM 학습 중...\n')
    losses = train_image_editing(vlm, edit_loader, num_epochs=6, lr=1e-4)
    return execution_step_12, losses


@app.cell
def _(execution_step_12, losses, plt):
    # 원본 코드 셀 순서와 모델·데이터 변경의 실행 순서를 보존합니다.
    execution_step_13 = execution_step_12 + 1
    # 학습 손실 그래프
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(losses)+1), losses, marker='o', linewidth=2, markersize=8)
    plt.xlabel('에포크', fontsize=12)
    plt.ylabel('손실', fontsize=12)
    plt.title('이미지 편집 학습 손실', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    return (execution_step_13,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6단계: 이미지 편집 테스트

    모델이 지시에 따라 이미지를 편집하는지 확인합니다.
    """)
    return


@app.cell
def _(Image, ImageEditor, execution_step_13, json):
    # 원본 코드 셀 순서와 모델·데이터 변경의 실행 순서를 보존합니다.
    execution_step_14 = execution_step_13 + 1
    def predict_edit_operation(model, image, instruction, image_processor, tokenizer, device):
        """지시에서 편집 연산을 예측합니다.
        
        반환:
            'operation'과 'params'를 담은 딕셔너리
        """
        model.eval()
    
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        image = image.convert('RGB')
    
        pixel_values = image_processor(image, return_tensors="pt").pixel_values.to(device)
    
        prompt = f"Edit instruction: {instruction}\nEdit:"
    
        response = model.generate(
            pixel_values,
            prompt=prompt,
            max_new_tokens=50,
            temperature=0.3,
            do_sample=False,  # JSON에는 탐욕적 디코딩 사용
        )
    
        # JSON 문법뿐 아니라 편집 응답의 자료형과 지원 연산도 확인합니다.
        fallback = {'operation': 'grayscale', 'params': {}}
        if not isinstance(response, str):
            return fallback

        # 응답에서 JSON 추출
        try:
            # 응답 형식: "Edit instruction: ... Edit: {json}"
            if "Edit:" in response:
                json_str = response.split("Edit:")[-1].strip()
            else:
                json_str = response
        
            # JSON 파싱
            edit_op = json.loads(json_str)
        except json.JSONDecodeError:
            return fallback

        if not isinstance(edit_op, dict):
            return fallback
        operation = edit_op.get('operation')
        params = edit_op.get('params', {})
        if not isinstance(operation, str) or operation not in ImageEditor.OPERATIONS:
            return fallback
        if not isinstance(params, dict):
            return fallback
        return {'operation': operation, 'params': params}


    def edit_image_with_instruction(model, image, instruction, image_processor, tokenizer, device):
        """전체 파이프라인: 지시 → 연산 예측 → 편집 적용.
        """
        # 연산 예측
        edit_op = predict_edit_operation(model, image, instruction, image_processor, tokenizer, device)
    
        # 연산 적용
        operation = edit_op.get('operation', 'grayscale')
        params = edit_op.get('params', {})
    
        try:
            edited_image = ImageEditor.apply_operation(image, operation, params)
            return edited_image, edit_op
        except Exception as e:
            print(f'연산 적용 오류: {e}')
            return image, edit_op

    return edit_image_with_instruction, execution_step_14


@app.cell
def _(
    device,
    edit_image_with_instruction,
    execution_step_14,
    image_processor,
    image_samples,
    plt,
    random,
    tokenizer,
    vlm,
):
    # 원본 코드 셀 순서와 모델·데이터 변경의 실행 순서를 보존합니다.
    execution_step_15 = execution_step_14 + 1
    # 샘플 이미지 테스트
    test_instructions = ['make it grayscale', 'increase brightness', 'make it warmer', 'blur the image', 'rotate 90 degrees', 'flip horizontally']
    test_indices = [0, 10, 50, 100]
    for _idx in test_indices[:2]:
        if _idx >= len(image_samples):
            continue
        original_image = image_samples[_idx]['image'].convert('RGB')
        sample_instructions = random.sample(test_instructions, 3)
        _fig, _axes = plt.subplots(1, 4, figsize=(20, 5))
        _axes[0].imshow(original_image)
        _axes[0].set_title('원본', fontsize=12, fontweight='bold')
        _axes[0].axis('off')
        for _i, _instruction in enumerate(sample_instructions):  # 이미지 2장 테스트
            _edited, _edit_op = edit_image_with_instruction(vlm, original_image, _instruction, image_processor, tokenizer, device)
            _axes[_i + 1].imshow(_edited)
            _title = f"'{_instruction}'\n{_edit_op['operation']}"
            if _edit_op['params']:
                _title += f"\n{_edit_op['params']}"
            _axes[_i + 1].set_title(_title, fontsize=10)  # 무작위 지시 3개 시도
            _axes[_i + 1].axis('off')
        plt.suptitle(f'이미지 편집 시연 - 샘플 {_idx}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()
        print('=' * 70)  # 원본 / 편집 결과
    return (execution_step_15,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7단계: 상호작용형 갤러리

    다양한 편집 연산의 적용 전후를 비교합니다.
    """)
    return


@app.cell
def _(
    device,
    edit_image_with_instruction,
    execution_step_15,
    image_processor,
    image_samples,
    plt,
    tokenizer,
    vlm,
):
    # 원본 코드 셀 순서와 모델·데이터 변경의 실행 순서를 보존합니다.
    execution_step_16 = execution_step_15 + 1
    # 전체 연산 갤러리 생성
    gallery_image = image_samples[5]['image'].convert('RGB')
    gallery_instructions = ['make it grayscale', 'make it brighter', 'increase saturation', 'make it warmer', 'blur the image', 'sharpen the image', 'rotate 90 degrees', 'flip horizontally']
    _fig, _axes = plt.subplots(3, 3, figsize=(18, 18))
    _axes = _axes.flatten()
    _axes[0].imshow(gallery_image)
    _axes[0].set_title('원본 이미지', fontsize=14, fontweight='bold', color='green')
    _axes[0].axis('off')
    for _i, _instruction in enumerate(gallery_instructions):
        _edited, _edit_op = edit_image_with_instruction(vlm, gallery_image, _instruction, image_processor, tokenizer, device)
        _axes[_i + 1].imshow(_edited)
        _title = f'{_instruction.upper()}\n'
        _title += f"Operation: {_edit_op['operation']}"
        if _edit_op['params']:
            params_str = ', '.join([f'{k}={v}' for k, v in _edit_op['params'].items()])
            _title += f'\nParams: {params_str}'
        _axes[_i + 1].set_title(_title, fontsize=10)
    # 원본
        _axes[_i + 1].axis('off')
    plt.suptitle('이미지 편집 갤러리: 자연어 → 자동 편집', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    # 편집 결과
    plt.show()
    return (execution_step_16,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 8단계: 모델 저장하기
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
    save_dir = str(output_dir / "mini-vlm-edit")
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
    }, f"{save_dir}/mini_vlm_edit.pt")

    tokenizer.save_pretrained(f"{save_dir}/tokenizer")
    image_processor.save_pretrained(f"{save_dir}/image_processor")

    print(f'모델 저장 위치: {save_dir}/')
    print(f'저장 내용: {os.listdir(save_dir)}')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 정리

    자연어 지시를 이해하는 **이미지 편집 VLM**을 만들었습니다.

    ### 구현한 내용

    1. **이미지 편집 라이브러리**: 그레이스케일, 흐림, 회전, 따뜻한 색조 등 20개 이상의 연산
    2. **지시 해석기**: 자연어에서 편집 연산을 예측하는 VLM
    3. **합성 데이터셋**: 2,000개 이상의 학습 예시 생성
    4. **종단 간 파이프라인**: 지시 → JSON → 연산 적용

    ### 구조: 지시 → 매개변수 → 편집

    ```
    입력: [이미지] + "make it warmer"(더 따뜻하게)
           ↓
    VLM 예측: {"operation": "warm", "params": {"strength": 0.3}}
           ↓
    PIL 적용: ImageEditor.warm(image, strength=0.3)
           ↓
    출력: [주황색 색조가 추가된 이미지]
    ```

    ### 주요 설계 결정

    **픽셀 생성 대신 매개변수 기반 편집을 쓰는 이유는?**

    | 측면 | 이 실습의 접근 | 확산(InstructPix2Pix) |
    |--------|--------------|-----------------------------|
    | **모델 크기** | 매개변수 135M | 매개변수 1B+ |
    | **속도** | 0.1초 미만(즉시) | 이미지당 5~10초 |
    | **신뢰성** | 결정론적 | 확률적 |
    | **제어 가능성** | 정확한 매개변수 | 프롬프트 기반(가변적) |
    | **품질** | 기하 변환에 양호 | 복잡한 편집에 우수 |
    | **학습 데이터** | 합성 2천 개 | 수백만 개 예시 |

    **이 실습의 접근에 적합한 작업:**
    - ✅ 색상 조정, 필터, 기하 변환
    - ✅ 일관되고 반복 가능한 편집
    - ✅ 실시간 애플리케이션
    - ✅ 소형 모델 배포
    - ✅ 해석 가능한 연산

    **확산 모델에 적합한 작업:**
    - ✅ "사람에게 모자를 씌워줘" 같은 복잡한 의미적 변경
    - ✅ 스타일 변환
    - ✅ 인페인팅, 아웃페인팅
    - ✅ 사실적인 사진 편집

    ### 지원하는 연산(20개 이상)

    **색상 조정:**
    - 그레이스케일, 세피아, 반전
    - 밝기, 대비, 채도, 선명도
    - 따뜻하거나 차가운 색온도

    **기하 변환:**
    - 회전, 좌우·상하 뒤집기
    - 크기 조정, 자르기

    **필터:**
    - 흐리게, 선명하게
    - 경계 검출, 엠보싱, 윤곽선
    - 포스터화, 솔라리제이션

    ### 지시 예시

    아래 영어 문자열은 실제 학습 형식입니다.

    ```text
    "make it grayscale"          → {"operation": "grayscale"}
    "increase brightness"        → {"operation": "brightness", "params": {"factor": 1.5}}
    "make it warmer"             → {"operation": "warm", "params": {"strength": 0.3}}
    "blur the image"             → {"operation": "blur", "params": {"radius": 5}}
    "rotate 90 degrees"          → {"operation": "rotate", "params": {"angle": 90}}
    "flip horizontally"          → {"operation": "flip_horizontal"}
    ```

    순서대로 흑백 변환, 밝기 증가, 따뜻한 색조, 흐림, 90도 회전, 좌우 뒤집기를 뜻합니다.

    ### 학습 전략

    **합성 데이터 생성:**
    1. COCO/Flickr에서 이미지 2,000장 불러오기
    2. 각 이미지마다 연산을 무작위로 선택하기
    3. 템플릿에서 자연어 지시문 생성하기
    4. 연산과 매개변수를 JSON 출력으로 만들기
    5. 지시에서 JSON을 예측하도록 VLM 학습하기

    **효과적인 이유:**
    - 실제 편집 전후 이미지 쌍이 필요하지 않음
    - 학습 데이터를 제한 없이 생성 가능
    - 연산이 결정론적이므로 정답이 명확함
    - 모델이 연산 → 매개변수 대응을 학습

    ### 한계

    - **의미적 이해 없음**: 모자 추가나 객체 제거 불가
    - **매개변수화된 연산으로 제한**: 새로운 연산 자체를 학습할 수 없음
    - **템플릿 기반**: 지시가 학습 템플릿과 맞아야 함
    - **조합 미지원**: 한 지시에서 여러 편집을 이어서 수행할 수 없음
    - **기본 PIL 연산**: Photoshop/GIMP에 비해 제한적

    ### 다음 단계

    1. **다단계 편집**: "흑백으로 바꾼 다음 대비를 높여줘"처럼 연산 연결
    2. **매개변수 학습**: VLM이 정확한 매개변수 값 예측
    3. **맞춤 연산**: 비네팅, 렌즈 흐림 등 고급 필터 추가
    4. **혼합 방식**: 의미적 편집에 확산 모델 결합
    5. **상호작용형 개선**: "파란색을 더해줘"로 이전 편집 조정
    6. **압축**: 다음 노트북에서 엣지 배포용 모델 양자화

    ### 관련 연구와의 비교

    | 모델 | 접근 | 연도 |
    |-------|----------|------|
    | **InstructPix2Pix** | 확산 기반 | 2023 |
    | **ControlNet** | 조건부 확산 | 2023 |
    | **Pix2Pix** | GAN 기반 | 2017 |
    | **이 실습 모델** | 매개변수 기반 편집 | 2026 |

    **이 접근의 특징:**
    - 가벼움(매개변수 135M 대 1B+)
    - 빠름(수 초 대신 즉시 처리)
    - 해석 가능(JSON 연산)
    - 신뢰성(결정론적 처리)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 참고 자료

    ### 이미지 편집 모델
    - [InstructPix2Pix](https://arxiv.org/abs/2211.09800): 확산 기반 지시 편집
    - [ControlNet](https://arxiv.org/abs/2302.05543): 조건부 확산 제어
    - [Pix2Pix](https://arxiv.org/abs/1611.07004): 이미지 간 변환
    - [CLIP](https://arxiv.org/abs/2103.00020): 시각·언어 사전 학습

    ### 이미지 처리
    - [Pillow(PIL)](https://pillow.readthedocs.io/): Python 이미지 라이브러리
    - [ImageEnhance](https://pillow.readthedocs.io/en/stable/reference/ImageEnhance.html): 색상·대비·밝기
    - [ImageFilter](https://pillow.readthedocs.io/en/stable/reference/ImageFilter.html): 필터와 효과

    ### 이전 노트북
    - [최소 구성의 VLM](part-01-minimal-vlm.py)
    - [멀티태스크 VLM](part-04-multi-task.py)
    - [사고 과정](part-07-chain-of-thought.py)
    - [시각적 그라운딩·분할](part-08-referring-segmentation.py)
    """)
    return


if __name__ == "__main__":
    app.run()
