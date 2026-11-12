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
    return (execution_step_0,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # VLM 유도 이미지 생성: 텍스트·이미지에서 픽셀로

    Nipun Batra · 2026-01-03
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # VLM 유도 이미지 생성

    이 노트북에서는 시각·언어 모델(VLM)이 이미지 이해를 넘어 **이미지 생성**에도 어떻게 쓰이는지 살펴봅니다.

    1. **텍스트 → 이미지 생성**: Stable Diffusion으로 텍스트 프롬프트에서 이미지 생성
    2. **VLM 유도 개선**: CLIP/VLM으로 생성 품질을 평가하고 개선
    3. **이미지 조건부 생성**: 입력 이미지에 기반한 편집, 인페인팅, 변형
    4. **반복 개선**: VLM 비평 → 재생성 → 개선

    ## VLM으로 생성을 유도하는 이유

    Stable Diffusion이나 DALL-E 같은 텍스트 → 이미지 모델은 텍스트만으로 이미지를 생성합니다. VLM을 활용하면 다음이 가능합니다.
    - **품질 확인**: 생성 이미지가 프롬프트와 일치하는지 확인
    - **피드백 제공**: "세부 묘사 추가", "색상이 틀림" 등의 개선안 제안
    - **반복 개선**: 생성 → 비평 → 수정 → 반복
    - **멀티모달 조건**: 텍스트와 참조 이미지 함께 사용

    ## 구조 개요

    ```
    ┌─────────────────────────────────────────────┐
    │            VLM 유도 이미지 생성 파이프라인      │
    └─────────────────────────────────────────────┘

    입력: "빨간 모자를 쓴 고양이"
       ↓
    [Stable Diffusion] → 생성 이미지 v1
       ↓
    [VLM 비평기] → "모자가 빨간색이 아니라 파란색입니다. 고양이는 좋습니다."
       ↓
    [프롬프트 개선] → "선명한 빨간 모자를 쓴 고양이, 사실적인 모습"
       ↓
    [Stable Diffusion] → 개선된 생성 이미지 v2
       ↓
    [VLM 비평기] → "완벽하게 일치합니다!"
    ```

    이제 실습을 시작합니다.
    """)
    return


@app.cell
def _(execution_step_0):
    # 원본 코드 셀 순서와 모델·데이터 변경의 실행 순서를 보존합니다.
    execution_step_1 = execution_step_0 + 1
    import os

    return (execution_step_1,)


@app.cell
def _(execution_step_1):
    # 원본 코드 셀 순서와 모델·데이터 변경의 실행 순서를 보존합니다.
    execution_step_2 = execution_step_1 + 1
    import torch
    import torch.nn as nn
    from diffusers import StableDiffusionPipeline, StableDiffusionInpaintPipeline, DPMSolverMultistepScheduler
    from transformers import AutoProcessor, CLIPModel, AutoTokenizer, AutoModel, AutoImageProcessor
    from PIL import Image, ImageDraw, ImageFont
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
    from tqdm import tqdm
    import gc

    print(f'PyTorch 버전: {torch.__version__}')
    print(f'CUDA 사용 가능 여부: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'CUDA 장치: {torch.cuda.get_device_name(0)}')
        print(f'GPU 메모리: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.2f} GB')
    return (
        AutoProcessor,
        CLIPModel,
        DPMSolverMultistepScheduler,
        Image,
        ImageDraw,
        StableDiffusionInpaintPipeline,
        StableDiffusionPipeline,
        execution_step_2,
        plt,
        torch,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Stable Diffusion으로 텍스트에서 이미지 생성

    대표적인 오픈소스 텍스트 → 이미지 모델 **Stable Diffusion**으로 시작합니다. 빠른 생성을 위해 더 작은 **SD 1.5**를 사용합니다.
    """)
    return


@app.cell
def _(
    DPMSolverMultistepScheduler,
    StableDiffusionPipeline,
    execution_step_2,
    torch,
):
    # 원본 코드 셀 순서와 모델·데이터 변경의 실행 순서를 보존합니다.
    execution_step_3 = execution_step_2 + 1
    # Stable Diffusion 1.5 로딩(작고 빠름)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print('Stable Diffusion 1.5 로딩 중...')
    sd_pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        safety_checker=None,  # 속도를 위해 비활성화
    )

    # 더 빠른 스케줄러 사용(DPM++ 2M)
    sd_pipe.scheduler = DPMSolverMultistepScheduler.from_config(sd_pipe.scheduler.config)
    sd_pipe = sd_pipe.to(device)

    # 메모리 최적화 활성화
    if torch.cuda.is_available():
        sd_pipe.enable_attention_slicing()
        sd_pipe.enable_vae_slicing()

    print('Stable Diffusion을 불러왔습니다.')
    return device, execution_step_3, sd_pipe


@app.cell
def _(device, execution_step_3, plt, sd_pipe, torch):
    # 원본 코드 셀 순서와 모델·데이터 변경의 실행 순서를 보존합니다.
    execution_step_4 = execution_step_3 + 1

    def generate_image(prompt, num_inference_steps=25, guidance_scale=7.5, seed=42):
        """Stable Diffusion으로 텍스트 프롬프트에서 이미지를 생성합니다.
        """
        generator = torch.Generator(device=device).manual_seed(seed)
        with torch.inference_mode():
            image = sd_pipe(prompt=prompt, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale, generator=generator).images[0]
        return image
    test_prompts = ['a photograph of a cat wearing a red hat, high quality', 'a beautiful sunset over mountains, realistic photography', 'a robot reading a book in a library, digital art']
    _fig, _axes = plt.subplots(1, 3, figsize=(15, 5))
    for _i, _prompt in enumerate(test_prompts):
        print(f'생성 중: {_prompt}')
        image = generate_image(_prompt, seed=42 + _i)
        _axes[_i].imshow(image)
        _axes[_i].set_title(_prompt[:40] + '...', fontsize=10)
    # 기본 생성 테스트
        _axes[_i].axis('off')
    plt.tight_layout()
    plt.show()
    return execution_step_4, generate_image


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. 품질 평가용 VLM 비평기

    다음을 수행하는 **VLM 비평기**를 만듭니다.
    1. 생성 이미지가 프롬프트와 일치하는지 확인
    2. 색상 오류나 객체 누락 같은 문제 식별
    3. 개선을 위한 피드백 제공

    이미지·텍스트 유사도 점수에는 **CLIP**을 사용하며, 상세한 비평에는 언어 모델을 추가로 활용할 수 있습니다.
    """)
    return


@app.cell
def _(AutoProcessor, CLIPModel, device, execution_step_4):
    # 원본 코드 셀 순서와 모델·데이터 변경의 실행 순서를 보존합니다.
    execution_step_5 = execution_step_4 + 1
    # 이미지·텍스트 정합성 평가용 CLIP 로딩
    print('CLIP 로딩 중...')
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    clip_processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
    clip_model = clip_model.to(device)
    clip_model.eval()
    print('CLIP을 불러왔습니다.')
    return clip_model, clip_processor, execution_step_5


@app.cell
def _(
    clip_model,
    clip_processor,
    device,
    execution_step_5,
    generate_image,
    plt,
    torch,
):
    # 원본 코드 셀 순서와 모델·데이터 변경의 실행 순서를 보존합니다.
    execution_step_6 = execution_step_5 + 1
    def compute_clip_score(image, text):
        """이미지와 텍스트 사이의 CLIP 유사도를 계산합니다.
        """
        inputs = clip_processor(
            text=[text],
            images=image,
            return_tensors="pt",
            padding=True
        ).to(device)
    
        with torch.no_grad():
            outputs = clip_model(**inputs)
            # 이미지와 텍스트 임베딩의 코사인 유사도
            logits_per_image = outputs.logits_per_image
            similarity = logits_per_image.item()
    
        return similarity

    def critique_image(image, prompt, attributes_to_check=None):
        """프롬프트를 기준으로 생성 이미지를 비평합니다.
        """
        # 전체 유사도
        overall_score = compute_clip_score(image, prompt)
    
        # 지정된 경우 세부 속성 확인
        attribute_scores = {}
        if attributes_to_check:
            for attr in attributes_to_check:
                attr_score = compute_clip_score(image, attr)
                attribute_scores[attr] = attr_score
    
        # 간단한 비평 로직
        critique = {
            "overall_score": overall_score,
            "attribute_scores": attribute_scores,
            "verdict": '잘 일치합니다.' if overall_score > 25 else '일치도가 낮아 개선이 필요합니다.'
        }
    
        return critique

    # 비평 테스트
    test_image = generate_image("a cat wearing a red hat", seed=42)
    critique = critique_image(
        test_image,
        "a cat wearing a red hat",
        attributes_to_check=[
            "a cat",
            "a red hat",
            "a blue hat",  # 틀린 설명이므로 점수가 더 낮아야 함
            "a dog",  # 틀린 설명이므로 점수가 더 낮아야 함
        ]
    )

    print('\n비평 결과:')
    print(f'전체 점수: {critique['overall_score']:.2f}')
    print(f'판정: {critique['verdict']}')
    print('\n속성별 점수:')
    for attr, score in critique['attribute_scores'].items():
        print(f"  '{attr}': {score:.2f}")

    plt.figure(figsize=(5, 5))
    plt.imshow(test_image)
    plt.title(f'점수: {critique['overall_score']:.2f}\n{critique['verdict']}')
    plt.axis('off')
    plt.show()
    return compute_clip_score, critique_image, execution_step_6


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. VLM 피드백을 통한 반복 개선

    다음과 같은 **반복 개선 루프**를 구현합니다.
    1. 초기 프롬프트로 이미지 생성
    2. VLM으로 이미지 비평
    3. 점수가 낮으면 강조 표현·네거티브 프롬프트 등을 추가해 수정
    4. 수정한 프롬프트로 재생성
    5. 품질 임계값에 도달할 때까지 반복
    """)
    return


@app.cell
def _(critique_image, execution_step_6, generate_image, plt):
    # 원본 코드 셀 순서와 모델·데이터 변경의 실행 순서를 보존합니다.
    execution_step_7 = execution_step_6 + 1

    def refine_prompt_based_on_critique(original_prompt, critique, iteration):
        """VLM 비평을 바탕으로 프롬프트를 개선합니다.
        """  # 간단한 개선 전략
        refinements = [f'{original_prompt}, highly detailed', f'{original_prompt}, high quality, professional photography', f'{original_prompt}, 8k, ultra realistic, sharp focus']  # 1. 강조 키워드 추가
        refined_prompt = refinements[min(iteration, len(refinements) - 1)]  # 2. 세부 묘사 확대
        return refined_prompt  # 3. 품질 향상 표현 추가

    def iterative_generate_with_refinement(prompt, max_iterations=3, target_score=28.0, seed=42):
        """VLM 유도 반복 개선으로 이미지를 생성합니다.
        """
        history = []
        current_prompt = prompt
        for iteration in range(max_iterations):
            print(f'\n--- 반복 {iteration + 1} ---')
            print(f'프롬프트: {current_prompt}')  # 반복 횟수에 따라 다른 개선 방식 적용
            image = generate_image(current_prompt, seed=seed + iteration)
            critique = critique_image(image, prompt)
            score = critique['overall_score']
            print(f'점수: {score:.2f}')
            print(f'판정: {critique['verdict']}')
            history.append({'iteration': iteration + 1, 'prompt': current_prompt, 'image': image, 'score': score, 'critique': critique})
            if score >= target_score:
                print(f'✓ 목표 점수 도달! ({score:.2f} >= {target_score})')
                break
            if iteration < max_iterations - 1:
                current_prompt = refine_prompt_based_on_critique(prompt, critique, iteration)
        return history
    _test_prompt = 'a red sports car in front of a modern building'
    _history = iterative_generate_with_refinement(_test_prompt, max_iterations=3, target_score=27.0, seed=100)
    _fig, _axes = plt.subplots(1, len(_history), figsize=(5 * len(_history), 5))  # 이미지 생성
    if len(_history) == 1:
        _axes = [_axes]
    for _i, _entry in enumerate(_history):  # 비평
        _axes[_i].imshow(_entry['image'])
        _axes[_i].set_title(f'반복 {_entry['iteration']}\n점수: {_entry['score']:.2f}', fontsize=10)
        _axes[_i].axis('off')
    plt.tight_layout()
    plt.show()
    print('\n=== 개선 결과 정리 ===')
    for _entry in _history:  # 이력 저장
    # 반복 개선 테스트
    # 진행 과정 시각화
        print(f'반복 {_entry['iteration']}: 점수 {_entry['score']:.2f} - {_entry['prompt'][:60]}...')  # 목표 도달 여부 확인 / 다음 반복의 프롬프트 개선
    return (execution_step_7,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. 이미지 조건부 생성(인페인팅)

    텍스트 → 이미지 생성 외에도 **이미지 조건부 생성**을 수행할 수 있습니다.
    - **인페인팅**: 마스크 영역 채우기
    - **이미지 간 변환**: 기존 이미지 변환
    - **변형 생성**: 변화를 준 유사 이미지 생성

    Stable Diffusion으로 인페인팅을 구현합니다.
    """)
    return


@app.cell
def _(StableDiffusionInpaintPipeline, device, execution_step_7, torch):
    # 원본 코드 셀 순서와 모델·데이터 변경의 실행 순서를 보존합니다.
    execution_step_8 = execution_step_7 + 1
    # 인페인팅 파이프라인 로딩
    print('Stable Diffusion 인페인팅 로딩 중...')
    inpaint_pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-inpainting",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        safety_checker=None,
    )
    inpaint_pipe = inpaint_pipe.to(device)

    if torch.cuda.is_available():
        inpaint_pipe.enable_attention_slicing()
        inpaint_pipe.enable_vae_slicing()

    print('인페인팅 파이프라인을 불러왔습니다.')
    return execution_step_8, inpaint_pipe


@app.cell
def _(
    Image,
    ImageDraw,
    device,
    execution_step_8,
    generate_image,
    inpaint_pipe,
    plt,
    torch,
):
    # 원본 코드 셀 순서와 모델·데이터 변경의 실행 순서를 보존합니다.
    execution_step_9 = execution_step_8 + 1

    def create_circular_mask(image_size, center, radius):
        """인페인팅용 원형 마스크를 생성합니다.
        """
        mask = Image.new('L', image_size, 0)
        draw = ImageDraw.Draw(mask)
        draw.ellipse([(center[0] - radius, center[1] - radius), (center[0] + radius, center[1] + radius)], fill=255)
        return mask

    def inpaint_image(image, mask, prompt, num_inference_steps=25, seed=42):
        """프롬프트에 따라 이미지의 마스크 영역을 채웁니다.
        """
        generator = torch.Generator(device=device).manual_seed(seed)
        image = image.resize((512, 512))
        mask = mask.resize((512, 512))
        with torch.inference_mode():
            result = inpaint_pipe(prompt=prompt, image=image, mask_image=mask, num_inference_steps=num_inference_steps, generator=generator).images[0]
        return result
    base_image = generate_image('a room with white walls and wooden floor', seed=200)
    base_image = base_image.resize((512, 512))  # SD 요구 사항에 맞게 512×512로 크기 조정
    mask = create_circular_mask((512, 512), center=(256, 256), radius=100)
    inpaint_prompts = ['a red chair in the center of the room', 'a green plant in a pot', 'a modern lamp']
    _fig, _axes = plt.subplots(2, 3, figsize=(15, 10))
    _axes[0, 0].imshow(base_image)
    _axes[0, 0].set_title('원본 이미지')
    _axes[0, 0].axis('off')
    _axes[0, 1].imshow(mask, cmap='gray')
    _axes[0, 1].set_title('마스크(인페인팅 영역)')
    _axes[0, 1].axis('off')
    _axes[0, 2].axis('off')
    for _i, _prompt in enumerate(inpaint_prompts):
        print(f'인페인팅: {_prompt}')
        inpainted = inpaint_image(base_image, mask, _prompt, seed=200 + _i)
        _axes[1, _i].imshow(inpainted)
    # 기반 이미지 생성
        _axes[1, _i].set_title(f'인페인팅 결과: {_prompt}', fontsize=9)
        _axes[1, _i].axis('off')
    plt.tight_layout()
    # 중앙 영역 마스크 생성
    # 여러 객체로 인페인팅
    # 원본과 마스크 표시
    # 인페인팅 결과 생성
    plt.show()
    return base_image, execution_step_9, inpaint_image, mask


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. 품질 제어를 포함한 VLM 유도 인페인팅

    인페인팅과 VLM 비평을 결합해 **품질을 제어하는 이미지 편집**을 수행합니다.
    1. 사용자가 편집 영역과 원하는 내용을 지정
    2. 해당 영역에 인페인팅 수행
    3. VLM이 편집 결과와 설명의 일치 여부 확인
    4. 일치하지 않으면 프롬프트를 개선해 재시도
    """)
    return


@app.cell
def _(base_image, critique_image, execution_step_9, inpaint_image, mask, plt):
    # 원본 코드 셀 순서와 모델·데이터 변경의 실행 순서를 보존합니다.
    execution_step_10 = execution_step_9 + 1

    def vlm_guided_inpainting(image, mask, prompt, max_attempts=3, target_score=25.0, seed=42):
        """VLM 품질 제어를 사용해 인페인팅합니다.
        """
        attempts = []
        current_prompt = prompt
        for attempt in range(max_attempts):
            print(f'\n--- 시도 {attempt + 1} ---')
            print(f'프롬프트: {current_prompt}')
            result = inpaint_image(image, mask, current_prompt, seed=seed + attempt)
            critique = critique_image(result, prompt)
            score = critique['overall_score']  # 인페인팅
            print(f'점수: {score:.2f}')
            attempts.append({'attempt': attempt + 1, 'prompt': current_prompt, 'result': result, 'score': score})
            if score >= target_score:  # 인페인팅 결과 비평
                print(f'✓ 품질 임계값에 도달했습니다.')
                break
            if attempt < max_attempts - 1:
                current_prompt = f'{prompt}, highly detailed, realistic'
        best = max(attempts, key=lambda x: x['score'])
        return (best, attempts)
    test_inpaint_prompt = 'a beautiful red rose in a vase'
    _best, attempts = vlm_guided_inpainting(base_image, mask, test_inpaint_prompt, max_attempts=2, target_score=24.0, seed=300)
    _fig, _axes = plt.subplots(1, len(attempts) + 2, figsize=(5 * (len(attempts) + 2), 5))
    _axes[0].imshow(base_image)
    _axes[0].set_title('원본')
    _axes[0].axis('off')
    _axes[1].imshow(mask, cmap='gray')
    _axes[1].set_title('마스크')
    _axes[1].axis('off')
    for _i, att in enumerate(attempts):
        _axes[_i + 2].imshow(att['result'])  # 프롬프트 개선
        _marker = '✓' if att == _best else ''
        _axes[_i + 2].set_title(f'시도 {att['attempt']} {_marker}\n점수: {att['score']:.2f}', fontsize=9)
        _axes[_i + 2].axis('off')
    plt.tight_layout()  # 가장 좋은 결과 반환
    # VLM 유도 인페인팅 테스트
    # 시도 결과 시각화
    plt.show()
    return (execution_step_10,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. 여러 프롬프트로 생성하고 선택하기

    다른 프롬프트·시드로 **여러 변형**을 생성한 뒤 VLM이 **가장 좋은 결과를 선택**하도록 하는 전략입니다.
    """)
    return


@app.cell
def _(compute_clip_score, execution_step_10, generate_image, plt):
    # 원본 코드 셀 순서와 모델·데이터 변경의 실행 순서를 보존합니다.
    execution_step_11 = execution_step_10 + 1

    def generate_and_select_best(base_prompt, num_variations=4, seeds=None):
        """여러 변형을 생성하고 VLM으로 최상의 결과를 선택합니다.
        """
        if seeds is None:
            seeds = list(range(num_variations))
        prompt_variations = [base_prompt, f'{base_prompt}, professional photography', f'{base_prompt}, digital art, trending on artstation', f'{base_prompt}, ultra realistic, 8k, detailed']  # 표현을 달리한 프롬프트 변형
        candidates = []
        print(f'생성할 변형 수: {num_variations}개...')
        for i in range(num_variations):
            prompt = prompt_variations[i % len(prompt_variations)]
            seed = seeds[i]
            print(f'  변형 {i + 1}: 시드={seed}')
            image = generate_image(prompt, seed=seed, num_inference_steps=20)
            score = compute_clip_score(image, base_prompt)
            candidates.append({'index': i, 'prompt': prompt, 'seed': seed, 'image': image, 'score': score})
        best = max(candidates, key=lambda x: x['score'])
        print(f'\n✓ 최상의 결과: 변형 {best['index'] + 1} (점수: {best['score']:.2f})')
        return (best, candidates)
    _test_prompt = 'a magical forest with glowing mushrooms'
    _best, candidates = generate_and_select_best(_test_prompt, num_variations=4, seeds=[10, 20, 30, 40])
    _fig, _axes = plt.subplots(1, 4, figsize=(20, 5))
    for _i, cand in enumerate(candidates):
        _axes[_i].imshow(cand['image'])
        _marker = '★ BEST' if cand == _best else ''  # 기본 프롬프트를 기준으로 점수 계산
        _axes[_i].set_title(f'변형 {cand['index'] + 1} {_marker}\n점수: {cand['score']:.2f}\n시드: {cand['seed']}', fontsize=9)
        _axes[_i].axis('off')
    plt.tight_layout()
    # 여러 생성 결과 중 선택 테스트
    # 모든 후보 시각화
    plt.show()  # 가장 좋은 결과 선택
    return (execution_step_11,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7. 심화: 캡션 기반 개선 루프

    **캡션 VLM**으로 생성 이미지를 설명한 다음, 캡션과 원래 프롬프트를 비교해 차이를 찾습니다.
    """)
    return


@app.cell
def _(device, execution_step_11):
    # 원본 코드 셀 순서와 모델·데이터 변경의 실행 순서를 보존합니다.
    execution_step_12 = execution_step_11 + 1
    # 간단한 캡션 모델 BLIP 로딩
    from transformers import BlipProcessor, BlipForConditionalGeneration

    print('캡션 생성용 BLIP 로딩 중...')
    blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    blip_model = blip_model.to(device)
    blip_model.eval()
    print('BLIP을 불러왔습니다.')
    return blip_model, blip_processor, execution_step_12


@app.cell
def _(
    blip_model,
    blip_processor,
    compute_clip_score,
    device,
    execution_step_12,
    generate_image,
    plt,
    torch,
):
    # 원본 코드 셀 순서와 모델·데이터 변경의 실행 순서를 보존합니다.
    execution_step_13 = execution_step_12 + 1

    def caption_image(image):
        """BLIP으로 이미지 캡션을 생성합니다.
        """
        inputs = blip_processor(image, return_tensors='pt').to(device)
        with torch.no_grad():
            generated_ids = blip_model.generate(**inputs, max_length=50)
            caption = blip_processor.decode(generated_ids[0], skip_special_tokens=True)
        return caption

    def caption_based_refinement(prompt, max_iterations=3, seed=42):
        """캡션 기반 피드백 루프로 생성합니다.
        """
        history = []
        current_prompt = prompt
        for iteration in range(max_iterations):
            print(f'\n--- 반복 {iteration + 1} ---')
            print(f'프롬프트: {current_prompt}')
            image = generate_image(current_prompt, seed=seed + iteration, num_inference_steps=20)
            generated_caption = caption_image(image)
            print(f"생성한 캡션: '{generated_caption}'")
            score = compute_clip_score(image, prompt)  # 생성
            print(f'CLIP 점수: {score:.2f}')
            history.append({'iteration': iteration + 1, 'prompt': current_prompt, 'image': image, 'caption': generated_caption, 'score': score})
            if score > 27.0:  # 결과의 캡션 생성
                print('✓ 양호한 품질에 도달했습니다.')
                break
            if iteration < max_iterations - 1:
                current_prompt = f'{prompt}, {generated_caption}, high quality'  # 정합성 점수 계산
        return history
    _test_prompt = 'a cyberpunk city at night with neon lights'
    _history = caption_based_refinement(_test_prompt, max_iterations=2, seed=500)
    _fig, _axes = plt.subplots(1, len(_history), figsize=(7 * len(_history), 7))
    if len(_history) == 1:
        _axes = [_axes]
    for _i, _entry in enumerate(_history):
        _axes[_i].imshow(_entry['image'])
        _axes[_i].set_title(f'반복 {_entry['iteration']}\n점수: {_entry['score']:.2f}\n캡션: {_entry['caption'][:50]}...', fontsize=9)
        _axes[_i].axis('off')
    plt.tight_layout()
    # 캡션 기반 개선 테스트
    # 시각화
    plt.show()  # 품질 충족 여부 확인 / 원래 프롬프트와 생성 캡션의 정보를 결합해 개선
    return (execution_step_13,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 정리와 비교

    지금까지 살펴본 VLM 유도 생성 전략을 비교합니다.
    """)
    return


@app.cell
def _(execution_step_13):
    # 원본 코드 셀 순서와 모델·데이터 변경의 실행 순서를 보존합니다.
    execution_step_14 = execution_step_13 + 1
    import pandas as pd

    strategies = {
        '전략': [
            '기본 텍스트 → 이미지',
            '반복 개선',
            '여러 변형 중 선택',
            'VLM 유도 인페인팅',
            '캡션 기반 루프',
        ],
        '설명': [
            '텍스트 프롬프트에서 한 번 생성',
            '생성 → 비평 → 프롬프트 개선 → 재생성',
            '변형 N개 생성 → VLM이 최상의 결과 선택',
            '영역 채우기 → VLM 품질 확인 → 필요 시 재시도',
            '생성 → 캡션 → 프롬프트와 비교 → 개선',
        ],
        '장점': [
            '빠르고 단순함',
            '반복하며 품질 향상',
            '다양한 후보 탐색',
            '정밀한 영역 제어',
            '캡션을 통한 상세 피드백',
        ],
        '단점': [
            '품질 제어 없음',
            '여러 번 생성 필요',
            '높은 연산 비용(N회 생성)',
            '마스크 지정 필요',
            '캡션에서 세부 정보 누락 가능',
        ],
        '적합한 용도': [
            '빠른 시제품 제작',
            '고품질 단일 이미지',
            '후보 중 최상의 결과 선택',
            '특정 영역 편집',
            '생성 실패 원인 이해',
        ],
    }

    df = pd.DataFrame(strategies)
    print('\nVLM 유도 생성 전략 비교:')
    print(df.to_string(index=False))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 핵심 내용

    ### 1. **품질 검사자로서의 VLM**
    - CLIP 점수로 이미지·텍스트 정합성 확인
    - 품질이 낮은 결과를 자동으로 제외
    - 품질 임계값에 도달할 때까지 반복

    ### 2. **반복 개선의 효과**
    - 단순한 프롬프트 수정으로도 결과를 크게 개선 가능
    - 원문에서는 보통 2~3회 반복으로 품질 목표에 도달한다고 설명
    - 이전 실패에서 얻은 피드백을 다음 반복에 반영

    ### 3. **여러 변형 중 선택**
    - 다양한 시드·프롬프트로 여러 후보 생성
    - VLM이 가장 잘 맞는 결과 선택
    - 비용이 더 들지만 해 공간을 넓게 탐색

    ### 4. **이미지 조건부 생성**
    - 인페인팅으로 영역을 정밀하게 제어
    - VLM으로 편집 결과와 설명의 일치 확인
    - 반복 이미지 편집 흐름에 유용

    ### 5. **캡션 기반 피드백**
    - 캡션으로 실제 생성된 내용 확인
    - 캡션과 프롬프트를 비교해 차이 식별
    - 반복적으로 발생하는 생성 실패 진단

    ## 실무 적용 방향

    **상용 시스템:**
    - VLM 점수를 활용한 **여러 변형 중 선택**
    - 사용 목적에 맞는 품질 임계값 설정
    - 분석을 위한 점수·프롬프트 기록

    **사용자용 앱:**
    - 여러 후보를 보여 주고 사용자와 VLM이 함께 선택
    - 자동 개선을 포함한 "다시 생성" 기능 구현
    - 편집 작업에 인페인팅 활용

    **연구:**
    - 캡션 기반 루프로 모델 동작 이해
    - CLIP, BLIP, 맞춤형 모델 등 여러 VLM 비평기 실험
    - VLM을 판별기로 사용하는 적대적 개선 시도

    ## 다음 단계

    1. **VLM 비평기 미세 조정**: 특정 분야의 점수 평가 개선
    2. **개선 방식 학습**: 프롬프트 수정안을 예측하는 모델 학습
    3. **멀티모달 융합**: CLIP 점수, 캡션, 객체 탐지 결합
    4. **상호작용형 편집**: VLM과 함께 사용자 피드백 반영
    5. **효율 개선**: 증류·양자화 VLM으로 빠르게 비평

    **더 읽을 자료:**
    - Stable Diffusion: https://github.com/Stability-AI/stablediffusion
    - CLIP: https://github.com/openai/CLIP
    - BLIP: https://github.com/salesforce/BLIP
    - Diffusers 라이브러리: https://github.com/huggingface/diffusers
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## GPU 메모리 정리
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### GPU 메모리 해제 (marimo)

    실습을 마치면 marimo의 **커널 재시작**을 사용하세요. marimo는 셀 출력과 의존성에 객체 참조를 보관하므로 Jupyter의 `globals()`/`del` 방식 대신 커널을 재시작해 모델과 GPU 메모리를 해제합니다.
    """)
    return


if __name__ == "__main__":
    app.run()
