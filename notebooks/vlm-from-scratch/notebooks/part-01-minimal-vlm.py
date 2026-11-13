import marimo

__generated_with = "0.25.2"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## marimo 실행

    노트북을 실행하면 모델·데이터 다운로드부터 학습·평가·저장까지 원본 순서대로 진행합니다. 설정을 바꾼 뒤에는 커널을 재시작하세요. 산출물은 저장소 루트 `outputs/vlm-from-scratch/<실행 ID>/`에 저장합니다. 후속 파트에서 가중치를 재사용하려면 같은 `VLM_RUN_ID`로 실행하세요.
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
    # 처음부터 만드는 최소 구성의 시각·언어 모델

    Nipun Batra · 2025-12-24
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 소개

    시각·언어 모델(Vision-Language Model, VLM)은 AI가 이미지를 이해하고 추론하는 방식을 크게 바꾸었습니다. GPT-4V, LLaVA, Gemini 같은 모델은 이미지를 설명하고, 이미지에 관한 질문에 답하며, 복잡한 시각적 지시도 따를 수 있습니다.

    이 모델들은 실제로 어떻게 동작할까요? VLM의 핵심은 다음 세 구성 요소입니다.

    1. **비전 인코더**: 이미지를 의미 있는 특징 표현으로 변환합니다.
    2. **프로젝션 층**: 시각 임베딩 공간과 언어 임베딩 공간을 연결합니다.
    3. **언어 모델**: 시각 특징을 조건으로 텍스트를 생성합니다.

    이 노트북에서는 공개된 소형 모델을 사용해 최소 구성의 VLM을 직접 만듭니다.
    - **시각**: Google의 ViT-Large(매개변수 304M)
    - **언어**: Hugging Face의 SmolLM-360M(매개변수 360M)
    - **데이터셋**: Flickr8k(교육용으로 일부만 사용)

    목표는 **학습과 구조 이해**입니다. 상용 모델을 구축하기 위한 실습은 아닙니다.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 환경 설정
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    프로젝트 디렉토리(`notebooks/vlm-from-scratch/`)에서 `uv sync --locked`로 의존성을 설치하세요.
    """)
    return


@app.cell
def _(execution_step_0):
    # 원본 코드 셀 순서와 모델·데이터 변경의 실행 순서를 보존합니다.
    execution_step_1 = execution_step_0 + 1
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
    from PIL import Image
    import requests
    from io import BytesIO
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
    import warnings
    warnings.filterwarnings('ignore')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'사용 장치: {device}')

    # torch 버전 출력
    print(f'Torch 버전: {torch.__version__}')
    return (
        AutoModelForCausalLM,
        AutoTokenizer,
        BytesIO,
        DataLoader,
        Dataset,
        F,
        Image,
        ViTImageProcessor,
        ViTModel,
        device,
        execution_step_1,
        load_dataset,
        nn,
        plt,
        requests,
        torch,
        tqdm,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    비전 인코더는 이미지를 의미 있는 특징 벡터의 시퀀스로 바꿉니다. 사전 학습된 비전 트랜스포머(Vision Transformer, ViT)를 사용하며, 처리 과정은 다음과 같습니다.

    1. 이미지를 16×16 크기의 패치로 나눕니다.
    2. 각 패치를 임베딩으로 투영합니다.
    3. 트랜스포머 층으로 패치를 처리합니다.
    4. 패치별 특징과 CLS 토큰으로 이루어진 특징 시퀀스를 출력합니다.

    224×224 이미지를 16×16 패치로 나누면 다음과 같습니다.
    - (224/16) × (224/16) = 14 × 14 = 196개 패치
    - CLS 토큰 1개를 더해 총 **197개 토큰**
    - 각 토큰의 차원은 1024(ViT-Large 기준)
    """)
    return


@app.cell
def _(ViTImageProcessor, ViTModel, device, execution_step_1):
    # 원본 코드 셀 순서와 모델·데이터 변경의 실행 순서를 보존합니다.
    execution_step_2 = execution_step_1 + 1
    # 사전 학습된 ViT-Large 불러오기(성능 향상을 위해 ViT-Base보다 큰 모델 사용)
    vision_model_name = "google/vit-large-patch16-224"
    vision_encoder = ViTModel.from_pretrained(vision_model_name)
    image_processor = ViTImageProcessor.from_pretrained(vision_model_name)

    # 학습하지 않을 비전 인코더의 가중치 고정
    for param in vision_encoder.parameters():
        param.requires_grad = False

    vision_encoder = vision_encoder.to(device)
    vision_encoder.eval()

    print(f'비전 인코더 은닉 차원: {vision_encoder.config.hidden_size}')
    print(f'패치 수: {(224 // 16) ** 2} + CLS 토큰 1개 = {(224 // 16) ** 2 + 1}개 토큰')
    return execution_step_2, image_processor, vision_encoder, vision_model_name


@app.cell
def _(
    BytesIO,
    Image,
    device,
    execution_step_2,
    image_processor,
    plt,
    requests,
    torch,
    vision_encoder,
):
    # 원본 코드 셀 순서와 모델·데이터 변경의 실행 순서를 보존합니다.
    execution_step_3 = execution_step_2 + 1
    # 샘플 이미지로 비전 인코더 테스트
    _url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
    response = requests.get(_url)
    sample_image = Image.open(BytesIO(response.content))
    plt.figure(figsize=(6, 6))
    plt.imshow(sample_image)
    plt.title('COCO 샘플 이미지')
    plt.axis('off')
    plt.show()
    _inputs = image_processor(sample_image, return_tensors='pt').to(device)
    with torch.no_grad():
    # 이미지 처리
        _vision_outputs = vision_encoder(**_inputs)
    image_features = _vision_outputs.last_hidden_state
    print(f'이미지 특징 shape: {image_features.shape}')
    print(f'  - 배치 크기: {image_features.shape[0]}')
    print(f'  - 시퀀스 길이(패치 + CLS): {image_features.shape[1]}')
    # 패치 특징 시퀀스 추출(pooler 출력 제외)
    print(f'  - 은닉 차원: {image_features.shape[2]}')
    return execution_step_3, image_features


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2단계: 언어 모델

    언어 모델로는 Hugging Face의 작지만 성능이 좋은 **SmolLM-360M**을 사용합니다.
    - 매개변수 360M개(135M 모델보다 크며 성능이 더 좋음)
    - 은닉 차원 960
    - 비교적 작은 하드웨어에서도 실행 가능

    핵심은 ViT-Large의 시각 특징(1024차원)을 LLM의 임베딩 공간(960차원)으로 **투영**해야 한다는 점입니다.
    """)
    return


@app.cell
def _(AutoModelForCausalLM, AutoTokenizer, device, execution_step_3):
    # 원본 코드 셀 순서와 모델·데이터 변경의 실행 순서를 보존합니다.
    execution_step_4 = execution_step_3 + 1
    # 언어 모델 불러오기(성능 향상을 위해 SmolLM-360M 사용)
    lm_model_name = "HuggingFaceTB/SmolLM-360M"
    tokenizer = AutoTokenizer.from_pretrained(lm_model_name)
    language_model = AutoModelForCausalLM.from_pretrained(lm_model_name)

    # 패딩 토큰이 없으면 추가
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    language_model = language_model.to(device)

    print(f'언어 모델 은닉 차원: {language_model.config.hidden_size}')
    print(f'어휘 크기: {language_model.config.vocab_size}')
    print(f'층 수: {language_model.config.num_hidden_layers}')
    return execution_step_4, language_model, lm_model_name, tokenizer


@app.cell
def _(device, execution_step_4, language_model, tokenizer, torch):
    # 원본 코드 셀 순서와 모델·데이터 변경의 실행 순서를 보존합니다.
    execution_step_5 = execution_step_4 + 1
    # 언어 모델 테스트
    test_text = 'A photo of'
    _inputs = tokenizer(test_text, return_tensors='pt').to(device)
    with torch.no_grad():
        outputs = language_model.generate(**_inputs, max_new_tokens=20, do_sample=True, temperature=0.7, pad_token_id=tokenizer.pad_token_id)
    print(f'생성된 텍스트: {tokenizer.decode(outputs[0], skip_special_tokens=True)}')
    return (execution_step_5,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3단계: 프로젝션 층

    프로젝션 층은 시각과 언어를 연결하는 **다리**입니다. 다음과 같이 변환합니다.
    - **입력**: 시각 특징 (batch, 197, 1024)
    - **출력**: 언어 모델에 맞는 임베딩 (batch, 197, 960)

    간단한 다층 프로젝션을 사용합니다.
    1. 선형 층(1024 → 960)
    2. GELU 활성화 함수
    3. 안정성을 위한 LayerNorm

    LLaVA에서 사용하는 방식과 비슷하지만 더 단순합니다.
    """)
    return


@app.cell
def _(device, execution_step_5, language_model, nn, torch, vision_encoder):
    # 원본 코드 셀 순서와 모델·데이터 변경의 실행 순서를 보존합니다.
    execution_step_6 = execution_step_5 + 1
    class VisionProjector(nn.Module):
        """시각 특징을 언어 모델의 임베딩 공간으로 투영합니다.
        """
    
        def __init__(self, vision_dim, language_dim):
            super().__init__()
            self.projection = nn.Sequential(
                nn.Linear(vision_dim, language_dim),
                nn.GELU(),
                nn.LayerNorm(language_dim),
                nn.Linear(language_dim, language_dim),  # 정렬 성능을 높이기 위한 추가 층
            )
    
        def forward(self, vision_features):
            """시각 특징을 언어 임베딩 공간으로 투영합니다.
            
            인자:
                vision_features: (batch, seq_len, vision_dim)
            반환:
                projected: (batch, seq_len, language_dim)
            """
            return self.projection(vision_features)


    # 프로젝터 생성
    vision_dim = vision_encoder.config.hidden_size  # 1024(ViT-Large의 은닉 차원)
    language_dim = language_model.config.hidden_size  # 960(SmolLM-360M의 은닉 차원)

    projector = VisionProjector(vision_dim, language_dim).to(device)

    print(f'프로젝터: {vision_dim} -> {language_dim}')
    print(f'학습 가능한 매개변수: {sum((p.numel() for p in projector.parameters())):,}')
    return (
        VisionProjector,
        execution_step_6,
        language_dim,
        projector,
        vision_dim,
    )


@app.cell
def _(execution_step_6, image_features, projector, torch):
    # 원본 코드 셀 순서와 모델·데이터 변경의 실행 순서를 보존합니다.
    execution_step_7 = execution_step_6 + 1
    # 프로젝터 테스트
    with torch.no_grad():
        projected_features = projector(image_features)

    print(f'원본 시각 특징: {image_features.shape}')
    print(f'투영한 특징: {projected_features.shape}')
    return (execution_step_7,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4단계: 전체 VLM 구조

    이제 구성 요소를 하나의 모델로 결합합니다. 순전파 과정은 다음과 같습니다.

    1. **이미지 인코딩** → 시각 특징 (batch, 197, 768)
    2. **투영** → 언어 공간의 특징 (batch, 197, 576)
    3. **텍스트 임베딩** → 텍스트 임베딩 (batch, text_len, 576)
    4. **연결** → [image_embeds, text_embeds] (batch, 197 + text_len, 576)
    5. **생성** → LLM으로 캡션 토큰 생성
    """)
    return


@app.cell
def _(
    AutoModelForCausalLM,
    AutoTokenizer,
    F,
    ViTModel,
    VisionProjector,
    device,
    execution_step_7,
    language_model,
    nn,
    projector,
    tokenizer,
    torch,
    vision_encoder,
):
    # 원본 코드 셀 순서와 모델·데이터 변경의 실행 순서를 보존합니다.
    execution_step_8 = execution_step_7 + 1
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
        
            # 비전 인코더 가중치 고정
            for param in self.vision_encoder.parameters():
                param.requires_grad = False
    
        def encode_image(self, pixel_values):
            """이미지를 인코딩하고 언어 공간으로 투영합니다.
            """
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
            """학습용 순전파입니다.
            
            인자:
                pixel_values: 이미지 텐서 (batch, 3, 224, 224)
                input_ids: 텍스트 토큰 ID (batch, text_len)
                attention_mask: 텍스트 어텐션 마스크 (batch, text_len)
                labels: 손실 계산용 정답 토큰 ID (batch, text_len)
            """
            batch_size = pixel_values.shape[0]
        
            # 1. 이미지 인코딩 및 투영
            image_embeds = self.encode_image(pixel_values)  # (batch, 197, hidden)
            num_image_tokens = image_embeds.shape[1]
        
            # 2. LLM의 임베딩 층에서 텍스트 임베딩 추출
            text_embeds = self.language_model.get_input_embeddings()(input_ids)  # (batch, text_len, hidden)
        
            # 3. 연결: [이미지 토큰] [텍스트 토큰]
            combined_embeds = torch.cat([image_embeds, text_embeds], dim=1)
        
            # 4. 결합한 시퀀스의 어텐션 마스크 생성
            image_attention = torch.ones(
                (batch_size, num_image_tokens),
                dtype=attention_mask.dtype,
                device=attention_mask.device
            )
            combined_attention = torch.cat([image_attention, attention_mask], dim=1)
        
            # 5. 레이블 생성: 이미지 토큰은 -100(손실에서 제외), 텍스트는 실제 레이블 사용
            if labels is not None:
                image_labels = torch.full(
                    (batch_size, num_image_tokens),
                    fill_value=-100,  # -100은 CrossEntropyLoss에서 무시됨
                    dtype=labels.dtype,
                    device=labels.device
                )
                combined_labels = torch.cat([image_labels, labels], dim=1)
            else:
                combined_labels = None
        
            # 6. 언어 모델로 순전파
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
            max_new_tokens = 50,
            temperature = 0.7,
            do_sample = True,
        ):
            """이미지의 캡션을 생성합니다.
            """
            self.eval()
        
            # 이미지 인코딩
            image_embeds = self.encode_image(pixel_values)  # (1, 197, hidden)
        
            # 프롬프트로 시작
            prompt = "This image shows"
            prompt_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(pixel_values.device)
            prompt_embeds = self.language_model.get_input_embeddings()(prompt_ids)
        
            # 이미지와 프롬프트 임베딩 결합
            combined_embeds = torch.cat([image_embeds, prompt_embeds], dim=1)
        
            # 토큰을 하나씩 생성
            generated_ids = prompt_ids.clone()
        
            for _ in range(max_new_tokens):
                # 현재 임베딩 추출
                current_embeds = self.language_model.get_input_embeddings()(generated_ids)
                full_embeds = torch.cat([image_embeds, current_embeds], dim=1)
            
                # 순전파
                outputs = self.language_model(inputs_embeds=full_embeds)
                next_token_logits = outputs.logits[:, -1, :]
            
                # 샘플링 또는 탐욕적 선택
                if do_sample:
                    probs = F.softmax(next_token_logits / temperature, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = next_token_logits.argmax(dim=-1, keepdim=True)
            
                generated_ids = torch.cat([generated_ids, next_token], dim=1)
            
                # EOS이면 중단
                if next_token.item() == self.tokenizer.eos_token_id:
                    break
        
            return self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)


    # VLM 생성
    vlm = MiniVLM(
        vision_encoder=vision_encoder,
        language_model=language_model,
        projector=projector,
        tokenizer=tokenizer,
    )
    vlm = vlm.to(device)

    # 매개변수 수 계산
    total_params = sum(p.numel() for p in vlm.parameters())
    trainable_params = sum(p.numel() for p in vlm.parameters() if p.requires_grad)

    print(f'전체 매개변수: {total_params:,}')
    print(f'학습 가능한 매개변수: {trainable_params:,}')
    print(f'고정된 매개변수: {total_params - trainable_params:,}')
    return MiniVLM, execution_step_8, vlm


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5단계: Flickr8k 데이터셋 불러오기

    이미지 8,000장으로 구성된 대표적인 이미지 캡션 벤치마크 Flickr8k의 일부를 사용합니다. 이미지마다 캡션 5개가 있으며, 학습할 때 이미지별로 하나를 무작위로 선택합니다.
    """)
    return


@app.cell
def _(execution_step_8, load_dataset):
    # 원본 코드 셀 순서와 모델·데이터 변경의 실행 순서를 보존합니다.
    execution_step_9 = execution_step_8 + 1
    # Flickr8k 캡션 데이터셋 불러오기
    dataset = load_dataset("jxie/flickr8k", split="train")

    # 더 나은 학습을 위해 샘플 2,000개 사용
    num_samples = 2000
    dataset = dataset.shuffle(seed=42).select(range(num_samples))

    print(f'데이터셋 크기: {len(dataset)}')
    print(f'샘플 항목의 키: {dataset[0].keys()}')
    return dataset, execution_step_9


@app.cell
def _(dataset, execution_step_9, plt):
    # 원본 코드 셀 순서와 모델·데이터 변경의 실행 순서를 보존합니다.
    execution_step_10 = execution_step_9 + 1
    # 샘플 확인: Flickr8k에는 caption_0부터 caption_4까지 있음
    _sample = dataset[0]
    print(f'이미지: {type(_sample['image'])}')
    print(f'캡션:')
    for _i in range(5):
        print(f"  {_i}: {_sample[f'caption_{_i}']}")
    plt.figure(figsize=(6, 6))
    plt.imshow(_sample['image'])
    plt.title(f'캡션: {_sample['caption_0'][:60]}...')
    plt.axis('off')
    plt.show()
    return (execution_step_10,)


@app.cell
def _(
    DataLoader,
    Dataset,
    dataset,
    execution_step_10,
    image_processor,
    tokenizer,
):
    # 원본 코드 셀 순서와 모델·데이터 변경의 실행 순서를 보존합니다.
    execution_step_11 = execution_step_10 + 1
    import random

    class Flickr8kDataset(Dataset):
        """Flickr8k 이미지·캡션 쌍 데이터셋입니다.
        """
    
        def __init__(self, hf_dataset, image_processor, tokenizer, max_length=64):
            self.dataset = hf_dataset
            self.image_processor = image_processor
            self.tokenizer = tokenizer
            self.max_length = max_length
    
        def __len__(self):
            return len(self.dataset)
    
        def __getitem__(self, idx):
            item = self.dataset[idx]
        
            # 이미지 처리
            image = item['image'].convert('RGB')
            pixel_values = self.image_processor(image, return_tensors="pt").pixel_values.squeeze(0)
        
            # 캡션 5개 중 하나를 무작위로 선택
            caption_idx = random.randint(0, 4)
            caption = item[f'caption_{caption_idx}']
        
            # 캡션 토큰화
            encoding = self.tokenizer(
                caption,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
        
            input_ids = encoding['input_ids'].squeeze(0)
            attention_mask = encoding['attention_mask'].squeeze(0)
        
            # 레이블: 패딩 토큰을 -100으로 마스킹
            labels = input_ids.clone()
            labels[attention_mask == 0] = -100
        
            return {
                'pixel_values': pixel_values,
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'labels': labels,
            }


    # 데이터셋과 데이터로더 생성
    train_dataset = Flickr8kDataset(dataset, image_processor, tokenizer)

    train_loader = DataLoader(
        train_dataset,
        batch_size=4,  # 메모리 사용을 줄이기 위한 작은 배치
        shuffle=True,
        num_workers=0,  # 멀티프로세싱 문제 방지
    )

    print(f'배치 수: {len(train_loader)}')
    return execution_step_11, train_loader


@app.cell
def _(execution_step_11, train_loader):
    # 원본 코드 셀 순서와 모델·데이터 변경의 실행 순서를 보존합니다.
    execution_step_12 = execution_step_11 + 1
    # 배치 테스트
    batch = next(iter(train_loader))
    print(f'배치 pixel_values shape: {batch['pixel_values'].shape}')
    print(f'배치 input_ids shape: {batch['input_ids'].shape}')
    print(f'배치 labels shape: {batch['labels'].shape}')
    return (execution_step_12,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6단계: 학습 전 모델의 예측(기준선)

    학습하기 전에 VLM이 무엇을 생성하는지 살펴봅니다. 프로젝션 층의 가중치가 무작위이므로 모델은 이미지를 이해하지 못하고 무작위이거나 의미 없는 텍스트를 생성하게 됩니다. 이를 기준선으로 삼아 학습 후의 개선 정도를 비교합니다.
    """)
    return


@app.cell
def _(BytesIO, Image, execution_step_12, requests):
    # 원본 코드 셀 순서와 모델·데이터 변경의 실행 순서를 보존합니다.
    execution_step_13 = execution_step_12 + 1
    def generate_caption(model, image, image_processor, device):
        """이미지의 캡션을 생성합니다.
        """
        model.eval()
    
        # 이미지 처리
        if isinstance(image, str):
            # 이미지 URL
            response = requests.get(image)
            image = Image.open(BytesIO(response.content)).convert('RGB')
        elif not isinstance(image, Image.Image):
            image = Image.fromarray(image)
    
        pixel_values = image_processor(image, return_tensors="pt").pixel_values.to(device)
    
        # 생성
        caption = model.generate(
            pixel_values,
            max_new_tokens=30,
            temperature=0.8,
            do_sample=True,
        )
    
        return caption, image

    return execution_step_13, generate_caption


@app.cell
def _(
    dataset,
    device,
    execution_step_13,
    generate_caption,
    image_processor,
    vlm,
):
    # 원본 코드 셀 순서와 모델·데이터 변경의 실행 순서를 보존합니다.
    execution_step_14 = execution_step_13 + 1
    # 학습 전후 비교용 테스트 이미지와 정답 캡션 보관
    test_indices = [0, 10, 50, 100]
    test_samples = [(dataset[i]['image'].convert('RGB'), dataset[i]['caption_0']) for i in test_indices]
    print('=' * 60)
    # 학습 전 모델로 캡션 생성
    print('학습 전 모델 예측(프로젝터의 무작위 가중치)')
    print('=' * 60)
    untrained_captions = []
    for _i, (image, _gt_caption) in enumerate(test_samples):
        _caption, _ = generate_caption(vlm, image, image_processor, device)
        untrained_captions.append(_caption)
        print(f'\n이미지 {_i + 1}:')
        print(f'  생성: {_caption}')
        print(f'  GT: {_gt_caption[:80]}...')
    return execution_step_14, test_samples, untrained_captions


@app.cell
def _(execution_step_14, plt, test_samples, untrained_captions):
    execution_step_15 = execution_step_14 + 1
    import textwrap

    def wrap_text(text, width=40):
        """제목이 잘 보이도록 텍스트를 줄바꿈합니다.
        """
        return '\n'.join(textwrap.wrap(text, width=width))
    _fig, _axes = plt.subplots(2, 2, figsize=(14, 16))
    _axes = _axes.flatten()
    for _i, ((image_1, _gt_caption), _ax) in enumerate(zip(test_samples, _axes)):
        _ax.imshow(image_1)
        gen_text = wrap_text(f'학습 전: {untrained_captions[_i]}', 45)
        _gt_text = wrap_text(f'정답: {_gt_caption}', 45)
        _ax.set_xlabel(f'{gen_text}\n\n{_gt_text}', fontsize=9, ha='center')
        _ax.set_xticks([])
        _ax.set_yticks([])
    plt.suptitle('학습 전 VLM 예측(무작위 출력)', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.show()
    return execution_step_15, wrap_text


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7단계: 학습 루프

    이제 VLM을 학습합니다. 학습 대상은 다음 두 구성 요소입니다.
    1. **프로젝션 층**: 시각과 언어를 연결합니다.
    2. **언어 모델**: 캡션 생성을 학습합니다.

    비전 인코더의 가중치는 고정합니다.
    """)
    return


@app.cell
def _(device, execution_step_15, output_dir, torch, tqdm):
    # 원본 코드 셀 순서와 모델·데이터 변경의 실행 순서를 보존합니다.
    execution_step_16 = execution_step_15 + 1
    import os

    def train_vlm(model, train_loader, num_epochs=3, lr=0.0001, checkpoint_path=str(output_dir / 'vlm_checkpoint.pt')):
        """중단·재개용 체크포인트를 지원하는 VLM 학습 함수입니다.
        """
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable_params, lr=lr)  # 학습 가능한 매개변수만 최적화
        start_epoch = 0
        losses = []
        if os.path.exists(checkpoint_path):
            print(f'체크포인트 발견: {checkpoint_path}')  # 학습 재개용 체크포인트 불러오기 시도
            try:
                checkpoint = torch.load(checkpoint_path, map_location='cpu')
                start_epoch = checkpoint.get('epoch', 0)
                losses = checkpoint.get('losses', [])
                model.projector.load_state_dict(checkpoint['projector_state_dict'])
                model.language_model.load_state_dict(checkpoint['language_model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                for state in optimizer.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.to(device)
                print(f'다음 에포크부터 재개: {start_epoch}')
            except Exception as e:  # 옵티마이저 상태를 장치로 이동
                print(f'체크포인트 로딩 실패: {e}')
        model.train()
        model.vision_encoder.eval()
        try:
            for epoch in range(start_epoch, num_epochs):
                epoch_loss = 0
                progress_bar = tqdm(train_loader, desc=f'에포크 {epoch + 1}/{num_epochs}')
                for batch in progress_bar:
                    pixel_values = batch['pixel_values'].to(device)
                    input_ids = batch['input_ids'].to(device)  # 비전 인코더는 평가 모드 유지
                    attention_mask = batch['attention_mask'].to(device)
                    labels = batch['labels'].to(device)
                    outputs = model(pixel_values=pixel_values, input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    loss = outputs.loss
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                    optimizer.step()
                    epoch_loss = epoch_loss + loss.item()  # 배치를 장치로 이동
                    progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
                avg_loss = epoch_loss / len(train_loader)
                losses.append(avg_loss)
                print(f'에포크 {epoch + 1} - 평균 손실: {avg_loss:.4f}')
                torch.save({'epoch': epoch + 1, 'losses': losses, 'projector_state_dict': model.projector.state_dict(), 'language_model_state_dict': model.language_model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, checkpoint_path)
                print(f'체크포인트 저장 위치: {checkpoint_path}')  # 순전파
        except KeyboardInterrupt:
            print('\n' + '=' * 70)
            print('학습을 중단했습니다.')
            print(f'완료한 에포크 수: {len(losses)}에포크')
            torch.save({'epoch': len(losses), 'losses': losses, 'projector_state_dict': model.projector.state_dict(), 'language_model_state_dict': model.language_model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, checkpoint_path)
            print(f'체크포인트 저장 위치: {checkpoint_path}')
            print('체크포인트에서 재개하려면 학습을 다시 실행하세요.')
            print('=' * 70)
        return losses  # 역전파 / 매 에포크 후 체크포인트 저장 / 중단 시 체크포인트 저장

    return execution_step_16, os, train_vlm


@app.cell
def _(execution_step_16, output_dir, train_loader, train_vlm, vlm):
    # 원본 코드 셀 순서와 모델·데이터 변경의 실행 순서를 보존합니다.
    execution_step_17 = execution_step_16 + 1
    # 더 큰 데이터셋으로 더 많은 에포크 학습
    # Ctrl+C로 학습을 중단한 뒤 다시 실행하면 재개 가능
    losses = train_vlm(vlm, train_loader, num_epochs=6, lr=2e-4, checkpoint_path=str(output_dir / "vlm_caption_checkpoint.pt"))
    return execution_step_17, losses


@app.cell
def _(execution_step_17, losses, plt):
    # 원본 코드 셀 순서와 모델·데이터 변경의 실행 순서를 보존합니다.
    execution_step_18 = execution_step_17 + 1
    # 학습 손실 그래프
    plt.figure(figsize=(8, 4))
    plt.plot(range(1, len(losses)+1), losses, marker='o')
    plt.xlabel('에포크')
    plt.ylabel('손실')
    plt.title('학습 손실')
    plt.grid(True, alpha=0.3)
    plt.show()
    return (execution_step_18,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 8단계: 학습 후 모델의 예측 — 학습 전후 비교

    같은 이미지에 대한 학습 전후의 예측을 비교합니다. 얼마나 개선되었는지 확인해 보세요.
    """)
    return


@app.cell
def _(
    device,
    execution_step_18,
    generate_caption,
    image_processor,
    test_samples,
    untrained_captions,
    vlm,
):
    execution_step_19 = execution_step_18 + 1
    print('=' * 60)
    print('학습을 마친 모델의 예측')
    print('=' * 60)
    trained_captions = []
    for _i, (image_2, _gt_caption) in enumerate(test_samples):
        _caption, _ = generate_caption(vlm, image_2, image_processor, device)
        trained_captions.append(_caption)
        print(f'\n이미지 {_i + 1}:')
        print(f'  학습 전: {untrained_captions[_i]}')
        print(f'  학습 후: {_caption}')
        print(f'  정답: {_gt_caption[:80]}...')
    return execution_step_19, trained_captions


@app.cell
def _(
    execution_step_19,
    plt,
    test_samples,
    trained_captions,
    untrained_captions,
    wrap_text,
):
    execution_step_20 = execution_step_19 + 1
    _fig, _axes = plt.subplots(4, 2, figsize=(16, 24))
    for _i, (image_3, _gt_caption) in enumerate(test_samples):
        _axes[_i, 0].imshow(image_3)
        untrained_text = wrap_text(f'학습 전: {untrained_captions[_i]}', 40)
        _axes[_i, 0].set_xlabel(untrained_text, fontsize=9, color='red', ha='center')
        _axes[_i, 0].set_xticks([])
        _axes[_i, 0].set_yticks([])
        _axes[_i, 1].imshow(image_3)
        trained_text = wrap_text(f'학습 후: {trained_captions[_i]}', 40)
        _gt_text = wrap_text(f'정답: {_gt_caption}', 40)
        _axes[_i, 1].set_xlabel(f'{trained_text}\n\n{_gt_text}', fontsize=9, color='green', ha='center')
        _axes[_i, 1].set_xticks([])
        _axes[_i, 1].set_yticks([])
    plt.suptitle('학습 전후 비교', fontsize=16, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.show()
    return (execution_step_20,)


@app.cell
def _(
    device,
    execution_step_20,
    generate_caption,
    image_processor,
    plt,
    vlm,
    wrap_text,
):
    execution_step_21 = execution_step_20 + 1
    test_urls = ['http://images.cocodataset.org/val2017/000000039769.jpg', 'http://images.cocodataset.org/val2017/000000037777.jpg', 'http://images.cocodataset.org/val2017/000000087038.jpg']
    _fig, _axes = plt.subplots(1, 3, figsize=(15, 6))
    for _url, _ax in zip(test_urls, _axes):
        _caption, image_4 = generate_caption(vlm, _url, image_processor, device)
        _ax.imshow(image_4)
        caption_text = wrap_text(f'생성: {_caption}', 35)
        _ax.set_xlabel(caption_text, fontsize=10, ha='center')
        _ax.set_xticks([])
        _ax.set_yticks([])
    plt.suptitle('학습 세트에 없는 새 이미지에서 VLM 테스트', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.show()
    return (execution_step_21,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 9단계: 모델 구조 이해하기

    VLM 내부에서 어떤 일이 일어나는지 시각화합니다.
    """)
    return


@app.cell
def _(
    dataset,
    device,
    execution_step_21,
    image_processor,
    tokenizer,
    torch,
    vlm,
):
    execution_step_22 = execution_step_21 + 1
    _sample = dataset[0]
    image_5 = _sample['image'].convert('RGB')
    pixel_values = image_processor(image_5, return_tensors='pt').pixel_values.to(device)
    with torch.no_grad():
        _vision_outputs = vlm.vision_encoder(pixel_values=pixel_values)
        vision_features = _vision_outputs.last_hidden_state
        print(f'1단계 - 비전 인코더 출력: {vision_features.shape}')
        print(f'         shape 의미: (배치={vision_features.shape[0]}, 패치={vision_features.shape[1]}, 차원={vision_features.shape[2]})')
        projected = vlm.projector(vision_features)
        print(f'\n2단계 - 투영 후: {projected.shape}')
        print(f'         shape 의미: (배치={projected.shape[0]}, 패치={projected.shape[1]}, 언어 모델 차원={projected.shape[2]})')
        sample_text = 'A photo of'
        text_ids = tokenizer.encode(sample_text, return_tensors='pt').to(device)
        text_embeds = vlm.language_model.get_input_embeddings()(text_ids)
        print(f'\n3단계 - 텍스트 임베딩: {text_embeds.shape}')
        print(f'         shape 의미: (배치={text_embeds.shape[0]}, 토큰={text_embeds.shape[1]}, 언어 모델 차원={text_embeds.shape[2]})')
        combined = torch.cat([projected, text_embeds], dim=1)
        print(f'\n4단계 - 결합한 임베딩: {combined.shape}')
        print(f'         전체 시퀀스: {projected.shape[1]}개 이미지 토큰 + {text_embeds.shape[1]}개 텍스트 토큰 = {combined.shape[1]}')
    return execution_step_22, image_5, vision_features


@app.cell
def _(execution_step_22, image_5, plt, torch, vision_features):
    execution_step_23 = execution_step_22 + 1
    patch_features = vision_features[0, 1:, :]
    patch_norms = torch.norm(patch_features, dim=-1)
    patch_grid = patch_norms.reshape(14, 14).cpu().numpy()
    _fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.imshow(image_5)
    ax1.set_title('원본 이미지')
    ax1.axis('off')
    im = ax2.imshow(patch_grid, cmap='hot')
    ax2.set_title('패치별 시각 특징의 크기')
    ax2.axis('off')
    plt.colorbar(im, ax=ax2, fraction=0.046)
    plt.tight_layout()
    plt.show()
    return (execution_step_23,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 10단계: 나중에 사용할 모델 저장하기

    후속 노트북의 지시 미세 조정이나 객체 탐지 등에 사용할 수 있도록 학습한 모델을 저장합니다.
    """)
    return


@app.cell
def _(
    execution_step_23,
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
    execution_step_24 = execution_step_23 + 1
    save_dir = str(output_dir / 'mini-vlm-flickr8k')
    os.makedirs(save_dir, exist_ok=True)
    # 모델 저장 디렉토리 생성
    torch.save(vlm.projector.state_dict(), f'{save_dir}/projector.pt')
    torch.save({'projector_state_dict': vlm.projector.state_dict(), 'language_model_state_dict': vlm.language_model.state_dict(), 'config': {'vision_model_name': vision_model_name, 'lm_model_name': lm_model_name, 'vision_dim': vision_dim, 'language_dim': language_dim}}, f'{save_dir}/mini_vlm_full.pt')
    tokenizer.save_pretrained(f'{save_dir}/tokenizer')
    # 프로젝터 가중치 저장(주요 학습 구성 요소)
    image_processor.save_pretrained(f'{save_dir}/image_processor')
    print(f'모델 저장 위치: {save_dir}/')
    # 전체 VLM 상태 저장(프로젝터 + 언어 모델 가중치)
    # 쉽게 불러오도록 토크나이저와 이미지 처리기 설정도 저장
    print(f'저장 내용: {os.listdir(save_dir)}')
    return (execution_step_24,)


@app.cell
def _(MiniVLM, VisionProjector, execution_step_24, torch):
    # 원본 코드 셀 순서와 모델·데이터 변경의 실행 순서를 보존합니다.
    execution_step_25 = execution_step_24 + 1
    # 모델 로딩 보조 함수(다음 노트북에서 활용)
    def load_mini_vlm(save_dir, device='cuda'):
        """저장한 MiniVLM 모델을 불러옵니다.
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer, ViTModel, ViTImageProcessor
    
        # 체크포인트 불러오기
        checkpoint = torch.load(f"{save_dir}/mini_vlm_full.pt", map_location=device)
        config = checkpoint['config']
    
        # 구성 요소 재생성
        vision_encoder = ViTModel.from_pretrained(config['vision_model_name'])
        language_model = AutoModelForCausalLM.from_pretrained(config['lm_model_name'])
        tokenizer = AutoTokenizer.from_pretrained(f"{save_dir}/tokenizer")
        image_processor = ViTImageProcessor.from_pretrained(f"{save_dir}/image_processor")
    
        # 프로젝터 재생성 및 가중치 로딩
        projector = VisionProjector(config['vision_dim'], config['language_dim'])
        projector.load_state_dict(checkpoint['projector_state_dict'])
    
        # 언어 모델 가중치 로딩
        language_model.load_state_dict(checkpoint['language_model_state_dict'])
    
        # VLM 생성
        vlm = MiniVLM(vision_encoder, language_model, projector, tokenizer)
        vlm = vlm.to(device)
    
        return vlm, image_processor, tokenizer

    print('후속 노트북에서 모델을 불러올 보조 함수 load_mini_vlm을 정의했습니다.')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 정리

    세 가지 핵심 구성 요소를 갖춘 최소 구성의 시각·언어 모델을 만들었습니다.

    | 구성 요소 | 모델 | 매개변수 | 학습 여부 |
    |-----------|-------|------------|----------|
    | 비전 인코더 | ViT-Large-16-224 | 304M | 고정 |
    | 프로젝션 | 2층 MLP | 약 2M | 학습 |
    | 언어 모델 | SmolLM-360M | 360M | 학습 |

    ### 핵심 내용

    1. **구조**: VLM은 프로젝션 층으로 비전 인코더와 언어 모델을 연결합니다.
    2. **학습**: 비전 인코더는 고정하고 프로젝터와 LLM만 학습합니다.
    3. **데이터**: 이미지·캡션 쌍(Flickr8k)을 통해 이미지 설명을 학습합니다.
    4. **규모 확장**: 상용 VLM은 더 큰 모델과 훨씬 많은 데이터를 사용합니다.
    5. **학습 재개**: Ctrl+C로 중단한 뒤 다시 실행하면 체크포인트에서 재개할 수 있습니다.

    ### 이 최소 모델의 한계

    - 상용 모델의 수백만 장에 비해 적은 학습 이미지(2,000장)
    - 정교한 교차 어텐션 대신 단순한 프로젝션(MLP) 사용
    - 대화형 상호작용을 위한 지시 학습 없음
    - 시각적 그라운딩이나 객체 탐지 기능 없음

    ### 다음 단계

    다음 글에서는 이 모델을 사용해 다음을 실습합니다.
    1. 객체 탐지 데이터로 **지시 미세 조정**하기
    2. "What objects are in this image?"(이 이미지에 어떤 객체가 있나요?) 같은 질문에 답하도록 학습하기
    3. 시각적 그라운딩 살펴보기

    저장된 모델은 실행 산출물의 `mini-vlm-flickr8k/`에 있으며 `load_mini_vlm()`으로 불러올 수 있습니다.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 정리 작업: GPU 메모리 해제

    실습을 마치면 아래 안내에 따라 GPU 메모리를 해제해 다른 작업에 사용할 수 있도록 합니다.
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
