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
    ### 추가 메모리 관리 팁

    **메모리를 정리할 시점:**
    - 학습·평가를 마치고 새 노트북을 시작하기 전
    - 큰 모델 사이를 전환할 때
    - CUDA 메모리 부족 오류가 발생했을 때

    **각 명령의 역할:**
    - `del model`: 모델을 가리키는 Python 참조 제거
    - `gc.collect()`: 사용하지 않는 Python 객체 메모리 회수
    - `torch.cuda.empty_cache()`: 캐시된 GPU 메모리를 시스템에 반환

    **메모리 사용량 확인:**
    터미널에서 `nvidia-smi`를 실행하거나 Python에서 다음을 실행합니다.
    ```python
    print(f"할당된 메모리: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    print(f"예약된 메모리: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
    print(f"최대 할당 메모리: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
    ```

    **참고:** 메모리를 정리한 뒤 모델을 다시 사용하려면 모델을 다시 불러와야 합니다. marimo에서는 셀에 보관된 참조도 해제하도록 커널을 재시작하세요.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### GPU 메모리 해제 (marimo)

    실습을 마치면 marimo의 **커널 재시작**을 사용하세요. marimo는 셀 출력과 의존성에 객체 참조를 보관하므로 Jupyter의 `globals()`/`del` 방식 대신 커널을 재시작해 모델과 GPU 메모리를 해제합니다.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 다중 이미지 시각·언어 모델: 비교, 추론, 이해

    Nipun Batra · 2025-12-27
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 소개

    지금까지 다음을 처리하는 VLM을 만들었습니다.
    1. [이미지 캡션](part-01-minimal-vlm.py)
    2. [객체 탐지](part-02-object-detection.py)
    3. [시각적 질의응답](part-03-visual-qa.py)
    4. [멀티태스크 학습](part-04-multi-task.py)

    하지만 이 모델들은 모두 **한 번에 이미지 한 장만** 처리했습니다. 다음과 같은 작업을 하려면 어떻게 해야 할까요?
    - 시간에 따른 이미지 비교: *이 두 위성 이미지 사이에 무엇이 바뀌었나요?*
    - 시퀀스 분석: *이 시계열 이미지에서 어떤 변화가 일어나나요?*
    - 다중 이미지 추론: *어느 이미지에 산불이 보이나요?*

    이것이 **시계열·다중 이미지 VLM**의 역할입니다. 지구 관측용 [TEOChat](https://arxiv.org/abs/2410.06234), GPT-4V, Gemini 등의 모델에서 아이디어를 얻었습니다.

    ### 다중 이미지 구조

    TEOChat의 접근(그림 2/3)을 따라 **시간에 걸쳐 공유하는 비전 인코딩**과 **이미지 식별자**를 사용합니다.

    ```
    [IMG1 토큰] <image_1> [IMG2 토큰] <image_2> [텍스트: "What changed?"]
           ↓                       ↓                            ↓
      공유 비전 인코더        공유 비전 인코더                  언어 모델
    ```

    **TEOChat의 핵심 설계 원칙:**
    1. **시간에 걸친 가중치 공유**: 같은 비전 인코더가 모든 이미지를 처리합니다.
    2. **이미지 식별자**: 특정 이미지를 가리키는 특수 토큰을 사용합니다.
    3. **시간적 추론**: 모델이 시퀀스의 이미지들을 비교하고 추론하도록 학습합니다.

    ### 예시

    | 작업 | 입력 | 출력 |
    |------|-------|--------|
    | **비교** | `[IMG1] <image_1> [IMG2] <image_2>: What changed?` | `"The forest is now cleared"`(숲이 사라짐) |
    | **선택** | `Which shows a cat? [IMG1] [IMG2] [IMG3]` | `"Image 2"`(두 번째 이미지) |
    | **시간적 변화** | `What happens? [IMG1] [IMG2] [IMG3]` | `"Vegetation decreases over time"`(시간이 지나며 식생 감소) |
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
    import os
    import warnings
    warnings.filterwarnings('ignore')


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'사용 장치: {device}')
    return (
        AutoModelForCausalLM,
        AutoTokenizer,
        DataLoader,
        Dataset,
        F,
        ViTImageProcessor,
        ViTModel,
        device,
        execution_step_1,
        load_dataset,
        nn,
        os,
        plt,
        random,
        torch,
        tqdm,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1단계: 다중 이미지 VLM 구조

    단일 이미지 VLM과 다른 핵심 사항은 다음과 같습니다.
    1. 순전파에서 **여러 이미지 입력받기**
    2. 텍스트 토큰 사이에 **이미지 토큰을 끼워 넣기**
    3. **이미지 위치 표식** 사용(예: `<image_1>`, `<image_2>`)
    """)
    return


@app.cell
def _(execution_step_1, nn, torch):
    # 원본 코드 셀 순서와 모델·데이터 변경의 실행 순서를 보존합니다.
    execution_step_2 = execution_step_1 + 1
    # 이전과 같은 프로젝터

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

    return VisionProjector, execution_step_2


@app.cell
def _(
    AutoModelForCausalLM,
    AutoTokenizer,
    F,
    ViTModel,
    VisionProjector,
    execution_step_2,
    nn,
    torch,
):
    # 원본 코드 셀 순서와 모델·데이터 변경의 실행 순서를 보존합니다.
    execution_step_3 = execution_step_2 + 1
    # 교차 인코딩을 사용하는 다중 이미지 VLM

    class MultiImageVLM(nn.Module):
        """이미지·텍스트 교차 인코딩으로 여러 이미지를 처리하는 VLM입니다.
        """
    
        # 이미지 위치용 특수 토큰
        IMAGE_TOKEN_TEMPLATE = "<image_{}>"
    
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
            """단일 이미지를 인코딩합니다.
            """
            with torch.no_grad():
                vision_outputs = self.vision_encoder(pixel_values=pixel_values)
            image_features = vision_outputs.last_hidden_state
            projected = self.projector(image_features)
            return projected
    
        def encode_images(self, pixel_values_list):
            """여러 이미지를 인코딩합니다.
            
            인자:
                pixel_values_list: (batch, 3, 224, 224) 텐서 목록
            
            반환:
                (batch, num_patches, hidden_dim) 텐서 목록
            """
            return [self.encode_image(pv) for pv in pixel_values_list]
    
        def forward(
            self,
            pixel_values_list,  # 이미지 텐서 목록
            input_ids,
            attention_mask,
            image_positions = None,  # 시퀀스에서 각 이미지를 삽입할 위치
            labels = None,
        ):
            """여러 이미지를 사용하는 순전파입니다.
            
            인자:
                pixel_values_list: 이미지별 [batch, 3, 224, 224] 텐서 목록
                input_ids: 텍스트 토큰 [batch, seq_len]
                attention_mask: 어텐션 마스크 [batch, seq_len]
                image_positions: 이미지를 삽입할 위치 목록
                                 None이면 모든 이미지를 앞에 배치
                labels: 정답 레이블 [batch, seq_len]
            """
            batch_size = input_ids.shape[0]
        
            # 모든 이미지 인코딩
            image_embeds_list = self.encode_images(pixel_values_list)
        
            # 텍스트 임베딩 추출
            text_embeds = self.language_model.get_input_embeddings()(input_ids)
        
            # 단순한 전략: 모든 이미지를 텍스트 앞에 배치
            # [IMG1 토큰] [IMG2 토큰] ... [텍스트 토큰]
            combined_embeds = torch.cat(image_embeds_list + [text_embeds], dim=1)
        
            # 이미지용 어텐션 마스크 생성
            total_image_tokens = sum(img_emb.shape[1] for img_emb in image_embeds_list)
            image_attention = torch.ones(
                (batch_size, total_image_tokens),
                dtype=attention_mask.dtype,
                device=attention_mask.device
            )
            combined_attention = torch.cat([image_attention, attention_mask], dim=1)
        
            # 레이블 생성
            if labels is not None:
                image_labels = torch.full(
                    (batch_size, total_image_tokens),
                    fill_value=-100,
                    dtype=labels.dtype,
                    device=labels.device
                )
                combined_labels = torch.cat([image_labels, labels], dim=1)
            else:
                combined_labels = None
        
            # 언어 모델 순전파
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
            pixel_values_list,  # 이미지 텐서 목록
            prompt,
            max_new_tokens = 50,
            temperature = 0.7,
            do_sample = True,
        ):
            """여러 이미지와 프롬프트에서 텍스트를 생성합니다.
            """
            self.eval()
        
            # 모든 이미지 인코딩
            image_embeds_list = self.encode_images(pixel_values_list)
        
            # 모든 이미지 임베딩 연결
            all_image_embeds = torch.cat(image_embeds_list, dim=1)  # [1, total_patches, hidden]
        
            # 프롬프트 인코딩
            prompt_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(pixel_values_list[0].device)
            generated_ids = prompt_ids.clone()
        
            # 토큰을 하나씩 생성
            for _ in range(max_new_tokens):
                current_embeds = self.language_model.get_input_embeddings()(generated_ids)
                full_embeds = torch.cat([all_image_embeds, current_embeds], dim=1)
            
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

    return MultiImageVLM, execution_step_3


@app.cell
def _(
    AutoModelForCausalLM,
    AutoTokenizer,
    MultiImageVLM,
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

    vlm = MultiImageVLM(vision_encoder, language_model, projector, tokenizer).to(device)
    print(f"비전 인코더: {vision_model_name} (은닉 차원={vision_dim})")
    print(f"언어 모델: {lm_model_name} (은닉 차원={language_dim})")
    print(f"모델 로딩 장치: {device}")
    print(f"학습 가능한 매개변수: {sum(p.numel() for p in vlm.parameters() if p.requires_grad):,}")
    return (execution_step_4, image_processor, language_dim, lm_model_name, tokenizer, vision_dim, vision_model_name, vlm)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2단계: 다중 이미지 데이터셋 만들기

    합성 다중 이미지 작업을 만듭니다.
    1. **이미지 비교**: "What's different between image 1 and image 2?"(이미지 1과 2는 무엇이 다른가요?)
    2. **이미지 선택**: "Which image shows a dog?"(어느 이미지에 개가 있나요?)
    3. **시퀀스 이해**: "Describe what happens across these images."(이 이미지들에 걸쳐 일어나는 일을 설명하세요.)
    """)
    return


@app.cell
def _(execution_step_4, load_dataset):
    # 원본 코드 셀 순서와 모델·데이터 변경의 실행 순서를 보존합니다.
    execution_step_5 = execution_step_4 + 1
    # 다중 이미지 작업 생성을 위한 Flickr8k 로딩
    flickr_dataset = load_dataset("jxie/flickr8k", split="train").shuffle(seed=42)

    print(f'Flickr8k 데이터셋: {len(flickr_dataset)}개 샘플')
    print(f'이 데이터셋에서 다중 이미지 작업을 만듭니다.')
    return execution_step_5, flickr_dataset


@app.cell
def _(Dataset, execution_step_5, random):
    # 원본 코드 셀 순서와 모델·데이터 변경의 실행 순서를 보존합니다.
    execution_step_6 = execution_step_5 + 1
    # 이미지 비교 데이터셋
    # 작업: 두 이미지 중 첫 이미지에 비추어 두 번째 이미지 설명

    class ImageComparisonDataset(Dataset):
        """두 이미지를 비교하고 차이를 설명합니다.
        """
    
        COMPARISON_PROMPTS = [
            "Image 1: [IMG1] Image 2: [IMG2] What's in image 2?",
            "Compare these images: [IMG1] [IMG2]. Describe the second image.",
            "First image: [IMG1] Second image: [IMG2] Describe the second one.",
        ]
    
        def __init__(self, hf_dataset, image_processor, tokenizer, num_pairs=500, max_length=128):
            self.dataset = hf_dataset
            self.image_processor = image_processor
            self.tokenizer = tokenizer
            self.num_pairs = num_pairs
            self.max_length = max_length
        
            # 무작위 쌍 생성
            self.pairs = []
            for i in range(num_pairs):
                idx1 = random.randint(0, len(hf_dataset) - 1)
                idx2 = random.randint(0, len(hf_dataset) - 1)
                while idx2 == idx1:
                    idx2 = random.randint(0, len(hf_dataset) - 1)
                self.pairs.append((idx1, idx2))
    
        def __len__(self):
            return len(self.pairs)
    
        def __getitem__(self, idx):
            idx1, idx2 = self.pairs[idx]
        
            # 이미지 추출
            img1 = self.dataset[idx1]['image'].convert('RGB')
            img2 = self.dataset[idx2]['image'].convert('RGB')
        
            # 이미지 처리
            pv1 = self.image_processor(img1, return_tensors="pt").pixel_values.squeeze(0)
            pv2 = self.image_processor(img2, return_tensors="pt").pixel_values.squeeze(0)
        
            # 두 번째 이미지의 캡션 추출(생성 목표)
            caption2 = self.dataset[idx2][f'caption_{random.randint(0, 4)}']
        
            # 프롬프트 생성
            prompt = random.choice(self.COMPARISON_PROMPTS)
            full_text = f"{prompt} {caption2}{self.tokenizer.eos_token}"
        
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
        
            # 프롬프트 마스킹
            prompt_tokens = self.tokenizer.encode(prompt, add_special_tokens=False)
            prompt_len = len(prompt_tokens)
        
            labels = input_ids.clone()
            labels[:prompt_len] = -100
            labels[attention_mask == 0] = -100
        
            return {
                'pixel_values_list': [pv1, pv2],  # 이미지 2장 목록
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'labels': labels,
            }

    return ImageComparisonDataset, execution_step_6


@app.cell
def _(Dataset, execution_step_6, random):
    # 원본 코드 셀 순서와 모델·데이터 변경의 실행 순서를 보존합니다.
    execution_step_7 = execution_step_6 + 1
    # 이미지 선택 데이터셋
    # 작업: N개 이미지 중 설명에 해당하는 이미지 찾기

    class ImageSelectionDataset(Dataset):
        """설명에 해당하는 이미지를 선택합니다.
        """
    
        def __init__(self, hf_dataset, image_processor, tokenizer, num_samples=500, num_images=3, max_length=128):
            self.dataset = hf_dataset
            self.image_processor = image_processor
            self.tokenizer = tokenizer
            self.num_samples = num_samples
            self.num_images = num_images
            self.max_length = max_length
        
            # 무작위 이미지 집합 생성
            self.image_sets = []
            for i in range(num_samples):
                indices = random.sample(range(len(hf_dataset)), num_images)
                target_idx = random.randint(0, num_images - 1)  # 설명할 이미지 선택
                self.image_sets.append((indices, target_idx))
    
        def __len__(self):
            return len(self.image_sets)
    
        def __getitem__(self, idx):
            indices, target_idx = self.image_sets[idx]
        
            # 이미지 추출
            images = [self.dataset[i]['image'].convert('RGB') for i in indices]
            pixel_values_list = [
                self.image_processor(img, return_tensors="pt").pixel_values.squeeze(0)
                for img in images
            ]
        
            # 대상 이미지 설명 추출
            target_caption = self.dataset[indices[target_idx]][f'caption_{random.randint(0, 4)}']
        
            # 프롬프트 형식: "Which image shows {description}? Image 1: [IMG1] Image 2: [IMG2] ..."
            # 답변 형식: "Image {target_idx + 1}"
            prompt = f"Which image shows: {target_caption}?"
            answer = f"Image {target_idx + 1}"
            full_text = f"{prompt} {answer}{self.tokenizer.eos_token}"
        
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
        
            # 프롬프트 마스킹
            prompt_tokens = self.tokenizer.encode(prompt, add_special_tokens=False)
            prompt_len = len(prompt_tokens)
        
            labels = input_ids.clone()
            labels[:prompt_len] = -100
            labels[attention_mask == 0] = -100
        
            return {
                'pixel_values_list': pixel_values_list,  # N개 이미지 목록
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'labels': labels,
            }

    return ImageSelectionDataset, execution_step_7


@app.cell
def _(
    ImageComparisonDataset,
    ImageSelectionDataset,
    execution_step_7,
    flickr_dataset,
    image_processor,
    tokenizer,
):
    # 원본 코드 셀 순서와 모델·데이터 변경의 실행 순서를 보존합니다.
    execution_step_8 = execution_step_7 + 1
    # 데이터셋 생성
    comparison_dataset = ImageComparisonDataset(flickr_dataset, image_processor, tokenizer, num_pairs=800)
    selection_dataset = ImageSelectionDataset(flickr_dataset, image_processor, tokenizer, num_samples=400, num_images=3)

    print(f'이미지 비교 쌍: {len(comparison_dataset)}')
    print(f'이미지 선택 샘플: {len(selection_dataset)}')
    print(f'전체 다중 이미지 샘플: {len(comparison_dataset) + len(selection_dataset)}')
    return comparison_dataset, execution_step_8, selection_dataset


@app.cell
def _(execution_step_8, torch):
    # 원본 코드 셀 순서와 모델·데이터 변경의 실행 순서를 보존합니다.
    execution_step_9 = execution_step_8 + 1
    # 가변 개수의 이미지 처리를 위한 사용자 정의 collate 함수

    def multi_image_collate_fn(batch):
        """다중 이미지 배치를 결합하는 함수입니다.
        
        샘플마다 이미지 개수가 다른 경우를 처리합니다.
        """
        # 배치에서 최대 이미지 수 찾기
        max_images = max(len(item['pixel_values_list']) for item in batch)
    
        # 이미지 수가 적은 샘플을 0으로 패딩
        batch_pixel_values_list = [[] for _ in range(max_images)]
    
        for item in batch:
            num_images = len(item['pixel_values_list'])
            for i in range(max_images):
                if i < num_images:
                    batch_pixel_values_list[i].append(item['pixel_values_list'][i])
                else:
                    # 0으로 패딩
                    batch_pixel_values_list[i].append(torch.zeros_like(item['pixel_values_list'][0]))
    
        # 각 이미지 위치별로 텐서 쌓기
        batch_pixel_values_list = [torch.stack(imgs) for imgs in batch_pixel_values_list]
    
        # 다른 필드는 일반 방식으로 쌓기
        input_ids = torch.stack([item['input_ids'] for item in batch])
        attention_mask = torch.stack([item['attention_mask'] for item in batch])
        labels = torch.stack([item['labels'] for item in batch])
    
        return {
            'pixel_values_list': batch_pixel_values_list,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
        }

    return execution_step_9, multi_image_collate_fn


@app.cell
def _(
    DataLoader,
    comparison_dataset,
    execution_step_9,
    multi_image_collate_fn,
    selection_dataset,
):
    # 원본 코드 셀 순서와 모델·데이터 변경의 실행 순서를 보존합니다.
    execution_step_10 = execution_step_9 + 1
    # 데이터로더 생성
    from torch.utils.data import ConcatDataset

    multi_image_dataset = ConcatDataset([comparison_dataset, selection_dataset])

    train_loader = DataLoader(
        multi_image_dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=multi_image_collate_fn,
        num_workers=0,
    )

    print(f'전체 학습 샘플: {len(multi_image_dataset)}')
    print(f'배치 수: {len(train_loader)}')
    return execution_step_10, train_loader


@app.cell
def _(execution_step_10, tokenizer, train_loader):
    # 원본 코드 셀 순서와 모델·데이터 변경의 실행 순서를 보존합니다.
    execution_step_11 = execution_step_10 + 1
    # 데이터로더 테스트
    batch = next(iter(train_loader))
    print(f'배치 pixel_values_list 길이: {len(batch['pixel_values_list'])}')
    print(f'첫 이미지 배치 shape: {batch['pixel_values_list'][0].shape}')
    print(f'입력 ID shape: {batch['input_ids'].shape}')

    # 샘플 디코딩
    print(f'\n샘플 텍스트:')
    print(tokenizer.decode(batch['input_ids'][0], skip_special_tokens=True)[:150])
    return (execution_step_11,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3단계: 다중 이미지 학습
    """)
    return


@app.cell
def _(device, execution_step_11, torch, tqdm):
    # 원본 코드 셀 순서와 모델·데이터 변경의 실행 순서를 보존합니다.
    execution_step_12 = execution_step_11 + 1
    def train_multi_image_vlm(model, train_loader, num_epochs=6, lr=1e-4):
        """다중 이미지 VLM을 학습합니다.
        """
    
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable_params, lr=lr)
    
        model.train()
        model.vision_encoder.eval()
    
        losses = []
    
        for epoch in range(num_epochs):
            epoch_loss = 0
            progress_bar = tqdm(train_loader, desc=f'에포크 {epoch + 1}/{num_epochs}')
        
            for batch in progress_bar:
                # 장치로 이동
                pixel_values_list = [pv.to(device) for pv in batch['pixel_values_list']]
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
            
                outputs = model(
                    pixel_values_list=pixel_values_list,
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
    
        return losses

    return execution_step_12, train_multi_image_vlm


@app.cell
def _(execution_step_12, train_loader, train_multi_image_vlm, vlm):
    # 원본 코드 셀 순서와 모델·데이터 변경의 실행 순서를 보존합니다.
    execution_step_13 = execution_step_12 + 1
    # 다중 이미지 모델 학습
    losses = train_multi_image_vlm(vlm, train_loader, num_epochs=6, lr=1e-4)
    return execution_step_13, losses


@app.cell
def _(execution_step_13, losses, plt):
    # 원본 코드 셀 순서와 모델·데이터 변경의 실행 순서를 보존합니다.
    execution_step_14 = execution_step_13 + 1
    # 학습 손실 그래프
    plt.figure(figsize=(8, 4))
    plt.plot(range(1, len(losses)+1), losses, marker='o', linewidth=2)
    plt.xlabel('에포크')
    plt.ylabel('손실')
    plt.title('다중 이미지 VLM 학습 손실')
    plt.grid(True, alpha=0.3)
    plt.show()
    return (execution_step_14,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4단계: 다중 이미지 처리 능력 테스트
    """)
    return


@app.cell
def _(device, execution_step_14, flickr_dataset, image_processor, plt, vlm):
    # 원본 코드 셀 순서와 모델·데이터 변경의 실행 순서를 보존합니다.
    execution_step_15 = execution_step_14 + 1
    # 테스트 1: 이미지 비교
    print('=' * 70)
    print('테스트 1: 이미지 비교')
    print('=' * 70)
    idx1, idx2 = (2000, 2005)
    # 무작위 이미지 2장 추출
    img1 = flickr_dataset[idx1]['image']  # 학습 범위 밖의 샘플
    img2 = flickr_dataset[idx2]['image']
    caption2 = flickr_dataset[idx2]['caption_0']
    pv1 = image_processor(img1, return_tensors='pt').pixel_values.to(device)
    pv2 = image_processor(img2, return_tensors='pt').pixel_values.to(device)
    # 이미지 처리
    _prompt = "Image 1: [IMG1] Image 2: [IMG2] What's in image 2?"
    _response = vlm.generate([pv1, pv2], _prompt, max_new_tokens=40, temperature=0.7)
    if 'image 2?' in _response:
    # 생성
        _answer = _response.split('image 2?')[-1].strip()
    else:
        _answer = _response
    # 답변 추출
    print(f'\n프롬프트: {_prompt}')
    print(f'생성: {_answer}')
    print(f'정답 캡션: {caption2}')
    _fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    ax1.imshow(img1)
    ax1.set_title('이미지 1', fontsize=12)
    ax1.axis('off')
    ax2.imshow(img2)
    ax2.set_title(f'이미지 2\n모델: {_answer[:50]}...', fontsize=10)
    # 이미지 표시
    ax2.axis('off')
    plt.tight_layout()
    plt.show()
    return (execution_step_15,)


@app.cell
def _(device, execution_step_15, flickr_dataset, image_processor, plt, vlm):
    # 원본 코드 셀 순서와 모델·데이터 변경의 실행 순서를 보존합니다.
    execution_step_16 = execution_step_15 + 1
    # 테스트 2: 이미지 선택
    print('\n' + '=' * 70)
    print('테스트 2: 이미지 선택')
    print('=' * 70)
    indices = [2010, 2015, 2020]
    # 이미지 3장 추출
    images = [flickr_dataset[i]['image'] for i in indices]
    captions = [flickr_dataset[i]['caption_0'] for i in indices]
    target_idx = 1
    target_caption = captions[target_idx]
    # 설명할 이미지 하나 선택
    pixel_values_list = [image_processor(img, return_tensors='pt').pixel_values.to(device) for img in images]
    _prompt = f'Which image shows: {target_caption}?'
    _response = vlm.generate(pixel_values_list, _prompt, max_new_tokens=10, do_sample=False)
    # 이미지 처리
    if 'shows:' in _response:
        _answer = _response.split('shows:')[-1].split('?')[-1].strip()
    else:
        _answer = _response
    print(f'\n프롬프트: {_prompt}')
    # 생성
    print(f'생성: {_answer}')
    print(f'정답: 이미지 {target_idx + 1}')
    _fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for i, (img, ax) in enumerate(zip(images, axes)):
        ax.imshow(img)
        title = f'이미지 {i + 1}'
        if i == target_idx:
            title += ' ✓ (Target)'
        ax.set_title(title, fontsize=12, color='green' if i == target_idx else 'black')
        ax.axis('off')
    plt.suptitle(f'질문: {target_caption[:50]}...\n모델 답변: {_answer}', fontsize=10)
    plt.tight_layout()
    # 이미지 표시
    plt.show()
    return (execution_step_16,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5단계: 다중 이미지 모델 저장하기
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
    # 모델 저장
    save_dir = str(output_dir / "mini-vlm-multiimage")
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
    }, f"{save_dir}/mini_vlm_multiimage.pt")

    tokenizer.save_pretrained(f"{save_dir}/tokenizer")
    image_processor.save_pretrained(f"{save_dir}/image_processor")

    print(f'다중 이미지 모델 저장 위치: {save_dir}/')
    print(f'저장 내용: {os.listdir(save_dir)}')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 정리

    지구 관측용 TEOChat에서 아이디어를 얻어 **시계열·다중 이미지 VLM**을 만들었습니다.

    ### 구현한 내용

    1. **다중 이미지 구조**: 가변 개수의 이미지 처리
    2. **시간에 걸쳐 공유하는 인코딩**: TEOChat처럼 같은 비전 인코더가 모든 이미지 처리
    3. **두 가지 다중 이미지 작업**:
       - 이미지 비교(이미지 2장)
       - 이미지 선택(이미지 3장)

    ### TEOChat에서 얻은 핵심 내용

    - **시간에 걸친 가중치 공유**: 인코더를 따로 두는 것보다 효율적입니다.
    - **단순 연결도 유효**: 모든 이미지를 텍스트 앞에 배치해 기준선으로 삼습니다.
    - **시간적 추론의 발현**: 모델이 이미지 비교를 암묵적으로 학습합니다.
    - **합성 데이터**: 단일 이미지 데이터셋에서 다중 이미지 작업을 만듭니다.

    ### 구조의 발전 과정

    ```
    단일 이미지: [IMG 토큰] [텍스트 토큰]
                        ↓
    이 실습의 다중 이미지: [IMG1 토큰] [IMG2 토큰] [IMG3 토큰] [텍스트 토큰]
                        ↓              ↓              ↓
    TEOChat의 다중 이미지: [IMG1 토큰] <image_1> [IMG2 토큰] <image_2> [텍스트]
    ```

    ### TEOChat과의 비교

    | 특징 | TEOChat | 이 실습 모델 |
    |---------|---------|-----------|
    | **시간에 걸친 인코더 공유** | ✓ | ✓ |
    | **다중 이미지** | ✓ | ✓ |
    | **이미지 식별자** | ✓ (`<image_1>`) | ✗ (후속 과제) |
    | **이미지·텍스트 토큰 교차 배치** | ✓ | 부분 지원 |
    | **규모** | LLaMA 2 (7B) | SmolLM (135M) |
    | **분야** | 지구 관측 | 일반 이미지 |

    ### 한계

    - **이미지 식별자 없음**: 특정 이미지를 위치로 가리킬 수 없음
    - **단순 연결**: 실제 교차 배치가 아니라 이미지를 텍스트 앞에 배치
    - **소형 모델**: 추론 능력 제한
    - **합성 작업**: 실제 시계열·지구 관측 데이터가 아님

    ### 다음 단계: TEOChat 방식의 구조로 확장

    1. **이미지 토큰 추가**: `<image_1>`, `<image_2>` 식별자 구현
    2. **실제 교차 배치**: 텍스트의 임의 위치에 이미지 삽입
    3. **시계열 데이터셋**: 실제 변화 탐지·시계열 데이터 사용
    4. **영상 이해**: 영상 프레임을 시계열로 처리
    5. **이미지 간 어텐션**: 이미지 사이를 명시적으로 참조하도록 구성
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 참고 자료

    - [TEOChat: 시간적 지구 관측 데이터를 위한 대규모 시각·언어 어시스턴트](https://arxiv.org/abs/2410.06234): 그림 2/3에서 시간에 걸쳐 공유하는 인코더 구조 확인
    - [처음부터 만드는 최소 구성의 VLM](part-01-minimal-vlm.py)
    - [멀티태스크 VLM](part-04-multi-task.py)
    - [Flamingo: 퓨샷 학습을 위한 시각·언어 모델](https://arxiv.org/abs/2204.14198)
    - [NLVR2: 시각적 추론을 위한 자연어](https://lil.nlp.cornell.edu/nlvr/)
    - [Spot-the-Diff: 이미지 차이 캡션 생성](https://github.com/harsh19/spot-the-diff)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## GPU 메모리 관리

    이 노트북을 실행한 뒤 다른 노트북이나 작업을 시작하기 전에 GPU 메모리를 해제할 수 있습니다.
    """)
    return


if __name__ == "__main__":
    app.run()
