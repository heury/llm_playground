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
    # 멀티태스크 시각·언어 모델: 캡션, 탐지, 질의응답

    Nipun Batra · 2025-12-26
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 소개

    이전 노트북에서는 작업별 VLM을 만들었습니다.
    1. [캡션 모델](part-01-minimal-vlm.py): 이미지 설명
    2. [객체 탐지 모델](part-02-object-detection.py): 경계 상자를 포함한 JSON 출력
    3. [VQA 모델](part-03-visual-qa.py): 질문에 답변

    이제 세 기능을 **하나의 통합 멀티태스크 모델**로 결합합니다. GPT-4V, Gemini, LLaVA 같은 현대 VLM이 여러 시각·언어 작업을 처리하는 방식과 비슷합니다.

    ### 멀티태스크 모델의 장점

    - **하나의 모델**로 여러 작업 처리
    - **공유 표현**을 통한 일반화 개선
    - **작업 간 전이**: 한 작업에서 배운 내용이 다른 작업에도 도움
    - **실용성**: 세 모델 대신 하나의 모델만 배포

    ### 하나의 모델로 세 가지 작업 수행

    | 작업 | 입력 | 출력 |
    |------|-------|--------|
    | **캡션** | `"Describe this image."` | `"A dog playing in the park."` |
    | **객체 탐지(OD)** | `"What objects? Output JSON."` | `{"objects": [{"label": "dog", "bbox": [...]}]}` |
    | **VQA** | `"Question: What color? Answer:"` | `"brown"` |

    예시의 영어 프롬프트와 출력은 실제 학습 형식을 보여 줍니다.
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
    from torch.utils.data import DataLoader, Dataset, ConcatDataset
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
    import json
    import textwrap
    import random
    import os
    import warnings
    warnings.filterwarnings('ignore')

    import supervision as sv


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'사용 장치: {device}')
    return (
        AutoModelForCausalLM,
        AutoTokenizer,
        ConcatDataset,
        DataLoader,
        Dataset,
        F,
        Image,
        ViTImageProcessor,
        ViTModel,
        device,
        execution_step_1,
        json,
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
    ## 1단계: 기반 VLM 구조 불러오기

    캡션을 학습한 모델에서 시작해 세 작업을 동시에 미세 조정합니다.
    """)
    return


@app.cell
def _(
    AutoModelForCausalLM,
    AutoTokenizer,
    F,
    ViTModel,
    execution_step_1,
    nn,
    torch,
):
    # 원본 코드 셀 순서와 모델·데이터 변경의 실행 순서를 보존합니다.
    execution_step_2 = execution_step_1 + 1
    # 모델 구조(앞과 동일)

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
            max_new_tokens = 50,
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

    return MiniVLM, VisionProjector, execution_step_2


@app.cell
def _(
    AutoModelForCausalLM,
    AutoTokenizer,
    MiniVLM,
    ViTImageProcessor,
    ViTModel,
    VisionProjector,
    device,
    execution_step_2,
    os,
    output_dir,
    torch,
):
    # 원본 코드 셀 순서와 모델·데이터 변경의 실행 순서를 보존합니다.
    execution_step_3 = execution_step_2 + 1
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

    vlm = MiniVLM(vision_encoder, language_model, projector, tokenizer).to(device)
    print(f"비전 인코더: {vision_model_name} (은닉 차원={vision_dim})")
    print(f"언어 모델: {lm_model_name} (은닉 차원={language_dim})")
    print(f"모델 로딩 장치: {device}")
    print(f"학습 가능한 매개변수: {sum(p.numel() for p in vlm.parameters() if p.requires_grad):,}")
    return (execution_step_3, image_processor, language_dim, lm_model_name, tokenizer, vision_dim, vision_model_name, vlm)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2단계: 멀티태스크 데이터셋 만들기

    세 데이터셋을 결합합니다.
    1. 캡션용 **Flickr8k**(1,000개 샘플)
    2. 객체 탐지용 **Animals-OD**(400개 샘플)
    3. 질의응답용 **VQAv2**(1,000개 샘플)

    각 샘플에는 작업별 프롬프트 형식을 사용합니다.
    """)
    return


@app.cell
def _(execution_step_3, load_dataset):
    # 원본 코드 셀 순서와 모델·데이터 변경의 실행 순서를 보존합니다.
    execution_step_4 = execution_step_3 + 1
    # 세 데이터셋 모두 불러오기
    print('데이터셋 로딩 중...')
    caption_dataset = load_dataset('jxie/flickr8k', split='train').shuffle(seed=42).select(range(1000))
    # 1. 캡션 데이터셋
    print(f'캡션 데이터셋: {len(caption_dataset)}개 샘플')
    od_train = load_dataset('Francesco/animals-ij5d2', split='train')
    od_val = load_dataset('Francesco/animals-ij5d2', split='validation')
    # 2. 객체 탐지 데이터셋
    from datasets import concatenate_datasets
    od_dataset = concatenate_datasets([od_train, od_val]).select(range(400))
    od_category_names = od_dataset.features['objects']['category'].feature.names
    print(f'객체 탐지 데이터셋: {len(od_dataset)}개 샘플')
    vqa_dataset_stream = load_dataset('lmms-lab/VQAv2', split='validation', streaming=True)
    vqa_samples = []
    for _i, _sample in enumerate(vqa_dataset_stream):
    # 3. VQA 데이터셋
        if _i >= 1000:
            break
        vqa_samples.append(_sample)
    print(f'VQA 데이터셋: {len(vqa_samples)}개 샘플')
    print(f'\n전체 학습 샘플: {len(caption_dataset) + len(od_dataset) + len(vqa_samples)}')
    return (
        caption_dataset,
        execution_step_4,
        od_category_names,
        od_dataset,
        vqa_samples,
    )


@app.cell
def _(execution_step_4, json):
    # 원본 코드 셀 순서와 모델·데이터 변경의 실행 순서를 보존합니다.
    execution_step_5 = execution_step_4 + 1
    # 이전 노트북의 보조 함수

    def get_most_common_answer(answers):
        """VQA 주석에서 가장 빈도가 높은 답변을 구합니다.
        """
        answer_counts = {}
        for ans in answers:
            a = ans['answer']
            answer_counts[a] = answer_counts.get(a, 0) + 1
        return max(answer_counts, key=answer_counts.get)

    def create_od_json(objects, width, height, category_names):
        """경계 상자 주석을 JSON 형식으로 변환합니다.
        """
        result = {"objects": []}
    
        for bbox, cat_id in zip(objects['bbox'], objects['category']):
            x, y, w, h = bbox
            # 0~1 범위로 정규화
            norm_bbox = [
                round(x / width, 3),
                round(y / height, 3),
                round(w / width, 3),
                round(h / height, 3)
            ]
            result["objects"].append({
                "label": category_names[cat_id],
                "bbox": norm_bbox
            })
    
        return json.dumps(result)

    return create_od_json, execution_step_5, get_most_common_answer


@app.cell
def _(
    Dataset,
    create_od_json,
    execution_step_5,
    get_most_common_answer,
    random,
):
    # 원본 코드 셀 순서와 모델·데이터 변경의 실행 순서를 보존합니다.
    execution_step_6 = execution_step_5 + 1
    # 작업별 프롬프트를 사용하는 멀티태스크 데이터셋
    # 중요: 배치 문제를 피하도록 모든 작업에서 같은 max_length 사용

    UNIFIED_MAX_LENGTH = 256  # 모든 작업에 가장 긴 max_length 사용

    class CaptionTaskDataset(Dataset):
        """캡션 작업용 데이터셋입니다.
        """
    
        def __init__(self, hf_dataset, image_processor, tokenizer, max_length=UNIFIED_MAX_LENGTH):
            self.dataset = hf_dataset
            self.image_processor = image_processor
            self.tokenizer = tokenizer
            self.max_length = max_length
            self.task_name = "caption"
    
        def __len__(self):
            return len(self.dataset)
    
        def __getitem__(self, idx):
            item = self.dataset[idx]
        
            # 이미지 처리
            image = item['image'].convert('RGB')
            pixel_values = self.image_processor(image, return_tensors="pt").pixel_values.squeeze(0)
        
            # 무작위 캡션
            caption_idx = random.randint(0, 4)
            caption = item[f'caption_{caption_idx}']
        
            # 형식: "Describe this image. {caption}"
            prompt = "Describe this image."
            full_text = f"{prompt} {caption}{self.tokenizer.eos_token}"
        
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
        
            # 프롬프트 마스킹(응답만 학습)
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
                'task': self.task_name,
            }


    class ODTaskDataset(Dataset):
        """객체 탐지 작업용 데이터셋입니다.
        """
    
        OD_PROMPTS = [
            "What objects are in this image? Output as JSON.",
            "Detect all objects and return JSON with bounding boxes.",
            "List objects with locations in JSON format.",
        ]
    
        def __init__(self, hf_dataset, image_processor, tokenizer, category_names, max_length=UNIFIED_MAX_LENGTH):
            self.dataset = hf_dataset
            self.image_processor = image_processor
            self.tokenizer = tokenizer
            self.category_names = category_names
            self.max_length = max_length
            self.task_name = "object_detection"
    
        def __len__(self):
            return len(self.dataset)
    
        def __getitem__(self, idx):
            item = self.dataset[idx]
        
            # 이미지 처리
            image = item['image'].convert('RGB')
            pixel_values = self.image_processor(image, return_tensors="pt").pixel_values.squeeze(0)
        
            # JSON 응답 생성
            instruction = random.choice(self.OD_PROMPTS)
            response = create_od_json(item['objects'], item['width'], item['height'], self.category_names)
        
            full_text = f"{instruction} {response}{self.tokenizer.eos_token}"
        
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
        
            # 지시 마스킹
            instruction_tokens = self.tokenizer.encode(instruction, add_special_tokens=False)
            instruction_len = len(instruction_tokens)
        
            labels = input_ids.clone()
            labels[:instruction_len] = -100
            labels[attention_mask == 0] = -100
        
            return {
                'pixel_values': pixel_values,
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'labels': labels,
                'task': self.task_name,
            }


    class VQATaskDataset(Dataset):
        """VQA 작업용 데이터셋입니다.
        """
    
        def __init__(self, samples, image_processor, tokenizer, max_length=UNIFIED_MAX_LENGTH):
            self.samples = samples
            self.image_processor = image_processor
            self.tokenizer = tokenizer
            self.max_length = max_length
            self.task_name = "vqa"
    
        def __len__(self):
            return len(self.samples)
    
        def __getitem__(self, idx):
            sample = self.samples[idx]
        
            # 이미지 처리
            image = sample['image'].convert('RGB')
            pixel_values = self.image_processor(image, return_tensors="pt").pixel_values.squeeze(0)
        
            # 질문·답변 추출
            question = sample['question']
            answer = get_most_common_answer(sample['answers'])
        
            # 형식: "Question: {q} Answer: {a}"
            prompt = f"Question: {question} Answer:"
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
                'pixel_values': pixel_values,
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'labels': labels,
                'task': self.task_name,
            }

    return CaptionTaskDataset, ODTaskDataset, VQATaskDataset, execution_step_6


@app.cell
def _(
    CaptionTaskDataset,
    ConcatDataset,
    DataLoader,
    ODTaskDataset,
    VQATaskDataset,
    caption_dataset,
    execution_step_6,
    image_processor,
    od_category_names,
    od_dataset,
    tokenizer,
    vqa_samples,
):
    # 원본 코드 셀 순서와 모델·데이터 변경의 실행 순서를 보존합니다.
    execution_step_7 = execution_step_6 + 1
    # 작업별 개별 데이터셋 생성
    caption_task = CaptionTaskDataset(caption_dataset, image_processor, tokenizer)
    od_task = ODTaskDataset(od_dataset, image_processor, tokenizer, od_category_names)
    vqa_task = VQATaskDataset(vqa_samples, image_processor, tokenizer)

    print(f'캡션 작업 샘플: {len(caption_task)}')
    print(f'객체 탐지 작업 샘플: {len(od_task)}')
    print(f'VQA 작업 샘플: {len(vqa_task)}')

    # 멀티태스크 데이터셋으로 결합
    multitask_dataset = ConcatDataset([caption_task, od_task, vqa_task])

    multitask_loader = DataLoader(
        multitask_dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0,
    )

    print(f'\n전체 멀티태스크 학습 샘플: {len(multitask_dataset)}')
    print(f'배치 수: {len(multitask_loader)}')
    return caption_task, execution_step_7, multitask_loader, od_task, vqa_task


@app.cell
def _(caption_task, execution_step_7, od_task, tokenizer, vqa_task):
    # 원본 코드 셀 순서와 모델·데이터 변경의 실행 순서를 보존합니다.
    execution_step_8 = execution_step_7 + 1
    # 작업별 배치 확인
    print('작업별 샘플:\n')

    # 캡션 샘플
    caption_sample = caption_task[0]
    print('캡션 작업:')
    print(tokenizer.decode(caption_sample['input_ids'], skip_special_tokens=True)[:100])

    # 객체 탐지 샘플
    od_sample = od_task[0]
    print('\n객체 탐지 작업:')
    print(tokenizer.decode(od_sample['input_ids'], skip_special_tokens=True)[:150])

    # VQA 샘플
    vqa_sample = vqa_task[0]
    print('\nVQA 작업:')
    print(tokenizer.decode(vqa_sample['input_ids'], skip_special_tokens=True)[:100])
    return (execution_step_8,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3단계: 멀티태스크 학습

    세 작업을 동시에 학습합니다. 모델은 다음을 배웁니다.
    - 프롬프트로부터 작업 유형 파악하기
    - 작업에 맞는 출력 형식 생성하기
    - 여러 작업에서 시각적 이해를 공유하기
    """)
    return


@app.cell
def _(device, execution_step_8, torch, tqdm):
    # 원본 코드 셀 순서와 모델·데이터 변경의 실행 순서를 보존합니다.
    execution_step_9 = execution_step_8 + 1
    def train_multitask_vlm(model, train_loader, num_epochs=8, lr=1e-4):
        """여러 작업으로 VLM을 학습합니다.
        """
    
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable_params, lr=lr)
    
        model.train()
        model.vision_encoder.eval()
    
        losses = []
        task_losses = {'caption': [], 'object_detection': [], 'vqa': []}
    
        for epoch in range(num_epochs):
            epoch_loss = 0
            epoch_task_losses = {'caption': 0, 'object_detection': 0, 'vqa': 0}
            epoch_task_counts = {'caption': 0, 'object_detection': 0, 'vqa': 0}
        
            progress_bar = tqdm(train_loader, desc=f'에포크 {epoch + 1}/{num_epochs}')
        
            for batch in progress_bar:
                pixel_values = batch['pixel_values'].to(device)
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                tasks = batch['task']
            
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
            
                # 작업별 손실 추적
                for task in tasks:
                    if task in epoch_task_losses:
                        epoch_task_losses[task] += loss.item() / len(tasks)
                        epoch_task_counts[task] += 1
            
                progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})
        
            avg_loss = epoch_loss / len(train_loader)
            losses.append(avg_loss)
        
            # 작업별 평균 손실
            for task in task_losses:
                if epoch_task_counts[task] > 0:
                    task_losses[task].append(epoch_task_losses[task] / epoch_task_counts[task])
                else:
                    task_losses[task].append(0)
        
            print(f'에포크 {epoch + 1} - 손실: {avg_loss:.4f} | 캡션: {task_losses['caption'][-1]:.4f} | OD: {task_losses['object_detection'][-1]:.4f} | VQA: {task_losses['vqa'][-1]:.4f}')
    
        return losses, task_losses

    return execution_step_9, train_multitask_vlm


@app.cell
def _(execution_step_9, multitask_loader, train_multitask_vlm, vlm):
    # 원본 코드 셀 순서와 모델·데이터 변경의 실행 순서를 보존합니다.
    execution_step_10 = execution_step_9 + 1
    # 멀티태스크 모델 학습
    losses, task_losses = train_multitask_vlm(vlm, multitask_loader, num_epochs=8, lr=1e-4)
    return execution_step_10, losses, task_losses


@app.cell
def _(execution_step_10, losses, plt, task_losses):
    # 원본 코드 셀 순서와 모델·데이터 변경의 실행 순서를 보존합니다.
    execution_step_11 = execution_step_10 + 1
    # 학습 곡선 표시
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))

    # 전체 손실
    ax1.plot(range(1, len(losses)+1), losses, marker='o', label='전체', linewidth=2)
    ax1.set_xlabel('에포크')
    ax1.set_ylabel('손실')
    ax1.set_title('멀티태스크 학습 손실')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # 작업별 손실
    ax2.plot(range(1, len(losses)+1), task_losses['caption'], marker='o', label='캡션', linewidth=2)
    ax2.plot(range(1, len(losses)+1), task_losses['object_detection'], marker='s', label='객체 탐지', linewidth=2)
    ax2.plot(range(1, len(losses)+1), task_losses['vqa'], marker='^', label='VQA', linewidth=2)
    ax2.set_xlabel('에포크')
    ax2.set_ylabel('손실')
    ax2.set_title('작업별 학습 손실')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.show()
    return (execution_step_11,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4단계: 세 작업 모두 평가하기

    각 작업으로 멀티태스크 모델을 테스트해 세 작업을 모두 처리할 수 있는지 확인합니다.
    """)
    return


@app.cell
def _(Image, execution_step_11):
    # 원본 코드 셀 순서와 모델·데이터 변경의 실행 순서를 보존합니다.
    execution_step_12 = execution_step_11 + 1
    # 생성용 보조 함수

    def generate_caption(model, image, image_processor, device):
        """캡션을 생성합니다.
        """
        model.eval()
    
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        image = image.convert('RGB')
    
        pixel_values = image_processor(image, return_tensors="pt").pixel_values.to(device)
    
        response = model.generate(
            pixel_values,
            prompt="Describe this image.",
            max_new_tokens=40,
            temperature=0.7,
            do_sample=True,
        )
    
        if "Describe this image." in response:
            return response.split("Describe this image.")[-1].strip()
        return response


    def generate_od_response(model, image, image_processor, device):
        """객체 탐지 JSON을 생성합니다.
        """
        model.eval()
    
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        image = image.convert('RGB')
    
        pixel_values = image_processor(image, return_tensors="pt").pixel_values.to(device)
    
        response = model.generate(
            pixel_values,
            prompt="What objects are in this image? Output as JSON.",
            max_new_tokens=150,
            temperature=0.3,
            do_sample=True,
        )
    
        return response


    def generate_vqa_answer(model, image, question, image_processor, device):
        """VQA 답변을 생성합니다.
        """
        model.eval()
    
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        image = image.convert('RGB')
    
        pixel_values = image_processor(image, return_tensors="pt").pixel_values.to(device)
    
        prompt = f"Question: {question} Answer:"
    
        response = model.generate(
            pixel_values,
            prompt=prompt,
            max_new_tokens=10,
            temperature=1.0,
            do_sample=False,
        )
    
        if "Answer:" in response:
            answer = response.split("Answer:")[-1].strip()
        else:
            answer = response
    
        # 정리
        answer = answer.split('.')[0].strip()
        words = answer.split()
        if len(words) > 5:
            answer = ' '.join(words[:5])
    
        return answer

    return (
        execution_step_12,
        generate_caption,
        generate_od_response,
        generate_vqa_answer,
    )


@app.cell
def _(
    caption_dataset,
    device,
    execution_step_12,
    generate_caption,
    image_processor,
    vlm,
):
    # 원본 코드 셀 순서와 모델·데이터 변경의 실행 순서를 보존합니다.
    execution_step_13 = execution_step_12 + 1
    # 작업 1 테스트: 캡션
    print('=' * 70)
    print('작업 1: 이미지 캡션')
    print('=' * 70)
    caption_test_indices = [1000, 1005, 1010]
    for _idx in caption_test_indices:  # 학습 세트 외부의 샘플
        if _idx >= len(caption_dataset):
            _idx = _idx % len(caption_dataset)
        _sample = caption_dataset[_idx]
        _image = _sample['image']
        gt_caption = _sample['caption_0']
        pred_caption = generate_caption(vlm, _image, image_processor, device)
        print(f'\n이미지 {_idx}:')
        print(f'  예측: {pred_caption}')
        print(f'  GT:        {gt_caption}')
    return (execution_step_13,)


@app.cell
def _(
    create_od_json,
    device,
    execution_step_13,
    generate_od_response,
    image_processor,
    load_dataset,
    od_category_names,
    vlm,
):
    # 원본 코드 셀 순서와 모델·데이터 변경의 실행 순서를 보존합니다.
    execution_step_14 = execution_step_13 + 1
    # 작업 2 테스트: 객체 탐지
    print('\n' + '=' * 70)
    print('작업 2: 객체 탐지')
    print('=' * 70)
    od_test = load_dataset('Francesco/animals-ij5d2', split='test')
    # 테스트 세트 불러오기
    for _idx in [0, 5]:
        if _idx >= len(od_test):
            continue
        _sample = od_test[_idx]
        _image = _sample['image']
        gt_json = create_od_json(_sample['objects'], _sample['width'], _sample['height'], od_category_names)
        pred_response = generate_od_response(vlm, _image, image_processor, device)
        print(f'\n테스트 이미지 {_idx}:')
        print(f'  예측: {pred_response[:150]}...')
        print(f'  GT:        {gt_json[:150]}...')
    return execution_step_14, od_test


@app.cell
def _(
    device,
    execution_step_14,
    generate_vqa_answer,
    get_most_common_answer,
    image_processor,
    load_dataset,
    vlm,
):
    # 원본 코드 셀 순서와 모델·데이터 변경의 실행 순서를 보존합니다.
    execution_step_15 = execution_step_14 + 1
    # 작업 3 테스트: VQA
    print('\n' + '=' * 70)
    print('작업 3: 시각적 질의응답')
    print('=' * 70)
    vqa_test_stream = load_dataset('lmms-lab/VQAv2', split='validation', streaming=True)
    # 테스트용 VQA 샘플 추가 로딩
    vqa_test_samples = []
    for _i, _sample in enumerate(vqa_test_stream):
        if _i < 1000:
            continue  # 학습 샘플 건너뛰기
        if _i >= 1010:
            break
        vqa_test_samples.append(_sample)
    for _i, _sample in enumerate(vqa_test_samples[:5]):
        _image = _sample['image']
        question = _sample['question']
        gt_answer = get_most_common_answer(_sample['answers'])
        pred_answer = generate_vqa_answer(vlm, _image, question, image_processor, device)
        print(f'\nQ: {question}')
        print(f'  예측: {pred_answer}')
        print(f'  GT:        {gt_answer}')
    return (execution_step_15,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5단계: 멀티태스크 시연

    같은 이미지에 대해 세 작업을 모두 수행합니다.
    """)
    return


@app.cell
def _(
    device,
    execution_step_15,
    generate_caption,
    generate_od_response,
    generate_vqa_answer,
    image_processor,
    od_test,
    plt,
    vlm,
):
    # 원본 코드 셀 순서와 모델·데이터 변경의 실행 순서를 보존합니다.
    execution_step_16 = execution_step_15 + 1
    # 하나의 이미지로 세 작업 모두 시연
    test_image = od_test[0]['image']

    # 작업 1: 캡션
    caption = generate_caption(vlm, test_image, image_processor, device)
    print('캡션:')
    print(f"  {caption}")

    # 작업 2: 객체 탐지
    od_response = generate_od_response(vlm, test_image, image_processor, device)
    print('\n객체 탐지:')
    print(f"  {od_response[:200]}...")

    # 작업 3: VQA
    questions = [
        "What animal is this?",
        "What color is it?",
        "Is it indoors or outdoors?"
    ]
    print('\nVQA:')
    for q in questions:
        answer = generate_vqa_answer(vlm, test_image, q, image_processor, device)
        print(f"  Q: {q}")
        print(f"  A: {answer}")

    # 이미지 표시
    plt.figure(figsize=(6, 6))
    plt.imshow(test_image)
    plt.title(f'멀티태스크 테스트\n캡션: {caption[:50]}...', fontsize=10)
    plt.axis('off')
    plt.show()
    return (execution_step_16,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6단계: 멀티태스크 모델 저장하기
    """)
    return


@app.cell
def _(
    execution_step_16,
    image_processor,
    language_dim,
    lm_model_name,
    od_category_names,
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
    # 멀티태스크 모델 저장
    save_dir = str(output_dir / "mini-vlm-multitask")
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
        'multitask_config': {
            'tasks': ['caption', 'object_detection', 'vqa'],
            'od_category_names': od_category_names,
        }
    }, f"{save_dir}/mini_vlm_multitask.pt")

    tokenizer.save_pretrained(f"{save_dir}/tokenizer")
    image_processor.save_pretrained(f"{save_dir}/image_processor")

    print(f'멀티태스크 모델 저장 위치: {save_dir}/')
    print(f'저장 내용: {os.listdir(save_dir)}')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 정리

    다음 작업을 처리하는 **통합 멀티태스크 VLM**을 만들었습니다.

    ### 하나의 모델로 세 가지 작업 수행

    1. **이미지 캡션**: 자연어 설명
    2. **객체 탐지**: 경계 상자를 포함한 구조화된 JSON
    3. **시각적 질의응답**: 짧고 사실적인 답변

    ### 수행한 작업

    1. 캡션을 학습한 모델에서 시작
    2. 멀티태스크 데이터셋 구성(총 2,400개 샘플)
    3. 작업별 프롬프트로 학습
    4. 세 작업 모두 평가

    ### 핵심 내용

    - **작업 프롬프트**가 서로 다른 출력 형식을 유도합니다.
    - **공유 비전 인코더**는 모든 작업에 유용한 표현을 제공합니다.
    - **멀티태스크 학습**은 일반화 성능을 높일 수 있습니다.
    - **하나의 모델**이 개별 모델 세 개보다 실용적입니다.

    ### 모델의 발전 과정

    ```
    기반 모델(ViT + SmolLM)
        │
        └── 캡션 학습
                │
                ├── 객체 탐지 미세 조정(전용)
                ├── VQA 미세 조정(전용)
                └── 멀티태스크 미세 조정(통합) ← 이 노트북
    ```

    ### 비교: 전용 모델과 멀티태스크 모델

    | 방식 | 장점 | 단점 |
    |----------|------|------|
    | **전용 모델** | 작업별 성능이 더 좋음 | 모델 3개 배포·유지 필요 |
    | **멀티태스크 모델** | 모델 1개, 학습 내용 공유 | 작업별 정확도가 다소 낮음 |

    ### 한계

    - 소형 모델(매개변수 135M)
    - 작업별 학습 데이터 부족
    - 작업별 최적화 없음
    - 상용 서비스용으로 준비된 모델이 아닌 교육용 모델

    ### 다음 단계

    1. **작업 추가**: 이미지 편집, 시각적 그라운딩, OCR
    2. **태스크 라우팅**: 프롬프트에서 작업 자동 감지
    3. **더 큰 모델**: SmolLM-1.7B나 Qwen2-VL로 확장
    4. **더 나은 데이터셋**: COCO, Visual Genome 등 사용
    5. **작업 균형 조정**: 작업별 샘플링 비율 조정
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## GPU 메모리 관리

    실습을 마친 뒤 GPU 메모리를 해제하려면 아래 안내를 따르세요. 여러 노트북을 순서대로 실행할 때 특히 중요합니다.
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
