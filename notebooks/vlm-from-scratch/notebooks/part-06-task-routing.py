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
    # 멀티태스크 VLM을 위한 지능형 태스크 라우팅

    Nipun Batra · 2025-12-28
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 소개

    [멀티태스크 VLM 노트북](part-04-multi-task.py)에서는 캡션, 객체 탐지, VQA를 처리하는 모델을 만들었습니다. 다만 **작업을 명시하는 프롬프트**가 필요했습니다.

    - 캡션: `"Describe this image."`
    - 객체 탐지: `"What objects? Output JSON."`
    - VQA: `"Question: {q} Answer:"`

    사용자가 형식을 지정하지 않고 자연스럽게 질문하면 어떨까요? 모델은 사용자의 의도를 **자동으로 감지**해야 합니다.

    | 사용자 입력 | 감지한 작업 | 출력 |
    |------------|--------------|--------|
    | `"What is in this image?"`(이미지에 무엇이 있나요?) | 캡션 | `"A dog playing..."` |
    | `"Detect objects"`(객체를 탐지하세요) | 객체 탐지 | `{"objects": [...]}` |
    | `"Is the dog happy?"`(개가 행복해 보이나요?) | VQA | `"yes"` |

    ### 태스크 라우팅의 두 가지 접근

    1. **접두사 기반 라우팅**(기본)
       - `[CAPTION]`, `[OD]`, `[VQA]` 태그 추가
       - 모델이 태그에 맞는 출력 형식을 학습

    2. **지능형 라우팅**(심화)
       - 별도의 분류기가 사용자 질문에서 작업을 예측
       - 적절한 생성 모드로 연결
       - 유연하고 사용하기 편리함

    이 노트북에서는 **두 방식 모두** 구현합니다.
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
    import random
    import os
    import warnings
    warnings.filterwarnings('ignore')


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
    ## 1단계: 기반 멀티태스크 VLM 불러오기

    멀티태스크 학습을 마친 모델에서 시작합니다.
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
    # 앞과 같은 VLM 구조

    class VisionProjector(nn.Module):
        """시각 특징을 언어 모델의 임베딩 공간으로 투영합니다.
        """
    
        def __init__(self, vision_dim: int, language_dim: int):
            super().__init__()
            self.projection = nn.Sequential(
                nn.Linear(vision_dim, language_dim),
                nn.GELU(),
                nn.LayerNorm(language_dim),
                nn.Linear(language_dim, language_dim),
            )
    
        def forward(self, vision_features: torch.Tensor) -> torch.Tensor:
            return self.projection(vision_features)


    class MiniVLM(nn.Module):
        """최소 구성의 시각·언어 모델입니다.
        """
    
        def __init__(
            self,
            vision_encoder: ViTModel,
            language_model: AutoModelForCausalLM,
            projector: VisionProjector,
            tokenizer: AutoTokenizer,
        ):
            super().__init__()
            self.vision_encoder = vision_encoder
            self.language_model = language_model
            self.projector = projector
            self.tokenizer = tokenizer
        
            for param in self.vision_encoder.parameters():
                param.requires_grad = False
    
        def encode_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
            with torch.no_grad():
                vision_outputs = self.vision_encoder(pixel_values=pixel_values)
            image_features = vision_outputs.last_hidden_state
            projected = self.projector(image_features)
            return projected
    
        def forward(
            self,
            pixel_values: torch.Tensor,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            labels: torch.Tensor = None,
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
            pixel_values: torch.Tensor,
            prompt: str,
            max_new_tokens: int = 50,
            temperature: float = 0.7,
            do_sample: bool = True,
        ) -> str:
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
    return (execution_step_3, image_processor, language_dim, lm_model_name, tokenizer, vision_dim, vision_model_name, vlm)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2단계: 접근 1 — 접두사 기반 작업 태그

    프롬프트 앞에 `[CAPTION]`, `[OD]`, `[VQA]` 같은 특수 토큰을 붙입니다. 모델은 태그에 맞는 출력을 생성하도록 학습합니다.

    ### 학습 형식
    ```
    [CAPTION] A dog playing in the park.
    [OD] {"objects": [{"label": "dog", "bbox": [...]}]}
    [VQA] Question: What color? Answer: brown
    ```
    """)
    return


@app.cell
def _(execution_step_3, load_dataset):
    # 원본 코드 셀 순서와 모델·데이터 변경의 실행 순서를 보존합니다.
    execution_step_4 = execution_step_3 + 1
    # 데이터셋 로딩(멀티태스크 노트북과 동일)
    print('데이터셋 로딩 중...')
    caption_dataset = load_dataset('jxie/flickr8k', split='train').shuffle(seed=42).select(range(500))
    print(f'캡션 데이터셋: {len(caption_dataset)}개 샘플')
    od_train = load_dataset('Francesco/animals-ij5d2', split='train')
    od_val = load_dataset('Francesco/animals-ij5d2', split='validation')
    from datasets import concatenate_datasets
    od_dataset = concatenate_datasets([od_train, od_val]).select(range(200))
    od_category_names = od_dataset.features['objects']['category'].feature.names
    print(f'객체 탐지 데이터셋: {len(od_dataset)}개 샘플')
    vqa_dataset_stream = load_dataset('lmms-lab/VQAv2', split='validation', streaming=True)
    vqa_samples = []
    for _i, sample in enumerate(vqa_dataset_stream):
        if _i >= 500:
            break
        vqa_samples.append(sample)
    print(f'VQA 데이터셋: {len(vqa_samples)}개 샘플')
    print(f'\n합계: {len(caption_dataset) + len(od_dataset) + len(vqa_samples)}개 샘플')
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
    # 보조 함수

    def get_most_common_answer(answers):
        answer_counts = {}
        for ans in answers:
            a = ans['answer']
            answer_counts[a] = answer_counts.get(a, 0) + 1
        return max(answer_counts, key=answer_counts.get)

    def create_od_json(objects, width, height, category_names):
        result = {"objects": []}
        for bbox, cat_id in zip(objects['bbox'], objects['category']):
            x, y, w, h = bbox
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
    # 작업 태그를 포함한 데이터셋

    UNIFIED_MAX_LENGTH = 256

    class TaggedCaptionDataset(Dataset):
        """[CAPTION] 태그를 사용하는 캡션 데이터셋입니다.
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
            image = item['image'].convert('RGB')
            pixel_values = self.image_processor(image, return_tensors="pt").pixel_values.squeeze(0)
        
            caption = item[f'caption_{random.randint(0, 4)}']
        
            # 작업 태그 추가
            prompt = "[CAPTION]"
            full_text = f"{prompt} {caption}{self.tokenizer.eos_token}"
        
            encoding = self.tokenizer(
                full_text,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
        
            input_ids = encoding['input_ids'].squeeze(0)
            attention_mask = encoding['attention_mask'].squeeze(0)
        
            # 태그 마스킹(응답만 학습)
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


    class TaggedODDataset(Dataset):
        """[OD] 태그를 사용하는 객체 탐지 데이터셋입니다.
        """
    
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
            image = item['image'].convert('RGB')
            pixel_values = self.image_processor(image, return_tensors="pt").pixel_values.squeeze(0)
        
            response = create_od_json(item['objects'], item['width'], item['height'], self.category_names)
        
            # 작업 태그 추가
            prompt = "[OD]"
            full_text = f"{prompt} {response}{self.tokenizer.eos_token}"
        
            encoding = self.tokenizer(
                full_text,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
        
            input_ids = encoding['input_ids'].squeeze(0)
            attention_mask = encoding['attention_mask'].squeeze(0)
        
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


    class TaggedVQADataset(Dataset):
        """[VQA] 태그를 사용하는 VQA 데이터셋입니다.
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
            image = sample['image'].convert('RGB')
            pixel_values = self.image_processor(image, return_tensors="pt").pixel_values.squeeze(0)
        
            question = sample['question']
            answer = get_most_common_answer(sample['answers'])
        
            # 작업 태그 추가
            prompt = f"[VQA] {question}"
            full_text = f"{prompt} {answer}{self.tokenizer.eos_token}"
        
            encoding = self.tokenizer(
                full_text,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
        
            input_ids = encoding['input_ids'].squeeze(0)
            attention_mask = encoding['attention_mask'].squeeze(0)
        
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

    return (
        TaggedCaptionDataset,
        TaggedODDataset,
        TaggedVQADataset,
        execution_step_6,
    )


@app.cell
def _(
    ConcatDataset,
    DataLoader,
    TaggedCaptionDataset,
    TaggedODDataset,
    TaggedVQADataset,
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
    # 태그 데이터셋 생성
    tagged_caption = TaggedCaptionDataset(caption_dataset, image_processor, tokenizer)
    tagged_od = TaggedODDataset(od_dataset, image_processor, tokenizer, od_category_names)
    tagged_vqa = TaggedVQADataset(vqa_samples, image_processor, tokenizer)

    print(f'태그가 있는 캡션: {len(tagged_caption)}')
    print(f'태그가 있는 객체 탐지: {len(tagged_od)}')
    print(f'태그가 있는 VQA: {len(tagged_vqa)}')

    # 결합
    tagged_dataset = ConcatDataset([tagged_caption, tagged_od, tagged_vqa])
    tagged_loader = DataLoader(tagged_dataset, batch_size=4, shuffle=True, num_workers=0)

    print(f'\n태그 샘플 합계: {len(tagged_dataset)}')
    return (
        execution_step_7,
        tagged_caption,
        tagged_loader,
        tagged_od,
        tagged_vqa,
    )


@app.cell
def _(execution_step_7, tagged_caption, tagged_od, tagged_vqa, tokenizer):
    # 원본 코드 셀 순서와 모델·데이터 변경의 실행 순서를 보존합니다.
    execution_step_8 = execution_step_7 + 1
    # 태그 형식 확인
    print('태그 샘플:')
    print('\n캡션:')
    print(tokenizer.decode(tagged_caption[0]['input_ids'], skip_special_tokens=True)[:80])

    print("\nOD:")
    print(tokenizer.decode(tagged_od[0]['input_ids'], skip_special_tokens=True)[:120])

    print('\nVQA:')
    print(tokenizer.decode(tagged_vqa[0]['input_ids'], skip_special_tokens=True)[:80])
    return (execution_step_8,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3단계: 작업 태그로 학습하기
    """)
    return


@app.cell
def _(device, execution_step_8, torch, tqdm):
    # 원본 코드 셀 순서와 모델·데이터 변경의 실행 순서를 보존합니다.
    execution_step_9 = execution_step_8 + 1
    def train_vlm(model, train_loader, num_epochs=5, lr=1e-4):
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable_params, lr=lr)
    
        model.train()
        model.vision_encoder.eval()
    
        losses = []
    
        for epoch in range(num_epochs):
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
    
        return losses

    return execution_step_9, train_vlm


@app.cell
def _(execution_step_9, tagged_loader, train_vlm, vlm):
    # 원본 코드 셀 순서와 모델·데이터 변경의 실행 순서를 보존합니다.
    execution_step_10 = execution_step_9 + 1
    # 작업 태그로 학습
    losses = train_vlm(vlm, tagged_loader, num_epochs=5, lr=1e-4)
    return execution_step_10, losses


@app.cell
def _(execution_step_10, losses, plt):
    # 원본 코드 셀 순서와 모델·데이터 변경의 실행 순서를 보존합니다.
    execution_step_11 = execution_step_10 + 1
    # 손실 그래프
    plt.figure(figsize=(8, 4))
    plt.plot(range(1, len(losses)+1), losses, marker='o', linewidth=2)
    plt.xlabel('에포크')
    plt.ylabel('손실')
    plt.title('태그 기반 라우팅 학습 손실')
    plt.grid(True, alpha=0.3)
    plt.show()
    return (execution_step_11,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4단계: 태그 기반 라우팅 테스트
    """)
    return


@app.cell
def _(caption_dataset, device, execution_step_11, image_processor, plt, vlm):
    # 원본 코드 셀 순서와 모델·데이터 변경의 실행 순서를 보존합니다.
    execution_step_12 = execution_step_11 + 1
    # 태그로 테스트
    _test_image = caption_dataset[450]['image']
    print('태그 기반 라우팅 테스트:\n')
    # 각 작업 태그 테스트
    pixel_values = image_processor(_test_image, return_tensors='pt').pixel_values.to(device)
    caption = vlm.generate(pixel_values, '[CAPTION]', max_new_tokens=40, temperature=0.7)
    # 캡션
    print(f'[CAPTION] 태그: {caption}')
    od_response = vlm.generate(pixel_values, '[OD]', max_new_tokens=150, temperature=0.3)
    print(f'\n[OD] 태그: {od_response[:120]}...')
    vqa_response = vlm.generate(pixel_values, '[VQA] What is the main color?', max_new_tokens=10, do_sample=False)
    # OD
    print(f'\n[VQA] 태그: {vqa_response}')
    plt.figure(figsize=(6, 6))
    plt.imshow(_test_image)
    # 시각적 질의응답(VQA)
    plt.title('테스트 이미지', fontsize=12)
    plt.axis('off')
    # 이미지 표시
    plt.show()
    return (execution_step_12,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5단계: 접근 2 — 자연어 질문 기반 지능형 라우팅

    자연어 질문에서 작업을 예측하는 가벼운 분류기를 만듭니다.
    - `"What's in this image?"`(이미지에 무엇이 있나요?) → 캡션
    - `"Find all objects"`(모든 객체를 찾으세요) → 객체 탐지
    - `"Is there a dog?"`(개가 있나요?) → VQA
    """)
    return


@app.cell
def _(execution_step_12, vqa_samples):
    # 원본 코드 셀 순서와 모델·데이터 변경의 실행 순서를 보존합니다.
    execution_step_13 = execution_step_12 + 1
    # 합성 라우팅 데이터용 작업별 질문 패턴
    CAPTION_QUERIES = ['Describe this image', "What's in this picture?", 'Tell me about this image', 'What do you see?', 'Caption this', 'What is shown here?']
    OD_QUERIES = ['Detect objects', 'Find all objects', 'What objects are present?', 'List objects with locations', 'Object detection', 'Show me bounding boxes']
    VQA_TEMPLATES = ['What {}', 'Is {}', 'How many {}', 'Where {}', 'Can you see {}', 'Does {}']
    routing_data = []
    for _query in CAPTION_QUERIES:
        routing_data.append((_query, 0))
    for _query in OD_QUERIES:
        routing_data.append((_query, 1))
    for _i in range(min(20, len(vqa_samples))):
        routing_data.append((vqa_samples[_i]['question'], 2))
    print(f'라우팅 학습 샘플: {len(routing_data)}')
    print(f'  캡션: {sum((1 for _, task in routing_data if task == 0))}')
    print(f'  OD: {sum((1 for _, task in routing_data if task == 1))}')
    # 합성 라우팅 학습 데이터 생성
    # 캡션 질문
    # 객체 탐지 질문
    # VQA 질문(실제 질문 일부 사용)
    print(f'  VQA: {sum((1 for _, task in routing_data if task == 2))}')  # 0 = 캡션 / 1 = 객체 탐지 / 2 = VQA
    return execution_step_13, routing_data


@app.cell
def _(device, execution_step_13, nn, tokenizer, torch):
    # 원본 코드 셀 순서와 모델·데이터 변경의 실행 순서를 보존합니다.
    execution_step_14 = execution_step_13 + 1
    # 간단한 텍스트 기반 작업 분류기

    class TaskClassifier(nn.Module):
        """텍스트 질문에서 작업을 분류합니다.
        """
    
        def __init__(self, tokenizer, hidden_dim=128, num_tasks=3):
            super().__init__()
            self.tokenizer = tokenizer
            self.embedding = nn.Embedding(tokenizer.vocab_size, hidden_dim)
            self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True, bidirectional=True)
            self.classifier = nn.Linear(hidden_dim * 2, num_tasks)
    
        def forward(self, input_ids):
            # input_ids: (batch, seq_len)
            embeds = self.embedding(input_ids)  # (batch, seq_len, hidden)
            lstm_out, (h_n, c_n) = self.lstm(embeds)  # h_n: (2, batch, hidden)
        
            # 정방향·역방향 은닉 상태 연결
            hidden = torch.cat([h_n[0], h_n[1]], dim=1)  # (batch, hidden*2)
        
            logits = self.classifier(hidden)  # (batch, num_tasks)
            return logits


    task_classifier = TaskClassifier(tokenizer).to(device)
    print(f'작업 분류기 매개변수: {sum((p.numel() for p in task_classifier.parameters())):,}')
    return execution_step_14, task_classifier


@app.cell
def _(
    device,
    execution_step_14,
    nn,
    random,
    routing_data,
    task_classifier,
    tokenizer,
    torch,
):
    # 원본 코드 셀 순서와 모델·데이터 변경의 실행 순서를 보존합니다.
    execution_step_15 = execution_step_14 + 1
    # 작업 분류기 학습

    def train_task_classifier(model, data, num_epochs=10, lr=1e-3):
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
    
        model.train()
    
        for epoch in range(num_epochs):
            total_loss = 0
            correct = 0
        
            # 데이터 섞기
            random.shuffle(data)
        
            for query, task_label in data:
                # 질문 토큰화
                tokens = tokenizer(query, return_tensors='pt', padding='max_length', 
                                 max_length=32, truncation=True)
                input_ids = tokens['input_ids'].to(device)
            
                # 순전파
                logits = model(input_ids)
                target = torch.tensor([task_label]).to(device)
            
                loss = criterion(logits, target)
            
                # 역전파
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
                total_loss += loss.item()
                pred = logits.argmax(dim=1)
                correct += (pred == target).sum().item()
        
            avg_loss = total_loss / len(data)
            accuracy = correct / len(data) * 100
            print(f'에포크 {epoch + 1}/{num_epochs} - 손실: {avg_loss:.4f}, 정확도: {accuracy:.1f}%')

    train_task_classifier(task_classifier, routing_data, num_epochs=20)
    return (execution_step_15,)


@app.cell
def _(device, execution_step_15, task_classifier, tokenizer, torch):
    # 원본 코드 셀 순서와 모델·데이터 변경의 실행 순서를 보존합니다.
    execution_step_16 = execution_step_15 + 1
    # 작업 분류기 테스트

    def predict_task(query, model, tokenizer):
        """질문에서 작업을 예측합니다.
        """
        model.eval()
        tokens = tokenizer(query, return_tensors='pt', padding='max_length', max_length=32, truncation=True)
        input_ids = tokens['input_ids'].to(device)
        with torch.no_grad():
            logits = model(input_ids)
            pred = logits.argmax(dim=1).item()
        task_names = ['CAPTION', 'OD', 'VQA']
        return task_names[pred]
    test_queries = ["What's in this image?", 'Detect all objects', 'Is there a dog?', 'Describe the scene', 'Find objects with bounding boxes', 'What color is the sky?']
    print('작업 분류 결과:\n')
    for _query in test_queries:
        _task = predict_task(_query, task_classifier, tokenizer)
        print(f"질문: '{_query}'")
    # 테스트 질문
        print(f'  → 예측한 작업: {_task}\n')
    return execution_step_16, predict_task


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6단계: 자동 라우팅을 포함한 통합 추론
    """)
    return


@app.cell
def _(Image, execution_step_16, predict_task):
    # 원본 코드 셀 순서와 모델·데이터 변경의 실행 순서를 보존합니다.
    execution_step_17 = execution_step_16 + 1
    def auto_generate(image, query, vlm_model, classifier_model, image_processor, tokenizer, device):
        """질문에 따라 자동 라우팅하고 적절한 응답을 생성합니다.
        """
    
        # 1단계: 작업 분류
        task = predict_task(query, classifier_model, tokenizer)
    
        # 2단계: 이미지 준비
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        pixel_values = image_processor(image, return_tensors="pt").pixel_values.to(device)
    
        # 3단계: 감지된 작업에 따라 생성
        if task == 'CAPTION':
            prompt = "[CAPTION]"
            response = vlm_model.generate(pixel_values, prompt, max_new_tokens=40, temperature=0.7)
        elif task == 'OD':
            prompt = "[OD]"
            response = vlm_model.generate(pixel_values, prompt, max_new_tokens=150, temperature=0.3)
        else:  # 시각적 질의응답(VQA)
            prompt = f"[VQA] {query}"
            response = vlm_model.generate(pixel_values, prompt, max_new_tokens=10, do_sample=False)
    
        # 답변 추출
        if task in response:
            answer = response.split(task)[-1].strip()
        else:
            answer = response
    
        return task, answer

    return auto_generate, execution_step_17


@app.cell
def _(
    auto_generate,
    caption_dataset,
    device,
    execution_step_17,
    image_processor,
    plt,
    task_classifier,
    tokenizer,
    vlm,
):
    # 원본 코드 셀 순서와 모델·데이터 변경의 실행 순서를 보존합니다.
    execution_step_18 = execution_step_17 + 1
    # 자동 라우팅 시연
    _test_image = caption_dataset[400]['image']
    queries = ["What's in this picture?", 'Detect all objects', 'Is there a person?']
    print('AUTO-ROUTING DEMO\n' + '=' * 70)
    for _query in queries:
        _task, answer = auto_generate(_test_image, _query, vlm, task_classifier, image_processor, tokenizer, device)
        print(f"\n질문: '{_query}'")
        print(f'감지한 작업: {_task}')
        print(f'Response: {answer[:100]}...' if len(answer) > 100 else f'Response: {answer}')
        print('-' * 70)
    plt.figure(figsize=(6, 6))
    plt.imshow(_test_image)
    plt.title('자동 라우팅용 테스트 이미지', fontsize=12)
    plt.axis('off')
    # 이미지 표시
    plt.show()
    return (execution_step_18,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7단계: 모델 저장하기
    """)
    return


@app.cell
def _(
    execution_step_18,
    image_processor,
    language_dim,
    lm_model_name,
    os,
    output_dir,
    task_classifier,
    tokenizer,
    torch,
    vision_dim,
    vision_model_name,
    vlm,
):
    # 원본 코드 셀 순서와 모델·데이터 변경의 실행 순서를 보존합니다.
    execution_step_19 = execution_step_18 + 1
    # 태그 기반 VLM과 작업 분류기 저장
    save_dir = str(output_dir / "mini-vlm-routing")
    os.makedirs(save_dir, exist_ok=True)

    # 태그를 사용하는 VLM 저장
    torch.save({
        'projector_state_dict': vlm.projector.state_dict(),
        'language_model_state_dict': vlm.language_model.state_dict(),
        'config': {
            'vision_model_name': vision_model_name,
            'lm_model_name': lm_model_name,
            'vision_dim': vision_dim,
            'language_dim': language_dim,
        },
    }, f"{save_dir}/vlm_with_tags.pt")

    # 작업 분류기 저장
    torch.save({
        'state_dict': task_classifier.state_dict(),
        'config': {
            'hidden_dim': 128,
            'num_tasks': 3,
        },
    }, f"{save_dir}/task_classifier.pt")

    tokenizer.save_pretrained(f"{save_dir}/tokenizer")
    image_processor.save_pretrained(f"{save_dir}/image_processor")

    print(f'모델 저장 위치: {save_dir}/')
    print(f'저장 내용: {os.listdir(save_dir)}')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 정리

    태스크 라우팅의 **두 가지 접근**을 구현했습니다.

    ### 1. 접두사 기반 태그(기본)

    - `[CAPTION]`, `[OD]`, `[VQA]` 태그 추가
    - 모델이 태그별로 다른 출력 형식을 학습
    - **장점**: 단순하고 VLM 학습에 통합 가능
    - **단점**: 사용자가 태그를 알아야 함

    ### 2. 자연어 질문 기반 지능형 라우팅(심화)

    - 별도의 분류기가 질문에서 작업 예측
    - 적절한 생성 모드로 연결
    - **장점**: 자연어 인터페이스로 사용하기 편리함
    - **단점**: 추가 분류기와 2단계 추론 필요

    ### 라우팅 파이프라인

    ```
    사용자 질문 → 작업 분류기 → 작업 예측 → 태그를 붙여 VLM에 전달 → 출력 생성
    "What's       (LSTM 기반)    (캡션)        [CAPTION] 프롬프트
     in image?"
    ```

    ### 핵심 내용

    - **작업 태그**는 가볍고 효과적입니다.
    - **질문 분류**는 자연스러운 상호작용을 가능하게 합니다.
    - **혼합 방식**은 두 접근을 결합해 유연성을 높입니다.
    - **합성 라우팅 데이터**도 단순한 분류기에는 잘 작동합니다.

    ### 비교

    | 방식 | 사용자 경험 | 복잡도 | 정확도 |
    |----------|----------------|------------|----------|
    | **명시적 태그** | 태그 사용 필요 | 낮음 | 완벽 |
    | **지능형 라우팅** | 자연어 사용 | 중간 | 약 90~95% |
    | **혼합** | 두 방식 모두 지원 | 중간 | 가장 좋음 |

    ### 한계

    - 단순한 LSTM 분류기 사용(BERT로 개선 가능)
    - 작은 합성 라우팅 데이터셋
    - 모호한 질문을 처리하지 않음
    - "caption and detect" 같은 여러 작업을 함께 요구하는 질문 미지원

    ### 다음 단계

    1. **더 나은 분류기**: 사전 학습된 BERT/DistilBERT 사용
    2. **라우팅 데이터 확대**: 실제 사용자 질문 수집
    3. **신뢰도 점수**: 추가 확인 질문이 필요한 시점 판단
    4. **멀티태스크 라우팅**: "캡션을 만들고 객체를 찾아줘" 처리
    5. **능동 학습**: 오류를 바탕으로 분류기 개선
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 참고 자료

    - [멀티태스크 VLM](part-04-multi-task.py)
    - [목적 지향 대화 시스템](https://arxiv.org/abs/2003.07490)
    - [전문가 혼합 모델](https://arxiv.org/abs/1701.06538): 유사한 라우팅 개념
    - [텍스트 분류를 위한 BERT](https://arxiv.org/abs/1810.04805)
    """)
    return


if __name__ == "__main__":
    app.run()
