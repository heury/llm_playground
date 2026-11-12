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
    # 객체 탐지를 위한 VLM 지시 미세 조정

    Nipun Batra · 2025-12-25
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 소개

    [이전 노트북](part-01-minimal-vlm.py)에서는 최소 구성의 시각·언어 모델(VLM)을 직접 만들고 이미지 캡션 데이터로 학습했습니다. 모델은 자연어로 이미지를 설명하는 방법을 배웠습니다.

    이제 한 단계 더 나아가 객체 탐지를 위한 **지시 미세 조정**을 진행합니다. 이미지를 설명하는 데서 나아가 다음을 학습시킵니다.

    1. *"What objects are in this image?"*(이 이미지에 어떤 객체가 있나요?) 같은 질문 이해하기
    2. 경계 상자 좌표를 포함한 구조화된 JSON 출력하기
    3. 객체 범주 식별하기

    이는 Kosmos-2나 Florence-2 같은 모델이 시각적 그라운딩 작업을 처리하는 방식과 비슷합니다.

    ### 출력 형식

    다음과 같은 JSON을 출력하도록 학습합니다.
    ```json
    {"objects": [{"label": "cat", "bbox": [x, y, w, h]}, {"label": "dog", "bbox": [x, y, w, h]}]}
    ```

    `bbox`는 0~1 범위로 정규화한 좌표입니다.
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
    import json
    import textwrap
    import os
    import warnings
    warnings.filterwarnings('ignore')

    # 시각화용 supervision 가져오기
    import supervision as sv


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'사용 장치: {device}')
    return (
        AutoModelForCausalLM,
        AutoTokenizer,
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
        np,
        os,
        plt,
        sv,
        torch,
        tqdm,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1단계: 사전 학습된 VLM 불러오기

    이전 노트북의 VLM을 불러옵니다. 이전 노트북을 아직 실행하지 않았다면 모델을 처음부터 다시 구성합니다.
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
    # 먼저 모델 구조 정의(이전 노트북과 동일)

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
            prompt: str = "What objects are in this image?",
            max_new_tokens: int = 150,
            temperature: float = 0.3,
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
    vision_model_name = "google/vit-large-patch16-224"
    lm_model_name = "HuggingFaceTB/SmolLM-360M"
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
    ## 2단계: 객체 탐지 데이터셋 불러오기

    경계 상자 주석이 있는 작은 동물 탐지 데이터셋을 사용합니다. 학습 이미지 700장으로 구성되며 고양이, 개, 말, 소 등의 범주가 있습니다.
    """)
    return


@app.cell
def _(execution_step_3, load_dataset):
    # 원본 코드 셀 순서와 모델·데이터 변경의 실행 순서를 보존합니다.
    execution_step_4 = execution_step_3 + 1
    # 학습을 위해 동물 탐지 데이터 전체 사용
    od_dataset = load_dataset('Francesco/animals-ij5d2', split='train')

    # 데이터를 늘리기 위해 검증 데이터도 로딩
    val_data = load_dataset('Francesco/animals-ij5d2', split='validation')

    # 학습·검증 데이터를 결합해 학습 데이터 확대
    from datasets import concatenate_datasets
    od_dataset = concatenate_datasets([od_dataset, val_data])

    print(f'데이터셋 크기: {len(od_dataset)} (학습·검증 데이터 결합)')
    print(f'데이터 특성: {od_dataset.features}')

    # 범주 이름 추출
    category_names = od_dataset.features['objects']['category'].feature.names
    print(f'\n범주: {category_names}')
    return category_names, execution_step_4, od_dataset


@app.cell
def _(category_names, execution_step_4, np, od_dataset, plt, sv):
    # 원본 코드 셀 순서와 모델·데이터 변경의 실행 순서를 보존합니다.
    execution_step_5 = execution_step_4 + 1
    # 샘플을 확인하고 supervision으로 시각화
    _sample = od_dataset[0]
    print(f'이미지 크기: {_sample['width']}x{_sample['height']}')
    print(f'객체: {_sample['objects']}')

    # supervision의 Detections로 변환
    def create_sv_detections(objects, category_names):
        """데이터셋 객체를 supervision의 Detections로 변환합니다.
        """
        bboxes = []
        class_ids = []
        for bbox, cat_id in zip(objects['bbox'], objects['category']):
            x, y, w, h = bbox
            bboxes.append([x, y, x + w, y + h])
            class_ids.append(cat_id)  # supervision은 xyxy 형식 사용
        if len(bboxes) == 0:
            return sv.Detections.empty()
        return sv.Detections(xyxy=np.array(bboxes), class_id=np.array(class_ids))
    box_annotator = sv.BoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(text_scale=0.5, text_thickness=1)
    image_np = np.array(_sample['image'])
    detections = create_sv_detections(_sample['objects'], category_names)
    labels = [category_names[i] for i in detections.class_id]
    annotated = box_annotator.annotate(image_np.copy(), detections)
    annotated = label_annotator.annotate(annotated, detections, labels=labels)
    plt.figure(figsize=(5, 5))
    # 주석 표시 도구 생성
    plt.imshow(annotated)
    plt.title('정답 경계 상자를 표시한 샘플(supervision 사용)', fontsize=12)
    plt.axis('off')
    # 이미지에 주석 표시
    plt.show()
    return create_sv_detections, execution_step_5


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3단계: 지시 데이터셋 만들기

    객체 탐지 주석을 지시·응답 쌍으로 변환해야 합니다.

    **입력(지시):** `"What objects are in this image? Output as JSON."`

    **출력(응답):** `{"objects": [{"label": "cat", "bbox": [0.1, 0.2, 0.3, 0.4]}]}`

    좌표를 일관되게 표현하도록 경계 상자를 0~1 범위로 정규화합니다.
    """)
    return


@app.cell
def _(category_names, execution_step_5, json, od_dataset):
    # 원본 코드 셀 순서와 모델·데이터 변경의 실행 순서를 보존합니다.
    execution_step_6 = execution_step_5 + 1

    def create_od_json(objects, width, height, category_names):
        """경계 상자 주석을 JSON 형식으로 변환합니다.
        """
        result = {'objects': []}
        for bbox, cat_id in zip(objects['bbox'], objects['category']):
            x, y, w, h = bbox
            norm_bbox = [round(x / width, 3), round(y / height, 3), round(w / width, 3), round(h / height, 3)]  # 0~1 범위로 정규화
            result['objects'].append({'label': category_names[cat_id], 'bbox': norm_bbox})
        return json.dumps(result)
    _sample = od_dataset[0]
    json_output = create_od_json(_sample['objects'], _sample['width'], _sample['height'], category_names)
    print('JSON 출력 예시:')
    # 샘플로 테스트
    print(json.dumps(json.loads(json_output), indent=2))
    return create_od_json, execution_step_6


@app.cell
def _(
    DataLoader,
    Dataset,
    category_names,
    create_od_json,
    execution_step_6,
    image_processor,
    od_dataset,
    tokenizer,
):
    # 원본 코드 셀 순서와 모델·데이터 변경의 실행 순서를 보존합니다.
    execution_step_7 = execution_step_6 + 1
    import random
    INSTRUCTION_PROMPTS = ['What objects are in this image? Output as JSON.', 'Detect all objects in this image and return JSON with bounding boxes.', 'List the objects with their locations in JSON format.', 'Find all objects and output their bounding boxes as JSON.']
    # 다양성을 위한 여러 지시 프롬프트

    class ODInstructionDataset(Dataset):
        """객체 탐지 지시 미세 조정용 데이터셋입니다.
        """

        def __init__(self, hf_dataset, image_processor, tokenizer, category_names, max_length=256):
            self.dataset = hf_dataset
            self.image_processor = image_processor
            self.tokenizer = tokenizer
            self.category_names = category_names
            self.max_length = max_length

        def __len__(self):
            return len(self.dataset)

        def __getitem__(self, idx):
            item = self.dataset[idx]
            image = item['image'].convert('RGB')
            pixel_values = self.image_processor(image, return_tensors='pt').pixel_values.squeeze(0)
            instruction = random.choice(INSTRUCTION_PROMPTS)
            response = create_od_json(item['objects'], item['width'], item['height'], self.category_names)
            full_text = f'{instruction} {response}{self.tokenizer.eos_token}'
            encoding = self.tokenizer(full_text, max_length=self.max_length, padding='max_length', truncation=True, return_tensors='pt')
            input_ids = encoding['input_ids'].squeeze(0)
            attention_mask = encoding['attention_mask'].squeeze(0)  # 이미지 처리
            instruction_tokens = self.tokenizer.encode(instruction, add_special_tokens=False)
            instruction_len = len(instruction_tokens)
            labels = input_ids.clone()
            labels[:instruction_len] = -100  # 지시·응답 쌍 생성
            labels[attention_mask == 0] = -100
            return {'pixel_values': pixel_values, 'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}
    train_dataset = ODInstructionDataset(od_dataset, image_processor, tokenizer, category_names)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0)  # 형식: "<instruction> <response><eos>"
    print(f'학습 샘플 수: {len(train_dataset)}')  # 중요: EOS 토큰으로 JSON 뒤에서 생성을 끝내도록 학습
    print(f'배치 수: {len(train_loader)}')
    _sample = train_dataset[0]
    decoded = tokenizer.decode(_sample['input_ids'])  # 토큰화
    print(f'\nEOS 토큰을 포함한 샘플:')
    # 데이터셋과 데이터로더 생성
    # 학습 데이터의 EOS 토큰 확인
    print(decoded[:200] + '...')  # 레이블 생성: 지시·패딩은 마스킹하고 응답과 EOS에만 손실 계산
    return INSTRUCTION_PROMPTS, execution_step_7, train_loader


@app.cell
def _(execution_step_7, tokenizer, train_loader):
    # 원본 코드 셀 순서와 모델·데이터 변경의 실행 순서를 보존합니다.
    execution_step_8 = execution_step_7 + 1
    # 배치 확인
    batch = next(iter(train_loader))
    print(f'배치 pixel_values shape: {batch['pixel_values'].shape}')
    print(f'배치 input_ids shape: {batch['input_ids'].shape}')

    # 첫 샘플을 디코딩해 형식 확인
    print(f'\n샘플 텍스트(첫 항목):')
    print(tokenizer.decode(batch['input_ids'][0], skip_special_tokens=True)[:300])
    return (execution_step_8,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4단계: 학습 전 테스트

    지시 미세 조정 전에 모델이 무엇을 출력하는지 확인합니다. 이전에 캡션을 학습했으므로 구조화된 JSON 대신 캡션을 생성할 것으로 예상합니다.
    """)
    return


@app.cell
def _(Image, execution_step_8):
    # 원본 코드 셀 순서와 모델·데이터 변경의 실행 순서를 보존합니다.
    execution_step_9 = execution_step_8 + 1
    def generate_od_response(model, image, image_processor, device, prompt="What objects are in this image? Output as JSON."):
        """이미지의 객체 탐지 응답을 생성합니다.
        """
        model.eval()
    
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        image = image.convert('RGB')
    
        pixel_values = image_processor(image, return_tensors="pt").pixel_values.to(device)
    
        # 구조화된 JSON 출력을 위해 탐욕적 디코딩 사용(do_sample=False)
        # 더 안정적이고 일관된 출력 생성
        response = model.generate(
            pixel_values,
            prompt=prompt,
            max_new_tokens=150,
            temperature=1.0,  # do_sample=False이면 무시됨
            do_sample=False,  # JSON용 탐욕적 디코딩
        )
    
        return response  # 원시 응답을 반환하고 정리는 parse_od_response에 맡김

    return execution_step_9, generate_od_response


@app.cell
def _(
    category_names,
    create_od_json,
    device,
    execution_step_9,
    generate_od_response,
    image_processor,
    od_dataset,
    vlm,
):
    # 원본 코드 셀 순서와 모델·데이터 변경의 실행 순서를 보존합니다.
    execution_step_10 = execution_step_9 + 1
    # 학습 전 샘플 테스트 및 응답 보관
    test_indices = [0, 5, 10]
    test_samples_data = []  # 나중에 비교하도록 저장
    print('=' * 70)
    print('지시 미세 조정 전 출력(캡션 생성 예상)')
    print('=' * 70)
    before_responses = []
    for _idx in test_indices:
        _sample = od_dataset[_idx]
        _response = generate_od_response(vlm, _sample['image'], image_processor, device)
        before_responses.append(_response)
        test_samples_data.append({'idx': _idx, 'image': _sample['image'], 'objects': _sample['objects'], 'width': _sample['width'], 'height': _sample['height']})
        _gt_json = create_od_json(_sample['objects'], _sample['width'], _sample['height'], category_names)
        print(f'\n이미지 {_idx}:')
        print(f'  모델 출력: {_response[:100]}...')
        print(f'  예상 JSON: {_gt_json[:80]}...')
    return before_responses, execution_step_10, test_samples_data


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5단계: 지시 미세 조정

    객체 탐지를 위한 구조화된 JSON을 출력하도록 모델을 미세 조정합니다.
    """)
    return


@app.cell
def _(device, execution_step_10, os, output_dir, torch, tqdm):
    # 원본 코드 셀 순서와 모델·데이터 변경의 실행 순서를 보존합니다.
    execution_step_11 = execution_step_10 + 1
    def train_vlm_od(model, train_loader, num_epochs=5, lr=1e-4, checkpoint_path=str(output_dir / "od_checkpoint.pt")):
        """재개용 체크포인트를 지원하며 VLM을 객체 탐지용으로 학습합니다.
        """
    
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable_params, lr=lr)
    
        # 학습 재개용 체크포인트 불러오기 시도
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
                # 옵티마이저 상태를 장치로 이동
                for state in optimizer.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.to(device)
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
            
                # 에포크마다 체크포인트 저장
                torch.save({
                    'epoch': epoch + 1,
                    'losses': losses,
                    'projector_state_dict': model.projector.state_dict(),
                    'language_model_state_dict': model.language_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, checkpoint_path)
                print(f'체크포인트 저장 위치: {checkpoint_path}')
    
        except KeyboardInterrupt:
            print("\n" + "="*70)
            print('학습을 중단했습니다.')
            print(f'완료한 에포크 수: {len(losses)}에포크')
            # 중단 시 체크포인트 저장
            torch.save({
                'epoch': len(losses),
                'losses': losses,
                'projector_state_dict': model.projector.state_dict(),
                'language_model_state_dict': model.language_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, checkpoint_path)
            print(f'체크포인트 저장 위치: {checkpoint_path}')
            print('체크포인트에서 재개하려면 학습을 다시 실행하세요.')
            print("="*70)
    
        return losses

    return execution_step_11, train_vlm_od


@app.cell
def _(execution_step_11, output_dir, train_loader, train_vlm_od, vlm):
    # 원본 코드 셀 순서와 모델·데이터 변경의 실행 순서를 보존합니다.
    execution_step_12 = execution_step_11 + 1
    # 결합한 데이터셋으로 더 많은 에포크 학습
    # Ctrl+C로 학습을 중단한 뒤 다시 실행하면 재개 가능
    losses = train_vlm_od(vlm, train_loader, num_epochs=10, lr=1e-4, checkpoint_path=str(output_dir / "od_checkpoint.pt"))
    return execution_step_12, losses


@app.cell
def _(execution_step_12, losses, plt):
    # 원본 코드 셀 순서와 모델·데이터 변경의 실행 순서를 보존합니다.
    execution_step_13 = execution_step_12 + 1
    # 학습 손실 그래프
    plt.figure(figsize=(8, 4))
    plt.plot(range(1, len(losses)+1), losses, marker='o')
    plt.xlabel('에포크')
    plt.ylabel('손실')
    plt.title('지시 미세 조정 손실')
    plt.grid(True, alpha=0.3)
    plt.show()
    return (execution_step_13,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6단계: 학습 후 테스트

    지시 미세 조정 전후의 출력을 비교합니다.
    """)
    return


@app.cell
def _(
    before_responses,
    category_names,
    create_od_json,
    device,
    execution_step_13,
    generate_od_response,
    image_processor,
    test_samples_data,
    vlm,
):
    # 원본 코드 셀 순서와 모델·데이터 변경의 실행 순서를 보존합니다.
    execution_step_14 = execution_step_13 + 1
    # 같은 샘플의 학습 후 결과를 학습 전과 비교
    print('=' * 70)
    print('지시 미세 조정 전후 비교')
    print('=' * 70)
    after_responses = []
    for _i, (_sample_data, _before_resp) in enumerate(zip(test_samples_data, before_responses)):
        _response = generate_od_response(vlm, _sample_data['image'], image_processor, device)
        after_responses.append(_response)
        _gt_json = create_od_json(_sample_data['objects'], _sample_data['width'], _sample_data['height'], category_names)
        print(f"\n{'=' * 70}")
        print(f'이미지 {_sample_data['idx']}:')
        print(f"{'=' * 70}")
        print(f'학습 전(캡션 모델): {_before_resp[:120]}...')
        print(f'학습 후(지시 미세 조정): {_response[:120]}...')
        print(f'정답: {_gt_json[:120]}...')
    return after_responses, execution_step_14


@app.cell
def _(create_sv_detections, execution_step_14, json, np, plt, sv):
    # 원본 코드 셀 순서와 모델·데이터 변경의 실행 순서를 보존합니다.
    execution_step_15 = execution_step_14 + 1

    def parse_od_response(response):
        """모델 응답의 JSON을 파싱하며 잘못된 형식도 가능한 범위에서 처리합니다.
        """
        try:  # JSON 시작 위치 찾기
            start = response.find('{"objects"')
            if start == -1:
                start = response.find('{')
            if start == -1:
                return None
            brace_count = 0
            bracket_count = 0  # 중괄호 짝을 맞춰 올바른 JSON 끝 찾기
            end = start
            for i, char in enumerate(response[start:], start):
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                elif char == '[':
                    bracket_count += 1
                elif char == ']':
                    bracket_count -= 1
                if brace_count == 0 and i > start:
                    end = i + 1
                    break
            if end > start:
                json_str = response[start:end]  # 모든 중괄호가 짝을 이루면 유효한 JSON 객체 종료
                try:
                    return json.loads(json_str)
                except json.JSONDecodeError:
                    pass
            idx = response.find(']}', start)
            if idx != -1:
                json_str = response[start:idx + 2]
                try:
                    return json.loads(json_str)
                except json.JSONDecodeError:
                    try:
                        return json.loads(json_str + '}')  # 대체 경로: ]} 패턴을 찾아 유효한 JSON 구성
                    except:
                        pass
            obj_start = response.find('[', start)
            obj_end = response.find(']', obj_start)
            if obj_start != -1 and obj_end != -1:
                try:
                    array_str = response[obj_start:obj_end + 1]  # 빠진 닫는 중괄호 추가 시도
                    return {'objects': json.loads(array_str)}
                except:
                    pass
        except Exception as e:
            pass
        return None  # 마지막 대안: objects 배열을 찾아 재구성
    test_cases = ['What objects are in this image? Output as JSON. {"objects": [{"label": "cat", "bbox": [0.022, 0.006, 0.773, 0.994]}]}', 'What objects are in this image? Output as JSON. {"objects": [{"label": "cat", "bbox": [0.022, 0.006, 0.773, 0.994]}]}]}]}', 'What objects are in this image? Output as JSON. {"objects": [{"label": "cat", "bbox": [0.0, 0.0, 0.727, 0.994]}']
    print('다양한 출력으로 파서 테스트:')
    for _i, test in enumerate(test_cases):
        _parsed = parse_od_response(test)
        status = 'OK' if _parsed else 'FAILED'
        print(f'  테스트 {_i + 1}: {status} -> {_parsed}')

    def create_sv_detections_from_json(parsed_json, width, height, category_names):
        """파싱한 JSON을 supervision의 Detections로 변환합니다.
        """
        if not parsed_json or 'objects' not in parsed_json:
            return (sv.Detections.empty(), [])
        bboxes = []
        class_ids = []
        labels_list = []
    # 잘못된 여러 출력으로 파서 테스트
        label_to_id = {name: i for i, name in enumerate(category_names)}
        for obj in parsed_json['objects']:
            try:
                nx, ny, nw, nh = obj['bbox']
                x1 = nx * width
                y1 = ny * height
                x2 = (nx + nw) * width
                y2 = (ny + nh) * height
                bboxes.append([x1, y1, x2, y2])
                label = obj['label']
                labels_list.append(label)
                class_ids.append(label_to_id.get(label, 0))
            except:
                continue
        if len(bboxes) == 0:
            return (sv.Detections.empty(), [])
        return (sv.Detections(xyxy=np.array(bboxes), class_id=np.array(class_ids)), labels_list)

    def visualize_comparison_sv(image, response, gt_objects, width, height, category_names, title=''):
        """supervision으로 예측과 정답을 시각화합니다.
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
        image_np = np.array(image)  # 레이블에서 범주 ID로 역매핑 생성
        gt_box_annotator = sv.BoxAnnotator(thickness=3, color=sv.Color.GREEN)
        gt_label_annotator = sv.LabelAnnotator(text_scale=0.6, text_thickness=2, color=sv.Color.GREEN)
        pred_box_annotator = sv.BoxAnnotator(thickness=3, color=sv.Color.RED)
        pred_label_annotator = sv.LabelAnnotator(text_scale=0.6, text_thickness=2, color=sv.Color.RED)
        gt_detections = create_sv_detections(gt_objects, category_names)
        gt_labels = [category_names[i] for i in gt_detections.class_id]  # 정규화를 되돌리고 xyxy로 변환
        gt_annotated = gt_box_annotator.annotate(image_np.copy(), gt_detections)
        gt_annotated = gt_label_annotator.annotate(gt_annotated, gt_detections, labels=gt_labels)
        ax1.imshow(gt_annotated)
        ax1.set_title('정답', fontsize=14, color='green', fontweight='bold')
        ax1.axis('off')
        parsed = parse_od_response(response)
        pred_annotated = image_np.copy()
        if parsed and 'objects' in parsed:
            pred_detections, pred_labels = create_sv_detections_from_json(parsed, width, height, category_names)
            if len(pred_detections) > 0:
                pred_annotated = pred_box_annotator.annotate(pred_annotated, pred_detections)
                pred_annotated = pred_label_annotator.annotate(pred_annotated, pred_detections, labels=pred_labels)
            ax2.set_title('모델 예측', fontsize=14, color='red', fontweight='bold')
        else:
            ax2.set_title('모델 예측(파싱 실패)', fontsize=14, color='red', fontweight='bold')
        ax2.imshow(pred_annotated)
        ax2.axis('off')
        if title:
            plt.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.show()
        return parsed  # 다른 색의 주석 도구 생성: 정답(왼쪽), 예측(오른쪽)

    return (
        create_sv_detections_from_json,
        execution_step_15,
        parse_od_response,
        visualize_comparison_sv,
    )


@app.cell
def _(
    after_responses,
    before_responses,
    category_names,
    create_sv_detections,
    create_sv_detections_from_json,
    execution_step_15,
    np,
    parse_od_response,
    plt,
    sv,
    test_samples_data,
):
    # 원본 코드 셀 순서와 모델·데이터 변경의 실행 순서를 보존합니다.
    execution_step_16 = execution_step_15 + 1
    # supervision으로 학습 전후 시각적 비교
    # 정답, 학습 전, 학습 후의 3열 표시

    def visualize_before_after_sv(image, before_response, after_response, gt_objects, width, height, category_names, title=''):
        """supervision으로 정답·학습 전·학습 후를 시각화합니다.
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        image_np = np.array(image)
        gt_box = sv.BoxAnnotator(thickness=3, color=sv.Color.GREEN)
        gt_label = sv.LabelAnnotator(text_scale=0.5, text_thickness=2, color=sv.Color.GREEN)  # 주석 도구
        pred_box = sv.BoxAnnotator(thickness=3, color=sv.Color.RED)
        pred_label = sv.LabelAnnotator(text_scale=0.5, text_thickness=2, color=sv.Color.RED)
        gt_detections = create_sv_detections(gt_objects, category_names)
        gt_labels = [category_names[i] for i in gt_detections.class_id]
        gt_img = gt_box.annotate(image_np.copy(), gt_detections)
        gt_img = gt_label.annotate(gt_img, gt_detections, labels=gt_labels)  # 1. 정답
        axes[0].imshow(gt_img)
        axes[0].set_title('정답', fontsize=14, color='green', fontweight='bold')
        axes[0].axis('off')
        before_parsed = parse_od_response(before_response)
        before_img = image_np.copy()
        if before_parsed and 'objects' in before_parsed:
            before_det, before_labels = create_sv_detections_from_json(before_parsed, width, height, category_names)
            if len(before_det) > 0:
                before_img = pred_box.annotate(before_img, before_det)  # 2. 학습 전(유효한 JSON이 없을 가능성이 높음)
                before_img = pred_label.annotate(before_img, before_det, labels=before_labels)
        axes[1].imshow(before_img)
        axes[1].set_title('학습 전\n(JSON 없음)', fontsize=14, color='orange', fontweight='bold')
        axes[1].axis('off')
        after_parsed = parse_od_response(after_response)
        after_img = image_np.copy()
        if after_parsed and 'objects' in after_parsed:
            after_det, after_labels = create_sv_detections_from_json(after_parsed, width, height, category_names)
            if len(after_det) > 0:
                after_img = pred_box.annotate(after_img, after_det)
                after_img = pred_label.annotate(after_img, after_det, labels=after_labels)
            axes[2].set_title('학습 후\n(JSON 출력)', fontsize=14, color='red', fontweight='bold')  # 3. 학습 후
        else:
            axes[2].set_title('학습 후\n(파싱 실패)', fontsize=14, color='red', fontweight='bold')
        axes[2].imshow(after_img)
        axes[2].axis('off')
        if title:
            plt.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.show()
    for _i, (_sample_data, _before_resp, after_resp) in enumerate(zip(test_samples_data, before_responses, after_responses)):
        print(f"\n{'=' * 70}")
        print(f'이미지 {_sample_data['idx']}')
        print(f'학습 전: {_before_resp[:60]}...')
        print(f'학습 후: {after_resp[:60]}...')
        print(f"{'=' * 70}")
    # 모든 테스트 샘플의 학습 전후 시각화
        visualize_before_after_sv(_sample_data['image'], _before_resp, after_resp, _sample_data['objects'], _sample_data['width'], _sample_data['height'], category_names, title=f"Image {_sample_data['idx']} - Before vs After Instruction Tuning")
    return (execution_step_16,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7단계: 새로운 이미지로 테스트

    모델이 보지 못한 검증 세트의 이미지로 테스트합니다.
    """)
    return


@app.cell
def _(
    category_names,
    create_od_json,
    device,
    execution_step_16,
    generate_od_response,
    image_processor,
    load_dataset,
    visualize_comparison_sv,
    vlm,
):
    # 원본 코드 셀 순서와 모델·데이터 변경의 실행 순서를 보존합니다.
    execution_step_17 = execution_step_16 + 1
    # 최종 평가용으로 완전히 보지 못한 테스트 세트 로딩
    test_dataset = load_dataset('Francesco/animals-ij5d2', split='test')
    print(f'테스트 세트 크기: {len(test_dataset)}')
    for _idx in [0, 5, 10]:
    # 테스트 이미지 일부 확인
        if _idx >= len(test_dataset):
            continue
        _sample = test_dataset[_idx]
        _response = generate_od_response(vlm, _sample['image'], image_processor, device)
        _gt_json = create_od_json(_sample['objects'], _sample['width'], _sample['height'], category_names)
        print(f"\n{'=' * 70}")
        print(f'테스트 이미지 {_idx}')
        print(f'모델 출력: {_response}')
        print(f'정답: {_gt_json}')
        print(f"{'=' * 70}")
        visualize_comparison_sv(_sample['image'], _response, _sample['objects'], _sample['width'], _sample['height'], category_names, title=f'Test Image {_idx} - Unseen Data')
    return execution_step_17, test_dataset


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7b단계: 평가 지표(IoU, mAP)

    표준 객체 탐지 지표를 사용해 모델을 평가합니다.

    - **IoU(교집합/합집합 비율)**: 예측 상자와 정답 상자가 겹치는 정도
    - **정밀도/재현율**: 여러 IoU 임계값에서 평가
    - **mAP(평균 정밀도의 평균)**: 표준 객체 탐지 지표로, 보통 IoU=0.5에서 평가
    """)
    return


@app.cell
def _(execution_step_17):
    # 원본 코드 셀 순서와 모델·데이터 변경의 실행 순서를 보존합니다.
    execution_step_18 = execution_step_17 + 1
    def compute_iou(box1, box2):
        """정규화된 [x, y, w, h] 형식의 두 상자 사이 IoU를 계산합니다.
        
        인자:
            box1, box2: [x, y, w, h] 정규화 좌표 (0~1)
        
        반환:
            IoU 값 (0~1)
        """
        # x1, y1, x2, y2로 변환
        x1_1, y1_1 = box1[0], box1[1]
        x2_1, y2_1 = box1[0] + box1[2], box1[1] + box1[3]
    
        x1_2, y1_2 = box2[0], box2[1]
        x2_2, y2_2 = box2[0] + box2[2], box2[1] + box2[3]
    
        # 교집합
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
    
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
    
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
    
        # 합집합
        area1 = box1[2] * box1[3]
        area2 = box2[2] * box2[3]
        union = area1 + area2 - intersection
    
        if union <= 0:
            return 0.0
    
        return intersection / union


    def evaluate_detection(pred_objects, gt_objects, iou_threshold=0.5):
        """정답을 기준으로 탐지 예측을 평가합니다.
        
        인자:
            pred_objects: {"label": str, "bbox": [x,y,w,h]}의 예측 목록
            gt_objects: {"label": str, "bbox": [x,y,w,h]}의 정답 목록
            iou_threshold: 매칭으로 판단할 IoU 임계값
        
        반환:
            TP, FP, FN, precision, recall, avg_iou를 담은 딕셔너리
        """
        if not pred_objects:
            pred_objects = []
        if not gt_objects:
            gt_objects = []
    
        tp = 0  # 참 양성
        fp = 0  # 거짓 양성
        matched_gt = set()
        ious = []
    
        for pred in pred_objects:
            best_iou = 0
            best_gt_idx = -1
        
            for gt_idx, gt in enumerate(gt_objects):
                if gt_idx in matched_gt:
                    continue
            
                # 레이블 일치 확인
                if pred['label'] != gt['label']:
                    continue
            
                iou = compute_iou(pred['bbox'], gt['bbox'])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
        
            if best_iou >= iou_threshold and best_gt_idx != -1:
                tp += 1
                matched_gt.add(best_gt_idx)
                ious.append(best_iou)
            else:
                fp += 1
    
        fn = len(gt_objects) - len(matched_gt)  # 매칭되지 않은 정답
    
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        avg_iou = sum(ious) / len(ious) if ious else 0
    
        return {
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'precision': precision,
            'recall': recall,
            'avg_iou': avg_iou,
            'num_pred': len(pred_objects),
            'num_gt': len(gt_objects),
        }


    def parse_gt_to_objects(objects_dict, width, height, category_names):
        """정답 데이터셋 형식을 객체 목록으로 변환합니다.
        """
        result = []
        for bbox, cat_id in zip(objects_dict['bbox'], objects_dict['category']):
            x, y, w, h = bbox
            result.append({
                'label': category_names[cat_id],
                'bbox': [x/width, y/height, w/width, h/height]  # 정규화
            })
        return result


    print('평가 함수를 정의했습니다.')
    return evaluate_detection, execution_step_18, parse_gt_to_objects


@app.cell
def _(
    category_names,
    device,
    evaluate_detection,
    execution_step_18,
    generate_od_response,
    image_processor,
    parse_gt_to_objects,
    parse_od_response,
    test_dataset,
    tqdm,
    vlm,
):
    # 원본 코드 셀 순서와 모델·데이터 변경의 실행 순서를 보존합니다.
    execution_step_19 = execution_step_18 + 1
    # 테스트 세트 평가
    print('테스트 세트 평가 중...')
    print('=' * 70)
    all_results = []
    total_tp, total_fp, total_fn = (0, 0, 0)
    all_ious = []
    for _idx in tqdm(range(len(test_dataset))):
        _sample = test_dataset[_idx]
        _response = generate_od_response(vlm, _sample['image'], image_processor, device)
        _parsed = parse_od_response(_response)
        gt_objects = parse_gt_to_objects(_sample['objects'], _sample['width'], _sample['height'], category_names)
        pred_objects = _parsed.get('objects', []) if _parsed else []  # 예측 생성
        result = evaluate_detection(pred_objects, gt_objects, iou_threshold=0.5)
        all_results.append(result)
        total_tp += result['tp']
        total_fp += result['fp']  # 정답 추출
        total_fn += result['fn']
        if result['avg_iou'] > 0:
            all_ious.append(result['avg_iou'])
    overall_precision = total_tp / (total_tp + total_fp) if total_tp + total_fp > 0 else 0
    overall_recall = total_tp / (total_tp + total_fn) if total_tp + total_fn > 0 else 0
    f1_score = 2 * (overall_precision * overall_recall) / (overall_precision + overall_recall) if overall_precision + overall_recall > 0 else 0
    mean_iou = sum(all_ious) / len(all_ious) if all_ious else 0
    print(f"\n{'=' * 70}")  # 예측 추출
    print('평가 결과(IoU 임계값 = 0.5)')
    print(f"{'=' * 70}")
    print(f'테스트 샘플: {len(test_dataset)}')  # 평가
    print(f'\n탐지 지표:')
    print(f'  참 양성(TP): {total_tp}')
    print(f'  거짓 양성(FP): {total_fp}')
    print(f'  거짓 음성(FN): {total_fn}')
    print(f'\n  정밀도: {overall_precision:.3f}')
    print(f'  재현율: {overall_recall:.3f}')
    print(f'  F1 점수: {f1_score:.3f}')
    print(f'\n위치 추정 지표:')
    print(f'  매칭된 상자의 평균 IoU: {mean_iou:.3f}')
    # 전체 지표 계산
    print(f"{'=' * 70}")
    return (execution_step_19,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 8단계: 미세 조정한 모델 저장하기
    """)
    return


@app.cell
def _(
    INSTRUCTION_PROMPTS,
    category_names,
    execution_step_19,
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
    execution_step_20 = execution_step_19 + 1
    # 지시 미세 조정한 모델 저장
    save_dir = str(output_dir / "mini-vlm-od")
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
        'od_config': {
            'category_names': category_names,
            'instruction_prompts': INSTRUCTION_PROMPTS,
        }
    }, f"{save_dir}/mini_vlm_od.pt")

    tokenizer.save_pretrained(f"{save_dir}/tokenizer")
    image_processor.save_pretrained(f"{save_dir}/image_processor")

    print(f'모델 저장 위치: {save_dir}/')
    print(f'저장 내용: {os.listdir(save_dir)}')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 정리

    객체 탐지를 위해 VLM을 지시 미세 조정했습니다.

    ### 수행한 작업

    1. 캡션 노트북에서 **사전 학습된 VLM 불러오기**(ViT-Large + SmolLM-360M)
    2. 생성 종료 시점을 가르칠 EOS 토큰을 포함한 **지시 데이터셋 만들기**
    3. 학습 데이터를 늘리기 위해 **학습·검증 데이터 결합하기**(이미지 800장)
    4. 재개 가능한 체크포인트를 사용해 **10에포크 미세 조정하기**
    5. IoU, 정밀도, 재현율, F1 등의 **지표로 평가하기**

    ### JSON 생성을 위한 주요 수정

    모델이 불필요한 `}]}]`를 추가로 생성하던 문제에 다음과 같이 대응했습니다.
    1. **EOS 토큰 누락**: 학습 데이터의 JSON 뒤에 `<eos>` 추가
    2. **샘플링의 무작위성**: 탐욕적 디코딩으로 변경(`do_sample=False`)
    3. **후처리**: 처음으로 유효한 `]}`가 나타난 지점 뒤를 잘라냄

    ### 핵심 내용

    - **EOS 토큰의 중요성**: 생성을 정확히 언제 끝낼지 가르칩니다.
    - **구조화된 출력에는 탐욕적 디코딩**: JSON은 샘플링보다 안정적으로 생성할 수 있습니다.
    - **지시 형식의 중요성**: 다양한 프롬프트가 일반화에 도움이 됩니다.
    - **레이블 마스킹**: 지시 부분이 아닌 응답 부분에만 손실을 계산하는 것이 중요합니다.

    ### 평가 결과

    다음 객체 탐지 지표를 확인할 수 있습니다.
    - **정밀도**: 예측 상자 중 올바른 상자의 비율
    - **재현율**: 정답 상자 중 탐지한 상자의 비율
    - **IoU**: 경계 상자 위치 추정의 품질
    - **F1 점수**: 정밀도와 재현율의 조화 평균

    ### 한계

    - 적은 학습 이미지(800장)
    - 제한된 객체 범주(11개 클래스)
    - 이미지 대부분에 객체가 하나만 존재
    - 교육용 모델로, 상용 서비스용으로 준비된 모델은 아님
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 참고 자료

    - [이전 노트북: 처음부터 만드는 최소 구성의 VLM](part-01-minimal-vlm.py)
    - [Kosmos-2: 멀티모달 대규모 언어 모델의 그라운딩](https://arxiv.org/abs/2306.14824)
    - [Florence-2: 통합 표현의 발전](https://arxiv.org/abs/2311.06242)
    - [동물 탐지 데이터셋](https://huggingface.co/datasets/Francesco/animals-ij5d2)
    - [Hugging Face 객체 탐지 안내](https://huggingface.co/docs/datasets/en/object_detection)
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
 
    """)
    return


if __name__ == "__main__":
    app.run()
