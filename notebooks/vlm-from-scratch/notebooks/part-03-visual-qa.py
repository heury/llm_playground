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
    # 시각적 질의응답을 위한 VLM 지시 미세 조정

    Nipun Batra · 2025-12-25
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 소개

    이전 노트북에서는 다음을 실습했습니다.
    1. [최소 구성의 VLM을 직접 만들고](part-01-minimal-vlm.py) 이미지 캡션 학습하기
    2. [객체 탐지를 위한 지시 미세 조정](part-02-object-detection.py)으로 JSON 출력하기

    이제 **시각적 질의응답(Visual Question Answering, VQA)**을 다룹니다. 이미지에 관한 자연어 질문에 답하도록 모델을 학습시킵니다.

    ### 예시

    | 이미지 | 질문 | 답변 |
    |-------|----------|--------|
    | 🖼️ 고양이 사진 | "What animal is this?"(어떤 동물인가요?) | "cat"(고양이) |
    | 🖼️ 해변 풍경 | "What is the weather like?"(날씨가 어떤가요?) | "sunny"(맑음) |
    | 🖼️ 주방 | "How many people are there?"(사람이 몇 명 있나요?) | "2" |

    객체 탐지의 구조화된 JSON과 달리 VQA에는 질문에 대한 짧고 직접적인 **자연어 답변**이 필요합니다.
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
    import textwrap
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
        Image,
        ViTImageProcessor,
        ViTModel,
        device,
        execution_step_1,
        load_dataset,
        nn,
        os,
        plt,
        textwrap,
        torch,
        tqdm,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1단계: 사전 학습된 VLM 불러오기(캡션 모델)

    객체 탐지(OD) 모델 대신 **캡션을 학습한 모델**(`mini-vlm-flickr8k/`)을 불러옵니다. 이유는 다음과 같습니다.
    - 캡션 모델은 유창한 자연어를 생성합니다.
    - 객체 탐지 모델은 JSON 출력에 치우쳐 있습니다.
    - VQA에는 자연어 답변이 필요합니다.
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
    # 모델 구조(이전 노트북과 동일)

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
            max_new_tokens = 20,
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
    ## 2단계: VQA 데이터셋 불러오기

    대표적인 시각적 질의응답 벤치마크 VQAv2를 사용합니다. 각 샘플에는 다음이 들어 있습니다.
    - 이미지
    - 이미지에 관한 질문
    - 여러 사람이 작성한 답변(가장 빈도가 높은 답변 사용)
    """)
    return


@app.cell
def _(execution_step_3, load_dataset):
    # 원본 코드 셀 순서와 모델·데이터 변경의 실행 순서를 보존합니다.
    execution_step_4 = execution_step_3 + 1
    # VQAv2 데이터셋 로딩(충분히 큰 검증 분할 사용)
    # 큰 데이터셋을 처리하기 위해 스트리밍 사용
    vqa_dataset_stream = load_dataset('lmms-lab/VQAv2', split='validation', streaming=True)
    num_samples = 4000
    # 학습용으로 더 큰 부분집합 선택
    print(f'불러올 샘플 수: {num_samples}개(VQAv2)...')  # 2,000개에서 확대
    vqa_samples = []
    for _i, _sample in enumerate(vqa_dataset_stream):
        if _i >= num_samples:
            break
        vqa_samples.append(_sample)
        if (_i + 1) % 1000 == 0:
            print(f'  불러온 샘플 수: {_i + 1}개 샘플...')
    print(f'\n불러온 샘플 수: {len(vqa_samples)}개 VQA 샘플')
    print(f'샘플 키: {vqa_samples[0].keys()}')
    return execution_step_4, vqa_samples


@app.cell
def _(execution_step_4, plt, vqa_samples):
    # 원본 코드 셀 순서와 모델·데이터 변경의 실행 순서를 보존합니다.
    execution_step_5 = execution_step_4 + 1
    # 샘플 일부 확인

    def get_most_common_answer(answers):
        """주석 작성자들의 답변 중 가장 빈도가 높은 답변을 구합니다.
        """
        answer_counts = {}
        for ans in answers:
            a = ans['answer']
            answer_counts[a] = answer_counts.get(a, 0) + 1
        return max(answer_counts, key=answer_counts.get)
    # 예시 일부 표시
    _fig, _axes = plt.subplots(2, 3, figsize=(15, 10))
    _axes = _axes.flatten()
    for _i, _ax in enumerate(_axes):
        _sample = vqa_samples[_i]
        _image = _sample['image']
        _question = _sample['question']
        answer = get_most_common_answer(_sample['answers'])
        _ax.imshow(_image)
        _ax.set_title(f'질문: {_question}\nA: {answer}', fontsize=10, wrap=True)
        _ax.axis('off')
    plt.suptitle('VQAv2 데이터셋 샘플', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
    return execution_step_5, get_most_common_answer


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3단계: VQA 지시 데이터셋 만들기

    각 샘플의 형식은 다음과 같습니다.
    - **입력:** `"Question: {question} Answer:"`
    - **정답:** `"{answer}"`

    모델은 이미지와 질문을 보고 답변을 완성하는 방법을 학습합니다.
    """)
    return


@app.cell
def _(
    DataLoader,
    Dataset,
    execution_step_5,
    get_most_common_answer,
    image_processor,
    tokenizer,
    vqa_samples,
):
    # 원본 코드 셀 순서와 모델·데이터 변경의 실행 순서를 보존합니다.
    execution_step_6 = execution_step_5 + 1
    class VQADataset(Dataset):
        """VQA 지시 미세 조정용 데이터셋입니다.
        """

        def __init__(self, samples, image_processor, tokenizer, max_length=64):
            self.samples = samples
            self.image_processor = image_processor
            self.tokenizer = tokenizer
            self.max_length = max_length

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            sample = self.samples[idx]

            # 이미지 처리
            image = sample['image'].convert('RGB')
            pixel_values = self.image_processor(image, return_tensors="pt").pixel_values.squeeze(0)

            # 질문과 답변 추출
            question = sample['question']
            answer = get_most_common_answer(sample['answers'])

            # 형식: "Question: {q} Answer: {a}<eos>"
            # EOS 토큰으로 답변 뒤에서 생성을 끝내도록 학습
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

            # 레이블 생성: 프롬프트를 마스킹하고 답변과 EOS만 학습
            prompt_tokens = self.tokenizer.encode(prompt, add_special_tokens=False)
            prompt_len = len(prompt_tokens)

            labels = input_ids.clone()
            labels[:prompt_len] = -100  # 프롬프트에는 손실을 계산하지 않음
            labels[attention_mask == 0] = -100  # 패딩에는 손실을 계산하지 않음

            return {
                'pixel_values': pixel_values,
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'labels': labels,
            }


    # 학습·테스트 분할 생성(총 4,000개 중 학습 3,600개, 테스트 400개)
    train_samples = vqa_samples[:3600]
    test_samples = vqa_samples[3600:]

    train_dataset = VQADataset(train_samples, image_processor, tokenizer)
    test_dataset = VQADataset(test_samples, image_processor, tokenizer)

    train_loader = DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True,
        num_workers=0,
    )

    print(f'학습 샘플: {len(train_dataset)}')
    print(f'테스트 샘플: {len(test_dataset)}')
    print(f'배치 수: {len(train_loader)}')
    return execution_step_6, test_samples, train_loader


@app.cell
def _(execution_step_6, tokenizer, train_loader):
    # 원본 코드 셀 순서와 모델·데이터 변경의 실행 순서를 보존합니다.
    execution_step_7 = execution_step_6 + 1
    # 배치 확인
    batch = next(iter(train_loader))
    print(f'배치 pixel_values shape: {batch['pixel_values'].shape}')
    print(f'배치 input_ids shape: {batch['input_ids'].shape}')

    # 첫 샘플 디코딩
    print(f'\n샘플 텍스트:')
    print(tokenizer.decode(batch['input_ids'][0], skip_special_tokens=True))
    return (execution_step_7,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4단계: 학습 전 테스트

    지시 미세 조정 전에 캡션 모델이 VQA 질문에 어떻게 답하는지 확인합니다.
    """)
    return


@app.cell
def _(Image, execution_step_7):
    # 원본 코드 셀 순서와 모델·데이터 변경의 실행 순서를 보존합니다.
    execution_step_8 = execution_step_7 + 1
    def generate_vqa_answer(model, image, question, image_processor, device):
        """시각적 질문에 대한 답변을 생성합니다.
        """
        model.eval()

        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        image = image.convert('RGB')

        pixel_values = image_processor(image, return_tensors="pt").pixel_values.to(device)

        prompt = f"Question: {question} Answer:"

        # 짧고 사실적인 답변을 위해 탐욕적 디코딩 사용(샘플링 없음)
        response = model.generate(
            pixel_values,
            prompt=prompt,
            max_new_tokens=10,  # VQA 답변은 짧음
            temperature=1.0,
            do_sample=False,  # 탐욕적 디코딩은 VQA에서 더 안정적
        )

        # 답변 부분만 추출
        if "Answer:" in response:
            answer = response.split("Answer:")[-1].strip()
        else:
            answer = response

        # 정리: VQA 답변은 짧으므로 처음 몇 단어만 사용
        answer = answer.split('.')[0].strip()  # 첫 마침표에서 중단
        words = answer.split()
        if len(words) > 5:
            answer = ' '.join(words[:5])  # 최대 5단어로 제한

        return answer, response

    return execution_step_8, generate_vqa_answer


@app.cell
def _(
    device,
    execution_step_8,
    generate_vqa_answer,
    get_most_common_answer,
    image_processor,
    test_samples,
    vlm,
):
    # 원본 코드 셀 순서와 모델·데이터 변경의 실행 순서를 보존합니다.
    execution_step_9 = execution_step_8 + 1
    # 학습 전 샘플 일부 테스트
    test_indices = [0, 3, 6, 9, 12]
    test_data_for_comparison = []
    print('=' * 70)
    print('VQA 지시 미세 조정 전')
    print('(캡션을 학습했으며 질의응답은 학습하지 않은 모델)')
    print('=' * 70)
    before_answers = []
    for idx in test_indices:
        _sample = test_samples[idx]
        _image = _sample['image']
        _question = _sample['question']
        _gt_answer = get_most_common_answer(_sample['answers'])
        _pred_answer, full_response = generate_vqa_answer(vlm, _image, _question, image_processor, device)
        before_answers.append(_pred_answer)
        test_data_for_comparison.append({'image': _image, 'question': _question, 'gt_answer': _gt_answer})
        print(f'\nQ: {_question}')
        print(f'  Model: {_pred_answer[:50]}...' if len(_pred_answer) > 50 else f'  Model: {_pred_answer}')
        print(f'  GT: {_gt_answer}')
    return before_answers, execution_step_9, test_data_for_comparison


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5단계: VQA 지시 미세 조정

    질문에 짧고 직접적으로 답하도록 모델을 학습합니다.
    """)
    return


@app.cell
def _(device, execution_step_9, os, output_dir, torch, tqdm):
    # 원본 코드 셀 순서와 모델·데이터 변경의 실행 순서를 보존합니다.
    execution_step_10 = execution_step_9 + 1
    def train_vlm_vqa(model, train_loader, num_epochs=8, lr=2e-4, checkpoint_path=str(output_dir / "vqa_checkpoint.pt")):
        """재개용 체크포인트를 지원하며 VLM을 VQA용으로 학습합니다.
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

    return execution_step_10, train_vlm_vqa


@app.cell
def _(execution_step_10, output_dir, train_loader, train_vlm_vqa, vlm):
    # 원본 코드 셀 순서와 모델·데이터 변경의 실행 순서를 보존합니다.
    execution_step_11 = execution_step_10 + 1
    # VQA 학습: 더 나은 수렴을 위해 에포크 수 확대
    # Ctrl+C로 학습을 중단한 뒤 다시 실행하면 재개 가능
    losses = train_vlm_vqa(vlm, train_loader, num_epochs=12, lr=2e-4, checkpoint_path=str(output_dir / "vqa_checkpoint.pt"))
    return execution_step_11, losses


@app.cell
def _(execution_step_11, losses, plt):
    # 원본 코드 셀 순서와 모델·데이터 변경의 실행 순서를 보존합니다.
    execution_step_12 = execution_step_11 + 1
    # 학습 손실 그래프
    plt.figure(figsize=(8, 4))
    plt.plot(range(1, len(losses)+1), losses, marker='o')
    plt.xlabel('에포크')
    plt.ylabel('손실')
    plt.title('VQA 지시 미세 조정 손실')
    plt.grid(True, alpha=0.3)
    plt.show()
    return (execution_step_12,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6단계: 학습 후 테스트 — 학습 전후 비교
    """)
    return


@app.cell
def _(
    before_answers,
    device,
    execution_step_12,
    generate_vqa_answer,
    image_processor,
    test_data_for_comparison,
    vlm,
):
    # 원본 코드 셀 순서와 모델·데이터 변경의 실행 순서를 보존합니다.
    execution_step_13 = execution_step_12 + 1
    # 같은 샘플을 학습 후 테스트
    print('=' * 70)
    print('VQA 지시 미세 조정 전후 비교')
    print('=' * 70)
    after_answers = []
    for _i, _data in enumerate(test_data_for_comparison):
        _pred_answer, _ = generate_vqa_answer(vlm, _data['image'], _data['question'], image_processor, device)
        after_answers.append(_pred_answer)
        print(f"\n{'=' * 70}")
        print(f'질문: {_data['question']}')
        print(f"{'=' * 70}")
        print(f'BEFORE: {before_answers[_i][:60]}...' if len(before_answers[_i]) > 60 else f'BEFORE: {before_answers[_i]}')
        print(f'학습 후: {_pred_answer}')
        print(f"GT:     {_data['gt_answer']}")
    return after_answers, execution_step_13


@app.cell
def _(
    after_answers,
    before_answers,
    execution_step_13,
    plt,
    test_data_for_comparison,
    textwrap,
):
    # 원본 코드 셀 순서와 모델·데이터 변경의 실행 순서를 보존합니다.
    execution_step_14 = execution_step_13 + 1
    # 시각적 비교

    def wrap_text(text, width=30):
        return '\n'.join(textwrap.wrap(str(text), width=width))
    _fig, _axes = plt.subplots(len(test_data_for_comparison), 1, figsize=(12, 4 * len(test_data_for_comparison)))
    for _i, (_data, before, after) in enumerate(zip(test_data_for_comparison, before_answers, after_answers)):
        _ax = _axes[_i]
        _ax.imshow(_data['image'])
        title = f'질문: {_data['question']}\n'
        title += f'BEFORE: {before[:40]}...\n' if len(before) > 40 else f'BEFORE: {before}\n'
        title += f'학습 후: {after}\n'
        title += f'정답: {_data['gt_answer']}'
        color = 'green' if after.lower().strip() == _data['gt_answer'].lower().strip() else 'black'
        _ax.set_title(title, fontsize=10, color=color)
        _ax.axis('off')
    plt.suptitle('VQA 학습 전후 비교', fontsize=14, fontweight='bold', y=1.01)  # 학습 후 결과가 정답과 같으면 초록색, 다르면 빨간색
    plt.tight_layout()
    plt.show()
    return (execution_step_14,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7단계: 더 많은 테스트 샘플로 평가하기
    """)
    return


@app.cell
def _(
    device,
    execution_step_14,
    generate_vqa_answer,
    get_most_common_answer,
    image_processor,
    test_samples,
    tqdm,
    vlm,
):
    # 원본 코드 셀 순서와 모델·데이터 변경의 실행 순서를 보존합니다.
    execution_step_15 = execution_step_14 + 1
    # 더 많은 테스트 샘플로 평가
    num_eval = min(50, len(test_samples))
    exact_correct = 0
    soft_correct = 0
    results = []
    print(f'평가할 샘플 수: {num_eval}개 테스트 샘플...')

    def soft_match(pred, gt):
        """예측에 정답이 포함되는지 확인합니다(예: green and white 안의 green).
        """
        pred_lower = pred.lower().strip()
        gt_lower = gt.lower().strip()
        if pred_lower == gt_lower:
            return (True, True)  # 완전 일치
        if pred_lower.startswith(gt_lower):
            return (False, True)
        if gt_lower in pred_lower:  # 예측의 시작에 정답이 있는지 확인
            return (False, True)
        return (False, False)
    for _i in tqdm(range(num_eval)):  # 예측에 정답이 포함되는지 확인
        _sample = test_samples[_i]
        _pred_answer, _ = generate_vqa_answer(vlm, _sample['image'], _sample['question'], image_processor, device)
        _gt_answer = get_most_common_answer(_sample['answers'])
        is_exact, is_soft = soft_match(_pred_answer, _gt_answer)
        if is_exact:
            exact_correct += 1
        if is_soft:
            soft_correct += 1
        results.append({'question': _sample['question'], 'predicted': _pred_answer, 'ground_truth': _gt_answer, 'exact': is_exact, 'soft': is_soft})
    exact_acc = exact_correct / num_eval * 100
    soft_acc = soft_correct / num_eval * 100
    print(f'\n완전 일치 정확도: {exact_acc:.1f}% ({exact_correct}/{num_eval})')
    print(f'포함 일치 정확도: {soft_acc:.1f}% ({soft_correct}/{num_eval})')
    return execution_step_15, results


@app.cell
def _(execution_step_15, results):
    # 원본 코드 셀 순서와 모델·데이터 변경의 실행 순서를 보존합니다.
    execution_step_16 = execution_step_15 + 1
    # 정답·오답 예시 표시
    print("\n" + "="*70)
    print('완전 일치 정답')
    print("="*70)
    for r in [x for x in results if x['exact']][:5]:
        print(f'질문: {r['question']}')
        print(f"A: {r['predicted']} (GT: {r['ground_truth']})")
        print()

    print("\n" + "="*70)
    print('포함 일치 정답(답변 포함)')
    print("="*70)
    for r in [x for x in results if x['soft'] and not x['exact']][:5]:
        print(f'질문: {r['question']}')
        print(f'예측: {r['predicted']} | GT: {r['ground_truth']}')
        print()

    print("\n" + "="*70)
    print('잘못된 예측')
    print("="*70)
    for r in [x for x in results if not x['soft']][:5]:
        print(f'질문: {r['question']}')
        print(f'예측: {r['predicted']} | GT: {r['ground_truth']}')
        print()
    return (execution_step_16,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 8단계: VQA 모델 저장하기
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
    # VQA 미세 조정 모델 저장
    save_dir = str(output_dir / "mini-vlm-vqa")
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
    }, f"{save_dir}/mini_vlm_vqa.pt")

    tokenizer.save_pretrained(f"{save_dir}/tokenizer")
    image_processor.save_pretrained(f"{save_dir}/image_processor")

    print(f'모델 저장 위치: {save_dir}/')
    print(f'저장 내용: {os.listdir(save_dir)}')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 정리

    시각적 질의응답을 위해 VLM을 지시 미세 조정했습니다.

    ### 수행한 작업

    1. **캡션을 학습한 VLM 불러오기**(객체 탐지 모델과는 작업이 다름)
    2. 학습 샘플 1,800개로 **VQAv2 데이터셋 사용하기**
    3. **질문·답변 형식 만들기**: `"Question: {q} Answer: {a}"`
    4. 짧고 직접적인 답변을 배우도록 **8에포크 미세 조정하기**
    5. 완전 일치 정확도로 **평가하기**

    ### 핵심 내용

    - **작업에 맞춘 조정**: 객체 탐지 → VQA 대신 캡션 모델 → VQA 모델 경로를 사용합니다.
    - **짧은 답변**: VQA는 보통 1~3단어 답변을 요구합니다.
    - **프롬프트 형식**: `"Question: ... Answer:"`는 작업 구조를 명확하게 합니다.
    - **레이블 마스킹**: 답변 부분만 학습합니다.

    ### 모델 계보

    ```
    기반 모델(ViT + SmolLM)
        │
        └── 캡션 학습(Flickr8k)
                │
                ├── 객체 탐지 지시 학습 → mini-vlm-od/
                │
                └── VQA 지시 학습 → mini-vlm-vqa/  ← 이 노트북
    ```

    ### 한계

    - 소형 언어 모델(135M LLM)로 인한 추론 능력의 한계
    - 엄격한 완전 일치 정확도("2"와 "two"도 오답 처리)
    - 제한된 학습 데이터(1,800개 샘플)
    - 복잡한 추론이나 개수 세기 능력 부족

    ### 다음 단계

    1. **멀티태스크 모델**: 캡션, 객체 탐지, VQA를 하나의 모델로 통합하기
    2. **더 나은 평가**: 답변 표현의 차이를 고려하는 VQA 정확도 지표 사용하기
    3. **더 큰 모델**: 추론 능력을 높이기 위해 SmolLM-360M이나 1.7B 사용하기
    4. **사고 과정**: 최종 답변 전에 추론 과정을 추가하기
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 참고 자료

    - [처음부터 만드는 최소 구성의 VLM](part-01-minimal-vlm.py)
    - [객체 탐지를 위한 VLM 지시 미세 조정](part-02-object-detection.py)
    - [VQAv2 데이터셋](https://visualqa.org/)
    - [Hugging Face의 VQAv2](https://huggingface.co/datasets/lmms-lab/VQAv2)
    - [LLaVA: 시각 지시 학습](https://llava-vl.github.io/)
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
