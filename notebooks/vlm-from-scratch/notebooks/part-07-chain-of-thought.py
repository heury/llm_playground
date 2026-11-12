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


@app.cell
def _(execution_step_0):
    # 원본 코드 셀 순서와 모델·데이터 변경의 실행 순서를 보존합니다.
    execution_step_1 = execution_step_0 + 1
    # 체크포인트를 삭제하고 처음부터 시작하려면 아래 주석 해제
    # import os
    # if (output_dir / "cot_training_checkpoint.pt").exists():
    #     (output_dir / "cot_training_checkpoint.pt").unlink()
    #     print("체크포인트를 삭제했습니다. 처음부터 학습합니다.")
    return (execution_step_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # VLM의 사고 과정 기반 시각적 추론

    Nipun Batra · 2025-12-30
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 소개

    이 시리즈에서는 이미지 캡션, 객체 탐지, 질의응답, 멀티태스크 처리가 가능한 모델을 만들었습니다. 그렇다면 모델은 본 내용을 **추론**할 수도 있을까요?

    이전 노트북:
    1. [최소 구성의 VLM](part-01-minimal-vlm.py): 기본 캡션 생성
    2. [객체 탐지](part-02-object-detection.py): 구조화된 출력
    3. [VQA](part-03-visual-qa.py): 질의응답
    4. [멀티태스크 VLM](part-04-multi-task.py): 통합 모델
    5. [다중 이미지 VLM](part-05-multi-image.py): 시간적 추론
    6. [태스크 라우팅](part-06-task-routing.py): 자동 작업 감지

    이번에는 사고 과정(Chain-of-Thought, CoT) 추론으로 VLM이 **단계별로 생각하도록** 학습합니다.

    ### 사고 과정 추론이란?

    질문에 바로 답하는 대신 먼저 추론 과정을 설명합니다.

    **CoT 없이 답변:**
    ```
    질문: 빨간 객체와 파란 객체 중 어느 쪽이 더 많나요?
    답변: 빨간 객체
    ```

    **CoT를 사용한 답변:**
    ```
    질문: 빨간 객체와 파란 객체 중 어느 쪽이 더 많나요?
    답변: 단계별로 세어 보겠습니다.
        1. 빨간 객체는 공, 정육면체, 원기둥으로 3개입니다.
        2. 파란 객체는 구와 작은 정육면체로 2개입니다.
        3. 빨간 객체 3개 > 파란 객체 2개
        정답: 빨간 객체가 더 많습니다.
    ```

    ### 사고 과정을 사용하는 이유

    **장점:**
    - **정확도 향상**: 복잡한 문제를 단계로 나눕니다.
    - **해석 가능성**: 모델의 사고 과정을 볼 수 있습니다.
    - **디버깅**: 추론이 실패한 지점을 찾을 수 있습니다.
    - **신뢰성**: 답변을 받아들이기 전에 논리를 검토할 수 있습니다.

    **연구 배경:**
    - GPT-4V는 복잡한 시각적 추론에 CoT를 사용합니다.
    - Gemini는 추론 과정을 보여 줍니다.
    - 원문에서 소개한 최근 논문들은 CoT로 정확도가 20~30% 향상된다고 보고합니다.

    ### 데이터셋: A-OKVQA(Augmented OK-VQA)

    다음을 포함하는 **A-OKVQA**를 사용합니다.
    - 복잡한 추론 질문
    - 추론 근거(사고 과정)
    - 객관식 답변
    - COCO의 실제 이미지

    예시(학습 데이터의 영어 형식 유지):
    ```json
    {
      "question": "Why is the person holding an umbrella?",
      "rationales": [
        "It is raining",
        "The ground is wet and the person is protecting themselves from rain"
      ],
      "answer": "It is raining"
    }
    ```

    이 예시는 사람이 우산을 든 이유를 묻고, 젖은 땅과 비를 피하는 모습을 근거로 "비가 오기 때문"이라고 답합니다.

    ### 구현할 내용

    1. 추론 근거가 포함된 A-OKVQA 데이터셋 불러오기
    2. 사고 과정을 생성하도록 VLM 학습하기
    3. 추론 품질 평가하기
    4. CoT 방식과 직접 답변 방식 비교하기
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 환경 설정
    """)
    return


@app.cell
def _(execution_step_1):
    # 원본 코드 셀 순서와 모델·데이터 변경의 실행 순서를 보존합니다.
    execution_step_2 = execution_step_1 + 1
    import os

    return execution_step_2, os


@app.cell
def _(execution_step_2):
    # 원본 코드 셀 순서와 모델·데이터 변경의 실행 순서를 보존합니다.
    execution_step_3 = execution_step_2 + 1
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer, ViTModel, ViTImageProcessor
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
    import warnings
    warnings.filterwarnings('ignore')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'사용 장치: {device}')
    print(f'GPU 장치: {(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')}')
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
        execution_step_3,
        load_dataset,
        nn,
        plt,
        random,
        torch,
        tqdm,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1단계: 기반 VLM 불러오기

    이전 노트북에서 학습한 멀티태스크 VLM에서 시작합니다.
    """)
    return


@app.cell
def _(
    AutoModelForCausalLM,
    AutoTokenizer,
    F,
    ViTModel,
    execution_step_3,
    nn,
    torch,
):
    # 원본 코드 셀 순서와 모델·데이터 변경의 실행 순서를 보존합니다.
    execution_step_4 = execution_step_3 + 1
    # VLM 구조(앞과 동일)

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
            max_new_tokens: int = 100,
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

    return MiniVLM, VisionProjector, execution_step_4


@app.cell
def _(
    AutoModelForCausalLM,
    AutoTokenizer,
    MiniVLM,
    ViTImageProcessor,
    ViTModel,
    VisionProjector,
    device,
    execution_step_4,
    os,
    output_dir,
    torch,
):
    # 원본 코드 셀 순서와 모델·데이터 변경의 실행 순서를 보존합니다.
    execution_step_5 = execution_step_4 + 1
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
    return (execution_step_5, image_processor, language_dim, lm_model_name, tokenizer, vision_dim, vision_model_name, vlm)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2단계: A-OKVQA 데이터셋 불러오기

    A-OKVQA(Augmented OK-VQA)는 추론이 필요한 질문과 답변을 설명하는 근거를 포함합니다.
    """)
    return


@app.cell
def _(execution_step_5, load_dataset):
    # 원본 코드 셀 순서와 모델·데이터 변경의 실행 순서를 보존합니다.
    execution_step_6 = execution_step_5 + 1
    # A-OKVQA 데이터셋 로딩
    print('A-OKVQA 데이터셋 로딩 중...')
    try:
        aokvqa_dataset = load_dataset('HuggingFaceM4/A-OKVQA', split='train', streaming=True)
        aokvqa_samples = []  # Hugging Face의 A-OKVQA
        for _i, _sample in enumerate(aokvqa_dataset):
            if _i >= 2000:
                break  # 샘플 수집
            aokvqa_samples.append(_sample)
        print(f'불러온 수: {len(aokvqa_samples)}개 A-OKVQA 샘플')
    except Exception as e:  # 학습에 샘플 2,000개 사용
        print(f'A-OKVQA 로딩 실패: {e}')
        print('합성 추론 근거를 사용하는 VQAv2로 전환합니다...')
        vqa_dataset = load_dataset('lmms-lab/VQAv2', split='train', streaming=True)
        aokvqa_samples = []
        for _i, _sample in enumerate(vqa_dataset):
            if _i >= 2000:
                break
            aokvqa_samples.append(_sample)
        print(f'불러온 수: {len(aokvqa_samples)}개 VQAv2 샘플(합성 추론 근거 추가 예정)')  # 대체 경로: VQAv2를 사용하고 합성 추론 근거 생성
    return aokvqa_samples, execution_step_6


@app.cell
def _(aokvqa_samples, execution_step_6, plt):
    # 원본 코드 셀 순서와 모델·데이터 변경의 실행 순서를 보존합니다.
    execution_step_7 = execution_step_6 + 1
    # 샘플 확인
    if aokvqa_samples:
        _sample = aokvqa_samples[0]
        print('샘플 키:', _sample.keys())
        print('\n샘플:')
        if 'question' in _sample:
            print(f'  질문: {_sample['question']}')
        if 'rationales' in _sample:
            print(f'  추론 근거: {_sample['rationales'][:2]}...')
        if 'choices' in _sample:
            print(f'  선택지: {_sample['choices']}')  # 처음 2개
        if 'correct_choice_idx' in _sample:
            print(f'  정답 선택지: {_sample['correct_choice_idx']}')
        if 'image' in _sample:
            plt.figure(figsize=(6, 6))
            plt.imshow(_sample['image'])
            plt.title(f'질문: {_sample.get('question', 'N/A')[:50]}...')
            plt.axis('off')
            plt.show()  # 이미지 표시
    return (execution_step_7,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3단계: 사고 과정 데이터셋 만들기

    모델이 단계별로 추론하도록 다음 형식으로 데이터를 구성합니다.

    **형식:**
    ```
    Question: {question}
    Let me think step by step:
    {rationale}
    Answer: {answer}
    ```

    질문 뒤에 추론 근거를 쓰고 마지막에 정답을 제시합니다.
    """)
    return


@app.cell
def _(execution_step_7, random):
    # 원본 코드 셀 순서와 모델·데이터 변경의 실행 순서를 보존합니다.
    execution_step_8 = execution_step_7 + 1
    # 보조 함수

    def get_most_common_answer(answers):
        """VQA 주석에서 가장 빈도가 높은 답변을 구합니다.
        """
        if isinstance(answers, list) and len(answers) > 0:
            if isinstance(answers[0], dict) and 'answer' in answers[0]:
                answer_counts = {}
                for ans in answers:
                    a = ans['answer']
                    answer_counts[a] = answer_counts.get(a, 0) + 1
                return max(answer_counts, key=answer_counts.get)
            elif isinstance(answers[0], str):
                return answers[0]
        return str(answers)


    def create_synthetic_rationale(question, answer):
        """대체 경로에서 VQAv2 샘플의 간단한 추론 근거를 생성합니다.
        """
        # 간단한 템플릿 기반 추론 근거
        templates = [
            f"Looking at the image, I can see that {answer}.",
            f"Based on the visual information, the answer is {answer}.",
            f"Examining the image carefully, {answer} is the correct answer.",
            f"I observe that {answer}.",
        ]
        return random.choice(templates)


    def format_cot_prompt(question, rationale, answer):
        """질문·근거·답변을 CoT 프롬프트로 구성합니다.
        """
        prompt = f"Question: {question}\nLet me think step by step:\n"
        response = f"{rationale}\nAnswer: {answer}"
        return prompt, response


    print('보조 함수를 정의했습니다.')
    return (
        create_synthetic_rationale,
        execution_step_8,
        format_cot_prompt,
        get_most_common_answer,
    )


@app.cell
def _(
    DataLoader,
    Dataset,
    aokvqa_samples,
    create_synthetic_rationale,
    execution_step_8,
    format_cot_prompt,
    get_most_common_answer,
    image_processor,
    random,
    tokenizer,
):
    # 원본 코드 셀 순서와 모델·데이터 변경의 실행 순서를 보존합니다.
    execution_step_9 = execution_step_8 + 1
    # 사고 과정 데이터셋

    class CoTReasoningDataset(Dataset):
        """사고 과정 기반 시각적 추론 데이터셋입니다.
        """
    
        def __init__(self, samples, image_processor, tokenizer, max_length=256):
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
            question = sample.get('question', '')
        
            # 답변 추출
            if 'choices' in sample and 'correct_choice_idx' in sample:
                # A-OKVQA 형식
                answer = sample['choices'][sample['correct_choice_idx']]
            elif 'answers' in sample:
                # VQAv2 형식
                answer = get_most_common_answer(sample['answers'])
            else:
                answer = sample.get('answer', 'unknown')
        
            # 추론 근거 추출
            if 'rationales' in sample and sample['rationales']:
                # A-OKVQA에는 추론 근거가 있음
                rationale = random.choice(sample['rationales'])
            else:
                # VQAv2용 합성 추론 근거 생성
                rationale = create_synthetic_rationale(question, answer)
        
            # CoT 형식으로 구성
            prompt, response = format_cot_prompt(question, rationale, answer)
            full_text = f"{prompt}{response}{self.tokenizer.eos_token}"
        
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
        
            # 프롬프트 마스킹(추론과 답변만 학습)
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
    cot_dataset = CoTReasoningDataset(aokvqa_samples, image_processor, tokenizer)

    cot_loader = DataLoader(
        cot_dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0,
    )

    print(f'CoT 데이터셋: {len(cot_dataset)}개 샘플')
    print(f'배치 수: {len(cot_loader)}')
    return cot_dataset, cot_loader, execution_step_9


@app.cell
def _(cot_dataset, execution_step_9, tokenizer):
    # 원본 코드 셀 순서와 모델·데이터 변경의 실행 순서를 보존합니다.
    execution_step_10 = execution_step_9 + 1
    # CoT 형식 확인
    sample_item = cot_dataset[0]
    decoded = tokenizer.decode(sample_item['input_ids'], skip_special_tokens=True)
    print('CoT 형식 샘플:')
    print(decoded[:300])
    print("...")
    return (execution_step_10,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4단계: 사고 과정 학습

    답변하기 전에 사고 과정을 생성하도록 VLM을 미세 조정합니다.
    """)
    return


@app.cell
def _(device, execution_step_10, os, output_dir, torch, tqdm):
    # 원본 코드 셀 순서와 모델·데이터 변경의 실행 순서를 보존합니다.
    execution_step_11 = execution_step_10 + 1
    def train_cot_vlm(model, train_loader, num_epochs=6, lr=1e-4, checkpoint_path=str(output_dir / "training_checkpoint.pt")):
        """사고 과정 추론으로 VLM을 학습합니다.
        
        기능:
        - 에포크마다 체크포인트 저장
        - 중단한 지점에서 재개 가능
        - 중단되더라도 손실 목록 반환
        """
    
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable_params, lr=lr)
    
        # 기존 체크포인트 로딩 시도
        start_epoch = 0
        losses = []
    
        if os.path.exists(checkpoint_path):
            print(f'체크포인트 발견 위치: {checkpoint_path}')
            try:
                checkpoint = torch.load(checkpoint_path, map_location='cpu')
                start_epoch = checkpoint.get('epoch', 0)
                losses = checkpoint.get('losses', [])
                model.projector.load_state_dict(checkpoint['projector_state_dict'])
                model.language_model.load_state_dict(checkpoint['language_model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                print(f'다음 에포크부터 재개: {start_epoch}, 지금까지의 손실: {losses}')
            except Exception as e:
                print(f'체크포인트 로딩 실패: {e}')
                print('처음부터 학습합니다...')
    
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
            print(f'지금까지의 손실: {losses}')
            print(f'체크포인트 저장 위치: {checkpoint_path}')
            print('이 셀을 다시 실행하면 학습을 재개할 수 있습니다.')
            print("="*70)
    
        return losses

    return execution_step_11, train_cot_vlm


@app.cell
def _(cot_loader, execution_step_11, output_dir, train_cot_vlm, vlm):
    # 원본 코드 셀 순서와 모델·데이터 변경의 실행 순서를 보존합니다.
    execution_step_12 = execution_step_11 + 1
    # CoT 모델 학습
    print('사고 과정 VLM 학습 중...\n')
    print('기능:')
    print('  - 에포크마다 체크포인트 저장')
    print('  - 중단 후 재개 가능(이 셀을 다시 실행)')
    print('  - 중단해도 손실 목록 보존\n')

    losses = train_cot_vlm(vlm, cot_loader, num_epochs=8, lr=1e-4, checkpoint_path=str(output_dir / "cot_training_checkpoint.pt"))
    return execution_step_12, losses


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 선택 사항: 처음부터 학습하려면 체크포인트 정리

    체크포인트에서 재개하는 대신 처음부터 학습하려면 먼저 체크포인트 삭제용 주석 코드를 확인하세요.
    """)
    return


@app.cell
def _(execution_step_12, losses, plt):
    # 원본 코드 셀 순서와 모델·데이터 변경의 실행 순서를 보존합니다.
    execution_step_13 = execution_step_12 + 1
    # 학습 손실 그래프
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(losses)+1), losses, marker='o', linewidth=2, markersize=8)
    plt.xlabel('에포크', fontsize=12)
    plt.ylabel('손실', fontsize=12)
    plt.title('사고 과정 학습 손실', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    return (execution_step_13,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5단계: 사고 과정 추론 테스트

    모델이 사고 과정을 생성하는지 확인합니다.
    """)
    return


@app.cell
def _(Image, execution_step_13):
    # 원본 코드 셀 순서와 모델·데이터 변경의 실행 순서를 보존합니다.
    execution_step_14 = execution_step_13 + 1
    def generate_cot_answer(model, image, question, image_processor, device, max_tokens=150):
        """CoT 추론과 답변을 생성합니다.
        """
        model.eval()
    
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        image = image.convert('RGB')
    
        pixel_values = image_processor(image, return_tensors="pt").pixel_values.to(device)
    
        # CoT 프롬프트
        prompt = f"Question: {question}\nLet me think step by step:\n"
    
        response = model.generate(
            pixel_values,
            prompt=prompt,
            max_new_tokens=max_tokens,
            temperature=0.7,
            do_sample=True,
        )
    
        # 추론과 답변 추출
        if "Let me think step by step:" in response:
            reasoning_part = response.split("Let me think step by step:")[-1].strip()
        else:
            reasoning_part = response
    
        return reasoning_part

    return execution_step_14, generate_cot_answer


@app.cell
def _(
    aokvqa_samples,
    device,
    execution_step_14,
    generate_cot_answer,
    image_processor,
    vlm,
):
    # 원본 코드 셀 순서와 모델·데이터 변경의 실행 순서를 보존합니다.
    execution_step_15 = execution_step_14 + 1
    # 학습 샘플로 테스트
    print('=' * 70)
    print('사고 과정 추론 예시')
    print('=' * 70)
    test_indices = [0, 10, 50, 100, 200]
    for _idx in test_indices:
        if _idx >= len(aokvqa_samples):
            continue
        _sample = aokvqa_samples[_idx]
        _image = _sample['image']
        _question = _sample.get('question', '')
        cot_response = generate_cot_answer(vlm, _image, _question, image_processor, device)
        print(f"\n{'=' * 70}")
        print(f'예시 {_idx}')
        print(f'질문: {_question}')
        print(f'\n모델의 사고 과정:')  # CoT 응답 생성
        print(cot_response)
        if 'rationales' in _sample and _sample['rationales']:
            print(f'\n정답 추론 근거(여러 개 중 하나):')
            print(_sample['rationales'][0])  # 정답이 있으면 표시
    return (execution_step_15,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6단계: 시각적 비교 — CoT와 직접 답변

    같은 이미지에 대해 CoT를 사용한 경우와 사용하지 않은 경우를 비교합니다.
    """)
    return


@app.cell
def _(Image, execution_step_15):
    # 원본 코드 셀 순서와 모델·데이터 변경의 실행 순서를 보존합니다.
    execution_step_16 = execution_step_15 + 1
    def generate_direct_answer(model, image, question, image_processor, device):
        """추론 과정 없이 직접 답변을 생성합니다.
        """
        model.eval()
    
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        image = image.convert('RGB')
    
        pixel_values = image_processor(image, return_tensors="pt").pixel_values.to(device)
    
        # 직접 답변 프롬프트(CoT 없음)
        prompt = f"Question: {question} Answer:"
    
        response = model.generate(
            pixel_values,
            prompt=prompt,
            max_new_tokens=20,  # 직접 답변은 더 짧게 생성
            temperature=0.7,
            do_sample=True,
        )
    
        if "Answer:" in response:
            answer = response.split("Answer:")[-1].strip()
        else:
            answer = response
    
        return answer

    return execution_step_16, generate_direct_answer


@app.cell
def _(
    aokvqa_samples,
    device,
    execution_step_16,
    generate_cot_answer,
    generate_direct_answer,
    image_processor,
    plt,
    vlm,
):
    # 원본 코드 셀 순서와 모델·데이터 변경의 실행 순서를 보존합니다.
    execution_step_17 = execution_step_16 + 1
    # 나란히 비교
    import textwrap

    def wrap_text(text, width=50):
        return '\n'.join(textwrap.wrap(text, width=width))
    comparison_indices = [5, 15, 25]
    _fig, axes = plt.subplots(len(comparison_indices), 1, figsize=(14, 6 * len(comparison_indices)))
    if len(comparison_indices) == 1:
        axes = [axes]
    for _i, _idx in enumerate(comparison_indices):
        if _idx >= len(aokvqa_samples):
            continue
        _sample = aokvqa_samples[_idx]
        _image = _sample['image']
        _question = _sample.get('question', '')
        direct = generate_direct_answer(vlm, _image, _question, image_processor, device)
        cot = generate_cot_answer(vlm, _image, _question, image_processor, device, max_tokens=100)
        axes[_i].imshow(_image)
        title_text = f'질문: {_question}\n\n'
        title_text += f'직접 답변: {direct[:80]}...\n\n'
        title_text += f'사고 과정: {cot[:150]}...'  # 두 방식 모두 생성
        axes[_i].set_title(wrap_text(title_text, 90), fontsize=10, loc='left')
        axes[_i].axis('off')
    plt.suptitle('사고 과정과 직접 답변 비교', fontsize=16, fontweight='bold', y=1.01)
    plt.tight_layout()  # 표시
    plt.show()
    return (execution_step_17,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7단계: 정량 평가

    CoT가 정확도에 미치는 영향을 측정합니다.
    """)
    return


@app.cell
def _(
    aokvqa_samples,
    device,
    execution_step_17,
    generate_cot_answer,
    generate_direct_answer,
    get_most_common_answer,
    image_processor,
    random,
    tqdm,
    vlm,
):
    # 원본 코드 셀 순서와 모델·데이터 변경의 실행 순서를 보존합니다.
    execution_step_18 = execution_step_17 + 1
    # 테스트 세트에서 간단히 평가
    def evaluate_cot(model, samples, image_processor, device, num_samples=50):
        """CoT 추론 품질을 평가합니다.
        """
    
        results = {
            'cot': {'correct': 0, 'total': 0, 'has_reasoning': 0},
            'direct': {'correct': 0, 'total': 0}
        }
    
        eval_samples = random.sample(samples, min(num_samples, len(samples)))
    
        for sample in tqdm(eval_samples, desc='평가 중'):
            image = sample['image']
            question = sample.get('question', '')
        
            # 정답 추출
            if 'choices' in sample and 'correct_choice_idx' in sample:
                gt_answer = sample['choices'][sample['correct_choice_idx']].lower()
            elif 'answers' in sample:
                gt_answer = get_most_common_answer(sample['answers']).lower()
            else:
                continue
        
            # CoT 평가
            cot_response = generate_cot_answer(model, image, question, image_processor, device, max_tokens=100)
        
            # CoT에서 최종 답변 추출
            if "Answer:" in cot_response:
                cot_answer = cot_response.split("Answer:")[-1].strip().lower()
            else:
                cot_answer = cot_response.split()[-1].lower() if cot_response else ""
        
            # 단순 직접 답변이 아닌 추론이 있는지 확인
            has_reasoning = len(cot_response.split()) > 10
            if has_reasoning:
                results['cot']['has_reasoning'] += 1
        
            # 정답 여부 확인(유연한 일치)
            if gt_answer in cot_answer or cot_answer in gt_answer:
                results['cot']['correct'] += 1
            results['cot']['total'] += 1
        
            # 직접 답변 평가
            direct_answer = generate_direct_answer(model, image, question, image_processor, device).lower()
        
            if gt_answer in direct_answer or direct_answer in gt_answer:
                results['direct']['correct'] += 1
            results['direct']['total'] += 1
    
        return results


    # 평가
    print('CoT와 직접 답변 평가 중...\n')
    eval_results = evaluate_cot(vlm, aokvqa_samples, image_processor, device, num_samples=100)

    print("\n" + "="*70)
    print('평가 결과')
    print("="*70)

    cot_acc = eval_results['cot']['correct'] / eval_results['cot']['total'] * 100
    direct_acc = eval_results['direct']['correct'] / eval_results['direct']['total'] * 100
    reasoning_pct = eval_results['cot']['has_reasoning'] / eval_results['cot']['total'] * 100

    print(f'\n사고 과정:')
    print(f'  정확도: {cot_acc:.1f}% ({eval_results['cot']['correct']}/{eval_results['cot']['total']})')
    print(f'  추론 생성 비율: {reasoning_pct:.1f}%의 응답')

    print(f'\n직접 답변:')
    print(f'  정확도: {direct_acc:.1f}% ({eval_results['direct']['correct']}/{eval_results['direct']['total']})')

    improvement = cot_acc - direct_acc
    print(f'\nCoT 사용 시 개선: {improvement:+.1f}%')

    if improvement > 0:
        print('✓ 사고 과정이 도움이 됩니다.')
    else:
        print('참고: CoT에 추가 학습이나 더 나은 프롬프트가 필요할 수 있습니다.')
    return cot_acc, direct_acc, execution_step_18, reasoning_pct


@app.cell
def _(cot_acc, direct_acc, execution_step_18, plt, reasoning_pct):
    # 원본 코드 셀 순서와 모델·데이터 변경의 실행 순서를 보존합니다.
    execution_step_19 = execution_step_18 + 1
    # 결과 시각화
    _fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    methods = ['직접 답변', '사고 과정']
    # 정확도 비교
    accuracies = [direct_acc, cot_acc]
    colors = ['skyblue', 'lightcoral']
    bars = ax1.bar(methods, accuracies, color=colors, edgecolor='black', linewidth=1.5)
    ax1.set_ylabel('정확도(%)', fontsize=12)
    ax1.set_title('정확도: 직접 답변과 사고 과정', fontsize=14, fontweight='bold')
    ax1.set_ylim(0, 100)
    ax1.grid(axis='y', alpha=0.3)
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2.0, height, f'{acc:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
    # 막대에 값 표시
    labels = ['추론 있음', '추론 없음']
    sizes = [reasoning_pct, 100 - reasoning_pct]
    colors2 = ['lightgreen', 'lightgray']
    ax2.pie(sizes, labels=labels, colors=colors2, autopct='%1.1f%%', startangle=90, textprops={'fontsize': 11})
    ax2.set_title('CoT 응답 특성', fontsize=14, fontweight='bold')
    # CoT 특성
    plt.tight_layout()
    plt.show()
    return (execution_step_19,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 8단계: CoT 모델 저장하기
    """)
    return


@app.cell
def _(
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
    # CoT 학습 모델 저장
    save_dir = str(output_dir / "mini-vlm-cot")
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
    }, f"{save_dir}/mini_vlm_cot.pt")

    tokenizer.save_pretrained(f"{save_dir}/tokenizer")
    image_processor.save_pretrained(f"{save_dir}/image_processor")

    print(f'CoT 모델 저장 위치: {save_dir}/')
    print(f'저장 내용: {os.listdir(save_dir)}')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 정리

    **사고 과정 기반 시각적 추론 VLM**을 만들었습니다.

    ### 구현한 내용

    1. **CoT 데이터셋**: 추론 근거가 있는 질문(A-OKVQA 또는 VQAv2 + 합성 근거)
    2. **CoT 학습 형식**: "Let me think step by step:" 프롬프트
    3. **추론 생성**: 단계별 설명 출력
    4. **평가**: CoT와 직접 답변 비교

    ### 핵심 내용

    **사고 과정의 장점:**
    - **해석 가능성**: 모델의 추론 과정 확인
    - **디버깅**: 논리적 오류 식별
    - **정확도**: 복잡한 추론 작업에서 향상 가능
    - **신뢰성**: 답변을 받아들이기 전에 추론 검토

    **학습 형식:**
    ```
    Question: {question}
    Let me think step by step:
    {step 1}
    {step 2}
    {step 3}
    Answer: {final_answer}
    ```

    **프롬프트 설계:**
    - "Let me think step by step"은 추론 모드를 유도합니다.
    - "First,", "Then,", "Finally," 같은 순서 표식도 사용할 수 있습니다.
    - "Answer:"는 최종 답변의 시작을 나타냅니다.

    ### 상용 모델과의 비교

    아래 수치는 원문에 제시된 비교를 옮긴 것이며 이 번역 작업에서 재측정한 결과는 아닙니다.

    | 특징 | 이 실습 모델 | GPT-4V | Gemini |
    |---------|-----------|---------|--------|
    | **CoT 추론** | ✓ | ✓ | ✓ |
    | **명시적 단계** | ✓ | ✓ | ✓ |
    | **규모** | 135M | 1T+ | 1T+ |
    | **정확도** | 약 40~50% | 약 80~90% | 약 80~90% |
    | **학습 데이터** | 2천 개 샘플 | 수백만 개 | 수백만 개 |

    ### 한계

    - **소형 모델**: 매개변수 135M으로 추론 깊이 제한
    - **적은 학습 데이터**: 상용 모델의 수백만 개에 비해 적은 2,000개 샘플
    - **단순한 근거**: 사람의 추론만큼 상세하지 않음
    - **자기 수정 없음**: 잘못된 추론을 고치지 않음
    - **합성 근거**: VQAv2 대체 경로에서 템플릿 사용

    ### 연구 배경(2024~2025년)

    **원문에서 소개한 발전 방향:**
    - **자기 일관성(Self-consistency)**: 여러 CoT 경로를 생성하고 답변에 투표
    - **쉬운 문제부터 풀기(Least-to-most prompting)**: 하위 문제로 분해
    - **사고 트리(Tree-of-thoughts)**: 여러 추론 분기 탐색
    - **자기 개선(Self-refinement)**: 자신의 추론을 비평하고 개선

    **주요 논문:**
    - Chain-of-Thought Prompting(Wei 외, 2022)
    - Self-Consistency(Wang 외, 2023)
    - Visual Chain-of-Thought(Zhang 외, 2024)

    ### 다음 단계

    1. **여러 단계 검증**: 자기 점검 메커니즘 추가
    2. **도구 사용**: 추론 중 계산기·검색 같은 외부 도구 호출
    3. **사고 트리**: 여러 추론 경로 탐색
    4. **더 큰 모델**: SmolLM-1.7B나 Qwen2-VL로 확장
    5. **더 나은 데이터셋**: CLEVR, ViQuAE, ScienceQA 사용
    6. **태스크 라우팅과 결합**: CoT 사용 시점 자동 감지
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 참고 자료

    ### 데이터셋
    - [A-OKVQA: Augmented OK-VQA](https://arxiv.org/abs/2206.01718): 근거가 포함된 질문
    - [VQAv2](https://visualqa.org/): 시각적 질의응답
    - [CLEVR](https://cs.stanford.edu/people/jcjohns/clevr/): 조합적 추론

    ### 사고 과정 연구
    - [Chain-of-Thought Prompting(Wei 외, 2022)](https://arxiv.org/abs/2201.11903): 최초 CoT 논문
    - [Self-Consistency(Wang 외, 2023)](https://arxiv.org/abs/2203.11171): 여러 추론 경로
    - [Least-to-Most Prompting](https://arxiv.org/abs/2205.10625): 문제 분해
    - [Tree of Thoughts(Yao 외, 2023)](https://arxiv.org/abs/2305.10601): 추론 트리 탐색

    ### 시각적 추론
    - [Visual Chain-of-Thought](https://arxiv.org/abs/2305.02317): 시각 작업을 위한 CoT
    - [Multimodal CoT(Zhang 외, 2023)](https://arxiv.org/abs/2302.00923): 시각·언어 추론

    ### 이전 노트북
    - [최소 구성의 VLM](part-01-minimal-vlm.py)
    - [VQA 지시 미세 조정](part-03-visual-qa.py)
    - [멀티태스크 VLM](part-04-multi-task.py)
    - [태스크 라우팅](part-06-task-routing.py)
    """)
    return


if __name__ == "__main__":
    app.run()
