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
    # VLM 압축: 양자화, 가지치기, 지식 증류

    Nipun Batra · 2026-01-02
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # VLM 압축: 모델을 더 작고 빠르게 만들기

    이 노트북에서는 성능 손실을 줄이면서 시각·언어 모델(VLM)의 효율을 높이는 **모델 압축 기법**을 살펴봅니다. 다음 세 기법을 시연합니다.

    1. **양자화**: FP32 가중치를 INT8로 변환 → **양자화 대상 가중치의 저장량을 약 1/4로 축소**
    2. **구조적 가지치기**: 어텐션 헤드나 FFN 차원 전체 제거 → **속도 2배 향상**
    3. **지식 증류**: 작은 학생 모델이 큰 교사 모델을 모방하도록 학습 → **처음부터 학습할 때보다 높은 정확도**

    ## VLM을 압축하는 이유

    - **배포 제약**: 모바일·엣지 장치의 제한된 메모리
    - **비용 절감**: 추론 비용 감소와 처리 속도 향상
    - **에너지 효율**: 연산량 감소로 소비 전력 절감
    - **실시간 애플리케이션**: 상호작용 시 응답 시간 단축

    ## 기준 모델

    **이미지 캡션 VLM**을 기준 모델로 사용합니다.
    - 비전 인코더: ViT-Base(매개변수 85M)
    - 언어 모델: SmolLM-135M(매개변수 135M)
    - 프로젝터: 선형 층(매개변수 2M)
    - **합계**: 별도로 추가한 LM 헤드까지 포함한 실제 매개변수 수와 저장 크기는 아래 셀에서 계산합니다.

    ## 목표 압축 결과

    다음은 원문의 이론적 예시 목표입니다. 실제 기준 모델의 크기나 이 실습의 측정 결과와는 구분합니다.

    | 기법 | 모델 크기 | 추론 속도 | 정확도 하락 |
    |-----------|------------|-----------------|---------------|
    | 기준 모델(FP32) | 888 MB | 1.0배 | 0% |
    | INT8 양자화 | 222 MB | 1.5~2배 | 2% 미만 |
    | 가지치기(50%) | 444 MB | 2배 | 5% 미만 |
    | 증류(소형 모델) | 300 MB | 2.5배 | 3% 미만 |
    | 결합 | 150 MB | 3~4배 | 7% 미만 |

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
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoModel, AutoTokenizer, AutoImageProcessor
    from datasets import load_dataset
    from PIL import Image
    import numpy as np
    from tqdm import tqdm
    import time
    import gc
    import copy

    print(f'PyTorch 버전: {torch.__version__}')
    print(f'CUDA 사용 가능 여부: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'CUDA 장치: {torch.cuda.get_device_name(0)}')
        print(f'GPU 메모리: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.2f} GB')
    return (
        AutoImageProcessor,
        AutoModel,
        AutoTokenizer,
        DataLoader,
        Dataset,
        F,
        copy,
        execution_step_2,
        load_dataset,
        nn,
        time,
        torch,
        tqdm,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 사전 학습된 VLM 불러오기(기준 모델)

    사전 학습된 이미지 캡션 모델을 기준 모델로 불러옵니다. 모델이 없다면 간단한 버전을 학습하는 방식으로 진행할 수 있습니다.
    """)
    return


@app.cell
def _(AutoModel, execution_step_2, nn, torch):
    # 원본 코드 셀 순서와 모델·데이터 변경의 실행 순서를 보존합니다.
    execution_step_3 = execution_step_2 + 1
    class SimpleVLM(nn.Module):
        """이미지 캡션용 간단한 시각·언어 모델입니다.
        """
        def __init__(self, vision_model_name="google/vit-base-patch16-224", 
                     language_model_name="HuggingFaceTB/SmolLM-135M"):
            super().__init__()
        
            # 비전 인코더
            self.vision_encoder = AutoModel.from_pretrained(vision_model_name)
            self.vision_hidden_size = self.vision_encoder.config.hidden_size  # ViT-Base의 차원은 768
        
            # 언어 모델
            self.language_model = AutoModel.from_pretrained(language_model_name)
            self.language_hidden_size = self.language_model.config.hidden_size  # SmolLM의 차원은 576
        
            # 프로젝션 층
            self.projector = nn.Linear(self.vision_hidden_size, self.language_hidden_size)
        
            # 생성용 LM 헤드
            self.lm_head = nn.Linear(self.language_hidden_size, 
                                     self.language_model.config.vocab_size, bias=False)
        
        def forward(self, pixel_values, input_ids, attention_mask=None, labels=None):
            # 이미지 인코딩
            vision_outputs = self.vision_encoder(pixel_values=pixel_values)
            image_features = vision_outputs.last_hidden_state  # [batch, 197, 768]
        
            # 언어 공간으로 투영
            image_embeds = self.projector(image_features)  # [batch, 197, 576]
        
            # 텍스트 임베딩 추출(SmolLM/Llama는 embeddings.word_embeddings 대신 embed_tokens 사용)
            text_embeds = self.language_model.embed_tokens(input_ids)  # [batch, seq_len, 576]
        
            # 이미지와 텍스트 임베딩 연결
            combined_embeds = torch.cat([image_embeds, text_embeds], dim=1)  # [batch, 197+seq_len, 576]
        
            # 결합 시퀀스의 어텐션 마스크 생성
            batch_size = pixel_values.shape[0]
            image_attention = torch.ones(batch_size, image_embeds.shape[1], 
                                         dtype=torch.long, device=pixel_values.device)
            if attention_mask is None:
                attention_mask = torch.ones_like(input_ids)
            combined_attention = torch.cat([image_attention, attention_mask], dim=1)
        
            # 언어 모델 순전파
            outputs = self.language_model(
                inputs_embeds=combined_embeds,
                attention_mask=combined_attention,
                output_hidden_states=True,
                return_dict=True
            )
        
            hidden_states = outputs.last_hidden_state
            logits = self.lm_head(hidden_states)
        
            loss = None
            if labels is not None:
                # 다음 토큰 예측을 위해 로짓과 레이블 이동
                shift_logits = logits[:, image_embeds.shape[1]:-1, :].contiguous()
                shift_labels = labels[:, 1:].contiguous()
            
                # 손실 계산
                loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
                loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), 
                              shift_labels.view(-1))
        
            return {"loss": loss, "logits": logits, "hidden_states": hidden_states}
    
        def count_parameters(self):
            """학습 가능한 전체 매개변수 수를 계산합니다.
            """
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
        def get_model_size_mb(self):
            """packed 양자화 가중치를 포함한 state_dict 직렬화 크기(MiB)입니다.

            GPU 메모리 사용량과는 다른 지표입니다.
            """
            from io import BytesIO

            with BytesIO() as buffer:
                torch.save(self.state_dict(), buffer)
                return buffer.tell() / 1024**2

    return SimpleVLM, execution_step_3


@app.cell
def _(SimpleVLM, execution_step_3, torch):
    # 원본 코드 셀 순서와 모델·데이터 변경의 실행 순서를 보존합니다.
    execution_step_4 = execution_step_3 + 1
    # 기준 모델 초기화
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    baseline_model = SimpleVLM().to(device)

    print(f'기준 모델 통계:')
    print(f'  전체 매개변수: {baseline_model.count_parameters() / 1000000.0:.1f}M')
    print(f'  모델 크기: {baseline_model.get_model_size_mb():.1f} MiB')
    print(f'\n구성 요소별 통계:')
    print(f'  비전 인코더: {sum((p.numel() for p in baseline_model.vision_encoder.parameters())) / 1000000.0:.1f}M')
    print(f'  언어 모델: {sum((p.numel() for p in baseline_model.language_model.parameters())) / 1000000.0:.1f}M')
    print(f'  프로젝터: {sum((p.numel() for p in baseline_model.projector.parameters())) / 1000000.0:.1f}M')
    print(f'  LM 헤드: {sum((p.numel() for p in baseline_model.lm_head.parameters())) / 1000000.0:.1f}M')
    return baseline_model, device, execution_step_4


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 평가 데이터셋 준비

    빠른 평가와 벤치마크를 위해 COCO의 작은 부분집합을 사용합니다.
    """)
    return


@app.cell
def _(AutoImageProcessor, AutoTokenizer, execution_step_4, load_dataset):
    # 원본 코드 셀 순서와 모델·데이터 변경의 실행 순서를 보존합니다.
    execution_step_5 = execution_step_4 + 1
    # 토크나이저와 이미지 처리기 로딩
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M")
    tokenizer.pad_token = tokenizer.eos_token
    image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")

    # Flickr8k 데이터셋 로딩(간단하며 trust_remote_code 불필요)
    dataset = load_dataset("jxie/flickr8k", split="train")
    dataset = dataset.shuffle(seed=42).select(range(500))  # 속도를 위한 작은 부분집합
    print(f'불러온 수: {len(dataset)}개 평가 예시')
    return dataset, execution_step_5, image_processor, tokenizer


@app.cell
def _(
    DataLoader,
    Dataset,
    dataset,
    execution_step_5,
    image_processor,
    tokenizer,
):
    # 원본 코드 셀 순서와 모델·데이터 변경의 실행 순서를 보존합니다.
    execution_step_6 = execution_step_5 + 1
    class CaptionDataset(Dataset):
        def __init__(self, dataset, image_processor, tokenizer, max_length=64):
            self.dataset = dataset
            self.image_processor = image_processor
            self.tokenizer = tokenizer
            self.max_length = max_length
    
        def __len__(self):
            return len(self.dataset)
    
        def __getitem__(self, idx):
            item = self.dataset[idx]
            image = item['image'].convert('RGB')
            # flickr8k는 caption_0, caption_1 등을 사용하므로 첫 캡션 선택
            caption = item['caption_0']
        
            # 이미지 처리
            pixel_values = self.image_processor(image, return_tensors="pt")["pixel_values"][0]
        
            # 캡션 처리
            prompt = "Caption: "
            full_text = f"{prompt}{caption}{self.tokenizer.eos_token}"
        
            encoding = self.tokenizer(
                full_text,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
        
            input_ids = encoding["input_ids"][0]
            attention_mask = encoding["attention_mask"][0]
        
            # 레이블 생성(프롬프트 토큰 마스킹)
            labels = input_ids.clone()
            prompt_len = len(self.tokenizer(prompt, add_special_tokens=False)["input_ids"])
            labels[:prompt_len] = -100
            labels[attention_mask == 0] = -100
        
            return {
                "pixel_values": pixel_values,
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
                "caption": caption
            }

    eval_dataset = CaptionDataset(dataset, image_processor, tokenizer)
    eval_loader = DataLoader(eval_dataset, batch_size=8, shuffle=False)
    print(f'생성한 DataLoader의 배치 수: {len(eval_loader)}개 배치')
    return eval_loader, execution_step_6


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 기준 모델 평가

    기준 모델의 추론 속도와 손실을 측정합니다.
    """)
    return


@app.cell
def _(
    baseline_model,
    device,
    eval_loader,
    execution_step_6,
    time,
    torch,
    tqdm,
):
    # 원본 코드 셀 순서와 모델·데이터 변경의 실행 순서를 보존합니다.
    execution_step_7 = execution_step_6 + 1
    def evaluate_model(model, dataloader, num_batches=50):
        """모델 성능을 평가합니다.
        """
        model.eval()
        total_loss = 0
        total_time = 0
        num_samples = 0
    
        with torch.no_grad():
            for i, batch in enumerate(tqdm(dataloader, total=num_batches, desc='평가 중')):
                if i >= num_batches:
                    break
            
                pixel_values = batch["pixel_values"].to(device)
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
            
                # 추론 시간 측정
                start_time = time.time()
                outputs = model(pixel_values, input_ids, attention_mask, labels)
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                end_time = time.time()
            
                total_loss += outputs["loss"].item() * pixel_values.size(0)
                total_time += (end_time - start_time)
                num_samples += pixel_values.size(0)
    
        avg_loss = total_loss / num_samples
        avg_time_per_sample = total_time / num_samples
        throughput = num_samples / total_time
    
        return {
            "loss": avg_loss,
            "time_per_sample": avg_time_per_sample,
            "throughput": throughput
        }

    print('기준 모델 평가 중...')
    baseline_results = evaluate_model(baseline_model, eval_loader)
    print(f'\n기준 모델 결과:')
    print(f'  손실: {baseline_results['loss']:.4f}')
    print(f'  샘플당 시간: {baseline_results['time_per_sample'] * 1000:.2f} ms')
    print(f'  처리량: {baseline_results['throughput']:.2f}개 샘플/초')
    return baseline_results, execution_step_7


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. INT8 양자화

    **양자화**는 FP32(32비트 부동소수점) 가중치를 INT8(8비트 정수)로 변환합니다. 양자화 대상 가중치의 저장량은 약 **1/4로 축소**됩니다. 임베딩처럼 양자화하지 않는 층과 저장 메타데이터도 있으므로 전체 모델의 압축률은 별도로 측정해야 합니다.

    ### 동작 원리
    - **가중치 양자화**: 가중치를 INT8로 변환
    - **활성값 양자화**: 활성값을 INT8로 변환
    - **보정**: 샘플 데이터에서 스케일과 영점(zero-point) 계산
    - **추론**: INT8 연산 사용(지원 하드웨어에서 더 빠르게 실행 가능)

    실습에서는 간단한 **PyTorch 동적 양자화**를 사용합니다.
    """)
    return


@app.cell
def _(baseline_model, copy, execution_step_7, nn, torch):
    # 원본 코드 셀 순서와 모델·데이터 변경의 실행 순서를 보존합니다.
    execution_step_8 = execution_step_7 + 1
    # 양자화 모델 생성(동적 양자화는 CPU에서 수행해야 함)
    baseline_model_cpu = copy.deepcopy(baseline_model).cpu()
    baseline_model_cpu.eval()

    # 선형 층에 동적 양자화 적용
    # 참고: PyTorch 동적 양자화는 CPU에서만 동작
    quantized_model = torch.quantization.quantize_dynamic(
        baseline_model_cpu,
        {nn.Linear},  # 모든 Linear 층 양자화
        dtype=torch.qint8
    )

    print(f'\n양자화 모델 통계:')
    print(f'  모델 크기: {quantized_model.get_model_size_mb():.1f} MiB')
    print(f'  크기 감소 비율: {baseline_model.get_model_size_mb() / quantized_model.get_model_size_mb():.2f}x')
    print(f'\n참고: 동적 양자화는 CPU에서만 동작합니다(PyTorch 제약).')
    return execution_step_8, quantized_model


@app.cell
def _(
    baseline_model,
    baseline_results,
    eval_loader,
    execution_step_8,
    quantized_model,
    time,
    torch,
    tqdm,
):
    # 원본 코드 셀 순서와 모델·데이터 변경의 실행 순서를 보존합니다.
    execution_step_9 = execution_step_8 + 1
    # CPU에서 양자화 모델 평가
    def evaluate_model_cpu(model, dataloader, num_batches=20):
        """CPU에서 모델을 평가합니다(양자화 모델용).
        """
        model.eval()
        total_loss = 0
        total_time = 0
        num_samples = 0
    
        with torch.no_grad():
            for i, batch in enumerate(tqdm(dataloader, total=num_batches, desc='평가 중(CPU)')):
                if i >= num_batches:
                    break
            
                # 양자화 모델용 데이터는 CPU에 유지
                pixel_values = batch["pixel_values"]  # 데이터로더에서 이미 CPU 상태
                input_ids = batch["input_ids"]
                attention_mask = batch["attention_mask"]
                labels = batch["labels"]
            
                # 추론 시간 측정
                start_time = time.time()
                outputs = model(pixel_values, input_ids, attention_mask, labels)
                end_time = time.time()
            
                total_loss += outputs["loss"].item() * pixel_values.size(0)
                total_time += (end_time - start_time)
                num_samples += pixel_values.size(0)
    
        avg_loss = total_loss / num_samples
        avg_time_per_sample = total_time / num_samples
        throughput = num_samples / total_time
    
        return {
            "loss": avg_loss,
            "time_per_sample": avg_time_per_sample,
            "throughput": throughput
        }

    print('CPU에서 양자화 모델 평가 중...')
    print('(참고: 양자화 모델은 CPU에서 실행되므로 CPU·GPU 속도의 직접 비교에는 한계가 있습니다.)')
    quantized_results = evaluate_model_cpu(quantized_model, eval_loader, num_batches=10)

    print(f'\n양자화 결과(CPU):')
    print(f'  손실: {quantized_results['loss']:.4f} (기준 모델: {baseline_results['loss']:.4f})')
    print(f'  손실 변화: {(quantized_results['loss'] - baseline_results['loss']) / baseline_results['loss'] * 100:+.2f}%')
    print(f'  샘플당 시간: {quantized_results['time_per_sample'] * 1000:.2f} ms (CPU)')
    print(f'  처리량: {quantized_results['throughput']:.2f}개 샘플/초(CPU)')
    print(f'\n주요 이점: 모델 크기 축소, 기존 {baseline_model.get_model_size_mb():.0f} MiB → {quantized_model.get_model_size_mb():.0f} MiB ({baseline_model.get_model_size_mb() / quantized_model.get_model_size_mb():.1f}배 축소)')
    return (execution_step_9,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. 구조적 가지치기

    **가지치기**는 중요도가 낮은 가중치나 어텐션 헤드·FFN 차원 같은 구조 전체를 제거해 모델 크기와 연산량을 줄입니다.

    ### 가지치기의 종류
    - **비구조적**: 개별 가중치 제거 → 희소 행렬(특수 하드웨어 필요)
    - **구조적**: 행·열·헤드 전체 제거 → 더 작은 밀집 행렬(일반 하드웨어에서 동작)

    중요도 점수를 기준으로 어텐션 헤드와 FFN 차원을 제거하는 **구조적 가지치기**를 살펴봅니다.
    """)
    return


@app.cell
def _(baseline_model, device, eval_loader, execution_step_9, torch, tqdm):
    # 원본 코드 셀 순서와 모델·데이터 변경의 실행 순서를 보존합니다.
    execution_step_10 = execution_step_9 + 1
    def compute_head_importance(model, dataloader, num_batches=20):
        """어텐션 헤드별 중요도 점수를 계산합니다.
        """
        model.eval()
    
        # 층·헤드 수 확인(SmolLM/Llama는 encoder.layer 대신 layers 사용)
        num_layers = len(model.language_model.layers)
        num_heads = model.language_model.config.num_attention_heads
    
        # 중요도 점수 초기화
        head_importance = torch.zeros(num_layers, num_heads).to(device)
    
        with torch.no_grad():
            for i, batch in enumerate(tqdm(dataloader, total=num_batches, desc='중요도 계산 중')):
                if i >= num_batches:
                    break
            
                pixel_values = batch["pixel_values"].to(device)
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
            
                # 순전파
                outputs = model(pixel_values, input_ids, attention_mask)
            
                # 헤드별 어텐션 점수 계산
                # 어텐션 크기를 기준으로 한 단순화된 중요도 지표
                for layer_idx in range(num_layers):
                    # 해당 층의 어텐션 가중치 추출
                    # 실제로는 모델에서 어텐션 가중치를 추출해야 함
                    # 현재는 임시 무작위 중요도 사용
                    head_importance[layer_idx] += torch.rand(num_heads).to(device)
    
        # 정규화
        head_importance /= num_batches
    
        return head_importance

    def prune_attention_heads(model, head_importance, prune_ratio=0.3):
        """중요도가 가장 낮은 어텐션 헤드를 가지치기합니다.
        """
        # 중요도 점수 펼치기
        flat_importance = head_importance.view(-1)
        num_heads_total = flat_importance.size(0)
        num_to_prune = int(num_heads_total * prune_ratio)
    
        # 가지치기할 낮은 중요도 헤드의 인덱스 추출
        _, indices_to_prune = torch.topk(flat_importance, num_to_prune, largest=False)
    
        print(f'가지치기할 헤드 수: {num_to_prune} / 전체 {num_heads_total}개 어텐션 헤드({prune_ratio * 100:.0f}%)')
    
        # 실제로는 여기서 모델 구조를 수정해야 함
        # 시연을 위해 마스크 생성
        head_mask = torch.ones_like(flat_importance)
        head_mask[indices_to_prune] = 0
    
        return head_mask.view_as(head_importance)

    # 헤드 중요도 계산
    print('어텐션 헤드 중요도 계산 중...')
    head_importance = compute_head_importance(baseline_model, eval_loader)

    # 헤드의 30% 가지치기
    head_mask = prune_attention_heads(baseline_model, head_importance, prune_ratio=0.3)
    print(f'남은 헤드 수: {head_mask.sum().item():.0f} / {head_mask.numel()}')
    return (execution_step_10,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **참고:** 완전한 구조적 가지치기를 구현하려면 모델 구조를 수정해야 합니다. 위 코드는 개념을 보여 줍니다. 실제 적용 과정은 다음과 같습니다.
    1. 중요도가 낮은 헤드·차원 식별
    2. 차원을 줄인 새 모델 생성
    3. 중요한 가중치 복사
    4. 가지치기한 모델 미세 조정
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. 지식 증류

    **지식 증류**는 작은 "학생" 모델이 큰 "교사" 모델을 모방하도록 학습하는 방법입니다. 학생은 다음에서 배웁니다.
    - **소프트 레이블**: 교사의 출력 확률(하드 레이블보다 많은 정보 포함)
    - **은닉 상태**: 교사의 중간 표현

    ### 증류 손실
    ```
    L = α * L_hard(student, labels) + (1-α) * L_soft(student, teacher)
    L_soft = KL_divergence(student_logits/T, teacher_logits/T)
    ```

    T는 확률분포를 부드럽게 만드는 온도입니다.
    """)
    return


@app.cell
def _(AutoModel, baseline_model, device, execution_step_10, nn, torch):
    # 원본 코드 셀 순서와 모델·데이터 변경의 실행 순서를 보존합니다.
    execution_step_11 = execution_step_10 + 1
    class TinyVLM(nn.Module):
        """증류용 소형 학생 VLM입니다.
        """
        def __init__(self):
            super().__init__()
        
            # 더 작은 비전 인코더나 증류 버전 사용
            self.vision_encoder = AutoModel.from_pretrained("google/vit-base-patch16-224")
            self.vision_hidden_size = 768
        
            # 더 작은 언어 모델 사용
            self.language_model = AutoModel.from_pretrained("HuggingFaceTB/SmolLM-135M")
            self.language_hidden_size = 576
        
            # 실제로는 작은 LM을 사용하지만 시연에서는 같은 크기 사용
            # 실제 증류에는 SmolLM-35M이나 별도의 소형 구조 사용
        
            self.projector = nn.Linear(self.vision_hidden_size, self.language_hidden_size)
            self.lm_head = nn.Linear(self.language_hidden_size, 
                                     self.language_model.config.vocab_size, bias=False)
    
        def forward(self, pixel_values, input_ids, attention_mask=None, labels=None):
            # SimpleVLM과 같은 순전파
            vision_outputs = self.vision_encoder(pixel_values=pixel_values)
            image_features = vision_outputs.last_hidden_state
            image_embeds = self.projector(image_features)
        
            # SmolLM/Llama는 embed_tokens 사용
            text_embeds = self.language_model.embed_tokens(input_ids)
            combined_embeds = torch.cat([image_embeds, text_embeds], dim=1)
        
            batch_size = pixel_values.shape[0]
            image_attention = torch.ones(batch_size, image_embeds.shape[1], 
                                         dtype=torch.long, device=pixel_values.device)
            if attention_mask is None:
                attention_mask = torch.ones_like(input_ids)
            combined_attention = torch.cat([image_attention, attention_mask], dim=1)
        
            outputs = self.language_model(
                inputs_embeds=combined_embeds,
                attention_mask=combined_attention,
                output_hidden_states=True,
                return_dict=True
            )
        
            hidden_states = outputs.last_hidden_state
            logits = self.lm_head(hidden_states)
        
            loss = None
            if labels is not None:
                shift_logits = logits[:, image_embeds.shape[1]:-1, :].contiguous()
                shift_labels = labels[:, 1:].contiguous()
                loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
                loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), 
                              shift_labels.view(-1))
        
            return {"loss": loss, "logits": logits, "hidden_states": hidden_states}
    
        def count_parameters(self):
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
        def get_model_size_mb(self):
            """packed 양자화 가중치를 포함한 state_dict 직렬화 크기(MiB)입니다."""
            from io import BytesIO

            with BytesIO() as buffer:
                torch.save(self.state_dict(), buffer)
                return buffer.tell() / 1024**2

    # 학생 모델 생성
    student_model = TinyVLM().to(device)
    print(f'\n학생 모델 통계:')
    print(f'  전체 매개변수: {student_model.count_parameters() / 1000000.0:.1f}M')
    print(f'  모델 크기: {student_model.get_model_size_mb():.1f} MiB')
    print(f'  교사 대비 크기 감소 비율: {baseline_model.get_model_size_mb() / student_model.get_model_size_mb():.2f}배 축소')
    return execution_step_11, student_model


@app.cell
def _(F, device, execution_step_11, nn, torch, tqdm):
    # 원본 코드 셀 순서와 모델·데이터 변경의 실행 순서를 보존합니다.
    execution_step_12 = execution_step_11 + 1
    def distillation_loss(student_logits, teacher_logits, labels, temperature=2.0, alpha=0.5):
        """하드·소프트 목표를 결합한 증류 손실을 계산합니다.
        """
        # 하드 손실(실제 레이블을 사용한 표준 교차 엔트로피)
        loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
        hard_loss = loss_fct(student_logits.view(-1, student_logits.size(-1)), labels.view(-1))
    
        # 소프트 손실(교사와의 KL 발산)
        # 온도를 적용해 분포를 부드럽게 만듦
        student_soft = F.log_softmax(student_logits / temperature, dim=-1)
        teacher_soft = F.softmax(teacher_logits / temperature, dim=-1)
    
        # KL 발산
        soft_loss = F.kl_div(student_soft, teacher_soft, reduction='batchmean') * (temperature ** 2)
    
        # 손실 결합
        total_loss = alpha * hard_loss + (1 - alpha) * soft_loss
    
        return total_loss, hard_loss, soft_loss

    def train_distillation(student, teacher, train_loader, num_epochs=3, lr=1e-4, 
                          temperature=2.0, alpha=0.5):
        """지식 증류로 학생 모델을 학습합니다.
        """
        teacher.eval()  # 교사는 평가 모드
        student.train()
    
        optimizer = torch.optim.AdamW(student.parameters(), lr=lr)
    
        losses = []
    
        for epoch in range(num_epochs):
            epoch_loss = 0
            pbar = tqdm(train_loader, desc=f'에포크 {epoch + 1}/{num_epochs}')
        
            for batch in pbar:
                pixel_values = batch["pixel_values"].to(device)
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
            
                # 교사 예측 추출(기울기 없음)
                with torch.no_grad():
                    teacher_outputs = teacher(pixel_values, input_ids, attention_mask)
                    teacher_logits = teacher_outputs["logits"]
            
                # 학생 예측 추출
                student_outputs = student(pixel_values, input_ids, attention_mask)
                student_logits = student_outputs["logits"]
            
                # 로짓 정렬(이미지 토큰 건너뛰기)
                image_seq_len = teacher_outputs["hidden_states"].shape[1] - input_ids.shape[1]
                student_logits_aligned = student_logits[:, image_seq_len:-1, :]
                teacher_logits_aligned = teacher_logits[:, image_seq_len:-1, :]
                labels_aligned = labels[:, 1:]
            
                # 증류 손실 계산
                loss, hard_loss, soft_loss = distillation_loss(
                    student_logits_aligned, teacher_logits_aligned, labels_aligned,
                    temperature=temperature, alpha=alpha
                )
            
                # 역전파 및 최적화
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
                epoch_loss += loss.item()
                pbar.set_postfix({"loss": loss.item(), "hard": hard_loss.item(), "soft": soft_loss.item()})
        
            avg_loss = epoch_loss / len(train_loader)
            losses.append(avg_loss)
            print(f'에포크 {epoch + 1} - 평균 손실: {avg_loss:.4f}')
    
        return losses

    print('\n참고: 전체 증류 학습에는 수 시간이 걸릴 수 있습니다.')
    print('시연에서는 학습 설정을 보여 줍니다.')
    print('실제 학습 명령: train_distillation(student_model, baseline_model, train_loader)')
    return (execution_step_12,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 압축 기법 결합: 양자화 + 증류

    높은 압축률을 위해 **여러 기법을 결합**할 수 있습니다.
    1. 증류로 작은 학생 모델 학습
    2. 학생 모델을 INT8로 양자화
    3. 선택 사항: 양자화 전에 학생 모델 가지치기

    원문에서 제시한 기대 효과는 다음과 같습니다.
    - **크기를 1/8~1/10로 축소**: 작은 구조 + INT8
    - **속도 3~4배 향상**: 적은 매개변수 + INT8 연산
    - **정확도 하락 10% 미만**: 증류로 지식 보존
    """)
    return


@app.cell
def _(baseline_model, copy, execution_step_12, nn, student_model, torch):
    # 원본 코드 셀 순서와 모델·데이터 변경의 실행 순서를 보존합니다.
    execution_step_13 = execution_step_12 + 1
    # 증류 학습 후 학생 양자화
    # student_model_trained = train_distillation(...)  # 여기서 실제 학습 가능

    # 시연에서는 학습 전 학생을 양자화
    student_quantized = torch.quantization.quantize_dynamic(
        copy.deepcopy(student_model),
        {nn.Linear},
        dtype=torch.qint8
    )

    print(f'\n압축 기법 결합 결과:')
    print(f'  원본 모델: {baseline_model.get_model_size_mb():.1f} MiB')
    print(f'  증류 학생: {student_model.get_model_size_mb():.1f} MiB')
    print(f'  양자화 학생: {student_quantized.get_model_size_mb():.1f} MiB')
    print(f'  전체 압축률: {baseline_model.get_model_size_mb() / student_quantized.get_model_size_mb():.2f}x')
    return execution_step_13, student_quantized


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 압축 기법 비교표

    모든 압축 기법을 비교해 정리합니다.

    실측 크기는 packed 양자화 가중치까지 포함한 `state_dict`의 직렬화 크기(MiB)입니다.
    FP32와 INT8에 같은 저장 방식을 적용하며, GPU 메모리 사용량과는 구분합니다.
    """)
    return


@app.cell
def _(
    baseline_model,
    execution_step_13,
    quantized_model,
    student_model,
    student_quantized,
):
    # 원본 코드 셀 순서와 모델·데이터 변경의 실행 순서를 보존합니다.
    execution_step_14 = execution_step_13 + 1
    import pandas as pd

    # 비교표 생성
    comparison_data = {
        '기법': [
            '기준 모델(FP32)',
            'INT8 양자화',
            '구조적 가지치기(30%)',
            '지식 증류',
            '양자화 학생',
        ],
        'state_dict 크기(MiB)': [
            f"{baseline_model.get_model_size_mb():.1f}",
            f"{quantized_model.get_model_size_mb():.1f}",
            f"{baseline_model.get_model_size_mb() * 0.7:.1f}",  # 추정값
            f"{student_model.get_model_size_mb():.1f}",
            f"{student_quantized.get_model_size_mb():.1f}",
        ],
        '크기 감소 비율': [
            "1.0x",
            f"{baseline_model.get_model_size_mb() / quantized_model.get_model_size_mb():.1f}x",
            "1.4x",
            f"{baseline_model.get_model_size_mb() / student_model.get_model_size_mb():.1f}x",
            f"{baseline_model.get_model_size_mb() / student_quantized.get_model_size_mb():.1f}x",
        ],
        '예상 속도 향상': [
            "1.0x",
            "1.5-2x",
            "2x",
            "1.5x",
            "3-4x",
        ],
        '예상 정확도 하락': [
            "0%",
            "<2%",
            "3-5%",
            "<3%",
            "<7%",
        ],
        '구현 복잡도': [
            "N/A",
            '쉬움(1줄)',
            '어려움(구조 변경)',
            '중간(학습 필요)',
            '쉬움(증류 후 양자화)',
        ]
    }

    df_comparison = pd.DataFrame(comparison_data)
    print('\n압축 기법 비교:')
    print(df_comparison.to_string(index=False))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 핵심 내용

    ### 1. **양자화(가장 간단함)**
    - ✅ PyTorch에서 한 줄로 구현 가능
    - ✅ 양자화 대상 가중치의 저장량을 약 1/4로 축소(전체 모델의 비율은 실측)
    - ✅ CPU에서 속도 1.5~2배 향상(GPU에서는 효과가 더 작음)
    - ⚠️ 최대 속도 향상을 위해 INT8 하드웨어 지원 필요

    ### 2. **가지치기(가장 복잡함)**
    - ✅ 구조적 가지치기로 속도 2~3배 향상 가능
    - ✅ 일반 하드웨어에서 동작
    - ❌ 모델 구조 수정 필요
    - ❌ 가지치기 후 미세 조정 필요

    ### 3. **증류(정확도 보존에 유리)**
    - ✅ 정확도 손실이 적은 소형 모델
    - ✅ 교사 모델의 지식 활용
    - ❌ 학습 시간과 데이터 필요
    - ❌ 사전 학습된 교사 모델 필요

    ### 4. **결합(전체적인 효과)**
    - ✅ 정확도 하락 10% 미만으로 8~10배 압축
    - ✅ 속도 3~4배 향상
    - ✅ 실제 배포에 활용 가능
    - 권장 순서: **증류 → 양자화**

    ## 실무 적용 방향

    **빠른 배포:**
    - 간단하고 효과적인 INT8 양자화 사용

    **최대 압축:**
    - 증류로 작은 학생 모델 학습
    - 이어서 INT8 양자화

    **최대 속도:**
    - 가지치기와 양자화 결합
    - INT8 가속 하드웨어에 배포

    **엣지 장치:**
    - 세 기법 모두 결합
    - INT4 같은 더 강한 압축 고려

    위 성능 수치는 원문에서 제시한 기대 효과입니다. 실제 효과는 모델·하드웨어에서 확인해야 합니다.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## GPU 메모리 정리

    압축 실험을 마친 뒤 GPU 메모리를 해제합니다.
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
    ## 다음 단계

    자신의 VLM에 적용하려면 다음 순서로 진행합니다.

    1. **양자화부터 시작**: 적은 작업으로 빠르게 효과 확인
    2. **증류 시도**: 학습 데이터와 연산 자원이 있을 때 적용
    3. **가지치기 탐색**: 맞춤 구조와 추가 속도 향상 추구
    4. **세심한 벤치마크**: 크기·속도·정확도를 모두 측정
    5. **단계적 배포**: 실제 서비스에 적용하기 전에 압축 모델 테스트

    **더 읽을 자료:**
    - PyTorch 양자화: https://pytorch.org/docs/stable/quantization.html
    - DistilBERT 논문: https://arxiv.org/abs/1910.01108
    - 복권 가설(The Lottery Ticket Hypothesis, 가지치기): https://arxiv.org/abs/1803.03635
    - QLoRA(4비트 양자화 + LoRA): https://arxiv.org/abs/2305.14314
    """)
    return


if __name__ == "__main__":
    app.run()
