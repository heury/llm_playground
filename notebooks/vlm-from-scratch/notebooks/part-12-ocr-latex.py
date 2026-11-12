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
    # OCR·LaTeX 생성을 위한 VLM: 이미지에서 구조화된 텍스트로

    Nipun Batra · 2026-01-04
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # OCR·LaTeX 생성을 위한 VLM

    이 노트북에서는 시각·언어 모델이 **광학 문자 인식(OCR)**과 **구조화된 문서 이해**에 어떻게 활용되는지 살펴봅니다.

    1. **기본 OCR**: 현대 VLM으로 이미지에서 텍스트 추출
    2. **수식 OCR**: 손글씨·인쇄 수식을 LaTeX로 변환
    3. **문서 구조**: 표, 목록, 제목, 배치 이해
    4. **표 → LaTeX**: 표 추출 및 LaTeX 코드 생성
    5. **전체 문서 → LaTeX**: 종단 간 문서 변환

    ## OCR에 VLM을 사용하는 이유

    Tesseract, PaddleOCR 같은 전통적 OCR은 다음을 처리하기 어렵습니다.
    - **수학 표기**: $\int_0^\infty e^{-x^2} dx = \frac{\sqrt{\pi}}{2}$
    - **복잡한 배치**: 다단 구성, 중첩 표, 그림
    - **손글씨**: 특히 수학 기호
    - **문맥 이해**: "0"과 "O", "1"과 "l" 구별

    VLM의 강점은 다음과 같습니다.
    - ✅ **의미적 문맥** 이해: 수식, 코드, 자연어
    - ✅ **구조화된 출력** 생성: LaTeX, Markdown, JSON
    - ✅ 공간 추론을 통한 **복잡한 배치** 처리
    - ✅ **손글씨와 인쇄 텍스트** 모두 처리

    ## 사용할 모델

    1. **TrOCR**: 트랜스포머 기반 OCR(텍스트 추출)
    2. **Nougat**: 학술 문서 이해(수식, 표)
    3. **Pix2Tex**: 수식 이미지를 LaTeX로 변환
    4. **직접 미세 조정한 VLM**: 맞춤형 OCR → LaTeX 모델

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
    from torch.utils.data import Dataset, DataLoader
    from transformers import (
        TrOCRProcessor, VisionEncoderDecoderModel,
        AutoProcessor, AutoModelForVision2Seq,
        AutoTokenizer, AutoModel, AutoImageProcessor
    )
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
    from datasets import load_dataset
    from tqdm import tqdm
    import re
    import gc

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
        Image,
        ImageDraw,
        ImageFont,
        TrOCRProcessor,
        VisionEncoderDecoderModel,
        execution_step_2,
        load_dataset,
        nn,
        plt,
        torch,
        tqdm,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. TrOCR을 이용한 기본 OCR

    **TrOCR**은 다음을 결합한 트랜스포머 기반 OCR 모델입니다.
    - 이미지용 비전 트랜스포머(ViT) 인코더
    - 텍스트 생성용 RoBERTa 디코더

    전통적 OCR과 달리 텍스트를 자기회귀적으로 생성하며 언어 문맥을 활용할 수 있습니다.
    """)
    return


@app.cell
def _(TrOCRProcessor, VisionEncoderDecoderModel, execution_step_2, torch):
    # 원본 코드 셀 순서와 모델·데이터 변경의 실행 순서를 보존합니다.
    execution_step_3 = execution_step_2 + 1
    # 손글씨용 TrOCR 로딩
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print('손글씨용 TrOCR 로딩 중...')
    trocr_processor = TrOCRProcessor.from_pretrained("microsoft/trocr-large-handwritten")
    trocr_model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-large-handwritten")
    trocr_model = trocr_model.to(device)
    trocr_model.eval()
    print('TrOCR을 불러왔습니다.')
    return device, execution_step_3, trocr_model, trocr_processor


@app.cell
def _(
    device,
    execution_step_3,
    load_dataset,
    plt,
    torch,
    trocr_model,
    trocr_processor,
):
    # 원본 코드 셀 순서와 모델·데이터 변경의 실행 순서를 보존합니다.
    execution_step_4 = execution_step_3 + 1

    def ocr_image_trocr(image):
        """TrOCR로 이미지에서 텍스트를 추출합니다.
        """
        # IAM의 흑백 이미지와 RGBA 입력도 ViT 처리기가 받는 RGB로 통일합니다.
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        image = image.convert('RGB')
        pixel_values = trocr_processor(image, return_tensors='pt').pixel_values.to(device)
        with torch.no_grad():
            generated_ids = trocr_model.generate(pixel_values)
            generated_text = trocr_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return generated_text
    print('IAM Handwriting 데이터셋 로딩 중...')
    iam_dataset = load_dataset('Teklia/IAM-line', split='test[:10]')
    # IAM Handwriting 데이터셋 테스트
    _fig, _axes = plt.subplots(3, 2, figsize=(12, 10))
    _axes = _axes.flatten()
    for _i in range(min(6, len(iam_dataset))):
    # 예시 표시
        sample = iam_dataset[_i]
        image = sample['image']
        ground_truth = sample['text']
        _predicted = ocr_image_trocr(image)
        _axes[_i].imshow(image, cmap='gray')
        _axes[_i].set_title(f'정답: {ground_truth}\n예측: {_predicted}', fontsize=8)
        _axes[_i].axis('off')
    plt.tight_layout()
    plt.show()  # OCR 수행 / 표시
    return execution_step_4, iam_dataset, ocr_image_trocr


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. 수식 OCR로 LaTeX 생성

    전통적 OCR은 수학 표기를 처리하기 어렵습니다. 수학적 구조를 이해하는 **전문 모델**을 사용합니다.

    ### 접근
    1. **Pix2Tex/LaTeX-OCR**: 이미지 → LaTeX
    2. 수식 이미지와 LaTeX 레이블로 VLM 미세 조정
    3. IM2LATEX, CROHME(손글씨 수식) 데이터셋 활용

    이 실습에서는 간단한 VLM 기반 수식 OCR을 만듭니다.
    """)
    return


@app.cell
def _(Image, execution_step_4, plt):
    # 원본 코드 셀 순서와 모델·데이터 변경의 실행 순서를 보존합니다.
    execution_step_5 = execution_step_4 + 1
    # 정답 LaTeX를 알고 있는 합성 수식 이미지 생성
    # 실제로는 IM2LATEX-100K 데이터셋 사용
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import mathtext  # 비대화형 백엔드
    from io import BytesIO

    def latex_to_image(latex_string, fontsize=20, dpi=150):
        """LaTeX 수식을 이미지로 렌더링합니다.
        """
        fig = plt.figure(figsize=(6, 1))
        fig.text(0.5, 0.5, f'${latex_string}$', fontsize=fontsize, ha='center', va='center')
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=dpi, bbox_inches='tight', pad_inches=0.1)
        plt.close(fig)
        buf.seek(0)  # 버퍼에 저장
        image = Image.open(buf).convert('RGB')
        return image
    latex_examples = ['\\int_0^\\infty e^{-x^2} dx = \\frac{\\sqrt{\\pi}}{2}', 'E = mc^2', '\\nabla \\times \\mathbf{E} = -\\frac{\\partial \\mathbf{B}}{\\partial t}', '\\sum_{n=1}^\\infty \\frac{1}{n^2} = \\frac{\\pi^2}{6}', 'f(x) = \\frac{1}{\\sigma\\sqrt{2\\pi}} e^{-\\frac{(x-\\mu)^2}{2\\sigma^2}}', '\\lim_{x \\to 0} \\frac{\\sin x}{x} = 1']
    _fig, _axes = plt.subplots(3, 2, figsize=(12, 8))
    _axes = _axes.flatten()
    equation_images = []
    for _i, _latex in enumerate(latex_examples):
        _img = latex_to_image(_latex, fontsize=24)
    # LaTeX 수식 예시
        equation_images.append((_img, _latex))
        _axes[_i].imshow(_img)
        _axes[_i].set_title(f'LaTeX: {_latex[:40]}...', fontsize=8)
        _axes[_i].axis('off')
    plt.tight_layout()
    plt.show()
    # 이미지 생성
    print(f'\n생성한 수: {len(equation_images)}장 수식 이미지')
    return execution_step_5, latex_to_image


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. 수식 OCR을 위한 VLM 미세 조정

    이전 구조와 비슷한 소형 VLM을 미세 조정해 수식 이미지에서 LaTeX를 예측합니다.
    """)
    return


@app.cell
def _(AutoModel, device, execution_step_5, nn, torch):
    # 원본 코드 셀 순서와 모델·데이터 변경의 실행 순서를 보존합니다.
    execution_step_6 = execution_step_5 + 1
    class EquationOCRVLM(nn.Module):
        """수식 OCR과 LaTeX 생성에 특화된 VLM입니다.
        """
        def __init__(self):
            super().__init__()
        
            # 비전 인코더
            self.vision_encoder = AutoModel.from_pretrained("google/vit-base-patch16-224")
            self.vision_hidden_size = self.vision_encoder.config.hidden_size
        
            # 언어 모델(디코더)
            self.language_model = AutoModel.from_pretrained("HuggingFaceTB/SmolLM-135M")
            self.language_hidden_size = self.language_model.config.hidden_size
        
            # 프로젝터
            self.projector = nn.Linear(self.vision_hidden_size, self.language_hidden_size)
        
            # 언어 모델 헤드
            self.lm_head = nn.Linear(self.language_hidden_size,
                                     self.language_model.config.vocab_size, bias=False)
    
        def forward(self, pixel_values, input_ids, attention_mask=None, labels=None):
            # 이미지 인코딩
            vision_outputs = self.vision_encoder(pixel_values=pixel_values)
            image_features = vision_outputs.last_hidden_state
            image_embeds = self.projector(image_features)
        
            # 텍스트 임베딩 추출
            text_embeds = self.language_model.get_input_embeddings()(input_ids)
        
            # 연결
            combined_embeds = torch.cat([image_embeds, text_embeds], dim=1)
        
            # 어텐션 마스크
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
                return_dict=True
            )
        
            hidden_states = outputs.last_hidden_state
            logits = self.lm_head(hidden_states)
        
            # 손실 계산
            loss = None
            if labels is not None:
                shift_logits = logits[:, image_embeds.shape[1]:-1, :].contiguous()
                shift_labels = labels[:, 1:].contiguous()
                loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
                loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)),
                              shift_labels.view(-1))
        
            return {"loss": loss, "logits": logits}

    # 모델 초기화
    equation_vlm = EquationOCRVLM().to(device)
    print(f'수식 OCR VLM 매개변수: {sum((p.numel() for p in equation_vlm.parameters())) / 1000000.0:.1f}M')
    return equation_vlm, execution_step_6


@app.cell
def _(
    AutoImageProcessor,
    AutoTokenizer,
    DataLoader,
    Dataset,
    execution_step_6,
    latex_to_image,
):
    # 원본 코드 셀 순서와 모델·데이터 변경의 실행 순서를 보존합니다.
    execution_step_7 = execution_step_6 + 1
    # 시연용 합성 데이터셋 생성
    # 실제 데이터셋: IM2LATEX-100K, https://zenodo.org/record/56198

    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M")
    tokenizer.pad_token = tokenizer.eos_token
    image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")

    # 확장 LaTeX 데이터셋
    training_latex = [
        r"x^2 + y^2 = r^2",
        r"\frac{a}{b} + \frac{c}{d} = \frac{ad + bc}{bd}",
        r"\sin^2(x) + \cos^2(x) = 1",
        r"\int x^n dx = \frac{x^{n+1}}{n+1} + C",
        r"\frac{d}{dx} e^x = e^x",
        r"\sum_{i=1}^n i = \frac{n(n+1)}{2}",
        r"\prod_{i=1}^n i = n!",
        r"\lim_{n \to \infty} \left(1 + \frac{1}{n}\right)^n = e",
        r"\nabla \cdot \mathbf{F} = \frac{\partial F_x}{\partial x} + \frac{\partial F_y}{\partial y} + \frac{\partial F_z}{\partial z}",
        r"E[X] = \sum_{i} x_i p(x_i)",
        r"\sigma^2 = E[(X - \mu)^2]",
        r"P(A|B) = \frac{P(B|A)P(A)}{P(B)}",
        r"\det(A) = \sum_{\sigma \in S_n} \text{sgn}(\sigma) \prod_{i=1}^n a_{i,\sigma(i)}",
        r"\|\mathbf{v}\| = \sqrt{v_1^2 + v_2^2 + \cdots + v_n^2}",
        r"\mathbf{A}\mathbf{x} = \lambda \mathbf{x}",
    ]

    class EquationDataset(Dataset):
        def __init__(self, latex_equations, tokenizer, image_processor, max_length=128):
            self.equations = latex_equations
            self.tokenizer = tokenizer
            self.image_processor = image_processor
            self.max_length = max_length
    
        def __len__(self):
            return len(self.equations)
    
        def __getitem__(self, idx):
            latex = self.equations[idx]
        
            # 이미지 생성
            image = latex_to_image(latex, fontsize=20)
        
            # 이미지 처리
            pixel_values = self.image_processor(image, return_tensors="pt")["pixel_values"][0]
        
            # LaTeX 텍스트 처리
            prompt = "LaTeX: "
            full_text = f"{prompt}{latex}{self.tokenizer.eos_token}"
        
            encoding = self.tokenizer(
                full_text,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
        
            input_ids = encoding["input_ids"][0]
            attention_mask = encoding["attention_mask"][0]
        
            # 레이블 생성
            labels = input_ids.clone()
            prompt_len = len(self.tokenizer(prompt, add_special_tokens=False)["input_ids"])
            labels[:prompt_len] = -100
            labels[attention_mask == 0] = -100
        
            return {
                "pixel_values": pixel_values,
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
                "latex": latex
            }

    # 시연용 작은 데이터셋 생성
    train_dataset = EquationDataset(training_latex * 20, tokenizer, image_processor)  # 샘플 수를 늘리기 위해 반복
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

    print(f'학습 데이터셋 크기: {len(train_dataset)}')
    return execution_step_7, image_processor, tokenizer, train_loader


@app.cell
def _(device, equation_vlm, execution_step_7, plt, torch, tqdm, train_loader):
    # 원본 코드 셀 순서와 모델·데이터 변경의 실행 순서를 보존합니다.
    execution_step_8 = execution_step_7 + 1
    def train_equation_vlm(model, train_loader, num_epochs=5, lr=1e-4):
        """수식 OCR 모델을 학습합니다.
        """
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        model.train()
    
        losses = []
    
        for epoch in range(num_epochs):
            epoch_loss = 0
            pbar = tqdm(train_loader, desc=f'에포크 {epoch + 1}/{num_epochs}')
        
            for batch in pbar:
                pixel_values = batch["pixel_values"].to(device)
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
            
                outputs = model(pixel_values, input_ids, attention_mask, labels)
                loss = outputs["loss"]
            
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
                epoch_loss += loss.item()
                pbar.set_postfix({"loss": loss.item()})
        
            avg_loss = epoch_loss / len(train_loader)
            losses.append(avg_loss)
            print(f'에포크 {epoch + 1} - 평균 손실: {avg_loss:.4f}')
    
        return losses

    print('수식 OCR 모델 학습 중...')
    print('참고: 제한된 데이터의 간단한 시연입니다. 실제 모델에는 IM2LATEX-100K가 필요합니다.')
    losses = train_equation_vlm(equation_vlm, train_loader, num_epochs=3, lr=1e-4)

    # 학습 곡선 표시
    plt.figure(figsize=(8, 5))
    plt.plot(losses, marker='o')
    plt.xlabel('에포크')
    plt.ylabel('손실')
    plt.title('수식 OCR 학습 손실')
    plt.grid(True)
    plt.show()
    return (execution_step_8,)


@app.cell
def _(
    device,
    equation_vlm,
    execution_step_8,
    image_processor,
    latex_to_image,
    plt,
    tokenizer,
    torch,
):
    # 원본 코드 셀 순서와 모델·데이터 변경의 실행 순서를 보존합니다.
    execution_step_9 = execution_step_8 + 1

    def predict_latex(model, image, tokenizer, image_processor, max_length=128):
        """수식 이미지에서 LaTeX를 예측합니다.
        """
        model.eval()
        pixel_values = image_processor(image, return_tensors='pt')['pixel_values'].to(device)  # 이미지 처리
        prompt = 'LaTeX: '
        input_ids = tokenizer(prompt, return_tensors='pt')['input_ids'].to(device)
        with torch.no_grad():  # 프롬프트로 시작
            vision_outputs = model.vision_encoder(pixel_values=pixel_values)
            image_features = vision_outputs.last_hidden_state
            image_embeds = model.projector(image_features)
            generated_tokens = []  # 생성
            for _ in range(max_length):
                text_embeds = model.language_model.get_input_embeddings()(input_ids)  # 텍스트 임베딩 추출
                combined_embeds = torch.cat([image_embeds, text_embeds], dim=1)
                image_attention = torch.ones(1, image_embeds.shape[1], dtype=torch.long, device=device)
                text_attention = torch.ones_like(input_ids)
                combined_attention = torch.cat([image_attention, text_attention], dim=1)
                outputs = model.language_model(inputs_embeds=combined_embeds, attention_mask=combined_attention, return_dict=True)  # 자기회귀 생성
                logits = model.lm_head(outputs.last_hidden_state)
                next_token_logits = logits[:, -1, :]
                next_token = torch.argmax(next_token_logits, dim=-1)  # 텍스트 임베딩 추출
                if next_token.item() == tokenizer.eos_token_id:
                    break
                generated_tokens.append(next_token.item())  # 연결
                input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
            predicted_latex = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        return predicted_latex  # 어텐션 마스크
    test_equations = ['\\int_0^\\pi \\sin(x) dx = 2', '\\sum_{k=0}^n \\binom{n}{k} = 2^n', 'e^{i\\pi} + 1 = 0']
    _fig, _axes = plt.subplots(len(test_equations), 1, figsize=(10, 3 * len(test_equations)))
    if len(test_equations) == 1:
        _axes = [_axes]
    for _i, _latex in enumerate(test_equations):  # 순전파
        _img = latex_to_image(_latex, fontsize=24)
        _predicted = predict_latex(equation_vlm, _img, tokenizer, image_processor)
        _axes[_i].imshow(_img)
        _axes[_i].set_title(f'정답: {_latex}\n예측: {_predicted}', fontsize=10)
        _axes[_i].axis('off')
    plt.tight_layout()
    # 새 수식으로 테스트
    plt.show()  # EOS 확인 / 디코딩 / 이미지 생성 / 예측 / 표시
    return (execution_step_9,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. 문서 구조 이해

    VLM은 수식을 넘어 **문서 배치**를 이해하고 Markdown이나 LaTeX 같은 구조화된 형식으로 변환할 수 있습니다.

    다음을 추출하는 예시를 살펴봅니다.
    - **제목**과 계층 구조
    - **표**와 LaTeX 변환
    - **목록**: 글머리표·번호 목록
    - **코드 블록**
    """)
    return


@app.cell
def _(Image, ImageDraw, ImageFont, execution_step_9, plt):
    # 원본 코드 셀 순서와 모델·데이터 변경의 실행 순서를 보존합니다.
    execution_step_10 = execution_step_9 + 1
    # 합성 문서 이미지 생성
    def create_document_image():
        """여러 요소를 담은 간단한 문서 이미지를 만듭니다.
        """
        img = Image.new('RGB', (800, 600), color='white')
        draw = ImageDraw.Draw(img)
    
        try:
            font_title = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 28)
            font_heading = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
            font_text = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
        except:
            font_title = font_heading = font_text = ImageFont.load_default()
    
        y = 30
    
        # 제목
        draw.text((50, y), "Research Paper: Deep Learning", fill='black', font=font_title)
        y += 50
    
        # 1절
        draw.text((50, y), "1. Introduction", fill='black', font=font_heading)
        y += 35
        draw.text((70, y), "Deep learning has revolutionized AI.", fill='black', font=font_text)
        y += 30
        draw.text((70, y), "Key contributions:", fill='black', font=font_text)
        y += 25
        draw.text((90, y), "• Novel architecture design", fill='black', font=font_text)
        y += 25
        draw.text((90, y), "• Improved training efficiency", fill='black', font=font_text)
        y += 40
    
        # 표가 포함된 2절
        draw.text((50, y), "2. Results", fill='black', font=font_heading)
        y += 35
    
        # 간단한 표 그리기
        table_x = 70
        draw.rectangle([table_x, y, table_x+400, y+100], outline='black', width=2)
        draw.line([table_x, y+30, table_x+400, y+30], fill='black', width=1)
        draw.line([table_x+200, y, table_x+200, y+100], fill='black', width=1)
    
        draw.text((table_x+50, y+5), "Model", fill='black', font=font_text)
        draw.text((table_x+250, y+5), "Accuracy", fill='black', font=font_text)
        draw.text((table_x+50, y+40), "ResNet-50", fill='black', font=font_text)
        draw.text((table_x+250, y+40), "92.3%", fill='black', font=font_text)
        draw.text((table_x+50, y+70), "Our Model", fill='black', font=font_text)
        draw.text((table_x+250, y+70), "95.1%", fill='black', font=font_text)
    
        return img

    doc_image = create_document_image()
    plt.figure(figsize=(10, 7))
    plt.imshow(doc_image)
    plt.title('합성 문서')
    plt.axis('off')
    plt.show()
    return (execution_step_10,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. 표 추출 및 LaTeX 생성

    표는 전통적 OCR에서 특히 처리하기 어렵습니다. VLM은 다음을 수행할 수 있습니다.
    1. 표의 행·열·셀 구조 탐지
    2. 셀 내용 추출
    3. LaTeX 표 코드 생성

    미세 조정한 VLM을 활용하는 접근을 살펴봅니다.
    """)
    return


@app.cell
def _(execution_step_10):
    # 원본 코드 셀 순서와 모델·데이터 변경의 실행 순서를 보존합니다.
    execution_step_11 = execution_step_10 + 1
    # 예시: 표를 LaTeX로 변환
    # 학습 데이터는 표 이미지와 LaTeX 코드의 쌍으로 구성

    table_latex_example = r"""
    \begin{tabular}{|l|r|}
    \hline
    Model & Accuracy \\
    \hline
    ResNet-50 & 92.3\% \\
    Our Model & 95.1\% \\
    \hline
    \end{tabular}
    """

    print('표의 예상 LaTeX 출력:')
    print(table_latex_example)

    # 실제 서비스에서는 다음을 수행
    # 1. (table_image, latex_code) 쌍으로 VLM 미세 조정
    # 2. PubTables-1M, FinTabNet 같은 데이터셋 사용
    # 3. 표 이미지에서 LaTeX 구조 예측
    return (execution_step_11,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. 비교: VLM과 전통적 OCR

    여러 문서 유형에서 VLM 기반 OCR과 전통적 Tesseract를 비교합니다.
    """)
    return


@app.cell
def _(execution_step_11, iam_dataset, ocr_image_trocr, plt):
    # 원본 코드 셀 순서와 모델·데이터 변경의 실행 순서를 보존합니다.
    execution_step_12 = execution_step_11 + 1
    try:
        import pytesseract
        tesseract_available = True
    except ImportError:
        tesseract_available = False
        print('Tesseract를 사용할 수 없습니다. 프로젝트에서 uv sync --locked --extra ocr를 실행하고 시스템 Tesseract를 설치하세요.')

    def compare_ocr_methods(image, ground_truth):
        """TrOCR과 Tesseract를 비교합니다.
        """
        results = {}
    
        # TrOCR 모델
        trocr_text = ocr_image_trocr(image)
        results['TrOCR'] = trocr_text
    
        # Tesseract 엔진
        if tesseract_available:
            tesseract_text = pytesseract.image_to_string(image)
            results['Tesseract'] = tesseract_text.strip()
    
        results['Ground Truth'] = ground_truth
    
        return results

    # 손글씨 샘플 테스트
    if len(iam_dataset) > 0:
        test_sample = iam_dataset[0]
        comparison = compare_ocr_methods(test_sample['image'], test_sample['text'])
    
        print('\nOCR 비교:')
        for method, text in comparison.items():
            print(f"\n{method}:")
            print(f"  '{text}'")
    
        plt.figure(figsize=(8, 3))
        plt.imshow(test_sample['image'], cmap='gray')
        plt.title('테스트 이미지')
        plt.axis('off')
        plt.show()
    return (execution_step_12,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 정리: VLM의 OCR 기능

    VLM 기반 OCR에 관해 배운 내용을 정리합니다.
    """)
    return


@app.cell
def _(execution_step_12):
    # 원본 코드 셀 순서와 모델·데이터 변경의 실행 순서를 보존합니다.
    execution_step_13 = execution_step_12 + 1
    import pandas as pd

    comparison_data = {
        '작업': [
            '인쇄 텍스트 OCR',
            '손글씨 텍스트 OCR',
            '수식',
            '표 구조',
            '문서 배치',
            '코드 인식',
        ],
        '전통적 OCR(Tesseract)': [
            '양호(95%+)',
            '낮음(60~70%)',
            '매우 낮음(30% 미만)',
            '낮음(후처리 필요)',
            '낮음(줄 단위 처리만)',
            '보통(70~80%)',
        ],
        'VLM 기반 OCR(TrOCR, Nougat)': [
            '우수(97%+)',
            '양호(85~90%)',
            '양호(미세 조정 시 80~90%)',
            '양호(전문 모델 사용)',
            '우수(계층 구조 이해)',
            '우수(구문 고려)',
        ],
        '적합한 모델·접근': [
            'TrOCR-base(인쇄)',
            'TrOCR-large(손글씨)',
            'Pix2Tex, IM2LATEX 기반 맞춤 VLM',
            'Nougat, PubTables-1M 기반 맞춤 VLM',
            "Nougat, LayoutLM",
            'CodeBERT, 맞춤 VLM',
        ],
    }

    df = pd.DataFrame(comparison_data)
    print('\nOCR 작업 비교:')
    print(df.to_string(index=False))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 핵심 내용

    ### 1. **문맥을 고려하는 OCR에서의 VLM 강점**
    - 수식, 코드, 표의 의미적 구조 이해
    - 언어 문맥으로 모호함 해소
    - LaTeX, Markdown, JSON 등 구조화된 출력 생성

    ### 2. **수식 OCR의 활용 가치**
    - 전통적 OCR은 수학 표기에서 어려움을 겪음
    - 원문은 이미지 → LaTeX 변환 정확도로 80~90%를 제시
    - 좋은 결과를 위해 IM2LATEX-100K로 미세 조정

    ### 3. **단순 문자 인식을 넘어선 문서 이해**
    - 제목, 목록, 표 등 문서 구조 보존
    - 형식을 갖춘 LaTeX 문서 생성 가능
    - Nougat은 학술 논문에서 높은 성능을 보여 주는 사례

    ### 4. **미세 조정의 중요성**
    - 기반 모델도 유용하지만 분야별 미세 조정이 중요
    - (image, structured_text) 쌍 데이터셋 필요
    - 소형 모델(매개변수 135M)로도 좋은 결과 가능

    ### 5. **상용 서비스 고려 사항**
    - 비용·품질의 균형을 위해 VLM과 전통적 OCR 결합
    - 수식, 표, 손글씨 같은 복잡한 사례에 VLM 사용
    - 단순 인쇄 텍스트에는 더 빠르고 저렴한 Tesseract 사용

    ## 실무 적용 방향

    **학술 논문 OCR:**
    - 해당 작업을 위해 설계된 **Nougat** 사용
    - LaTeX 수식이 포함된 Markdown 출력
    - 다단 배치, 표, 수식 처리

    **수학 과제·필기:**
    - IM2LATEX나 손글씨용 CROHME로 VLM 미세 조정
    - ViT 인코더 + GPT 디코더 방식 활용
    - 회전·잡음 등 데이터 증강으로 강건성 향상

    **업무 문서:**
    - 양식·송장에 LayoutLM이나 Donut 활용
    - 규칙 기반 추출과 결합
    - 모호하거나 복잡한 사례에 VLM 사용

    **여러 유형이 섞인 문서:**
    - 단계적 처리: Tesseract → 신뢰도가 낮으면 TrOCR → 수식이면 수식용 VLM
    - 비용 효율과 정확도를 함께 고려

    ## 학습용 데이터셋

    - **IM2LATEX-100K**: 수식 → LaTeX(https://zenodo.org/record/56198)
    - **CROHME**: 손글씨 수식 → LaTeX
    - **IAM Handwriting**: 손글씨 텍스트 줄
    - **PubTables-1M**: 논문의 표 → 구조
    - **arXiv 데이터셋**: LaTeX 소스가 있는 전체 논문

    ## 더 읽을 자료

    - TrOCR: https://arxiv.org/abs/2109.10282
    - Nougat: https://arxiv.org/abs/2308.13418
    - Pix2Tex: https://github.com/lukas-blecher/LaTeX-OCR
    - LayoutLM: https://arxiv.org/abs/1912.13318
    - Donut: https://arxiv.org/abs/2111.15664
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
