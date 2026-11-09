import marimo

__generated_with = "0.25.2"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Chapter_6_Excercise_Finetuning_Classification_LoRA

    원본 `Chapter_6_Excercise_Finetuning_Classification_LoRA.ipynb`의 설명·모델·학습 설정을 유지한 marimo 변환본입니다.
    `gpt/`에서 `uv sync --locked` 후 `uv run marimo edit "Chapter_6_Excercise_Finetuning_Classification_LoRA.py"`로 실행하세요.
    데이터·토크나이저·사전 학습 가중치는 첫 실행에 다운로드될 수 있습니다.
    5~7장의 전체 학습에는 충분한 메모리와 GPU 실행 환경을 권장합니다.
    실행 산출물과 데이터는 `outputs/gpt/<실행 ID>/`, 재사용 가중치는 `gpt/models/gpt2/`에 저장합니다.
    이번 변환에서는 전체 학습을 실행하지 않았습니다.
    셀을 수정하면 의존 셀이 다시 실행되므로 학습 셀은 marimo의 지연 실행 모드에서 실행을 관리하세요.
    """)
    return


@app.cell
def _():
    import marimo as mo
    from marimo_runtime import prepare_run
    output_dir = prepare_run('chapter_6_excercise_finetuning_classification_lora')
    mo.md(f"실행 산출물: `{output_dir}`")
    return mo, output_dir


@app.cell
def _():
    from pathlib import Path
    Path('models/gpt2').mkdir(parents=True, exist_ok=True)
    import math
    import time
    import requests
    import pandas as pd
    import torch
    import torch.nn as nn
    import tiktoken
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt
    # 로컬 모듈 임포트 (previous_chapters.py가 필요합니다)
    from previous_chapters import download_and_unzip_spam_data, create_balanced_dataset, random_split, SpamDataset, GPTModel, download_model, calc_accuracy_loader, train_classifier_simple, plot_values  # 그래프 저장을 위해 추가  # 그래프 그리기 함수 추가

    return (
        DataLoader,
        GPTModel,
        Path,
        SpamDataset,
        calc_accuracy_loader,
        create_balanced_dataset,
        download_and_unzip_spam_data,
        download_model,
        math,
        nn,
        pd,
        plot_values,
        random_split,
        requests,
        tiktoken,
        time,
        torch,
        train_classifier_simple,
    )


@app.cell
def _():
    # -----------------------------------------------------------------------------
    # 1. 설정 및 하이퍼파라미터
    # -----------------------------------------------------------------------------
    BASE_CONFIG = {
        "vocab_size": 50257,
        "context_length": 1024,
        "drop_rate": 0.0,
        "qkv_bias": True,
        "emb_dim": 768, "n_layers": 12, "n_heads": 12
    }

    BATCH_SIZE = 8
    NUM_EPOCHS = 5
    LEARNING_RATE = 8e-4
    LORA_RANK = 16
    LORA_ALPHA = 16
    return (
        BASE_CONFIG,
        BATCH_SIZE,
        LEARNING_RATE,
        LORA_ALPHA,
        LORA_RANK,
        NUM_EPOCHS,
    )


@app.cell
def _(math, nn, torch):
    ###중요: LoRA 구현 부분
    # -----------------------------------------------------------------------------
    # 2. LoRA 클래스 및 함수 정의
    # -----------------------------------------------------------------------------
    class LoRALayer(nn.Module):
        def __init__(self, in_dim, out_dim, rank, alpha):
            """
            LoRA(Low-Rank Adaptation) 레이어 초기화
            """
            super().__init__()
            self.A = nn.Parameter(torch.empty(in_dim, rank))
            nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        
            self.B = nn.Parameter(torch.zeros(rank, out_dim))
        
            self.alpha = alpha
            self.rank = rank

        def forward(self, x):
            # region [LoRA 어댑터 적용]
            x = (self.alpha / self.rank) * (x @ self.A @ self.B)
            # endregion
            return x

    class LinearWithLoRA(nn.Module):
        """
        기존의 Linear 레이어를 감싸서 LoRA 어댑터를 추가한 클래스
        """
        def __init__(self, linear, rank, alpha):
            super().__init__()
            self.linear = linear
            self.lora = LoRALayer(
                linear.in_features, linear.out_features, rank, alpha
            )

        def forward(self, x):
            # region [기존 Linear와 LoRA 출력 합산]
            return self.linear(x) + self.lora(x)
            # endregion
        
    def replace_linear_with_lora(model, rank, alpha):
        """
        모델 내의 모든 Linear 레이어를 찾아 LoRA가 적용된 레이어로 교체
        """
        for name, module in model.named_children():
            if isinstance(module, torch.nn.Linear):
                # 기존 Linear 레이어를 LinearWithLoRA로 교체
                setattr(model, name, LinearWithLoRA(module, rank, alpha))
            else:
                # 재귀 호출
                replace_linear_with_lora(module, rank, alpha)

    return (replace_linear_with_lora,)


@app.cell
def _(
    BASE_CONFIG,
    BATCH_SIZE,
    DataLoader,
    GPTModel,
    LEARNING_RATE,
    LORA_ALPHA,
    LORA_RANK,
    NUM_EPOCHS,
    Path,
    SpamDataset,
    calc_accuracy_loader,
    create_balanced_dataset,
    download_and_unzip_spam_data,
    download_model,
    output_dir,
    pd,
    plot_values,
    random_split,
    replace_linear_with_lora,
    requests,
    tiktoken,
    time,
    torch,
    train_classifier_simple,
):
    def main():
        # 시드 설정
        torch.manual_seed(123)
    
        # 장치 설정
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"사용 장치: {device}")

        # --- 데이터셋 준비 ---
        print("데이터셋 준비 중...")
        url = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
        zip_path = f"{output_dir}/datas/sms_spam_collection.zip"
        extracted_path = f"{output_dir}/datas/sms_spam_collection"
        data_file_path = Path(extracted_path) / "SMSSpamCollection.tsv"

        try:
            download_and_unzip_spam_data(url, zip_path, extracted_path, data_file_path)
        except (requests.exceptions.RequestException, TimeoutError):
            # 백업 URL 시도
            url = "https://f001.backblazeb2.com/file/LLMs-from-scratch/sms%2Bspam%2Bcollection.zip"
            download_and_unzip_spam_data(url, zip_path, extracted_path, data_file_path)

        df = pd.read_csv(data_file_path, sep="\t", header=None, names=["Label", "Text"])
        balanced_df = create_balanced_dataset(df)
        balanced_df["Label"] = balanced_df["Label"].map({"ham": 0, "spam": 1})

        train_df, validation_df, test_df = random_split(balanced_df, 0.7, 0.1)
    
        # 임시 파일 저장 (Dataset 클래스 사용을 위해)
        train_df.to_csv(f"{output_dir}/datas/train.csv", index=None)
        validation_df.to_csv(f"{output_dir}/datas/validation.csv", index=None)
        test_df.to_csv(f"{output_dir}/datas/test.csv", index=None)

        tokenizer = tiktoken.get_encoding("gpt2")
        train_dataset = SpamDataset(f"{output_dir}/datas/train.csv", max_length=None, tokenizer=tokenizer)
        val_dataset = SpamDataset(f"{output_dir}/datas/validation.csv", max_length=train_dataset.max_length, tokenizer=tokenizer)
        test_dataset = SpamDataset(f"{output_dir}/datas/test.csv", max_length=train_dataset.max_length, tokenizer=tokenizer)

        train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
        val_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, drop_last=False)
        test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, drop_last=False)

        # --- 모델 준비 ---
        print("모델 로드 중...")

        file_name = "gpt2-small-124M.pth"
        model_path = f"./models/gpt2/{file_name}"
        download_model(file_name, model_path)

        # Policy Model (학습 대상)
        model = GPTModel(BASE_CONFIG)
        model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
        model.to(device)

        # 출력 헤드 교체 (분류용)
        num_classes = 2
        model.out_head = torch.nn.Linear(in_features=768, out_features=num_classes)
    
        # --- LoRA 적용 ---
        print("LoRA 레이어 적용 및 파라미터 동결 중...")
        # 1. 모든 파라미터 동결
        for param in model.parameters():
            param.requires_grad = False
    
        # 2. Linear 레이어를 LoRA로 교체 (LoRA 파라미터는 새로 생성되므로 requires_grad=True 상태)
        replace_linear_with_lora(model, rank=LORA_RANK, alpha=LORA_ALPHA)
    
        model.to(device)

        # 학습 가능 파라미터 확인
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"총 학습 가능 LoRA 파라미터 수: {total_params:,}")

        # --- 학습 실행 ---
        print("학습 시작...")
        optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.1)
    
        start_time = time.time()
        # 반환값 캡처 (loss 및 accuracy 기록)
        train_losses, val_losses, train_accs, val_accs, examples_seen = train_classifier_simple(
            model, train_loader, val_loader, optimizer, device,
            num_epochs=NUM_EPOCHS, eval_freq=50, eval_iter=5,
        )
        end_time = time.time()
        print(f"학습 완료 (소요 시간: {(end_time - start_time) / 60:.2f}분)")

        # --- 결과 시각화 (Loss) ---
        print("학습 결과 시각화 중...")
        epochs_tensor = torch.linspace(0, NUM_EPOCHS, len(train_losses))
        examples_seen_tensor = torch.linspace(0, examples_seen, len(train_losses))
    
        # plot_values 함수 호출
        plot_values(epochs_tensor, examples_seen_tensor, train_losses, val_losses, label="loss", output_dir=output_dir)
    
        # --- 최종 평가 ---
        print("최종 모델 평가 중...")
        train_acc = calc_accuracy_loader(train_loader, model, device)
        val_acc = calc_accuracy_loader(val_loader, model, device)
        test_acc = calc_accuracy_loader(test_loader, model, device)

        print(f"훈련 정확도: {train_acc*100:.2f}%")
        print(f"검증 정확도: {val_acc*100:.2f}%")
        print(f"테스트 정확도: {test_acc*100:.2f}%")

    return (main,)


@app.cell
def _(main):
    main()
    return


if __name__ == "__main__":
    app.run()
