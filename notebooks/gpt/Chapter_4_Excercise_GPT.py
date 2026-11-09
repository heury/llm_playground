import marimo

__generated_with = "0.25.2"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Chapter_4_Excercise_GPT

    원본 `Chapter_4_Excercise_GPT.ipynb`의 설명·모델·학습 설정을 유지한 marimo 변환본입니다.
    `gpt/`에서 `uv sync --locked` 후 `uv run marimo edit "Chapter_4_Excercise_GPT.py"`로 실행하세요.
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
    output_dir = prepare_run('chapter_4_excercise_gpt')
    mo.md(f"실행 산출물: `{output_dir}`")
    return (mo,)


@app.cell
def _():
    import tiktoken
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader

    #####################################
    # Chapter 2: 데이터 로딩 및 처리
    #####################################

    class GPTDatasetV1(Dataset):
        """
        GPT 학습을 위한 데이터셋 클래스입니다.
        텍스트를 입력받아 토큰화하고, 입력(input)과 타겟(target) 쌍을 만듭니다.
        """
        def __init__(self, txt, tokenizer, max_length, stride):
            self.input_ids = []
            self.target_ids = []

            # 1. 전체 텍스트를 토큰화합니다.
            # <|endoftext|> 같은 특수 토큰도 허용하여 인코딩합니다.
            token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})

            # 2. 슬라이딩 윈도우 방식으로 데이터를 조각냅니다.
            # stride만큼 이동하면서 max_length 길이의 덩어리(chunk)를 만듭니다.
            for i in range(0, len(token_ids) - max_length, stride):
                input_chunk = token_ids[i:i + max_length]
                # 타겟은 입력보다 한 칸 뒤의 토큰들입니다 (다음 토큰 예측 과제).
                target_chunk = token_ids[i + 1: i + max_length + 1]
            
                # 텐서로 변환하여 저장
                self.input_ids.append(torch.tensor(input_chunk))
                self.target_ids.append(torch.tensor(target_chunk))

        def __len__(self):
            # 데이터셋의 총 샘플 수 반환
            return len(self.input_ids)

        def __getitem__(self, idx):
            # 특정 인덱스의 입력과 타겟 쌍 반환
            return self.input_ids[idx], self.target_ids[idx]


    def create_dataloader_v1(txt, batch_size=4, max_length=256,
                             stride=128, shuffle=True, drop_last=True, num_workers=0):
        """
        텍스트 데이터를 받아 학습에 사용할 DataLoader를 생성하는 함수입니다.
        """
        # 토크나이저 초기화 (GPT-2용 BPE 인코딩 사용)
        tokenizer = tiktoken.get_encoding("gpt2")

        # 데이터셋 생성
        dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

        # 데이터로더 생성 (배치 단위로 데이터를 묶어주고 셔플링 수행)
        dataloader = DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)

        return dataloader

    return nn, tiktoken, torch


@app.cell
def _(nn, torch):
    #####################################
    # Chapter 3: 어텐션 메커니즘 (모델의 핵심)
    #####################################
    class MultiHeadAttention(nn.Module):
        """
        멀티 헤드 셀프 어텐션 (Multi-Head Self-Attention) 모듈입니다.
        입력 데이터 간의 관계성을 여러 관점(Head)에서 병렬로 학습합니다.
        """
        def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
            super().__init__()
            assert d_out % num_heads == 0, "출력 차원(d_out)은 헤드 수(num_heads)로 나누어 떨어져야 합니다."

            self.d_out = d_out
            self.num_heads = num_heads
            self.head_dim = d_out // num_heads  # 각 헤드가 담당할 차원 크기

            # Query, Key, Value를 만들기 위한 선형 투영 레이어들
            self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
            self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
            self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        
            # 멀티 헤드 결과를 하나로 합친 후 통과시키는 출력 레이어
            self.out_proj = nn.Linear(d_out, d_out) 
            self.dropout = nn.Dropout(dropout)
        
            # Causal Mask (인과적 마스킹) 생성: 미래의 토큰을 보지 못하게 함
            # 상삼각 행렬(대각선 위쪽)을 1로 채워서 나중에 마스킹에 사용
            self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))

        def forward(self, x):
            b, num_tokens, d_in = x.shape # b: 배치 크기, num_tokens: 시퀀스 길이

            # 1. Q, K, V 계산
            keys = self.W_key(x)     # Shape: (b, num_tokens, d_out)
            queries = self.W_query(x)
            values = self.W_value(x)

            # 2. 헤드 나누기 (Multi-head splitting)
            # 차원을 변형하여 여러 헤드가 병렬로 처리할 수 있게 함
            # (b, num_tokens, d_out) -> (b, num_tokens, num_heads, head_dim)
            keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
            values = values.view(b, num_tokens, self.num_heads, self.head_dim)
            queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

            # 3. 차원 순서 변경 (Transpose)
            # (b, num_tokens, num_heads, head_dim) -> (b, num_heads, num_tokens, head_dim)
            # 이렇게 하면 (num_tokens, head_dim) 행렬이 헤드 개수만큼 독립적으로 존재하게 됨
            keys = keys.transpose(1, 2)
            queries = queries.transpose(1, 2)
            values = values.transpose(1, 2)

            # 4. Scaled Dot-Product Attention 계산
            # Query와 Key의 내적 (유사도 계산)
            attn_scores = queries @ keys.transpose(2, 3)  # 결과 Shape: (b, num_heads, num_tokens, num_tokens)

            # 5. 마스킹 (Masking)
            # 현재 시점보다 미래의 토큰 정보를 참조하지 못하게 -inf로 가림
            mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
            attn_scores.masked_fill_(mask_bool, -torch.inf)

            # 6. 소프트맥스 및 드롭아웃
            # 점수를 확률로 변환 (합이 1이 되도록)
            attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
            attn_weights = self.dropout(attn_weights)

            # 7. Value와의 가중치 합 (Context Vector 계산)
            # Shape: (b, num_heads, num_tokens, head_dim)
            context_vec = (attn_weights @ values).transpose(1, 2) 

            # 8. 헤드 결합 (Concatenation)
            # 나눠졌던 헤드들을 다시 원래의 d_out 차원으로 합침
            context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        
            # 9. 최종 선형 투영
            context_vec = self.out_proj(context_vec) 

            return context_vec

    return (MultiHeadAttention,)


@app.cell
def _(nn, torch):
    #####################################
    # Chapter 4: GPT 아키텍처 구성 요소
    #####################################
    class LayerNorm(nn.Module):
        """
        층 정규화 (Layer Normalization): 학습 안정성을 높임
        """
        def __init__(self, emb_dim):
            super().__init__()
            self.eps = 1e-5
            self.scale = nn.Parameter(torch.ones(emb_dim)) # 학습 가능한 스케일 파라미터 (Gamma)
            self.shift = nn.Parameter(torch.zeros(emb_dim)) # 학습 가능한 시프트 파라미터 (Beta)

        def forward(self, x):
            mean = x.mean(dim=-1, keepdim=True)
            var = x.var(dim=-1, keepdim=True, unbiased=False)
            norm_x = (x - mean) / torch.sqrt(var + self.eps)
            return self.scale * norm_x + self.shift


    class GELU(nn.Module):
        """
        GELU 활성화 함수: GPT 계열에서 주로 사용하는 비선형 함수
        """
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return 0.5 * x * (1 + torch.tanh(
                torch.sqrt(torch.tensor(2.0 / torch.pi)) *
                (x + 0.044715 * torch.pow(x, 3))
            ))


    class FeedForward(nn.Module):
        """
        피드 포워드 네트워크 (Feed-Forward Network)
        어텐션이 모은 정보를 각 토큰별로 개별적으로 가공하는 역할
        보통 임베딩 차원을 4배로 늘렸다가 다시 줄임
        """
        def __init__(self, cfg):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
                GELU(),
                nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
            )

        def forward(self, x):
            return self.layers(x)

    return FeedForward, LayerNorm


@app.cell
def _(FeedForward, LayerNorm, MultiHeadAttention, nn, torch):
    ##중요 : 아래는 GPT 모델 전체 구조를 정의하는 코드입니다

    #####################################
    # Chapter 4: GPT 아키텍처 조립
    #####################################
    class TransformerBlock(nn.Module):
        """
        표준 트랜스포머 블록 (Decoder Block)
        구조: LayerNorm -> Attention -> Add(Residual) -> LayerNorm -> FeedForward -> Add(Residual)
        """
        def __init__(self, cfg):
            super().__init__()
            self.att = MultiHeadAttention(
                d_in=cfg["emb_dim"],
                d_out=cfg["emb_dim"],
                context_length=cfg["context_length"],
                num_heads=cfg["n_heads"],
                dropout=cfg["drop_rate"],
                qkv_bias=cfg["qkv_bias"])
            self.ff = FeedForward(cfg)
            self.norm1 = LayerNorm(cfg["emb_dim"])
            self.norm2 = LayerNorm(cfg["emb_dim"])
            self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

        def forward(self, x):
            # region [어텐션 블록 (Residual Connection 적용)]
            shortcut = x
            x = self.norm1(x) # Pre-LayerNorm 방식
            x = self.att(x)   
            x = self.drop_shortcut(x)
            x = x + shortcut  # 원본 입력을 더해줌 (기울기 소실 방지)
            # endregion

            # region [피드 포워드 블록 (Residual Connection 적용)]
            shortcut = x
            x = self.norm2(x)
            x = self.ff(x)
            x = self.drop_shortcut(x)
            x = x + shortcut  # 원본 입력 다시 더함
            # endregion

            return x
    class GPTModel(nn.Module):
        """
        전체 GPT 모델 구조 정의
        Embedding -> Transformer Blocks -> Final Norm -> Output Head
        """
        def __init__(self, cfg):
            super().__init__()
            # 토큰 임베딩 (단어 -> 벡터)
            self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
            # 위치 임베딩 (위치 정보 -> 벡터)
            self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
            self.drop_emb = nn.Dropout(cfg["drop_rate"])

            # 트랜스포머 블록 쌓기 (n_layers 만큼)
            self.trf_blocks = nn.Sequential(
                *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])

            # 최종 정규화 및 출력 헤드
            self.final_norm = LayerNorm(cfg["emb_dim"])
            self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

        def forward(self, in_idx):
            batch_size, seq_len = in_idx.shape
        
            # region [임베딩 생성]
            tok_embeds = self.tok_emb(in_idx)
            pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
            # endregion

            # region [토큰 임베딩과 위치 임베딩 합산]
            x = tok_embeds + pos_embeds 
            # endregion

            x = self.drop_emb(x)
        
            # region [트랜스포머 블록 통과]
            x = self.trf_blocks(x)
            # endregion

            # region [최종 출력 계산]
            x = self.final_norm(x)
            logits = self.out_head(x) # 각 단어에 대한 예측 점수 (Logits)
            # endregion
        
            return logits

    return (GPTModel,)


@app.cell
def _(torch):
    def generate_text_simple(model, idx, max_new_tokens, context_size):
        """
        간단한 텍스트 생성 루프
        현재 문맥을 넣어 다음 토큰을 예측하고, 이를 다시 문맥에 추가하여 반복함
        """
        # idx: 현재 문맥의 토큰 인덱스들 (Batch, Time)
        for _ in range(max_new_tokens):

            # 모델이 지원하는 최대 길이(context_size)를 넘지 않도록 자름
            idx_cond = idx[:, -context_size:]

            # 모델 예측 (기울기 계산 불필요)
            with torch.no_grad():
                logits = model(idx_cond)

            # 마지막 타임스텝의 예측값만 가져옴 (다음 단어 예측이므로)
            # (batch, n_token, vocab_size) -> (batch, vocab_size)
            logits = logits[:, -1, :]

            # 가장 확률(로짓값)이 높은 토큰 선택 (Greedy Decoding)
            idx_next = torch.argmax(logits, dim=-1, keepdim=True) 

            # 예측된 토큰을 현재 시퀀스 뒤에 이어 붙임
            idx = torch.cat((idx, idx_next), dim=1) 

        return idx

    return (generate_text_simple,)


@app.cell
def _(GPTModel, generate_text_simple, tiktoken, torch):
    def main():
        # GPT-2 Small 모델 설정값 (124M 파라미터)
        GPT_CONFIG_124M = {
            "vocab_size": 50257,     # 단어 집합 크기
            "context_length": 1024,  # 최대 문맥 길이
            "emb_dim": 768,          # 임베딩 차원
            "n_heads": 12,           # 어텐션 헤드 수
            "n_layers": 12,          # 레이어 수
            "drop_rate": 0.1,        # 드롭아웃 비율
            "qkv_bias": False        # Q,K,V 편향 사용 여부
        }

        torch.manual_seed(123)
    
        # 모델 초기화 및 평가 모드 설정 (드롭아웃 비활성화)
        model = GPTModel(GPT_CONFIG_124M)
        model.eval()

        start_context = "Hello, I am"

        # 입력 텍스트 인코딩
        tokenizer = tiktoken.get_encoding("gpt2")
        encoded = tokenizer.encode(start_context)
        encoded_tensor = torch.tensor(encoded).unsqueeze(0) # 배치 차원 추가

        print(f"\n{50*'='}\n{22*' '}IN\n{50*'='}")
        print("\nInput text:", start_context)
        print("Encoded input text:", encoded)
        print("encoded_tensor.shape:", encoded_tensor.shape)

        # 텍스트 생성 실행
        out = generate_text_simple(
            model=model,
            idx=encoded_tensor,
            max_new_tokens=10,  # 10개의 새로운 토큰 생성
            context_size=GPT_CONFIG_124M["context_length"]
        )
    
        # 생성된 결과 디코딩
        decoded_text = tokenizer.decode(out.squeeze(0).tolist())

        print(f"\n\n{50*'='}\n{22*' '}OUT\n{50*'='}")
        print("\nOutput:", out)
        print("Output length:", len(out[0]))
        print("Output text:", decoded_text)

    main()
    return


if __name__ == "__main__":
    app.run()
