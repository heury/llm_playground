import marimo

__generated_with = "0.25.2"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Chapter_3_Excercise_Attention

    원본 `Chapter_3_Excercise_Attention.ipynb`의 설명·모델·학습 설정을 유지한 marimo 변환본입니다.
    `gpt/`에서 `uv sync --locked` 후 `uv run marimo edit "Chapter_3_Excercise_Attention.py"`로 실행하세요.
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
    output_dir = prepare_run('chapter_3_excercise_attention')
    mo.md(f"실행 산출물: `{output_dir}`")
    return (mo,)


@app.cell
def _():
    ### 중요: 이 코드는 인과적 어텐션 메커니즘을 구현한 PyTorch 모듈입니다.
    import torch
    import torch.nn as nn

    class CausalAttention(nn.Module):
        def __init__(self, d_in, d_out, context_length, dropout, qkv_bias=False):
            """
            Args:
                d_in: 입력 벡터의 차원 크기
                d_out: 출력(및 쿼리/키/밸류) 벡터의 차원 크기
                context_length: 모델이 한 번에 처리할 수 있는 최대 문맥 길이 (토큰 수)
                dropout: 드롭아웃 확률
                qkv_bias: 선형 레이어에 편향(bias)을 사용할지 여부
            """
            super().__init__()
        
            # 1. 쿼리(Query), 키(Key), 밸류(Value)를 만들기 위한 선형 투영 레이어 정의
            # 입력 벡터(d_in)를 각각의 목적에 맞는 벡터(d_out)로 변환합니다.
            self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
            self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
            self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

            # 2. 과적합(Overfitting) 방지를 위한 드롭아웃 설정
            self.dropout = nn.Dropout(dropout)

            # 3. 인과적 마스크(Causal Mask) 생성 및 버퍼 등록
            # 'register_buffer'를 사용하면 역전파(학습) 대상은 아니지만, 모델의 상태(state_dict)로 저장됩니다.
            # torch.triu(..., diagonal=1): 대각선(0) 위쪽 삼각형 부분만 1로 채웁니다.
            # 즉, '미래의 정보' 위치에 1을 표시하여 나중에 가릴(masking) 준비를 합니다.
            self.register_buffer(
                'mask', 
                torch.triu(torch.ones(context_length, context_length), diagonal=1)
            )

        def forward(self, x):
            # x.shape: [배치 크기(b), 토큰 개수(num_tokens), 입력 차원(d_in)]
            b, num_tokens, d_in = x.shape 

            # 입력 x를 통과시켜 현재 시점의 관심사(Query), 검색 대상(Key), 정보 내용(Value)을 추출합니다.
            # region [Q, K, V 벡터 계산]
            keys = self.W_key(x)        # Shape: [b, num_tokens, d_out]
            queries = self.W_query(x)   # Shape: [b, num_tokens, d_out]
            values = self.W_value(x)    # Shape: [b, num_tokens, d_out]
            # endregion

            # Query와 Key의 내적(Dot Product)을 통해 각 토큰 간의 관련성을 구합니다.
            # keys.transpose(1, 2): 행렬 곱을 위해 차원을 뒤집습니다. (d_out 차원끼리 곱해짐)
            # region [어텐션 스코어(유사도) 계산]
            attn_scores = queries @ keys.transpose(1, 2) 
            #endregion

            # mask가 1인 위치(미래 시점의 토큰들)를 -무한대(-inf)로 채웁니다.
            # 이렇게 하면 나중에 Softmax를 거칠 때 확률이 0이 되어, 미래 정보를 참조하지 못하게 됩니다.
            # [:num_tokens, :num_tokens]: 입력 길이가 context_length보다 짧을 때를 대비해 크기를 맞춥니다.
            # region [인과적 마스킹 (Masking) - 미래 정보 차단]
            attn_scores.masked_fill_(
                self.mask.bool()[:num_tokens, :num_tokens], 
                -torch.inf
            ) 
            # endregion

            # 스케일링(/ keys.shape[-1]**0.5): 차원이 커질수록 내적 값이 커져 기울기 소실이 오는 것을 방지합니다.
            # Softmax: 점수를 확률(0~1 사이, 합은 1)로 변환합니다.
            # region [어텐션 가중치(Weights) 계산 및 스케일링]
            attn_weights = torch.softmax(
                attn_scores / keys.shape[-1]**0.5, dim=-1
            )
            # endregion

            # 계산된 가중치 중 일부를 무작위로 0으로 만들어 모델이 특정 토큰에만 의존하는 것을 막습니다.
            # region [드롭아웃 적용]
            attn_weights = self.dropout(attn_weights)
            # endregion

            # 어텐션 가중치(확률)를 기반으로 Value(정보)들을 가중 합산합니다.
            # 결과적으로 "현재 토큰과 관련이 깊은 과거 토큰들의 정보"가 진하게 섞인 벡터가 됩니다.
            # region [문맥 벡터(Context Vector) 생성]
            context_vec = attn_weights @ values 
            # endregion
        
            return context_vec

    return CausalAttention, nn, torch


@app.cell
def _(CausalAttention, torch):
    inputs = torch.tensor([[0.43, 0.15, 0.89], [0.55, 0.87, 0.66], [0.57, 0.85, 0.64], [0.22, 0.58, 0.33], [0.77, 0.25, 0.1], [0.05, 0.8, 0.55]])
    _d_in = inputs.shape[1]
    _d_out = 2
    batch = torch.stack((inputs, inputs), dim=0)
    torch.manual_seed(123)
    _context_length = batch.shape[1]
    print('context_length:', _context_length)
    ca = CausalAttention(_d_in, _d_out, _context_length, 0.0)
    _context_vecs = ca(batch)
    print(_context_vecs)
    print('context_vecs.shape:', _context_vecs.shape)
    return (batch,)


@app.cell
def _(CausalAttention, batch, nn, torch):
    class MultiHeadAttentionWrapper(nn.Module):

        def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
            super().__init__()  # 1. d_out이 num_heads로 나누어떨어지는지 확인
            assert d_out % num_heads == 0, 'd_out must be divisible by num_heads'
            self.head_dim = d_out // num_heads
            self.heads = nn.ModuleList([CausalAttention(d_in, self.head_dim, context_length, dropout, qkv_bias) for _ in range(num_heads)])  # 각 헤드가 담당할 차원 (2 / 2 = 1)
            self.out_proj = nn.Linear(d_out, d_out)
      # 2. 각 헤드의 출력 차원을 head_dim으로 설정
        def forward(self, x):
            context_vec = torch.cat([head(x) for head in self.heads], dim=-1)
            return self.out_proj(context_vec)
    torch.manual_seed(123)
    _context_length = batch.shape[1]
    _d_in, _d_out = (3, 2)  # 3. 최종 출력 차원은 d_out과 일치하도록 설정
    _mha = MultiHeadAttentionWrapper(_d_in, _d_out, _context_length, 0.0, num_heads=2)
    _context_vecs = _mha(batch)
    print(_context_vecs)
    print('context_vecs.shape:', _context_vecs.shape)  # 각 헤드의 결과(2, 6, 1)를 합치면 (2, 6, 2)가 됨  # This is the number of tokens
    return


@app.cell
def _(batch, nn, torch):
    class MultiHeadAttention(nn.Module):

        def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
            super().__init__()
            assert d_out % num_heads == 0, 'd_out must be divisible by num_heads'
            self.d_out = d_out
            self.num_heads = num_heads
            self.head_dim = d_out // num_heads
            self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)  # Reduce the projection dim to match desired output dim
            self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
            self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
            self.out_proj = nn.Linear(d_out, d_out)
            self.dropout = nn.Dropout(dropout)
            self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1))  # Linear layer to combine head outputs

        def forward(self, x):
            b, num_tokens, d_in = x.shape
            keys = self.W_key(x)
            queries = self.W_query(x)
            values = self.W_value(x)
            keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
            values = values.view(b, num_tokens, self.num_heads, self.head_dim)
            queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)
            keys = keys.transpose(1, 2)  # As in `CausalAttention`, for inputs where `num_tokens` exceeds `context_length`, 
            queries = queries.transpose(1, 2)  # this will result in errors in the mask creation further below. 
            values = values.transpose(1, 2)  # In practice, this is not a problem since the LLM (chapters 4-7) ensures that inputs  
            attn_scores = queries @ keys.transpose(2, 3)  # do not exceed `context_length` before reaching this forward method.
            mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
            attn_scores.masked_fill_(mask_bool, -torch.inf)  # Shape: (b, num_tokens, d_out)
            attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
            attn_weights = self.dropout(attn_weights)
            context_vec = (attn_weights @ values).transpose(1, 2)
            context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)  # We implicitly split the matrix by adding a `num_heads` dimension
            context_vec = self.out_proj(context_vec)  # Unroll last dim: (b, num_tokens, d_out) -> (b, num_tokens, num_heads, head_dim)
            return context_vec
    torch.manual_seed(123)
    batch_size, _context_length, _d_in = batch.shape
    _d_out = 2
    _mha = MultiHeadAttention(_d_in, _d_out, _context_length, 0.0, num_heads=2)  # Transpose: (b, num_tokens, num_heads, head_dim) -> (b, num_heads, num_tokens, head_dim)
    _context_vecs = _mha(batch)
    print(_context_vecs)
    print('context_vecs.shape:', _context_vecs.shape)  # Compute scaled dot-product attention (aka self-attention) with a causal mask  # Dot product for each head  # Original mask truncated to the number of tokens and converted to boolean  # Use the mask to fill attention scores  # Shape: (b, num_tokens, num_heads, head_dim)  # Combine heads, where self.d_out = self.num_heads * self.head_dim  # optional projection
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <img src="images/llm_from_scratch/ch03_compressed/26.webp" width="600px">
    """)
    return


if __name__ == "__main__":
    app.run()
