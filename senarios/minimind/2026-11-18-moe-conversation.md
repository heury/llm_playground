# MiniMind MoE 대화 원본

## 사용자

minimind model에서 moe 관련 코드를 이해하기 쉽게  보여줘

## 어시스턴트

기본 MiniMind 모델의 MoE 코드와 설정을 확인해서, 토큰이 전문가에게 배분되고 결과가 합쳐지는 흐름을 설명하겠습니다.

기본 MiniMind의 MoE는 **“토큰마다 사용할 FFN을 고르는 구조”**입니다. 핵심 코드는 [model_minimind.py의 `MOEFeedForward`](/home/mind4u/dev/llm_playground/projects/minimind/model/model_minimind.py:148)에 있습니다.

**1. 모델에서 MoE가 들어가는 위치**

Transformer 블록의 FFN 부분을 MoE로 교체합니다. Attention은 그대로 사용합니다.

```python
self.mlp = (
    FeedForward(config)
    if not config.use_moe
    else MOEFeedForward(config)
)
```

```text
입력
 ↓
Attention + 잔차 연결
 ↓
정규화
 ↓
MoE: 전문가 선택 → 선택된 FFN 실행 → 결과 합산
 ↓
잔차 연결
 ↓
출력
```

각 Transformer 블록은 **자기 라우터와 전문가들을 따로** 가집니다.

**2. 전문가 하나는 평범한 FFN**

[FeedForward 코드](/home/mind4u/dev/llm_playground/projects/minimind/model/model_minimind.py:136)는 다음과 같습니다.

```python
def forward(self, x):
    return self.down_proj(
        self.act_fn(self.gate_proj(x)) * self.up_proj(x)
    )
```

기본 활성화 함수가 SiLU이므로 SwiGLU 형태입니다.

```text
                  ┌→ gate_proj → SiLU ─┐
토큰 벡터 x ──────┤                    × → down_proj → 출력
                  └→ up_proj ──────────┘
```

여기서 `gate_proj`는 FFN 내부 연산입니다. 아래에서 나오는 **전문가 선택용 `self.gate`와는 다른 층**입니다.

**3. 라우터가 토큰마다 전문가를 선택**

MoE를 만드는 코드는 크게 두 부분입니다.

```python
# 토큰 벡터 → 전문가별 점수
self.gate = nn.Linear(
    config.hidden_size,
    config.num_experts,
    bias=False,
)

# 서로 다른 가중치를 가진 FFN들
self.experts = nn.ModuleList([
    FeedForward(
        config,
        intermediate_size=config.moe_intermediate_size,
    )
    for _ in range(config.num_experts)
])
```

기본 설정은 **전문가 4개, 토큰당 1개 선택**입니다. `use_moe` 자체는 기본적으로 `False`입니다.

실제 선택 과정입니다.

```python
batch_size, seq_len, hidden_dim = x.shape

# [배치, 토큰 수, 차원] → [전체 토큰 수, 차원]
x_flat = x.view(-1, hidden_dim)

# 각 토큰에 대해 전문가별 확률 계산
scores = F.softmax(self.gate(x_flat), dim=-1)

# 확률이 높은 전문가 k개 선택
topk_weight, topk_idx = torch.topk(
    scores,
    k=self.config.num_experts_per_tok,
    dim=-1,
    sorted=False,
)
```

예를 들어 한 토큰의 라우터 확률이 다음과 같다고 합시다.

| 전문가 | 확률 |
|---|---:|
| 0 | 0.10 |
| 1 | 0.60 |
| 2 | 0.25 |
| 3 | 0.05 |

- `top-1`: 전문가 1을 선택합니다.
- `top-2`: 전문가 1과 2를 선택합니다.

선택은 **문장 전체가 아니라 토큰별로** 이루어집니다. 전문가는 사전에 “수학 담당”처럼 지정하지 않으며, 학습하면서 역할이 형성될 수 있습니다.

**4. 선택한 전문가를 실행하고 원래 위치에 합산**

아래는 원본의 핵심 루프에 설명을 붙인 코드입니다.

```python
y = torch.zeros_like(x_flat)

for i, expert in enumerate(self.experts):
    # 전문가 i를 선택한 위치: [전체 토큰 수, k]
    mask = (topk_idx == i)

    if mask.any():
        # 전문가 i가 처리해야 할 토큰 번호
        token_idx = mask.any(dim=-1).nonzero().flatten()

        # 해당 토큰이 이 전문가에 부여한 가중치
        weight = topk_weight[mask].view(-1, 1)

        # 선택된 토큰만 이 전문가의 FFN에 통과
        expert_output = expert(x_flat[token_idx])

        # 가중치를 곱하고 토큰의 원래 위치에 누적
        y.index_add_(
            0,
            token_idx,
            (expert_output * weight).to(y.dtype),
        )

return y.view(batch_size, seq_len, hidden_dim)
```

`index_add_`가 중요한 이유는 **한 토큰이 여러 전문가를 선택할 수 있기 때문**입니다.

```text
토큰 A → 전문가 1 → 출력 × 0.706 ─┐
                                  ├→ 더하기 → 토큰 A의 최종 출력
토큰 A → 전문가 2 → 출력 × 0.294 ─┘
```

위 가중치는 앞의 확률 `0.60`, `0.25`를 선택된 전문가끼리 정규화한 값입니다.

```python
# top-k, k > 1일 때 기본 동작
topk_weight = topk_weight / (
    topk_weight.sum(dim=-1, keepdim=True) + 1e-20
)
```

**5. 이 구현에서 눈여겨볼 학습 처리**

**① 기본 top-1에서는 출력 가중치가 1입니다.**

단순히 `p / p`로 정규화하면 라우터로 가는 기울기가 사라지므로, 다음 특수 처리를 사용합니다.

```python
top1 = torch.topk(
    F.softmax(self.gate(x_flat.detach()), dim=-1),
    k=1,
    dim=-1,
    sorted=False,
)[0]

topk_weight = top1 - top1.detach() + 1.0
```

- 순전파: `top1 - top1.detach()`가 0이므로 가중치는 **1**입니다.
- 역전파: `top1` 쪽에는 기울기가 남아 **라우터 가중치를 학습**합니다.
- `x_flat.detach()` 때문에 이 재계산 경로에서는 입력 `x`로 기울기가 흐르지 않습니다. 전문가를 통과하는 경로에서는 입력으로 기울기가 흐릅니다.

**② 특정 전문가에만 몰리지 않도록 보조 손실을 더합니다.**

```python
load = F.one_hot(
    topk_idx, self.config.num_experts
).float().mean(0)

self.aux_loss = (
    (load * scores.mean(0)).sum()
    * self.config.num_experts
    * self.config.router_aux_loss_coef
)
```

실제 선택 빈도와 평균 라우터 확률을 이용해 전문가 사용 편중을 완화합니다. 각 층의 보조 손실을 모아 [학습 코드](/home/mind4u/dev/llm_playground/projects/minimind/trainer/train_pretrain.py:35)에서 더합니다.

```python
loss = res.loss + res.aux_loss
```

**③ 선택되지 않은 전문가도 계산 그래프에 연결합니다.**

```python
elif self.training:
    y[0, 0] += 0 * sum(p.sum() for p in expert.parameters())
```

출력값은 바꾸지 않으면서, 사용되지 않은 전문가의 매개변수도 계산 그래프에 포함하는 처리입니다.

읽을 때는 **`gate` → `topk` → 전문가별 토큰 수집 → `expert(...)` → `index_add_`** 순서로 따라가면 전체 동작을 이해하기 쉽습니다.
