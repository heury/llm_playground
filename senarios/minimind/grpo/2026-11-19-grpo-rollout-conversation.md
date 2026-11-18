# MiniMind GRPO rollout과 토큰별 loss 계산 대화

저장일: 2026-11-19 (Asia/Seoul)

GRPO rollout과 rollout engine, 토큰별 loss 계산에 관한 세 차례의 사용자 요청과
어시스턴트 최종 답변을 보관합니다. 도구 호출·진행 설명과 IDE/환경 메타데이터는
제외했습니다. 소스 링크는 저장소 기준 상대 경로로 표기했습니다.
마지막 답변에서 언급하는 DPO 설명은 [DPO loss 계산 대화](../dpo/2026-11-19-dpo-loss-conversation.md)에 있습니다.

## 사용자

grpo에서 rollout를 통해 다양한 답변을 만드는 과정을 설명해

## 어시스턴트

GRPO에서 **rollout은 현재 정책 모델로 답변을 실제 생성하는 과정**입니다. 같은 질문을 여러 개 복제하고, 각 답변의 다음 토큰을 확률적으로 뽑아 다양한 답변을 만듭니다.

MiniMind의 [rollout 코드](../../../projects/minimind/trainer/rollout_engine.py#L71)를 간단히 풀면 다음과 같습니다.

**1. 같은 질문을 여러 개 복제합니다.**

```python
# 실제 MiniMind 기본값은 질문당 6개
num_generations = 3

# [질문 A, 질문 B]
# → [질문 A, 질문 A, 질문 A, 질문 B, 질문 B, 질문 B]
input_ids = prompt_ids.repeat_interleave(num_generations, dim=0)
mask = attention_mask.repeat_interleave(num_generations, dim=0)
```

복제된 질문은 같지만, **각 행에서 토큰을 따로 샘플링**합니다.

**2. 다음 토큰을 확률적으로 선택하며 답변을 이어갑니다.**

핵심 원리는 다음과 같습니다.

```python
# 생성 원리를 보여주는 단순화 코드
with torch.no_grad():
    for _ in range(max_new_tokens):
        logits = model(input_ids).logits[:, -1, :]
        probs = torch.softmax(logits / 0.8, dim=-1)

        # 각 답변마다 다음 토큰 하나를 확률적으로 선택
        next_token = torch.multinomial(probs, num_samples=1)
        input_ids = torch.cat([input_ids, next_token], dim=1)

        # 실제 구현에서는 EOS 종료와 패딩도 처리
```

예를 들어 같은 질문에서 다음 토큰의 분포가 이렇다고 해보겠습니다.

```text
질문: 사과 2개에 3개를 더하면?

다음 토큰 확률: "5" 60%, "합계" 25%, "6" 15%

답변 1 → "5" 선택
답변 2 → "합계" 선택
답변 3 → "6" 선택
```

그다음에는 **각자 지금까지 생성한 토큰을 조건으로** 다음 토큰을 뽑습니다.

```text
답변 1: 5개입니다.
답변 2: 합계는 5개입니다.
답변 3: 6개입니다.
```

이는 원리 설명용 예시이며, 실제 생성에서는 같은 답변이 중복될 수도 있습니다.

**3. MiniMind에서는 `generate()`로 이 과정을 실행합니다.**

```python
with torch.no_grad():
    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=mask,
        max_new_tokens=256,
        do_sample=True,       # 확률에 따라 토큰 선택
        temperature=0.8,
        num_return_sequences=1,  # 질문을 이미 복제했으므로 행당 1개
    )
```

다양성을 만드는 핵심은 `do_sample=True`입니다.

- `temperature`가 낮으면 높은 확률의 토큰에 집중합니다.
- `temperature`가 높으면 낮은 확률의 토큰도 더 자주 선택합니다.
- MiniMind의 생성 함수는 기본적으로 `top_k=50`, `top_p=0.85`로 후보 토큰도 제한합니다.

**4. 생성된 답변들을 평가하고, 같은 질문 안에서 비교합니다.**

```python
# 예: 같은 질문에서 나온 답변 3개의 보상
rewards = torch.tensor([1.0, 1.0, 0.0])

advantages = (
    rewards - rewards.mean()
) / (rewards.std(unbiased=False) + 1e-4)

# 약 [+0.707, +0.707, -1.414]
```

이후 GRPO loss를 통해 상대적으로 좋은 답변의 생성 확률을 높이고, 나쁜 답변의 확률을 낮춥니다. rollout은 역전파 없이 수행하고, **학습할 때 생성된 답변을 모델에 다시 넣어 로그 확률과 기울기를 계산**합니다.

전체 흐름은 **질문 복제 → 확률적 답변 생성 → 보상 평가 → 그룹 내 비교 → 모델 업데이트**입니다.

## 사용자

grpo에서 rollout engine를 통해 다양한 답변을 생성하는 과정을 설명해

## 어시스턴트

MiniMind의 **rollout engine은 정책 모델로 질문당 여러 답변을 생성하고, 학습에 필요한 토큰과 로그 확률을 묶어 반환하는 역할**입니다. 답변의 다양성은 같은 질문에서 다음 토큰을 여러 번 확률적으로 뽑으면서 생깁니다.

**1. 학습 코드가 engine에 생성을 요청합니다.**

[train_grpo.py](../../../projects/minimind/trainer/train_grpo.py#L105)에서는 다음처럼 호출합니다.

```python
result = rollout_engine.rollout(
    prompt_ids=prompt_ids,         # 질문 토큰 [B, P]
    attention_mask=attention_mask,
    num_generations=6,             # 질문당 답변 6개
    max_new_tokens=256,            # 예시 길이 제한
    temperature=0.8,
)
```

여기서 `B`는 질문 개수, `P`는 패딩을 포함한 질문 길이입니다.

**2. engine이 질문을 답변 개수만큼 복제합니다.**

기본 [TorchRolloutEngine](../../../projects/minimind/trainer/rollout_engine.py#L71)은 다음처럼 배치를 확장합니다.

```python
expanded_ids = prompt_ids.repeat_interleave(6, dim=0)
expanded_mask = attention_mask.repeat_interleave(6, dim=0)

# 질문이 2개라면:
# [A, B]
# → [A, A, A, A, A, A, B, B, B, B, B, B]
# shape: [2, P] → [12, P]
```

같은 모델이 이 12개 입력을 배치로 처리합니다.

**3. 각 행에서 다음 토큰을 따로 샘플링합니다.**

```python
with torch.no_grad():
    output_ids = model.generate(
        input_ids=expanded_ids,
        attention_mask=expanded_mask,
        num_return_sequences=1,  # 이미 질문을 6배 복제했음
        do_sample=True,
        temperature=0.8,
        max_new_tokens=256,
    )
```

`generate()` 내부의 핵심은 다음과 같습니다.

```python
# 각 행의 다음 토큰 점수
logits = model(...).logits[:, -1, :] / temperature

# 실제 MiniMind에서는 여기서 top_k, top_p로 후보도 제한
probs = torch.softmax(logits, dim=-1)

# 같은 질문이어도 각 행에서 별도로 추첨
next_token = torch.multinomial(probs, num_samples=1)
```

처음에는 같은 질문의 확률 분포가 같아도, 추첨 결과가 달라질 수 있습니다. 한 토큰이 달라지면 이후에는 서로 다른 문맥으로 생성이 이어집니다.

```text
같은 질문: “사과 2개에 3개를 더하면?”

복제 1 → “5”    → “개” → “입니다.”
복제 2 → “합계” → “는” → “5개입니다.”
복제 3 → “6”    → “개” → “입니다.”
```

위는 원리 설명용 예시입니다. **샘플링은 서로 다른 답변을 보장하지 않으므로 중복 답변도 나올 수 있습니다.** 각 답변은 EOS를 생성하거나 최대 길이에 도달하면 끝납니다.

**4. engine이 생성 결과를 학습용으로 정리합니다.**

주요 반환값은 다음과 같습니다.

| 반환값 | 내용 |
|---|---|
| `output_ids` | 질문과 답변을 합친 토큰 |
| `completion_ids` | 생성한 답변 토큰 |
| `completions` | 사람이 읽을 수 있는 답변 문자열 |
| `per_token_logps` | 생성 토큰별 로그 확률 |
| `prompt_lens`, `completion_mask` | 질문 길이와 답변의 유효 위치 정보 |

Torch engine은 생성 후 모델을 다시 실행해 토큰별 로그 확률을 계산합니다. 학습 코드는 이를 `old_per_token_logps`로 보관하고, 업데이트할 모델의 로그 확률과 비교합니다.

이후 **답변별 보상 계산 → 같은 질문의 6개 답변끼리 advantage 계산 → GRPO loss로 업데이트**가 이어집니다. 보상 평가와 loss 계산은 engine 밖의 학습 코드가 담당합니다.

MiniMind에는 `SGLangRolloutEngine`도 있습니다. 이 엔진은 복제한 질문을 HTTP로 추론 서버에 보내 생성하고 로그 확률을 받아옵니다. 두 엔진 모두 **질문당 여러 샘플을 생성해 학습 코드에 돌려준다**는 역할은 같습니다.

## 사용자

loss계산을 할 때 token을 하나씩 예측하면서 계산해서 합산하는 거야?

## 어시스턴트

네. **각 위치에서 다음 토큰에 부여한 로그 확률을 계산**합니다. 다만 생성할 때와 loss를 계산할 때의 실행 방식이 다릅니다.

- **Rollout 생성:** 토큰을 하나씩 뽑아 답변을 이어갑니다.
- **Loss 계산:** 이미 생성된 답변 전체를 입력해, 각 위치의 다음 토큰 예측을 한 번의 순전파로 병렬 계산합니다.

예를 들어 rollout에서 `"5 개 입니다"`를 생성했다면:

```text
입력 문맥                  평가할 다음 토큰
질문                       "5"
질문 + "5"                 "개"
질문 + "5 개"              "입니다"
```

Causal attention 때문에 각 위치는 앞의 토큰만 볼 수 있습니다.

```python
# 질문과 생성된 답변 전체를 입력
logits = model(output_ids).logits

# 위치 t의 출력으로 실제 다음 토큰(t+1)의 로그 확률 계산
log_probs = logits[:, :-1].log_softmax(dim=-1)
token_logps = log_probs.gather(
    -1, output_ids[:, 1:].unsqueeze(-1)
).squeeze(-1)
```

**GRPO에서는 이 로그 확률로 토큰별 loss를 만든 다음, 답변 토큰에 대해 평균**냅니다.

```python
# 세부 GRPO 수식은 생략
token_loss = grpo_token_loss(
    current_logps, old_logps, ref_logps, advantage
)

# 질문과 패딩을 제외하고 답변별 평균 → 배치 평균
loss = (
    (token_loss * answer_mask).sum(-1)
    / answer_mask.sum(-1).clamp(min=1)
).mean()
```

즉, **토큰별로 계산한 뒤 합산하는 것은 맞지만, MiniMind GRPO는 답변 길이로 나눠 평균까지 냅니다.** 앞서 본 DPO는 답변의 토큰 로그 확률을 먼저 합산한 뒤, chosen/rejected를 비교해 loss를 계산한다는 차이가 있습니다.
