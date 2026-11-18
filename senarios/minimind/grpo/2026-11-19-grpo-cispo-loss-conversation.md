# MiniMind GRPO·CISPO loss 계산 대화

저장일: 2026-11-19 (Asia/Seoul)

GRPO loss 계산 과정과 CISPO·GRPO의 차이에 관한 두 차례의 사용자 요청 및
어시스턴트 최종 답변을 보관합니다. 도구 호출·진행 설명과 IDE/환경 메타데이터는
제외했습니다. 소스 링크는 저장소 기준 상대 경로로 표기했습니다.
숫자 예시는 계산 설명용이며, 기울기 비교는 PyTorch로 확인했습니다.

## 사용자

grpo loss를 계산하는 과정을 간단한 코드와 함께 설명해

## 어시스턴트

GRPO loss는 **같은 질문에서 생성한 답변들을 비교해, 상대적으로 좋은 답변의 생성 확률은 높이고 나쁜 답변의 확률은 낮추는 손실**입니다.

MiniMind의 [실제 계산 코드](../../../projects/minimind/trainer/train_grpo.py#L153)를 간단히 풀면 다음과 같습니다.

**1. 답변별 보상을 상대 점수로 바꿉니다.**

한 질문에 답변 3개를 생성했다고 해보겠습니다.

| 생성한 답변 | 보상 | 상대 점수: advantage |
|---|---:|---:|
| 정답 A | 1 | 약 +0.707 |
| 오답 B | 0 | 약 −1.414 |
| 정답 C | 1 | 약 +0.707 |

```python
advantages = (
    rewards - rewards.mean()
) / (rewards.std(unbiased=False) + 1e-4)
```

그룹 평균보다 보상이 높으면 양수, 낮으면 음수입니다. **답변 하나의 advantage를 그 답변의 생성 토큰들에 적용**합니다.

**2. 답변을 생성했을 때와 현재의 토큰 확률을 비교합니다.**

```python
ratio = torch.exp(logp - old_logp)
```

- `old_logp`: 답변을 생성할 당시 모델의 토큰 로그 확률
- `logp`: 현재 학습 중인 모델의 토큰 로그 확률
- `ratio = 1.2`: 해당 토큰의 확률이 이전의 1.2배

좋은 답변은 확률을 높이는 방향으로, 나쁜 답변은 낮추는 방향으로 학습합니다.

**3. 변화가 지나치게 커지는 방향의 학습 이득을 제한합니다.**

```python
clipped_ratio = ratio.clamp(0.8, 1.2)

objective = torch.minimum(
    ratio * advantages,
    clipped_ratio * advantages,
)
```

이것이 PPO 방식의 clipping입니다. 확률 자체를 강제로 범위 안에 넣는 것은 아닙니다.

**4. 기준 모델에서 너무 멀어지는 것을 억제하고, 실제 생성 토큰만 평균냅니다.**

전체 과정을 합친 간단한 코드입니다. **질문 하나, 답변 G개**를 가정합니다.

```python
import torch

def grpo_loss(logp, old_logp, ref_logp, rewards, mask,
              epsilon=0.2, beta=0.1):
    # logp, old_logp, ref_logp, mask: [G, 생성 길이]
    # rewards: [G]
    # mask: 유효한 생성 토큰은 1, 패딩/EOS 이후는 0

    # ① 같은 질문의 답변끼리 보상 비교
    advantage = (
        rewards - rewards.mean()
    ) / (rewards.std(unbiased=False) + 1e-4)
    advantage = advantage.detach()[:, None]

    # ② 생성 당시 모델 대비 현재 모델의 확률 비율
    ratio = torch.exp(logp - old_logp.detach())

    # ③ Clipped policy objective
    clipped = ratio.clamp(1 - epsilon, 1 + epsilon)
    objective = torch.minimum(
        ratio * advantage,
        clipped * advantage,
    )

    # ④ 고정된 기준 모델과의 차이에 대한 KL 추정 페널티
    delta = ref_logp.detach() - logp
    kl = torch.exp(delta) - delta - 1

    token_loss = -objective + beta * kl

    # 답변별 유효 토큰 평균 → 답변들 평균
    response_loss = (
        (token_loss * mask).sum(dim=1)
        / mask.sum(dim=1).clamp(min=1)
    )
    return response_loss.mean()
```

여기서 `logp`는 **모델이 실제 생성한 토큰들에 부여하는 로그 확률**입니다. 반환된 loss에 `backward()`를 호출해 모델을 업데이트합니다.

처음에는 현재 모델과 생성 당시 모델이 같아 `ratio=1`이고 loss 값이 0에 가까울 수도 있습니다. 그래도 **loss의 기울기는 0이 아닐 수 있어 학습이 진행됩니다.**

현재 MiniMind의 기본 설정은 `--loss_type cispo`입니다. 위 코드는 **`--loss_type grpo` 분기**에 해당하며, 실제 학습에서는 MoE 보조 손실과 gradient accumulation 처리도 추가됩니다.

## 사용자

cispo와 grpo loss 계산하는 것을 비교해서 설명해

## 어시스턴트

MiniMind에서는 둘 다 **그룹 내 보상으로 advantage를 계산**하지만, 확률 변화를 제한하는 방식이 다릅니다.

- **GRPO:** 좋은 방향으로 확률이 충분히 바뀌면, 해당 토큰의 정책 기울기가 0이 될 수 있습니다.
- **CISPO:** 확률 비율을 제한된 **학습 가중치**로 사용하고, 로그 확률을 통해 계속 학습합니다.

아래는 [현재 MiniMind 구현](../../../projects/minimind/trainer/train_grpo.py#L164) 기준입니다.

**1. 공통으로 계산하는 값**

```python
# 같은 질문에서 생성한 답변들의 상대적 보상
advantage = (rewards - rewards.mean()) / (
    rewards.std(unbiased=False) + 1e-4
)

# 현재 확률 / 답변 생성 당시 확률
ratio = torch.exp(logp - old_logp)
```

`advantage > 0`이면 생성 확률을 높이고, `< 0`이면 낮추는 방향으로 학습합니다.

**2. GRPO: clipping한 목적함수 중 작은 값 선택**

```python
clipped_ratio = ratio.clamp(0.8, 1.2)

loss = -torch.minimum(
    ratio * advantage,
    clipped_ratio * advantage,
)
```

예를 들어 좋은 답변의 토큰에서:

```text
advantage = +1
ratio     = 1.5

원래 목적함수 = 1.5 × 1 = 1.5
제한 목적함수 = 1.2 × 1 = 1.2
선택한 값     = 1.2
```

이 구간에서는 현재 확률이 더 커져도 선택된 값이 `1.2`로 일정하므로, **그 토큰의 정책 기울기는 0**입니다.

**3. CISPO: 제한된 비율을 가중치로 사용**

```python
weight = ratio.clamp(max=5.0).detach()

loss = -weight * advantage * logp
```

여기서 핵심은 `detach()`입니다.

```text
weight: 역전파할 때 고정된 숫자로 취급
logp:   기울기가 흘러 모델을 업데이트
```

앞과 같은 조건이면:

```python
loss = -1.5 * 1.0 * logp
```

따라서 **확률을 높이는 방향의 학습이 계속됩니다.** 비율이 `6`이어도 가중치만 `5`로 제한되고, `logp`를 통한 기울기는 남습니다.

현재 구현은 비율의 **상한만 5로 제한**합니다.

**4. 기울기로 비교하면 차이가 명확합니다.**

KL 페널티를 제외하고 직접 계산한 결과입니다.

| Advantage | 확률 비율 | GRPO의 `∂loss/∂logp` | CISPO의 `∂loss/∂logp` |
|---:|---:|---:|---:|
| +1 | 1.5 | 0 | −1.5 |
| −1 | 0.5 | 0 | +0.5 |
| −1 | 1.5 | +1.5 | +1.5 |
| +1 | 6.0 | 0 | −5.0 |

경사하강에서는 기울기가 음수면 `logp`를 높이고, 양수면 낮춥니다. GRPO도 **나쁜 답변의 확률이 증가하는 등 잘못된 방향의 변화에는 기울기를 유지**합니다.

두 방식 모두 마지막에는 동일한 KL 페널티를 더하고, 패딩을 제외한 생성 토큰을 평균냅니다.

```python
delta = ref_logp - logp
kl = torch.exp(delta) - delta - 1

token_loss = policy_token_loss + beta * kl
loss = ((token_loss * mask).sum(-1)
        / mask.sum(-1).clamp(min=1)).mean()
```

여기서 `old_logp`는 **답변을 생성한 모델**, `ref_logp`는 **고정된 기준 모델**의 로그 확률입니다. GRPO의 정책 기울기가 0인 구간에서도 KL 페널티에 의한 업데이트는 남을 수 있습니다.

또한 **두 방식의 loss 숫자 자체를 서로 비교해 성능을 판단하면 안 됩니다.** 목적함수 형태가 달라, 기울기와 실제 보상·평가 성능을 확인해야 합니다.
