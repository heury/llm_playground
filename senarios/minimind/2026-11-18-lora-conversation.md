# MiniMind LoRA 대화 원본

## 사용자

minimind의 lora에 대해 설명해

## 어시스턴트

MiniMind의 LoRA 구현과 학습 코드를 확인하고, 어떤 가중치를 추가해 학습하는지 쉽게 설명하겠습니다.

> 이 요청은 코드 확인 후 중단되었으며, 아래 후속 요청에 이어서 설명했습니다.

## 사용자

minimind의 lora에 대해 이해하기 쉽게 설명해

## 어시스턴트

MiniMind의 LoRA는 **기존 모델의 가중치를 고정하고, 작은 보정용 가중치만 추가로 학습하는 방법**입니다.

예를 들어 일반적인 대화를 하던 MiniMind에 의료 질문·답변을 학습시킨다고 해보겠습니다. 모델 전체를 수정하는 대신, 기존 계산 옆에 작은 보정 경로를 붙입니다.

```text
                  기존 가중치 W → 기존 결과 ─┐
입력 x ──────────┤                          + → 최종 결과
                  작은 A → 작은 B → 보정값 ─┘
                  ↑ 이 두 가중치만 학습
```

MiniMind에서는 실제로 다음처럼 계산합니다.

```python
최종 결과 = 기존 레이어(x) + B(A(x))
```

**1. A와 B는 왜 작을까요?**

기존 레이어가 768차원 입력을 768차원 출력으로 바꾼다면, LoRA는 중간을 작은 차원으로 줄입니다. 이 크기를 **rank**라고 하며 기본값은 16입니다.

```text
기존 경로: 768 → 768
LoRA 경로: 768 → 16 → 768
```

| 구분 | 가중치 수 |
|---|---:|
| 기존 768 × 768 행렬 | 589,824개 |
| LoRA의 A와 B | 24,576개 |

이 예에서는 원래 행렬 크기의 약 **4.2%에 해당하는 가중치만 학습**합니다. 이는 레이어 하나의 비교이며, 전체 모델의 비율은 다릅니다.

[LoRA 구현](/home/mind4u/dev/llm_playground/projects/minimind/model/model_lora.py:6)도 간단합니다.

```python
self.A = nn.Linear(in_features, rank, bias=False)
self.B = nn.Linear(rank, out_features, bias=False)

def forward(self, x):
    return self.B(self.A(x))
```

**2. 처음에는 기존 모델과 똑같이 동작합니다.**

MiniMind는 A를 작은 난수로, **B를 모두 0으로 초기화**합니다.

```text
처음: B(A(x)) = 0
최종 결과 = 기존 결과 + 0
```

학습하면서 보정값이 생기므로, 기존 모델의 동작에서 출발해 원하는 답변 방식으로 조정됩니다.

**3. 학습할 때는 A와 B만 바뀝니다.**

[학습 코드](/home/mind4u/dev/llm_playground/projects/minimind/trainer/train_lora.py:120)는 다음 순서로 동작합니다.

```text
기존 MiniMind 가중치 불러오기
        ↓
LoRA 경로 추가
        ↓
기존 가중치 고정
        ↓
질문·정답 데이터로 학습
        ↓
LoRA의 A·B만 업데이트
```

기존 모델도 계산에는 참여합니다. 다만 **가중치 업데이트와 옵티마이저 상태를 LoRA에 집중**하므로 전체 미세 조정보다 학습 메모리를 줄일 수 있습니다.

**4. MiniMind는 어느 레이어에 붙일까요?**

현재 구현은 이름 대신 다음 조건으로 대상을 고릅니다.

```python
isinstance(module, nn.Linear) and module.in_features == module.out_features
```

즉, **입력·출력 차원이 같은 Linear 레이어**에 붙입니다. 현재 기본 설정에서는 Attention의 `q_proj`와 `o_proj`가 해당합니다. GQA 때문에 출력 차원이 더 작은 `k_proj`, `v_proj`에는 붙지 않습니다.

**5. 학습한 LoRA는 따로 저장하고 다시 사용할 수 있습니다.**

- **분리해서 사용:** 기존 모델을 불러온 뒤 LoRA 가중치를 추가로 불러옵니다.
- **병합해서 사용:** 보정값을 원래 가중치에 합쳐 일반 모델처럼 저장합니다.

병합은 다음 계산입니다.

```python
새 가중치 = 기존 가중치 + B.weight @ A.weight
```

따라서 같은 기본 MiniMind를 두고, 의료용·말투 변경용 등 **작은 LoRA 파일을 목적에 맞게 바꿔 끼우는 방식**으로 사용할 수 있습니다.

## 사용자

lora로 학습할 때 메모리가 얼마나 줄어드는지 테스트해봐

## 어시스턴트

MiniMind의 전체 미세 조정과 LoRA 학습을 같은 모델·배치·시퀀스 길이로 실행해 GPU 최대 메모리를 비교하겠습니다. 먼저 로컬 GPU와 실행 환경을 확인하겠습니다.

로컬 RTX 3080 Ti 12GB에서 CUDA 동작을 확인했습니다. MiniMind 기본 768차원·8레이어 모델에 BF16 혼합 정밀도를 적용하고, 전체 학습과 rank 16 LoRA를 비교하겠습니다. 메모리 측정용 짧은 학습을 배치 1·8·32에서 각각 반복해, 배치가 커질 때 절감률이 어떻게 달라지는지도 확인하겠습니다.

배치 1에서는 최대 GPU 메모리가 전체 학습 약 1,262MiB에서 LoRA 약 511MiB로 줄었고, 3회 반복값도 같았습니다. 현재 배치 8·32를 측정 중입니다. 사전학습 가중치 대신 같은 초기 가중치와 합성 토큰을 사용해, 학습 품질과 분리해서 메모리만 비교하고 있습니다.

18회 측정을 모두 마쳤습니다. 최대 GPU 메모리는 LoRA에서 배치 1 기준 59.5%, 배치 8 기준 40.6%, 배치 32 기준 34.0% 줄었습니다. 학습 가중치는 약 6,391만 개에서 39만 개로 줄지만, 기본 모델과 중간 계산값도 메모리를 사용하므로 전체 메모리가 같은 비율로 줄지는 않습니다. 결과와 재현 방법을 보고서로 저장하겠습니다.

실측 결과, **MiniMind 기본 배치 32에서는 LoRA가 최대 GPU 메모리를 약 34% 줄였습니다.**

조건: RTX 3080 Ti, 768차원·8레이어, 길이 340토큰, BF16 혼합 정밀도, LoRA rank 16. 각 조건을 3회 반복했습니다.

| 배치 크기 | 전체 학습 | LoRA | 메모리 절감 |
|---|---:|---:|---:|
| 1 | 1,262 MiB | 511 MiB | **59.5%** |
| 8 | 2,177 MiB | 1,293 MiB | **40.6%** |
| 32 | 5,998 MiB | 3,959 MiB | **34.0%** |

학습 가중치는 **약 6,391만 개 → 39만 개로 99.4% 감소**했습니다. 하지만 기본 모델과 중간 계산값은 메모리에 남으므로, 전체 메모리가 같은 비율로 줄지는 않습니다.

같은 초기 가중치와 합성 토큰으로 짧게 학습한 **메모리 비교 실험**이며, 학습 품질은 평가하지 않았습니다.

[상세 보고서·그래프·재현 방법](/home/mind4u/dev/llm_playground/reports/minimind/20261118-234200-lora-memory.md)

## 사용자

lora에서 학습할 때 A와 B 행렬의 값이 어떻게 바뀌는지 테스트해봐

## 어시스턴트

LoRA 구현과 이전 LoRA 대화를 확인했습니다. 원본 `apply_lora`와 `train_lora.py` 기본 설정(rank 16, AdamW lr 1e-4, clip 1.0)을 그대로 쓰고, 감정 분류 과제(`dair-ai/emotion`, 학습 2,000개)로 300 step 동안 A·B·ΔW=BA를 측정하는 스크립트를 작성했습니다. 학습 후 병합 결과를 비교할 때 `state_dict()`가 텐서 참조를 돌려주는 문제를 발견해, 동결 가중치 비교를 병합 전으로 옮겼습니다.

실제 학습을 300 step 돌리며 A·B를 측정했습니다. **B가 먼저 움직이고, A는 값은 바뀌어도 방향은 거의 그대로이며, 학습된 변화 대부분은 B에 담깁니다.**

모델은 Dense 64M이고, 원본 `apply_lora`로 `q_proj`와 `o_proj` 16개 층에 rank 16 LoRA를 붙였습니다. 설정은 `train_lora.py` 기본값(AdamW lr 1e-4, clip 1.0)을 따랐고, 과제는 앞의 감정 분류입니다.

| step | ‖grad A‖ | ‖grad B‖ | A 변화율 | A 방향 유지(cos) | ‖B‖ | ‖BA‖/‖W‖ | BA 유효 rank |
|---|---|---|---|---|---|---|---|
| 0 | – | – | 0 | 1 | **0** | 0 | – |
| 1 | **0.000** | 0.41 | 0.0001% | 1.000 | 0.011 | 0.02% | 13.5 |
| 10 | 0.14 | 0.41 | 3.9% | 0.999 | 0.094 | 0.18% | 11.7 |
| 50 | 0.03 | 0.35 | 17.4% | 0.989 | 0.299 | 0.67% | 9.2 |
| 300 | 0.05 | 0.50 | **21.8%** | **0.984** | **0.353** | **0.84%** | **8.6** |

- **첫 step에는 B만 움직입니다.** B=0이면 A의 기울기 `Bᵀ·(…)`도 0이 되기 때문입니다. step 1에서 grad A는 정확히 0이었습니다. 이때 A가 바뀐 1e-6은 weight decay(lr×0.01)뿐입니다. B의 원소는 모두 약 lr(1e-4)만큼 움직였는데, Adam의 첫 업데이트가 원소마다 크기 lr에 가깝기 때문입니다.
- **A의 값은 바뀌지만 방향은 거의 그대로입니다.** 기울기는 B가 A보다 대략 3~20배 큽니다. 그래도 Adam이 원소별로 정규화하므로 A도 초기값 대비 22% 바뀌었습니다. 그런데 A가 투영하는 16차원 공간과 초기 공간의 cos는 0.984입니다. 처음 무작위로 정한 방향을 거의 유지하므로, 학습된 변화는 대부분 B에 담깁니다.
- **변화는 초반에 몰립니다.** ‖BA‖ 증가의 약 79%가 처음 50 step에 일어났습니다. 후반에는 B의 방향이 거의 고정됩니다(연속 측정 간 cos 0.9998).
- **rank 16을 다 쓰지 않습니다.** BA의 유효 rank는 13.5에서 8.6으로 줄었습니다. 뒤쪽 층(5~7층)은 3~6으로 한두 방향에 집중됩니다.
- **원본 가중치는 거의 그대로입니다.** 원본 가중치 W는 바뀌지 않았고, 학습된 변화량 BA는 W의 1% 미만입니다. `W + BA`로 병합한 모델과 LoRA 모델의 logit 차이는 최대 1.8e-5였습니다.

**주의할 점:** 300 step으로는 학습이 덜 끝났습니다. 정확도는 29.1%에서 46.7%로 올랐지만, 같은 데이터로 전체 SFT를 하면 88.1%입니다. 수렴할 때까지 학습하면 A·B의 모습이 달라질 수 있습니다. seed는 하나만 썼고, 학습한 LoRA 가중치는 저장하지 않았습니다.

- 실행기: [lora_ab_dynamics.py](/home/mind4u/dev/llm_playground/projects/minimind/scripts/knowledge_sft/lora_ab_dynamics.py)
- [상세 보고서](/home/mind4u/dev/llm_playground/reports/minimind/20261119-091519-lora-ab-dynamics.md)
- 산출물: `outputs/minimind/20261119-091519-lora-ab-dynamics/` (시점별 16개 층의 값이 `per_layer.json`에 있음)

## 사용자

lora에서 학습할 때 A행렬과 B행렬을 모두 0으로 초기화했을 때와 A행렬을 랜덤, B행렬을 0으로 초기화했을 때를 step 3번 학습해서 비교해

## 어시스턴트

두 초기화 방식을 같은 batch 3개로 각각 3 step 학습하고, step마다 기울기와 값의 변화를 기록하는 비교 스크립트를 작성해 실행했습니다.

A와 B를 모두 0으로 초기화하면 3 step 내내 기울기가 정확히 0이어서 아무것도 학습되지 않았습니다. 반면 A 랜덤·B 0(원본 방식)은 첫 step부터 B가 움직였고, 두 번째 step부터 A도 움직였습니다.

두 조건 모두 같은 모델, 같은 batch 3개, AdamW lr 1e-4로 학습했습니다.

| 초기화 | step | loss | ‖grad A‖ | ‖grad B‖ | ‖B‖ | ‖BA‖ | 출력 변화 |
|---|---|---|---|---|---|---|---|
| A 랜덤·B 0 | 1 | 2.4903 | **0** | 2.036 | 0.044 | 0.025 | 0.127 |
| | 2 | 2.7664 | 0.128 | 2.550 | 0.083 | 0.047 | 0.272 |
| | 3 | 1.9169 | 0.166 | 1.724 | 0.121 | 0.068 | 0.419 |
| A 0·B 0 | 1 | 2.4903 | **0** | **0** | 0 | 0 | **0** |
| | 2 | 2.7688 | **0** | **0** | 0 | 0 | **0** |
| | 3 | 1.9442 | **0** | **0** | 0 | 0 | **0** |

출력 변화는 첫 문장에서 학습 전과 비교한 logit 변화의 최댓값입니다.

**왜 이렇게 되는지**

LoRA 출력은 `h = Wx + B(Ax)`입니다. 손실의 출력 기울기를 `g`라고 하면 A와 B의 기울기는 다음과 같습니다.
- `∂L/∂B = g·(Ax)ᵀ`
- `∂L/∂A = Bᵀg·xᵀ`

- **A 0·B 0:** A=0이면 B의 기울기가 0이고, B=0이면 A의 기울기도 0입니다. 둘이 서로를 0에 묶어 두므로 몇 step을 돌려도 그대로입니다. weight decay도 0에 곱해져서 아무 효과가 없습니다. 실측에서도 A·B의 0이 아닌 원소가 끝까지 0개였고, 출력 변화도 0이었습니다. LoRA를 붙이지 않은 것과 같습니다.
- **A 랜덤·B 0:** 처음에는 BA=0이라 기본 모델의 출력이 그대로 유지됩니다. 그래서 step 1의 loss가 두 조건에서 2.4903으로 같습니다. 하지만 Ax가 0이 아니라서 step 1에 B의 기울기가 바로 생깁니다. B가 0이 아니게 된 step 2부터는 A에도 기울기가 생깁니다.
- **A 0·B 0 쪽 step 2·3의 loss**(2.7688, 1.9442)는 기본 모델이 그 batch에서 내는 값입니다. A 랜덤 쪽이 조금씩 낮은 것은 3 step 동안 생긴 작은 차이일 뿐이고, 성능을 비교한 결과는 아닙니다.

- 실행기: [lora_init_compare.py](/home/mind4u/dev/llm_playground/projects/minimind/scripts/knowledge_sft/lora_init_compare.py)
- [상세 보고서](/home/mind4u/dev/llm_playground/reports/minimind/20261119-091902-lora-init-compare.md)
- 산출물: `outputs/minimind/20261119-091902-lora-init-compare/`

## 사용자

lora 구조와 학습에 관련된 코드를 간단한 코드로 설명해

## 어시스턴트

MiniMind의 LoRA 코드는 크게 **구조**([model_lora.py](/home/mind4u/dev/llm_playground/projects/minimind/model/model_lora.py))와 **학습**([train_lora.py](/home/mind4u/dev/llm_playground/projects/minimind/trainer/train_lora.py)) 두 부분입니다. 핵심만 줄여서 순서대로 설명하겠습니다.

### 1. LoRA 모듈: 작은 행렬 두 개

```python
class LoRA(nn.Module):
    def __init__(self, in_features, out_features, rank):
        super().__init__()
        self.A = nn.Linear(in_features, rank, bias=False)   # 768 → 16 (줄이기)
        self.B = nn.Linear(rank, out_features, bias=False)  # 16 → 768 (되돌리기)
        self.A.weight.data.normal_(mean=0.0, std=0.02)      # A는 작은 난수
        self.B.weight.data.zero_()                          # B는 0

    def forward(self, x):
        return self.B(self.A(x))                            # 보정값 = B·A·x
```

- 768×768 행렬 하나(약 59만 개)를 학습하는 대신, 768×16과 16×768 두 개(약 2.5만 개)만 학습합니다.
- **B가 0이라서 처음에는 보정값이 0입니다.** 그래서 LoRA를 붙여도 처음에는 원래 모델과 출력이 같습니다.
- **A는 0이 아니어야 합니다.** 둘 다 0이면 기울기가 영원히 0이어서 학습이 시작되지 않습니다(앞의 3 step 실험에서 확인).

### 2. 모델에 붙이기: 기존 Linear의 forward를 감싸기

```python
def apply_lora(model, rank=16):
    for name, module in model.named_modules():
        # 입력·출력 크기가 같은 Linear에만 붙임 → q_proj, o_proj
        if isinstance(module, nn.Linear) and module.in_features == module.out_features:
            lora = LoRA(module.in_features, module.out_features, rank)
            module.lora = lora                       # 하위 모듈로 등록 → 이름에 "lora"가 들어감
            original_forward = module.forward

            def forward_with_lora(x, layer1=original_forward, layer2=lora):
                return layer1(x) + layer2(x)         # W·x + B·A·x

            module.forward = forward_with_lora
```

- 모델 코드는 고치지 않고, 해당 층의 `forward`만 "원래 출력 + LoRA 출력"으로 바꿔 끼웁니다.
- 기본 설정에서는 `q_proj`와 `o_proj`(8층 × 2 = 16개)에 붙습니다. GQA라서 출력이 더 작은 `k_proj`, `v_proj`에는 붙지 않습니다.
- 기본값 `layer1=..., layer2=...`는 반복문 안에서 층마다 자기 LoRA가 묶이도록 하려는 것입니다(파이썬 클로저 함정을 피함).

### 3. 학습: 원래 가중치는 고정하고 LoRA만 옵티마이저에 넣기

```python
model, tokenizer = init_model(lm_config, "full_sft")
apply_lora(model, rank=16)

lora_params = []
for name, param in model.named_parameters():
    if "lora" in name:
        param.requires_grad = True       # A, B만 학습
        lora_params.append(param)
    else:
        param.requires_grad = False      # 원래 모델은 고정

optimizer = optim.AdamW(lora_params, lr=1e-4)
```

학습 루프 자체는 일반 SFT와 똑같습니다.

```python
for input_ids, labels in loader:
    with autocast_ctx:
        loss = model(input_ids, labels=labels).loss   # 순전파에서 W·x + B·A·x 계산
    loss.backward()                                   # 기울기는 A, B에만 생김
    torch.nn.utils.clip_grad_norm_(lora_params, 1.0)
    optimizer.step()
    optimizer.zero_grad()
```

- 원래 모델도 순전파에는 참여하지만 기울기와 Adam 상태를 저장하지 않습니다. 그래서 메모리가 줄어듭니다(앞의 실험에서 batch 32 기준 34%).
- 학습 중에는 B가 먼저 움직이고, A는 처음 방향을 거의 유지합니다(앞의 A·B 변화 실험).

### 4. 저장과 병합

```python
# 저장: 이름에 lora가 들어간 A, B만 따로 저장 → 작은 파일
save_lora(model, "lora_medical_768.pth")

# 병합: 보정값을 원래 가중치에 더해서 일반 모델로 만듦
W_new = W + B.weight @ A.weight      # [768×16] @ [16×768] = [768×768]
```

- **따로 쓰기:** 기본 모델을 불러오고 `apply_lora`로 붙인 뒤 `load_lora`로 A·B를 채웁니다. 용도별 LoRA 파일을 바꿔 끼울 수 있습니다.
- **병합해서 쓰기:** `merge_lora`가 `W + BA`를 계산해 저장합니다. 추론할 때 LoRA 경로가 없어지므로 추가 연산도 없습니다. 앞의 실험에서 병합 전후 logit 차이는 최대 1.8e-5였습니다.

### 한 줄 요약

`h = W·x + B·(A·x)`에서 **W는 고정하고 A·B만 학습**하며, B=0에서 출발해 처음 출력을 그대로 둔 채 조금씩 보정값을 키웁니다.
