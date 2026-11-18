# MiniMind 지식 증류 loss 계산 대화

저장일: 2026-11-19 (Asia/Seoul)

지식 증류 loss에 관한 사용자 요청과 어시스턴트 최종 답변을 보관합니다.
도구 호출·진행 설명과 IDE/환경 메타데이터는 제외했습니다.
소스 링크는 저장소 기준 상대 경로로 표기했습니다.

## 사용자

minimind의 knowledge distillation loss에 대해 간단한 코드와 함께 설명해

## 어시스턴트

MiniMind의 지식 증류는 **정답 토큰을 맞히는 학습(CE)**과 **교사 모델의 다음 토큰 확률 분포를 따라가는 학습(KL)**을 함께 합니다.

[train_distillation.py](../../projects/minimind/trainer/train_distillation.py#L25)의 계산을 간단히 풀면 다음과 같습니다.

```python
import torch
import torch.nn.functional as F

# 같은 문장을 학생과 교사에게 입력
student_logits = student(input_ids).logits[:, :-1, :]

teacher.eval()
with torch.no_grad():
    teacher_logits = teacher(input_ids).logits[:, :-1, :]

# 현재 위치의 출력으로 다음 토큰 예측
targets = labels[:, 1:]
valid = targets != -100  # 학습 대상 위치만 선택

s = student_logits[valid]  # [유효 토큰 수, vocab]
t = teacher_logits[valid]
y = targets[valid]

# ① 정답 토큰을 맞히는 손실
ce_loss = F.cross_entropy(s, y)

# ② 교사의 확률 분포를 따라가는 손실
T = 1.5
teacher_probs = F.softmax(t / T, dim=-1)
student_log_probs = F.log_softmax(s / T, dim=-1)

kd_loss = F.kl_div(
    student_log_probs,
    teacher_probs,
    reduction="batchmean",
) * T**2

# ③ 두 손실 결합
alpha = 0.5
loss = alpha * ce_loss + (1 - alpha) * kd_loss
loss.backward()  # 학생 모델만 업데이트
```

위 코드는 교사와 학생이 같은 토큰 어휘를 쓴다고 가정합니다.

**CE와 증류 손실의 차이**는 한 위치의 예측을 보면 이해하기 쉽습니다.

```text
문맥: “대한민국의 수도는”
정답 토큰: “서울”

후보       서울    부산    도쿄
교사       0.90    0.08    0.02
학생       0.50    0.30    0.20
```

- **CE:** 정답인 `"서울"`의 확률을 높입니다.
- **KL:** `"서울"`뿐 아니라 다른 후보까지 포함해 학생의 분포가 교사의 분포에 가까워지도록 합니다.

여기서 KL의 방향은 **`KL(교사 || 학생)`**입니다. 코드에서는 학생의 **로그 확률**을 첫 인자로, 교사의 **확률**을 두 번째 인자로 전달합니다.

`T`는 분포를 부드럽게 만드는 온도입니다. `T > 1`이면 낮은 확률의 후보들 간 차이도 학습하기 쉬워지고, `T²`는 온도로 작아지는 기울기 크기를 보정합니다.

MiniMind 기본값은 `alpha=0.5`, `T=1.5`입니다. **각 유효 위치에서 어휘 전체의 KL을 계산한 뒤, 유효 토큰 전체에 대해 평균**냅니다. 학생이 MoE라면 CE 쪽에 보조 손실도 더합니다.

GRPO rollout과 달리, 이 과정에서는 **데이터에 있는 문장을 두 모델에 그대로 넣어 모든 위치를 병렬 평가**합니다. 교사가 답변을 새로 생성할 필요는 없습니다.
