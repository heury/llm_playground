# MiniMind DPO loss 계산 대화

저장일: 2026-11-19 (Asia/Seoul)

DPO loss 계산 과정에 관한 사용자 요청과 어시스턴트 최종 답변을 보관합니다.
도구 호출·진행 설명과 IDE/환경 메타데이터는 제외했습니다.
소스 링크는 저장소 기준 상대 경로로 표기했습니다.

## 사용자

minimin에서 dpo loss 계산하는 과정을 간단한 코드로 설명해

## 어시스턴트

MiniMind의 DPO는 **선호 답변(chosen)을 비선호 답변(rejected)보다 더 선호하도록** 학습합니다. 이때 학습 전 상태로 고정한 참조 모델과 비교합니다.

실제 구현은 [train_dpo.py](../../../projects/minimind/trainer/train_dpo.py#L25)에 있으며, 핵심을 줄이면 다음과 같습니다.

```python
import torch.nn.functional as F

def answer_log_prob(model, x, y, mask):
    # x: 현재까지의 토큰, y: 한 칸 뒤의 정답 토큰
    logits = model(x).logits                 # [B, T, vocab]
    log_probs = F.log_softmax(logits, dim=-1)

    # 각 위치에서 실제 정답 토큰의 log 확률 선택
    token_log_probs = log_probs.gather(
        -1, y.unsqueeze(-1)
    ).squeeze(-1)                            # [B, T]

    # assistant 답변 부분만 합산 (길이 평균이 아님)
    return (token_log_probs * mask).sum(-1)  # [B]

# chosen과 rejected는 같은 질문에 대한 두 답변
policy_chosen = answer_log_prob(policy, xc, yc, mc)
policy_rejected = answer_log_prob(policy, xr, yr, mr)

with torch.no_grad():  # 참조 모델은 고정
    ref_chosen = answer_log_prob(reference, xc, yc, mc)
    ref_rejected = answer_log_prob(reference, xr, yr, mr)

# 각 모델이 chosen을 rejected보다 얼마나 선호하는지
policy_gap = policy_chosen - policy_rejected
ref_gap = ref_chosen - ref_rejected

# 참조 모델보다 선호 격차가 커지도록 학습
beta = 0.15
loss = -F.logsigmoid(beta * (policy_gap - ref_gap)).mean()
loss.backward()
```

계산 흐름은 **토큰별 log 확률 → 답변 전체 합산 → chosen/rejected 차이 → 참조 모델과의 차이 → loss**입니다.

예를 들어 다음과 같다면:

```text
학습 모델: chosen = -8, rejected = -12 → 격차 4
참조 모델: chosen = -9, rejected = -11 → 격차 2

개선된 격차 = 4 - 2 = 2
loss = -log(sigmoid(0.15 × 2)) ≈ 0.554
```

두 모델의 격차가 같으면 loss는 약 `0.693`입니다. **학습 모델의 chosen 선호 격차가 참조 모델보다 커질수록 loss가 작아집니다.**

실제 MiniMind는 chosen/rejected를 한 배치로 묶어 계산하며, 최종 학습 loss에는 `outputs.aux_loss`도 더합니다.
