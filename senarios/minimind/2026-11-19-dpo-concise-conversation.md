# MiniMind 간결한 답변 DPO 실험 대화

저장일: 2026-11-19 (Asia/Seoul)

DPO 시나리오 제안부터 학습 결과 보고까지 사용자 요청과 어시스턴트 답변을 보관합니다. 도구 호출·실행 로그와 IDE/환경 메타데이터는 제외했습니다. 진행 설명은 별도로 표시했습니다.

이전 Full SFT 대화: [2026-11-19-full-sft-knowledge-conversation.md](2026-11-19-full-sft-knowledge-conversation.md).

## 사용자

간단한 dpo 학습을 진행하는 시나리오 알려줘

## 어시스턴트 — 진행 설명

MiniMind의 DPO 학습 코드와 데이터 형식을 확인해서, 작은 데이터로 학습 전후를 비교할 수 있는 시나리오를 제안하겠습니다.

## 어시스턴트 — 시나리오 제안

가장 간단한 시나리오는 **“같은 정답이라도 간결하고 직접적인 답변을 선호하도록 DPO 학습하기”**입니다. 앞선 Full SFT 실험에 이어, 이번에는 **답변 방식이 바뀌는지** 확인할 수 있습니다.

시작 모델은 원본 `caspar/MiniMind-Dense-64M-Mixed-SFT-v1.0`을 사용하면 좋습니다. 이전 상품 지식 SFT 모델은 일반 응답 퇴행이 커서 실험 결과를 해석하기 어렵습니다.

**1. 선호 답변 쌍을 만듭니다.**

두 답변의 정답은 같게 유지하고, 불필요한 서두와 반복 여부에만 차이를 둡니다.

| 질문 | chosen: 선호 답변 | rejected: 비선호 답변 |
|---|---|---|
| What is the capital of France? | The capital of France is Paris. | That is an interesting question! Let me answer it carefully. The capital of France is Paris. So, Paris is the answer. |
| What is 2 + 3? | 2 + 3 = 5. | Thank you for asking! Adding two and three gives five. Therefore, the answer is five. |
| What is the opposite of hot? | The opposite of hot is cold. | This is a question about opposites. Hot has an opposite, and that opposite is cold. So the answer is cold. |

숫자·상식·단어 뜻 등 **학습 100쌍, 검증 20쌍, 최종 평가 30쌍** 정도로 시작합니다. 같은 질문의 표현 변형은 같은 분할에 넣어 평가 누출을 막습니다.

현재 [DPODataset](../../projects/minimind/dataset/lm_dataset.py)의 JSONL 형식은 다음과 같습니다.

```json
{"chosen":[{"role":"user","content":"What is the capital of France?"},{"role":"assistant","content":"The capital of France is Paris."}],"rejected":[{"role":"user","content":"What is the capital of France?"},{"role":"assistant","content":"That is an interesting question! Let me answer it carefully. The capital of France is Paris. So, Paris is the answer."}]}
```

두 대화의 **질문은 반드시 동일**해야 합니다.

**2. 작은 설정으로 학습합니다.**

| 항목 | 실험용 초기 설정 |
|---|---|
| 학습 방식 | 전체 매개변수 DPO |
| Reference 모델 | 시작 모델의 고정 복사본 |
| Epochs | 2 |
| Batch size | 4쌍 |
| Learning rate | `5e-7` |
| Beta | `0.1` |
| 문맥 길이 | 두 답변 모두 채팅 형식 포함 340토큰 미만 |

이는 소규모 실험의 출발값입니다. 저장소의 기존 짧은 문맥 DPO 실험도 LR `5e-7`, beta `0.1`을 사용합니다. 답변이 잘리는 샘플은 제외하고, 작은 업데이트가 보존되도록 FP32로 저장하는 편이 좋습니다.

**3. 학습 전후를 두 가지로 평가합니다.**

- **답변 쌍 점수:** 미학습 질문에서 chosen과 rejected의 로그확률 차이가 학습 전보다 chosen 쪽으로 이동했는지 확인합니다. 길이 차이의 영향을 볼 수 있도록 토큰 평균 로그확률도 함께 기록합니다.
- **실제 생성:** 동일한 질문에 greedy로 답하게 해서 정답률, 답변 길이, 불필요한 서두·반복을 비교합니다.

간결함만 측정하면 `Paris` 대신 엉뚱한 한 단어를 출력해도 좋아 보일 수 있으므로, **정답 유지 여부를 반드시 함께 확인**해야 합니다. 설명·코드 작성처럼 긴 답변이 필요한 질문도 별도로 넣어 과도하게 짧아지는지 관찰합니다.

예시 성공 기준은 **“평가 질문의 정답률이 유지되고, 불필요한 서두·반복이 줄어드는 것”**입니다. DPO loss 감소만으로 성공이라고 판단하지 않습니다.

이 실험으로 확인할 수 있는 것은 **간결한 답변에 대한 선호 학습**입니다. 새로운 지식 학습이나 전반적인 추론 능력 향상까지 의미하지는 않습니다. 아직 학습은 실행하지 않았습니다.

## 사용자

이 시나리오로 dpo 학습을 진행하고 결과 보고해

## 어시스턴트 — 진행 설명

원본 SFT 모델에서 시작해 간결한 정답을 선호하는 DPO를 실행하겠습니다. 학습 전후의 정답률과 답변 길이를 비교하고, 설명이나 코드가 필요한 질문에서도 응답이 유지되는지 확인하겠습니다.

정답은 같고 군더더기만 다른 답변 쌍 150개를 준비했습니다. 학습 100개·검증 20개·최종 평가 30개로 나누고, 제안한 LR `5e-7`, beta `0.1`, 2 epochs로 실행합니다. 짧아진 오답을 개선으로 오인하지 않도록 생성 답변의 정확도도 함께 확인하겠습니다.

2 epochs 학습과 체크포인트 재로딩 검증을 마쳤습니다. 선호 답변 쌍의 DPO 점수는 개선됐지만, 이것만으로 실제 답변이 좋아졌다고 판단할 수는 없습니다. 현재 미학습 질문의 생성 결과를 전후 비교하며 오답과 반복을 확인하고 있습니다.

DPO loss는 `0.693 → 0.036`으로 줄었지만, 미학습 질문의 답변 품질은 개선되지 않았습니다. 평균 길이는 82.1토큰에서 74.9토큰으로 줄었으나 반복은 소폭 늘었고, 맞던 태국 수도도 학습 후 틀렸습니다. 30개 답변을 모두 확인했으며, 자동 채점의 과대평가를 보정한 수동 판정도 보고서에 기록하겠습니다.

## 어시스턴트 — 최종 결과

DPO 학습을 완료했습니다. **선호 답변의 확률은 개선됐지만, 실제 응답 품질 개선에는 실패했습니다.**

원본 `caspar/MiniMind-Dense-64M-Mixed-SFT-v1.0`에서 시작해 학습 100쌍, 2 epochs, LR `5e-7`, beta `0.1`로 진행했습니다.

| 미학습 평가 30문항 | 학습 전 | 학습 후 |
|---|---:|---:|
| DPO loss | 0.693 | **0.036** |
| 핵심 정답률¹ | 10.0% | **6.7%** |
| 평균 답변 길이 | 82.1토큰 | 74.9토큰 |
| 중복 4단어 구간 수 | 224 | 231 |

¹ 전후 답변 30개를 모두 직접 검토했습니다.

예를 들어 태국 수도는 `Bangkok`에서 오답인 `Vietnam`으로 바뀌었습니다. `20 - 8`에는 짧게 `The answer is 20.`이라고 답했습니다. **길이는 줄었지만 정답 유지와 반복 감소를 달성하지 못했습니다.**

고정 reference의 불변성과 저장한 체크포인트의 정확한 재로딩을 확인했습니다. 원본 모델은 보존했습니다.

[상세 보고서](../../reports/minimind/20261119-001400-dpo-concise.md) · [학습된 체크포인트](../../outputs/minimind/20261119-001400-dpo-concise/dpo_concise_768.pth)

## 사용자

이 대화들을 /home/mind4u/dev/llm_playground/senarios/minimind에 저장해
