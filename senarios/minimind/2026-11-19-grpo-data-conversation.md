# MiniMind GRPO 데이터 확인 대화

저장일: 2026-11-19 (Asia/Seoul)

GRPO 데이터 샘플, 채팅 템플릿 변환, 왼쪽 패딩 결과에 관한 사용자 요청과
어시스턴트 최종 답변을 보관합니다. 도구 호출·진행 설명과 IDE/환경 메타데이터는
제외했습니다. 샘플 질문과 답변은 형식 설명을 위한 예시이며 실제 학습 데이터 발췌가 아닙니다.
템플릿 및 토큰 길이·패딩 수치는 로컬 MiniMind 토크나이저로 확인했습니다.
소스 링크는 저장소 기준 상대 경로로 표기했습니다.

## 사용자

grpo 데이터 샘플을 보여줘

## 어시스턴트

MiniMind의 GRPO 데이터는 **질문을 주고, 모델이 여러 답변을 생성한 뒤 보상으로 비교하는 방식**입니다. 현재 코드에는 두 가지 입력 형식이 있습니다.

아래는 **형식 설명을 위해 만든 예시**입니다.

**1. 일반 질문: 보상 모델로 답변 평가**

JSONL 한 줄에 대화 하나를 넣습니다.

```jsonl
{"conversations":[{"role":"user","content":"인공지능이 무엇인지 쉽게 설명해줘."},{"role":"assistant","content":"인공지능은 컴퓨터가 학습하고 판단하도록 만드는 기술입니다."}]}
{"conversations":[{"role":"user","content":"친구에게 생일 축하 메시지를 써줘."},{"role":"assistant","content":"생일 축하해! 오늘 행복한 하루 보내!"}]}
```

현재 [RLAIFDataset](../../projects/minimind/dataset/lm_dataset.py#L199)은 **마지막 assistant 답변을 제외한 대화**를 프롬프트로 사용합니다. 위에 적힌 답변을 따라 쓰도록 학습하는 것이 아닙니다.

```text
질문: 인공지능이 무엇인지 쉽게 설명해줘.
                    ↓
모델이 여러 답변 생성
                    ↓
보상 모델과 길이·반복 등의 규칙으로 평가
                    ↓
그룹 안에서 보상이 높은 답변을 더 선호하도록 학습
```

**2. 수학 문제: 정답과 비교해 평가**

현재 [GSM8KDataset](../../projects/minimind/dataset/gsm8k_dataset.py#L12)은 다음 형태의 로컬 JSONL도 지원합니다.

```jsonl
{"question":"Mina has 3 apples and buys 2 more. How many apples does she have?","answer":"3 + 2 = 5.\n#### 5"}
{"question":"There are 4 bags with 6 marbles each. How many marbles are there in total?","answer":"4 * 6 = 24.\n#### 24"}
```

여기서 `answer`는 **보상 계산에 쓰는 정답**이고, 모델 입력에는 들어가지 않습니다.

예를 들어 첫 문제에서:

| 모델이 생성한 답변 예시 | 정답 여부 |
|---|---|
| `3 + 2 = 5. #### 5` | 맞음 |
| `3 + 2 = 6. #### 6` | 틀림 |
| `The answer is 5. #### 5` | 맞음 |

**SFT는 준비된 정답 답변을 따라 쓰도록 학습하고, GRPO는 모델이 직접 생성한 여러 답변을 보상으로 비교해 학습한다는 차이가 있습니다.**

## 사용자

템플릿으로 변환한 데이터 보여줘

## 어시스턴트

앞서 만든 샘플을 **현재 로컬 MiniMind 토크나이저의 실제 채팅 템플릿**으로 변환하면 다음과 같습니다.

**1. 일반 질문 GRPO — 사고 모드 ON**

원본 데이터:

```json
{
  "conversations": [
    {"role": "user", "content": "인공지능이 무엇인지 쉽게 설명해줘."},
    {"role": "assistant", "content": "인공지능은 컴퓨터가 학습하고 판단하도록 만드는 기술입니다."}
  ]
}
```

모델에 들어가는 문자열:

```text
<|im_start|>user
인공지능이 무엇인지 쉽게 설명해줘.<|im_end|>
<|im_start|>assistant
<think>
```

**기존 assistant 답변은 빠지고**, 모델이 `<think>` 뒤부터 직접 생성합니다. 위 예시는 system 메시지가 추가되지 않은 경우입니다.

**2. 수학 문제 GRPO**

원본 데이터:

```json
{
  "question": "Mina has 3 apples and buys 2 more. How many apples does she have?",
  "answer": "3 + 2 = 5.\n#### 5"
}
```

모델에 들어가는 문자열:

```text
<|im_start|>user
Mina has 3 apples and buys 2 more. How many apples does she have?
Solve step by step. End your answer with #### followed by the final number.<|im_end|>
<|im_start|>assistant
```

정답은 모델 입력과 분리해 보관합니다.

```text
보상 계산용 정답:
3 + 2 = 5.
#### 5
```

이 프롬프트에서 여러 답변을 생성한 뒤, 각 답변의 최종 숫자를 정답 `5`와 비교해 보상을 계산합니다.

## 사용자

패딩한 결과 보여줘

## 어시스턴트

MiniMind의 GRPO는 **왼쪽 패딩**을 사용합니다. 실제 토크나이저로 길이가 다른 두 질문을 변환한 결과입니다. 두 질문 모두 앞서 보여드린 수학 지시문과 채팅 템플릿을 포함합니다.

| 질문 | 원래 토큰 수 | 추가 패딩 | 최종 길이 |
|---|---:|---:|---:|
| `1 + 1 = ?` | 39 | 14 | 53 |
| `Mina has 3 apples ...` | 53 | 0 | 53 |

**패딩한 문자열**

```text
샘플 1:
[PAD] × 14
<|im_start|>user
1 + 1 = ?
Solve step by step. End your answer with #### followed by the final number.<|im_end|>
<|im_start|>assistant

샘플 2:
<|im_start|>user
Mina has 3 apples and buys 2 more. How many apples does she have?
Solve step by step. End your answer with #### followed by the final number.<|im_end|>
<|im_start|>assistant
```

`[PAD] × 14`는 표시를 줄인 표현입니다. 실제로는 **`<|endoftext|>` 토큰 14개**가 앞에 붙으며, 토큰 ID는 `0`입니다.

**Attention mask**

아래도 반복 구간을 줄여 표시했습니다.

```text
샘플 1: [0 × 14] + [1 × 39]
샘플 2:           [1 × 53]
```

- `0`: 패딩 위치이므로 Attention에서 참고하지 않습니다.
- `1`: 실제 프롬프트 위치입니다.

최종 텐서 크기는 둘 다 같습니다.

```text
input_ids.shape      = [2, 53]
attention_mask.shape = [2, 53]
```

이렇게 하면 **두 프롬프트의 마지막 위치가 나란히 맞춰져**, 그 뒤에서 답변 생성을 시작할 수 있습니다. 정답 `answer`는 이 입력에 넣지 않고 보상 계산용으로 따로 보관합니다.
