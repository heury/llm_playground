# MiniMind SFT 데이터 확인 대화

저장일: 2026-11-19 (Asia/Seoul)

SFT 샘플, 채팅 템플릿, input/target, padding을 확인한 사용자 요청과
어시스턴트 최종 답변을 보관합니다. 도구 호출·진행 설명과 IDE/환경 메타데이터는
제외했습니다. 데이터는 형식 설명을 위한 예시이며 실제 학습 데이터 발췌나 실행 결과가 아닙니다.
소스 링크는 저장소 기준 상대 경로로 표기했습니다.

## 사용자

sft 데이터 샘플 보여줘

## 어시스턴트

MiniMind **SFT 데이터 예시**입니다. JSONL 각 줄에 질문과 답변을 넣습니다.

```jsonl
{"conversations":[{"role":"user","content":"강아지는 어떻게 울어?"},{"role":"assistant","content":"강아지는 멍멍 하고 짖어요."}]}
{"conversations":[{"role":"user","content":"What sound does a dog make?"},{"role":"assistant","content":"A dog says woof!"}]}
{"conversations":[{"role":"user","content":"1 더하기 2는?"},{"role":"assistant","content":"3입니다."}]}
```

**`assistant` 답변 부분을 예측하도록 학습**합니다. 실제 데이터 발췌가 아닌 [SFT 로더](../../projects/minimind/dataset/lm_dataset.py) 형식에 맞춘 예시입니다.

## 사용자

템플릿으로 변환한 데이터 보여줘

## 어시스턴트

앞의 SFT 샘플을 [채팅 템플릿](../../projects/minimind/model/chat_template.jinja)으로 변환하면 다음 문자열이 됩니다.

```text
<|im_start|>user
강아지는 어떻게 울어?<|im_end|>
<|im_start|>assistant
<think>

</think>

강아지는 멍멍 하고 짖어요.<|im_end|>
```

실제 SFT 로더는 후처리에서 빈 `<think>` 블록을 80% 확률로 제거하므로, 다음 형태도 사용합니다.

```text
<|im_start|>user
강아지는 어떻게 울어?<|im_end|>
<|im_start|>assistant
강아지는 멍멍 하고 짖어요.<|im_end|>
```

학습 대상은 **assistant 답변과 끝 토큰 `<|im_end|>`**입니다.

## 사용자

input과 target 데이터로 보여줘

## 어시스턴트

빈 `<think>`가 제거된 SFT 샘플 기준입니다.

**Input — 모델에 넣는 전체 대화**
```text
<|im_start|>user
강아지는 어떻게 울어?<|im_end|>
<|im_start|>assistant
강아지는 멍멍 하고 짖어요.<|im_end|>
```

**Target(labels) — 답변 부분만 학습**
```text
[질문과 assistant 헤더에 해당하는 토큰은 모두 -100]
강아지는 멍멍 하고 짖어요.<|im_end|>
```

`-100`은 손실 계산에서 제외합니다. 학습할 때는 한 칸 밀어서 **앞의 토큰으로 다음 토큰을 예측**합니다.

| Input으로 지금까지 본 내용 | 예측할 Target |
|---|---|
| 질문 + assistant 헤더 | 답변의 첫 토큰 |
| 질문 + assistant 헤더 + 답변 첫 토큰 | 답변의 두 번째 토큰 |
| 질문 + assistant 헤더 + 답변 전체 | `<\|im_end\|>` |

## 사용자

padding한 결과 보여줘

## 어시스턴트

최대 길이까지 **오른쪽에 padding**을 붙입니다. 아래는 실제 토큰 ID 대신 텍스트로 표현한 결과입니다.

**Input**
```text
<|im_start|>user
강아지는 어떻게 울어?<|im_end|>
<|im_start|>assistant
강아지는 멍멍 하고 짖어요.<|im_end|>
[PAD] [PAD] [PAD] ...
```

**Target(labels) — Input과 같은 길이**
```text
[-100 ... 질문과 assistant 헤더 ... -100]
강아지는 멍멍 하고 짖어요.<|im_end|>
-100  -100  -100  ...
```

질문과 padding은 `-100`으로 가려 **답변 부분만 손실을 계산**합니다.
