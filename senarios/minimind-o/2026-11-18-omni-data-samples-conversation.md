# MiniMind-O 데이터 샘플 대화 원본

## 사용자

omni model 데이터 샘플 보여줘

## 어시스턴트

MiniMind-O의 데이터 로더와 로컬 예제를 확인해서, 모달리티별 실제 데이터 형식을 보여드리겠습니다.

MiniMind-O는 **대화 텍스트에 음성·이미지 데이터를 결합**해 학습합니다. 로컬에서 실제 T2A 학습 데이터 2개를 읽었습니다.

출처: [train_t2a.parquet](/home/mind4u/dev/llm_playground/outputs/notebooks/20261117-204708-omni-training/workspace/dataset/train_t2a.parquet) — 총 1,024개 샘플.

**1. 실제 샘플: 텍스트 질문 → 텍스트·음성 답변**

```json
{
  "conversations": [
    {
      "role": "user",
      "content": "Add a possessive 's to the following noun: house"
    },
    {
      "role": "assistant",
      "content": "The possessive form of \"house\" is \"house's.\""
    }
  ],
  "answer_audios": [
    [1049, 811, 1626, 1562, 306, 714, 1044, 924]
  ]
}
```

`answer_audios`는 **앞의 8개 코드만 표시**했습니다. 또한 실제 Parquet의 `conversations`는 위 배열을 JSON 문자열로 저장합니다. 여기서는 읽기 쉽게 풀었습니다.

```text
입력 텍스트
  “house에 소유격 's를 붙여줘”
           ↓
       MiniMind-O
           ↓
  ┌────────┴─────────┐
텍스트 답변         음성 코드
“house's”           [1049, 811, ...]
                         ↓ Mimi 디코더
                     말소리 파형
```

`answer_audios`의 숫자는 단어 ID나 음성 파형 값이 아니라 **Mimi 음성 코덱의 코드 ID**입니다. 8개 코드북을 사용하므로, 저장된 배열을 8개씩 묶으면 한 음성 프레임에 해당합니다.

```text
첫 프레임: [1049, 811, 1626, 1562, 306, 714, 1044, 924]
다음 프레임: 이어지는 8개 코드
...
```

**2. 실제 샘플: 문법 질문**

```json
{
  "conversations": [
    {
      "role": "user",
      "content": "In the sentence, \"Please carefully read the instructions before starting the task,\" identify the main verb and its type."
    },
    {
      "role": "assistant",
      "content": "The main verb in the sentence is \"read,\" and it is an action verb."
    }
  ],
  "answer_audios": [
    [1049, 811, 1626, 1477, 306, 165, 1504, 424]
  ]
}
```

이 샘플도 정답 텍스트와 그 답변의 음성 코드를 함께 제공합니다. 음성 코드는 앞부분만 표시했습니다.

**3. 음성·이미지 샘플은 어떤 필드를 추가하나?**

[실제 데이터 로더](/home/mind4u/dev/llm_playground/projects/minimind-o/dataset/omni_dataset.py:203)가 읽는 형식입니다.

| 유형 | 입력 | 정답 | 주요 필드 |
|---|---|---|---|
| T2A | 텍스트 질문 | 텍스트 + 음성 | `conversations`, `answer_audios` |
| A2A | 음성 질문 | 텍스트 + 음성 | 위 필드 + `question_audios` |
| I2T | 이미지 + 텍스트 질문 | 텍스트 | `conversations`, `image_bytes` |

- `question_audios`: 사용자 발화별 **음성 파일 바이트** 목록
- `answer_audios`: 어시스턴트 응답별 **Mimi 코드 배열** 목록
- `image_bytes`: 이미지 바이트. 대화 내용의 `<image>`와 연결
- `spk_emb`: 선택적인 192차원 화자 벡터
- `ref_audios`: 선택적인 참조 음성의 Mimi 코드

예를 들어 이미지 데이터의 대화 부분은 다음 형태입니다. **이 예시는 형식 설명용입니다.**

```json
[
  {
    "role": "user",
    "content": "<image>\nWhat is in this picture?"
  },
  {
    "role": "assistant",
    "content": "A cat is sitting on a desk."
  }
]
```

이미지 자체는 `content` 문자열에 넣지 않고, 별도의 `image_bytes`에 저장합니다.
