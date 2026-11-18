# MiniMind 스팸 분류 모델 구현·학습 대화

저장일: 2026-11-19 (Asia/Seoul)

스팸 분류 모델의 수정·학습 요청, 학습 결과, 모델 수정 내역에 관한 사용자 요청과
어시스턴트 최종 답변을 보관합니다. 도구 호출·진행 설명과 IDE/환경 메타데이터는
제외했습니다. 소스·산출물 링크는 저장소 기준 상대 경로로 표기했습니다.
실제 실행 결과와 검증 근거는 연결된 보고서 및 outputs에 별도로 보존되어 있습니다.

## 사용자

minimind 모델을 수정해서 스팸 여부를 판단하는 모델을 만들어서 학습을 진행할려고 해. 모델은 아래 기준으로 수정해

- 출력은 스팸과 스팸아님 두개만 출력하도록 해.
- 마지막 transformer, final normalize, 출력 헤드만 학습하도록 하고 나머지는 학습이 안되도록 고정해.

학습 후 결과를 보고해

## 어시스턴트

모델 구현과 학습을 완료했습니다.

- 출력: **`스팸아님` / `스팸`** 두 클래스
- 학습: **마지막 Transformer 블록·최종 RMSNorm·출력 헤드만** 갱신
- 학습 매개변수: 전체 6,391만 개 중 약 **738만 개**
- 나머지 가중치는 학습 전후 해시가 동일함을 확인했습니다.

영어 [UCI SMS 데이터](https://archive.ics.uci.edu/dataset/228/sms+spam+collection)의 독립 테스트 **516건** 결과입니다.

| 지표 | 결과 |
|---|---:|
| 정확도 | **99.03%** |
| 스팸 정밀도 | 98.36% |
| 스팸 재현율 | 93.75% |
| 스팸 F1 | **96.00%** |
| 정상 문자 오탐률 | 0.22% |

정상 문자 **1건을 오탐**했고, 스팸 **4건을 놓쳤습니다**. 8에포크 학습 후 검증 성능으로 4에포크 모델을 선택했습니다. 관련 검사 8개와 저장·재로드 검증도 통과했습니다.

[학습 모델](../../outputs/minimind/20261119-001300-spam-last-block/best/) · [상세 보고서·실행 방법](../../reports/minimind/20261119-001300-spam-last-block.md)

한국어 스팸 판정 성능은 평가하지 않았습니다.

## 사용자

모델 수정한 내역을 보여줘

## 어시스턴트

기존 MiniMind 백본을 재사용하는 **`MiniMindForSpamClassification` 클래스를 추가**했습니다. 주요 변경은 다음과 같습니다.

**1. 출력 헤드를 2클래스 분류용으로 교체**

```python
# 기존 언어 모델
self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

# 새 스팸 분류 모델
self.lm_head = nn.Linear(hidden_size, 2, bias=True)
```

출력은 `[배치, 2]`이며, 클래스는 다음과 같습니다.

```python
LABELS = {0: '스팸아님', 1: '스팸'}
```

입력 임베딩과 출력 헤드의 크기가 달라지므로 가중치 공유도 해제했습니다.

```python
self.config.tie_word_embeddings = False
```

코드: [분류 모델 초기화](../../projects/minimind/model/model_spam.py#L14)

**2. 마지막 Transformer·최종 정규화·헤드만 학습**

```python
prefixes = (
    f'model.layers.{len(self.model.layers)-1}.',
    'model.norm.',
    'lm_head.',
)

for name, p in self.named_parameters():
    p.requires_grad_(name.startswith(prefixes))
```

8레이어 모델에서는 다음 범위가 적용됩니다.

| 부분 | 학습 여부 |
|---|---|
| 입력 임베딩 | 고정 |
| Transformer 0~6번 | 고정 |
| Transformer 7번 — 마지막 블록 | **학습** |
| 최종 RMSNorm | **학습** |
| 2클래스 출력 헤드 | **학습** |

고정 블록은 `eval()` 상태로 유지해 dropout도 비활성화합니다.

코드: [학습 범위 설정](../../projects/minimind/model/model_spam.py#L31)

**3. 문장 전체를 읽은 EOS 위치에서 분류**

```text
[BOS] 문자 내용 [EOS] [PAD] …
                  ↑
           이 위치의 벡터로 분류
```

```python
last = attention_mask.sum(-1).long() - 1
rows = torch.arange(input_ids.shape[0], device=input_ids.device)

hidden, _, _ = self.model(
    input_ids,
    attention_mask=attention_mask,
    use_cache=False,
)
logits = self.lm_head(hidden[rows, last])
```

패딩 위치는 제외하고, 마지막 유효 토큰이 EOS인지 검사합니다.

코드: [순전파와 분류](../../projects/minimind/model/model_spam.py#L44)

**4. 손실을 문장 단위 분류 손실로 변경**

```python
loss = nn.functional.cross_entropy(logits, labels)
```

정답은 토큰별 문장이 아니라 `[0, 1, …]` 형태입니다. 추론에서는 두 점수 중 큰 쪽을 선택합니다.

추가한 파일은 다음과 같습니다.

| 파일 | 역할 |
|---|---|
| [model_spam.py](../../projects/minimind/model/model_spam.py) | 분류 모델·백본 로드·저장 |
| [spam_dataset.py](../../projects/minimind/dataset/spam_dataset.py) | BOS/EOS·패딩·라벨 처리 |
| [train_spam.py](../../projects/minimind/trainer/train_spam.py) | 데이터 분리·학습·평가 |
| [eval_spam.py](../../projects/minimind/eval_spam.py) | 문자 입력 후 판정 출력 |
| [test_spam_classifier.py](../../projects/minimind/trainer/test_spam_classifier.py) | 학습 범위·출력·재로드 검증 |
