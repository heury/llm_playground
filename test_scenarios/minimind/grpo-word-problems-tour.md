# 노트북 기반 MiniMind 산수 문장제 GRPO 시나리오

작성일: 2026-11-19. 상태: **학습·평가 실행 완료**.

실제 결과: [실행 보고서](../../reports/minimind/20261119-004300-grpo-word-problems.md).

## 목적과 근거

[minimind_grpo_tour.py](../../notebooks/minimind/minimind_grpo_tour.py)의 데이터 생성,
GSM8K 형식 보상, rollout, 전후 평가 방식을 참고한다. 원본 SFT 모델이 가끔
맞힐 수 있는 한 단계 문장제에서 정답 샘플의 확률을 높일 수 있는지 확인한다.
참고 소스 SHA256은 문서 끝에 기록한다. 노트북의 과거 측정값은 재실행 결과로 취급하지 않는다.

앞선 숫자 전용 실험은 32토큰 안에서 정답 숫자만 출력해야 보상을 줬다.
훈련 1,024샘플의 보상이 모두 0이어서 가중치가 변하지 않았다.
이번에는 SFT에서 사용한 풀이 형식에 맞춰 영어 문장제와 `#### 숫자`를 쓰고,
풀이를 위한 최대 생성 길이를 96토큰으로 늘린다. 별도 SFT나 assistant 답변
접두어를 추가하지 않고 원본 모델에서 독립적으로 시작한다.

## 1. 모델과 데이터 고정

- 모델: `caspar/MiniMind-Dense-64M-Mixed-SFT-v1.0`.
- 리비전: `7597c22aa4d3e7b2ac824b4b4be01158cccf1b8d`.
- 원본 safetensors/config/tokenizer를 사용하고 로컬 native MiniMind 구현에 로딩한다.
  이전 상품 SFT, DPO, 숫자 전용 GRPO 산출물은 시작점으로 쓰지 않는다.
- 토크나이저는 고정한 모델 리비전에서 가져온다. 노트북은 로컬 model/의
  토크나이저를 사용하므로, 재현 실행에서는 두 토크나이저와 템플릿의 동일 여부를 기록한다.
- 공개 chat template, `open_thinking=False`, 빈 think 블록 제거를 전후 동일하게 적용한다.

노트북의 `make_problem` 알고리즘을 그대로 사용한다.

| 항목 | 구성 |
|---|---|
| 피연산자 | a,b 각각 2~9 |
| 종류 | 한 단계 덧셈 / 곱셈 문장제 |
| 이름 | Tom, Anna, Ben, Lily, Sam, Mia, Leo, Emma |
| 물건 | apples, pencils, cookies, books, marbles, stickers, cards, balloons |
| 생성 시드 | `random.Random(0)` |
| 생성 방법 | 덧셈·곱셈 번갈아 생성, 질문 문자열 중복 제외 |
| 학습 | 처음 128개: 덧셈 64 + 곱셈 64 |
| 최종 평가 | 다음 256개: 덧셈 128 + 곱셈 128 |

입력 예:

```text
Tom has 3 apples. Tom gets 4 more apples. How many apples does Tom have now?
Solve step by step. End your answer with #### followed by the final number.
```

```text
Anna buys 3 boxes. Each box has 4 pencils. How many pencils does Anna buy in total?
Solve step by step. End your answer with #### followed by the final number.
```

JSONL의 정답은 각각 `Tom has 3 + 4 = 7 apples now.\n#### 7`,
`Anna buys 3 * 4 = 12 pencils in total.\n#### 12`와 같이 저장한다.
정답 열은 보상 함수에서만 읽으며 정책 프롬프트에는 포함하지 않는다.
[GSM8KDataset](../../projects/minimind/dataset/gsm8k_dataset.py)을 사용해
질문 뒤 지시문과 chat template를 구성한다.

**평가 범위:** 노트북과 동일하게 질문 문자열만 분리한다. 이름/물건이 달라도
같은 a,b 계산식이 훈련과 평가에 걸쳐 나타날 수 있다. 따라서 이 평가는 새로운
문장 표면에서의 정답 출력 및 행동 개선이며, 미학습 계산식에 대한 일반화를
증명하지 않는다. 계산식 일반화를 별도로 보려면 교환된 a,b까지 묶어 데이터
분할부터 다시 설계하고 별개 실험으로 기록한다.

## 2. 보상은 기존 정답 채점 함수 재사용

[score_gsm8k_response](../../projects/minimind/trainer/gsm8k_reward.py)를 그대로 쓴다.

| 응답 | 보상 |
|---|---:|
| 풀이 후 마지막 줄 `#### 12`, 실제 정답 12 | 1 |
| 풀이에 12가 있어도 마지막 줄 `#### 15` | 0 |
| 숫자 12만 출력 | 0: 이 실험의 형식 미준수 |
| `####`가 없거나 둘 이상 | 0 |
| 형식은 맞지만 숫자 오답 | 0 |

풀이 텍스트를 허용하며, 최종 줄의 숫자만 채점한다. Decimal 비교를 사용한다.
형식 자체·길이·짧음에 대한 보너스나 별도 보상 모델은 없다.
최종 정답이 맞아도 풀이 자체가 맞았다고 판정하지 않는다.

## 3. 사전 점검에서 학습 신호 확인

노트북 전체를 자동 실행하면 다운로드/학습 셀이 실행될 수 있으므로, 문서 참조만으로
노트북을 실행하지 않는다. 실제 실행 시 필요한 셀/함수를 분리한 실행기를 사용한다.

1. 모델/토크나이저/데이터 해시와 분할 중복을 검증한다. 각 prompt 길이와
   prompt+96 길이를 측정해 실제 전체 문맥 340토큰 이내인지 확인한다.
   초과 시 질문을 자르지 않고 설정/문제를 검토한다.
2. 학습 128개 중 생성 순서 기준 첫 덧셈 8개·곱셈 8개를 고정해 16문항을 고른다.
   정답률로 쉬운 문제만 골라내지 않는다.
3. 각 문제에서 6개씩 생성해 96개를 기존 보상 함수로 채점한다.
4. 정답 샘플 수, 형식 오류, 생성 한도 도달과 **mixed-reward 그룹 비율**을 기록한다.

본 학습 진입 기준은 정답 샘플이 1개 이상이고, 16그룹 중 최소 4그룹에서
보상이 다른 답변이 존재하는 것이다. 25%는 이번 실험의 사전 운영 기준으로,
알고리즘의 필수 임계값이나 노트북의 보장값은 아니다.
미달 시 본 학습을 시작하지 않고 진단 결과를 보고한다. 모든 그룹이 같은 보상이면
다시 많은 optimizer 단계를 호출해도 정답 선호를 위한 신호가 없다.
SFT 준비나 문제 난도 변경이 필요하면 새 실험으로 분리하고 시작점부터 다시 평가한다.
최종 평가 256문제의 결과로 진입 기준이나 하이퍼파라미터를 조정하지 않는다.

## 4. 학습 설정

| 설정 | 값 |
|---|---|
| 전체 매개변수 학습 | 63,912,192개 |
| Reference | 시작 SFT 모델의 독립적인 고정 복사본 |
| Epochs / 질문 batch | 2 / 4 |
| 질문당 생성 G | 6 |
| 학습량 | 32 steps/epoch, 총 64 optimizer steps, 총 1,536 rollout |
| LR | 3e-7, cosine으로 3e-8까지 감소 |
| KL beta / clip epsilon | 0.1 / 0.2 |
| Optimizer | AdamW, betas=(0.9,0.999), weight decay=0.01 |
| Gradient clip / accumulation | 1.0 / 1 |
| Precision | FP32 가중치/상태, BF16 autocast |
| Prompt 최대 / 생성 최대 | 256 / 96토큰; 실제 전체 길이 별도 확인 |
| 훈련 seed | epoch별 42+epoch, epoch는 0부터 |
| Rollout | temperature 0.8, top-p 0.85, top-k 50 |
| Loss | **`grpo` 명시** |

노트북은 기본 위젯이 `cispo`다. 이번 주 실험은 사용자 요청에 맞춰 `grpo`를
명시한다. 노트북 기본값의 수치를 그대로 재현했다고 표현하지 않는다.
노트북의 원본 `grpo_train_epoch` 호출 방식과 `TorchRolloutEngine`,
`calculate_gsm8k_rewards_and_stats`를 우선 재사용한다. global 변수 설정과
checkpoint 경로만 실행용 래퍼에서 주입하고 원본 함수를 바꾼다면 변경점을 기록한다.

원본 sampler의 temperature/top-p/top-k를 재현하므로 무변형 on-policy sampling과
동일하다고 주장하지 않는다. rollout 및 재계산 log-prob의 의미·수치 차이를
실제 샘플에서 확인하고 사용한 정밀도와 배치 형태를 기록한다. 동일 초기 가중치에서
reference KL이 수치 오차 범위인지 확인한다. old/new ratio에 큰 차이가 있으면
학습을 진행하기 전에 원인을 확인한다.

그룹 advantage는 질문별 평균과 population std+1e-4로 계산하고 EOS 첫 토큰까지
포함한다. EOS 뒤 토큰, prompt, padding은 학습 대상에서 제외한다. native generate의
inference tensor는 gradient 계산 전에 clone한다. 한 rollout당 한 update이며,
이 경우 GRPO/CISPO clipping 차이를 충분히 비교한 실험이라고 해석하지 않는다.

## 5. 전후 평가와 판정

학습 전과 마지막 epoch의 FP32 checkpoint strict 재로딩 후, 같은 평가 256개에 대해:

- Greedy 1회: 최종 답 기준 전체/덧셈/곱셈 정답률.
- Sampling 4회: temperature 0.8, top-p 0.85, top-k 50, 고정 seed 0.
  총 1,024개 답변의 정답률이며 pass@4와 구분한다.
- 형식 오류율, 길이 한도 도달률, 평균 생성 길이.
- 정답 숫자가 본문에 등장했지만 최종 답은 틀린 비율.
  이는 숫자 포함 탐지 지표이며 올바른 중간 계산을 했다는 증거는 아니다.
- 동일 8개 일반 질문(상식·설명·코드)의 원문 응답 비교.

훈련 중에는 reward, mixed-group 비율, advantage std, KL, gradient norm,
응답 길이, 정답/오답/형식 오류를 기록한다. loss 값만으로 학습 성공을 판단하지 않는다.
weight decay가 있으면 보상 gradient 없이도 가중치가 바뀔 수 있어,
가중치 변화만으로 정답 학습을 입증하지 않는다.

사전 성공 기준은 평가 greedy 정답 수가 늘고, 형식 오류율이 증가하지 않으며,
일반 질문에서 명백한 응답 붕괴가 관찰되지 않는 것이다. 작은 차이는 단일 seed의
탐색 결과로 보고하고 유의한 향상이라고 단정하지 않는다. 실패도 같은 지표로 기록한다.
최종 평가로 checkpoint를 선택하거나 추가 epoch/LR를 탐색하지 않는다.

검증 항목은 frozen reference 불변, 유한 gradient/가중치, optimizer 단계 수,
보상 gradient가 존재한 단계 수, FP32 가중치 변화량, strict 재로딩 후 tensor/logit
일치다. 원본 trainer의 FP16 저장 외에 최종 FP32 가중치를 별도로 보존한다.

## 6. 산출물과 보고

이 문서는 모델 프로젝트의 실행 시나리오다. 산출물은
`outputs/minimind/<실행-ID>-grpo-word-problems/`, 보고서는
`reports/minimind/<실행-ID>-grpo-word-problems.md`에 저장한다.
노트북 자체를 실습 환경에서 실행하는 경우에는 해당 README의
`outputs/notebooks/`, `reports/notebooks/` 규칙을 따른다.

데이터 JSONL, source/와 해시, 고정 모델/토크나이저 리비전, 환경 버전,
96개 사전 점검 응답, 1,536개 훈련 rollout 원문·보상, 단계별 지표,
전후 greedy/sampling 응답, 일반 질문, FP32 가중치와 재로딩 검증을 보관한다.
trainer의 resume checkpoint도 root outputs 아래로 지정하고 기존 ../checkpoints를
덮어쓰지 않는다. 실행 전 원본 저장 경로를 확인한다.

참고 노트북의 설명에는 과거 greedy 약 13%→38%가 기재되어 있다.
이는 현재 실행에서 재검증한 수치가 아니며 이번 시나리오의 예상 결과나 보장이 아니다.
작성 당시에는 미실행 상태였으며, 이후 실행한 측정값과 검증 범위는 상단 실행 보고서에 기록했다.

참고 소스 SHA256: `89be1737db1440a70073b2f01e355905e93b971dc3366be0a0c25dbb706267dd`
