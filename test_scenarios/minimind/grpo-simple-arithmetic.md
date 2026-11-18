# 간단한 산수 GRPO

## 목적과 데이터

원본 `caspar/MiniMind-Dense-64M-Mixed-SFT-v1.0`의 고정 리비전
`7597c22aa4d3e7b2ac824b4b4be01158cccf1b8d`에서 숫자 정답에 대한
규칙 보상으로 GRPO가 학습되는지 확인한다. 이전 상품 SFT/DPO 모델은 쓰지 않는다.

피연산자 0~9의 덧셈(교환된 중복 제거)과 음수가 나오지 않는 뺄셈을 열거한다.
고정 프롬프트 예제 `1+1=2`, `4-1=3`은 모든 데이터 분할에서 제외한다.
남은 108개 계산식을 seed 42로 학습 64, 검증 12, 평가 32로 나눈다.
모든 분할에서 덧셈/뺄셈 수는 같다. 같은 계산식은 분할 간 공유하지 않는다.
고정 예제 두 개는 모든 질문에 들어가며, 해당 문제의 정답은 프롬프트에 넣지 않는다.

## 보상과 실행

응답 전체에서 공백을 제거한 결과가 정답인 ASCII 정수이면 1점, 그 외 0점.
설명 중 정답 숫자가 등장하는 것에는 보상을 주지 않는다. 형식·길이 보너스 없음.
2 epochs, 질문 batch 2, 질문당 8개 샘플, 최대 생성 32토큰, LR 1e-6,
KL beta .04, GRPO clip .2, AdamW weight decay 0, gradient clip 1, seed 42.
temperature 1, top-p 1, top-k 0, 답변을 제한하는 강제 디코딩 없음.
별도 SFT 준비 없이 시작하며 reference는 원본의 별도 고정 복사본이다.

작업 디렉토리 `projects/minimind/`:

```bash
uv run python scripts/knowledge_sft/grpo_arithmetic.py \
  --base /tmp/minimind-knowledge-base \
  --output ../../outputs/minimind/<실행-ID>
```

출력 폴더는 실행 전에 없어야 한다. 전체 FP32 매개변수/옵티마이저와 BF16
autocast를 사용한다. 프롬프트 길이가 같은 문제끼리 묶고 EOS 이후를 마스킹한다.
한 rollout당 한 on-policy update를 수행하며 old logps는 같은 업데이트 전
forward에서 detach한다. CISPO가 아닌 clipped GRPO 목적을 사용한다.

## 평가와 실패 판정

- 시작 전 학습 문제 12개×8샘플로 정답 보상과 mixed-reward 그룹이 있는지 점검한다.
- train/validation/test 전체를 전후 동일 greedy/32-token 조건으로 평가한다.
- 최종 epoch를 사용하며 validation으로 checkpoint를 선택하지 않는다.
- 마지막 숫자 일치는 참고만 한다. 정답 문장이 아닌 우연한 숫자 일치도 발생한다.
- 모든 훈련 샘플의 원문·보상, 그룹 보상 차이, KL, gradient norm과 가중치 변화량을 기록한다.
- 보상이 그룹 내 모두 같으면 task gradient가 0이다. policy/reference도 같으면
  KL gradient도 0일 수 있다. 이때 optimizer.step을 호출한 횟수를 실질적 학습으로 보고하지 않는다.
- 여섯 일반 질문의 전후 응답, 고정 reference 불변, FP32 저장/strict reload/logit 일치를 확인한다.
- 생성 길이·출력 형식 때문에 생긴 실패와 산수 계산 능력 전체를 구분한다.
  개선이 없으면 실패 결과를 그대로 보고하며 SFT 효과를 GRPO 효과로 섞지 않는다.

세부 동작: [GRPO.md](../../projects/minimind/scripts/knowledge_sft/GRPO.md).
실행기: [grpo_arithmetic.py](../../projects/minimind/scripts/knowledge_sft/grpo_arithmetic.py).
산출물은 root `outputs/minimind/<실행-ID>/`, 보고서는 `reports/minimind/<실행-ID>.md`.
