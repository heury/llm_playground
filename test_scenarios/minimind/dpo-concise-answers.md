# 간결한 정답을 선호하는 MiniMind DPO 実験

## 목적과 시작점

원본 `caspar/MiniMind-Dense-64M-Mixed-SFT-v1.0`, 리비전
`7597c22aa4d3e7b2ac824b4b4be01158cccf1b8d`에서 시작한다.
상품 지식 SFT 체크포인트는 사용하지 않는다. 같은 정답을 가진 두 응답 중
짧고 직접적인 답변을 chosen, 서두와 정답 반복이 있는 답변을 rejected로 구성한다.
새 지식이나 전반적 추론 향상을 검증하는 실험이 아니다.

## 데이터와 학습

seed 42로 150개 사실/질문(산술 60, 수도 30, 반의어 30, 용어 정의 30)을
학습 100, 검증 20, 최종 평가 30으로 나눈다. 동일 사실/질문은 분할을 넘지 않는다.
rejected는 chosen의 정답을 그대로 두 번 넣고 5가지 서두 템플릿으로 감싼다.
모든 split에서 서두 스타일은 공유되며 새 서두로의 일반화는 평가하지 않는다.
chosen/rejected의 사용자 메시지는 동일하게 유지한다.

2 epochs, batch 4쌍, LR 5e-7 일정, beta 0.1, AdamW weight decay 0,
clip 1, FP32 매개변수/옵티마이저와 BF16 autocast. 총 50 updates.
reference는 시작 모델의 별도 고정 복사본이다. native DPO loss와 assistant/EOS
마스킹을 재사용한다. 공개 모델의 빈 think 블록 제거를 항상 적용한다.
두 응답 모두 채팅 마커 포함 340토큰 미만이어야 하며 절단하지 않는다.

작업 디렉토리는 `projects/minimind/`:

```bash
uv run python scripts/knowledge_sft/dpo_concise.py \
  --base /tmp/minimind-knowledge-base \
  --output ../../outputs/minimind/<실행-ID>
```

출력 폴더는 실행 전에 없어야 한다. 다운로드 및 세부 동작은
[DPO.md](../../projects/minimind/scripts/knowledge_sft/DPO.md)를 참조한다.
실행기: [dpo_concise.py](../../projects/minimind/scripts/knowledge_sft/dpo_concise.py).

## 평가와 판정

- 검증은 진단용이며 항상 마지막 epoch를 사용한다. 평가 집합으로 설정을 조정하지 않는다.
- DPO loss, reference 대비 margin 변화, 원시 시퀀스/토큰 평균 로그확률 선호를 기록한다.
  원시 시퀀스 확률은 짧은 답변에 유리하며 시작점의 reference 대비 변화는 모두 tie다.
- 학습 100개·평가 30개에서 전후 greedy 생성, 최대 96 새 토큰을 비교한다.
- 설명·코드·절차를 요구하는 추가 8개 문항은 최대 128토큰으로 전후 비교한다.
- 정확도, 길이(EOS 제외), 생성 한도 도달, 서두 표현, 중복 4단어 span을 함께 기록한다.
- 자동 채점은 수치/단어/키워드 근사치다. 평가 30개 및 대조 8개의 전후 원문을
  모두 직접 읽어 자동 오판을 별도 기록한다. 핵심 정답과 부연의 오류를 구분한다.
- 성공은 정답 유지와 불필요한 서두/반복 감소가 함께 나타나는 경우다.
  baseline에 측정 대상 서두가 없으면 서두 감소 성공으로 보고하지 않는다.
  짧아진 오답이나 DPO loss 감소만으로 통과라고 하지 않는다.
- reference hash 불변, 유한 loss/gradient/가중치, 실제 매개변수 변화,
  FP32 checkpoint strict 재로딩 및 점수 일치도 확인한다.

산출물은 `outputs/minimind/<실행-ID>/`, 실행 보고서는
`reports/minimind/<실행-ID>.md`에 저장한다. 단일 seed·작은 합성 데이터·한정된
길이의 실험이며 전체 일반 능력 보존이나 폭넓은 사람 선호 정렬을 증명하지 않는다.
