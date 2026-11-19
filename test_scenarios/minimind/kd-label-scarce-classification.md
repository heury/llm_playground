# 라벨이 부족한 감정 분류에서 MoE → Dense 지식 증류

## 목적과 시작점

- 교사: `caspar/MiniMind-MoE-198M-A64M-Mixed-SFT-v1.0`, 리비전 `a742772a0ad4db7480064bd2abccc174062a3de8`
- 학생: `caspar/MiniMind-Dense-64M-Mixed-SFT-v1.0`, 리비전 `7597c22aa4d3e7b2ac824b4b4be01158cccf1b8d`
  (MoE 스냅샷에는 토크나이저가 없으므로 같은 어휘 6400의 Dense 토크나이저를 공용으로 쓴다.)

공개 체크포인트끼리는 교사가 학생보다 낫지 않다(GSM8K CE 1.257 vs 1.240,
[distillation-tour 보고서](../../reports/notebooks/20261113-132831-distillation-tour.md)).
그래서 교사에게만 라벨 2,000개를 주어 특화하고, 학생은 라벨 200개와 라벨 없는 문장
2,000개만 가진 상황에서 증류가 라벨 부족을 얼마나 메우는지 본다.

## 데이터

`dair-ai/emotion` (config `split`, 리비전 `cab853a1dbdf4c42c2b3ef2173804746df8825fe`),
6클래스 sadness, joy, love, anger, fear, surprise. train 16,000개를 seed 42로 섞어 겹치지 않게 나눈다.

| 묶음 | 크기 | 용도 |
| --- | --- | --- |
| 교사 라벨 | train 0~1999 | 교사 SFT, 상한 대조군 S5 |
| 학생 라벨 | train 2000~2199 | 모든 학생 실험의 라벨 |
| 라벨 없는 문장 | train 2200~4199 | S3·S4 (정답 라벨은 사용하지 않음) |
| 평가 | test 2,000개 전체 | 최종 판정. 설정 조정에 쓰지 않음 |

프롬프트(영어, 기반 모델이 영어 중심):
`Classify the emotion expressed in the text. Choose one of: sadness, joy, love, anger, fear, surprise.\nText: {text}\nAnswer with one word.`
답변은 라벨 단어 하나이고, 빈 think 블록은 제거한다. 손실은 답변 토큰과 EOS에만 둔다.

## 학습 (모든 설정은 실행 전에 고정)

공통: 전체 매개변수, FP32 AdamW(wd 0.01), BF16 autocast, batch 16, clip 1,
lr 3e-5 코사인(원본 `get_lr` 식: 0.1~1.0배), KD 온도 T=2, `T²·KL(teacher‖student)` 전체 어휘.

- **교사 SFT**: 교사 라벨 2,000개 CE(+ MoE aux loss), 375 step(3 epoch), seed 42.
- 학생 실험은 모두 **250 step**(샘플 4,000개)이며 seed 42·43·44로 3번 반복한다.
  - **S1 CE-200**: 학생 라벨 200개 CE만 사용
  - **S2 KD-200**: 같은 200개에 `0.5·CE + 0.5·KD`
  - **S3 KD-200+U2000**: S2에 라벨 없는 2,000개를 더하고 이 문장들은 KD만 쓴다. 이어지는 답변은
    교사가 6개 후보 중 가장 높게 점수를 준 라벨이다(teacher-forced).
  - **S4 Pseudo-200+U2000**: S3와 같은 시퀀스에 CE만 쓴다(교사의 hard label). soft와 hard를 비교하는 대조군
  - **S5 CE-2000**: 교사 라벨 2,000개 CE. 정답 라벨이 충분할 때의 상한

## 평가와 판정

- 주 지표: test 2,000개에서 6개 후보의 `라벨+EOS` 로그확률 합 argmax로 매긴 정확도와 macro-F1
- 보조 지표: 6개 후보로 정규화한 분포의 정답 NLL, ECE(15 bins), 교사 argmax와의 일치율,
  `KL(교사‖학생)`(6클래스)
- 형식 확인: test 앞 300개의 greedy 생성(최대 8토큰)에서 유효한 라벨 비율과 정확도
- 학습 전 교사·학생의 zero-shot 점수도 기록한다.

판정(사전 고정):
1. **전제**: 특화된 교사의 정확도가 S1 평균보다 5%p 이상 높아야 한다. 아니면 지식 전달은 판정하지 않는다.
2. **증류 효과**: S3 평균 정확도가 S1 평균보다 높고, 그 차이가 seed 표준편차의 2배보다 커야 한다.
3. S2와 S1, S3와 S4의 비교는 효과 크기만 보고한다(통과·실패를 판정하지 않음).
4. 손실·기울기·가중치가 유한해야 하고, 교사 가중치는 증류 중 바뀌지 않아야 한다(해시 확인).

단일 데이터셋·영어·짧은 학습 실험이다. 일반 능력 보존이나 다른 과제로의 일반화는 증명하지 않는다.
실행기: [kd_classification.py](../../projects/minimind/scripts/knowledge_sft/kd_classification.py),
산출물은 `outputs/minimind/<실행 ID>/`, 보고서는 `reports/minimind/<실행 ID>.md`.
