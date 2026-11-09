# MiniMind 지식 증류 노트북 검증

- 실행일: 2026-11-09, Asia/Seoul.
- 대상: `notebooks/minimind_distillation.py`, 실행 안내: `notebooks/README.md`.
- 기준 커밋: `e2c613e1b240114cae92215d1f6d680e9a6f5159`. 원본 모델·trainer·dataset은 수정하지 않았다.
- 산출물: [outputs/notebooks/20261109-201717-distillation-notebook/](../../outputs/notebooks/20261109-201717-distillation-notebook/).
- 환경: Linux CPU, Python 3.12.15, PyTorch 2.15.0+cpu, transformers 4.57.6, datasets 5.0.1, marimo 0.25.2, matplotlib 3.11.2. 기존 `notebooks/.venv`·`uv.lock`을 사용했다. 의존성 변경 없음.

## 수업 구성과 원본 대응

hard label/soft target과 텍스트 증류/logits 증류를 구분한 뒤 세 후보의 KL을 수동 계산하여 원본 `trainer.train_distillation.distillation_loss`와 비교한다. 온도·CE 비중 UI, 학생 logits gradient와 교사 gradient 없음, batchmean의 유효 토큰 평균을 확인한다.

실제 `SFTDataset`과 로컬 WordLevel 토크나이저·chat template를 사용한다. 줄바꿈을 별도 토큰으로 보존한다. system을 명시해 확률적 system 추가를 피하고, think/tool 데이터는 사용하지 않는다. 첫 대화는 두 턴이며 유효 정답 수는 `[10,5,5,5]`, 전체 25개다. 위치별 입력·원래 label·다음 label·손실 마스크 표로 assistant 헤더, 응답, EOS+줄바꿈, PAD를 구분한다. 절단 관찰 길이는 8/24/64이고 학습 길이는 항상 64다. 실습 손실 함수는 유효 정답이 전혀 없는 배치를 ValueError로 거부한다.

동물 소리 4개 대화는 노트북에 직접 작성되어 외부 모델·데이터 리비전이 없다. 교사는 Dense hidden 48, 2층, FFN 96이고 학생은 Dense hidden 24, 1층, FFN 48이다. 두 모델 모두 Q heads 4, KV heads 2, dropout 0, 최대 위치 64, flash attention 비활성화다. 교사는 seed 42에서 160 step SFT로 준비하고, seed 7 초기 학생을 동일 복사하여 CE와 CE+KL로 각각 80 step 학습한다. 모든 학습은 full batch 4, AdamW lr 0.005, weight_decay 0, clipping norm 1.0, float32 CPU, torch threads 2다. 교사 logits는 고정 입력에서 no_grad로 한 번 계산해 캐시한다.

원본은 이미 SFT된 교사·학생을 읽고 기본 MoE 교사 → Dense 학생을 사용한다. 이 실습은 다운로드 없이 실행하기 위해 교사를 직접 준비하고 학생을 무작위 초기화한다. 원본과 차이를 본문에 명시했다. 어휘 logits 자르기의 정렬 한계, MoE 학생 aux_loss에 alpha가 곱해지는 점, 함수와 CLI의 alpha 기본값 차이도 설명한다. 마지막에는 예측 질문·해설·5개 수정 과제와 완료 기준이 있다.

## 실행 명령

작업 디렉토리: `/home/mind4u/dev/llm_playground/notebooks`.

```bash
MPLBACKEND=Agg MPLCONFIGDIR=/tmp/distillation-mpl \
XDG_CONFIG_HOME=/tmp/distillation-config \
.venv/bin/python minimind_distillation.py

XDG_CONFIG_HOME=/tmp/distillation-config \
timeout 60 .venv/bin/marimo check minimind_distillation.py --strict --format json \
  > ../outputs/notebooks/20261109-201717-distillation-notebook/check.json

MPLBACKEND=Agg MPLCONFIGDIR=/tmp/distillation-mpl \
XDG_CONFIG_HOME=/tmp/distillation-config \
.venv/bin/python ../outputs/notebooks/20261109-201717-distillation-notebook/validate.py \
  > ../outputs/notebooks/20261109-201717-distillation-notebook/validation.log 2>&1

MPLBACKEND=Agg MPLCONFIGDIR=/tmp/distillation-mpl \
XDG_CONFIG_HOME=/tmp/distillation-config \
timeout 60 .venv/bin/marimo export html minimind_distillation.py \
  -o ../outputs/notebooks/20261109-201717-distillation-notebook/minimind_distillation.html \
  > ../outputs/notebooks/20261109-201717-distillation-notebook/export.log 2>&1
```

`marimo check`와 HTML export는 내부 로컬 소켓을 사용하는 실행기 때문에 샌드박스 제한 밖에서 수행했다. 기본 스크립트와 경계값 검증은 제한 안에서 실행했다. 세션 캐시는 이 노트북 것만 산출물의 `marimo-session/`으로 옮겼다.

## 결과

기본 스크립트, 검증 스크립트, 정적 검사, HTML export 모두 최종 종료 코드 0. 정적 검사 문제 0개. HTML 약 355 KiB. `validation.json`에 환경과 전체 결과, 각 실행의 `students-*/metrics.json`에 소스 SHA-256·학습 곡선 원자료·확률·생성을 기록했다. PNG 곡선도 확인했다.

| alpha / 온도 / 관찰 길이 | CE 기준선 최종 CE | 비교 학생 최종 CE | 기준선 KL(τ=1) | 비교 학생 KL(τ=1) | 학생 첫 소리 생성 정답 |
| --- | --- | --- | --- | --- | --- |
| 0.5 / 2 / 24 | 0.148371 | 0.138477 | 0.143355 | 0.135521 | 둘 다 2/4 |
| 1 / 1 / 8 | 0.148371 | 0.148371 | 0.143355 | 0.143355 | 둘 다 2/4 |
| 0 / 4 / 64 | 0.148371 | 0.139611 | 0.143355 | 0.136693 | 둘 다 2/4 |

교사 CE는 0.001263, 유효 정답률 100%, 첫 소리 생성 4/4다. 초기 학생 CE는 3.311637이며 학습 후 학생 유효 토큰 정답률은 세 설정 모두 92%다. EOS·줄바꿈·반복되는 응답 토큰도 포함하는 정답률이므로 소리 생성 2/4와 구별해야 한다. 기본 증류의 온도 보정 KL은 0.478472, CE 기준선은 0.871732다. **교사 분포 적합도가 개선됐지만 생성 정답 수는 개선되지 않았다.** 교사 자체 KL이 미세한 음수로 계산되는 경우(-1.49e-10)는 float32 오차이며 본문에서도 수치 오차를 설명한다.

검증 스크립트의 모든 assertion 통과:

- 수동 KL과 원본 함수 일치, KL 방향, N배 반복 후 batchmean 불변.
- teacher eval·requires_grad False·gradient 없음 및 학습 전후 state_dict 불변.
- 두 학생이 초기 가중치에서 실제 갱신되며 alpha=1에서 두 학생 가중치 완전 동일.
- 마스킹된 logits gradient 0, 유효 위치 gradient 존재.
- 마스킹된 logits를 임의로 크게 바꿔도 CE/KD 불변.
- 전부 -100인 배치를 ValueError로 거부.
- JSONL → 실제 Dataset → DataLoader, 패딩 마스크·토큰 사전·유효 정답 수 검증.

검증 실행 경로:

- 기본: `outputs/notebooks/20261109-201913-d0d5a9-distillation/students-8e9054/`.
- alpha=1: `outputs/notebooks/20261109-201915-981dac-distillation/students-8620c3/`.
- alpha=0: `outputs/notebooks/20261109-201917-159455-distillation/students-3c70f2/`.

## 실패·경고와 검증 한계

초기 검증 보조 스크립트가 marimo 실행 후 닫힌 matplotlib figure를 찾다가 IndexError가 발생했다. 노트북 실행 오류는 아니었으며, 반환된 곡선 원자료로 그림을 다시 그리도록 수정 후 세 설정 전체를 재검증했다. HTML export 종료 시 multiprocessing resource_tracker가 semaphore 1개 정리 경고를 남겼으나 HTML 생성과 모든 셀 실행은 완료됐다. 최초 샌드박스 내 정적 검사는 완료되지 않아, 로컬 소켓 사용이 가능한 제한 밖 검사 결과를 최종 판정에 사용했다.

별도 검증 분할·실제 한국어 모델 가중치·서브워드 tokenizer·MoE·GPU·AMP·DDP·누적·원본 trainer 전체·checkpoint 재개는 실행하지 않았다. 학생의 암기 성능과 KL 적합도만 관찰했으며 일반화, 대형 모델 효율·속도·메모리 절감률은 주장하지 않는다. 모델 체크포인트는 저장하지 않는다. 원본 trainer나 다른 프로젝트의 전체 테스트는 변경 범위에 포함하지 않았다.
