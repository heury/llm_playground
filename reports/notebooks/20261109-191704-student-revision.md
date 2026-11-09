# 수강생 관점 검토 반영 및 검증

- 실행일: 2026-11-09 (Asia/Seoul)
- 대상: `notebooks/minimind_structure.py`, `notebooks/minimind_lora.py`, `notebooks/README.md`
- 기준: [수강생 관점 검토](20261109-191249-student-review.md)의 9개 개선점.
- 산출물: [`outputs/notebooks/20261109-191704-student-revision/`](../../outputs/notebooks/20261109-191704-student-revision/)
- 환경: Linux x86_64, Python 3.12.15, torch 2.15.0+cpu, transformers 4.57.6, marimo 0.25.2, matplotlib 3.11.2. 기존 `notebooks/uv.lock`과 환경을 사용했다.

## 반영 내용

| 검토 항목 | 반영 |
| --- | --- |
| Embedding과 LM head의 연결 | 고양이 ID가 선택하는 임베딩 행을 직접 비교하고, 마지막 hidden state × LM head의 전치 행렬을 원본 logits와 비교 |
| 과밀한 Attention 셀 | 사람이 정한 3토큰·1 head 계산부터 시작해 실제 Q/K/V, RoPE·GQA, 점수·mask·가중합을 별도 셀로 분리 |
| 점수·확률·손실·생성 연결 | 학습 전후 어휘별 logits·확률과 한 정답의 cross entropy 비교, argmax·토큰 추가·eos 종료를 직접 반복 |
| LoRA 행렬 계산 | 원본 LoRA 클래스로 직사각형 선형층의 Wx, Ax, B(Ax), 합산·병합 출력과 rank 상한을 관찰 |
| 동결과 역전파 | 새 선형층에서 두 step 동안 W/A/B의 requires_grad, gradient norm, 가중치 변화량을 표로 관찰. eval과 no_grad도 구분 |
| 시각화의 문장 대응 | 같은 첫 블록·첫 head의 학습 전후 Attention을 실제 토큰 축과 상세 표로 비교. 한글 폰트가 없으면 위치 번호와 토큰 표로 대체 |
| 과제 판정 기준 | 예상 shape, 관측 shape, 손실, 정답 수, rank별 매개변수·파일 크기 등 기록 항목과 힌트·해설 추가 |
| 학습 대상과 전체 크기의 혼동 | 원본 W 보존·A/B 추가를 명시하고 원본 모델·어댑터·병합 모델의 보유량·학습 대상·파일 크기를 구분 |
| marimo 사용 안내 | 셀 데코레이터, 인자·return, 셀 내부 변수, 반응형 재실행과 모델 초기화 규칙을 설명 |

원본 MiniMind 모델과 LoRA 구현은 수정하지 않았다. 데이터는 기존 네 문장이고 외부 모델·데이터 리비전이나 체크포인트는 없다. 원본과 실습 소스 SHA-256은 `metrics.json`에 기록했다. 무작위 초기화 seed 42, LoRA seed 7, 추가 작은 계산의 seed 11/12를 사용한다.

기본 모델은 hidden size 32, 2층, Query head 4개, KV head 2개, intermediate size 64인 Dense 모델이다. 전체 학습은 AdamW lr=0.005, 기본 100 step, LoRA는 rank 4, AdamW lr=0.01, 120 step으로 유지했다. 새 gradient 관찰은 별도의 입력 4/출력 3/rank 2 선형층에서 SGD lr=0.1로 두 step만 수행한다. 생성은 greedy, top_k=0, top_p=1, temperature=1, 최대 5개 새 토큰이다.

## 실행 명령과 결과

저장소 루트에서 최종 전체 셀 및 지표 검증:

```bash
XDG_CONFIG_HOME=/tmp/minimind-marimo-config \
MPLCONFIGDIR=/tmp/minimind-marimo-mpl MPLBACKEND=Agg \
notebooks/.venv/bin/python \
  outputs/notebooks/20261109-191704-student-revision/validate.py
```

`validate.py`는 각 노트북의 `app.run()`을 실행하고 기본 정답 문자열·확률 손실·생성 종료를 추가 검증한다. 그림 참조를 수집하기 위해 `plt.subplots`를 동일 반환값을 유지하는 수집 함수로 감싸 PNG로 보관했다. `validation.log`, `metrics.json`, 네 개 PNG에 결과를 저장했다. 종료 코드 0이며 최종 실행에는 warning이 없었다.

`notebooks/`를 작업 디렉토리로 사용한 최종 정적 검사와 HTML 실행:

```bash
XDG_CONFIG_HOME=/tmp/minimind-marimo-config \
.venv/bin/marimo check minimind_structure.py minimind_lora.py --strict --format json

XDG_CONFIG_HOME=/tmp/minimind-marimo-config \
MPLCONFIGDIR=/tmp/minimind-marimo-mpl MPLBACKEND=Agg \
.venv/bin/marimo export html minimind_structure.py \
  -o ../outputs/notebooks/20261109-191704-student-revision/minimind_structure.html

XDG_CONFIG_HOME=/tmp/minimind-marimo-config \
MPLCONFIGDIR=/tmp/minimind-marimo-mpl MPLBACKEND=Agg \
.venv/bin/marimo export html minimind_lora.py \
  -o ../outputs/notebooks/20261109-191704-student-revision/minimind_lora.html
```

정적 검사와 두 HTML 내보내기 모두 종료 코드 0이었다. 검사와 HTML 실행 커널의 내부 소켓 사용을 위해 해당 명령은 로컬 소켓 제한 밖에서 실행했다.

- marimo strict: 두 파일, 문제 0건 (`check.json`).
- HTML: 모델 약 409 KiB, LoRA 약 314 KiB. 실행 로그는 `model-export.log`, `lora-export.log`에 보관했다.

## 수치 검증 결과

| 항목 | 결과 |
| --- | --- |
| 모델의 원래 정답 생성 | 4/4 |
| 전체 학습 최종 손실 | 0.239934 |
| 고양이 는 다음의 야옹 한 위치 손실 | 2.949241 → 0.014988 |
| 직접 작성한 생성 반복 | 야옹 → 한다 → . → eos, 원본 generate와 일치 |
| KV cache 최대 오차 | 0 |
| LoRA의 새 정답 생성 | 4/4 |
| LoRA 최종 손실 | 0.242767 |
| 복원 / 병합 최대 오차 | 0.001121 / 0.005561 |

작은 gradient 관찰에서는 W는 두 step 모두 gradient가 없고 변화량이 0이었다. A는 첫 step에서 gradient가 0, 두 번째 step에서 gradient norm 0.004259였다. B는 첫 step에서 gradient norm 0.015059였다. B를 0으로 초기화한 경우의 첫 갱신을 실제로 확인했다.

같은 float16 저장 기준 파일 크기는 원본 45,947 bytes, 어댑터 5,261 bytes, 병합 모델 46,009 bytes였다. 이 모델의 파일 비교이며 전체 메모리 절감률을 측정한 결과는 아니다.

새 내장 assertion과 기존 assertion 모두 통과했다. 새 검증은 ID와 임베딩 행 일치, 단일 head의 확률 합과 미래 확률 0, RoPE 길이 보존, 공유 K 일치, LM head 직접 계산, 한 위치의 손실 일치, 단일층 LoRA의 합산과 병합 일치, 초기 A gradient 0과 다음 갱신, eval과 no_grad 구분을 포함한다.

## 시각·문서 확인과 한계

Attention 비교 PNG를 직접 열어 한글 토큰 축, query/key 축, 학습 전후 두 그림을 읽을 수 있음을 확인했다. 브라우저 편집기 전체 조작은 검증하지 않았다. README 상대 링크, Python 구문, 줄 끝 공백도 확인했다.

초기 임시 스크립트 실행도 두 파일 모두 종료 코드 0이었으나, 임베딩 행을 scalar로 표시할 때 warning이 발생해 detach를 추가했다. 초기 로그는 `initial-*-script.log`로 보존하여 최종 `validation.log`, HTML 실행 로그와 구분했다. 자동 생성 세션 캐시는 `marimo-session/`으로 옮겼다.

이번 검증은 기본 CPU 설정과 내장 샘플에 한정한다. 모든 rank·step 조합, GPU, MoE, 다른 운영체제, 한글 폰트 없는 환경, 별도 데이터의 일반화, 실제 수강생 이해도는 검증하지 않았다. 이전 rank 경계 검증을 이번 새 셀의 통과 결과로 해석하지 않는다.
