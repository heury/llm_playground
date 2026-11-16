# LeRobot 정책 실습

구조 해부 노트북은 [smolvla_tour.py](smolvla_tour.py)입니다. [smolvla.md](smolvla.md)는 같은 구현에 맞춘 구조 안내입니다.
PyTorch 텐서·역전파와 Transformer 어텐션·RoPE·KV 캐시 기초를 아는 독자를 대상으로 합니다.
본문은 고정 입력으로 VLM·행동 전문가 조립, 블록 마스크, Flow Matching, KV 캐시에 집중합니다.
VLM 분석, 장난감 과제 학습·평가, 비동기 큐는 보충 노트북에서 진행합니다.

| 노트북 | 내용 | 자원 |
| --- | --- | --- |
| [smolvla_tour.py](smolvla_tour.py) | 핵심 구조와 동작 원리 | 설명·작은 예제는 CPU, 공개 가중치는 GPU 권장 |
| [smolvla_vlm_experiments.py](smolvla_vlm_experiments.py) | VQA, 층별 좌표 프로브, 절단 전후 연산 시간 | 모델 로드·실험 버튼, GPU 권장 |
| [smolvla_training.py](smolvla_training.py) | 기본 평가, 파인튜닝, 폐루프 평가, 지시문·노이즈 비교 | 정책 로드·학습·평가 버튼, GPU 권장 |
| [smolvla_async_execution.py](smolvla_async_execution.py) | 행동 큐, trim, 청크 혼합 | CPU, 다운로드 없음 |
| [flow_matching_intro.py](flow_matching_intro.py) | 1차원 Flow Matching 입문: 직선 경로, 속도장 학습, Euler 샘플링 | CPU, 학습 버튼(수십 초), 다운로드 없음 |

각 노트북은 별도 앱으로 열며 다른 앱의 실행 상태나 메모리를 요구하지 않습니다.
VLM 분석과 학습 노트북은 `smolvla_tour.py`의 이름 있는 정의 셀을 재사용합니다.
모델 구현을 수정할 때는 본문의 `expert_definitions`, `flow_definitions`를 확인하세요.
장면 생성·정답 궤적·환경 이동·성공 판정은 학습 노트북에 있으며, VLM 분석은 그 `scene_definitions` 셀을 재사용합니다.

## 설치와 실행

Python 3.12 / Linux 환경에서 이 디렉토리를 기준으로 실행합니다.

```bash
cd notebooks/lerobot
uv sync --locked
uv run marimo edit smolvla_tour.py
# 보충 실습은 파일명을 바꿔 실행합니다.
uv run marimo edit smolvla_vlm_experiments.py
uv run marimo edit smolvla_training.py
uv run marimo edit smolvla_async_execution.py
```

marimo Python은 `notebooks/lerobot/.venv/bin/python`입니다. 다른 가상환경을 쓰고 있다면 `deactivate` 후 실행하세요.
`lerobot` 패키지는 설치하지 않으며, 실제 핵심 코드는 노트북 내부에 이식했습니다.

## 학습 경로와 실행 비용

| 경로 | 내용 | 실행 조건 |
| --- | --- | --- |
| 가중치 없이 읽기 | 본문 1~8절 설명/코드, 3절 재배열, 6절 작은 마스크, 비동기 실행 노트북 | CPU, 다운로드 불필요 |
| 공개 가중치 실험 | 본문 2절 로드 → 3절 커넥터 확인 → 5절 조립 → 6절 실제 마스크·8절 캐시 비교 | 각 실행 버튼, GPU 권장 |
| 학습·평가 | 본문 7절 토이 학습, 학습 노트북의 기본 평가·파인튜닝·학습 후 평가 | 각 실행 버튼 |

- 2절 로드 버튼은 SmolVLM2(약 2 GB)와 정책(약 0.9 GB)을 Hub에서 받아 `~/.cache/huggingface/`에 둡니다.
  5절 조립은 내려받은 로컬 가중치로 별도 fp32 정책을 만듭니다. 전체 VLM과 정책이 함께 메모리를 사용합니다.
  VLM 분석 노트북은 VLM만 받고, 학습 노트북은 fp32 정책만 메모리에 유지합니다.
- 7절 토이 모델도 3000 step 학습하므로 별도 버튼으로 실행합니다. 입력을 바꾸면 의존 셀이 다시 계산되지만
  긴 실험·학습·반복 평가는 실행 버튼을 다시 눌러야 합니다.
- 학습 노트북의 기본 모델과 학습 후 평가는 같은 100개 장면, 장면 seed 123, noise seed 456으로 비교합니다.
  지시문 교체도 같은 장면·같은 noise seed를 쓰며, 여러 seed 반복은 별도 실험입니다.
- 파인튜닝은 RTX 3080 Ti의 과거 측정에서 1500 step 약 7분·최대 약 7 GB였습니다.
  현재 환경의 보장값이 아닙니다. CPU 기본값은 구조 확인용 20 step이며 학습/추론이 느릴 수 있습니다.
- 학습 가중치는 메모리에만 있습니다. 모델 로드 재실행·앱/커널 재시작 때 학습 결과를 잃을 수 있고 로그만 디스크에 남습니다.
  CUDA나 가중치가 없으면 설명·작은 마스크·큐 시뮬레이션으로 진행할 수 있습니다.

## 원본과 단순화

- [LeRobot SmolVLA 원본](https://github.com/huggingface/lerobot/tree/e6ffde60312365c5452c46189a26872483943154/src/lerobot/policies/smolvla),
  `policies/common/flow_matching.py`, `vla_utils.py`, `inference/execution.py`의 고정 리비전
  `e6ffde60312365c5452c46189a26872483943154`를 비교합니다.
- 원본은 Apache License 2.0, Copyright 2025–2026 The HuggingFace Inc. team입니다. 안내와 출처를 유지합니다.
- VLM: [`HuggingFaceTB/SmolVLM2-500M-Video-Instruct`](https://huggingface.co/HuggingFaceTB/SmolVLM2-500M-Video-Instruct),
  revision `4cde9d208e19519c36917aa170fffc7e381aa0ad`.
- 정책: [`lerobot/smolvla_base`](https://huggingface.co/lerobot/smolvla_base),
  revision `5e8d12a6e2975b0e5e5fce7c8caf47c371d257b6`.
- fp32 이식 구현이며 프로세서·일부 설정 옵션·RTC·실제 서버/로봇 제어는 생략했습니다.
  2D 점 과제의 성공은 실물 성능 재현이 아닙니다. 정규화·입출력 단위·시간 규약은 구조 안내에서 구분합니다.
- 비동기 실행 노트북은 고정 tick 지연의 가짜 정책으로 큐 정렬·혼합을 비교합니다. 네트워크/스레드 스케줄링이나 실시간 주기를 보장하지 않습니다.

## ACT 단계별 실습

ACT 실습은 구조 이해와 선택 실험으로 나눕니다. 두 노트북은 각각 실행할 수 있으며,
실험 노트북은 구조 노트북의 이름 있는 셀을 가져와 같은 모델 구현을 사용합니다.

| 노트북 | 내용 | 실행 조건 |
| --- | --- | --- |
| [act_tour.py](act_tour.py) | 고정 배치·패딩, 위치 임베딩, ResNet-18, DETR 층, CVAE, ACT 조립, L1 + KL, `z=0` 추론, 시간 앙상블 | 작은 CPU 계산, 다운로드·반복 학습 없음 |
| [act_experiments.py](act_experiments.py) | 2D 갈림길 환경·시연, MSE/L1 MLP 비교, 실제 크기 매개변수 집계, β별 학습·잠재 변수 분석, 청크 실행·외란 평가, 다른 정책 비교 | MLP·ACT 학습·반복 평가는 각 버튼으로 실행 |

```bash
# notebooks/lerobot/에서 실행
uv run marimo edit act_tour.py
uv run marimo edit act_experiments.py
```

[act.md](act.md)는 관련 구조 설명입니다. 본문의 고정 수열·빈 이미지는 텐서 흐름 확인용이며 행동 성능을 검증하지 않습니다.
두 노트북 모두 다운로드 없이 ResNet-18을 무작위 초기화합니다.
보충 실험의 MLP는 CPU 약 10초, ACT는 RTX 3080 Ti의 과거 측정에서 3000 step에 모델당 약 1분이었습니다.
CPU 기본값은 600 step입니다. 같은 seed라도 GPU 비결정성으로 성공률이 크게 달라질 수 있으므로
보충 실험의 기준 실행 표와 한계를 함께 읽으세요. 학습 가중치는 메모리에만 보관하고 로그·지표는 루트 `outputs/lerobot/`에 저장합니다.
LeRobot `policies/act` 고정 리비전 `e6ffde60312365c5452c46189a26872483943154`(Apache License 2.0)의 구조를 따라
축소·재작성했으며, 실제 로봇 성능 재현이나 공개 체크포인트 호환은 다루지 않습니다.

## 산출물

학습 로그·평가 지표는 저장소 루트 `outputs/lerobot/<실행 ID>/`, 검토·검증 보고서는
`reports/lerobot/<실행 ID>.md`에 둡니다. 모델·학습·데이터를 실제로 실행한 범위와 구문/작은 계산 검사를 구분합니다.
