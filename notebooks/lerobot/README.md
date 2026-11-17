# LeRobot 정책 실습

SmolVLA 실습 노트북은 [smolvla_tour.py](smolvla_tour.py)입니다. lerobot의 `modeling_smolvla.py`,
`smolvlm_with_expert.py`와 같은 구조·함수 이름을 따르는 미니 SmolVLA를 만들고 장난감 과제로 학습합니다.
범용 Transformer 연산·장난감 데이터·optimizer 루프는 [smolvla_helpers.py](smolvla_helpers.py)에 있으며,
노트북이 이 파일을 import하므로 같은 폴더에 두어야 합니다.

| 노트북 | 내용 | 자원 |
| --- | --- | --- |
| [smolvla_tour.py](smolvla_tour.py) | 설정, pixel shuffle, prefix·블록 마스크, 레이어 스킵·SA/CA 교대 Action Expert, 시간 임베딩, Flow Matching 학습(회귀 비교), KV 캐시·Euler 추론, Action queue, 비동기 추론 시뮬레이션 | CPU 가능, 학습 버튼(CPU 약 2~3분), 다운로드 없음 |
| [flow_matching_tutor.py](flow_matching_tutor.py) | 1차원 Flow Matching 입문: 직선 경로, 속도장 학습, Euler 샘플링 | CPU, 학습 버튼(수십 초), 다운로드 없음 |

## 설치와 실행

Python 3.12 / Linux 환경에서 이 디렉토리를 기준으로 실행합니다.

```bash
cd notebooks/lerobot
uv sync --locked
uv run marimo edit smolvla_tour.py
```

marimo Python은 `notebooks/lerobot/.venv/bin/python`입니다. 다른 가상환경을 쓰고 있다면 `deactivate` 후 실행하세요.
`lerobot` 패키지는 설치하지 않습니다. `smolvla_tour.py`의 10절(실제 `lerobot/smolvla_base` 살펴보기)은
스위치로만 실행되며 `lerobot[smolvla]` 설치와 약 1 GB 다운로드가 필요하므로 이 환경에서는 동작하지 않습니다.

## 원본과 단순화

- [LeRobot SmolVLA 원본](https://github.com/huggingface/lerobot/tree/main/src/lerobot/policies/smolvla)
  (Apache License 2.0, Copyright The HuggingFace Inc. team)의 구조를 축소·재작성했습니다.
- 작은 차원, GQA 생략, state/action 정규화 생략, 무작위 초기화 VLM(고정) 등 단순화는 코드 주석에 `[실제]`로 표시했습니다.
- 장난감 과제(장애물 우회 2D 경로)의 성공은 실물 성능 재현이 아닙니다.

## ACT 실습

[act_tour.py](act_tour.py)는 LeRobot `modeling_act.py`를 핵심만 남겨 다시 구현하고, 2D 장애물 우회 과제
(좌·우 우회 시연이 50:50으로 섞인 다봉 분포)로 학습·추론까지 확인합니다.

| 절 | 내용 |
| --- | --- |
| 0–1 | `ACTConfig` 축소판, 행동 청크 라벨(`action_delta_indices`)과 `action_is_pad`, 정규화와 배치 구성 |
| 2–4 | ResNet-18 layer4 → 1×1 Conv 이미지 토큰, 2D 사인파 위치, DETR식 인코더(위치는 Q·K에만), 0 입력 + 학습형 쿼리 디코더 |
| 5–7 | CVAE(사후분포·사전분포·정책 구분, KL, 재매개변수화), `MiniACT` 조립, 마스크된 L1 + β·KL |
| 8–9 | CVAE β=10 / β=0.01 / VAE 없음 세 모델 학습 비교, z 샘플링으로 posterior collapse 확인 |
| 10–11 | 행동 큐와 Temporal Ensembling, 폐루프 롤아웃(충돌률·도달률·jerk), 실습 과제 |
| 부록 | 2D 위치 임베딩 유사도, Temporal Ensembling 구현 동등성 검증 (각 버튼) |

```bash
# notebooks/lerobot/에서 실행
uv run marimo edit act_tour.py
```

- 학습은 버튼으로 시작하며 세 모델을 기본 2000 step씩 학습합니다. GPU가 있으면 자동으로 사용하고, CPU에서는 오래 걸릴 수 있습니다.
- `MiniACT`의 기본값 `pretrained=True`는 torchvision ImageNet ResNet-18 가중치(약 45 MB)를 내려받습니다.
  6절 shape 확인 셀부터 다운로드가 일어나므로 오프라인이면 `MiniACT(cfg, pretrained=False)`로 바꾸세요.
- 마스크된 L1은 패딩이 아닌 원소 수로 평균합니다. LeRobot은 패딩 자리를 0으로 만든 뒤 전체 원소 수로 평균하므로 값의 스케일이 약간 다릅니다.
- 원본: [LeRobot ACT](https://github.com/huggingface/lerobot/tree/main/src/lerobot/policies/act)
  (Apache License 2.0, Copyright The HuggingFace Inc. team), [ACT 원 저자 코드](https://github.com/tonyzhaozh/act).
  여러 카메라·`env_state` 처리는 생략했고, 실제 로봇 성능 재현이나 공개 체크포인트 호환은 다루지 않습니다.
  학습 가중치는 메모리에만 있습니다.

## 산출물

학습 로그·평가 지표는 저장소 루트 `outputs/lerobot/<실행 ID>/`, 검토·검증 보고서는
`reports/lerobot/<실행 ID>.md`에 둡니다. 모델·학습·데이터를 실제로 실행한 범위와 구문/작은 계산 검사를 구분합니다.
