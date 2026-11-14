# LeRobot 정책 실습

LeRobot 정책 구현을 직접 학습시키고 뜯어보는 marimo 노트북 모음입니다.

| 노트북 | 내용 |
|---|---|
| [act_tour.py](act_tour.py) | ACT(Action Chunking with Transformers): 2D 집어서 옮기기 과제, 학습 배치·패딩, 위치 임베딩, 인코더·디코더, CVAE 손실, 단일 step BC와의 비교, 행동 큐·시간 앙상블, 잠재 변수 `z` 분석, SmolVLA 비교 |
| [smolvla_tour.py](smolvla_tour.py) | SmolVLA: 작은 SmolVLM(비전 인코더·픽셀 셔플·Llama형 디코더) VQA 사전 학습, VLM+행동 전문가 구조(자기/교차 어텐션 교대), 블록 어텐션 마스크, Flow Matching 학습, KV 캐시 추론, 행동 큐·폐루프 평가, 다봉 분포, 무작위 VLM 비교 |

## 원본 코드

- [huggingface/lerobot `src/lerobot/policies/act`](https://github.com/huggingface/lerobot/tree/e6ffde60312365c5452c46189a26872483943154/src/lerobot/policies/act)
  (revision `e6ffde60312365c5452c46189a26872483943154`)
- 노트북의 ACT 코드는 원본(Apache License 2.0, Copyright 2024 Tony Z. Zhao and The HuggingFace Inc. team)을
  옮겨 온 것입니다. `lerobot` 패키지는 설치하지 않습니다.
- [huggingface/lerobot `src/lerobot/policies/smolvla`](https://github.com/huggingface/lerobot/tree/e6ffde60312365c5452c46189a26872483943154/src/lerobot/policies/smolvla)
  및 `policies/common/flow_matching.py`, `policies/common/vla_utils.py` (같은 revision)
- 노트북의 SmolVLA 코드는 원본(Apache License 2.0, Copyright 2025–2026 The HuggingFace Inc. team)을 옮겨 와
  `transformers` 없이 동작하도록 축소·수정한 것입니다. SmolVLM2 가중치는 내려받지 않고 작은 VLM을 노트북에서 직접 학습합니다.

## 설치와 실행

Python 3.12 환경을 사용합니다. 이 디렉토리에서 실행하세요.

```bash
cd notebooks/lerobot
uv sync --locked
uv run marimo edit act_tour.py
```

marimo의 Python 환경은 `notebooks/lerobot/.venv/bin/python`입니다.
다른 가상환경이 활성화되어 있으면 `uv`가 경고를 낼 수 있으니 `deactivate` 후 실행하세요.

- 데이터는 노트북이 즉석에서 만듭니다. 외부 데이터·가중치를 내려받지 않습니다.
- ACT는 6절 버튼을 누르면 학습 3회가 실행됩니다. GPU 기본 3000 step, CPU 기본 500 step입니다. 본문의 과거 시간·성공률은 현재 실행의 보장값이 아닙니다.
- SmolVLA는 `uv run marimo edit smolvla_tour.py`로 엽니다. 4절 버튼을 누르면 VLM 사전 학습과 정책 학습이 이어집니다.
  CPU에서는 step 수를 자동으로 줄이며 결과 품질도 낮아질 수 있습니다.
  14절의 무작위 VLM 비교는 버튼을 눌러야 실행됩니다.

실행 산출물은 `outputs/lerobot/<실행 ID>/`, 보고서는 `reports/lerobot/<실행 ID>.md`에 저장합니다.

학습 모델은 메모리에만 유지됩니다. marimo는 변수 의존성에 따라 실행하므로 설정·학습 셀 변경은 하위 학습·평가를 다시 실행할 수 있습니다.
처음부터 비교하려면 커널을 재시작하고 같은 시드로 실행하세요. 평가 함수는 시작할 때 행동 큐를 초기화합니다.
두 과제는 합성 데이터 기반 구조 실습이며, 실제 로봇 성능이나 공개 SmolVLA 체크포인트 성능을 재현하지 않습니다.
