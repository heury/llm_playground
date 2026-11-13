# LeRobot 정책 실습

LeRobot 정책 구현을 직접 학습시키고 뜯어보는 marimo 노트북 모음입니다.

| 노트북 | 내용 |
|---|---|
| [act_policy_lab.py](act_policy_lab.py) | ACT(Action Chunking with Transformers): 데이터 청크·패딩, 위치 임베딩, ResNet 백본, 인코더·디코더, CVAE 손실, 잠재 변수 분석, 행동 큐와 시간 앙상블 비교 |

## 원본 코드

- [huggingface/lerobot `src/lerobot/policies/act`](https://github.com/huggingface/lerobot/tree/main/src/lerobot/policies/act)
  (revision `e6ffde60312365c5452c46189a26872483943154`)
- 노트북의 `ACTConfig`, `ACT`, `ACTEncoder(Layer)`, `ACTDecoder(Layer)`, `ACTPolicy`,
  `ACTTemporalEnsembler`, 위치 임베딩 코드는 원본(Apache License 2.0,
  Copyright 2024 Tony Z. Zhao and The HuggingFace Inc. team)을 단순화한 것입니다.
  `lerobot` 패키지는 설치하지 않으며 `PreTrainedPolicy`, processor 파이프라인, 데이터셋은
  최소 기능만 노트북 안에서 다시 구현합니다.

## 설치와 실행

Python 3.12 환경을 사용합니다. 이 디렉토리에서 실행하세요.

```bash
cd notebooks/lerobot
uv sync --locked
uv run marimo edit act_policy_lab.py
```

marimo의 Python 환경은 `notebooks/lerobot/.venv/bin/python`입니다.
다른 가상환경이 활성화되어 있으면 `uv`가 경고를 낼 수 있으니 `deactivate` 후 실행하세요.

## 실행 시간과 장치

- 데이터는 노트북이 즉석에서 만듭니다. 외부 데이터·가중치를 내려받지 않습니다
  (ResNet18은 ImageNet 가중치 없이 무작위 초기화).
- 7장 학습은 폼의 **학습 시작** 버튼을 눌러야 실행됩니다. 기본값(2000 스텝 × 3개 설정)은
  RTX 3080 Ti에서 설정당 40~70초 걸렸습니다. CPU에서는 스텝 수를 줄이세요.
- 9-1장 롤아웃 비교는 **비교 실행** 버튼으로 시작합니다.

## 무인 실행(검증용)

환경 변수로 버튼을 누르지 않고 끝까지 실행할 수 있습니다.

```bash
cd notebooks/lerobot
ACT_LAB_AUTORUN=1 ACT_LAB_STEPS=200 uv run marimo export html act_policy_lab.py \
  -o ../../outputs/lerobot/<실행 ID>/act_policy_lab.html
```

- `ACT_LAB_AUTORUN=1`: 학습 폼과 비교 버튼을 기본값으로 실행
- `ACT_LAB_STEPS=N`: 학습 스텝 수를 N으로 덮어쓰기 (0 또는 미설정이면 폼 값 사용)

실행 산출물은 `outputs/lerobot/<실행 ID>/`, 보고서는 `reports/lerobot/<실행 ID>.md`에 저장합니다.
