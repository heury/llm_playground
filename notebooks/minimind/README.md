# MiniMind marimo 실습

원본 모델과 trainer를 직접 가져와 데이터·텐서 계산·손실·학습 결과를 연결하는 실습입니다.
Python 함수·클래스, 텐서 shape와 행렬 곱, 기본 확률·미분을 선수 지식으로 가정합니다.
각 노트북의 도입에서 추가 선수 지식과 준비물을 확인하세요.

## 환경과 실행

저장소 루트에서 시작합니다. 각 모델 프로젝트의 별도 환경과 구분하세요.

```bash
cd notebooks/minimind
uv sync --locked
uv run marimo edit minimind_model_tour.py
```

marimo에서 사용할 Python은 `notebooks/minimind/.venv/bin/python`입니다.
Python 3.12와 의존성은 이 디렉토리의 `pyproject.toml`, `uv.lock`으로 관리합니다.
Linux/Windows용 torch는 CUDA 13.2 빌드이며, GPU 사용에는 호환 NVIDIA 드라이버가 필요합니다.
모델 구조·사전 학습·SFT·LoRA·DPO 코드는 CPU 대체 경로가 있지만 같은 규모의 학습은 느릴 수 있습니다.
GRPO·증류·VLM·Omni 실습의 학습 경로는 CUDA GPU를 요구합니다.

파일을 열면 의존 셀이 계산되며, 일부 노트북은 **가중치·데이터 다운로드와 학습을 자동으로 시작**합니다.
도입의 다운로드 크기·VRAM·소요 시간과 각 실행 버튼 안내를 먼저 읽으세요.
CPU에서 설명을 읽는 것과 전체 학습이 가능한 것은 다릅니다. 기존 측정 시간·결과는 현재 장치의 보장이 아닙니다.
GPU가 없거나 자원이 부족하면 학습 셀을 실행하지 않고 설명과 원본 코드 대조부터 진행하세요.

## 권장 읽기 순서

| 노트북 | 학습 목표 | 선행 실습 |
| --- | --- | --- |
| [모델 구조](minimind_model_tour.py) | RMSNorm·RoPE·GQA·MLP·CE·생성의 shape와 원본 연산 | 기본 텐서 연산 |
| [사전 학습](minimind_pretrain_tour.py) | 다음 토큰 라벨·누적·스케줄·체크포인트 | 모델 구조 |
| [SFT](minimind_sft_tour.py) | assistant 마스크·정체성 학습·망각 | 사전 학습 |
| [샘플링](minimind_sampling_tour.py) | greedy·temperature·top-k·top-p·repetition penalty, 원본 `generate` 재현 (GPU 필요) | 모델 구조 |
| [LoRA](minimind_lora_tour.py) | 저랭크 갱신·동결·저장·병합 | 모델 구조, SFT |
| [DPO](minimind_dpo_tour.py) | 선호 쌍·로그 확률비·reference | SFT |
| [GRPO](minimind_grpo_tour.py) | rollout·보상·advantage·GRPO/CISPO | SFT, 로그 확률 |
| [증류](minimind_distillation_tour.py) | 교사 분포·KL·온도·CE 혼합 | SFT |
| [VLM 구조](minimind_vlm_tour.py) | 시각 특징·Projector·placeholder 결합 | 모델 구조, SFT |
| [VLM 학습](minimind_vlm_training_tour.py) | 정렬→시각 SFT·원본 trainer·재개 | VLM 구조, 사전 학습 |
| [Omni 구조](minimind_omni_tour.py) | Thinker/Talker·코드북·지연·음성 학습 | SFT, VLM 구조 |
| [VLM에서 Omni로](minimind_omni_vlm_tour.py) | 시각 능력 보존과 음성 학습 비교 | VLM 구조, Omni 구조 |
| [Omni trainer](minimind_omni_training_tour.py) | 원본 trainer·학습 모드·재개·드문 갱신 | Omni 구조, VLM에서 Omni로 |

## 읽기·재실행·결과 해석

위젯을 바꾸면 해당 값에 의존하는 셀과 결과가 다시 계산됩니다. 이전 학습을 이어가는지,
베이스 가중치에서 다시 시작하는지는 각 학습 절의 안내와 모델 초기화 코드를 확인하세요.
비교할 때 데이터·시드·학습량·평가 입력을 고정하고, 손실뿐 아니라 생성 결과와 기존 능력도 함께 봅니다.

설명 수식은 LaTeX, 표는 Markdown, 구조·흐름 도식은 Mermaid로 표시합니다.
결과표는 Markdown이 기본이며, 접힌 **정렬·검색·페이지 보기**에서 기존 표 기능을 사용할 수 있습니다.
`notebook_display.py`는 표시 보조 모듈이며 실행할 실습 노트북은 아닙니다.

## 원본과 산출물

언어 실습은 `../../projects/minimind/`, VLM은 `../../projects/minimind-v/`,
Omni는 `../../projects/minimind-o/`의 실제 구현을 사용합니다. 원본 코드의 라이선스·저작권 안내를 유지합니다.
가중치와 데이터는 소스와 별개이며, 노트북의 모델·데이터 리비전과 각 모델 카드의 조건을 확인하세요.

새 실행 결과는 `../../outputs/notebooks/<실행 ID>/`, 보고서는
`../../reports/notebooks/<실행 ID>-*.md`에 저장합니다. 현재 검토의 정적 점수·구문 검사와
실제 학습·추론·화면 검증을 구분하여 기록합니다.
