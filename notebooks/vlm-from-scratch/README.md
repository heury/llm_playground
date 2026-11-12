# 처음부터 만드는 시각·언어 모델(VLM)

간단한 이미지 캡션 모델부터 고급 멀티모달 AI 시스템까지, 12부 실습으로 시각·언어 모델을 직접 만듭니다.

## 실습 구성

| 파트 | 주제 | 구현할 내용 |
|------|-------|-------------------|
| 01 | [최소 구성의 VLM](notebooks/part-01-minimal-vlm.py) | Flickr8k로 학습하는 ViT + SmolLM 캡션 모델 |
| 02 | [객체 탐지](notebooks/part-02-object-detection.py) | 경계 상자를 JSON으로 출력하는 지시 미세 조정 모델 |
| 03 | [시각적 질의응답](notebooks/part-03-visual-qa.py) | 이미지에 관한 질문에 답하는 VQA 모델 |
| 04 | [멀티태스크](notebooks/part-04-multi-task.py) | 캡션·객체 탐지·VQA 통합 모델 |
| 05 | [다중 이미지](notebooks/part-05-multi-image.py) | 이미지 시퀀스의 시간적 변화 추론 |
| 06 | [태스크 라우팅](notebooks/part-06-task-routing.py) | 프롬프트에서 작업 유형 자동 감지 |
| 07 | [사고 과정](notebooks/part-07-chain-of-thought.py) | 단계별 시각적 추론 |
| 08 | [지칭 분할](notebooks/part-08-referring-segmentation.py) | "왼쪽의 개" → 다각형 마스크 |
| 09 | [이미지 편집](notebooks/part-09-image-editing.py) | 자연어 지시에 따른 이미지 편집 |
| 10 | [모델 압축](notebooks/part-10-compression.py) | 양자화·가지치기·지식 증류 |
| 11 | [이미지 생성](notebooks/part-11-image-generation.py) | VLM으로 유도하는 Stable Diffusion 생성 |
| 12 | [OCR → LaTeX](notebooks/part-12-ocr-latex.py) | 수식 인식 |

## 모델 구조

```
이미지 → 비전 인코더(ViT) → 프로젝션 층 → 언어 모델(SmolLM) → 출력
        매개변수 86~304M       선형 층        매개변수 135~360M
```

## 시작하기

이 저장소에서는 12개 실습을 한국어로 번역한 `notebooks/part-*.py` marimo 앱으로 실행합니다.
설명·주석·로그·그래프와 비교표를 번역했으며, 모델 입력 프롬프트·학습 데이터·특수 토큰·코드 식별자는 원문을 유지합니다.
Python 3.12 환경을 사용하며, 다음 명령은 이 저장소의 루트에서 시작합니다.

```bash
cd notebooks/vlm-from-scratch
uv sync --locked
export VLM_RUN_ID="$(date +%Y%m%d-%H%M%S)-vlm"
uv run marimo edit notebooks/
```

Part 1부터 순서대로 실행하세요. 모든 노트북은 별도의 실행 버튼 없이 시작합니다.
모델·데이터 다운로드, 학습, 평가와 저장까지 진행합니다. 코드 셀의 의존성에
실행 순서를 명시해 학습 전후 비교와 모델 변경 순서를 유지했습니다.
셀 수정 시 자동 재실행은 끄고(`.marimo.toml`의 `lazy`), 설정을 바꿨다면
커널을 재시작해 처음부터 실행하세요. GPU 메모리 해제도 커널 재시작으로 합니다.

실행 결과와 체크포인트는 저장소 루트의 `outputs/vlm-from-scratch/<VLM_RUN_ID>/`에
저장합니다. 여러 파트에서 같은 실행 ID를 사용하면 Part 1의 `mini-vlm-flickr8k/`,
Part 4의 `mini-vlm-multitask/` 가중치를 후속 파트에서 찾아 재사용합니다.
재개할 때는 이전 `VLM_RUN_ID`를 지정하세요. 생략하면 앱 실행 시각으로 ID를 생성하므로
다른 앱과 자동으로 공유되지 않습니다. 실행 ID에는 영문, 숫자, 밑줄, 하이픈만 허용합니다.
원본에 있던 가중치 부재 시 초기 모델로 진행하는 동작은 유지했습니다.

2~9부는 체크포인트가 있으면 저장된 `config`의 모델 이름과 차원으로 구성 요소를
만든 뒤 가중치를 복원합니다. 저장된 토크나이저와 이미지 전처리 설정도 함께
사용하므로 1부의 ViT-Large/SmolLM-360M을 후속 파트에서 그대로 재사용할 수 있습니다.
체크포인트가 없을 때만 각 파트의 기본 모델을 사용합니다. 본문의 기본 모델
크기와 실제 VRAM 요구량은 불러온 체크포인트에 따라 달라질 수 있습니다.

GPU 선택은 실행 전에 `CUDA_VISIBLE_DEVICES=0 uv run marimo edit notebooks/`처럼
환경변수로 지정합니다. 원본의 GPU 1 고정 설정은 제거했습니다.
Part 12에서 Tesseract 비교가 필요하면 `uv sync --locked --extra ocr`로 설치하고,
별도로 시스템의 Tesseract 실행 파일과 언어 데이터를 준비하세요.

실습 소스는 marimo `.py`로 관리하며 원본 Jupyter 노트북은 제거했습니다.
Quarto 웹사이트의 실습 링크는 `.py` 소스를 내려받도록 연결되어 있습니다.
웹사이트 빌드는 실습 코드를 실행하지 않으며, 실습 실행은 위 marimo 명령을 사용하세요.
`requirements.txt`는 호환용 목록이고 재현 가능한 환경은 `pyproject.toml`과 `uv.lock`을 사용합니다.

그래프는 설치된 Noto Sans CJK, NanumGothic, 맑은 고딕, AppleGothic 중 한글 글꼴을 선택합니다.
그래프의 한글이 깨지면 운영체제에 해당 글꼴을 설치한 뒤 커널을 재시작하세요.

검증은 프로젝트 디렉토리에서 `uv run marimo check --strict notebooks/*.py`로 합니다.
실행 보고서는 저장소 루트 `reports/vlm-from-scratch/`에 기록합니다.

다운로드 없는 회귀 검사는 같은 디렉토리에서 실행합니다.

```bash
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 uv run python -m unittest discover -s tests -v
```

회귀 검사는 작은 무작위 초기화 ViT/Llama와 임시 체크포인트로 구성 복원·가중치
일치·JSON 응답 검증·OCR 전처리/학습/생성·INT8 용량 측정을 확인합니다.
실제 가중치의 GPU 학습·생성 품질 검증과는 구분합니다.
10부의 실측 용량은 양자화 packed 가중치를 포함한 `state_dict` 직렬화 크기(MiB)이며,
GPU 메모리 사용량을 뜻하지 않습니다.

## 원본 웹사이트

원본 프로젝트의 웹사이트: [https://nipunbatra.github.io/vlm-from-scratch/](https://nipunbatra.github.io/vlm-from-scratch/)

## 준비 사항

- Python 3.12(uv로 관리)
- PyTorch 2.12 계열(현재 uv 프로젝트 기준)
- 선택한 모델과 배치 크기에 충분한 VRAM을 갖춘 호환 GPU(파트별 요구량은 다름)
- 딥러닝 기초 지식

## 실습의 목표

코드를 직접 작성하고 각 부분의 동작을 설명합니다. 실습을 마치면 현대 VLM의 기본 동작 원리를 이해할 수 있습니다.

## 라이선스

MIT 라이선스입니다. 교육 목적으로 자유롭게 활용할 수 있습니다.
