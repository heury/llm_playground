# GPT 실습

Python 3.12와 uv를 사용하는 노트북 및 모델 실험 프로젝트입니다.
아래 명령은 `notebooks/gpt/` 디렉토리에서 실행합니다.

```bash
uv sync --locked
uv run marimo edit .
```

VS Code에서는 marimo 편집기와 `notebooks/gpt/.venv/bin/python` 환경을 사용합니다.

## marimo 실습

챕터 실습 9개를 `.py` marimo 앱으로 제공합니다.
기존 Jupyter 원본의 설명과 모델·학습 설정을 유지하며, `.ipynb` 파일은 제거했습니다.
`notebooks/gpt/`에서 실행하세요.

```bash
uv sync --locked
uv run marimo edit "Chapter_1_Exercise_Vector Space.py"
# 또는 디렉토리에서 실습 선택
uv run marimo edit .

# 전체 앱의 구문·셀 의존성 검사 (학습을 실행하지 않음)
uv run marimo check --strict Chapter*.py
```

| 장 | marimo 앱 |
| --- | --- |
| 1. 단어 벡터 | [Vector Space](Chapter_1_Exercise_Vector%20Space.py) |
| 2. 데이터셋·임베딩 | [Dataset](Chapter_2_Exercise_Dataset.py) |
| 3. Attention | [Attention](Chapter_3_Excercise_Attention.py) |
| 4. GPT 구조·생성 | [GPT](Chapter_4_Excercise_GPT.py) |
| 5. 사전 학습 | [Pretraining](Chapter_5_Excercise_Pretraining.py) |
| 6. 분류 미세 조정 | [Classification](Chapter_6_Excercise_Finetuning_Classification.py) |
| 6. LoRA | [Classification LoRA](Chapter_6_Excercise_Finetuning_Classification_LoRA.py) |
| 7. 지시 미세 조정 | [Follow Instructions](Chapter_7_Exercise_Follow_Instructions.py) |
| 7. DPO | [Follow Instructions DPO](Chapter_7_Exercise_Follow_Instructions_dpo.py) |

Jupyter의 `%pip` 대신 uv가 `marimo`와 `gensim`까지 설치합니다.
데이터·토크나이저·가중치는 첫 실행에 인터넷이 필요할 수 있습니다.
실행별 데이터, CSV, 그래프, 학습 체크포인트와 생성 결과는
`outputs/gpt/<한국 시각>-<실습명>-<고유 ID>/`에 저장하고,
다운로드한 GPT-2 가중치는 기존 `notebooks/gpt/models/gpt2/` 경로를 재사용합니다.
Gensim·Hugging Face·tiktoken의 다운로드 캐시는 각 라이브러리 기본 경로를 사용합니다.

5~7장은 원본의 전체 모델과 학습 횟수를 유지하므로 충분한 메모리와 GPU가 필요할 수 있습니다.
marimo 설정의 Runtime에서 **On cell change → lazy**를 선택하면
셀을 수정할 때 이어지는 학습을 수동으로 실행할 수 있습니다.
`uv run python <앱.py>`는 전체 셀을 실행하므로 실제 학습을 진행할 때 사용하세요.
이번 변환의 검증 범위와 실행 기록은
[marimo 변환 검증 보고서](../../reports/gpt/20261109-203900-marimo.md)를 참고하세요.

## 모델 구조 개선 marimo 실습

기존 `model_improvements/`의 스크립트 9개는 아래 marimo 앱으로 변환해 이
디렉토리에 통합했습니다. 모델 클래스와 생성 함수는 유지하고, CLI 옵션을
화면의 설정 입력으로 바꿨습니다. CodeTour도 새 파일과 줄 번호를 가리킵니다.

| 실습 | marimo 앱 |
| --- | --- |
| KV cache | [gpt_with_kv_cache.py](gpt_with_kv_cache.py) |
| Multi-Head Attention | [gpt_with_kv_mha.py](gpt_with_kv_mha.py) |
| Grouped-Query Attention | [gpt_with_kv_gqa.py](gpt_with_kv_gqa.py) |
| Multi-Head Latent Attention | [gpt_with_kv_mla.py](gpt_with_kv_mla.py) |
| DeepSeek Sparse Attention 실험 | [gpt_with_kv_dsa.py](gpt_with_kv_dsa.py) |
| Sliding Window Attention | [gpt_with_kv_swa.py](gpt_with_kv_swa.py) |
| 레이어 간 KV 공유 | [gpt_with_kv_sharing.py](gpt_with_kv_sharing.py) |
| Dense FFN / MoE | [gpt_with_kv_moe.py](gpt_with_kv_moe.py) |
| Gated MHA / Gated DeltaNet | [gpt_with_deltanet.py](gpt_with_deltanet.py) |

```bash
uv run --locked marimo edit gpt_with_kv_gqa.py
uv run --locked marimo check --strict gpt_with_*.py
```

설정을 확인하고 **실험 실행**을 누르면 실행합니다. 버튼을 누르기 전에는
모델 생성과 토크나이저 다운로드를 시작하지 않습니다. GPT 실습은 기존 스크립트의
기본값(임베딩 768, 12층, 생성 200토큰)을 유지합니다. 작은 실험은 임베딩 32,
4헤드, 2층, 생성 8토큰 등으로 줄이고 GQA의 KV 그룹 및 KV 공유 레이어 수도 맞추세요.
모델은 무작위 가중치로 초기화하며 사전 학습 가중치를 내려받지 않습니다.
DeltaNet 앱은 작은 합성 입력으로 두 모듈의 순전파·역전파를 확인합니다.

실행별 설정·결과는 `outputs/gpt/<실행 ID>/result.json`, 콘솔 출력은
같은 폴더의 `run.log`에 저장합니다. DSA는 dense attention에 선택 마스크를
적용하는 교육용 구현이므로 sparse kernel의 속도 향상을 측정하는 용도는 아닙니다.
변환본의 정적 검사와 작은 CPU 모델 검증은
[모델 구조 개선 변환 보고서](../../reports/gpt/20261109-204756-model-improvements-marimo.md)에 기록했습니다.

노트북에서 사용하는 `previous_chapters.py`는
[heury/llm_hands_on 원본](https://github.com/heury/llm_hands_on/blob/main/previous_chapters.py)에서 가져왔습니다.
원본의 저작권 및 라이선스 안내를 파일 상단에 유지합니다.
데이터와 모델 가중치도 실습에 따라 별도로 내려받아야 합니다.

검증 보고서는 저장소 루트의 [reports/gpt/](../../reports/gpt/),
현재 보관 중인 실행 산출물은 [outputs/gpt/](../../outputs/gpt/)에서 확인합니다.
기존 안내의 `test_scenarios/gpt/README.md`와 `validation/README.md`는 현재 체크아웃에 없습니다.
