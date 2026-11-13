# Unsloth 실습

Python 3.12 / Linux x86_64 환경에서 Unsloth 노트북을 실행하는 uv 프로젝트입니다.
학습 의존성을 `pyproject.toml`에서 관리하며, 노트북 원본의 설치 버전 고정은 제거했습니다.
각 노트북은 Installation 셀과 그 위의 소개·뉴스를 제거하고 실습 본문부터 시작합니다.
패키지 설치는 아래 uv 명령으로 터미널에서 진행합니다.

## 설치 및 실행

아래 명령은 이 `notebooks/unsloth/` 디렉토리에서 실행합니다.

```bash
# A40에서 검증한 잠금 파일의 패키지 조합으로 설치
uv sync --locked
uv run marimo edit .
```

VS Code에서는 marimo 편집기와 `.venv/bin/python` 환경을 사용합니다.

## marimo 노트북

실습 10개를 marimo `.py` 파일로 제공합니다. `.ipynb` 원본은 제거했습니다.
설명·코드와 원래 학습 설정을 유지하고, 코드 셀 사이에 `execution_step_*` 의존성을
추가해 모델 변경 → 데이터 준비 → 학습 → 추론·저장 순서를 보존했습니다.
`scripts/`의 기존 Jupyter 실행 유틸리티는 별도로 준비한 외부 `.ipynb` 입력에만 사용합니다.
Pod 실행 시 `UNSLOTH_VALIDATION_SOURCE`에 외부 원본과 환경 파일이 있는 디렉토리를 지정하세요.

아래 명령도 `notebooks/unsloth/`에서 실행합니다. 파일명의 괄호 때문에 경로를 따옴표로 감쌉니다.

```bash
uv sync --locked
uv run --locked marimo edit 'Qwen3_5_(4B)-Decision.py'
```

| 실습 | marimo 파일 | 추가 옵션 |
| --- | --- | --- |
| Decision | [Qwen3_5_(4B)-Decision.py](Qwen3_5_%284B%29-Decision.py) | 기본 |
| Decision 내부 구조 | [Qwen3_5_(4B)-Decision_Internals.py](Qwen3_5_%284B%29-Decision_Internals.py) | 기본 (1~8부 CPU) |
| Vision | [Qwen3_5_(4B)_Vision.py](Qwen3_5_%284B%29_Vision.py) | 기본 |
| Vision GRPO | [Qwen3_5_(4B)_Vision_GRPO.py](Qwen3_5_%284B%29_Vision_GRPO.py) | 기본 |
| Knowledge Distillation | [Qwen3_(0.6B)-Knowledge_Distillation.py](Qwen3_%280.6B%29-Knowledge_Distillation.py) | 기본 |
| DPO | [Qwen3_5_(4B)-DPO.py](Qwen3_5_%284B%29-DPO.py) | 기본 |
| GRPO (vLLM) | [Qwen3_(4B)-GRPO.py](Qwen3_%284B%29-GRPO.py) | `--extra grpo` |
| QAT | [Qwen3_(4B)_Instruct-QAT.py](Qwen3_%284B%29_Instruct-QAT.py) | `--extra qat` |
| Phone deployment | [Qwen3_(0_6B)-Phone_Deployment.py](Qwen3_%280_6B%29-Phone_Deployment.py) | `--extra phone` |
| Tool calling | [Qwen3_(4B)-Tool_Calling.py](Qwen3_%284B%29-Tool_Calling.py) | `--extra tools` |

예를 들어 GRPO 실습은 다음과 같이 실행합니다.

```bash
uv sync --locked --extra grpo
uv run --locked --extra grpo marimo edit 'Qwen3_(4B)-GRPO.py'
```

프로젝트의 `.marimo.toml`은 설명과 실행 버튼을 처음 열 때 표시하도록 설정합니다.
**GPU 실습 실행** 버튼을 누르면 모델·데이터 다운로드부터 학습·추론·저장까지
원본 순서로 진행합니다. 버튼을 누르기 전에는 GPU 코드가 실행되지 않습니다.
설정을 수정한 뒤에는 커널을 재시작하고 처음부터 실행하세요. 실행 중 셀 편집은
해당 셀과 이후 학습을 다시 실행할 수 있습니다.

각 실행은 날짜·시간·임의 ID가 포함된 `../../outputs/unsloth/<실행 ID>/`를 만듭니다.
체크포인트, LoRA 및 선택적 내보내기 파일과 Phone의 ExecuTorch 중간 파일도
해당 실행 폴더에 저장합니다. Hugging Face의 모델·데이터 다운로드 캐시는 기존 설정을 사용합니다.
marimo가 이전 셀의 변수를 유지하므로 GRPO의 SFT 데이터셋은 이후에도 남아 있습니다.

변환본의 검증 범위는 Python 구문, marimo 정적 검사, 셀 의존성 및 버튼을 누르지 않은
초기 실행입니다. 변환본으로 GPU 학습·추론·내보내기는 다시 실행하지 않았으며,
아래 A40 검증 기록은 원본 `.ipynb`에 대한 결과입니다.
변환 방식은 [marimo의 Jupyter 마이그레이션 안내](https://docs.marimo.io/guides/coming_from/jupyter/)를 참고했습니다.

## 판단 모델 내부 구조 실습

[`Qwen3_5_(4B)-Decision_Internals.py`](Qwen3_5_%284B%29-Decision_Internals.py)는 Decision 실습의
API 아래를 단계별로 살펴보는 교육용 노트북입니다. Unsloth의 `decision_from_lm.py`, `decision.py`,
`decision_datasets.py`와 포함된 Cloudflare Clef 구현을 참고했습니다.
설치된 `unsloth` 패키지를 초기화하지 않고 필요한 원본 파일만 경로로 불러오므로 1~8부는 GPU 없이 실행됩니다.
데이터 형식, 스키마 증강, 프롬프트 span 인코딩, `JointSchemaHead` 순전파, 손실 함수, 작은 무작위 백본의
2단계 CPU 학습(헤드만 → 백본 포함), 온도 보정과 추론을 다룹니다. 9부는 GPU에서 같은 과정을
`FastDecisionModel`·`DecisionTrainer`로 실행하며 기본 백본은 `unsloth/Qwen3.5-0.8B`입니다.
검증 범위는 [검증 보고서](../../reports/unsloth/20261113-151500-decision-internals.md)에 기록했습니다.

## GPU 환경과 추가 옵션

잠금 파일은 CUDA 13.0용 패키지를 사용하므로, GPU 실행에는 이를 지원하는 NVIDIA
드라이버가 필요합니다. CPU만 있는 환경에서 학습 노트북을 실행할 수는 없습니다.
일반 Decision, Vision, Vision GRPO, Knowledge Distillation 실습은 기본 환경을 사용합니다.
[`Qwen3_5_(4B)-DPO.py`](Qwen3_5_%284B%29-DPO.py)도 기본 환경을 사용합니다.
Zephyr DPO 원본을 `unsloth/Qwen3.5-4B`와 Qwen 채팅 템플릿, 현재 TRL API에 맞춘
사본이며, 형식·구문·채팅 경계만 확인했고 GPU 학습은 아직 실행하지 않았습니다.
학습 결과와 LoRA adapter는 `../../outputs/unsloth/<실행 ID>/`에 저장합니다.
다음 실습은 추가 패키지를 선택하세요.

| 실습 | 검증한 환경 설치 | 실행 |
| --- | --- | --- |
| Qwen3 GRPO (vLLM) | `uv sync --locked --extra grpo` | `uv run --locked --extra grpo marimo edit "Qwen3_(4B)-GRPO.py"` |
| QAT | `uv sync --locked --extra qat` | `uv run --locked --extra qat marimo edit "Qwen3_(4B)_Instruct-QAT.py"` |
| Phone deployment | `uv sync --locked --extra phone` | `uv run --locked --extra phone marimo edit "Qwen3_(0_6B)-Phone_Deployment.py"` |
| Tool calling | `uv sync --locked --extra tools` | `uv run --locked --extra tools marimo edit "Qwen3_(4B)-Tool_Calling.py"` |

`grpo`, `qat`, `phone`은 환경을 전환하며 하나씩 선택합니다. `--all-extras`는 사용하지 않습니다.
`tools`는 다른 옵션과 함께 선택할 수 있습니다.
환경을 바꾸기 전에는 실행 중인 커널을 종료하고, 설치 후 커널을 다시 시작하세요.
동시에 여러 환경이 필요하면 별도의 가상환경을 사용합니다.
테스트 실행 결과는 실습 소스에 덮어쓰지 않고 별도 검증 기록에 보관합니다.

## 최신 버전과 재현성

`uv sync --upgrade`는 새로운 버전을 다시 조회합니다.
`uv sync --locked`는 저장된 `uv.lock` 조합을 재현하며, 추가 옵션도 동일하게 지정합니다.
패키지 상호 의존성은 유지합니다. `--no-deps`나 의존성 강제 덮어쓰기는 사용하지 않습니다.

2026-10-27 A40 검증 환경:

- Unsloth / Unsloth Zoo: 2026.10.9
- 기본 환경: Transformers 5.17.0, TRL 1.15.0, PyTorch 2.14.1
- GRPO: vLLM 0.32.0이 요구하는 PyTorch 2.13.0 사용
- QAT: MSLK 1.3.0의 바이너리 호환성에 맞춰 PyTorch 2.13.0 사용
- Phone: 노트북에서 사용하는 ExecuTorch를 직접 설치하며 Transformers 5.17.0 사용

PyPI에는 더 최신 Transformers / TRL도 있지만 Unsloth의 상한 제약이 있어
동시에 선택할 수 없습니다. Phone에서 사용하지 않는 `optimum-executorch`는 제거했습니다.
이 패키지가 요구하는 Transformers 5.0.0rc1은 최신 TorchAO의 저장 API와 충돌했습니다.
선택된 버전은 `uv.lock`에서 확인할 수 있습니다.
`xformers`와 QAT의 `mslk`는 PyTorch 공식 CUDA 13 인덱스에서 설치합니다.
PyPI의 CUDA 12.8용 `xformers`를 설치한 기존 환경은 같은 버전 번호 때문에
일반 동기화만으로 교체되지 않을 수 있으므로, 한 번 다음과 같이 재설치하세요.

```bash
uv sync --locked --reinstall-package xformers
# 추가 옵션을 사용하는 환경에서는 위 명령에도 --extra qat 등을 지정합니다.
```

증류 노트북의 GKD 클래스는 최신 TRL의 `trl.experimental.gkd` 경로를 사용합니다.
`uv sync --upgrade`로 패키지 조합을 바꾸면 GPU 실행을 다시 검증해야 합니다.

## 검증 범위

2026-10-27에 로컬 Python 3.12 환경을 `uv sync --locked`로 설치했습니다.
노트북 8개의 형식 검사와 전체 코드 셀 164개의 Python/IPython 구문 검사를 통과했습니다.
각 노트북을 별도의 Runpod A40 Pod와 uv 환경에서 실행해 전체 164개 코드 셀의 실행을
완료했습니다. 원본 학습 스텝을 유지했고, Phone은 ExecuTorch 내보내기 후 실제 CPU에서
2토큰 순전파와 KV 캐시도 확인했습니다. 테스트 Pod 8개는 모두 중단했습니다.
노트북별 실행 조건, 패키지·코드 수정, 실행 증거와 검증 범위는
[A40 통합 실행 결과](../../reports/unsloth/20261027-a40-validation.md)에 기록했습니다.

출처 및 패키지 관리 안내:

- [원본 Unsloth 노트북](https://github.com/unslothai/notebooks/tree/main/nb)
- [Unsloth 설치 안내](https://unsloth.ai/docs/get-started/install/pip-install)
- [uv 잠금 파일 및 동기화](https://docs.astral.sh/uv/concepts/projects/sync/)
