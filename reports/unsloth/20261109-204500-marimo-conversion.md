# Unsloth 노트북 marimo 변환 검증

2026-11-09, Asia/Seoul. `unsloth/`의 Jupyter 노트북 9개를 같은 이름의 marimo
Python 파일로 변환했다. 원본 `.ipynb`는 보존했다.

## 변경 내용

- 원본 코드 셀 173개와 Markdown 설명을 보존하고 marimo 변환기가 중복 변수를 별도 이름으로 변환했다.
- 모든 원본 코드 셀에 `execution_step_*` 의존성을 추가했다. Unsloth 패치·환경 변수 설정, 모델 변경, 학습, 추론, 저장 및 외부 변환 명령의 순서를 유지한다.
- 실행 버튼과 `mo.stop`으로 모델·데이터 다운로드 및 학습 시작을 제어한다. `.marimo.toml`은 초기 화면을 표시하고 버튼 클릭 후 의존 셀을 실행하도록 설정했다.
- 실행별 출력 폴더를 `outputs/unsloth/<날짜-시간-임의 ID-실습명>/`으로 지정했다. TRL 체크포인트, adapter, 병합·양자화 출력 및 Phone의 ExecuTorch 변환 파일을 이 경로로 변경했다. 다운로드 캐시는 기존 설정을 유지한다.
- DPO의 출력 경로 초기화를 공통 초기화 셀로 대체했다. Phone의 `output_name`은 파일명으로 유지하고 `output_dir`에 실행 폴더를 지정했다.
- GRPO 보상 함수의 전역 출력 횟수를 셀 내부 상태로 바꿨다. 이전 SFT 데이터셋은 marimo가 셀 변수를 보존하므로 삭제하지 않는다. 따라서 Jupyter 원본보다 CPU 메모리를 더 유지할 수 있다.
- marimo 의존성과 `uv.lock`을 갱신했다. README에 파일 목록·extra별 실행 명령·검증 범위를 추가하고 기존 A40 보고서의 끊어진 링크를 수정했다.

## 환경과 명령

작업 디렉토리는 `/home/mind4u/dev/llm_playground`이다.
검사에는 기존 `notebooks/.venv/bin/python`의 Python 3.12.15와 marimo 0.25.2를 사용했다.
Unsloth 학습 환경을 재설치하거나 GPU 리소스를 생성하지 않았다.

```bash
notebooks/.venv/bin/marimo check --strict unsloth/*.py
MPLCONFIGDIR=/tmp/unsloth-marimo-mpl notebooks/.venv/bin/python \
  outputs/unsloth/20261109-204500-marimo-conversion/validate.py
git diff --check
```

잠금 파일 확인은 `unsloth/`에서 실행했다.

```bash
UV_CACHE_DIR=/tmp/unsloth-uv-cache uv lock --offline --check
```

## 결과와 한계

| 확인 항목 | 결과 |
| --- | --- |
| Python 구문·컴파일 | 9/9 통과 |
| `marimo check --strict` | 종료 코드 0 |
| 원본 Markdown 보존 | 9/9 통과 |
| 의존 그래프 초기화·미정의 참조 검사 | 9/9 통과 |
| 코드 셀 순서 의존성 | 173/173 통과 |
| 실행 버튼 미클릭 상태의 `app.run()` | 9/9 통과 |
| 초기 실행의 모델·학습 패키지 로딩 방지 | 9/9 통과 |
| README 상대 링크 | 통과; 기존 보고서 링크 수정 후 재검사 |
| uv 잠금 파일 일치 | 328개 패키지, 통과 |
| `git diff --check` | 통과 |

초기 실행은 `mo`와 `run_training`만 정의하고, 출력 디렉토리 생성과 모든 GPU 코드가
대기하는 것을 확인했다. 브라우저 UI 클릭, 실제 모델 로딩, GPU 학습·추론,
ExecuTorch 내보내기 및 모바일 실행은 수행하지 않았다. 모델·데이터를 다운로드하지
않았으므로 현재 리비전은 확인하지 않았다. 모델 ID, 데이터셋 ID와 학습 설정은
원본을 따르며, 과거 Jupyter A40 결과를 marimo 실행 결과로 간주하지 않는다.

산출물: [outputs/unsloth/20261109-204500-marimo-conversion/](../../outputs/unsloth/20261109-204500-marimo-conversion).
`validation.json`과 로그, 검증 스크립트, `manifest.json`의 소스·변환본·잠금 파일 SHA256을 보관했다.
`outputs/`는 Git 제외 경로다.
