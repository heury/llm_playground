# VLM from Scratch marimo 변환 검증

- 실행 ID: `20261111-105050-marimo-conversion` (Asia/Seoul)
- 작업 대상: `notebooks/vlm-from-scratch/notebooks/part-*.ipynb` → 같은 이름의 `.py` 12개.
- 원본 리비전: 내부 저장소 `93475a3247b8c8ae7e4f73b73fd2c3e8c48ba5ce`. 원본 파일 SHA-256은 산출물의 `manifest.json` 참조.
- 산출물: [outputs/vlm-from-scratch/20261111-105050-marimo-conversion/](../../outputs/vlm-from-scratch/20261111-105050-marimo-conversion/)

## 변경 내용

원본 설명·모델 구조·학습·평가 코드를 marimo 셀로 변환했다. marimo 변환기가 중복 셀 변수에 지역 이름을 부여하고, 코드 셀마다 `execution_step_N` 의존성을 연결해 모델과 리스트 변경, 학습 전후 비교, pyplot 상태의 원본 실행 순서를 보존했다. 명시적인 실행 버튼을 누르기 전에는 다운로드와 학습 코드가 실행되지 않는다.

설치 셀은 `uv sync --locked` 안내로 바꾸고 IPython `%config`를 제거했다. Part 7의 중복 초기화 셀과 GPU 1 강제 설정을 제거했다. GPU 선택은 실행 전 환경변수로 한다. `globals()`/다른 셀의 변수를 `del`하는 Jupyter 정리 코드는 marimo 커널 재시작 안내로 대체했다. Part 5 첫머리에 있던 초기화 전 삭제 코드도 같은 방식으로 처리했다. 원본 raw frontmatter의 제목·저자·날짜는 Markdown으로 보존했다.

모델 및 체크포인트 경로는 저장소 루트 `outputs/vlm-from-scratch/<VLM_RUN_ID>/`로 옮겼다. 동일 실행 ID의 Part 1/4 결과를 후속 파트에서 읽을 수 있다. 실행 ID를 지정하지 않으면 각 앱 시작 시 시각 기반으로 생성한다. 원본에서 가중치가 없을 때 초기 모델로 진행하는 경로는 유지했다.

독립 Python 3.12용 `pyproject.toml`, `uv.lock`, `.python-version`을 추가했다. marimo 설정은 셀 수정 시 수동 실행(`lazy`)이다. README의 실습 링크와 실행 방법, 앱 내부의 60개 로컬 링크를 확인했다. 기존 Quarto 출판 경로를 유지하기 위해 `.ipynb` 원본은 수정·삭제하지 않았다. `.py`와 `.ipynb`는 자동 동기화되지 않는다.

## 환경과 명령

검사 환경은 Linux, CPython 3.12.15, marimo 0.25.2다. 이미 설치된 `notebooks/minimind/.venv/bin/python` 및 `marimo`를 검사 도구로 사용했다. 이 프로젝트의 전체 GPU 의존성 환경은 설치하지 않았다. uv는 110개 패키지를 잠갔으며 `uv.lock`에 정확한 버전이 기록되어 있다.

저장소 루트에서 실행:

```bash
notebooks/minimind/.venv/bin/marimo check --strict notebooks/vlm-from-scratch/notebooks/*.py
UV_CACHE_DIR=/tmp/vlm-uv-cache uv lock --check --project notebooks/vlm-from-scratch
notebooks/minimind/.venv/bin/python outputs/vlm-from-scratch/20261111-105050-marimo-conversion/validate_marimo.py
git diff --check
git -C notebooks/vlm-from-scratch diff --check
```

`validate_marimo.py`는 Python AST 구문 검사, marimo 앱 import 및 의존성 그래프 초기화, 순환·중복 정의·미정의 참조 검사, 순차 실행 토큰 연결을 검사한다. 각 앱은 다음 명령과 동일한 방식으로 프로젝트 디렉토리에서 HTML을 내보냈다:

```bash
# 작업 디렉토리: notebooks/vlm-from-scratch/
../minimind/.venv/bin/marimo export html notebooks/part-01-minimal-vlm.py \
  -o ../../outputs/vlm-from-scratch/20261111-105050-marimo-conversion/part-01-minimal-vlm.html -f
```

## 결과

| 앱 | 전체 셀 | 순차 코드 셀 | 구문·그래프 | 실행 전 HTML |
| --- | ---: | ---: | --- | --- |
| `part-01-minimal-vlm.py` | 45 | 25 | 통과 | 통과 |
| `part-02-object-detection.py` | 41 | 20 | 통과 | 통과 |
| `part-03-visual-qa.py` | 36 | 17 | 통과 | 통과 |
| `part-04-multi-task.py` | 34 | 17 | 통과 | 통과 |
| `part-05-multi-image.py` | 34 | 17 | 통과 | 통과 |
| `part-06-task-routing.py` | 35 | 19 | 통과 | 통과 |
| `part-07-chain-of-thought.py` | 37 | 20 | 통과 | 통과 |
| `part-08-referring-segmentation.py` | 33 | 17 | 통과 | 통과 |
| `part-09-image-editing.py` | 34 | 17 | 통과 | 통과 |
| `part-10-compression.py` | 32 | 14 | 통과 | 통과 |
| `part-11-image-generation.py` | 30 | 14 | 통과 | 통과 |
| `part-12-ocr-latex.py` | 28 | 13 | 통과 | 통과 |

- `marimo check --strict`: 종료 코드 0.
- 미정의 참조, 순환, 중복 정의: 12개 모두 없음.
- 모든 순차 코드 셀의 직전 토큰 참조 및 연속 번호: 통과.
- HTML 내보내기: 12개 모두 종료 코드 0. 세션 JSON의 예상하지 않은 오류도 0개다. 실행 버튼 앞의 `mo.stop`에 따른 `ancestor-stopped`는 예상된 대기 상태로 구분했다.
- 로컬 Markdown 링크 60개: 대상 존재 확인.
- `uv lock --check`: 종료 코드 0, 110 패키지.
- 외부/내부 저장소 `git diff --check`: 통과.

## 미실행 및 한계

GPU 학습·추론, 모델·데이터 다운로드, 전체 의존성 설치, 체크포인트 재개와 파트 간 실제 가중치 로딩은 실행하지 않았다. 모델·데이터 리비전은 원본 코드의 저장소 이름을 유지했으며 외부 파일을 가져와 검증하지 않았다. 따라서 이 보고서는 변환 형식, 의존성 그래프 및 실행 전 상태의 렌더링을 검증하며 학습 수렴·생성 품질·GPU 메모리 요구량을 보장하지 않는다. 브라우저에서의 직접 UI 조작과 기존 Quarto 웹사이트 빌드도 수행하지 않았다.
