# VLM from Scratch 한국어 번역 검증

- 실행 ID: `20261111-130859-korean-translation` (Asia/Seoul)
- 대상: `notebooks/vlm-from-scratch/notebooks/part-*.py` 12개와 프로젝트 README.
- 산출물: [outputs/vlm-from-scratch/20261111-130859-korean-translation/](../../outputs/vlm-from-scratch/20261111-130859-korean-translation/)

## 번역 범위

12개 앱의 제목·본문·표·도식·참고 자료 설명을 한국어로 옮겼다. 비어 있지 않은 Markdown 셀은 모두 한국어를 포함하며 `mo.md` 호출은 총 197개다. 코드의 고유 영문 설명 주석 605종, 함수·클래스 설명 125종(등장 횟수 149개)을 번역했다. 텐서 shape 표기와 주석 처리된 코드는 유지했다. 학습 로그, 진행 표시, 그래프 제목·축·범례와 비교표도 번역했다.

모델·데이터셋 식별자, 다운로드 URL, 코드 식별자, 파일 경로, JSON 키, 실제 학습·추론에 전달되는 프롬프트와 정답 문자열, 특수 토큰은 유지했다. 본문의 예시는 실제 영어 형식을 보여 주거나 한국어 뜻을 함께 제시한다. 원문에 포함된 목표 수치와 과거 성능 설명을 이번 작업에서 측정한 결과로 보고하지 않았다.

그래프의 한글 표시를 위해 pyplot을 사용하는 11개 앱에 설치된 한글 글꼴 선택을 추가했다. Noto Sans CJK KR/JP, NanumGothic, Malgun Gothic, AppleGothic 순서로 확인하고 유니코드 마이너스를 비활성화한다. 외부 글꼴 다운로드는 수행하지 않는다. README에 번역 범위와 글꼴 준비 방법을 추가했다.

## 코드 보존 검사

산출물 `before/`에 저장한 작업 시작 시점의 소스와 현재 파일을 Python AST로 비교했다. 문서 문자열, 화면 출력 문자열, 비교표, 그래프 표시 문자열, 새 글꼴 설정만 정규화하고 나머지 구조를 비교했다. 모델 생성·순전파·역전파·최적화·마스킹·토큰화·생성 설정·체크포인트 처리가 12개 모두 동일했다. 프롬프트 등 모델에 전달되는 문자열은 일반 문자열로 비교 대상에 남겼다.

URL과 로컬 링크의 대상 목록도 번역 전후가 동일한지 확인했고 로컬 링크의 실제 파일이 존재하는지 검사했다. 실제 모델·데이터를 다운로드하지 않았으므로 외부 리비전 검증은 해당하지 않는다. 원본 데이터·모델 이름과 설정을 유지했다.

## 실행 환경과 명령

Linux / CPython 3.12.15 / marimo 0.25.2에서 검사했다. 기존 `notebooks/minimind/.venv/bin/python`과 `marimo`를 검사 도구로 사용했으며 이 프로젝트의 GPU 환경을 새로 설치하지 않았다. 한글 그래프 검사는 matplotlib 3.11.2와 `Noto Sans CJK KR`를 사용했다.

작업 디렉토리: 저장소 루트.

```bash
notebooks/minimind/.venv/bin/marimo check --strict notebooks/vlm-from-scratch/notebooks/*.py
notebooks/minimind/.venv/bin/python outputs/vlm-from-scratch/20261111-130859-korean-translation/vlm_check_translation_ast.py
notebooks/minimind/.venv/bin/python outputs/vlm-from-scratch/20261111-130859-korean-translation/vlm_validate_ko.py
git diff --check
git -C notebooks/vlm-from-scratch diff --check
```

검증 스크립트는 이번 세션의 `/tmp/vlm-ko-run.txt`에 저장한 산출물 경로를 읽는다. 재실행 시 해당 파일에 이 산출물 디렉토리의 절대 경로를 지정해야 한다. `vlm_validate_ko.py`는 각 앱을 import해 그래프를 검사한 뒤 프로젝트 디렉토리에서 실행 버튼을 누르지 않고 `marimo export html`을 수행한다. 생성된 `ancestor-stopped`는 버튼의 `mo.stop`에 따른 정상 대기 상태로 구분했다.

## 결과

| 앱 | Markdown 셀 | 순차 실행 코드 셀 | 그래프·구문·코드 보존 | 실행 전 HTML |
| --- | ---: | ---: | --- | --- |
| `part-01-minimal-vlm.py` | 19 | 25 | 통과 | 통과 |
| `part-02-object-detection.py` | 20 | 20 | 통과 | 통과 |
| `part-03-visual-qa.py` | 18 | 17 | 통과 | 통과 |
| `part-04-multi-task.py` | 16 | 17 | 통과 | 통과 |
| `part-05-multi-image.py` | 16 | 17 | 통과 | 통과 |
| `part-06-task-routing.py` | 15 | 19 | 통과 | 통과 |
| `part-07-chain-of-thought.py` | 16 | 20 | 통과 | 통과 |
| `part-08-referring-segmentation.py` | 15 | 17 | 통과 | 통과 |
| `part-09-image-editing.py` | 16 | 17 | 통과 | 통과 |
| `part-10-compression.py` | 17 | 14 | 통과 | 통과 |
| `part-11-image-generation.py` | 15 | 14 | 통과 | 통과 |
| `part-12-ocr-latex.py` | 14 | 13 | 통과 | 통과 |

- `marimo check --strict`: 종료 코드 0.
- 순환·중복 정의·미정의 참조: 없음. 순차 실행 토큰 연결 유지.
- 실행 전 HTML 내보내기: 12개 모두 종료 코드 0, 예상하지 않은 세션 오류 없음.
- 계산 코드·모델 입력 보존: 12개 모두 통과(`ast-preservation.json`).
- 한글 그래프 이미지 저장: 통과, 글리프 누락 경고 0개(`korean-font-smoke.png`).
- 링크 대상 보존 및 로컬 파일 존재: 통과.
- 외부·내부 저장소 `git diff --check`: 통과.

## 한계와 미실행 항목

GPU 학습·추론, 모델·데이터 다운로드, 모델 출력의 한국어화, 외부 문헌의 성능 주장 재검증, 브라우저의 직접 조작은 수행하지 않았다. 모델 입력과 데이터는 영어를 유지하므로 실제 생성 결과 역시 영어일 수 있다. 화면 검증은 실행 전 상태와 별도의 한글 그래프 렌더링을 대상으로 한다. 실제 학습 품질이나 GPU 메모리 사용량을 보장하지 않는다. 시스템에 한글 글꼴이 없는 다른 환경에서는 README에 따라 글꼴을 설치해야 한다.
