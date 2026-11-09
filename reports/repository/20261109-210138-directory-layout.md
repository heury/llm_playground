# 디렉토리 구조 변경 검증

- 실행일: 2026-11-09 (Asia/Seoul)
- 작업 디렉토리: `/home/mind4u/dev/llm_playground`
- 산출물: `outputs/repository/20261109-210138-directory-layout/`
- 환경: Linux, Python 3.12.3, 캐시에서 읽은 marimo 0.25.2.
- 모델·데이터 리비전: 해당 없음. 모델·데이터 다운로드와 학습은 실행하지 않음.

## 적용 내용

- `gpt/` → `notebooks/gpt/`
- 기존 `notebooks/` → `notebooks/minimind/`
- `unsloth/` → `notebooks/unsloth/`
- MiniMind 계열 8개 프로젝트 → `projects/<기존 이름>/`
- `pdf/` → `references/slides/`
- 프로젝트 내부의 과거 실행 폴더 5개 → 루트 `outputs/minimind/`, `outputs/minimind-o/` (대상 이름 충돌 없음).
- 각 프로젝트 내부 구성, 개별 의존성 환경과 잠금 파일 유지.
- 노트북의 모델 import·소스 해시·산출물 경로, 로컬 클라우드 컨트롤러의 저장소 루트 경로, README 링크, CodeTour, AGENTS 안내 갱신.
- 기존 작업의 수정·추가·삭제 상태를 보존하며 이동. Git stage·commit은 수행하지 않음.

## 검증 결과

| 검사 | 결과 |
| --- | --- |
| 이동 전 실제 존재했던 파일 | 650개 모두 새 위치에 존재 |
| Python AST 구문 검사 | 282개 통과 |
| 노트북 JSON 구조 | 18개 통과 |
| pyproject.toml 및 존재하는 uv.lock의 TOML 구문 | 프로젝트 11개 통과 |
| 잠금 파일 내용 비교 | 10개 모두 HEAD와 동일 |
| marimo 기본 정적 검사 규칙 | 노트북 40개, 진단 0개 |
| 주요 README 상대 링크 | 76개 유효 |
| CodeTour 파일 대상 | 38개 유효 |
| 노트북의 정적 모델 경로 표현식 | 15개 유효 |
| git diff --check | 통과 |

## 실행 명령과 재확인

저장소 루트에서 실행:

```bash
python3 outputs/repository/20261109-210138-directory-layout/verify_layout.py
XDG_CONFIG_HOME=/tmp/llm-reorg-config python3 outputs/repository/20261109-210138-directory-layout/check_marimo_static.py
git diff --check
```

`verification.json`, `marimo-static.json`, `locks.json`, `diff-check.log`에 결과를 기록했다.
`original-files.json`은 이동 전 파일 목록이며 파일 내용이나 인증 정보는 포함하지 않는다.

## 검증 한계와 미실행 항목

프로젝트 전체 환경 설치, 모델 순전파·역전파, 전체 단위 테스트, GPU 학습 및 클라우드 작업은 실행하지 않았다.
변경은 디렉토리와 경로 참조에 한정하며 모델 연산은 변경하지 않았다.
기존 가상 환경은 이 작업 환경에 없으며 설치하지 않았다. 각 프로젝트에서 `uv sync --locked`로 준비한다.
`minimind-j2`에는 이동 전부터 `uv.lock`이 없으므로 해당 프로젝트는 잠금 파일 비교 대상이 아니다.

일반 `marimo check --strict` CLI는 이 실행 환경의 비동기 파일 탐색에서 대기해 중단했다.
대신 캐시의 동일한 marimo 버전에서 파일을 직접 읽고 기본 정적 검사 엔진을 호출해 40개를 검사했다.
이는 학습 셀이나 모델 코드를 실행하지 않으며, 정식 프로젝트 환경 설치에 대한 검증은 아니다.
캐시에서 모듈을 읽은 재현 스크립트는 이 환경의 캐시 경로를 사용한다.

과거 보고서의 실행 명령과 기록은 당시 경로를 포함할 수 있다. 현재 실행 경로는 루트 README와 각 프로젝트 README를 따른다.
원래 GPT README에 있던 현재 체크아웃에 없는 검증 문서 링크는 실제 `reports/gpt/`, `outputs/gpt/` 경로로 교체했다.
