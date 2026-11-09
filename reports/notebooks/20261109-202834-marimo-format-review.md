# marimo 두 번째 셀의 Markdown 분류 검토

- 대상: `notebooks/`의 Python 노트북 13개.
- 산출물: `outputs/notebooks/20261109-202834-marimo-format-review/`.
- 환경: `notebooks/.venv`, Python 3.12.15, marimo 0.25.2. 설치된 VS Code 확장은 `marimo-team.vscode-marimo-0.19.1-linux-x64`.

## 원인과 수정

VS Code 확장의 `dist/extension.js`에 있는 셀 분류 함수는 Markdown parser가 지원하는 코드라도 문자열 prefix에 `f`가 있으면 텍스트 셀로 분류하지 않는다. Python 값을 실행해야 하는 동적 Markdown을 코드 셀로 보존하는 동작이다. marimo 자체의 웹 편집기는 f-string Markdown 편집을 지원하므로, f-string 자체가 잘못된 marimo 구문이라는 뜻은 아니다.

SFT, 사전 학습, PPO, VLM 미세 조정의 두 번째 도입 셀은 `mo.md(f"""...{output_dir}...""")`를 사용했다. 이를 `mo.md(r"""...""")` 정적 문자열로 바꾸고 `output_dir` 셀 인자를 제거했다. 도입부는 `outputs/notebooks/<실행 ID>/`라는 공통 위치를 안내한다. 실제 계산·저장 코드는 유지했다.

DPO와 지식 증류 도입부에 이미 있던 `hide_code=True`를 제거했다. 다른 7개 파일의 도입 셀은 정적 Markdown으로 정상 인식되어 수정하지 않았다. `README.md`에 VS Code의 정적/동적 Markdown 분류 차이를 설명했다. 학습 결과를 삽입하는 다른 동적 설명 셀은 유지한다.

## 검증

작업 디렉토리: 저장소의 `notebooks/`.

```bash
.venv/bin/python ../outputs/notebooks/20261109-202834-marimo-format-review/verify.py
```

검증 스크립트는 13개 파일의 Python 구문, marimo 파싱, marimo 기본 lint 규칙 전체(순환 의존·중복 정의 등 포함), 두 번째 셀의 정적 Markdown 추출과 제목 HTML 생성, 숨김 설정 부재를 확인한다. 수정 전 사본과 AST를 비교해 두 번째 셀 외 계산 셀이 유지되는지도 검사한다. 결과는 `validation.json`, 변경 내용은 `changes.diff`, 수정 전 사본은 `before/`에 보관한다. 모델 가중치나 데이터셋을 사용하지 않았으며 모델·데이터 리비전은 해당 없음이다.

일반 `marimo check --strict --format json *.py`는 이 실행 환경에서 비동기 파일 탐색/읽기 단계의 대기가 끝나지 않았다. 규칙 검사를 생략하지 않고, 동기 파일 읽기 후 동일한 `RuleEngine.create_default().check_notebook(...)`로 전체 기본 규칙을 실행했다. CLI 결과의 빈 JSON 파일은 통과 근거로 사용하지 않았다.

VS Code 화면 조작·스크린샷 검증과 전체 학습 실행은 수행하지 않았다. 설치된 확장의 분류 조건, 정적 Markdown 인식, 실제 Markdown 제목 렌더링까지 확인한 범위다. 이미 열린 탭의 셀 분류가 남아 있으면 파일을 닫았다가 marimo 편집기로 다시 열어 확인한다.

참고: [marimo Markdown 편집 안내](https://marimo-team.github.io/jupyter-book-marimo/tutorials/markdown/), [marimo 출력 안내](https://docs.marimo.io/guides/outputs/). VS Code의 f-string 분류 조건은 설치된 확장 소스로 직접 확인했다.
