# 모델 구조 개선 스크립트의 marimo 변환

2026-11-09 (Asia/Seoul). `gpt/model_improvements/`의 Python 파일 9개를
`gpt/gpt_with_*.py` marimo 앱으로 변환하고 원래 디렉토리를 제거했다.

## 변경 범위

KV cache, MHA, GQA, MLA, DSA, SWA, 레이어 간 KV 공유, MoE, DeltaNet을
각각 독립된 앱으로 제공한다. import, 모델 클래스, 생성 함수, 설정과 실행 결과를
별도 셀로 구성했다. 기존 클래스와 생성 함수의 연산 코드는 유지했다.
원본의 저작권·라이선스·출처 주석도 파일 상단에 유지했다.

CLI의 argparse 옵션은 marimo 설정 입력으로 바꿨다. 기존 기본값을 보존하며,
MLA의 `latent_dim=None`은 드롭다운의 `자동`으로 선택한다. KV cache 원본의
고정 모델 크기와 생성 길이도 같은 기본값의 설정으로 노출했다. MoE의 FFN
프로파일 기록 리스트는 별도 셀에 정의했다. DeltaNet에는 작은 합성 입력으로
Gated MHA와 Gated DeltaNet의 순전파·역전파를 확인하는 실습을 추가했다.

실행 버튼 전에는 모델을 생성하거나 토크나이저를 다운로드하지 않는다.
버튼을 누르면 기존 기본값 또는 수정한 설정으로 실행하고, `prepare_run`을 통해
`outputs/gpt/<실행 ID>/result.json`과 `run.log`에 설정·결과·출력을 저장한다.
GPT 모델은 무작위 가중치로 초기화한다.

README의 명령·파일 목록과 루트 AGENTS 안내를 갱신했다. CodeTour 5개의
파일 경로 및 기존 코드 위치에 대응하는 줄 번호를 변경했다. 다른 Chapter 앱과
`previous_chapters.py`, 프로젝트 의존성과 잠금 파일은 이번 작업에서 수정하지 않았다.

## 검증 환경과 명령

작업 디렉토리: `/home/mind4u/dev/llm_playground`.

- Python 3.12.15, marimo 0.25.2, PyTorch 2.15.0+cpu.
- 기존 `notebooks/.venv/bin/python`을 검사에 사용했다.
- 해당 환경에 없던 tiktoken 0.14.0과 의존성은 uv로 `/tmp/gpt-improvements-validation-deps`에 설치했다. 기존 가상환경은 변경하지 않았다.
- `gpt/uv.lock`의 PyTorch 환경을 설치한 검증은 아니다.
- 모델·데이터 리비전: 로컬 소스와 합성 token ID를 사용했다. 외부 모델 가중치, 데이터셋, 토크나이저 파일 다운로드는 수행하지 않았다.

```bash
notebooks/.venv/bin/marimo check --strict gpt/gpt_with_*.py
PYTHONPATH=/tmp/gpt-improvements-validation-deps \
MPLCONFIGDIR=/tmp/gpt-improvements-mpl \
notebooks/.venv/bin/python \
  outputs/gpt/20261109-204756-model-improvements-marimo/validate.py
git diff --check
```

검증 스크립트는 실행 중 작업 디렉토리를 `gpt/`로 설정한다. 제거한 원본은
산출물의 `originals.json`에서 읽으므로 검사를 다시 실행할 수 있다.

## 결과

| 검사 | 결과 |
| --- | --- |
| Python 구문·컴파일 | 9/9 통과 |
| `marimo check --strict` | 종료 코드 0 |
| 기존 클래스·생성 함수 AST 비교 | 9/9 동일, docstring 들여쓰기는 정규화 후 비교 |
| 앱 그래프·미정의 참조 검사 | 9/9 통과 |
| 원본 기본값과 UI 초기값 비교 | 9/9 통과 |
| 버튼 미클릭 초기 실행 | 9/9 통과, 모델 실행·출력 폴더 생성 대기 확인 |
| CPU 순전파·역전파 | 9/9 통과 |
| 원본과 변환본 출력 비교 | GPT 8개 및 DeltaNet 파일의 두 모듈 모두 동일 |
| 캐시 사용·미사용 logits 및 greedy 생성 비교 | GPT 8/8 통과 |
| DeltaNet 추가 실습 함수 | 통과 |
| CodeTour 파일·줄 번호 유효성 | 5개 파일 통과 |
| 원본 디렉토리 제거 | 원본 내용이 변하지 않았음을 확인한 뒤 9개 파일과 디렉토리 제거 |
| `git diff --check` | 통과 |

GPT 검증 설정은 CPU float32, 어휘 32, 임베딩 16, 4헤드, 2층, 문맥 길이 16이다.
MLA latent 8, GQA KV 그룹 2, 공유 KV 생성 레이어 1, MoE expert 2/top-2,
DSA indexer 2헤드·차원 4/top-k 4, SWA window 4/stride 2를 사용했다.
DeltaNet 비교에는 `[2,8,16]`, 추가 실습 함수에는 기본 `[2,8,32]` 입력을 사용했다.
기울기의 유한성과 원본 대비 출력 일치, 캐시 logits의 허용 오차 내 일치 및 생성
token ID의 완전 일치를 판정 근거로 삼았다.

브라우저 UI 클릭, 기본 124M 규모의 전체 생성 벤치마크, 실제 tokenizer를 사용한
텍스트 생성, GPU 실행 및 학습은 수행하지 않았다. 처리량·메모리 성능은 검증하지
않았으며 기존 모델의 알고리즘 전체를 재검증한 결과로 해석하지 않는다.

산출물: [outputs/gpt/20261109-204756-model-improvements-marimo/](../../outputs/gpt/20261109-204756-model-improvements-marimo).
`validation.json`에 버전·파일 SHA256·앱별 판정, `validation.log`에 실행 출력,
`validate.py`에 재실행 절차, `originals.json`에 제거 전 소스와 위치 매핑을 보관했다.
`outputs/`는 Git 제외 경로다.
