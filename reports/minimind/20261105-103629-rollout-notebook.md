# Rollout 실습 노트북 실행 검증

- 대상: [rollout_engine_walkthrough.ipynb](../../minimind/notebooks/rollout_engine_walkthrough.ipynb)
- 결과: **passed**, 코드 셀 11개, 2.095초
- 작업 디렉토리: `minimind/`
- 명령: `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 /tmp/minimind-v-exercise-venv/bin/python /tmp/verify_rollout_notebook.py`
- 방법: JSON/셀 구문 검사 후 모든 코드 셀을 같은 Python namespace에서 순서대로 실행하고 assert 확인. Jupyter 커널을 통한 실행은 수행하지 않음.
- 환경: Python 3.13.13, PyTorch 2.6.0+cpu, Transformers 4.57.6, CPU. 프로젝트 권장 Python 3.12 환경 대신 기존의 사용 가능한 Python 3.13 환경에서 검증함.
- 소스 리비전: `e2c613e1b240114cae92215d1f6d680e9a6f5159`. 이번에 추가한 노트북은 커밋 전 작업 트리 파일.
- 모델: 무작위 MiniMind(hidden_size=32, 1 layer, vocab=64), 초기화 seed=7, 생성 seed=42.
- 데이터: 노트북 내 64개 단어와 프롬프트 2개, EOS 마스크용 합성 입력. 외부 모델·데이터 리비전 없음.
- 산출물: `outputs/minimind/20261105-103629-rollout-notebook/`의 `executed.ipynb`, `execution.log`, `status.json`. 원본 노트북의 출력/실행 번호는 비워 둠.

## 판정 근거

입력 복제의 질문별 그룹 순서, 반환 필드와 크기, 원본 rollout과 단계별 재현 결과의 일치,
전체 logits 및 음의 교차 엔트로피와 토큰별 로그 확률의 일치, 첫 EOS/패딩 경계,
rollout의 gradient 차단과 재계산 후 backward 연결, 설정 변경 후 크기와 seed 재현성을 assert로 확인했다.

## 재현 방법

`minimind/`에서 `uv sync --locked` 후 노트북을 열고 `minimind/.venv/bin/python`을 커널로 선택해 Run All을 수행한다.
위 `/tmp/`의 검증 실행기는 이번 로컬 검증에만 사용했으며 저장소 배포 파일이 아니다.
실행 결과를 보존하려면 출력이 있는 사본을 해당 실행의 `outputs/` 디렉토리에 저장한다.

## 미실행 및 한계

Jupyter UI/커널 및 Python 3.12 환경에서의 실행, DDP, GPU/autocast, SGLang,
실제 체크포인트/토크나이저, 전체 GRPO 학습과 응답 품질은 검증하지 않았다.
무작위 가중치의 작은 CPU 실행 결과이며 실제 학습의 성공이나 성능을 의미하지 않는다.
