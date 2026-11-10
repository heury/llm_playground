# 구조 노트북 학습 전후 테이블 분리 검증

- 대상: `notebooks/minimind/minimind_structure.py`
- 변경: 생성 결과, 후보 토큰 점수·확률, Attention 확률을 각각 학습 전/학습 후 두 표로 표시한다. 각 표 위에 시점 제목을 붙이고 공통 열 이름을 사용한다.
- 실행 디렉토리: `notebooks/minimind/`
- 실행 명령: `.venv/bin/python -m marimo check minimind_structure.py`, `.venv/bin/python minimind_structure.py`
- 환경: Python 3.12.15, marimo 0.25.2, PyTorch 2.15.0+cpu.
- 모델·데이터: 현재 작업 트리의 MiniMind 구현과 노트북 내부 합성 데이터. 외부 가중치·데이터를 사용하지 않음.
- 결과: 두 명령 모두 종료 코드 0. 기본 설정의 CPU 학습과 노트북 assertion을 포함한 전체 셀 실행 성공.
- 산출물: `outputs/notebooks/20261110-structure-separate-tables/check.log`, `outputs/notebooks/20261110-structure-separate-tables/run.log`.
- 제한: IDE 화면의 시각적 배치는 직접 확인하지 않음. Matplotlib 설정 디렉토리 쓰기 권한 경고가 있었으며 임시 캐시로 실행 완료.
