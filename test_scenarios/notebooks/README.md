# MiniMind 강의 실습 검증 시나리오

대상은 `notebooks/minimind/minimind_*.py` 14개 marimo 앱입니다.
프로젝트의 기존 `.venv`를 사용하고, 각 앱을 별도 프로세스에서 실행합니다.
입력은 소스에 정의된 tiny 무작위 모델·합성 데이터와 저장소의 로컬 tokenizer입니다.
가중치·데이터 다운로드나 GPU 학습 없이 실행합니다.

## 공통 절차

1. 저장소 루트에서 한국 시각의 실행 ID로 `outputs/notebooks/<실행 ID>/`를 만듭니다.
2. 대상 파일·README·실제 참조 구현의 SHA-256과 Python/패키지 버전을 보존합니다.
3. `notebooks/minimind/`에서 다음 정적 검사를 실행해 로그를 저장합니다.

   ```bash
   .venv/bin/python -m marimo check --strict minimind_*.py
   ```

4. 같은 디렉토리에서 각 앱을 새 프로세스로 export합니다. `<출력 경로>`는 위 실행 폴더 안의 HTML입니다.

   ```bash
   HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 MPLBACKEND=Agg \
     .venv/bin/python -m marimo export html <노트북.py> -o <출력 경로>
   ```

   Matplotlib 캐시도 실행 폴더 안의 쓰기 가능한 `MPLCONFIGDIR`로 지정할 수 있습니다.
5. 종료 코드 0뿐 아니라 HTML의 셀 결과에 오류가 없는지, 수치·설명이 대응하는지 확인합니다.
   export와 브라우저/IDE의 실제 화면 확인을 구분합니다.
6. 변형은 원본을 덮어쓰지 않는 실행 사본으로 진행합니다. 파일 위치가 달라지면
   사본의 `repo_root`를 저장소 절대 경로로 고정합니다. 원본과 변형 사본·변경값을 보관합니다.

## 기본 경로

모든 14개 노트북의 기본값으로 구문·셀 의존성·내장 assert를 통과해야 합니다.
모델 성능·학습 효과가 통과했다는 의미는 아닙니다. 실제 실행 결과는 해당 실행 보고서에 기록합니다.

## 과제 변형 입력·판정 기준

| 시나리오 | 실행 사본에서 바꿀 입력 | 판정 기준 |
| --- | --- | --- |
| 부정확한 교사 | distillation의 `teacher_steps=1` | 같은 history 원소 비교로 중단하지 않고 학생 비교까지 도달. `metrics.json`의 teacher_steps도 1 |
| 텍스트 LLM 없이 정렬 | v_finetuning의 기본 실험 검사 해제, `aligned_model=load_vlm("none",freeze=2,...)` | 기본 LLM과 가중치 동일성을 강요하지 않고 8절 관찰 표·복원 검사까지 도달 |
| 정렬 생략 | 같은 검사 해제, `sft_model=load_vlm("llm",freeze=1,...)` | 정렬 체크포인트와 동일성 비교로 중단하지 않고 실제 확률·greedy 결과 표시 |
| 이미지 신호 제거 | 같은 검사 해제, 모든 도형의 색을 흰색으로 변경 | 기본 성공 임계값을 요구하지 않고 정상/교환 이미지 비교표까지 도달. 낮은 성능도 관찰로 보존 |
| GRPO 그룹 1개 | `train_G=1`, 새 정책·reference·optimizer 사용, `weight_decay=0` | advantage=0인 조건에서 정책 state_dict의 전후 값이 정확히 같음. loss gradient와 optimizer 상태/decay의 조건을 구분 |
| Omni 생성 예산 8 | `stream_token_budget=8` (기본 10과 비교) | len assert가 같은 예산을 쓰며 표·메트릭까지 도달. 처음 8개 스트림 반환은 프레임 None |

변형 과제에서 해제하는 것은 기본 실험의 초기 가중치 동일성·고정된 성능 기대값뿐입니다.
shape·동결·저장/복원 검사는 유지합니다. 과제 뒤 원본 설정으로 돌아갑니다.

## 기록

소스·사본·diff·로그·HTML·수치·환경은 `outputs/notebooks/<실행 ID>/`,
명령·작업 디렉토리·모델/데이터 리비전·판정 근거·실패·미실행·한계는
`reports/notebooks/<실행 ID>-summary.md`에 기록합니다.
위 표는 재사용할 기대 조건이며 실제 실행 통과 기록은 아닙니다.
