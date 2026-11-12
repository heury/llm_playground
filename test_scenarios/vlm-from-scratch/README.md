# VLM from Scratch 실행 검토 시나리오

작업 디렉토리는 `notebooks/vlm-from-scratch/`, 인터프리터는 해당 프로젝트 `.venv/bin/python`을 사용한다. 먼저 프로젝트 README와 루트 AGENTS.md를 확인한다.

## 준비 및 실행

1. Python·torch·torchvision·transformers·marimo 버전과 `torch.cuda.is_available()`를 기록한다. 가중치·데이터 캐시와 디스크 여유를 확인한다.
2. 결과는 저장소 루트 `outputs/vlm-from-scratch/<실행 ID>/`, 보고서는 `reports/vlm-from-scratch/<실행 ID>.md`에 저장한다.
3. 각 앱을 개별 프로세스에서 `app.run()`으로 실행한다. 별도의 실행 버튼 없이 모델·데이터 로딩부터 진행한다.
4. 다운로드 없이 준비 상태를 확인할 때는 `HF_HUB_OFFLINE=1`, `HF_DATASETS_OFFLINE=1`을 명시한다. 이때 캐시 누락은 환경 제약이며 전체 실행 통과로 판정하지 않는다.
5. 소형 CPU 검증에서는 실제 HF ViT/Llama 구조와 합성 PIL 데이터로 원본 함수의 순전파·역전파·토큰 생성·배치를 검사한다. 전체 실행 결과와 분리한다.
6. 실제 GPU·가중치·데이터를 사용하는 통합 실행은 별도 시나리오로 기록하고 모델·데이터 리비전과 학습 설정을 명시한다.

## 판정 기준

- 유한한 손실, 프로젝터 기울기, 고정된 비전 인코더 확인.
- 데이터 배치의 패딩 레이블이 -100이며 학습할 응답 토큰이 존재하는지 확인.
- 체크포인트 저장·재개 시 손실 이력과 에포크가 이어지는지 확인.
- 양자화 후 실제 순전파와 packed 가중치를 포함한 크기 측정 확인.
- OCR의 학습 및 추론 경로에서 실제 SmolLM/Llama 임베딩 API 확인.
- 마스크·다각형 변환, 이미지 편집 연산, LaTeX 렌더링 확인.
- 정적 검사나 소형 모델 통과를 실제 학습·생성 품질 통과로 보고하지 않는다.

첫 실행의 재현 스크립트와 상세 입력은 `outputs/vlm-from-scratch/20261111-135349-execution-review/`에 보관했다. [검토 보고서](../../reports/vlm-from-scratch/20261111-135349-execution-review.md) 참조.

## A40 원격 검증

[Runpod A40 보고서](../../reports/vlm-from-scratch/20261111-140926-runpod-a40.md)와 `outputs/vlm-from-scratch/20261111-140926-runpod-a40/remote-results/`의 실행 스크립트를 참고한다. 실제 GPU 실행은 사용자가 요청했을 때 수행하고, 요청한 GPU 재고·요금과 CUDA 하한을 확인한다. 이 검증에서는 1에포크·최대 2배치, 실제 가중치, 데이터 16행을 반복한 64행 뷰를 사용했다. 연속 실행과 호환 체크포인트를 사용하는 별도 실행을 구분한다.

실행 완료와 정확도·학습 품질을 분리하고, 9부의 JSON 응답 타입, 12부의 흑백/RGB 입력 및 Llama 임베딩 접근, 10부 packed 가중치의 용량 포함 여부를 확인한다. 8부의 COCO fallback을 RefCOCO 검증으로 표시하지 않는다. 원격 결과를 회수해 로컬에서 확인한 뒤 해당 검증에서 생성한 Pod만 삭제하고 새 조회의 not_found를 기록한다.

## A40 오류 수정 후 로컬 회귀 검사

`notebooks/vlm-from-scratch/`에서 `HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 .venv/bin/python -m unittest discover -s tests -v`로 실행한다. 2~9부의 체크포인트 구성·전체 가중치·우선순위와 순전파/생성, 9부 응답 스키마, 12부 RGB/임베딩, 10부 INT8 직렬화 크기를 검사한다. GPU 실제 가중치 실행과는 분리해 판정한다. [수정 및 검증 보고서](../../reports/vlm-from-scratch/20261111-145314-notebook-fixes.md) 참조.
