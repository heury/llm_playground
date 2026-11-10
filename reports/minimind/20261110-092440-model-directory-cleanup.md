# model 디렉토리 정리

이전 토크나이저 학습·검증 기록 6개(corpus_audit.json, data_validation.json, dataset_manifest.json, requirements-resolved.txt, training_info.json, validation.json)를 `outputs/minimind/20261110-092440-model-directory-cleanup/archived-model-metadata/`로 이동했다. `__pycache__/`는 제거했다.

남긴 파일은 __init__.py, model_minimind.py, model_lora.py 및 현재 HF 토크나이저의 tokenizer.json, tokenizer_config.json, special_tokens_map.json, chat_template.jinja 총 7개이며 SHA-256으로 내용 불변을 확인했다.

코드 참조를 확인했다. 기존 Runpod 실행기의 SOURCE/model 메타데이터 참조는 원격 `/shared-volume/` 학습 산출물 경로이므로 로컬 model 정리에 따라 바꾸지 않았다. train_tokenizer.py는 새 학습 출력 디렉토리에 training_info.json을 생성하며 기존 로컬 기록은 읽지 않는다.

작업 디렉토리: `/home/mind4u/dev/llm_playground`. 검증 인터프리터: `projects/minimind/.venv/bin/python` (PYTHONDONTWRITEBYTECODE=1).
Python 3.12.15, transformers 4.57.6, tokenizers 0.22.2.
모델 Python 파일 3개의 AST 구문 검사 및 AutoTokenizer 로컬 로딩, backend 전체·chat template·샘플 인코딩·대화 토큰화를 기존 HF 원본과 비교하여 PASS했다.
HF 원본: caspar/MiniMind-Dense-64M-Mixed-SFT-v1.0 @ 7597c22aa4d3e7b2ac824b4b4be01158cccf1b8d.
가중치·데이터셋을 사용하지 않았으며 학습·모델 추론·전체 테스트는 실행하지 않았다.

산출물: `outputs/minimind/20261110-092440-model-directory-cleanup/` (archived-model-metadata/, changes.json, verification.json).
