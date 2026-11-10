# MiniMind 추가 토크나이저 정리

`projects/minimind/model/fineweb-edu-score-3.5/` 전체(기존 토크나이저와 해당 학습 메타데이터), 중복 `vocab.json`, `merges.txt`를 제거했다.
현재 Hugging Face 토크나이저의 tokenizer.json, tokenizer_config.json, special_tokens_map.json, chat_template.jinja는 유지했다.

백업과 결과: `outputs/minimind/20261110-091112-tokenizer-cleanup/` (backup/, changes.json, verification.json).

작업 디렉토리: `/home/mind4u/dev/llm_playground`.
검증 인터프리터: `projects/minimind/.venv/bin/python`.
Python 3.12.15, transformers 4.57.6, tokenizers 0.22.2.
`AutoTokenizer.from_pretrained(path, local_files_only=True)`로 로딩하여 원본과 backend, chat template, 한국어·영어·특수 토큰 인코딩 및 대화 입력 토큰화를 비교했다. 모두 PASS, vocab 6400.
비교 원본은 `caspar/MiniMind-Dense-64M-Mixed-SFT-v1.0` 리비전 `7597c22aa4d3e7b2ac824b4b4be01158cccf1b8d`의 기존 다운로드 사본이다.
데이터셋과 모델 가중치는 사용하지 않았으며 학습·모델 추론 및 전체 단위 테스트는 실행하지 않았다.
