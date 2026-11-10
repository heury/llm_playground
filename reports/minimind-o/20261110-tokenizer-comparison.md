# 세 MiniMind 모델 토크나이저 비교

확인일: 2026-11-10 (Asia/Seoul). 세 모델은 동일한 토크나이저를 사용한다. 가중치는 다운로드하거나 실행하지 않았다.

## 대상과 리비전

- omni: `caspar/MiniMind-O-VLM-Audio-Mini-5090-20261110` / `982f2e40f696013a4645937b164b3025d4d3d4f0`
- vlm: `caspar/MiniMind-V-Dense-64M-LLaVA-Short-v1.0` / `4e6f87f184b27f0692e1f8b3afed959de254894b`
- lm: `caspar/MiniMind-Dense-64M-Mixed-SFT-v1.0` / `7597c22aa4d3e7b2ac824b4b4be01158cccf1b8d`

## 근거

- 세 tokenizer.json은 429,052 bytes이며 SHA-256이 모두 `312258c4f87cd509a830f418a8e430a4cf203d2c17827e324c8baf5946755417`로 일치한다.
- BPE vocab 6,400개, merges, added_tokens, normalizer, pre_tokenizer, post_processor, decoder를 포함한 JSON 전체가 같다.
- BOS=1, EOS=2, PAD=0. 이미지 토큰=12, 오디오 시작=14, 끝=15, 패딩=16. 이 토큰들은 세 모델 모두 같은 ID를 갖는다.
- Omni와 LM은 tokenizer_config.json, special_tokens_map.json, chat_template.jinja도 바이트 단위로 동일하다.
- VLM은 동일한 chat template을 tokenizer_config.json에 내장한다. Omni와 LM은 별도 chat_template.jinja에 둔다. VLM에는 빈 extra_special_tokens 필드가 생략되어 있다.
- AutoTokenizer로 로딩한 실제 chat_template은 세 모델 모두 완전히 동일하다.
- 한국어·영어·중국어·이미지/오디오 특수 토큰 4개 입력의 encode 결과와 대화 템플릿 렌더링 결과가 모두 일치한다.

## 재현 및 환경

- 작업 디렉토리: `/home/mind4u/dev/llm_playground`
- 파일 수집 및 비교: `python3 outputs/minimind-o/20261110-tokenizer-comparison/compare.py`
- 실행 비교 환경: `projects/minimind/.venv/bin/python`, Python 3.12.15, transformers 4.57.6, tokenizers 0.22.2.
- 실행 비교: 각 다운로드 폴더를 `AutoTokenizer.from_pretrained(path, local_files_only=True)`로 로딩하여 encode 및 apply_chat_template 비교. 입력과 전체 결과는 runtime.json에 기록했다.
- 산출물: `outputs/minimind-o/20261110-tokenizer-comparison/` (수집 파일, compare.py, comparison.json, runtime.json).

## 검증 범위

- 지정 리비전의 텍스트 토크나이저와 대화 템플릿 비교 결과다. 모델 추론, 학습, 오디오 코덱과 비전 전처리 호환성은 검증하지 않았다.
- 이 확인에는 데이터셋을 사용하지 않았다.
