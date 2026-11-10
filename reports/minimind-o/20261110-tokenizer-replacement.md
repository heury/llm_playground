# 로컬 MiniMind 토크나이저 교체

확인일: 2026-11-10 (Asia/Seoul).

## 변경

Hugging Face `caspar/MiniMind-Dense-64M-Mixed-SFT-v1.0`의 리비전
`7597c22aa4d3e7b2ac824b4b4be01158cccf1b8d`에서 앞선 비교 작업 중 내려받은 파일을 사용했다.
세 HF 모델의 tokenizer.json과 실제 대화 템플릿은 동일하다.

다음 5개 프로젝트의 `model/`에 tokenizer.json, tokenizer_config.json,
special_tokens_map.json, chat_template.jinja를 동일하게 적용했다.

- projects/minimind
- projects/minimind-v
- projects/minimind-o
- projects/minimind-a
- projects/minimind-j2

minimind와 minimind-j2의 tokenizer.json은 이미 같아 내용 변경이 없다.
설정은 HF LM/Omni 배포 형태에 맞추어 대화 템플릿을 별도 jinja 파일로 제공한다.
기존 vocab.json과 merges.txt는 새 토크나이저와 일치함을 확인하여 유지했다.

독립 토크나이저 사본이 없는 minimind-j/d/la에는 새 파일을 만들지 않았다.
minimind-j의 기본 ByteTokenizer와 명시적으로 선택하는 외부 tokenizer 경로도 변경하지 않았다.
별도 실험 이력인 minimind/model/fineweb-edu-score-3.5 및 체크포인트/캐시는 교체 범위에서 제외했다.

## 검증 결과

5개 경로 모두 PASS:

- 네 배포 파일의 SHA-256이 HF 원본과 일치.
- AutoTokenizer 로컬 로딩, 전체 vocab/토큰 ID 및 backend 설정 일치.
- vocab 6,400, BOS=1, EOS=2, PAD=0.
- 한국어·영어·중국어·이미지/오디오 특수 토큰 입력 4개의 encode 결과 일치.
- 대화 템플릿 전체 일치와 thinking on/off 토큰화 결과 일치.
- 기존 vocab.json/merges.txt와 새 tokenizer.json의 일관성 확인.

## 실행 및 산출물

작업 디렉토리: `/home/mind4u/dev/llm_playground`.

```bash
projects/minimind/.venv/bin/python outputs/minimind-o/20261110-tokenizer-replacement/replace_and_verify.py > outputs/minimind-o/20261110-tokenizer-replacement/verification.log
```

이 스크립트는 최초 교체와 백업용이며 기존 백업을 덮어쓰지 않도록 재실행 시 중단한다.
환경: Python 3.12.15, transformers 4.57.6, tokenizers 0.22.2.
데이터셋과 모델 가중치를 사용하지 않았다.

산출물: `outputs/minimind-o/20261110-tokenizer-replacement/`.
원본 파일 백업은 `backup/<프로젝트>/`, 전후 해시 및 검증 결과는 `manifest.json`,
실행 출력은 `verification.log`에 있다.

## 한계 및 기존 가중치

V/O/A의 기존 토크나이저는 HF 버전과 vocab이 달랐다. 공통 토큰 중 2,030개는
ID가 다르고, 기존 전용 토큰 4,078개가 새 토큰 4,078개로 교체된다.
따라서 기존 토크나이저로 학습한 로컬 가중치는 이 교체만으로 호환되지 않는다.
이전 가중치에는 백업 토크나이저를 사용하거나, 새 토크나이저와 함께 학습된 HF 가중치를 사용해야 한다.
가중치 변환·교체·학습과 전체 프로젝트 테스트는 실행하지 않았다.
