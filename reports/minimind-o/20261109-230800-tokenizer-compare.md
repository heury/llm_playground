# 세 MiniMind 모델의 토크나이저 비교

확인일: 2026-11-09 (Asia/Seoul).

## 결론

MiniMind-Dense Mixed-SFT와 MiniMind-V LLaVA-Short의 토크나이저는 동일하다. MiniMind-O Omni mini는 다른 어휘와 BPE 병합 규칙을 사용하므로 앞의 두 모델 토크나이저로 교체하면 안 된다. 어휘 크기가 같아도 같은 ID가 다른 텍스트를 뜻할 수 있어 임베딩 및 출력 head와 불일치한다.

| 항목 | Dense Mixed-SFT | V LLaVA-Short | O Omni mini |
|---|---|---|---|
| 어휘 수 | 6,400 | 6,400 | 6,400 |
| BPE 병합 규칙 수 | 6,108 | 6,108 | 6,108 |
| tokenizer.json | Dense와 동일 | Dense와 바이트 단위 동일 | Dense/V와 다름 |
| 추가 토큰 정의 | 동일 | 동일 | 동일 |
| 정규화·사전 분할·후처리·디코더 | 동일 | 동일 | 동일 |
| 유효 채팅 템플릿 | 동일 | 동일 | 동일 |

Dense/V와 Omni 사이에 공통 토큰 문자열은 2,322개이고, 문자열과 ID가 모두 같은 토큰은 292개다. 나머지 어휘 및 병합 순서가 다르다. 전체 tokenizer.json을 비교했으며 샘플만으로 동일성을 판정하지 않았다.

세 모델의 주요 ID는 PAD/UNK `<|endoftext|>`=0, BOS `<|im_start|>`=1, EOS `<|im_end|>`=2, `<|image_pad|>`=12, `<|audio_start|>`=14, `<|audio_end|>`=15, `<|audio_pad|>`=16, `<think>`=25, `</think>`=26으로 같다.

Dense와 Omni의 tokenizer_config.json, special_tokens_map.json 및 별도 chat_template.jinja는 각각 바이트 단위로 동일하다. V는 채팅 템플릿을 tokenizer_config.json에 내장하고, Dense/Omni의 빈 extra_special_tokens={} 항목이 없지만, 실제 유효 채팅 템플릿 내용은 같다. 설정 파일의 동일성은 어휘 동일성을 보장하지 않는다.

## 실제 실행

Transformers AutoTokenizer로 로컬 다운로드 파일을 읽어 영어·한국어·중국어·모달 특수 토큰 입력 5개를 인코딩했다. Dense와 V는 모든 샘플에서 같은 ID를 반환했고 Omni는 일반 문장에서 다른 ID를 반환했다. 사용자 메시지 하나에 add_generation_prompt=True를 적용한 채팅 문자열은 세 모델 모두 일치했다.

`Hello, world!` (`add_special_tokens=False`):
- Dense/V: `[75, 497, 114, 47, 1009, 36]`
- Omni: `[1602, 47, 1707, 36]`

모달 토큰 입력 `<|image_pad|><|audio_start|><|audio_pad|><|audio_end|>`는 모두 `[12, 14, 16, 15]`다. 토큰 존재만으로 해당 모델이 모달 입력을 처리할 수 있음을 뜻하지는 않는다.

현재 IDE의 `projects/minimind-v/model/tokenizer.json`은 Omni의 tokenizer.json과 SHA256이 같다. 공개된 V LLaVA-Short 체크포인트의 토크나이저와는 다르므로 그 체크포인트 실행 시 해당 Hub 저장소의 토크나이저를 사용해야 한다. 소스 파일이나 원격 저장소는 변경하지 않았다.

## 리비전과 해시

- llm: https://huggingface.co/caspar/MiniMind-Dense-64M-Mixed-SFT-v1.0/tree/7597c22aa4d3e7b2ac824b4b4be01158cccf1b8d
  - 리비전: `7597c22aa4d3e7b2ac824b4b4be01158cccf1b8d`
  - tokenizer.json SHA256: `312258c4f87cd509a830f418a8e430a4cf203d2c17827e324c8baf5946755417`
- vlm: https://huggingface.co/caspar/MiniMind-V-Dense-64M-LLaVA-Short-v1.0/tree/4e6f87f184b27f0692e1f8b3afed959de254894b
  - 리비전: `4e6f87f184b27f0692e1f8b3afed959de254894b`
  - tokenizer.json SHA256: `312258c4f87cd509a830f418a8e430a4cf203d2c17827e324c8baf5946755417`
- omni: https://huggingface.co/caspar/MiniMind-O-Mini-LLaVA-5090-20261025/tree/b7b050060232c91df87ceff4aa8a507cc8021191
  - 리비전: `b7b050060232c91df87ceff4aa8a507cc8021191`
  - tokenizer.json SHA256: `71f32c68cf63a15355a8fc171b7594b3d41870fe0ddb54fc6aefa55f73a4a668`

## 환경·재현·검증 한계

작업 디렉토리: `/home/mind4u/dev/llm_playground`.
Python 3.13.13, huggingface_hub 0.36.2, transformers 4.57.6, tokenizers 0.22.2.

```bash
HF_HUB_CACHE=/tmp/minimind-tokenizer-hf-cache /tmp/minimind-v-exercise-venv/bin/python outputs/minimind-o/20261109-230800-tokenizer-compare/compare.py
```

파일 원본·전체 비교·리비전·해시: `outputs/minimind-o/20261109-230800-tokenizer-compare/`의 모델별 하위 폴더와 `comparison.json`. 실제 인코딩과 채팅 템플릿 실행 결과: 같은 폴더의 `runtime.json`.
모델 가중치 로딩·생성·학습은 실행하지 않았다. 이번 비교는 지정 리비전의 토크나이저 파일과 토크나이저 런타임 동작에 한정하며 데이터·가중치 리비전의 새 검증은 수행하지 않았다.
