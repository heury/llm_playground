# minimind_v_finetuning.py 검토·재구성 검증

- 대상: `notebooks/minimind/minimind_v_finetuning.py`, `notebooks/minimind/README.md` (4-2 행 설명)
- 실행 위치: `notebooks/minimind/`
- 명령: `.venv/bin/python minimind_v_finetuning.py`, `.venv/bin/marimo check minimind_v_finetuning.py`
- 환경: torch 2.15.0+cpu, transformers 4.57.6, marimo 0.25.2 (CPU, 약 10초)
- 산출물: `outputs/notebooks/20261110-200000-6100ff-vlm-finetuning/`
  (`training-*/` 아래 `llm_32.pth`, `pretrain_vlm_32.pth`, `sft_vlm_32.pth`, `metrics.json`, `image-use-probe.json`, `loss-curves.png`, `restore-check.json`)

## 검토에서 찾은 핵심 문제

기존 노트북은 **무작위 LLM**에서 Projector 정렬을 수행했습니다. 정렬 뒤 greedy 응답은 `assistant ×6`이었고,
이미지를 바꿨을 때 CE 증가는 0.02~0.04에 그쳤습니다. 이미지 사용은 SFT에서만 나타났습니다.
원본(`from_weight=llm`, `freeze_llm=2`)에서 정렬이 의미를 갖는 이유, 즉 **학습된 LLM의 임베딩 공간에
Projector를 맞추는 것**이 재현되지 않았습니다.

사전 실험(스크래치)으로 두 경우를 비교했습니다.
- 일반 텍스트 형식만 학습한 LLM: Projector만 학습해서는 색을 배우지 못했습니다(정렬 CE ≈ 0.35, 색 확률 약 0.5).
- **문장 속 색 단어를 따라 말하는** 텍스트 LLM: Projector만 학습해 색 확률 0.92~0.98, 이미지를 바꾸면 0.01.

후자를 채택했습니다.

## 변경 내용

| 절 | 변경 |
| --- | --- |
| 도입 | 원본 3단계 파이프라인(텍스트 LLM → 정렬 → SFT)과 단계별 학습 목표·확인 방법 표 |
| 1 | 텍스트 LLM용 대화(이미지 없음) 추가, 정렬·SFT 검증 split 분리 (`val_alignment`, `val_sft`) |
| 2 | `base_model` 대신 processor만 로드 |
| 3 | label 표에서 PAD 행 제외, shape 흐름 다이어그램을 이 절로 이동 |
| 4 (신규) | ① 텍스트 LLM 학습(원본 `VLMDataset.create_chat_prompt`·`generate_labels` 사용) → `llm_32.pth` 저장(Projector 제외, 원본 `llm_768.pth`와 같은 구성) |
| 5 | 동결 정책 표 정리, **LLM 동결 이유·학습률 80배 차이·첫·마지막 층만 푸는 이유** 설명 추가 |
| 삭제 | 구 4절(교체 검사)과 구 5절(수동 CE·gradient 표): VLM 노트북 4·6절과 중복 |
| 학습 전 점검 | 인위적으로 EOS를 뺀 "PAD가 정답이 되는" 사례 삭제(절단 경로에서는 PAD가 붙지 않아 발생하지 않음). `max_length=8` 사례만 유지 |
| 6 | 정렬을 `load_vlm("llm", freeze=2)`에서 시작, 로딩된 LLM 가중치가 저장본과 같은지 assert |
| 8 | 지표를 답변 평균 CE에서 **색 토큰 정답 확률(정상/바꾼 이미지)** 과 greedy 정확 일치로 교체. 단계별 판정 assert 추가 |
| 10 | 과제 재구성: 텍스트 LLM 없이 정렬, 정렬 생략, SFT 동결 정책, 흰 이미지, Projector |

## 판정 근거 (`image-use-probe.json`, 기본 설정 정렬 80 / SFT 80 step)

| 단계 | 짧은 질문 정상 / 바꾼 | 지시 질문 정상 / 바꾼 | 지시 greedy 일치 |
| --- | --- | --- | --- |
| ① 텍스트 LLM | 0.496 / 0.497 | 0.066 / 0.032 | 0/4 (`빨강`) |
| ② 정렬 뒤 | 0.992 / 0.001 | 0.763 / 0.000 | 0/4 (`빨강`) |
| ③ SFT 뒤 | 0.024 / 0.038 | 0.991 / 0.001 | 4/4 (`색 은 빨강`) |

- 텍스트 LLM이 문장 속 색 단어를 답할 확률: 0.99 이상
- ③의 짧은 질문 확률 하락은 응답 형식을 덮어쓴 결과입니다(첫 토큰이 `색`). 노트북 8절에 망각 사례로 설명했습니다.
- 슬라이더 극단값(20/20, 160/160 step)에서도 8절 assert가 모두 통과했습니다.
- SFT 가중치 복원 시 logits 정확 일치, 저장 파일에 `vision_encoder.*` 없음.

## 한계

- 무작위 SigLIP과 3층 LLM을 사용한 두 색 사각형 과제입니다. 실제 가중치·데이터·AMP·DDP·원본 trainer 전체는 검증하지 않았습니다.
- 학습률(텍스트 3e-3, 정렬 2e-2, SFT 3e-3)은 작은 모델에 맞춘 값이며 원본 값(4e-4, 5e-6)이 아닙니다.
- marimo 편집기 화면 렌더링은 직접 보지 않았습니다. 스크립트 실행과 표 데이터만 확인했습니다.
