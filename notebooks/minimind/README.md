# MiniMind 실습 — marimo

`projects/minimind`, `projects/minimind-v`, `projects/minimind-o`의 **실제 클래스와 함수**를 가져와
LLM의 구조, 미세 조정, 선호·강화 학습과 추론(reasoning), 멀티모달 확장을 단계별로 실습하는 노트북 모음입니다.
모든 노트북은 한국어 설명과 작은 합성 데이터로 **CPU에서 다운로드 없이** 실행되며, 각각 단독 실행할 수 있습니다.

## 학습 경로

뒤 노트북은 앞 노트북에서 설명한 내용을 다시 자세히 설명하지 않으므로 아래 순서를 권장합니다.

```text
1부 구조         structure ─┬─ feedforward → moe            (구조 심화, 선택)
                            ├─ sampling                     (생성 심화, 선택)
                            │
2부 학습         pretraining → sft → lora → distillation
                                    │
3부 정렬·추론               dpo → ppo → grpo(+reasoning)
                                    │
4부 멀티모달                vlm → v_finetuning → omni
```

| 단계 | 노트북 | 핵심 질문 | 사용하는 원본 |
| --- | --- | --- | --- |
| 1-1 | [모델 구조와 학습](minimind_structure.py) | 토큰 ID가 Embedding·Attention(RoPE 주파수·GQA·패딩 마스크)·MLP를 지나 다음 토큰 확률이 되는 과정, 생성과 KV cache(RoPE 위치) | `minimind/model/model_minimind.py` |
| 1-2 | [FeedForward](minimind_feedforward.py) | SwiGLU는 무엇을 계산하고, 왜 위치를 섞지 않는가 | `FeedForward`, `MiniMindBlock` |
| 1-3 | [MoE](minimind_moe.py) | router·top-k·가중합, 보조 손실, top-1 straight-through | `MOEFeedForward` |
| 1-4 | [생성과 샘플링](minimind_sampling.py) | greedy vs sampling, temperature·top-k·top-p·repetition penalty, 원본 `generate` 재현, 배치 종료 처리 | `MiniMindForCausalLM.generate` |
| 2-1 | [사전 학습](minimind_pretraining.py) | 원문 → PretrainDataset → 정답 이동·패딩 → 누적·clip·학습률 → 체크포인트 | `train_pretrain.py`, `lm_dataset.py`, `trainer_utils.py` |
| 2-2 | [SFT](minimind_sft.py) | 대화 template, assistant 정답 마스크, **실제 template의 `<think>` 추론 데이터** | `train_full_sft.py`, `SFTDataset` |
| 2-3 | [LoRA](minimind_lora.py) | 같은 목표에서 저랭크 A/B만 학습, 저장·복원·병합 | `model_lora.py` |
| 2-4 | [지식 증류](minimind_distillation.py) | hard label 대신 교사 분포(온도·KL) | `train_distillation.py` |
| 3-1 | [DPO](minimind_dpo.py) | 선호 쌍과 고정 reference로 하는 오프라인 정렬 | `train_dpo.py`, `DPODataset` |
| 3-2 | [PPO](minimind_ppo.py) | rollout·보상·Critic·GAE·clip·reference KL | `train_ppo.py`, `RLAIFDataset` |
| 3-3 | [GRPO와 추론](minimind_grpo.py) | Critic 없는 그룹 advantage, **think 형식 보상 vs GSM8K 정답 보상**, CISPO | `train_grpo.py`, `gsm8k_reward.py` |
| 4-1 | [시각·언어 모델 구조](minimind_vlm.py) | 패치 → SigLIP → Projector → 이미지 자리 교체, gradient 경로, 다중 이미지, KV cache | `minimind-v/model/model_vlm.py` |
| 4-2 | [VLM 미세 조정](minimind_v_finetuning.py) | 텍스트 LLM 준비 → Projector 정렬 → 시각 지시 SFT, 동결 정책·학습률, 단계별 이미지 교환 검사 | `train_pretrain_vlm.py`, `train_sft_vlm.py` |
| 4-3 | [Omni](minimind_omni.py) | 음성 입력, Thinker→Talker, 8개 코드북 지연 배치·손실, 스트리밍, VAD | `minimind-o/model/model_omni.py` |

**단계별 연결점**

- **구조 → 학습:** 구조 노트북 5절의 "같은 배치 반복 학습"이 사전 학습 노트북에서 실제 Dataset·학습률·누적으로 확장됩니다.
- **사전 학습 → SFT:** 손실은 같은 다음 토큰 CE이고, 정답 위치만 assistant 응답으로 바뀝니다.
- **SFT → LoRA·증류·DPO:** SFT·DPO는 *학습 목적*, Full/LoRA는 *갱신 대상*, 증류는 *정답 신호의 종류*를 바꿉니다.
- **SFT → GRPO(추론):** SFT 5-1절에서 `reasoning_content`가 `<think>` 블록으로 학습되는 것을 보고, GRPO 3-1절에서 그 형식과 최종 답에 보상을 주는 방법을 비교합니다.
- **텍스트 → 멀티모달:** 이미지·음성은 Projector로 언어 차원에 맞춘 뒤 **텍스트 자리 일부를 교체**하여 같은 causal LM에 들어갑니다. 정답 마스크는 SFT와 같습니다.

## 수강생 실습 방법

Python 함수, 텐서 shape와 행렬 곱(`@`)을 알고 시작합니다. 각 절의 **예측 → 실행 → 관찰 → 해설**을 따라
먼저 예상 결과를 적고 실행 결과와 비교하세요. 각 노트북 마지막에는 수정 과제와 힌트·해설이 있습니다.

- 셀은 **설명(마크다운) → 실습 코드(순수 Python) → 결과 표시** 순서입니다. 결과 표·그림·위젯을 만드는
  marimo 코드(`mo.vstack`, `mo.ui.table`, `mo.ui.slider` 등)는 `hide_code=True` 셀로 분리해 숨겼으므로,
  보이는 코드 셀에는 모델·데이터·학습 코드만 있습니다. UI 값은 학습 셀에서 `.value`로만 읽습니다.
- `@app.cell`과 셀 함수의 인자·return은 marimo의 셀 간 데이터 전달이며 모델 구조가 아닙니다.
  `_`로 시작하는 변수는 셀 내부의 중간 계산입니다.
- UI(학습 step, rank, β 등)를 바꾸면 관련 셀이 같은 초기 모델에서 다시 실행되므로 학습이 누적되지 않습니다.
- 학습 손실, 검증 손실, 실제 생성 결과는 서로 다른 지표입니다. 함께 기록하세요.

## 실행

Python 3.12와 [uv](https://docs.astral.sh/uv/)가 필요합니다. 최초 의존성 설치에만 인터넷이 필요합니다.
Linux/Windows에는 CPU용 PyTorch를, macOS에는 PyPI 빌드를 설치합니다.

```bash
cd notebooks/minimind
uv sync --locked

# 편집기에서 하나씩 열기 (노트북마다 별도 프로세스)
uv run marimo edit minimind_structure.py
```

서버 없이 전체 셀을 실행해 검증하려면(모든 assertion이 실행되며, 실패하면 종료 코드가 0이 아닙니다):

```bash
cd notebooks/minimind
uv run marimo check minimind_*.py
for f in minimind_*.py; do uv run python "$f" || echo "FAILED: $f"; done
```

결과를 HTML로 보관하려면(모든 셀을 실제 실행합니다):

```bash
RUN_ID="$(date +%Y%m%d-%H%M%S)-marimo"
mkdir -p "../../outputs/notebooks/$RUN_ID"
for f in minimind_*.py; do
  uv run marimo export html "$f" -o "../../outputs/notebooks/$RUN_ID/${f%.py}.html"
done
# 마크다운 셀의 다이어그램은 public/ 상대 경로로 참조하므로 함께 복사합니다.
cp -r public "../../outputs/notebooks/$RUN_ID/"
```

### 다이어그램

구조·흐름 그림은 `public/diagrams/<이름>.mmd`(mermaid 원본)와 미리 렌더링한 `<이름>.png`로 관리합니다.
VS Code marimo 확장은 `mo.mermaid` 출력을 렌더링하지 못하고, 정적 `mo.md(r"""…""")` 셀만 텍스트 셀로 보여 줍니다.
그래서 노트북에서는 마크다운 셀 안에 `<img src="public/diagrams/<이름>.png" width="…">`로 PNG를 넣습니다.
`.mmd`를 수정한 뒤에는 PNG를 다시 만드세요.

```bash
cd notebooks/minimind/public/diagrams
for f in *.mmd; do npx -y @mermaid-js/mermaid-cli@11.4.2 -i "$f" -o "${f%.mmd}.png" -s 2 -b white; done
```

MiniMind-V·Omni 노트북은 다른 프로젝트와 이름이 같은 `model`/`dataset` 패키지를 가져오므로
같은 Python 프로세스에서 다른 노트북을 import하지 말고 각각 새 프로세스로 실행하세요.
참고: [marimo 스크립트 실행](https://docs.marimo.io/guides/scripts/),
[HTML 내보내기](https://docs.marimo.io/guides/exporting/static_html/),
[uv의 CPU용 PyTorch 설정](https://docs.astral.sh/uv/guides/integration/pytorch/).

## 산출물

실행마다 저장소 루트의 `outputs/notebooks/<시각>-<고유 ID>-<노트북>/`이 만들어집니다.
UI 설정을 바꿔 학습을 다시 실행하면 그 안에 `training-<고유 ID>/`(증류는 `students-<고유 ID>/`)가 추가됩니다.
구조·FeedForward·MoE·LoRA 노트북은 파일을 남기지 않습니다(LoRA의 저장·복원 검증은 임시 디렉토리에서 정리).

| 노트북 | 접미사 | 주요 파일 |
| --- | --- | --- |
| 사전 학습 | `-pretraining` | JSONL, tokenizer, `training-*/`의 metrics·microsteps·생성·`pretrain_lab_32*.pth` |
| SFT | `-sft` | 대화·추론 관찰용 JSONL, tokenizer, 기준 가중치, `training-*/`의 기록·생성·`full_sft_32.pth` |
| 지식 증류 | `-distillation` | 대화 JSONL, tokenizer, `students-*/metrics.json`(가중치 저장 없음) |
| DPO | `-dpo` | 선호 JSONL, tokenizer, `training-*/`의 정책 가중치·config·metrics |
| PPO | `-ppo` | RLAIF JSONL, tokenizer, `training-*/`의 metrics·곡선·Actor/Critic 체크포인트 |
| GRPO | `-grpo` | RLAIF JSONL, metrics, 보상 곡선, 참조 trainer 소스 사본 |
| VLM 구조 | `-vlm` | tiny SigLIP, 이미지·대화 Parquet, `summary.json` |
| VLM 미세 조정 | `-vlm-finetuning` | Parquet, tokenizer, tiny SigLIP, config, `training-*/`의 단계별 가중치·metrics·이미지 교환 결과 |
| Omni | `-omni` | `metrics.json`(손실·gradient·cache 오차·원본 소스 해시) |

자동 생성되는 `__marimo__/` 세션 캐시는 버전 관리에서 제외합니다. 보관하려면 해당 `outputs/notebooks/<실행 ID>/`로 옮기세요.

## 공통 설정과 주의점

- 기본 언어 모델은 은닉 차원 32, 2층, Q head 4개, KV head 2개인 Dense 모델입니다. 작은 CPU 행렬을 위해 PyTorch 스레드를 2개로 설정합니다.
- 대부분 실제 서브워드 토크나이저 대신 **공백 단위 단어 사전**을 사용해 토큰을 눈으로 읽을 수 있게 합니다.
  실제 MiniMind tokenizer(`<|im_start|>`·`<think>` template)는 SFT 5-1절, GRPO 1·3-1절, VLM 9절에서 데이터 경로 관찰에만 사용합니다.
- 원본 언어 모델은 `labels`를 내부에서 한 칸 이동합니다. Dataset에서 미리 이동하지 않습니다.
  예외: DPODataset·OmniDataset은 이미 이동한 입력/정답을 돌려주고, 증류·PPO·GRPO trainer는 모델 밖에서 직접 이동합니다.
- `apply_lora`는 정사각형 Linear에만 적용하고 원본 가중치를 동결하지 않습니다. **원본 복사 → 동결 → LoRA 적용** 순서를 지키고 적용 후 `deepcopy`하지 마세요.
- 원본 trainer의 AMP·DDP·wandb·torch.compile·중간 재개는 CPU 입문 실습에서 실행하지 않습니다. 학습률 등은 작은 모델에 맞춰 원본 기본값과 다르게 설정했습니다.
- 모든 결과는 작은 합성 데이터의 관찰이며 한국어 일반화, 이미지 인식, 음성 품질, 대규모 학습 성능을 평가하지 않습니다.

원본 클래스와 함수를 직접 사용하며 원본 파일은 수정하지 않습니다. 저작권·라이선스는
[minimind/LICENSE](../../projects/minimind/LICENSE), [minimind-v/LICENSE](../../projects/minimind-v/LICENSE),
[minimind-o/LICENSE](../../projects/minimind-o/LICENSE)를 확인하세요.
