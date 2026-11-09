# MiniMind 실습 — marimo

VLM 실습은 **모델 구조 → SFT → [시각·언어 모델](minimind_vlm.py)** 순서로 읽는 것을 권장합니다. 작은 실제 SigLIP과 MiniMindVLM을 무작위 초기화하여 CPU에서 실행합니다. 그림과 텐서 표로 이미지 패치부터 생성까지 확인하고 실제 tokenizer·VLMDataset의 답변 마스킹도 관찰합니다. 실행마다 `outputs/notebooks/<시각>-<고유 ID>-vlm/`에 작은 인코더, 이미지·대화 Parquet, 데이터 캐시와 `summary.json`을 저장합니다. 공개 체크포인트의 이미지 인식 성능은 평가하지 않습니다.

모델 구조, 사전 학습, SFT, FeedForward, MoE, LoRA, DPO, 지식 증류를 별도 marimo 노트북으로 제공합니다. 모두 한국어 설명과 작은 샘플 데이터를 포함하며, 각각 단독 실행할 수 있습니다.

| 노트북 | 내용 | 사용하는 원본 |
| --- | --- | --- |
| [시각·언어 모델(VLM)](minimind_vlm.py) | 패치 그림, 실제 SigLIP·Projector, 이미지 임베딩 교체, causal 연결, 답변 손실·gradient, 다중 이미지·KV cache·생성, 실제 VLMDataset | `minimind-v/model/model_vlm.py`, `minimind-v/dataset/lm_dataset.py` |
| [모델 구조와 학습](minimind_structure.py) | ID와 임베딩 행 비교, 한 head의 작은 Attention, MiniMind의 Q/K/V·RoPE·GQA, SwiGLU, 학습, logits·확률·손실, 한 토큰씩 생성, KV cache | `minimind/model/model_minimind.py` |
| [사전 학습](minimind_pretraining.py) | 원본 PretrainDataset, JSONL·절단·패딩·정답 이동, DataLoader, 누적·clipping·학습률, 검증·생성·체크포인트 | train_pretrain.py, lm_dataset.py, trainer_utils |
| [SFT 미세 조정](minimind_sft.py) | 원본 SFTDataset, 대화 template·assistant 마스킹·정답 이동, 기준 모델 준비와 Full SFT, 검증·생성·복원 | train_full_sft.py, lm_dataset.py |
| [FeedForward](minimind_feedforward.py) | SwiGLU 직접 계산, FFN과 블록의 문맥 전파 비교, Linear·TwoLinear·SiLU MLP·SwiGLU의 동일 회귀 과제 비교 | 원본 `FeedForward`, `MiniMindBlock` |
| [MoE](minimind_moe.py) | 고정 router·가중합, Dense/MoE 동일 입력 비교, scalar gradient, 선택 결과 기반 검증, 보조 손실 계수별 LM 사용량 비교 | 원본 `MOEFeedForward`, `MiniMindForCausalLM` |
| [LoRA](minimind_lora.py) | 기준 모델 준비, 단일 선형층의 A/B 계산·병합, 두 step의 gradient 관찰, 실제 모델 적용·학습, 저장·복원·파일 크기 비교 | `minimind/model/model_minimind.py`, `minimind/model/model_lora.py` |
| [지식 증류](minimind_distillation.py) | hard/soft target, 온도·수동 KL, 실제 SFTDataset 마스크·정답 이동, 교사 동결, CE/CE+KL 학생 비교·확률·생성 | `train_distillation.py`, `lm_dataset.py` |
| [GRPO fine tuning](minimind_grpo.py) | 실제 RLAIFDataset, 그룹 보상·advantage, logps·EOS 마스크, GRPO/CISPO, 작은 MiniMind의 CPU 정책 학습 | `train_grpo.py`, `lm_dataset.py`, `model_minimind.py` |
| [DPO](minimind_dpo.py) | 실제 DPODataset·응답 마스크·정답 이동, 로그 확률·참조 대비 선호 차이, β와 gradient, 작은 SFT 준비·DPO 학습·검증·생성 | `lm_dataset.py`, `train_dpo.py` |
| [PPO fine tuning](minimind_ppo.py) | 실제 RLAIFDataset, rollout·EOS 마스크, Critic·GAE 손계산, policy/value clip·두 KL 지표, CPU PPO 학습·저장·복원 | `train_ppo.py`, `lm_dataset.py` |

모델 구조 실습을 먼저 읽고 LoRA 실습을 진행하는 순서를 권장합니다. LoRA 노트북은 네 문장으로 기준 모델을 직접 준비하므로, 모델 노트북을 실행하거나 체크포인트를 저장할 필요가 없습니다. 기준 모델을 만드는 전체 매개변수 학습과 이후의 LoRA 학습을 구분해 설명합니다.

구조를 더 살펴보려면 **모델 구조 → FeedForward → MoE** 순서로 진행하세요. FFN/MoE 노트북은 `model_minimind.py`의 원본 클래스를 직접 사용합니다. FeedForward는 입력 4차원의 작은 특징 변환과 별도 검증 데이터를 사용하고, MoE는 전문가 3개와 토큰 6개에서 직접 계산한 뒤 작은 토큰 ID 언어 모델을 학습합니다.

학습 파이프라인은 **모델 구조 → 사전 학습 → SFT → LoRA** 순서로 살펴보세요. 사전 학습 노트북은 실제 `PretrainDataset`으로 학습 8개·검증 4개의 로컬 JSONL을 읽고 작은 단어 토크나이저를 직접 만듭니다. 원본 trainer의 CPU 흐름을 축소해 microstep과 optimizer step, 불완전 누적 그룹, 유효 토큰 평균 손실과 perplexity를 구분합니다. 원본의 `loss/accumulation_steps`를 유지하므로 길이가 다른 microbatch와 마지막 작은 그룹의 평균 방식도 설명합니다. 원본 trainer 전체, AMP·DDP·wandb는 실행하지 않습니다.

지식 증류는 **모델 구조 → 사전 학습 → SFT → 지식 증류** 순서를 권장하며 단독 실행도 가능합니다. 원본 `distillation_loss`와 `SFTDataset`을 직접 사용해 온도·수동 KL·다음 토큰 정렬·두 턴의 응답 마스크를 확인합니다. CPU에서 작은 Dense 교사를 준비하고 같은 초기 학생을 CE/CE+KL로 비교합니다. 온도·CE 비중 UI, 절단 관찰, 예측 질문·해설·과제를 제공합니다. 원본의 SFT 체크포인트/MoE 교사 대신 무작위 초기화와 Dense 교사를 사용하며, 4개 학습 대화에서 관찰하므로 일반화 성능 평가는 포함하지 않습니다.

## 수강생 실습 방법

Python 함수, 텐서 shape와 행렬 곱(`@`)을 알고 시작합니다. 각 절의 **예측 → 실행 → 관찰 → 해설**을 따라 먼저 예상 결과를 적고 실행 결과와 비교하세요. `@app.cell`과 셀 함수의 인자·return은 marimo의 셀 간 데이터 전달이며 모델 자체의 구조가 아닙니다. `_`로 시작하는 변수는 셀 내부의 중간 계산에 씁니다.

모델 실습은 사람이 정한 Q/K/V의 작은 계산에서 시작하여 실제 MiniMind로 확장합니다. 학습 전후에는 같은 토큰과 head의 Attention 그림·표, 다음 토큰의 확률을 비교하고, greedy 생성의 각 반복 입력을 확인합니다. 한글 폰트가 없는 환경에서는 Attention 그림 축에 위치 번호를 표시하며 표에서 실제 토큰과 연결합니다.

LoRA 실습은 작은 직사각형 Linear에서 분기 계산을 먼저 확인합니다. 그다음 첫 backward에서 A의 gradient가 0이고 B가 갱신된 뒤 A도 학습되는 모습을 관찰합니다. 이후 실제 모델의 적용 층, 학습 대상 수, 생성 결과, 저장 크기를 비교하세요. 모델 전체 보유량·학습 대상·어댑터 파일은 서로 다른 값입니다. 학습 대상 비율로 전체 메모리 절감률을 계산하지 않습니다.

마지막 과제에는 기록할 지표와 힌트·해설을 넣었습니다. 손실 감소와 문장별 정답 수를 따로 기록하고, 작은 데이터에서 모든 rank가 정답이어도 학습 대상 수와 파일 크기를 비교하세요.

FeedForward에서는 중간 차원과 학습 횟수를, MoE에서는 토큰당 전문가 수와 결합 가중치 정규화를 바꿀 수 있습니다. FFN의 SiLU gate는 특징별 곱셈이고, MoE의 softmax router는 전문가 선택이라는 차이를 확인하세요. MoE의 다음 토큰 주 손실과 이미 계수가 반영된 보조 손실은 따로 반환되므로 학습 셀에서 한 번 합산합니다.

FFN의 구조 비교는 같은 데이터·step·optimizer를 사용합니다. TwoLinear와 SiLU MLP의 up/down은 SwiGLU와 같은 초기 가중치에서 시작하지만 모델별 매개변수 수는 다릅니다. 고정 shape 계산의 M=8과 학습용 M UI는 별도 실험입니다. FFN 단독과 Attention 포함 블록의 출력 변화 표로 “위치를 섞지 않는 연산”과 “문맥이 없는 입력”을 구분하세요.

MoE의 **고정 router 관찰(1~3절)**과 **LM 재학습(5절)**은 UI가 분리되어 있습니다. 앞 절 정규화 변경이 뒤 LM을 재학습시키지 않습니다. 4절 부호 UI는 실제 선택 전문가에 따라 검증하므로 +1/-1 모두 실습할 수 있습니다. LM은 동일 초기 가중치를 복사하여 보조 손실 계수 0/0.01을 각각 학습하고, 전문가별 선택 횟수·평균 확률의 전후 표와 그래프를 보여 줍니다. 계수 0의 aux가 0인 것과 균등 선택은 다르므로, 계수 없는 균형 항도 함께 읽으세요. 선택 횟수는 EOS 포함 입력 8위치, 주 손실은 다음 정답이 있는 6위치를 사용합니다.

## 실행

Python 3.12와 [uv](https://docs.astral.sh/uv/)가 필요합니다. 저장소 루트에서:

```bash
cd notebooks/minimind
uv sync --locked

# 모델 구조와 학습
uv run marimo edit minimind_structure.py

# LoRA (다른 터미널에서 실행하거나 앞 편집기를 종료한 뒤 실행)
uv run marimo edit minimind_lora.py

uv run marimo edit minimind_pretraining.py
uv run marimo edit minimind_distillation.py
uv run marimo edit minimind_grpo.py
uv run marimo edit minimind_dpo.py
uv run marimo edit minimind_sft.py
uv run marimo edit minimind_vlm.py

# FeedForward / MoE도 각각 편집기로 실행
uv run marimo edit minimind_feedforward.py
uv run marimo edit minimind_moe.py
```

일반 Jupyter 커널 대신 marimo 편집기에서 설명과 코드를 함께 봅니다. 최초 의존성 설치에는 인터넷이 필요하지만, 실습은 모델·데이터 다운로드나 GPU 없이 CPU로 실행됩니다. 저장소 디렉토리 배치를 유지해 주세요. Linux/Windows에는 CPU용 PyTorch를 설치하고 macOS에서는 PyPI 빌드를 사용합니다. 의존성은 `notebooks/minimind/.venv`와 `uv.lock`으로 관리합니다.

VS Code marimo 확장에서 두 번째 도입 셀을 텍스트 셀로 표시하려면 `mo.md(r"""...""")`처럼 정적 문자열을 사용합니다. `mo.md(f"""...{값}...""")`는 동적 Python 표현식이므로 확장에서 코드 셀로 표시합니다. 도입 셀에는 공통 산출물 경로를 안내하고 실제 실행 경로는 뒤의 결과 셀에서 확인합니다. `hide_code=True`로 코드를 숨기는 설정과 Markdown 셀 분류는 별개입니다.

서버 없이 전체 셀을 실행하려면 같은 `notebooks/minimind/` 디렉토리에서:

```bash
uv run marimo check minimind_structure.py minimind_lora.py minimind_feedforward.py minimind_moe.py minimind_pretraining.py minimind_distillation.py minimind_dpo.py minimind_sft.py
uv run python minimind_structure.py
uv run python minimind_pretraining.py
uv run python minimind_distillation.py
uv run python minimind_grpo.py
uv run python minimind_dpo.py
uv run python minimind_sft.py
uv run python minimind_lora.py
uv run python minimind_feedforward.py
uv run python minimind_moe.py
```

결과를 HTML로 보관하려면:

```bash
RUN_ID="$(date +%Y%m%d-%H%M%S)-marimo"
mkdir -p "../../outputs/notebooks/$RUN_ID"
uv run marimo export html minimind_structure.py \
  -o "../../outputs/notebooks/$RUN_ID/minimind_structure.html"
uv run marimo export html minimind_pretraining.py \
  -o "../../outputs/notebooks/$RUN_ID/minimind_pretraining.html"
uv run marimo export html minimind_distillation.py \
  -o "../../outputs/notebooks/$RUN_ID/minimind_distillation.html"
uv run marimo export html minimind_sft.py \
  -o "../../outputs/notebooks/$RUN_ID/minimind_sft.html"
uv run marimo export html minimind_dpo.py \
  -o "../../outputs/notebooks/$RUN_ID/minimind_dpo.html"
uv run marimo export html minimind_lora.py \
  -o "../../outputs/notebooks/$RUN_ID/minimind_lora.html"
uv run marimo export html minimind_feedforward.py \
  -o "../../outputs/notebooks/$RUN_ID/minimind_feedforward.html"
uv run marimo export html minimind_moe.py \
  -o "../../outputs/notebooks/$RUN_ID/minimind_moe.html"
```

HTML 내보내기는 모든 셀을 실제 실행합니다. [marimo 스크립트 실행](https://docs.marimo.io/guides/scripts/), [HTML 내보내기](https://docs.marimo.io/guides/exporting/static_html/), [uv의 CPU용 PyTorch 설정](https://docs.astral.sh/uv/guides/integration/pytorch/)을 참고하세요. 자동 생성되는 `__marimo__/` 세션 캐시는 버전 관리에서 제외하며, 보관할 때는 실행 종료 후 해당 `outputs/notebooks/<실행 ID>/`로 옮깁니다.

사전 학습 노트북은 실행할 때마다 `outputs/notebooks/<시각>-<고유 ID>-pretraining/`을 만들어 원문 JSONL, tokenizer, datasets 캐시를 저장합니다. 학습 설정을 바꿀 때마다 그 안에 별도 `training-<고유 ID>/`를 만들어 loss·생성 결과·체크포인트를 남깁니다. 저장·복원 검증은 optimizer 갱신 경계의 같은 추가 배치를 비교하며, 원본 trainer의 DataLoader 중간 재개까지 검증하는 것은 아닙니다.

## 실습 설정과 주의점

- 기본 모델은 은닉 차원 32, 2층, Q head 4개, KV head 2개인 Dense 모델입니다. CPU의 작은 행렬 연산을 위해 PyTorch 스레드를 2개로 설정합니다.
- 모델 노트북에서는 전체 학습 횟수를, LoRA 노트북에서는 rank를 화면에서 바꿀 수 있습니다. 각 노트북 마지막에 별도의 수정 과제가 있습니다.
- 실제 서브워드 토크나이저 대신 공백 기반 단어 사전을 사용합니다. 사전에 없는 단어를 쓰려면 샘플 데이터에 추가하세요.
- `labels`는 입력과 같은 위치를 유지합니다. 모델 내부에서 다음 토큰으로 이동하며 패딩 정답 `-100`을 무시합니다.
- `apply_lora`는 정사각형 Linear만 수정하고, 원본 가중치는 별도로 동결해야 합니다. 기본 설정은 Q/O에 적용됩니다. 설정 변경 후 실제 적용 층 표를 확인하세요.
- `apply_lora`는 원래 층을 참조하는 함수로 `forward`를 바꿉니다. 적용 후 `deepcopy`하지 않고 **원본 복사 → 동결 → LoRA 적용** 순서를 지킵니다.
- 어댑터 저장·병합은 원본 함수의 float16 저장으로 작은 오차가 생깁니다. 검증용 파일은 임시 디렉토리에 저장한 뒤 정리합니다. 영구 산출물은 `outputs/notebooks/<실행 ID>/`에 보관하세요.
- 모델·LoRA는 네 문장 암기 실험이며 한국어 일반화 성능은 측정하지 않습니다. FeedForward는 숫자 회귀의 학습·검증 분할을 사용하고, MoE는 작은 특징과 토큰 ID를 사용합니다. 대규모 사전 학습·SFT·RL, GPU 분산 MoE와 속도 비교는 포함하지 않습니다.

원본 클래스와 함수를 직접 사용하며, 저작권·라이선스 안내는 [minimind/LICENSE](../../projects/minimind/LICENSE)를 확인하세요. 기존 통합 파일 `minimind_structure_lora.py`는 두 파일로 대체했습니다.

## SFT 실습

`minimind_sft.py`는 원본 `SFTDataset`으로 학습 대화 8개와 검증 질문 4개를 읽습니다. 학습 사전으로 만든 로컬 tokenizer와 chat template을 사용하고, 여러 턴에서 assistant 본문과 종료 토큰만 정답으로 남는지 표로 확인합니다. 질문도 문맥으로 입력되며, labels의 `-100`은 손실 제외 표시입니다. 정답 이동은 원본 모델 내부에서 한 번만 수행합니다.

일반 문장 4개로 기준 모델을 100 step 준비한 뒤 복사하여 전체 매개변수 SFT를 실행합니다. 학습·검증 응답 토큰 CE, greedy 응답 일치와 EOS 종료를 비교하고 가중치 저장·복원도 확인합니다. epoch와 누적 수를 UI에서 바꿀 수 있습니다. 학습 전 길이 절단과 정답 0개, 종료 없는 발화 뒤 PAD가 정답에 포함될 수 있는 원본 동작도 살펴봅니다.

각 실행의 데이터·tokenizer·기준 가중치는 `outputs/notebooks/<시각>-<고유 ID>-sft/`, 설정별 기록·생성·SFT 가중치는 그 아래 `training-<고유 ID>/`에 저장됩니다. CPU float32 실습으로 AMP·DDP·원본 trainer 전체·중간 재개는 실행하지 않습니다. 작은 동물 소리 과제의 새 질문 조합 평가이며 범용 한국어 성능을 측정하지 않습니다. 원본 trainer와 데이터 코드는 수정하지 않습니다.

DPO는 SFT·다음 토큰 예측을 이해한 뒤 진행하세요. 실제 `DPODataset`과 원본 손실 함수를 사용하며, 공손한 응답을 선호하는 합성 6쌍으로 학습하고 2쌍으로 검증합니다. 두 응답을 모두 학습한 작은 SFT 기준 모델을 준비한 뒤 policy/ref로 복사하므로 별도 가중치가 필요 없습니다. 숫자 실험의 Δ·β와 실제 DPO의 β·step UI는 분리되어 있습니다. 응답별 로그 확률 **합**, 참조 대비 격차, raw gap과 greedy 생성을 구분하세요. 실행 산출물은 `outputs/notebooks/<시각>-<고유 ID>-dpo/`, 설정별 가중치·config·지표는 그 안의 `training-<고유 ID>/`에 저장합니다. 기본값은 CPU SFT 100 step과 DPO 40 step이며 원본 trainer의 AMP·DDP·재개는 검증하지 않습니다.

## GRPO 실습

[GRPO fine tuning](minimind_grpo.py)은 SFT의 다음 토큰 학습을 알고 진행하는 실습입니다.
원본 `RLAIFDataset`과 저장소의 chat tokenizer로 마지막 assistant 응답이 입력에서 빠지는 것을 확인한 뒤,
숫자 예제로 그룹 advantage, 다음 토큰 logps 정렬, EOS 마스크와 GRPO/CISPO 손실을 계산합니다.
원본 trainer의 기본 `loss_type`은 `cispo`이므로 GRPO 분기를 따로 구분합니다.

마지막에는 별도 9개 토큰 사전과 작은 실제 MiniMind로 SFT 출발 정책을 준비하고,
두 산술 질문에 한 토큰 응답을 샘플링해 60 step GRPO를 수행합니다.
외부 보상 모델 대신 정답 일치 규칙을 사용하고 학습 전후 정답 확률과 기대 보상을 비교합니다.
모든 셀은 CPU에서 실행하며, 원본 trainer 전체·실제 대화 모델·GSM8K 일반화는 검증하지 않습니다.
실행마다 `outputs/notebooks/<시각>-<고유 ID>-grpo/`에 JSONL, datasets 캐시,
지표와 참조 trainer 소스 사본을 저장합니다. 파라미터를 바꾸면 학습 셀이 다시 실행됩니다.

```bash
# notebooks/minimind/에서 실행
uv run marimo export html minimind_grpo.py \
  -o ../../outputs/notebooks/<실행-ID>/minimind_grpo.html
```


## PPO 실습

[minimind_ppo.py](minimind_ppo.py)는 SFT를 이해한 뒤 진행합니다. 실제 `RLAIFDataset`에서 마지막 assistant 응답이 프롬프트에서 제외되는지 확인하고, 작은 SFT 모델을 준비하여 두 인사 중 공손한 응답을 보상으로 더 자주 선택하게 만듭니다. Actor·old policy·reference·Critic·reward의 역할, 다음 토큰 위치와 EOS 마스크, GAE·return·advantage 정규화, 부호별 clip gradient를 순서대로 계산합니다.

```bash
cd notebooks/minimind
uv run marimo edit minimind_ppo.py
# 전체 셀 실행
uv run python minimind_ppo.py
uv run marimo check minimind_ppo.py
```

CPU에서 다운로드 없이 실행합니다. rollout 횟수와 reference KL 계수를 바꿀 수 있으며, 같은 4개 질문에 대한 정확한 응답 경로 확률과 샘플 보상, 두 KL 지표, clipfrac, Critic loss를 비교합니다. 별도 일반화 평가는 아닙니다. 산출물은 `outputs/notebooks/<시각>-<고유 ID>-ppo/`, 설정별 지표·응답 trace·학습 곡선·Actor/Critic 체크포인트는 그 아래 `training-*/`에 저장합니다. 원본의 외부 보상 모델을 규칙 보상으로 대체하고 CPU 단일 프로세스 PPO를 직접 구현하므로 원본 trainer의 AMP·DDP·외부 rollout 엔진·중간 재개는 검증하지 않습니다.

지식 증류 노트북은 `outputs/notebooks/<시각>-<고유 ID>-distillation/`에 대화 JSONL·토크나이저·datasets 캐시를 저장합니다. 설정별 `students-<고유 ID>/metrics.json`에 CE·KL·정답률·확률·생성 토큰·곡선 원자료·소스 해시를 보관합니다. 모델 체크포인트는 저장하지 않습니다.


## Omni 실습

[minimind_omni.py](minimind_omni.py)는 `minimind-o/model/model_omni.py`의 실제 클래스로 Thinker–Talker와 멀티모달 입력을 살펴보는 한국어 실습입니다. 모델 구조 노트북 다음에 진행하세요. 텍스트·음향 9행 입력, projector와 placeholder 교체, 중간 bridge, 공유/개별 음향 embedding·head, 8개 코드북 지연 배치, 두 CE와 STOP 가중치, 역전파·projector 동결, KV cache, 스트리밍, 화자 조건과 VAD를 다룹니다. Thinker 전달 배율을 바꾸며 두 출력의 변화를 비교할 수 있습니다.

```bash
cd notebooks/minimind
uv sync --locked
uv run marimo edit minimind_omni.py
# 전체 셀 실행 및 정적 검사
uv run python minimind_omni.py
uv run marimo check minimind_omni.py
```

CPU·합성 텐서·작은 무작위 모델로 실행합니다. 원본의 외부 인코더 로딩만 생략하고, 음성·이미지 경로 실험에는 고정 숫자를 반환하는 교육용 인코더를 사용합니다. 원본 모듈 import에 필요한 `soundfile`, `librosa`, `onnxruntime`도 uv로 관리합니다. SenseVoice·SigLIP·Mimi 가중치와 실제 데이터는 다운로드하지 않으며, 인식·음성 합성 품질을 평가하지 않습니다. `OmniDataset`·원본 trainer 전체도 실행하지 않습니다. 실행마다 `outputs/notebooks/<시각>-<고유 ID>-omni/metrics.json`에 손실·gradient·cache 오차·원본 소스 해시를 저장합니다.

## Vision LLM 미세 조정 실습

[minimind_v_finetuning.py](minimind_v_finetuning.py)는 MiniMind-V의 실제 `VLMDataset`, `MiniMindVLM`, `init_vlm_model`을 이용해 **이미지·대화 → assistant 정답 마스킹 → Projector 정렬 → 시각 지시 SFT**를 연결합니다. 기존 SFT 실습 다음에 읽으세요.

```bash
cd notebooks/minimind
uv sync --locked
uv run marimo edit minimind_v_finetuning.py

# UI 없이 기본 설정으로 전체 셀 실행
uv run python minimind_v_finetuning.py
```

CPU·오프라인 실습입니다. 빨강/파랑 사각형 이미지와 Parquet를 직접 만들고 작은 SigLIP·3층 LLM을 무작위 초기화하므로 가중치나 데이터를 다운로드하지 않습니다. token별 label 표, 시각 특징 교체 흐름도, gradient와 동결 정책 표, 두 단계 loss 그래프를 확인합니다. 정렬과 SFT의 step 수는 UI에서 바꿀 수 있습니다. 실제 사전 학습 모델의 이미지 이해 성능을 재현하는 실습은 아닙니다.

학습 뒤에는 위치가 다른 검증 이미지에서 응답 생성·EOS 종료·정확 일치와 **정답을 유지한 이미지 교환 CE**를 비교합니다. Projector만 학습하는 정렬과 Projector·LLM 첫/마지막 층을 학습하는 SFT에서 실제 가중치 변화도 검사합니다. 원본 두 trainer의 루프를 CPU float32로 축소하며, AMP·DDP·MoE·원본 `train_epoch` 전체·중간 재개는 실행하지 않습니다. 실데이터로 확장할 때의 원본 기본값과 경로 제약은 마지막 절에 설명합니다.

각 실행은 `outputs/notebooks/<시각>-<고유 ID>-vlm/`에 Parquet·tokenizer·tiny SigLIP·config를 저장합니다. step 설정별 `training-<고유 ID>/`에는 단계별 가중치, `metrics.json`, `generation-and-image-swap.json`, `restore-check.json`이 생성됩니다. 새 marimo 프로세스로 실행하여 다른 프로젝트의 동명 model/dataset 패키지와 충돌하지 않게 하세요.
