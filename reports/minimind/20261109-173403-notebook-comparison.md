# GPT 원본과 MiniMind 생성 노트북의 내용 비교

검토 ID: 20261109-173403-notebook-comparison. 검토 대상: `gpt/` 원본 9개, `notebooks/` 생성본 10개 및 공통 helper·MiniMind 관련 구현. 모든 노트북의 코드·Markdown 셀을 읽고 비교했다. 셀 번호는 **0부터 시작하는 JSON cells 인덱스**다.

## 결론

생성본은 MiniMind의 기본 연산과 학습 절차를 이해하는 입문 과정으로 구성되어 있지만, 원본의 모든 실습을 옮긴 것은 아니다. 특히 **별도 테스트 질문 평가, 분류 백본의 부분 미세 조정, 직접 작성하는 생성 루프, 슬라이딩 윈도 데이터 구성, 동일 분류 문제의 LoRA 실험**이 빠졌거나 과제로만 남았다. 이전의 실행 통과 결과는 기본 코드 동작의 확인이며 원본 대비 내용 완전성을 의미하지 않는다.

이번 요청은 비교 검토이므로 노트북·helper·uv 설정을 수정하거나 학습을 다시 실행하지 않았다. 아래 보강안은 제안이며 구현 완료 항목이 아니다.

## 먼저 보강할 내용

### 1. 별도 테스트 질문에 대한 SFT·DPO 평가가 빠졌다 — 우선순위 높음

**원본 근거**

- [원본 SFT](../../notebooks/gpt/Chapter_7_Exercise_Follow_Instructions.ipynb) 셀 0의 `main`: train/test/valid를 분리하고, 학습 후 모든 test entry에 `model_response`를 생성하여 JSON으로 저장한다.
- [원본 DPO](../../notebooks/gpt/Chapter_7_Exercise_Follow_Instructions_dpo.ipynb) 셀 0의 `evaluate_dpo_loss_loader`: 별도 validation의 loss와 chosen/rejected reward를 계산한다. 셀 4는 test 질문에 대한 reference와 policy의 생성 결과를 나란히 비교한다.
- [원본 LoRA 분류](../../notebooks/gpt/Chapter_6_Excercise_Finetuning_Classification_LoRA.ipynb) 셀 3은 train/valid/test accuracy를 모두 계산한다. 일반 분류 원본은 test loader를 준비하지만 기본 실행 끝에서는 test accuracy를 별도로 출력하지 않으므로 두 원본의 범위를 혼동하면 안 된다.

**생성본 상태**

- [생성 SFT](../../notebooks/Chapter_7_Exercise_MiniMind_SFT.ipynb) 셀 4·6·8: `Say hello.`라는 **학습 질문**을 생성 전후 비교한다. valid CE는 계산하지만 독립 test set과 test 응답 저장은 없다. 새 질문 생성은 셀 11의 과제다.
- [생성 DPO](../../notebooks/Chapter_9_Exercise_MiniMind_DPO.ipynb) 셀 4·8: 같은 4쌍으로 학습하고 동일 4쌍의 loss/margin을 확인한다. `sft_valid_loss`는 보조적인 SFT 응답 CE이며 **별도 선호 검증 데이터의 DPO loss가 아니다**. test 질문에 대한 reference/policy 생성 비교도 없다.
- [생성 분류](../../notebooks/Chapter_6_Exercise_MiniMind_Classification.ipynb)는 train/valid만 있고 별도 test가 없다.

**영향과 보강안**

학습 목적함수의 개선과 새로운 입력에 대한 성능을 구별하는 실습이 부족하다. 작은 데이터 조건을 유지하면서 질문 단위 train/valid/test를 분리하고, SFT test 응답 JSON과 DPO reference/policy 비교표를 추가하는 것이 우선이다. DPO는 별도 선호 쌍의 loss·chosen/rejected reward·margin 곡선도 기록할 수 있다. 표본이 적다는 한계는 계속 명시해야 한다.

### 2. 분류에서 백본 일부를 실제로 미세 조정하는 과정이 과제로 축소됐다 — 우선순위 높음

**원본 근거**: [일반 분류 원본](../../notebooks/gpt/Chapter_6_Excercise_Finetuning_Classification.ipynb) 셀 1은 분류 head뿐 아니라 `trf_blocks[-1]`과 `final_norm`을 unfreeze하여 업데이트한다.

**생성본 상태**: [생성 분류](../../notebooks/Chapter_6_Exercise_MiniMind_Classification.ipynb) 셀 4에서 백본 전체를 고정하고, `features()`의 `torch.no_grad()` 안에서 hidden state를 미리 계산한다. 셀 6은 head만 학습한다. 마지막 블록 학습은 셀 7의 선택 과제다.

**영향과 보강안**: 현재 기본 실행은 동결된 특징 위의 분류 head 학습이다. 분류 목적에 맞춰 Transformer 표현을 바꾸는 과정은 체험하지 않는다. 마지막 MiniMind block과 최종 norm을 학습하는 두 번째 모드를 추가하고 head-only와 비교하는 것이 좋다. 이때 `requires_grad=True`만 바꾸면 부족하다. 미리 계산한 특징과 `no_grad()` 경로를 바꾸고 매 step 백본을 다시 forward해야 gradient가 흐른다.

### 3. 자기회귀 생성 루프를 직접 작성하는 셀이 빠졌다 — 우선순위 높음

**원본 근거**: [원본 모델 조립](../../notebooks/gpt/Chapter_4_Excercise_GPT.ipynb) 셀 4의 `generate_text_simple`, [원본 사전 학습](../../notebooks/gpt/Chapter_5_Excercise_Pretraining.ipynb) 셀 3의 `generate`는 마지막 위치 logits 선택 → argmax → ID 이어 붙이기를 직접 구현한다. 원본 5장에는 context cropping과 EOS 검사도 있다.

**생성본 상태**: [생성 모델](../../notebooks/Chapter_4_Exercise_MiniMind_Architecture.ipynb) 셀 8은 `model.generate(...)`를 호출한다. cache와 full forward의 일치 검사는 있지만 생성 루프 자체는 원본 모델 API 안에 있다. [helper](../../notebooks/minimind_lab.py)의 `respond()`도 동일하다.

**영향과 보강안**: 다음 토큰 예측을 반복해 문장이 되는 과정을 API 호출로 지나간다. cache 없는 5~10줄의 greedy 생성 루프를 먼저 만들고 매 step 선택한 ID·문자열·입력 길이를 출력한 뒤 MiniMind `generate`와 결과를 비교하면 좋다. GPT의 context cropping을 RoPE+cache에 그대로 복사하지 말고 MiniMind의 지원 길이와 위치 의미를 별도로 설명해야 한다.

### 4. 슬라이딩 윈도·stride와 DataLoader 구성 실습이 빠졌다 — 우선순위 높음

**원본 근거**: [원본 데이터](../../notebooks/gpt/Chapter_2_Exercise_Dataset.ipynb) 셀 2~4는 `GPTDatasetV1`에서 긴 텍스트를 `max_length`, `stride`로 자르고, `__len__`, `__getitem__`, DataLoader 설정과 첫 두 배치를 직접 확인한다.

**생성본 상태**: [생성 데이터](../../notebooks/Chapter_2_Exercise_MiniMind_Dataset.ipynb) 셀 4는 helper의 `pretrain_data()`·`loader()`를 호출한다. 실제 `PretrainDataset`을 사용하는 장점은 있으나, 긴 원문이 여러 학습 샘플로 바뀌는 과정을 만들지는 않는다. helper의 loader는 batch_size=4, num_workers=0을 고정하며 나머지 기본값 설명도 제한적이다.

**영향과 보강안**: 데이터 준비가 이미 완성된 기능처럼 보일 수 있다. 짧은 ID 배열을 stride=1과 stride=window로 잘라 샘플 수·중복을 비교하는 독립 셀과 `shuffle/drop_last`의 차이를 넣으면 좋다. **MiniMind 현재 데이터셋의 동작 자체를 sliding window라고 설명해서는 안 된다.** 현재 MiniMind는 JSONL 레코드를 길이 제한하는 방식이며, GPT 예제의 이동된 y를 그대로 MiniMind labels에 전달하면 이중 shift가 된다.

### 5. 같은 문제에서 분류 미세 조정과 LoRA를 비교하는 연결이 사라졌다 — 우선순위 중간

**원본 근거**: [일반 분류 원본](../../notebooks/gpt/Chapter_6_Excercise_Finetuning_Classification.ipynb)과 [LoRA 분류 원본](../../notebooks/gpt/Chapter_6_Excercise_Finetuning_Classification_LoRA.ipynb)은 같은 SMS spam 과제와 데이터 전처리를 사용한다. LoRA 원본은 어댑터 클래스를 직접 정의한다.

**생성본 상태**: 6장은 감정 단어 분류, [8장](../../notebooks/Chapter_8_Exercise_MiniMind_LoRA.ipynb)은 SFT 기반 브랜딩 응답 생성이다. 어댑터 구현도 직접 작성하는 대신 실제 `model_lora.py`를 가져온다. 따라서 파라미터 수는 관찰할 수 있지만 같은 문제의 일반 미세 조정과 LoRA 결과를 직접 비교하지 못한다.

**보강안**: MiniMind trainer에 맞춘 생성 LoRA는 유지하되, 같은 적응 대화 데이터로 full FT와 LoRA를 각각 실행해 업데이트 파라미터 수·loss·별도 질문 응답을 비교한다. 분류 LoRA를 추가한다면 비정방 분류 head는 원본 `apply_lora` 대상이 아니므로 head의 학습 여부를 명시해야 한다. `B(Ax)`를 작은 행렬로 직접 계산해 모듈 출력과 비교하는 셀도 원본의 구현 학습을 보완한다.

## 그 밖에 축소되거나 생략된 내용

| 항목 | 원본 위치 | 생성본의 상태 | 권장 보강 |
|---|---|---|---|
| 학습된 단어 벡터의 의미적 이웃·outlier·유추/군집 탐색 | 원본 1장 셀 2 및 외부 데모 안내 | 생성 1장은 랜덤 임베딩의 cosine과 행 업데이트만 확인. 의미 관계는 학습하지 않음 | 학습된 로컬 벡터가 있을 때만 선택 실습. 랜덤 벡터를 의미 유사성의 증거로 쓰면 안 됨 |
| 여러 tokenizer의 실제 비교 | 원본 2장 셀 9 Qwen/EXAONE | 생성 1장은 하나의 MiniMind tokenizer로 한·영 비교 | 선택 tokenizer 또는 미리 저장한 비교 결과로 길이·조각 차이 관찰. 언어별 우열을 일반화하지 않음 |
| HF 모델 파일의 역할 | 원본 2장 셀 10 | 생성본은 checkpoint·tokenizer 해시를 설명하지만 config/weights/generation_config/vocab/merges 역할표는 없음 | 2장에 작은 로컬 파일 목록과 역할표 추가. 실제 없는 파일은 없다고 표시 |
| single-head → 여러 독립 head → vectorized multi-head 순서 | 원본 3장 셀 0~3 | 생성 3장은 바로 4 Q heads/2 KV heads의 GQA 수동 계산으로 진입 | head 하나를 먼저 계산한 후 concat하고 batched 계산과 비교. Q/K/V·mask 수식 자체는 이미 유지되어 있음 |
| Transformer block을 직접 조립하기 | 원본 4장 셀 3 | 생성 4장은 원본 모델 import, hook shape, norm/FFN 수식 검증 | 기존 MiniMindBlock의 residual 두 단계를 작은 셀로 재현해 실제 block 출력과 비교 |
| 사전 학습 중간 생성·tokens_seen/global_step | 원본 5장 셀 2·4 | 생성 5장은 epoch loss와 마지막 생성. 중간 생성은 7장 SFT에 있음. 처리 토큰 축 없음 | 중간 생성과 유효/패딩 포함/감독 토큰 수를 구분해 기록 |
| 분류 데이터 불균형·train 길이 기준 처리·학습 곡선 | 원본 일반 분류 셀 0~1 | 생성 6장은 이미 균형인 8개 단어, train/valid 마지막 정확도만 출력 | 작은 불균형 예제와 분류 train/valid loss·accuracy 곡선 추가 |
| 외부 사전 학습 가중치로 전이 학습 | 원본 분류·SFT의 `load_gpt2_model` | 생성본은 5장에서 8문장으로 직접 학습한 작은 모델만 사용 | CPU 기본 실행을 유지하고, 호환되는 로컬 공개 가중치를 읽는 선택 경로를 별도로 제공. 파일 존재·설정 일치를 먼저 검사 |

위 항목은 같은 중요도가 아니다. 외부 벡터·tokenizer·공개 가중치는 오프라인·경량 실행을 위한 범위 축소로 볼 수 있다. 반면 직접 생성·데이터 샘플 구성·별도 평가·백본 업데이트는 외부 다운로드 없이도 작은 예제로 보강할 수 있다.

## 누락으로 분류하면 안 되는 MiniMind 전환 차이

### 모델 구조 변경

- GPT의 학습된 절대 위치 임베딩 → MiniMind RoPE.
- LayerNorm → RMSNorm, GELU FFN → SwiGLU, 일반 MHA → GQA.
- 가중치 공유·KV cache·MoE는 생성본에서 추가되거나 더 명확하게 검증하는 내용이다.

GPT의 원래 구성 요소를 그대로 복원할 필요는 없다. 구조 비교표와 block 연산 재현을 보강하는 편이 MiniMind 학습 목적에 맞다.

### SFT의 학습 대상이 원본과 다름

[원본 SFT](../../notebooks/gpt/Chapter_7_Exercise_Follow_Instructions.ipynb)의 `custom_collate_fn`은 padding을 제외하지만 기본 경로에서 prompt 전체를 응답처럼 별도로 -100 처리하지 않는다. 따라서 프롬프트를 포함한 sequence의 다음 토큰 loss를 계산한다. 생성본은 MiniMind `SFTDataset`을 사용해 **assistant 응답과 종료 구분자**만 감독한다.

생성본의 assistant-only mask는 누락이 아니라 MiniMind 방식으로의 변경이다. 다만 두 방식의 loss 값은 같은 대상으로 계산한 값이 아니므로 직접 비교할 수 없음을 예제에 명시하는 것이 좋다.

### DPO의 로그 확률 집계가 원본과 다름

[원본 DPO](../../notebooks/gpt/Chapter_7_Exercise_Follow_Instructions_dpo.ipynb) 셀 0의 `compute_logprobs`는 선택된 응답 토큰 log probability를 **평균**한다. MiniMind `trainer/train_dpo.py::dpo_loss`와 생성 9장은 마스크된 토큰 log probability를 **합산**한다.

생성본은 MiniMind 구현에 맞다. 다만 길이가 다른 chosen/rejected에서 평균과 합산은 달라지고 margin 및 beta의 수치적 척도도 달라진다. 예를 들어 token log probability가 모두 -1이면 길이 2/4의 합은 -2/-4지만 평균은 둘 다 -1이다. 원본과 같은 결과라고 설명해서는 안 된다. 길이가 다른 작은 응답으로 두 집계를 비교하는 설명 셀이 유용하다.

또한 원본 DPO는 주석에 SFT 모델이라고 적지만 실제 기본 코드는 `gpt2-medium-355M.pth`를 로드한다. 이번 비교에서는 그 파일의 실제 학습 이력을 검증하지 않았다. 생성본이 7장의 명시적 `sft.pt`를 사용하는 연결은 오히려 더 분명하다.

### LoRA 대상과 scaling이 원본과 다름

원본은 모든 Linear를 재귀적으로 감싸고 `(alpha/rank)·xAB`를 더한다. MiniMind는 입력/출력 차원이 같은 Linear에만 붙이고 `B(Ax)`를 그대로 더한다. 생성 8장은 이 차이를 설명하고 적용 대상을 출력한다. 현재 설정에서 q_proj/o_proj가 대상인 것은 k_proj/v_proj를 빼먹은 오류가 아니다.

### 원본 주석의 기능과 실제 구현을 구별해야 함

원본 5장 셀 3의 `generate`는 주석에 temperature/top-k를 설명하지만 함수 인자와 본문은 greedy 생성이다. 생성본에서 temperature/top-k 실험이 기본 셀에 없다는 것은 확인할 수 있으나, **원본 5장이 이미 그 비교 실습을 실행했다고 보아서는 안 된다.** `previous_chapters.py`의 별도 generate에는 관련 기능이 있지만 노트북 기본 호출의 사용 범위와 구분해야 한다.

## 장별 대응 요약

| 원본 | 생성본 | 판정 |
|---|---|---|
| 1장 Vector Space | 생성 1장 | 토큰/벡터 기초로 재구성. 학습된 의미 공간 탐색은 빠짐 |
| 2장 Dataset | 생성 1~2장 | shift·마스킹은 강화. sliding window·tokenizer 비교·파일 구조는 축소 |
| 3장 Attention | 생성 3장 | 핵심 수동 연산 유지 및 GQA/RoPE 강화. single-head부터의 단계적 구현은 축소 |
| 4장 GPT | 생성 4장 | MiniMind 구조로 전환 및 cache/MoE 추가. 직접 block 조립·생성 루프는 축소 |
| 5장 Pretraining | 생성 5장 | 학습·검증·저장 유지 및 optimizer 원리 보강. step/token 추적·중간 생성은 축소 |
| 6장 Classification | 생성 6장 | head 학습만 기본 실행. 실제 마지막 블록/norm fine-tuning은 과제로 축소 |
| 6장 LoRA Classification | 생성 8장 | 생성 과제로 변경. 같은 분류 문제의 비교·test 평가·직접 adapter 구현은 빠짐 |
| 7장 Instructions | 생성 7장 | MiniMind assistant-only SFT로 전환. test 질문 생성·응답 파일 저장은 빠짐 |
| 7장 DPO | 생성 9장 | 핵심 loss·reference 고정 유지. 별도 선호 검증·test 생성·reward 곡선은 빠짐 |
| 원본 대응 없음 | 생성 10장 | GRPO 숫자 연산의 선택 확장 |

## 보강 순서 제안

1. 7·9장에 별도 test 질문과 생성 결과 저장, 9장에 held-out 선호 평가를 추가한다.
2. 6장에 head-only와 마지막 block/norm fine-tuning의 두 모드를 추가한다.
3. 4장에 직접 greedy 생성 루프와 residual block 재현을 추가한다.
4. 2장에 window/stride·DataLoader 실험, 모델 파일 역할표를 추가한다.
5. 8장에 같은 데이터·같은 시작점에서 full FT와 LoRA를 비교하는 실습을 추가한다.
6. 의미 공간·다중 tokenizer·공개 가중치 실습은 별도의 선택 확장으로 둔다.

## 근거·명령·검증 범위

작업 디렉토리: 저장소 루트. 외부 웹사이트·모델·데이터를 내려받지 않았다. Python 표준 라이브러리로 `.ipynb` JSON을 읽고 코드의 함수 정의·호출·데이터 분할·mask·평가 경로를 확인했다. `rg`로 원본 helper와 MiniMind 구현을 대조했다. 학습 실행은 하지 않았으며 이전 실행 결과를 이번 검토의 새 통과 결과로 보고하지 않는다.

주요 읽기 명령 유형:

```bash
rg --files gpt notebooks -g '*.ipynb' -g '*.py' -g 'README.md' -g '!**/.venv/**'
# Python: json.loads(path.read_text())로 각 셀을 읽고 ast.parse로 함수 정의를 확인
# MiniMind model_lora.py, trainer/train_dpo.py 및 생성 helper를 함께 비교
```

별도 패키지 설치나 uv lock 변경은 없었다. Python 버전 등 검토 환경과 파일 SHA-256, 셀별 함수·제목 목록은 [scope.json](../../outputs/minimind/20261109-173403-notebook-comparison/scope.json), [inventory.json](../../outputs/minimind/20261109-173403-notebook-comparison/inventory.json)에 보관했다. 외부 벡터 데모의 현재 가용성, 원본의 전체 실행 성공, 공개 가중치의 실제 학습 이력, 제안한 보강안의 실행 결과는 이번 범위에 포함되지 않는다.
