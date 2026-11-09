# 수강생 관점의 학습 장애 보강과 실행 검증

실행 ID: `20261109-180606-student-revision`. [수강생 관점 검토](20261109-175707-student-review.md)에 따라 학습 원리·원본 코드 연결·과제 오류를 수정했다. 원본 GPT 노트북과 MiniMind 모델/trainer 소스는 수정하지 않았다. 기존 uv 프로젝트와 lock도 유지했다.

## 결과

노트북 **10개 / 코드 셀 68개**가 새 Jupyter 커널에서 전부 통과했다. 총 실행 시간은 55.89초다. 별도 GQA 과제 검사는 유효 설정 5개와 잘못된 설정 3개 모두 기대대로 통과했다. 이는 코드와 설명용 수치·불변성의 확인이며 실제 수강생의 이해도를 측정한 결과는 아니다.

배포 원본에는 출력과 실행 번호를 남기지 않았다. 실행 사본·로그·지표·JSONL·그림·가중치는 [outputs 실행 폴더](../../outputs/minimind/20261109-180606-student-revision)에 보관한다.

## 검토 항목별 반영

| 학습 장애 | 수정 |
|---|---|
| GQA 과제에서 KV heads=1/4가 reshape 오류 | Q/KV head 수·head_dim·batch·sequence·KV 반복 수를 config와 입력에서 유도한다. 수동 계산·미래 변경 검사도 같은 길이를 사용한다. |
| 설정이 helper 내부에 숨음 | 모든 노트북 setup에 MODEL_OPTIONS를 표시한다. make_config가 나눗셈·양수·짝수 head_dim을 검증하고 head_dim을 재계산한다. |
| wildcard import로 출처가 불명확 | helper import와 실제 MiniMind model/dataset import를 구분하고 B/T/D/V 및 도구 역할을 설명한다. |
| gradient/SGD 기초가 없음 | 1장에 scalar w·x·target 예제로 loss·gradient·학습률·업데이트를 숫자로 계산하고 해설을 제공한다. |
| logits·CE·학습 연결이 짧음 | 4장에 V=3의 softmax→정답 확률→-log(p)→CE→gradient→SGD 전후 확률을 확인한다. Attention softmax와 vocabulary softmax의 축을 구분한다. |
| RoPE가 norm 보존 검사뿐 | 3장에서 같은 회전/다른 회전의 내적을 비교하고, Q/K norm·GQA와 위치 차이의 역할을 설명한다. |
| 개념 사용/설명 순서가 어긋남 | 4장은 norm/FFN·residual·직접 생성 후 cache/MoE로, 5장은 기본 학습 후 누적·AMP·재개로 이동한다. |
| 문맥화 hidden state 과제가 너무 이름 | 1장 과제를 뒤로 연결하고 4장에서 같은 마지막 ID·같은 embedding·다른 문맥 hidden state를 직접 확인한다. |
| Dataset/helper 경로가 가려짐 | 2장에서 실제 JSONL 한 줄→Dataset[0]→DataLoader batch를 비교하고 원본 __getitem__을 읽는다. |
| 원본 trainer로 넘어가는 지도가 없음 | 5장에서 __main__→args/seed/config→init_model/Dataset/optimizer→resume/sampler→train_epoch→저장의 지도를 제공한다. AST로 원본 소스와 optimizer 생성 위치를 확인한다. |
| SFT/LoRA 원본 코드 연결 부족 | 7장·8장에 dataset·from_weight·requires_grad·optimizer 대상·apply_lora forward 교체를 찾는 안내와 해설을 추가한다. |
| Run All과 이해도 혼동 | 실행 전 예상값·실행 후 설명 문제·접이식 해설, 원본 함수에서 근거를 찾는 질문을 추가한다. |

scalar SGD 예제에서 loss=4, gradient=-8, learning rate=0.1, 새 w=1.8을 검증했다. logits 기초 예제는 처음 정답 확률 1/3과 CE≈1.0986을 확인하고 SGD 뒤 정답 확률이 증가함을 검증했다. 이 예제는 logits 자체를 Parameter로 둔 설명용이며 실제 모델은 logits를 만드는 가중치에 gradient가 전달됨을 명시했다.

## 전체 실행

작업 디렉토리: 저장소 루트. 실제 커널 작업 디렉토리: `/home/mind4u/dev/llm_playground/notebooks`.

```bash
notebooks/.venv/bin/python notebooks/scripts/validate_notebooks.py --run-id 20261109-180606-student-revision
notebooks/.venv/bin/python notebooks/scripts/check_attention_exercises.py --output-dir outputs/minimind/20261109-180606-student-revision/attention-exercises
```

첫 명령은 현재 uv Python의 새 커널로 각 장을 실행한다. 두 번째 명령은 **3장 실제 GQA 수동 계산·mask·causal 검사 셀**을 읽어 설정과 예제 입력 shape를 바꿔 실행한다. 별도 복제 구현만 검사한 것이 아니다.

| 장 | 주제 | 실행 셀 | 판정 | 시간(초) |
|---|---|---:|---|---:|
| 1 | 토큰과 임베딩 벡터 | 7 | 통과 | 4.86 |
| 2 | 데이터셋과 정답 마스킹 | 8 | 통과 | 3.54 |
| 3 | Causal Attention·GQA·RoPE | 6 | 통과 | 3.74 |
| 4 | MiniMind 모델 전체 흐름 | 11 | 통과 | 3.87 |
| 5 | 다음 토큰 예측 사전 학습 | 10 | 통과 | 5.61 |
| 6 | 분류를 위한 파인튜닝 | 4 | 통과 | 5.17 |
| 7 | 지시를 따르는 SFT | 7 | 통과 | 13.12 |
| 8 | LoRA로 적은 파라미터만 학습하기 | 6 | 통과 | 5.40 |
| 9 | 선호 응답 쌍으로 DPO 학습하기 | 6 | 통과 | 6.91 |
| 10 | 선택 확장: GRPO 손실 이해 | 3 | 통과 | 3.67 |

[status.json](../../outputs/minimind/20261109-180606-student-revision/validation/status.json)에 상태·환경이 있으며 [static-audit.json](../../outputs/minimind/20261109-180606-student-revision/validation/static-audit.json)에 schema·구문·상대 링크·빈 출력·노트북 해시가 있다. 실행 후 source 해시가 그대로인지, README 링크가 유효한지도 확인했다.

## 설정 변경 과제 회귀 결과

| hidden size | Q heads | KV heads | batch | T | 수동/원본 최대 오차 | causal 최대 오차 |
|---:|---:|---:|---:|---:|---:|---:|
| 64 | 4 | 1 | 1 | 6 | 0.0e+00 | 0.0e+00 |
| 64 | 4 | 2 | 1 | 6 | 0.0e+00 | 0.0e+00 |
| 64 | 4 | 4 | 1 | 6 | 0.0e+00 | 0.0e+00 |
| 96 | 6 | 3 | 2 | 7 | 0.0e+00 | 0.0e+00 |
| 128 | 8 | 2 | 2 | 9 | 0.0e+00 | 0.0e+00 |

추가로 hidden_size=64/Q heads=3, Q heads=4/KV heads=3, hidden_size=12/Q heads=4(head_dim=3)를 모두 ValueError와 설명 메시지로 거부했다. 회귀 결과는 [results.json](../../outputs/minimind/20261109-180606-student-revision/attention-exercises/results.json)에 있다. 단순히 기대한 에러가 났다는 것이 아니라 유효한 과제 설정은 실제 연산이 통과하고 잘못된 조합만 사전에 거부하는지 구분했다.

## 환경·리비전

Python: `3.12.15 (main, Oct 21 2026, 17:47:06) [Clang 23.1.3 ]`. 실행 경로: `/home/mind4u/dev/llm_playground/notebooks/minimind/.venv/bin/python`. 기본 CPU float32, hidden_size=64, layers=2, Q/KV heads=4/2, vocab=6400. 외부 가중치 없이 무작위 초기화와 고정 예제 JSONL로 학습했다.

- torch: 2.14.1
- transformers: 4.57.6
- datasets: 5.0.1
- numpy: 2.5.4
- matplotlib: 3.11.2
- ipykernel: 7.4.0
- jupyterlab: 4.6.4
- nbclient: 0.11.0
- nbformat: 5.11.1

모델·dataset·trainer 소스·tokenizer·helper·검증 스크립트·lock·실제 JSONL의 SHA-256과 git HEAD는 [input-manifest.json](../../outputs/minimind/20261109-180606-student-revision/validation/input-manifest.json)에 기록했다. 외부 모델·데이터 리비전은 사용하지 않았다.

## 관찰과 검증 한계

이번 SFT 별도 test 4개 중 종료 token/앞뒤 공백을 제거한 단순 exact match는 **1/4**, test CE는 0.4426였다. 코드 통과와 일반화 품질은 별개다. 셀 순서를 바꾸면 이전 워밍업 예제가 소비하던 난수 순서도 달라질 수 있으므로 이전 실행의 metric 값과 비트 단위 동일성을 요구하지 않는다.

원본 trainer 읽기 셀은 파일을 AST/inspect로 읽어 출력할 뿐 원본 CLI 학습을 호출하지 않는다. 실제 GPU AMP/DDP·장시간 학습·온라인 RL은 미실행이다. 로컬 GloVe·추가 HF tokenizer·공개 MiniMind 가중치 선택 셀은 None 분기만 실행했으므로 외부 리소스 호환성까지 통과한 것은 아니다.

수강생의 이해도·소요 시간·오답률은 직접 측정하지 않았다. 수정한 자료가 실제 학습 효과를 보장한다고 주장하지 않으며, 추가한 예측 문제·해설·원본 추적 질문으로 수업에서 확인해야 한다. 이번 검사에서 코드 셀 실패는 없었다.
