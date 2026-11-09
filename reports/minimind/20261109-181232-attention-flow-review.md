# 3장 Attention 계산 흐름의 수강생 관점 검토

검토 ID: 20261109-181232-attention-flow-review. 대상: [Chapter_3_Exercise_MiniMind_Attention.ipynb](../../notebooks/Chapter_3_Exercise_MiniMind_Attention.ipynb). 셀 번호는 0부터 시작하는 JSON 인덱스다.

## 판단

**행렬곱과 PyTorch shape 연산에 익숙하면 계산 순서를 추적할 수 있다. 하지만 처음 배우는 수강생이 “특정 token 하나의 출력 벡터가 어떻게 만들어졌는지”를 혼자 설명하기에는 중간 단계와 숫자 예시가 부족하다.** 현재 자료는 올바른 수동 연산과 실제 모듈의 결과 일치 검증이 강점이다. 그 계산을 이해하는 학습 경로는 더 잘게 나눌 필요가 있다.

이번은 설명·코드·원본 모듈의 정적 검토다. 새로운 실행이나 학습, 노트북 수정은 하지 않았다. 기존 설정 변경 회귀 기록의 5개 유효 조합에서 수동/원본 최대 오차가 0이었던 사실은 참고 자료로만 사용했다.

## 계산 순서의 정확성

셀 8→10은 다음 MiniMind 연산 순서를 따른다. [원본 Attention.forward](../../projects/minimind/model/model_minimind.py)의 기본 cache 없는 float32·flash_attn=False·eval 경로와 대응한다.

```text
입력 h
→ Q/K/V 선형 투영
→ head로 나누기
→ Q/K norm
→ Q/K RoPE
→ KV head 공유와 축 교환
→ QKᵀ / √head_dim
→ 미래 위치 mask
→ 입력 위치에 대한 softmax
→ V 가중합
→ head 합치기
→ o_proj
→ 출력
```

Q/K norm, RoPE, GQA 반복, causal softmax, 출력 투영을 누락하지 않았다. 셀 4의 독립 head 계산과 batched 계산 비교, 셀 6의 회전 내적 예제, 셀 12의 미래 입력 변경 불변성 검사는 유용하다. KV cache·padding mask·학습 dropout의 모든 분기는 이 셀의 범위가 아니며 기본 계산을 설명하기 위해 반드시 여기서 실행해야 하는 것은 아니다.

## 우선 보강할 부분

### 1. V 가중합부터 출력까지를 한 줄에 묶었다 — 높음

셀 10:

```python
manual = attn.o_proj((probs @ v_heads).transpose(1, 2).reshape(batch_size, sequence_length, -1))
```

수강생은 동시에 행렬곱·축 교환·head concat·출력 투영을 해석해야 한다. 현재 출력은 첫 head의 확률 행렬이고, 실제 문맥 벡터와 최종 출력의 중간 shape는 출력하지 않는다. Attention이 반환하는 것이 확률인지 벡터인지 혼동하기 쉽다.

권장: `context_per_head = probs @ v_heads`, `context_by_token = ...transpose(...)`, `concatenated = ...reshape(...)`, `output = attn.o_proj(...)`를 별도 단계로 분리한다. 각 shape와 한 token 벡터의 일부 값을 출력한다. **probs는 입력 위치에 대한 가중치, context는 V를 섞은 벡터, 최종 output은 o_proj를 지난 벡터**라고 명시한다.

### 2. 한 token의 실제 숫자로 가중합하는 예제가 없다 — 높음

셀 4는 랜덤 Linear로 Q/K/V를 만든 뒤 shape를 보여 주고, 셀 10도 확률 행렬 전체를 출력한다. 두 셀 모두 한 출력이 `p0*v0 + p1*v1 + ...`로 만들어지는 과정을 숫자로 보여 주지는 않는다.

권장: token 3개, head_dim=2의 고정 Q/K/V를 사용한다. 두 번째 token에 대해 score 두 개·mask된 미래·softmax 확률·V 가중합을 직접 계산하고 행렬곱 결과와 비교한다. 학습된 의미를 보여 주는 예제가 아니라 계산을 손으로 추적하는 예제임을 명시한다.

### 3. score 행·열과 softmax 축이 명시적이지 않다 — 높음

셀 9는 “행별 확률 합은 1”이라고 하지만 어떤 행과 열이 무엇인지, 왜 `softmax(-1)`인지 충분히 설명하지 않는다. 초보자는 Q/K token 축을 반대로 해석할 수 있다.

권장: `scores[b,h,i,j]`가 “출력 위치 i의 query가 입력 위치 j의 key를 얼마나 참조하는가”임을 정의한다. 마지막 축 j를 정규화한다고 표시한다. i=1인 행에는 j=0,1만 남고 j>1은 0임을 출력한다. 대각선은 현재 token을 허용하며 그 위치에서 **다음 token**을 예측하므로 정답 누출이 아니라는 설명도 연결한다.

### 4. shape 전환표와 GQA head 대응이 부족하다 — 중간

셀 8은 Q/K/V shape를 출력하지만 KV 반복·transpose 이후 shape는 보여 주지 않는다. `repeat_kv(..., q_count // kv_count)`가 어느 head를 공유하는지 번호로 보이지 않는다.

현재 기본 설정의 기대 흐름:

| 단계 | shape | 의미 |
|---|---|---|
| h | [1,6,64] | 6개 token의 입력 hidden vector |
| q_proj / k_proj / v_proj | [1,6,64] / [1,6,32] / [1,6,32] | Q와 KV 투영 크기가 다름 |
| head로 분리한 Q / K,V | [1,6,4,16] / [1,6,2,16] | Q head 4개, KV head 2개 |
| norm·RoPE | 위와 같음 | Q/K 값은 바뀌지만 shape는 유지 |
| 반복·transpose 후 Q/K/V | [1,4,6,16] | KV를 Q head 수에 맞추고 head 축을 앞으로 |
| scores / probs | [1,4,6,6] | 출력 위치×참조 위치 |
| head별 context | [1,4,6,16] | 각 head가 만든 가중합 벡터 |
| transpose·concat | [1,6,64] | token별로 4개 head를 합침 |
| o_proj | [1,6,64] | head 정보를 섞어 출력 hidden 크기로 투영 |

MiniMind repeat_kv에서 KV0는 Q0/Q1에, KV1은 Q2/Q3에 대응한다. 이런 대응표와 단계별 shape 출력이 있으면 view·transpose가 임의의 변형이 아니라는 점을 이해하기 쉽다.

### 5. scaling·Q/K/V 역할 설명이 사용보다 늦거나 짧다 — 중간

셀 4가 Q/K/V 투영과 `√d` scaling을 먼저 사용하고, 역할 비유는 셀 7에서 설명한다. 왜 Q와 K를 서로 다른 가중치로 만들고 왜 V를 마지막에 섞는지, scaling이 없는 경우 어떤 문제가 생길 수 있는지에 대한 질문이 남는다.

권장: 셀 4 전에 Q=찾는 조건, K=매칭할 특성, V=가져올 내용이라는 연산 관점의 설명을 둔다. Q/K 점수가 커지면 softmax가 뾰족해질 수 있어 차원 크기를 보정하려고 √d로 나눈다는 직관을 추가한다. 이 통계적 설명이 Q/K norm·RoPE를 포함한 모든 설정에서 정확히 동일한 분포를 보장한다는 의미는 아니라고 구분한다.

### 6. 과제와 기호의 작은 혼동도 정리하면 좋다 — 중간

- setup 설명의 `V=어휘 크기`와 Attention의 Value tensor V가 같은 문자다. 이 장에서는 `V_vocab`과 `values` 또는 Value를 구분하면 좋다.
- 셀 11의 “앞 네 위치”는 기본 길이 6일 때만 맞는다. 실제 코드는 `sequence_length-2`를 사용하므로 “마지막 두 위치 이전”으로 쓰는 편이 설정 변경에도 맞다.
- causal mask 제거 과제는 수동 `future` mask와 실제 모듈의 `attn.is_causal`이 별개라는 안내가 필요하다. 수동 mask만 지우면 미래 확률 0 assert나 수동/모듈 일치 assert에서 먼저 실패할 수 있으며, 이후 원본 모듈의 미래 변경 검사만으로 의도한 음성 실험을 설명할 수 없다. 어느 경로를 바꾸고 어느 assert가 실패할지 예측하게 해야 한다.
- 이 장은 학습 loss를 줄이는 실습이 아닌데 제출 기준 뒤에 공통 문구 “손실 하락은…”이 붙어 있다. 이 장의 확인은 수치 일치·미래 차단·shape라는 문구가 더 정확하다.

## 수강생이 답할 수 있어야 할 최소 질문

1. 입력 token 하나가 Q/K/V 세 벡터로 바뀔 때 각 W의 역할은 무엇인가?
2. `scores[...,i,j]`의 i와 j가 무엇이며 왜 마지막 축에 softmax하는가?
3. 현재 위치는 허용하고 미래만 막는 이유는 무엇인가?
4. 확률 행렬과 `probs @ values`의 shape와 의미는 어떻게 다른가?
5. head를 나누었다가 다시 합치는 이유와 o_proj의 역할은 무엇인가?
6. Q heads=4/KV heads=2일 때 각 query head가 사용하는 KV head를 번호로 설명할 수 있는가?
7. 원본 Attention.forward에서 자신이 계산한 각 단계의 코드를 찾을 수 있는가?

현재 코드로 7번의 연산 대조와 수치 검증은 할 수 있다. 2·4·5번을 직접 설명하도록 하려면 token별 숫자 예제와 중간 tensor의 의미를 보강하는 것이 우선이다.

## 제안하는 절 구성

1. 전체 계산 지도와 Q/K/V 역할.
2. 고정된 작은 Q/K/V로 token 하나의 score→mask→softmax→가중합.
3. head 분리·head별 계산·concat·출력 투영.
4. MiniMind의 norm·RoPE·GQA를 같은 지도에 추가.
5. 원본 Attention.forward와 단계별 비교.
6. 미래 입력 변경, KV head 변경, mask 제거 과제.

기존 셀과 검사를 대부분 유지하면서 설명·중간 출력·고정 숫자 예제를 추가하는 방식으로 보강할 수 있다.

## 근거와 검증 범위

작업 디렉토리: 저장소 루트. Python 표준 라이브러리로 노트북의 모든 셀을 읽고 원본 Attention.forward를 대조했다. 새로운 학습·실행·패키지 변경은 없다. 파일 hash와 셀별 원문은 [review-evidence.json](../../outputs/minimind/20261109-181232-attention-flow-review/review-evidence.json)에 보관했다.

기존 [GQA 회귀 결과](../../outputs/minimind/20261109-180606-student-revision/attention-exercises/results.json)는 코드 일치의 참고이며 이번 새 실행 결과가 아니다. 실제 수강생의 이해도나 학습 시간을 측정하지 않았다. 노트북은 수정하지 않았다.
