# MiniMind KV cache 대화 원본

## 사용자

minimind에서 kv cache하는 과정을 설명해

## 어시스턴트

MiniMind 코드에서 KV cache가 생성되고 재사용되는 부분을 찾아, 추론 흐름에 맞춰 설명하겠습니다.

MiniMind의 KV cache는 **이미 처리한 토큰의 Key·Value를 각 Transformer 레이어에 저장해 두고, 다음 토큰을 생성할 때 재사용하는 방식**입니다.

로컬 `projects/minimind` 코드 기준으로 설명하겠습니다.

1. **첫 호출에서는 프롬프트 전체를 처리합니다.**

   프롬프트가 `[A, B, C]`라면 각 레이어에서 세 토큰의 Q·K·V를 계산합니다. K·V는 캐시에 저장하고, 마지막 위치의 출력으로 다음 토큰 `D`를 예측합니다.

2. **다음 호출에서는 새 토큰만 모델에 넣습니다.**

   [`generate()`](/home/mind4u/dev/llm_playground/projects/minimind/model/model_minimind.py:265)의 핵심입니다.

   ```python
   past_len = past_key_values[0][0].shape[1] if past_key_values else 0

   outputs = self.forward(
       input_ids[:, past_len:],
       attention_mask,
       past_key_values,
       use_cache=use_cache,
   )
   ```

   `input_ids`에는 전체 토큰이 있지만, 캐시에 들어 있는 길이만큼 잘라냅니다. 따라서 `[A, B, C, D]` 중 **`D`만 새로 계산**합니다.

3. **새 K·V를 기존 캐시 뒤에 붙입니다.**

   각 레이어의 [`Attention.forward()`](/home/mind4u/dev/llm_playground/projects/minimind/model/model_minimind.py:111)에서 다음 작업을 합니다.

   ```python
   # 새로 입력된 토큰의 Q·K·V 계산
   xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)

   # 중간의 reshape, QK 정규화, RoPE 적용 생략

   if past_key_value is not None:
       xk = torch.cat([past_key_value[0], xk], dim=1)
       xv = torch.cat([past_key_value[1], xv], dim=1)

   past_kv = (xk, xv) if use_cache else None
   ```

   현재 토큰 `D`의 Query가 전체 `[A, B, C, D]`의 Key를 조회하고, 그 결과로 Value를 가중합합니다.

   ```text
   Q: D 하나
   K: A B C D
   V: A B C D
          ↓
   D 위치의 출력 → 다음 토큰 E 예측
   ```

4. **갱신된 캐시를 다음 생성 단계에 전달합니다.**

   [`MiniMindModel.forward()`](/home/mind4u/dev/llm_playground/projects/minimind/model/model_minimind.py:217)는 레이어별 캐시를 모아 반환합니다.

   ```python
   past_key_values = [
       (레이어0의 K, 레이어0의 V),
       (레이어1의 K, 레이어1의 V),
       ...
   ]
   ```

   생성 루프는 이를 받아 다음 호출에 사용합니다.

   ```python
   past_key_values = outputs.past_key_values if use_cache else None
   ```

전체 흐름은 다음과 같습니다.

| 생성 단계 | 이번에 계산하는 입력 | 계산 후 캐시 | 예측하는 토큰 |
|---|---|---|---|
| 첫 단계: prefill | A, B, C | A, B, C의 K·V | D |
| 다음 단계: decode | D | A, B, C, D의 K·V | E |
| 다음 단계: decode | E | A, B, C, D, E의 K·V | F |

위치 정보도 이어집니다. 캐시 길이가 3이면 새 토큰에는 **위치 3의 RoPE**를 적용합니다. 과거 K에는 이미 RoPE가 적용되어 있으므로 다시 적용하지 않습니다.

캐시 텐서 모양은 레이어마다 `[배치, 누적 토큰 수, KV 헤드 수, 헤드 차원]`입니다. MiniMind 기본 설정은 Q 헤드 8개·KV 헤드 4개의 GQA이며, **4개 KV 헤드 상태로 저장한 뒤** Attention 계산에 맞춰 확장합니다.

이렇게 하면 과거 토큰의 Q·K·V와 레이어 출력을 반복 계산하는 비용을 줄일 수 있습니다. 다만 새 Query가 과거 Key 전체를 조회하는 연산은 계속 필요하므로, 문장이 길어지면 캐시 메모리와 Attention 비용도 증가합니다.
