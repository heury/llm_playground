# MiniMind-O 음성 출력 과정 대화 원본

## 사용자

omni model에서 audio가 출력되는 과정을 이해하기 쉽게 설명해

## 어시스턴트

MiniMind-O의 Talker가 음성 코드를 생성하고 Mimi가 파형으로 복원하는 과정을 실제 코드로 확인하겠습니다.

MiniMind-O의 음성 출력은 **Thinker가 만든 의미 표현을 Talker가 음성 코드로 바꾸고, Mimi 디코더가 실제 소리로 복원하는 과정**입니다.

```text
사용자 입력: “봄을 한 문장으로 소개해줘”
                    ↓
                 Thinker
                    │
          ┌─────────┴─────────┐
          ↓                   ↓
     텍스트 출력          중간 hidden state
   “봄은 꽃이 피고…”           ↓
                           Talker
                              ↓
                     8개 코드북의 음성 코드
                              ↓
                         Mimi 디코더
                              ↓
                      실제 음성 파형 (24 kHz)
```

아래는 저장소의 실제 코드 기준입니다. 문장과 코드 숫자는 설명용 예시입니다.

**1. Thinker가 답변을 생성하면서 의미 벡터를 전달합니다**

Thinker는 Transformer를 통과하며 입력을 해석합니다. 최종 hidden state에서는 텍스트를 예측하고, **중간 층의 hidden state는 Talker로 전달**합니다.

[해당 코드](/home/mind4u/dev/llm_playground/projects/minimind-o/model/model_omni.py:288)를 간략히 표현하면 다음과 같습니다.

```python
for i, layer in enumerate(self.thinker.layers):
    hidden_states, present = layer(...)

    if i == self.config.bridge_layer:
        bridge_states = hidden_states

# 마지막 층 출력 → 다음 텍스트 토큰의 점수
text_logits = self.thinker.lm_head(
    self.thinker.norm(hidden_states)
)
```

기본 8층 Thinker에서는 `bridge_layer=3`, 즉 **네 번째 층의 출력**을 Talker에 연결합니다.

Talker는 완성된 답변 문자열을 기다리는 대신, 생성 과정에서 나오는 이 벡터를 받습니다.

**2. Talker는 의미 벡터와 이전 음성 코드를 함께 봅니다**

다음 소리를 예측하려면 “무슨 내용을 말하는가”와 “지금까지 어떤 소리를 냈는가”가 모두 필요합니다.

```text
Thinker의 의미 벡터           이전 음성 코드들
        ↓                          ↓
    embed_proj              임베딩 → codec_proj
        │                          │
        └──────── 가중합 ──────────┘
                       ↓
                Talker Transformer
```

[실제 결합 코드](/home/mind4u/dev/llm_playground/projects/minimind-o/model/model_omni.py:296)는 다음과 같습니다.

```python
talker_emb = self.talker.embed_tokens(audio_ids)

hidden_states = (
    self.talker.embed_proj(bridge_states) * self.talker.text_scale
    + self.talker.codec_proj(talker_emb) * self.talker.audio_scale
)
```

기본 Talker는 **4개의 Transformer 블록**을 사용합니다. 생성된 음성 코드는 다음 단계의 입력으로 다시 들어갑니다.

**3. Talker가 8개 코드북의 코드 점수를 예측합니다**

Talker가 바로 파형을 출력하는 것은 아닙니다. Mimi가 이해하는 **이산적인 음성 코드 ID**를 예측합니다.

한 음성 프레임은 다음처럼 코드 8개로 표현됩니다.

```text
한 프레임
[코드북0, 코드북1, 코드북2, 코드북3, 코드북4, 코드북5, 코드북6, 코드북7]
[ 1049,     811,    1626,    1562,     306,     714,    1044,     924 ]
```

8개 코드북은 같은 음성 구간을 여러 층으로 표현합니다. **각 코드가 단어 하나나 음소 하나를 뜻하는 것은 아닙니다.**

[`TalkerHead`](/home/mind4u/dev/llm_playground/projects/minimind-o/model/model_omni.py:58)는 공통 선형층과 코드북별 adapter를 사용합니다.

```python
def forward(self, x):
    base_out = self.base(x)

    return [
        base_out + adapter(x)
        for adapter in self.adapters
    ]
```

각 출력은 해당 코드북에서 다음에 선택할 코드의 점수입니다. 기본 설정에서는 특수 토큰을 포함해 **각각 2,112개 후보**를 가집니다.

**4. 코드북마다 시작 시점을 한 단계씩 늦춥니다**

이 구현의 중요한 부분입니다. 8개 코드북이 모두 같은 프레임을 동시에 예측하지 않고, **코드북별로 한 단계씩 지연**시킵니다.

생각 모드를 끈 일반 생성 경로를 단순화하면 다음과 같습니다.

```text
생성 단계    코드북0    코드북1    코드북2    …    코드북7
   0            ·          ·          ·               ·
   1           a₀          ·          ·               ·
   2           a₁         b₀          ·               ·
   3           a₂         b₁         c₀              ·
   ⋮
   8           a₇         b₆         c₅      …       h₀
```

`·`은 아직 생성하지 않는 자리입니다. 아래첨자가 같은 코드는 같은 음성 프레임에 해당합니다.

```text
첫 프레임 = [a₀, b₀, c₀, d₀, e₀, f₀, g₀, h₀]
```

[생성 루프](/home/mind4u/dev/llm_playground/projects/minimind-o/model/model_omni.py:362)의 핵심 조건입니다.

```python
audio_step = step - 1

for i, logits in enumerate(out.audio_logits):
    if audio_step < i:
        # 코드북 i는 아직 시작 전
        code = self.audio_pad_token
    else:
        # 해당 코드북에서 다음 코드 선택
        ...
```

따라서 **한 번의 forward에서 8개 코드북의 점수를 계산하되, 서로 다른 시간 위치의 코드를 생성**합니다. 지연을 되돌려 정렬하면 Mimi에 전달할 프레임이 됩니다.

**5. Mimi 디코더가 코드들을 파형으로 복원합니다**

정렬된 프레임들을 모으면 다음 구조입니다.

```text
프레임 0: [a₀, b₀, c₀, …, h₀]
프레임 1: [a₁, b₁, c₁, …, h₁]
프레임 2: [a₂, b₂, c₂, …, h₂]
...
```

[평가 코드](/home/mind4u/dev/llm_playground/projects/minimind-o/eval_omni.py:69)는 이를 `[배치, 코드북, 프레임]` 형태로 바꿔 디코딩합니다.

```python
# [프레임 수, 8] → [1, 8, 프레임 수]
mimi_codes = torch.tensor(
    codes,
    dtype=torch.long,
).T.unsqueeze(0)

audio = model.mimi_model.decode(mimi_codes).audio_values
```

기본 Mimi 설정의 프레임률은 **초당 12.5프레임**입니다.

```text
코드 8개로 이루어진 프레임 1개 ≈ 소리 80 ms
프레임 125개                  ≈ 소리 10초
```

복원된 파형은 **24 kHz**로 저장합니다.

```python
sf.write(
    wav_path,
    audio.squeeze().float().cpu().numpy(),
    24000,
)
```

모델은 음성 코드 프레임을 순차적으로 내보낼 수 있습니다. 현재 `eval_omni.py`는 프레임을 모두 모은 뒤 디코딩해 파일로 저장합니다.

**6. 목소리와 종료 시점은 별도로 제어합니다**

| 요소 | 역할 |
|---|---|
| `spk_emb` | 192차원 화자 벡터를 Talker 입력에 주입 |
| `ref_codes` | 참조 음성 코드를 초기 문맥에 넣어 목소리 조건 제공 |
| 종료 코드 | 각 코드북의 음성 종료 여부 표시 |

텍스트가 끝나도 음성 코드 생성이 남아 있을 수 있으므로, 생성 루프는 **텍스트 종료와 8개 코드북의 종료를 함께 확인**합니다. 최대 생성 길이에 도달해도 종료됩니다.

전체 코드를 읽을 때는 **`bridge_states` → Talker 입력 결합 → `audio_logits` → 지연 정렬 → `mimi_model.decode()`** 순서로 따라가면 됩니다.
