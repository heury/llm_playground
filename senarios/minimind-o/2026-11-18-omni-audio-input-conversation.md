# MiniMind-O 음성 입력 과정 대화 원본

## 사용자

omni 모델의 audio 입력 과정을 이해하기 쉽게 보여줘

## 어시스턴트

MiniMind-O에서 음성 파일이 특징 벡터로 바뀌고, 텍스트 토큰과 합쳐져 Thinker에 들어가는 경로를 실제 코드로 확인하겠습니다.

MiniMind-O는 **음성을 특징 벡터로 바꿔서, 텍스트 임베딩과 같은 공간에 넣습니다.** 예를 들어 “하늘은 왜 파란가요?”라는 음성이 들어오면 다음 경로를 거칩니다.

```text
음성 파일 (.wav / .mp3)
        ↓
① 파형 읽기 → 모노 변환 → 16 kHz로 리샘플링
        ↓
② 음향 특징 추출: [T, 560]
        ↓
③ SenseVoice 인코더: [T, 512]
        ↓
④ Audio Projector: [T, 768]
        ↓
⑤ <|audio_pad|> 자리의 임베딩을 음성 벡터로 교체
        ↓
⑥ Thinker Transformer
        ↓
답변 텍스트 + Talker를 통한 답변 음성
```

`T`는 음성 특징의 시간축 길이입니다. 위 차원은 기본 설정 기준입니다.

**1. 음성 파일을 숫자 배열로 읽습니다**

[음성 전처리 코드](/home/mind4u/dev/llm_playground/projects/minimind-o/dataset/omni_dataset.py:79)의 핵심입니다.

```python
wav, sr = sf.read(audio_path)

# 스테레오라면 좌우 채널을 평균내어 모노로 변환
if wav.ndim > 1:
    wav = wav.mean(axis=1)

# 입력 음성의 샘플링 주파수를 16 kHz로 맞춤
if sr != 16000:
    wav = librosa.resample(
        wav.astype(float),
        orig_sr=sr,
        target_sr=16000,
    )
```

예를 들어 3초짜리 음성이라면, 16 kHz에서 약 48,000개의 파형 값이 됩니다.

```text
“하늘은 왜 파란가요?”
        ↓
[0.001, 0.004, -0.002, ...]  약 48,000개
```

**2. 파형에서 음향 특징을 추출합니다**

```python
inputs = audio_processor(
    wav.astype(np.float32),
    sampling_rate=16000,
    return_tensors="pt",
    return_attention_mask=True,
)
```

[Audio Processor](/home/mind4u/dev/llm_playground/projects/minimind-o/model/model_omni.py:78)는 SenseVoice의 frontend를 호출합니다.

```python
fbank, flen = self.frontend(
    wav,
    torch.tensor([wav.size(1)]),
)
```

파형을 짧은 시간 구간으로 분석해 **소리의 주파수 성분을 담은 특징**을 만듭니다. 이 구현에서 LFR 처리를 거친 특징의 차원은 560입니다.

```text
파형: [샘플 수]
          ↓
특징: [T, 560]

시간 0 → [특징 560개]
시간 1 → [특징 560개]
시간 2 → [특징 560개]
...
```

배치 차원을 붙이면 모델에 전달되는 `audio_inputs`는 `[B, T, 560]`입니다. `audio_lens`에는 패딩을 제외한 유효 길이를 전달합니다.

**3. SenseVoice가 음성 표현을 만듭니다**

[음성 인코딩 코드](/home/mind4u/dev/llm_playground/projects/minimind-o/model/model_omni.py:138)를 간략히 쓰면 다음과 같습니다.

```python
with torch.no_grad():
    emb, _ = self.audio_encoder(
        valid_fbank,
        valid_lens,
    )
```

```text
음향 특징 [B, T, 560]
          ↓ SenseVoice
음성 표현 [B, T, 512]
```

이 경로에서는 **음성을 문자로 전사하는 단계를 거치지 않습니다.** SenseVoice의 인코더가 만든 연속 벡터를 사용합니다.

SenseVoice 인코더는 동결되어 있어 학습 중 가중치가 바뀌지 않습니다.

**4. Projector가 Thinker의 차원에 맞춥니다**

SenseVoice의 출력은 512차원이고, 기본 Thinker는 768차원입니다. [Audio Projector](/home/mind4u/dev/llm_playground/projects/minimind-o/model/model_omni.py:31)가 이를 연결합니다.

```python
self.mlp = nn.Sequential(
    nn.LayerNorm(512),
    nn.Linear(512, 768),
    nn.GELU(),
    nn.Linear(768, 768),
)
```

```text
음성 벡터 512차원
        ↓ 학습 가능한 변환
Thinker가 받을 수 있는 벡터 768차원
```

Projector는 학습 설정에 따라 업데이트되며, 음성 표현을 언어 모델이 활용할 수 있도록 맞추는 역할을 합니다.

**5. 음성 벡터가 들어갈 자리를 프롬프트에 만듭니다**

[추론 코드](/home/mind4u/dev/llm_playground/projects/minimind-o/eval_omni.py:182)는 유효 음성 길이만큼 `<|audio_pad|>`를 반복합니다.

```python
audio_token_len = valid_len or 1
prompt = model.config.audio_special_token * audio_token_len
```

가령 음성 특징이 4개라면 다음과 같은 프롬프트를 만듭니다.

```text
user:
<|audio_pad|><|audio_pad|><|audio_pad|><|audio_pad|>
assistant:
```

이 토큰은 **음성 벡터가 들어갈 자리를 표시**합니다. 음성 내용 자체는 별도의 `audio_inputs`로 전달됩니다.

**6. 자리표시자의 임베딩을 실제 음성 벡터로 교체합니다**

[모델의 forward](/home/mind4u/dev/llm_playground/projects/minimind-o/model/model_omni.py:265)는 먼저 평소처럼 텍스트 임베딩을 만듭니다.

```python
hidden_states = self.thinker.dropout(
    self.thinker.embed_tokens(text_ids)
)

audio_features = self.encode_audio_inputs(
    audio_inputs,
    audio_lens,
)

hidden_states = self.inject_audio_features(
    text_ids,
    hidden_states,
    audio_features,
    seq_length,
)
```

[`inject_audio_features`](/home/mind4u/dev/llm_playground/projects/minimind-o/model/model_omni.py:161)가 하는 일을 그림으로 보면 다음과 같습니다.

```text
교체 전
[ user ] [audio_pad] [audio_pad] [audio_pad] [audio_pad] [assistant]
             ↓           ↓           ↓           ↓
교체 후
[ user ] [음성벡터₀] [음성벡터₁] [음성벡터₂] [음성벡터₃] [assistant]
```

**토큰 ID는 유지하고, 해당 위치의 임베딩 벡터를 교체**합니다. 모든 위치가 같은 768차원이므로 Thinker는 이 시퀀스를 그대로 처리할 수 있습니다. 음성과 함께 텍스트 질문을 넣으면 텍스트 위치의 임베딩도 함께 유지됩니다.

이후 Thinker의 Attention이 음성 위치들을 참고해 답변을 생성합니다. KV cache를 사용하는 생성에서는 최초 프롬프트 처리 시 음성을 주입하고, 이후에는 캐시를 활용합니다.

앞에서 본 데이터 필드와 연결하면 다음과 같습니다.

| 데이터 | 처리 경로 |
|---|---|
| `question_audios`: 사용자 음성 바이트 | 파형 → 음향 특징 → SenseVoice → Projector → Thinker |
| `answer_audios`: 정답 음성의 Mimi 코드 | Talker의 음성 생성 학습에 사용 |

따라서 **입력 음성을 이해하는 SenseVoice 경로**와 **답변 음성을 생성하는 Mimi 코드 경로**를 구분해서 읽으면 코드가 훨씬 명확해집니다.
