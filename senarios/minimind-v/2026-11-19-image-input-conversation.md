# MiniMind-V 이미지 입력 과정 대화 원본

## 사용자

minimind-v에서 이미지가 입력되는 과정을 간단한 코드로 설명해

## 어시스턴트

MiniMind-V는 이미지를 64개의 "가짜 토큰 벡터"로 바꾼 뒤, 텍스트 안에 미리 비워 둔 64칸에 끼워 넣습니다. 그 뒤로는 일반 언어 모델과 똑같이 처리합니다.

## 전체 흐름

```
이미지 ─► SigLIP 인코더(고정) ─► [64, 768] ─► Projector(MLP) ─► [64, hidden]
                                                                      │
텍스트 "<image>\n이건 뭐야?"                                            │ 교체
  └► "<|image_pad|>"×64 + "\n이건 뭐야?" ─► 토큰 임베딩 [L, hidden] ◄──┘
                                                │
                                         Transformer 층 ─► 다음 토큰 예측
```

## 간단한 코드로 보면

```python
import torch, torch.nn as nn

IMAGE_ID  = 12     # '<|image_pad|>' 토큰 ID (VLMConfig.image_ids)
N_IMG_TOK = 64     # 이미지 하나가 차지하는 토큰 수 (image_token_len)
hidden    = 512    # 언어 모델 hidden_size

# 1) 프롬프트: '<image>'를 자리표시자 토큰 64개로 펼친다
prompt = "<image>\n이 그림을 설명해줘"
prompt = prompt.replace("<image>", "<|image_pad|>" * N_IMG_TOK)
input_ids = tokenizer(prompt)                      # [..., 12, 12, ..., 12, ...]

# 2) 이미지 → 픽셀 텐서 (SigLIP processor, 256×256)
pixel_values = processor(images=img)["pixel_values"]   # [1, 3, 256, 256]

# 3) 비전 인코더(고정): 32×32 패치 → 8×8 = 64개 패치 특징
with torch.no_grad():
    feats = vision_encoder(pixel_values).last_hidden_state  # [1, 64, 768]

# 4) Projector: 비전 공간 768 → 언어 모델 공간 hidden
proj = nn.Sequential(nn.LayerNorm(768), nn.Linear(768, hidden),
                     nn.GELU(), nn.Linear(hidden, hidden))
img_emb = proj(feats)                                       # [1, 64, hidden]

# 5) 텍스트 임베딩을 만든 뒤 자리표시자 위치를 이미지 벡터로 덮어쓴다
h = embed_tokens(input_ids)                                 # [1, L, hidden]
pos = (input_ids[0] == IMAGE_ID).nonzero().squeeze(-1)      # 연속된 64개 위치
h[0, pos] = img_emb[0]

# 6) 이후는 일반 LM과 같다
for layer in layers:
    h = layer(h)
logits = lm_head(norm(h))
```

## 실제 코드와 대응

| 단계 | 위치 |
|---|---|
| `<image>` → `<\|image_pad\|>`×64 | [lm_dataset.py:65](/home/mind4u/dev/llm_playground/projects/minimind-v/dataset/lm_dataset.py#L65), [eval_vlm.py:62](/home/mind4u/dev/llm_playground/projects/minimind-v/eval_vlm.py#L62) |
| SigLIP 로드 및 파라미터 고정 | [model_vlm.py](/home/mind4u/dev/llm_playground/projects/minimind-v/model/model_vlm.py) `get_vision_model` |
| 인코더 출력 `last_hidden_state` | `get_image_embeddings` |
| Projector (LayerNorm→Linear→GELU→Linear) | `MMVisionProjector` |
| 자리표시자 구간을 이미지 벡터로 교체 | `count_vision_proj` |
| 이미지 주입은 첫 스텝에서만 | `forward`의 `start_pos == 0` 조건 |

## 짚어 둘 점

- **자리표시자 수와 패치 수가 같아야 합니다.** `siglip2-base-p32-256`은 256/32 = 8, 즉 8×8 = 64개 패치를 내므로 `image_token_len=64`와 맞습니다. 인코더를 바꾸면 이 값도 바꿔야 합니다.
- **실제 구현은 텐서를 잘라 붙입니다.** 위 예시처럼 인덱스에 대입하지 않고 `torch.cat((hb[:start], 이미지벡터, hb[i:]))`로 이어 붙입니다. 연속된 마커 구간을 하나 찾을 때마다 다음 이미지(`k += 1`)를 넣기 때문에, 한 샘플에 이미지를 여러 장 넣을 수 있습니다.
- **생성할 때는 이미지를 한 번만 넣습니다.** 첫 forward에서 이미지 벡터가 KV cache에 들어가고, 이후 토큰을 생성할 때는 `start_pos > 0`이라 이미지를 다시 처리하지 않습니다.
- **학습되는 부분:** 비전 인코더는 항상 고정입니다. 사전 정렬 단계에서는 주로 Projector를 학습하고, SFT에서는 LLM까지 함께 학습합니다. 정확한 범위는 trainer의 `freeze_llm` 설정을 확인하세요.
