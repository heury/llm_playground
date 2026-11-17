# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "torch",
#     "numpy",
#     "matplotlib",
# ]
# ///

import marimo

__generated_with = "0.17.0"
app = marimo.App(width="medium", app_title="SmolVLA 실습")


@app.cell
def _():
    import marimo as mo
    import math
    import time
    from collections import deque
    from dataclasses import dataclass

    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    return F, dataclass, deque, math, mo, nn, np, plt, time, torch


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # SmolVLA 실습 — 코드로 보는 핵심 개념

    이 노트북은 `lerobot`의 SmolVLA 구현
    ([`modeling_smolvla.py`](https://github.com/huggingface/lerobot/blob/main/src/lerobot/policies/smolvla/modeling_smolvla.py),
    [`smolvlm_with_expert.py`](https://github.com/huggingface/lerobot/blob/main/src/lerobot/policies/smolvla/smolvlm_with_expert.py))
    의 **구조와 함수 이름을 그대로 따르는 미니 SmolVLA**를 직접 만들어 보고, 장난감 과제로 학습시켜 봅니다.

    | 절 | 개념 | 대응하는 lerobot 코드 |
    |---|---|---|
    | 1 | 설정값 | `SmolVLAConfig` |
    | 2 | 시각 토큰 축소 (pixel shuffle) | `SmolVLMWithExpertModel.embed_image` → `connector` |
    | 3 | Prefix 구성과 attention mask | `VLAFlowMatching.embed_prefix`, `make_att_2d_masks` |
    | 4 | 레이어 스킵 · Action Expert · CA/SA 교차 | `SmolVLMWithExpertModel.__init__`, `forward`, `forward_attn_layer`, `forward_cross_attn_layer` |
    | 5 | 시간 임베딩 | `VLAFlowMatching.embed_suffix`, `create_sinusoidal_pos_embedding` |
    | 6 | Flow Matching 학습 | `VLAFlowMatching.forward`, `make_flow_matching_inputs` |
    | 7 | 추론: KV 캐시 + Euler 적분 | `sample_actions`, `denoise_step`, `euler_integrate` |
    | 8 | Action queue | `SmolVLAPolicy.select_action` |
    | 9 | (선택 심화) 비동기 추론 | `async_inference/robot_client.py` |
    | 10 | (선택) 실제 `smolvla_base` 살펴보기 | — |

    **학습 흐름:** 2~7절에서 시각 토큰 축소 → prefix/suffix → Action Expert → Flow Matching → 추론을
    살펴보고, 8절에서 예측한 행동을 제어 루프에 전달합니다. **9절은 선택 심화(실행 시스템)**,
    10절은 실제 모델과의 구조 비교입니다.

    범용 Transformer 연산·장난감 데이터 제작·optimizer 루프는 같은 폴더의 `smolvla_helpers.py`로
    분리했습니다. 실행할 때 두 파일을 함께 두세요. 시각화 셀은 그림을 먼저 볼 수 있도록 코드를 접어 두었습니다.

    > 실제 가중치를 내려받지 않고 **CPU에서도 몇 분 안에** 모두 실행됩니다 (학습 포함).
    > 단순화한 부분(GQA 생략, 작은 차원, 정규화 생략 등)은 코드 주석에 `[실제]`로 표시했습니다.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. 설정값

    실제 SmolVLA 값과 미니 버전 값을 나란히 적었습니다. **비율(레이어 절반, 폭 0.75배, 2층마다 SA)** 은 그대로 유지합니다.
    """)
    return


@app.cell
def _(dataclass, torch):
    @dataclass
    class MiniConfig:
        # ---- VLM 백본 ----                      [실제 SmolVLA]
        vlm_hidden: int = 64                     # 960 (SmolLM2 hidden)
        num_heads: int = 4                       # 15 (KV head는 5개, GQA)
        total_vlm_layers: int = 8                # 32
        num_vlm_layers: int = 4                  # 16  ← 레이어 스킵: 앞쪽 절반만 사용
        # ---- Action Expert ----
        expert_width_multiplier: float = 0.75    # 0.75 ← expert hidden = 0.75 × VLM hidden
        self_attn_every_n_layers: int = 2        # 2    ← 0,2,4..번 층은 SA, 나머지는 CA
        # ---- 입력 ----
        image_size: int = 32                     # 512
        patch_size: int = 4                      # 16 (SigLIP)
        scale_factor: int = 2                    # 4  (pixel shuffle)
        vocab_size: int = 8                      # ~49k
        max_state_dim: int = 8                   # 32 (로봇마다 다른 차원을 0으로 패딩)
        max_action_dim: int = 8                  # 32
        action_dim: int = 2                      # 로봇에 따라 다름 (SO-100: 6)
        # ---- Flow matching / chunking ----
        chunk_size: int = 16                     # 50
        num_steps: int = 10                      # 10 (Euler 적분 횟수)
        min_period: float = 4e-3                 # 4e-3
        max_period: float = 4.0                  # 4.0

    cfg = MiniConfig()
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print("device:", device)
    return MiniConfig, cfg, device


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. 시각 토큰 축소 — Pixel Shuffle

    SmolVLM의 connector는 SigLIP 패치 토큰을 **공간적으로 r×r개씩 묶어 채널 방향으로 쌓습니다** (space-to-depth).
    토큰 수는 1/r² 로 줄고, 채널은 r² 배로 늘어난 뒤 선형층으로 LM hidden 차원에 맞춥니다.

    - 실제: 512×512 이미지 → 16px 패치 → 32×32 = **1024 토큰** → r=4 → **64 토큰**
    - SmolVLA는 SmolVLM의 이미지 타일링(splitting)도 끄고 **전역 이미지 1장**만 씁니다.

    아래 `pixel_shuffle`은 transformers의 `Idefics3Connector.pixel_shuffle`과 같은 코드입니다.
    """)
    return


@app.cell
def _():
    def pixel_shuffle(x, scale_factor):
        """(B, N, C) → (B, N / r², C · r²).  transformers Idefics3Connector와 동일."""
        bsz, seq, embed_dim = x.size()
        height = width = int(seq**0.5)
        x = x.view(bsz, height, width, embed_dim)
        x = x.view(bsz, height, width // scale_factor, embed_dim * scale_factor)
        x = x.permute(0, 2, 1, 3)
        x = x.reshape(bsz, width // scale_factor, height // scale_factor, embed_dim * scale_factor**2)
        x = x.permute(0, 2, 1, 3)
        x = x.reshape(bsz, seq // scale_factor**2, embed_dim * scale_factor**2)
        return x
    return (pixel_shuffle,)


@app.cell
def _(pixel_shuffle, torch):
    # 실제 크기로 shape만 확인해 보기 (SigLIP hidden = 768)
    _siglip_out = torch.randn(1, (512 // 16) ** 2, 768)
    _shuffled = pixel_shuffle(_siglip_out, scale_factor=4)
    print("SigLIP 출력      :", tuple(_siglip_out.shape), "← 이미지 1장당 1024 토큰")
    print("pixel shuffle 후 :", tuple(_shuffled.shape), "← 64 토큰, 채널 16배")
    print("connector Linear : 12288 → 960 (LM hidden)")

    # 어떤 패치들이 하나의 토큰으로 묶이는지 확인 (4x4 격자, r=2)
    _ids = torch.arange(16).float().view(1, 16, 1)
    print("\n4x4 패치 번호:\n", _ids.view(4, 4).long())
    print("r=2 shuffle 후 각 토큰에 묶인 패치 번호:\n", pixel_shuffle(_ids, 2)[0].long())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Prefix 구성과 Attention Mask

    `embed_prefix`는 **[카메라1 토큰 | 카메라2 토큰 | … | 언어 토큰 | state 토큰]** 을 이어 붙입니다.
    Suffix(`embed_suffix`)는 **노이즈 섞인 action 토큰 50개**입니다.

    마스크는 openpi/big_vision에서 온 `make_att_2d_masks` 하나로 만듭니다. 규칙은 단 하나:

    > 토큰 *i* 는 `cumsum(att_masks)[j] <= cumsum(att_masks)[i]` 인 토큰 *j* 만 볼 수 있다.

    `att_masks` 값이 `0`이면 "앞 토큰과 같은 블록", `1`이면 "새 블록 시작"입니다. SmolVLA의 설정:

    | 토큰 | att_mask | 결과 |
    |---|---|---|
    | 이미지, 언어 | 0 | 하나의 블록 → 서로 **양방향**으로 봄 |
    | state | **1** | 새 블록 → 이미지·언어는 state를 **못 봄**, state는 앞을 모두 봄 |
    | action × chunk | **1, 1, 1, …** | 토큰마다 새 블록 → action끼리 **causal**, prefix 전체는 봄 |

    핵심: **prefix는 action(suffix)을 절대 보지 않습니다.** → prefix의 K/V는 노이즈와 무관 → 추론 때 1번만 계산해 캐시(7절).
    """)
    return


@app.cell
def _(torch):
    def make_att_2d_masks(pad_masks, att_masks):
        """lerobot/openpi와 동일. pad_masks: 패딩이 아니면 True, att_masks: 새 블록이면 1."""
        cumsum = torch.cumsum(att_masks, dim=1)
        att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
        pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
        return att_2d_masks & pad_2d_masks
    return (make_att_2d_masks,)


@app.cell
def _(mo):
    mask_n_img = mo.ui.slider(1, 6, value=3, label="카메라당 이미지 토큰 수")
    mask_n_lang_pad = mo.ui.slider(0, 3, value=1, label="언어 패딩 토큰 수")
    mask_n_act = mo.ui.slider(2, 8, value=5, label="action 토큰 수 (chunk)")
    mo.hstack([mask_n_img, mask_n_lang_pad, mask_n_act])
    return mask_n_act, mask_n_img, mask_n_lang_pad


@app.cell
def _(make_att_2d_masks, mask_n_act, mask_n_img, mask_n_lang_pad, plt, torch):
    _n_img, _n_lang, _n_pad, _n_act = mask_n_img.value, 3, mask_n_lang_pad.value, mask_n_act.value
    _labels = (
        [f"cam1_{i}" for i in range(_n_img)] + [f"cam2_{i}" for i in range(_n_img)]
        + [f"lang{i}" for i in range(_n_lang)] + ["<pad>"] * _n_pad
        + ["state"] + [f"act{i}" for i in range(_n_act)]
    )
    # embed_prefix: 이미지·언어 = 0, state = 1 / embed_suffix: action = 1
    _att = [0] * (2 * _n_img) + [0] * (_n_lang + _n_pad) + [1] + [1] * _n_act
    _pad = [True] * (2 * _n_img + _n_lang) + [False] * _n_pad + [True] * (1 + _n_act)
    _mask = make_att_2d_masks(torch.tensor([_pad]), torch.tensor([_att]))[0]

    _fig, _ax = plt.subplots(figsize=(6.5, 6.5))
    _ax.imshow(_mask, cmap="Blues", vmin=0, vmax=1.4)
    _ax.set_xticks(range(len(_labels)), _labels, rotation=90, fontsize=8)
    _ax.set_yticks(range(len(_labels)), _labels, fontsize=8)
    _ax.set_xlabel("key (attended to)")
    _ax.set_ylabel("query (attending)")
    _n_prefix = 2 * _n_img + _n_lang + _n_pad + 1
    _ax.axhline(_n_prefix - 0.5, color="tab:orange", lw=2)
    _ax.axvline(_n_prefix - 0.5, color="tab:orange", lw=2)
    _ax.set_title("SmolVLA attention mask (blue = can attend)\norange line: prefix | suffix")
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. VLM + Action Expert

    ### 4.1 VLM 빌딩 블록과 Expert의 연결

    VLM 쪽과 expert 쪽 모두 **같은 모양의 decoder layer**를 씁니다
    (`input_layernorm → self_attn(q/k/v/o_proj) → post_attention_layernorm → mlp`, HF Llama 계열과 동일한 속성 이름).
    범용 연산(`RMSNorm`, RoPE, attention, 기본 decoder, 미니 vision encoder)은
    **`smolvla_helpers.py`**에서 가져옵니다. 세부 구현은 필요할 때 그 파일에서 확인하세요.
    아래에는 **pixel shuffle을 연결하는 connector와 VLM 조립 과정**을 남겼습니다.

    중요한 사실 하나: expert는 hidden 폭만 줄였고 **head 수·head_dim은 VLM과 같습니다.**
    그래서 q/k/v 차원(`heads × head_dim`)이 두 모델에서 똑같고, SA 층에서 둘의 Q/K/V를 **한 시퀀스로 이어 붙일 수 있습니다.**
    """)
    return


@app.cell
def _(cfg, nn, pixel_shuffle, torch):
    from smolvla_helpers import (
        MiniDecoderLayer,
        MiniVisionModel,
        RMSNorm,
        apply_rope,
        eager_attention,
    )

    class MiniConnector(nn.Module):
        """pixel shuffle + Linear → LM hidden"""
        def __init__(self, c, vis_dim=32):
            super().__init__()
            self.scale_factor = c.scale_factor
            self.proj = nn.Linear(vis_dim * c.scale_factor**2, c.vlm_hidden)

        def forward(self, x):
            return self.proj(pixel_shuffle(x, self.scale_factor))

    class MiniVLM(nn.Module):
        """SmolVLM 흉내: vision_model + connector + text_model(embed_tokens, layers, norm)"""
        def __init__(self, c):
            super().__init__()
            head_dim = c.vlm_hidden // c.num_heads
            self.vision_model = MiniVisionModel(c)
            self.connector = MiniConnector(c)
            self.embed_tokens = nn.Embedding(c.vocab_size, c.vlm_hidden)
            self.layers = nn.ModuleList(
                [MiniDecoderLayer(c.vlm_hidden, c.num_heads, head_dim) for _ in range(c.total_vlm_layers)]
            )
            self.norm = RMSNorm(c.vlm_hidden)

    print("VLM 레이어 1개 파라미터:", sum(p.numel() for p in MiniDecoderLayer(cfg.vlm_hidden, 4, 16).parameters()))
    return MiniDecoderLayer, MiniVLM, RMSNorm, apply_rope, eager_attention


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 4.2 `SmolVLMWithExpertModel` — 이 노트북의 핵심 클래스

    `__init__`에서 일어나는 일 (lerobot 코드와 순서 동일):

    1. **레이어 스킵** — `text_model.layers = text_model.layers[:num_vlm_layers]`. 한 줄로 VLM 상위 절반을 버립니다.
    2. **Action expert 생성** — VLM text config를 복사해 `hidden_size`만 0.75배, 레이어 수는 (스킵 후) VLM과 동일.
    3. **CA 층의 k/v_proj 교체** — CA 층에서 expert는 자기 hidden이 아니라 **VLM의 K/V(이미 투영된 값)** 를 입력으로 받으므로,
       `k_proj`, `v_proj`의 입력 차원을 VLM의 `kv_heads × head_dim`으로 바꿉니다.
    4. **Freeze** — `train_expert_only=True`이면 VLM 전체를 고정하고, `train()`을 오버라이드해 VLM을 항상 eval 모드로 둡니다.

    `forward`의 레이어 루프는 층마다 둘 중 하나를 고릅니다.

    | 경로 | 언제 | 무엇을 하나 |
    |---|---|---|
    | `forward_attn_layer` (SA) | `layer_idx % 2 == 0` 또는 prefix 캐시 채우기 | prefix·suffix의 Q/K/V를 **이어 붙여** 한 번에 attention (2D 마스크 적용) |
    | `forward_cross_attn_layer` (CA) | 나머지 층 | prefix는 자기들끼리 SA, expert는 **Q만 자기 것**, K/V는 VLM 것을 expert의 k/v_proj로 변환해서 사용 |

    마지막으로 각 스트림은 **자기 모델의** `o_proj → residual → mlp → residual`을 거칩니다.
    """)
    return


@app.cell
def _(MiniDecoderLayer, MiniVLM, RMSNorm, apply_rope, eager_attention, nn, torch):
    class MiniVLMWithExpert(nn.Module):
        def __init__(self, c, train_expert_only=True):
            super().__init__()
            self.vlm = MiniVLM(c)
            # ① 레이어 스킵: 앞쪽 num_vlm_layers개만 남김
            self.vlm.layers = self.vlm.layers[: c.num_vlm_layers]
            self.num_vlm_layers = len(self.vlm.layers)

            # ② Action expert: 폭 0.75배, 레이어 수 동일, head 수/head_dim 동일
            self.head_dim = c.vlm_hidden // c.num_heads
            self.expert_hidden_size = int(c.vlm_hidden * c.expert_width_multiplier)
            self.lm_expert_layers = nn.ModuleList(
                [MiniDecoderLayer(self.expert_hidden_size, c.num_heads, self.head_dim) for _ in range(self.num_vlm_layers)]
            )
            self.lm_expert_norm = RMSNorm(self.expert_hidden_size)

            # ③ CA 층: k/v_proj 입력 = VLM의 K/V (kv_heads*head_dim) 차원
            self.self_attn_every_n_layers = c.self_attn_every_n_layers
            kv_dim = c.num_heads * self.head_dim
            for i, layer in enumerate(self.lm_expert_layers):
                if i % self.self_attn_every_n_layers == 0:
                    continue  # SA 층은 그대로
                layer.self_attn.k_proj = nn.Linear(kv_dim, kv_dim, bias=False)
                layer.self_attn.v_proj = nn.Linear(kv_dim, kv_dim, bias=False)

            # ④ Freeze: VLM은 고정, expert만 학습
            self.train_expert_only = train_expert_only
            if train_expert_only:
                for p in self.vlm.parameters():
                    p.requires_grad = False

            self.calls = {"vlm_layer": 0, "expert_layer": 0}  # 7절에서 KV 캐시 효과 측정용

        def train(self, mode=True):
            # lerobot과 동일: model.train()을 불러도 고정된 VLM은 eval 모드 유지 (dropout 등 비활성)
            super().train(mode)
            if self.train_expert_only:
                self.vlm.eval()
            return self

        def is_cross_attn_layer(self, layer_idx):
            return layer_idx % self.self_attn_every_n_layers != 0

        def embed_image(self, image):
            return self.vlm.connector(self.vlm.vision_model(image))

        def embed_language_tokens(self, tokens):
            return self.vlm.embed_tokens(tokens)

        # ---------------- SA 층: prefix + suffix를 한 시퀀스로 ----------------
        def forward_attn_layer(self, model_layers, inputs_embeds, layer_idx, position_ids,
                               attention_mask, past_key_values, fill_kv_cache):
            qs, ks, vs = [], [], []
            for i, hidden_states in enumerate(inputs_embeds):
                if hidden_states is None:
                    continue
                layer = model_layers[i][layer_idx]
                h = layer.input_layernorm(hidden_states)
                shape = (*h.shape[:-1], -1, self.head_dim)
                qs.append(layer.self_attn.q_proj(h).view(shape))
                ks.append(layer.self_attn.k_proj(h).view(shape))
                vs.append(layer.self_attn.v_proj(h).view(shape))
            q, k, v = torch.cat(qs, 1), torch.cat(ks, 1), torch.cat(vs, 1)   # 토큰 축으로 concat
            q, k = apply_rope(q, position_ids), apply_rope(k, position_ids)

            if past_key_values is not None:
                if fill_kv_cache:              # prefix만 넣은 첫 호출: K/V 저장
                    past_key_values[layer_idx] = (k, v)
                else:                          # denoise 단계: [prefix 캐시; suffix] 로 확장
                    k = torch.cat([past_key_values[layer_idx][0], k], dim=1)
                    v = torch.cat([past_key_values[layer_idx][1], v], dim=1)
                    # [실제] DynamicCache에 append 후 denoise_step 끝에서 crop()으로 되돌림
            return [eager_attention(attention_mask, q, k, v)]   # 출력 1개 (나중에 스트림별로 자름)

        # ---------------- CA 층: expert는 VLM의 K/V만 참조 ----------------
        def forward_cross_attn_layer(self, model_layers, inputs_embeds, layer_idx, position_ids,
                                     attention_mask, past_key_values):
            prefix, suffix = inputs_embeds
            att_outputs = []
            if prefix is not None:             # 학습: prefix도 이 층을 통과 (prefix끼리 SA)
                L = prefix.shape[1]
                layer = model_layers[0][layer_idx]
                h = layer.input_layernorm(prefix)
                shape = (*h.shape[:-1], -1, self.head_dim)
                q = apply_rope(layer.self_attn.q_proj(h).view(shape), position_ids[:, :L])
                k = apply_rope(layer.self_attn.k_proj(h).view(shape), position_ids[:, :L])
                v = layer.self_attn.v_proj(h).view(shape)
                att_outputs.append(eager_attention(attention_mask[:, :L, :L], q, k, v))
                expert_position_id = position_ids[:, L:]
            else:                              # 추론: prefix K/V는 캐시에서 읽기만
                k, v = past_key_values[layer_idx]
                expert_position_id = position_ids
                att_outputs.append(None)

            ex = model_layers[1][layer_idx]
            h = ex.input_layernorm(suffix)
            eq = ex.self_attn.q_proj(h).view(*h.shape[:-1], -1, self.head_dim)
            # VLM의 K/V(B, Lk, H, D)를 펼쳐서 expert 전용 k/v_proj로 한 번 더 투영
            ek = ex.self_attn.k_proj(k.reshape(*k.shape[:2], -1)).view(*k.shape[:2], -1, self.head_dim)
            ev = ex.self_attn.v_proj(v.reshape(*v.shape[:2], -1)).view(*v.shape[:2], -1, self.head_dim)
            # expert의 위치는 0부터 다시 시작
            expert_position_id = expert_position_id - expert_position_id.min(dim=1, keepdim=True).values
            eq = apply_rope(eq, expert_position_id)
            # 마스크: suffix 행 × prefix 열만 → action끼리는 이 층에서 서로 보지 않음
            mask = attention_mask[:, -suffix.shape[1]:, : ek.shape[1]]
            att_outputs.append(eager_attention(mask, eq, ek, ev))
            return att_outputs

        # ---------------- 전체 forward ----------------
        def forward(self, attention_mask, position_ids, past_key_values, inputs_embeds, use_cache):
            model_layers = [self.vlm.layers, self.lm_expert_layers]
            fill_kv_cache = use_cache and past_key_values is None
            if fill_kv_cache:
                past_key_values = {}

            for layer_idx in range(self.num_vlm_layers):
                if fill_kv_cache or not self.is_cross_attn_layer(layer_idx):
                    att = self.forward_attn_layer(model_layers, inputs_embeds, layer_idx, position_ids,
                                                  attention_mask, past_key_values, fill_kv_cache)[0]
                    # concat 출력을 스트림별 길이로 다시 자름
                    lens = [h.shape[1] for h in inputs_embeds if h is not None]
                    parts = iter(att.split(lens, dim=1))
                    att_outputs = [next(parts) if h is not None else None for h in inputs_embeds]
                else:
                    att_outputs = self.forward_cross_attn_layer(model_layers, inputs_embeds, layer_idx,
                                                                position_ids, attention_mask, past_key_values)

                outputs_embeds = []
                for i, hidden_states in enumerate(inputs_embeds):
                    if hidden_states is None:
                        outputs_embeds.append(None)
                        continue
                    self.calls["vlm_layer" if i == 0 else "expert_layer"] += 1
                    layer = model_layers[i][layer_idx]
                    out = layer.self_attn.o_proj(att_outputs[i]) + hidden_states   # 각자의 o_proj
                    out = layer.mlp(layer.post_attention_layernorm(out)) + out
                    outputs_embeds.append(out)
                inputs_embeds = outputs_embeds

            norms = [self.vlm.norm, self.lm_expert_norm]
            outputs = [norms[i](h) if h is not None else None for i, h in enumerate(inputs_embeds)]
            return outputs, past_key_values
    return (MiniVLMWithExpert,)


@app.cell
def _(MiniVLMWithExpert, cfg, mo):
    _m = MiniVLMWithExpert(cfg)
    _rows = []
    for _i in range(_m.num_vlm_layers):
        _ex = _m.lm_expert_layers[_i].self_attn
        _rows.append({
            "layer": _i,
            "type": "CA (cross-attn)" if _m.is_cross_attn_layer(_i) else "SA (self-attn)",
            "VLM q_proj (out, in)": tuple(_m.vlm.layers[_i].self_attn.q_proj.weight.shape),
            "expert q_proj (out, in)": tuple(_ex.q_proj.weight.shape),
            "expert k_proj (out, in)": tuple(_ex.k_proj.weight.shape),
        })
    _vlm_n = sum(p.numel() for p in _m.vlm.parameters())
    _exp_n = sum(p.numel() for p in _m.lm_expert_layers.parameters())
    _train_n = sum(p.numel() for p in _m.parameters() if p.requires_grad)
    mo.vstack([
        mo.md(f"""
    - VLM 레이어: {cfg.total_vlm_layers}개 중 **{_m.num_vlm_layers}개만** 남음 (레이어 스킵)
    - expert hidden: {cfg.vlm_hidden} × {cfg.expert_width_multiplier} = **{_m.expert_hidden_size}**
    - 파라미터: VLM {_vlm_n:,} / expert {_exp_n:,} / **학습 대상 {_train_n:,}** (VLM은 freeze)

    표에서 expert의 `k_proj` 입력 차원을 보세요. SA 층은 자기 hidden({_m.expert_hidden_size}),
    CA 층은 **VLM의 K 차원({cfg.vlm_hidden})** 을 입력으로 받습니다.
    """),
        mo.ui.table(_rows, selection=None),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Suffix: 노이즈 행동 + 시간 임베딩

    `embed_suffix`는 노이즈 섞인 행동 `x_t`와 flow 시간 `t`를 하나의 토큰열로 만듭니다.

    ```
    action_emb = action_in_proj(x_t)                         # (B, 50, d_expert)
    time_emb   = sinusoidal(t)  → 50개 토큰에 복사             # (B, 50, d_expert)
    x          = mlp_out(silu(mlp_in(concat(action_emb, time_emb))))
    ```

    시간은 [0, 1] 범위의 연속값이라 `min_period=4e-3 ~ max_period=4.0`으로 주기를 잡아 작은 차이도 구분되게 합니다.
    """)
    return


@app.cell
def _(cfg, math, plt, torch):
    def create_sinusoidal_pos_embedding(time, dimension, min_period, max_period):
        """lerobot과 동일 (단, float64 대신 float32 사용)."""
        fraction = torch.linspace(0.0, 1.0, dimension // 2, device=time.device)
        period = min_period * (max_period / min_period) ** fraction
        scaling_factor = 1.0 / period * 2 * math.pi
        sin_input = time[..., None] * scaling_factor
        return torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=-1)

    _t = torch.linspace(0, 1, 101)
    _emb = create_sinusoidal_pos_embedding(_t, 48, cfg.min_period, cfg.max_period)
    _fig, _ax = plt.subplots(figsize=(8, 2.8))
    _ax.imshow(_emb.T, aspect="auto", cmap="RdBu", extent=[0, 1, 48, 0])
    _ax.set_xlabel("flow time t")
    _ax.set_ylabel("embedding dim")
    _ax.set_title("sinusoidal time embedding (top: short periods, bottom: long periods)")
    _fig.tight_layout()
    _fig
    return (create_sinusoidal_pos_embedding,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. Flow Matching

    SmolVLA(openpi 계열)는 **t = 1이 노이즈, t = 0이 정답 행동**인 규약(`FlowConvention.NOISE_AT_ONE`)을 씁니다.

    $$x_t = t\,\epsilon + (1-t)\,A, \qquad u_t = \epsilon - A, \qquad \mathcal{L} = \lVert v_\theta(x_t, t, o) - u_t \rVert^2$$

    - `sample_time`: $t \sim \text{Beta}(1.5, 1.0) \times 0.999 + 0.001$ → **노이즈 쪽(t≈1)을 더 자주** 학습
    - 손실은 next-token CE가 아니라 **속도 벡터에 대한 MSE** — forward 1번으로 계산
    - 추론: $t=1$의 노이즈에서 시작해 $x \leftarrow x + \Delta t \cdot v$, $\Delta t = -1/10$ 을 10번 반복
    """)
    return


@app.cell
def _(torch):
    def make_flow_matching_inputs(actions, noise, time):
        """NOISE_AT_ONE 규약: t=1 → 노이즈, t=0 → 정답"""
        t = time[:, None, None]
        x_t = t * noise + (1 - t) * actions
        u_t = noise - actions
        return x_t, u_t

    def euler_integrate(denoise_fn, noise, num_steps, return_all=False):
        """t: 1 → 0 으로 적분. [실제] lerobot.policies.common.flow_matching.euler_integrate (RTC 옵션 포함)"""
        dt = -1.0 / num_steps
        x_t, history = noise, [noise]
        for step in range(num_steps):
            time = torch.full((noise.shape[0],), 1.0 + step * dt, device=noise.device)
            v_t = denoise_fn(x_t, time)
            x_t = x_t + dt * v_t
            history.append(x_t)
        return (x_t, history) if return_all else x_t
    return euler_integrate, make_flow_matching_inputs


@app.cell
def _(make_flow_matching_inputs, plt, torch):
    # 직관 확인: 1D 정답 A=2, 노이즈 eps 여러 개 → x_t는 직선 경로, 속도는 t와 무관한 상수
    _beta = torch.distributions.Beta(1.5, 1.0).sample((20000,)) * 0.999 + 0.001
    _A = torch.full((5, 1, 1), 2.0)
    _eps = torch.randn(5, 1, 1)
    _ts = torch.linspace(0, 1, 21)
    _paths = torch.stack([make_flow_matching_inputs(_A, _eps, torch.full((5,), float(_tt)))[0][:, 0, 0] for _tt in _ts])

    _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(10, 3.2))
    _ax1.hist(_beta.numpy(), bins=50, color="tab:blue")
    _ax1.set_title("sample_time: Beta(1.5,1)*0.999+0.001")
    _ax1.set_xlabel("t  (1 = noise)")
    _ax2.plot(_ts, _paths, marker=".")
    _ax2.axhline(2.0, color="k", ls="--", lw=1, label="action A=2")
    _ax2.invert_xaxis()
    _ax2.set_xlabel("t  (integrate 1 → 0)")
    _ax2.set_ylabel("x_t")
    _ax2.set_title("straight paths from noise to action")
    _ax2.legend()
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 6.1 `VLAFlowMatching` — 모든 조각 합치기

    lerobot의 `VLAFlowMatching`과 같은 메서드 구성입니다. 눈여겨볼 곳:

    - `embed_prefix`: 이미지/언어 임베딩에 `× sqrt(dim)` 스케일, **state는 `state_proj` 선형층 → 토큰 1개**
    - `forward` (학습): prefix + suffix를 **한 번에** VLM+expert에 통과, `position_ids = cumsum(pad_masks) - 1`
    - `sample_actions` (추론): prefix만 먼저 통과시켜 **KV 캐시 생성** → `denoise_step`을 10번 호출
    - `denoise_step`: 마스크는 `(suffix 길이) × (prefix + suffix 길이)`, suffix 위치는 prefix 길이에서 이어짐
    """)
    return


@app.cell
def _(
    F,
    MiniVLMWithExpert,
    create_sinusoidal_pos_embedding,
    euler_integrate,
    make_att_2d_masks,
    make_flow_matching_inputs,
    math,
    nn,
    torch,
):
    class MiniVLAFlowMatching(nn.Module):
        def __init__(self, c):
            super().__init__()
            self.config = c
            self.vlm_with_expert = MiniVLMWithExpert(c)
            eh = self.vlm_with_expert.expert_hidden_size
            self.state_proj = nn.Linear(c.max_state_dim, c.vlm_hidden)       # state → VLM 토큰 1개
            self.action_in_proj = nn.Linear(c.max_action_dim, eh)
            self.action_out_proj = nn.Linear(eh, c.max_action_dim)
            self.action_time_mlp_in = nn.Linear(2 * eh, eh)
            self.action_time_mlp_out = nn.Linear(eh, eh)

        def sample_noise(self, shape, device):
            return torch.randn(shape, device=device)

        def sample_time(self, bsize, device):
            t = torch.distributions.Beta(1.5, 1.0).sample((bsize,))
            return (t * 0.999 + 0.001).to(device)

        def embed_prefix(self, images, img_masks, lang_tokens, lang_masks, state):
            embs, pad_masks, att_masks = [], [], []
            for img, img_mask in zip(images, img_masks):
                img_emb = self.vlm_with_expert.embed_image(img)
                img_emb = img_emb * math.sqrt(img_emb.shape[-1])
                bsize, n = img_emb.shape[:2]
                embs.append(img_emb)
                pad_masks.append(img_mask[:, None].expand(bsize, n))
                att_masks += [0] * n                              # 이미지: 블록 0
            lang_emb = self.vlm_with_expert.embed_language_tokens(lang_tokens)
            embs.append(lang_emb * math.sqrt(lang_emb.shape[-1]))
            pad_masks.append(lang_masks)
            att_masks += [0] * lang_emb.shape[1]                  # 언어: 블록 0 (이미지와 양방향)

            state_emb = self.state_proj(state)[:, None, :]        # (B, 1, d)
            embs.append(state_emb)
            pad_masks.append(torch.ones(bsize, 1, dtype=torch.bool, device=state.device))
            att_masks += [1]                                      # state: 새 블록

            embs = torch.cat(embs, dim=1)
            pad_masks = torch.cat(pad_masks, dim=1)
            att_masks = torch.tensor(att_masks, dtype=torch.bool, device=state.device)[None].expand(bsize, -1)
            return embs, pad_masks, att_masks

        def embed_suffix(self, noisy_actions, timestep):
            action_emb = self.action_in_proj(noisy_actions)
            time_emb = create_sinusoidal_pos_embedding(
                timestep, self.vlm_with_expert.expert_hidden_size, self.config.min_period, self.config.max_period
            )
            time_emb = time_emb[:, None, :].expand_as(action_emb)
            x = self.action_time_mlp_in(torch.cat([action_emb, time_emb], dim=2))
            x = self.action_time_mlp_out(F.silu(x))
            bsize, n = x.shape[:2]
            pad_masks = torch.ones(bsize, n, dtype=torch.bool, device=x.device)
            att_masks = torch.ones(bsize, n, dtype=torch.bool, device=x.device)   # action: 토큰마다 새 블록 → causal
            return x, pad_masks, att_masks

        def predict_velocity(self, images, img_masks, lang_tokens, lang_masks, state, x_t, time):
            """forward의 앞부분 (캐시 없이 prefix+suffix 한 번에). 회귀 baseline에서 재사용하려고 분리."""
            prefix_embs, prefix_pad, prefix_att = self.embed_prefix(images, img_masks, lang_tokens, lang_masks, state)
            suffix_embs, suffix_pad, suffix_att = self.embed_suffix(x_t, time)
            pad_masks = torch.cat([prefix_pad, suffix_pad], dim=1)
            att_masks = torch.cat([prefix_att, suffix_att], dim=1)
            att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
            position_ids = torch.cumsum(pad_masks, dim=1) - 1
            (_, suffix_out), _ = self.vlm_with_expert.forward(
                attention_mask=att_2d_masks, position_ids=position_ids, past_key_values=None,
                inputs_embeds=[prefix_embs, suffix_embs], use_cache=False,
            )
            return self.action_out_proj(suffix_out[:, -self.config.chunk_size:])

        def forward(self, images, img_masks, lang_tokens, lang_masks, state, actions, noise=None, time=None):
            """학습: per-element loss (B, chunk, max_action_dim) 반환"""
            if noise is None:
                noise = self.sample_noise(actions.shape, actions.device)
            if time is None:
                time = self.sample_time(actions.shape[0], actions.device)
            x_t, u_t = make_flow_matching_inputs(actions, noise, time)
            v_t = self.predict_velocity(images, img_masks, lang_tokens, lang_masks, state, x_t, time)
            return F.mse_loss(u_t, v_t, reduction="none")

        @torch.no_grad()
        def sample_actions(self, images, img_masks, lang_tokens, lang_masks, state, noise=None, return_all=False):
            bsize = state.shape[0]
            if noise is None:
                noise = self.sample_noise((bsize, self.config.chunk_size, self.config.max_action_dim), state.device)
            prefix_embs, prefix_pad, prefix_att = self.embed_prefix(images, img_masks, lang_tokens, lang_masks, state)
            prefix_att_2d = make_att_2d_masks(prefix_pad, prefix_att)
            prefix_pos = torch.cumsum(prefix_pad, dim=1) - 1
            # ① prefix만 1번 통과 → 층별 K/V 캐시
            _, past_key_values = self.vlm_with_expert.forward(
                attention_mask=prefix_att_2d, position_ids=prefix_pos, past_key_values=None,
                inputs_embeds=[prefix_embs, None], use_cache=True,
            )
            # ② expert만 num_steps번 실행
            return euler_integrate(
                lambda x_t, t: self.denoise_step(prefix_pad, past_key_values, x_t, t),
                noise, self.config.num_steps, return_all=return_all,
            )

        def denoise_step(self, prefix_pad_masks, past_key_values, x_t, timestep):
            suffix_embs, suffix_pad, suffix_att = self.embed_suffix(x_t, timestep)
            bsize, prefix_len = prefix_pad_masks.shape
            suffix_len = suffix_pad.shape[1]
            prefix_pad_2d = prefix_pad_masks[:, None, :].expand(bsize, suffix_len, prefix_len)
            suffix_att_2d = make_att_2d_masks(suffix_pad, suffix_att)
            full_att_2d = torch.cat([prefix_pad_2d, suffix_att_2d], dim=2)        # (B, Ls, Lp + Ls)
            prefix_offsets = prefix_pad_masks.sum(dim=-1)[:, None]
            position_ids = prefix_offsets + torch.cumsum(suffix_pad, dim=1) - 1   # prefix 다음 위치부터
            (_, suffix_out), _ = self.vlm_with_expert.forward(
                attention_mask=full_att_2d, position_ids=position_ids, past_key_values=past_key_values,
                inputs_embeds=[None, suffix_embs], use_cache=True,
            )
            return self.action_out_proj(suffix_out[:, -self.config.chunk_size:])
    return (MiniVLAFlowMatching,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6.2 장난감 과제: "장애물을 피해 목표로 가기"

    | 입력 | 내용 |
    |---|---|
    | 이미지 (32×32) | 경로 한가운데에 **흰 사각형 장애물**이 있거나 없음 |
    | 언어 | `"go right"` (목표 (1, 0)) 또는 `"go up"` (목표 (0, 1)) |
    | state | 시작 위치 (원점 근처, 약간의 노이즈) |
    | action chunk | 앞으로 16스텝의 (x, y) 웨이포인트 |

    장애물이 있으면 시연자는 **위로 돌아가거나 아래로 돌아갑니다 (50:50)** → 정답 분포가 **다봉(multimodal)** 입니다.
    모델은 **언어로 목표를, 이미지로 장애물 유무를** 알아내야 합니다.

    데이터 제작은 보조 함수에 맡깁니다. 여기서는 아래 텐서가 모델의 어느 입력으로 들어가는지 확인하세요.

    | 텐서 | shape (`B`: 배치 크기) | 연결 |
    |---|---|---|
    | `images[0]` | `(B, 3, 32, 32)` | 이미지 → connector → prefix |
    | `lang_tokens` | `(B, 2)` | 언어 → prefix |
    | `state` | `(B, max_state_dim)` | `state_proj` → prefix 토큰 1개 |
    | `actions` | `(B, chunk_size, max_action_dim)` | 노이즈와 섞어 suffix 입력 및 속도 정답 구성 |

    핵심은 장애물을 그리는 방법보다 **동일한 조건에서도 우회 행동이 두 갈래**라는 점입니다.

    > [실제] lerobot은 state/action을 데이터셋 통계로 정규화한 뒤 `max_*_dim`까지 0으로 패딩합니다. 여기서는 패딩만 합니다.
    """)
    return


@app.cell
def _(cfg):
    from smolvla_helpers import make_toy_batch

    def make_batch(bsize, device, lang=None, obstacle=None):
        return make_toy_batch(cfg, bsize, device, lang=lang, obstacle=obstacle)

    return (make_batch,)


@app.cell(hide_code=True)
def _(cfg, make_batch, plt):
    _b = make_batch(64, "cpu")
    _fig, _axes = plt.subplots(1, 3, figsize=(11, 3.4))
    for _k in range(2):
        _idx = (_b["obstacle"] == bool(_k)).nonzero().flatten()[0].item()
        _axes[_k].imshow(_b["images"][0][_idx].permute(1, 2, 0), extent=[-0.5, 1.5, -0.5, 1.5])
        _axes[_k].set_title(f"image (obstacle={bool(_k)}), lang={'right' if _b['lang_id'][_idx] == 0 else 'up'}")
    _a = _b["actions"][..., : cfg.action_dim]
    for _j in range(64):
        _c = "tab:red" if _b["obstacle"][_j] else "tab:gray"
        _axes[2].plot(_a[_j, :, 0], _a[_j, :, 1], color=_c, alpha=0.5, lw=1)
    _axes[2].set_aspect("equal")
    _axes[2].set_title("demo action chunks (red: with obstacle)")
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6.3 학습: Flow Matching vs 단순 회귀(MSE)

    **같은 네트워크**를 두 가지 방식으로 학습합니다.
    회귀 baseline은 생성 방식의 차이를 관찰하기 위한 비교 실험입니다.
    아래 두 손실 함수를 비교하고, **Flow Matching도 MSE를 쓰지만 정답이 행동이 아닌 속도**라는 점을 확인하세요.
    optimizer·스케줄러·gradient clipping 루프는 `smolvla_helpers.train_model`에 분리했습니다.

    - **Flow matching** (SmolVLA): `losses = model.forward(...)` — 위에서 만든 그대로
    - **회귀 baseline**: 입력 `x_t`를 0으로, `t=1`로 고정하고 출력이 바로 행동이 되도록 MSE 학습

    VLM은 freeze되어 있으므로 실제로 학습되는 것은 expert, `state_proj`, action/time 투영층뿐입니다.
    (여기서는 VLM이 랜덤 초기화 상태지만, 고정된 랜덤 특징만으로도 이 과제는 풀립니다.)
    """)
    return


@app.cell
def _(mo):
    train_steps = mo.ui.slider(300, 3000, value=1500, step=100, label="학습 스텝")
    train_button = mo.ui.run_button(label="▶ 두 모델 학습 시작")
    mo.hstack([train_steps, train_button], justify="start")
    return train_button, train_steps


@app.cell
def _(cfg, torch):
    def flow_loss(model, b):
        losses = model.forward(b["images"], b["img_masks"], b["lang_tokens"], b["lang_masks"], b["state"], b["actions"])
        return losses[:, :, : cfg.action_dim].mean()      # 패딩된 차원은 loss에서 제외

    def regression_loss(model, b):
        zeros = torch.zeros_like(b["actions"])
        ones = torch.ones(zeros.shape[0], device=zeros.device)
        pred = model.predict_velocity(b["images"], b["img_masks"], b["lang_tokens"], b["lang_masks"], b["state"], zeros, ones)
        return ((pred - b["actions"])[:, :, : cfg.action_dim] ** 2).mean()

    return flow_loss, regression_loss


@app.cell
def _(MiniVLAFlowMatching, cfg, device, flow_loss, make_batch, mo, regression_loss, torch, train_button, train_steps):
    from smolvla_helpers import train_model

    mo.stop(not train_button.value, mo.md("⬆️ **학습 시작** 버튼을 누르세요. (GPU 수십 초, CPU 약 2~3분)"))
    if device.type == "cpu":
        torch.set_num_threads(min(4, torch.get_num_threads()))

    torch.manual_seed(0)
    flow_model = MiniVLAFlowMatching(cfg).to(device)
    flow_hist = train_model(
        flow_model, flow_loss, make_batch, device, train_steps.value,
        mo.status.progress_bar(range(train_steps.value), title="flow matching 학습"),
    )
    torch.manual_seed(0)
    reg_model = MiniVLAFlowMatching(cfg).to(device)
    reg_hist = train_model(
        reg_model, regression_loss, make_batch, device, train_steps.value,
        mo.status.progress_bar(range(train_steps.value), title="회귀 baseline 학습"),
    )
    return flow_hist, flow_model, reg_hist, reg_model


@app.cell(hide_code=True)
def _(flow_hist, np, plt, reg_hist):
    def _smooth(x, k=25):
        return np.convolve(x, np.ones(k) / k, mode="valid")

    _fig, _axes = plt.subplots(1, 2, figsize=(10, 3))
    _axes[0].plot(_smooth(flow_hist))
    _axes[0].set_title("flow matching loss  ||v - (eps - A)||^2")
    _axes[1].plot(_smooth(reg_hist), color="tab:red")
    _axes[1].set_title("regression loss  ||pred - A||^2")
    for _a in _axes:
        _a.set_yscale("log")
        _a.set_xlabel("step")
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(cfg, device, flow_model, make_batch, plt, reg_model, torch):
    torch.manual_seed(1)
    _fig, _axes = plt.subplots(2, 2, figsize=(9, 8))
    for _li, _lname in enumerate(["right", "up"]):
        for _oi, _obs in enumerate([False, True]):
            _ax = _axes[_oi, _li]
            _b = make_batch(30, device, lang=_li, obstacle=_obs)
            _flow = flow_model.sample_actions(_b["images"], _b["img_masks"], _b["lang_tokens"], _b["lang_masks"], _b["state"])
            with torch.no_grad():
                _reg = reg_model.predict_velocity(
                    _b["images"], _b["img_masks"], _b["lang_tokens"], _b["lang_masks"], _b["state"],
                    torch.zeros_like(_b["actions"]), torch.ones(30, device=device),
                )
            for _j in range(30):
                _f = _flow[_j, :, : cfg.action_dim].cpu()
                _r = _reg[_j, :, : cfg.action_dim].cpu()
                _ax.plot(_f[:, 0], _f[:, 1], color="tab:blue", alpha=0.4, lw=1, label="flow matching" if _j == 0 else None)
                _ax.plot(_r[:, 0], _r[:, 1], color="tab:red", alpha=0.6, lw=1, label="regression" if _j == 0 else None)
            if _obs:
                _ax.add_patch(plt.Circle((0.5, 0.0) if _li == 0 else (0.0, 0.5), 0.12, color="k", alpha=0.3))
            _ax.set_xlim(-0.6, 1.3)
            _ax.set_ylim(-0.6, 1.3)
            _ax.set_aspect("equal")
            _ax.set_title(f'lang="go {_lname}", obstacle={_obs}')
            _ax.legend(loc="lower left", fontsize=8)
    _fig.suptitle("30 sampled action chunks per condition")
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **관찰 포인트**

    - 장애물이 없을 때는 두 방법 모두 직선을 잘 냅니다.
    - 장애물이 있을 때, **회귀(빨강)는 "위로 돌기"와 "아래로 돌기"의 평균 → 장애물을 정면으로 통과**합니다 (mode averaging).
    - **Flow matching(파랑)은 노이즈에 따라 위 또는 아래 경로 중 하나를 고릅니다.** 이것이 VLA가 생성 모델을 쓰는 이유입니다.
    - 언어(`right`/`up`)와 이미지(장애물 유무)를 모두 반영하는지도 확인해 보세요.

    ## 7. 추론: KV 캐시 + Euler 적분

    아래 슬라이더로 10번의 적분 스텝 동안 노이즈가 행동 궤적으로 바뀌는 과정을 보세요.
    """)
    return


@app.cell
def _(cfg, mo):
    denoise_step_slider = mo.ui.slider(0, cfg.num_steps, value=0, label="Euler step k (t = 1 - k/10)", full_width=True)
    denoise_step_slider
    return (denoise_step_slider,)


@app.cell
def _(cfg, denoise_step_slider, device, flow_model, make_batch, plt, torch):
    torch.manual_seed(2)
    _b = make_batch(30, device, lang=0, obstacle=True)
    _, _hist = flow_model.sample_actions(
        _b["images"], _b["img_masks"], _b["lang_tokens"], _b["lang_masks"], _b["state"], return_all=True
    )
    _k = denoise_step_slider.value
    _x = _hist[_k][..., : cfg.action_dim].cpu()
    _fig, _ax = plt.subplots(figsize=(5, 4.5))
    for _j in range(30):
        _ax.plot(_x[_j, :, 0], _x[_j, :, 1], marker=".", ms=3, lw=0.8, alpha=0.6)
    _ax.add_patch(plt.Circle((0.5, 0.0), 0.12, color="k", alpha=0.3))
    _ax.set_xlim(-2.5, 2.5)
    _ax.set_ylim(-2.5, 2.5)
    _ax.set_aspect("equal")
    _ax.set_title(f"step {_k}/{cfg.num_steps}   t = {1 - _k / cfg.num_steps:.1f}")
    _fig
    return


@app.cell
def _(cfg, device, flow_model, make_batch, mo, time, torch):
    # KV 캐시 효과: 레이어 호출 횟수와 시간 비교
    _b = make_batch(64, device)
    _vwe = flow_model.vlm_with_expert

    _vwe.calls = {"vlm_layer": 0, "expert_layer": 0}
    _t0 = time.perf_counter()
    flow_model.sample_actions(_b["images"], _b["img_masks"], _b["lang_tokens"], _b["lang_masks"], _b["state"])
    _t_cache = time.perf_counter() - _t0
    _calls_cache = dict(_vwe.calls)

    # 캐시 없이: 매 step마다 prefix+suffix 전체를 다시 계산
    _vwe.calls = {"vlm_layer": 0, "expert_layer": 0}
    _t0 = time.perf_counter()
    with torch.no_grad():
        _x = torch.randn(64, cfg.chunk_size, cfg.max_action_dim, device=device)
        for _s in range(cfg.num_steps):
            _tt = torch.full((64,), 1.0 - _s / cfg.num_steps, device=device)
            _v = flow_model.predict_velocity(_b["images"], _b["img_masks"], _b["lang_tokens"], _b["lang_masks"], _b["state"], _x, _tt)
            _x = _x - _v / cfg.num_steps
    _t_nocache = time.perf_counter() - _t0
    _calls_nocache = dict(_vwe.calls)

    mo.md(f"""
    | | VLM 레이어 호출 | expert 레이어 호출 | 시간 |
    |---|---|---|---|
    | **KV 캐시 사용** (`sample_actions`) | {_calls_cache['vlm_layer']} (= {_vwe.num_vlm_layers}층 × **1회**) | {_calls_cache['expert_layer']} (= {_vwe.num_vlm_layers}층 × {cfg.num_steps}회) | {_t_cache * 1000:.1f} ms |
    | 캐시 없음 | {_calls_nocache['vlm_layer']} (= {_vwe.num_vlm_layers}층 × {cfg.num_steps}회) | {_calls_nocache['expert_layer']} | {_t_nocache * 1000:.1f} ms |

    실제 SmolVLA에서는 VLM이 expert보다 훨씬 크고(이미지 토큰도 많음) 이 차이가 추론 속도를 좌우합니다.
    prefix가 suffix를 보지 않도록 한 **마스크 설계(3절) 덕분에** 가능한 최적화입니다.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 8. Action Queue — `SmolVLAPolicy.select_action`

    로봇 제어 루프는 매 스텝 행동 **1개**를 요구합니다. `select_action`은 큐가 비었을 때만 chunk를 새로 예측하고,
    그 앞쪽 `n_action_steps`개를 큐에 넣어 하나씩 꺼내 줍니다.

    ```python
    # lerobot SmolVLAPolicy.select_action (요약)
    if len(self._queues[ACTION]) == 0:
        actions = self._get_action_chunk(batch)                         # (B, chunk, action_dim)
        self._queues[ACTION].extend(actions.transpose(0, 1)[: self.config.n_action_steps])
    return self._queues[ACTION].popleft()
    ```
    `n_action_steps`가 작을수록 새 관측을 더 자주 반영(반응성↑)하지만 추론 횟수가 늘어납니다.
    """)
    return


@app.cell
def _(mo):
    n_action_steps_slider = mo.ui.slider(1, 16, value=8, label="n_action_steps")
    n_action_steps_slider
    return (n_action_steps_slider,)


@app.cell
def _(cfg, deque, device, flow_model, make_batch, mo, n_action_steps_slider):
    class MiniPolicy:
        def __init__(self, model, n_action_steps):
            self.model, self.n_action_steps = model, n_action_steps
            self.queue = deque()
            self.num_inferences = 0

        def select_action(self, batch):
            if len(self.queue) == 0:
                actions = self.model.sample_actions(
                    batch["images"], batch["img_masks"], batch["lang_tokens"], batch["lang_masks"], batch["state"]
                )[:, :, : cfg.action_dim]
                self.queue.extend(actions.transpose(0, 1)[: self.n_action_steps])
                self.num_inferences += 1
            return self.queue.popleft()

    _policy = MiniPolicy(flow_model, n_action_steps_slider.value)
    _b = make_batch(1, device, lang=1, obstacle=False)
    _log = []
    for _step in range(32):
        _before = _policy.num_inferences
        _a = _policy.select_action(_b)
        if _policy.num_inferences > _before:
            _log.append(_step)
    mo.md(
        f"32 제어 스텝 동안 추론 **{_policy.num_inferences}회** — 추론이 일어난 스텝: `{_log}`"
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 9. (선택 심화) 비동기 추론 (Asynchronous Inference)

    모델 내부를 공부하는 경우 이 절은 건너뛰어도 됩니다.
    이 시뮬레이션은 신경망을 호출하지 않고 **행동 큐와 추론 지연**만 다룹니다.

    동기 방식에서는 큐가 빈 뒤에야 추론을 요청하므로 **추론 지연 동안 로봇이 멈춥니다.**
    비동기 방식(`RobotClient` ↔ `PolicyServer`)은 큐 잔량 비율이 **임계값 g (`chunk_size_threshold`) 이하**가 되면
    미리 관측을 보내고, 새 chunk가 도착하면 **이미 지나간 시점의 행동은 버리고** 나머지로 큐를 갱신합니다.

    아래는 이를 단순화한 시뮬레이션입니다 (1 tick = 행동 1개 실행 시간).
    """)
    return


@app.cell
def _(mo):
    async_g = mo.ui.slider(0.0, 1.0, value=0.6, step=0.05, label="임계값 g (chunk_size_threshold)")
    async_latency = mo.ui.slider(1, 40, value=15, label="추론 지연 (tick)")
    mo.hstack([async_g, async_latency], justify="start")
    return async_g, async_latency


@app.cell(hide_code=True)
def _(async_g, async_latency, deque, plt):
    def simulate(mode, g, latency, chunk=50, T=300):
        queue, pending = deque(), None          # pending = (도착 tick, 관측 시점의 로봇 step)
        robot_step, idle, n_req, q_hist = 0, 0, 0, []
        for tick in range(T):
            if pending is not None and tick >= pending[0]:
                obs_step = pending[1]
                new_chunk = range(obs_step, obs_step + chunk)               # 각 행동의 timestep
                queue = deque(a for a in new_chunk if a >= robot_step)      # 지난 행동은 버림, 겹치는 구간은 새 chunk로 교체
                pending = None
            need = len(queue) == 0 if mode == "sync" else len(queue) / chunk <= g
            if need and pending is None:
                pending = (tick + latency, robot_step)
                n_req += 1
            if queue:
                queue.popleft()
                robot_step += 1
            else:
                idle += 1
            q_hist.append(len(queue))
        return q_hist, idle, robot_step, n_req

    _fig, _ax = plt.subplots(figsize=(12, 3.2))
    for _mode, _c in [("sync", "tab:gray"), ("async", "tab:blue")]:
        _q, _idle, _done, _nreq = simulate(_mode, async_g.value, async_latency.value)
        _ax.plot(_q, color=_c, label=f"{_mode}: idle {_idle} ticks, actions {_done}, requests {_nreq}")
    _ax.axhline(async_g.value * 50, color="tab:blue", ls=":", lw=1, label="g × chunk_size")
    _ax.set_xlabel("tick")
    _ax.set_ylabel("action queue length")
    _ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1))
    _ax.set_title("sync vs async inference (chunk_size = 50)")
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    - **g = 0** → 큐가 빌 때만 요청 = 동기 방식과 동일
    - g를 키우면 idle이 사라지지만 **요청 횟수(=GPU 연산)** 가 늘어납니다. `g × chunk ≥ 지연`이면 idle이 0이 됩니다.
    - [실제] lerobot은 관측이 직전 것과 관절 공간에서 거의 같으면 요청을 건너뛰는 **관측 필터링**,
      겹치는 구간을 섞는 `aggregate_fn_name` 옵션도 제공합니다.

    ## 10. (선택) 실제 `lerobot/smolvla_base` 살펴보기

    `pip install "lerobot[smolvla]"` 가 되어 있다면 스위치를 켜서 실제 모델에서 같은 구조를 확인할 수 있습니다 (~1GB 다운로드).
    """)
    return


@app.cell
def _(mo):
    real_switch = mo.ui.switch(label="실제 smolvla_base 로드")
    real_switch
    return (real_switch,)


@app.cell
def _(mo, real_switch):
    mo.stop(not real_switch.value)
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    real_policy = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base")
    _vwe = real_policy.model.vlm_with_expert
    _text = _vwe.get_vlm_model().text_model
    _rows = []
    for _i, _el in enumerate(_vwe.lm_expert.layers):
        _is_sa = _i % _vwe.self_attn_every_n_layers == 0
        _rows.append({
            "layer": _i,
            "type": "SA" if _is_sa else "CA",
            "VLM q_proj": tuple(_text.layers[_i].self_attn.q_proj.weight.shape),
            "expert q_proj": tuple(_el.self_attn.q_proj.weight.shape),
            "expert k_proj": tuple(_el.self_attn.k_proj.weight.shape),
        })

    def _count(m, trainable=False):
        return sum(p.numel() for p in m.parameters() if (p.requires_grad or not trainable))

    mo.vstack([
        mo.md(f"""
    - VLM 텍스트 레이어 수: **{len(_text.layers)}** (원래 32층 중 앞쪽 절반)
    - VLM hidden / expert hidden: **{_vwe.config.text_config.hidden_size} / {_vwe.expert_hidden_size}**
    - 전체 파라미터: **{_count(real_policy) / 1e6:.0f}M**, 그중 학습 대상: **{_count(real_policy, True) / 1e6:.0f}M**
    - vision encoder: {_count(_vwe.get_vlm_model().vision_model) / 1e6:.0f}M,
      action expert: {_count(_vwe.lm_expert) / 1e6:.0f}M
    """),
        mo.ui.table(_rows, selection=None),
    ])
    return SmolVLAPolicy, real_policy


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 연습 문제

    1. **마스크 실험** — `embed_suffix`의 `att_masks`를 전부 `0`으로 바꾸면(action끼리 양방향) 3절의 그림이 어떻게 변하나요?
       `embed_prefix`에서 state의 att_mask를 `0`으로 바꾸면?
    2. **CA vs SA** — `MiniConfig(self_attn_every_n_layers=1)`(모든 층 SA)로 바꿔 다시 학습해 보세요.
       4.2절 표에서 expert `k_proj` 모양이 어떻게 바뀌나요?
    3. **적분 스텝 수** — 학습된 모델에서 `flow_model.config.num_steps`를 1, 3, 10으로 바꿔 샘플을 비교해 보세요.
       1스텝이면 회귀 모델과 비슷해지는 이유는?
    4. **레이어 스킵** — `num_vlm_layers=2`로 줄이면 expert 레이어 수와 학습 파라미터 수가 어떻게 바뀌나요?
    5. **(선택 심화) 비동기 추론** — 추론 지연이 30 tick일 때 idle을 0으로 만드는 최소 g는? 왜 그 값인가요?
    6. **코드 읽기** — 실제 `smolvlm_with_expert.py`의 `denoise_step` 끝에 있는 `layer.crop(...)`은 왜 필요할까요?
       이 노트북의 `forward_attn_layer`는 왜 crop이 필요 없나요?
    """)
    return


if __name__ == "__main__":
    app.run()
