"""SmolVLA 실습용 보조 구현.

범용 Transformer 연산, 합성 데이터 생성, optimizer 루프를 모아 둡니다.
SmolVLA의 마스크, pixel shuffle, action expert, flow matching은 노트북에 있습니다.
이 파일은 교육용 미니 구현이며 사전학습된 SmolVLA가 아닙니다.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F


def apply_rope(x, positions, max_wavelength=10_000):
    """lerobot 코드와 동일. x: (B, L, H, D), positions: (B, L)"""
    d_half = x.shape[-1] // 2
    dtype = x.dtype
    x = x.to(torch.float32)
    freq_exponents = (2.0 / x.shape[-1]) * torch.arange(d_half, dtype=torch.float32, device=x.device)
    timescale = max_wavelength**freq_exponents
    radians = positions[..., None].to(torch.float32) / timescale[None, None, :]
    radians = radians[..., None, :]
    sin, cos = torch.sin(radians), torch.cos(radians)
    x1, x2 = x.split(d_half, dim=-1)
    res = torch.empty_like(x)
    res[..., :d_half] = x1 * cos - x2 * sin
    res[..., d_half:] = x2 * cos + x1 * sin
    return res.to(dtype)


def eager_attention(attention_mask, q, k, v):
    """q: (B, Lq, H, D), k/v: (B, Lk, H, D), attention_mask: (B, Lq, Lk) bool.
    [실제] eager_attention_forward는 GQA(KV head 5개 → 15개로 복제)도 처리함."""
    B, Lq, H, D = q.shape
    q, k, v = (t.transpose(1, 2) for t in (q, k, v))           # (B, H, L, D)
    att = (q @ k.transpose(-1, -2)) * D**-0.5
    att = att.masked_fill(~attention_mask[:, None], torch.finfo(att.dtype).min)
    out = att.softmax(dim=-1) @ v
    return out.transpose(1, 2).reshape(B, Lq, H * D)


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class MiniDecoderLayer(nn.Module):
    def __init__(self, hidden, num_heads, head_dim):
        super().__init__()
        attn_dim = num_heads * head_dim
        self.input_layernorm = RMSNorm(hidden)
        self.self_attn = nn.Module()
        self.self_attn.head_dim = head_dim
        self.self_attn.q_proj = nn.Linear(hidden, attn_dim, bias=False)
        self.self_attn.k_proj = nn.Linear(hidden, attn_dim, bias=False)
        self.self_attn.v_proj = nn.Linear(hidden, attn_dim, bias=False)
        self.self_attn.o_proj = nn.Linear(attn_dim, hidden, bias=False)
        self.post_attention_layernorm = RMSNorm(hidden)
        self.mlp = nn.Sequential(nn.Linear(hidden, 4 * hidden), nn.SiLU(), nn.Linear(4 * hidden, hidden))


class MiniVisionModel(nn.Module):
    """[실제] SigLIP ViT. 여기서는 patchify + 인코더 1층."""
    def __init__(self, c, dim=32):
        super().__init__()
        self.patch_embed = nn.Conv2d(3, dim, c.patch_size, c.patch_size)
        self.encoder = nn.TransformerEncoderLayer(dim, 4, 4 * dim, dropout=0.0, batch_first=True)

    def forward(self, pixel_values):
        x = self.patch_embed(pixel_values).flatten(2).transpose(1, 2)    # (B, N패치, dim)
        return self.encoder(x)


def make_toy_batch(cfg, bsize, device, lang=None, obstacle=None):
    """lang: 0=right, 1=up / obstacle: bool. None이면 랜덤."""
    H = cfg.chunk_size
    lang_id = torch.randint(0, 2, (bsize,)) if lang is None else torch.full((bsize,), lang)
    obs = torch.randint(0, 2, (bsize,)).bool() if obstacle is None else torch.full((bsize,), obstacle)
    start = 0.05 * torch.randn(bsize, 2)
    goal = torch.where(lang_id[:, None] == 0, torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0]))
    direction = goal - start
    perp = torch.stack([-direction[:, 1], direction[:, 0]], dim=1)
    perp = perp / perp.norm(dim=1, keepdim=True)
    side = torch.randint(0, 2, (bsize,)) * 2 - 1                      # 위로/아래로 돌아가기
    t = torch.arange(1, H + 1) / H
    bump = 0.4 * torch.sin(math.pi * t)[None] * (side * obs)[:, None]
    traj = start[:, None] + t[None, :, None] * direction[:, None] + bump[..., None] * perp[:, None]

    images = torch.zeros(bsize, 3, cfg.image_size, cfg.image_size)
    for b in torch.nonzero(obs).flatten().tolist():
        mx, my = (0.5, 0.0) if lang_id[b] == 0 else (0.0, 0.5)        # 경로 중간
        col = int((mx + 0.5) / 2 * cfg.image_size)
        row = int((1.5 - my) / 2 * cfg.image_size)
        images[b, :, row - 5 : row + 5, col - 5 : col + 5] = 1.0         # 흰 사각형

    return {
        "images": [images.to(device)],                                # 카메라 1대
        "img_masks": [torch.ones(bsize, dtype=torch.bool, device=device)],
        "lang_tokens": torch.stack([torch.ones_like(lang_id), lang_id + 2], dim=1).to(device),
        "lang_masks": torch.ones(bsize, 2, dtype=torch.bool, device=device),
        "state": F.pad(start, (0, cfg.max_state_dim - 2)).to(device),
        "actions": F.pad(traj, (0, cfg.max_action_dim - 2)).to(device),
        "obstacle": obs,
        "lang_id": lang_id,
    }


def train_model(model, loss_fn, make_batch, device, steps, progress):
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=3e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    history = []
    for _ in progress:
        loss = loss_fn(model, make_batch(64, device))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 10.0)
        opt.step()
        sched.step()
        history.append(loss.item())
    return history

