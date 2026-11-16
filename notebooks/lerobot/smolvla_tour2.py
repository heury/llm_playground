import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # SmolVLA 행동 전문가 단순화 스케치와 샘플 테스트

    [SmolVLA 구조 해부](smolvla_tour.py)의 행동 전문가, Flow Matching, 블록 마스크, KV 캐시를
    짧은 코드로 다시 쓴 스케치입니다. 정의 셀마다 **작은 텐서로 바로 확인하는 테스트 셀**이 붙어 있습니다.
    모든 셀은 CPU에서 실행되며 다운로드가 없습니다. 장난감 학습만 버튼으로 실행합니다.
    `notebooks/lerobot`에서 `uv run marimo edit smolvla_tour2.py`로 엽니다.

    이해를 돕기 위한 단순화이며 원본과 다음이 다릅니다.

    | 항목 | 이 스케치 | 원본 SmolVLA ([smolvla_tour.py](smolvla_tour.py)) |
    | --- | --- | --- |
    | 시간 규약 | $t=0$ 노이즈 → $t=1$ 행동, `dt = +1/N` | $t=1$ 노이즈 → $t=0$ 행동, `dt = -1/N` (7절) |
    | 시간 표집 | 균등분포 | Beta(1.5, 1) (`sample_time`) |
    | 층 구성 | 블록마다 Cross + Self 둘 다 | Self 층과 Cross 층 교대 (5절) |
    | 시간 결합 | 행동 임베딩에 더함 | 행동 임베딩과 이어 붙여 MLP 통과 |
    | RoPE·멀티헤드 | 생략 (혼합 층은 단일 헤드) | 적용 |
    """)
    return


@app.cell
def _():
    import math

    import matplotlib.font_manager as font_manager
    import matplotlib.pyplot as plt
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    _installed = {f.name for f in font_manager.fontManager.ttflist}
    for _name in ["Noto Sans CJK KR", "NanumGothic", "Malgun Gothic", "AppleGothic", "Noto Sans Mono CJK KR"]:
        if _name in _installed:
            plt.rcParams["font.family"] = _name
            break
    plt.rcParams["axes.unicode_minus"] = False
    return F, math, nn, plt, torch


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. 시간 임베딩

    스칼라 시간 $t \in [0, 1]$을 여러 주기의 sin/cos로 펼쳐 벡터로 만듭니다.
    원본 `create_sinusoidal_pos_embedding`과 같은 주기 범위(4e-3 ~ 4.0)를 씁니다.
    """)
    return


@app.cell
def _(math, torch):
    def sinusoidal_embedding(timestep: torch.Tensor, dim: int, min_period: float = 4e-3, max_period: float = 4.0):
        """timestep: (B,) → (B, dim). 앞 절반은 sin, 뒤 절반은 cos."""
        fraction = torch.linspace(0.0, 1.0, dim // 2, device=timestep.device)
        period = min_period * (max_period / min_period) ** fraction
        angle = timestep[:, None] * (2 * math.pi / period)
        return torch.cat([torch.sin(angle), torch.cos(angle)], dim=-1)

    return (sinusoidal_embedding,)


@app.cell
def _(mo, plt, sinusoidal_embedding, torch):
    # 테스트: 모양과 서로 다른 t가 서로 다른 벡터가 되는지 확인
    _t = torch.tensor([0.0, 0.25, 0.5, 1.0])
    _emb = sinusoidal_embedding(_t, 16)
    assert _emb.shape == (4, 16)
    assert torch.allclose(_emb[0, :8], torch.zeros(8)) and torch.allclose(_emb[0, 8:], torch.ones(8))  # t=0 → sin 0, cos 1

    _fig, _ax = plt.subplots(figsize=(7, 2.2))
    _im = _ax.imshow(_emb.numpy(), aspect="auto", cmap="RdBu", vmin=-1, vmax=1)
    _ax.set_yticks(range(4), [f"t={v:.2f}" for v in _t.tolist()])
    _ax.set_xlabel("임베딩 차원 (앞 8: sin, 뒤 8: cos)")
    _fig.colorbar(_im, ax=_ax)
    _fig.tight_layout()
    mo.vstack([mo.md(f"`sinusoidal_embedding(t, 16)` 모양: `{tuple(_emb.shape)}`, t=0 검사 통과"), _fig])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. 행동 전문가 블록

    한 블록은 (1) VLM 문맥을 보는 Cross-Attention, (2) 행동 청크 안의 인과적 Self-Attention, (3) MLP로 이루어집니다.
    """)
    return


@app.cell
def _(nn, torch):
    class ActionExpertBlock(nn.Module):
        def __init__(self, hidden_dim: int, vlm_dim: int, num_heads: int):
            super().__init__()
            self.norm1 = nn.RMSNorm(hidden_dim)
            # 1. VLM의 시각/언어/상태 컨텍스트(KV Cache)를 참조하는 Cross-Attention
            self.cross_attn = nn.MultiheadAttention(
                embed_dim=hidden_dim, kdim=vlm_dim, vdim=vlm_dim,
                num_heads=num_heads, batch_first=True
            )

            self.norm2 = nn.RMSNorm(hidden_dim)
            # 2. Action Chunk 내부의 시간적(Temporal) 관계를 학습하는 Causal Self-Attention
            self.self_attn = nn.MultiheadAttention(
                embed_dim=hidden_dim, num_heads=num_heads, batch_first=True
            )

            self.norm3 = nn.RMSNorm(hidden_dim)
            self.mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 4),
                nn.SiLU(),
                nn.Linear(hidden_dim * 4, hidden_dim)
            )

        def forward(self, x: torch.Tensor, vlm_kv_cache: tuple[torch.Tensor, torch.Tensor]):
            vlm_k, vlm_v = vlm_kv_cache

            # (1) Cross-Attention: Query는 액션(x), Key/Value는 VLM의 중간 레이어 출력
            h = self.norm1(x)
            attn_out, _ = self.cross_attn(query=h, key=vlm_k, value=vlm_v)
            x = x + attn_out

            # (2) Causal Self-Attention: 미래 액션 스텝 간의 일관성 유지
            #     nn.MultiheadAttention의 bool 마스크는 True가 "보지 못함"입니다.
            h = self.norm2(x)
            seq_len = x.size(1)
            causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
            attn_out, _ = self.self_attn(query=h, key=h, value=h, attn_mask=causal_mask)
            x = x + attn_out

            # (3) Feed-Forward Network
            x = x + self.mlp(self.norm3(x))
            return x

    return (ActionExpertBlock,)


@app.cell
def _(ActionExpertBlock, mo, torch):
    # 테스트: 배치 2, 행동 청크 6스텝, 전문가 폭 48, VLM 폭 64, VLM 토큰 5개
    torch.manual_seed(0)
    _block = ActionExpertBlock(hidden_dim=48, vlm_dim=64, num_heads=4).eval()
    _x = torch.randn(2, 6, 48)
    _kv = (torch.randn(2, 5, 64), torch.randn(2, 5, 64))
    with torch.no_grad():
        _y = _block(_x, _kv)
        # 인과성 검사: 마지막 행동 스텝만 바꾸면 앞쪽 5스텝 출력은 그대로여야 합니다.
        _x2 = _x.clone()
        _x2[:, -1] += 10.0
        _y2 = _block(_x2, _kv)
        # 문맥 의존 검사: VLM K/V를 바꾸면 모든 스텝 출력이 바뀝니다.
        _y3 = _block(_x, (torch.randn(2, 5, 64), torch.randn(2, 5, 64)))
    _past_same = torch.allclose(_y[:, :-1], _y2[:, :-1], atol=1e-5)
    _last_diff = not torch.allclose(_y[:, -1], _y2[:, -1])
    _ctx_diff = (_y - _y3).abs().amax(dim=-1).min().item() > 1e-4
    assert _y.shape == _x.shape and _past_same and _last_diff and _ctx_diff

    mo.md(f"""
    | 검사 | 결과 |
    | --- | --- |
    | 입력 → 출력 모양 | `{tuple(_x.shape)}` → `{tuple(_y.shape)}` |
    | 마지막 스텝을 바꿔도 앞 5스텝 출력 불변 (인과 마스크) | {_past_same} |
    | 마지막 스텝 출력은 변함 | {_last_diff} |
    | VLM K/V를 바꾸면 모든 스텝 출력이 변함 (Cross-Attention) | {_ctx_diff} |
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. 행동 전문가 전체

    노이즈 섞인 행동 $x_t$와 시간 $t$를 받아, 블록마다 대응하는 VLM 층의 K/V를 참조해 속도 $v_\theta$를 출력합니다.
    """)
    return


@app.cell
def _(ActionExpertBlock, nn, sinusoidal_embedding, torch):
    class SmolVLAActionExpert(nn.Module):
        def __init__(self, action_dim: int, chunk_size: int, hidden_dim: int = 720, vlm_dim: int = 960, num_layers: int = 16, num_heads: int = 12):
            super().__init__()
            self.chunk_size = chunk_size

            # 노이즈가 섞인 액션(x_t)과 타임스텝(t) 임베딩을 결합하는 입력 프로젝션
            self.action_in_proj = nn.Linear(action_dim, hidden_dim)
            self.time_mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )

            # VLM의 첫 N개 레이어에서 추출한 KV 캐시와 1:1 매칭되는 트랜스포머 블록들
            self.layers = nn.ModuleList([
                ActionExpertBlock(hidden_dim=hidden_dim, vlm_dim=vlm_dim, num_heads=num_heads)
                for _ in range(num_layers)
            ])

            self.final_norm = nn.RMSNorm(hidden_dim)
            # 최종 속도 벡터(Velocity field) 출력 프로젝션
            self.action_out_proj = nn.Linear(hidden_dim, action_dim)

        def forward(self, noisy_actions: torch.Tensor, timestep: torch.Tensor, vlm_kv_caches: list):
            """
            noisy_actions: (Batch, Chunk_Size, Action_Dim) - 보간된 액션 x_t
            timestep:      (Batch,) - [0, 1] 사이의 Flow 시간 t
            vlm_kv_caches: SmolVLM2의 상위 num_layers개 레이어에서 추출된 [(K_1, V_1), ...] 리스트
            """
            # 1. 액션 임베딩 + 사인파(Sinusoidal) 타임스텝 임베딩 주입
            action_emb = self.action_in_proj(noisy_actions)             # (B, K, D)
            t_emb = self.time_mlp(sinusoidal_embedding(timestep, action_emb.size(-1))) # (B, D)

            # 타임스텝 조건을 모든 액션 청크 토큰에 더해줌
            x = action_emb + t_emb.unsqueeze(1)                         # (B, K, D)

            # 2. 레이어별로 VLM의 해당 층 KV Cache를 참조하며 통과
            for layer, kv_cache in zip(self.layers, vlm_kv_caches):
                x = layer(x, kv_cache)

            # 3. 예측된 속도 벡터(v_theta) 반환
            velocity = self.action_out_proj(self.final_norm(x))         # (B, K, Action_Dim)
            return velocity

    return (SmolVLAActionExpert,)


@app.cell
def _(SmolVLAActionExpert, mo, torch):
    # 테스트: 작은 설정(2층, 폭 48)과 층마다 하나씩 만든 가짜 VLM K/V
    torch.manual_seed(0)
    _expert = SmolVLAActionExpert(action_dim=2, chunk_size=6, hidden_dim=48, vlm_dim=64, num_layers=2, num_heads=4).eval()
    _kv_caches = [(torch.randn(3, 5, 64), torch.randn(3, 5, 64)) for _ in range(2)]
    _x_t = torch.randn(3, 6, 2)
    with torch.no_grad():
        _v0 = _expert(_x_t, torch.zeros(3), _kv_caches)
        _v1 = _expert(_x_t, torch.ones(3), _kv_caches)
    assert _v0.shape == _x_t.shape
    _n_params = sum(p.numel() for p in _expert.parameters())

    _full = SmolVLAActionExpert(action_dim=32, chunk_size=50)  # 기본값(720/960/16층) 크기만 계산
    _n_full = sum(p.numel() for p in _full.parameters())
    mo.md(f"""
    - 입력 `x_t` `{tuple(_x_t.shape)}` + `t` `(3,)` + K/V {len(_kv_caches)}층 → 속도 `{tuple(_v0.shape)}`
    - 같은 `x_t`라도 `t=0`과 `t=1`의 출력 차이 평균: `{(_v0 - _v1).abs().mean():.4f}` (시간 조건이 반영됨)
    - 작은 설정 매개변수: {_n_params:,}개 / 기본 설정(720, 960, 16층): {_n_full / 1e6:.1f}M개
      (원본 행동 전문가는 Self/Cross 교대 구조라 수치가 다릅니다)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Flow Matching 학습 손실과 Euler 샘플링

    이 스케치의 규약은 $x_t = (1-t)\,\epsilon + t\,a$, 목표 속도 $u_t = a - \epsilon$입니다.
    추론은 $t=0$의 노이즈에서 시작해 $x \leftarrow x + v\,\Delta t$를 반복합니다. VLM K/V는 모든 스텝에서 재사용합니다.
    """)
    return


@app.cell
def _(nn, torch):
    # [학습 시: Conditional Flow Matching Loss]
    def compute_loss(expert, gt_actions, vlm_kv_caches):
        batch_size = gt_actions.size(0)
        noise = torch.randn_like(gt_actions)                         # x_0 ~ N(0, I)
        t = torch.rand(batch_size, device=gt_actions.device)         # t ~ Uniform(0, 1)

        # 선형 보간 경로: x_t = (1 - t) * noise + t * gt_actions
        t_view = t.view(-1, 1, 1)
        noisy_actions = (1 - t_view) * noise + t_view * gt_actions

        # 목표 속도 벡터 (Target Velocity): u_t = x_1 - x_0
        target_velocity = gt_actions - noise

        # 모델이 예측한 속도 벡터와 MSE 손실 계산
        pred_velocity = expert(noisy_actions, t, vlm_kv_caches)
        return nn.functional.mse_loss(pred_velocity, target_velocity)


    # [추론 시: Euler 적분 샘플링 (예: 10스텝)]
    @torch.no_grad()
    def sample_actions(expert, vlm_kv_caches, shape, num_steps=10, generator=None):
        device = vlm_kv_caches[0][0].device
        x = torch.randn(shape, device=device, generator=generator)  # 순수 노이즈(t=0)에서 시작
        dt = 1.0 / num_steps

        for i in range(num_steps):
            t = torch.full((shape[0],), i * dt, device=device)
            # VLM KV 캐시는 재계산하지 않고 10스텝 동안 계속 재사용 (핵심 속도 최적화!)
            v = expert(x, t, vlm_kv_caches)
            x = x + v * dt                                           # Euler 스텝 이동

        return x                                                     # 최종 생성된 Action Chunk (t=1)

    return compute_loss, sample_actions


@app.cell
def _(mo, torch):
    # 장난감 과제: 문맥 A이면 사인 곡선, 문맥 B이면 원 궤적을 그리는 2D 행동 청크 16스텝
    TOY_CHUNK, TOY_LAYERS, TOY_VLM_DIM = 16, 2, 32
    _s = torch.linspace(0, 1, TOY_CHUNK)
    toy_targets = torch.stack([
        torch.stack([_s * 2 - 1, torch.sin(2 * torch.pi * _s) * 0.8], dim=-1),                 # A
        torch.stack([torch.cos(2 * torch.pi * _s) * 0.8, torch.sin(2 * torch.pi * _s) * 0.8], dim=-1),  # B
    ])  # (2, 16, 2)
    _g = torch.Generator().manual_seed(1)
    # 문맥마다 고정된 가짜 VLM K/V (층별, 토큰 4개). 실제로는 이미지·지시문·상태가 만든 값입니다.
    toy_contexts = [
        (torch.randn(2, 4, TOY_VLM_DIM, generator=_g), torch.randn(2, 4, TOY_VLM_DIM, generator=_g))
        for _ in range(TOY_LAYERS)
    ]
    toy_train_btn = mo.ui.run_button(label="장난감 과제 학습 (CPU, 수 초)")
    mo.vstack([
        mo.md("문맥 두 개(가짜 K/V)에 서로 다른 목표 궤적을 대응시켜, 행동 전문가가 문맥에 따라 다른 청크를 생성하는지 봅니다."),
        toy_train_btn,
    ])
    return TOY_CHUNK, TOY_LAYERS, TOY_VLM_DIM, toy_contexts, toy_targets, toy_train_btn


@app.cell
def _(
    SmolVLAActionExpert,
    TOY_CHUNK,
    TOY_LAYERS,
    TOY_VLM_DIM,
    compute_loss,
    mo,
    toy_contexts,
    toy_targets,
    toy_train_btn,
    torch,
):
    mo.stop(not toy_train_btn.value, mo.md("버튼을 누르면 학습합니다."))
    torch.manual_seed(0)
    toy_expert = SmolVLAActionExpert(
        action_dim=2, chunk_size=TOY_CHUNK, hidden_dim=48, vlm_dim=TOY_VLM_DIM, num_layers=TOY_LAYERS, num_heads=4
    )
    _opt = torch.optim.AdamW(toy_expert.parameters(), lr=2e-3)
    # 배치 64: 문맥 A/B를 32개씩 반복
    _idx = torch.arange(64) % 2
    _gt = toy_targets[_idx]
    _kv = [(k[_idx], v[_idx]) for k, v in toy_contexts]
    toy_losses = []
    with mo.status.spinner("학습 중..."):
        for _step in range(600):
            _loss = compute_loss(toy_expert, _gt, _kv)
            _opt.zero_grad()
            _loss.backward()
            _opt.step()
            toy_losses.append(_loss.item())
    toy_expert.eval()
    return toy_expert, toy_losses


@app.cell
def _(plt, sample_actions, toy_contexts, toy_expert, toy_losses, toy_targets, torch):
    _g = torch.Generator().manual_seed(0)
    _n = 8  # 문맥마다 노이즈 8개로 샘플링
    _fig, _axes = plt.subplots(1, 3, figsize=(12, 3.6))
    _axes[0].plot(toy_losses, lw=0.8)
    _axes[0].set_yscale("log")
    _axes[0].set_title("Flow Matching 손실")
    _axes[0].set_xlabel("스텝")
    for _c, (_ax, _name) in enumerate(zip(_axes[1:], ["문맥 A (사인)", "문맥 B (원)"])):
        _kv = [(k[_c : _c + 1].expand(_n, -1, -1), v[_c : _c + 1].expand(_n, -1, -1)) for k, v in toy_contexts]
        _samples = sample_actions(toy_expert, _kv, (_n, toy_targets.shape[1], 2), num_steps=10, generator=_g)
        for _s in _samples:
            _ax.plot(_s[:, 0], _s[:, 1], "-", color="tab:blue", alpha=0.35, lw=1)
        _ax.plot(toy_targets[_c, :, 0], toy_targets[_c, :, 1], "k.--", lw=1, label="목표")
        _err = (_samples - toy_targets[_c]).abs().mean().item()
        _ax.set_title(f"{_name}: 샘플 {_n}개, 평균 오차 {_err:.3f}")
        _ax.set_aspect("equal")
        _ax.legend(fontsize=8, loc="lower right")
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    같은 전문가와 같은 노이즈 분포를 써도 **K/V(문맥)만 바뀌면 다른 궤적**이 나옵니다.
    600스텝 학습으로는 목표에 정확히 붙지 않습니다(평균 오차 약 0.2~0.4). 다만 자기 문맥 목표 쪽 오차가 다른 문맥 목표보다 작습니다.
    `num_steps`를 1, 3, 10으로 바꿔 Euler 적분 횟수에 따른 차이도 확인해 보세요.

    ## 5. 2D 블록 어텐션 마스크

    `att_masks`의 누적합으로 블록을 나눕니다. 값이 0이면 앞 토큰과 같은 블록, 1이면 새 블록을 시작합니다.
    Query는 누적합이 자기 이하인 Key만 볼 수 있습니다. 반환값은 `True`가 "볼 수 있음"입니다.
    """)
    return


@app.cell
def _(torch):
    def make_att_2d_masks(pad_masks: torch.Tensor, att_masks: torch.Tensor) -> torch.Tensor:
        """
        pad_masks: (B, N) - 유효한 토큰은 True, 패딩은 False
        att_masks: (B, N) - Prefix(VLM) 구간은 0, Suffix(Action) 구간은 1 (또는 스텝별 증가값)
        """
        # 누적합(cumsum)을 이용하면 Prefix(0)는 0으로 유지되고, Suffix(1)는 1 이상이 됨
        cumsum = torch.cumsum(att_masks, dim=1)

        # Key의 cumsum <= Query의 cumsum 인 위치만 허용 (True)
        # - Query가 Prefix(0)일 때: Key가 Prefix(0)인 경우만 True (0 <= 0), Suffix(1)는 False (1 <= 0 거짓)
        # - Query가 Suffix(1)일 때: Key가 Prefix(0)와 Suffix(1) 모두 True (0 <= 1, 1 <= 1 참)
        att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]

        # 패딩 토큰 마스킹 결합
        pad_2d_masks = pad_masks[:, None, :] & pad_masks[:, :, None]
        return att_2d_masks & pad_2d_masks

    return (make_att_2d_masks,)


@app.cell
def _(make_att_2d_masks, mo, plt, torch):
    # 테스트: 이미지 2 + 지시문 2(마지막은 패딩) + 상태 1 + 행동 3 = 토큰 8개
    _labels = ["img0", "img1", "txt0", "pad", "state", "act0", "act1", "act2"]
    _pad = torch.tensor([[1, 1, 1, 0, 1, 1, 1, 1]], dtype=torch.bool)
    _cases = {
        "행동 인과적 (원본 SmolVLA): [0,0,0,0, 1, 1,1,1]": torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1]]),
        "행동 양방향: [0,0,0,0, 1, 1,0,0]": torch.tensor([[0, 0, 0, 0, 1, 1, 0, 0]]),
    }
    _fig, _axes = plt.subplots(1, 2, figsize=(10, 4.4))
    _masks = {}
    for _ax, (_title, _att) in zip(_axes, _cases.items()):
        _m = make_att_2d_masks(_pad, _att)[0]
        _masks[_title] = _m
        _ax.imshow(_m.numpy(), cmap="Greens", vmin=0, vmax=1)
        _ax.set_xticks(range(8), _labels, rotation=60, fontsize=8)
        _ax.set_yticks(range(8), _labels, fontsize=8)
        _ax.set_xlabel("Key (볼 대상)")
        _ax.set_ylabel("Query")
        _ax.set_title(_title, fontsize=9)
    _fig.tight_layout()

    _causal = list(_masks.values())[0]
    _checks = {
        "이미지는 행동·상태를 못 봄": not _causal[0, 4:].any().item(),
        "상태는 접두부를 보지만 행동은 못 봄": _causal[4, [0, 1, 2]].all().item() and not _causal[4, 5:].any().item(),
        "act1은 act0·act1만 보고 act2는 못 봄": _causal[6, 5:7].all().item() and not _causal[6, 7].item(),
        "패딩 토큰은 행·열 모두 차단": not _causal[3].any().item() and not _causal[:, 3].any().item(),
    }
    assert all(_checks.values())
    mo.vstack([_fig, mo.md("\n".join(f"- {k}: **{v}**" for k, v in _checks.items()))])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. VLM·행동 전문가 혼합 층과 KV 캐시

    VLM 토큰과 행동 토큰의 Q/K/V를 시퀀스 축으로 이어 붙여 **어텐션 한 번**으로 계산하고,
    결과를 다시 나눠 각자의 출력 투영·MLP를 통과시킵니다. 두 경로는 hidden 폭이 달라도
    Q/K/V 투영 폭만 같으면 연결됩니다.
    블록 마스크 때문에 VLM 토큰은 행동을 보지 않으므로, VLM K/V는 Euler 스텝이 바뀌어도 그대로입니다.
    그래서 추론 2스텝째부터는 VLM 경로를 건너뛰고 캐시만 씁니다.
    """)
    return


@app.cell
def _(F, nn, torch):
    class SmolVLAMixtureLayer(nn.Module):
        def __init__(self, vlm_layer, expert_layer):
            super().__init__()
            self.vlm = vlm_layer         # SmolVLM2의 트랜스포머 레이어 (960 dim 등)
            self.expert = expert_layer   # Action Expert의 트랜스포머 레이어 (720 dim 등)

        def forward(
            self,
            vlm_embeds: torch.Tensor | None,     # 추론 2스텝째부터는 None (KV 캐시 재사용)
            action_embeds: torch.Tensor,         # (B, K, D_expert)
            att_2d_mask: torch.Tensor,           # (B, Total_Len, Total_Len)
            kv_cache: tuple | None = None
        ):
            # 1. [Prefix 경로] VLM 토큰이 주어진 경우에만 Q, K, V 계산 (학습 시 또는 추론 0스텝)
            if vlm_embeds is not None:
                vlm_norm = self.vlm.input_layernorm(vlm_embeds)
                vlm_q = self.vlm.self_attn.q_proj(vlm_norm)
                vlm_k = self.vlm.self_attn.k_proj(vlm_norm)
                vlm_v = self.vlm.self_attn.v_proj(vlm_norm)
                # (여기에 RoPE 위치 임베딩 적용)
                kv_cache = (vlm_k, vlm_v)        # 다음 Flow 스텝을 위해 KV 캐시 저장
            else:
                # 추론의 1~9번째 Euler 스텝에서는 VLM 연산을 완전히 건너뜀!
                vlm_k, vlm_v = kv_cache

            # 2. [Suffix 경로] Action 토큰의 Q, K, V 계산 (VLM과 차원이 달라도 Head 차원만 맞추면 됨)
            act_norm = self.expert.input_layernorm(action_embeds)
            act_q = self.expert.self_attn.q_proj(act_norm)
            act_k = self.expert.self_attn.k_proj(act_norm)
            act_v = self.expert.self_attn.v_proj(act_norm)

            # 3. [단일 어텐션 통합] 시퀀스 길이(Seq_Len) 축을 따라 K, V 결합
            if vlm_embeds is not None:
                # 학습 시: VLM과 Action 모두 동시에 업데이트
                full_q = torch.cat([vlm_q, act_q], dim=1)
                full_k = torch.cat([vlm_k, act_k], dim=1)
                full_v = torch.cat([vlm_v, act_v], dim=1)

                attn_out = F.scaled_dot_product_attention(full_q, full_k, full_v, attn_mask=att_2d_mask)

                # 결과를 다시 Prefix와 Suffix로 분리하여 각자의 Output Proj & MLP 통과
                vlm_len = vlm_embeds.size(1)
                vlm_out = vlm_embeds + self.vlm.mlp(self.vlm.post_attn_norm(self.vlm.self_attn.o_proj(attn_out[:, :vlm_len])))
                act_out = action_embeds + self.expert.mlp(self.expert.post_attn_norm(self.expert.self_attn.o_proj(attn_out[:, vlm_len:])))
                return vlm_out, act_out, kv_cache

            else:
                # 추론 중(Step > 0): Query는 Action만 사용하고, Key/Value만 캐시된 VLM + 현재 Action 결합
                full_k = torch.cat([vlm_k, act_k], dim=1)
                full_v = torch.cat([vlm_v, act_v], dim=1)

                # Suffix 영역에 해당하는 마스크 슬라이스만 사용
                act_mask = att_2d_mask[:, -action_embeds.size(1):, :]
                attn_out = F.scaled_dot_product_attention(act_q, full_k, full_v, attn_mask=act_mask)

                act_out = action_embeds + self.expert.mlp(self.expert.post_attn_norm(self.expert.self_attn.o_proj(attn_out)))
                return None, act_out, kv_cache

    return (SmolVLAMixtureLayer,)


@app.cell
def _(nn):
    class ToyAttn(nn.Module):
        """q/k/v_proj: hidden → attn_dim, o_proj: attn_dim → hidden (단일 헤드)"""

        def __init__(self, hidden, attn_dim):
            super().__init__()
            self.q_proj = nn.Linear(hidden, attn_dim, bias=False)
            self.k_proj = nn.Linear(hidden, attn_dim, bias=False)
            self.v_proj = nn.Linear(hidden, attn_dim, bias=False)
            self.o_proj = nn.Linear(attn_dim, hidden, bias=False)

    class ToyLayer(nn.Module):
        """SmolVLAMixtureLayer가 기대하는 속성 이름만 갖춘 장난감 트랜스포머 층"""

        def __init__(self, hidden, attn_dim):
            super().__init__()
            self.input_layernorm = nn.RMSNorm(hidden)
            self.self_attn = ToyAttn(hidden, attn_dim)
            self.post_attn_norm = nn.RMSNorm(hidden)
            self.mlp = nn.Sequential(nn.Linear(hidden, hidden * 2), nn.SiLU(), nn.Linear(hidden * 2, hidden))

    return (ToyLayer,)


@app.cell
def _(SmolVLAMixtureLayer, ToyLayer, make_att_2d_masks, mo, torch):
    # 테스트: VLM 폭 96, 전문가 폭 48, 공통 어텐션 폭 32. VLM 토큰 5개 + 행동 4개.
    torch.manual_seed(0)
    _layer = SmolVLAMixtureLayer(ToyLayer(96, 32), ToyLayer(48, 32)).eval()
    _B, _L, _K = 2, 5, 4
    _vlm = torch.randn(_B, _L, 96)
    _att = torch.tensor([[0] * _L + [1] * _K]).expand(_B, -1)
    _mask = make_att_2d_masks(torch.ones(_B, _L + _K, dtype=torch.bool), _att)

    with torch.no_grad():
        # Euler 0스텝: VLM과 행동을 함께 넣어 캐시 생성
        _act0 = torch.randn(_B, _K, 48)
        _vlm_out, _act_full0, _cache = _layer(_vlm, _act0, _mask)
        # Euler 1스텝: 행동만 바뀜. (a) 전체 재계산 vs (b) 캐시 재사용
        _act1 = torch.randn(_B, _K, 48)
        _, _act_full1, _ = _layer(_vlm, _act1, _mask)
        _none, _act_cached1, _ = _layer(None, _act1, _mask, kv_cache=_cache)
        # 비교용: 마스크 없이 VLM도 행동을 보게 하면 VLM 출력이 행동에 의존하게 됨
        _open = torch.ones_like(_mask)
        _vlm_open0, _, _ = _layer(_vlm, _act0, _open)
        _vlm_open1, _, _ = _layer(_vlm, _act1, _open)
        _vlm_out1, _, _ = _layer(_vlm, _act1, _mask)

    _same = torch.allclose(_act_full1, _act_cached1, atol=1e-5)
    _vlm_fixed = torch.allclose(_vlm_out, _vlm_out1, atol=1e-6)
    _vlm_leak = not torch.allclose(_vlm_open0, _vlm_open1, atol=1e-4)
    assert _none is None and _same and _vlm_fixed and _vlm_leak
    mo.md(f"""
    | 검사 | 결과 |
    | --- | --- |
    | 출력 모양: VLM / 행동 | `{tuple(_vlm_out.shape)}` / `{tuple(_act_full0.shape)}` |
    | 캐시 K/V 모양 | `{tuple(_cache[0].shape)}` (VLM 토큰 {_L}개 × 어텐션 폭 32) |
    | 행동이 바뀌어도 VLM 출력 불변 (블록 마스크) | {_vlm_fixed} |
    | 1스텝 행동 출력: 전체 재계산 = 캐시 재사용 (최대 차이 `{(_act_full1 - _act_cached1).abs().max():.1e}`) | {_same} |
    | 마스크를 열면 VLM 출력이 행동에 따라 변함 (캐시 불가) | {_vlm_leak} |
    """)
    return


if __name__ == "__main__":
    app.run()
