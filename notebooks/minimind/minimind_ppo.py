import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import copy
    import json
    import random
    import sys
    from datetime import datetime
    from pathlib import Path
    from uuid import uuid4
    from zoneinfo import ZoneInfo

    import datasets
    import marimo as mo
    import matplotlib.pyplot as plt
    import torch
    import torch.nn.functional as F
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root / "projects/minimind"))
    from dataset.lm_dataset import RLAIFDataset
    from model.model_minimind import MiniMindConfig, MiniMindForCausalLM

    output_dir = repo_root / "outputs" / "notebooks" / (
        datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y%m%d-%H%M%S")
        + f"-{uuid4().hex[:6]}-ppo"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    datasets.config.HF_DATASETS_CACHE = output_dir / "dataset-cache"
    datasets.disable_progress_bars()
    torch.set_num_threads(2)
    return (F, MiniMindConfig, MiniMindForCausalLM, PreTrainedTokenizerFast,
            RLAIFDataset, Tokenizer, copy, json, mo, models, output_dir,
            plt, pre_tokenizers, random, torch, uuid4)


@app.cell
def _(mo):
    mo.md(r"""
    # MiniMind PPO fine tuning: 생성한 답변을 보상으로 개선하기

    - SFT: 모범 답변을 따라 합니다.
    - **PPO: 모델이 직접 만든 답변을 평가합니다. 예상보다 좋은 행동의 확률을 높입니다.**

    이 노트북의 과제: 인사할 줄 아는 작은 모델이 **공손한 인사**를 더 자주 고르게 합니다.

    **흐름:** 프롬프트 → rollout → 보상 → Critic·GAE → clipped loss → 재생성.

    - 원본 RLAIFDataset과 MiniMind 모델을 그대로 씁니다.
    - PPO 루프는 셀에서 읽기 쉽게 줄였습니다(GPU·외부 보상 모델 없음).

    **순서:** SFT → DPO → **PPO** → GRPO.
    DPO의 "참조 대비 변화"가 여기서는 KL 페널티로 나옵니다.
    보상은 **모델이 만든 응답**에 줍니다.

    CPU로 실행하며 다운로드는 없습니다. 산출물: `outputs/notebooks/<실행 ID>/`.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. 네 역할과 두 개의 고정 기준

    | 역할 | 출력 | PPO에서 갱신? |
    | --- | --- | --- |
    | Actor $\pi_\theta$ | 다음 토큰의 확률 | 예 |
    | old policy $\pi_{old}$ | 이번 rollout을 만들 때의 토큰 log probability | 아니요. 이번 배치에서 수치로 저장 |
    | Reference $\pi_{ref}$ | SFT 직후 기준 확률 | 아니요. 학습 내내 고정 |
    | Critic $V_\phi(s_t)$ | 이 상태 이후 얻을 보상의 예상값 | 예 |
    | Reward | 완성된 답변 점수 | 아니요. 이 실습은 고정 규칙 |

    - **상태 $s_t$:** 프롬프트 + 지금까지 만든 토큰
    - **행동 $a_t$:** 다음 토큰
    - **Critic:** 행동 전에 미래 보상을 예측합니다(확률 아닌 실수).
    - **Reward:** 완성된 답을 평가합니다.
    - **old vs reference:** old는 rollout마다 새로 저장합니다. reference는 끝까지 같습니다.

    **보상 규칙:** 원본은 보상 모델 점수에 길이·형식·반복 패널티를 더합니다.
    여기서는 단순하게 바꿨습니다. **`안녕하세요 [EOS]`면 +1, 나머지는 -1.**
    규칙 보상으로도 PPO 계산을 익히기에는 충분합니다.

    **예측:** 보상이 높아도 Critic 예상보다 낮으면 advantage의 부호는?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 2. 원본 RLAIFDataset은 무엇을 반환할까?

    **예측:** JSONL의 마지막 assistant 답변이 PPO의 정답 label로 반환될까요?
    마지막 응답에 `정답누출확인`이라는 표시를 넣어 확인합니다.

    - `create_chat_prompt`는 `conversations[:-1]`만 씁니다.
    - `__getitem__`은 `prompt`와 빈 `answer`를 돌려줍니다.
    - 그래서 데이터의 마지막은 assistant여야 합니다. 앞 턴의 assistant는 문맥으로 남습니다.

    **기타:**

    - system을 직접 넣어 무작위 추가를 막았습니다.
    - `thinking_ratio=0`입니다. 사고 모드는 쓰지 않습니다.
    - Dataset의 `max_length`는 저장만 되고 쓰이지 않습니다. 절단·왼쪽 패딩은 trainer가 합니다.
      그래서 Dataset에는 3, 토큰화에는 32를 줘서 확인합니다.
    """)
    return


@app.cell
def _(PreTrainedTokenizerFast, RLAIFDataset, Tokenizer, json, models, output_dir, pre_tokenizers, random):
    questions = ["인사", "인사 부탁", "인사 부탁 해", "인사 해"]
    records = [{"conversations": [
        {"role": "system", "content": "인사 도우미"},
        {"role": "user", "content": q},
        {"role": "assistant", "content": "정답누출확인"},
    ]} for q in questions]
    vocab = ["[PAD]", "[BOS]", "[EOS]", "[UNK]", "system", "user", "assistant",
             "인사", "도우미", "부탁", "해", "안녕", "안녕하세요", "정답누출확인"]
    _backend = Tokenizer(models.WordLevel({w: i for i, w in enumerate(vocab)}, unk_token="[UNK]"))
    _backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=_backend, pad_token="[PAD]",
        bos_token="[BOS]", eos_token="[EOS]", unk_token="[UNK]", padding_side="left")
    tokenizer.chat_template = (
        "{% for m in messages %}{{ bos_token + m['role'] + ' ' + m['content'] + ' ' + eos_token + ' ' }}{% endfor %}"
        "{% if add_generation_prompt %}{{ bos_token + 'assistant ' }}{% endif %}"
    )
    _path = output_dir / "rlaif.jsonl"
    _path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")
    random.seed(42)
    rl_dataset = RLAIFDataset(str(_path), tokenizer, max_length=3, thinking_ratio=0.0)
    prompts = [rl_dataset[i]["prompt"] for i in range(len(rl_dataset))]
    assert all("정답누출확인" not in p for p in prompts)
    assert all(rl_dataset[i]["answer"] == "" for i in range(len(rl_dataset)))
    prompt_batch = tokenizer(prompts, padding=True, truncation=True, max_length=32, return_tensors="pt")
    assert prompt_batch.input_ids.shape[1] > rl_dataset.max_length
    tokenizer.save_pretrained(output_dir / "tokenizer")
    return (prompt_batch, prompts, questions, rl_dataset, tokenizer, vocab,)


@app.cell(hide_code=True)
def _(mo, prompt_batch, prompts, questions, rl_dataset):
    # ▸ 결과 표시 — 실행하면 표·그림이 나타납니다 (marimo 표시 코드)
    mo.vstack([
        mo.ui.table([{"질문": q, "실제 prompt": p, "answer": rl_dataset[i]["answer"],
                      "왼쪽 패딩 포함 ID": str(prompt_batch.input_ids[i].tolist())}
                     for i, (q, p) in enumerate(zip(questions, prompts))], selection=None),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 3. PPO 출발점: 작은 SFT 모델 준비
    """)
    return


@app.cell
def _(MiniMindConfig, MiniMindForCausalLM, copy, prompt_batch, tokenizer, torch):
    torch.manual_seed(42)
    config = MiniMindConfig(hidden_size=32, num_hidden_layers=1, num_attention_heads=4,
        num_key_value_heads=2, intermediate_size=64, max_position_embeddings=64,
        vocab_size=len(tokenizer), dropout=0.0, flash_attn=False,
        bos_token_id=tokenizer.bos_token_id, eos_token_id=tokenizer.eos_token_id)
    base_actor = MiniMindForCausalLM(config)
    _opt = torch.optim.AdamW(base_actor.parameters(), lr=0.004)
    # 같은 질문에 두 인사를 같은 횟수 제공: PPO가 선택을 바꿀 여지를 남긴다.
    _prefix = prompt_batch.input_ids.repeat_interleave(2, dim=0)
    _mask = prompt_batch.attention_mask.repeat_interleave(2, dim=0)
    _answers = torch.tensor([[tokenizer.convert_tokens_to_ids(w), tokenizer.eos_token_id]
                            for _ in range(4) for w in ["안녕", "안녕하세요"]])
    _ids = torch.cat([_prefix, _answers], dim=1)
    _full_mask = torch.cat([_mask, torch.ones_like(_answers)], dim=1)
    _labels = torch.full_like(_ids, -100)
    _labels[:, _prefix.size(1):] = _answers
    warmup_losses = []
    for _step in range(100):
        _opt.zero_grad()
        _loss = base_actor(_ids, attention_mask=_full_mask, labels=_labels).loss
        _loss.backward()
        torch.nn.utils.clip_grad_norm_(base_actor.parameters(), 1.0)
        _opt.step()
        warmup_losses.append(float(_loss.detach()))
    base_actor.eval()
    reference = copy.deepcopy(base_actor).eval().requires_grad_(False)

    class TeachingCritic(MiniMindForCausalLM):
        """train_ppo.CriticModel의 value head와 추가 norm 동작을 유지한다."""
        def __init__(self, cfg):
            super().__init__(cfg)
            self.value_head = torch.nn.Linear(cfg.hidden_size, 1)

        def forward(self, input_ids, attention_mask=None):
            hidden = self.model(input_ids=input_ids, attention_mask=attention_mask)[0]
            return self.value_head(self.model.norm(hidden)).squeeze(-1)
    return (TeachingCritic, base_actor, config, reference, warmup_losses,)


@app.cell(hide_code=True)
def _(mo, warmup_losses):
    # ▸ 결과 설명 — 실행하면 계산 값이 들어간 설명이 나타납니다 (marimo 표시 코드)
    mo.md(f"""
    무작위 모델로 바로 PPO를 하면 쓸 만한 응답이 거의 안 나옵니다.
    그래서 먼저 **SFT 100 step**으로 시작 모델을 만듭니다.

    - 같은 질문에 `안녕 [EOS]`, `안녕하세요 [EOS]`를 같은 횟수로 줍니다.
    - 모범 응답 label은 이 단계에서만 씁니다.
    - SFT loss: **{warmup_losses[0]:.3f} → {warmup_losses[-1]:.3f}**.

    이후 PPO는 샘플링한 응답과 규칙 보상만 씁니다.

    **모델 구성:**

    - Actor: 어휘 크기의 logits를 냅니다.
    - Critic: 위치마다 값 하나를 냅니다. SFT 가중치로 시작하고 value head만 새로 만듭니다.
    - reference: SFT 모델의 동결 사본입니다.

    원본 `CriticModel`은 norm을 한 번 더 적용합니다. 여기서도 그대로 둡니다.
    """)
    return


@app.cell
def _(F, torch):
    def masked_mean(x, mask):
        return (x * mask).sum() / mask.sum().clamp(min=1)

    def response_logps(model, ids, full_mask, positions):
        # logits[:, j]가 ids[:, j+1]을 예측한다.
        logits = model(ids, attention_mask=full_mask).logits[:, :-1]
        all_logps = F.log_softmax(logits, dim=-1).gather(2, ids[:, 1:, None]).squeeze(-1)
        return all_logps.gather(1, positions)

    @torch.no_grad()
    def collect_rollout(actor, batch, tok, max_new=3):
        ids = batch.input_ids.clone()
        full_mask = batch.attention_mask.clone()
        B, P = ids.shape
        alive = torch.ones(B, dtype=torch.bool)
        tokens, masks, logps = [], [], []
        for _ in range(max_new):
            logits = actor(ids, attention_mask=full_mask).logits[:, -1]
            # 전체 어휘, temperature=1, top-k/top-p 없음: old/new의 확률 정의 일치.
            distribution = torch.distributions.Categorical(logits=logits)
            sampled = distribution.sample()
            sampled = torch.where(alive, sampled, tok.pad_token_id)
            tokens.append(sampled)
            masks.append(alive.clone())  # EOS는 유효, EOS 다음부터 무효
            logps.append(distribution.log_prob(sampled))
            ids = torch.cat([ids, sampled[:, None]], dim=1)
            full_mask = torch.cat([full_mask, alive[:, None].long()], dim=1)
            alive = alive & sampled.ne(tok.eos_token_id)
        completion = torch.stack(tokens, dim=1)
        mask = torch.stack(masks, dim=1).float()
        positions = torch.arange(max_new)[None, :].expand(B, -1) + P - 1
        # 이 과제는 EOS가 없으면 미완성으로 판정한다.
        polite_id = tok.convert_tokens_to_ids("안녕하세요")
        successful = (completion[:, 0] == polite_id) & (completion[:, 1] == tok.eos_token_id)
        rewards = successful.float() * 2 - 1
        return {"ids": ids, "full_mask": full_mask, "completion": completion,
                "mask": mask, "positions": positions, "old_logp": torch.stack(logps, dim=1),
                "rewards": rewards}

    @torch.no_grad()
    def gae(rewards, values, mask, gamma=1.0, lam=0.95):
        # 원본과 같이 마지막 유효 토큰에 외부 보상을 넣고 끝의 bootstrap은 0.
        token_rewards = torch.zeros_like(values)
        last = mask.sum(1).long() - 1
        token_rewards[torch.arange(len(last)), last] = rewards
        values = values * mask
        running = torch.zeros(values.size(0))
        advantages = torch.zeros_like(values)
        for t in reversed(range(values.size(1))):
            next_value = values[:, t + 1] if t + 1 < values.size(1) else 0.0
            delta = token_rewards[:, t] + gamma * next_value - values[:, t]
            running = (delta + gamma * lam * running) * mask[:, t]
            advantages[:, t] = running
        returns = advantages + values  # 정규화 전 advantage로 계산!
        mean = masked_mean(advantages, mask)
        var = masked_mean((advantages - mean).square(), mask)
        normalized = (advantages - mean) * torch.rsqrt(var + 1e-8) * mask
        return normalized, returns, advantages, token_rewards

    return collect_rollout, gae, masked_mean, response_logps


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. rollout과 한 칸 이동: 어떤 위치가 행동의 확률일까?

    **shape:** 프롬프트 폭 P(왼쪽 패딩 포함), 생성 폭 R.

    - 입력: `[B,P+R]`
    - 응답 log prob·value·mask: `[B,R]`

    **위치:** 첫 행동 `ids[:,P]`의 확률은 `logits[:,P-1]`에 있습니다.
    Critic도 행동 전 위치 P-1을 읽습니다. 원본의 `logp_pos = prompt_lens - 1 + resp_idx`와 같습니다.

    **예측:** `[안녕하세요, EOS, PAD]`에서 몇 개가 loss에 들어갈까요?
    **2개**입니다. EOS도 고른 행동입니다. 프롬프트와 EOS 뒤는 뺍니다.

    - 원본은 completion mask와 첫 EOS 위치로 mask를 만듭니다.
    - 여기서는 생성 중에 alive mask를 기록합니다.
    - EOS 전에 PAD가 뽑히면 그것도 행동으로 셉니다.

    **샘플링:** 원본은 temperature=0.8입니다.
    여기서는 계산을 맞추려고 **temperature=1, 전체 어휘에서 샘플링**합니다.
    아래 오차는 저장한 old logp와 다시 계산한 값의 차이입니다.
    """)
    return


@app.cell
def _(base_actor, collect_rollout, prompt_batch, response_logps, tokenizer, torch):
    torch.manual_seed(123)
    example = collect_rollout(base_actor, prompt_batch, tokenizer)
    _check = response_logps(base_actor, example["ids"], example["full_mask"], example["positions"])
    rollout_logp_error = float(((_check.detach() - example["old_logp"]) * example["mask"]).abs().max())
    assert rollout_logp_error < 1e-5
    return (example, rollout_logp_error,)


@app.cell(hide_code=True)
def _(example, mo, rollout_logp_error, tokenizer):
    # ▸ 결과 표시 — 실행하면 표·그림이 나타납니다 (marimo 표시 코드)
    mo.vstack([
        mo.ui.table([{"샘플": i, "응답 토큰": str(tokenizer.convert_ids_to_tokens(row.tolist())),
                      "mask": str(example["mask"][i].tolist()),
                      "logits/value 위치": str(example["positions"][i].tolist()),
                      "보상": float(example["rewards"][i])}
                     for i, row in enumerate(example["completion"])], selection=None),
        mo.md(f"유효 토큰 old logp 최대 오차: `{rollout_logp_error:.2e}`"),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. 마지막에 받은 보상을 앞 행동까지 전달하기: GAE

    $\delta_t=r_t+\gamma V(s_{t+1})-V(s_t)$

    $A_t=\delta_t+\gamma\lambda A_{t+1}$. 뒤에서 앞으로 계산합니다.

    **예제:** $\gamma=1$, $\lambda=0.95$, 끝 다음 value=0.

    - 첫 행 마지막: $\delta=1-0.4=0.6$
    - 첫 행 가운데: $A=(0.4-0.3)+0.95\times0.6=0.67$

    **예측:** 보상이 마지막에만 있으면 첫 토큰의 advantage는 0일까요?

    **Critic 목표:** return = (정규화 전) advantage + old value.

    **Actor용 advantage:** 유효 토큰 전체로 평균·분산 정규화합니다.
    정규화하면 부호가 바뀔 수 있습니다. 원본값과 따로 보세요.

    **마스킹:** 두 번째 응답의 패딩 value=999는 0으로 가려져 GAE에 안 들어갑니다.

    **EOS 없이 끝나면:** 마지막 bootstrap을 0으로 둡니다(원본과 같음). 종료로 보는 선택입니다.
    시간 제한만 걸린 환경이라면 다음 value로 bootstrap해야 합니다.
    """)
    return


@app.cell
def _(gae, torch):
    gae_demo_mask = torch.tensor([[1., 1., 1.], [1., 1., 0.]])
    gae_demo_values = torch.tensor([[0.2, 0.3, 0.4], [0.1, 0.2, 999.]])
    gae_demo_normalized, gae_demo_returns, gae_demo_raw, gae_demo_rewards = gae(torch.tensor([1., -1.]), gae_demo_values, gae_demo_mask)
    assert torch.allclose(gae_demo_raw[0], torch.tensor([0.7365, 0.67, 0.6]), atol=1e-6)
    assert gae_demo_raw[1, 2] == 0 and gae_demo_returns[1, 2] == 0
    return (
        gae_demo_mask,
        gae_demo_normalized,
        gae_demo_raw,
        gae_demo_returns,
        gae_demo_rewards,
        gae_demo_values,
    )


@app.cell(hide_code=True)
def _(gae_demo_mask, gae_demo_normalized, gae_demo_raw, gae_demo_returns, gae_demo_rewards, gae_demo_values, mo):
    # ▸ 결과 표시 — 실행하면 표·그림이 나타납니다 (marimo 표시 코드)
    mo.vstack([
        mo.ui.table([{"샘플": b, "t": t, "유효": bool(gae_demo_mask[b,t]),
                      "원래 V": float(gae_demo_values[b,t]), "토큰 보상": float(gae_demo_rewards[b,t]),
                      "raw A": round(float(gae_demo_raw[b,t]), 4), "return": round(float(gae_demo_returns[b,t]), 4),
                      "정규화 A": round(float(gae_demo_normalized[b,t]), 4)}
                     for b in range(2) for t in range(3)], selection=None),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. PPO clip은 모든 변화를 막는가?

    $\rho_t=\exp(\log\pi_\theta(a_t|s_t)-\log\pi_{old}(a_t|s_t))$

    $L_{clip}=\mathrm{mean}_{valid}\left[\max(-A_t\rho_t,
    -A_t\mathrm{clip}(\rho_t,1-\epsilon,1+\epsilon))\right]$

    목적에 마이너스를 붙여 **loss**로 바꿨기 때문에 `max`입니다.

    - A>0이면 확률을 올리고, A<0이면 내립니다.
    - 좋은 방향으로 너무 많이 가면 더 이상 이득을 주지 않습니다.

    **예측:** A<0, ratio=1.4이면 gradient가 0일까요? 표에서 확인하세요.
    **아닙니다.** 나쁜 행동을 더 고르게 됐으니 되돌리는 gradient가 남습니다.
    clip은 변화량의 강제 상한이 아닙니다.
    """)
    return


@app.cell
def _(torch):
    clip_rows = []
    for _a in [1., -1.]:
        for _r in [0.6, 1., 1.4]:
            _ratio = torch.tensor(_r, requires_grad=True)
            _unclipped = -_a * _ratio
            _clipped = -_a * _ratio.clamp(0.8, 1.2)
            _loss = torch.maximum(_unclipped, _clipped)
            _loss.backward()
            clip_rows.append({"A": _a, "ratio": _r, "일반 loss": float(_unclipped.detach()),
                          "clip loss": round(float(_clipped.detach()), 3),
                          "선택 loss": round(float(_loss.detach()), 3),
                          "d loss / d ratio": float(_ratio.grad)})
    return (clip_rows,)


@app.cell(hide_code=True)
def _(clip_rows, mo):
    # ▸ 결과 표시 — 실행하면 표·그림이 나타납니다 (marimo 표시 코드)
    mo.vstack([
        mo.ui.table(clip_rows, selection=None),
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 7. Critic loss, reference penalty, 두 KL 지표

    $V_{clip}=\mathrm{clip}(V_\phi,V_{old}-c,V_{old}+c)$

    $L_V=\frac12\mathrm{mean}_{valid}\max((V_\phi-G)^2,(V_{clip}-G)^2)$

    $d=\log\pi_{ref}-\log\pi_\theta$, $k_{ref}=\exp(d)-d-1$

    $L=L_{clip}+\beta\,\mathrm{mean}_{valid}(k_{ref})+c_v L_V$

    - KL은 보상에 넣지 않고 **Actor loss에 따로 더합니다**(원본과 같음).
    - Dense라서 보조 손실은 0입니다.
    - old value와 return은 detach한 고정 목표입니다.

    | 지표 | 비교 기준 | 쓰임 |
    | --- | --- | --- |
    | approx_kl = mean(0.5 × log_ratio²) | 이번 old policy | 한 rollout의 업데이트 조기 종료 |
    | kl_ref = mean(exp(d) − d − 1) | SFT reference | 기준 정책에서 멀어지는 것을 완화 |
    | clipfrac = mean(abs(ratio−1)>epsilon) | 이번 old policy | clip 구간을 벗어난 토큰 비율 |

    - 샘플 토큰으로 구한 근사값입니다. 정확한 KL이 아닙니다.
    - clipfrac는 gradient가 0인 비율과 다릅니다(6절 clip 표 참고).
    - 보상만 보지 말고 이 지표, Critic loss, 응답 길이도 함께 보세요.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 8. 실제 MiniMind PPO 업데이트

    - rollout 한 번 = 응답 32개
    - mini-batch 16, 최대 3회 재사용
    - 설정을 바꾸면 같은 SFT 모델에서 다시 시작합니다.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    # ▸ 실습 설정 위젯 — 실행하면 슬라이더·선택 상자가 나타납니다 (marimo 표시 코드)
    rounds_ui = mo.ui.slider(4, 32, step=4, value=20, label="새 rollout 수")
    beta_ui = mo.ui.slider(0.0, 0.5, step=0.02, value=0.02, label="reference KL 계수 β")
    mo.vstack([
               rounds_ui, beta_ui])
    return beta_ui, rounds_ui


@app.cell
def _(TeachingCritic, base_actor, beta_ui, collect_rollout, config, copy, gae, json, masked_mean, output_dir, prompt_batch, reference, response_logps, rounds_ui, tokenizer, torch, uuid4):
    torch.manual_seed(2026)
    actor = copy.deepcopy(base_actor).train()
    critic = TeachingCritic(config)
    critic.load_state_dict(base_actor.state_dict(), strict=False)
    actor_opt = torch.optim.AdamW(actor.parameters(), lr=0.0005)
    critic_opt = torch.optim.AdamW(critic.parameters(), lr=0.001)
    _batch = copy.deepcopy(prompt_batch)
    for _key in _batch:
        _batch[_key] = _batch[_key].repeat(8, 1)
    _ref_state = {k: v.clone() for k, v in reference.state_dict().items()}

    @torch.no_grad()
    def evaluate_policy(model):
        # 같은 네 질문의 정확한 2토큰 경로 확률. 샘플링 변동과 구분한다.
        ids = prompt_batch.input_ids
        mask = prompt_batch.attention_mask
        first = torch.full((len(ids), 1), tokenizer.convert_tokens_to_ids("안녕하세요"))
        p_first = model(ids, attention_mask=mask).logits[:, -1].softmax(-1).gather(1, first).squeeze(1)
        second_ids = torch.cat([ids, first], dim=1)
        second_mask = torch.cat([mask, torch.ones_like(first)], dim=1)
        p_eos = model(second_ids, attention_mask=second_mask).logits[:, -1].softmax(-1)[:, tokenizer.eos_token_id]
        return float((p_first * p_eos).mean())

    before_probability = evaluate_policy(actor)
    history, traces = [], []
    for _round in range(rounds_ui.value):
        # 새 rollout마다 현재 Actor로 샘플링. old 통계는 아래 3회 재사용 동안 고정.
        rollout = collect_rollout(actor, _batch, tokenizer)
        with torch.no_grad():
            old_values = critic(rollout["ids"], rollout["full_mask"]).gather(1, rollout["positions"]) * rollout["mask"]
            ref_logp = response_logps(reference, rollout["ids"], rollout["full_mask"], rollout["positions"])
            advantages, returns, raw_adv, token_rewards = gae(rollout["rewards"], old_values, rollout["mask"])
        _max_old_error = float(((response_logps(actor, rollout["ids"], rollout["full_mask"], rollout["positions"]).detach()
                                 - rollout["old_logp"]) * rollout["mask"]).abs().max())
        assert _max_old_error < 1e-5
        _logs, _stopped = [], False
        for _epoch in range(3):
            for _indices in torch.randperm(32).split(16):
                _m = rollout["mask"][_indices]
                _new_logp = response_logps(actor, rollout["ids"][_indices], rollout["full_mask"][_indices], rollout["positions"][_indices])
                _v = critic(rollout["ids"][_indices], rollout["full_mask"][_indices]).gather(1, rollout["positions"][_indices])
                _log_ratio = _new_logp - rollout["old_logp"][_indices]
                _ratio = _log_ratio.exp()
                _approx = masked_mean(0.5 * _log_ratio.square(), _m)
                if float(_approx.detach()) > 0.25:
                    _stopped = True
                    break  # CPU 단일 프로세스이므로 DDP용 0-loss backward 불필요
                _a = advantages[_indices]
                _policy = masked_mean(torch.maximum(-_a * _ratio, -_a * _ratio.clamp(0.8, 1.2)), _m)
                _d = ref_logp[_indices] - _new_logp
                _kl_ref = masked_mean(_d.exp() - _d - 1, _m)
                _v_clip = torch.clamp(_v, old_values[_indices] - 0.2, old_values[_indices] + 0.2)
                _value_loss = 0.5 * masked_mean(torch.maximum((_v - returns[_indices]).square(),
                                                                            (_v_clip - returns[_indices]).square()), _m)
                _loss = _policy + beta_ui.value * _kl_ref + 0.5 * _value_loss
                assert torch.isfinite(_loss)
                actor_opt.zero_grad()
                critic_opt.zero_grad()
                _loss.backward()
                torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
                torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
                actor_opt.step()
                critic_opt.step()
                _logs.append([float(x.detach()) for x in [_policy, _value_loss, _approx, _kl_ref,
                              masked_mean((_ratio - 1).abs().gt(0.2).float(), _m)]])
            if _stopped:
                break
        _means = torch.tensor(_logs).mean(0).tolist() if _logs else [0.] * 5
        history.append({"round": _round + 1, "sample_reward": float(rollout["rewards"].mean()),
                        "success_probability": evaluate_policy(actor), "avg_length": float(rollout["mask"].sum(1).mean()),
                        **dict(zip(["policy_loss", "value_loss", "approx_kl", "kl_ref", "clipfrac"], _means)),
                        "updates": len(_logs), "early_stop": _stopped})
        if _round in [0, rounds_ui.value - 1]:
            for _i in range(8):
                _valid = rollout["mask"][_i].bool()
                traces.append({"round": _round + 1, "prompt_index": _i % 4,
                    "tokens": tokenizer.convert_ids_to_tokens(rollout["completion"][_i][_valid].tolist()),
                    "reward": float(rollout["rewards"][_i]), "mask": rollout["mask"][_i].tolist(),
                    "old_logp": rollout["old_logp"][_i][_valid].tolist(),
                    "old_values": old_values[_i][_valid].tolist(), "raw_advantage": raw_adv[_i][_valid].tolist(),
                    "normalized_advantage": advantages[_i][_valid].tolist(), "returns": returns[_i][_valid].tolist()})
    after_probability = evaluate_policy(actor)
    assert all(torch.equal(v, _ref_state[k]) for k, v in reference.state_dict().items())
    assert all(p.grad is None for p in reference.parameters())
    assert any(not torch.equal(v, base_actor.state_dict()[k]) for k, v in actor.state_dict().items())
    assert any(not torch.equal(v, base_actor.state_dict()[k]) for k, v in critic.state_dict().items() if k in base_actor.state_dict())
    training_dir = output_dir / f"training-r{rounds_ui.value}-b{beta_ui.value:.2f}-{uuid4().hex[:6]}"
    training_dir.mkdir(parents=True, exist_ok=True)
    results = {"config": config.to_dict(), "rounds": rounds_ui.value, "kl_coef": beta_ui.value,
               "seed": 2026, "sampling_temperature": 1.0, "max_new_tokens": 3,
               "batch_size": 32, "mini_batch_size": 16, "ppo_update_iters": 3,
               "gamma": 1.0, "lam": 0.95, "clip_epsilon": 0.2, "cliprange_value": 0.2,
               "vf_coef": 0.5, "early_stop_kl": 0.25, "actor_lr": 0.0005, "critic_lr": 0.001,
               "before_probability": before_probability, "after_probability": after_probability,
               "history": history, "traces": traces,
               "checks": {"reference_unchanged": True, "actor_and_critic_updated": True,
                          "initial_ratio_one": True, "finite_losses": True}}
    (training_dir / "metrics.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    torch.save({"actor": actor.state_dict(), "critic": critic.state_dict(),
                "actor_optimizer": actor_opt.state_dict(), "critic_optimizer": critic_opt.state_dict(),
                "config": config.to_dict()}, training_dir / "ppo.pt")
    _restored = type(base_actor)(config)
    _restored.load_state_dict(torch.load(training_dir / "ppo.pt", weights_only=True)["actor"])
    assert abs(evaluate_policy(_restored) - after_probability) < 1e-7
    results["checks"]["actor_save_restore"] = True
    (training_dir / "metrics.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    return (after_probability, before_probability, history, results, traces, training_dir,)


@app.cell(hide_code=True)
def _(after_probability, before_probability, history, mo, training_dir):
    # ▸ 결과 표시 — 실행하면 표·그림이 나타납니다 (marimo 표시 코드)
    mo.vstack([
        mo.md(f"""공손한 2토큰 응답의 **정확한 경로 확률 평균**: {before_probability:.1%} → {after_probability:.1%}.

        - 학습한 4개 프롬프트로 쟀습니다. 일반화 평가가 아닙니다.
        - 보상은 업데이트 **전** 샘플, 경로 확률은 업데이트 **후** 값입니다.
        - 진단 loss/KL은 각 mini-batch 업데이트 직전 값의 평균입니다.
        - 확인한 것: reference 동결, Actor·Critic 갱신, 첫 ratio≈1, 유한 loss, 저장·복원.

        저장 경로: `{training_dir}`"""),
        mo.ui.table(history, selection=None),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 첫 rollout과 마지막 rollout의 실제 응답
    응답 길이, 보상, old value, advantage(원본·정규화), return을 함께 보세요.
    """)
    return


@app.cell(hide_code=True)
def _(history, mo, plt, traces, training_dir):
    # ▸ 결과 표시 — 실행하면 표·그림이 나타납니다 (marimo 표시 코드)
    _fig, _axes = plt.subplots(1, 3, figsize=(12, 3))
    _x = [r["round"] for r in history]
    for _ax, _keys, _title in zip(_axes,
            [["sample_reward", "success_probability"], ["approx_kl", "kl_ref"], ["value_loss", "clipfrac"]],
            ["Reward and success probability", "Two KL diagnostics", "Value loss and clip fraction"]):
        for _key in _keys:
            _ax.plot(_x, [r[_key] for r in history], label=_key)
        _ax.set(xlabel="Rollout round", title=_title)
        _ax.legend(fontsize=8)
    _fig.tight_layout()
    _fig.savefig(training_dir / "learning_curves.png", dpi=140)
    mo.vstack([_fig, 
               mo.ui.table(traces, selection=None)])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 9. 원본 코드로 돌아가기

    | 이 실습 | `train_ppo.py`의 대응 |
    | --- | --- |
    | 원본 RLAIFDataset, 왼쪽 패딩 | Dataset 생성 → tokenizer 호출 |
    | collect_rollout, old_logp 저장 | rollout_engine.rollout → per_token_logps |
    | EOS 포함 mask, positions | resp_policy_mask, logp_pos |
    | 고정 규칙 ±1 | calculate_rewards: RM + 길이·형식·반복 점수 |
    | TeachingCritic, old_values | CriticModel, old_resp_values |
    | gae | token_rewards → 역순 GAE → returns → advantage 정규화 |
    | 3회 재사용, mini-batch 16 | ppo_update_iters, mini_batch_size |
    | clip loss + reference penalty + value loss | policy_loss + vf_coef × value_loss + aux_loss |
    | approx KL 초과 시 break | stop_ppo와 DDP용 0-loss backward |

    **이 실습의 설정:** CPU float32, Dense, dropout=0, 누적 1, 고정 학습률.

    **실행하지 않은 것:** AMP, DDP, scheduler, MoE 보조 손실, SGLang 동기화, 중간 재개.

    **주의할 점:**

    - 외부 엔진(SGLang 등)은 `update_policy`로 가중치를 맞춰야 합니다.
    - old logp는 **실제로 샘플링한 정책**의 값이어야 합니다.
    - 전체 trainer에는 SFT 가중치, tokenizer, RLAIF 데이터, 보상 모델이 필요합니다.
    - 이 노트북의 가중치는 768차원 원본 모델과 호환되지 않습니다.

    ## 10. 수강생 과제와 확인 질문

    1. **β=0과 0.5를 비교하세요.** 경로 확률, kl_ref, approx_kl, clipfrac를 적으세요.
       힌트: old는 계속 바뀌므로 두 KL이 같을 필요는 없습니다.
       해설: β는 reference에서 멀어지는 비용입니다. 효과는 직접 관찰하세요.
    2. **λ를 0과 1로 바꿔 손으로 계산하세요.** 5절 첫 응답의 advantage는
       `[0.1,0.1,0.6]`, `[0.8,0.7,0.6]`입니다.
       λ=0은 한 단계 TD입니다. λ=1이면 return은 모두 최종 보상 1입니다.
    3. **A=-1, ratio=0.6의 clip gradient를 설명하세요.**
       나쁜 행동 확률을 이미 충분히 낮췄으므로 추가 이득이 없습니다.
       ratio=1.4이면 되돌리는 gradient가 남습니다.
    4. **보상을 `안녕 [EOS]`에 주도록 바꾸세요.**
       collect_rollout의 보상과 evaluate_policy의 측정 경로를 함께 고치세요.
       해설: 보상이 오르는 건 기준에 맞춘 것뿐입니다. 공손함을 이해한 게 아닙니다.

    **자기 점검:**

    - 모범 label 없이 어떻게 학습되나요?
    - old logp와 return은 왜 고정하나요?
    - EOS 다음 토큰은 왜 빼나요?
    - ratio는 왜 처음에 1이어야 하나요?
    - Critic 예상이 정확하면 advantage는?

    **참고 코드:** `minimind/trainer/train_ppo.py`, `minimind/dataset/lm_dataset.py`,
    `minimind/trainer/rollout_engine.py`, `minimind/model/model_minimind.py`.
    라이선스는 `minimind/LICENSE`를 보세요.

    **다음 실습:** `minimind_grpo.py`.

    - Critic 대신 **같은 질문의 여러 응답 평균**을 기준선으로 씁니다.
    - think 형식과 정답 보상으로 추론 모델을 학습합니다.
    - 6절의 clip 표는 GRPO에서도 같아서 반복하지 않습니다.
    """)
    return


if __name__ == "__main__":
    app.run()
