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

    SFT가 모범 답변의 다음 토큰을 맞힌다면, **PPO는 모델이 직접 생성한 답변을
    평가하고, 예상보다 좋은 행동의 확률을 높입니다.** 이 노트북에서는 이미 인사를
    할 줄 아는 작은 모델이 보상을 통해 공손한 인사를 더 자주 선택하도록 학습합니다.

    `minimind/trainer/train_ppo.py`와 `minimind/dataset/lm_dataset.py`를 따라
    **프롬프트 → rollout → 보상 → Critic·GAE → clipped loss → 재생성**을 실행합니다.
    원본 RLAIFDataset과 MiniMind 모델을 직접 사용합니다. PPO 학습 루프는
    전역 변수·GPU·외부 보상 모델 의존성을 없애고 셀에서 읽을 수 있게 축소했습니다.

    모델 구조 → SFT → 이 실습 순서를 권장합니다. Python, 텐서 shape, 역전파,
    다음 토큰 확률을 알고 시작하세요. 각 절의 **예측 → 실행 → 관찰 → 해설**을 따릅니다.
    `@app.cell`의 인자·return은 marimo 셀 간 데이터 전달입니다.
    모델·데이터 다운로드 없이 CPU에서 실행하며 산출물은 `outputs/notebooks/<실행 ID>/`에 저장합니다.
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

    상태 $s_t$는 프롬프트와 이미 생성한 토큰, 행동 $a_t$는 다음 토큰입니다.
    Critic의 값은 확률이 아니라 실수입니다. Reward는 실제 생성 결과를 평가하고,
    Critic은 행동 전 상태에서 미래 보상을 예측합니다.
    **old는 매 rollout마다 새로 저장하지만 reference는 계속 같습니다.**
    원본은 `LMForRewardModel` 점수에 길이·think 형식·반복 패널티를 더합니다.
    여기서는 **정확히 `안녕하세요 [EOS]`이면 +1, 나머지는 -1**인 규칙으로 대체합니다.
    PPO는 피드백으로 정책을 최적화하는 알고리즘이므로 규칙 보상으로도 계산을 익힐 수 있습니다.
    **예측:** 높은 보상을 받은 답도 Critic의 예상보다 나쁘면 advantage는 어떤 부호일까요?
    """)
    return


@app.cell
def _(PreTrainedTokenizerFast, RLAIFDataset, Tokenizer, json, mo, models,
      output_dir, pre_tokenizers, random):
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
    mo.vstack([
        mo.md("""
        ## 2. 원본 RLAIFDataset은 무엇을 반환할까?

        **예측:** JSONL의 마지막 assistant 답변이 PPO의 정답 label로 반환될까요?
        마지막 응답에 추적 문자열 `정답누출확인`을 넣어 확인합니다.
        원본 `create_chat_prompt`는 `conversations[:-1]`만 템플릿에 넣고,
        `__getitem__`은 `prompt`와 빈 `answer`를 반환합니다. 따라서 마지막 원소가
        assistant인 데이터를 준비해야 합니다. 이전 턴 assistant는 문맥으로 남습니다.

        system을 명시하여 무작위 system 추가를 피했습니다. `thinking_ratio=0`이며
        이 작은 템플릿은 사고 모드를 구현하지 않습니다. 원본 Dataset의 `max_length`는
        저장만 하고 절단에 쓰지 않습니다. 실제 절단·왼쪽 패딩은 trainer의 tokenizer가 수행합니다.
        여기서도 Dataset에 3을 주고 실제 토큰화에는 32를 주어 이를 확인합니다.
        """),
        mo.ui.table([{"질문": q, "실제 prompt": p, "answer": rl_dataset[i]["answer"],
                      "왼쪽 패딩 포함 ID": str(prompt_batch.input_ids[i].tolist())}
                     for i, (q, p) in enumerate(zip(questions, prompts))], selection=None),
    ])
    return prompt_batch, prompts, rl_dataset, tokenizer, vocab


@app.cell
def _(F, MiniMindConfig, MiniMindForCausalLM, copy, mo, prompt_batch, tokenizer, torch):
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

    mo.md(f"""
    ## 3. PPO 출발점: 작은 SFT 모델 준비

    무작위 모델에 바로 PPO를 적용하면 쓸 만한 응답을 샘플링하기 어렵습니다.
    여기서는 같은 질문에 `안녕 [EOS]`, `안녕하세요 [EOS]`를 동일 횟수 제공하여
    **SFT 100 step**으로 시작 모델을 준비합니다. 이 단계만 모범 응답 label을 사용합니다.
    SFT loss: **{warmup_losses[0]:.3f} → {warmup_losses[-1]:.3f}**.
    이후 PPO에서는 샘플링한 응답과 규칙 보상만 사용합니다.

    Actor는 어휘 크기의 logits, Critic은 위치당 값 하나를 출력합니다.
    원본 `CriticModel`은 `self.model`이 이미 norm한 출력에 norm을 한 번 더 적용합니다.
    이 실습도 대응을 위해 그 동작을 유지합니다. 별도의 Critic을 SFT 가중치에서
    초기화하고 value head만 새로 만듭니다. reference는 SFT 모델의 동결 사본입니다.
    """)
    return TeachingCritic, base_actor, config, reference, warmup_losses


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


@app.cell
def _(base_actor, collect_rollout, mo, prompt_batch, response_logps, tokenizer, torch):
    torch.manual_seed(123)
    example = collect_rollout(base_actor, prompt_batch, tokenizer)
    _check = response_logps(base_actor, example["ids"], example["full_mask"], example["positions"])
    rollout_logp_error = float(((_check.detach() - example["old_logp"]) * example["mask"]).abs().max())
    assert rollout_logp_error < 1e-5
    mo.vstack([
        mo.md(r"""
        ## 4. rollout과 한 칸 이동: 어떤 위치가 행동의 확률일까?

        프롬프트 패딩 폭을 P, 생성 폭을 R이라고 하면 입력은 `[B,P+R]`,
        응답 log probability·value·mask는 `[B,R]`입니다. **P는 왼쪽 패딩을 포함한 폭**입니다.
        첫 행동 `ids[:,P]`의 log probability는 `logits[:,P-1]`에서 가져옵니다.
        Critic도 행동 전 상태인 위치 P-1을 읽습니다. 원본의
        `logp_pos = prompt_lens - 1 + resp_idx`와 같습니다.

        **예측:** `[안녕하세요, EOS, PAD]` 중 몇 위치가 loss에 들어갈까요?
        정답은 2개입니다. EOS도 선택한 행동입니다. 프롬프트와 EOS 이후 위치는 제외합니다.
        원본은 rollout의 completion mask와 첫 EOS 위치를 결합합니다.
        이 실습은 생성 중 alive mask를 기록합니다. EOS 전에 PAD가 샘플링되면 그것도
        실제 행동으로 취급하며 토큰 ID만 보고 지우지 않습니다.

        원본 호출은 temperature=0.8입니다. 여기서는 샘플링 분포와 ratio에 쓰는 분포를
        명확히 맞추기 위해 **temperature=1, top-k/top-p 없이 전체 어휘에서 샘플링**합니다.
        아래 오차는 생성 시 저장한 old logp와 전체 시퀀스 재계산 결과를 비교한 값입니다.
        """),
        mo.ui.table([{"샘플": i, "응답 토큰": str(tokenizer.convert_ids_to_tokens(row.tolist())),
                      "mask": str(example["mask"][i].tolist()),
                      "logits/value 위치": str(example["positions"][i].tolist()),
                      "보상": float(example["rewards"][i])}
                     for i, row in enumerate(example["completion"])], selection=None),
        mo.md(f"유효 토큰 old logp 최대 오차: `{rollout_logp_error:.2e}`"),
    ])
    return example, rollout_logp_error


@app.cell
def _(gae, mo, torch):
    _mask = torch.tensor([[1., 1., 1.], [1., 1., 0.]])
    _values = torch.tensor([[0.2, 0.3, 0.4], [0.1, 0.2, 999.]])
    _norm, _returns, _raw, _rewards = gae(torch.tensor([1., -1.]), _values, _mask)
    assert torch.allclose(_raw[0], torch.tensor([0.7365, 0.67, 0.6]), atol=1e-6)
    assert _raw[1, 2] == 0 and _returns[1, 2] == 0
    mo.vstack([
        mo.md(r"""
        ## 5. 마지막에 받은 보상을 앞 행동까지 전달하기: GAE

        $\delta_t=r_t+\gamma V(s_{t+1})-V(s_t)$

        $A_t=\delta_t+\gamma\lambda A_{t+1}$, 뒤에서 앞으로 계산합니다.
        이 예제는 $\gamma=1$, $\lambda=0.95$, 끝의 다음 value=0입니다.
        첫 행에서 마지막 $\delta=1-0.4=0.6$,
        가운데 $A=(0.4-0.3)+0.95\times0.6=0.67$입니다.
        **예측:** 마지막에만 보상을 넣어도 첫 토큰의 advantage가 0일까요?

        Critic의 학습 목표는 **return = 정규화 전 advantage + old value**입니다.
        Actor용 advantage는 유효 토큰 전체의 평균·분산으로 정규화합니다.
        정규화하면 부호가 달라질 수 있으므로 표의 원본값과 따로 읽으세요.
        두 번째 응답의 패딩 value=999는 0으로 마스킹하여 GAE에 들어오지 않습니다.
        원본도 old value를 마스킹하고 뒤에서 누적합니다. 이 구현은 매 단계 mask를 명시합니다.

        길이 제한으로 EOS 없이 끝나도 원본처럼 마지막 value의 bootstrap을 0으로 둡니다.
        이는 종료로 취급하는 선택입니다. 계속되는 환경에서 시간 제한만 걸린 경우에는
        다음 상태 value로 bootstrap하는 설계가 필요합니다.
        """),
        mo.ui.table([{"샘플": b, "t": t, "유효": bool(_mask[b,t]),
                      "원래 V": float(_values[b,t]), "토큰 보상": float(_rewards[b,t]),
                      "raw A": round(float(_raw[b,t]), 4), "return": round(float(_returns[b,t]), 4),
                      "정규화 A": round(float(_norm[b,t]), 4)}
                     for b in range(2) for t in range(3)], selection=None),
    ])
    return


@app.cell
def _(mo, torch):
    _rows = []
    for _a in [1., -1.]:
        for _r in [0.6, 1., 1.4]:
            _ratio = torch.tensor(_r, requires_grad=True)
            _unclipped = -_a * _ratio
            _clipped = -_a * _ratio.clamp(0.8, 1.2)
            _loss = torch.maximum(_unclipped, _clipped)
            _loss.backward()
            _rows.append({"A": _a, "ratio": _r, "일반 loss": float(_unclipped.detach()),
                          "clip loss": round(float(_clipped.detach()), 3),
                          "선택 loss": round(float(_loss.detach()), 3),
                          "d loss / d ratio": float(_ratio.grad)})
    mo.vstack([
        mo.md(r"""
        ## 6. PPO clip은 모든 변화를 막는가?

        $\rho_t=\exp(\log\pi_\theta(a_t|s_t)-\log\pi_{old}(a_t|s_t))$

        $L_{clip}=\mathrm{mean}_{valid}\left[\max(-A_t\rho_t,
        -A_t\mathrm{clip}(\rho_t,1-\epsilon,1+\epsilon))\right]$

        최대화할 목적에 마이너스를 붙여 **최소화할 loss**로 썼으므로 `max`입니다.
        A>0이면 확률을 높이고 A<0이면 낮추되, 유리한 방향의 과도한 이동에서 이득을 제한합니다.
        **예측:** A<0, ratio=1.4이면 gradient가 0일까요? 표에서 확인하세요.
        이 경우 나쁜 행동을 더 많이 선택했으므로 되돌리는 gradient가 남습니다.
        clip은 확률이나 매개변수의 변화에 대한 강제 상한이 아닙니다.
        """), mo.ui.table(_rows, selection=None),
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

    원본은 KL을 token reward에 넣지 않고 **Actor loss에 별도 더합니다.** 이 실습도 같습니다.
    MoE가 아니므로 보조 손실은 0입니다. old value와 return은 detach한 고정 목표입니다.

    | 지표 | 비교 기준 | 쓰임 |
    | --- | --- | --- |
    | approx_kl = mean(0.5 × log_ratio²) | 이번 old policy | 한 rollout의 업데이트 조기 종료 |
    | kl_ref = mean(exp(d) − d − 1) | SFT reference | 기준 정책에서 멀어지는 것을 완화 |
    | clipfrac = mean(abs(ratio−1)>epsilon) | 이번 old policy | clip 구간을 벗어난 토큰 비율 |

    이들은 샘플 토큰에서 구하는 지표이며 어휘 전체를 합산한 정확한 KL이 아닙니다.
    clipfrac는 실제로 gradient가 0인 토큰 비율과도 다릅니다. clip 예제의 부호를 떠올려 보세요.
    보상 상승과 함께 이 지표·Critic loss·응답 길이를 봐야 합니다.
    """)
    return


@app.cell
def _(mo):
    rounds_ui = mo.ui.slider(4, 32, step=4, value=20, label="새 rollout 수")
    beta_ui = mo.ui.slider(0.0, 0.5, step=0.02, value=0.02, label="reference KL 계수 β")
    mo.vstack([mo.md("## 8. 실제 MiniMind PPO 업데이트\n설정 변경 시 같은 SFT 모델에서 다시 시작합니다. 한 rollout=32응답, mini-batch=16, 최대 3회 재사용입니다."),
               rounds_ui, beta_ui])
    return beta_ui, rounds_ui


@app.cell
def _(TeachingCritic, base_actor, beta_ui, collect_rollout, config, copy, gae,
      json, masked_mean, mo, output_dir, prompt_batch, reference,
      response_logps, rounds_ui, tokenizer, torch, uuid4):
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
    mo.vstack([
        mo.md(f"""공손한 2토큰 응답의 **정확한 경로 확률 평균**: {before_probability:.1%} → {after_probability:.1%}.
        같은 4개 프롬프트에서 측정했습니다. 학습 문제에 대한 점검이며 별도 일반화 평가는 아닙니다.
        매 round 보상은 업데이트 **전** 샘플, 경로 확률은 업데이트 **후** 정책의 값입니다.
        진단 loss/KL은 실제 수행한 mini-batch 업데이트 직전 값의 평균입니다.
        reference 동결, Actor·Critic 갱신, 최초 ratio≈1, 유한 loss, Actor 저장·복원을 확인했습니다.
        저장 경로: `{training_dir}`"""),
        mo.ui.table(history, selection=None),
    ])
    return after_probability, before_probability, history, results, traces, training_dir


@app.cell
def _(history, mo, plt, traces, training_dir):
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
    mo.vstack([_fig, mo.md("### 첫 rollout과 마지막 rollout의 실제 응답\n응답 길이·보상과 old value, raw/정규화 advantage, return을 연결해 읽으세요."),
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

    이 실습은 CPU float32·Dense·dropout=0·누적=1·고정 학습률입니다.
    원본의 AMP, DDP, scheduler, MoE 보조 손실, SGLang 정책 동기화와 중간 재개는 실행하지 않습니다.
    원본 Torch 엔진은 Actor 객체를 참조하지만 외부 엔진은 `update_policy`로 가중치를
    동기화해야 합니다. old logp는 **실제로 샘플링에 사용한 정책**의 값이어야 합니다.
    전체 trainer를 실행하려면 SFT 가중치·원본 tokenizer·RLAIF 데이터·보상 모델이 필요합니다.
    이 노트북의 작은 어휘·가중치는 원본 기본 768차원 모델과 호환되지 않습니다.

    ## 10. 수강생 과제와 확인 질문

    1. **β=0과 0.5를 비교하세요.** 같은 seed·round 수로 최종 경로 확률, kl_ref,
       approx_kl, clipfrac를 기록하세요. 힌트: old는 갱신되므로 두 KL이 같을 필요가 없습니다.
       해설: β는 reference 이탈에 대한 비용이며 보상 상승 속도와 최종 값은 관찰해야 합니다.
    2. **GAE의 λ를 0과 1로 바꿔 손계산하세요.** 5절 첫 응답에서 raw advantage는 각각
       `[0.1,0.1,0.6]`, `[0.8,0.7,0.6]`입니다. λ=0은 한 단계 TD,
       λ=1·γ=1·끝 bootstrap=0이면 return은 세 위치 모두 최종 보상 1이 됩니다.
    3. **A=-1, ratio=0.6의 clip gradient를 설명하세요.** 이미 나쁜 행동의 확률을
       충분히 낮췄기 때문에 surrogate에서 추가 이득을 주지 않습니다.
       ratio=1.4인 경우에는 바로잡는 gradient가 남습니다.
    4. **보상을 `안녕 [EOS]`에 주도록 바꾸세요.** collect_rollout의 보상과
       evaluate_policy의 측정 경로를 함께 수정하고, 반대 인사의 선택 확률을 기록하세요.
       해설: 보상 상승은 작성한 평가 기준에 맞추는 것이며 공손함을 자동으로 이해했다는 뜻은 아닙니다.

    **자기 점검:** 왜 PPO에서는 모범 응답 label이 없어도 학습되는가?
    왜 old logp와 return을 고정하는가? 왜 EOS 다음 토큰을 제외하는가?
    왜 ratio가 처음에는 1이어야 하는가? Critic의 예상이 정확하면 advantage는 어떻게 되는가?

    참고한 저장소 코드: `minimind/trainer/train_ppo.py`,
    `minimind/dataset/lm_dataset.py`, `minimind/trainer/rollout_engine.py`,
    `minimind/model/model_minimind.py`. 원본 라이선스는 `minimind/LICENSE`를 확인하세요.
    """)
    return


if __name__ == "__main__":
    app.run()
