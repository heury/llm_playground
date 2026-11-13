import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import ast
    import copy
    import json
    import random
    import re
    import sys
    from datetime import datetime
    from pathlib import Path
    from types import SimpleNamespace
    from uuid import uuid4
    from zoneinfo import ZoneInfo

    import datasets
    import marimo as mo
    import matplotlib.pyplot as plt
    import torch
    from transformers import AutoTokenizer

    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root / "projects/minimind"))
    from dataset.lm_dataset import RLAIFDataset
    from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
    from trainer.gsm8k_reward import score_gsm8k_response

    torch.set_num_threads(2)
    run_dir = repo_root / "outputs" / "notebooks" / (
        datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y%m%d-%H%M%S") + "-" + uuid4().hex[:8] + "-grpo"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    datasets.config.HF_DATASETS_CACHE = run_dir / "dataset-cache"
    datasets.disable_progress_bars()
    return (ast, copy, json, random, re, SimpleNamespace,
            mo, plt, torch, repo_root, RLAIFDataset, MiniMindConfig,
            MiniMindForCausalLM, AutoTokenizer, run_dir, score_gsm8k_response)


@app.cell
def _(mo):
    mo.md(r"""
    # MiniMind GRPO와 추론(reasoning) 모델: 같은 질문의 여러 답으로 정책 개선하기

    **학습 목표:** `train_grpo.py`의 흐름을 설명할 수 있게 됩니다.

    질문 → 여러 응답 → 보상 → 그룹 내 advantage → 토큰 log prob → 손실 → 갱신.

    원본 함수로 **추론 모델의 보상**도 확인합니다.

    **순서:** SFT → DPO → PPO 다음입니다. ratio·clip·reference KL은 다시 설명하지 않습니다.

    1. 실제 `RLAIFDataset`과 토크나이저로 입력 형식을 봅니다.
    2. 보상·advantage를 계산합니다. think 형식 보상과 정답(GSM8K) 보상을 비교합니다.
    3. 마스크와 GRPO/CISPO 손실을 계산합니다.
    4. 작은 MiniMind로 응답을 샘플링하고 CPU에서 GRPO 학습합니다.

    다운로드와 GPU는 필요 없습니다.
    마지막 학습은 두 질문, 한 토큰 응답으로 줄인 교육용 실험입니다.
    대화 성능이나 GSM8K 성능은 평가하지 않습니다. 외부 보상 모델도 쓰지 않습니다.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. SFT와 GRPO의 학습 신호

    | 구분 | SFT | 이 코드의 GRPO |
    | --- | --- | --- |
    | 입력 | 질문과 정답 응답 | 질문과 직접 생성한 G개 응답 |
    | 목표 | 정답 토큰의 확률 높이기 | 그룹에서 상대 보상이 높은 응답의 확률 높이기 |
    | 신호 | 토큰별 labels | 응답별 스칼라 reward → advantage |
    | 기준선 | 해당 없음 | 같은 질문 응답들의 평균 보상 |
    | 추가 모델 | 학습할 정책 | 고정 reference, model 보상일 때 보상 모델 |

    - **Critic 대신** 그룹 통계를 씁니다. reward model은 여전히 있을 수 있습니다.
    - 정책은 SFT 모델에서 시작합니다.
    - reference: 시작 시점 정책의 고정 사본입니다.
    - old policy: 이번 응답을 만든 정책입니다. reference와 다릅니다.

    **예측:** 질문마다 응답이 하나뿐이면 비교가 될까요?
    아래에서 G=1이거나 보상이 모두 같을 때를 확인합니다.
    """)
    return


@app.cell
def _(AutoTokenizer, RLAIFDataset, json, random, repo_root, run_dir):
    chat_tokenizer = AutoTokenizer.from_pretrained(repo_root / "projects/minimind/model", local_files_only=True)
    _records = [
        {"conversations": [
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": "2 + 2는 얼마인가요?"},
            {"role": "assistant", "content": "4입니다. 이 문장은 GRPO 입력에서 제외됩니다."},
        ]},
        {"conversations": [
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": "2 + 3은 얼마인가요?"},
            {"role": "assistant", "content": "5입니다."},
        ]},
    ]
    data_path = run_dir / "rlaif_toy.jsonl"
    data_path.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in _records) + "\n", encoding="utf-8")
    random.seed(42)
    rl_dataset = RLAIFDataset(str(data_path), chat_tokenizer, max_length=8, thinking_ratio=0.0)
    rl_samples = [rl_dataset[i] for i in range(len(rl_dataset))]
    assert all(s["answer"] == "" for s in rl_samples)
    assert "이 문장은" not in rl_samples[0]["prompt"]
    return (chat_tokenizer, data_path, rl_dataset, rl_samples,)


@app.cell(hide_code=True)
def _(mo, rl_samples):
    # ▸ 결과 표시 — 실행하면 표·그림이 나타납니다 (marimo 표시 코드)
    mo.vstack([mo.md("**실제 Dataset 반환값**"), mo.ui.table(rl_samples, selection=None),
               mo.md("첫 프롬프트:\n```text\n" + rl_samples[0]["prompt"] + "\n```" )])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. `RLAIFDataset`은 정답 labels를 만들지 않는다

    - `create_chat_prompt()`는 `conversations[:-1]`만 씁니다.
    - 그래서 **마지막 메시지는 assistant여야 합니다.** 마지막이 user면 질문이 빠집니다.
    - 반환값: `{'prompt': 문자열, 'answer': ''}`. model 보상 경로는 answer를 안 씁니다.
    - GSM8K 경로는 `GSM8KDataset`과 숫자 정답을 따로 씁니다.

    **기타:**

    - system이 없으면 무작위로 추가됩니다. 여기서는 직접 넣었습니다.
    - `thinking_ratio`는 `open_thinking`으로 전달됩니다. 품질 점수가 아닙니다.
    - `max_length`는 저장만 됩니다. 절단은 trainer가 합니다.

    **관찰:** `max_length=8`이어도 전체 문자열이 나옵니다.
    trainer는 왼쪽 패딩으로 토큰화하고 마지막 `max_seq_len`개를 남깁니다.
    질문이 잘리면 과제가 바뀝니다. GSM8K 경로는 길이를 넘으면 오류를 냅니다.
    """)
    return


@app.cell
def _(chat_tokenizer, rl_samples):
    encoded_prompts = chat_tokenizer([s["prompt"] for s in rl_samples],
        return_tensors="pt", padding=True, padding_side="left",
        add_special_tokens=False, return_token_type_ids=False)
    B, G, R = 2, 3, 4
    expanded_prompts = encoded_prompts["input_ids"].repeat_interleave(G, dim=0)
    return (B, G, R, encoded_prompts, expanded_prompts,)


@app.cell(hide_code=True)
def _(B, G, R, encoded_prompts, expanded_prompts, mo):
    # ▸ 결과 표시 — 실행하면 표·그림이 나타납니다 (marimo 표시 코드)
    mo.vstack([mo.ui.table([
        {"응답 행": i, "질문 인덱스": i // G, "그룹 내 번호": i % G}
        for i in range(B * G)
    ], selection=None), mo.md(f"질문 shape: `{tuple(encoded_prompts['input_ids'].shape)}` → "
        f"반복 후 `{tuple(expanded_prompts.shape)}`. 생성 응답은 `[B×G, R] = [{B*G}, {R}]`." )])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. 응답별 보상과 그룹별 advantage

    응답 순서는 **질문별로 묶입니다**: `q0/a0, q0/a1, q0/a2, q1/a0, ...`.
    그래야 `rewards.view(B, G)`에서 질문이 섞이지 않습니다.

    $$A_i = \frac{r_i - \operatorname{mean}(r_{group})}
    {\operatorname{std}(r_{group},\;unbiased=False)+10^{-4}}$$

    응답 하나의 advantage는 그 응답의 모든 토큰이 나눠 씁니다.
    어느 토큰이 좋았는지는 알려 주지 않습니다.

    **예측:** `[0, 1, 2]`와 `[5, 5, 5]`의 advantage는?
    평균이 5인 두 번째 그룹이 더 좋은 신호일까요?
    """)
    return


@app.cell
def _(torch):
    def group_advantages(rewards, num_generations):
        groups = rewards.view(-1, num_generations)
        means = groups.mean(1).repeat_interleave(num_generations)
        stds = groups.std(1, unbiased=False).repeat_interleave(num_generations)
        return (rewards - means) / (stds + 1e-4)

    example_rewards = torch.tensor([0., 1., 2., 5., 5., 5.])
    example_advantages = group_advantages(example_rewards, 3)
    assert torch.equal(example_advantages[3:], torch.zeros(3))
    assert torch.equal(group_advantages(example_rewards, 1), torch.zeros(6))
    return (example_advantages, example_rewards, group_advantages,)


@app.cell(hide_code=True)
def _(example_advantages, example_rewards, mo):
    # ▸ 결과 표시 — 실행하면 표·그림이 나타납니다 (marimo 표시 코드)
    mo.ui.table([
        {"행": i, "질문": i // 3, "reward": float(r), "advantage": float(a)}
        for i, (r, a) in enumerate(zip(example_rewards, example_advantages))
    ], selection=None)
    return


@app.cell
def _(mo):
    mo.md(r"""
    **해설:** 첫 그룹은 약 `[-1.2246, 0, 1.2246]`, 두 번째는 모두 0입니다.

    - 모두 오답이어도, **모두 정답이어도** 보상 신호는 0입니다.
    - 그래도 정책이 reference와 다르면 KL 항은 작동합니다.
    - advantage 평균이 0이라는 것만으로는 학습 상태를 알 수 없습니다.

    ### 원본 model 보상의 구성

    `calculate_rewards()`는 네 가지를 더합니다.

    - 전체 응답 길이(문자 수)
    - `</think>` 앞부분 길이와 태그 수
    - 답변의 반복 패널티(`rep_penalty`: 정규식으로 나눈 n-gram 반복)
    - 보상 모델 점수 `reward_model.get_score(messages, answer)`

    태그가 맞아도 추론은 틀릴 수 있습니다. 내용보다 점수만 높이는 행동도 살펴보세요.

    다음 셀은 **원본 두 함수만 AST로 읽어 실행**합니다(trainer 전체 import는 피함).
    보상 모델 점수는 항상 0.2로 대신합니다(stub). 보상 모델 품질 검증이 아닙니다.
    """)
    return


@app.cell
def _(SimpleNamespace, ast, re, repo_root, rl_samples, torch):
    trainer_source = (repo_root / "projects/minimind/trainer/train_grpo.py").read_text(encoding="utf-8")
    _tree = ast.parse(trainer_source)
    _nodes = [n for n in _tree.body if isinstance(n, ast.FunctionDef)
              and n.name in {"rep_penalty", "calculate_rewards"}]
    _namespace = {"torch": torch, "re": re,
                  "args": SimpleNamespace(device="cpu", num_generations=3)}
    exec(compile(ast.Module(body=_nodes, type_ignores=[]), "train_grpo_reward_excerpt", "exec"), _namespace)

    class ConstantRewardModel:
        def get_score(self, messages, answer):
            return 0.2

    calculate_rewards = _namespace["calculate_rewards"]
    reward_responses = ["4", "The answer is four. This is a short explanation.",
                        "word " * 12]
    original_rewards = calculate_rewards(
        [rl_samples[0]["prompt"]], reward_responses, ConstantRewardModel())
    return (
        ConstantRewardModel,
        calculate_rewards,
        original_rewards,
        reward_responses,
        trainer_source,
    )


@app.cell(hide_code=True)
def _(mo, original_rewards, reward_responses):
    # ▸ 결과 표시 — 실행하면 표·그림이 나타납니다 (marimo 표시 코드)
    mo.ui.table([{"응답": s, "원본 보상(stub 포함)": float(r)}
                 for s, r in zip(reward_responses, original_rewards)], selection=None)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 3-1. 추론(reasoning) 모델: think 형식과 두 종류의 보상

    MiniMind의 응답은 `<think>추론</think>` 뒤에 최종 답을 씁니다(SFT 노트북 5-1절).

    - SFT: 사람이 쓴 추론을 **따라 씁니다.**
    - GRPO: 모델이 **스스로 쓴** 추론과 답을 보상으로 평가합니다.

    **프롬프트:** `thinking_ratio` 확률로 `open_thinking=True`가 됩니다(GRPO 기본 0.9).

    - True: 프롬프트가 `<think>\n`에서 끝납니다. 모델은 추론부터 씁니다.
    - False: 빈 블록 `<think>\n\n</think>\n\n`까지 넣습니다. 바로 답합니다.

    | 보상 경로 | 무엇을 점수로 주는가 | think에 대한 처리 |
    | --- | --- | --- |
    | `calculate_rewards` (기본, 보상 모델) | 전체 길이 + think 길이·태그 형식 + 반복 패널티 + 보상 모델 점수 | think 20~300자 +1.0, 아니면 −0.5; `</think>` 정확히 1개 +0.25, 아니면 −0.25 |
    | `score_gsm8k_response` (`--reward_type gsm8k`) | 마지막 줄 `#### 숫자`가 정답과 같으면 1, 아니면 0 | 닫힌 think 0~1개만 허용. **형식만으로는 점수 없음** |

    **예측:** 형식은 같고 숫자만 틀린 응답은 두 보상에서 몇 점일까요?

    - 아래 응답은 `open_thinking=True` 뒤에 생성됐다고 가정합니다.
    - GSM8K 프롬프트는 `assistant\n`에서 끝나서 모델이 `<think>`부터 씁니다.
      그래서 GSM8K 채점에는 앞에 `<think>\n`을 붙여 넣습니다.
    """)
    return


@app.cell
def _(ConstantRewardModel, RLAIFDataset, calculate_rewards, chat_tokenizer, data_path, random, score_gsm8k_response):
    random.seed(42)
    thinking_prompt = RLAIFDataset(str(data_path), chat_tokenizer, thinking_ratio=1.0)[0]["prompt"]
    direct_prompt = RLAIFDataset(str(data_path), chat_tokenizer, thinking_ratio=0.0)[0]["prompt"]
    assert thinking_prompt.endswith("<think>\n")
    assert direct_prompt.endswith("<think>\n\n</think>\n\n")
    _reasoning = "2와 2를 더하면 4가 됩니다. 계산을 한 번 더 확인했습니다."
    reasoning_cases = [
        ("올바른 추론 + 정답", _reasoning + "\n</think>\n\n#### 4"),
        ("너무 짧은 추론 + 정답", "4\n</think>\n\n#### 4"),
        ("형식은 같고 숫자만 오답", _reasoning.replace("4가", "5가") + "\n</think>\n\n#### 5"),
        ("</think> 두 번", _reasoning + "\n</think>\n</think>\n\n#### 4"),
        ("추론 없이 답만", "</think>\n\n#### 4"),
        ("정답 표기(####) 없음", _reasoning + "\n</think>\n\n답은 4입니다."),
    ]
    _format_rewards = calculate_rewards([thinking_prompt] * 2, [r for _, r in reasoning_cases], ConstantRewardModel())
    reasoning_rows = []
    for (_name, _response), _format in zip(reasoning_cases, _format_rewards):
        _score, _reason = score_gsm8k_response("<think>\n" + _response, "2 + 2 = 4\n#### 4")
        reasoning_rows.append({"응답 종류": _name, "completion": repr(_response),
                               "calculate_rewards (stub 0.2 포함)": round(float(_format), 3),
                               "GSM8K 점수": _score, "GSM8K 판정": _reason})
    _by_name = {r["응답 종류"]: r for r in reasoning_rows}
    # 형식 보상은 숫자의 정오를 보지 않고, GSM8K 보상은 think 길이를 보지 않습니다.
    assert _by_name["올바른 추론 + 정답"]["calculate_rewards (stub 0.2 포함)"] == _by_name["형식은 같고 숫자만 오답"]["calculate_rewards (stub 0.2 포함)"]
    assert _by_name["올바른 추론 + 정답"]["GSM8K 점수"] == 1 and _by_name["형식은 같고 숫자만 오답"]["GSM8K 점수"] == 0
    assert _by_name["너무 짧은 추론 + 정답"]["GSM8K 점수"] == 1
    assert _by_name["올바른 추론 + 정답"]["calculate_rewards (stub 0.2 포함)"] > _by_name["너무 짧은 추론 + 정답"]["calculate_rewards (stub 0.2 포함)"]
    assert _by_name["</think> 두 번"]["GSM8K 판정"] == "format_error"
    return (direct_prompt, reasoning_cases, reasoning_rows, thinking_prompt,)


@app.cell(hide_code=True)
def _(direct_prompt, mo, reasoning_rows, thinking_prompt):
    # ▸ 결과 표시 — 실행하면 표·그림이 나타납니다 (marimo 표시 코드)
    mo.vstack([
        mo.md(f"`thinking_ratio=1` 프롬프트 끝: `{thinking_prompt[-30:]!r}`\n\n"
              f"`thinking_ratio=0` 프롬프트 끝: `{direct_prompt[-30:]!r}`"),
        mo.ui.table(reasoning_rows, selection=None),
        mo.md('**해설:** 기본 보상은 **길이·태그 형식**도 평가합니다. 이 예제는 보상 모델 점수를 고정했으므로 숫자가 틀려도 점수가 같습니다. 실제 보상 모델은 내용을 평가하지만, 형식 점수만 높이는 행동(reward hacking)이 생길 수 있습니다.\n\nGSM8K는 **최종 정답**만 봅니다. 추론이 짧아도 맞으면 1점입니다. GRPO는 같은 질문의 응답끼리 보상을 비교합니다.\n\n여기서는 보상만 계산합니다. 7절에서는 한 토큰 산술 과제로 학습합니다.'),
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 4. 다음 토큰 정렬과 EOS 마스크

    `logits[:, t]`는 `input_ids[:, t+1]`의 분포입니다.

    1. `log_softmax → gather(다음 토큰 ID)`로 `[N, T-1]`을 만듭니다.
    2. `logp_pos = prompt_lens - 1 + arange(R)`로 응답 위치만 고릅니다.

    **예:** P=3, 응답 `[7, EOS, PAD, PAD]` → logits **2, 3, 4, 5번**을 읽습니다.

    - 손실에는 첫 EOS까지 **포함한** 두 위치만 씁니다.
    - EOS가 없으면 유효 길이까지 씁니다.
    - `prompt_lens`: Torch 엔진은 왼쪽 패딩 포함 폭, SGLang은 행별 실제 길이입니다.
    """)
    return


@app.cell
def _(torch):
    def response_logps(model, output_ids, attention_mask, prompt_lens, response_width):
        logits = model(output_ids, attention_mask=attention_mask).logits[:, :-1, :]
        next_logps = logits.log_softmax(-1).gather(2, output_ids[:, 1:, None]).squeeze(-1)
        positions = prompt_lens[:, None] - 1 + torch.arange(response_width)[None, :]
        return next_logps.gather(1, positions)

    def response_mask(completion_ids, valid_mask, eos_id):
        is_eos = (completion_ids == eos_id) & valid_mask.bool()
        eos_idx = torch.full((len(is_eos),), is_eos.size(1) - 1, dtype=torch.long)
        has_eos = is_eos.any(1)
        eos_idx[has_eos] = is_eos.int().argmax(1)[has_eos]
        return ((torch.arange(is_eos.size(1))[None, :] <= eos_idx[:, None]) & valid_mask.bool()).int()

    mask_ids = torch.tensor([[7, 2, 0, 0], [7, 8, 9, 0], [2, 2, 2, 2]])
    mask_valid = torch.tensor([[1, 1, 1, 1], [1, 1, 1, 0], [1, 1, 1, 1]])
    mask_demo = response_mask(mask_ids, mask_valid, 2)
    assert mask_demo.tolist() == [[1, 1, 0, 0], [1, 1, 1, 0], [1, 0, 0, 0]]
    return (mask_demo, mask_ids, mask_valid, response_logps, response_mask,)


@app.cell(hide_code=True)
def _(mask_demo, mask_ids, mask_valid, mo):
    # ▸ 결과 표시 — 실행하면 표·그림이 나타납니다 (marimo 표시 코드)
    mo.ui.table([{"행": i, "응답 ID": str(ids.tolist()), "유효 길이": str(v.tolist()),
                  "최종 마스크": str(m.tolist())}
                 for i, (ids, v, m) in enumerate(zip(mask_ids, mask_valid, mask_demo))], selection=None)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 5. GRPO 목적함수와 CISPO 분기

    선택 토큰의 log prob: 현재 정책 $\ell$, 생성 당시 $\ell_{old}$, reference $\ell_{ref}$.

    $$\rho=\exp(\ell-\ell_{old}),\qquad
    k=\exp(\ell_{ref}-\ell)-(\ell_{ref}-\ell)-1$$
    $$L_{it}=-\min(\rho_{it}A_i,\;\operatorname{clip}(\rho_{it},1-\epsilon,1+\epsilon)A_i)+\beta k_{it}$$

    - old와 reference는 고정합니다. **현재 logps만 미분**합니다.
    - advantage가 양수면 확률을 올리고, 음수면 내립니다.
    - clip과 $k$는 PPO 노트북 6~7절과 같습니다.

    **PPO와 다른 점은 advantage의 출처뿐입니다.**

    - PPO: Critic·GAE로 토큰별 advantage
    - GRPO: 같은 질문 응답들의 평균·표준편차로 응답별 advantage 하나

    원본 로그의 `KL_ref`는 단순 `ref_logp - logp` 평균입니다. 위 $k$와 다르고 음수일 수 있습니다.

    **평균 방식:** 응답마다 토큰 평균 → 응답끼리 평균.
    전체 토큰 평균(긴 응답의 비중이 커짐)과 다릅니다.

    **주의:** `--loss_type` 기본값은 **cispo**입니다. GRPO는 `--loss_type grpo`로 지정하세요.
    CISPO 식은 `clamp(ratio, max=epsilon_high).detach() * A * logp`입니다. PPO clip과 다릅니다.
    """)
    return


@app.cell
def _(torch):
    def policy_objective(logps, old_logps, ref_logps, advantages, mask,
                         beta=0.1, epsilon=0.2, loss_type="grpo", epsilon_high=5.0):
        delta = ref_logps - logps
        token_kl = delta.exp() - delta - 1
        ratio = (logps - old_logps).exp()
        if loss_type == "cispo":
            surrogate = ratio.clamp(max=epsilon_high).detach() * advantages[:, None] * logps
        else:
            surrogate = torch.minimum(ratio * advantages[:, None],
                                      ratio.clamp(1-epsilon, 1+epsilon) * advantages[:, None])
        token_loss = -surrogate + beta * token_kl
        loss = ((token_loss * mask).sum(1) / mask.sum(1).clamp(min=1)).mean()
        return loss, ratio, token_kl

    # ratio=1이고 손실 값이 0이어도 gradient가 0인 것은 아닙니다.
    probe = torch.tensor([[-1.], [-1.]], requires_grad=True)
    probe_loss, _, _ = policy_objective(probe, probe.detach(), probe.detach(),
                                         torch.tensor([1., -1.]), torch.ones(2, 1))
    probe_loss.backward()
    assert probe.grad[0] < 0 and probe.grad[1] > 0
    return (policy_objective, probe, probe_loss,)


@app.cell(hide_code=True)
def _(mo, probe, probe_loss):
    # ▸ 결과 설명 — 실행하면 계산 값이 들어간 설명이 나타납니다 (marimo 표시 코드)
    mo.md(f"ratio=1, advantage=[+1, -1] 확인: loss={probe_loss.item():.3f}, "
          f"logp의 gradient={probe.grad.flatten().tolist()}. 경사 하강에서는 gradient가 음수인 logp가 올라갑니다.")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 6. 작은 SFT 정책 준비

    여기부터 실제 `MiniMindForCausalLM`을 씁니다.

    - **교육용 사전:** 토큰 하나 = 숫자 하나. 앞의 chat tokenizer와 다릅니다.
    - 질문: `2 + 2 =`, `2 + 3 =`. 응답: 숫자 한 토큰.

    **준비:** SFT 예시 10개로 학습합니다.

    - 질문마다 정답 3번, 오답 2번을 보여 줍니다.
    - 그래야 정답과 오답이 모두 나오는 정책이 됩니다.
    - 이 단계만 정답 labels를 씁니다. GRPO는 reward만 씁니다.

    실제로는 이미 언어 능력이 있는 SFT 체크포인트에서 시작합니다.
    """)
    return


@app.cell
def _(MiniMindConfig, MiniMindForCausalLM, torch):
    toy_vocab = ["<pad>", "<bos>", "<eos>", "+", "=", "2", "3", "4", "5"]
    toy_prompts = torch.tensor([[1, 5, 3, 5, 4], [1, 5, 3, 6, 4]])
    toy_targets = torch.tensor([7, 8])
    torch.manual_seed(42)
    toy_config = MiniMindConfig(hidden_size=32, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, intermediate_size=64,
        max_position_embeddings=32, vocab_size=len(toy_vocab), dropout=0.0,
        flash_attn=False, use_moe=False, bos_token_id=1, eos_token_id=2)
    sft_policy = MiniMindForCausalLM(toy_config)
    _sft_inputs = toy_prompts.repeat_interleave(5, 0)
    _sft_answers = torch.tensor([7, 7, 7, 8, 8, 7, 7, 8, 8, 8])
    _sft_optimizer = torch.optim.AdamW(sft_policy.parameters(), lr=0.005)
    for _step in range(120):
        _logits = sft_policy(_sft_inputs).logits[:, -1, :]
        _loss = torch.nn.functional.cross_entropy(_logits, _sft_answers)
        _sft_optimizer.zero_grad()
        _loss.backward()
        _sft_optimizer.step()
    sft_policy.eval()
    with torch.no_grad():
        initial_probs = sft_policy(toy_prompts).logits[:, -1].softmax(-1)
    return (initial_probs, sft_policy, toy_config, toy_prompts, toy_targets, toy_vocab,)


@app.cell(hide_code=True)
def _(initial_probs, mo, toy_prompts, toy_targets, toy_vocab):
    # ▸ 결과 표시 — 실행하면 표·그림이 나타납니다 (marimo 표시 코드)
    mo.ui.table([{"질문": " ".join(toy_vocab[i] for i in ids[1:]),
                  "P(4)": float(probs[7]), "P(5)": float(probs[8]),
                  "정답 확률": float(probs[target])}
                 for ids, probs, target in zip(toy_prompts, initial_probs, toy_targets)], selection=None)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 7. 생성 → 보상 → backward를 60회 반복하기

    **설정:** B=2, G=8, 응답은 한 토큰.

    - 매 step 현재 정책으로 새 응답을 만들고 `old_logps`를 저장합니다.
    - reference는 SFT 정책의 고정 사본입니다.
    - 보상: 정답이면 1, 아니면 0. 원본 보상 모델의 **대체 과제**입니다.

    **샘플링:** `temperature=1, top_k=0, top_p=1`.
    원본은 temperature 0.8로 생성하고 logps는 원래 logits로 다시 계산합니다.
    여기서는 그 차이를 없애고 손실의 의미부터 봅니다.

    **ratio:** rollout마다 한 번만 갱신하므로 ratio는 항상 약 1입니다.
    **clip 효과는 PPO 노트북 6절에서 확인하세요.**
    rollout을 재사용하거나 생성 정책이 뒤처지면 ratio가 1에서 멀어집니다.

    **생략한 것:** MoE aux, AMP, DDP, scheduler, 원격 rollout. (Dense·FP32·누적 1)
    이 실습은 `weight_decay=0`으로 두고 보상·KL의 효과를 확인합니다.
    AdamW의 기본 weight decay는 gradient가 0이어도 가중치를 바꿀 수 있습니다.
    """)
    return


@app.cell
def _(copy, group_advantages, policy_objective, response_logps, response_mask, sft_policy, torch, toy_prompts, toy_targets):
    policy = copy.deepcopy(sft_policy).train()
    reference = copy.deepcopy(sft_policy).eval().requires_grad_(False)
    _ref_before = {k: v.clone() for k, v in reference.state_dict().items()}
    # G=1 과제에서 보상 gradient가 없는 경우와 weight decay를 분리합니다.
    _grpo_optimizer = torch.optim.AdamW(policy.parameters(), lr=0.0003, weight_decay=0.0)
    train_G, train_beta, train_steps = 8, 0.01, 60
    _expanded = toy_prompts.repeat_interleave(train_G, 0)
    _gold = toy_targets.repeat_interleave(train_G)
    _prompt_widths = torch.full((len(_expanded),), toy_prompts.size(1), dtype=torch.long)
    training_history = []
    first_rollout = []
    torch.manual_seed(123)
    for _step in range(train_steps):
        with torch.no_grad():
            _outputs = policy.generate(input_ids=_expanded, attention_mask=torch.ones_like(_expanded),
                max_new_tokens=1, temperature=1.0, top_k=0, top_p=1.0, do_sample=True,
                eos_token_id=2).clone()  # generate의 inference tensor를 일반 tensor로 복사
            _completion = _outputs[:, toy_prompts.size(1):]
            _full_mask = torch.ones_like(_outputs)
            _old = response_logps(policy, _outputs, _full_mask, _prompt_widths, 1).detach()
            _ref = response_logps(reference, _outputs, _full_mask, _prompt_widths, 1)
            _rewards = (_completion[:, 0] == _gold).float()
            _advantages = group_advantages(_rewards, train_G)
        _current = response_logps(policy, _outputs, _full_mask, _prompt_widths, 1)
        _mask = response_mask(_completion, torch.ones_like(_completion), 2)
        _policy_loss, _ratio, _kl = policy_objective(
            _current, _old, _ref, _advantages, _mask, beta=train_beta)
        assert torch.allclose(_ratio, torch.ones_like(_ratio), atol=1e-5)
        _grpo_optimizer.zero_grad()
        _policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        _grpo_optimizer.step()
        with torch.no_grad():
            _probs = policy(toy_prompts).logits[:, -1].softmax(-1)
            _expected_reward = _probs.gather(1, toy_targets[:, None]).mean().item()
        training_history.append({"step": _step + 1, "sample_reward": _rewards.mean().item(),
            "expected_reward": _expected_reward, "loss": _policy_loss.item(),
            "kl_penalty": (_kl * _mask).sum().item() / _mask.sum().item(),
            "mixed_group_rate": (_rewards.view(-1, train_G).std(1, unbiased=False) > 0).float().mean().item()})
        if _step == 0:
            first_rollout = [{"질문": i // train_G, "생성 ID": int(c), "reward": float(r), "advantage": float(a)}
                             for i, (c, r, a) in enumerate(zip(_completion[:, 0], _rewards, _advantages))]
    assert all(torch.equal(v, _ref_before[k]) for k, v in reference.state_dict().items())
    assert all(p.grad is None for p in reference.parameters())
    return (first_rollout, policy, reference, train_G, train_beta, train_steps, training_history,)


@app.cell(hide_code=True)
def _(first_rollout, mo, training_history):
    # ▸ 결과 표시 — 실행하면 표·그림이 나타납니다 (marimo 표시 코드)
    mo.vstack([mo.md("**첫 rollout: 같은 질문 안에서 reward와 advantage를 비교하세요.**"),
               mo.ui.table(first_rollout, selection=None),
               mo.ui.table(training_history[::5] + training_history[-1:], selection=None)])
    return


@app.cell(hide_code=True)
def _(training_history, plt, mo, run_dir):
    # ▸ 결과 표시 — 실행하면 표·그림이 나타납니다 (marimo 표시 코드)
    _fig, _ax = plt.subplots(figsize=(8, 3))
    _ax.plot([r["step"] for r in training_history], [r["sample_reward"] for r in training_history],
             alpha=0.6, label="sample reward (16 responses)")
    _ax.plot([r["step"] for r in training_history], [r["expected_reward"] for r in training_history],
             label="expected reward (exact on 2 prompts)")
    _ax.set(xlabel="Optimizer step", ylabel="Reward", ylim=(-0.05, 1.05))
    _ax.legend()
    _fig.tight_layout()
    _fig.savefig(run_dir / "reward_curve.png", dpi=150)
    mo.as_html(_fig)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 8. 학습 전후를 비교하고 결과 저장하기

    **관찰할 값:**

    - 샘플 보상은 뽑힌 16개 응답에 따라 달라집니다.
    - 정확한 기대 보상 = 두 질문의 정답 확률 평균.
    - 학습 손실은 SFT처럼 꾸준히 줄지 않습니다. advantage·ratio·KL에 따라 움직입니다.

    아래 표는 두 질문의 확률과 greedy 답을 비교합니다.
    둘 다 학습에 쓴 질문이므로 **과제 안에서의 변화 확인**입니다.
    일반화, 긴 문장 생성, 실제 체크포인트 성능은 검증하지 않습니다.
    """)
    return


@app.cell
def _(first_rollout, initial_probs, json, policy, run_dir, torch, toy_prompts, toy_targets, toy_vocab, train_G, train_beta, train_steps, trainer_source, training_history):
    with torch.no_grad():
        final_probs = policy(toy_prompts).logits[:, -1].softmax(-1)
    comparison = [{"질문": " ".join(toy_vocab[i] for i in ids[1:]),
                   "정답": toy_vocab[target], "학습 전 정답 확률": float(before[target]),
                   "학습 후 정답 확률": float(after[target]),
                   "학습 전 greedy": toy_vocab[int(before.argmax())],
                   "학습 후 greedy": toy_vocab[int(after.argmax())]}
                  for ids, target, before, after in zip(toy_prompts, toy_targets, initial_probs, final_probs)]
    metrics = {"config": {"seed_sft": 42, "seed_grpo": 123, "B": 2, "G": train_G,
                          "beta": train_beta, "steps": train_steps, "loss_type": "grpo"},
               "comparison": comparison, "history": training_history, "first_rollout": first_rollout,
               "scope": "CPU toy one-token GRPO; not external reward model or full trainer"}
    (run_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "train_grpo_source.py.txt").write_text(trainer_source, encoding="utf-8")
    return (comparison, metrics,)


@app.cell(hide_code=True)
def _(comparison, mo, run_dir):
    # ▸ 결과 표시 — 실행하면 표·그림이 나타납니다 (marimo 표시 코드)
    mo.vstack([mo.ui.table(comparison, selection=None), mo.md(f"실행 결과 저장 경로: `{run_dir}`")])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 9. 원본 코드로 돌아가기

    | 노트북 | `train_grpo.py`에서 찾을 부분 |
    | --- | --- |
    | 프롬프트 만들기 | `RLAIFDataset` → `batch['prompt']` → 왼쪽 패딩·절단 |
    | G개 응답 | `rollout_engine.rollout()`와 `num_generations` |
    | 이번 생성의 기준 | `old_per_token_logps.detach()` |
    | 고정 기준 모델 | `ref_model.eval().requires_grad_(False)` |
    | 응답 점수 | `calculate_rewards()` 또는 GSM8K 규칙 보상 |
    | 그룹 상대 신호 | `view(-1, G)`, 평균, `std(unbiased=False)` |
    | 선택 토큰 확률 | log_softmax·gather·`logp_pos` |
    | 손실·마스크 | `completion_mask`, ratio, KL, grpo/cispo 분기 |
    | 갱신 | `(policy_loss + aux_loss) / accumulation_steps` → backward → clip → step |

    **엔진 차이:** Torch 엔진은 학습 정책을 그대로 씁니다.
    SGLang은 정책을 주기적으로 동기화합니다. 그래서 old logps와 정책 지연을 함께 봐야 합니다.

    **누적:** 원본은 여러 microbatch에 걸쳐 누적합니다.
    덜 찬 마지막 그룹도 `accumulation_steps`로 나눕니다.

    **실제 대화 모델로 확장할 때 필요한 것:**

    - `rlaif.jsonl` (마지막이 assistant)
    - 구조가 맞는 `full_sft_*.pth`
    - 로컬 tokenizer
    - 외부 보상 모델과 실행 환경
    - GSM8K 경로는 별도 데이터 클래스와 `trainer/gsm8k_reward.py`

    이 노트북의 CPU 환경은 원본 GPU trainer 환경이 아닙니다.

    ## 10. 수강생 과제와 확인 기준

    1. **G를 8→1로 바꾸기 전에 예측하세요.** 새 optimizer에서 정책 = reference,
       advantage = 0이고 이 실습처럼 `weight_decay=0`이면 업데이트가 없습니다.
       이미 쌓인 optimizer 상태를 재사용하는 경우와는 구분하세요.
       G, beta, 전후 확률을 적으세요.
    2. **보상을 `[0,0,0]`, `[1,1,1]`로 바꾸세요.** 둘 다 advantage가 0입니다.
       reward 평균과 mixed group 비율을 적으세요.
    3. **`train_beta`를 0 / 0.01 / 0.1로 비교하세요.** 기대 보상, KL, 정답 확률을 적으세요.
       작은 과제라 변화가 일정하지 않아도 됩니다.
    4. **EOS 마스크를 모두 1로 바꾸면?** 4절에서 빠져야 할 위치를 적으세요.
       7절은 길이 1이라 차이가 없습니다. 긴 예제로 확인하세요.
    5. **`loss_type='cispo'`로 바꾸세요.** 식과 detach 위치를 설명하고 최종 확률을 비교하세요.
       손실 값 크기만으로 비교하지 마세요.
    6. **긴 응답 두 개라면?** "응답별 평균 → 전체 평균"과 "전체 토큰 평균"을 손으로 비교하세요.
       길이가 다르면 결과가 다릅니다.
    7. **3-1절 보상 표에 응답을 추가하세요.** 의미 없는 긴 think + 정답이면 두 보상은 몇 점일까요?
       형식 보상만으로 학습하면 어떤 위험이 있을까요?

    **제출:** 설정, 첫 그룹의 응답·보상·advantage, 마스크, 전후 정답 확률, 기대 보상·KL 곡선과 설명.
    실행 결과는 `outputs/notebooks/<실행 ID>/`, 보고서는 `reports/notebooks/<실행 ID>.md`.

    **원본·라이선스:** `minimind/trainer/train_grpo.py`, `minimind/trainer/gsm8k_reward.py`,
    `minimind/dataset/lm_dataset.py`, `minimind/model/model_minimind.py`, `minimind/LICENSE`.

    **다음 파트:** 텍스트 LLM을 마쳤습니다. 이제 멀티모달입니다.

    - `minimind_vlm.py`: 이미지 입력 구조
    - `minimind_v_finetuning.py`: VLM 2단계 학습
    - `minimind_omni.py`: 음성 입출력
    """)
    return


if __name__ == "__main__":
    app.run()
