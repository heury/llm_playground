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
        datetime.now().astimezone().strftime("%Y%m%d-%H%M%S") + "-" + uuid4().hex[:8] + "-grpo"
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

    **학습 목표:** `train_grpo.py`와 `lm_dataset.py`를 읽고, 질문 → 여러 응답 → 보상 →
    그룹 내 advantage → 응답 토큰의 log probability → 손실 → 가중치 갱신을 설명합니다.
    또한 `<think>` 블록을 쓰는 **추론 모델에 어떤 보상을 주는지**를 원본 보상 함수로 확인합니다.
    SFT → DPO → PPO 다음 순서를 권장합니다. PPO의 ratio·clip·reference KL은 다시 자세히 설명하지 않습니다.

    1. 실제 `RLAIFDataset`과 저장소의 토크나이저로 입력 형식을 확인합니다.
    2. 작은 숫자로 보상·advantage를 계산하고, think 형식 보상과 정답(GSM8K) 보상을 비교합니다.
    3. 마스크·GRPO/CISPO 손실을 계산합니다.
    4. 작은 실제 MiniMind에서 응답을 샘플링하고 CPU로 GRPO fine tuning합니다.

    모델·데이터 다운로드와 GPU가 필요하지 않습니다. 마지막 학습은 두 질문과 한 토큰 응답으로
    축소한 교육용 실험이며, 한국어 대화 성능이나 GSM8K 성능을 평가하지 않습니다.
    원본 trainer 전체나 외부 보상 모델을 실행하는 실습은 아닙니다.
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

    **Critic 없이** 그룹 통계를 씁니다. 별도 reward model이 없는 것과는 다릅니다.
    정책은 SFT 모델에서 시작하고 reference는 시작 시 정책의 고정 복사본입니다.
    old policy는 이번 응답을 생성했을 때의 정책이며 reference와 역할이 다릅니다.

    **예측:** 질문마다 응답을 하나만 만들면 상대 비교가 가능할까요?
    아래에서 G=1 또는 보상이 모두 같을 때 advantage가 어떻게 되는지 확인합니다.
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

    `create_chat_prompt()`는 `conversations[:-1]`을 템플릿에 넣습니다.
    **마지막 메시지가 assistant 정답인 데이터**를 전제로 하므로 마지막이 user이면 질문 자체가 빠집니다.
    반환값은 `{'prompt': 문자열, 'answer': ''}`입니다. 원본 model 보상 경로는 이 빈 answer를 사용하지 않습니다.
    GSM8K 경로는 별도 `GSM8KDataset`과 숫자 정답을 사용합니다.

    `pre_processing_chat()`은 system이 없으면 확률적으로 추가합니다. 여기서는 system을 명시했습니다.
    `thinking_ratio`는 `open_thinking`으로 템플릿에 전달됩니다. thinking의 품질 점수는 아닙니다.
    `max_length`는 저장만 하며 여기서 절단하지 않습니다. 실제 절단은 trainer가 토큰화한 뒤 수행합니다.

    **관찰:** 위에서는 `max_length=8`을 주어도 문자열 전체가 반환됩니다.
    원본은 `padding_side='left'`, `add_special_tokens=False`로 토큰화하고 마지막 `max_seq_len`개를 남깁니다.
    질문을 자르면 과제가 바뀔 수 있으며, GSM8K 경로는 초과 시 오류를 냅니다.
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

    응답은 **질문 우선 순서**입니다: `q0/a0, q0/a1, q0/a2, q1/a0, ...`.
    그래야 `rewards.view(B, G)`에서 서로 다른 질문이 섞이지 않습니다.

    $$A_i = \frac{r_i - \operatorname{mean}(r_{group})}
    {\operatorname{std}(r_{group},\;unbiased=False)+10^{-4}}$$

    응답 하나의 advantage를 그 응답의 모든 유효 생성 토큰에 공유합니다.
    정답 단어가 어느 위치에 있었는지 직접 알려 주는 토큰별 보상은 아닙니다.

    **예측:** `[0, 1, 2]`와 `[5, 5, 5]`의 advantage를 계산하세요.
    평균이 5인 두 번째 그룹이 더 좋은 학습 신호를 낼까요?
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
    **해설:** 첫 그룹은 약 `[-1.2246, 0, 1.2246]`, 두 번째 그룹은 모두 0입니다.
    전부 오답인 경우뿐 아니라 **전부 정답인 경우도** policy 보상 항이 0이 됩니다.
    이때도 정책이 reference와 달라져 있으면 KL 항이 작동할 수 있습니다.
    전체 advantage 평균이 0이라는 사실만으로 학습 성공을 판단할 수 없습니다.

    ### 원본 model 보상은 어떻게 구성되는가?

    `calculate_rewards()`는 전체 응답 길이(문자 수), `</think>` 앞부분 길이·태그 수,
    답변 부분의 반복 패널티, `reward_model.get_score(messages, answer)`를 합산합니다.
    `rep_penalty()`는 정규식으로 나눈 토큰의 n-gram 반복을 셉니다. 모델 토크나이저 ID가 아닙니다.
    태그 검사만으로 추론이 정확하다고 판단할 수 없습니다. 보상만 높이는 우회 행동도 관찰해야 합니다.

    다음 셀은 **원본 두 함수만 AST로 읽어 실행**합니다. trainer import의 전역 설정과 외부 의존성을 피하며,
    보상 모델 점수는 항상 0.2인 모형(stub)으로 대체합니다. 실제 보상 모델의 품질 검증은 아닙니다.
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

    SFT 노트북 5-1절에서 본 것처럼 MiniMind의 assistant 응답은 `<think>추론</think>` 뒤에 최종 답을 씁니다.
    SFT는 사람이 쓴 추론 문장을 **따라 쓰게** 하고, GRPO는 모델이 **스스로 쓴** 추론과 답을 보상으로 평가합니다.

    **프롬프트:** `RLAIFDataset`은 `thinking_ratio` 확률로 `open_thinking=True`를 template에 넘깁니다
    (원본 GRPO 기본 0.9). 그러면 프롬프트가 `<think>\n`에서 끝나 모델은 추론부터 이어 씁니다.
    False이면 빈 블록 `<think>\n\n</think>\n\n`까지 넣어 바로 답하게 합니다.

    | 보상 경로 | 무엇을 점수로 주는가 | think에 대한 처리 |
    | --- | --- | --- |
    | `calculate_rewards` (기본, 보상 모델) | 전체 길이 + think 길이·태그 형식 + 반복 패널티 + 보상 모델 점수 | think 20~300자 +1.0, 아니면 −0.5; `</think>` 정확히 1개 +0.25, 아니면 −0.25 |
    | `score_gsm8k_response` (`--reward_type gsm8k`) | 마지막 줄 `#### 숫자`가 정답과 같으면 1, 아니면 0 | 닫힌 think 0~1개만 허용. **형식만으로는 점수 없음** |

    **예측:** 추론 형식은 같고 최종 숫자만 틀린 응답은 두 보상에서 각각 어떤 점수를 받을까요?
    아래 응답은 `open_thinking=True` 프롬프트 뒤에 생성된 completion이라고 가정합니다.
    GSM8K 프롬프트는 빈 think를 제거해 `assistant\n`에서 끝나므로, 모델이 `<think>`부터 직접 씁니다.
    그래서 GSM8K 채점에는 앞에 `<think>\n`을 붙인 같은 응답을 넣습니다.
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
        mo.md("**해설:** 기본 보상은 **추론의 모양**(길이·태그 수)에 점수를 주므로 숫자가 틀려도 같은 점수를 받습니다. "
              "실제로는 보상 모델 점수가 내용을 평가하지만, 형식 항만 높이는 우회 행동(reward hacking)이 생길 수 있습니다. "
              "GSM8K 보상은 **최종 답의 정오**만 보므로 짧은 추론도 정답이면 1점입니다. "
              "추론 모델의 GRPO는 이런 보상을 같은 질문의 여러 응답 사이에서 비교해, 더 나은 추론·답을 쓴 응답의 확률을 높입니다. "
              "여기서는 보상 계산만 확인합니다. 아래 7절의 학습은 한 토큰 산술 과제로 축소합니다."),
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 4. 다음 토큰 정렬과 EOS 마스크

    `logits[:, t]`는 `input_ids[:, t+1]`의 분포입니다.
    `log_softmax → gather(실제 다음 토큰 ID)`로 `[N, T-1]`을 만들고,
    `logp_pos = prompt_lens - 1 + arange(R)`로 응답 예측 위치만 고릅니다.

    아래에서 P=3이고 응답이 `[7, EOS, PAD, PAD]`이면 logits의 **2, 3, 4, 5번**을 읽습니다.
    그중 손실에는 첫 EOS까지 **포함**한 두 위치만 씁니다. 프롬프트와 EOS 이후 위치는 제외합니다.
    EOS가 없으면 rollout의 유효 길이까지 사용합니다.
    `prompt_lens`는 Torch 엔진에서 왼쪽 패딩을 포함한 프롬프트 폭이며,
    SGLang 엔진에서는 행별 실제 프롬프트 길이입니다.
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

    현재 정책의 선택 토큰 log probability를 $\ell$, 생성 당시 값을 $\ell_{old}$,
    고정 reference 값을 $\ell_{ref}$라고 합시다.

    $$\rho=\exp(\ell-\ell_{old}),\qquad
    k=\exp(\ell_{ref}-\ell)-(\ell_{ref}-\ell)-1$$
    $$L_{it}=-\min(\rho_{it}A_i,\;\operatorname{clip}(\rho_{it},1-\epsilon,1+\epsilon)A_i)+\beta k_{it}$$

    `old`와 `reference` logps는 detach/no_grad로 고정하고 **현재 logps만 미분**합니다.
    양의 advantage는 선택 응답 확률을 높이는 방향, 음의 값은 낮추는 방향입니다.
    clip과 $k$(reference KL 추정)는 PPO 노트북 6~7절과 같은 식입니다. 부호별 clip gradient 표는 그곳에서 확인하세요.
    **PPO와의 차이는 advantage의 출처뿐**입니다. PPO는 Critic·GAE로 토큰별 advantage를 만들고,
    GRPO는 같은 질문 응답들의 보상 평균·표준편차로 응답별 advantage 하나를 만들어 모든 토큰에 공유합니다.
    원본 로그의 `KL_ref`는 단순 `ref_logp - logp` 평균이므로 이 항과 다르고 음수일 수도 있습니다.

    원본은 응답마다 유효 토큰 평균을 낸 뒤 응답 평균을 냅니다. 긴 응답이 토큰 수만큼
    더 큰 가중치를 받는 전체 토큰 평균과 구분하세요.

    **주의:** 원본 `--loss_type` 기본값은 **cispo**입니다. GRPO를 선택하려면 `--loss_type grpo`를 명시합니다.
    CISPO는 `clamp(ratio, max=epsilon_high).detach() * A * logp`를 사용하며 PPO clip과 다른 식입니다.
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
          f"logp의 gradient={probe.grad.flatten().tolist()}. 경사 하강은 gradient가 음수인 logp를 높입니다.")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 6. 작은 SFT 정책 준비

    아래부터는 실제 `MiniMindForCausalLM`을 사용합니다. 한 토큰이 숫자 하나를 뜻하는
    **별도 교육용 사전**을 쓰며, 앞의 실제 chat tokenizer와 ID 체계가 다릅니다.
    질문은 `2 + 2 =`와 `2 + 3 =`, 응답은 숫자 한 토큰입니다.

    GRPO 전에 열 SFT 예시로 모델을 준비합니다. 각 질문에 정답을 세 번, 오답을 두 번 보여 주어
    두 답을 모두 생성할 수 있는 출발점을 만듭니다. 여러 답이 나오는 정책을 만드는 준비 단계입니다.
    실제 학습에서는 이미 언어 능력을 갖춘 SFT 체크포인트에서 시작합니다.
    이 준비 단계에서는 정답 labels를 쓰고, 이후 GRPO에서는 생성 응답의 reward를 씁니다.
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

    B=2, G=8이며 한 응답은 한 토큰입니다. 매 step 현재 정책으로 새 응답을 생성하고
    `old_logps`를 no_grad로 저장합니다. reference는 SFT 정책의 고정 복사본입니다.
    정답 숫자와 일치하면 1, 아니면 0인 규칙 보상을 씁니다. 이것은 원본 model 보상의 **대체 과제**입니다.

    샘플링은 `temperature=1, top_k=0, top_p=1`로 하여 전체 softmax와 logps를 맞춥니다.
    원본 Torch rollout은 temperature 0.8과 generate 기본 필터로 생성하지만 logps는 원시 logits로
    재계산합니다. 여기서는 그 차이를 제거해 손실의 의미를 먼저 관찰합니다.

    매 rollout당 optimizer를 한 번 갱신하므로 갱신 전 ratio는 항상 약 1입니다.
    **clip 효과는 PPO 노트북 6절의 숫자 예제로 확인**합니다. 동일 rollout을 여러 번 재사용하거나
    생성 정책이 뒤처지면 ratio가 1에서 멀어질 수 있습니다.
    이 실습은 Dense·FP32·누적 1이며 MoE aux, AMP, DDP, scheduler와 원격 rollout은 생략합니다.
    """)
    return


@app.cell
def _(copy, group_advantages, policy_objective, response_logps, response_mask, sft_policy, torch, toy_prompts, toy_targets):
    policy = copy.deepcopy(sft_policy).train()
    reference = copy.deepcopy(sft_policy).eval().requires_grad_(False)
    _ref_before = {k: v.clone() for k, v in reference.state_dict().items()}
    _grpo_optimizer = torch.optim.AdamW(policy.parameters(), lr=0.0003)
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

    **관찰할 값:** 샘플 보상은 16개 응답의 우연에 흔들립니다.
    정확한 기대 보상은 두 질문 각각의 정답 토큰 확률 평균입니다.
    학습 손실은 advantage 부호, ratio, KL에 함께 좌우되므로 SFT loss처럼 단조 감소를 기대하지 마세요.

    아래 표는 같은 두 질문의 확률과 greedy 답을 비교합니다.
    두 질문은 모두 학습에 사용했으므로 이것은 **과제 내 행동 변화 확인**입니다.
    새 산술 문제의 일반화, 긴 문장 생성, 실제 SFT 체크포인트 성능은 검증하지 않습니다.
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

    Torch 엔진은 학습 정책 객체를 참조합니다. SGLang은 별도 정책을 저장 주기에 동기화하므로
    old logps의 의미와 정책 지연을 함께 살펴보세요. 원본의 누적은 여러 microbatch에 걸친 것이며,
    마지막 불완전 누적 그룹도 고정 `accumulation_steps`로 나누는 구현입니다.

    실제 대화 모델로 확장할 때 준비할 것은 `rlaif.jsonl`(마지막 assistant 포함),
    맞는 구조의 `full_sft_*.pth`, 로컬 tokenizer, 외부 보상 모델과 호환 실행 환경입니다.
    GSM8K 경로는 별도 데이터 클래스와 `trainer/gsm8k_reward.py`를 읽으세요.
    이 노트북의 CPU 환경을 원본 GPU trainer 실행 환경으로 간주하지 마세요.

    ## 10. 수강생 과제와 확인 기준

    1. **G를 8→1로 변경하기 전에 예측하세요.** 준비 정책과 reference가 같고 모든 advantage가
       0이면 보상 항의 업데이트가 없습니다. G와 beta, 학습 전후 확률을 기록하세요.
    2. **두 그룹의 보상을 각각 `[0,0,0]`, `[1,1,1]`로 바꾸세요.**
       둘 다 advantage가 0입니다. reward 평균과 mixed group 비율을 함께 기록하세요.
    3. **`train_beta`를 0 / 0.01 / 0.1로 비교하세요.** 같은 seed에서 기대 보상·KL·정답 확률을
       기록하세요. 이 작은 과제에서 수치 변화가 단조적일 필요는 없습니다.
    4. **EOS 마스크를 전부 1로 바꾸면?** 4절에서 제외되어야 할 위치를 적으세요.
       7절은 길이 1이므로 차이가 없습니다. 길이 여러 개인 예제에서 검증해야 합니다.
    5. **목적함수의 `loss_type='cispo'`로 바꾸세요.** surrogate 식과 detach 위치를 설명하고
       최종 확률을 비교하세요. 손실 값의 절댓값만으로 두 알고리즘을 평가하지 마세요.
    6. **긴 응답 두 개로 확장한다면?** 응답별 평균 후 전체 평균과 전체 토큰 평균을
       손으로 비교하세요. 응답 길이가 다를 때 서로 다른 목적함수입니다.

    **제출:** 설정, 첫 그룹의 응답·보상·advantage, 마스크, 전후 정답 확률,
    기대 보상·KL 곡선과 설명을 기록하세요. 실행 결과는 `outputs/notebooks/<실행 ID>/`,
    보고서는 `reports/notebooks/<실행 ID>.md`에 저장합니다.

    7. **추론 보상 표(3-1절)에 응답을 추가하세요.** think는 길지만 의미 없는 반복 문장과 정답을 쓰면
       두 보상은 각각 몇 점일까요? 형식 보상만으로 학습할 때의 위험을 설명하세요.

    원본 구현과 라이선스는 `minimind/trainer/train_grpo.py`, `minimind/trainer/gsm8k_reward.py`,
    `minimind/dataset/lm_dataset.py`, `minimind/model/model_minimind.py`, `minimind/LICENSE`를 확인하세요.

    **다음 파트:** 텍스트 LLM의 구조·학습·정렬·추론을 마쳤습니다. 멀티모달은 `minimind_vlm.py`(이미지 입력 구조)
    → `minimind_v_finetuning.py`(VLM 2단계 학습) → `minimind_omni.py`(음성 입출력)로 이어집니다.
    """)
    return


if __name__ == "__main__":
    app.run()
