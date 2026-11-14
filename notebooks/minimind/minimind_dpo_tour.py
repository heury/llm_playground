import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import sys as _display_sys
    from pathlib import Path as _DisplayPath

    _display_dir = str(_DisplayPath(__file__).resolve().parent)
    if _display_dir not in _display_sys.path:
        _display_sys.path.insert(0, _display_dir)
    from notebook_display import result_table

    return (result_table,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **실행 안내:** 저장소 루트에서 `cd notebooks/minimind` → `uv sync --locked` →
    `uv run marimo edit minimind_dpo_tour.py`. 환경·준비물과 권장 순서는 [공통 README](README.md)를 참고하세요.
    위젯 변경 시 의존 셀이 다시 실행됩니다. 모델 크기·데이터·학습 설정 변경 전에는 해당 절의 재실행 범위를 확인하세요.
    """)
    return


@app.cell
def _():
    import copy
    import json
    import math
    import sys
    import tempfile
    from contextlib import nullcontext
    from pathlib import Path

    import datasets
    import marimo as mo
    import matplotlib.pyplot as plt
    import torch
    import torch.nn.functional as F
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer

    # 노트북 위치(notebooks/minimind/)에서 두 단계 위가 저장소 루트입니다.
    repo_root = Path(__file__).resolve().parents[2]
    minimind_dir = repo_root / "projects/minimind"
    sys.path.insert(0, str(minimind_dir))

    # train_dpo.py의 함수와, 그 스크립트가 쓰는 원본 클래스를 그대로 가져옵니다.
    from dataset.lm_dataset import DPODataset, post_processing_chat
    from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
    from trainer.train_dpo import dpo_loss, logits_to_log_probs
    from trainer.trainer_utils import SkipBatchSampler, get_lr, setup_seed

    # GPU가 있으면 GPU, 없으면 CPU (train_dpo.py의 --device 기본값과 같은 규칙)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        torch.set_num_threads(4)  # CPU로 실행할 때 64M 모델 학습 속도를 위해 4개 스레드 사용

    def autocast_ctx():
        """train_dpo.py의 '3. 혼합 정밀도 설정': GPU는 bf16 autocast, CPU는 그대로 fp32"""
        return nullcontext() if device == "cpu" else torch.autocast(device_type="cuda", dtype=torch.bfloat16)

    datasets.disable_progress_bars()

    tokenizer = AutoTokenizer.from_pretrained(minimind_dir / "model")
    return (
        DPODataset,
        DataLoader,
        F,
        MiniMindConfig,
        MiniMindForCausalLM,
        Path,
        SkipBatchSampler,
        autocast_ctx,
        copy,
        device,
        dpo_loss,
        get_lr,
        hf_hub_download,
        json,
        load_file,
        logits_to_log_probs,
        math,
        mo,
        plt,
        post_processing_chat,
        setup_seed,
        tempfile,
        tokenizer,
        torch,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.vstack([
        mo.md(r"""
    # MiniMind DPO(Direct Preference Optimization) 따라가기

    SFT는 "좋은 답 하나"를 보여 주고 따라 하게 합니다. 그런데 좋은 답을 직접 쓰기는 어려워도,
    **두 답 중 어느 쪽이 나은지 고르기**는 쉬운 경우가 많습니다. DPO는 이런 **선호 쌍**
    (선호 답 `chosen`, 비선호 답 `rejected`)으로 모델을 조정합니다.
    MiniMind에서는 `projects/minimind/trainer/train_dpo.py`가 이 일을 합니다.

    이 노트북은 실제로 SFT까지 학습된 **64M MiniMind 모델**
    [`caspar/MiniMind-Dense-64M-Mixed-SFT-v1.0`](https://huggingface.co/caspar/MiniMind-Dense-64M-Mixed-SFT-v1.0)에서 출발합니다.
    `train_dpo.py`의 `--from_weight full_sft`에 해당하는 모델입니다.

    **준비:** 처음 실행할 때 Hugging Face에서 가중치(약 128MB)를 내려받으므로 인터넷이 필요합니다.
    **NVIDIA GPU가 있으면 자동으로 GPU를 씁니다**(`notebooks/minimind`의 uv 환경은 CUDA 13.2용 PyTorch를 설치합니다).
    GPU가 없으면 CPU로 실행되며 전체 2분 정도 걸립니다(4스레드 기준).

    **사전 지식:** SFT 노트북(`minimind_sft_tour.py`)의 채팅 템플릿과 "assistant 답변만 정답" 개념.

    ## 0. 큰 그림

    """),
        mo.mermaid(r"""
    flowchart LR
     P["사전 학습"] --> S["SFT: full_sft 가중치"]
     S --> D["DPO"]
     D --> A["policy: SFT 복사본, 학습"]
     D --> R["reference: SFT 복사본, 고정"]
    """),
        mo.md(r"""

    한 쌍에 대해 DPO가 하는 일은 한 문장입니다.
    **"기준 모델보다, chosen의 확률은 상대적으로 올리고 rejected의 확률은 상대적으로 내려라."**

    $$
    \mathcal{L}_{\mathrm{DPO}} = -\log \sigma\Big(\beta \big[
    \underbrace{(\log \pi_\theta(y_c|x) - \log \pi_\theta(y_r|x))}_{\text{policy의 선호 차이}} -
    \underbrace{(\log \pi_{\mathrm{ref}}(y_c|x) - \log \pi_{\mathrm{ref}}(y_r|x))}_{\text{reference의 선호 차이}}
    \big]\Big)
    $$

    **`train_full_sft.py`와 비교**

    | | SFT | DPO |
    | --- | --- | --- |
    | 데이터 | 대화 하나 (`conversations`) | 같은 질문의 두 답 (`chosen`, `rejected`) |
    | 모델 | 1개 | **2개**: 학습하는 policy + 고정된 reference |
    | 손실 | 정답 토큰의 cross entropy | **두 답의 로그 확률 차이**로 만든 `dpo_loss` |
    | 시작 가중치 | `pretrain` | `full_sft` |
    | 학습률 | 1e-5 | **4e-8** (SFT의 250분의 1) |
    | 그 외 | | `--beta 0.15`, `--batch_size 4`, `--epochs 1`, `--max_seq_len 1024` |

    별도의 보상 모델도, 학습 중 답을 생성하는 과정(rollout)도 없습니다. 미리 모아 둔 쌍으로만 학습하는 **오프라인** 방식입니다.
    """)
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. 베이스 모델 불러오기

    Hugging Face 저장소에는 `trust_remote_code`용 모델 코드도 들어 있지만, 여기서는 그 코드를 실행하지 않습니다.
    **설정(`config.json`)과 가중치(`model.safetensors`)만** 버전 `v1.0`으로 고정해 내려받고,
    저장소의 원본 `MiniMindForCausalLM`에 넣습니다. 그래서 지금까지 본 MiniMind 코드가 그대로 쓰입니다.

    모델 카드에 따르면 이 모델은 FineWeb-Edu로 사전 학습한 뒤 UltraChat·SmolTalk·GSM8K 대화로 SFT한 **영어** 모델입니다.
    카드는 "실험용이며, 반복·사실 오류·계산 오류가 있을 수 있다"고 밝힙니다.
    또 추론 프롬프트에서 **빈 think 블록을 지워야** SFT 학습 형식과 맞는다고 안내합니다. 이 노트북도 그렇게 합니다.
    """)
    return


@app.cell
def _(
    MiniMindConfig,
    MiniMindForCausalLM,
    device,
    hf_hub_download,
    json,
    load_file,
):
    BASE_REPO = "caspar/MiniMind-Dense-64M-Mixed-SFT-v1.0"
    BASE_REVISION = "v1.0"  # 버전 고정: 저장소가 바뀌어도 같은 가중치

    _config_path = hf_hub_download(BASE_REPO, "config.json", revision=BASE_REVISION)
    _weights_path = hf_hub_download(BASE_REPO, "model.safetensors", revision=BASE_REVISION)

    base_config = MiniMindConfig(**json.load(open(_config_path)))
    sft_model = MiniMindForCausalLM(base_config)
    _state = {k: v.float() for k, v in load_file(_weights_path).items()}  # fp16 → fp32
    _missing, _unexpected = sft_model.load_state_dict(_state, strict=False)
    sft_model.to(device).eval()  # 가중치는 fp32로 두고, 학습할 때만 bf16 autocast를 씁니다

    # lm_head는 Embedding과 같은 가중치라 파일에 따로 저장되지 않습니다.
    assert _missing == ["lm_head.weight"] and not _unexpected
    assert sft_model.lm_head.weight is sft_model.model.embed_tokens.weight
    base_param_count = sum(p.numel() for p in sft_model.parameters())
    return BASE_REPO, BASE_REVISION, base_config, base_param_count, sft_model


@app.cell(hide_code=True)
def _(
    BASE_REPO,
    BASE_REVISION,
    base_config,
    base_param_count,
    device,
    mo,
    torch,
):
    _where = f"GPU (`{torch.cuda.get_device_name(0)}`)" if device == "cuda" else "CPU"
    mo.md(f"""
    `{BASE_REPO}` (`{BASE_REVISION}`): 매개변수 **{base_param_count:,}개**, hidden {base_config.hidden_size},
    {base_config.num_hidden_layers}층, Query head {base_config.num_attention_heads}개 / KV head {base_config.num_key_value_heads}개, 어휘 {base_config.vocab_size}.

    실행 장치: **{_where}**
    """)
    return


@app.cell
def _(device, tokenizer, torch):
    def prompt_ids(question):
        """추론 프롬프트. 모델 카드 안내대로 빈 think 블록을 지웁니다."""
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": question}], tokenize=False, add_generation_prompt=True
        ).replace("<think>\n\n</think>\n\n", "")
        return tokenizer(text, add_special_tokens=False).input_ids

    @torch.inference_mode()
    def generate_answer(model, question, max_new_tokens=60):
        """greedy 생성. (답, 생성 토큰 수, eos로 멈췄는지)를 돌려줍니다."""
        x = torch.tensor([prompt_ids(question)], device=device)
        y = model.generate(inputs=x, max_new_tokens=max_new_tokens, do_sample=False, temperature=1.0,
                           top_p=1.0, top_k=0, eos_token_id=tokenizer.eos_token_id)
        new = y[0, x.shape[1]:].tolist()
        stopped = tokenizer.eos_token_id in new
        if stopped:
            new = new[: new.index(tokenizer.eos_token_id)]
        return tokenizer.decode(new, skip_special_tokens=True).strip(), len(new), stopped

    return (generate_answer,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. 선호 데이터 만들기: "짧고 정확하게 답하고 멈추기"

    이 SFT 모델은 간단한 질문에도 **장황하게 이어 쓰거나**, 답을 끝맺지 못하거나, 틀린 답을 냅니다.
    이 행동을 선호 쌍으로 고쳐 봅니다.

    - **chosen:** 사람이 쓴 짧고 정확한 한 문장 답
    - **rejected:** **SFT 모델이 실제로 생성한 답** (greedy, 최대 48토큰)

    rejected를 모델 자신의 출력으로 만들면 "모델이 실제로 하는 행동"을 직접 낮추게 됩니다.
    실제 선호 데이터(예: UltraFeedback)도 모델들의 생성 답을 사람이나 AI가 비교해 만듭니다.

    질문 32개 중 **24개는 학습**, **8개는 평가**(학습에 없던 질문)에 씁니다.
    아래 셀은 SFT 모델로 32개 답을 생성합니다(CPU에서는 20초 정도).
    """)
    return


@app.cell
def _(generate_answer, sft_model):
    train_qa = [
        ("What is the capital of France?", "The capital of France is Paris."),
        ("What color is the sky on a clear day?", "It is blue."),
        ("How many legs does a spider have?", "A spider has eight legs."),
        ("What is 2 + 3?", "2 + 3 = 5."),
        ("Name a fruit that is yellow.", "A banana is yellow."),
        ("What is the largest planet in our solar system?", "Jupiter is the largest planet in our solar system."),
        ("What do bees make?", "Bees make honey."),
        ("How many days are in a week?", "There are seven days in a week."),
        ("What is the boiling point of water at sea level in Celsius?", "Water boils at 100 degrees Celsius at sea level."),
        ("What is the opposite of hot?", "The opposite of hot is cold."),
        ("Which animal is known as the king of the jungle?", "The lion is known as the king of the jungle."),
        ("What gas do plants absorb from the air?", "Plants absorb carbon dioxide from the air."),
        ("How many continents are there?", "There are seven continents."),
        ("What is the freezing point of water in Celsius?", "Water freezes at 0 degrees Celsius."),
        ("Who wrote Romeo and Juliet?", "William Shakespeare wrote Romeo and Juliet."),
        ("What is the capital of Japan?", "The capital of Japan is Tokyo."),
        ("How many hours are in a day?", "There are 24 hours in a day."),
        ("What is the main language spoken in Brazil?", "Portuguese is the main language spoken in Brazil."),
        ("What is H2O commonly called?", "H2O is commonly called water."),
        ("What shape has three sides?", "A triangle has three sides."),
        ("Which planet is known as the Red Planet?", "Mars is known as the Red Planet."),
        ("What is the tallest animal in the world?", "The giraffe is the tallest animal in the world."),
        ("How many minutes are in an hour?", "There are 60 minutes in an hour."),
        ("What do cows drink?", "Cows drink water."),
    ]
    test_qa = [
        ("What is the capital of Italy?", "The capital of Italy is Rome."),
        ("How many legs does a dog have?", "A dog has four legs."),
        ("What color is grass?", "Grass is green."),
        ("What is 10 - 4?", "10 - 4 = 6."),
        ("Which ocean is the largest?", "The Pacific Ocean is the largest."),
        ("What is the chemical symbol for gold?", "The chemical symbol for gold is Au."),
        ("How many months are in a year?", "There are twelve months in a year."),
        ("What do caterpillars turn into?", "Caterpillars turn into butterflies."),
    ]

    # rejected: SFT 모델 자신의 답
    model_answers = {q: generate_answer(sft_model, q, max_new_tokens=48)[0] for q, _ in train_qa + test_qa}
    return model_answers, test_qa, train_qa


@app.cell
def _(
    DPODataset,
    Path,
    json,
    model_answers,
    tempfile,
    test_qa,
    tokenizer,
    train_qa,
):
    def chat(question, answer):
        return [{"role": "user", "content": question}, {"role": "assistant", "content": answer}]

    def pair_rows(qa):
        return [{"chosen": chat(q, a), "rejected": chat(q, model_answers[q])} for q, a in qa]

    data_dir = Path(tempfile.mkdtemp(prefix="minimind_dpo_data_"))
    for _name, _qa in [("dpo_train.jsonl", train_qa), ("dpo_test.jsonl", test_qa)]:
        with open(data_dir / _name, "w", encoding="utf-8") as _f:
            for _row in pair_rows(_qa):
                _f.write(json.dumps(_row, ensure_ascii=False) + "\n")

    dpo_max_len = 96  # 원본 기본값은 1024. 질문 + 48토큰 답이 들어가는 길이
    train_ds = DPODataset(str(data_dir / "dpo_train.jsonl"), tokenizer, max_length=dpo_max_len)
    test_ds = DPODataset(str(data_dir / "dpo_test.jsonl"), tokenizer, max_length=dpo_max_len)
    return data_dir, test_ds, train_ds


@app.cell(hide_code=True)
def _(data_dir, mo, model_answers, result_table, train_qa):
    _line = open(data_dir / "dpo_train.jsonl", encoding="utf-8").readline().strip()
    mo.vstack([
        result_table(
            [{"질문": q, "chosen (사람이 씀)": a, "rejected (SFT 모델의 답)": model_answers[q]} for q, a in train_qa],
            selection=None, page_size=8,
        ),
        mo.md(f"`dpo.jsonl`과 같은 형식의 한 줄:\n```json\n{_line[:600]}{'...' if len(_line) > 600 else ''}\n```"),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. 데이터: `DPODataset`

    `__getitem__`은 chosen과 rejected를 **각각 SFT처럼** 처리합니다.

    ```python
    chosen_prompt = tokenizer.apply_chat_template(chosen, tokenize=False, add_generation_prompt=False)
    remove_empty_think = random.random() > 0.2                       # 두 답에 "같은" 선택을 적용
    chosen_prompt = post_processing_chat(chosen_prompt, remove_empty_think=remove_empty_think)
    chosen_encoding = tokenizer(chosen_prompt, truncation=True, max_length=max_length, padding='max_length')
    loss_mask = self.generate_loss_mask(input_ids)                   # assistant 답변 위치 = 1
    x, y, mask = input_ids[:-1], input_ids[1:], loss_mask[1:]        # 여기서 한 칸 밀기
    ```

    SFT와 다른 점:

    - **한 칸 밀기를 데이터셋에서 합니다.** `x`가 입력, `y`가 각 위치에서 맞힐 다음 토큰, `mask`가 그 토큰이 답변인지 표시합니다.
      SFT는 `labels`를 그대로 두고 모델 `forward`가 밀었습니다. DPO는 `forward`에 `labels`를 주지 않고 직접 손실을 계산하기 때문입니다.
    - 정답 표시가 `-100` 대신 `0/1` **마스크**입니다.
    - 빈 think 블록 삭제 여부를 chosen과 rejected에 **똑같이** 적용합니다. 두 답이 형식 차이 없이 내용만 달라야 하기 때문입니다.
    """)
    return


@app.cell
def _(tokenizer, torch, train_ds):
    pair = train_ds[0]
    assert torch.equal(pair["x_chosen"][1:], pair["y_chosen"][:-1])  # x와 y는 한 칸 밀린 관계
    chosen_target = tokenizer.decode(pair["y_chosen"][pair["mask_chosen"] == 1])
    rejected_target = tokenizer.decode(pair["y_rejected"][pair["mask_rejected"] == 1])
    assert chosen_target.endswith("<|im_end|>\n") and rejected_target.endswith("<|im_end|>\n")
    return chosen_target, pair, rejected_target


@app.cell(hide_code=True)
def _(chosen_target, mo, pair, rejected_target, result_table, tokenizer):
    def _rows(prefix):
        x, y, m = pair[f"x_{prefix}"], pair[f"y_{prefix}"], pair[f"mask_{prefix}"]
        n = int((x != tokenizer.pad_token_id).sum())
        return [{"위치": t, "x (입력)": tokenizer.convert_ids_to_tokens(int(x[t])).replace("Ċ", "\\n"),
                 "y (다음 토큰)": tokenizer.convert_ids_to_tokens(int(y[t])).replace("Ċ", "\\n"),
                 "mask": int(m[t])} for t in range(n)]

    mo.vstack([
        mo.hstack([
            mo.vstack([mo.md("**chosen**"), result_table(_rows("chosen"), selection=None, page_size=30)]),
            mo.vstack([mo.md("**rejected**"), result_table(_rows("rejected"), selection=None, page_size=30)]),
        ], widths="equal"),
        mo.md(f"`mask=1`인 다음 토큰: chosen `{chosen_target!r}` ({int(pair['mask_chosen'].sum())}개), "
              f"rejected `{rejected_target!r}` ({int(pair['mask_rejected'].sum())}개). 질문과 템플릿 부분은 두 시퀀스가 같습니다."),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. 답 하나의 로그 확률: `logits_to_log_probs`

    DPO는 "모델이 이 답을 낼 확률"을 비교합니다. 답의 확률은 토큰 확률의 곱이므로, 로그를 취하면 **합**이 됩니다.

    $$
    \log \pi(y|x) = \sum_{t \in \text{답변}} \log p(y_t \mid x, y_{<t})
    $$

    ```python
    def logits_to_log_probs(logits, labels):
        log_probs = F.log_softmax(logits, dim=2)                          # [B, T, V]
        return torch.gather(log_probs, dim=2, index=labels.unsqueeze(2)).squeeze(-1)   # [B, T]: 실제 다음 토큰의 log p

    # dpo_loss 안에서
    policy_log_probs = (policy_log_probs * mask).sum(dim=1)               # 답변 위치만 더함 → [B]
    ```

    **평균이 아니라 합**이라는 점이 중요합니다. 긴 답일수록 더할 항이 많아 로그 확률이 더 작아집니다(더 음수).
    SFT 모델로 학습 쌍 전체의 값을 계산하고, 첫 쌍은 직접 계산과 비교합니다.
    """)
    return


@app.cell
def _(
    F,
    device,
    logits_to_log_probs,
    pair,
    post_processing_chat,
    sft_model,
    tokenizer,
    torch,
):
    def fixed_eval_items(ds):
        """평가용 입력. DPODataset.__getitem__과 같은 처리지만 무작위 가공을 끕니다.

        __getitem__은 꺼낼 때마다 20% 확률로 빈 think 블록을 남기므로, 같은 쌍도 호출마다 입력이 달라집니다.
        reference와 DPO 모델을 비교하려면 두 모델이 '똑같은' 입력을 봐야 하므로, 빈 think를 항상 지운
        (생성 프롬프트와 같은) 형식으로 한 번만 만들어 재사용합니다.
        """
        items = []
        for sample in ds.samples:
            item = {}
            for kind in ("chosen", "rejected"):
                text = tokenizer.apply_chat_template(sample[kind], tokenize=False, add_generation_prompt=False)
                text = post_processing_chat(text, remove_empty_think=True)
                ids = tokenizer(text, truncation=True, max_length=ds.max_length, padding="max_length")["input_ids"]
                mask = ds.generate_loss_mask(ids)
                item[f"x_{kind}"] = torch.tensor(ids[:-1])
                item[f"y_{kind}"] = torch.tensor(ids[1:])
                item[f"mask_{kind}"] = torch.tensor(mask[1:])
            assert "<think>" not in tokenizer.decode(item["x_chosen"])
            items.append(item)
        return items

    @torch.no_grad()
    def sequence_log_probs(model, items):
        """고정된 평가 입력의 log π(chosen|x), log π(rejected|x)와 답변 토큰 수 (평가는 fp32)"""
        out = {}
        for kind in ("chosen", "rejected"):
            x = torch.stack([e[f"x_{kind}"] for e in items]).to(device)
            y = torch.stack([e[f"y_{kind}"] for e in items]).to(device)
            mask = torch.stack([e[f"mask_{kind}"] for e in items]).to(device)
            out[kind] = (logits_to_log_probs(model(x).logits, y) * mask).sum(dim=1).cpu()
            out[f"{kind}_tokens"] = mask.sum(dim=1).cpu()
        return out

    # 첫 쌍의 chosen을 직접 계산해 원본과 비교
    with torch.no_grad():
        _x, _y, _m = (pair[k][None].to(device) for k in ("x_chosen", "y_chosen", "mask_chosen"))
        _logits = sft_model(_x).logits
        first_logp_orig = (logits_to_log_probs(_logits, _y) * _m).sum().item()
        _probs = F.softmax(_logits[0], dim=-1)
        first_logp_manual = sum(torch.log(_probs[t, _y[0, t]]).item() for t in range(_y.shape[1]) if _m[0, t] == 1)
    assert abs(first_logp_orig - first_logp_manual) < 1e-2
    return (
        first_logp_manual,
        first_logp_orig,
        fixed_eval_items,
        sequence_log_probs,
    )


@app.cell
def _(fixed_eval_items, sequence_log_probs, sft_model, test_ds, train_ds):
    train_eval_items = fixed_eval_items(train_ds)
    test_eval_items = fixed_eval_items(test_ds)
    ref_train_lp = sequence_log_probs(sft_model, train_eval_items)
    ref_test_lp = sequence_log_probs(sft_model, test_eval_items)
    return ref_test_lp, ref_train_lp, test_eval_items


@app.cell(hide_code=True)
def _(
    first_logp_manual,
    first_logp_orig,
    mo,
    ref_train_lp,
    result_table,
    train_qa,
):
    _rows = []
    for _i, (_q, _) in enumerate(train_qa[:8]):
        _c, _r = ref_train_lp["chosen"][_i].item(), ref_train_lp["rejected"][_i].item()
        _nc, _nr = int(ref_train_lp["chosen_tokens"][_i]), int(ref_train_lp["rejected_tokens"][_i])
        _rows.append({"질문": _q, "log π(chosen)": f"{_c:.1f} ({_nc}토큰)", "log π(rejected)": f"{_r:.1f} ({_nr}토큰)",
                      "토큰당 평균 chosen / rejected": f"{_c / _nc:.2f} / {_r / _nr:.2f}"})
    mo.vstack([
        mo.md(f"첫 쌍 chosen: 원본 `{first_logp_orig:.3f}` = 직접 계산 `{first_logp_manual:.3f}`"),
        result_table(_rows, selection=None),
        mo.md(r"""
    rejected는 **모델 자신의 greedy 답**이라 토큰당 확률이 높습니다(평균 log p가 0에 가까움).
    반면 합으로 보면 rejected가 길어서 더 음수인 경우가 많습니다. DPO는 이 값을 **reference와의 차이**로 비교하므로
    길이 자체가 손실을 정하지는 않지만, 긴 답일수록 바뀔 수 있는 토큰이 많아 로그 확률이 크게 움직입니다.
    """),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. `dpo_loss` 뜯어보기

    ```python
    def dpo_loss(ref_log_probs, policy_log_probs, mask, beta):
        ref_log_probs = (ref_log_probs * mask).sum(dim=1)          # 시퀀스마다 하나의 값
        policy_log_probs = (policy_log_probs * mask).sum(dim=1)
        # 배치 앞 절반은 chosen, 뒤 절반은 rejected (train_epoch가 torch.cat으로 붙여 넣음)
        chosen_ref, reject_ref = ref_log_probs[:B // 2], ref_log_probs[B // 2:]
        chosen_pol, reject_pol = policy_log_probs[:B // 2], policy_log_probs[B // 2:]
        pi_logratios = chosen_pol - reject_pol                     # policy가 chosen을 얼마나 더 선호하나
        ref_logratios = chosen_ref - reject_ref                    # reference가 chosen을 얼마나 더 선호하나
        logits = pi_logratios - ref_logratios                      # 선호가 기준보다 얼마나 "늘었나"
        return (-F.logsigmoid(beta * logits)).mean()
    ```

    - `logits`(아래에서 **margin**) = $m=(\log\pi_c-\log\pi^{\mathrm{ref}}_c)-(\log\pi_r-\log\pi^{\mathrm{ref}}_r)$.
      $\beta(\log\pi-\log\pi_{\mathrm{ref}})$를 **암묵적 보상(implicit reward)** 이라 부르며, 위에서 정의한 margin $m$에는 β가 없으므로 **암묵적 보상 차이는 $\beta m$**입니다.
    - 학습 시작 때 policy와 reference가 같으므로 margin = 0, 손실 = $-\log\sigma(0)=\ln2\approx0.693$입니다.
    - margin이 커질수록 손실이 0에 가까워지고 **기울기가 줄어듭니다.** 기울기 크기는 $\beta\sigma(-\beta m)$에 비례합니다.
      이미 잘 구분하는 쌍의 갱신은 작아집니다. β는 기울기의 배율과 감소 속도 모두에 영향을 주며, 유한한 margin에서 기울기가 정확히 0이 되는 것은 아닙니다.


    한 쌍의 손실 $\mathcal{L}=-\log\sigma(\beta m)$에 대해
    $\partial\mathcal{L}/\partial m=-\beta\sigma(-\beta m)$입니다. 양의 유한한 $\beta$와 유한한 $m$에서는 0이 아닙니다.
    """)
    return


@app.cell
def _(F, dpo_loss, math, torch):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        _ref = torch.randn(4, 6)     # 배치 4 = chosen 2 + rejected 2, 토큰 6개
        _pol = _ref + 0.3 * torch.randn(4, 6)
        _mask = torch.ones(4, 6)
    _beta = 0.15

    # 직접 계산
    _r, _p = (_ref * _mask).sum(1), (_pol * _mask).sum(1)
    _margin = (_p[:2] - _p[2:]) - (_r[:2] - _r[2:])
    loss_manual = (-F.logsigmoid(_beta * _margin)).mean().item()
    assert abs(dpo_loss(_ref, _pol, _mask, _beta).item() - loss_manual) < 1e-6

    # policy == reference이면 margin 0 → ln 2
    loss_at_start = dpo_loss(_ref, _ref, _mask, _beta).item()
    assert abs(loss_at_start - math.log(2)) < 1e-6
    return (loss_at_start,)


@app.cell(hide_code=True)
def _(loss_at_start, mo, plt, torch):
    _u = torch.linspace(-20, 40, 400)
    _fig, (_a1, _a2) = plt.subplots(1, 2, figsize=(10, 3))
    for _b in [0.05, 0.15, 0.5, 2.0]:
        _a1.plot(_u, -torch.nn.functional.logsigmoid(_b * _u), label=f"beta={_b}")
        _a2.plot(_u, _b * torch.sigmoid(-_b * _u), label=f"beta={_b}")
    _a1.set_xlabel("margin")
    _a1.set_title("DPO loss = -log sigmoid(beta * margin)")
    _a2.set_xlabel("margin")
    _a2.set_title("gradient size = beta * sigmoid(-beta * margin)")
    for _a in (_a1, _a2):
        _a.axvline(0, color="gray", lw=0.5)
        _a.legend(fontsize=8)
    _fig.tight_layout()
    _plot = mo.as_html(_fig)
    plt.close(_fig)
    mo.vstack([
        _plot,
        mo.md(f"원본 `dpo_loss`가 직접 계산과 같고, policy = reference일 때 `{loss_at_start:.4f}` = ln 2 입니다. "
              "오른쪽 그림: β가 크면 margin이 조금만 커져도 기울기가 0에 가까워집니다."),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. DPO 학습

    ### 6-1. `train_epoch` 옮기기

    `train_dpo.py`의 핵심 부분입니다.

    ```python
    model, tokenizer = init_model(lm_config, args.from_weight)        # policy: 학습함
    ref_model, _ = init_model(lm_config, args.from_weight)            # reference: 같은 가중치
    ref_model.eval(); ref_model.requires_grad_(False)                 # 고정

    x = torch.cat([x_chosen, x_rejected], dim=0)                      # 한 배치로 붙여 한 번에 forward
    with torch.no_grad():
        ref_log_probs = logits_to_log_probs(ref_model(x).logits, y)
    policy_log_probs = logits_to_log_probs(model(x).logits, y)
    loss = dpo_loss(ref_log_probs, policy_log_probs, mask, beta) + outputs.aux_loss
    ```

    - reference는 학습 내내 **SFT 모델 그대로**입니다. 같은 쌍에 대해 `ref_logratios`는 항상 같은 상수입니다.
    - 64M 모델 두 개(policy, reference)와 AdamW 상태가 메모리에 올라갑니다(약 1.1GB). reference는 기울기를 계산하지 않습니다.
    - GPU에서는 원본처럼 forward와 손실 계산을 **bf16 autocast** 안에서 합니다. bf16은 float32와 표현 범위가 같아
      `GradScaler`가 필요 없습니다(원본도 `--dtype float16`일 때만 켭니다). CPU에서는 fp32 그대로입니다.
    - 아래 `run_dpo`는 이 루프를 그대로 옮기고, 그래프용 기록만 추가했습니다(저장·재개 제외).
    """)
    return


@app.cell
def _(
    DataLoader,
    SkipBatchSampler,
    autocast_ctx,
    copy,
    device,
    dpo_loss,
    get_lr,
    logits_to_log_probs,
    setup_seed,
    sft_model,
    torch,
    train_ds,
):
    def run_dpo(args, history=None):
        model = copy.deepcopy(sft_model).train()                                   # policy
        ref_model = copy.deepcopy(sft_model).eval().requires_grad_(False)          # reference
        optimizer = torch.optim.AdamW(model.parameters(), lr=args["learning_rate"])
        for epoch in range(args["epochs"]):
            setup_seed(42 + epoch); indices = torch.randperm(len(train_ds)).tolist()
            loader = DataLoader(train_ds, batch_sampler=SkipBatchSampler(indices, args["batch_size"], 0))
            iters = len(loader)
            for step, batch in enumerate(loader, start=1):
                x = torch.cat([batch["x_chosen"], batch["x_rejected"]], dim=0).to(device)
                y = torch.cat([batch["y_chosen"], batch["y_rejected"]], dim=0).to(device)
                mask = torch.cat([batch["mask_chosen"], batch["mask_rejected"]], dim=0).to(device)

                lr = get_lr(epoch * iters + step, args["epochs"] * iters, args["learning_rate"])
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr

                with autocast_ctx():
                    with torch.no_grad():
                        ref_log_probs = logits_to_log_probs(ref_model(x).logits, y)
                    outputs = model(x)
                    policy_log_probs = logits_to_log_probs(outputs.logits, y)
                    dpo_loss_val = dpo_loss(ref_log_probs, policy_log_probs, mask, beta=args["beta"])
                    loss = dpo_loss_val + outputs.aux_loss

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args["grad_clip"])
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                if history is not None:  # 노트북 추가: 암묵적 보상 기록
                    with torch.no_grad():
                        _diff = ((policy_log_probs - ref_log_probs) * mask).sum(1)   # log π − log ref
                        _h = _diff.shape[0] // 2
                        history.append({
                            "loss": dpo_loss_val.item(),
                            "chosen": _diff[:_h].mean().item(),        # log π_c − log ref_c
                            "rejected": _diff[_h:].mean().item(),      # log π_r − log ref_r
                            "accuracy": (_diff[:_h] > _diff[_h:]).float().mean().item(),
                        })
        return model.eval()

    return (run_dpo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 6-2. 실행해 보기

    원본 기본값(`4e-8`, 1 epoch)은 3천 쌍 이상의 데이터를 전제로 한 값이라, 쌍 24개로는 거의 아무것도 바뀌지 않습니다.
    그래서 학습률을 키웠습니다. 기본값 `5e-6`, 3 epochs(쌍 24개 / batch 4 → 18 step)입니다.
    설정을 바꾸면 SFT 모델에서 다시 학습합니다(GPU에서는 수 초, CPU에서는 약 40초).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    dpo_beta = mo.ui.dropdown(options={"0.02": 0.02, "0.15": 0.15, "1.0": 1.0}, value="0.15", label="beta")
    dpo_lr = mo.ui.dropdown(options={"1e-6": 1e-6, "5e-6": 5e-6, "2e-5": 2e-5}, value="5e-6", label="learning_rate")
    dpo_epochs = mo.ui.dropdown(options={"1": 1, "2": 2, "3": 3}, value="3", label="epochs")
    mo.hstack([dpo_beta, dpo_lr, dpo_epochs], justify="start", gap=2)
    return dpo_beta, dpo_epochs, dpo_lr


@app.cell
def _(dpo_beta, dpo_epochs, dpo_lr, run_dpo):
    dpo_args = {"beta": dpo_beta.value, "learning_rate": dpo_lr.value, "epochs": dpo_epochs.value,
                "batch_size": 4, "grad_clip": 1.0}
    dpo_history = []
    dpo_model = run_dpo(dpo_args, history=dpo_history)
    return dpo_args, dpo_history, dpo_model


@app.cell
def _(
    dpo_model,
    generate_answer,
    ref_test_lp,
    sequence_log_probs,
    sft_model,
    test_eval_items,
    test_qa,
    torch,
):
    dpo_test_lp = sequence_log_probs(dpo_model, test_eval_items)  # reference와 같은 고정 입력

    def preference_rate(lp):
        """평가 쌍에서 σ(log π(chosen) − log π(rejected))의 평균: chosen 쪽을 고를 확률"""
        return torch.sigmoid(lp["chosen"] - lp["rejected"]).mean().item()

    sft_test_gen = [generate_answer(sft_model, q) for q, _ in test_qa]
    dpo_test_gen = [generate_answer(dpo_model, q) for q, _ in test_qa]

    def summarize(gens):
        return {"평균 길이(토큰)": sum(g[1] for g in gens) / len(gens), "eos로 멈춤": sum(g[2] for g in gens)}

    test_summary = {
        "SFT (DPO 전)": {"선호 판별률": preference_rate(ref_test_lp), **summarize(sft_test_gen)},
        "DPO 후": {"선호 판별률": preference_rate(dpo_test_lp), **summarize(dpo_test_gen),
                   "Δ log π(chosen)": (dpo_test_lp["chosen"] - ref_test_lp["chosen"]).mean().item(),
                   "Δ log π(rejected)": (dpo_test_lp["rejected"] - ref_test_lp["rejected"]).mean().item(),
                   "chosen이 떨어진 쌍": int((dpo_test_lp["chosen"] < ref_test_lp["chosen"]).sum())},
    }
    return dpo_test_gen, sft_test_gen, test_summary


@app.cell(hide_code=True)
def _(
    dpo_args,
    dpo_history,
    dpo_test_gen,
    mo,
    plt,
    result_table,
    sft_test_gen,
    test_qa,
    test_summary,
):
    _fig, (_a1, _a2) = plt.subplots(1, 2, figsize=(10, 3))
    _a1.plot([h["loss"] for h in dpo_history], label="dpo_loss")
    _a1.plot([h["accuracy"] for h in dpo_history], label="reward accuracy")
    _a1.axhline(0.693, ls=":", color="gray")
    _a1.set_xlabel("step")
    _a1.legend(fontsize=8)
    _a2.plot([h["chosen"] for h in dpo_history], label="chosen: log pi - log ref")
    _a2.plot([h["rejected"] for h in dpo_history], label="rejected: log pi - log ref")
    _a2.axhline(0, color="gray", lw=0.5)
    _a2.set_xlabel("step (training batch)")
    _a2.legend(fontsize=8)
    _fig.tight_layout()
    _plot = mo.as_html(_fig)
    plt.close(_fig)

    _s, _d = test_summary["SFT (DPO 전)"], test_summary["DPO 후"]
    mo.vstack([
        mo.md(f"설정: β={dpo_args['beta']}, lr={dpo_args['learning_rate']}, {dpo_args['epochs']} epochs"),
        _plot,
        mo.md(f"""
    **학습에 없던 질문 8개**에서 측정

    | | 선호 판별률 | 평균 답 길이 | eos로 멈춤 |
    | --- | --- | --- | --- |
    | SFT (DPO 전) | {_s['선호 판별률']:.2f} | {_s['평균 길이(토큰)']:.1f} 토큰 | {_s['eos로 멈춤']}/8 |
    | DPO 후 | {_d['선호 판별률']:.2f} | {_d['평균 길이(토큰)']:.1f} 토큰 | {_d['eos로 멈춤']}/8 |

    reference 대비 평균 변화: `Δ log π(chosen)` = {_d['Δ log π(chosen)']:+.2f}, `Δ log π(rejected)` = {_d['Δ log π(rejected)']:+.2f}.
    chosen의 로그 확률이 오히려 떨어진 쌍: **{_d['chosen이 떨어진 쌍']}/8**
    (선호 판별률 = $\\sigma(\\log\\pi_c-\\log\\pi_r)$의 평균. 두 모델 모두 빈 think를 지운 같은 고정 입력으로 계산. 최대 60토큰까지 생성)
    """),
        result_table(
            [{"질문": q, "기대하는 답": a, "SFT 답": s[0], "DPO 후 답": d[0]}
             for (q, a), s, d in zip(test_qa, sft_test_gen, dpo_test_gen)],
            selection=None,
        ),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **읽는 법** (기본 설정 β=0.15, lr=5e-6, 3 epochs. 설정을 바꾸면 결과도 바뀝니다.)

    - **손실**은 ln 2(점선)에서 시작해 내려가고, **reward accuracy**(배치에서 chosen의 보상이 rejected보다 큰 비율)는 1에 가까워집니다.
    - **학습에 없던 질문**에서도 답이 짧아지고 eos에서 멈추는 쪽으로 바뀝니다. "짧게 답하고 멈추기"는 질문 내용과 상관없는
      **말투(스타일)** 라서 새 질문에도 옮겨 갑니다.
    - **변화의 대부분은 rejected 쪽입니다.** 평균적으로 chosen은 조금 오르고 rejected는 크게 떨어집니다.
      그리고 일부 쌍에서는 **chosen의 확률도 함께 떨어집니다.** rejected가 더 많이 떨어지면 선호 판별률은 오르기 때문입니다.
      DPO는 두 답의 **차이**만 키우므로, "좋은 답의 확률을 올렸다"고 단정하면 안 됩니다.
    - **평가 입력은 고정해야 합니다.** `DPODataset`은 꺼낼 때마다 빈 think 블록을 무작위로 남기므로, 평가할 때 매번 데이터셋에서
      새로 꺼내면 reference와 DPO 모델이 서로 다른 형식을 보게 되어 `Δ log π`가 크게 틀어집니다(`fixed_eval_items`).
    - 하지만 **내용이 맞아지지는 않습니다.** 모델이 모르는 사실(예: 개의 다리 수)은 짧게 틀릴 뿐입니다.
      일부 답이 맞아지기도 하지만, 같은 질문이 학습에 없으므로 짧게 답하다 보니 생긴 변화로 보는 편이 안전합니다.
      이 실험은 **응답 선호 조정**에 초점을 둡니다. 선호 답변의 사실도 학습할 수 있지만,
      DPO 손실 감소가 사실 정확성이나 새로운 질문의 정답률 개선을 보장하지는 않습니다.
    - **학습률 `2e-5`로 올리면:** 손실과 선호 판별률은 더 좋아지지만, 답이 질문과 무관한 짧은 문장으로 무너집니다(과최적화).
      rejected의 로그 확률이 크게 떨어지면서 모델이 원래 능력까지 잃은 것입니다. **DPO 지표만 보고 판단하면 안 되는 이유**입니다.
    - **학습률 `1e-6`:** 이 정도 데이터·step으로는 거의 변하지 않습니다. 원본의 `4e-8`은 이보다도 훨씬 작습니다.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7. β와 reference의 역할

    손실 식만 보면 reference 항은 학습 중 바뀌지 않는 상수라서 쓸모없어 보입니다. reference는 각 응답 쌍의 로그 확률비를 비교하는 **고정 기준점**입니다.

    - 손실은 reference 대비 chosen의 상대 로그 확률비가 커질수록 줄어듭니다. $\beta m$이 충분히 크면 기울기가 매우 작아지지만, 유한한 $m$에서 자동으로 학습이 멈추는 것은 아닙니다.
    - β가 크면 같은 양의 margin에서 기울기가 더 빨리 감소하는 경향이 있습니다. 실제 reference와의 거리는 학습률·학습 횟수·데이터에도 달려 있습니다.
      β는 선호 최적화와 reference 기준의 절충에 관여합니다. 응답마다 KL 상한을 직접 보장하는 장치는 아닙니다.
    - 기울기에도 β가 곱해집니다. Adam의 모멘트·epsilon·과거 갱신 상태가 영향을 주므로 그 배율이 항상 상쇄된다고 단정할 수 없습니다.
      그래서 step 수가 적으면 β를 바꿔도 결과가 비슷할 수 있습니다.

    위젯에서 β를 0.02와 1.0으로 바꾸어 `Δ log π(rejected)`와 답을 비교해 보세요. 학습률 `2e-5`처럼 많이 움직이는 설정에서 차이가 더 잘 보입니다.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 8. 실제 규모로 실행하려면

    이 노트북은 작은 선호 데이터로 몇 step만 학습합니다. 실제로는 SFT 가중치(`projects/minimind/out/full_sft_768.pth`)와
    `dpo.jsonl`(53MB, `projects/minimind/README_ko.md` 참고)을 준비해 GPU에서 실행합니다.

    ```bash
    cd projects/minimind/trainer
    python train_dpo.py                         # full_sft_768.pth에서 시작 → ../out/dpo_768.pth
    python train_dpo.py --beta 0.1 --learning_rate 4e-8
    ```

    | 인자 | 기본값 | 이 노트북 |
    | --- | --- | --- |
    | `--from_weight` | `full_sft` | `caspar/MiniMind-Dense-64M-Mixed-SFT-v1.0` |
    | `--learning_rate` | 4e-8 ("망각 방지를 위해 ≤5e-8 권장") | 5e-6 |
    | `--beta` | 0.15 | 0.15 (위젯으로 변경) |
    | `--batch_size` | 4 (쌍 4개 → forward는 8개) | 4 |
    | `--epochs` | 1 | 3 |
    | `--max_seq_len` | 1024 | 96 |
    | `--dtype` | bfloat16 (GPU) | bfloat16 autocast (GPU) / fp32 (CPU) |

    **참고 기록:** 저장소의 `projects/minimind/prompts/dpo_training.md`에는 같은 베이스 모델로 UltraFeedback에서 고른 3,000쌍을
    2 epochs DPO한 기록이 있습니다. 테스트 150쌍에서 DPO 손실은 0.6931 → 0.6457로 줄었지만, 선호 판별률(46.67%)과
    응답 품질에는 뚜렷한 개선이 없었습니다. 다양한 실제 선호 데이터는 이 노트북의 "짧게 답하기"처럼 하나의 뚜렷한 방향이 없어서,
    작은 모델과 작은 학습률로는 변화가 잘 드러나지 않을 수 있습니다.

    ## 9. 정리

    | 항목 | 핵심 |
    | --- | --- |
    | 데이터 | 같은 질문의 chosen·rejected, 데이터셋에서 한 칸 밀어 `x`, `y`, `mask` |
    | 로그 확률 | 답변 토큰 log p의 **합** (`logits_to_log_probs` + mask) |
    | 모델 | policy(학습) + reference(SFT 모델 고정), chosen·rejected를 한 배치로 붙여 forward |
    | 손실 | $-\log\sigma(\beta[(\ell_c-\ell_r)-(\ell^{\mathrm{ref}}_c-\ell^{\mathrm{ref}}_r)])$($\ell$은 응답 로그 확률의 합), 시작값 ln 2 |
    | 효과 | 응답 선호를 조정하며 선호 데이터의 내용도 학습할 수 있음. 사실 정확성 개선은 보장하지 않음 |
    | 주의 | 학습률이 크면 손실은 좋아져도 답이 무너짐. 손실·판별률만 보지 말고 생성 결과를 확인 |

    ## 10. 직접 해 볼 과제

    각 과제를 하기 전에 **결과를 먼저 예측**해 적고, 바꿔 실행한 뒤 비교하세요.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({
        "과제 1. 학습률을 2e-5로 (6-2절)": mo.md("""
    학습률만 `2e-5`로 바꾸고 손실, 선호 판별률, `Δ log π(rejected)`, 답을 기본 설정과 비교하세요.

    *해설:* 손실은 0에 가깝고 선호 판별률도 오르지만, 답은 질문과 상관없는 짧은 문장이 됩니다.
    rejected 로그 확률이 크게 떨어지면서 모델의 원래 능력까지 무너진 과최적화입니다.
    """),
        "과제 2. β 비교 (7절)": mo.md(r"""
    학습률 `2e-5`에서 β를 0.02와 1.0으로 바꿔 보세요. 어느 쪽이 SFT 모델의 답에 더 가깝게 남을까요?

    *해설:* β가 크면 같은 양의 margin에서 기울기가 더 빨리 작아지는 경향이 있습니다. 생성 결과와 로그 확률 변화를 함께 확인하고, 학습률과 step 수를 고정해 비교하세요.
    """),
        "과제 3. chosen과 rejected 바꾸기 (2절)": mo.md("""
    `pair_rows`에서 chosen과 rejected를 서로 바꾸면 어떻게 될까요?

    *해설:* 모델은 장황한 답을 더 선호하도록 바뀝니다. DPO는 무엇이 "좋은지" 모르고 **주어진 선호를 따를 뿐**입니다.
    선호 데이터의 품질과 방향이 결과를 결정합니다.
    """),
        "과제 4. 틀린 chosen 넣기 (2절)": mo.md("""
    `train_qa`의 답 몇 개를 짧지만 틀린 답(예: `"A spider has six legs."`)으로 바꾸면 평가 질문의 답은 어떻게 될까요?

    *해설:* 모델은 "짧게 답하기"를 그대로 배우고, 틀린 사실도 선호 대상으로 학습합니다.
    DPO는 사실 여부를 검사하지 않으므로 선호 데이터의 정확성을 사람이 보장해야 합니다.
    """),
        "과제 5. 빈 think 블록 남기기 (1절)": mo.md("""
    `prompt_ids`에서 `.replace("<think>\\n\\n</think>\\n\\n", "")`를 지우고 SFT 모델의 답을 보세요.

    *해설:* 모델 카드가 경고하듯 SFT 학습 형식과 프롬프트가 달라져 답이 달라질 수 있습니다.
    학습 형식과 추론 형식을 맞추는 것이 중요하다는 점을 확인하는 과제입니다.
    """),
    })
    return


if __name__ == "__main__":
    app.run()
