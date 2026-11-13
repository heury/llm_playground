import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import itertools
    import json
    import math
    import random
    import shutil
    import sys
    import tempfile
    from contextlib import nullcontext
    from pathlib import Path

    import datasets
    import marimo as mo
    import matplotlib.pyplot as plt
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer

    # 노트북 위치(notebooks/minimind/)에서 두 단계 위가 저장소 루트입니다.
    repo_root = Path(__file__).resolve().parents[2]
    minimind_dir = repo_root / "projects/minimind"
    sys.path.insert(0, str(minimind_dir))

    # train_pretrain.py가 쓰는 원본 클래스와 함수를 그대로 가져옵니다.
    from dataset.lm_dataset import PretrainDataset
    from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
    from trainer.trainer_utils import SkipBatchSampler, get_lr, lm_checkpoint, setup_seed

    torch.set_num_threads(2)
    datasets.disable_progress_bars()  # jsonl을 읽을 때의 진행 막대 숨기기

    tokenizer = AutoTokenizer.from_pretrained(minimind_dir / "model")

    # train_pretrain.py의 "3. 혼합 정밀도 설정"과 같습니다 (--device cuda:0, --dtype bfloat16).
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16
    autocast_ctx = nullcontext() if device.type == "cpu" else torch.autocast("cuda", dtype=dtype)
    return (
        DataLoader,
        MiniMindConfig,
        MiniMindForCausalLM,
        Path,
        PretrainDataset,
        SkipBatchSampler,
        autocast_ctx,
        device,
        dtype,
        get_lr,
        itertools,
        json,
        lm_checkpoint,
        math,
        mo,
        plt,
        random,
        setup_seed,
        shutil,
        tempfile,
        tokenizer,
        torch,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # MiniMind 사전 학습(Pretrain) 따라가기

    **사전 학습**은 아무 지시도 없는 평범한 글을 잔뜩 읽히면서 **다음 토큰 맞히기**만 반복해,
    무작위 가중치의 모델을 "말이 되는 글을 이어 쓰는" 모델로 만드는 단계입니다.
    MiniMind에서는 `projects/minimind/trainer/train_pretrain.py`가 이 일을 합니다.

    이 노트북은 그 스크립트를 부품별로 나누어 **원본 함수를 그대로** 실행합니다.
    모델은 스크립트 기본값과 **같은 크기**(은닉 차원 768, 8층, 약 64M 매개변수)이고, GPU에서 bf16 혼합 정밀도로 학습합니다.
    실제 데이터(GB 단위) 대신 같은 형식의 작은 합성 말뭉치를 쓰므로 학습 한 번이 수 초면 끝납니다.
    (GPU 메모리 약 2GB 사용. GPU가 없으면 CPU fp32로 동작하지만 매우 느립니다.)

    **사전 지식:** Python, PyTorch의 `loss.backward()`와 `optimizer.step()`.
    MiniMind 모델이 `[B, T]` 토큰을 받아 `[B, T, V]` logits를 낸다는 정도만 알면 됩니다.

    ## 0. 큰 그림

    ```text
    jsonl 파일 {"text": ...}
        │  PretrainDataset            ① 토큰화, <bos>/<eos>, 패딩, 정답(labels)
        ▼
    (input_ids, labels)  [T]
        │  randperm + SkipBatchSampler + DataLoader   ⑥ 섞기, 배치, 재개 시 건너뛰기
        ▼
    배치 [B, T] ──▶ model(input_ids, labels) ──▶ loss             ② ③
                                         │
                       loss / 누적 수 → backward  (여러 번 쌓기)     ③
                                         │ 누적 수마다
                 기울기 자르기(clip) → AdamW step(학습률 get_lr)     ③ ④
                                         │ save_interval마다
                        가중치 파일 + 재개용 체크포인트 저장          ⑧
    ```

    **`train_pretrain.py` 지도** (원형 숫자는 이 노트북의 절)

    | 스크립트의 단계 | 하는 일 | 절 |
    | --- | --- | --- |
    | 1. 환경·시드 | `setup_seed`, 여러 GPU(DDP) 초기화 | 2, 9 |
    | 2. 모델 설정·체크포인트 확인 | `MiniMindConfig`, `--from_resume`이면 `lm_checkpoint`로 불러오기 | 2, 8 |
    | 3. 혼합 정밀도 | GPU에서 `autocast`(bf16/fp16), CPU는 그대로 fp32 | 5 |
    | 5. 모델·데이터·옵티마이저 | `init_model`, `PretrainDataset`, `GradScaler`, `AdamW` | 1, 2, 3 |
    | 6. 상태 복원 | 모델·옵티마이저·scaler·epoch·step 복원 | 8 |
    | 8. 학습 | epoch마다 섞고 `train_epoch` 실행 | 6, 7 |
    | `train_epoch` | 학습률 → forward → 누적 backward → clip → step → 로그 → 저장 | 3, 4, 7 |
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. 데이터: `PretrainDataset`

    사전 학습 데이터는 한 줄에 글 하나인 jsonl 파일입니다. 실제 MiniMind는 `pretrain_t2t_mini.jsonl`(1.2GB) 같은 파일을 씁니다.

    ```json
    {"text": "Transformer는 자기 어텐션으로 문맥 관계를 학습합니다."}
    ```

    여기서는 **주어 6개 × 동사 3개 × 목적어 4개 = 72개**의 짧은 영어 문장을 만들어 같은 형식으로 저장합니다.
    이 중 12개 조합은 **학습에 쓰지 않고 따로 떼어 두어**(검증용), 모델이 외운 것이 아니라 문장 구조를 배웠는지 확인합니다.
    (원본 스크립트에는 검증 단계가 없습니다. 검증 손실은 이 노트북에서 추가로 재는 값입니다.)
    """)
    return


@app.cell
def _(Path, PretrainDataset, itertools, json, random, tempfile, tokenizer):
    subjects = ["The cat", "The dog", "A bird", "My friend", "The teacher", "A child"]
    verbs = ["eats", "sees", "likes"]
    objects = ["an apple", "the ball", "a book", "the fish"]

    all_sentences = [f"{s} {v} {o}." for s, v, o in itertools.product(subjects, verbs, objects)]
    random.Random(0).shuffle(all_sentences)
    val_sentences, train_sentences = all_sentences[:12], all_sentences[12:]

    def write_jsonl(path, texts):
        with open(path, "w", encoding="utf-8") as f:
            for text in texts:
                f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")

    data_dir = Path(tempfile.mkdtemp(prefix="minimind_pretrain_data_"))
    train_path, val_path = data_dir / "pretrain_train.jsonl", data_dir / "pretrain_val.jsonl"
    write_jsonl(train_path, train_sentences)
    write_jsonl(val_path, val_sentences)

    max_seq_len = 16  # 원본 기본값은 340. 우리 문장은 짧으므로 16이면 충분합니다.
    train_ds = PretrainDataset(str(train_path), tokenizer, max_length=max_seq_len)
    val_ds = PretrainDataset(str(val_path), tokenizer, max_length=max_seq_len)
    return all_sentences, max_seq_len, train_ds, train_path, val_ds, val_sentences, write_jsonl, data_dir


@app.cell(hide_code=True)
def _(mo, train_ds, train_path, val_ds, val_sentences):
    _head = "".join(open(train_path, encoding="utf-8").readlines()[:3])
    mo.md(f"""
    학습 파일의 처음 세 줄 (`{train_path.name}`):

    ```json
    {_head.strip()}
    ```

    학습 문장 **{len(train_ds)}개**, 검증 문장 **{len(val_ds)}개** (예: `{val_sentences[0]}`, `{val_sentences[1]}`)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 1-1. 샘플 하나 뜯어보기

    `PretrainDataset.__getitem__`은 글 하나를 다음처럼 바꿉니다.

    ```python
    tokens = tokenizer(text, add_special_tokens=False, max_length=max_length - 2, truncation=True).input_ids
    tokens = [bos_token_id] + tokens + [eos_token_id]                     # 시작·끝 표시
    input_ids = tokens + [pad_token_id] * (max_length - len(tokens))      # 길이 맞추기
    labels = input_ids.clone(); labels[input_ids == pad_token_id] = -100  # 패딩은 정답에서 제외
    ```

    - `<|im_start|>`(bos)로 시작하고 `<|im_end|>`(eos)로 끝납니다. 모델은 **글이 끝나는 곳에서 eos를 내는 법**도 배웁니다.
    - 모든 샘플이 같은 길이 `max_length`가 되도록 `<|endoftext|>`(pad)를 채웁니다.
    - `labels`는 `input_ids`의 **복사본**입니다. 한 칸 미는 일은 모델 `forward`가 합니다
      (`logits[:, :-1]`과 `labels[:, 1:]`를 비교). 그래서 위치 `t`의 출력이 위치 `t+1`의 토큰을 맞힙니다.
    - 정답이 `-100`인 위치는 `cross_entropy(..., ignore_index=-100)`에서 빠집니다.
    """)
    return


@app.cell
def _(max_seq_len, tokenizer, train_ds):
    sample_input_ids, sample_labels = train_ds[0]
    sample_real_len = int((sample_labels != -100).sum())  # bos + 본문 + eos

    assert sample_input_ids.shape == (max_seq_len,)
    assert sample_input_ids[0] == tokenizer.bos_token_id
    assert sample_input_ids[sample_real_len - 1] == tokenizer.eos_token_id
    assert (sample_input_ids[sample_real_len:] == tokenizer.pad_token_id).all()
    assert (sample_labels[sample_real_len:] == -100).all()
    return sample_input_ids, sample_labels, sample_real_len


@app.cell(hide_code=True)
def _(mo, sample_input_ids, sample_labels, tokenizer):
    def _tok(i):
        return tokenizer.convert_ids_to_tokens(int(i))

    _rows = []
    for _t in range(len(sample_input_ids)):
        _next = int(sample_labels[_t + 1]) if _t + 1 < len(sample_labels) else None
        _rows.append({
            "위치 t": _t,
            "input_ids": int(sample_input_ids[_t]),
            "입력 토큰": _tok(sample_input_ids[_t]),
            "labels": int(sample_labels[_t]),
            "위치 t에서 맞힐 토큰 (labels[t+1])": "—" if _next is None else ("무시(-100)" if _next == -100 else _tok(_next)),
        })
    mo.vstack([
        mo.ui.table(_rows, selection=None, page_size=20),
        mo.md("`Ġ`는 단어 앞 공백입니다. 첫 토큰 bos는 맞힐 대상이 아니고, 마지막 실제 토큰 다음에는 **eos**를 맞혀야 합니다."),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 1-2. 너무 긴 글은 잘립니다

    본문은 `max_length - 2` 토큰까지만 남기고 자릅니다(`truncation=True`). bos와 eos 두 자리를 남겨 두기 때문입니다.
    잘린 글도 **eos로 끝납니다.** 실제로 끝난 글이 아닌데도 모델은 거기서 끝난다고 배우게 되므로,
    실제 데이터는 대부분 `max_seq_len`보다 짧게 준비합니다.
    """)
    return


@app.cell
def _(PretrainDataset, data_dir, max_seq_len, tokenizer, write_jsonl):
    long_text = "The cat eats an apple and the dog sees the ball and a bird likes the fish every day."
    write_jsonl(data_dir / "long.jsonl", [long_text])
    long_ids, _ = PretrainDataset(str(data_dir / "long.jsonl"), tokenizer, max_length=max_seq_len)[0]

    long_full_len = len(tokenizer(long_text, add_special_tokens=False).input_ids)
    assert long_ids[-1] == tokenizer.eos_token_id  # 잘려도 eos로 끝남
    long_kept = tokenizer.decode(long_ids[1:-1])
    return long_full_len, long_kept, long_text


@app.cell(hide_code=True)
def _(long_full_len, long_kept, long_text, max_seq_len, mo):
    mo.md(f"""
    원문 ({long_full_len} 토큰): `{long_text}`

    `max_length={max_seq_len}`로 남은 본문 ({max_seq_len - 2} 토큰): `{long_kept}` → 바로 뒤에 eos
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. 모델 준비: `setup_seed`와 첫 손실

    `--from_weight none`(기본값)이면 `init_model`은 **무작위 가중치**의 `MiniMindForCausalLM`을 만듭니다.
    `setup_seed`가 Python·NumPy·PyTorch 난수를 모두 고정하므로, 같은 시드면 같은 초기 가중치가 나옵니다.

    모델 설정은 스크립트와 같습니다: `MiniMindConfig(hidden_size=768, num_hidden_layers=8, use_moe=False)`.
    토크나이저도 실제 MiniMind 것(어휘 6,400개)을 씁니다. 모델은 `init_model`처럼 만든 뒤 `model.to(device)`로 GPU에 올립니다.
    학습 전 모델은 6,400개 토큰 중 아무거나 고르는 것과 비슷하므로 손실이 `ln(6400) ≈ 8.76` 근처에서 시작합니다.
    """)
    return


@app.cell
def _(MiniMindConfig, MiniMindForCausalLM, device, math, setup_seed, torch, train_ds):
    # train_pretrain.py 기본 인자: --hidden_size 768 --num_hidden_layers 8 --use_moe 0
    lm_config = MiniMindConfig(hidden_size=768, num_hidden_layers=8, use_moe=False)

    def init_pretrain_model(seed=42):
        """train_pretrain.py의 setup_seed(args.seed) → init_model(..., from_weight='none', device=args.device)"""
        setup_seed(seed)
        return MiniMindForCausalLM(lm_config).to(device)

    def stack_samples(ds, indices):
        ids, labels = zip(*(ds[i] for i in indices))
        return torch.stack(ids).to(device), torch.stack(labels).to(device)

    initial_model = init_pretrain_model().eval()
    first_ids, first_labels = stack_samples(train_ds, range(8))
    with torch.no_grad():
        initial_loss = initial_model(first_ids, labels=first_labels).loss.item()
    uniform_loss = math.log(lm_config.vocab_size)
    model_params = sum(p.numel() for p in initial_model.parameters())
    return init_pretrain_model, initial_loss, initial_model, lm_config, model_params, stack_samples, uniform_loss


@app.cell(hide_code=True)
def _(device, initial_loss, mo, model_params, torch, uniform_loss):
    _gpu = f" ({torch.cuda.get_device_name(device)})" if device.type == "cuda" else ""
    mo.md(f"""
    모델 매개변수 **{model_params / 1e6:.1f}M**, 장치 `{device}`{_gpu}

    학습 전 손실 `{initial_loss:.3f}` ≈ 균등 분포 손실 `ln(6400) = {uniform_loss:.3f}`
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 2-1. `attention_mask` 없이 패딩을 넣어도 될까?

    `train_epoch`는 `model(input_ids, labels=labels)`만 부르고 `attention_mask`를 주지 않습니다. 그래도 되는 이유:

    1. 패딩은 **항상 뒤쪽**에 있습니다(오른쪽 패딩).
    2. MiniMind Attention은 **causal**이라 각 위치는 자기보다 앞만 봅니다. 그래서 실제 토큰은 뒤의 패딩을 볼 수 없습니다.
    3. 패딩 위치의 출력은 정답이 `-100`이라 손실에서 빠집니다.

    즉 패딩은 계산만 낭비할 뿐 결과에는 영향을 주지 않습니다. 패딩을 붙인 입력과 잘라낸 입력으로 확인합니다.
    """)
    return


@app.cell
def _(device, initial_model, sample_input_ids, sample_real_len, torch, train_ds):
    _ids = sample_input_ids[None].to(device)
    with torch.no_grad():
        logits_with_pad = initial_model(_ids).logits[0, :sample_real_len]
        logits_no_pad = initial_model(_ids[:, :sample_real_len]).logits[0]
    pad_effect = (logits_with_pad - logits_no_pad).abs().max().item()
    assert pad_effect < 1e-4

    _all_labels = [train_ds[i][1] for i in range(len(train_ds))]
    pad_ratio = sum(int((l == -100).sum()) for l in _all_labels) / sum(l.numel() for l in _all_labels)
    return pad_effect, pad_ratio


@app.cell(hide_code=True)
def _(mo, pad_effect, pad_ratio):
    mo.md(f"""
    실제 토큰 위치의 logits 차이: `{pad_effect:.1e}` (사실상 0). 패딩은 결과를 바꾸지 않습니다.

    다만 이 데이터에서는 전체 칸의 **{pad_ratio:.0%}** 가 패딩이라 그만큼 계산이 낭비됩니다.
    실제 학습에서 `max_seq_len`을 데이터 길이에 맞게 고르는 이유입니다.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. 학습 한 step 해부하기

    `train_epoch`의 핵심 부분입니다.

    ```python
    res = model(input_ids, labels=labels)
    loss = res.loss + res.aux_loss          # aux_loss: MoE 균형 손실 (MoE가 아니면 0)
    loss = loss / args.accumulation_steps
    scaler.scale(loss).backward()           # 기울기를 계산해 p.grad에 "더함"

    if step % args.accumulation_steps == 0 or step == iters:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)              # AdamW로 가중치 갱신
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
    ```

    ### 3-1. forward → backward → clip → AdamW step

    배치 하나로 한 번 갱신해 보고, 무엇이 바뀌는지 봅니다. 학습률은 원본 기본값 `5e-4`입니다.
    `scaler`는 fp16일 때만 하는 일이 있으므로(5절) 여기서는 생략합니다.
    3절은 값을 정확히 비교하려고 `autocast` 없이 fp32로 계산합니다. 7절의 실제 학습은 bf16 `autocast`를 씁니다.
    """)
    return


@app.cell
def _(first_ids, first_labels, init_pretrain_model, torch):
    step_model = init_pretrain_model().train()
    step_optimizer = torch.optim.AdamW(step_model.parameters(), lr=5e-4)

    step_res = step_model(first_ids, labels=first_labels)
    step_loss = step_res.loss + step_res.aux_loss
    step_loss.backward()

    # clip_grad_norm_은 자르기 "전"의 전체 기울기 크기를 돌려줍니다.
    grad_norm_before = torch.nn.utils.clip_grad_norm_(step_model.parameters(), max_norm=1.0).item()
    grad_norm_after = torch.norm(torch.stack([p.grad.norm() for p in step_model.parameters()])).item()
    assert grad_norm_after <= 1.0 + 1e-4

    step_optimizer.step()
    step_optimizer.zero_grad(set_to_none=True)
    with torch.no_grad():
        loss_after_one_step = step_model(first_ids, labels=first_labels).loss.item()

    # AdamW는 매개변수마다 이동 평균 두 개(exp_avg, exp_avg_sq)를 저장합니다.
    _first_param = next(step_model.parameters())
    adam_state_keys = list(step_optimizer.state[_first_param].keys())
    adam_state_numel = sum(
        v.numel() for s in step_optimizer.state.values() for v in s.values() if torch.is_tensor(v) and v.dim() > 0
    )
    model_numel = sum(p.numel() for p in step_model.parameters())
    return adam_state_keys, adam_state_numel, grad_norm_after, grad_norm_before, loss_after_one_step, model_numel, step_loss


@app.cell(hide_code=True)
def _(adam_state_keys, adam_state_numel, grad_norm_after, grad_norm_before, loss_after_one_step, mo, model_numel, step_loss):
    mo.md(f"""
    | 항목 | 값 |
    | --- | --- |
    | 갱신 전 손실 | `{step_loss.item():.4f}` |
    | 한 번 갱신 후 같은 배치 손실 | `{loss_after_one_step:.4f}` |
    | 기울기 전체 크기 (clip 전 → 후, `grad_clip=1.0`) | `{grad_norm_before:.3f}` → `{grad_norm_after:.3f}` |
    | AdamW 상태 | `{adam_state_keys}` |
    | AdamW 상태 원소 수 / 모델 매개변수 수 | `{adam_state_numel:,}` / `{model_numel:,}` (= {adam_state_numel / model_numel:.0f}배) |

    - **clip:** 기울기 전체 크기가 `grad_clip`보다 크면 방향은 두고 크기만 줄입니다. 가끔 튀는 큰 기울기가 학습을 망치지 않게 합니다.
    - **AdamW 상태:** 매개변수마다 기울기의 이동 평균(`exp_avg`)과 제곱의 이동 평균(`exp_avg_sq`)을 저장합니다.
      그래서 옵티마이저가 **모델의 2배 메모리**를 쓰고, 재개용 체크포인트가 가중치 파일보다 훨씬 큽니다(8절).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 3-2. 기울기 누적(gradient accumulation)

    GPU 메모리에 큰 배치가 들어가지 않을 때, 작은 배치(micro-batch) 여러 개의 기울기를 **더해서** 한 번에 갱신합니다.
    `backward()`는 `p.grad`를 덮어쓰지 않고 **더하기** 때문에 가능합니다.
    기본값은 `batch_size=32`, `accumulation_steps=8`이므로 실제로는 **256개 문장마다 한 번** 갱신합니다.

    - 각 micro-batch 손실을 `accumulation_steps`로 나누어 더하면 **micro-batch 손실들의 평균**에 대한 기울기가 됩니다.
    - 큰 배치 하나의 손실은 **모든 정답 토큰의 평균**입니다. 두 값은 micro-batch마다 정답 토큰 수가 같을 때만 정확히 같습니다.

    샘플 4개를 큰 배치 1개(4개)로 계산한 기울기와, micro-batch 2개(2개씩) 누적한 기울기를 비교합니다.
    """)
    return


@app.cell
def _(init_pretrain_model, stack_samples, torch, train_ds):
    def flat_grad(model):
        return torch.cat([p.grad.flatten() for p in model.parameters()])

    def big_batch_grad(indices):
        model = init_pretrain_model().train()
        ids, labels = stack_samples(train_ds, indices)
        model(ids, labels=labels).loss.backward()
        return flat_grad(model)

    def accumulated_grad(indices, accumulation_steps=2):
        model = init_pretrain_model().train()
        size = len(indices) // accumulation_steps
        for k in range(accumulation_steps):
            ids, labels = stack_samples(train_ds, indices[k * size:(k + 1) * size])
            (model(ids, labels=labels).loss / accumulation_steps).backward()  # 더해짐
        return flat_grad(model)

    # 정답 토큰 수별로 샘플을 묶습니다.
    _by_len = {}
    for _i in range(len(train_ds)):
        _by_len.setdefault(int((train_ds[_i][1] != -100).sum()), []).append(_i)
    _sizes = sorted(_by_len)
    same_len_idx = next(v for v in _by_len.values() if len(v) >= 4)[:4]      # 모두 같은 길이
    mixed_len_idx = _by_len[_sizes[0]][:2] + _by_len[_sizes[-1]][:2]       # 짧은 2개 + 긴 2개

    def rel_diff(a, b):
        return ((a - b).norm() / b.norm()).item()

    accum_same_diff = rel_diff(accumulated_grad(same_len_idx), big_batch_grad(same_len_idx))
    accum_mixed_diff = rel_diff(accumulated_grad(mixed_len_idx), big_batch_grad(mixed_len_idx))
    assert accum_same_diff < 1e-5
    return accum_mixed_diff, accum_same_diff, mixed_len_idx, same_len_idx


@app.cell(hide_code=True)
def _(accum_mixed_diff, accum_same_diff, mixed_len_idx, mo, same_len_idx, train_ds):
    def _n(idx):
        return [int((train_ds[i][1] != -100).sum()) for i in idx]

    mo.md(f"""
    | 샘플 4개 | 샘플별 토큰 수 | 누적 기울기 vs 큰 배치 기울기 (상대 차이) |
    | --- | --- | --- |
    | 길이가 모두 같음 | `{_n(same_len_idx)}` | `{accum_same_diff:.1e}` (같음) |
    | 짧은 2개 + 긴 2개 | `{_n(mixed_len_idx)}` | `{accum_mixed_diff:.1e}` (조금 다름) |

    길이가 다르면 짧은 micro-batch의 토큰 하나가 조금 더 큰 비중을 갖습니다. 실제 학습에서는 이 정도 차이를 감수하고 씁니다.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. 학습률 스케줄: `get_lr`

    ```python
    def get_lr(current_step, total_steps, lr):
        return lr * (0.1 + 0.45 * (1 + math.cos(math.pi * current_step / total_steps)))
    ```

    - `current_step = 0`이면 `0.1 + 0.45·2 = 1.0`배, 끝(`current_step = total_steps`)이면 `0.1 + 0.45·0 = 0.1`배입니다.
    - 즉 **설정한 학습률에서 시작해 코사인 곡선을 따라 그 10%까지** 줄어듭니다. 처음엔 크게 움직이고 끝에는 세밀하게 다듬습니다.
    - MiniMind는 처음에 학습률을 서서히 올리는 warmup을 쓰지 않습니다.
    - `train_epoch`는 **micro-batch마다** `get_lr(epoch * iters + step, epochs * iters, lr)`로 학습률을 다시 정합니다.
      `step`은 micro-batch 번호이므로 누적을 쓰면 실제 갱신 때 쓰이는 값은 누적 마지막 micro-batch의 학습률입니다.
    """)
    return


@app.cell
def _(get_lr):
    lr_demo_base = 5e-4  # train_pretrain.py 기본 학습률
    lr_demo_total = 1000
    lr_demo_curve = [get_lr(s, lr_demo_total, lr_demo_base) for s in range(lr_demo_total + 1)]
    assert abs(lr_demo_curve[0] - lr_demo_base) < 1e-12
    assert abs(lr_demo_curve[-1] - 0.1 * lr_demo_base) < 1e-12
    return lr_demo_base, lr_demo_curve


@app.cell(hide_code=True)
def _(lr_demo_base, lr_demo_curve, mo, plt):
    _fig, _ax = plt.subplots(figsize=(5.5, 2.8))
    _ax.plot(lr_demo_curve)
    _ax.axhline(lr_demo_base, ls=":", color="gray")
    _ax.axhline(0.1 * lr_demo_base, ls=":", color="gray")
    _ax.set_xlabel("step (of total 1000)")
    _ax.set_ylabel("learning rate")
    _ax.set_title("get_lr(step, 1000, 5e-4)")
    _fig.tight_layout()
    _plot = mo.as_html(_fig)
    plt.close(_fig)
    _plot
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. 혼합 정밀도: `autocast`와 `GradScaler`

    GPU에서는 계산을 16비트로 하면 빠르고 메모리도 적게 씁니다. `train_pretrain.py`의 설정:

    ```python
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))
    ```

    - `autocast` 안에서는 행렬 곱 같은 연산을 16비트로 하고, 가중치 자체는 32비트로 유지합니다. **CPU에서는 아무것도 하지 않습니다.**
    - 기본값 `bfloat16`은 float32와 표현 범위가 같아 작은 기울기도 0이 되지 않습니다. 대신 유효 숫자가 적습니다.
    - `float16`은 범위가 좁아 아주 작은 기울기가 **0으로 사라집니다(underflow).** 그래서 `GradScaler`가 손실에 큰 수(기본 2¹⁶)를
      곱해 backward하고, 갱신 직전 `unscale_`로 다시 나눕니다. clip은 **unscale 뒤에** 해야 실제 크기로 자릅니다.

    작은 기울기 값이 각 형식에서 어떻게 되는지 확인합니다.
    """)
    return


@app.cell
def _(torch):
    small_grads = torch.tensor([1e-3, 1e-5, 1e-7, 1e-9])
    grad_scale = 2.0 ** 16  # GradScaler의 기본 시작 배율

    as_fp16 = small_grads.half().float()
    as_fp16_scaled = (small_grads * grad_scale).half().float() / grad_scale  # 곱해서 16비트 → 다시 나눔
    as_bf16 = small_grads.bfloat16().float()

    assert as_fp16[-1] == 0 and as_fp16_scaled[-1] > 0
    return as_bf16, as_fp16, as_fp16_scaled, small_grads


@app.cell(hide_code=True)
def _(as_bf16, as_fp16, as_fp16_scaled, mo, small_grads):
    mo.ui.table(
        [
            {"원래 값 (fp32)": f"{a:.1e}", "fp16": f"{b:.3e}", "fp16 + 배율 2¹⁶": f"{c:.3e}", "bf16": f"{d:.3e}"}
            for a, b, c, d in zip(small_grads.tolist(), as_fp16.tolist(), as_fp16_scaled.tolist(), as_bf16.tolist())
        ],
        selection=None,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    fp16에서는 `1e-9`가 0이 되고 `1e-7`도 크게 틀어지지만, 배율을 곱하면 살아남습니다. bf16은 작은 값도 표현하지만 유효 숫자가 3자리 정도입니다.
    이 노트북의 학습(7, 8절)은 원본 기본값처럼 GPU에서 **bf16 `autocast`** 로 계산합니다.
    bf16이므로 `GradScaler(enabled=False)`는 아무 일도 하지 않습니다(GPU가 없으면 fp32).

    ## 6. 섞기와 배치: `randperm`과 `SkipBatchSampler`

    ```python
    setup_seed(args.seed + epoch); indices = torch.randperm(len(train_ds)).tolist()
    skip = start_step if (epoch == start_epoch and start_step > 0) else 0
    batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
    ```

    - epoch마다 시드를 `seed + epoch`로 다시 정해 순서를 섞습니다. 그래서 **같은 epoch는 언제 실행해도 같은 순서**입니다.
    - `SkipBatchSampler`는 순서대로 `batch_size`개씩 묶되, 앞의 `skip`개 배치는 건너뜁니다.
      중단된 학습을 재개할 때 이미 학습한 배치를 다시 보지 않기 위한 장치입니다(8절).
    - 여러 GPU(DDP)에서는 `indices` 대신 데이터를 GPU별로 나누는 `DistributedSampler`를 씁니다.
    """)
    return


@app.cell
def _(SkipBatchSampler, setup_seed, torch, train_ds):
    def epoch_batches(epoch, skip=0, seed=42, batch_size=8):
        setup_seed(seed + epoch)
        indices = torch.randperm(len(train_ds)).tolist()
        return list(SkipBatchSampler(indices, batch_size, skip))

    batches_epoch0 = epoch_batches(0)
    batches_epoch1 = epoch_batches(1)
    batches_epoch0_skip3 = epoch_batches(0, skip=3)

    assert batches_epoch0 == epoch_batches(0)          # 같은 epoch → 같은 순서
    assert batches_epoch0 != batches_epoch1             # epoch이 다르면 다른 순서
    assert batches_epoch0_skip3 == batches_epoch0[3:]   # 앞 3개 배치만 건너뜀
    return batches_epoch0, batches_epoch0_skip3, batches_epoch1


@app.cell(hide_code=True)
def _(batches_epoch0, batches_epoch0_skip3, batches_epoch1, mo):
    _n = max(len(batches_epoch0), len(batches_epoch1))
    mo.vstack([
        mo.ui.table(
            [
                {"배치 번호": k + 1,
                 "epoch 0": str(batches_epoch0[k]) if k < len(batches_epoch0) else "",
                 "epoch 1": str(batches_epoch1[k]) if k < len(batches_epoch1) else "",
                 "epoch 0, skip=3": str(batches_epoch0[k]) if k >= 3 else "(건너뜀)"}
                for k in range(_n)
            ],
            selection=None, page_size=20,
        ),
        mo.md(f"샘플 {sum(map(len, batches_epoch0))}개, `batch_size=8` → 배치 {len(batches_epoch0)}개 (마지막 배치는 남은 샘플만). "
              f"skip=3이면 {len(batches_epoch0_skip3)}개 배치만 나옵니다."),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7. 전체 학습 실행

    ### 7-1. `train_epoch`를 함수 하나로 옮기기

    아래 `run_pretraining`은 `train_pretrain.py`의 "8. 학습 시작" 반복문과 `train_epoch`를 **줄 단위로 그대로** 옮긴 것입니다.
    노트북에서 쓰기 위해 바꾼 점만 표시했습니다.

    - 원본의 `args`, `model` 같은 전역 변수 대신 인자로 받습니다.
    - `history`, `logs`, `val_fn`은 그래프와 검증 손실을 위해 **노트북에서 추가**한 기록입니다.
    - `stop_at=(epoch, step)`은 8절에서 "학습이 중간에 끊긴 상황"을 흉내 내기 위해 추가했습니다.
    - DDP, wandb, `torch.compile`은 뺐습니다. 장치 이동(`.to(device)`)과 `autocast`는 원본과 같습니다.
    """)
    return


@app.cell
def _(DataLoader, Path, SkipBatchSampler, autocast_ctx, device, get_lr, lm_checkpoint, lm_config, setup_seed, torch, train_ds):
    def run_pretraining(model, optimizer, scaler, args, *, start_epoch=0, start_step=0,
                        out_dir=None, ckpt_dir=None, stop_at=None, history=None, logs=None, val_fn=None):
        for epoch in range(start_epoch, args["epochs"]):
            # ── train_pretrain.py "8. 학습 시작" ──
            setup_seed(args["seed"] + epoch); indices = torch.randperm(len(train_ds)).tolist()
            skip = start_step if (epoch == start_epoch and start_step > 0) else 0
            batch_sampler = SkipBatchSampler(indices, args["batch_size"], skip)
            loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=0)
            iters = len(loader) + skip

            # ── train_epoch ──
            for step, (input_ids, labels) in enumerate(loader, start=skip + 1):
                input_ids = input_ids.to(device)
                labels = labels.to(device)
                lr = get_lr(epoch * iters + step, args["epochs"] * iters, args["learning_rate"])
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr

                with autocast_ctx:
                    res = model(input_ids, labels=labels)
                    loss = res.loss + res.aux_loss
                    loss = loss / args["accumulation_steps"]

                scaler.scale(loss).backward()

                updated = step % args["accumulation_steps"] == 0 or step == iters
                if updated:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args["grad_clip"])
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)

                current_loss = loss.item() * args["accumulation_steps"]
                if history is not None:  # 노트북 추가: 그래프용 기록
                    history.append({"epoch": epoch, "step": step, "loss": current_loss, "lr": lr, "updated": updated})
                if logs is not None and (step % args["log_interval"] == 0 or step == iters):
                    logs.append(f"Epoch:[{epoch + 1}/{args['epochs']}]({step}/{iters}), "
                                f"loss: {current_loss:.4f}, lr: {lr:.8f}")

                if out_dir is not None and (step % args["save_interval"] == 0 or step == iters):
                    model.eval()
                    ckp = Path(out_dir) / f"pretrain_{lm_config.hidden_size}.pth"   # 추론·다음 단계용 가중치
                    Path(out_dir).mkdir(parents=True, exist_ok=True)
                    torch.save({k: v.half().cpu() for k, v in model.state_dict().items()}, ckp)
                    lm_checkpoint(lm_config, weight="pretrain", model=model, optimizer=optimizer,
                                  scaler=scaler, epoch=epoch, step=step, save_dir=str(ckpt_dir))  # 재개용
                    model.train()

                if stop_at == (epoch, step):  # 노트북 추가: 중단 흉내
                    return

            if val_fn is not None:  # 노트북 추가: epoch마다 검증 손실
                val_fn(epoch)
    return (run_pretraining,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 7-2. 실행해 보기

    설정을 바꾸면 같은 초기 모델에서 다시 학습합니다(GPU에서 약 5초). 학습률 기본값은 원본과 같은 `5e-4`입니다.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    pt_epochs = mo.ui.slider(2, 40, step=2, value=20, label="epochs", show_value=True)
    pt_lr = mo.ui.dropdown(options={"5e-4": 5e-4, "1e-3": 1e-3, "5e-3": 5e-3, "2e-2": 2e-2, "1e-1": 1e-1}, value="5e-4", label="learning_rate")
    pt_batch = mo.ui.dropdown(options={"4": 4, "8": 8, "16": 16}, value="8", label="batch_size")
    pt_accum = mo.ui.dropdown(options={"1": 1, "2": 2, "4": 4}, value="2", label="accumulation_steps")
    mo.hstack([pt_epochs, pt_lr, pt_batch, pt_accum], justify="start", gap=2)
    return pt_accum, pt_batch, pt_epochs, pt_lr


@app.cell
def _(device, dtype, init_pretrain_model, pt_accum, pt_batch, pt_epochs, pt_lr, run_pretraining, stack_samples, torch, val_ds):
    pt_args = {
        "epochs": pt_epochs.value, "batch_size": pt_batch.value,
        "accumulation_steps": pt_accum.value, "learning_rate": pt_lr.value,
        "grad_clip": 1.0, "seed": 42, "log_interval": 4, "save_interval": 1000,
    }

    pt_model = init_pretrain_model(pt_args["seed"]).train()
    pt_optimizer = torch.optim.AdamW(pt_model.parameters(), lr=pt_args["learning_rate"])
    pt_scaler = torch.amp.GradScaler(device.type, enabled=(dtype == torch.float16))  # 원본: fp16일 때만 enabled

    _val_ids, _val_labels = stack_samples(val_ds, range(len(val_ds)))
    pt_history, pt_logs, pt_val = [], [], []

    def record_val_loss(epoch):
        pt_model.eval()
        with torch.no_grad():
            pt_val.append({"epoch": epoch, "seen_steps": len(pt_history),
                           "val_loss": pt_model(_val_ids, labels=_val_labels).loss.item()})
        pt_model.train()

    run_pretraining(pt_model, pt_optimizer, pt_scaler, pt_args,
                    history=pt_history, logs=pt_logs, val_fn=record_val_loss)
    pt_model.eval()
    pt_updates = sum(h["updated"] for h in pt_history)
    return pt_args, pt_history, pt_logs, pt_model, pt_updates, pt_val


@app.cell
def _(all_sentences, device, initial_model, math, pt_model, tokenizer, torch, train_ds):
    def continue_text(model, prompt, sample=False, seed=0):
        """bos + prompt 뒤를 이어 씁니다. 학습 데이터도 bos로 시작했으므로 생성도 bos로 시작합니다."""
        ids = torch.tensor([[tokenizer.bos_token_id] + tokenizer(prompt, add_special_tokens=False).input_ids], device=device)
        with torch.random.fork_rng(devices=[device] if device.type == "cuda" else []):
            torch.manual_seed(seed)
            out = model.generate(ids, max_new_tokens=10, do_sample=sample, top_k=0, top_p=1.0,
                                 temperature=1.0, eos_token_id=tokenizer.eos_token_id)
        return tokenizer.decode(out[0, ids.shape[1]:], skip_special_tokens=True)

    gen_prompts = ["The cat", "A child likes", "My friend"]
    gen_rows = [
        {"프롬프트": p, "학습 전 (greedy)": continue_text(initial_model, p),
         "학습 후 (greedy)": continue_text(pt_model, p),
         "학습 후 (샘플링 ×3)": " | ".join(continue_text(pt_model, p, sample=True, seed=s) for s in range(3))}
        for p in gen_prompts
    ]

    # 참고선: 72개 조합이 고르게 나온다면 문장 하나에서 줄일 수 없는 불확실성은 ln(72)입니다.
    _targets = sum(int((train_ds[i][1] != -100).sum()) - 1 for i in range(len(train_ds)))  # bos는 정답이 아님
    loss_floor_ref = math.log(len(all_sentences)) * len(train_ds) / _targets
    return gen_rows, loss_floor_ref


@app.cell(hide_code=True)
def _(gen_rows, loss_floor_ref, mo, plt, pt_args, pt_history, pt_logs, pt_updates, pt_val):
    _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(10, 3.2))
    _ax1.plot([h["loss"] for h in pt_history], lw=0.8, label="train loss (micro-batch)")
    _ax1.plot([v["seen_steps"] for v in pt_val], [v["val_loss"] for v in pt_val], "o-", ms=3, label="val loss (unseen sentences)")
    _ax1.axhline(loss_floor_ref, ls=":", color="gray", label="reference floor")
    _ax1.set_yscale("log")
    _ax1.set_xlabel("micro-batch step")
    _ax1.legend(fontsize=8)
    _ax2.plot([h["lr"] for h in pt_history])
    _ax2.set_xlabel("micro-batch step")
    _ax2.set_title("learning rate (get_lr)")
    _fig.tight_layout()
    _plot = mo.as_html(_fig)
    plt.close(_fig)

    mo.vstack([
        _plot,
        mo.md(f"""
    micro-batch **{len(pt_history)}개**, 실제 가중치 갱신 **{pt_updates}번**
    (`accumulation_steps={pt_args['accumulation_steps']}`). 마지막 학습 손실 `{pt_history[-1]['loss']:.3f}`,
    검증 손실 `{pt_val[-1]['val_loss']:.3f}`, 참고선 `{loss_floor_ref:.3f}`.
    """),
        mo.md("**원본과 같은 형식의 로그 (마지막 8줄)**\n```text\n" + "\n".join(pt_logs[-8:]) + "\n```"),
        mo.ui.table(gen_rows, selection=None),
        mo.md(r"""
    **읽는 법**

    - **참고선:** 문장의 주어·동사·목적어는 앞 내용으로 알 수 없는 무작위 선택입니다. 72개 조합이 고르게 나온다면
      문장당 `ln(72)`만큼은 어떤 모델도 줄일 수 없습니다. 이를 정답 토큰 수로 나눈 값입니다. 손실이 0이 되지 않는 것이 정상입니다.
      학습 문장은 72개 중 60개뿐이라 선택지가 조금 적으므로, 학습 손실은 참고선보다 약간 낮아질 수 있습니다.
    - **검증 손실**이 학습 손실과 함께 내려가면, 학습에 없던 조합도 잘 예측한다는 뜻입니다. 문장을 외운 것이 아니라
      "주어 + 동사 + 목적어 + 마침표 + 끝" 구조를 배웠습니다.
    - 생성은 문장 하나를 쓰고 **eos에서 멈춥니다**. 1-1절에서 본 eos 정답 덕분입니다.
    - 학습률을 `2e-2`, `1e-1`로 올려 손실 곡선을 비교해 보세요.
    """),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 8. 체크포인트 저장과 재개

    `save_interval`마다 파일 두 종류를 저장합니다.

    | 파일 | 저장 위치 (원본 기본값) | 내용 | 용도 |
    | --- | --- | --- | --- |
    | `pretrain_768.pth` | `../out/` | 가중치만, **float16** | 추론, 다음 단계 SFT의 `--from_weight pretrain` |
    | `pretrain_768_resume.pth` | `../checkpoints/` | 가중치(fp32) + **옵티마이저 상태** + scaler + epoch + step + GPU 수 | `--from_resume 1`로 이어서 학습 |

    (`lm_checkpoint`는 `../checkpoints/`에도 float16 가중치 파일을 하나 더 씁니다.)

    재개할 때 스크립트는 모델·옵티마이저·scaler 상태를 되돌리고, 끊긴 epoch의 순서를 같은 시드로 다시 만든 뒤
    `SkipBatchSampler`로 이미 본 배치를 건너뜁니다. 그러면 **끊기지 않은 학습과 완전히 같은 결과**가 나와야 합니다.

    아래에서 세 번 학습합니다.

    - **A:** 끊김 없이 4 epoch 학습
    - **B:** 같은 설정으로 시작해 epoch 2의 6번째 배치에서 강제 중단 (`save_interval=4`, 마지막 저장은 4번째 배치)
    - **C:** 새 모델·옵티마이저를 만들고 B의 체크포인트에서 `train_pretrain.py --from_resume 1`과 같은 순서로 재개

    실제 크기 모델이라 저장할 때마다 임시 폴더에 약 1GB(fp16 가중치 2개 + fp32 가중치·AdamW 상태)를 씁니다.
    """)
    return


@app.cell
def _(Path, device, dtype, init_pretrain_model, lm_checkpoint, lm_config, run_pretraining, shutil, tempfile, torch):
    resume_args = {"epochs": 4, "batch_size": 8, "accumulation_steps": 2, "learning_rate": 5e-4,
                   "grad_clip": 1.0, "seed": 42, "log_interval": 1000, "save_interval": 4}

    def fresh_training_state(args):
        model = init_pretrain_model(args["seed"]).train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=args["learning_rate"])
        scaler = torch.amp.GradScaler(device.type, enabled=(dtype == torch.float16))
        return model, optimizer, scaler

    def interrupted_then_resumed(args, stop_at):
        root = Path(tempfile.mkdtemp(prefix="minimind_pretrain_ckpt_"))
        # B: 중단될 학습
        model_b, opt_b, scaler_b = fresh_training_state(args)
        run_pretraining(model_b, opt_b, scaler_b, args, out_dir=root / "out",
                        ckpt_dir=root / "checkpoints", stop_at=stop_at)
        # C: train_pretrain.py의 "2. 체크포인트 확인"과 "6. 상태 복원"
        ckp_data = lm_checkpoint(lm_config, weight="pretrain", save_dir=str(root / "checkpoints"))
        model_c, opt_c, scaler_c = fresh_training_state(args)
        model_c.load_state_dict(ckp_data["model"])
        opt_c.load_state_dict(ckp_data["optimizer"])
        scaler_c.load_state_dict(ckp_data["scaler"])
        run_pretraining(model_c, opt_c, scaler_c, args,
                        start_epoch=ckp_data["epoch"], start_step=ckp_data.get("step", 0))
        return model_c, ckp_data, root

    def max_param_diff(m1, m2):
        return max((p - q).abs().max().item() for p, q in zip(m1.parameters(), m2.parameters()))

    # A: 끊김 없는 학습
    model_a, _opt_a, _scaler_a = fresh_training_state(resume_args)
    run_pretraining(model_a, _opt_a, _scaler_a, resume_args)

    model_c, resume_ckp, _ckpt_root = interrupted_then_resumed(resume_args, stop_at=(1, 6))
    # 저장된 파일 목록과 크기만 기록하고, 약 1GB의 임시 체크포인트는 지웁니다.
    ckpt_files = [(str(p.relative_to(_ckpt_root)), p.stat().st_size) for p in sorted(_ckpt_root.rglob("*.pth"))]
    shutil.rmtree(_ckpt_root)
    resume_diff = max_param_diff(model_a, model_c)
    assert resume_diff == 0.0  # 비트 단위로 같음
    return ckpt_files, fresh_training_state, interrupted_then_resumed, max_param_diff, resume_args, resume_ckp, resume_diff


@app.cell(hide_code=True)
def _(ckpt_files, mo, resume_ckp, resume_diff):
    _opt_state = resume_ckp["optimizer"]["state"]
    mo.vstack([
        mo.md("저장된 파일 (확인 후 임시 폴더는 삭제했습니다)"),
        mo.ui.table(
            [{"파일": name, "크기": f"{size / 2**20:.0f} MB"} for name, size in ckpt_files],
            selection=None,
        ),
        mo.md(f"""
    재개용 파일의 내용: `{list(resume_ckp)}`

    - 저장 시점: epoch `{resume_ckp['epoch']}` (0부터 셈), step `{resume_ckp['step']}`, GPU 수 `{resume_ckp['world_size']}`
    - 옵티마이저 상태: 매개변수 {len(_opt_state)}개 각각 `{list(next(iter(_opt_state.values())).keys())}`
    - 재개 파일이 가중치 파일보다 큰 이유: fp32 가중치(2배) + AdamW 상태(가중치의 2배)

    **A(끊김 없음)와 C(중단 후 재개)의 최종 가중치 최대 차이: `{resume_diff}`** → 완전히 같습니다.
    """),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 8-1. 주의: 저장 간격과 기울기 누적

    체크포인트에는 모델·옵티마이저 상태만 있고 **누적 중인 기울기(`p.grad`)는 없습니다.**
    `save_interval`이 `accumulation_steps`의 배수가 아니면, 누적 도중에 저장된 체크포인트에서 재개할 때
    그때까지 쌓인 기울기가 사라집니다. `save_interval=3`, `accumulation_steps=2`로 같은 실험을 합니다
    (epoch 2의 5번째 배치에서 중단, 마지막 저장은 누적 도중인 3번째 배치).
    """)
    return


@app.cell
def _(fresh_training_state, interrupted_then_resumed, max_param_diff, resume_args, run_pretraining, shutil):
    misaligned_args = {**resume_args, "save_interval": 3}
    _model_a, _opt_a, _scaler_a = fresh_training_state(misaligned_args)
    run_pretraining(_model_a, _opt_a, _scaler_a, misaligned_args)
    _model_c, misaligned_ckp, _root = interrupted_then_resumed(misaligned_args, stop_at=(1, 5))
    misaligned_diff = max_param_diff(_model_a, _model_c)
    shutil.rmtree(_root)  # 약 1GB 임시 체크포인트 정리
    assert misaligned_diff > 0
    return misaligned_ckp, misaligned_diff


@app.cell(hide_code=True)
def _(misaligned_ckp, misaligned_diff, mo):
    mo.md(f"""
    재개 지점: epoch `{misaligned_ckp['epoch']}`, step `{misaligned_ckp['step']}` (누적 2개 중 1개만 쌓인 상태).
    끊김 없는 학습과의 최종 가중치 최대 차이: **`{misaligned_diff:.2e}`** → 결과가 달라졌습니다.

    원본 기본값은 `save_interval=1000`, `accumulation_steps=8`로 1000이 8의 배수라 문제가 없습니다.
    둘 중 하나를 바꿀 때는 이 관계를 지켜야 재개 결과가 정확합니다.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    _D, _bs, _acc, _T = 768, 32, 8, 340
    mo.md(rf"""
    ## 9. 실제 규모로 실행하려면

    이 노트북은 실제 크기의 모델을 쓰지만 데이터는 합성 문장 60개뿐입니다. 실제 사전 학습은 다음처럼 실행합니다
    (데이터는 `projects/minimind/README_ko.md`의 안내에 따라 `projects/minimind/dataset/`에 내려받습니다).

    ```bash
    cd projects/minimind/trainer
    python train_pretrain.py                                  # GPU 1개, 기본 설정
    torchrun --nproc_per_node 2 train_pretrain.py             # GPU 여러 개 (DDP)
    python train_pretrain.py --from_resume 1                  # 끊긴 학습 이어서
    ```

    | 인자 | 기본값 | 이 노트북 |
    | --- | --- | --- |
    | `--hidden_size`, `--num_hidden_layers` | {_D}, 8 (약 64M 매개변수) | 같음 |
    | `--data_path` | `../dataset/pretrain_t2t_mini.jsonl` (1.2GB) | 합성 문장 60개 |
    | `--max_seq_len` | {_T} | 16 |
    | `--batch_size` × `--accumulation_steps` | {_bs} × {_acc} = {_bs * _acc}문장 / 갱신 | 8 × 2 |
    | `--learning_rate` | 5e-4 (→ 5e-5까지 코사인) | 같음 |
    | `--epochs` | 2 | 20 |
    | `--device`, `--dtype` | cuda:0, bfloat16 | 같음 (GPU가 없으면 CPU fp32) |
    | `--grad_clip` | 1.0 | 1.0 |
    | `--save_interval` | 1000 | 학습 1000 (재개 실험 4) |

    갱신 한 번에 최대 {_bs * _acc} × {_T} = **{_bs * _acc * _T:,} 토큰**을 봅니다.

    **여러 GPU(DDP)에서 달라지는 점**

    - GPU마다 프로세스가 하나씩 뜨고, 각자 시드 `seed + rank`로 시작합니다. `DistributedSampler`가 데이터를 GPU별로 나눕니다.
    - 로그와 저장은 0번 프로세스만 합니다(`is_main_process`).
    - 재개 시 GPU 수가 바뀌면 `lm_checkpoint`가 `step`을 `저장 시 GPU 수 / 현재 GPU 수` 비율로 바꿉니다.

    결과 `../out/pretrain_768.pth`는 다음 단계인 SFT(`train_full_sft.py --from_weight pretrain`)의 출발점이 됩니다.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 10. 정리

    | 단계 | 원본 | 핵심 |
    | --- | --- | --- |
    | 데이터 | `PretrainDataset` | bos + 본문 + eos + pad, 정답은 입력 복사본(패딩만 -100), 한 칸 밀기는 모델 안에서 |
    | 패딩 | `model(input_ids, labels)` | 오른쪽 패딩 + causal attention이라 mask 없이도 결과 동일 |
    | 갱신 | `train_epoch` | loss / 누적 수 → backward 누적 → clip → AdamW step |
    | 학습률 | `get_lr` | 설정값 → 10%까지 코사인 감소, warmup 없음, micro-batch마다 계산 |
    | 정밀도 | `autocast`, `GradScaler` | bf16 기본, fp16일 때만 배율 조정, CPU는 fp32 |
    | 순서 | `setup_seed(seed+epoch)`, `SkipBatchSampler` | epoch별 고정 순서, 재개 시 본 배치 건너뛰기 |
    | 저장 | `torch.save` + `lm_checkpoint` | fp16 가중치(다음 단계용) + 재개용 전체 상태 |

    ## 11. 직접 해 볼 과제

    각 과제를 하기 전에 **결과를 먼저 예측**해 적고, 바꿔 실행한 뒤 비교하세요.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({
        "과제 1. 학습률 바꾸기 (7-2절)": mo.md("""
    학습률을 `5e-4`, `5e-3`, `2e-2`, `1e-1`로 바꿔 손실 곡선과 생성 결과를 비교하세요.

    *해설:* `5e-4`(원본 기본값)는 수십 step 만에 손실이 1 아래로 내려갑니다. `5e-3`부터는 내려가는 속도가 느려지고,
    `2e-2`·`1e-1`은 첫 갱신 직후 손실이 오히려 크게 튀었다가(1e-1은 수십까지) 끝까지 회복하지 못합니다.
    clip은 기울기 크기만 자르고, AdamW는 학습률만큼 모든 가중치를 움직이므로 학습률이 크면 64M 모델이 한 번에 망가집니다.
    좋은 학습률은 모델 크기와 데이터에 따라 달라서, 작은 모델로 찾은 값을 그대로 옮길 수 없습니다.
    """),
        "과제 2. 누적 수 바꾸기 (7-2절)": mo.md("""
    `batch_size=8`에서 `accumulation_steps`를 1, 2, 4로 바꾸면 실제 갱신 횟수와 최종 손실은 어떻게 될까요?

    *해설:* 갱신 횟수가 대략 1/누적 수로 줄어듭니다. 같은 epoch 수라면 갱신이 적어 손실이 덜 내려갑니다.
    누적은 "큰 배치를 흉내 내는 방법"이지 공짜로 학습을 더 하는 방법이 아닙니다.
    """),
        "과제 3. 검증 문장을 처음 보는 주어로 (1절)": mo.md("""
    `val_sentences`를 `"The teacher"`로 시작하는 문장 12개로, `train_sentences`를 나머지로 바꾸면 검증 손실은 어떻게 될까요?

    *해설:* 학습 데이터에 `teacher`가 한 번도 나오지 않으므로, 그 토큰을 예측하는 위치의 손실이 크게 남습니다.
    검증 손실이 학습 손실보다 확실히 높아집니다. 모델은 본 적 없는 단어는 만들어 내지 못합니다.
    """),
        "과제 4. 저장 간격 (8-1절)": mo.md("""
    `misaligned_args`의 `save_interval`을 6으로 바꾸고, `stop_at`을 `(1, 7)`로 바꾸면 차이는 0일까요?

    *해설:* 마지막 저장이 6번째 배치(누적 2의 배수)라 누적 중인 기울기가 없으므로 차이가 0이 되고, `assert misaligned_diff > 0`이 실패합니다.
    """),
        "과제 5. 너무 짧은 max_seq_len (1절)": mo.md("""
    `max_seq_len`을 6으로 줄이면 학습 데이터는 어떻게 바뀌고, 생성 결과는 어떻게 될까요?

    *해설:* 본문이 앞 4토큰만 남고 eos가 붙습니다(예: `The cat eats an`). 모델은 목적어 중간에서 글이 끝난다고 배우므로,
    생성도 마침표 없이 중간에 끊긴 문장에서 멈춥니다.
    """),
    })
    return


if __name__ == "__main__":
    app.run()
