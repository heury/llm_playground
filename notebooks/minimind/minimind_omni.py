import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import copy
    import hashlib
    import importlib.util
    import json
    import sys
    from datetime import datetime
    from pathlib import Path
    from types import SimpleNamespace
    from uuid import uuid4
    from zoneinfo import ZoneInfo

    import marimo as mo
    import torch
    import torch.nn.functional as F

    repo_root = Path(__file__).resolve().parents[2]
    # model이라는 공통 이름 대신 별도 패키지 이름으로 원본을 로드합니다.
    # 다른 MiniMind 실습을 같은 프로세스에서 import해도 프로젝트가 섞이지 않습니다.
    _name = "course_minimind_omni"
    _spec = importlib.util.spec_from_file_location(
        _name, repo_root / "projects/minimind-o/model/__init__.py",
        submodule_search_locations=[str(repo_root / "projects/minimind-o/model")],
    )
    _package = importlib.util.module_from_spec(_spec)
    sys.modules[_name] = _package
    _spec.loader.exec_module(_package)
    from course_minimind_omni.model_omni import (
        MiniMindOmni, OmniConfig, TalkerEmbedding, TalkerHead,
        RealtimeSession,
    )

    torch.set_num_threads(2)
    return (
        F, MiniMindOmni, OmniConfig, RealtimeSession, SimpleNamespace,
        TalkerEmbedding, TalkerHead, ZoneInfo, copy, datetime, hashlib,
        json, mo, repo_root, torch, uuid4,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # MiniMind-O: 보고 듣고 생각하고 말하는 모델

    **학습 목표:** `minimind-o/model/model_omni.py`의 실제 연산을 따라가며 설명합니다.

    1. 서로 다른 입력을 같은 은닉 차원으로 잇는 법
    2. Thinker와 Talker의 역할
    3. 텍스트와 음향 코드 8개의 시간 정렬, 그리고 손실

    **멀티모달 파트의 마지막 노트북**입니다(예상 60~90분).
    먼저 `minimind_vlm.py`와 `minimind_v_finetuning.py`를 읽으세요.
    이미지 경로는 같아서 짧게 봅니다. 새로운 것에 집중합니다:
    **음성 입력, Thinker→Talker 연결, 음향 코드북 8개 출력.**

    <img src="public/diagrams/omni.png" alt="MiniMind-O: Thinker와 Talker 연결 흐름" width="480">

    - *Omni*는 여러 모달리티를 다룬다는 뜻입니다.
    - 출력은 **텍스트와 음성 코드**입니다. 이미지는 만들지 않습니다.
    - Thinker: 이미지·음성이 들어간 임베딩을 읽습니다.
    - Talker: 문장을 다시 토큰화하지 않고 **중간 은닉 상태**를 받습니다.

    **실행 환경:** CPU, 작은 무작위 모델, 합성 텐서만 씁니다.

    - 다운로드, 녹음, 실제 인식·합성, 대규모 학습은 하지 않습니다.
    - 원본 클래스를 쓰되 외부 인코더 로딩만 뺍니다. 대신 가짜 인코더를 붙입니다.
    - 의미 있는 답이나 들을 수 있는 음성은 나오지 않습니다.
    """)
    return


@app.cell
def _(MiniMindOmni, OmniConfig, torch):
    class ClassroomOmni(MiniMindOmni):
        """외부 가중치 로딩만 생략. forward와 projector·삽입·생성은 원본 그대로."""
        @staticmethod
        def load_sensevoice(path):
            return None, None

        @staticmethod
        def load_vision(path):
            return None, None

    config = OmniConfig(
        hidden_size=32, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, intermediate_size=64,
        vocab_size=64, max_position_embeddings=64, dropout=0.0,
        flash_attn=False, use_moe=False,
        num_talker_hidden_layers=1, talker_hidden_size=32,
        audio_hidden_size=6, image_hidden_size=5, image_token_len=3,
        spk_emb_size=4, bridge_layer=0,
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        initial_model = ClassroomOmni(config).cpu().eval()
    assert initial_model.thinker is initial_model.model
    assert initial_model.thinker.lm_head is initial_model.lm_head
    assert initial_model.lm_head.weight is initial_model.thinker.embed_tokens.weight
    return (ClassroomOmni, config, initial_model,)


@app.cell(hide_code=True)
def _(mo):
    # ▸ 결과 표시 — 실행하면 표·그림이 나타납니다 (marimo 표시 코드)
    mo.ui.table([
        {"설정": "Thinker", "실습": "2층, D=32, Q=4, KV=2", "원본 기본": "8층, D=768, Q=8, KV=4"},
        {"설정": "Talker", "실습": "1층, D=32, Q=8, KV=4", "원본 기본": "4층, D=768, Q=8, KV=4"},
        {"설정": "bridge_layer (0부터)", "실습": "0 → 첫 블록 출력", "원본 기본": "8 // 2 - 1 = 3 → 네 번째 블록 출력"},
        {"설정": "텍스트 어휘 / 음향 어휘", "실습": "64 / 2112", "원본 기본": "6400 / 2112"},
        {"설정": "이미지 자리 수", "실습": "3", "원본 기본": "64"},
    ], selection=None)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. 입력은 왜 9개 행일까?

    - `[B,T]`를 넣으면: `forward`가 음향 8행을 모두 PAD로 채웁니다.
    - `[B,9,T]`를 넣으면: 앞 8행은 음향 코드북 0~7, 마지막 행은 텍스트입니다.

    **코드북:** 음성 프레임을 표현하는 이산 코드 사전입니다.
    프레임마다 코드북 8개의 ID를 씁니다. 8개가 서로 다른 시점이라는 뜻은 아닙니다.

    | 값 | 텍스트 행에서 | 음향 행에서 |
    | --- | --- | --- |
    | 16 | 음성 특징을 넣을 자리(`<\|audio_pad\|>`) | 일반 음향 코드 ID일 수 있음 |
    | 12 | 이미지 특징을 넣을 자리(`<\|image_pad\|>`) | 일반 음향 코드 ID일 수 있음 |
    | 2049 / 2050 / 2051 | 이 실습 텍스트 사전에 없음 | PAD / STOP / 화자 자리 |

    **텍스트와 음향은 토큰 공간이 다릅니다.** 텍스트의 음성 자리 16과 음향 PAD 2049를 구분하세요.
    아래 숫자는 설명용 ID입니다. 실제 토크나이저 결과가 아닙니다.

    **예측:** `[B,T]`와 PAD로 채운 `[B,9,T]`의 출력은 같을까요?
    """)
    return


@app.cell
def _(config, initial_model, torch):
    text_ids = torch.tensor([[1, 12, 12, 12, 9, 16, 16, 16, 10, 11, 2]])
    audio_ids = torch.full((1, 8, text_ids.size(1)), config.audio_pad_token, dtype=torch.long)
    packed_ids = torch.cat([audio_ids, text_ids.unsqueeze(1)], dim=1)
    with torch.no_grad():
        _plain = initial_model(text_ids)
        packed_output = initial_model(packed_ids)
    torch.testing.assert_close(_plain.logits, packed_output.logits)
    for _left, _right in zip(_plain.audio_logits, packed_output.audio_logits):
        torch.testing.assert_close(_left, _right)
    return (audio_ids, packed_ids, packed_output, text_ids,)


@app.cell(hide_code=True)
def _(mo, packed_ids, packed_output):
    # ▸ 결과 표시 — 실행하면 표·그림이 나타납니다 (marimo 표시 코드)
    mo.vstack([
        mo.md(f"입력 `{tuple(packed_ids.shape)}` → 텍스트 `{tuple(packed_output.logits.shape)}`, "
              f"음향은 길이 `{len(packed_output.audio_logits)}`인 list, 원소마다 `{tuple(packed_output.audio_logits[0].shape)}`"),
        mo.ui.table([
            {"행": _i, "종류": "텍스트" if _i == 8 else f"음향 코드북 {_i}",
             "ID": str(packed_ids[0, _i].tolist())} for _i in range(9)
        ], selection=None),
        mo.md("검증: 2차원 입력과 PAD 음향을 붙인 3차원 입력의 두 출력이 같습니다."),
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. Projector는 차원을 바꾸고, 자리 교체는 길이를 맞춘다

    이미지 경로는 VLM 노트북 3~4절과 같습니다.
    새 `MMAudioProjector`도 같은 구조입니다: `LayerNorm → Linear → GELU → Linear`.
    **마지막 차원만** 바꾸고 토큰 수는 그대로입니다.
    다른 점은 음성 교체 함수 `inject_audio_features`의 규칙입니다(아래 해설).

    아래 특징은 인코더 출력이라고 가정한 숫자입니다. SenseVoice·SigLIP을 재현하지 않습니다.

    **예측:**

    - 이미지 자리 3개에 특징 3개를 넣으면 길이 T가 바뀔까요?
    - 음성 특징이 자리보다 많으면 남는 특징은?
    """)
    return


@app.cell
def _(initial_model, text_ids, torch):
    raw_audio = torch.arange(18, dtype=torch.float32).reshape(1, 3, 6) / 10
    raw_vision = torch.arange(15, dtype=torch.float32).reshape(1, 3, 5) / 10
    with torch.no_grad():
        base_embeddings = initial_model.thinker.embed_tokens(text_ids)
        projected_audio = initial_model.audio_proj(raw_audio)
        projected_vision = initial_model.vision_proj(raw_vision)
        _audio_injected = initial_model.inject_audio_features(
            text_ids, base_embeddings, [projected_audio[0]], text_ids.size(1),
        )
        injected_embeddings = initial_model.count_vision_proj(
            text_ids, _audio_injected, projected_vision, text_ids.size(1),
        )
    torch.testing.assert_close(injected_embeddings[:, 1:4], projected_vision)
    torch.testing.assert_close(injected_embeddings[:, 5:8], projected_audio)
    torch.testing.assert_close(injected_embeddings[:, [0, 4, 8, 9, 10]], base_embeddings[:, [0, 4, 8, 9, 10]])
    assert injected_embeddings.shape == base_embeddings.shape
    return (
        base_embeddings,
        injected_embeddings,
        projected_audio,
        projected_vision,
        raw_audio,
        raw_vision,
    )


@app.cell(hide_code=True)
def _(base_embeddings, injected_embeddings, mo, text_ids):
    # ▸ 결과 표시 — 실행하면 표·그림이 나타납니다 (marimo 표시 코드)
    mo.ui.table([
        {"위치": _t, "텍스트 ID": int(text_ids[0, _t]),
         "입력 출처": "이미지 특징" if 1 <= _t <= 3 else "음성 특징" if 5 <= _t <= 7 else "텍스트 임베딩",
         "교체 전후 L2 차이": float((injected_embeddings[0, _t] - base_embeddings[0, _t]).norm())}
        for _t in range(text_ids.size(1))
    ], selection=None)
    return


@app.cell
def _(mo):
    mo.md(r"""
    **관찰 해설:** 교체된 위치만 바뀝니다. 전체 길이는 그대로입니다.

    **음성 (`inject_audio_features`):**

    - 첫 연속 음성 구간에 `min(특징 수, 자리 수)`개만 넣습니다.
    - 남는 자리는 텍스트 임베딩 그대로입니다.
    - 뒤의 음성 구간에는 특징을 다시 넣지 않습니다.

    **이미지 (`count_vision_proj`):**

    - 이미지별 연속 구간을 교체합니다.
    - 특징이 부족하면 길이가 줄 수 있습니다. 자리 수와 특징 수를 맞추세요.
    - `image_token_len=64`로 설정해도 projector가 64개를 만들어 주지는 않습니다.

    ## 3. 실제 forward 안에서 두 경로 만나기

    가짜 인코더는 항상 위의 합성 특징을 돌려줍니다. 파형·픽셀은 이해하지 않습니다.
    대신 실제 경로를 그대로 지납니다:
    **`encode_audio_inputs` → projector → 자리 교체 → Thinker → Talker.**

    - 외부 인코더는 `no_grad`로 호출됩니다. projector에는 gradient가 흐릅니다.
    - 원본은 인코더를 `object.__setattr__`로 붙여 하위 모듈 등록에서 뺍니다.
    - 그래서 실제 실행에서는 인코더를 장치로 따로 옮겨야 합니다(`eval_omni.py` 참고).
    """)
    return


@app.cell
def _(SimpleNamespace, copy, initial_model, raw_audio, raw_vision, torch):
    class FixedAudioEncoder(torch.nn.Module):
        def __init__(self, features):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(()), requires_grad=False)
            self.register_buffer("features", features)

        def forward(self, fbank, lengths):
            return self.features.expand(fbank.size(0), -1, -1), lengths

    class FixedVisionEncoder(torch.nn.Module):
        def __init__(self, features):
            super().__init__()
            self.register_buffer("features", features)

        def forward(self, pixel_values):
            return SimpleNamespace(last_hidden_state=self.features.expand(pixel_values.size(0), -1, -1))

    multimodal_model = copy.deepcopy(initial_model)
    object.__setattr__(multimodal_model, "audio_encoder", FixedAudioEncoder(raw_audio))
    object.__setattr__(multimodal_model, "vision_encoder", FixedVisionEncoder(raw_vision))
    # 원본이 all-zero 입력을 '음성/이미지 없음'으로 판정하므로 1로 채웁니다.
    modal_inputs = {
        "audio_inputs": torch.ones(1, 3, 6),
        "audio_lens": torch.tensor([3]),
        "pixel_values": torch.ones(1, 3, 2, 2),
    }
    return modal_inputs, multimodal_model


@app.cell(hide_code=True)
def _(mo):
    # ▸ 실습 설정 위젯 — 실행하면 슬라이더·선택 상자가 나타납니다 (marimo 표시 코드)
    bridge_scale = mo.ui.slider(0.0, 5.0, step=0.5, value=3.0, label="Thinker 전달 배율 text_scale")
    mo.vstack([
        mo.md("예측: Thinker→Talker 배율을 바꾸면 텍스트 logits도 달라질까요? 아래 표에서 두 경로를 비교하세요."),
        bridge_scale,
    ])
    return (bridge_scale,)


@app.cell
def _(bridge_scale, copy, injected_embeddings, modal_inputs, multimodal_model, packed_ids, torch):
    observed_model = copy.deepcopy(multimodal_model)
    with torch.no_grad():
        observed_model.talker.text_scale.fill_(bridge_scale.value)
    traces = {}
    _handles = []

    def capture(name):
        def hook(module, inputs, output):
            value = output[0] if isinstance(output, tuple) else output
            traces[name] = value.detach().clone()
        return hook

    _handles.append(observed_model.thinker.layers[0].register_forward_pre_hook(
        lambda module, inputs: traces.update(injected=inputs[0].detach().clone())
    ))
    _handles.append(observed_model.thinker.layers[0].register_forward_hook(capture("bridge")))
    _handles.append(observed_model.thinker.norm.register_forward_hook(capture("thinker_final")))
    _handles.append(observed_model.talker.norm.register_forward_hook(capture("talker_final")))
    try:
        with torch.no_grad():
            multimodal_output = observed_model(packed_ids, **modal_inputs)
    finally:
        for _h in _handles:
            _h.remove()
    torch.testing.assert_close(traces["injected"], injected_embeddings)
    with torch.no_grad():
        _baseline = multimodal_model(packed_ids, **modal_inputs)
    text_diff = float((multimodal_output.logits - _baseline.logits).abs().max())
    audio_diff = float((multimodal_output.audio_logits[0] - _baseline.audio_logits[0]).abs().max())
    assert text_diff == 0
    if bridge_scale.value != 3.0:
        assert audio_diff > 0
    return (audio_diff, multimodal_output, text_diff, traces,)


@app.cell(hide_code=True)
def _(audio_diff, mo, text_diff, traces):
    # ▸ 결과 표시 — 실행하면 표·그림이 나타납니다 (marimo 표시 코드)
    mo.vstack([
        mo.ui.table([{"지점": _k, "shape": str(tuple(_v.shape)), "L2 norm": float(_v.norm())}
                     for _k, _v in traces.items()], selection=None),
        mo.md(f"기본 배율 3 대비 최대 차이: 텍스트 **{text_diff:.6f}**, 코드북 0 **{audio_diff:.6f}**.\n\n"
              "Talker 입력 = `embed_proj(bridge) × text_scale + codec_proj(audio_embedding) × audio_scale`. "
              "배율은 학습되는 Parameter입니다. 0으로 해도 codec 경로와 bias 등은 남습니다. "
              "연결 방향만 확인하는 실험입니다. 생성 품질 비교가 아닙니다."),
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 4. 8개 코드북을 어떻게 읽고 예측할까?

    - `TalkerEmbedding`: 코드북마다 **공유 embedding + 개별 adapter**. 8개를 평균합니다.
    - `TalkerHead`: 은닉 상태 하나에 **공유 Linear + 개별 adapter**. logits 8개를 만듭니다.

    코드북 8개는 같은 시점의 은닉 상태를 읽습니다.
    하지만 지연 배치(5절) 때문에 각각 다른 프레임을 예측합니다.
    미래 텍스트 8개를 한 번에 만드는 것이 아닙니다.

    아래는 작은 모듈(rank=4)로 수식을 직접 계산합니다. 전체 Omni는 원본 rank=256을 씁니다.

    **예측:** head 8개의 결과는 같을까요? 공유 항은 같고 adapter는 다릅니다.
    """)
    return


@app.cell
def _(TalkerEmbedding, TalkerHead, torch):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(7)
        _emb = TalkerEmbedding(20, 8, num_layers=8, rank=4)
        _head = TalkerHead(8, 20, num_layers=8, rank=4)
        _codes = torch.randint(0, 20, (1, 8, 3))
        _hidden = torch.randn(1, 3, 8)
    manual_emb = sum(_emb.base(_codes[:, _i]) + _emb.adapters[_i](_codes[:, _i]) for _i in range(8)) / 8
    torch.testing.assert_close(_emb(_codes), manual_emb)
    head_outputs = _head(_hidden)
    for _i in range(8):
        torch.testing.assert_close(head_outputs[_i], _head.base(_hidden) + _head.adapters[_i](_hidden))
    return (head_outputs, manual_emb,)


@app.cell(hide_code=True)
def _(head_outputs, manual_emb, mo):
    # ▸ 결과 설명 — 실행하면 계산 값이 들어간 설명이 나타납니다 (marimo 표시 코드)
    mo.md(f"Embedding 검증 통과: `[1,8,3] → {tuple(manual_emb.shape)}`. "
          f"Head 검증 통과: `{len(head_outputs)}`개 × `{tuple(head_outputs[0].shape)}`. "
          f"head 0과 1의 최대 차이: `{float((head_outputs[0] - head_outputs[1]).detach().abs().max()):.4f}`.")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 5. 지연 배치: 같은 음성 프레임을 대각선으로 보내기

    원본 데이터셋 규칙을 숫자로 재현합니다:
    `Y_audio_layers[k][assistant_start + k + 1 + f] = code[k,f]`.

    - 코드북 번호 k가 클수록 늦게 시작합니다.
    - 입력은 `Y[:, :-1]`, 정답은 `Y[:, 1:]`입니다.
    - 그래서 위치 t, 코드북 k의 정답 프레임은 `f = t - assistant_start - k`입니다.
    - **teacher forcing:** 학습 때 이전 정답 코드를 입력으로 줍니다.

    표의 `k0:f0`, `k1:f0`, … `k7:f0`는 같은 프레임의 서로 다른 코드입니다.

    **예측:** 코드북 7의 첫 정답은 코드북 0보다 몇 칸 늦을까요?
    """)
    return


@app.cell
def _(config, torch):
    assistant_start = 3
    frame_count = 2
    sequence_length = 16
    target_text_sequence = torch.tensor([[1, 7, 8, 20, 21, 2] + [0] * 10])
    original_audio = torch.full((1, 8, sequence_length), config.audio_pad_token, dtype=torch.long)
    original_audio_labels = torch.full_like(original_audio, -100)
    for _k in range(8):
        _start = assistant_start + _k + 1
        _codes = torch.tensor([10 + _k, 30 + _k, config.audio_stop_token])
        original_audio[0, _k, _start:_start + 3] = _codes
        original_audio_labels[0, _k, _start:_start + 3] = _codes
    training_ids = torch.cat([original_audio[:, :, :-1], target_text_sequence[:, None, :-1]], dim=1)
    text_targets = target_text_sequence[:, 1:].clone()
    # assistant 응답의 두 토큰과 EOS만 학습 대상으로 남깁니다.
    text_targets[:, :assistant_start - 1] = -100
    text_targets[:, 5:] = -100
    audio_targets = original_audio_labels[:, :, 1:]
    delay_rows = []
    for _k in range(8):
        _row = {"코드북": _k}
        for _t in range(sequence_length - 1):
            _v = int(audio_targets[0, _k, _t])
            _f = _t - assistant_start - _k
            _row[f"t={_t}"] = "·" if _v == -100 else "STOP" if _v == config.audio_stop_token else f"k{_k}:f{_f}"
        delay_rows.append(_row)
        assert int(audio_targets[0, _k, assistant_start + _k]) == 10 + _k
    return (assistant_start, audio_targets, delay_rows, text_targets, training_ids,)


@app.cell(hide_code=True)
def _(audio_targets, delay_rows, mo, text_targets, training_ids):
    # ▸ 결과 표시 — 실행하면 표·그림이 나타납니다 (marimo 표시 코드)
    mo.vstack([
        mo.ui.table(delay_rows, selection=None),
        mo.md(f"모델 입력 `{tuple(training_ids.shape)}`, 텍스트 정답 `{tuple(text_targets.shape)}`, "
              f"음향 정답 `{tuple(audio_targets.shape)}`. `·`는 손실 제외(-100)입니다. "
              "입력과 정답이 이미 한 칸 밀려 있어서 모델 안에서 또 옮기지 않습니다."),
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 6. 텍스트 CE + 음향 CE로 한 번 역전파하기

    `MiniMindOmni.forward`는 손실을 계산하지 않습니다. 학습 스크립트가 직접 CE를 계산합니다.
    일반 MiniMind LM처럼 내부에서 shift하지 않습니다.

    - **텍스트:** 유효 응답 토큰 CE의 평균.
    - **음향:** 코드북마다 CE 합 ÷ 유효 토큰 수. STOP(2050)은 **10배**. 그 뒤 8개 평균.
    - **MoE면** `aux_loss` 추가. 여기서는 Dense라 0입니다.

    **주의:**

    - STOP 가중치는 분모에 들어가지 않습니다. 가중 평균이 아닙니다.
    - 정답이 없는 코드북은 0이지만, 나눗셈은 항상 8로 합니다.

    아래 함수는 `train_sft_omni.py`의 손실 계산을 줄여 옮긴 것입니다.
    합성 데이터로 구조만 확인합니다. `OmniDataset`, Parquet, 증강, scheduled sampling은 쓰지 않습니다.
    """)
    return


@app.cell
def _(F, torch):
    def omni_losses(output, text_targets, audio_targets):
        raw_text = F.cross_entropy(output.logits.reshape(-1, output.logits.size(-1)), text_targets.reshape(-1), reduction="none")
        valid_text = text_targets.reshape(-1) != -100
        text_loss = (raw_text * valid_text).sum() / (valid_text.sum() + 1e-9)
        audio_loss = output.audio_logits[0].sum() * 0
        for k, logits in enumerate(output.audio_logits):
            target = audio_targets[:, k].reshape(-1)
            valid = target != -100
            raw = F.cross_entropy(logits.reshape(-1, logits.size(-1)), target, reduction="none")
            weight = 1 + (target == 2050).float() * 9
            if valid.any():
                audio_loss = audio_loss + (raw * valid * weight).sum() / (valid.sum() + 1e-9)
        return text_loss, audio_loss / 8
    return (omni_losses,)


@app.cell
def _(audio_targets, copy, initial_model, omni_losses, text_targets, torch, training_ids):
    gradient_model = copy.deepcopy(initial_model)
    gradient_model.train()
    _out = gradient_model(training_ids)
    _text_loss, _audio_loss = omni_losses(_out, text_targets, audio_targets)
    # 음향 손실만으로 어느 Thinker 층까지 연결되는지 확인합니다.
    (_audio_loss + _out.aux_loss).backward()
    gradient_rows = []
    for _name, _module in [
        ("Thinker 첫 블록 (bridge까지)", gradient_model.thinker.layers[0]),
        ("Thinker 두 번째 블록 (bridge 이후)", gradient_model.thinker.layers[1]),
        ("Talker", gradient_model.talker),
    ]:
        _norm = sum(float(_p.grad.square().sum()) for _p in _module.parameters() if _p.grad is not None) ** 0.5
        gradient_rows.append({"모듈": _name, "음향 CE만의 gradient norm": _norm})
    assert gradient_rows[0]["음향 CE만의 gradient norm"] > 0
    assert gradient_rows[1]["음향 CE만의 gradient norm"] == 0
    gradient_model.zero_grad(set_to_none=True)
    with torch.no_grad():
        _initial_out = gradient_model(training_ids)
        initial_losses = tuple(float(_v) for _v in omni_losses(_initial_out, text_targets, audio_targets))
    _optimizer = torch.optim.AdamW(gradient_model.parameters(), lr=1e-3)
    for _step in range(5):
        _optimizer.zero_grad(set_to_none=True)
        _out = gradient_model(training_ids)
        _tl, _al = omni_losses(_out, text_targets, audio_targets)
        _loss = _tl + _al + _out.aux_loss
        assert torch.isfinite(_loss)
        _loss.backward()
        _optimizer.step()
    with torch.no_grad():
        final_losses = tuple(float(_v) for _v in omni_losses(gradient_model(training_ids), text_targets, audio_targets))
    assert sum(final_losses) < sum(initial_losses)
    return (final_losses, gradient_rows, initial_losses,)


@app.cell(hide_code=True)
def _(final_losses, gradient_rows, initial_losses, mo):
    # ▸ 결과 표시 — 실행하면 표·그림이 나타납니다 (marimo 표시 코드)
    mo.vstack([
        mo.ui.table(gradient_rows, selection=None),
        mo.ui.table([
            {"시점": "학습 전", "텍스트 CE": initial_losses[0], "음향 CE": initial_losses[1]},
            {"시점": "5 step 후", "텍스트 CE": final_losses[0], "음향 CE": final_losses[1]},
        ], selection=None),
        mo.md("음향 손실은 bridge 이전 Thinker까지만 갑니다. bridge 이후 블록에는 경로가 없습니다. "
              "뒤쪽 Thinker는 텍스트 CE가 학습시킵니다. "
              "5 step 결과는 같은 합성 배치의 손실 감소일 뿐입니다. 일반화·음질 지표가 아닙니다."),
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 7. projector 학습과 인코더 동결은 어떻게 함께 가능할까?

    `no_grad`는 **그 범위 안의 연산만** 기록하지 않습니다.

    - 음성 인코더 호출은 no_grad 안에 있습니다.
    - `self.audio_proj(emb)`는 밖에 있습니다.
    - 그래서 고정 인코더 위의 projector를 학습할 수 있습니다. 이미지도 같습니다.

    **예측:** `mode=audio_proj`로 나머지를 동결해도, 음향 CE가 projector까지 갈까요?
    """)
    return


@app.cell
def _(audio_targets, copy, modal_inputs, multimodal_model, omni_losses, packed_ids, text_ids, text_targets):
    _model = copy.deepcopy(multimodal_model)
    for _p in _model.parameters():
        _p.requires_grad_(False)
    for _p in _model.audio_proj.parameters():
        _p.requires_grad_(True)
    # 길이가 같은 앞 11위치로 합성 정답을 잘라 두 출력 CE를 계산합니다.
    _output = _model(packed_ids, **modal_inputs)
    _tl, _al = omni_losses(_output, text_targets[:, :text_ids.size(1)], audio_targets[:, :, :text_ids.size(1)])
    (_tl + _al + _output.aux_loss).backward()
    projector_grad_norm = sum(float(_p.grad.square().sum()) for _p in _model.audio_proj.parameters() if _p.grad is not None) ** 0.5
    assert projector_grad_norm > 0
    assert all(_p.grad is None for _p in _model.audio_encoder.parameters())
    assert all(_p.grad is None for _p in _model.thinker.parameters())
    return (projector_grad_norm,)


@app.cell(hide_code=True)
def _(mo, projector_grad_norm):
    # ▸ 결과 설명 — 실행하면 계산 값이 들어간 설명이 나타납니다 (marimo 표시 코드)
    mo.md(f"Audio Projector gradient norm: **{projector_grad_norm:.6f}**. "
          "음성 인코더와 Thinker의 parameter에는 gradient가 없습니다. "
          "단, Thinker 연산을 no_grad로 감싸면 projector로 가는 경로도 끊깁니다.\n\n"
          "원본 trainer 모드: `audio_proj`, `vision_proj`는 전체를 동결한 뒤 해당 projector만 켭니다. "
          "`all`은 동결 설정을 따로 확인하세요.")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 8. KV cache와 스트리밍: 언제 무엇을 출력할까?

    학습은 이전 정답을 입력으로 씁니다.
    생성은 **직접 뽑은 텍스트·음향 코드를 다음 입력에 붙입니다.**

    - `stream_generate`: 첫 호출은 프롬프트 전체, 이후는 마지막 한 위치만 처리합니다.
    - cache: Thinker 층들 뒤에 Talker 층들이 이어진 list입니다.
    - 음성·이미지 삽입은 `start_pos == 0`일 때만 합니다.

    아래는 같은 토큰을 넣어 cache 있을 때와 없을 때의 logits를 비교합니다.
    """)
    return


@app.cell
def _(config, initial_model, packed_ids, torch):
    _prefix = packed_ids[:, :, :6]
    _last = packed_ids[:, :, 6:7]
    with torch.no_grad():
        cache_prefix_output = initial_model(_prefix, use_cache=True)
        _cached = initial_model(_last, past_key_values=cache_prefix_output.past_key_values, use_cache=True)
        _full = initial_model(packed_ids[:, :, :7], use_cache=False)
    cache_text_error = float((_cached.logits[:, -1] - _full.logits[:, -1]).abs().max())
    cache_audio_error = max(float((_a[:, -1] - _b[:, -1]).abs().max()) for _a, _b in zip(_cached.audio_logits, _full.audio_logits))
    torch.testing.assert_close(_cached.logits[:, -1], _full.logits[:, -1], atol=1e-5, rtol=1e-4)
    for _a, _b in zip(_cached.audio_logits, _full.audio_logits):
        torch.testing.assert_close(_a[:, -1], _b[:, -1], atol=1e-5, rtol=1e-4)
    assert len(cache_prefix_output.past_key_values) == config.num_hidden_layers + config.num_talker_hidden_layers
    return (cache_audio_error, cache_prefix_output, cache_text_error,)


@app.cell(hide_code=True)
def _(cache_audio_error, cache_prefix_output, cache_text_error, mo):
    # ▸ 결과 설명 — 실행하면 계산 값이 들어간 설명이 나타납니다 (marimo 표시 코드)
    mo.md(f"cache 원소 `{len(cache_prefix_output.past_key_values)}`개, 첫 K shape "
          f"`{tuple(cache_prefix_output.past_key_values[0][0].shape)}`. "
          f"최대 logits 오차: 텍스트 `{cache_text_error:.2e}`, 음향 `{cache_audio_error:.2e}`. "
          "CPU float32에서 cache 결과가 같은지만 확인했습니다. "
          "실제 음성 스트리밍 검증은 아닙니다.")
    return


@app.cell
def _(mo):
    mo.md(r"""
    **일반 모드(`open_thinking=False`)의 타이밍:**

    - 생성 step s에서 `audio_step = s - 1`.
    - 코드북 k는 s ≥ k+1부터 시작합니다. 그래서 **step 0에는 음향 코드가 없습니다.**
    - step 8부터 같은 프레임의 코드 8개를 모아 반환합니다: `[audio_codes[k][s - 7 + k]]`.
    - 리스트에 처음 PAD도 들어 있어서, 인덱스가 곧 프레임 번호는 아닙니다.

    **`yield` 형식:** `(누적 텍스트 또는 None, 8개 audio_frame 또는 None)`.

    - 텍스트 EOS 뒤에도 음향 종료를 기다리며 계속 생성할 수 있습니다.
    - 종료 판정은 `code >= 2048`입니다. STOP 2050만 보는 게 아닙니다.
    - `open_thinking=True`는 `think_end_ids`를 찾은 뒤 시점을 늦추는 별도 경로입니다. 아래는 일반 모드만 봅니다.

    **주의:** `generate(stream=False)`는 마지막 yield를 그대로 반환합니다. 토큰 텐서가 아닙니다.
    사용 예는 `eval_omni.py`의 `stream=True` 반복을 보세요. 샘플링 코드는 배치 1 전제입니다.
    """)
    return


@app.cell
def _():
    stream_schedule_rows = []
    for _s in range(10):
        _row = {"생성 step": _s, "텍스트": f"새 토큰 {_s}", "반환 프레임": "f0" if _s == 8 else "f1" if _s == 9 else "아직 없음"}
        for _k in range(8):
            _row[f"k{_k}"] = "PAD" if _s - 1 < _k else f"f{_s - 1 - _k}"
        stream_schedule_rows.append(_row)
    # 원본의 역지연 인덱스가 모든 k에서 프레임 f0를 가리키는지 확인.
    for _k in range(8):
        _list_index = 8 - 7 + _k
        assert _list_index - 1 - _k == 0
    return (stream_schedule_rows,)


@app.cell(hide_code=True)
def _(mo, stream_schedule_rows):
    # ▸ 결과 표시 — 실행하면 표·그림이 나타납니다 (marimo 표시 코드)
    mo.ui.table(stream_schedule_rows, selection=None)
    return


@app.cell
def _(initial_model, torch):
    with torch.random.fork_rng(devices=[]), torch.no_grad():
        torch.manual_seed(123)
        stream_steps = list(initial_model.generate(
            torch.tensor([[1, 7, 8]]), eos_token_id=-1, max_new_tokens=10,
            stream=True, return_audio_codes=True, temperature=0.7,
        ))
    assert len(stream_steps) == 10
    assert all(_text is not None and _text.shape == (1, _i + 1) for _i, (_text, _frame) in enumerate(stream_steps))
    assert all(_frame is None for _text, _frame in stream_steps[:8])
    return (stream_steps,)


@app.cell(hide_code=True)
def _(mo, stream_steps):
    # ▸ 결과 표시 — 실행하면 표·그림이 나타납니다 (marimo 표시 코드)
    mo.ui.table([
        {"step": _i, "누적 텍스트 길이": _text.size(1),
         "audio_frame": str(_frame), "코드 수": len(_frame) if _frame is not None else 0}
        for _i, (_text, _frame) in enumerate(stream_steps)
    ], selection=None)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 9. 화자 조건과 실시간 VAD는 어디에 있을까?

    **화자 조건:**

    - 첫 코드북 ID가 `audio_spk_token=2051`인 위치에서 `spk_proj(spk_emb)`가 embedding을 대신합니다.
    - 화자 벡터 차원: 실습 4, 원본 192.
    - `ref_codes`는 참조 음향 코드입니다(프롬프트 끝에 우측 정렬). 화자 벡터와 별개입니다.
    - 화자 정보가 있다고 목소리 복제 품질이 보장되지는 않습니다.

    **파형 복원:**

    - `model_omni.py`의 출력은 이산 코드입니다.
    - `eval_omni.py`가 `[1,8,F]`로 쌓아 `MimiModel.decode`에 넘기고 24kHz로 저장합니다.
    - 입력(SenseVoice 16kHz)과 출력(24kHz) 주파수가 다릅니다.
    - 이 노트북은 Mimi를 쓰지 않아서 소리는 들을 수 없습니다.

    **실시간 제어:**

    - `SileroVAD`: ONNX 모델로 발화 확률을 계산합니다.
    - `RealtimeSession`: 무음·발화 길이를 세어 `listening`, `speech_end`, `interrupt`를 돌려줍니다.
    - 학습되는 attention 블록이 아닌 별도 제어 계층입니다.

    아래는 정해진 확률을 내는 가짜 VAD로 상태 전이만 확인합니다.
    """)
    return


@app.cell
def _(config, copy, initial_model, packed_ids, torch):
    _model = copy.deepcopy(initial_model)
    _ids = packed_ids.clone()
    _ids[:, :8, 0] = config.audio_spk_token
    _speaker = torch.ones(1, config.spk_emb_size)
    _seen = []
    _handle = _model.talker.codec_proj.register_forward_pre_hook(
        lambda module, inputs: _seen.append(inputs[0].detach().clone())
    )
    try:
        with torch.no_grad():
            _without = _model(_ids)
            _with = _model(_ids, spk_emb=_speaker)
            _expected = _model.talker.spk_proj(_speaker)
    finally:
        _handle.remove()
    torch.testing.assert_close(_seen[1][:, 0], _expected)
    torch.testing.assert_close(_seen[0][:, 1:], _seen[1][:, 1:])
    torch.testing.assert_close(_without.logits, _with.logits)
    speaker_logit_diff = float((_without.audio_logits[0] - _with.audio_logits[0]).abs().max())
    assert speaker_logit_diff > 0
    return (speaker_logit_diff,)


@app.cell(hide_code=True)
def _(mo, speaker_logit_diff):
    # ▸ 결과 설명 — 실행하면 계산 값이 들어간 설명이 나타납니다 (marimo 표시 코드)
    mo.md(f"화자 자리의 embedding이 spk_proj 출력과 같습니다. "
          f"다른 자리와 텍스트 logits는 그대로입니다. "
          f"코드북 0 logits 최대 차이: `{speaker_logit_diff:.5f}`. "
          "조건이 전달되는지만 확인했습니다. 화자 유사도 측정은 아닙니다.")
    return


@app.cell
def _(RealtimeSession, torch):
    class ScriptedVAD:
        def __init__(self, probabilities):
            self.values = list(probabilities)
            self.reset()

        def reset(self):
            self.position = 0

        def __call__(self, chunk, sr):
            value = self.values[self.position]
            self.position += 1
            return value

    class ClassroomSession(RealtimeSession):
        def __init__(self, probabilities):
            self.vad = ScriptedVAD(probabilities)
            self.sr, self.threshold = 16000, 0.8
            self.min_speech, self.min_silence = 2048, 2048
            self.reset()

    _session = ClassroomSession([0.0, 0.95, 0.95, 0.0, 0.0])
    _chunk = torch.zeros(1024).numpy()
    events = [_session.push_chunk(_chunk) for _ in range(5)]
    assert events[-1] == "speech_end"
    _interrupt_session = ClassroomSession([0.95, 0.95])
    _interrupt_session.generating = True
    interrupt_events = [_interrupt_session.push_chunk(_chunk) for _ in range(2)]
    assert interrupt_events[-1] == "interrupt"
    return (events, interrupt_events,)


@app.cell(hide_code=True)
def _(events, interrupt_events, mo):
    # ▸ 결과 설명 — 실행하면 계산 값이 들어간 설명이 나타납니다 (marimo 표시 코드)
    mo.md(f"발화→무음 이벤트: `{events}`\n\n생성 중 발화: `{interrupt_events}`\n\n"
          "1024샘플 = 16kHz에서 64ms. 여기서는 최소 발화·무음을 2청크(128ms)로 두었습니다. "
          "원본 기본 최소 무음은 800ms입니다. 확률은 정해 준 값이라 VAD 정확도는 평가하지 않습니다.")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 10. 이해 확인과 수정 과제

    | 과제 | 먼저 예측할 것 | 실행 후 기록할 것 |
    | --- | --- | --- |
    | 3절 UI의 text_scale을 3→0으로 변경 | 텍스트/음향 중 어느 출력이 바뀔까? | 각 logits 최대 차이 |
    | config의 bridge_layer를 0→1로 변경 | 음향 CE gradient가 어느 블록까지 흐를까? | 6절에서 두 블록 norm; 기존 두 번째=0 assert도 새 가설에 맞게 수정 |
    | raw_audio를 2프레임으로 줄이기 | 음성 자리 3개 중 남는 위치는? | 2절의 교체 slice·assert와 modal_inputs 길이를 함께 수정 |
    | 5절 STOP을 제외해 보기 | 음향 CE가 얼마나 달라질까? | 토큰별 CE·유효 정답 수; 종료 학습에 미칠 영향 |
    | 생성 max_new_tokens를 8→10 비교 | 프레임을 반환할 기회가 있는가? | None/8코드 프레임 개수; 특수 코드가 먼저 나오면 프레임이 없을 수도 있음 |

    **스스로 설명하기**

    1. projector가 차원을 바꿔도 placeholder 수는 왜 자동으로 안 맞을까?
    2. Talker는 왜 최종 logits 대신 중간 hidden state를 받을까?
    3. head 8개가 동시에 예측하는데, 같은 프레임을 모으려면 왜 기다려야 할까?
    4. labels가 이미 밀려 있는데 또 shift하면 무엇이 깨질까?
    5. 인코더 동결과 모델 전체 no_grad는 무엇이 다를까?

    **해설:**

    1. projector는 마지막 차원만 바꿉니다. 토큰 수는 그대로입니다.
    2. bridge는 연속 의미 표현을 넘깁니다. 음향 손실이 앞쪽 Thinker까지 흐릅니다.
    3. 코드북 k는 k+1칸 늦습니다. 같은 프레임을 모으려면 지연을 되돌려야 합니다.
    4. 입력 t ↔ 정답 t+1 대응이 t+2로 어긋납니다.
    5. 고정된 층도 입력에 대한 미분은 계산합니다. 그래서 앞쪽 projector를 학습할 수 있습니다.

    ## 다음으로 읽을 원본

    저장소 루트 기준 경로입니다.

    - `minimind-o/model/model_omni.py`: 이 노트북의 구성 요소, forward, 스트리밍, VAD.
    - `minimind-o/dataset/omni_dataset.py`: 대화 template, 마지막 assistant 마스크,
      8행 코드 전개, `assistant_start + layer_idx + 1`, labels의 `[1:]`.
    - `minimind-o/trainer/train_sft_omni.py`: 두 CE, STOP 가중치, projector 모드, collate.
    - `minimind-o/eval_omni.py`: 실제 tokenizer·인코더·가중치·Mimi 로딩 및 파형 저장.
    - `minimind-o/webui/web_demo.py`: 실시간 오디오와 끼어들기 제어.
    - `minimind-o/README_ko.md` 및 `minimind-o/LICENSE`: 리소스 준비, 실행 방법, 라이선스.

    **실제 가중치로 확장할 때:** config 차원, special ID, 코드북 수, placeholder 길이를 확인하세요.
    프로젝트 전용 환경에서 실행하세요.
    이 노트북의 작은 가중치는 공개 체크포인트와 호환되지 않습니다.
    """)
    return


@app.cell
def _(ZoneInfo, cache_audio_error, cache_text_error, datetime, final_losses, gradient_rows, hashlib, initial_losses, json, projector_grad_norm, repo_root, uuid4):
    _run_id = datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y%m%d-%H%M%S") + "-" + uuid4().hex[:6] + "-omni"
    artifact_dir = repo_root / "outputs/notebooks" / _run_id
    artifact_dir.mkdir(parents=True, exist_ok=True)
    _source = repo_root / "projects/minimind-o/model/model_omni.py"
    _metrics = {
        "source": str(_source.relative_to(repo_root)),
        "source_sha256": hashlib.sha256(_source.read_bytes()).hexdigest(),
        "initial_losses": initial_losses,
        "final_losses": final_losses,
        "audio_only_gradients": gradient_rows,
        "audio_projector_gradient_norm": projector_grad_norm,
        "cache_text_max_error": cache_text_error,
        "cache_audio_max_error": cache_audio_error,
        "scope": "CPU synthetic tensors; fixed fake encoders; no pretrained weights or real audio",
    }
    (artifact_dir / "metrics.json").write_text(json.dumps(_metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    return (artifact_dir,)


@app.cell(hide_code=True)
def _(artifact_dir, mo, repo_root):
    # ▸ 결과 설명 — 실행하면 계산 값이 들어간 설명이 나타납니다 (marimo 표시 코드)
    mo.md(f"이번 실행의 수치 기록: `{artifact_dir.relative_to(repo_root)}/metrics.json`. "
          "UI를 바꾸면 관련 셀과 저장이 다시 실행됩니다. 이전 결과는 다른 폴더에 남습니다.")
    return


if __name__ == "__main__":
    app.run()
