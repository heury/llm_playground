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

    **학습 목표:** `minimind-o/model/model_omni.py`의 실제 연산을 따라
    ① 서로 다른 입력을 같은 은닉 차원으로 연결하고 ② Thinker와 Talker를 구분하며
    ③ 텍스트와 8개 음향 코드의 시간 정렬 및 손실을 설명합니다.

    Python, `[배치, 길이, 차원]` 텐서와 다음 토큰 예측을 알고 시작하세요.
    **모델 구조 노트북 → 이 노트북 → 실제 Omni 데이터·추론 코드** 순서를 권장합니다.
    예상 60~90분. 각 절에서 **예측 → 실행 → 관찰 → 해설**을 진행하세요.

    ```text
    텍스트 ID ───────────────→ 텍스트 임베딩 ───────────┐
    음성 → fbank → SenseVoice → Audio Projector ── 자리 교체 ┤
    이미지 → SigLIP → Vision Projector ─────────── 자리 교체 ┘
                                     ↓
                               Thinker 블록들 → norm → LM head → 텍스트 logits
                                     │ 중간 bridge_states
                                     ↓ embed_proj × text_scale
    이전 음향 코드 8개 → TalkerEmbedding → codec_proj × audio_scale
                                     ↓ 두 경로 합산
                               Talker 블록들 → norm → 8개 head → 음향 코드
                                                                         ↓
                                                           외부 Mimi 디코더 → 파형
    ```

    여기서 *Omni*는 여러 모달리티를 다루는 모델이라는 뜻입니다. 이 구현의 출력은
    **텍스트와 음성 코드**이며 이미지를 생성하는 경로는 없습니다.
    Thinker는 이미지·음성을 넣은 임베딩을 처리하므로 멀티모달 문맥을 읽습니다.
    Talker는 생성된 문장을 다시 토큰화하는 대신 **중간 은닉 상태**를 전달받습니다.

    기본 실습은 CPU, 작은 무작위 모델, 합성 텐서만 사용합니다.
    모델·데이터 다운로드, 음성 녹음, 실제 인식·합성, 대규모 학습을 수행하지 않습니다.
    실제 원본 클래스를 사용하되 외부 인코더 로딩만 생략하고, 뒤에서 명시적인 교육용
    가짜 인코더를 연결합니다. 의미 있는 답변이나 들을 수 있는 음성을 만드는 실습은 아닙니다.
    marimo의 셀 인자와 return은 셀 사이 데이터 전달입니다. UI 변경은 관련 셀을 다시 실행합니다.
    """)
    return


@app.cell
def _(MiniMindOmni, OmniConfig, mo, torch):
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
    mo.ui.table([
        {"설정": "Thinker", "실습": "2층, D=32, Q=4, KV=2", "원본 기본": "8층, D=768, Q=8, KV=4"},
        {"설정": "Talker", "실습": "1층, D=32, Q=8, KV=4", "원본 기본": "4층, D=768, Q=8, KV=4"},
        {"설정": "bridge_layer (0부터)", "실습": "0 → 첫 블록 출력", "원본 기본": "8 // 2 - 1 = 3 → 네 번째 블록 출력"},
        {"설정": "텍스트 어휘 / 음향 어휘", "실습": "64 / 2112", "원본 기본": "6400 / 2112"},
        {"설정": "이미지 자리 수", "실습": "3", "원본 기본": "64"},
    ], selection=None)
    return ClassroomOmni, config, initial_model


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. 입력은 왜 9개 행일까?

    `[B,T]`이면 원본 `forward`가 8개 음향 행을 전부 PAD로 만듭니다.
    `[B,9,T]`이면 앞 8개 행은 음향 코드북 0~7, 마지막 행은 텍스트입니다.
    **코드북**은 음성 프레임을 표현하는 이산 코드들의 사전입니다. 이 구현은 프레임마다
    8개 코드북의 ID를 쓰며, 8개가 서로 다른 시점이라는 뜻은 아닙니다.

    | 값 | 텍스트 행에서 | 음향 행에서 |
    | --- | --- | --- |
    | 16 | 음성 특징을 넣을 자리(`<\|audio_pad\|>`) | 일반 음향 코드 ID일 수 있음 |
    | 12 | 이미지 특징을 넣을 자리(`<\|image_pad\|>`) | 일반 음향 코드 ID일 수 있음 |
    | 2049 / 2050 / 2051 | 이 실습 텍스트 사전에 없음 | PAD / STOP / 화자 자리 |

    **토큰 공간은 별개**입니다. 음성 입력용 텍스트 자리 16과 출력 음향 PAD 2049를 구분하세요.
    아래 숫자는 설명용 ID로 실제 한국어 토크나이저를 통과한 문장이 아닙니다.
    예측: `[B,T]`와 PAD로 채운 `[B,9,T]`의 텍스트·음향 출력은 같을까요?
    """)
    return


@app.cell
def _(config, initial_model, mo, torch):
    text_ids = torch.tensor([[1, 12, 12, 12, 9, 16, 16, 16, 10, 11, 2]])
    audio_ids = torch.full((1, 8, text_ids.size(1)), config.audio_pad_token, dtype=torch.long)
    packed_ids = torch.cat([audio_ids, text_ids.unsqueeze(1)], dim=1)
    with torch.no_grad():
        _plain = initial_model(text_ids)
        _packed = initial_model(packed_ids)
    torch.testing.assert_close(_plain.logits, _packed.logits)
    for _left, _right in zip(_plain.audio_logits, _packed.audio_logits):
        torch.testing.assert_close(_left, _right)
    mo.vstack([
        mo.md(f"입력 `{tuple(packed_ids.shape)}` → 텍스트 `{tuple(_packed.logits.shape)}`, "
              f"음향은 길이 `{len(_packed.audio_logits)}`인 list, 원소마다 `{tuple(_packed.audio_logits[0].shape)}`"),
        mo.ui.table([
            {"행": _i, "종류": "텍스트" if _i == 8 else f"음향 코드북 {_i}",
             "ID": str(packed_ids[0, _i].tolist())} for _i in range(9)
        ], selection=None),
        mo.md("검증: 2차원 입력과 PAD 음향을 붙인 3차원 입력의 두 출력이 같습니다."),
    ])
    return audio_ids, packed_ids, text_ids


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. Projector는 차원을 바꾸고, 자리 교체는 길이를 맞춘다

    `MMAudioProjector`와 `MMVisionProjector` 모두
    `LayerNorm → Linear → GELU → Linear`입니다. `[B,N,D_encoder]`를
    `[B,N,D_thinker]`로 바꾸며 **토큰 수 N을 줄이는 연산은 없습니다**.
    `MMVisionProjector`의 `source_tokens`, `target_tokens` 인자는 현재 구현에서 사용되지 않습니다.

    아래에서는 인코더 출력이라고 가정한 숫자 특징을 직접 만듭니다.
    외부 SenseVoice나 SigLIP의 의미 표현을 재현하지 않습니다.
    예측: 이미지 자리 3개에 특징 3개를 넣으면 전체 길이 T는 바뀔까요?
    음성 특징이 자리보다 길면 남는 특징은 어떻게 될까요?
    """)
    return


@app.cell
def _(initial_model, mo, text_ids, torch):
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
    mo.ui.table([
        {"위치": _t, "텍스트 ID": int(text_ids[0, _t]),
         "입력 출처": "이미지 특징" if 1 <= _t <= 3 else "음성 특징" if 5 <= _t <= 7 else "텍스트 임베딩",
         "교체 전후 L2 차이": float((injected_embeddings[0, _t] - base_embeddings[0, _t]).norm())}
        for _t in range(text_ids.size(1))
    ], selection=None)
    return base_embeddings, injected_embeddings, projected_audio, projected_vision, raw_audio, raw_vision


@app.cell
def _(mo):
    mo.md(r"""
    **관찰 해설:** 교체가 일어난 위치만 바뀌고 전체 길이는 유지됩니다.
    `inject_audio_features`는 첫 연속 음성 자리 구간에 `min(특징 수, 자리 수)`만 넣으며,
    남는 자리는 텍스트 임베딩으로 남습니다. 뒤의 음성 자리 구간에는 같은 특징을 다시 쓰지 않습니다.
    `count_vision_proj`는 이미지별 연속 자리 구간을 교체합니다. 특징 수가 부족하면 길이가
    줄어들 수 있으므로 실제 데이터에서도 이미지 자리 수와 특징 수를 맞춰야 합니다.
    `image_token_len=64`라는 설정만으로 projector가 64개를 만들어 주지는 않습니다.

    ## 3. 실제 forward 안에서 두 경로 만나기

    다음 가짜 인코더는 항상 위의 합성 특징을 반환합니다. 파형·픽셀을 이해하지 않습니다.
    대신 **원본 `encode_audio_inputs` → projector → 자리 교체 → Thinker → Talker**를
    실제로 통과시킵니다. 외부 인코더는 `no_grad`로 호출되지만, projector에는 gradient가 흐릅니다.
    원본은 외부 인코더를 `object.__setattr__`로 붙여 일반 하위 모듈 등록에서 제외합니다.
    실제 실행에서는 `eval_omni.py`처럼 인코더를 장치로 따로 옮기는 코드도 확인하세요.
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


@app.cell
def _(mo):
    bridge_scale = mo.ui.slider(0.0, 5.0, step=0.5, value=3.0, label="Thinker 전달 배율 text_scale")
    mo.vstack([
        mo.md("예측: Thinker→Talker 배율을 바꾸면 텍스트 logits도 달라질까요? 아래 표에서 두 경로를 비교하세요."),
        bridge_scale,
    ])
    return (bridge_scale,)


@app.cell
def _(bridge_scale, copy, injected_embeddings, mo, modal_inputs, multimodal_model, packed_ids, torch):
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
    _text_diff = float((multimodal_output.logits - _baseline.logits).abs().max())
    _audio_diff = float((multimodal_output.audio_logits[0] - _baseline.audio_logits[0]).abs().max())
    assert _text_diff == 0
    if bridge_scale.value != 3.0:
        assert _audio_diff > 0
    mo.vstack([
        mo.ui.table([{"지점": _k, "shape": str(tuple(_v.shape)), "L2 norm": float(_v.norm())}
                     for _k, _v in traces.items()], selection=None),
        mo.md(f"기본 배율 3 대비 최대 차이: 텍스트 **{_text_diff:.6f}**, 코드북 0 **{_audio_diff:.6f}**.\n\n"
              "Talker 입력 = `embed_proj(bridge) × text_scale + codec_proj(audio_embedding) × audio_scale`. "
              "배율은 학습 가능한 Parameter입니다. 0으로 만들어도 codec 경로와 각 층의 bias 등이 남습니다. "
              "이 실험은 한쪽 방향 연결을 확인하며 생성 품질을 비교하지 않습니다."),
    ])
    return multimodal_output, traces


@app.cell
def _(mo):
    mo.md(r"""
    ## 4. 8개 코드북을 어떻게 읽고 예측할까?

    `TalkerEmbedding`: 코드북마다 **공유 embedding + 개별 adapter**를 구한 후 8개를 평균합니다.
    `TalkerHead`: 하나의 은닉 상태에 **공유 Linear + 개별 adapter**를 적용해 8개 logits를 만듭니다.
    각 코드북은 같은 시점의 Talker 은닉 상태를 읽지만, 학습 데이터의 지연 배치 때문에
    각기 다른 오디오 프레임을 예측하게 됩니다. 한 번에 미래 텍스트 8개를 생성하는 연산은 아닙니다.

    아래 작은 독립 모듈만 rank=4로 만들고 원본 수식을 직접 계산합니다.
    전체 Omni의 head/embedding은 원본 기본 rank=256을 그대로 사용합니다.
    예측: head 8개의 결과가 항상 같을까요? 공유 항만 같고 개별 adapter는 다릅니다.
    """)
    return


@app.cell
def _(TalkerEmbedding, TalkerHead, mo, torch):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(7)
        _emb = TalkerEmbedding(20, 8, num_layers=8, rank=4)
        _head = TalkerHead(8, 20, num_layers=8, rank=4)
        _codes = torch.randint(0, 20, (1, 8, 3))
        _hidden = torch.randn(1, 3, 8)
    _manual_emb = sum(_emb.base(_codes[:, _i]) + _emb.adapters[_i](_codes[:, _i]) for _i in range(8)) / 8
    torch.testing.assert_close(_emb(_codes), _manual_emb)
    _outputs = _head(_hidden)
    for _i in range(8):
        torch.testing.assert_close(_outputs[_i], _head.base(_hidden) + _head.adapters[_i](_hidden))
    mo.md(f"Embedding 검증 통과: `[1,8,3] → {tuple(_manual_emb.shape)}`. "
          f"Head 검증 통과: `{len(_outputs)}`개 × `{tuple(_outputs[0].shape)}`. "
          f"head 0과 1의 최대 차이: `{float((_outputs[0] - _outputs[1]).detach().abs().max()):.4f}`.")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 5. 지연 배치: 같은 음성 프레임을 대각선으로 보내기

    원본 데이터셋의 `Y_audio_layers[k][assistant_start + k + 1 + f] = code[k,f]`를
    숫자로 재현합니다. 코드북 번호 k가 클수록 늦게 시작합니다.
    학습 입력은 `Y[:, :-1]`, 정답은 `Y[:, 1:]`입니다.
    따라서 출력 위치 t의 코드북 k 정답 프레임은 `f = t - assistant_start - k`입니다.
    **teacher forcing**은 학습 때 이전 시점의 정답 코드를 입력으로 제공한다는 뜻입니다.

    아래 표에서 `k0:f0`, `k1:f0`, … `k7:f0`는 같은 음성 프레임의 서로 다른 코드입니다.
    예측: 코드북 7의 첫 정답은 코드북 0보다 몇 위치 늦게 등장할까요?
    """)
    return


@app.cell
def _(config, mo, torch):
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
    _rows = []
    for _k in range(8):
        _row = {"코드북": _k}
        for _t in range(sequence_length - 1):
            _v = int(audio_targets[0, _k, _t])
            _f = _t - assistant_start - _k
            _row[f"t={_t}"] = "·" if _v == -100 else "STOP" if _v == config.audio_stop_token else f"k{_k}:f{_f}"
        _rows.append(_row)
        assert int(audio_targets[0, _k, assistant_start + _k]) == 10 + _k
    mo.vstack([
        mo.ui.table(_rows, selection=None),
        mo.md(f"모델 입력 `{tuple(training_ids.shape)}`, 텍스트 정답 `{tuple(text_targets.shape)}`, "
              f"음향 정답 `{tuple(audio_targets.shape)}`. `·`는 손실 제외(-100)입니다. "
              "입력과 정답이 이미 한 칸 어긋나 있으므로 모델 안에서 또 이동하지 않습니다."),
    ])
    return assistant_start, audio_targets, text_targets, training_ids


@app.cell
def _(mo):
    mo.md(r"""
    ## 6. 텍스트 CE + 음향 CE로 한 번 역전파하기

    `MiniMindOmni.forward`는 labels로 주 손실을 계산하지 않습니다. 학습 스크립트가
    텍스트 logits와 음향 logits의 CE를 계산합니다. 일반 MiniMind LM의 내부 shift와 다릅니다.

    - 텍스트: 유효한 응답 토큰 CE의 평균.
    - 음향: 코드북별로 유효 토큰 CE를 합하되 STOP(2050)은 **10배**, 유효 토큰 수로 나눈 후 8개 평균.
    - MoE일 때 `aux_loss`를 추가. 이 Dense 실습의 aux는 0입니다.

    STOP의 가중치를 분모에는 적용하지 않으므로 이것은 가중치 합으로 정규화한 평균이 아닙니다.
    정답이 없는 코드북은 0으로 기여하지만 마지막 나눗셈은 항상 8입니다.
    아래 함수는 `train_sft_omni.py`의 손실 계산을 축소해 옮겼습니다.
    합성 입력·정답으로 구조만 검증하며 실제 `OmniDataset`·Parquet·증강·scheduled sampling은 실행하지 않습니다.
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
def _(audio_targets, copy, initial_model, mo, omni_losses, text_targets, torch, training_ids):
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
    mo.vstack([
        mo.ui.table(gradient_rows, selection=None),
        mo.ui.table([
            {"시점": "학습 전", "텍스트 CE": initial_losses[0], "음향 CE": initial_losses[1]},
            {"시점": "5 step 후", "텍스트 CE": final_losses[0], "음향 CE": final_losses[1]},
        ], selection=None),
        mo.md("음향 손실은 bridge 이전 Thinker까지 전달됩니다. bridge 이후 블록에는 "
              "이 손실의 경로가 없습니다. 공동 손실에서는 텍스트 CE가 뒤쪽 Thinker도 학습합니다. "
              "5 step 결과는 동일 합성 배치의 손실 감소이며 일반화·음질 지표가 아닙니다."),
    ])
    return final_losses, gradient_rows, initial_losses


@app.cell
def _(mo):
    mo.md(r"""
    ## 7. projector 학습과 인코더 동결은 어떻게 함께 가능할까?

    `no_grad`는 **그 범위 안의 연산**을 기록하지 않습니다. 원본 음성 인코더 호출은
    no_grad 안에 있지만 `self.audio_proj(emb)`는 밖에 있습니다.
    이미지 경로도 같은 방식입니다. 고정 인코더 위에 붙은 projector를 학습할 수 있습니다.
    예측: `mode=audio_proj`처럼 나머지를 동결하면 Thinker가 동결되어도 음향 CE가 projector에 도달할까요?
    """)
    return


@app.cell
def _(audio_targets, copy, mo, modal_inputs, multimodal_model, omni_losses, packed_ids, text_ids, text_targets, torch):
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
    mo.md(f"Audio Projector gradient norm: **{projector_grad_norm:.6f}**. "
          "고정 음성 인코더와 Thinker의 parameter gradient는 없습니다. "
          "Thinker 연산 자체를 no_grad로 감싸면 projector로 돌아가는 경로도 끊기므로 구분하세요.\n\n"
          "원본 trainer의 `all`은 동결 설정을 별도로 확인해야 하며, `audio_proj`와 `vision_proj`는 "
          "전체 parameter를 먼저 동결한 뒤 해당 projector만 다시 활성화합니다.")
    return (projector_grad_norm,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 8. KV cache와 스트리밍: 언제 무엇을 출력할까?

    학습에서는 정답의 이전 값을 사용하지만 생성에서는 **직접 뽑은 텍스트·음향 코드를 다음 입력에 붙입니다**.
    `stream_generate`는 첫 호출에서 프롬프트 전체를 처리하고 이후 마지막 한 위치만 처리합니다.
    cache는 Thinker 층들 뒤에 Talker 층들이 이어지는 list입니다.
    음성·이미지 특징 삽입은 `start_pos == 0`일 때만 실행합니다.
    아래는 샘플링 대신 동일 토큰을 주어 cache 유무의 logits를 비교합니다.
    """)
    return


@app.cell
def _(config, initial_model, mo, packed_ids, torch):
    _prefix = packed_ids[:, :, :6]
    _last = packed_ids[:, :, 6:7]
    with torch.no_grad():
        _first = initial_model(_prefix, use_cache=True)
        _cached = initial_model(_last, past_key_values=_first.past_key_values, use_cache=True)
        _full = initial_model(packed_ids[:, :, :7], use_cache=False)
    cache_text_error = float((_cached.logits[:, -1] - _full.logits[:, -1]).abs().max())
    cache_audio_error = max(float((_a[:, -1] - _b[:, -1]).abs().max()) for _a, _b in zip(_cached.audio_logits, _full.audio_logits))
    torch.testing.assert_close(_cached.logits[:, -1], _full.logits[:, -1], atol=1e-5, rtol=1e-4)
    for _a, _b in zip(_cached.audio_logits, _full.audio_logits):
        torch.testing.assert_close(_a[:, -1], _b[:, -1], atol=1e-5, rtol=1e-4)
    assert len(_first.past_key_values) == config.num_hidden_layers + config.num_talker_hidden_layers
    mo.md(f"cache 원소 `{len(_first.past_key_values)}`개, 첫 K shape "
          f"`{tuple(_first.past_key_values[0][0].shape)}`. "
          f"최대 logits 오차: 텍스트 `{cache_text_error:.2e}`, 음향 `{cache_audio_error:.2e}`. "
          "이는 CPU float32의 텍스트·코드 입력 cache 동등성 검사입니다. "
          "외부 인코더를 포함한 실제 음성 스트리밍 검증은 아닙니다.")
    return cache_audio_error, cache_text_error


@app.cell
def _(mo):
    mo.md(r"""
    일반 모드(`open_thinking=False`)에서 생성 step s의 `audio_step = s - 1`입니다.
    코드북 k는 s ≥ k+1에서 시작하므로, **첫 텍스트 토큰을 뽑는 step 0에는 음향 코드가 없습니다**.
    step 8부터 `[audio_codes[k][s - 7 + k]]`를 모아 같은 프레임의 8개 코드를 반환할 수 있습니다.
    리스트에는 처음의 PAD도 저장되기 때문에 아래 인덱스는 단순한 프레임 번호가 아닙니다.

    `yield`는 `(생성 텍스트 누적값 또는 None, 8개짜리 audio_frame 또는 None)`입니다.
    텍스트 EOS 이후에도 음향 코드북의 종료를 기다리며 생성할 수 있습니다.
    코드에서 종료는 `code >= 2048`로 판정합니다. STOP 2050 하나만 검사하는 구현은 아닙니다.
    `open_thinking=True`일 때는 `think_end_ids` 탐지 후 기준 시점을 늦추는 별도 경로가 있습니다.
    아래 표와 검증은 일반 모드만 다룹니다.

    `generate(stream=False)`는 이 구현에서 마지막 yield를 그대로 반환하므로 일반적인
    Transformers의 토큰 텐서 반환을 가정하지 마세요. 실제 사용 예는 `eval_omni.py`의
    `stream=True` 반복을 확인하세요. 현재 샘플링 코드는 배치 1을 전제로 합니다.
    """)
    return


@app.cell
def _(mo):
    _rows = []
    for _s in range(10):
        _row = {"생성 step": _s, "텍스트": f"새 토큰 {_s}", "반환 프레임": "f0" if _s == 8 else "f1" if _s == 9 else "아직 없음"}
        for _k in range(8):
            _row[f"k{_k}"] = "PAD" if _s - 1 < _k else f"f{_s - 1 - _k}"
        _rows.append(_row)
    # 원본의 역지연 인덱스가 모든 k에서 프레임 f0를 가리키는지 확인.
    for _k in range(8):
        _list_index = 8 - 7 + _k
        assert _list_index - 1 - _k == 0
    mo.ui.table(_rows, selection=None)
    return


@app.cell
def _(initial_model, mo, torch):
    # 실제 원본 stream_generate도 짧게 실행합니다. 임의 출력이며 디코딩하지 않습니다.
    with torch.random.fork_rng(devices=[]), torch.no_grad():
        torch.manual_seed(123)
        _stream = list(initial_model.generate(
            torch.tensor([[1, 7, 8]]), eos_token_id=-1, max_new_tokens=10,
            stream=True, return_audio_codes=True, temperature=0.7,
        ))
    assert len(_stream) == 10
    assert all(_text is not None and _text.shape == (1, _i + 1) for _i, (_text, _frame) in enumerate(_stream))
    assert all(_frame is None for _text, _frame in _stream[:8])
    mo.ui.table([
        {"step": _i, "누적 텍스트 길이": _text.size(1),
         "audio_frame": str(_frame), "코드 수": len(_frame) if _frame is not None else 0}
        for _i, (_text, _frame) in enumerate(_stream)
    ], selection=None)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 9. 화자 조건과 실시간 VAD는 어디에 있을까?

    **화자 조건:** 첫 코드북 ID가 `audio_spk_token=2051`인 위치에서
    `spk_proj(spk_emb)`가 Talker embedding을 대체합니다. 이 실습은 화자 벡터 차원을 4로
    줄였고 원본 기본은 192입니다. `ref_codes`는 프롬프트 끝에 우측 정렬하는 참조 음향 코드이며
    화자 벡터와 별개입니다. 화자 벡터·코드가 있다고 복제 음질을 보장하는 것은 아닙니다.

    **파형 복원:** `model_omni.py`가 내보내는 것은 이산 코드입니다.
    `eval_omni.py`는 프레임을 `[1,8,F]`로 쌓아 외부 `MimiModel.decode`에 전달하고
    24kHz로 저장합니다. 입력 SenseVoice의 16kHz 전처리와 출력 주파수를 구분하세요.
    이 노트북은 Mimi를 로드하지 않으므로 코드 번호를 음성으로 들려주지 않습니다.

    **실시간 제어:** `SileroVAD`는 ONNX 모델의 발화 확률을 계산하고,
    `RealtimeSession`은 무음 길이·발화 길이를 세어 `listening`, `speech_end`, `interrupt`를
    반환합니다. 이 제어 계층은 Thinker/Talker의 학습 가능한 attention 블록이 아닙니다.
    아래는 실제 ONNX 대신 정해진 확률을 반환하는 VAD만 연결해 원본 상태 전이를 검사합니다.
    """)
    return


@app.cell
def _(config, copy, initial_model, mo, packed_ids, torch):
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
    _difference = float((_without.audio_logits[0] - _with.audio_logits[0]).abs().max())
    assert _difference > 0
    mo.md(f"화자 자리의 Talker embedding이 spk_proj 출력과 일치합니다. "
          f"다른 자리의 embedding은 같고, 텍스트 logits도 같습니다. "
          f"코드북 0 logits 최대 차이: `{_difference:.5f}`. "
          "이는 조건 전달 검사이며 실제 화자 유사도 측정은 아닙니다.")
    return


@app.cell
def _(RealtimeSession, mo, torch):
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
    _events = [_session.push_chunk(_chunk) for _ in range(5)]
    assert _events[-1] == "speech_end"
    _interrupt_session = ClassroomSession([0.95, 0.95])
    _interrupt_session.generating = True
    _interrupt_events = [_interrupt_session.push_chunk(_chunk) for _ in range(2)]
    assert _interrupt_events[-1] == "interrupt"
    mo.md(f"발화→무음 이벤트: `{_events}`\n\n생성 중 발화: `{_interrupt_events}`\n\n"
          "1024샘플은 16kHz에서 64ms입니다. 여기서는 2청크(128ms)를 최소 발화·무음으로 "
          "설정했습니다. 원본의 기본 최소 무음은 800ms입니다. 확률은 정해 준 값이며 VAD 정확도를 평가하지 않습니다.")
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

    1. 왜 이미지 projector의 차원 변환만으로 placeholder 수가 자동으로 맞춰지지 않을까?
    2. 왜 Talker는 Thinker의 최종 logits 대신 중간 hidden state를 받을까?
    3. 8개 head가 같은 시점에 예측하더라도 같은 음성 프레임을 모으려면 왜 기다려야 할까?
    4. labels가 이미 한 칸 이동했는데 한 번 더 shift하면 어떤 대응이 깨질까?
    5. 인코더의 동결과 모델 전체를 no_grad로 감싸는 것의 차이는 무엇일까?

    **해설:** projector는 마지막 특징 차원만 바꿉니다. bridge는 연속 의미 표현을 전달하고
    음향 손실이 앞부분 Thinker까지 흐르게 합니다. 코드북 k는 k+1 위치 지연되므로
    같은 프레임의 8개 코드를 모으려면 역지연 정렬이 필요합니다.
    두 번째 shift는 입력 t와 정답 t+1의 대응을 t+2로 바꿉니다.
    고정 층도 autograd가 입력에 대한 미분을 계산하면 앞쪽 projector를 학습할 수 있습니다.

    ## 다음으로 읽을 원본

    저장소 루트 기준 경로입니다.

    - `minimind-o/model/model_omni.py`: 이 노트북의 구성 요소, forward, 스트리밍, VAD.
    - `minimind-o/dataset/omni_dataset.py`: 대화 template, 마지막 assistant 마스크,
      8행 코드 전개, `assistant_start + layer_idx + 1`, labels의 `[1:]`.
    - `minimind-o/trainer/train_sft_omni.py`: 두 CE, STOP 가중치, projector 모드, collate.
    - `minimind-o/eval_omni.py`: 실제 tokenizer·인코더·가중치·Mimi 로딩 및 파형 저장.
    - `minimind-o/webui/web_demo.py`: 실시간 오디오와 끼어들기 제어.
    - `minimind-o/README_ko.md` 및 `minimind-o/LICENSE`: 리소스 준비, 실행 방법, 라이선스.

    실제 가중치로 확장할 때는 기본 config의 차원·토크나이저 special ID·코드북 수·
    placeholder 길이를 확인하고 프로젝트의 전용 환경에서 진행하세요.
    이 노트북의 축소 가중치는 공개 체크포인트와 호환되지 않습니다.
    """)
    return


@app.cell
def _(cache_audio_error, cache_text_error, datetime, final_losses, gradient_rows, hashlib, initial_losses, json, mo, projector_grad_norm, repo_root, uuid4, ZoneInfo):
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
    mo.md(f"이번 실행의 수치 기록: `{artifact_dir.relative_to(repo_root)}/metrics.json`. "
          "UI를 바꾸면 관련 셀과 수치 저장도 다시 실행됩니다. 이전 결과는 별도 실행 폴더로 남습니다.")
    return (artifact_dir,)


if __name__ == "__main__":
    app.run()
