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
    # SmolVLM2 보충 실습: VQA와 좌표 프로브

    [SmolVLA 구조 해부](smolvla_tour.py)에서 분리한 선택 실험입니다.
    텍스트 생성과 층별 좌표의 선형 해독 가능성을 확인합니다. 이 결과만으로 행동 성능이나 최적 절단 깊이를 판정하지 않습니다.

    실행 순서: 모델 로드 → VQA 또는 좌표 프로브 → 층별 연산 시간 측정.
    `notebooks/lerobot`에서 `uv run marimo edit smolvla_vlm_experiments.py`로 엽니다.
    모델 로드 버튼은 SmolVLM2 약 2 GB를 다운로드할 수 있습니다. GPU에서는 bf16, CPU에서는 fp32입니다.
    정책 가중치는 필요하지 않습니다. 각 실험은 자체 버튼으로 실행하며, 모델을 다시 로드하면 결과가 무효화됩니다.
    장면 정의는 학습 노트북, 전처리는 구조 해부 노트북의 이름 있는 셀을 재사용합니다. 다른 앱을 실행할 필요는 없습니다.

    원본: [Hugging Face LeRobot SmolVLA](https://github.com/huggingface/lerobot/tree/e6ffde60312365c5452c46189a26872483943154/src/lerobot/policies/smolvla)
    및 같은 리비전의 `policies/common/flow_matching.py`, `policies/common/vla_utils.py`, `inference/execution.py`.
    고정 리비전 `e6ffde60312365c5452c46189a26872483943154`, Apache License 2.0,
    Copyright 2025–2026 The HuggingFace Inc. team. 원본을 축소·수정한 실습입니다.
    """)
    return


@app.cell
def _():
    import time

    import matplotlib.font_manager as font_manager
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    import torch.nn.functional as F

    # 그림의 한글 표시용 글꼴: 설치된 것 중 첫 번째를 사용합니다.
    _installed = {f.name for f in font_manager.fontManager.ttflist}
    for _name in ["Noto Sans CJK KR", "NanumGothic", "Malgun Gothic", "AppleGothic", "Noto Sans Mono CJK KR"]:
        if _name in _installed:
            plt.rcParams["font.family"] = _name
            break
    plt.rcParams["axes.unicode_minus"] = False
    return F, plt, time, torch


@app.cell
def _(torch):
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ON_GPU = DEVICE.type == "cuda"
    VLM_ID = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
    VLM_REV = "4cde9d208e19519c36917aa170fffc7e381aa0ad"
    return DEVICE, ON_GPU, VLM_ID, VLM_REV


@app.cell
def _(torch):
    from smolvla_training import scene_definitions

    _, _defs = scene_definitions.run(torch=torch)
    render_scenes = _defs["render_scenes"]
    sample_scenes = _defs["sample_scenes"]
    return render_scenes, sample_scenes


@app.cell
def _(F):
    from smolvla_tour import image_preprocessing

    _, _defs = image_preprocessing.run(F=F)
    resize_with_pad = _defs["resize_with_pad"]
    return (resize_with_pad,)


@app.cell
def _(mo):
    load_btn = mo.ui.run_button(label="SmolVLM2 불러오기")
    load_btn
    return (load_btn,)


@app.cell
def _(DEVICE, ON_GPU, VLM_ID, VLM_REV, load_btn, mo, torch):
    mo.stop(not load_btn.value, mo.md("모델 로드 버튼을 눌러 시작하세요."))
    from transformers import AutoModelForImageTextToText, AutoTokenizer

    with mo.status.spinner("SmolVLM2 불러오는 중..."):
        tok = AutoTokenizer.from_pretrained(VLM_ID, revision=VLM_REV)
        vlm_full = AutoModelForImageTextToText.from_pretrained(
            VLM_ID, revision=VLM_REV, dtype=torch.bfloat16 if ON_GPU else torch.float32
        ).to(DEVICE).eval()
    return tok, vlm_full


@app.cell
def _(mo, vlm_full):
    _v, _t = vlm_full.config.vision_config, vlm_full.config.text_config
    _n_patch = (_v.image_size // _v.patch_size) ** 2
    _sf = vlm_full.config.scale_factor
    mo.md(f"""
    | 부분 | 설정 | 값 |
    | --- | --- | --- |
    | 비전 인코더 | 입력 크기 / 패치 / 층 / 폭 | {_v.image_size}px / {_v.patch_size}px / {_v.num_hidden_layers} / {_v.hidden_size} |
    | | 패치 수 | ({_v.image_size}/{_v.patch_size})² = **{_n_patch}** |
    | 커넥터 | Pixel Shuffle 배율 | {_sf} → 토큰 {_n_patch} / {_sf}² = **{_n_patch // _sf**2}**, 폭 {_v.hidden_size}×{_sf}² = {_v.hidden_size * _sf**2} → {_t.hidden_size} |
    | 텍스트 디코더 | 층 / 폭 / Q 헤드 / KV 헤드 / head_dim | {_t.num_hidden_layers} / {_t.hidden_size} / {_t.num_attention_heads} / {_t.num_key_value_heads} / {_t.head_dim} |
    | | RoPE theta (VLM 원래 값) | {_t.rope_parameters["rope_theta"] if hasattr(_t, "rope_parameters") else _t.rope_theta} |
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. VLM이 정말 이미지를 읽는지 VQA로 확인

    SmolVLM2의 채팅 형식은 이미지 자리에 `<image>` 토큰 64개를 넣고, 모델 안에서 이 자리를 **비전 토큰 64개로 바꿔 끼웁니다**.
    `AutoProcessor`가 하는 일을 손으로 재현합니다(512×512 한 장이면 이미지 분할 없이 전역 이미지 1개입니다).

    ```text
    <|im_start|>User:<fake_token_around_image><global-img><image>×64<fake_token_around_image>질문<end_of_utterance>\nAssistant:
    ```

    픽셀은 SmolVLA와 같게 `resize_with_pad`로 512×512에 맞춘 뒤 `[0,1] → [-1,1]`로 바꿉니다(SigLIP의 mean=std=0.5 정규화와 같음).
    """)
    return


@app.function
def vqa_prompt(question):
    return (
        "<|im_start|>User:<fake_token_around_image><global-img>"
        + "<image>" * 64 + "<fake_token_around_image>" + question
        + "<end_of_utterance>\nAssistant:"
    )


@app.cell
def _(mo):
    vqa_question = mo.ui.text(value="Is the red dot on the left or the right side of the image?", full_width=True, label="질문")
    vqa_question
    return (vqa_question,)


@app.cell
def _(mo):
    vqa_btn = mo.ui.run_button(label='VQA 질문 실행')
    vqa_btn
    return (vqa_btn,)


@app.cell
def _(
    DEVICE,
    mo,
    plt,
    render_scenes,
    resize_with_pad,
    tok,
    torch,
    vlm_full,
    vqa_btn,
    vqa_question,
):
    mo.stop(not vqa_btn.value, mo.md('질문을 고친 뒤 VQA 실행 버튼을 누르세요.'))
    _red, _blue = torch.tensor([[-0.5, -0.4]]), torch.tensor([[0.5, 0.5]])
    _img = render_scenes(_red, _blue)
    _pixels = (resize_with_pad(_img, 512, 512) * 2 - 1).to(DEVICE, vlm_full.dtype)[:, None]  # [B, 이미지 수, 3, H, W]
    _ids = tok(vqa_prompt(vqa_question.value), return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        _out = vlm_full.generate(**_ids, pixel_values=_pixels, max_new_tokens=40, do_sample=False)
    _answer = tok.decode(_out[0, _ids.input_ids.shape[1] :], skip_special_tokens=True)
    print("프롬프트 토큰 수:", _ids.input_ids.shape[1], "(이 중 <image> 64개)")
    print("답:", _answer.strip())
    _fig, _ax = plt.subplots(figsize=(2.6, 2.6))
    _ax.imshow(_img[0].permute(1, 2, 0), extent=[-1, 1, 1, -1])
    _ax.set_xticks([]), _ax.set_yticks([])
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    질문을 바꾸고 실제 그림의 색·위치·개수와 답을 대조하세요.
    SmolVLA는 VLM의 답변 텍스트 대신 중간 층의 K/V를 행동 생성에 씁니다.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. 층별 좌표 프로브와 레이어 절단

    SmolVLA는 VLM 텍스트 디코더 32층 중 **앞쪽(입력 쪽) 16층**만 남기고 나머지를 잘라냅니다(`num_vlm_layers=16`).
    원본 코드는 `text_model.layers = text_model.layers[:num_vlm_layers]` 한 줄입니다.
    [smolvla.md](smolvla.md)의 구조 안내와 같은 **하위(앞쪽) 층**입니다.

    뒤쪽 층을 줄이면 계산·캐시 비용을 낮출 수 있습니다. 행동에 필요한 표현이 앞쪽 층에도 남는다는 가설을
    **선형 프로브**로 확인합니다. 무작위 장면 768개에 `"Where is the red dot?"`를 묻고, 층마다 마지막 토큰의 은닉 상태(960차원)에서
    빨간 점 좌표를 **선형 회귀(ridge)** 로 맞혀 봅니다. 학습 512개, 평가 256개의 결정계수 $R^2$를 비교합니다.
    표준화 통계는 학습 부분에서만 구하고 ridge 계수는 고정 10입니다. $R^2$의 기준은 평가 표적 평균이며 음수가 나올 수도 있습니다.

    ridge 해는 $W=(X^TX+\lambda I)^{-1}X^TY$입니다. $X$는 표준화한 은닉 특징과 상수 열, $Y$는 2D 좌표입니다.
    특징 960개가 학습 장면 512개보다 많아 해가 불안정할 수 있으므로, $\lambda>0$로 큰 가중치를 억제하고 연립방정식을 안정화합니다.
    코드에서는 역행렬을 직접 만들지 않고 `torch.linalg.solve`로 풉니다. $\lambda=10$은 이 비교에서 고정한 값이며 최적값이라는 뜻은 아닙니다.
    """)
    return


@app.cell
def _(mo):
    probe_btn = mo.ui.run_button(label="선형 프로브 실험 실행 (GPU 약 10초)")
    probe_btn
    return (probe_btn,)


@app.cell
def _(
    DEVICE,
    mo,
    probe_btn,
    render_scenes,
    resize_with_pad,
    sample_scenes,
    time,
    tok,
    torch,
    vlm_full,
):
    mo.stop(not probe_btn.value, mo.md("*버튼을 누르면 실행합니다.*"))
    _g = torch.Generator().manual_seed(0)
    _n_train, _n_test = 512, 256
    probe_red, probe_blue, _ = sample_scenes(_n_train + _n_test, generator=_g)
    _ids = tok(vqa_prompt("Where is the red dot?"), return_tensors="pt").to(DEVICE)
    _feats = []
    _t0 = time.time()
    with torch.no_grad(), mo.status.progress_bar(total=(_n_train + _n_test) // 32, title="은닉 상태 추출") as _bar:
        for _i in range(0, _n_train + _n_test, 32):
            _pix = (resize_with_pad(render_scenes(probe_red[_i : _i + 32], probe_blue[_i : _i + 32]), 512, 512) * 2 - 1)
            _pix = _pix.to(DEVICE, vlm_full.dtype)[:, None]
            _b = _pix.shape[0]
            _o = vlm_full.model(
                input_ids=_ids.input_ids.expand(_b, -1),
                attention_mask=_ids.attention_mask.expand(_b, -1),
                pixel_values=_pix,
                output_hidden_states=True,
            )
            # hidden_states[0] = 입력 임베딩, hidden_states[k] = k번째 층 출력
            _feats.append(torch.stack([_h[:, -1].float().cpu() for _h in _o.hidden_states], 1))
            _bar.update()
    probe_feats = torch.cat(_feats)

    def _ridge_r2(X, y, lam=10.0):
        Xtr, Xte, ytr, yte = X[:_n_train], X[_n_train:], y[:_n_train], y[_n_train:]
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
        Xtr = torch.cat([(Xtr - mu) / sd, torch.ones(len(Xtr), 1)], 1)
        Xte = torch.cat([(Xte - mu) / sd, torch.ones(len(Xte), 1)], 1)
        W = torch.linalg.solve(Xtr.T @ Xtr + lam * torch.eye(Xtr.shape[1]), Xtr.T @ ytr)
        return (1 - ((Xte @ W - yte) ** 2).sum() / ((yte - yte.mean(0)) ** 2).sum()).item()

    probe_layers = list(range(probe_feats.shape[1]))
    probe_r2_red = [_ridge_r2(probe_feats[:, _l], probe_red) for _l in probe_layers]
    probe_r2_blue = [_ridge_r2(probe_feats[:, _l], probe_blue) for _l in probe_layers]
    print(f"추출 {time.time() - _t0:.0f}초, 특징 shape {tuple(probe_feats.shape)}")
    return probe_layers, probe_r2_blue, probe_r2_red


@app.cell
def _(plt, probe_layers, probe_r2_blue, probe_r2_red):
    _fig, _ax = plt.subplots(figsize=(7, 3))
    _ax.plot(probe_layers, probe_r2_red, "o-", color="tab:red", ms=3, label="빨간 점 좌표 (질문 대상)")
    _ax.plot(probe_layers, probe_r2_blue, "s-", color="tab:blue", ms=3, label="파란 점 좌표")
    _ax.axvline(16, color="k", ls="--", lw=1)
    _ax.text(16.3, 0.05, "SmolVLA가 쓰는 마지막 층(16)", fontsize=8)
    _ax.set_xlabel("층 번호 (0 = 입력 임베딩)")
    _ax.set_ylabel("평가 R²")
    _ax.set_ylim(min(-0.05, min(probe_r2_red + probe_r2_blue) - 0.05), 1.05)
    _ax.legend(fontsize=8)
    _ax.grid(alpha=0.3)
    _fig.tight_layout()
    print(f"R²(빨강): 16층 {probe_r2_red[16]:.3f}, 32층 {probe_r2_red[-1]:.3f}, 최대 {max(probe_r2_red):.3f}")
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    층 0의 마지막 텍스트 토큰은 아직 이미지를 보지 않았습니다. 실제 $R^2$를 읽고 층별 차이와
    16층 이후의 변화가 얼마나 큰지 확인하세요. $R^2=1$은 완벽한 예측, 0은 평가 표적의 평균 수준, 음수는 그보다 나쁜 예측입니다.
    프로브 결과는 좌표의 선형 해독 가능성이며 최적 절단 깊이나 실제 행동 성공을 입증하지는 않습니다. 이 실험은 마지막 토큰 하나만 봤지만, SmolVLA의 행동 전문가는 각 층의
    **모든 접두부 토큰의 K/V**를 참조하므로 더 많은 정보를 쓸 수 있습니다.

    **속도 측면:** 잘라내면 텍스트 층 연산과 KV 캐시 메모리가 절반이 됩니다. 비전 인코더(1024패치 × 12층)는 그대로 남으므로
    전체 접두부 시간이 정확히 절반이 되지는 않습니다. 다음 셀에서 두 부분을 따로 재 봅니다
    (RTX 3080 Ti, bf16 측정 예: 비전 6.4 ms, 텍스트 32층 22.0 ms → 16층 11.3 ms).
    """)
    return


@app.cell
def _(
    DEVICE,
    ON_GPU,
    mo,
    probe_btn,
    render_scenes,
    resize_with_pad,
    sample_scenes,
    time,
    tok,
    torch,
    vlm_full,
):
    mo.stop(not probe_btn.value)

    def _timeit(fn, reps=5):
        fn()
        if ON_GPU:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(reps):
            fn()
        if ON_GPU:
            torch.cuda.synchronize()
        return (time.perf_counter() - t0) / reps * 1000

    _r, _b, _ = sample_scenes(1)
    _pix = (resize_with_pad(render_scenes(_r, _b), 512, 512) * 2 - 1).to(DEVICE, vlm_full.dtype)
    _ids = tok(vqa_prompt("go to the red dot"), return_tensors="pt").to(DEVICE)
    _m = vlm_full.model
    with torch.no_grad():
        _img_tokens = _m.connector(_m.vision_model(pixel_values=_pix).last_hidden_state)
        _emb = _m.text_model.get_input_embeddings()(_ids.input_ids)
        _emb[_ids.input_ids == vlm_full.config.image_token_id] = _img_tokens.reshape(-1, _img_tokens.shape[-1])
        _all_layers = _m.text_model.layers
        _t_vision = _timeit(lambda: _m.connector(_m.vision_model(pixel_values=_pix).last_hidden_state))
        _t_text = {}
        try:
            for _nl in (32, 16):
                _m.text_model.layers = _all_layers[:_nl]
                _t_text[_nl] = _timeit(lambda: _m.text_model(inputs_embeds=_emb))
        finally:
            _m.text_model.layers = _all_layers  # 32층 VLM을 원래대로 되돌립니다.
    mo.md(f"""
    배치 1, 접두부 {_emb.shape[1]}토큰 기준 (ms):

    | 비전 인코더 + 커넥터 | 텍스트 32층 | 텍스트 16층 |
    | --- | --- | --- |
    | {_t_vision:.1f} | {_t_text[32]:.1f} | {_t_text[16]:.1f} |
    """)
    return


if __name__ == "__main__":
    app.run()
