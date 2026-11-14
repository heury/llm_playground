import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import itertools
    import json
    import re
    import sys
    import time
    from pathlib import Path

    _here = str(Path(__file__).resolve().parent)
    if _here not in sys.path:
        sys.path.insert(0, _here)

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    import physicalai_common as pc
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    RUN_DIR = pc.make_run_dir("ch07-smolvla")
    return Path, RUN_DIR, device, itertools, json, mo, np, pc, plt, re, time


@app.cell(hide_code=True)
def _(device, mo):
    try:
        import lerobot

        _ok = True
    except ImportError:
        _ok = False
    mo.stop(not _ok, mo.callout(mo.md("`lerobot`가 없습니다. `uv sync --locked --extra lerobot` 후 다시 여세요."), kind="danger"))
    mo.md(rf"""
    # 7장. 자연어로 로봇 움직이기: SmolVLA

    6장의 ACT는 카메라와 관절각만 봅니다. "빨간 블록을 상자에"라는 과제는 학습 데이터에 박혀 있을 뿐 입력으로 바꿀 수 없습니다.
    **VLA**(Vision-Language-Action) 모델은 이미지·관절각과 함께 **자연어 명령**을 입력으로 받습니다.

    [SmolVLA](https://huggingface.co/lerobot/smolvla_base)는 Hugging Face가 공개한 약 4.5억 파라미터 VLA입니다.

    ```mermaid
    flowchart LR
      I[카메라 이미지] --> V[SmolVLM2<br/>비전·언어 모델]
      T["명령 문장<br/>'Pick up the red cube…'"] --> V
      S[관절각] --> V
      V -->|KV 캐시| A[Action Expert<br/>Flow Matching]
      N[무작위 noise] --> A
      A --> C[50스텝 행동 묶음]
    ```

    | 절 | 내용 | 자원 |
    | --- | --- | --- |
    | 7-1 | 사전학습 모델 `lerobot/smolvla_base` 불러오기 | 다운로드 약 1 GB, GPU 권장 |
    | 7-2 | 사전학습 모델을 우리 시뮬레이터에 그대로 넣어 보기 | GPU, 5 Episode 약 1~2분 |
    | 7-3 | 우리 시연 데이터로 파인튜닝 (`lerobot-train`) | GPU 8 GB 이상, 10,000 step 수 시간 |
    | 7-4 | 파인튜닝 모델 평가 (공개 체크포인트 가능) | 다운로드 약 720 MB |
    | 7-5 | 명령 문장을 바꾸면 행동이 달라지는가 | GPU, 수 분 |

    현재 장치 **`{device}`**. 내부 구조(VLM + Action Expert, 블록 어텐션 마스크, Flow Matching)는
    [`notebooks/lerobot/smolvla_tour.py`](../lerobot/smolvla_tour.py)에서 다룹니다.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7-1. 사전학습 SmolVLA 불러오기

    `SmolVLAPolicy.from_pretrained`는 정책 가중치와 함께 SmolVLM2 백본 설정·토크나이저를 Hub에서 받습니다.
    처음 한 번 약 1 GB를 `~/.cache/huggingface/`에 내려받습니다.
    SmolVLA는 카메라를 `observation.images.camera1`, `camera2` 이름으로 받습니다. 우리 장면의 top → camera1, wrist → camera2로 연결합니다.
    """)
    return


@app.cell
def _(mo):
    base_btn = mo.ui.run_button(label="lerobot/smolvla_base 불러오기")
    base_btn
    return (base_btn,)


@app.cell
def _(base_btn, device, mo, pc, time):
    mo.stop(not base_btn.value, mo.md("버튼을 누르면 모델을 내려받아 GPU에 올립니다."))
    _t0 = time.time()
    with mo.status.spinner("SmolVLA 불러오는 중..."):
        base_policy, base_pre, base_post = pc.load_smolvla("lerobot/smolvla_base", device)
    _cfg = base_policy.config
    _n = sum(p.numel() for p in base_policy.parameters())
    _train = sum(p.numel() for p in base_policy.parameters() if p.requires_grad)
    mo.md(f"""
    불러오기 {time.time() - _t0:.0f} 초.

    | 항목 | 값 |
    | --- | --- |
    | 전체 파라미터 | {_n / 1e6:.0f}M (학습 가능 {_train / 1e6:.0f}M) |
    | VLM 백본 | `{_cfg.vlm_model_name}` |
    | chunk_size / n_action_steps | {_cfg.chunk_size} / {_cfg.n_action_steps} |
    | Flow Matching 적분 step | {_cfg.num_steps} |
    | 이미지 입력 | {list(k for k in _cfg.input_features if 'image' in k)} → {_cfg.resize_imgs_with_padding} 패딩·리사이즈 |
    | 상태·행동 차원 | 최대 {_cfg.max_state_dim} / {_cfg.max_action_dim} (짧으면 0으로 채움) |
    """)
    return base_policy, base_post, base_pre


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7-2. 사전학습 모델을 그대로 써 보기

    `smolvla_base`는 커뮤니티가 실물 SO-100/101로 모은 데이터로 학습했습니다. 우리 시뮬레이터는 카메라 위치·조명·화질·관절 범위가
    모두 다릅니다. 같은 문장을 줘도 행동이 맞을지 확인해 봅니다. 책의 측정에서도 사전학습 모델 그대로는 거의 성공하지 못했습니다.

    SmolVLA는 행동을 만들 때마다 Flow Matching의 시작 noise를 무작위로 뽑으므로 Episode마다 noise seed를 고정해 재현합니다.
    """)
    return


@app.cell
def _(mo, pc):
    base_task = mo.ui.text(value=pc.TASK, label="명령", full_width=True)
    base_episodes = mo.ui.slider(1, 10, value=3, label="Episode")
    base_eval_btn = mo.ui.run_button(label="사전학습 모델 평가")
    mo.vstack([base_task, mo.hstack([base_episodes, base_eval_btn], justify="start")])
    return base_episodes, base_eval_btn, base_task


@app.cell
def _(
    RUN_DIR,
    base_episodes,
    base_eval_btn,
    base_policy,
    base_post,
    base_pre,
    base_task,
    device,
    mo,
    np,
    pc,
):
    mo.stop(not base_eval_btn.value, mo.md("버튼을 누르면 평가합니다."))
    _rng = np.random.default_rng(1000)
    _rows, _video = [], None
    with mo.status.progress_bar(total=base_episodes.value, title="smolvla_base 평가") as _bar:
        for _i in range(base_episodes.value):
            _cube = pc.random_cube(_rng)
            pc.seed_noise(_i)
            _r = pc.run_policy_episode(base_policy, base_pre, base_post, device, cube=_cube, task=base_task.value,
                                       image_keys=pc.SMOLVLA_KEYS, video=(_i == 0))
            if _i == 0 and _r["frames"]:
                _video = pc.save_video(_r["frames"], RUN_DIR / "smolvla_base_ep1.mp4")
            _rows.append(f"Episode {_i + 1}: 블록 ({_cube[0]:+.3f}, {_cube[1]:+.3f}) → {_r['result']}")
            _bar.update()
    mo.hstack([mo.video(src=_video, loop=True, width=420) if _video else mo.md(""), mo.md("\n\n".join(_rows))],
              align="center")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7-3. 우리 시연 데이터로 파인튜닝

    6장에서 기록한 데이터셋으로 `smolvla_base`를 이어서 학습합니다. 데이터셋의 카메라 이름(top, wrist)을
    SmolVLA가 기대하는 이름(camera1, camera2)으로 바꾸는 `--rename_map`이 핵심입니다.
    원본 책 설정은 10,000 step, batch 8이며 RTX 3060 기준 수 시간이 걸렸습니다. 학습을 건너뛰려면 7-4의 공개 체크포인트를 쓰세요.

    ```bash
    lerobot-train --policy.path=lerobot/smolvla_base \
      --dataset.repo_id=physicalai/so101_pickplace_sim --dataset.root=outputs/physicalai/datasets/so101_pickplace_sim \
      --rename_map='{"observation.images.top": "observation.images.camera1", "observation.images.wrist": "observation.images.camera2"}' \
      --policy.device=cuda --policy.push_to_hub=false --steps=10000 --batch_size=8 --trackio.enable=false \
      --output_dir=outputs/physicalai/checkpoints/train/smolvla_so101
    ```
    """)
    return


@app.cell
def _(mo, pc):
    _names = sorted(p.name for p in pc.DATASET_DIR.glob("*") if (p / "meta" / "info.json").is_file())
    ft_dataset = mo.ui.dropdown(_names or ["(없음)"], value=(_names or ["(없음)"])[0], label="데이터셋")
    ft_job = mo.ui.text(value="smolvla_so101", label="작업 이름")
    ft_steps = mo.ui.dropdown({"100 (동작 확인)": 100, "2,000": 2000, "10,000 (책 설정)": 10000},
                              value="10,000 (책 설정)", label="학습 step")
    ft_btn = mo.ui.run_button(label="SmolVLA 파인튜닝 시작")
    mo.hstack([ft_dataset, ft_job, ft_steps, ft_btn], justify="start")
    return ft_btn, ft_dataset, ft_job, ft_steps


@app.cell
def _(mo):
    get_vla_version, set_vla_version = mo.state(0)
    return get_vla_version, set_vla_version


@app.cell
def _(
    RUN_DIR,
    device,
    ft_btn,
    ft_dataset,
    ft_job,
    ft_steps,
    json,
    mo,
    pc,
    re,
    set_vla_version,
):
    mo.stop(not ft_btn.value, mo.md("버튼을 누르면 학습합니다. 오래 걸리므로 처음에는 '100 (동작 확인)'으로 명령이 도는지만 확인하세요."))
    mo.stop(device.type != "cuda", mo.callout(mo.md("CUDA GPU가 없어 학습을 실행하지 않습니다."), kind="warn"))
    _root = pc.DATASET_DIR / ft_dataset.value
    _out = pc.CHECKPOINT_DIR / "train" / ft_job.value
    mo.stop(not (_root / "meta" / "info.json").is_file(), mo.md("데이터셋이 없습니다. 6장 6-2에서 기록하세요."))
    mo.stop(_out.exists(), mo.callout(mo.md(f"`{_out}`가 이미 있습니다. 작업 이름을 바꾸세요."), kind="warn"))
    _steps = ft_steps.value
    _rename = json.dumps({"observation.images.top": "observation.images.camera1",
                          "observation.images.wrist": "observation.images.camera2"})
    _cmd = [pc.lerobot_cli("lerobot-train"), "--policy.path=lerobot/smolvla_base",
            f"--dataset.repo_id=physicalai/{ft_dataset.value}", f"--dataset.root={_root}", f"--rename_map={_rename}",
            "--policy.device=cuda", "--policy.push_to_hub=false", f"--output_dir={_out}", f"--job_name={ft_job.value}",
            f"--steps={_steps}", "--batch_size=8", f"--save_freq={max(_steps // 4, 50)}",
            f"--log_freq={max(_steps // 40, 10)}", "--trackio.enable=false"]

    def _on_line(line):
        _m = re.search(r"step:(\S+) .*?loss:([\d.eE+-]+)", line)
        if _m:
            mo.output.replace(mo.md(f"학습 중... step {_m.group(1)}/{_steps:,}, loss {float(_m.group(2)):.4f}"))

    _log = RUN_DIR / f"train_{ft_job.value}.log"
    mo.output.replace(mo.md("학습 준비 중..."))
    _code = pc.stream_command(_cmd, _log, cwd=pc.REPO_ROOT, on_line=_on_line)
    mo.stop(_code != 0, mo.callout(mo.md(f"학습 실패 (exit {_code}). 로그 `{_log}`"), kind="danger"))
    set_vla_version(lambda v: v + 1)
    mo.md(f"학습 완료: `{_out / 'checkpoints/last/pretrained_model'}`")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7-4. 파인튜닝한 SmolVLA 평가

    공개 체크포인트 `smolvla_so101_v2_10k`는 원본 책 저자가 시연 데이터로 10,000 step 파인튜닝한 모델입니다(약 720 MB).
    평가 루프는 6장과 같고, 카메라 이름과 명령 문장만 다릅니다.
    """)
    return


@app.cell
def _(mo, pc):
    vla_dl_btn = mo.ui.run_button(label=f"공개 SmolVLA 체크포인트 내려받기 ({pc.RELEASE_FILES['smolvla_so101_v2_10k'][2]})")
    vla_dl_btn
    return (vla_dl_btn,)


@app.cell
def _(mo, pc, vla_dl_btn):
    vla_release_dir = pc.CHECKPOINT_DIR / "smolvla_so101_v2_10k"
    if vla_dl_btn.value:
        with mo.status.spinner("내려받는 중..."):
            pc.download_release("smolvla_so101_v2_10k")
    return (vla_release_dir,)


@app.cell
def _(get_vla_version, mo, pc, vla_release_dir):
    get_vla_version()
    _cands = {"공개: smolvla_so101_v2_10k": str(p) for p in pc.find_pretrained_dirs(vla_release_dir)}
    for _p in sorted((pc.CHECKPOINT_DIR / "train").glob("*/checkpoints/last/pretrained_model")):
        if '"type": "smolvla"' in (_p / "config.json").read_text():
            _cands[f"내 학습: {_p.parts[-4]}"] = str(_p)
    _empty = {"(체크포인트 없음)": ""}
    vla_ckpt = mo.ui.dropdown(_cands or _empty, value=next(iter(_cands or _empty)), label="체크포인트")
    vla_episodes = mo.ui.slider(5, 30, value=10, step=5, label="평가 Episode")
    vla_eval_btn = mo.ui.run_button(label="평가")
    mo.hstack([vla_ckpt, vla_episodes, vla_eval_btn], justify="start")
    return vla_ckpt, vla_episodes, vla_eval_btn


@app.cell
def _(Path, device, mo, pc, vla_ckpt, vla_eval_btn):
    mo.stop(not vla_eval_btn.value, mo.md("체크포인트를 고르고 평가 버튼을 누르세요."))
    mo.stop(not vla_ckpt.value or not (Path(vla_ckpt.value) / "model.safetensors").is_file(),
            mo.md("체크포인트가 없습니다. 위에서 내려받거나 7-3에서 학습하세요."))
    with mo.status.spinner("SmolVLA 불러오는 중..."):
        vla_policy, vla_pre, vla_post = pc.load_smolvla(vla_ckpt.value, device)
    vla_loaded_from = vla_ckpt.value
    return vla_loaded_from, vla_policy, vla_post, vla_pre


@app.cell
def _(
    RUN_DIR,
    device,
    json,
    mo,
    np,
    pc,
    time,
    vla_episodes,
    vla_loaded_from,
    vla_policy,
    vla_post,
    vla_pre,
):
    _rng = np.random.default_rng(1000)
    _records, _video, _t0 = [], None, time.time()
    with mo.status.progress_bar(total=vla_episodes.value, title="SmolVLA 평가") as _bar:
        for _i in range(vla_episodes.value):
            _cube = pc.random_cube(_rng)
            pc.seed_noise(_i)
            _r = pc.run_policy_episode(vla_policy, vla_pre, vla_post, device, cube=_cube,
                                       image_keys=pc.SMOLVLA_KEYS, video=(_i == 0))
            if _i == 0 and _r["frames"]:
                _video = pc.save_video(_r["frames"], RUN_DIR / "smolvla_ft_ep1.mp4")
            _records.append({"cube_xy": list(_cube), "noise_seed": _i, "success": _r["success"], "t": _r["t"]})
            _bar.update()
    _k = sum(r["success"] for r in _records)
    _lo, _hi = pc.wilson(_k, len(_records))
    (RUN_DIR / "smolvla_eval.json").write_text(json.dumps(
        {"checkpoint": vla_loaded_from, "seed": 1000, "episodes": _records, "elapsed_seconds": time.time() - _t0}, indent=2))
    mo.hstack([mo.video(src=_video, loop=True, width=420) if _video else mo.md(""),
               mo.md(f"### 성공률 {_k}/{len(_records)} = {100 * _k / len(_records):.0f}%\n\n"
                     f"95% 신뢰구간 {100 * _lo:.0f}~{100 * _hi:.0f}%, {time.time() - _t0:.0f} 초")], align="center")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7-5. 명령 문장을 바꾸면 행동이 달라지는가

    같은 장면(블록 (0.24, 0))에서 문장만 바꿔 4초 동안 실행하고, 손끝의 최종 위치를 비교합니다.
    성공 여부가 아니라 **언어가 행동에 영향을 주는가**를 보는 실험입니다.

    SmolVLA의 행동에는 noise도 영향을 주므로 두 가지 차이를 나눠 잽니다.

    - **문장 효과**: noise seed를 고정하고 문장만 바꿨을 때 최종 위치가 서로 얼마나 다른가
    - **noise 효과**: 문장을 고정하고 noise seed만 바꿨을 때 최종 위치가 서로 얼마나 다른가

    문장 효과가 noise 효과보다 뚜렷하게 커야 "언어가 행동을 바꾼다"고 말할 수 있습니다.
    파인튜닝 데이터의 문장이 하나뿐이었다면 결과가 어떻게 나올지 먼저 예측해 보세요.
    """)
    return


@app.cell
def _(mo):
    tv_tasks = mo.ui.text_area(value="\n".join([
        "Pick up the red cube and put it in the blue box.",
        "Move the cube to the left.",
        "Do nothing.",
        "Raise your arm up high.",
    ]), label="비교할 문장 (한 줄에 하나)", full_width=True, rows=4)
    tv_seeds = mo.ui.slider(2, 5, value=3, label="noise seed 개수")
    tv_btn = mo.ui.run_button(label="문장 비교 실행 (7-4에서 불러온 모델)")
    mo.vstack([tv_tasks, mo.hstack([tv_seeds, tv_btn], justify="start")])
    return tv_btn, tv_seeds, tv_tasks


@app.cell
def _(
    RUN_DIR,
    device,
    itertools,
    json,
    mo,
    np,
    pc,
    plt,
    tv_btn,
    tv_seeds,
    tv_tasks,
    vla_loaded_from,
    vla_policy,
    vla_post,
    vla_pre,
):
    mo.stop(not tv_btn.value, mo.md("7-4에서 모델을 불러온 뒤 버튼을 누르세요."))
    _tasks = [t.strip() for t in tv_tasks.value.splitlines() if t.strip()]
    _seeds = list(range(tv_seeds.value))

    def _final_pos(task, seed):
        pc.seed_noise(seed)
        _r = pc.run_policy_episode(vla_policy, vla_pre, vla_post, device, cube=(0.24, 0.0), task=task,
                                   image_keys=pc.SMOLVLA_KEYS, max_seconds=4.0, stop_on_success=False)
        return _r["final_site"]

    _final = {}
    with mo.status.progress_bar(total=len(_tasks) * len(_seeds) + 1, title="문장 × noise seed") as _bar:
        _repeat = np.linalg.norm(_final_pos(_tasks[0], 0) - _final_pos(_tasks[0], 0))   # 재현성 확인
        _bar.update()
        for _s in _seeds:
            for _t in _tasks:
                _final[(_s, _t)] = _final_pos(_t, _s)
                _bar.update()

    def _pairwise(points):
        return float(np.mean([np.linalg.norm(a - b) for a, b in itertools.combinations(points, 2)]))

    _by_sentence = np.mean([_pairwise([_final[(s, t)] for t in _tasks]) for s in _seeds])
    _by_noise = np.mean([_pairwise([_final[(s, t)] for s in _seeds]) for t in _tasks])
    (RUN_DIR / "task_variation.json").write_text(json.dumps({
        "checkpoint": vla_loaded_from, "tasks": _tasks, "seeds": _seeds, "repeat_diff_m": float(_repeat),
        "final": {f"{s}|{t}": v.tolist() for (s, t), v in _final.items()},
        "sentence_effect_m": float(_by_sentence), "noise_effect_m": float(_by_noise)}, indent=2, ensure_ascii=False))
    _fig, _ax = plt.subplots(figsize=(5, 4))
    for _j, _t in enumerate(_tasks):
        _pts = np.array([_final[(s, _t)] for s in _seeds])
        _ax.scatter(_pts[:, 0], _pts[:, 1], label=f"task {_j + 1}", s=40)
    _ax.scatter([0.24], [0.0], marker="s", color="red", label="cube")
    _ax.set(xlabel="x (m)", ylabel="y (m)", title="Gripper position after 4 s", aspect="equal")
    _ax.legend(fontsize=7)
    _ax.grid(alpha=0.3)
    _fig.tight_layout()
    _rows = "\n".join(f"| task {j + 1} | {t} |" for j, t in enumerate(_tasks))
    mo.hstack([_fig, mo.md(f"""
    | 번호 | 문장 |
    | --- | --- |
    {_rows}

    - 같은 문장·같은 seed 두 번 실행한 차이: **{_repeat * 100:.2f} cm** (0이면 재현됨)
    - 문장만 바꿨을 때 최종 위치 차이 평균: **{_by_sentence * 100:.1f} cm**
    - noise만 바꿨을 때 최종 위치 차이 평균: **{_by_noise * 100:.1f} cm**
    """)], align="center")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 정리와 다음 장

    - VLA는 문장을 입력으로 받지만, **파인튜닝 데이터의 문장이 하나뿐이면 문장을 무시하는 법을 배웁니다.**
      이미지만 보고도 같은 행동을 하면 손실이 줄기 때문입니다.
    - 사전학습 모델의 능력은 학습 환경(카메라·로봇·조명)과 우리 환경이 다르면 거의 이어지지 않습니다. 결국 우리 환경의 시연으로 파인튜닝해야 합니다.

    **확인 문제**

    1. 문장이 행동을 바꾸게 하려면 시연 데이터를 어떻게 모아야 할까요? (8장 8-1에서 직접 해 봅니다)
    2. noise seed를 고정하지 않으면 7-5의 비교에 어떤 문제가 생기나요?

    다음 [8장](ch08_generalization.py)에서는 블록 세 개 중 문장이 가리키는 색을 집게 하고, 처음 보는 환경에서 정책을 시험합니다.
    """)
    return


if __name__ == "__main__":
    app.run()
