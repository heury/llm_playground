import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import json
    import re
    import shutil
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
    RUN_DIR = pc.make_run_dir("ch06-act")
    return Path, RUN_DIR, device, json, mo, np, pc, plt, re, shutil, time, torch


@app.cell(hide_code=True)
def _(device, mo):
    try:
        import lerobot

        _lerobot = f"LeRobot {lerobot.__version__}"
    except ImportError:
        _lerobot = None
    mo.stop(_lerobot is None, mo.callout(mo.md(
        "`lerobot`가 설치되어 있지 않습니다. `notebooks/physicalai/`에서 `uv sync --locked --extra lerobot` 후 "
        "marimo를 다시 시작하세요."), kind="danger"))
    mo.md(rf"""
    # 6장. 시연을 따라 하는 정책: LeRobot ACT

    5장에서 보상만으로는 긴 과제(집기 → 들기 → 옮기기 → 놓기)를 배우기 어렵다는 것을 봤습니다.
    **모방학습(Imitation Learning)** 은 잘하는 사람(여기서는 3장의 Rule 기반 Pick and Place)의 시연을 모아
    "이 장면에서는 이 행동"을 지도학습으로 배웁니다.

    | 절 | 내용 | 필요 자원 |
    | --- | --- | --- |
    | 6-1 | LeRobot 데이터셋 형식 (Hub 공개 데이터셋 메타데이터) | 인터넷 |
    | 6-2 | 시뮬레이터에서 시연 데이터셋 기록 | CPU, 50 Episode 약 2분 |
    | 6-3 | 기록한 데이터셋 뜯어보기 | CPU |
    | 6-4 | `lerobot-train`으로 ACT 학습 | NVIDIA GPU, 20,000 step 약 15~30분 |
    | 6-5 | 학습한 ACT 평가 (또는 공개 체크포인트) | GPU 권장, 10 Episode 약 30초 |
    | 6-6 | Action chunk 실행 길이와 Temporal Ensembling 비교 | GPU 권장 |

    현재 장치 **`{device}`**, {_lerobot}. GPU가 없으면 6-4를 건너뛰고 6-5에서 공개 체크포인트(약 190 MB)를 내려받아 평가하세요.
    ACT 신경망 내부(인코더·디코더·CVAE)는 [`notebooks/lerobot/act_tour.py`](../lerobot/act_tour.py)에서 자세히 다룹니다.

    데이터셋은 `outputs/physicalai/datasets/`, 학습 결과와 공개 체크포인트는 `outputs/physicalai/checkpoints/`에 두어
    7·8장에서 다시 씁니다.
    """)
    return


@app.cell
def _():
    from lerobot.datasets import LeRobotDataset, LeRobotDatasetMetadata
    return LeRobotDataset, LeRobotDatasetMetadata


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6-0. 모방학습과 ACT 한눈에 보기

    ```mermaid
    flowchart LR
      S[시연자<br/>3장 Rule 기반] -->|관측·행동 기록 20 Hz| D[(LeRobotDataset)]
      D -->|lerobot-train| P[ACT 정책]
      O[관측: 관절각 6 + top·wrist 이미지] --> P
      P -->|50스텝 행동 묶음| R[로봇 ctrl 6개]
      R --> O
    ```

    - **관측**: 관절각 6개(`observation.state`) + 위·손목 카메라 이미지 2장(240×320).
    - **행동**: 6개 구동기의 목표 각도(`action`). 시연 당시 컨트롤러가 그 순간 명령하던 값입니다.
    - **ACT**(Action Chunking with Transformers): 행동 하나가 아니라 **앞으로 50스텝(2.5초)의 행동 묶음**을 한 번에 예측합니다.
      한 스텝씩 고르면 작은 오차가 쌓여 시연 분포 밖으로 벗어나기 쉬운데(compounding error), 묶음으로 예측하면 그 횟수가 줄어듭니다.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6-1. LeRobot 데이터셋 형식

    LeRobot 데이터셋은 `meta/info.json`(특징·fps·개수), Parquet 표(관절각·행동·시간), MP4 영상(카메라)으로 이루어집니다.
    Hugging Face Hub의 공개 SO-101 데이터셋 메타데이터만 읽어 봅니다(영상은 내려받지 않습니다).
    """)
    return


@app.cell
def _(mo):
    hub_btn = mo.ui.run_button(label="Hub 공개 데이터셋 메타데이터 읽기")
    hub_repo = mo.ui.text(value="lerobot/svla_so101_pickplace", label="repo_id", full_width=False)
    mo.hstack([hub_repo, hub_btn], justify="start")
    return hub_btn, hub_repo


@app.cell
def _(LeRobotDatasetMetadata, hub_btn, hub_repo, mo):
    mo.stop(not hub_btn.value, mo.md("버튼을 누르면 Hub에서 메타데이터(수 MB)를 `~/.cache/huggingface/lerobot/`에 받습니다."))
    _meta = LeRobotDatasetMetadata(hub_repo.value)
    _rows = ["| feature | dtype | shape |", "| --- | --- | --- |"] + [
        f"| `{k}` | {v['dtype']} | {tuple(v['shape'])} |" for k, v in _meta.features.items()]
    mo.md(f"""
    - robot_type `{_meta.robot_type}`, fps {_meta.fps}, Episode {_meta.total_episodes}개, 프레임 {_meta.total_frames}개
      (Episode당 평균 {_meta.total_frames / _meta.total_episodes:.0f})
    - 카메라 `{_meta.camera_keys}`
    - task 예: {list(_meta.tasks.index)[:3]}

    {chr(10).join(_rows)}
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6-2. 시뮬레이터에서 시연 데이터셋 만들기

    사람이 조종 장치(teleoperation)로 시연하는 대신 3장의 Rule 기반 Pick and Place가 시연합니다.
    `RecordingRobot`은 `SO101Sim`과 같지만 시뮬레이션이 1/20초 흐를 때마다 관측과 행동을 `dataset.add_frame()`으로 기록합니다.
    실패한 시연(블록을 놓친 경우)은 버립니다. 나쁜 시연까지 따라 하지 않도록 하기 위해서입니다.
    """)
    return


@app.cell
def _(mo, pc):
    import inspect as _inspect

    mo.accordion({"RecordingRobot 소스 보기": mo.md("```python\n" + _inspect.getsource(pc.RecordingRobot) + "```"),
                  "scripted_pick_and_place 소스 보기": mo.md("```python\n" + _inspect.getsource(pc.scripted_pick_and_place) + "```")})
    return


@app.cell
def _(mo):
    get_ds_version, set_ds_version = mo.state(0)
    return get_ds_version, set_ds_version


@app.cell
def _(mo):
    rec_name = mo.ui.text(value="so101_pickplace_sim", label="데이터셋 이름")
    rec_episodes = mo.ui.slider(5, 150, value=50, step=5, label="Episode 수")
    rec_seed = mo.ui.number(value=0, label="seed")
    rec_overwrite = mo.ui.checkbox(value=False, label="같은 이름이 있으면 지우고 새로 기록")
    rec_btn = mo.ui.run_button(label="시연 기록")
    mo.vstack([mo.hstack([rec_name, rec_episodes, rec_seed], justify="start"),
               mo.hstack([rec_overwrite, rec_btn], justify="start")])
    return rec_btn, rec_episodes, rec_name, rec_overwrite, rec_seed


@app.cell
def _(
    LeRobotDataset,
    RUN_DIR,
    json,
    mo,
    np,
    pc,
    rec_btn,
    rec_episodes,
    rec_name,
    rec_overwrite,
    rec_seed,
    set_ds_version,
    shutil,
    time,
):
    mo.stop(not rec_btn.value, mo.md("버튼을 누르면 기록합니다. 이미 기록한 데이터셋이 있으면 6-3부터 진행해도 됩니다."))
    _root = pc.DATASET_DIR / rec_name.value
    if _root.exists():
        mo.stop(not rec_overwrite.value, mo.callout(mo.md(
            f"`{_root}`가 이미 있습니다. 지우고 새로 기록하려면 체크박스를 켜고, 남겨 두려면 이름을 바꾸세요."), kind="warn"))
        shutil.rmtree(_root)
    _ds = LeRobotDataset.create(repo_id=f"physicalai/{rec_name.value}", fps=pc.FPS, features=pc.DATASET_FEATURES,
                                root=_root, robot_type="so101_sim", use_videos=True)
    _rng = np.random.default_rng(int(rec_seed.value))
    _log, _t0 = [], time.time()
    with mo.status.progress_bar(total=rec_episodes.value, title="시연 기록") as _bar:
        for _ep in range(rec_episodes.value):
            _cube = pc.random_cube(_rng)
            _robot = pc.RecordingRobot(dataset=_ds, cube_pos=_cube, box_pos=pc.BOX_XY)
            _ok = pc.scripted_pick_and_place(_robot, demo=True)
            _n = _robot.n_recorded
            _robot.close()
            if _ok:
                _ds.save_episode()
            else:
                _ds.clear_episode_buffer()
            _log.append({"episode": _ep + 1, "cube_xy": list(_cube), "frames": _n, "saved": bool(_ok)})
            _bar.update()
    _ds.finalize()
    set_ds_version(lambda v: v + 1)
    (RUN_DIR / f"record_{rec_name.value}.json").write_text(json.dumps(_log, indent=2))
    _saved = sum(r["saved"] for r in _log)
    mo.md(f"완료: 시도 {len(_log)}회 중 **{_saved}개 저장**, {len(_log) - _saved}개 실패(버림), "
          f"총 프레임 {_ds.meta.total_frames}, {time.time() - _t0:.0f} 초. 위치 `{_root}`")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6-3. 기록한 데이터셋 뜯어보기

    `LeRobotDataset[i]`는 프레임 하나를 dict로 돌려줍니다. 이미지는 영상에서 디코딩해 `(C, H, W)` float 0~1로 줍니다.
    아래 그림에서 `action`(점선)이 `state`(실선)보다 **앞서 움직이는** 것을 확인하세요. 행동은 "지금 향하는 목표 각도"라서
    관절이 실제로 따라가기 전에 먼저 바뀝니다. 그리퍼(joint 5)는 블록에 막혀 명령만큼 닫히지 않습니다.
    """)
    return


@app.cell
def _(get_ds_version, mo, pc):
    get_ds_version()
    _names = sorted(p.name for p in pc.DATASET_DIR.glob("*") if (p / "meta" / "info.json").is_file())
    insp_name = mo.ui.dropdown(_names or ["(없음)"], value=(_names or ["(없음)"])[0], label="데이터셋")
    insp_episode = mo.ui.number(value=0, start=0, label="Episode 번호")
    insp_btn = mo.ui.run_button(label="열기")
    mo.hstack([insp_name, insp_episode, insp_btn], justify="start")
    return insp_btn, insp_episode, insp_name


@app.cell
def _(LeRobotDataset, RUN_DIR, insp_btn, insp_episode, insp_name, mo, np, pc, plt):
    mo.stop(not insp_btn.value, mo.md("기록한 데이터셋을 고르고 버튼을 누르세요."))
    _root = pc.DATASET_DIR / insp_name.value
    mo.stop(not (_root / "meta" / "info.json").is_file(), mo.md(f"`{_root}`에 데이터셋이 없습니다. 6-2에서 먼저 기록하세요."))
    _ds = LeRobotDataset(f"physicalai/{insp_name.value}", root=_root, episodes=[int(insp_episode.value)])
    _sample = _ds[0]
    _rows = ["| key | shape | dtype |", "| --- | --- | --- |"]
    for _k, _v in _sample.items():
        _rows.append(f"| `{_k}` | {tuple(_v.shape) if hasattr(_v, 'shape') else repr(_v)} | {getattr(_v, 'dtype', type(_v).__name__)} |")
    _states = np.stack([_ds[i]["observation.state"].numpy() for i in range(_ds.num_frames)])
    _actions = np.stack([_ds[i]["action"].numpy() for i in range(_ds.num_frames)])
    _t = np.arange(_ds.num_frames) / _ds.meta.fps
    _fig, _axes = plt.subplots(6, 1, figsize=(7, 7), sharex=True)
    for _j in range(6):
        _axes[_j].plot(_t, _states[:, _j], label="state")
        _axes[_j].plot(_t, _actions[:, _j], "--", label="action")
        _axes[_j].set_ylabel(pc.JOINT_NAMES[_j], fontsize=7)
        _axes[_j].grid(alpha=0.3)
    _axes[0].legend(fontsize=7, loc="upper right")
    _axes[-1].set_xlabel("time (s)")
    _fig.tight_layout()
    _fig.savefig(RUN_DIR / f"episode{int(insp_episode.value)}_state_action.png", dpi=120)
    _imgs = [mo.image((_ds[i][k].permute(1, 2, 0).numpy() * 255).astype(np.uint8), caption=f"{k.split('.')[-1]} t={i / _ds.meta.fps:.1f}s")
             for i in (0, _ds.num_frames // 2) for k in _ds.meta.camera_keys]
    mo.vstack([mo.md(f"전체 Episode {_ds.meta.total_episodes}개, 프레임 {_ds.meta.total_frames}개 / 이 Episode {_ds.num_frames}프레임"),
               mo.hstack([mo.md("\n".join(_rows)), _fig], align="start"), mo.hstack(_imgs, wrap=True)])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6-4. ACT 학습: `lerobot-train`

    LeRobot의 학습 명령을 그대로 실행합니다. 원본 책의 설정은 20,000 step, batch 16, `chunk_size = n_action_steps = 50`입니다.
    RTX 3080 Ti에서 step당 약 0.05초(20,000 step 약 15~20분)이며 GPU 메모리는 2 GB 미만입니다. 처음에는 ResNet18 가중치(45 MB)를 내려받습니다.
    로그는 실행 폴더의 `train_<작업 이름>.log`에 저장되고, 이 셀이 끝날 때까지 노트북의 다른 셀은 기다립니다.

    학습 결과는 `outputs/physicalai/checkpoints/train/<작업 이름>/checkpoints/last/pretrained_model`에 생깁니다.
    같은 작업 이름의 폴더가 있으면 LeRobot이 덮어쓰지 않으므로 이름을 바꾸세요.
    """)
    return


@app.cell
def _(get_ds_version, mo, pc):
    get_ds_version()
    _names = sorted(p.name for p in pc.DATASET_DIR.glob("*") if (p / "meta" / "info.json").is_file())
    train_dataset = mo.ui.dropdown(_names or ["(없음)"], value=(_names or ["(없음)"])[0], label="학습 데이터셋")
    train_job = mo.ui.text(value="act_so101", label="작업 이름")
    train_steps = mo.ui.dropdown({"200 (동작 확인)": 200, "2,000": 2000, "5,000": 5000, "20,000 (책 설정)": 20000},
                                 value="20,000 (책 설정)", label="학습 step")
    train_btn = mo.ui.run_button(label="ACT 학습 시작")
    mo.hstack([train_dataset, train_job, train_steps, train_btn], justify="start")
    return train_btn, train_dataset, train_job, train_steps


@app.cell
def _(mo):
    get_ckpt_version, set_ckpt_version = mo.state(0)
    return get_ckpt_version, set_ckpt_version


@app.cell
def _(
    RUN_DIR,
    device,
    mo,
    np,
    pc,
    plt,
    re,
    set_ckpt_version,
    train_btn,
    train_dataset,
    train_job,
    train_steps,
):
    mo.stop(not train_btn.value, mo.md("버튼을 누르면 학습을 시작합니다. GPU가 없다면 건너뛰고 6-5로 가세요."))
    mo.stop(device.type != "cuda", mo.callout(mo.md("CUDA GPU가 없어 학습을 실행하지 않습니다."), kind="warn"))
    _root = pc.DATASET_DIR / train_dataset.value
    _out = pc.CHECKPOINT_DIR / "train" / train_job.value
    mo.stop(not (_root / "meta" / "info.json").is_file(), mo.md("학습할 데이터셋이 없습니다. 6-2에서 먼저 기록하세요."))
    mo.stop(_out.exists(), mo.callout(mo.md(f"`{_out}`가 이미 있습니다. 작업 이름을 바꾸세요."), kind="warn"))
    _steps = train_steps.value
    _cmd = [pc.lerobot_cli("lerobot-train"),
            f"--dataset.repo_id=physicalai/{train_dataset.value}", f"--dataset.root={_root}",
            "--policy.type=act", "--policy.chunk_size=50", "--policy.n_action_steps=50",
            "--policy.device=cuda", "--policy.push_to_hub=false",
            f"--output_dir={_out}", f"--job_name={train_job.value}", f"--steps={_steps}", "--batch_size=16",
            f"--save_freq={max(_steps // 4, 100)}", f"--log_freq={max(_steps // 40, 10)}", "--trackio.enable=false"]
    _losses = []

    def _on_line(line):
        _m = re.search(r"step:(\S+) .*?loss:([\d.eE+-]+)", line)
        if _m:
            _step = float(_m.group(1).replace("K", "e3").replace("M", "e6"))
            _losses.append((_step, float(_m.group(2))))
            mo.output.replace(mo.md(f"학습 중... step {_step:,.0f}/{_steps:,}, loss {_losses[-1][1]:.3f}"))

    _log = RUN_DIR / f"train_{train_job.value}.log"
    mo.output.replace(mo.md("학습 시작 (처음 몇십 초는 데이터·모델 준비)..."))
    _code = pc.stream_command(_cmd, _log, cwd=pc.REPO_ROOT, on_line=_on_line)
    mo.stop(_code != 0, mo.callout(mo.md(f"학습 실패 (exit {_code}). 로그: `{_log}`"), kind="danger"))
    set_ckpt_version(lambda v: v + 1)
    _fig, _ax = plt.subplots(figsize=(5, 3))
    if _losses:
        _l = np.array(_losses)
        _ax.semilogy(_l[:, 0], _l[:, 1])
    _ax.set(xlabel="step", ylabel="loss (L1 + KL)", title=f"ACT training: {train_job.value}")
    _ax.grid(alpha=0.3)
    _fig.tight_layout()
    _fig.savefig(RUN_DIR / f"train_{train_job.value}_loss.png", dpi=120)
    mo.vstack([mo.md(f"학습 완료. 체크포인트 `{_out / 'checkpoints/last/pretrained_model'}`, 로그 `{_log}`"), _fig])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6-5. 학습한 ACT로 Pick and Place

    평가 루프는 1장의 세 줄과 같습니다. 1/20초마다 관측(관절각 + 카메라 2장)을 찍어 정책에 넣고, 나온 목표 각도 6개를
    `data.ctrl`에 씁니다. 매번 새 관측을 쓰는 **닫힌 고리(closed loop)** 입니다.
    12초 안에 블록이 상자 안(`is_in_box`)에 처음 들어오면 성공입니다.

    ```python
    obs = preprocess(make_observation(robot, cams, image_keys, task, device))   # 정규화 포함
    action = postprocess(policy.select_action(obs))                             # 역정규화
    robot.data.ctrl[:6] = action
    robot.step(25)                                                              # 0.05초
    ```

    공개 체크포인트 `act_so101_v2_10k`는 원본 책 저자가 시연 117개 데이터셋으로 30,000 step 학습한 ACT입니다.
    이 노트북 환경(LeRobot 0.7, MuJoCo 4)에서 처음 10개 위치로 시험했을 때 4/10 성공했습니다. 책의 측정 환경과 버전이 달라
    수치가 다를 수 있습니다.
    """)
    return


@app.cell
def _(mo, pc):
    _tag, _file, _size = pc.RELEASE_FILES["act_so101_v2_10k"]
    act_dl_btn = mo.ui.run_button(label=f"공개 ACT 체크포인트 내려받기 ({_size})")
    act_dl_btn
    return (act_dl_btn,)


@app.cell
def _(act_dl_btn, mo, pc):
    act_release_dir = pc.CHECKPOINT_DIR / "act_so101_v2_10k"
    if act_dl_btn.value:
        with mo.status.spinner("내려받는 중..."):
            pc.download_release("act_so101_v2_10k")
    return (act_release_dir,)


@app.cell
def _(act_release_dir, get_ckpt_version, mo, pc):
    get_ckpt_version()
    _cands = {}
    for _p in pc.find_pretrained_dirs(act_release_dir):
        _cands["공개: act_so101_v2_10k"] = str(_p)
    for _p in sorted((pc.CHECKPOINT_DIR / "train").glob("*/checkpoints/last/pretrained_model")):
        if '"type": "act"' in (_p / "config.json").read_text():
            _cands[f"내 학습: {_p.parts[-4]}"] = str(_p)
    act_ckpt = mo.ui.dropdown(_cands or {"(체크포인트 없음)": ""}, value=next(iter(_cands or {"(체크포인트 없음)": ""})),
                              label="체크포인트")
    act_custom = mo.ui.text(value="", placeholder="직접 경로 입력 (선택)", label="또는 경로", full_width=True)
    act_episodes = mo.ui.slider(5, 50, value=10, step=5, label="평가 Episode")
    act_seed = mo.ui.number(value=1000, label="블록 위치 seed")
    act_eval_btn = mo.ui.run_button(label="평가")
    mo.vstack([mo.hstack([act_ckpt, act_episodes, act_seed, act_eval_btn], justify="start"), act_custom])
    return act_ckpt, act_custom, act_episodes, act_eval_btn, act_seed


@app.cell
def _(
    Path,
    RUN_DIR,
    act_ckpt,
    act_custom,
    act_episodes,
    act_eval_btn,
    act_seed,
    device,
    json,
    mo,
    np,
    pc,
    time,
    torch,
):
    mo.stop(not act_eval_btn.value, mo.md("체크포인트를 고르고 평가 버튼을 누르세요."))
    act_ckpt_path = act_custom.value.strip() or act_ckpt.value
    mo.stop(not act_ckpt_path or not (Path(act_ckpt_path) / "model.safetensors").is_file(),
            mo.md("체크포인트가 없습니다. 위에서 내려받거나 6-4에서 학습하세요."))
    _policy, _pre, _post = pc.load_act(act_ckpt_path, device)
    torch.manual_seed(int(act_seed.value))
    _rng = np.random.default_rng(int(act_seed.value))
    act_eval_records, _video, _t0 = [], None, time.time()
    with mo.status.progress_bar(total=act_episodes.value, title="ACT 평가") as _bar:
        for _i in range(act_episodes.value):
            _cube = pc.random_cube(_rng)
            _r = pc.run_policy_episode(_policy, _pre, _post, device, cube=_cube, video=(_i == 0))
            if _i == 0:
                _video = pc.save_video(_r["frames"], RUN_DIR / "act_eval_ep1.mp4") if _r["frames"] else None
            act_eval_records.append({"episode": _i + 1, "cube_xy": list(_cube), "success": _r["success"], "t": _r["t"]})
            _bar.update()
    _k = sum(r["success"] for r in act_eval_records)
    _lo, _hi = pc.wilson(_k, len(act_eval_records))
    (RUN_DIR / "act_eval.json").write_text(json.dumps({
        "checkpoint": act_ckpt_path, "seed": int(act_seed.value), "device": str(device),
        "successes": _k, "episodes": act_eval_records, "elapsed_seconds": time.time() - _t0}, indent=2))
    act_eval_summary = (f"### 성공률 {_k}/{len(act_eval_records)} = {100 * _k / len(act_eval_records):.0f}% "
                        f"(95% 신뢰구간 {100 * _lo:.0f}~{100 * _hi:.0f}%)")
    act_eval_video = _video
    return act_eval_records, act_eval_summary, act_eval_video


@app.cell
def _(act_eval_records, act_eval_summary, act_eval_video, mo, np, plt):
    _xy = np.array([r["cube_xy"] for r in act_eval_records])
    _ok = np.array([r["success"] for r in act_eval_records])
    _fig, _ax = plt.subplots(figsize=(4.5, 3.8))
    _ax.scatter(_xy[_ok, 0], _xy[_ok, 1], c="tab:green", label="success")
    _ax.scatter(_xy[~_ok, 0], _xy[~_ok, 1], c="tab:red", marker="x", label="fail")
    _ax.add_patch(plt.Rectangle((0.0, 0.17), 0.1, 0.1, fill=False, color="tab:blue"))
    _ax.set(aspect="equal", xlabel="x (m)", ylabel="y (m)", title="ACT outcomes by initial cube position")
    _ax.legend(fontsize=8)
    _ax.grid(alpha=0.3)
    _fig.tight_layout()
    _times = [r["t"] for r in act_eval_records if r["success"]]
    mo.vstack([mo.md(act_eval_summary + (f"\n\n성공까지 평균 {np.mean(_times):.1f}초" if _times else "")),
               mo.hstack([mo.video(src=act_eval_video, loop=True, width=420) if act_eval_video else mo.md(""), _fig],
                         align="center")])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    10회 평가의 신뢰구간은 매우 넓습니다. 두 설정을 비교할 때는 같은 블록 위치(같은 seed)에서 충분히 많이 평가해야 합니다.

    ## 6-6. Action chunk를 얼마나 실행할까

    ACT는 50스텝 묶음을 예측하지만 그중 몇 스텝을 실행하고 다시 추론할지는 실행 시점에 고를 수 있습니다.

    | 설정 | 동작 | 특징 |
    | --- | --- | --- |
    | `n_action_steps = 50` | 50스텝(2.5초) 실행 후 재추론 | 빠르지만 그동안 관측 변화에 반응하지 못함 |
    | `n_action_steps = 10` | 10스텝마다 재추론 | 반응성 ↑, 묶음 경계에서 행동이 끊길 수 있음 |
    | Temporal Ensembling (0.01) | 매 스텝 재추론, 겹치는 예측을 지수 가중 평균 | 부드럽지만 추론 50배 |

    같은 체크포인트·같은 블록 위치에서 설정만 바꿔 비교합니다(설정당 10 Episode면 약 1~3분).
    """)
    return


@app.cell
def _(mo):
    chunk_modes = mo.ui.multiselect(["n_action_steps=50", "n_action_steps=25", "n_action_steps=10", "ensemble=0.01"],
                                    value=["n_action_steps=50", "n_action_steps=10", "ensemble=0.01"], label="비교할 설정")
    chunk_episodes = mo.ui.slider(5, 30, value=10, step=5, label="설정당 Episode")
    chunk_btn = mo.ui.run_button(label="비교 실행")
    mo.hstack([chunk_modes, chunk_episodes, chunk_btn], justify="start")
    return chunk_btn, chunk_episodes, chunk_modes


@app.cell
def _(
    Path,
    RUN_DIR,
    act_ckpt,
    act_custom,
    chunk_btn,
    chunk_episodes,
    chunk_modes,
    device,
    json,
    mo,
    np,
    pc,
    torch,
):
    mo.stop(not chunk_btn.value, mo.md("6-5에서 고른 체크포인트로 비교합니다."))
    _ckpt = act_custom.value.strip() or act_ckpt.value
    mo.stop(not _ckpt or not (Path(_ckpt) / "model.safetensors").is_file(), mo.md("체크포인트가 없습니다."))
    _rows = ["| 설정 | 성공 | 95% 신뢰구간 | 성공까지 평균 시간 |", "| --- | ---: | --- | ---: |"]
    _all = {}
    with mo.status.progress_bar(total=len(chunk_modes.value) * chunk_episodes.value, title="chunk 설정 비교") as _bar:
        for _mode in chunk_modes.value:
            _key, _val = _mode.split("=")
            _kw = {"ensemble": float(_val)} if _key == "ensemble" else {"n_action_steps": int(_val)}
            _policy, _pre, _post = pc.load_act(_ckpt, device, **_kw)
            _rng = np.random.default_rng(1000)
            torch.manual_seed(1000)
            _res = []
            for _ in range(chunk_episodes.value):
                _r = pc.run_policy_episode(_policy, _pre, _post, device, cube=pc.random_cube(_rng))
                _res.append({"success": _r["success"], "t": _r["t"]})
                _bar.update()
            _k = sum(r["success"] for r in _res)
            _lo, _hi = pc.wilson(_k, len(_res))
            _ts = [r["t"] for r in _res if r["success"]]
            _rows.append(f"| `{_mode}` | {_k}/{len(_res)} | {100 * _lo:.0f}~{100 * _hi:.0f}% | "
                         f"{np.mean(_ts):.1f} s |" if _ts else f"| `{_mode}` | {_k}/{len(_res)} | {100 * _lo:.0f}~{100 * _hi:.0f}% | - |")
            _all[_mode] = _res
    (RUN_DIR / "act_chunk_comparison.json").write_text(json.dumps({"checkpoint": _ckpt, "results": _all}, indent=2))
    mo.md("\n".join(_rows))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 정리와 다음 장

    - 모방학습은 **좋은 시연 데이터**가 성능을 좌우합니다. 시연 수, 블록 위치의 다양성, 실패 시연 제거가 모두 중요합니다.
    - ACT는 행동 묶음을 예측해 오차 누적을 줄입니다. 실행 길이는 반응성과 매끄러움 사이의 선택입니다.
    - 학습한 정책은 시연에서 본 범위(블록 위치·조명·카메라) 안에서만 믿을 수 있습니다. 8장에서 그 경계를 시험합니다.

    **확인 문제**

    1. 시연을 10개만 기록해 학습하면 성공률이 어떻게 될까요? 6-2와 6-4를 작은 설정으로 돌려 확인해 보세요.
    2. 행동으로 "목표 각도" 대신 "다음 프레임의 관절각"을 기록하면 무엇이 달라질까요?

    다음 [7장](ch07_smolvla.py)에서는 이미지와 함께 **자연어 명령**을 입력으로 받는 VLA 모델 SmolVLA를 다룹니다.
    """)
    return


if __name__ == "__main__":
    app.run()
