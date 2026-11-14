import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import json
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
    RUN_DIR = pc.make_run_dir("ch08-generalization")
    return Path, RUN_DIR, device, json, mo, np, pc, plt, shutil, time, torch


@app.cell(hide_code=True)
def _(device, mo):
    mo.md(rf"""
    # 8장. 처음 보는 상황에서도 동작하는가: 일반화 시험

    학습한 정책은 **학습 때 본 분포** 안에서 잘 동작하도록 만들어졌습니다. 이 장에서는 그 경계를 세 방향으로 시험합니다.

    | 절 | 질문 | 모델 | 자원 |
    | --- | --- | --- | --- |
    | 8-1 | 블록 세 개 중 **문장이 가리키는 색**을 집을 수 있나 | SmolVLA (3색 데이터로 파인튜닝) | 공개 체크포인트 720 MB, GPU |
    | 8-2 | 학습 때와 **다른 블록 위치**에서도 되나 | ACT (고정 위치 vs 무작위 위치 학습) | GPU, 학습 약 10분 |
    | 8-3 | **조명·바닥·카메라·블록 색**이 바뀌면 | ACT, 4장 색 검출 | GPU 권장 |

    현재 장치 **`{device}`**. 6·7장에서 내려받은 공개 체크포인트(`outputs/physicalai/checkpoints/`)를 그대로 씁니다.

    ```text
    Training Environment ≠ Test Environment
    ```
    """)
    return


@app.cell
def _():
    from lerobot.datasets import LeRobotDataset
    return (LeRobotDataset,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 8-1. 색을 지정해 집게 하기

    장면에 빨강·초록·노랑 블록이 서로 7 cm 이상 떨어져 놓입니다. 명령은 `"Pick up the {color} cube and put it in the blue box."`입니다.
    7장에서 본 것처럼 문장이 하나뿐인 데이터로 학습하면 VLA는 문장을 무시합니다. 그래서 **시연마다 집을 색을 바꾸고 문장에 그 색을 넣어**
    데이터를 모읍니다.

    판정은 세 가지입니다. **성공**: 지정한 색만 상자 안 / **오답**: 다른 색이 상자 안 / **실패**: 12초 안에 아무것도 넣지 못함.
    """)
    return


@app.cell
def _(mo, np, pc):
    _rng = np.random.default_rng(2000)
    _layout = pc.random_layout(_rng)
    _robot = pc.SO101Sim(cube_pos=_layout[0], box_pos=pc.BOX_XY, render=False,
                         extra_cubes=[("green", _layout[1]), ("yellow", _layout[2])])
    _top = _robot.snapshot(camera="top", width=320, height=240)
    _fixed = _robot.snapshot(camera="fixed", width=320, height=240)
    _robot.close()
    mo.hstack([mo.image(_top, caption="top 카메라 (평가 seed 2000의 첫 배치)"), mo.image(_fixed, caption="fixed 카메라"),
               mo.md("배치마다 세 색을 차례로 시킵니다. 같은 배치의 세 명령에는 같은 noise seed를 써서 "
                     "차이가 문장에서만 오게 합니다.")], align="center")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### (선택) 3색 시연 데이터셋 기록과 파인튜닝

    책의 공개 체크포인트 `smolvla_multicolor_v2_20k`는 3색 시연 240개로 20,000 step 파인튜닝한 모델입니다.
    직접 해 보려면 아래에서 데이터를 기록한 뒤 7장 7-3의 학습 셀에서 데이터셋을 `so101_multicolor_sim`으로 골라 학습하세요
    (120 Episode 기록 약 5분, 학습은 수 시간).
    """)
    return


@app.cell
def _(mo):
    mc_episodes = mo.ui.slider(30, 240, value=120, step=30, label="성공 시연 수")
    mc_overwrite = mo.ui.checkbox(value=False, label="있으면 지우고 새로 기록")
    mc_btn = mo.ui.run_button(label="3색 시연 기록")
    mo.hstack([mc_episodes, mc_overwrite, mc_btn], justify="start")
    return mc_btn, mc_episodes, mc_overwrite


@app.cell
def _(
    LeRobotDataset,
    RUN_DIR,
    json,
    mc_btn,
    mc_episodes,
    mc_overwrite,
    mo,
    np,
    pc,
    shutil,
):
    mo.stop(not mc_btn.value, mo.md("버튼을 누르면 기록합니다."))
    _root = pc.DATASET_DIR / "so101_multicolor_sim"
    if _root.exists():
        mo.stop(not mc_overwrite.value, mo.callout(mo.md(f"`{_root}`가 이미 있습니다."), kind="warn"))
        shutil.rmtree(_root)
    _ds = LeRobotDataset.create(repo_id="physicalai/so101_multicolor_sim", fps=pc.FPS, features=pc.DATASET_FEATURES,
                                root=_root, robot_type="so101_sim", use_videos=True)
    _rng = np.random.default_rng(0)
    _saved, _failed, _log = {c: 0 for c in pc.COLORS}, 0, []
    with mo.status.progress_bar(total=mc_episodes.value, title="3색 시연 기록") as _bar:
        while sum(_saved.values()) < mc_episodes.value and len(_log) < mc_episodes.value * 10:
            _color = pc.COLORS[sum(_saved.values()) % 3]        # 성공할 때마다 다음 색으로 순환
            _layout = pc.random_layout(_rng)
            _robot = pc.RecordingRobot(dataset=_ds, task=pc.TASK_FMT.format(color=_color), cube_pos=_layout[0],
                                       box_pos=pc.BOX_XY, extra_cubes=[("green", _layout[1]), ("yellow", _layout[2])])
            _ok = pc.scripted_pick_and_place(_robot, color=_color, demo=True)
            _robot.close()
            if _ok:
                _ds.save_episode()
                _saved[_color] += 1
                _bar.update()
            else:
                _ds.clear_episode_buffer()
                _failed += 1
            _log.append({"color": _color, "layout": _layout, "saved": bool(_ok)})
    _ds.finalize()
    (RUN_DIR / "record_multicolor.json").write_text(json.dumps(_log, indent=2))
    mo.md(f"저장 {_saved}, 실패 {_failed}, 총 프레임 {_ds.meta.total_frames}. 위치 `{_root}`")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 3색 SmolVLA 평가
    """)
    return


@app.cell
def _(mo, pc):
    mc_dl_btn = mo.ui.run_button(label=f"공개 3색 SmolVLA 내려받기 ({pc.RELEASE_FILES['smolvla_multicolor_v2_20k'][2]})")
    mc_dl_btn
    return (mc_dl_btn,)


@app.cell
def _(mc_dl_btn, mo, pc):
    if mc_dl_btn.value:
        with mo.status.spinner("내려받는 중..."):
            pc.download_release("smolvla_multicolor_v2_20k")
    _cands = {f"공개: {d.name}": str(p) for d in [pc.CHECKPOINT_DIR / "smolvla_multicolor_v2_20k"]
              for p in pc.find_pretrained_dirs(d)}
    for _p in sorted((pc.CHECKPOINT_DIR / "train").glob("*/checkpoints/last/pretrained_model")):
        if '"type": "smolvla"' in (_p / "config.json").read_text():
            _cands[f"내 학습: {_p.parts[-4]}"] = str(_p)
    _empty = {"(체크포인트 없음)": ""}
    mc_ckpt = mo.ui.dropdown(_cands or _empty, value=next(iter(_cands or _empty)), label="체크포인트")
    mc_layouts = mo.ui.slider(2, 20, value=5, label="배치 수 (배치당 3색)")
    mc_eval_btn = mo.ui.run_button(label="색 지정 평가")
    mo.hstack([mc_ckpt, mc_layouts, mc_eval_btn], justify="start")
    return mc_ckpt, mc_eval_btn, mc_layouts


@app.cell
def _(Path, device, mc_ckpt, mc_eval_btn, mo, pc):
    mo.stop(not mc_eval_btn.value, mo.md("버튼을 누르면 평가합니다. 배치 5개(15 Episode)에 GPU에서 약 5분 걸립니다."))
    mo.stop(not mc_ckpt.value or not (Path(mc_ckpt.value) / "model.safetensors").is_file(), mo.md("체크포인트가 없습니다."))
    with mo.status.spinner("SmolVLA 불러오는 중..."):
        mc_policy, mc_pre, mc_post = pc.load_smolvla(mc_ckpt.value, device)
    mc_loaded = mc_ckpt.value
    return mc_loaded, mc_policy, mc_post, mc_pre


@app.cell
def _(
    RUN_DIR,
    device,
    json,
    mc_layouts,
    mc_loaded,
    mc_policy,
    mc_post,
    mc_pre,
    mo,
    np,
    pc,
):
    _rng = np.random.default_rng(2000)
    _stats = {c: {"성공": 0, "오답": 0, "실패": 0} for c in pc.COLORS}
    _records, _closest = [], []
    with mo.status.progress_bar(total=mc_layouts.value * 3, title="3색 평가") as _bar:
        for _L in range(mc_layouts.value):
            _layout = pc.random_layout(_rng)
            for _color in pc.COLORS:
                _best = {c: 9.0 for c in pc.COLORS}

                def _track(robot, _best=_best):
                    _s = robot.site_pose()[0]
                    for _c in pc.COLORS:
                        _best[_c] = min(_best[_c], float(np.linalg.norm(_s[:2] - robot.cube_pos(_c)[:2])))

                pc.seed_noise(_L)                                  # 한 배치의 세 문장은 같은 noise
                _r = pc.run_policy_episode(mc_policy, mc_pre, mc_post, device, layout=_layout, color=_color,
                                           image_keys=pc.SMOLVLA_KEYS, on_step=_track)
                _near = min(_best, key=_best.get)
                _stats[_color][_r["result"].split("(")[0]] += 1
                _closest.append(_near == _color)
                _records.append({"layout": _L, "color": _color, "result": _r["result"], "closest_block": _near,
                                 "min_dist_cm": {c: round(100 * v, 1) for c, v in _best.items()}})
                _bar.update()
    (RUN_DIR / "multicolor_eval.json").write_text(json.dumps(
        {"checkpoint": mc_loaded, "seed": 2000, "records": _records, "stats": _stats}, indent=2, ensure_ascii=False))
    _n = len(_records)
    _ok = sum(s["성공"] for s in _stats.values())
    _hit = sum(_closest)
    _lo, _hi = pc.wilson(_hit, _n)
    _rows = ["| 명령 색 | 성공 | 오답 | 실패 |", "| --- | ---: | ---: | ---: |"] + [
        f"| {c} | {s['성공']} | {s['오답']} | {s['실패']} |" for c, s in _stats.items()]
    mo.md("\n".join(_rows) + f"""

    - 전체 성공률 **{_ok}/{_n} = {100 * _ok / _n:.0f}%**
    - 손끝이 가장 가까이 간 블록이 명령한 색과 일치한 비율 **{_hit}/{_n} = {100 * _hit / _n:.0f}%**
      (95% 신뢰구간 {100 * _lo:.0f}~{100 * _hi:.0f}%). 문장과 무관하게 움직이면 33% 근처입니다.

    집기 성공률이 낮더라도 두 번째 지표가 33%보다 뚜렷하게 높으면 "문장을 보고 대상을 고르기는 한다"고 볼 수 있습니다.
    책의 측정에서 SmolVLA는 3색 시연 179~240개, 최대 20,000 step 학습으로도 이 과제를 거의 해내지 못했습니다(9장에서 LLM Planner와 비교).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 8-2. 블록 위치 일반화: 고정 위치로 학습하면

    블록을 **항상 (0.24, 0)** 에만 두고 시연 20개를 기록해 ACT를 학습합니다. 그리고 학습 때 본 위치 A와 처음 보는 위치 B, C,
    무작위 위치에서 평가해 무작위 위치로 학습한 6장 ACT와 비교합니다.

    | 위치 | 좌표 |
    | --- | --- |
    | A (학습 위치) | (0.24, 0.00) |
    | B | (0.22, 0.08) |
    | C | (0.20, −0.10) |
    | R | 무작위 (seed 1000) |

    고정 위치 데이터로 학습하면 `observation.state`·이미지와 무관하게 "늘 같은 궤적"을 외우기 쉽습니다.
    """)
    return


@app.cell
def _(mo):
    fx_btn = mo.ui.run_button(label="고정 위치 시연 20개 기록 + ACT 8,000 step 학습 (약 10분)")
    fx_btn
    return (fx_btn,)


@app.cell
def _(LeRobotDataset, RUN_DIR, device, fx_btn, mo, pc):
    mo.stop(not fx_btn.value, mo.md("이미 학습했다면 아래 평가로 바로 가세요."))
    mo.stop(device.type != "cuda", mo.callout(mo.md("CUDA GPU가 필요합니다."), kind="warn"))
    _root = pc.DATASET_DIR / "so101_fixed_sim"
    _out = pc.CHECKPOINT_DIR / "train" / "act_fixed"
    mo.stop(_out.exists(), mo.md(f"`{_out}`가 이미 있습니다. 아래 평가로 가세요."))
    if not (_root / "meta" / "info.json").is_file():
        _ds = LeRobotDataset.create(repo_id="physicalai/so101_fixed_sim", fps=pc.FPS, features=pc.DATASET_FEATURES,
                                    root=_root, robot_type="so101_sim", use_videos=True)
        with mo.status.progress_bar(total=20, title="고정 위치 시연") as _bar:
            for _ in range(20):
                _robot = pc.RecordingRobot(dataset=_ds, cube_pos=(0.24, 0.0), box_pos=pc.BOX_XY)
                _ok = pc.scripted_pick_and_place(_robot, demo=True)
                _robot.close()
                _ds.save_episode() if _ok else _ds.clear_episode_buffer()
                _bar.update()
        _ds.finalize()
    _cmd = [pc.lerobot_cli("lerobot-train"), "--dataset.repo_id=physicalai/so101_fixed_sim", f"--dataset.root={_root}",
            "--policy.type=act", "--policy.chunk_size=50", "--policy.n_action_steps=50", "--policy.device=cuda",
            "--policy.push_to_hub=false", f"--output_dir={_out}", "--job_name=act_fixed", "--steps=8000",
            "--batch_size=16", "--save_freq=4000", "--log_freq=200", "--trackio.enable=false"]
    with mo.status.spinner("ACT 학습 중 (8,000 step)..."):
        _code = pc.stream_command(_cmd, RUN_DIR / "train_act_fixed.log", cwd=pc.REPO_ROOT)
    mo.stop(_code != 0, mo.callout(mo.md(f"학습 실패. 로그 `{RUN_DIR / 'train_act_fixed.log'}`"), kind="danger"))
    mo.md("학습 완료. 아래 평가에서 `act_fixed`를 고르세요.")
    return


@app.cell
def _(fx_btn, mo, pc):
    fx_btn.value   # 학습이 끝나면 목록을 다시 읽는다
    _act = {"공개: act_so101_v2_10k": str(p) for p in pc.find_pretrained_dirs(pc.CHECKPOINT_DIR / "act_so101_v2_10k")}
    for _p in sorted((pc.CHECKPOINT_DIR / "train").glob("*/checkpoints/last/pretrained_model")):
        if '"type": "act"' in (_p / "config.json").read_text():
            _act[f"내 학습: {_p.parts[-4]}"] = str(_p)
    act_choices = _act or {"(ACT 체크포인트 없음)": ""}
    gen_models = mo.ui.multiselect(act_choices, value=list(act_choices)[:2], label="비교할 ACT")
    gen_episodes = mo.ui.slider(5, 20, value=10, step=5, label="위치당 Episode")
    gen_btn = mo.ui.run_button(label="위치 일반화 평가")
    mo.hstack([gen_models, gen_episodes, gen_btn], justify="start")
    return act_choices, gen_btn, gen_episodes, gen_models


@app.cell
def _(
    RUN_DIR,
    act_choices,
    device,
    gen_btn,
    gen_episodes,
    gen_models,
    json,
    mo,
    np,
    pc,
    torch,
):
    mo.stop(not gen_btn.value, mo.md("ACT를 고르고 버튼을 누르세요. 모델 2개 × 위치 4곳 × 10회 = 약 3분."))
    _positions = {"A (0.24, 0.00)": (0.24, 0.0), "B (0.22, 0.08)": (0.22, 0.08), "C (0.20, -0.10)": (0.20, -0.10), "R 무작위": None}
    _names = {v: k for k, v in act_choices.items()}
    _table, _all = {}, {}
    with mo.status.progress_bar(total=len(gen_models.value) * len(_positions) * gen_episodes.value, title="위치 일반화") as _bar:
        for _ckpt in gen_models.value:
            _policy, _pre, _post = pc.load_act(_ckpt, device)
            _label = _names.get(_ckpt, _ckpt)
            for _pos_name, _pos in _positions.items():
                _rng = np.random.default_rng(1000)
                torch.manual_seed(1000)
                _ok = 0
                for _ in range(gen_episodes.value):
                    _cube = _pos if _pos is not None else pc.random_cube(_rng)
                    _ok += pc.run_policy_episode(_policy, _pre, _post, device, cube=_cube)["success"]
                    _bar.update()
                _table[(_label, _pos_name)] = _ok
    (RUN_DIR / "position_generalization.json").write_text(json.dumps(
        {f"{m}|{p}": v for (m, p), v in _table.items()}, indent=2, ensure_ascii=False))
    _models = list(dict.fromkeys(m for m, _ in _table))
    _rows = ["| 모델 | " + " | ".join(_positions) + " |", "| --- |" + " ---: |" * len(_positions)]
    for _m in _models:
        _rows.append(f"| {_m} | " + " | ".join(f"{_table[(_m, p)]}/{gen_episodes.value}" for p in _positions) + " |")
    mo.md("\n".join(_rows) + "\n\n블록 위치가 고정된 A에서는 같은 위치를 반복해도 결과가 거의 같습니다(ACT는 결정적). "
          "R의 성공률이 \"처음 보는 위치\"에 대한 실제 실력입니다.")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 8-3. 환경이 바뀌면: 조명·바닥·카메라·블록 색

    학습 환경에서 한 가지씩만 바꾼 변형 9가지입니다. 먼저 각 변형에서 위 카메라가 보는 이미지를 확인합니다.
    """)
    return


@app.cell
def _():
    VARIANTS = {
        "baseline":     ("학습 환경 그대로", {}),
        "light_dim":    ("조명 절반 밝기", {"light_diffuse": (0.3, 0.3, 0.3)}),
        "light_bright": ("조명 두 배 밝기", {"light_diffuse": (1.2, 1.2, 1.2)}),
        "light_side":   ("조명을 옆으로 (그림자 방향 변화)", {"light_pos": (0.6, 0.6, 0.8)}),
        "floor_dark":   ("바닥을 어둡게", {"floor_rgba": (0.4, 0.4, 0.45, 1)}),
        "floor_red":    ("바닥을 붉은 톤으로 (블록과 비슷한 색)", {"floor_rgba": (0.9, 0.6, 0.6, 1)}),
        "cam_shift":    ("위 카메라를 5 cm 옆으로", {"top_cam_pos": (0.2, 0.05, 0.8)}),
        "cam_high":     ("위 카메라를 10 cm 높게", {"top_cam_pos": (0.2, 0.0, 0.9)}),
        "cube_blue":    ("블록을 파란색으로 (상자와 같은 계열)", {"cube_rgba": (0.1, 0.1, 0.9, 1)}),
    }
    return (VARIANTS,)


@app.cell
def _(VARIANTS, mo, pc):
    _imgs = []
    for _name, (_desc, _kw) in VARIANTS.items():
        _robot = pc.SO101Sim(cube_pos=(0.24, 0.0), box_pos=pc.BOX_XY, render=False, **_kw)
        _cam = pc.SimCamera(_robot.model, "top", pc.IMG_W, pc.IMG_H)
        _rgb, _ = _cam.capture(_robot.data)
        _cam.close()
        _robot.close()
        _imgs.append(mo.image(_rgb, width=220, caption=f"{_name}: {_desc}"))
    mo.hstack(_imgs, wrap=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 4장의 색 검출은 어떻게 되나

    학습이 없는 Rule 기반 인식부터 봅니다. 같은 블록 위치 8곳에서 `locate_object(…, "red")`의 오차를 잽니다.
    """)
    return


@app.cell
def _(VARIANTS, mo, np, pc):
    _rows = ["| 변형 | 검출 | 평균 오차 (mm) |", "| --- | ---: | ---: |"]
    for _name, (_desc, _kw) in VARIANTS.items():
        _rng = np.random.default_rng(3000)
        _errs, _found = [], 0
        for _ in range(8):
            _cube = pc.random_cube(_rng)
            _robot = pc.SO101Sim(cube_pos=_cube, box_pos=pc.BOX_XY, render=False, **_kw)
            _cam = pc.SimCamera(_robot.model, "top")
            _est, _ = pc.locate_object(_cam, _robot.data, "red")
            if _est is not None:
                _found += 1
                _errs.append(np.linalg.norm(_est[:2] - _robot.cube_pos()[:2]) * 1000)
            _cam.close()
            _robot.close()
        _rows.append(f"| `{_name}` | {_found}/8 | {np.mean(_errs):.1f} |" if _errs else f"| `{_name}` | 0/8 | - |")
    mo.md("\n".join(_rows) + "\n\n`cube_blue`에서 \"red\"를 찾지 못하는 것은 당연합니다. 규칙은 명시적이라 실패 이유가 분명합니다. "
          "`cam_shift`·`cam_high`에서도 정확한 이유는 장면의 카메라 위치를 그대로 알고 역투영하기 때문입니다. "
          "실제 로봇에서는 카메라 보정(calibration)이 틀리면 오차가 생깁니다.")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### ACT는 어떻게 되나

    6장 ACT(무작위 위치 학습)를 모든 변형에서 **같은 블록 위치들(seed 3000)** 로 평가합니다.
    ACT는 이미지를 직접 보고 행동을 내므로, 카메라 위치가 바뀌면 "카메라가 옮겨졌다"는 것을 알 방법이 없습니다.
    """)
    return


@app.cell
def _(VARIANTS, act_choices, mo):
    rob_ckpt = mo.ui.dropdown(act_choices, value=list(act_choices)[0], label="ACT")
    rob_variants = mo.ui.multiselect(list(VARIANTS), value=list(VARIANTS), label="변형")
    rob_episodes = mo.ui.slider(5, 20, value=10, step=5, label="변형당 Episode")
    rob_btn = mo.ui.run_button(label="환경 변화 평가")
    mo.vstack([mo.hstack([rob_ckpt, rob_episodes, rob_btn], justify="start"), rob_variants])
    return rob_btn, rob_ckpt, rob_episodes, rob_variants


@app.cell
def _(
    Path,
    RUN_DIR,
    VARIANTS,
    device,
    json,
    mo,
    np,
    pc,
    plt,
    rob_btn,
    rob_ckpt,
    rob_episodes,
    rob_variants,
    time,
    torch,
):
    mo.stop(not rob_btn.value, mo.md("버튼을 누르면 평가합니다. 변형 9개 × 10회 ≈ 3~5분."))
    mo.stop(not rob_ckpt.value or not (Path(rob_ckpt.value) / "model.safetensors").is_file(), mo.md("ACT 체크포인트가 없습니다."))
    _policy, _pre, _post = pc.load_act(rob_ckpt.value, device)
    _res, _t0 = {}, time.time()
    with mo.status.progress_bar(total=len(rob_variants.value) * rob_episodes.value, title="환경 변화") as _bar:
        for _name in rob_variants.value:
            _rng = np.random.default_rng(3000)                 # 모든 변형에 같은 블록 위치들
            torch.manual_seed(3000)
            _ok = 0
            for _ in range(rob_episodes.value):
                _ok += pc.run_policy_episode(_policy, _pre, _post, device, cube=pc.random_cube(_rng),
                                             scene_kw=VARIANTS[_name][1])["success"]
                _bar.update()
            _res[_name] = _ok
    (RUN_DIR / "robustness.json").write_text(json.dumps(
        {"checkpoint": rob_ckpt.value, "seed": 3000, "episodes": rob_episodes.value, "successes": _res,
         "elapsed_seconds": time.time() - _t0}, indent=2))
    _fig, _ax = plt.subplots(figsize=(7, 3))
    _names = list(_res)
    _rates = [100 * _res[n] / rob_episodes.value for n in _names]
    _ax.bar(_names, _rates, color=["tab:gray" if n == "baseline" else "tab:blue" for n in _names])
    _ax.set(ylabel="success rate (%)", ylim=(0, 100), title=f"ACT under scene changes ({rob_episodes.value} ep each)")
    _ax.tick_params(axis="x", rotation=40)
    _ax.grid(alpha=0.3, axis="y")
    _fig.tight_layout()
    _fig.savefig(RUN_DIR / "robustness.png", dpi=120)
    _rows = ["| 변형 | 설명 | 성공 |", "| --- | --- | ---: |"] + [
        f"| `{n}` | {VARIANTS[n][0]} | {_res[n]}/{rob_episodes.value} |" for n in _names]
    mo.hstack([mo.md("\n".join(_rows)), _fig], align="center")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 정리와 다음 장

    | 무엇이 바뀌었나 | Rule 기반 (4장) | 학습 정책 (ACT/SmolVLA) |
    | --- | --- | --- |
    | 블록 위치 | 정답 좌표만 맞으면 됨 | 학습 때 본 범위 안에서만 |
    | 조명·바닥 | HSV 범위가 맞으면 됨, 아니면 명확히 실패 | 이미지 분포가 바뀌면 조용히 성능 저하 |
    | 카메라 위치 | 보정값을 고치면 됨 | 다시 시연하고 학습해야 함 |
    | 문장(색 지정) | 해당 없음 | 문장마다 다른 시연이 충분히 있어야 함 |

    일반화를 높이는 일반적인 방법은 **학습 데이터를 다양하게** 만드는 것입니다(도메인 랜덤화: 조명·색·카메라를 무작위로 바꿔 기록).
    직접 해 보려면 `SO101Sim`(또는 `RecordingRobot`)에 넘기는 장면 인자(`light_diffuse`, `floor_rgba`, `cube_rgba` 등)를 Episode마다 무작위로 바꿔 6장 6-2의 기록 셀을 고쳐 보세요.

    **확인 문제**

    1. 8-3에서 가장 크게 성능이 떨어진 변형은 무엇이고, ACT 입력의 어느 부분 때문일까요?
    2. 8-1에서 손끝이 명령한 색으로 가는 비율은 높지만 집기는 실패한다면, 데이터나 학습을 어떻게 바꿔야 할까요?

    다음 [9장](ch09_llm_planner.py)에서는 정책을 학습하지 않고, LLM이 3장의 도구를 골라 쓰게 해서 8-1의 색 지정 과제에 다시 도전합니다.
    """)
    return


if __name__ == "__main__":
    app.run()
