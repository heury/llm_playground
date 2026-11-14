import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import platform
    import sys
    from pathlib import Path

    _here = str(Path(__file__).resolve().parent)
    if _here not in sys.path:
        sys.path.insert(0, _here)

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    import physicalai_common as pc   # MUJOCO_GL을 먼저 정한 뒤 mujoco를 import한다
    import mujoco

    RUN_DIR = pc.make_run_dir("ch02-mujoco")
    return RUN_DIR, mo, mujoco, np, pc, platform, plt, sys


@app.cell(hide_code=True)
def _(RUN_DIR, mo):
    mo.md(rf"""
    # 2장. MuJoCo로 가상 세계와 로봇팔 만들기

    1장의 `env`를 물리 법칙이 있는 세계로 바꿉니다. **MuJoCo**는 Google DeepMind가 관리하는 오픈소스 물리 엔진이며
    로봇 연구에서 널리 씁니다. 이 장에서 다루는 것:

    1. 개발 환경 확인
    2. XML 몇 줄로 세계 만들기: `MjModel`(설계도)과 `MjData`(현재 상태)
    3. 질량·마찰·중력·공기저항을 바꿔 가며 물체 떨어뜨리기
    4. SO-101 로봇팔(관절 5개 + 그리퍼)을 불러와 관절 하나씩 움직이기

    영상과 그림은 `{RUN_DIR.relative_to(RUN_DIR.parents[2])}`에 저장됩니다.
    화면이 없는 Linux에서는 `MUJOCO_GL=egl`을 자동으로 씁니다(`physicalai_common.py` 첫 부분).
    """)
    return


@app.cell(hide_code=True)
def _(mo, mujoco, np, platform, sys):
    import os

    import cv2
    import imageio
    import torch

    _cuda = torch.cuda.is_available()
    _gpu = f"{torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory / 2**30:.1f} GB)" if _cuda else "없음"
    mo.md(f"""
    ## 2-1. 개발 환경 확인

    | 항목 | 값 |
    | --- | --- |
    | OS | {platform.platform()} |
    | Python | {sys.version.split()[0]} |
    | NumPy | {np.__version__} |
    | PyTorch | {torch.__version__} (CUDA 사용 가능: {_cuda}) |
    | GPU | {_gpu} |
    | MuJoCo | {mujoco.__version__} |
    | OpenCV | {cv2.__version__} |
    | imageio | {imageio.__version__} |
    | MUJOCO_GL | `{os.environ.get("MUJOCO_GL", "(미지정: 창 기반 OpenGL)")}` |

    GPU는 6~8장의 학습·추론에서 필요합니다. 1~5장은 CPU로 충분합니다.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2-2. MuJoCo 첫 실행: 공 하나 떨어뜨리기

    MuJoCo 세계는 **MJCF**라는 XML로 적습니다. 바닥(`plane`)과 1 m 높이의 공 하나가 전부입니다.
    `freejoint`는 "이 body는 6자유도(위치 3 + 회전 3)로 자유롭게 움직인다"는 뜻입니다.
    """)
    return


@app.cell
def _(mujoco):
    FIRST_XML = """
    <mujoco>
      <option gravity="0 0 -9.81"/>
      <worldbody>
        <light pos="0 0 3"/>
        <geom name="floor" type="plane" size="2 2 0.1" rgba="0.8 0.8 0.8 1"/>
        <body name="ball" pos="0 0 1.0">
          <freejoint/>
          <geom type="sphere" size="0.05" mass="0.1" rgba="1 0 0 1"/>
        </body>
      </worldbody>
    </mujoco>
    """
    first_model = mujoco.MjModel.from_xml_string(FIRST_XML)   # 세계의 '설계도' (바뀌지 않는 값)
    first_data = mujoco.MjData(first_model)                    # 세계의 '현재 상태' (시간에 따라 바뀜)
    return first_data, first_model


@app.cell
def _(first_data, first_model, mo, mujoco, np):
    _ball = mujoco.mj_name2id(first_model, mujoco.mjtObj.mjOBJ_BODY, "ball")
    mujoco.mj_forward(first_model, first_data)          # 초기 상태에서 위치 등 파생값 계산
    first_t, first_z = [], []
    for _ in range(1001):                                # 2초
        first_t.append(first_data.time)
        first_z.append(first_data.xpos[_ball][2])        # xpos: 각 body의 월드 좌표
        mujoco.mj_step(first_model, first_data)          # timestep만큼 물리 진행
    first_t, first_z = np.array(first_t), np.array(first_z)

    mo.md(f"""
    - `timestep = {first_model.opt.timestep}` s: `mj_step` 한 번에 흐르는 시간
    - body {first_model.nbody}개(world 포함), geom {first_model.ngeom}개, `nq = {first_model.nq}`
      (freejoint 위치 3 + quaternion 4)
    - `mj_forward`는 시간을 흘리지 않고 위치·속도 같은 파생값만 계산합니다. 3장의 FK·IK에서 다시 씁니다.
    """)
    return first_t, first_z


@app.cell
def _(first_t, first_z, np, plt):
    _analytic = 1.0 - 0.5 * 9.81 * first_t**2
    _fig, _ax = plt.subplots(figsize=(7, 3))
    _ax.plot(first_t, first_z, label="MuJoCo ball z")
    _ax.plot(first_t, np.maximum(_analytic, 0.05), "--", label="1 - g t^2 / 2 (until contact)")
    _ax.axhline(0.05, color="gray", lw=0.8, ls=":")
    _ax.set(xlabel="time (s)", ylabel="height (m)", title="Free fall: simulation vs. formula")
    _ax.legend()
    _ax.grid(alpha=0.3)
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    공은 반지름 5 cm라서 중심이 z = 0.05에서 멈춥니다. 바닥과 닿기 전까지는 공식과 거의 겹칩니다.

    ## 2-3. 책상 위에 물체 떨어뜨리기

    책상(고정), 빨간 블록, 파란 공, 깃털을 떨어뜨립니다. 깃털은 얇은 타원체로 근사하고 MuJoCo의 유체 모델로
    공기저항과 양력을 줍니다(교육용으로 조정한 계수이며 실제 깃털의 측정값은 아닙니다).
    값을 바꾸면 시뮬레이션과 영상이 다시 계산됩니다. 달의 중력은 -1.62입니다.
    """)
    return


@app.cell
def _(mo):
    drop_gravity = mo.ui.slider(-20.0, -0.5, value=-9.81, step=0.01, label="중력 (m/s²)")
    drop_mass = mo.ui.slider(0.01, 2.0, value=0.1, step=0.01, label="블록 질량 (kg)")
    drop_friction = mo.ui.slider(0.0, 2.0, value=1.0, step=0.05, label="마찰 계수")
    drop_density = mo.ui.slider(0.0, 5.0, value=1.225, step=0.025, label="공기 밀도 (kg/m³)")
    drop_wind = mo.ui.slider(0.0, 5.0, value=0.0, step=0.1, label="+x 바람 (m/s)")
    mo.vstack([mo.hstack([drop_gravity, drop_mass, drop_friction], justify="start"),
               mo.hstack([drop_density, drop_wind], justify="start")])
    return drop_density, drop_friction, drop_gravity, drop_mass, drop_wind


@app.cell
def _():
    _barb = '      <geom type="capsule" fromto="0 {y0} 0.002 {x} {y1} 0.002" size="0.0006" mass="0" contype="0" conaffinity="0" rgba="0.65 0.50 0.28 1"/>'
    _barbs = "\n".join(_barb.format(y0=y0, x=s * x, y1=y1)
                       for y0, x, y1 in [(-0.04, 0.020, -0.02), (-0.02, 0.025, 0.0), (0.0, 0.023, 0.025), (0.025, 0.016, 0.045)]
                       for s in (1, -1))
    DROP_XML = """
    <mujoco model="drop_cube">
      <option gravity="0 0 {gravity}" timestep="0.002" integrator="implicitfast"
              density="{air_density}" viscosity="{air_viscosity}"/>
      <visual><global offwidth="640" offheight="480"/></visual>
      <worldbody>
        <light pos="0 0 3" dir="0 0 -1"/>
        <geom name="floor" type="plane" size="2 2 0.1" rgba="0.9 0.9 0.9 1"/>
        <!-- 책상: joint가 없어 세계에 고정. 윗면 높이 0.4 m -->
        <body name="table" pos="0 0 0.2">
          <geom type="box" size="0.4 0.3 0.2" rgba="0.6 0.4 0.2 1" friction="{friction} 0.005 0.0001"/>
        </body>
        <body name="cube" pos="0.1 0 0.9" euler="20 30 0">
          <freejoint/>
          <geom type="box" size="0.03 0.03 0.03" mass="{mass}" rgba="1 0.1 0.1 1" friction="{friction} 0.005 0.0001"/>
        </body>
        <body name="ball" pos="-0.2 0.1 1.1">
          <freejoint/>
          <geom type="sphere" size="0.03" mass="0.1" rgba="0.1 0.3 1 1"/>
        </body>
        <!-- 깃털 0.1 g. fluidcoef: 정면 항력, 측면 항력, 회전 저항, Kutta 양력, Magnus 양력 -->
        <body name="feather" pos="0 -0.16 0.9" euler="15 20 -25">
          <freejoint/>
          <geom name="feather_vane" type="ellipsoid" size="0.026 0.065 0.002" mass="0.00009"
                rgba="0.95 0.88 0.65 1" fluidshape="ellipsoid" fluidcoef="1.2 0.25 0.5 4 1"/>
          <geom name="feather_shaft" type="capsule" fromto="0 -0.085 0 0 0.06 0" size="0.0015"
                mass="0.00001" rgba="0.45 0.30 0.12 1"/>
    BARBS
        </body>
        <camera name="side" mode="targetbody" target="table" pos="0.9 -0.9 0.9"/>
      </worldbody>
    </mujoco>
    """.replace("BARBS", _barbs)
    return (DROP_XML,)


@app.cell
def _(
    DROP_XML,
    RUN_DIR,
    drop_density,
    drop_friction,
    drop_gravity,
    drop_mass,
    drop_wind,
    mujoco,
    np,
    pc,
):
    _xml = DROP_XML.format(mass=drop_mass.value, friction=drop_friction.value, gravity=drop_gravity.value,
                           air_density=drop_density.value,
                           air_viscosity=1.8e-5 if drop_density.value > 0 else 0)  # 밀도 0이면 점성도 끈다
    _model = mujoco.MjModel.from_xml_string(_xml)
    _model.opt.wind[:] = (drop_wind.value, 0.0, 0.0)      # 시간에 따라 바뀌지 않는 바람
    _data = mujoco.MjData(_model)
    _ids = {n: mujoco.mj_name2id(_model, mujoco.mjtObj.mjOBJ_BODY, n) for n in ["cube", "ball", "feather"]}
    mujoco.mj_forward(_model, _data)
    _renderer = mujoco.Renderer(_model, height=480, width=640)
    _frames, _log = [], {"t": [], **{n: [] for n in _ids}}
    _spf = int(1 / (30 * _model.opt.timestep))
    for _step in range(int(4.0 / _model.opt.timestep) + 1):
        if _step % 10 == 0:
            _log["t"].append(_data.time)
            for _n, _i in _ids.items():
                _log[_n].append(_data.xpos[_i].copy())
        if _step % _spf == 0:
            _renderer.update_scene(_data, camera="side")
            _frames.append(_renderer.render().copy())
        mujoco.mj_step(_model, _data)
    _renderer.close()
    drop_log = {k: np.array(v) for k, v in _log.items()}
    drop_video = pc.save_video(_frames, RUN_DIR / "drop_cube.mp4", fps=30)
    return drop_log, drop_video


@app.cell
def _(drop_log, drop_video, mo, plt):
    _fig, _axes = plt.subplots(1, 2, figsize=(10, 3.2))
    for _n, _c in [("cube", "tab:red"), ("ball", "tab:blue"), ("feather", "goldenrod")]:
        _axes[0].plot(drop_log["t"], drop_log[_n][:, 2], color=_c, label=_n)
    _axes[0].set(xlabel="time (s)", ylabel="z (m)", title="Height")
    _axes[1].plot(drop_log["feather"][:, 0], drop_log["feather"][:, 1], color="goldenrod")
    _axes[1].set(xlabel="x (m)", ylabel="y (m)", title="Feather path seen from above")
    _axes[0].legend()
    for _ax in _axes:
        _ax.grid(alpha=0.3)
    _fig.tight_layout()
    mo.hstack([mo.video(src=drop_video, controls=True, loop=True, width=420), _fig], align="center")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **관찰 포인트**

    - 공기 밀도를 0으로 하면 깃털도 블록과 같은 속도로 떨어집니다(진공 낙하).
    - 바람을 켜면 깃털만 크게 밀립니다. 질량에 비해 면적이 큰 물체일수록 유체 힘의 영향이 큽니다.
    - 마찰을 0으로 하면 블록이 책상 위에서 미끄러집니다. 3장 이후 블록을 집을 때 마찰이 중요한 이유입니다.

    ## 2-4. SO-101 로봇팔 불러오기

    [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie)의 [`robotstudio_so101`](https://live.mujoco.org/?model=github:google-deepmind/mujoco_menagerie/main/robotstudio_so101/scene.xml) 모델을 씁니다.
    처음 실행하면 SO-101 폴더만 `notebooks/physicalai/third_party/`에 내려받습니다(수십 MB).
    SO-101은 오픈소스 저가형 로봇팔이며 LeRobot 커뮤니티가 많이 씁니다.

    - **joint**(관절): 움직일 수 있는 축. `qpos`에 각도가 저장됩니다.
    - **actuator**(구동기): joint를 움직이는 모터. `ctrl`에 목표 각도를 넣으면 위치 제어기가 따라갑니다.
    """)
    return


@app.cell
def _(mo, mujoco, pc):
    arm_model = mujoco.MjModel.from_xml_path(pc.so101_xml("scene.xml"))
    _rows = ["| id | joint | 가동 범위 (rad) | actuator | ctrl 범위 |", "| ---: | --- | --- | --- | --- |"]
    for _a in range(arm_model.nu):
        _jid = arm_model.actuator_trnid[_a][0]
        _lo, _hi = arm_model.jnt_range[_jid]
        _clo, _chi = arm_model.actuator_ctrlrange[_a]
        _rows.append(f"| {_a} | `{mujoco.mj_id2name(arm_model, mujoco.mjtObj.mjOBJ_JOINT, _jid)}` | "
                     f"[{_lo:+.3f}, {_hi:+.3f}] | `{mujoco.mj_id2name(arm_model, mujoco.mjtObj.mjOBJ_ACTUATOR, _a)}` | "
                     f"[{_clo:+.3f}, {_chi:+.3f}] |")
    mo.md(f"body {arm_model.nbody}개, joint {arm_model.njnt}개, actuator {arm_model.nu}개, "
          f"`nq = {arm_model.nq}`, `nu = {arm_model.nu}`\n\n" + "\n".join(_rows))
    return (arm_model,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 관절 하나씩 움직이기

    각 구동기에 차례로 가동 범위의 60%를 명령하고 1초 기다린 뒤 0으로 되돌립니다.
    손(`gripper` body) 위치가 어떻게 바뀌는지 표와 영상으로 확인합니다.
    """)
    return


@app.cell
def _(RUN_DIR, arm_model, mo, mujoco, pc):
    _data = mujoco.MjData(arm_model)
    mujoco.mj_forward(arm_model, _data)
    _hand = mujoco.mj_name2id(arm_model, mujoco.mjtObj.mjOBJ_BODY, "gripper")
    _renderer = mujoco.Renderer(arm_model, height=480, width=640)
    _cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(arm_model, _cam)
    _cam.azimuth, _cam.elevation, _cam.distance = 150, -25, 0.9
    _cam.lookat[:] = [0.0, -0.1, 0.15]
    _spf = int(round(1 / (30 * arm_model.opt.timestep)))
    _frames = []

    def _run(steps):
        for _s in range(steps):
            mujoco.mj_step(arm_model, _data)
            if _s % _spf == 0:
                _renderer.update_scene(_data, camera=_cam)
                _frames.append(_renderer.render().copy())

    _hold = int(1.0 / arm_model.opt.timestep)
    _fmt = lambda p: ", ".join(f"{v:+.3f}" for v in p)   # noqa: E731
    _rows = ["| actuator | 명령 ctrl | 실제 관절각 | 손 위치 (x, y, z) m |", "| --- | ---: | ---: | --- |",
             f"| (초기 자세) | 0 | - | {_fmt(_data.xpos[_hand])} |"]
    for _a in range(arm_model.nu):
        _lo, _hi = arm_model.actuator_ctrlrange[_a]
        _target = 0.6 * _hi if abs(_hi) > abs(_lo) else 0.6 * _lo
        _data.ctrl[:] = 0.0
        _data.ctrl[_a] = _target
        _run(_hold)
        _actual = _data.qpos[arm_model.jnt_qposadr[arm_model.actuator_trnid[_a][0]]]
        _rows.append(f"| `{mujoco.mj_id2name(arm_model, mujoco.mjtObj.mjOBJ_ACTUATOR, _a)}` | {_target:+.3f} | "
                     f"{_actual:+.3f} | {_fmt(_data.xpos[_hand])} |")
        _data.ctrl[:] = 0.0
        _run(_hold)
    _renderer.close()
    _video = pc.save_video(_frames, RUN_DIR / "robot_arm_joints.mp4", fps=30)
    mo.hstack([mo.video(src=_video, controls=True, loop=True, width=420), mo.md("\n".join(_rows))], align="center")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    명령한 각도와 실제 관절각이 조금 다를 수 있습니다. 구동기는 목표를 향해 힘을 내는 **위치 제어기**이고,
    중력과 관절 한계가 그 결과에 영향을 줍니다. `gripper` 관절은 손 body 위치를 바꾸지 않습니다.

    ### 직접 해 보기: 슬라이더로 자세 만들기

    여섯 구동기의 목표 각도를 정하면 1.5초 동안 시뮬레이션한 뒤의 자세를 보여 줍니다.
    3장에서 "원하는 손 위치를 만드는 관절각"을 역으로 계산하는 방법(IK)을 배웁니다.
    """)
    return


@app.cell
def _(arm_model, mo, mujoco):
    _names = [mujoco.mj_id2name(arm_model, mujoco.mjtObj.mjOBJ_ACTUATOR, a) for a in range(arm_model.nu)]
    pose_sliders = mo.ui.array([
        mo.ui.slider(float(arm_model.actuator_ctrlrange[a][0]), float(arm_model.actuator_ctrlrange[a][1]),
                     value=0.0, step=0.01, label=_names[a])
        for a in range(arm_model.nu)])
    pose_sliders
    return (pose_sliders,)


@app.cell
def _(arm_model, mo, mujoco, np, pose_sliders):
    _data = mujoco.MjData(arm_model)
    _data.ctrl[:] = np.array(pose_sliders.value)
    for _ in range(int(1.5 / arm_model.opt.timestep)):
        mujoco.mj_step(arm_model, _data)
    _cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(arm_model, _cam)
    _cam.azimuth, _cam.elevation, _cam.distance = 150, -25, 0.9
    _cam.lookat[:] = [0.0, -0.1, 0.15]
    _r = mujoco.Renderer(arm_model, height=360, width=480)
    _r.update_scene(_data, camera=_cam)
    _img = _r.render().copy()
    _r.close()
    _hand = _data.xpos[mujoco.mj_name2id(arm_model, mujoco.mjtObj.mjOBJ_BODY, "gripper")]
    mo.hstack([mo.image(_img), mo.md(f"손 위치 (x, y, z) = `{np.round(_hand, 3)}` m")], align="center")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 정리와 다음 장

    | 개념 | MuJoCo 이름 | 쓰임 |
    | --- | --- | --- |
    | 설계도 | `MjModel` | 질량, 관절 범위, `opt.timestep`, 카메라 |
    | 현재 상태 | `MjData` | `qpos`(관절각·자유물체 자세), `ctrl`(구동기 명령), `xpos`(body 위치), `time` |
    | 시간 진행 | `mj_step` | 물리 시뮬레이션 한 스텝 |
    | 운동학만 계산 | `mj_forward` | 시간을 흘리지 않고 위치·자세 갱신 |

    **확인 문제**

    1. `timestep`을 0.01로 키우면 공의 낙하 곡선은 어떻게 달라지나요? 너무 키우면 어떤 문제가 생길까요?
    2. `shoulder_pan`만 움직였을 때 손의 z 좌표가 거의 그대로인 이유는 무엇인가요?

    다음 [3장](ch03_kinematics_pick_place.py)에서는 관절각과 손끝 위치의 관계(FK·IK)를 배우고 블록을 집어 상자에 넣습니다.
    """)
    return


if __name__ == "__main__":
    app.run()
