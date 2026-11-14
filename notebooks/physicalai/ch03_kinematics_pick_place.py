import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import inspect
    import json
    import sys
    from pathlib import Path

    _here = str(Path(__file__).resolve().parent)
    if _here not in sys.path:
        sys.path.insert(0, _here)

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    import physicalai_common as pc
    import mujoco

    np.set_printoptions(precision=3, suppress=True)
    RUN_DIR = pc.make_run_dir("ch03-kinematics")
    return RUN_DIR, inspect, json, mo, mujoco, np, pc, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 3장. 좌표계·FK·IK와 Rule 기반 Pick and Place

    로봇에게 "블록을 집어"라고 시키려면 두 방향의 계산이 필요합니다.

    - **Forward Kinematics(FK)**: 관절각 → 손끝 위치. 계산이 쉽고 답이 하나입니다.
    - **Inverse Kinematics(IK)**: 원하는 손끝 위치 → 관절각. 답이 없거나 여러 개일 수 있어 반복 계산으로 풉니다.

    이 장의 마지막에는 AI 없이 다섯 단계 상태 기계로 빨간 블록을 파란 상자에 넣습니다.
    이 Rule 기반 Pick and Place는 6장에서 **시연 데이터를 만드는 시연자**가 되고, 9장에서 LLM과 비교하는 기준이 됩니다.

    장면은 `physicalai_common.load_scene()`이 만듭니다. SO-101 + 바닥 + 빨간 블록(한 변 3 cm) + 파란 상자 + 카메라 5대입니다.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3-1. Pose 읽기: 위치 + 자세

    **Pose**는 위치(x, y, z)와 자세(회전)를 합친 것입니다. MuJoCo는 body의 자세를 quaternion `(w, x, y, z)`로 저장합니다.
    quaternion은 계산에 유리하지만 사람이 읽기 어려워 Roll·Pitch·Yaw(각 축 회전각)로도 바꿔 봅니다.
    """)
    return


@app.cell
def _(np):
    def quat_to_rpy(q):
        """MuJoCo quaternion (w, x, y, z) → roll, pitch, yaw (rad)."""
        w, x, y, z = q
        roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
        pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1, 1))
        yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        return np.array([roll, pitch, yaw])
    return (quat_to_rpy,)


@app.cell
def _(mo):
    pose_yaw = mo.ui.slider(-180, 180, value=45, step=5, label="블록을 z축으로 돌린 각도 (deg)")
    pose_yaw
    return (pose_yaw,)


@app.cell
def _(mo, mujoco, np, pc, pose_yaw, quat_to_rpy):
    _model = pc.load_scene(cube_pos=(0.25, 0.0), box_pos=(0.0, 0.25))
    _data = mujoco.MjData(_model)
    mujoco.mj_resetDataKeyframe(_model, _data, 0)                # 로봇을 home 자세로
    # freejoint의 qpos: [x, y, z, qw, qx, qy, qz]
    _adr = _model.jnt_qposadr[mujoco.mj_name2id(_model, mujoco.mjtObj.mjOBJ_JOINT, "cube_free")]
    _yaw = np.radians(pose_yaw.value)
    _data.qpos[_adr + 3: _adr + 7] = [np.cos(_yaw / 2), 0, 0, np.sin(_yaw / 2)]   # z축 회전 quaternion
    mujoco.mj_forward(_model, _data)

    _rows = ["| body | 위치 (x, y, z) m | quaternion (w, x, y, z) | RPY (deg) |", "| --- | --- | --- | --- |"]
    for _name in ["cube", "box", "gripper", "base"]:
        _bid = mujoco.mj_name2id(_model, mujoco.mjtObj.mjOBJ_BODY, _name)
        _rows.append(f"| `{_name}` | {np.round(_data.xpos[_bid], 3)} | {np.round(_data.xquat[_bid], 3)} | "
                     f"{np.round(np.degrees(quat_to_rpy(_data.xquat[_bid])), 1)} |")
    _site = mujoco.mj_name2id(_model, mujoco.mjtObj.mjOBJ_SITE, "gripperframe")
    _R = _data.site_xmat[_site].reshape(3, 3)
    _base = mujoco.mj_name2id(_model, mujoco.mjtObj.mjOBJ_BODY, "base")
    _cube = mujoco.mj_name2id(_model, mujoco.mjtObj.mjOBJ_BODY, "cube")
    _rel = _data.xmat[_base].reshape(3, 3).T @ (_data.xpos[_cube] - _data.xpos[_base])
    mo.md("\n".join(_rows) + f"""

    - 손끝 기준점 site `gripperframe` 위치 {np.round(_data.site_xpos[_site], 3)}, x축(손가락 방향) {np.round(_R[:, 0], 3)}
    - 로봇 base 좌표계에서 본 블록 위치 `R_baseᵀ (p_cube − p_base)` = {np.round(_rel, 3)}

    SO-101의 base는 월드 원점에 회전 없이 놓여 있어 두 좌표가 같습니다. 로봇을 옮기거나 돌리면 달라집니다.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3-2. Forward Kinematics

    ### 2관절 평면 로봇팔

    링크 길이 $l_1, l_2$, 관절각 $\theta_1, \theta_2$일 때 손끝은

    $$
    x = l_1\cos\theta_1 + l_2\cos(\theta_1+\theta_2),\qquad
    y = l_1\sin\theta_1 + l_2\sin(\theta_1+\theta_2)
    $$

    두 번째 링크의 방향은 **누적 각도** $\theta_1+\theta_2$입니다.
    """)
    return


@app.cell
def _(np):
    def fk_2link(theta1, theta2, l1=0.12, l2=0.14):
        """평면 2관절 로봇팔. (손끝, 팔꿈치) 좌표를 돌려준다."""
        elbow = np.array([l1 * np.cos(theta1), l1 * np.sin(theta1)])
        hand = elbow + l2 * np.array([np.cos(theta1 + theta2), np.sin(theta1 + theta2)])
        return hand, elbow
    return (fk_2link,)


@app.cell
def _(mo):
    fk_t1 = mo.ui.slider(-180, 180, value=30, label="θ1 (deg)")
    fk_t2 = mo.ui.slider(-180, 180, value=60, label="θ2 (deg)")
    mo.hstack([fk_t1, fk_t2], justify="start")
    return fk_t1, fk_t2


@app.cell
def _(fk_2link, fk_t1, fk_t2, np, plt):
    _hand, _elbow = fk_2link(np.radians(fk_t1.value), np.radians(fk_t2.value))
    # 손끝이 닿을 수 있는 영역: 모든 관절각 조합의 FK
    _g = np.radians(np.arange(-180, 181, 4))
    _cloud = np.array([fk_2link(a, b)[0] for a in _g for b in _g])
    _fig, _ax = plt.subplots(figsize=(4.5, 4.5))
    _ax.scatter(_cloud[:, 0], _cloud[:, 1], s=1, color="lightgray", label="reachable (all angles)")
    _ax.plot([0, _elbow[0], _hand[0]], [0, _elbow[1], _hand[1]], "-o", lw=3, color="tab:blue")
    _ax.annotate(f"hand ({_hand[0]:.3f}, {_hand[1]:.3f})", _hand, textcoords="offset points", xytext=(5, 5))
    _ax.set(xlim=(-0.3, 0.3), ylim=(-0.3, 0.3), aspect="equal", title="2-link planar arm FK")
    _ax.grid(alpha=0.3)
    _ax.legend(loc="lower left", fontsize=8)
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    회색 영역의 안쪽 구멍은 $|l_1 - l_2| = 2$ cm보다 가까운 곳, 바깥 경계는 $l_1 + l_2 = 26$ cm입니다.
    손이 닿지 않는 곳은 IK도 답을 찾을 수 없습니다.

    ### SO-101의 FK: `mj_forward` 한 번

    관절 5개짜리 3차원 로봇팔은 식을 손으로 쓰기 번거롭습니다. MuJoCo는 관절각을 `qpos`에 넣고 `mj_forward`만 부르면
    모든 body와 site의 위치를 계산합니다. 물리 시뮬레이션(`mj_step`)이 아니라 운동학 계산입니다.
    """)
    return


@app.cell
def _(mo, mujoco, np, pc):
    fk_model = pc.load_scene()
    fk_data = mujoco.MjData(fk_model)
    fk_site = mujoco.mj_name2id(fk_model, mujoco.mjtObj.mjOBJ_SITE, "gripperframe")
    _poses = [([0, 0, 0, 0, 0], "모든 관절 0: 팔을 앞으로 뻗음"),
              ([0, -1.5, 0.5, 0, 0], "home(대기) 자세: 팔을 위로 세움"),
              ([0.5, -1.5, 0.5, 0, 0], "대기 자세에서 pan만 0.5 rad"),
              ([0, -1.57, 1.57, 1.2, 0], "손목을 꺾어 손가락이 바닥을 향함"),
              ([0, 0.0, 0.3, 1.27, 0], "블록 위 (IK가 찾는 자세와 비슷)")]
    _rows = ["| q = [pan, lift, elbow, wrist_flex, roll] | 설명 | 손끝 (x, y, z) m |", "| --- | --- | --- |"]
    for _q, _desc in _poses:
        fk_data.qpos[:5] = _q
        mujoco.mj_forward(fk_model, fk_data)
        _rows.append(f"| {_q} | {_desc} | {np.round(fk_data.site_xpos[fk_site], 3)} |")
    mo.md("\n".join(_rows))
    return fk_data, fk_model, fk_site


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 관절을 조금 움직이면 손끝은 얼마나 움직이나: Jacobian

    관절 하나를 $\varepsilon$만큼 바꿨을 때 손끝 이동량을 $\varepsilon$으로 나누면 **Jacobian의 한 열**이 됩니다.
    IK는 이 정보를 써서 "손끝을 목표 쪽으로 보내려면 각 관절을 얼마나 바꿔야 하나"를 계산합니다.
    아래 표는 수치 미분 값과 MuJoCo의 `mj_jacSite`가 계산한 해석적 값을 나란히 놓은 것입니다.
    """)
    return


@app.cell
def _(fk_data, fk_model, fk_site, mo, mujoco, np, pc):
    _q0 = np.array([0, -1.0, 1.0, 1.2, 0.0])
    fk_data.qpos[:5] = _q0
    mujoco.mj_forward(fk_model, fk_data)
    _p0 = fk_data.site_xpos[fk_site].copy()
    _jacp = np.zeros((3, fk_model.nv))
    mujoco.mj_jacSite(fk_model, fk_data, _jacp, None, fk_site)
    _eps = 0.01
    _rows = ["| 관절 | 수치 미분 dp/dq (m/rad) | mj_jacSite 열 | 크기 |", "| --- | --- | --- | ---: |"]
    for _i, _n in enumerate(pc.ARM_JOINTS):
        _q = _q0.copy()
        _q[_i] += _eps
        fk_data.qpos[:5] = _q
        mujoco.mj_forward(fk_model, fk_data)
        _dp = (fk_data.site_xpos[fk_site] - _p0) / _eps
        _rows.append(f"| `{_n}` | {np.round(_dp, 3)} | {np.round(_jacp[:, _i], 3)} | {np.linalg.norm(_dp):.3f} |")
    mo.md(f"기준 자세 q = {_q0.tolist()}\n\n" + "\n".join(_rows) +
          "\n\n`wrist_roll`은 손가락 축을 따라 도는 관절이라 손끝 위치가 거의 바뀌지 않습니다.")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3-3. Inverse Kinematics: Damped Least Squares

    목표 위치 $p^*$와 현재 손끝 $p(q)$의 오차 $e = p^* - p(q)$를 줄이는 관절 변화량 $\Delta q$를 반복해서 구합니다.

    $$
    \Delta q = J^\top (J J^\top + \lambda^2 I)^{-1} e,\qquad q \leftarrow \operatorname{clip}(q + \alpha\,\Delta q)
    $$

    - $J$: Jacobian. 손가락 방향까지 맞출 때는 위치 3행 + 방향 3행 = 6행입니다.
    - $\lambda$(damping): 팔이 쭉 펴진 특이 자세 근처에서 $\Delta q$가 폭발하지 않게 막습니다.
    - clip: 관절 가동 범위 밖으로 나가지 않게 자릅니다.

    아래는 반복마다 오차를 기록하도록 조금 고친 IK입니다. 원본은 `physicalai_common.solve_ik`에 있습니다.
    """)
    return


@app.cell
def _(mujoco, np, pc):
    def ik_with_trace(model, data, target_pos, target_dir=pc.DOWN, max_iters=300, tol=1e-3,
                      damping=1e-2, step=0.5):
        """solve_ik와 같지만 반복마다 (위치 오차, 방향 오차)를 기록한다."""
        site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, pc.SITE)
        jids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in pc.ARM_JOINTS]
        qadr, dadr = model.jnt_qposadr[jids], model.jnt_dofadr[jids]
        lo, hi = model.jnt_range[jids, 0], model.jnt_range[jids, 1]
        scratch = mujoco.MjData(model)
        scratch.qpos[:] = data.qpos
        target_dir = None if target_dir is None else np.asarray(target_dir, float) / np.linalg.norm(target_dir)
        jacp, jacr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
        trace = []
        for _ in range(max_iters):
            mujoco.mj_forward(model, scratch)
            err = np.asarray(target_pos) - scratch.site_xpos[site]
            rot = np.zeros(3)
            if target_dir is not None:
                rot = np.cross(scratch.site_xmat[site].reshape(3, 3)[:, 0], target_dir)
                err = np.concatenate([err, rot])
            trace.append((np.linalg.norm(err[:3]), np.linalg.norm(rot)))
            if np.linalg.norm(err) < tol:
                break
            mujoco.mj_jacSite(model, scratch, jacp, jacr, site)
            J = jacp[:, dadr] if target_dir is None else np.vstack([jacp, jacr])[:, dadr]
            dq = J.T @ np.linalg.solve(J @ J.T + damping**2 * np.eye(J.shape[0]), err)
            scratch.qpos[qadr] = np.clip(scratch.qpos[qadr] + step * dq, lo, hi)
        return scratch.qpos[qadr].copy(), np.array(trace)
    return (ik_with_trace,)


@app.cell
def _(mo):
    ik_x = mo.ui.slider(0.05, 0.45, value=0.24, step=0.01, label="목표 x (m)")
    ik_y = mo.ui.slider(-0.25, 0.25, value=0.0, step=0.01, label="목표 y (m)")
    ik_z = mo.ui.slider(0.0, 0.35, value=0.08, step=0.01, label="목표 z (m)")
    ik_down = mo.ui.checkbox(value=True, label="손가락을 아래로 향하게")
    ik_damping = mo.ui.dropdown({"0.001": 1e-3, "0.01": 1e-2, "0.1": 1e-1}, value="0.01", label="damping λ")
    mo.vstack([mo.hstack([ik_x, ik_y, ik_z], justify="start"), mo.hstack([ik_down, ik_damping], justify="start")])
    return ik_damping, ik_down, ik_x, ik_y, ik_z


@app.cell
def _(ik_damping, ik_down, ik_with_trace, ik_x, ik_y, ik_z, mo, mujoco, np, pc, plt):
    _robot = pc.SO101Sim(cube_pos=(0.24, 0.0), box_pos=pc.BOX_XY, render=False)
    _target = np.array([ik_x.value, ik_y.value, ik_z.value])
    _q, _trace = ik_with_trace(_robot.model, _robot.data, _target,
                               pc.DOWN if ik_down.value else None, damping=ik_damping.value)
    # 구한 관절각을 그대로 넣고 FK로 확인 (시뮬레이션 없이 자세만 보여 준다)
    _robot.data.qpos[_robot.arm_qadr] = _q
    mujoco.mj_forward(_robot.model, _robot.data)
    _reached = _robot.site_pose()[0]
    _img = _robot.snapshot(camera="fixed", width=480, height=360)
    _robot.close()

    _fig, _ax = plt.subplots(figsize=(4.5, 3))
    _ax.semilogy(_trace[:, 0], label="position error (m)")
    if ik_down.value:
        _ax.semilogy(_trace[:, 1] + 1e-9, label="direction error")
    _ax.axhline(1e-3, color="gray", ls=":", lw=0.8)
    _ax.set(xlabel="iteration", title="IK convergence")
    _ax.legend(fontsize=8)
    _ax.grid(alpha=0.3)
    _fig.tight_layout()
    _ok = np.linalg.norm(_reached - _target) < 0.01
    mo.vstack([
        mo.hstack([mo.image(_img), _fig], align="center"),
        mo.md(f"관절각 {np.round(_q, 3)} rad, 반복 {len(_trace)}회, 실제 손끝 {np.round(_reached, 3)}, "
              f"목표와 거리 **{np.linalg.norm(_reached - _target) * 100:.2f} cm** "
              f"({'도달' if _ok else '도달 못 함: 작업 영역 밖이거나 손가락 방향 제약과 충돌'})"),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **실험**: x를 0.45로 늘리면 오차가 0으로 가지 않습니다(팔 길이 밖). 같은 점이라도 "손가락 아래로" 제약을 끄면
    도달할 수 있는 경우가 있습니다. 제약이 많을수록 도달 가능한 영역이 좁아집니다.
    damping을 0.001로 줄이면 수렴은 빨라질 수 있지만 경계 근처에서 관절이 크게 튈 수 있습니다.

    ## 3-4. IK로 손을 블록 위로 보내기

    IK로 관절각을 구한 뒤 `move_arm`이 현재 명령에서 목표 관절각까지 **선형 보간**하며 1~1.5초에 걸쳐 움직입니다.
    한 번에 목표를 주면 구동기가 급하게 움직여 흔들리기 때문입니다.
    """)
    return


@app.cell
def _(inspect, mo, pc):
    mo.accordion({
        "SO101Sim.move_arm / move_to 소스 보기": mo.md(
            "```python\n" + inspect.getsource(pc.SO101Sim.move_arm) + "\n" + inspect.getsource(pc.SO101Sim.move_to) + "```"),
        "solve_ik 원본 소스 보기": mo.md("```python\n" + inspect.getsource(pc.solve_ik) + "```"),
    })
    return


@app.cell
def _(RUN_DIR, mo, np, pc):
    _robot = pc.SO101Sim(cube_pos=(0.24, 0.0), box_pos=pc.BOX_XY, render=True, camera="fixed")
    _cube = _robot.cube_pos()
    _robot.open_gripper(0.5)
    _log = []
    for _z, _sec in [(0.08, 1.5), (0.02, 1.0)]:             # 블록 위 8 cm → 손끝이 블록을 감싸는 2 cm
        _target = np.array([_cube[0], _cube[1], _z])
        _q, _err = _robot.move_to(_target, seconds=_sec)
        _robot.hold(0.3)
        _p = _robot.site_pose()[0]
        _log.append(f"| z = {_z} | {np.round(_q, 3)} | {_err:.4f} | {np.linalg.norm(_p - _target) * 100:.2f} cm | "
                    f"{np.round(_robot.finger_dir(), 2)} |")
    _video = _robot.save_video(RUN_DIR / "ik_reach.mp4")
    _robot.close()
    mo.hstack([mo.video(src=_video, loop=True, width=420),
               mo.md("| 목표 | 관절각 | IK 잔차 | 실제 거리 | 손가락 방향 |\n| --- | --- | ---: | ---: | --- |\n" + "\n".join(_log))],
              align="center")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    IK 잔차는 계산상의 오차, 실제 거리는 시뮬레이션 후의 오차입니다. 둘이 다른 이유는 위치 제어기가 중력을 이기며 따라가는
    동안 생기는 작은 처짐 때문입니다.

    ## 3-5. Rule 기반 Pick and Place

    다섯 단계를 차례로 실행하는 **상태 기계(state machine)** 입니다. 블록과 상자 위치는 시뮬레이터에서 직접 읽습니다.
    4장에서는 이 두 줄만 카메라 추정으로 바꿉니다.

    ```mermaid
    flowchart LR
      A[1 Approach<br/>블록 위 8 cm] --> B[2 Grasp<br/>2 cm로 내려가 닫기]
      B --> C[3 Lift<br/>12 cm로 들기]
      C -->|블록 z > 6 cm| D[4 Move<br/>상자 위로]
      C -->|놓침| X[실패]
      D --> E[5 Place<br/>6 cm에서 벌리기]
    ```
    """)
    return


@app.cell
def _(np, pc):
    APPROACH_Z, GRASP_Z, LIFT_Z, PLACE_Z = 0.08, 0.02, 0.12, 0.06   # 손끝 목표 높이 (바닥 기준, m)


    def above(xy, z):
        return np.array([xy[0], xy[1], z])


    def is_in_box(cube_pos, box_pos, half=0.05):
        dx, dy = abs(cube_pos[0] - box_pos[0]), abs(cube_pos[1] - box_pos[1])
        return dx < half and dy < half and cube_pos[2] < 0.05


    def pick_and_place(robot, log=print):
        """다섯 단계를 실행하고 성공 여부를 돌려준다."""
        cube = robot.cube_pos()          # 시뮬레이터에서 직접 읽는 State (4장에서 카메라로 바뀜)
        box = robot.box_pos()

        log("1 Approach")
        robot.open_gripper(0.5)
        _, err = robot.move_to(above(cube, APPROACH_Z), seconds=1.2)
        log(f"   IK 오차 {err:.4f}  손끝 {np.round(robot.site_pose()[0], 3)}")

        log("2 Grasp")
        robot.move_to(above(cube, GRASP_Z), seconds=0.8)
        robot.hold(0.2)
        robot.close_gripper(0.8)
        log(f"   집게 각도 {robot.gripper_opening():+.3f} rad (블록에 막혀 덜 닫히면 잡았다는 신호, 완전히 닫힘 = {pc.GRIPPER_CLOSED})")

        log("3 Lift")
        robot.move_to(above(cube, LIFT_Z), seconds=1.0)
        robot.hold(0.2)
        lifted = robot.cube_pos()[2] > 0.06
        log(f"   블록 높이 {robot.cube_pos()[2]:.3f} m → {'집었음' if lifted else '놓쳤음'}")
        if not lifted:
            return False

        log("4 Move")
        robot.move_to(above(box, LIFT_Z), seconds=1.5)
        robot.hold(0.2)

        log("5 Place")
        robot.move_to(above(box, PLACE_Z), seconds=0.8)
        robot.hold(0.2)
        robot.open_gripper(0.5)
        robot.move_to(above(box, LIFT_Z), seconds=0.8)
        robot.hold(0.5)

        ok = is_in_box(robot.cube_pos(), box)
        log(f"   블록 최종 위치 {np.round(robot.cube_pos(), 3)}  상자 {np.round(box[:2], 3)} → {'성공' if ok else '실패'}")
        return ok
    return (pick_and_place,)


@app.cell
def _(mo):
    pp_cube_x = mo.ui.slider(0.12, 0.32, value=0.24, step=0.01, label="블록 x (m)")
    pp_cube_y = mo.ui.slider(-0.2, 0.2, value=0.0, step=0.01, label="블록 y (m)")
    mo.hstack([pp_cube_x, pp_cube_y], justify="start")
    return pp_cube_x, pp_cube_y


@app.cell
def _(RUN_DIR, mo, pc, pick_and_place, pp_cube_x, pp_cube_y):
    _robot = pc.SO101Sim(cube_pos=(pp_cube_x.value, pp_cube_y.value), box_pos=pc.BOX_XY, render=True, camera="fixed")
    _lines = []
    _ok = pick_and_place(_robot, log=_lines.append)
    _video = _robot.save_video(RUN_DIR / "pick_and_place.mp4")
    _robot.close()
    mo.hstack([mo.video(src=_video, loop=True, width=440),
               mo.md("```text\n" + "\n".join(_lines) + f"\n\n결과: {'성공' if _ok else '실패'}\n```")], align="center")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3-6. 무작위 위치에서 성공률 재기

    한 번의 성공은 우연일 수 있습니다. 블록을 로봇 앞 부채꼴(반지름 0.17~0.27 m, 각도 ±40°)에 무작위로 놓고 여러 번 반복합니다.
    5장 이후의 강화학습·모방학습 정책도 같은 영역, 같은 판정(`is_in_box`)으로 평가합니다.
    """)
    return


@app.cell
def _(mo):
    trials_n = mo.ui.slider(5, 50, value=10, step=5, label="시도 횟수")
    trials_seed = mo.ui.number(value=0, label="seed")
    trials_btn = mo.ui.run_button(label="무작위 위치로 반복 실행")
    mo.hstack([trials_n, trials_seed, trials_btn], justify="start")
    return trials_btn, trials_n, trials_seed


@app.cell
def _(RUN_DIR, json, mo, np, pc, pick_and_place, trials_btn, trials_n, trials_seed):
    mo.stop(not trials_btn.value, mo.md("버튼을 누르면 실행합니다. 10회에 약 10초 걸립니다(영상 없이)."))
    _rng = np.random.default_rng(int(trials_seed.value))
    trial_records = []
    with mo.status.progress_bar(total=trials_n.value, title="Pick and Place 반복") as _bar:
        for _i in range(trials_n.value):
            _cube = pc.random_cube(_rng)
            _robot = pc.SO101Sim(cube_pos=_cube, box_pos=pc.BOX_XY, render=False)
            _ok = pick_and_place(_robot, log=lambda *a: None)
            trial_records.append({"trial": _i + 1, "cube_xy": list(_cube), "success": bool(_ok),
                                  "final_cube_xyz": _robot.cube_pos().round(4).tolist()})
            _robot.close()
            _bar.update()
    (RUN_DIR / "pick_and_place_trials.json").write_text(
        json.dumps({"seed": int(trials_seed.value), "box_xy": pc.BOX_XY, "trials": trial_records}, indent=2))
    return (trial_records,)


@app.cell
def _(mo, np, plt, trial_records):
    _xy = np.array([r["cube_xy"] for r in trial_records])
    _ok = np.array([r["success"] for r in trial_records])
    _fig, _ax = plt.subplots(figsize=(4.8, 4))
    _th = np.linspace(-0.7, 0.7, 50)
    for _r in (0.17, 0.27):
        _ax.plot(_r * np.cos(_th), _r * np.sin(_th), color="gray", lw=0.8)
    _ax.scatter(_xy[_ok, 0], _xy[_ok, 1], c="tab:green", label="success")
    _ax.scatter(_xy[~_ok, 0], _xy[~_ok, 1], c="tab:red", marker="x", label="fail")
    _ax.add_patch(plt.Rectangle((0.0, 0.17), 0.1, 0.1, fill=False, color="tab:blue", label="box"))
    _ax.scatter([0], [0], marker="s", color="k", label="robot base")
    _ax.set(aspect="equal", xlabel="x (m)", ylabel="y (m)", title="Initial cube positions")
    _ax.legend(fontsize=8, loc="lower left")
    _ax.grid(alpha=0.3)
    _fig.tight_layout()
    mo.hstack([_fig, mo.md(f"### 성공률 {_ok.sum()}/{len(_ok)} = {_ok.mean() * 100:.0f}%\n\n"
                           "실패가 있다면 어느 쪽 가장자리에 몰려 있는지 보세요. 3-7에서 원인을 찾습니다.")],
              align="center")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3-7. 작업 영역 훑기: 어디까지 집을 수 있나

    블록 위치를 격자로 바꿔 가며 "접근 → 내려가기 → 닫기 → 들기"만 시도하고 IK 오차와 성공 여부를 기록합니다.
    9장의 LLM 도구 계층이 목표를 반지름 0.12~0.30 m로 자르는 근거가 이 실험입니다.
    """)
    return


@app.cell
def _(mo):
    sweep_btn = mo.ui.run_button(label="작업 영역 훑기 실행 (약 20초)")
    sweep_btn
    return (sweep_btn,)


@app.cell
def _(mo, np, pc, plt, sweep_btn):
    mo.stop(not sweep_btn.value, mo.md("버튼을 누르면 8 × 5 = 40개 위치를 시험합니다."))
    _xs = [0.14, 0.16, 0.20, 0.24, 0.28, 0.30, 0.32, 0.34]
    _ys = [-0.12, -0.06, 0.0, 0.06, 0.12]
    _res = np.zeros((len(_ys), len(_xs)))
    _err = np.zeros_like(_res)
    with mo.status.progress_bar(total=len(_xs) * len(_ys), title="작업 영역 훑기") as _bar:
        for _i, _y in enumerate(_ys):
            for _j, _x in enumerate(_xs):
                _robot = pc.SO101Sim(cube_pos=(_x, _y), render=False)
                _robot.open_gripper(0.5)
                _robot.move_to([_x, _y, 0.08], seconds=1.0)
                _, _e = _robot.move_to([_x, _y, 0.02], seconds=0.8)
                _robot.hold(0.2)
                _robot.close_gripper(0.8)
                _robot.move_to([_x, _y, 0.12], seconds=1.0)
                _robot.hold(0.3)
                _res[_i, _j] = _robot.cube_pos()[2] > 0.06
                _err[_i, _j] = _e
                _robot.close()
                _bar.update()
    _fig, _axes = plt.subplots(1, 2, figsize=(10, 3.2))
    for _ax, _m, _t, _cmap in [(_axes[0], _res, "lifted (1 = success)", "RdYlGn"),
                               (_axes[1], np.log10(_err + 1e-6), "log10 IK residual at grasp height", "viridis")]:
        _im = _ax.imshow(_m, origin="lower", cmap=_cmap, aspect="auto")
        _ax.set(xticks=range(len(_xs)), xticklabels=_xs, yticks=range(len(_ys)), yticklabels=_ys,
                xlabel="cube x (m)", ylabel="cube y (m)", title=_t)
        _fig.colorbar(_im, ax=_ax)
    _fig.tight_layout()
    mo.vstack([_fig, mo.md(f"성공 {int(_res.sum())}/{_res.size}. 너무 가깝거나(x ≤ 0.14) 먼 곳(x ≥ 0.32)에서 "
                           "손가락을 수직으로 유지하지 못해 IK 잔차가 커지고 집기에 실패합니다.")])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 정리와 다음 장

    - FK는 `mj_forward` 한 번, IK는 Jacobian으로 오차를 줄이는 반복 계산입니다.
    - Rule 기반 Pick and Place는 블록 위치를 **정확히 알면** 잘 동작합니다. 그 위치를 어디서 알아낼지가 다음 문제입니다.
    - 위치 판정 `is_in_box`, 블록 위치 범위 `random_cube`, 높이 상수 `APPROACH_Z` 등은 이후 장에서
      `physicalai_common`의 같은 이름으로 다시 씁니다.

    **확인 문제**

    1. `GRASP_Z`를 0.03으로 올리면 어떻게 되나요? 0.01로 내리면요? (힌트: 블록 중심은 0.015)
    2. Lift 단계에서 `move_to`의 `seconds`를 0.3으로 줄이면 성공률이 어떻게 변할까요? 직접 바꿔 3-6을 다시 실행해 보세요.

    다음 [4장](ch04_vision.py)에서는 블록 위치를 시뮬레이터에서 읽는 대신 카메라 이미지로 찾습니다.
    """)
    return


if __name__ == "__main__":
    app.run()
