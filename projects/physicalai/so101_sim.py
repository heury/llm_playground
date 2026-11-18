"""SO-101 로봇팔 MuJoCo 시뮬레이터.

장면: 바닥 + 색깔 블록(한 변 3 cm, 기본은 빨강 하나) + 파란 상자 + 카메라("top", "view").
손목 카메라("wrist_cam")는 로봇 모델(so101.xml)에 들어 있습니다.

로봇 모델은 MuJoCo Menagerie에서 처음 한 번만 내려받아 third_party/에 둡니다.
"""

import os
import subprocess

import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MENAGERIE_URL = "https://github.com/google-deepmind/mujoco_menagerie.git"
MENAGERIE_DIR = os.environ.get("MENAGERIE_DIR", os.path.join(HERE, "third_party", "mujoco_menagerie"))

ARM_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
SITE = "gripperframe"       # 손끝 기준점 (두 손가락 사이). 이 site의 x축이 손가락 방향
DOWN = (0.0, 0.0, -1.0)     # 위에서 내려 집을 때 손가락 방향
GRIPPER_OPEN = 1.2          # rad
GRIPPER_CLOSED = -0.1       # rad (약간 더 조이도록 음수)
BOX_XY = (0.05, 0.22)       # 상자 위치 (고정)
CUBE_RGBA = {"red": "0.9 0.1 0.1 1", "green": "0.1 0.7 0.2 1",
             "yellow": "0.95 0.85 0.1 1", "purple": "0.6 0.2 0.8 1"}

CUBE_XML = """
    <body name="cube_{color}" pos="{x} {y} 0.015">
      <freejoint/>
      <geom type="box" size="0.015 0.015 0.015" mass="0.02" rgba="{rgba}"
            friction="1.5 0.005 0.0001" condim="4"/>
    </body>"""

SCENE_XML = """
<mujoco model="so101_pickplace">
  <include file="so101.xml"/>
  <option timestep="0.002" integrator="implicitfast"/>
  <visual>
    <global offwidth="640" offheight="480"/>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0"/>
  </visual>
  <asset>
    <texture type="2d" name="grid" builtin="checker" mark="edge" rgb1="0.85 0.85 0.85"
             rgb2="0.75 0.75 0.75" markrgb="0.6 0.6 0.6" width="300" height="300"/>
    <material name="grid" texture="grid" texuniform="true" texrepeat="8 8" reflectance="0.1"/>
  </asset>
  <worldbody>
    <light pos="0 0 1.5" dir="0 0 -1" directional="true" diffuse="0.6 0.6 0.6"/>
    <geom name="floor" type="plane" size="1 1 0.05" material="grid"/>
{cubes}

    <!-- 상자: 바닥 + 낮은 벽 4개 -->
    <body name="box" pos="{box_x} {box_y} 0">
      <geom type="box" size="0.05 0.05 0.003" pos="0 0 0.003" rgba="0.2 0.4 0.9 1"/>
      <geom type="box" size="0.05 0.003 0.015" pos="0 0.05 0.015" rgba="0.2 0.4 0.9 1"/>
      <geom type="box" size="0.05 0.003 0.015" pos="0 -0.05 0.015" rgba="0.2 0.4 0.9 1"/>
      <geom type="box" size="0.003 0.05 0.015" pos="0.05 0 0.015" rgba="0.2 0.4 0.9 1"/>
      <geom type="box" size="0.003 0.05 0.015" pos="-0.05 0 0.015" rgba="0.2 0.4 0.9 1"/>
    </body>

    <camera name="top" pos="0.2 0 0.8" quat="1 0 0 0"/>
    <!-- 영상 확인용 대각선 카메라 (데이터셋에는 쓰지 않음) -->
    <body name="view_target" pos="0.15 0.05 0.03"/>
    <camera name="view" mode="targetbody" target="view_target" pos="0.65 -0.5 0.4"/>
  </worldbody>

  <keyframe>
    <!-- 대기 자세: 팔을 위로 세워 바닥의 물체와 카메라 시야를 피한다 -->
    <key name="home" qpos="0 -1.5 0.5 0 0 0{cube_qpos}" ctrl="0 -1.5 0.5 0 0 0"/>
  </keyframe>
</mujoco>
"""


def scene_path(colors):
    """so101.xml과 같은 폴더에 장면 XML을 써 두고 경로를 돌려준다.

    MuJoCo는 include·mesh 경로를 메인 XML 파일이 있는 폴더 기준으로 찾기 때문이다.
    """
    robot_dir = os.path.join(MENAGERIE_DIR, "robotstudio_so101")
    if not os.path.isdir(robot_dir):
        print(f"SO-101 모델을 {MENAGERIE_DIR}에 내려받습니다 ...")
        subprocess.run(["git", "clone", "--depth", "1", "--filter=blob:none", "--sparse",
                        MENAGERIE_URL, MENAGERIE_DIR], check=True)
        subprocess.run(["git", "-C", MENAGERIE_DIR, "sparse-checkout", "set", "robotstudio_so101"], check=True)
    # 블록은 처음에 x = 0.25 줄에 6 cm 간격으로 둔다 (reset에서 원하는 위치로 옮긴다)
    places = [(0.25, 0.06 * i) for i in range(len(colors))]
    cubes = "".join(CUBE_XML.format(color=c, x=x, y=y, rgba=CUBE_RGBA[c]) for c, (x, y) in zip(colors, places))
    cube_qpos = "".join(f"  {x} {y} 0.015 1 0 0 0" for x, y in places)
    xml = SCENE_XML.format(cubes=cubes, cube_qpos=cube_qpos, box_x=BOX_XY[0], box_y=BOX_XY[1])
    path = os.path.join(robot_dir, f"pickplace_scene_{'_'.join(colors)}.xml")
    with open(path, "w", encoding="utf-8") as f:
        f.write(xml)
    return path


class SO101Sim:
    """관절 목표를 주고 시간을 흘려보내는 로봇. step()마다 on_step(sim)을 호출한다."""

    def __init__(self, colors=("red",)):
        self.colors = list(colors)
        self.model = mujoco.MjModel.from_xml_path(scene_path(self.colors))
        self.data = mujoco.MjData(self.model)
        m = self.model
        jid = [m.joint(n).id for n in ARM_JOINTS]
        self.arm_qadr = m.jnt_qposadr[jid]
        self.arm_dadr = m.jnt_dofadr[jid]
        self.arm_lo, self.arm_hi = m.jnt_range[jid, 0], m.jnt_range[jid, 1]
        self.arm_act = np.array([m.actuator(n).id for n in ARM_JOINTS])
        self.grip_act = m.actuator("gripper").id
        self.site_id = m.site(SITE).id
        self.cube_qadr = {c: m.jnt_qposadr[m.body(f"cube_{c}").jntadr[0]] for c in self.colors}
        self.on_step = None
        self.reset()

    def reset(self, cube_xy=None):
        """대기 자세로 돌아가고 블록을 놓는다. cube_xy: {색: (x, y)}, 없는 색은 처음 자리."""
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        for color, xy in (cube_xy or {}).items():
            a = self.cube_qadr[color]
            self.data.qpos[a:a + 7] = [*xy, 0.015, 1, 0, 0, 0]
        mujoco.mj_forward(self.model, self.data)
        self.n_steps = 0

    # ---------- 상태 읽기 ----------
    def cube_pos(self, color="red"):
        return self.data.body(f"cube_{color}").xpos.copy()

    def box_pos(self):
        return self.data.body("box").xpos.copy()

    # ---------- 시간 진행 ----------
    def step(self, n=1):
        for _ in range(n):
            if self.on_step is not None:
                self.on_step(self)
            mujoco.mj_step(self.model, self.data)
            self.n_steps += 1

    def hold(self, seconds):
        self.step(int(seconds / self.model.opt.timestep))

    def move_arm(self, q_target, seconds=1.0):
        """현재 명령값에서 목표 관절각까지 선형 보간하며 이동한다."""
        start = self.data.ctrl[self.arm_act].copy()
        n = max(1, int(seconds / self.model.opt.timestep))
        for i in range(1, n + 1):
            a = i / n
            self.data.ctrl[self.arm_act] = (1 - a) * start + a * q_target
            self.step()

    def set_gripper(self, value, seconds):
        self.data.ctrl[self.grip_act] = value
        self.hold(seconds)

    def move_to(self, target_pos, seconds=1.0):
        """손끝을 target_pos로, 손가락은 아래를 향하게 이동한다."""
        self.move_arm(self.ik(target_pos), seconds)

    # ---------- Inverse Kinematics ----------
    def ik(self, target_pos, target_dir=DOWN):
        """손끝을 target_pos로, 손가락(site의 x축)을 target_dir로 향하게 하는 팔 관절각 5개.

        수치 IK는 시작 자세에 따라 관절 한계에 걸려 멈출 수 있다. 그래서 현재 자세와
        대기 자세(home) 두 곳에서 각각 풀어 보고 손끝 위치 오차가 작은 쪽을 고른다.
        """
        home = self.model.key_qpos[0][self.arm_qadr]
        q1, err1 = self._solve_ik(self.data.qpos[self.arm_qadr], target_pos, target_dir)
        q2, err2 = self._solve_ik(home, target_pos, target_dir)
        return q1 if err1 <= err2 else q2

    def _solve_ik(self, q_start, target_pos, target_dir, iters=300, tol=1e-3, damping=1e-2, step=0.5):
        """Damped Least Squares: dq = Jᵀ (J Jᵀ + λ²I)⁻¹ err 를 반복한다. (관절각, 위치 오차)

        실제 상태를 건드리지 않도록 MjData 사본에서 운동학(mj_forward)만 계산한다.
        """
        m = self.model
        d = mujoco.MjData(m)
        d.qpos[:] = self.data.qpos
        d.qpos[self.arm_qadr] = q_start
        target_pos = np.asarray(target_pos, dtype=float)
        target_dir = np.asarray(target_dir, dtype=float)
        jacp, jacr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        for _ in range(iters):
            mujoco.mj_forward(m, d)
            finger_dir = d.site_xmat[self.site_id].reshape(3, 3)[:, 0]
            err = np.concatenate([target_pos - d.site_xpos[self.site_id],   # 위치 오차
                                  np.cross(finger_dir, target_dir)])         # 방향 오차
            if np.linalg.norm(err) < tol:
                break
            mujoco.mj_jacSite(m, d, jacp, jacr, self.site_id)
            J = np.vstack([jacp, jacr])[:, self.arm_dadr]
            dq = J.T @ np.linalg.solve(J @ J.T + damping**2 * np.eye(6), err)
            q = d.qpos[self.arm_qadr] + step * dq
            d.qpos[self.arm_qadr] = np.clip(q, self.arm_lo, self.arm_hi)
        mujoco.mj_forward(m, d)
        return d.qpos[self.arm_qadr].copy(), np.linalg.norm(target_pos - d.site_xpos[self.site_id])
