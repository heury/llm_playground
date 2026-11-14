"""Physical AI 실습 노트북의 공통 도우미 모듈.

원본: 《로봇 없이 시작하는 Physical AI 따라하기》 예제 코드
(https://github.com/ady95/physicalai_tutorial, MIT License)의 common/ 과
여러 장에서 다시 쓰는 함수(3장 Pick and Place, 6장 시연 기록 등)를 한 파일로 모았습니다.

노트북에서 직접 만들어 보는 코드는 각 장의 셀에 있고, 이 모듈은 다음 장에서 그 결과를
다시 쓸 때 import 합니다. 예를 들어 3장 노트북에서 만든 pick_and_place()와 같은 동작을
이 모듈의 scripted_pick_and_place()가 합니다.

이 모듈은 실습 노트북이 아닙니다. 노트북에서 `import physicalai_common as pc`로 씁니다.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

# MuJoCo는 import 시점에 렌더링 백엔드를 고릅니다. 화면이 없는 Linux에서는 EGL을 씁니다.
# 사용자가 MUJOCO_GL을 직접 지정했다면 그 값을 유지합니다.
if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
    os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("SVT_LOG", "0")  # LeRobot 영상 인코더(SVT-AV1)의 장황한 로그 끄기

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

# ---------------------------------------------------------------------------
# 경로
# ---------------------------------------------------------------------------
NOTEBOOK_DIR = Path(__file__).resolve().parent
REPO_ROOT = NOTEBOOK_DIR.parents[1]
OUTPUT_ROOT = REPO_ROOT / "outputs" / "physicalai"
CHECKPOINT_DIR = OUTPUT_ROOT / "checkpoints"   # 릴리스 체크포인트·학습 결과 (여러 실행이 공유)
DATASET_DIR = OUTPUT_ROOT / "datasets"         # 기록한 LeRobot 데이터셋 (여러 실행이 공유)


def make_run_dir(tag: str) -> Path:
    """outputs/physicalai/<날짜-시간-tag>/ 실행 폴더를 만들어 돌려준다."""
    path = OUTPUT_ROOT / f"{datetime.now():%Y%m%d-%H%M%S}-{tag}"
    path.mkdir(parents=True, exist_ok=True)
    return path


# ---------------------------------------------------------------------------
# MuJoCo Menagerie (SO-101 로봇 모델)
# ---------------------------------------------------------------------------
MENAGERIE_URL = "https://github.com/google-deepmind/mujoco_menagerie.git"
DEFAULT_MENAGERIE_DIR = NOTEBOOK_DIR / "third_party" / "mujoco_menagerie"
ROBOT_DIRS = ["robotstudio_so101"]


def menagerie_dir() -> Path:
    """Menagerie 경로. 없으면 SO-101 폴더만 sparse checkout으로 내려받는다(수십 MB, 처음 한 번).

    이미 받아 둔 사본이 있으면 환경변수 MENAGERIE_DIR로 지정하세요.
    """
    path = Path(os.environ.get("MENAGERIE_DIR", DEFAULT_MENAGERIE_DIR))
    if not (path / "robotstudio_so101").is_dir():
        print(f"[menagerie] {path}에 SO-101 모델이 없어 내려받습니다 ...", flush=True)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.is_dir():
            subprocess.run(["git", "clone", "--depth", "1", "--filter=blob:none", "--sparse",
                            MENAGERIE_URL, str(path)], check=True)
        subprocess.run(["git", "-C", str(path), "sparse-checkout", "set", *ROBOT_DIRS], check=True)
    return path


def so101_xml(scene: str = "scene.xml") -> str:
    """SO-101 MJCF 경로. "scene.xml"은 바닥이 포함된 장면, "so101.xml"은 로봇만."""
    return str(menagerie_dir() / "robotstudio_so101" / scene)


# ---------------------------------------------------------------------------
# 장면: SO-101 + 바닥 + 블록 + 상자 + 카메라
# ---------------------------------------------------------------------------
SCENE_TEMPLATE = """
<mujoco model="physicalai_scene">
  <include file="so101.xml"/>

  <option timestep="0.002" integrator="implicitfast"/>
  <visual>
    <global offwidth="640" offheight="480"/>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0"/>
  </visual>

  <asset>
    <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="0.85 0.85 0.85"
             rgb2="0.75 0.75 0.75" markrgb="0.6 0.6 0.6" width="300" height="300"/>
    <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="8 8" reflectance="0.1"/>
  </asset>

  <worldbody>
    <light pos="{light_pos}" dir="0 0 -1" directional="true" diffuse="{light_diffuse}"/>
    <geom name="floor" type="plane" size="1 1 0.05" material="groundplane" rgba="{floor_rgba}"/>

    <!-- 집을 물체: 빨간 블록 (한 변 3 cm) -->
    <body name="cube" pos="{cube_x} {cube_y} 0.015">
      <freejoint name="cube_free"/>
      <geom name="cube_geom" type="box" size="0.015 0.015 0.015" mass="0.02"
            rgba="{cube_rgba}" friction="1.5 0.005 0.0001" condim="4"/>
    </body>
{extra_cubes}

    <!-- 놓을 곳: 상자 (바닥 + 낮은 벽 4개) -->
    <body name="box" pos="{box_x} {box_y} 0">
      <geom type="box" size="0.05 0.05 0.003" pos="0 0 0.003" rgba="0.2 0.4 0.9 1"/>
      <geom type="box" size="0.05 0.003 0.015" pos="0 0.05 0.015" rgba="0.2 0.4 0.9 1"/>
      <geom type="box" size="0.05 0.003 0.015" pos="0 -0.05 0.015" rgba="0.2 0.4 0.9 1"/>
      <geom type="box" size="0.003 0.05 0.015" pos="0.05 0 0.015" rgba="0.2 0.4 0.9 1"/>
      <geom type="box" size="0.003 0.05 0.015" pos="-0.05 0 0.015" rgba="0.2 0.4 0.9 1"/>
    </body>

    <!-- 카메라: 위에서(top), 확대(top_zoom), 고정 관찰용(fixed), 블록을 따라가는 옆·정면 카메라 -->
    <camera name="top" pos="{top_cam_pos}" quat="1 0 0 0"/>
    <camera name="top_zoom" pos="0.15 0.05 0.8" quat="1 0 0 0" fovy="34"/>
    <body name="cam_target" pos="0.15 0.05 0.03"/>
    <camera name="fixed" mode="targetbody" target="cam_target" pos="0.65 -0.5 0.4"/>
    <camera name="side" mode="targetbody" target="cube" pos="0.55 -0.55 0.35"/>
    <camera name="front" mode="targetbody" target="cube" pos="0.75 0 0.3"/>
  </worldbody>

  <keyframe>
    <!-- 대기 자세: 팔을 위로 세워 손을 41 cm 높이에 둔다 -->
    <key name="home" qpos="0 -1.5 0.5 0 0 0 {cube_x} {cube_y} 0.015 1 0 0 0{extra_qpos}" ctrl="0 -1.5 0.5 0 0 0"/>
  </keyframe>
</mujoco>
"""

CUBE_COLORS = {"red": "0.9 0.1 0.1 1", "green": "0.1 0.7 0.2 1", "yellow": "0.95 0.85 0.1 1"}

EXTRA_CUBE_TEMPLATE = """
    <body name="cube_{name}" pos="{x} {y} 0.015">
      <freejoint name="cube_{name}_free"/>
      <geom name="cube_{name}_geom" type="box" size="0.015 0.015 0.015" mass="0.02"
            rgba="{rgba}" friction="1.5 0.005 0.0001" condim="4"/>
    </body>"""


def scene_xml(cube_pos=(0.25, 0.0), box_pos=(0.0, 0.25), extra_cubes=None,
              light_pos=(0.0, 0.0, 1.5), light_diffuse=(0.6, 0.6, 0.6), floor_rgba=(1, 1, 1, 1),
              top_cam_pos=(0.2, 0.0, 0.8), cube_rgba=(0.9, 0.1, 0.1, 1)) -> str:
    """장면 MJCF 문자열.

    extra_cubes: [("green", (x, y)), ("yellow", (x, y))] 처럼 추가 블록 (8장).
    light_pos / light_diffuse / floor_rgba / top_cam_pos / cube_rgba: 8장 환경 변화 실험용.
    """
    extra_xml, extra_qpos = "", ""
    for name, (x, y) in (extra_cubes or []):
        extra_xml += EXTRA_CUBE_TEMPLATE.format(name=name, x=x, y=y, rgba=CUBE_COLORS[name])
        extra_qpos += f" {x} {y} 0.015 1 0 0 0"
    fmt = lambda v: " ".join(f"{float(a):g}" for a in v)  # noqa: E731
    return SCENE_TEMPLATE.format(cube_x=cube_pos[0], cube_y=cube_pos[1],
                                 box_x=box_pos[0], box_y=box_pos[1],
                                 extra_cubes=extra_xml, extra_qpos=extra_qpos,
                                 light_pos=fmt(light_pos), light_diffuse=fmt(light_diffuse),
                                 floor_rgba=fmt(floor_rgba), top_cam_pos=fmt(top_cam_pos),
                                 cube_rgba=fmt(cube_rgba))


def load_scene(**scene_kw):
    """장면 MjModel을 만든다.

    MuJoCo는 include·mesh 경로를 메인 XML 파일의 폴더 기준으로 찾기 때문에 장면 XML을 so101.xml 옆에
    임시 파일로 쓰고, 컴파일한 뒤 바로 지운다(블록 위치마다 파일이 쌓이지 않도록).
    """
    robot_dir = os.path.dirname(so101_xml("so101.xml"))
    fd, tmp = tempfile.mkstemp(prefix="physicalai_scene_", suffix=".xml", dir=robot_dir)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(scene_xml(**scene_kw))
        return mujoco.MjModel.from_xml_path(tmp)
    finally:
        os.remove(tmp)


# ---------------------------------------------------------------------------
# 수치 IK (Damped Least Squares)
# ---------------------------------------------------------------------------
def solve_ik(model, data, site_name, target_pos, target_dir=None, arm_joint_names=None,
             axis=0, max_iters=300, tol=1e-3, damping=1e-2, step=0.5):
    """site 위치를 target_pos로, (선택) site의 axis축을 target_dir로 맞추는 관절각과 최종 오차.

    MuJoCo의 Jacobian(mj_jacSite)으로 "관절을 조금 움직이면 손끝이 어디로 가는가"를 선형 근사하고
    dq = J^T (J J^T + λ²I)^-1 err 로 관절을 조금씩 고친다. 물리 시뮬레이션은 돌리지 않는다.
    """
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in arm_joint_names]
    qpos_adr = np.array([model.jnt_qposadr[j] for j in joint_ids])
    dof_adr = np.array([model.jnt_dofadr[j] for j in joint_ids])
    lo = model.jnt_range[joint_ids, 0]
    hi = model.jnt_range[joint_ids, 1]

    scratch = mujoco.MjData(model)          # 실제 시뮬레이션 상태를 건드리지 않기 위한 사본
    scratch.qpos[:] = data.qpos
    target_pos = np.asarray(target_pos, dtype=float)
    if target_dir is not None:
        target_dir = np.asarray(target_dir, dtype=float)
        target_dir = target_dir / np.linalg.norm(target_dir)

    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    err_norm = np.inf
    for _ in range(max_iters):
        mujoco.mj_forward(model, scratch)
        pos = scratch.site_xpos[site_id]
        err = target_pos - pos                                   # 위치 오차 (3)
        if target_dir is not None:
            dir_cur = scratch.site_xmat[site_id].reshape(3, 3)[:, axis]
            err = np.concatenate([err, np.cross(dir_cur, target_dir)])   # 방향 오차 (3)
        err_norm = np.linalg.norm(err)
        if err_norm < tol:
            break
        mujoco.mj_jacSite(model, scratch, jacp, jacr, site_id)
        J = jacp[:, dof_adr] if target_dir is None else np.vstack([jacp, jacr])[:, dof_adr]
        JJt = J @ J.T
        dq = J.T @ np.linalg.solve(JJt + damping**2 * np.eye(JJt.shape[0]), err)
        scratch.qpos[qpos_adr] = np.clip(scratch.qpos[qpos_adr] + step * dq, lo, hi)
    return scratch.qpos[qpos_adr].copy(), err_norm


# ---------------------------------------------------------------------------
# SO-101 시뮬레이션 로봇 래퍼
# ---------------------------------------------------------------------------
ARM_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
GRIPPER_JOINT = "gripper"
SITE = "gripperframe"           # 손끝 기준점 (두 손가락 사이). x축이 손가락이 향하는 방향
DOWN = (0.0, 0.0, -1.0)         # 위에서 내려 집을 때 손가락 방향
GRIPPER_OPEN = 1.2              # rad
GRIPPER_CLOSED = -0.1           # rad (약간 더 조이도록 음수)


class SO101Sim:
    """관절 목표를 주고 시간을 흘려보내고(move_arm), 위치를 읽고(site_pose, cube_pos), 영상을 찍는다."""

    def __init__(self, cube_pos=(0.25, 0.0), box_pos=(0.0, 0.25), render=True,
                 camera="fixed", fps=30, width=640, height=480, **scene_kw):
        self.model = load_scene(cube_pos=cube_pos, box_pos=box_pos, **scene_kw)
        self.data = mujoco.MjData(self.model)
        m = self.model
        jid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)  # noqa: E731
        self.arm_qadr = np.array([m.jnt_qposadr[jid(n)] for n in ARM_JOINTS])
        self.arm_act = np.array([mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in ARM_JOINTS])
        self.grip_act = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, GRIPPER_JOINT)
        self.grip_qadr = m.jnt_qposadr[jid(GRIPPER_JOINT)]
        self.site_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, SITE)
        self.cube_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "cube")
        self.box_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "box")
        self.fps = fps
        self.steps_per_frame = int(round(1 / (fps * m.opt.timestep)))
        self.frames = []
        self.camera = camera
        self.renderer = mujoco.Renderer(m, height=height, width=width) if render else None
        self.reset()

    # ----- 상태 -----
    def reset(self):
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)   # keyframe "home"
        mujoco.mj_forward(self.model, self.data)
        self._step_count = 0

    @property
    def q_arm(self):
        return self.data.qpos[self.arm_qadr].copy()

    def site_pose(self):
        """손끝 기준점의 위치(3)와 회전행렬(3x3)."""
        return self.data.site_xpos[self.site_id].copy(), self.data.site_xmat[self.site_id].reshape(3, 3).copy()

    def cube_pos(self, name=None):
        """블록 위치. name이 None/"red"이면 빨간 블록, 아니면 cube_<name>."""
        if name is None or name == "red":
            return self.data.xpos[self.cube_id].copy()
        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"cube_{name}")
        return self.data.xpos[bid].copy()

    def box_pos(self):
        return self.data.xpos[self.box_id].copy()

    def gripper_opening(self):
        return float(self.data.qpos[self.grip_qadr])

    def finger_dir(self):
        return self.site_pose()[1][:, 0]

    # ----- 시간 진행 -----
    def step(self, n=1):
        for _ in range(n):
            mujoco.mj_step(self.model, self.data)
            self._step_count += 1
            if self.renderer is not None and self._step_count % self.steps_per_frame == 0:
                self.renderer.update_scene(self.data, camera=self.camera)
                self.frames.append(self.renderer.render().copy())

    def hold(self, seconds):
        self.step(int(seconds / self.model.opt.timestep))

    def move_arm(self, q_target, seconds=1.0):
        """현재 ctrl에서 목표 관절각까지 선형 보간하며 이동."""
        q_target = np.asarray(q_target, dtype=float)
        start = self.data.ctrl[self.arm_act].copy()
        n = max(1, int(seconds / self.model.opt.timestep))
        for i in range(n):
            a = (i + 1) / n
            self.data.ctrl[self.arm_act] = (1 - a) * start + a * q_target
            self.step()

    def set_gripper(self, value, seconds=0.5):
        self.data.ctrl[self.grip_act] = value
        self.hold(seconds)

    def open_gripper(self, seconds=0.5):
        self.set_gripper(GRIPPER_OPEN, seconds)

    def close_gripper(self, seconds=0.7):
        self.set_gripper(GRIPPER_CLOSED, seconds)

    # ----- IK -----
    def ik(self, target_pos, target_dir=DOWN, **kw):
        return solve_ik(self.model, self.data, SITE, target_pos, target_dir, ARM_JOINTS, axis=0, **kw)

    def move_to(self, target_pos, target_dir=DOWN, seconds=1.0, **kw):
        q, err = self.ik(target_pos, target_dir, **kw)
        self.move_arm(q, seconds)
        return q, err

    # ----- 영상 -----
    def save_video(self, path):
        if not self.frames:
            return None
        return save_video(self.frames, path, fps=self.fps)

    def snapshot(self, camera=None, width=640, height=480):
        r = mujoco.Renderer(self.model, height=height, width=width)
        try:
            r.update_scene(self.data, camera=camera or self.camera)
            return r.render().copy()
        finally:
            r.close()

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None


def save_video(frames, path, fps=30):
    """프레임 목록을 mp4(H.264)로 저장하고 마지막 프레임을 _last.png로 함께 저장한다."""
    import imageio

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(path, frames, fps=fps, macro_block_size=8)
    imageio.imwrite(path.with_name(path.stem + "_last.png"), frames[-1])
    return path


# ---------------------------------------------------------------------------
# 가상 카메라와 색 검출 (4장)
# ---------------------------------------------------------------------------
COLOR_RANGES = {   # HSV 범위 (OpenCV: H 0~179, S 0~255, V 0~255)
    "red": [((0, 120, 70), (10, 255, 255)), ((170, 120, 70), (179, 255, 255))],
    "blue": [((100, 120, 70), (130, 255, 255))],
    "yellow": [((20, 120, 70), (35, 255, 255))],
    "green": [((40, 80, 70), (80, 255, 255))],
}


class SimCamera:
    """MuJoCo 카메라 하나를 RGB + Depth로 찍는 객체."""

    def __init__(self, model, camera="top", width=640, height=480):
        self.model = model
        self.cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera)
        self.width, self.height = width, height
        self.rgb_renderer = mujoco.Renderer(model, height=height, width=width)
        self.depth_renderer = mujoco.Renderer(model, height=height, width=width)
        self.depth_renderer.enable_depth_rendering()
        fovy = np.radians(model.cam_fovy[self.cam_id])
        self.f = (height / 2) / np.tan(fovy / 2)          # 초점 거리 (픽셀)
        self.cx, self.cy = width / 2, height / 2

    def capture(self, data):
        """(rgb [H,W,3] uint8, depth [H,W] float m)."""
        self.rgb_renderer.update_scene(data, camera=self.cam_id)
        rgb = self.rgb_renderer.render().copy()
        self.depth_renderer.update_scene(data, camera=self.cam_id)
        depth = self.depth_renderer.render().copy()
        return rgb, depth

    def pixel_to_world(self, data, u, v, depth):
        """픽셀 (u, v)와 depth(카메라 축 방향 거리) → 월드 좌표."""
        p_cam = np.array([(u - self.cx) / self.f * depth,
                          -(v - self.cy) / self.f * depth,    # 이미지 v는 아래가 +, 카메라 y는 위가 +
                          -depth])                            # MuJoCo 카메라는 -z를 본다
        R = data.cam_xmat[self.cam_id].reshape(3, 3)
        return R @ p_cam + data.cam_xpos[self.cam_id]

    def close(self):
        self.rgb_renderer.close()
        self.depth_renderer.close()


def color_mask(rgb, color):
    import cv2

    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lo, hi in COLOR_RANGES[color]:
        mask |= cv2.inRange(hsv, np.array(lo), np.array(hi))
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))


def find_color_center(rgb, color, min_area=20):
    """RGB 이미지에서 color의 가장 큰 덩어리 중심 (u, v), 면적, bbox. 없으면 None."""
    import cv2

    mask = color_mask(rgb, color)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(c)
    if area < min_area:
        return None
    m = cv2.moments(c)
    return {"center": (m["m10"] / m["m00"], m["m01"] / m["m00"]), "area": area,
            "bbox": cv2.boundingRect(c), "mask": mask}


def locate_object(cam, data, color, rgb=None, depth=None):
    """카메라로 color 물체의 월드 좌표를 추정한다. (world_xyz, 검출 정보) 또는 (None, None)."""
    if rgb is None:
        rgb, depth = cam.capture(data)
    det = find_color_center(rgb, color)
    if det is None:
        return None, None
    u, v = det["center"]
    ui, vi = int(round(u)), int(round(v))
    patch = depth[max(vi - 1, 0): vi + 2, max(ui - 1, 0): ui + 2]   # 가장자리 배경 depth를 피하는 중앙값
    return cam.pixel_to_world(data, u, v, float(np.median(patch))), det


# ---------------------------------------------------------------------------
# 3장 Pick and Place (이후 장에서 시연자·판정 기준으로 재사용)
# ---------------------------------------------------------------------------
APPROACH_Z = 0.08     # 블록 위 8 cm에서 접근
GRASP_Z = 0.02        # 블록 중심(0.015)보다 5 mm 위
LIFT_Z = 0.12         # 들어 올리는 높이
PLACE_Z = 0.06        # 상자 위에서 놓는 높이
BOX_XY = (0.05, 0.22)


def above(xy, z):
    return np.array([xy[0], xy[1], z])


def is_in_box(cube_pos, box_pos, half=0.05):
    """블록이 상자 안(가로·세로 5 cm 이내, 높이 5 cm 미만)에 있으면 성공."""
    dx, dy = abs(cube_pos[0] - box_pos[0]), abs(cube_pos[1] - box_pos[1])
    return dx < half and dy < half and cube_pos[2] < 0.05


def random_cube(rng):
    """로봇 앞 부채꼴(반지름 0.17~0.27 m, 각도 ±0.7 rad)에서 블록 위치를 뽑는다."""
    r, th = rng.uniform(0.17, 0.27), rng.uniform(-0.7, 0.7)
    return (float(r * np.cos(th)), float(r * np.sin(th)))


def scripted_pick_and_place(robot, color=None, target=None, box=None, demo=False):
    """Approach → Grasp → Lift → Move → Place. 성공 여부를 돌려준다.

    target: 집을 좌표. 주지 않으면 시뮬레이터의 정답 위치(3장), 주면 카메라 추정값(4장) 등을 쓴다.
    demo=True: 6·8장 시연 기록과 같은 시간 배분(중간 정지 생략). 공개 체크포인트가 이 궤적으로 학습되었다.
    """
    cube = robot.cube_pos(color) if target is None else np.asarray(target)
    box = robot.box_pos() if box is None else np.asarray(box)
    pause = (lambda s: None) if demo else robot.hold   # noqa: E731
    robot.open_gripper(0.5)
    robot.move_to(above(cube, APPROACH_Z), seconds=1.2)
    robot.move_to(above(cube, GRASP_Z), seconds=0.8)
    robot.hold(0.2)
    robot.close_gripper(0.8)
    robot.move_to(above(cube, LIFT_Z), seconds=1.0)
    pause(0.2)
    if robot.cube_pos(color)[2] < 0.06:
        return False
    robot.move_to(above(box, LIFT_Z), seconds=1.5)
    pause(0.2)
    robot.move_to(above(box, PLACE_Z), seconds=0.8)
    robot.hold(0.2)
    robot.open_gripper(0.5)
    robot.move_to(above(box, LIFT_Z), seconds=0.8)
    robot.hold(0.3 if demo else 0.5)
    return bool(is_in_box(robot.cube_pos(color), robot.box_pos()))


# ---------------------------------------------------------------------------
# 6~8장: 시연 기록과 정책 실행
# ---------------------------------------------------------------------------
FPS = 20
IMG_H, IMG_W = 240, 320
TASK = "Pick up the red cube and put it in the blue box."
JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
COLORS = ["red", "green", "yellow"]
TASK_FMT = "Pick up the {color} cube and put it in the blue box."
ACT_KEYS = {"top": "observation.images.top", "wrist": "observation.images.wrist"}
SMOLVLA_KEYS = {"top": "observation.images.camera1", "wrist": "observation.images.camera2"}

DATASET_FEATURES = {
    "observation.state": {"dtype": "float32", "shape": (6,), "names": JOINT_NAMES},
    "action": {"dtype": "float32", "shape": (6,), "names": JOINT_NAMES},
    "observation.images.top": {"dtype": "video", "shape": (IMG_H, IMG_W, 3),
                               "names": ["height", "width", "channels"]},
    "observation.images.wrist": {"dtype": "video", "shape": (IMG_H, IMG_W, 3),
                                 "names": ["height", "width", "channels"]},
}


class RecordingRobot(SO101Sim):
    """SO101Sim과 같지만 1/FPS 초마다 (관측, 행동) 프레임을 dataset.add_frame()으로 기록한다."""

    def __init__(self, dataset=None, task=TASK, top_camera="top", **kw):
        kw.setdefault("render", False)
        super().__init__(**kw)
        self.dataset = dataset
        self.task = task
        self.cams = {"top": SimCamera(self.model, top_camera, IMG_W, IMG_H),
                     "wrist": SimCamera(self.model, "wrist_cam", IMG_W, IMG_H)}
        self.record_every = int(round(1 / (FPS * self.model.opt.timestep)))
        self.n_recorded = 0

    def observe(self):
        state = self.data.qpos[:6].astype(np.float32)
        imgs = {k: cam.capture(self.data)[0] for k, cam in self.cams.items()}
        return state, imgs

    def step(self, n=1):
        for _ in range(n):
            if self.dataset is not None and self._step_count % self.record_every == 0:
                state, imgs = self.observe()
                self.dataset.add_frame({
                    "observation.state": state,
                    "observation.images.top": imgs["top"],
                    "observation.images.wrist": imgs["wrist"],
                    "action": self.data.ctrl[:6].astype(np.float32),   # 지금 명령 중인 목표 각도
                    "task": self.task,
                })
                self.n_recorded += 1
            super().step(1)

    def close(self):
        for cam in self.cams.values():
            cam.close()
        super().close()


def random_layout(rng, n=3, min_gap=0.07):
    """작업 영역 안에서 서로 7 cm 이상 떨어진 위치 n개 (8장 3색 블록)."""
    pts = []
    while len(pts) < n:
        r, th = rng.uniform(0.17, 0.27), rng.uniform(-0.7, 0.7)
        p = np.array([r * np.cos(th), r * np.sin(th)])
        if all(np.linalg.norm(p - q) > min_gap for q in pts):
            pts.append(p)
    return [tuple(map(float, p)) for p in pts]


def make_observation(robot, cams, image_keys, task, device):
    """정책 입력 dict: 관절각 6개 + 카메라 이미지 (C,H,W) 0~1 + 명령 문장."""
    import torch

    state = torch.from_numpy(robot.data.qpos[:6].astype(np.float32))
    obs = {"observation.state": state.unsqueeze(0).to(device), "task": [task]}
    for k, cam in cams.items():
        rgb, _ = cam.capture(robot.data)
        obs[image_keys[k]] = (torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0).unsqueeze(0).to(device)
    return obs


def run_policy_episode(policy, preprocess, postprocess, device, cube=None, layout=None, color="red",
                       task=None, max_seconds=12.0, video=False, image_keys=ACT_KEYS,
                       scene_kw=None, top_camera="top", box=BOX_XY, on_step=None, stop_on_success=True):
    """학습한 정책(ACT/SmolVLA)으로 episode 하나를 닫힌 고리로 실행한다.

    layout(빨강, 초록, 노랑 위치)을 주면 3색 블록 장면이 되고 color 블록을 상자에 넣어야 성공이다.
    on_step(robot)을 주면 매 프레임 호출한다(손끝 궤적 기록 등).
    stop_on_success=False이면 성공해도 max_seconds까지 계속 실행한다(7장 문장 비교).
    돌려주는 값: {"result": "성공"|"오답(...)"|"실패", "success": bool, "t": 초, "frames": [...]}.
    """
    import torch

    scene_kw = dict(scene_kw or {})
    if layout is not None:
        cube = layout[0]
        scene_kw["extra_cubes"] = [("green", layout[1]), ("yellow", layout[2])]
    task = task or (TASK_FMT.format(color=color) if layout is not None else TASK)
    robot = SO101Sim(cube_pos=cube, box_pos=box, render=video, camera="fixed", **scene_kw)
    cams = {"top": SimCamera(robot.model, top_camera, IMG_W, IMG_H),
            "wrist": SimCamera(robot.model, "wrist_cam", IMG_W, IMG_H)}
    colors = COLORS if layout is not None else ["red"]
    steps_per_frame = int(round(1 / (FPS * robot.model.opt.timestep)))
    policy.reset()
    out = {"result": "실패", "success": False, "t": None}
    try:
        for i in range(int(max_seconds * FPS)):
            obs = preprocess(make_observation(robot, cams, image_keys, task, device))
            with torch.inference_mode():
                action = postprocess(policy.select_action(obs))
            robot.data.ctrl[:6] = action.squeeze(0).cpu().numpy()
            robot.step(steps_per_frame)
            if on_step is not None:
                on_step(robot)
            in_box = [c for c in colors if is_in_box(robot.cube_pos(c), robot.box_pos())]
            if in_box and out["t"] is None:
                ok = in_box == [color]
                out = {"result": "성공" if ok else f"오답({','.join(in_box)})", "success": ok,
                       "t": (i + 1) / FPS}
                if stop_on_success:
                    robot.hold(0.5)
                    break
        out["frames"] = robot.frames
        out["final_site"] = robot.site_pose()[0]
        return out
    finally:
        for cam in cams.values():
            cam.close()
        robot.close()


def load_act(checkpoint, device, n_action_steps=None, ensemble=None):
    """ACT 체크포인트와 전처리기·후처리기(정규화 통계 포함)를 불러온다.

    n_action_steps: chunk 중 실제로 실행할 스텝 수(기본: 학습 설정값).
    ensemble: Temporal Ensembling 계수(예: 0.01). 주면 매 스텝 다시 추론해 겹치는 행동을 가중 평균한다.
    """
    from lerobot.policies import make_pre_post_processors
    from lerobot.policies.act import ACTPolicy
    from lerobot.policies.act.modeling_act import ACTTemporalEnsembler

    checkpoint = str(checkpoint)
    policy = ACTPolicy.from_pretrained(checkpoint).to(device).eval()
    if ensemble is not None:
        policy.config.temporal_ensemble_coeff = ensemble
        policy.config.n_action_steps = 1
        policy.temporal_ensembler = ACTTemporalEnsembler(ensemble, policy.config.chunk_size)
    elif n_action_steps is not None:
        policy.config.n_action_steps = n_action_steps
    preprocess, postprocess = make_pre_post_processors(
        policy.config, checkpoint, preprocessor_overrides={"device_processor": {"device": str(device)}})
    return policy, preprocess, postprocess


def load_smolvla(checkpoint, device):
    """SmolVLA 체크포인트(Hub 이름 또는 폴더)와 전처리기·후처리기를 불러온다."""
    from lerobot.policies import make_pre_post_processors
    from lerobot.policies.smolvla import SmolVLAPolicy

    checkpoint = str(checkpoint)
    policy = SmolVLAPolicy.from_pretrained(checkpoint).to(device).eval()
    preprocess, postprocess = make_pre_post_processors(
        policy.config, checkpoint, preprocessor_overrides={"device_processor": {"device": str(device)}})
    return policy, preprocess, postprocess


def stream_command(command, log_path, cwd=None, env=None, on_line=None):
    """명령을 실행하며 출력을 log_path에 쓰고, 줄마다 on_line(line)을 부른다. 종료 코드를 돌려준다."""
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    full_env = {**os.environ, "PYTHONUNBUFFERED": "1", **(env or {})}
    with log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + " ".join(map(str, command)) + "\n")
        proc = subprocess.Popen([str(c) for c in command], cwd=cwd, env=full_env, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in proc.stdout:
            log.write(line)
            log.flush()
            if on_line is not None:
                on_line(line)
        return proc.wait()


def seed_noise(seed):
    """SmolVLA 행동 생성(Flow Matching)의 시작 noise를 고정한다."""
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# 릴리스 체크포인트 (6~9장)
# ---------------------------------------------------------------------------
RELEASE_BASE = "https://github.com/ady95/physicalai_tutorial/releases/download"
RELEASE_FILES = {
    "act_so101_v2_10k": ("v0.1-checkpoints", "act_so101_v2_10k.tar.gz", "약 190 MB"),
    "smolvla_so101_v2_10k": ("v0.1-checkpoints", "smolvla_so101_v2_10k.tar.gz", "약 720 MB"),
    "smolvla_multicolor_v2_20k": ("v0.1-checkpoints", "smolvla_multicolor_v2_20k.tar.gz", "약 720 MB"),
    "ch09_traces": ("v0.2-ch09-traces", "ch09_traces.tar.gz", "약 90 KB"),
}


def download_release(name: str) -> Path:
    """릴리스 압축 파일을 outputs/physicalai/checkpoints/<name>/ 에 내려받아 푼다(이미 있으면 건너뜀)."""
    import tarfile
    import urllib.request

    tag, filename, _ = RELEASE_FILES[name]
    target = CHECKPOINT_DIR / name
    if target.is_dir() and any(target.iterdir()):
        return target
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    archive = CHECKPOINT_DIR / filename
    if not archive.is_file():
        tmp = archive.with_suffix(".part")
        print(f"내려받는 중: {RELEASE_BASE}/{tag}/{filename}", flush=True)
        urllib.request.urlretrieve(f"{RELEASE_BASE}/{tag}/{filename}", tmp)
        tmp.replace(archive)
    target.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive) as tar:
        tar.extractall(target, filter="data")
    return target


def find_pretrained_dirs(root: Path) -> list[Path]:
    """root 아래에서 config.json과 model.safetensors가 함께 있는 폴더(정책 체크포인트)를 찾는다."""
    root = Path(root)
    return sorted(p.parent for p in root.rglob("model.safetensors") if (p.parent / "config.json").is_file())


def wilson(k, n, z=1.96):
    """성공 k / 시행 n 비율의 95% Wilson 신뢰구간."""
    import math

    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def lerobot_cli(name="lerobot-train"):
    """현재 가상환경의 LeRobot 명령 경로."""
    return str(Path(sys.executable).parent / name)


def timer():
    t0 = time.time()
    return lambda: time.time() - t0
