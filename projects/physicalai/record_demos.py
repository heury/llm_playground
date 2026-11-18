"""Pick and Place 시연 데이터셋 만들기 (LeRobotDataset 형식)

규칙 기반 Pick and Place 프로그램이 사람 시연자를 대신합니다.
블록 위치를 무작위로 바꿔 가며 episode를 실행하고, 1/FPS 초마다 (관측, 행동)을 기록합니다.

    observation.state         관절각 6개 (팔 5 + 집게 1)
    observation.images.top    위 카메라 이미지
    observation.images.wrist  손목 카메라 이미지
    action                    그 순간 구동기 6개에 명령한 목표 각도

실행 (notebooks/physicalai 의 uv 환경 사용):
    MUJOCO_GL=egl ../../notebooks/physicalai/.venv/bin/python record_demos.py --episodes 50
"""

import argparse
import os
import shutil
import time

os.environ.setdefault("SVT_LOG", "0")    # 영상 인코더(SVT-AV1) 로그 끄기

import mujoco
import numpy as np

from so101_sim import GRIPPER_CLOSED, GRIPPER_OPEN, SO101Sim

FPS = 20
IMG_H, IMG_W = 240, 320
CAMERAS = {"top": "top", "wrist": "wrist_cam"}     # 데이터셋 키 → MuJoCo 카메라 이름
TASK = "Pick up the red cube and put it in the blue box."
JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_ROOT = os.path.join(REPO_ROOT, "outputs", "physicalai", "datasets", "so101_pickplace_sim")

# 손끝의 목표 높이 (바닥 기준, m)
APPROACH_Z = 0.08   # 블록 위에서 접근
GRASP_Z = 0.02      # 블록 중심(0.015)보다 5 mm 위에서 집기
LIFT_Z = 0.10       # 들어 올려 옮기는 높이 (더 높으면 상자 앞쪽 위에서 IK가 닿지 않는다)
PLACE_Z = 0.06      # 상자 위에서 놓는 높이


def pick_and_place(robot, color="red", place_offset=(0.0, 0.0)):
    """color 블록을 상자 중심 + place_offset 위치에 옮긴다. Approach → Grasp → Lift → Move → Place.

    성공 여부(블록이 상자 안 바닥 근처에 있는지)를 돌려준다.
    """
    cube, box = robot.cube_pos(color), robot.box_pos()
    target = box[:2] + np.asarray(place_offset)
    at = lambda xy, z: np.array([xy[0], xy[1], z])  # noqa: E731

    robot.set_gripper(GRIPPER_OPEN, 0.5)
    robot.move_to(at(cube, APPROACH_Z), seconds=1.2)    # Approach
    robot.move_to(at(cube, GRASP_Z), seconds=0.8)       # Grasp
    robot.hold(0.2)
    robot.set_gripper(GRIPPER_CLOSED, 0.8)
    robot.move_to(at(cube, LIFT_Z), seconds=1.0)        # Lift
    if robot.cube_pos(color)[2] < 0.06:
        return False                                    # 놓쳤다
    robot.move_to(at(target, LIFT_Z), seconds=1.5)      # Move
    robot.move_to(at(target, PLACE_Z), seconds=0.8)     # Place
    robot.hold(0.2)
    robot.set_gripper(GRIPPER_OPEN, 0.5)
    robot.move_to(at(target, LIFT_Z), seconds=0.8)
    robot.hold(0.3)
    return in_box(robot, color)


def in_box(robot, color="red"):
    """블록이 상자 벽 안쪽, 바닥 근처(높이 5 cm 미만)에 있는지."""
    cube, box = robot.cube_pos(color), robot.box_pos()
    return abs(cube[0] - box[0]) < 0.05 and abs(cube[1] - box[1]) < 0.05 and cube[2] < 0.05


def random_cube_xy(rng):
    """로봇 앞쪽 부채꼴(반지름 0.17~0.27 m, 각도 ±40°)에서 블록 위치를 뽑는다."""
    r, th = rng.uniform(0.17, 0.27), rng.uniform(-0.7, 0.7)
    return float(r * np.cos(th)), float(r * np.sin(th))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--root", default=DEFAULT_ROOT, help="데이터셋을 저장할 폴더")
    ap.add_argument("--repo-id", default="physicalai/so101_pickplace_sim")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--keep-failed", action="store_true", help="실패한 시연도 저장")
    ap.add_argument("--overwrite", action="store_true", help="--root 폴더가 있으면 지우고 새로 기록")
    args = ap.parse_args()

    if os.path.exists(args.root):
        if not args.overwrite:
            ap.error(f"{args.root}가 이미 있습니다. --overwrite 또는 다른 --root를 주세요.")
        shutil.rmtree(args.root)

    from lerobot.datasets import LeRobotDataset

    image = {"dtype": "video", "shape": (IMG_H, IMG_W, 3), "names": ["height", "width", "channels"]}
    features = {
        "observation.state": {"dtype": "float32", "shape": (6,), "names": JOINT_NAMES},
        "action": {"dtype": "float32", "shape": (6,), "names": JOINT_NAMES},
        **{f"observation.images.{key}": image for key in CAMERAS},
    }
    dataset = LeRobotDataset.create(repo_id=args.repo_id, fps=FPS, features=features,
                                    root=args.root, robot_type="so101_sim", use_videos=True)

    robot = SO101Sim()
    renderer = mujoco.Renderer(robot.model, height=IMG_H, width=IMG_W)
    record_every = round(1 / (FPS * robot.model.opt.timestep))   # 시뮬레이션 몇 step마다 한 프레임

    def record(sim):
        """물리 step 직전에 호출된다. 1/FPS 초마다 (관측, 행동) 한 프레임을 버퍼에 넣는다."""
        if sim.n_steps % record_every != 0:
            return
        frame = {
            "observation.state": sim.data.qpos[:6].astype(np.float32),
            "action": sim.data.ctrl[:6].astype(np.float32),
            "task": TASK,
        }
        for key, cam in CAMERAS.items():
            renderer.update_scene(sim.data, camera=cam)
            frame[f"observation.images.{key}"] = renderer.render().copy()
        dataset.add_frame(frame)

    robot.on_step = record
    rng = np.random.default_rng(args.seed)
    t0, saved = time.time(), 0
    for ep in range(args.episodes):
        cube = random_cube_xy(rng)
        robot.reset({"red": cube})
        ok = pick_and_place(robot)
        if ok or args.keep_failed:
            dataset.save_episode()
            saved += 1
        else:
            dataset.clear_episode_buffer()
        print(f"episode {ep + 1:3d}/{args.episodes}  블록 ({cube[0]:+.3f}, {cube[1]:+.3f})  "
              f"{'성공' if ok else '실패'}  {'저장' if ok or args.keep_failed else '버림'}  "
              f"({time.time() - t0:.0f} s)")

    renderer.close()
    dataset.finalize()
    print(f"\n완료: {saved}/{args.episodes} episode 저장, 총 {dataset.meta.total_frames} 프레임")
    print(f"데이터셋 위치: {args.root}")


if __name__ == "__main__":
    main()
