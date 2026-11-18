"""Pick and Place 동작 테스트 + 영상 저장

record_demos.py의 시연 프로그램을 블록 위치를 바꿔 가며 실행하고,
대각선 카메라("view") 영상과 회차별 결과(JSON)를 남깁니다.

실행 (notebooks/physicalai 의 uv 환경 사용):
    MUJOCO_GL=egl ../../notebooks/physicalai/.venv/bin/python test_pick_place.py --trials 5
"""

import argparse
import json
import os
import time

import imageio
import mujoco
import numpy as np

from record_demos import REPO_ROOT, pick_and_place, random_cube_xy
from so101_sim import SO101Sim

VIDEO_FPS = 30


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--camera", default="view", help="영상 카메라 (view, top, wrist_cam)")
    ap.add_argument("--out", default=None, help="결과 폴더 (기본: outputs/physicalai/<시각>-pick-place-test)")
    args = ap.parse_args()
    out = args.out or os.path.join(REPO_ROOT, "outputs", "physicalai",
                                   time.strftime("%Y%m%d-%H%M%S") + "-pick-place-test")
    os.makedirs(out, exist_ok=True)

    robot = SO101Sim()
    renderer = mujoco.Renderer(robot.model, height=480, width=640)
    render_every = round(1 / (VIDEO_FPS * robot.model.opt.timestep))
    frames = []

    def capture(sim):
        if sim.n_steps % render_every == 0:
            renderer.update_scene(sim.data, camera=args.camera)
            frames.append(renderer.render().copy())

    robot.on_step = capture
    rng = np.random.default_rng(args.seed)
    results = []
    with imageio.get_writer(os.path.join(out, "all_trials.mp4"), fps=VIDEO_FPS) as all_writer:
        for i in range(1, args.trials + 1):
            cube = random_cube_xy(rng)
            robot.reset({"red": cube})
            frames.clear()
            ok = pick_and_place(robot)
            robot.hold(0.5)                                  # 마지막 장면을 잠깐 더 보여 준다
            video = f"trial_{i:02d}.mp4"
            imageio.mimsave(os.path.join(out, video), frames, fps=VIDEO_FPS)
            for f in frames:
                all_writer.append_data(f)
            final = robot.cube_pos()
            results.append({"trial": i, "cube_xy": list(cube), "final_cube_xyz": final.round(4).tolist(),
                            "success": bool(ok), "video": video})
            print(f"trial {i:2d}/{args.trials}  블록 ({cube[0]:+.3f}, {cube[1]:+.3f})  "
                  f"→ 최종 ({final[0]:+.3f}, {final[1]:+.3f}, {final[2]:.3f})  {'성공' if ok else '실패'}")

    renderer.close()
    n_ok = sum(r["success"] for r in results)
    summary = {"seed": args.seed, "camera": args.camera, "box_xy": robot.box_pos()[:2].tolist(),
               "success": n_ok, "trials": args.trials, "results": results}
    with open(os.path.join(out, "results.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n성공률 {n_ok}/{args.trials}   결과 폴더: {out}")


if __name__ == "__main__":
    main()
