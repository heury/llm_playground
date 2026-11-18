"""여러 색깔 블록을 색깔별로 한 번씩 상자에 옮기는 테스트 + 영상 저장

바닥에 빨강·초록·노랑·보라 블록을 무작위로 흩어 놓고, 정해진 색 순서대로 하나씩 집어
상자 안의 서로 다른 자리(네 귀퉁이)에 놓습니다. 회차마다 블록 배치가 바뀝니다.

실행 (notebooks/physicalai 의 uv 환경 사용):
    MUJOCO_GL=egl ../../notebooks/physicalai/.venv/bin/python test_color_pick.py --trials 3
"""

import argparse
import json
import os
import time

import cv2
import imageio
import mujoco
import numpy as np

from record_demos import REPO_ROOT, in_box, pick_and_place
from so101_sim import CUBE_RGBA, SO101Sim

VIDEO_FPS = 30
COLORS = ["red", "green", "yellow", "purple"]
# 상자 중심 기준 놓을 자리 (상자 안쪽 반폭 4.7 cm, 블록 반폭 1.5 cm)
PLACE_OFFSETS = [(-0.022, -0.022), (0.022, -0.022), (-0.022, 0.022), (0.022, 0.022)]
MIN_GAP = 0.08      # 블록 사이 최소 거리 (벌린 집게가 옆 블록을 건드리지 않도록)


def random_layout(rng, colors):
    """로봇 앞 부채꼴(반지름 0.17~0.27 m, ±40°)에 서로 MIN_GAP 이상 떨어지게 블록을 배치한다."""
    while True:
        pts = []
        for _ in range(200):
            r, th = rng.uniform(0.17, 0.27), rng.uniform(-0.7, 0.7)
            p = np.array([r * np.cos(th), r * np.sin(th)])
            if all(np.linalg.norm(p - q) >= MIN_GAP for q in pts):
                pts.append(p)
                if len(pts) == len(colors):
                    return {c: tuple(float(v) for v in p) for c, p in zip(colors, pts)}


def label(frame, text, color):
    """영상 왼쪽 위에 현재 목표를 적는다."""
    rgb = tuple(int(255 * float(v)) for v in CUBE_RGBA[color].split()[:3]) if color else (255, 255, 255)
    cv2.rectangle(frame, (10, 10), (34, 34), rgb, -1)
    for thickness, fg in ((4, (0, 0, 0)), (2, (255, 255, 255))):      # 검은 테두리 + 흰 글씨
        cv2.putText(frame, text, (44, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, fg, thickness, cv2.LINE_AA)
    return frame


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--camera", default="view", help="영상 카메라 (view, top, wrist_cam)")
    ap.add_argument("--out", default=None, help="결과 폴더 (기본: outputs/physicalai/<시각>-color-pick-test)")
    args = ap.parse_args()
    out = args.out or os.path.join(REPO_ROOT, "outputs", "physicalai",
                                   time.strftime("%Y%m%d-%H%M%S") + "-color-pick-test")
    os.makedirs(out, exist_ok=True)

    robot = SO101Sim(colors=COLORS)
    renderer = mujoco.Renderer(robot.model, height=480, width=640)
    render_every = round(1 / (VIDEO_FPS * robot.model.opt.timestep))
    frames, caption = [], {"text": "", "color": None}

    def capture(sim):
        if sim.n_steps % render_every == 0:
            renderer.update_scene(sim.data, camera=args.camera)
            frames.append(label(renderer.render().copy(), caption["text"], caption["color"]))

    robot.on_step = capture
    rng = np.random.default_rng(args.seed)
    results = []
    with imageio.get_writer(os.path.join(out, "all_trials.mp4"), fps=VIDEO_FPS) as all_writer:
        for i in range(1, args.trials + 1):
            layout = random_layout(rng, COLORS)
            robot.reset(layout)
            frames.clear()
            picks = []
            for k, (color, offset) in enumerate(zip(COLORS, PLACE_OFFSETS), 1):
                caption.update(text=f"trial {i}  {k}/{len(COLORS)}  {color}", color=color)
                ok = pick_and_place(robot, color, offset)
                picks.append({"color": color, "start_xy": list(layout[color]), "placed": bool(ok)})
                print(f"trial {i}  {k}/{len(COLORS)} {color:6s} ({layout[color][0]:+.3f}, {layout[color][1]:+.3f})"
                      f"  → {'성공' if ok else '실패'}")
            n_in = sum(in_box(robot, c) for c in COLORS)
            caption.update(text=f"trial {i}  done: {n_in}/{len(COLORS)} in box", color=None)
            robot.hold(1.0)                                   # 마지막 상태를 잠깐 보여 준다
            for p in picks:                                   # 모든 동작이 끝난 뒤에도 상자 안에 있는지
                p["in_box_at_end"] = bool(in_box(robot, p["color"]))
                p["final_xyz"] = robot.cube_pos(p["color"]).round(4).tolist()
            video = f"trial_{i:02d}.mp4"
            imageio.mimsave(os.path.join(out, video), frames, fps=VIDEO_FPS)
            for f in frames:
                all_writer.append_data(f)
            n_end = sum(p["in_box_at_end"] for p in picks)
            results.append({"trial": i, "picks": picks, "in_box_at_end": n_end, "video": video})
            print(f"trial {i}  최종: 상자 안 {n_end}/{len(COLORS)}\n")

    renderer.close()
    total = sum(r["in_box_at_end"] for r in results)
    summary = {"seed": args.seed, "camera": args.camera, "colors": COLORS, "place_offsets": PLACE_OFFSETS,
               "box_xy": robot.box_pos()[:2].tolist(), "in_box_at_end": total,
               "total_picks": len(COLORS) * args.trials, "results": results}
    with open(os.path.join(out, "results.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"전체: {total}/{len(COLORS) * args.trials} 블록이 상자 안   결과 폴더: {out}")


if __name__ == "__main__":
    main()
