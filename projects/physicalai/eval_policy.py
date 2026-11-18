"""학습한 정책(SmolVLA)을 여러 색깔 블록이 있는 장면에서 평가하고 영상으로 남긴다.

조건 (모든 조건이 같은 블록 배치 목록을 쓰므로 서로 비교할 수 있다)
    red_only       빨간 블록만 있는 장면, 명령 "red"  (학습 데이터와 같은 조건)
    distractors    빨강·초록·노랑·보라 블록, 명령 "red"  (다른 색 블록이 방해가 되는가)
    lang_<색>      같은 4색 장면, 명령 "<색>"            (명령 문장을 따르는가)

판정 (episode마다 하나)
    성공    목표 색 블록만 상자 안에 들어감
    오답    다른 색 블록이 상자 안에 들어감
    실패    제한 시간 안에 아무 블록도 상자에 넣지 못함

실행 (notebooks/physicalai 의 uv 환경, GPU 권장):
    MUJOCO_GL=egl ../../notebooks/physicalai/.venv/bin/python eval_policy.py \\
        --policy caspar/smolvla_so101_pickplace_sim_150_rtx5090_20261118 --episodes 20
"""

import argparse
import json
import os
import time

import cv2
import imageio
import mujoco
import numpy as np
import torch

from record_demos import FPS, IMG_H, IMG_W, REPO_ROOT, in_box
from so101_sim import CUBE_RGBA, SO101Sim
from test_color_pick import COLORS, random_layout

TASK_FMT = "Pick up the {color} cube and put it in the blue box."   # 학습 문장의 색만 바꾼다
LABEL = {"성공": "SUCCESS", "오답": "WRONG CUBE", "실패": "FAIL"}      # 영상 자막용 (OpenCV는 한글 불가)


def load_policy(repo_id, revision, device):
    """Hub의 SmolVLA 체크포인트와 전처리기·후처리기(정규화·카메라 이름 변환 포함)를 불러온다."""
    from huggingface_hub import snapshot_download
    from lerobot.policies import make_pre_post_processors
    from lerobot.policies.smolvla import SmolVLAPolicy

    path = snapshot_download(repo_id, revision=revision, allow_patterns=["*.json", "*.safetensors", "*.md"])
    policy = SmolVLAPolicy.from_pretrained(path).to(device).eval()
    preprocess, postprocess = make_pre_post_processors(
        policy.config, path, preprocessor_overrides={"device_processor": {"device": str(device)}})
    return policy, preprocess, postprocess


class Evaluator:
    def __init__(self, policy, preprocess, postprocess, device, max_seconds, camera):
        self.policy, self.pre, self.post, self.device = policy, preprocess, postprocess, device
        self.max_steps = int(max_seconds * FPS)
        self.camera = camera
        self.sims = {}                                   # 색 구성별 시뮬레이터 (한 번만 만든다)

    def sim(self, colors):
        key = tuple(colors)
        if key not in self.sims:
            robot = SO101Sim(colors=colors)
            self.sims[key] = (robot, mujoco.Renderer(robot.model, IMG_H, IMG_W),
                              mujoco.Renderer(robot.model, 480, 640))
        return self.sims[key]

    def observe(self, robot, obs_renderer, task):
        """학습 데이터와 같은 형식: 관절각 6개 + top·wrist 이미지 (1,3,H,W) 0~1 + 명령 문장."""
        obs = {"observation.state": torch.from_numpy(robot.data.qpos[:6].astype(np.float32))[None].to(self.device),
               "task": [task]}
        for key, cam in (("top", "top"), ("wrist", "wrist_cam")):
            obs_renderer.update_scene(robot.data, camera=cam)
            img = torch.from_numpy(obs_renderer.render().copy()).permute(2, 0, 1).float() / 255.0
            obs[f"observation.images.{key}"] = img[None].to(self.device)
        return obs

    def run(self, colors, layout, target, seed, caption):
        """episode 하나를 실행하고 (결과 dict, 영상 프레임 목록)을 돌려준다."""
        robot, obs_r, vid_r = self.sim(colors)
        robot.reset({c: layout[c] for c in colors})
        task = TASK_FMT.format(color=target)
        self.policy.reset()
        torch.manual_seed(seed)                          # Flow Matching 시작 noise 고정
        steps_per_frame = round(1 / (FPS * robot.model.opt.timestep))
        frames, max_z = [], {c: 0.0 for c in colors}
        result, t_done = "실패", None

        def frame(text):
            vid_r.update_scene(robot.data, camera=self.camera)
            img = vid_r.render().copy()
            rgb = tuple(int(255 * float(v)) for v in CUBE_RGBA[target].split()[:3])
            cv2.rectangle(img, (10, 10), (34, 34), rgb, -1)
            for th, fg in ((4, (0, 0, 0)), (2, (255, 255, 255))):
                cv2.putText(img, text, (44, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, fg, th, cv2.LINE_AA)
            frames.append(img)

        for i in range(self.max_steps):
            with torch.inference_mode():
                action = self.post(self.policy.select_action(self.pre(self.observe(robot, obs_r, task))))
            robot.data.ctrl[:6] = action.squeeze(0).cpu().numpy()
            robot.step(steps_per_frame)
            for c in colors:
                max_z[c] = max(max_z[c], float(robot.cube_pos(c)[2]))
            frame(f"{caption}  t={(i + 1) / FPS:4.1f}s")
            placed = [c for c in colors if in_box(robot, c)]
            if placed:
                result = "성공" if placed == [target] else "오답"
                t_done = (i + 1) / FPS
                for _ in range(10):                      # 결과 장면을 0.5초 더 보여 준다
                    robot.step(steps_per_frame)
                    frame(f"{caption}  {LABEL[result]}")
                break
        else:
            for _ in range(10):
                frame(f"{caption}  {LABEL[result]}")

        return {"target": target, "task": task, "seed": seed, "result": result, "time_s": t_done,
                "in_box": [c for c in colors if in_box(robot, c)],
                "lifted": [c for c in colors if max_z[c] > 0.05],          # 5 cm 넘게 들어 올린 블록
                "start_xy": {c: list(layout[c]) for c in colors}}, frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default="caspar/smolvla_so101_pickplace_sim_150_rtx5090_20261118")
    ap.add_argument("--revision", default=None, help="Hub 리비전 (커밋 해시 권장)")
    ap.add_argument("--episodes", type=int, default=20, help="red_only·distractors 조건의 episode 수")
    ap.add_argument("--lang-episodes", type=int, default=10, help="lang_<색> 조건별 episode 수")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-seconds", type=float, default=15.0)
    ap.add_argument("--camera", default="view", help="영상 카메라")
    ap.add_argument("--out", default=None, help="결과 폴더 (기본: outputs/physicalai/<시각>-smolvla-multiblock-eval)")
    args = ap.parse_args()
    out = args.out or os.path.join(REPO_ROOT, "outputs", "physicalai",
                                   time.strftime("%Y%m%d-%H%M%S") + "-smolvla-multiblock-eval")
    os.makedirs(os.path.join(out, "videos"), exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy, pre, post = load_policy(args.policy, args.revision, device)
    ev = Evaluator(policy, pre, post, device, args.max_seconds, args.camera)

    rng = np.random.default_rng(args.seed)
    layouts = [random_layout(rng, COLORS) for _ in range(max(args.episodes, args.lang_episodes))]
    conditions = [("red_only", ["red"], "red", args.episodes),
                  ("distractors", COLORS, "red", args.episodes)]
    conditions += [(f"lang_{c}", COLORS, c, args.lang_episodes) for c in COLORS if c != "red"]

    records, t0 = [], time.time()
    for name, colors, target, n in conditions:
        with imageio.get_writer(os.path.join(out, "videos", f"{name}.mp4"), fps=FPS) as writer:
            for ep in range(n):
                rec, frames = ev.run(colors, layouts[ep], target, args.seed * 1000 + ep, f"{name} #{ep + 1}")
                rec.update(condition=name, episode=ep + 1)
                records.append(rec)
                for f in frames:
                    writer.append_data(f)
                t = f"{rec['time_s']:.1f}s" if rec["time_s"] else "-"
                print(f"{name:12s} {ep + 1:2d}/{n}  {rec['result']}  t={t:>5s}  상자 {rec['in_box']}  "
                      f"들어올림 {rec['lifted']}  ({time.time() - t0:.0f} s)", flush=True)

    summary = {}
    for name, _, target, n in conditions:
        rs = [r for r in records if r["condition"] == name]
        summary[name] = {k: sum(r["result"] == k for r in rs) for k in ("성공", "오답", "실패")}
        summary[name]["episodes"] = n
        summary[name]["target_lifted"] = sum(target in r["lifted"] for r in rs)
        print(f"{name:12s} 성공 {summary[name]['성공']}/{n}  오답 {summary[name]['오답']}  "
              f"실패 {summary[name]['실패']}  목표 블록 들어올림 {summary[name]['target_lifted']}")
    with open(os.path.join(out, "results.json"), "w", encoding="utf-8") as f:
        json.dump({"policy": args.policy, "revision": args.revision, "seed": args.seed,
                   "max_seconds": args.max_seconds, "summary": summary, "episodes": records},
                  f, ensure_ascii=False, indent=2)
    print(f"결과 폴더: {out}")


if __name__ == "__main__":
    main()
