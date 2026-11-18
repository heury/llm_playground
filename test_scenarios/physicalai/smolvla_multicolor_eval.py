"""3색(빨강·초록·노랑) 장면에서 SmolVLA가 명령한 색 블록을 옮기는지 평가한다.

notebooks/physicalai/physicalai_common.py의 장면·배치·판정(8장과 동일, 학습 데이터 생성기와 같은 설정)을 쓴다.
배치 하나마다 세 색을 차례로 명령하고, 같은 배치의 세 명령은 같은 noise seed를 써서 차이가 문장에서만 오게 한다.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import imageio
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
NB = Path(sys.argv[sys.argv.index("--nb") + 1]) if "--nb" in sys.argv else None
sys.path.insert(0, str(NB))
import physicalai_common as pc  # noqa: E402

RGB = {"red": (230, 25, 25), "green": (25, 180, 50), "yellow": (240, 215, 25)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nb", required=True)
    ap.add_argument("--policy", default="caspar/smolvla_so101_multicolor_sim_rtx5090_20261118")
    ap.add_argument("--revision", default="17611a55590b66f2d03060f200be8daa258cbf5c")
    ap.add_argument("--layouts", type=int, default=10)
    ap.add_argument("--seed", type=int, default=2000)
    ap.add_argument("--max-seconds", type=float, default=12.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    (out / "videos").mkdir(parents=True, exist_ok=True)

    from huggingface_hub import snapshot_download
    path = snapshot_download(args.policy, revision=args.revision, allow_patterns=["*.json", "*.safetensors"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy, pre, post = pc.load_smolvla(path, device)

    rng = np.random.default_rng(args.seed)
    records, t0 = [], time.time()
    writers = {c: imageio.get_writer(out / "videos" / f"cmd_{c}.mp4", fps=30) for c in pc.COLORS}
    for L in range(args.layouts):
        layout = pc.random_layout(rng)
        for color in pc.COLORS:
            best = {c: 9.0 for c in pc.COLORS}
            max_z = {c: 0.0 for c in pc.COLORS}

            def track(robot, best=best, max_z=max_z):
                s = robot.site_pose()[0]
                for c in pc.COLORS:
                    p = robot.cube_pos(c)
                    best[c] = min(best[c], float(np.linalg.norm(s[:2] - p[:2])))
                    max_z[c] = max(max_z[c], float(p[2]))

            pc.seed_noise(L)
            r = pc.run_policy_episode(policy, pre, post, device, layout=layout, color=color,
                                      max_seconds=args.max_seconds, video=True,
                                      image_keys=pc.SMOLVLA_KEYS, on_step=track)
            near = min(best, key=best.get)
            res = r["result"].split("(")[0]
            for f in r["frames"]:
                f = np.ascontiguousarray(f)
                cv2.rectangle(f, (10, 10), (34, 34), RGB[color], -1)
                txt = f"layout {L} cmd={color}  -> {dict(성공='SUCCESS', 오답='WRONG', 실패='FAIL')[res]}"
                for th, fg in ((4, (0, 0, 0)), (2, (255, 255, 255))):
                    cv2.putText(f, txt, (44, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, fg, th, cv2.LINE_AA)
                writers[color].append_data(f)
            rec = {"layout": L, "color": color, "result": r["result"], "t": r["t"],
                   "closest_block": near, "min_dist_cm": {c: round(100 * v, 1) for c, v in best.items()},
                   "lifted": [c for c in pc.COLORS if max_z[c] > 0.05],
                   "positions": dict(zip(pc.COLORS, layout))}
            records.append(rec)
            print(f"layout {L:2d} cmd={color:6s} {r['result']:10s} t={r['t']}  closest={near}  "
                  f"lifted={rec['lifted']}  ({time.time() - t0:.0f}s)", flush=True)
    for w in writers.values():
        w.close()

    stats = {c: {k: sum(r["color"] == c and r["result"].startswith(k) for r in records)
                 for k in ("성공", "오답", "실패")} for c in pc.COLORS}
    for c in pc.COLORS:
        rs = [r for r in records if r["color"] == c]
        stats[c]["closest_is_target"] = sum(r["closest_block"] == c for r in rs)
        stats[c]["target_lifted"] = sum(c in r["lifted"] for r in rs)
        stats[c]["n"] = len(rs)
    n = len(records)
    ok = sum(s["성공"] for s in stats.values())
    hit = sum(s["closest_is_target"] for s in stats.values())
    # 같은 배치에서 세 명령이 모두 같은 블록으로 갔는지 (언어 무시 여부)
    same = sum(len({r["closest_block"] for r in records if r["layout"] == L}) == 1 for L in range(args.layouts))
    summary = {"episodes": n, "success": ok, "success_ci95": pc.wilson(ok, n),
               "closest_is_target": hit, "closest_ci95": pc.wilson(hit, n),
               "layouts_all_cmds_same_block": same, "per_color": stats}
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    (out / "results.json").write_text(json.dumps(
        {"policy": args.policy, "revision": args.revision, "seed": args.seed, "max_seconds": args.max_seconds,
         "summary": summary, "records": records}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
