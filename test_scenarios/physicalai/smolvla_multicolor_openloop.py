"""학습 데이터 관측을 그대로 넣고 (1) 예측 행동 chunk와 데이터 행동의 오차, (2) 문장의 색만 바꿨을 때 예측 차이를 잰다."""
import json, sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, sys.argv[1])
import physicalai_common as pc
from huggingface_hub import snapshot_download
from lerobot.datasets import LeRobotDataset

out = Path(sys.argv[2])
REV_M, REV_D = "17611a55590b66f2d03060f200be8daa258cbf5c", "1d47ca0df5c9b958acd005817fdb5bf6a1e42bd7"
path = snapshot_download("caspar/smolvla_so101_multicolor_sim_rtx5090_20261118", revision=REV_M,
                         allow_patterns=["*.json", "*.safetensors"])
dev = torch.device("cuda")
policy, pre, post = pc.load_smolvla(path, dev)
ds = LeRobotDataset("caspar/so101_multicolor_sim", revision=REV_D)
ep = ds.meta.episodes
res = []
for e in [0, 1, 2, 75, 76, 77, 147, 148, 149]:
    start = int(ep["dataset_from_index"][e]); L = int(ep["length"][e])
    task = ep["tasks"][e][0]
    for t in [0, 30, 60, 100]:
        fr = ds[start + t]
        gt = torch.stack([ds.hf_dataset[start + t + k]["action"] for k in range(min(50, L - t))]).numpy()
        def chunk(task_text, seed=0):
            obs = {"observation.state": fr["observation.state"][None].to(dev), "task": [task_text],
                   "observation.images.top": fr["observation.images.top"][None].to(dev),
                   "observation.images.wrist": fr["observation.images.wrist"][None].to(dev)}
            pc.seed_noise(seed)
            with torch.inference_mode():
                a = policy.predict_action_chunk(pre(obs))
            return post(a).squeeze(0).cpu().numpy()
        pred = chunk(task)
        mae = np.abs(pred[: len(gt)] - gt).mean(0)
        others = {c: float(np.abs(chunk(pc.TASK_FMT.format(color=c)) - pred).mean())
                  for c in pc.COLORS if c not in task}
        noise = float(np.abs(chunk(task, seed=1) - pred).mean())
        r = {"episode": e, "task": task, "t": t, "mae_per_joint": mae.round(4).tolist(), "mae": float(mae.mean()),
             "diff_other_color_prompt": others, "diff_noise_seed": noise}
        res.append(r)
        print(e, task.split()[3], t, f"mae={r['mae']:.4f}", {k: round(v, 4) for k, v in others.items()},
              f"noise={noise:.4f}", flush=True)
(out / "openloop.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))
print("mean mae", np.mean([r["mae"] for r in res]),
      "mean prompt diff", np.mean([v for r in res for v in r["diff_other_color_prompt"].values()]),
      "mean noise diff", np.mean([r["diff_noise_seed"] for r in res]))
