"""Đánh giá một policy LeRobot đã train trên bộ seed cố định, ra bảng giống bảng của policy viết tay.

Cách dùng (môi trường có LeRobot, vd D:/vla/venv-lerobot):
    python -m sim_engine.learning.eval_policy --policy D:/vla/outputs/train/act_box4_v1/checkpoints/last/pretrained_model \
        --kind box --size 0.04 --limit 100 --out D:/vla/eval/act_box4_v1.json

--limit N: chỉ chạy N seed đầu của bộ đánh giá (cách đều trong 774 seed) để có số nhanh; bỏ --limit để chạy đủ 774.
Kết quả: tỉ lệ thành công chung, theo bán kính (r < 37 cm / r >= 37 cm), danh sách seed hỏng; ghi JSON từng seed.
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

from .. import config
from . import sim_setup


def load_policy(path, device):
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import get_policy_class, make_pre_post_processors

    cfg = PreTrainedConfig.from_pretrained(path)
    cfg.device = device
    policy = get_policy_class(cfg.type).from_pretrained(path, config=cfg)
    policy.to(device).eval()
    pre, post = make_pre_post_processors(cfg, pretrained_path=path,
                                         preprocessor_overrides={"device_processor": {"device": device}})
    return policy, pre, post


def run_episode(seed, policy, pre, post, device, kind, size, task, max_steps):
    """Cho policy AI tự chạy một ván. Trả dict kết quả."""
    import torch
    from lerobot.common.control_utils import predict_action

    env, options = sim_setup.make_env(seed, kind, size)
    obs, _ = env.reset(seed=seed, options=options)
    policy.reset()
    pre.reset()
    post.reset()
    info, grasped, t = {}, False, 0
    for t in range(max_steps):
        frame = sim_setup.to_frame(obs, np.zeros(7), task)
        observation = {k: v for k, v in frame.items() if k.startswith("observation.")}
        action = predict_action(observation, policy, torch.device(device), pre, post, use_amp=False, task=task)
        obs, _, terminated, truncated, info = env.step(sim_setup.to_env_action(action.cpu().numpy()))
        grasped = grasped or bool(env._grasped)
        if info.get("success") or truncated:
            break
    env.close()
    xy = info.get("box_pos", [0, 0])[:2] if info else [0, 0]
    x0, y0 = sim_setup.seed_scene(seed)[0]
    r = math.hypot(x0 - config.PICK_PAN_XY[0], y0 - config.PICK_PAN_XY[1])
    return {"seed": seed, "ok": bool(info.get("success")), "steps": t + 1, "grasped": grasped, "r": round(r, 3),
            "end_xy": [round(float(v), 3) for v in xy]}


def summarize(rows):
    def rate(sel):
        return f"{sum(r['ok'] for r in sel)}/{len(sel)} ({100 * sum(r['ok'] for r in sel) / max(1, len(sel)):.1f} %)"
    near = [r for r in rows if r["r"] < 0.37]
    far = [r for r in rows if r["r"] >= 0.37]
    return {"all": rate(rows), "r<37cm": rate(near), "r>=37cm": rate(far),
            "grasped_but_failed": [r["seed"] for r in rows if r["grasped"] and not r["ok"]],
            "never_grasped": [r["seed"] for r in rows if not r["grasped"]]}


def main():
    parser = argparse.ArgumentParser(description="Đánh giá policy LeRobot trên bộ seed cố định")
    parser.add_argument("--policy", required=True, help="thư mục pretrained_model của checkpoint")
    parser.add_argument("--kind", choices=["box", "sphere"], default="box")
    parser.add_argument("--size", type=float, default=0.04)
    parser.add_argument("--color-name", default="red")
    parser.add_argument("--limit", type=int, default=None, help="chỉ chạy N seed (cách đều) của bộ 774 seed")
    parser.add_argument("--max-steps", type=int, default=config.MAX_EPISODE_STEPS)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", default=None, help="file JSON kết quả từng seed")
    args = parser.parse_args()

    seeds = sim_setup.EVAL_SEEDS
    if args.limit:
        seeds = [seeds[i] for i in np.linspace(0, len(seeds) - 1, args.limit).round().astype(int)]
    task = sim_setup.TASKS[args.kind].format(color=args.color_name)
    policy, pre, post = load_policy(args.policy, args.device)
    rows, t0 = [], time.time()
    for k, seed in enumerate(seeds):
        row = run_episode(seed, policy, pre, post, args.device, args.kind, args.size, task, args.max_steps)
        rows.append(row)
        print(f"[{k + 1}/{len(seeds)}] seed {seed}: {'OK' if row['ok'] else 'HỎNG'}"
              f"{' (đã kẹp)' if row['grasped'] and not row['ok'] else ''} {row['steps']} bước | "
              f"thành công {sum(r['ok'] for r in rows)}/{len(rows)} | {time.time() - t0:.0f} s", flush=True)
    summary = summarize(rows)
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps({"policy": args.policy, "kind": args.kind, "size": args.size, "summary": summary,
                                              "rows": rows}, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
