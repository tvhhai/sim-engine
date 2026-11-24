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


def _recorder(env):
    """Recorder của replay chuẩn (tools/make_replay) dùng cho policy AI: AI không có phase / IK nên chỉ cần id 5 khớp tay."""
    from types import SimpleNamespace

    from ..tools.make_replay import Recorder
    joint_ids = [env.model.joint(name).id for name in config.JOINT_NAMES[:5]]
    return Recorder(env, SimpleNamespace(_joint_ids=joint_ids, _current_joint_targets=None))


def run_episode(seed, policy, pre, post, device, kind, size, task, max_steps, record=False):
    """Cho policy AI tự chạy một ván. Trả (dict kết quả, ván replay hoặc None)."""
    import torch
    from lerobot.common.control_utils import predict_action

    env, options = sim_setup.make_env(seed, kind, size)
    obs, info = env.reset(seed=seed, options=options)
    rec = _recorder(env) if record else None
    frames = [rec.frame(0, 0, 0, 0, info, 1.0)] if rec else []
    policy.reset()
    pre.reset()
    post.reset()
    grasped, t = False, 0
    for t in range(max_steps):
        frame = sim_setup.to_frame(obs, np.zeros(7), task)
        observation = {k: v for k, v in frame.items() if k.startswith("observation.")}
        action = predict_action(observation, policy, torch.device(device), pre, post, use_amp=False, task=task)
        env_action = sim_setup.to_env_action(action.cpu().numpy())
        obs, _, terminated, truncated, info = env.step(env_action)
        grasped = grasped or bool(env._grasped)
        if rec:
            frames.append(rec.frame(t + 1, 0, 0, t + 1, info, env_action[5]))
        if info.get("success") or truncated:
            break
    x0, y0 = sim_setup.seed_scene(seed)[0]
    r = math.hypot(x0 - config.PICK_PAN_XY[0], y0 - config.PICK_PAN_XY[1])
    row = {"seed": seed, "ok": bool(info.get("success")), "steps": t + 1, "grasped": grasped, "r": round(r, 3),
           "end_xy": [round(float(v), 3) for v in info["box_pos"][:2]]}
    episode = None
    if rec:
        from ..tools.make_replay import bowl_geom_list
        episode = {"seed": seed, "bowl": [round(float(v), 4) for v in env.get_bowl_pos()], "bowl_geoms": bowl_geom_list(env),
                   "success": row["ok"], "steps": len(frames) - 1, "frames": frames}
    env.close()
    return row, episode


def label(row):
    if row["ok"]:
        return "Thành công"
    return "Đã kẹp nhưng hỏng" if row["grasped"] else "Chưa kẹp được"


def write_replay(name, policy_name, kind, size, rows, episodes, per_group):
    """Ghi results/replays/<name>.json (Studio tự liệt kê): tối đa per_group ván mỗi nhóm (thành công / đã kẹp / chưa kẹp)."""
    from .. import paths
    from ..tools import make_replay

    kind_vi = "hộp" if kind == "box" else "bóng"
    keep, count = [], {}
    for row, ep in zip(rows, episodes):
        group = label(row)
        if count.get(group, 0) >= per_group:
            continue
        count[group] = count.get(group, 0) + 1
        tags = [f"AI: {policy_name}", group, f"vật cách trục khớp 1 {row['r'] * 100:.0f} cm", f"{row['steps']} bước"]
        ep.update(name=f"seed {row['seed']} · {policy_name} · {'OK' if row['ok'] else 'FAIL'}", group=group, tags=tags)
        keep.append(ep)
    env, _ = sim_setup.make_env(sim_setup.EVAL_SEEDS[0], kind, size, images=False)
    out_dir = paths.RESULTS / "replays"
    out_dir.mkdir(parents=True, exist_ok=True)
    data = make_replay.write_replay(env, keep, str(out_dir / f"{name}.html"), policy_name)
    env.close()
    ok = sum(r["ok"] for r in rows)
    data["scene"]["title"] = f"{policy_name} · {kind_vi} {size * 100:g} cm · {ok}/{len(rows)} thành công · {len(keep)} ván"
    with open(out_dir / f"{name}.json", "w", encoding="utf-8") as f:
        json.dump(data, f, separators=(",", ":"), ensure_ascii=False)
    print(f"Đã ghi replay results/replays/{name}.json ({len(keep)} ván): mở Studio > Replay > Nguồn replay")


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
    parser.add_argument("--replay", default=None, help="tên replay ghi vào results/replays/<tên>.json để xem trên Studio")
    parser.add_argument("--replay-per-group", type=int, default=15, help="số ván tối đa mỗi nhóm trong replay (file nhỏ, mở nhanh)")
    parser.add_argument("--name", default=None, help="tên policy hiện trên replay (mặc định: tên thư mục train)")
    args = parser.parse_args()

    seeds = sim_setup.EVAL_SEEDS
    if args.limit:
        seeds = [seeds[i] for i in np.linspace(0, len(seeds) - 1, args.limit).round().astype(int)]
    task = sim_setup.TASKS[args.kind].format(color=args.color_name)
    policy, pre, post = load_policy(args.policy, args.device)
    policy_name = args.name or next((p.name for p in Path(args.policy).parents if p.parent.name == "train"), "AI")
    rows, episodes, t0 = [], [], time.time()
    for k, seed in enumerate(seeds):
        row, episode = run_episode(seed, policy, pre, post, args.device, args.kind, args.size, task, args.max_steps,
                                   record=bool(args.replay))
        rows.append(row)
        episodes.append(episode)
        print(f"[{k + 1}/{len(seeds)}] seed {seed}: {'OK' if row['ok'] else 'HỎNG'}"
              f"{' (đã kẹp)' if row['grasped'] and not row['ok'] else ''} {row['steps']} bước | "
              f"thành công {sum(r['ok'] for r in rows)}/{len(rows)} | {time.time() - t0:.0f} s", flush=True)
    summary = summarize(rows)
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps({"policy": args.policy, "kind": args.kind, "size": args.size, "summary": summary,
                                              "rows": rows}, ensure_ascii=False, indent=1), encoding="utf-8")
    if args.replay:
        write_replay(args.replay, policy_name, args.kind, args.size, rows, episodes, args.replay_per_group)


if __name__ == "__main__":
    main()
