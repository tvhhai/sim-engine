"""Ghi dataset LeRobot từ policy viết tay (thầy giáo): chạy từng seed, CHỈ giữ ván thành công.

Cách dùng (môi trường có LeRobot, vd D:/vla/venv-lerobot):
    python -m sim_engine.learning.record_dataset --episodes 200 --kind box --size 0.04 \
        --root D:/vla/datasets/so101_box4_v1 --repo-id local/so101_box4_v1

Seed bắt đầu từ sim_setup.TRAIN_SEED_START (3000), không trùng bộ seed đánh giá. Ván hỏng bị bỏ và chạy seed kế tiếp cho tới khi đủ
--episodes ván thành công. In tiến độ và ghi thêm <root>/record_log.json (seed nào được giữ, seed nào hỏng).
"""

import argparse
import contextlib
import io
import json
import time
from pathlib import Path

from .. import config
from ..policies import default_policy
from . import sim_setup


def run_episode(seed, kind, size, color, task):
    """Chạy thầy giáo một ván có ảnh. Trả (thành công?, danh sách khung hình)."""
    env, options = sim_setup.make_env(seed, kind, size, color)
    policy = default_policy(env)
    obs, _ = env.reset(seed=seed, options=options)
    policy.reset()
    frames, info = [], {}
    for _ in range(config.MAX_EPISODE_STEPS):
        with contextlib.redirect_stdout(io.StringIO()):
            action = policy.select_action({"state": obs["state"], "images": {}, "task": task})
        frames.append(sim_setup.to_frame(obs, action, task))
        obs, _, _, truncated, info = env.step(action)
        if truncated or policy._phase > 10:                # phase 11 = đã về home xong
            break
    _, info, _ = env.settle(action, info)
    env.close()
    return bool(info.get("success")), frames


def main():
    parser = argparse.ArgumentParser(description="Ghi LeRobotDataset từ policy viết tay (chỉ ván thành công)")
    parser.add_argument("--episodes", type=int, default=200, help="số ván THÀNH CÔNG cần ghi")
    parser.add_argument("--kind", choices=["box", "sphere"], default="box")
    parser.add_argument("--size", type=float, default=0.04, help="cạnh hộp / đường kính bóng (m)")
    parser.add_argument("--color", default="#e5484d")
    parser.add_argument("--color-name", default="red", help="tên màu trong câu nhiệm vụ")
    parser.add_argument("--seed-start", type=int, default=sim_setup.TRAIN_SEED_START)
    parser.add_argument("--root", required=True, help="thư mục dataset (chưa tồn tại)")
    parser.add_argument("--repo-id", default="local/so101_dataset")
    args = parser.parse_args()

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    task = sim_setup.TASKS[args.kind].format(color=args.color_name)
    ds = LeRobotDataset.create(args.repo_id, sim_setup.FPS, sim_setup.features(), root=args.root, robot_type="so101_follower",
                               use_videos=True, streaming_encoding=True)   # mã hoá video song song khi đang chạy ván kế
    log, seed, t0 = {"kept": [], "failed": []}, args.seed_start, time.time()
    try:
        while len(log["kept"]) < args.episodes:
            ok, frames = run_episode(seed, args.kind, args.size, args.color, task)
            if ok:
                for frame in frames:
                    ds.add_frame(frame)
                ds.save_episode()
                log["kept"].append({"seed": seed, "frames": len(frames)})
            else:
                log["failed"].append(seed)
            done = len(log["kept"])
            print(f"seed {seed}: {'giữ' if ok else 'HỎNG, bỏ'} ({len(frames)} khung) | đã có {done}/{args.episodes} ván, "
                  f"bỏ {len(log['failed'])}, {time.time() - t0:.0f} s", flush=True)
            seed += 1
    finally:
        ds.finalize()
        log.update(kind=args.kind, size=args.size, task=task, seconds=round(time.time() - t0))
        (Path(args.root) / "record_log.json").write_text(json.dumps(log, indent=1), encoding="utf-8")
    print(f"Xong: {len(log['kept'])} ván, {sum(k['frames'] for k in log['kept'])} khung hình -> {args.root}")


if __name__ == "__main__":
    main()
