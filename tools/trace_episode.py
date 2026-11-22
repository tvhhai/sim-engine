"""In log từng bước của 1 episode, từ đơn giản đến chi tiết, để làm quen cách hệ thống hoạt động.

Cách chạy (đứng ở thư mục chứa sim_engine, đã kích hoạt .venv):
    python -m sim_engine.trace_episode --seed 0 --level 1   # chỉ in lúc đổi phase (ít dòng nhất)
    python -m sim_engine.trace_episode --seed 0 --level 2   # bảng mỗi 10 bước
    python -m sim_engine.trace_episode --seed 0 --level 3   # bảng mỗi bước, thêm action và góc khớp
    python -m sim_engine.trace_episode --seed 2 --level 2 --every 5   # tự chọn số bước giữa 2 dòng

Ba mức log:
    Mức 1: chỉ in khi policy đổi phase. Trả lời "policy đi qua những giai đoạn nào, lúc nào?"
    Mức 2: thêm bảng định kỳ: phase, vị trí tay, vị trí hộp, độ mở kẹp, có đang kẹp không.
    Mức 3: in cả action (7 số policy ra lệnh) và góc 5 khớp (độ) ở mỗi bước.
"""

import argparse
import importlib

import numpy as np

from .. import config
from ..env import SO101PickPlaceEnv


def fmt(v):
    """Định dạng vector 3D thành chuỗi gọn."""
    return "(" + ", ".join(f"{x:+.3f}" for x in v) + ")"


def main():
    parser = argparse.ArgumentParser(description="In log từng bước của 1 episode")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--level", type=int, choices=[1, 2, 3], default=1)
    parser.add_argument("--every", type=int, default=0,
                        help="in 1 dòng mỗi N bước (mặc định: mức 2 = 10, mức 3 = 1)")
    parser.add_argument("--policy", default=config.DEFAULT_POLICY)
    args = parser.parse_args()

    every = args.every or (10 if args.level == 2 else 1)
    module_name, class_name = args.policy.split(":")
    policy_cls = getattr(importlib.import_module(module_name), class_name)

    env = SO101PickPlaceEnv(render_images=False)   # tắt ảnh camera cho nhanh
    policy = policy_cls(env)

    obs, info = env.reset(seed=args.seed)
    policy.reset()
    print(f"Episode seed={args.seed} | hộp ban đầu {fmt(info['box_pos'])} | bát {fmt(info['bowl_pos'])}")
    print(f"Tối đa {config.MAX_EPISODE_STEPS} bước (~{config.MAX_EPISODE_STEPS / 30:.0f} giây)\n")

    if args.level >= 2:
        header = "bước | phase | tay (x, y, z)            | hộp (x, y, z)             | kẹp mở | đang kẹp"
        if args.level == 3:
            header += " | action[0..6]                                  | góc khớp 5 khớp (độ)"
        print(header)
        print("-" * len(header))

    last_phase = None
    for step in range(config.MAX_EPISODE_STEPS):
        # 1) Policy nhìn quan sát và quyết định action
        action = policy.select_action({"state": obs["state"], "images": {}, "task": config.DEFAULT_TASK})
        # 2) Env thực hiện action và trả về quan sát mới
        obs, reward, terminated, truncated, info = env.step(action)

        phase = getattr(policy, "_phase", None)
        phase_changed = phase != last_phase
        last_phase = phase

        if args.level == 1:
            if phase_changed:
                print(f"bước {step:3d}: bắt đầu phase {phase}")
        else:
            if phase_changed or step % every == 0:
                mark = "*" if phase_changed else " "   # dấu * = vừa đổi phase
                line = (f"{step:4d}{mark}| {phase!s:>5} | {fmt(info['ee_pos'])} | {fmt(info['box_pos'])} | "
                        f"{obs['state'][6]:6.1f} | {str(env._grasped):>8}")
                if args.level == 3:
                    line += " | " + np.array2string(action, precision=2, suppress_small=True, floatmode="fixed")
                    line += " | " + np.array2string(obs["state"][:5], precision=1, suppress_small=True, floatmode="fixed")
                print(line)

        if terminated or truncated:
            break

    print()
    print(f"Kết thúc ở bước {step + 1}: {'THÀNH CÔNG' if info['success'] else 'THẤT BẠI'} "
          f"| hộp cách bát {info['box_to_bowl_dist']:.3f} m")
    if args.level >= 2:
        print("Chú thích: dấu * = vừa đổi phase | 'kẹp mở' từ 0 (đóng) đến 100 (mở) | 'đang kẹp' = cả hai hàm đang chạm hộp (kẹp thật bằng ma sát)")
    env.close()


if __name__ == "__main__":
    main()
