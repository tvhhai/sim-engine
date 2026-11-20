"""Chấm điểm policy trên nhiều seed và xem timeline từng phase, để debug.

Cách chạy (đứng ở thư mục chứa sim_engine, đã kích hoạt .venv):
    python -m sim_engine.check_scripted                    # chấm seed 0-9 (mặc định)
    python -m sim_engine.check_scripted --seeds 0-49       # chấm 50 seed
    python -m sim_engine.check_scripted --seeds 2,3 -v     # chỉ seed 2 và 3, in timeline chi tiết
    python -m sim_engine.check_scripted --policy sim_engine.policies.my_policy:MyPolicy

"Thành công" được tính trong SO101PickPlaceEnv._get_info() (file env.py):
hộp nằm trong bán kính bát (theo chiều ngang) và đúng độ cao. Mỗi seed cho ra một vị trí hộp
khác nhau (dịch ngẫu nhiên ±3 cm), cùng seed thì kết quả luôn lặp lại giống hệt.
"""

import argparse
import importlib

import numpy as np

from .. import config
from ..env import SO101PickPlaceEnv


def parse_seeds(text):
    """Đổi chuỗi như '0-9' hoặc '2,3' hoặc '0-4,10' thành danh sách seed."""
    seeds = []
    for part in text.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            seeds.extend(range(int(a), int(b) + 1))
        elif part:
            seeds.append(int(part))
    return seeds


def load_policy_class(path):
    """Nạp class policy từ chuỗi 'ten.module:TenClass'."""
    module_name, class_name = path.split(":")
    return getattr(importlib.import_module(module_name), class_name)


def run_episode(env, policy, seed):
    """Chạy 1 episode với seed cho trước. Trả về (info cuối, số bước, timeline, cảnh báo)."""
    obs, info = env.reset(seed=seed)
    policy.reset()

    box_start = info["box_pos"].copy()   # vị trí hộp lúc bắt đầu
    timeline = []                        # mỗi phần tử: ghi lại lúc policy đổi phase
    last_phase = None
    warnings = []
    ever_grasped = False                 # đã từng kẹp được hộp chưa

    for step in range(config.MAX_EPISODE_STEPS):
        action = policy.select_action(
            {"state": obs["state"], "images": obs.get("images", {}), "task": config.DEFAULT_TASK}
        )
        obs, reward, terminated, truncated, info = env.step(action)

        # Ghi lại mỗi lần đổi phase (nếu policy có thuộc tính _phase, như ScriptedPolicy)
        phase = getattr(policy, "_phase", None)
        if phase != last_phase:
            timeline.append({
                "step": step, "phase": phase,
                "ee": info["ee_pos"].round(3), "box": info["box_pos"].round(3),
                "grasped": env._grasped,
            })
            last_phase = phase

        # Cảnh báo: hộp bị xê dịch trước khi được kẹp lần đầu (nghĩa là tay đã gạt trúng hộp).
        # Sau khi đã kẹp thì hộp được mang đi nên dịch chuyển là bình thường, không cảnh báo nữa.
        ever_grasped = ever_grasped or env._grasped
        if not ever_grasped and not warnings:
            moved = float(np.linalg.norm(info["box_pos"][:2] - box_start[:2]))
            if moved > 0.01:
                warnings.append(
                    f"hộp bị dịch {moved * 100:.1f} cm trước khi kẹp (bước {step}, phase {phase}) "
                    f"-> có thể tay đã gạt trúng hộp"
                )

        if terminated or truncated or (phase is not None and phase > 10):
            break

    return info, step + 1, timeline, warnings


def main():
    parser = argparse.ArgumentParser(description="Chấm điểm policy trên nhiều seed")
    parser.add_argument("--seeds", default="0-9", help="ví dụ 0-9, hoặc 2,3, hoặc 0-4,10")
    parser.add_argument("--policy", default="sim_engine.policies.scripted_policy:ScriptedPolicy",
                        help="module:Class của policy cần chấm")
    parser.add_argument("--box-range", type=float, default=None, help="hộp ngẫu nhiên trong ± giá trị này (mét)")
    parser.add_argument("--bowl-range", type=float, default=None, help="bát ngẫu nhiên trong ± giá trị này (mét)")
    parser.add_argument("-v", "--verbose", action="store_true", help="in timeline từng phase")
    args = parser.parse_args()

    seeds = parse_seeds(args.seeds)
    policy_cls = load_policy_class(args.policy)
    env = SO101PickPlaceEnv(render_images=False, end_on_success=False,
                            box_range=args.box_range, bowl_range=args.bowl_range)   # chạy đủ mọi phase, chấm ở bước cuối
    policy = policy_cls(env)

    print(f"Policy: {args.policy} | số seed: {len(seeds)}")
    ok = 0
    failed = []
    for seed in seeds:
        info, steps, timeline, warnings = run_episode(env, policy, seed)
        ok += int(info["success"])
        status = "OK  " if info["success"] else "FAIL"
        print(f"  seed {seed:3d}: {status} | {steps:3d} bước | hộp cách bát {info['box_to_bowl_dist']:.3f} m")
        if not info["success"]:
            failed.append(seed)
        for w in warnings:
            print(f"      ! {w}")
        if args.verbose or not info["success"]:
            for t in timeline:
                print(f"      bước {t['step']:3d} -> phase {t['phase']} | tay {t['ee']} | "
                      f"hộp {t['box']} | đang kẹp: {t['grasped']}")

    print(f"\nKết quả: {ok}/{len(seeds)} thành công ({100 * ok / len(seeds):.0f}%)")
    if failed:
        print(f"Seed thất bại: {failed}")
    env.close()


if __name__ == "__main__":
    main()
