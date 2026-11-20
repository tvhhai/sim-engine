"""Quét vùng GẮP ĐƯỢC THẬT của cánh tay: đặt hộp ở lưới điểm theo toạ độ cực quanh trục khớp 1, chạy trọn một ván bằng policy
hiện tại, ghi lại ván nào thành công. Kết quả ghi vào docs/pick_scan.json; replay vẽ vùng gắp được từ đó (không còn vẽ theo
tầm với hình học của IK) và env lấy hình quạt đặt hộp ngẫu nhiên từ config.PICK_REGION (đặt dựa trên bản đồ này).

Cách dùng (đứng ở thư mục chứa sim_engine, đã kích hoạt .venv):
    python -m sim_engine.tools.scan_pick_region                    # quét r 0.08-0.42 m (cách 2 cm), góc -130..+120° (cách 10°)
    python -m sim_engine.tools.scan_pick_region --workers 5        # chạy song song 5 tiến trình
Góc đo từ hướng tay (trục +y) về phía +x: x = trục_x + r*sin(góc), y = trục_y + r*cos(góc).
"""

import argparse
import contextlib
import io
import json
import math
import multiprocessing as mp
import os

import numpy as np

from .. import config
from ..env import SO101PickPlaceEnv

PAN = np.array(config.PICK_PAN_XY, dtype=float)


def grid(r_min, r_max, r_step, a_min, a_max, a_step):
    bowl = np.array(config.BOWL_DEFAULT_POS[:2], dtype=float)
    pts = []
    for r in np.arange(r_min, r_max + 1e-9, r_step):
        for a in range(a_min, a_max + 1, a_step):
            th = math.radians(a)
            xy = PAN + r * np.array([math.sin(th), math.cos(th)])
            if abs(xy[0]) > config.TABLE_HALF[0] - 0.03 or abs(xy[1]) > config.TABLE_HALF[1] - 0.03:
                continue                                                  # sát mép bàn
            if np.linalg.norm(xy - bowl) < config.BOX_BOWL_CLEARANCE:
                continue                                                  # sát bát
            pts.append((round(float(r), 3), a, np.round(xy, 4)))      # làm tròn 4 số lẻ NGAY ở đây: seed cố định dùng đúng toạ độ này nên tái hiện được từng điểm hỏng
    return pts


def run_points(pts):
    from ..policies.scripted_policy import ScriptedPolicy
    env = SO101PickPlaceEnv(render_images=False, end_on_success=False)
    rows = []
    for r, a, xy in pts:
        policy = ScriptedPolicy(env)
        obs, info = env.reset(seed=0, options={"box_xy": [float(xy[0]), float(xy[1])]})
        policy.reset()
        for t in range(config.MAX_EPISODE_STEPS):
            with contextlib.redirect_stdout(io.StringIO()):
                act = policy.select_action({"state": obs["state"], "images": {}, "task": config.DEFAULT_TASK})
            obs, _, _, trunc, info = env.step(act)
            if trunc or policy._phase > 10:
                break
        rows.append({"r": r, "a": a, "xy": [round(float(xy[0]), 4), round(float(xy[1]), 4)], "ok": bool(info["success"]), "steps": t})
    env.close()
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--r", type=float, nargs=3, default=[0.08, 0.42, 0.02], metavar=("MIN", "MAX", "STEP"))
    ap.add_argument("--angle", type=int, nargs=3, default=[-130, 120, 10], metavar=("MIN", "MAX", "STEP"))
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--update-seeds", action="store_true", help="ghi lại khối SCAN-SEEDS trong config.py: các điểm quét hỏng trong vùng gắp thành seed cố định 7..")
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--out", default=os.path.join(here, "docs", "pick_scan.json"))
    args = ap.parse_args()
    pts = grid(*args.r, *args.angle)
    chunks = [pts[k::args.workers] for k in range(args.workers)]
    print(f"quét {len(pts)} điểm bằng {args.workers} tiến trình ...", flush=True)
    with mp.get_context("spawn").Pool(args.workers) as pool:
        parts = pool.map(run_points, chunks)
    rows = sorted((r for part in parts for r in part), key=lambda d: (d["r"], d["a"]))
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"pan_xy": PAN.tolist(), "angle_from": "+y về phía +x", "points": rows}, f, ensure_ascii=False)
    ok = sum(r["ok"] for r in rows)
    print(f"thành công {ok}/{len(rows)} điểm. Đã ghi {args.out}")
    if args.update_seeds:
        update_seeds(rows)


def update_seeds(rows):
    """Viết lại khối SCAN-SEEDS trong config.py: seed 7.. = điểm quét hỏng nằm trong PICK_REGION; sau đó là seed pan ±100/±110 (r 15, 25 cm)."""
    R = config.PICK_REGION
    fails = sorted((p for p in rows if not p["ok"] and R["r_min"] - 1e-9 <= p["r"] <= R["r_max"] + 1e-9
                    and R["angle_min_deg"] <= p["a"] <= R["angle_max_deg"]), key=lambda p: (p["r"], p["a"]))
    lines, seed = [], 7
    for p in fails:
        lines.append(f'    {seed}: ("quét hỏng: r = {p["r"] * 100:.0f} cm, góc {p["a"]:+d}°", ({p["xy"][0]:.4f}, {p["xy"][1]:.4f})),')
        seed += 1
    first_pan = seed
    for a in (100, 110, -100, -110):
        for r in (0.15, 0.25):
            th = math.radians(a)
            x, y = PAN[0] + r * math.sin(th), PAN[1] + r * math.cos(th)
            lines.append(f'    {seed}: ("pan {a:+d}°, r = {r * 100:.0f} cm (kiểm tra vùng gắp đối xứng)", ({x:.4f}, {y:.4f})),')
            seed += 1
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.py")
    with open(path, encoding="utf-8") as f:
        text = f.read()
    a0 = text.index("    # >>> SCAN-SEEDS")
    a1 = text.index("    # <<< SCAN-SEEDS")
    nl = chr(10)
    head = text[a0:text.index(nl, a0) + 1]
    with open(path, "w", encoding="utf-8") as f:
        f.write(text[:a0] + head + nl.join(lines) + nl + text[a1:])
    print(f"đã ghi {len(fails)} seed quét hỏng (7-{first_pan - 1}) và 8 seed pan ({first_pan}-{seed - 1}) vào config.py")


if __name__ == "__main__":
    main()
