"""Quét vùng GẮP ĐƯỢC THẬT của cánh tay: đặt hộp ở lưới điểm theo toạ độ cực quanh trục khớp 1, chạy trọn một ván bằng policy
hiện tại, ghi lại ván nào thành công. Kết quả ghi vào docs/pick_scan.json; replay vẽ vùng gắp được từ đó (không còn vẽ theo
tầm với hình học của IK) và env lấy hình quạt đặt hộp ngẫu nhiên từ config.PICK_REGION (đặt dựa trên bản đồ này).

Cách dùng (đứng ở thư mục chứa sim_engine, đã kích hoạt .venv):
    python -m sim_engine.tools.scan_pick_region                    # quét r 0.08-0.42 m (cách 2 cm), góc -130..+120° (cách 10°)
    python -m sim_engine.tools.scan_pick_region --workers 5        # chạy song song 5 tiến trình
Góc đo từ hướng tay (trục +y) về phía +x: x = trục_x + r*sin(góc), y = trục_y + r*cos(góc).
"""

import argparse
import re
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
        _, info, _ = env.settle(act, info)                       # hộp còn lắc trong bát thì chờ nằm yên rồi mới chấm
        rows.append({"r": r, "a": a, "xy": [round(float(xy[0]), 4), round(float(xy[1]), 4)], "ok": bool(info["success"]), "steps": t})
    env.close()
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--r", type=float, nargs=3, default=[0.08, 0.42, 0.02], metavar=("MIN", "MAX", "STEP"))
    ap.add_argument("--angle", type=int, nargs=3, default=[-130, 120, 10], metavar=("MIN", "MAX", "STEP"))
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--seeds-only", action="store_true", help="không quét lại: đọc docs/pick_scan.json rồi chỉ cập nhật khối SCAN-SEEDS")
    ap.add_argument("--update-seeds", action="store_true", help="ghi lại khối SCAN-SEEDS trong config.py: các điểm quét hỏng trong vùng gắp thành seed cố định 7..")
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--out", default=os.path.join(here, "docs", "pick_scan.json"))
    args = ap.parse_args()
    if args.seeds_only:
        with open(args.out, encoding="utf-8") as f:
            update_seeds(json.load(f)["points"])
        return
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
    """Viết lại khối SCAN-SEEDS trong config.py, theo thứ tự:
        1. điểm quét HỎNG nằm trong PICK_REGION (tái hiện từng điểm);
        2. seed pan +-100/+-110 (r 15, 25 cm): kiểm tra vùng gắp đối xứng;
        3. điểm NGOÀI vùng gắp nhưng quét GẮP ĐƯỢC (r 8, 10, 38, 40 cm; 5 hướng mỗi vòng): để xem xét rồi mới kết luận vùng chắc chắn không gắp;
        4. hộp NGOÀI TẦM chắc chắn (khớp 1 không xoay tới, góc quá +-115 độ): policy về home ngay, không chờ hết hạn."""
    R = config.PICK_REGION
    in_region = lambda p: R["r_min"] - 1e-9 <= p["r"] <= R["r_max"] + 1e-9 and R["angle_min_deg"] <= p["a"] <= R["angle_max_deg"]
    fails = sorted((p for p in rows if not p["ok"] and in_region(p)), key=lambda p: (p["r"], p["a"]))
    lines, seed = [], 7
    for p in fails:
        lines.append(f'    {seed}: ("quét hỏng: r = {p["r"] * 100:.0f} cm, góc {p["a"]:+d}°", ({p["xy"][0]:.4f}, {p["xy"][1]:.4f})),')
        seed += 1
    n_fail = len(fails)
    for a in (100, 110, -100, -110):
        for r in (0.15, 0.25):
            th = math.radians(a)
            x, y = PAN[0] + r * math.sin(th), PAN[1] + r * math.cos(th)
            lines.append(f'    {seed}: ("pan {a:+d}°, r = {r * 100:.0f} cm (kiểm tra vùng gắp đối xứng)", ({x:.4f}, {y:.4f})),')
            seed += 1
    n_outside = 0
    for ring in (0.08, 0.10, 0.38, 0.40):
        ok_pts = [p for p in rows if p["ok"] and abs(p["r"] - ring) < 1e-6 and not in_region(p) and abs(p["a"]) <= config.PAN_LIMIT_DEG]
        for want in (-90, -45, 0, 45, 90):
            if ok_pts:
                p = min(ok_pts, key=lambda q: abs(q["a"] - want))
                ok_pts.remove(p)
                lines.append(f'    {seed}: ("ngoài vùng nhưng gắp được: r = {p["r"] * 100:.0f} cm, góc {p["a"]:+d}°", ({p["xy"][0]:.4f}, {p["xy"][1]:.4f})),')
                seed += 1
                n_outside += 1
    n_unreach = 0
    for a, r in ():                                        # vùng đỏ (chắc chắn không gắp được) không còn seed: bỏ các seed ngoài tầm
        th = math.radians(a)
        x, y = PAN[0] + r * math.sin(th), PAN[1] + r * math.cos(th)
        lines.append(f'    {seed}: ("ngoài tầm (khớp 1 không xoay tới): góc {a:+d}°, r = {r * 100:.0f} cm", ({x:.4f}, {y:.4f})),')
        seed += 1
        n_unreach += 1
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.py")
    with open(path, encoding="utf-8") as f:
        text = f.read()
    a0 = text.index("    # >>> SCAN-SEEDS")
    a1 = text.index("    # <<< SCAN-SEEDS")
    nl = chr(10)
    head = text[a0:text.index(nl, a0) + 1]
    # GỘP: các seed đã có GIỮ NGUYÊN số và vị trí (kể cả khi chúng đã hết hỏng, để còn xem lại quá trình sửa); điểm mới được thêm ở cuối.
    pat = re.compile(r'^\s+(\d+): \("(.*)", \((-?[\d.]+), (-?[\d.]+)\)\),\s*$')
    existing = {}
    for ln in text[a0:a1].splitlines():
        m = pat.match(ln)
        if m:
            existing[int(m.group(1))] = (m.group(2), float(m.group(3)), float(m.group(4)))
    kept = [(sd, name, x, y) for sd, (name, x, y) in sorted(existing.items())]
    nxt = max(max(existing, default=6), config.FIXED_SEED_FIRST_EXTRA - 1) + 1     # seed mới từ 200 trở đi: không đè lên seed 0-199
    added = 0
    for ln in lines:
        m = pat.match(ln)
        name, x, y = m.group(2), float(m.group(3)), float(m.group(4))
        if any(abs(x - ex) < 1e-3 and abs(y - ey) < 1e-3 for _, _, ex, ey in kept):
            continue
        kept.append((nxt, name, x, y))
        nxt += 1
        added += 1
    out = [f'    {sd}: ("{name}", ({x:.4f}, {y:.4f})),' for sd, name, x, y in kept]
    with open(path, "w", encoding="utf-8") as f:
        f.write(text[:a0] + head + nl.join(out) + nl + text[a1:])
    print(f"seed cố định: giữ {len(existing)} seed cũ, thêm {added} seed mới (tổng 1-{nxt - 1})")


if __name__ == "__main__":
    main()
