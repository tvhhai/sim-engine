"""Nghiên cứu VÙNG HỖN HỢP: chạy lại từng điểm quét nằm ngoài hình quạt gắp được (và các điểm hỏng trong quạt), phân loại NGUYÊN NHÂN hỏng
(chưa kẹp được / kẹp được nhưng không nhấc / nhấc rồi tuột / hết bước ở phase nào) và gom theo vòng bán kính, bên trái-phải.
Kết quả: results/mixed_zone_study.json + bảng in ra.

    python -m sim_engine.tools.mixed_zone_study [--workers N]
"""
import argparse, collections, contextlib, io, json, math, multiprocessing as mp, os

import numpy as np

from .. import box_sampling, config, paths
from ..env import SO101PickPlaceEnv
from ..policies import default_policy

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS = str(paths.RESULTS)


def load_points():
    pts = {}
    for name in ("pick_scan.json", "pick_scan_outer.json", "pick_scan_inner.json"):
        with open(os.path.join(DOCS, name), encoding="utf-8") as f:
            for p in json.load(f)["points"]:
                pts[(round(p["r"], 3), p["a"])] = p
    return pts


def in_sector(p):
    R = config.PICK_REGION
    return R["r_min"] - 1e-9 <= p["r"] <= R["r_max"] + 1e-9 and R["angle_min_deg"] <= p["a"] <= R["angle_max_deg"]


def run_one(p):
    env = SO101PickPlaceEnv(render_images=False, end_on_success=False)
    pol = default_policy(env)
    obs, info = env.reset(seed=0, options={"box_xy": [float(p["xy"][0]), float(p["xy"][1])]})
    pol.reset()
    z0 = float(env.get_box_pos()[2])
    grasped = False
    zmax = z0
    exits = collections.Counter()
    last_phase = -1
    for t in range(config.MAX_EPISODE_STEPS):
        with contextlib.redirect_stdout(io.StringIO()):
            act = pol.select_action({"state": obs["state"], "images": {}, "task": config.DEFAULT_TASK})
        ph = pol._phase
        obs, _, _, trunc, info = env.step(act)
        if pol._phase != ph and pol.last_exit in ("timeout", "stall"):
            exits[ph] += 1
        grasped |= bool(env._grasped)
        zmax = max(zmax, float(env.get_box_pos()[2]))
        if trunc or pol._phase > 10:
            break
    _, info, _ = env.settle(act, info)                       # hộp còn lắc trong bát thì chờ nằm yên rồi mới chấm
    ok = bool(info.get("success"))
    if ok:
        kind = "ok"
    elif not grasped:
        kind = "chưa kẹp được"
    elif zmax < z0 + 0.03:
        kind = "kẹp được nhưng không nhấc"
    else:
        kind = "nhấc rồi tuột/không vào bát"
    return {"r": p["r"], "a": p["a"], "ok": ok, "kind": kind, "steps": t, "timeouts": sorted(exits), "phase_end": pol._phase}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    pts = [p for p in load_points().values() if abs(p["a"]) <= config.PAN_LIMIT_DEG and p["r"] <= config.PICK_UNREACHABLE_RADIUS
           and (not in_sector(p) or not p["ok"])]
    print(f"chạy {len(pts)} điểm trong vùng hỗn hợp (ngoài quạt hoặc hỏng trong quạt) ...")
    with mp.get_context("spawn").Pool(args.workers) as pool:
        rows = pool.map(run_one, pts, chunksize=4)
    json.dump(rows, open(os.path.join(DOCS, "mixed_zone_study.json"), "w", encoding="utf-8"), ensure_ascii=False)
    kinds = ["ok", "chưa kẹp được", "kẹp được nhưng không nhấc", "nhấc rồi tuột/không vào bát"]
    print(f"{'r(cm)':>6} {'bên':>4} " + " ".join(f"{k[:14]:>14}" for k in kinds))
    for r in sorted({x["r"] for x in rows}):
        for side, f in (("-x", lambda a: a < 0), ("0", lambda a: a == 0), ("+x", lambda a: a > 0)):
            sel = [x for x in rows if x["r"] == r and f(x["a"])]
            if sel:
                c = collections.Counter(x["kind"] for x in sel)
                print(f"{r * 100:6.0f} {side:>4} " + " ".join(f"{c.get(k, 0):14d}" for k in kinds))
    tc = collections.Counter(ph for x in rows if not x["ok"] for ph in x["timeouts"])
    print("phase hết hạn trong các ván hỏng:", dict(sorted(tc.items())))


if __name__ == "__main__":
    main()
