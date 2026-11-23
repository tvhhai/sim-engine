"""Chạy các seed với vật KHÔNG phải hộp (bóng, hộp cỡ khác) và ghi replay để quan sát.

Cách dùng (đứng ở thư mục chứa sim_engine, đã kích hoạt .venv):
    python -m sim_engine.tools.object_replay --kind sphere --size 0.03                     # chấm cả 774 seed rồi ghi replay các seed hỏng + 12 seed đối chứng
    python -m sim_engine.tools.object_replay --kind sphere --size 0.03 --seeds 63,87,2299  # chỉ vài seed (ghi luôn, không chấm lại)
    python -m sim_engine.tools.object_replay --kind sphere --size 0.04 --name sphere4      # đặt tên file kết quả

Vật được đặt đúng vị trí hộp của từng seed (bát theo seed). Kết quả: results/replays/<tên>.html và .json; Studio liệt kê mọi file
trong results/replays/ ở danh sách "Nguồn replay".
"""

import argparse
import contextlib
import io
import json
import math
import os
import time
from multiprocessing import get_context

from .. import config, paths


def _seed_scene(seed):
    from ..env import SO101PickPlaceEnv
    base = SO101PickPlaceEnv(render_images=False, end_on_success=False)
    base.reset(seed=seed)
    xy = [float(v) for v in base.get_box_pos()[:2]]
    bowl = [float(v) for v in base.get_bowl_pos()[:2]]
    base.close()
    return xy, bowl


def _make_env(xy, kind, size):
    from ..env import SO101PickPlaceEnv
    obj = {"id": "target", "type": kind, "x": xy[0], "y": xy[1], "yaw": 0, "size": size, "color": "#e5484d"}
    return SO101PickPlaceEnv(render_images=False, end_on_success=False, lab_objects=[obj])


def sweep_seed(args):
    """Chấm một seed (không ghi từng bước): trả {seed, ok, steps, r, ang}."""
    from ..policies import default_policy
    seed, kind, size = args
    xy, bowl = _seed_scene(seed)
    env = _make_env(xy, kind, size)
    policy = default_policy(env)
    obs, _ = env.reset(seed=seed, options={"bowl_xy": bowl})
    policy.reset()
    info = {}
    for t in range(config.MAX_EPISODE_STEPS):
        with contextlib.redirect_stdout(io.StringIO()):
            action = policy.select_action({"state": obs["state"], "images": {}, "task": ""})
        obs, _, _, trunc, info = env.step(action)
        if trunc or policy._phase > 10:
            break
    _, info, _ = env.settle(action, info)
    env.close()
    dx, dy = xy[0] - config.PICK_PAN_XY[0], xy[1] - config.PICK_PAN_XY[1]
    return {"seed": seed, "ok": bool(info["success"]), "steps": t, "r": round(math.hypot(dx, dy), 3), "ang": round(math.degrees(math.atan2(dx, dy)))}


def record_seed(args):
    """Chạy một seed và ghi từng bước (như make_replay), kèm nhãn mô tả vì sao hỏng."""
    from ..policies import default_policy
    from .make_replay import Recorder
    seed, kind, size, label = args
    xy, bowl = _seed_scene(seed)
    env = _make_env(xy, kind, size)
    policy = default_policy(env)
    with contextlib.redirect_stdout(io.StringIO()):
        ep = Recorder(env, policy).run(seed, options={"bowl_xy": bowl})
    frames = ep["frames"]
    z0 = frames[0]["box"][2]
    lift = (max(f["box"][2] for f in frames) - z0) * 100
    grasped = any(f["gr"] for f in frames)
    fell = frames[-1]["box"][2] < 0.3                                    # rơi khỏi mặt bàn (z mặt bàn 0,42)
    r = math.hypot(xy[0] - config.PICK_PAN_XY[0], xy[1] - config.PICK_PAN_XY[1])
    tags = [label, "THÀNH CÔNG" if ep["success"] else "HỎNG", f"hộp cách trục khớp 1 {r * 100:.0f} cm"]
    if not ep["success"]:
        tags.append("đã kẹp được" if grasped else "chưa kẹp được")
        tags.append(f"nhấc lên tối đa {lift:.1f} cm")
    if fell:
        tags.append("RƠI KHỎI BÀN")
    kind_vi = "bóng" if kind == "sphere" else "hộp"
    ep.update(name=f"seed {seed} · {kind_vi} {size * 100:g} cm · {'OK' if ep['success'] else 'FAIL'}", group=label, tags=tags)
    return ep, env.bowl_color


def main():
    parser = argparse.ArgumentParser(description="Chạy các seed với vật không phải hộp và ghi replay để quan sát")
    parser.add_argument("--kind", choices=["sphere", "box"], default="sphere")
    parser.add_argument("--size", type=float, default=0.03, help="cạnh hộp / đường kính bóng (m)")
    parser.add_argument("--seeds", default=None, help="ví dụ 63,87,2299: ghi thẳng các seed này, không chấm cả bộ")
    parser.add_argument("--sweep-json", default=None, help="kết quả chấm đã có (list {seed, ok}) để khỏi chấm lại")
    parser.add_argument("--ok-sample", type=int, default=12, help="số seed thành công (cách đều) ghi thêm để đối chứng")
    parser.add_argument("--name", default=None, help="tên file kết quả trong results/replays (mặc định <kind><cm>)")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = parser.parse_args()
    from .make_replay import parse_seeds, write_replay

    name = args.name or f"{args.kind}{args.size * 100:g}"
    ctx = get_context("spawn")
    jobs = []
    if args.seeds:
        jobs = [(s, "Chọn tay") for s in parse_seeds(args.seeds)]
    else:
        if args.sweep_json:
            rows = json.load(open(args.sweep_json, encoding="utf-8"))
        else:
            seeds = list(range(200)) + sorted(k for k in config.BOX_FIXED_SEEDS if k >= 200) + list(range(2000, 2500))
            t0 = time.time()
            with ctx.Pool(args.workers) as pool:
                rows = pool.map(sweep_seed, [(s, args.kind, args.size) for s in seeds], chunksize=4)
            print(f"chấm {len(rows)} seed trong {time.time() - t0:.0f} s")
        bad = [r["seed"] for r in rows if not r["ok"]]
        good = [r["seed"] for r in rows if r["ok"]]
        print(f"{args.kind} {args.size * 100:g} cm: gắp được {len(good)}/{len(rows)}; hỏng {len(bad)}")
        step = max(1, len(good) // max(1, args.ok_sample))
        jobs = [(s, "Hỏng") for s in bad] + [(s, "Đối chứng thành công") for s in good[::step][:args.ok_sample]]
    with ctx.Pool(args.workers) as pool:
        results = pool.map(record_seed, [(s, args.kind, args.size, label) for s, label in jobs], chunksize=2)
    episodes = [ep for ep, _ in results]
    env = _make_env([0.15, 0.0], args.kind, args.size)
    out_dir = paths.RESULTS / "replays"
    out_dir.mkdir(parents=True, exist_ok=True)
    html = out_dir / f"{name}.html"
    data = write_replay(env, episodes, str(html), config.DEFAULT_POLICY)
    data["scene"]["title"] = f"{'Bóng' if args.kind == 'sphere' else 'Hộp'} {args.size * 100:g} cm · {len(episodes)} ván"
    with open(out_dir / f"{name}.json", "w", encoding="utf-8") as f:
        json.dump(data, f, separators=(",", ":"), ensure_ascii=False)
    ok = sum(e["success"] for e in episodes)
    print(f"Đã ghi {html} và .json: {len(episodes)} ván ({ok} thành công)")


if __name__ == "__main__":
    main()
