"""Chấm một dải seed lớn (mặc định 0-199) rồi CHỌN bộ seed để xem lại trong replay.

Cách dùng (đứng ở thư mục chứa sim_engine, đã kích hoạt .venv):
    python -m sim_engine.select_seeds                    # chấm seed 0-199 bằng KẸP THẬT, chọn 30 seed
    python -m sim_engine.select_seeds --count 40         # chọn 40 seed
    python -m sim_engine.select_seeds --pool 0-499       # chấm dải lớn hơn

Cách chọn (để 30 seed trong replay không trùng lặp và đáng xem):
    1. Ưu tiên các seed HỎNG, seed có PHASE HẾT THỜI GIAN (timeout), seed CHẬM bất thường. Mỗi kiểu lỗi
       (chữ ký = hỏng/ok + kiểu hỏng + các phase timeout) có ít nhất một đại diện.
    2. Trong cùng ưu tiên, chọn seed có vị trí hộp (và bát nếu có random bát) XA các seed đã chọn nhất,
       để không lấy các vị trí chỉ lệch nhau một chút.
    3. Còn chỗ thì lấp bằng các seed xa nhau nhất.
    4. Nếu số seed hỏng nhiều hơn --count thì đưa TẤT CẢ seed hỏng lên (nhóm "hỏng còn lại" sau 30 seed đầu).

Kết quả ghi vào docs/replay_seeds.json (make_replay đọc file này) và in báo cáo ra màn hình.
Bộ chấm 200 seed (pool) không đổi: cùng seed thì cùng vị trí hộp.
"""

import argparse
import collections
import contextlib
import importlib
import io
import json
import math
import os
import time

import numpy as np

from .. import config
from .. import box_sampling
from ..box_sampling import fixed_seed_name
from ..env import SO101PickPlaceEnv
from ..policies.scripted_policy import ScriptedPolicy
from . import make_replay as _mr
from .make_replay import Recorder, parse_seeds, parallel_map, default_workers


# Hai trường hợp cực biên luôn đứng đầu danh sách replay: hộp XA nhất và GẦN nhất mà cánh tay phải xử lý.
# (đặt tay vị trí hộp qua env.reset(options=...), nên không phụ thuộc vùng random ±3 cm)
# Các trường hợp cực biên là SEED 1-6 (config.BOX_FIXED_SEEDS): env.reset(seed=k) đặt hộp đúng vị trí đã định, nên chúng được chấm
# cùng 200 seed như mọi seed khác và luôn có mặt trong replay.
SHOULDER_XY = (0.0, -0.15)      # trục khớp 1
MOVE_PHASES = (0, 1, 2, 3, 5, 6, 7, 9)   # các phase kết thúc khi tay tới đích (hoặc quá PHASE_TIMEOUT_STEPS bước)
PHASE_TIMEOUT = ScriptedPolicy.PHASE_TIMEOUT_STEPS


def failed(r):
    """Hỏng THẬT: không thành công và không phải seed ngoài tầm (ngoài tầm là bỏ qua có chủ ý, không tính là hỏng)."""
    return (not r["success"]) and not r.get("skip")


def evaluate_seed(rec, seed):
    """Chạy 1 seed, trả về dict thống kê dùng để chọn và để báo cáo."""
    with contextlib.redirect_stdout(io.StringIO()):
        ep = rec.run(seed)
    frames = ep["frames"]
    # timeout thật = phase kết thúc vì quá hạn hoặc vì KẸT (policy ghi lý do ở `last_exit`), không phải vì tay đã tới nơi
    timeouts = sorted({f["ph"] for f in frames if f["phn"] != f["ph"] and f.get("ex") in ("timeout", "stall")})
    retries = sum(1 for f in frames if f["ph"] == 4 and f["phn"] == 2)                 # phase 4 không kẹp được, hạ lại phase 2
    drops = sum(1 for f in frames if f["ph"] in (5, 6, 7) and f["phn"] == 1)           # làm rớt hộp khi đang mang, quay lại gắp
    regrasps = sum(1 for f in frames if f["ph"] == 9 and f["phn"] == 1)                # rút lên rồi thấy hộp ngoài bát, gắp lại
    z0 = frames[0]["box"][2]
    roll_flips = max((f.get("rf", 0) for f in frames), default=0)          # đổi roll vì chạm bát
    bowl_avoid = any(f.get("ba") for f in frames)                         # roll chọn sẵn để tránh bát
    ever_grasped = any(f["gr"] for f in frames)
    lifted = max(f["box"][2] for f in frames) > z0 + 0.03
    if ep["success"]:
        kind = "ok"
    elif not ever_grasped:
        kind = "chưa kẹp được"
    elif not lifted:
        kind = "kẹp được nhưng không nhấc lên"
    else:
        kind = "nhấc lên rồi tuột/không vào bát"
    skip = box_sampling.is_unreachable(frames[0]["box"][:2])
    if skip:
        kind = "ngoài tầm (bỏ qua có chủ ý)"
    cam_min = min((f["cd"] for f in frames if "cd" in f), default=None)          # khoảng cách camera cổ tay nhỏ nhất tới bàn/vai/cánh tay (mm)
    return {
        "cam_min": cam_min,
        "skip": skip,
        "seed": seed, "success": bool(ep["success"]), "steps": ep["steps"], "timeouts": timeouts, "retries": retries,
        "drops": drops, "regrasps": regrasps, "roll_flips": roll_flips, "bowl_avoid": bowl_avoid,
        "kind": kind, "box0": frames[0]["box"][:2], "bowl": ep["bowl"][:2],
    }


def _eval_task(task):
    """Chạy trong tiến trình con: chấm một seed bằng Recorder của tiến trình đó."""
    seed, _ = task
    return evaluate_seed(_mr._W["rec"], seed)


def features(rows, box_range, bowl_range, box_center=(0.15, 0.0)):
    """Toạ độ hộp (và bát nếu bát random) chuẩn hoá về cỡ [-1, 1] để đo "xa nhau" giữa các seed."""
    out = []
    for r in rows:
        f = [r["box0"][0] / 0.40, r["box0"][1] / 0.40]
        if bowl_range > 0:
            f += [(r["bowl"][0] + 0.15) / bowl_range, (r["bowl"][1] - 0.0) / bowl_range]
        out.append(np.array(f))
    return out


def farthest(cands, chosen, feat):
    """Trong `cands` chọn seed có khoảng cách tới tập `chosen` lớn nhất (lấy khoảng cách tới seed gần nhất)."""
    if not chosen:
        return cands[0]
    return max(cands, key=lambda i: min(np.linalg.norm(feat[i] - feat[j]) for j in chosen))


def reach(r):
    """Khoảng cách từ trục khớp 1 tới hộp / bát, và từ hộp tới bát (m)."""
    box = math.hypot(r["box0"][0] - SHOULDER_XY[0], r["box0"][1] - SHOULDER_XY[1])
    bowl = math.hypot(r["bowl"][0] - SHOULDER_XY[0], r["bowl"][1] - SHOULDER_XY[1])
    gap = math.hypot(r["box0"][0] - r["bowl"][0], r["box0"][1] - r["bowl"][1])
    return box, bowl, gap


def reach_picks(rows):
    """Seed ở các biên của tầm với: tay vươn tới hộp xa/gần nhất, hộp cách bát xa/gần nhất, và 4 tổ hợp ngược nhau.

    Trả về {chỉ số: [nhãn]}. Bát đứng yên nên "tay vươn tới bát" không đổi; khác nhau giữa các seed là vị trí hộp,
    nên 4 tổ hợp là: (hộp xa trục, hộp xa bát), (xa trục, gần bát), (gần trục, xa bát), (gần trục, gần bát).
    """
    rb = np.array([reach(r)[0] for r in rows])
    gp = np.array([reach(r)[2] for r in rows])
    zr = (rb - rb.mean()) / (rb.std() or 1)
    zg = (gp - gp.mean()) / (gp.std() or 1)
    picks = {}

    def add(i, label):
        picks.setdefault(int(i), []).append(label)

    add(rb.argmax(), "tay vươn tới hộp XA nhất")
    add(rb.argmin(), "tay vươn tới hộp GẦN nhất")
    add(gp.argmax(), "hộp cách bát XA nhất")
    add(gp.argmin(), "hộp cách bát GẦN nhất")
    add((zr + zg).argmax(), "hộp xa trục + xa bát")
    add((zr - zg).argmax(), "hộp xa trục + gần bát")
    add((-zr + zg).argmax(), "hộp gần trục + xa bát")
    add((-zr - zg).argmax(), "hộp gần trục + gần bát")
    return picks


def select(rows, count, feat, slow_steps):
    n = len(rows)

    def signature(i):
        r = rows[i]
        return (r["success"], r["kind"], tuple(r["timeouts"]), r["steps"] >= slow_steps,
                r["retries"] > 0, r["drops"] > 0, r["regrasps"] > 0)

    bad = [i for i in range(n) if failed(rows[i]) or rows[i]["timeouts"] or rows[i]["steps"] >= slow_steps
           or rows[i]["retries"] or rows[i]["drops"] or rows[i]["regrasps"]]
    chosen = [i for i in range(n) if rows[i]["seed"] in config.BOX_FIXED_SEEDS]       # seed 1-6: các trường hợp cực biên, luôn có mặt
    # 0) các biên của tầm với (hộp xa/gần trục, xa/gần bát và 4 tổ hợp)
    picks = reach_picks(rows)
    for i in picks:
        if i not in chosen:
            chosen.append(i)
    # 1) MỌI seed hỏng (nếu có chỗ), rồi mỗi chữ ký lỗi (timeout / kẹp lại / làm rớt / gắp lại / chậm) có một đại diện
    for i in [i for i in range(n) if failed(rows[i])]:
        if i not in chosen and len(chosen) < count:
            chosen.append(i)
    groups = collections.defaultdict(list)
    for i in bad:
        groups[signature(i)].append(i)
    for key in sorted(groups, key=lambda k: len(groups[k])):
        if len(chosen) >= count:
            break
        if not any(signature(j) == key for j in chosen):
            chosen.append(farthest(groups[key], chosen, feat))
    # 2) còn chỗ: lấp bằng seed ưu tiên (có chuyện xấu) xa nhất, rồi tới mọi seed xa nhất
    for pool in (bad, list(range(n))):
        while len(chosen) < count:
            left = [i for i in pool if i not in chosen]
            if not left:
                break
            chosen.append(farthest(left, chosen, feat))
    return chosen, bad, picks


def main():
    ap = argparse.ArgumentParser(description="Chấm dải seed lớn và chọn bộ seed để xem lại")
    ap.add_argument("--pool", default="0-199", help="dải seed để chấm, ví dụ 0-199")
    ap.add_argument("--count", type=int, default=30, help="số seed trong nhóm chính")
    ap.add_argument("--policy", default="sim_engine.policies.scripted_policy:ScriptedPolicy")
    ap.add_argument("--box-range", type=float, default=None)
    ap.add_argument("--bowl-range", type=float, default=None)
    ap.add_argument("--slow-factor", type=float, default=1.5, help="chậm = số bước >= hệ số này x số bước trung vị")
    ap.add_argument("--slow-min", type=int, default=100, help="và tối thiểu bằng số bước này")
    ap.add_argument("--repin", action="store_true", help="bỏ danh sách seed đã ghim, chọn lại từ đầu rồi ghim bộ mới")
    ap.add_argument("--workers", type=int, default=default_workers(), help="số tiến trình chạy song song (mặc định: số lõi - 1; 1 = tuần tự)")
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--out", default=os.path.join(here, "docs", "replay_seeds.json"))
    args = ap.parse_args()

    seeds = parse_seeds(args.pool)
    seeds = seeds + [k for k in sorted(config.BOX_FIXED_SEEDS) if k not in set(seeds)]      # seed cố định nằm ngoài dải chấm cũng được chấm
    t_start = time.time()
    rows = parallel_map(_eval_task, [(s, None) for s in seeds], args.policy, args.box_range, args.bowl_range, args.workers,
                        progress=lambda k, n: print(f"  đã chấm {k}/{n} seed...", flush=True) if k % 25 == 0 else None)
    print(f"  chấm xong trong {time.time() - t_start:.1f} s ({min(args.workers, len(seeds))} tiến trình)")
    probe = SO101PickPlaceEnv(render_images=False, box_range=args.box_range, bowl_range=args.bowl_range)
    box_r, bowl_r = probe.box_range, probe.bowl_range
    probe.close()

    steps = np.array([r["steps"] for r in rows])
    median = float(np.median(steps))
    slow_steps = max(args.slow_min, args.slow_factor * median)
    fails = [r for r in rows if failed(r)]
    n_skip = sum(1 for r in rows if r.get("skip"))
    to_count = collections.Counter(p for r in rows for p in r["timeouts"])
    slow = [r["seed"] for r in rows if r["steps"] >= slow_steps]
    kinds = collections.Counter(r["kind"] for r in rows)

    print(f"\n=== Chấm {len(rows)} seed (kẹp THẬT) ===")
    main_rows = [r for r in rows if r["seed"] < config.FIXED_SEED_FIRST_EXTRA]
    extra_rows = [r for r in rows if r["seed"] >= config.FIXED_SEED_FIRST_EXTRA]
    print(f"200 seed chính (0-{config.FIXED_SEED_FIRST_EXTRA - 1}): thành công {sum(1 for r in main_rows if r['success'] and not r.get('skip'))}/{sum(1 for r in main_rows if not r.get('skip'))}")
    print(f"seed cố định thêm ({len(extra_rows)} seed từ {config.FIXED_SEED_FIRST_EXTRA}): gắp được {sum(1 for r in extra_rows if r['success'])}, ngoài tầm bỏ qua {sum(1 for r in extra_rows if r.get('skip'))}, hỏng {sum(1 for r in extra_rows if failed(r))}")
    print(f"tổng: thành công {len(rows) - n_skip - len(fails)}/{len(rows) - n_skip} (không tính {n_skip} seed ngoài tầm: về home ngay) | bước trung vị {median:.0f} | chậm (>= {slow_steps:.0f} bước): {len(slow)} seed")
    print("kiểu kết quả:", dict(kinds))
    print("số seed có timeout theo phase:", dict(sorted(to_count.items())))

    feat = features(rows, box_r, bowl_r)
    chosen, bad, picks = select(rows, args.count, feat, slow_steps)
    # GHIM seed: bộ seed xem trong replay không bị đổi khi một seed hết hỏng (xanh), để còn xem lại quá trình sửa.
    # docs/replay_pins.json giữ danh sách; mỗi lần chấm: seed cố định + seed đã ghim + MỌI seed hiện đang hỏng chưa có trong danh sách.
    # Seed hỏng mới được thêm vào ghim; seed đã ghim không bao giờ bị bỏ (trừ khi chạy --repin).
    pins_path = os.path.join(os.path.dirname(args.out), "replay_pins.json")
    pinned = []
    if not args.repin and os.path.exists(pins_path):
        with open(pins_path, encoding="utf-8") as f:
            pinned = json.load(f).get("pinned", [])
    by_seed = {r["seed"]: i for i, r in enumerate(rows)}
    fixed_idx = [i for i, r in enumerate(rows) if r["seed"] in config.BOX_FIXED_SEEDS]
    if not pinned:                                                # lần đầu (hoặc --repin): ghim bộ vừa chọn
        pinned = [rows[i]["seed"] for i in chosen if rows[i]["seed"] not in config.BOX_FIXED_SEEDS]
    new_fail = [r["seed"] for r in rows if failed(r) and r["seed"] not in pinned and r["seed"] not in config.BOX_FIXED_SEEDS]
    pinned = pinned + new_fail
    with open(pins_path, "w", encoding="utf-8") as f:
        json.dump({"pinned": pinned, "note": "seed xem trong replay; thêm seed hỏng mới, không bỏ seed đã xanh. --repin để chọn lại"}, f)
    chosen = fixed_idx + [by_seed[sd] for sd in pinned if sd in by_seed]
    fail_idx = [i for i, r in enumerate(rows) if failed(r)]
    rest = [i for i in fail_idx if i not in chosen] if len(fail_idx) > args.count else []

    base_path = os.path.join(os.path.dirname(args.out), "replay_baseline.json")
    base_failed, base_label, base_seeds = None, "", set()
    if os.path.exists(base_path):
        with open(base_path, encoding="utf-8") as f:
            bj = json.load(f)
        base_failed, base_label = set(bj["failed"]), bj.get("label", "mốc trước")
        base_seeds = set(bj.get("seeds", []))                                 # seed đã được chấm ở mốc trước (seed mới thì chưa có để so)

    def tags(r):
        t = []
        t.append("NGOÀI TẦM: về home ngay" if r.get("skip") else ("THÀNH CÔNG" if r["success"] else "HỎNG: " + r["kind"]))
        if r["timeouts"]:
            t.append("timeout phase " + ",".join(map(str, r["timeouts"])))
        if r["retries"]:
            t.append(f"thử kẹp lại {r['retries']} lần")
        if r.get("roll_flips"):
            t.append(f"chạm bát, đổi roll {r['roll_flips']} lần")
        if r.get("bowl_avoid"):
            t.append("roll chọn sẵn tránh bát")
        if r["drops"]:
            t.append(f"làm rớt hộp {r['drops']} lần")
        if r["regrasps"]:
            t.append(f"gắp lại sau khi rút {r['regrasps']} lần")
        if r["steps"] >= slow_steps:
            t.append(f"chậm ({r['steps']} bước)")
        cm = r.get("cam_min")
        if cm is not None and cm < 3.0:
            t.append(f"CAMERA XUYÊN bàn/vai {cm:.1f} mm" if cm < 0 else f"camera sát bàn/vai ({cm:.1f} mm)")
        if base_failed is not None and not r.get("skip"):                     # so với mốc trước khi tối ưu
            if r["success"] and r["seed"] in base_failed:
                t.insert(0, "ĐÃ CẢI THIỆN: trước hỏng, nay xanh (" + base_label + ")")
            elif not r["success"] and r["seed"] not in base_failed:
                t.insert(0, ("MỚI HỎNG so với " + base_label) if r["seed"] in base_seeds else "seed mới (chưa có ở " + base_label + ")")
        return t

    def pack(i, group):
        r = rows[i]
        box_r, _, gap = reach(r)
        t = picks.get(i, []) + tags(r)
        t.append(f"hộp cách trục khớp 1 {box_r * 100:.1f} cm, cách bát {gap * 100:.1f} cm")
        fixed = fixed_seed_name(r["seed"])
        e = {"seed": r["seed"], "group": group if i not in picks else "biên tầm với (xa/gần trục, xa/gần bát)", "tags": t,
             "steps": r["steps"], "box0": [round(v, 4) for v in r["box0"]]}
        if fixed:
            e["name"] = f"seed {r['seed']} · {fixed}"
            e["group"] = f"{len(config.BOX_FIXED_SEEDS)} seed cố định (trường hợp cực biên)"
        return e

    selected = [pack(i, f"{args.count} seed chọn (đa dạng, ưu tiên hỏng/timeout/chậm)") for i in chosen]
    others = [pack(i, "seed hỏng còn lại") for i in sorted(rest, key=lambda j: rows[j]["seed"])]

    dmin = min((np.linalg.norm(feat[a] - feat[b]) for a in chosen for b in chosen if a < b), default=0)
    print(f"\nNhóm chính {len(chosen)} seed: hỏng {sum(1 for i in chosen if failed(rows[i]))}, "
          f"có timeout {sum(1 for i in chosen if rows[i]['timeouts'])}, chậm {sum(1 for i in chosen if rows[i]['steps'] >= slow_steps)}; "
          f"khoảng cách nhỏ nhất giữa hai seed (chuẩn hóa, 2.0 = bề rộng cả ô): {dmin:.2f}")
    print("seed nhóm chính:", [rows[i]["seed"] for i in chosen])
    if others:
        print(f"Số seed hỏng nhiều hơn {args.count}: đưa thêm {len(others)} seed hỏng còn lại lên replay.")

    data = {
        "pool": args.pool, "count": args.count, "slow_steps": slow_steps,
        "summary": {"success": len(rows) - n_skip - len(fails), "total": len(rows) - n_skip, "skipped_unreachable": n_skip, "median_steps": median, "kinds": dict(kinds),
                    "timeouts_by_phase": {str(k): v for k, v in sorted(to_count.items())}, "slow_seeds": slow},
        "selected": selected, "rest": others,
        "all": [{"seed": r["seed"], "skip": r.get("skip", False), "success": r["success"], "steps": r["steps"], "timeouts": r["timeouts"], "kind": r["kind"],
                 "retries": r["retries"], "drops": r["drops"], "regrasps": r["regrasps"], "box0": [round(v, 4) for v in r["box0"]]} for r in rows],
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    print(f"\nĐã ghi {args.out}. Chạy tiếp: python -m sim_engine.make_replay")


if __name__ == "__main__":
    main()
