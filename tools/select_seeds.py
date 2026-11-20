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

import numpy as np

from ..env import SO101PickPlaceEnv
from ..policies.scripted_policy import ScriptedPolicy
from .make_replay import Recorder, parse_seeds


# Hai trường hợp cực biên luôn đứng đầu danh sách replay: hộp XA nhất và GẦN nhất mà cánh tay phải xử lý.
# (đặt tay vị trí hộp qua env.reset(options=...), nên không phụ thuộc vùng random ±3 cm)
# Các trường hợp cực biên, đặt tay vị trí hộp qua env.reset(options=...) (không phụ thuộc vùng random ±3 cm).
# Vị trí lấy từ bản đồ vùng gắp được (quét theo cực quanh trục khớp 1, góc đo từ hướng tay): gắp ổn tới r ≈ 0.38 m
# (r = 0.40 trở lên hỏng), tới góc ±80° (±100° hỏng), và r ≥ 0.10 m. Bát cố định ở (-0.15, 0).
EXTREME_CASES = [
    {"name": "CASE 1: hộp XA nhất (tay vươn tối đa)", "seed": 0, "box_xy": (0.000, 0.230)},
    {"name": "CASE 2: hộp GẦN nhất (sát đế, lệch một bên)", "seed": 0, "box_xy": (0.060, -0.046)},
    {"name": "CASE 3: hộp XA bát nhất + tay vươn xa (chéo)", "seed": 0, "box_xy": (0.355, -0.087)},
    {"name": "CASE 4: hộp XA bát + tay vươn gần", "seed": 0, "box_xy": (0.118, -0.129)},
    {"name": "CASE 5: hộp SÁT bát + tay vươn xa", "seed": 0, "box_xy": (-0.260, 0.000)},
    {"name": "CASE 6: hộp SÁT bát + tay vươn gần", "seed": 0, "box_xy": (-0.118, -0.129)},
]
for _c in EXTREME_CASES:
    _c["options"] = {"box_xy": list(_c["box_xy"])}
SHOULDER_XY = (0.0, -0.15)      # trục khớp 1
MOVE_PHASES = (0, 1, 2, 3, 5, 6, 7, 9)   # các phase kết thúc khi tay tới đích (hoặc quá PHASE_TIMEOUT_STEPS bước)
PHASE_TIMEOUT = ScriptedPolicy.PHASE_TIMEOUT_STEPS


def evaluate_seed(rec, seed):
    """Chạy 1 seed, trả về dict thống kê dùng để chọn và để báo cáo."""
    with contextlib.redirect_stdout(io.StringIO()):
        ep = rec.run(seed)
    frames = ep["frames"]
    # timeout thật = phase kết thúc vì quá PHASE_TIMEOUT_STEPS bước (không phải vì tay đã tới nơi)
    timeouts = sorted({f["ph"] for f in frames if f["phn"] != f["ph"] and f["ph"] in MOVE_PHASES and f["ps"] > PHASE_TIMEOUT})
    retries = sum(1 for f in frames if f["ph"] == 4 and f["phn"] == 2)                 # phase 4 không kẹp được, hạ lại phase 2
    drops = sum(1 for f in frames if f["ph"] in (5, 6, 7) and f["phn"] == 1)           # làm rớt hộp khi đang mang, quay lại gắp
    regrasps = sum(1 for f in frames if f["ph"] == 9 and f["phn"] == 1)                # rút lên rồi thấy hộp ngoài bát, gắp lại
    z0 = frames[0]["box"][2]
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
    return {
        "seed": seed, "success": bool(ep["success"]), "steps": ep["steps"], "timeouts": timeouts, "retries": retries,
        "drops": drops, "regrasps": regrasps,
        "kind": kind, "box0": frames[0]["box"][:2], "bowl": ep["bowl"][:2],
    }


def features(rows, box_range, bowl_range, box_center=(0.15, 0.0)):
    """Vị trí chuẩn hóa về [-1, 1] để đo "xa nhau" công bằng giữa hộp và bát."""
    out = []
    for r in rows:
        f = [(r["box0"][0] - box_center[0]) / box_range, (r["box0"][1] - box_center[1]) / box_range]
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

    bad = [i for i in range(n) if (not rows[i]["success"]) or rows[i]["timeouts"] or rows[i]["steps"] >= slow_steps
           or rows[i]["retries"] or rows[i]["drops"] or rows[i]["regrasps"]]
    chosen = []
    # 0) các biên của tầm với (hộp xa/gần trục, xa/gần bát và 4 tổ hợp)
    picks = reach_picks(rows)
    for i in picks:
        if i not in chosen:
            chosen.append(i)
    # 1) MỌI seed hỏng (nếu có chỗ), rồi mỗi chữ ký lỗi (timeout / kẹp lại / làm rớt / gắp lại / chậm) có một đại diện
    for i in [i for i in range(n) if not rows[i]["success"]]:
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
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--out", default=os.path.join(here, "docs", "replay_seeds.json"))
    args = ap.parse_args()

    module_name, class_name = args.policy.split(":")
    policy_cls = getattr(importlib.import_module(module_name), class_name)
    env = SO101PickPlaceEnv(render_images=False, end_on_success=False,
                            box_range=args.box_range, bowl_range=args.bowl_range)
    rec = Recorder(env, policy_cls(env))

    seeds = parse_seeds(args.pool)
    rows = []
    for k, s in enumerate(seeds):
        rows.append(evaluate_seed(rec, s))
        if (k + 1) % 25 == 0:
            print(f"  đã chấm {k + 1}/{len(seeds)} seed...", flush=True)
    box_r, bowl_r = env.box_range, env.bowl_range
    env.close()

    steps = np.array([r["steps"] for r in rows])
    median = float(np.median(steps))
    slow_steps = max(args.slow_min, args.slow_factor * median)
    fails = [r for r in rows if not r["success"]]
    to_count = collections.Counter(p for r in rows for p in r["timeouts"])
    slow = [r["seed"] for r in rows if r["steps"] >= slow_steps]
    kinds = collections.Counter(r["kind"] for r in rows)

    print(f"\n=== Chấm {len(rows)} seed (kẹp THẬT) ===")
    print(f"thành công {len(rows) - len(fails)}/{len(rows)} | bước trung vị {median:.0f} | chậm (>= {slow_steps:.0f} bước): {len(slow)} seed")
    print("kiểu kết quả:", dict(kinds))
    print("số seed có timeout theo phase:", dict(sorted(to_count.items())))

    feat = features(rows, box_r, bowl_r)
    chosen, bad, picks = select(rows, args.count, feat, slow_steps)
    fail_idx = [i for i, r in enumerate(rows) if not r["success"]]
    rest = [i for i in fail_idx if i not in chosen] if len(fail_idx) > args.count else []

    def tags(r):
        t = []
        t.append("THÀNH CÔNG" if r["success"] else "HỎNG: " + r["kind"])
        if r["timeouts"]:
            t.append("timeout phase " + ",".join(map(str, r["timeouts"])))
        if r["retries"]:
            t.append(f"thử kẹp lại {r['retries']} lần")
        if r["drops"]:
            t.append(f"làm rớt hộp {r['drops']} lần")
        if r["regrasps"]:
            t.append(f"gắp lại sau khi rút {r['regrasps']} lần")
        if r["steps"] >= slow_steps:
            t.append(f"chậm ({r['steps']} bước)")
        return t

    def pack(i, group):
        r = rows[i]
        box_r, _, gap = reach(r)
        t = picks.get(i, []) + tags(r)
        t.append(f"hộp cách trục khớp 1 {box_r * 100:.1f} cm, cách bát {gap * 100:.1f} cm")
        return {"seed": r["seed"], "group": group if i not in picks else "biên tầm với (xa/gần trục, xa/gần bát)", "tags": t,
                "steps": r["steps"], "box0": [round(v, 4) for v in r["box0"]]}

    extremes = []
    for c in EXTREME_CASES:
        r = math.hypot(c["box_xy"][0] - SHOULDER_XY[0], c["box_xy"][1] - SHOULDER_XY[1])
        gap = math.hypot(c["box_xy"][0] + 0.15, c["box_xy"][1])
        extremes.append({"seed": c["seed"], "name": c["name"], "options": c["options"], "group": f"{len(EXTREME_CASES)} trường hợp cực biên (xa/gần trục, xa/sát bát)",
                         "tags": [f"hộp ({c['box_xy'][0]:.3f}, {c['box_xy'][1]:.3f}), cách trục khớp 1 {r * 100:.1f} cm, cách bát {gap * 100:.1f} cm"], "steps": 0,
                         "box0": list(c["box_xy"])})
    selected = extremes + [pack(i, f"{args.count} seed chọn (đa dạng, ưu tiên hỏng/timeout/chậm)") for i in chosen]
    others = [pack(i, "seed hỏng còn lại") for i in sorted(rest, key=lambda j: rows[j]["seed"])]

    dmin = min((np.linalg.norm(feat[a] - feat[b]) for a in chosen for b in chosen if a < b), default=0)
    print(f"\nNhóm chính {len(chosen)} seed: hỏng {sum(1 for i in chosen if not rows[i]['success'])}, "
          f"có timeout {sum(1 for i in chosen if rows[i]['timeouts'])}, chậm {sum(1 for i in chosen if rows[i]['steps'] >= slow_steps)}; "
          f"khoảng cách nhỏ nhất giữa hai seed (chuẩn hóa, 2.0 = bề rộng cả ô): {dmin:.2f}")
    print("seed nhóm chính:", [rows[i]["seed"] for i in chosen])
    if others:
        print(f"Số seed hỏng nhiều hơn {args.count}: đưa thêm {len(others)} seed hỏng còn lại lên replay.")

    data = {
        "pool": args.pool, "count": args.count, "slow_steps": slow_steps,
        "summary": {"success": len(rows) - len(fails), "total": len(rows), "median_steps": median, "kinds": dict(kinds),
                    "timeouts_by_phase": {str(k): v for k, v in sorted(to_count.items())}, "slow_seeds": slow},
        "selected": selected, "rest": others,
        "all": [{"seed": r["seed"], "success": r["success"], "steps": r["steps"], "timeouts": r["timeouts"], "kind": r["kind"],
                 "retries": r["retries"], "drops": r["drops"], "regrasps": r["regrasps"], "box0": [round(v, 4) for v in r["box0"]]} for r in rows],
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    print(f"\nĐã ghi {args.out}. Chạy tiếp: python -m sim_engine.make_replay")


if __name__ == "__main__":
    main()
