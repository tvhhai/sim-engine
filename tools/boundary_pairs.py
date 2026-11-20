"""Phân tích BIÊN của vùng gắp được: các vòng bán kính mà gắp được và không gắp được XEN KẼ, và các CẶP ĐỐI XỨNG (r, +góc) / (r, -góc)
cho kết quả khác nhau (một bên gắp được, bên đối xứng thì không).

Dữ liệu vào: các file quét dày ở biên (tools/scan_pick_region.py --r ... --angle ... --out ...), mặc định docs/pick_scan_outer.json (vòng ngoài)
và docs/pick_scan_inner.json (vòng trong sát đế), cộng docs/pick_scan.json (lưới thường). Kết quả ghi docs/boundary_pairs.json và in bảng.

Cách dùng (đứng ở thư mục chứa sim_engine, đã kích hoạt .venv):
    python -m sim_engine.tools.boundary_pairs                  # chỉ phân tích và in
    python -m sim_engine.tools.boundary_pairs --update-seeds   # thêm mỗi thành viên của cặp bất đối xứng thành seed cố định (200+)
Quy ước góc: đo từ hướng tay (+y) về phía +x quanh trục khớp 1; góc dương = bên +x, góc âm = bên -x (phía bát).
"""

import argparse
import collections
import json
import os
import re

from .. import config

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS = os.path.join(HERE, "docs")


def load():
    pts = {}
    for name in ("pick_scan.json", "pick_scan_outer.json", "pick_scan_inner.json"):
        path = os.path.join(DOCS, name)
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                for p in json.load(f)["points"]:
                    pts[(round(p["r"], 3), p["a"])] = p            # điểm quét sau ghi đè điểm trước (cùng r, góc)
    return pts


def analyse(pts):
    rings = sorted({r for r, _ in pts})
    ring_stats = []
    for r in rings:
        row = [p for (rr, a), p in pts.items() if rr == r and abs(a) <= config.PAN_LIMIT_DEG]
        ok = sum(p["ok"] for p in row)
        ring_stats.append({"r": r, "ok": ok, "fail": len(row) - ok, "total": len(row), "mixed": 0 < ok < len(row)})
    pairs = []
    for (r, a), p in sorted(pts.items()):
        if a <= 0 or abs(a) > config.PAN_LIMIT_DEG:
            continue
        q = pts.get((r, -a))
        if q is None:
            continue
        pairs.append({"r": r, "a": a, "plus": p, "minus": q, "differ": p["ok"] != q["ok"]})
    return ring_stats, pairs


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--update-seeds", action="store_true", help="thêm thành viên các cặp bất đối xứng vào khối SCAN-SEEDS của config.py (số seed từ 200)")
    ap.add_argument("--rings", type=float, nargs="*", default=[0.38, 0.40, 0.42], help="chỉ thêm seed cho các vòng bán kính này (m); mặc định 3 mốc ngoài vùng gắp: 0.38 0.40 0.42")
    args = ap.parse_args()
    pts = load()
    ring_stats, pairs = analyse(pts)
    R = config.PICK_REGION
    print("VÒNG BÁN KÍNH (chỉ góc trong giới hạn khớp 1 ±%d°):" % config.PAN_LIMIT_DEG)
    print("   r(cm)  gắp được / hỏng   xen kẽ?   (trong vùng gắp %.0f-%.0f cm)" % (R["r_min"] * 100, R["r_max"] * 100))
    for s in ring_stats:
        tag = "XEN KẼ" if s["mixed"] else ("hỏng hết" if s["ok"] == 0 else "gắp hết")
        inside = "trong vùng" if R["r_min"] <= s["r"] <= R["r_max"] else ""
        print(f"   {s['r'] * 100:5.0f}    {s['ok']:3d} / {s['fail']:3d}       {tag:8s} {inside}")
    diff = [p for p in pairs if p["differ"]]
    print(f"\nCẶP ĐỐI XỨNG (r, +góc) và (r, -góc): {len(pairs)} cặp, {len(diff)} cặp KHÁC KẾT QUẢ:")
    by_r = collections.defaultdict(list)
    for p in diff:
        by_r[p["r"]].append(p)
    for r in sorted(by_r):
        row = ", ".join(f"±{p['a']}° ({'+ gắp được, - hỏng' if p['plus']['ok'] else '- gắp được, + hỏng'})" for p in by_r[r])
        print(f"   r = {r * 100:4.0f} cm: {row}")
    out = {"rings": ring_stats, "asymmetric_pairs": [{"r": p["r"], "a": p["a"], "plus_ok": p["plus"]["ok"], "minus_ok": p["minus"]["ok"],
                                                     "plus_xy": p["plus"]["xy"], "minus_xy": p["minus"]["xy"]} for p in diff]}
    with open(os.path.join(DOCS, "boundary_pairs.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)
    if args.update_seeds:
        update_seeds([p for p in diff if any(abs(p['r'] - r) < 1e-6 for r in args.rings)])


def update_seeds(diff):
    """Thêm hai thành viên của mỗi cặp bất đối xứng làm seed cố định mới (giữ nguyên các seed đã có)."""
    path = os.path.join(HERE, "config.py")
    with open(path, encoding="utf-8") as f:
        text = f.read()
    a0, a1 = text.index("    # >>> SCAN-SEEDS"), text.index("    # <<< SCAN-SEEDS")
    nl = chr(10)
    pat = re.compile(r'^\s+(\d+): \("(.*)", \((-?[\d.]+), (-?[\d.]+)\)\),\s*$')
    kept = []
    for ln in text[a0:a1].splitlines():
        m = pat.match(ln)
        if m:
            kept.append((int(m.group(1)), m.group(2), float(m.group(3)), float(m.group(4))))
    nxt = max(max((k[0] for k in kept), default=6), config.FIXED_SEED_FIRST_EXTRA - 1) + 1
    added = 0
    for p in diff:
        for sign, q in ((+1, p["plus"]), (-1, p["minus"])):
            x, y = q["xy"]
            if any(abs(x - kx) < 1e-3 and abs(y - ky) < 1e-3 for _, _, kx, ky in kept):
                continue
            kept.append((nxt, f'cặp đối xứng r = {p["r"] * 100:.0f} cm, góc {sign * p["a"]:+d}° ({"gắp được" if q["ok"] else "hỏng"}; đối xứng {"hỏng" if q["ok"] else "gắp được"})', x, y))
            nxt += 1
            added += 1
    head = text[a0:text.index(nl, a0) + 1]
    out = [f'    {sd}: ("{name}", ({x:.4f}, {y:.4f})),' for sd, name, x, y in sorted(kept)]
    with open(path, "w", encoding="utf-8") as f:
        f.write(text[:a0] + head + nl.join(out) + nl + text[a1:])
    print(f"đã thêm {added} seed cặp đối xứng vào config.py (tổng seed cố định đến {nxt - 1})")


if __name__ == "__main__":
    main()
