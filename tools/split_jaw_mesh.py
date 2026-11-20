"""Chia mesh hàm kẹp thành nhiều khối nhỏ để va chạm (bao lồi) bám sát hình thật hơn.

Vì sao: MuJoCo dùng BAO LỒI của cả mesh để tính va chạm. Hàm kẹp có chỗ lõm, bao lồi lấp kín chỗ lõm đó nên hộp
bị "chạm" sớm vài mm so với bề mặt nhìn thấy (đo thấy 2.6-6.5 mm, tối đa 11.6 mm). Chia mesh thành các cụm tam giác,
mỗi cụm có bao lồi riêng, thì tổng các bao lồi bám sát hơn nhiều.

Cách dùng (đứng ở thư mục chứa sim_engine):
    python -m sim_engine.tools.split_jaw_mesh                 # ghi các file STL vào assets/so101/split/ và in dòng XML
    python -m sim_engine.tools.split_jaw_mesh --parts 12 8    # số khối của hàm cố định, hàm di động

Cách chia: chia đôi liên tiếp cụm có kích thước lớn nhất theo trục dài nhất của nó, cắt tại trung vị tâm tam giác,
tới khi đủ số khối. Chỉ cần numpy.
"""

import argparse
import os

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MESH_DIR = os.path.join(HERE, "assets", "so101")
OUT_DIR = os.path.join(MESH_DIR, "split")
STL = np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")])


def read_stl(path):
    with open(path, "rb") as f:
        f.read(80)
        n = int(np.frombuffer(f.read(4), "<u4")[0])
        return np.frombuffer(f.read(n * 50), STL, count=n)


def write_stl(path, tris):
    out = np.zeros(len(tris), STL)
    out["v"] = tris
    e1, e2 = tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0]
    nrm = np.cross(e1, e2)
    out["n"] = nrm / np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-12)
    with open(path, "wb") as f:
        f.write(b"split jaw part".ljust(80, b" "))
        f.write(np.uint32(len(tris)).tobytes())
        f.write(out.tobytes())


def split(tris, parts):
    """Chia `tris` (n, 3, 3) thành `parts` cụm bằng cách chia đôi dần cụm có kích thước lớn nhất."""
    groups = [np.arange(len(tris))]
    cent = tris.mean(axis=1)
    while len(groups) < parts:
        ext = [np.ptp(tris[g].reshape(-1, 3), axis=0).max() for g in groups]
        k = int(np.argmax(ext))
        g = groups.pop(k)
        axis = int(np.argmax(np.ptp(tris[g].reshape(-1, 3), axis=0)))
        cut = np.median(cent[g, axis])
        left, right = g[cent[g, axis] <= cut], g[cent[g, axis] > cut]
        if len(left) < 4 or len(right) < 4:               # không chia được nữa
            groups.append(g)
            break
        groups += [left, right]
    return groups


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--parts", type=int, nargs=2, default=[12, 8], metavar=("FIXED", "MOVING"))
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    for name, stl, parts in (("fixed_jaw", "wrist_roll_follower_so101_v1.stl", args.parts[0]),
                             ("moving_jaw", "moving_jaw_so101_v1.stl", args.parts[1])):
        tris = read_stl(os.path.join(MESH_DIR, stl))["v"].astype(np.float64)
        groups = split(tris, parts)
        print(f"{name}: {len(tris)} tam giac -> {len(groups)} khoi")
        for k, g in enumerate(groups):
            write_stl(os.path.join(OUT_DIR, f"{name}_part{k:02d}.stl"), tris[g])
            print(f'    <mesh name="{name}_part{k:02d}" file="split/{name}_part{k:02d}.stl"/>')


if __name__ == "__main__":
    main()
