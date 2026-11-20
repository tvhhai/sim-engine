"""Tạo mesh cho giá đỡ và module camera cổ tay của SO-101 (model gốc chỉ cho vị trí camera, không có hình giá đỡ).

Ảnh robot thật: camera USB nhỏ (bo mạch vuông ~3.2 cm, ống kính tròn) gắn trên một trụ nhô lên từ thân bộ kẹp, nghiêng nhìn
xuống phía mũi hàm. Ở đây dựng lại bằng hình hộp và hình trụ, theo đúng tư thế camera `arm_cam` đã có trong model:
vị trí và hướng nhìn lấy từ XML, nên hình vẽ và ảnh camera khớp nhau.

Cách dùng (đứng ở thư mục chứa sim_engine):
    python -m sim_engine.tools.make_camera_mount            # ghi assets/so101/wrist_cam_bracket.stl và wrist_cam_module.stl

Toạ độ ghi trong khung body "gripper" (cùng khung với camera arm_cam trong XML). Hai file:
    wrist_cam_bracket.stl: trụ nghiêng + tấm giữ bo mạch (nhựa in 3D, cùng màu thân robot)
    wrist_cam_module.stl : bo mạch + khối ống kính + ống kính (màu tối)
"""

import os

import numpy as np

from .split_jaw_mesh import write_stl

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(HERE, "assets", "so101")

# Tư thế camera arm_cam trong khung body gripper (khớp với XML): vị trí, và các trục của camera (MuJoCo: nhìn theo -z)
CAM_POS = np.array([-0.00296, 0.05436, -0.05310])
CAM_QUAT = np.array([0.954498, -0.284308, -0.085070, 0.029438])
POST_BASE = np.array([-0.003, 0.021, -0.030])        # chân trụ: nằm chìm trong mặt trên thân bộ kẹp (y ≈ 0.024)


def quat_to_mat(q):
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def box(center, axes, half):
    """Hình hộp có hướng: `axes` là ma trận 3x3 (cột = trục), `half` = nửa kích thước theo 3 trục. Trả (12, 3, 3) tam giác."""
    corners = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], float) * half
    v = center + corners @ axes.T
    quads = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
    tris = []
    for a, b, c, d in quads:
        tris += [[v[a], v[b], v[c]], [v[a], v[c], v[d]]]
    return np.array(tris)


def cylinder(center, axis, radius, length, n=28):
    """Hình trụ tâm `center`, dọc theo `axis`, có nắp hai đầu. Trả (4n, 3, 3) tam giác."""
    axis = axis / np.linalg.norm(axis)
    t = np.cross(axis, [1.0, 0, 0] if abs(axis[0]) < 0.9 else [0, 1.0, 0])
    t /= np.linalg.norm(t)
    b = np.cross(axis, t)
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
    ring = [(np.cos(a) * t + np.sin(a) * b) * radius for a in ang]
    top, bot = center + axis * length / 2, center - axis * length / 2
    tris = []
    for k in range(n):
        p, q = ring[k], ring[(k + 1) % n]
        tris += [[bot + p, top + p, top + q], [bot + p, top + q, bot + q], [top, top + p, top + q], [bot, bot + q, bot + p]]
    return np.array(tris)


def build():
    R = quat_to_mat(CAM_QUAT)
    right, up, back = R[:, 0], R[:, 1], R[:, 2]          # camera nhìn theo -back
    look = -back
    axes = np.column_stack([right, up, look])            # hộp: x = phải, y = lên, z = hướng nhìn

    # Giá đỡ: tấm giữ bo mạch phía sau + trụ nghiêng nối xuống thân bộ kẹp
    holder_c = CAM_POS - 0.0165 * look
    holder = box(holder_c, axes, np.array([0.0185, 0.0185, 0.0020]))
    post_top = holder_c - 0.0025 * look
    d = post_top - POST_BASE
    length = np.linalg.norm(d)
    pz = d / length
    px = np.cross([0.0, 0.0, 1.0], pz)
    px /= np.linalg.norm(px)
    py = np.cross(pz, px)
    post = box((POST_BASE + post_top) / 2, np.column_stack([px, py, pz]), np.array([0.0085, 0.0045, length / 2]))
    bracket = np.concatenate([holder, post])

    # Module camera: bo mạch vuông, khối đế ống kính, ống kính tròn (đầu ống kính chạm đúng vị trí camera trong XML)
    pcb = box(CAM_POS - 0.0105 * look, axes, np.array([0.0160, 0.0160, 0.0008]))
    base = box(CAM_POS - 0.0070 * look, axes, np.array([0.0072, 0.0072, 0.0040]))
    barrel = cylinder(CAM_POS - 0.0040 * look, look, 0.0050, 0.0080)
    module = np.concatenate([pcb, base, barrel])
    return bracket, module


def main():
    bracket, module = build()
    write_stl(os.path.join(OUT, "wrist_cam_bracket.stl"), bracket)
    write_stl(os.path.join(OUT, "wrist_cam_module.stl"), module)
    print(f"ghi {len(bracket)} + {len(module)} tam giac vao {OUT}")


if __name__ == "__main__":
    main()
