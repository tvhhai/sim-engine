"""Tạo mesh cho giá đỡ và module camera cổ tay của SO-101 (model gốc chỉ cho vị trí camera, không có hình giá đỡ).

Ảnh robot thật: camera USB nhỏ (bo mạch vuông ~3.2 cm, ống kính tròn) gắn trên một trụ dẹt nhô lên từ thân bộ kẹp, nghiêng nhìn
xuống phía mũi hàm. Ở đây dựng lại bằng các khối bo tròn góc (không còn các tấm phẳng góc cạnh): trụ dẹt thon dần, khay đỡ bo mạch
bo góc, bo mạch bo góc, khối đế ống kính, ống kính tròn có viền. Vị trí và hướng nhìn lấy từ camera `arm_cam` trong XML nên
hình vẽ và ảnh camera khớp nhau.

Cách dùng (đứng ở thư mục chứa sim_engine):
    python -m sim_engine.tools.make_camera_mount            # ghi assets/so101/wrist_cam_bracket.stl và wrist_cam_module.stl

Toạ độ ghi trong khung body "gripper" (cùng khung với camera arm_cam trong XML). Hai file:
    wrist_cam_bracket.stl: trụ dẹt + khay đỡ bo mạch (nhựa in 3D, cùng màu thân robot)
    wrist_cam_module.stl : bo mạch + đế ống kính + ống kính + đầu nối (màu tối)
"""

import os

import numpy as np

from .split_jaw_mesh import write_stl

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(HERE, "assets", "so101")

# Tư thế camera arm_cam trong khung body gripper (khớp với XML): vị trí, và các trục của camera (MuJoCo: nhìn theo -z)
CAM_POS = np.array([-0.00296, 0.05436, -0.05310])
CAM_QUAT = np.array([0.954498, -0.284308, -0.085070, 0.029438])
POST_BASE = np.array([-0.003, 0.019, -0.030])        # chân trụ: chìm trong mặt trên thân bộ kẹp (y ≈ 0.024)


def quat_to_mat(q):
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def rounded_rect(w, h, r, seg=5):
    """Chu vi hình chữ nhật w x h bo góc bán kính r: mảng (4*(seg+1), 2), ngược chiều kim đồng hồ."""
    r = min(r, w / 2 - 1e-6, h / 2 - 1e-6)
    pts = []
    for k, (cx, cy) in enumerate([(w / 2 - r, h / 2 - r), (-w / 2 + r, h / 2 - r), (-w / 2 + r, -h / 2 + r), (w / 2 - r, -h / 2 + r)]):
        for a in np.linspace(k * np.pi / 2, (k + 1) * np.pi / 2, seg + 1):
            pts.append((cx + r * np.cos(a), cy + r * np.sin(a)))
    return np.array(pts)


def loft(c0, c1, axes, size0, size1, radius, seg=5):
    """Khối nối hai mặt cắt chữ nhật bo góc: tâm c0 -> c1, `axes` = cột (x, y, z) của khung mặt cắt (z là hướng từ c0 sang c1).

    size0, size1 = (rộng, dày) ở hai đầu, nên có thể thon dần. Có nắp hai đầu. Trả (n, 3, 3) tam giác.
    """
    def ring(c, size):
        p = rounded_rect(size[0], size[1], radius, seg)
        return c + p[:, :1] * axes[:, 0] + p[:, 1:] * axes[:, 1]

    a, b = ring(c0, size0), ring(c1, size1)
    n = len(a)
    tris = []
    for k in range(n):
        j = (k + 1) % n
        tris += [[a[k], a[j], b[j]], [a[k], b[j], b[k]], [c0, a[j], a[k]], [c1, b[k], b[j]]]
    return np.array(tris)


def cylinder(center, axis, radius, length, n=32):
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


def slab(center, axes, w, h, thickness, radius):
    """Tấm bo góc w x h dày `thickness`, mặt phẳng theo (x, y) của axes, tâm `center`, dày theo z của axes."""
    z = axes[:, 2]
    return loft(center - z * thickness / 2, center + z * thickness / 2, axes, (w, h), (w, h), radius)


def build():
    R = quat_to_mat(CAM_QUAT)
    right, up, back = R[:, 0], R[:, 1], R[:, 2]          # camera nhìn theo -back
    look = -back
    axes = np.column_stack([right, up, look])            # khung camera: x = phải, y = lên, z = hướng nhìn

    # --- Giá đỡ ---
    tray_c = CAM_POS - 0.0172 * look
    tray = slab(tray_c, axes, 0.036, 0.036, 0.0030, 0.006)                  # khay đỡ bo mạch
    lip = slab(tray_c + 0.0022 * look, axes, 0.0345, 0.0345, 0.0016, 0.0055)  # gờ nhỏ quanh bo mạch
    post_top = tray_c - 0.0010 * look
    d = post_top - POST_BASE
    length = np.linalg.norm(d)
    pz = d / length
    px = np.cross([0.0, 0.0, 1.0], pz)
    px /= np.linalg.norm(px)
    py = np.cross(pz, px)
    post = loft(POST_BASE, post_top, np.column_stack([px, py, pz]), (0.022, 0.0105), (0.016, 0.0075), 0.0032)   # trụ dẹt thon dần
    bracket = np.concatenate([tray, lip, post])

    # --- Module camera ---
    pcb = slab(CAM_POS - 0.0118 * look, axes, 0.032, 0.032, 0.0016, 0.0035)                   # bo mạch bo góc
    holder = slab(CAM_POS - 0.0082 * look, axes, 0.0150, 0.0150, 0.0056, 0.0020)              # đế ống kính
    barrel = cylinder(CAM_POS - 0.0042 * look, look, 0.0055, 0.0084)                          # thân ống kính
    ring_ = cylinder(CAM_POS - 0.0010 * look, look, 0.0042, 0.0020)                           # viền kính ở đầu
    conn = slab(CAM_POS - 0.0138 * look - up * 0.0150, axes, 0.0110, 0.0040, 0.0022, 0.0008)   # đầu nối cáp ở cạnh dưới bo mạch
    module = np.concatenate([pcb, holder, barrel, ring_, conn])
    return bracket, module


def main():
    bracket, module = build()
    write_stl(os.path.join(OUT, "wrist_cam_bracket.stl"), bracket)
    write_stl(os.path.join(OUT, "wrist_cam_module.stl"), module)
    print(f"ghi {len(bracket)} + {len(module)} tam giac vao {OUT}")


if __name__ == "__main__":
    main()
