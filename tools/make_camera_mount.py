"""Tạo mesh cho giá đỡ camera cổ tay của SO-101 có KHỚP GẬP (model gốc chỉ cho vị trí camera, không có hình giá đỡ).

Cấu tạo (theo cách hay làm trên robot thật): một trụ dẹt gắn trên thân bộ kẹp, đầu trụ có hai khớp chữ C ôm lấy chốt;
khớp giữa của tay đỡ camera nằm giữa hai khớp C, nên tay đỡ gập quanh chốt (trục song song trục x của khung gripper).
Gập ít hay nhiều thì camera nhìn xa hay gần mũi hàm.

Hai phần, ghi ra hai nhóm file STL:
    wrist_cam_post.stl   phần CỐ ĐỊNH: trụ dẹt + hai khớp C + chốt, toạ độ trong khung body "gripper"
    wrist_cam_link.stl   phần GẬP được: khớp giữa + tay đỡ + khay bo mạch, toạ độ trong khung body "wrist_cam_arm"
    wrist_cam_module.stl bo mạch + ống kính, cũng trong khung "wrist_cam_arm" (màu tối)
Body "wrist_cam_arm" nằm ở chốt H, xoay quanh trục x một góc "tilt" (độ gập); camera arm_cam là con của body này.

Cách dùng (đứng ở thư mục chứa sim_engine):
    python -m sim_engine.tools.make_camera_mount        # ghi lại 3 file STL (độ gập không ảnh hưởng mesh, chỉ đổi quat trong XML)
Độ gập: đổi `config.WRIST_CAM_TILT_DEG` (env áp vào body lúc tạo môi trường), hoặc kéo thanh "Gập cam" trong replay để xem thử.
"""

import os

import numpy as np

from .split_jaw_mesh import write_stl

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(HERE, "assets", "so101")

# Tư thế camera ở độ gập 0 (khớp với tư thế cộng đồng SO-101 MuJoCo), trong khung body gripper
CAM_POS = np.array([-0.00296, 0.05436, -0.05310])
CAM_QUAT = np.array([0.954498, -0.284308, -0.085070, 0.029438])
POST_BASE = np.array([-0.003, 0.019, -0.030])        # chân trụ: chìm trong mặt trên thân bộ kẹp (y ≈ 0.024)
HINGE = np.array([-0.003, 0.046, -0.034])            # tâm chốt gập H (trục song song x)
X = np.array([1.0, 0.0, 0.0])


def quat_to_mat(q):
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def outward(tris, mid):
    """Đổi chiều các tam giác sao cho pháp tuyến hướng RA NGOÀI vật liệu. `mid(P)` trả về điểm nằm trong vật liệu gần tam giác có
    tâm P. MuJoCo/OpenGL tô theo pháp tuyến nên mặt bị ngược sẽ tối hoặc lủng; hình dựng bằng tay phải đảm bảo chiều đúng."""
    tris = np.array(tris, float)
    n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    cen = tris.mean(axis=1)
    inner = np.array([mid(P) for P in cen])
    bad = ((cen - inner) * n).sum(axis=1) < 0
    tris[bad] = tris[bad][:, [0, 2, 1]]
    return tris


def convex(tris):
    """Vật lồi (hộp bo góc, trụ, tấm): mặt nào hướng vào tâm thì đảo lại."""
    c = np.array(tris, float).reshape(-1, 3).mean(axis=0)
    return outward(tris, lambda P: c)


def rounded_rect(w, h, r, seg=8):
    r = min(r, w / 2 - 1e-6, h / 2 - 1e-6)
    pts = []
    for k, (cx, cy) in enumerate([(w / 2 - r, h / 2 - r), (-w / 2 + r, h / 2 - r), (-w / 2 + r, -h / 2 + r), (w / 2 - r, -h / 2 + r)]):
        for a in np.linspace(k * np.pi / 2, (k + 1) * np.pi / 2, seg + 1):
            pts.append((cx + r * np.cos(a), cy + r * np.sin(a)))
    return np.array(pts)


def loft(c0, c1, axes, size0, size1, radius, seg=8):
    """Khối nối hai mặt cắt chữ nhật bo góc (thon dần), có nắp hai đầu."""
    def ring(c, size):
        p = rounded_rect(size[0], size[1], radius, seg)
        return c + p[:, :1] * axes[:, 0] + p[:, 1:] * axes[:, 1]

    a, b = ring(c0, size0), ring(c1, size1)
    n = len(a)
    tris = []
    for k in range(n):
        j = (k + 1) % n
        tris += [[a[k], a[j], b[j]], [a[k], b[j], b[k]], [c0, a[j], a[k]], [c1, b[k], b[j]]]
    return convex(tris)


def frame_from_z(z):
    """Khung (x, y, z) có trục z cho trước."""
    z = z / np.linalg.norm(z)
    x = np.cross([0.0, 0.0, 1.0], z) if abs(z[2]) < 0.95 else np.cross([0.0, 1.0, 0.0], z)
    x /= np.linalg.norm(x)
    return np.column_stack([x, np.cross(z, x), z])


def cylinder(center, axis, radius, length, n=32):
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
    return convex(tris)


def c_ring(center, axis, opening, r_out, r_in, length, gap_deg=110.0, n=22):
    """Khớp chữ C: ống rỗng (bán kính trong r_in, ngoài r_out, dài `length` dọc `axis`) bị cắt một khe rộng gap_deg
    hướng về `opening` (vuông góc trục). gap_deg = 0 cho ống kín (khớp giữa)."""
    axis = axis / np.linalg.norm(axis)
    o = opening - axis * np.dot(opening, axis)
    o /= np.linalg.norm(o)
    p = np.cross(axis, o)
    full = gap_deg <= 0
    a0, a1 = (0.0, 2 * np.pi) if full else (np.radians(gap_deg / 2), 2 * np.pi - np.radians(gap_deg / 2))
    angs = np.linspace(a0, a1, n + 1)
    half = axis * length / 2

    def pt(r, a, side):
        return center + r * (np.cos(a) * o + np.sin(a) * p) + side * half

    tris = []
    for k in range(n):
        A, B = angs[k], angs[k + 1]
        for r, flip in ((r_out, False), (r_in, True)):                       # mặt ngoài và mặt trong
            q = [pt(r, A, -1), pt(r, B, -1), pt(r, B, 1), pt(r, A, 1)]
            tris += ([[q[0], q[2], q[1]], [q[0], q[3], q[2]]] if flip else [[q[0], q[1], q[2]], [q[0], q[2], q[3]]])
        for side in (-1, 1):                                                  # hai nắp phẳng (vành khuyên)
            q = [pt(r_out, A, side), pt(r_out, B, side), pt(r_in, B, side), pt(r_in, A, side)]
            tris += [[q[0], q[1], q[2]], [q[0], q[2], q[3]]]
    if not full:                                                              # hai mặt cắt của khe
        for a in (a0, a1):
            q = [pt(r_in, a, -1), pt(r_out, a, -1), pt(r_out, a, 1), pt(r_in, a, 1)]
            tris += [[q[0], q[1], q[2]], [q[0], q[2], q[3]]]

    rmid, amin, amax = (r_out + r_in) / 2, angs[0], angs[-1]

    def mid(P):                      # điểm trong vật liệu gần P: cùng góc (kẹp cách mép khe một chút) và cùng độ cao dọc trục
        v = P - center
        s = np.clip(np.dot(v, axis), -length / 2, length / 2)
        a = np.arctan2(np.dot(v, p), np.dot(v, o)) % (2 * np.pi)
        if not full:
            a = np.clip(a, amin + 0.25, amax - 0.25)
        return center + rmid * (np.cos(a) * o + np.sin(a) * p) + s * axis
    return outward(tris, mid)


def soft_slab(center, axes, w, h, thickness, radius, edge=0.0025, steps=4, seg=8):
    """Tấm bo góc VÀ bo cạnh (cạnh nối mặt trước/sau với thành bo cung tròn bán kính `edge`), nên các mặt kề nhau lệch nhau
    chỉ vài chục độ: MuJoCo làm mượt pháp tuyến đẹp, không còn hiện tượng hình chóp trên mặt phẳng lớn."""
    z = axes[:, 2]
    ts = np.linspace(0, np.pi / 2, steps + 1)
    back = [(-thickness / 2 + edge - edge * np.sin(t), edge - edge * np.cos(t)) for t in ts][::-1]   # từ mặt sau tiến vào giữa
    front = [(thickness / 2 - edge + edge * np.sin(t), edge - edge * np.cos(t)) for t in ts]         # từ giữa ra mặt trước
    zs, ins = zip(*(back + front))
    rings = []
    for zk, ik in zip(zs, ins):
        pts = rounded_rect(w - 2 * ik, h - 2 * ik, max(radius - ik, 1e-4), seg)
        rings.append(center + z * zk + pts[:, :1] * axes[:, 0] + pts[:, 1:] * axes[:, 1])
    tris = []
    n = len(rings[0])
    for a, b in zip(rings[:-1], rings[1:]):
        for k in range(n):
            j = (k + 1) % n
            tris += [[a[k], a[j], b[j]], [a[k], b[j], b[k]]]
    for ring, c in ((rings[0], rings[0].mean(axis=0)), (rings[-1], rings[-1].mean(axis=0))):
        for k in range(n):
            tris.append([c, ring[k], ring[(k + 1) % n]])
    return convex(tris)


def slab(center, axes, w, h, thickness, radius):
    z = axes[:, 2]
    return loft(center - z * thickness / 2, center + z * thickness / 2, axes, (w, h), (w, h), radius)


def build():
    R = quat_to_mat(CAM_QUAT)
    right, up, back = R[:, 0], R[:, 1], R[:, 2]          # camera nhìn theo -back
    look = -back
    axes = np.column_stack([right, up, look])            # khung camera: x = phải, y = lên, z = hướng nhìn
    H = HINGE

    # ---- phần CỐ ĐỊNH (khung gripper): trụ dẹt, hai khớp C ôm chốt, chốt ----
    ear = 0.0085                                          # tâm hai khớp C lệch khỏi mặt phẳng giữa
    post_top = H - np.array([0, 0.0035, 0])
    d = post_top - POST_BASE
    pz = d / np.linalg.norm(d)
    px = np.cross([0.0, 0.0, 1.0], pz)
    px /= np.linalg.norm(px)
    post = loft(POST_BASE, post_top, np.column_stack([px, np.cross(pz, px), pz]), (0.022, 0.0105), (0.0235, 0.0075), 0.0032)
    open_dir = look + np.array([0, 0.4, 0])               # khe chữ C hướng về phía camera (mở cho tay đỡ gập thoải mái)
    c_left = c_ring(H + X * (-ear), X, open_dir, 0.0062, 0.0027, 0.0052, gap_deg=95)
    c_right = c_ring(H + X * (ear), X, open_dir, 0.0062, 0.0027, 0.0052, gap_deg=95)
    pin = cylinder(H, X, 0.0027, 0.0288)
    post_mesh = np.concatenate([post, c_left, c_right, pin])

    # ---- phần GẬP được (khung wrist_cam_arm, gốc tại H): khớp giữa, tay đỡ, HỘP camera liền khối ----
    # Hộp camera là MỘT khối bo góc (như vỏ in 3D ôm bo mạch), chỉ có ống kính nhô ra mặt trước; tay đỡ cắm thẳng vào lưng hộp
    # nên không còn các tấm mỏng chồng lên nhau.
    cam_local = CAM_POS - H
    case_c = cam_local - 0.0118 * look                                   # tâm hộp: mặt trước lùi 4.8 mm sau đầu ống kính
    case = soft_slab(case_c, axes, 0.0360, 0.0360, 0.0140, 0.0070)
    knuckle = c_ring(np.zeros(3), X, look, 0.0060, 0.0029, 0.0098, gap_deg=0)          # khớp giữa kín, nằm giữa hai khớp C
    to_case = case_c                                                      # tay đỡ cắm vào giữa hộp
    arm = loft(np.zeros(3), to_case, frame_from_z(to_case), (0.0098, 0.0062), (0.0140, 0.0080), 0.0030)
    link = np.concatenate([knuckle, arm, case])

    # phần tối màu: ống kính (thân + viền kính) và vành đen quanh miệng ống kính trên mặt trước hộp
    barrel = cylinder(cam_local - 0.0039 * look, look, 0.0058, 0.0058)                      # thân ống kính nhô ra khỏi mặt hộp
    bezel = cylinder(cam_local - 0.0042 * look, look, 0.0080, 0.0012)                       # vành đen quanh miệng ống kính
    glass = cylinder(cam_local - 0.0005 * look, look, 0.0042, 0.0010)                       # mặt kính ở đầu
    module = np.concatenate([bezel, barrel, glass])
    return post_mesh, link, module


def main():
    post, link, module = build()
    write_stl(os.path.join(OUT, "wrist_cam_post.stl"), post)
    write_stl(os.path.join(OUT, "wrist_cam_link.stl"), link)
    write_stl(os.path.join(OUT, "wrist_cam_module.stl"), module)
    print(f"ghi {len(post)} + {len(link)} + {len(module)} tam giac vao {OUT}")
    print("body wrist_cam_arm: pos =", " ".join(f"{v:.5f}" for v in HINGE))
    print("camera arm_cam (cục bộ): pos =", " ".join(f"{v:.5f}" for v in (CAM_POS - HINGE)), "quat =", " ".join(f"{v:.6f}" for v in CAM_QUAT))


if __name__ == "__main__":
    main()
