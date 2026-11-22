"""Phòng thí nghiệm khớp: kéo thanh trượt để chỉnh tay robot, xem số liệu và va chạm ngay bên cạnh.

Cách dùng (đứng ở thư mục chứa sim_engine, đã kích hoạt .venv):
    python -m sim_engine.joint_lab              # mở http://localhost:8765 trong trình duyệt
    python -m sim_engine.joint_lab --port 9000 --no-open

Dùng để làm gì:
    - Kéo 5 khớp + độ mở ngón, xem tay đi đâu (nhìn từ cạnh và từ trên), có chạm bàn / bát / hộp không,
      độ hở tới mặt bàn, trục kẹp nghiêng bao nhiêu so với phương thẳng đứng, hai má kẹp ôm hai MẶT
      hay hai GÓC của hộp.
    - Đặt hộp và bát ở vị trí tùy ý (kể cả xoay hộp).
    - "Giải IK" tới một điểm đích, có thể bật ràng buộc "kẹp thẳng đứng" và kéo về tư thế vòm, để so với
      cách chỉnh tay của bạn.
    - "Lấy tư thế của policy": nạp góc khớp mà policy hiện tại có ở cuối một phase của một seed.

Đây là tư thế TĨNH (không chạy vật lý, không có trọng lực): chỉ dùng hình học và va chạm của MuJoCo
tại đúng tư thế đó. Chụp màn hình trang này gửi cho người khác là đủ để họ thấy mọi con số.
"""

import argparse
import json
import math
import os
import sys
import mimetypes
from pathlib import Path
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import mujoco
import numpy as np

from .. import config, paths
from ..env import SO101PickPlaceEnv
from ..tools.make_replay import arm_model_export, bowl_geom_list, pick_scan_points, unreachable_region
from .api import handle_get as studio_get, handle_post as studio_post
from .runner import LabRun
from .scene import normalize_objects

ARM = ["base", "shoulder", "upper_arm", "lower_arm", "wrist", "gripper"]
JAW = "moving_jaw_so101_v1"      # hàm kẹp di động (hàm cố định thuộc body "gripper")
ARM_ALL = ARM + [JAW]
TABLE_TOP_Z = 0.42
HIDE_PAIRS = {frozenset(p) for p in [("table", "world"), ("base", "table"), ("base", "world"),
                                      ("box", "table"), ("bowl", "table"), ("bowl", "world"), ("box", "world")]}


def hull2(pts):
    """Bao lồi 2D (monotone chain)."""
    P = sorted(set((round(float(a), 5), round(float(b), 5)) for a, b in pts))
    if len(P) < 3:
        return [list(p) for p in P]
    cr = lambda o, a, b: (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lo, up = [], []
    for p in P:
        while len(lo) >= 2 and cr(lo[-2], lo[-1], p) <= 0:
            lo.pop()
        lo.append(p)
    for p in reversed(P):
        while len(up) >= 2 and cr(up[-2], up[-1], p) <= 0:
            up.pop()
        up.append(p)
    return [list(p) for p in lo[:-1] + up[:-1]]


def skew(a):
    return np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])


class Lab:
    def __init__(self, lab_objects=None, target_id=None):
        self.env = SO101PickPlaceEnv(render_images=False, lab_objects=lab_objects, target_id=target_id)
        self.m, self.d = self.env.model, self.env.data
        m = self.m
        self.lock = threading.Lock()
        self.jid = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n) for n in config.JOINT_NAMES[:5]]
        self.qadr = np.array([m.jnt_qposadr[j] for j in self.jid])
        self.dof = [m.jnt_dofadr[j] for j in self.jid]
        self.lo = np.array([m.jnt_range[j][0] for j in self.jid])
        self.hi = np.array([m.jnt_range[j][1] for j in self.jid])
        self.fl = m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "gripper")]   # góc hàm kẹp di động (rad)
        self.box_adr = m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "box_joint")]
        self.bowl = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "bowl")
        self.site = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "ee_site")
        body = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)
        self.arm_ids = [body(n) for n in ARM]
        self.all_ids = [body(n) for n in ARM_ALL + ["wrist_cam_arm"]]
        self.wr = body("gripper")
        geom = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n)
        names = lambda g: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
        self.jaw_fixed = [g for g in range(m.ngeom) if names(g).startswith("fixed_jaw_col")]      # mỗi hàm gồm nhiều khối va chạm
        self.jaw_moving = [g for g in range(m.ngeom) if names(g).startswith("moving_jaw_col")]
        self.kin, self.mesh = arm_model_export(self.env)
        self.bname = lambda b: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(b))
        self.ik = mujoco.MjData(m)
        self.policy_env = None
        self.reach, self.reach_status = None, "idle"
        self.home = [round(float(np.degrees(v)), 1) for v in config.HOME_QPOS_RAD[:5]]
        self.bowl_default = m.body_pos[self.bowl][:2].copy()
        self.runner = LabRun()
        self.exact_model = None

    def set_scene(self, request):
        objects = normalize_objects(request["objects"], request.get("target_id"))
        signature = lambda items: [(o["id"], o["type"], o["color"], o["body"]) for o in items]
        if signature(objects) != signature(self.env.lab_objects):
            replacement = Lab(objects, request.get("target_id"))
            old = self.env
            keep = {key: getattr(self, key) for key in ("lock", "runner", "reach", "reach_status", "exact_model")}
            self.__dict__.update(replacement.__dict__)
            self.__dict__.update(keep)
            old.close()
        self.env.lab_objects = objects
        target = next(obj for obj in objects if obj["body"] == "box")
        self.set_pose(dict(request, bx=target["x"], by=target["y"], byaw=target["yaw"]))
        for obj in objects:
            if obj["body"] == "box":
                continue
            body = self.m.body(obj["body"]).id
            adr = self.m.jnt_qposadr[self.m.body_jntadr[body]]
            half = math.radians(obj["yaw"]) / 2
            self.d.qpos[adr:adr + 7] = [obj["x"], obj["y"], TABLE_TOP_Z + config.BOX_HALF, math.cos(half), 0, 0, math.sin(half)]
        mujoco.mj_forward(self.m, self.d)

    # ---------- đặt tư thế và đọc số liệu ----------
    def set_pose(self, p):
        d = self.d
        d.qpos[self.qadr] = np.radians([p["pan"], p["lift"], p["elbow"], p["wflex"], p["wroll"]])
        d.qpos[self.fl] = p["grip"]
        yaw = math.radians(p["byaw"])
        d.qpos[self.box_adr:self.box_adr + 3] = [p["bx"], p["by"], TABLE_TOP_Z + config.BOX_HALF]
        d.qpos[self.box_adr + 3:self.box_adr + 7] = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
        d.qvel[:] = 0
        self.m.body_pos[self.bowl][:2] = [p["wx"], p["wy"]]
        mujoco.mj_forward(self.m, d)

    def geom_pts(self, g):
        """Các điểm 3D lấy mẫu trên mặt vật (để vẽ bao lồi và tính độ cao thấp nhất)."""
        m, d = self.m, self.d
        pos, R, s = d.geom_xpos[g], d.geom_xmat[g].reshape(3, 3), m.geom_size[g]
        t = int(m.geom_type[g])
        if t == int(mujoco.mjtGeom.mjGEOM_BOX):
            return [pos + R @ (np.array([sx, sy, sz]) * s) for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]
        ang = np.linspace(0, 2 * math.pi, 20, endpoint=False)
        if t == int(mujoco.mjtGeom.mjGEOM_CYLINDER):
            return [pos + R @ np.array([s[0] * math.cos(a), s[0] * math.sin(a), sz * s[1]]) for a in ang for sz in (-1, 1)]
        if t == int(mujoco.mjtGeom.mjGEOM_CAPSULE):
            ends = [pos + R[:, 2] * s[1] * k for k in (-1, 1)]
            out = []
            for e in ends:       # hình chiếu của mặt cầu là hình tròn bán kính r theo mọi hướng
                out += [e + s[0] * np.array([math.cos(a), math.sin(a), 0]) for a in ang]
                out += [e + s[0] * np.array([math.cos(a), 0, math.sin(a)]) for a in ang]
            return out
        if t == int(mujoco.mjtGeom.mjGEOM_MESH):
            mid = m.geom_dataid[g]
            v = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
            return list(v[::25] @ R.T + pos)
        return [pos]

    def geom_3d(self, g, pts):
        """Dữ liệu để vẽ 3D: hộp = 8 đỉnh, trụ = 2 đầu trục + bán kính, capsule = 2 đầu + bán kính."""
        m, d = self.m, self.d
        t, R, pos, s = int(m.geom_type[g]), d.geom_xmat[g].reshape(3, 3), d.geom_xpos[g], m.geom_size[g]
        r3 = lambda v: [round(float(x), 4) for x in v]
        if t == int(mujoco.mjtGeom.mjGEOM_BOX):
            return {"k": "box", "c": [r3(q) for q in pts]}
        if t == int(mujoco.mjtGeom.mjGEOM_CYLINDER):
            return {"k": "cyl", "a": r3(pos - R[:, 2] * s[1]), "b": r3(pos + R[:, 2] * s[1]), "r": float(s[0]),
                    "u": r3(R[:, 0]), "v": r3(R[:, 1])}
        return {"k": "cap", "a": r3(pos - R[:, 2] * s[1]), "b": r3(pos + R[:, 2] * s[1]), "r": float(s[0])}

    def jaw_dev(self):
        """Độ lệch (độ) giữa trục hai má kẹp và các mặt của hộp: 0 = ôm 2 mặt đối diện, 45 = kẹp vào góc."""
        d = self.d
        jaw = d.geom_xpos[self.jaw_moving].mean(axis=0) - d.geom_xpos[self.jaw_fixed].mean(axis=0)
        jaw_yaw = math.degrees(math.atan2(jaw[1], jaw[0])) % 180
        q = d.qpos[self.box_adr + 3:self.box_adr + 7]
        box_yaw = math.degrees(math.atan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2))) % 90
        rel = (jaw_yaw - box_yaw) % 90
        return min(rel, 90 - rel)

    def best_roll(self, p):
        """Quét góc wrist_roll (giữ nguyên các khớp khác), trả về góc làm hai má song song với một cặp mặt hộp, gần góc hiện tại nhất."""
        keep = self.d.qpos[self.qadr[4]]
        cands = []
        for deg in np.arange(-180, 180.01, 1.0):
            self.d.qpos[self.qadr[4]] = math.radians(deg)
            mujoco.mj_kinematics(self.m, self.d)
            cands.append((self.jaw_dev(), float(deg)))
        self.d.qpos[self.qadr[4]] = keep
        mujoco.mj_forward(self.m, self.d)
        best = min(c[0] for c in cands)
        good = [deg for dev, deg in cands if dev <= best + 0.6]
        cur = p["wroll"]
        pick = min(good, key=lambda v: abs(v - cur))
        return {"wroll": pick, "dev": round(best, 2)}

    def snapshot(self):
        m, d = self.m, self.d
        touch, contacts = {}, []
        for i in range(d.ncon):
            c = d.contact[i]
            if c.dist > 0.0005:
                continue
            a, b = self.bname(m.geom_bodyid[c.geom1]), self.bname(m.geom_bodyid[c.geom2])
            if frozenset((a, b)) in HIDE_PAIRS or (a == b):
                continue
            contacts.append({"a": a, "b": b, "pen_mm": round(-c.dist * 1000, 2)})
            for x, y in ((a, b), (b, a)):
                kind = "table" if y in ("table", "world") else y if y in ("bowl", "box") else "arm"
                touch.setdefault(x, set()).add(kind)

        geoms, clearance = [], {}
        for g in range(m.ngeom):
            body = self.bname(m.geom_bodyid[g])
            if int(m.geom_type[g]) == int(mujoco.mjtGeom.mjGEOM_PLANE) or body in ("table", "world"):
                continue
            if body in ARM_ALL:                       # cánh tay thật: vẽ bằng lưới (xem "pose"), ở đây chỉ tính độ hở tới bàn
                if m.geom_group[g] == 3 and m.geom_type[g] == int(mujoco.mjtGeom.mjGEOM_MESH):
                    zmin = min(p[2] for p in self.geom_pts(g))
                    if body != "base":
                        clearance[body] = min(clearance.get(body, 9.0), zmin - TABLE_TOP_Z)
                continue
            pts = self.geom_pts(g)
            geoms.append({
                "body": body, "solid": bool(m.geom_contype[g]), "m": self.geom_3d(g, pts),
                "side": hull2([(p[0], p[2]) for p in pts]), "top": hull2([(p[0], p[1]) for p in pts]),
            })
        P = [d.xpos[b].tolist() for b in self.arm_ids]
        ee = d.site_xpos[self.site].copy()
        axis = ee - d.xpos[self.wr]
        axis /= np.linalg.norm(axis)
        tilt = math.degrees(math.acos(max(-1, min(1, -axis[2]))))
        jaw = d.geom_xpos[self.jaw_moving].mean(axis=0) - d.geom_xpos[self.jaw_fixed].mean(axis=0)
        jaw_yaw = math.degrees(math.atan2(jaw[1], jaw[0])) % 180
        box = d.xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "box")].copy()
        q = d.qpos[self.box_adr + 3:self.box_adr + 7]
        box_yaw = math.degrees(math.atan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2))) % 90
        rel = (jaw_yaw - box_yaw) % 90
        face_dev = min(rel, 90 - rel)
        ju = jaw / (np.linalg.norm(jaw) + 1e-9)
        off = box - ee
        ang = np.degrees(d.qpos[self.qadr])
        near = [bool(min(a - math.degrees(l), math.degrees(h) - a) < 3) for a, l, h in zip(ang, self.lo, self.hi)]
        return {
            "geoms": geoms, "joints_xyz": P, "ee": ee.tolist(), "axis": axis.tolist(), "tilt": tilt,
            "jaw_yaw": jaw_yaw, "face_dev": face_dev, "jaw_deg": math.degrees(float(d.qpos[self.fl])),
            "pose": {self.bname(b): {"p": d.xpos[b].round(5).tolist(), "R": d.xmat[b].round(5).tolist()} for b in self.all_ids},
            "box": box.tolist(), "box_off_jaw_mm": float(off @ ju) * 1000, "box_off_dist_mm": float(np.linalg.norm(off)) * 1000,
            "angles": ang.tolist(), "near_limit": near,
            "limits": [[math.degrees(l), math.degrees(h)] for l, h in zip(self.lo, self.hi)],
            "contacts": contacts, "touch": {k: sorted(v) for k, v in touch.items()},
            "clearance_mm": {k: round(v * 1000, 1) for k, v in clearance.items()},
            "bowl": [float(v) for v in d.xpos[self.bowl]],
        }

    # ---------- IK (độc lập với policy, để thử công thức) ----------
    def solve_ik(self, p, target, vertical, w, K, pref, mask, iters=150, free=(1, 1, 1, 1, 1)):
        m, ik = self.m, self.ik
        ik.qpos[:] = self.d.qpos[:]
        ik.qvel[:] = 0
        mujoco.mj_forward(m, ik)
        jacp, jacr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        q_pref, kk, down = np.radians(pref), K * np.array(mask, float), np.array([0, 0, -1.0])
        used = 0
        for it in range(iters):
            ee = ik.site_xpos[self.site]
            err = np.array(target) - ee
            a = ee - ik.xpos[self.wr]
            a /= np.linalg.norm(a)
            e_o = down - a
            used = it
            if np.linalg.norm(err) < 0.0005 and (not vertical or np.linalg.norm(e_o) < 0.01):
                break
            mujoco.mj_jacSite(m, ik, jacp, jacr, self.site)
            J = jacp[:, self.dof]
            if vertical:
                Jc = np.vstack([J, w * (-skew(a) @ jacr[:, self.dof])])
                ec = np.concatenate([err, w * e_o])
            else:
                Jc, ec = J, err
            Jc = Jc * np.r_[np.array(free, float)][None, :]                    # cột của khớp bị khóa = 0
            Jp = Jc.T @ np.linalg.inv(Jc @ Jc.T + 0.05 ** 2 * np.eye(Jc.shape[0]))
            q = ik.qpos[self.qadr]
            dq = (Jp @ ec + (np.eye(5) - Jp @ Jc) @ (kk * (q_pref - q))) * np.array(free, float)   # khớp không "free" thì đứng yên
            ik.qpos[self.qadr] = np.clip(q + 0.4 * dq, self.lo, self.hi)     # tôn trọng giới hạn khớp
            mujoco.mj_forward(m, ik)
        ee = ik.site_xpos[self.site]
        a = ee - ik.xpos[self.wr]
        a /= np.linalg.norm(a)
        return {"angles": np.degrees(ik.qpos[self.qadr]).round(1).tolist(),
                "err_mm": round(float(np.linalg.norm(np.array(target) - ee)) * 1000, 2),
                "tilt": round(math.degrees(math.acos(max(-1, min(1, -a[2])))), 1), "iters": used + 1}

    # ---------- vùng với tới ----------
    def pick_scan(self):
        """Các điểm đã quét gắp thật (cùng bộ điểm với replay): [[x, y, 1 nếu gắp được], ...]."""
        return pick_scan_points()

    REACH_FILE = paths.REACH_ENVELOPE
    REACH_BANDS = [(0.455, "gắp (đầu tay z 0.455)"), (0.50, "thả (z 0.50)"), (0.62, "trên cao an toàn (z 0.62)")]

    def _reach_solve(self, rd, target, q0, vertical, w=0.3, iters=150):
        """IK độc lập (không dùng dữ liệu chung của server) để dò xem có tư thế nào đưa đầu tay tới `target` không."""
        m = self.m
        rd.qpos[:] = self.env._init_qpos
        rd.qpos[self.qadr] = q0
        rd.qvel[:] = 0
        mujoco.mj_forward(m, rd)
        jacp, jacr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        down = np.array([0, 0, -1.0])
        for _ in range(iters):
            ee = rd.site_xpos[self.site]
            err = np.array(target) - ee
            a = ee - rd.xpos[self.wr]
            a /= np.linalg.norm(a)
            e_o = down - a
            if np.linalg.norm(err) < 0.0005 and (not vertical or np.linalg.norm(e_o) < 0.01):
                break
            mujoco.mj_jacSite(m, rd, jacp, jacr, self.site)
            J = jacp[:, self.dof]
            if vertical:
                Jc = np.vstack([J, w * (-skew(a) @ jacr[:, self.dof])])
                ec = np.concatenate([err, w * e_o])
            else:
                Jc, ec = J, err
            Jp = Jc.T @ np.linalg.inv(Jc @ Jc.T + 0.05 ** 2 * np.eye(Jc.shape[0]))
            rd.qpos[self.qadr] = np.clip(rd.qpos[self.qadr] + 0.4 * (Jp @ ec), self.lo, self.hi)
            mujoco.mj_forward(m, rd)
        ee = rd.site_xpos[self.site]
        a = ee - rd.xpos[self.wr]
        a /= np.linalg.norm(a)
        tilt = math.degrees(math.acos(max(-1, min(1, -a[2]))))
        hit = False
        for i in range(rd.ncon):
            c = rd.contact[i]
            b1, b2 = self.bname(m.geom_bodyid[c.geom1]), self.bname(m.geom_bodyid[c.geom2])
            if c.dist < 0 and ((b1 in ARM_ALL[1:]) and b2 in ("table", "world") or
                               (b2 in ARM_ALL[1:]) and b1 in ("table", "world")):
                hit = True
        return (float(np.linalg.norm(np.array(target) - ee)), tilt, hit,
                np.degrees(rd.qpos[self.qadr]).round(0).tolist(), rd.qpos[self.qadr].copy())

    @staticmethod
    def _intervals(rs, step):
        out = []
        for r in sorted(rs):
            if out and r - out[-1][1] <= step * 1.01:
                out[-1][1] = r
            else:
                out.append([r, r])
        return [[round(a, 3), round(b, 3)] for a, b in out]

    def compute_reach(self, r_step=0.01, starts=10):
        """Với mỗi độ cao đầu tay (gắp, thả, trên cao) và mỗi bán kính tính từ trục khớp 1: có tư thế nào
        (a) đưa tay tới đúng đó, không chạm bàn (hướng tay tự do) và (b) làm được điều đó với tay kẹp gần thẳng đứng?
        Trả về các khoảng bán kính khả thi. Vùng với tới là hình vành khuyên quanh trục khớp 1 vì khớp 1 xoay tự do.

        Dò bằng cách quét bán kính liên tục (ngoài vào trong, rồi trong ra ngoài); mỗi bước xuất phát từ nghiệm của bước
        trước (nếu có) và thêm vài điểm xuất phát ngẫu nhiên, để không bỏ sót nghiệm như khi chỉ thử ngẫu nhiên."""
        rd = mujoco.MjData(self.m)
        rng = np.random.default_rng(0)
        mujoco.mj_forward(self.m, self.d)
        base = [float(v) for v in self.d.xpos[self.arm_ids[1]][:2]]
        radii = np.arange(0.02, 0.42, r_step)
        bands = []
        for z, label in self.REACH_BANDS:
            free, vert, best = set(), set(), {}
            for order in (radii[::-1], radii):
                prev = {"free": None, "vert": None}
                for r in order:
                    tgt = [base[0] + r, base[1], z]
                    for mode in ("free", "vert"):
                        cands = ([prev[mode]] if prev[mode] is not None else []) + [rng.uniform(self.lo, self.hi) for _ in range(starts)]
                        prev[mode] = None
                        for q0 in cands:
                            err, tilt, hit, ang, q = self._reach_solve(rd, tgt, q0, mode == "vert")
                            if err < 0.003 and not hit and (mode == "free" or tilt < 25):
                                prev[mode] = q
                                key = round(float(r), 3)
                                if mode == "free":
                                    free.add(key)
                                    if key not in best or tilt < best[key]["tilt"]:
                                        best[key] = {"tilt": round(tilt, 0), "angles": ang}
                                else:
                                    vert.add(key)
                                break
            bands.append({"z": z, "label": label, "free": self._intervals(free, r_step),
                          "vertical": self._intervals(vert, r_step), "best": {str(k): v for k, v in best.items()}})
        return {"base": base, "r_step": r_step, "bands": bands}

    def ensure_reach(self, force=False):
        """Nạp vùng với tới từ file lưu sẵn, hoặc tính (khoảng 1-2 phút) rồi lưu."""
        try:
            if not force and os.path.exists(self.REACH_FILE):
                with open(self.REACH_FILE, encoding="utf-8") as f:
                    self.reach = json.load(f)
                self.reach_status = "ready"
                return
            self.reach_status = "computing"
            print("Đang tính vùng với tới (một lần, khoảng 1-2 phút)...", flush=True)
            self.reach = self.compute_reach()
            with open(self.REACH_FILE, "w", encoding="utf-8") as f:
                json.dump(self.reach, f, ensure_ascii=False)
            self.reach_status = "ready"
            print("Xong vùng với tới, đã lưu vào", self.REACH_FILE, flush=True)
        except Exception as e:
            self.reach_status = f"error: {e}"

    # ---------- tư thế của policy ----------
    def policy_pose(self, seed, phase):
        from ..policies import default_policy
        if self.policy_env is None:
            self.policy_env = SO101PickPlaceEnv(render_images=False, end_on_success=False)
        env = self.policy_env
        pol = default_policy(env)
        obs, info = env.reset(seed=seed)
        pol.reset()
        got = None
        for _ in range(config.MAX_EPISODE_STEPS):
            ph = pol._phase
            if ph > phase:
                break
            action = pol.select_action({"state": obs["state"], "images": {}, "task": config.DEFAULT_TASK})
            obs, *_ = env.step(action)
            if ph == phase:
                got = (np.degrees(env.data.qpos[self.qadr]).round(1).tolist(), float(env.data.qpos[self.fl]),
                       env.get_box_pos().tolist())
        if got is None:
            return {"error": f"policy không chạy tới phase {phase} ở seed {seed}"}
        return {"angles": got[0], "grip": got[1], "box": got[2]}


PAGE = r"""<!doctype html><html lang="vi"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Phòng thí nghiệm khớp</title>
<style>
:root{--bg:#f6f7f9;--card:#fff;--ink:#1c2128;--muted:#6b7280;--line:#d8dce2;--arm:#6b7280;--warn:#d97706;--bad:#cf222e;--ok:#1a9a3a;--bowl:#2f6fd0;--box:#e24b4a;--table:#e6d7b5;--hand:#c026d3}
@media (prefers-color-scheme:dark){:root{--bg:#14171c;--card:#1d2128;--ink:#e6e9ee;--muted:#9aa3af;--line:#343a44;--table:#4a4130}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:13px/1.35 system-ui,Segoe UI,sans-serif}
h1{font-size:16px;margin:8px 12px 0}.sub{color:var(--muted);margin:0 12px 6px}
.wrap{display:grid;grid-template-columns:minmax(0,1.5fr) minmax(330px,.8fr) minmax(330px,.9fr);gap:8px;padding:8px 12px}
@media (max-width:1250px){.wrap{grid-template-columns:1fr}}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:8px 10px}
.card h2{font-size:11px;letter-spacing:.07em;color:var(--muted);margin:0 0 4px;text-transform:uppercase}
svg{width:100%;height:auto;display:block;margin:0 auto;background:var(--card)}
.views2{display:grid;grid-template-columns:1fr 1fr;gap:8px}
.row{display:grid;grid-template-columns:78px 1fr 62px;gap:6px;align-items:center;margin:2px 0}
.row input[type=range]{width:100%}.row input[type=number]{width:62px;font:inherit;padding:1px 3px}
.mono{font-family:ui-monospace,Consolas,monospace}.tag{display:inline-block;border-radius:9px;padding:0 7px;font-size:12px;font-weight:600;margin:1px 2px 1px 0}
.t-table{background:color-mix(in srgb,var(--warn) 22%,transparent);color:var(--warn)}.t-bowl{background:color-mix(in srgb,#a855f7 22%,transparent);color:#a855f7}.t-box{background:color-mix(in srgb,var(--bad) 18%,transparent);color:var(--bad)}.t-ok{background:color-mix(in srgb,var(--ok) 18%,transparent);color:var(--ok)}
table{border-collapse:collapse;width:100%}td,th{padding:1px 5px;border-bottom:1px solid var(--line);text-align:right;font-size:12.5px}th{color:var(--muted);font-weight:600}td:first-child,th:first-child{text-align:left}
button{font:inherit;padding:2px 9px;border:1px solid var(--line);border-radius:6px;background:var(--card);color:var(--ink);cursor:pointer}button:hover{border-color:var(--ink)}
.bad{color:var(--bad);font-weight:700}.warn{color:var(--warn);font-weight:700}.good{color:var(--ok);font-weight:700}.muted{color:var(--muted)}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:4px 10px}.grid2 label{display:flex;gap:4px;align-items:center}.grid2 input[type=number]{width:70px;font:inherit}
</style></head><body>
<h1>Joint Lab · đặt cảnh và chạy thử</h1><p class="sub">Kéo hộp đỏ / bát xanh trong hình nhìn từ trên. Kéo dấu đầu tay màu tím ở hình trên hoặc cạnh để giải IK; thanh trượt chỉnh từng khớp. Bấm Chạy để mô phỏng từ tư thế hiện tại (có 50 bước ổn định vật lý ban đầu).</p>
<div class="card" style="margin:8px 12px"><button id="bRun">▶ Chạy mô phỏng · seed lab</button> <a id="replayLink" href="/replay-lab" target="_blank" hidden>Mở replay mới nhất ↗</a>
 <span id="runStatus" role="status" aria-live="polite">Chỉ chạy 1 trường hợp lab; mỗi lần chạy ghi đè kết quả riêng, không chạy các seed cũ.</span></div>
<div class="wrap">
<div>
 <div class="card"><h2>3D · kéo chuột = xoay · lăn = zoom · nhấp đúp = mặc định <span style="float:right"><button id="v1">chéo</button> <button id="v2">trước</button> <button id="v3">cạnh</button> <button id="v4">trên</button></span></h2><canvas id="c3" style="width:100%;height:400px;display:block"></canvas></div>
<div class="views2" style="margin-top:8px">
  <svg id="side" viewBox="0 0 600 420" style="display:none"></svg>
  <div class="card"><h2>Từ trên xuống (x ngang, y dọc) <label style="float:right;font-weight:400;text-transform:none;letter-spacing:0"><input type="checkbox" id="showReach" checked> vùng với tới</label></h2><svg id="top" viewBox="0 0 600 500"></svg></div>
 </div>
</div>
<div>
 <div class="card"><h2>Khớp (độ) và hàm kẹp</h2><div id="sl"></div>
  <div style="margin-top:6px"><button id="bHome">về home</button> <button id="bOpen">mở hàm kẹp</button> <button id="bClose">đóng hàm kẹp</button></div></div>
 <div class="card" style="margin-top:8px"><h2>Hộp và bát (cm)</h2><div id="ob"></div></div>
 <div class="card" style="margin-top:8px"><h2>Nút nhanh (xong thì chỉ cần tinh chỉnh khớp)</h2>
  <div style="display:flex;flex-wrap:wrap;gap:4px">
   <button data-q="above-box">đích → trên hộp</button><button data-q="grasp">đích → vị trí gắp</button>
   <button data-q="above-bowl">đích → trên bát</button><button data-q="place">đích → vị trí thả</button>
   <button data-q="pan-box">pan quay về hộp</button><button data-q="pan-bowl">pan quay về bát</button>
   <button data-q="vertical">ép kẹp thẳng đứng (giữ đầu tay)</button><button data-q="roll">wroll ôm 2 mặt hộp</button>
  </div><div id="qres" class="mono muted" style="margin-top:4px"></div></div>
 <div class="card" style="margin-top:8px"><h2>Giải IK tới một điểm đích</h2>
  <div class="grid2"><label>x (cm)<input type="number" id="tx" value="13.6" step="0.5"></label><label>y (cm)<input type="number" id="ty" value="-1.2" step="0.5"></label>
  <label>z (cm)<input type="number" id="tz" value="45.5" step="0.5"></label><label><input type="checkbox" id="tv"> tay kẹp thẳng đứng</label>
  <label>trọng số<input type="number" id="tw" value="0.05" step="0.01"></label><label>K vòm<input type="number" id="tk" value="0.1" step="0.05"></label>
  <label>lift ưa thích<input type="number" id="pl" value="10"></label><label>elbow ưa thích<input type="number" id="pe" value="-15"></label>
  <label>wflex ưa thích<input type="number" id="pw" value="60"></label><label></label></div>
  <div style="margin-top:6px"><button id="bIk">giải IK từ tư thế hiện tại</button> <span id="ikres" class="mono muted"></span></div></div>
 <div class="card" style="margin-top:8px"><h2>Tư thế của policy</h2>
  <label>seed <input type="number" id="ps" value="2" style="width:60px"></label> <label>cuối phase <input type="number" id="pp" value="3" min="0" max="10" style="width:50px"></label>
  <button id="bPol">nạp tư thế</button> <span id="polres" class="muted"></span></div>
</div>
<div>
 <div class="card"><h2>Số liệu</h2><div id="info"></div></div>
</div>
</div>
<section id="labReplay" class="card" style="margin:8px 12px" hidden><h2>Replay · trường hợp lab mới nhất</h2><iframe id="replayFrame" title="Replay mô phỏng Joint Lab" style="width:100%;height:850px;border:0"></iframe></section>
<script>
const NAMES=[["pan","1 pan (xoay thân)"],["lift","2 lift (nâng vai)"],["elbow","3 elbow (gập khuỷu)"],["wflex","4 wflex (gập cổ tay)"],["wroll","5 wroll (xoay cổ tay)"]];
const S={pan:0,lift:0,elbow:0,wflex:0,wroll:0,grip:1.2,bx:0.15,by:0,byaw:0,wx:-0.15,wy:0};
let GOPEN=1.2,GCLOSE=-0.1,MODEL=null;
let LIM=[[-180,180],[-90,90],[-180,0],[-90,90],[-180,180]],busy=false,again=false,last=null,REACH=null;
const BANDC=["#0e9aa7","#d97706","#8b8f98"];
async function loadReach(){try{const r=await (await fetch("/reach")).json();if(r.status==="ready"){REACH=r.reach;if(last)render(last)}else{setTimeout(loadReach,3000)}}catch(e){setTimeout(loadReach,5000)}}
let PICK=null,PSCAN=[];
const sector=(n=48)=>{const P=PICK,pts=[];for(let q=0;q<=n;q++){const a=(P.angle_min_deg+(P.angle_max_deg-P.angle_min_deg)*q/n)*Math.PI/180;pts.push([P.pan[0]+P.r_max*Math.sin(a),P.pan[1]+P.r_max*Math.cos(a)])}
  for(let q=n;q>=0;q--){const a=(P.angle_min_deg+(P.angle_max_deg-P.angle_min_deg)*q/n)*Math.PI/180;pts.push([P.pan[0]+P.r_min*Math.sin(a),P.pan[1]+P.r_min*Math.cos(a)])}return pts};
const cpath=(cx,cy,r)=>`M${cx+r},${cy}a${r},${r} 0 1 0 ${-2*r},0a${r},${r} 0 1 0 ${2*r},0`;
function reachTop(J){
  if(!(REACH||PICK)||!$("showReach").checked)return "";
  const X=x=>(x+0.45)*666.7,Y=y=>(0.30-y)*666.7;let s="";
  if(PICK){   // vùng gắp được THẬT: hình quạt đo bằng quét gắp; chấm = gắp được, x đỏ = hỏng
    s+=`<polygon points="${sector().map(([x,y])=>X(x)+","+Y(y)).join(" ")}" fill="#0e9aa7" fill-opacity=".16" stroke="#0e9aa7" stroke-width="1.6" stroke-dasharray="6 4"/>`;
    PSCAN.forEach(([x,y,ok])=>{const a=X(x),b=Y(y);s+=ok?`<circle cx="${a}" cy="${b}" r="2.6" fill="#0e9aa7" opacity=".6"/>`:`<path d="M${a-3.4},${b-3.4}l6.8,6.8m0,-6.8l-6.8,6.8" stroke="#cf222e" stroke-width="1.8" opacity=".9"/>`});
  }
  if(!REACH)return s;
  const b=J.joints_xyz[1],cx=X(b[0]),cy=Y(b[1]);
  REACH.bands.forEach((bd,i)=>{const c=BANDC[i];
    bd.free.forEach(([a,r])=>{s+=`<path d="${cpath(cx,cy,(r+REACH.r_step/2)*666.7)}${cpath(cx,cy,Math.max(0,a-REACH.r_step/2)*666.7)}" fill="${c}" fill-opacity="${i===0?.13:.05}" fill-rule="evenodd" stroke="${c}" stroke-width="1.6" stroke-opacity=".9"/>`});
    bd.vertical.forEach(([a,r])=>{s+=`<circle cx="${cx}" cy="${cy}" r="${(r+REACH.r_step/2)*666.7}" fill="none" stroke="${c}" stroke-width="1.6" stroke-dasharray="6 4"/>`+(a>0.03?`<circle cx="${cx}" cy="${cy}" r="${Math.max(0,a-REACH.r_step/2)*666.7}" fill="none" stroke="${c}" stroke-width="1.6" stroke-dasharray="6 4"/>`:"")});
  });return s;
}
function reachInfo(J){
  if(!REACH)return `<div class="muted" style="margin-top:6px">Đang tính vùng với tới (lần đầu mất 1-2 phút)…</div>`;
  const b=J.joints_xyz[1],rb=Math.hypot(J.box[0]-b[0],J.box[1]-b[1]),rw=Math.hypot(J.bowl[0]-b[0],J.bowl[1]-b[1]);
  const inn=(iv,r)=>iv.some(([a,c])=>r>=a-REACH.r_step/2&&r<=c+REACH.r_step/2),fmt=iv=>iv.length?iv.map(([a,c])=>(a*100).toFixed(0)+"–"+(c*100).toFixed(0)).join(", ")+" cm":"không";
  const mark=ok=>ok?'<span class="good">✓</span>':'<span class="bad">✗</span>';
  let h=`<div style="margin-top:8px"><b>Vùng với tới (hình học IK)</b> <span class="muted">(bán kính từ trục khớp 1, KHÔNG tính giới hạn xoay khớp 1 hay va chạm; nét liền = hướng tay tự do, nét đứt = tay kẹp gần thẳng đứng). Vùng GẮP ĐƯỢC thật là hình quạt xanh có chấm quét.</span></div><table><tr><th>độ cao đầu tay</th><th>tự do</th><th>thẳng đứng</th></tr>`;
  REACH.bands.forEach((bd,i)=>{h+=`<tr><td><span style="color:${BANDC[i]}">●</span> ${bd.label}</td><td class="mono">${fmt(bd.free)}</td><td class="mono">${fmt(bd.vertical)}</td></tr>`});
  h+=`</table><div style="margin-top:4px">Hộp cách trục <span class="mono">${(rb*100).toFixed(1)} cm</span>: vùng gắp tự do ${mark(inn(REACH.bands[0].free,rb))} · thẳng đứng ${mark(inn(REACH.bands[0].vertical,rb))}</div>`;
  h+=`<div>Bát cách trục <span class="mono">${(rw*100).toFixed(1)} cm</span>: vùng thả tự do ${mark(inn(REACH.bands[1].free,rw))} · thẳng đứng ${mark(inn(REACH.bands[1].vertical,rw))} · trên cao ${mark(inn(REACH.bands[2].free,rw))}</div>`;
  return h;
}
const $=id=>document.getElementById(id);
function row(id,label,lo,hi,step,val,fmt){return `<div class="row"><span>${label}</span><input type="range" id="r_${id}" min="${lo}" max="${hi}" step="${step}" value="${val}"><input type="number" id="n_${id}" min="${lo}" max="${hi}" step="${step}" value="${val}"></div>`}
function build(){
  $("sl").innerHTML=NAMES.map(([k,l],i)=>row(k,l,LIM[i][0],LIM[i][1],0.5,S[k])).join("")+row("grip","hàm kẹp (°)",-10,100,0.5,S.grip*180/Math.PI);
  $("ob").innerHTML=row("bx","hộp x",-45,45,0.1,S.bx*100)+row("by","hộp y",-45,30,0.1,S.by*100)+row("byaw","hộp xoay °",-90,90,1,S.byaw)+row("wx","bát x",-45,45,0.1,S.wx*100)+row("wy","bát y",-45,30,0.1,S.wy*100);
  for(const k of ["pan","lift","elbow","wflex","wroll","grip","bx","by","byaw","wx","wy"]){
    const r=$("r_"+k),n=$("n_"+k);
    const set=v=>{v=+v;S[k]=(k==="grip")?v*Math.PI/180:(["bx","by","wx","wy"].includes(k)?v/100:v);r.value=v;n.value=v;go()};
    r.oninput=()=>set(r.value);n.onchange=()=>set(n.value);
  }
}
function syncUI(){for(const [k,s] of [["pan",1],["lift",1],["elbow",1],["wflex",1],["wroll",1],["grip",180/Math.PI],["bx",100],["by",100],["byaw",1],["wx",100],["wy",100]]){const v=Math.round(S[k]*s*10)/10;$("r_"+k).value=v;$("n_"+k).value=v}}
async function go(){
  if(busy){again=true;return}busy=true;
  try{const q=new URLSearchParams(S);const r=await fetch("/pose?"+q);last=await r.json();render(last);
  }finally{busy=false;if(again){again=false;go()}}
}
const f=(v,n=3)=>(+v).toFixed(n);
function touchColor(body,T){const t=T[body]||[];return t.includes("box")?"var(--bad)":t.includes("table")?"var(--warn)":t.includes("bowl")?"#a855f7":null}
/* ---------- cánh tay THẬT: lưới tam giác giản lược của từng link, đặt theo tư thế MuJoCo trả về ---------- */
function armWorld(J){const out={};if(!MODEL)return out;
  for(const nm in MODEL.mesh){const T=J.pose[nm];if(!T)continue;const R=T.R,p=T.p,v=MODEL.mesh[nm].v,w=new Float64Array(v.length);
    for(let i=0;i<v.length;i+=3){w[i]=R[0]*v[i]+R[1]*v[i+1]+R[2]*v[i+2]+p[0];w[i+1]=R[3]*v[i]+R[4]*v[i+1]+R[5]*v[i+2]+p[1];w[i+2]=R[6]*v[i]+R[7]*v[i+1]+R[8]*v[i+2]+p[2]}
    out[nm]=w}return out}
const ARMCOL="#e8b923";
function linkColor(J,nm){const t=J.touch[nm]||[];return t.includes("box")?"#e5484d":t.includes("table")?"#f59e0b":t.includes("bowl")?"#a855f7":ARMCOL}
function hull2(pts){const P=pts.slice().sort((a,b)=>a[0]-b[0]||a[1]-b[1]);if(P.length<3)return P;const cr=(o,a,b)=>(a[0]-o[0])*(b[1]-o[1])-(a[1]-o[1])*(b[0]-o[0]);const lo=[],up=[];
  for(const p of P){while(lo.length>=2&&cr(lo[lo.length-2],lo[lo.length-1],p)<=0)lo.pop();lo.push(p)}
  for(const p of P.slice().reverse()){while(up.length>=2&&cr(up[up.length-2],up[up.length-1],p)<=0)up.pop();up.push(p)}
  return lo.slice(0,-1).concat(up.slice(0,-1))}
function armHulls(J,fx,fy,a,b){let s="";const W3=armWorld(J);
  for(const nm in W3){const w=W3[nm],pts=[];for(let i=0;i<w.length;i+=6)pts.push([fx(w[i+a]),fy(w[i+b])]);
    s+=`<polygon points="${hull2(pts).map(p=>p[0].toFixed(1)+","+p[1].toFixed(1)).join(" ")}" fill="${linkColor(J,nm)}" fill-opacity=".78" stroke="#3a2c00" stroke-opacity=".55" stroke-width=".8"/>`}
  return s}
function render(J){
  const X=x=>(x+0.45)*666.7,Z=z=>(0.88-z)*666.7,Y=y=>(0.30-y)*666.7;
  const col=g=>{if(g.body==="box")return["var(--box)","var(--box)",.9];if(g.body==="bowl")return["var(--bowl)","var(--bowl)",.3];
    const c=touchColor(g.body,J.touch);return c?[c,c,.45]:[g.solid?"var(--arm)":"#888",g.solid?"var(--arm)":"#888",g.solid?.3:.12]};
  const poly=(pts,fx,fy,c)=>`<polygon points="${pts.map(p=>fx(p[0])+","+fy(p[1])).join(" ")}" fill="${c[0]}" fill-opacity="${c[2]}" stroke="${c[1]}" stroke-width="1.2"/>`;
  const grid=(horiz)=>{let s="";for(let v=-0.45;v<=0.451;v+=0.05){s+=`<line x1="${X(v)}" y1="0" x2="${X(v)}" y2="${horiz?500:420}" stroke="var(--muted)" stroke-opacity=".15"/><text x="${X(v)}" y="${(horiz?500:420)-3}" font-size="9" text-anchor="middle" fill="var(--muted)">${v.toFixed(2)}</text>`}return s};
  // cạnh
  let s=grid(false);for(let v=0.30;v<=0.88;v+=0.05)s+=`<line x1="0" y1="${Z(v)}" x2="600" y2="${Z(v)}" stroke="var(--muted)" stroke-opacity=".15"/><text x="3" y="${Z(v)-2}" font-size="9" fill="var(--muted)">${v.toFixed(2)}</text>`;
  s+=`<rect x="0" y="${Z(0.42)}" width="600" height="${Z(0.38)-Z(0.42)}" fill="var(--table)"/><line x1="0" y1="${Z(0.42)}" x2="600" y2="${Z(0.42)}" stroke="var(--muted)" stroke-width="1.5"/>`;
  J.geoms.forEach(g=>s+=poly(g.side,X,Z,col(g)));
  s+=armHulls(J,X,Z,0,2);
  const P=J.joints_xyz,ee=J.ee,ax=J.axis;
  s+=`<line x1="${X(ee[0])}" y1="${Z(ee[2])}" x2="${X(ee[0]+ax[0]*0.08)}" y2="${Z(ee[2]+ax[2]*0.08)}" stroke="var(--hand)" stroke-dasharray="4 3"/>`;
  s+=`<line x1="${X(ee[0])}" y1="${Z(ee[2])}" x2="${X(ee[0])}" y2="${Z(ee[2]-0.08)}" stroke="var(--muted)" stroke-dasharray="1 3" stroke-opacity=".6"/>`;
  for(let k=1;k<=5;k++)s+=`<circle cx="${X(P[k][0])}" cy="${Z(P[k][2])}" r="8" fill="var(--card)" stroke="var(--ink)"/><text x="${X(P[k][0])}" y="${Z(P[k][2])+3.5}" font-size="10" font-weight="700" text-anchor="middle" fill="var(--ink)">${k}</text>`;
  s+=`<path d="M${X(ee[0])-8},${Z(ee[2])}H${X(ee[0])+8}M${X(ee[0])},${Z(ee[2])-8}V${Z(ee[2])+8}" stroke="var(--hand)" stroke-width="2.5"/>`;
  $("side").innerHTML=s;
  $("side").innerHTML+=dragHandle(X(ee[0]),Z(ee[2]),"arm","Đầu tay · kéo XZ","var(--hand)");
  // trên
  s=grid(true);for(let v=-0.45;v<=0.301;v+=0.05)s+=`<line x1="0" y1="${Y(v)}" x2="600" y2="${Y(v)}" stroke="var(--muted)" stroke-opacity=".15"/><text x="3" y="${Y(v)-2}" font-size="9" fill="var(--muted)">${v.toFixed(2)}</text>`;
  s=`<rect x="0" y="0" width="600" height="500" fill="var(--table)" fill-opacity=".6"/>`+s+reachTop(J);
  J.geoms.forEach(g=>s+=poly(g.top,X,Y,col(g)));
  s+=armHulls(J,X,Y,0,1);
  for(let k=1;k<=5;k++)s+=`<circle cx="${X(P[k][0])}" cy="${Y(P[k][1])}" r="8" fill="var(--card)" stroke="var(--ink)"/><text x="${X(P[k][0])}" y="${Y(P[k][1])+3.5}" font-size="10" font-weight="700" text-anchor="middle" fill="var(--ink)">${k}</text>`;
  s+=`<path d="M${X(ee[0])-8},${Y(ee[1])}H${X(ee[0])+8}M${X(ee[0])},${Y(ee[1])-8}V${Y(ee[1])+8}" stroke="var(--hand)" stroke-width="2.5"/>`;
  $("top").innerHTML=s;
  $("top").innerHTML+=dragHandle(X(J.box[0]),Y(J.box[1]),"box","Hộp · kéo XY","var(--box)")+dragHandle(X(J.bowl[0]),Y(J.bowl[1]),"bowl","Bát · kéo XY","var(--bowl)")+dragHandle(X(ee[0]),Y(ee[1]),"arm","Đầu tay · kéo XY","var(--hand)");
  draw3d(J);info(J);
}
function dragHandle(x,y,kind,label,color){return `<circle data-drag="${kind}" cx="${x}" cy="${y}" r="13" fill="${color}" fill-opacity=".2" stroke="${color}" stroke-width="2.5" style="cursor:grab"><title>${label}</title></circle>`}
// Capture on the SVG, whose identity survives render(), rather than a replaced handle.
for(const id of ["top","side"]){
  const svg=$(id);let drag=null,pending=null,solving=false;
  svg.style.touchAction="none";
  const solve=async()=>{if(solving)return;solving=true;try{while(pending){const target=pending;pending=null;await doIk(target)}}catch(e){$("ikres").textContent=e.message}finally{solving=false}};
  const move=e=>{if(!drag)return;const pt=new DOMPoint(e.clientX,e.clientY).matrixTransform(svg.getScreenCTM().inverse());
    const x=Math.max(-.45,Math.min(.45,pt.x/666.7-.45));
    const y=Math.max(-.45,Math.min(.30,.30-pt.y/666.7));
    if(drag.kind==="box"||drag.kind==="bowl"){const k=drag.kind==="box"?"b":"w";S[k+"x"]=x;S[k+"y"]=y;syncUI();go()}
    else{const target=id==="top"?[x,y,drag.ee[2]]:[x,drag.ee[1],Math.max(.30,Math.min(.88,.88-pt.y/666.7))];setT(...target);pending=target;solve()}
  };
  svg.onpointerdown=e=>{const kind=e.target.dataset.drag;if(!kind||!last)return;e.preventDefault();drag={kind,ee:[...last.ee]};svg.setPointerCapture(e.pointerId)};
  svg.onpointermove=move;
  svg.onpointerup=e=>{move(e);drag=null;if(svg.hasPointerCapture(e.pointerId))svg.releasePointerCapture(e.pointerId)};
  svg.onpointercancel=()=>{drag=null;pending=null};
}
/* ---------- 3D tự vẽ bằng canvas 2D ---------- */
const CAM0={yaw:-0.95,pitch:0.5,dist:0.7},CAM={...CAM0,t:[0.0,0.0,0.5]};
const css=n=>getComputedStyle(document.documentElement).getPropertyValue("--"+n).trim();
function makeProj(w,h){
  const cp=Math.cos(CAM.pitch),sp=Math.sin(CAM.pitch),cy=Math.cos(CAM.yaw),sy=Math.sin(CAM.yaw);
  const cam=[CAM.t[0]+CAM.dist*cp*cy,CAM.t[1]+CAM.dist*cp*sy,CAM.t[2]+CAM.dist*sp],f=[-cp*cy,-cp*sy,-sp];
  let r=[f[1],-f[0],0];const rl=Math.hypot(r[0],r[1])||1;r=[r[0]/rl,r[1]/rl,0];
  const u=[r[1]*f[2]-r[2]*f[1],r[2]*f[0]-r[0]*f[2],r[0]*f[1]-r[1]*f[0]],fp=Math.min(w,h)*1.45;
  const P=v=>{const d=[v[0]-cam[0],v[1]-cam[1],v[2]-cam[2]],zc=d[0]*f[0]+d[1]*f[1]+d[2]*f[2];return [w/2+fp*(d[0]*r[0]+d[1]*r[1])/zc,h/2-fp*(d[0]*u[0]+d[1]*u[1]+d[2]*u[2])/zc,zc]};
  P.fp=fp;P.r=r;P.u=u;return P;
}
const EDG=[[0,1],[2,3],[4,5],[6,7],[0,2],[1,3],[4,6],[5,7],[0,4],[1,5],[2,6],[3,7]],FAC=[[0,1,3,2],[4,5,7,6],[0,1,5,4],[2,3,7,6],[0,2,6,4],[1,3,7,5]];
function draw3d(J){
  const cv=$("c3"),w=cv.clientWidth,h=cv.clientHeight;if(!w||!h)return;
  const dpr=window.devicePixelRatio||1;if(cv.width!==Math.round(w*dpr)||cv.height!==Math.round(h*dpr)){cv.width=Math.round(w*dpr);cv.height=Math.round(h*dpr)}
  const g=cv.getContext("2d");g.setTransform(dpr,0,0,dpr,0,0);g.clearRect(0,0,w,h);
  const P=makeProj(w,h),K={};["arm","warn","bad","bowl","box","hand","ink","muted","card","table"].forEach(n=>K[n]=css(n));
  const line=(a,b,c,lw,al,dash)=>{const p=P(a),q=P(b);g.globalAlpha=al??1;g.strokeStyle=c;g.lineWidth=lw;g.setLineDash(dash||[]);g.beginPath();g.moveTo(p[0],p[1]);g.lineTo(q[0],q[1]);g.stroke();g.globalAlpha=1;g.setLineDash([])};
  const poly=(pts,fill,stroke,lw,al)=>{g.beginPath();pts.forEach((v,k)=>{const p=P(v);k?g.lineTo(p[0],p[1]):g.moveTo(p[0],p[1])});g.closePath();g.globalAlpha=al??1;if(fill){g.fillStyle=fill;g.fill()}if(stroke){g.strokeStyle=stroke;g.lineWidth=lw||1;g.stroke()}g.globalAlpha=1};
  const TZ=0.42;
  poly([[-.45,-.40,TZ],[.45,-.40,TZ],[.45,.40,TZ],[-.45,.40,TZ]],K.table,K.muted,1,.6);
  for(let v=-0.4;v<=0.401;v+=0.1){line([v,-.40,TZ],[v,.40,TZ],K.muted,.5,.3);line([-.45,v,TZ],[.45,v,TZ],K.muted,.5,.3)}
  if(PICK&&$("showReach").checked){poly(sector().map(([x,y])=>[x,y,TZ+0.001]),"#0e9aa7","#0e9aa7",1.2,.22)}
  if(REACH&&$("showReach").checked){const bb=J.joints_xyz[1],ringAt=(rad,n=72)=>Array.from({length:n},(_,k)=>[bb[0]+rad*Math.cos(2*Math.PI*k/n),bb[1]+rad*Math.sin(2*Math.PI*k/n),TZ+0.001]);
    REACH.bands.forEach((bd,i)=>{const c=BANDC[i];bd.free.forEach(([a,r])=>{poly(ringAt(r+REACH.r_step/2),null,c,1.5,.9);if(a>0.03)poly(ringAt(a-REACH.r_step/2),null,c,1.5,.9)});
      bd.vertical.forEach(([a,r])=>{g.setLineDash([6,4]);poly(ringAt(r+REACH.r_step/2),null,c,1.3,.9);g.setLineDash([])})})}
  const colorOf=gm=>{const c=touchColor(gm.body,J.touch);if(gm.body==="box")return[K.box,.92];if(gm.body==="bowl")return[K.bowl,.28];
    return c?[c.startsWith("var")?css(c.slice(6,-1)):c,.55]:[gm.solid?K.arm:"#888",gm.solid?.3:.15]};
  const items=J.geoms.map(gm=>{const m=gm.m;let c;
    if(m.k==="box"){c=[0,1,2].map(a=>m.c.reduce((s,v)=>s+v[a],0)/8)}else c=[0,1,2].map(a=>(m.a[a]+m.b[a])/2);
    return {gm,m,z:P(c)[2]}}).sort((a,b)=>b.z-a.z);
  for(const {gm,m} of items){const [col,al]=colorOf(gm);
    if(m.k==="box"){FAC.map(f=>({f,z:f.reduce((s,k)=>s+P(m.c[k])[2],0)/4})).sort((a,b)=>b.z-a.z).forEach(({f})=>poly(f.map(i=>m.c[i]),col,null,0,al*.55));EDG.forEach(([a,b])=>line(m.c[a],m.c[b],col,1.2,Math.min(1,al+.35)))}
    else if(m.k==="cyl"){const ring=pt=>Array.from({length:20},(_,i)=>{const t=i/20*2*Math.PI,c=Math.cos(t)*m.r,s=Math.sin(t)*m.r;return[pt[0]+c*m.u[0]+s*m.v[0],pt[1]+c*m.u[1]+s*m.v[1],pt[2]+c*m.u[2]+s*m.v[2]]});
      const ra=ring(m.a),rb=ring(m.b);poly(ra,col,col,1,al);poly(rb,col,col,1,al);for(let i=0;i<20;i+=5)line(ra[i],rb[i],col,1,Math.min(1,al+.3))}
    else{const mid=[0,1,2].map(a=>(m.a[a]+m.b[a])/2),wpx=2*m.r*P.fp/Math.max(.001,P(mid)[2]);g.lineCap="round";line(m.a,m.b,col,wpx,al*.55);line(m.a,m.b,col,2,Math.min(1,al+.4));g.lineCap="butt"}}
  {const W3=armWorld(J),tris=[],L=[0.35,-0.45,0.82],ln=Math.hypot(...L);
   for(const nm in W3){const w=W3[nm],F=MODEL.mesh[nm].f,n=w.length/3,sc=new Float64Array(n*3);
    for(let k=0;k<n;k++){const q=P([w[3*k],w[3*k+1],w[3*k+2]]);sc[3*k]=q[0];sc[3*k+1]=q[1];sc[3*k+2]=q[2]}
    const c=linkColor(J,nm),base=[1,3,5].map(k=>parseInt(c.slice(k,k+2),16));
    for(let t=0;t<F.length;t+=3){const i0=F[t],i1=F[t+1],i2=F[t+2],ux=w[3*i1]-w[3*i0],uy=w[3*i1+1]-w[3*i0+1],uz=w[3*i1+2]-w[3*i0+2],vx=w[3*i2]-w[3*i0],vy=w[3*i2+1]-w[3*i0+1],vz=w[3*i2+2]-w[3*i0+2];
      const nx=uy*vz-uz*vy,ny=uz*vx-ux*vz,nz=ux*vy-uy*vx,nl=Math.hypot(nx,ny,nz)||1,sh=0.42+0.58*Math.abs((nx*L[0]+ny*L[1]+nz*L[2])/(nl*ln));
      tris.push([(sc[3*i0+2]+sc[3*i1+2]+sc[3*i2+2])/3,i0,i1,i2,sc,base,sh])}}
   tris.sort((p,q)=>q[0]-p[0]);g.lineJoin="round";g.lineWidth=.6;
   for(const [,i0,i1,i2,sc,base,sh] of tris){const col=`rgb(${(base[0]*sh)|0},${(base[1]*sh)|0},${(base[2]*sh)|0})`;g.fillStyle=col;g.strokeStyle=col;g.beginPath();g.moveTo(sc[3*i0],sc[3*i0+1]);g.lineTo(sc[3*i1],sc[3*i1+1]);g.lineTo(sc[3*i2],sc[3*i2+1]);g.closePath();g.fill();g.stroke()}}
  const ee=J.ee,ax=J.axis,A=J.joints_xyz;
  line(ee,[ee[0]+ax[0]*.09,ee[1]+ax[1]*.09,ee[2]+ax[2]*.09],K.hand,1.5,1,[4,3]);line(ee,[ee[0],ee[1],ee[2]-.09],K.muted,1,.7,[1,3]);line(ee,[ee[0],ee[1],TZ],K.hand,1,.6,[2,3]);
  for(let k=1;k<=5;k++){const q=P(A[k]);g.fillStyle=K.card;g.strokeStyle=K.ink;g.lineWidth=1.3;g.beginPath();g.arc(q[0],q[1],8,0,7);g.fill();g.stroke();g.fillStyle=K.ink;g.font="bold 10px system-ui,sans-serif";g.textAlign="center";g.fillText(k,q[0],q[1]+3.5)}
  {const q=P(ee);g.strokeStyle=K.hand;g.lineWidth=2.5;g.beginPath();g.moveTo(q[0]-8,q[1]);g.lineTo(q[0]+8,q[1]);g.moveTo(q[0],q[1]-8);g.lineTo(q[0],q[1]+8);g.stroke()}
  // la bàn trục ở góc
  {const r=P.r,u=P.u,ox=40,oy=h-36,L=30;g.fillStyle=K.card;g.globalAlpha=.65;g.beginPath();g.arc(ox,oy,L+12,0,7);g.fill();g.globalAlpha=1;
   [[[1,0,0],"#d33","x"],[[0,1,0],"#2a2","y"],[[0,0,1],"#38e","z"]].forEach(([d,c,n])=>{const sx=d[0]*r[0]+d[1]*r[1],sy=-(d[0]*u[0]+d[1]*u[1]+d[2]*u[2]);
    g.strokeStyle=c;g.fillStyle=c;g.lineWidth=2.4;g.beginPath();g.moveTo(ox,oy);g.lineTo(ox+L*sx,oy+L*sy);g.stroke();g.font="bold 12px system-ui,sans-serif";g.textAlign="center";g.fillText(n,ox+(L+8)*sx,oy+(L+8)*sy+4)})}
}
(function cam(){
  const cv=$("c3");let drag=null;
  cv.onpointerdown=e=>{drag=[e.clientX,e.clientY];cv.setPointerCapture(e.pointerId)};
  cv.onpointermove=e=>{if(!drag)return;CAM.yaw-=(e.clientX-drag[0])*.008;CAM.pitch=Math.max(-.2,Math.min(1.5,CAM.pitch+(e.clientY-drag[1])*.008));drag=[e.clientX,e.clientY];if(last)draw3d(last)};
  cv.onpointerup=()=>{drag=null};
  cv.onwheel=e=>{e.preventDefault();CAM.dist=Math.max(.3,Math.min(2.5,CAM.dist*(e.deltaY>0?1.1:.9)));if(last)draw3d(last)};
  cv.ondblclick=()=>{Object.assign(CAM,CAM0);if(last)draw3d(last)};
  const view=(y,p)=>()=>{CAM.yaw=y;CAM.pitch=p;CAM.dist=CAM0.dist;if(last)draw3d(last)};
  $("v1").onclick=view(-.95,.5);$("v2").onclick=view(Math.PI,.15);$("v3").onclick=view(-Math.PI/2,.05);$("v4").onclick=view(-Math.PI/2,1.5);
  new ResizeObserver(()=>{if(last)draw3d(last)}).observe(cv);
})();
function info(J){
  const tag=(k,t)=>`<span class="tag t-${k}">${t}</span>`;
  let h=`<table><tr><th>khớp</th><th>góc (°)</th><th>giới hạn</th></tr>`+NAMES.map(([k,l],i)=>`<tr><td>${l}</td><td class="mono">${f(J.angles[i],1)}</td><td class="mono ${J.near_limit[i]?"warn":"muted"}">${f(J.limits[i][0],0)}…${f(J.limits[i][1],0)}${J.near_limit[i]?" (sát)":""}</td></tr>`).join("")+`</table>`;
  h+=`<div style="margin-top:6px"><b>Đầu tay (giữa 2 ngón)</b> <span class="mono">x ${f(J.ee[0])}  y ${f(J.ee[1])}  z ${f(J.ee[2])}</span></div>`;
  const tc=J.tilt<15?"good":J.tilt<40?"warn":"bad";
  h+=`<div>Trục kẹp nghiêng so với thẳng đứng: <span class="mono ${tc}">${f(J.tilt,1)}°</span> <span class="muted">(0° = chúc thẳng xuống)</span></div>`;
  const fc=J.face_dev<10?"good":J.face_dev<30?"warn":"bad";
  h+=`<div>Hàm kẹp mở <span class="mono">${f(J.jaw_deg,1)}°</span> (1 hàm cố định + 1 hàm di động) · Hai hàm so với mặt hộp: lệch <span class="mono ${fc}">${f(J.face_dev,1)}°</span> <span class="muted">(0° = ôm 2 MẶT đối diện, 45° = kẹp vào GÓC)</span></div>`;
  h+=`<div>hộp cách đầu tay <span class="mono">${f(J.box_off_dist_mm,1)} mm</span> (dọc trục má ${f(J.box_off_jaw_mm,1)} mm)</div>`;
  const T=J.touch,list=(k)=>Object.entries(T).filter(([b,v])=>v.includes(k)).map(([b])=>b);
  h+=`<div style="margin-top:6px"><b>Va chạm</b> `+(["table","bowl","box"].map(k=>{const l=list(k);return l.length?tag(k,{table:"bàn",bowl:"bát",box:"hộp"}[k]+": "+l.join(", ")):""}).join("")||tag("ok","không chạm gì"))+`</div>`;
  if(J.contacts.length)h+=`<table style="margin-top:4px"><tr><th>cặp</th><th>xuyên (mm)</th></tr>`+J.contacts.map(c=>`<tr><td>${c.a} ↔ ${c.b}</td><td class="mono">${c.pen_mm}</td></tr>`).join("")+`</table>`;
  h+=`<div style="margin-top:6px"><b>Độ hở tới mặt bàn</b> <span class="muted">(âm = đã xuyên)</span></div><table>`+Object.entries(J.clearance_mm).map(([b,v])=>`<tr><td>${b}</td><td class="mono ${v<5?"bad":v<20?"warn":""}">${v} mm</td></tr>`).join("")+`</table>`;
  const wz=J.bowl[2]+0.04;h+=`<div class="muted" style="margin-top:6px">Bát: tâm (${f(J.bowl[0])}, ${f(J.bowl[1])}), miệng z=${f(wz,2)} · hộp: (${f(J.box[0])}, ${f(J.box[1])}, ${f(J.box[2])})</div>`;
  h+=reachInfo(J);
  $("info").innerHTML=h;
}
$("bHome").onclick=()=>{Object.assign(S,HOME);syncUI();go()};
$("bOpen").onclick=()=>{S.grip=GOPEN;syncUI();go()};$("bClose").onclick=()=>{S.grip=GCLOSE;syncUI();go()};
async function doIk(target,{vertical=$("tv").checked,free="1,1,1,1,1",label=""}={}){
  const q=new URLSearchParams({...S,x:target[0],y:target[1],z:target[2],vertical:vertical?1:0,w:$("tw").value,K:$("tk").value,pl:$("pl").value,pe:$("pe").value,pw:$("pw").value,free});
  const r=await (await fetch("/ik?"+q)).json();
  if(r.error){$("ikres").textContent=r.error;return}
  Object.assign(S,{pan:r.angles[0],lift:r.angles[1],elbow:r.angles[2],wflex:r.angles[3],wroll:r.angles[4]});syncUI();
  const t=`${label} sai số ${r.err_mm} mm · nghiêng ${r.tilt}° · ${r.iters} vòng`;$("ikres").textContent=t;$("qres").textContent=t;go();
}
$("bIk").onclick=()=>doIk([$("tx").value/100,$("ty").value/100,$("tz").value/100]);
const setT=(x,y,z)=>{$("tx").value=(x*100).toFixed(1);$("ty").value=(y*100).toFixed(1);$("tz").value=(z*100).toFixed(1)};
document.querySelectorAll("[data-q]").forEach(btn=>btn.onclick=async()=>{
  const k=btn.dataset.q,ee=last?last.ee:[0,0,0.5];
  if(k==="above-box"){setT(S.bx,S.by,0.62);doIk([S.bx,S.by,0.62],{label:"trên hộp:"})}
  else if(k==="grasp"){setT(S.bx,S.by,0.455);doIk([S.bx,S.by,0.455],{label:"vị trí gắp:"})}
  else if(k==="above-bowl"){setT(S.wx,S.wy,0.62);doIk([S.wx,S.wy,0.62],{label:"trên bát:"})}
  else if(k==="place"){setT(S.wx,S.wy,0.48);doIk([S.wx,S.wy,0.48],{label:"vị trí thả:"})}
  else if(k==="pan-box"){doIk([S.bx,S.by,ee[2]],{free:"1,0,0,0,0",vertical:false,label:"pan về hộp:"})}
  else if(k==="pan-bowl"){doIk([S.wx,S.wy,ee[2]],{free:"1,0,0,0,0",vertical:false,label:"pan về bát:"})}
  else if(k==="vertical"){setT(ee[0],ee[1],ee[2]);$("tv").checked=true;doIk(ee,{vertical:true,label:"thẳng đứng:"})}
  else if(k==="roll"){const r=await (await fetch("/roll?"+new URLSearchParams(S))).json();S.wroll=r.wroll;syncUI();$("qres").textContent=`wroll = ${r.wroll}° (lệch mặt hộp còn ${r.dev}°)`;go()}
});
$("bPol").onclick=async()=>{
  const r=await (await fetch(`/policy?seed=${$("ps").value}&phase=${$("pp").value}`)).json();
  if(r.error){$("polres").textContent=r.error;return}
  Object.assign(S,{pan:r.angles[0],lift:r.angles[1],elbow:r.angles[2],wflex:r.angles[3],wroll:r.angles[4],grip:r.grip,bx:r.box[0],by:r.box[1]});
  syncUI();$("polres").textContent="đã nạp";go();
};
$("showReach").onchange=()=>{if(last)render(last)};
let HOME={};
let polling=false;
async function pollRun(){
  if(polling)return;polling=true;
  try{const r=await (await fetch("/run-status")).json();$("bRun").disabled=r.status==="running";
    if(r.status==="running"){$("runStatus").textContent="Đang chạy riêng seed lab theo cảnh lúc bấm Chạy…";setTimeout(pollRun,700)}
    else if(r.status==="done"){$("runStatus").textContent=`Đã chạy ${r.steps} bước · ${r.success?"THÀNH CÔNG":"THẤT BẠI"}. Replay là cảnh lúc bấm Chạy; chỉnh tiếp rồi chạy lại để cập nhật.`;$("replayLink").hidden=false;$("replayLink").href=r.replay;$("labReplay").hidden=false;if($("replayFrame").getAttribute("src")!==r.replay)$("replayFrame").src=r.replay}
    else if(r.status==="error"){$("runStatus").textContent="Chạy lỗi: "+r.error}
  }catch(e){$("bRun").disabled=false;$("runStatus").textContent="Không kết nối được máy chủ: "+e.message}finally{polling=false}
}
$("bRun").onclick=async()=>{
  $("bRun").disabled=true;$("runStatus").textContent="Đang gửi cảnh hiện tại…";
  try{const r=await (await fetch("/run",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(S)})).json();if(r.error)throw new Error(r.error);pollRun()}
  catch(e){$("runStatus").textContent="Chạy lỗi: "+e.message;$("bRun").disabled=false}
};
(async()=>{const r=await (await fetch("/meta")).json();LIM=r.limits;HOME=r.home;GOPEN=r.gopen;GCLOSE=r.gclose;Object.assign(S,r.home,r.saved_pose||{});PICK=r.pick_region;PSCAN=r.pick_scan||[];MODEL=await (await fetch("/model")).json();build();syncUI();go();loadReach();pollRun()})();
</script></body></html>"""


def make_handler(lab):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, body, ctype="application/json"):
            data = body if isinstance(body, bytes) else body.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", ctype + "; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            try:
                self.wfile.write(data)
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                pass  # The editor cancels stale preview requests during a drag.

        def do_POST(self):
            if studio_post(self, lab):
                return
            if self.path not in ("/run", "/scene-pose"):
                self.send_error(404)
                return
            # Only accept same-origin JSON; a remote site cannot submit a rollout form.
            if self.headers.get("Content-Type", "").split(";")[0] != "application/json":
                self.send_error(415)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 8192:
                    raise ValueError("Invalid request size")
                pose = json.loads(self.rfile.read(length))
                if self.path == "/run":
                    self._send(json.dumps(lab.runner.start(pose)))
                else:
                    with lab.lock:
                        lab.set_scene(pose)
                        self._send(json.dumps(lab.snapshot()))
            except (ValueError, KeyError, TypeError) as exc:
                self._send(json.dumps({"error": str(exc)}))

        def do_GET(self):
            u = urlparse(self.path)
            q = {k: v[0] for k, v in parse_qs(u.query).items()}
            if studio_get(self, lab, u.path, q):
                return
            fl = lambda k, d=0.0: float(q.get(k, d))
            pose = lambda: dict(pan=fl("pan"), lift=fl("lift"), elbow=fl("elbow"), wflex=fl("wflex"), wroll=fl("wroll"),
                                grip=fl("grip", config.GRIPPER_OPEN_RAD), bx=fl("bx", 0.15), by=fl("by"), byaw=fl("byaw"),
                                wx=fl("wx", -0.15), wy=fl("wy"))
            try:
                if u.path == "/" and (paths.STUDIO_BUILD / "index.html").is_file():
                    self.send_response(302)
                    self.send_header("Location", "/lab/")
                    self.end_headers()
                elif u.path == "/" or u.path == "/legacy":
                    self._send(PAGE, "text/html")
                elif u.path.startswith("/lab/"):
                    root = paths.STUDIO_BUILD.resolve()
                    path = (root / (u.path[5:] or "index.html")).resolve()
                    if path.is_relative_to(root) and path.is_file():
                        self._send(path.read_bytes(), mimetypes.guess_type(path.name)[0] or "application/octet-stream")
                    else:
                        self.send_error(404)
                elif u.path == "/lab-replay-data":
                    path = lab.runner.directory / "replay_lab.json"
                    if path.is_file():
                        self._send(path.read_bytes())
                    else:
                        self._send(json.dumps({"episodes": []}))
                elif u.path == "/run-status":
                    self._send(json.dumps(lab.runner.snapshot()))
                elif u.path == "/replay-lab":
                    path = lab.runner.directory / "replay_lab.html"
                    if path.is_file():
                        self._send(path.read_bytes(), "text/html")
                    else:
                        self.send_error(404, "Chua co replay lab")
                elif u.path == "/meta":
                    h = lab.home
                    self._send(json.dumps({"limits": [[math.degrees(a), math.degrees(b)] for a, b in zip(lab.lo, lab.hi)],
                                           "home": {"pan": h[0], "lift": h[1], "elbow": h[2], "wflex": h[3], "wroll": h[4],
                                                    "grip": config.GRIPPER_OPEN_RAD},
                                           "saved_pose": lab.runner.saved_pose,
                                           "table_half": list(config.TABLE_HALF),
                                           "bowl_geoms": bowl_geom_list(lab.env),
                                           "bowl_origin": lab.env.get_bowl_pos().tolist(),
                                           "gopen": config.GRIPPER_OPEN_RAD, "gclose": config.GRIPPER_CLOSED_RAD,
                                           "pick_region": dict(config.PICK_REGION, pan=list(config.PICK_PAN_XY)),
                                           "pick_scan": lab.pick_scan(),
                                           "unreachable_region": unreachable_region(),
                                           "glim": [float(lab.m.jnt_range[lab.m.jnt_qposadr.tolist().index(lab.fl)][0]) if False else -10, 100]}))
                elif u.path == "/model":
                    if q.get("exact") == "1":
                        with lab.lock:
                            if lab.exact_model is None:
                                kin, mesh = arm_model_export(lab.env, exact=True)
                                lab.exact_model = {"kin": kin, "mesh": mesh}
                        self._send(json.dumps(lab.exact_model))
                    else:
                        self._send(json.dumps({"kin": lab.kin, "mesh": lab.mesh}))
                elif u.path == "/pose":
                    with lab.lock:
                        lab.set_pose(pose())
                        self._send(json.dumps(lab.snapshot()))
                elif u.path == "/ik":
                    with lab.lock:
                        lab.set_pose(pose())
                        free = [int(v) for v in q.get("free", "1,1,1,1,1").split(",")]
                        r = lab.solve_ik(pose(), [fl("x"), fl("y"), fl("z")], int(fl("vertical")) == 1, fl("w", 0.05), fl("K", 0.1),
                                         [0, fl("pl", 10), fl("pe", -15), fl("pw", 60), 0], [0, 1, 1, 1, 0], free=free)
                        self._send(json.dumps(r))
                elif u.path == "/reach":
                    self._send(json.dumps({"status": lab.reach_status, "reach": lab.reach}))
                elif u.path == "/roll":
                    with lab.lock:
                        lab.set_pose(pose())
                        self._send(json.dumps(lab.best_roll(pose())))
                elif u.path == "/policy":
                    with lab.lock:
                        self._send(json.dumps(lab.policy_pose(int(fl("seed")), int(fl("phase")))))
                else:
                    self.send_error(404)
            except Exception as e:      # báo lỗi ra trang thay vì làm sập server
                self._send(json.dumps({"error": f"{type(e).__name__}: {e}"}))
    return H


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="Phòng thí nghiệm khớp: kéo thanh trượt chỉnh tay robot")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-open", action="store_true", help="không tự mở trình duyệt")
    ap.add_argument("--recompute-reach", action="store_true", help="tính lại vùng với tới (mặc định dùng file results/reach_envelope.json nếu có)")
    args = ap.parse_args()
    lab = Lab()
    threading.Thread(target=lab.ensure_reach, kwargs={"force": args.recompute_reach}, daemon=True).start()
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(lab))
    url = f"http://127.0.0.1:{args.port}"
    print(f"Phòng thí nghiệm khớp đang chạy tại {url}  (Ctrl+C để dừng)")
    if not args.no_open:
        webbrowser.open(url)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()


if __name__ == "__main__":
    main()
