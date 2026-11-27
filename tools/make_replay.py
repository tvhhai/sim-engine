"""Ghi nhiều ván chạy ra file HTML để xem lại từng bước trên trình duyệt (replay).

Cách dùng (đứng ở thư mục chứa sim_engine, đã kích hoạt .venv):
    python -m sim_engine.tools.make_replay                    # seed lấy từ results/replay_seeds.json (xem select_seeds), mặc định 0-29
    python -m sim_engine.tools.make_replay --seeds 0,2,3      # chỉ vài seed (file nhỏ hơn, ghi nhanh hơn)
    python -m sim_engine.tools.make_replay --rebuild          # chỉ dựng lại HTML từ results/replay.json sau khi sửa giao diện
    python -m sim_engine.tools.make_replay --policy sim_engine.policies.my_policy:MyPolicy --out results/replay_my.html

Kết quả: results/replay.html (bản canvas) và results/replay.json (cùng dữ liệu, Studio React đọc file này).
Sau đó mở file HTML bằng trình duyệt (bấm đúp). File chạy offline, không cần server, không cần mạng.
Mỗi lần bạn sửa policy, chạy lại lệnh này để có dữ liệu mới.

Đây là bản xem lại các con số thật mà mô phỏng đã tính, vẽ lại thành hình 2D đơn giản.
Nó không chạy vật lý. Để xem va chạm thật, dùng step_policy hoặc watch_policy (cửa sổ 3D).
"""

import argparse
import contextlib
import importlib
import io
import json
import math
import os
import time

import mujoco
import numpy as np

from .. import config, paths
from ..env import SO101PickPlaceEnv

# Các khớp nối của cánh tay, theo thứ tự từ đế tới cổ tay (để vẽ "xương" cánh tay)
ARM_BODIES = ["base", "shoulder", "upper_arm", "lower_arm", "wrist", "gripper"]
JAW_BODY = "moving_jaw_so101_v1"          # hàm kẹp di động (khớp "gripper"); hàm cố định thuộc body "gripper"
MESH_CELL = 0.005                         # ô lưới (m) khi gộp đỉnh để giảm số tam giác; nhỏ hơn = chi tiết hơn, file lớn hơn


def _quat_to_mat(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def decimate(verts, faces, cell):
    """Giảm số tam giác bằng cách gộp các đỉnh cùng một ô lưới (vertex clustering). Trả (đỉnh, mặt) đã gọn."""
    key = np.floor(verts / cell).astype(np.int64)
    _, first, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
    inv = inv.reshape(-1)
    newv = np.zeros((len(first), 3))
    np.add.at(newv, inv, verts)
    newv /= np.bincount(inv, minlength=len(first))[:, None]
    f = inv[faces]
    f = f[(f[:, 0] != f[:, 1]) & (f[:, 1] != f[:, 2]) & (f[:, 0] != f[:, 2])]
    f = np.unique(np.sort(f, axis=1), axis=0) if len(f) else f
    return newv, f


def cameras_export(env):
    """Các camera có sẵn trong model (webcam, arm_cam): vị trí, hướng, fovy; camera gắn vào body thì lưu tư thế cục bộ."""
    m, d = env.model, env.data
    mujoco.mj_forward(m, d)
    out = {}
    for c in range(m.ncam):
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_CAMERA, c)
        b = int(m.cam_bodyid[c])
        if b == 0:     # gắn vào thế giới: lấy tư thế world
            pos, R = d.cam_xpos[c], d.cam_xmat[c].reshape(3, 3)
            body = None
        else:          # gắn vào một body: tư thế cục bộ, HTML tự cộng động học thuận
            pos, R = m.cam_pos[c], _quat_to_mat(m.cam_quat[c])
            body = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b)
        out[name] = {"body": body, "pos": [round(float(x), 5) for x in pos],
                     "R": [[round(float(x), 5) for x in row] for row in R], "fovy": float(m.cam_fovy[c])}
    return out


def arm_model_export(env, exact=False):
    """Cánh tay THẬT để HTML tự vẽ: cây khung (vị trí/hướng cục bộ + trục khớp) và lưới tam giác đã giản lược của từng
    body, toạ độ trong khung body. Mỗi khung hình chỉ cần góc khớp (jq, fo) rồi HTML tự tính động học thuận.
    """
    m = env.model
    names = ARM_BODIES + [JAW_BODY, "wrist_cam_arm"]      # + giá đỡ camera gập được (con của gripper)
    kin, meshes = [], {}
    for n in names:
        b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)
        par = int(m.body_parentid[b])
        jid = int(m.body_jntadr[b]) if m.body_jntnum[b] else -1
        kin.append({
            "name": n,
            "parent": mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, par) if par > 0 else None,
            "pos": [round(float(x), 6) for x in m.body_pos[b]],
            "quat": [round(float(x), 6) for x in m.body_quat[b]],
            "joint": mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, jid) if jid >= 0 else None,
        })
        V, T, colors = [], [], []
        for g in range(m.ngeom):
            if m.geom_bodyid[g] != b or m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH or m.geom_group[g] != 2:
                continue
            mid = m.geom_dataid[g]
            va, vn = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
            fa, fn = m.mesh_faceadr[mid], m.mesh_facenum[mid]
            v = m.mesh_vert[va:va + vn].astype(float)
            v = v @ _quat_to_mat(m.geom_quat[g]).T + m.geom_pos[g]       # khung body
            small = (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_MESH, mid) or "").startswith("wrist_cam")   # vật nhỏ (camera): giữ chi tiết
            nv, nf = (v, m.mesh_face[fa:fa + fn]) if exact else decimate(v, m.mesh_face[fa:fa + fn], 0.0015 if small else MESH_CELL)
            T.append(nf + sum(len(x) for x in V))
            material_id = m.geom_matid[g]
            rgba = m.mat_rgba[material_id] if material_id >= 0 else m.geom_rgba[g]
            colors.append([len(nf), [round(float(channel) * 255) for channel in rgba[:3]], (2 if "module" in (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_MESH, mid) or "") else 1) if small else 0])   # phần tử 3: 1 = mesh camera (không đổi màu theo va chạm), 2 = phần ống kính (được ưu tiên vẽ trước mặt hộp camera)
            V.append(nv)
        if V:
            meshes[n] = {"v": [round(float(x), 7 if exact else 4) for x in np.concatenate(V).ravel()],
                         "f": np.concatenate(T).ravel().astype(int).tolist(), "colors": colors}
    return kin, meshes


_W = {}   # trạng thái của tiến trình con: {"rec": Recorder}


def _init_worker(policy_spec, box_range, bowl_range):
    module_name, class_name = policy_spec.split(":")
    policy_cls = getattr(importlib.import_module(module_name), class_name)
    env = SO101PickPlaceEnv(render_images=False, end_on_success=False, box_range=box_range, bowl_range=bowl_range)
    _W["rec"] = Recorder(env, policy_cls(env))


def _record_task(task):
    seed, options = task
    with contextlib.redirect_stdout(io.StringIO()):
        return _W["rec"].run(seed, options=options)


def parallel_map(fn, tasks, policy_spec, box_range=None, bowl_range=None, workers=1, progress=None):
    """Chạy fn(task) cho từng task, mỗi tiến trình con có Recorder riêng (môi trường độc lập). Giữ nguyên thứ tự kết quả.

    Mỗi ván tách biệt hoàn toàn (env.reset xoá mọi trạng thái) nên chạy song song cho CÙNG kết quả như chạy tuần tự.
    workers <= 1: chạy ngay trong tiến trình này (dễ gỡ lỗi)."""
    if workers <= 1 or len(tasks) <= 1:
        _init_worker(policy_spec, box_range, bowl_range)
        out = []
        for k, t in enumerate(tasks):
            out.append(fn(t))
            if progress:
                progress(k + 1, len(tasks))
        return out
    import multiprocessing as mp
    out = []
    with mp.get_context("spawn").Pool(min(workers, len(tasks)), initializer=_init_worker, initargs=(policy_spec, box_range, bowl_range)) as pool:
        for k, r in enumerate(pool.imap(fn, tasks, chunksize=1)):
            out.append(r)
            if progress:
                progress(k + 1, len(tasks))
    return out


def default_workers():
    return max(1, (os.cpu_count() or 2) - 1)


def parse_seeds(text):
    """Đổi chuỗi như '0-9' hoặc '2,3' hoặc '0-4,10' thành danh sách seed."""
    seeds = []
    for part in text.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            seeds.extend(range(int(a), int(b) + 1))
        elif part:
            seeds.append(int(part))
    return seeds


def bowl_geom_list(env):
    """Hình học thật của bát ở vị trí HIỆN TẠI (đáy trụ + 24 tấm tường hẹp xếp vòng tròn), toạ độ thế giới."""
    body = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "bowl")
    return [
        {
            "name": mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, g),
            "type": "cylinder" if env.model.geom_type[g] == mujoco.mjtGeom.mjGEOM_CYLINDER else "box",
            "pos": r3(env.data.geom_xpos[g]),
            "size": [round(float(x), 4) for x in env.model.geom_size[g]],
            "yaw": round(float(np.arctan2(env.data.geom_xmat[g][3], env.data.geom_xmat[g][0])), 4),
        }
        for g in range(env.model.ngeom) if env.model.geom_bodyid[g] == body
    ]


def r3(v):
    """Làm tròn vector về 3 chữ số thập phân (mét), đổi sang list để ghi JSON."""
    return [round(float(x), 3) for x in v]


class Recorder:
    """Chạy 1 ván và ghi lại từng bước thành list các dict."""

    def __init__(self, env, policy):
        self.env = env
        self.policy = policy
        self.model = env.model
        self.data = env.data
        self.target = None
        self.threshold = None
        self.arm_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, n) for n in ARM_BODIES]
        self.skip_bodies = {
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "table"),
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "world"),
        }
        bid = lambda n: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, n)
        geoms = lambda b: [g for g in range(self.model.ngeom) if self.model.geom_bodyid[g] == b]
        self._cam_pairs = [(a, w) for a in geoms(bid("wrist_cam_arm")) for n in ("base", "shoulder", "upper_arm", "table") for w in geoms(bid(n))]

    def cam_contacts(self):
        """Tên các khâu đang chạm camera cổ tay (vai, cánh tay trên, đế, bàn, bát, hộp...), không tính gá camera, cổ tay và hàm kẹp."""
        if not hasattr(self, "_cam_gids"):
            bid = lambda n: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, n)
            self._cam_body = bid("wrist_cam_arm")
            self._cam_own = {self._cam_body, bid("gripper"), bid("wrist"), bid("moving_jaw_so101_v1")}
        names = set()
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            b1, b2 = self.model.geom_bodyid[c.geom1], self.model.geom_bodyid[c.geom2]
            if self._cam_body in (b1, b2):
                other = b2 if b1 == self._cam_body else b1
                if other not in self._cam_own:
                    names.add(mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, other))
        return sorted(n for n in names if n)

    def cam_clearance_mm(self):
        """Khoảng cách nhỏ nhất (mm) từ camera cổ tay tới đế, vai, cánh tay trên và mặt bàn; âm = camera lún vào. Camera có va chạm (geom wrist_cam_*_col)
        nên chỉ lún vài mm khi bị ép; đây là phép đo hình học để cảnh báo (xem nhãn "CAMERA" ở select_seeds)."""
        fromto = np.zeros(6)
        return min(mujoco.mj_geomDistance(self.model, self.data, a, w, 0.1, fromto) for a, w in self._cam_pairs) * 1000.0

    def box_contacts(self):
        """Tên các body (trừ bàn) đang chạm hộp ở thời điểm hiện tại."""
        box_b = self.env._box_body
        names = set()
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            b1, b2 = self.model.geom_bodyid[c.geom1], self.model.geom_bodyid[c.geom2]
            if box_b in (b1, b2):
                other = b2 if b1 == box_b else b1
                if other not in self.skip_bodies:
                    names.add(mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, other))
        return sorted(n for n in names if n)

    def table_contacts(self):
        """Tên các khúc tay đang chạm bàn hoặc sàn (không tính hộp và bát)."""
        env_bodies = {
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, n) for n in ("box", "bowl")
        }
        names = set()
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            b1, b2 = self.model.geom_bodyid[c.geom1], self.model.geom_bodyid[c.geom2]
            if b1 in self.skip_bodies or b2 in self.skip_bodies:
                other = b2 if b1 in self.skip_bodies else b1
                if other not in self.skip_bodies and other not in env_bodies:
                    names.add(mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, other))
        return sorted(n for n in names if n)

    def table_points(self):
        """Toạ độ (x, y, z) các điểm tay chạm bàn hoặc sàn, để Studio tô tím lên mặt bàn."""
        env_bodies = {mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, n) for n in ("box", "bowl")}
        pts = []
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            b1, b2 = self.model.geom_bodyid[c.geom1], self.model.geom_bodyid[c.geom2]
            if b1 in self.skip_bodies or b2 in self.skip_bodies:
                other = b2 if b1 in self.skip_bodies else b1
                if other not in self.skip_bodies and other not in env_bodies:
                    pts.append(r3(c.pos))
        return pts[:12]

    def bowl_contacts(self):
        """Tên các khúc tay / ngón kẹp đang chạm bát (không tính bàn và hộp)."""
        bowl_b = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "bowl")
        names = set()
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            b1, b2 = self.model.geom_bodyid[c.geom1], self.model.geom_bodyid[c.geom2]
            if bowl_b in (b1, b2):
                other = b2 if b1 == bowl_b else b1
                if other not in self.skip_bodies and other != self.env._box_body:
                    names.add(mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, other))
        return sorted(n for n in names if n)

    def joint_deg(self):
        """Góc thật của 5 khớp tay (độ)."""
        return [round(float(np.degrees(self.data.qpos[self.model.jnt_qposadr[j]])), 1) for j in self.policy._joint_ids]

    def ik_info(self):
        """Kết quả IK của bước vừa rồi: 5 góc khớp đích (độ) và sai số (m) giữa tay dự đoán và đích.

        IK chạy trên bản sao dữ liệu, chỉ tính hình học, không biết va chạm. Nên sai số nhỏ
        nghĩa là "IK tin rằng tay sẽ tới đích", chưa chắc tay thật tới được.
        """
        if self.target is None or self.policy._current_joint_targets is None:
            return None, None
        deg = [round(float(np.degrees(a)), 1) for a in self.policy._current_joint_targets]
        shadow = self.policy._ik_data.site_xpos[self.policy._ee_site]
        return deg, round(float(np.linalg.norm(shadow - self.target)), 4)

    def finger_open(self):
        adr = self.model.jnt_qposadr[self.env._gripper_jnt]
        return round(float(self.data.qpos[adr]), 4)

    def frame(self, step, ph, phn, ps, info, action_grip):
        ik_deg, ik_err = self.ik_info()
        result = {
            "ik": ik_deg,                                # 5 góc khớp đích do IK tính (độ), None nếu phase chờ
            "ie": ik_err,                                # sai số IK (m): tay dự đoán so với đích
            "jq": self.joint_deg(),                      # 5 góc khớp thật (độ)
            "tp": self.table_points(),                   # điểm chạm bàn (toạ độ thế giới)
            "tb": self.table_contacts(),                 # khúc tay đang chạm bàn
            "cc": self.cam_contacts(),                   # khâu đang chạm camera cổ tay (vai, cánh tay trên, bàn...)
            "t": step,                                   # tổng số bước đã chạy
            "ph": ph, "phn": phn, "ps": ps,              # phase vừa chạy, phase kế, bước trong phase
            "ee": r3(info["ee_pos"]),                    # điểm giữa hai ngón kẹp
            "box": r3(info["box_pos"]),
            "bq": [round(float(x), 4) for x in self.data.xquat[self.env._box_body]],   # hướng của hộp (quaternion w,x,y,z): để vẽ hộp bị nghiêng/lật
            "tg": r3(self.target) if self.target is not None else None,   # đích của phase
            "th": self.threshold if self.target is not None else None,    # ngưỡng để sang phase kế
            "fo": self.finger_open(),                    # góc hàm kẹp di động (rad): lớn = mở
            "gr": bool(self.env._grasped),               # cả hai hàm đang chạm hộp (kẹp thật)
            "ag": round(float(action_grip), 2),          # lệnh kẹp: +1 mở, -1 đóng
            "rc": getattr(self.policy, "_roll_stage", None),             # phục hồi sau khi chạm bát: lift / turn / None
            "rf": getattr(self.policy, "_roll_flips", 0),               # số lần đã đổi roll vì chạm bát
            "ba": bool(getattr(self.policy, "_pred_changed", False)),   # roll đã được DỰ ĐOÁN đổi để tránh bát
            "ex": getattr(self.policy, "last_exit", None),   # lý do phase kết thúc ở bước này: arrived / stall / timeout
            "hd": round(float(info["box_to_bowl_dist"]), 3),   # khoảng cách ngang hộp-bát
            "ok": bool(info["success"]),
            "cd": round(self.cam_clearance_mm(), 1),     # khoảng cách camera tới bàn/vai/cánh tay (mm), âm = xuyên
            "bs": round(float(info["box_speed"]), 3),    # tốc độ hộp (m/s): chỉ tính thành công khi hộp đã nằm yên
            "arm": [r3(self.data.xpos[i]) for i in self.arm_ids],
            "ct": self.box_contacts(),
            "bc": self.bowl_contacts(),                  # khúc tay / ngón đang chạm bát
        }
        if self.env.lab_objects:
            result["objects"] = [dict(obj, pos=self.data.xpos[self.model.body(obj["body"]).id].tolist(),
                                      quat=self.data.xquat[self.model.body(obj["body"]).id].tolist())
                                 for obj in self.env.lab_objects]
            result["pose"] = {name: {"p": self.data.xpos[self.model.body(name).id].tolist(),
                                      "R": self.data.xmat[self.model.body(name).id].tolist()}
                              for name in ARM_BODIES + [JAW_BODY, "wrist_cam_arm"]}
        return result

    def run(self, seed, options=None, *, continue_scene=False):
        env, policy = self.env, self.policy
        if continue_scene:
            env._step_count = 0
            obs, info = env._get_obs(), env._get_info()
        else:
            obs, info = env.reset(seed=seed, options=options)
        policy.reset()
        self.target = None
        self.threshold = None
        frames = [self.frame(0, 0, 0, 0, info, 1.0)]     # khung 0: trước khi chạy bước nào
        blocked = (options or {}).get("blocked_reason")
        if blocked:
            return {"seed": seed, "bowl": r3(env.get_bowl_pos()), "bowl_geoms": bowl_geom_list(env),
                    "success": False, "steps": 0, "frames": frames, "blocked_reason": blocked}
        ps = 0
        for step in range(1, config.MAX_EPISODE_STEPS + 1):
            before = policy._phase
            self.target = None
            with contextlib.redirect_stdout(io.StringIO()):   # chặn print debug của policy
                action = policy.select_action({"state": obs["state"], "images": {}, "task": config.DEFAULT_TASK})
            after = policy._phase
            self.target, self.threshold = getattr(policy, "motion_target", None) or (None, None)
            obs, reward, terminated, truncated, info = env.step(action)
            ps = policy._phase_step if after == before else ps + 1
            frames.append(self.frame(step, before, after, ps, info, action[5]))
            if after != before:
                ps = 0
            if terminated or truncated or after > 10:
                break
        if not (terminated or truncated):                                    # policy xong mà hộp còn lắc trong bát: chờ nằm yên (có ghi khung)
            n_wait = 0
            while n_wait < config.SETTLE_WAIT_STEPS and env.box_settling(info):
                obs, reward, terminated, truncated, info = env.step(action)
                n_wait += 1
                step += 1
                frames.append(self.frame(step, after, after, 0, info, action[5]))
        return {
            "seed": seed,
            "bowl": r3(self.env.get_bowl_pos()),               # vị trí bát của ván này (có thể khác nhau giữa các seed)
            "bowl_geoms": bowl_geom_list(self.env),
            "success": bool(info["success"]),
            "steps": len(frames) - 1,
            "frames": frames,
        }


SCAN_FILES = ("pick_scan.json", "pick_scan_inner.json", "pick_scan_outer.json")


def pick_scan_points():
    """Mọi điểm đã quét gắp thật (tools/scan_pick_region.py, lưới thường + vòng trong + vòng ngoài): [[x, y, 1 nếu gắp được], ...].
    Replay canvas, Studio Lab và Studio Replay cùng đọc qua hàm này."""
    points = []
    for filename in SCAN_FILES:
        path = paths.result(filename)
        if path.is_file():
            with open(path, encoding="utf-8") as f:
                points.extend([[*point["xy"], int(point["ok"])] for point in json.load(f)["points"]])
    return points


def unreachable_region():
    """Giới hạn dừng của policy (vùng đỏ): ngoài góc khớp 1 ±(giới hạn + biên), quá xa, quá sát đế thì về home ngay."""
    return {"pan": list(config.PICK_PAN_XY), "angle_deg": config.PAN_LIMIT_DEG + config.PAN_UNREACHABLE_MARGIN_DEG,
            "radius_m": config.PICK_UNREACHABLE_RADIUS, "inner_radius_m": config.PICK_UNREACHABLE_INNER_RADIUS}


def pick_zone(x, y):
    """Vùng của một vị trí hộp: "green" (trong hình quạt gắp được), "red" (ngoài giới hạn dừng) hoặc "mixed".
    Cùng quy tắc với pickZone trong static/replay/replay.js và web/joint-lab/src/zones.ts."""
    region, stop = config.PICK_GREEN, unreachable_region()
    dx, dy = x - config.PICK_PAN_XY[0], y - config.PICK_PAN_XY[1]
    r, angle = math.hypot(dx, dy), math.degrees(math.atan2(dx, dy))
    if r > stop["radius_m"] or r < stop["inner_radius_m"] or abs(angle) > stop["angle_deg"]:
        return "red"
    if (r < region["r_min"] - 1e-9 or r > region["r_max"] + 1e-9
            or angle < region["angle_min_deg"] - 1e-9 or angle > region["angle_max_deg"] + 1e-9):
        return "mixed"
    return "green"


def write_replay(env, episodes, out, policy_name):
    """Write existing episodes without running any seeds or updating batch metadata."""
    # Thông tin cố định của cảnh để HTML vẽ đúng tỷ lệ
    scene = {
        "bowl": r3(env.get_bowl_pos()),
        "bowl_radius": env.bowl_inner,   # ngưỡng ngang để vào bát = lòng bát (tâm hộp phải nằm trong vòng này)
        "bowl_height": env.bowl_height,
        "bowl_color": env.bowl_color,
        "bowl_inner": env.bowl_inner,      # lòng bát: tâm hộp phải cách tâm bát ít hơn mức này
        "settled_speed": config.SETTLED_SPEED,       # hộp phải chậm hơn mức này (m/s) mới tính là nằm yên
        "box_half": config.BOX_HALF,
        "unreachable_region": unreachable_region(),   # giới hạn đỏ dùng chung với policy
        "bowl_z_margin": config.BOX_IN_BOWL_MARGIN,   # tâm hộp phải thấp hơn miệng bát chừng này mới tính trong bát
        "table": list(config.TABLE_HALF),            # nửa kích thước bàn (x, y)
        "max_steps": config.MAX_EPISODE_STEPS,
        "cam_tilt_deg": config.WRIST_CAM_TILT_DEG,     # độ gập hiện tại của giá đỡ camera cổ tay (để thanh "Gập cam" trong replay tính độ gập thêm)
        "step_dt": config.PHYSICS_SUBSTEPS * config.PHYSICS_TIMESTEP,   # giây mô phỏng thật mỗi bước điều khiển (để phát replay đúng thời gian thực)
        "joints": config.JOINT_NAMES[:5],
        # Hình học THẬT của bát (đọc từ model): đáy hình trụ + 8 tấm tường hình hộp. Toạ độ tâm tính theo thế giới.
        "bowl_geoms": bowl_geom_list(env),
        "arm_bodies": ARM_BODIES,            # đoạn k nằm giữa khớp k và khớp k+1 (xem JS: đánh số khớp 1-5)
        "policy": policy_name,
    }
    scene["kin"], scene["arm_mesh"] = arm_model_export(env)
    scene["cams"] = cameras_export(env)           # camera trong model, để HTML dựng lại góc nhìn của chúng     # cánh tay THẬT: cây khung + lưới tam giác để HTML tự vẽ
    # Vùng với tới của cánh tay (do joint_lab tính và lưu); nếu chưa có thì bỏ qua và nhắc cách tạo
    # Vùng GẮP ĐƯỢC THẬT: hình quạt trong config.PICK_GREEN và các điểm đã quét (tools/scan_pick_region.py -> results/pick_scan.json)
    scene["pick_region"] = dict(config.PICK_GREEN, pan=list(config.PICK_PAN_XY), bowl_clearance=config.BOX_BOWL_CLEARANCE)
    scan_points = pick_scan_points()
    if scan_points:
        scene["pick_scan"] = scan_points
    if paths.REACH_ENVELOPE.exists():
        with open(paths.REACH_ENVELOPE, encoding="utf-8") as f:
            scene["reach"] = json.load(f)
    else:
        print("Chưa có results/reach_envelope.json nên replay chưa vẽ vùng với tới. Chạy python -m sim_engine.studio một lần để tính.")
    data = {"scene": scene, "episodes": episodes}
    render_html(data, out)
    return data


def render_html(data, out):
    """Ghép template + css/js của static/replay với dữ liệu replay thành một file HTML mở offline."""
    payload = json.dumps(data, separators=(",", ":"), ensure_ascii=False)
    template_path = paths.REPLAY_ASSETS / "template.html"
    with open(template_path, encoding="utf-8") as f:
        template = f.read()
    asset_dir = paths.REPLAY_ASSETS
    for marker, filename in (
        ("/*__SHARED_CSS__*/", "accessibility.css"),
        ("/*__REPLAY_CSS__*/", "replay.css"),
        ("/*__REPLAY_JS__*/", "replay.js"),
    ):
        with open(os.path.join(asset_dir, filename), encoding="utf-8") as asset:
            template = template.replace(marker, asset.read())
    marker = "/*__DATA__*/null"
    if marker not in template:
        raise SystemExit(f"Không tìm thấy {marker} trong {template_path}")
    with open(out, "w", encoding="utf-8") as f:
        f.write(template.replace(marker, payload))


def rebuild(out):
    """Dựng lại file HTML từ file JSON cùng tên (không chạy lại seed nào): dùng sau khi sửa giao diện static/replay.
    Cập nhật luôn điểm quét và vùng đỏ trong scene theo results/ hiện tại."""
    json_path = os.path.splitext(out)[0] + ".json"
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    data["scene"]["pick_scan"] = pick_scan_points()
    data["scene"]["unreachable_region"] = unreachable_region()
    data["scene"]["pick_region"] = dict(config.PICK_GREEN, pan=list(config.PICK_PAN_XY), bowl_clearance=config.BOX_BOWL_CLEARANCE)
    render_html(data, out)
    with open(json_path, "w", encoding="utf-8") as f:            # Studio đọc file JSON: giữ hai bản cùng dữ liệu
        json.dump(data, f, separators=(",", ":"), ensure_ascii=False)
    print(f"Đã dựng lại {out} từ {json_path} ({len(data['episodes'])} ván, không chạy lại mô phỏng).")



def main():
    parser = argparse.ArgumentParser(description="Ghi các ván chạy ra file HTML để xem lại")
    parser.add_argument("--seeds", default=None, help="ví dụ 0-29, hoặc 2,3, hoặc 0-4,10. Mặc định: đọc --seed-file, "
                        "nếu không có thì 0-29")
    parser.add_argument("--seed-file", default=str(paths.REPLAY_SEEDS),
                        help="file do select_seeds ghi: danh sách seed đã chọn kèm nhãn (hỏng, timeout, chậm...)")
    parser.add_argument("--policy", default=config.DEFAULT_POLICY)
    parser.add_argument("--box-range", type=float, default=None, help="hộp ngẫu nhiên trong ± giá trị này (mét), mặc định theo config")
    parser.add_argument("--bowl-range", type=float, default=None, help="bát ngẫu nhiên trong ± giá trị này (mét), mặc định theo config")
    parser.add_argument("--out", default=None, help="mặc định results/replay.html (kèm results/replay.json cùng dữ liệu)")
    parser.add_argument("--workers", type=int, default=default_workers(), help="số tiến trình chạy song song (mặc định: số lõi - 1; 1 = tuần tự)")
    parser.add_argument("--rebuild", action="store_true",
                        help="chỉ dựng lại HTML từ file .json cùng tên (vài giây), không chạy lại seed: dùng sau khi sửa static/replay")
    args = parser.parse_args()
    if args.out is None:
        args.out = str(paths.REPLAY_HTML)
    if args.rebuild:
        rebuild(args.out)
        return

    module_name, class_name = args.policy.split(":")
    policy_cls = getattr(importlib.import_module(module_name), class_name)
    env = SO101PickPlaceEnv(render_images=False, end_on_success=False,
                            box_range=args.box_range, bowl_range=args.bowl_range)
    policy = policy_cls(env)
    rec = Recorder(env, policy)

    # Danh sách seed: --seeds (nếu có), hoặc file do select_seeds ghi (kèm nhãn), hoặc 0-29
    meta = {}
    if args.seeds is not None:
        seed_list = parse_seeds(args.seeds)
    elif os.path.exists(args.seed_file):
        with open(args.seed_file, encoding="utf-8") as f:
            chosen = json.load(f)
        entries = chosen["selected"] + chosen["rest"]
        seed_list = [(e.get("name") or e["seed"]) for e in entries]      # trường hợp cực biên có tên riêng thay cho seed
        meta = {(e.get("name") or e["seed"]): e for e in entries}
    else:
        seed_list = parse_seeds("0-29")

    # Lần chạy trước (nếu có): để in thêm "(+N)" / "(-N)" bước so với lần đó, biết mình vừa tối ưu hay làm chậm đi
    prev_path = paths.REPLAY_PREV_STEPS
    try:
        with open(prev_path, encoding="utf-8") as f:
            prev = json.load(f)
    except (OSError, ValueError):
        prev = {}
    now = {}

    def delta(key, steps, success):
        p = prev.get(str(key))
        if p is None:
            return "(mới)"
        tag = "" if p["success"] == success else (" ĐÃ SỬA" if success else " MỚI HỎNG")
        d = steps - p["steps"]
        return f"({d:+d}{tag})" if d or tag else f"(=){tag}"

    tasks = []
    for key in seed_list:
        m0 = meta.get(key)
        tasks.append((m0["seed"] if key in meta else key, (m0 or {}).get("options")))
    t_start = time.time()
    results = parallel_map(_record_task, tasks, args.policy, args.box_range, args.bowl_range, args.workers)
    print(f"  chạy {len(tasks)} ván bằng {min(args.workers, len(tasks))} tiến trình trong {time.time() - t_start:.1f} s", flush=True)

    episodes = []
    for key, ep in zip(seed_list, results):
        seed = meta[key]["seed"] if key in meta else key
        m = meta.get(key)
        if m:
            ep["group"] = m["group"]
            ep["tags"] = m["tags"]
            if m.get("name"):
                ep["name"] = m["name"]
        episodes.append(ep)
        label = ep.get("name") or f"seed {seed:3d}"
        print(f"  {label}: {'OK  ' if ep['success'] else 'FAIL'} | {ep['steps']:3d} bước {delta(key, ep['steps'], ep['success'])}")
        now[str(key)] = {"steps": ep["steps"], "success": bool(ep["success"])}

    # Tổng kết so với lần trước (chỉ tính các ván có ở cả hai lần)
    both = [k for k in now if k in prev]
    if both:
        dsum = sum(now[k]["steps"] - prev[k]["steps"] for k in both)
        better = sum(now[k]["steps"] < prev[k]["steps"] for k in both)
        worse = sum(now[k]["steps"] > prev[k]["steps"] for k in both)
        print(f"  So với lần chạy trước ({len(both)} ván chung): tổng {dsum:+d} bước, {better} ván nhanh hơn, {worse} ván chậm hơn, "
              f"{len(both) - better - worse} ván không đổi")
    prev.update(now)                                   # chạy một phần seed thì chỉ ghi đè phần đã chạy
    with open(prev_path, "w", encoding="utf-8") as f:
        json.dump(prev, f, ensure_ascii=False)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    data = write_replay(env, episodes, args.out, args.policy)
    # Cùng dữ liệu dạng JSON cạnh file HTML: Studio React (và mọi công cụ khác) đọc thẳng, không phải bóc từ HTML
    json_out = os.path.splitext(args.out)[0] + ".json"
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(data, f, separators=(",", ":"), ensure_ascii=False)

    ok = sum(e["success"] for e in episodes)
    print(f"\nĐã ghi {args.out} ({os.path.getsize(args.out) / 1e6:.1f} MB) và {json_out}, {ok}/{len(episodes)} ván thành công.")
    print("Mở file đó bằng trình duyệt (bấm đúp).")
    env.close()


if __name__ == "__main__":
    main()
