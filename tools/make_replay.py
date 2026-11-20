"""Ghi nhiều ván chạy ra file HTML để xem lại từng bước trên trình duyệt (replay).

Cách dùng (đứng ở thư mục chứa sim_engine, đã kích hoạt .venv):
    python -m sim_engine.make_replay                    # seed lấy từ docs/replay_seeds.json (xem select_seeds), mặc định 0-29
    python -m sim_engine.make_replay --seeds 0,2,3      # chỉ vài seed (file nhỏ hơn, ghi nhanh hơn)
    python -m sim_engine.make_replay --policy sim_engine.policies.my_policy:MyPolicy --out docs/replay_my.html

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
import os
import time

import mujoco
import numpy as np

from .. import config
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


def arm_model_export(env):
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
            nv, nf = decimate(v, m.mesh_face[fa:fa + fn], 0.0015 if small else MESH_CELL)
            T.append(nf + sum(len(x) for x in V))
            material_id = m.geom_matid[g]
            rgba = m.mat_rgba[material_id] if material_id >= 0 else m.geom_rgba[g]
            colors.append([len(nf), [round(float(channel) * 255) for channel in rgba[:3]], (2 if "module" in (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_MESH, mid) or "") else 1) if small else 0])   # phần tử 3: 1 = mesh camera (không đổi màu theo va chạm), 2 = phần ống kính (được ưu tiên vẽ trước mặt hộp camera)
            V.append(nv)
        if V:
            meshes[n] = {"v": [round(float(x), 4) for x in np.concatenate(V).ravel()],
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
        return {
            "ik": ik_deg,                                # 5 góc khớp đích do IK tính (độ), None nếu phase chờ
            "ie": ik_err,                                # sai số IK (m): tay dự đoán so với đích
            "jq": self.joint_deg(),                      # 5 góc khớp thật (độ)
            "tb": self.table_contacts(),                 # khúc tay đang chạm bàn
            "t": step,                                   # tổng số bước đã chạy
            "ph": ph, "phn": phn, "ps": ps,              # phase vừa chạy, phase kế, bước trong phase
            "ee": r3(info["ee_pos"]),                    # điểm giữa hai ngón kẹp
            "box": r3(info["box_pos"]),
            "bq": [round(float(x), 4) for x in self.data.xquat[self.env._box_body]],   # hướng của hộp (quaternion w,x,y,z): để vẽ hộp bị nghiêng/lật
            "tg": r3(self.target) if self.target is not None else None,   # đích của phase
            "th": self.threshold if self.target is not None else None,    # ngưỡng để sang phase kế
            "fo": self.finger_open(),                    # góc hàm kẹp di động (rad): lớn = mở
            "gr": bool(self.env._grasped),               # hộp đang dính vào tay
            "ag": round(float(action_grip), 2),          # lệnh kẹp: +1 mở, -1 đóng
            "rc": getattr(self.policy, "_roll_stage", None),             # phục hồi sau khi chạm bát: lift / turn / None
            "rf": getattr(self.policy, "_roll_flips", 0),               # số lần đã đổi roll vì chạm bát
            "ba": bool(getattr(self.policy, "_pred_changed", False)),   # roll đã được DỰ ĐOÁN đổi để tránh bát
            "ex": getattr(self.policy, "last_exit", None),   # lý do phase kết thúc ở bước này: arrived / stall / timeout
            "hd": round(float(info["box_to_bowl_dist"]), 3),   # khoảng cách ngang hộp-bát
            "ok": bool(info["success"]),
            "bs": round(float(info["box_speed"]), 3),    # tốc độ hộp (m/s): chỉ tính thành công khi hộp đã nằm yên
            "arm": [r3(self.data.xpos[i]) for i in self.arm_ids],
            "ct": self.box_contacts(),
            "bc": self.bowl_contacts(),                  # khúc tay / ngón đang chạm bát
        }

    def run(self, seed, options=None):
        env, policy = self.env, self.policy
        obs, info = env.reset(seed=seed, options=options)
        policy.reset()
        self.target = None
        self.threshold = None
        frames = [self.frame(0, 0, 0, 0, info, 1.0)]     # khung 0: trước khi chạy bước nào
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
        return {
            "seed": seed,
            "bowl": r3(self.env.get_bowl_pos()),               # vị trí bát của ván này (có thể khác nhau giữa các seed)
            "bowl_geoms": bowl_geom_list(self.env),
            "success": bool(info["success"]),
            "steps": len(frames) - 1,
            "frames": frames,
        }


def main():
    parser = argparse.ArgumentParser(description="Ghi các ván chạy ra file HTML để xem lại")
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parser.add_argument("--seeds", default=None, help="ví dụ 0-29, hoặc 2,3, hoặc 0-4,10. Mặc định: đọc --seed-file, "
                        "nếu không có thì 0-29")
    parser.add_argument("--seed-file", default=os.path.join(here, "docs", "replay_seeds.json"),
                        help="file do select_seeds ghi: danh sách seed đã chọn kèm nhãn (hỏng, timeout, chậm...)")
    parser.add_argument("--policy", default="sim_engine.policies.scripted_policy:ScriptedPolicy")
    parser.add_argument("--box-range", type=float, default=None, help="hộp ngẫu nhiên trong ± giá trị này (mét), mặc định theo config")
    parser.add_argument("--bowl-range", type=float, default=None, help="bát ngẫu nhiên trong ± giá trị này (mét), mặc định theo config")
    parser.add_argument("--out", default=None, help="mặc định docs/replay.html")
    parser.add_argument("--workers", type=int, default=default_workers(), help="số tiến trình chạy song song (mặc định: số lõi - 1; 1 = tuần tự)")
    args = parser.parse_args()
    if args.out is None:
        args.out = os.path.join(here, "docs", "replay.html")

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
    prev_path = os.path.join(here, "docs", "replay_prev_steps.json")
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

    # Thông tin cố định của cảnh để HTML vẽ đúng tỷ lệ
    scene = {
        "bowl": r3(env.get_bowl_pos()),
        "bowl_radius": config.BOWL_INNER_RADIUS,   # ngưỡng ngang để vào bát = lòng bát (tâm hộp phải nằm trong vòng này)
        "bowl_height": config.BOWL_HEIGHT,
        "bowl_inner": config.BOWL_INNER_RADIUS,      # lòng bát: tâm hộp phải cách tâm bát ít hơn mức này
        "settled_speed": config.SETTLED_SPEED,       # hộp phải chậm hơn mức này (m/s) mới tính là nằm yên
        "box_half": config.BOX_HALF,
        "unreachable_region": {"pan": list(config.PICK_PAN_XY), "angle_deg": config.PAN_LIMIT_DEG + config.PAN_UNREACHABLE_MARGIN_DEG, "radius_m": config.PICK_UNREACHABLE_RADIUS},   # giới hạn đỏ dùng chung với policy
        "bowl_z_margin": config.BOX_IN_BOWL_MARGIN,   # tâm hộp phải thấp hơn miệng bát chừng này mới tính trong bát
        "table": list(config.TABLE_HALF),            # nửa kích thước bàn (x, y)
        "max_steps": config.MAX_EPISODE_STEPS,
        "cam_tilt_deg": config.WRIST_CAM_TILT_DEG,     # độ gập hiện tại của giá đỡ camera cổ tay (để thanh "Gập cam" trong replay tính độ gập thêm)
        "step_dt": config.PHYSICS_SUBSTEPS * config.PHYSICS_TIMESTEP,   # giây mô phỏng thật mỗi bước điều khiển (để phát replay đúng thời gian thực)
        "joints": config.JOINT_NAMES[:5],
        # Hình học THẬT của bát (đọc từ model): đáy hình trụ + 8 tấm tường hình hộp. Toạ độ tâm tính theo thế giới.
        "bowl_geoms": bowl_geom_list(env),
        "arm_bodies": ARM_BODIES,            # đoạn k nằm giữa khớp k và khớp k+1 (xem JS: đánh số khớp 1-5)
        "policy": args.policy,
    }
    scene["kin"], scene["arm_mesh"] = arm_model_export(env)
    scene["cams"] = cameras_export(env)           # camera trong model, để HTML dựng lại góc nhìn của chúng     # cánh tay THẬT: cây khung + lưới tam giác để HTML tự vẽ
    # Vùng với tới của cánh tay (do joint_lab tính và lưu); nếu chưa có thì bỏ qua và nhắc cách tạo
    # Vùng GẮP ĐƯỢC THẬT: hình quạt trong config.PICK_REGION và các điểm đã quét (tools/scan_pick_region.py -> docs/pick_scan.json)
    scene["pick_region"] = dict(config.PICK_REGION, pan=list(config.PICK_PAN_XY), bowl_clearance=config.BOX_BOWL_CLEARANCE)
    scan_points = []
    for filename in ("pick_scan.json", "pick_scan_inner.json", "pick_scan_outer.json"):
        scan_path = os.path.join(here, "docs", filename)
        if os.path.isfile(scan_path):
            with open(scan_path, encoding="utf-8") as f:
                scan_points.extend([[*point["xy"], int(point["ok"])] for point in json.load(f)["points"]])
    if scan_points:
        scene["pick_scan"] = scan_points
    reach_path = os.path.join(here, "docs", "reach_envelope.json")
    if os.path.exists(reach_path):
        with open(reach_path, encoding="utf-8") as f:
            scene["reach"] = json.load(f)
    else:
        print("Chưa có docs/reach_envelope.json nên replay chưa vẽ vùng với tới. Chạy python -m sim_engine.joint_lab một lần để tính.")
    payload = json.dumps({"scene": scene, "episodes": episodes}, separators=(",", ":"), ensure_ascii=False)

    template_path = os.path.join(here, "docs", "replay_template.html")
    with open(template_path, encoding="utf-8") as f:
        template = f.read()
    asset_dir = os.path.join(here, "docs", "ui")
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
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(template.replace(marker, payload))

    ok = sum(e["success"] for e in episodes)
    print(f"\nĐã ghi {args.out} ({os.path.getsize(args.out) / 1e6:.1f} MB), {ok}/{len(episodes)} ván thành công.")
    print("Mở file đó bằng trình duyệt (bấm đúp).")
    env.close()


if __name__ == "__main__":
    main()
