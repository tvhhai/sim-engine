"""Ghi nhiều ván chạy ra file HTML để xem lại từng bước trên trình duyệt (replay).

Cách dùng (đứng ở thư mục chứa sim_engine, đã kích hoạt .venv):
    python -m sim_engine.make_replay                    # ghi seed 0-29 ra sim_engine/docs/replay.html
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

import mujoco
import numpy as np

from . import config
from .env import SO101PickPlaceEnv

# Các khớp nối của cánh tay, theo thứ tự từ đế tới cổ tay (để vẽ "xương" cánh tay)
ARM_BODIES = ["base_link", "shoulder_link", "upper_arm_link", "forearm_link", "wrist_flex_link", "wrist_roll_link"]


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
        self.finger_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, n) for n in ("left_finger", "right_finger")
        ]
        self.skip_bodies = {
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "table"),
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "world"),
        }
        original = policy._move_toward

        def wrapped(target_pos, ee_pos, threshold=0.02):
            self.target = np.array(target_pos, dtype=float)
            self.threshold = threshold
            return original(target_pos, ee_pos, threshold)

        policy._move_toward = wrapped   # bọc để biết đích và ngưỡng mà không sửa file policy

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
        adr = self.model.jnt_qposadr[self.env._finger_left_jnt]
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
            "fo": self.finger_open(),                    # độ mở 1 ngón kẹp (m)
            "gr": bool(self.env._grasped),               # hộp đang dính vào tay
            "ag": round(float(action_grip), 2),          # lệnh kẹp: +1 mở, -1 đóng
            "hd": round(float(info["box_to_bowl_dist"]), 3),   # khoảng cách ngang hộp-bát
            "ok": bool(info["success"]),
            "bs": round(float(info["box_speed"]), 3),    # tốc độ hộp (m/s): chỉ tính thành công khi hộp đã nằm yên
            "arm": [r3(self.data.xpos[i]) for i in self.arm_ids],
            "wq": [round(float(x), 6) for x in self.data.xquat[self.arm_ids[-1]]],
            "fg": [r3(self.data.xpos[i]) for i in self.finger_ids],
            "ct": self.box_contacts(),
            "bc": self.bowl_contacts(),                  # khúc tay / ngón đang chạm bát
        }

    def run(self, seed):
        env, policy = self.env, self.policy
        obs, info = env.reset(seed=seed)
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
            obs, reward, terminated, truncated, info = env.step(action)
            ps = policy._phase_step if after == before else ps + 1
            frames.append(self.frame(step, before, after, ps, info, action[5]))
            if after != before:
                ps = 0
            if terminated or truncated or after > 10:
                break
        return {
            "seed": seed,
            "success": bool(info["success"]),
            "steps": len(frames) - 1,
            "frames": frames,
        }


def main():
    parser = argparse.ArgumentParser(description="Ghi các ván chạy ra file HTML để xem lại")
    parser.add_argument("--seeds", default="0-29", help="ví dụ 0-29, hoặc 2,3, hoặc 0-4,10")
    parser.add_argument("--policy", default="sim_engine.policies.scripted_policy:ScriptedPolicy")
    parser.add_argument("--real-grasp", action="store_true", help="tắt kẹp giả, hộp chỉ được giữ bằng ma sát thật")
    here = os.path.dirname(os.path.abspath(__file__))
    parser.add_argument("--out", default=None, help="mặc định docs/replay.html (kẹp giả) hoặc docs/replay_real.html (--real-grasp)")
    args = parser.parse_args()
    if args.out is None:
        args.out = os.path.join(here, "docs", "replay_real.html" if args.real_grasp else "replay.html")

    module_name, class_name = args.policy.split(":")
    policy_cls = getattr(importlib.import_module(module_name), class_name)
    env = SO101PickPlaceEnv(render_images=False, real_grasp=args.real_grasp, end_on_success=False)
    policy = policy_cls(env)
    rec = Recorder(env, policy)

    episodes = []
    for seed in parse_seeds(args.seeds):
        ep = rec.run(seed)
        episodes.append(ep)
        print(f"  seed {seed:3d}: {'OK  ' if ep['success'] else 'FAIL'} | {ep['steps']:3d} bước")

    # Thông tin cố định của cảnh để HTML vẽ đúng tỷ lệ
    scene = {
        "bowl": r3(env.get_bowl_pos()),
        "bowl_radius": config.BOWL_INNER_RADIUS,   # ngưỡng ngang để vào bát = lòng bát (tâm hộp phải nằm trong vòng này)
        "bowl_height": config.BOWL_HEIGHT,
        "bowl_inner": config.BOWL_INNER_RADIUS,      # lòng bát: tâm hộp phải cách tâm bát ít hơn mức này
        "settled_speed": config.SETTLED_SPEED,       # hộp phải chậm hơn mức này (m/s) mới tính là nằm yên
        "real_grasp": bool(args.real_grasp),
        "box_half": config.BOX_HALF,
        "close_threshold": env._gripper_close_threshold,
        "grasp_threshold": env._grasp_threshold,
        "max_steps": config.MAX_EPISODE_STEPS,
        "joints": config.JOINT_NAMES[:5],
        "gripper": [
            {"origin": origin, "offset": r3(env.model.geom_pos[g]), "half": r3(env.model.geom_size[g]),
             "body": mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_BODY, env.model.geom_bodyid[g])}
            for origin, name in [(-1, "gripper_mount"), (0, "left_finger_slider"), (0, "left_finger_pad"),
                                 (1, "right_finger_slider"), (1, "right_finger_pad")]
            for g in [env.model.geom(name).id]
        ],
        # Hình học THẬT của bát (đọc từ model): đáy hình trụ + 8 tấm tường hình hộp. Toạ độ tâm tính theo thế giới.
        "bowl_geoms": [
            {
                "name": mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, g),
                "type": "cylinder" if env.model.geom_type[g] == mujoco.mjtGeom.mjGEOM_CYLINDER else "box",
                "pos": r3(env.data.geom_xpos[g]),
                "size": [round(float(x), 4) for x in env.model.geom_size[g]],
                "yaw": round(float(np.arctan2(env.data.geom_xmat[g][3], env.data.geom_xmat[g][0])), 4),
            }
            for g in range(env.model.ngeom)
            if env.model.geom_bodyid[g] == mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "bowl")
        ],
        "capsule_radii": {
            ARM_BODIES[k]: float(env.model.geom_size[g, 0])
            for k, body in enumerate(rec.arm_ids)
            for g in range(env.model.ngeom)
            if env.model.geom_bodyid[g] == body and env.model.geom_type[g] == mujoco.mjtGeom.mjGEOM_CAPSULE
        },
        "arm_bodies": ARM_BODIES,            # đoạn k nằm giữa khớp k và khớp k+1 (xem JS: đánh số khớp 1-5)
        "policy": args.policy,
    }
    payload = json.dumps({"scene": scene, "episodes": episodes}, separators=(",", ":"), ensure_ascii=False)

    template_path = os.path.join(here, "docs", "replay_template.html")
    with open(template_path, encoding="utf-8") as f:
        template = f.read()
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
