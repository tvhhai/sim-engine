"""Dùng chung cho ghi dataset và đánh giá policy AI: dựng env theo seed và đổi quan sát của env sang "ngôn ngữ" LeRobot.

Một khung hình LeRobot gồm:
    observation.state          6 số: 5 góc khớp (độ) + độ mở kẹp (0 đóng .. 100 mở)
    observation.images.front   ảnh camera trước (webcam), H x W x 3, uint8
    observation.images.wrist   ảnh camera cổ tay (arm_cam), H x W x 3, uint8
    action                     6 số: lệnh 5 khớp + lệnh kẹp, thang [-1, 1] như env.step nhận (env nhận 7 số, số kẹp lặp 2 lần)
    task                       câu mô tả nhiệm vụ (để sau này dùng VLA nhận lệnh chữ)

Seed: vật đặt đúng vị trí hộp của seed trong env chuẩn (box_sampling), bát theo seed. Seed ghi dữ liệu (TRAIN_SEEDS) và seed đánh giá
(EVAL_SEEDS, bộ 774 seed quen thuộc) KHÔNG trùng nhau, để kết quả đánh giá đo khả năng tổng quát chứ không phải học thuộc.
"""

import numpy as np

from .. import config
from ..env import SO101PickPlaceEnv
from ..renderer import OffscreenRenderer

FPS = config.CONTROL_FREQ                      # 30 khung hình / giây (thực tế 31,25 Hz, xem config.PHYSICS_SUBSTEPS)
IMAGE_H, IMAGE_W = 240, 320                    # nhỏ hơn 640x480 gốc để train vừa GPU 4 GB
CAMERAS = {"front": "webcam", "wrist": "arm_cam"}
STATE_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
TASKS = {"box": "Pick up the {color} box and place it in the bowl", "sphere": "Pick up the {color} ball and place it in the bowl"}
EVAL_SEEDS = list(range(200)) + sorted(k for k in config.BOX_FIXED_SEEDS if k >= 200) + list(range(2000, 2500))
TRAIN_SEED_START = 3000                        # seed 3000+ phủ cùng 200 ô vùng gắp như seed 0-199 nhưng điểm rơi khác


def features():
    """Mô tả các trường của một khung hình cho LeRobotDataset.create."""
    img = {"dtype": "video", "shape": (IMAGE_H, IMAGE_W, 3), "names": ["height", "width", "channels"]}
    vec = {"dtype": "float32", "shape": (6,), "names": STATE_NAMES}
    return {"observation.state": vec, **{f"observation.images.{k}": dict(img) for k in CAMERAS}, "action": dict(vec)}


def seed_scene(seed):
    """Vị trí hộp và bát của seed trong env chuẩn."""
    base = SO101PickPlaceEnv(render_images=False, end_on_success=False)
    base.reset(seed=seed)
    xy, bowl = base.get_box_pos()[:2].tolist(), base.get_bowl_pos()[:2].tolist()
    base.close()
    return xy, bowl


def make_env(seed, kind="box", size=0.04, color="#e5484d", images=True):
    """Env có một vật (hộp / bóng cỡ `size`) đặt ở vị trí hộp của seed. Trả (env, options cho reset)."""
    xy, bowl = seed_scene(seed)
    obj = {"id": "target", "type": kind, "x": xy[0], "y": xy[1], "yaw": 0, "size": size, "color": color}
    env = SO101PickPlaceEnv(render_images=False, end_on_success=False, lab_objects=[obj])
    if images:                                 # renderer nhỏ thay cho 640x480 mặc định
        env.render_images = True
        env._renderer = OffscreenRenderer(env.model, IMAGE_W, IMAGE_H)
    return env, {"bowl_xy": bowl}


def to_frame(obs, action, task):
    """Quan sát env + action (7 số) -> một khung hình LeRobot."""
    s = obs["state"]
    frame = {
        "observation.state": np.array([*s[:5], s[6]], dtype=np.float32),   # bỏ ô state[5] luôn = 0 (quirk của repo gốc)
        "action": np.array(action[:6], dtype=np.float32),
        "task": task,
    }
    for key, cam in CAMERAS.items():
        frame[f"observation.images.{key}"] = obs["images"][cam]
    return frame


def to_env_action(action6):
    """Action 6 số của mạng -> 7 số env.step nhận (lệnh kẹp lặp cho cả hai ô)."""
    a = np.clip(np.asarray(action6, dtype=np.float32).reshape(-1)[:6], -1, 1)
    return np.array([*a, a[5]], dtype=np.float32)
