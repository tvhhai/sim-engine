"""Dùng chung cho ghi dataset và đánh giá policy AI: dựng env theo seed và đổi quan sát của env sang "ngôn ngữ" LeRobot.

Vì sao cần file này?
    Env của repo (env.py) trả quan sát theo kiểu riêng: obs["state"] là mảng 7 số, obs["images"] là dict ảnh theo tên camera trong
    MJCF. LeRobot thì muốn mỗi khung hình là một dict có tên trường chuẩn ("observation.state", "observation.images.<tên>", "action").
    File này là "bộ phiên dịch" giữa hai bên. record_dataset.py (ghi dữ liệu) và eval_policy.py (chấm AI) cùng dùng nó, nên dữ liệu
    lúc train và lúc chạy thật chắc chắn có cùng định dạng. Nếu hai bên lệch nhau (vd thứ tự khớp khác) thì AI học xong sẽ chạy sai.

Một khung hình LeRobot gồm:
    observation.state          6 số: 5 góc khớp (độ) + độ mở kẹp (0 đóng .. 100 mở)
    observation.images.front   ảnh camera trước (webcam), H x W x 3, uint8
    observation.images.wrist   ảnh camera cổ tay (arm_cam), H x W x 3, uint8
    action                     6 số: lệnh 5 khớp + lệnh kẹp, thang [-1, 1] như env.step nhận (env nhận 7 số, số kẹp lặp 2 lần)
    task                       câu mô tả nhiệm vụ (để sau này dùng VLA nhận lệnh chữ)

Seed: vật đặt đúng vị trí hộp của seed trong env chuẩn (box_sampling), bát theo seed. Seed ghi dữ liệu (TRAIN_SEEDS) và seed đánh giá
(EVAL_SEEDS, bộ 774 seed quen thuộc) KHÔNG trùng nhau, để kết quả đánh giá đo khả năng tổng quát chứ không phải học thuộc.

Muốn đổi gì thì sửa ở đây:
    - độ phân giải ảnh       -> IMAGE_H, IMAGE_W   (lớn hơn: thấy hộp rõ hơn nhưng train chậm, tốn GPU)
    - camera có sẵn          -> CAMERAS            (tên bên phải phải là tên <camera> trong assets/*.xml)
    - chọn camera cho 1 dataset -> record_dataset --cameras top,wrist  (không cần sửa code)
    - thêm trường quan sát    -> features() và to_frame()  (sửa CẢ HAI cho khớp, rồi ghi lại dataset)
Sau mỗi thay đổi ở đây phải ghi lại dataset và train lại: dataset cũ có định dạng cũ.
"""

import numpy as np

from .. import config
from ..env import SO101PickPlaceEnv
from ..renderer import OffscreenRenderer

# --- Hằng số định dạng dữ liệu -------------------------------------------------------------------------------------------------

FPS = config.CONTROL_FREQ                      # 30 khung hình / giây = số bước điều khiển mỗi giây (thực tế 31,25 Hz, xem
                                               # config.PHYSICS_SUBSTEPS). LeRobot dùng số này để biết "1 giây là bao nhiêu khung".
IMAGE_H, IMAGE_W = 240, 320                    # kích thước ảnh lưu vào dataset. Nhỏ hơn 640x480 gốc để train vừa GPU 4 GB.
                                               # Đổi lại: hộp 4 cm trong camera trước chỉ còn khoảng 6x6 điểm ảnh.
CAMERAS = {"front": "webcam", "wrist": "arm_cam", "top": "topcam"}   # tên trong dataset -> tên camera trong MJCF.
                                               # front: camera cố định chéo phía trước (hộp 4 cm ~6x6 điểm ảnh);
                                               # wrist: camera gắn cổ tay, đi theo tay (lúc ở home KHÔNG thấy hộp);
                                               # top: camera nhìn thẳng từ trên xuống, thấy cả bàn (hộp ~12x12 điểm ảnh).
CAMERAS.update(camera1="topcam", camera2="arm_cam")
                                               # Tên camera của SmolVLA: smolvla_base được train sẵn với camera1/2/3. Khi fine-tune ta
                                               # đổi top -> camera1, wrist -> camera2 (rename_map trong configs/smolvla_finetune.yaml);
                                               # checkpoint SmolVLA vì vậy đòi camera1/camera2, hai dòng này để eval / DAgger dựng đúng.
DEFAULT_CAMERAS = ("front", "wrist")           # bộ camera của dataset v1
ENV_STATE_NAMES = ["box_x", "box_y", "box_z", "bowl_x", "bowl_y"]
                                               # "thông tin ăn gian" (privileged): toạ độ thật của vật và bát, đọc thẳng từ mô phỏng.
                                               # Robot thật không có; chỉ dùng để thí nghiệm "nếu AI biết chắc vật ở đâu thì
                                               # gắp được bao nhiêu" -> tách lỗi do NHÌN khỏi lỗi do ĐIỀU KHIỂN.
STATE_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]   # tên 6 số trong state/action
TASKS = {"box": "Pick up the {color} box and place it in the bowl", "sphere": "Pick up the {color} ball and place it in the bowl"}
                                               # câu lệnh chữ. ACT KHÔNG đọc câu này (chỉ lưu kèm); SmolVLA sau này mới đọc.
EVAL_SEEDS = list(range(200)) + sorted(k for k in config.BOX_FIXED_SEEDS if k >= 200) + list(range(2000, 2500))
                                               # bộ 774 seed chấm điểm, giống bộ dùng để chấm policy viết tay -> so sánh được.
TRAIN_SEED_START = 3000                        # seed 3000+ phủ cùng 200 ô vùng gắp như seed 0-199 nhưng điểm rơi khác.
                                               # Không bao giờ ghi dữ liệu từ EVAL_SEEDS, nếu không điểm chấm sẽ "ăn gian".


def features(cameras=DEFAULT_CAMERAS, env_state=False):
    """Mô tả các trường của một khung hình cho LeRobotDataset.create.

    LeRobot cần biết trước mỗi trường có kiểu gì, kích thước bao nhiêu, để tạo file parquet / video đúng.
    - "video": ảnh được nén thành mp4 (nhẹ hơn lưu từng ảnh rất nhiều); shape là (cao, rộng, 3 kênh màu).
    - "float32" shape (6,): một vector 6 số; "names" là tên từng số (để đọc dataset dễ hiểu, và để LeRobot khớp với robot thật).
    cameras: các camera đưa vào (tên khoá trong CAMERAS). env_state=True: thêm "observation.environment_state" (toạ độ vật, bát).
    ACT tự dùng MỌI trường observation.* có trong dataset làm đầu vào, nên chọn camera ở đây = chọn mạng được nhìn gì.
    """
    img = {"dtype": "video", "shape": (IMAGE_H, IMAGE_W, 3), "names": ["height", "width", "channels"]}
    vec = {"dtype": "float32", "shape": (6,), "names": STATE_NAMES}
    # dict(img) / dict(vec): tạo bản sao cho từng trường, để LeRobot sửa trường này không ảnh hưởng trường kia.
    feats = {"observation.state": vec, **{f"observation.images.{k}": dict(img) for k in cameras}, "action": dict(vec)}
    if env_state:
        feats["observation.environment_state"] = {"dtype": "float32", "shape": (len(ENV_STATE_NAMES),), "names": ENV_STATE_NAMES}
    return feats


def seed_scene(seed):
    """Vị trí hộp và bát của seed trong env chuẩn.

    Env chuẩn tự rải hộp theo seed (box_sampling.py). Ta dựng env chuẩn, reset với seed, đọc vị trí ra rồi đóng lại.
    Nhờ đó "seed 23" trong AI và trong replay policy viết tay là CÙNG một cảnh -> so sánh trực tiếp được.
    """
    base = SO101PickPlaceEnv(render_images=False, end_on_success=False)   # không cần ảnh, chỉ cần vị trí -> nhanh
    base.reset(seed=seed)
    xy, bowl = base.get_box_pos()[:2].tolist(), base.get_bowl_pos()[:2].tolist()   # [:2] = chỉ lấy x, y (bỏ độ cao z)
    base.close()
    return xy, bowl


def make_env(seed, kind="box", size=0.04, color="#e5484d", images=True):
    """Env có một vật (hộp / bóng cỡ `size`) đặt ở vị trí hộp của seed. Trả (env, options cho reset).

    Dùng chế độ "lab_objects" của env (giống Studio Lab) để đổi được loại vật và kích thước; env chuẩn chỉ có hộp 3 cm.
    options = {"bowl_xy": ...} phải truyền vào env.reset(seed=..., options=options) để bát nằm đúng chỗ của seed.
    images=False dùng khi không cần ảnh (thí nghiệm chỉ toạ độ, hoặc dựng phần mô tả cảnh của replay): nhanh hơn nhiều.

    Env tự render 2 camera của config.CAMERA_NAMES ở 640x480; ta TẮT cái đó và tự render bằng renderer nhỏ (320x240) trong
    to_frame, chỉ những camera thí nghiệm cần (xem env._lr_renderer).
    """
    xy, bowl = seed_scene(seed)
    obj = {"id": "target", "type": kind, "x": xy[0], "y": xy[1], "yaw": 0, "size": size, "color": color}
    env = SO101PickPlaceEnv(render_images=False, end_on_success=False, lab_objects=[obj])
    # MuJoCo vẽ ảnh từ camera ảo trong mô hình (không phải ảnh chụp thật). Gắn renderer vào env để to_frame dùng.
    env._lr_renderer = OffscreenRenderer(env.model, IMAGE_W, IMAGE_H) if images else None
    env._renderer = env._lr_renderer           # để env.close() đóng luôn renderer (giải phóng bộ nhớ đồ hoạ)
    return env, {"bowl_xy": bowl}


def to_frame(env, obs, action, task, cameras=DEFAULT_CAMERAS, env_state=False):
    """Trạng thái hiện tại của env + action (7 số) -> một khung hình LeRobot.

    obs["state"] của env có 7 số: 5 góc khớp, 1 ô luôn bằng 0 (quirk của repo gốc), 1 số độ mở kẹp. Ta bỏ ô số 5.
    action của env có 7 số: 5 khớp + 2 ô kẹp giống hệt nhau. Ta chỉ lấy 6 số đầu.
    Khi đánh giá (eval_policy), action chưa có nên truyền np.zeros(7); trường "action" khi đó bị bỏ qua.
    Ảnh: render từ env.data ngay lúc gọi (cùng thời điểm với obs vừa nhận).
    """
    s = obs["state"]
    frame = {
        "observation.state": np.array([*s[:5], s[6]], dtype=np.float32),   # 5 khớp + kẹp; bỏ ô state[5] luôn = 0
        "action": np.array(action[:6], dtype=np.float32),
        "task": task,
    }
    for key in cameras:                        # vd key="wrist" -> render camera "arm_cam" -> frame["observation.images.wrist"]
        frame[f"observation.images.{key}"] = env._lr_renderer.render(env.data, CAMERAS[key])
    if env_state:                              # toạ độ thật của vật (x, y, z) và bát (x, y)
        box, bowl = env.get_box_pos(), env.get_bowl_pos()
        frame["observation.environment_state"] = np.array([*box, *bowl[:2]], dtype=np.float32)
    return frame


def spec_from_features(keys):
    """Ngược lại của features(): từ tên các trường đầu vào của một mạng đã train, suy ra (camera, có env_state không).
    eval_policy dùng để dựng đúng loại quan sát mà mạng đó được train, không cần ghi tay."""
    # Bỏ những camera ta không có (vd camera3 / empty_camera_0 của SmolVLA): mạng tự bù ảnh trống cho chúng.
    cameras = tuple(k.split(".")[-1] for k in keys if k.startswith("observation.images.") and k.split(".")[-1] in CAMERAS)
    return cameras, "observation.environment_state" in keys


def to_env_action(action6):
    """Action 6 số của mạng -> 7 số env.step nhận (lệnh kẹp lặp cho cả hai ô).

    np.clip(-1, 1): mạng có thể đoán số hơi vượt khoảng (vd 1.03); env chỉ nhận [-1, 1] nên cắt lại cho an toàn.
    reshape(-1)[:6]: mạng có thể trả mảng dạng (1, 6); làm phẳng rồi lấy 6 số.
    """
    a = np.clip(np.asarray(action6, dtype=np.float32).reshape(-1)[:6], -1, 1)
    return np.array([*a, a[5]], dtype=np.float32)
