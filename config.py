"""Cấu hình trung tâm của SimEngine (mọi hằng số dùng chung nằm ở đây)."""

import os
import numpy as np

# Tên các khớp, đúng thứ tự như robot SO-101 thật.
# Thứ tự này quan trọng: action[0] điều khiển khớp đầu tiên, action[1] khớp thứ hai, ...
JOINT_NAMES = [
    "shoulder_pan",   # xoay đế (yaw)
    "shoulder_lift",  # nhấc vai lên/xuống
    "elbow_flex",     # gập khuỷu tay
    "wrist_flex",     # gập cổ tay
    "wrist_roll",     # xoay cổ tay
    "gripper",        # kẹp
]

N_JOINTS = 6  # tổng số khớp trong JOINT_NAMES (thực tế là 5 khớp tay + 1 gripper; env dùng JOINT_NAMES[:5] cho tay)
N_ACTIONS = 7  # 5 khớp tay + 2 số cho gripper (action[5] và action[6] giống nhau, đều là lệnh kẹp)

# Giới hạn khớp (radian), dùng cho actuator của MuJoCo
JOINT_LIMITS_RAD = {
    "shoulder_pan":  (-np.pi, np.pi),
    "shoulder_lift": (-np.pi / 2, np.pi / 2),
    "elbow_flex":    (-np.pi, 0.0),
    "wrist_flex":    (-np.pi / 2, np.pi / 2),
    "wrist_roll":    (-np.pi, np.pi),
    "gripper":       (-0.1745, 1.7453),  # góc hàm kẹp di động (rad); lớn = mở
}

# Tư thế "nhà" (home) tính bằng radian: GẤP GỌN sát đế như robot thật lúc nghỉ (đã kiểm tra không va chạm bàn / tay tự chạm)
HOME_QPOS_RAD = np.array([0.0, -1.6, 1.55, 0.9, 0.0, 1.2])
GRIPPER_OPEN_RAD = 1.2      # lệnh mở hết cỡ (hàm di động), action[5] = +1
GRIPPER_CLOSED_RAD = -0.1   # lệnh đóng/bóp chặt, action[5] = -1
GRIPPER_REST_RAD = GRIPPER_CLOSED_RAD   # tư thế nghỉ/gấp gọn: kẹp KHÉP (như robot thật khi cất), bắt đầu và kết thúc ván đều thế
WRIST_CAM_TILT_DEG = 0.0       # độ gập của giá đỡ camera cổ tay quanh chốt (độ). 0 = tư thế mặc định; dương/âm đổi góc nhìn lên/xuống mũi hàm. Kéo thanh "Gập cam" trong replay để xem thử

# Cấu hình camera (2 camera: webcam nhìn toàn cảnh và arm_cam gắn ở cổ tay)
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_NAMES = ["webcam", "arm_cam"]

# Vật lý
CONTROL_FREQ = 30  # Hz: tần số điều khiển, tức 1 lần policy ra action mỗi 1/30 giây
PHYSICS_TIMESTEP = 0.002  # giây: bước thời gian của MuJoCo
# Số bước vật lý chạy cho mỗi 1 bước điều khiển.
# Lưu ý: int(1/30/0.002) = int(16.67) = 16, nên tần số thực tế là 31.25 Hz chứ không đúng 30.
PHYSICS_SUBSTEPS = int(1.0 / CONTROL_FREQ / PHYSICS_TIMESTEP)

# Episode
MAX_EPISODE_STEPS = 500  # tối đa 500 bước (~16 giây ở 30 Hz) rồi bị coi là hết giờ

# Task: câu lệnh mặc định (VLA sẽ dùng câu này làm input ngôn ngữ)
DEFAULT_TASK = "Pick up the box and place it in the bowl"

# Đường dẫn
ASSETS_DIR = os.path.join(os.path.dirname(__file__), "assets")
SCENE_XML = os.path.join(ASSETS_DIR, "so101_real_scene.xml")
DEFAULT_DATASET_DIR = os.path.join(os.path.dirname(__file__), "..", "datasets", "sim_demos")
DEFAULT_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "sim_engine")

# Vị trí vật thể (mét, hệ tọa độ world, gốc ở giữa bàn)
BOX_DEFAULT_POS = np.array([0.15, 0.0, 0.44])   # hộp, nằm trên mặt bàn
BOWL_DEFAULT_POS = np.array([-0.15, 0.0, 0.41])  # bát, nằm trên mặt bàn
BOWL_RADIUS = 0.05   # (cũ) ngưỡng ngang của tiêu chí thành công kiểu cũ; tiêu chí mới dùng BOWL_INNER_RADIUS
BOWL_HEIGHT = 0.04   # chiều cao tường bát: miệng bát = z của bát + BOWL_HEIGHT
BOWL_INNER_RADIUS = 0.042   # lòng bát tới mặt TRONG của tường (tường dày 0.3 cm nên 0.045 - 0.003)
TABLE_HALF = (0.45, 0.40)   # nửa kích thước mặt bàn (x, y), mét: bàn 90 x 80 cm (khớp với assets/so101_real_scene.xml)
BOX_HALF = 0.015            # nửa cạnh hộp (hộp 3 cm)
BOX_RANDOM_RANGE = 0.03     # hộp được đặt ngẫu nhiên trong ±3 cm quanh BOX_DEFAULT_POS (theo x, y)
BOWL_RANDOM_RANGE = 0.0     # bát đứng yên; đặt > 0 để bát cũng ngẫu nhiên trong ±giá trị này

# ---- Vị trí hộp ngẫu nhiên: HÌNH QUẠT quanh trục khớp 1, đo thật bằng quét gắp (tools/scan_pick_region.py -> docs/pick_scan.json) ----
PICK_PAN_XY = (0.0, -0.15)           # trục khớp 1 (x, y) trong world
PICK_REGION = {                       # vùng hộp đặt được: r = khoảng cách tới trục khớp 1; góc đo từ hướng tay (+y) về phía +x
    "r_min": 0.10, "r_max": 0.36,     # đo: r = 0.08 hay 0.38 trở đi chỗ gắp chỗ hỏng; r >= 0.40 hỏng hết
    "angle_min_deg": -110.0, "angle_max_deg": 90.0,   # đo: -120 và +100 trở đi hỏng (phía sau cánh tay, khớp 1 không với tới)
}
BOX_BOWL_CLEARANCE = 0.10            # hộp cách tâm bát tối thiểu (m)
BOX_SAMPLING = "sector"              # "sector": 200 seed chia đều các ô góc x bán kính trong hình quạt (phủ đủ vùng gắp); "square": ô ±BOX_RANDOM_RANGE cũ
BOX_SECTOR_CELLS = (10, 20)          # (số khoảng góc, số vòng bán kính chia đều DIỆN TÍCH): 10 x 20 = 200 ô, mỗi seed 0-199 một ô
# Seed cố định cho các trường hợp cực biên (thay cho 6 "case" trước đây): seed -> (tên, (x, y) của hộp)
BOX_FIXED_SEEDS = {
    1: ("hộp XA nhất (tay vươn tối đa)", (0.000, 0.230)),
    2: ("hộp GẦN nhất (sát đế, lệch một bên)", (0.060, -0.046)),
    3: ("hộp XA bát nhất + tay vươn xa (chéo)", (0.355, -0.087)),
    4: ("hộp XA bát + tay vươn gần", (0.118, -0.129)),
    5: ("hộp SÁT bát + tay vươn xa", (-0.260, 0.000)),
    6: ("hộp SÁT bát + tay vươn gần", (-0.118, -0.129)),
}
SETTLED_SPEED = 0.05        # hộp chuyển động chậm hơn mức này (m/s) mới coi là "đã nằm yên"

# Đường dẫn Python (dùng cho phím 3 Train trong giao diện).
# CẢNH BÁO: đây là đường dẫn kiểu Linux (bin/python3), trên Windows sẽ không tồn tại.
PYTHON = os.path.join(
    os.path.dirname(__file__), "..", "lerobot", ".venv", "bin", "python3"
)
