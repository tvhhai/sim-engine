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
# Policy viết tay mặc định cho MỌI công cụ (Lab, quét vùng gắp, make_replay, select_seeds, ghi demo, đánh giá, viewer...):
# bản kết hợp chạy V1 (scripted_policy) rồi chuyển sang V2 (scripted_v2) khi V1 bỏ cuộc mà hộp còn gắp được. 774/774 seed.
# Muốn dùng policy khác cho một lệnh thì truyền --policy module:Lớp.
DEFAULT_POLICY = "sim_engine.policies.scripted_hybrid:ScriptedPolicyHybrid"

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
BOX_IN_BOWL_MARGIN = 0.005  # "hộp trong bát": TÂM hộp phải thấp hơn miệng bát ít nhất chừng này (m). Không dùng "đỉnh hộp thấp hơn miệng" (tâm < miệng - nửa cạnh):
                            # hộp nằm nghiêng tựa thành bát có tâm cao hơn mức đó vài mm (seed 9, 38) dù vẫn nằm trọn trong lòng bát, bị chấm hỏng oan
BOX_RANDOM_RANGE = 0.03     # hộp được đặt ngẫu nhiên trong ±3 cm quanh BOX_DEFAULT_POS (theo x, y)
BOWL_RANDOM_RANGE = 0.0     # bát đứng yên; đặt > 0 để bát cũng ngẫu nhiên trong ±giá trị này

# ---- Vị trí hộp ngẫu nhiên: HÌNH QUẠT quanh trục khớp 1, đo thật bằng quét gắp (tools/scan_pick_region.py -> results/pick_scan.json) ----
PICK_PAN_XY = (0.0, -0.15)           # trục khớp 1 (x, y) trong world
PICK_REGION = {                       # vùng hộp đặt được: r = khoảng cách tới trục khớp 1; góc đo từ hướng tay (+y) về phía +x
    "r_min": 0.12, "r_max": 0.42,     # đo (quét lại sau khi sửa roll hộp xa): r 14-32 cm gắp hết; r 37-42 cm gắp 94-98% mỗi vòng; r 43 cm 71%; r 44 cm gần như hỏng hết
    "angle_min_deg": -110.0, "angle_max_deg": 110.0,   # đo: +-120 hỏng (ngoài giới hạn khớp 1 = +-110); +-100/+-110 gắp được như nhau
}
# Vùng XANH (gắp được) hiển thị trong Studio / replay và vùng Lab cho phép đặt vật cần gắp / bát. Đo bằng quét NHIỀU GÓC XOAY hộp
# (0-75°, mỗi điểm 6 ván): r 8-42 cm, góc +-110° gắp được >= 90% (hầu hết 100%); r 6-7 cm 52-79%; r 43 cm 58%; r 44 cm 4%; góc +-120° 0%.
# Giữa vùng xanh và giới hạn đỏ là vùng hỗn hợp (r 4-8 cm và 42-44 cm). PICK_REGION ở trên chỉ dùng để ĐẶT HỘP NGẪU NHIÊN (box_sampling):
# đổi nó sẽ dời vị trí của cả 200 seed nên giữ nguyên.
PICK_GREEN = {"r_min": 0.08, "r_max": 0.42, "angle_min_deg": -110.0, "angle_max_deg": 110.0}
PAN_LIMIT_DEG = 110.0                # giới hạn khớp 1 (pan) trong XML: +-110 độ so với hướng tay
PAN_UNREACHABLE_MARGIN_DEG = 5.0     # hộp lệch quá PAN_LIMIT_DEG + biên này thì CHẮC CHẮN không với tới: policy về home ngay (đo: +-120 hỏng hết)
PICK_UNREACHABLE_INNER_RADIUS = 0.04  # m: hộp cách trục khớp 1 gần hơn mức này nằm chồng lên ĐẾ (đế ±5,5 cm x, -7..+2,6 cm y; quét r 2-4 cm: 0/69 gắp được) -> chắc chắn không gắp
PICK_UNREACHABLE_RADIUS = 0.44      # m: outer scan still finds successes at 0.43 m. Keep 0.44 m as the cutoff.
BOX_BOWL_CLEARANCE = 0.10            # hộp cách tâm bát tối thiểu (m)
BOX_SAMPLING = "sector"              # "sector": 200 seed chia đều các ô góc x bán kính trong hình quạt (phủ đủ vùng gắp); "square": ô ±BOX_RANDOM_RANGE cũ
BOX_SECTOR_CELLS = (10, 20)          # (số khoảng góc, số vòng bán kính chia đều DIỆN TÍCH): 10 x 20 = 200 ô, mỗi seed 0-199 một ô
# Seed cố định cho các trường hợp cực biên (thay cho 6 "case" trước đây): seed -> (tên, (x, y) của hộp)
FIXED_SEED_FIRST_EXTRA = 200          # seed cố định thêm về sau (điểm quét hỏng mới, ngoài vùng, ngoài tầm) đánh số từ đây; 0-199 là 200 seed phủ vùng gắp
BOX_FIXED_SEEDS = {
    1: ("hộp XA nhất (tay vươn tối đa)", (0.000, 0.230)),
    2: ("hộp GẦN nhất (sát đế, lệch một bên)", (0.060, -0.046)),
    3: ("hộp XA bát nhất + tay vươn xa (chéo)", (0.355, -0.087)),
    4: ("hộp XA bát + tay vươn gần", (0.118, -0.129)),
    5: ("hộp SÁT bát + tay vươn xa", (-0.260, 0.000)),
    6: ("hộp SÁT bát + tay vươn gần", (-0.118, -0.129)),
    # >>> SCAN-SEEDS (tools/scan_pick_region.py --update-seeds ghi lại khối này: seed 7.. = điểm quét hỏng TRONG vùng gắp, tiếp theo là seed pan ±100/±110)
    7: ("quét hỏng: r = 12 cm, góc -60°", (-0.1039, -0.0900)),
    8: ("quét hỏng: r = 12 cm, góc +50°", (0.0919, -0.0729)),
    9: ("quét hỏng: r = 34 cm, góc -80°", (-0.3348, -0.0910)),
    10: ("quét hỏng: r = 34 cm, góc +80°", (0.3348, -0.0910)),
    11: ("pan +100°, r = 15 cm (kiểm tra vùng gắp đối xứng)", (0.1477, -0.1760)),
    12: ("pan +100°, r = 25 cm (kiểm tra vùng gắp đối xứng)", (0.2462, -0.1934)),
    13: ("pan +110°, r = 15 cm (kiểm tra vùng gắp đối xứng)", (0.1410, -0.2013)),
    14: ("pan +110°, r = 25 cm (kiểm tra vùng gắp đối xứng)", (0.2349, -0.2355)),
    15: ("pan -100°, r = 15 cm (kiểm tra vùng gắp đối xứng)", (-0.1477, -0.1760)),
    16: ("pan -100°, r = 25 cm (kiểm tra vùng gắp đối xứng)", (-0.2462, -0.1934)),
    17: ("pan -110°, r = 15 cm (kiểm tra vùng gắp đối xứng)", (-0.1410, -0.2013)),
    18: ("pan -110°, r = 25 cm (kiểm tra vùng gắp đối xứng)", (-0.2349, -0.2355)),
    200: ("quét hỏng: r = 32 cm, góc -70°", (-0.3007, -0.0406)),
    201: ("ngoài vùng nhưng gắp được: r = 8 cm, góc -90°", (-0.0800, -0.1500)),
    202: ("ngoài vùng nhưng gắp được: r = 8 cm, góc -50°", (-0.0613, -0.0986)),
    203: ("ngoài vùng nhưng gắp được: r = 8 cm, góc +0°", (0.0000, -0.0700)),
    204: ("ngoài vùng nhưng gắp được: r = 8 cm, góc +50°", (0.0613, -0.0986)),
    205: ("ngoài vùng nhưng gắp được: r = 8 cm, góc +90°", (0.0800, -0.1500)),
    206: ("ngoài vùng nhưng gắp được: r = 10 cm, góc -90°", (-0.1000, -0.1500)),
    207: ("ngoài vùng nhưng gắp được: r = 10 cm, góc -50°", (-0.0766, -0.0857)),
    208: ("ngoài vùng nhưng gắp được: r = 10 cm, góc +0°", (0.0000, -0.0500)),
    209: ("ngoài vùng nhưng gắp được: r = 10 cm, góc +40°", (0.0643, -0.0734)),
    210: ("ngoài vùng nhưng gắp được: r = 10 cm, góc +90°", (0.1000, -0.1500)),
    211: ("ngoài vùng nhưng gắp được: r = 38 cm, góc -90°", (-0.3800, -0.1500)),
    212: ("ngoài vùng nhưng gắp được: r = 38 cm, góc -40°", (-0.2443, 0.1411)),
    213: ("ngoài vùng nhưng gắp được: r = 38 cm, góc +0°", (0.0000, 0.2300)),
    214: ("ngoài vùng nhưng gắp được: r = 38 cm, góc +40°", (0.2443, 0.1411)),
    215: ("ngoài vùng nhưng gắp được: r = 38 cm, góc +90°", (0.3800, -0.1500)),
    216: ("ngoài vùng nhưng gắp được: r = 40 cm, góc -90°", (-0.4000, -0.1500)),
    217: ("ngoài vùng nhưng gắp được: r = 40 cm, góc -50°", (-0.3064, 0.1071)),
    218: ("ngoài vùng nhưng gắp được: r = 40 cm, góc -40°", (-0.2571, 0.1564)),
    219: ("ngoài vùng nhưng gắp được: r = 40 cm, góc +40°", (0.2571, 0.1564)),
    220: ("ngoài vùng nhưng gắp được: r = 40 cm, góc +90°", (0.4000, -0.1500)),
    225: ("cặp đối xứng r = 38 cm, góc +5° (gắp được; đối xứng hỏng)", (0.0331, 0.2286)),
    226: ("cặp đối xứng r = 38 cm, góc -5° (hỏng; đối xứng gắp được)", (-0.0331, 0.2286)),
    227: ("cặp đối xứng r = 38 cm, góc +85° (hỏng; đối xứng gắp được)", (0.3786, -0.1169)),
    228: ("cặp đối xứng r = 38 cm, góc -85° (gắp được; đối xứng hỏng)", (-0.3786, -0.1169)),
    229: ("cặp đối xứng r = 38 cm, góc +95° (gắp được; đối xứng hỏng)", (0.3786, -0.1831)),
    230: ("cặp đối xứng r = 38 cm, góc -95° (hỏng; đối xứng gắp được)", (-0.3786, -0.1831)),
    231: ("cặp đối xứng r = 38 cm, góc +100° (gắp được; đối xứng hỏng)", (0.3742, -0.2160)),
    232: ("cặp đối xứng r = 38 cm, góc -100° (hỏng; đối xứng gắp được)", (-0.3742, -0.2160)),
    233: ("cặp đối xứng r = 40 cm, góc +65° (gắp được; đối xứng hỏng)", (0.3625, 0.0190)),
    234: ("cặp đối xứng r = 40 cm, góc -65° (hỏng; đối xứng gắp được)", (-0.3625, 0.0190)),
    235: ("cặp đối xứng r = 40 cm, góc +70° (gắp được; đối xứng hỏng)", (0.3759, -0.0132)),
    236: ("cặp đối xứng r = 40 cm, góc -70° (hỏng; đối xứng gắp được)", (-0.3759, -0.0132)),
    237: ("cặp đối xứng r = 40 cm, góc +75° (gắp được; đối xứng hỏng)", (0.3864, -0.0465)),
    238: ("cặp đối xứng r = 40 cm, góc -75° (hỏng; đối xứng gắp được)", (-0.3864, -0.0465)),
    239: ("cặp đối xứng r = 40 cm, góc +80° (hỏng; đối xứng gắp được)", (0.3939, -0.0805)),
    240: ("cặp đối xứng r = 40 cm, góc -80° (gắp được; đối xứng hỏng)", (-0.3939, -0.0805)),
    241: ("cặp đối xứng r = 40 cm, góc +85° (gắp được; đối xứng hỏng)", (0.3985, -0.1151)),
    242: ("cặp đối xứng r = 40 cm, góc -85° (hỏng; đối xứng gắp được)", (-0.3985, -0.1151)),
    243: ("cặp đối xứng r = 40 cm, góc +110° (gắp được; đối xứng hỏng)", (0.3759, -0.2868)),
    244: ("cặp đối xứng r = 40 cm, góc -110° (hỏng; đối xứng gắp được)", (-0.3759, -0.2868)),
    245: ("cặp đối xứng r = 42 cm, góc +55° (gắp được; đối xứng hỏng)", (0.3440, 0.0909)),
    246: ("cặp đối xứng r = 42 cm, góc -55° (hỏng; đối xứng gắp được)", (-0.3440, 0.0909)),
    247: ("cặp đối xứng r = 42 cm, góc +60° (hỏng; đối xứng gắp được)", (0.3637, 0.0600)),
    248: ("cặp đối xứng r = 42 cm, góc -60° (gắp được; đối xứng hỏng)", (-0.3637, 0.0600)),
    249: ("cặp đối xứng r = 42 cm, góc +75° (gắp được; đối xứng hỏng)", (0.4057, -0.0413)),
    250: ("cặp đối xứng r = 42 cm, góc -75° (hỏng; đối xứng gắp được)", (-0.4057, -0.0413)),
    251: ("cặp đối xứng r = 42 cm, góc +100° (hỏng; đối xứng gắp được)", (0.4136, -0.2229)),
    252: ("cặp đối xứng r = 42 cm, góc -100° (gắp được; đối xứng hỏng)", (-0.4136, -0.2229)),
    253: ("cặp đối xứng r = 42 cm, góc +105° (hỏng; đối xứng gắp được)", (0.4057, -0.2587)),
    254: ("cặp đối xứng r = 42 cm, góc -105° (gắp được; đối xứng hỏng)", (-0.4057, -0.2587)),
    # <<< SCAN-SEEDS
    # Seed đặt tay (ngoài khối SCAN-SEEDS, công cụ quét không ghi đè): hộp ở chỗ seed 83 nhưng bát đối xứng qua mặt phẳng x = 0 (xem BOWL_FIXED_SEEDS)
    1000: ("hộp như seed 83 (góc +107°, r 35 cm), bát đối xứng bên kia (góc -107°): tay xoay 214°", (0.3360, -0.2550)),
    1001: ("quét hỏng vùng xanh: r = 12 cm, góc -110°", (-0.1128, -0.1910)),
    1002: ("quét hỏng vùng xanh: r = 12 cm, góc +40°", (0.0771, -0.0581)),
    1003: ("quét hỏng vùng xanh: r = 12 cm, góc +110°", (0.1128, -0.1910)),
    1004: ("quét hỏng vùng xanh: r = 34 cm, góc -110°", (-0.3195, -0.2663)),
    1005: ("quét hỏng vùng xanh: r = 34 cm, góc +40°", (0.2185, 0.1105)),
    1006: ("quét hỏng vùng xanh: r = 36 cm, góc -110°", (-0.3383, -0.2731)),
    1007: ("quét hỏng vùng xanh: r = 36 cm, góc +30°", (0.1800, 0.1618)),
    1008: ("quét hỏng vùng xanh: r = 37 cm, góc -110°", (-0.3477, -0.2765)),
    1009: ("quét hỏng vùng xanh: r = 37 cm, góc -80°", (-0.3644, -0.0858)),
    1010: ("quét hỏng vùng xanh: r = 37 cm, góc -25°", (-0.1564, 0.1853)),
    1011: ("quét hỏng vùng xanh: r = 37 cm, góc -20°", (-0.1265, 0.1977)),
    1012: ("quét hỏng vùng xanh: r = 37 cm, góc +55°", (0.3031, 0.0622)),
    1013: ("quét hỏng vùng xanh: r = 37 cm, góc +60°", (0.3204, 0.0350)),
    1014: ("quét hỏng vùng xanh: r = 38 cm, góc -110°", (-0.3571, -0.2800)),
    1015: ("quét hỏng vùng xanh: r = 38 cm, góc +45°", (0.2687, 0.1187)),
    1016: ("quét hỏng vùng xanh: r = 39 cm, góc -110°", (-0.3665, -0.2834)),
    1017: ("quét hỏng vùng xanh: r = 40 cm, góc +45°", (0.2828, 0.1328)),
    1018: ("quét hỏng vùng xanh: r = 41 cm, góc -110°", (-0.3853, -0.2902)),
    1019: ("quét hỏng vùng xanh: r = 41 cm, góc -55°", (-0.3359, 0.0852)),
    1020: ("quét hỏng vùng xanh: r = 41 cm, góc +45°", (0.2899, 0.1399)),
    1021: ("quét hỏng vùng xanh: r = 41 cm, góc +50°", (0.3141, 0.1135)),
    1022: ("quét hỏng vùng xanh: r = 42 cm, góc -110°", (-0.3947, -0.2936)),
}
# Seed có BÁT đặt tay (mặc định bát đứng yên ở BOWL_DEFAULT_POS): seed -> (x, y) của bát. env.reset(options={"bowl_xy": ...}) vẫn thắng.
BOWL_FIXED_SEEDS = {
    1000: (-0.3360, -0.2550),
}
SETTLE_WAIT_STEPS = 60       # policy xong (phase 11) mà hộp đã trong bát, đã thả nhưng còn đang trượt/lắc: chờ tối đa chừng này bước cho hộp nằm yên rồi mới chấm (xem env.box_settling)
SETTLED_SPEED = 0.05        # hộp chuyển động chậm hơn mức này (m/s) mới coi là "đã nằm yên"

# Đường dẫn Python (dùng cho phím 3 Train trong giao diện).
# CẢNH BÁO: đây là đường dẫn kiểu Linux (bin/python3), trên Windows sẽ không tồn tại.
PYTHON = os.path.join(
    os.path.dirname(__file__), "..", "lerobot", ".venv", "bin", "python3"
)
