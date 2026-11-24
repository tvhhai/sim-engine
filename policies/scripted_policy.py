"""Policy viết tay (scripted policy) cho gắp hộp bỏ vào bát: đi theo các waypoint, dùng IK.

Ý tưởng: không có AI, chỉ có code cứng "đưa tay tới điểm A, kẹp, nhấc, đưa tới điểm B, nhả".
Mỗi điểm đích là một vị trí 3D (x, y, z) của đầu tay. IK (Inverse Kinematics) tính ra
các góc khớp cần có để đầu tay tới được điểm đó.

IK chạy trên một bản sao dữ liệu MuJoCo ("shadow copy") nên không làm nhiễu mô phỏng thật.
Đầu ra là action đã chuẩn hóa về [-1, 1] để đưa vào env.step().
"""

import math

import numpy as np
import mujoco

from .base import BasePolicy
from .. import box_sampling, config


class ScriptedPolicy(BasePolicy):
    """Thực hiện pick-and-place theo chuỗi waypoint trong không gian Cartesian (x, y, z).

    Chiến lược tiếp cận: nhấc tay THẲNG LÊN từ home trước, rồi mới di chuyển ngang ở độ cao
    an toàn, để cánh tay không quét ngang qua vùng làm việc và gạt trúng vật thể.

    Các phase (giai đoạn), đánh số 0..11:
        0: Nhấc thẳng lên từ vị trí home
        1: Di chuyển ngang tới phía trên hộp (ở độ cao an toàn)
        2: Hạ xuống vị trí chuẩn bị kẹp (pre-grasp)
        3: Hạ xuống vị trí kẹp (grasp)
        4: Đóng kẹp (chờ một lúc)
        5: Nhấc thẳng lên
        6: Di chuyển tới phía trên bát (độ cao an toàn)
        7: Hạ xuống trong bát
        8: Mở kẹp (chờ một lúc)
        9: Rút lên khỏi bát
       10: Về lại home
       11: Xong (giữ nguyên)
    """

    # Hằng số của bộ IK và của phase (đặt tên để khỏi lẫn "số ma thuật" trong code)
    IK_ITERATIONS = 40        # số vòng lặp IK tối đa mỗi bước
    IK_TOLERANCE = 0.001      # dừng IK khi đầu tay cách đích < 1 mm (trên bản sao)
    IK_DAMPING = 0.05         # hệ số giảm chấn của damped least squares (ổn định gần điểm kỳ dị)
    IK_STEP_GAIN = 0.4        # mỗi vòng chỉ đi 40% bước tính được, cho an toàn
    PHASE_TIMEOUT_STEPS = 200  # trần mặc định cho một phase di chuyển (bước), dùng khi phase không có hạn riêng ở PHASE_TIMEOUTS
    # Hạn riêng cho phase di chuyển (bước). Phase 1 chạy tốt xong trong ≤ 30 bước (p99 = 15) nên 60 là dư; quá hạn nghĩa là đích sai, không đứng đợi 200 bước (6.7 s).
    PHASE_TIMEOUTS = {1: 60}
    PLACE_RECOVERY_LIMIT = 1
    PLACE_RETRACT_TIMEOUT = 45
    STALL_PHASES = (1,)        # chỉ phase 1 dùng phát hiện kẹt; các phase kẹp/mang hộp dựa vào hạn mặc định (đã đo: rút ngắn chúng làm hỏng kẹp)
    IK_LIMIT_PHASES = (5, 6, 7)  # phase nào IK kẹp góc khớp trong giới hạn thật. Đo 200 seed: (5, 6, 7) -> 198/200; cả mọi phase -> 195/200; không phase nào -> 197/200
    IK_RESPECT_LIMITS = False  # True: mọi lời giải IK bị kẹp trong giới hạn khớp. Đo 200 seed: bật cho mọi phase làm 3-4 seed mất kẹp (91, 149...)
    # Có giới hạn khớp thì IK khoá khớp bão hoà (xem _solve_ik) thay vì chỉ cắt góc, cộng khởi tạo khớp 1 theo hướng đích ở PAN_BEARING_PHASES.
    # Đo 274 seed: 271 -> 271 (1018 hết hỏng, 55 hỏng: seed mong manh, bản cũ cũng rơi hộp 2 lần khi nhấc); 14 seed sát đế chậm hơn 30+ bước.
    # Khoá khớp mà không khởi tạo khớp 1: seed 14 kẹt ở phase 6 (270/274)
    PAN_BEARING_PHASES = (6,)
    PAN_BEARING_DEG = 30.0     # chỉ khởi tạo lại khi hướng đích lệch góc khớp 1 hiện tại hơn chừng này
    PHASE5_ADAPTIVE_Z = False  # True: phase 5 cũng hạ độ cao nhấc xuống mức tới được. Đo 200 seed: 192-194/200 (xấu hơn), nên TẮT
    PHASE1_ADAPTIVE_Z = True   # phase 1 hạ độ cao đích xuống mức tới được
    REACTIVE_ROLL = True       # tay chạm bát lúc hạ xuống kẹp -> nhấc lên, đổi roll, kẹp lại (xem _flip_roll)
    PREDICT_BOWL_ROLL = True   # dự đoán TRƯỚC khi hạ: chọn sẵn roll mà hàm kẹp mở rộng không chạm bát (xem _predict_roll)
    ROLL_TURN_MODE = "half"    # "full": xoay tại chỗ tới roll mới rồi hạ; "half": xoay tại chỗ nửa đường, nửa còn lại xoay trong lúc hạ xuống
    PREDICT_PATH = True        # dự đoán xét cả quãng xoay roll (ở độ cao lơ lửng), không chỉ hai tư thế cuối
    PREDICT_PRETURN = True     # roll dự đoán khác roll hiện tại nhiều thì xoay TẠI CHỖ ở độ cao lơ lửng (cuối phase 1) rồi mới hạ, không xoay lúc đang hạ gần bát
    # Chọn roll để camera cổ tay không chạm vai/cánh tay/đế/bàn (xem _cam_geom_distance), xét cho MỌI hộp. Trước chỉ xét r < 0,16 m:
    # đo 774 seed xét mọi hộp cho kết quả trùng hệt (ở xa camera không thể chạm vai), nên bỏ ngưỡng
    CAM_AVOID = True
    # Kiểm tra hàm kẹp mở không chạm hộp khi hạ (JAW_BOX_MARGIN) cho hộp gần hơn chừng này (m). Trước chỉ áp cho r < 0,16 m: seed 1011
    # (r 37 cm) bị hàm cố định gạt hộp ra 25 mm lúc hạ ở phase 3, hộp lọt ra vùng xa rồi rơi khi nhấc. Đo 274 seed: 0,16 -> 273/274;
    # 0,385 -> 274/274 (17 seed đổi, không seed nào chậm hơn 30 bước); mọi hộp (1,0) -> 273 (152 ở r 39,4 cm hỏng). Ngưỡng nằm giữa 1011 và 152,
    # tức là canh theo đúng hai seed này: thêm seed mới ở r 37-40 cm thì nên đo lại
    JAW_AVOID_R = 0.385        # đo lại 774 seed: xét mọi hộp -> 772 (152 hỏng, rơi hộp 11 -> 19), giữ ngưỡng
    JAW_BOX_MARGIN = 0.0008    # hộp sát chân: hàm kẹp (mở hết cỡ) phải không chạm hộp ở tư thế kẹp (>= chừng này, m); chạm = đẩy hộp vào đế. Đo seed 205: roll chạm hộp -3..-9 mm đều hỏng
    FAR_FIXED_ROLL = True      # hộp r >= FAR_FIXED_R: roll mặc định -87°, không canh theo yaw hộp (xem _far_fixed_roll). Quét roll ép sẵn seed 244/246/248/252/254: roll ±87..93° gắp được, roll canh yaw (vd -123°) hỏng
    FAR_FIXED_R = 0.395        # gộp vào FAR_ROLL_R (0,33) -> 770/774 (50, 211, 2312 hỏng), giữ ngưỡng
    FAR_PREF_WEIGHT = 0.1      # hộp xa (r >= FAR_ROLL_R) canh roll theo yaw: trọng số kéo về roll ưa thích -87° (hộp thường 0,02). Seed 172 (r 36 cm, +32°) ở 0,02 canh ra roll ~3° làm hỏng; 0,05 và 0,1 đều 252/252, 0,1 ít bước hơn
    FAR_ROLL_ORDER = (-100, 40)  # thứ tự thử (độ). Quét roll ép sẵn: -100° gắp được seed 217-220/245/247/254 (-87° nằm ngay mép vực: 245 hỏng; -96° làm hỏng 219); seed 244 (-110°, mọi roll <= 25° ngoài tầm khớp 1) chỉ gắp được ở roll >= 40°
    FAR_FIXED_ERR = 0.014      # sai số IK (có giới hạn khớp) tối đa của đích kẹp (m); seed 244 ở -96°: 19 mm, phase 3 hết hạn 200 bước
    FAR_FIXED_CAM = 0.003      # camera cổ tay phải cách bàn/vai tối thiểu chừng này (m) ở tư thế chuẩn bị kẹp và kẹp: không cắm vào bàn (roll +93° cắm 27 mm)
    # Hộp sát giới hạn khớp 1 (lệch >= PAN_LIMIT_NEAR_DEG): đích kẹp = tâm hộp lệch GRASP_SIDE_OFFSET về phía hàm cố định, nên một nửa số roll
    # song song mặt hộp đặt đầu tay RA NGOÀI ±110°. IK không giới hạn khớp ở phase 2-3 vẫn ra lệnh pan -112°, khớp thật dừng ở -110° và hàm cố
    # định đè lên hộp (seed 1004: roll -105° sai số 16,7 mm). Roll đó sai số (có giới hạn khớp) > PAN_LIMIT_TRIGGER thì xoay thêm k·90° để
    # hàm CỐ ĐỊNH nằm phía trong tầm khớp 1, hàm di động ở phía ngoài đẩy hộp vào khi khép (roll +75°: 2,2 mm).
    # Đo 274 seed: 270 -> 271 (1004 hết hỏng; 17, 1001, 1003 nhanh hơn 20-40 bước). Ngưỡng 4 mm làm hỏng 244 (roll -95° sai 4-10 mm vẫn gắp được);
    # thử kẹp theo hướng bán kính (hàm cố định về phía đế) thì 268/274: kẹp chéo góc hộp, hỏng 28, 231, 244
    # Hộp xa (r >= DRAG_R): tay vươn gần hết cỡ nên trục kẹp nghiêng 47-66° so với phương thẳng đứng (seed 55, mọi roll -100..+80°), hai hàm ôm
    # hộp xiên và hộp tuột khi nhấc. Ván nào thành công thì lần kẹp cuối ở r <= 38,2 cm (hộp đã bị xê dịch vào sau lần rơi). Nên kẹp, KÉO TRƯỢT
    # hộp vào r = DRAG_TARGET, nhả rồi gắp lại (xem _drag_box). Đo 274 seed (camera có va chạm): 271 -> 273 (55, 1019, 1022 hết hỏng; 1011 hỏng:
    # hộp bị đẩy ra 41 cm ở các lần gắp hỏng trước), trung vị 164,5 -> 180,5 bước. DRAG_R 0,39: cũng 273 nhưng chậm hơn; DRAG_TARGET 0,36: 272.
    # Chỉ kéo sau khi đã rơi hộp một lần: 271 (không có tác dụng)
    # Đường hạ có kế hoạch cho hộp r < PLAN_R_MAX (xem _lower_to_box): phase 2 đi ngang ở độ cao lơ lửng tới NGAY TRÊN điểm kẹp và xoay roll xong,
    # phase 3 hạ THẲNG đứng. Đường cũ dừng phase 1 trên TÂM hộp rồi dời ngang ~2,6 cm ở 1 cm trên đỉnh hộp với hàm mở hết cỡ, hạ trong lúc còn dời
    # ngang (ngưỡng 2,5 cm) và roll còn xoay: seed 1011 hàm cố định gạt hộp 22 mm rồi phải đổi roll. Đo 274 seed: 274 -> 274, đổi roll 15 -> 9 lần,
    # 1011 162 -> 133 bước, trung bình +2,6 bước (8: +83 vì vẫn chạm hộp một lần rồi đổi roll). Áp cho mọi hộp: 264-273 (hộp xa tay vươn hết cỡ không
    # hạ thẳng được; r 38-39,5 cm thì 152/1016 hỏng); sau khi kéo hộp (_drag_box) cũng dùng đường cũ (hộp có thể còn nghiêng)
    PLANNED_DESCENT = True
    PLAN_R_MAX = 0.38
    # Đã thử thay PLAN_R_MAX bằng phép đo (774 seed): hạ thẳng khi có roll mà đường hạ sạch -> 764 (hộp xa tay nghiêng 47-50°, kẹp xiên rồi tuột);
    # thêm điều kiện trục tay nghiêng <= 44° / 46° -> 753 / 747 (vùng giữa mất đường hạ thẳng). Giữ ngưỡng bán kính
    PLAN_SAMPLES = 5           # số điểm kiểm tra dọc đường hạ
    PLAN_REACH_ERR = 0.015     # sai số IK (có giới hạn khớp) tối đa dọc đường hạ (m)
    PLAN_BOWL_MARGIN = 0.005   # hàm mở cách bát tối thiểu (m), chỉ xét khi hộp cách bát < 20 cm
    HOVER_ABOVE = 0.03         # độ cao lơ lửng = pre_grasp_z + chừng này (thấp hơn nếu phase 1 thấp hơn; tối thiểu pre_grasp_z + 1,5 cm)
    HOVER_NEAR, HOVER_FAR = 0.01, 0.025
    HOVER_STALL, HOVER_STALL_ANY = 5, 10
    PLAN_ROLL_SETTLE_DEG = 4.0
    PLAN_ROLL_WAIT = 25
    DRAG_FAR = True
    DRAG_R = 0.40
    DRAG_TARGET = 0.33
    DRAG_LIFT = 0.004          # nhấc nhẹ khi kéo để bớt ma sát với bàn (m)
    STAGED_LIFT = True         # phase 0: nâng khớp khuỷu (3) và cổ tay lên trước, khớp vai (2) chỉ nâng sau khi cánh tay đã cao (seed 2164: vai nâng trước quét trúng bóng sát đế)
    STAGED_LIFT_R_MAX = None   # None = áp cho mọi vị trí vật; số (m) = chỉ áp khi vật cách trục khớp 1 nhỏ hơn
    DRAG_UP_SPHERE = 0.02      # bóng xa: kẹp xong NHẤC THẲNG lên chừng này rồi mới kéo về (kéo sát bàn bóng lăn khỏi hai hàm, seed 1019)
    STAGED_LIFT_ELBOW_DEG = 6.0
    STAGED_LIFT_MAX_STEPS = 30
    DRAG_RELEASE_STEPS = 12
    DRAG_RAISE = 0.03          # nhấc tay lên chừng này sau khi nhả rồi mới gắp lại (m)
    PAN_LIMIT_ROLL = True
    PAN_LIMIT_NEAR_DEG = 100.0
    PAN_LIMIT_TRIGGER = 0.010
    PAN_LIMIT_ERR_OK = 0.004
    CAM_LIFT_CHECK = True          # kiểm tra camera cổ tay cả dọc đường NHẤC hộp (phase 5): hộp sát đế, camera quay vào trong sẽ quét vai / cánh tay trên khi khớp 3 mở ra
    # Nhấc hộp sát đế: dời đích RA NGOÀI theo bán kính tới r >= LIFT_MIN_R (mở khớp 3 ra khi nhấc thay vì để IK kéo tay vào phía đế, nơi camera cọ vào vai).
    # Đo 274 + 500 seed: 274/274 và 499/500 như cũ; camera xuyên 8 -> 3 và 1 -> 0 seed; số bước TB 184,2 -> 182,3 và 169,3 -> 168,2.
    # 14 cm: -0,4 / -1,4 bước; 18 cm: hộp ở 18 cm nhanh hơn nhưng 17 seed chậm hơn; 24 cm: hỏng 202, 207. None = nhấc thẳng như cũ
    # Bóng nhanh hơn mức này (m/s) thì nâng tay và đợi đứng yên rồi mới hạ (chỉ áp cho bóng). Đo 774 seed bóng 3 cm: bật 0,005 -> sửa 4 seed nhưng làm
    # hỏng thêm 33 (760 -> 731): bóng còn rung nhẹ lúc tiếp cận nên tay chờ vô ích rồi hết giờ. Nên TẮT (None); thử ngưỡng khác trước khi bật.
    REST_SPEED = None
    REST_WAIT_MAX = 80             # tối đa chừng này bước đợi trong một ván
    LIFT_MIN_R = 0.16
    LIFT_EXT_RISE = 0.015          # m: hộp lên cao chừng này rồi mới bắt đầu dời ra ngoài (tránh kéo hộp lê trên bàn)
    LIFT_SIM_STEPS = 70            # tối đa chừng này bước khi mô phỏng đường nhấc (_lift_cam_clearance)
    CAM_LIFT_MARGIN = 0.002        # camera cách vai / cánh tay trên / đế tối thiểu chừng này (m) ở các độ cao nhấc
    CAM_CLEARANCE_MARGIN = 0.010   # camera phải cách vai/cánh tay/đế/bàn tối thiểu chừng này (m)
    BOWL_CLEARANCE_MARGIN = 0.010   # hàm kẹp (mở hết cỡ) phải cách bát tối thiểu chừng này (m) ở tư thế chuẩn bị kẹp và tư thế kẹp
    JAW_STUCK_STEPS = 6        # hàm CỐ ĐỊNH nằm trên đỉnh hộp (chạm hộp, đầu tay còn cao hơn đáy > 1,2 cm) liền chừng này bước ở phase 3 thì nhấc lên, đổi roll rồi hạ lại (như khi chạm bát). Seed 1001/1003: hàm đè lên hộp, phase 3 hết hạn
    # Phase 4, TRƯỚC khi đóng kẹp: đo trục kẹp trên tư thế THẬT (không phải tư thế IK dự đoán) rồi xoay wrist_roll cho song song cặp mặt hộp.
    # Roll chọn ở _choose_roll tính trên lời giải IK; tư thế thật lệch khỏi đó vài độ (seed 1002: -8,6°) nên hàm cố định chạm GÓC hộp trước,
    # hộp xoay tới +30° trong lúc khép và kẹp trượt. Đo 274 seed: 266 -> 270 (1002, 1009, 1010, 1011 hết hỏng, không seed nào mất)
    GRASP_ALIGN = True
    GRASP_ALIGN_TOL_DEG = 2.0  # lệch trục kẹp - mặt hộp dưới chừng này thì đóng kẹp luôn
    GRASP_ALIGN_MAX_DEG = 20.0  # chỉ chỉnh roll trong ± chừng này quanh roll đã chọn
    GRASP_ALIGN_STEPS = 12     # tối đa chừng này bước chờ chỉnh roll; hết thì đóng kẹp như cũ
    BOWL_HIT_STEPS = 3         # tay chạm bát liền chừng này bước khi hạ xuống kẹp (phase 2-4) thì nhấc lên, xoay wrist_roll rồi kẹp lại
    ROLL_FLIP_OFFSETS = (math.pi / 2, -math.pi / 2, math.pi, -math.pi)   # thứ tự thử khi đổi roll (rad so với roll đang giữ): 90° trước, 180° sau
    # Đo 200 seed + 57 vị trí quanh bát: dự đoán + phản ứng (90°, 45° tại chỗ + 45° lúc hạ) -> 194/200 (seed 7: 157 bước, 198: 102, 99: 126);
    # chỉ phản ứng 180° -> 193/200; dự đoán + phản ứng 180° đầy đủ -> 194/200 nhưng chậm (seed 7: 238); 90° đầy đủ -> seed 7 hỏng.
    ROLL_CLEAR_MARGINS = (0.035, 0.025, 0.015)   # nhấc lên cao hơn miệng bát chừng này (m) trước khi xoay roll; lấy mức đầu tiên với tới được
    ROLL_TURN_STEPS = 60       # tối đa chừng này bước để xoay roll tại chỗ
    VIA_ANGLE_DEG = 165.0      # mang hộp tới bát cách tay quá chừng này độ (quanh khớp 1): đi qua điểm trung gian ở giữa (IK một bước sẽ chọn chiều xoay pan ngược và kẹt ở giới hạn)
    # >0: phase 1 tiến đích đầu tay từng đoạn chừng này (m) cho MỌI hộp: đi thẳng ở độ cao lơ lửng, không võng xuống quét hộp khi khớp vượt đích.
    # Trước chỉ áp cho r >= TCP1_R (0,38 rồi 0,37): seed 2187 (r 37,98 cm, roll đã xoay sớm từ FAR_ROLL_R) võng xuống đẩy hộp ra ngoài tầm.
    # Đo 774 seed: áp cho mọi hộp 773/774 như ngưỡng 0,37 (trung bình +1,1 bước), bỏ được một ngưỡng bán kính
    TCP_STEP1 = 0.02
    TCP_STEP = 0.02            # C: >0: ở phase 2-3 giới hạn mỗi tick đích đầu tay tiến tối đa chừng này (m) trước khi giải IK
    FAR_ROLL_R = 0.33          # hộp xa khớp 1 hơn mức này (m): kẹp với roll FAR_ROLL_DEG thay vì roll gần tư thế nghỉ (kẹp ổn hơn khi hộp trượt/ma sát thấp)
    FAR_ROLL_EARLY = True      # True: xoay roll song song lúc di chuyển ra; False: tới phía trên hộp mới xoay tại chỗ
    FAR_ROLL_DEG = -87.0       # -87 (không phải +93): +93 làm cụm cam đâm xuống bàn (-11 mm) ở 7/43 hộp xa; -87 luôn cách bàn >= 30 mm
    MAX_ROLL_FLIPS = 3         # số lần đổi roll tối đa trong một ván (thử lần lượt: xoay 180°, +90°, -90° so với roll đang giữ)
    STALL_STEPS = 8            # (trước 20) đầu tay không tiến gần đích thêm STALL_PROGRESS trong chừng này bước liền thì coi là KẸT, sang phase kế
    # Đo 274 + 500 seed: 20 -> 8 bước không hỏng seed nào, 49 seed nhanh hơn, 1 chậm hơn; 5 hoặc 3 bước hỏng 203 / 208 (hộp sát đế, đường hạ nhạy)
    STALL_PROGRESS = 0.001     # mét

    # Ưu tiên PHỤ về tư thế "vòm" (khớp 2, 3, 4 cùng cong về một phía, khuỷu ở trên cao) để cánh tay
    # tránh võng xuống chạm bàn và bát. Đây chỉ là xu hướng "kéo nhẹ", không phải ràng buộc cứng:
    # IK vẫn tự do uốn khác đi (kể cả duỗi thẳng 2-3-4 để vươn xa) khi đích đòi hỏi.
    POSTURE_GAIN = 0.1        # K: kéo mạnh cỡ nào (0 = tắt). Quá lớn (>=0.3) thì cản việc tới đích.
    POSTURE_PREF_DEG = (0, 10, -15, 60, 0)   # góc ưa thích (độ): pan, lift, elbow, wrist_flex, wrist_roll
    POSTURE_MASK = (0, 1, 1, 1, 0)           # khớp nào được kéo (pan và wrist_roll thì không)

    # Ở các phase gần bát, ép (mềm) tay kẹp chúc xuống thẳng đứng: má kẹp mở theo phương thẳng đứng thì không
    # quét ngang vào thành bát, và có thể thả sát tâm bát. "Mềm" nghĩa là cộng vào bài toán IK với trọng số nhỏ
    # chứ không bắt buộc: khớp có giới hạn gập nên nhiều vị trí không thể thẳng đứng hoàn toàn.
    TOOL_DOWN_PHASES = (2, 3, 4, 5, 6, 7, 8, 9)   # từ lúc hạ xuống hộp tới lúc thả (phase 0, 1 đi ở độ cao cao, tay không thẳng đứng được)
    TOOL_DOWN_WEIGHT = 0.02   # trọng số hướng so với vị trí. 0.03 kém hơn (vẫn chạm bát), >= 0.1 làm đầu tay không tới đích.
    PLACE_Z = 0.475   # độ cao đầu tay lúc thả (miệng bát ở z 0.46). Đo trên 100 seed:
                              #   bát cố định:  0.48 -> 199/200, 0.50 -> 198/200 (gần như bằng nhau)
                              #   bát ngẫu nhiên ±3 cm: 0.48 -> 96/100 (chạm bát 41 ván), 0.50 -> 100/100 (chạm bát 0)
                              #   bát ngẫu nhiên ±5 cm: 0.48 -> 87/100, 0.50 -> 97/100
                              # nên chọn 0.50: ngón còn cao hơn miệng bát, ít vướng khi bát lệch khỏi chỗ gốc.
    # Kẹp THẬT với SO-101: hàm cố định nhọn như mũi kim, hàm di động xoay quanh bản lề nên hai hàm tạo hình chữ V.
    # Đo trên model: mặt hàm cố định ở sát điểm ee (≈ +0.1 cm dọc trục kẹp), mặt hàm di động ở +1.5 cm khi góc 0 rad
    # và rộng ra ~0.65 cm mỗi 0.1 rad, nên hộp 3 cm nằm giữa hai hàm khi hàm mở ~0.25 rad. Vì vậy:
    #   - điểm ee phải lệch khỏi tâm hộp GRASP_SIDE_OFFSET theo trục kẹp (về phía hàm cố định), không đặt ngay trên hộp
    #     (đặt ngay trên hộp thì mũi hàm cố định đè lên mặt hộp);
    #   - ee hạ xuống GRASP_Z (ngang tâm hộp) để hai hàm ôm ở phần giữa thân hộp;
    #   - wrist_roll xoay để trục kẹp song song với một cặp mặt hộp (nếu không hai hàm kẹp vào GÓC hộp).
    GRASP_Z = 0.43
    GRASP_SIDE_OFFSET = 0.022
    CARRY_Z = 0.51            # độ cao đầu tay khi mang: tay kẹp thẳng đứng được ở độ cao này, miệng bát z 0.46 nên hộp (đáy ~0.49) vẫn cao hơn
    LIFT_STEP_LIMIT = 0.01    # phase 5-7: mỗi bước, lệnh khớp (thang -1..1) đổi tối đa chừng này, để hộp không bị giật
    PLACE_ARRIVE_THRESHOLD = 0.04   # phase 7: kẹp thẳng đứng làm tay thiếu 2-4 cm tới đích, nên nới ngưỡng "tới nơi"
    # Lệnh mở kẹp lúc thả (action > 0.5 thì env coi là thả). 0.55 mở mỗi ngón ~33 mm, hai má hở ~67 mm: lớn hơn
    # đường chéo hộp (42 mm) để hộp rơi xéo vẫn lọt, nhưng nhỏ hơn mở hết (82 mm, rộng hơn lòng bát 84 mm).
    RELEASE_OPEN_CMD = 0.55
    RELEASE_CLEAR_RAD = 0.8   # phase 8 kết thúc khi hộp đã rời tay (hai hàm hết cùng chạm) VÀ hàm di động mở tới góc này (rad)
    SETTLE_STILL_STEPS = 3    # số bước liền hộp phải chậm hơn SETTLED_SPEED mới coi là đã nằm yên
    SETTLE_WAIT_MAX = 15      # sau khi rút lên (phase 9), chờ tối đa chừng này bước cho hộp nằm yên rồi mới kiểm tra hộp có trong bát không
    REGRASP_R_MIN = 0.08      # hộp rơi ra ngoài bát: chỉ gắp lại khi cách trục khớp 1 trong khoảng này (m), ngoài tầm thì về home
    REGRASP_R_MAX = 0.31
    RELEASE_RAMP = 0.10       # lệnh mở kẹp tăng tối đa chừng này mỗi bước (thang -1..1)
    # 0.10: mở kẹp đủ chậm để hàm không kéo lật hộp khi thả (hàm kẹp chia khối ôm sát hơn). Đo trên 400 seed: 0.10 -> 400/400 (trung vị 158),
    # 0.07 -> 400/400 (trung vị 164), 0.085 -> 195/200, 0.15 -> 192/200. Kết quả nhạy ở khoảng này, nên đổi thì phải đo lại.

    def __init__(self, env):
        self.env = env
        self.model = env.model
        self.data = env.data

        # Bản sao dữ liệu MuJoCo dùng riêng cho IK: thử các góc khớp "trên giấy" mà không
        # làm robot thật cử động
        self._ik_data = mujoco.MjData(self.model)

        self._joint_names = config.JOINT_NAMES[:5]
        self._joint_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)
            for n in self._joint_names
        ]
        self._dof_ids = [self.model.jnt_dofadr[jid] for jid in self._joint_ids]
        self._wrist_roll_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "gripper")
        self._gripper_joint = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "gripper")
        self._qpos_adr = np.array([self.model.jnt_qposadr[jid] for jid in self._joint_ids])  # vị trí của 5 khớp trong data.qpos
        self._jnt_lo = self.model.jnt_range[self._joint_ids, 0] + 0.01       # giới hạn góc 5 khớp (chừa 0.01 rad), IK không được vượt
        self._jnt_hi = self.model.jnt_range[self._joint_ids, 1] - 0.01
        self._ee_site = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "ee_site")

        self._phase = 0                 # đang ở phase nào
        self._phase_step = 0            # đã ở phase này bao nhiêu bước (dùng cho timeout và chờ)
        self._retries = 0               # đã thử kẹp lại bao nhiêu lần (phase 4 quay về phase 2)
        self._max_retries = 2           # tối đa bao nhiêu lần thử lại
        self._rest_wait = 0
        self._profile_alt = False       # bóng: đang dùng cấu hình kẹp của hộp thay vì của bóng (xem _switch_object_profile)
        self._last_finger = None        # độ mở ngón ở bước trước (để biết ngón đã ngừng khép chưa)
        self._gripper_target = 1.0      # lệnh kẹp: +1 mở, -1 đóng
        self._current_joint_targets = None   # góc khớp đích mà IK vừa tính (radian)
        self._smoothed_action = None    # action đã làm mượt (để chuyển động không giật)
        self._prev_grip = -1.0          # lệnh kẹp ở bước trước (để giới hạn tốc độ mở); ván bắt đầu với kẹp khép
        self._regrasps = 0              # đã kẹp lại bao nhiêu lần vì hộp tuột khi đang mang
        self._still = 0                 # số bước liền hộp đang nằm yên (sau khi rút lên)
        self._settle_wait = 0           # số bước đã chờ hộp nằm yên sau khi rút lên (phase 9)
        self._best_dist = 0.0           # khoảng cách tốt nhất tới đích trong phase hiện tại (để phát hiện kẹt)
        self._stall = 0                 # số bước liền không tiến thêm
        self.last_exit = None           # lý do thoát phase ở bước này: arrived / stall / timeout (None = vẫn trong phase)
        self._p1_z = None               # độ cao đích của phase 1 (hạ thấp nếu độ cao an toàn nằm ngoài tầm với)
        self._lift_xy0, self._lift_z0 = None, None   # vị trí đầu tay lúc bắt đầu nhấc (xem _lift_xy)
        self._lift_z = None             # độ cao đích của phase 5 (nhấc hộp), cũng hạ thấp khi hộp sát đế
        self._bowl_hits = 0             # số bước liền tay đang chạm bát khi hạ xuống kẹp
        self._jaw_stuck = 0             # số bước liền hàm cố định đè lên hộp khi hạ xuống kẹp
        self._align_steps = 0           # số bước đã chờ chỉnh roll cho song song mặt hộp ở đầu phase 4 (xem GRASP_ALIGN)        self._roll_flips = 0           # đã đổi roll bao nhiêu lần vì tay chạm bát
        self._roll_pref = None          # roll ưu tiên sau khi đổi (None = chọn roll gần roll hiện tại nhất)
        self._cam_ok_key = None         # vị trí hộp đã kiểm camera và roll mặc định đủ xa
        self._pred_changed = False      # roll được DỰ ĐOÁN đổi để tránh bát (khác roll gần nhất)
        self._preturn_key = None        # vị trí hộp đã xoay-trước-khi-hạ cho
        self._far_roll = None           # roll cho hộp xa (đặt ở reset)
        self._far_fix_key, self._far_fix_roll = None, None   # vị trí hộp và roll đã chốt cho hộp r >= FAR_FIXED_R (xem _far_fixed_roll)
        self._pan_key, self._pan_roll = None, None   # (vị trí hộp, roll gốc) đã xét và roll thay thế cho hộp sát giới hạn khớp 1 (xem _pan_limit_roll)
        self._drag, self._drag_n, self._drag_goal, self._dragged = None, 0, None, False
        self._drag_goal_after = None   # kéo hộp xa vào trong (xem _drag_box)
        self._plan = None               # kế hoạch đường hạ: roll, điểm kẹp G, độ cao lơ lửng (xem _make_descent_plan)
        self._via_done = False          # phase 6 đã đi qua điểm trung gian (bát lệch quá VIA_ANGLE_DEG)
        self._tcp_goal = None
        self._pred_key = None           # vị trí hộp đã dự đoán roll cho (để dự đoán lại khi hộp bị đẩy đi)
        self._roll_turn_target = None   # góc roll tại chỗ của bước "turn"
        self._roll_stage = None         # None | "lift" (nhấc thẳng lên khỏi miệng bát) | "turn" (xoay wrist_roll tại chỗ)
        self._roll_z = None             # độ cao nhấc lên để xoay roll
        self._arm_body_ids = {mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, n) for n in
                              ("shoulder", "upper_arm", "lower_arm", "wrist", "gripper", "moving_jaw_so101_v1", "wrist_cam_arm")}
        self._place_retries = 0
        self._place_retract_target = None
        self._place_hold_action = None
        self._lost_steps = 0            # số bước liên tiếp hộp nằm xa ee khi đang mang
        self._roll_hold = None          # góc wrist_roll đã chọn để trục kẹp song song mặt hộp (None = chưa chọn)
        self._clamp_dir = np.array([1.0, 0.0, 0.0])   # hướng trục kẹp trong mặt phẳng ngang (từ hàm cố định sang hàm di động)
        self._home_ee = None            # vị trí đầu tay lúc ở home
        # Hệ số làm mượt theo từng phase (chỉ áp cho 5 khớp tay, kẹp không qua đây): alpha càng lớn thì
        # lệnh khớp bám đích IK càng nhanh; 1.0 = không làm mượt, lệnh = đích IK ngay.
        self._phase_smoothing = {
            0: 0.8,   # nhấc lên: nhanh
            1: 0.8,   # tới phía trên hộp: nhanh
            2: 0.7,   # hạ xuống pre-grasp: vừa
            3: 0.7,   # hạ xuống kẹp: vừa
            4: 1.0,   # đóng kẹp: tức thì
            5: 0.5,   # nhấc hộp: chậm hơn (và bị giới hạn LIFT_STEP_LIMIT mỗi bước)
            6: 0.5,   # chuyển hộp tới bát: chậm hơn (cũng bị giới hạn LIFT_STEP_LIMIT)
            7: 0.5,   # hạ vào bát: chậm hơn (cũng bị giới hạn LIFT_STEP_LIMIT)
            8: 1.0,   # mở kẹp: tức thì
            9: 1.0,   # rút lên: không làm mượt
            10: 0.8,  # về home: khá nhanh (đích là góc khớp cố định, lệnh nhảy tới 0.8 rad/bước)
        }

        # Các độ cao (z trong hệ tọa độ world, đơn vị mét)
        self._safe_z = self.CARRY_Z   # độ cao an toàn để di chuyển ngang (đi tới hộp và mang hộp), tránh va vật thể
        self._pre_grasp_z = 0.46   # độ cao chuẩn bị kẹp
        self._grasp_z = self.GRASP_Z   # độ cao kẹp: đầu tay ngang tâm hộp
        self._place_z = self.PLACE_Z   # độ cao thả hộp vào bát

    def _apply_object_profile(self):
        """Cấu hình kẹp theo VẬT cần gắp (loại và cỡ). Hộp 3 cm: giữ nguyên mọi thông số đã chỉnh.
        - Cỡ: điểm ee lệch khỏi tâm vật GRASP_SIDE_OFFSET = 7 mm + nửa cỡ (22 mm cho vật 3 cm: tâm vật nằm sát mặt hàm cố định), và độ cao kẹp
          bám tâm vật (hộp: thấp hơn tâm 5 mm; 0,43 m ứng với hộp 3 cm), nên đổi cỡ không làm hàm cố định đè lên vật.
        - Bóng: không có mặt phẳng nên không căn roll theo mặt hộp, và kẹp ở đúng đường kính (đầu tay ngang tâm bóng, thấp hơn 1 mm): ở độ cao của
          hộp hàm di động chạm bóng ở nửa trên và ép bóng xuống, nên khi nhấc bóng trượt ra. Đo 43 seed bóng hỏng + 60 seed đạt: căn roll tắt + nâng 4 mm
          -> sửa 39/43, không hỏng thêm."""
        kind, size = getattr(self.env, "target_type", "box"), float(getattr(self.env, "target_size", 0.03))
        if kind != "sphere" and abs(size - 0.03) < 1e-6:
            return
        self.GRASP_SIDE_OFFSET = 0.007 + size / 2
        if kind != "sphere" or self._profile_alt:               # hộp (cỡ khác 3 cm) hoặc cấu hình hộp khi cấu hình bóng đã thử mà hỏng
            self.GRASP_ALIGN = type(self).GRASP_ALIGN
            self._grasp_z = 0.42 + size / 2 - 0.005
            self._pre_grasp_z = self._grasp_z + 0.03
            return
        self.GRASP_ALIGN = False
        self._grasp_z = 0.42 + size / 2 - 0.001
        self._pre_grasp_z = self._grasp_z + 0.03

    def _switch_object_profile(self):
        """Kẹp bóng hỏng (không kẹp được hoặc tuột khi nhấc): lần thử kế tiếp đổi sang cấu hình kia (bóng <-> hộp). Vài seed (khớp 1 sát giới hạn,
        sát đế) chỉ đạt với căn roll và độ cao của hộp, số còn lại chỉ đạt với cấu hình bóng; thử lần lượt cả hai thay vì chọn một."""
        if getattr(self.env, "target_type", "box") == "sphere":
            self._profile_alt = not self._profile_alt
            self._apply_object_profile()
            self._plan = None
    def reset(self):
        """Gọi mỗi đầu episode: đưa policy về phase 0."""
        self._profile_alt = False
        self._rest_wait = 0
        self._apply_object_profile()
        self._phase = 0
        self._phase_step = 0
        self._retries = 0
        self._last_finger = None
        self._gripper_target = 1.0
        self._current_joint_targets = None
        self._smoothed_action = None
        self._home_ee = self.env.get_ee_pos()
        self._roll_hold = None
        self._regrasps = 0
        self._place_retries = 0
        self._place_retract_target = None
        self._place_hold_action = None
        self._lost_steps = 0
        self._prev_grip = -1.0
        self._settle_wait = 0
        self._still = 0
        self._best_dist, self._stall, self.last_exit, self._p1_z, self._lift_z = 0.0, 0, None, None, None
        self._lift_xy0, self._lift_z0 = None, None
        self._bowl_hits, self._roll_flips, self._roll_pref, self._roll_stage, self._roll_z = 0, 0, None, None, None
        self._jaw_stuck = 0
        self._align_steps = 0
        self._pred_key,self._roll_turn_target, self._pred_changed, self._preturn_key, self._cam_ok_key = None, None, False, None, None
        self._far_roll = None
        self._via_done = False
        self._far_fix_key, self._far_fix_roll = None, None
        self._pan_key, self._pan_roll = None, None
        self._drag, self._drag_n, self._drag_goal, self._dragged = None, 0, None, False
        self._drag_goal_after = None
        self._plan = None
        self._tcp_goal = None
        # Hộp nằm ngoài giới hạn góc hoặc bán kính gắp: không thử gắp, về home ngay (phase 10) thay vì chờ hết hạn từng phase
        self.unreachable = bool(box_sampling.is_unreachable(self.env.get_box_pos()[:2]))
        bx = self.env.get_box_pos()
        if self.FAR_ROLL_R and math.hypot(bx[0] - config.PICK_PAN_XY[0], bx[1] - config.PICK_PAN_XY[1]) >= self.FAR_ROLL_R:
            self._roll_pref = math.radians(self.FAR_ROLL_DEG)
            self._far_roll = self._roll_pref
        if self.unreachable:
            self._phase = 10

    def select_action(self, observation: dict) -> np.ndarray:
        """Hàm chính: nhìn tình hình hiện tại, trả về action (7 số trong [-1, 1]).

        Policy viết tay này không dùng `observation` mà hỏi thẳng env vị trí hộp và bát
        ("gian lận" so với policy AI, vì AI chỉ được nhìn ảnh camera và góc khớp).
        """
        self.motion_target = None
        self.last_exit = None                                  # đặt lại mỗi bước; _move_toward ghi lý do nếu phase kết thúc ở bước này
        if self._place_hold_action is not None:
            return self._place_hold_action.copy()

        box_pos = self.env.get_box_pos()
        bowl_pos = self.env.get_bowl_pos()
        ee_pos = self.env.get_ee_pos()

        # Tầm gắp trên bàn chỉ áp dụng cho hộp chưa được giữ.
        if self._phase <= 4 and not self.env._grasped and box_sampling.is_unreachable(box_pos[:2]):
            self._abort_unreachable_pick()

        # Hộp tuột khi đang mang (nằm xa ee > 5 cm vài bước liền): quay lại phase 1 để kẹp lại (tối đa 2 lần)
        if 5 <= self._phase <= 7 and self._drag is None:              # đang kéo/nhả hộp (_drag_box): hộp không cần bám tay
            self._lost_steps = self._lost_steps + 1 if np.linalg.norm(box_pos - ee_pos) > 0.05 else 0
            if self._lost_steps > 3 and box_sampling.is_unreachable(box_pos[:2]):
                self._abort_unreachable_pick()
            elif self._lost_steps > 3 and self._regrasps < 2:
                self._regrasps += 1
                self._switch_object_profile()
                self._place_retract_target = None
                self._lost_steps = 0
                self._phase, self._phase_step = 1, 0
                self._roll_hold = None
                self._last_finger = None

        if self.REACTIVE_ROLL and self._phase in (2, 3, 4) and not self.env._grasped:
            # chỉ đếm khi tay chạm bát MÀ hàm kẹp chưa chạm hộp: hàm đã chạm hộp nghĩa là việc kẹp đang tiến triển, đừng cắt ngang
            self._bowl_hits = self._bowl_hits + 1 if (self._arm_touches_bowl() and not self._jaw_touches_box()) else 0
            if self._bowl_hits >= self.BOWL_HIT_STEPS and self._roll_flips < self.MAX_ROLL_FLIPS:
                self._flip_roll()
            # hàm cố định đè lên đỉnh hộp: đầu tay không hạ tới đáy được (phase 3 chờ hết hạn). Đổi roll để hàm lệch sang bên khác
            stuck = self._phase == 3 and ee_pos[2] > self._grasp_z + 0.012 and self._fixed_jaw_on_box()
            self._jaw_stuck = self._jaw_stuck + 1 if stuck else 0
            if self._jaw_stuck >= self.JAW_STUCK_STEPS and self._roll_flips < self.MAX_ROLL_FLIPS:
                self._jaw_stuck = 0
                self._flip_roll()

        match self._phase:
            case 0:
                self._lift_from_home(ee_pos)
            case 1:
                self._approach_box(box_pos, ee_pos)
            case 2:
                self._lower_to_box(ee_pos, self._pre_grasp_z)
            case 3:
                self._lower_to_box(ee_pos, self._grasp_z)
            case 4:
                self._close_gripper()
            case 5:
                self._lift_box(ee_pos)
            case 6:
                if not self._carry_via(bowl_pos, ee_pos):
                    self._move_box_to_bowl(box_pos, bowl_pos, ee_pos, self._safe_z, 0.02)
            case 7:
                self._lower_into_bowl(box_pos, bowl_pos, ee_pos)
                if self._place_hold_action is not None:
                    return self._place_hold_action.copy()
            case 8:
                self._release_box()
            case 9:
                self._retract_from_bowl(box_pos, bowl_pos, ee_pos)
            case 10:
                self._return_home()
            case _:
                return self._build_action_from_current()

        return self._build_action()

    def _lift_from_home(self, ee_pos):
        # Nhấc thẳng lên từ home (giữ nguyên x, y; chỉ đổi z)
        self._gripper_target = 1.0
        target = np.array([self._home_ee[0], self._home_ee[1], self._safe_z])
        if self.STAGED_LIFT and self._phase_step < self.STAGED_LIFT_MAX_STEPS and (self.STAGED_LIFT_R_MAX is None or self._box_pan_r() < self.STAGED_LIFT_R_MAX):
            q = self._solve_ik(target)
            now = self.data.qpos[self._qpos_adr]
            if abs(float(now[2] - q[2])) > math.radians(self.STAGED_LIFT_ELBOW_DEG):
                # giai đoạn 1: khuỷu và cổ tay tới góc đích, vai (khớp 2) giữ ở tư thế home; xong mới nâng vai
                q[1] = config.HOME_QPOS_RAD[1]
                self._current_joint_targets = q
                self.motion_target = (target.copy(), 0.02)
                self._phase_step += 1
                return
        self._move_toward(target, ee_pos, threshold=0.02)

    def _approach_box(self, box_pos, ee_pos):
        if self._roll_stage is not None:
            self._recover_roll(ee_pos)
            return
        # Di chuyển ngang tới phía trên hộp ở độ cao an toàn
        if self.FAR_ROLL_EARLY and self._far_roll is not None and self._roll_hold is None:
            self._roll_hold = self._far_roll                  # xoay roll song song lúc đi ra; không áp ở phase 0 (nhấc từ home: khoá roll làm IK không tới được home)
        if self._phase_step == 0 or self._p1_z is None:
            self._p1_z = self._reachable_z(box_pos[:2]) if self.PHASE1_ADAPTIVE_Z else self._safe_z
        target = np.array([box_pos[0], box_pos[1], self._p1_z])
        self._gripper_target = 1.0
        self._move_toward(target, ee_pos, threshold=0.025)
        if self._phase == 2 and self.PREDICT_BOWL_ROLL and self.PREDICT_PRETURN:
            self._maybe_preturn(box_pos, ee_pos)

    def _rest_hold(self, ee_pos):
        """Bóng đang lăn (bị hàm đẩy hoặc vừa va chạm): nâng tay lên độ cao lơ lửng ngay trên chỗ đang đứng và ĐỢI bóng đứng yên rồi mới hạ gắp
        (hạ tiếp khi bóng còn lăn thì hàm đuổi theo và đẩy bóng đi xa hơn). Chỉ áp cho bóng; tối đa REST_WAIT_MAX bước cho cả ván. Trả True nếu bước này dành cho việc đợi."""
        if (self.REST_SPEED is None or getattr(self.env, "target_type", "box") != "sphere" or self._rest_wait >= self.REST_WAIT_MAX):
            return False
        adr = self.env._box_dof_adr
        if float(np.linalg.norm(self.data.qvel[adr:adr + 3])) <= self.REST_SPEED:
            return False
        self._rest_wait += 1
        if self._phase == 3:                                      # đang hạ: quay lại phase 2 (lơ lửng) để hạ lại từ trên
            self._phase, self._phase_step, self._plan = 2, 0, None
        hz = self._pre_grasp_z + self.HOVER_ABOVE
        self._gripper_target = 1.0
        self._current_joint_targets = self._solve_ik(np.array([ee_pos[0], ee_pos[1], max(float(ee_pos[2]), hz)]))
        if self._roll_hold is not None:
            self._current_joint_targets[4] = self._roll_hold
        return True

    def _lower_to_box(self, ee_pos, height):
        if self._rest_hold(ee_pos):
            return
        if self.PLANNED_DESCENT and not self._dragged and self._box_pan_r() < self.PLAN_R_MAX:
            if self._phase == 2:
                self._descent_hover(ee_pos)
            else:
                self._descent_drop(ee_pos, height)
            return
        # Đường hạ cũ (hộp xa / sau khi kéo hộp): đích kẹp tính lại mỗi bước, hạ chéo từ phía trên tâm hộp
        self._plan = None
        target = self._grasp_target(height)
        self._gripper_target = 1.0
        self._align_steps = 0
        self._move_toward(target, ee_pos, threshold=0.025)

    def _descent_hover(self, ee_pos):
        """Phase 2 của đường hạ có kế hoạch: đi ngang ở độ cao lơ lửng tới NGAY TRÊN điểm kẹp G, roll khoá theo kế hoạch.
        Lập lại kế hoạch mỗi bước (tay còn ở trên cao, an toàn) để roll hội tụ theo tư thế thật. Sang phase 3 khi đã tới sát (< HOVER_NEAR),
        hoặc ngừng tiến HOVER_STALL bước mà cách < HOVER_FAR, hoặc ngừng tiến HOVER_STALL_ANY bước (điểm lơ lửng ngoài tầm, vd hộp sát đế);
        và roll đã xoay xong (lệch < PLAN_ROLL_SETTLE_DEG, chờ tối đa PLAN_ROLL_WAIT bước)."""
        self._make_descent_plan()
        p = self._plan
        self._roll_hold, self._clamp_dir = p["roll"], p["clamp"]
        self._gripper_target = 1.0
        self._align_steps = 0
        H = np.array([p["G"][0], p["G"][1], p["hz"]])
        step = self._phase_step + 1
        self._move_toward(H, ee_pos, threshold=self.HOVER_NEAR)
        roll_ok = abs(float(self.data.qpos[self._qpos_adr[4]]) - p["roll"]) < math.radians(self.PLAN_ROLL_SETTLE_DEG)
        if self._phase == 3:
            if not roll_ok and step <= self.PLAN_ROLL_WAIT:              # tới sát nhưng roll chưa xoay xong: đứng chờ ở trên, chưa hạ
                self._phase, self._phase_step, self.last_exit = 2, step, None
            return
        if roll_ok or self._phase_step > self.PLAN_ROLL_WAIT:
            if float(np.linalg.norm(ee_pos - H)) < self.HOVER_FAR and self._stall >= self.HOVER_STALL:
                self.last_exit = "arrived"
                self._advance_phase()
            elif self._stall >= self.HOVER_STALL_ANY:
                self.last_exit = "stall"
                self._advance_phase()

    def _descent_drop(self, ee_pos, height):
        """Phase 3 của đường hạ có kế hoạch: hạ THẲNG đứng xuống G với roll đã chốt; đóng kẹp (phase 4) khi cách độ cao kẹp < 8 mm,
        hoặc ngừng tiến 4 bước mà cách < 2,5 cm (đường cũ dừng ở 2,5 cm nên có lúc khép kẹp gần đỉnh hộp)."""
        if self._plan is None:
            self._make_descent_plan()
        p = self._plan
        self._roll_hold, self._clamp_dir = p["roll"], p["clamp"]
        self._gripper_target = 1.0
        G = np.array([p["G"][0], p["G"][1], height])
        self._move_toward(G, ee_pos, threshold=0.008)
        if self._phase == 3 and float(np.linalg.norm(ee_pos - G)) < 0.025 and self._stall >= 4:
            self.last_exit = "arrived"
            self._advance_phase()

    def _make_descent_plan(self):
        """Chọn roll và điểm kẹp G cho đường hạ thẳng: roll của _choose_roll nếu đường hạ sạch (xem _descent_clearance), nếu không thì roll ± k·90°
        gần nhất mà đường hạ sạch; không roll nào sạch thì giữ roll của _choose_roll."""
        box = self.env.get_box_pos()
        hz = max(self._pre_grasp_z + 0.015, min(self._pre_grasp_z + self.HOVER_ABOVE, self._p1_z if self._p1_z is not None else self._safe_z))
        roll0, _ = self._choose_roll(self._solve_ik(np.array([box[0], box[1], self._pre_grasp_z])))
        lo, hi = self.model.jnt_range[self._joint_ids[4]]
        cands = [r for r in [roll0] + [roll0 + k * math.pi / 2 for k in (1, -1, 2, -2)] if lo + 0.05 <= r <= hi - 0.05]
        res = [(r,) + self._descent_clearance(box, r, hz) for r in cands]
        ok = [x for x in res if x[1]]
        pick = ok[0] if ok else res[0]
        self._plan = {"roll": pick[0], "G": pick[2], "clamp": pick[3], "hz": hz}

    def _descent_clearance(self, box, r, hz):
        """Đo đường hạ thẳng từ độ cao hz xuống GRASP_Z với roll r (PLAN_SAMPLES điểm, IK có giới hạn khớp, hàm mở hết cỡ): hàm-hộp >= JAW_BOX_MARGIN,
        camera-bàn/vai >= FAR_FIXED_CAM, sai số với tới <= PLAN_REACH_ERR, gần bát thì hàm-bát >= PLAN_BOWL_MARGIN. Trả (sạch?, G, hướng kẹp)."""
        saved, self._phase = self._phase, 3
        try:
            above = np.array([box[0], box[1], self._grasp_z])
            q = self._solve_ik(above, roll=r); q[4] = r
            c = self._clamp_direction(q)
            G = above - self.GRASP_SIDE_OFFSET * np.append(c[:2], 0.0)
            near_bowl = np.linalg.norm(box[:2] - self.env.get_bowl_pos()[:2]) <= 0.20
            worst = 1e9
            for z in np.linspace(hz, self._grasp_z, self.PLAN_SAMPLES):
                tgt = np.array([G[0], G[1], z])
                q = self._solve_ik(tgt, roll=r, limits=True); q[4] = r
                err = float(np.linalg.norm(self._ik_data.site_xpos[self._ee_site] - tgt))
                s = min(self._jaw_box_distance(q) / self.JAW_BOX_MARGIN, self._cam_geom_distance(q) / self.FAR_FIXED_CAM,
                        self.PLAN_REACH_ERR / max(err, 1e-6))
                if near_bowl:
                    s = min(s, self._bowl_geom_distance(q, config.GRIPPER_OPEN_RAD) / self.PLAN_BOWL_MARGIN)
                worst = min(worst, s)
            return worst >= 1.0, G, c
        finally:
            self._phase = saved

    def _close_gripper(self):
        # Đóng kẹp. Kết thúc SỚM khi env báo hộp đã được kẹp VÀ ngón đã ngừng khép.
        # Quá 30 bước (~1 giây) mà chưa kẹp được thì quay lại phase 2 để căn lại và hạ xuống lần nữa
        # (đích tính lại từ vị trí hộp hiện tại), tối đa self._max_retries lần.
        if self._align_roll_before_close():
            return
        self._gripper_target = -1.0
        self._phase_step += 1
        fo = self.data.qpos[self.model.jnt_qposadr[self.env._gripper_jnt]]
        settled = self._last_finger is not None and abs(fo - self._last_finger) < 0.004
        self._last_finger = fo
        if self.env._grasped and settled:
            self._last_finger = None
            self._advance_phase()
        elif self._phase_step > 30:
            self._last_finger = None
            if self._retries < self._max_retries:
                self._retries += 1
                self._switch_object_profile()
                self._phase, self._phase_step = 2, 0     # thử lại từ phase 2
                self._roll_hold = None                   # hộp có thể đã xoay: chọn lại roll
            else:
                self._advance_phase()                    # hết lượt thử: đi tiếp như cũ

    def _align_roll_before_close(self):
        """Đầu phase 4 (hàm còn mở, chưa chạm hộp): trục kẹp ở tư thế THẬT lệch mặt hộp quá GRASP_ALIGN_TOL_DEG thì giữ kẹp mở, chỉ xoay
        wrist_roll tới góc song song (các khớp khác giữ đích IK cũ). Trả True nếu bước này dành cho việc chỉnh roll.
        Hộp xa (roll chốt ở _far_fixed_roll) cố ý không canh theo yaw hộp nên bỏ qua.
        Mỗi bước tìm trong ± GRASP_ALIGN_MAX_DEG quanh roll VỪA chỉnh nên qua nhiều bước roll có thể trôi xa hơn (seed 1011: 52° sau 9 bước, hộp
        bị hàm gạt nghiêng nên yaw nhảy). Đã thử chốt quanh roll ban đầu và bỏ qua khi hộp nghiêng > 10°: 1011 vẫn hỏng, 1010 hỏng thêm (272/274)."""
        if (not self.GRASP_ALIGN or self._phase_step > 0 or self._align_steps >= self.GRASP_ALIGN_STEPS
                or self._far_fix_key is not None or self._current_joint_targets is None or self._jaw_touches_box()):
            return False
        q = self.data.qpos[self._qpos_adr].copy()
        yaw = self._box_yaw()

        def dev(r):
            qq = q.copy()
            qq[4] = r
            c = self._clamp_direction(qq)
            return (math.atan2(c[1], c[0]) - yaw + math.pi / 4) % (math.pi / 2) - math.pi / 4

        if abs(dev(q[4])) <= math.radians(self.GRASP_ALIGN_TOL_DEG):
            return False
        lo, hi = self.model.jnt_range[self._joint_ids[4]]
        base = self._roll_hold if self._roll_hold is not None else q[4]
        cands = [base + d for d in np.radians(np.arange(-self.GRASP_ALIGN_MAX_DEG, self.GRASP_ALIGN_MAX_DEG + 0.01, 0.5))]
        cands = [r for r in cands if lo + 0.02 <= r <= hi - 0.02]
        if not cands:
            return False
        self._roll_hold = min(cands, key=lambda r: abs(dev(r)))
        self._current_joint_targets[4] = self._roll_hold
        self._gripper_target = 1.0
        self._align_steps += 1
        return True

    def _box_yaw(self):
        """Góc yaw của hộp (rad) quanh trục z."""
        qb = self.data.xquat[self.env._box_body]
        return math.atan2(2 * (qb[0] * qb[3] + qb[1] * qb[2]), 1 - 2 * (qb[2] ** 2 + qb[3] ** 2))

    def _lift_box(self, ee_pos):
        # Nhấc thẳng lên (giữ nguyên x, y của tay). Sát đế thì độ cao an toàn nằm ngoài tầm với: nhấc tới độ cao tới được
        if self._drag_box(ee_pos):
            return
        if self._phase_step == 0 or self._lift_z is None or self._lift_xy0 is None:
            self._lift_z = self._reachable_z(ee_pos[:2], z_lo=self._grasp_z + 0.04) if self.PHASE5_ADAPTIVE_Z else self._safe_z
            self._lift_xy0, self._lift_z0 = np.array(ee_pos[:2], dtype=float), float(ee_pos[2])
        target = np.array([*self._lift_xy(self._lift_xy0, self._lift_z0, ee_pos), self._lift_z])
        self._gripper_target = -1.0
        self._move_toward(target, ee_pos, threshold=0.02)

    def _lift_xy(self, xy0, z0, cur):
        """Điểm (x, y) đích khi nhấc: giữ nguyên x, y; hộp sát đế (r < LIFT_MIN_R) thì dời dần RA NGOÀI theo hướng bán kính tới r = LIFT_MIN_R
        khi hộp đã lên LIFT_EXT_RISE trở đi. Độ cao an toàn gần như ngoài tầm với của hộp sát đế nên nhấc thẳng làm IK kéo tay vào phía đế,
        khớp 3 gập lại và camera cọ vào vai; phase 6 vốn duỗi tay ra tới bát, nên duỗi sớm trong lúc nhấc không tốn thêm đường."""
        cur_xy, z = np.asarray(cur[:2], dtype=float), float(cur[2])
        if self.LIFT_MIN_R is None:
            return cur_xy                                          # như cũ: x, y đích = x, y hiện tại của tay
        pan = np.array(config.PICK_PAN_XY)
        v = xy0 - pan
        r0 = float(np.linalg.norm(v))
        if r0 < 1e-6 or r0 >= self.LIFT_MIN_R:
            return cur_xy
        ramp = min(1.0, max(0.0, (z - z0) / self.LIFT_EXT_RISE))
        return pan + v / r0 * (r0 + (self.LIFT_MIN_R - r0) * ramp)

    def _box_pan_r(self):
        """Khoảng cách ngang (m) từ hộp tới trục khớp 1."""
        b = self.env.get_box_pos()
        return math.hypot(b[0] - config.PICK_PAN_XY[0], b[1] - config.PICK_PAN_XY[1])

    def _drag_box(self, ee_pos):
        """Đầu phase 5, hộp xa (r >= DRAG_R): không nhấc mà KÉO TRƯỢT hộp trên bàn theo hướng bán kính về phía trục khớp 1 tới r = DRAG_TARGET,
        nhả kẹp, nhấc tay lên DRAG_RAISE rồi gắp lại từ phase 2. Mỗi ván kéo tối đa một lần. Trả True nếu bước này dành cho việc kéo.
        Ba bước: "pull" (kẹp chặt, kéo), "release" (mở kẹp, giữ yên), "raise" (nhấc tay khỏi hộp)."""
        if (self.DRAG_FAR and self._drag is None and not self._dragged and self._phase_step == 0
                and self._box_pan_r() >= self.DRAG_R):
            pan = np.array(config.PICK_PAN_XY)
            v = ee_pos[:2] - pan
            r = float(np.linalg.norm(v))
            r_t = r - (self._box_pan_r() - self.DRAG_TARGET)                 # tay dời đúng quãng hộp cần dời
            self._drag_goal = np.array([*(pan + v * (r_t / r)), ee_pos[2] + self.DRAG_LIFT])
            self._drag, self._dragged = "pull", True
            if getattr(self.env, "target_type", "box") == "sphere" and self.DRAG_UP_SPHERE:
                self._drag = "up"                                              # bóng: nhấc lên trước, kéo về sau
                self._drag_goal_after = self._drag_goal + np.array([0, 0, self.DRAG_UP_SPHERE - self.DRAG_LIFT])
                self._drag_goal = np.array([ee_pos[0], ee_pos[1], ee_pos[2] + self.DRAG_UP_SPHERE])
        if self._drag == "up":
            self._gripper_target = -1.0
            self._move_toward(self._drag_goal, ee_pos, threshold=0.006, timeout_steps=20)
            if self._phase != 5 or self.last_exit:
                self._phase, self._phase_step, self.last_exit = 5, 0, None
                self._drag, self._drag_goal = "pull", self._drag_goal_after
            return True
        if self._drag == "pull":
            self._gripper_target = -1.0
            self._move_toward(self._drag_goal, ee_pos, threshold=0.006, timeout_steps=120)
            if self._phase != 5 or self.last_exit:                            # tới nơi hoặc hết hạn: ở lại phase 5, sang nhả
                self._phase, self._phase_step = 5, 0
                if self._drag_goal_after is not None and self.env._grasped:   # bóng đã được mang trên không và vẫn đang kẹp: nhấc tiếp luôn, không nhả gắp lại
                    self._drag, self._drag_goal_after, self.last_exit = None, None, None
                    return True
                self._drag, self._drag_n = "release", 0
            return True
        if self._drag == "release":
            self._gripper_target = 1.0
            self._drag_n += 1
            if self._drag_n >= self.DRAG_RELEASE_STEPS:
                self._drag = "raise"
                self._drag_goal = np.array([ee_pos[0], ee_pos[1], ee_pos[2] + self.DRAG_RAISE])
                self._phase_step = 0
            return True
        if self._drag == "raise":
            self._gripper_target = 1.0
            self._move_toward(self._drag_goal, ee_pos, threshold=0.008, timeout_steps=40)
            if self._phase != 5 or self.last_exit:                            # đã nhấc khỏi hộp: gắp lại từ phase 2 ở chỗ mới
                self._drag = None
                self._phase, self._phase_step = 2, 0
                self._roll_hold, self._last_finger, self._retries = None, None, 0
                self._far_fix_key, self._far_fix_roll = None, None             # hộp đã vào gần: chọn roll lại từ đầu
                self._best_dist, self._stall, self.last_exit = 0.0, 0, None
            return True
        return False

    def _carry_via(self, bowl_pos, ee_pos):
        """Phase 6: bát nằm lệch quá VIA_ANGLE_DEG so với tay (quanh khớp 1) thì trước hết đi tới điểm trung gian ở độ cao an toàn, nằm giữa hai góc
        (đi qua hướng thẳng phía trước, trong giới hạn khớp 1). Trả True nếu bước này dành cho điểm trung gian."""
        if self._via_done:
            return False
        pan = np.array(config.PICK_PAN_XY)
        a_ee = math.degrees(math.atan2(ee_pos[0] - pan[0], ee_pos[1] - pan[1]))
        a_bw = math.degrees(math.atan2(bowl_pos[0] - pan[0], bowl_pos[1] - pan[1]))
        if abs(a_ee - a_bw) <= self.VIA_ANGLE_DEG:
            self._via_done = True
            return False
        mid = math.radians((a_ee + a_bw) / 2)
        r = float(np.clip((np.linalg.norm(ee_pos[:2] - pan) + np.linalg.norm(bowl_pos[:2] - pan)) / 2, 0.20, 0.30))
        via = np.array([pan[0] + r * math.sin(mid), pan[1] + r * math.cos(mid), self._safe_z])
        self._gripper_target = -1.0
        self._move_toward(via, ee_pos, threshold=0.04)
        if self._phase != 6:                                       # tới nơi hoặc hết hạn: ở lại phase 6, đi tiếp tới bát
            self._phase, self._phase_step = 6, 0
            self._best_dist, self._stall = 0.0, 0
            self.last_exit = None
            self._via_done = True
        return True

    def _move_box_to_bowl(self, box_pos, bowl_pos, ee_pos, height, threshold, *, advance_on_failure=True):
        target = np.array([bowl_pos[0], bowl_pos[1], height])
        # Căn tâm hộp với tâm bát khi hộp nằm lệch khỏi đầu kẹp.
        target[:2] -= np.clip((box_pos - ee_pos)[:2], -0.04, 0.04)
        self._gripper_target = -1.0
        self._move_toward(target, ee_pos, threshold=threshold, advance_on_failure=advance_on_failure)

    def _lower_into_bowl(self, box_pos, bowl_pos, ee_pos):
        """Phase 7 chỉ cho phép mở kẹp khi đã tới đích thả."""
        if self._place_retract_target is not None:
            self._move_toward(
                self._place_retract_target, ee_pos, threshold=0.02,
                advance_on_failure=False, timeout_steps=self.PLACE_RETRACT_TIMEOUT,
            )
            if self.last_exit == "arrived":
                # _move_toward resets the counters. Retry from above the bowl.
                self._phase = 6
                self._place_retract_target = None
                self.last_exit = "place_retry"
            elif self.last_exit:
                self._place_hold_action = self._build_action_from_current()
                self.last_exit = "place_hold"
            return

        self._move_box_to_bowl(
            box_pos, bowl_pos, ee_pos, self._place_z, self.PLACE_ARRIVE_THRESHOLD,
            advance_on_failure=False,
        )
        if self.last_exit != "timeout":
            return

        if not self.env._grasped or self._place_retries >= self.PLACE_RECOVERY_LIMIT:
            self._place_hold_action = self._build_action_from_current()
            self.last_exit = "place_hold"
            return

        self._place_retries += 1
        self._place_retract_target = np.array([ee_pos[0], ee_pos[1], self._safe_z])
        self._phase_step, self._best_dist, self._stall = 0, 0.0, 0
        self.last_exit = "place_retract"
        # Stop issuing the lowering goal as soon as we start the retry.
        self.motion_target = (self._place_retract_target.copy(), 0.02)
        self._current_joint_targets = self._solve_ik(self._place_retract_target)
        if self._roll_hold is not None:
            self._current_joint_targets[4] = self._roll_hold

    def _release_box(self):
        # Mở kẹp vừa phải. Xong khi hộp đã rời tay và hàm đã mở đủ rộng để rút lên không kéo theo hộp
        # (không chờ hộp nằm yên ở đây: việc đó làm sau khi rút lên, ở cuối phase 9). Tối đa 20 bước.
        self._gripper_target = self.RELEASE_OPEN_CMD
        self._phase_step += 1
        jaw = self.data.qpos[self.model.jnt_qposadr[self.env._gripper_jnt]]
        if (not self.env._grasped and jaw >= self.RELEASE_CLEAR_RAD) or self._phase_step > 20:
            self._settle_wait = 0
            self._still = 0
            self._advance_phase()

    def _retract_from_bowl(self, box_pos, bowl_pos, ee_pos):
        # Rút tay lên khỏi bát (vẫn mở vừa, để ngón không quét vào thành bát khi rút)
        target = np.array([bowl_pos[0], bowl_pos[1], self._safe_z])
        self._gripper_target = self.RELEASE_OPEN_CMD
        self._move_toward(target, ee_pos, threshold=0.025)
        if self._phase == 10:                       # đã rút lên tới nơi: xem hộp đang ở đâu rồi mới đi tiếp
            self._after_retract(box_pos, bowl_pos)

    def _abort_unreachable_pick(self):
        self.unreachable = True
        self.last_exit = "unreachable"
        self._phase, self._phase_step = 10, 0
        self._place_retract_target = None

    def _return_home(self):
        # Về tư thế GẤP GỌN ban đầu: đi thẳng trong không gian khớp tới HOME_QPOS_RAD (không qua IK để chắc chắn gập đúng kiểu)
        self._gripper_target = -1.0     # cất tay thì khép kẹp lại, giống tư thế ban đầu
        self._current_joint_targets = np.array(config.HOME_QPOS_RAD[:5], dtype=float)
        self._phase_step += 1
        err = np.max(np.abs(self.data.qpos[self._qpos_adr] - self._current_joint_targets))
        if err < 0.05 or self._phase_step > self.PHASE_TIMEOUT_STEPS:
            self._advance_phase()

    def _after_retract(self, box_pos, bowl_pos):
        """Cuối phase 9 (tay đã rút lên trên bát): chờ hộp nằm yên, rồi quyết định.

        - hộp nằm yên trong bát            -> phase 10 (về home)
        - hộp nằm ngoài bát, trong tầm gắp -> quay lại phase 1 gắp lại (dùng chung giới hạn _regrasps của phase 5-7)
        - hộp ngoài tầm gắp, hoặc hết lượt  -> phase 10
        Đang chờ thì giữ ở phase 9 (tay vẫn ở trên bát), tối đa SETTLE_WAIT_MAX bước.
        """
        if box_sampling.is_unreachable(box_pos[:2]):
            self._abort_unreachable_pick()
            return

        speed = float(np.linalg.norm(self.data.qvel[self.env._box_dof_adr:self.env._box_dof_adr + 3]))
        self._still = self._still + 1 if speed < config.SETTLED_SPEED else 0
        if self._still < self.SETTLE_STILL_STEPS and self._settle_wait < self.SETTLE_WAIT_MAX:   # hộp phải yên vài bước liền, không chỉ một khoảnh khắc
            self._settle_wait += 1
            self._phase = 9
            return
        # "trong bát" chỉ xét theo chiều ngang: hộp nằm trong lòng bát nhưng còn hơi nghiêng/gác lên thành (z cao hơn vài mm)
        # sẽ tự nằm xuống, KHÔNG được coi là rơi ra ngoài (đo thấy gây gắp lại sai ở một số seed)
        in_bowl = np.linalg.norm(box_pos[:2] - bowl_pos[:2]) < self.env.bowl_inner
        pan = self.data.xpos[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "shoulder")]
        r = float(np.linalg.norm(box_pos[:2] - pan[:2]))
        on_table = box_pos[2] < config.BOX_DEFAULT_POS[2] + 0.015      # nằm trên bàn, không gác trên miệng bát
        if (not in_bowl and on_table and self.REGRASP_R_MIN <= r <= self.REGRASP_R_MAX and self._regrasps < 2):
            self._regrasps += 1
            self._phase, self._phase_step = 1, 0
            self._roll_hold = None
            self._last_finger = None
            self._lost_steps = 0

    def _reachable_z(self, xy, z_lo=None):
        """Độ cao đích (phase 1 và phase 5): cao nhất trong [z_lo, độ cao an toàn] mà IK (có giới hạn khớp) còn tới được trong 4 mm.

        Hộp sát đế (r < 15 cm) thì độ cao an toàn CARRY_Z nằm ngoài tầm với của cánh tay: IK không giới hạn khớp từng báo "tới được", nhưng
        khớp thật dừng ở giới hạn nên đầu tay kẹt cách đích ~4 cm cho tới khi hết hạn phase. Hạ độ cao đích xuống mức tới được thì phase 1 kết
        thúc ngay khi tới nơi. Nếu không độ cao nào tới được thì lấy độ cao có sai số IK nhỏ nhất."""
        z_lo = self._pre_grasp_z + 0.02 if z_lo is None else z_lo
        best = (1e9, self._safe_z)
        z = self._safe_z
        while z >= z_lo - 1e-9:
            target = np.array([xy[0], xy[1], z])
            self._solve_ik(target, limits=True)      # chỉ để THĂM DÒ tầm với: có kẹp giới hạn khớp nên sai số IK phản ánh đúng thực tế
            err = float(np.linalg.norm(self._ik_data.site_xpos[self._ee_site] - target))
            if err < 0.004:
                return z
            if err < best[0]:
                best = (err, z)
            z -= 0.01
        return best[1]

    def annotation(self):
        """Chú thích hành vi của bước vừa chọn, để ghi cạnh dữ liệu huấn luyện (không phải action).

        phase: phase hiện tại; recovery: 0 bình thường, 1 đang nhấc lên khỏi bát, 2 đang xoay roll tại chỗ (đều là PHỤC HỒI sau khi chạm bát);
        roll_flips: đã đổi roll vì chạm bát bao nhiêu lần; bowl_avoid: 1 nếu roll được chọn sẵn từ đầu để tránh bát (không có va chạm xảy ra)."""
        stage = {None: 0, "lift": 1, "turn": 2}[self._roll_stage]
        return {"phase": int(self._phase), "recovery": stage, "roll_flips": int(self._roll_flips), "bowl_avoid": int(self._pred_changed),
                "unreachable": int(self.unreachable)}

    def _arm_touches_bowl(self):
        """True nếu có khâu tay (kể cả hàm kẹp) đang chạm bát."""
        bowl = self.env._bowl_body
        for c in self.data.contact[:self.data.ncon]:
            b1, b2 = self.model.geom_bodyid[c.geom1], self.model.geom_bodyid[c.geom2]
            if (b1 == bowl and b2 in self._arm_body_ids) or (b2 == bowl and b1 in self._arm_body_ids):
                return True
        return False

    def _fixed_jaw_on_box(self):
        """True nếu hàm CỐ ĐỊNH (thân gripper) đang chạm hộp."""
        box = self.env._box_body
        jaw = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "gripper")
        for c in self.data.contact[:self.data.ncon]:
            b1, b2 = self.model.geom_bodyid[c.geom1], self.model.geom_bodyid[c.geom2]
            if (b1 == box and b2 == jaw) or (b2 == box and b1 == jaw):
                return True
        return False

    def _jaw_touches_box(self):
        """True nếu hàm kẹp (cố định hoặc di động) đang chạm hộp."""
        box = self.env._box_body
        jaws = {mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, n) for n in ("gripper", "moving_jaw_so101_v1")}
        for c in self.data.contact[:self.data.ncon]:
            b1, b2 = self.model.geom_bodyid[c.geom1], self.model.geom_bodyid[c.geom2]
            if (b1 == box and b2 in jaws) or (b2 == box and b1 in jaws):
                return True
        return False

    def _flip_roll(self):
        """Tay chạm bát lúc hạ xuống kẹp: nhấc THẲNG lên một đoạn ngắn (chỉ cao hơn miệng bát vài cm, không phải về độ cao an toàn như
        phase 0), xoay wrist_roll TẠI CHỖ ở độ cao đó, rồi hạ xuống kẹp lại (phase 2). Việc xoay thực hiện ở `_recover_roll`.

        Hộp vuông nên trục kẹp song song mặt hộp đúng với 4 góc roll cách nhau 90°. Đổi góc đổi phía hàm di động (đang mở rộng)
        quay về, để nó không quét vào thành bát. Thử lần lượt xoay 180°, rồi -180°, +90°, -90° (bỏ phương án ngoài giới hạn khớp)."""
        lo, hi = self.model.jnt_range[self._joint_ids[4]]
        r0 = self._roll_hold if self._roll_hold is not None else float(self.data.qpos[self._qpos_adr[4]])
        cands = [r0 + off for off in self.ROLL_FLIP_OFFSETS]
        cands = [r for r in cands if lo + 0.05 <= r <= hi - 0.05]
        if cands:
            self._roll_pref = cands[self._roll_flips % len(cands)]
        self._roll_flips += 1
        self._roll_hold = None
        self._last_finger = None
        self._bowl_hits = 0
        self._roll_stage, self._roll_z = "lift", None
        self._roll_turn_target = r0 + (self._roll_pref - r0) / 2 if (self.ROLL_TURN_MODE == "half" and self._roll_pref is not None) else self._roll_pref
        self._phase, self._phase_step = 1, 0
        self._best_dist, self._stall = 0.0, 0

    def _clear_height(self, xy):
        """Độ cao đầu tay để xoay roll mà không quét vào bát: cao hơn miệng bát một chút, lấy mức đầu tiên IK với tới được."""
        rim = self.env.get_bowl_pos()[2] + self.env.bowl_height
        for margin in self.ROLL_CLEAR_MARGINS:
            z = rim + margin
            target = np.array([xy[0], xy[1], z])
            self._solve_ik(target, limits=True)
            if float(np.linalg.norm(self._ik_data.site_xpos[self._ee_site] - target)) < 0.004:
                return z
        return self._reachable_z(xy, z_lo=rim + 0.01)

    def _recover_roll(self, ee_pos):
        """Hai bước sau khi tay chạm bát (xem _flip_roll): "lift" nhấc thẳng lên, "turn" xoay wrist_roll tại chỗ; xong thì sang phase 2."""
        self._gripper_target = 1.0
        if self._roll_z is None:
            self._roll_z = self._clear_height(ee_pos[:2])
        if self._roll_stage == "lift":
            target = np.array([ee_pos[0], ee_pos[1], self._roll_z]) if self._phase_step == 0 else self._roll_target
            self._roll_target = target
            self._move_toward(target, ee_pos, threshold=0.012)
            if self._phase != 1:                                   # đã lên tới độ cao: ở lại phase 1 và chuyển sang xoay roll
                self._phase, self._phase_step = 1, 0
                self._best_dist, self._stall = 0.0, 0
                self._roll_stage = "turn"
                self.last_exit = None
            return
        # "turn": giữ nguyên vị trí đầu tay, chỉ xoay wrist_roll tới roll ưu tiên
        target = np.array(self._roll_target)
        q = self._solve_ik(target)
        if self._roll_turn_target is not None:
            q[4] = self._roll_turn_target
        self._current_joint_targets = q
        self.motion_target = (target.copy(), 0.02)
        self._phase_step += 1
        roll_err = abs(float(self.data.qpos[self._qpos_adr[4]]) - (self._roll_turn_target if self._roll_turn_target is not None else 0.0))
        if self._roll_turn_target is None or roll_err < 0.06 or self._phase_step > self.ROLL_TURN_STEPS:
            self._roll_stage, self._roll_z = None, None
            self._advance_phase()                                  # sang phase 2: hạ xuống kẹp bằng roll mới

    def _advance_phase(self):
        """Chuyển sang phase kế tiếp và đặt lại bộ đếm bước của phase."""
        self._phase += 1
        self._phase_step = 0
        self._best_dist, self._stall = 0.0, 0

    def _grasp_target(self, z):
        """Đích của ee khi hạ xuống hộp: chọn roll theo mặt hộp, rồi lệch khỏi tâm hộp GRASP_SIDE_OFFSET theo trục kẹp."""
        box = self.env.get_box_pos()
        above = np.array([box[0], box[1], z])
        roll, clamp = self._choose_roll(self._solve_ik(above))
        self._roll_hold, self._clamp_dir = roll, clamp
        return above - self.GRASP_SIDE_OFFSET * np.append(clamp[:2], 0.0)

    def _clamp_direction(self, q5):
        """Hướng trục kẹp (ngang, vector đơn vị) ở tư thế khớp q5: hàm di động xoay quanh trục bản lề a, trục kẹp = a × xuống."""
        ik = self._ik_data
        ik.qpos[:] = self.data.qpos[:]
        ik.qpos[self._qpos_adr] = q5
        mujoco.mj_kinematics(self.model, ik)
        c = np.cross(ik.xaxis[self._gripper_joint], [0.0, 0.0, -1.0])
        return c / (np.linalg.norm(c) + 1e-9)

    def _choose_roll(self, q5):
        """Chọn góc wrist_roll để trục kẹp song song với một cặp mặt hộp (gần góc hiện tại nhất, để khỏi xoay vòng).

        Quét roll trong giới hạn khớp; với mỗi góc tính hướng trục kẹp rồi lệch của nó so với mặt hộp (mod 90 độ).
        """
        if self.FAR_FIXED_ROLL and self._roll_stage is None and self._roll_flips == 0:      # đã phải đổi roll (hàm đè hộp/chạm bát) thì theo roll mới, không ép lại roll cố định
            fixed = self._far_fixed_roll(q5)
            if fixed is not None:
                return fixed
        box_yaw = self._box_yaw()
        lo, hi = self.model.jnt_range[self._joint_ids[4]]
        cur = self._roll_pref if self._roll_pref is not None else (self._roll_hold if self._roll_hold is not None else self.data.qpos[self._qpos_adr[4]])
        best = None
        for r in np.radians(np.arange(-157.0, 163.1, 2.0)):
            if not lo <= r <= hi:
                continue
            q = np.array(q5, dtype=float)
            q[4] = r
            c = self._clamp_direction(q)
            dev = (math.atan2(c[1], c[0]) - box_yaw + math.pi / 4) % (math.pi / 2) - math.pi / 4
            score = abs(dev) + (self.FAR_PREF_WEIGHT if self._far_roll is not None else 0.02) * abs(r - cur)
            if best is None or score < best[0]:
                best = (score, r, c)
        roll, clamp = best[1], best[2]
        if self.PREDICT_BOWL_ROLL and self._roll_stage is None:
            roll2 = self._predict_roll(roll, q5)
            if roll2 is not None and abs(roll2 - roll) > 1e-6:
                roll, clamp = roll2, self._clamp_direction(np.where(np.arange(5) == 4, roll2, np.asarray(q5, dtype=float)))
        if self.PAN_LIMIT_ROLL:
            roll3 = self._pan_limit_roll(roll)
            if roll3 is not None:
                roll, clamp = roll3, self._clamp_direction(np.where(np.arange(5) == 4, roll3, np.asarray(q5, dtype=float)))
        return roll, clamp

    def _grasp_reach(self, box, r):
        """Đo tầm với khi kẹp hộp ở `box` với roll `r`: (sai số IK có giới hạn khớp lớn nhất, khoảng cách camera cổ tay nhỏ nhất tới bàn/vai)
        ở tư thế chuẩn bị kẹp và tư thế kẹp (m). Tính trên bản sao IK với trục kẹp thẳng xuống như phase 3."""
        saved, self._phase = self._phase, 3
        try:
            err, cam = 0.0, 1e9
            for z in (self._pre_grasp_z, self._grasp_z):
                above = np.array([box[0], box[1], z])
                q = self._solve_ik(above, roll=r); q[4] = r
                c = self._clamp_direction(q)
                tgt = above - self.GRASP_SIDE_OFFSET * np.append(c[:2], 0.0)
                q = self._solve_ik(tgt, roll=r); q[4] = r
                cam = min(cam, self._cam_geom_distance(q))
                self._solve_ik(tgt, roll=r, limits=True)
                err = max(err, float(np.linalg.norm(self._ik_data.site_xpos[self._ee_site] - tgt)))
            return err, cam
        finally:
            self._phase = saved

    def _pan_limit_roll(self, roll):
        """Hộp sát giới hạn khớp 1: roll `roll` đặt đích kẹp ngoài tầm (sai số > PAN_LIMIT_TRIGGER) thì trả roll song song mặt hộp khác (roll ± k·90°)
        gần nhất mà sai số <= PAN_LIMIT_ERR_OK và camera cách bàn/vai >= FAR_FIXED_CAM; None nếu không cần đổi hoặc không có roll nào đạt.
        Không áp cho roll đã chốt ở _far_fixed_roll (đã tự xét tầm khớp 1). Kết quả nhớ theo vị trí hộp và roll gốc."""
        box = self.env.get_box_pos()
        pan = config.PICK_PAN_XY
        if abs(math.degrees(math.atan2(box[0] - pan[0], box[1] - pan[1]))) < self.PAN_LIMIT_NEAR_DEG:
            return None
        if self._far_fix_roll is not None and abs(roll - self._far_fix_roll) < 1e-6:
            return None
        key = (round(float(box[0]), 3), round(float(box[1]), 3), round(float(roll), 2))
        if self._pan_key != key:
            self._pan_key, self._pan_roll = key, None
            err, cam = self._grasp_reach(box, roll)
            if err > self.PAN_LIMIT_TRIGGER or cam < self.FAR_FIXED_CAM:
                lo, hi = self.model.jnt_range[self._joint_ids[4]]
                ok = []
                for k in (-3, -2, -1, 1, 2, 3):
                    r = roll + k * math.pi / 2
                    if not lo + 0.05 <= r <= hi - 0.05:
                        continue
                    e, c = self._grasp_reach(box, r)
                    if e <= self.PAN_LIMIT_ERR_OK and c >= self.FAR_FIXED_CAM:
                        ok.append((abs(k), r))
                if ok:
                    self._pan_roll = min(ok)[1]
        return self._pan_roll

    def _far_fixed_roll(self, q5):
        """Hộp r >= FAR_FIXED_R: không canh roll theo yaw hộp mà thử lần lượt các roll trong FAR_ROLL_ORDER (-87° đầu tiên), lấy roll đầu tiên mà ở
        tư thế chuẩn bị kẹp và tư thế kẹp: đích kẹp trong tầm khớp 1 (sai số IK có giới hạn <= FAR_FIXED_ERR) và camera cổ tay cách bàn/vai
        >= FAR_FIXED_CAM (roll dương lớn làm camera cắm xuống bàn). Không roll nào đạt thì lấy roll ít vi phạm nhất. Roll được chốt một lần (chọn lại nếu hộp bị đẩy > 2 cm). Trả (roll, hướng kẹp), hoặc None nếu hộp chưa đủ xa."""
        box = self.env.get_box_pos()
        if math.hypot(box[0] - config.PICK_PAN_XY[0], box[1] - config.PICK_PAN_XY[1]) < self.FAR_FIXED_R:
            return None
        key = (float(box[0]), float(box[1]))
        if self._far_fix_key is None or math.hypot(key[0] - self._far_fix_key[0], key[1] - self._far_fix_key[1]) > 0.02:
            lo, hi = self.model.jnt_range[self._joint_ids[4]]
            best = None
            for deg in self.FAR_ROLL_ORDER:
                r = math.radians(deg)
                if not lo + 0.05 <= r <= hi - 0.05:
                    continue
                err, cam = self._grasp_reach(box, r)
                bad = max(err / self.FAR_FIXED_ERR, self.FAR_FIXED_CAM / max(cam, 1e-4))
                if best is None or bad < best[1]:
                    best = (r, bad)
                if bad <= 1.0:
                    break
            self._far_fix_roll = best[0] if best else math.radians(self.FAR_ROLL_DEG)
            self._far_fix_key = key
        r = self._far_fix_roll
        q = np.array(q5, dtype=float); q[4] = r
        return r, self._clamp_direction(q)

    def _maybe_preturn(self, box_pos, ee_pos):
        """Vừa tới phía trên hộp (hết phase 1). Nếu roll dự đoán để tránh bát khác roll hiện tại nhiều thì xoay roll TẠI CHỖ ở độ cao lơ lửng
        trước, rồi mới hạ (phase 2). Nếu để khớp xoay trong lúc hạ gần bát thì hàm kẹp mở rộng quét qua các góc roll va vào bát."""
        key = (round(float(box_pos[0]), 3), round(float(box_pos[1]), 3))
        if self._preturn_key == key:
            return
        self._preturn_key = key
        saved = self._phase
        self._phase = 3
        try:
            self._roll_hold = None
            self._grasp_target(self._pre_grasp_z)                      # chọn roll (có dự đoán tránh bát)
        finally:
            self._phase = saved
        roll = self._roll_hold
        self._roll_hold = None
        cur = float(self.data.qpos[self._qpos_adr[4]])
        if self._roll_pref is None or abs(roll - cur) < 0.35:
            return
        self._roll_stage, self._roll_turn_target = "turn", roll
        self._roll_target = np.array(ee_pos, dtype=float)
        self._roll_z = float(ee_pos[2])
        self._phase, self._phase_step = 1, 0
        self._best_dist, self._stall = 0.0, 0

    def _bowl_geom_distance(self, q5, grip):
        """Khoảng cách nhỏ nhất (m) từ các khâu tay + hàm kẹp tới bát khi tay ở tư thế q5 và hàm mở góc `grip` (trên bản sao IK)."""
        ik = self._ik_data
        ik.qpos[:] = self.data.qpos[:]
        ik.qpos[self._qpos_adr] = q5
        ik.qpos[self.model.jnt_qposadr[self.env._gripper_jnt]] = grip
        self._fk(ik)
        if not hasattr(self, "_clear_pairs"):
            bowl = self.env._bowl_body
            arm = [g for g in range(self.model.ngeom) if self.model.geom_bodyid[g] in self._arm_body_ids and self.model.geom_contype[g] != 0]
            walls = [g for g in range(self.model.ngeom) if self.model.geom_bodyid[g] == bowl]
            self._clear_pairs = [(a, w) for a in arm for w in walls]
        fromto = np.zeros(6)
        return min(mujoco.mj_geomDistance(self.model, ik, a, w, 0.06, fromto) for a, w in self._clear_pairs)

    def _cam_geom_distance(self, q5):
        """Khoảng cách nhỏ nhất (m) từ camera cổ tay (wrist_cam_arm) tới vai, cánh tay trên, đế và mặt bàn khi tay ở tư thế q5 (trên bản sao IK).
        Đo hình học trên bản sao IK (camera có va chạm trong mô phỏng thật, nhưng bản sao không chạy va chạm); âm = camera xuyên vào khâu đó."""
        ik = self._ik_data
        ik.qpos[:] = self.data.qpos[:]
        ik.qpos[self._qpos_adr] = q5
        self._fk(ik)
        if not hasattr(self, "_cam_pairs"):
            bid = lambda n: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, n)
            geoms = lambda b: [g for g in range(self.model.ngeom) if self.model.geom_bodyid[g] == b]
            cam = geoms(bid("wrist_cam_arm"))
            walls = [g for n in ("base", "shoulder", "upper_arm", "table") for g in geoms(bid(n))]
            self._cam_pairs = [(a, w) for a in cam for w in walls]
        fromto = np.zeros(6)
        return min(mujoco.mj_geomDistance(self.model, ik, a, w, 0.06, fromto) for a, w in self._cam_pairs)

    def _lift_cam_clearance(self, q_grasp, roll):
        """Mô phỏng HÌNH HỌC phase 5 từ tư thế kẹp: mỗi bước giải IK tới (x, y hiện tại, độ cao an toàn) rồi đưa lệnh khớp tới đó theo đúng
        làm mượt và giới hạn tốc độ của phase 5; trả khoảng cách camera nhỏ nhất (m) tới vai / cánh tay trên / đế / bàn dọc đường nhấc.
        Phải mô phỏng từng bước vì tay không đi thẳng đứng: hộp sát đế thì độ cao an toàn gần như ngoài tầm, IK kéo tay vào phía đế và camera
        quay vào trong sẽ quét vai (seed 4: roll +81° chạm vai 35 bước). Không chạy vật lý, chỉ động học nên rẻ."""
        saved = self._phase
        self._phase = 5
        ik = self._ik_data
        q = np.array(q_grasp, dtype=float)
        smoothed = self._to_action(q)
        alpha = self._phase_smoothing.get(5, 0.7)
        lo, hi = self.env._ctrl_ranges[:5].T
        clear = 1e9
        xy0 = z0 = None
        try:
            for _ in range(self.LIFT_SIM_STEPS):
                ik.qpos[:] = self.data.qpos[:]
                ik.qpos[self._qpos_adr] = q
                self._fk(ik)
                ee = ik.site_xpos[self._ee_site].copy()
                if xy0 is None:
                    xy0, z0 = ee[:2].copy(), float(ee[2])
                if ee[2] >= self._safe_z - 0.02:
                    break
                q_ik = self._solve_ik(np.array([*self._lift_xy(xy0, z0, ee), self._safe_z]), roll=roll, q0=q)
                step = np.clip(alpha * (self._to_action(q_ik) - smoothed), -self.LIFT_STEP_LIMIT, self.LIFT_STEP_LIMIT)
                smoothed = smoothed + step
                q = (smoothed + 1.0) / 2.0 * (hi - lo) + lo
                q[4] = roll
                clear = min(clear, self._cam_geom_distance(q))
        finally:
            self._phase = saved
        return clear

    def _jaw_box_distance(self, q5):
        """Khoảng cách nhỏ nhất (m) từ các khâu tay + hàm kẹp mở hết cỡ tới HỘP ở tư thế q5 (trên bản sao IK); âm = hàm cắm vào hộp, tức là khi hạ sẽ đẩy hộp."""
        ik = self._ik_data
        ik.qpos[:] = self.data.qpos[:]
        ik.qpos[self._qpos_adr] = q5
        ik.qpos[self.model.jnt_qposadr[self.env._gripper_jnt]] = config.GRIPPER_OPEN_RAD
        self._fk(ik)
        if not hasattr(self, "_jawbox_pairs"):
            box = [g for g in range(self.model.ngeom) if self.model.geom_bodyid[g] == self.env._box_body]
            arm = [g for g in range(self.model.ngeom) if self.model.geom_bodyid[g] in self._arm_body_ids and self.model.geom_contype[g] != 0]
            self._jawbox_pairs = [(a, b) for a in arm for b in box]
        fromto = np.zeros(6)
        return min(mujoco.mj_geomDistance(self.model, ik, a, b, 0.06, fromto) for a, b in self._jawbox_pairs)

    def _predict_roll(self, roll, q5):
        """Bước 1 (trước khi hạ): với bốn góc roll song song mặt hộp (cách nhau 90°), tính khoảng cách hàm kẹp mở hết cỡ tới bát ở tư thế
        chuẩn bị kẹp và tư thế kẹp; chọn góc đủ xa bát (>= BOWL_CLEARANCE_MARGIN) và gần roll hiện tại nhất; nếu không góc nào đủ xa thì
        lấy góc xa nhất. Chỉ tính khi hộp nằm gần bát (< 20 cm); tính lại khi hộp bị đẩy đi. Trả về None nếu không cần đổi."""
        box = self.env.get_box_pos()
        bowl = self.env.get_bowl_pos()
        need_bowl = np.linalg.norm(box[:2] - bowl[:2]) <= 0.20
        box_r = float(np.linalg.norm(box[:2] - np.array(config.PICK_PAN_XY)))
        need_cam = self.CAM_AVOID
        need_jaw = self.CAM_AVOID and box_r < self.JAW_AVOID_R
        if not need_bowl and not need_cam and not need_jaw:
            return None
        key = (round(float(box[0]), 3), round(float(box[1]), 3))
        if self._pred_key == key and self._roll_pref is not None:
            return self._roll_pref
        if not need_bowl and self._cam_ok_key == key:
            return None
        lo, hi = self.model.jnt_range[self._joint_ids[4]]
        cur = float(self.data.qpos[self._qpos_adr[4]])
        saved_phase = self._phase
        self._phase = 3                                           # IK với trục kẹp thẳng xuống, như lúc thật hạ xuống kẹp
        results = []
        try:
            for k in range(-2, 3):
                r = roll + k * math.pi / 2
                if not lo + 0.05 <= r <= hi - 0.05:
                    continue
                clear, cam_clear, jaw_clear, lift_clear = 1e9, 1e9, 1e9, 1e9
                for z in (self._pre_grasp_z, self._grasp_z):
                    above = np.array([box[0], box[1], z])
                    q = self._solve_ik(above, roll=r)
                    q[4] = r
                    c = self._clamp_direction(q)
                    tgt = above - self.GRASP_SIDE_OFFSET * np.append(c[:2], 0.0)
                    q = self._solve_ik(tgt, roll=r)
                    q[4] = r
                    if need_bowl:
                        clear = min(clear, self._bowl_geom_distance(q, config.GRIPPER_OPEN_RAD))
                    if need_cam:
                        cam_clear = min(cam_clear, self._cam_geom_distance(q))
                    if need_jaw:
                        jaw_clear = min(jaw_clear, self._jaw_box_distance(q))
                if need_cam and self.CAM_LIFT_CHECK:                   # đường nhấc: IK tới độ cao an toàn BẮT ĐẦU TỪ tư thế kẹp (như lúc nhấc thật), rồi nội suy khớp
                    lift_clear = self._lift_cam_clearance(q.copy(), r)   # q: tư thế kẹp (cuối vòng for trên)
                if self.PREDICT_PATH:                                  # đường xoay: từ roll hiện tại tới r, ở độ cao lơ lửng phía trên hộp
                    hover = self._p1_z if self._p1_z is not None else self._pre_grasp_z + 0.02
                    for f in (0.25, 0.5, 0.75, 1.0):
                        ri = cur + (r - cur) * f
                        above = np.array([box[0], box[1], hover])
                        q = self._solve_ik(above, roll=ri)
                        q[4] = ri
                        if need_bowl:
                            clear = min(clear, self._bowl_geom_distance(q, config.GRIPPER_OPEN_RAD))
                        if need_cam:
                            cam_clear = min(cam_clear, self._cam_geom_distance(q))
                # điểm theo từng điều kiện (đơn vị = mức tối thiểu, >= 1 là đạt): bát, camera (cứng), hàm kẹp-hộp (mềm)
                hard = min(clear / self.BOWL_CLEARANCE_MARGIN if need_bowl else 1e9, cam_clear / self.CAM_CLEARANCE_MARGIN if need_cam else 1e9,
                           lift_clear / self.CAM_LIFT_MARGIN if need_cam and self.CAM_LIFT_CHECK else 1e9)
                jaw_s = jaw_clear / self.JAW_BOX_MARGIN if need_jaw else 1e9
                results.append((r, min(hard, jaw_s), hard, jaw_s))
        finally:
            self._phase = saved_phase
        if not results:
            return None
        ok = [x for x in results if x[1] >= 1.0]
        if not need_bowl and any(abs(x[0] - roll) < 1e-6 for x in ok):   # chỉ vướng camera mà roll song song mặt hộp đã đủ xa: giữ nguyên, không ghi _roll_pref
            self._cam_ok_key = key                                        # (đổi roll không cần thiết làm hộp rơi nghiêng; ghi _roll_pref còn đổi cả logic xoay-trước)
            return None
        if ok:
            pick = min(ok, key=lambda x: abs(x[0] - cur))
        else:                                                         # không roll nào đạt cả ba: giữ điều kiện cứng (bát, camera), trong đó lấy hàm kẹp xa hộp nhất
            hard_ok = [x for x in results if x[2] >= 1.0]
            pick = max(hard_ok, key=lambda x: x[3]) if hard_ok else max(results, key=lambda x: x[2])
        self._pred_key = key
        self._roll_pref = pick[0]
        self._pred_changed = abs(pick[0] - roll) > 1e-6
        return pick[0]

    def _solve_ik(self, target_pos, roll=None, limits=None, q0=None):
        """IK: tìm góc khớp (radian) của 5 khớp để đầu tay (ee_site) tới target_pos.

        Chạy trên bản sao dữ liệu `_ik_data` nên không làm robot thật cử động. Chỉ tính HÌNH HỌC:
        không biết va chạm nhưng có kẹp góc khớp trong giới hạn thật. Nhiệm vụ chính là vị trí (x, y, z) của đầu tay.
        Hai nhiệm vụ phụ (đều "mềm"):

        1. Ở các phase TOOL_DOWN_PHASES: trục kẹp (từ cổ tay tới đầu tay) chúc thẳng xuống. Cộng thêm 3 hàng
           vào bài toán, nhân với TOOL_DOWN_WEIGHT:  [J_vị_trí ; w·J_hướng] · dq = [lỗi_vị_trí ; w·lỗi_hướng]
        2. Phần tự do còn lại (không gian null) dùng để kéo về tư thế vòm (xem POSTURE_*):

            dq = J⁺ · e  +  (I − J⁺J) · K · mask · (q_pref − q)

        Số hạng sau nằm trong không gian null của J nên (xấp xỉ bậc nhất) không làm đầu tay lệch khỏi đích.

        Khi có giới hạn khớp (`use_lim`): khớp đang ở giới hạn mà bước dq còn đẩy ra ngoài thì bị KHOÁ (bỏ khỏi J) và giải lại, để phần
        chuyển động còn thiếu dồn sang các khớp khác. Chỉ cắt góc (np.clip) thì vòng sau J vẫn dồn chuyển động vào khớp đã kẹt.
        """
        if roll is None:
            roll = self._roll_hold
        use_lim = (self.IK_RESPECT_LIMITS or self._phase in self.IK_LIMIT_PHASES) if limits is None else limits
        ik = self._ik_data
        # Bắt đầu từ tư thế thật hiện tại, nên lời giải "gần" tư thế đang có
        ik.qpos[:] = self.data.qpos[:]
        if q0 is not None:                                         # bắt đầu từ tư thế cho trước (vd tư thế kẹp) thay vì tư thế thật hiện tại
            ik.qpos[self._qpos_adr] = q0
        ik.qvel[:] = 0
        free0 = np.ones(5)
        if roll is not None:                                       # khoá wrist_roll ở góc `roll`: IK chỉ dùng 4 khớp còn lại
            ik.qpos[self._qpos_adr[4]] = roll
            free0[4] = 0.0
        if self._phase in self.PAN_BEARING_PHASES:
            # Khớp 1 bắt đầu từ HƯỚNG của đích quanh trục khớp 1 nếu lệch xa góc hiện tại. IK tuyến tính từ pan +109° tới bát ở -45° chọn
            # chiều xoay ra giới hạn +110°, khoá khớp 1 rồi gập 2-3-4 xuống và kẹt (seed 14, phase 6 hết hạn)
            pan = config.PICK_PAN_XY
            b = math.atan2(target_pos[0] - pan[0], target_pos[1] - pan[1])
            if abs(b - float(self.data.qpos[self._qpos_adr[0]])) > math.radians(self.PAN_BEARING_DEG):
                ik.qpos[self._qpos_adr[0]] = float(np.clip(b, self._jnt_lo[0], self._jnt_hi[0]))
        if use_lim:
            ik.qpos[self._qpos_adr] = np.clip(ik.qpos[self._qpos_adr], self._jnt_lo, self._jnt_hi)
        self._fk(ik)

        jacp = np.zeros((3, self.model.nv))
        tool_down = self._phase in self.TOOL_DOWN_PHASES
        jacr = np.zeros_like(jacp) if tool_down else None
        eye5 = np.eye(5)
        q_pref = np.radians(self.POSTURE_PREF_DEG)
        posture_k = self.POSTURE_GAIN * np.array(self.POSTURE_MASK, dtype=float)   # K · mask
        damping = self.IK_DAMPING ** 2 * np.eye(6 if tool_down else 3)
        down = np.array([0.0, 0.0, -1.0])
        for _ in range(self.IK_ITERATIONS):
            ee = ik.site_xpos[self._ee_site]
            error = target_pos - ee                                # còn cách đích bao xa (vector 3D)
            if tool_down:
                axis = ee - ik.xpos[self._wrist_roll_body]             # trục kẹp: từ cổ tay tới đầu tay
                axis /= np.linalg.norm(axis)
                axis_error = down - axis                               # còn lệch khỏi "thẳng xuống" bao nhiêu
            if np.linalg.norm(error) < self.IK_TOLERANCE and (not tool_down or np.linalg.norm(axis_error) < 0.02):
                break

            # Jacobian J (3 x 5): "xoay mỗi khớp một chút thì đầu tay dịch chuyển thế nào"
            mujoco.mj_jacSite(self.model, ik, jacp, jacr, self._ee_site)
            J = jacp[:, self._dof_ids]
            if tool_down:
                # xoay khớp làm trục kẹp đổi hướng:  d(axis) = ω × axis = −[axis]× · ω, với ω = Jr · dq
                J_axis = -self._skew(axis) @ jacr[:, self._dof_ids]
                J_full = np.vstack([J, self.TOOL_DOWN_WEIGHT * J_axis])
                err_all = np.concatenate([error, self.TOOL_DOWN_WEIGHT * axis_error])
            else:
                J_full, err_all = J, error

            q = ik.qpos[self._qpos_adr]
            free = free0.copy()
            for _ in range(5):                                     # tối đa 5 lần khoá thêm khớp bão hoà
                # Damped least squares: dq nhỏ nhất sao cho J_all @ dq ≈ err_all.
                # Damping giúp ổn định khi robot gần tư thế "kỳ dị" (singularity), ví dụ khi 2-3-4 thẳng hàng.
                J_all = J_full * free[None, :]
                J_pinv = J_all.T @ np.linalg.inv(J_all @ J_all.T + damping)
                dq = J_pinv @ err_all * free
                # Nhiệm vụ phụ: kéo khớp về tư thế vòm, chỉ trong không gian null
                dq = dq + ((eye5 - J_pinv @ J_all) @ (posture_k * (q_pref - q))) * free
                if not use_lim:
                    break
                sat = (((q <= self._jnt_lo + 1e-4) & (dq < 0)) | ((q >= self._jnt_hi - 1e-4) & (dq > 0))) & (free > 0)
                if not sat.any():
                    break
                free[sat] = 0.0

            ik.qpos[self._qpos_adr] += self.IK_STEP_GAIN * dq
            if use_lim:
                ik.qpos[self._qpos_adr] = np.clip(ik.qpos[self._qpos_adr], self._jnt_lo, self._jnt_hi)   # IK phải tôn trọng giới hạn khớp thật
            self._fk(ik)

        return ik.qpos[self._qpos_adr].copy()

    def _fk(self, data):
        """Động học thuận cho IK: chỉ tính vị trí các link và jacobian (mj_kinematics + mj_comPos), không chạy va chạm và động lực học như
        mj_forward. IK gọi hàm này ~40 lần mỗi bước nên đây là chỗ tốn thời gian nhất của cả mô phỏng; kết quả IK không đổi."""
        mujoco.mj_kinematics(self.model, data)
        mujoco.mj_comPos(self.model, data)

    @staticmethod
    def _skew(a):
        """Ma trận phản đối xứng [a]× sao cho [a]× @ v = a × v."""
        return np.array([[0.0, -a[2], a[1]], [a[2], 0.0, -a[0]], [-a[1], a[0], 0.0]])

    def _move_toward(self, target_pos, ee_pos, threshold=0.02, *, advance_on_failure=True, timeout_steps=None):
        """Hướng đầu tay về target_pos trong bước này, và quyết định có sang phase kế không.

        Tham số:
            target_pos: đích 3D (x, y, z) của đầu tay, mét, do phase hiện tại tính (vd. phía trên hộp).
            ee_pos:     vị trí đầu tay LÚC ĐẦU BƯỚC này (chưa áp dụng action của bước này).
            threshold:  khoảng cách (mét) dưới mức này thì coi là "đã tới" và sang phase kế.

        Làm hai việc: (1) gọi IK để có góc khớp đích, lưu vào `_current_joint_targets` cho
        `_build_action` dùng; (2) đếm bước và chuyển phase khi tay đã tới gần đích HOẶC đã ở phase
        này quá PHASE_TIMEOUT_STEPS bước. Phase hay chạm timeout là dấu hiệu tay không tới được đích.

        Lưu ý: khoảng cách đo bằng `ee_pos` của đầu bước, nên "tới nơi" được nhận ra chậm 1 bước
        (không đáng kể).
        """
        self.motion_target = (target_pos.copy(), threshold)
        ik_target = target_pos
        far1 = self.TCP_STEP1 > 0 and self._phase == 1
        if (self.TCP_STEP > 0 and self._phase in (2, 3)) or far1:   # C: đích đầu tay tiến từng đoạn nhỏ
            if self._tcp_goal is None or self._phase_step == 0:
                self._tcp_goal = np.array(ee_pos, dtype=float)
            d = target_pos - self._tcp_goal
            n = float(np.linalg.norm(d))
            stp = self.TCP_STEP1 if far1 else self.TCP_STEP
            self._tcp_goal = target_pos.copy() if n <= stp else self._tcp_goal + d * (stp / n)
            ik_target = self._tcp_goal
        else:
            self._tcp_goal = None
        self._current_joint_targets = self._solve_ik(ik_target)
        if self._roll_hold is not None:                            # đã chọn roll thì giữ nguyên tới hết ván
            self._current_joint_targets[4] = self._roll_hold

        self._phase_step += 1
        dist = float(np.linalg.norm(ee_pos - target_pos))
        if self._phase_step == 1 or dist < self._best_dist - self.STALL_PROGRESS:
            self._best_dist, self._stall = dist, 0
        else:
            self._stall += 1
        cap = timeout_steps if timeout_steps is not None else self.PHASE_TIMEOUTS.get(self._phase, self.PHASE_TIMEOUT_STEPS)
        if dist < threshold:
            self.last_exit = "arrived"
        elif self._phase_step > cap:
            self.last_exit = "timeout"
        elif self._stall >= self.STALL_STEPS and self._phase in self.STALL_PHASES:
            self.last_exit = "stall"
        if self.last_exit == "arrived" or (advance_on_failure and self.last_exit):
            self._advance_phase()

    def _to_action(self, angles):
        """Đổi góc 5 khớp (radian) sang thang [-1, 1] của action, theo khoảng điều khiển của từng actuator."""
        lo, hi = self.env._ctrl_ranges[:5].T
        return 2.0 * (np.asarray(angles) - lo) / (hi - lo) - 1.0

    def _build_action(self) -> np.ndarray:
        """Đổi góc khớp đích (radian) thành action chuẩn hóa [-1, 1], rồi làm mượt."""
        raw = np.zeros(7, dtype=np.float32)
        if self._current_joint_targets is not None:
            raw[:5] = self._to_action(self._current_joint_targets)
        raw[5] = raw[6] = self._gripper_target
        raw = np.clip(raw, -1.0, 1.0)

        # Làm mượt theo hàm mũ (exponential smoothing) để tốc độ nhìn đều, không giật.
        # smoothed += alpha * (đích - smoothed); alpha càng lớn càng bám đích nhanh.
        alpha = self._phase_smoothing.get(self._phase, 0.7)
        if self._smoothed_action is None:
            self._smoothed_action = raw.copy()
        else:
            step = alpha * (raw[:5] - self._smoothed_action[:5])
            if 5 <= self._phase <= 7:                         # mang hộp: không đổi lệnh khớp quá nhanh
                step = np.clip(step, -self.LIFT_STEP_LIMIT, self.LIFT_STEP_LIMIT)
            self._smoothed_action[:5] += step
            # gripper: đóng ngay; MỞ thì tăng dần (tối đa RELEASE_RAMP mỗi bước) để hai hàm không bật hộp văng ra khi thả
            self._smoothed_action[5:] = raw[5:]
            self._smoothed_action[5] = min(raw[5], self._prev_grip + self.RELEASE_RAMP)
            self._smoothed_action[6] = self._smoothed_action[5]
        self._prev_grip = float(self._smoothed_action[5])
        return self._smoothed_action.copy()

    def _build_action_from_current(self) -> np.ndarray:
        """Tạo action giữ nguyên tư thế hiện tại (dùng khi đã xong phase 11)."""
        action = np.zeros(7, dtype=np.float32)
        action[:5] = self._to_action(self.data.qpos[self._qpos_adr])
        action[5] = action[6] = self._gripper_target
        return np.clip(action, -1.0, 1.0)
