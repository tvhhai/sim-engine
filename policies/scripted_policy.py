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
from .. import config


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
    PHASE5_ADAPTIVE_Z = False  # True: phase 5 cũng hạ độ cao nhấc xuống mức tới được. Đo 200 seed: 192-194/200 (xấu hơn), nên TẮT
    PHASE1_ADAPTIVE_Z = True   # phase 1 hạ độ cao đích xuống mức tới được
    BOWL_HIT_STEPS = 3         # tay chạm bát liền chừng này bước khi hạ xuống kẹp (phase 2-4) thì nhấc lên, xoay wrist_roll rồi kẹp lại
    ROLL_FLIP_OFFSETS = (math.pi, -math.pi, math.pi / 2, -math.pi / 2)   # thứ tự thử khi đổi roll (rad so với roll đang giữ)
    ROLL_CLEAR_MARGINS = (0.035, 0.025, 0.015)   # nhấc lên cao hơn miệng bát chừng này (m) trước khi xoay roll; lấy mức đầu tiên với tới được
    ROLL_TURN_STEPS = 60       # tối đa chừng này bước để xoay roll tại chỗ
    MAX_ROLL_FLIPS = 3         # số lần đổi roll tối đa trong một ván (thử lần lượt: xoay 180°, +90°, -90° so với roll đang giữ)
    STALL_STEPS = 20           # đầu tay không tiến gần đích thêm STALL_PROGRESS trong chừng này bước liền thì coi là KẸT, sang phase kế
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
        self._lift_z = None             # độ cao đích của phase 5 (nhấc hộp), cũng hạ thấp khi hộp sát đế
        self._bowl_hits = 0             # số bước liền tay đang chạm bát khi hạ xuống kẹp
        self._roll_flips = 0            # đã đổi roll bao nhiêu lần vì tay chạm bát
        self._roll_pref = None          # roll ưu tiên sau khi đổi (None = chọn roll gần roll hiện tại nhất)
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

    def reset(self):
        """Gọi mỗi đầu episode: đưa policy về phase 0."""
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
        self._bowl_hits, self._roll_flips, self._roll_pref, self._roll_stage, self._roll_z = 0, 0, None, None, None

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

        # Hộp tuột khi đang mang (nằm xa ee > 5 cm vài bước liền): quay lại phase 1 để kẹp lại (tối đa 2 lần)
        if 5 <= self._phase <= 7:
            self._lost_steps = self._lost_steps + 1 if np.linalg.norm(box_pos - ee_pos) > 0.05 else 0
            if self._lost_steps > 3 and self._regrasps < 2:
                self._regrasps += 1
                self._place_retract_target = None
                self._lost_steps = 0
                self._phase, self._phase_step = 1, 0
                self._roll_hold = None
                self._last_finger = None

        if self._phase in (2, 3, 4) and not self.env._grasped:
            # chỉ đếm khi tay chạm bát MÀ hàm kẹp chưa chạm hộp: hàm đã chạm hộp nghĩa là việc kẹp đang tiến triển, đừng cắt ngang
            self._bowl_hits = self._bowl_hits + 1 if (self._arm_touches_bowl() and not self._jaw_touches_box()) else 0
            if self._bowl_hits >= self.BOWL_HIT_STEPS and self._roll_flips < self.MAX_ROLL_FLIPS:
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
        self._move_toward(target, ee_pos, threshold=0.02)

    def _approach_box(self, box_pos, ee_pos):
        if self._roll_stage is not None:
            self._recover_roll(ee_pos)
            return
        # Di chuyển ngang tới phía trên hộp ở độ cao an toàn
        if self._phase_step == 0 or self._p1_z is None:
            self._p1_z = self._reachable_z(box_pos[:2]) if self.PHASE1_ADAPTIVE_Z else self._safe_z
        target = np.array([box_pos[0], box_pos[1], self._p1_z])
        self._gripper_target = 1.0
        self._move_toward(target, ee_pos, threshold=0.025)

    def _lower_to_box(self, ee_pos, height):
        target = self._grasp_target(height)
        self._gripper_target = 1.0
        self._move_toward(target, ee_pos, threshold=0.025)

    def _close_gripper(self):
        # Đóng kẹp. Kết thúc SỚM khi env báo hộp đã được kẹp VÀ ngón đã ngừng khép.
        # Quá 30 bước (~1 giây) mà chưa kẹp được thì quay lại phase 2 để căn lại và hạ xuống lần nữa
        # (đích tính lại từ vị trí hộp hiện tại), tối đa self._max_retries lần.
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
                self._phase, self._phase_step = 2, 0     # thử lại từ phase 2
                self._roll_hold = None                   # hộp có thể đã xoay: chọn lại roll
            else:
                self._advance_phase()                    # hết lượt thử: đi tiếp như cũ

    def _lift_box(self, ee_pos):
        # Nhấc thẳng lên (giữ nguyên x, y của tay). Sát đế thì độ cao an toàn nằm ngoài tầm với: nhấc tới độ cao tới được
        if self._phase_step == 0 or self._lift_z is None:
            self._lift_z = self._reachable_z(ee_pos[:2], z_lo=self._grasp_z + 0.04) if self.PHASE5_ADAPTIVE_Z else self._safe_z
        target = np.array([ee_pos[0], ee_pos[1], self._lift_z])
        self._gripper_target = -1.0
        self._move_toward(target, ee_pos, threshold=0.02)

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
        speed = float(np.linalg.norm(self.data.qvel[self.env._box_dof_adr:self.env._box_dof_adr + 3]))
        self._still = self._still + 1 if speed < config.SETTLED_SPEED else 0
        if self._still < self.SETTLE_STILL_STEPS and self._settle_wait < self.SETTLE_WAIT_MAX:   # hộp phải yên vài bước liền, không chỉ một khoảnh khắc
            self._settle_wait += 1
            self._phase = 9
            return
        # "trong bát" chỉ xét theo chiều ngang: hộp nằm trong lòng bát nhưng còn hơi nghiêng/gác lên thành (z cao hơn vài mm)
        # sẽ tự nằm xuống, KHÔNG được coi là rơi ra ngoài (đo thấy gây gắp lại sai ở một số seed)
        in_bowl = np.linalg.norm(box_pos[:2] - bowl_pos[:2]) < config.BOWL_INNER_RADIUS
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

    def _arm_touches_bowl(self):
        """True nếu có khâu tay (kể cả hàm kẹp) đang chạm bát."""
        bowl = self.env._bowl_body
        for c in self.data.contact[:self.data.ncon]:
            b1, b2 = self.model.geom_bodyid[c.geom1], self.model.geom_bodyid[c.geom2]
            if (b1 == bowl and b2 in self._arm_body_ids) or (b2 == bowl and b1 in self._arm_body_ids):
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
        self._phase, self._phase_step = 1, 0
        self._best_dist, self._stall = 0.0, 0

    def _clear_height(self, xy):
        """Độ cao đầu tay để xoay roll mà không quét vào bát: cao hơn miệng bát một chút, lấy mức đầu tiên IK với tới được."""
        rim = self.env.get_bowl_pos()[2] + config.BOWL_HEIGHT
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
        if self._roll_pref is not None:
            q[4] = self._roll_pref
        self._current_joint_targets = q
        self.motion_target = (target.copy(), 0.02)
        self._phase_step += 1
        roll_err = abs(float(self.data.qpos[self._qpos_adr[4]]) - (self._roll_pref if self._roll_pref is not None else 0.0))
        if self._roll_pref is None or roll_err < 0.06 or self._phase_step > self.ROLL_TURN_STEPS:
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
        qb = self.data.xquat[self.env._box_body]
        box_yaw = math.atan2(2 * (qb[0] * qb[3] + qb[1] * qb[2]), 1 - 2 * (qb[2] ** 2 + qb[3] ** 2))
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
            score = abs(dev) + 0.02 * abs(r - cur)
            if best is None or score < best[0]:
                best = (score, r, c)
        return best[1], best[2]

    def _solve_ik(self, target_pos, roll=None, limits=None):
        """IK: tìm góc khớp (radian) của 5 khớp để đầu tay (ee_site) tới target_pos.

        Chạy trên bản sao dữ liệu `_ik_data` nên không làm robot thật cử động. Chỉ tính HÌNH HỌC:
        không biết va chạm nhưng có kẹp góc khớp trong giới hạn thật. Nhiệm vụ chính là vị trí (x, y, z) của đầu tay.
        Hai nhiệm vụ phụ (đều "mềm"):

        1. Ở các phase TOOL_DOWN_PHASES: trục kẹp (từ cổ tay tới đầu tay) chúc thẳng xuống. Cộng thêm 3 hàng
           vào bài toán, nhân với TOOL_DOWN_WEIGHT:  [J_vị_trí ; w·J_hướng] · dq = [lỗi_vị_trí ; w·lỗi_hướng]
        2. Phần tự do còn lại (không gian null) dùng để kéo về tư thế vòm (xem POSTURE_*):

            dq = J⁺ · e  +  (I − J⁺J) · K · mask · (q_pref − q)

        Số hạng sau nằm trong không gian null của J nên (xấp xỉ bậc nhất) không làm đầu tay lệch khỏi đích.
        """
        if roll is None:
            roll = self._roll_hold
        ik = self._ik_data
        # Bắt đầu từ tư thế thật hiện tại, nên lời giải "gần" tư thế đang có
        ik.qpos[:] = self.data.qpos[:]
        ik.qvel[:] = 0
        free = np.ones(5)
        if roll is not None:                                       # khoá wrist_roll ở góc `roll`: IK chỉ dùng 4 khớp còn lại
            ik.qpos[self._qpos_adr[4]] = roll
            free[4] = 0.0
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
                J_all = np.vstack([J, self.TOOL_DOWN_WEIGHT * J_axis])
                err_all = np.concatenate([error, self.TOOL_DOWN_WEIGHT * axis_error])
            else:
                J_all, err_all = J, error

            # Damped least squares: dq nhỏ nhất sao cho J_all @ dq ≈ err_all.
            # Damping giúp ổn định khi robot gần tư thế "kỳ dị" (singularity), ví dụ khi 2-3-4 thẳng hàng.
            J_all = J_all * free[None, :]
            J_pinv = J_all.T @ np.linalg.inv(J_all @ J_all.T + damping)
            dq = J_pinv @ err_all * free

            # Nhiệm vụ phụ: kéo khớp về tư thế vòm, chỉ trong không gian null
            q = ik.qpos[self._qpos_adr]
            dq = dq + ((eye5 - J_pinv @ J_all) @ (posture_k * (q_pref - q))) * free

            ik.qpos[self._qpos_adr] += self.IK_STEP_GAIN * dq
            if (self.IK_RESPECT_LIMITS or self._phase in self.IK_LIMIT_PHASES) if limits is None else limits:
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
        self._current_joint_targets = self._solve_ik(target_pos)
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
