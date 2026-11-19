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
    PHASE_TIMEOUT_STEPS = 200  # ở một phase quá số bước này thì bỏ cuộc và sang phase kế

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
    GRASP_SIDE_OFFSET = 0.016
    CARRY_Z = 0.51            # độ cao đầu tay khi mang: tay kẹp thẳng đứng được ở độ cao này, miệng bát z 0.46 nên hộp (đáy ~0.49) vẫn cao hơn
    LIFT_STEP_LIMIT = 0.01    # phase 5-7: mỗi bước, lệnh khớp (thang -1..1) đổi tối đa chừng này, để hộp không bị giật
    PLACE_ARRIVE_THRESHOLD = 0.04   # phase 7: kẹp thẳng đứng làm tay thiếu 2-4 cm tới đích, nên nới ngưỡng "tới nơi"
    # Lệnh mở kẹp lúc thả (action > 0.5 thì env coi là thả). 0.55 mở mỗi ngón ~33 mm, hai má hở ~67 mm: lớn hơn
    # đường chéo hộp (42 mm) để hộp rơi xéo vẫn lọt, nhưng nhỏ hơn mở hết (82 mm, rộng hơn lòng bát 84 mm).
    RELEASE_OPEN_CMD = 0.55

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
        self._ee_site = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "ee_site")

        self._phase = 0                 # đang ở phase nào
        self._phase_step = 0            # đã ở phase này bao nhiêu bước (dùng cho timeout và chờ)
        self._retries = 0               # đã thử kẹp lại bao nhiêu lần (phase 4 quay về phase 2)
        self._max_retries = 2           # tối đa bao nhiêu lần thử lại
        self._last_finger = None        # độ mở ngón ở bước trước (để biết ngón đã ngừng khép chưa)
        self._gripper_target = 1.0      # lệnh kẹp: +1 mở, -1 đóng
        self._current_joint_targets = None   # góc khớp đích mà IK vừa tính (radian)
        self._smoothed_action = None    # action đã làm mượt (để chuyển động không giật)
        self._regrasps = 0              # đã kẹp lại bao nhiêu lần vì hộp tuột khi đang mang
        self._lost_steps = 0            # số bước liên tiếp hộp nằm xa ee khi đang mang
        self._roll_hold = None          # góc wrist_roll đã chọn để trục kẹp song song mặt hộp (None = chưa chọn)
        self._clamp_dir = np.array([1.0, 0.0, 0.0])   # hướng trục kẹp trong mặt phẳng ngang (từ hàm cố định sang hàm di động)
        self._home_ee = None            # vị trí đầu tay lúc ở home
        # Hệ số làm mượt theo từng phase: càng lớn thì bám đích càng nhanh
        self._phase_smoothing = {
            0: 0.8,   # nhấc lên: nhanh
            1: 0.8,   # tới phía trên hộp: nhanh
            2: 0.7,   # hạ xuống pre-grasp: vừa
            3: 0.7,   # hạ xuống kẹp: vừa
            4: 1.0,   # đóng kẹp: tức thì
            5: 0.5,   # nhấc hộp: chậm hơn
            6: 0.5,   # chuyển hộp tới bát: chậm hơn
            7: 0.5,   # hạ vào bát: chậm hơn
            8: 1.0,   # mở kẹp: tức thì
            9: 0.4,   # rút lên: chậm
            10: 0.4,  # về home: chậm
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
        self._lost_steps = 0

    def select_action(self, observation: dict) -> np.ndarray:
        """Hàm chính: nhìn tình hình hiện tại, trả về action (7 số trong [-1, 1]).

        Policy viết tay này không dùng `observation` mà hỏi thẳng env vị trí hộp và bát
        ("gian lận" so với policy AI, vì AI chỉ được nhìn ảnh camera và góc khớp).
        """
        box_pos = self.env.get_box_pos()
        bowl_pos = self.env.get_bowl_pos()
        ee_pos = self.env.get_ee_pos()

        # Hộp tuột khi đang mang (nằm xa ee > 5 cm vài bước liền): quay lại phase 1 để kẹp lại (tối đa 2 lần)
        if 5 <= self._phase <= 7:
            self._lost_steps = self._lost_steps + 1 if np.linalg.norm(box_pos - ee_pos) > 0.05 else 0
            if self._lost_steps > 3 and self._regrasps < 2:
                self._regrasps += 1
                self._lost_steps = 0
                self._phase, self._phase_step = 1, 0
                self._roll_hold = None
                self._last_finger = None

        if self._phase == 0:
            # Nhấc thẳng lên từ home (giữ nguyên x, y; chỉ đổi z)
            target = np.array([self._home_ee[0], self._home_ee[1], self._safe_z])
            self._gripper_target = 1.0
            self._move_toward(target, ee_pos, threshold=0.02)

        elif self._phase == 1:
            # Di chuyển ngang tới phía trên hộp ở độ cao an toàn
            target = np.array([box_pos[0], box_pos[1], self._safe_z])
            self._gripper_target = 1.0
            self._move_toward(target, ee_pos, threshold=0.025)

        elif self._phase == 2:
            # Hạ xuống pre-grasp
            target = np.array([box_pos[0], box_pos[1], self._pre_grasp_z])
            self._gripper_target = 1.0
            self._move_toward(target, ee_pos, threshold=0.025)

        elif self._phase == 3:
            # Hạ xuống vị trí kẹp
            target = np.array([box_pos[0], box_pos[1], self._grasp_z])
            self._gripper_target = 1.0
            self._move_toward(target, ee_pos, threshold=0.025)

        elif self._phase == 4:
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
                else:
                    self._advance_phase()                    # hết lượt thử: đi tiếp như cũ

        elif self._phase == 5:
            # Nhấc thẳng lên (giữ nguyên x, y của tay)
            target = np.array([ee_pos[0], ee_pos[1], self._safe_z])
            self._gripper_target = -1.0
            self._move_toward(target, ee_pos, threshold=0.02)

        elif self._phase == 6:
            # Di chuyển tới phía trên bát ở độ cao an toàn
            target = np.array([bowl_pos[0], bowl_pos[1], self._safe_z])
            target[:2] -= np.clip((box_pos - ee_pos)[:2], -0.04, 0.04)   # hộp nằm lệch khỏi ee (kẹp lệch bên): đưa HỘP, không phải ee, tới tâm bát
            self._gripper_target = -1.0
            self._move_toward(target, ee_pos, threshold=0.02)

        elif self._phase == 7:
            # Hạ hộp xuống trong bát
            target = np.array([bowl_pos[0], bowl_pos[1], self._place_z])
            target[:2] -= np.clip((box_pos - ee_pos)[:2], -0.04, 0.04)
            self._gripper_target = -1.0
            self._move_toward(target, ee_pos, threshold=self.PLACE_ARRIVE_THRESHOLD)

        elif self._phase == 8:
            # Mở kẹp vừa phải: chờ 20 bước cho hộp rơi vào bát
            self._gripper_target = self.RELEASE_OPEN_CMD
            self._phase_step += 1
            if self._phase_step > 20:
                self._advance_phase()

        elif self._phase == 9:
            # Rút tay lên khỏi bát (vẫn mở vừa, để ngón không quét vào thành bát khi rút)
            target = np.array([bowl_pos[0], bowl_pos[1], self._safe_z])
            self._gripper_target = self.RELEASE_OPEN_CMD
            self._move_toward(target, ee_pos, threshold=0.025)

        elif self._phase == 10:
            # Về lại vị trí home
            target = np.array([self._home_ee[0], self._home_ee[1], self._home_ee[2]])
            self._gripper_target = 1.0
            self._move_toward(target, ee_pos, threshold=0.025)

        else:
            # Xong: giữ nguyên tư thế hiện tại
            return self._build_action_from_current()

        return self._build_action()

    def _advance_phase(self):
        """Chuyển sang phase kế tiếp và đặt lại bộ đếm bước của phase."""
        self._phase += 1
        self._phase_step = 0

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
        cur = self._roll_hold if self._roll_hold is not None else self.data.qpos[self._qpos_adr[4]]
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

    def _solve_ik(self, target_pos, roll=None):
        """IK: tìm góc khớp (radian) của 5 khớp để đầu tay (ee_site) tới target_pos.

        Chạy trên bản sao dữ liệu `_ik_data` nên không làm robot thật cử động. Chỉ tính HÌNH HỌC:
        không biết va chạm và không giới hạn góc khớp. Nhiệm vụ chính là vị trí (x, y, z) của đầu tay.
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
        mujoco.mj_forward(self.model, ik)

        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        eye5 = np.eye(5)
        q_pref = np.radians(self.POSTURE_PREF_DEG)
        posture_k = self.POSTURE_GAIN * np.array(self.POSTURE_MASK, dtype=float)   # K · mask
        tool_down = self._phase in self.TOOL_DOWN_PHASES
        down = np.array([0.0, 0.0, -1.0])
        for _ in range(self.IK_ITERATIONS):
            ee = ik.site_xpos[self._ee_site]
            error = target_pos - ee                                # còn cách đích bao xa (vector 3D)
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
            J_pinv = J_all.T @ np.linalg.inv(J_all @ J_all.T + self.IK_DAMPING ** 2 * np.eye(J_all.shape[0]))
            dq = J_pinv @ err_all * free

            # Nhiệm vụ phụ: kéo khớp về tư thế vòm, chỉ trong không gian null
            q = ik.qpos[self._qpos_adr]
            dq = dq + ((eye5 - J_pinv @ J_all) @ (posture_k * (q_pref - q))) * free

            ik.qpos[self._qpos_adr] += self.IK_STEP_GAIN * dq
            mujoco.mj_forward(self.model, ik)

        return ik.qpos[self._qpos_adr].copy()

    @staticmethod
    def _skew(a):
        """Ma trận phản đối xứng [a]× sao cho [a]× @ v = a × v."""
        return np.array([[0.0, -a[2], a[1]], [a[2], 0.0, -a[0]], [-a[1], a[0], 0.0]])

    def _move_toward(self, target_pos, ee_pos, threshold=0.02):
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
        if self._phase in (2, 3):
            # Hạ xuống hộp: chọn roll theo hướng mặt hộp rồi dịch điểm ee sang bên (xem GRASP_SIDE_OFFSET)
            roll, clamp = self._choose_roll(self._solve_ik(target_pos))
            self._roll_hold, self._clamp_dir = roll, clamp
            target_pos = np.array(target_pos, dtype=float) - self.GRASP_SIDE_OFFSET * np.append(clamp[:2], 0.0)
            self._current_joint_targets = self._solve_ik(target_pos, roll)
        else:
            self._current_joint_targets = self._solve_ik(target_pos)
            if self._roll_hold is not None:                        # đã chọn roll thì giữ nguyên tới hết ván
                self._current_joint_targets[4] = self._roll_hold

        self._phase_step += 1
        arrived = np.linalg.norm(ee_pos - target_pos) < threshold
        if arrived or self._phase_step > self.PHASE_TIMEOUT_STEPS:
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
            self._smoothed_action[5:] = raw[5:]  # gripper đổi ngay lập tức, không làm mượt
        return self._smoothed_action.copy()

    def _build_action_from_current(self) -> np.ndarray:
        """Tạo action giữ nguyên tư thế hiện tại (dùng khi đã xong phase 11)."""
        action = np.zeros(7, dtype=np.float32)
        action[:5] = self._to_action(self.data.qpos[self._qpos_adr])
        action[5] = action[6] = self._gripper_target
        return np.clip(action, -1.0, 1.0)
