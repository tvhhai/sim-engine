"""Policy viết tay (scripted policy) cho gắp hộp bỏ vào bát: đi theo các waypoint, dùng IK.

Ý tưởng: không có AI, chỉ có code cứng "đưa tay tới điểm A, kẹp, nhấc, đưa tới điểm B, nhả".
Mỗi điểm đích là một vị trí 3D (x, y, z) của đầu tay. IK (Inverse Kinematics) tính ra
các góc khớp cần có để đầu tay tới được điểm đó.

IK chạy trên một bản sao dữ liệu MuJoCo ("shadow copy") nên không làm nhiễu mô phỏng thật.
Đầu ra là action đã chuẩn hóa về [-1, 1] để đưa vào env.step().
"""

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
    TOOL_DOWN_PHASES = (6, 7, 8, 9)
    TOOL_DOWN_WEIGHT = 0.05   # trọng số hướng so với vị trí. 0.03 kém hơn (vẫn chạm bát), >= 0.1 làm đầu tay không tới đích.
    PLACE_Z = 0.50            # độ cao đầu tay lúc thả (miệng bát ở z 0.46). Đo trên 100 seed:
                              #   bát cố định:  0.48 -> 199/200, 0.50 -> 198/200 (gần như bằng nhau)
                              #   bát ngẫu nhiên ±3 cm: 0.48 -> 96/100 (chạm bát 41 ván), 0.50 -> 100/100 (chạm bát 0)
                              #   bát ngẫu nhiên ±5 cm: 0.48 -> 87/100, 0.50 -> 97/100
                              # nên chọn 0.50: ngón còn cao hơn miệng bát, ít vướng khi bát lệch khỏi chỗ gốc.
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
        self._safe_z = 0.62        # độ cao an toàn để di chuyển ngang, tránh va vật thể
        self._pre_grasp_z = 0.46   # độ cao chuẩn bị kẹp
        self._grasp_z = 0.455      # độ cao kẹp: đầu tay ngang tâm hộp, má kẹp ôm hai bên hộp (0.434 làm ngón đè bàn nên khép chậm)
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

    def select_action(self, observation: dict) -> np.ndarray:
        """Hàm chính: nhìn tình hình hiện tại, trả về action (7 số trong [-1, 1]).

        Policy viết tay này không dùng `observation` mà hỏi thẳng env vị trí hộp và bát
        ("gian lận" so với policy AI, vì AI chỉ được nhìn ảnh camera và góc khớp).
        """
        box_pos = self.env.get_box_pos()
        bowl_pos = self.env.get_bowl_pos()
        ee_pos = self.env.get_ee_pos()

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
            self._gripper_target = -1.0
            self._move_toward(target, ee_pos, threshold=0.02)

        elif self._phase == 7:
            # Hạ hộp xuống trong bát
            target = np.array([bowl_pos[0], bowl_pos[1], self._place_z])
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

    def _solve_ik(self, target_pos):
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
        ik = self._ik_data
        # Bắt đầu từ tư thế thật hiện tại, nên lời giải "gần" tư thế đang có
        ik.qpos[:] = self.data.qpos[:]
        ik.qvel[:] = 0
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
            J_pinv = J_all.T @ np.linalg.inv(J_all @ J_all.T + self.IK_DAMPING ** 2 * np.eye(J_all.shape[0]))
            dq = J_pinv @ err_all

            # Nhiệm vụ phụ: kéo khớp về tư thế vòm, chỉ trong không gian null
            q = ik.qpos[self._qpos_adr]
            dq = dq + (eye5 - J_pinv @ J_all) @ (posture_k * (q_pref - q))

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
        self._current_joint_targets = self._solve_ik(target_pos)

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
            self._smoothed_action[:5] += alpha * (raw[:5] - self._smoothed_action[:5])
            self._smoothed_action[5:] = raw[5:]  # gripper đổi ngay lập tức, không làm mượt
        return self._smoothed_action.copy()

    def _build_action_from_current(self) -> np.ndarray:
        """Tạo action giữ nguyên tư thế hiện tại (dùng khi đã xong phase 11)."""
        action = np.zeros(7, dtype=np.float32)
        action[:5] = self._to_action(self.data.qpos[self._qpos_adr])
        action[5] = action[6] = self._gripper_target
        return np.clip(action, -1.0, 1.0)
