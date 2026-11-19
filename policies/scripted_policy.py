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
        self._place_z = 0.48       # độ cao thả hộp vào bát

    def reset(self):
        """Gọi mỗi đầu episode: đưa policy về phase 0."""
        self._phase = 0
        self._phase_step = 0
        self._retries = 0               # đã thử kẹp lại bao nhiêu lần
        self._last_finger = None        # độ mở ngón ở bước trước (để biết ngón đã ngừng khép chưa)
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

        print(f"phase={self._phase} bước_trong_phase={self._phase_step} tay={ee_pos.round(3)}")

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
            fo = self.data.qpos[self.model.jnt_qposadr[self.env._finger_left_jnt]]
            settled = self._last_finger is not None and abs(fo - self._last_finger) < 0.0003
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
            self._move_toward(target, ee_pos, threshold=0.02)

        elif self._phase == 8:
            # Mở kẹp: chờ 20 bước cho hộp rơi vào bát
            self._gripper_target = 1.0
            self._phase_step += 1
            if self._phase_step > 20:
                self._advance_phase()

        elif self._phase == 9:
            # Rút tay lên khỏi bát
            target = np.array([bowl_pos[0], bowl_pos[1], self._safe_z])
            self._gripper_target = 1.0
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

    def _move_toward(self, target_pos, ee_pos, threshold=0.02):
        """Dùng IK để tính góc khớp đưa đầu tay tới target_pos.

        Chuyển phase khi: tay đã tới gần đích (khoảng cách < threshold mét),
        HOẶC đã ở phase này quá 200 bước (timeout, tránh kẹt vĩnh viễn).
        Nếu hay thấy phase bị timeout 200 bước thì đó là dấu hiệu tay không tới được đích.
        """
        dist = np.linalg.norm(ee_pos - target_pos)

        # Sao chép trạng thái thật sang bản sao IK
        self._ik_data.qpos[:] = self.data.qpos[:]
        self._ik_data.qvel[:] = 0
        mujoco.mj_forward(self.model, self._ik_data)

        # Lặp tối đa 40 lần, mỗi lần nhích các khớp một chút để đầu tay gần đích hơn
        for _ in range(40):
            shadow_ee = self._ik_data.site_xpos[self._ee_site].copy()
            error = target_pos - shadow_ee      # còn cách đích bao xa (vector 3D)
            if np.linalg.norm(error) < 0.001:   # đã tới đích (sai số < 1 mm)
                break

            # Jacobian J: "xoay mỗi khớp một chút thì đầu tay dịch chuyển thế nào"
            jacp = np.zeros((3, self.model.nv))
            mujoco.mj_jacSite(self.model, self._ik_data, jacp, None, self._ee_site)
            J = jacp[:, self._dof_ids]

            # Damped least squares: giải dq sao cho J @ dq ≈ error.
            # lam (damping) giúp ổn định khi robot gần tư thế "kỳ dị" (singularity).
            lam = 0.05
            dq = J.T @ np.linalg.solve(J @ J.T + lam**2 * np.eye(3), error)

            # Cộng dq (nhân 0.4 để bước nhỏ, an toàn) vào góc khớp của bản sao IK
            for i, jid in enumerate(self._joint_ids):
                self._ik_data.qpos[self.model.jnt_qposadr[jid]] += dq[i] * 0.4
            mujoco.mj_forward(self.model, self._ik_data)

        # Kết quả IK: góc đích (radian) của 5 khớp tay
        self._current_joint_targets = np.array([
            self._ik_data.qpos[self.model.jnt_qposadr[jid]]
            for jid in self._joint_ids
        ])

        self._phase_step += 1
        if dist < threshold or self._phase_step > 200:
            self._advance_phase()

    def _build_action(self) -> np.ndarray:
        """Đổi góc khớp đích (radian) thành action chuẩn hóa [-1, 1], rồi làm mượt."""
        raw = np.zeros(7, dtype=np.float32)
        if self._current_joint_targets is not None:
            for i in range(5):
                lo, hi = self.env._ctrl_ranges[i]
                raw[i] = 2.0 * (self._current_joint_targets[i] - lo) / (hi - lo) - 1.0
        raw[5] = self._gripper_target
        raw[6] = self._gripper_target
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
        for i, jid in enumerate(self._joint_ids):
            lo, hi = self.env._ctrl_ranges[i]
            qpos = self.data.qpos[self.model.jnt_qposadr[jid]]
            action[i] = 2.0 * (qpos - lo) / (hi - lo) - 1.0
        action[5] = self._gripper_target
        action[6] = self._gripper_target
        return np.clip(action, -1.0, 1.0)
