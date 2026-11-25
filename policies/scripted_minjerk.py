"""Bản thử: thay cách làm mượt của policy viết tay bằng quỹ đạo minimum-jerk (đa thức bậc 5).

Policy gốc (ScriptedPolicy) đi tới đích theo hai lớp:
  1. `_move_toward`: ở phase 1-3, đích đầu tay tiến đều tối đa TCP_STEP (2 cm) mỗi bước -> tốc độ không đổi, đến nơi thì dừng đột ngột.
  2. `_build_action`: làm mượt hàm mũ, mỗi bước lệnh khớp đi alpha (0,5-0,8) phần quãng còn lại -> nhanh lúc đầu, chậm dần, bám đuổi
     (đích mới xuất hiện là tay giật mạnh nhất ngay bước đầu).

Bản này thay cả hai bằng minimum-jerk: mỗi đoạn đường (mỗi phase) đi từ điểm đầu p0 tới đích theo
    s(τ) = 10τ³ − 15τ⁴ + 6τ⁵,   τ = bước / T  (0 -> 1)
s(0)=0, s(1)=1, vận tốc và gia tốc ở hai đầu bằng 0: tay tăng tốc êm, chạy nhanh nhất giữa đường (1,875 lần tốc độ trung bình),
rồi giảm tốc êm. Đây là quỹ đạo làm "độ giật" (đạo hàm của gia tốc) nhỏ nhất (Flash & Hogan 1985: tay người với tới vật đi gần đúng
như vậy). Đích được đọc lại mỗi bước nên hộp xê dịch thì đường đi vẫn bám theo.

  - Đoạn đi bằng `_move_toward` (đích là điểm 3D): nội suy minimum-jerk TRONG KHÔNG GIAN 3D rồi giải IK mỗi bước -> đầu tay đi
    thẳng (như TCP_STEP cũ) nhưng tốc độ êm. Lệnh khớp không làm mượt thêm (alpha = 1) vì quỹ đạo đã êm sẵn.
  - Đoạn đích là góc khớp (về home, nhấc từ home theo từng khớp): nội suy minimum-jerk trong không gian khớp.
  - Kẹp và các giới hạn an toàn (LIFT_STEP_LIMIT khi mang hộp, RELEASE_RAMP khi thả) giữ nguyên.

Chỉ đổi V1; bản kết hợp vẫn chuyển sang V2 (không đổi) khi V1 bỏ cuộc.
Chạy thử: python -m sim_engine.tools.make_replay --policy sim_engine.policies.scripted_minjerk:ScriptedPolicyHybridMinJerk --seeds 83,200
"""

import math

import numpy as np

from .scripted_hybrid import ScriptedPolicyHybrid
from .scripted_policy import ScriptedPolicy
from .scripted_v2 import ScriptedPolicyV2


def min_jerk(tau):
    """s(τ) = 10τ³ − 15τ⁴ + 6τ⁵, kẹp τ trong [0, 1]."""
    t = min(max(tau, 0.0), 1.0)
    return t * t * t * (10.0 + t * (-15.0 + 6.0 * t))


class ScriptedPolicyMinJerk(ScriptedPolicy):
    MJ_TCP_PEAK = 0.02     # m/bước: tốc độ đỉnh của đầu tay (giữa đường), bằng TCP_STEP cũ (0,6 m/s ở 30 Hz)
    MJ_JOINT_PEAK = 0.12   # thang action (-1..1)/bước: tốc độ đỉnh của khớp ở đoạn đích là góc khớp (0,06: về home 27 bước, gốc 14)
    # Phase đi bằng đường 3D minimum-jerk. Phase 0 (nhấc khỏi home) giữ cách cũ: đường thẳng 3D từ home đẩy cổ tay vào giới hạn 95°,
    # đầu tay dừng cách đích 2,02 cm > ngưỡng 2 cm nên phase 0 chờ hết hạn 200 bước (đo ở cả 5 seed thử)
    MJ_PHASES = (1, 2, 3, 5, 6, 7, 9)
    MJ_JOINT_PHASES = (10,)    # phase đích là góc khớp đi bằng minimum-jerk trong không gian khớp (về home)
    MJ_MIN_STEPS = 4       # đoạn ngắn nhất (bước): đích gần mấy cũng không giật trong 1-2 bước
    MJ_TIMEOUT_SHARE = 0.6 # đoạn không dài quá 60% hạn của phase, để còn thời gian bám đích trước khi phase hết hạn

    def __init__(self, env):
        super().__init__(env)
        self.TCP_STEP = self.TCP_STEP1 = 0.0                   # tắt bước tiến đều cũ: minimum-jerk thay nó
        self._mj_reset()

    def reset(self):
        super().reset()
        self._mj_reset()

    def _mj_reset(self):
        self._mj_seg = None        # đoạn 3D đang đi: (phase, p0, T)
        self._mj_jseg = None       # đoạn khớp đang đi: (phase, q0, T)
        self._mj_cart = False      # bước này đã đi bằng đoạn 3D chưa (thì _build_action không làm mượt thêm)
        self._mj_k = 0

    def select_action(self, observation):
        self._mj_cart = False
        return super().select_action(observation)

    def _move_toward(self, target_pos, ee_pos, threshold=0.02, *, advance_on_failure=True, timeout_steps=None):
        if self._phase not in self.MJ_PHASES:
            return super()._move_toward(target_pos, ee_pos, threshold, advance_on_failure=advance_on_failure, timeout_steps=timeout_steps)
        target_pos = np.asarray(target_pos, dtype=float)
        if self._mj_seg is None or self._mj_seg[0] != self._phase or self._phase_step == 0:
            dist = float(np.linalg.norm(target_pos - ee_pos))
            cap = timeout_steps if timeout_steps is not None else self.PHASE_TIMEOUTS.get(self._phase, self.PHASE_TIMEOUT_STEPS)
            T = max(self.MJ_MIN_STEPS, math.ceil(1.875 * dist / self.MJ_TCP_PEAK))
            T = min(T, max(self.MJ_MIN_STEPS, int(cap * self.MJ_TIMEOUT_SHARE)))
            self._mj_seg, self._mj_k = (self._phase, np.array(ee_pos, dtype=float), T), 0
        phase, p0, T = self._mj_seg
        self._mj_k += 1
        via = p0 + (target_pos - p0) * min_jerk(self._mj_k / T)  # điểm trên đường thẳng p0 -> đích, theo nhịp minimum-jerk
        # Dùng lại _move_toward gốc để giải IK và đếm bước / chuyển phase, nhưng IK nhắm `via`; khoảng cách tới đích vẫn đo tới đích thật
        self.motion_target = (target_pos.copy(), threshold)
        self._current_joint_targets = self._solve_ik(via)
        if self._roll_hold is not None:
            self._current_joint_targets[4] = self._roll_hold
        self._mj_cart = True
        self._count_and_advance(target_pos, ee_pos, threshold, advance_on_failure, timeout_steps)

    def _count_and_advance(self, target_pos, ee_pos, threshold, advance_on_failure, timeout_steps):
        """Phần đếm bước / phát hiện kẹt / chuyển phase, chép nguyên từ ScriptedPolicy._move_toward."""
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

    def _build_action(self):
        raw = np.zeros(7, dtype=np.float32)
        if self._current_joint_targets is not None:
            raw[:5] = self._to_action(self._current_joint_targets)
        raw[5] = raw[6] = self._gripper_target
        raw = np.clip(raw, -1.0, 1.0)
        if self._smoothed_action is None:
            self._smoothed_action = raw.copy()
        else:
            if self._mj_cart:
                goal = raw[:5]                                      # đường 3D đã êm: lệnh khớp = đích IK
                self._mj_jseg = None
            elif self._phase not in self.MJ_JOINT_PHASES:           # phase giữ cách cũ: làm mượt hàm mũ
                alpha = self._phase_smoothing.get(self._phase, 0.7)
                goal = self._smoothed_action[:5] + alpha * (raw[:5] - self._smoothed_action[:5])
                self._mj_jseg = None
            else:                                                   # đích là góc khớp: minimum-jerk trong không gian khớp
                if self._mj_jseg is None or self._mj_jseg[0] != self._phase:
                    q0 = self._smoothed_action[:5].copy()
                    T = max(self.MJ_MIN_STEPS, math.ceil(1.875 * float(np.max(np.abs(raw[:5] - q0))) / self.MJ_JOINT_PEAK))
                    self._mj_jseg, self._mj_jk = (self._phase, q0, T), 0
                phase, q0, T = self._mj_jseg
                self._mj_jk += 1
                goal = q0 + (raw[:5] - q0) * min_jerk(self._mj_jk / T)
            step = goal - self._smoothed_action[:5]
            if 5 <= self._phase <= 7:                               # mang hộp: giữ giới hạn cũ
                step = np.clip(step, -self.LIFT_STEP_LIMIT, self.LIFT_STEP_LIMIT)
            self._smoothed_action[:5] += step
            self._smoothed_action[5:] = raw[5:]
            self._smoothed_action[5] = min(raw[5], self._prev_grip + self.RELEASE_RAMP)
            self._smoothed_action[6] = self._smoothed_action[5]
        self._prev_grip = float(self._smoothed_action[5])
        return self._smoothed_action.copy()


class ScriptedPolicyHybridMinJerk(ScriptedPolicyHybrid):
    """Bản kết hợp V1 + V2 như mặc định, chỉ thay V1 bằng bản minimum-jerk."""

    def __init__(self, env):
        self.env = env
        self.v1 = ScriptedPolicyMinJerk(env)
        self.v2 = ScriptedPolicyV2(env)
        self._active = self.v1
        self.switched = False
