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

from copy import copy
import math
from typing import NamedTuple

import mujoco
import numpy as np

from .. import config
from .scripted_hybrid import ScriptedPolicyHybrid
from .scripted_policy import ScriptedPolicy
from .scripted_v2 import ScriptedPolicyV2


def min_jerk(tau):
    """s(τ) = 10τ³ − 15τ⁴ + 6τ⁵, kẹp τ trong [0, 1]."""
    t = min(max(tau, 0.0), 1.0)
    return t * t * t * (10.0 + t * (-15.0 + 6.0 * t))


def carry_in(tau):
    """h(τ) = τ − 6τ³ + 8τ⁴ − 3τ⁵: đa thức bậc 5 có h(0)=h(1)=0, h'(0)=1, h'(1)=0, h''(0)=h''(1)=0.
    Cộng v0·T·h(τ) vào đường minimum-jerk để đoạn mới BẮT ĐẦU với vận tốc v0 của đoạn trước (không dừng ở chỗ nối), vẫn tới đích êm."""
    t = min(max(tau, 0.0), 1.0)
    return t * (1.0 + t * t * (-6.0 + t * (8.0 - 3.0 * t)))


class ScriptedPolicyMinJerk(ScriptedPolicy):
    MJ_TCP_PEAK = 0.03     # m/bước: tốc độ đỉnh của đầu tay (giữa đường), 0,9 m/s ở 30 Hz. 0,02 (= TCP_STEP cũ): 4/5 seed thử chậm hơn
    MJ_APPROACH_PEAK = None  # Tốc độ đỉnh riêng cho pha 1, trước khi hạ xuống vật.
                           # bản gốc 5-8% vì tốc độ trung bình của minimum-jerk chỉ bằng 1/1,875 tốc độ đỉnh
    # Cặp (phase trước, phase sau) nối vận tốc: tay đi liền một mạch, không dừng ở chỗ đổi phase. Bản đầu dừng hẳn ở mọi chỗ nối nên
    # nhìn "khựng". Không nối 2 -> 3 (lơ lửng -> hạ thẳng xuống hộp): phải dừng trên hộp để hạ thẳng đứng, không quét ngang vào hộp
    MJ_BLEND = {(1, 2), (5, 6), (6, 6), (6, 7)}
    MJ_FINISH_PHASES = (3, 7)  # phase chỉ được "tới nơi" khi đã đi hết đoạn minimum-jerk (xem _count_and_advance)
    MJ_JOINT_PEAK = 0.12   # thang action (-1..1)/bước: tốc độ đỉnh của khớp ở đoạn đích là góc khớp (0,06: về home 27 bước, gốc 14)
    # Phase đi bằng đường 3D minimum-jerk. Phase 0 (nhấc khỏi home) giữ cách cũ: đường thẳng 3D từ home đẩy cổ tay vào giới hạn 95°,
    # đầu tay dừng cách đích 2,02 cm > ngưỡng 2 cm nên phase 0 chờ hết hạn 200 bước (đo ở cả 5 seed thử)
    MJ_PHASES = (1, 2, 3, 5, 6, 7, 9)
    MJ_JOINT_PHASES = (10,)    # phase đích là góc khớp đi bằng minimum-jerk trong không gian khớp (về home)
    MJ_MIN_STEPS = 4       # đoạn ngắn nhất (bước): đích gần mấy cũng không giật trong 1-2 bước
    MJ_TIMEOUT_SHARE = 0.6 # đoạn không dài quá 60% hạn của phase, để còn thời gian bám đích trước khi phase hết hạn

    # --- Kéo vòng cung bằng khớp 1 (hộp sát giới hạn khớp 1 và xa đế) ---
    # Hộp nằm ở góc gần ±110° (giới hạn khớp 1) và xa đế: kẹp không xoay được cho song song mặt hộp mà vẫn hạ tới độ cao kẹp, gắp thường
    # hỏng (774 seed: 3 seed còn hỏng 55, 1016, 1022 đều ở -109..-110°, r 39-42 cm). Ý người dùng: đặt kẹp lên hộp, khép, XOAY KHỚP 1 vào
    # trong để ma sát kéo hộp theo vòng cung ra khỏi sát giới hạn, nhả, rồi gắp lại bình thường. Thử ở seed 55 (sau khi kéo hộp vào 38,5 cm):
    # tay chỉ hạ tới ~45,7 cm (đè lên nóc hộp), hộp theo được ~5° (-110° -> -105°) rồi trượt; sau đó policy gắp được với roll -45/0/90°.
    ARC = True
    ARC_MARGIN_DEG = 3.0       # hộp cách giới hạn khớp 1 dưới chừng này thì kéo vòng cung
    ARC_R_MIN = 0.37           # chỉ hộp xa đế (gần đế thì gắp được dù sát giới hạn: seed 13, 14, 17 r 15-25 cm vẫn xanh)
    ARC_ROLL_DEG = 45.0        # roll khi đè lên hộp: -45° ở phía khớp 1 âm, +45° ở phía dương (seed 55: -45° kéo được, 45° không)
    ARC_SWEEP_DEG = 25.0       # xoay khớp 1 vào trong chừng này (hộp theo được ít hơn vì trượt)
    ARC_PRESS_Z = 0.430        # đích đầu tay khi đè (thường không tới được; IK dừng ở chỗ thấp nhất với được)
    ARC_STEPS = {"above": 20, "down": 15, "close": 12, "sweep": 30, "release": 10, "raise": 15}

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
        self._mj_last = None       # điểm via của bước trước và vận tốc (m/bước), để nối đoạn
        self._mj_vel = np.zeros(3)
        self._arc, self._arc_done, self._arc_k, self._arc_q0, self._arc_goal = None, False, 0, None, None

    def select_action(self, observation):
        self._mj_cart = False
        if (self.ARC and self._arc is None and not self._arc_done and self._phase == 2 and self._phase_step == 0
                and self._arc_needed()):
            self._arc_start()
        if self._arc is not None:
            return self._arc_step()
        return super().select_action(observation)

    # --- kéo vòng cung ---
    def _box_bearing(self):
        """(góc của hộp quanh trục khớp 1 theo quy ước góc khớp 1 (độ), khoảng cách tới trục (m))."""
        b, pan = self.env.get_box_pos(), config.PICK_PAN_XY
        return math.degrees(math.atan2(b[0] - pan[0], b[1] - pan[1])), math.hypot(b[0] - pan[0], b[1] - pan[1])

    def _arc_needed(self):
        # Chỉ cho hộp: bóng lăn khỏi kẹp khi bị đè và xoay (thử 10 seed sát giới hạn: bóng chỉ dịch 0-0,9 cm, bóng 3 cm hỏng 8/10 so với 3/10 khi
        # không kéo). Muốn dùng cho bóng phải làm khác: ôm hẳn bóng (kẹp sâu hơn tâm) và xoay chậm.
        if getattr(self.env, "target_type", "box") == "sphere":
            return False
        ang, r = self._box_bearing()
        lim = math.degrees(float(self._jnt_hi[0] if ang > 0 else -self._jnt_lo[0]))
        return abs(ang) >= lim - self.ARC_MARGIN_DEG and r >= self.ARC_R_MIN

    def _arc_start(self):
        ang, _ = self._box_bearing()
        side = 1.0 if ang > 0 else -1.0
        b = self.env.get_box_pos()
        roll = math.radians(side * self.ARC_ROLL_DEG)
        q_above = self._solve_ik(np.array([b[0], b[1], self._pre_grasp_z + 0.03]), roll=roll, limits=True)
        q_press = self._solve_ik(np.array([b[0], b[1], self.ARC_PRESS_Z]), roll=roll, limits=True, q0=q_above)
        q_sweep = q_press.copy()
        q_sweep[0] = float(np.clip(q_press[0] - side * math.radians(self.ARC_SWEEP_DEG), self._jnt_lo[0], self._jnt_hi[0]))
        q_raise = q_sweep.copy()
        q_raise[1] -= math.radians(30)
        q_raise[2] += math.radians(15)
        self._arc_plan = {"above": (q_above, 1.0), "down": (q_press, 1.0), "close": (q_press, -1.0),
                          "sweep": (q_sweep, -1.0), "release": (q_sweep, 1.0), "raise": (q_raise, 1.0)}
        self._arc, self._arc_k, self._arc_done = "above", 0, True
        self._arc_q0 = self.data.qpos[self._qpos_adr].copy()
        self.last_exit = "arc_start"

    def _arc_step(self):
        """Một bước của thao tác kéo vòng cung: nội suy minimum-jerk trong không gian khớp tới tư thế của chặng hiện tại."""
        order = list(self.ARC_STEPS)
        q_goal, grip = self._arc_plan[self._arc]
        n = self.ARC_STEPS[self._arc]
        self._arc_k += 1
        s = min_jerk(self._arc_k / n)
        self._current_joint_targets = self._arc_q0 + (q_goal - self._arc_q0) * s
        self._gripper_target = grip
        self.motion_target = None
        self._phase_step = 0
        if self._arc_k >= n:
            i = order.index(self._arc)
            self._arc_q0, self._arc_k = q_goal.copy(), 0
            if i + 1 < len(order):
                self._arc = order[i + 1]
            else:                                                       # xong: gắp lại từ phase 2 như sau khi kéo hộp vào
                self._arc = None
                self._phase, self._phase_step, self._plan = 2, 0, None
                self._roll_hold, self._last_finger, self._retries = None, None, 0
                self._far_fix_key, self._far_fix_roll = None, None
                self._best_dist, self._stall, self.last_exit = 0.0, 0, "arc_done"
                self._mj_seg = self._mj_last = None
        return self._build_action()

    def _move_toward(self, target_pos, ee_pos, threshold=0.02, *, advance_on_failure=True, timeout_steps=None):
        if self._phase not in self.MJ_PHASES or timeout_steps is not None:
            # timeout_steps riêng = các đoạn kéo hộp lại gần (hạn 20-120 bước) và rút lên khi đặt hỏng: giữ cách cũ
            self._mj_seg = self._mj_last = None
            return super()._move_toward(target_pos, ee_pos, threshold, advance_on_failure=advance_on_failure, timeout_steps=timeout_steps)
        target_pos = np.asarray(target_pos, dtype=float)
        if self._mj_seg is None or self._mj_seg[0] != self._phase or self._phase_step == 0:
            prev = self._mj_seg[0] if self._mj_seg is not None else None
            if (prev, self._phase) in self.MJ_BLEND and self._mj_last is not None:
                p0, v0 = self._mj_last.copy(), self._mj_vel.copy()  # đi tiếp từ điểm via cuối, giữ vận tốc
            else:
                p0, v0 = np.array(ee_pos, dtype=float), np.zeros(3)
                self._mj_last = None                                 # via cũ thuộc đoạn khác: không dùng để tính vận tốc
            dist = float(np.linalg.norm(target_pos - p0))
            cap = timeout_steps if timeout_steps is not None else self.PHASE_TIMEOUTS.get(self._phase, self.PHASE_TIMEOUT_STEPS)
            peak = self.MJ_APPROACH_PEAK if self._phase == 1 and self.MJ_APPROACH_PEAK is not None else self.MJ_TCP_PEAK
            T = max(self.MJ_MIN_STEPS, math.ceil(1.875 * dist / peak))
            T = min(T, max(self.MJ_MIN_STEPS, int(cap * self.MJ_TIMEOUT_SHARE)))
            self._mj_seg, self._mj_k = (self._phase, p0, T, v0), 0
        phase, p0, T, v0 = self._mj_seg
        self._mj_k += 1
        tau = self._mj_k / T
        # điểm trên đường p0 -> đích theo nhịp minimum-jerk, cộng phần vận tốc mang từ đoạn trước (tắt dần về 0)
        via = p0 + (target_pos - p0) * min_jerk(tau) + v0 * T * carry_in(tau)
        if self._mj_last is not None:
            self._mj_vel = via - self._mj_last
        self._mj_last = via.copy()
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
        moving = self._mj_seg is not None and self._mj_k < self._mj_seg[2]
        if self._phase_step == 1 or dist < self._best_dist - self.STALL_PROGRESS or moving:
            # đang đi đoạn minimum-jerk: đầu đoạn tay đi rất chậm (cố ý), không tính là "kẹt". Bản trước đếm kẹt ở đây nên phase 3 của
            # đường hạ thẳng (thoát khi kẹt 4 bước) khép kẹp quá sớm (seed 2462 lần gắp lại)
            self._best_dist, self._stall = min(dist, self._best_dist) if self._phase_step > 1 else dist, 0
        else:
            self._stall += 1
        cap = timeout_steps if timeout_steps is not None else self.PHASE_TIMEOUTS.get(self._phase, self.PHASE_TIMEOUT_STEPS)
        # Phase 7 (hạ vào bát): ngưỡng "tới nơi" rộng 4 cm (PLACE_ARRIVE_THRESHOLD) nên tay vừa vào phase đã "tới" và thả ở 8,5 cm trên bàn.
        # Bản gốc cũng chỉ ở phase 7 một bước, nhưng lệnh hạ của bước đó còn tác dụng khi mở kẹp nên tay tụt xuống ~5 cm rồi mới nhấc lên.
        # Ở đây chờ đi hết đoạn hạ (τ = 1) rồi mới cho "tới nơi": thả ở độ cao PLACE_Z, không cần tụt rồi nhấc
        # Phase 3 (hạ xuống kẹp, đường hạ chéo cho hộp xa) cũng vậy: ngưỡng 2,5 cm làm kẹp khép khi đoạn hạ chưa xong, ôm cạn hơn ~4 mm,
        # hộp tuột (seed 2462: kẹp khép hết, về home)
        landing = self._phase in self.MJ_FINISH_PHASES and self._mj_seg is not None and self._mj_k < self._mj_seg[2]
        if dist < threshold and not landing:
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
                limit = self._carry_step_limit()
                step = np.clip(step, -limit, limit)
            self._smoothed_action[:5] += step
            self._smoothed_action[5:] = raw[5:]
            self._smoothed_action[5] = min(raw[5], self._prev_grip + self.RELEASE_RAMP)
            self._smoothed_action[6] = self._smoothed_action[5]
        self._prev_grip = float(self._smoothed_action[5])
        return self._smoothed_action.copy()


class ScriptedBoxMinJerk(ScriptedPolicyMinJerk):
    """Phương án kẹp hộp với độ cao, độ lệch và cách rời tư thế nghỉ khác."""

    DRAG_UP_BOX = .02  # Hộp xa: thử nhấc 2 cm rồi thu tay về khi vẫn giữ hộp.

    def __init__(self, env, height_delta, side_delta, staged=True, carry="retry", fixed_roll=None, keep_approach_roll=False):
        super().__init__(env)
        self._height_delta = height_delta
        self._side_delta = side_delta
        self.STAGED_LIFT = staged
        self.DRAG_FAR = carry != "direct"
        self._drag_lift = self.DRAG_UP_BOX if carry == "hold" else 0.0
        self._keep_approach_roll = keep_approach_roll
        self.ARC = not keep_approach_roll
        self._fixed_roll = None if fixed_roll is None else math.radians(fixed_roll)
        if self._fixed_roll is not None:
            self.FAR_FIXED_ROLL = self.PREDICT_BOWL_ROLL = self.PAN_LIMIT_ROLL = False
            self.FAR_ROLL_R = None

    def select_action(self, observation):
        if self._keep_approach_roll and self._phase == 2 and self._phase_step == 0 and self._fixed_roll is None:
            self._fixed_roll = float(self.data.qpos[self._qpos_adr[4]])
            self.GRASP_ALIGN = False
        return super().select_action(observation)

    def _apply_object_profile(self):
        super()._apply_object_profile()
        self._grasp_z += self._height_delta
        self.GRASP_SIDE_OFFSET += self._side_delta
        if self._fixed_roll is not None:
            self.GRASP_ALIGN = False

    def _drag_clearance_lift(self):
        return self._drag_lift

    def _choose_roll(self, q5):
        if self._fixed_roll is None:
            return super()._choose_roll(q5)
        q = np.array(q5, dtype=float)
        q[4] = self._fixed_roll
        return q[4], self._clamp_direction(q)


class ScriptedSphereMinJerk(ScriptedPolicyMinJerk):
    """Biến thể đường gắp bóng để thử khi quỹ đạo cơ sở làm lăn hoặc tuột bóng."""

    def __init__(self, env, bearing_offset, height_delta, side_delta):
        super().__init__(env)
        self._bearing_offset = bearing_offset
        self._height_delta = height_delta
        self._side_delta = side_delta

    def _apply_object_profile(self):
        super()._apply_object_profile()
        self.GRASP_ALIGN = False
        self._grasp_z += self._height_delta
        self.GRASP_SIDE_OFFSET += self._side_delta

    def _box_yaw(self):
        # Bóng không có mặt để căn. Có thể ưu tiên hướng kẹp theo bán kính thay cho yaw của vật.
        if self._bearing_offset is None:
            return super()._box_yaw()
        radial = self.env.get_box_pos()[:2] - config.PICK_PAN_XY
        return math.atan2(radial[1], radial[0]) + math.radians(self._bearing_offset)


class ScriptedSphereIntercept(ScriptedPolicyMinJerk):
    """Đón bóng đã lăn ngay sau khi khởi tạo cảnh, rồi nâng theo policy thường."""

    def __init__(self, env, horizon, roll, close_offset):
        super().__init__(env)
        self._horizon = horizon
        self._intercept_roll = math.radians(roll)
        self._close_step = horizon + close_offset

    def reset(self):
        super().reset()
        self._intercept_step = 0
        self._intercept_start = self.data.qpos[self._qpos_adr].copy()
        adr = self.env._box_dof_adr
        velocity = self.data.qvel[adr:adr + 2]
        xy = self.env.get_box_pos()[:2] + velocity * (self._horizon / config.CONTROL_FREQ)
        self._phase = 3
        self._roll_hold = self._intercept_roll
        target = np.array([*xy, self._grasp_z])
        q = self._solve_ik(target, roll=self._roll_hold, limits=True)
        target[:2] -= self.GRASP_SIDE_OFFSET * self._clamp_direction(q)[:2]
        self._intercept_goal = self._solve_ik(target, roll=self._roll_hold, limits=True)
        self._intercept_target = target

    def select_action(self, observation):
        if self._intercept_step is None:
            return super().select_action(observation)
        if self.env._grasped and self._intercept_step >= self._horizon - 4:
            self._finish_intercept(self._smoothed_action, 5)
            return super().select_action(observation)
        self._intercept_step += 1
        self._phase_step = self._intercept_step
        fraction = min_jerk(self._intercept_step / (self._horizon - 3))
        self._current_joint_targets = self._intercept_start + fraction * (self._intercept_goal - self._intercept_start)
        self.motion_target = (self._intercept_target, 0.008)
        grip = -1.0 if self._intercept_step >= self._close_step else 1.0
        action = np.clip(np.array([*self._to_action(self._current_joint_targets), grip, grip]), -1.0, 1.0)
        self._smoothed_action = action.copy()
        if self._intercept_step >= self._horizon + 8:
            self._roll_hold = None
            self._finish_intercept(action, 1)
        return action

    def _finish_intercept(self, action, phase):
        self._intercept_step = None
        self._phase, self._phase_step = phase, 0
        self._smoothed_action = action.copy()
        self._gripper_target = self._prev_grip = float(action[5])


class _TrialResult(NamedTuple):
    ok: bool
    steps: int
    score: tuple[bool, int, int]
    revisits: int
    pad_heights: tuple[float, float] | None
    pulled: bool
    contacts: tuple[int, int, int] = (0, 0, 0)


def _carry_contacts(env, bodies):
    floor, bowl, camera, camera_mount = bodies
    table_hit = bowl_hit = camera_hit = False
    geom_body = env.model.geom_bodyid
    object_body = env._box_body
    for contact in env.data.contact:
        a, b = geom_body[contact.geom1], geom_body[contact.geom2]
        if a in floor or b in floor:
            other = b if a in floor else a
            table_hit |= other not in floor and other not in (object_body, bowl)
        if bowl in (a, b):
            other = b if a == bowl else a
            bowl_hit |= other not in floor and other != object_body
        if camera in (a, b):
            other = b if a == camera else a
            camera_hit |= other not in camera_mount
    return table_hit, camera_hit, bowl_hit


class ScriptedPolicyHybridMinJerk(ScriptedPolicyHybrid):
    """Bản kết hợp V1 + V2 như mặc định, chỉ thay V1 bằng bản minimum-jerk."""

    # Hộp 4 cm: nhấc các khớp đồng thời sửa seed 204 và 248. Nâng điểm kẹp 4 mm sửa seed 87
    # và dời điểm kẹp theo trục hàm 4 mm sửa seed 244.
    BOX_GRASPS = (
        (0, 0, False), (.004, 0, True), (-.004, 0, True), (0, .004, True),
        (0, -.004, True), (.004, .004, True), (-.004, .004, True),
    )
    # Seed 245 (r = 42 cm): roll -100° và điểm kẹp thấp hơn 4 mm giảm từ hai lượt xuống một.
    BOX_FAR_FIXED_GRASP = (-.004, 0, True, "direct", -100)
    # Hộp 133: giữ roll -87° cuối pha 1, lệch trục hàm 4 mm, bỏ kéo vòng cung và kéo lại (373 → 213 bước).
    BOX_EDGE_GRASPS = ((0, .004, False, "direct", None, True), (0, 0, False, "direct", None, True),
                       (.004, 0, False, "direct", None, True), (0, -.004, False, "direct", None, True),
                       (0, 0, False, "hold", None, True))
    # Hộp 55 và 1018: ở pha kéo, chờ tới 120 bước không giúp tay tới đích. Chấm thử 24 bước (435 → 342/345).
    BOX_EDGE_PULL_TIMEOUT = 24

    # Thử trên bản sao vật lý. Với ngưỡng 0,018, pha đưa ngang giảm từ 31 xuống 18 bước ở seed hộp
    # 87 và từ 137 xuống 80 ở seed hộp 133. Các pha kẹp/nhấc/hạ vẫn dùng giới hạn cũ.
    TIMINGS = ((.018, .20, 1.60, 2.50, None), (.014, .17, 1.40, 2.25, None), (None, .20, 1.20, 2.50, None),
               # Seed hộp 179: tăng tốc pha tiếp cận 30 → 45 mm/bước, giảm 154 → 150 bước.
               (None, .12, 1.20, 2.00, .045), (.012, .20, 1.30, 2.50, None))
    # Ván bóng 2278 có thể mất 500 bước khi thử lại bằng min-jerk, trong khi V1 từng gắp xong trong 138.
    SLOW_SPHERE_CYCLE_STEPS = 300

    # Seed 8 và 201: nâng điểm kẹp 4 mm tránh má kẹp đẩy quả bóng trước khi nâng.
    # Hướng và độ lệch khác giúp kiểm tra điểm kẹp quanh quả cầu khi bị giới hạn khớp.
    SPHERE_PROFILES = (
        (None, .004, 0), (None, -.004, 0),
        (0, 0, 0), (30, 0, 0), (60, 0, 0),
        "v1", "v2",
        (0, .004, 0), (30, .004, 0), (60, .004, 0),
        (0, -.004, 0), (30, -.004, 0), (60, -.004, 0),
        (0, 0, .004), (30, 0, .004), (60, 0, .004),
        (0, 0, -.004), (30, 0, -.004), (60, 0, -.004),
    )
    # Seed 2317: bóng đã lăn 0,31 m/s sau reset. Đón trước 14 bước gắp được với roll 45° hoặc -135°.
    SPHERE_INTERCEPT_SPEED = .10
    SPHERE_INTERCEPTS = (("intercept", 14, 45, -3), ("intercept", 14, -135, -3), ("intercept", 18, -45, 0))
    # Bóng 2278 ở bước nhấc 1 cm: hai má chạm cao hơn tâm 4,0 và 14,4 mm.
    # Xoay theo bán kính và dịch 4 mm giữ 138 bước, đưa điểm chạm còn 1,0 và 4,6 mm trên tâm.
    SPHERE_HIGH_PAD_MIN = .003
    SPHERE_HIGH_PAD_MAX = .012
    SPHERE_CENTER_BAND = .006
    SPHERE_CENTER_LIFT = .010
    SPHERE_CENTER_GRASPS = ((0, 0, .004), (0, -.004, -.004), (0, -.008, -.004), (60, -.008, 0))
    # Kiểm tra ván bóng trên 180 bước có lặp pha. Seed 2447, đổi hướng và điểm kẹp giảm 294 → 204 bước.
    SPHERE_RETRY_STEP_THRESHOLD = 180
    SPHERE_RETRY_GRASPS = ((30, 0, 0), (60, 0, 0), "v1", "v2")

    def __init__(self, env):
        self.env = env
        # Chọn quỹ đạo bằng dự báo vật lý cho bài một vật 4 cm.
        sphere = getattr(env, "target_type", "box") == "sphere"
        plan = abs(env.target_size - .04) < 1e-6 and len(env.lab_objects) == 1
        self._plan_box = plan and not sphere
        self._plan_sphere = plan and sphere
        self.v1 = ScriptedPolicy(env) if sphere else ScriptedPolicyMinJerk(env)
        self.v2 = ScriptedPolicyV2(env)
        self._active = self.v1
        self.switched = False

    def reset(self):
        if self._plan_box:
            self._reset_profile(*self._choose_profile((*self.BOX_GRASPS, "v1", "v2")))
        elif self._plan_sphere:
            adr = self.env._box_dof_adr
            speed = np.linalg.norm(self.env.data.qvel[adr:adr + 2])
            intercepts = self.SPHERE_INTERCEPTS if speed >= self.SPHERE_INTERCEPT_SPEED else ()
            self._reset_profile(*self._choose_profile((*intercepts, *self.SPHERE_PROFILES)))
        else:
            super().reset()

    def _reset_profile(self, profile, timing=None):
        if profile in ("v1", "v2"):
            self.v1 = ScriptedPolicy(self.env)
        elif profile is None:
            self.v1 = ScriptedPolicyMinJerk(self.env)
        elif self._plan_box:
            self.v1 = ScriptedBoxMinJerk(self.env, *profile)
        elif profile[0] == "intercept":
            self.v1 = ScriptedSphereIntercept(self.env, *profile[1:])
        else:
            self.v1 = ScriptedSphereMinJerk(self.env, *profile)
        self.v2.VMAX_CARRY = type(self.v2).VMAX_CARRY
        self.v2.VMAX_HOME = None
        if timing is not None:
            transfer, home_peak, carry_v2, home_v2, approach, *pull_timeout = timing
            self.v1.TRANSFER_STEP_LIMIT = transfer
            if pull_timeout:
                self.v1.DRAG_PULL_TIMEOUT = pull_timeout[0]
            if isinstance(self.v1, ScriptedPolicyMinJerk):
                self.v1.MJ_JOINT_PEAK = home_peak
                self.v1.MJ_APPROACH_PEAK = approach
            else:
                self.v1._phase_smoothing[10] = 1.0
            self.v2.VMAX_CARRY, self.v2.VMAX_HOME = carry_v2, home_v2
        super().reset()
        if profile == "v2":
            self.v2.reset()
            self._active, self.switched = self.v2, True

    def _choose_profile(self, profiles):
        """Kiểm tra phương án trên bản sao MuJoCo trước khi thầy thực thi trên cảnh thật."""
        env = copy(self.env)
        env.data = mujoco.MjData(env.model)
        env.render_images = False
        probe = type(self)(env)
        mujoco.mj_copyData(env.data, env.model, self.env.data)
        at_box_edge = self._plan_box and probe.v1._arc_needed()
        profile, timing, result = self._timed_profile(probe, profiles, self._plan_box and not at_box_edge)
        if result is None or not self.TIMINGS:
            return profile, timing
        if at_box_edge:
            for candidate in self.BOX_EDGE_GRASPS:
                direct = self._trial(probe, candidate, self.TIMINGS[0])
                if (direct.steps < result.steps and direct.score[1] == 1
                        and self._trial_preserves_grasp(direct, result)):
                    profile, timing, result = candidate, self.TIMINGS[0], direct
            if result.pulled and result.revisits:
                short_pull = (*(timing or self.TIMINGS[0]), self.BOX_EDGE_PULL_TIMEOUT)
                pulled = self._trial(probe, profile, short_pull)
                if pulled.steps < result.steps and self._trial_preserves_grasp(pulled, result):
                    timing = short_pull
        if self._plan_sphere:
            return self._refine_sphere_pick(probe, profile, timing, result)
        return profile, timing

    @staticmethod
    def _trial_preserves_grasp(candidate, baseline):
        return (candidate.ok and candidate.revisits <= baseline.revisits
                and all(a <= b for a, b in zip(candidate.score, baseline.score)))

    def _refine_sphere_pick(self, probe, profile, timing, result):
        pads = result.pad_heights
        high = pads and min(pads) >= self.SPHERE_HIGH_PAD_MIN and max(pads) >= self.SPHERE_HIGH_PAD_MAX
        slow_retry = result.revisits and result.steps > self.SPHERE_RETRY_STEP_THRESHOLD
        if not (high or slow_retry):
            return profile, timing

        candidates = (*self.SPHERE_CENTER_GRASPS, *self.SPHERE_RETRY_GRASPS) if slow_retry else self.SPHERE_CENTER_GRASPS
        for candidate in candidates:
            if (candidate, self.TIMINGS[0]) == (profile, timing):
                continue
            trial = self._trial(probe, candidate, self.TIMINGS[0])
            if trial.steps > result.steps or not self._trial_preserves_grasp(trial, result):
                continue
            if high and not (trial.pad_heights and max(map(abs, trial.pad_heights)) <= self.SPHERE_CENTER_BAND):
                continue
            if high or trial.steps < result.steps or trial.revisits < result.revisits:
                profile, timing, result = candidate, self.TIMINGS[0], trial
                if high and not slow_retry:
                    break
        return profile, timing

    def _timed_profile(self, probe, profiles, prefer_single_grasp):
        if prefer_single_grasp and probe.v1._box_pan_r() >= probe.v1.DRAG_R:
            # Seed 87 và 100: dời điểm kẹp 4 mm và nâng trực tiếp giúp gắp một lần.
            grips = ((0, .004, True), (0, 0, True), *self.BOX_GRASPS)
            lifts = tuple((*grip, carry) for carry in ("direct", "hold") for grip in dict.fromkeys(grips))
            profiles = (*lifts, *profiles, self.BOX_FAR_FIXED_GRASP)
        best, best_result = "v1", None
        for profile in (None, *profiles):
            result = self._trial(probe, profile)
            if not result.ok:
                continue
            score = result.score
            if best_result is None or score < best_result.score:
                best, best_result = profile, result
            if not prefer_single_grasp or score == (False, 1, 1):
                break
        if best_result is not None and self.TIMINGS:
            original_steps, original_score = best_result.steps, best_result.score
            chosen_timing = None
            shortest = best_result
            for timing in self.TIMINGS:
                result = self._trial(probe, best, timing)
                if result.ok and result.steps < shortest.steps and all(a <= b for a, b in zip(result.score, original_score)):
                    if not self._plan_sphere or result.steps < self.SLOW_SPHERE_CYCLE_STEPS:
                        return best, timing, result
                    chosen_timing, shortest = timing, result
                    break
            if self._plan_sphere and shortest.steps >= self.SLOW_SPHERE_CYCLE_STEPS:
                for profile in ("v1", "v2"):
                    if profile == best:
                        continue
                    result = self._trial(probe, profile)
                    if result.ok and result.steps < shortest.steps and all(a <= b for a, b in zip(result.score, original_score)):
                        best, chosen_timing, shortest = profile, None, result
                if shortest.steps < original_steps:
                    return best, chosen_timing, shortest
            # Nếu đổi nhịp chưa rút được bước, thử một cách kẹp có sẵn để tránh vòng thử lại dài.
            for profile in (None, *profiles):
                if profile == best:
                    continue
                result = self._trial(probe, profile)
                if result.ok and result.steps < original_steps and all(a <= b for a, b in zip(result.score, original_score)):
                    return profile, None, result
        return best, None, best_result

    def _trial(self, probe, profile, timing=None, *, track_contacts=False):
        env = probe.env
        mujoco.mj_copyData(env.data, env.model, self.env.data)
        env._step_count = self.env._step_count
        env._grasped = self.env._grasped
        probe._reset_profile(profile, timing)
        obs = env._get_obs()
        closes, descents, last_grip = 0, 0, 1.0
        grasped = env._grasped
        previous_phase = None
        entered = set()
        revisits = 0
        pad_heights = None
        pulled = False
        contacts = [0, 0, 0]
        if track_contacts:
            body = lambda name: mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, name)
            camera = body("wrist_cam_arm")
            bodies = ({body("table"), body("world")}, body("bowl"), camera,
                      {camera, body("gripper"), body("wrist"), body("moving_jaw_so101_v1")})
        initial_z = env.data.xpos[env._box_body, 2]
        for step in range(1, config.MAX_EPISODE_STEPS + 1):
            phase = probe._phase
            pulled |= getattr(probe._active, "_drag", None) == "pull"
            if phase != previous_phase:
                revisits += phase in entered
                entered.add(phase)
                previous_phase = phase
            action = probe.select_action(obs)
            obs, _, terminated, truncated, info = env.step(action)
            if track_contacts and 5 <= phase <= 7:
                for index, hit in enumerate(_carry_contacts(env, bodies)):
                    contacts[index] += int(hit)
            closes += phase == 4 and last_grip >= 0 > action[5]
            descents += phase == 2 and probe._phase == 3
            last_grip = action[5]
            grasped |= env._grasped
            if self._plan_sphere and pad_heights is None and 3 <= phase <= 6 and env._grasped:
                center_z = env.data.xpos[env._box_body, 2]
                if center_z - initial_z >= self.SPHERE_CENTER_LIFT:
                    pad_heights = self._loaded_pad_heights(env, center_z)
            if terminated or truncated or probe._phase > 10:
                break
        _, info, waited = env.settle(action, info)
        return _TrialResult(info["success"], step + waited, (not grasped, closes, descents), revisits, pad_heights,
                            pulled, tuple(contacts))

    @staticmethod
    def _loaded_pad_heights(env, center_z):
        """Lấy cao độ điểm đang chịu lực lớn nhất của mỗi má kẹp so với tâm bóng."""
        pads = {}
        force = np.empty(6)
        for i in range(env.data.ncon):
            contact = env.data.contact[i]
            for pad, other in ((contact.geom1, contact.geom2), (contact.geom2, contact.geom1)):
                name = env._pad_geoms.get(pad)
                if name is None or env.model.geom_bodyid[other] != env._box_body:
                    continue
                mujoco.mj_contactForce(env.model, env.data, i, force)
                if force[0] > .05 and (name not in pads or force[0] > pads[name][0]):
                    pads[name] = float(force[0]), float(contact.pos[2] - center_z)
        if len(pads) == 2:
            return pads["fixed"][1], pads["moving"][1]
        return None


class _FinalTiming(NamedTuple):
    original: tuple | None
    carry: tuple[float, float] | None = None
    lower: float = 0.
    place_threshold: float | None = None
    short_arc: bool = False


class ScriptedPolicyFinal(ScriptedPolicyHybridMinJerk):
    """Chọn phương án ngắn nhất qua lượt thử vật lý cho cảnh một vật 4 cm."""

    # Nhịp 0,030/2,0 rút bước trên PPO nhưng làm hỏng 7 hộp; nhịp 0,024/1,8 là ứng viên dự phòng.
    FAST_CARRY = (.030, 2.0)
    MEDIUM_CARRY = (.024, 1.8)
    # Thử riêng 7 ca đặt hỏng: hạ 3–6 mm sửa 6 ca, hạ 9 mm và siết ngưỡng tới đích sửa 2087.
    LOWER_PLACEMENTS = ((.003, None), (.006, None), (.009, .025))
    # Hộp 1016 giảm 373 → 242 bước khi rút vòng cung từ 102 còn 82 bước.
    SHORT_ARC_STEPS = {"above": 16, "down": 12, "close": 12, "sweep": 24, "release": 8, "raise": 10}

    def _reset_profile(self, profile, timing=None):
        if not isinstance(timing, _FinalTiming):
            super()._reset_profile(profile, timing)
            return
        super()._reset_profile(profile, timing.original)
        if timing.carry is not None:
            self.v1.TRANSFER_STEP_LIMIT, self.v2.VMAX_CARRY = timing.carry
        self.v1._place_z -= timing.lower
        if timing.place_threshold is not None:
            self.v1.PLACE_ARRIVE_THRESHOLD = timing.place_threshold
        if timing.short_arc:
            self.v1.ARC_STEPS = self.SHORT_ARC_STEPS

    def _trial(self, probe, profile, timing=None):
        result = super()._trial(probe, profile, timing, track_contacts=isinstance(timing, _FinalTiming))
        if self._planning_results is not None:
            self._planning_results[profile, timing] = result
        return result

    def __init__(self, env):
        super().__init__(env)
        self._planning_results = None

    def _choose_profile(self, profiles):
        tried = {}
        self._planning_results = tried
        try:
            profile, original = super()._choose_profile(profiles)
        finally:
            self._planning_results = None
        if not self.TIMINGS:
            return profile, original

        shadow = copy(self.env)
        shadow.data = mujoco.MjData(shadow.model)
        shadow.render_images = False
        mujoco.mj_copyData(shadow.data, shadow.model, self.env.data)
        probe = type(self)(shadow)
        at_edge = self._plan_box and probe.v1._arc_needed()
        baseline = tried.get((profile, original))
        if baseline is None:
            baseline = self._trial(probe, profile, original)
        best, best_timing = baseline, original
        baseline_contacts = None

        def consider(timing):
            nonlocal best, best_timing, baseline_contacts
            result = self._trial(probe, profile, timing)
            if (result.ok and not result.score[0] and (not best.ok or result.steps < best.steps)
                    and (not baseline.ok or self._trial_preserves_grasp(result, baseline))):
                if any(result.contacts) and baseline_contacts is None:
                    baseline_contacts = self._trial(probe, profile, _FinalTiming(original)).contacts
                if not any(result.contacts) or all(a <= b for a, b in zip(result.contacts, baseline_contacts)):
                    best, best_timing = result, timing
            return result

        fast = consider(_FinalTiming(original, self.FAST_CARRY))
        if best_timing == original:
            consider(_FinalTiming(original, self.MEDIUM_CARRY))
        if self._plan_box and not fast.ok:
            for lower, threshold in self.LOWER_PLACEMENTS:
                consider(_FinalTiming(original, self.FAST_CARRY, lower, threshold))
        direct_roll = isinstance(profile, tuple) and len(profile) >= 6 and profile[5]
        if at_edge and profile not in ("v1", "v2") and not direct_roll:
            arc = consider(_FinalTiming(original, short_arc=True))
            if arc.ok:
                consider(_FinalTiming(original, self.FAST_CARRY, short_arc=True))
        return profile, best_timing
