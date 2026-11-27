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
            T = max(self.MJ_MIN_STEPS, math.ceil(1.875 * dist / self.MJ_TCP_PEAK))
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
                step = np.clip(step, -self.LIFT_STEP_LIMIT, self.LIFT_STEP_LIMIT)
            self._smoothed_action[:5] += step
            self._smoothed_action[5:] = raw[5:]
            self._smoothed_action[5] = min(raw[5], self._prev_grip + self.RELEASE_RAMP)
            self._smoothed_action[6] = self._smoothed_action[5]
        self._prev_grip = float(self._smoothed_action[5])
        return self._smoothed_action.copy()


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


class ScriptedPolicyHybridMinJerk(ScriptedPolicyHybrid):
    """Bản kết hợp V1 + V2 như mặc định, chỉ thay V1 bằng bản minimum-jerk."""

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

    def __init__(self, env):
        self.env = env
        # Thầy dùng dự báo vật lý cho bài một bóng 4 cm. Lab nhiều vật dùng policy chạy trực tiếp.
        sphere = getattr(env, "target_type", "box") == "sphere"
        self._plan_sphere = sphere and abs(env.target_size - .04) < 1e-6 and len(env.lab_objects) == 1
        self.v1 = ScriptedPolicy(env) if sphere else ScriptedPolicyMinJerk(env)
        self.v2 = ScriptedPolicyV2(env)
        self._active = self.v1
        self.switched = False

    def reset(self):
        if self._plan_sphere:
            self._reset_sphere_profile(self._choose_sphere_profile())
        else:
            super().reset()

    def _reset_sphere_profile(self, profile):
        if profile in ("v1", "v2"):
            self.v1 = ScriptedPolicy(self.env)
        elif profile is None:
            self.v1 = ScriptedPolicyMinJerk(self.env)
        elif profile[0] == "intercept":
            self.v1 = ScriptedSphereIntercept(self.env, *profile[1:])
        else:
            self.v1 = ScriptedSphereMinJerk(self.env, *profile)
        super().reset()
        if profile == "v2":
            self.v2.reset()
            self._active, self.switched = self.v2, True

    def _choose_sphere_profile(self):
        """Kiểm tra phương án trên bản sao MuJoCo trước khi thầy thực thi trên cảnh thật."""
        env = copy(self.env)
        env.data = mujoco.MjData(env.model)
        env.render_images = False
        probe = type(self)(env)
        adr = self.env._box_dof_adr
        speed = np.linalg.norm(self.env.data.qvel[adr:adr + 2])
        intercepts = self.SPHERE_INTERCEPTS if speed >= self.SPHERE_INTERCEPT_SPEED else ()
        for profile in (None, *intercepts, *self.SPHERE_PROFILES):
            mujoco.mj_copyData(env.data, env.model, self.env.data)
            env._step_count = self.env._step_count
            env._grasped = self.env._grasped
            probe._reset_sphere_profile(profile)
            obs = env._get_obs()
            for _ in range(config.MAX_EPISODE_STEPS):
                action = probe.select_action(obs)
                obs, _, terminated, truncated, info = env.step(action)
                if terminated or truncated or probe._phase > 10:
                    break
            _, info, _ = env.settle(action, info)
            if info["success"]:
                return profile
        return "v1"
