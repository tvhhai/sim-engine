"""Scripted policy V2: lập kế hoạch kẹp MỘT LẦN mỗi lần thử, rồi chạy theo waypoint (không tính lại đích mỗi bước như V1).

Tham khảo cách các repo khác làm gắp bằng code cứng:
  - ManiSkill PickCube: tính tư thế kẹp từ hộp bao quanh vật, điểm chuẩn bị lùi dọc trục kẹp, đi thẳng tới điểm kẹp, đóng kẹp, mang đi.
  - ACT scripted_policy: danh sách waypoint nội suy theo thời gian.
  - asdqwe9501/Soarm101_Mujoco (expert/ik.py): IK DLS nhiều điểm xuất phát, khớp 1 khởi tạo bằng atan2, chọn lời giải ít nghiêng nhất.
  - sxi9/so101-workcell: chọn cặp mặt hộp theo khoảng trống của hàm kẹp, kiểm tra sau mỗi kỹ năng và thử lại.

Ba phần:
  1. IK (_ik): DLS trên 4 khớp (wrist_roll khoá), luôn trong giới hạn khớp (khớp chạm giới hạn mà bước còn đẩy ra ngoài thì bị khoá),
     nhiều điểm xuất phát; mục tiêu chính là vị trí, mục tiêu phụ (trọng số nhỏ) là trục kẹp chúc xuống.
  2. Kế hoạch kẹp (_plan_grasp): thử 4 cặp mặt hộp; với mỗi cặp tìm roll để trục kẹp song song mặt hộp TẠI tư thế kẹp, kiểm tra va chạm
     dọc đường hạ thẳng (hàm mở - hộp, camera - bàn/vai, hàm - bát), chọn phương án sạch và ít nghiêng nhất.
  3. Thực hiện: di chuyển tự do = nội suy min-jerk trong không gian khớp; hạ / nhấc / kéo = đường thẳng Cartesian (IK từng điểm,
     khởi động từ điểm trước). Hết mỗi đoạn chờ tay bám kịp lệnh rồi mới kiểm tra và sang phase kế.

Đánh số phase giữ như V1 để công cụ chấm (select_seeds, make_replay) so sánh được:
  0 nhấc khỏi home, 1 tới phía trên điểm kẹp, 2 hạ tới điểm chuẩn bị, 3 hạ tới điểm kẹp, 4 đóng kẹp, 5 nhấc (hoặc kéo hộp xa vào),
  6 mang tới bát, 7 hạ vào bát, 8 thả, 9 rút lên + kiểm tra, 10 về home, 11 xong.

Chạy thử: python -m sim_engine.tools.select_seeds --policy sim_engine.policies.scripted_v2:ScriptedPolicyV2
"""

import math

import mujoco
import numpy as np

from .base import BasePolicy
from .. import box_sampling, config


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def _minjerk(s):
    return s * s * s * (10 - 15 * s + 6 * s * s)


class ScriptedPolicyV2(BasePolicy):
    # Hình học (m, hệ world), cùng giá trị V1
    GRASP_Z = 0.43             # đầu tay ngang tâm hộp
    PRE_Z = 0.465              # điểm chuẩn bị: hàm còn cao hơn đỉnh hộp (0,45)
    HOVER_Z = 0.49             # độ cao lơ lửng phía trên điểm kẹp
    CARRY_Z = 0.51
    PLACE_Z = 0.475
    GRASP_SIDE_OFFSET = 0.022  # đầu tay lệch khỏi tâm hộp về phía hàm cố định, theo trục kẹp

    # IK
    IK_ITERS = 60
    IK_DAMPING = 0.03
    IK_TOL = 0.001
    IK_MAX_DQ = 0.2            # rad mỗi vòng lặp
    IK_STARTS = 6              # số điểm xuất phát (1 từ tư thế hiện tại, 1 khớp 1 theo hướng đích, còn lại ngẫu nhiên)
    IK_OK = 0.003              # sai số dưới mức này coi là tới được; trong các lời giải tới được chọn lời giải ít nghiêng nhất
    TOOL_DOWN_W = 0.05         # trọng số mục tiêu "trục kẹp chúc xuống" so với vị trí

    # Chuyển động
    DT = config.PHYSICS_SUBSTEPS * config.PHYSICS_TIMESTEP
    VMAX = 2.0                 # rad/s, di chuyển tự do
    VMAX_CARRY = 1.2           # rad/s, khi đang mang hộp
    CART_SPEED = 0.15          # m/s, đường thẳng không mang hộp
    CART_SPEED_SLOW = 0.08     # m/s, hạ cuối tới điểm kẹp, nhấc hộp, kéo hộp
    SETTLE_TOL = 0.03          # rad: hết đoạn, chờ tới khi mọi khớp lệch lệnh dưới mức này
    SETTLE_MAX = 12            # bước chờ tối đa

    # Kẹp
    CLOSE_STEPS = 25
    MAX_GRASP_TRIES = 3
    MAX_REGRASPS = 2
    JAW_MARGIN = 0.001         # hàm mở cách hộp tối thiểu dọc đường hạ (tới ngay trên đỉnh hộp)
    CAM_MARGIN = 0.003         # camera cổ tay cách bàn/vai/cánh tay tối thiểu
    BOWL_MARGIN = 0.005        # hàm mở cách bát tối thiểu (chỉ xét khi hộp gần bát)
    SELF_MARGIN = 0.003        # cổ tay / hàm / camera cách đế, vai, cánh tay trên tối thiểu
    BLOCK_TOL = 0.1            # rad: hết đoạn mà khớp còn lệch lệnh quá mức này = tay bị chặn (chạm), lập kế hoạch lại
    REACH_OK = 0.006           # sai số IK tối đa ở điểm kẹp để coi phương án là "tới được"
    REACH_MAX = 0.015          # sai số lớn hơn mức này: phương án bị loại (hộp xa thật sự có sai số 5-14 mm nên không loại ở REACH_OK)
    PATH_SAMPLES = 5

    # Hộp xa: tay vươn hết cỡ, trục kẹp nghiêng nhiều, hộp tuột khi nhấc (đo ở V1). Kẹp, kéo trượt vào r = DRAG_TARGET, nhả, gắp lại
    DRAG_R = 0.40
    DRAG_TILT_DEG = 25.0       # cũng kéo khi kế hoạch kẹp nghiêng quá chừng này hoặc sai số với tới quá DRAG_ERR (hộp ở r > DRAG_TARGET + 1 cm)
    DRAG_ERR = 0.005
    DRAG_TARGET = 0.33
    RADIAL_R = 0.38            # hộp xa hơn mức này (chưa kéo): thêm phương án kẹp theo hướng bán kính để kéo vào
    DRAG_LIFT = 0.004
    DRAG_RELEASE_STEPS = 12

    RELEASE_OPEN_CMD = 0.55
    RELEASE_RAMP = 0.10
    RELEASE_CLEAR_RAD = 0.8
    SETTLE_WAIT_MAX = 15
    SETTLE_STILL_STEPS = 3

    def __init__(self, env):
        self.env = env
        self.model = env.model
        self.data = env.data
        self._ik_data = mujoco.MjData(self.model)
        m = self.model
        self._joint_ids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n) for n in config.JOINT_NAMES[:5]]
        self._qpos_adr = np.array([m.jnt_qposadr[j] for j in self._joint_ids])
        self._dof_ids = [m.jnt_dofadr[j] for j in self._joint_ids]
        self._lo = m.jnt_range[self._joint_ids, 0] + 0.01
        self._hi = m.jnt_range[self._joint_ids, 1] - 0.01
        self._ee_site = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "ee_site")
        self._gripper_body = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "gripper")
        self._gripper_joint = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "gripper")
        self._grip_adr = m.jnt_qposadr[self._gripper_joint]
        bid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)
        geoms = lambda b: [g for g in range(m.ngeom) if m.geom_bodyid[g] == b and m.geom_contype[g] != 0]
        arm = [bid(n) for n in ("shoulder", "upper_arm", "lower_arm", "wrist", "gripper", "moving_jaw_so101_v1", "wrist_cam_arm")]
        arm_geoms = [g for b in arm for g in geoms(b)]
        self._jaw_box_pairs = [(a, b) for a in arm_geoms for b in geoms(env._box_body)]
        self._jaw_bowl_pairs = [(a, b) for a in arm_geoms for b in geoms(env._bowl_body)]
        # tay tự chạm: khâu cuối (cổ tay, hai hàm, camera) với khâu gần đế (đế, vai, cánh tay trên). Seed 205: hàm cố định tì vào vai
        distal = [g for n in ("wrist", "gripper", "moving_jaw_so101_v1", "wrist_cam_arm") for g in geoms(bid(n))]
        proximal = [g for n in ("base", "shoulder", "upper_arm") for g in geoms(bid(n))]
        self._self_pairs = [(a, b) for a in distal for b in proximal]
        cam = [g for g in range(m.ngeom) if m.geom_bodyid[g] == bid("wrist_cam_arm")]
        self._cam_pairs = [(a, w) for a in cam for n in ("base", "shoulder", "upper_arm", "table") for w in
                           [g for g in range(m.ngeom) if m.geom_bodyid[g] == bid(n)]]
        self.reset()

    # ------------------------------------------------------------------ vòng đời

    def reset(self):
        self._phase, self._phase_step = 0, 0
        self.last_exit, self.motion_target = None, None
        self._traj = []                     # danh sách lệnh 5 khớp (rad) còn phải chạy trong phase này
        self._settle = 0
        self._entered = False               # phase hiện tại đã dựng quỹ đạo chưa
        self._q_cmd = self.data.qpos[self._qpos_adr].copy()
        self._current_joint_targets = self._q_cmd.copy()
        self._grip_target, self._grip_cmd = 1.0, -1.0
        self._plan = None
        self._tries, self._regrasps, self._dragged = 0, 0, False
        self._drag_stage, self._drag_n = None, 0
        self._last_finger, self._lost, self._still, self._settle_wait = None, 0, 0, 0
        self._replan_lift = False           # phase 2 phải nhấc lên rồi đi lại (thử kẹp lại / sau khi kéo hộp)
        self._blocked = False
        self._failed_faces, self._failed_key = set(), None   # cặp mặt hộp đã thử hỏng ở vị trí hộp này
        self.unreachable = bool(box_sampling.is_unreachable(self.env.get_box_pos()[:2]))
        if self.unreachable:
            self._goto(10)

    def annotation(self):
        return {"phase": int(self._phase), "recovery": 0, "roll_flips": 0, "bowl_avoid": 0, "unreachable": int(self.unreachable)}

    # ------------------------------------------------------------------ IK

    def _fk(self, q):
        d = self._ik_data
        d.qpos[:] = self.data.qpos[:]
        d.qpos[self._qpos_adr] = q
        mujoco.mj_kinematics(self.model, d)
        mujoco.mj_comPos(self.model, d)
        return d

    def _tool_axis(self, d):
        a = d.site_xpos[self._ee_site] - d.xpos[self._gripper_body]
        return a / (np.linalg.norm(a) + 1e-9)

    def _ik_descend(self, target, roll, q, tool_w):
        q = np.clip(np.array(q, dtype=float), self._lo, self._hi)
        q[4] = roll
        m = self.model
        jacp, jacr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        down = np.array([0.0, 0.0, -1.0])
        lam = self.IK_DAMPING ** 2 * np.eye(6)
        for _ in range(self.IK_ITERS):
            d = self._fk(q)
            ee = d.site_xpos[self._ee_site]
            e = target - ee
            if np.linalg.norm(e) < self.IK_TOL:
                break
            axis = self._tool_axis(d)
            mujoco.mj_jacSite(m, d, jacp, jacr, self._ee_site)
            J = jacp[:, self._dof_ids]
            skew = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
            J_all = np.vstack([J, tool_w * (-skew @ jacr[:, self._dof_ids])])
            e_all = np.concatenate([e, tool_w * (down - axis)])
            free = np.array([1.0, 1.0, 1.0, 1.0, 0.0])
            for _ in range(4):                                     # khoá khớp đang ở giới hạn mà bước còn đẩy ra ngoài, giải lại
                Jf = J_all * free[None, :]
                dq = Jf.T @ np.linalg.solve(Jf @ Jf.T + lam, e_all) * free
                sat = (((q <= self._lo + 1e-4) & (dq < 0)) | ((q >= self._hi - 1e-4) & (dq > 0))) & (free > 0)
                if not sat.any():
                    break
                free[sat] = 0.0
            q = np.clip(q + np.clip(dq, -self.IK_MAX_DQ, self.IK_MAX_DQ), self._lo, self._hi)
            q[4] = roll
        d = self._fk(q)
        err = float(np.linalg.norm(target - d.site_xpos[self._ee_site]))
        tilt = math.acos(float(np.clip(-self._tool_axis(d)[2], -1.0, 1.0)))
        return q, err, tilt

    def _bearing(self, p):
        """Góc khớp 1 hướng về điểm p (atan2 quanh trục khớp 1; 0 = hướng +y, dương về phía +x)."""
        return float(np.clip(math.atan2(p[0] - config.PICK_PAN_XY[0], p[1] - config.PICK_PAN_XY[1]), self._lo[0], self._hi[0]))

    def _ik(self, target, roll, q0, multi=True, tool_w=None):
        """IK nhiều điểm xuất phát. Trong các lời giải sai số < IK_OK chọn lời giải ít nghiêng nhất (kèm phạt nhẹ khoảng cách tới q0);
        không có thì lấy lời giải sai số nhỏ nhất. Trả (q, sai số, góc nghiêng trục kẹp so với phương thẳng đứng)."""
        starts = self._starts(target, q0) if multi else [np.array(q0, dtype=float)]
        sols = self._ik_all(target, roll, starts, tool_w)
        return sols[0]

    def _ik_all(self, target, roll, starts, tool_w=None):
        """Giải IK từ mọi điểm xuất phát; trả danh sách (q, sai số, nghiêng) đã bỏ trùng, xếp: tới được (< IK_OK) và ít nghiêng trước."""
        tool_w = self.TOOL_DOWN_W if tool_w is None else tool_w
        out = []
        for s in starts:
            q, err, tilt = self._ik_descend(target, roll, s, tool_w)
            if all(float(np.max(np.abs(q - o[0]))) > 0.1 for o in out):
                out.append((q, err, tilt))
        key = lambda o: (0, o[2] + 0.05 * float(np.linalg.norm(o[0][:4] - starts[0][:4]))) if o[1] < self.IK_OK else (1, o[1])
        return sorted(out, key=key)

    def _starts(self, target, q0):
        starts = [np.array(q0, dtype=float)]
        s = starts[0].copy(); s[0] = self._bearing(target); starts.append(s)
        rng = np.random.default_rng(0)
        for _ in range(self.IK_STARTS - 2):
            s = rng.uniform(self._lo, self._hi); s[0] = self._bearing(target) + rng.normal(0.0, 0.15); starts.append(s)
        return starts

    # ------------------------------------------------------------------ đo hình học trên bản sao IK

    def _clamp_angle(self, q):
        """Góc (rad, quanh z) của trục kẹp ngang (từ hàm cố định sang hàm di động) ở tư thế q."""
        d = self._fk(q)
        c = np.cross(d.xaxis[self._gripper_joint], [0.0, 0.0, -1.0])
        return math.atan2(c[1], c[0])

    def _min_dist(self, q, pairs, grip=None):
        d = self._fk(q)
        if grip is not None:
            d.qpos[self._grip_adr] = grip
            mujoco.mj_kinematics(self.model, d)
        fromto = np.zeros(6)
        return min((mujoco.mj_geomDistance(self.model, d, a, b, 0.06, fromto) for a, b in pairs), default=1.0)

    def _box_yaw(self):
        qb = self.data.xquat[self.env._box_body]
        return math.atan2(2 * (qb[0] * qb[3] + qb[1] * qb[2]), 1 - 2 * (qb[2] ** 2 + qb[3] ** 2))

    # ------------------------------------------------------------------ kế hoạch kẹp

    def _roll_for_clamp(self, G, phi, q0):
        """Tìm roll để trục kẹp có góc phi tại điểm kẹp G (Newton với đạo hàm số; IK giải lại sau mỗi lần chỉnh roll)."""
        r = float(q0[4])
        q = q0
        for it in range(5):
            q, err, tilt = self._ik(G, r, q, multi=(it == 0))
            a = self._clamp_angle(q)
            diff = _wrap(phi - a)
            if abs(diff) < math.radians(1.0):
                break
            q2 = q.copy(); q2[4] = r + 0.05
            k = _wrap(self._clamp_angle(q2) - a) / 0.05
            if abs(k) < 0.2:
                k = 1.0 if k >= 0 else -1.0
            r = r + diff / k
            for w in (0.0, 2 * math.pi, -2 * math.pi):            # đưa roll về trong giới hạn khớp nếu được
                if self._lo[4] <= r + w <= self._hi[4]:
                    r += w
                    break
            else:
                return None
        q, err, tilt = self._ik(G, r, q, multi=False)
        if abs(_wrap(phi - self._clamp_angle(q))) > math.radians(3.0):
            return None
        return r, q, err, tilt

    def _plan_grasp(self):
        """Thử 4 cặp mặt hộp (trục kẹp = yaw hộp + k·90°). Mỗi phương án: điểm kẹp G, roll, tư thế lơ lửng / chuẩn bị / kẹp, và kiểm tra
        dọc đường hạ thẳng từ HOVER_Z tới ngay trên đỉnh hộp (hàm mở không chạm hộp) và tới GRASP_Z (camera, bát).
        Chọn: sạch và tới được trước, rồi ít nghiêng nhất, rồi roll ít phải xoay nhất."""
        box = self.env.get_box_pos()
        if self._failed_key is not None and np.linalg.norm(box[:2] - self._failed_key) > 0.02:   # hộp đã dời hẳn chỗ: quên các lần hỏng cũ
            self._failed_faces, self._failed_key = set(), None
        yaw = self._box_yaw()
        q_cur = self.data.qpos[self._qpos_adr].copy()
        near_bowl = np.linalg.norm(box[:2] - self.env.get_bowl_pos()[:2]) <= 0.20
        top = box[2] + config.BOX_HALF
        plans = []
        phis = [_wrap(yaw + k * math.pi / 2) for k in range(4)]
        if not self._dragged and self._box_r(box) >= self.RADIAL_R:
            # Hộp xa: thêm phương án kẹp THEO HƯỚNG BÁN KÍNH (hàm cố định phía đế) để điểm kẹp gần đế hơn GRASP_SIDE_OFFSET và với tới được;
            # chỉ dùng để kéo hộp vào (_need_drag), không mang đi. Seed 87 (r 40,5 cm): 4 cặp mặt đều sai số 11 mm, hai hàm khép vào khoảng trống
            phis.append(math.atan2(box[1] - config.PICK_PAN_XY[1], box[0] - config.PICK_PAN_XY[0]))
        for k, phi in enumerate(phis):
            c = np.array([math.cos(phi), math.sin(phi)])
            G = np.array([box[0] - self.GRASP_SIDE_OFFSET * c[0], box[1] - self.GRASP_SIDE_OFFSET * c[1], self.GRASP_Z])
            sol = self._roll_for_clamp(G, phi, q_cur)
            if sol is None:
                continue
            r, q_g0, _, _ = sol
            failed = k in self._failed_faces
            # Mỗi cặp mặt xét nhiều lời giải IK ở điểm kẹp (vd tay nghiêng để hàm tránh vai khi hộp sát đế), lấy lời giải có đường hạ sạch
            cands = self._ik_all(G, r, self._starts(G, q_g0))
            cands = [c for c in cands if c[1] <= self.REACH_MAX][:4] or cands[:1]       # bỏ lời giải không tới được (vd kẹp chúc ngược lên)
            for q_g, err, tilt in cands:
                q_h, _, _ = self._ik(np.array([G[0], G[1], self.HOVER_Z]), r, q_g, multi=False)
                clear, q = 1e9, q_h
                for z in np.linspace(self.HOVER_Z, self.GRASP_Z, self.PATH_SAMPLES):   # đúng thứ tự lúc chạy: từ trên xuống, khởi động từ điểm trước
                    q, _, _ = self._ik(np.array([G[0], G[1], z]), r, q, multi=False)
                    s = min(self._min_dist(q, self._cam_pairs) / self.CAM_MARGIN,
                            self._min_dist(q, self._self_pairs, config.GRIPPER_OPEN_RAD) / self.SELF_MARGIN)
                    if z > top + 0.004:
                        s = min(s, self._min_dist(q, self._jaw_box_pairs, config.GRIPPER_OPEN_RAD) / self.JAW_MARGIN)
                    if near_bowl:
                        s = min(s, self._min_dist(q, self._jaw_bowl_pairs, config.GRIPPER_OPEN_RAD) / self.BOWL_MARGIN)
                    clear = min(clear, s)
                err = max(err, float(np.linalg.norm(self._fk(q).site_xpos[self._ee_site] - G)))
                ok = clear >= 1.0 and err <= self.REACH_OK
                # thứ tự: chưa hỏng ở lần trước, tới được, sạch va chạm, rồi ít nghiêng / ít phải xoay roll
                score = (int(failed), 0 if err <= self.REACH_MAX else 1, 0 if ok else 1, 0 if clear >= 1.0 else 1,
                         tilt + 0.15 * abs(r - q_cur[4]) + 20 * max(0.0, err - self.REACH_OK))
                plans.append({"score": score, "k": k, "roll": r, "G": G, "q_h": q_h, "err": err, "tilt": tilt, "clear": clear, "ok": ok})
        if not plans:
            return None
        return min(plans, key=lambda p: p["score"])

    # ------------------------------------------------------------------ dựng quỹ đạo

    def _seg_joint(self, q_from, q_to, vmax):
        dq = np.asarray(q_to) - np.asarray(q_from)
        n = max(3, int(math.ceil(float(np.max(np.abs(dq))) / vmax / self.DT)))
        return [np.asarray(q_from) + dq * _minjerk((i + 1) / n) for i in range(n)]

    def _seg_line(self, p_from, p_to, roll, q_from, speed):
        """Đường thẳng Cartesian từ p_from tới p_to, roll giữ nguyên. Trả (danh sách lệnh khớp, q cuối)."""
        p_from, p_to = np.asarray(p_from, float), np.asarray(p_to, float)
        dist = float(np.linalg.norm(p_to - p_from))
        n = max(2, int(math.ceil(dist / speed / self.DT)))
        out, q = [], np.array(q_from, dtype=float)
        for i in range(n):                                          # mỗi bước một điểm IK, khởi động từ điểm trước (lời giải liền mạch)
            q, _, _ = self._ik(p_from + (p_to - p_from) * _minjerk((i + 1) / n), roll, q, multi=False)
            out.append(q.copy())
        return out, q

    # ------------------------------------------------------------------ điều khiển

    def _goto(self, phase, exit_reason=None):
        self._phase, self._phase_step = phase, 0
        self._traj, self._settle, self._entered = [], 0, False
        self.last_exit = exit_reason

    def _ee(self):
        return self.env.get_ee_pos()

    def _box_r(self, p=None):
        b = self.env.get_box_pos() if p is None else p
        return math.hypot(b[0] - config.PICK_PAN_XY[0], b[1] - config.PICK_PAN_XY[1])

    def _segment_done(self):
        """Quỹ đạo đã chạy hết và tay đã bám kịp lệnh (hoặc chờ quá SETTLE_MAX bước)."""
        self._blocked = False
        if self._traj:
            return False
        err = float(np.max(np.abs(self.data.qpos[self._qpos_adr] - self._q_cmd)))
        self._settle += 1
        if err < self.SETTLE_TOL:
            return True
        if self._settle > self.SETTLE_MAX:
            self._blocked = err > self.BLOCK_TOL
            return True
        return False

    def _hover_ready_traj(self):
        """Quỹ đạo từ tư thế hiện tại tới tư thế lơ lửng đã lập kế hoạch ngay trên điểm kẹp (nhấc thẳng lên trước nếu đang thấp)."""
        p = self._plan
        traj, q = [], self._q_cmd.copy()
        ee = self._ee_cmd_pos()
        if ee[2] < self.HOVER_Z - 0.01:
            seg, q = self._seg_line(ee, [ee[0], ee[1], self.HOVER_Z], q[4], q, self.CART_SPEED)
            traj += seg
        traj += self._seg_joint(q, p["q_h"], self.VMAX)
        return traj

    def _fail_face(self, reason):
        """Lần thử với cặp mặt hộp hiện tại hỏng: ghi lại để lần lập kế hoạch sau ưu tiên cặp mặt khác (thử lại phải khác lần trước)."""
        if self._plan is not None:
            self._failed_faces.add(self._plan["k"])
        self._failed_key = self.env.get_box_pos()[:2].copy()
        self.last_exit = reason

    def _retry_from_above(self, reason):
        """Nhấc lên, lập kế hoạch lại (tránh cặp mặt vừa hỏng) và hạ lại; quá MAX_GRASP_TRIES lần thì về home."""
        self._fail_face(reason)
        self._tries += 1
        if self._tries < self.MAX_GRASP_TRIES:
            self._replan_lift = True
            self._goto(2, reason)
        else:
            self._goto(10, reason)

    def select_action(self, observation):
        self.motion_target = None
        exit_now = None
        box, ee = self.env.get_box_pos(), self._ee()
        grasped = self.env._grasped

        if self._phase <= 4 and not grasped and self._phase not in (10, 11) and box_sampling.is_unreachable(box[:2]):
            self.unreachable = True
            self._goto(10, "unreachable")

        if 5 <= self._phase <= 7 and self._drag_stage is None:     # hộp tuột khi đang mang
            self._lost = self._lost + 1 if np.linalg.norm(box - ee) > 0.05 else 0
            if self._lost > 3:
                self._lost = 0
                if self._regrasps < self.MAX_REGRASPS and not box_sampling.is_unreachable(box[:2]):
                    self._fail_face("lost")
                    self._regrasps += 1
                    self._plan, self._replan_lift, self._tries = None, True, 0
                    self._goto(1, "lost")
                else:
                    self._goto(10, "lost")

        ph = self._phase
        if ph == 0:
            if not self._entered:
                home = self._ee()
                q_up, _, _ = self._ik(np.array([home[0], home[1], self.CARRY_Z]), self._q_cmd[4], self._q_cmd, multi=False, tool_w=0.0)
                self._traj = self._seg_joint(self._q_cmd, q_up, self.VMAX)
                self._entered = True
            if self._segment_done():
                self._goto(1, "arrived")
        elif ph == 1:
            if not self._entered:
                self._plan = self._plan_grasp()
                if self._plan is None:
                    self._goto(10, "no_plan")
                else:
                    self._traj = self._hover_ready_traj()
                    self._replan_lift = False
                    self._entered = True
            elif self._segment_done():
                if self._blocked:
                    self._retry_from_above("blocked")
                else:
                    self._goto(2, "arrived")
            self._grip_target = 1.0
        elif ph == 2:
            if not self._entered:
                if self._replan_lift or self._plan is None:              # thử lại / sau khi kéo hộp: lập kế hoạch lại và đi lại từ trên cao
                    self._plan = self._plan_grasp()
                    if self._plan is None:
                        self._goto(10, "no_plan")
                        return self._action()
                    self._traj = self._hover_ready_traj()
                    self._replan_lift = False
                    p = self._plan
                    H = np.array([p["G"][0], p["G"][1], self.HOVER_Z])
                    seg, _ = self._seg_line(H, [p["G"][0], p["G"][1], self.PRE_Z], p["roll"], p["q_h"], self.CART_SPEED)
                    self._traj += seg
                else:
                    p = self._plan
                    seg, _ = self._seg_line(self._ee_cmd_pos(), [p["G"][0], p["G"][1], self.PRE_Z], p["roll"], self._q_cmd, self.CART_SPEED)
                    self._traj = seg
                self._entered = True
            elif self._segment_done():
                if self._blocked:
                    self._retry_from_above("blocked")
                else:
                    self._goto(3, "arrived")
            self._grip_target = 1.0
        elif ph == 3:
            if not self._entered:
                p = self._plan
                seg, _ = self._seg_line(self._ee_cmd_pos(), p["G"], p["roll"], self._q_cmd, self.CART_SPEED_SLOW)
                self._traj = seg
                self._entered = True
            elif self._segment_done():
                if self._blocked:
                    self._retry_from_above("blocked")
                else:
                    self._goto(4, "arrived")
            self._grip_target = 1.0
        elif ph == 4:
            self._grip_target = -1.0
            self._phase_step += 1
            fo = float(self.data.qpos[self._grip_adr])
            settled = self._last_finger is not None and abs(fo - self._last_finger) < 0.004
            self._last_finger = fo
            if grasped and settled:
                self._last_finger = None
                self._goto(5, "arrived")
            elif self._phase_step > self.CLOSE_STEPS:
                self._last_finger = None
                self._retry_from_above("timeout")
            return self._action()
        elif ph == 5:
            self._grip_target = -1.0
            if self._drag_stage is None and not self._entered and not self._dragged and self._need_drag():
                self._drag_stage, self._dragged = "pull", True
            if self._drag_stage is not None:
                self._run_drag()
            else:
                if not self._entered:
                    ee = self._ee_cmd_pos()
                    seg, _ = self._seg_line(ee, [ee[0], ee[1], self.CARRY_Z], self._plan["roll"], self._q_cmd, self.CART_SPEED_SLOW)
                    self._traj = seg
                    self._entered = True
                elif self._segment_done():
                    self._goto(6, "arrived")
        elif ph == 6:
            self._grip_target = -1.0
            if not self._entered:
                bowl = self.env.get_bowl_pos()
                off = np.clip((box - ee)[:2], -0.04, 0.04)
                tgt = np.array([bowl[0] - off[0], bowl[1] - off[1], self.CARRY_Z])
                self._place_xy = tgt[:2].copy()
                q_t, _, _ = self._ik(tgt, self._q_cmd[4], self._q_cmd, multi=True, tool_w=0.1)
                self._traj = self._seg_joint(self._q_cmd, q_t, self.VMAX_CARRY)
                self._entered = True
            elif self._segment_done():
                self._goto(7, "arrived")
        elif ph == 7:
            self._grip_target = -1.0
            if not self._entered:
                seg, _ = self._seg_line(self._ee_cmd_pos(), [self._place_xy[0], self._place_xy[1], self.PLACE_Z], self._q_cmd[4], self._q_cmd,
                                        self.CART_SPEED_SLOW)
                self._traj = seg
                self._entered = True
            elif self._segment_done():
                self._goto(8, "arrived")
        elif ph == 8:
            self._grip_target = self.RELEASE_OPEN_CMD
            self._phase_step += 1
            fo = float(self.data.qpos[self._grip_adr])
            if (not grasped and fo >= self.RELEASE_CLEAR_RAD) or self._phase_step > 20:
                self._still, self._settle_wait = 0, 0
                self._goto(9, "arrived")
            return self._action()
        elif ph == 9:
            self._grip_target = self.RELEASE_OPEN_CMD
            if not self._entered:
                ee = self._ee_cmd_pos()
                seg, _ = self._seg_line(ee, [ee[0], ee[1], self.CARRY_Z], self._q_cmd[4], self._q_cmd, self.CART_SPEED)
                self._traj = seg
                self._entered = True
            elif self._segment_done():
                self._after_retract(box)
        elif ph == 10:
            self._grip_target = -1.0
            if not self._entered:
                self._traj = self._seg_joint(self._q_cmd, np.array(config.HOME_QPOS_RAD[:5], dtype=float), self.VMAX)
                self._entered = True
            elif self._segment_done():
                self._goto(11, "arrived")
        if self._phase not in (4, 8):
            self._phase_step += 1
        return self._action()

    def _need_drag(self):
        """Hộp xa (r >= DRAG_R), hoặc kế hoạch kẹp nghiêng quá DRAG_TILT_DEG / sai số với tới quá DRAG_ERR mà hộp còn xa hơn DRAG_TARGET + 1 cm:
        kẹp xiên dễ tuột khi nhấc (V1, seed 55: lần kẹp nào ở r >= 41 cm cũng tuột), nên kéo vào trước."""
        r = self._box_r()
        if r >= self.DRAG_R:
            return True
        p = self._plan
        return p is not None and r > self.DRAG_TARGET + 0.01 and (p["tilt"] > math.radians(self.DRAG_TILT_DEG) or p["err"] > self.DRAG_ERR)

    def _ee_cmd_pos(self):
        """Vị trí đầu tay ứng với lệnh khớp hiện tại (điểm bắt đầu của đoạn kế, không phụ thuộc tay thật trễ bao nhiêu)."""
        return self._fk(self._q_cmd).site_xpos[self._ee_site].copy()

    def _run_drag(self):
        """Kéo hộp xa vào: "pull" kéo trượt theo hướng bán kính tới r = DRAG_TARGET (kẹp chặt, nhấc DRAG_LIFT), "release" mở kẹp,
        rồi lập kế hoạch lại và gắp lại từ phase 2 (nhấc lên trước)."""
        if self._drag_stage == "pull":
            self._grip_target = -1.0
            if not self._entered:
                ee = self._ee_cmd_pos()
                pan = np.array(config.PICK_PAN_XY)
                v = ee[:2] - pan
                r = float(np.linalg.norm(v))
                r_t = r - (self._box_r() - self.DRAG_TARGET)
                goal = np.array([*(pan + v * (r_t / r)), ee[2] + self.DRAG_LIFT])
                seg, _ = self._seg_line(ee, goal, self._q_cmd[4], self._q_cmd, self.CART_SPEED_SLOW)
                self._traj = seg
                self._entered = True
            elif self._segment_done():
                self._drag_stage, self._drag_n = "release", 0
        elif self._drag_stage == "release":
            self._grip_target = 1.0
            self._drag_n += 1
            if self._drag_n >= self.DRAG_RELEASE_STEPS:
                self._drag_stage = None
                self._plan, self._replan_lift, self._tries = None, True, 0
                self._goto(2, "drag")

    def _after_retract(self, box):
        """Phase 9 xong (tay đã ở trên bát): chờ hộp nằm yên, hộp ngoài bát mà còn gắp được thì gắp lại, không thì về home."""
        if box_sampling.is_unreachable(box[:2]):
            self.unreachable = True
            self._goto(10, "unreachable")
            return
        adr = self.env._box_dof_adr
        speed = float(np.linalg.norm(self.data.qvel[adr:adr + 3]))
        self._still = self._still + 1 if speed < config.SETTLED_SPEED else 0
        if self._still < self.SETTLE_STILL_STEPS and self._settle_wait < self.SETTLE_WAIT_MAX:
            self._settle_wait += 1
            return
        in_bowl = np.linalg.norm(box[:2] - self.env.get_bowl_pos()[:2]) < self.env.bowl_inner
        on_table = box[2] < config.BOX_DEFAULT_POS[2] + 0.015
        if not in_bowl and on_table and self._regrasps < self.MAX_REGRASPS:
            self._regrasps += 1
            self._plan, self._replan_lift, self._tries = None, True, 0
            self._goto(1, "regrasp")
        else:
            self._goto(10, "arrived")

    def _action(self):
        if self._traj:
            self._q_cmd = np.asarray(self._traj.pop(0), dtype=float)
        self._current_joint_targets = self._q_cmd.copy()
        lo, hi = self.env._ctrl_ranges[:5].T
        a = np.zeros(7, dtype=np.float32)
        a[:5] = 2.0 * (self._q_cmd - lo) / (hi - lo) - 1.0
        # kẹp: đóng ngay, mở tăng dần để hai hàm không bật hộp văng ra
        g = self._grip_target if self._grip_target < self._grip_cmd else min(self._grip_target, self._grip_cmd + self.RELEASE_RAMP)
        self._grip_cmd = g
        a[5] = a[6] = g
        if self._plan is not None and self._phase in (1, 2, 3):
            self.motion_target = (np.array(self._plan["G"], dtype=float), 0.01)
        return np.clip(a, -1.0, 1.0)
