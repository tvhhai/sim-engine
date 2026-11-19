"""Chạy policy TỪNG BƯỚC bằng phím bấm, vừa xem cửa sổ vừa xem log, để quan sát chậm.

Cách dùng (đứng ở thư mục chứa sim_engine, đã kích hoạt .venv):
    python -m sim_engine.step_policy                 # seed 2 (seed hay thất bại)
    python -m sim_engine.step_policy --seed 0        # seed 0 (thành công), để so sánh
    python -m sim_engine.step_policy --policy sim_engine.policies.my_policy:MyPolicy
    python -m sim_engine.step_policy --seed 2 --goto 8   # chạy sẵn 8 bước rồi dừng (số bước lấy từ replay.html)
    python -m sim_engine.step_policy --seed 2 --goto 8 --viewer   # cửa sổ MuJoCo viewer: xoay chuột, thấy va chạm thật

Chế độ --viewer dùng phím riêng: ENTER/SPACE = 1 bước, U/BACKSPACE = LÙI 1 bước, G = tới phase kế, K = tự chạy, J = chạy lại, +/- = tốc độ.

Phím (bấm vào cửa sổ ảnh trước):
    SPACE / ENTER : chạy ĐÚNG 1 bước (1/30 giây)
    U / BACKSPACE : LÙI 1 bước (reset cùng seed rồi chạy lại tới bước trước)
    P             : chạy tới khi policy ĐỔI PHASE rồi dừng (mỗi bước vẫn in log, chạy chậm)
    A             : bật/tắt chạy liên tục (vẫn chậm, bấm A lần nữa để dừng)
    + / -         : chạy chậm hơn / nhanh hơn (cho P và A)
    R             : chạy lại ván từ đầu cùng seed
    N / B         : seed kế / seed trước
    Q hoặc ESC    : thoát

Trên ảnh:
    chấm XANH LÁ  = ĐÍCH của phase (policy muốn điểm giữa hai ngón kẹp tới đó)
    dấu + TÍM     = ĐIỂM GIỮA HAI NGÓN KẸP thật (ee_site)
    chấm ĐỎ       = tâm hộp
    chấm + vòng XANH DƯƠNG = tâm bát và vòng "vào bát" (hộp phải nằm trong vòng này)
    đường vàng nối đầu tay với đích: càng ngắn thì càng gần tới
Log ở terminal: mỗi dòng là 1 bước, có đích, khoảng cách tới đích, ngưỡng để sang phase.

Tùy chọn --headless: không mở cửa sổ, tự chạy P liên tục tới hết ván, in log (dùng để thử nhanh).
"""

import argparse
import contextlib
import importlib
import io
import os
import sys
import time

import mujoco
import numpy as np

from . import config
from .env import SO101PickPlaceEnv

PHASE_DESC = {
    0: "nhấc thẳng lên từ home",
    1: "bay ngang tới phía TRÊN HỘP (độ cao an toàn)",
    2: "hạ xuống pre-grasp (gần hộp)",
    3: "hạ xuống vị trí KẸP",
    4: "đóng kẹp, chờ 30 bước",
    5: "nhấc hộp lên",
    6: "bay ngang tới phía TRÊN BÁT",
    7: "hạ hộp xuống bát",
    8: "mở kẹp thả hộp, chờ 20 bước",
    9: "rút tay lên khỏi bát",
    10: "về lại home",
    11: "XONG (giữ nguyên)",
}


def fmt(v):
    """Vector 3D thành chuỗi gọn."""
    return "(" + ", ".join(f"{x:+.3f}" for x in v) + ")"


def project(env, cam_id, width, height, point):
    """Đổi điểm 3D (world) thành pixel (u, v) trên ảnh của camera. Trả None nếu ở sau camera."""
    pos = env.data.cam_xpos[cam_id]
    rot = env.data.cam_xmat[cam_id].reshape(3, 3)
    p = rot.T @ (np.asarray(point) - pos)          # đổi sang hệ tọa độ của camera
    if p[2] >= -1e-6:                               # camera nhìn theo trục -z
        return None
    f = (height / 2) / np.tan(np.radians(env.model.cam_fovy[cam_id]) / 2)
    return int(width / 2 + f * p[0] / -p[2]), int(height / 2 - f * p[1] / -p[2])


class Runner:
    """Giữ env + policy, chạy từng bước và ghi lại thông tin để in log / vẽ."""

    def __init__(self, policy_path, seed, real_grasp=False):
        module_name, class_name = policy_path.split(":")
        policy_cls = getattr(importlib.import_module(module_name), class_name)
        self.env = SO101PickPlaceEnv(render_images=False, real_grasp=real_grasp, end_on_success=False)
        self.policy = policy_cls(self.env)
        self.seed = seed
        self.target = None          # đích của phase hiện tại (None nếu phase chờ)
        self.threshold = None
        self._patch_move_toward()
        self.reset()

    def _patch_move_toward(self):
        """Bọc _move_toward để biết đích và ngưỡng của policy mà KHÔNG phải sửa file policy."""
        original = self.policy._move_toward

        def wrapped(target_pos, ee_pos, threshold=0.02):
            self.target = np.array(target_pos, dtype=float)
            self.threshold = threshold
            return original(target_pos, ee_pos, threshold)

        self.policy._move_toward = wrapped

    def reset(self, quiet=False):
        self.obs, self.info = self.env.reset(seed=self.seed)
        self.policy.reset()
        self.step_no = 0
        self.finished = False
        self.last_log = ""
        self.target = None
        self.threshold = None
        self.phase_run = 0          # phase của bước VỪA chạy (khớp với dòng log)
        self.phase_next = 0         # phase sẽ chạy ở bước kế (khác phase_run khi vừa đổi phase)
        self.phase_step = 0         # đây là bước thứ mấy trong phase_run
        if not quiet:
            print(f"\n===== Ván mới: seed={self.seed} | hộp {fmt(self.info['box_pos'])} | bát {fmt(self.info['bowl_pos'])} =====")
            print(f">>> phase 0: {PHASE_DESC[0]}")

    def go_to_step(self, n):
        """Về đúng bước thứ n (n = 0 là trước khi chạy bước nào). Dùng cho nút LÙI.

        Mô phỏng không "chạy ngược" được, nên cách làm là reset cùng seed rồi chạy lại n bước
        (kết quả giống hệt lần trước vì mọi thứ đều xác định). n-1 bước đầu chạy lặng lẽ,
        bước cuối in log như bình thường để bạn thấy dòng của bước đó.
        """
        n = max(0, n)
        self.reset(quiet=True)
        with contextlib.redirect_stdout(io.StringIO()):
            for _ in range(max(0, n - 1)):
                if not self.step_once():
                    break
        if n == 0:
            print("[lùi] về bước 0 (chưa chạy bước nào)")
        else:
            print(f"[lùi] về bước {n}:")
            self.step_once()

    def back(self, n=1):
        """Lùi n bước so với bước hiện tại."""
        self.go_to_step(self.step_no - n)

    def finger_open(self):
        """Độ mở 1 ngón kẹp (mét). Cần < _gripper_close_threshold thì mới tính là đã khép."""
        adr = self.env.model.jnt_qposadr[self.env._finger_left_jnt]
        return float(self.env.data.qpos[adr])

    def contact_text(self):
        """Các va chạm đang có (MuJoCo thật), ví dụ 'forearm_link↔box (0.3mm)'. Bỏ qua hộp/bát nằm yên trên bàn."""
        m, d = self.env.model, self.env.data
        name = lambda g: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[g]))
        passive = {"table", "world", "box", "bowl"}
        out = []
        for i in range(d.ncon):
            c = d.contact[i]
            a, b = name(c.geom1), name(c.geom2)
            if a in passive and b in passive and (a in ("table", "world") or b in ("table", "world")):
                continue
            out.append(f"{a}↔{b} (xuyên {-c.dist * 1000:.1f}mm)")
        return ", ".join(out) if out else "không"

    def step_once(self):
        """Chạy đúng 1 bước. Trả về True nếu còn chạy tiếp được."""
        if self.finished:
            return False
        phase_before = self.policy._phase
        self.target = None
        # policy có thể có print debug, chặn lại để log không bị lẫn
        with contextlib.redirect_stdout(io.StringIO()):
            action = self.policy.select_action(
                {"state": self.obs["state"], "images": {}, "task": config.DEFAULT_TASK}
            )
        phase_after = self.policy._phase
        target, threshold = self.target, self.threshold
        self.obs, reward, terminated, truncated, self.info = self.env.step(action)
        self.step_no += 1

        info = self.info
        self.phase_run, self.phase_next = phase_before, phase_after
        # policy đặt lại bộ đếm về 0 khi đổi phase, nên tự đếm số bước của phase vừa chạy
        self.phase_step = self.policy._phase_step if phase_after == phase_before else self.phase_step + 1
        ee, box = info["ee_pos"], info["box_pos"]
        if target is not None:
            dist = float(np.linalg.norm(ee - target))
            goal = f"đích {fmt(target)} | cách đích {dist * 100:5.1f} cm (ngưỡng {threshold * 100:.1f})"
            # Kết quả IK: sai số = tay DỰ ĐOÁN (trên bản sao, không có va chạm) so với đích.
            # Nhỏ mà "cách đích" vẫn lớn nghĩa là IK tin tới được nhưng tay thật bị cản.
            ik_err = float(np.linalg.norm(self.policy._ik_data.site_xpos[self.policy._ee_site] - target))
            ik_deg = np.degrees(self.policy._current_joint_targets).round(0).astype(int).tolist()
            goal += f" | IK sai số {ik_err * 100:.2f} cm, góc IK {ik_deg}"
        else:
            goal = f"chờ {self.policy._phase_step if phase_after == phase_before else 'xong'}"
        line = (f"bước {self.step_no:3d} | phase {phase_before:2d} (bước {self.policy._phase_step:3d}) | "
                f"tay {fmt(ee)} | {goal} | hộp {fmt(box)} | "
                f"ngón mở {self.finger_open():.3f} | dính: {self.env._grasped} | va chạm: {self.contact_text()}")
        print(line)
        self.last_log = line

        if phase_after != phase_before:
            if target is not None:
                reason = "tới đích" if dist < threshold else "TIMEOUT 200 bước (không tới được đích)"
            else:
                reason = "hết thời gian chờ"
            print(f">>> phase {phase_before} xong ({reason}). Sang phase {phase_after}: "
                  f"{PHASE_DESC.get(phase_after, '?')}")

        if terminated or truncated or phase_after > 10:
            self.finished = True
            print(f"===== KẾT THÚC sau {self.step_no} bước: "
                  f"{'THÀNH CÔNG' if info['success'] else 'THẤT BẠI'} | hộp cách bát {info['box_to_bowl_dist']:.3f} m =====")
        return not self.finished

    def run_until_phase_change(self, on_frame=None):
        """Chạy tới khi đổi phase hoặc hết ván. on_frame() được gọi sau mỗi bước (để vẽ + chờ)."""
        start = self.policy._phase
        while self.step_once():
            if on_frame is not None and on_frame():   # on_frame trả True = người dùng bấm phím, dừng lại
                break
            if self.policy._phase != start:
                break


def run_viewer(runner, goto_n):
    """Cửa sổ MuJoCo viewer chính hãng: xoay/zoom bằng chuột, thấy ĐIỂM và LỰC va chạm thật.

    Điều khiển bằng phím (bấm khi cửa sổ viewer đang được chọn):
        ENTER : chạy 1 bước      G : chạy tới khi đổi phase      K : bật/tắt tự chạy
        U hoặc BACKSPACE : LÙI 1 bước (chạy lại từ đầu tới bước trước)
        J     : chạy lại ván     + / - : chạy nhanh / chậm hơn khi tự chạy
    Chuột: kéo trái = xoay, kéo phải = dời, lăn = zoom. Điểm va chạm (chấm cam) và mũi tên lực
    đã được bật sẵn. Trong bảng Rendering bên trái có thể bật thêm nhóm geom (Group) để xem khối va chạm.
    Chấm xanh lá = ĐÍCH của phase, chấm tím = điểm giữa hai ngón (ee_site).
    """
    import queue
    import mujoco.viewer

    cmds = queue.Queue()
    KEYS = {257: "step", 335: "step", 32: "step", ord("G"): "phase", ord("K"): "auto", ord("J"): "reset", 259: "back", ord("U"): "back", ord("+"): "fast", 61: "fast", ord("-"): "slow", 45: "slow"}

    def on_key(keycode):
        if os.environ.get("STEP_DEBUG_KEYS"):
            print(f"[phím] mã {keycode}", flush=True)
        if keycode in KEYS:
            cmds.put(KEYS[keycode])

    model, data = runner.env.model, runner.env.data
    model.vis.scale.contactwidth = 0.012      # to hơn cho dễ thấy
    model.vis.scale.contactheight = 0.012
    model.vis.scale.forcewidth = 0.02
    model.vis.map.force = 0.02                # chiều dài mũi tên lực
    auto, delay = False, 0.12

    def markers(v):
        """Vẽ thêm chấm đích (xanh lá) và đầu tay (tím) bằng geom trang trí của viewer."""
        scn = v.user_scn
        scn.ngeom = 0
        pts = []
        if runner.target is not None:
            pts.append((runner.target, [0.1, 0.85, 0.2, 0.9], 0.009))
        pts.append((runner.env.get_ee_pos(), [0.85, 0.2, 0.85, 0.9], 0.006))
        for pos, rgba, r in pts:
            mujoco.mjv_initGeom(scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE, np.array([r, 0, 0]),
                                np.asarray(pos, dtype=float), np.eye(3).flatten(), np.array(rgba, dtype=np.float32))
            scn.ngeom += 1

    with mujoco.viewer.launch_passive(model, data, key_callback=on_key, show_left_ui=True, show_right_ui=False) as v:
        v.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
        v.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True
        v.cam.lookat[:] = [0.0, -0.02, 0.47]
        v.cam.distance, v.cam.azimuth, v.cam.elevation = 0.9, 140, -25
        if goto_n:
            with v.lock():
                goto_n_steps = goto_n
                with contextlib.redirect_stdout(io.StringIO()):
                    for _ in range(goto_n_steps):
                        if not runner.step_once():
                            break
            print(f"[goto] đã chạy sẵn {runner.step_no} bước.")
        with v.lock():
            markers(v)
        v.sync()
        print("Cửa sổ viewer đứng yên cho tới khi bạn bấm phím (nó chỉ chạy khi có lệnh). Hãy bấm CHUỘT vào cửa sổ viewer trước.")
        print("Viewer: ENTER hoặc SPACE = 1 bước | U hoặc BACKSPACE = lùi 1 bước | G = tới phase kế | K = tự chạy | J = chạy lại | +/- = tốc độ. Log và 'va chạm' in ở đây.")
        while v.is_running():
            try:
                cmd = cmds.get(timeout=delay if auto else 0.03)
            except queue.Empty:
                cmd = "step" if auto else None
                v.sync()                          # luôn sync đều đặn để viewer nhận phím và vẽ lại
            if cmd == "fast":
                delay = max(0.01, delay * 0.6); continue
            if cmd == "slow":
                delay = min(1.5, delay * 1.6); continue
            if cmd == "auto":
                auto = not auto
                continue
            if cmd is None:
                continue
            with v.lock():
                if cmd == "step":
                    if not runner.step_once():
                        auto = False
                elif cmd == "phase":
                    runner.run_until_phase_change()
                elif cmd == "back":
                    runner.back()
                    auto = False
                elif cmd == "reset":
                    runner.reset()
                    auto = False
                    if goto_n:
                        with contextlib.redirect_stdout(io.StringIO()):
                            for _ in range(goto_n):
                                if not runner.step_once():
                                    break
                markers(v)
            v.sync()


def main():
    parser = argparse.ArgumentParser(description="Chạy policy từng bước bằng phím bấm")
    parser.add_argument("--viewer", action="store_true", help="dùng cửa sổ MuJoCo viewer (xoay được, thấy va chạm thật)")
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--real-grasp", action="store_true", help="tắt kẹp giả, hộp chỉ được giữ bằng ma sát thật")
    parser.add_argument("--policy", default="sim_engine.policies.scripted_policy:ScriptedPolicy")
    parser.add_argument("--headless", action="store_true", help="không mở cửa sổ, chạy hết ván và in log")
    parser.add_argument("--goto", type=int, default=0, help="chạy sẵn N bước (không in log) rồi mới dừng chờ phím; "
                        "dùng số 'tổng bước' trên replay.html để demo dừng đúng chỗ đó")
    args = parser.parse_args()

    runner = Runner(args.policy, args.seed, real_grasp=args.real_grasp)

    def goto(n):
        """Chạy lặng lẽ n bước đầu để demo dừng đúng bước đang xem trên replay.html."""
        with contextlib.redirect_stdout(io.StringIO()):
            for _ in range(n):
                if not runner.step_once():
                    break
        print(f"[goto] đã chạy sẵn {runner.step_no} bước; bấm SPACE để chạy tiếp từng bước.")

    if args.viewer:
        run_viewer(runner, args.goto)
        runner.env.close()
        return

    if args.goto:
        goto(args.goto)

    if args.headless:
        while not runner.finished:
            runner.run_until_phase_change()
        runner.env.close()
        return

    try:
        import cv2
    except ImportError:
        print("Cần opencv: pip install opencv-python (hoặc dùng --headless)")
        sys.exit(1)

    from .renderer import OffscreenRenderer

    W, H = 960, 720
    main_r = OffscreenRenderer(runner.env.model, W, H)
    wrist_r = OffscreenRenderer(runner.env.model, 320, 240)
    cam_id = int(runner.env.model.camera("webcam").id)
    delay_ms = 120            # thời gian nghỉ giữa 2 bước khi chạy P hoặc A
    auto = False
    window = "step_policy (SPACE=1 buoc, P=toi phase ke, A=tu chay, Q=thoat)"

    def draw():
        """Vẽ khung hình hiện tại kèm chú thích."""
        env, pol = runner.env, runner.policy
        frame = cv2.cvtColor(main_r.render(env.data, "webcam"), cv2.COLOR_RGB2BGR)
        ee, box = env.get_ee_pos(), env.get_box_pos()

        bowl = env.get_bowl_pos()
        p_bowl = project(env, cam_id, W, H, bowl)
        p_ee = project(env, cam_id, W, H, ee)
        p_box = project(env, cam_id, W, H, box)
        p_tg = project(env, cam_id, W, H, runner.target) if runner.target is not None else None
        if p_ee and p_tg:
            cv2.line(frame, p_ee, p_tg, (0, 220, 255), 2)             # vàng: tay -> đích
        if p_bowl:
            # vòng xanh dương: hộp phải nằm trong vòng này (theo chiều ngang) mới tính là vào bát
            edge = project(env, cam_id, W, H, bowl + np.array([config.BOWL_INNER_RADIUS, 0, 0]))
            if edge:
                cv2.circle(frame, p_bowl, max(4, abs(edge[0] - p_bowl[0])), (217, 102, 51), 2)
            cv2.circle(frame, p_bowl, 4, (217, 102, 51), -1)
        if p_box:
            cv2.circle(frame, p_box, 6, (0, 0, 255), -1)                # đỏ: tâm hộp
        if p_tg:
            cv2.circle(frame, p_tg, 10, (0, 220, 0), 2)                 # xanh lá: đích
            cv2.circle(frame, p_tg, 2, (0, 220, 0), -1)
        if p_ee:
            cv2.drawMarker(frame, p_ee, (255, 0, 255), cv2.MARKER_CROSS, 18, 2)   # tím: điểm giữa hai ngón kẹp

        wrist = cv2.cvtColor(wrist_r.render(env.data, "arm_cam"), cv2.COLOR_RGB2BGR)
        wrist = cv2.resize(wrist, (240, 180))
        frame[H - 190:H - 10, W - 250:W - 10] = wrist

        info = runner.info
        bowl_p = info["bowl_pos"]
        horiz = info["box_to_bowl_dist"]
        z_ok = bowl_p[2] < box[2] < bowl_p[2] + config.BOWL_HEIGHT - config.BOX_HALF   # đỉnh hộp thấp hơn miệng bát
        # Dùng phase của bước VỪA chạy để khớp với dòng log ở terminal
        phase_txt = f"phase {runner.phase_run} (buoc {runner.phase_step} trong phase)"
        if runner.phase_next != runner.phase_run:
            phase_txt += f"  -> vua xong, sang phase {runner.phase_next}"
        RED, MAG, GRN, WHT, BLU = (80, 80, 255), (255, 0, 255), (0, 220, 0), (255, 255, 255), (255, 150, 90)
        lines = [
            (f"seed {runner.seed} | tong {runner.step_no} buoc | {phase_txt}", WHT),
            (f"tay (giua 2 ngon, tim) {fmt(ee)}", MAG),
            (f"hop (do)  {fmt(box)}", RED),
            (f"bat (xanh) {fmt(bowl_p)}", BLU),
        ]
        if runner.target is not None:
            d = float(np.linalg.norm(ee - runner.target)) * 100
            lines.append((f"dich (xanh la) {fmt(runner.target)} | tay cach dich {d:.1f} cm, nguong {runner.threshold * 100:.1f}", GRN))
        lines.append((f"ngon mo {runner.finger_open():.3f} (can <{env._gripper_close_threshold}) | hop dinh vao tay: {env._grasped}", WHT))
        lines.append((f"vao bat? ngang {horiz * 100:.1f} cm (can <{config.BOWL_INNER_RADIUS * 100:.1f}) | do cao hop: {'OK' if z_ok else 'chua'}", BLU))
        if runner.finished:
            ok = runner.info["success"]
            lines.append(("KET THUC: " + ("THANH CONG" if ok else "THAT BAI") + "  (R = chay lai)",
                          (0, 255, 0) if ok else (0, 140, 255)))
        for i, (text, color) in enumerate(lines):
            cv2.putText(frame, text, (10, 28 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 2 if i == 0 else 1)
        cv2.putText(frame, "SPACE=1 buoc | U=lui 1 buoc | P=toi phase ke | A=tu chay | +/-=toc do | R N B | Q",
                    (10, H - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.imshow(window, frame)

    pending = [255]           # phím bấm giữa lúc đang chạy P / A (255 = không có phím)

    def frame_and_wait():
        """Vẽ rồi nghỉ delay_ms. Nếu có phím bấm: nhớ lại và trả True để dừng đợt chạy hiện tại."""
        draw()
        key = cv2.waitKey(delay_ms) & 0xFF
        if key != 255:
            pending[0] = key
            return True
        return False

    print("Bấm vào cửa sổ ảnh rồi dùng phím. SPACE = 1 bước, P = tới phase kế, A = tự chạy, Q = thoát.")
    draw()
    while True:
        if pending[0] != 255:
            key, pending[0] = pending[0], 255
        elif auto:
            frame_and_wait()
            key, pending[0] = pending[0], 255
        else:
            key = cv2.waitKey(30) & 0xFF
        if key in (ord("q"), 27):
            break
        elif key in (ord(" "), 13):
            runner.step_once()
        elif key in (ord("u"), 8):
            auto = False
            runner.back()
        elif key == ord("p"):
            runner.run_until_phase_change(on_frame=frame_and_wait)
        elif key == ord("a"):
            auto = not auto
        elif key in (ord("+"), ord("=")):
            delay_ms = max(10, int(delay_ms * 0.6))
            print(f"[tốc độ] nghỉ {delay_ms} ms giữa 2 bước")
        elif key in (ord("-"), ord("_")):
            delay_ms = min(1500, int(delay_ms * 1.6))
            print(f"[tốc độ] nghỉ {delay_ms} ms giữa 2 bước")
        elif key == ord("r"):
            runner.reset()
        elif key in (ord("n"), ord("b")):
            runner.seed += 1 if key == ord("n") else -1
            runner.reset()
            if args.goto:
                goto(args.goto)
        if auto and not runner.step_once():
            auto = False
        if key != 255 or auto:
            draw()

    cv2.destroyAllWindows()
    main_r.close()
    wrist_r.close()
    runner.env.close()
    print("Đã thoát.")


if __name__ == "__main__":
    main()
