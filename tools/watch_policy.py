"""Xem policy chạy trong cửa sổ và TỰ NẠP LẠI khi bạn lưu file code (hot reload).

Cách dùng (đứng ở thư mục chứa sim_engine, đã kích hoạt .venv):
    python -m sim_engine.watch_policy                 # xem seed 2 (seed hay thất bại)
    python -m sim_engine.watch_policy --seed 3        # chọn seed
    python -m sim_engine.watch_policy --policy sim_engine.policies.my_policy:MyPolicy

Cách hoạt động: mỗi khung hình, script kiểm tra giờ sửa (mtime) của các file bên dưới.
Nếu có file đổi, nó nạp lại module (importlib.reload), tạo lại env và policy mới,
rồi chạy lại episode từ đầu với CÙNG seed. Cửa sổ không bị đóng.

File được theo dõi: config.py, renderer.py, env.py, policies/base.py, scripted_policy.py, scripted_v2.py, file policy, so101_scene.xml.

Phím: SPACE = tạm dừng / chạy tiếp | R = chạy lại episode | N / B = seed kế / seed trước | Q hoặc ESC = thoát
Giới hạn: chỉ chạy lại episode, không giữ trạng thái đang dở. Nếu file có lỗi cú pháp,
cửa sổ hiện thông báo lỗi và chờ bạn sửa rồi lưu lại.

Tùy chọn --headless: không mở cửa sổ, chỉ in log ra terminal (dùng để thử nhanh, không cần màn hình).
"""

import argparse
import importlib
import os
import sys
import time
import traceback

import numpy as np

from .. import config


def watched_files(policy_module):
    """Danh sách file cần theo dõi thay đổi."""
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    files = [
        os.path.join(here, "config.py"),
        os.path.join(here, "renderer.py"),
        os.path.join(here, "env.py"),
        os.path.join(here, "policies", "base.py"),
        config.SCENE_XML,
    ]
    # bản kết hợp V1+V2 nằm ở file riêng: theo dõi cả hai file policy con
    files += [os.path.join(here, "policies", name) for name in ("scripted_policy.py", "scripted_v2.py")]
    if policy_module is not None and getattr(policy_module, "__file__", None):
        files.append(policy_module.__file__)
    return files


def snapshot(files):
    """Ghi lại giờ sửa của từng file, để so sánh lần sau xem file nào đổi."""
    return {f: os.path.getmtime(f) for f in files if os.path.exists(f)}


def reload_all(policy_path):
    """Nạp lại toàn bộ module theo đúng thứ tự phụ thuộc, rồi tạo env + policy mới.

    Thứ tự quan trọng: module nào được import bởi module khác thì phải nạp lại TRƯỚC.
        config -> renderer -> env -> policies.base -> module policy
    Trả về (env, policy, policy_module).
    """
    from .. import renderer, env as env_mod
    from ..policies import base

    importlib.reload(config)
    importlib.reload(renderer)
    importlib.reload(env_mod)
    importlib.reload(base)
    from ..policies import scripted_policy, scripted_v2          # bản kết hợp dùng hai module này: nạp lại trước
    importlib.reload(scripted_policy)
    importlib.reload(scripted_v2)

    module_name, class_name = policy_path.split(":")
    policy_module = importlib.import_module(module_name)
    policy_module = importlib.reload(policy_module)
    policy_cls = getattr(policy_module, class_name)

    env = env_mod.SO101PickPlaceEnv(render_images=False)
    policy = policy_cls(env)
    return env, policy, policy_module


def main():
    parser = argparse.ArgumentParser(description="Xem policy chạy, tự nạp lại khi lưu file")
    parser.add_argument("--seed", type=int, default=2, help="seed vị trí hộp (mặc định 2, một seed hay thất bại)")
    parser.add_argument("--policy", default=config.DEFAULT_POLICY)
    parser.add_argument("--headless", action="store_true", help="không mở cửa sổ, chỉ in log")
    parser.add_argument("--max-frames", type=int, default=0, help="tự thoát sau N khung hình (0 = không giới hạn)")
    args = parser.parse_args()

    cv2 = None
    if not args.headless:
        try:
            import cv2 as _cv2
            cv2 = _cv2
        except ImportError:
            print("Cần opencv: pip install opencv-python (hoặc dùng --headless)")
            sys.exit(1)

    from ..renderer import OffscreenRenderer

    seed = args.seed
    env, policy, policy_module = reload_all(args.policy)
    files = watched_files(policy_module)
    mtimes = snapshot(files)

    def new_episode():
        """Reset env và policy theo seed hiện tại."""
        obs, info = env.reset(seed=seed)
        policy.reset()
        print(f"[episode] seed={seed}")
        return obs, info

    def make_renderers():
        main_r = OffscreenRenderer(env.model, 960, 720)
        wrist_r = OffscreenRenderer(env.model, 320, 240)
        return main_r, wrist_r

    main_r = wrist_r = None
    if cv2 is not None:
        main_r, wrist_r = make_renderers()

    obs, info = new_episode()
    step = 0
    paused = False
    error_msg = None
    finished = False
    finish_time = 0.0
    last_phase = None
    frame_count = 0
    print("Đang chạy. Sửa và LƯU file policy để thấy kết quả mới. (Q để thoát)")

    while True:
        frame_count += 1
        if args.max_frames and frame_count > args.max_frames:
            break

        # 1) Kiểm tra file nào vừa được lưu -> nạp lại
        current = snapshot(files)
        changed = [os.path.basename(f) for f in current if current[f] != mtimes.get(f)]
        if changed:
            print(f"[reload] phát hiện thay đổi: {', '.join(changed)}")
            try:
                if main_r is not None:
                    main_r.close()
                    wrist_r.close()
                env.close()
                env, policy, policy_module = reload_all(args.policy)
                files = watched_files(policy_module)
                if cv2 is not None:
                    main_r, wrist_r = make_renderers()
                obs, info = new_episode()
                step, finished, last_phase, error_msg = 0, False, None, None
            except Exception:
                # Lỗi (ví dụ gõ sai cú pháp): báo lỗi và chờ bạn sửa rồi lưu lại
                error_msg = traceback.format_exc().strip().splitlines()[-1]
                print("[reload] LỖI, sửa file rồi lưu lại:")
                traceback.print_exc()
            mtimes = snapshot(files)

        # 2) Chạy 1 bước nếu không tạm dừng / không lỗi / chưa xong
        if not paused and error_msg is None and not finished:
            action = policy.select_action({"state": obs["state"], "images": {}, "task": config.DEFAULT_TASK})
            obs, reward, terminated, truncated, info = env.step(action)
            step += 1

            phase = getattr(policy, "_phase", None)
            if phase != last_phase:
                print(f"  bước {step:3d} -> phase {phase} | tay {info['ee_pos'].round(3)} | hộp {info['box_pos'].round(3)}")
                last_phase = phase

            if terminated or truncated or (phase is not None and phase > 10):
                finished = True
                finish_time = time.time()
                print(f"[xong] success={info['success']} sau {step} bước | hộp cách bát {info['box_to_bowl_dist']:.3f} m")

        # 3) Tự chạy lại episode sau 1.5 giây khi đã xong (để bạn thấy lặp lại liên tục)
        if finished and time.time() - finish_time > 1.5 and error_msg is None:
            obs, info = new_episode()
            step, finished, last_phase = 0, False, None

        # 4) Vẽ cửa sổ (nếu không phải headless)
        if cv2 is not None:
            frame = cv2.cvtColor(main_r.render(env.data, "webcam"), cv2.COLOR_RGB2BGR)
            wrist = cv2.cvtColor(wrist_r.render(env.data, "arm_cam"), cv2.COLOR_RGB2BGR)
            wrist = cv2.resize(wrist, (240, 180))
            h, w = frame.shape[:2]
            frame[h - 190:h - 10, w - 250:w - 10] = wrist

            if error_msg is not None:
                text, color = f"LOI: {error_msg[:70]}", (60, 60, 255)
            elif finished:
                text = ("SUCCESS" if info["success"] else "FAIL") + f" | seed {seed} | {step} buoc"
                color = (0, 255, 0) if info["success"] else (0, 140, 255)
            else:
                text = f"seed {seed} | buoc {step} | phase {getattr(policy, '_phase', '?')}" + (" | TAM DUNG" if paused else "")
                color = (255, 255, 255)
            cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            # Thông tin thêm để quan sát (cv2.putText không hiển thị được tiếng Việt có dấu nên dùng không dấu)
            if error_msg is None:
                ee, box = info["ee_pos"], info["box_pos"]
                lines = [
                    f"tay  x={ee[0]:+.3f} y={ee[1]:+.3f} z={ee[2]:+.3f}",
                    f"hop  x={box[0]:+.3f} y={box[1]:+.3f} z={box[2]:+.3f}",
                    f"tay-hop {info['ee_to_box_dist'] * 100:5.1f} cm | dang kep: {env._grasped} | kep mo: {obs['state'][6]:.0f}",
                ]
                for i, line in enumerate(lines):
                    cv2.putText(frame, line, (10, 58 + 24 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
            cv2.imshow("watch_policy (Q = thoat)", frame)
            key = cv2.waitKey(33) & 0xFF
        else:
            time.sleep(0.005)
            key = 255

        # 5) Xử lý phím
        if key in (ord("q"), 27):
            break
        if key == ord(" "):
            paused = not paused
        elif key == ord("r"):
            obs, info = new_episode()
            step, finished, last_phase = 0, False, None
        elif key in (ord("n"), ord("b")):
            seed += 1 if key == ord("n") else -1
            obs, info = new_episode()
            step, finished, last_phase = 0, False, None

    if cv2 is not None:
        cv2.destroyAllWindows()
        main_r.close()
        wrist_r.close()
    env.close()
    print("Đã thoát.")


if __name__ == "__main__":
    main()
