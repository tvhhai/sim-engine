"""Bước 1 của học bắt chước: ghi dataset LeRobot từ policy viết tay ("thầy giáo"), CHỈ giữ ván thành công.

Cách dùng (môi trường có LeRobot, vd D:/vla/venv-lerobot):
    python -m sim_engine.learning.record_dataset --episodes 200 --kind box --size 0.04 \
        --root D:/vla/datasets/so101_box4_v1 --repo-id local/so101_box4_v1

Ý tưởng: AI (ACT) không tự nghĩ ra cách gắp. Nó học "nhìn thấy cảnh này thì làm động tác này" từ các ví dụ. Ví dụ lấy từ policy viết
tay (policies/scripted_hybrid.py): cho thầy giáo chạy từng seed, mỗi bước ghi lại (ảnh 2 camera, góc khớp, lệnh thầy ra). Ván thầy làm
hỏng bị bỏ, vì AI sẽ bắt chước cả cái sai.

Seed bắt đầu từ sim_setup.TRAIN_SEED_START (3000), không trùng bộ seed đánh giá. Ván hỏng bị bỏ và chạy seed kế tiếp cho tới khi đủ
--episodes ván thành công. In tiến độ và ghi thêm <root>/record_log.json (seed nào được giữ, seed nào hỏng).

Kết quả trong --root:
    videos/observation.images.{front,wrist}/...mp4   ảnh hai camera, nén video
    data/...parquet                                    bảng số: state, action, ván số mấy, khung số mấy
    meta/                                              mô tả dataset: fps, các trường, thống kê (trung bình, độ lệch) để chuẩn hoá
    record_log.json                                    seed giữ / bỏ (file riêng của repo này, LeRobot không dùng)

Muốn dữ liệu khác thì sửa ở đâu:
    - thêm ván / đổi vật / cỡ vật      -> tham số dòng lệnh (--episodes, --kind, --size)
    - thầy giáo gắp sai hay vụng       -> sửa policies/ (AI giỏi nhất cũng chỉ ngang thầy)
    - muốn ghi thêm ván ở vùng xa      -> đổi cách chọn seed trong main() (hiện chạy seed liên tiếp)
    - camera / toạ độ vật đưa vào        -> --cameras top,wrist   --env-state
    - thêm nhiễu để AI học cách sửa sai -> --noise 0.02 (đẩy tay lệch, ghi lệnh ĐÚNG của thầy ở chỗ lệch; xem run_episode)
"""

import argparse
import contextlib
import io
import json
import time
from pathlib import Path

import numpy as np

from .. import config
from ..policies import default_policy
from . import sim_setup

# Nhiễu khi ghi (--noise): action là thang [-1, 1] trên cả biên độ khớp (~3,8 rad), nên 0,02 ~ 2 độ ~ 1-1,5 cm ở đầu kẹp.
NOISE_START_PROB = 0.03                         # mỗi bước có 3 % khả năng bắt đầu một lần đẩy lệch (~1 lần mỗi 33 bước)
NOISE_STEPS = 10                                # mỗi lần đẩy lệch kéo dài 10 bước (~0,3 s) để tay lệch thật, rồi thầy kéo về


def run_episode(seed, kind, size, color, task, cameras=sim_setup.DEFAULT_CAMERAS, env_state=False, noise=0.0, rng=None):
    """Chạy thầy giáo một ván có ảnh. Trả (thành công?, danh sách khung hình).

    Khung hình chưa được ghi vào dataset ngay: phải chờ hết ván mới biết thành công hay không.
    noise > 0: thỉnh thoảng làm tay lệch khỏi đường thầy chọn (xem NOISE_*), nhưng khung hình vẫn ghi lệnh ĐÚNG của thầy ở
    trạng thái đã lệch đó -> AI được thấy "lệch rồi thì sửa thế này" (ý tưởng của DART / DAgger).
    """
    env, options = sim_setup.make_env(seed, kind, size, color, images=bool(cameras))   # env có hộp ở vị trí của seed
    policy = default_policy(env)                                  # thầy giáo: policy viết tay mặc định (config.DEFAULT_POLICY, bản min-jerk)
    obs, _ = env.reset(seed=seed, options=options)                # đặt tay về home, đặt hộp và bát; obs = quan sát đầu tiên
    policy.reset()                                                # xoá trạng thái cũ của thầy (phase về 0)
    frames, info, push = [], {}, None
    for t in range(config.MAX_EPISODE_STEPS):                     # tối đa 500 bước (~16 giây)
        with contextlib.redirect_stdout(io.StringIO()):           # thầy giáo in nhiều dòng debug; nuốt đi cho log gọn
            action = policy.select_action({"state": obs["state"], "images": {}, "task": task})
            # thầy KHÔNG nhìn ảnh ("images": {}): nó đọc vị trí hộp thẳng từ mô phỏng. Ảnh chỉ để ghi lại cho AI học.
        # ghi cặp (quan sát TRƯỚC khi làm, lệnh thầy ra) = 1 ví dụ
        frames.append(sim_setup.to_frame(env, obs, action, task, cameras, env_state))
        executed = action
        if noise > 0 and not env._grasped:                       # chỉ đẩy lệch lúc chưa kẹp: lệch lúc đang cầm dễ làm rơi vật
            if push is None and rng.random() < NOISE_START_PROB:
                push = [rng.normal(0, noise, 5), NOISE_STEPS]    # độ lệch cho 5 khớp, giữ trong NOISE_STEPS bước
            if push is not None:
                executed = action.copy()
                executed[:5] = action[:5] + push[0]               # làm lệnh đã lệch; khung hình vẫn ghi lệnh đúng của thầy
                push[1] -= 1
                push = push if push[1] > 0 else None
        obs, _, _, truncated, info = env.step(executed)           # làm lệnh: mô phỏng chạy tiếp 1 bước, trả quan sát mới
        if truncated or policy._phase > 10:                       # hết giờ, hoặc thầy đã xong (phase 11 = đã về home)
            break
    _, info, _ = env.settle(action, info)                         # chờ hộp nằm yên trong bát rồi mới chấm thành công
    env.close()
    return bool(info.get("success")), frames


def main():
    # --- Đọc tham số dòng lệnh ---
    parser = argparse.ArgumentParser(description="Ghi LeRobotDataset từ policy viết tay (chỉ ván thành công)")
    parser.add_argument("--episodes", type=int, default=200, help="số ván THÀNH CÔNG cần ghi")
    parser.add_argument("--kind", choices=["box", "sphere"], default="box")
    parser.add_argument("--size", type=float, default=0.04, help="cạnh hộp / đường kính bóng (m)")
    parser.add_argument("--color", default="#e5484d")
    parser.add_argument("--color-name", default="red", help="tên màu trong câu nhiệm vụ")
    parser.add_argument("--seed-start", type=int, default=sim_setup.TRAIN_SEED_START)
    parser.add_argument("--seed-step", type=int, default=1,
                        help="bước nhảy seed: chạy song song K phần thì phần k dùng --seed-start S+k --seed-step K, các phần không bao giờ trùng seed (dải liền nhau thì phần có ván hỏng lấn sang dải phần sau)")
    parser.add_argument("--root", required=True, help="thư mục dataset (chưa tồn tại)")
    parser.add_argument("--repo-id", default="local/so101_dataset")   # tên dataset kiểu Hugging Face; "local/..." = không đẩy lên mạng
    parser.add_argument("--cameras", default=",".join(sim_setup.DEFAULT_CAMERAS),
                        help="camera đưa vào dataset, cách nhau dấu phẩy: front, wrist, top; để trống = không ảnh")
    parser.add_argument("--env-state", action="store_true", help="thêm toạ độ thật của vật và bát (thí nghiệm, robot thật không có)")
    parser.add_argument("--noise", type=float, default=0.0, help="độ lệch đẩy tay khi ghi (vd 0.02); 0 = không đẩy")
    args = parser.parse_args()
    cameras = tuple(c for c in args.cameras.split(",") if c)
    rng = np.random.default_rng(0)                                # nhiễu cố định theo seed 0: chạy lại ra cùng dataset

    # Import LeRobot ở trong hàm (không ở đầu file): môi trường Studio không cài LeRobot vẫn import được module này.
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    task = sim_setup.TASKS[args.kind].format(color=args.color_name)   # vd "Pick up the red box and place it in the bowl"
    # Tạo dataset rỗng. robot_type chỉ là nhãn mô tả. use_videos=True: ảnh lưu thành mp4.
    # streaming_encoding=True: nén video song song trong lúc ván kế đang chạy (nhanh hơn ~5 lần so với nén sau mỗi ván).
    ds = LeRobotDataset.create(args.repo_id, sim_setup.FPS, sim_setup.features(cameras, args.env_state), root=args.root,
                               robot_type="so101_follower", use_videos=bool(cameras), streaming_encoding=bool(cameras))
    log, seed, t0 = {"kept": [], "failed": []}, args.seed_start, time.time()
    try:
        while len(log["kept"]) < args.episodes:                  # chạy tới khi đủ số ván THÀNH CÔNG
            ok, frames = run_episode(seed, args.kind, args.size, args.color, task, cameras, args.env_state, args.noise, rng)
            if ok:
                for frame in frames:
                    ds.add_frame(frame)                           # đưa từng khung vào bộ đệm của ván hiện tại
                ds.save_episode()                                 # chốt ván: ghi parquet, đẩy ảnh đi nén video
                log["kept"].append({"seed": seed, "frames": len(frames)})
            else:
                log["failed"].append(seed)                        # ván hỏng: bỏ hết khung, chỉ ghi lại seed để biết
            done = len(log["kept"])
            print(f"seed {seed}: {'giữ' if ok else 'HỎNG, bỏ'} ({len(frames)} khung) | đã có {done}/{args.episodes} ván, "
                  f"bỏ {len(log['failed'])}, {time.time() - t0:.0f} s", flush=True)
            seed += args.seed_step
    finally:
        # finally: chạy cả khi bấm Ctrl+C giữa chừng -> các ván đã lưu vẫn dùng được.
        ds.finalize()                                             # đóng file video / parquet, ghi thống kê vào meta/
        log.update(kind=args.kind, size=args.size, task=task, cameras=list(cameras), env_state=args.env_state, noise=args.noise,
                   seed_start=args.seed_start, seed_step=args.seed_step,
                   seconds=round(time.time() - t0))
        (Path(args.root) / "record_log.json").write_text(json.dumps(log, indent=1), encoding="utf-8")
    print(f"Xong: {len(log['kept'])} ván, {sum(k['frames'] for k in log['kept'])} khung hình -> {args.root}")


if __name__ == "__main__":
    main()
