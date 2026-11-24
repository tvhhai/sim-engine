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
    - thêm nhiễu để AI học cách sửa sai -> run_episode() (vd thỉnh thoảng cộng nhiễu vào action trước env.step, nhưng vẫn ghi action
                                          ĐÚNG của thầy vào khung hình)
"""

import argparse
import contextlib
import io
import json
import time
from pathlib import Path

from .. import config
from ..policies import default_policy
from . import sim_setup


def run_episode(seed, kind, size, color, task):
    """Chạy thầy giáo một ván có ảnh. Trả (thành công?, danh sách khung hình).

    Khung hình chưa được ghi vào dataset ngay: phải chờ hết ván mới biết thành công hay không.
    """
    env, options = sim_setup.make_env(seed, kind, size, color)   # env có hộp ở vị trí của seed, bật render ảnh
    policy = default_policy(env)                                  # thầy giáo: policy viết tay mặc định (scripted_hybrid)
    obs, _ = env.reset(seed=seed, options=options)                # đặt tay về home, đặt hộp và bát; obs = quan sát đầu tiên
    policy.reset()                                                # xoá trạng thái cũ của thầy (phase về 0)
    frames, info = [], {}
    for _ in range(config.MAX_EPISODE_STEPS):                     # tối đa 500 bước (~16 giây)
        with contextlib.redirect_stdout(io.StringIO()):           # thầy giáo in nhiều dòng debug; nuốt đi cho log gọn
            action = policy.select_action({"state": obs["state"], "images": {}, "task": task})
            # thầy KHÔNG nhìn ảnh ("images": {}): nó đọc vị trí hộp thẳng từ mô phỏng. Ảnh chỉ để ghi lại cho AI học.
        frames.append(sim_setup.to_frame(obs, action, task))     # ghi cặp (quan sát TRƯỚC khi làm, lệnh thầy ra) = 1 ví dụ
        obs, _, _, truncated, info = env.step(action)             # làm lệnh đó: mô phỏng chạy tiếp 1 bước, trả quan sát mới
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
    parser.add_argument("--root", required=True, help="thư mục dataset (chưa tồn tại)")
    parser.add_argument("--repo-id", default="local/so101_dataset")   # tên dataset kiểu Hugging Face; "local/..." = không đẩy lên mạng
    args = parser.parse_args()

    # Import LeRobot ở trong hàm (không ở đầu file): môi trường Studio không cài LeRobot vẫn import được module này.
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    task = sim_setup.TASKS[args.kind].format(color=args.color_name)   # vd "Pick up the red box and place it in the bowl"
    # Tạo dataset rỗng. robot_type chỉ là nhãn mô tả. use_videos=True: ảnh lưu thành mp4.
    # streaming_encoding=True: nén video song song trong lúc ván kế đang chạy (nhanh hơn ~5 lần so với nén sau mỗi ván).
    ds = LeRobotDataset.create(args.repo_id, sim_setup.FPS, sim_setup.features(), root=args.root, robot_type="so101_follower",
                               use_videos=True, streaming_encoding=True)
    log, seed, t0 = {"kept": [], "failed": []}, args.seed_start, time.time()
    try:
        while len(log["kept"]) < args.episodes:                  # chạy tới khi đủ số ván THÀNH CÔNG
            ok, frames = run_episode(seed, args.kind, args.size, args.color, task)
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
            seed += 1
    finally:
        # finally: chạy cả khi bấm Ctrl+C giữa chừng -> các ván đã lưu vẫn dùng được.
        ds.finalize()                                             # đóng file video / parquet, ghi thống kê vào meta/
        log.update(kind=args.kind, size=args.size, task=task, seconds=round(time.time() - t0))
        (Path(args.root) / "record_log.json").write_text(json.dumps(log, indent=1), encoding="utf-8")
    print(f"Xong: {len(log['kept'])} ván, {sum(k['frames'] for k in log['kept'])} khung hình -> {args.root}")


if __name__ == "__main__":
    main()
