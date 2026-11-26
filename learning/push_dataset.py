"""Bước phụ: đẩy một dataset LeRobot lên Hugging Face Hub (chia sẻ cho team, xem trực tuyến, gọi bằng repo_id ở bất kỳ máy nào).

    python -m sim_engine.learning.push_dataset --root D:/vla/datasets/so101_box4_mj --repo-id tvhhai/so101_box4_mj --private
    python -m sim_engine.learning.push_dataset --root ... --repo-id tvhhai/so101_box4_mj_50 --first 50 --private   # bản con 50 ván
    python -m sim_engine.learning.push_dataset --root ... --repo-id ... --dry-run    # chỉ in sẽ đẩy gì, không đẩy

Cần đăng nhập trước: `hf auth login` (máy nhà) hoặc biến môi trường HF_TOKEN (Colab: Secrets -> HF_TOKEN), token loại Write.
--private: repo chỉ mình (và người được mời) xem được; đổi sang public sau trên trang Settings của repo.
--first N: chỉ giữ N ván đầu (tạo bản sao ở <root>_first<N>, dataset gốc không đổi).

Đẩy xong: trang dataset https://huggingface.co/datasets/<repo-id>, xem từng ván (video + góc khớp) ở
https://huggingface.co/spaces/lerobot/visualize_dataset?path=<repo-id>. Train từ dataset trên Hub: --dataset.repo_id=<repo-id> (bỏ root).
"""

import argparse
from pathlib import Path

# Mô tả hiện trên trang dataset (card). {n} = số ván, {frames} = số khung.
DESCRIPTION = """Mô phỏng MuJoCo cánh tay **SO-101** gắp hộp đỏ 4 cm bỏ vào bát (repo: github.com/tvhhai/sim-engine, nhánh my-work).

- **{n} ván, {frames} khung, 30 khung/giây.** Chỉ giữ ván thành công.
- **Thầy giáo:** policy viết tay `scripted_minjerk:ScriptedPolicyHybridMinJerk` (IK + quỹ đạo minimum-jerk; 94-99 % thành công
  trên 774 seed chấm).
- **Đầu vào:** `observation.images.top` (camera trên cao, 320x240), `observation.images.wrist` (camera cổ tay), `observation.state`
  (5 góc khớp tính bằng độ + độ mở kẹp 0-100).
- **Action:** 5 góc khớp đích (rad) + lệnh kẹp (-1 khép .. 1 mở).
- **Seed:** 3000-4999 (bộ chấm dùng 0-2499 nên không trùng). Câu lệnh: "Pick up the red box and place it in the bowl".
- **Mốc so sánh:** ACT train trên dữ liệu kiểu này + 4 vòng DAgger, gộp 3 checkpoint: 84,8 % trên 774 seed.
"""


def main():
    parser = argparse.ArgumentParser(description="Đẩy dataset LeRobot lên Hugging Face Hub")
    parser.add_argument("--root", required=True, help="thư mục dataset ở máy")
    parser.add_argument("--repo-id", required=True, help="tên trên Hub: <tài khoản>/<tên>, vd tvhhai/so101_box4_mj")
    parser.add_argument("--first", type=int, default=None, help="chỉ đẩy N ván đầu")
    parser.add_argument("--private", action="store_true", help="repo riêng tư")
    parser.add_argument("--dry-run", action="store_true", help="chỉ in thông tin, không đẩy")
    args = parser.parse_args()

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    # repo_id ở đây là tên trên Hub; dữ liệu vẫn đọc từ root ở máy.
    ds = LeRobotDataset(args.repo_id, root=args.root, video_backend="pyav")
    if args.first and args.first < ds.num_episodes:
        from lerobot.datasets.dataset_tools import delete_episodes
        out = Path(f"{args.root}_first{args.first}")
        drop = list(range(args.first, ds.num_episodes))           # bỏ các ván từ N trở đi
        ds = delete_episodes(ds, drop, output_dir=out, repo_id=args.repo_id)
        print(f"Bản con {args.first} ván: {out}")
    print(f"{args.repo_id}: {ds.num_episodes} ván, {ds.num_frames} khung, camera {[k for k in ds.features if 'images' in k]}, "
          f"{'riêng tư' if args.private else 'công khai'}")
    if args.dry_run:
        return
    ds.push_to_hub(tags=["LeRobot", "so101", "mujoco", "simulation", "pick-and-place"], license="apache-2.0",
                   private=args.private,
                   dataset_description=DESCRIPTION.format(n=ds.num_episodes, frames=ds.num_frames))
    print(f"Đã đẩy: https://huggingface.co/datasets/{args.repo_id}")
    print(f"Xem từng ván: https://huggingface.co/spaces/lerobot/visualize_dataset?path={args.repo_id}")


if __name__ == "__main__":
    main()
