"""Gộp nhiều dataset LeRobot thành một (vd dataset gốc 500 ván + các ván DAgger) để train trên tất cả.

    python -m sim_engine.learning.merge_datasets --out D:/vla/datasets/so101_box4_topwrist_500_dagger1 \
        --repo-id local/so101_box4_topwrist_500_dagger1 \
        D:/vla/datasets/so101_box4_topwrist_500 D:/vla/datasets/so101_box4_topwrist_dagger1

Các dataset phải cùng trường dữ liệu (cùng camera, cùng có / không toạ độ vật) và cùng fps. Dùng aggregate_datasets của LeRobot:
nối video, bảng số, đánh số lại ván và tính lại thống kê chuẩn hoá cho dataset mới. Dataset nguồn không bị đổi.
"""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Gộp nhiều LeRobotDataset")
    parser.add_argument("sources", nargs="+", help="thư mục các dataset nguồn")
    parser.add_argument("--out", required=True, help="thư mục dataset gộp (chưa tồn tại)")
    parser.add_argument("--repo-id", required=True)
    args = parser.parse_args()

    from lerobot.datasets.aggregate import aggregate_datasets

    roots = [Path(s) for s in args.sources]
    repo_ids = [f"local/{r.name}" for r in roots]               # chỉ là nhãn; dữ liệu đọc từ roots
    aggregate_datasets(repo_ids, args.repo_id, roots=roots, aggr_root=Path(args.out))
    print(f"Đã gộp {len(roots)} dataset -> {args.out}")


if __name__ == "__main__":
    main()
