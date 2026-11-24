"""Bước 2: chạy lerobot-train với tham số đọc từ một file YAML trong learning/configs/ (một file = một thí nghiệm, dễ so sánh và ghi lại).

    python -m sim_engine.learning.train learning/configs/act_box4_v1.yaml            # train
    python -m sim_engine.learning.train learning/configs/act_box4_v1.yaml --steps=200  # ghi đè tham số khi chạy thử

Tham số thêm sau tên file (dạng --khoá=giá_trị) được chuyển thẳng cho lerobot-train và ghi đè giá trị trong YAML.

File này KHÔNG chứa vòng lặp train. Vòng lặp nằm trong LeRobot (lệnh lerobot-train, file lerobot/scripts/lerobot_train.py), tóm tắt:
    lặp `steps` lần:
        lấy ngẫu nhiên `batch_size` khung hình từ dataset (ảnh + state + 50 action thầy làm tiếp theo)
        mạng ACT đoán 50 action
        loss = sai lệch giữa đoán và thầy (L1) + một phần nhỏ KL (giữ "biến ẩn" của ACT gọn)
        lan truyền ngược, optimizer chỉnh trọng số cho loss nhỏ đi
        cứ `save_freq` bước lưu một checkpoint vào output_dir/checkpoints/
Ta chỉ cần chọn tham số. Vì vậy khi muốn "sửa train", thường là tạo file YAML mới trong learning/configs/, không sửa file này.
"""

import subprocess
import sys
from pathlib import Path

import yaml


def main():
    if len(sys.argv) < 2:                       # không truyền file YAML -> in hướng dẫn (docstring ở trên) rồi thoát
        sys.exit(__doc__)
    cfg = yaml.safe_load(Path(sys.argv[1]).read_text(encoding="utf-8"))   # YAML -> dict, vd {"policy.type": "act", "steps": 20000}
    extra = sys.argv[2:]                        # các tham số gõ thêm sau tên file, vd ["--steps=200"]
    overridden = {a.split("=", 1)[0].lstrip("-") for a in extra}   # tên các khoá bị ghi đè, vd {"steps"}
    # Mỗi khoá YAML thành "--khoá=giá_trị". Bool viết thường (true/false) vì lerobot-train đọc kiểu đó.
    # Khoá đã gõ thêm trên dòng lệnh thì bỏ bản trong YAML, để bản dòng lệnh thắng.
    args = [f"--{k}={str(v).lower() if isinstance(v, bool) else v}" for k, v in cfg.items() if k not in overridden] + extra
    # lerobot-train.exe nằm cạnh python.exe trong thư mục Scripts của venv (Windows); Linux/Mac không có ".exe".
    exe = Path(sys.executable).with_name("lerobot-train.exe" if sys.platform == "win32" else "lerobot-train")
    print(" ".join([exe.name, *args]), flush=True)   # in lệnh thật sự chạy, để copy chạy tay hoặc ghi vào sổ thí nghiệm
    sys.exit(subprocess.call([str(exe), *args]))     # chạy và trả mã lỗi của nó (0 = thành công)


if __name__ == "__main__":
    main()
