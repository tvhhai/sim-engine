"""Chạy lerobot-train với tham số đọc từ một file YAML trong learning/configs/ (một file = một thí nghiệm, dễ so sánh và ghi lại).

    python -m sim_engine.learning.train learning/configs/act_box4_v1.yaml            # train
    python -m sim_engine.learning.train learning/configs/act_box4_v1.yaml --steps=200  # ghi đè tham số khi chạy thử

Tham số thêm sau tên file (dạng --khoá=giá_trị) được chuyển thẳng cho lerobot-train và ghi đè giá trị trong YAML.
"""

import subprocess
import sys
from pathlib import Path

import yaml


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    cfg = yaml.safe_load(Path(sys.argv[1]).read_text(encoding="utf-8"))
    extra = sys.argv[2:]
    overridden = {a.split("=", 1)[0].lstrip("-") for a in extra}
    args = [f"--{k}={str(v).lower() if isinstance(v, bool) else v}" for k, v in cfg.items() if k not in overridden] + extra
    exe = Path(sys.executable).with_name("lerobot-train.exe" if sys.platform == "win32" else "lerobot-train")
    print(" ".join([exe.name, *args]), flush=True)
    sys.exit(subprocess.call([str(exe), *args]))


if __name__ == "__main__":
    main()
