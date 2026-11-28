"""Công cụ thống kê chuẩn hoá (mean / std / min / max / quantile) của state và action khi train tiếp một checkpoint LeRobot.

Vì sao cần: LeRobot 0.7.0, khi train tiếp bằng `--policy.path` (không phải `--resume`), thay thống kê trong bộ tiền xử lý (chuẩn hoá
state, action mục tiêu) và hậu xử lý (giải chuẩn hoá action) bằng thống kê của dataset MỚI (`lerobot_train.py` dòng 665-692). Mạng đã học
với thang đo cũ nên cùng một đầu ra bị giải thành action khác. Phép thử 0 bước train (2026-11-28): SmolVLA 30k chỉ đổi thống kê sang
dataset hộp + bóng đã tụt 22/31 -> 14/31 trên cùng seed.

Ba lệnh:
  swap  --ckpt A --dataset D --out O   chép checkpoint A sang O, thay thống kê state / action bằng của dataset D (phép thử 0 bước train)
  keep  --ckpt A --dataset D --out O   dựng dataset O = D (data / videos là liên kết, không chép) nhưng meta/stats.json mang thống kê
                                       state / action của checkpoint A: train tiếp trên O thì LeRobot giữ đúng thang đo của A
  check --ckpt A --other B             so thống kê state / action trong processor của hai checkpoint (vd sau khi train trên O, checkpoint
                                       mới phải giữ nguyên thống kê của A)
"""

import argparse
import json
import os
import shutil
from pathlib import Path

KEYS = ("action", "observation.state")   # chỉ hai đặc trưng chuẩn hoá MEAN_STD; ảnh là IDENTITY nên không đụng


def _processor_files(ckpt):
    return sorted(Path(ckpt).glob("policy_*processor_step_*normalizer_processor.safetensors"))


def _load(path):
    from safetensors.torch import load_file
    return load_file(str(path))


def ckpt_stats(ckpt):
    """Thống kê state / action mà checkpoint thực sự dùng (lấy từ bộ hậu xử lý; bộ tiền xử lý phải giống, kiểm ở check)."""
    out = {}
    for f in _processor_files(ckpt):
        for name, t in _load(f).items():
            feat, stat = name.rsplit(".", 1)
            if feat in KEYS:
                out.setdefault(feat, {})[stat] = t.flatten().tolist()
    return out


def swap(ckpt, dataset, out):
    """Checkpoint O = trọng số của A + thống kê state / action của dataset D (cả tiền xử lý lẫn hậu xử lý)."""
    import torch
    from safetensors.torch import save_file
    stats = json.loads((Path(dataset) / "meta" / "stats.json").read_text(encoding="utf-8"))
    out = Path(out)
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(ckpt, out)
    changed = 0
    for f in _processor_files(out):
        tensors = _load(f)
        for name in list(tensors):
            feat, stat = name.rsplit(".", 1)
            if feat in KEYS and stat in stats.get(feat, {}):
                new = torch.tensor(stats[feat][stat], dtype=tensors[name].dtype).reshape(tensors[name].shape)
                changed += int(not torch.equal(new, tensors[name]))
                tensors[name] = new
        save_file(tensors, str(f))
    print(f"swap: {out} ({changed} tensor đổi giá trị)")


def keep(ckpt, dataset, out):
    """Dataset O: liên kết tới data / videos của D, meta chép riêng, meta/stats.json lấy thống kê state / action của checkpoint A."""
    dataset, out = Path(dataset), Path(out)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    for item in dataset.iterdir():
        if item.name == "meta":
            shutil.copytree(item, out / "meta")
        else:
            os.symlink(item.resolve(), out / item.name, target_is_directory=item.is_dir())
    stats_path = out / "meta" / "stats.json"
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    old = ckpt_stats(ckpt)
    for feat in KEYS:
        if feat not in old:
            raise SystemExit(f"checkpoint {ckpt} không có thống kê {feat}")
        for stat, values in old[feat].items():
            stats.setdefault(feat, {})[stat] = values
    stats_path.write_text(json.dumps(stats, indent=4), encoding="utf-8")
    print(f"keep: {out} (state / action theo {ckpt})")


def check(ckpt, other):
    """In độ lệch lớn nhất giữa thống kê state / action của hai checkpoint, và giữa tiền / hậu xử lý trong từng checkpoint."""
    worst = 0.0
    for path in (ckpt, other):
        files = _processor_files(path)
        maps = [{k: v for k, v in _load(f).items() if k.rsplit(".", 1)[0] in KEYS} for f in files]
        for k in set(maps[0]) & set(maps[-1]):
            worst = max(worst, float((maps[0][k] - maps[-1][k]).abs().max()))
    a, b = ckpt_stats(ckpt), ckpt_stats(other)
    diffs = {f"{feat}.{stat}": max(abs(x - y) for x, y in zip(a[feat][stat], b[feat][stat]))
             for feat in KEYS for stat in ("mean", "std") if feat in a and feat in b}
    print("lệch tiền / hậu xử lý trong cùng checkpoint:", round(worst, 6))
    for k, v in diffs.items():
        print(f"{k}: lệch lớn nhất {v:.6f}")
    return max(diffs.values(), default=0.0)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("cmd", choices=["swap", "keep", "check"])
    parser.add_argument("--ckpt", required=True, help="thư mục pretrained_model của checkpoint")
    parser.add_argument("--dataset", help="thư mục dataset LeRobot (có meta/stats.json)")
    parser.add_argument("--out", help="thư mục kết quả (bị xoá nếu đã có)")
    parser.add_argument("--other", help="checkpoint thứ hai (lệnh check)")
    parser.add_argument("--max-diff", type=float, default=None, help="check: lệch mean / std vượt mức này thì thoát mã 1")
    args = parser.parse_args()
    if args.cmd == "swap":
        swap(args.ckpt, args.dataset, args.out)
    elif args.cmd == "keep":
        keep(args.ckpt, args.dataset, args.out)
    else:
        diff = check(args.ckpt, args.other)
        if args.max_diff is not None and diff > args.max_diff:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
