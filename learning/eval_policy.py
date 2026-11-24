"""Bước 3: đánh giá một policy LeRobot đã train trên bộ seed cố định, ra bảng giống bảng của policy viết tay.

Cách dùng (môi trường có LeRobot, vd D:/vla/venv-lerobot):
    python -m sim_engine.learning.eval_policy --policy D:/vla/outputs/train/act_box4_v1/checkpoints/last/pretrained_model \
        --kind box --size 0.04 --limit 100 --out D:/vla/eval/act_box4_v1.json --replay act_box4_v1

--limit N: chỉ chạy N seed đầu của bộ đánh giá (cách đều trong 774 seed) để có số nhanh; bỏ --limit để chạy đủ 774.
--replay TÊN: ghi thêm results/replays/TÊN.json để xem từng ván trên Studio (Replay > Nguồn replay).
Kết quả: tỉ lệ thành công chung, theo bán kính (r < 37 cm / r >= 37 cm), danh sách seed hỏng; ghi JSON từng seed.

Khác với lúc train: ở đây KHÔNG có thầy giáo. Mạng tự nhìn ảnh, tự ra lệnh, mô phỏng chạy theo lệnh đó, rồi mạng nhìn kết quả và ra lệnh
tiếp (gọi là "vòng kín", closed loop). Sai một chút ở bước trước sẽ làm cảnh ở bước sau khác với những gì mạng từng thấy trong dữ liệu.
Đó là lý do loss lúc train rất thấp mà tỉ lệ gắp vẫn có thể thấp.

Lưu ý: chạy trên GPU nên kết quả mỗi lần chạy lệch nhau một chút (vài %). So hai bản train thì chênh dưới ~5 % chưa nói lên gì.
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

from .. import config
from . import sim_setup


def load_policy(path, device, n_action_steps=None, temporal_ensemble=None):
    """Nạp mạng đã train từ thư mục checkpoint. Trả (mạng, bộ tiền xử lý, bộ hậu xử lý).

    Thư mục `path` (pretrained_model/) chứa:
        config.json                      loại mạng (act), kích thước, chunk_size...
        model.safetensors                các trọng số đã học
        policy_preprocessor*.json/...    cách chuẩn hoá đầu vào (trừ trung bình, chia độ lệch, tính từ dataset)
        policy_postprocessor*.json/...   cách đổi đầu ra của mạng về lại thang action của env
    """
    # Import trong hàm: môi trường không có LeRobot vẫn import được module này (vd để chạy test).
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import get_policy_class, make_pre_post_processors

    cfg = PreTrainedConfig.from_pretrained(path)                 # đọc config.json
    cfg.device = device                                          # "cuda" (GPU) hoặc "cpu"
    # Hai tham số chỉ ảnh hưởng lúc CHẠY (không cần train lại), để thử cách mạng dùng chuỗi action nó đoán:
    #   n_action_steps: làm bao nhiêu action trong 50 action đoán được rồi mới nhìn ảnh và đoán lại (nhỏ = phản ứng nhanh hơn).
    #   temporal_ensemble: đoán lại MỖI bước và lấy trung bình có trọng số các lần đoán chồng lên nhau (cách bài báo ACT dùng);
    #   bắt buộc n_action_steps = 1.
    if temporal_ensemble is not None:
        cfg.temporal_ensemble_coeff, n_action_steps = temporal_ensemble, 1
    if n_action_steps is not None:
        cfg.n_action_steps = n_action_steps
    policy = get_policy_class(cfg.type).from_pretrained(path, config=cfg)   # cfg.type = "act" -> lớp ACTPolicy, nạp trọng số
    policy.to(device).eval()                                     # eval(): tắt dropout, chế độ chạy thật chứ không phải train
    # pre: dict quan sát (numpy) -> tensor đã chuẩn hoá trên GPU; post: tensor mạng đoán -> action thang thật.
    pre, post = make_pre_post_processors(cfg, pretrained_path=path,
                                         preprocessor_overrides={"device_processor": {"device": device}})
    return policy, pre, post


def _recorder(env):
    """Recorder của replay chuẩn (tools/make_replay) dùng cho policy AI.

    Recorder vốn được viết cho policy viết tay: nó hỏi policy đang ở phase nào, đích IK ở đâu. Mạng AI không có những thứ đó (nó
    đoán thẳng góc khớp từ ảnh), nên ta đưa vào một "policy giả" chỉ có id 5 khớp tay. Vì vậy replay AI luôn hiện phase 0.
    """
    from types import SimpleNamespace

    from ..tools.make_replay import Recorder
    joint_ids = [env.model.joint(name).id for name in config.JOINT_NAMES[:5]]
    return Recorder(env, SimpleNamespace(_joint_ids=joint_ids, _current_joint_targets=None))


def run_episode(seed, policy, pre, post, device, kind, size, task, max_steps, record=False):
    """Cho policy AI tự chạy một ván. Trả (dict kết quả, ván replay hoặc None nếu record=False)."""
    import torch
    from lerobot.common.control_utils import predict_action

    # Mạng được train với những đầu vào nào (camera nào, có toạ độ vật không) thì dựng đúng những đầu vào đó.
    cameras, env_state = sim_setup.spec_from_features(policy.config.input_features)
    env, options = sim_setup.make_env(seed, kind, size, images=bool(cameras))   # cùng cảnh với seed đó của policy viết tay
    obs, info = env.reset(seed=seed, options=options)
    rec = _recorder(env) if record else None
    frames = [rec.frame(0, 0, 0, 0, info, 1.0)] if rec else []   # khung 0 của replay: trước khi làm gì
    # Xoá trí nhớ của ván trước. ACT giữ một hàng đợi action: mỗi lần đoán 50 action (chunk_size), làm dần n_action_steps cái
    # rồi mới đoán lại. Không reset thì ván mới sẽ làm nốt action còn thừa của ván cũ.
    policy.reset()
    pre.reset()
    post.reset()
    grasped, t, close_err = False, 0, None
    for t in range(max_steps):
        # 1) Đổi quan sát của env sang định dạng LeRobot (giống hệt lúc ghi dataset), bỏ trường "action" và "task".
        frame = sim_setup.to_frame(env, obs, np.zeros(7), task, cameras, env_state)
        observation = {k: v for k, v in frame.items() if k.startswith("observation.")}
        # 2) Mạng ra lệnh. predict_action làm gọn: chuẩn hoá (pre) -> mạng -> đổi về thang thật (post).
        #    Thực tế mạng chỉ chạy thật sự mỗi n_action_steps bước; các bước giữa lấy action kế tiếp trong hàng đợi.
        action = predict_action(observation, policy, torch.device(device), pre, post, use_amp=False, task=task)
        env_action = sim_setup.to_env_action(action.cpu().numpy())   # 6 số -> 7 số env nhận
        # 3) Mô phỏng làm lệnh đó và trả quan sát mới. Vòng lặp quay lại bước 1 với quan sát mới (vòng kín).
        obs, _, terminated, truncated, info = env.step(env_action)
        grasped = grasped or bool(env._grasped)                  # đã từng kẹp được vật (hai hàm cùng chạm) lúc nào chưa
        if close_err is None and env_action[5] < 0:              # lần đầu mạng ra lệnh khép kẹp (lệnh kẹp < 0)
            # đầu kẹp cách tâm vật bao xa (mm) theo x, y, z ở đúng lúc đó: "kẹp vào không khí" lệch bao nhiêu, lệch hướng nào
            close_err = [round(float(v) * 1000) for v in np.asarray(info["ee_pos"]) - np.asarray(info["box_pos"])]
        if rec:
            frames.append(rec.frame(t + 1, 0, 0, t + 1, info, env_action[5]))   # phase luôn 0: AI không có phase
        if info.get("success") or truncated:                     # vật nằm yên trong bát, hoặc hết 500 bước
            break
    # r = khoảng cách vật tới trục khớp 1 (cm): để chia bảng theo vùng gần / xa giống bảng policy viết tay.
    x0, y0 = sim_setup.seed_scene(seed)[0]
    r = math.hypot(x0 - config.PICK_PAN_XY[0], y0 - config.PICK_PAN_XY[1])
    row = {"seed": seed, "ok": bool(info.get("success")), "steps": t + 1, "grasped": grasped, "r": round(r, 3),
           "end_xy": [round(float(v), 3) for v in info["box_pos"][:2]],   # end_xy: vật nằm đâu khi hết ván
           "close_err_mm": close_err}                                      # lệch đầu kẹp - vật lúc khép lần đầu (mm)
    episode = None
    if rec:
        from ..tools.make_replay import bowl_geom_list
        episode = {"seed": seed, "bowl": [round(float(v), 4) for v in env.get_bowl_pos()], "bowl_geoms": bowl_geom_list(env),
                   "success": row["ok"], "steps": len(frames) - 1, "frames": frames}
    env.close()
    return row, episode


def label(row):
    """Nhóm kết quả của một ván. Hai kiểu hỏng gợi ý hai nguyên nhân khác nhau:
    chưa kẹp được  -> tìm / tiếp cận vật kém (nhìn sai vị trí, tay lệch dần);
    đã kẹp nhưng hỏng -> kẹp lỏng, rơi giữa đường, thả trượt bát."""
    if row["ok"]:
        return "Thành công"
    return "Đã kẹp nhưng hỏng" if row["grasped"] else "Chưa kẹp được"


def write_replay(name, policy_name, kind, size, rows, episodes):
    """Ghi results/replays/<name>.json (Studio tự liệt kê mọi file trong thư mục này) và bản .html mở offline."""
    from .. import paths
    from ..tools import make_replay

    kind_vi = "hộp" if kind == "box" else "bóng"
    keep = []
    for row, ep in zip(rows, episodes):
        if ep is None:                                           # ván không được giữ (nhóm đã đủ số lượng)
            continue
        group = label(row)
        tags = [f"AI: {policy_name}", group, f"vật cách trục khớp 1 {row['r'] * 100:.0f} cm", f"{row['steps']} bước"]
        ep.update(name=f"seed {row['seed']} · {policy_name} · {'OK' if row['ok'] else 'FAIL'}", group=group, tags=tags)
        keep.append(ep)
    # Phần "scene" của replay (bàn, cánh tay, camera, vùng gắp) lấy từ một env bất kỳ cùng loại vật.
    env, _ = sim_setup.make_env(sim_setup.EVAL_SEEDS[0], kind, size, images=False)
    out_dir = paths.RESULTS / "replays"
    out_dir.mkdir(parents=True, exist_ok=True)
    data = make_replay.write_replay(env, keep, str(out_dir / f"{name}.html"), policy_name)
    env.close()
    ok = sum(r["ok"] for r in rows)
    data["scene"]["title"] = f"{policy_name} · {kind_vi} {size * 100:g} cm · {ok}/{len(rows)} thành công · {len(keep)} ván"
    with open(out_dir / f"{name}.json", "w", encoding="utf-8") as f:
        json.dump(data, f, separators=(",", ":"), ensure_ascii=False)   # separators gọn: file nhỏ hơn
    print(f"Đã ghi replay results/replays/{name}.json ({len(keep)} ván): mở Studio > Replay > Nguồn replay")


def summarize(rows):
    """Bảng tóm tắt: tỉ lệ chung, gần (r < 37 cm), xa (r >= 37 cm), và danh sách seed theo kiểu hỏng."""
    def rate(sel):
        return f"{sum(r['ok'] for r in sel)}/{len(sel)} ({100 * sum(r['ok'] for r in sel) / max(1, len(sel)):.1f} %)"
    near = [r for r in rows if r["r"] < 0.37]
    far = [r for r in rows if r["r"] >= 0.37]
    return {"all": rate(rows), "r<37cm": rate(near), "r>=37cm": rate(far),
            "grasped_but_failed": [r["seed"] for r in rows if r["grasped"] and not r["ok"]],
            "never_grasped": [r["seed"] for r in rows if not r["grasped"]]}


def main():
    # --- Tham số dòng lệnh ---
    parser = argparse.ArgumentParser(description="Đánh giá policy LeRobot trên bộ seed cố định")
    parser.add_argument("--policy", required=True, help="thư mục pretrained_model của checkpoint")
    parser.add_argument("--kind", choices=["box", "sphere"], default="box")
    parser.add_argument("--size", type=float, default=0.04)
    parser.add_argument("--color-name", default="red")
    parser.add_argument("--limit", type=int, default=None, help="chỉ chạy N seed (cách đều) của bộ 774 seed")
    parser.add_argument("--max-steps", type=int, default=config.MAX_EPISODE_STEPS)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", default=None, help="file JSON kết quả từng seed")
    parser.add_argument("--replay", default=None, help="tên replay ghi vào results/replays/<tên>.json để xem trên Studio")
    parser.add_argument("--replay-per-group", type=int, default=15, help="số ván tối đa mỗi nhóm trong replay (file nhỏ, mở nhanh)")
    parser.add_argument("--name", default=None, help="tên policy hiện trên replay (mặc định: tên thư mục train)")
    parser.add_argument("--n-action-steps", type=int, default=None, help="ghi đè số action làm trước khi đoán lại (không cần train lại)")
    parser.add_argument("--temporal-ensemble", type=float, default=None, help="vd 0.01: đoán mỗi bước, trung bình các lần đoán")
    parser.add_argument("--seeds", default=None, help="danh sách seed riêng, vd 3000-3019 (mặc định bộ 774 seed đánh giá)")
    args = parser.parse_args()

    # --- Chọn seed: --limit N lấy N seed cách đều trong 774 seed (phủ cả gần lẫn xa, cả seed 0-199 lẫn 2000-2499) ---
    if args.seeds:                                               # vd "3000-3019" (seed đã train) để xem mạng học thuộc được chưa
        from ..tools.make_replay import parse_seeds
        seeds = parse_seeds(args.seeds)
    else:
        seeds = sim_setup.EVAL_SEEDS
    if args.limit:
        seeds = [seeds[i] for i in np.linspace(0, len(seeds) - 1, args.limit).round().astype(int)]
    task = sim_setup.TASKS[args.kind].format(color=args.color_name)
    policy, pre, post = load_policy(args.policy, args.device, args.n_action_steps, args.temporal_ensemble)
    # Tên hiện trên replay: thư mục ngay dưới ".../train/", vd D:/vla/outputs/train/act_box4_v1/... -> "act_box4_v1".
    policy_name = args.name or next((p.name for p in Path(args.policy).parents if p.parent.name == "train"), "AI")

    # --- Chạy từng seed ---
    rows, episodes, per_group, t0 = [], [], {}, time.time()
    for k, seed in enumerate(seeds):
        row, episode = run_episode(seed, policy, pre, post, args.device, args.kind, args.size, task, args.max_steps,
                                   record=bool(args.replay))
        if episode is not None:                                  # chỉ giữ tối đa --replay-per-group ván mỗi nhóm (đỡ tốn RAM)
            group = label(row)
            per_group[group] = per_group.get(group, 0) + 1
            if per_group[group] > args.replay_per_group:
                episode = None
        rows.append(row)
        episodes.append(episode)
        print(f"[{k + 1}/{len(seeds)}] seed {seed}: {'OK' if row['ok'] else 'HỎNG'}"
              f"{' (đã kẹp)' if row['grasped'] and not row['ok'] else ''} {row['steps']} bước | "
              f"thành công {sum(r['ok'] for r in rows)}/{len(rows)} | {time.time() - t0:.0f} s", flush=True)

    # --- Tổng kết và ghi file ---
    summary = summarize(rows)
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps({"policy": args.policy, "kind": args.kind, "size": args.size, "summary": summary,
                                              "rows": rows}, ensure_ascii=False, indent=1), encoding="utf-8")
    if args.replay:
        write_replay(args.replay, policy_name, args.kind, args.size, rows, episodes)


if __name__ == "__main__":
    main()
