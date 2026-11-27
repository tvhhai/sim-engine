"""DAgger (kiểu "thầy tiếp quản"): cho AI tự lái, tới một lúc thì thầy giáo cầm lái làm nốt ván, ghi lại đoạn thầy làm.

Cách dùng (môi trường có LeRobot):
    python -m sim_engine.learning.dagger --policy D:/vla/outputs/train/act_box4_topwrist/checkpoints/last/pretrained_model \
        --episodes 200 --root D:/vla/datasets/so101_box4_topwrist_dagger1 --repo-id local/so101_box4_topwrist_dagger1

Vì sao cần: dataset gốc chỉ có cảnh THẦY gặp (tay luôn đúng đường). Khi AI tự chạy, mỗi bước lệch một chút, tay tới những chỗ thầy
chưa từng đi qua, AI chưa thấy cảnh đó bao giờ nên đoán sai thêm (lỗi cộng dồn). DAgger lấp lỗ hổng đó: đưa AI vào đúng những cảnh
NÓ tự gây ra, rồi hỏi thầy "ở cảnh này thì làm gì", ghi lại câu trả lời để train thêm.

DAgger gốc hỏi thầy MỖI bước, nhưng vẫn để AI lái. Thầy của ta là policy viết tay có trạng thái (phase, kế hoạch kẹp), không trả lời
được "một bước lẻ" giữa chừng, nên ta làm biến thể hay dùng với thầy kiểu này (gần HG-DAgger / "expert takeover"):
    1. reset ván (seed mới, không trùng seed đánh giá), AI tự chạy t_b bước, t_b ngẫu nhiên trong [--branch-min, --branch-max]
    2. ở bước t_b: thầy mới (policy mặc định bản minimum-jerk) reset TỪ TƯ THẾ HIỆN TẠI và làm nốt ván
    3. ghi các khung của đoạn thầy làm (ảnh + góc khớp + lệnh thầy). Thầy làm hỏng thì bỏ ván.
Không tách nhánh khi AI đang kẹp vật (thầy mới reset sẽ mở kẹp làm rơi vật): chờ tới khi không kẹp nữa, quá 60 bước thì bỏ ván.
Ván AI tự thành công trước t_b thì không có gì để sửa: bỏ (vẫn đếm để biết AI đang giỏi cỡ nào).

Kết quả là một dataset LeRobot RIÊNG. Gộp với dataset gốc bằng learning/merge_datasets.py rồi train tiếp
(learning/configs/act_finetune.yaml). Có thể lặp nhiều vòng: train -> dagger -> gộp -> train ... (notebook Colab mục 6).
"""

import argparse
import contextlib
import io
import json
import time
from pathlib import Path

import numpy as np

from .. import config
from . import sim_setup
from .eval_policy import act, camera_spec, load_policy

GRASP_WAIT_MAX = 60      # AI đang kẹp vật ở bước t_b: chờ tối đa chừng này bước cho tới lúc không kẹp nữa rồi mới tách nhánh
TEACHER_FRAMES_MAX = 400 # thầy mất nhiều bước hơn thế (lòng vòng, thử lại nhiều lần) thì bỏ ván: dạy AI lòng vòng chẳng ích gì
                         # (chạy thử: 1/3 ván thầy dùng tròn 500 khung)


def make_teacher(env):
    """Thầy giáo tiếp quản giữa ván: bản minimum-jerk + V2 dự phòng. reset() lấy tư thế HIỆN TẠI làm điểm xuất phát (phase 0 nhấc
    thẳng lên từ chỗ đang đứng rồi đi tới vật), nên dùng được ở bất kỳ lúc nào tay không kẹp vật.

    Hộp đã nằm trong lòng bát (theo chiều ngang) mà chưa tính thành công (AI vừa thả, hộp đang rơi / gác nghiêng trên thành): thầy bắt đầu
    từ phase 8 (mở kẹp, rút lên khỏi bát, chờ hộp nằm yên, rồi về home; hộp rơi ra ngoài thì tự gắp lại). Trước đây thầy bắt đầu từ phase 0
    và cố gắp hộp RA KHỎI bát: kẹp hụt 3 lần rồi kẹt 200 bước ở phase 5. Đo 93 điểm tách nhánh (AI act_box4_topwrist, seed 6000-6119):
    10/10 ván "thầy lòng vòng" đều là ca này. Ca này lại dạy đúng việc ACT hay quên: thả xong thì rút tay về home."""
    from ..policies.scripted_minjerk import ScriptedPolicyHybridMinJerk
    teacher = ScriptedPolicyHybridMinJerk(env)
    teacher.reset()
    box, bowl = env.get_box_pos(), env.get_bowl_pos()
    if not env._grasped and float(np.linalg.norm(box[:2] - bowl[:2])) < env.bowl_inner:
        teacher.v1._phase, teacher.v1._phase_step = 8, 0
    return teacher


def run_episode(seed, members, device, kind, size, task, t_branch):
    """Một ván DAgger. members = [(mạng, pre, post), ...]: một mạng, hoặc nhiều checkpoint lái chung (trung bình action, xem
    eval_policy.act). Trả (kết quả: "ai_ok" / "no_branch" / "teacher_ok" / "teacher_fail", số bước AI đã lái, danh sách khung)."""
    cameras, env_state = camera_spec(members[0])
    env, options = sim_setup.make_env(seed, kind, size, images=bool(cameras))
    obs, info = env.reset(seed=seed, options=options)
    for policy, pre, post in members:
        policy.reset(); pre.reset(); post.reset()
    try:
        # 1) AI lái tới t_b (kéo dài thêm tối đa GRASP_WAIT_MAX bước nếu lúc đó đang kẹp vật)
        t = 0
        while t < t_branch or env._grasped:
            if t >= t_branch + GRASP_WAIT_MAX or t >= config.MAX_EPISODE_STEPS:
                return "no_branch", t, []
            frame = sim_setup.to_frame(env, obs, np.zeros(7), task, cameras, env_state)
            observation = {k: v for k, v in frame.items() if k.startswith("observation.")}
            action = act(members, observation, device, task)
            obs, _, _, _, info = env.step(sim_setup.to_env_action(action.cpu().numpy()))
            t += 1
            if info.get("success"):
                return "ai_ok", t, []
        # 2) thầy tiếp quản, có đủ MAX_EPISODE_STEPS bước của riêng mình
        env._step_count = 0
        teacher = make_teacher(env)
        frames = []
        for _ in range(config.MAX_EPISODE_STEPS):
            with contextlib.redirect_stdout(io.StringIO()):       # thầy in nhiều dòng debug
                action = teacher.select_action({"state": obs["state"], "images": {}, "task": task})
            # 3) ghi (quan sát TRƯỚC khi làm, lệnh thầy) như record_dataset
            frames.append(sim_setup.to_frame(env, obs, action, task, cameras, env_state))
            obs, _, _, truncated, info = env.step(action)
            if truncated or teacher._phase > 10:
                break
        _, info, _ = env.settle(action, info)
        if not info.get("success"):
            return "teacher_fail", t, frames
        return ("teacher_ok" if len(frames) <= TEACHER_FRAMES_MAX else "teacher_slow"), t, frames
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description="DAgger: AI lái, thầy tiếp quản, ghi đoạn thầy sửa thành dataset LeRobot")
    parser.add_argument("--policy", required=True, nargs="+",
                        help="thư mục pretrained_model của checkpoint AI lái; nhiều thư mục = gộp (trung bình action). Vòng 4 cho "
                             "mạng gộp 3 checkpoint (84,8 %) lái: mạng giỏi hơn thì lệch ở chỗ khó hơn, thầy sửa đúng chỗ còn yếu")
    parser.add_argument("--episodes", type=int, default=200, help="số ván THẦY SỬA THÀNH CÔNG cần ghi")
    parser.add_argument("--kind", choices=["box", "sphere"], default="box")
    parser.add_argument("--size", type=float, default=0.04)
    parser.add_argument("--color", default="#e5484d")
    parser.add_argument("--color-name", default="red")
    parser.add_argument("--seed-start", type=int, default=6000, help="seed riêng cho DAgger (dataset gốc dùng 3000+, đánh giá dưới 3000)")
    parser.add_argument("--branch-min", type=int, default=15, help="AI lái ít nhất chừng này bước rồi thầy mới tiếp quản")
    parser.add_argument("--branch-max", type=int, default=250)
    parser.add_argument("--root", required=True, help="thư mục dataset mới (chưa tồn tại)")
    parser.add_argument("--repo-id", default="local/so101_dagger")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    members = [load_policy(path, args.device) for path in args.policy]
    cameras, env_state = camera_spec(members[0])
    task = sim_setup.TASKS[args.kind].format(color=args.color_name)
    ds = LeRobotDataset.create(args.repo_id, sim_setup.FPS, sim_setup.features(cameras, env_state), root=args.root,
                               robot_type="so101_follower", use_videos=bool(cameras), streaming_encoding=bool(cameras))
    rng = np.random.default_rng(0)
    log, seed, t0 = {"kept": [], "skipped": []}, args.seed_start, time.time()
    try:
        while len(log["kept"]) < args.episodes:
            t_branch = int(rng.integers(args.branch_min, args.branch_max + 1))
            result, t_ai, frames = run_episode(seed, members, args.device, args.kind, args.size, task, t_branch)
            if result == "teacher_ok":
                for frame in frames:
                    ds.add_frame(frame)
                ds.save_episode()
                log["kept"].append({"seed": seed, "ai_steps": t_ai, "frames": len(frames)})
            else:
                log["skipped"].append({"seed": seed, "result": result, "ai_steps": t_ai})
            n = {k: sum(1 for s in log["skipped"] if s["result"] == k) for k in ("ai_ok", "no_branch", "teacher_fail", "teacher_slow")}
            print(f"seed {seed}: {result} (AI lái {t_ai} bước, thầy {len(frames)} khung) | đã ghi {len(log['kept'])}/{args.episodes}, "
                  f"AI tự xong {n['ai_ok']}, không tách được {n['no_branch']}, thầy hỏng {n['teacher_fail']}, thầy lòng vòng "
                  f"{n['teacher_slow']}, {time.time() - t0:.0f} s",
                  flush=True)
            seed += 1
    finally:
        ds.finalize()
        log.update(policy=args.policy, kind=args.kind, size=args.size, task=task, cameras=list(cameras), env_state=env_state,
                   branch=[args.branch_min, args.branch_max], seconds=round(time.time() - t0))
        (Path(args.root) / "dagger_log.json").write_text(json.dumps(log, indent=1), encoding="utf-8")
    print(f"Xong: {len(log['kept'])} ván -> {args.root}")


if __name__ == "__main__":
    main()
