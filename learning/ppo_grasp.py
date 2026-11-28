"""Thử PPO chọn nhịp gắp cao cấp. Mỗi hành động điều khiển một ván MuJoCo."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from copy import copy
import contextlib
import io
import json
import math
from pathlib import Path

import mujoco
import numpy as np

from .. import box_sampling, config, paths
from ..policies.scripted_minjerk import ScriptedPolicyHybridMinJerk
from ..tools.object_replay import _make_env, _seed_scene

ACTIONS = ("teacher", "carry24", "carry30", "arc82")
# Ba ứng viên đã đo trong pilot 5 seed. Hộp 1016 giảm 373 → 242 nhờ vòng cung 82.
ARC82 = dict(zip(("above", "down", "close", "sweep", "release", "raise"), (16, 12, 12, 24, 8, 10)))
CARRY_LIMITS = {1: (.024, 1.8), 2: (.030, 2.0)}
FEATURES = 8


class GraspTrial(ScriptedPolicyHybridMinJerk):
    def __init__(self, env):
        super().__init__(env)
        self.plan = None

    def _choose_profile(self, profiles):
        self.plan = super()._choose_profile(profiles)
        return self.plan

    def apply_action(self, action):
        if action in CARRY_LIMITS:
            self.v1.TRANSFER_STEP_LIMIT, self.v2.VMAX_CARRY = CARRY_LIMITS[action]
        elif action == 3:
            self.v1.ARC_STEPS = ARC82

    def guarded_action(self, action):
        """Giữ nhịp của thầy nếu lượt thử PPO không giữ và đặt được vật."""
        if not action:
            return 0
        shadow = copy(self.env)
        shadow.data = mujoco.MjData(self.env.model)
        shadow.render_images = False
        mujoco.mj_copyData(shadow.data, self.env.model, self.env.data)
        probe = GraspTrial(shadow)
        probe._reset_profile(*self.plan)
        probe.apply_action(action)
        result = rollout(shadow, probe, shadow._get_obs())
        return action if result["ok"] and result["held"] else 0


def geometry(xy):
    x, y = np.asarray(xy) - config.PICK_PAN_XY
    return math.hypot(x, y), math.atan2(x, y)


def describe(env):
    obj = env.get_box_pos()
    target = env.get_bowl_pos()
    radius, angle = geometry(obj[:2])
    _, bowl_angle = geometry(target[:2])
    speed = np.linalg.norm(env.data.qvel[env._box_dof_adr:env._box_dof_adr + 2])
    sphere = env.target_type == "sphere"
    edge = not sphere and radius >= .37 and abs(angle) >= math.radians(107)
    features = [
        float(sphere), radius / .45, angle / math.pi, bowl_angle / math.pi,
        np.linalg.norm(obj[:2] - target[:2]) / .75, speed, float(edge), obj[2] - .42,
    ]
    return [float(v) for v in features], [True, True, True, edge]


def rollout(env, actor, obs):
    phase_counts = Counter()
    visits = Counter()
    previous_phase = None
    previous_grip = 1.
    held = False
    closes = 0
    for step in range(1, config.MAX_EPISODE_STEPS + 1):
        phase = actor._phase
        phase_counts[phase] += 1
        if phase != previous_phase:
            visits[phase] += 1
            previous_phase = phase
        action = actor.select_action(obs)
        closes += int(2 <= phase <= 5 and previous_grip >= 0 > action[5])
        previous_grip = action[5]
        obs, _, _, truncated, info = env.step(action)
        held |= env._grasped
        if truncated or actor._phase > 10:
            break
    _, info, waited = env.settle(action, info)
    return dict(ok=bool(info["success"]), held=bool(held), steps=step, waited=waited,
                revisits=sum(n - 1 for n in visits.values()), closes=closes, carry=phase_counts[6])


def reward(result, baseline):
    if not (result["ok"] and result["held"]):
        return -5.
    faster = (baseline["steps"] + baseline["waited"] - result["steps"] - result["waited"]) / 100
    repeats = max(0, result["revisits"] - baseline["revisits"])
    closures = max(0, result["closes"] - baseline["closes"])
    return float(np.clip(faster, -2., 2.) - .5 * repeats - .5 * closures)


def work(job):
    kind, seed, mode, weights = job
    xy, bowl = _seed_scene(seed)
    env = _make_env(xy, kind, .04)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            options = {"bowl_xy": bowl}
            obs, _ = env.reset(seed=seed, options=options)
            features, allowed = describe(env)
            actor = GraspTrial(env)
            actor.reset()
            if mode == "eval":
                action = predict(weights, features, allowed)
                actor.apply_action(action)
                result = rollout(env, actor, obs)
                return dict(kind=kind, seed=seed, action=action, result=result, plan=actor.plan)
            results = {"0": rollout(env, actor, obs)}
            for action in range(1, len(ACTIONS)):
                if allowed[action]:
                    obs, _ = env.reset(seed=seed, options=options)
                    actor._reset_profile(*actor.plan)
                    actor.apply_action(action)
                    results[str(action)] = rollout(env, actor, obs)
            return dict(kind=kind, seed=seed, x=features, mask=allowed, results=results)
    finally:
        env.close()


def predict(weights, features, allowed):
    hidden = np.tanh(np.asarray(weights[0]) @ features + weights[1])
    logits = np.asarray(weights[2]) @ hidden + weights[3]
    return int(np.argmax(np.where(allowed, logits, -np.inf)))


def jobs_for(mode, weights=None):
    if mode == "eval":
        seeds = sorted(set(range(200)) | {s for s in config.BOX_FIXED_SEEDS if s >= 200} | set(range(2000, 2500)))
        assert len(seeds) == 774
        return [(k, s, mode, weights) for s in seeds for k in ("box", "sphere")]
    lo, hi, count, edge_count = (3000, 6000, 24, 10) if mode == "train" else (2500, 3000, 12, 5)
    rng = np.random.default_rng(799)
    seeds = list(map(int, rng.permutation(np.arange(lo, hi))))
    standard = seeds[:count]
    edge = []
    for seed in seeds[count:]:
        r, a = geometry(box_sampling.box_xy_for_seed(seed))
        if r >= .372 and abs(a) >= math.radians(107.2):
            edge.append(seed)
            if len(edge) >= edge_count:
                break
    return [(k, s, mode, None) for s in standard for k in ("box", "sphere")] + [("box", s, mode, None) for s in edge]


def fit(folder):
    import torch
    from torch import nn

    torch.manual_seed(73)
    torch.set_num_threads(1)
    rows = [json.loads(line) for line in (folder / "train.jsonl").read_text().splitlines()]
    valid = [json.loads(line) for line in (folder / "val.jsonl").read_text().splitlines()]
    x = torch.tensor([r["x"] for r in rows], dtype=torch.float32)
    mask = torch.tensor([r["mask"] for r in rows])
    values = torch.tensor([[reward(r["results"][str(a)], r["results"]["0"]) if allowed else 0.
                            for a, allowed in enumerate(r["mask"])] for r in rows], dtype=torch.float32)
    actor = nn.Sequential(nn.Linear(FEATURES, 32), nn.Tanh(), nn.Linear(32, len(ACTIONS)))
    critic = nn.Sequential(nn.Linear(FEATURES, 32), nn.Tanh(), nn.Linear(32, 1))
    optimizer = torch.optim.Adam([*actor.parameters(), *critic.parameters()], lr=.003)
    history = []
    best = None
    for update in range(1, 161):
        # Một quyết định mỗi ván. Return là thưởng cuối ván, tương đương GAE với ván một bước.
        with torch.no_grad():
            state = torch.randint(len(x), (512,))
            inputs = x[state]
            allowed = mask[state]
            old = torch.distributions.Categorical(logits=actor(inputs).masked_fill(~allowed, -1e9))
            action = old.sample()
            old_logp = old.log_prob(action)
            returns = values[state, action]
            adv = returns - critic(inputs).squeeze(1)
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        for _ in range(4):
            for batch in torch.randperm(len(action)).split(64):
                dist = torch.distributions.Categorical(logits=actor(inputs[batch]).masked_fill(~allowed[batch], -1e9))
                ratio = (dist.log_prob(action[batch]) - old_logp[batch]).exp()
                clipped = torch.clamp(ratio, .8, 1.2)
                objective = torch.minimum(ratio * adv[batch], clipped * adv[batch]).mean()
                value_loss = (critic(inputs[batch]).squeeze(1) - returns[batch]).square().mean()
                loss = -objective + .5 * value_loss - .01 * dist.entropy().mean()
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_([*actor.parameters(), *critic.parameters()], .5)
                optimizer.step()
        if update == 1 or update % 5 == 0:
            weights = [p.detach().tolist() for p in actor.parameters()]
            decisions = [predict(weights, r["x"], r["mask"]) for r in valid]
            outcomes = [r["results"][str(a)] for r, a in zip(valid, decisions)]
            failed = sum(not (r["ok"] and r["held"]) for r in outcomes)
            total = sum(r["steps"] + r["waited"] for r in outcomes)
            revisits = sum(r["revisits"] for r in outcomes)
            key = (failed, total, revisits)
            entry = dict(update=update, mean_reward=float(returns.mean()), failed=failed,
                         total_steps=total, revisits=revisits, actions=dict(Counter(decisions)))
            history.append(entry)
            if best is None or key < best:
                best = key
                (folder / "model.json").write_text(json.dumps(dict(weights=weights, validation=entry)), encoding="utf8")
    (folder / "ppo_history.json").write_text(json.dumps(history, indent=2), encoding="utf8")
    print("PPO validation", best, len(valid), flush=True)


def collect(folder, mode, workers):
    file = folder / (mode + ".jsonl")
    done = set()
    if file.exists():
        for line in file.read_text(encoding="utf8").splitlines():
            row = json.loads(line)
            done.add((row["kind"], row["seed"]))
    weights = json.loads((folder / "model.json").read_text(encoding="utf8"))["weights"] if mode == "eval" else None
    jobs = [job for job in jobs_for(mode, weights) if job[:2] not in done]
    print(mode, len(done), "done,", len(jobs), "pending", flush=True)
    with file.open("a", encoding="utf8", buffering=1) as out, ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(work, job): job for job in jobs}
        from concurrent.futures import as_completed
        for i, future in enumerate(as_completed(futures), 1):
            row = future.result()
            print(json.dumps(row), file=out)
            if i % 12 == 0 or i == len(jobs):
                print(mode, i, "/", len(jobs), flush=True)


def check_guard(row):
    kind, seed = row["kind"], row["seed"]
    xy, bowl = _seed_scene(seed)
    env = _make_env(xy, kind, .04)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            obs, _ = env.reset(seed=seed, options={"bowl_xy": bowl})
            actor = GraspTrial(env)
            actor.plan = tuple(tuple(p) if isinstance(p, list) else p for p in row["plan"])
            actor._reset_profile(*actor.plan)
            action = actor.guarded_action(row["action"])
            expected = row["action"] if row["result"]["ok"] and row["result"]["held"] else 0
            if action != expected:
                raise ValueError(f"Dự báo và ván PPO khác kết quả ở {kind} {seed}")
            actor.apply_action(action)
            result = rollout(env, actor, obs)
            if action == row["action"] and result != row["result"]:
                raise ValueError(f"Đo lại phương án PPO khác kết quả ở {kind} {seed}")
            return dict(kind=kind, seed=seed, action=action, result=result)
    finally:
        env.close()


def verify_guard(folder, workers):
    rows = [json.loads(line) for line in (folder / "eval.jsonl").read_text(encoding="utf8").splitlines()]
    if len({(r["kind"], r["seed"]) for r in rows}) != 1548:
        raise ValueError("Chấm xong 774 seed cho mỗi vật trước khi kiểm tra phương án dự phòng.")
    rejected = [r for r in rows if r["action"] and not (r["result"]["ok"] and r["result"]["held"])]
    samples = [r for kind in ("box", "sphere") for r in
               [r for r in rows if r["kind"] == kind and r["action"] and r["result"]["ok"] and r["result"]["held"]][:3]]
    with (folder / "guard-check.jsonl").open("w", encoding="utf8", buffering=1) as out:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for row in pool.map(check_guard, rejected + samples):
                print(json.dumps(row), file=out)
                print("guard", row["kind"], row["seed"], row["action"], row["result"]["ok"], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("collect", "fit", "eval", "guard-check"))
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--out", type=Path, default=paths.RESULTS / "replays" / "ppo-grasp4")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.stage == "collect":
        collect(args.out, "train", args.workers)
        collect(args.out, "val", args.workers)
    elif args.stage == "fit":
        fit(args.out)
    elif args.stage == "eval":
        collect(args.out, "eval", args.workers)
    else:
        verify_guard(args.out, args.workers)


if __name__ == "__main__":
    main()
