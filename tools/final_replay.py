"""Chấm policy mặc định và lưu đúng một replay cuối cho mỗi seed chuẩn."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import contextlib
import gzip
import io
import json
from pathlib import Path

from .. import config, paths
from ..policies import policy_class
from ..policies.scripted_minjerk import ScriptedPolicyFinal, _FinalTiming
from ..studio.store import atomic_json
from .make_replay import Recorder, pick_zone, replay_scene
from .object_replay import _make_env, _seed_scene

KINDS = ("box", "sphere")


class RecordedFinal(ScriptedPolicyFinal):
    def _choose_profile(self, profiles):
        profile, self.selected_timing = super()._choose_profile(profiles)
        return profile, self.selected_timing


def record(job):
    directory, kind, seed = job
    xy, bowl = _seed_scene(seed)
    env = _make_env(xy, kind, .04)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            actor = RecordedFinal(env)
            episode = Recorder(env, actor).run(seed, options={"bowl_xy": bowl})
        variant = actor.selected_timing
        tags = ["Policy cuối", "Thành công" if episode["success"] else "Hỏng"]
        if isinstance(variant, _FinalTiming):
            if variant.carry:
                tags.append(f"Mang ngang {variant.carry[0]:g}")
            if variant.lower:
                tags.append(f"Hạ điểm thả {variant.lower*1000:g} mm")
            if variant.short_arc:
                tags.append("Vòng cung 82 bước")
        label = "Hộp" if kind == "box" else "Bóng"
        episode.update(name=f"{label} · seed {seed} · {episode['steps']} bước", group="Policy cuối", tags=tags)
        target = directory / f"{kind}4-final" / f"{seed}.json.gz"
        temporary = target.with_suffix(".tmp")
        with gzip.open(temporary, "wt", encoding="utf8", compresslevel=3) as f:
            json.dump(episode, f, ensure_ascii=False, separators=(",", ":"))
        temporary.replace(target)

        frames = episode["frames"][1:]
        phases = Counter(frame["ph"] for frame in frames if frame["ph"] <= 10)
        visits = Counter()
        previous = None
        for frame in frames:
            phase = frame["ph"]
            if phase <= 10 and phase != previous:
                visits[phase] += 1
            previous = phase
        return {
            "kind": kind, "seed": seed, "name": episode["name"], "success": episode["success"],
            "steps": episode["steps"], "control_steps": sum(phases.values()), "tags": tags,
            "zone": pick_zone(*episode["frames"][0]["box"][:2]),
            "revisits": sum(n - 1 for n in visits.values()), "phase6": phases[6],
            "grasped": any(f["gr"] for f in frames),
            "carry_contacts": [sum(bool(f[key]) for f in frames if f["ph"] in (5, 6, 7))
                               for key in ("tb", "cc", "bc")],
        }
    finally:
        env.close()


def seeds():
    values = tuple(sorted(set(range(200)) | {s for s in config.BOX_FIXED_SEEDS if s >= 200} | set(range(2000, 2500))))
    assert len(values) == 774
    return values


def review_contacts(directory, file, rows, workers):
    # Kết quả đầu có thể đã được ghi trước khi bật bộ lọc tiếp xúc. Ván không chạm
    # giữ nguyên lựa chọn, trừ phương án vòng cung có thể sắp lại thứ tự ứng viên.
    jobs = [(directory, r["kind"], r["seed"]) for r in rows if not r.get("checked_contacts")
            and (any(r["carry_contacts"]) or "Vòng cung 82 bước" in r["tags"])]
    if not jobs:
        return rows
    print(f"Kiểm tra lại {len(jobs)} ván có tiếp xúc khi mang hoặc có vòng cung ngắn.", flush=True)
    checked = {}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        tasks = [pool.submit(record, job) for job in jobs]
        for task in as_completed(tasks):
            row = task.result()
            checked[row["kind"], row["seed"]] = {**row, "checked_contacts": True}
            print("Kiểm tra", row["kind"], row["seed"], row["steps"], row["carry_contacts"], row["success"], flush=True)
    raw = file.with_name("full-before-contact-review.jsonl")
    if not raw.exists():
        raw.write_bytes(file.read_bytes())
    reviewed = [checked.get((r["kind"], r["seed"]), r) for r in rows]
    temporary = file.with_suffix(".tmp")
    temporary.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in reviewed), encoding="utf8")
    temporary.replace(file)
    return reviewed


def publish(directory, rows):
    expected = {(kind, seed) for kind in KINDS for seed in seeds()}
    if {(r["kind"], r["seed"]) for r in rows} != expected or len(rows) != len(expected):
        raise ValueError("Thiếu ván cuối trong bộ chuẩn.")
    if not all(row["success"] and row["grasped"] for row in rows):
        raise ValueError("Còn ván hỏng hoặc không giữ vật, cần điều tra trước khi công bố Replay cuối.")
    for kind in KINDS:
        folder = directory / f"{kind}4-final"
        episodes = sorted((r for r in rows if r["kind"] == kind), key=lambda r: r["seed"])
        if not all((folder / f"{r['seed']}.json.gz").is_file() for r in episodes):
            raise ValueError(f"Thiếu file replay {kind}")
        env = _make_env([.15, 0], kind, .04)
        try:
            scene = replay_scene(env, config.DEFAULT_POLICY)
        finally:
            env.close()
        scene["title"] = ("Hộp" if kind == "box" else "Bóng") + " 4 cm · Policy cuối · 774 seed"
        metadata = [dict({key: r[key] for key in ("seed", "name", "success", "steps", "tags", "zone")},
                         group="Policy cuối") for r in episodes]
        atomic_json(directory / f"{kind}4-final.json", {"scene": scene, "episode_files": True, "episodes": metadata})


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--sample", nargs="+", metavar="LOẠI:SEED")
    ap.add_argument("--publish", action="store_true")
    args = ap.parse_args()
    if policy_class() is not ScriptedPolicyFinal:
        raise ValueError("Không xác định được policy mặc định.")
    directory = paths.RESULTS / "replays"
    if args.sample:
        directory /= "final-study"
    directory.mkdir(parents=True, exist_ok=True)
    for kind in KINDS:
        (directory / f"{kind}4-final").mkdir(exist_ok=True)
    file = directory / "final4-audit.jsonl" if args.sample else directory / "final-study" / "full.jsonl"
    file.parent.mkdir(parents=True, exist_ok=True)
    previous = [json.loads(line) for line in file.read_text(encoding="utf8").splitlines()] if file.exists() else []
    if args.publish:
        publish(directory, review_contacts(directory, file, previous, args.workers))
        return
    focus = (229, 230, 1014, 2026, 2087, 2286, 2379, 1016, 133, 1000, 55, 2047)
    ordered = (*focus, *(seed for seed in seeds() if seed not in focus))
    cases = [(part.split(":")[0], int(part.split(":")[1])) for part in args.sample] if args.sample else [
        (kind, seed) for seed in ordered for kind in KINDS
    ]
    completed = {(r["kind"], r["seed"]) for r in previous}
    jobs = [(directory, kind, seed) for kind, seed in cases if (kind, seed) not in completed]
    print(f"Đã có {len(completed)} ván, chạy {len(jobs)} ván.", flush=True)
    with file.open("a", encoding="utf8", buffering=1) as out, ProcessPoolExecutor(max_workers=args.workers) as pool:
        tasks = [pool.submit(record, job) for job in jobs]
        for index, task in enumerate(as_completed(tasks), 1):
            row = task.result()
            print(json.dumps(row, ensure_ascii=False), file=out)
            if args.sample or index % 20 == 0 or index == len(jobs) or not row["success"]:
                print(index, "/", len(jobs), row["kind"], row["seed"], row["steps"], row["success"], row["tags"], flush=True)


if __name__ == "__main__":
    main()
