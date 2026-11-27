"""Single-case asynchronous Joint Lab rollout, isolated from batch replays."""

import json
import math
import os
from pathlib import Path
import threading
import time

from .. import paths
from ..env import SO101PickPlaceEnv
from .. import config
from ..policies import default_policy
from ..tools.make_replay import Recorder, bowl_geom_list, write_replay
from .scene import check_bowl_capacity, check_bowl_fits, in_workspace, normalize_bowl, normalize_objects
from .store import StudioStore

LAB_SEED = 2147483647


def run_all(env, options):
    """Chạy từng vật trên một mô phỏng liên tục và chấm lại mọi vật ở khung cuối."""
    env.reset(seed=LAB_SEED, options=options)
    objects = sorted(env.lab_objects, key=lambda obj: obj["body"] != "box")
    frames, outcomes = [], {}
    for index, obj in enumerate(objects):
        ident = obj["id"]
        env.select_lab_target(obj)
        info = env._get_info()
        if info["success"]:
            outcomes[ident] = "placed"
            continue
        if not in_workspace(info["box_pos"][:2]):
            outcomes[ident] = "unreachable"
            continue
        attempt = Recorder(env, default_policy(env)).run(LAB_SEED, continue_scene=True)
        offset = len(frames) - 1 if frames else 0
        for frame in attempt["frames"][1 if frames else 0:]:
            frames.append(dict(frame, t=offset + frame["t"], target_id=ident, target_index=index + 1))
        outcomes[ident] = "placed" if attempt["success"] else "failed"
        if env._grasped:
            # Dừng để lượt kế tiếp không mở hàm khi tay còn giữ vật của lượt trước.
            break
    placements = []
    for obj in objects:
        env.select_lab_target(obj)
        ident = obj["id"]
        status = outcomes.get(ident, "not_attempted")
        if env._get_info()["success"]:
            status = "placed"
        elif status == "placed":
            status = "displaced"
        placements.append({"id": ident, "status": status})
    if not frames:
        env.select_lab_target(objects[0])
        rec = Recorder(env, default_policy(env))
        frames = [rec.frame(0, 0, 0, 0, env._get_info(), 1.0)]
    return {"seed": LAB_SEED, "bowl": env.get_bowl_pos().tolist(), "bowl_geoms": bowl_geom_list(env), "frames": frames,
            "success": all(result["status"] == "placed" for result in placements),
            "steps": len(frames) - 1, "placements": placements, "mode": "all"}


class LabRun:
    def __init__(self, directory=None):
        self.directory = Path(directory) if directory else paths.STUDIO_DATA
        self.lock = threading.Lock()
        self.state = {"status": "idle"}
        self.store = StudioStore(self.directory)
        try:
            self.saved_pose = json.loads((self.directory / "lab_case.json").read_text(encoding="utf-8"))["pose"]
        except (OSError, ValueError, KeyError):
            self.saved_pose = None

    def snapshot(self):
        with self.lock:
            return dict(self.state)

    def start(self, pose):
        if not isinstance(pose, dict):
            raise ValueError("Cảnh phải là một JSON object")
        keys = ("pan", "lift", "elbow", "wflex", "wroll", "grip", "bx", "by", "byaw", "wx", "wy")
        objects = pose.get("objects")
        target_id = pose.get("target_id")
        bowl = pose.get("bowl")
        mode = pose.get("mode", "single")
        if mode not in ("single", "all") or (mode == "all" and objects is None):
            raise ValueError("Chế độ gắp không hợp lệ")
        pose = {key: float(pose[key]) for key in keys}
        if not all(math.isfinite(value) for value in pose.values()):
            raise ValueError("Tọa độ và góc phải là số hữu hạn")
        if any(abs(pose[key]) > 0.45 for key in ("bx", "by", "wx", "wy")):
            raise ValueError("Hộp và bát phải nằm trong hệ tọa độ ±45 cm")
        if objects is not None:
            pose["objects"] = normalize_objects(objects, target_id)
            pose["bowl"] = normalize_bowl(bowl)
            validate_bowl = check_bowl_capacity if mode == "all" else check_bowl_fits
            validate_bowl(pose["objects"], pose["bowl"])
            target = next(obj for obj in pose["objects"] if obj["body"] == "box")
            pose.update(target_id=target["id"], bx=target["x"], by=target["y"], byaw=target["yaw"])
        if mode == "all":
            pose["mode"] = mode
        with self.lock:
            if self.state["status"] == "running":
                return dict(self.state)
            self.state = {"status": "running", "run_id": str(time.time_ns())}
            self.saved_pose = pose
            threading.Thread(target=self._run, args=(pose,), daemon=True).start()
            return dict(self.state)

    def _run(self, pose):
        env = None
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            options = {
                "box_xy": [pose["bx"], pose["by"]],
                "bowl_xy": [pose["wx"], pose["wy"]],
                "box_yaw": math.radians(pose["byaw"]),
                "arm_qpos": [math.radians(pose[k]) for k in ("pan", "lift", "elbow", "wflex", "wroll")],
                "gripper_qpos": pose["grip"],
            }
            blocked = None
            if not in_workspace(options["bowl_xy"]):
                blocked = "Bát ngoài vùng thao tác: giữ nguyên cánh tay"
            elif pose.get("mode") != "all" and not in_workspace(options["box_xy"]):
                blocked = "Vật cần gắp ngoài vùng thao tác: giữ nguyên cánh tay"
            if blocked:
                options.update(blocked_reason=blocked, skip_settle=True)
            case = {"name": "lab", "seed": LAB_SEED, "pose": pose, "options": options}
            case_path = self.directory / "lab_case.json"
            case_tmp = case_path.with_suffix(".tmp")
            case_tmp.write_text(json.dumps(case, ensure_ascii=False, indent=2), encoding="utf-8")
            os.replace(case_tmp, case_path)
            env = SO101PickPlaceEnv(render_images=False, end_on_success=False,
                                    lab_objects=pose.get("objects"), target_id=pose.get("target_id"),
                                    lab_bowl=pose.get("bowl"))
            all_objects = pose.get("mode") == "all"
            episode = run_all(env, options) if all_objects and not blocked else Recorder(env, default_policy(env)).run(LAB_SEED, options=options)
            episode.update(name="lab", group="mixed", tags=["LAB · lần chạy mới nhất"])
            if "placements" in episode:
                placed = sum(result["status"] == "placed" for result in episode["placements"])
                episode["tags"].append(f"Gắp tất cả · {placed}/{len(episode['placements'])} vật trong bát")
            episode["initial_options"] = options
            out = self.directory / "replay_lab.html"
            temporary = out.with_suffix(".tmp")
            payload = write_replay(env, [episode], str(temporary), config.DEFAULT_POLICY)
            data_path = self.directory / "replay_lab.json"
            data_tmp = data_path.with_suffix(".json.tmp")
            data_tmp.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
            os.replace(data_tmp, data_path)
            os.replace(temporary, out)
            with self.lock:
                self.state.update(status="done", success=episode["success"], steps=episode["steps"],
                                  blocked_reason=blocked,
                                  replay="/replay-lab?run=" + self.state["run_id"])
        except Exception as exc:
            with self.lock:
                self.state.update(status="error", error=f"{type(exc).__name__}: {exc}")
        finally:
            if env is not None:
                env.close()
