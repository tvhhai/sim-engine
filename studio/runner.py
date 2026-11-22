"""Single-case asynchronous Joint Lab rollout, isolated from batch replays."""

import json
import math
import os
from pathlib import Path
import threading
import time

from .. import paths
from ..env import SO101PickPlaceEnv
from ..policies.scripted_policy import ScriptedPolicy
from ..tools.make_replay import Recorder, write_replay
from .scene import in_workspace, normalize_objects
from .store import StudioStore

LAB_SEED = 2147483647


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
        pose = {key: float(pose[key]) for key in keys}
        if not all(math.isfinite(value) for value in pose.values()):
            raise ValueError("Tọa độ và góc phải là số hữu hạn")
        if any(abs(pose[key]) > 0.45 for key in ("bx", "by", "wx", "wy")):
            raise ValueError("Hộp và bát phải nằm trong hệ tọa độ ±45 cm")
        if objects is not None:
            pose["objects"] = normalize_objects(objects, target_id)
            target = next(obj for obj in pose["objects"] if obj["body"] == "box")
            pose.update(target_id=target["id"], bx=target["x"], by=target["y"], byaw=target["yaw"])
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
            elif not in_workspace(options["box_xy"]):
                blocked = "Vật cần gắp ngoài vùng thao tác: giữ nguyên cánh tay"
            if blocked:
                options.update(blocked_reason=blocked, skip_settle=True)
            case = {"name": "lab", "seed": LAB_SEED, "pose": pose, "options": options}
            case_path = self.directory / "lab_case.json"
            case_tmp = case_path.with_suffix(".tmp")
            case_tmp.write_text(json.dumps(case, ensure_ascii=False, indent=2), encoding="utf-8")
            os.replace(case_tmp, case_path)
            env = SO101PickPlaceEnv(render_images=False, end_on_success=False,
                                    lab_objects=pose.get("objects"), target_id=pose.get("target_id"))
            episode = Recorder(env, ScriptedPolicy(env)).run(LAB_SEED, options=options)
            episode.update(name="lab", group="mixed", tags=["LAB · lần chạy mới nhất"])
            episode["initial_options"] = options
            out = self.directory / "replay_lab.html"
            temporary = out.with_suffix(".tmp")
            payload = write_replay(env, [episode], str(temporary), "sim_engine.policies.scripted_policy:ScriptedPolicy")
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
