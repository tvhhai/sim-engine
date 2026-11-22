"""Studio screen APIs kept separate from the legacy Lab request handler."""

import json
import math
from pathlib import Path

import mujoco

from .. import config
from .studio_store import render_full_replay


def scene_from_source(store, source, index=0):
    if source.startswith("run:"):
        return store.case(source[4:])
    if source == "lab":
        return json.loads((store.directory / "lab_case.json").read_text(encoding="utf-8"))
    payload = store.payload(source)
    episode = payload["episodes"][index]
    frame = episode["frames"][0]
    keys = ("pan", "lift", "elbow", "wflex", "wroll")
    q = frame.get("bq", [1, 0, 0, 0])
    yaw = math.degrees(math.atan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2)))
    objects = frame.get("objects") or [{"id": "object-1", "type": "box", "color": "#ed8067",
                                        "x": frame["box"][0], "y": frame["box"][1], "yaw": yaw}]
    target = next((obj for obj in objects if obj.get("body") == "box"), objects[0])
    pose = dict(zip(keys, frame["jq"]), grip=frame["fo"], bx=target["x"], by=target["y"],
                byaw=target["yaw"], wx=episode["bowl"][0], wy=episode["bowl"][1],
                objects=objects, target_id=target["id"])
    return {"name": episode.get("name", f"seed {episode['seed']}"), "seed": episode["seed"], "pose": pose}


def handle_get(handler, lab, path, query):
    if not path.startswith("/studio/"):
        return False
    store = lab.runner.store
    try:
        source = query.get("source", "lab")
        if path == "/studio/catalog":
            result = store.catalog()
        elif path == "/studio/runs":
            result = {"runs": store.list_runs()}
        elif path == "/studio/case":
            result = scene_from_source(store, source, int(query.get("episode", "0")))
        elif path in ("/studio/replay-data", "/studio/replay-view"):
            payload = store.payload(source)
            if path.endswith("replay-view"):
                handler._send(render_full_replay(payload, Path(__file__).resolve().parents[1] / "docs"), "text/html")
                return True
            if "episode" in query:
                index = int(query["episode"])
                if not 0 <= index < len(payload["episodes"]):
                    raise ValueError("Episode không tồn tại")
                result = {"scene": payload["scene"], "episodes": [payload["episodes"][index]]}
            else:
                result = payload
        elif path == "/studio/health":
            result = {"status": "ok", "mujoco": mujoco.__version__, "control_hz": config.CONTROL_FREQ,
                      "physics_dt": config.PHYSICS_TIMESTEP, "max_steps": config.MAX_EPISODE_STEPS,
                      "scene": Path(config.SCENE_XML).name, "run": lab.runner.snapshot(),
                      "workspace": dict(config.PICK_REGION, pan=list(config.PICK_PAN_XY)),
                      "capabilities": ["lab", "full-replay", "multi-object", "run-history", "scene-import-export"]}
        else:
            handler.send_error(404)
            return True
        handler._send(json.dumps(result, ensure_ascii=False))
    except (OSError, ValueError, KeyError, IndexError) as exc:
        handler._send(json.dumps({"error": str(exc)}, ensure_ascii=False))
    return True


def handle_post(handler, lab):
    if not handler.path.startswith("/studio/"):
        return False
    if handler.headers.get("Content-Type", "").split(";")[0] != "application/json":
        handler.send_error(415)
        return True
    try:
        parts = handler.path.split("/")
        if len(parts) != 5 or parts[1:3] != ["studio", "runs"] or parts[4] != "meta":
            handler.send_error(404)
            return True
        length = int(handler.headers.get("Content-Length", "0"))
        if not 0 < length <= 8192:
            raise ValueError("Invalid request size")
        changes = json.loads(handler.rfile.read(length))
        if not isinstance(changes, dict):
            raise ValueError("Metadata phải là object")
        result = lab.runner.store.update(parts[3], changes)
        handler._send(json.dumps(result, ensure_ascii=False))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        handler._send(json.dumps({"error": str(exc)}, ensure_ascii=False))
    return True
