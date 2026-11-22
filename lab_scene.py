"""Validated Lab scene definitions shared by the editor, MuJoCo and replay."""

import copy
import math
from pathlib import Path
import re
import xml.etree.ElementTree as ET

import mujoco

from . import config


def in_workspace(xy):
    dx, dy = xy[0] - config.PICK_PAN_XY[0], xy[1] - config.PICK_PAN_XY[1]
    radius = math.hypot(dx, dy)
    angle = math.degrees(math.atan2(dx, dy))
    region = config.PICK_REGION
    return (region["r_min"] - 1e-8 <= radius <= region["r_max"] + 1e-8
            and region["angle_min_deg"] - 1e-8 <= angle <= region["angle_max_deg"] + 1e-8)


def normalize_objects(objects, target_id=None):
    if not isinstance(objects, list) or not 1 <= len(objects) <= 5:
        raise ValueError("Cảnh phải có từ 1 đến 5 vật thể")
    result, ids = [], set()
    for i, obj in enumerate(objects):
        if not isinstance(obj, dict):
            raise ValueError("Vật thể không hợp lệ")
        ident = str(obj.get("id", f"object-{i + 1}"))
        if not re.fullmatch(r"[a-zA-Z0-9_-]{1,40}", ident) or ident in ids:
            raise ValueError("ID vật thể phải duy nhất")
        ids.add(ident)
        kind, color = obj.get("type", "box"), obj.get("color", "#e85d55")
        if kind not in ("box", "sphere") or not re.fullmatch(r"#[0-9a-fA-F]{6}", str(color)):
            raise ValueError("Loại hoặc màu vật thể không hợp lệ")
        values = {key: float(obj.get(key, 0)) for key in ("x", "y", "yaw")}
        if not all(math.isfinite(v) for v in values.values()):
            raise ValueError("Tọa độ và góc phải hữu hạn")
        if abs(values["x"]) > config.TABLE_HALF[0] or abs(values["y"]) > config.TABLE_HALF[1]:
            raise ValueError("Vật thể phải nằm trong mặt bàn")
        result.append(dict(id=ident, type=kind, color=color, **values))
    target_id = target_id or result[0]["id"]
    if target_id not in ids:
        raise ValueError("Vật cần gắp không tồn tại")
    for i, obj in enumerate(result):
        obj["body"] = "box" if obj["id"] == target_id else f"lab_object_{i}"
    return result


def scene_model(objects):
    """Keep the calibrated robot and contact parameters; add real free bodies."""
    source = Path(config.SCENE_XML)
    root = ET.parse(source).getroot()
    compiler = root.find("compiler")
    compiler.set("meshdir", str((source.parent / compiler.get("meshdir", ".")).resolve()))
    world = root.find("worldbody")
    original = world.find("body[@name='box']")
    world.remove(original)
    for obj in objects:
        body = copy.deepcopy(original)
        body.set("name", obj["body"])
        body.set("pos", f"{obj['x']} {obj['y']} {0.42 + config.BOX_HALF}")
        angle = math.radians(obj["yaw"]) / 2
        body.set("quat", f"{math.cos(angle)} 0 0 {math.sin(angle)}")
        body.find("freejoint").set("name", "box_joint" if obj["body"] == "box" else obj["body"] + "_joint")
        geom = body.find("geom")
        geom.set("name", "box_geom" if obj["body"] == "box" else obj["body"] + "_geom")
        geom.set("type", obj["type"])
        geom.set("size", "0.015" if obj["type"] == "sphere" else "0.015 0.015 0.015")
        geom.attrib.pop("material", None)
        rgb = [int(obj["color"][k:k + 2], 16) / 255 for k in (1, 3, 5)]
        geom.set("rgba", " ".join(map(str, [*rgb, 1])))
        world.append(body)
    return mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
