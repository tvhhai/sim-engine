"""Validated Lab scene definitions shared by the editor, MuJoCo and replay."""

import copy
import math
from pathlib import Path
import re
import xml.etree.ElementTree as ET

import mujoco

from .. import config


def in_workspace(xy):
    dx, dy = xy[0] - config.PICK_PAN_XY[0], xy[1] - config.PICK_PAN_XY[1]
    radius = math.hypot(dx, dy)
    angle = math.degrees(math.atan2(dx, dy))
    region = config.PICK_GREEN
    return (region["r_min"] - 1e-8 <= radius <= region["r_max"] + 1e-8
            and region["angle_min_deg"] - 1e-8 <= angle <= region["angle_max_deg"] + 1e-8)


DEFAULT_SIZE, MIN_SIZE, MAX_SIZE = 0.03, 0.02, 0.06          # cạnh hộp / đường kính bóng (m)
BOWL_WALL = 0.003                                             # nửa độ dày tường bát (m)
DEFAULT_BOWL = {"color": "#3366d9", "radius": 0.045, "height": 0.04}   # radius = bán kính tới TÂM tường; lòng bát = radius - BOWL_WALL


def normalize_bowl(bowl=None):
    """Cấu hình bát của Lab: màu, bán kính (3-9 cm) và chiều cao tường (2-8 cm). None = bát mặc định."""
    cfg = dict(DEFAULT_BOWL)
    if bowl:
        if not isinstance(bowl, dict):
            raise ValueError("Cấu hình bát không hợp lệ")
        cfg.update({k: bowl[k] for k in cfg if k in bowl})
    cfg["radius"], cfg["height"] = float(cfg["radius"]), float(cfg["height"])
    if not re.fullmatch(r"#[0-9a-fA-F]{6}", str(cfg["color"])):
        raise ValueError("Màu bát không hợp lệ")
    if not (0.03 <= cfg["radius"] <= 0.09 and 0.02 <= cfg["height"] <= 0.08):
        raise ValueError("Bát: bán kính 3-9 cm, chiều cao 2-8 cm")
    return cfg


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
        values["size"] = float(obj.get("size", DEFAULT_SIZE))          # hộp: cạnh; bóng: đường kính (m)
        if not MIN_SIZE <= values["size"] <= MAX_SIZE:
            raise ValueError(f"Kích thước vật phải từ {MIN_SIZE * 100:.0f} đến {MAX_SIZE * 100:.0f} cm")
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


def scene_model(objects, bowl=None):
    """Keep the calibrated robot and contact parameters; add real free bodies and an optionally resized / recoloured bowl."""
    bowl = normalize_bowl(bowl)
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
        half_size = obj["size"] / 2
        body.set("pos", f"{obj['x']} {obj['y']} {0.42 + half_size}")
        angle = math.radians(obj["yaw"]) / 2
        body.set("quat", f"{math.cos(angle)} 0 0 {math.sin(angle)}")
        body.find("freejoint").set("name", "box_joint" if obj["body"] == "box" else obj["body"] + "_joint")
        geom = body.find("geom")
        geom.set("name", "box_geom" if obj["body"] == "box" else obj["body"] + "_geom")
        geom.set("type", obj["type"])
        geom.set("size", f"{half_size}" if obj["type"] == "sphere" else f"{half_size} {half_size} {half_size}")
        geom.attrib.pop("material", None)
        rgb = [int(obj["color"][k:k + 2], 16) / 255 for k in (1, 3, 5)]
        geom.set("rgba", " ".join(map(str, [*rgb, 1])))
        world.append(body)
    _resize_bowl(root, bowl)
    return mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))


def _resize_bowl(root, bowl, segments=24):
    """Dựng lại bát: đáy trụ bán kính radius và 24 tấm tường hẹp xếp vòng tròn cao `height`, đổi màu vật liệu bowl_mat."""
    body = root.find("worldbody").find("body[@name='bowl']")
    for geom in [g for g in body.findall("geom") if g.get("name", "").startswith("bowl_wall_")]:
        body.remove(geom)
    base = body.find("geom[@name='bowl_base']")
    r, h = bowl["radius"], bowl["height"]
    base.set("size", f"{r} 0.003")
    half_len = 1.34 * math.pi * r / segments                    # chồng nhẹ lên nhau như bát gốc để không hở khe
    for k in range(segments):
        a = 2 * math.pi * k / segments
        ET.SubElement(body, "geom", {
            "name": f"bowl_wall_{k:02d}", "type": "box", "size": f"{BOWL_WALL} {half_len:.5f} {h / 2:.5f}",
            "pos": f"{r * math.cos(a):.5f} {r * math.sin(a):.5f} {h / 2:.5f}", "euler": f"0 0 {a:.5f}",
            "material": "bowl_mat", "contype": "1", "conaffinity": "1"})
    rgb = [int(bowl["color"][i:i + 2], 16) / 255 for i in (1, 3, 5)]
    root.find("asset").find("material[@name='bowl_mat']").set("rgba", " ".join(f"{v:.4f}" for v in [*rgb, 1]))
