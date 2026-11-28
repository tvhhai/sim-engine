"""Local run archive and replay adapters. Reading a replay never runs physics."""

from datetime import datetime, timezone
import gzip
import json
import os
from pathlib import Path
import re
import shutil
import threading

from .. import paths
from ..tools.make_replay import pick_zone


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    os.replace(temporary, path)


class StudioStore:
    def __init__(self, directory, baseline=None):
        self.directory = Path(directory)
        # Bộ replay chung (make_replay): đọc replay.json; chỉ có replay.html (bản cũ) thì bóc dữ liệu từ HTML
        self.baseline = Path(baseline) if baseline else paths.REPLAY_JSON
        self.runs = self.directory / "studio_runs"
        self.lock = threading.RLock()
        self.cache = None
        self.file_cache = {}

    def _run_dir(self, run_id):
        if not re.fullmatch(r"[0-9]{15,25}", str(run_id)):
            raise ValueError("ID lần chạy không hợp lệ")
        return self.runs / str(run_id)

    def archive(self, run_id, case, payload, created_at=None):
        episode = payload["episodes"][0]
        record = {
            "id": str(run_id), "title": "Lab · " + str(run_id)[-6:],
            "created_at": created_at or datetime.now(timezone.utc).isoformat(),
            "success": episode["success"], "steps": episode["steps"],
            "blocked_reason": episode.get("blocked_reason"),
            "objects": len(case["pose"].get("objects", [])) or 1,
            "seed": case["seed"], "favorite": False, "notes": "",
            "duration": round(episode["steps"] * payload["scene"].get("step_dt", 1 / 30), 3),
        }
        with self.lock:
            folder = self._run_dir(run_id)
            atomic_json(folder / "case.json", case)
            atomic_json(folder / "payload.json", payload)
            # Publish the metadata last so listings never expose a partial archive.
            atomic_json(folder / "meta.json", record)
        return record

    def delete(self, run_id):
        """Xoá hẳn một lần chạy đã lưu (cả cảnh, dữ liệu replay và ghi chú)."""
        with self.lock:
            folder = self._run_dir(run_id)
            if not (folder / "meta.json").is_file():
                raise ValueError("Không có lần chạy này")
            shutil.rmtree(folder)

    def delete_lab(self):
        """Xoá replay của lần chạy Lab gần nhất (replay_lab.json/.html). Cảnh đã đặt (lab_case.json) vẫn giữ để Lab mở lại đúng cảnh."""
        with self.lock:
            removed = False
            for name in ("replay_lab.json", "replay_lab.html"):
                path = self.directory / name
                if path.is_file():
                    path.unlink()
                    removed = True
            if not removed:
                raise ValueError("Chưa có replay Lab để xoá")

    def list_runs(self):
        with self.lock:
            records = []
            if self.runs.is_dir():
                for path in self.runs.glob("*/meta.json"):
                    try:
                        records.append(json.loads(path.read_text(encoding="utf-8")))
                    except (OSError, ValueError):
                        continue
            return sorted(records, key=lambda record: record["id"], reverse=True)

    def update(self, run_id, changes):
        with self.lock:
            path = self._run_dir(run_id) / "meta.json"
            record = json.loads(path.read_text(encoding="utf-8"))
            if "title" in changes:
                title = str(changes["title"]).strip()
                if not 1 <= len(title) <= 120:
                    raise ValueError("Tên phải có từ 1 đến 120 ký tự")
                record["title"] = title
            if "notes" in changes:
                record["notes"] = str(changes["notes"])[:4000]
            if "favorite" in changes:
                if not isinstance(changes["favorite"], bool):
                    raise ValueError("favorite phải là boolean")
                record["favorite"] = changes["favorite"]
            atomic_json(path, record)
            return record

    def case(self, run_id):
        return json.loads((self._run_dir(run_id) / "case.json").read_text(encoding="utf-8"))

    def _has_final_replays(self):
        replay_dir = paths.RESULTS / "replays"
        return self.baseline == paths.REPLAY_JSON and all(
            (replay_dir / name).is_file() for name in ("box4-final.json", "sphere4-final.json")
        )

    def payload(self, source):
        if source == "baseline":
            if self._has_final_replays():
                return self.payload("file:box4-final")
            path = self.baseline
            if path.suffix == ".json" and not path.is_file() and path.with_suffix(".html").is_file():
                path = path.with_suffix(".html")
            signature = (path, path.stat().st_mtime_ns, path.stat().st_size)
            with self.lock:
                if self.cache is None or self.cache[0] != signature:
                    text = path.read_text(encoding="utf-8")
                    if path.suffix == ".json":
                        payload = json.loads(text)
                    else:
                        marker = "const DATA = "
                        if marker not in text:
                            raise ValueError("replay.html không chứa dữ liệu replay")
                        payload = json.JSONDecoder().raw_decode(text.split(marker, 1)[1].lstrip())[0]
                    self.cache = (signature, payload)
                return self.cache[1]
        if source.startswith("file:"):                        # results/replays/<tên>.json do tools/object_replay (bóng, vật cỡ khác...) ghi
            name = source[5:]
            if not re.fullmatch(r"[A-Za-z0-9_-]{1,60}", name):
                raise ValueError("Nguồn replay không hợp lệ")
            path = paths.RESULTS / "replays" / f"{name}.json"
            signature = (path, path.stat().st_mtime_ns, path.stat().st_size)
            with self.lock:
                cached = self.file_cache.get(name)
                if cached is None or cached[0] != signature:
                    cached = (signature, json.loads(path.read_text(encoding="utf-8")))
                    self.file_cache[name] = cached
                return cached[1]
        if source == "lab":
            return json.loads((self.directory / "replay_lab.json").read_text(encoding="utf-8"))
        if source.startswith("run:"):
            return json.loads((self._run_dir(source[4:]) / "payload.json").read_text(encoding="utf-8"))
        raise ValueError("Nguồn replay không hợp lệ")

    @staticmethod
    def episode(source, payload, index):
        """Đọc một ván đã lưu. Bộ lớn chỉ giữ metadata trong catalog."""
        if not 0 <= index < len(payload["episodes"]):
            raise ValueError("Episode không tồn tại")
        entry = payload["episodes"][index]
        if not payload.get("episode_files"):
            return entry
        name = "box4-final" if source == "baseline" else source.removeprefix("file:")
        seed = entry["seed"]
        if (source != "baseline" and not source.startswith("file:")) or not re.fullmatch(r"[A-Za-z0-9_-]{1,60}", name) or type(seed) is not int:
            raise ValueError("Đường dẫn ván replay không hợp lệ")
        file = paths.RESULTS / "replays" / name / f"{seed}.json.gz"
        with gzip.open(file, "rt", encoding="utf-8") as handle:
            episode = json.load(handle)
        if episode["seed"] != seed:
            raise ValueError("Seed replay khác danh sách")
        return episode

    def catalog(self):
        sources = []
        extra = [(f"file:{p.stem}", None) for p in sorted((paths.RESULTS / "replays").glob("*.json"))] if (paths.RESULTS / "replays").is_dir() else []
        final = self._has_final_replays()
        if final:
            history = {f"file:{kind}4{suffix}" for kind in ("box", "sphere")
                       for suffix in ("", "-history", "-pilot5", "-ppo-review")}
            # Chỉ hiện bộ cuối trong danh sách thầy. Link replay cũ vẫn đọc được.
            extra = [item for item in extra if item[0] not in history and item[0] != "file:box4-final"]
        extra.sort(key=lambda item: not item[0].endswith("4-final"))
        for ident, title in (("lab", "Lab · lần chạy mới nhất"), ("baseline", None if final else "Bộ replay hiện tại"), *extra):
            try:
                payload = self.payload(ident)
                title = title or payload.get("scene", {}).get("title") or ident[5:]
                sources.append({"id": ident, "title": title, "episodes": [
                    {"index": i, "seed": e["seed"], "name": e.get("name", f"seed {e['seed']}"),
                     "success": e["success"], "steps": e["steps"], "tags": e.get("tags", []),
                     "blocked_reason": e.get("blocked_reason"),
                     "zone": e.get("zone") if payload.get("episode_files") else (
                         pick_zone(*e["frames"][0]["box"][:2]) if e.get("frames") else None)}
                    for i, e in enumerate(payload["episodes"])]})
            except (OSError, ValueError, KeyError):
                continue
        return {"sources": sources, "runs": self.list_runs()}


def render_full_replay(payload, assets_directory):
    """Reuse the real replay UI, including cameras, diagnostics and phase logs."""
    directory = Path(assets_directory)
    template = (directory / "template.html").read_text(encoding="utf-8")
    # The replay stylesheet is complete; the optional CDN is unnecessary in Studio.
    template = re.sub(r'<script src="https://cdn\.tailwindcss\.com/[^\"]+"></script>', "", template)
    template = re.sub(r'<script>tailwind\.config.*?</script>', "", template)
    template = template.replace("/*__REPLAY_JS__*/", "window.STUDIO_WEBGL = true;\n/*__REPLAY_JS__*/")
    for marker, name in (("/*__SHARED_CSS__*/", "accessibility.css"),
                         ("/*__REPLAY_CSS__*/", "replay.css"), ("/*__REPLAY_JS__*/", "replay.js")):
        template = template.replace(marker, (directory / name).read_text(encoding="utf-8"))
    data = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c")
    template = template.replace("/*__DATA__*/null", data)
    bridge = (directory / "studio_bridge.js").read_text(encoding="utf-8")
    return template.replace("</body>", "<script>" + bridge + "</script></body>")
