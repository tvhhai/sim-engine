"""Đường dẫn dùng chung của sim_engine: mọi tool đọc/ghi file qua đây thay vì tự ghép "docs/...".

    docs/            tài liệu (hướng dẫn Markdown, ảnh)
    results/         kết quả sinh ra: quét vùng gắp, bộ seed replay, replay.html + replay.json (hai file sau không commit)
    results/studio/  dữ liệu lúc chạy Studio: lab_case.json, replay_lab.*, studio_runs/ (không commit)
    static/replay/   bản replay canvas: template.html, replay.js, css, studio_bridge.js
    static/studio/   bản build của web/joint-lab (Studio React)
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent
DOCS = ROOT / "docs"
RESULTS = ROOT / "results"
STUDIO_DATA = RESULTS / "studio"
REPLAY_ASSETS = ROOT / "static" / "replay"
STUDIO_BUILD = ROOT / "static" / "studio"

REPLAY_HTML = RESULTS / "replay.html"      # bản canvas mở offline (bấm đúp)
REPLAY_JSON = RESULTS / "replay.json"      # cùng dữ liệu, Studio React đọc file này
REPLAY_SEEDS = RESULTS / "replay_seeds.json"
REPLAY_PREV_STEPS = RESULTS / "replay_prev_steps.json"
PICK_SCAN = RESULTS / "pick_scan.json"
REACH_ENVELOPE = RESULTS / "reach_envelope.json"


def result(name):
    """Đường dẫn một file kết quả trong results/ (vd "pick_scan_outer.json")."""
    return RESULTS / name
