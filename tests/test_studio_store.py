import json
from pathlib import Path
import tempfile
import unittest

from sim_engine.tools.studio_store import StudioStore, render_full_replay
from sim_engine.tools.studio_api import scene_from_source


def payload(steps=1):
    frame = {"jq": [1, 2, 3, 4, 5], "fo": .8, "box": [.15, .03, .435], "bq": [1, 0, 0, 0]}
    return {"scene": {"step_dt": .032}, "episodes": [{"seed": 7, "success": True, "steps": steps,
             "frames": [frame], "bowl": [-.15, 0, .42]}]}


class StudioStoreTests(unittest.TestCase):
    def test_archive_is_immutable_when_latest_is_replaced(self):
        with tempfile.TemporaryDirectory() as directory:
            store = StudioStore(directory)
            case = {"seed": 99, "pose": {"objects": [{"id": "one"}], "pan": 5}}
            store.archive("1795295445070733900", case, payload(10))
            store.archive("1795295445070733901", dict(case, seed=100), payload(20))
            self.assertEqual([record["steps"] for record in store.list_runs()], [20, 10])
            self.assertEqual(store.payload("run:1795295445070733900")["episodes"][0]["steps"], 10)
            self.assertEqual(store.case("1795295445070733900")["pose"]["pan"], 5)
            store.update("1795295445070733900", {"title": "Gắp màu đỏ", "notes": "Ghi chú", "favorite": True})
            record = store.list_runs()[1]
            self.assertEqual((record["title"], record["favorite"]), ("Gắp màu đỏ", True))
            self.assertEqual(store.payload("run:1795295445070733900")["episodes"][0]["steps"], 10)

    def test_baseline_catalog_and_case_do_not_modify_source(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "replay.html"
            contents = "<script>const DATA = " + json.dumps(payload()) + ";const other=42;</script>"
            path.write_text(contents, encoding="utf-8")
            store = StudioStore(directory)
            catalog = store.catalog()
            self.assertEqual(catalog["sources"][0]["id"], "baseline")
            self.assertEqual(catalog["sources"][0]["episodes"][0]["seed"], 7)
            case = scene_from_source(store, "baseline", 0)
            self.assertEqual(case["pose"]["pan"], 1)
            self.assertEqual(case["pose"]["objects"][0]["x"], .15)
            self.assertEqual(path.read_text(encoding="utf-8"), contents)

    def test_source_paths_are_validated(self):
        with tempfile.TemporaryDirectory() as directory:
            store = StudioStore(directory)
            for source in ("../../secret", "run:../../secret", "run:1"):
                with self.subTest(source=source), self.assertRaises(ValueError):
                    store.payload(source)

    def test_existing_latest_is_imported_only_once(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "replay_lab.json").write_text(json.dumps(payload()))
            (root / "lab_case.json").write_text(json.dumps({"seed": 0, "pose": {}}))
            store = StudioStore(directory)
            store.seed_latest()
            store.seed_latest()
            self.assertEqual(len(store.list_runs()), 1)

    def test_full_view_reuses_all_replay_tools_without_cdn(self):
        root = Path(__file__).resolve().parents[1] / "docs"
        html = render_full_replay(payload(), root)
        for item in ("drawTop", "drawSide", "draw3d", "drawCams", "buildTimeline", "buildLog", "drawChk", "studio-replay-command"):
            self.assertIn(item, html)
        self.assertNotIn("cdn.tailwindcss.com", html)
        self.assertNotIn("/*__DATA__*/null", html)


if __name__ == "__main__":
    unittest.main()
