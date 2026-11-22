"""Integration checks for one-case rollouts, using real MuJoCo and replay export."""

import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from sim_engine import config
from sim_engine.studio.runner import LAB_SEED, LabRun
from sim_engine.tools.make_replay import Recorder


def pose():
    angles = [float(v * 180 / 3.141592653589793) for v in config.HOME_QPOS_RAD[:5]]
    return dict(zip(("pan", "lift", "elbow", "wflex", "wroll"), angles),
                grip=config.GRIPPER_OPEN_RAD, bx=.15, by=0, byaw=0, wx=-.15, wy=0)


def wait(runner):
    deadline = time.monotonic() + 30
    while runner.snapshot()["status"] == "running" and time.monotonic() < deadline:
        time.sleep(.02)
    return runner.snapshot()


def payload(path):
    text = path.read_text(encoding="utf-8")
    start = text.index('const DATA = ') + len('const DATA = ')
    return json.JSONDecoder().raw_decode(text[start:])[0]


class LabRunTests(unittest.TestCase):
    def test_replaces_only_lab_and_restores_submitted_pose(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("replay.html", "replay_seeds.json", "replay_prev_steps.json"):
                (root / name).write_text("untouched", encoding="utf-8")
            runner = LabRun(root)
            first = runner.start(pose())
            self.assertEqual(wait(runner)["status"], "done")
            second_pose = dict(pose(), bx=.18, by=.03, wy=-.03, byaw=30, pan=15)
            second = runner.start(second_pose)
            self.assertNotEqual(first["run_id"], second["run_id"])
            result = wait(runner)
            self.assertEqual(result["status"], "done", result)
            data = payload(root / "replay_lab.html")
            self.assertEqual(len(data["episodes"]), 1)
            episode = data["episodes"][0]
            self.assertEqual((episode["name"], episode["seed"]), ("lab", LAB_SEED))
            self.assertEqual(episode["initial_options"]["box_xy"], [.18, .03])
            self.assertEqual(episode["initial_options"]["bowl_xy"], [-.15, -.03])
            self.assertAlmostEqual(episode["initial_options"]["arm_qpos"][0], .2617993878)
            self.assertAlmostEqual(episode["frames"][0]["box"][0], .18, delta=.003)
            self.assertEqual(LabRun(root).saved_pose, second_pose)
            for name in ("replay.html", "replay_seeds.json", "replay_prev_steps.json"):
                self.assertEqual((root / name).read_text(), "untouched")

    def test_duplicate_start_is_not_queued(self):
        entered, release = threading.Event(), threading.Event()
        real_run = Recorder.run

        def blocked(recorder, *args, **kwargs):
            entered.set()
            release.wait(10)
            return real_run(recorder, *args, **kwargs)

        with tempfile.TemporaryDirectory() as directory, patch.object(Recorder, "run", blocked):
            runner = LabRun(directory)
            first = runner.start(pose())
            try:
                self.assertTrue(entered.wait(10))
                self.assertEqual(runner.start(dict(pose(), bx=.2))["run_id"], first["run_id"])
            finally:
                release.set()
            self.assertEqual(wait(runner)["status"], "done")

    def test_errors_leave_previous_replay_intact_and_allow_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "replay_lab.html"
            out.write_text("previous replay")
            runner = LabRun(directory)
            with self.assertRaises(ValueError):
                runner.start(dict(pose(), bx=float("nan")))
            with patch.object(Recorder, "run", side_effect=RuntimeError("test failure")):
                runner.start(pose())
                self.assertEqual(wait(runner)["status"], "error")
            self.assertEqual(out.read_text(), "previous replay")
            runner.start(pose())
            self.assertEqual(wait(runner)["status"], "done")


if __name__ == "__main__":
    unittest.main()
