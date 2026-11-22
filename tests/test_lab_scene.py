import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from sim_engine.env import SO101PickPlaceEnv
from sim_engine.studio.runner import LabRun
from sim_engine.studio.scene import normalize_objects
from sim_engine.studio.server import Lab
from test_lab_run import pose, wait


def objects(count=5):
    return [dict(id=f"obj-{i}", type="sphere" if i % 2 else "box", color="#77bbdd",
                 x=.14 + i * .045, y=.02, yaw=20) for i in range(count)]


class LabSceneTests(unittest.TestCase):
    def test_five_objects_are_physical_bodies_with_selected_target(self):
        env = SO101PickPlaceEnv(render_images=False, lab_objects=objects(), target_id="obj-1")
        try:
            env.reset(seed=123, options={"skip_settle": True})
            self.assertEqual(len(env.lab_objects), 5)
            self.assertEqual(int(env.model.geom("box_geom").type[0]), 2)  # mjGEOM_SPHERE
            for obj in env.lab_objects:
                body = env.model.body(obj["body"]).id
                self.assertGreater(env.model.body_mass[body], 0)
                self.assertEqual(env.model.body_jntnum[body], 1)
                np.testing.assert_allclose(env.data.xpos[body][:2], [obj["x"], obj["y"]])
            self.assertTrue(any(name.startswith("fixed_jaw_col") for name in [env.model.geom(i).name for i in range(env.model.ngeom)]))
        finally:
            env.close()

    def test_invalid_count_duplicate_ids_and_color_are_rejected(self):
        for bad in ([], objects(6), [objects(1)[0]] * 2, [dict(objects(1)[0], color="red")]):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                normalize_objects(bad)

    def test_blocked_bowl_and_target_have_zero_steps_and_hold_joint_pose(self):
        for coords in ({"wx": .44, "wy": .39}, {"bx": -.44, "by": -.38}):
            with self.subTest(coords=coords), tempfile.TemporaryDirectory() as directory:
                p = dict(pose(), pan=15, lift=-30, elbow=20, wflex=40, **coords)
                runner = LabRun(directory)
                runner.start(p)
                state = wait(runner)
                self.assertEqual(state["status"], "done", state)
                self.assertEqual(state["steps"], 0)
                self.assertTrue(state["blocked_reason"])
                data = json.loads((Path(directory) / "replay_lab.json").read_text(encoding="utf-8"))
                frame = data["episodes"][0]["frames"][0]
                self.assertEqual(frame["jq"][:4], [15, -30, 20, 40])

    def test_multi_object_rollout_records_every_body_and_pose(self):
        with tempfile.TemporaryDirectory() as directory:
            runner = LabRun(directory)
            runner.start(dict(pose(), objects=objects(), target_id="obj-0"))
            state = wait(runner)
            self.assertEqual(state["status"], "done", state)
            data = json.loads((Path(directory) / "replay_lab.json").read_text(encoding="utf-8"))
            frames = data["episodes"][0]["frames"]
            self.assertGreater(len(frames), 1)
            for frame in frames:
                self.assertEqual(len(frame["objects"]), 5)
                self.assertIn("wrist_cam_arm", frame["pose"])
                self.assertEqual(len(frame["objects"][1]["quat"]), 4)

    def test_preview_switches_target_and_preserves_runner(self):
        lab = Lab()
        runner = lab.runner
        try:
            lab.set_scene(dict(pose(), objects=objects(2), target_id="obj-1"))
            self.assertIs(lab.runner, runner)
            self.assertEqual(len(lab.env.lab_objects), 2)
            np.testing.assert_allclose(lab.env.get_box_pos()[:2], [.185, .02])
            self.assertIn("wrist_cam_arm", lab.snapshot()["pose"])
        finally:
            lab.env.close()


if __name__ == "__main__":
    unittest.main()
