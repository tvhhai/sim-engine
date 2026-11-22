"""Interactive demo recording screen."""

import os
import time
from contextlib import ExitStack

from .. import config
from ..dataset import DatasetRecorder
from ..env import SO101PickPlaceEnv
from ..policies import default_policy
from ..workflows.record import record_policy_step


def record_screen(cv2, window, draw_menu, put, show_message, width, height, colors):
    green, white, gray, _ = colors

    while True:
        save_dir = config.DEFAULT_DATASET_DIR
        n_existing = len([f for f in os.listdir(save_dir) if f.endswith(".hdf5")]) if os.path.exists(save_dir) else 0

        img = draw_menu("Record Episodes", [
            ("s", f"Start Recording  (scripted policy, 10 episodes)", green),
            ("n", f"Dataset: {save_dir}", white),
            ("", f"Existing episodes: {n_existing}", gray),
            ("", "", white),
            ("", "The scripted policy will automatically perform", gray),
            ("", "pick-and-place. Each episode is saved as HDF5.", gray),
            ("", "Images from both cameras are recorded.", gray),
            ("", "", white),
            ("b", "Back to menu", gray),
        ])
        cv2.imshow(window, img)
        key = cv2.waitKey(100)

        if key == ord("b") or key == 27: return
        if key == ord("s"):
            _run_recording(cv2, 10, save_dir, window, put, show_message, width, height, colors)

def _run_recording(cv2, num_episodes, save_dir, window, put, show_message, width, height, colors):
    _, _, gray, red = colors

    with ExitStack() as resources:
        env = SO101PickPlaceEnv(render_images=True)
        resources.callback(env.close)
        policy = default_policy(env)
        recorder = DatasetRecorder(save_dir=save_dir)
        pip_w, pip_h = 200, 150
        successes = 0

        for ep in range(num_episodes):
            obs, info = env.reset(seed=ep * 7 + int(time.time()) % 100)
            policy.reset()
            recorder.start_episode(task=config.DEFAULT_TASK)

            for step in range(config.MAX_EPISODE_STEPS):
                obs, _, _, truncated, info = record_policy_step(
                    env, policy, recorder, obs, config.DEFAULT_TASK,
                )

                # Reuse the post-step camera images already rendered by the environment.
                images = obs["images"]
                frame_bgr = cv2.resize(cv2.cvtColor(images["webcam"], cv2.COLOR_RGB2BGR), (width, height))
                wrist_bgr = cv2.resize(cv2.cvtColor(images["arm_cam"], cv2.COLOR_RGB2BGR), (pip_w, pip_h))
                y1 = height - pip_h - 10
                x1 = width - pip_w - 10
                frame_bgr[y1:y1+pip_h, x1:x1+pip_w] = wrist_bgr
                put(frame_bgr, "Wrist", (x1, y1 - 5), gray, 0.4, 1)

                # Status bar
                status = f"RECORDING  |  Episode {ep+1}/{num_episodes}  |  Step {step}  |  Success: {successes}/{ep}"
                put(frame_bgr, status, (10, 30), red, 0.65, 2)

                # Recording indicator (blinking red dot)
                if (step // 10) % 2 == 0:
                    cv2.circle(frame_bgr, (width - 30, 30), 10, (0, 0, 255), -1)

                cv2.imshow(window, frame_bgr)
                k = cv2.waitKey(20)
                if k == 27:  # ESC to abort
                    recorder.end_episode(success=False)
                    return

                if policy._phase > 10 or truncated:
                    break

            success = info.get("success", False)
            if success:
                successes += 1
            recorder.end_episode(success=success)

    # Show summary
    show_message(cv2, f"Recording Complete!  {successes}/{num_episodes} successful",
                  f"Saved to: {save_dir}", 3000)
