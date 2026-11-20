"""HDF5 dataset recorder and PyTorch-compatible loader."""

import hashlib
import json
import os
import tempfile
from pathlib import Path
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from . import config


class DatasetRecorder:
    """Records episodes to HDF5 files."""

    def __init__(self, save_dir=None):
        self.save_dir = save_dir or config.DEFAULT_DATASET_DIR
        os.makedirs(self.save_dir, exist_ok=True)
        self._buffers = None
        self._task = None
        indices = (
            int(path.stem.removeprefix("episode_"))
            for path in Path(self.save_dir).glob("episode_*.hdf5")
            if path.stem.removeprefix("episode_").isdigit()
        )
        self._episode_idx = max(indices, default=-1) + 1

    def start_episode(self, task: str = None):
        self._task = task or config.DEFAULT_TASK
        self._buffers = {
            "states": [],
            "images_webcam": [],
            "images_arm_cam": [],
            "actions": [],
            "ann_phase": [],
            "ann_recovery": [],
            "ann_roll_flips": 0,
            "ann_bowl_avoid": 0,
        }

    def record_step(self, obs: dict, action: np.ndarray, annotation: dict = None):
        """annotation (tùy chọn, từ policy.annotation()): chú thích từng bước, lưu ở nhóm `annotations`, không đổi action."""
        assert self._buffers is not None, "Call start_episode() first"
        if annotation is not None:
            self._buffers["ann_phase"].append(annotation["phase"])
            self._buffers["ann_recovery"].append(annotation["recovery"])
            self._buffers["ann_roll_flips"] = max(self._buffers["ann_roll_flips"], annotation["roll_flips"])
            self._buffers["ann_bowl_avoid"] = max(self._buffers["ann_bowl_avoid"], annotation["bowl_avoid"])
        self._buffers["states"].append(obs["state"].copy())
        self._buffers["actions"].append(action.copy())
        if "images" in obs:
            self._buffers["images_webcam"].append(obs["images"]["webcam"].copy())
            self._buffers["images_arm_cam"].append(obs["images"]["arm_cam"].copy())

    def end_episode(self, success: bool = False):
        assert self._buffers is not None, "Call start_episode() first"

        n_steps = len(self._buffers["states"])
        fd, temporary_path = tempfile.mkstemp(dir=self.save_dir, suffix=".tmp")
        os.close(fd)
        try:
            with h5py.File(temporary_path, "w") as f:
                obs_g = f.create_group("observations")
                obs_g.create_dataset("state", data=np.array(self._buffers["states"], dtype=np.float32))

                if self._buffers["images_webcam"]:
                    img_g = obs_g.create_group("images")
                    img_g.create_dataset(
                        "webcam",
                        data=np.array(self._buffers["images_webcam"], dtype=np.uint8),
                        compression="gzip", compression_opts=4,
                    )
                    img_g.create_dataset(
                        "arm_cam",
                        data=np.array(self._buffers["images_arm_cam"], dtype=np.uint8),
                        compression="gzip", compression_opts=4,
                    )

                f.create_dataset("actions", data=np.array(self._buffers["actions"], dtype=np.float32))
                if self._buffers["ann_phase"]:
                    ann = f.create_group("annotations")
                    ann.create_dataset("phase", data=np.array(self._buffers["ann_phase"], dtype=np.int8))
                    ann.create_dataset("recovery", data=np.array(self._buffers["ann_recovery"], dtype=np.int8))
                    ann.attrs["layout"] = "phase 0-11; recovery 0 normal / 1 lifting off the bowl / 2 turning wrist_roll (steps to filter or down-weight)"
                    f.attrs["roll_flips"] = self._buffers["ann_roll_flips"]
                    f.attrs["bowl_avoid"] = self._buffers["ann_bowl_avoid"]
                f.attrs["contract_version"] = 1
                f.attrs["source"] = "mujoco"
                f.attrs["scene_sha256"] = hashlib.sha256(Path(config.SCENE_XML).read_bytes()).hexdigest()
                f.attrs["physics_fps"] = 1 / (config.PHYSICS_SUBSTEPS * config.PHYSICS_TIMESTEP)
                f.attrs["joint_order"] = json.dumps(config.JOINT_NAMES)
                f.attrs["state_layout"] = "5 joint angles in degrees; reserved zero; gripper opening percent"
                f.attrs["action_layout"] = "5 normalized joint targets; normalized gripper target (positive opens); unused"
                f.attrs["task"] = self._task
                f.attrs["fps"] = config.CONTROL_FREQ
                f.attrs["success"] = success
                f.attrs["num_steps"] = n_steps
            while True:
                fname = f"episode_{self._episode_idx:04d}.hdf5"
                fpath = os.path.join(self.save_dir, fname)
                try:
                    # Publish a complete file without replacing an existing episode.
                    if os.name == "nt":
                        os.rename(temporary_path, fpath)
                    else:
                        os.link(temporary_path, fpath)
                    break
                except FileExistsError:
                    self._episode_idx += 1
        finally:
            Path(temporary_path).unlink(missing_ok=True)

        print(f"  Saved {fname} ({n_steps} steps, success={success})")
        self._episode_idx += 1
        self._buffers = None
        return fpath


class SimDataset(Dataset):
    """PyTorch Dataset that loads episodes from HDF5 files.

    Returns (state, action) pairs for training state-based policies,
    or (state, images, action) if load_images=True.
    """

    def __init__(self, data_dir=None, load_images=False):
        self.data_dir = data_dir or config.DEFAULT_DATASET_DIR
        self.load_images = load_images

        # Scan for episode files
        self.episode_files = sorted([
            os.path.join(self.data_dir, f)
            for f in os.listdir(self.data_dir)
            if f.endswith(".hdf5")
        ])
        assert len(self.episode_files) > 0, f"No episodes found in {self.data_dir}"

        # Build index: (episode_idx, step_idx) for each global index
        self._index = []
        self._states = []
        self._actions = []
        self._tasks = []

        for ep_idx, fpath in enumerate(self.episode_files):
            with h5py.File(fpath, "r") as f:
                states = f["observations/state"][:]
                actions = f["actions"][:]
                task = f.attrs.get("task", config.DEFAULT_TASK)

                self._states.append(states)
                self._actions.append(actions)
                self._tasks.append(task)

                if load_images:
                    for camera in config.CAMERA_NAMES:
                        key = f"observations/images/{camera}"
                        if key not in f or len(f[key]) != len(states):
                            raise ValueError(f"{fpath}: missing or incomplete camera {camera}")

                for step in range(len(states)):
                    self._index.append((ep_idx, step))

        # Concatenate for easy normalization
        all_states = np.concatenate(self._states, axis=0)
        all_actions = np.concatenate(self._actions, axis=0)

        self.state_mean = all_states.mean(axis=0).astype(np.float32)
        self.state_std = all_states.std(axis=0).astype(np.float32) + 1e-6
        self.action_mean = all_actions.mean(axis=0).astype(np.float32)
        self.action_std = all_actions.std(axis=0).astype(np.float32) + 1e-6

        print(f"Loaded {len(self.episode_files)} episodes, {len(self._index)} total steps")

    def __len__(self):
        return len(self._index)

    def __getitem__(self, idx):
        ep_idx, step_idx = self._index[idx]

        state = self._states[ep_idx][step_idx].astype(np.float32)
        action = self._actions[ep_idx][step_idx].astype(np.float32)

        # Normalize
        state_norm = (state - self.state_mean) / self.state_std
        action_norm = (action - self.action_mean) / self.action_std

        result = {
            "state": torch.from_numpy(state_norm),
            "action": torch.from_numpy(action_norm),
            "state_raw": torch.from_numpy(state),
            "action_raw": torch.from_numpy(action),
        }

        if self.load_images:
            with h5py.File(self.episode_files[ep_idx], "r") as episode:
                webcam = episode["observations/images/webcam"][step_idx]
                arm_cam = episode["observations/images/arm_cam"][step_idx]
            # HWC uint8 -> CHW float32 [0, 1]
            result["webcam"] = torch.from_numpy(webcam).permute(2, 0, 1).float() / 255.0
            result["arm_cam"] = torch.from_numpy(arm_cam).permute(2, 0, 1).float() / 255.0

        return result

    def get_normalization_stats(self):
        """Return normalization statistics for use during inference."""
        return {
            "state_mean": self.state_mean,
            "state_std": self.state_std,
            "action_mean": self.action_mean,
            "action_std": self.action_std,
        }
