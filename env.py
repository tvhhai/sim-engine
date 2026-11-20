"""Môi trường Gymnasium cho bài toán SO-101 gắp hộp bỏ vào bát (pick-and-place) trong MuJoCo.

Cách hiểu nhanh: env là "sân tập". Vòng lặp luôn là:
    obs, info = env.reset()                       # đặt lại sân tập, nhận quan sát đầu tiên
    loop:
        action = policy(obs)                      # policy nhìn quan sát, quyết định hành động
        obs, reward, terminated, truncated, info = env.step(action)   # env thực hiện hành động
"""

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from . import box_sampling, config
from .renderer import OffscreenRenderer


class SO101PickPlaceEnv(gym.Env):
    """Robot SO-101 ảo thực hiện gắp hộp bỏ vào bát.

    Observation (quan sát):
        state:  (7,) float32 — [0..4] = 5 góc khớp tay (đơn vị ĐỘ), [5] = LUÔN BẰNG 0 (ô trống, xem _get_obs),
                [6] = độ mở kẹp (0 = đóng, 100 = mở)
        images: dict gồm ảnh 2 camera webcam và arm_cam, mỗi ảnh (H, W, 3) uint8 RGB

    Action (hành động):
        (7,) float32, mỗi số nằm trong [-1, 1]:
        5 số đầu = góc đích của 5 khớp tay (đã chuẩn hóa), số 6 và 7 = lệnh kẹp (-1 đóng, +1 mở)
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": config.CONTROL_FREQ}

    def __init__(self, render_mode=None, render_images=True, end_on_success=True,
                 box_range=None, bowl_range=None):
        # render_images=False: không render ảnh camera trong obs -> chạy nhanh hơn nhiều
        # (policy viết tay không cần ảnh nên thường tắt đi)
        # Kẹp THẬT: hộp chỉ được giữ bằng va chạm và ma sát thật của hai hàm kẹp (không có kẹp giả / snap-grasp).
        # end_on_success=False: ván KHÔNG dừng khi vừa thỏa điều kiện thành công, để policy chạy đủ mọi phase
        #   (thả, rút tay, về home). Khi đó đọc info["success"] ở bước CUỐI để biết hộp có còn nằm trong bát không.
        # box_range / bowl_range: nửa cạnh (mét) của ô vuông mà hộp / bát được đặt ngẫu nhiên trong đó mỗi lần reset.
        #   Mặc định lấy từ config (hộp ±3 cm, bát đứng yên). Cùng seed thì cùng vị trí.
        super().__init__()
        self.box_range = config.BOX_RANDOM_RANGE if box_range is None else box_range
        # box_range đặt tay (không None) nghĩa là người gọi muốn ô vuông cũ; mặc định theo config.BOX_SAMPLING
        self.box_sampling = "square" if box_range is not None else config.BOX_SAMPLING
        self.bowl_range = config.BOWL_RANDOM_RANGE if bowl_range is None else bowl_range
        self.render_mode = render_mode
        self.render_images = render_images
        self.end_on_success = end_on_success

        # Nạp mô hình MuJoCo từ file XML (robot, bàn, hộp, bát, camera)
        self.model = mujoco.MjModel.from_xml_path(config.SCENE_XML)
        # Độ gập của giá đỡ camera cổ tay (body wrist_cam_arm gập quanh trục x của chốt): đặt theo config
        cam_arm = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "wrist_cam_arm")
        if cam_arm >= 0:
            half = np.radians(config.WRIST_CAM_TILT_DEG) / 2
            self.model.body_quat[cam_arm] = [np.cos(half), np.sin(half), 0.0, 0.0]
        self.model.opt.timestep = config.PHYSICS_TIMESTEP
        # model = "bản thiết kế" (cố định); data = "trạng thái hiện tại" (thay đổi mỗi bước)
        self.data = mujoco.MjData(self.model)

        # Renderer chụp ảnh từ camera ảo (chạy ngoài màn hình, offscreen)
        if self.render_images:
            self._renderer = OffscreenRenderer(
                self.model, config.CAMERA_WIDTH, config.CAMERA_HEIGHT
            )
        else:
            self._renderer = None

        # Tra sẵn ID của các khớp / vật thể theo tên, để khỏi tra lại mỗi bước
        self._arm_joint_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            for name in config.JOINT_NAMES[:5]  # 5 khớp tay (không gồm gripper)
        ]
        # Khớp hàm kẹp (robot thật chỉ có 1 hàm di động, hàm kia cố định)
        self._gripper_jnt = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "gripper"
        )
        self._box_body = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "box"
        )
        self._bowl_body = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "bowl"
        )
        # ee_site = điểm "đầu tay" (end-effector), nằm giữa 2 má kẹp. IK nhắm điểm này tới đích.
        self._ee_site = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "ee_site"
        )
        self._bowl_default_xy = self.model.body_pos[self._bowl_body][:2].copy()   # vị trí bát gốc trong file XML

        # Khoảng điều khiển (min, max) của từng actuator, dùng để đổi action [-1,1] ra giá trị thật
        self._ctrl_ranges = self.model.actuator_ctrlrange.copy()

        self._grasped = False                 # THÔNG TIN: cả hai hàm kẹp đang cùng chạm hộp (không điều khiển gì cả)
        # Hai má kẹp, dùng để biết "hộp đang bị kẹp thật" (cả hai má cùng chạm hộp) 
        # Mỗi hàm gồm nhiều khối va chạm nhỏ (fixed_jaw_col_00..., moving_jaw_col_00...): {id geom: "fixed" | "moving"}
        self._pad_geoms = {}
        for g in range(self.model.ngeom):
            gname = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
            if gname.startswith("fixed_jaw_col"):
                self._pad_geoms[g] = "fixed"
            elif gname.startswith("moving_jaw_col"):
                self._pad_geoms[g] = "moving"
        box_joint = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "box_joint")
        self._box_qpos_adr = self.model.jnt_qposadr[box_joint]
        self._box_dof_adr = self.model.jnt_dofadr[box_joint]

        # Nhớ trạng thái ban đầu để reset() khôi phục
        mujoco.mj_forward(self.model, self.data)
        self._init_qpos = self.data.qpos.copy()
        self._init_qvel = self.data.qvel.copy()

        # Khai báo không gian hành động và quan sát (theo chuẩn Gymnasium)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(7,), dtype=np.float32)

        obs_spaces = {"state": spaces.Box(-np.inf, np.inf, shape=(7,), dtype=np.float32)}
        if self.render_images:
            obs_spaces["images"] = spaces.Dict({
                name: spaces.Box(0, 255, shape=(config.CAMERA_HEIGHT, config.CAMERA_WIDTH, 3), dtype=np.uint8)
                for name in config.CAMERA_NAMES
            })
        self.observation_space = spaces.Dict(obs_spaces)

        self._step_count = 0
        self._viewer = None

    # ------------------------------------------------------------------
    # Giao diện Gymnasium: reset / step / render / close
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        """Bắt đầu episode mới. Đặt robot về home, dịch hộp (và bát, nếu bowl_range > 0) ngẫu nhiên.

        seed: cùng seed => cùng vị trí hộp => kết quả lặp lại được (rất hữu ích khi debug).
        options (tùy chọn): {"box_xy": [x, y], "bowl_xy": [x, y]} đặt thẳng vị trí hộp / bát, thay cho vị trí ngẫu nhiên.
        Trả về (obs, info).
        """
        super().reset(seed=seed)

        # Khôi phục trạng thái ban đầu
        mujoco.mj_resetData(self.model, self.data)      # xoá cả bộ nhớ ấm của solver (qacc_warmstart...): ván sau không phụ thuộc ván trước
        self.data.qpos[:] = self._init_qpos
        self.data.qvel[:] = self._init_qvel
        self.data.ctrl[:] = 0.0

        # Đặt tay về tư thế home
        for i, jname in enumerate(config.JOINT_NAMES[:5]):
            jid = self._arm_joint_ids[i]
            qadr = self.model.jnt_qposadr[jid]
            self.data.qpos[qadr] = config.HOME_QPOS_RAD[i]

        # Kẹp khép ở tư thế nghỉ (policy sẽ mở ra khi tiếp cận)
        fl_qadr = self.model.jnt_qposadr[self._gripper_jnt]
        self.data.qpos[fl_qadr] = config.GRIPPER_REST_RAD

        # Dịch vị trí hộp ngẫu nhiên trong ±box_range theo x và y (z giữ nguyên)
        box_qpos_adr = self._box_qpos_adr
        self.data.qpos[box_qpos_adr:box_qpos_adr + 3] = config.BOX_DEFAULT_POS + self.np_random.uniform(
            low=[-self.box_range, -self.box_range, 0.0], high=[self.box_range, self.box_range, 0.0]
        )
        if self.box_sampling == "sector":          # phủ đều vùng gắp được theo seed (xem box_sampling.py); seed 1-6 là các trường hợp cực biên
            self.data.qpos[box_qpos_adr:box_qpos_adr + 2] = box_sampling.box_xy_for_seed(seed)
        # Bát: luôn khôi phục về chỗ gốc, rồi (nếu bật) dịch ngẫu nhiên trong ±bowl_range. Rút số ngẫu nhiên SAU hộp
        # để vị trí hộp của mỗi seed không đổi khi bật random bát.
        self.model.body_pos[self._bowl_body][:2] = self._bowl_default_xy
        if self.bowl_range > 0:
            self.model.body_pos[self._bowl_body][:2] += self.np_random.uniform(-self.bowl_range, self.bowl_range, 2)
        if "bowl_xy" in (options or {}):
            self.model.body_pos[self._bowl_body][:2] = options["bowl_xy"]
        opts = options or {}
        if "box_xy" in opts:                      # đặt tay vị trí hộp (không ảnh hưởng số ngẫu nhiên đã rút ở trên)
            self.data.qpos[box_qpos_adr:box_qpos_adr + 2] = opts["box_xy"]
        # Đặt hướng hộp về mặc định (quaternion đơn vị = không xoay)
        self.data.qpos[box_qpos_adr + 3:box_qpos_adr + 7] = [1, 0, 0, 0]
        # Hộp đứng yên (vận tốc = 0)
        box_dof_adr = self._box_dof_adr
        self.data.qvel[box_dof_adr:box_dof_adr + 6] = 0.0

        # Xóa trạng thái kẹp
        self._grasped = False

        # Đặt lệnh điều khiển (ctrl) bằng đúng vị trí hiện tại để robot không giật
        for i in range(5):
            jid = self._arm_joint_ids[i]
            qadr = self.model.jnt_qposadr[jid]
            self.data.ctrl[i] = self.data.qpos[qadr]
        self.data.ctrl[5] = config.GRIPPER_REST_RAD  # hàm di động khép

        # Cho vật lý "lắng xuống" 50 bước (hộp chạm bàn, tay ổn định)
        for _ in range(50):
            mujoco.mj_step(self.model, self.data)

        self._step_count = 0
        obs = self._get_obs()
        return obs, self._get_info()

    def step(self, action):
        """Thực hiện 1 hành động (1 bước điều khiển ~ 1/30 giây).

        Trả về (obs, reward, terminated, truncated, info):
            terminated = True khi THÀNH CÔNG (hộp đã vào bát)
            truncated  = True khi hết giờ (đủ MAX_EPISODE_STEPS bước mà chưa xong)
        """
        action = np.clip(action, -1.0, 1.0).astype(np.float32)

        # Đổi action từ thang [-1, 1] sang khoảng điều khiển thật của từng actuator.
        # 5 số đầu = góc đích của 5 khớp tay (MuJoCo tự kéo khớp tới góc đó bằng position actuator)
        for i in range(5):
            lo, hi = self._ctrl_ranges[i]
            self.data.ctrl[i] = lo + (action[i] + 1.0) * 0.5 * (hi - lo)

        # Gripper: action[5] trong [-1,1] -> góc đích của hàm di động [GRIPPER_CLOSED_RAD, GRIPPER_OPEN_RAD].
        gripper_val = config.GRIPPER_CLOSED_RAD + (action[5] + 1.0) * 0.5 * (config.GRIPPER_OPEN_RAD - config.GRIPPER_CLOSED_RAD)
        self.data.ctrl[5] = gripper_val

        # action[6] không được dùng (robot thật chỉ có 1 hàm di động)

        # Chạy vật lý: 1 bước điều khiển = PHYSICS_SUBSTEPS bước vật lý nhỏ (16 bước x 0.002s)
        for _ in range(config.PHYSICS_SUBSTEPS):
            mujoco.mj_step(self.model, self.data)

        # _grasped chỉ là THÔNG TIN ("cả hai hàm đang chạm hộp"), không gán vị trí hộp
        self._grasped = self._both_pads_touch_box()

        self._step_count += 1
        obs = self._get_obs()
        info = self._get_info()

        reward = self._compute_reward(info)
        terminated = bool(info["success"]) and self.end_on_success
        truncated = self._step_count >= config.MAX_EPISODE_STEPS

        return obs, reward, terminated, truncated, info

    def render(self):
        """Trả về ảnh webcam (rgb_array) hoặc mở cửa sổ viewer tương tác (human)."""
        if self.render_mode == "rgb_array" and self._renderer is not None:
            return self._renderer.render(self.data, "webcam")
        elif self.render_mode == "human":
            self._render_human()
        return None

    def close(self):
        """Giải phóng tài nguyên renderer và viewer."""
        if self._renderer is not None:
            self._renderer.close()
        if self._viewer is not None:
            self._viewer.close()

    # ------------------------------------------------------------------
    # Hàm nội bộ (bắt đầu bằng dấu gạch dưới _)
    # ------------------------------------------------------------------

    def _both_pads_touch_box(self):
        """True nếu CẢ HAI má kẹp đang chạm hộp (dấu hiệu hộp đang bị kẹp thật)."""
        touching = set()
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            for pad, other in ((c.geom1, c.geom2), (c.geom2, c.geom1)):
                if pad in self._pad_geoms and self.model.geom_bodyid[other] == self._box_body:
                    touching.add(self._pad_geoms[pad])
        return len(touching) == 2

    def _get_obs(self):
        """Lấy quan sát hiện tại: state (7 số) và ảnh camera (nếu bật render_images)."""
        # Góc 5 khớp tay, đổi từ radian sang độ.
        # Lưu ý: mảng có 6 ô nhưng vòng lặp dưới chỉ điền 5 ô đầu, nên ô thứ 6 (state[5]) luôn = 0.
        # Độ mở kẹp được nối thêm vào cuối nên nằm ở state[6]. (Đây là quirk của repo gốc.)
        joint_pos = np.zeros(6, dtype=np.float32)
        for i, jid in enumerate(self._arm_joint_ids):
            qadr = self.model.jnt_qposadr[jid]
            joint_pos[i] = np.rad2deg(self.data.qpos[qadr])

        # Độ mở kẹp: 0 (đóng) đến 100 (mở hết)
        fl_qadr = self.model.jnt_qposadr[self._gripper_jnt]
        gripper_pos = self.data.qpos[fl_qadr]
        span = config.GRIPPER_OPEN_RAD - config.GRIPPER_CLOSED_RAD
        gripper_normalized = np.float32(np.clip((gripper_pos - config.GRIPPER_CLOSED_RAD) / span * 100.0, 0, 100))

        state = np.append(joint_pos, gripper_normalized).astype(np.float32)

        obs = {"state": state}
        if self.render_images and self._renderer is not None:
            obs["images"] = {
                name: self._renderer.render(self.data, name)
                for name in config.CAMERA_NAMES
            }
        return obs

    def _get_info(self):
        """Thông tin phụ (không đưa vào policy AI): vị trí vật thể và cờ THÀNH CÔNG.

        success = hộp THẬT SỰ nằm trong bát, tức là cả 4 điều cùng đúng:
            1. tâm hộp cách tâm bát (theo chiều ngang) < BOWL_INNER_RADIUS (lòng bát, tới mặt trong tường)
            2. TÂM hộp thấp hơn miệng bát ít nhất BOX_IN_BOWL_MARGIN (5 mm), nên hộp nằm bên trong chứ không gác trên miệng. Dùng TÂM chứ
               không dùng đỉnh: hộp nằm nghiêng tựa thành bát vẫn ở trong bát dù một góc nhô cao hơn miệng. Hộp gác trên miệng có tâm
               ngang thành bát (xy >= lòng bát) và cao hơn miệng, nên vẫn bị loại bởi điều 1 và 2
            3. hộp đã được THẢ ra (không còn bị kẹp giữ)
            4. hộp đã nằm yên (tốc độ < SETTLED_SPEED), nên hộp đang rơi hay nảy chưa tính
        Đây chính là chỗ "chấm điểm" một episode thành công hay không.
        """
        box_pos = self.data.xpos[self._box_body].copy()
        bowl_pos = self.data.xpos[self._bowl_body].copy()
        ee_pos = self.data.site_xpos[self._ee_site].copy()

        horizontal_dist = np.linalg.norm(box_pos[:2] - bowl_pos[:2])
        rim_z = bowl_pos[2] + config.BOWL_HEIGHT
        box_speed = float(np.linalg.norm(self.data.qvel[self._box_dof_adr:self._box_dof_adr + 3]))
        in_xy = horizontal_dist < config.BOWL_INNER_RADIUS
        in_z = bowl_pos[2] < box_pos[2] < rim_z - config.BOX_IN_BOWL_MARGIN
        released = not self._grasped
        settled = box_speed < config.SETTLED_SPEED
        success = in_xy and in_z and released and settled

        return {
            "success": bool(success),
            "box_speed": box_speed,
            "in_bowl_xy": bool(in_xy),
            "in_bowl_z": bool(in_z),
            "box_pos": box_pos,
            "bowl_pos": bowl_pos,
            "ee_pos": ee_pos,
            "ee_to_box_dist": float(np.linalg.norm(ee_pos - box_pos)),
            "box_to_bowl_dist": float(np.linalg.norm(box_pos[:2] - bowl_pos[:2])),
        }

    def _compute_reward(self, info):
        """Phần thưởng (reward) mỗi bước, dùng cho RL. Imitation learning không cần tới.

        - Càng gần hộp càng ít bị trừ điểm.
        - Hộp được nhấc lên khỏi bàn: +1, rồi càng gần bát càng đỡ bị trừ.
        - Hộp vào bát: +5.
        """
        reward = 0.0
        # Shaping: khuyến khích tay lại gần hộp
        reward -= 0.1 * info["ee_to_box_dist"]
        # Thưởng khi hộp được nhấc lên khỏi mặt bàn
        if info["box_pos"][2] > 0.47:
            reward += 1.0
            # Shaping: khuyến khích đưa hộp lại gần bát
            reward -= 0.1 * info["box_to_bowl_dist"]
        # Thưởng khi đặt hộp vào bát
        if info["success"]:
            reward += 5.0
        return reward

    def _render_human(self):
        """Mở cửa sổ viewer tương tác của MuJoCo (chế độ render_mode='human')."""
        if self._viewer is None:
            import mujoco.viewer
            self._viewer = mujoco.viewer.launch_passive(self.model, self.data)
        else:
            self._viewer.sync()

    # ------------------------------------------------------------------
    # Hàm tiện ích: policy gọi để lấy vị trí vật thể (dùng cho policy viết tay)
    # ------------------------------------------------------------------

    def get_ee_pos(self):
        """Vị trí đầu tay (end-effector) trong hệ tọa độ world, đơn vị mét."""
        return self.data.site_xpos[self._ee_site].copy()

    def get_box_pos(self):
        """Vị trí hộp trong hệ tọa độ world."""
        return self.data.xpos[self._box_body].copy()

    def get_bowl_pos(self):
        """Vị trí bát trong hệ tọa độ world."""
        return self.data.xpos[self._bowl_body].copy()

    @property
    def arm_joint_ids(self):
        return self._arm_joint_ids

    @property
    def ee_site_id(self):
        return self._ee_site
