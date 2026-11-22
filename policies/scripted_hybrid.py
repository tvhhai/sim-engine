"""Kết hợp V1 và V2: chạy V1 (ScriptedPolicy) trước; khi V1 bỏ cuộc mà hộp vẫn nằm trên bàn trong tầm gắp thì chuyển sang V2
(ScriptedPolicyV2) gắp tiếp ngay trong ván đó.

Đo trên 274 + 500 seed: V1 274/274 và 499/500, bản kết hợp 274/274 và 500/500. Seed V1 hỏng (2063) là V1 thả hộp, hộp nảy ra khỏi bát
rơi ở r 38 cm, ngoài tầm gắp lại của V1 (REGRASP_R_MAX 0,31) nên V1 về home; V2 gắp lại được.

Chạy thử: python -m sim_engine.tools.select_seeds --policy sim_engine.policies.scripted_hybrid:ScriptedPolicyHybrid
"""

import numpy as np

from .base import BasePolicy
from .scripted_policy import ScriptedPolicy
from .scripted_v2 import ScriptedPolicyV2
from .. import box_sampling, config


class ScriptedPolicyHybrid(BasePolicy):
    def __init__(self, env):
        self.env = env
        self.v1 = ScriptedPolicy(env)
        self.v2 = ScriptedPolicyV2(env)
        self._active = self.v1
        self.switched = False

    def reset(self):
        self.v1.reset()
        self._active, self.switched = self.v1, False

    def __getattr__(self, name):
        # Recorder / select_seeds đọc _phase, _phase_step, motion_target, _current_joint_targets, ... của policy đang chạy
        return getattr(self.__dict__["_active"], name)

    def _v1_gave_up(self):
        """V1 vừa vào phase 10 (về home) mà hộp chưa vào bát, vẫn nằm yên trên bàn trong vùng gắp."""
        if self._active is not self.v1 or self.v1._phase != 10 or self.v1.unreachable:
            return False
        box, bowl = self.env.get_box_pos(), self.env.get_bowl_pos()
        in_bowl = np.linalg.norm(box[:2] - bowl[:2]) < config.BOWL_INNER_RADIUS
        on_table = box[2] < config.BOX_DEFAULT_POS[2] + 0.015
        return not in_bowl and on_table and not box_sampling.is_unreachable(box[:2])

    def select_action(self, observation):
        action = self._active.select_action(observation)
        if not self.switched and self._v1_gave_up():
            self.switched = True
            self.v2.reset()
            self.v2._goto(1, "v1_gave_up")          # lập kế hoạch từ tư thế hiện tại (nhấc lên trước nếu đang thấp)
            self._active = self.v2
            action = self.v2.select_action(observation)
        return action

    def annotation(self):
        return self._active.annotation()
