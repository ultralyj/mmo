"""Position targets interpolated with explicit hardware velocity bounds."""
from __future__ import annotations

from dataclasses import dataclass
import torch
from mani_skill.agents.controllers.pd_joint_pos import PDJointPosController, PDJointPosControllerConfig


class HardwarePDJointPosController(PDJointPosController):
    def set_action(self, action):
        desired = self._preprocess_action(action)
        start = self._target_qpos.clone()
        if not hasattr(self, "_velocity_bounds"):
            self._velocity_bounds = torch.tensor([self.articulation._objs[0].active_joints[int(i)].max_velocity
                                   for i in self.active_joint_indices.cpu().tolist()],
                                  device=self.device, dtype=desired.dtype) / self.control_freq
        bounds = self._velocity_bounds
        self._start_qpos = start
        self._target_qpos = start + torch.clamp(desired - start, -bounds, bounds)
        self._step = 0
        self._step_size = (self._target_qpos - start) / self._sim_steps

    def before_simulation_step(self):
        super().before_simulation_step()
        if self.scene.gpu_sim_enabled:
            # BaseEnv uploads once before interpolation updates; upload every substep.
            self.scene.px.gpu_apply_articulation_target_position()


@dataclass
class HardwarePDJointPosControllerConfig(PDJointPosControllerConfig):
    controller_cls = HardwarePDJointPosController
