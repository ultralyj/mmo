"""ManiSkill agent for the Wuji Hand2 Beta 2 right hand."""

from __future__ import annotations

from pathlib import Path

from .hardware import load_hand_hardware, collision_ignore_masks
from .controller import HardwarePDJointPosControllerConfig

from mani_skill.agents.base_agent import BaseAgent
from mani_skill.agents.controllers.pd_joint_pos import (
    PDJointPosControllerConfig,
    PDJointPosController,
)


class WujiHand2Agent(BaseAgent):
    """A fixed-wrist, twenty-DoF hand agent loaded from the local URDF."""

    uid = "wuji_hand2_beta2_right"
    urdf_path = str(
        Path(__file__).resolve().parents[3]
        / "assets/wuji-description/hand2/hand2_beta2/body/urdf/right.urdf"
    )
    fix_root_link = True
    # Legacy checkpoints used disabled self collision; physical configs enable
    # adjacency/vendor exclusions only, on every batched articulation.
    disable_self_collisions = True

    def __init__(
        self,
        *args,
        stiffness: float = 200.0,
        damping: float = 10.0,
        force_limit: float = 5.0,
        friction: float = 1.0,
        actuator_profile: str = "legacy",
        asset_gain_scale: float = 1.0,
        self_collisions: bool = False,
        rate_limit_targets: bool = False,
        **kwargs,
    ) -> None:
        """Configure the joint controller and uniform hand collision material."""

        if actuator_profile not in ("legacy", "asset"):
            raise ValueError("actuator_profile must be legacy or asset")
        if asset_gain_scale <= 0:
            raise ValueError("asset_gain_scale must be positive")
        self.actuator_profile = actuator_profile
        self.asset_gain_scale = asset_gain_scale
        self.disable_self_collisions = not self_collisions
        self.rate_limit_targets = rate_limit_targets
        self.hardware, self.excluded_collision_pairs = load_hand_hardware(self.urdf_path)
        self.stiffness = stiffness
        self.damping = damping
        self.force_limit = force_limit
        self.urdf_config = {
            "material": {
                "static_friction": friction,
                "dynamic_friction": friction,
                "restitution": 0.0,
            }
        }
        super().__init__(*args, **kwargs)

    def _after_init(self):
        if self.actuator_profile == "asset":
            for articulation in self.robot._objs:
                for joint, name in zip(articulation.active_joints, [j.name for j in self.robot.active_joints]):
                    hardware = self.hardware[name]
                    joint.set_max_velocity(hardware.velocity_rad_s)
                    joint.set_armature([hardware.armature_kg_m2])
                # BaseAgent normally disables gravity for passive-force balancing.
                for link in articulation.links:
                    link.disable_gravity = False
        if not self.disable_self_collisions:
            masks = collision_ignore_masks([link.name for link in self.robot.links if link._bodies[0].collision_shapes], self.excluded_collision_pairs)
            for index in range(len(self.robot._objs)):
                for link in self.robot.links:
                    for shape in link._bodies[index].collision_shapes:
                        groups = list(shape.collision_groups)
                        groups[2] = masks[link.name]
                        groups[3] = 1000 + index
                        shape.set_collision_groups(groups)

    @property
    def _controller_configs(self):
        names = [joint.name for joint in self.robot.active_joints]
        self._drive_stiffness = [self.hardware[n].stiffness_nm_rad * self.asset_gain_scale if self.actuator_profile == "asset" else self.stiffness for n in names]
        self._drive_damping = [self.hardware[n].damping_nm_s_rad * self.asset_gain_scale if self.actuator_profile == "asset" else self.damping for n in names]
        self._drive_limits = [min(self.force_limit, self.hardware[n].effort_nm) if self.actuator_profile == "asset" else self.force_limit for n in names]
        return dict(
            pd_joint_pos=(HardwarePDJointPosControllerConfig if self.rate_limit_targets else PDJointPosControllerConfig)(
                [joint.name for joint in self.robot.active_joints],
                lower=None,
                upper=None,
                stiffness=self._drive_stiffness,
                damping=self._drive_damping,
                force_limit=self._drive_limits,
                interpolate=self.rate_limit_targets,
                normalize_action=False,
            )
        )
