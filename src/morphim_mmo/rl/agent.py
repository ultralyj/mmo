"""ManiSkill agent for the Wuji Hand2 Beta 2 right hand."""

from __future__ import annotations

from pathlib import Path

from mani_skill.agents.base_agent import BaseAgent
from mani_skill.agents.controllers.pd_joint_pos import (
    PDJointPosControllerConfig,
)


class WujiHand2Agent(BaseAgent):
    """A fixed-wrist, twenty-DoF hand agent loaded from the local URDF."""

    uid = "wuji_hand2_beta2_right"
    urdf_path = str(
        Path(__file__).resolve().parents[3]
        / "assets/wuji-description/hand2/hand2_beta2/body/urdf/right.urdf"
    )
    fix_root_link = True
    disable_self_collisions = False

    def __init__(
        self,
        *args,
        stiffness: float = 200.0,
        damping: float = 10.0,
        force_limit: float = 5.0,
        friction: float = 1.0,
        **kwargs,
    ) -> None:
        """Configure the joint controller and uniform hand collision material."""

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

    @property
    def _controller_configs(self):
        return dict(
            pd_joint_pos=PDJointPosControllerConfig(
                [joint.name for joint in self.robot.active_joints],
                lower=None,
                upper=None,
                stiffness=self.stiffness,
                damping=self.damping,
                force_limit=self.force_limit,
                normalize_action=False,
            )
        )
