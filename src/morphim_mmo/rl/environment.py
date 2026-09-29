"""ManiSkill environment for Morphometric Imitation residual RL."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
import sapien
import sapien.physx as physx
import torch
import trimesh
import yourdfpy
from pyroki.collision import RobotCollision

from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import SimConfig

from .agent import WujiHand2Agent
from .config import ResidualRLConfig
from .observation import future_reference_window, privileged_teacher_observation
from .reference import ResidualRLReference
from .reward import average_distance_error, morphometric_reward, transform_object_points
from .termination import (
    TerminationState,
    termination_reasons,
    update_termination_state,
)


def _quaternion_multiply_wxyz(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Multiply batches of wxyz quaternions."""

    left_w, left_xyz = left[..., :1], left[..., 1:]
    right_w, right_xyz = right[..., :1], right[..., 1:]
    xyz = (
        left_w * right_xyz
        + right_w * left_xyz
        + torch.linalg.cross(left_xyz, right_xyz, dim=-1)
    )
    return torch.cat([left_w * right_w - (left_xyz * right_xyz).sum(-1, keepdim=True), xyz], dim=-1)


class MorphometricResidualEnv(BaseEnv):
    """A hand-only environment whose fixed-base wrist follows an MMO reference."""

    SUPPORTED_ROBOTS = [WujiHand2Agent]
    agent: WujiHand2Agent

    def __init__(
        self,
        *,
        reference: ResidualRLReference,
        config: ResidualRLConfig,
        num_envs: int = 1,
        sim_backend: str = "auto",
        **kwargs: Any,
    ) -> None:
        self.reference = reference
        self.config = config
        self._reference_step: torch.Tensor | None = None
        self._active_step: torch.Tensor | None = None
        self._position_offset: torch.Tensor | None = None
        self._yaw_offset: torch.Tensor | None = None
        self._object_anchor: torch.Tensor | None = None
        super().__init__(
            num_envs=num_envs,
            obs_mode="state",
            reward_mode="dense",
            control_mode="pd_joint_pos",
            robot_uids=WujiHand2Agent,
            sim_config=SimConfig(
                sim_freq=config.control.simulation_frequency_hz,
                control_freq=config.control.control_frequency_hz,
            ),
            sim_backend=sim_backend,
            **kwargs,
        )

    def _load_agent(self, options: dict) -> None:
        initial_pose = self.reference.base_pose_position_wxyz[0]
        self.agent = WujiHand2Agent(
            self.scene,
            self._control_freq,
            "pd_joint_pos",
            initial_pose=sapien.Pose(
                p=initial_pose[:3].tolist(),
                q=initial_pose[3:].tolist(),
            ),
            stiffness=self.config.control.stiffness,
            damping=self.config.control.damping,
            force_limit=self.config.control.force_limit,
            friction=self.config.physics.hand_friction,
        )

    def _load_scene(self, options: dict) -> None:
        joint_names = tuple(joint.name for joint in self.agent.robot.active_joints)
        if set(joint_names) != set(self.reference.joint_names):
            raise ValueError(
                "robot and reference joint orders differ: "
                f"{joint_names} != {self.reference.joint_names}"
            )
        joint_indices = [
            self.reference.joint_names.index(name) for name in joint_names
        ]

        device = self.device
        self._env_ids = torch.arange(self.num_envs, device=device)
        self._reference_qpos = torch.as_tensor(
            self.reference.qpos[:, joint_indices],
            dtype=torch.float32,
            device=device,
        )
        self._reference_base_pose = torch.as_tensor(
            self.reference.base_pose_position_wxyz,
            dtype=torch.float32,
            device=device,
        )
        self._reference_object_pose = torch.as_tensor(
            self.reference.object_pose_position_wxyz,
            dtype=torch.float32,
            device=device,
        )
        self._reference_contact = torch.as_tensor(
            self.reference.reference_contact, dtype=torch.bool, device=device
        )
        self._reference_table_height = torch.as_tensor(
            self.reference.table_height_m, dtype=torch.float32, device=device
        )
        self._object_points = torch.as_tensor(
            self.reference.object_points_local_m,
            dtype=torch.float32,
            device=device,
        )

        limits = self.agent.robot.get_qlimits()
        if limits.ndim == 3:
            limits = limits[0]
        self._joint_lower = limits[:, 0].to(device=device, dtype=torch.float32)
        self._joint_upper = limits[:, 1].to(device=device, dtype=torch.float32)
        self._reference_step = torch.zeros(
            self.num_envs, dtype=torch.int64, device=device
        )
        self._active_step = torch.zeros_like(self._reference_step)
        self._position_offset = torch.zeros(
            self.num_envs, 3, dtype=torch.float32, device=device
        )
        self._yaw_offset = torch.zeros(
            self.num_envs, dtype=torch.float32, device=device
        )
        self._object_anchor = self._reference_object_pose[
            self._reference_step, :3
        ].clone()
        self._previous_action = self._reference_qpos[0].repeat(
            self.num_envs, 1
        )
        self._previous_qpos = self._previous_action.clone()
        self._object_scale = torch.ones(
            self.num_envs, 3, dtype=torch.float32, device=device
        )
        self._object_mass = torch.full(
            (self.num_envs,),
            self.config.physics.object_mass_kg,
            dtype=torch.float32,
            device=device,
        )
        self._object_friction = torch.full(
            (self.num_envs,),
            self.config.physics.object_friction,
            dtype=torch.float32,
            device=device,
        )
        self._object_center_of_mass = torch.zeros(
            self.num_envs, 3, dtype=torch.float32, device=device
        )
        self._table_friction = torch.full(
            (self.num_envs,),
            self.config.physics.table_friction,
            dtype=torch.float32,
            device=device,
        )
        self._termination_state = TerminationState.zeros(self.num_envs, device)
        self._robot_links = tuple(self.agent.robot.links)
        urdf = yourdfpy.URDF.load(
            self.agent.urdf_path,
            load_meshes=False,
            load_collision_meshes=True,
        )
        self._link_surface_points: list[tuple[Any, torch.Tensor]] = []
        for link_index, link in enumerate(self._robot_links):
            mesh = RobotCollision._get_trimesh_collision_geometries(
                urdf, link.name
            )
            if mesh.is_empty or mesh.area <= 0:
                continue
            sampled, _ = trimesh.sample.sample_surface(
                mesh, 16, seed=link_index
            )
            self._link_surface_points.append(
                (
                    link,
                    torch.as_tensor(
                        sampled, dtype=torch.float32, device=device
                    ),
                )
            )
        prefixes = (
            "r_thumb",
            "r_index_finger",
            "r_middle_finger",
            "r_ring_finger",
            "r_pinky",
        )
        self._finger_links = tuple(
            tuple(
                link
                for link in self._robot_links
                if link.name.startswith(prefix)
            )
            for prefix in prefixes
        )

        object_material = physx.PhysxMaterial(
            static_friction=self.config.physics.object_friction,
            dynamic_friction=self.config.physics.object_friction,
            restitution=0.1,
        )
        table_material = physx.PhysxMaterial(
            static_friction=self.config.physics.table_friction,
            dynamic_friction=self.config.physics.table_friction,
            restitution=0.1,
        )
        initial_object_pose = self.reference.object_pose_position_wxyz[0]
        object_builder = self.scene.create_actor_builder()
        object_builder.set_initial_pose(
            sapien.Pose(
                p=initial_object_pose[:3].tolist(),
                q=initial_object_pose[3:].tolist(),
            )
        )
        object_builder.add_convex_collision_from_file(
            str(self.reference.object_mesh_path),
            material=object_material,
        )
        object_builder.add_visual_from_file(str(self.reference.object_mesh_path))
        points = self.reference.object_points_local_m
        extents = np.ptp(points, axis=0)
        mass = self.config.physics.object_mass_kg
        inertia = mass / 12.0 * np.asarray(
            [
                extents[1] ** 2 + extents[2] ** 2,
                extents[0] ** 2 + extents[2] ** 2,
                extents[0] ** 2 + extents[1] ** 2,
            ],
            dtype=np.float32,
        )
        object_builder.set_mass_and_inertia(
            mass,
            sapien.Pose(p=[0.0, 0.0, 0.0]),
            tuple(float(value) for value in inertia),
        )
        self.object = object_builder.build_dynamic("reference_object")

        table_center = self.reference.table_center_xy_m
        table_half = self.reference.table_half_extents_xy_m
        table_builder = self.scene.create_actor_builder()
        table_builder.set_initial_pose(
            sapien.Pose(p=[float(table_center[0]), float(table_center[1]), -0.01])
        )
        table_builder.add_box_collision(
            half_size=(float(table_half[0]), float(table_half[1]), 0.01),
            material=table_material,
        )
        table_builder.add_box_visual(
            half_size=(float(table_half[0]), float(table_half[1]), 0.01)
        )
        self.table = table_builder.build_static("table")

    def _transformed_pose(
        self,
        pose: torch.Tensor,
        env_idx: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply an episode's tabletop rigid transform to a reference pose."""

        if env_idx is None:
            env_idx = self._env_ids
        position, quaternion = pose[..., :3], pose[..., 3:]
        relative = position - self._object_anchor[env_idx]
        yaw = self._yaw_offset[env_idx, None]
        rotation = torch.stack(
            [
                relative[..., 0] * torch.cos(yaw[..., 0])
                - relative[..., 1] * torch.sin(yaw[..., 0]),
                relative[..., 0] * torch.sin(yaw[..., 0])
                + relative[..., 1] * torch.cos(yaw[..., 0]),
                relative[..., 2],
            ],
            dim=-1,
        )
        yaw_quaternion = torch.stack(
            [
                torch.cos(self._yaw_offset[env_idx] / 2.0),
                torch.zeros_like(self._yaw_offset[env_idx]),
                torch.zeros_like(self._yaw_offset[env_idx]),
                torch.sin(self._yaw_offset[env_idx] / 2.0),
            ],
            dim=-1,
        )
        return torch.cat(
            [
                self._object_anchor[env_idx]
                + self._position_offset[env_idx]
                + rotation,
                _quaternion_multiply_wxyz(yaw_quaternion, quaternion),
            ],
            dim=-1,
        )

    def _base_pose_for_step(
        self, step: torch.Tensor, env_idx: torch.Tensor | None = None
    ) -> torch.Tensor:
        pose = self._reference_base_pose[step]
        return self._transformed_pose(pose, env_idx)

    def _object_pose_for_step(
        self, step: torch.Tensor, env_idx: torch.Tensor | None = None
    ) -> torch.Tensor:
        pose = self._reference_object_pose[step]
        return self._transformed_pose(pose, env_idx)

    def _contact_state(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        finger_contact = torch.zeros(
            (self.num_envs, len(self._finger_links)),
            dtype=torch.bool,
            device=self.device,
        )
        maximum_object_force = torch.zeros(self.num_envs, device=self.device)
        for finger_index, links in enumerate(self._finger_links):
            for link in links:
                force = self.scene.get_pairwise_contact_forces(link, self.object)
                force_norm = torch.linalg.vector_norm(force, dim=-1)
                finger_contact[:, finger_index] |= (
                    force_norm > self.config.reward.contact_force_threshold_n
                )
                maximum_object_force = torch.maximum(
                    maximum_object_force, force_norm
                )

        table_force = torch.zeros(self.num_envs, device=self.device)
        for link in self._robot_links:
            force = self.scene.get_pairwise_contact_forces(link, self.table)
            table_force = torch.maximum(
                table_force, torch.linalg.vector_norm(force, dim=-1)
            )
        surface_heights = []
        for link, local_points in self._link_surface_points:
            pose = link.pose.raw_pose
            surface_heights.append(
                transform_object_points(
                    local_points, pose[:, :3], pose[:, 3:]
                )[..., 2].min(dim=-1).values
            )
        minimum_height = torch.stack(surface_heights, dim=-1).min(dim=-1).values
        return finger_contact, maximum_object_force, table_force, minimum_height

    def _task_metrics(self, step: torch.Tensor) -> dict[str, torch.Tensor]:
        observed_pose = self.object.pose.raw_pose
        reference_pose = self._object_pose_for_step(step)
        add_error = average_distance_error(
            self._object_points,
            observed_pose[:, :3],
            observed_pose[:, 3:],
            reference_pose[:, :3],
            reference_pose[:, 3:],
        )
        observed_contact, maximum_object_force, table_force, minimum_height = (
            self._contact_state()
        )
        reference_contact = self._reference_contact[step]
        reward = morphometric_reward(
            add_error,
            observed_contact,
            reference_contact,
            self.config.reward.add_alpha,
        )
        return {
            **reward,
            "add_error": add_error,
            "observed_contact": observed_contact,
            "maximum_object_contact_force": maximum_object_force,
            "table_contact_force": table_force,
            "minimum_robot_height": minimum_height,
            "table_height": self._reference_table_height[step],
        }

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict) -> None:
        randomization = self.config.randomization
        if randomization.enabled and randomization.random_start:
            start = torch.randint(
                0,
                self.reference.frame_count,
                (len(env_idx),),
                device=self.device,
            )
        else:
            start = torch.zeros_like(self._reference_step[env_idx])
        self._reference_step[env_idx] = start
        self._active_step[env_idx] = start

        anchor = self._reference_object_pose[start, :3]
        self._object_anchor[env_idx] = anchor
        if randomization.enabled:
            self._position_offset[env_idx] = torch.cat(
                [
                    torch.empty(
                        (len(env_idx), 2), device=self.device
                    ).uniform_(
                        -randomization.object_position_range_m,
                        randomization.object_position_range_m,
                    ),
                    torch.zeros((len(env_idx), 1), device=self.device),
                ],
                dim=-1,
            )
            self._yaw_offset[env_idx] = torch.empty(
                (len(env_idx),), device=self.device
            ).uniform_(
                -randomization.object_yaw_range_rad,
                randomization.object_yaw_range_rad,
            )
        else:
            self._position_offset[env_idx] = 0.0
            self._yaw_offset[env_idx] = 0.0

        start_qpos = self._reference_qpos[start]
        self._previous_action[env_idx] = start_qpos
        self._previous_qpos[env_idx] = start_qpos
        self._termination_state.reset(env_idx)

        root_pose = self._base_pose_for_step(
            self._reference_step[env_idx], env_idx
        )
        self.agent.robot.root_pose = Pose.create_from_pq(
            p=root_pose[:, :3], q=root_pose[:, 3:]
        )
        self.agent.robot.set_qpos(start_qpos)
        self.agent.robot.set_qvel(
            torch.zeros(
                (len(env_idx), len(self.agent.robot.active_joints)),
                device=self.device,
            )
        )
        object_pose = self._object_pose_for_step(
            self._reference_step[env_idx], env_idx
        )
        self.object.set_pose(
            Pose.create_from_pq(
                p=object_pose[:, :3], q=object_pose[:, 3:]
            )
        )

    def _before_control_step(self) -> None:
        self._active_step = self._reference_step.clone()
        root_pose = self._base_pose_for_step(self._active_step)
        self.agent.robot.root_pose = Pose.create_from_pq(
            p=root_pose[:, :3], q=root_pose[:, 3:]
        )
        if self.config.randomization.enabled:
            external_force = torch.empty(
                (self.num_envs, 3), device=self.device
            ).uniform_(
                -self.config.randomization.external_force_range_n,
                self.config.randomization.external_force_range_n,
            )
            if self.scene.gpu_sim_enabled:
                self.object.apply_force(external_force)
            else:
                for body_index, body in enumerate(self.object._bodies):
                    body.add_force_at_point(
                        force=external_force[body_index].detach().cpu().numpy(),
                        point=(body.pose * body.cmass_local_pose).p,
                    )

    def _after_control_step(self) -> None:
        metrics = self._task_metrics(self._active_step)
        update_termination_state(
            self._termination_state,
            metrics["add_error"],
            metrics["contact_mismatch"],
            self.config.termination.ema_decay,
        )
        self._reference_step = torch.clamp(
            self._reference_step + 1, max=self.reference.frame_count - 1
        )

    def step(
        self, action: np.ndarray | torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        result = super().step(action)
        observation, reward, terminated, truncated, info = result
        truncated = torch.logical_or(
            truncated, self._active_step >= self.reference.frame_count - 1
        )
        return observation, reward, terminated, truncated, info

    def evaluate(self) -> dict[str, torch.Tensor]:
        metrics = self._task_metrics(self._active_step)
        reasons = termination_reasons(
            self._termination_state,
            metrics["minimum_robot_height"],
            metrics["table_height"],
            metrics["maximum_object_contact_force"],
            metrics["table_contact_force"],
            self.config.termination,
        )
        success = (
            (self._active_step >= self.reference.frame_count - 1)
            & ~reasons["fail"]
            & metrics["observed_contact"].any(dim=-1)
            & (metrics["add_error"] <= self.config.reward.final_add_success_m)
        )
        return {
            **metrics,
            **reasons,
            "add_ema": self._termination_state.add_ema.clone(),
            "contact_mismatch_ema": (
                self._termination_state.contact_mismatch_ema.clone()
            ),
            "table_clearance": (
                metrics["minimum_robot_height"] - metrics["table_height"]
            ),
            "success": success,
        }

    def _get_obs_agent(self) -> dict[str, torch.Tensor]:
        return {"qpos": self.agent.robot.get_qpos()}

    def _get_obs_extra(self, info: dict) -> dict[str, torch.Tensor]:
        if "observed_contact" in info:
            observed_contact = info["observed_contact"]
            minimum_height = info["minimum_robot_height"]
        else:
            observed_contact, _, _, minimum_height = self._contact_state()
        reference_qpos_window = future_reference_window(
            self._reference_qpos,
            self._reference_step,
            self.config.control.future_steps,
        )
        reference_contact_window = future_reference_window(
            self._reference_contact,
            self._reference_step,
            self.config.control.future_steps,
        )
        observation = privileged_teacher_observation(
            qpos=self.agent.robot.get_qpos(),
            object_pose=self.object.pose.raw_pose,
            reference_object_pose=self._object_pose_for_step(
                self._reference_step
            ),
            reference_qpos_window=reference_qpos_window,
            observed_contact=observed_contact,
            reference_contact_window=reference_contact_window,
            minimum_robot_height=minimum_height,
            table_height=self._reference_table_height[self._reference_step],
            object_scale=self._object_scale,
            object_mass=self._object_mass,
            object_friction=self._object_friction,
            object_center_of_mass=self._object_center_of_mass,
            table_friction=self._table_friction,
            previous_action=self._previous_action,
            previous_qpos=self._previous_qpos,
        )
        observation.pop("qpos")
        return observation

    def compute_dense_reward(
        self,
        obs: Any,
        action: torch.Tensor,
        info: dict,
    ) -> torch.Tensor:
        return info["reward"]

    def compute_normalized_dense_reward(
        self,
        obs: Any,
        action: torch.Tensor,
        info: dict,
    ) -> torch.Tensor:
        return info["reward"]

    @property
    def current_reference_qpos(self) -> torch.Tensor:
        return self._reference_qpos[self._reference_step]


class ResidualActionWrapper(gym.Wrapper):
    """Convert bounded residual actions into paper Eq. q_cmd = q_ref + delta q."""

    def __init__(self, env: MorphometricResidualEnv) -> None:
        super().__init__(env)
        self.env = env
        residual_limit = env.config.control.residual_limit_rad
        self.action_space = gym.spaces.Box(
            low=-residual_limit,
            high=residual_limit,
            shape=env.action_space.shape,
            dtype=np.float32,
        )

    def step(
        self, action: np.ndarray | torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        base_env = self.env.unwrapped
        residual = torch.as_tensor(action, device=base_env.device)
        if residual.ndim == 1:
            residual = residual.unsqueeze(0)
        residual = torch.clamp(
            residual,
            -base_env.config.control.residual_limit_rad,
            base_env.config.control.residual_limit_rad,
        )
        q_command = base_env.current_reference_qpos + residual
        q_command = torch.clamp(
            q_command, base_env._joint_lower, base_env._joint_upper
        )
        base_env._previous_qpos = base_env.agent.robot.get_qpos().clone()
        base_env._previous_action = q_command.clone()
        return self.env.step(q_command)
