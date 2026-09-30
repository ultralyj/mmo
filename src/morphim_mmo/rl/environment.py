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
from scipy.spatial.transform import Rotation

from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import SimConfig

from .agent import WujiHand2Agent
from .config import ResidualRLConfig
from .observation import future_reference_window, privileged_teacher_observation
from .reference import ResidualRLReference
from .reward import average_distance_error, morphometric_reward
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


def _angular_velocity_between_wxyz(
    start: torch.Tensor, target: torch.Tensor, duration_s: float
) -> torch.Tensor:
    """Return the constant world-frame angular velocity joining two poses."""

    if duration_s <= 0:
        raise ValueError("quaternion interpolation duration must be positive")
    start = torch.nn.functional.normalize(start, dim=-1)
    target = torch.nn.functional.normalize(target, dim=-1)
    target = torch.where(
        (start * target).sum(dim=-1, keepdim=True) < 0, -target, target
    )
    conjugate = torch.cat([start[..., :1], -start[..., 1:]], dim=-1)
    delta = _quaternion_multiply_wxyz(target, conjugate)
    vector = delta[..., 1:]
    vector_norm = torch.linalg.vector_norm(vector, dim=-1, keepdim=True)
    angle = 2.0 * torch.atan2(vector_norm, delta[..., :1].clamp_min(1e-12))
    axis = vector / vector_norm.clamp_min(1e-12)
    return axis * (angle / duration_s)


def _grasp_bias_activation(
    reference_contact: torch.Tensor,
    lookahead_steps: int,
    minimum_fingers: int,
    release_decay: float,
) -> torch.Tensor:
    """Build a smooth pre-contact gate for a robot-specific grasp correction."""

    if lookahead_steps < 0:
        raise ValueError("grasp contact lookahead must be non-negative")
    if minimum_fingers <= 0:
        raise ValueError("minimum grasp finger count must be positive")
    if not 0.0 <= release_decay < 1.0:
        raise ValueError("grasp release decay must be in [0, 1)")
    active = reference_contact.sum(dim=-1) >= minimum_fingers
    gate = active.to(torch.float32)
    for offset in range(1, lookahead_steps + 1):
        gate[:-offset] = torch.maximum(
            gate[:-offset], active[offset:].to(torch.float32)
        )
    for frame in range(1, len(gate)):
        gate[frame] = torch.maximum(gate[frame], gate[frame - 1] * release_decay)
    return gate


def _mesh_mass_properties(
    mesh: trimesh.Trimesh, mass_kg: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return center, principal inertia, and principal-frame wxyz rotation."""

    if mass_kg <= 0:
        raise ValueError("object mass must be positive")
    if not mesh.is_watertight or not np.isfinite(mesh.volume) or mesh.volume <= 0:
        raise ValueError("object mesh must be watertight with positive volume")
    inertia_tensor = np.asarray(mesh.moment_inertia) * (mass_kg / mesh.volume)
    principal_inertia, principal_axes = trimesh.inertia.principal_axis(
        inertia_tensor
    )
    if np.linalg.det(principal_axes) < 0:
        principal_axes[-1] *= -1
    principal_to_mesh = principal_axes.T
    quaternion_xyzw = Rotation.from_matrix(principal_to_mesh).as_quat()
    quaternion_wxyz = quaternion_xyzw[[3, 0, 1, 2]]
    return (
        np.asarray(mesh.center_mass, dtype=np.float32),
        np.asarray(principal_inertia, dtype=np.float32),
        np.asarray(quaternion_wxyz, dtype=np.float32),
    )


class MorphometricResidualEnv(BaseEnv):
    """A hand-only environment whose fixed-base wrist follows an MMO reference."""

    SUPPORTED_ROBOTS = [WujiHand2Agent]
    agent: WujiHand2Agent

    @property
    def _default_human_render_camera_configs(self) -> CameraConfig:
        """Return a camera framing the hand, apple, and tabletop."""

        pose = sapien_utils.look_at(
            eye=[0.48, -0.92, 0.38], target=[0.0, -0.47, 0.12]
        )
        return CameraConfig(
            "render_camera", pose, 640, 480, 1.0, 0.01, 10.0
        )

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
        self._root_pose_start: torch.Tensor | None = None
        self._root_pose_target: torch.Tensor | None = None
        self._root_substep = 0
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
            enhanced_determinism=True,
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
        ).clone()
        self._reference_object_pose = torch.as_tensor(
            self.reference.object_pose_position_wxyz,
            dtype=torch.float32,
            device=device,
        )
        self._reference_contact = torch.as_tensor(
            self.reference.reference_contact, dtype=torch.bool, device=device
        )
        grasp_bias = torch.as_tensor(
            self.config.control.grasp_bias_rad,
            dtype=torch.float32,
            device=device,
        )
        if grasp_bias.shape != (self._reference_qpos.shape[-1],):
            raise ValueError(
                "grasp_bias_rad must contain one value per active robot joint"
            )
        grasp_gate = _grasp_bias_activation(
            self._reference_contact,
            self.config.control.grasp_contact_lookahead_steps,
            self.config.control.grasp_min_contact_fingers,
            self.config.control.grasp_release_decay,
        )
        self._reference_qpos += grasp_gate[:, None] * grasp_bias[None, :]
        wrist_bias = torch.as_tensor(self.config.control.wrist_bias_m, device=device)
        if wrist_bias.shape != (3,):
            raise ValueError("wrist_bias_m must have three entries")
        self._reference_base_pose[:, :3] += grasp_gate[:, None] * wrist_bias
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
        self._initial_object_height = torch.zeros(self.num_envs, device=device)
        self._termination_state = TerminationState.zeros(self.num_envs, device)
        self._robot_links = tuple(self.agent.robot.links)
        urdf = yourdfpy.URDF.load(
            self.agent.urdf_path,
            load_meshes=False,
            load_collision_meshes=True,
        )
        surface_links = []
        surface_points = []
        for link_index, link in enumerate(self._robot_links):
            mesh = RobotCollision._get_trimesh_collision_geometries(
                urdf, link.name
            )
            if mesh.is_empty or mesh.area <= 0:
                continue
            sampled, _ = trimesh.sample.sample_surface(
                mesh, 16, seed=link_index
            )
            surface_links.append(link)
            surface_points.append(
                torch.as_tensor(sampled, dtype=torch.float32, device=device)
            )
        self._surface_links = tuple(surface_links)
        self._link_surface_points = torch.stack(surface_points)
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
        mass = self.config.physics.object_mass_kg
        object_mesh = trimesh.load_mesh(
            self.reference.object_mesh_path, process=False
        )
        if not isinstance(object_mesh, trimesh.Trimesh):
            object_mesh = trimesh.util.concatenate(
                tuple(object_mesh.geometry.values())
            )
        center_of_mass, inertia, inertia_quaternion = _mesh_mass_properties(
            object_mesh, mass
        )
        object_builder.set_mass_and_inertia(
            mass,
            sapien.Pose(
                p=center_of_mass.tolist(), q=inertia_quaternion.tolist()
            ),
            tuple(float(value) for value in inertia),
        )
        self.object = object_builder.build_dynamic("reference_object")
        self._object_center_of_mass[:] = torch.as_tensor(
            center_of_mass, dtype=torch.float32, device=device
        )

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
        self._randomize_scene_physics()
        if self.scene.gpu_sim_enabled:
            object_pairs = []
            table_pairs = []
            for link in self._robot_links:
                object_pairs.extend(zip(link._bodies, self.object._bodies))
                table_pairs.extend(zip(link._bodies, self.table._bodies))
            self._object_contact_query = (
                self.scene.px.gpu_create_contact_pair_impulse_query(object_pairs)
            )
            self._table_contact_query = (
                self.scene.px.gpu_create_contact_pair_impulse_query(table_pairs)
            )
            link_to_index = {
                link.name: index for index, link in enumerate(self._robot_links)
            }
            self._finger_link_indices = tuple(
                torch.as_tensor(
                    [link_to_index[link.name] for link in links],
                    dtype=torch.int64,
                    device=device,
                )
                for links in self._finger_links
            )

    def _randomize_scene_physics(self) -> None:
        """Sample physical parameters before PhysX GPU buffers are initialized.

        Parameters persist per scene; episode position/yaw and forces are
        resampled separately. Rebuilding with a new seed samples new physics.
        """
        randomization = self.config.randomization
        if not randomization.enabled:
            return
        rng = np.random.default_rng(self.config.ppo.seed)
        for bounds in (randomization.mass_scale_range,
                       randomization.friction_scale_range,
                       randomization.pd_gain_scale_range):
            if len(bounds) != 2 or not 0 < bounds[0] <= bounds[1]:
                raise ValueError("physical randomization ranges must be positive ordered pairs")
        for index, body in enumerate(self.object._bodies):
            mass_scale = rng.uniform(*randomization.mass_scale_range)
            friction_scale = rng.uniform(*randomization.friction_scale_range)
            gain_scale = rng.uniform(*randomization.pd_gain_scale_range)
            body.mass *= mass_scale
            body.inertia = np.asarray(body.inertia) * mass_scale
            self._object_mass[index] = body.mass
            self._object_friction[index] = self.config.physics.object_friction * friction_scale
            material = physx.PhysxMaterial(
                static_friction=float(self._object_friction[index]),
                dynamic_friction=float(self._object_friction[index]), restitution=0.0)
            for shape in body.collision_shapes:
                shape.physical_material = material
            articulation = self.agent.robot._objs[index]
            hand_friction = self.config.physics.hand_friction * rng.uniform(*randomization.friction_scale_range)
            hand_material = physx.PhysxMaterial(static_friction=hand_friction, dynamic_friction=hand_friction, restitution=0.0)
            for link in articulation.links:
                for shape in link.collision_shapes:
                    shape.physical_material = hand_material
            for joint in articulation.active_joints:
                joint.set_drive_properties(self.config.control.stiffness * gain_scale,
                                           self.config.control.damping * gain_scale,
                                           force_limit=self.config.control.force_limit)

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
        if self.scene.gpu_sim_enabled:
            link_count = len(self._robot_links)
            self.scene.px.gpu_query_contact_pair_impulses(
                self._object_contact_query
            )
            object_forces = (
                self._object_contact_query.cuda_impulses.torch()
                .clone()
                .reshape(link_count, self.num_envs, 3)
                .transpose(0, 1)
                / self.scene.px.timestep
            )
            object_force_norm = torch.linalg.vector_norm(object_forces, dim=-1)
            finger_contact = torch.stack(
                [
                    object_force_norm[:, indices].amax(dim=-1)
                    > self.config.reward.contact_force_threshold_n
                    for indices in self._finger_link_indices
                ],
                dim=-1,
            )
            maximum_object_force = object_force_norm.amax(dim=-1)

            self.scene.px.gpu_query_contact_pair_impulses(
                self._table_contact_query
            )
            table_forces = (
                self._table_contact_query.cuda_impulses.torch()
                .clone()
                .reshape(link_count, self.num_envs, 3)
                .transpose(0, 1)
                / self.scene.px.timestep
            )
            table_force = torch.linalg.vector_norm(
                table_forces, dim=-1
            ).amax(dim=-1)
        else:
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

        poses = torch.stack(
            [link.pose.raw_pose for link in self._surface_links], dim=1
        )
        points = self._link_surface_points[None]
        quaternion = poses[..., 3:]
        quaternion = quaternion / torch.linalg.vector_norm(
            quaternion, dim=-1, keepdim=True
        ).clamp_min(1e-12)
        vector = quaternion[..., None, 1:]
        first_cross = torch.linalg.cross(vector, points, dim=-1)
        rotated = points + 2.0 * (
            quaternion[..., None, :1] * first_cross
            + torch.linalg.cross(vector, first_cross, dim=-1)
        )
        world_points = rotated + poses[..., None, :3]
        minimum_height = world_points[..., 2].amin(dim=(-1, -2))
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
            "object_height": observed_pose[:, 2],
            "object_lift": observed_pose[:, 2] - self._initial_object_height,
            "reference_object_height": reference_pose[:, 2],
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
        self._root_pose_start = root_pose.clone()
        self._root_pose_target = root_pose.clone()
        self.agent.robot.set_qpos(start_qpos)
        self.agent.robot.set_qvel(
            torch.zeros(
                (len(env_idx), len(self.agent.robot.active_joints)),
                device=self.device,
            )
        )
        self.agent.robot.set_root_linear_velocity(torch.zeros((len(env_idx), 3), device=self.device))
        self.agent.robot.set_root_angular_velocity(torch.zeros((len(env_idx), 3), device=self.device))
        object_pose = self._object_pose_for_step(
            self._reference_step[env_idx], env_idx
        )
        self.object.set_pose(
            Pose.create_from_pq(
                p=object_pose[:, :3], q=object_pose[:, 3:]
            )
        )

        self._initial_object_height[env_idx] = object_pose[:, 2]
        # Pose reset does not clear dynamic-body momentum in SAPIEN.
        self.object.set_linear_velocity(torch.zeros((len(env_idx), 3), device=self.device))
        self.object.set_angular_velocity(torch.zeros((len(env_idx), 3), device=self.device))

    def _before_control_step(self) -> None:
        self._active_step = self._reference_step.clone()
        self._root_pose_start = self.agent.robot.root_pose.raw_pose.clone()
        self._root_pose_target = self._base_pose_for_step(self._active_step)
        self._root_substep = 0

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

    def _before_simulation_step(self) -> None:
        """Interpolate the kinematic wrist at the physics simulation rate."""

        self._root_substep += 1
        fraction = self._root_substep / self._sim_steps_per_control
        start = self._root_pose_start
        target = self._root_pose_target
        position = torch.lerp(start[:, :3], target[:, :3], fraction)
        start_quaternion = start[:, 3:]
        target_quaternion = target[:, 3:]
        target_quaternion = torch.where(
            (start_quaternion * target_quaternion).sum(dim=-1, keepdim=True) < 0,
            -target_quaternion,
            target_quaternion,
        )
        quaternion = torch.nn.functional.normalize(
            torch.lerp(start_quaternion, target_quaternion, fraction), dim=-1
        )
        root_pose = torch.cat([position, quaternion], dim=-1)
        self.agent.robot.root_pose = Pose.create_from_pq(
            p=root_pose[:, :3], q=root_pose[:, 3:]
        )
        control_duration = 1.0 / self.config.control.control_frequency_hz
        self.agent.robot.root_linear_velocity = (
            target[:, :3] - start[:, :3]
        ) / control_duration
        self.agent.robot.root_angular_velocity = _angular_velocity_between_wxyz(
            start[:, 3:], target[:, 3:], control_duration
        )
        if self.scene.gpu_sim_enabled:
            # Root-pose writes occur after BaseEnv's controller-target upload.
            self.scene.px.gpu_apply_articulation_root_pose()
            self.scene.px.gpu_apply_articulation_root_velocity()

    def _after_control_step(self) -> None:
        # BaseEnv fetches GPU state only after this hook; EMA must use this step.
        if self.scene.gpu_sim_enabled:
            self.scene._gpu_fetch_all()
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
        # BaseEnv treats success as termination. Here only failures terminate;
        # reaching the reference horizon is an explicit task truncation.
        terminated = info["fail"].clone()
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
            "reference_step": self._active_step.clone(),
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
