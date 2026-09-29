"""Skeleton heat diffusion and MMO linear-blend retargeting (equations 11--12)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from .urdf import UrdfModel


@dataclass(frozen=True)
class RobotSkeleton:
    """Robot skeleton nodes, hierarchy, rest geometry, and orientation hints."""

    node_names: tuple[str, ...]
    link_names: tuple[str, ...]
    rest_positions: np.ndarray
    rest_link_rotations: np.ndarray
    parent_indices: np.ndarray
    orientation_child_indices: np.ndarray
    edges: np.ndarray


def build_robot_skeleton(
    model: UrdfModel,
    wrist_link: str,
    tip_joints: Mapping[str, str],
    orientation_root_joint: str,
) -> RobotSkeleton:
    """Build wrist + actuated-joint + fingertip nodes from a hand URDF."""

    selected_joints = [*model.actuated_joints]
    for joint_name in tip_joints.values():
        selected_joints.append(model.joint_by_name[joint_name])
    node_names = (wrist_link, *(joint.name for joint in selected_joints))
    link_names = (wrist_link, *(joint.child for joint in selected_joints))
    node_by_joint = {joint.name: index + 1 for index, joint in enumerate(selected_joints)}
    joint_from_child = {joint.child: joint for joint in model.joints}

    parent_indices = [-1]
    for joint in selected_joints:
        link = joint.parent
        while True:
            if link == wrist_link:
                parent_indices.append(0)
                break
            parent_joint = joint_from_child.get(link)
            if parent_joint is None:
                raise ValueError(f"cannot connect {joint.name!r} to wrist {wrist_link!r}")
            if parent_joint.name in node_by_joint:
                parent_indices.append(node_by_joint[parent_joint.name])
                break
            link = parent_joint.parent
    parent_indices_array = np.asarray(parent_indices, dtype=np.int32)
    edges = np.asarray(
        [(parent, child) for child, parent in enumerate(parent_indices) if parent >= 0],
        dtype=np.int32,
    )

    link_transforms, joint_positions = model.forward_kinematics()
    rest_positions = np.asarray(
        [
            link_transforms[wrist_link][:3, 3],
            *(joint_positions[joint.name] for joint in selected_joints),
        ],
        dtype=np.float64,
    )
    rest_link_rotations = np.asarray(
        [link_transforms[name][:3, :3] for name in link_names], dtype=np.float64
    )

    children: list[list[int]] = [[] for _ in node_names]
    for child, parent in enumerate(parent_indices_array):
        if parent >= 0:
            children[int(parent)].append(child)
    orientation_children = np.empty(len(node_names), dtype=np.int32)
    root_child = node_by_joint[orientation_root_joint]
    for index, candidates in enumerate(children):
        if index == 0:
            orientation_children[index] = root_child
        elif candidates:
            orientation_children[index] = candidates[0]
        else:
            orientation_children[index] = index

    return RobotSkeleton(
        node_names=node_names,
        link_names=link_names,
        rest_positions=rest_positions,
        rest_link_rotations=rest_link_rotations,
        parent_indices=parent_indices_array,
        orientation_child_indices=orientation_children,
        edges=edges,
    )


def one_dimensional_laplacian(
    points: np.ndarray, edges: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return the 1D finite-element stiffness matrix and lumped mass.

    Edge stiffness ``1 / length`` is the curve/skeleton analogue of the
    cotangent Laplacian used on triangle meshes.
    """

    points = np.asarray(points, dtype=np.float64)
    edges = np.asarray(edges, dtype=np.int32)
    count = len(points)
    laplacian = np.zeros((count, count), dtype=np.float64)
    mass = np.zeros(count, dtype=np.float64)
    for left, right in edges:
        length = float(np.linalg.norm(points[left] - points[right]))
        if length <= 1e-9:
            raise ValueError("skeleton edges must have positive length")
        stiffness = 1.0 / length
        laplacian[left, left] += stiffness
        laplacian[right, right] += stiffness
        laplacian[left, right] -= stiffness
        laplacian[right, left] -= stiffness
        mass[left] += length / 2.0
        mass[right] += length / 2.0
    if np.any(mass <= 0):
        raise ValueError("every skeleton node must be incident to an edge")
    return laplacian, mass


def heat_diffusion_weights(
    skeleton_points: np.ndarray,
    edges: np.ndarray,
    influence_points: np.ndarray,
    *,
    source_sigma_m: float = 0.012,
    diffusion_scale: float = 0.05,
) -> np.ndarray:
    """Diffuse proximity seeds over a 1D skeleton and row-normalize weights."""

    if source_sigma_m <= 0 or diffusion_scale < 0:
        raise ValueError("source_sigma_m must be positive and diffusion_scale non-negative")
    points = np.asarray(skeleton_points, dtype=np.float64)
    influences = np.asarray(influence_points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("skeleton_points must have shape (N, 3)")
    if influences.ndim != 2 or influences.shape[1] != 3:
        raise ValueError("influence_points must have shape (K, 3)")
    laplacian, mass = one_dimensional_laplacian(points, edges)
    squared_distances = np.sum(
        (points[:, None, :] - influences[None, :, :]) ** 2, axis=-1
    )
    seeds = np.exp(-squared_distances / (2.0 * source_sigma_m**2))
    seeds /= np.maximum(np.sum(seeds, axis=1, keepdims=True), 1e-12)
    edge_lengths = np.linalg.norm(points[edges[:, 0]] - points[edges[:, 1]], axis=1)
    diffusion_time = diffusion_scale * float(np.median(edge_lengths) ** 2)
    system = np.diag(mass) + diffusion_time * laplacian
    weights = np.linalg.solve(system, mass[:, None] * seeds)
    weights = np.maximum(weights, 0.0)
    weights /= np.maximum(np.sum(weights, axis=1, keepdims=True), 1e-12)
    return weights.astype(np.float32)


def linear_blend_targets(
    rest_positions: np.ndarray,
    weights: np.ndarray,
    influence_transforms: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply equations 11--12 and return positions plus blended linear maps."""

    rest_positions = np.asarray(rest_positions)
    weights = np.asarray(weights)
    influence_transforms = np.asarray(influence_transforms)
    blended = np.einsum("nk,kab->nab", weights, influence_transforms)
    homogeneous = np.concatenate(
        [rest_positions, np.ones((len(rest_positions), 1), dtype=rest_positions.dtype)],
        axis=1,
    )
    positions = np.einsum("nab,nb->na", blended, homogeneous)[:, :3]
    return positions, blended[:, :3, :3]


def target_link_poses(
    skeleton: RobotSkeleton,
    target_positions: np.ndarray,
    blended_linear_maps: np.ndarray,
) -> np.ndarray:
    """Construct calibrated link poses using z-to-child and Gram--Schmidt."""

    target_positions = np.asarray(target_positions, dtype=np.float64)
    blended_linear_maps = np.asarray(blended_linear_maps, dtype=np.float64)
    count = len(skeleton.node_names)
    poses = np.tile(np.eye(4), (count, 1, 1))
    poses[:, :3, 3] = target_positions
    for index in range(count):
        child = int(skeleton.orientation_child_indices[index])
        if child == index:
            parent = int(skeleton.parent_indices[index])
            rest_z = skeleton.rest_positions[index] - skeleton.rest_positions[parent]
            target_z = target_positions[index] - target_positions[parent]
        else:
            rest_z = skeleton.rest_positions[child] - skeleton.rest_positions[index]
            target_z = target_positions[child] - target_positions[index]
        rest_z /= np.linalg.norm(rest_z)
        target_z /= np.linalg.norm(target_z)

        rest_x_hint = skeleton.rest_link_rotations[index, :, 0]
        rest_x = rest_x_hint - np.dot(rest_x_hint, rest_z) * rest_z
        if np.linalg.norm(rest_x) < 1e-7:
            rest_x_hint = skeleton.rest_link_rotations[index, :, 1]
            rest_x = rest_x_hint - np.dot(rest_x_hint, rest_z) * rest_z
        rest_x /= np.linalg.norm(rest_x)
        rest_y = np.cross(rest_z, rest_x)
        rest_frame = np.column_stack([rest_x, rest_y, rest_z])
        calibration = rest_frame.T @ skeleton.rest_link_rotations[index]

        target_x_hint = blended_linear_maps[index] @ rest_x
        target_x = target_x_hint - np.dot(target_x_hint, target_z) * target_z
        if np.linalg.norm(target_x) < 1e-7:
            target_x_hint = blended_linear_maps[index] @ rest_y
            target_x = target_x_hint - np.dot(target_x_hint, target_z) * target_z
        target_x /= np.linalg.norm(target_x)
        target_y = np.cross(target_z, target_x)
        target_frame = np.column_stack([target_x, target_y, target_z])
        poses[index, :3, :3] = target_frame @ calibration
    return poses


def relative_influence_transforms(
    current_transforms: np.ndarray,
    reference_transforms: np.ndarray,
    influence_bone_indices: Sequence[int],
) -> np.ndarray:
    """Build the relative MANO transforms used in equation 11."""

    current = np.asarray(current_transforms)[np.asarray(influence_bone_indices)]
    reference = np.asarray(reference_transforms)[np.asarray(influence_bone_indices)]
    return current @ np.linalg.inv(reference)
