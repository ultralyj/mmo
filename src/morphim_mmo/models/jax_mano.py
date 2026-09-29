"""Differentiable single-frame MANO linear-blend skinning in JAX."""

from __future__ import annotations

import jax.numpy as jnp
from collections.abc import Sequence


def axis_angle_to_matrix(rotvecs: jnp.ndarray) -> jnp.ndarray:
    """Convert axis-angle vectors to rotation matrices, matching SMPL-X."""

    rotvecs = jnp.asarray(rotvecs)
    angles = jnp.linalg.norm(rotvecs + 1e-8, axis=-1, keepdims=True)
    directions = rotvecs / angles
    x, y, z = jnp.moveaxis(directions, -1, 0)
    zeros = jnp.zeros_like(x)
    skew = jnp.stack(
        [zeros, -z, y, z, zeros, -x, -y, x, zeros], axis=-1
    ).reshape(rotvecs.shape[:-1] + (3, 3))
    identity = jnp.broadcast_to(jnp.eye(3, dtype=rotvecs.dtype), skew.shape)
    sin = jnp.sin(angles)[..., None]
    cosine = jnp.cos(angles)[..., None]
    return identity + sin * skew + (1.0 - cosine) * (skew @ skew)


def mano_lbs(
    template: jnp.ndarray,
    pose_blends: jnp.ndarray,
    joint_regressor: jnp.ndarray,
    blend_weights: jnp.ndarray,
    parents: Sequence[int],
    global_orient: jnp.ndarray,
    hand_pose: jnp.ndarray,
    translation: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Pose one beta-zero MANO template and return vertices, joints, rotations."""

    vertices, joints, rotations, _ = mano_lbs_with_transforms(
        template,
        pose_blends,
        joint_regressor,
        blend_weights,
        parents,
        global_orient,
        hand_pose,
        translation,
    )
    return vertices, joints, rotations


def mano_lbs_with_transforms(
    template: jnp.ndarray,
    pose_blends: jnp.ndarray,
    joint_regressor: jnp.ndarray,
    blend_weights: jnp.ndarray,
    parents: Sequence[int],
    global_orient: jnp.ndarray,
    hand_pose: jnp.ndarray,
    translation: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Pose MANO and also return its 16 rigid rest-to-world transforms.

    The returned transforms are the same corrected transforms used by MANO
    linear-blend skinning, with global translation included.  Consequently,
    ``A_current @ inv(A_reference)`` maps a point from a reference MANO pose to
    the current pose, as required by MMO equation 11.
    """

    template = jnp.asarray(template)
    pose_blends = jnp.asarray(pose_blends)
    joint_regressor = jnp.asarray(joint_regressor)
    blend_weights = jnp.asarray(blend_weights)
    parent_indices = tuple(int(parent) for parent in parents)
    parents_array = jnp.asarray(parent_indices)
    full_pose = jnp.concatenate([global_orient.reshape(1, 3), hand_pose.reshape(15, 3)])
    rotations = axis_angle_to_matrix(full_pose)
    pose_feature = (rotations[1:] - jnp.eye(3, dtype=template.dtype)).reshape(-1)
    posed_vertices = template + jnp.einsum("p,pvc->vc", pose_feature, pose_blends)
    rest_joints = joint_regressor @ template

    relative_joints = jnp.concatenate(
        [
            rest_joints[:1],
            rest_joints[1:] - rest_joints[parents_array[1:]],
        ],
        axis=0,
    )
    transform_top = jnp.concatenate(
        [rotations, relative_joints[..., None]], axis=2
    )
    transform_bottom = jnp.broadcast_to(
        jnp.asarray([0.0, 0.0, 0.0, 1.0], dtype=template.dtype),
        (len(parent_indices), 1, 4),
    )
    local_transforms = jnp.concatenate([transform_top, transform_bottom], axis=1)

    transform_chain = [local_transforms[0]]
    for index in range(1, len(parent_indices)):
        transform_chain.append(
            transform_chain[parent_indices[index]] @ local_transforms[index]
        )
    transforms = jnp.stack(transform_chain)
    posed_joints = transforms[:, :3, 3]

    rest_homogeneous = jnp.concatenate(
        [rest_joints, jnp.zeros((len(parent_indices), 1), dtype=template.dtype)], axis=1
    )
    transformed_rest = (transforms @ rest_homogeneous[..., None])[..., 0]
    correction = jnp.zeros_like(transforms).at[:, :, 3].set(transformed_rest)
    relative_transforms = transforms - correction
    vertex_transforms = jnp.einsum("vj,jab->vab", blend_weights, relative_transforms)
    posed_homogeneous = jnp.concatenate(
        [posed_vertices, jnp.ones((len(posed_vertices), 1), dtype=template.dtype)], axis=1
    )
    vertices = (vertex_transforms @ posed_homogeneous[..., None])[:, :3, 0]
    world_transforms = relative_transforms.at[:, :3, 3].add(translation)
    return vertices + translation, posed_joints + translation, rotations, world_transforms
