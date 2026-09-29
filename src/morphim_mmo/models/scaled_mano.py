"""Scaled-MANO equations (paper equations 5 and 6).

This module only transforms the MANO model arrays. Posing/LBS is deliberately a
separate backend because the licensed MANO model archive is not part of the repo.
All functions use JAX arrays so the six morphology scales remain differentiable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import jax.numpy as jnp
import numpy as np


@dataclass(frozen=True)
class FingerSpec:
    """MANO indices needed to scale one anatomical finger."""

    name: str
    mcp_joint_index: int
    skinning_joint_indices: tuple[int, ...]


@dataclass(frozen=True)
class ScaledManoArrays:
    """Differentiable MANO arrays after palm and finger scaling."""

    template: jnp.ndarray
    shape_blends: jnp.ndarray
    pose_blends: jnp.ndarray
    rest_joints: jnp.ndarray


def _validate_model_arrays(
    template: jnp.ndarray,
    joint_regressor: jnp.ndarray,
    blend_weights: jnp.ndarray,
) -> None:
    """Validate model-array dimensions required by the scaling equations."""

    if template.ndim != 2 or template.shape[1] != 3:
        raise ValueError(f"template must have shape (V, 3), got {template.shape}")
    if joint_regressor.ndim != 2 or joint_regressor.shape[1] != template.shape[0]:
        raise ValueError("joint_regressor must have shape (J, V)")
    if blend_weights.shape != (template.shape[0], joint_regressor.shape[0]):
        raise ValueError("blend_weights must have shape (V, J)")


def scale_mano_arrays(
    template: jnp.ndarray,
    shape_blends: jnp.ndarray,
    pose_blends: jnp.ndarray,
    joint_regressor: jnp.ndarray,
    blend_weights: jnp.ndarray,
    palm_scale: float | jnp.ndarray,
    finger_scales: Mapping[str, float | jnp.ndarray],
    finger_specs: Sequence[FingerSpec],
    wrist_joint_index: int = 0,
) -> ScaledManoArrays:
    """Apply the paper's two-stage palm/finger scaling to MANO arrays.

    `shape_blends` and `pose_blends` may have arbitrary leading dimensions but
    must end in `(V, 3)`. The finger updates are accumulated from the same
    palm-scaled arrays, matching Eq. (5) and avoiding ordering artifacts where
    soft membership weights overlap.
    """

    template = jnp.asarray(template)
    shape_blends = jnp.asarray(shape_blends)
    pose_blends = jnp.asarray(pose_blends)
    joint_regressor = jnp.asarray(joint_regressor)
    blend_weights = jnp.asarray(blend_weights)
    _validate_model_arrays(template, joint_regressor, blend_weights)

    if shape_blends.shape[-2:] != template.shape:
        raise ValueError("shape_blends must end in (V, 3)")
    if pose_blends.shape[-2:] != template.shape:
        raise ValueError("pose_blends must end in (V, 3)")
    if set(finger_scales) != {spec.name for spec in finger_specs}:
        raise ValueError("finger_scales keys must exactly match finger_specs names")

    palm_scale = jnp.asarray(palm_scale, dtype=template.dtype)
    wrist = joint_regressor[wrist_joint_index] @ template
    palm_template = wrist + palm_scale * (template - wrist)
    palm_shape_blends = palm_scale * shape_blends
    palm_pose_blends = palm_scale * pose_blends

    template_delta = jnp.zeros_like(palm_template)
    blend_multiplier = jnp.ones((template.shape[0],), dtype=template.dtype)
    for spec in finger_specs:
        membership = jnp.sum(
            blend_weights[:, jnp.asarray(spec.skinning_joint_indices)], axis=1
        )
        mcp = joint_regressor[spec.mcp_joint_index] @ palm_template
        alpha_minus_one = jnp.asarray(
            finger_scales[spec.name], dtype=template.dtype
        ) / palm_scale - 1.0
        template_delta = template_delta + (
            membership[:, None] * alpha_minus_one * (palm_template - mcp)
        )
        blend_multiplier = blend_multiplier + membership * alpha_minus_one

    scaled_template = palm_template + template_delta
    broadcast_shape = (1,) * (shape_blends.ndim - 2) + (template.shape[0], 1)
    multiplier = blend_multiplier.reshape(broadcast_shape)
    scaled_shape_blends = palm_shape_blends * multiplier
    scaled_pose_blends = palm_pose_blends * multiplier
    rest_joints = joint_regressor @ scaled_template
    return ScaledManoArrays(
        template=scaled_template,
        shape_blends=scaled_shape_blends,
        pose_blends=scaled_pose_blends,
        rest_joints=rest_joints,
    )


def initial_scale_ratios(
    mano_joints: np.ndarray,
    mano_tips: Mapping[str, np.ndarray],
    robot_wrist: np.ndarray,
    robot_mcps: Mapping[str, np.ndarray],
    robot_tips: Mapping[str, np.ndarray],
    finger_specs: Sequence[FingerSpec],
    wrist_joint_index: int = 0,
) -> tuple[float, dict[str, float]]:
    """Compute the URDF-derived initialization described in Section IV-A."""

    names = [spec.name for spec in finger_specs]
    if set(names) != set(mano_tips) or set(names) != set(robot_mcps) or set(names) != set(robot_tips):
        raise ValueError("MANO tips, robot MCPs, and robot tips must cover the same fingers")

    mano_wrist = np.asarray(mano_joints[wrist_joint_index], dtype=float)
    robot_wrist = np.asarray(robot_wrist, dtype=float)
    mano_palm_lengths = []
    robot_palm_lengths = []
    finger_ratios: dict[str, float] = {}
    for spec in finger_specs:
        mano_mcp = np.asarray(mano_joints[spec.mcp_joint_index], dtype=float)
        robot_mcp = np.asarray(robot_mcps[spec.name], dtype=float)
        mano_palm_lengths.append(np.linalg.norm(mano_mcp - mano_wrist))
        robot_palm_lengths.append(np.linalg.norm(robot_mcp - robot_wrist))
        mano_length = np.linalg.norm(np.asarray(mano_tips[spec.name]) - mano_mcp)
        robot_length = np.linalg.norm(np.asarray(robot_tips[spec.name]) - robot_mcp)
        if mano_length <= 0:
            raise ValueError(f"MANO {spec.name} MCP-to-tip length must be positive")
        finger_ratios[spec.name] = float(robot_length / mano_length)

    mano_mean = float(np.mean(mano_palm_lengths))
    if mano_mean <= 0:
        raise ValueError("mean MANO wrist-to-MCP distance must be positive")
    palm_scale = float(np.mean(robot_palm_lengths) / mano_mean)
    return palm_scale, finger_ratios
