"""Least-squares residuals for MMO equations 7--10."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp


@dataclass(frozen=True)
class ContactLossWeights:
    """Scalar weights for contact, table, and pose residual groups."""

    contact: float = 1.0
    table: float = 1.0
    pose: float = 1.0


def _weighted_mean_residual(values: jnp.ndarray, weight: float) -> jnp.ndarray:
    """Scale a residual block so its squared norm is a weighted mean."""

    # The paper averages squared Euclidean norms over vertices/joints, not over
    # individual xyz coordinates. For scalar residuals this is unchanged.
    count = max(int(values.shape[0]), 1)
    return jnp.sqrt(jnp.asarray(weight) / count) * values.reshape(-1)


def morphology_residuals(
    mano_joints: jnp.ndarray,
    robot_joints: jnp.ndarray,
    mano_tips: jnp.ndarray,
    robot_tips: jnp.ndarray,
    joint_weight: float = 1.0,
    fingertip_weight: float = 1.0,
) -> jnp.ndarray:
    """Residual vector whose squared norm is Eq. (7)."""

    joint_delta = jnp.asarray(mano_joints) - jnp.asarray(robot_joints)
    tip_delta = jnp.asarray(mano_tips) - jnp.asarray(robot_tips)
    return jnp.concatenate(
        [
            _weighted_mean_residual(joint_delta, joint_weight),
            _weighted_mean_residual(tip_delta, fingertip_weight),
        ]
    )


def so3_geodesic_angles(
    rotations: jnp.ndarray,
    reference_rotations: jnp.ndarray,
) -> jnp.ndarray:
    """Numerically stable SO(3) geodesic angles in radians."""

    relative = jnp.swapaxes(reference_rotations, -1, -2) @ rotations
    cos_angle = (jnp.trace(relative, axis1=-2, axis2=-1) - 1.0) / 2.0
    return jnp.arccos(jnp.clip(cos_angle, -1.0, 1.0))


def so3_log_vectors(
    rotations: jnp.ndarray,
    reference_rotations: jnp.ndarray,
) -> jnp.ndarray:
    """Return stable tangent vectors for ``reference.T @ rotation``.

    Their norms are the SO(3) geodesic angles.  The small-angle branch avoids
    the undefined derivative of ``arccos`` at an exact pose match, which is
    important because contact matching starts on the pose-regularization
    target for many frames.
    """

    relative = jnp.swapaxes(reference_rotations, -1, -2) @ rotations
    vee_sine = 0.5 * jnp.stack(
        [
            relative[..., 2, 1] - relative[..., 1, 2],
            relative[..., 0, 2] - relative[..., 2, 0],
            relative[..., 1, 0] - relative[..., 0, 1],
        ],
        axis=-1,
    )
    sine_squared = jnp.sum(vee_sine * vee_sine, axis=-1)
    safe_sine = jnp.sqrt(jnp.maximum(sine_squared, 1e-12))
    cosine = jnp.clip(
        (jnp.trace(relative, axis1=-2, axis2=-1) - 1.0) / 2.0,
        -1.0,
        1.0,
    )
    angle = jnp.arctan2(safe_sine, cosine)
    large_angle_scale = angle / safe_sine
    scale = jnp.where(sine_squared < 1e-8, 1.0, large_angle_scale)
    return scale[..., None] * vee_sine


def contact_residuals(
    vertices: jnp.ndarray,
    contact_indices: jnp.ndarray,
    target_positions: jnp.ndarray,
    joint_rotations: jnp.ndarray,
    reference_rotations: jnp.ndarray,
    table_height: float,
    weights: ContactLossWeights = ContactLossWeights(),
    include_table: bool = True,
    contact_mask: jnp.ndarray | None = None,
    table_mask: float | jnp.ndarray = 1.0,
) -> jnp.ndarray:
    """Residual vector for five-finger contact matching (Eq. 8 without L_d).

    ``contact_mask`` permits fixed-size padding, so a complete trajectory can
    use one JAX compilation despite varying contact counts. Five-finger hands
    with one-to-one anatomical mapping correctly omit the coupled-finger loss.
    """

    contact_delta = vertices[contact_indices] - target_positions
    if contact_mask is None:
        contact_mask = jnp.ones((contact_delta.shape[0],), dtype=vertices.dtype)
    contact_mask = jnp.asarray(contact_mask, dtype=vertices.dtype)
    contact_count = jnp.maximum(jnp.sum(contact_mask), 1.0)
    contact_scale = jnp.sqrt(jnp.asarray(weights.contact) / contact_count)
    chunks = [(contact_scale * contact_mask[:, None] * contact_delta).reshape(-1)]
    if include_table:
        penetration = (
            jnp.asarray(table_mask, dtype=vertices.dtype)
            * jnp.maximum(0.0, table_height - vertices[:, 2])
        )
        chunks.append(_weighted_mean_residual(penetration, weights.table))
    pose_vectors = so3_log_vectors(joint_rotations, reference_rotations)
    chunks.append(_weighted_mean_residual(pose_vectors, weights.pose))
    return jnp.concatenate(chunks)
