"""Privileged teacher observation assembly from paper Section V."""

from __future__ import annotations

import torch


def future_reference_window(
    trajectory: torch.Tensor,
    step: torch.Tensor,
    future_steps: int,
) -> torch.Tensor:
    """Gather current through future reference values, clamping at the end."""

    offsets = torch.arange(future_steps + 1, device=step.device)
    indices = (step[:, None] + offsets[None, :]).clamp_max(trajectory.shape[0] - 1)
    return trajectory[indices]


def privileged_teacher_observation(
    *,
    qpos: torch.Tensor,
    object_pose: torch.Tensor,
    reference_object_pose: torch.Tensor,
    reference_qpos_window: torch.Tensor,
    observed_contact: torch.Tensor,
    reference_contact_window: torch.Tensor,
    minimum_robot_height: torch.Tensor,
    table_height: torch.Tensor,
    object_scale: torch.Tensor,
    object_mass: torch.Tensor,
    object_friction: torch.Tensor,
    object_center_of_mass: torch.Tensor,
    table_friction: torch.Tensor,
    previous_action: torch.Tensor,
    previous_qpos: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Return every privileged observation group listed in paper Section V."""

    return {
        "qpos": qpos,
        "object_pose": object_pose,
        "reference_object_pose": reference_object_pose,
        "reference_qpos_window": reference_qpos_window.flatten(start_dim=1),
        "observed_contact": observed_contact.to(torch.float32),
        "reference_contact_window": reference_contact_window.to(
            torch.float32
        ).flatten(start_dim=1),
        "minimum_robot_height": minimum_robot_height[:, None],
        "table_height": table_height[:, None],
        "object_scale": object_scale,
        "object_mass": object_mass[:, None],
        "object_friction": object_friction[:, None],
        "object_center_of_mass": object_center_of_mass,
        "table_friction": table_friction[:, None],
        "previous_action": previous_action,
        "previous_qpos": previous_qpos,
    }
