"""Paper-exact object/contact reward terms for residual RL."""

from __future__ import annotations

import torch


def transform_object_points(
    points: torch.Tensor,
    position: torch.Tensor,
    quaternion_wxyz: torch.Tensor,
) -> torch.Tensor:
    """Transform local object points by batched world poses."""

    quaternion = quaternion_wxyz / torch.linalg.vector_norm(
        quaternion_wxyz, dim=-1, keepdim=True
    ).clamp_min(1e-12)
    w, x, y, z = quaternion.unbind(dim=-1)
    rotation = torch.stack(
        [
            1 - 2 * (y * y + z * z),
            2 * (x * y - z * w),
            2 * (x * z + y * w),
            2 * (x * y + z * w),
            1 - 2 * (x * x + z * z),
            2 * (y * z - x * w),
            2 * (x * z - y * w),
            2 * (y * z + x * w),
            1 - 2 * (x * x + y * y),
        ],
        dim=-1,
    ).reshape(*quaternion.shape[:-1], 3, 3)
    return torch.einsum("...ij,pj->...pi", rotation, points) + position[..., None, :]


def average_distance_error(
    object_points: torch.Tensor,
    observed_position: torch.Tensor,
    observed_quaternion_wxyz: torch.Tensor,
    reference_position: torch.Tensor,
    reference_quaternion_wxyz: torch.Tensor,
) -> torch.Tensor:
    """Compute equation 13 ADD between observed and reference object poses."""

    observed = transform_object_points(
        object_points, observed_position, observed_quaternion_wxyz
    )
    reference = transform_object_points(
        object_points, reference_position, reference_quaternion_wxyz
    )
    return torch.linalg.vector_norm(observed - reference, dim=-1).mean(dim=-1)


def contact_matching_reward(
    observed_contact: torch.Tensor,
    reference_contact: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute equation 14 using counts of contacting robot and human fingers."""

    if observed_contact.shape != reference_contact.shape:
        raise ValueError("observed and reference contact tensors must have equal shape")
    maximum_fingers = observed_contact.shape[-1]
    observed_count = observed_contact.to(torch.float32).sum(dim=-1)
    human_count = reference_contact.to(torch.float32).sum(dim=-1)
    goal_count = torch.clamp(human_count, max=float(maximum_fingers))
    mismatch = torch.abs(goal_count - observed_count) / float(maximum_fingers)
    return 1.0 - mismatch, mismatch, goal_count


def morphometric_reward(
    add_error: torch.Tensor,
    observed_contact: torch.Tensor,
    reference_contact: torch.Tensor,
    alpha: float,
) -> dict[str, torch.Tensor]:
    """Compute the paper's multiplicative residual-RL reward."""

    contact_reward, mismatch, goal_count = contact_matching_reward(
        observed_contact, reference_contact
    )
    object_reward = torch.exp(-alpha * add_error)
    return {
        "reward": object_reward * contact_reward,
        "object_reward": object_reward,
        "contact_reward": contact_reward,
        "contact_mismatch": mismatch,
        "goal_contact_count": goal_count,
    }
