"""Stateful termination logic for residual-RL episodes."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .config import TerminationConfig


@dataclass
class TerminationState:
    """Per-environment exponential moving averages and initialization flags."""

    add_ema: torch.Tensor
    contact_mismatch_ema: torch.Tensor
    initialized: torch.Tensor

    @classmethod
    def zeros(cls, count: int, device: torch.device | str) -> "TerminationState":
        """Create an empty termination state for parallel environments."""

        return cls(
            add_ema=torch.zeros(count, device=device),
            contact_mismatch_ema=torch.zeros(count, device=device),
            initialized=torch.zeros(count, dtype=torch.bool, device=device),
        )

    def reset(self, indices: torch.Tensor) -> None:
        """Reset selected parallel environments in place."""

        self.add_ema[indices] = 0.0
        self.contact_mismatch_ema[indices] = 0.0
        self.initialized[indices] = False


def update_termination_state(
    state: TerminationState,
    add_error: torch.Tensor,
    contact_mismatch: torch.Tensor,
    decay: float,
) -> None:
    """Update both EMAs without bias from zero at the first time step."""

    first = ~state.initialized
    state.add_ema = torch.where(
        first, add_error, decay * state.add_ema + (1.0 - decay) * add_error
    )
    state.contact_mismatch_ema = torch.where(
        first,
        contact_mismatch,
        decay * state.contact_mismatch_ema + (1.0 - decay) * contact_mismatch,
    )
    state.initialized[:] = True


def termination_reasons(
    state: TerminationState,
    minimum_robot_height: torch.Tensor,
    table_height: torch.Tensor,
    maximum_object_contact_force: torch.Tensor,
    table_contact_force: torch.Tensor,
    config: TerminationConfig,
) -> dict[str, torch.Tensor]:
    """Evaluate paper tracking, contact, collision, and force terminations."""

    reasons = {
        "terminate_add_ema": state.add_ema > config.add_ema_threshold_m,
        "terminate_contact_ema": (
            state.contact_mismatch_ema > config.contact_mismatch_ema_threshold
        ),
        "terminate_table_height": minimum_robot_height < table_height,
        "terminate_table_force": table_contact_force
        > config.table_collision_force_n,
        "terminate_contact_force": maximum_object_contact_force
        > config.maximum_contact_force_n,
    }
    reasons["fail"] = torch.stack(tuple(reasons.values()), dim=-1).any(dim=-1)
    return reasons
