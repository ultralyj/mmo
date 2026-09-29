"""Focused regression tests for the residual-RL implementation."""

from __future__ import annotations

import numpy as np
import torch

from morphim_mmo.rl.ppo import ActorCritic, _single_env_action_limit
from morphim_mmo.rl.reference import _reference_finger_contacts


def test_empty_detected_contact_frame_ignores_padded_indices() -> None:
    """A zero-count frame must not turn nearest-vertex padding into contact."""

    states = _reference_finger_contacts(
        contact_offsets=np.array([0, 2, 4]),
        contact_indices=np.array([0, 1, 2, 3]),
        vertex_labels=np.array([0, 1, 2, 3]),
        finger_count=5,
        detected_contact_counts=np.array([0, 2]),
    )

    np.testing.assert_array_equal(states[0], np.zeros(5, dtype=bool))
    np.testing.assert_array_equal(
        states[1], np.array([False, False, True, True, False])
    )


def test_squashed_policy_is_bounded_and_log_probability_is_consistent() -> None:
    """PPO must optimize the same bounded residual action sent to the simulator."""

    torch.manual_seed(3)
    policy = ActorCritic(
        observation_dim=7,
        action_dim=4,
        hidden_sizes=(8,),
        action_limit=torch.full((4,), 0.25),
    )
    observations = torch.randn(16, 7)

    actions, sampled_log_probability, _, _ = policy.act(observations)
    evaluated_log_probability, _, _ = policy.evaluate(observations, actions)

    assert torch.all(actions.abs() < 0.25)
    torch.testing.assert_close(
        sampled_log_probability, evaluated_log_probability, atol=2e-5, rtol=2e-5
    )


def test_vectorized_action_bounds_reduce_to_one_environment() -> None:
    """ManiSkill's batched Box bounds must not retain the environment axis."""

    batched_high = np.full((128, 20), 0.25, dtype=np.float32)

    limit = _single_env_action_limit(batched_high, 20, torch.device("cpu"))

    assert limit.shape == (20,)
    torch.testing.assert_close(limit, torch.full((20,), 0.25))
