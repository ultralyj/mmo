"""Focused regression tests for the residual-RL implementation."""

from __future__ import annotations

import numpy as np
import torch
import trimesh

from morphim_mmo.rl.environment import _grasp_bias_activation, _mesh_mass_properties
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


def test_mesh_mass_properties_use_mesh_center_and_principal_inertia() -> None:
    """Object dynamics should use its watertight mesh rather than a box proxy."""

    mesh = trimesh.creation.box(extents=(0.08, 0.10, 0.12))
    mesh.apply_translation((0.01, -0.02, 0.03))

    center, inertia, quaternion = _mesh_mass_properties(mesh, mass_kg=0.15)

    np.testing.assert_allclose(center, (0.01, -0.02, 0.03), atol=1e-7)
    expected = np.sort(
        0.15
        / 12.0
        * np.array((0.10**2 + 0.12**2, 0.08**2 + 0.12**2, 0.08**2 + 0.10**2))
    )
    np.testing.assert_allclose(np.sort(inertia), expected, rtol=1e-6)
    np.testing.assert_allclose(np.linalg.norm(quaternion), 1.0, atol=1e-7)


def test_grasp_bias_gate_anticipates_contact_and_releases_smoothly() -> None:
    """Robot grasp correction should close before contact and fade afterward."""

    contacts = torch.zeros((8, 5), dtype=torch.bool)
    contacts[4, :3] = True

    gate = _grasp_bias_activation(contacts, 2, 3, 0.5)

    torch.testing.assert_close(
        gate, torch.tensor([0.0, 0.0, 1.0, 1.0, 1.0, 0.5, 0.25, 0.125])
    )


def test_success_at_reference_horizon_is_truncation_not_failure(monkeypatch) -> None:
    """ManiSkill's success termination must not contaminate failure statistics."""
    from types import SimpleNamespace
    from mani_skill.envs.sapien_env import BaseEnv
    from morphim_mmo.rl.environment import MorphometricResidualEnv

    env = MorphometricResidualEnv.__new__(MorphometricResidualEnv)
    env.reference = SimpleNamespace(frame_count=3)
    env._active_step = torch.tensor([2, 1])
    monkeypatch.setattr(BaseEnv, "step", lambda self, action: (
        torch.zeros(2, 1), torch.ones(2), torch.tensor([True, True]),
        torch.tensor([False, False]),
        {"fail": torch.tensor([False, True]), "success": torch.tensor([True, False])},
    ))
    _, _, terminated, truncated, _ = env.step(torch.zeros(2, 20))
    torch.testing.assert_close(terminated, torch.tensor([False, True]))
    torch.testing.assert_close(truncated, torch.tensor([True, False]))
