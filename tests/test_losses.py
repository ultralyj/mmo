import jax
import jax.numpy as jnp
import numpy as np

from morphim_mmo.optimization.losses import (
    ContactLossWeights,
    contact_residuals,
    morphology_residuals,
    so3_log_vectors,
)


def test_morphology_residual_norm_matches_paper_mean() -> None:
    joints = jnp.asarray([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    tips = jnp.asarray([[0.0, 0.0, 3.0]])
    residual = morphology_residuals(joints, jnp.zeros_like(joints), tips, jnp.zeros_like(tips))
    assert np.isclose(float(residual @ residual), (1.0 + 4.0) / 2.0 + 9.0)


def test_contact_table_and_pose_losses_are_zero_for_match() -> None:
    vertices = jnp.asarray([[0.0, 0.0, 0.1], [1.0, 0.0, 0.2]])
    rotations = jnp.tile(jnp.eye(3), (2, 1, 1))
    residual = contact_residuals(
        vertices,
        jnp.asarray([1]),
        vertices[jnp.asarray([1])],
        rotations,
        rotations,
        table_height=0.0,
        weights=ContactLossWeights(),
    )
    np.testing.assert_allclose(np.asarray(residual), 0.0, atol=1e-7)


def test_pose_log_residual_has_finite_gradient_at_exact_match() -> None:
    reference = jnp.tile(jnp.eye(3), (2, 1, 1))

    def loss(rotations: jnp.ndarray) -> jnp.ndarray:
        residual = so3_log_vectors(rotations, reference)
        return jnp.sum(residual * residual)

    gradient = jax.grad(loss)(reference)
    assert np.isfinite(np.asarray(gradient)).all()


def test_padded_contacts_ignore_masked_entries() -> None:
    vertices = jnp.asarray([[0.0, 0.0, 0.1], [1.0, 0.0, 0.2]])
    rotations = jnp.tile(jnp.eye(3), (2, 1, 1))
    residual = contact_residuals(
        vertices,
        jnp.asarray([1, 0]),
        jnp.asarray([[0.5, 0.0, 0.2], [100.0, 100.0, 100.0]]),
        rotations,
        rotations,
        table_height=0.0,
        contact_mask=jnp.asarray([1.0, 0.0]),
    )
    assert np.isclose(float(residual @ residual), 0.25)
