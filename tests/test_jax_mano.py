import jax.numpy as jnp
import numpy as np

from morphim_mmo.models.jax_mano import axis_angle_to_matrix


def test_axis_angle_identity_and_quarter_turn() -> None:
    rotations = np.asarray(
        axis_angle_to_matrix(jnp.asarray([[0.0, 0.0, 0.0], [0.0, 0.0, np.pi / 2]]))
    )
    np.testing.assert_allclose(rotations[0], np.eye(3), atol=1e-7)
    np.testing.assert_allclose(
        rotations[1],
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        atol=1e-6,
    )
