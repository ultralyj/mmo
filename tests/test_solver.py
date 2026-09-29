import jax.numpy as jnp
import numpy as np

from morphim_mmo.optimization.solver import solve_lm


def test_lm_uses_jax_jacobian_and_recovers_target() -> None:
    result = solve_lm(lambda x: x - jnp.asarray([1.0, -2.0]), np.zeros(2))
    assert result.success
    np.testing.assert_allclose(result.parameters, [1.0, -2.0], atol=1e-6)
