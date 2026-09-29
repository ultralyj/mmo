import jax.numpy as jnp
import numpy as np

from morphim_mmo.models.scaled_mano import (
    FingerSpec,
    initial_scale_ratios,
    scale_mano_arrays,
)
from morphim_mmo.optimization.jaxls_mmo import (
    _bounded_parameters,
    _raw_parameters,
    _scale_mano_dense,
)


def test_two_stage_scaling_matches_equation_five() -> None:
    template = jnp.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    regressor = jnp.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    weights = jnp.asarray([[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
    zeros = jnp.zeros((1, 3, 3))
    result = scale_mano_arrays(
        template,
        zeros,
        zeros,
        regressor,
        weights,
        palm_scale=2.0,
        finger_scales={"index": 3.0},
        finger_specs=[FingerSpec("index", 1, (1,))],
    )
    # Wrist remains fixed. MCP moves from x=1 to x=2 under palm scaling. The
    # distal point is then 1.5x as far from that scaled MCP: 2 + 1.5*2 = 5.
    np.testing.assert_allclose(np.asarray(result.template[[0, 1, 2], 0]), [0.0, 2.0, 5.0])


def test_initial_scale_ratios() -> None:
    specs = [FingerSpec("index", 1, (1,))]
    palm, fingers = initial_scale_ratios(
        mano_joints=np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        mano_tips={"index": np.asarray([3.0, 0.0, 0.0])},
        robot_wrist=np.asarray([0.0, 0.0, 0.0]),
        robot_mcps={"index": np.asarray([2.0, 0.0, 0.0])},
        robot_tips={"index": np.asarray([8.0, 0.0, 0.0])},
        finger_specs=specs,
    )
    assert palm == 2.0
    assert fingers == {"index": 3.0}


def test_dense_jaxls_scaling_matches_reference_implementation() -> None:
    """The vectorized JAXLS scaling path must preserve equations 5--6."""

    template = jnp.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    regressor = jnp.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    weights = jnp.asarray([[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
    pose_blends = jnp.ones((2, 3, 3))
    reference = scale_mano_arrays(
        template,
        jnp.zeros_like(pose_blends),
        pose_blends,
        regressor,
        weights,
        palm_scale=2.0,
        finger_scales={"index": 3.0},
        finger_specs=[FingerSpec("index", 1, (1,))],
    )
    dense_template, dense_pose_blends = _scale_mano_dense(
        template,
        pose_blends,
        regressor,
        jnp.asarray([2.0, 3.0]),
        weights[:, 1, None],
        jnp.asarray([1]),
        jnp.asarray(0),
    )
    np.testing.assert_allclose(dense_template, reference.template)
    np.testing.assert_allclose(dense_pose_blends, reference.pose_blends)


def test_jaxls_bound_reparameterization_round_trip() -> None:
    """Bounded JAXLS coordinates must preserve an interior initialization."""

    lower = jnp.asarray([-2.0, 1.0, -0.5])
    upper = jnp.asarray([4.0, 5.0, 0.5])
    parameters = jnp.asarray([-1.0, 3.0, 0.25])
    raw = _raw_parameters(parameters, lower, upper)
    recovered = _bounded_parameters(raw, lower, upper)
    np.testing.assert_allclose(recovered, parameters, rtol=1e-6, atol=1e-6)
    assert np.all(np.asarray(recovered) > np.asarray(lower))
    assert np.all(np.asarray(recovered) < np.asarray(upper))
