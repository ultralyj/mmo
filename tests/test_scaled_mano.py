import jax.numpy as jnp
import numpy as np

from morphim_mmo.models.scaled_mano import (
    FingerSpec,
    initial_scale_ratios,
    scale_mano_arrays,
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
