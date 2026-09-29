import numpy as np
import pytest

from morphim_mmo.optimization.metrics import point_rmse, rotation_errors
from morphim_mmo.models.configuration import morphology_pose_vector
from morphim_mmo.utils.frames import parse_frame_indices


def test_parse_frame_indices_supports_all_alias_csv_and_range() -> None:
    """All CLI selection forms should produce increasing validated indices."""

    np.testing.assert_array_equal(parse_frame_indices("all", 4), [0, 1, 2, 3])
    np.testing.assert_array_equal(
        parse_frame_indices("peak", 5, aliases={"peak": 3}), [3]
    )
    np.testing.assert_array_equal(parse_frame_indices("0,2,4", 5), [0, 2, 4])
    np.testing.assert_array_equal(parse_frame_indices("1:5:2", 5), [1, 3])


def test_parse_frame_indices_rejects_invalid_sequences() -> None:
    """Sequential optimizers must reject empty, repeated, or reversed frames."""

    for specification in ("2,2", "3,1", "5", "1:1", "0:4:-1"):
        with pytest.raises(ValueError):
            parse_frame_indices(specification, 4)


def test_shared_point_and_rotation_metrics() -> None:
    """Shared metrics should preserve the former optimizer calculations."""

    assert point_rmse(np.zeros((2, 3)), np.ones((2, 3))) == pytest.approx(np.sqrt(3))
    identity = np.eye(3)[None]
    quarter_turn = np.asarray([[[0, -1, 0], [1, 0, 0], [0, 0, 1]]])
    np.testing.assert_allclose(rotation_errors(identity, quarter_turn), np.pi / 2)


def test_morphology_pose_vector_preserves_requested_dtype() -> None:
    """Shared morphology packing must preserve optimizer precision choices."""

    result = {
        "global_axis_angle_rad": [0.0, 0.1, 0.2],
        "hand_pose_axis_angle_rad": np.zeros((15, 3)).tolist(),
        "translation_m": [0.3, 0.4, 0.5],
    }
    parameters = morphology_pose_vector(result, dtype=np.float64)
    assert parameters.shape == (51,)
    assert parameters.dtype == np.float64
