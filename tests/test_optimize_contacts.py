import numpy as np

from morphim_mmo.data.contact import table_top_heights


def test_table_top_heights_under_rigid_transform() -> None:
    vertices = np.asarray([[0.0, -1.0, 0.0], [0.0, 2.0, 0.0]])
    transform = np.eye(4)
    transform[:3, :3] = np.asarray(
        [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]]
    )
    transform[2, 3] = 3.0
    heights = table_top_heights(vertices, transform[None])
    np.testing.assert_allclose(heights, [5.0])
