import numpy as np

from morphim_mmo.models.skeleton_skinning import (
    RobotSkeleton,
    heat_diffusion_weights,
    linear_blend_targets,
    one_dimensional_laplacian,
    target_link_poses,
)


def test_one_dimensional_laplacian_is_symmetric_and_constant_nullspace() -> None:
    points = np.asarray([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 3.0]])
    edges = np.asarray([[0, 1], [1, 2]])
    laplacian, mass = one_dimensional_laplacian(points, edges)
    np.testing.assert_allclose(laplacian, laplacian.T)
    np.testing.assert_allclose(laplacian @ np.ones(3), 0.0)
    np.testing.assert_allclose(mass, [0.5, 1.5, 1.0])


def test_heat_weights_are_nonnegative_partition_of_unity() -> None:
    points = np.asarray([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 2.0]])
    edges = np.asarray([[0, 1], [1, 2]])
    influences = points[[0, 2]]
    weights = heat_diffusion_weights(
        points, edges, influences, source_sigma_m=0.4, diffusion_scale=0.1
    )
    assert np.all(weights >= 0.0)
    np.testing.assert_allclose(weights.sum(axis=1), 1.0, atol=1e-6)
    assert weights[0, 0] > weights[0, 1]
    assert weights[2, 1] > weights[2, 0]


def test_identity_blending_reconstructs_calibrated_rest_link_poses() -> None:
    rest_positions = np.asarray([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    skeleton = RobotSkeleton(
        node_names=("root", "tip"),
        link_names=("root", "tip"),
        rest_positions=rest_positions,
        rest_link_rotations=np.tile(np.eye(3), (2, 1, 1)),
        parent_indices=np.asarray([-1, 0]),
        orientation_child_indices=np.asarray([1, 1]),
        edges=np.asarray([[0, 1]]),
    )
    weights = np.asarray([[0.8, 0.2], [0.1, 0.9]])
    influence_transforms = np.tile(np.eye(4), (2, 1, 1))
    positions, linear_maps = linear_blend_targets(
        rest_positions, weights, influence_transforms
    )
    poses = target_link_poses(skeleton, positions, linear_maps)
    np.testing.assert_allclose(poses[:, :3, 3], rest_positions)
    np.testing.assert_allclose(poses[:, :3, :3], np.tile(np.eye(3), (2, 1, 1)))
