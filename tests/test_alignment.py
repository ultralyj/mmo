import numpy as np

from morphim_mmo.optimization.alignment import initial_rigid_alignment, palm_frame


def test_palm_frame_is_orthonormal_and_right_handed() -> None:
    frame = palm_frame(
        np.zeros(3),
        np.asarray([1.0, 1.0, 0.0]),
        np.asarray([0.0, 2.0, 0.0]),
    )
    np.testing.assert_allclose(frame.T @ frame, np.eye(3), atol=1e-8)
    assert np.linalg.det(frame) > 0.0


def test_initial_alignment_maps_palm_landmarks() -> None:
    angle = np.pi / 2
    rotation = np.asarray(
        [[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]]
    )
    translation = np.asarray([1.0, 2.0, 3.0])
    wrist = np.zeros(3)
    thumb = np.asarray([1.0, 1.0, 0.0])
    middle = np.asarray([0.0, 2.0, 0.0])
    result_rotation, _, result_translation = initial_rigid_alignment(
        wrist,
        thumb,
        middle,
        rotation @ wrist + translation,
        rotation @ thumb + translation,
        rotation @ middle + translation,
    )
    np.testing.assert_allclose(result_rotation, rotation, atol=1e-8)
    np.testing.assert_allclose(result_translation, translation, atol=1e-8)
