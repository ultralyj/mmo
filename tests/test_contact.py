import numpy as np

from morphim_mmo.data.contact import (
    build_contact_targets,
    build_rigid_contact_targets,
    detect_contact_indices,
)


def test_contact_threshold_is_strictly_less_than_five_mm() -> None:
    hand = np.asarray([[0.0, 0.0, 0.0], [0.004, 0.0, 0.0], [0.005, 0.0, 0.0]])
    obj = np.asarray([[0.0, 0.0, 0.0]])
    np.testing.assert_array_equal(detect_contact_indices(hand, obj), [0, 1])


def test_empty_frames_use_max_contact_indices_but_current_positions() -> None:
    hands = np.asarray(
        [
            [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [0.004, 0.0, 0.0]],
        ]
    )
    objects = np.asarray([[[0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0]]])
    targets = build_contact_targets(hands, objects)
    assert targets.max_contact_frame == 1
    np.testing.assert_array_equal(targets.source_counts, [0, 2])
    np.testing.assert_array_equal(targets.indices[0], [0, 1])
    np.testing.assert_allclose(targets.positions[0], hands[0])


def test_rigid_contact_matches_expanded_world_object() -> None:
    hands = np.array(
        [
            [[1.0, 2.0, 0.0], [2.0, 2.0, 0.0]],
            [[0.0, 1.0, 0.0], [0.0, 2.0, 0.0]],
        ]
    )
    object_local = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    rotations = np.array(
        [
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        ]
    )
    translations = np.array([[1.0, 2.0, 0.0], [0.0, 1.0, 0.0]])
    expanded = np.stack(
        [
            object_local @ rotation.T + translation
            for rotation, translation in zip(rotations, translations, strict=True)
        ]
    )
    expected = build_contact_targets(hands, expanded, threshold_m=1e-4)
    actual = build_rigid_contact_targets(
        hands, object_local, rotations, translations, threshold_m=1e-4
    )
    for left, right in zip(actual.indices, expected.indices, strict=True):
        np.testing.assert_array_equal(left, right)
