"""Reference hand-object contact extraction and temporal target completion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class ContactTargets:
    """Per-frame MANO contact indices and corresponding world-space targets."""

    indices: tuple[np.ndarray, ...]
    positions: tuple[np.ndarray, ...]
    source_counts: np.ndarray
    max_contact_frame: int


def table_top_heights(
    vertices_local: np.ndarray, transforms: np.ndarray
) -> np.ndarray:
    """Compute the upper world-z extent of a rigid table at every frame."""

    vertices_local = np.asarray(vertices_local, dtype=np.float64)
    transforms = np.asarray(transforms, dtype=np.float64)
    if vertices_local.ndim != 2 or vertices_local.shape[1] != 3:
        raise ValueError("vertices_local must have shape (V, 3)")
    if transforms.ndim != 3 or transforms.shape[1:] != (4, 4):
        raise ValueError("transforms must have shape (T, 4, 4)")
    # With world = local @ R.T + t, world z is local dot R[2, :] + t_z.
    return np.asarray(
        [
            np.max(vertices_local @ transform[2, :3]) + transform[2, 3]
            for transform in transforms
        ],
        dtype=np.float32,
    )


def detect_contact_indices(
    hand_vertices: np.ndarray,
    object_vertices: np.ndarray,
    threshold_m: float = 0.005,
) -> np.ndarray:
    """Return hand vertex indices within `threshold_m` of an object vertex."""

    hand_vertices = np.asarray(hand_vertices, dtype=float)
    object_vertices = np.asarray(object_vertices, dtype=float)
    if hand_vertices.ndim != 2 or hand_vertices.shape[1] != 3:
        raise ValueError("hand_vertices must have shape (V, 3)")
    if object_vertices.ndim != 2 or object_vertices.shape[1] != 3:
        raise ValueError("object_vertices must have shape (O, 3)")
    if object_vertices.shape[0] == 0:
        raise ValueError("object_vertices cannot be empty")
    if threshold_m <= 0:
        raise ValueError("threshold_m must be positive")
    distances, _ = cKDTree(object_vertices).query(hand_vertices, k=1)
    return np.flatnonzero(distances < threshold_m).astype(np.int32)


def build_contact_targets(
    hand_vertices_by_frame: np.ndarray,
    object_vertices_by_frame: np.ndarray | Sequence[np.ndarray],
    threshold_m: float = 0.005,
) -> ContactTargets:
    """Detect contacts and apply the paper's max-contact fill for empty frames."""

    hands = np.asarray(hand_vertices_by_frame, dtype=float)
    if hands.ndim != 3 or hands.shape[2] != 3:
        raise ValueError("hand_vertices_by_frame must have shape (T, V, 3)")
    if len(object_vertices_by_frame) != hands.shape[0]:
        raise ValueError("hand and object trajectories must have the same length")

    detected = tuple(
        detect_contact_indices(hands[t], np.asarray(object_vertices_by_frame[t]), threshold_m)
        for t in range(hands.shape[0])
    )
    counts = np.asarray([len(indices) for indices in detected], dtype=np.int32)
    max_frame = int(np.argmax(counts))
    fallback = detected[max_frame]
    completed = tuple(indices if len(indices) else fallback.copy() for indices in detected)
    positions = tuple(hands[t, indices].copy() for t, indices in enumerate(completed))
    return ContactTargets(
        indices=completed,
        positions=positions,
        source_counts=counts,
        max_contact_frame=max_frame,
    )


def build_rigid_contact_targets(
    hand_vertices_by_frame: np.ndarray,
    object_vertices_local: np.ndarray,
    object_rotations: np.ndarray,
    object_translations: np.ndarray,
    threshold_m: float = 0.005,
) -> ContactTargets:
    """Extract contacts without expanding a rigid object mesh for every frame."""

    hands = np.asarray(hand_vertices_by_frame, dtype=float)
    object_vertices = np.asarray(object_vertices_local, dtype=float)
    rotations = np.asarray(object_rotations, dtype=float)
    translations = np.asarray(object_translations, dtype=float)
    if hands.ndim != 3 or hands.shape[2] != 3:
        raise ValueError("hand_vertices_by_frame must have shape (T, V, 3)")
    if object_vertices.ndim != 2 or object_vertices.shape[1] != 3:
        raise ValueError("object_vertices_local must have shape (O, 3)")
    if object_vertices.shape[0] == 0:
        raise ValueError("object_vertices_local cannot be empty")
    frame_count = hands.shape[0]
    if rotations.shape != (frame_count, 3, 3):
        raise ValueError("object_rotations must have shape (T, 3, 3)")
    if translations.shape != (frame_count, 3):
        raise ValueError("object_translations must have shape (T, 3)")
    if threshold_m <= 0:
        raise ValueError("threshold_m must be positive")

    tree = cKDTree(object_vertices)
    detected = []
    for hand, rotation, translation in zip(hands, rotations, translations, strict=True):
        hand_local = (hand - translation) @ rotation
        distances, _ = tree.query(hand_local, k=1)
        detected.append(np.flatnonzero(distances < threshold_m).astype(np.int32))
    detected_tuple = tuple(detected)
    counts = np.asarray([len(indices) for indices in detected_tuple], dtype=np.int32)
    max_frame = int(np.argmax(counts))
    fallback = detected_tuple[max_frame]
    completed = tuple(
        indices if len(indices) else fallback.copy() for indices in detected_tuple
    )
    positions = tuple(hands[t, indices].copy() for t, indices in enumerate(completed))
    return ContactTargets(
        indices=completed,
        positions=positions,
        source_counts=counts,
        max_contact_frame=max_frame,
    )
