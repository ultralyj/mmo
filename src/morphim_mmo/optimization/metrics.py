"""Numerical metrics shared by the three matching stages."""

from __future__ import annotations

import numpy as np


def point_rmse(left: np.ndarray, right: np.ndarray) -> float:
    """Return root-mean-square Euclidean distance between paired points."""

    delta = np.asarray(left) - np.asarray(right)
    return float(np.sqrt(np.mean(np.sum(delta**2, axis=-1))))


def rotation_errors(actual: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Return geodesic SO(3) errors in radians for paired rotation matrices."""

    relative = np.swapaxes(np.asarray(actual), -1, -2) @ np.asarray(target)
    cosine = np.clip(
        (np.trace(relative, axis1=-2, axis2=-1) - 1.0) / 2.0,
        -1.0,
        1.0,
    )
    return np.arccos(cosine)


def rotation_error_statistics(
    actual: np.ndarray, target: np.ndarray
) -> tuple[float, float]:
    """Return mean and maximum geodesic SO(3) errors in radians."""

    errors = rotation_errors(actual, target)
    return float(np.mean(errors)), float(np.max(errors))

