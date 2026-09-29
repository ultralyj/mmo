"""Palm-frame initialization for morphology matching."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


def palm_frame(wrist: np.ndarray, thumb_tip: np.ndarray, middle_tip: np.ndarray) -> np.ndarray:
    """
    Construct a right-handed orthonormal palm frame from three landmarks.

    The longitudinal y-axis points wrist-to-middle-tip. The thumb direction is
    Gram--Schmidt projected into x; z is their cross product. Columns are axes
    expressed in the input coordinate system.
    """

    wrist = np.asarray(wrist, dtype=float)
    y_axis = np.asarray(middle_tip, dtype=float) - wrist
    y_norm = np.linalg.norm(y_axis)
    if y_norm <= 1e-10:
        raise ValueError("wrist and middle fingertip must be distinct")
    y_axis /= y_norm

    thumb = np.asarray(thumb_tip, dtype=float) - wrist
    x_axis = thumb - np.dot(thumb, y_axis) * y_axis
    x_norm = np.linalg.norm(x_axis)
    if x_norm <= 1e-10:
        raise ValueError("thumb and middle directions cannot be collinear")
    x_axis /= x_norm
    z_axis = np.cross(x_axis, y_axis)
    z_axis /= np.linalg.norm(z_axis)
    x_axis = np.cross(y_axis, z_axis)
    return np.column_stack((x_axis, y_axis, z_axis))


def initial_rigid_alignment(
    mano_wrist: np.ndarray,
    mano_thumb_tip: np.ndarray,
    mano_middle_tip: np.ndarray,
    robot_wrist: np.ndarray,
    robot_thumb_tip: np.ndarray,
    robot_middle_tip: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return MANO-to-robot rotation matrix, axis-angle, and translation."""

    mano_frame = palm_frame(mano_wrist, mano_thumb_tip, mano_middle_tip)
    robot_frame = palm_frame(robot_wrist, robot_thumb_tip, robot_middle_tip)
    rotation = robot_frame @ mano_frame.T
    translation = np.asarray(robot_wrist) - rotation @ np.asarray(mano_wrist)
    return rotation, Rotation.from_matrix(rotation).as_rotvec(), translation
