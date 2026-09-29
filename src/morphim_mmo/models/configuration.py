"""Translate hand YAML mappings into validated MANO model inputs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .mano import ManoModelArrays
from .scaled_mano import FingerSpec, ScaledManoArrays, scale_mano_arrays


def finger_specs_from_config(config: Mapping[str, Any]) -> tuple[FingerSpec, ...]:
    """Build ordered MANO finger-scaling specifications from hand config."""

    semantics = config["mano_semantics"]
    return tuple(
        FingerSpec(
            str(name),
            int(semantics[name]["mcp_joint_index"]),
            tuple(int(value) for value in semantics[name]["skinning_joint_indices"]),
        )
        for name in config["fingers"]
    )


def scaled_mano_from_config(
    arrays: ManoModelArrays,
    config: Mapping[str, Any],
    scales: Sequence[float] | np.ndarray,
) -> ScaledManoArrays:
    """Apply the configured palm/finger semantics to six morphology scales."""

    names = tuple(str(name) for name in config["fingers"])
    values = np.asarray(scales)
    if values.shape != (len(names) + 1,):
        raise ValueError(f"expected {len(names) + 1} morphology scales")
    return scale_mano_arrays(
        arrays.template,
        arrays.shape_blends,
        arrays.pose_blends,
        arrays.joint_regressor,
        arrays.blend_weights,
        values[0],
        dict(zip(names, values[1:], strict=True)),
        finger_specs_from_config(config),
        wrist_joint_index=int(config["mano_semantics"]["wrist_joint_index"]),
    )


def morphology_pose_vector(
    result: Mapping[str, Any], *, dtype: np.dtype[Any] | type = np.float32
) -> np.ndarray:
    """Pack a morphology JSON pose into MANO's 51-parameter frame layout."""

    parameters = np.concatenate(
        [
            np.asarray(result["global_axis_angle_rad"], dtype=dtype),
            np.asarray(result["hand_pose_axis_angle_rad"], dtype=dtype).reshape(-1),
            np.asarray(result["translation_m"], dtype=dtype),
        ]
    )
    if parameters.shape != (51,):
        raise ValueError("morphology pose must contain 51 parameters")
    return parameters
