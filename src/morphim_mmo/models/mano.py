"""Loading and validation for the licensed MANO v1.2 model files."""

from __future__ import annotations

from dataclasses import dataclass
import inspect
from pathlib import Path
from typing import Literal

import numpy as np
import torch


@dataclass(frozen=True)
class ManoModelArrays:
    """Validated NumPy arrays extracted from an SMPL-X MANO model."""

    template: np.ndarray
    shape_blends: np.ndarray
    pose_blends: np.ndarray
    joint_regressor: np.ndarray
    blend_weights: np.ndarray
    faces: np.ndarray
    parents: np.ndarray
    hand_components: np.ndarray
    hand_mean: np.ndarray


def enable_legacy_chumpy_compatibility() -> None:
    """Restore names needed to unpickle the 2018 MANO/Chumpy archive."""

    if not hasattr(inspect, "getargspec"):
        inspect.getargspec = inspect.getfullargspec  # type: ignore[attr-defined]
    aliases = {
        "bool": np.bool_,
        "int": int,
        "float": float,
        "complex": complex,
        "object": object,
        "str": str,
        "unicode": str,
    }
    for name, value in aliases.items():
        if name not in np.__dict__:
            setattr(np, name, value)


def load_mano_model(
    model_directory: str | Path,
    side: Literal["left", "right"] = "right",
    *,
    use_pca: bool = False,
    num_pca_comps: int = 24,
    flat_hand_mean: bool = True,
):
    """Load MANO through SMPL-X after enabling its legacy pickle compatibility."""

    enable_legacy_chumpy_compatibility()
    from smplx.body_models import MANO

    return MANO(
        str(model_directory),
        is_rhand=side == "right",
        use_pca=use_pca,
        num_pca_comps=num_pca_comps,
        flat_hand_mean=flat_hand_mean,
    )


def extract_model_arrays(model) -> ManoModelArrays:
    """Convert SMPL-X storage layouts to the layouts used by Scaled MANO."""

    def cpu(value: torch.Tensor) -> np.ndarray:
        """Detach a tensor and return an owned CPU NumPy copy."""

        return value.detach().cpu().numpy().copy()

    def as_numpy(value) -> np.ndarray:
        """Return an owned NumPy copy from either tensor or array storage."""

        return cpu(value) if isinstance(value, torch.Tensor) else np.asarray(value).copy()

    template = cpu(model.v_template)
    vertex_count = template.shape[0]
    shapedirs = cpu(model.shapedirs).transpose(2, 0, 1)
    posedirs = cpu(model.posedirs).reshape(-1, vertex_count, 3)
    arrays = ManoModelArrays(
        template=template,
        shape_blends=shapedirs,
        pose_blends=posedirs,
        joint_regressor=cpu(model.J_regressor),
        blend_weights=cpu(model.lbs_weights),
        faces=np.asarray(model.faces, dtype=np.int32).copy(),
        parents=cpu(model.parents).astype(np.int32),
        hand_components=as_numpy(
            getattr(model, "hand_components", model.np_hand_components)
        ),
        hand_mean=as_numpy(model.hand_mean),
    )
    validate_model_arrays(arrays)
    return arrays


def validate_model_arrays(arrays: ManoModelArrays) -> None:
    """Validate canonical MANO v1.2 dimensions and finite values."""

    expected = {
        "template": (778, 3),
        "shape_blends": (10, 778, 3),
        "pose_blends": (135, 778, 3),
        "joint_regressor": (16, 778),
        "blend_weights": (778, 16),
        "faces": (1538, 3),
        "parents": (16,),
    }
    for name, shape in expected.items():
        actual = getattr(arrays, name).shape
        if actual != shape:
            raise ValueError(f"MANO {name} expected shape {shape}, got {actual}")
    if not all(np.isfinite(getattr(arrays, name)).all() for name in expected):
        raise ValueError("MANO model contains non-finite values")


def neutral_geometry(model, tip_vertex_indices: list[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return flat-pose vertices, 16 joints, and five configured fingertips."""

    batch_size = 1
    hand_pose_size = model.num_pca_comps if model.use_pca else 45
    output = model(
        global_orient=torch.zeros((batch_size, 3), dtype=model.dtype),
        hand_pose=torch.zeros((batch_size, hand_pose_size), dtype=model.dtype),
        transl=torch.zeros((batch_size, 3), dtype=model.dtype),
        return_verts=True,
    )
    vertices = output.vertices[0].detach().cpu().numpy()
    joints = output.joints[0, :16].detach().cpu().numpy()
    tips = vertices[np.asarray(tip_vertex_indices, dtype=np.int32)]
    return vertices, joints, tips
