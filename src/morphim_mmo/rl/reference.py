"""Load MMO outputs as synchronized residual-RL reference trajectories."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from ..data.contact import table_top_heights
from ..models.configuration import finger_specs_from_config
from ..models.mano import extract_model_arrays, load_mano_model
from ..utils.io import load_yaml


def _matrix_poses_to_position_wxyz(transforms: np.ndarray) -> np.ndarray:
    """Convert homogeneous matrices to position-plus-wxyz pose vectors."""

    xyzw = Rotation.from_matrix(transforms[:, :3, :3]).as_quat()
    wxyz = xyzw[:, [3, 0, 1, 2]]
    return np.concatenate([transforms[:, :3, 3], wxyz], axis=-1).astype(
        np.float32
    )


def _finger_vertex_labels(
    mano_dir: Path,
    robot_config: dict,
    minimum_membership: float = 0.05,
) -> np.ndarray:
    """Assign MANO vertices to the dominant configured finger skinning chain."""

    model = load_mano_model(mano_dir, "right", use_pca=False, flat_hand_mean=True)
    arrays = extract_model_arrays(model)
    specs = finger_specs_from_config(robot_config)
    membership = np.stack(
        [
            arrays.blend_weights[:, np.asarray(spec.skinning_joint_indices)].sum(
                axis=1
            )
            for spec in specs
        ],
        axis=-1,
    )
    labels = np.argmax(membership, axis=-1).astype(np.int32)
    labels[np.max(membership, axis=-1) < minimum_membership] = -1
    return labels


def _reference_finger_contacts(
    contact_offsets: np.ndarray,
    contact_indices: np.ndarray,
    vertex_labels: np.ndarray,
    finger_count: int,
) -> np.ndarray:
    """Convert ragged human contact vertices into per-finger Boolean states."""

    frame_count = len(contact_offsets) - 1
    states = np.zeros((frame_count, finger_count), dtype=np.bool_)
    for frame in range(frame_count):
        start, stop = int(contact_offsets[frame]), int(contact_offsets[frame + 1])
        labels = vertex_labels[contact_indices[start:stop]]
        valid = labels[labels >= 0]
        if len(valid):
            states[frame, np.unique(valid)] = True
    return states


@dataclass(frozen=True)
class ResidualRLReference:
    """Synchronized robot, object, contact, and table reference arrays."""

    timestamp_s: np.ndarray
    joint_names: tuple[str, ...]
    qpos: np.ndarray
    base_pose_position_wxyz: np.ndarray
    object_pose_position_wxyz: np.ndarray
    reference_contact: np.ndarray
    table_height_m: np.ndarray
    object_points_local_m: np.ndarray
    object_mesh_path: Path
    world_translation_m: np.ndarray

    @property
    def frame_count(self) -> int:
        """Return the number of synchronized reference frames."""

        return len(self.timestamp_s)

    @classmethod
    def load(
        cls,
        *,
        robot_pose_path: str | Path,
        contacts_path: str | Path,
        robot_config_path: str | Path,
        mano_dir: str | Path,
        object_mesh_path: str | Path,
        reference_stride: int,
        object_point_count: int = 512,
    ) -> "ResidualRLReference":
        """Load, synchronize, subsample, and table-normalize MMO references."""

        if reference_stride <= 0:
            raise ValueError("reference_stride must be positive")
        robot = np.load(robot_pose_path, allow_pickle=False)
        contacts = np.load(contacts_path, allow_pickle=False)
        config = load_yaml(Path(robot_config_path))
        if len(robot["timestamp_s"]) != len(contacts["timestamp_s"]):
            raise ValueError("robot and contact trajectories have different lengths")
        if not np.allclose(robot["timestamp_s"], contacts["timestamp_s"]):
            raise ValueError("robot and contact timestamps are not synchronized")

        table_heights = table_top_heights(
            contacts["table_vertices_local_m"], contacts["table_to_world"]
        ).astype(np.float32)
        table_origin = float(np.median(table_heights))
        world_translation = np.asarray([0.0, 0.0, -table_origin], dtype=np.float32)

        base_wxyz_xyz = np.asarray(robot["base_wxyz_xyz"], dtype=np.float32)
        base_pose = np.concatenate(
            [base_wxyz_xyz[:, 4:7] + world_translation, base_wxyz_xyz[:, :4]],
            axis=-1,
        )
        object_pose = _matrix_poses_to_position_wxyz(contacts["object_to_world"])
        object_pose[:, :3] += world_translation

        vertex_labels = _finger_vertex_labels(Path(mano_dir), config)
        reference_contact = _reference_finger_contacts(
            contacts["contact_offsets"],
            contacts["contact_indices"],
            vertex_labels,
            len(config["fingers"]),
        )
        vertices = np.asarray(contacts["object_vertices_local_m"], dtype=np.float32)
        if object_point_count <= 0:
            raise ValueError("object_point_count must be positive")
        sample_indices = np.linspace(
            0, len(vertices) - 1, min(object_point_count, len(vertices)), dtype=np.int32
        )
        selected = np.arange(0, len(robot["timestamp_s"]), reference_stride)
        return cls(
            timestamp_s=np.asarray(robot["timestamp_s"])[selected],
            joint_names=tuple(str(name) for name in robot["joint_names"]),
            qpos=np.asarray(robot["joint_positions_rad"], dtype=np.float32)[selected],
            base_pose_position_wxyz=base_pose[selected],
            object_pose_position_wxyz=object_pose[selected],
            reference_contact=reference_contact[selected],
            table_height_m=(table_heights + world_translation[2])[selected],
            object_points_local_m=vertices[sample_indices],
            object_mesh_path=Path(object_mesh_path).resolve(),
            world_translation_m=world_translation,
        )
