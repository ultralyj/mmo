"""Export GRAB rigid-object poses and MMO reference contact targets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import trimesh

from ...data.contact import build_rigid_contact_targets
from ...data.grab import GrabRigidMotion, load_grab_rigid_motion
from ...utils.io import save_npz


def _pack_ragged(values: tuple[np.ndarray, ...]) -> tuple[np.ndarray, np.ndarray]:
    """Pack variable-length arrays into offsets plus one flat array."""

    counts = np.fromiter((len(value) for value in values), dtype=np.int64)
    offsets = np.concatenate([np.zeros(1, dtype=np.int64), np.cumsum(counts)])
    if offsets[-1] == 0:
        shape = (0,) + values[0].shape[1:]
        return offsets, np.empty(shape, dtype=values[0].dtype)
    return offsets, np.concatenate(values, axis=0)


def _world_vertices(motion: GrabRigidMotion, frame: int) -> np.ndarray:
    """Transform one rigid object's local vertices into world coordinates."""

    rotation = motion.rotations()[frame]
    return motion.vertices_local_m @ rotation.T + motion.translation_m[frame]


def main() -> None:
    """Export rigid object/table motion and completed MANO contact targets."""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sequence",
        type=Path,
        default=Path("../data/raw/grab/s1/apple_lift.npz"),
    )
    parser.add_argument(
        "--hand-trajectory",
        type=Path,
        default=Path("../data/processed/s1/apple_lift_rhand_mano.npz"),
    )
    parser.add_argument("--threshold-m", type=float, default=0.005)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("../data/processed/s1/apple_lift_contacts.npz"),
    )
    parser.add_argument(
        "--preview",
        type=Path,
        default=Path("outputs/apple_lift_max_contact_frame.ply"),
    )
    args = parser.parse_args()

    with np.load(args.hand_trajectory, allow_pickle=False) as hand_data:
        hand_vertices = np.asarray(hand_data["hand_vertices_m"], dtype=np.float64)
        hand_faces = np.asarray(hand_data["hand_faces"], dtype=np.int32)
        timestamps = np.asarray(hand_data["timestamp_s"], dtype=np.float64)
    object_motion = load_grab_rigid_motion(args.sequence, section="object")
    table_motion = load_grab_rigid_motion(args.sequence, section="table")
    if hand_vertices.shape[0] != object_motion.frame_count:
        raise ValueError("Hand and object frame counts differ")

    object_rotations = object_motion.rotations()
    targets = build_rigid_contact_targets(
        hand_vertices,
        object_motion.vertices_local_m,
        object_rotations,
        object_motion.translation_m,
        args.threshold_m,
    )
    index_offsets, contact_indices = _pack_ragged(targets.indices)
    position_offsets, contact_positions = _pack_ragged(targets.positions)
    if not np.array_equal(index_offsets, position_offsets):
        raise RuntimeError("Contact index and position packing disagree")

    save_npz(
        args.output,
        timestamp_s=timestamps,
        threshold_m=np.asarray(args.threshold_m),
        contact_offsets=index_offsets,
        contact_indices=contact_indices,
        contact_positions_m=contact_positions.astype(np.float32),
        detected_contact_counts=targets.source_counts,
        max_contact_frame=np.asarray(targets.max_contact_frame, dtype=np.int32),
        object_name=np.asarray(object_motion.name),
        object_vertices_local_m=object_motion.vertices_local_m.astype(np.float32),
        object_faces=object_motion.faces,
        object_to_world=object_motion.transforms(),
        table_vertices_local_m=table_motion.vertices_local_m.astype(np.float32),
        table_faces=table_motion.faces,
        table_to_world=table_motion.transforms(),
        source_sequence=np.asarray(str(args.sequence)),
        source_hand_trajectory=np.asarray(str(args.hand_trajectory)),
    )

    frame = targets.max_contact_frame
    hand_mesh = trimesh.Trimesh(hand_vertices[frame], hand_faces, process=False)
    object_mesh = trimesh.Trimesh(
        _world_vertices(object_motion, frame), object_motion.faces, process=False
    )
    hand_mesh.visual.vertex_colors = [230, 170, 130, 255]
    object_mesh.visual.vertex_colors = [120, 185, 110, 255]
    args.preview.parent.mkdir(parents=True, exist_ok=True)
    trimesh.util.concatenate([hand_mesh, object_mesh]).export(args.preview)

    print(
        json.dumps(
            {
                "frames": object_motion.frame_count,
                "threshold_m": args.threshold_m,
                "frames_with_detected_contact": int(
                    np.count_nonzero(targets.source_counts)
                ),
                "max_contact_frame": targets.max_contact_frame,
                "max_contact_vertices": int(targets.source_counts.max()),
                "packed_contact_targets": int(len(contact_indices)),
                "output": str(args.output),
                "preview": str(args.preview),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
