"""Export a reproducible world-space MANO trajectory from a GRAB sequence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import trimesh

from ...data.grab import decode_grab_hand, load_grab_hand_motion
from ...utils.io import save_npz


def main() -> None:
    """Decode and export a world-space MANO trajectory from one GRAB sequence."""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sequence",
        type=Path,
        default=Path("../data/raw/grab/s1/apple_lift.npz"),
    )
    parser.add_argument(
        "--mano-dir",
        type=Path,
        default=Path("../data/models/mano/mano_v1_2/models"),
    )
    parser.add_argument("--side", choices=("left", "right"), default="right")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("../data/processed/s1/apple_lift_rhand_mano.npz"),
    )
    parser.add_argument(
        "--preview",
        type=Path,
        default=Path("outputs/apple_lift_rhand_frame0000.ply"),
    )
    args = parser.parse_args()

    motion = load_grab_hand_motion(args.sequence, side=args.side)
    decoded = decode_grab_hand(
        motion,
        args.mano_dir,
        batch_size=args.batch_size,
        device=args.device,
    )
    timestamps = np.arange(motion.frame_count, dtype=np.float64) / motion.framerate_hz
    save_npz(
        args.output,
        frame_index=np.arange(motion.frame_count, dtype=np.int32),
        timestamp_s=timestamps,
        mano_global_orient=motion.global_orient,
        mano_hand_pose=motion.hand_pose,
        mano_translation_m=motion.translation_m,
        hand_vertices_m=decoded.vertices_m,
        hand_joints_m=decoded.joints_m,
        hand_faces=decoded.faces,
        subject=np.asarray(motion.subject),
        side=np.asarray(motion.side),
        framerate_hz=np.asarray(motion.framerate_hz),
        source_sequence=np.asarray(str(args.sequence)),
        source_template=np.asarray(str(motion.template_path)),
    )

    args.preview.parent.mkdir(parents=True, exist_ok=True)
    trimesh.Trimesh(
        vertices=decoded.vertices_m[0],
        faces=decoded.faces,
        process=False,
    ).export(args.preview)

    summary = {
        "subject": motion.subject,
        "side": motion.side,
        "frames": motion.frame_count,
        "framerate_hz": motion.framerate_hz,
        "vertices_shape": list(decoded.vertices_m.shape),
        "joints_shape": list(decoded.joints_m.shape),
        "output": str(args.output),
        "preview": str(args.preview),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
