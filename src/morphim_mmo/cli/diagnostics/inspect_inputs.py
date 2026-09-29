"""Validate the current licensed MANO and first GRAB trajectory inputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ...models.mano import extract_model_arrays, load_mano_model, neutral_geometry
from ...utils.io import load_yaml


def main() -> None:
    """Validate licensed MANO/GRAB inputs and report their key dimensions."""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mano-dir",
        type=Path,
        default=Path("../data/models/mano/mano_v1_2/models"),
    )
    parser.add_argument(
        "--sequence",
        type=Path,
        default=Path("../data/raw/grab/s1/apple_lift.npz"),
    )
    parser.add_argument("--config", type=Path, default=Path("configs/sharpa_wave_right.yaml"))
    args = parser.parse_args()

    config = load_yaml(args.config)
    model = load_mano_model(args.mano_dir, "right", use_pca=False, flat_hand_mean=True)
    arrays = extract_model_arrays(model)
    tip_indices = [
        config["mano_semantics"][name]["tip_vertex_index"] for name in config["fingers"]
    ]
    vertices, joints, tips = neutral_geometry(model, tip_indices)

    sequence = np.load(args.sequence, allow_pickle=True)
    object_data = sequence["object"].item()
    table_data = sequence["table"].item()
    dependencies = [object_data["object_mesh"], table_data["table_mesh"]]
    for side in ("lhand", "rhand"):
        dependencies.append(sequence[side].item()["vtemp"])
    grab_root = args.sequence.parents[1]

    payload = {
        "mano": {
            "template_shape": list(arrays.template.shape),
            "joint_count": int(arrays.joint_regressor.shape[0]),
            "face_count": int(arrays.faces.shape[0]),
            "neutral_vertices_finite": bool(np.isfinite(vertices).all()),
            "neutral_joints_shape": list(joints.shape),
            "neutral_tips_shape": list(tips.shape),
        },
        "grab": {
            "subject": str(sequence["sbj_id"]),
            "object": str(sequence["obj_name"]),
            "intent": str(sequence["motion_intent"]),
            "frames": int(sequence["n_frames"]),
            "framerate_hz": float(sequence["framerate"]),
            "right_hand_pose_shape": list(sequence["rhand"].item()["params"]["hand_pose"].shape),
            "referenced_files": {
                value: (grab_root / value).exists() for value in dependencies
            },
        },
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
