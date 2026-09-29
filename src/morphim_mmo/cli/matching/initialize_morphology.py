"""Compute the paper's URDF-derived Scaled-MANO initialization."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ...models.mano import extract_model_arrays, load_mano_model
from ...models.configuration import finger_specs_from_config
from ...models.scaled_mano import initial_scale_ratios, scale_mano_arrays
from ...models.urdf import UrdfModel, extract_semantic_points
from ...optimization.alignment import initial_rigid_alignment
from ...utils.io import load_yaml, write_json


def main() -> None:
    """Compute scale-ratio and rigid-palm initialization for morphology fitting."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/sharpa_wave_right.yaml"))
    parser.add_argument(
        "--mano-dir", type=Path, default=Path("../data/models/mano/mano_v1_2/models")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/morphology_initialization.json")
    )
    args = parser.parse_args()

    config = load_yaml(args.config)
    finger_names = config["fingers"]
    mano_semantics = config["mano_semantics"]
    specs = finger_specs_from_config(config)

    mano_model = load_mano_model(args.mano_dir, "right", use_pca=False, flat_hand_mean=True)
    arrays = extract_model_arrays(mano_model)
    mano_joints = arrays.joint_regressor @ arrays.template
    mano_tips = {
        name: arrays.template[mano_semantics[name]["tip_vertex_index"]]
        for name in finger_names
    }

    urdf = UrdfModel.from_file(config["urdf"])
    robot_semantics = config["robot_semantics"]
    robot_wrist, robot_mcps, robot_tips = extract_semantic_points(
        urdf,
        config["anatomical_wrist_link"],
        {name: robot_semantics[name]["mcp_joint"] for name in finger_names},
        {name: robot_semantics[name]["tip_joint"] for name in finger_names},
    )
    palm_scale, finger_scales = initial_scale_ratios(
        mano_joints,
        mano_tips,
        robot_wrist,
        robot_mcps,
        robot_tips,
        specs,
        wrist_joint_index=mano_semantics["wrist_joint_index"],
    )
    scaled = scale_mano_arrays(
        arrays.template,
        arrays.shape_blends,
        arrays.pose_blends,
        arrays.joint_regressor,
        arrays.blend_weights,
        palm_scale,
        finger_scales,
        specs,
        wrist_joint_index=mano_semantics["wrist_joint_index"],
    )
    scaled_template = np.asarray(scaled.template)
    scaled_joints = np.asarray(scaled.rest_joints)
    scaled_tips = {
        name: scaled_template[mano_semantics[name]["tip_vertex_index"]]
        for name in finger_names
    }
    wrist_index = mano_semantics["wrist_joint_index"]
    rotation, axis_angle, _ = initial_rigid_alignment(
        scaled_joints[wrist_index],
        scaled_tips["thumb"],
        scaled_tips["middle"],
        robot_wrist,
        robot_tips["thumb"],
        robot_tips["middle"],
    )
    # MANO global orientation rotates about the regressed wrist; it does not
    # rotate the wrist position about the world origin.
    translation = robot_wrist - scaled_joints[wrist_index]
    payload = {
        "status": "initialization_only_not_optimized",
        "hand": config["hand"],
        "robot_source_revision": config["source_revision"],
        "mano_side": "right",
        "shape_beta": [0.0] * 10,
        "scales": {"palm": palm_scale, **finger_scales},
        "global_rotation_matrix": rotation.tolist(),
        "global_axis_angle_rad": axis_angle.tolist(),
        "translation_m": translation.tolist(),
    }
    write_json(args.output, payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
