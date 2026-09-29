"""Generate dense Wuji link pose targets from contact-matched Scaled MANO."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np

from ...models.configuration import morphology_pose_vector, scaled_mano_from_config
from ...models.jax_mano import mano_lbs_with_transforms
from ...models.mano import extract_model_arrays, load_mano_model
from ...models.skeleton_skinning import (
    build_robot_skeleton,
    heat_diffusion_weights,
    linear_blend_targets,
    relative_influence_transforms,
    target_link_poses,
)
from ...models.urdf import UrdfModel
from ...utils.io import load_json, load_yaml, save_npz, write_json


def main() -> None:
    """Diffuse MANO motion onto robot nodes and emit dense link-pose targets."""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=Path("configs/wuji_hand2_beta2_right.yaml")
    )
    parser.add_argument(
        "--mano-dir", type=Path, default=Path("../data/models/mano/mano_v1_2/models")
    )
    parser.add_argument(
        "--morphology-json",
        type=Path,
        default=Path("outputs/wuji_hand2/morphology_match.json"),
    )
    parser.add_argument(
        "--contact-match",
        type=Path,
        default=Path("outputs/wuji_hand2/contact_match_full.npz"),
    )
    parser.add_argument("--source-sigma-m", type=float, default=0.012)
    parser.add_argument("--diffusion-scale", type=float, default=0.05)
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/wuji_hand2/robot_targets.npz")
    )
    parser.add_argument(
        "--output-json", type=Path, default=Path("outputs/wuji_hand2/robot_targets.json")
    )
    args = parser.parse_args()

    started = time.perf_counter()
    config = load_yaml(args.config)
    morphology = load_json(args.morphology_json)
    contact = np.load(args.contact_match, allow_pickle=False)
    parameters = contact["optimized_parameters"]
    if parameters.ndim != 2 or parameters.shape[1] != 51:
        raise ValueError("contact parameters must have shape (T, 51)")

    finger_names = list(config["fingers"])
    semantics = config["mano_semantics"]
    mano_model = load_mano_model(args.mano_dir, "right", use_pca=False, flat_hand_mean=True)
    arrays = extract_model_arrays(mano_model)
    scales = np.asarray(morphology["optimized_scales"], dtype=np.float32)
    scaled = scaled_mano_from_config(arrays, config, scales)

    template = jnp.asarray(scaled.template)
    pose_blends = jnp.asarray(scaled.pose_blends)
    joint_regressor = jnp.asarray(arrays.joint_regressor)
    blend_weights = jnp.asarray(arrays.blend_weights)
    parents = tuple(int(parent) for parent in arrays.parents)

    def forward(frame_parameters: jnp.ndarray):
        """Pose Scaled MANO and expose the rigid transforms used for blending."""

        return mano_lbs_with_transforms(
            template,
            pose_blends,
            joint_regressor,
            blend_weights,
            parents,
            frame_parameters[:3],
            frame_parameters[3:48],
            frame_parameters[48:51],
        )

    morphology_parameters = morphology_pose_vector(morphology)
    morph_vertices, morph_joints, _, morph_transforms = forward(
        jnp.asarray(morphology_parameters)
    )
    morph_vertices = np.asarray(morph_vertices)
    morph_joints = np.asarray(morph_joints)
    morph_transforms = np.asarray(morph_transforms)
    tip_vertex_indices = np.asarray(
        [semantics[name]["tip_vertex_index"] for name in finger_names], dtype=np.int32
    )
    influence_points = np.concatenate(
        [morph_joints, morph_vertices[tip_vertex_indices]], axis=0
    )
    influence_bone_indices = np.asarray(
        [*range(16), *(semantics[name]["skinning_joint_indices"][-1] for name in finger_names)],
        dtype=np.int32,
    )

    urdf = UrdfModel.from_file(config["urdf"])
    skeleton = build_robot_skeleton(
        urdf,
        config["anatomical_wrist_link"],
        {name: config["robot_semantics"][name]["tip_joint"] for name in finger_names},
        config["robot_semantics"]["middle"]["mcp_joint"],
    )
    diffusion_weights = heat_diffusion_weights(
        skeleton.rest_positions,
        skeleton.edges,
        influence_points,
        source_sigma_m=args.source_sigma_m,
        diffusion_scale=args.diffusion_scale,
    )

    batched_forward = jax.jit(jax.vmap(lambda value: forward(value)[3]))
    current_transforms = np.asarray(batched_forward(jnp.asarray(parameters)))
    target_positions = np.empty(
        (len(parameters), len(skeleton.node_names), 3), dtype=np.float32
    )
    target_poses = np.empty(
        (len(parameters), len(skeleton.node_names), 4, 4), dtype=np.float32
    )
    for frame in range(len(parameters)):
        relative = relative_influence_transforms(
            current_transforms[frame], morph_transforms, influence_bone_indices
        )
        positions, linear_maps = linear_blend_targets(
            skeleton.rest_positions, diffusion_weights, relative
        )
        target_positions[frame] = positions
        target_poses[frame] = target_link_poses(skeleton, positions, linear_maps)

    identity_relative = np.tile(np.eye(4), (len(influence_bone_indices), 1, 1))
    identity_positions, identity_linear = linear_blend_targets(
        skeleton.rest_positions, diffusion_weights, identity_relative
    )
    identity_poses = target_link_poses(skeleton, identity_positions, identity_linear)
    rest_link_transforms, _ = urdf.forward_kinematics()
    expected_rest_poses = np.asarray(
        [rest_link_transforms[name] for name in skeleton.link_names]
    )
    identity_position_error = float(
        np.max(np.linalg.norm(identity_poses[:, :3, 3] - expected_rest_poses[:, :3, 3], axis=1))
    )
    identity_rotation_error = float(
        np.max(np.abs(identity_poses[:, :3, :3] - expected_rest_poses[:, :3, :3]))
    )

    save_npz(
        args.output,
        frame_index=contact["frame_index"],
        timestamp_s=contact["timestamp_s"],
        target_link_poses=target_poses,
        target_node_positions_m=target_positions,
        target_link_names=np.asarray(skeleton.link_names),
        skeleton_node_names=np.asarray(skeleton.node_names),
        skeleton_rest_positions_m=skeleton.rest_positions.astype(np.float32),
        skeleton_parent_indices=skeleton.parent_indices,
        skeleton_edges=skeleton.edges,
        mano_influence_points_m=influence_points.astype(np.float32),
        mano_influence_bone_indices=influence_bone_indices,
        diffusion_weights=diffusion_weights,
        morphology_scales=scales,
    )
    summary = {
        "status": "generated",
        "hand": config["hand"],
        "frame_count": int(len(parameters)),
        "robot_node_count": len(skeleton.node_names),
        "mano_influence_count": int(len(influence_points)),
        "source_sigma_m": args.source_sigma_m,
        "diffusion_scale": args.diffusion_scale,
        "weight_row_sum_max_error": float(
            np.max(np.abs(np.sum(diffusion_weights, axis=1) - 1.0))
        ),
        "morphology_identity_position_error_m": identity_position_error,
        "morphology_identity_rotation_matrix_max_error": identity_rotation_error,
        "elapsed_seconds": time.perf_counter() - started,
        "output": str(args.output),
        "method_note": (
            "1D finite-element/cotangent analogue with backward-Euler heat diffusion; "
            "the paper does not publish diffusion hyperparameters."
        ),
    }
    write_json(args.output_json, summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
