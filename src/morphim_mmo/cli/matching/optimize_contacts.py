"""Per-frame contact matching for a morphology-aligned MANO model.

Implements Section IV-B / equations 8--10 of MMO for a five-finger robot.
The coupled-finger term is absent because every human finger maps one-to-one.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import least_squares

from ...data.contact import table_top_heights
from ...models.configuration import morphology_pose_vector, scaled_mano_from_config
from ...models.jax_mano import axis_angle_to_matrix, mano_lbs
from ...models.mano import extract_model_arrays, load_mano_model
from ...optimization.losses import ContactLossWeights, contact_residuals
from ...optimization.jaxls_mmo import solve_contact_trajectory_jaxls
from ...optimization.metrics import point_rmse, rotation_error_statistics
from ...utils.frames import parse_frame_indices
from ...utils.io import load_json, load_yaml, save_npz, write_json


def _contact_slice(
    offsets: np.ndarray,
    flat_indices: np.ndarray,
    flat_positions: np.ndarray,
    frame: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract one frame from flattened ragged contact arrays."""

    start, stop = int(offsets[frame]), int(offsets[frame + 1])
    return flat_indices[start:stop], flat_positions[start:stop]


def _pack_contacts(
    indices: np.ndarray, positions: np.ndarray, capacity: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pad variable-size contacts to a fixed JAX compilation shape."""

    count = len(indices)
    if count > capacity:
        raise ValueError(f"contact count {count} exceeds padding capacity {capacity}")
    padded_indices = np.zeros(capacity, dtype=np.int32)
    padded_positions = np.zeros((capacity, 3), dtype=np.float32)
    mask = np.zeros(capacity, dtype=np.float32)
    padded_indices[:count] = indices
    padded_positions[:count] = positions
    mask[:count] = 1.0
    return padded_indices, padded_positions, mask


def main() -> None:
    """Run sequential contact matching for selected GRAB frames."""

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
        "--hand-trajectory",
        type=Path,
        default=Path("../data/processed/s1/apple_lift_rhand_mano.npz"),
    )
    parser.add_argument(
        "--contacts",
        type=Path,
        default=Path("../data/processed/s1/apple_lift_contacts.npz"),
    )
    parser.add_argument(
        "--frames",
        default="max-contact",
        help="all, max-contact, comma-separated indices, or START:STOP[:STRIDE]",
    )
    parser.add_argument("--max-evaluations", type=int, default=200)
    parser.add_argument(
        "--backend",
        choices=("jaxls", "scipy"),
        default="jaxls",
        help="JAXLS runs the sequential trajectory on GPU; SciPy is the baseline.",
    )
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--contact-weight", type=float)
    parser.add_argument("--table-weight", type=float)
    parser.add_argument("--pose-weight", type=float)
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/wuji_hand2/contact_match.npz")
    )
    parser.add_argument(
        "--output-json", type=Path, default=Path("outputs/wuji_hand2/contact_match.json")
    )
    args = parser.parse_args()

    config = load_yaml(args.config)
    morphology = load_json(args.morphology_json)
    hand_data = np.load(args.hand_trajectory, allow_pickle=False)
    contact_data = np.load(args.contacts, allow_pickle=False)
    hand_frame_index = hand_data["frame_index"]
    timestamps = hand_data["timestamp_s"]
    reference_global_orient = hand_data["mano_global_orient"]
    reference_hand_pose = hand_data["mano_hand_pose"]
    reference_vertices = hand_data["hand_vertices_m"]
    contact_timestamps = contact_data["timestamp_s"]
    contact_offsets = contact_data["contact_offsets"]
    flat_contact_indices = contact_data["contact_indices"]
    flat_contact_positions = contact_data["contact_positions_m"]
    table_vertices_local = contact_data["table_vertices_local_m"]
    table_to_world = contact_data["table_to_world"]
    frame_count = len(hand_frame_index)
    if len(contact_timestamps) != frame_count:
        raise ValueError("hand and contact trajectories have different frame counts")
    if not np.allclose(timestamps, contact_timestamps):
        raise ValueError("hand and contact timestamps do not match")
    if args.progress_every <= 0:
        raise ValueError("progress-every must be positive")

    max_contact_frame = int(contact_data["max_contact_frame"])
    selected_frames = parse_frame_indices(
        args.frames,
        frame_count,
        aliases={"max-contact": max_contact_frame},
    )
    contact_capacity = int(np.max(np.diff(contact_offsets)))
    table_heights = table_top_heights(table_vertices_local, table_to_world)

    mano_model = load_mano_model(args.mano_dir, "right", use_pca=False, flat_hand_mean=True)
    arrays = extract_model_arrays(mano_model)
    finger_names = list(config["fingers"])
    semantics = config["mano_semantics"]
    scales = np.asarray(morphology["optimized_scales"], dtype=np.float32)
    scaled = scaled_mano_from_config(arrays, config, scales)

    template = jnp.asarray(scaled.template)
    pose_blends = jnp.asarray(scaled.pose_blends)
    joint_regressor = jnp.asarray(arrays.joint_regressor)
    blend_weights = jnp.asarray(arrays.blend_weights)
    parents = tuple(int(parent) for parent in arrays.parents)
    configured_weights = config["loss_weights"]
    weights = ContactLossWeights(
        contact=(
            float(args.contact_weight)
            if args.contact_weight is not None
            else float(configured_weights["contact"])
        ),
        table=(
            float(args.table_weight)
            if args.table_weight is not None
            else float(configured_weights["table"])
        ),
        pose=(
            float(args.pose_weight)
            if args.pose_weight is not None
            else float(configured_weights["pose"])
        ),
    )
    if min(weights.contact, weights.table, weights.pose) < 0:
        raise ValueError("loss weights must be non-negative")

    def forward(parameters: jnp.ndarray):
        """Evaluate the fixed-morphology MANO model for one pose vector."""

        return mano_lbs(
            template,
            pose_blends,
            joint_regressor,
            blend_weights,
            parents,
            parameters[:3],
            parameters[3:48],
            parameters[48:51],
        )

    def residual(
        parameters: jnp.ndarray,
        contact_indices: jnp.ndarray,
        target_positions: jnp.ndarray,
        contact_mask: jnp.ndarray,
        reference_rotations: jnp.ndarray,
        table_height: jnp.ndarray,
        table_mask: jnp.ndarray,
    ) -> jnp.ndarray:
        """Assemble contact, table, and pose residuals for one frame."""

        vertices, _, rotations = forward(parameters)
        return contact_residuals(
            vertices,
            contact_indices,
            target_positions,
            rotations,
            reference_rotations,
            table_height,
            weights,
            include_table=True,
            contact_mask=contact_mask,
            table_mask=table_mask,
        )

    residual_jit = jax.jit(residual)
    jacobian_jit = jax.jit(jax.jacfwd(residual, argnums=0))

    morphology_parameters = morphology_pose_vector(
        morphology,
        dtype=np.float32 if args.backend == "jaxls" else np.float64,
    )
    tip_indices = np.asarray(
        [semantics[name]["tip_vertex_index"] for name in finger_names], dtype=np.int32
    )

    started = time.perf_counter()
    padded_indices_by_frame = []
    padded_positions_by_frame = []
    contact_masks_by_frame = []
    reference_rotations_by_frame = []
    table_loss_enabled = []
    contact_slices = []
    for frame_value in selected_frames:
        frame = int(frame_value)
        indices, positions = _contact_slice(
            contact_offsets, flat_contact_indices, flat_contact_positions, frame
        )
        padded_indices, padded_positions, contact_mask = _pack_contacts(
            indices, positions, contact_capacity
        )
        reference_pose = np.concatenate(
            [reference_global_orient[frame], reference_hand_pose[frame]]
        ).reshape(16, 3)
        reference_rotations = np.asarray(
            axis_angle_to_matrix(jnp.asarray(reference_pose))
        )
        table_height = float(table_heights[frame])
        # This is the exact omission condition stated in Section IV-B.
        use_table = not bool(
            np.all(reference_vertices[frame, tip_indices, 2] < table_height)
        )
        padded_indices_by_frame.append(padded_indices)
        padded_positions_by_frame.append(padded_positions)
        contact_masks_by_frame.append(contact_mask)
        reference_rotations_by_frame.append(reference_rotations)
        table_loss_enabled.append(use_table)
        contact_slices.append((indices, positions))

    padded_indices_array = np.asarray(padded_indices_by_frame, dtype=np.int32)
    padded_positions_array = np.asarray(padded_positions_by_frame, dtype=np.float32)
    contact_masks_array = np.asarray(contact_masks_by_frame, dtype=np.float32)
    reference_rotations_array = np.asarray(
        reference_rotations_by_frame, dtype=np.float32
    )
    selected_table_heights = np.asarray(
        table_heights[selected_frames], dtype=np.float32
    )
    table_masks_array = np.asarray(table_loss_enabled, dtype=np.float32)

    if args.backend == "jaxls":
        (
            initial_parameter_trajectory,
            optimized_parameter_trajectory,
            evaluation_array,
            convergence_array,
        ) = solve_contact_trajectory_jaxls(
            jnp.asarray(morphology_parameters),
            template,
            pose_blends,
            joint_regressor,
            blend_weights,
            jnp.asarray(padded_indices_array),
            jnp.asarray(padded_positions_array),
            jnp.asarray(contact_masks_array),
            jnp.asarray(reference_rotations_array),
            jnp.asarray(selected_table_heights),
            jnp.asarray(table_masks_array),
            jnp.asarray(weights.contact, dtype=jnp.float32),
            jnp.asarray(weights.table, dtype=jnp.float32),
            jnp.asarray(weights.pose, dtype=jnp.float32),
            args.max_evaluations,
        )
        optimized_parameter_trajectory.block_until_ready()
        initial_parameter_trajectory = np.asarray(initial_parameter_trajectory)
        optimized_parameters = np.asarray(optimized_parameter_trajectory)
        evaluations = np.asarray(evaluation_array, dtype=np.int32)
        converged = np.asarray(convergence_array, dtype=np.bool_)
        successes = np.isfinite(optimized_parameters).all(axis=1)
    else:
        warm_parameters = morphology_parameters
        initial_parameters = []
        optimized_parameters_list = []
        evaluations_list = []
        successes_list = []
        for frame_inputs in zip(
            padded_indices_array,
            padded_positions_array,
            contact_masks_array,
            reference_rotations_array,
            selected_table_heights,
            table_masks_array,
            strict=True,
        ):
            dynamic_args = tuple(jnp.asarray(value) for value in frame_inputs)

            def scipy_residual(parameters: np.ndarray) -> np.ndarray:
                """Convert the compiled JAX residual to SciPy's NumPy interface."""

                return np.asarray(
                    residual_jit(jnp.asarray(parameters), *dynamic_args),
                    dtype=np.float64,
                )

            def scipy_jacobian(parameters: np.ndarray) -> np.ndarray:
                """Convert the exact JAX Jacobian to SciPy's NumPy interface."""

                return np.asarray(
                    jacobian_jit(jnp.asarray(parameters), *dynamic_args),
                    dtype=np.float64,
                )

            initial_parameters.append(warm_parameters.copy())
            result = least_squares(
                scipy_residual,
                warm_parameters,
                jac=scipy_jacobian,
                method="lm",
                max_nfev=args.max_evaluations,
            )
            warm_parameters = result.x
            optimized_parameters_list.append(result.x.copy())
            evaluations_list.append(int(result.nfev))
            successes_list.append(bool(result.success))
        initial_parameter_trajectory = np.asarray(initial_parameters)
        optimized_parameters = np.asarray(optimized_parameters_list)
        evaluations = np.asarray(evaluations_list, dtype=np.int32)
        successes = np.asarray(successes_list, dtype=np.bool_)
        converged = successes.copy()

    batched_forward = jax.jit(jax.vmap(forward))
    combined_parameters = jnp.asarray(
        np.concatenate([initial_parameter_trajectory, optimized_parameters], axis=0)
    )
    all_vertices, all_joints, all_rotations = batched_forward(combined_parameters)
    all_vertices = np.asarray(all_vertices)
    all_joints = np.asarray(all_joints)
    all_rotations = np.asarray(all_rotations)
    trajectory_length = len(selected_frames)
    initial_vertices_by_frame = all_vertices[:trajectory_length]
    optimized_vertices_by_frame = all_vertices[trajectory_length:]
    optimized_joints_by_frame = all_joints[trajectory_length:]
    optimized_rotations_by_frame = all_rotations[trajectory_length:]

    initial_contact_rmse = []
    optimized_contact_rmse = []
    initial_table_penetration = []
    optimized_table_penetration = []
    optimized_pose_mean = []
    optimized_pose_max = []
    for output_index, frame_value in enumerate(selected_frames):
        frame = int(frame_value)
        indices, positions = contact_slices[output_index]
        initial_vertices = initial_vertices_by_frame[output_index]
        vertices = optimized_vertices_by_frame[output_index]
        reference_rotations = reference_rotations_array[output_index]
        rotations = optimized_rotations_by_frame[output_index]
        table_height = float(selected_table_heights[output_index])
        use_table = bool(table_loss_enabled[output_index])
        pose_mean, pose_max = rotation_error_statistics(
            reference_rotations, rotations
        )
        initial_contact_rmse.append(point_rmse(initial_vertices[indices], positions))
        optimized_contact_rmse.append(point_rmse(vertices[indices], positions))
        initial_table_penetration.append(
            float(np.max(np.maximum(0.0, table_height - initial_vertices[:, 2])))
            if use_table
            else 0.0
        )
        optimized_table_penetration.append(
            float(np.max(np.maximum(0.0, table_height - vertices[:, 2])))
            if use_table
            else 0.0
        )
        optimized_pose_mean.append(pose_mean)
        optimized_pose_max.append(pose_max)
        if (
            output_index == 0
            or output_index + 1 == len(selected_frames)
            or (output_index + 1) % args.progress_every == 0
            or not successes[output_index]
        ):
            print(
                f"[{output_index + 1}/{len(selected_frames)}] frame {frame}: "
                f"contact {initial_contact_rmse[-1] * 1000:.2f} -> "
                f"{optimized_contact_rmse[-1] * 1000:.2f} mm, "
                f"iterations={int(evaluations[output_index])}, "
                f"success={bool(successes[output_index])}, "
                f"converged={bool(converged[output_index])}",
                flush=True,
            )

    elapsed = time.perf_counter() - started
    result_arrays = {
        "frame_index": selected_frames,
        "timestamp_s": timestamps[selected_frames],
        "optimized_parameters": np.asarray(optimized_parameters, dtype=np.float32),
        "initial_vertices_m": np.asarray(initial_vertices_by_frame, dtype=np.float32),
        "optimized_vertices_m": np.asarray(optimized_vertices_by_frame, dtype=np.float32),
        "optimized_joints_m": np.asarray(optimized_joints_by_frame, dtype=np.float32),
        "reference_vertices_m": reference_vertices[selected_frames],
        "mano_faces": arrays.faces,
        "tip_vertex_indices": tip_indices,
        "table_height_m": table_heights[selected_frames],
        "table_loss_enabled": np.asarray(table_loss_enabled, dtype=np.bool_),
        "initial_contact_rmse_m": np.asarray(initial_contact_rmse, dtype=np.float32),
        "optimized_contact_rmse_m": np.asarray(optimized_contact_rmse, dtype=np.float32),
        "initial_table_penetration_m": np.asarray(initial_table_penetration, dtype=np.float32),
        "optimized_table_penetration_m": np.asarray(optimized_table_penetration, dtype=np.float32),
        "optimized_pose_mean_rad": np.asarray(optimized_pose_mean, dtype=np.float32),
        "optimized_pose_max_rad": np.asarray(optimized_pose_max, dtype=np.float32),
        "optimizer_success": np.asarray(successes, dtype=np.bool_),
        "optimizer_converged": np.asarray(converged, dtype=np.bool_),
        "optimizer_evaluations": np.asarray(evaluations, dtype=np.int32),
        "morphology_scales": scales,
    }
    save_npz(args.output, **result_arrays)
    summary = {
        "status": "optimized" if all(successes) else "optimizer_stopped_on_some_frames",
        "hand": config["hand"],
        "source_sequence": str(hand_data["source_sequence"]),
        "frames_specification": args.frames,
        "processed_frame_count": int(len(selected_frames)),
        "first_frame": int(selected_frames[0]),
        "last_frame": int(selected_frames[-1]),
        "max_contact_frame": max_contact_frame,
        "contact_padding_capacity": contact_capacity,
        "loss_weights": {
            "contact": weights.contact,
            "table": weights.table,
            "pose": weights.pose,
        },
        "loss_weights_note": "Reproduction values; the paper does not publish weights.",
        "backend": args.backend,
        "solver": (
            "on-device JAXLS Levenberg-Marquardt inside jax.lax.scan"
            if args.backend == "jaxls"
            else "SciPy Levenberg-Marquardt with exact JAX Jacobian"
        ),
        "jax_backend": jax.default_backend(),
        "jax_devices": [str(device) for device in jax.devices()],
        "warm_start_note": (
            "First processed frame starts from morphology matching; each later processed "
            "frame starts from the preceding processed solution. Use --frames all for the "
            "paper's exact consecutive-frame schedule."
        ),
        "mean_initial_contact_rmse_mm": float(np.mean(initial_contact_rmse) * 1000.0),
        "mean_optimized_contact_rmse_mm": float(np.mean(optimized_contact_rmse) * 1000.0),
        "max_optimized_table_penetration_mm": float(
            np.max(optimized_table_penetration) * 1000.0
        ),
        "mean_pose_deviation_deg": float(np.rad2deg(np.mean(optimized_pose_mean))),
        "successful_frames": int(np.count_nonzero(successes)),
        "converged_frames": int(np.count_nonzero(converged)),
        "elapsed_seconds": elapsed,
        "throughput_frames_per_second": float(len(selected_frames) / elapsed),
        "output": str(args.output),
    }
    write_json(args.output_json, summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
