"""Optimize Scaled MANO to the neutral Sharpa Wave skeleton (paper Eq. 7)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np

from ...models.jax_mano import mano_lbs
from ...models.configuration import finger_specs_from_config
from ...models.mano import extract_model_arrays, load_mano_model
from ...models.scaled_mano import initial_scale_ratios, scale_mano_arrays
from ...models.urdf import UrdfModel, extract_semantic_points
from ...optimization.alignment import initial_rigid_alignment
from ...optimization.losses import morphology_residuals
from ...optimization.jaxls_mmo import solve_morphology_jaxls
from ...optimization.metrics import point_rmse
from ...optimization.solver import solve_bounded_least_squares, solve_lm
from ...utils.io import load_yaml, save_npz, write_json


def main() -> None:
    """Fit Scaled MANO to neutral robot joints and fingertips."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/sharpa_wave_right.yaml"))
    parser.add_argument(
        "--mano-dir", type=Path, default=Path("../data/models/mano/mano_v1_2/models")
    )
    parser.add_argument("--max-evaluations", type=int, default=500)
    parser.add_argument(
        "--backend",
        choices=("jaxls", "scipy"),
        default="jaxls",
        help="Use GPU-compatible JAXLS or the SciPy baseline.",
    )
    parser.add_argument("--solver", choices=("bounded", "lm"), default="bounded")
    parser.add_argument(
        "--pose-component-limit-rad",
        type=float,
        default=0.35,
        help="Per-axis trust region around the zero MANO hand pose.",
    )
    parser.add_argument(
        "--output-json", type=Path, default=Path("outputs/morphology_match.json")
    )
    parser.add_argument(
        "--output-geometry", type=Path, default=Path("outputs/morphology_match.npz")
    )
    args = parser.parse_args()

    config = load_yaml(args.config)
    finger_names = config["fingers"]
    mano_semantics = config["mano_semantics"]
    robot_semantics = config["robot_semantics"]
    specs = finger_specs_from_config(config)

    mano_model = load_mano_model(args.mano_dir, "right", use_pca=False, flat_hand_mean=True)
    arrays = extract_model_arrays(mano_model)
    urdf = UrdfModel.from_file(config["urdf"])
    link_transforms, robot_joint_positions = urdf.forward_kinematics()
    robot_wrist, robot_mcps, robot_tips_by_name = extract_semantic_points(
        urdf,
        config["anatomical_wrist_link"],
        {name: robot_semantics[name]["mcp_joint"] for name in finger_names},
        {name: robot_semantics[name]["tip_joint"] for name in finger_names},
    )

    mano_rest_joints = arrays.joint_regressor @ arrays.template
    mano_tips_by_name = {
        name: arrays.template[mano_semantics[name]["tip_vertex_index"]]
        for name in finger_names
    }
    palm_scale, finger_scale_map = initial_scale_ratios(
        mano_rest_joints,
        mano_tips_by_name,
        robot_wrist,
        robot_mcps,
        robot_tips_by_name,
        specs,
        wrist_joint_index=mano_semantics["wrist_joint_index"],
    )
    initial_scales = np.asarray(
        [palm_scale] + [finger_scale_map[name] for name in finger_names], dtype=np.float32
    )
    initial_scaled = scale_mano_arrays(
        arrays.template,
        arrays.shape_blends,
        arrays.pose_blends,
        arrays.joint_regressor,
        arrays.blend_weights,
        initial_scales[0],
        dict(zip(finger_names, initial_scales[1:], strict=True)),
        specs,
        wrist_joint_index=mano_semantics["wrist_joint_index"],
    )
    initial_template = np.asarray(initial_scaled.template)
    initial_joints_rest = np.asarray(initial_scaled.rest_joints)
    initial_tips = {
        name: initial_template[mano_semantics[name]["tip_vertex_index"]]
        for name in finger_names
    }
    rotation, axis_angle, _ = initial_rigid_alignment(
        initial_joints_rest[mano_semantics["wrist_joint_index"]],
        initial_tips["thumb"],
        initial_tips["middle"],
        robot_wrist,
        robot_tips_by_name["thumb"],
        robot_tips_by_name["middle"],
    )
    # MANO's global orientation rotates about the regressed wrist, so unlike a
    # conventional origin-centred rigid transform it does not rotate the wrist
    # position itself.
    translation = robot_wrist - initial_joints_rest[mano_semantics["wrist_joint_index"]]

    mano_joint_indices = [mano_semantics["wrist_joint_index"]]
    robot_joint_targets = [robot_wrist]
    for name in finger_names:
        mcp_index = mano_semantics[name]["mcp_joint_index"]
        mano_joint_indices.extend([mcp_index, mcp_index + 1, mcp_index + 2])
        robot_joint_targets.extend(
            [
                robot_joint_positions[robot_semantics[name]["mcp_joint"]],
                robot_joint_positions[robot_semantics[name]["pip_joint"]],
                robot_joint_positions[robot_semantics[name]["dip_joint"]],
            ]
        )
    mano_joint_indices = np.asarray(mano_joint_indices, dtype=np.int32)
    robot_joint_targets = np.asarray(robot_joint_targets, dtype=np.float32)
    tip_vertex_indices = np.asarray(
        [mano_semantics[name]["tip_vertex_index"] for name in finger_names],
        dtype=np.int32,
    )
    robot_tip_targets = np.asarray(
        [robot_tips_by_name[name] for name in finger_names], dtype=np.float32
    )

    template = jnp.asarray(arrays.template)
    shape_blends = jnp.asarray(arrays.shape_blends)
    pose_blends = jnp.asarray(arrays.pose_blends)
    joint_regressor = jnp.asarray(arrays.joint_regressor)
    blend_weights = jnp.asarray(arrays.blend_weights)
    parents = tuple(int(parent) for parent in arrays.parents)
    joint_indices = jnp.asarray(mano_joint_indices)
    tip_indices = jnp.asarray(tip_vertex_indices)
    joint_targets = jnp.asarray(robot_joint_targets)
    tip_targets = jnp.asarray(robot_tip_targets)

    def forward(parameters: jnp.ndarray):
        """Decode log-scales and pose parameters into posed MANO geometry."""

        scales = jnp.exp(parameters[:6])
        scaled = scale_mano_arrays(
            template,
            shape_blends,
            pose_blends,
            joint_regressor,
            blend_weights,
            scales[0],
            {name: scales[index + 1] for index, name in enumerate(finger_names)},
            specs,
            wrist_joint_index=mano_semantics["wrist_joint_index"],
        )
        vertices, joints, rotations = mano_lbs(
            scaled.template,
            scaled.pose_blends,
            joint_regressor,
            blend_weights,
            parents,
            parameters[6:9],
            parameters[9:54],
            parameters[54:57],
        )
        return scales, vertices, joints, rotations

    def residual(parameters: jnp.ndarray) -> jnp.ndarray:
        """Evaluate the paper's joint and fingertip morphology residuals."""

        _, vertices, joints, _ = forward(parameters)
        return morphology_residuals(
            joints[joint_indices],
            joint_targets,
            vertices[tip_indices],
            tip_targets,
            joint_weight=float(config["loss_weights"]["joint"]),
            fingertip_weight=float(config["loss_weights"]["fingertip"]),
        )

    initial_parameters = np.concatenate(
        [np.log(initial_scales), axis_angle, np.zeros(45), translation]
    )
    _, initial_vertices, initial_posed_joints, _ = forward(jnp.asarray(initial_parameters))
    lower_bounds = np.concatenate(
        [
            np.log(initial_scales * 0.5),
            np.full(3, -np.pi),
            np.full(45, -args.pose_component_limit_rad),
            translation - 0.2,
        ]
    )
    upper_bounds = np.concatenate(
        [
            np.log(initial_scales * 1.5),
            np.full(3, np.pi),
            np.full(45, args.pose_component_limit_rad),
            translation + 0.2,
        ]
    )
    started = time.perf_counter()
    if args.backend == "jaxls":
        finger_memberships = np.stack(
            [
                np.sum(
                    arrays.blend_weights[:, np.asarray(spec.skinning_joint_indices)],
                    axis=1,
                )
                for spec in specs
            ],
            axis=1,
        ).astype(np.float32)
        finger_mcp_indices = np.asarray(
            [spec.mcp_joint_index for spec in specs], dtype=np.int32
        )
        solution_parameters, iterations, converged = solve_morphology_jaxls(
            jnp.asarray(initial_parameters, dtype=jnp.float32),
            jnp.asarray(lower_bounds, dtype=jnp.float32),
            jnp.asarray(upper_bounds, dtype=jnp.float32),
            template,
            pose_blends,
            joint_regressor,
            blend_weights,
            jnp.asarray(finger_memberships),
            jnp.asarray(finger_mcp_indices),
            jnp.asarray(mano_semantics["wrist_joint_index"], dtype=jnp.int32),
            joint_indices,
            tip_indices,
            joint_targets,
            tip_targets,
            jnp.asarray(config["loss_weights"]["joint"], dtype=jnp.float32),
            jnp.asarray(config["loss_weights"]["fingertip"], dtype=jnp.float32),
            args.max_evaluations,
        )
        solution_parameters.block_until_ready()
        solution_parameters = np.asarray(solution_parameters)
        result_success = bool(np.isfinite(solution_parameters).all())
        result_message = (
            "JAXLS convergence criterion reached"
            if bool(np.asarray(converged))
            else "JAXLS returned a finite solution at its stopping condition"
        )
        result_optimality = None
        result_evaluations = int(np.asarray(iterations))
        method = "bounded JAXLS Levenberg-Marquardt on JAX"
        bounds_note = (
            "Finite baseline bounds are enforced by a differentiable tanh "
            "reparameterization."
        )
    elif args.solver == "lm":
        result = solve_lm(
            residual, initial_parameters, max_evaluations=args.max_evaluations
        )
        solution_parameters = result.parameters
        result_success = result.success
        result_message = result.message
        result_optimality = result.optimality
        result_evaluations = result.evaluations
        method = "Levenberg-Marquardt with exact JAX Jacobian"
        bounds_note = "Unbounded, matching the solver stated in the paper."
    else:
        result = solve_bounded_least_squares(
            residual,
            initial_parameters,
            lower_bounds,
            upper_bounds,
            max_evaluations=args.max_evaluations,
        )
        solution_parameters = result.parameters
        result_success = result.success
        result_message = result.message
        result_optimality = result.optimality
        result_evaluations = result.evaluations
        method = "bounded trust-region reflective with exact JAX Jacobian"
        bounds_note = (
            "Scale bounds are 0.5x--1.5x initialization; "
            f"local hand axis-angle components are bounded to "
            f"+/-{args.pose_component_limit_rad:g} rad."
        )
    elapsed = time.perf_counter() - started
    scales, vertices, joints, rotations = forward(jnp.asarray(solution_parameters))
    verified_residual = np.asarray(residual(jnp.asarray(solution_parameters)))
    verified_cost = float(0.5 * np.sum(verified_residual**2))
    verified_jit_residual = np.asarray(
        jax.jit(residual)(jnp.asarray(solution_parameters))
    )
    verified_jit_cost = float(0.5 * np.sum(verified_jit_residual**2))

    initial_vertices = np.asarray(initial_vertices)
    initial_posed_joints = np.asarray(initial_posed_joints)
    vertices = np.asarray(vertices)
    joints = np.asarray(joints)
    rotations = np.asarray(rotations)
    scales = np.asarray(scales)
    metrics = {
        "initial_joint_rmse_m": point_rmse(
            initial_posed_joints[mano_joint_indices], robot_joint_targets
        ),
        "optimized_joint_rmse_m": point_rmse(
            joints[mano_joint_indices], robot_joint_targets
        ),
        "initial_fingertip_rmse_m": point_rmse(
            initial_vertices[tip_vertex_indices], robot_tip_targets
        ),
        "optimized_fingertip_rmse_m": point_rmse(
            vertices[tip_vertex_indices], robot_tip_targets
        ),
    }
    payload = {
        "status": "optimized" if result_success else "optimizer_stopped",
        "hand": config["hand"],
        "robot_source_revision": config["source_revision"],
        "mano_side": "right",
        "shape_beta": [0.0] * 10,
        "scale_order": ["palm", *finger_names],
        "initial_scales": initial_scales.tolist(),
        "optimized_scales": scales.tolist(),
        "global_axis_angle_rad": solution_parameters[6:9].tolist(),
        "hand_pose_axis_angle_rad": solution_parameters[9:54].reshape(15, 3).tolist(),
        "translation_m": solution_parameters[54:57].tolist(),
        "metrics": metrics,
        "optimizer": {
            "method": method,
            "backend": args.backend,
            "bounds_note": bounds_note,
            "success": result_success,
            "message": result_message,
            "cost": verified_cost,
            "verified_cost": verified_cost,
            "verified_jit_cost": verified_jit_cost,
            "optimality": result_optimality,
            "evaluations": result_evaluations,
            "elapsed_seconds": elapsed,
            "jax_backend": jax.default_backend(),
            "jax_devices": [str(device) for device in jax.devices()],
        },
        "loss_weights_note": "Placeholder weights: the paper does not publish values.",
    }
    write_json(args.output_json, payload)
    save_npz(
        args.output_geometry,
        initial_vertices_m=initial_vertices,
        optimized_vertices_m=vertices,
        mano_faces=arrays.faces,
        initial_joints_m=initial_posed_joints,
        optimized_joints_m=joints,
        optimized_joint_rotations=rotations,
        mano_correspondence_joint_indices=mano_joint_indices,
        mano_tip_vertex_indices=tip_vertex_indices,
        robot_joint_targets_m=robot_joint_targets,
        robot_tip_targets_m=robot_tip_targets,
        optimized_scales=scales,
        initial_scales=initial_scales,
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
