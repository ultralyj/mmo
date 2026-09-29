"""Sequential floating-wrist PyRoki IK for MMO robot pose recovery."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import jaxlie
import jaxls
import numpy as np
import pyroki as pk
from pyroki.collision import RobotCollision
import trimesh
import yourdfpy

from ...optimization.metrics import point_rmse, rotation_errors
from ...utils.frames import parse_frame_indices
from ...utils.io import load_yaml, save_npz, write_json


@jaxls.Cost.factory
def temporal_residual(
    values: jaxls.VarValues,
    joint_var: jaxls.Var[jax.Array],
    base_var: jaxls.Var[jaxlie.SE3],
    previous_joints: jax.Array,
    previous_base: jaxlie.SE3,
    joint_weight: jax.Array,
    base_translation_weight: jax.Array,
    base_orientation_weight: jax.Array,
) -> jax.Array:
    """Penalize frame-to-frame joint and floating-base motion."""

    joint_delta = (values[joint_var] - previous_joints) * joint_weight
    base_delta = (previous_base.inverse() @ values[base_var]).log()
    base_delta = base_delta * jnp.concatenate(
        [
            jnp.full((3,), base_translation_weight),
            jnp.full((3,), base_orientation_weight),
        ]
    )
    return jnp.concatenate([joint_delta, base_delta])


@jaxls.Cost.factory
def sampled_table_residual(
    values: jaxls.VarValues,
    robot: pk.Robot,
    joint_var: jaxls.Var[jax.Array],
    base_var: jaxls.Var[jaxlie.SE3],
    sample_points_local: jax.Array,
    sample_link_indices: jax.Array,
    table_height: jax.Array,
    weight: jax.Array,
) -> jax.Array:
    """Penalize sampled collision-surface points below the table plane."""

    base_link_poses = jaxlie.SE3(robot.forward_kinematics(values[joint_var]))
    sample_link_poses = jaxlie.SE3(
        base_link_poses.wxyz_xyz[sample_link_indices]
    )
    points_base = sample_link_poses.apply(sample_points_local)
    points_world = values[base_var].apply(points_base)
    penetration = jnp.maximum(0.0, table_height - points_world[:, 2])
    return penetration * weight / jnp.sqrt(float(sample_points_local.shape[0]))


def _same_finger_ignore_pairs(link_names: tuple[str, ...]) -> tuple[tuple[str, str], ...]:
    """Ignore expected intra-finger and wrist capsule overlaps."""

    def family(name: str) -> str:
        """Map a Wuji link name to its finger collision family."""

        for value, marker in (
            ("thumb", "_thumb_"),
            ("index", "_index_finger_"),
            ("middle", "_middle_finger_"),
            ("ring", "_ring_finger_"),
            ("pinky", "_pinky_"),
        ):
            if marker in name:
                return value
        return "wrist" if "wrist" in name else name

    ignored = []
    for index, left in enumerate(link_names):
        for right in link_names[index + 1 :]:
            if family(left) == family(right) or "wrist" in (family(left), family(right)):
                ignored.append((left, right))
    return tuple(ignored)


def _sample_link_surfaces(
    urdf: yourdfpy.URDF,
    link_names: tuple[str, ...],
    samples_per_link: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample deterministic local points on each link's collision surface."""

    points = []
    indices = []
    for link_index, link_name in enumerate(link_names):
        mesh = RobotCollision._get_trimesh_collision_geometries(urdf, link_name)
        if mesh.is_empty or mesh.area <= 0:
            continue
        sampled, _ = trimesh.sample.sample_surface(
            mesh, samples_per_link, seed=link_index
        )
        points.append(sampled)
        indices.extend([link_index] * len(sampled))
    if not points:
        raise ValueError("URDF has no sampleable collision surfaces")
    return np.concatenate(points).astype(np.float32), np.asarray(indices, dtype=np.int32)


@jdc.jit
def _solve_frame(
    robot: pk.Robot,
    collision: RobotCollision,
    target_pose_matrices: jax.Array,
    target_link_indices: jax.Array,
    previous_joints: jax.Array,
    previous_base_wxyz_xyz: jax.Array,
    surface_points: jax.Array,
    surface_link_indices: jax.Array,
    table_height: jax.Array,
    position_weight: jax.Array,
    orientation_weight: jax.Array,
    joint_velocity_weight: jax.Array,
    base_translation_velocity_weight: jax.Array,
    base_orientation_velocity_weight: jax.Array,
    self_collision_weight: jax.Array,
    self_collision_margin: jax.Array,
    table_weight: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Solve one warm-started floating-base IK frame with safety costs."""

    joint_var = robot.joint_var_cls(0)
    previous_base = jaxlie.SE3(previous_base_wxyz_xyz)

    class FloatingBaseVar(
        jaxls.Var[jaxlie.SE3],
        default_factory=lambda: previous_base,
        tangent_dim=jaxlie.SE3.tangent_dim,
        retract_fn=jaxls.SE3Var.retract_fn,
    ):
        """Per-frame SE(3) variable initialized from the preceding solution."""


    base_var = FloatingBaseVar(0)
    target_poses = jaxlie.SE3.from_matrix(target_pose_matrices)
    costs = [
        pk.costs.pose_cost_with_base(
            robot,
            joint_var,
            base_var,
            target_poses,
            target_link_indices,
            position_weight,
            orientation_weight,
        ),
        temporal_residual(
            joint_var,
            base_var,
            previous_joints,
            previous_base,
            joint_velocity_weight,
            base_translation_velocity_weight,
            base_orientation_velocity_weight,
        ),
        pk.costs.self_collision_cost(
            robot,
            collision,
            joint_var,
            self_collision_margin,
            self_collision_weight,
        ),
        sampled_table_residual(
            robot,
            joint_var,
            base_var,
            surface_points,
            surface_link_indices,
            table_height,
            table_weight,
        ),
        pk.costs.limit_constraint(robot, joint_var),
    ]
    solution = (
        jaxls.LeastSquaresProblem(costs=costs, variables=[joint_var, base_var])
        .analyze()
        .solve(
            initial_vals=jaxls.VarValues.make(
                [joint_var.with_value(previous_joints), base_var]
            ),
            linear_solver="dense_cholesky",
            trust_region=jaxls.TrustRegionConfig(lambda_initial=1.0),
            termination=jaxls.TerminationConfig(max_iterations=100),
            verbose=False,
        )
    )
    return solution[joint_var], solution[base_var].wxyz_xyz


def _world_link_poses(robot: pk.Robot, joints: np.ndarray, base: np.ndarray) -> np.ndarray:
    """Evaluate every robot link pose in the world frame."""

    link_poses = jaxlie.SE3(robot.forward_kinematics(jnp.asarray(joints)))
    world = jaxlie.SE3(jnp.asarray(base)) @ link_poses
    return np.asarray(world.as_matrix())


def main() -> None:
    """Recover a sequential Wuji trajectory from dense link-pose targets."""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=Path("configs/wuji_hand2_beta2_right.yaml")
    )
    parser.add_argument(
        "--targets", type=Path, default=Path("outputs/wuji_hand2/robot_targets.npz")
    )
    parser.add_argument(
        "--contact-match",
        type=Path,
        default=Path("outputs/wuji_hand2/contact_match_full.npz"),
    )
    parser.add_argument(
        "--frames",
        default="1732",
        help="all, comma-separated indices, or START:STOP[:STRIDE]",
    )
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--position-weight", type=float)
    parser.add_argument("--orientation-weight", type=float)
    parser.add_argument("--self-collision-weight", type=float)
    parser.add_argument("--table-weight", type=float)
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/wuji_hand2/robot_pose.npz")
    )
    parser.add_argument(
        "--output-json", type=Path, default=Path("outputs/wuji_hand2/robot_pose.json")
    )
    args = parser.parse_args()

    config = load_yaml(args.config)
    settings = dict(config["pose_recovery"])
    for argument, key in (
        (args.position_weight, "position_residual_weight"),
        (args.orientation_weight, "orientation_residual_weight"),
        (args.self_collision_weight, "self_collision_residual_weight"),
        (args.table_weight, "table_residual_weight"),
    ):
        if argument is not None:
            settings[key] = argument
    targets = np.load(args.targets, allow_pickle=False)
    contact = np.load(args.contact_match, allow_pickle=False)
    frame_count = len(targets["frame_index"])
    selected = parse_frame_indices(args.frames, frame_count)

    urdf_path = Path(config["urdf"]).resolve()
    urdf = yourdfpy.URDF.load(
        str(urdf_path), load_meshes=False, load_collision_meshes=True
    )
    robot = pk.Robot.from_urdf(
        urdf, default_joint_cfg=np.zeros(len(urdf.actuated_joints), dtype=np.float32)
    )
    collision = RobotCollision.from_urdf(
        urdf,
        user_ignore_pairs=_same_finger_ignore_pairs(robot.links.names),
        ignore_immediate_adjacents=True,
    )
    samples, sample_link_indices = _sample_link_surfaces(
        urdf,
        robot.links.names,
        int(settings["table_surface_samples_per_link"]),
    )
    target_link_names = tuple(str(value) for value in targets["target_link_names"])
    target_link_indices = np.asarray(
        [robot.links.names.index(name) for name in target_link_names], dtype=np.int32
    )

    lower = np.asarray(robot.joints.lower_limits)
    upper = np.asarray(robot.joints.upper_limits)
    previous_joints = np.clip(np.zeros(robot.joints.num_actuated_joints), lower, upper)
    first_target = targets["target_link_poses"][int(selected[0]), 0]
    previous_base = np.asarray(jaxlie.SE3.from_matrix(jnp.asarray(first_target)).wxyz_xyz)
    table_heights = contact["table_height_m"]

    joint_trajectory = []
    base_trajectory = []
    position_rmse = []
    orientation_mean = []
    table_penetration = []
    self_collision_minimum = []
    started = time.perf_counter()
    for output_index, frame in enumerate(selected):
        frame = int(frame)
        previous_joints, previous_base = _solve_frame(
            robot,
            collision,
            jnp.asarray(targets["target_link_poses"][frame]),
            jnp.asarray(target_link_indices),
            jnp.asarray(previous_joints),
            jnp.asarray(previous_base),
            jnp.asarray(samples),
            jnp.asarray(sample_link_indices),
            jnp.asarray(table_heights[frame]),
            jnp.asarray(settings["position_residual_weight"]),
            jnp.asarray(settings["orientation_residual_weight"]),
            jnp.asarray(settings["joint_velocity_residual_weight"]),
            jnp.asarray(settings["base_translation_velocity_residual_weight"]),
            jnp.asarray(settings["base_orientation_velocity_residual_weight"]),
            jnp.asarray(settings["self_collision_residual_weight"]),
            jnp.asarray(settings["self_collision_margin_m"]),
            jnp.asarray(settings["table_residual_weight"]),
        )
        previous_joints = np.asarray(previous_joints)
        previous_base = np.asarray(previous_base)
        world_poses = _world_link_poses(robot, previous_joints, previous_base)
        actual = world_poses[target_link_indices]
        target = targets["target_link_poses"][frame]
        position_rmse.append(point_rmse(actual[:, :3, 3], target[:, :3, 3]))
        orientation_mean.append(
            float(np.mean(rotation_errors(actual[:, :3, :3], target[:, :3, :3])))
        )
        sample_world = np.einsum(
            "pij,pj->pi",
            world_poses[sample_link_indices, :3, :3],
            samples,
        ) + world_poses[sample_link_indices, :3, 3]
        table_penetration.append(
            float(np.max(np.maximum(0.0, table_heights[frame] - sample_world[:, 2])))
        )
        collision_distances = collision.compute_self_collision_distance(
            robot, jnp.asarray(previous_joints)
        )
        self_collision_minimum.append(float(np.min(np.asarray(collision_distances))))
        joint_trajectory.append(previous_joints.copy())
        base_trajectory.append(previous_base.copy())
        if (
            output_index == 0
            or output_index + 1 == len(selected)
            or (output_index + 1) % args.progress_every == 0
        ):
            print(
                f"[{output_index + 1}/{len(selected)}] frame {frame}: "
                f"position={position_rmse[-1] * 1000:.2f} mm, "
                f"orientation={np.rad2deg(orientation_mean[-1]):.2f} deg, "
                f"table={table_penetration[-1] * 1000:.2f} mm",
                flush=True,
            )

    elapsed = time.perf_counter() - started
    joint_trajectory = np.asarray(joint_trajectory, dtype=np.float32)
    base_trajectory = np.asarray(base_trajectory, dtype=np.float32)
    save_npz(
        args.output,
        frame_index=targets["frame_index"][selected],
        timestamp_s=targets["timestamp_s"][selected],
        joint_names=np.asarray(robot.joints.actuated_names),
        joint_positions_rad=joint_trajectory,
        base_wxyz_xyz=base_trajectory,
        target_link_names=np.asarray(target_link_names),
        target_link_indices=target_link_indices,
        position_rmse_m=np.asarray(position_rmse, dtype=np.float32),
        orientation_mean_rad=np.asarray(orientation_mean, dtype=np.float32),
        max_table_penetration_m=np.asarray(table_penetration, dtype=np.float32),
        minimum_self_collision_distance_m=np.asarray(
            self_collision_minimum, dtype=np.float32
        ),
        joint_lower_limits_rad=lower,
        joint_upper_limits_rad=upper,
    )
    summary = {
        "status": "optimized",
        "hand": config["hand"],
        "frames_specification": args.frames,
        "processed_frame_count": int(len(selected)),
        "mean_position_rmse_mm": float(np.mean(position_rmse) * 1000.0),
        "mean_orientation_error_deg": float(np.rad2deg(np.mean(orientation_mean))),
        "max_table_penetration_mm": float(np.max(table_penetration) * 1000.0),
        "minimum_self_collision_distance_mm": float(
            np.min(self_collision_minimum) * 1000.0
        ),
        "max_joint_limit_violation_rad": float(
            max(
                np.max(lower - joint_trajectory),
                np.max(joint_trajectory - upper),
                0.0,
            )
        ),
        "elapsed_seconds": elapsed,
        "throughput_frames_per_second": float(len(selected) / elapsed),
        "weights": settings,
        "weights_note": "Reproduction values; the paper does not publish IK weights.",
        "pyroki_revision": "388e43e1fc0d0ee382968d3dd72970fd62a0450c",
        "output": str(args.output),
    }
    write_json(args.output_json, summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
