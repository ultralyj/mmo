"""GPU-oriented JAXLS solvers for MMO morphology and contact matching."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import jaxls

from ..models.jax_mano import MANO_PARENT_INDICES, mano_lbs
from .losses import ContactLossWeights, contact_residuals, morphology_residuals


class MorphologyRawVar(
    jaxls.Var[jax.Array], default_factory=lambda: jnp.zeros((57,), dtype=jnp.float32)
):
    """Unconstrained coordinates mapped smoothly into morphology bounds."""


def _bounded_parameters(
    raw_parameters: jax.Array,
    lower_bounds: jax.Array,
    upper_bounds: jax.Array,
) -> jax.Array:
    """Map unconstrained values into finite open bounds with ``tanh``."""

    midpoint = (lower_bounds + upper_bounds) / 2.0
    half_range = (upper_bounds - lower_bounds) / 2.0
    return midpoint + half_range * jnp.tanh(raw_parameters)


def _raw_parameters(
    bounded_parameters: jax.Array,
    lower_bounds: jax.Array,
    upper_bounds: jax.Array,
) -> jax.Array:
    """Map bounded initialization values into unconstrained coordinates."""

    midpoint = (lower_bounds + upper_bounds) / 2.0
    half_range = (upper_bounds - lower_bounds) / 2.0
    normalized = (bounded_parameters - midpoint) / half_range
    return jnp.arctanh(jnp.clip(normalized, -0.999999, 0.999999))


def _scale_mano_dense(
    template: jax.Array,
    pose_blends: jax.Array,
    joint_regressor: jax.Array,
    scales: jax.Array,
    finger_memberships: jax.Array,
    finger_mcp_indices: jax.Array,
    wrist_joint_index: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Vectorized form of Scaled MANO equations 5--6 for JAXLS tracing."""

    palm_scale = scales[0]
    finger_scales = scales[1:]
    wrist = joint_regressor[wrist_joint_index] @ template
    palm_template = wrist + palm_scale * (template - wrist)
    palm_pose_blends = palm_scale * pose_blends
    mcps = joint_regressor[finger_mcp_indices] @ palm_template
    alpha_minus_one = finger_scales / palm_scale - 1.0
    offsets = palm_template[:, None, :] - mcps[None, :, :]
    template_delta = jnp.sum(
        finger_memberships[..., None] * alpha_minus_one[None, :, None] * offsets,
        axis=1,
    )
    blend_multiplier = 1.0 + finger_memberships @ alpha_minus_one
    return (
        palm_template + template_delta,
        palm_pose_blends * blend_multiplier[None, :, None],
    )


@jaxls.Cost.factory(jac_mode="forward")
def morphology_matching_cost(
    values: jaxls.VarValues,
    raw_var: MorphologyRawVar,
    lower_bounds: jax.Array,
    upper_bounds: jax.Array,
    template: jax.Array,
    pose_blends: jax.Array,
    joint_regressor: jax.Array,
    blend_weights: jax.Array,
    finger_memberships: jax.Array,
    finger_mcp_indices: jax.Array,
    wrist_joint_index: jax.Array,
    mano_joint_indices: jax.Array,
    tip_vertex_indices: jax.Array,
    robot_joint_targets: jax.Array,
    robot_tip_targets: jax.Array,
    joint_weight: jax.Array,
    fingertip_weight: jax.Array,
) -> jax.Array:
    """Evaluate bounded Scaled-MANO morphology matching for one robot hand."""

    parameters = _bounded_parameters(values[raw_var], lower_bounds, upper_bounds)
    scales = jnp.exp(parameters[:6])
    scaled_template, scaled_pose_blends = _scale_mano_dense(
        template,
        pose_blends,
        joint_regressor,
        scales,
        finger_memberships,
        finger_mcp_indices,
        wrist_joint_index,
    )
    vertices, joints, _ = mano_lbs(
        scaled_template,
        scaled_pose_blends,
        joint_regressor,
        blend_weights,
        MANO_PARENT_INDICES,
        parameters[6:9],
        parameters[9:54],
        parameters[54:57],
    )
    return morphology_residuals(
        joints[mano_joint_indices],
        robot_joint_targets,
        vertices[tip_vertex_indices],
        robot_tip_targets,
        joint_weight,
        fingertip_weight,
    )


@jdc.jit
def solve_morphology_jaxls(
    initial_parameters: jax.Array,
    lower_bounds: jax.Array,
    upper_bounds: jax.Array,
    template: jax.Array,
    pose_blends: jax.Array,
    joint_regressor: jax.Array,
    blend_weights: jax.Array,
    finger_memberships: jax.Array,
    finger_mcp_indices: jax.Array,
    wrist_joint_index: jax.Array,
    mano_joint_indices: jax.Array,
    tip_vertex_indices: jax.Array,
    robot_joint_targets: jax.Array,
    robot_tip_targets: jax.Array,
    joint_weight: jax.Array,
    fingertip_weight: jax.Array,
    max_iterations: jdc.Static[int],
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Solve bounded morphology matching with GPU-compatible JAXLS LM."""

    raw_var = MorphologyRawVar(0)
    initial_raw = _raw_parameters(
        initial_parameters, lower_bounds, upper_bounds
    )
    cost = morphology_matching_cost(
        raw_var,
        lower_bounds,
        upper_bounds,
        template,
        pose_blends,
        joint_regressor,
        blend_weights,
        finger_memberships,
        finger_mcp_indices,
        wrist_joint_index,
        mano_joint_indices,
        tip_vertex_indices,
        robot_joint_targets,
        robot_tip_targets,
        joint_weight,
        fingertip_weight,
    )
    solution, summary = (
        jaxls.LeastSquaresProblem(costs=[cost], variables=[raw_var])
        .analyze()
        .solve(
            initial_vals=jaxls.VarValues.make([raw_var.with_value(initial_raw)]),
            linear_solver="dense_cholesky",
            trust_region=jaxls.TrustRegionConfig(),
            termination=jaxls.TerminationConfig(
                max_iterations=max_iterations,
                cost_tolerance=1e-8,
                gradient_tolerance=1e-6,
                parameter_tolerance=1e-8,
            ),
            verbose=False,
            return_summary=True,
        )
    )
    parameters = _bounded_parameters(solution[raw_var], lower_bounds, upper_bounds)
    return parameters, summary.iterations, jnp.any(summary.termination_criteria)


class ContactParametersVar(
    jaxls.Var[jax.Array], default_factory=lambda: jnp.zeros((51,), dtype=jnp.float32)
):
    """Euclidean MANO global, articulated-pose, and translation parameters."""


@jaxls.Cost.factory(jac_mode="forward")
def contact_matching_cost(
    values: jaxls.VarValues,
    parameters_var: ContactParametersVar,
    template: jax.Array,
    pose_blends: jax.Array,
    joint_regressor: jax.Array,
    blend_weights: jax.Array,
    contact_indices: jax.Array,
    target_positions: jax.Array,
    contact_mask: jax.Array,
    reference_rotations: jax.Array,
    table_height: jax.Array,
    table_mask: jax.Array,
    contact_weight: jax.Array,
    table_weight: jax.Array,
    pose_weight: jax.Array,
) -> jax.Array:
    """Evaluate the fixed-shape MMO contact residual for one trajectory frame."""

    parameters = values[parameters_var]
    vertices, _, rotations = mano_lbs(
        template,
        pose_blends,
        joint_regressor,
        blend_weights,
        MANO_PARENT_INDICES,
        parameters[:3],
        parameters[3:48],
        parameters[48:51],
    )
    return contact_residuals(
        vertices,
        contact_indices,
        target_positions,
        rotations,
        reference_rotations,
        table_height,
        ContactLossWeights(
            contact=contact_weight,
            table=table_weight,
            pose=pose_weight,
        ),
        include_table=True,
        contact_mask=contact_mask,
        table_mask=table_mask,
    )


@jdc.jit
def solve_contact_frame_jaxls(
    initial_parameters: jax.Array,
    template: jax.Array,
    pose_blends: jax.Array,
    joint_regressor: jax.Array,
    blend_weights: jax.Array,
    contact_indices: jax.Array,
    target_positions: jax.Array,
    contact_mask: jax.Array,
    reference_rotations: jax.Array,
    table_height: jax.Array,
    table_mask: jax.Array,
    contact_weight: jax.Array,
    table_weight: jax.Array,
    pose_weight: jax.Array,
    max_iterations: jdc.Static[int],
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Solve one contact frame entirely with JAXLS Levenberg--Marquardt."""

    parameters_var = ContactParametersVar(0)
    cost = contact_matching_cost(
        parameters_var,
        template,
        pose_blends,
        joint_regressor,
        blend_weights,
        contact_indices,
        target_positions,
        contact_mask,
        reference_rotations,
        table_height,
        table_mask,
        contact_weight,
        table_weight,
        pose_weight,
    )
    solution, summary = (
        jaxls.LeastSquaresProblem(costs=[cost], variables=[parameters_var])
        .analyze()
        .solve(
            initial_vals=jaxls.VarValues.make(
                [parameters_var.with_value(initial_parameters)]
            ),
            linear_solver="dense_cholesky",
            trust_region=jaxls.TrustRegionConfig(),
            termination=jaxls.TerminationConfig(max_iterations=max_iterations),
            verbose=False,
            return_summary=True,
        )
    )
    return (
        solution[parameters_var],
        summary.iterations,
        jnp.any(summary.termination_criteria),
    )


@jdc.jit
def solve_contact_trajectory_jaxls(
    initial_parameters: jax.Array,
    template: jax.Array,
    pose_blends: jax.Array,
    joint_regressor: jax.Array,
    blend_weights: jax.Array,
    contact_indices: jax.Array,
    target_positions: jax.Array,
    contact_masks: jax.Array,
    reference_rotations: jax.Array,
    table_heights: jax.Array,
    table_masks: jax.Array,
    contact_weight: jax.Array,
    table_weight: jax.Array,
    pose_weight: jax.Array,
    max_iterations: jdc.Static[int],
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Solve a trajectory on-device with sequential warm starts via ``lax.scan``."""

    def solve_step(
        warm_parameters: jax.Array,
        frame_inputs: tuple[jax.Array, ...],
    ) -> tuple[jax.Array, tuple[jax.Array, ...]]:
        """Solve one scan step and expose both warm and optimized parameters."""

        (
            frame_indices,
            frame_targets,
            frame_contact_mask,
            frame_reference_rotations,
            frame_table_height,
            frame_table_mask,
        ) = frame_inputs
        optimized, iterations, terminated = solve_contact_frame_jaxls(
            warm_parameters,
            template,
            pose_blends,
            joint_regressor,
            blend_weights,
            frame_indices,
            frame_targets,
            frame_contact_mask,
            frame_reference_rotations,
            frame_table_height,
            frame_table_mask,
            contact_weight,
            table_weight,
            pose_weight,
            max_iterations,
        )
        return optimized, (warm_parameters, optimized, iterations, terminated)

    _, outputs = jax.lax.scan(
        solve_step,
        initial_parameters,
        (
            contact_indices,
            target_positions,
            contact_masks,
            reference_rotations,
            table_heights,
            table_masks,
        ),
    )
    return outputs
