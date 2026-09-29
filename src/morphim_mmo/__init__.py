"""Minimal implementation of the three pre-RL MMO kinematic stages."""

from .data.contact import (
    ContactTargets,
    build_contact_targets,
    build_rigid_contact_targets,
    detect_contact_indices,
)
from .models.scaled_mano import (
    FingerSpec,
    ScaledManoArrays,
    initial_scale_ratios,
    scale_mano_arrays,
)
from .optimization.losses import ContactLossWeights, contact_residuals, morphology_residuals

__all__ = [
    "ContactLossWeights",
    "ContactTargets",
    "FingerSpec",
    "ScaledManoArrays",
    "build_contact_targets",
    "build_rigid_contact_targets",
    "contact_residuals",
    "detect_contact_indices",
    "initial_scale_ratios",
    "morphology_residuals",
    "scale_mano_arrays",
]
