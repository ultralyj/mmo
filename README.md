# MorphIM minimal MMO pipeline

This directory implements the three kinematic stages of Morphometric
Optimization used before reinforcement learning:

1. morphology matching with the six-scale Scaled MANO model;
2. per-frame contact matching with contact, table, and pose losses;
3. heat-diffused robot targets and sequential robot pose recovery with IK.

Reinforcement learning is intentionally out of scope for this milestone.

## Code layout

The package is organized by responsibility; command workflows contain only
argument parsing and stage orchestration:

```text
src/morphim_mmo/
├── data/             # GRAB loading and contact-target construction
├── models/           # MANO, Scaled MANO, URDF, and skeleton skinning
├── optimization/     # alignment, losses, metrics, and solver adapters
├── visualization/    # shared headless renderer and URDF mesh loading
├── utils/            # YAML/JSON/NPZ I/O and frame selection
└── cli/
    ├── data/         # dataset export commands
    ├── diagnostics/  # input/robot inspection commands
    ├── matching/     # three MMO matching stages
    └── rendering/    # artifact rendering commands
```

Stable command names are declared in `pyproject.toml`, so callers do not need
to depend on internal module paths.

## Current status

- The official right-hand Sharpa Wave URDF and meshes are pinned under
  `assets/sharpa-urdf-usd-xml` at commit
  `0d19cac602f46456b819e4b6a2c09a74982c9a3e`.
- The 22-DoF neutral robot skeleton and one-to-one five-finger semantic mapping
  are defined in `configs/sharpa_wave_right.yaml`.
- Scaled MANO equations 5--6, initial scale ratios, equation 7 residuals,
  contact detection/fill, equations 8--10 (with the Sharpa-inapplicable coupled
  finger term omitted), and JAX-autodifferentiated LM are implemented.
- Real Sharpa morphology matching is executable with bounded nonlinear least
  squares and an exact JAX Jacobian. The bounds prevent periodic axis-angle and
  scale-collapse degeneracies observed with unconstrained LM. A local
  +/-0.35-rad hand-pose trust region also removes axial-twist solutions that
  match sparse keypoints but visibly distort the MANO surface, while preserving
  the paper's equation 7 alignment objective.
- MANO v1.2 and GRAB subject `s1` are validated and extracted. The real
  `apple_lift` sequence has 2,602 frames at 120 Hz and uses the right hand.
- GRAB's subject meshes, object meshes/settings, subject settings, and SMPL-X
  correspondence data are validated and extracted. The subject-specific `s1`
  right-hand template and apple/table geometry are usable; the full
  `apple_lift` right-hand trajectory and 5 mm reference contacts can be
  exported without expanding a copy of the dense object mesh per frame.
- Contact matching now runs sequentially over the complete GRAB trajectory with
  fixed morphology scales, max-contact index fill, object-contact, table, and
  SO(3) pose terms. Fixed-size padded contacts allow one JAX compilation for
  all varying contact counts.
- Loss weights are reproduction values because the paper does not publish its
  values. For Wuji Hand2, a max-contact-frame sweep selected pose weight
  `1e-3`; contact weight is `1.0`, and a penetration sweep selected table
  weight `100.0`.
- Robot target generation implements the paper's one-dimensional skeleton heat
  diffusion and linear blend skinning construction. The complete 2,602-frame
  target trajectory passes identity and rotation-consistency checks.
- Robot pose recovery uses a floating 6-DoF wrist and Wuji Hand2's 20 actuated
  joints, solved consecutively with PyRoki warm starts, joint limits, temporal
  regularization, capsule self-collision, and sampled hand/table collision.
  The balanced full run reaches 4.96 mm mean target RMSE and 20.99 degrees mean
  orientation error, with 0.55 mm maximum sampled table penetration.

## Environment

```bash
uv sync --python 3.11
uv run pytest
uv run morphim-inspect-robot
uv run morphim-inspect-inputs
uv run morphim-export-grab-hand
uv run morphim-export-grab-contacts
JAX_PLATFORMS=cpu uv run morphim-init-morphology
JAX_PLATFORMS=cpu uv run morphim-match-morphology
uv run morphim-render-morphology
PYOPENGL_PLATFORM=egl uv run morphim-render-paper-style
```

The environment uses PyTorch CUDA 12.8 wheels and JAX CUDA 13 wheels. PyRoki is
pinned to commit `388e43e1fc0d0ee382968d3dd72970fd62a0450c` for reproducible
IK behavior. All are installed by `uv sync`.

## Required licensed inputs

Validated MANO v1.2 models are under `../data/models/mano/`, and GRAB subject
`s1` is under `../data/raw/grab/s1/`, and the available licensed GRAB tools are
under `../data/raw/grab/tools/`. Do not commit or redistribute those files.

The MANO pickle loader, legacy Chumpy compatibility layer, subject-specific
GRAB hand decoder, rigid object reconstruction, reference contact extractor,
and three-stage kinematic optimization are implemented and validated.

## Wuji Hand 2 Beta 2

The same morphology pipeline also supports the right, no-mount Wuji Hand 2
Beta 2 model pinned under `assets/wuji-description` at revision
`c2cd7f8d1ef8b6dc8cb907c17daa5a88b4442d95`:

```bash
JAX_PLATFORMS=cpu uv run morphim-match-morphology \
  --config configs/wuji_hand2_beta2_right.yaml \
  --output-json outputs/wuji_hand2/morphology_match.json \
  --output-geometry outputs/wuji_hand2/morphology_match.npz
PYOPENGL_PLATFORM=egl uv run morphim-render-paper-style \
  --config configs/wuji_hand2_beta2_right.yaml \
  --geometry outputs/wuji_hand2/morphology_match.npz \
  --output outputs/wuji_hand2/morphology_paper_style.png \
  --aligned-output outputs/wuji_hand2/morphology_aligned_mano_smooth.png

# Paper-exact consecutive warm start over all 2,602 apple_lift frames.
uv run morphim-match-contacts \
  --frames all \
  --output outputs/wuji_hand2/contact_match_full.npz \
  --output-json outputs/wuji_hand2/contact_match_full.json

PYOPENGL_PLATFORM=egl uv run morphim-render-contacts \
  --result outputs/wuji_hand2/contact_match_full.npz \
  --frame 1732 \
  --output outputs/wuji_hand2/contact_matching.png

# Paper equations 11--12: heat diffusion skinning and robot pose targets.
uv run morphim-generate-robot-targets \
  --contact-match outputs/wuji_hand2/contact_match_full.npz \
  --output outputs/wuji_hand2/robot_targets.npz \
  --output-json outputs/wuji_hand2/robot_targets.json

# Consecutive warm-started floating-base IK over the complete trajectory.
uv run morphim-solve-robot-ik \
  --targets outputs/wuji_hand2/robot_targets.npz \
  --frames all \
  --output outputs/wuji_hand2/robot_pose_full.npz \
  --output-json outputs/wuji_hand2/robot_pose_full.json

PYOPENGL_PLATFORM=egl uv run morphim-render-robot-pose \
  --robot-pose outputs/wuji_hand2/robot_pose_full.npz \
  --frame 1732 \
  --output outputs/wuji_hand2/robot_pose_recovery.png

# Object-stabilized contact animation around the maximum-contact frame.
PYOPENGL_PLATFORM=egl uv run morphim-render-contact-gif \
  --frames 1450:1951:8 \
  --fps 15 \
  --output outputs/wuji_hand2/contact_recovery.gif
```

The paper does not publish its heat-diffusion or IK residual weights. Values in
`configs/wuji_hand2_beta2_right.yaml` are clearly marked reproduction values.
The Wuji asset contains a hand but no arm, so pose recovery treats the wrist as
a floating base; this is the direct hand-only analogue of the paper's arm-hand
IK stage.
