# MorphIM minimal MMO pipeline

This directory implements the three kinematic stages of Morphometric
Optimization used before reinforcement learning:

1. morphology matching with the six-scale Scaled MANO model;
2. per-frame contact matching with contact, table, and pose losses;
3. heat-diffused robot targets and sequential robot pose recovery with IK.

Residual reinforcement learning now supports a validated Wuji Hand2 apple
lift-and-hold task and a subsequent domain-randomization stage. See
[the repair log](docs/rl_repair_log.md) for measured results, reproduction
commands, checkpoints, and the limits of the hand-only setup.

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
  contact detection/fill, and equations 8--10 (with the Sharpa-inapplicable
  coupled-finger term omitted) are implemented in JAX.
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
- Morphology and contact matching use JAXLS Levenberg--Marquardt by default.
  Contact matching keeps the complete sequential warm-start trajectory on the
  accelerator with `jax.lax.scan`; fixed-size padded contacts avoid recompiling
  for different contact counts. The original SciPy implementations remain
  available through `--backend scipy` for regression comparisons.
- Loss weights are reproduction values because the paper does not publish its
  values. For Wuji Hand2, a max-contact-frame sweep selected pose weight
  `1e-3`; contact weight is `1.0`, and a penetration sweep selected table
  weight `100.0`.
- Robot target generation implements the paper's one-dimensional skeleton heat
  diffusion and linear blend skinning construction. The complete 2,602-frame
  target trajectory passes identity and rotation-consistency checks.
- Robot pose recovery uses a floating 6-DoF wrist and Wuji Hand2's 20 actuated
  joints, solved consecutively with PyRoki/JAXLS warm starts, joint limits,
  temporal regularization, capsule self-collision, and sampled hand/table
  collision. Its default `scan` execution keeps the trajectory and batched
  metrics on the accelerator.
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
uv run morphim-init-morphology
uv run morphim-match-morphology
uv run morphim-render-morphology
PYOPENGL_PLATFORM=egl uv run morphim-render-paper-style
```

The environment uses PyTorch CUDA 12.8 wheels and JAX CUDA 13 wheels. JAXLS is
pinned to commit `50a58be88c5ef74532f09e3f55268b4f02c490e3`, and PyRoki is pinned
to `388e43e1fc0d0ee382968d3dd72970fd62a0450c`. All are installed by
`uv sync`. Set `JAX_PLATFORMS=cpu` explicitly when a CUDA device is unavailable.

## Accelerator backends and measured throughput

The JAXLS backend follows the paper's implementation choice and is the default.
The morphology variables use a smooth `tanh` reparameterization so the finite
bounds from the stable SciPy baseline are preserved. Contact frames are still
solved in chronological order, but the warm-start loop runs as a compiled GPU
scan rather than crossing the Python/SciPy boundary for every LM step.

On the current server, including first-call compilation and output metric
calculation, a 100-frame contact window ran at 35.44 FPS with JAXLS versus
28.22 FPS with SciPy. The complete 2,602-frame JAXLS run took 10.81 seconds
(240.66 FPS), with 3.39 mm mean contact RMSE. The paper's reported MMO speed
excludes robot IK; pose recovery is therefore reported separately. A 500-frame
IK window ran at 45.37 FPS with the GPU scan versus 44.45 FPS with per-frame
host dispatch, also including compilation.

These paths currently use one JAX device. `jax_devices` in each JSON result
records every visible device, but the implementation does not shard one
trajectory across multiple GPUs because consecutive warm starts are dependent.

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
uv run morphim-match-morphology \
  --backend jaxls \
  --config configs/wuji_hand2_beta2_right.yaml \
  --output-json outputs/wuji_hand2/morphology_match_jaxls.json \
  --output-geometry outputs/wuji_hand2/morphology_match_jaxls.npz
PYOPENGL_PLATFORM=egl uv run morphim-render-paper-style \
  --config configs/wuji_hand2_beta2_right.yaml \
  --geometry outputs/wuji_hand2/morphology_match_jaxls.npz \
  --output outputs/wuji_hand2/morphology_paper_style.png \
  --aligned-output outputs/wuji_hand2/morphology_aligned_mano_smooth.png

# Paper-exact consecutive warm start over all 2,602 apple_lift frames.
uv run morphim-match-contacts \
  --backend jaxls \
  --morphology-json outputs/wuji_hand2/morphology_match_jaxls.json \
  --frames all \
  --output outputs/wuji_hand2/contact_match_jaxls_full.npz \
  --output-json outputs/wuji_hand2/contact_match_jaxls_full.json

PYOPENGL_PLATFORM=egl uv run morphim-render-contacts \
  --result outputs/wuji_hand2/contact_match_jaxls_full.npz \
  --frame 1732 \
  --output outputs/wuji_hand2/contact_matching.png

# Paper equations 11--12: heat diffusion skinning and robot pose targets.
uv run morphim-generate-robot-targets \
  --morphology-json outputs/wuji_hand2/morphology_match_jaxls.json \
  --contact-match outputs/wuji_hand2/contact_match_jaxls_full.npz \
  --output outputs/wuji_hand2/robot_targets_jaxls.npz \
  --output-json outputs/wuji_hand2/robot_targets_jaxls.json

# Consecutive warm-started floating-base IK over the complete trajectory.
uv run morphim-solve-robot-ik \
  --execution scan \
  --targets outputs/wuji_hand2/robot_targets_jaxls.npz \
  --contact-match outputs/wuji_hand2/contact_match_jaxls_full.npz \
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
