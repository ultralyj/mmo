# Input contract for the first real trajectory

All coordinates use metres and all rotations use axis-angle radians unless a
field explicitly states otherwise.

## MANO model

Expected source: a complete licensed MANO v1.2 right-hand pickle. Before use,
the loader must validate:

- 778 template vertices;
- 16 regressed/skinning joints including the wrist;
- template, shape directions, pose directions, joint regressor, blend weights,
  kinematic tree, and mesh faces are present;
- the standard joint and fingertip ordering recorded in
  `configs/sharpa_wave_right.yaml` matches the model.

The hand shape is fixed to `beta = 0` for morphology matching, as in the paper.

## GRAB sequence package

The first trajectory loader will normalize one GRAB sequence into an `.npz`
containing:

| Field | Shape | Meaning |
| --- | --- | --- |
| `frame_index` | `(T,)` | original GRAB frame index |
| `timestamp_s` | `(T,)` | monotonically increasing time |
| `mano_global_orient` | `(T, 3)` | wrist global axis-angle |
| `mano_hand_pose` | `(T, 45)` | 15 local joint axis-angles |
| `mano_translation_m` | `(T, 3)` | world translation |
| `object_vertices_local_m` | `(O, 3)` | immutable object mesh |
| `object_faces` | `(F, 3)` | triangle indices |
| `object_to_world` | `(T, 4, 4)` | object poses |
| `table_height_m` | scalar or `(T,)` | world-frame table plane height |

Derived reference MANO vertices, contact indices, and contact target positions
will be written to a separate processed artifact, never back into the raw data.

## Output of the two-stage milestone

1. One morphology result with six positive scales, global transform, MANO hand
   pose, optimizer diagnostics, source revisions, and config hash.
2. Per-frame contact-matched global orientation, translation, local hand pose,
   contact indices, each loss component, convergence state, and previous-frame
   warm-start provenance.
3. A geometry-only replay containing reference and optimized meshes. No robot
   IK joint trajectory is part of this milestone.
