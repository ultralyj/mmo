"""Render a ManiSkill GIF of the MMO kinematic hand-object reference."""

from __future__ import annotations

import argparse
from pathlib import Path

from ...rl.config import load_residual_rl_config
from ...rl.environment import MorphometricResidualEnv, ResidualActionWrapper
from ...rl.reference import ResidualRLReference
from ...rl.visualization import render_kinematic_reference_gif


def main() -> None:
    """Load the apple reference and render it without physics or a policy."""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=Path("configs/rl/wuji_hand2_apple.yaml")
    )
    parser.add_argument(
        "--robot-pose",
        type=Path,
        default=Path("outputs/wuji_hand2/robot_pose_full_safe.npz"),
    )
    parser.add_argument(
        "--contacts",
        type=Path,
        default=Path("../data/processed/s1/apple_lift_contacts.npz"),
    )
    parser.add_argument(
        "--robot-config",
        type=Path,
        default=Path("configs/wuji_hand2_beta2_right.yaml"),
    )
    parser.add_argument(
        "--mano-dir",
        type=Path,
        default=Path("../data/models/mano/mano_v1_2/models"),
    )
    parser.add_argument(
        "--object-mesh",
        type=Path,
        default=Path(
            "../data/raw/grab/tools/object_meshes/contact_meshes/apple.ply"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/rl/diagnostics/kinematic-reference.gif"),
    )
    parser.add_argument("--frame-stride", type=int, default=2)
    parser.add_argument("--fps", type=float, default=15.0)
    args = parser.parse_args()

    config = load_residual_rl_config(args.config)
    reference = ResidualRLReference.load(
        robot_pose_path=args.robot_pose,
        contacts_path=args.contacts,
        robot_config_path=args.robot_config,
        mano_dir=args.mano_dir,
        object_mesh_path=args.object_mesh,
        reference_stride=config.control.reference_stride,
    )
    environment = ResidualActionWrapper(
        MorphometricResidualEnv(
            reference=reference,
            config=config,
            num_envs=1,
            sim_backend="cpu",
            render_mode="rgb_array",
        )
    )
    try:
        render_kinematic_reference_gif(
            environment,
            args.output,
            fps=args.fps,
            frame_stride=args.frame_stride,
        )
    finally:
        environment.close()
    print(f"Saved kinematic reference GIF: {args.output}")


if __name__ == "__main__":
    main()
