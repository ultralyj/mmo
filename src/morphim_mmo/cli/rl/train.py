"""Train a Morphometric Imitation residual-RL teacher policy."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from ...rl.config import load_residual_rl_config
from ...rl.environment import MorphometricResidualEnv, ResidualActionWrapper
from ...rl.ppo import PPOTrainer
from ...rl.reference import ResidualRLReference


def main() -> None:
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
        "--mano-dir", type=Path, default=Path("../data/models/mano/mano_v1_2/models")
    )
    parser.add_argument(
        "--object-mesh",
        type=Path,
        default=Path(
            "../data/raw/grab/tools/object_meshes/contact_meshes/apple.ply"
        ),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/rl/wuji_hand2_apple")
    )
    parser.add_argument("--num-envs", type=int)
    parser.add_argument("--rollout-steps", type=int)
    parser.add_argument("--total-timesteps", type=int)
    parser.add_argument("--sim-backend", default="auto")
    parser.add_argument("--checkpoint-interval", type=int, default=10)
    args = parser.parse_args()

    config = load_residual_rl_config(args.config)
    overrides = {}
    if args.num_envs is not None:
        overrides["num_envs"] = args.num_envs
    if args.rollout_steps is not None:
        overrides["rollout_steps"] = args.rollout_steps
    if args.total_timesteps is not None:
        overrides["total_timesteps"] = args.total_timesteps
    if overrides:
        config = replace(
            config, ppo=replace(config.ppo, **overrides)
        )

    reference = ResidualRLReference.load(
        robot_pose_path=args.robot_pose,
        contacts_path=args.contacts,
        robot_config_path=args.robot_config,
        mano_dir=args.mano_dir,
        object_mesh_path=args.object_mesh,
        reference_stride=config.control.reference_stride,
    )
    environment = MorphometricResidualEnv(
        reference=reference,
        config=config,
        num_envs=config.ppo.num_envs,
        sim_backend=args.sim_backend,
    )
    environment = ResidualActionWrapper(environment)
    trainer = PPOTrainer(environment, config.ppo)
    trainer.train(
        output_dir=args.output_dir,
        checkpoint_interval=args.checkpoint_interval,
    )
    environment.close()

if __name__ == "__main__":
    main()