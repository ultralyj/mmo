"""Train a Morphometric Imitation residual-RL teacher policy."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from ...rl.config import load_residual_rl_config
from ...rl.environment import MorphometricResidualEnv, ResidualActionWrapper
from ...rl.ppo import PPOTrainer
from ...rl.reference import ResidualRLReference
from ...rl.visualization import render_policy_gif


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
    parser.add_argument(
        "--initial-log-std",
        type=float,
        help="override the initial Gaussian policy log standard deviation",
    )
    parser.add_argument("--checkpoint", type=Path, help="warm-start policy weights for the next curriculum stage")
    parser.add_argument("--sim-backend", default="auto")
    parser.add_argument("--checkpoint-interval", type=int, default=10)
    parser.add_argument(
        "--gif-interval",
        type=int,
        default=5,
        help="render one diagnostic policy GIF every N PPO iterations; 0 disables",
    )
    parser.add_argument("--gif-frame-stride", type=int, default=4)
    parser.add_argument("--gif-fps", type=float, default=15.0)
    args = parser.parse_args()

    config = load_residual_rl_config(args.config)
    overrides = {}
    if args.num_envs is not None:
        overrides["num_envs"] = args.num_envs
    if args.rollout_steps is not None:
        overrides["rollout_steps"] = args.rollout_steps
    if args.total_timesteps is not None:
        overrides["total_timesteps"] = args.total_timesteps
    if args.initial_log_std is not None:
        overrides["initial_log_std"] = args.initial_log_std
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
        reference_end_step=config.control.reference_end_step,
        hold_steps=config.control.hold_steps,
    )
    environment = MorphometricResidualEnv(
        reference=reference,
        config=config,
        num_envs=config.ppo.num_envs,
        sim_backend=args.sim_backend,
    )
    environment = ResidualActionWrapper(environment)
    render_environment = None
    visualization_callback = None
    if args.gif_interval > 0:
        render_environment = ResidualActionWrapper(
            MorphometricResidualEnv(
                reference=reference,
                config=config,
                num_envs=1,
                # Keep rendering isolated from the batched GPU PhysX scene.
                sim_backend="cpu",
                render_mode="rgb_array",
            )
        )

        def visualization_callback(policy, iteration, output_path):
            """Render the current deterministic policy in an isolated scene."""

            render_policy_gif(
                policy,
                render_environment,
                output_path,
                fps=args.gif_fps,
                frame_stride=args.gif_frame_stride,
                seed=config.ppo.seed,
            )

    trainer = PPOTrainer(environment, config.ppo)
    if args.checkpoint is not None:
        import torch
        checkpoint = torch.load(args.checkpoint, map_location=trainer.device, weights_only=False)
        trainer.policy.load_state_dict(checkpoint["policy"])
    try:
        trainer.train(
            output_dir=args.output_dir,
            checkpoint_interval=args.checkpoint_interval,
            visualization_interval=args.gif_interval,
            visualization_callback=visualization_callback,
        )
    finally:
        environment.close()
        if render_environment is not None:
            render_environment.close()

if __name__ == "__main__":
    main()
