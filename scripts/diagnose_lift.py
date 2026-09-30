"""Evaluate complete physical episodes without moving the object after reset."""
import argparse
import json
from dataclasses import replace
from pathlib import Path

import torch

from morphim_mmo.rl.config import load_residual_rl_config
from morphim_mmo.rl.environment import MorphometricResidualEnv, ResidualActionWrapper
from morphim_mmo.rl.ppo import ActorCritic, _single_env_action_limit
from morphim_mmo.rl.reference import ResidualRLReference


def load_reference(config):
    return ResidualRLReference.load(
        robot_pose_path="outputs/wuji_hand2/robot_pose_full_safe.npz",
        contacts_path="../data/processed/s1/apple_lift_contacts.npz",
        robot_config_path="configs/wuji_hand2_beta2_right.yaml",
        mano_dir="../data/models/mano/mano_v1_2/models",
        object_mesh_path="../data/raw/grab/tools/object_meshes/contact_meshes/apple.ply",
        reference_stride=config.control.reference_stride,
        reference_end_step=config.control.reference_end_step,
        hold_steps=config.control.hold_steps,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/rl/wuji_hand2_apple_physical_grasp.yaml")
    parser.add_argument("--output", default="outputs/rl/diagnostics/baseline.json")
    parser.add_argument("--checkpoint")
    parser.add_argument("--num-envs", type=int, default=128)
    parser.add_argument("--seed", type=int, default=101)
    args = parser.parse_args()
    config = load_residual_rl_config(args.config)
    config = replace(config, ppo=replace(config.ppo, seed=args.seed))
    reference = load_reference(config)
    env = ResidualActionWrapper(MorphometricResidualEnv(
        reference=reference, config=config, num_envs=args.num_envs, sim_backend="gpu"))
    try:
        observation, _ = env.reset(seed=args.seed)
        base = env.unwrapped
        policy = None
        if args.checkpoint:
            checkpoint = torch.load(args.checkpoint, map_location=base.device, weights_only=False)
            policy = ActorCritic(
                observation.shape[-1], 20, tuple(checkpoint["config"]["hidden_sizes"]),
                _single_env_action_limit(env.action_space.high, 20, base.device)).to(base.device)
            policy.load_state_dict(checkpoint["policy"])
            policy.eval()
        initial_height = base.object.pose.p[:, 2].clone()
        peak = initial_height.clone()
        failed = torch.zeros(args.num_envs, dtype=torch.bool, device=base.device)
        max_force = torch.zeros(args.num_envs, device=base.device)
        min_clearance = torch.full_like(max_force, float("inf"))
        max_self_force = torch.zeros_like(max_force)
        hold_contact = torch.zeros((args.num_envs, 5), device=base.device)
        hold_frames = 0
        rows = []
        with torch.no_grad():
            for step in range(reference.frame_count):
                action = torch.zeros((args.num_envs, 20), device=base.device)
                if policy is not None:
                    action = policy.action_limit * torch.tanh(policy.forward(observation)[0].mean)
                observation, reward, terminated, truncated, info = env.step(action)
                failed |= info["fail"]
                peak = torch.maximum(peak, base.object.pose.p[:, 2])
                max_force = torch.maximum(max_force, info["maximum_object_contact_force"])
                min_clearance = torch.minimum(min_clearance, info["table_clearance"])
                if config.control.hold_steps and step >= reference.frame_count - config.control.hold_steps:
                    hold_contact += info["observed_contact"].float()
                    hold_frames += 1
                max_self_force = torch.maximum(max_self_force, info["maximum_self_contact_force"])
                row = {
                    "step": step,
                    "height": float(base.object.pose.p[:, 2].mean()),
                    "reference_height": float(base._object_pose_for_step(base._active_step)[:, 2].mean()),
                    "add": float(info["add_error"].mean()),
                    "fingers": float(info["observed_contact"].float().sum(-1).mean()),
                    "force": float(info["maximum_object_contact_force"].max()),
                    "observed_contact": info["observed_contact"].float().mean(0).tolist(),
                    "reference_contact": base._reference_contact[step].tolist(),
                    "finger_force_n": info["finger_contact_force"].mean(0).tolist(),
                    "failed": int(failed.sum()),
                    "self_force_n": float(info["maximum_self_contact_force"].max()),
                    "termination_reasons": {k:int(v.sum()) for k,v in info.items() if k.startswith("terminate_")},
                    "clearance": float(info["table_clearance"].min()),
                }
                rows.append(row)
                if step % 25 == 0 or step == reference.frame_count - 1:
                    print(row, flush=True)
        success = info["success"] & ~failed
        final_lift = base.object.pose.p[:, 2] - initial_height
        # In addition to the paper criterion, insist on sustained physical lift.
        lift_success = success & (final_lift >= 0.04)
        hold_fraction = hold_contact / max(hold_frames, 1)
        final_target = base._reference_contact[-1]
        finger_match_success = lift_success & torch.where(final_target, hold_fraction >= .9, hold_fraction <= .1).all(-1)
        result = {
            "seed": args.seed, "checkpoint": args.checkpoint,
            "config": args.config, "frames": reference.frame_count,
            "initial_height_m": float(initial_height.mean()),
            "peak_lift_m": (peak - initial_height).tolist(),
            "final_lift_m": final_lift.tolist(),
            "final_height_m": base.object.pose.p[:, 2].tolist(),
            "final_add_m": info["add_error"].tolist(),
            "max_contact_force_n": max_force.tolist(),
            "max_self_contact_force_n": max_self_force.tolist(),
            "minimum_table_clearance_m": min_clearance.tolist(),
            "success": success.tolist(), "lift_success": lift_success.tolist(),
            "finger_order": ["thumb", "index", "middle", "ring", "pinky"],
            "hold_contact_fraction": hold_fraction.tolist(),
            "finger_match_success": finger_match_success.tolist(),
            "rows": rows,
        }
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2))
        print(f"Success: {int(success.sum())}/{args.num_envs}; sustained >=4cm: {int(lift_success.sum())}/{args.num_envs}; hold finger match: {int(finger_match_success.sum())}/{args.num_envs}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
