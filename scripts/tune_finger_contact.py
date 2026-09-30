"""Search physical lift corrections while preserving reference finger identity."""
import argparse
import json
from dataclasses import replace
from pathlib import Path

import torch
from diagnose_lift import load_reference
from morphim_mmo.rl.config import load_residual_rl_config
from morphim_mmo.rl.environment import MorphometricResidualEnv, ResidualActionWrapper, _grasp_bias_activation

parser = argparse.ArgumentParser()
parser.add_argument('--config', default='configs/rl/wuji_hand2_apple.yaml')
parser.add_argument('--rounds', type=int, default=35)
parser.add_argument('--num-envs', type=int, default=512)
parser.add_argument('--output', default='outputs/rl/diagnostics/cem_fingers.json')
args = parser.parse_args()
config = load_residual_rl_config(args.config)
config = replace(config, reward=replace(config.reward, contact_mode='per_finger', require_reference_contact=True))
reference = load_reference(config)
env = ResidualActionWrapper(MorphometricResidualEnv(reference=reference, config=config,
                         num_envs=args.num_envs, sim_backend='gpu'))
base = env.unwrapped
base.reset(seed=1)
gate = _grasp_bias_activation(base._reference_contact, config.control.grasp_contact_lookahead_steps,
                             config.control.grasp_min_contact_fingers, config.control.grasp_release_decay)
mean = torch.zeros(23, device=base.device)
std = torch.full_like(mean, .25); std[20:] = .012
best = mean.clone(); best_score = -float('inf'); history = []
original_base_pose = base._base_pose_for_step
candidates = None

def corrected_pose(step, env_idx=None):
    pose = original_base_pose(step, env_idx).clone()
    ids = base._env_ids if env_idx is None else env_idx
    if candidates is not None:
        pose[:, :3] += gate[step, None] * candidates[ids, 20:]
    return pose

base._base_pose_for_step = corrected_pose
try:
    with torch.no_grad():
        for iteration in range(args.rounds):
            torch.manual_seed(300 + iteration)
            candidates = mean + std * torch.randn(args.num_envs, 23, device=base.device)
            candidates[:, :20].clamp_(-.7, .7); candidates[:, 20:].clamp_(-.04, .04)
            candidates[0] = best; candidates[1] = 0
            env.reset(seed=1)
            score = torch.zeros(args.num_envs, device=base.device)
            failed = torch.zeros_like(score, dtype=torch.bool)
            coverage = torch.zeros((args.num_envs, 5), device=base.device)
            initial_z = base.object.pose.p[:, 2].clone()
            peak = torch.zeros_like(score)
            max_force = torch.zeros_like(score)
            for step in range(reference.frame_count):
                q = (base.current_reference_qpos + gate[step] * candidates[:, :20]).clamp(base._joint_lower, base._joint_upper)
                _, _, _, _, info = base.step(q)
                failed |= info['fail']
                peak = torch.maximum(peak, info['object_lift'])
                max_force = torch.maximum(max_force, info['maximum_object_contact_force'])
                observed = info['observed_contact']; target = base._reference_contact[step]
                score -= (60 if step >= 60 else 2) * info['add_error']
                if step >= 60:
                    score += 2 * (observed & target).float().sum(-1) - (observed & ~target).float().sum(-1)
                if step >= 77:
                    coverage += observed.float()
                    score -= 100 * (.04 - info['object_lift']).clamp_min(0)
                score -= 5 * (info['terminate_table_height'] | info['terminate_table_force']).float()
                score -= .02 * (info['maximum_object_contact_force'] - 100).clamp_min(0)
            score -= 30 * failed.float()
            elite = score.topk(max(8, args.num_envs // 10)).indices
            mean = .2 * mean + .8 * candidates[elite].mean(0)
            std = .2 * std + .8 * candidates[elite].std(0)
            std[:20].clamp_min_(.025); std[20:].clamp_min_(.001)
            index = int(score.argmax())
            if float(score[index]) > best_score:
                best_score = float(score[index]); best = candidates[index].clone()
            row = dict(round=iteration, score=float(score[index]), best_score=best_score,
                       peak_lift_m=float(peak[index]), final_lift_m=float(info['object_lift'][index]),
                       final_add_m=float(info['add_error'][index]), max_force_n=float(max_force[index]),
                       failed=bool(failed[index]), hold_contact_fraction=(coverage[index] / 30).tolist(),
                       candidate_delta=candidates[index].tolist(), best_delta=best.tolist())
            history.append(row); print(json.dumps(row), flush=True)
            path = Path(args.output); path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(history, indent=2))
finally:
    env.close()
