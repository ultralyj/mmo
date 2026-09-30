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
parser.add_argument('--config', default='configs/rl/wuji_hand2_apple_physical_tuned.yaml')
parser.add_argument('--rounds', type=int, default=30)
parser.add_argument('--num-envs', type=int, default=256)
parser.add_argument('--output', default='outputs/rl/diagnostics/cem_physical_tuned.json')
args = parser.parse_args()
config = load_residual_rl_config(args.config)
config = replace(config, reward=replace(config.reward, contact_mode='per_finger', require_reference_contact=True))
reference = load_reference(config)
env = ResidualActionWrapper(MorphometricResidualEnv(reference=reference, config=config,
                         num_envs=args.num_envs, sim_backend='gpu'))
base = env.unwrapped
base.reset(seed=1)
gate = _grasp_bias_activation(base._reference_contact, config.control.grasp_contact_lookahead_steps,
                             config.control.grasp_min_contact_fingers, config.control.grasp_release_decay, config.control.grasp_ramp_steps)
names = [j.name for j in base.agent.robot.active_joints]
pinky_indices = [i for i,n in enumerate(names) if "pinky" in n]
initial_bias = torch.tensor([config.control.grasp_bias_by_joint[n] for n in names],device=base.device)
mean = torch.zeros(23, device=base.device)
mean[:20] = -.5 * initial_bias
std = torch.full_like(mean, .3); std[20:] = .012
std[pinky_indices] = 0
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
tip_links = [next(link for link in base._robot_links if link.name == name) for name in ['r_thumb_tip','r_index_finger_tip','r_middle_finger_tip','r_ring_finger_tip','r_pinky_tip']]
from morphim_mmo.rl.reward import transform_object_points
try:
    with torch.no_grad():
        for iteration in range(args.rounds):
            torch.manual_seed(300 + iteration)
            candidates = mean + std * torch.randn(args.num_envs, 23, device=base.device)
            candidates[:, :20].clamp_(-.7, .7); candidates[:, 20:].clamp_(-.04, .04)
            candidates[0] = best; candidates[1, :20] = -initial_bias
            candidates[:, pinky_indices] = 0
            env.reset(seed=1)
            score = torch.zeros(args.num_envs, device=base.device)
            failed = torch.zeros_like(score, dtype=torch.bool)
            coverage = torch.zeros((args.num_envs, 5), device=base.device)
            initial_z = base.object.pose.p[:, 2].clone()
            peak = torch.zeros_like(score)
            max_force = torch.zeros_like(score)
            max_self_force = torch.zeros_like(score)
            first_fail = torch.full((args.num_envs,), -1, dtype=torch.int64, device=base.device)
            reason_totals = {}
            for step in range(reference.frame_count):
                q = (base.current_reference_qpos + gate[step] * candidates[:, :20]).clamp(base._joint_lower, base._joint_upper)
                _, _, _, _, info = base.step(q)
                first_fail = torch.where((first_fail < 0) & info['fail'], step, first_fail)
                failed |= info['fail']
                max_self_force = torch.maximum(max_self_force, info['maximum_self_contact_force'])
                for reason,value in info.items():
                    if reason.startswith('terminate_'):
                        if reason not in reason_totals: reason_totals[reason] = torch.zeros_like(score)
                        reason_totals[reason] += value.float()
                peak = torch.maximum(peak, info['object_lift'])
                max_force = torch.maximum(max_force, info['maximum_object_contact_force'])
                observed = info['observed_contact']; target = base._reference_contact[step]
                score -= (60 if step >= 60 else 2) * info['add_error']
                if step >= 60:
                    score += 2 * (observed & target).float().sum(-1) - (observed & ~target).float().sum(-1)
                    tips = torch.stack([link.pose.p for link in tip_links], dim=1)
                    object_pose = base.object.pose.raw_pose
                    points = transform_object_points(base._object_points[::4], object_pose[:, :3], object_pose[:, 3:])
                    distance = torch.linalg.vector_norm(tips[:, :, None] - points[:, None], dim=-1).amin(-1)
                    score -= 50 * (distance * target).sum(-1)
                if step >= 77:
                    coverage += observed.float()
                    score -= 100 * (.04 - info['object_lift']).clamp_min(0)
                score -= 5 * (info['terminate_table_height'] | info['terminate_table_force']).float()
                score -= 2 * info['maximum_self_contact_force']
                score -= .02 * (info['maximum_object_contact_force'] - 100).clamp_min(0)
            score -= 500 * failed.float()
            elite = score.topk(max(8, args.num_envs // 10)).indices
            mean = .2 * mean + .8 * candidates[elite].mean(0)
            std = .2 * std + .8 * candidates[elite].std(0)
            std[:20].clamp_min_(.025); std[20:].clamp_min_(.001)
            std[pinky_indices] = 0
            index = int(score.argmax())
            if float(score[index]) > best_score:
                best_score = float(score[index]); best = candidates[index].clone()
            row = dict(round=iteration, score=float(score[index]), best_score=best_score,
                       peak_lift_m=float(peak[index]), final_lift_m=float(info['object_lift'][index]),
                       final_add_m=float(info['add_error'][index]), max_force_n=float(max_force[index]),
                       failed=bool(failed[index]), first_failure_step=int(first_fail[index]), max_self_force_n=float(max_self_force[index]),
                       failure_reasons={k:int(v[index]) for k,v in reason_totals.items()}, hold_contact_fraction=(coverage[index] / 30).tolist(),
                       candidate_delta=candidates[index].tolist(), best_delta=best.tolist())
            history.append(row); print(json.dumps(row), flush=True)
            path = Path(args.output); path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(history, indent=2))
finally:
    env.close()
