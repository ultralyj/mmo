"""Render a saved policy in GPU PhysX and verify sustained successful lifting."""
import argparse
import json
from dataclasses import replace
from pathlib import Path

import torch
import sapien
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from diagnose_lift import load_reference
from morphim_mmo.rl.config import load_residual_rl_config
from morphim_mmo.rl.environment import MorphometricResidualEnv, ResidualActionWrapper
from morphim_mmo.rl.ppo import ActorCritic, _single_env_action_limit
from morphim_mmo.rl.visualization import _rgb_image, _annotate_frame
from morphim_mmo.utils.io import save_gif

parser = argparse.ArgumentParser()
parser.add_argument('--config', default='configs/rl/wuji_hand2_apple.yaml')
parser.add_argument('--checkpoint', required=True)
parser.add_argument('--output', required=True)
parser.add_argument('--seed', type=int, default=101)
args = parser.parse_args()
config = load_residual_rl_config(args.config)
config = replace(config, ppo=replace(config.ppo, seed=args.seed))
reference = load_reference(config)
class CloseupLiftEnv(MorphometricResidualEnv):
    @property
    def _default_human_render_camera_configs(self):
        anchor = self.reference.object_pose_position_wxyz[0, :3]
        eye = anchor + [.28, -.38, .24]
        target = anchor + [0, 0, .05]
        return CameraConfig("render_camera", sapien_utils.look_at(eye=eye, target=target),
                            800, 600, .85, .01, 10.0)


env = ResidualActionWrapper(CloseupLiftEnv(
    reference=reference, config=config, num_envs=1, sim_backend='gpu', render_mode='rgb_array'))
try:
    observation, info = env.reset(seed=args.seed)
    base = env.unwrapped
    checkpoint = torch.load(args.checkpoint, map_location=base.device, weights_only=False)
    policy = ActorCritic(observation.shape[-1], 20, tuple(checkpoint['config']['hidden_sizes']),
                         _single_env_action_limit(env.action_space.high, 20, base.device)).to(base.device)
    policy.load_state_dict(checkpoint['policy']); policy.eval()
    frames = [_annotate_frame(_rgb_image(env.render()), 0, 0.0, info)]
    failed = False
    peak_lift = 0.0
    hold_contact = torch.zeros(5, device=base.device)
    hold_frames = 0
    with torch.no_grad():
        for step in range(reference.frame_count):
            action = policy.action_limit * torch.tanh(policy.forward(observation)[0].mean)
            observation, reward, terminated, truncated, info = env.step(action)
            failed |= bool(info['fail'][0])
            if config.control.hold_steps and step >= reference.frame_count - config.control.hold_steps:
                hold_contact += info['observed_contact'][0].float()
                hold_frames += 1
            peak_lift = max(peak_lift, float(info['object_lift'][0]))
            frames.append(_annotate_frame(_rgb_image(env.render()), step + 1, float(reward[0]), info))
            if bool((terminated | truncated)[0]):
                break
    success = bool(info['success'][0]) and not failed and float(info['object_lift'][0]) >= .04
    hold_fraction = hold_contact / max(hold_frames, 1)
    target = base._reference_contact[-1]
    finger_match = bool(torch.where(target, hold_fraction >= .9, hold_fraction <= .1).all())
    if config.reward.require_reference_contact:
        success = success and finger_match
    output = Path(args.output)
    save_gif(frames, output, fps=30)
    metadata = dict(success=success, seed=args.seed, checkpoint=args.checkpoint,
                    peak_lift_m=peak_lift, final_lift_m=float(info['object_lift'][0]),
                    final_add_m=float(info['add_error'][0]), failed=failed,
                    finger_order=['thumb','index','middle','ring','pinky'],
                    hold_contact_fraction=hold_fraction.tolist(),
                    final_finger_force_n=info['finger_contact_force'][0].tolist(),
                    finger_match_success=finger_match)
    output.with_suffix('.json').write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata), flush=True)
    if not success:
        raise RuntimeError('Rendered episode did not pass the sustained-lift criterion')
finally:
    env.close()
