"""Verify actual loaded hardware and per-scene collision masks on GPU."""
import itertools
import argparse
import numpy as np
import torch
from diagnose_lift import load_reference
from morphim_mmo.rl.config import load_residual_rl_config
from morphim_mmo.rl.environment import MorphometricResidualEnv, ResidualActionWrapper

parser = argparse.ArgumentParser()
parser.add_argument('--config', default='configs/rl/wuji_hand2_apple_physical_grasp.yaml')
args = parser.parse_args()
config = load_residual_rl_config(args.config)
env = ResidualActionWrapper(MorphometricResidualEnv(reference=load_reference(config), config=config,
                                                  num_envs=4, sim_backend='gpu'))
try:
    env.reset(seed=101)
    base = env.unwrapped
    names = [j.name for j in base.agent.robot.active_joints]
    for articulation in base.agent.robot._objs:
        for name,joint in zip(names, articulation.active_joints):
            hardware = base.agent.hardware[name]
            assert joint.force_limit <= hardware.effort_nm + 1e-6
            assert abs(joint.max_velocity - hardware.velocity_rad_s) < 1e-5
            np.testing.assert_allclose(joint.armature, hardware.armature_kg_m2, rtol=1e-6)
        assert all(not link.disable_gravity for link in articulation.links)
    links = [link for link in base._robot_links if link._bodies[0].collision_shapes]
    for index in range(4):
        for left,right in itertools.combinations(links,2):
            a=left._bodies[index].collision_shapes[0].collision_groups
            b=right._bodies[index].collision_shapes[0].collision_groups
            ignored=bool(a[2]&b[2]) and ((a[3]&0xffff)==(b[3]&0xffff))
            assert ignored == (frozenset([left.name,right.name]) in base.agent.excluded_collision_pairs)
    for index,name in enumerate(names):
        if 'pinky' in name:
            assert float(env._residual_limits[index]) <= .050001
    for _ in range(5):
        env.step(torch.zeros((4,20),device=base.device))
    print('Hardware effort/velocity/armature, gravity, 4-scene collision matrix and pinky action bounds passed.')
finally:
    env.close()
