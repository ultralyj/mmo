"""Record actual substep motion and mesh self-overlap, not just task success."""
import argparse
import json
from dataclasses import replace
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import torch
import trimesh
import yourdfpy
from pyroki.collision import RobotCollision
from diagnose_lift import load_reference
from morphim_mmo.rl.config import load_residual_rl_config
from morphim_mmo.rl.environment import MorphometricResidualEnv, ResidualActionWrapper
from morphim_mmo.rl.ppo import ActorCritic, _single_env_action_limit

parser = argparse.ArgumentParser()
parser.add_argument('--config', default='configs/rl/wuji_hand2_apple_fingers.yaml')
parser.add_argument('--checkpoint')
parser.add_argument('--output', default='outputs/rl/diagnostics/physics_audit_legacy.json')
parser.add_argument('--backend', default='gpu')
parser.add_argument('--seed', type=int, default=101)
args = parser.parse_args()
c = load_residual_rl_config(args.config)
c = replace(c, ppo=replace(c.ppo, seed=args.seed))
reference = load_reference(c)
env = ResidualActionWrapper(MorphometricResidualEnv(reference=reference, config=c,
                            num_envs=1, sim_backend=args.backend))
base = env.unwrapped
observation, _ = env.reset(seed=args.seed)
policy = None
if args.checkpoint:
    checkpoint = torch.load(args.checkpoint, map_location=base.device, weights_only=False)
    policy = ActorCritic(observation.shape[-1], 20, tuple(checkpoint['config']['hidden_sizes']),
                        _single_env_action_limit(env.action_space.high, 20, base.device)).to(base.device)
    policy.load_state_dict(checkpoint['policy']); policy.eval()

urdf_xml = ET.parse(base.agent.urdf_path).getroot()
limits = {j.get('name'): {k:float(v) for k,v in j.find('limit').attrib.items()}
          for j in urdf_xml.findall('joint') if j.find('limit') is not None}
excluded = {frozenset([j.find('parent').get('link'), j.find('child').get('link')])
            for j in urdf_xml.findall('joint')}
mjcf = ET.parse(Path(base.agent.urdf_path).parents[1] / 'mjcf/right.xml').getroot()
excluded.update(frozenset([x.get('body1'), x.get('body2')]) for x in mjcf.findall('contact/exclude'))
asset_actuators = {x.get('joint'):dict(kp=float(x.get('kp')), kv=float(x.get('kv'))) for x in mjcf.findall('actuator/position')}
joint_names = [j.name for j in base.agent.robot.active_joints]
settings = []
for name,joint in zip(joint_names,base.agent.robot._objs[0].active_joints):
    settings.append(dict(name=name, urdf=limits[name], asset_pd=asset_actuators[name],
                         stiffness=joint.stiffness, damping=joint.damping, force_limit=joint.force_limit,
                         max_velocity=joint.max_velocity, armature=np.asarray(joint.armature).tolist()))
excluded = base.agent.excluded_collision_pairs
robot_urdf = yourdfpy.URDF.load(base.agent.urdf_path, load_meshes=False, load_collision_meshes=True)
manager = trimesh.collision.CollisionManager()
shape_records = []
for link in base._robot_links:
    mesh = RobotCollision._get_trimesh_collision_geometries(robot_urdf, link.name)
    if mesh.is_empty:
        continue
    manager.add_object(link.name, mesh.convex_hull)
    for shape in link._bodies[0].collision_shapes:
        shape_records.append(dict(link=link.name, type=type(shape).__name__,
                                  collision_groups=list(shape.collision_groups),
                                  contact_offset=float(shape.contact_offset), rest_offset=float(shape.rest_offset),
                                  disable_gravity=bool(link._bodies[0].disable_gravity)))
records=[]; collisions={}; worst_depth=0.0
original_hook=base._after_simulation_step

def observe_substep():
    global worst_depth
    original_hook()
    if base.scene.gpu_sim_enabled:
        base.scene._gpu_fetch_all()
    q=base.agent.robot.get_qpos()[0].cpu().numpy()
    v=base.agent.robot.get_qvel()[0].cpu().numpy()
    target=base.agent.controller._target_qpos[0].cpu().numpy()
    if base.agent.controller.config.interpolate:
        target=(base.agent.controller._start_qpos + base.agent.controller._step * base.agent.controller._step_size)[0].cpu().numpy()
    # This demand estimate is not a measurement of PhysX's implicit drive force.
    demand=np.array([x['stiffness'] for x in settings])*(target-q)-np.array([x['damping'] for x in settings])*v
    for link in base._robot_links:
        if link.name in manager._objs:
            manager.set_transform(link.name,link.pose.to_transformation_matrix()[0].cpu().numpy())
    _, _, contacts=manager.in_collision_internal(return_names=True,return_data=True)
    max_depth=0.0
    for contact in contacts:
        pair=frozenset(contact.names)
        if pair in excluded:
            continue
        depth=float(contact.depth)
        if depth < 1e-5:
            continue
        label=' / '.join(sorted(pair)); collisions[label]=max(collisions.get(label,0.0),depth)
        max_depth=max(max_depth,depth)
    worst_depth=max(worst_depth,max_depth)
    records.append(dict(control_step=int(base._active_step[0]),qpos=q.tolist(),qvel=v.tolist(),
                        drive_target=target.tolist(),pd_demand_estimate_nm=demand.tolist(),
                        self_penetration_m=max_depth))

base._after_simulation_step=observe_substep
try:
    failed=False
    with torch.no_grad():
        for step in range(reference.frame_count):
            action=torch.zeros((1,20),device=base.device) if policy is None else policy.action_limit*torch.tanh(policy.forward(observation)[0].mean)
            observation,_,_,_,info=env.step(action); failed |= bool(info['fail'][0])
    q=np.array([x['qpos'] for x in records]);v=np.array([x['qvel'] for x in records]);targets=np.array([x['drive_target'] for x in records])
    raw=reference.qpos[:,[reference.joint_names.index(n) for n in joint_names]]
    corrected=base._reference_qpos.cpu().numpy()
    summary=[]
    for k,name in enumerate(joint_names):
        l=limits[name];margin=np.minimum(q[:,k]-l['lower'],l['upper']-q[:,k])
        summary.append(dict(name=name,raw_reference_final_rad=float(raw[-1,k]),
                            corrected_reference_final_rad=float(corrected[-1,k]),actual_final_rad=float(q[-1,k]),
                            actual_min_rad=float(q[:,k].min()),actual_max_rad=float(q[:,k].max()),
                            max_actual_speed_rad_s=float(np.abs(v[:,k]).max()),
                            velocity_limit_rad_s=l['velocity'],min_limit_margin_rad=float(margin.min()),
                            command_limit_violation_rad=float(np.maximum(l['lower']-targets[:,k],targets[:,k]-l['upper']).clip(0).max()),
                            speed_exceed_substeps=int((np.abs(v[:,k])>l['velocity']+1e-3).sum()),
                            configured_effort_to_urdf_ratio=settings[k]['force_limit']/l['effort']))
    result=dict(articulation_solver_position_iterations=base.agent.robot._objs[0].solver_position_iterations,
                articulation_solver_velocity_iterations=base.agent.robot._objs[0].solver_velocity_iterations,
                config=args.config,checkpoint=args.checkpoint,backend=args.backend,seed=args.seed,
                drive_settings=settings,collision_shapes=shape_records,joints=summary,
                worst_self_penetration_m=worst_depth,
                nonexcluded_self_overlaps=sorted(collisions.items(),key=lambda x:-x[1]),
                success=bool(info['success'][0]) and not failed,final_lift_m=float(info['object_lift'][0]),
                finger_force_n=info['finger_contact_force'][0].tolist(),substeps=records,
                notes=['PD demand is an explicit-law estimate, not a measured motor torque.',
                       'FCL checks convex hulls of URDF collision meshes, excluding only adjacency and vendor MJCF exclusions.',
                       'The URDF has 31 link frames, including 26 collision-bearing rigid links; it does not model internal gears, tendons or backlash.'])
    path=Path(args.output);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k not in ['substeps','collision_shapes','drive_settings']},indent=2),flush=True)
finally:
    env.close()
