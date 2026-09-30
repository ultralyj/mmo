"""Batched simulation-in-the-loop CEM for bounded reference grasp corrections."""
import argparse, json, time
from pathlib import Path
import numpy as np
import torch
from diagnose_lift import load_reference
from morphim_mmo.rl.config import load_residual_rl_config
from morphim_mmo.rl.environment import MorphometricResidualEnv, ResidualActionWrapper, _grasp_bias_activation

p=argparse.ArgumentParser(); p.add_argument('--rounds',type=int,default=15); p.add_argument('--num-envs',type=int,default=256); p.add_argument('--steps',type=int,default=260); p.add_argument('--output',default='outputs/rl/diagnostics/cem.json'); a=p.parse_args()
c=load_residual_rl_config('configs/rl/wuji_hand2_apple.yaml'); r=load_reference(c); e=ResidualActionWrapper(MorphometricResidualEnv(reference=r,config=c,num_envs=a.num_envs,sim_backend='gpu')); b=e.unwrapped; e.reset(seed=1)
# Reference bias is already incorporated by environment. Optimize a bounded additional correction.
gate=_grasp_bias_activation(b._reference_contact,c.control.grasp_contact_lookahead_steps,c.control.grasp_min_contact_fingers,c.control.grasp_release_decay)
mean=torch.zeros(20,device=b.device); std=torch.full_like(mean,.18); best=mean.clone(); best_score=-1e9; history=[]
with torch.no_grad():
 for it in range(a.rounds):
    torch.manual_seed(100+it); candidates=(mean+std*torch.randn(a.num_envs,20,device=b.device)).clamp(-.5,.5); candidates[0]=best; candidates[1]=0
    e.reset(seed=1); total=torch.zeros(a.num_envs,device=b.device); failed=torch.zeros_like(total,dtype=torch.bool); peak=total.clone(); z0=b.object.pose.p[:,2].clone(); maxforce=total.clone()
    for t in range(min(a.steps,r.frame_count)):
        # Direct q command deliberately allows searching a new reference beyond policy residual bounds.
        q=(b.current_reference_qpos+gate[t]*candidates).clamp(b._joint_lower,b._joint_upper)
        _,rew,term,trunc,info=b.step(q)
        failed |= term
        total += rew - .005*info['maximum_object_contact_force'].clamp_min(100).sub(100) - 2*info['terminate_table_height'].float() - 2*info['terminate_table_force'].float()
        peak=torch.maximum(peak,b.object.pose.p[:,2]-z0); maxforce=torch.maximum(maxforce,info['maximum_object_contact_force'])
    elite=total.topk(max(8,a.num_envs//10)).indices; mean=.25*mean+.75*candidates[elite].mean(0); std=(.25*std+.75*candidates[elite].std(0)).clamp_min(.015)
    ix=int(total.argmax()); score=float(total[ix]);
    if score>best_score: best_score=score; best=candidates[ix].clone()
    row={'round':it,'score':score,'peak_lift':float(peak[ix]),'final_add':float(info['add_error'][ix]),'max_force':float(maxforce[ix]),'failed':bool(failed[ix]),'delta_bias':best.tolist()}; history.append(row); print(json.dumps(row),flush=True)
    path=Path(a.output); path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(history,indent=2))
e.close()
