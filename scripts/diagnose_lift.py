"""Measure complete physical rollouts; never move the dynamic object manually."""
import argparse, json
from pathlib import Path
import torch
from morphim_mmo.rl.config import load_residual_rl_config
from morphim_mmo.rl.reference import ResidualRLReference
from morphim_mmo.rl.environment import MorphometricResidualEnv, ResidualActionWrapper


def load_reference(config):
    return ResidualRLReference.load(robot_pose_path='outputs/wuji_hand2/robot_pose_full_safe.npz', contacts_path='../data/processed/s1/apple_lift_contacts.npz', robot_config_path='configs/wuji_hand2_beta2_right.yaml', mano_dir='../data/models/mano/mano_v1_2/models', object_mesh_path='../data/raw/grab/tools/object_meshes/contact_meshes/apple.ply', reference_stride=config.control.reference_stride)


def main():
    p=argparse.ArgumentParser(); p.add_argument('--config',default='configs/rl/wuji_hand2_apple.yaml'); p.add_argument('--output',default='outputs/rl/diagnostics/baseline.json'); p.add_argument('--num-envs',type=int,default=16); a=p.parse_args()
    c=load_residual_rl_config(a.config); r=load_reference(c)
    e=ResidualActionWrapper(MorphometricResidualEnv(reference=r,config=c,num_envs=a.num_envs,sim_backend='gpu'))
    e.reset(seed=1); b=e.unwrapped; z0=b.object.pose.p[:,2].clone(); peak=z0.clone(); failed=torch.zeros(a.num_envs,dtype=torch.bool,device=b.device); rows=[]
    with torch.no_grad():
        for t in range(r.frame_count):
            _,reward,term,trunc,info=e.step(torch.zeros((a.num_envs,20),device=b.device)); failed |= term; peak=torch.maximum(peak,b.object.pose.p[:,2]);
            if t%25==0 or t==r.frame_count-1:
                row={'step':t,'height':float(b.object.pose.p[:,2].mean()),'reference_height':float(b._object_pose_for_step(b._active_step)[:,2].mean()),'add':float(info['add_error'].mean()),'fingers':float(info['observed_contact'].float().sum(-1).mean()),'force':float(info['maximum_object_contact_force'].max()),'failed':int(failed.sum())}; rows.append(row); print(row,flush=True)
    result={'initial_height_m':float(z0.mean()),'peak_lift_m':(peak-z0).tolist(),'final_height_m':b.object.pose.p[:,2].tolist(),'success':(info['success'] & ~failed).tolist(),'rows':rows}
    path=Path(a.output); path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(result,indent=2)); e.close()
if __name__=='__main__': main()
