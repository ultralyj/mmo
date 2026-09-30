"""Real GPU checks for reset momentum, partial resets and reference ownership."""
import numpy as np
import torch
from diagnose_lift import load_reference
from morphim_mmo.rl.config import load_residual_rl_config
from morphim_mmo.rl.environment import MorphometricResidualEnv, ResidualActionWrapper

c=load_residual_rl_config('configs/rl/wuji_hand2_apple.yaml')
r=load_reference(c); original=r.base_pose_position_wxyz.copy()
e=ResidualActionWrapper(MorphometricResidualEnv(reference=r,config=c,num_envs=4,sim_backend='gpu'))
e.reset(seed=1); b=e.unwrapped
np.testing.assert_array_equal(r.base_pose_position_wxyz,original)
b.object.set_linear_velocity(torch.ones((4,3),device=b.device))
b.object.set_angular_velocity(torch.ones((4,3),device=b.device))
b.scene._gpu_apply_all(); b.scene._gpu_fetch_all()
e.reset(options={'env_idx':torch.tensor([1,3],device=b.device)})
torch.testing.assert_close(b.object.linear_velocity[[1,3]],torch.zeros((2,3),device=b.device))
torch.testing.assert_close(b.object.angular_velocity[[1,3]],torch.zeros((2,3),device=b.device))
torch.testing.assert_close(b.object.linear_velocity[[0,2]],torch.ones((2,3),device=b.device))
e.reset(seed=1)
_,_,_,_,info=e.step(torch.zeros((4,20),device=b.device))
torch.testing.assert_close(info['add_ema'],info['add_error'],atol=1e-6,rtol=1e-6)
np.testing.assert_array_equal(r.base_pose_position_wxyz,original)
e.close(); print('GPU reset, partial reset, current-step EMA and reference ownership checks passed.')
