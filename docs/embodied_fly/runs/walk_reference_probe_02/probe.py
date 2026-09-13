import json,time
from pathlib import Path
import numpy as np,mujoco,torch
from embodied_fly.body import FlyEnvironment
from embodied_fly.teacher import TeacherOracle
from embodied_fly.wing_position import normalize_targets,wing_actuators
from embodied_fly.physical_contract import physical_contract
from embodied_fly.provenance import evidence
out=Path('outputs/embodied_fly/walk_reference_probe_02');out.mkdir(exist_ok=False)
torch.set_num_threads(2)
env=FlyEnvironment('wing_position');oracle=TeacherOracle(env,Path('assets/embodied_fly/teachers/walking.npz'))
results=[]
for mode in ('receding','world_path'):
 for speed in (1.,2.):
  env.reset(yaw=.05);oracle.set_reference(speed,0,5,heading=.05)
  wing=normalize_targets(env.model,env.data.qpos[env.wing_angle_indices]);q=[];a=[];v=[];u=[];warn=0;start=time.perf_counter()
  for step in range(2500):
   act=oracle.act(step,mode);act[wing_actuators(env.model)]=wing
   q.append(env.data.qpos.copy());a.append(act.copy());v.append(env.anatomical_velocity());u.append(env.data.xmat[env.thorax_id,8])
   env.step(act)
  key=f'{mode}_{speed:g}'
  np.savez_compressed(out/(key+'.npz'),qpos=q,action=a,velocity=v,upright=u)
  r={'case':key,'seconds':time.perf_counter()-start,'stable':bool(min(u)>.5),'forward_cm':float(env.data.qpos[0]*np.cos(.05)+env.data.qpos[1]*np.sin(.05)),'speed_rmse_cm_s':float(np.sqrt(np.mean((np.array(v)[:,3]-speed)**2))),'yaw_rmse_rad_s':float(np.sqrt(np.mean(np.array(v)[:,2]**2))),'max_forbidden_load':env.maximum_disallowed_ground_force,'warnings':int(env.data.warning.number.sum())}
  results.append(r);print(json.dumps(r),flush=True)
(out/'report.json').write_text(json.dumps({'provenance':evidence(),'physical_contract':physical_contract(env.model),'learning':False,'cases':results},indent=2)+'\n')
