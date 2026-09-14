import hashlib,json,time
from pathlib import Path
import numpy as np
from embodied_fly.batch import FlyBatch
from embodied_fly.hover_only import HoverOnlyTasks
from embodied_fly.pid_imitation import BatchPID
from embodied_fly.observations import append_horizontal_error

started=time.perf_counter()
p=Path('docs/embodied_fly/runs/pid_movement_imitation_01')
r=json.loads((p/'training.json').read_text())
e=FlyBatch(7,2,16,preset='wing_position',wing_response='instant',physics_hz=1000)
tasks=HoverOnlyTasks(e,22);pid=BatchPID(e,tasks)
e.ages[:]=100  # Diagnostic clock assignment; no physical stepping.
origin=np.column_stack((e.requested_xy_cm,e.requested_height_cm)).copy()
delta=np.array([[0,0,0],[.15,0,0],[-.15,0,0],[0,.15,0],[0,-.15,0],[0,0,.15],[0,0,-.15]])
target=origin+delta
e.requested_xy_cm[:]=target[:,:2];e.requested_height_cm[:]=target[:,2]
before={k:v.copy() for k,v in e.fields.items()}
obs=e.observation()
expected=np.einsum('nji,nj->ni',e.fields['xmat'][:,e.template.thorax_id].reshape(-1,3,3),np.column_stack((delta[:,:2],np.zeros(7))))[:,:2]
np.testing.assert_allclose(obs[:,397:399]-obs[0,397:399],expected,atol=2e-7)
np.testing.assert_allclose(obs[:,396]*2,target[:,2],atol=2e-7)
np.testing.assert_allclose(obs[:,395]*2,e.fields['qpos'][:,2],atol=2e-7)
pred=pid.act()
acc=np.array([c.desired_acceleration for c in pid.controllers])
np.testing.assert_allclose(acc[1:,0:3]-acc[0],delta[1:]*(np.array([15,15,900])+np.array([3,3,1600])*.002),atol=2e-4)
for k,v in before.items():np.testing.assert_array_equal(v,e.fields[k])
# Pure coordinate fixture at 90 degrees; no body pose or simulation changes.
rot=np.array([[[0.,-1,0],[1,0,0],[0,0,1]]])
local=append_horizontal_error(np.zeros((1,0)),np.zeros((1,3)),rot,np.array([[.15,0]]),np.ones(1))
np.testing.assert_allclose(local,[[0,-.15]],atol=1e-7)
collected=[]
for trace in r['traces']:
 with np.load(Path('outputs/embodied_fly/pid_movement_imitation_01')/trace['file']) as d:
  o=d['observation']
  collected.append(o[:,:,395:399].reshape(-1,4))
o=np.concatenate(collected)
q=(50,90,95,99,100)
report={'scope':'Read-only command-path and saved-teacher-data audit. Zero training updates and zero physical steps; scratch state/clock and pure rotation fixtures only.','physical_steps':0,'training_updates':0,'actor_command_observation_indices':{'current_height_div_2cm':395,'target_height_div_2cm':396,'body_frame_horizontal_error_div_cm':[397,398]},'target_observation_encoding_passes':True,'world_to_body_90_degree_fixture_passes':True,'pid_acceleration_command_sign_and_gain_passes':True,'pid_query_changes_no_live_world_fields':True,'counterfactual_target_offsets_mm':(delta*10).tolist(),'counterfactual_pid_accelerations_cm_s2':acc.tolist(),'teacher_state_observations':len(o),'teacher_error_quantiles_percent':q,'teacher_horizontal_error_mm_quantiles':np.percentile(np.linalg.norm(o[:,2:4],axis=1)*10,q).tolist(),'teacher_altitude_error_mm_quantiles':np.percentile(np.abs(o[:,1]-o[:,0])*20,q).tolist(),'teacher_current_target_height_correlation':float(np.corrcoef(o[:,0],o[:,1])[0,1]),'elapsed_audit_seconds':time.perf_counter()-started,'limitations':'Checks command encoding and current-state PID label signs; does not prove the actor learned a stabilizing mapping or rule out other bugs. Accurate PID tracking yields small position-error examples even when the requested target moves.'}
(p/'command_path_audit.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
