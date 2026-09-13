"""Replay measured failed hover states; no neural inference or physics rollout."""
import json
from pathlib import Path
import mujoco
import numpy as np
from embodied_fly.provenance import sha256
from embodied_fly.state_hover import wing_commands
from embodied_fly.wing_position import reference_torque_to_position, wing_actuators

source = Path('outputs/embodied_fly/position_fullbody_01_evaluation')
model = mujoco.MjModel.from_binary_path(str(source/'model.mjb'))
capture = np.load(source/'hover.npz')
ids = wing_actuators(model)
joints = model.actuator_trnid[ids, 0]
qi, vi = model.jnt_qposadr[joints], model.jnt_dofadr[joints]
rows = []
for step in (0, 1, 5, 10, 20, 40, 80, 100, 200):
    q, v = capture['qpos'][step:step+1], capture['qvel'][step:step+1]
    data = mujoco.MjData(model)
    data.qpos[:], data.qvel[:] = q[0], v[0]
    mujoco.mj_forward(model, data)
    velocity = np.zeros(6)
    mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_XBODY,
                            model.body('walker/thorax').id, velocity, 1)
    torque = wing_commands(q[:, qi], v[:, vi], velocity[None], q[:,2],
                          capture['requested_height_cm'][step:step+1], model.qpos_spring[qi])
    target = reference_torque_to_position(model, torque, q[:,qi], v[:,vi], .002)
    rows.append({'seconds':step*.002, 'height_cm':float(q[0,2]),
                 'sweep_angles':q[0,qi[[0,3]]].tolist(),
                 'sweep_velocities':v[0,vi[[0,3]]].tolist(),
                 'actor':capture['action'][step,ids[[0,3]]].tolist(),
                 'current_state_reference':target[0,[0,3]].tolist()})
report = {'source_capture_sha256':sha256(source/'hover.npz'),
          'source_model_sha256':sha256(source/'model.mjb'),
          'learning':False,'extra_physical_transitions':0,'samples':rows}
Path(__file__).with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n')
print('Saved replay diagnostic for nine measured states.')
