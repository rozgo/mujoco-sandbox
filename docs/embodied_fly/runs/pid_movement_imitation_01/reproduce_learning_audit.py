"""Frozen-weight implementation audit. No optimizer or physical rollouts."""
import argparse
import json
import time
from pathlib import Path

import mujoco
import numpy as np
import torch

from embodied_fly.evaluate import load_actor
from embodied_fly.pid_imitation import imitation_loss
from embodied_fly.ppo_timing import recurrent_forward
from embodied_fly.provenance import sha256, utc_now
from embodied_fly.wing_position import normalize_targets, wing_actuators


def main(args):
    start = time.perf_counter()
    torch.set_num_threads(4)
    torch.manual_seed(120999)
    device = torch.device('cuda')
    report = {'started_utc': utc_now(), 'optimizer_updates': 0, 'physical_steps': 0,
              'script_sha256': sha256(Path(__file__))}
    trace = np.load(args.trace / 'batch_00000.npz')
    model = mujoco.MjModel.from_binary_path(str(args.trace / 'model.mjb'))
    wings = wing_actuators(model)
    assert np.array_equal(wings, np.arange(14, 20))
    labels = trace['teacher_action']
    physical_targets = model.actuator_ctrlrange[wings, 0] + (labels[..., wings] + 1)*.5*np.diff(model.actuator_ctrlrange[wings], axis=1).ravel()
    reverse = normalize_targets(model, physical_targets)
    report['actuator_interface'] = {
        'named_wing_ids': wings.tolist(),
        'names': [model.actuator(int(i)).name for i in wings],
        'residual_readout_ids_match': True,
        'normalized_target_roundtrip_max_error': float(np.abs(reverse-labels[..., wings]).max()),
        'trace_executed_equals_teacher': bool(np.array_equal(trace['executed_action'], labels)),
        'limits': model.actuator_ctrlrange[wings].tolist(),
    }
    assert report['actuator_interface']['normalized_target_roundtrip_max_error'] < 1e-6
    print('Actuator interface passed', flush=True)

    actor, parent = load_actor(args.parent, args.graph, device)
    obs = torch.as_tensor(trace['observation'], device=device)
    state = actor.initial_state(obs.shape[1])
    actions = []
    with torch.no_grad():
        for frame in obs:
            output = actor(frame, state)
            state = output.state
            actions.append(output.action.cpu().numpy())
    delta = np.stack(actions)-trace['student_action']
    report['parent_training_capture_replay'] = {
        'frames': len(obs), 'worlds': obs.shape[1],
        'max_absolute_action_error': float(np.abs(delta).max()),
        'rms_action_error': float(np.sqrt((delta**2).mean())),
        'same_checkpoint_sha256': sha256(args.parent),
        'trace_sha256': sha256(args.trace/'batch_00000.npz'),
        'scope': 'Exact pre-first-update parent, stored observations, cold recurrent state; deployment forward reproduces training predictions',
    }
    assert np.abs(delta).max() < 2e-5
    print('Saved parent reproduces first training sequence', flush=True)
    del actor, state, obs
    torch.cuda.empty_cache()

    actor, final = load_actor(args.final, args.graph, device)
    frozen = {k: v.detach().cpu().clone() for k,v in actor.state_dict().items()}
    worlds = [0, 32, 34, 36]
    # First sequence establishes measured wing history; gradients are tested in
    # a second short, matched sequence. No fitting or weight changes occur.
    state = actor.initial_state(len(worlds))
    with torch.no_grad():
        for frame in trace['observation'][:,worlds]:
            state = actor(torch.as_tensor(frame, device=device), state).state
    initial_state = state.detach().clone()
    second = np.load(args.trace/'batch_00001.npz')
    base_obs = torch.as_tensor(second['observation'][:16, worlds], device=device)
    targets = torch.as_tensor(second['teacher_action'][:16, worlds], device=device)
    snapshots = []
    for recompute in (False, True):
        actor.train(recompute)
        actor.zero_grad(set_to_none=True)
        observed = base_obs.detach().clone().requires_grad_(True)
        state = initial_state.detach().clone()
        outputs, losses = [], []
        for index, frame in enumerate(observed):
            if recompute:
                result = recurrent_forward(actor, frame, state, torch.ones(len(worlds),device=device,dtype=torch.long),1.0,True)
            else:
                result = actor(frame, state)
            outputs.append(result.action)
            losses.append(imitation_loss(result.action, targets[index], wings)[0])
            state = result.state
        loss = torch.stack(losses).mean()
        loss.backward()
        gradients = {k:p.grad.detach().cpu().clone() for k,p in actor.named_parameters() if p.requires_grad}
        snapshots.append((torch.stack(outputs).detach().cpu(), gradients, observed.grad.detach().cpu(), float(loss.detach())))
    forward_delta = (snapshots[0][0]-snapshots[1][0]).abs().max().item()
    gradient_difference = {k: float((snapshots[0][1][k]-snapshots[1][1][k]).abs().max()) for k in snapshots[0][1]}
    grad = snapshots[1][1]['sensor_extension.weight']
    report['train_vs_deployment'] = {
        'frames': 16, 'worlds': len(worlds),
        'action_max_difference': forward_delta,
        'loss_direct': snapshots[0][3], 'loss_training_wrapper': snapshots[1][3],
        'max_parameter_gradient_difference': max(gradient_difference.values()),
        'input_gradient_max_difference': float((snapshots[0][2]-snapshots[1][2]).abs().max()),
        'all_trainable_gradients_finite': all(bool(torch.isfinite(g).all()) for g in snapshots[1][1].values()),
        'sensor_extension_gradient_column_l2': torch.linalg.vector_norm(grad,dim=0).tolist(),
        'raw_observation_gradient_last_four_l2': torch.linalg.vector_norm(snapshots[1][2].reshape(-1,399),dim=0)[-4:].tolist(),
        'gradient_l2_by_parameter': {k: float(g.norm()) for k,g in snapshots[1][1].items()},
    }
    assert forward_delta < 1e-6
    assert report['train_vs_deployment']['max_parameter_gradient_difference'] < 1e-5
    assert torch.all(torch.linalg.vector_norm(grad,dim=0)[12:] > 0)
    assert report['train_vs_deployment']['all_trainable_gradients_finite']
    print('Train/deployment values and gradients match; all four target/height columns receive gradients', flush=True)
    # Perturb only current/target altitude and target displacement at matched
    # measured states, preserving recurrent history. No physics claim is made.
    actor.eval()
    actor.zero_grad(set_to_none=True)
    matched = base_obs[0:1,0].detach().clone()
    memory = initial_state[:,0:1].clone()
    intervention = []
    for channel in (395,396,397,398):
        probe = torch.cat([matched.clone(),matched.clone()])
        # +/- 1 mm actual or target height; +/- 1 mm planar target error.
        amount = .05 if channel < 397 else .1
        probe[0,channel] += amount
        probe[1,channel] -= amount
        with torch.no_grad():
            prediction = actor(probe,memory.expand(-1,2).contiguous()).action.cpu().numpy()
        intervention.append({'input_index':channel, 'plus_minus_mm':1.0, 'wing_action_plus_minus_difference':(prediction[0,wings]-prediction[1,wings]).tolist()})
    report['one_action_input_interventions'] = intervention
    report['parameters_and_buffers_unchanged'] = all(torch.equal(v.detach().cpu(),frozen[k]) for k,v in actor.state_dict().items())
    assert report['parameters_and_buffers_unchanged']
    report['completed_utc'] = utc_now()
    report['wall_seconds'] = time.perf_counter()-start
    report['final_checkpoint_sha256'] = sha256(args.final)
    report['scope'] = 'Tests actuator indexing/normalization, exact initial captured training predictions, real MaleCNS train/eval path and gradients. No root-cause assertion or proof of successful feedback learning.'
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--parent',type=Path,required=True)
    p.add_argument('--final',type=Path,required=True)
    p.add_argument('--graph',type=Path,required=True)
    p.add_argument('--trace',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    main(p.parse_args())
