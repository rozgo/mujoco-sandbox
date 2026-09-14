"""Causal velocity-input probes through the frozen MaleCNS actor; never deployed."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from embodied_fly.evaluate import load_actor
from embodied_fly.physical_contract import physical_contract
from embodied_fly.provenance import evidence, sha256, utc_now
from embodied_fly.velocity_demonstrations import environment
from embodied_fly.velocity_hover import EVALUATION_EPISODES
from embodied_fly.velocity_motor import observation
from embodied_fly.wing_readout_ppo import motor_features


def sensitivity(values, amplitude):
    derivative = torch.stack(
        [(values[1 + 2 * i] - values[2 + 2 * i]) / (2 * amplitude) for i in range(3)]
    ).double()
    singular = torch.linalg.svdvals(derivative)
    return {
        "rms_derivative_per_cm_s": derivative.square().mean(1).sqrt().cpu().tolist(),
        "max_abs_derivative_per_cm_s": derivative.abs().amax(1).cpu().tolist(),
        "singular_values": singular.cpu().tolist(),
        "duplicate_baseline_max_error": float((values[0] - values[7]).abs().max()),
    }


@torch.no_grad()
def diagnose(args):
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(args.output)
    started = time.perf_counter()
    provenance = evidence()
    torch.set_num_threads(4)
    device = torch.device(args.device)
    actor, parent = load_actor(args.checkpoint, args.graph, device)
    actor.eval()
    env = environment(4, 4)
    assert physical_contract(env.model) == parent["physical_contract"]
    training = json.loads((args.run / "report.json").read_text())
    if sha256(args.checkpoint) != training["checkpoint_sha256"]:
        raise ValueError("Captured history must correspond to these weights")
    captured = []
    for case in training["evaluations"]["final"]["cases"]:
        path = args.run / "final" / case["file"]
        assert sha256(path) == case["sha256"]
        with np.load(path) as saved:
            captured.append({k: saved[k].copy() for k in saved.files})
    assert len(captured) == len(EVALUATION_EPISODES)
    # Recorded state replay only: no physics advancement or policy updates here.
    observations = []
    for step in range(2550):
        for key in ("qpos", "qvel", "act", "ctrl"):
            env.fields[key][:] = np.stack([c[key][step] for c in captured])
        env.previous_action[:] = (
            np.stack([c["action"][step - 1] for c in captured]) if step else 0
        )
        env.batch.forward()
        observations.append(observation(env, np.zeros((4, 4), np.float32)))
    obs = torch.as_tensor(np.stack(observations), device=device)
    memory = actor.initial_state(4)
    records = []
    replay_error = 0.0
    for step in range(2501):
        if step in (200, 1000, 2500):
            branch = memory[:, :1].repeat(1, 8)
            for advance in range(50):
                inputs = obs[step + advance, :1].repeat(8, 1)
                for axis in range(3):
                    inputs[1 + 2 * axis, 285 + axis] += args.amplitude
                    inputs[2 + 2 * axis, 285 + axis] -= args.amplitude
                result = actor(inputs, branch)
                branch = result.state
                if advance in (0, 4, 24, 49):
                    records.append(
                        {
                            "history_seconds": step * 0.002,
                            "probe_seconds": (advance + 1) * 0.002,
                            "motor_features": sensitivity(
                                motor_features(actor, branch), args.amplitude
                            ),
                            "wing_actions": sensitivity(
                                result.action[:, 14:20], args.amplitude
                            ),
                        }
                    )
        result = actor(obs[step], memory)
        memory = result.state
        original = torch.as_tensor(
            np.stack([c["action"][step] for c in captured]), device=device
        )
        replay_error = max(replay_error, float((result.action - original).abs().max()))
    report = {
        "provenance": provenance,
        "completed_utc": utc_now(),
        "wall_seconds": time.perf_counter() - started,
        "checkpoint_sha256": sha256(args.checkpoint),
        "physical_contract": parent["physical_contract"],
        "amplitude_cm_s": args.amplitude,
        "axes": ["forward", "left", "up"],
        "input_indices": [285, 286, 287],
        "history_replay_max_action_error": replay_error,
        "scope": "Replay the recorded physical history, then perturb only measured velocity inputs for 2-100 ms; continue other observations from the original capture. Counterfactual neural response, not closed-loop physical recovery or proof that a linear decoder suffices.",
        "records": records,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--amplitude", type=float, default=0.1)
    parser.add_argument("--device", default="cuda")
    diagnose(parser.parse_args())
