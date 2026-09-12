"""Free-body student evaluation: one checkpoint, no teacher calls or gait driver."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import mujoco
import numpy as np
import torch

from embodied_fly.body import CONTROL_DT, FlyEnvironment
from embodied_fly.brain import EmbodiedBrain, load_malecns


def load_actor(path, graph_path, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    graph_hash = hashlib.sha256((graph_path / "weights.npz").read_bytes()).hexdigest()
    if graph_hash != checkpoint["graph_sha256"]:
        raise ValueError("Checkpoint refers to a different graph")
    adjacency, sensory, descending, motor = load_malecns(graph_path)
    brain = EmbodiedBrain(
        adjacency,
        sensory,
        descending,
        motor,
        checkpoint["observation_size"],
        checkpoint["action_size"],
        checkpoint["config"]["internal_steps"],
    )
    brain.load_state_dict(checkpoint["state_dict"], strict=True)
    return brain.to(device).eval(), checkpoint


@torch.no_grad()
def evaluate(args):
    torch.set_num_threads(4)
    started = time.perf_counter()
    device = torch.device(args.device)
    actor, checkpoint = load_actor(args.checkpoint, args.graph, device)
    environment = FlyEnvironment()
    args.output.mkdir(parents=True, exist_ok=True)
    mujoco.mj_saveModel(environment.model, str(args.output / "model.mjb"))
    setup_seconds = time.perf_counter() - started
    rng = np.random.default_rng(80001)
    cases = [
        ("hold", 0.0, 0.0),
        ("slow_walk", 0.5, 0.0),
        ("walk", 1.0, 0.0),
        ("fast_walk", 2.0, 0.0),
        ("left", 1.5, 0.75),
        ("right", 1.5, -0.75),
    ]
    results = []
    for name, speed, turn in cases[: args.cases]:
        environment.reset(yaw=float(rng.uniform(-0.2, 0.2)))
        environment.command[:] = (speed, 0, turn)
        memory = actor.initial_state(1)
        qpos, qvel, activations, controls, utilities, actions = [], [], [], [], [], []
        upright, heights, speed_errors, neural_activity = [], [], [], []
        trace_ids = np.linspace(0, actor.core.neurons - 1, 256).astype(int)
        start_pos = environment.data.qpos[:3].copy()
        wall_start = time.perf_counter()
        numerical_failure = None
        for _ in range(round(args.seconds / CONTROL_DT)):
            observation = torch.as_tensor(environment.observation()[None], device=device)
            result = actor(observation, memory)
            memory = result.state
            action = result.action[0].cpu().numpy()
            actions.append(action)
            qpos.append(environment.data.qpos.copy())
            qvel.append(environment.data.qvel.copy())
            activations.append(environment.data.act.copy())
            controls.append(environment.data.ctrl.copy())
            utilities.append(result.utility_scores[0].cpu().numpy())
            neural_activity.append(memory[trace_ids, 0].cpu().numpy())
            try:
                environment.step(action)
            except RuntimeError as error:
                numerical_failure = str(error)
                break
            data = environment.data
            upright.append(float(data.xmat[environment.thorax_id, 8]))
            heights.append(float(data.qpos[2] * 0.01))
            velocity = np.empty(6)
            mujoco.mj_objectVelocity(
                environment.model,
                data,
                mujoco.mjtObj.mjOBJ_BODY,
                environment.thorax_id,
                velocity,
                1,
            )
            speed_errors.append(float(velocity[3] - speed))
        distance = (environment.data.qpos[:2] - start_pos[:2]) * 0.01
        # All cases use the same declared thresholds, including failures.
        stable = bool(upright) and min(upright) > 0.5 and min(heights) > 0.0006
        velocity_rmse = (
            float(np.sqrt(np.mean(np.square(speed_errors)))) if speed_errors else None
        )
        success = (
            stable and numerical_failure is None and velocity_rmse < max(0.5, speed * 0.5)
        )
        report = {
            "case": name,
            "command_cm_s_rad_s": [speed, turn],
            "simulated_seconds": len(actions) * CONTROL_DT,
            "wall_seconds": time.perf_counter() - wall_start,
            "stable": stable,
            "success": success,
            "velocity_rmse_cm_s": velocity_rmse,
            "minimum_upright": min(upright) if upright else None,
            "displacement_m": distance.tolist(),
            "numerical_failure": numerical_failure,
            "warning_count": int(environment.data.warning.number.sum()),
        }
        results.append(report)
        np.savez_compressed(
            args.output / f"{name}.npz",
            qpos=qpos,
            qvel=qvel,
            activation=activations,
            ctrl=controls,
            action=actions,
            utility=utilities,
            neural_activity=neural_activity,
            neural_ids=trace_ids,
        )
        print(json.dumps(report), flush=True)
    manifest = {
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "training_source_commit": checkpoint["source_commit"],
        "device": str(device),
        "physics": "native MuJoCo CPU",
        "setup_seconds": setup_seconds,
        "one_checkpoint_for_all_cases": True,
        "teacher_present": False,
        "scripted_gait_present": False,
        "evaluation_seed": 80001,
        "results": results,
        "physical_success_count": sum(r["success"] for r in results),
    }
    (args.output / "report.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seconds", type=float, default=2.0)
    parser.add_argument("--cases", type=int, default=6)
    evaluate(parser.parse_args())
