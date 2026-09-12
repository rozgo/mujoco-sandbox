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
from embodied_fly.brain import ACTIVITIES, EmbodiedBrain, load_malecns
from embodied_fly.neural_view import NeuralProjection
from embodied_fly.provenance import evidence, sha256, utc_now


def load_actor(path, graph_path, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    graph_hash = hashlib.sha256((graph_path / "weights.npz").read_bytes()).hexdigest()
    if graph_hash != checkpoint["graph_sha256"]:
        raise ValueError("Checkpoint refers to a different graph")
    if checkpoint.get("graph_metadata_sha256") is not None and checkpoint[
        "graph_metadata_sha256"
    ] != sha256(graph_path / "brain.npz"):
        raise ValueError("Checkpoint refers to different neuron metadata")
    adjacency, sensory, descending, motor = load_malecns(graph_path)
    for name, ids in (
        ("sensory_ids", sensory),
        ("descending_ids", descending),
        ("motor_ids", motor),
    ):
        if not np.array_equal(checkpoint["state_dict"][name].numpy(), ids):
            raise ValueError("Checkpoint neuron routing differs from graph metadata")
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
    if args.seconds <= 0 or not 1 <= args.cases <= 6:
        raise ValueError("Positive duration and one to six cases required")
    torch.set_num_threads(4)
    started = time.perf_counter()
    args.output.mkdir(parents=True, exist_ok=False)
    run_evidence = evidence()
    device = torch.device(args.device)
    actor, checkpoint = load_actor(args.checkpoint, args.graph, device)
    forced_activity = (
        torch.tensor([ACTIVITIES.index(args.diagnostic_activity)], device=device)
        if args.diagnostic_activity
        else None
    )
    environment = FlyEnvironment()
    projection = NeuralProjection.from_graph(args.graph, device) if args.neural_view else None
    mujoco.mj_saveModel(environment.model, str(args.output / "model.mjb"))
    setup_seconds = time.perf_counter() - started
    rng = np.random.default_rng(args.seed)
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
        upright, heights, speed_errors, yaw_errors, neural_activity = [], [], [], [], []
        neural_maps = []
        trace_ids = np.linspace(0, actor.core.neurons - 1, 256).astype(int)
        start_pos = environment.data.qpos[:3].copy()
        wall_start = time.perf_counter()
        numerical_failure = None
        for step in range(round(args.seconds / CONTROL_DT)):
            observation = torch.as_tensor(environment.observation()[None], device=device)
            result = actor(observation, memory, activity_override=forced_activity)
            memory = result.state
            action = result.action[0].cpu().numpy()
            if args.walking_action_mask:
                # Explicit curriculum diagnostic: passive wings/mouth/antennae,
                # not a supplied gait. Keep the unmasked result as the baseline.
                action = environment.walking_action(action)
            actions.append(action)
            qpos.append(environment.data.qpos.copy())
            qvel.append(environment.data.qvel.copy())
            activations.append(environment.data.act.copy())
            controls.append(environment.data.ctrl.copy())
            utilities.append(result.utility_scores[0].cpu().numpy())
            neural_activity.append(memory[trace_ids, 0].cpu().numpy())
            if projection is not None and step % 10 == 0:
                neural_maps.append(projection.project(memory)[0].astype(np.float16))
            try:
                environment.step(action)
            except RuntimeError as error:
                numerical_failure = str(error)
                break
            data = environment.data
            upright.append(float(data.xmat[environment.thorax_id, 8]))
            heights.append(float(data.qpos[2] * 0.01))
            velocity = environment.anatomical_velocity()
            speed_errors.append(float(velocity[3] - speed))
            yaw_errors.append(float(velocity[2] - turn))
        distance = (environment.data.qpos[:2] - start_pos[:2]) * 0.01
        # All cases use the same declared thresholds, including failures.
        stable = bool(upright) and min(upright) > 0.5 and min(heights) > 0.0006
        velocity_rmse = (
            float(np.sqrt(np.mean(np.square(speed_errors)))) if speed_errors else None
        )
        yaw_rmse = float(np.sqrt(np.mean(np.square(yaw_errors)))) if yaw_errors else None
        body_weight = environment.model.body_mass.sum() * 981
        support_ratio = environment.maximum_disallowed_ground_force / body_weight
        success = (
            stable
            and numerical_failure is None
            and velocity_rmse < max(0.5, speed * 0.5)
            and yaw_rmse < 0.5
            and support_ratio < 0.1
        )
        block = round(0.1 / CONTROL_DT)
        usable = len(speed_errors) // block * block
        block_errors = (
            np.column_stack([speed_errors, yaw_errors])[:usable].reshape(-1, block, 2).mean(1)
            if usable
            else None
        )
        report = {
            "case": name,
            "command_cm_s_rad_s": [speed, turn],
            "simulated_seconds": float(environment.data.time),
            "wall_seconds": time.perf_counter() - wall_start,
            "stable": stable,
            "success": success,
            "velocity_rmse_cm_s": velocity_rmse,
            "yaw_rmse_rad_s": yaw_rmse,
            "max_disallowed_ground_force_over_weight": support_ratio,
            "minimum_upright": min(upright) if upright else None,
            "block_mean_rmse_cm_s_rad_s": np.sqrt(np.square(block_errors).mean(0)).tolist()
            if block_errors is not None
            else None,
            "block_mean_window_seconds": block * CONTROL_DT,
            "block_mean_frames": usable,
            "block_mean_is_diagnostic_only": True,
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
            **(
                {
                    "neural_map": neural_maps,
                    "neural_occupancy": projection.occupancy,
                    "neural_map_stride": 10,
                }
                if projection is not None
                else {}
            ),
        )
        report["state_sha256"] = sha256(args.output / f"{name}.npz")
        print(json.dumps(report), flush=True)
    manifest = {
        "provenance": run_evidence,
        "completed_utc": utc_now(),
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "model_sha256": sha256(args.output / "model.mjb"),
        "training_source_commit": checkpoint["source_commit"],
        "graph_metadata_sha256": sha256(args.graph / "brain.npz"),
        "checkpoint_metadata_verification": "SHA-256 and routing"
        if checkpoint.get("graph_metadata_sha256")
        else "legacy checkpoint: exact graph weights and neuron routing; no stored metadata digest",
        "device": str(device),
        "physics": "native MuJoCo CPU",
        "setup_seconds": setup_seconds,
        "one_checkpoint_for_all_cases": True,
        "teacher_present": False,
        "training_method": checkpoint.get("method"),
        "diagnostic_activity_override": args.diagnostic_activity,
        "policy_acceptance_eligible": forced_activity is None,
        "scripted_gait_present": False,
        "walking_action_mask": args.walking_action_mask,
        "active_actuators": int((~environment.walking_inactive).sum())
        if args.walking_action_mask
        else environment.model.nu,
        "evaluation_seed": args.seed,
        "seed_role": "development/model-selection"
        if args.seed == 80001
        else "explicit alternate seed",
        "tracking_coordinate_frame": "anatomical thorax (mjOBJ_XBODY), x forward / z yaw",
        "actor_velocity_observation_frame": "principal inertia frame retained for v1 checkpoint compatibility",
        "neural_view": projection.report() if projection is not None else None,
        "results": results,
        "physical_success_count": sum(r["success"] for r in results),
    }
    (args.output / "report.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seconds", type=float, default=2.0)
    parser.add_argument("--cases", type=int, default=6)
    parser.add_argument("--walking-action-mask", action="store_true")
    parser.add_argument("--neural-view", action="store_true")
    parser.add_argument("--seed", type=int, default=80001)
    parser.add_argument(
        "--diagnostic-activity",
        choices=ACTIVITIES,
        help="Force a utility choice to isolate motor ability; not deployable-policy acceptance",
    )
    report = evaluate(parser.parse_args())
    raise SystemExit(
        0
        if report["policy_acceptance_eligible"]
        and report["physical_success_count"] == len(report["results"])
        else 2
    )
