"""Physical parameter search over two existing wing-sweep decoder rows.

Candidate gain/bias values are baked into ordinary actor weights on export.
The batched search hook only evaluates several constant candidate matrices at
once. It sees no task, clock or raw sensor. There is no deployed extra module.
"""

import argparse
import json
import time
from pathlib import Path

import mujoco
import numpy as np
import torch

from embodied_fly.batch import FlyBatch
from embodied_fly.evaluate import load_actor
from embodied_fly.ground_posture import GroundPosture
from embodied_fly.motor_focus import MotorTasks
from embodied_fly.physical_contract import physical_contract
from embodied_fly.provenance import evidence, sha256, utc_now
from embodied_fly.train import synchronize


def calibrated_state(state, rows, gain, bias):
    if len(rows) != 2 or len(set(rows)) != 2:
        raise ValueError("Two distinct sweep rows required")
    if not np.isfinite([gain, bias]).all() or gain <= 0:
        raise ValueError("Finite positive gain and finite bias required")
    result = {k: v.clone() for k, v in state.items()}
    weight, offset = result["motor_decoder.3.weight"], result["motor_decoder.3.bias"]
    if min(rows) < 0 or max(rows) >= len(offset):
        raise ValueError("Sweep row outside the actor")
    weight[rows] *= gain
    offset[rows] = offset[rows] * gain + bias
    return result


class CandidateRows:
    """Training-search arithmetic equivalent to constant final linear weights."""

    def __init__(self, actor, rows, candidates):
        self.rows = rows
        device = actor.core.bias.device
        values = torch.as_tensor(candidates, device=device, dtype=torch.float32)
        self.gain, self.bias = values[:, :1], values[:, 1:]
        self.handle = actor.motor_decoder[3].register_forward_hook(self.apply)

    def apply(self, module, inputs, logits):
        if len(logits) != len(self.gain):
            raise ValueError("Candidate/world count mismatch")
        result = logits.clone()
        result[:, self.rows] = logits[:, self.rows] * self.gain + self.bias
        return result

    def close(self):
        self.handle.remove()


def select_candidate(results):
    """Ground retention first; only promote a >=5% lower physical hover cost."""
    baseline = next(r for r in results if r["gain"] == 1 and r["bias"] == 0)
    for result in results:
        retention = True
        for i in (0, 1):
            current, old = result["cases"][i], baseline["cases"][i]
            retention &= current["stable"] and current["max_forbidden_load"] < 0.1
            for key, tolerance in (
                ("wing_rms_rad", 0.005),
                ("speed_rmse_cm_s", 0.05),
                ("yaw_rmse_rad_s", 0.05),
            ):
                retention &= current[key] <= 1.25 * old[key] + tolerance
            retention &= current["wing_max_rad"] <= old["wing_max_rad"] + 0.05
        result["ground_retention"] = bool(retention)
        result["hover_cost"] = float(
            np.mean(
                [c["root_rmse_cm"] + 2 * c["failed_fraction"] for c in result["cases"][2:]]
            )
        )
    eligible = [r for r in results if r["ground_retention"]]
    best = min(eligible, key=lambda r: r["hover_cost"]) if eligible else baseline
    improved = best["hover_cost"] < 0.95 * baseline["hover_cost"]
    return best if improved else baseline, bool(improved)


@torch.no_grad()
def search(args):
    if args.seconds <= 0 or not np.isfinite(args.seconds):
        raise ValueError("Positive physical window required")
    args.output.mkdir(parents=True, exist_ok=False)
    setup_start = time.perf_counter()
    provenance = evidence()
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    actor, parent = load_actor(args.resume, args.graph, device)
    if not actor.motor_only or actor.observation_size != 397:
        raise ValueError("Canonical motor-only actor required")
    actor.requires_grad_(False)
    env = FlyBatch(32, args.threads, 14, preset="wing_motion")
    if parent["physical_contract"] != physical_contract(env.model):
        raise ValueError("Search physical body differs from parent")
    tasks = MotorTasks(env, args.seed)
    tasks.task_ids = np.tile([0, 1, 2, 2], 8)
    tasks.reset(np.arange(32))
    initial = {k: np.tile(env.fields[k][:4], (8, 1)) for k in ("qpos", "qvel", "act", "ctrl")}
    command = np.tile(env.command[:4], (8, 1))
    requested = np.tile(env.requested_height_cm[:4], 8)
    heading = np.tile(tasks.heading[:4], 8)
    posture = GroundPosture(env, tasks.ground["qpos"])
    rows = [env.model.actuator(f"walker/wing_yaw_{side}").id for side in ("left", "right")]
    grid = [(g, b) for g in (0.85, 1.0, 1.15, 1.3) for b in (-0.02, 0.0, 0.02, 0.04)]
    mujoco.mj_saveModel(env.model, str(args.output / "model.mjb"))
    np.savez_compressed(
        args.output / "initial.npz", **initial, command=command, requested_height_cm=requested
    )
    synchronize(device)
    setup_seconds = time.perf_counter() - setup_start
    started_utc, started = utc_now(), time.perf_counter()
    results, traces, transitions = [], [], 0
    inference_steps = round(args.seconds / env.control_dt)
    for batch_index in range(2):
        candidates = grid[batch_index * 8 : (batch_index + 1) * 8]
        env.reset(np.arange(32), state=initial)
        env.command[:] = command
        env.requested_height_cm[:] = requested
        env.needs[:] = 0
        memory = actor.initial_state(32)
        hook = CandidateRows(actor, rows, np.repeat(candidates, 4, axis=0))
        records, metrics = [], []
        try:
            for step in range(inference_steps):
                observation = env.observation()
                prediction = actor(torch.as_tensor(observation, device=device), memory)
                memory = prediction.state
                action = prediction.action.cpu().numpy()
                if not np.isfinite(action).all() or np.abs(action).max() > 1:
                    raise RuntimeError("Invalid candidate command")
                records.append(
                    {
                        "qpos": env.fields["qpos"].copy(),
                        "qvel": env.fields["qvel"].copy(),
                        "activation": env.fields["act"].copy(),
                        "ctrl": env.fields["ctrl"].copy(),
                        "observation": observation.copy(),
                        "action": action.copy(),
                        "wing_activity": env.wing_forces.activity.copy(),
                        "wing_wrench": env.wing_forces.wrench.copy(),
                    }
                )
                env.step(action)
                transitions += 32
                target = initial["qpos"][:, :3].copy()
                distance = (step + 1) * env.control_dt * command[:, 0]
                target[:, 0] += distance * np.cos(heading)
                target[:, 1] += distance * np.sin(heading)
                velocity = env.velocity()
                metrics.append(
                    {
                        "root_error_cm": np.linalg.norm(
                            env.fields["qpos"][:, :3] - target, axis=1
                        ),
                        "height_cm": env.fields["qpos"][:, 2].copy(),
                        "upright": env.fields["xmat"][:, env.template.thorax_id, 8].copy(),
                        "failed": tasks.failed().copy(),
                        "speed_error": velocity[:, 3] - command[:, 0],
                        "yaw_error": velocity[:, 2].copy(),
                        "forbidden_load": env.forbidden_peak.copy() / env.body_weight,
                        **posture.measure(),
                    }
                )
        finally:
            hook.close()
        packed = {k: np.stack([r[k] for r in records]) for k in records[0]}
        measured = {k: np.stack([r[k] for r in metrics]) for k in metrics[0]}
        trace_path = args.output / f"batch_{batch_index}.npz"
        np.savez_compressed(
            trace_path,
            **packed,
            **{"metric_" + k: v for k, v in measured.items()},
            time=np.arange(inference_steps) * env.control_dt,
            candidates=np.asarray(candidates),
            task_ids=tasks.task_ids,
        )
        traces.append({"file": trace_path.name, "sha256": sha256(trace_path)})
        for i, (gain, bias) in enumerate(candidates):
            cases = []
            for j, name in enumerate(("stand", "walk", "hover_a", "hover_b")):
                ix = i * 4 + j
                rms = lambda key, m=measured, column=ix: float(
                    np.sqrt(np.square(m[key][:, column]).mean())
                )
                cases.append(
                    {
                        "case": name,
                        "stable": not bool(measured["failed"][:, ix].any()),
                        "failed_fraction": float(measured["failed"][:, ix].mean()),
                        "root_rmse_cm": rms("root_error_cm"),
                        "speed_rmse_cm_s": rms("speed_error"),
                        "yaw_rmse_rad_s": rms("yaw_error"),
                        "minimum_height_cm": float(measured["height_cm"][:, ix].min()),
                        "minimum_upright": float(measured["upright"][:, ix].min()),
                        "max_forbidden_load": float(measured["forbidden_load"][:, ix].max()),
                        "wing_rms_rad": float(
                            np.sqrt(measured["wings_angle_mse_rad2"][:, ix].mean())
                        ),
                        "wing_max_rad": float(measured["wing_max_deviation_rad"][:, ix].max()),
                    }
                )
            results.append({"gain": gain, "bias": bias, "cases": cases})
        print(
            json.dumps({"batch_completed": batch_index, "transitions": transitions}),
            flush=True,
        )
    selected, improved = select_candidate(results)
    state = calibrated_state(parent["state_dict"], rows, selected["gain"], selected["bias"])
    # All other body outputs and all upstream persistent state stay bitwise fixed.
    other = [i for i in range(actor.action_size) if i not in rows]
    for key, value in parent["state_dict"].items():
        if key in ("motor_decoder.3.weight", "motor_decoder.3.bias"):
            assert torch.equal(value[other], state[key][other])
        else:
            assert torch.equal(value, state[key])
    child = {
        k: parent[k]
        for k in (
            "observation_size",
            "sensor_extension_size",
            "action_size",
            "config",
            "graph_sha256",
            "graph_metadata_sha256",
            "physical_contract",
            "motor_only",
        )
    }
    child.update(
        state_dict=state,
        source_commit=provenance["source_commit"],
        parent_checkpoint_sha256=sha256(args.resume),
        method="physical search of existing sweep output weights",
        sweep_calibration={"gain": selected["gain"], "bias": selected["bias"]},
    )
    torch.save(child, args.output / "actor.pt")
    synchronize(device)
    report = {
        "provenance": provenance,
        "started_utc": started_utc,
        "completed_utc": utc_now(),
        "parent_checkpoint_sha256": sha256(args.resume),
        "checkpoint_sha256": sha256(args.output / "actor.pt"),
        "physical_contract": physical_contract(env.model),
        "model_sha256": sha256(args.output / "model.mjb"),
        "initial_sha256": sha256(args.output / "initial.npz"),
        "traces": traces,
        "method": child["method"],
        "searched_parameters": 2,
        "sweep_rows": rows,
        "all_other_state_unchanged": True,
        "runtime_hook": False,
        "teacher": False,
        "ground_retention_rule": "stable, permitted support; <=1.25x baseline RMS + tolerance; wing peak <= baseline +0.05rad",
        "hover_cost_rule": "mean over two airborne starts: root RMS cm + 2 * failed fraction",
        "selection_requires_relative_improvement": 0.05,
        "grid": results,
        "selected": selected,
        "improved_on_search": improved,
        "physical_acceptance": "Requires independent full-length evaluation; short search is not success",
        "seed": args.seed,
        "parallel_worlds": 32,
        "candidates_per_batch": 8,
        "cases_per_candidate": 4,
        "batches": 2,
        "control_hz": 500,
        "physics_hz": 5000,
        "seconds_per_case": inference_steps * env.control_dt,
        "physical_transitions": transitions,
        "aggregate_simulated_seconds": transitions * env.control_dt,
        "setup_seconds": setup_seconds,
        "search_and_save_seconds": time.perf_counter() - started,
        "neural_device": str(device),
        "physics_backend": "native CPU MuJoCo/mjbatch",
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device)
        if device.type == "cuda"
        else 0,
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "selected": selected,
                "improved_on_search": improved,
                "search_seconds": report["search_and_save_seconds"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("resume", "graph", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=2)
    parser.add_argument("--seed", type=int, default=96001)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    search(parser.parse_args())
