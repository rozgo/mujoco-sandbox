"""Fit existing wing rows from canonical motor histories and sensory probes.

Frozen graph features are the only decoder inputs. Counterfactual observations
are training examples, never physical transitions or a deployed controller.
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
from embodied_fly.motor_focus import TASKS, MotorTasks
from embodied_fly.motor_response import perturb_feedback
from embodied_fly.physical_contract import physical_contract
from embodied_fly.provenance import evidence, sha256, utc_now
from embodied_fly.train import synchronize
from embodied_fly.wing_motion import CONFIG
from embodied_fly.wing_readout import replace_wing_rows


def split_worlds(task_ids):
    """Hold out the last world of each task, including all its perturbations."""
    groups = [np.flatnonzero(task_ids == task) for task in range(3)]
    if any(len(ids) < 2 for ids in groups):
        raise ValueError("Need at least two worlds per task")
    valid = np.array([ids[-1] for ids in groups])
    train = np.array([i for i in range(len(task_ids)) if i not in valid])
    return train, valid


def variant_targets(q, v, qref, angle_delta, speed_delta):
    angle = np.repeat(q[:, None], 25, axis=1)
    speed = np.repeat(v[:, None], 25, axis=1)
    for i in range(6):
        angle[:, 1 + 2 * i, i] += angle_delta
        angle[:, 2 + 2 * i, i] -= angle_delta
        speed[:, 13 + 2 * i, i] += speed_delta
        speed[:, 14 + 2 * i, i] -= speed_delta
    return np.clip(
        (GroundPosture.wing_kp * (qref - angle) - GroundPosture.wing_kd * speed)
        / CONFIG.joint_torque_limit,
        -1,
        1,
    ).astype(np.float32)


@torch.no_grad()
def collect(args):
    if (
        args.worlds < 6
        or args.stride < 1
        or args.threads < 1
        or not np.isfinite([args.seconds, args.angle_delta, args.speed_delta]).all()
        or min(args.seconds, args.angle_delta, args.speed_delta) <= 0
    ):
        raise ValueError("Positive collection settings and six worlds required")
    args.output.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    provenance = evidence()
    torch.set_num_threads(4)
    device = torch.device(args.device)
    actor, checkpoint = load_actor(args.checkpoint, args.graph, device)
    actor.eval()
    env = FlyBatch(args.worlds, args.threads, 14, preset="wing_motion")
    contract = physical_contract(env.model)
    if not actor.motor_only or actor.observation_size != 397:
        raise ValueError("Requires canonical motor-only actor")
    if checkpoint.get("physical_contract") != contract:
        raise ValueError("Checkpoint and collection body must match exactly")
    tasks = MotorTasks(env, args.seed)
    train, valid = split_worlds(tasks.task_ids)
    ground = np.flatnonzero(tasks.task_ids != 2)
    air = np.flatnonzero(tasks.task_ids == 2)
    wings = np.array(
        [env.model.actuator(env.model.joint(j).name).id for j in env.template.wing_joint_ids]
    )
    qref = tasks.ground["qpos"][env.template.wing_angle_indices]
    mujoco.mj_saveModel(env.model, str(args.output / "model.mjb"))
    memory = actor.initial_state(env.n)
    rows, traces = [], []
    first_failure = np.full(env.n, -1, np.int64)
    synchronize(device)
    setup = time.perf_counter() - start
    start = time.perf_counter()
    for frame in range(round(args.seconds / env.control_dt)):
        obs = env.observation()
        q, v = (env.fields[k].copy() for k in ("qpos", "qvel"))
        result = actor(torch.as_tensor(obs, device=device), memory)
        action = result.action.cpu().numpy()
        if frame % args.stride == 0:
            variants = perturb_feedback(
                env.model,
                obs[ground],
                q[ground],
                v[ground],
                args.angle_delta,
                args.speed_delta,
            )
            inputs = np.concatenate((variants.reshape(-1, 397), obs[air]))
            worlds = np.concatenate((np.repeat(ground, 25), air))
            variant = np.concatenate(
                (np.tile(np.arange(25), len(ground)), np.zeros(len(air), int))
            )
            branch = actor(torch.as_tensor(inputs, device=device), memory[:, worlds])
            hidden = actor.motor_decoder[:3](branch.state[actor.motor_ids].T)
            target = np.concatenate(
                (
                    variant_targets(
                        q[ground][:, env.template.wing_angle_indices],
                        v[ground][:, env.template.wing_velocity_indices],
                        qref,
                        args.angle_delta,
                        args.speed_delta,
                    ).reshape(-1, 6),
                    action[air][:, wings],
                )
            )
            rows.append(
                {
                    "hidden": hidden.cpu().numpy(),
                    "target": target,
                    "world": worlds,
                    "variant": variant,
                    "task": tasks.task_ids[worlds],
                    "frame": np.full(len(worlds), frame),
                }
            )
        traces.append(
            {
                "observation": obs,
                "qpos": q,
                "qvel": v,
                "action": action,
                "activation": env.fields["act"].copy(),
                "ctrl": env.fields["ctrl"].copy(),
                "wing_activity": env.wing_forces.activity.copy(),
                "wing_wrench": env.wing_forces.wrench.copy(),
            }
        )
        memory = result.state
        env.step(action)
        failed = tasks.failed() & (first_failure < 0)
        first_failure[failed] = frame + 1
    synchronize(device)
    collection_seconds = time.perf_counter() - start
    packed = {k: np.concatenate([r[k] for r in rows]) for k in rows[0]}
    capture = {k: np.stack([r[k] for r in traces]) for k in traces[0]}
    for arrays in (packed, capture):
        if not all(np.isfinite(x).all() for x in arrays.values()):
            raise RuntimeError("Nonfinite readout corpus")
    np.testing.assert_array_equal(
        capture["observation"][1:, :, 297:375], capture["action"][:-1]
    )
    np.savez_compressed(
        args.output / "features.npz",
        **packed,
        training_worlds=train,
        validation_worlds=valid,
        wing_channels=wings,
    )
    np.savez_compressed(
        args.output / "capture.npz",
        **capture,
        task=tasks.task_ids,
        first_failure_frame=first_failure,
        initial_qpos=tasks.ground["qpos"],
    )
    report = {
        "schema": "canonical-wing-readout-features-v1",
        "provenance": provenance,
        "completed_utc": utc_now(),
        "checkpoint_sha256": sha256(args.checkpoint),
        "physical_contract": contract,
        "model_sha256": sha256(args.output / "model.mjb"),
        "cache_sha256": sha256(args.output / "features.npz"),
        "capture_sha256": sha256(args.output / "capture.npz"),
        "config": {k: v for k, v in vars(args).items() if not isinstance(v, Path)},
        "setup_seconds": setup,
        "collection_seconds": collection_seconds,
        "physical_transitions": len(traces) * env.n,
        "aggregate_simulated_seconds": len(traces) * env.n * env.control_dt,
        "physics_worlds": env.n,
        "physics_hz": 5000,
        "control_hz": 500,
        "physics_backend": "native CPU MuJoCo/mjbatch",
        "brain_device": str(device),
        "feature_examples": len(packed["world"]),
        "synthetic_examples": int(np.count_nonzero(packed["variant"])),
        "training_worlds": train.tolist(),
        "validation_worlds": valid.tolist(),
        "first_failure_frame": first_failure.tolist(),
        "warning_count": int(env.fields["warning"].sum()),
        "training_updates": 0,
        "ground_target": "bounded resting-wing feedback torque",
        "hover_target": "retain parent commands on captured hover histories",
        "feature_source": "existing 256-unit motor hidden layer after fixed MaleCNS activity",
        "execution": "parent actor only, no resets or teacher assistance; all failures retained",
        "limitations": "Perturbed sensory examples do not advance physics. Offline fit is not motor acceptance.",
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def balanced_weights(task, variant):
    weights = torch.zeros(len(task), dtype=torch.float64, device=task.device)
    for group in torch.unique(task):
        ids = task == group
        kinds = torch.unique(variant[ids] == 0)
        for nominal in kinds:
            mask = ids & ((variant == 0) == nominal)
            weights[mask] = 1 / (len(torch.unique(task)) * len(kinds) * int(mask.sum()))
    return weights


def ridge_delta(x, y, weights, parent, alpha):
    """Fit in existing hidden coordinates; regularize changes from parent rows."""
    if not np.isfinite(alpha) or alpha <= 0:
        raise ValueError("Positive finite ridge penalty required")
    x, y, parent = x.double(), y.double(), parent.double()
    residual = torch.atanh(y.clamp(-0.999, 0.999)) - x @ parent
    gram = x.T @ (weights[:, None] * x)
    gram += alpha * torch.eye(x.shape[1], device=x.device, dtype=x.dtype)
    return parent + torch.linalg.solve(gram, x.T @ (weights[:, None] * residual))


def calibrated_checkpoint(parent, weights, bias, wings):
    # Keep motor context, physical identity and observation schema; omit stale Adam state.
    result = {k: v for k, v in parent.items() if k != "optimizer_state_dict"}
    result["state_dict"] = replace_wing_rows(parent["state_dict"], weights, bias, wings)
    return result


def fit(args):
    if not args.alphas or not np.isfinite(args.alphas).all() or min(args.alphas) <= 0:
        raise ValueError("At least one positive finite ridge penalty required")
    args.output.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    provenance = evidence()
    torch.set_num_threads(4)
    device = torch.device(args.device)
    parent = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    source = json.loads((args.cache / "report.json").read_text())
    if (
        source["schema"] != "canonical-wing-readout-features-v1"
        or source["checkpoint_sha256"] != sha256(args.checkpoint)
        or source["cache_sha256"] != sha256(args.cache / "features.npz")
        or source["physical_contract"] != parent.get("physical_contract")
        or not parent.get("motor_only")
    ):
        raise ValueError("Canonical feature cache/checkpoint identity mismatch")
    with np.load(args.cache / "features.npz") as c:
        train, valid = c["training_worlds"], c["validation_worlds"]
        if set(train) & set(valid) or set(c["world"]) != set(train) | set(valid):
            raise ValueError("Invalid whole-world split")
        wings = c["wing_channels"].copy()
        data = []
        for ids in (train, valid):
            mask = np.isin(c["world"], ids)
            x = torch.as_tensor(c["hidden"][mask], device=device, dtype=torch.float64)
            x = torch.cat((x, torch.ones_like(x[:, :1])), 1)
            y = torch.as_tensor(c["target"][mask], device=device, dtype=torch.float64)
            task, variant = [
                torch.as_tensor(c[k][mask], device=device) for k in ("task", "variant")
            ]
            data.append((x, y, balanced_weights(task, variant), task, variant))
    state = parent["state_dict"]
    base = (
        torch.cat(
            (
                state["motor_decoder.3.weight"][wings].T,
                state["motor_decoder.3.bias"][wings][None],
            ),
            0,
        )
        .to(device)
        .double()
    )

    def metrics(coef, d):
        x, y, weight, task, variant = d
        mse = ((x @ coef).tanh() - y).square().mean(1)
        return {
            "balanced_mse": float(mse @ weight),
            "by_task": {TASKS[i]: float(mse[task == i].mean()) for i in range(3)},
            "nominal_by_task": {
                TASKS[i]: float(mse[(task == i) & (variant == 0)].mean()) for i in range(3)
            },
        }

    initial = metrics(base, data[1])
    synchronize(device)
    setup = time.perf_counter() - start
    start = time.perf_counter()
    candidates = []
    best, best_score, selected = None, float("inf"), None
    for alpha in args.alphas:
        coef = ridge_delta(*data[0][:3], base, alpha)
        valid_metrics = metrics(coef, data[1])
        candidates.append(
            {
                "alpha": alpha,
                "training": metrics(coef, data[0]),
                "validation": valid_metrics,
                "row_change_l2": float((coef - base).norm()),
            }
        )
        if valid_metrics["balanced_mse"] < best_score:
            best, best_score, selected = coef, valid_metrics["balanced_mse"], alpha
    synchronize(device)
    elapsed = time.perf_counter() - start
    if best is None or not torch.isfinite(best).all():
        raise RuntimeError("No finite fitted wing rows")
    fitted = calibrated_checkpoint(parent, best[:-1].T.float(), best[-1].float(), wings)
    fitted.update(
        source_commit=provenance["source_commit"],
        parent_checkpoint_sha256=sha256(args.checkpoint),
        method="frozen MaleCNS features; ridge calibration of six existing wing output rows",
        readout_calibration={"alpha": selected, "cache_sha256": source["cache_sha256"]},
    )
    torch.save(fitted, args.output / "actor.pt")
    report = {
        "provenance": provenance,
        "completed_utc": utc_now(),
        "method": fitted["method"],
        "parent_checkpoint_sha256": sha256(args.checkpoint),
        "checkpoint_sha256": sha256(args.output / "actor.pt"),
        "cache_report_sha256": sha256(args.cache / "report.json"),
        "cache_sha256": source["cache_sha256"],
        "physical_contract": parent["physical_contract"],
        "setup_seconds": setup,
        "fitting_seconds": elapsed,
        "device": str(device),
        "train_examples": len(data[0][0]),
        "validation_examples": len(data[1][0]),
        "physical_transitions": 0,
        "new_deployed_parameters": 0,
        "trained_existing_parameters": 1542,
        "selected_alpha": selected,
        "initial_validation": initial,
        "candidates": candidates,
        "scope": "Six existing output rows only; all upstream and non-wing parameters unchanged. Physical review required.",
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=("collect", "fit"))
    for name in ("checkpoint", "graph", "cache", "output"):
        p.add_argument("--" + name, type=Path, required=name in ("checkpoint", "output"))
    p.add_argument("--device", default="cuda")
    p.add_argument("--worlds", type=int, default=32)
    p.add_argument("--threads", type=int, default=16)
    p.add_argument("--seconds", type=float, default=2)
    p.add_argument("--seed", type=int, default=73009)
    p.add_argument("--stride", type=int, default=10)
    p.add_argument("--angle-delta", type=float, default=0.2)
    p.add_argument("--speed-delta", type=float, default=5)
    p.add_argument("--alphas", type=float, nargs="+", default=[1, 0.1, 0.01, 0.001, 0.0001])
    args = p.parse_args()
    if args.mode == "collect" and args.graph is None:
        p.error("Collection requires --graph")
    if args.mode == "fit" and args.cache is None:
        p.error("Fitting requires --cache")
    report = collect(args) if args.mode == "collect" else fit(args)
    print(json.dumps({k: v for k, v in report.items() if k != "provenance"}), flush=True)
