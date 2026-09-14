"""Local decoder-parameter probes through full MaleCNS and unmodified flight physics."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from embodied_fly.evaluate import load_actor
from embodied_fly.physical_contract import physical_contract
from embodied_fly.provenance import evidence, sha256, utc_now
from embodied_fly.recovery_starts import capture, restore
from embodied_fly.train import synchronize
from embodied_fly.velocity_demonstrations import environment
from embodied_fly.velocity_hover import failures, reward_from_recipe, start_states
from embodied_fly.velocity_motor import observation


def parameter_variants(delta):
    if not np.isfinite(delta) or not 0 < delta <= 0.01:
        raise ValueError("Small finite decoder perturbation required")
    offsets = np.zeros((16, 7), np.float32)
    names = ["unchanged"]
    for axis in range(7):
        offsets[1 + 2 * axis, axis] = delta
        offsets[2 + 2 * axis, axis] = -delta
        name = f"bias_{axis}" if axis < 6 else "collective_sweep_gain"
        names.extend([f"{name}_plus", f"{name}_minus"])
    names.append("unchanged_duplicate")
    return offsets, names


def varied_linear_output(layer, inputs, output, changes):
    """Equivalent to copied final-layer biases and sweep-row weights per world.

    Only the existing last six-output linear layer changes. The full upstream
    brain and remaining action logits still execute normally in every world.
    """
    if changes.shape != (len(output), 7):
        raise ValueError("One seven-parameter probe per world required")
    gain = output.new_zeros(output.shape)
    gain[:, [0, 3]] = changes[:, 6, None]
    return output + changes[:, :6] + gain * (output - layer.bias)


def response_jacobian(values, delta):
    return np.stack(
        [(values[1 + 2 * i] - values[2 + 2 * i]) / (2 * delta) for i in range(7)], axis=1
    )


@torch.no_grad()
def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    begin = time.perf_counter()
    provenance = evidence()
    torch.set_num_threads(4)
    device = torch.device(args.device)
    actor, checkpoint = load_actor(args.checkpoint, args.graph, device)
    actor.eval()
    if actor.wing_residual is None or not hasattr(actor.wing_residual, "network"):
        raise ValueError("Requires the preserved nonlinear wing readout")
    layer = actor.wing_residual.network[-1]
    before = {k: v.detach().clone() for k, v in actor.named_parameters()}
    env = environment(32, 16)
    if physical_contract(env.model) != checkpoint["physical_contract"]:
        raise ValueError("Diagnostic physical plant differs")
    bank_report = json.loads((args.bank / "report.json").read_text())
    if not bank_report["passed"] or bank_report["parent_checkpoint_sha256"] != sha256(
        args.checkpoint
    ):
        raise ValueError("Use the exact parent's audited recovery histories")
    if sha256(args.bank / "states.npz") != bank_report["states_sha256"]:
        raise ValueError("Recovery bank checksum differs")
    with np.load(args.bank / "states.npz") as data:
        bank = {k: data[k].copy() for k in data.files}
    reward = reward_from_recipe(env, bank_report["reward_recipe"])
    commands = np.zeros((32, 4), np.float32)
    offsets, names = parameter_variants(args.delta)
    changes = torch.as_tensor(np.tile(offsets, (2, 1)), device=device)
    memory = actor.initial_state(32)
    env.reset(np.arange(32), state=start_states(args.dataset, [0] * 32))
    conditions = []

    def save_condition(name):
        snap = capture(env, reward, memory)
        conditions.append((name, {k: v[:1].copy() for k, v in snap.items()}))

    # Build cold histories with the unchanged actor, not a teacher or pose replay.
    for step in range(301):
        if step in (0, 100, 300):
            save_condition(f"cold_{step * 0.002:g}s")
        if step == 300:
            break
        result = actor(torch.as_tensor(observation(env, commands), device=device), memory)
        memory = result.state
        env.step(result.action.cpu().numpy())
        reward()
    for age in (2, 4, 6):
        record = next(
            i
            for i, r in enumerate(bank_report["records"])
            if r["episode"] == 0 and r["parent_flight_seconds"] == age
        )
        saved = {k: np.repeat(v[record : record + 1], 32, axis=0) for k, v in bank.items()}
        restore(env, reward, memory, np.arange(32), saved)
        save_condition(f"warm_{age}s")
        if age == 2:
            for step in range(12):
                result = actor(
                    torch.as_tensor(observation(env, commands), device=device), memory
                )
                memory = result.state
                env.step(result.action.cpu().numpy())
                reward()
                if step + 1 in (4, 8):
                    save_condition(f"warm_{age + (step + 1) * 0.002:g}s")
    if len(conditions) != 8:
        raise RuntimeError("Expected eight cold/warm phase histories")
    synchronize(device)
    setup_seconds = time.perf_counter() - begin
    begin = time.perf_counter()
    original_force = env.wing_forces.advance
    force_samples = []

    def observe_force(*values):
        wrench = original_force(*values)
        force_samples.append(
            np.column_stack(
                (wrench[:, :2] / env.body_weight, env.wing_forces.lift / env.body_weight)
            )
        )
        return wrench

    env.wing_forces.advance = observe_force
    records = []
    steps = round(args.seconds / env.control_dt)
    for pair in range(0, len(conditions), 2):
        force_samples.clear()
        selected = conditions[pair : pair + 2]
        saved = {
            k: np.concatenate([np.repeat(s[k], 16, axis=0) for _, s in selected])
            for k in selected[0][1]
        }
        restore(env, reward, memory, np.arange(32), saved)
        initial_position = env.fields["qpos"][:, :3].copy()
        initial_velocity = env.fields["qvel"][:, :3].copy()
        first_failure = np.full(32, np.nan)
        minimum_height = env.fields["qpos"][:, 2].copy()
        minimum_up = env.fields["xmat"][:, env.template.thorax_id, 8].copy()
        trace, action_trace = [], []
        hook = layer.register_forward_hook(
            lambda m, x, y: varied_linear_output(m, x, y, changes)
        )
        try:
            for step in range(steps):
                result = actor(
                    torch.as_tensor(observation(env, commands), device=device), memory
                )
                memory = result.state
                action = result.action.cpu().numpy()
                action_trace.append(action[:, 14:20].copy())
                env.step(action)
                reward()
                trace.append(env.fields["qpos"][:, :3].copy())
                newly = failures(env) & np.isnan(first_failure)
                first_failure[newly] = (step + 1) * env.control_dt
                np.minimum(minimum_height, env.fields["qpos"][:, 2], out=minimum_height)
                np.minimum(
                    minimum_up,
                    env.fields["xmat"][:, env.template.thorax_id, 8],
                    out=minimum_up,
                )
        finally:
            hook.remove()
        if len(force_samples) != 2 * steps:
            raise RuntimeError("Force observer must capture every physical tick")
        forces = np.stack(force_samples)
        positions = np.stack(trace)
        actions = np.stack(action_trace)
        mean_force = forces.mean(axis=0)
        velocity_change = (env.fields["qvel"][:, :3] - initial_velocity) * 10
        for group, (name, _) in enumerate(selected):
            s = slice(group * 16, (group + 1) * 16)
            f, v = mean_force[s], velocity_change[s]
            j = response_jacobian(f, args.delta)
            record = {
                "condition": name,
                "force_columns": ["world_fx_weight", "world_fy_weight", "raw_lift_weight"],
                "initial_position_cm": initial_position[s][0].tolist(),
                "initial_velocity_mm_s": (initial_velocity[s][0] * 10).tolist(),
                "mean_forces": f.tolist(),
                "velocity_change_mm_s": v.tolist(),
                "force_jacobian_per_parameter_unit": j.tolist(),
                "force_jacobian_singular_values": np.linalg.svd(j, compute_uv=False).tolist(),
                "velocity_jacobian_mm_s_per_parameter_unit": response_jacobian(
                    v, args.delta
                ).tolist(),
                "minimum_heights_mm": (minimum_height[s] * 10).tolist(),
                "minimum_upright": minimum_up[s].tolist(),
                "first_failure_seconds": [
                    None if np.isnan(x) else float(x) for x in first_failure[s]
                ],
                "duplicate_position_error_cm": float(
                    np.max(np.abs(positions[:, s][:, 0] - positions[:, s][:, 15]))
                ),
                "duplicate_action_error": float(
                    np.max(np.abs(actions[:, s][:, 0] - actions[:, s][:, 15]))
                ),
            }
            records.append(record)
        np.savez_compressed(
            args.output / f"pair_{pair // 2:02d}.npz",
            force=forces,
            position=positions,
            action=actions,
            velocity_change=velocity_change,
        )
        print(
            json.dumps(
                {
                    "completed_conditions": [name for name, _ in selected],
                    "records": records[-2:],
                }
            ),
            flush=True,
        )
    synchronize(device)
    probe_seconds = time.perf_counter() - begin
    unchanged = all(torch.equal(v, before[k]) for k, v in actor.named_parameters())
    passed = unchanged and all(
        r["duplicate_position_error_cm"] <= 1e-4 and r["duplicate_action_error"] <= 1e-4
        for r in records
    )
    report = {
        "provenance": provenance,
        "completed_utc": utc_now(),
        "checkpoint_sha256": sha256(args.checkpoint),
        "bank_sha256": bank_report["states_sha256"],
        "physical_contract": checkpoint["physical_contract"],
        "setup_and_history_seconds": setup_seconds,
        "probe_seconds": probe_seconds,
        "worlds": 32,
        "physics_threads": 16,
        "seconds_per_probe": args.seconds,
        "history_collection_transitions": (300 + 12) * 32,
        "probe_transitions": 4 * 32 * steps,
        "parameter_delta": args.delta,
        "variants": names,
        "parameter_probe_matrix": offsets.tolist(),
        "records": records,
        "actor_parameters_unchanged": unchanged,
        "passed": passed,
        "scope": "Diagnostic parameter copies of the same decoder; full connectome live, original physics, no optimization, no new deployed parameter, no external force assistance. Eight histories from training episode zero only; not a broad capability evaluation.",
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    if not passed:
        raise RuntimeError("Unchanged controls or actor preservation check failed")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--graph", type=Path, required=True)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--bank", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--delta", type=float, default=0.001)
    p.add_argument("--seconds", type=float, default=0.128)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    if not np.isfinite(args.seconds) or not 0.02 <= args.seconds <= 0.5:
        raise ValueError("Use a short local probe between 20 and 500 ms")
    run(args)
