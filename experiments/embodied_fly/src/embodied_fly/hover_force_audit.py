"""Observe every physics-tick wing force while replaying recorded flights exactly."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from embodied_fly.physical_contract import physical_contract
from embodied_fly.provenance import evidence, sha256, utc_now
from embodied_fly.recovery_starts import restore
from embodied_fly.velocity_demonstrations import environment
from embodied_fly.velocity_hover import reward_from_recipe


def inspect_capture(folder, bank=None):
    report = json.loads((folder / "report.json").read_text())
    cases = report["cases"]
    captures = []
    for case in cases:
        file = folder / case["file"]
        if sha256(file) != case["sha256"]:
            raise ValueError("Capture checksum differs")
        with np.load(file) as data:
            captures.append(
                {
                    k: data[k].copy()
                    for k in ("qpos", "qvel", "act", "ctrl", "action", "post_position")
                }
            )
    env = environment(len(cases), len(cases))
    if physical_contract(env.model) != report["physical_contract"]:
        raise ValueError("Physical force audit must use the matching plant")
    if bank is None:
        env.reset(
            np.arange(env.n),
            state={
                k: np.stack([c[k][0] for c in captures])
                for k in ("qpos", "qvel", "act", "ctrl")
            },
        )
    else:
        bank_report = json.loads((bank / "report.json").read_text())
        if (
            sha256(bank / "states.npz") != bank_report["states_sha256"]
            or report["bank_sha256"] != bank_report["states_sha256"]
        ):
            raise ValueError("Recovery bank differs")
        with np.load(bank / "states.npz") as data:
            selected = np.array([c["record"] for c in cases])
            saved = {k: data[k][selected].copy() for k in data.files}
        score = reward_from_recipe(env, bank_report["reward_recipe"])
        memory = torch.zeros(saved["memory"].shape[1], env.n)
        restore(env, score, memory, np.arange(env.n), saved)

    samples = []
    original = env.wing_forces.advance

    def observed(*args):
        wrench = original(*args)
        # Copy only after the original force function. Return its identical result.
        samples.append(
            np.stack(
                (
                    env.wing_forces.lift / env.body_weight,
                    wrench[:, 2] / env.body_weight,
                    np.abs(np.asarray(args[1]).reshape(env.n, 2, 3)[:, :, 0]).mean(axis=1),
                ),
                axis=1,
            )
        )
        return wrench

    env.wing_forces.advance = observed
    stops = [
        round(c["first_failure_seconds"] * 500)
        if c["first_failure_seconds"] is not None
        else len(a["action"])
        for c, a in zip(cases, captures, strict=True)
    ]
    steps = min(stops)  # Strict common interval; never include post-failure forces.
    error = 0.0
    for step in range(steps):
        if bank is None:
            env.batch.forward()
        env.step(np.stack([a["action"][step] for a in captures]))
        error = max(
            error,
            float(
                np.max(
                    np.abs(
                        env.fields["qpos"][:, :3]
                        - np.stack([a["post_position"][step] for a in captures])
                    )
                )
            ),
        )
    samples = np.stack(samples)
    if len(samples) != 2 * steps:
        raise RuntimeError("Expected both 1 kHz force evaluations per action")
    windows = {}
    for start, end in ((0, 0.2), (0.2, 0.6), (0.6, 0.796), (2, 3), (2, 10)):
        if end > steps * 0.002:
            continue
        mean = samples[round(start * 1000) : round(end * 1000)].mean(axis=0)
        windows[f"{start:g}_{end:g}_seconds"] = {
            "mean_lift_body_weights": float(mean[:, 0].mean()),
            "mean_upward_wrench_body_weights": float(mean[:, 1].mean()),
            "mean_absolute_sweep_speed_rad_s": float(mean[:, 2].mean()),
            "per_case": mean.tolist(),
        }
    return {
        "capture_report_sha256": sha256(folder / "report.json"),
        "common_interval_seconds": steps * 0.002,
        "physics_ticks_per_world": len(samples),
        "worlds": env.n,
        "maximum_body_position_replay_error_cm": error,
        "windows": windows,
        "passed": error <= 1e-9,
    }


def run(args):
    if args.output.exists():
        raise FileExistsError("Preserve prior force audits")
    begin = time.perf_counter()
    captures = {}
    for label in ("parent", "midpoint", "final"):
        captures[f"cold_{label}"] = inspect_capture(args.run / label)
        captures[f"recovery_{label}"] = inspect_capture(args.recoveries / label, args.bank)
        print(
            json.dumps(
                {
                    "completed": label,
                    "cold": captures[f"cold_{label}"],
                    "recovery": captures[f"recovery_{label}"],
                }
            ),
            flush=True,
        )
    result = {
        "provenance": evidence(),
        "completed_utc": utc_now(),
        "wall_seconds": time.perf_counter() - begin,
        "captures": captures,
        "passed": all(c["passed"] for c in captures.values()),
        "scope": "Exact recorded-action physics replay; read-only force observer at all 1 kHz ticks, no neural inference, updates, additional force or post-failure samples. Raw lift differs from net upward wrench, which includes body response and drag.",
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    if not result["passed"]:
        raise RuntimeError("Force-observed replay differs from the recorded flight")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--recoveries", type=Path, required=True)
    p.add_argument("--bank", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    run(p.parse_args())
