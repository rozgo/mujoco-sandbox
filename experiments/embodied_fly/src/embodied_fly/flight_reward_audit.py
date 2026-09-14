"""Audit goal preferences without running physics or updating an actor."""

import argparse
import json
from pathlib import Path

import numpy as np

from embodied_fly.flight_tracking_reward import (
    DEFAULT_TRACKING,
    desired_velocity,
    velocity_scores,
)
from embodied_fly.provenance import evidence, sha256, utc_now
from embodied_fly.round_trip_tasks import batched_target_motion


def rates(position, velocity, target, reference_velocity, upright, angular_speed, new):
    score = lambda x: 1 / np.sqrt(1 + np.asarray(x) ** 2)
    wanted = (
        desired_velocity(position, target, reference_velocity)
        if new
        else np.zeros_like(position)
    )
    return (
        0.5
        + 2 * score((position[..., 2] - target[..., 2]) / 0.1)
        + score(np.linalg.norm(position[..., :2] - target[..., :2], axis=-1) / 0.1)
        + sum(velocity_scores(velocity, wanted).values())
        + 0.5 * np.clip(upright, 0, 1)
        + 0.1 * score(angular_speed)
    )


def totals(
    position, velocity, target, reference_velocity, upright, angular_speed, new, bad=None
):
    reward = (
        rates(position, velocity, target, reference_velocity, upright, angular_speed, new)
        * 0.002
    )
    if bad is not None:
        reward = np.where(bad, -1, reward)
        terminated = np.cumsum(bad, axis=0) > 0
        previously_terminated = np.concatenate(
            (np.zeros_like(terminated[:1]), terminated[:-1]), axis=0
        )
        reward = np.where(previously_terminated, 0, reward)
    return reward.sum(axis=0)


def schedule(times, origins, amplitude=0.15, time_scale=1.0):
    n = len(origins)
    target, velocity = batched_target_motion(
        np.repeat(times, n),
        np.tile(origins, (len(times), 1)),
        np.tile(np.arange(n), len(times)),
        np.full(len(times) * n, amplitude),
        time_scale,
    )
    return target.reshape(-1, n, 3), velocity.reshape(-1, n, 3)


def physical_capture(source):
    report = json.loads((source / "report.json").read_text())
    assert sha256(source / "capture.npz") == report["capture_sha256"]
    with np.load(source / "capture.npz") as z:
        # Last step has no following pre-step angular velocity in the archive.
        times = z["time"][:-1] + 0.002
        position = z["actual_position"][:-1]
        velocity = z["actual_velocity"][:-1]
        up = z["upright"][:-1]
        forbidden = z["forbidden"][:-1]
        angular = np.linalg.norm(z["qvel"][1:, :, 3:6], axis=-1)
    target, reference = schedule(times, np.asarray(report["origins_cm"]))
    bad = (position[..., 2] < 0.5) | (up < 0.5) | (forbidden > 0.1)
    result = {
        name: totals(position, velocity, target, reference, up, angular, new, bad).tolist()
        for name, new in [("old", False), ("new", True)]
    }
    return {
        "capture_sha256": report["capture_sha256"],
        "source_model_sha256": report["model_sha256"],
        "seconds_scored": len(times) * 0.002,
        "returns_excluding_effort": result,
        "failure_worlds": int(bad.any(axis=0).sum()),
    }


def synthetic_case(amplitude, scale):
    times = np.arange(1, 6001) * 0.002
    origins = np.tile([0.0, 0.0, 1.86651647], (7, 1))
    target, reference = schedule(times, origins, amplitude, scale)
    still = np.broadcast_to(origins, target.shape).copy()
    velocity = np.zeros_like(target)
    drift = still.copy()
    drift[..., 0] += times[:, None] * 0.8
    drift_velocity = velocity.copy()
    drift_velocity[..., 0] = 0.8
    phase = times[:, None] * 2 * np.pi
    oscillation = target.copy()
    oscillation[..., 1] += amplitude * np.sin(phase)
    oscillation_velocity = reference.copy()
    oscillation_velocity[..., 1] += amplitude * 2 * np.pi * np.cos(phase)
    no_return, no_return_velocity = schedule(
        np.minimum(times, 3 * scale), origins, amplitude, scale
    )
    no_return_velocity[times >= 3 * scale] = 0
    falling = still.copy()
    falling[..., 2] -= times[:, None] * 0.4
    falling_velocity = velocity.copy()
    falling_velocity[..., 2] = -0.4
    fixtures = {
        "tracking": (target, reference),
        "stationary": (still, velocity),
        "drift": (drift, drift_velocity),
        "oscillation": (oscillation, oscillation_velocity),
        "no_return": (no_return, no_return_velocity),
        "falling": (falling, falling_velocity),
    }
    results = {}
    for name, (p, v) in fixtures.items():
        bad = p[..., 2] < 0.5
        results[name] = {
            label: totals(
                p,
                v,
                target,
                reference,
                np.ones(p.shape[:-1]),
                np.zeros(p.shape[:-1]),
                new,
                bad,
            ).tolist()
            for label, new in [("old", False), ("new", True)]
        }
    margins = {
        name: (
            np.asarray(results["tracking"]["new"])[:6] - np.asarray(result["new"])[:6]
        ).tolist()
        for name, result in results.items()
        if name != "tracking"
    }
    return {
        "amplitude_mm": amplitude * 10,
        "time_scale": scale,
        "returns_excluding_effort": results,
        "tracking_advantage_by_case": margins,
        "passed": all(min(values) > 0 for values in margins.values()),
    }


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    synthetic = [synthetic_case(a, t) for a in (0.12, 0.15, 0.18) for t in (0.85, 1, 1.15)]
    captures = {
        name: physical_capture(Path(path))
        for name, path in {
            "pid": "outputs/embodied_fly/round_trip_reference_02",
            "parent": "outputs/embodied_fly/round_trip_parent_review_01",
            "child": "outputs/embodied_fly/round_trip_ppo_review_01",
        }.items()
    }
    assert len({r["source_model_sha256"] for r in captures.values()}) == 1
    pid = np.asarray(captures["pid"]["returns_excluding_effort"]["new"])
    margins = {
        name: (pid - np.asarray(captures[name]["returns_excluding_effort"]["new"])).tolist()
        for name in ("parent", "child")
    }
    report = {
        "provenance": evidence(),
        "completed_utc": utc_now(),
        "configuration": vars(DEFAULT_TRACKING),
        "synthetic_cases": synthetic,
        "physical_captures": captures,
        "pid_margin_over_recorded_learned_cases": margins,
        "passed": all(x["passed"] for x in synthetic)
        and all(min(x) > 0.048 for x in margins.values()),
        "scope": "Reward preference audit only. Synthetic fixtures are kinematic, not physically simulated. Recorded physical captures retain their actual states. No neural training or new physics.",
        "omitted_term": "Actuator effort is not recorded in these captures; its total magnitude is bounded by 0.024 per 12 s. The physical ranking gate uses a conservative 0.048 margin.",
        "simulation_steps": 0,
        "actor_updates": 0,
        "critic_updates": 0,
        "caveat": "These rankings do not prove policy learnability, local exploration quality or absence of all reward exploits.",
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "passed": report["passed"],
                "synthetic_configurations": len(synthetic),
                "minimum_tracking_advantage": min(
                    min(m) for x in synthetic for m in x["tracking_advantage_by_case"].values()
                ),
                "pid_margins": margins,
            },
            indent=2,
        )
    )
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    result = run(p.parse_args())
    raise SystemExit(0 if result["passed"] else 2)
