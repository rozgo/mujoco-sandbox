"""Gate a coordinated hover objective using recorded flights and counterexamples."""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from embodied_fly.hover_reward_replay import audit
from embodied_fly.provenance import evidence, utc_now
from embodied_fly.velocity_hover import reward_rates


def counterexamples(scale=0.5):
    """Synthetic physical outcomes, not executable controllers or flight evidence."""
    dt, horizon = 0.002, 5000
    gamma = np.exp(-dt / 2)

    def score(velocity, fail_at=None):
        velocity = np.broadcast_to(velocity, (horizon, 3)).copy()
        # The same causal 100 ms window, including its shorter initial history.
        cumulative = np.vstack([np.zeros(3), velocity.cumsum(axis=0)])
        end = np.arange(horizon) + 1
        start = np.maximum(end - 50, 0)
        mean = (cumulative[end] - cumulative[start]) / (end - start)[:, None]
        rates = sum(
            reward_rates(
                mean,
                np.zeros_like(mean),
                np.ones(horizon),
                horizontal_scale=scale,
                velocity_objective="vector",
            ).values()
        )
        rewards = rates * dt
        if fail_at is not None:
            rewards[fail_at] = -2
            rewards = rewards[: fail_at + 1]  # No post-failure bonus or padding.
        return {
            "return": float(rewards.sum()),
            "discounted_return": float(np.sum(rewards * gamma ** np.arange(len(rewards)))),
            "duration_seconds": len(rewards) * dt,
        }

    recovery = np.zeros((horizon, 3))
    recovery[:500, 0] = np.linspace(3, 0, 500)
    examples = {
        "stationary": score([0, 0, 0]),
        "horizontal_5_mm_s": score([0.5, 0, 0]),
        "vertical_5_mm_s": score([0, 0, -0.5]),
        "horizontal_15_mm_s": score([1.5, 0, 0]),
        "mixed_15_mm_s": score([0.9, 0, 1.2]),
        "persistent_30_mm_s": score([3, 0, 0]),
        "recover_from_30_mm_s": score(recovery),
        "immediate_failure": score([3, 0, 0], 0),
        "failure_after_one_second": score([3, 0, 0], 499),
    }
    checks = {}
    for metric in ("return", "discounted_return"):
        x = {k: v[metric] for k, v in examples.items()}
        checks[metric] = bool(
            x["stationary"]
            > x["recover_from_30_mm_s"]
            > x["persistent_30_mm_s"]
            > x["failure_after_one_second"]
            > x["immediate_failure"]
            and x["stationary"] > x["horizontal_5_mm_s"] > x["horizontal_15_mm_s"]
            and np.isclose(x["horizontal_5_mm_s"], x["vertical_5_mm_s"])
            and np.isclose(x["horizontal_15_mm_s"], x["mixed_15_mm_s"])
        )
    return {"examples": examples, "ordering_checks": checks, "passed": all(checks.values())}


def run(root, output):
    output.mkdir(parents=True, exist_ok=False)
    begin = time.perf_counter()
    synthetic = counterexamples()
    reports = {}
    for name, labels in (
        ("13", ["parent", "midpoint", "final"]),
        ("12", ["midpoint", "final"]),
    ):
        result = audit(root / f"velocity_hover_ppo_{name}", labels, candidate_scale=0.5)
        reports[name] = result
        (output / f"run_{name}.json").write_text(json.dumps(result, indent=2) + "\n")
        print(
            json.dumps({"completed_run": name, "evaluations": result["evaluations"]}),
            flush=True,
        )
    parent = reports["13"]["evaluations"]["parent"]
    reference = np.array(parent["candidate"]["per_case_return"])
    comparisons = {}
    for run_id, report in reports.items():
        for label, result in report["evaluations"].items():
            if (run_id, label) == ("13", "parent"):
                continue
            comparisons[f"run_{run_id}_{label}"] = bool(
                np.all(reference > np.array(result["candidate"]["per_case_return"]))
            )
    report = {
        "provenance": evidence(),
        "completed_utc": utc_now(),
        "wall_seconds": time.perf_counter() - begin,
        "candidate": {
            "objective": "vector",
            "velocity_scale_cm_s": 0.5,
            "maximum_velocity_rate": 3,
            "other_terms_changed": False,
            "velocity_window_seconds": 0.1,
        },
        "counterexamples": synthetic,
        "retained_parent_beats_sideways_regressions_per_case": comparisons,
        "physical_replay_passed": all(r["passed"] for r in reports.values()),
        "passed": synthetic["passed"]
        and all(comparisons.values())
        and all(r["passed"] for r in reports.values()),
        "scope": "Objective audit, not learning; matched recorded actions replayed through the original plant. Synthetic outcomes test return ordering, not realizability.",
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)
    if not report["passed"]:
        raise RuntimeError("Do not train: coordinated reward audit failed")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    run(args.root, args.output)
