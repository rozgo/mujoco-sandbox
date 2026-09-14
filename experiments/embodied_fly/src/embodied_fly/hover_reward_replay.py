"""Score saved actor flights by replaying their exact actions through the same plant."""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from embodied_fly.physical_contract import physical_contract
from embodied_fly.provenance import evidence, sha256, utc_now
from embodied_fly.velocity_demonstrations import environment
from embodied_fly.velocity_hover import reward_from_recipe, reward_rates


def audit(run, labels, *, candidate_scale=None):
    started = time.perf_counter()
    training = json.loads((run / "report.json").read_text())
    results = {}
    for label in labels:
        report = training["evaluations"][label]
        captures = []
        for case in report["cases"]:
            file = run / label / case["file"]
            if sha256(file) != case["sha256"] or not case["survived_ten_seconds"]:
                raise ValueError("Reward comparison requires verified complete paired flights")
            with np.load(file) as source:
                captures.append(
                    {
                        k: source[k].copy()
                        for k in ("qpos", "qvel", "act", "ctrl", "action", "post_position")
                    }
                )
        env = environment(len(captures), len(captures))
        if physical_contract(env.model) != training["physical_contract"]:
            raise ValueError("Physical contract differs during reward replay")
        env.reset(
            np.arange(env.n),
            state={
                k: np.stack([c[k][0] for c in captures])
                for k in ("qpos", "qvel", "act", "ctrl")
            },
        )
        recipe = training["recipe"]["reward"]
        score = reward_from_recipe(env, recipe)
        totals = {}
        returns = np.zeros(env.n)
        candidate_returns = np.zeros(env.n)
        candidate_totals = {}
        position_error = 0.0
        for step in range(len(captures[0]["action"])):
            env.batch.forward()  # Same pre-action refresh as the saved evaluations.
            env.step(np.stack([c["action"][step] for c in captures]))
            reward, failed, terms = score()
            if np.any(failed):
                raise RuntimeError("Complete-flight reward replay failed physically")
            returns += reward
            for name, value in terms.items():
                totals[name] = totals.get(name, np.zeros(env.n)) + value * env.control_dt
            if candidate_scale is not None:
                mean = score.mean()
                candidate_terms = reward_rates(
                    mean[:, :3],
                    mean[:, 3:],
                    env.fields["xmat"][:, env.template.thorax_id, 8],
                    horizontal_scale=candidate_scale,
                    velocity_objective="vector",
                )
                candidate_returns += sum(candidate_terms.values()) * env.control_dt
                for name, value in candidate_terms.items():
                    candidate_totals[name] = (
                        candidate_totals.get(name, np.zeros(env.n)) + value * env.control_dt
                    )
            position_error = max(
                position_error,
                float(
                    np.max(
                        np.abs(
                            env.fields["qpos"][:, :3]
                            - np.stack([c["post_position"][step] for c in captures])
                        )
                    )
                ),
            )
        results[label] = {
            "mean_return": float(returns.mean()),
            "per_case_return": returns.tolist(),
            "mean_term_totals": {k: float(v.mean()) for k, v in totals.items()},
            "maximum_body_position_replay_error_cm": position_error,
            "passed": position_error <= 1e-9,
        }
        if candidate_scale is not None:
            results[label]["candidate"] = {
                "velocity_scale_cm_s": candidate_scale,
                "mean_return": float(candidate_returns.mean()),
                "per_case_return": candidate_returns.tolist(),
                "mean_term_totals": {k: float(v.mean()) for k, v in candidate_totals.items()},
            }
    return {
        "provenance": evidence(),
        "completed_utc": utc_now(),
        "wall_seconds": time.perf_counter() - started,
        "source_training_report_sha256": sha256(run / "report.json"),
        "evaluations": results,
        "passed": all(v["passed"] for v in results.values()),
        "scope": "No learning or new neural inference; exact recorded actor actions, original reward, same physical plant",
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--labels", nargs="+", default=["parent", "midpoint", "final"])
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError("Preserve previous reward audit")
    report = audit(args.run, args.labels)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)
    if not report["passed"]:
        raise RuntimeError("Reward replay position check failed")
