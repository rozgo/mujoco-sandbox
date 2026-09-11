"""Apply the declared paired CPU/Warp learning gates without selecting checkpoints."""

import argparse
import json

from adaptive_locomotion.bodies import ROOT


def main(prefix="matched"):
    docs = ROOT / "docs/locomotion/warp_training"
    pairs = json.loads((docs / f"{prefix}_pairs.json").read_text())["pairs"]
    assert len(pairs) == 3 and {p["seed"] for p in pairs} == {2, 3, 4}
    timings, configurations, physical = [], [], []
    for pair in pairs:
        reports = []
        row = {"seed": pair["seed"]}
        for backend in ("mjbatch", "warp"):
            path = docs / f"{prefix}_{backend}_seed{pair['seed']}"
            report = json.loads((path / "training.json").read_text())
            reports.append(report)
            assert (
                report["iterations"] == report["max_iterations"]
                and report["optimizer_steps"] == 768
            )
            assert report["transitions"] == 589824
            row[f"{backend}_seconds"] = report["training_seconds"]
            row[f"{backend}_transitions_per_second"] = (
                report["transitions"] / report["training_seconds"]
            )
            evaluation = json.loads((path / "evaluation.json").read_text())
            physical.append(
                all(
                    r["survived"] and r["bad_support_control_windows"] == 0
                    for case in evaluation["cases"]
                    for r in case["rows"]
                )
            )
        cpu, warp = reports
        excluded = {
            "physics",
            "physics_backend",
            "physics_device",
            "warp_execution",
            "setup_seconds",
            "training_seconds",
            "cumulative_training_seconds",
            "progress",
            "checkpoint_sha256",
        }
        configurations.append(
            {
                k: [cpu.get(k), warp.get(k)]
                for k in cpu.keys() | warp.keys()
                if k not in excluded and cpu.get(k) != warp.get(k)
            }
        )
        row["speedup"] = row["mjbatch_seconds"] / row["warp_seconds"]
        timings.append(row)
    by_case = {
        case: {
            backend: sum(p["by_case"][case][backend] for p in pairs)
            for backend in ("cpu", "warp")
        }
        for case in pairs[0]["by_case"]
    }
    cpu_complete = sum(p["cpu_completed"] for p in pairs)
    warp_complete = sum(p["warp_completed"] for p in pairs)
    checks = {
        "matched_configurations": not any(configurations),
        "all_trials_upright_with_valid_support": all(physical),
        "pooled_completion_within_5pp": warp_complete / 216
        >= cpu_complete / 216 - 0.05,
        "no_body_more_than_two_trials_worse": all(
            c["warp"] >= c["cpu"] - 2 for c in by_case.values()
        ),
        "gait_retention_all_seeds": all(
            all(
                v
                for k, v in p["gait_comparison"]["checks"].items()
                if k != "all_tasks_support_valid"
            )
            for p in pairs
        ),
    }
    report = {
        "trials_per_backend": 216,
        "cpu_completed": cpu_complete,
        "warp_completed": warp_complete,
        "checks": checks,
        "passed": all(checks.values()),
        "configuration_differences": configurations,
        "timings": timings,
        "by_case": by_case,
        "pooled_training_speedup": sum(r["mjbatch_seconds"] for r in timings)
        / sum(r["warp_seconds"] for r in timings),
        "scope": f"Three paired seeds, fixed parent, {prefix} world/rollout layout and identical experience plus optimizer-step counts; development validation, not full curriculum retraining or general robustness.",
    }
    (docs / f"{prefix}_summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", choices=("matched", "scaled"), default="matched")
    main(**vars(parser.parse_args()))
