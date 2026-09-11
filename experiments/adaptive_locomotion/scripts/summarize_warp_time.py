"""Report the equal-time comparison separately from matched-update acceptance."""

import json

from adaptive_locomotion.bodies import ROOT


def main():
    directory = ROOT / "docs/locomotion/warp_training"
    pair = json.loads((directory / "time_pairs.json").read_text())["pairs"]
    assert len(pair) == 1 and pair[0]["seed"] == 2
    rows, configs = {}, []
    for backend in ("mjbatch", "warp"):
        path = directory / f"time_{backend}_seed2"
        train = json.loads((path / "training.json").read_text())
        evaluation = json.loads((path / "evaluation.json").read_text())
        configs.append(train)
        assert train["budget_seconds"] == 90 and train["num_envs"] == 4096
        assert train["max_iterations"] is None and train["stop_reason"] == "time_limit"
        assert train["minibatch_size"] == 3072 and train["epochs"] == 4
        rows[backend] = {
            "training_seconds": train["training_seconds"],
            "setup_seconds": train["setup_seconds"],
            "transitions": train["transitions"],
            "transitions_per_second": train["transitions"] / train["training_seconds"],
            "iterations": train["iterations"],
            "optimizer_steps": train["optimizer_steps"],
            "completed": sum(
                c["completed_with_allowed_support"] for c in evaluation["cases"]
            ),
            "attempted": sum(c["trials"] for c in evaluation["cases"]),
            "all_upright_with_valid_support": all(
                r["survived"] and r["bad_support_control_windows"] == 0
                for case in evaluation["cases"]
                for r in case["rows"]
            ),
        }
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
        "iterations",
        "optimizer_steps",
        "transitions",
    }
    cpu, warp = configs
    differences = {
        key: [cpu.get(key), warp.get(key)]
        for key in cpu.keys() | warp.keys()
        if key not in excluded and cpu.get(key) != warp.get(key)
    }
    report = {
        "seed": 2,
        "same_budget_configurations": not differences,
        "configuration_differences": differences,
        "results": rows,
        "training_throughput_ratio": rows["warp"]["transitions_per_second"]
        / rows["mjbatch"]["transitions_per_second"],
        "experience_ratio": rows["warp"]["transitions"]
        / rows["mjbatch"]["transitions"],
        "paired_evaluation": pair[0],
        "scope": "One predetermined equal-time pair from a pretrained parent. The deadline guard may discard the last rollout or stop an update between minibatches; actual optimizer steps are recorded. More experience is not necessarily better policy quality; this does not replace the three-seed matched-update acceptance.",
    }
    (directory / "time_summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {k: v for k, v in report.items() if k != "paired_evaluation"}, indent=2
        )
    )
    if differences or not all(
        r["all_upright_with_valid_support"] for r in rows.values()
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
