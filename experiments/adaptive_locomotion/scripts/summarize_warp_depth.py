"""Compare the 40-round CPU control with the completed 40-round Warp run."""

import hashlib
import json
import subprocess

from evaluate_visible_steps import compare

from adaptive_locomotion.bodies import ROOT


def main():
    docs = ROOT / "docs/locomotion/warp_training"
    labels = ("depth_mjbatch_seed2", "time_warp_seed2")
    reports, evaluations, rows, source_hashes, initial_hashes = [], [], [], [], []
    for label in labels:
        directory = docs / label
        report = json.loads((directory / "training.json").read_text())
        provenance = json.loads((directory / "checkpoint.json").read_text())
        evaluation = json.loads((directory / "evaluation.json").read_text())
        assert report["iterations"] == 40 and report["optimizer_steps"] == 5120
        assert report["transitions"] == 3932160
        assert not report["source_dirty"]
        reports.append(report)
        evaluations.append(evaluation)
        initial_hashes.append(provenance["initial_tensor_sha256"])
        source_hashes.append({})
        for file in ("train.py", "env.py", "policy.py"):
            path = f"experiments/adaptive_locomotion/src/adaptive_locomotion/{file}"
            result = subprocess.run(
                ["git", "show", f"{report['source_commit']}:{path}"],
                cwd=ROOT,
                check=True,
                capture_output=True,
            )
            source_hashes[-1][file] = hashlib.sha256(result.stdout).hexdigest()
        rows.append(
            {
                "label": label,
                "training_seconds": report["training_seconds"],
                "completed": sum(
                    c["completed_with_allowed_support"] for c in evaluation["cases"]
                ),
                "all_upright_with_valid_support": all(
                    r["survived"] and r["bad_support_control_windows"] == 0
                    for c in evaluation["cases"]
                    for r in c["rows"]
                ),
            }
        )
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
        "max_iterations",
        "budget_seconds",
        "stop_reason",
        "source_commit",
    }
    cpu, warp = reports
    differences = {
        k: [cpu.get(k), warp.get(k)]
        for k in cpu.keys() | warp.keys()
        if k not in excluded and cpu.get(k) != warp.get(k)
    }
    config_match = (
        not differences
        and len(set(initial_hashes)) == 1
        and source_hashes[0] == source_hashes[1]
    )
    report = {
        "same_learning_configuration": config_match,
        "configuration_differences": differences,
        "source_tensor_and_code_hashes": {
            "initial_parameters": initial_hashes,
            "code": source_hashes,
        },
        "results": rows,
        "speedup": rows[0]["training_seconds"] / rows[1]["training_seconds"],
        "by_case": {
            a["case"]: {
                "cpu": a["completed_with_allowed_support"],
                "warp": b["completed_with_allowed_support"],
            }
            for a, b in zip(
                evaluations[0]["cases"], evaluations[1]["cases"], strict=True
            )
        },
        "gait_comparison": compare(*evaluations),
        "scope": "One follow-up matched-update control after equal-time regression was observed; not an independent three-seed acceptance or a new checkpoint selection. Same network/reward implementation hashes are verified across documentation-only source revisions.",
    }
    (docs / "depth_summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: v
                for k, v in report.items()
                if k not in ("gait_comparison", "source_tensor_and_code_hashes")
            },
            indent=2,
        )
    )
    if not config_match or not all(r["all_upright_with_valid_support"] for r in rows):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
