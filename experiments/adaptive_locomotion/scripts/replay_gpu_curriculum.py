"""Replay archived walking reward stages on CUDA without importing trained weights."""

import argparse
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def utc():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stage_path(output, stage):
    return output / f"stage_{stage:02d}"


def worker(recipe, output, number):
    from adaptive_locomotion.train import load_checkpoint, train

    row = recipe["stages"][number - 1]
    run = stage_path(output, number)
    if (run / "initial.pt").exists():
        raise ValueError("Refusing to overwrite an existing training attempt")
    params = dict(row["parameters"])
    parent = stage_path(output, number - 1) / "policy.pt" if number > 1 else None
    ancestry = (
        load_checkpoint(parent)[1]["cumulative_training_seconds"] if parent else 0
    )
    mapping = {
        r["historical_checkpoint"]: stage_path(output, r["stage"]) / "policy.pt"
        for r in recipe["stages"][: number - 1]
    }
    for archived_key, argument in (
        ("historical_reference", "reference"),
        ("historical_single_reference", "single_reference"),
        ("historical_front_reference", "front_reference"),
    ):
        if row[archived_key]:
            params[argument] = mapping[row[archived_key]]
    train(
        output=run,
        resume=parent,
        seconds=300,
        allowance=ancestry + 301,
        extension_reason="Replay the accepted historical curriculum at its recorded PPO round count; no reward retuning or imported trained weights.",
        num_envs=recipe["gpu_num_envs"],
        minibatch_size=recipe["minibatch_size"],
        max_iterations=row["gpu_iterations"],
        physics_backend="warp",
        device="cuda",
        **params,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--through", type=int, default=21)
    parser.add_argument("--worker", type=int)
    args = parser.parse_args()
    recipe = json.loads(args.recipe.read_text())
    if args.worker:
        worker(recipe, args.output, args.worker)
        return
    if not 1 <= args.through <= len(recipe["stages"]):
        raise ValueError("Invalid final curriculum stage")
    args.output.mkdir(parents=True, exist_ok=True)
    recipe_hash = digest(args.recipe)
    records = []
    for row in recipe["stages"][: args.through]:
        run = stage_path(args.output, row["stage"])
        audit_path = run / "process.json"
        if audit_path.exists():
            audit = json.loads(audit_path.read_text())
            if audit["recipe_sha256"] != recipe_hash or not audit["completed"]:
                raise ValueError(
                    "Existing run failed or used another recipe; preserve it"
                )
            if digest(run / "policy.pt") != audit["checkpoint_sha256"]:
                raise ValueError("Completed checkpoint changed")
            records.append(audit)
            continue
        run.mkdir(parents=True, exist_ok=True)
        started, wall_start = utc(), time.perf_counter()
        hardware = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.used,utilization.gpu",
                "--format=csv",
            ],
            text=True,
        )
        command = [
            sys.executable,
            __file__,
            "--recipe",
            str(args.recipe),
            "--output",
            str(args.output),
            "--worker",
            str(row["stage"]),
        ]
        with (run / "process.log").open("w") as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        audit = {
            "stage": row["stage"],
            "recipe_sha256": recipe_hash,
            "started_utc": started,
            "finished_utc": utc(),
            "process_wall_seconds": time.perf_counter() - wall_start,
            "gpu_before": hardware,
            "returncode": result.returncode,
            "completed": False,
        }
        if (run / "training.json").exists():
            measured = json.loads((run / "training.json").read_text())
            audit.update(
                {
                    k: measured[k]
                    for k in (
                        "training_seconds",
                        "setup_seconds",
                        "transitions",
                        "iterations",
                        "optimizer_steps",
                        "checkpoint_sha256",
                        "cumulative_training_seconds",
                    )
                }
            )
            audit["completed"] = (
                result.returncode == 0
                and measured["iterations"] == row["gpu_iterations"]
                and measured["transitions"] == row["gpu_transitions"]
                and measured["optimizer_steps"] == row["gpu_iterations"] * 16
                and measured["stop_reason"] == "iteration_limit"
            )
        audit_path.write_text(json.dumps(audit, indent=2) + "\n")
        records.append(audit)
        (args.output / "progress.json").write_text(json.dumps(records, indent=2) + "\n")
        print(json.dumps(audit), flush=True)
        if not audit["completed"]:
            raise RuntimeError(
                f"Stage {row['stage']} did not complete; inspect its retained log"
            )


if __name__ == "__main__":
    main()
