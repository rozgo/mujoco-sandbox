"""Verify one GPU training lineage and aggregate its measured stage records."""

import argparse
import hashlib
import json
import math
from datetime import datetime
from pathlib import Path

import torch
from adaptive_locomotion.bodies import ROOT


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def aggregate(rows):
    fields = (
        "training_seconds",
        "setup_seconds",
        "process_wall_seconds",
        "transitions",
        "iterations",
        "optimizer_steps",
    )
    result = {k: sum(r[k] for r in rows) for k in fields}
    result.update(
        stages=len(rows),
        parallel_worlds=4096,
        transitions_per_training_second=result["transitions"]
        / result["training_seconds"],
        aggregate_simulated_hours=result["transitions"] * 0.02 / 3600,
        world_physics_steps=result["transitions"] * 10,
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Preserve previous reports; choose a new output file")
    recipe = json.loads(args.recipe.read_text())
    recipe_hash = digest(args.recipe)
    initial_path = args.run / "stage_01/initial.pt"
    initial = torch.load(initial_path, map_location="cpu", weights_only=False)
    assert initial["parent"] is None
    assert all(
        initial[k] == 0
        for k in (
            "transitions",
            "iterations",
            "optimizer_steps",
            "training_seconds",
            "cumulative_training_seconds",
        )
    )
    rows, hashes = [], set()
    ancestry = 0.0
    for stage in recipe["stages"]:
        run = args.run / f"stage_{stage['stage']:02d}"
        process = json.loads((run / "process.json").read_text())
        measured = json.loads((run / "training.json").read_text())
        checkpoint = run / "policy.pt"
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        sha = digest(checkpoint)
        assert process["completed"] and process["returncode"] == 0
        assert process["recipe_sha256"] == recipe_hash
        assert sha == process["checkpoint_sha256"] == measured["checkpoint_sha256"]
        assert measured["num_envs"] == 4096
        assert measured["horizon"] == 24 and measured["epochs"] == 4
        assert measured["minibatch_size"] == 3072
        assert measured["device"] == measured["actor_device"] == "cuda"
        assert measured["physics_backend"] == "warp"
        assert measured["source_dirty"] is False
        assert measured["physics_timestep_s"] == 0.002
        assert measured["control_timestep_s"] == 0.02
        assert measured["transitions"] == stage["gpu_transitions"]
        assert measured["iterations"] == stage["gpu_iterations"]
        assert measured["optimizer_steps"] == measured["iterations"] * 128
        assert measured["stop_reason"] == "iteration_limit"
        assert math.isclose(measured["ancestry_seconds"], ancestry, abs_tol=1e-6)
        if rows:
            previous = args.run / f"stage_{stage['stage'] - 1:02d}/policy.pt"
            assert saved["parent"] == str(previous.resolve().relative_to(ROOT))
        else:
            assert saved["parent"] is None
        for key in (
            "reference_sha256",
            "single_reference_sha256",
            "front_reference_sha256",
            "standing_reference_sha256",
        ):
            assert not measured.get(key) or measured[key] in hashes
        ancestry += measured["training_seconds"]
        assert math.isclose(
            saved["cumulative_training_seconds"], ancestry, abs_tol=1e-6
        )
        rows.append(
            dict(
                process,
                phase=stage["phase"],
                num_envs=measured["num_envs"],
                source_commit=measured["source_commit"],
                training_record_sha256=digest(run / "training.json"),
                walking_replay_rows=measured["walking_replay_rows"],
                standing_replay_rows=measured["standing_replay_rows"],
                reference_sha256=measured["reference_sha256"],
                standing_reference_sha256=measured["standing_reference_sha256"],
            )
        )
        hashes.add(sha)
    assert len({r["source_commit"] for r in rows}) == 1
    result = {
        "scope": "One random-initialized lineage; every stage uses 4096 CUDA physics worlds. Training references are earlier checkpoints from this same run. No claim of behavioral acceptance is made by this timing ledger.",
        "recipe_sha256": recipe_hash,
        "initial_checkpoint_sha256": digest(initial_path),
        "initial_parent": initial["parent"],
        "initial_transitions": initial["transitions"],
        "final_checkpoint_sha256": rows[-1]["checkpoint_sha256"],
        "started_utc": rows[0]["started_utc"],
        "finished_utc": rows[-1]["finished_utc"],
        "driver_elapsed_seconds": (
            datetime.fromisoformat(rows[-1]["finished_utc"])
            - datetime.fromisoformat(rows[0]["started_utc"])
        ).total_seconds(),
        "healthy_walking": aggregate(rows[:5]),
        "damage_adaptation": aggregate(rows[5:21]),
        "complete_walking": aggregate(rows[:21]),
        "standing": aggregate(rows[21:28]),
        "moving": aggregate(rows[28:]),
        "total": aggregate(rows),
        "stages": rows,
        "timing": "Learning includes physics, host observation/reward work, transfers and PPO optimization. Setup and process elapsed are separately measured. Evaluation, media and abandoned attempts are excluded from this lineage.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "stages"}), flush=True)


if __name__ == "__main__":
    main()
