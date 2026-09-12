"""Frozen-policy comparison with matched foot and terrain sliding friction."""

import hashlib
import json
import subprocess
import time

import mujoco
import numpy as np
from adaptive_locomotion.bodies import ROOT
from adaptive_locomotion.standing_evaluate import run_case
from adaptive_locomotion.train import load_checkpoint


def main():
    checkpoint = (
        ROOT / "assets/locomotion/checkpoints/standing/consolidate_120s_seed12.pt"
    )
    reference = json.loads(
        (ROOT / "docs/locomotion/standing/final/healthy_cpu.json").read_text()
    )
    reviewed = json.loads(
        (ROOT / "previews/locomotion/standing/failure_review.json").read_text()
    )["cases"][:8]
    terrain = [(c["surface"], False) for c in reviewed]
    controls = [("flat", True), ("pads_high", False), ("gap_fr", False)]
    net, _ = load_checkpoint(checkpoint)
    started = time.perf_counter()
    runs = []
    directory = ROOT / "outputs/locomotion/standing/friction_comparison"
    directory.mkdir(parents=True, exist_ok=True)
    for friction in (0.8, 1.0, 1.2):
        results = []
        for surface, transition in terrain + controls:
            result, frames, model = run_case(
                net,
                "healthy",
                surface,
                trials=4,
                seconds=12 if transition else 10,
                seed=9307,
                capture=True,
                physics_backend="mjbatch",
                support_friction=friction,
            )
            original = next(
                c
                for c in reference["cases"]
                if c["surface"] == surface and c["transition"] == transition
            )
            if friction == 0.8:
                assert result["rows"] == original["rows"], (
                    "Baseline drifted from original evaluation"
                )
            trajectory = directory / f"mu{friction:.1f}_{surface}.npz"
            np.savez_compressed(
                trajectory, **{k: np.asarray([f[k] for f in frames]) for k in frames[0]}
            )
            model_path = directory / f"mu{friction:.1f}_{surface}.mjb"
            buffer = np.empty(mujoco.mj_sizeModel(model), np.uint8)
            mujoco.mj_saveModel(model, buffer=buffer)
            model_path.write_bytes(buffer.tobytes())
            result.update(
                role="reviewed_terrain"
                if (surface, transition) in terrain
                else "passing_control",
                captured_trial=0,
                trajectory=str(trajectory.relative_to(ROOT)),
                trajectory_sha256=hashlib.sha256(trajectory.read_bytes()).hexdigest(),
                model=str(model_path.relative_to(ROOT)),
                model_sha256=hashlib.sha256(model_path.read_bytes()).hexdigest(),
                baseline_rows_identical=friction == 0.8
                and result["rows"] == original["rows"],
            )
            results.append(result)
        reviewed_rows = [row for r in results[:8] for row in r["rows"]]
        control_rows = [row for r in results[8:] for row in r["rows"]]
        summary = {
            "sliding_friction_feet_and_supports": friction,
            "terrain_passed": sum(row["passed"] for row in reviewed_rows),
            "terrain_total": len(reviewed_rows),
            "terrain_upright": sum(row["survived"] for row in reviewed_rows),
            "mean_terrain_max_drift_m": float(
                np.mean([row["max_idle_drift_m"] for row in reviewed_rows])
            ),
            "mean_terrain_max_tilt_deg": float(
                np.mean([row["max_idle_tilt_deg"] for row in reviewed_rows])
            ),
            "unintended_support_trials": sum(
                row["peak_unintended_support_n"] > 1 for row in reviewed_rows
            ),
            "controls_passed": sum(row["passed"] for row in control_rows),
            "controls_total": len(control_rows),
            "all_upright": all(row["survived"] for row in reviewed_rows + control_rows),
            "all_torque_limits": all(
                row["peak_torque_limit_ratio"] <= 1.0001
                for row in reviewed_rows + control_rows
            ),
            "max_sampled_penetration_m": max(
                row["max_penetration_m"] for row in reviewed_rows + control_rows
            ),
        }
        runs.append({"summary": summary, "cases": results})
        print(json.dumps(summary), flush=True)
    output = ROOT / "docs/locomotion/standing/TERRAIN_FRICTION_SWEEP.json"
    output.write_text(
        json.dumps(
            {
                "source_commit": subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
                ).strip(),
                "checkpoint_sha256": hashlib.sha256(
                    checkpoint.read_bytes()
                ).hexdigest(),
                "seed": 9307,
                "physics_backend": "mjbatch",
                "physics_timestep_s": 0.002,
                "scope": "Both allowed foot/stump sliding friction and collidable static terrain sliding friction set to the same coefficient before the batch copies its model. Other links, torsional/rolling coefficients, solver, geometry, reset perturbations, network and gates unchanged.",
                "selection": "Eight user-reviewed healthy terrain conditions plus three previously passing healthy controls; four paired trials per condition. Reuses the original holdout seed for diagnosis, not a new independent holdout or training result.",
                "new_training_seconds": 0,
                "aggregate_simulated_world_seconds": 3 * 4 * (10 * 10 + 12),
                "evaluation_and_capture_wall_seconds": time.perf_counter() - started,
                "runs": runs,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
