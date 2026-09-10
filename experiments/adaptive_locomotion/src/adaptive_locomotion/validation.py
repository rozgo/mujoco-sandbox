"""Frozen final cases, paired comparisons and scalar contact diagnostics.

Defined before final evaluation. No tuning or training on these cases.
"""

import json
from pathlib import Path

import mujoco
import numpy as np

from .bodies import PRESETS, BodySpec
from .evaluate import CASES, rollout
from .train import load_checkpoint

FINAL_CASES = {
    "final_healthy": (PRESETS["healthy"], "flat", None),
    "final_right_short": (BodySpec("final_right_short", (1, 0.62, 1, 1)), "flat", None),
    "final_left_pair": (BodySpec("final_left_pair", (0.78, 1, 0.86, 1)), "flat", None),
    "final_motor_loss": (
        BodySpec("final_motor_loss", (1, 1, 1, 0.88)),
        "flat",
        (4.2, 6, 0.35),
    ),
    "final_low_course": (
        BodySpec("final_low_course", (0.86, 1, 1, 0.74)),
        "heldout_steps",
        None,
    ),
    "final_missing_calf": (
        BodySpec("final_missing_calf", absent=("RR",)),
        "flat",
        None,
    ),
}
FINAL_SEED = 20260910


def wilson(k, n):
    z = 1.959963984540054
    p = k / n
    denominator = 1 + z * z / n
    middle = (p + z * z / (2 * n)) / denominator
    radius = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return [float(middle - radius), float(middle + radius)]


def validate(checkpoints, output, trials=32):
    CASES.update(FINAL_CASES)
    report = {
        "split": "frozen final; not used for training or checkpoint selection",
        "seed": FINAL_SEED,
        "trials_per_case": trials,
        "seconds": 12,
        "policies": {},
        "paired_differences": {},
    }
    for name, path in checkpoints.items():
        net, saved = load_checkpoint(path)
        rows = []
        for case in FINAL_CASES:
            result, _, _ = rollout(
                net, case, trials=trials, seed=FINAL_SEED, seconds=12
            )
            result["completion_wilson_95"] = wilson(result["completed_5m"], trials)
            rows.append(result)
            print(
                json.dumps(
                    {
                        "policy": name,
                        "case": case,
                        "completed": result["completed_5m"],
                        "trials": trials,
                    }
                ),
                flush=True,
            )
        report["policies"][name] = {
            "checkpoint": str(path),
            "training_seconds": saved["cumulative_training_seconds"],
            "cases": rows,
        }
    if "history" in report["policies"] and "blind" in report["policies"]:
        rng = np.random.default_rng(FINAL_SEED)
        for a, b in zip(
            report["policies"]["history"]["cases"],
            report["policies"]["blind"]["cases"],
            strict=True,
        ):
            diff = np.array(
                [
                    int(x["completed_5m"]) - int(y["completed_5m"])
                    for x, y in zip(a["rows"], b["rows"], strict=True)
                ]
            )
            draws = rng.choice(diff, (10000, len(diff)), replace=True).mean(1)
            report["paired_differences"][a["case"]] = {
                "history_minus_blind": float(diff.mean()),
                "bootstrap_95": np.quantile(draws, [0.025, 0.975]).tolist(),
            }
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(report, indent=2) + "\n")
    return report


def inspect_contacts(checkpoint, case="short_steps", seconds=12):
    net, _ = load_checkpoint(checkpoint)
    outcome, frames, model = rollout(net, case, trials=1, seconds=seconds, capture=True)
    data = mujoco.MjData(model)
    minimum = 0.0
    trunk_frames = 0
    for state in frames:
        data.qpos[:] = state["qpos"]
        data.qvel[:] = state["qvel"]
        mujoco.mj_forward(model, data)
        if data.ncon:
            minimum = min(minimum, float(np.min(data.contact.dist)))
            for c in data.contact:
                if (
                    model.geom_bodyid[c.geom1] == model.body("base").id
                    or model.geom_bodyid[c.geom2] == model.body("base").id
                ):
                    trunk_frames += 1
                    break
    fine, _, _ = rollout(net, case, trials=16, seconds=seconds, timestep=0.001)
    return {
        "case": case,
        "penetration_geometry_sample_interval_s": 0.02,
        "maximum_penetration_m": -minimum,
        "trunk_contact_frames": trunk_frames,
        "sampled_frames": len(frames),
        "original_outcome": outcome,
        "half_timestep_evaluation": fine,
        "note": "Contact geometry reconstructed from actual recorded states; penetration is sampled, not an every-substep bound.",
    }
