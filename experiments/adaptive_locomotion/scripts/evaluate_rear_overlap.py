"""Selection for one rear-overlap intervention; unchanged all-foot gait gates."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from evaluate_clearance import evaluate
from evaluate_visible_steps import compare as compare_visible

from adaptive_locomotion.bodies import ROOT

PARENT = ROOT / "assets/locomotion/checkpoints/limb_visible_steps_selected_seed2.pt"


def timing(report):
    rows = {}
    for g in report["gait_cases"]:
        if g["case"] not in ("lower_fl", "lower_fr", "whole_fl", "whole_fr"):
            continue
        r = [x["rear_timing"] for x in g["rows"]]
        rows[g["case"]] = {
            "overlap": float(
                np.mean([x["both_rear_feet_above_1cm_fraction"] for x in r])
            ),
            "separation": float(np.mean([x["separation_fraction"] for x in r]))
            if all(x["separation_fraction"] is not None for x in r)
            else None,
            "minimum_phase_samples": min(x["phase_samples"] for x in r),
        }
    return rows


def compare(parent, candidate):
    r = compare_visible(parent, candidate)
    a = timing(parent)
    b = timing(candidate)
    checks = r["checks"]
    checks["fr_overlap_halved_each"] = all(
        b[c]["overlap"] <= 0.5 * a[c]["overlap"] for c in ("lower_fr", "whole_fr")
    )
    checks["fr_phase_separation_at_least_quarter_cycle"] = all(
        b[c]["separation"] is not None and b[c]["separation"] >= 0.25
        for c in ("lower_fr", "whole_fr")
    )
    checks["fl_alternation_preserved"] = all(
        b[c]["separation"] is not None
        and b[c]["separation"] >= 0.30
        and b[c]["overlap"] <= a[c]["overlap"] + 0.01
        for c in ("lower_fl", "whole_fl")
    )
    checks["complete_rear_cycles"] = all(b[c]["minimum_phase_samples"] >= 8 for c in b)
    r.update(
        eligible=all(checks.values()),
        before_timing=a,
        after_timing=b,
        score=float(
            np.mean([b[c]["separation"] or 0 for c in ("lower_fr", "whole_fr")])
        ),
    )
    return r


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run", type=Path)
    args = p.parse_args()
    run = args.run.resolve()
    cache = run / "parent_dev.json"
    if cache.exists():
        parent = json.loads(cache.read_text())
        assert (
            parent["sha256"] == hashlib.sha256(PARENT.read_bytes()).hexdigest()
            and parent["seed"] == 9159
            and parent["physics_timestep_s"] == 0.0005
        )
    else:
        parent = evaluate(PARENT, cache, trials=8, seed=9159, timestep=0.0005)
    rows = []
    for name in ("iteration_0050.pt", "iteration_0100.pt", "policy.pt"):
        path = run / name
        if not path.exists():
            continue
        result = evaluate(
            path, run / f"{path.stem}_dev.json", trials=8, seed=9159, timestep=0.0005
        )
        row = compare(parent, result)
        rows.append(row)
        print(
            "CANDIDATE",
            json.dumps({k: v for k, v in row.items() if k != "feet"}),
            flush=True,
        )
    eligible = [r for r in rows if r["eligible"]]
    selected = max(eligible, key=lambda r: r["score"]) if eligible else None
    (run / "selection.json").write_text(
        json.dumps({"candidates": rows, "selected": selected}, indent=2) + "\n"
    )
    print("SELECTED", selected["checkpoint"] if selected else None, flush=True)
    return 0 if selected else 1


if __name__ == "__main__":
    raise SystemExit(main())
