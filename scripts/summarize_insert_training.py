"""Collect the insertion training record for the journal: learning curves and compute per run, checkpoint
evaluations in the C environment and the full simulation, the yardsticks on the same seeds, and the split
of the full-simulation placement error into aim, stroke and release.

Reads the trainer logs copied from the GPU machine (outputs/neural_insertion/puffer/logs), the evaluation
reports in docs/neural_insertion/policy/ and the recorded full-simulation traces. Writes
docs/neural_insertion/INSERT_RESULTS.json. Simulates nothing.

Usage: uv run --locked python scripts/summarize_insert_training.py
"""

import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))

from sixlegs.neural_insertion.training_log import boxes, history, summary  # noqa: E402

DOCS = ROOT/"docs/neural_insertion"
LOGS = ROOT/"outputs/neural_insertion/puffer/logs"
RUNS = {
    "insert_v1": ("First insertion environment", "stopped by hand after the 65.5M checkpoint: the stages could "
                  "still move after the stroke started, so the needle could move sideways in the tissue"),
    "insert_v2": ("Stages held from the start of the stroke; stroke timed as the full cycle", "completed"),
}
CHECKPOINTS = (("65.5M", "insert_v2_65M"), ("131M", "insert_v2_131M"), ("196.6M", "insert_v2_197M"),
               ("299.9M", "insert_v2_300M"))


def text(path):
    return path.read_text().strip() if path.exists() else None


def full(name):
    return json.loads((DOCS/"policy"/f"{name}.json").read_text())["full_simulation"]


def placements(fs):
    """Every placed thread's distance from its target (µm), by disturbance level."""
    return {lv: [round(s["placement_lateral_um"], 2) for r in fs["runs"] if str(r["level"]) == lv
                 for s in r["sites"] if s["passed"]] for lv in ("0", "1", "2")}


def error_budget(run_dir):
    """Thread end to the (moving) target at the stroke start, before release and after, per thread (µm)."""
    rows = []
    for run in sorted(run_dir.glob("L*_s*")):
        tr = np.load(run/"trace.npz", allow_pickle=True)
        ph, end, tgt = tr["phase"].astype(str), tr["eyelet"], tr["true_target"]
        for i in (0, 5, 1):
            idx = [np.where(ph == f"{p} {i}")[0] for p in ("policy approach", "insert", "snap back")]
            if any(len(k) == 0 for k in idx):
                continue
            a, b, c = (k[-1] for k in idx)
            e = lambda k: (end[k, :2]-tgt[k, :2])*1e3
            rows.append((e(a), e(b), e(c)))
    A, B, C = (np.array([r[k] for r in rows]) for k in range(3))
    stat = lambda x: {"median": float(np.median(np.linalg.norm(x, axis=1))),
                      "p90": float(np.percentile(np.linalg.norm(x, axis=1), 90)),
                      "max": float(np.linalg.norm(x, axis=1).max())}
    return {"threads": len(rows), "aim": stat(A), "after_stroke": stat(B), "placed": stat(C),
            "stroke_shift": stat(B-A), "release_shift": stat(C-B),
            "release_shift_over_15um": int((np.linalg.norm(C-B, axis=1) > 15).sum())}


def main():
    out = {"description": "Learned thread insertion (insert_core.h, PufferLib 5.0) evaluated in the C environment "
                          "and in the full thread simulation (policy_cycle.py); see INSERT_TRAINING.md.",
           "runs": {}, "compute": {}}
    for run, (change, outcome) in RUNS.items():
        rows = boxes((LOGS/f"{run}.log").read_text(errors="replace"))
        out["compute"][run] = summary(rows, LOGS/f"{run}_resources.csv")
        out["runs"][run] = {"change": change, "outcome": outcome, "commit": text(LOGS/f"{run}.commit"),
                            "started_utc": text(LOGS/f"{run}.start"), "history": history(rows)}
    final = json.loads((DOCS/"policy/insert_v2_300M.json").read_text())
    base_c = json.loads((DOCS/"INSERT_BASELINE_C_V2.json").read_text())
    pick = lambda s: {k: s[k] for k in ("success", "placement_um_median", "placement_um_p90", "thread_touch", "placed")}
    out["c_environment"] = {
        "seeds": "2,000,000-2,000,199 per level",
        "learned_deterministic": {lv: pick(final["c_environment"][f"deterministic_level{lv}"]) for lv in ("0", "1", "2")},
        "learned_sampled": {lv: pick(final["c_environment"][f"sampled_level{lv}"]) for lv in ("0", "1", "2")},
        "compensating": {lv: pick(base_c[f"compensating_level{lv}"]) for lv in ("0", "1", "2")},
        "aiming": {lv: pick(base_c[f"yardstick_level{lv}"]) for lv in ("0", "1", "2")}}
    tube = json.loads((DOCS/"TUBE_BASELINE.json").read_text())
    approved = {"summary": tube["summary"], "runs": tube["runs"]}
    out["full_simulation"] = {
        "seeds": "1001-1010 at levels 1 and 2, 1001 at level 0; sites 0, 5, 1",
        "summary": {"learned_deterministic": full("insert_v2_300M_deterministic")["summary"],
                    "learned_sampled": full("insert_v2_300M")["summary"],
                    "compensating": full("compensating_yardstick")["summary"],
                    "approved": tube["summary"]},
        "placements_um": {"learned_deterministic": placements(full("insert_v2_300M_deterministic")),
                          "compensating": placements(full("compensating_yardstick")),
                          "approved": placements(approved)}}
    for name, s in out["full_simulation"]["summary"]["approved"].items():
        lv = name.split("_")[1]
        s.setdefault("within_10um", int(sum(v < 10 for v in out["full_simulation"]["placements_um"]["approved"][lv])))
    progression = []
    for label, name in CHECKPOINTS:
        rep = json.loads((DOCS/"policy"/f"{name}.json").read_text())
        row = {"checkpoint": label,
               "c_sampled_level1": rep["c_environment"]["sampled_level1"]["success"],
               "c_deterministic_level1": rep["c_environment"]["deterministic_level1"]["success"],
               "c_deterministic_level2": rep["c_environment"]["deterministic_level2"]["success"],
               "full_sampled_within_10um": {lv: rep["full_simulation"]["summary"][f"level_{lv}"]["within_10um"] for lv in ("1", "2")},
               "full_sampled_median_um": {lv: rep["full_simulation"]["summary"][f"level_{lv}"]["placement_um_median"] for lv in ("1", "2")}}
        progression.append(row)
    out["checkpoints"] = progression
    out["error_budget"] = error_budget(ROOT/"outputs/neural_insertion/policy_eval/insert_v2_300M_deterministic")
    stats = json.loads((DOCS/"THREAD_END_STATS.json").read_text())
    out["thread_end"] = {"offset_settled_um": stats["offset_m"]["settled"]*1e6,
                         "release_jump_um": {k: stats["jump_m"][k]*1e6 for k in ("median", "p90", "max")},
                         "releases": len(stats["jump_m"]["samples"]) if isinstance(stats["jump_m"]["samples"], list)
                         else stats["jump_m"]["samples"]}
    (DOCS/"INSERT_RESULTS.json").write_text(json.dumps(out, indent=1)+"\n")
    print(json.dumps({k: out[k] for k in ("checkpoints", "error_budget")}, indent=1))


if __name__ == "__main__":
    main()
