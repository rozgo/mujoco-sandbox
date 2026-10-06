"""Scripted yardstick baseline for the thread-tube cycle: three sites per run over predetermined seeds.

Evaluation seeds 1001-1010 at levels 1 (nominal) and 2 (stress), and one undisturbed run (level 0 is
deterministic). Seeds were fixed before any training; later policies are compared on these same seeds and
levels. Writes docs/neural_insertion/TUBE_BASELINE.json and keeps each run's directory (report, trace,
scene) under outputs/neural_insertion/tube_baseline/.

Usage: uv run --locked python scripts/evaluate_tube_baseline.py [--workers 10]
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/"outputs/neural_insertion/tube_baseline"
SEEDS = tuple(range(1001, 1011))
SITES = (0, 5, 1)


def run(job):
    level, seed = job
    out = OUT/f"L{level}_s{seed}"
    started = time.time()
    proc = subprocess.run([sys.executable, "-m", "sixlegs.neural_insertion.tube_cycle", "--level", str(level),
                           "--seed", str(seed), "--sites", *map(str, SITES), "--output", str(out)],
                          cwd=ROOT, capture_output=True, text=True)
    report = json.loads((out/"report.json").read_text()) if (out/"report.json").exists() else {}
    checks = [e for e in report.get("events", []) if "check" in e]
    placed = [e for e in checks if e["check"].endswith("thread left in tissue")]
    return {"level": level, "seed": seed, "status": report.get("status", f"no report (exit {proc.returncode})"),
            "wall_s": round(time.time()-started, 1), "simulated_s": report.get("simulated_s"),
            "sites": [{"site": int(e["check"].split()[1].rstrip(":")), "passed": e["passed"],
                       "end_depth_mm": e.get("end_depth_mm"), "placement_lateral_um": 1e3*e.get("placement_lateral_mm", np.nan)}
                      for e in placed],
            "failed_checks": [e["check"] for e in checks if not e["passed"]],
            "prohibited_contacts": report.get("prohibited_count"), "disturbances": report.get("disturbances")}


def summarize(rows):
    out = {}
    for level in sorted({r["level"] for r in rows}):
        rs = [r for r in rows if r["level"] == level]
        sites = [s for r in rs for s in r["sites"]]
        placed = [s for s in sites if s["passed"]]
        lat = np.array([s["placement_lateral_um"] for s in placed])
        out[f"level_{level}"] = {
            "runs": len(rs), "runs_completed": sum(r["status"] == "completed" for r in rs),
            "sites_attempted": len(rs)*len(SITES), "threads_placed": len(placed),
            "placement_um_median": float(np.median(lat)) if len(lat) else None,
            "placement_um_p90": float(np.percentile(lat, 90)) if len(lat) else None,
            "placement_um_max": float(lat.max()) if len(lat) else None,
            "end_depth_mm_range": [float(min(s["end_depth_mm"] for s in placed)),
                                   float(max(s["end_depth_mm"] for s in placed))] if placed else None,
            "prohibited_contacts": int(sum(r["prohibited_contacts"] or 0 for r in rs))}
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = [(0, 1001)]+[(level, seed) for level in (1, 2) for seed in SEEDS]
    started = datetime.now(timezone.utc).isoformat()
    t0 = time.time()
    with ProcessPoolExecutor(args.workers) as pool:
        rows = []
        for row in pool.map(run, jobs):
            rows.append(row)
            print(f"L{row['level']} seed {row['seed']}: {row['status']}  "
                  + "  ".join(f"site {s['site']}: {s['placement_um'] if 'placement_um' in s else s['placement_lateral_um']:.0f} um" for s in row["sites"]), flush=True)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    result = {"controller": "scripted yardstick (tube_cycle.py): measured target, one correction from the measured "
                            "tip, fixed insertion stroke; no learning",
              "cycle": "thread-tube design, sites 0, 5, 1; stand-in constraints for sticking, release, tube hold "
                       "and tissue grip; fast thread physics",
              "seeds": {"evaluation": list(SEEDS), "note": "fixed before any training; level 0 is deterministic"},
              "levels": {"0": "undisturbed", "1": "nominal quality workcell", "2": "stress, twice nominal"},
              "started_utc": started, "wall_s_total": round(time.time()-t0, 1), "source_commit": commit,
              "summary": summarize(rows), "runs": rows}
    path = ROOT/"docs/neural_insertion/TUBE_BASELINE.json"
    path.write_text(json.dumps(result, indent=1, default=float)+"\n")
    print(json.dumps(result["summary"], indent=1))


if __name__ == "__main__":
    main()
