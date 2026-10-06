"""Evaluate an insertion controller in the C environment and in the full thread-tube simulation.

C environment (insert_core.h): the 200 predetermined evaluation seeds at levels 0, 1 and 2; a learned
checkpoint is run with its learned action noise (as in training; noise seeded per episode) and as its
deterministic mean. Full simulation (policy_cycle.py): three sites per run on the approved baseline's seeds
and levels (1001-1010 at levels 1 and 2, and 1001 at level 0), so results compare run for run with
docs/neural_insertion/TUBE_BASELINE.json. Writes docs/neural_insertion/policy/NAME.json and keeps each run's
directory under outputs/neural_insertion/policy_eval/NAME/.

Usage: uv run --locked python scripts/evaluate_insert_policy.py NAME (--weights W.bin | --scripted compensate)
           [--workers 10] [--skip-full]
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SEEDS = tuple(range(1001, 1011))
SITES = (0, 5, 1)


def c_evaluation(weights, scripted):
    from sixlegs.neural_insertion.align_policy import Policy
    from sixlegs.neural_insertion.insert_env import EVALUATION_SEEDS, evaluate, summary, yardstick

    out = {}
    if scripted:
        modes = {"scripted": yardstick(scripted == "compensate")}
    else:
        policy = Policy(weights, obs=24, actions=4)
        out["action_std"] = policy.std.tolist()

        def learned(sample):
            def act(env, obs):
                first = not hasattr(env, "_rng")
                if first:
                    env._rng = np.random.default_rng(7_000_000+env.seed) if sample else None
                return policy.act(obs, first, env._rng)
            return act
        modes = {"sampled": learned(True), "deterministic": learned(False)}
    for name, act in modes.items():
        for level in (0., 1., 2.):
            out[f"{name}_level{level:g}"] = s = summary(evaluate(act, EVALUATION_SEEDS, level))
            print(f"C {name} level {level:g}: success {s['success']:.3f} placed {s['placed']} median "
                  f"{s['placement_um_median']} timeout {s['timeout']:.3f} thread_touch {s['thread_touch']:.3f}", flush=True)
    return out


def run(job):
    out, level, seed, agent = job
    started = time.time()
    proc = subprocess.run([sys.executable, "-m", "sixlegs.neural_insertion.policy_cycle", *agent, "--level", str(level),
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
            "approaches": [{k: a.get(k) for k in ("site", "outcome", "policy_steps", "trigger_end_to_target_lateral_um",
                                                  "vessel_steps")} for a in report.get("approaches", [])],
            "failed_checks": [e["check"] for e in checks if not e["passed"]],
            "prohibited_contacts": report.get("prohibited_count"),
            "thread_touches": len(report.get("thread_touches", []))}


def summarize(rows):
    out = {}
    for level in sorted({r["level"] for r in rows}):
        rs = [r for r in rows if r["level"] == level]
        placed = [s for r in rs for s in r["sites"] if s["passed"]]
        lat = np.array([s["placement_lateral_um"] for s in placed])
        out[f"level_{level}"] = {
            "runs": len(rs), "runs_completed": sum(r["status"] == "completed" for r in rs),
            "sites_attempted": len(rs)*len(SITES), "threads_placed": len(placed),
            "within_10um": int((lat < 10).sum()),
            "placement_um_median": float(np.median(lat)) if len(lat) else None,
            "placement_um_p90": float(np.percentile(lat, 90)) if len(lat) else None,
            "placement_um_max": float(lat.max()) if len(lat) else None,
            "stops": [f"seed {r['seed']}: {r['status']}" for r in rs if r["status"] != "completed"],
            "prohibited_contacts": int(sum(r["prohibited_contacts"] or 0 for r in rs)),
            "thread_touches": int(sum(r["thread_touches"] for r in rs))}
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name")
    agent = parser.add_mutually_exclusive_group(required=True)
    agent.add_argument("--weights", type=Path)
    agent.add_argument("--scripted", choices=("aim", "compensate"))
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--skip-full", action="store_true")
    args = parser.parse_args()
    started = datetime.now(timezone.utc).isoformat()
    t0 = time.time()
    result = {"name": args.name, "started_utc": started,
              "source_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                                              text=True).stdout.strip()}
    if args.weights:
        result.update(weights=str(args.weights), weights_sha256=hashlib.sha256(args.weights.read_bytes()).hexdigest())
    else:
        result["controller"] = f"scripted ({args.scripted}), insert_core.h si_scripted"
    result["c_environment"] = c_evaluation(args.weights, args.scripted)
    if not args.skip_full:
        out = ROOT/"outputs/neural_insertion/policy_eval"/args.name
        agent = ["--weights", str(args.weights.resolve())] if args.weights else ["--scripted", args.scripted]
        jobs = [(out/f"L{level}_s{seed}", level, seed, agent) for level, seed in
                [(0, 1001)]+[(level, seed) for level in (1, 2) for seed in SEEDS]]
        rows = []
        with ProcessPoolExecutor(args.workers) as pool:
            for row in pool.map(run, jobs):
                rows.append(row)
                print(f"full L{row['level']} seed {row['seed']}: {row['status']}  "
                      + "  ".join(f"site {s['site']}: {s['placement_lateral_um']:.1f} um" for s in row["sites"]), flush=True)
        result["full_simulation"] = {"seeds": list(SEEDS), "sites": list(SITES), "summary": summarize(rows), "runs": rows,
                                     "actions": "learned noise, seeded per run (policy_cycle.py)" if args.weights else "scripted"}
    result["wall_s_total"] = round(time.time()-t0, 1)
    path = ROOT/"docs/neural_insertion/policy"/f"{args.name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=1, default=float)+"\n")
    print(json.dumps(result.get("full_simulation", {}).get("summary", {}), indent=1))


if __name__ == "__main__":
    main()
