"""One additional contact refinement, preserving the prior case and contact law."""

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from sixlegs.neural_insertion.der import build_provenance
from sixlegs.neural_insertion.der_benchmarks import difference, trial
from sixlegs.neural_insertion.rescaling import CableConfig
from sixlegs.neural_insertion.units import Units


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--case", default="contact_fine")
    args = parser.parse_args()
    source = args.baseline/"manifest.json"
    baseline = json.loads(source.read_text())
    old = baseline["cases"][args.case]
    prior_path = args.baseline/(args.case+".npz")
    if hashlib.sha256(prior_path.read_bytes()).hexdigest() != old["trajectory_sha256"]:
        raise ValueError("Prior trajectory hash mismatch")
    if not old["completed"]:
        raise ValueError("Cannot refine against an incomplete baseline")
    plugin = build_provenance()
    if plugin != baseline["plugin"]:
        raise ValueError("Physics plugin must match the prior run")
    args.output.mkdir(parents=True, exist_ok=True)
    out = args.output/"manifest.json"
    if out.exists():
        raise FileExistsError("Use a new output directory")
    started = time.perf_counter()
    report = {"started_utc": datetime.now(timezone.utc).isoformat(), "status": "running",
              "baseline_manifest_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
              "baseline_case": args.case, "plugin": plugin,
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "cases": {}, "gates": {}}
    out.write_text(json.dumps(report, indent=2)+"\n")
    config = replace(CableConfig(**old["config"]), dt_s=old["config"]["dt_s"]/2)
    result, trace = trial(config, duration=old["duration_s"], units=Units(**old["units"]),
                          projection=old["projection"], sample_interval_s=old["sample_interval_s"])
    path = args.output/"contact_fine.npz"
    np.savez_compressed(path, **trace)
    result["trajectory_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    report["cases"]["contact_fine"] = result
    prior = np.load(prior_path)
    for field in ("tip", "vertices"):
        value = difference(prior, trace, field)
        report["gates"][field+"_refinement_um"] = {"value": value, "limit_exclusive": 1.,
            "passed": bool(result["completed"] and value is not None and value < 1.)}
    penetration = max(old["peak_penetration_um"], result["peak_penetration_um"])
    report["gates"]["peak_penetration_um"] = {"value": penetration, "limit_exclusive": 2., "passed": penetration < 2.}
    report["success"] = all(g["passed"] for g in report["gates"].values())
    report["status"] = "tested_gates_passed" if report["success"] else "quality_gate_failed"
    report["elapsed_wall_s"] = time.perf_counter()-started
    report["finished_utc"] = datetime.now(timezone.utc).isoformat()
    report["ready_for_robot_or_rl"] = False
    out.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({"status": report["status"], "gates": report["gates"], "elapsed_wall_s": report["elapsed_wall_s"]}, indent=2))
    raise SystemExit(0 if report["success"] else 1)


if __name__ == "__main__":
    main()
