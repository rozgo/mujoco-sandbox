"""Two predefined extra halvings of the failed rigid normal-drop control."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from sixlegs.neural_insertion.contact_diagnostics import difference, trial


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output/"manifest.json"
    if path.exists():
        raise FileExistsError("Use a fresh output directory")
    baseline_path = args.baseline/"manifest.json"
    baseline = json.loads(baseline_path.read_text())
    key = "segment_normal_mm_g_s_fine"
    trajectory = args.baseline/(key+".npz")
    assert hashlib.sha256(trajectory.read_bytes()).hexdigest() == baseline["cases"][key]["trajectory_sha256"]
    with np.load(trajectory) as saved:
        previous = {k: saved[k] for k in ("vertices", "time")}
    started = time.perf_counter()
    report = {"started_utc": datetime.now(timezone.utc).isoformat(), "cases": {}, "comparisons": {},
              "status": "running", "baseline_case_trajectory_sha256": baseline["cases"][key]["trajectory_sha256"],
              "source_sha256": baseline["source_sha256"],
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "protocol": "Selected after initial normal-drop failure; exactly two extra timestep halvings in mm-g-s. No changes to physics or gates."}
    for dt in (3.90625e-8, 1.953125e-8):
        name = f"normal_dt_{dt*1e9:g}_ns"
        result, trace, xml = trial("segment_normal", dt)
        npz = args.output/(name+".npz")
        np.savez_compressed(npz, **trace)
        (args.output/(name+".xml")).write_text(xml)
        result["trajectory_sha256"] = hashlib.sha256(npz.read_bytes()).hexdigest()
        report["cases"][name] = result
        value = difference(previous, trace)
        report["comparisons"][name] = {"difference_um": value, "gate_um": 1.,
            "passed": bool(value is not None and value < 1 and result["completed"] and result["first_contact_s"] is not None)}
        previous = trace
        path.write_text(json.dumps(report, indent=2)+"\n")
        print(name, report["comparisons"][name], flush=True)
    report["success"] = all(c["passed"] for c in report["comparisons"].values())
    report["status"] = "refinement_comparisons_passed" if report["success"] else "refinement_comparisons_failed"
    report["elapsed_wall_s"] = time.perf_counter()-started
    report["simulated_s"] = sum(c["simulated_s"] for c in report["cases"].values())
    report["finished_utc"] = datetime.now(timezone.utc).isoformat()
    report["ready_for_robot_or_rl"] = False
    path.write_text(json.dumps(report, indent=2)+"\n")
    raise SystemExit(0 if report["success"] else 1)


if __name__ == "__main__":
    main()
