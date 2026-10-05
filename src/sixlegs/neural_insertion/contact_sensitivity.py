"""Predeclared tiny-perturbation and solver-tolerance controls for the DER chain."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import mujoco
import numpy as np

from .contact_audit import checked_trace
from .contact_diagnostics import load_fixture, kernel, centerline
from .units import MM_G


def trial(seed=None, epsilon=1e-12, tolerance=1e-12):
    started = time.perf_counter()
    _, m, d, xml = load_fixture("der", 7.8125e-8)
    m.opt.tolerance = tolerance
    before = centerline("der", m, d, MM_G)
    direction = np.zeros(m.nv)
    if seed is not None:
        # Initialization only. Perturb local Y rotations, preserving planar motion.
        direction[1::3] = np.random.default_rng(seed).normal(size=m.nv//3)
        direction *= epsilon/np.linalg.norm(direction)
        mujoco.mj_integratePos(m, d.qpos, direction, 1.)
        mujoco.mj_forward(m, d)
    initial_change = float(np.linalg.norm(centerline("der", m, d, MM_G)-before, axis=1).max()*1e6)
    trace = {k: [] for k in ("time", "vertices", "qpos", "qvel", "plugin_state")}

    def sample():
        trace["time"].append(float(d.time))
        trace["vertices"].append(centerline("der", m, d, MM_G))
        for k in ("qpos", "qvel", "plugin_state"):
            trace[k].append(getattr(d, k).copy())

    sample()
    steps, every = round(.03/m.opt.timestep), round(.0001/m.opt.timestep)
    max_pen, max_iter, first_contact = 0., 0, None
    failure = None
    for _ in range(0, steps, every):
        out = np.zeros(15)
        kernel().contact_chunk(m._address, d._address, every, m.geom("support").id, -1, out)
        max_pen = max(max_pen, out[1]/MM_G.length*1e6)
        max_iter = max(max_iter, int(out[11]))
        if first_contact is None and out[12] >= 0:
            first_contact = float(out[12])
        mujoco.mj_forward(m, d)
        sample()
        if out[14]:
            failure = "warning_or_nonfinite"
            break
    return {"seed": seed, "perturbation_norm_rad": epsilon if seed is not None else 0.,
            "initial_shape_change_um": initial_change, "tolerance": tolerance,
            "completed": failure is None, "failure": failure, "warnings": d.warning.number.tolist(),
            "peak_penetration_um": float(max_pen), "max_solver_iterations": max_iter,
            "first_contact_s": first_contact, "simulated_s": float(d.time),
            "wall_s": time.perf_counter()-started,
            "base_scene_sha256": hashlib.sha256(xml.encode()).hexdigest()}, {k: np.array(v) for k, v in trace.items()}


def run(baseline, output):
    baseline, output = Path(baseline), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    path = output/"manifest.json"
    if path.exists():
        raise FileExistsError("Use a fresh output directory")
    started = time.perf_counter()
    manifest = json.loads((baseline/"manifest.json").read_text())
    reference = checked_trace(baseline, manifest, "der_mm_g_s_fine")
    report = {"status": "running", "started_utc": datetime.now(timezone.utc).isoformat(),
              "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "baseline_manifest_sha256": hashlib.sha256((baseline/"manifest.json").read_bytes()).hexdigest(),
              "protocol": "Three predetermined planar perturbations: seeds 7,17,29; norm 1e-12 rad; same units and timestep. One unperturbed tolerance=0 control.",
              "cases": {}, "ready_for_robot_or_rl": False}
    for seed in (7, 17, 29, None):
        name = f"perturb_seed_{seed}" if seed is not None else "no_tolerance_exit"
        result, trace = trial(seed=seed, tolerance=1e-12 if seed is not None else 0.)
        p = output/(name+".npz")
        np.savez_compressed(p, **trace)
        result["trajectory_sha256"] = hashlib.sha256(p.read_bytes()).hexdigest()
        if result["completed"]:
            np.testing.assert_allclose(trace["time"], reference["time"], atol=1e-9, rtol=0)
            error = np.linalg.norm(trace["vertices"]-reference["vertices"], axis=-1).max(axis=1)*1e6
            result["max_shape_difference_um"] = float(error.max())
            result["threshold_crossings_s"] = {str(threshold): float(trace["time"][np.flatnonzero(error > threshold)[0]])
                                              if np.any(error > threshold) else None for threshold in (.001, .01, 1.)}
        report["cases"][name] = result
        path.write_text(json.dumps(report, indent=2)+"\n")
        print(name, result.get("max_shape_difference_um"), "um", result["wall_s"], "s wall", flush=True)
    report["status"] = "sensitivity_audit_complete_accuracy_unaccepted"
    report["simulated_s"] = sum(c["simulated_s"] for c in report["cases"].values())
    report["elapsed_wall_s"] = time.perf_counter()-started
    report["finished_utc"] = datetime.now(timezone.utc).isoformat()
    path.write_text(json.dumps(report, indent=2)+"\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    run(args.baseline, args.output)


if __name__ == "__main__":
    main()
