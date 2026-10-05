"""Reproducible microscale cable gates; exits 1 when a quality gate fails."""

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import time

import mujoco
import numpy as np

from .rescaling import CableConfig, cable_xml, trial
from .scene import ROOT
from .units import SI, MM_KG, MM_G, CM_G


def matched_error_um(a, b, field="tip"):
    """Maximum Euclidean error at identical sampled physical timestamps."""
    if a[field].shape != b[field].shape or a["time"].shape != b["time"].shape:
        raise ValueError("Comparisons require equal trajectory shapes")
    if not len(a["time"]) or not np.allclose(a["time"], b["time"], rtol=0, atol=1e-9):
        raise ValueError("Comparisons require matched physical timestamps")
    return float(np.linalg.norm(a[field] - b[field], axis=-1).max() * 1e6)


def source_hashes():
    directory = Path(__file__).parent
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (directory / "benchmarks.py", directory / "rescaling.py",
                      directory / "units.py", directory / "scene.py")}


def run_suite(output, suite="all"):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "manifest.json").exists():
        raise FileExistsError("Use a new output directory to preserve existing results")
    started = time.perf_counter()
    manifest = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "mujoco_version": mujoco.__version__, "numpy_version": np.__version__,
        "backend": "standard MuJoCo CPU", "machine": platform.machine(),
        "os": platform.system(), "suite": suite,
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "source_files_sha256": source_hashes(),
        "initial_conditions": "Deterministic straight clamped rod; no randomization",
        "timing_note": "trial wall_s includes compilation, Python stepping and telemetry; not pure physics time",
        "cases": {}, "gates": {}, "status": "running", "success": False,
    }
    traces = {}

    def save():
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    def run(name, config, units=MM_G, **kwargs):
        report, trace = trial(config, units, **kwargs)
        report["case"] = name
        manifest["cases"][name] = report
        (output / (name + ".xml")).write_text(cable_xml(config, units))
        (output / (name + ".json")).write_text(json.dumps(report, indent=2) + "\n")
        if trace is not None:
            traces[name] = trace
            path = output / (name + ".npz")
            np.savez_compressed(path, **trace)
            report["trajectory_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        save()
        print(f"{name}: compiled={report['compiled']} simulation={report['simulated_s']:.6g}s "
              f"wall={report['wall_s']:.3f}s", flush=True)
        return report

    def gate(name, value, limit):
        manifest["gates"][name] = {"value": value, "limit_exclusive": limit,
                                    "passed": value is not None and value < limit}

    save()
    bend = CableConfig(length_m=.01, segments=16, gravity_m_s2=0, floor_z_m=-.02)
    if suite in ("all", "units"):
        for units in (SI, MM_KG, MM_G, CM_G):
            run("unit_" + units.name, bend, units, duration=.3, force_n=(0, 0, -1e-8))
        names = ["unit_" + u.name for u in (MM_KG, MM_G, CM_G)]
        error = max(matched_error_um(traces[names[0]], traces[n]) for n in names[1:]) if all(n in traces for n in names) else None
        gate("bending_unit_equivalence_um", error, .01)
    if suite in ("all", "bending"):
        reference = 1e-8 * bend.length_m**3 / (3 * bend.ei)
        manifest["cantilever_reference_m"] = reference
        for segments in (16, 32, 64):
            run(f"bend_{segments}", replace(bend, segments=segments),
                duration=2, force_n=(0, 0, -1e-8))
        fine = abs(manifest["cases"]["bend_64"].get("final_tip_m", [0, 0, float("inf")])[2])
        medium = abs(manifest["cases"]["bend_32"].get("final_tip_m", [0, 0, float("inf")])[2])
        gate("cantilever_reference_relative_error", abs(fine - reference) / reference, .03)
        gate("cantilever_mesh_relative_change", abs(fine - medium) / reference, .03)
    if suite in ("all", "contact"):
        # Fixed frictional drop, no bending damping. No tuning based on outcome.
        contact = CableConfig(dt_s=1.5625e-7, contact_time_s=1e-5, relaxation_s=0)
        for name, dt in (("contact_coarse", contact.dt_s), ("contact_fine", contact.dt_s / 2)):
            run(name, replace(contact, dt_s=dt), duration=.06, sample_interval_s=.0001)
        gate("contact_timestep_tip_error_um",
             matched_error_um(traces["contact_coarse"], traces["contact_fine"]), 1)
        gate("contact_timestep_shape_error_um",
             matched_error_um(traces["contact_coarse"], traces["contact_fine"], "vertices"), 1)
        gate("peak_contact_penetration_um", max(
            manifest["cases"][n]["peak_support_penetration_um"]
            for n in ("contact_coarse", "contact_fine")), 2)
        manifest["unvalidated"] = [
            "Settled contact: the 60 ms window includes rebound",
            "Contact spatial refinement, calibrated material/contact, tool and tissue interaction",
        ]
    dynamic = [r for name, r in manifest["cases"].items() if name != "unit_m_kg_s"]
    manifest["gates"]["numerical_completion"] = {
        "passed": all(r.get("completed", False) for r in dynamic),
        "note": "SI compilation is an explicit diagnostic, excluded from dynamic completion",
    }
    manifest["gates"]["physical_mass"] = {"passed": all(
        r.get("compiled") and np.isclose(r["total_cable_mass_kg"],
            CableConfig(**r["config"]).mass_kg, rtol=1e-12, atol=0) for r in dynamic)}
    gate("segment_length_error_nm", max(
        r.get("max_segment_length_error_m", float("inf")) for r in dynamic) * 1e9, 1)
    manifest["success"] = all(g["passed"] for g in manifest["gates"].values())
    manifest["status"] = "passed" if manifest["success"] else "quality_gate_failed"
    manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["elapsed_wall_s"] = time.perf_counter() - started
    manifest["total_simulated_s"] = sum(r["simulated_s"] for r in manifest["cases"].values())
    manifest["summed_trial_wall_s"] = sum(r["wall_s"] for r in manifest["cases"].values())
    save()
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=("all", "units", "bending", "contact"), default="all")
    parser.add_argument("--output", type=Path, required=True, help="New directory; existing manifests are preserved")
    args = parser.parse_args()
    report = run_suite(args.output, args.suite)
    print(json.dumps({"status": report["status"], "gates": report["gates"]}, indent=2))
    raise SystemExit(0 if report["success"] else 1)


if __name__ == "__main__":
    main()
