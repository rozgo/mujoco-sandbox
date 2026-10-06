"""Largest thread timestep that passes the task-regime gates.

The 5 µs contact time constant, and therefore the 100 ns step, was chosen for a
0.34 m/s drop. At task speeds (mm/s placement, ~0.03 m/s settling contact) a
longer time constant may meet the same penetration gates at a much larger step.
This study measures that on the low-velocity settling fixture for both material
presets, both integrators and four time constants. Gates are unchanged:
< 1 µm under timestep halving, < 0.01 µm across unit systems, < 2 µm peak and
< 0.2 µm settled penetration, finite state, no warnings, actual contact.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import time

import mujoco
import numpy as np

from .contact_diagnostics import UNIT_SYSTEMS, difference, kernel
from .contact_laws import ContactLaw
from .contact_smoothing import trial
from .der import build_provenance
from .materials import MATERIALS
from .units import MM_G

TIME_CONSTANTS = (5e-6, 1e-5, 2e-5, 4e-5)
FRACTIONS = (2, 4, 8, 16)  # dt = tau/fraction; tau/2 is MuJoCo's refsafe limit
UNIT_FRACTION = 8
INTEGRATORS = ("RK4", "implicitfast")
DURATION_S = .06
GATES_UM = {"timestep": 1., "units": .01, "peak_penetration": 2., "settled_penetration": .2}


def law(tau):
    return ContactLaw(f"ramp_{tau*1e6:g}us", d0=mujoco.mjMINIMP, power=1., time_constant_s=tau)


def case_name(material, integrator, tau, fraction, units="mm_g_s"):
    return f"{material}_{integrator}_tau{tau*1e6:g}us_dt{fraction}_{units}"


def specs():
    out = []
    for material in MATERIALS:
        for integrator in INTEGRATORS:
            for tau in TIME_CONSTANTS:
                for fraction in FRACTIONS:
                    out.append((material, integrator, tau, fraction, MM_G))
                    if integrator == "RK4" and fraction == UNIT_FRACTION:
                        out += [(material, integrator, tau, fraction, u) for u in UNIT_SYSTEMS[1:]]
    return out


def settled_penetration_um(trace, config):
    final = trace["vertices"][-1, :, 2]-config["radius_m"]-config["floor_z_m"]
    return float(max(0., -final.min())*1e6)


def _worker(spec):
    material, integrator, tau, fraction, units = spec
    try:
        report, trace, xml = trial("der_settle", tau/fraction, units, law(tau), integrator,
                                   duration=DURATION_S, material=material)
    except Exception as exc:  # retain the failure, e.g. a rejected model
        return {"failure": f"{type(exc).__name__}: {exc}", "completed": False}, None, None
    return report, trace, xml


def run(output, jobs=1):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    path = output/"manifest.json"
    if path.exists():
        raise FileExistsError("Preserve previous runs: use a fresh directory")
    started = time.perf_counter()
    kernel()
    source = Path(__file__).parent
    report = {"status": "running", "started_utc": datetime.now(timezone.utc).isoformat(),
              "mujoco": mujoco.__version__, "backend": "MuJoCo CPU", "machine": platform.machine(),
              "plugin": build_provenance(), "fixture": "der_settle", "duration_s": DURATION_S,
              "materials": MATERIALS, "time_constants_s": TIME_CONSTANTS, "fractions": FRACTIONS,
              "integrators": INTEGRATORS, "gates_um": GATES_UM, "jobs": jobs,
              "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
                                [source/"thread_clock.py", source/"contact_smoothing.py", source/"contact_laws.py",
                                 source/"materials.py", source/"contact_kernel.cc", source/"der.py"]},
              "cases": {}, "comparisons": {}, "largest_passing_dt_s": {}, "ready_for_robot_or_rl": False,
              "timing_note": "Wall includes construction, stepping and telemetry across parallel workers."}
    traces = {}
    work = specs()

    def record(spec, result):
        material, integrator, tau, fraction, units = spec
        r, trace, xml = result
        name = case_name(material, integrator, tau, fraction, units.name)
        r.update(material=material, integrator=integrator, time_constant_s=tau, fraction=fraction,
                 dt_s=tau/fraction, units_name=units.name)
        if trace is not None:
            r["settled_penetration_um"] = settled_penetration_um(trace, r["config"])
            p = output/(name+".npz")
            np.savez_compressed(p, **trace)
            r["trajectory_sha256"] = hashlib.sha256(p.read_bytes()).hexdigest()
            traces[name] = trace
        report["cases"][name] = r
        path.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
        print(f"{name}: completed={r.get('completed')} peak={r.get('peak_penetration_um', float('nan')):.4g}um "
              f"settled={r.get('settled_penetration_um', float('nan')):.4g}um wall={r.get('wall_s', 0):.0f}s", flush=True)

    if jobs > 1:
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            for spec, result in zip(work, pool.map(_worker, work)):
                record(spec, result)
    else:
        for spec in work:
            record(spec, _worker(spec))

    def ok(name):
        c = report["cases"].get(name, {})
        return bool(c.get("completed") and c.get("first_contact_s") is not None)

    for material in MATERIALS:
        for integrator in INTEGRATORS:
            for tau in TIME_CONSTANTS:
                for coarse, fine in zip(FRACTIONS, FRACTIONS[1:]):
                    a, b = case_name(material, integrator, tau, coarse), case_name(material, integrator, tau, fine)
                    value = difference(traces[a], traces[b]) if a in traces and b in traces else None
                    pens = [report["cases"][x].get("peak_penetration_um", np.inf) for x in (a, b)]
                    settled = [report["cases"][x].get("settled_penetration_um", np.inf) for x in (a, b)]
                    report["comparisons"][f"{a}_vs_dt{fine}"] = {
                        "material": material, "integrator": integrator, "time_constant_s": tau,
                        "dt_s": tau/coarse, "reference_dt_s": tau/fine, "difference_um": value,
                        "peak_penetration_um": max(pens), "settled_penetration_um": max(settled),
                        "passed": bool(ok(a) and ok(b) and value is not None and value < GATES_UM["timestep"]
                                       and max(pens) < GATES_UM["peak_penetration"]
                                       and max(settled) < GATES_UM["settled_penetration"])}
                if integrator == "RK4":
                    base = case_name(material, integrator, tau, UNIT_FRACTION)
                    for u in UNIT_SYSTEMS[1:]:
                        other = case_name(material, integrator, tau, UNIT_FRACTION, u.name)
                        value = difference(traces[base], traces[other]) if base in traces and other in traces else None
                        report["comparisons"][f"{other}_units"] = {
                            "material": material, "integrator": integrator, "time_constant_s": tau,
                            "dt_s": tau/UNIT_FRACTION, "difference_um": value, "units": True,
                            "passed": bool(ok(base) and ok(other) and value is not None and value < GATES_UM["units"])}
        for integrator in INTEGRATORS:
            passing = [v["dt_s"] for v in report["comparisons"].values()
                       if v["material"] == material and v["integrator"] == integrator and not v.get("units") and v["passed"]]
            report["largest_passing_dt_s"][f"{material}_{integrator}"] = max(passing) if passing else None
    report["status"] = "clock_study_complete"
    report["elapsed_wall_s"] = time.perf_counter()-started
    report["simulated_s"] = sum(c.get("simulated_s", 0.) for c in report["cases"].values())
    report["finished_utc"] = datetime.now(timezone.utc).isoformat()
    path.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(json.dumps(report["largest_passing_dt_s"], indent=2), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run",))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=1)
    args = parser.parse_args()
    run(args.output, args.jobs)


if __name__ == "__main__":
    main()
