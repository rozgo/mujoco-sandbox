"""Continuous-entry contact law and RK4 stepping on the recorded contact fixtures.

The isolation and audit studies kept the baseline law (flat impedance 0.9999,
10 µs) and the first-order implicitfast integrator fixed. This experiment
changes only those two numerical choices, one at a time and together, on the
same fixtures, checkpoint, units, masses, dimensions and friction. It reports
the unchanged gates: < 1 µm under timestep halving, < 0.01 µm across unit
systems, < 2 µm peak penetration, finite state, no warnings, actual contact.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import time
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from .contact_audit import normal_acceleration, normal_fixed_step, normal_reference
from .contact_diagnostics import (CHECKPOINT, UNIT_SYSTEMS, centerline, checkpoint, difference,
                                  energy, kernel, model_xml, physical_signature)
from .contact_laws import INTEGRATORS, LAWS
from .der import build_provenance, numbers
from .rescaling import CableConfig
from .scene import ROOT
from .units import MM_G

RIGID = ("segment_normal", "segment_oblique")
CHAINS = ("chain_no_elastic", "der", "der_settle")
# der_settle: the same DER rod released straight and at rest with the support
# 50 µm below its lower surface. Tip impact is near 0.03 m/s and the rod then
# lies down and settles; this is closer to placing thread than the 6 mm drop.
SETTLE_CLEARANCE_M = 50e-6
DTS = (("coarse", 1.5625e-7), ("fine", 7.8125e-8))
FINER = ("finer", 3.90625e-8)
CANDIDATE = ("ramp_5us", "RK4")
BASELINE = ("flat", "implicitfast")
CONTROLS = (("flat", "RK4"), ("ramp_5us", "implicitfast"))
SEEDS = (7, 17, 29)
GATES_UM = {"timestep": 1., "units": .01, "penetration": 2.}


def fixture_config(kind, dt, friction=.3, law=LAWS["flat"], integrator="implicitfast"):
    """Base fixture kind and configuration; der_settle only moves the support."""
    case, state = checkpoint()
    c = replace(CableConfig(**case["config"]), dt_s=dt, friction=friction,
                contact_time_s=law.time_constant_s, integrator=integrator)
    if kind == "der_settle":
        return "der", replace(c, floor_z_m=-(c.radius_m+SETTLE_CLEARANCE_M)), state
    return kind, c, state


def variant_xml(kind, config, units, law, integrator):
    """Baseline fixture XML with only solimp/solref/integrator replaced."""
    if integrator not in INTEGRATORS:
        raise ValueError("Unsupported integrator")
    root = ET.fromstring(model_xml("der" if kind == "der_settle" else kind, config, units))
    root.set("model", f"contact_smoothing_{kind}_{law.name}_{integrator}")
    root.find("option").set("integrator", integrator)
    solimp, solref = numbers(law.solimp(units)), numbers(law.solref())
    for geom in list(root.find("default").iter("geom"))+list(root.find("worldbody").iter("geom")):
        geom.set("solimp", solimp)
        geom.set("solref", solref)
    return ET.tostring(root, encoding="unicode")


def load_variant(kind, dt, units=MM_G, law=LAWS["flat"], integrator="implicitfast", friction=.3):
    _, c, state = fixture_config(kind, dt, friction, law, integrator)
    xml = variant_xml(kind, c, units, law, integrator)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    if kind.startswith("segment"):
        if kind == "segment_oblique":
            data.qvel[0] = .05*units.length
    elif kind != "der_settle":  # der_settle keeps the compiled straight rest state
        data.qpos[:] = state["qpos"]  # ball-joint coordinates are unit independent
        data.qvel[:] = state["qvel"]
        if kind == "der":
            data.plugin_state[:] = state["plugin_state"]
    mujoco.mj_forward(model, data)
    if data.ncon:
        raise ValueError("Fixture starts before contact")
    return c, model, data, xml


def law_audit(kind, law, integrator, units=MM_G):
    """The variant must differ from the baseline model only in the intended fields."""
    base_kind, base_config, _ = fixture_config(kind, 7.8125e-8)
    base = mujoco.MjModel.from_xml_string(model_xml(base_kind, base_config, units))
    _, m, _, _ = load_variant(kind, 7.8125e-8, units, law, integrator)
    same = {}
    for field in ("body_mass", "body_inertia", "body_pos", "body_ipos", "geom_size", "geom_pos",
                  "geom_friction", "geom_condim", "dof_damping", "dof_armature", "opt.gravity", "opt.tolerance",
                  "opt.iterations", "opt.cone", "opt.solver", "opt.jacobian", "nplugin"):
        a = base.opt if field.startswith("opt.") else base
        b = m.opt if field.startswith("opt.") else m
        name = field.split(".")[-1]
        same[field] = bool(np.array_equal(np.asarray(getattr(a, name)), np.asarray(getattr(b, name))))
    expected = {"solimp": np.array(law.solimp(units)), "solref": np.array(law.solref())}
    ids = [g for g in range(m.ngeom) if m.geom_contype[g] or m.geom_conaffinity[g]]
    applied = {"solimp": bool(np.allclose(m.geom_solimp[ids], expected["solimp"], rtol=0, atol=0)),
               "solref": bool(np.allclose(m.geom_solref[ids], expected["solref"], rtol=0, atol=0)),
               "integrator": m.opt.integrator == (mujoco.mjtIntegrator.mjINT_RK4 if integrator == "RK4"
                                                  else mujoco.mjtIntegrator.mjINT_IMPLICITFAST)}
    return {"unchanged": same, "applied": applied,
            "passed": bool(all(same.values()) and all(applied.values()))}


def unit_audit(kind, law, integrator):
    signatures = {}
    for units in UNIT_SYSTEMS:
        _, m, d, _ = load_variant(kind, 1.5625e-7, units, law, integrator)
        signatures[units.name] = physical_signature(kind, m, d, units)
    reference = signatures[MM_G.name]
    result = {}
    for units in UNIT_SYSTEMS[1:]:
        errors = {}
        for name, a in reference.items():
            b = signatures[units.name][name]
            scale = max(float(np.max(abs(a))), 1e-30)
            errors[name] = float(np.max(abs(a-b))/scale)
        result[units.name] = {"maximum_relative_errors": errors, "passed": bool(max(errors.values()) < 1e-10)}
    return result


def frozen_law(law):
    """Same-state MuJoCo force versus the independent ramped formula, across units."""
    records = []
    for gap in (-.02e-6, -.3e-6, -.8e-6, -1.25e-6):
        for velocity in (-.343, 0., .02):
            expected = normal_acceleration(gap, velocity, law=law)
            cases = []
            for u in UNIT_SYSTEMS:
                c, m, d, _ = load_variant("segment_normal", 7.8125e-8, u, law, "implicitfast")
                d.qpos[2] = (c.floor_z_m+c.radius_m+gap)*u.length  # scratch evaluation only
                d.qvel[2] = velocity*u.length
                d.qacc_warmstart[:] = 0
                mujoco.mj_forward(m, d)
                measured = float(d.qacc[2]/u.length)
                cases.append({"units": u.name, "acceleration_m_s2": measured, "contacts": int(d.ncon),
                              "impedance": float(d.efc_KBIP[0, 2]) if d.nefc else None,
                              "acceleration_relative_error": abs(measured-expected)/max(9.81, abs(expected))})
            records.append({"gap_m": gap, "velocity_m_s": velocity, "independent_acceleration_m_s2": expected,
                            "independent_impedance": law.impedance(-gap), "cases": cases})
    maximum = max(c["acceleration_relative_error"] for r in records for c in r["cases"])
    return {"law": law.name, "cases": records, "maximum_relative_error": maximum, "gate": 1e-8,
            "passed": bool(maximum < 1e-8)}


def trial(kind, dt, units, law, integrator, duration=None, seed=None, epsilon=1e-12):
    """Live stepping only. A seed applies an initialization-only planar perturbation."""
    started = time.perf_counter()
    c, m, d, xml = load_variant(kind, dt, units, law, integrator)
    initial_change = 0.
    if seed is not None:
        before = centerline(kind, m, d, units)
        direction = np.zeros(m.nv)
        direction[1::3] = np.random.default_rng(seed).normal(size=m.nv//3)
        direction *= epsilon/np.linalg.norm(direction)
        mujoco.mj_integratePos(m, d.qpos, direction, 1.)
        mujoco.mj_forward(m, d)
        initial_change = float(np.linalg.norm(centerline(kind, m, d, units)-before, axis=1).max()*1e6)
    duration = (.03 if kind in ("chain_no_elastic", "der") else .06) if duration is None else duration
    sample_dt = .0001
    n, every = round(duration/dt), round(sample_dt/dt)
    if n < 1 or every < 1 or not math.isclose(n*dt, duration, rel_tol=1e-10) or not math.isclose(every*dt, sample_dt, rel_tol=1e-10):
        raise ValueError("Duration and sampling must match timestep")
    support = m.geom("support").id
    body = m.body("segment").id if kind.startswith("segment") else -1
    summary = np.zeros(15)
    first_contact = None
    trace = {k: [] for k in ("time", "vertices", "qpos", "qvel", "plugin_state", "energy_j")}

    def sample():
        trace["time"].append(float(d.time))
        trace["vertices"].append(centerline(kind, m, d, units))
        for name in ("qpos", "qvel", "plugin_state"):
            trace[name].append(getattr(d, name).copy())
        trace["energy_j"].append(energy(kind, m, d, units))

    sample()
    lib = kernel()
    for start in range(0, n, every):
        chunk = np.zeros(15)
        lib.contact_chunk(m._address, d._address, min(every, n-start), support, body, chunk)
        for index in (0, 3, 4, 5, 6, 7, 8, 9, 10):
            summary[index] += chunk[index]
        for index in (1, 2, 11, 14):
            summary[index] = max(summary[index], chunk[index])
        if first_contact is None and chunk[12] >= 0:
            first_contact = float(chunk[12])
        mujoco.mj_forward(m, d)
        sample()
        if chunk[14]:
            break
    trace = {k: np.array(v) for k, v in trace.items()}
    total = trace["energy_j"].sum(axis=1)
    # Positive normalization: largest recorded |kinetic|+|gravity|+|elastic| (zero at a straight rest start).
    scale = max(float(np.abs(trace["energy_j"]).sum(axis=1).max()), 1e-30)
    report = {"kind": kind, "config": asdict(c), "units": asdict(units), "law": asdict(law), "integrator": integrator,
              "seed": seed, "perturbation_norm_rad": epsilon if seed is not None else 0.,
              "initial_shape_change_um": initial_change, "duration_s": duration, "simulated_s": float(d.time),
              "completed": bool(not summary[14] and int(summary[0]) == n), "sample_interval_s": sample_dt,
              "initial_checkpoint_s": .03 if kind in ("chain_no_elastic", "der") else 0,
              "failure": "warning_or_nonfinite" if summary[14] else None,
              "peak_penetration_um": float(summary[1]/units.length*1e6),
              "peak_normal_force_n": float(summary[2]/units.force), "first_contact_s": first_contact,
              "touching_steps": int(summary[10]), "max_solver_iterations": int(summary[11]),
              "warnings": d.warning.number.tolist(),
              "max_energy_gain_fraction": float(max(0., np.max(total-total[0]))/scale),
              "final_energy_change_fraction": float((total[-1]-total[0])/scale),
              "scene_sha256": hashlib.sha256(xml.encode()).hexdigest(),
              "physical_total_mass_kg": float(m.body_mass[1:].sum()/units.mass),
              "wall_s": time.perf_counter()-started}
    if first_contact is not None:
        gap = trace["vertices"][:, :, 2].min(axis=1)-c.radius_m-c.floor_z_m
        after = trace["time"] > first_contact
        report["max_rebound_um"] = float(gap[after].max()*1e6)
        if kind == "der_settle":
            # Fraction of centerline points within one radius of the support at the end.
            final = trace["vertices"][-1, :, 2]-c.radius_m-c.floor_z_m
            report["final_points_within_radius_fraction"] = float(np.mean(final < c.radius_m))
            report["final_max_centerline_speed_m_s"] = float(np.linalg.norm(
                (trace["vertices"][-1]-trace["vertices"][-2])/(trace["time"][-1]-trace["time"][-2]), axis=1).max())
    return report, trace, xml


def case_name(kind, law, integrator, level, units, seed=None):
    name = f"{kind}_{law}_{integrator}_{units}_{level}"
    return name if seed is None else f"{name}_seed{seed}"


def _worker(spec):
    kind, dt, units, law, integrator, seed = spec
    report, trace, xml = trial(kind, dt, units, LAWS[law], integrator, seed=seed)
    return report, trace, xml


def matrix(kinds):
    """Predeclared cases: candidate everywhere, controls and baseline in mm–g–s."""
    specs = []
    for kind in kinds:
        combos = [(law, integrator, UNIT_SYSTEMS) for law in ("flat", "ramp_5us") for integrator in INTEGRATORS] \
            if kind in RIGID else [(*CANDIDATE, UNIT_SYSTEMS), (*BASELINE, (MM_G,)), *[(*c, (MM_G,)) for c in CONTROLS]]
        for law, integrator, systems in combos:
            for units in systems:
                for level, dt in DTS:
                    specs.append((kind, dt, units, law, integrator, None, level))
    if "der" in kinds:
        # Third halving for the candidate: a refinement trend, not a single pair.
        specs.append(("der", FINER[1], MM_G, *CANDIDATE, None, FINER[0]))
        for seed in SEEDS:
            specs.append(("der", DTS[1][1], MM_G, *CANDIDATE, seed, DTS[1][0]))
    return specs


def reference_comparison(report, traces, output):
    """Rigid normal drops against the independent continuous and discrete references."""
    times = np.arange(601)*.0001
    result = {}
    for law_name in ("flat", "ramp_5us"):
        law = LAWS[law_name]
        standard, receipt = normal_reference(times, law=law)
        tight, tight_receipt = normal_reference(times, tighter=True, law=law)
        np.savez_compressed(output/f"normal_reference_{law_name}.npz", time=times, standard=standard, tighter=tight)
        entry = {"continuous": {"standard": receipt, "tighter": tight_receipt,
                                "refinement_difference_um": float(np.max(abs(standard[:, 0]-tight[:, 0]))*1e6)},
                 "comparisons": {}}
        for integrator in INTEGRATORS:
            for level, dt in DTS:
                name = case_name("segment_normal", law_name, integrator, level, MM_G.name)
                if name not in traces:
                    continue
                gap = traces[name]["qpos"][:, 2]/MM_G.length-(-.006+20e-6)
                recurrence, discrete = normal_fixed_step(dt, law=law, method="euler" if integrator == "implicitfast" else "rk4")
                entry["comparisons"][name] = {
                    "timestep_s": dt, "integrator": integrator,
                    "mujoco_vs_continuous_um": float(np.max(abs(gap-tight[:, 0]))*1e6),
                    "mujoco_vs_independent_recurrence_um": float(np.max(abs(gap-recurrence[:, 0]))*1e6),
                    "recurrence_vs_continuous_um": float(np.max(abs(recurrence[:, 0]-tight[:, 0]))*1e6),
                    "independent_recurrence": discrete}
        result[law_name] = entry
    return result


def run(output, kinds=RIGID+CHAINS, jobs=1):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    path = output/"manifest.json"
    if path.exists():
        raise FileExistsError("Preserve previous runs: use a fresh directory")
    started = time.perf_counter()
    source = Path(__file__).parent
    kernel()  # build once before any worker starts
    report = {"status": "running", "started_utc": datetime.now(timezone.utc).isoformat(),
              "mujoco": mujoco.__version__, "backend": "MuJoCo CPU", "machine": platform.machine(),
              "plugin": build_provenance(),
              "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
                                [source/"contact_smoothing.py", source/"contact_laws.py", source/"contact_kernel.cc",
                                 source/"contact_audit.py", source/"contact_diagnostics.py", source/"der.py", source/"units.py"]},
              "checkpoint_sha256": hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest(),
              "laws": {k: asdict(v) for k, v in LAWS.items()}, "candidate": CANDIDATE, "baseline": BASELINE,
              "controls": CONTROLS, "gates_um": GATES_UM, "jobs": jobs,
              "protocol": "Same fixtures, checkpoint, masses, dimensions, friction and tolerance as the isolation study; "
                          "only solimp/solref and the integrator change. Candidate runs in all unit systems; "
                          "baseline/controls in mm-g-s. Three predeclared DER perturbation seeds at the fine step.",
              "law_audits": {}, "unit_audits": {}, "frozen_law": frozen_law(LAWS["ramp_5us"]),
              "cases": {}, "comparisons": {}, "sensitivity": {}, "ready_for_robot_or_rl": False,
              "timing_note": "Wall includes construction, native stepping and telemetry across parallel workers; not isolated compute time."}
    for kind in kinds:
        for law, integrator in ((*CANDIDATE,), (*BASELINE,), *CONTROLS):
            report["law_audits"][f"{kind}_{law}_{integrator}"] = law_audit(kind, LAWS[law], integrator)
        report["unit_audits"][kind] = unit_audit(kind, LAWS[CANDIDATE[0]], CANDIDATE[1])
    if not all(v["passed"] for v in report["law_audits"].values()) or \
            not all(v["passed"] for k in report["unit_audits"].values() for v in k.values()) or not report["frozen_law"]["passed"]:
        report["status"] = "compiled_audit_failed"
        path.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
        raise ValueError("Variant changes more than the intended fields or the ramped law disagrees with the formula")
    path.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    specs = matrix(kinds)
    traces = {}

    def record(spec, result):
        kind, dt, units, law, integrator, seed, level = spec
        result_report, trace, xml = result
        name = case_name(kind, law, integrator, level, units.name, seed)
        traj_path = output/(name+".npz")
        np.savez_compressed(traj_path, **trace)
        (output/(name+".xml")).write_text(xml)
        result_report["trajectory_sha256"] = hashlib.sha256(traj_path.read_bytes()).hexdigest()
        report["cases"][name] = result_report
        traces[name] = trace
        path.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
        print(f"{name}: completed={result_report['completed']} penetration={result_report['peak_penetration_um']:.4g}um "
              f"iterations={result_report['max_solver_iterations']} wall={result_report['wall_s']:.1f}s", flush=True)

    work = [(s[0], s[1], s[2], s[3], s[4], s[5]) for s in specs]
    if jobs > 1:
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            for spec, result in zip(specs, pool.map(_worker, work)):
                record(spec, result)
    else:
        for spec, w in zip(specs, work):
            record(spec, _worker(w))
    for kind in kinds:
        combos = [(law, integrator) for law in ("flat", "ramp_5us") for integrator in INTEGRATORS] if kind in RIGID \
            else [CANDIDATE, BASELINE, *CONTROLS]
        for law, integrator in combos:
            gated = (law, integrator) == CANDIDATE
            systems = UNIT_SYSTEMS if (kind in RIGID or gated) else (MM_G,)
            for units in systems:
                a, b = [case_name(kind, law, integrator, level, units.name) for level, _ in DTS]
                value = difference(traces[a], traces[b])
                report["comparisons"][f"{kind}_{law}_{integrator}_{units.name}_timestep"] = {
                    "difference_um": value, "cases": [a, b], "gate_um": GATES_UM["timestep"], "gated": gated,
                    "passed": bool(value is not None and value < GATES_UM["timestep"] and all(
                        report["cases"][x]["completed"] and report["cases"][x]["first_contact_s"] is not None for x in (a, b)))}
            if len(systems) > 1:
                for alternative in UNIT_SYSTEMS[1:]:
                    for level, _ in DTS:
                        a, b = [case_name(kind, law, integrator, level, u.name) for u in (MM_G, alternative)]
                        value = difference(traces[a], traces[b])
                        report["comparisons"][f"{kind}_{law}_{integrator}_{alternative.name}_{level}_units"] = {
                            "difference_um": value, "cases": [a, b], "gate_um": GATES_UM["units"], "gated": gated,
                            "passed": bool(value is not None and value < GATES_UM["units"] and all(
                                report["cases"][x]["completed"] and report["cases"][x]["first_contact_s"] is not None for x in (a, b)))}
            peak = max(report["cases"][case_name(kind, law, integrator, level, u.name)]["peak_penetration_um"]
                       for u in systems for level, _ in DTS)
            report["comparisons"][f"{kind}_{law}_{integrator}_penetration"] = {
                "peak_penetration_um": peak, "gate_um": GATES_UM["penetration"], "gated": gated,
                "passed": bool(peak < GATES_UM["penetration"])}
    if "der" in kinds:
        reference = traces[case_name("der", *CANDIDATE, DTS[1][0], MM_G.name)]
        finer = case_name("der", *CANDIDATE, FINER[0], MM_G.name)
        value = difference(reference, traces[finer])
        report["comparisons"]["der_"+"_".join(CANDIDATE)+"_mm_g_s_timestep_finer"] = {
            "difference_um": value, "cases": [case_name("der", *CANDIDATE, DTS[1][0], MM_G.name), finer],
            "gate_um": GATES_UM["timestep"], "gated": True,
            "passed": bool(value is not None and value < GATES_UM["timestep"] and report["cases"][finer]["completed"])}
        for seed in SEEDS:
            name = case_name("der", *CANDIDATE, DTS[1][0], MM_G.name, seed)
            trace = traces[name]
            error = np.linalg.norm(trace["vertices"]-reference["vertices"], axis=-1).max(axis=1)*1e6
            report["sensitivity"][name] = {
                "seed": seed, "max_shape_difference_um": float(error.max()),
                "initial_shape_change_um": report["cases"][name]["initial_shape_change_um"],
                "threshold_crossings_s": {str(t): (float(trace["time"][np.flatnonzero(error > t)[0]]) if np.any(error > t) else None)
                                          for t in (.001, .01, 1.)}}
    if "segment_normal" in kinds:
        report["independent_reference"] = reference_comparison(report, traces, output)
    gated = [v for v in report["comparisons"].values() if v["gated"]]
    report["success"] = bool(gated) and all(v["passed"] for v in gated)
    report["status"] = "candidate_gates_passed" if report["success"] else "candidate_gates_failed"
    report["elapsed_wall_s"] = time.perf_counter()-started
    report["simulated_s"] = sum(c["simulated_s"] for c in report["cases"].values())
    report["finished_utc"] = datetime.now(timezone.utc).isoformat()
    report["ready_for_robot_or_rl"] = False
    report["unvalidated"] = ["Settled support contact and release", "Physical contact/material calibration",
                             "Tool grasp and tissue interaction", "Robot-scale timestep coupling"]
    path.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(json.dumps({k: v for k, v in report["comparisons"].items() if v["gated"]}, indent=2), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run",))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--kinds", nargs="+", choices=RIGID+CHAINS, default=list(RIGID+CHAINS))
    parser.add_argument("--jobs", type=int, default=1)
    args = parser.parse_args()
    result = run(args.output, tuple(args.kinds), args.jobs)
    raise SystemExit(0 if result["success"] else 1)


if __name__ == "__main__":
    main()
