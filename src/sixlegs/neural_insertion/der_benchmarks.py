"""Recorded DER acceptance gates; successful execution is not physics acceptance."""

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import subprocess
import time

import mujoco
import numpy as np

from .der import build_provenance, der_xml, elastic_metrics, load_der, preview, vertices
from .rescaling import CableConfig, load_cable
from .units import MM_G, CM_G, MM_KG


def audit(projection="direct"):
    """Virtual-work and lifecycle checks; pose writes are scratch calculations."""
    c = CableConfig(length_m=.01, segments=8, gravity_m_s2=0, relaxation_s=0, floor_z_m=-.02)
    m, d = load_der(c, projection=projection)
    cases = {}
    vectors = {name: np.zeros(m.nv) for name in ("bend", "twist", "mixed")}
    vectors["bend"][11] = .5
    vectors["twist"][9] = .5
    vectors["mixed"][11] = .5
    vectors["mixed"][18] = .3
    for seed in (7, 17, 29):
        vectors[f"spatial_seed_{seed}"] = np.random.default_rng(seed).normal(0, .12, m.nv)
    for name, velocity in vectors.items():
        mujoco.mj_resetData(m, d)
        mujoco.mj_integratePos(m, d.qpos, velocity, 1.)
        mujoco.mj_forward(m, d)
        q = d.qpos.copy()
        force = d.qfrc_passive.copy()/MM_G.torque
        energy = elastic_metrics(m, d)
        errors = []
        gradients = []
        for epsilon in (1e-5, 1e-6):
            gradient = np.zeros(m.nv)
            for i in range(m.nv):
                direction = np.zeros(m.nv)
                direction[i] = 1
                energies = []
                for sign in (-1, 1):
                    d.qpos[:] = q
                    mujoco.mj_integratePos(m, d.qpos, direction, sign*epsilon)
                    mujoco.mj_forward(m, d)
                    e = elastic_metrics(m, d)
                    energies.append(e["bending_j"]+e["twisting_j"])
                gradient[i] = -(energies[1]-energies[0])/(2*epsilon)
            errors.append(float(np.linalg.norm(force-gradient)/np.linalg.norm(gradient)))
            gradients.append(gradient.tolist())
        cases[name] = {"energy": energy, "force_nm": force.tolist(),
                       "negative_energy_gradient_nm": gradients,
                       "difference_step_rad": [1e-5, 1e-6], "relative_error": errors}

    # A separate evolving case exercises plugin history, copy and restart state.
    c = replace(c, dt_s=1e-7)
    m, d = load_der(c, projection=projection)
    mujoco.mj_integratePos(m, d.qpos, vectors["mixed"], 1.)
    mujoco.mj_step(m, d, nstep=10)
    mujoco.mj_forward(m, d)
    reference = d.qfrc_passive.copy()
    state_reference = d.plugin_state.copy()
    for _ in range(10):
        mujoco.mj_forward(m, d)
    repeated_error = float(np.max(np.abs(d.qfrc_passive-reference))/MM_G.torque)
    query_state_error = float(np.max(np.abs(d.plugin_state-state_reference)))
    copied = mujoco.MjData(m)
    mujoco.mj_copyData(copied, m, d)
    mujoco.mj_forward(m, copied)
    copied_error = float(np.max(np.abs(copied.qfrc_passive-reference))/MM_G.torque)
    spec = mujoco.mjtState.mjSTATE_INTEGRATION
    saved = np.zeros(mujoco.mj_stateSize(m, spec))
    mujoco.mj_getState(m, d, saved, spec)
    restored = mujoco.MjData(m)
    mujoco.mj_setState(m, restored, saved, spec)
    mujoco.mj_forward(m, restored)
    restore_error = float(np.max(np.abs(restored.qfrc_passive-reference))/MM_G.torque)
    mujoco.mj_resetData(m, d)
    mujoco.mj_forward(m, d)
    fresh = mujoco.MjData(m)
    mujoco.mj_forward(m, fresh)
    reset_error = float(np.max(np.abs(d.qfrc_passive-fresh.qfrc_passive))/MM_G.torque)

    # Unit tests use binary-exact spacing and identical generalized coordinates.
    unit_forces, unit_vertices = [], []
    for units in (MM_G, CM_G, MM_KG):
        um, ud = load_der(c, units, projection=projection)
        mujoco.mj_integratePos(um, ud.qpos, vectors["mixed"], 1.)
        mujoco.mj_forward(um, ud)
        unit_forces.append(ud.qfrc_passive.copy()/units.torque)
        unit_vertices.append(vertices(um, ud, units))
    unit_relative = max(float(np.linalg.norm(f-unit_forces[0])/np.linalg.norm(unit_forces[0]))
                        for f in unit_forces[1:])
    native, _ = load_cable(c)
    mass_error = float(np.max(np.abs(native.body_mass-m.body_mass[:-1]))/MM_G.mass)
    inertia_error = float(np.max(np.abs(native.body_inertia-m.body_inertia[:-1]))/MM_G.torque)
    return {"projection": projection, "config": asdict(c), "virtual_work": cases,
            "max_virtual_work_relative_error": max(max(v["relative_error"]) for v in cases.values()),
            "repeat_force_error_nm": repeated_error, "copy_force_error_nm": copied_error,
            "restore_force_error_nm": restore_error, "reset_force_error_nm": reset_error,
            "query_plugin_state_error": query_state_error, "unit_force_relative_error": unit_relative,
            "unit_vertex_difference_m": max(float(np.max(np.abs(v-unit_vertices[0]))) for v in unit_vertices[1:]),
            "native_mass_max_difference_kg": mass_error, "native_inertia_max_difference_kg_m2": inertia_error,
            "audit_simulated_s": 10*c.dt_s}


def trial(config, *, duration, units=MM_G, projection="direct", force_n=(0, 0, 0),
          release_at=None, initial_bend=0., sample_interval_s=.0001):
    """Live stepping only; initial bend is an explicit reset, not a controller."""
    c = config
    steps = round(duration/c.dt_s)
    sample_steps = round(sample_interval_s/c.dt_s)
    if steps < 1 or sample_steps < 1 or not math.isclose(steps*c.dt_s, duration, rel_tol=1e-10):
        raise ValueError("Duration must match the step grid")
    if not math.isclose(sample_steps*c.dt_s, sample_interval_s, rel_tol=1e-10):
        raise ValueError("Sampling must match the step grid")
    if c.integrator != "implicitfast":
        raise ValueError("This acceptance fixture uses implicitfast")
    started = time.perf_counter()
    m, d = load_der(c, units, projection=projection)
    if initial_bend:
        velocity = np.zeros(m.nv)
        velocity[2::3] = initial_bend/(c.segments-1)
        mujoco.mj_integratePos(m, d.qpos, velocity, 1.)
    mujoco.mj_forward(m, d)
    ids = np.array([i for i in range(m.nbody) if m.body(i).name.startswith("thread_B")])
    support = m.geom("support").id
    end = m.body("thread_B_last").id
    site = m.site("thread_S_last").id
    force = np.asarray(force_n)*units.force
    trace = {name: [] for name in ("time", "vertices", "qpos", "qvel", "plugin_state", "energy_j",
                                   "tip", "support_force_n", "load_active")}
    max_pen = 0.
    final_quarter_pen = 0.
    max_niter = 0
    max_segment_error = 0.
    failed = None
    contact_force = np.zeros(6)

    def sample(active):
        nonlocal max_segment_error
        points = vertices(m, d, units)
        trace["time"].append(float(d.time))
        trace["vertices"].append(points)
        trace["tip"].append(points[-1])
        for name in ("qpos", "qvel", "plugin_state"):
            trace[name].append(getattr(d, name).copy())
        e = elastic_metrics(m, d, units)
        mv = np.zeros(m.nv)
        mujoco.mj_mulM(m, d, mv, d.qvel)
        kinetic = .5*d.qvel@mv/units.torque
        gravitational = -np.sum(m.body_mass[:, None]*d.xipos*m.opt.gravity)/units.torque
        trace["energy_j"].append([e["bending_j"], e["twisting_j"], kinetic, gravitational])
        reaction = 0.
        for i, contact in enumerate(d.contact):
            if support in (contact.geom1, contact.geom2):
                mujoco.mj_contactForce(m, d, i, contact_force)
                reaction += contact_force[0]/units.force
        trace["support_force_n"].append(reaction)
        trace["load_active"].append(active)
        max_segment_error = max(max_segment_error, float(np.max(np.abs(
            np.linalg.norm(np.diff(points, axis=0), axis=1)-c.length_m/c.segments))))

    sample(release_at is None or release_at > 0)
    for step in range(steps):
        mujoco.mj_step1(m, d)
        d.qfrc_applied[:] = 0
        active = release_at is None or step*c.dt_s < release_at
        if active:
            mujoco.mj_applyFT(m, d, force, np.zeros(3), d.site_xpos[site], end, d.qfrc_applied)
        for contact in d.contact:
            if support in (contact.geom1, contact.geom2):
                penetration = max(0., -contact.dist/units.length)
                max_pen = max(max_pen, penetration)
                if step*c.dt_s > .75*duration:
                    final_quarter_pen = max(final_quarter_pen, penetration)
        mujoco.mj_step2(m, d)
        max_niter = max(max_niter, int(np.max(d.solver_niter)))
        if np.any(d.warning.number) or not np.isfinite(d.qpos).all() or not np.isfinite(d.qvel).all():
            failed = "warning_or_nonfinite"
            break
        if (step+1) % sample_steps == 0 or step+1 == steps:
            mujoco.mj_forward(m, d)
            sample(active)
    # Preserve the actual partial trajectory on any failure.
    report = {"config": asdict(c), "units": asdict(units), "projection": projection,
              "duration_s": duration, "simulated_s": float(d.time), "wall_s": time.perf_counter()-started,
              "force_n": list(force_n), "release_at_s": release_at, "initial_bend_rad": initial_bend,
              "sample_interval_s": sample_interval_s, "failure": failed,
              "completed": failed is None and math.isclose(d.time, duration, rel_tol=1e-8),
              "warnings": d.warning.number.tolist(), "max_solver_iterations": max_niter,
              "peak_penetration_um": max_pen*1e6, "final_quarter_penetration_um": final_quarter_pen*1e6,
              "max_segment_length_error_m": max_segment_error,
              "physical_mass_kg": float(m.body_mass[ids].sum()/units.mass),
              "scene_sha256": hashlib.sha256(der_xml(c, units, projection=projection).encode()).hexdigest()}
    return report, {key: np.asarray(value) for key, value in trace.items()}


def difference(a, b, field="vertices"):
    if a[field].shape != b[field].shape or not np.allclose(a["time"], b["time"], rtol=0, atol=1e-9):
        return None
    return float(np.max(np.linalg.norm(a[field]-b[field], axis=-1))*1e6)


def run(output, suite="all"):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output/"manifest.json"
    if manifest_path.exists():
        raise FileExistsError("Use a fresh output directory")
    start = time.perf_counter()
    source = Path(__file__).parent
    report = {"started_utc": datetime.now(timezone.utc).isoformat(), "suite": suite,
              "mujoco": mujoco.__version__, "backend": "MuJoCo CPU + local DER plugin",
              "machine": platform.machine(), "os": platform.system(),
              "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
              "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in [source/"der.py", source/"der_benchmarks.py", source/"rescaling.py", source/"units.py"]},
              "plugin": build_provenance(), "gates": {}, "cases": {}, "status": "running"}
    traces = {}

    def save():
        manifest_path.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")

    def gate(name, value, limit):
        report["gates"][name] = {"value": value, "limit_exclusive": limit,
                                 "passed": bool(value is not None and value < limit)}

    def case(name, config, **kwargs):
        r, t = trial(config, **kwargs)
        traces[name] = t
        trajectory = output/(name+".npz")
        np.savez_compressed(trajectory, **t)
        r["trajectory_sha256"] = hashlib.sha256(trajectory.read_bytes()).hexdigest()
        report["cases"][name] = r
        save()
        print(f"{name}: complete={r['completed']} wall={r['wall_s']:.2f}s penetration={r['peak_penetration_um']:.6g}um", flush=True)
        return r, t

    if suite in ("all", "audit"):
        report["audit"] = {p: audit(p) for p in ("published", "direct")}
        a = report["audit"]["direct"]
        gate("virtual_work_relative_error", a["max_virtual_work_relative_error"], 1e-4)
        for name in ("repeat_force_error_nm", "copy_force_error_nm", "restore_force_error_nm", "reset_force_error_nm"):
            gate(name, a[name], 1e-15)
        gate("unit_force_relative_error", a["unit_force_relative_error"], 1e-9)
        gate("query_plugin_state_error", a["query_plugin_state_error"], 1e-15)
        gate("mass_difference_kg", a["native_mass_max_difference_kg"], 1e-20)
        gate("inertia_difference_kg_m2", a["native_inertia_max_difference_kg_m2"], 1e-25)
        save()
        print("force/lifecycle audit saved", flush=True)
    if suite in ("all", "bending"):
        base = CableConfig(length_m=.01, gravity_m_s2=0, floor_z_m=-.02, dt_s=2.5e-5)
        analytical = 1e-8*base.length_m**3/(3*base.ei)
        for n in (16, 32, 64):
            case(f"bend_{n}", replace(base, segments=n), duration=2., force_n=(0, 0, -1e-8), sample_interval_s=.002)
        fine = abs(traces["bend_64"]["tip"][-1, 2])
        medium = abs(traces["bend_32"]["tip"][-1, 2])
        gate("cantilever_relative_error", abs(fine-analytical)/analytical, .03)
        gate("cantilever_mesh_relative_change", abs(fine-medium)/analytical, .03)
        report["cantilever_analytical_m"] = analytical
    if suite in ("all", "relaxation"):
        base = CableConfig(length_m=.01, segments=16, gravity_m_s2=0, floor_z_m=-.02, relaxation_s=0)
        drift = []
        for name, dt in (("relax_coarse", 2e-6), ("relax_fine", 1e-6)):
            r, t = case(name, replace(base, dt_s=dt), duration=.01, initial_bend=.3)
            energy = t["energy_j"].sum(axis=1)
            drift.append(float(np.max(np.abs(energy-energy[0]))/energy[0]) if r["completed"] else None)
        gate("relaxation_energy_relative_drift", drift[-1], .001)
        report["relaxation_energy_drift_coarse_fine"] = drift
        report["gates"]["relaxation_energy_refines"] = {"passed": all(v is not None for v in drift) and drift[1] < drift[0]}
        gate("relaxation_timestep_shape_um", difference(traces["relax_coarse"], traces["relax_fine"]), 1.)
    if suite in ("all", "contact"):
        base = CableConfig(relaxation_s=0, contact_time_s=1e-5)
        for name, dt in (("contact_medium", 3.125e-7), ("contact_coarse", 1.5625e-7), ("contact_fine", 7.8125e-8)):
            case(name, replace(base, dt_s=dt), duration=.06)
        errors = [difference(traces[a], traces[b]) for a, b in (("contact_medium", "contact_coarse"), ("contact_coarse", "contact_fine"))]
        gate("contact_timestep_shape_um", errors[-1], 1.)
        gate("contact_timestep_tip_um", difference(traces["contact_coarse"], traces["contact_fine"], "tip"), 1.)
        report["contact_shape_refinement_um"] = errors
        gate("peak_contact_penetration_um", max(report["cases"][n]["peak_penetration_um"] for n in ("contact_medium", "contact_coarse", "contact_fine")), 2.)
    report["gates"]["numerical_completion"] = {"passed": all(r["completed"] for r in report["cases"].values())}
    report["success"] = all(g["passed"] for g in report["gates"].values())
    report["status"] = "tested_gates_passed" if report["success"] else "quality_gate_failed"
    report["unvalidated"] = ["Settled support contact", "Contact spatial/unit convergence",
                             "Upstream full buckling experiment reproduction", "Physical tool grasp/release",
                             "Quasistatic twist validity and measured material/contact", "Robot plant and tissue interaction"]
    report["ready_for_robot_or_rl"] = False
    report["finished_utc"] = datetime.now(timezone.utc).isoformat()
    report["elapsed_wall_s"] = time.perf_counter()-start
    report["simulated_s"] = sum(r["simulated_s"] for r in report["cases"].values())+sum(
        a["audit_simulated_s"] for a in report.get("audit", {}).values())
    report["timing_note"] = "Wall includes compilation, Python stepping and telemetry; not isolated physics compute."
    save()
    print(json.dumps({"status": report["status"], "gates": report["gates"]}, indent=2), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "preview"))
    parser.add_argument("--suite", choices=("all", "audit", "bending", "relaxation", "contact"), default="all")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "preview":
        print(preview(args.output))
    else:
        result = run(args.output, args.suite)
        raise SystemExit(0 if result["success"] else 1)


if __name__ == "__main__":
    main()
