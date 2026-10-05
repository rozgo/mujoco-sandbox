"""Focused contact audit; independent symmetric normal-impact reference."""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import mujoco
import numpy as np

from .contact_diagnostics import UNIT_SYSTEMS, load_fixture, kernel, centerline
from .contact_laws import LAWS
from .scene import ROOT
from .units import MM_G


def normal_acceleration(gap_m, velocity_m_s, gravity=9.81, impedance=.9999, time_constant=1e-5, law=None):
    """Closed-form two-contact QP solution for a horizontal, symmetric capsule.

    MuJoCo 3.12: each normal row has diagApprox=1/m and
    R=(1-d)/(d*m). Equal endpoint forces solve (2/m+R)*f=aref+g.
    Tangential forces and rotation vanish by symmetry. No MuJoCo calls here.
    With a `ContactLaw`, d is the pinned penetration-dependent impedance ramp
    while B and K keep using d_width, exactly as mj_makeImpedance does.
    """
    if gap_m > 0:
        return -gravity
    if law is not None:
        impedance, time_constant = law.d_width, law.time_constant_s
        ratio = law.damping_ratio
        d = law.impedance(-gap_m)
    else:
        ratio, d = 1., impedance
    damping = 2/(impedance*time_constant)
    stiffness = 1/(impedance*time_constant*ratio)**2
    aref = -damping*velocity_m_s-stiffness*d*gap_m
    effective_impedance = 2*d/(1+d)
    return -gravity+effective_impedance*max(0., aref+gravity)


def normal_reference(times, *, tighter=False, law=None):
    """Independent continuous-time ODE; analytic flight, adaptive DOP853 impact.

    This validates only the implemented symmetric contact law, not material data.
    SciPy is supplied by the existing optional wind extra.
    """
    from scipy.integrate import solve_ivp
    started = time.perf_counter()
    gravity, gap = 9.81, .006
    impact = np.sqrt(2*gap/gravity)
    result = np.empty((len(times), 2))
    flight = times < impact
    result[flight, 0] = gap-.5*gravity*times[flight]**2
    result[flight, 1] = -gravity*times[flight]
    solve = solve_ivp(lambda t, y: (y[1], normal_acceleration(*y, law=law)),
                     (impact, times[-1]), (0., -gravity*impact), method="DOP853",
                     t_eval=times[~flight], max_step=5e-7 if tighter else 1e-6,
                     rtol=5e-13 if tighter else 5e-12,
                     atol=(1e-17, 1e-13) if tighter else (1e-16, 1e-12))
    if not solve.success:
        raise RuntimeError(solve.message)
    result[~flight] = solve.y.T
    return result, {"method": "analytic flight + independent DOP853 contact ODE",
                    "law": law.name if law is not None else "flat", "tighter": tighter, "nfev": solve.nfev,
                    "first_contact_s": float(impact), "wall_s": time.perf_counter()-started,
                    "simulated_s": float(times[-1])}


def normal_fixed_step(dt, duration=.06, initial_gap=.006, initial_velocity=0., law=None, method="euler"):
    """Independent fixed-step recurrence in SI, without MuJoCo.

    `euler` is the semi-implicit recurrence of mj_Euler/mj_implicit; `rk4` is
    the classical explicit scheme of mj_RungeKutta applied to the same law.
    """
    started = time.perf_counter()
    gap, velocity = initial_gap, initial_velocity
    every = round(.0001/dt)
    trace = [[gap, velocity]]
    first_contact = None

    def rate(s, v):
        return v, normal_acceleration(s, v, law=law)

    for step in range(round(duration/dt)):
        acceleration = normal_acceleration(gap, velocity, law=law)
        if first_contact is None and acceleration != -9.81:
            first_contact = step*dt
        if method == "euler":
            velocity += dt*acceleration
            gap += dt*velocity
        elif method == "rk4":
            k1 = rate(gap, velocity)
            k2 = rate(gap+.5*dt*k1[0], velocity+.5*dt*k1[1])
            k3 = rate(gap+.5*dt*k2[0], velocity+.5*dt*k2[1])
            k4 = rate(gap+dt*k3[0], velocity+dt*k3[1])
            gap += dt*(k1[0]+2*k2[0]+2*k3[0]+k4[0])/6
            velocity += dt*(k1[1]+2*k2[1]+2*k3[1]+k4[1])/6
        else:
            raise ValueError("Unknown recurrence")
        if (step+1)%every == 0:
            trace.append([gap, velocity])
    return np.array(trace), {"method": "independent SI "+("semi-implicit Euler" if method == "euler" else "classical RK4")+" recurrence",
                            "law": law.name if law is not None else "flat", "first_contact_s": first_contact,
                            "simulated_s": duration, "wall_s": time.perf_counter()-started}


def frozen_normal():
    records = []
    for gap in (-.05e-6, -.5e-6, -1.25e-6):
        for velocity in (-.343, 0., .02, .05):
            expected = normal_acceleration(gap, velocity)
            cases = []
            for u in UNIT_SYSTEMS:
                c, m, d, _ = load_fixture("segment_normal", 7.8125e-8, u)
                # Scratch evaluation, not live trajectory advancement.
                d.qpos[2] = (c.floor_z_m+c.radius_m+gap)*u.length
                d.qvel[2] = velocity*u.length
                d.qacc_warmstart[:] = 0
                mujoco.mj_forward(m, d)
                measured = float(d.qacc[2]/u.length)
                force = float(d.qfrc_constraint[2]/u.force)
                cases.append({"units": asdict(u), "acceleration_m_s2": measured,
                              "support_force_n": force, "contacts": d.ncon,
                              "acceleration_relative_error": abs(measured-expected)/max(9.81, abs(expected)),
                              "R_times_mass": (d.efc_R*m.body_mass[1]).tolist(),
                              "solver_iterations": int(max(d.solver_niter))})
            records.append({"gap_m": gap, "velocity_m_s": velocity,
                            "independent_acceleration_m_s2": expected, "cases": cases})
    maximum = max(c["acceleration_relative_error"] for r in records for c in r["cases"])
    return {"cases": records, "maximum_relative_error": maximum,
            "gate": 1e-8, "passed": bool(maximum < 1e-8)}


def checked_trace(folder, manifest, name):
    p = folder/(name+".npz")
    if hashlib.sha256(p.read_bytes()).hexdigest() != manifest["cases"][name]["trajectory_sha256"]:
        raise ValueError("Trajectory hash mismatch: "+name)
    with np.load(p) as saved:
        return {k: saved[k] for k in saved.files}


def evaluate_same_state(kind, state, tolerance=1e-12):
    values = []
    for u in UNIT_SYSTEMS:
        _, m, d, _ = load_fixture(kind, 7.8125e-8, u)
        m.opt.tolerance = tolerance
        for name in ("qpos", "qvel", "plugin_state"):
            getattr(d, name)[:] = state[name]
        d.qacc_warmstart[:] = 0
        mujoco.mj_forward(m, d)
        contacts = [[int(c.geom1), int(c.geom2), int(c.dim)] for c in d.contact]
        values.append({"acceleration": d.qacc.copy(), "constraint": d.qfrc_constraint.copy()/u.torque,
                       "smooth": d.qacc_smooth.copy(), "contacts": contacts,
                       "iterations": int(max(d.solver_niter)), "warnings": d.warning.number.tolist()})
    records = []
    for u, v in zip(UNIT_SYSTEMS[1:], values[1:]):
        r = {"units": u.name, "same_contacts": v["contacts"] == values[0]["contacts"],
             "iterations": [values[0]["iterations"], v["iterations"]]}
        for field in ("acceleration", "constraint", "smooth"):
            a, b = values[0][field], v[field]
            r[field+"_relative_difference"] = float(np.max(abs(a-b))/max(np.max(abs(a)), 1e-30))
            r[field+"_absolute_difference"] = float(np.max(abs(a-b)))
        records.append(r)
    return {"tolerance": tolerance, "reference_contacts": values[0]["contacts"], "comparisons": records}


def frozen_chains(folder, manifest):
    report = {}
    for kind in ("chain_no_elastic", "der"):
        trace = checked_trace(folder, manifest, kind+"_mm_g_s_fine")
        records = []
        for stamp in (.0035, .0036, .004, .008, .012, .020, .026):
            i = int(np.argmin(abs(trace["time"]-stamp)))
            state = {k: trace[k][i] for k in ("qpos", "qvel", "plugin_state")}
            records.append({"time_s": float(trace["time"][i]),
                            "default": evaluate_same_state(kind, state),
                            "no_tolerance_exit": evaluate_same_state(kind, state, tolerance=0.)})
        report[kind] = records
    return report


def run(folder, output, refinement=None):
    folder, output = Path(folder), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    path = output/"manifest.json"
    if path.exists():
        raise FileExistsError("Use a fresh output directory")
    start = time.perf_counter()
    baseline = json.loads((folder/"manifest.json").read_text())
    report = {"started_utc": datetime.now(timezone.utc).isoformat(), "status": "running",
              "mujoco": mujoco.__version__, "backend": "MuJoCo CPU", "ready_for_robot_or_rl": False,
              "source_sha256": {Path(__file__).name: hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
              "baseline_manifest_sha256": hashlib.sha256((folder/"manifest.json").read_bytes()).hexdigest(),
              "frozen_normal": frozen_normal(), "frozen_chains": frozen_chains(folder, baseline)}
    path.write_text(json.dumps(report, indent=2)+"\n")
    print("Frozen-state comparisons saved", flush=True)
    times = np.arange(601)*.0001
    a, ar = normal_reference(times)
    b, br = normal_reference(times, tighter=True)
    np.savez_compressed(output/"normal_reference.npz", time=times, standard=a, tighter=b)
    report["independent_reference"] = {"standard": ar, "tighter": br,
        "refinement_difference_um": float(np.max(abs(a[:, 0]-b[:, 0]))*1e6),
        "trajectory_sha256": hashlib.sha256((output/"normal_reference.npz").read_bytes()).hexdigest(),
        "comparisons": {}}
    sources = [(folder, baseline, "segment_normal_mm_g_s_"+level) for level in ("coarse", "fine")]
    if refinement:
        refined = json.loads((Path(refinement)/"manifest.json").read_text())
        sources += [(Path(refinement), refined, key) for key in refined["cases"]]
    for src, manifest, name in sources:
        t = checked_trace(src, manifest, name)
        np.testing.assert_allclose(t["time"], times, atol=1e-9, rtol=0)
        gap = t["qpos"][:, 2]/MM_G.length-(-.006+20e-6)
        dt = manifest["cases"][name]["config"]["dt_s"]
        recurrence, receipt = normal_fixed_step(dt)
        np.savez_compressed(output/(name+"_independent.npz"), time=times, state=recurrence)
        report["independent_reference"]["comparisons"][name] = {
            "timestep_s": dt,
            "max_position_difference_um": float(np.max(abs(gap-b[:, 0]))*1e6),
            "max_velocity_difference_m_s": float(np.max(abs(t["qvel"][:, 2]/MM_G.length-b[:, 1]))),
            "first_contact_s": manifest["cases"][name]["first_contact_s"],
            "fixed_step_reference": receipt,
            "mujoco_vs_fixed_step_difference_um": float(np.max(abs(gap-recurrence[:, 0]))*1e6),
            "fixed_step_vs_continuous_difference_um": float(np.max(abs(recurrence[:, 0]-b[:, 0]))*1e6)}
    report["elapsed_wall_s"] = time.perf_counter()-start
    report["finished_utc"] = datetime.now(timezone.utc).isoformat()
    report["status"] = "audit_complete_accuracy_unaccepted"
    path.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report["independent_reference"], indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--normal-refinement", type=Path)
    args = parser.parse_args()
    run(args.baseline, args.output, args.normal_refinement)


if __name__ == "__main__":
    main()
