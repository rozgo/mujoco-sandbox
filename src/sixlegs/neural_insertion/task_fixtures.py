"""Task-regime thread fixtures: needle drag, normal loading and hook release.

Each fixture uses the DER thread (clamped root, straight along +X at rest) on a
support plane, plus one rigid 150 µm probe on slide joints. Probe motion comes
only from mj_step: bounded force motors for normal load (press, drag) and
force-limited position servos for sliding (drag, release). Commands update at a fixed 50 kHz servo rate, identical
for every timestep under comparison. The probe body has gravity compensation
(a documented applied force), so its own weight does not load the thread.

Gates are those of ACCEPTANCE.md: < 1 µm under timestep halving, < 0.01 µm
across unit systems, < 2 µm peak penetration over every contact at every step,
finite state, no warnings, actual contact; press adds force balance < 1%.
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

from .contact_diagnostics import UNIT_SYSTEMS, checkpoint, difference, kernel, model_xml
from .contact_laws import ContactLaw
from .der import build_provenance, numbers, vertices
from .materials import MATERIALS
from .rescaling import CableConfig
from .scene import add
from .units import MM_G

FIXTURES = ("drag", "press", "release")
COMMAND_PERIOD_S = 2e-5
SAMPLE_S = 1e-4
PROBE_RADIUS_M = 75e-6  # the workcell needle radius
PROBE_MASS_KG = 1e-3
PROBE_KP_N_M = 50.
PROBE_FORCE_LIMIT_N = .01
PRESS_FORCE_N = 1e-4
X_PROBE_M = {"drag": .030, "press": .030, "release": .040}
GATES_UM = {"timestep": 1., "units": .01, "peak_penetration": 2., "force_balance_relative": .01}


def minimum_jerk(a, b, duration, t):
    s = min(max(t/duration, 0.), 1.)
    return a+(b-a)*(10*s**3-15*s**4+6*s**5), (b-a)*(30*s**2-60*s**3+30*s**4)/duration


def schedule(name, t):
    """Probe command (position m or force N) and reference velocity at time t."""
    if name == "drag":  # see drag_schedule: preload and slide use two actuators
        raise ValueError("drag uses drag_schedule")
    if name == "press":  # force ramp to 100 µN, hold, unload
        if t < .005:
            return 0., 0.
        if t < .045:
            return -PRESS_FORCE_N*minimum_jerk(0., 1., .04, t-.005)[0], 0.
        if t < .07:
            return -PRESS_FORCE_N, 0.
        return -PRESS_FORCE_N*(1-minimum_jerk(0., 1., .02, t-.07)[0]), 0.
    if name == "release":  # settle on hook, slide it out past the thread at ~8 mm/s, settle
        if t < .03:
            return 0., 0.
        return minimum_jerk(0., 500e-6, .06, t-.03)
    raise ValueError(name)


DURATION_S = {"drag": .18, "press": .1, "release": .13}
DRAG_PRELOAD_N = 2e-5
DRAG_TRAVEL_M = 2e-4


def drag_schedule(t):
    """Needle across the thread: preload, slide 200 µm along its own axis at 2 mm/s, unload.

    Returns (slide position m, slide velocity m/s, normal force N, negative = down).
    """
    load = 0.
    if .005 <= t < .025:
        load = minimum_jerk(0., 1., .02, t-.005)[0]
    elif .025 <= t < .135:
        load = 1.
    elif .135 <= t < .155:
        load = 1-minimum_jerk(0., 1., .02, t-.135)[0]
    y, v = (0., 0.) if t < .025 else minimum_jerk(0., DRAG_TRAVEL_M, .1, t-.025)
    return y, v, -DRAG_PRELOAD_N*load


def fixture_config(name, dt, material, law):
    case, _ = checkpoint()
    c = replace(CableConfig(**case["config"]), dt_s=dt, contact_time_s=law.time_constant_s,
                integrator="RK4", **MATERIALS[material])
    support = -(c.radius_m+150e-6) if name == "release" else -c.radius_m
    return replace(c, floor_z_m=support)


def fixture_xml(name, config, units, law, integrator):
    root = ET.fromstring(model_xml("der", config, units))
    root.set("model", f"thread_task_{name}")
    root.find("option").set("integrator", integrator)
    L, M = units.length, units.mass
    world = root.find("worldbody")
    r = config.radius_m
    x = X_PROBE_M[name]
    # A needle lying across the resting thread; drag slides it along its own axis.
    if name == "drag":
        half = 5e-4
        pos, zaxis, joint_axis = (x, 0, r+PROBE_RADIUS_M), (0, 1, 0), (0, 0, 1)
    elif name == "press":  # horizontal probe across the thread, touching its top
        half = 3e-4
        pos, zaxis, joint_axis = (x, 0, r+PROBE_RADIUS_M), (0, 1, 0), (0, 0, 1)
    else:  # hook under the thread near its free end, touching its bottom
        half = 3e-4
        pos, zaxis, joint_axis = (x, 0, -r-PROBE_RADIUS_M), (0, 1, 0), (0, 1, 0)
    body = add(world, "body", name="probe", pos=numbers(np.array(pos)*L), gravcomp="1")
    add(body, "joint", name="probe_slide", type="slide", axis=numbers(joint_axis), damping="0")
    if name == "drag":
        add(body, "joint", name="probe_along", type="slide", axis="0 1 0", damping="0")
    add(body, "inertial", pos="0 0 0", mass=format(PROBE_MASS_KG*M, ".17g"),
        diaginertia=numbers(np.full(3, PROBE_MASS_KG*M*(1e-3*L)**2)))
    add(body, "geom", name="probe_geom", type="capsule", size=numbers((PROBE_RADIUS_M*L, half*L)), zaxis=numbers(zaxis),
        rgba=".69 .76 .79 1", condim="3", friction=numbers((config.friction, 1e-6*L, 1e-7*L)))
    actuators = add(root, "actuator")
    limit = PROBE_FORCE_LIMIT_N*units.force
    if name in ("press", "drag"):
        add(actuators, "motor", name="probe", joint="probe_slide", ctrlrange=numbers((-limit, limit)),
            forcerange=numbers((-limit, limit)))
        body.find("joint").set("damping", format(2*math.sqrt(PROBE_KP_N_M*PROBE_MASS_KG)*M, ".17g"))
        if name == "drag":
            kp, kv = PROBE_KP_N_M*M, 2*math.sqrt(PROBE_KP_N_M*PROBE_MASS_KG)*M
            add(actuators, "position", name="probe_along", joint="probe_along", kp=format(kp, ".17g"),
                kv=format(kv, ".17g"), forcerange=numbers((-limit, limit)))
    else:
        kp, kv = PROBE_KP_N_M*M, 2*math.sqrt(PROBE_KP_N_M*PROBE_MASS_KG)*M
        add(actuators, "position", name="probe", joint="probe_slide", kp=format(kp, ".17g"), kv=format(kv, ".17g"),
            forcerange=numbers((-limit, limit)))
    add(root.find("contact"), "exclude", body1="probe", body2="world")
    solimp, solref = numbers(law.solimp(units)), numbers(law.solref())
    for geom in list(root.find("default").iter("geom"))+list(world.iter("geom")):
        geom.set("solimp", solimp)
        geom.set("solref", solref)
    return ET.tostring(root, encoding="unicode")


def law_for(tau):
    return ContactLaw(f"ramp_{tau*1e6:g}us", d0=mujoco.mjMINIMP, power=1., time_constant_s=tau)


def support_reaction(model, data, support, units):
    total, wrench = 0., np.zeros(6)
    for i, c in enumerate(data.contact[:data.ncon]):
        if support in (c.geom1, c.geom2):
            mujoco.mj_contactForce(model, data, i, wrench)
            total += wrench[0]
    return total/units.force


def trial(name, dt, units, material, tau, integrator="RK4"):
    started = time.perf_counter()
    law = law_for(tau)
    c = fixture_config(name, dt, material, law)
    xml = fixture_xml(name, c, units, law, integrator)
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    per_command = round(COMMAND_PERIOD_S/dt)
    every = round(SAMPLE_S/COMMAND_PERIOD_S)
    duration = DURATION_S[name]
    n_commands = round(duration/COMMAND_PERIOD_S)
    if per_command < 1 or not math.isclose(per_command*dt, COMMAND_PERIOD_S, rel_tol=1e-9):
        raise ValueError("Timestep must divide the servo period")
    support, act = m.geom("support").id, m.actuator("probe").id
    qadr, dof = m.jnt_qposadr[m.joint("probe_slide").id], m.jnt_dofadr[m.joint("probe_slide").id]
    scale = units.force if name in ("press", "drag") else units.length
    kp = PROBE_KP_N_M*units.mass
    kv = 2*math.sqrt(PROBE_KP_N_M*PROBE_MASS_KG)*units.mass
    trace = {k: [] for k in ("time", "vertices", "probe_m", "command", "probe_force_n", "support_n", "qpos", "qvel")}

    def sample(command):
        mujoco.mj_forward(m, d)
        trace["time"].append(float(d.time))
        trace["vertices"].append(vertices(m, d, units))
        trace["probe_m"].append(float(d.qpos[qadr]/units.length))
        trace["command"].append(command)
        trace["probe_force_n"].append(float(d.actuator_force[act]/units.force))
        trace["support_n"].append(support_reaction(m, d, support, units))
        trace["qpos"].append(d.qpos.copy())
        trace["qvel"].append(d.qvel.copy())

    summary = np.zeros(15)
    first_contact, failure = None, None
    sample(0.)
    lib = kernel()
    along = m.actuator("probe_along").id if name == "drag" else None
    for k in range(n_commands):
        if name == "drag":
            y, vy, value = drag_schedule(k*COMMAND_PERIOD_S)
            d.ctrl[act] = value*units.force
            d.ctrl[along] = (y+kv/kp*vy)*units.length
        else:
            value, velocity = schedule(name, k*COMMAND_PERIOD_S)
            # Position servo: velocity feedforward removes kv lag; force motor: direct bounded force.
            d.ctrl[act] = value*scale if name == "press" else (value+kv/kp*velocity)*units.length
        chunk = np.zeros(15)
        lib.contact_chunk(m._address, d._address, per_command, -1, -1, chunk)
        summary[0] += chunk[0]
        summary[10] += chunk[10]
        for i in (1, 2, 11, 14):
            summary[i] = max(summary[i], chunk[i])
        if first_contact is None and chunk[12] >= 0:
            first_contact = float(chunk[12])
        if (k+1) % every == 0:
            sample(value)
        if chunk[14]:
            failure = "warning_or_nonfinite"
            break
    trace = {k: np.array(v) for k, v in trace.items()}
    report = {"fixture": name, "material": material, "time_constant_s": tau, "dt_s": dt, "integrator": integrator,
              "units": units.name, "config": asdict(c), "law": asdict(law), "duration_s": duration,
              "simulated_s": float(d.time), "completed": bool(failure is None and int(summary[0]) == n_commands*per_command),
              "failure": failure, "peak_penetration_um": float(summary[1]/units.length*1e6),
              "peak_contact_normal_force_n": float(summary[2]/units.force), "first_contact_s": first_contact,
              "touching_steps": int(summary[10]), "max_solver_iterations": int(summary[11]),
              "warnings": d.warning.number.tolist(), "scene_sha256": hashlib.sha256(xml.encode()).hexdigest(),
              "probe_max_force_fraction": float(np.max(np.abs(trace["probe_force_n"]))/PROBE_FORCE_LIMIT_N),
              "wall_s": time.perf_counter()-started}
    if name == "press":
        t = trace["time"]
        before = (t > .002) & (t < .005)
        hold = (t > .065) & (t <= .07)
        applied = -trace["command"][hold].mean()
        delta = trace["support_n"][hold].mean()-trace["support_n"][before].mean()
        report["press"] = {"applied_force_n": float(applied), "support_reaction_increase_n": float(delta),
                           "relative_balance_error": float(abs(delta-applied)/applied)}
    if name == "release":
        report["max_probe_tracking_error_m"] = float(np.max(np.abs(trace["probe_m"][1:]-trace["command"][1:])))
    if name == "drag":
        report["thread_max_lateral_displacement_m"] = float(np.abs(trace["vertices"][:, :, 1]).max())
    return report, trace, xml


def _worker(spec):
    name, dt, units, material, tau = spec
    try:
        return trial(name, dt, units, material, tau)
    except Exception as exc:
        return {"fixture": name, "failure": f"{type(exc).__name__}: {exc}", "completed": False}, None, None


def case_name(name, material, tau, dt, units):
    return f"{name}_{material}_tau{tau*1e6:g}us_dt{dt*1e9:g}ns_{units}"


def run(output, tau, dt, jobs=1, fixtures=FIXTURES):
    """Fixtures x materials at dt and dt/2 in mm-g-s, plus both other unit systems at dt."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    path = output/"manifest.json"
    if path.exists():
        raise FileExistsError("Preserve previous runs: use a fresh directory")
    started = time.perf_counter()
    kernel()
    source = Path(__file__).parent
    report = {"status": "running", "started_utc": datetime.now(timezone.utc).isoformat(), "mujoco": mujoco.__version__,
              "backend": "MuJoCo CPU", "machine": platform.machine(), "plugin": build_provenance(),
              "time_constant_s": tau, "dt_s": dt, "materials": MATERIALS, "gates_um": GATES_UM,
              "probe": {"radius_m": PROBE_RADIUS_M, "mass_kg": PROBE_MASS_KG, "kp_n_m": PROBE_KP_N_M,
                        "force_limit_n": PROBE_FORCE_LIMIT_N, "command_period_s": COMMAND_PERIOD_S, "gravcomp": True},
              "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
                                [source/"task_fixtures.py", source/"contact_kernel.cc", source/"contact_laws.py",
                                 source/"materials.py", source/"der.py"]},
              "cases": {}, "comparisons": {}, "ready_for_robot_or_rl": False}
    work = []
    for name in fixtures:
        for material in MATERIALS:
            work.append((name, dt, MM_G, material, tau))
            work.append((name, dt/2, MM_G, material, tau))
            work += [(name, dt, u, material, tau) for u in UNIT_SYSTEMS[1:]]
    traces = {}

    def record(spec, result):
        name, step, units, material, _ = spec
        r, trace, xml = result
        key = case_name(name, material, tau, step, units.name)
        if trace is not None:
            p = output/(key+".npz")
            np.savez_compressed(p, **trace)
            r["trajectory_sha256"] = hashlib.sha256(p.read_bytes()).hexdigest()
            traces[key] = trace
        report["cases"][key] = r
        path.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
        print(f"{key}: completed={r.get('completed')} peak={r.get('peak_penetration_um', float('nan')):.4g}um "
              f"wall={r.get('wall_s', 0):.0f}s {r.get('press', '')}", flush=True)

    if jobs > 1:
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            for spec, result in zip(work, pool.map(_worker, work)):
                record(spec, result)
    else:
        for spec in work:
            record(spec, _worker(spec))

    def ok(key):
        c = report["cases"].get(key, {})
        return bool(c.get("completed") and c.get("first_contact_s") is not None
                    and c.get("peak_penetration_um", np.inf) < GATES_UM["peak_penetration"])

    for name in fixtures:
        for material in MATERIALS:
            a, b = case_name(name, material, tau, dt, "mm_g_s"), case_name(name, material, tau, dt/2, "mm_g_s")
            value = difference(traces[a], traces[b]) if a in traces and b in traces else None
            report["comparisons"][f"{name}_{material}_timestep"] = {
                "difference_um": value, "gate_um": GATES_UM["timestep"],
                "passed": bool(ok(a) and ok(b) and value is not None and value < GATES_UM["timestep"])}
            for u in UNIT_SYSTEMS[1:]:
                other = case_name(name, material, tau, dt, u.name)
                value = difference(traces[a], traces[other]) if a in traces and other in traces else None
                report["comparisons"][f"{name}_{material}_{u.name}_units"] = {
                    "difference_um": value, "gate_um": GATES_UM["units"],
                    "passed": bool(ok(a) and ok(other) and value is not None and value < GATES_UM["units"])}
            if name == "press":
                errors = [report["cases"][k]["press"]["relative_balance_error"] for k in report["cases"]
                          if k.startswith(f"press_{material}_") and "press" in report["cases"][k]]
                report["comparisons"][f"press_{material}_force_balance"] = {
                    "max_relative_error": max(errors) if errors else None,
                    "gate": GATES_UM["force_balance_relative"],
                    "passed": bool(errors and max(errors) < GATES_UM["force_balance_relative"])}
    report["success"] = all(v["passed"] for v in report["comparisons"].values())
    report["status"] = "task_fixture_gates_passed" if report["success"] else "task_fixture_gates_failed"
    report["elapsed_wall_s"] = time.perf_counter()-started
    report["simulated_s"] = sum(c.get("simulated_s", 0.) for c in report["cases"].values())
    report["finished_utc"] = datetime.now(timezone.utc).isoformat()
    path.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(json.dumps(report["comparisons"], indent=2), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run",))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tau", type=float, required=True, help="Contact time constant, s")
    parser.add_argument("--dt", type=float, required=True, help="Coarser timestep, s; dt/2 is also run")
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--fixtures", nargs="+", choices=FIXTURES, default=list(FIXTURES))
    args = parser.parse_args()
    result = run(args.output, args.tau, args.dt, args.jobs, tuple(args.fixtures))
    raise SystemExit(0 if result["success"] else 1)


if __name__ == "__main__":
    main()
