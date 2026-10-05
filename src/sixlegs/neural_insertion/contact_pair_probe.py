"""Locate divergence in two unit-equivalent DER trajectories, step by step."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import mujoco
import numpy as np

from .contact_diagnostics import UNIT_SYSTEMS, load_fixture


def trial(alternative, duration=.01):
    started = time.perf_counter()
    systems = (UNIT_SYSTEMS[0], alternative)
    fixtures = [load_fixture("der", 7.8125e-8, u)[1:3] for u in systems]
    ids = [i for i in range(fixtures[0][0].nbody)
           if fixtures[0][0].body(i).name.startswith("thread_B") or fixtures[0][0].body(i).name == "thread_endpoint"]
    events, event_states, rows = {}, {}, []
    dt = fixtures[0][0].opt.timestep
    steps, sample = round(duration/dt), 128  # dense 10 µs telemetry
    peaks = {"shape_um": 0., "force_relative": 0., "smooth_acceleration_relative": 0.}
    failure = None
    for step in range(steps):
        for m, d in fixtures:
            mujoco.mj_step1(m, d)
        a, b = fixtures[0][1], fixtures[1][1]
        shape = float(np.linalg.norm(a.xpos[ids]/systems[0].length-b.xpos[ids]/systems[1].length, axis=1).max()*1e6)
        contacts = [[[int(c.geom1), int(c.geom2), int(c.dim)] for c in d.contact] for _, d in fixtures]
        same_contact = contacts[0] == contacts[1]
        # Preserve pre-integration states only for the first decisive events.
        pending = []
        for threshold in (1e-9, .001, .01, 1.):
            key = f"shape_above_{threshold:g}_um"
            if key not in events and shape > threshold:
                pending.append(key)
        if not same_contact and "contact_set_differs" not in events:
            pending.append("contact_set_differs")
        for key in pending:
            event_states[key] = {f"{name}_{j}": getattr(d, name).copy()
                                 for j, (_, d) in enumerate(fixtures)
                                 for name in ("qpos", "qvel", "plugin_state", "qacc_warmstart")}
        for m, d in fixtures:
            d.qfrc_applied[:] = 0
            mujoco.mj_step2(m, d)
        force_a, force_b = a.qfrc_constraint/systems[0].torque, b.qfrc_constraint/systems[1].torque
        absolute_force = float(np.max(abs(force_a-force_b)))
        relative_force = absolute_force/max(float(np.max(abs(force_a))), float(np.max(abs(force_b))), 1e-30)
        smooth = float(np.max(abs(a.qacc_smooth-b.qacc_smooth)))/max(float(np.max(abs(a.qacc_smooth))), 1e-30)
        if "force_difference_above_1e-6_relative" not in events and relative_force > 1e-6:
            pending.append("force_difference_above_1e-6_relative")
        event = {"time_s": step*dt, "shape_difference_um": shape, "force_relative_difference": relative_force,
                 "force_absolute_difference_nm": absolute_force, "smooth_acceleration_relative_difference": smooth,
                 "contact_sets": contacts}
        for key in pending:
            events[key] = event
        peaks["shape_um"] = max(peaks["shape_um"], shape)
        peaks["force_relative"] = max(peaks["force_relative"], relative_force)
        peaks["smooth_acceleration_relative"] = max(peaks["smooth_acceleration_relative"], smooth)
        if step%sample == 0:
            rows.append([step*dt, shape, relative_force, absolute_force, smooth, len(contacts[0]), len(contacts[1]), same_contact])
        if any(d.warning.number.any() or not np.isfinite(d.qpos).all() or not np.isfinite(d.qvel).all() for _, d in fixtures):
            failure = "warning_or_nonfinite"
            break
        # Match the baseline harness's extra forward evaluation every 100 µs.
        if (step+1)%1280 == 0:
            for m, d in fixtures:
                mujoco.mj_forward(m, d)
    return {"alternative_units": alternative.name, "events": events, "peaks": peaks,
            "failure": failure, "completed": failure is None, "wall_s": time.perf_counter()-started,
            "simulated_s_per_model": (step+1)*dt,
            "columns": ["time_s", "shape_um", "force_relative", "force_absolute_nm", "smooth_acceleration_relative", "contacts_a", "contacts_b", "same_contact"]}, np.array(rows), event_states


def run(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    path = output/"manifest.json"
    if path.exists():
        raise FileExistsError("Use a fresh output directory")
    start = time.perf_counter()
    report = {"status": "running", "started_utc": datetime.now(timezone.utc).isoformat(),
              "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "protocol": "DER; 78.125 ns; first 10 ms after portable pre-impact checkpoint; all physics substeps inspected; 10 us stored telemetry.",
              "cases": {}, "ready_for_robot_or_rl": False}
    for units in UNIT_SYSTEMS[1:]:
        result, trace, states = trial(units)
        p = output/(units.name+".npz")
        np.savez_compressed(p, telemetry=trace, **{event+"__"+key: value for event, state in states.items() for key, value in state.items()})
        result["trajectory_sha256"] = hashlib.sha256(p.read_bytes()).hexdigest()
        report["cases"][units.name] = result
        path.write_text(json.dumps(report, indent=2)+"\n")
        print(units.name, json.dumps(result["events"]), flush=True)
    report["elapsed_wall_s"] = time.perf_counter()-start
    report["aggregate_simulated_s"] = sum(2*c["simulated_s_per_model"] for c in report["cases"].values())
    report["finished_utc"] = datetime.now(timezone.utc).isoformat()
    report["status"] = "pair_audit_complete_accuracy_unaccepted"
    path.write_text(json.dumps(report, indent=2)+"\n")
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("output", type=Path)
    run(p.parse_args().output)


if __name__ == "__main__":
    main()
