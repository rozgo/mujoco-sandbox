"""Independent reference wind drives both MuJoCo evaluation runs."""

import json
import time
from pathlib import Path

import mujoco
import numpy as np

from .control import DRAG_DRONE, DRAG_LOAD, Controller, drag, reference, swing_angle
from .scene import DT, load_scene


class Calm:
    def at(self, t, points):
        return np.zeros((len(points), 2))

    def forecast(self, t, leads, points):
        return np.zeros((len(points), 2))


class Simulation:
    def __init__(self, weather=None):
        self.model, self.data = load_scene()
        self.control = Controller(self.model)
        self.weather = weather or Calm()
        self.ticks = 0
        self.snapshots = []
        self.qpos = []
        self.times = []
        self.forces = np.zeros((5, 3))
        self.drag_power = 0.0
        self.peak_swing = 0.0
        self.tracking = []
        self.collisions = set()
        self.max_motor = 0.0
        self.max_cable_error = 0.0
        self.velocity = np.zeros(6)
        self.plan_seconds = 0.0
        self.settled = False

    def apply_wind(self):
        m, d = self.model, self.data
        d.qfrc_applied[:] = 0.0
        self.drag_power = 0.0
        points = np.array(
            [d.site(f"rotor_{i}").xpos.copy() for i in range(4)]
            + [d.body("payload").xpos.copy()]
        )
        wind = self.weather.at(d.time, points[:, :2])
        for i, p in enumerate(points):
            body = m.body("drone" if i < 4 else "payload").id
            mujoco.mj_objectVelocity(
                m, d, mujoco.mjtObj.mjOBJ_BODY, body, self.velocity, 0
            )
            v = self.velocity[3:] + np.cross(self.velocity[:3], p - d.xipos[body])
            airflow = np.r_[wind[i], 0.0]
            force = drag(airflow, v, DRAG_DRONE / 4 if i < 4 else DRAG_LOAD)
            self.forces[i] = force
            self.drag_power += float(force @ (v - airflow))
            mujoco.mj_applyFT(m, d, force, np.zeros(3), p, body, d.qfrc_applied)

    def step(self, record=True):
        m, d, c = self.model, self.data, self.control
        if self.ticks % 50 == 0:
            t = time.perf_counter()
            c.plan(
                d, lambda leads, points: self.weather.forecast(d.time, leads, points)
            )
            self.plan_seconds += time.perf_counter() - t
        if self.ticks % 5 == 0:
            c.update(d)
        d.ctrl[:] += (1 - np.exp(-DT / 0.03)) * (c.command - d.ctrl)
        self.apply_wind()
        mujoco.mj_step(m, d)
        self.ticks += 1
        if not np.isfinite(d.qpos).all() or d.warning.number.any():
            raise RuntimeError(f"Unstable rigid-body simulation at {d.time:.2f}s")
        if d.qpos[2] < 0.15 or abs(d.qpos[:2]).max() > 5.8:
            raise RuntimeError(
                f"Aircraft left flight envelope at {d.time:.2f}s: {d.qpos[:3]}"
            )
        if not c.released:
            self.max_cable_error = max(
                self.max_cable_error,
                float(max(0.0, d.ten_length[0] - m.tendon_range[0, 1])),
            )
            if 6 < d.time < 32:
                self.peak_swing = max(self.peak_swing, swing_angle(d))
                self.tracking.append(
                    float(
                        np.linalg.norm(
                            d.body("payload").xpos[:2] - reference(d.time)[0][:2]
                        )
                    )
                )
        self.max_motor = max(self.max_motor, float(d.actuator_force.max()))
        for con in d.contact:
            names = [m.geom(int(g)).name for g in con.geom]
            if con.dist < 0 and any(n.startswith("gate_") for n in names if n):
                self.collisions.update(n for n in names if n)
        if record and self.ticks % 20 == 0:
            self.times.append(float(d.time))
            self.qpos.append(d.qpos.copy())
            self.snapshots.append(
                {
                    "time": float(d.time),
                    "phase": c.phase,
                    "released": c.released,
                    "swing_deg": swing_angle(d) if not c.released else 0.0,
                    "reference": reference(d.time)[0].tolist(),
                    "rotor_n": d.actuator_force.tolist(),
                    "wind_force_n": self.forces.tolist(),
                    "predicted_load": c.predicted_load.tolist(),
                    "wind_prediction": c.wind_prediction.tolist(),
                }
            )

    def report(self):
        d, c = self.data, self.control
        payload = d.body("payload").xpos
        touching = any(
            set(con.geom)
            == {self.model.geom("package").id, self.model.geom("delivery_platform").id}
            and con.dist < 0.001
            for con in d.contact
        )
        return {
            "weather": getattr(self.weather, "provenance", {"kind": "calm"}),
            "success": bool(
                c.released
                and touching
                and np.linalg.norm(payload[:2] - [2.7, 0]) < 0.35
                and not self.collisions
            ),
            "duration_s": float(d.time),
            "release_time_s": c.release_time,
            "payload_final_position": payload.tolist(),
            "payload_on_destination": bool(touching),
            "payload_tracking_rmse_m": float(np.sqrt(np.mean(np.square(self.tracking))))
            if self.tracking
            else 0.0,
            "peak_swing_deg": self.peak_swing,
            "gate_collisions": sorted(self.collisions),
            "max_rotor_thrust_n": self.max_motor,
            "max_cable_extension_m": self.max_cable_error,
            "warnings": d.warning.number.tolist(),
            "controller_wall_seconds": self.plan_seconds,
        }


def run(weather=None, output=None, duration=43.0):
    sim = Simulation(weather)
    begin = time.monotonic()
    next_log = 0
    while sim.data.time < duration:
        sim.step(record=output is not None)
        if sim.data.time >= next_log:
            print(
                f"{sim.data.time:5.1f}s {sim.control.phase:14} drone={sim.data.qpos[:3].round(2)} swing={swing_angle(sim.data):.1f}",
                flush=True,
            )
            next_log += 5
    report = sim.report()
    report["wall_seconds"] = time.monotonic() - begin
    if output:
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output / "trajectory.npz", time=sim.times, qpos=sim.qpos)
        (output / "telemetry.json").write_text(
            json.dumps(sim.snapshots, separators=(",", ":"))
        )
        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
