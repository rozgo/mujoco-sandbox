"""Shared physical stepping, checks, telemetry, and saved trajectory."""

import json
import time
from pathlib import Path

import mujoco
import numpy as np

from .control import Controller
from .fluid import Water
from .scene import DT, FINISH_X, WATER_Z, load_scene


class Simulation:
    def __init__(self, buoyancy_scale=1.0):
        self.model, self.data = load_scene()
        self.control = Controller(self.model, self.data)
        self.water = Water(self.model, buoyancy_scale)
        self.ticks = 0
        self.times = []
        self.qpos = []
        self.samples = []
        self.phases = []
        self.weight = self.control.mass * 9.81
        self.min_up = 1.0
        self.max_torque_fraction = 0.0
        self.float_seconds = 0.0
        self.max_float_streak = 0.0
        self.float_streak = 0.0
        self.peak_buoyancy = 0.0
        self.done_at = None
        self.max_penetration = 0.0
        self.attachment_contacts = set()
        self.max_drag_power = 0.0
        self.start = self.data.qpos[:3].copy()

    def step(self, record=True):
        m, d, c, w = self.model, self.data, self.control, self.water
        if self.ticks % 5 == 0:
            c.update(d, w, 0.01)
        w.apply(d, c.thrust)
        mujoco.mj_step(m, d)
        self.ticks += 1
        up = float(d.body("base").xmat[8])
        self.min_up = min(self.min_up, up)
        if not np.isfinite(d.qpos).all() or d.warning.number.any():
            raise RuntimeError("MuJoCo became unstable")
        if up < 0.4:
            raise RuntimeError(f"Robot tipped at {d.time:.2f}s")
        if abs(d.qpos[1]) > 1.0:
            raise RuntimeError(f"Robot left the crossing lane at {d.time:.2f}s")
        caps = m.actuator_forcerange[:, 1]
        self.max_torque_fraction = max(
            self.max_torque_fraction, float(np.max(abs(d.actuator_force) / caps))
        )
        self.peak_buoyancy = max(self.peak_buoyancy, float(sum(w.buoyancy)))
        self.max_drag_power = max(self.max_drag_power, w.drag_power)
        any_ground = False
        for contact in d.contact:
            bodies = m.geom_bodyid[contact.geom]
            names = [m.geom(int(g)).name for g in contact.geom]
            if 0 in bodies and np.any(bodies != 0) and contact.dist < 0.002:
                any_ground = True
                if contact.dist < 0:
                    self.max_penetration = max(
                        self.max_penetration, -float(contact.dist)
                    )
                if any(n and "float_shell" in n for n in names):
                    self.attachment_contacts.update(n for n in names if n)
        floating = (
            not any_ground
            and sum(w.buoyancy) > 0.8 * self.weight
            and c.phase == "FLOATING"
        )
        self.float_streak = self.float_streak + DT if floating else 0.0
        self.max_float_streak = max(self.max_float_streak, self.float_streak)
        self.float_seconds += DT if floating else 0.0
        if c.done and self.done_at is None:
            self.done_at = float(d.time)
        if record and self.ticks % 20 == 0:
            self.times.append(float(d.time))
            self.qpos.append(d.qpos.copy())
            self.phases.append(c.phase)
            self.samples.append(
                {
                    "time": float(d.time),
                    "phase": c.phase,
                    "stage": c.stage,
                    "buoyancy_n": w.buoyancy.tolist(),
                    "submerged": w.submerged.tolist(),
                    "buoyancy_centers": w.centers.tolist(),
                    "thrust_n": w.thrust.tolist(),
                    "foot_contacts": c.contacts(d).tolist(),
                    "ground_contact": any_ground,
                    "up": up,
                    "weight_n": self.weight,
                }
            )

    def report(self):
        d, c = self.data, self.control
        return {
            "success": bool(
                c.done
                and self.max_float_streak > 2
                and sum(self.water.buoyancy) < 1.0
                and c.contacts(d).all()
            ),
            "simulation_seconds": float(d.time),
            "finish_x": FINISH_X,
            "final_position": d.qpos[:3].tolist(),
            "mass_kg": self.control.mass,
            "weight_n": self.weight,
            "water_level_m": WATER_Z,
            "phases": c.events,
            "longest_unsupported_float_s": self.max_float_streak,
            "total_unsupported_float_s": self.float_seconds,
            "peak_buoyancy_n": self.peak_buoyancy,
            "final_buoyancy_n": float(sum(self.water.buoyancy)),
            "final_foot_contacts": c.contacts(d).tolist(),
            "min_body_up": self.min_up,
            "max_actuator_fraction": self.max_torque_fraction,
            "max_ground_penetration_m": self.max_penetration,
            "attachment_ground_contacts": sorted(self.attachment_contacts),
            "max_drag_power_w": self.max_drag_power,
            "warnings": d.warning.number.tolist(),
        }


def run(output=None, duration=650):
    sim = Simulation()
    start = time.monotonic()
    next_log = 0
    while sim.data.time < duration:
        sim.step()
        if sim.data.time >= next_log:
            print(
                f"{sim.data.time:6.1f}s {sim.control.phase:12} x={sim.data.qpos[0]:5.2f} buoy={sum(sim.water.buoyancy):6.1f} N",
                flush=True,
            )
            next_log += 30
        if sim.done_at is not None and sim.data.time > sim.done_at + 3.0:
            break
    report = sim.report()
    report["wall_seconds"] = time.monotonic() - start
    if output:
        folder = Path(output)
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        (folder / "telemetry.json").write_text(
            json.dumps(sim.samples, separators=(",", ":"))
        )
        np.savez_compressed(
            folder / "trajectory.npz", time=sim.times, qpos=sim.qpos, phase=sim.phases
        )
    return report
