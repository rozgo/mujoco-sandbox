"""One shared physical habitat. No scripted fly paths or pose-driven movement."""

import json
import time
from dataclasses import asdict

import mujoco as mj
import numpy as np

from .biology import Needs, Resources
from .motor import Motors
from .neural import NeuralPopulation
from .paths import OUTPUTS
from .scene import RESOURCE_POS, build
from .senses import Sensors
from .utility import ACTIONS, Thinker, handcrafted_weights


class Habitat:
    decision_dt = 0.02

    def __init__(
        self,
        n_flies=8,
        seed=101,
        weights=None,
        neural=True,
        device="cpu",
        vision=True,
        record=False,
        hazards=True,
        arena=None,
    ):
        self.n, self.seed = n_flies, seed
        self.rng = np.random.default_rng(seed)
        self.arena = arena or build(n_flies)
        self.m, self.d = self.arena.sim.mj_model, self.arena.sim.mj_data
        self.motors = Motors(self.arena, seed=seed)
        self.sensors = Sensors(self.arena, vision=vision)
        self.neural = NeuralPopulation(n_flies, device, seed) if neural else None
        self.neural_features = np.zeros((n_flies, 2, 2))
        w = handcrafted_weights() if weights is None else weights
        self.thinkers = [
            Thinker(weights=np.array(w, copy=True)) for _ in range(n_flies)
        ]
        self.needs = [
            Needs(
                energy=float(self.rng.uniform(0.28, 0.82)),
                hydration=float(self.rng.uniform(0.3, 0.9)),
                fatigue=float(self.rng.uniform(0, 0.55)),
            )
            for _ in range(n_flies)
        ]
        self.resources = Resources()
        self.hazards = hazards
        self.next_swat = float(self.rng.uniform(1.4, 2.4))
        self.swat_start = None
        self.swat_events = 0
        self.walk_angle = self.rng.uniform(-np.pi, np.pi, n_flies)
        self.commands = np.zeros((n_flies, 2))
        self.speed = np.zeros(n_flies)
        self.previous_xyz = self.d.xpos[self.arena.fly_body_ids].copy()
        self.impacts = np.zeros(n_flies)
        self.total_impulse = np.zeros(n_flies)
        self.max_contact_force = 0.0
        self.interfly_contacts = 0
        self.action_seconds = np.zeros((n_flies, 6))
        self.events = []
        self.last_event = [""] * n_flies
        self.frames = []
        self.qpos = []
        self.qvel = []
        self.ctrl = []
        self.record = record
        self.tick = 0
        self.max_actuator_ratio = 0.0
        self._geom_fly = np.full(self.m.ngeom, -1, int)
        for i, ids in enumerate(self.sensors.own_geoms):
            self._geom_fly[ids] = i
        self.force = np.zeros(6)
        self.warnings = []
        self.started = time.perf_counter()
        self.step_wall = 0.0

    def _swatter(self):
        t = float(self.d.time)
        raised = -0.82
        if self.hazards and self.swat_start is None and t >= self.next_swat:
            self.swat_start = t
            self.swat_events += 1
            self.events.append({"t": t, "type": "swatter_cycle", "fly": None})
        if self.swat_start is None:
            target = raised
        else:
            age = t - self.swat_start
            if age < 0.5:
                target = raised + (0.48) * age / 0.5
            elif age < 0.62:
                target = -0.34 + (0.375) * (age - 0.5) / 0.12
            elif age < 0.7:
                target = 0.035
            elif age < 1.0:
                target = 0.035 + (raised - 0.035) * (age - 0.7) / 0.3
            else:
                self.swat_start = None
                self.next_swat = t + float(self.rng.uniform(1.4, 2.4))
                target = raised
        self.d.ctrl[self.arena.swat_actuator] = target

    def _contacts(self, dt):
        geoms=self.d.contact.geom
        owners=self._geom_fly[geoms]
        self.interfly_contacts+=int(np.count_nonzero((owners[:,0]>=0)&(owners[:,1]>=0)&(owners[:,0]!=owners[:,1])))
        swat_ids=np.array(tuple(self.sensors.swat_ids))
        mask=(geoms[:,0,None]==swat_ids).any(axis=1)|(geoms[:,1,None]==swat_ids).any(axis=1)
        for k in np.flatnonzero(mask):
            c = self.d.contact[k]
            if c.efc_address < 0:
                continue
            a, b = int(c.geom1), int(c.geom2)
            i, j = self._geom_fly[[a, b]]
            target = (
                i
                if b in self.sensors.swat_ids
                else (j if a in self.sensors.swat_ids else -1)
            )
            if target < 0:
                continue
            mj.mj_contactForce(self.m, self.d, k, self.force)
            force = max(0, float(self.force[0]))
            self.max_contact_force = max(self.max_contact_force, force)
            # Time integral of measured compressive force, sampled every physics substep.
            self.impacts[target] += force * dt

    def _motor_command(self, i, p, action):
        bid = self.arena.fly_body_ids[i]
        forward = self.d.xmat[bid].reshape(3, 3)[:2, 0]
        forward = forward / max(np.linalg.norm(forward), 1e-9)
        if self.tick % 35 == 0:
            self.walk_angle[i] += float(self.rng.normal(0, 0.55))
        wander = np.array([np.cos(self.walk_angle[i]), np.sin(self.walk_angle[i])])
        desired = wander
        speed = 0.7
        if action == 1:
            desired = p.food_gradient
            if p.food > 0.78:
                speed = 0
        elif action == 2:
            desired = p.water_gradient
            if p.water > 0.76:
                speed = 0
        elif action == 3:
            speed = 0
        elif action == 4:
            desired = p.shade_gradient - p.heat_gradient * 4
            speed = 0 if p.shade > 0.6 else 0.85
        elif action == 5:
            desired = -p.heat_gradient * 10
            if np.linalg.norm(desired) < 0.002:
                desired = forward
            speed = 1.1
        if speed == 0:
            return np.zeros(2)
        if np.linalg.norm(desired) < 0.002:
            desired = wander
        desired = desired / max(np.linalg.norm(desired), 1e-9) + p.avoidance * 2.5
        angle = np.arctan2(
            forward[0] * desired[1] - forward[1] * desired[0], np.dot(forward, desired)
        )
        turn = float(np.clip(angle, -1.5, 1.5))
        # The steering neural readout can bias turning; no neuron is mapped straight to a joint.
        turn += 0.12 * float(
            self.neural_features[i, 1, 0] - self.neural_features[i, 1, 1]
        )
        return np.clip([speed - 0.55 * turn, speed + 0.55 * turn], -1.2, 1.2)

    def step(self):
        start = time.perf_counter()
        dt = self.decision_dt
        alive = np.array([n.alive for n in self.needs])
        percepts = [
            self.sensors.sample(
                i, self.resources.remaining, dt, refresh_eyes=self.tick % 5 == 0
            )
            for i in range(self.n)
        ]
        if self.neural is not None:
            self.neural_features = self.neural.step(percepts, alive)
        old_actions = [t.current for t in self.thinkers]
        for i, (p, n, t) in enumerate(zip(percepts, self.needs, self.thinkers)):
            if not n.alive:
                self.commands[i] = 0
                continue
            f = np.array(
                [
                    1,
                    1 - n.energy,
                    1 - n.hydration,
                    n.fatigue,
                    min(n.heat, 1),
                    p.threat,
                    p.food,
                    p.water,
                    p.shade,
                    p.crowding,
                    float(self.neural_features[i, 0].max()),
                    float(
                        abs(
                            self.neural_features[i, 1, 0]
                            - self.neural_features[i, 1, 1]
                        )
                    ),
                ]
            )
            action = t.tick(f, dt)
            if old_actions[i] != action:
                event = {
                    "t": float(self.d.time),
                    "fly": i,
                    "type": "choice",
                    "from": ACTIONS[old_actions[i]],
                    "to": ACTIONS[action],
                }
                self.events.append(event)
                self.last_event[i] = f"{event['from']} → {event['to']}"
            self.commands[i] = self._motor_command(i, p, action)
            self.action_seconds[i, action] += dt
        self.impacts.fill(0)
        for _ in range(round(dt / self.motors.timestep)):
            self._swatter()
            self.motors.apply(self.commands, alive)
            for _ in range(round(self.motors.timestep / self.m.opt.timestep)):
                mj.mj_step(self.m, self.d)
                self._contacts(self.m.opt.timestep)
        if not np.isfinite(self.d.qpos).all() or self.d.warning.number.any():
            raise RuntimeError(
                f"Numerical failure at t={self.d.time}: {self.d.warning.number}"
            )
        xyz = self.d.xpos[self.arena.fly_body_ids].copy()
        self.speed = np.linalg.norm(xyz[:, :2] - self.previous_xyz[:, :2], axis=1) / dt
        self.previous_xyz = xyz
        demand = np.zeros((self.n, 3))
        for i, (p, n, t) in enumerate(zip(percepts, self.needs, self.thinkers)):
            if not n.alive or self.speed[i] > 2.5 or not 0.5 < xyz[i, 2] < 1.8:
                continue
            distance = np.linalg.norm(RESOURCE_POS - xyz[i, :2], axis=1)
            if t.current == 1:
                patch = int(np.argmin(distance[:2]))
                if distance[patch] < 2.35:
                    demand[i, patch] = min(dt * 0.5, 1 - n.energy)
            if t.current == 2 and distance[2] < 2.35:
                demand[i, 2] = min(dt * 0.7, 1 - n.hydration)
        intake = self.resources.consume(demand)
        for i, (p, n) in enumerate(zip(percepts, self.needs)):
            was_alive = n.alive
            exposure = p.heat if self.hazards else 0.0
            n.advance(
                dt,
                self.speed[i],
                exposure,
                float(intake[i, :2].sum()),
                float(intake[i, 2]),
                float(self.impacts[i]),
            )
            self.total_impulse[i] += self.impacts[i]
            if was_alive and not n.alive:
                reason = (
                    "impact"
                    if self.impacts[i] > 0.005
                    else ("heat" if n.heat > 0.75 else "depletion")
                )
                self.events.append(
                    {
                        "t": float(self.d.time),
                        "type": "death",
                        "fly": i,
                        "reason": reason,
                    }
                )
                self.last_event[i] = f"Death / {reason}"
        limited = self.m.actuator_forcelimited.astype(bool)
        limits = np.maximum(abs(self.m.actuator_forcerange[limited]).max(axis=1), 1e-9)
        self.max_actuator_ratio = max(
            self.max_actuator_ratio,
            float((abs(self.d.actuator_force[limited]) / limits).max()),
        )
        if self.record:
            self._capture(percepts)
        self.tick += 1
        self.step_wall += time.perf_counter() - start

    def _capture(self, percepts):
        self.qpos.append(self.d.qpos.copy())
        self.qvel.append(self.d.qvel.copy())
        self.ctrl.append(self.d.ctrl.copy())
        flies = []
        for i, (p, n, t) in enumerate(zip(percepts, self.needs, self.thinkers)):
            flies.append(
                {
                    "alive": n.alive,
                    "action": t.current,
                    "scores": t.scores.tolist(),
                    "needs": n.values(),
                    "xyz": self.d.xpos[self.arena.fly_body_ids[i]].tolist(),
                    "speed": float(self.speed[i]),
                    "event": self.last_event[i],
                    "exposure": p.heat,
                    "threat": p.threat,
                    "neural_rates_hz": (
                        self.neural.rate[i].tolist() if self.neural else None
                    ),
                    "spikes": (
                        len(self.neural.last_spikes[i]) if self.neural else None
                    ),
                    "brain": (self.neural.activity(i) if self.neural else []),
                }
            )
        self.frames.append(
            {
                "t": float(self.d.time),
                "flies": flies,
                "resources": self.resources.remaining.tolist(),
            }
        )

    def run(self, seconds):
        for _ in range(round(seconds / self.decision_dt)):
            self.step()
        return self.report()

    def report(self):
        return {
            "seed": self.seed,
            "n_flies": self.n,
            "simulation_seconds": float(self.d.time),
            "wall_seconds": time.perf_counter() - self.started,
            "step_wall_seconds": self.step_wall,
            "neural_wall_seconds": self.neural.total_seconds if self.neural else 0.0,
            "neural_device": self.neural.brain.device if self.neural else "disabled",
            "physics": "native MuJoCo CPU",
            "vision": self.sensors.use_vision,
            "alive": sum(n.alive for n in self.needs),
            "needs": [asdict(n) for n in self.needs],
            "resources": self.resources.remaining.tolist(),
            "reward": float(np.mean([n.reward for n in self.needs])),
            "action_seconds": self.action_seconds.tolist(),
            "swatter_cycles": self.swat_events,
            "total_impulse_model_units": self.total_impulse.tolist(),
            "max_swat_force_model_units": self.max_contact_force,
            "max_actuator_limit_ratio": self.max_actuator_ratio,
            "interfly_contact_samples": self.interfly_contacts,
            "events": self.events,
            "warnings": self.d.warning.number.tolist(),
        }

    def save(self, name):
        root = OUTPUTS / name
        root.mkdir(parents=True, exist_ok=True)
        (root / "report.json").write_text(json.dumps(self.report(), indent=2))
        if self.record:
            np.savez_compressed(
                root / "physics.npz",
                qpos=self.qpos,
                qvel=self.qvel,
                ctrl=self.ctrl,
                times=[f["t"] for f in self.frames],
                seed=self.seed,
                n_flies=self.n,
            )
            (root / "telemetry.json").write_text(
                json.dumps({"mode": "replay", "n_flies": self.n, "frames": self.frames})
            )
        self.sensors.close()
        return root


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--flies", type=int, default=2)
    p.add_argument("--seconds", type=float, default=8.0)
    p.add_argument("--seed", type=int, default=101)
    p.add_argument("--device", default="cpu")
    p.add_argument("--no-neural", action="store_true")
    p.add_argument("--no-vision", action="store_true")
    p.add_argument("--record", action="store_true")
    p.add_argument("--name", default="development")
    a = p.parse_args()
    env = Habitat(
        a.flies,
        a.seed,
        device=a.device,
        neural=not a.no_neural,
        vision=not a.no_vision,
        record=a.record,
    )
    report = env.run(a.seconds)
    env.save(a.name)
    print(
        json.dumps(
            {
                k: v
                for k, v in report.items()
                if k not in ("events", "needs", "action_seconds")
            }
        ),
        flush=True,
    )
