"""World/sensor boundary, joint actuation, evaluation and replay logs."""

import json
import math
import time
from dataclasses import asdict
from pathlib import Path

import mujoco
import numpy as np

from .agent import Detection, FacilityMap, Observation, RoverAgent, wrap
from .radio import LoRaNetwork
from .scene import BUILDINGS, MARKERS, TRACK, WHEEL_RADIUS, WHEELBASE, load_scene


class RoverSimulation:
    def __init__(self, case="healthy", seed=0, duration=150, reticulum=False):
        self.model, self.data = load_scene()
        self.case = case
        self.seed = seed
        self.duration = duration
        self.bridge = None
        if reticulum:
            from .reticulum import ReticulumBridge

            self.bridge = ReticulumBridge()
        self.agents = [RoverAgent(i, FacilityMap(BUILDINGS), seed) for i in range(6)]
        self.network = LoRaNetwork(case, seed, BUILDINGS)
        self.ticks = 0
        self.inspected = {}
        self.duplicates = 0
        self.completion_time = None
        self.distance = np.zeros(6)
        self.last_positions = self.positions().copy()
        self.collisions = set()
        self.max_torque_fraction = 0.0
        self.max_penetration = 0.0
        self.samples = []
        self.qpos = []
        self.times = []
        self.next_record = 0.0
        self.route_events_seen = [0] * 6
        self.own_sightings = [{} for _ in range(6)]
        self.coverage = set()
        self.decision_log = []
        self.drive_ids = [
            [
                self.model.actuator(f"r{i}_{s}_drive").id
                for s in ("fl", "fr", "bl", "br")
            ]
            for i in range(6)
        ]
        self.steer_ids = [
            [self.model.actuator(f"r{i}_{s}_steer").id for s in ("fl", "fr")]
            for i in range(6)
        ]
        self.body_rovers = {}
        for b in range(1, self.model.nbody):
            name = self.model.body(b).name
            if name.startswith("r") and len(name) > 1 and name[1].isdigit():
                self.body_rovers[b] = int(name[1])

    def positions(self):
        return np.array([self.data.body(f"r{i}").xpos.copy() for i in range(6)])

    def observation(self, index):
        m, d = self.model, self.data
        body = d.body(f"r{index}")
        position = body.xpos
        rotation = body.xmat.reshape(3, 3)
        yaw = math.atan2(rotation[1, 0], rotation[0, 0])
        origin = d.site(f"r{index}_sensor").xpos.copy()
        visible = []
        geom = np.zeros(1, dtype=np.int32)
        for marker in range(len(MARKERS)):
            point = d.site(f"marker_{marker}").xpos
            delta = point - origin
            distance = np.linalg.norm(delta)
            if distance > 2.5:
                continue
            if abs(wrap(math.atan2(delta[1], delta[0]) - yaw)) > math.radians(65):
                continue
            hit = mujoco.mj_ray(
                m,
                d,
                origin,
                delta / max(distance, 1e-9),
                None,
                True,
                m.body(f"r{index}").id,
                geom,
            )
            if hit >= 0 and hit < distance - 0.04:
                continue
            visible.append(
                Detection(marker, float(point[0]), float(point[1]), float(d.time))
            )
            self.own_sightings[index].setdefault(marker, float(d.time))
        nearby = []
        for j, p in enumerate(self.positions()):
            if j != index and np.linalg.norm(p[:2] - position[:2]) < 2.0:
                nearby.append((float(p[0]), float(p[1])))
        adr = m.joint(f"r{index}_free").dofadr[0]
        speed = float(d.qvel[adr] * math.cos(yaw) + d.qvel[adr + 1] * math.sin(yaw))
        return Observation(
            float(d.time),
            float(position[0]),
            float(position[1]),
            yaw,
            speed,
            tuple(visible),
            tuple(nearby),
        )

    def actuate(self, index, speed, steering):
        d = self.data
        yaw_rate = speed * math.tan(steering) / WHEELBASE
        left = speed - yaw_rate * TRACK / 2
        right = speed + yaw_rate * TRACK / 2
        # Ackermann angles at the two front knuckles, plus inside/outside wheel rates.
        if abs(steering) > 0.001:
            radius = WHEELBASE / math.tan(steering)
            angles = [
                math.atan(WHEELBASE / (radius - TRACK / 2)),
                math.atan(WHEELBASE / (radius + TRACK / 2)),
            ]
        else:
            angles = [0.0, 0.0]
        ids = self.steer_ids[index]
        d.ctrl[ids] = np.clip(angles, -0.65, 0.65)
        velocities = (
            np.array(
                [
                    left / max(math.cos(angles[0]), 0.5),
                    right / max(math.cos(angles[1]), 0.5),
                    left,
                    right,
                ]
            )
            / WHEEL_RADIUS
        )
        d.ctrl[self.drive_ids[index]] = np.clip(velocities, -12, 12)

    def step(self, record=True):
        m, d = self.model, self.data
        antennas = np.array([d.site(f"r{i}_antenna").xpos.copy() for i in range(6)])
        self.network.advance(float(d.time), antennas)
        if self.ticks % 25 == 0:
            for i, agent in enumerate(self.agents):
                obs = self.observation(i)
                received = self.network.receive(i)
                if self.bridge:
                    decoded = []
                    for raw, sender, arrival, packet in received:
                        for payload in self.bridge.decode(i, raw):
                            decoded.append((payload, sender, arrival, packet))
                    received = decoded
                speed, steer, outgoing = agent.update(obs, received)
                self.actuate(i, speed, steer)
                outgoing.sort(
                    key=lambda message: {"done": 0, "claim": 1, "seen": 2}[message.kind]
                )
                for message in outgoing:
                    frames = (
                        self.bridge.encode(i, message.encode())
                        if self.bridge
                        else [message.encode()]
                    )
                    for frame in frames:
                        self.network.enqueue(i, frame, d.time)
                for event in agent.events[self.route_events_seen[i] :]:
                    self.decision_log.append(event)
                    if event["event"] == "inspected":
                        marker = event["marker"]
                        if marker in self.inspected:
                            self.duplicates += 1
                        else:
                            self.inspected[marker] = dict(event)
                self.route_events_seen[i] = len(agent.events)
            if len(self.inspected) == len(MARKERS) and self.completion_time is None:
                self.completion_time = float(d.time)
        mujoco.mj_step(m, d)
        self.ticks += 1
        positions = self.positions()
        self.distance += np.linalg.norm(
            positions[:, :2] - self.last_positions[:, :2], axis=1
        )
        self.last_positions = positions
        for p in positions:
            self.coverage.add((round(float(p[0])), round(float(p[1]))))
        if not np.isfinite(d.qpos).all() or d.warning.number.any():
            raise RuntimeError("MuJoCo instability")
        for i in range(6):
            if d.body(f"r{i}").xpos[2] < 0.08 or d.body(f"r{i}").xmat[8] < 0.5:
                raise RuntimeError(f"Rover {i} rolled over")
        caps = np.max(np.abs(m.actuator_forcerange), axis=1)
        self.max_torque_fraction = max(
            self.max_torque_fraction, float(np.max(abs(d.actuator_force) / caps))
        )
        for c in d.contact:
            a, b = [self.body_rovers.get(int(m.geom_bodyid[g])) for g in c.geom]
            if a is not None and b is not None and a != b and c.dist < -0.001:
                self.collisions.add(tuple(sorted((a, b))))
                self.max_penetration = max(self.max_penetration, -float(c.dist))
        if record and d.time >= self.next_record:
            self.qpos.append(d.qpos.copy())
            self.times.append(float(d.time))
            self.next_record += 0.05
            if len(self.times) % 4 == 1:
                self.samples.append(self.snapshot())

    def snapshot(self):
        return {
            "time": float(self.data.time),
            "inspected": sorted(self.inspected),
            "radio": self.network.report(),
            "agents": [
                {
                    "id": a.id,
                    "role": "inspector" if a.inspector else "scout",
                    "target": a.target,
                    "belief": a.belief.copy(),
                    "completed": sorted(a.completed),
                    "source": a.target_source,
                    "queue": len(self.network.queues[a.id]),
                    "radio_on": self.network.radio_on(a.id, self.data.time),
                }
                for a in self.agents
            ],
        }

    def report(self):
        remote = []
        for e in self.decision_log:
            if e["event"] == "route_changed" and e.get("source", "").startswith(
                "radio"
            ):
                own = self.own_sightings[e["rover"]].get(e["marker"], math.inf)
                if e["time"] < own:
                    remote.append(
                        {
                            **e,
                            "first_local_sighting": own if math.isfinite(own) else None,
                        }
                    )
        return {
            "case": self.case,
            "seed": self.seed,
            "duration_s": float(self.data.time),
            "complete": len(self.inspected) == len(MARKERS),
            "unique_inspections": len(self.inspected),
            "total_markers": len(MARKERS),
            "completion_time_s": self.completion_time,
            "duplicate_inspections": self.duplicates,
            "distance_m": self.distance.tolist(),
            "coverage_cells": len(self.coverage),
            "radio": self.network.report(),
            "remote_route_changes": remote,
            "inspection_events": list(self.inspected.values()),
            "vehicle_collision_pairs": sorted(self.collisions),
            "max_vehicle_penetration_m": self.max_penetration,
            "max_actuator_fraction": self.max_torque_fraction,
            "warnings": self.data.warning.number.tolist(),
            "radio_parameters": asdict(self.network.config),
            "transport": "reticulum-plain" if self.bridge else "direct-lora",
            "reticulum_version": self.bridge.version if self.bridge else None,
            "completion_records_known_per_rover": [
                len(a.completed) for a in self.agents
            ],
            "team_completion_knowledge_fraction": sum(
                len(a.completed) for a in self.agents
            )
            / (6 * len(MARKERS)),
        }


def run(
    case="healthy", seed=0, duration=150, output=None, record=True, reticulum=False
):
    sim = RoverSimulation(case, seed, duration, reticulum)
    started = time.monotonic()
    next_log = 0
    try:
        while sim.data.time < duration:
            sim.step(record)
            if sim.data.time >= next_log:
                print(
                    f"{case} {sim.data.time:6.1f}s | inspected {len(sim.inspected)}/8 | delivered {sim.network.counts['delivered']}",
                    flush=True,
                )
                next_log += 20
    finally:
        if sim.bridge:
            sim.bridge.close()
    report = sim.report()
    report["wall_seconds"] = time.monotonic() - started
    if output:
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        if record:
            np.savez_compressed(
                output / "trajectory.npz", qpos=sim.qpos, time=sim.times
            )
            (output / "telemetry.json").write_text(
                json.dumps(
                    {
                        "snapshots": sim.samples,
                        "packets": sim.network.events,
                        "decisions": sim.decision_log,
                    },
                    separators=(",", ":"),
                )
            )
    return report
