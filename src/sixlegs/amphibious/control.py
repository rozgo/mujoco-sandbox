"""Scripted crawl and floating posture, driven by joint torques and contact."""

import math

import mujoco
import numpy as np

from .scene import FINISH_X, LEGS, ground_height

SWIM_POSE = np.array([0.0, -1.50, -0.90] * 2 + [0.0, 1.30, -2.50] * 2)


def leg_ik(point, index):
    sign = 1 if index in (0, 2) else -1
    mount = np.array([0.1934 if index < 2 else -0.1934, sign * 0.0465, 0.0])
    x, y, z = np.asarray(point) - mount
    a = sign * 0.0955
    length = 0.213
    down = math.sqrt(max(0.005, y * y + z * z - a * a))
    ab = math.atan2(y, -z) - math.atan2(a, down)
    knee = -math.acos(
        np.clip((x * x + down * down - 2 * length**2) / (2 * length**2), -0.99, 0.99)
    )
    thigh = math.atan2(-x, down) - math.atan2(
        length * math.sin(knee), length * (1 + math.cos(knee))
    )
    return np.array([ab, thigh, knee])


class Controller:
    def __init__(self, model, data):
        self.m = model
        self.mass = float(model.body_subtreemass[model.body("base").id])
        self.feet = np.array([data.site(leg + "_toe").xpos.copy() for leg in LEGS])
        self.reference = data.qpos[:3].copy()
        self.reference[2] = 0.30
        self.order = [2, 0, 3, 1]
        self.index = 0
        self.stage = "SHIFT"
        self.stage_time = 0.0
        self.start = self.feet[0].copy()
        self.end = self.start.copy()
        self.phase = "SETTLE"
        self.swimming = False
        self.events = []
        self.done = False
        self.jp = np.zeros((3, model.nv))
        self.jr = np.zeros_like(self.jp)
        self.sites = [model.site(leg + "_toe").id for leg in LEGS]
        self.target = data.qpos[7:].copy()
        self.stance = np.ones(4, dtype=bool)
        self.thrust = np.zeros(2)
        self.support_forces = np.zeros((4, 3))
        self.foot_geom = {model.geom(leg).id: i for i, leg in enumerate(LEGS)}

    def contacts(self, d):
        touching = np.zeros(4, dtype=bool)
        for contact in d.contact:
            if contact.dist > 0.002:
                continue
            for g in contact.geom:
                if int(g) in self.foot_geom:
                    touching[self.foot_geom[int(g)]] = True
        return touching

    def update(self, d, water, dt):
        m = self.m
        x, y = d.qpos[:2]
        t = d.time
        old = self.phase
        buoy = float(sum(water.buoyancy))
        if self.done:
            self.phase = "COMPLETE"
        elif t < 2.0:
            self.phase = "SETTLE"
        elif self.phase == "EXIT" or x >= 1.60:
            self.phase = "EXIT"
            self.swimming = False
        elif self.swimming or x >= 0.10:
            self.phase = "FLOATING"
            self.swimming = True
        elif self.phase == "WATER ENTRY" or x >= -1.6:
            self.phase = "WATER ENTRY"
        else:
            self.phase = "APPROACH"
        if self.phase != old:
            self.events.append({"time": float(t), "phase": self.phase, "x": float(x)})
            if old == "FLOATING":
                self.feet = np.array(
                    [
                        [
                            x + (0.1934 if i < 2 else -0.1934),
                            0.175 if i in (0, 2) else -0.175,
                            0.0,
                        ]
                        for i in range(4)
                    ]
                )
                for foot in self.feet:
                    foot[2] = ground_height(*foot[:2]) + 0.022
                self.stage = "SHIFT"
                self.stage_time = 0.0
                self.index = 0
        if x > FINISH_X and abs(d.qvel[0]) < 0.08:
            self.done = True
            for foot in self.feet:
                foot[2] = ground_height(*foot[:2]) + 0.022
        self.thrust[:] = 0.0
        self.stage_time += dt
        rotation = d.body("base").xmat.reshape(3, 3)
        actual = np.array([d.site(s).xpos.copy() for s in self.sites])
        touching = self.contacts(d)
        if self.phase == "FLOATING":
            self.target[:] += np.clip(SWIM_POSE - self.target, -0.8 * dt, 0.8 * dt)
            self.stance[:] = False
            self.support_forces[:] = 0.0
            self.reference = d.qpos[:3].copy()
        else:
            # Re-anchor the local gait to the measured base. A position-only
            # script otherwise accumulates a fictitious advance when feet slip.
            error = d.qpos[:2] - self.reference[:2]
            correction = error * np.maximum(
                0.0, 1.0 - 0.035 / max(np.linalg.norm(error), 1e-9)
            )
            self.reference[:2] += correction
            self.feet[:, :2] += correction
            self.start[:2] += correction
            self.end[:2] += correction
            for i in range(4):
                if self.stage != "SWING" or i != self.order[self.index % 4]:
                    self.feet[i, 2] = ground_height(*self.feet[i, :2]) + 0.022
            walking = t >= 2.0 and not self.done
            leg = self.order[self.index % 4]
            support = [i for i in range(4) if i != leg]
            center = np.mean(self.feet[:, :2], axis=0) - [0.015, 0.0]
            if walking:
                center += 0.5 * (
                    np.mean(self.feet[support, :2], axis=0)
                    - np.mean(self.feet[:, :2], axis=0)
                )
            if (
                walking
                and self.stage == "SHIFT"
                and self.stage_time > 0.65
                and (
                    np.linalg.norm(
                        d.subtree_com[m.body("base").id][:2]
                        - np.mean(self.feet[support, :2], axis=0)
                    )
                    < 0.045
                    or self.stage_time > 2.0
                )
            ):
                self.stage = "SWING"
                self.stage_time = 0.0
                self.start = actual[leg].copy()
                self.end = self.feet[leg].copy()
                self.end[0] += 0.12
                # Predetermined lane and exact known surface heights: this
                # is a scripted terrain fixture, not autonomous perception.
                self.end[1] = 0.175 if leg in (0, 2) else -0.175
                self.end[2] = ground_height(*self.end[:2]) + 0.022
            if walking and self.stage == "SWING":
                p = min(1.0, self.stage_time / 0.65)
                s = p * p * (3 - 2 * p)
                self.feet[leg] = (1 - s) * self.start + s * self.end
                self.feet[leg, 2] += 0.07 * math.sin(math.pi * p) ** 2
                if p >= 1.0:
                    self.stage = "LAND"
                    self.stage_time = 0.0
            elif walking and self.stage == "LAND":
                self.feet[leg] = self.end.copy()
                if self.stage_time > 0.20 and (touching[leg] or self.stage_time > 0.9):
                    self.index += 1
                    self.stage = "SHIFT"
                    self.stage_time = 0.0
            self.stance[:] = True
            if walking and self.stage == "SWING":
                self.stance[leg] = False
            change = center - self.reference[:2]
            self.reference[:2] += change * min(
                1.0, 0.10 * dt / max(np.linalg.norm(change), 1e-9)
            )
            terrain = np.mean([ground_height(*f[:2]) for f in self.feet])
            # Reach the bed before transferring weight from floats to feet.
            clearance = 0.34 if self.phase == "EXIT" and x < 2.4 else 0.30
            self.reference[2] += float(
                np.clip(terrain + clearance - self.reference[2], -0.05 * dt, 0.05 * dt)
            )
            front = np.mean([ground_height(*f[:2]) for f in self.feet[:2]])
            rear = np.mean([ground_height(*f[:2]) for f in self.feet[2:]])
            pitch = float(np.clip(-math.atan2(front - rear, 0.3868), -0.35, 0.35))
            c, s = math.cos(pitch), math.sin(pitch)
            desired_rotation = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
            for i in range(4):
                self.target[i * 3 : i * 3 + 3] = leg_ik(
                    desired_rotation.T @ (self.feet[i] - self.reference), i
                )
            self.support_forces[:] = 0.0
            active = np.flatnonzero(self.stance)
            weight = max(0.0, self.mass * 9.81 - buoy)
            matrix = np.vstack((np.ones(len(active)), self.feet[active, :2].T))
            desired = np.r_[weight, weight * (self.reference[:2] + [0.015, 0.0])]
            loads = np.linalg.lstsq(matrix, desired, rcond=None)[0]
            loads = np.maximum(loads, 0.0)
            self.support_forces[active, 2] = loads * weight / max(sum(loads), 1e-9)
        if -0.75 < x < 2.8 and np.mean(water.submerged[:4]) > 0.05:
            yaw = math.atan2(rotation[1, 0], rotation[0, 0])
            desired_speed = 0.17 if self.phase == "FLOATING" else 0.055
            total = float(np.clip(5.0 + 45.0 * (desired_speed - d.qvel[0]), -6.0, 12.0))
            moment = float(np.clip(3.0 * (-2 * y - yaw) - 2 * d.qvel[5], -0.8, 0.8))
            self.thrust[:] = np.clip(
                [total / 2 - moment / 0.2, total / 2 + moment / 0.2], -6.0, 6.0
            )
        q = d.qpos[7:]
        v = d.qvel[6:]
        torque = 120.0 * (self.target - q) - 4.0 * v + d.qfrc_bias[6:]
        for i, site in enumerate(self.sites):
            mujoco.mj_jacSite(m, d, self.jp, self.jr, site)
            torque[i * 3 : i * 3 + 3] -= (
                self.jp[:, 6 + i * 3 : 9 + i * 3].T @ self.support_forces[i]
            )
        d.ctrl[:] = np.clip(
            torque, m.actuator_ctrlrange[:, 0], m.actuator_ctrlrange[:, 1]
        )
