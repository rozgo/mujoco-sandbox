"""Identical finite-horizon sling controllers; only the wind forecast changes."""

import math

import numpy as np
from scipy.linalg import expm

from .scene import CABLE, DRONE_MASS, LOAD_MASS, ROTOR_XY, START, YAW_SIGNS

HORIZON = 20
CONTROL_DT = 0.1
DRAG_DRONE = 0.14
DRAG_LOAD = 0.045


def reference(t):
    x = START[0]
    z = START[2]
    vx = vz = 0.0
    if t > 1:
        p = np.clip((t - 1) / 3, 0, 1)
        s = p * p * (3 - 2 * p)
        z = START[2] + (0.1 + 1.8 - START[2]) * s
        if p < 1:
            vz = (0.1 + 1.8 - START[2]) * 6 * p * (1 - p) / 3
    if t > 6:
        p = np.clip((t - 6) / 24, 0, 1)
        s = p * p * (3 - 2 * p)
        x = START[0] + 5.4 * s
        if p < 1:
            vx = 5.4 * 6 * p * (1 - p) / 24
    if t > 32:
        p = np.clip((t - 32) / 5, 0, 1)
        s = p * p * (3 - 2 * p)
        z = 1.9 + (1.185 - 1.9) * s
        if p < 1:
            vz = (1.185 - 1.9) * 6 * p * (1 - p) / 5
    return np.array([x, 0.0, z]), np.array([vx, 0.0, vz])


def drag(wind, velocity, coefficient):
    relative = np.asarray(wind) - np.asarray(velocity)
    return coefficient * np.linalg.norm(relative, axis=-1, keepdims=True) * relative


class Controller:
    def __init__(self, model):
        self.model = model
        self.phase = "LIFT"
        self.released = False
        self.release_time = None
        self.horizontal = np.zeros(2)
        self.command = np.full(4, (DRONE_MASS + LOAD_MASS) * 9.81 / 4)
        self.predicted_load = np.zeros((HORIZON, 2))
        self.wind_prediction = np.zeros((HORIZON, 2))
        self.mixer = np.vstack(
            (np.ones(4), ROTOR_XY[:, 1], -ROTOR_XY[:, 0], 0.018 * YAW_SIGNS)
        )
        md, ml, L = DRONE_MASS, LOAD_MASS, CABLE
        # Fifth state: actual horizontal rotor force. Attitude and rotor lag
        # matter: treating commanded force as instantaneous excites the sling.
        lag = 0.12
        a = np.array(
            [
                [0, 1, 0, 0, 0],
                [0, 0, ml * 9.81 / md, 0, 1 / md],
                [0, 0, 0, 1, 0],
                [0, 0, -9.81 * (1 + ml / md) / L, -0.15, -1 / (md * L)],
                [0, 0, 0, 0, -1 / lag],
            ]
        )
        b = np.array([[0], [0], [0], [0], [1 / lag]])
        e = np.array(
            [[0, 0], [1 / md, 0], [0, 0], [-1 / (md * L), 1 / (ml * L)], [0, 0]]
        )
        augmented = np.zeros((8, 8))
        augmented[:5, :5] = a
        augmented[:5, 5:6] = b
        augmented[:5, 6:] = e
        discrete = expm(augmented * CONTROL_DT)
        self.A = discrete[:5, :5]
        self.B = discrete[:5, 5]
        self.E = discrete[:5, 6:]
        # Penalize payload error, aircraft error, speed, swing and swing rate.
        load = np.array([1, 0, L, 0, 0])
        q = 65 * np.outer(load, load) + np.diag([5, 4, 12, 1.5, 0.01])
        qend = q * 3
        self.Q = np.kron(np.eye(HORIZON), q)
        self.Q[-5:, -5:] = qend
        self.S = np.vstack(
            [np.linalg.matrix_power(self.A, i + 1) for i in range(HORIZON)]
        )
        self.T = np.zeros((5 * HORIZON, HORIZON))
        for i in range(HORIZON):
            for j in range(i + 1):
                self.T[i * 5 : (i + 1) * 5, j] = (
                    np.linalg.matrix_power(self.A, i - j) @ self.B
                )
        difference = np.eye(HORIZON) - np.eye(HORIZON, k=-1)
        self.H = (
            self.T.T @ self.Q @ self.T
            + 0.08 * np.eye(HORIZON)
            + 0.10 * difference.T @ difference
        )
        self.K = np.linalg.solve(self.H, self.T.T @ self.Q)

    def plan(self, d, forecast):
        pos = d.body("drone").xpos.copy()
        payload = d.body("payload").xpos.copy()
        vel = d.qvel[:3]
        loadvel = d.qvel[6:9]
        refs = np.array(
            [reference(d.time + (i + 1) * CONTROL_DT)[0] for i in range(HORIZON)]
        )
        refvel = np.array(
            [reference(d.time + (i + 1) * CONTROL_DT)[1] for i in range(HORIZON)]
        )
        # Both controllers receive the same current full-field observation.
        # Forecast may advance it with FNO/PINO or persist it unchanged.
        points = refs[:, :2] + (pos[:2] - reference(d.time)[0][:2]) * 0.5
        winds = forecast(np.arange(1, HORIZON + 1) * CONTROL_DT, points)
        self.wind_prediction = winds.copy()
        fd = drag(winds, refvel[:, :2], DRAG_DRONE)
        fl = drag(winds, refvel[:, :2], DRAG_LOAD)
        theta = (payload[:2] - pos[:2]) / CABLE
        omega = (loadvel[:2] - vel[:2]) / CABLE
        for axis in range(2):
            applied = (
                sum(d.actuator_force) * d.body("drone").xmat.reshape(3, 3)[axis, 2]
            )
            state = np.array([pos[axis], vel[axis], theta[axis], omega[axis], applied])
            disturbances = []
            z = np.zeros(5)
            for j in range(HORIZON):
                z = self.A @ z + self.E @ np.array([fd[j, axis], fl[j, axis]])
                disturbances.extend(z)
            desired = np.column_stack(
                (
                    refs[:, axis],
                    refvel[:, axis],
                    np.zeros(HORIZON),
                    np.zeros(HORIZON),
                    np.zeros(HORIZON),
                )
            ).ravel()
            free = self.S @ state + np.array(disturbances)
            controls = -self.K @ (free - desired)
            self.horizontal[axis] = np.clip(controls[0], -6.0, 6.0)
            predicted = (free + self.T @ np.clip(controls, -6.0, 6.0)).reshape(-1, 5)
            self.predicted_load[:, axis] = predicted[:, 0] + CABLE * predicted[:, 2]

    def update(self, d):
        ref, rv = reference(d.time)
        if d.time < 6:
            self.phase = "LIFT"
        elif d.time < 30:
            self.phase = "CROSSWINDS"
        elif d.time < 32:
            self.phase = "ALIGN"
        else:
            self.phase = "LOWER PACKAGE"
        if self.released:
            self.phase = "DELIVERED"
            ref[2] = 1.6
        mass = DRONE_MASS + (0 if self.released else LOAD_MASS)
        if self.released:
            self.horizontal[:] = np.clip(
                -6 * (d.qpos[:2] - ref[:2]) - 4 * d.qvel[:2], -5, 5
            )
        force = np.r_[
            self.horizontal,
            mass * (9.81 + 10 * (ref[2] - d.qpos[2]) + 5 * (rv[2] - d.qvel[2])),
        ]
        z = force / max(np.linalg.norm(force), 1e-6)
        y = np.cross(z, [1.0, 0.0, 0.0])
        y /= max(np.linalg.norm(y), 1e-9)
        x = np.cross(y, z)
        desired = np.column_stack((x, y, z))
        rotation = d.body("drone").xmat.reshape(3, 3)
        error = desired.T @ rotation - rotation.T @ desired
        e = 0.5 * np.array([error[2, 1], error[0, 2], error[1, 0]])
        torque = -8.0 * e - 0.80 * d.qvel[3:6]
        total = float(force @ rotation[:, 2])
        self.command[:] = np.clip(
            np.linalg.solve(self.mixer, np.r_[total, torque]), 0, 10.0
        )
        # Release only after physical contact with the destination platform.
        touching = any(
            set(c.geom)
            == {self.model.geom("package").id, self.model.geom("delivery_platform").id}
            and c.dist < 0.001
            for c in d.contact
        )
        if d.time > 35 and touching and np.linalg.norm(d.qvel[6:9]) < 0.15:
            self.released = True
            self.release_time = self.release_time or float(d.time)
            self.model.tendon_range[0, 1] = 1000.0
            self.model.tendon_width[0] = 0.0


def swing_angle(d):
    delta = d.body("payload").xpos - d.site("hook").xpos
    return math.degrees(math.atan2(np.linalg.norm(delta[:2]), max(1e-6, -delta[2])))
