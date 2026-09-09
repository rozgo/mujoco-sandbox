"""Localized, smooth sampled-volume buoyancy and dissipative water drag.

This is an explicitly assumed calm-water model, not CFD or a NAUT replica.
"""

import math

import mujoco
import numpy as np

from .scene import FLOAT_HALF_LENGTH, FLOAT_RADIUS, LEGS, WATER_Z


def capsule_samples(radius=FLOAT_RADIUS, half=FLOAT_HALF_LENGTH, spacing=0.025):
    axes = [np.arange(-radius + spacing / 2, radius, spacing)] * 2 + [
        np.arange(-half - radius + spacing / 2, half + radius, spacing)
    ]
    points = np.array(np.meshgrid(*axes, indexing="ij")).reshape(3, -1).T
    q = points.copy()
    q[:, 2] = np.maximum(abs(q[:, 2]) - half, 0)
    points = points[np.sum(q * q, axis=1) < radius * radius]
    volume = math.pi * radius**2 * (2 * half) + 4 / 3 * math.pi * radius**3
    return points, volume


class Water:
    def __init__(self, model, buoyancy_scale=1.0):
        self.model = model
        self.scale = buoyancy_scale
        self.level = WATER_Z
        self.rho = 1000.0
        points, volume = capsule_samples()
        self.parts = [(model.body(leg + "_float").id, points, volume) for leg in LEGS]
        # Idealized sealed central hull matching the base collision box.
        axes = [
            np.linspace(-s, s, n) for s, n in zip((0.1881, 0.04675, 0.057), (12, 4, 4))
        ]
        hull = np.array(np.meshgrid(*axes, indexing="ij")).reshape(3, -1).T
        self.parts.append((model.body("base").id, hull, 8 * 0.1881 * 0.04675 * 0.057))
        self.volume = volume
        self.submerged = np.zeros(5)
        self.centers = np.zeros((5, 3))
        self.buoyancy = np.zeros(5)
        self.drag_power = 0.0
        self.thrust = np.zeros(2)
        self.velocity = np.zeros(6)

    def apply(self, data, thrust=(0.0, 0.0)):
        m = self.model
        data.qfrc_applied[:] = 0.0
        data.xfrc_applied[:] = 0.0
        self.drag_power = 0.0
        for i, (body, points, volume) in enumerate(self.parts):
            rotation = data.xmat[body].reshape(3, 3)
            world = points @ rotation.T + data.xpos[body]
            wet = np.clip((self.level - world[:, 2]) / 0.025 + 0.5, 0.0, 1.0)
            wet *= (
                (world[:, 0] > -1.25) & (world[:, 0] < 2.85) & (abs(world[:, 1]) < 1.35)
            )
            fraction = float(np.mean(wet))
            self.submerged[i] = fraction
            center = (
                np.sum(world * wet[:, None], axis=0) / max(float(np.sum(wet)), 1e-9)
                if fraction > 0
                else data.xpos[body].copy()
            )
            self.centers[i] = center
            self.buoyancy[i] = self.rho * 9.81 * volume * fraction * self.scale
            mujoco.mj_objectVelocity(
                m, data, mujoco.mjtObj.mjOBJ_BODY, body, self.velocity, 0
            )
            angular = self.velocity[:3].copy()
            velocity = self.velocity[3:] + np.cross(angular, center - data.xipos[body])
            # Linear plus quadratic drag opposes point velocity, with rotational
            # resistance. Multiplication by wet fraction confines it to water.
            drag = -fraction * (
                7.0 * velocity + 12.0 * np.linalg.norm(velocity) * velocity
            )
            torque = -fraction * (
                0.20 * angular + 0.10 * np.linalg.norm(angular) * angular
            )
            self.drag_power += float(drag @ velocity + torque @ angular)
            force = drag + np.array([0.0, 0.0, self.buoyancy[i]])
            mujoco.mj_applyFT(m, data, force, torque, center, body, data.qfrc_applied)
        self.thrust[:] = np.asarray(thrust) * min(
            1.0, float(np.mean(self.submerged[:4])) / 0.25
        )
        base = m.body("base").id
        forward = data.xmat[base].reshape(3, 3)[:, 0]
        for i, side in enumerate(("left", "right")):
            mujoco.mj_applyFT(
                m,
                data,
                forward * self.thrust[i],
                np.zeros(3),
                data.site(side + "_thruster").xpos,
                base,
                data.qfrc_applied,
            )
