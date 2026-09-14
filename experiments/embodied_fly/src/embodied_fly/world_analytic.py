"""Reduced analytical predictor; never writes to a live MuJoCo environment.

Independent massless wing hinges use implicit velocity integration, capped PD
torques and hard joint stops. The body uses initial-articulation locked inertia,
the declared instantaneous force law, rigid-body rotation and gravity. It omits
subsequent non-wing articulation, contact and soft joint-limit forces. Those
approximations are measured against the unchanged full MuJoCo plant.
"""

import mujoco
import numpy as np

from embodied_fly.wing_motion import WingMotionForces
from embodied_fly.world_data import DT, layout


def rotation_step(omega_dt):
    angle = np.linalg.norm(omega_dt, axis=-1)
    skew = np.zeros((*angle.shape, 3, 3))
    x, y, z = np.moveaxis(omega_dt, -1, 0)
    skew[..., 0, 1], skew[..., 0, 2] = -z, y
    skew[..., 1, 0], skew[..., 1, 2] = z, -x
    skew[..., 2, 0], skew[..., 2, 1] = -y, x
    a = np.sinc(angle / np.pi)
    b = 0.5 * np.sinc(angle / (2 * np.pi)) ** 2
    return np.eye(3) + a[..., None, None] * skew + b[..., None, None] * (skew @ skew)


def wing_step(q, v, command, parameters, dt):
    p = parameters
    target = p["lower"] + (np.clip(command, -1, 1) + 1) * 0.5 * (p["upper"] - p["lower"])
    actuator = p["kp"] * (target - q) - p["kv"] * v
    force = (
        np.clip(actuator, -p["limit"], p["limit"])
        - p["damping"] * v
        - p["stiffness"] * (q - p["spring"])
    )
    # Canonical MuJoCo Euler treats passive damping implicitly, while actuator
    # velocity feedback remains explicit. Match that actual compiled integrator.
    dv = dt * force / (p["armature"] + dt * p["damping"])
    v = v + dv
    raw = q + dt * v
    q = np.clip(raw, p["lower"], p["upper"])
    v = np.where(raw == q, v, 0)
    return q, v


class AnalyticalFly:
    def __init__(self, model):
        if model.opt.integrator != mujoco.mjtIntegrator.mjINT_EULER:
            raise ValueError("Analytical wing step requires the canonical Euler integrator")
        self.model = model
        self.indices = layout(model)
        ids = np.asarray(self.indices["wing_action"])
        joints = model.actuator_trnid[ids, 0]
        dofs = model.jnt_dofadr[joints]
        self.parameters = {
            "lower": model.actuator_ctrlrange[ids, 0],
            "upper": model.actuator_ctrlrange[ids, 1],
            "kp": model.actuator_gainprm[ids, 0],
            "kv": -model.actuator_biasprm[ids, 2],
            "limit": model.actuator_forcerange[ids, 1],
            "damping": model.dof_damping[dofs],
            "stiffness": model.jnt_stiffness[joints],
            "spring": model.qpos_spring[model.jnt_qposadr[joints]],
            "armature": model.dof_armature[dofs],
        }
        self.data = mujoco.MjData(model)
        self.matrix = np.empty((model.nv, model.nv))

    def inertia(self, qpos):
        matrices = []
        for row in qpos:
            self.data.qpos[:] = row
            self.data.qvel[:] = 0
            mujoco.mj_forward(self.model, self.data)
            mujoco.mj_fullM(self.model, self.data, self.matrix)
            m = self.matrix
            # Translational Schur complement gives locked inertia about COM.
            j = m[3:6, 3:6] - m[3:6, :3] @ np.linalg.solve(m[:3, :3], m[:3, 3:6])
            matrices.append(j.copy())
        return np.asarray(matrices)

    def predict(self, initial, actions, initial_qpos):
        x = np.asarray(initial, dtype=np.float64).copy()
        actions = np.asarray(actions)
        n, steps, _ = actions.shape
        if x.shape != (n, 30) or initial_qpos.shape != (n, self.model.nq):
            raise ValueError("One complete initial physical state per analytical rollout")
        inertia = self.inertia(initial_qpos)
        inverse = np.linalg.inv(inertia)
        law = WingMotionForces(self.model, n)
        force, path = [], []
        dt = self.model.opt.timestep
        if abs(dt - 0.001) > 1e-12:
            raise ValueError("Analytical comparator requires the declared 1 kHz plant")
        for t in range(steps):
            for _ in range(round(DT / dt)):
                rotation = x[:, 6:15].reshape(n, 3, 3)
                omega = x[:, 15:18]
                velocity_body = np.einsum("nji,nj->ni", rotation, x[:, 3:6])
                wrench = law.advance(
                    x[:, 18:24],
                    x[:, 24:30],
                    rotation,
                    np.concatenate((omega, velocity_body), axis=-1),
                    dt,
                )
                force.append(law.lift.copy() / law.weight)
                torque_body = np.einsum("nji,nj->ni", rotation, wrench[:, 3:])
                momentum = np.einsum("nij,nj->ni", inertia, omega)
                acceleration = np.einsum(
                    "nij,nj->ni", inverse, torque_body - np.cross(omega, momentum)
                )
                x[:, 15:18] += dt * acceleration
                rotation = rotation @ rotation_step(x[:, 15:18] * dt)
                x[:, 6:15] = rotation.reshape(n, 9)
                x[:, 3:6] += dt * (wrench[:, :3] / law.mass + self.model.opt.gravity)
                x[:, :3] += dt * x[:, 3:6]
                x[:, 18:24], x[:, 24:30] = wing_step(
                    x[:, 18:24],
                    x[:, 24:30],
                    actions[:, t, self.indices["wing_action"]],
                    self.parameters,
                    dt,
                )
            path.append(x.copy())
        return np.stack(path, axis=1), np.stack(force, axis=1)


def sampled_lift(metrics):
    """Lift at predicted 500 Hz wing states; not the 1 kHz truth observer."""
    q = np.asarray(metrics)[..., 18:24].reshape(*metrics.shape[:-1], 2, 3)
    v = np.asarray(metrics)[..., 24:30].reshape(*metrics.shape[:-1], 2, 3)
    activity = np.clip(np.abs(v[..., 0]) / 50, 0, 1.4)
    return 1.6 * (activity * (0.8 + 0.2 * np.cos(q[..., 2] + 1))).mean(axis=-1)
