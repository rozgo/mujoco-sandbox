"""Initial-form ground targets and measured posture, never a runtime override.

The imitation target provides corrective supervision from measured joint state.
State scores are evaluation diagnostics: no gradient passes through CPU MuJoCo.
"""

import mujoco
import numpy as np

from embodied_fly.wing_motion import CONFIG


class GroundPosture:
    wing_kp = 0.02
    wing_kd = 0.00015

    def __init__(self, env, qref):
        self.env = env
        self.qref = np.asarray(qref).copy()
        m, t = env.model, env.template
        self.wings = np.array([m.actuator(m.joint(j).name).id for j in t.wing_joint_ids])
        self.position_actuators = np.flatnonzero(
            (m.actuator_trntype == mujoco.mjtTrn.mjTRN_JOINT) & (m.actuator_biasprm[:, 1] < 0)
        )
        ctrl = np.zeros(m.nu)
        for a in self.position_actuators:
            j = m.actuator_trnid[a, 0]
            length = m.actuator_gear[a, 0] * self.qref[m.jnt_qposadr[j]]
            ctrl[a] = (
                -(m.actuator_biasprm[a, 0] + m.actuator_biasprm[a, 1] * length)
                / m.actuator_gainprm[a, 0]
            )
        self.rest_action = (
            2 * (np.clip(ctrl, t.low, t.high) - t.low) / (t.high - t.low) - 1
        ).astype(np.float32)
        legs = np.array(["_T" in name for name in t.joint_names])
        wings = np.isin(t.joint_ids, t.wing_joint_ids)
        self.groups = {"legs": legs, "wings": wings, "body": ~(legs | wings)}

    def wing_targets(self, ids=None):
        """Training labels from current measured state, never executed here."""
        e, t = self.env, self.env.template
        if ids is None:
            ids = np.arange(e.n)
        q = e.fields["qpos"][ids][:, t.wing_angle_indices]
        v = e.fields["qvel"][ids][:, t.wing_velocity_indices]
        torque = self.wing_kp * (self.qref[t.wing_angle_indices] - q) - self.wing_kd * v
        return np.clip(torque / CONFIG.joint_torque_limit, -1, 1).astype(np.float32)

    def targets(self, actions, task_ids):
        """Stand: all initial position targets. Walk: preserve leg teacher.

        Both ground tasks get feedback torques toward initial wing angles with
        zero angular velocity. Hover labels pass through unchanged. No pose writes.
        """
        result = actions.copy()
        result[task_ids == 0] = self.rest_action
        ids = np.flatnonzero(task_ids != 2)
        result[np.ix_(ids, self.wings)] = self.wing_targets(ids)
        return result

    def measure(self):
        e, t = self.env, self.env.template
        delta = e.fields["qpos"][:, t.qpos_indices] - self.qref[t.qpos_indices]
        velocity = e.fields["qvel"][:, t.qvel_indices]
        values = {}
        for name, mask in self.groups.items():
            values[name + "_angle_mse_rad2"] = np.square(delta[:, mask]).mean(1)
            values[name + "_velocity_mse_rad2_s2"] = np.square(velocity[:, mask]).mean(1)
        values["wing_max_deviation_rad"] = np.abs(delta[:, self.groups["wings"]]).max(1)
        values["body_height_loss_fraction"] = np.maximum(
            0, 1 - e.fields["qpos"][:, 2] / self.qref[2]
        )
        # Group-balanced score prevents 90+ body/leg hinges diluting six wings.
        cost = sum(values[name + "_angle_mse_rad2"] / 0.15**2 for name in self.groups) / 3
        values["initial_form_score"] = np.exp(
            -cost - (values["body_height_loss_fraction"] / 0.1) ** 2
        )
        return values

    def report(self):
        t = self.env.template
        return {
            "target": "canonical initial hinge pose; zero wing angular velocity on stand and walk",
            "initial_hinge_angles_rad": dict(
                zip(t.joint_names, self.qref[t.qpos_indices].tolist())
            ),
            "initial_root_height_cm": float(self.qref[2]),
            "wing_kp": self.wing_kp,
            "wing_kd": self.wing_kd,
            "stand_foot_adhesion": 0.0,
            "method": "bounded corrective imitation labels; measured posture scores are not PPO rewards",
            "passive_joints": "included in posture measurement; only existing actuators receive targets",
            "runtime_override": False,
            "gates": {
                "ground_wing_max_deviation_rad": 0.2,
                "ground_wing_velocity_rms_rad_s": 2.0,
                "stand_group_angle_rms_rad": 0.15,
                "stand_max_height_loss_fraction": 0.1,
            },
        }
