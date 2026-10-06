"""Measured robot motion: programmed servo, hold and target approach. SI units.

Movement comes only from mj_step with the scene's force-limited position
actuators. The servo is programmed control, not learned: per axis it adds
inertial, gravity/bias, viscous and Coulomb feedforward plus a bounded integral
term to the actuator's own PD, by offsetting the position command:

    ctrl = q_ref + (kv*v_ref + M*a_ref + qfrc_bias + damping*v_ref + I) / kp

so the applied force is kp*(q_ref-q) + kv*(v_ref-qdot) + feedforward + I,
clipped by the actuator force range. References are minimum-jerk joint-space
segments; for this Cartesian gantry that is a straight tool-tip line.
"""

from dataclasses import dataclass, field
import math

import mujoco
import numpy as np

from .scene import JOINTS, load_scene

JOINT_NAMES = tuple(JOINTS)
# The insertion and retainer slides hang against gravity at their retracted
# stops, where the command range cannot offset the sag. Carry both 0.5 mm out.
CARRY_M = {"insertion": 5e-4, "retainer": 5e-4}
HOVER_M = 1e-3  # needle tip above the target site
SAFE_M = 5e-3  # transit height above the target sites
WORK_Z_M = -6e-3  # Z stage working height; 2 mm from its lower travel limit
INTEGRAL_TIME_S = .05
TARGETS = tuple(range(6))
ROBOT_BODIES = ("stage_x", "stage_y", "stage_z", "insertion", "retainer")


def minimum_jerk(q0, q1, duration, t):
    """Position, velocity and acceleration on a minimum-jerk segment."""
    s = np.clip(t/duration, 0., 1.)
    dq = q1-q0
    p = 10*s**3-15*s**4+6*s**5
    v = (30*s**2-60*s**3+30*s**4)/duration
    a = (60*s-180*s**2+120*s**3)/duration**2
    return q0+dq*p, dq*v, dq*a


@dataclass
class Reference:
    """Piecewise minimum-jerk joint reference: (target q, move s, hold s) segments."""
    start: np.ndarray
    segments: list = field(default_factory=list)

    def add(self, target, move_s, hold_s=0.):
        self.segments.append((np.asarray(target, float), float(move_s), float(hold_s)))
        return self

    @property
    def duration(self):
        return sum(m+h for _, m, h in self.segments)

    def phase(self, t):
        """Segment index and whether the reference is moving at time t."""
        q0, elapsed = self.start, 0.
        for i, (q1, move, hold) in enumerate(self.segments):
            if t < elapsed+move+hold or i == len(self.segments)-1:
                return i, t < elapsed+move
            elapsed += move+hold
            q0 = q1
        return len(self.segments)-1, False

    def __call__(self, t):
        q0, elapsed = self.start, 0.
        for q1, move, hold in self.segments:
            if t < elapsed+move:
                return minimum_jerk(q0, q1, move, t-elapsed)
            elapsed += move
            if t < elapsed+hold:
                return q1.copy(), np.zeros_like(q1), np.zeros_like(q1)
            elapsed += hold
            q0 = q1
        return q0.copy(), np.zeros_like(q0), np.zeros_like(q0)


class Servo:
    def __init__(self, model, integral_time_s=INTEGRAL_TIME_S, names=JOINT_NAMES):
        self.model = model
        self.joint = np.array([model.joint(n).id for n in names])
        self.dof = model.jnt_dofadr[self.joint]
        self.qadr = model.jnt_qposadr[self.joint]
        self.act = np.array([model.actuator(n).id for n in names])
        self.kp = model.actuator_gainprm[self.act, 0].copy()
        self.kv = -model.actuator_biasprm[self.act, 2].copy()
        self.damping = model.dof_damping[self.dof].copy()
        self.friction = model.dof_frictionloss[self.dof].copy()
        self.force_limit = model.actuator_forcerange[self.act, 1].copy()
        self.ki = self.kp/integral_time_s
        self.integral = np.zeros(len(names))
        self._acc = np.zeros(model.nv)
        self._inertial = np.zeros(model.nv)

    def command(self, data, q_ref, v_ref, a_ref):
        """Set data.ctrl after mj_step1 (positions, velocities, bias and inertia current)."""
        m, dt = self.model, self.model.opt.timestep
        q = data.qpos[self.qadr]
        error = q_ref-q
        self._acc[:] = 0
        self._acc[self.dof] = a_ref
        mujoco.mj_mulM(m, data, self._inertial, self._acc)
        coulomb = self.friction*np.tanh(v_ref/1e-4)
        feedforward = self._inertial[self.dof]+data.qfrc_bias[self.dof]+self.damping*v_ref+coulomb
        # Bounded integral with conditional integration (no wind-up into saturation).
        limit = 2*self.friction+1e-12
        self.integral = np.clip(self.integral+self.ki*error*dt, -limit, limit)
        ctrl = q_ref+(self.kv*v_ref+feedforward+self.integral)/self.kp
        lo, hi = m.actuator_ctrlrange[self.act].T
        data.ctrl[self.act] = np.clip(ctrl, lo, hi)
        return error


def tip_offset(model):
    """Needle tip at zero joint position; the gantry maps joints linearly to the tip."""
    scratch = mujoco.MjData(model)  # scratch kinematics, never stepped
    mujoco.mj_kinematics(model, scratch)
    return scratch.site("needle_tip").xpos.copy()


def joints_for_tip(model, tip, z_stage=WORK_Z_M, retainer=CARRY_M["retainer"]):
    """Inverse kinematics for a world tip position.

    Z travels only 8 mm down, so the design reaches the surface with the needle
    slide: Z parks at a working height and the insertion slide supplies the rest.
    """
    q = np.zeros(len(JOINT_NAMES))
    delta = np.asarray(tip)-tip_offset(model)
    q[0], q[1], q[2] = delta[0], delta[1], z_stage
    q[3] = z_stage-delta[2]  # insertion axis is world -Z
    q[4] = retainer
    scratch = mujoco.MjData(model)
    scratch.qpos[[model.jnt_qposadr[model.joint(n).id] for n in JOINT_NAMES]] = q
    mujoco.mj_kinematics(model, scratch)
    if np.max(abs(scratch.site("needle_tip").xpos-tip)) > 1e-12:
        raise ValueError("Tip kinematics are not the expected Cartesian map")
    for i, name in enumerate(JOINT_NAMES):
        lo, hi = model.jnt_range[model.joint(name).id]
        if not lo <= q[i] <= hi:
            raise ValueError(f"{name} target {q[i]:.6f} m outside travel")
    return q


def hover_joints(model, data, target, height=HOVER_M):
    return joints_for_tip(model, data.site(f"target_{target}").xpos+(0, 0, height))


def robot_geoms(model):
    bodies = {model.body(n).id for n in ROBOT_BODIES}
    return np.array([model.geom_bodyid[g] in bodies for g in range(model.ngeom)])


def run(model, data, reference, *, record_every=1, callback=None):
    """Advance physics under the servo for the reference duration; return telemetry.

    The same loop serves headless runs, tests, recording and the live viewer
    (through `callback`, which must not change state).
    """
    servo = Servo(model)
    is_robot = robot_geoms(model)
    tip_site = model.site("needle_tip").id
    needle, phantom = model.geom("needle").id, model.geom("tissue_phantom").id
    steps = round(reference.duration/model.opt.timestep)
    keys = ("time", "qpos", "qvel", "q_ref", "v_ref", "ctrl", "force", "tip", "tip_ref", "segment", "moving")
    trace = {k: [] for k in keys}
    tip0 = tip_offset(model)
    contacts, min_gap, warnings = [], math.inf, 0
    for step in range(steps):
        mujoco.mj_step1(model, data)
        t = data.time
        q_ref, v_ref, a_ref = reference(t)
        servo.command(data, q_ref, v_ref, a_ref)
        for c in data.contact[:data.ncon]:
            if is_robot[c.geom1] != is_robot[c.geom2]:
                contacts.append({"time_s": float(t), "geoms": [model.geom(c.geom1).name, model.geom(c.geom2).name],
                                 "distance_m": float(c.dist)})
        if step % 10 == 0:
            min_gap = min(min_gap, mujoco.mj_geomDistance(model, data, needle, phantom, .05, None))
        mujoco.mj_step2(model, data)
        warnings += int(sum(w.number for w in data.warning))
        if step % record_every == 0 or step == steps-1:
            mujoco.mj_kinematics(model, data)  # derived pose for the post-step state
            segment, moving = reference.phase(data.time)
            r = reference(data.time)
            trace["time"].append(data.time)
            trace["qpos"].append(data.qpos[servo.qadr].copy())
            trace["qvel"].append(data.qvel[servo.dof].copy())
            trace["q_ref"].append(r[0])
            trace["v_ref"].append(r[1])
            trace["ctrl"].append(data.ctrl[servo.act].copy())
            trace["force"].append(data.actuator_force[servo.act].copy())
            trace["tip"].append(data.site_xpos[tip_site].copy())
            trace["tip_ref"].append(tip0+np.array((r[0][0], r[0][1], r[0][2]-r[0][3])))
            trace["segment"].append(segment)
            trace["moving"].append(moving)
        if callback is not None:
            callback(data)
        if not (np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()):
            break
    trace = {k: np.array(v) for k, v in trace.items()}
    return trace, {"contacts": contacts, "min_needle_phantom_gap_m": float(min_gap),
                   "mujoco_warnings": warnings, "force_limit_n": servo.force_limit.tolist()}


def hold_trial(duration=2.):
    """Start at rest on a command (initialization), then hold under the servo."""
    model, data = load_scene()
    q = np.array([0., 0., 0., CARRY_M["insertion"], CARRY_M["retainer"]])
    qadr = [model.jnt_qposadr[model.joint(n).id] for n in JOINT_NAMES]
    data.qpos[qadr] = q
    data.qvel[:] = 0
    mujoco.mj_forward(model, data)
    reference = Reference(q).add(q, 0., duration)
    trace, events = run(model, data, reference)
    drift = np.max(np.abs(trace["qpos"]-q), axis=0)
    tip_drift = float(np.max(np.linalg.norm(trace["tip"]-trace["tip"][0], axis=1)))
    saturated = np.max(np.abs(trace["force"])/np.array(events["force_limit_n"]), axis=0)
    report = {"duration_s": duration, "axis_drift_m": dict(zip(JOINT_NAMES, drift.tolist())),
              "tip_drift_m": tip_drift, "max_force_fraction": dict(zip(JOINT_NAMES, saturated.tolist())),
              "contacts": len(events["contacts"]), "mujoco_warnings": events["mujoco_warnings"],
              "simulated_s": float(trace["time"][-1])}
    report["passed"] = bool(max(drift) < 1e-6 and max(saturated) < 1 and not events["contacts"]
                            and not events["mujoco_warnings"])
    return report, trace


def approach_reference(model, data, targets, start=None, move_s=1.5, transit_s=1.0, hold_s=.8):
    """Raise to transit height, move above each target, descend to hover, hold."""
    start = np.zeros(len(JOINT_NAMES)) if start is None else start
    reference = Reference(start)
    reference.hover_segments, reference.labels = [], []
    for i, target in enumerate(targets):
        hover = hover_joints(model, data, target)
        above = hover_joints(model, data, target, SAFE_M)
        if i == 0:
            reference.add(above, move_s)
            reference.labels.append(("transit", target))
        else:
            previous_above = hover_joints(model, data, targets[i-1], SAFE_M)
            reference.add(previous_above, transit_s*.5)
            reference.add(above, transit_s)
            reference.labels += [("raise", targets[i-1]), ("transit", target)]
        reference.hover_segments.append(len(reference.segments))
        reference.add(hover, transit_s*.6, hold_s)
        reference.labels.append(("descend", target))
    return reference


def evaluate(trace, events, reference, targets, settle_window_s=.3, gate_m=1e-5):
    """Per-target hover error after each descent; tracking and force margins."""
    error = trace["tip"]-trace["tip_ref"]
    lateral, vertical = np.linalg.norm(error[:, :2], axis=1), np.abs(error[:, 2])
    windows, t = [], 0.
    for _, move, hold in reference.segments:
        windows.append((t+move, t+move+hold))
        t += move+hold
    hovers = []
    for target, index in zip(targets, reference.hover_segments):
        end_move, end_hold = windows[index]
        window = (trace["time"] >= end_move+settle_window_s) & (trace["time"] <= end_hold+1e-9)
        hovers.append({"target": int(target), "settled_after_s": settle_window_s,
                       "max_lateral_error_m": float(lateral[window].max()),
                       "max_vertical_error_m": float(vertical[window].max()),
                       "passed": bool(lateral[window].max() < gate_m and vertical[window].max() < gate_m)})
    moving = trace["moving"].astype(bool)
    force_fraction = np.max(np.abs(trace["force"])/np.array(events["force_limit_n"]), axis=0)
    report = {"targets": hovers,
              "tracking_rms_while_moving_m": float(np.sqrt(np.mean(np.sum(error[moving]**2, axis=1)))),
              "tracking_max_while_moving_m": float(np.linalg.norm(error[moving], axis=1).max()),
              "max_speed_m_s": float(np.linalg.norm(trace["qvel"][:, :3], axis=1).max()),
              "max_force_fraction": dict(zip(JOINT_NAMES, force_fraction.tolist())),
              "robot_environment_contacts": len(events["contacts"]),
              "min_needle_phantom_gap_m": events["min_needle_phantom_gap_m"],
              "mujoco_warnings": events["mujoco_warnings"], "simulated_s": float(trace["time"][-1]),
              "finite": bool(np.isfinite(trace["qpos"]).all())}
    report["passed"] = bool(all(h["passed"] for h in hovers) and not events["contacts"] and report["finite"]
                            and not events["mujoco_warnings"] and max(force_fraction) < 1)
    return report


READY = np.array([0., 0., 0., CARRY_M["insertion"], CARRY_M["retainer"]])


def start_ready(model, data):
    """Initialization: the validated hold pose, at rest."""
    qadr = [model.jnt_qposadr[model.joint(n).id] for n in JOINT_NAMES]
    data.qpos[qadr] = READY
    data.qvel[:] = 0
    mujoco.mj_forward(model, data)
    return model, data


def tour_reference(model, data, targets=TARGETS):
    return approach_reference(model, data, list(targets), start=READY)


def approach_suite():
    """Hold gate, each target approached from ready, then one six-target tour."""
    hold, _ = hold_trial()
    singles = []
    for target in TARGETS:
        model, data = start_ready(*load_scene())
        reference = tour_reference(model, data, [target])
        trace, events = run(model, data, reference)
        singles.append(evaluate(trace, events, reference, [target]))
    model, data = start_ready(*load_scene())
    reference = tour_reference(model, data)
    trace, events = run(model, data, reference)
    reference.force_limits = events["force_limit_n"]
    tour = evaluate(trace, events, reference, list(TARGETS))
    report = {"hold": hold, "single_approaches": singles, "tour": tour,
              "passed": bool(hold["passed"] and all(r["passed"] for r in singles) and tour["passed"])}
    return report, trace, reference
