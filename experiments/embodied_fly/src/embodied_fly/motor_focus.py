"""Motor-first curriculum: stand, walk, hover in one body with one graph actor.

References label bounded joint commands only during training. Evaluation runs the
same motor-only checkpoint with no utility selector, teacher or wing oscillator.
"""

import argparse
import json
import time
from collections import deque
from pathlib import Path

import mujoco
import numpy as np
import torch

from embodied_fly.batch import FlyBatch
from embodied_fly.brain import EmbodiedBrain, initialize_extended_actor, load_malecns
from embodied_fly.braking import BrakingTeacher
from embodied_fly.evaluate import load_actor
from embodied_fly.ground_posture import GroundPosture
from embodied_fly.motion_flight import initialize
from embodied_fly.motor_parameter_subset import (
    WingFeedbackSubset,
    WingOutputSubset,
    WingResidualSubset,
)
from embodied_fly.motor_retention import (
    FrozenMotorReference,
    ground_nonwing_loss,
    hover_start_weights,
    task_loss,
    task_mixtures,
)
from embodied_fly.neural_view import NeuralProjection
from embodied_fly.physical_contract import physical_contract
from embodied_fly.provenance import evidence, sha256, utc_now
from embodied_fly.state_hover import RECIPE as STATE_HOVER_RECIPE
from embodied_fly.train import synchronize
from embodied_fly.wing_motion import CONFIG
from embodied_fly.wing_position import reference_torque_to_position, wing_actuators
from embodied_fly.wing_response_learning import response_loss

TASKS = ("stand", "walk", "hover")


def motor_error(action, target, wing_channels, hovering, ground_wing_weight=0.0):
    """Keep all motor targets, with explicit wing accuracy in each task."""
    if not np.isfinite(ground_wing_weight) or ground_wing_weight < 0:
        raise ValueError("Ground wing weight must be finite and nonnegative")
    wing = (action[:, wing_channels] - target[:, wing_channels]).square().mean(-1)
    weight = torch.where(hovering, 2.0, ground_wing_weight)
    return (action - target).square().mean(-1) + weight * wing


class MotorTasks:
    def __init__(
        self,
        env,
        seed,
        task_set="all",
        wing_angle_perturbation=0.0,
        wing_speed_perturbation=0.0,
    ):
        if task_set not in ("all", "ground"):
            raise ValueError("Unknown motor curriculum task set")
        if (
            not np.isfinite([wing_angle_perturbation, wing_speed_perturbation]).all()
            or min(wing_angle_perturbation, wing_speed_perturbation) < 0
        ):
            raise ValueError("Wing reset perturbations must be finite and nonnegative")
        self.env = env
        self.rng = np.random.default_rng(seed)
        self.perturb_rng = np.random.default_rng(seed + 1973)
        self.wing_angle_perturbation = wing_angle_perturbation
        self.wing_speed_perturbation = wing_speed_perturbation
        self.task_ids = np.arange(env.n) % (3 if task_set == "all" else 2)
        self.ground = {
            k: getattr(env.template.data, k).copy() for k in ("qpos", "qvel", "act", "ctrl")
        }
        self.air_action = initialize(env.template)
        self.air = {k: getattr(env.template.data, k).copy() for k in self.ground}
        env.template.reset()
        self.start = np.zeros((env.n, 3))
        self.heading = np.zeros(env.n)
        self.reset(np.arange(env.n))

    def reset(self, ids):
        ids = np.asarray(ids, np.int64)
        if not len(ids):
            return
        hovering = self.task_ids[ids] == 2
        state = {
            k: np.stack([(self.air if h else self.ground)[k] for h in hovering])
            for k in self.ground
        }
        heading = self.rng.uniform(-0.15, 0.15, len(ids))
        state["qpos"][:, 3:7] = 0
        state["qpos"][:, 3] = np.cos(heading / 2)
        state["qpos"][:, 6] = np.sin(heading / 2)
        state["qpos"][hovering, 2] = self.rng.uniform(1.8, 2.2, int(hovering.sum()))
        # Training initialization only. The corrective target stays the original
        # canonical resting pose; neither live state nor evaluation is overwritten.
        ground = np.flatnonzero(~hovering)
        t, m = self.env.template, self.env.model
        if self.wing_angle_perturbation and len(ground):
            key = np.ix_(ground, t.wing_angle_indices)
            angles = state["qpos"][key] + self.perturb_rng.uniform(
                -self.wing_angle_perturbation, self.wing_angle_perturbation, (len(ground), 6)
            )
            limited = m.jnt_limited[t.wing_joint_ids].astype(bool)
            limits = m.jnt_range[t.wing_joint_ids]
            state["qpos"][key] = np.clip(
                angles,
                np.where(limited, limits[:, 0], -np.inf),
                np.where(limited, limits[:, 1], np.inf),
            )
        if self.wing_speed_perturbation and len(ground):
            state["qvel"][np.ix_(ground, t.wing_velocity_indices)] += self.perturb_rng.uniform(
                -self.wing_speed_perturbation, self.wing_speed_perturbation, (len(ground), 6)
            )
        self.env.reset(ids, state=state)
        self.env.command[ids] = 0
        self.env.command[ids, 0] = (self.task_ids[ids] == 1).astype(float)
        self.env.requested_height_cm[ids] = np.where(hovering, state["qpos"][:, 2], 0)
        self.env.needs[ids] = 0
        self.start[ids] = state["qpos"][:, :3]
        self.heading[ids] = heading

    def failed(self):
        height = self.env.fields["qpos"][:, 2]
        upright = self.env.fields["xmat"][:, self.env.template.thorax_id, 8]
        return (upright < 0.5) | (height < np.where(self.task_ids == 2, 0.5, 0.06))


class MotorTeacher:
    """Batched current-state corrections; this object never controls evaluation."""

    def __init__(
        self,
        tasks,
        teacher_path,
        device,
        ground_posture=False,
        hover_reference="clock",
        stand_initial_form=False,
        walking_reference="receding",
    ):
        if hover_reference not in ("clock", "state", "state-position"):
            raise ValueError("Unknown hover reference")
        if tasks.env.preset == "wing_position" and (
            not ground_posture or hover_reference not in ("state", "state-position")
        ):
            raise ValueError(
                "Position pilot requires ground posture and measured-state hover reference"
            )
        self.hover_reference = hover_reference
        if stand_initial_form and not ground_posture:
            raise ValueError("Initial-form stand supervision requires ground posture")
        self.stand_initial_form = stand_initial_form
        self.tasks, self.env = tasks, tasks.env
        if walking_reference not in ("receding", "anchored"):
            raise ValueError("Unknown walking reference")
        self.walking_reference = walking_reference
        self.ground = BrakingTeacher(
            self.env,
            teacher_path,
            device,
            track_command=True,
            reference_tasks=tasks if walking_reference == "anchored" else None,
        )
        self.posture = (
            GroundPosture(self.env, tasks.ground["qpos"]) if ground_posture else None
        )
        self.channels = np.array(
            [
                self.env.model.actuator(self.env.model.joint(j).name).id
                for j in self.env.template.wing_joint_ids
            ]
        )

    @torch.no_grad()
    def act(self, ground_actions=None):
        e, t, c = self.env, self.tasks, CONFIG
        actions = (
            self.ground.act().cpu().numpy()
            if ground_actions is None
            else np.asarray(ground_actions).copy()
        )
        if actions.shape != (e.n, e.model.nu) or not np.isfinite(actions).all():
            raise ValueError("Invalid ground reference actions")
        if self.posture is not None and ground_actions is None:
            actions = self.posture.targets(actions, t.task_ids)
        elif self.posture is not None:
            if self.stand_initial_form:
                actions[t.task_ids == 0] = self.posture.rest_action
            ids = np.flatnonzero(t.task_ids != 2)
            actions[np.ix_(ids, self.channels)] = self.posture.wing_targets(ids)
        ids = np.flatnonzero(t.task_ids == 2)
        if not len(ids):
            return actions
        if self.hover_reference in ("state", "state-position"):
            from embodied_fly.state_hover import wing_commands

            actions[ids] = t.air_action
            forward_error = None
            if self.hover_reference == "state-position":
                displacement = e.fields["qpos"][ids, :3] - t.start[ids]
                displacement[:, 2] = 0
                rotation = e.fields["xmat"][ids, e.template.thorax_id].reshape(-1, 3, 3)
                forward_error = np.einsum("ni,ni->n", displacement, rotation[:, :, 0])
            actions[np.ix_(ids, self.channels)] = wing_commands(
                e.fields["qpos"][ids][:, e.template.wing_angle_indices],
                e.fields["qvel"][ids][:, e.template.wing_velocity_indices],
                e.velocity()[ids],
                e.fields["qpos"][ids, 2],
                e.requested_height_cm[ids],
                e.model.qpos_spring[e.template.wing_angle_indices],
                forward_position_error=forward_error,
            )
            if e.preset == "wing_position":
                actions[np.ix_(ids, self.channels)] = reference_torque_to_position(
                    e.model,
                    actions[np.ix_(ids, self.channels)],
                    e.fields["qpos"][ids][:, e.template.wing_angle_indices],
                    e.fields["qvel"][ids][:, e.template.wing_velocity_indices],
                    e.control_dt,
                )
            return actions
        angle = e.fields["qpos"][ids][:, e.template.wing_angle_indices]
        velocity = e.fields["qvel"][ids][:, e.template.wing_velocity_indices]
        body_velocity = e.velocity()[ids]
        omega = 2 * np.pi * 12
        phase = omega * e.ages[ids] * e.control_dt
        amplitude = np.clip(
            c.reference_sweep_speed / (c.lift_weight_multiplier * 48)
            + 0.8 * (e.requested_height_cm[ids] - e.fields["qpos"][ids, 2])
            - 0.045 * body_velocity[:, 5],
            0.12,
            1.15,
        )
        stroke = 0.7 + np.clip(-0.1 * body_velocity[:, 3], -0.6, 0.6)
        desired = np.column_stack((amplitude * np.sin(phase), stroke, -np.ones(len(ids))))
        speed = np.column_stack((amplitude * omega * np.cos(phase), np.zeros((len(ids), 2))))
        acceleration = np.column_stack(
            (-amplitude * omega**2 * np.sin(phase), np.zeros((len(ids), 2)))
        )
        desired, speed, acceleration = (
            np.tile(a, (1, 2)) for a in (desired, speed, acceleration)
        )
        spring = e.model.qpos_spring[e.template.wing_angle_indices]
        torque = (
            c.angular_armature * acceleration
            + c.joint_damping * speed
            + c.joint_stiffness * (desired - spring)
            + 0.02 * (desired - angle)
            + 0.00015 * (speed - velocity)
        )
        actions[ids] = t.air_action
        actions[np.ix_(ids, self.channels)] = np.clip(torque / c.joint_torque_limit, -1, 1)
        return actions


def load_motor_actor(path, graph_path, device):
    old, parent = load_actor(path, graph_path, device)
    if old.observation_size == 397:
        actor = old
    else:
        graph, sensory, descending, motor = load_malecns(graph_path)
        actor = EmbodiedBrain(
            graph, sensory, descending, motor, 397, 78, old.internal_steps, 14
        )
        initialize_extended_actor(actor, parent["state_dict"])
        actor.core.adjacency, actor.core.transpose = old.core.adjacency, old.core.transpose
        actor = actor.to(device)
    actor.set_motor_only()
    return actor, parent


def review(args):
    """Matched three-task physical reference or autonomous student evaluation."""
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    provenance = evidence()
    torch.set_num_threads(4)
    device = torch.device(args.device)
    actor = checkpoint = None
    response = getattr(args, "wing_response", None)
    if args.mode != "reference":
        actor, checkpoint = load_actor(args.resume, args.graph, device)
        recorded = checkpoint["physical_contract"].get("wing_response", "filtered")
        if response is not None and response != recorded:
            raise ValueError("Wing response must match the checkpoint's recorded physics")
        response = recorded
    env = FlyBatch(
        3,
        3,
        14,
        preset=getattr(args, "preset", "wing_motion"),
        wing_response=response or "filtered",
        physics_hz=checkpoint["physical_contract"]["physics_hz"] if checkpoint else None,
    )
    tasks = MotorTasks(env, args.seed)
    posture = GroundPosture(env, tasks.ground["qpos"])
    posture_enabled = getattr(args, "ground_posture", False)
    reference = memory = projection = None
    if args.mode == "reference":
        reference = MotorTeacher(
            tasks,
            args.teacher,
            device,
            posture_enabled,
            getattr(args, "hover_reference", "clock"),
            getattr(args, "stand_initial_form", False),
            getattr(args, "walking_reference", "receding"),
        )
    else:
        posture_enabled |= checkpoint.get("config", {}).get("ground_posture", False)
        if checkpoint["physical_contract"] != physical_contract(env.model):
            raise ValueError("Evaluation must use the checkpoint's exact physical fly")
        if not actor.motor_only or actor.observation_size != 397:
            raise ValueError(
                "Motor review requires an explicit motor-only 397-input checkpoint"
            )
        actor.eval()
        memory = actor.initial_state(3)
        if args.neural_view:
            projection = NeuralProjection.from_graph(args.graph, device)
    mujoco.mj_saveModel(env.model, str(args.output / "model.mjb"))
    setup = time.perf_counter() - started
    rows = []
    maps = []
    metrics = []
    started = time.perf_counter()
    failure = None
    for step in range(round(args.seconds / env.control_dt)):
        obs = env.observation()
        with torch.no_grad():
            if actor is not None:
                result = actor(torch.as_tensor(obs, device=device), memory)
                memory = result.state
                action = result.action.cpu().numpy()
                if projection is not None and step % 10 == 0:
                    maps.append(projection.project(memory).astype(np.float16))
            else:
                action = reference.act()
        rows.append(
            {
                "qpos": env.fields["qpos"].copy(),
                "qvel": env.fields["qvel"].copy(),
                "activation": env.fields["act"].copy(),
                "ctrl": env.fields["ctrl"].copy(),
                "observation": obs.copy(),
                "action": action.copy(),
                "time": np.full(3, step * env.control_dt),
                "command": env.command.copy(),
                "requested_height_cm": env.requested_height_cm.copy(),
                "wing_activity": env.wing_forces.activity.copy(),
                "wing_wrench": env.wing_forces.wrench.copy(),
                "wing_actuator_force": env.fields["actuator_force"][
                    :, wing_actuators(env.model)
                ].copy(),
                "initial_qpos": np.repeat(tasks.ground["qpos"][None], 3, axis=0),
            }
        )
        try:
            env.step(action)
        except RuntimeError as error:
            failure = str(error)
            break
        target = tasks.start.copy()
        distance = (step + 1) * env.control_dt * env.command[:, 0]
        target[:, 0] += distance * np.cos(tasks.heading)
        target[:, 1] += distance * np.sin(tasks.heading)
        v = env.velocity()
        metrics.append(
            {
                "height": env.fields["qpos"][:, 2].copy() * 0.01,
                "upright": env.fields["xmat"][:, env.template.thorax_id, 8].copy(),
                "speed_error": v[:, 3] - env.command[:, 0],
                "yaw_error": v[:, 2],
                "root_error": np.linalg.norm(env.fields["qpos"][:, :3] - target, axis=1)
                * 0.01,
                "forbidden_load": env.forbidden_peak.copy() / env.body_weight,
                "wing_torque_max": np.abs(
                    env.fields["actuator_force"][:, wing_actuators(env.model)]
                ).max(axis=1),
                **posture.measure(),
            }
        )
    elapsed = time.perf_counter() - started
    results = []
    for i, task in enumerate(TASKS):
        arrays = {k: np.stack([r[k][i] for r in rows]) for k in rows[0]}
        if projection is not None:
            arrays.update(
                neural_map=np.stack([m[i] for m in maps]),
                neural_occupancy=projection.occupancy,
                neural_map_stride=10,
            )
        np.savez_compressed(args.output / f"{task}.npz", **arrays)
        measurements = {k: np.array([m[k][i] for m in metrics]) for k in metrics[0]}
        stable = measurements["upright"].min() > 0.5 and measurements["height"].min() > (
            0.005 if task == "hover" else 0.0006
        )
        rmse = lambda key, values=measurements: float(np.sqrt(np.mean(values[key] ** 2)))
        support = float(measurements["forbidden_load"].max())
        passed = stable and support < 0.1 and failure is None
        if task == "hover":
            passed &= rmse("root_error") < 0.005
        else:
            passed &= rmse("speed_error") < 0.5 and rmse("yaw_error") < 0.5
        pose_metrics = {
            name + "_angle_rms_rad": float(
                np.sqrt(measurements[name + "_angle_mse_rad2"].mean())
            )
            for name in posture.groups
        }
        pose_metrics.update(
            wing_velocity_rms_rad_s=float(
                np.sqrt(measurements["wings_velocity_mse_rad2_s2"].mean())
            ),
            wing_max_deviation_rad=float(measurements["wing_max_deviation_rad"].max()),
            max_body_height_loss_fraction=float(
                measurements["body_height_loss_fraction"].max()
            ),
            mean_initial_form_score=float(measurements["initial_form_score"].mean()),
        )
        pose_pass = None
        if task != "hover":
            pose_pass = (
                pose_metrics["wing_max_deviation_rad"] < 0.2
                and pose_metrics["wing_velocity_rms_rad_s"] < 2
            )
            if task == "stand":
                pose_pass &= (
                    max(pose_metrics[k + "_angle_rms_rad"] for k in posture.groups) < 0.15
                )
                pose_pass &= pose_metrics["max_body_height_loss_fraction"] < 0.1
            if posture_enabled:
                passed &= pose_pass
        results.append(
            {
                "case": task,
                "stable": bool(stable),
                "success": bool(passed),
                "root_tracking_rmse_m": rmse("root_error"),
                "speed_rmse_cm_s": rmse("speed_error"),
                "yaw_rmse_rad_s": rmse("yaw_error"),
                "minimum_height_m": float(measurements["height"].min()),
                "post_action_wing_torque_max_CGS": float(
                    measurements["wing_torque_max"].max()
                ),
                "minimum_upright": float(measurements["upright"].min()),
                "max_forbidden_ground_force_over_weight": support,
                "state_sha256": sha256(args.output / f"{task}.npz"),
                "posture": pose_metrics,
                "ground_posture_pass": bool(pose_pass) if pose_pass is not None else None,
            }
        )
    report = {
        "provenance": provenance,
        "completed_utc": utc_now(),
        "mode": args.mode,
        "motor_only": actor is not None,
        "teacher_present": reference is not None,
        "hover_reference": getattr(args, "hover_reference", "clock") if reference else None,
        "state_hover_recipe": dict(STATE_HOVER_RECIPE)
        if reference and reference.hover_reference in ("state", "state-position")
        else None,
        "reference_forward_position_hold": {
            "enabled": reference is not None and reference.hover_reference == "state-position",
            "stroke_speed_gain_rad_per_cm_s": 0.6,
            "stroke_position_gain_rad_per_cm": 0.3,
            "body_force_or_pose_override": False,
            "scope": "fore-aft wing-stroke correction; no lateral position servo",
        },
        "student_present": actor is not None,
        "controller": "student" if actor else "reference",
        "one_checkpoint_for_all_cases": actor is not None,
        "policy_acceptance_eligible": actor is not None,
        "physical_preset": env.preset,
        "physical_contract": physical_contract(env.model),
        "physics_hz": 5000,
        "control_hz": 500,
        "environment": env.template.report(),
        "model_sha256": sha256(args.output / "model.mjb"),
        "setup_seconds": setup,
        "stepping_and_capture_seconds": elapsed,
        "simulated_seconds_per_world": len(metrics) * env.control_dt,
        "parallel_physics_worlds": 3,
        "seed": args.seed,
        "checkpoint_sha256": sha256(args.resume) if actor else None,
        "teacher_sha256": sha256(args.teacher) if reference else None,
        "neural_view": projection.report() if projection is not None else None,
        "warning_count": int(env.fields["warning"].sum()),
        "numerical_failure": failure,
        "ground_posture": posture.report(),
        "posture_gates_enabled": bool(posture_enabled),
        "results": results,
        "scope": "motor primitives from declared initial states; no takeoff, landing or learned utility",
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {k: v for k, v in report.items() if k not in ("environment", "provenance")}
        ),
        flush=True,
    )
    return report


def train(args):
    if args.worlds < 3 or min(args.seconds, args.sequence, args.lr) <= 0:
        raise ValueError("Need positive training settings and at least three worlds")
    if not 0 <= args.teacher_mix <= 1:
        raise ValueError("Teacher mixture must be in [0,1]")
    subset_mode = getattr(args, "trainable_subset", "all")
    if subset_mode not in ("all", "wing-output", "wing-residual", "wing-feedback"):
        raise ValueError("Unknown motor training parameter subset")
    retain_ground = getattr(args, "retain_ground", False)
    ground_weight = getattr(args, "ground_retention_weight", 1.0)
    task_loss({0: torch.tensor(0.0)}, ground_weight)
    walking_reference = getattr(args, "walking_reference", "receding")
    if walking_reference == "anchored" and retain_ground:
        raise ValueError("Choose anchored walking supervision or retained actor labels")
    if ground_weight != 1.0 and not retain_ground and walking_reference != "anchored":
        raise ValueError("Ground retention weighting requires --retain-ground")
    nonwing_retention_weight = getattr(args, "nonwing_retention_weight", 0.0)
    feedback_lr = getattr(args, "feedback_lr", 0.003)
    if (
        not np.isfinite([nonwing_retention_weight, feedback_lr]).all()
        or nonwing_retention_weight < 0
        or feedback_lr <= 0
    ):
        raise ValueError(
            "Finite nonnegative retention and positive feedback learning rate required"
        )
    if nonwing_retention_weight and not retain_ground:
        raise ValueError("Non-wing retention requires a frozen ground reference")
    mixture_by_task = task_mixtures(
        np.arange(3),
        args.teacher_mix,
        getattr(args, "hover_teacher_mix", None),
        getattr(args, "walk_teacher_mix", None),
    )
    start_duration = getattr(args, "hover_start_seconds", 0.05)
    start_weight = getattr(args, "hover_start_weight", 1.0)
    hover_start_weights(np.arange(3), np.zeros(3), 0.002, start_duration, start_weight)
    response_weight = getattr(args, "wing_response_loss", 0.0)
    response_worlds = getattr(args, "wing_response_worlds", 8)
    if not np.isfinite(response_weight) or response_weight < 0 or response_worlds < 1:
        raise ValueError("Invalid wing-response learning settings")
    if response_weight and not getattr(args, "ground_posture", False):
        raise ValueError("Wing-response supervision requires ground posture targets")
    if getattr(args, "preset", "wing_motion") == "wing_position" and response_weight:
        raise ValueError("Legacy torque-response loss is incompatible with position targets")
    response_rng = np.random.default_rng(args.seed + 1101)
    hover_feedback_weight = getattr(args, "hover_feedback_loss", 0.0)
    if not np.isfinite(hover_feedback_weight) or hover_feedback_weight < 0:
        raise ValueError("Finite nonnegative hover feedback weight required")
    if hover_feedback_weight and (
        getattr(args, "preset", "wing_motion") != "wing_position"
        or getattr(args, "task_set", "all") != "all"
        or getattr(args, "hover_reference", "clock") != "state"
    ):
        raise ValueError(
            "Hover feedback requires position body, all tasks and state reference"
        )
    hover_feedback_rng = np.random.default_rng(args.seed + 2203)
    synthetic_hover_inputs = 0
    switch_seconds = getattr(args, "ground_switch_seconds", 0.0)
    if not np.isfinite(switch_seconds) or switch_seconds < 0:
        raise ValueError("Finite nonnegative ground command interval required")
    if switch_seconds and (
        not retain_ground
        or not getattr(args, "ground_posture", False)
        or not getattr(args, "stand_initial_form", False)
        or getattr(args, "task_set", "all") != "all"
        or getattr(args, "preset", "wing_motion") != "wing_position"
        or np.any(mixture_by_task[:2] != 0)
        or subset_mode != "all"
    ):
        raise ValueError(
            "Command curriculum requires unassisted ground actions, all-task position learning, ground retention and initial-form targets"
        )
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    provenance = evidence()
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    actor, parent = load_motor_actor(args.resume, args.graph, device)
    actor.train()
    response = parent.get("physical_contract", {}).get("wing_response", "filtered")
    if getattr(args, "wing_response", None) not in (None, response):
        raise ValueError("Wing response must match the checkpoint's recorded physics")
    env = FlyBatch(
        args.worlds,
        args.threads,
        14,
        preset=getattr(args, "preset", "wing_motion"),
        wing_response=response,
        physics_hz=parent.get("physical_contract", {}).get("physics_hz"),
    )
    if parent.get("motor_only") and parent.get("physical_contract") != physical_contract(
        env.model
    ):
        raise ValueError("Motor continuation must use the parent's exact physical fly")
    tasks = MotorTasks(
        env,
        args.seed,
        getattr(args, "task_set", "all"),
        getattr(args, "wing_angle_perturbation", 0.0),
        getattr(args, "wing_speed_perturbation", 0.0),
    )
    active_tasks = np.unique(tasks.task_ids)
    teacher = MotorTeacher(
        tasks,
        args.teacher,
        device,
        getattr(args, "ground_posture", False),
        getattr(args, "hover_reference", "clock"),
        getattr(args, "stand_initial_form", False),
        walking_reference,
    )
    posture = GroundPosture(env, tasks.ground["qpos"])
    mujoco.mj_saveModel(env.model, str(args.output / "model.mjb"))
    memory = actor.initial_state(args.worlds)
    retainer = FrozenMotorReference(actor, args.worlds) if retain_ground else None
    curriculum = None
    if switch_seconds:
        from embodied_fly.motor_curriculum import GroundCommandCurriculum

        curriculum = GroundCommandCurriculum(tasks, args.transition_worlds, switch_seconds)
    subset_class = {
        "wing-output": WingOutputSubset,
        "wing-residual": WingResidualSubset,
        "wing-feedback": WingFeedbackSubset,
    }.get(subset_mode)
    subset = subset_class(actor, teacher.channels) if subset_class else None
    optimizer = torch.optim.Adam(
        [
            {"params": actor.sensor_extension.parameters(), "lr": feedback_lr},
            {"params": actor.wing_residual.parameters(), "lr": args.lr},
        ]
        if subset_mode == "wing-feedback"
        else [p for p in actor.parameters() if p.requires_grad],
        lr=args.lr,
    )
    retention_initial = (
        {k: v.detach().cpu().clone() for k, v in retainer.actor.state_dict().items()}
        if retainer is not None
        else None
    )
    mixture = mixture_by_task[tasks.task_ids]
    nonwing_channels = np.setdiff1d(np.arange(env.model.nu), teacher.channels)
    same_history_body_delta = 0.0 if subset is not None and retainer is not None else None
    frozen = {
        k: v.detach().clone()
        for k, v in actor.state_dict().items()
        if k.startswith(("utility_head.", "intention_encoder."))
    }
    initial_core = {k: v.detach().clone() for k, v in actor.core.named_parameters()}
    transitions = updates = 0
    startup_presentations = 0
    episodes = []
    trace = deque(maxlen=64)
    gradient_audit = None
    subset_gradient_audit = None
    collection_seconds = optimization_seconds = 0.0
    synchronize(device)
    setup = time.perf_counter() - started
    started_utc = utc_now()
    started = time.perf_counter()
    with (args.output / "progress.jsonl").open("x") as log:
        while time.perf_counter() - started < args.seconds:
            start = time.perf_counter()
            memory = memory.detach()
            losses = []
            response_errors = []
            retention_errors = []
            per_task = [[] for _ in TASKS]
            hover_feedback_error = None
            for sequence_step in range(args.sequence):
                if curriculum is not None:
                    curriculum.advance(memory)
                obs = env.observation()
                observation = torch.as_tensor(obs, device=device)
                ground_actions = (
                    retainer.act(observation).cpu().numpy() if retainer is not None else None
                )
                if curriculum is not None:
                    ground_actions = curriculum.supervised_actions(ground_actions, teacher)
                target = torch.as_tensor(teacher.act(ground_actions), device=device)
                memory_before = memory
                if hover_feedback_weight and sequence_step == 0:
                    from embodied_fly.hover_feedback import response_loss as hover_response

                    hover_feedback_error, count = hover_response(
                        actor, memory_before, env, obs, tasks.task_ids, 8, hover_feedback_rng
                    )
                    synthetic_hover_inputs += count
                result = actor(observation, memory)
                memory = result.state
                flight = torch.as_tensor(tasks.task_ids == 2, device=device)
                error = motor_error(
                    result.action,
                    target,
                    teacher.channels,
                    flight,
                    getattr(args, "ground_wing_loss", 0.0),
                )
                startup_weights = hover_start_weights(
                    tasks.task_ids, env.ages, env.control_dt, start_duration, start_weight
                )
                error = error * torch.as_tensor(startup_weights, device=device)
                startup_presentations += int((startup_weights > 1).sum())
                group_losses = {
                    i: error[torch.as_tensor(tasks.task_ids == i, device=device)].mean()
                    for i in active_tasks
                }
                loss = task_loss(group_losses, ground_weight)
                if nonwing_retention_weight:
                    retention_error = ground_nonwing_loss(
                        result.action, ground_actions, tasks.task_ids, nonwing_channels
                    )
                    loss = loss + nonwing_retention_weight * retention_error
                    retention_errors.append(float(retention_error.detach()))
                if response_weight:
                    correction = response_loss(
                        actor,
                        memory_before,
                        env,
                        posture,
                        obs,
                        tasks.task_ids,
                        response_worlds,
                        response_rng,
                    )
                    loss = loss + response_weight * correction
                    response_errors.append(float(correction.detach()))
                losses.append(loss)
                for i, value in group_losses.items():
                    per_task[i].append(float(value.detach()))
                student = result.action.detach().cpu().numpy()
                if same_history_body_delta is not None:
                    same_history_body_delta = max(
                        same_history_body_delta,
                        float(
                            np.abs(
                                student[:, nonwing_channels]
                                - ground_actions[:, nonwing_channels]
                            ).max()
                        ),
                    )
                target_np = target.cpu().numpy()
                executed = (1 - mixture[:, None]) * student + mixture[:, None] * target_np
                trace.append(
                    {
                        "qpos": env.fields["qpos"].copy(),
                        "qvel": env.fields["qvel"].copy(),
                        "activation": env.fields["act"].copy(),
                        "ctrl": env.fields["ctrl"].copy(),
                        "observation": obs.copy(),
                        "student_action": student.copy(),
                        "teacher_action": target_np.copy(),
                        "executed_action": executed.copy(),
                        "teacher_mix": mixture.copy(),
                        "supervision_weight": startup_weights.copy(),
                        "wing_activity": env.wing_forces.activity.copy(),
                        "wing_wrench": env.wing_forces.wrench.copy(),
                        "wing_actuator_force": env.fields["actuator_force"][
                            :, wing_actuators(env.model)
                        ].copy(),
                        "requested_height_cm": env.requested_height_cm.copy(),
                        "task_ids": tasks.task_ids.copy(),
                        "command": env.command.copy(),
                    }
                )
                env.step(executed)
                transitions += args.worlds
                failed = tasks.failed()
                done = failed | (env.ages >= round(args.episode_seconds / env.control_dt))
                ids = np.flatnonzero(done)
                for i in ids:
                    info = None
                    if failed[i]:
                        window = list(trace)[-min(int(env.ages[i]), len(trace)) :]
                        file = args.output / f"failure_{len(episodes):05d}.npz"
                        np.savez_compressed(
                            file, **{k: np.stack([r[k][i] for r in window]) for k in window[0]}
                        )
                        info = {
                            "file": file.name,
                            "sha256": sha256(file),
                            "frames": len(window),
                        }
                    episodes.append(
                        {
                            "task": TASKS[tasks.task_ids[i]],
                            "command_transition_world": bool(
                                curriculum is not None and i in curriculum.ids
                            ),
                            "failed": bool(failed[i]),
                            "simulated_seconds": float(env.ages[i] * env.control_dt),
                            "teacher_mix": float(mixture[i]),
                            "failure_trace": info,
                        }
                    )
                if curriculum is not None:
                    curriculum.prepare_reset(ids)
                tasks.reset(ids)
                memory = actor.reset_worlds(memory, torch.as_tensor(done, device=device))
                if retainer is not None:
                    retainer.reset(torch.as_tensor(done, device=device))
            synchronize(device)
            collection_seconds += time.perf_counter() - start
            start = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            loss = torch.stack(losses).mean()
            if hover_feedback_error is not None:
                loss = loss + hover_feedback_weight * hover_feedback_error
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite motor loss")
            loss.backward()
            if subset is not None and subset_gradient_audit is None:
                subset_gradient_audit = subset.gradient_audit()
            if subset is None and gradient_audit is None:
                gradient_audit = {
                    n: {
                        "l2": float(p.grad.norm()),
                        "finite": bool(torch.isfinite(p.grad).all()),
                    }
                    for n, p in actor.core.named_parameters()
                }
                if not all(x["finite"] and x["l2"] > 0 for x in gradient_audit.values()):
                    raise RuntimeError("Core must receive motor gradients")
            torch.nn.utils.clip_grad_norm_(actor.parameters(), 1)
            optimizer.step()
            synchronize(device)
            optimization_seconds += time.perf_counter() - start
            updates += 1
            if updates == 1 or updates % 10 == 0:
                row = {
                    "seconds": time.perf_counter() - started,
                    "updates": updates,
                    "transitions": transitions,
                    "task_motor_loss": {
                        task: float(np.mean(values)) if values else None
                        for task, values in zip(TASKS, per_task)
                    },
                    "wing_response_loss": float(np.mean(response_errors))
                    if response_errors
                    else None,
                    "hover_feedback_response_loss": float(hover_feedback_error.detach())
                    if hover_feedback_error is not None
                    else None,
                    "synthetic_hover_inputs": synthetic_hover_inputs,
                    "ground_nonwing_retention_mse": float(np.mean(retention_errors))
                    if retention_errors
                    else None,
                    "completed_episodes": len(episodes),
                    "failed_episodes": sum(e["failed"] for e in episodes),
                    "posture_by_task": {
                        task: {
                            key: float(value[tasks.task_ids == i].mean())
                            for key, value in posture.measure().items()
                        }
                        if i in active_tasks
                        else None
                        for i, task in enumerate(TASKS)
                    },
                }
                log.write(json.dumps(row) + "\n")
                log.flush()
                print(json.dumps(row), flush=True)
    synchronize(device)
    elapsed = time.perf_counter() - started
    subset_report = subset.verify_and_report() if subset is not None else {"mode": "all"}
    if subset is not None:
        subset_report["same_history_nonwing_action_max_delta"] = same_history_body_delta
        subset_report["same_history_world_actions_checked"] = (
            transitions if same_history_body_delta is not None else 0
        )
        if (
            subset_mode != "wing-feedback"
            and same_history_body_delta is not None
            and same_history_body_delta > 1e-4
        ):
            raise RuntimeError(
                "Non-wing outputs diverged from the frozen parent on identical histories"
            )
    assert all(torch.equal(actor.state_dict()[k], v) for k, v in frozen.items())
    if retainer is not None:
        assert all(
            torch.equal(v.detach().cpu(), retention_initial[k])
            for k, v in retainer.actor.state_dict().items()
        )
    retention = {
        "enabled": retain_ground,
        "checkpoint_sha256": sha256(args.resume) if retain_ground else None,
        "weights_unchanged": True if retain_ground else None,
        "ground_task_loss_weight": ground_weight,
        "additional_nonwing_mse_weight": nonwing_retention_weight,
        "independent_recurrent_state": retain_ground,
        "runtime_module": False,
        "transition_world_label_override": curriculum is not None,
        "ground_wing_targets": "initial-pose corrective labels"
        if teacher.posture is not None
        else "reference actor"
        if retain_ground
        else "walking teacher",
    }
    config = {k: v for k, v in vars(args).items() if not isinstance(v, Path)}
    config["internal_steps"] = actor.internal_steps
    config["state_hover_recipe"] = (
        dict(STATE_HOVER_RECIPE) if teacher.hover_reference == "state" else None
    )
    checkpoint = {
        "state_dict": {k: v.detach().cpu() for k, v in actor.state_dict().items()},
        "optimizer_state_dict": optimizer.state_dict(),
        "motor_only": True,
        "wing_residual_enabled": actor.wing_residual is not None,
        "wing_residual_hidden": actor.wing_residual_hidden,
        "physical_contract": physical_contract(env.model),
        "observation_size": 397,
        "sensor_extension_size": 14,
        "action_size": 78,
        "config": config,
        "source_commit": provenance["source_commit"],
        "graph_sha256": parent["graph_sha256"],
        "graph_metadata_sha256": sha256(args.graph / "brain.npz"),
        "parent_checkpoint_sha256": sha256(args.resume),
        "method": "motor-only online imitation on actual teacher/student-mixture physical states",
        "parameter_subset": subset_report,
        "ground_retention": retention,
        "command_curriculum": curriculum.report() if curriculum is not None else None,
        "teacher_mix_by_task": dict(zip(TASKS, mixture_by_task.tolist())),
        "startup_supervision": {
            "hover_seconds": start_duration,
            "weight": start_weight,
            "weighted_presentations": startup_presentations,
            "extra_physics_transitions": 0,
            "runtime_module": False,
        },
        "ground_posture": posture.report() if teacher.posture is not None else None,
        "wing_response_supervision": {
            "weight": response_weight,
            "ground_worlds_sampled_per_action": response_worlds if response_weight else 0,
            "kind": "paired synthetic wing feedback, not extra physical transitions",
        },
        "hover_feedback_supervision": {
            "weight": hover_feedback_weight,
            "synthetic_sensor_inputs": synthetic_hover_inputs,
            "paired_worlds_per_chunk": 8 if hover_feedback_weight else 0,
            "height_delta_cm": 0.2,
            "vertical_speed_delta_cm_s": 2.0,
            "difference_scales": {"height": 0.02, "vertical_speed": 0.01},
            "runtime_module": False,
            "extra_physical_transitions": 0,
        },
    }
    torch.save(checkpoint, args.output / "actor.pt")
    report = {
        "provenance": provenance,
        "started_utc": started_utc,
        "completed_utc": utc_now(),
        "config": config,
        "motor_only": True,
        "physical_contract": physical_contract(env.model),
        "utility_and_intention_weights_unchanged": True,
        "graph_sha256": parent["graph_sha256"],
        "parent_checkpoint_sha256": sha256(args.resume),
        "checkpoint_sha256": sha256(args.output / "actor.pt"),
        "model_sha256": sha256(args.output / "model.mjb"),
        "setup_seconds": setup,
        "training_seconds": elapsed,
        "collection_forward_seconds": collection_seconds,
        "backward_optimization_seconds": optimization_seconds,
        "updates": updates,
        "transitions": transitions,
        "aggregate_simulated_seconds": transitions * env.control_dt,
        "parallel_physics_worlds": args.worlds,
        "worlds_by_task": dict(zip(TASKS, np.bincount(tasks.task_ids, minlength=3).tolist())),
        "physics_backend": "native CPU MuJoCo/mjbatch",
        "brain_device": str(device),
        "physics_hz": 5000,
        "control_hz": 500,
        "neurons": actor.core.neurons,
        "connections": actor.core.connections,
        "actor_parameters": sum(p.numel() for p in actor.parameters()),
        "trainable_parameters": sum(p.numel() for p in actor.parameters() if p.requires_grad),
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device)
        if device.type == "cuda"
        else 0,
        "core_gradient_audit": gradient_audit,
        "parameter_subset": subset_report,
        "subset_gradient_audit": subset_gradient_audit,
        "core_parameter_changes": {
            k: float((v.detach() - initial_core[k]).norm())
            for k, v in actor.core.named_parameters()
        },
        "completed_episodes": episodes,
        "teacher_present_during_collection": True,
        "ground_retention": retention,
        "command_curriculum": checkpoint["command_curriculum"],
        "hover_feedback_supervision": checkpoint["hover_feedback_supervision"],
        "teacher_mix_by_task": dict(zip(TASKS, mixture_by_task.tolist())),
        "startup_supervision": {
            "hover_seconds": start_duration,
            "weight": start_weight,
            "weighted_presentations": startup_presentations,
            "extra_physics_transitions": 0,
            "runtime_module": False,
        },
        "loss": "normalized weighted mean of active task motor MSE; hover adds 2x wing MSE; ground wing and task weights explicit; no utility loss",
        "training_tasks": [TASKS[i] for i in active_tasks],
        "ground_wing_loss_weight": getattr(args, "ground_wing_loss", 0.0),
        "ground_posture": posture.report() if teacher.posture is not None else None,
        "wing_response_supervision": {
            "weight": response_weight,
            "ground_worlds_sampled_per_action": response_worlds if response_weight else 0,
            "angle_perturbation_rad": 0.05,
            "speed_perturbation_rad_s": 2.0,
            "kind": "paired synthetic wing feedback from copied actual prior neural state",
            "normalization": "finite difference divided by the requested unsaturated response",
            "extra_physics_worlds": 0,
            "runtime_module": False,
        },
        "optimizer": "fresh Adam for declared motor-only parameter subset",
        "physical_success": "Training assistance is not student acceptance; run independent review",
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {k: v for k, v in report.items() if k not in ("provenance", "completed_episodes")}
        ),
        flush=True,
    )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("reference", "train", "evaluate"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--preset", choices=("wing_motion", "wing_position"), default="wing_motion"
    )
    parser.add_argument("--teacher", type=Path)
    parser.add_argument("--wing-response", choices=("filtered", "instant"))
    parser.add_argument(
        "--hover-reference", choices=("clock", "state", "state-position"), default="clock"
    )
    parser.add_argument(
        "--walking-reference", choices=("receding", "anchored"), default="receding"
    )
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--graph", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seconds", type=float, default=2)
    parser.add_argument("--seed", type=int, default=71001)
    parser.add_argument("--worlds", type=int, default=32)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--sequence", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument(
        "--trainable-subset",
        choices=("all", "wing-output", "wing-residual", "wing-feedback"),
        default="all",
    )
    parser.add_argument("--teacher-mix", type=float, default=0.8)
    parser.add_argument("--hover-teacher-mix", type=float)
    parser.add_argument("--walk-teacher-mix", type=float)
    parser.add_argument("--hover-start-seconds", type=float, default=0.05)
    parser.add_argument("--hover-start-weight", type=float, default=1)
    parser.add_argument("--retain-ground", action="store_true")
    parser.add_argument(
        "--stand-initial-form",
        action="store_true",
        help="Training labels hold the full initial stand pose; retain parent only for walking",
    )
    parser.add_argument("--ground-retention-weight", type=float, default=1.0)
    parser.add_argument("--nonwing-retention-weight", type=float, default=0.0)
    parser.add_argument("--feedback-lr", type=float, default=0.003)
    parser.add_argument("--ground-wing-loss", type=float, default=0.0)
    parser.add_argument("--wing-response-loss", type=float, default=0.0)
    parser.add_argument("--wing-response-worlds", type=int, default=8)
    parser.add_argument("--hover-feedback-loss", type=float, default=0.0)
    parser.add_argument("--ground-switch-seconds", type=float, default=0.0)
    parser.add_argument("--transition-worlds", type=int, default=12)
    parser.add_argument(
        "--ground-posture",
        action="store_true",
        help="Teach initial-form standing and restoring wing torques on both ground tasks",
    )
    parser.add_argument("--episode-seconds", type=float, default=2)
    parser.add_argument(
        "--task-set",
        choices=("all", "ground"),
        default="all",
        help="Training curriculum only; review always evaluates all three commands",
    )
    parser.add_argument(
        "--wing-angle-perturbation",
        type=float,
        default=0.0,
        help="Ground training reset disturbance in radians; clipped to joint limits",
    )
    parser.add_argument(
        "--wing-speed-perturbation",
        type=float,
        default=0.0,
        help="Ground training reset disturbance in radians/second",
    )
    parser.add_argument("--neural-view", action="store_true")
    args = parser.parse_args()
    if args.mode in ("train", "reference") and args.teacher is None:
        parser.error("Reference and training require --teacher")
    if args.mode in ("train", "evaluate") and (args.resume is None or args.graph is None):
        parser.error("Student requires --resume and --graph")
    result = train(args) if args.mode == "train" else review(args)
    if args.mode != "train" and not all(r["success"] for r in result["results"]):
        raise SystemExit(2)
