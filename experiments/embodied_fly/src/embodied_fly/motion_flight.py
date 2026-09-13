"""Reference and student captures for flight dynamics driven by wing motion.

The reference is a training-only wing trajectory controller. The student executes
all 78 graph-actor outputs, with no runtime teacher or phase input. Neither path
supplies body forces: both use the same measured-wing force model in FlyEnvironment.
"""

import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np

from embodied_fly.body import FlyEnvironment
from embodied_fly.provenance import evidence, sha256, utc_now
from embodied_fly.wing_motion import CONFIG


def initialize(env, height=2.0, heading=0.0):
    env.reset(yaw=heading)
    env.data.qpos[2] = height
    env.data.qpos[7:] = env.model.qpos_spring[7:]
    env.data.qpos[env.wing_angle_indices] = (0, 0.7, -1, 0, 0.7, -1)
    raw = np.zeros(env.model.nu)
    for i in range(env.model.nu):
        if env.model.actuator_trntype[i] == mujoco.mjtTrn.mjTRN_JOINT:
            jid = env.model.actuator_trnid[i, 0]
            if "wing_" not in env.model.joint(jid).name:
                raw[i] = env.data.qpos[env.model.jnt_qposadr[jid]]
    raw = np.clip(raw, env.low, env.high)
    env.data.ctrl[:] = raw
    filtered = env.model.actuator_actadr >= 0
    env.data.act[env.model.actuator_actadr[filtered]] = raw[filtered]
    mujoco.mj_forward(env.model, env.data)
    return (2 * (raw - env.low) / (env.high - env.low) - 1).astype(np.float32)


class WingReference:
    """Training-only wing commands. Privileged height and phase are declared."""

    def __init__(self, env, base_action, *, height=2, speed=0, frequency=12, phase=0):
        self.env, self.base = env, base_action.copy()
        self.height, self.speed, self.frequency, self.phase = height, speed, frequency, phase
        self.channels = np.array(
            [env.model.actuator(env.model.joint(j).name).id for j in env.wing_joint_ids]
        )

    def act(self):
        e, c = self.env, CONFIG
        omega = 2 * np.pi * self.frequency
        phase = omega * e.data.time + self.phase
        # Mean |sweep velocity| = 4 * frequency * amplitude for a sine stroke.
        hover_amplitude = c.reference_sweep_speed / (
            c.lift_weight_multiplier * 4 * self.frequency
        )
        velocity = e.anatomical_velocity()
        amplitude = np.clip(
            hover_amplitude + 0.8 * (self.height - e.data.qpos[2]) - 0.045 * velocity[5],
            0.12,
            1.15,
        )
        stroke = 0.7 + np.clip(0.1 * (self.speed - velocity[3]), -0.6, 0.6)
        desired = np.tile([amplitude * np.sin(phase), stroke, -1.0], 2)
        speed = np.tile([amplitude * omega * np.cos(phase), 0, 0], 2)
        acceleration = np.tile([-amplitude * omega**2 * np.sin(phase), 0, 0], 2)
        q = e.data.qpos[e.wing_angle_indices]
        v = e.data.qvel[e.wing_velocity_indices]
        spring = e.model.qpos_spring[e.wing_angle_indices]
        torque = (
            c.angular_armature * acceleration
            + c.joint_damping * speed
            + c.joint_stiffness * (desired - spring)
            + 0.02 * (desired - q)
            + 0.00015 * (speed - v)
        )
        action = self.base.copy()
        action[self.channels] = np.clip(torque / c.joint_torque_limit, -1, 1)
        return action


def run(args):
    if args.seconds <= 0 or not np.isfinite([args.seconds, args.height, args.speed]).all():
        raise ValueError("Finite positive duration and finite task settings required")
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    env = FlyEnvironment("wing_motion")
    base = initialize(env, args.height, args.heading)
    env.command[:] = (args.speed, 0, 0)
    oracle = WingReference(env, base, height=args.height, speed=args.speed, phase=args.phase)
    actor = checkpoint = memory = projection = None
    if args.controller == "student":
        import torch

        from embodied_fly.evaluate import load_actor

        torch.set_num_threads(4)
        actor, checkpoint = load_actor(args.checkpoint, args.graph, torch.device(args.device))
        memory = actor.initial_state(1)
        if args.neural_view:
            from embodied_fly.neural_view import NeuralProjection

            projection = NeuralProjection.from_graph(args.graph, torch.device(args.device))
    mujoco.mj_saveModel(env.model, str(args.output / "model.mjb"))
    setup = time.perf_counter() - started
    rows = {
        k: []
        for k in (
            "qpos",
            "qvel",
            "activation",
            "ctrl",
            "time",
            "action",
            "observation",
            "utility",
            "activity",
            "wing_activity",
            "wing_wrench",
        )
    }
    maps, heights, uprights, errors = [], [], [], []
    started = time.perf_counter()
    failure = None
    for step in range(round(args.seconds / env.control_dt)):
        observation = env.observation()
        if actor is not None:
            import torch

            with torch.no_grad():
                result = actor(
                    torch.as_tensor(
                        env.observation(True, wing_angles=True)[None], device=args.device
                    ),
                    memory,
                )
            memory = result.state
            action = result.action[0].cpu().numpy()
            utility = result.utility_scores[0].cpu().numpy()
            if projection is not None and step % 10 == 0:
                maps.append(projection.project(memory)[0].astype(np.float16))
        else:
            action = oracle.act() if args.controller == "reference" else base.copy()
            utility = np.array([0, 1, 0, 0, 0, 0], np.float32)
        for k, value in (
            ("qpos", env.data.qpos),
            ("qvel", env.data.qvel),
            ("activation", env.data.act),
            ("ctrl", env.data.ctrl),
            ("time", env.data.time),
            ("action", action),
            ("observation", observation),
            ("utility", utility),
            ("activity", np.argmax(utility)),
            ("wing_activity", env.wing_forces.activity[0]),
            ("wing_wrench", env.wing_forces.wrench[0]),
        ):
            rows[k].append(np.array(value, copy=True))
        try:
            env.step(action)
        except RuntimeError as error:
            failure = str(error)
            break
        heights.append(float(env.data.qpos[2] * 0.01))
        uprights.append(float(env.data.xmat[env.thorax_id, 8]))
        target = np.array(
            [
                args.speed * env.data.time * np.cos(args.heading),
                args.speed * env.data.time * np.sin(args.heading),
                args.height,
            ]
        )
        errors.append(float(np.linalg.norm(env.data.qpos[:3] - target) * 0.01))
    if projection is not None:
        rows.update(
            neural_map=maps, neural_occupancy=projection.occupancy, neural_map_stride=10
        )
    np.savez_compressed(args.output / "flight.npz", **rows)
    elapsed = time.perf_counter() - started
    rmse = float(np.sqrt(np.mean(np.square(errors)))) if errors else None
    stable = bool(heights) and min(heights) > 0.005 and min(uprights) > 0.5
    ratio = env.maximum_disallowed_ground_force / env.wing_forces.weight
    success = stable and rmse < 0.005 and ratio < 0.1 and failure is None
    report = {
        "provenance": evidence(),
        "completed_utc": utc_now(),
        "environment": env.report(),
        "controller": args.controller,
        "student_present": actor is not None,
        "teacher_present": args.controller == "reference",
        "scripted_gait_present": False,
        "policy_acceptance_eligible": actor is not None,
        "reference_description": "training-only sinusoidal wing joint trajectory with height/speed feedback"
        if args.controller == "reference"
        else None,
        "checkpoint_sha256": sha256(args.checkpoint) if actor is not None else None,
        "graph_sha256": checkpoint["graph_sha256"] if checkpoint else None,
        "physical_preset": "wing_motion",
        "control_hz": 500,
        "physics_hz": 5000,
        "setup_seconds": setup,
        "stepping_capture_and_state_write_wall_seconds": elapsed,
        "requested_seconds": args.seconds,
        "simulated_seconds": env.data.time,
        "initial_height_cm": args.height,
        "speed_cm_s": args.speed,
        "heading_rad": args.heading,
        "reference_phase_rad": args.phase if args.controller == "reference" else None,
        "root_tracking_rmse_m": rmse,
        "minimum_root_height_m": min(heights) if heights else None,
        "minimum_upright": min(uprights) if uprights else None,
        "max_forbidden_ground_force_over_weight": ratio,
        "warning_count": int(env.data.warning.number.sum()),
        "numerical_failure": failure,
        "success": success,
        "scope": "airborne development trial; no takeoff or landing claim",
        "results": [
            {
                "case": "flight",
                "stable": stable,
                "success": success,
                "gate_label": "Wing-motion flight gate",
            }
        ],
        "gates": {
            "minimum_height_m": 0.005,
            "minimum_upright": 0.5,
            "maximum_root_rmse_m": 0.005,
            "maximum_prohibited_support_over_weight": 0.1,
        },
        "model_sha256": sha256(args.output / "model.mjb"),
        "state_sha256": sha256(args.output / "flight.npz"),
        "force_state": "wing_activity is causal filter state; wing_wrench is the previous applied contribution",
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "environment"}), flush=True)
    return report


def collect(args):
    """Predetermined reference episodes; preserve all outcomes, including failures."""
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    rng = np.random.default_rng(args.seed)
    episodes = []
    first = None
    for episode in range(args.episodes):
        options = vars(args) | {
            "controller": "reference",
            "episodes": 0,
            "output": args.output / f"reference_{episode:03d}",
            "phase": float(rng.uniform(0, 2 * np.pi))
            if args.cold_start_phase is None
            else args.cold_start_phase,
            "heading": float(rng.uniform(-0.2, 0.2)),
            "height": float(rng.uniform(1.8, 2.2)),
            "speed": (0, 0, 1, 2)[episode % 4],
        }
        report = run(SimpleNamespace(**options))
        source = options["output"]
        target = args.output / f"episode_{episode:03d}.npz"
        (source / "flight.npz").replace(target)
        if first is None:
            first = report
            (source / "model.mjb").replace(args.output / "model.mjb")
        else:
            if report["model_sha256"] != first["model_sha256"]:
                raise RuntimeError("Corpus model changed between episodes")
            (source / "model.mjb").unlink()  # identical generated duplicate only
        episodes.append(
            {
                "episode": episode,
                "failure": report["numerical_failure"],
                "physical_failure": not report["success"],
                "final_upright": report["minimum_upright"],
                "state_sha256": sha256(target),
                "frames": round(args.seconds * 500),
                "report": str(source.relative_to(args.output) / "report.json"),
                "phase": options["phase"],
                "speed_cm_s": options["speed"],
            }
        )
    manifest = {
        "provenance": evidence(),
        "completed_utc": utc_now(),
        "seed": args.seed,
        "environment": first["environment"],
        "model_sha256": first["model_sha256"],
        "control_hz": 500,
        "physical_preset": "wing_motion",
        "role": "flight",
        "episodes": episodes,
        "collection_wall_seconds": time.perf_counter() - started,
        "physical_transitions": sum(e["frames"] for e in episodes),
        "aggregate_simulated_seconds": args.seconds * args.episodes,
        "controller": "training-only wing reference; not a learned brain",
        "cold_start_phase": args.cold_start_phase,
        "startup_scope": "All episodes initialize stationary wings. A fixed reference phase gives consistent cold-start action labels; phase is never a student input.",
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        json.dumps(
            {k: v for k, v in manifest.items() if k not in ("environment", "provenance")}
        ),
        flush=True,
    )
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--controller", choices=("reference", "student", "inactive"), required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--graph", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seconds", type=float, default=1)
    parser.add_argument("--height", type=float, default=2)
    parser.add_argument("--speed", type=float, default=0)
    parser.add_argument("--heading", type=float, default=0)
    parser.add_argument("--phase", type=float, default=0)
    parser.add_argument("--neural-view", action="store_true")
    parser.add_argument("--episodes", type=int, default=0)
    parser.add_argument("--seed", type=int, default=61001)
    parser.add_argument("--cold-start-phase", type=float)
    config = parser.parse_args()
    if config.controller == "student" and (config.checkpoint is None or config.graph is None):
        parser.error("Student requires --checkpoint and --graph")
    if config.episodes:
        if config.episodes < 4 or config.controller != "reference":
            parser.error("Collection requires at least four reference episodes")
        result = collect(config)
        raise SystemExit(
            0 if all(not e["physical_failure"] for e in result["episodes"]) else 2
        )
    raise SystemExit(0 if run(config)["success"] else 2)
