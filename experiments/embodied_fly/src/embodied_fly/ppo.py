"""Recurrent physical-outcome PPO with explicit, measured imitation rehearsal.

The deployed graph actor is unchanged. A separate critic and motor exploration
distribution exist only during training. Physics is native MuJoCo on CPU threads.
"""

import argparse
import json
import math
import time
from collections import deque
from pathlib import Path

import mujoco
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from embodied_fly.batch import FlyBatch
from embodied_fly.body import CONTROL_DT
from embodied_fly.evaluate import load_actor
from embodied_fly.flight_outcome import FlightOutcomeReward, FlightResets
from embodied_fly.physical_contract import physical_contract
from embodied_fly.ppo_critic import StandardizedValueNetwork, fit_critic, value_quality
from embodied_fly.ppo_timing import physical_timescales, recurrent_forward
from embodied_fly.provenance import evidence, sha256, utc_now
from embodied_fly.train import load_episodes, sample, synchronize


class Critic(nn.Module):
    """Training-only value of causal observation plus preceding descending state."""

    def __init__(self, brain, standardize_inputs=False):
        super().__init__()
        network_type = StandardizedValueNetwork if standardize_inputs else nn.Sequential
        self.network = network_type(
            nn.Linear(brain.observation_size + len(brain.descending_ids), 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
        )

    def features(self, brain, observation, state):
        obs = ((observation - brain.observation_mean) / brain.observation_std).clamp(-10, 10)
        return torch.cat([obs, state[brain.descending_ids].T.detach()], dim=-1).detach()

    def forward(self, brain, observation, state):
        return self.network(self.features(brain, observation, state)).squeeze(-1)


def motor_distribution(output, active, log_std, minimum_std=0.01):
    if not 0 < minimum_std <= 0.15:
        raise ValueError("Exploration floor must be positive and no greater than 0.15")
    location = torch.atanh(output.action[:, active].clamp(-0.9999, 0.9999))
    return torch.distributions.Normal(
        location, log_std.clamp(math.log(minimum_std), math.log(0.15)).exp()
    )


def reset_motor_exploration(log_std, optimizer, standard_deviation):
    """Explicit distribution migration; preserve actor/critic weights and Adam.

    Discard only the exploration parameter's moments so its old optimizer state
    cannot immediately undo the requested new scale.
    """
    if not 0 < standard_deviation <= 0.15:
        raise ValueError("Exploration standard deviation must be positive and <=.15")
    with torch.no_grad():
        log_std.fill_(math.log(standard_deviation))
    optimizer.state.pop(log_std, None)


def joint_log_probability(output, distribution, latent_action, activity, *, motor_only=False):
    # Stable tanh change-of-variables; sum over this stage's active channels.
    jacobian = 2 * (math.log(2) - latent_action - F.softplus(-2 * latent_action))
    motor = (distribution.log_prob(latent_action) - jacobian).sum(-1)
    if motor_only:
        return motor
    utility = torch.distributions.Categorical(logits=output.utility_logits).log_prob(activity)
    return motor + utility


def advantages(rewards, values, final_value, done, gamma, gae_lambda):
    """Timeout bootstrap is included in rewards; neither reset leaks next episode."""
    result = torch.zeros_like(rewards)
    carry = torch.zeros_like(final_value)
    following = final_value
    for t in reversed(range(len(rewards))):
        live = (~done[t]).float()
        delta = rewards[t] + gamma * following * live - values[t]
        carry = delta + gamma * gae_lambda * live * carry
        result[t] = carry
        following = values[t]
    return result, result + values


REWARD_RECIPE = {
    "velocity_tracking_rate": 2.0,
    "yaw_tracking_rate": 0.5,
    "upright_alive_rate": 0.5,
    "tilt_cost_rate": 0.5,
    "vertical_speed_cost_rate": 0.05,
    "forbidden_support_cost_rate": 2.0,
    "action_change_cost_rate": 0.01,
    "physical_failure_penalty": 1.0,
    "velocity_filter_seconds": 0.05,
    "moving_velocity_width_cm_s": 0.5,
    "holding_velocity_width_cm_s": 0.2,
    "yaw_width_rad_s": 0.75,
    "failure_upright_below": 0.5,
    "failure_height_below_cm": 0.06,
    "units": "rates multiplied by 0.002 s each action; failure penalty once",
}


class OutcomeReward:
    def __init__(self, env, stationary_cost=0.0, stationary_turn_cost=0.0):
        if min(stationary_cost, stationary_turn_cost) < 0:
            raise ValueError("Stationary costs must be nonnegative")
        self.env = env
        self.filtered_velocity = np.zeros((env.n, 6))
        self.stationary_cost = stationary_cost
        self.stationary_turn_cost = stationary_turn_cost
        self.recipe = REWARD_RECIPE | {
            "stationary_translation_cost_rate": stationary_cost,
            "stationary_rotation_cost_rate": stationary_turn_cost,
            "stationary_speed_scale_cm_s": 0.25,
            "stationary_yaw_scale_rad_s": 0.5,
            "stationary_cost_formula": "zero command only: -weight * magnitude / (scale + magnitude)",
        }

    def reset(self, ids):
        self.filtered_velocity[ids] = 0

    def __call__(self, previous_action):
        env = self.env
        alpha = -np.expm1(-CONTROL_DT / REWARD_RECIPE["velocity_filter_seconds"])
        velocity = env.velocity()
        self.filtered_velocity += alpha * (velocity - self.filtered_velocity)
        upright = env.fields["xmat"][:, env.template.thorax_id, 8]
        failed = (upright < 0.5) | (env.fields["qpos"][:, 2] < 0.06)
        width = np.where(np.linalg.norm(env.command[:, :2], axis=1) < 0.01, 0.2, 0.5)
        velocity_error = self.filtered_velocity[:, 3:5] - env.command[:, :2]
        terms = {
            "velocity": 2.0 * np.exp(-np.sum(velocity_error**2, axis=1) / width**2),
            "yaw": 0.5
            * np.exp(-(((self.filtered_velocity[:, 2] - env.command[:, 2]) / 0.75) ** 2)),
            "alive": 0.5 * (~failed),
            "tilt": -0.5 * (1 - upright) ** 2,
            "vertical": -0.05 * np.minimum((velocity[:, 5] / 0.5) ** 2, 10),
            "support": -2.0 * np.minimum(env.forbidden_peak / env.body_weight, 5),
            "action_change": -0.01
            * np.mean((env.previous_action - previous_action) ** 2, axis=1),
        }
        # Bounded rational costs keep a useful distinction between drifting holds
        # after the original narrow tracking exponential has almost saturated.
        # A nonzero translational OR rotational command leaves these costs off.
        stationary = np.linalg.norm(env.command, axis=1) < 1e-6
        speed = np.linalg.norm(self.filtered_velocity[:, 3:5], axis=1)
        turn = np.abs(self.filtered_velocity[:, 2])
        terms["stationary_translation"] = (
            -self.stationary_cost * stationary * speed / (0.25 + speed)
        )
        terms["stationary_rotation"] = (
            -self.stationary_turn_cost * stationary * turn / (0.5 + turn)
        )
        reward = sum(terms.values()) * CONTROL_DT - failed.astype(float)
        return reward.astype(np.float32), failed, terms


def core_gradient_stats(brain, loss):
    """Audit one objective before mixing gradients; do not mislabel supervision as RL."""
    named = list(brain.core.named_parameters())
    gradients = torch.autograd.grad(loss, [p for _, p in named], retain_graph=True)
    return {
        name: {
            "l2": float(g.norm()),
            "finite": bool(torch.isfinite(g).all()),
            "nonzero_cells": int((g != 0).sum()),
        }
        for (name, _), g in zip(named, gradients)
    }


def train(args):
    if args.horizon % args.sequence or min(args.worlds, args.horizon, args.sequence) < 1:
        raise ValueError(
            "Positive rollout dimensions and horizon divisible by sequence required"
        )
    motor_ground = getattr(args, "motor_ground", False)
    motor_all = getattr(args, "motor_all", False)
    motor_mode = motor_ground or motor_all
    hover_physical = getattr(args, "hover_physical", False)
    hover_only = getattr(args, "hover_only", False)
    round_trip = getattr(args, "round_trip", False)
    bounded_hover = getattr(args, "bounded_hover_reward", False)
    reset_critic = getattr(args, "reset_critic", False)
    reset_exploration = getattr(args, "reset_exploration", False)
    hover_speed_scale = getattr(args, "hover_vertical_speed_scale", 5.0)
    if not np.isfinite(hover_speed_scale) or hover_speed_scale <= 0:
        raise ValueError("Hover vertical-speed scale must be finite and positive")
    if hover_speed_scale != 5.0 and not bounded_hover:
        raise ValueError("Custom hover speed scale requires bounded hover reward")
    if bounded_hover and not hover_only:
        raise ValueError("Bounded hover reward requires hover-only training")
    if round_trip and (
        not (hover_only and bounded_hover and motor_all)
        or args.episode_seconds != 12
        or args.worlds < 12
        or args.worlds % 2
    ):
        raise ValueError(
            "Round trips require bounded hover motor training, 12 s episodes and even worlds >=12"
        )
    if hover_only and (not hover_physical or args.preset != "wing_position"):
        raise ValueError("Hover-only requires the physical hover motor curriculum")
    critic_lr = getattr(args, "critic_lr", 3e-4)
    if not np.isfinite(critic_lr) or critic_lr <= 0:
        raise ValueError("Critic learning rate must be finite and positive")
    recompute = getattr(args, "checkpoint_activations", False)
    independent_critic = getattr(args, "independent_critic", False)
    critic_epochs = getattr(args, "critic_epochs", None)
    critic_epochs = args.epochs if critic_epochs is None else critic_epochs
    standardize_critic = getattr(args, "critic_standardize_inputs", False)
    shuffle_critic = getattr(args, "critic_shuffle_transitions", False)
    if critic_epochs < 1 or (
        (standardize_critic or shuffle_critic or critic_epochs != args.epochs)
        and not independent_critic
    ):
        raise ValueError("Separate critic epochs/input calibration require independent critic")
    if independent_critic and args.epochs < 1:
        raise ValueError("Independent critic requires at least one training epoch")
    if hover_physical and (
        not motor_all
        or getattr(args, "motor_retention_weight", 0)
        or args.rehearsal is not None
        or getattr(args, "wing_supervision", 0)
    ):
        raise ValueError("Hover physical PPO requires all motors and no imitation losses")
    critic_warmup = getattr(args, "critic_warmup_rollouts", 0)
    if critic_warmup < 0:
        raise ValueError("Critic warmup rollouts must be nonnegative")
    if critic_warmup and not motor_all:
        raise ValueError("Critic warmup currently requires the all-motor curriculum")
    minimum_noise = getattr(args, "minimum_noise", 0.01)
    if not 0 < minimum_noise <= args.noise <= 0.15:
        raise ValueError("Initial exploration must lie between its positive floor and 0.15")
    retention_weight = getattr(args, "motor_retention_weight", 0.0)
    if motor_ground and motor_all:
        raise ValueError("Choose one motor curriculum")
    if motor_all and (
        args.preset != "wing_position" or args.rehearsal is not None or args.worlds < 3
    ):
        raise ValueError("All-motor PPO requires wing_position and no legacy corpus")
    if (
        not np.isfinite(retention_weight)
        or retention_weight < 0
        or (retention_weight and not motor_all)
    ):
        raise ValueError("Finite nonnegative retention requires all-motor PPO")
    wing_weight = getattr(args, "wing_supervision", 0.0)
    if not np.isfinite(wing_weight) or wing_weight < 0 or (wing_weight and not motor_ground):
        raise ValueError("Wing supervision requires a finite nonnegative motor-ground weight")
    if motor_ground and (args.preset != "wing_motion" or args.rehearsal is not None):
        raise ValueError(
            "Motor ground PPO requires wing_motion and no legacy rehearsal corpus"
        )
    if not motor_mode and args.rehearsal is None:
        raise ValueError("Utility PPO requires its declared rehearsal corpus")
    if (
        not motor_mode
        and args.preset in ("flight", "wing_motion")
        and args.flight_resets is None
    ):
        raise ValueError("Flight PPO requires declared training-split airborne resets")
    args.output.mkdir(parents=True, exist_ok=False)
    started_setup = time.perf_counter()
    run_evidence = evidence()
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    # Critic shuffling must not consume the actor/physical sampling RNG streams.
    critic_rng = np.random.default_rng(args.seed ^ 0xC8171C)
    device = torch.device(args.device)
    brain, parent = load_actor(args.resume, args.graph, device)
    if (
        hover_only
        and (
            parent.get("config", {}).get("bounded_hover_reward", False) != bounded_hover
            or parent.get("config", {}).get("hover_vertical_speed_scale", 5.0)
            != hover_speed_scale
        )
        and not reset_critic
    ):
        raise ValueError("Changed hover reward requires an explicit fresh critic")
    brain.train()
    if (
        parent.get("config", {}).get("critic_standardize_inputs", False) != standardize_critic
        and "critic_state_dict" in parent
        and not reset_critic
    ):
        raise ValueError("Changed critic input calibration requires --reset-critic")
    critic = Critic(brain, standardize_critic).to(device)
    if brain.motor_only != motor_mode:
        raise ValueError(
            "Motor-only actors require a motor curriculum; utility actors require utility PPO"
        )
    wing_response = parent.get("physical_contract", {}).get("wing_response", "filtered")
    env = FlyBatch(
        args.worlds,
        args.threads,
        brain.sensor_extension_size,
        preset=args.preset,
        wing_response=wing_response,
        physics_hz=parent.get("physical_contract", {}).get("physics_hz"),
    )
    contract = physical_contract(env.model) if motor_mode else None
    if motor_mode and parent.get("physical_contract") != contract:
        raise ValueError("Motor PPO must match the parent physical fly")
    if hover_only and (
        contract.get("wing_response") != "instant"
        or contract["physics_hz"] != 1000
        or brain.sensor_extension_size != 16
    ):
        raise ValueError(
            "Hover-only requires explicit 1 kHz plant and horizontal-feedback migration"
        )
    frozen = (
        {
            k: v.detach().clone()
            for k, v in brain.state_dict().items()
            if k.startswith(("utility_head.", "intention_encoder."))
        }
        if motor_mode
        else {}
    )
    tasks = None
    if motor_mode:
        from embodied_fly.motor_focus import MotorTasks
        from embodied_fly.motor_outcome import AllMotorReward, GroundMotorReward

        tasks = MotorTasks(
            env,
            args.seed,
            "all" if motor_all else "ground",
            getattr(args, "wing_angle_perturbation", 0.0),
            getattr(args, "wing_speed_perturbation", 0.0),
        )
        if hover_physical:
            from embodied_fly.hover_ppo import HoverPPOTasks

            tasks = HoverPPOTasks(env, args.seed)
            if hover_only:
                from embodied_fly.hover_only import HoverOnlyTasks

                tasks = HoverOnlyTasks(env, args.seed)
                if round_trip:
                    from embodied_fly.round_trip_tasks import RoundTripTasks

                    tasks = RoundTripTasks(env, args.seed)
                if bounded_hover:
                    tasks.promotion_rate = 5.5
    control_dt = env.control_dt
    time_scale = control_dt / CONTROL_DT
    # CLI discount factors retain their original 2 ms physical horizons.
    gamma, gae_lambda = args.gamma**time_scale, args.gae_lambda**time_scale
    timing = physical_timescales(control_dt, gamma, gae_lambda, args.horizon, args.sequence)
    mujoco.mj_saveModel(env.model, str(args.output / "model.mjb"))
    trace = deque(maxlen=128)
    flight_resets = (
        FlightResets(env, args.flight_resets, rng)
        if not motor_mode and args.preset in ("flight", "wing_motion")
        else None
    )
    reward_fn = (
        FlightOutcomeReward(env)
        if flight_resets
        else OutcomeReward(env, args.stationary_cost, args.stationary_turn_cost)
    )
    if tasks is not None:
        reward_fn = AllMotorReward(tasks) if motor_all else GroundMotorReward(tasks)
        if hover_physical:
            from embodied_fly.hover_ppo import HoverPPOReward

            reward_fn.flight = HoverPPOReward(env)
            reward_fn.recipe["version"] = "stand_walk_hover_physical_v2"
            reward_fn.recipe["hover"] = reward_fn.flight.recipe
            if hover_only:
                from embodied_fly.hover_only import HoverOnlyReward

                reward_fn = HoverOnlyReward(env)
                if bounded_hover:
                    from embodied_fly.hover_only import HoverBalancedReward

                    reward_fn = HoverBalancedReward(env, hover_speed_scale)
    active = torch.as_tensor(
        np.ones(env.model.nu, bool)
        if flight_resets or motor_mode
        else ~env.template.walking_inactive,
        device=device,
    )
    log_std = nn.Parameter(
        torch.full((int(active.sum()),), math.log(args.noise), device=device)
    )
    parameters = [p for p in brain.parameters() if p.requires_grad] + [log_std]
    optimizer = torch.optim.Adam(parameters, lr=args.lr, eps=1e-5)
    value_optimizer = torch.optim.Adam(critic.parameters(), lr=critic_lr, eps=1e-5)
    optimizer_resumed = (
        parent.get("method", "").startswith("recurrent physical-outcome PPO")
        and parent.get("config", {}).get("preset", "walking") == args.preset
        and parent.get("motor_only", False) == motor_mode
    )
    if optimizer_resumed:
        # Same actor/critic/exploration ordering as the preceding PPO stage.
        # Imitation checkpoints use another optimizer layout and cannot resume it.
        if not reset_critic:
            critic.load_state_dict(parent["critic_state_dict"], strict=True)
        optimizer.load_state_dict(parent["optimizer_state_dict"])
        if not reset_critic:
            value_optimizer.load_state_dict(parent["value_optimizer_state_dict"])
        with torch.no_grad():
            log_std.copy_(parent["log_std"].to(device))
        for group in optimizer.param_groups:
            group["lr"] = args.lr
        for group in value_optimizer.param_groups:
            group["lr"] = critic_lr
    if reset_exploration:
        reset_motor_exploration(log_std, optimizer, args.noise)
    initial_exploration = (
        log_std.detach().clamp(math.log(minimum_noise), math.log(0.15)).exp().cpu().tolist()
    )
    from embodied_fly.motor_retention import FrozenMotorReference

    retainer = FrozenMotorReference(brain, args.worlds) if retention_weight else None
    retained_state = (
        {k: v.detach().cpu().clone() for k, v in retainer.actor.state_dict().items()}
        if retainer is not None
        else None
    )
    core_initial = {n: p.detach().clone() for n, p in brain.core.named_parameters()}
    rehearsal_hz = None
    validation_ids = set()
    if args.rehearsal is not None:
        # Preserve the same whole-episode held-out split as the imitation experiments.
        episodes = load_episodes(
            args.rehearsal,
            brain.sensor_extension_size >= 6,
            brain.sensor_extension_size >= 12,
            brain.sensor_extension_size == 14,
        )
        rehearsal_manifest = json.loads((args.rehearsal / "manifest.json").read_text())
        rehearsal_hz = rehearsal_manifest.get(
            "control_hz", rehearsal_manifest.get("environment", {}).get("control_hz", 500)
        )
        rehearsal_time_scale = 1 / rehearsal_hz / CONTROL_DT
        validation_ids = set(
            np.random.default_rng(1193).permutation(len(episodes))[
                : max(1, len(episodes) // 4)
            ]
        )
        rehearsal = [episode for i, episode in enumerate(episodes) if i not in validation_ids]
    cases = np.array(
        [
            [0, 0, 0],
            [0, 0, 0],
            [0.5, 0, 0],
            [1, 0, 0],
            [2, 0, 0],
            [1.5, 0, 0.75],
            [1.5, 0, -0.75],
        ],
        np.float32,
    )
    task_ids = (
        tasks.task_ids
        if tasks is not None
        else flight_resets.task_ids
        if flight_resets
        else np.arange(args.worlds) % len(cases)
    )

    def reset_worlds(ids):
        if tasks is not None:
            tasks.reset(ids)
        elif flight_resets:
            flight_resets.reset(ids)
        else:
            env.reset(ids, yaw=rng.uniform(-0.2, 0.2, len(ids)))

    if not flight_resets and tasks is None:
        env.command[:] = cases[task_ids]
    reset_worlds(np.arange(args.worlds))
    reward_fn.reset(np.arange(args.worlds))
    episode_return = np.zeros(args.worlds)
    episode_start = env.fields["qpos"][:, :2].copy()
    episode_records = []
    memory = brain.initial_state(args.worlds)
    observation = torch.as_tensor(env.observation(), device=device)
    synchronize(device)
    setup_seconds = time.perf_counter() - started_setup
    counters = {
        "rollouts": 0,
        "ppo_updates": 0,
        "critic_updates": 0,
        "critic_sample_presentations": 0,
        "critic_updates_after_actor_stop": 0,
        "actor_kl_stops": 0,
        "independent_critic_seconds": 0.0,
        "critic_warmup_rollouts": 0,
        "transitions": 0,
        "rehearsal_frames": 0,
        "wing_supervised_presentations": 0,
        "collection_seconds": 0.0,
        "optimization_seconds": 0.0,
        "ppo_optimization_seconds": 0.0,
        "rehearsal_seconds": 0.0,
    }
    training_started_utc = utc_now()
    start = time.perf_counter()
    gradient_audit = wing_gradient_audit = retention_gradient_audit = None
    failure = None
    progress = []
    task_transitions = {name: 0 for name in ("stand", "walk", "hover")}
    try:
        while time.perf_counter() - start < args.seconds:
            if hover_only:
                tasks.advance_curriculum()
            elif hover_physical:
                tasks.widening = min(
                    1.0, counters["transitions"] * control_dt / args.worlds / 10.0
                )
            warming_critic = counters["rollouts"] < critic_warmup
            collect_start = time.perf_counter()
            obs_buf, latent_buf, activity_buf, logp_buf, values, rewards, dones = (
                [],
                [],
                [],
                [],
                [],
                [],
                [],
            )
            wing_target_buf = []
            retention_buf = []
            state_starts = []
            critic_features = []
            physical_rewards = []
            term_sums = {}
            with torch.no_grad():
                for t in range(args.horizon):
                    if t % args.sequence == 0:
                        state_starts.append(memory.clone())
                    if independent_critic:
                        features = critic.features(brain, observation, memory)
                        critic_features.append(features)
                        value = critic.network(features).squeeze(-1)
                    else:
                        value = critic(brain, observation, memory)
                    output = brain(
                        observation, memory, sample_activity=True, time_scale=time_scale
                    )
                    distribution = motor_distribution(output, active, log_std, minimum_noise)
                    latent = distribution.sample()
                    action = output.action.clone()
                    action[:, active] = latent.tanh()
                    logp = joint_log_probability(
                        output, distribution, latent, output.activity, motor_only=motor_mode
                    )
                    if wing_weight:
                        wing_target = reward_fn.posture.wing_targets()
                        wing_target_buf.append(torch.as_tensor(wing_target, device=device))
                    if retainer is not None:
                        retention_buf.append(retainer.act(observation).detach())
                    previous = env.previous_action.copy()
                    if round_trip:
                        previous_target = np.column_stack(
                            (env.requested_xy_cm, env.requested_height_cm)
                        )
                    next_observation = env.step(action.cpu().numpy())
                    if round_trip:
                        tasks.after_step()
                        next_observation = env.observation()
                    trace.append(
                        {
                            "observation": observation.cpu().numpy().copy(),
                            "qpos": env.fields["qpos"].copy(),
                            "qvel": env.fields["qvel"].copy(),
                            "activation": env.fields["act"].copy(),
                            "ctrl": env.fields["ctrl"].copy(),
                            "action": env.previous_action.copy(),
                            "utility": output.utility_scores.cpu().numpy(),
                        }
                    )
                    if retainer is not None:
                        trace[-1]["ground_reference_action"] = (
                            retention_buf[-1].cpu().numpy().copy()
                        )
                    if wing_weight:
                        trace[-1]["wing_target"] = wing_target.copy()
                    if env.wing_forces is not None:
                        trace[-1]["wing_activity"] = env.wing_forces.activity.copy()
                        trace[-1]["wing_wrench"] = env._wing_applied.copy()
                    reward, failed, terms = reward_fn(previous)
                    if motor_mode:
                        for i, name in enumerate(task_transitions):
                            task_transitions[name] += int(np.sum(task_ids == i))
                    trace[-1]["physical_reward"] = reward.copy()
                    trace[-1]["failed"] = failed.copy()
                    trace[-1]["reward_rates"] = np.stack(list(terms.values()), axis=1)
                    trace[-1]["task_id"] = task_ids.copy()
                    trace[-1]["requested_height_cm"] = env.requested_height_cm.copy()
                    if round_trip:
                        trace[-1]["requested_xy_cm"] = env.requested_xy_cm.copy()
                        trace[-1]["pre_action_target_cm"] = previous_target
                        trace[-1]["observer_route_id"] = tasks.route_ids.copy()
                    physical_rewards.append(float(reward.mean()))
                    for key, term in terms.items():
                        term_sums[key] = (
                            term_sums.get(key, 0.0) + float(term.mean()) / args.horizon
                        )
                    episode_return += reward
                    timeout = env.ages >= round(args.episode_seconds / control_dt)
                    done = failed | timeout
                    next_tensor = torch.as_tensor(next_observation, device=device)
                    reward_tensor = torch.as_tensor(reward, device=device)
                    if np.any(timeout & ~failed):
                        final = critic(brain, next_tensor, output.state)
                        reward_tensor += (
                            gamma * final * torch.as_tensor(timeout & ~failed, device=device)
                        )
                    obs_buf.append(observation)
                    latent_buf.append(latent)
                    activity_buf.append(output.activity)
                    logp_buf.append(logp)
                    values.append(value)
                    rewards.append(reward_tensor)
                    dones.append(torch.as_tensor(done, device=device))
                    ids = np.flatnonzero(done)
                    for i in ids:
                        failure_trace = None
                        if failed[i]:
                            failure_path = (
                                args.output / f"failure_{len(episode_records):05d}.npz"
                            )
                            window = list(trace)[-min(int(env.ages[i]), len(trace)) :]
                            np.savez_compressed(
                                failure_path,
                                **{
                                    key: np.stack([frame[key][i] for frame in window])
                                    for key in window[0]
                                },
                            )
                            failure_trace = {
                                "file": failure_path.name,
                                "sha256": sha256(failure_path),
                                "frames": len(window),
                                "scope": f"last up to {128 * control_dt:g} s; post-action physical states",
                            }
                        episode_records.append(
                            {
                                "case_id": int(task_ids[i]),
                                "task": ("stand", "walk", "hover")[task_ids[i]]
                                if motor_mode
                                else None,
                                "failed": bool(failed[i]),
                                "failure_trace": failure_trace,
                                "simulated_seconds": float(env.ages[i] * control_dt),
                                "return": float(episode_return[i]),
                                "distance_cm": float(
                                    np.linalg.norm(
                                        env.fields["qpos"][i, :2] - episode_start[i]
                                    )
                                ),
                            }
                        )
                        if hover_only:
                            if round_trip:
                                episode_records[-1].update(tasks.episode_metrics(i))
                                episode_records[-1]["complete_target_sequence_tracking"] &= (
                                    bool(timeout[i] and not failed[i])
                                )
                            tasks.observe_episode(episode_records[-1])
                    reset_worlds(ids)
                    reward_fn.reset(ids)
                    episode_return[ids] = 0
                    episode_start[ids] = env.fields["qpos"][ids, :2]
                    if not flight_resets and tasks is None:
                        task_ids[ids] = rng.integers(len(cases), size=len(ids))
                        env.command[ids] = cases[task_ids[ids]]
                    memory = brain.reset_worlds(output.state, dones[-1])
                    if retainer is not None:
                        retainer.reset(dones[-1])
                    observation = torch.as_tensor(env.observation(), device=device)
                final_value = critic(brain, observation, memory)
            data = {
                key: torch.stack(buffer)
                for key, buffer in (
                    ("obs", obs_buf),
                    ("latent", latent_buf),
                    ("activity", activity_buf),
                    ("logp", logp_buf),
                    ("value", values),
                    ("reward", rewards),
                    ("done", dones),
                )
            }
            if wing_weight:
                data["wing_target"] = torch.stack(wing_target_buf)
            if retainer is not None:
                data["ground_reference_action"] = torch.stack(retention_buf)
            if independent_critic:
                data["critic_features"] = torch.stack(critic_features)
                critic_features.clear()
            adv, returns = advantages(
                data["reward"],
                data["value"],
                final_value,
                data["done"],
                gamma,
                gae_lambda,
            )
            critic_quality = {}
            if motor_mode:
                critic_quality = value_quality(
                    data["value"], returns, task_ids, task_transitions
                )
            adv = (adv - adv.mean()) / adv.std().clamp_min(1e-6)
            synchronize(device)
            counters["collection_seconds"] += time.perf_counter() - collect_start
            counters["transitions"] += args.horizon * args.worlds
            update_start = time.perf_counter()
            kl_values, policy_losses, value_losses, wing_losses = [], [], [], []
            retention_losses = []
            stop_epoch = False
            critic_fit = None
            critic_after = {}
            for _ in range(0 if independent_critic and warming_critic else args.epochs):
                for chunk in rng.permutation(len(state_starts)):
                    a, b = chunk * args.sequence, (chunk + 1) * args.sequence
                    state = state_starts[chunk].detach()
                    new_logps, predictions = [], []
                    entropy, wing_errors, retention_errors = [], [], []
                    for t in range(a, b):
                        if not independent_critic:
                            predictions.append(critic(brain, data["obs"][t], state))
                        # Critic warmup needs no actor graph. Later updates can
                        # recompute the identical actor to fit longer sequences.
                        with torch.set_grad_enabled(not warming_critic):
                            result = recurrent_forward(
                                brain,
                                data["obs"][t],
                                state,
                                data["activity"][t],
                                time_scale,
                                recompute,
                            )
                        dist = motor_distribution(result, active, log_std, minimum_noise)
                        new_logps.append(
                            joint_log_probability(
                                result,
                                dist,
                                data["latent"][t],
                                data["activity"][t],
                                motor_only=motor_mode,
                            )
                        )
                        entropy.append(
                            dist.entropy().mean()
                            if motor_mode
                            else torch.distributions.Categorical(logits=result.utility_logits)
                            .entropy()
                            .mean()
                        )
                        if wing_weight:
                            wing_errors.append(
                                F.mse_loss(
                                    result.action[:, reward_fn.posture.wings],
                                    data["wing_target"][t],
                                )
                            )
                        if retainer is not None:
                            ground = torch.as_tensor(task_ids != 2, device=device)
                            retention_errors.append(
                                F.mse_loss(
                                    result.action[ground],
                                    data["ground_reference_action"][t, ground],
                                )
                            )
                        state = brain.reset_worlds(result.state, data["done"][t])
                    new_logp = torch.stack(new_logps)
                    log_ratio = new_logp - data["logp"][a:b]
                    ratio = log_ratio.exp()
                    approx_kl = ((ratio - 1) - log_ratio).mean()
                    kl_values.append(float(approx_kl.detach()))
                    if approx_kl.detach() > args.target_kl:
                        stop_epoch = True
                        break
                    policy_loss = torch.maximum(
                        -adv[a:b] * ratio, -adv[a:b] * ratio.clamp(0.8, 1.2)
                    ).mean()
                    value_loss = (
                        torch.zeros((), device=device)
                        if independent_critic
                        else F.mse_loss(torch.stack(predictions), returns[a:b])
                    )
                    physical_loss = policy_loss - args.entropy * torch.stack(entropy).mean()
                    wing_loss = torch.stack(wing_errors).mean() if wing_weight else None
                    loss = physical_loss + (wing_weight * wing_loss if wing_weight else 0)
                    retention_loss = (
                        torch.stack(retention_errors).mean() if retainer is not None else None
                    )
                    if retention_loss is not None:
                        loss = loss + retention_weight * retention_loss
                    if not torch.isfinite(loss + value_loss):
                        raise RuntimeError("Nonfinite PPO loss")
                    optimizer.zero_grad(set_to_none=True)
                    if (
                        not warming_critic
                        and gradient_audit is None
                        and (wing_weight or retention_weight)
                    ):
                        gradient_audit = core_gradient_stats(brain, physical_loss)
                        if wing_weight:
                            wing_gradient_audit = core_gradient_stats(
                                brain, wing_weight * wing_loss
                            )
                        if retention_weight:
                            retention_gradient_audit = core_gradient_stats(
                                brain, retention_weight * retention_loss
                            )
                        if not all(
                            x["finite"] and x["l2"] > 0 for x in gradient_audit.values()
                        ):
                            raise RuntimeError("Physical reward did not reach neural core")
                    if not warming_critic:
                        loss.backward()
                    if not warming_critic and gradient_audit is None:
                        gradient_audit = {
                            n: {
                                "l2": float(p.grad.norm()),
                                "finite": bool(torch.isfinite(p.grad).all()),
                                "nonzero_cells": int((p.grad != 0).sum()),
                            }
                            for n, p in brain.core.named_parameters()
                        }
                        if not all(
                            item["finite"] and item["l2"] > 0
                            for item in gradient_audit.values()
                        ):
                            raise RuntimeError("Physical reward did not reach neural core")
                    if not warming_critic:
                        nn.utils.clip_grad_norm_(parameters, 1.0)
                        optimizer.step()
                    if not independent_critic:
                        value_optimizer.zero_grad(set_to_none=True)
                        value_loss.backward()
                        nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
                        value_optimizer.step()
                        counters["critic_updates"] += 1
                        counters["critic_sample_presentations"] += args.sequence * args.worlds
                        value_losses.append(float(value_loss.detach()))
                    counters["ppo_updates"] += int(not warming_critic)
                    if wing_weight:
                        counters["wing_supervised_presentations"] += (
                            args.sequence * args.worlds
                        )
                        wing_losses.append(float(wing_loss.detach()))
                    if retention_loss is not None:
                        retention_losses.append(float(retention_loss.detach()))
                    policy_losses.append(float(policy_loss.detach()))
                if stop_epoch:
                    break
            counters["actor_kl_stops"] += int(stop_epoch)
            if independent_critic:
                synchronize(device)
                critic_start = time.perf_counter()
                critic_fit = fit_critic(
                    critic,
                    value_optimizer,
                    data["critic_features"],
                    returns,
                    args.sequence,
                    critic_epochs,
                    critic_rng,
                    shuffle=shuffle_critic,
                )
                if motor_mode:
                    critic_after = value_quality(
                        critic_fit["predictions"], returns, task_ids, task_transitions
                    )
                del critic_fit["predictions"]
                value_losses.extend(critic_fit.pop("losses"))
                counters["critic_updates"] += critic_fit["updates"]
                counters["critic_sample_presentations"] += critic_fit["sample_presentations"]
                counters["critic_updates_after_actor_stop"] += (
                    critic_fit["updates"] if stop_epoch else 0
                )
                synchronize(device)
                counters["independent_critic_seconds"] += time.perf_counter() - critic_start
            synchronize(device)
            ppo_seconds = time.perf_counter() - update_start
            counters["ppo_optimization_seconds"] += ppo_seconds
            rehearsal_start = time.perf_counter()
            rehearsal_loss = None
            if args.rehearsal is not None:
                # One short explicit rehearsal batch per rollout, including reset contexts.
                reset_start = rng.random() < 0.25
                burnin = 0 if reset_start else 8
                demo = sample(
                    rehearsal,
                    rng,
                    burnin + args.sequence,
                    args.worlds,
                    device,
                    reset_start=reset_start,
                )
                state = brain.initial_state(args.worlds)
                with torch.no_grad():
                    for t in range(burnin):
                        state = brain(
                            demo["observation"][t], state, time_scale=rehearsal_time_scale
                        ).state
                rehearsal_losses = []
                for t in range(burnin, burnin + args.sequence):
                    result = brain(
                        demo["observation"][t], state, time_scale=rehearsal_time_scale
                    )
                    state = result.state
                    rehearsal_losses.append(
                        F.mse_loss(result.action, demo["action"][t])
                        + 0.02 * F.cross_entropy(result.utility_logits, demo["activity"][t])
                    )
                rehearsal_loss = torch.stack(rehearsal_losses).mean()
                optimizer.zero_grad(set_to_none=True)
                (args.rehearsal_weight * rehearsal_loss).backward()
                nn.utils.clip_grad_norm_(parameters, 1.0)
                optimizer.step()
            synchronize(device)
            rehearsal_seconds = (
                time.perf_counter() - rehearsal_start if args.rehearsal else 0.0
            )
            counters["rehearsal_seconds"] += rehearsal_seconds
            counters["optimization_seconds"] += ppo_seconds + rehearsal_seconds
            counters["rehearsal_frames"] += (
                args.sequence * args.worlds if args.rehearsal else 0
            )
            counters["rollouts"] += 1
            counters["critic_warmup_rollouts"] += int(warming_critic)
            row = {
                **counters,
                "elapsed_seconds": time.perf_counter() - start,
                "actor_updates_enabled": not warming_critic,
                "mean_physical_reward": float(np.mean(physical_rewards)),
                "reward_rates": term_sums,
                "max_kl": max(kl_values, default=None),
                "policy_loss": float(np.mean(policy_losses)) if policy_losses else None,
                "value_loss": float(np.mean(value_losses)) if value_losses else None,
                "wing_supervision_mse": float(np.mean(wing_losses)) if wing_losses else None,
                "motor_retention_mse": float(np.mean(retention_losses))
                if retention_losses
                else None,
                "rehearsal_loss": float(rehearsal_loss.detach())
                if rehearsal_loss is not None
                else None,
                "completed_episodes": len(episode_records),
                "failed_episodes": sum(e["failed"] for e in episode_records),
                "critic_by_task_before_update": critic_quality,
                "critic_fit_on_collected_rollout": critic_fit,
                "critic_by_task_after_update": critic_after,
                "transitions_by_task": task_transitions.copy(),
                "hover_reset_widening": tasks.widening if hover_physical else None,
            }
            progress.append(row)
            print(json.dumps(row), flush=True)
            with (args.output / "progress.jsonl").open("a") as stream:
                stream.write(json.dumps(row) + "\n")
    except Exception as error:
        failure = f"{type(error).__name__}: {error}"
        np.savez_compressed(args.output / "error_snapshot.npz", **env.fields)
        raise
    finally:
        synchronize(device)
        elapsed = time.perf_counter() - start
        config = {k: v for k, v in vars(args).items() if not isinstance(v, Path)}
        config["internal_steps"] = brain.internal_steps
        if motor_mode:
            config["ground_posture"] = True
        if retainer is not None:
            assert all(
                torch.equal(v.cpu(), retained_state[k])
                for k, v in retainer.actor.state_dict().items()
            )
        checkpoint = {
            "wing_residual_enabled": brain.wing_residual is not None,
            "wing_residual_hidden": brain.wing_residual_hidden,
            "state_dict": {k: v.detach().cpu() for k, v in brain.state_dict().items()},
            "observation_size": brain.observation_size,
            "sensor_extension_size": brain.sensor_extension_size,
            "action_size": brain.action_size,
            "config": config,
            "graph_sha256": parent["graph_sha256"],
            "graph_metadata_sha256": sha256(args.graph / "brain.npz"),
            "source_commit": run_evidence["source_commit"],
            "parent_checkpoint_sha256": sha256(args.resume),
            "method": (
                "recurrent physical-outcome PPO, motor-only closed-flight curriculum"
                if round_trip
                else "recurrent physical-outcome PPO, motor-only hover-first curriculum"
                if hover_only
                else "recurrent physical-outcome PPO plus corrective wing supervision"
                if wing_weight
                else "recurrent physical-outcome PPO, motor-only all-command curriculum"
                if motor_all
                else "recurrent physical-outcome PPO, motor-only ground curriculum"
            )
            if motor_mode
            else "recurrent physical-outcome PPO plus explicit imitation rehearsal",
            "motor_only": motor_mode,
            "physical_contract": contract,
            "critic_state_dict": critic.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "value_optimizer_state_dict": value_optimizer.state_dict(),
            "log_std": log_std.detach().cpu(),
        }
        torch.save(checkpoint, args.output / "actor.pt")
        report = {
            "provenance": run_evidence,
            "config": config,
            "reward_recipe": reward_fn.recipe,
            "motor_only": motor_mode,
            "physical_contract": contract,
            "training_tasks": list(tasks.report()["transitions_by_target_sequence"])
            if round_trip
            else ["hover"]
            if hover_only
            else ["stand", "walk", "hover"]
            if motor_all
            else ["stand", "walk"]
            if motor_ground
            else None,
            "worlds_by_task": {
                name: int((task_ids == i).sum())
                for i, name in enumerate(("stand", "walk", "hover"))
            }
            if motor_mode
            else None,
            "utility_and_intention_weights_unchanged": all(
                torch.equal(brain.state_dict()[k], v) for k, v in frozen.items()
            )
            if motor_mode
            else None,
            "active_motor_channels": int(active.sum()),
            "teacher_present_during_collection": bool(wing_weight or retention_weight),
            "executed_teacher_actions": False,
            "wing_supervision": {
                "weight": wing_weight,
                "channels": reward_fn.posture.wings.tolist() if wing_weight else [],
                "target": "bounded corrective torque from pre-action measured wing q/qvel",
                "kp": reward_fn.posture.wing_kp if wing_weight else None,
                "kd": reward_fn.posture.wing_kd if wing_weight else None,
                "loss": "mean squared normalized action error over six wing channels only",
                "included_in_ppo_optimization_seconds": True,
                "deployed_module": False,
            },
            "motor_retention": {
                "weight": retention_weight,
                "parent_checkpoint_sha256": sha256(args.resume)
                if retainer is not None
                else None,
                "weights_unchanged": retainer is not None,
                "scope": "all 78 mean motor outputs, ground worlds only",
                "runtime_module": False,
                "extra_physics_transitions": 0,
            },
            "core_gradient_audit_from_ground_retention": retention_gradient_audit,
            "training_started_utc": training_started_utc,
            "completed_utc": utc_now(),
            "setup_seconds": setup_seconds,
            "training_wall_seconds": elapsed,
            **counters,
            "physics_backend": "native MuJoCo CPU / mjbatch",
            "physics_threads": env.batch.num_threads,
            "neural_device": str(device),
            "neural_device_name": torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else "CPU",
            "parallel_physics_worlds": args.worlds,
            "aggregate_simulated_seconds": counters["transitions"] * control_dt,
            "physics_hz": 1 / env.model.opt.timestep,
            "control_hz": 1 / control_dt,
            "neural_time_scale": time_scale,
            "effective_gamma": gamma,
            "effective_gae_lambda": gae_lambda,
            "physical_timescales": timing,
            "activation_recomputation": recompute,
            "critic_schedule": {
                "learning_rate": critic_lr,
                "independent_of_actor_kl": independent_critic,
                "epochs": critic_epochs,
                "sample_order": "shuffled time/world transitions"
                if shuffle_critic
                else "shuffled contiguous time windows",
                "input_standardization": "first training rollout mean/std, frozen thereafter; floor .05 and clip +/-10"
                if standardize_critic
                else "none beyond actor observation statistics",
                "features": "normalized observation + preceding descending state",
                "feature_source": "saved during physical collection"
                if independent_critic
                else "actor recurrent replay during optimization",
                "targets": "fixed timeout-aware GAE returns from the collected rollout",
                "updates_per_complete_rollout": critic_epochs * args.horizon // args.sequence
                if independent_critic
                else None,
                "actor_or_physics_replay_for_value_fit": not independent_critic,
                "deployed_module": False,
            },
            "transitions_by_task": task_transitions,
            "aggregate_simulated_seconds_by_task": {
                k: v * control_dt for k, v in task_transitions.items()
            },
            "hover_reset_curriculum": tasks.report() if hover_physical else None,
            "airborne_resets": flight_resets.report if flight_resets else None,
            "rehearsal_control_hz": rehearsal_hz,
            "transitions_per_training_wall_second": counters["transitions"] / elapsed,
            "actor_parameters": sum(p.numel() for p in brain.parameters()),
            "critic_parameters": sum(p.numel() for p in critic.parameters()),
            "core_gradient_audit_from_physical_reward": gradient_audit,
            "core_gradient_audit_from_weighted_wing_supervision": wing_gradient_audit,
            "core_changes": {
                n: float((p.detach() - core_initial[n]).norm())
                for n, p in brain.core.named_parameters()
            },
            "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device)
            if device.type == "cuda"
            else 0,
            "parent_checkpoint_sha256": sha256(args.resume),
            "checkpoint_sha256": sha256(args.output / "actor.pt"),
            "model_sha256": sha256(args.output / "model.mjb"),
            "flight_force_model": env.wing_forces.report() if env.wing_forces else None,
            "rehearsal_manifest_sha256": sha256(args.rehearsal / "manifest.json")
            if args.rehearsal
            else None,
            "rehearsal_episode_sha256": {
                p.name: sha256(p) for p in sorted(args.rehearsal.glob("episode_*.npz"))
            }
            if args.rehearsal
            else {},
            "rehearsal_validation_indices": sorted(int(i) for i in validation_ids),
            "completed_episodes": episode_records,
            "failure": failure,
            "physical_success": "Not established by training reward; independent evaluation required",
            "optimizer_resumed": optimizer_resumed,
            "exploration_reset_explicitly": reset_exploration,
            "initial_exploration_std": initial_exploration,
            "critic_reset_for_new_reward": reset_critic,
            "optimizer_initialization": (
                "retained actor Adam; explicitly reset exploration and its Adam; "
                + ("new critic/critic Adam" if reset_critic else "retained critic/critic Adam")
            )
            if optimizer_resumed and reset_exploration
            else "retained actor Adam/exploration; new critic and critic Adam"
            if optimizer_resumed and reset_critic
            else "retained actor/critic Adam and exploration from PPO parent"
            if optimizer_resumed
            else "new PPO and critic Adam; parent actor and normalization retained",
            "critic_warmup": {
                "requested_rollouts": critic_warmup,
                "completed_rollouts": counters["critic_warmup_rollouts"],
                "actor_and_exploration_frozen_during_warmup": True,
                "included_in_training_wall_seconds": True,
                "extra_physics_worlds": 0,
            },
        }
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--resume", type=Path, required=True)
    parser.add_argument("--rehearsal", type=Path)
    parser.add_argument("--motor-ground", action="store_true")
    parser.add_argument("--motor-all", action="store_true")
    parser.add_argument("--hover-physical", action="store_true")
    parser.add_argument("--hover-only", action="store_true")
    parser.add_argument(
        "--round-trip",
        action="store_true",
        help="Explicit mixed hover and closed-flight curriculum",
    )
    parser.add_argument("--bounded-hover-reward", action="store_true")
    parser.add_argument(
        "--hover-vertical-speed-scale",
        type=float,
        default=5.0,
        help="Bounded hover reward vertical-speed scale in cm/s; changed reward requires --reset-critic",
    )
    parser.add_argument("--reset-critic", action="store_true")
    parser.add_argument("--reset-exploration", action="store_true")
    parser.add_argument("--critic-lr", type=float, default=3e-4)
    parser.add_argument("--checkpoint-activations", action="store_true")
    parser.add_argument("--independent-critic", action="store_true")
    parser.add_argument("--critic-epochs", type=int)
    parser.add_argument("--critic-standardize-inputs", action="store_true")
    parser.add_argument("--critic-shuffle-transitions", action="store_true")
    parser.add_argument("--motor-retention-weight", type=float, default=0.0)
    parser.add_argument("--critic-warmup-rollouts", type=int, default=0)
    parser.add_argument("--wing-supervision", type=float, default=0.0)
    parser.add_argument("--wing-angle-perturbation", type=float, default=0.0)
    parser.add_argument("--wing-speed-perturbation", type=float, default=0.0)
    parser.add_argument(
        "--preset",
        choices=("walking", "flight", "wing_motion", "wing_position"),
        default="walking",
    )
    parser.add_argument("--flight-resets", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--worlds", type=int, default=32)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--seconds", type=float, default=300)
    parser.add_argument("--horizon", type=int, default=64)
    parser.add_argument("--sequence", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--episode-seconds", type=float, default=2)
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--noise", type=float, default=0.04)
    parser.add_argument("--minimum-noise", type=float, default=0.01)
    parser.add_argument("--target-kl", type=float, default=0.03)
    parser.add_argument("--entropy", type=float, default=0.001)
    parser.add_argument("--rehearsal-weight", type=float, default=1.0)
    parser.add_argument("--stationary-cost", type=float, default=0.0)
    parser.add_argument("--stationary-turn-cost", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=28001)
    train(parser.parse_args())
