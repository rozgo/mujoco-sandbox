"""Independent trajectory measurements of stride, cadence, slip and body bounce."""

import numpy as np
import torch

from .bodies import CONTROL_DT, LEGS
from .evaluate import lane_command, make_case
from .train import load_checkpoint


def inspect_stride(checkpoint, trials=16, seed=9137, seconds=12, case="healthy"):
    torch.set_num_threads(1)
    net, saved = load_checkpoint(checkpoint)
    env = make_case(case, trials=trials, seed=seed)
    positions, forces, base, velocity, angular_velocity = [], [], [], [], []
    try:
        for _ in range(round(seconds / CONTROL_DT)):
            lane_command(env)
            with torch.no_grad():
                action, _ = net(
                    torch.as_tensor(env.obs()),
                    torch.as_tensor(env.context),
                    torch.as_tensor(env.history),
                )
            env.step(action.numpy())
            positions.append(env.tip_positions.copy())
            forces.append(env.tip_forces.copy())
            base.append(env.pos.copy())
            velocity.append(env.vel.copy())
            angular_velocity.append(env.gyro.copy())
    finally:
        env.close()
    positions, forces, base, velocity, angular_velocity = map(
        np.asarray, (positions, forces, base, velocity, angular_velocity)
    )
    contact = forces > 1
    # Offline contact debounce bridges one-sample force dropouts. This diagnostic
    # uses successive landings, unlike the reward's liftoff-to-landing travel.
    contact[1:-1] |= contact[:-2] & contact[2:]
    start = round(1 / CONTROL_DT)
    rows = []
    for trial in range(trials):
        legs = []
        for leg in range(4):
            hits = (
                np.flatnonzero(contact[1:, trial, leg] & ~contact[:-1, trial, leg]) + 1
            )
            hits = hits[hits >= start]
            lengths = np.diff(positions[hits, trial, leg, 0])
            periods = np.diff(hits) * CONTROL_DT
            modes = contact[start:, trial, leg]
            changes = np.flatnonzero(modes[1:] != modes[:-1]) + 1
            intervals = np.diff(changes) * CONTROL_DT
            stance = intervals[modes[changes[:-1]]]
            swing = intervals[~modes[changes[:-1]]]
            legs.append(
                {
                    "leg": LEGS[leg],
                    "completed_cycles": len(lengths),
                    "mean_stride_m": float(np.mean(lengths)) if len(lengths) else None,
                    "mean_cycle_seconds": float(np.mean(periods))
                    if len(periods)
                    else None,
                    "duty_fraction": float(modes.mean()),
                    "mean_completed_stance_s": float(stance.mean())
                    if len(stance)
                    else None,
                    "mean_completed_swing_s": float(swing.mean())
                    if len(swing)
                    else None,
                    "mean_contact_force_magnitude_n": float(
                        forces[start:, trial, leg].mean()
                    ),
                    "strides_per_second": float(1 / np.mean(periods))
                    if len(periods)
                    else None,
                }
            )
        foot_velocity = np.diff(positions[:, trial, :, :2], axis=0) / CONTROL_DT
        planted = contact[1:, trial] & contact[:-1, trial]
        slip2 = np.sum(foot_velocity[start:] ** 2, -1)[planted[start:]]
        duty = np.array([x["duty_fraction"] for x in legs])
        load = np.array([x["mean_contact_force_magnitude_n"] for x in legs])
        share = load / max(load.sum(), 1e-8)
        rows.append(
            {
                "trial": trial,
                "legs": legs,
                "mean_left_right_duty_gap": float(
                    np.abs(duty[[0, 2]] - duty[[1, 3]]).mean()
                ),
                "mean_left_right_load_share_gap": float(
                    np.abs(share[[0, 2]] - share[[1, 3]]).mean()
                ),
                "roll_pitch_rate_rms_rad_s": float(
                    np.sqrt(np.mean(angular_velocity[start:, trial, :2] ** 2))
                ),
                "mean_stride_m": float(np.mean([x["mean_stride_m"] for x in legs]))
                if all(x["mean_stride_m"] is not None for x in legs)
                else None,
                "mean_forward_speed_mps": float(np.mean(velocity[start:, trial, 0])),
                "flight_fraction": float(np.mean(~(forces[start:, trial] > 1).any(1))),
                "base_height_std_m": float(np.std(base[start:, trial, 2])),
                "base_height_min_max_m": [
                    float(base[start:, trial, 2].min()),
                    float(base[start:, trial, 2].max()),
                ],
                "planted_foot_horizontal_speed_rms_mps": float(np.sqrt(np.mean(slip2)))
                if len(slip2)
                else None,
            }
        )
    keys = (
        "mean_stride_m",
        "mean_forward_speed_mps",
        "flight_fraction",
        "base_height_std_m",
        "planted_foot_horizontal_speed_rms_mps",
        "mean_left_right_duty_gap",
        "mean_left_right_load_share_gap",
        "roll_pitch_rate_rms_rad_s",
    )
    return {
        "checkpoint": str(checkpoint),
        "training_seconds": saved["cumulative_training_seconds"],
        "trials": trials,
        "case": case,
        "seed": seed,
        "seconds": seconds,
        "measurement": "after first second; 20 ms positions/contact forces; consecutive landings per foot, one-sample contact debounce",
        "summary": {
            k: float(np.mean([r[k] for r in rows]))
            if all(r[k] is not None for r in rows)
            else None
            for k in keys
        },
        "rows": rows,
        "leg_summary": [
            {
                "leg": name,
                **{
                    key: float(np.mean([row["legs"][j][key] for row in rows]))
                    if all(row["legs"][j][key] is not None for row in rows)
                    else None
                    for key in (
                        "duty_fraction",
                        "mean_completed_stance_s",
                        "mean_completed_swing_s",
                        "mean_contact_force_magnitude_n",
                        "mean_stride_m",
                        "mean_cycle_seconds",
                        "strides_per_second",
                    )
                },
            }
            for j, name in enumerate(LEGS)
        ],
    }
