"""Independent trajectory measurements of stride, cadence, slip and body bounce."""

import numpy as np
import torch

from .bodies import CONTROL_DT, LEGS
from .evaluate import lane_command, make_case
from .train import load_checkpoint


def inspect_stride(checkpoint, trials=16, seed=9137, seconds=12):
    torch.set_num_threads(1)
    net, saved = load_checkpoint(checkpoint)
    env = make_case("healthy", trials=trials, seed=seed)
    positions, forces, base, velocity = [], [], [], []
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
    finally:
        env.close()
    positions, forces, base, velocity = map(
        np.asarray, (positions, forces, base, velocity)
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
            legs.append(
                {
                    "leg": LEGS[leg],
                    "completed_cycles": len(lengths),
                    "mean_stride_m": float(np.mean(lengths)) if len(lengths) else None,
                    "mean_cycle_seconds": float(np.mean(periods))
                    if len(periods)
                    else None,
                    "strides_per_second": float(1 / np.mean(periods))
                    if len(periods)
                    else None,
                }
            )
        foot_velocity = np.diff(positions[:, trial, :, :2], axis=0) / CONTROL_DT
        planted = contact[1:, trial] & contact[:-1, trial]
        slip2 = np.sum(foot_velocity[start:] ** 2, -1)[planted[start:]]
        rows.append(
            {
                "trial": trial,
                "legs": legs,
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
    )
    return {
        "checkpoint": str(checkpoint),
        "training_seconds": saved["cumulative_training_seconds"],
        "trials": trials,
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
    }
