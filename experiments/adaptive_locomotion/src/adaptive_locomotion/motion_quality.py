"""Fixed-window motion measurements, including absolute high-frequency motion."""

import numpy as np


def measure(q, action, gyro, velocity, height, support, dt=0.02):
    """Inputs are time x trials x active channels; no filtering of live control.

    Spectral RMS uses a demeaned Hann window and one-sided Parseval weights.
    Measuring absolute motion avoids a misleading improvement in a power ratio
    caused merely by increasing low-frequency motion. All trials are included,
    even failures, so completion must be checked separately before comparison.
    """
    q, action, gyro, velocity, height, support = map(
        np.asarray, (q, action, gyro, velocity, height, support)
    )
    window = np.hanning(len(q))
    spectrum = np.abs(np.fft.rfft((q - q.mean(0)) * window[:, None, None], axis=0)) ** 2
    weights = np.full(len(spectrum), 2.0)
    weights[0] = 1
    if len(q) % 2 == 0:
        weights[-1] = 1
    power = spectrum * weights[:, None, None] / (len(q) * (window**2).sum())
    frequency = np.fft.rfftfreq(len(q), dt)
    high = power[frequency > 6].sum(0).mean(1)
    total = power[1:].sum(0).mean(1)
    values = {
        "action_step_rms": np.sqrt(np.mean(np.diff(action, axis=0) ** 2, axis=(0, 2))),
        "joint_above_6hz_rms_rad": np.sqrt(high),
        "joint_above_6hz_power_fraction": high / np.maximum(total, 1e-12),
        "roll_pitch_rate_rms_rad_s": np.sqrt(
            np.mean(np.sum(gyro[:, :, :2] ** 2, 2), 0)
        ),
        "vertical_velocity_rms_mps": np.sqrt(np.mean(velocity[:, :, 2] ** 2, 0)),
        "forward_speed_mps": velocity[:, :, 0].mean(0),
        "height_std_m": height.std(0),
    }
    return {
        "window_s": [1.0, 12.0],
        "spectral_window": "Hann, demeaned, power normalized, active joints only",
        "summary": {k: float(v.mean()) for k, v in values.items()},
        "per_trial": [
            {k: float(v[i]) for k, v in values.items()} for i in range(q.shape[1])
        ],
        "mean_support_duty_FL_FR_RL_RR": (support > 1).mean(axis=(0, 1)).tolist(),
    }
