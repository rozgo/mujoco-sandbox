"""Read-only motion diagnostics for recorded hover; never a training reward."""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from embodied_fly.provenance import sha256, utc_now
from embodied_fly.velocity_exercise import rolling_velocity
from embodied_fly.wing_position import wing_actuators


def analyze(run, label):
    report = json.loads((run / "report.json").read_text())
    evaluation = report["evaluations"][label]
    model = mujoco.MjModel.from_binary_path(str(run / "model.mjb"))
    addresses = model.jnt_qposadr[model.actuator_trnid[wing_actuators(model), 0]]
    cases = []
    for case in evaluation["cases"]:
        file = run / label / case["file"]
        if sha256(file) != case["sha256"]:
            raise ValueError("Capture checksum mismatch")
        # Partial failure windows are not silently compared with complete flights.
        if not case["survived_ten_seconds"]:
            cases.append({"episode": case["episode"], "complete_window": False})
            continue
        with np.load(file) as capture:
            times = capture["time"]
            window = (times >= 1) & (times < 10)
            speed = capture["measured_velocity"] * 10  # cm/s -> mm/s
            rapid_speed = (speed - rolling_velocity(speed))[window]
            height = capture["qpos"][window, 2] * 10
            t = times[window]
            detrended_height = height - np.polyval(np.polyfit(t, height, 1), t)
            sweeps = capture["qpos"][window][:, addresses[[0, 3]]]
            frequencies = np.fft.rfftfreq(len(t), d=0.002)
            band = (frequencies >= 1) & (frequencies <= 100)
            spectrum = np.abs(np.fft.rfft(sweeps - sweeps.mean(0), axis=0))
            peaks = frequencies[band][np.argmax(spectrum[band], axis=0)]
            cases.append(
                {
                    "episode": case["episode"],
                    "complete_window": True,
                    "sweep_spectral_peak_hz_left_right": peaks.tolist(),
                    "sweep_peak_to_peak_rad_left_right": np.ptp(sweeps, axis=0).tolist(),
                    "raw_body_speed_rms_mm_s": float(
                        np.sqrt(np.mean(np.sum(speed[window] ** 2, axis=1)))
                    ),
                    "body_speed_minus_100ms_mean_rms_mm_s": float(
                        np.sqrt(np.mean(np.sum(rapid_speed**2, axis=1)))
                    ),
                    "linear_detrended_height_rms_mm": float(
                        np.sqrt(np.mean(detrended_height**2))
                    ),
                    "linear_detrended_height_range_mm": float(np.ptp(detrended_height)),
                }
            )
    return {
        "training_report_sha256": sha256(run / "report.json"),
        "label": label,
        "cases": cases,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, action="append", required=True)
    parser.add_argument(
        "--label", default="final", choices=("pid", "parent", "midpoint", "final")
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Preserve prior diagnostics")
    results = {run.name: analyze(run, args.label) for run in args.run}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "completed_utc": utc_now(),
                "window_seconds": [1, 10],
                "scope": "Recorded physical movement only; no weights, rewards or dynamics changed",
                "interpretation": "Spectral peak is not a guaranteed cycle count. Detrended height includes slow curvature as well as bobbing. Do not substitute these diagnostics for flight acceptance.",
                "runs": results,
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps(results))
