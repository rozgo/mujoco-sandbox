"""Read-only analysis of preserved captures; no new physical simulation."""
import json
from pathlib import Path

import mujoco
import numpy as np

from embodied_fly.observations import wing_angle_indices
from embodied_fly.provenance import sha256, utc_now


def peak_frequency(values):
    values = values[500:]
    values = values - np.linspace(values[0], values[-1], len(values))
    frequencies = np.fft.rfftfreq(len(values), 0.002)
    amplitude = np.abs(np.fft.rfft(values))
    amplitude[(frequencies < 1) | (frequencies > 100)] = 0
    return float(frequencies[np.argmax(amplitude)])


result = {"completed_utc": utc_now(), "new_physical_transitions": 0, "training_updates": 0}
for kind, name in (
    ("learned", "position_sustain_retention_01_evaluation"),
    ("reference", "hover_recovery_01"),
):
    p = Path("outputs/embodied_fly") / name
    r = json.loads((p / "report.json").read_text())
    model = mujoco.MjModel.from_binary_path(str(p / "model.mjb"))
    assert sha256(p / "model.mjb") == r["model_sha256"]
    if kind == "learned":
        path = p / "hover.npz"
        expected = next(c["state_sha256"] for c in r["results"] if c["case"] == "hover")
        qpos = np.load(path)["qpos"]
    else:
        path = p / "trajectories.npz"
        expected = r["trajectory_sha256"]
        i = next(i for i, c in enumerate(r["cases"])
                 if c["initial_height_cm"] == 2 and c["initial_vertical_speed_cm_s"] == 0)
        qpos = np.load(path)["qpos"][:, i]
    assert sha256(path) == expected
    height = qpos[:, 2]
    wings = qpos[:, wing_angle_indices(model)]
    result[kind] = {
        "source_report_sha256": sha256(p / "report.json"),
        "source_trajectory_sha256": expected,
        "initial_height_cm": float(height[0]),
        "height_frequency_hz": peak_frequency(height),
        "height_full_span_cm": float(np.ptp(height)),
        "height_last_second_span_cm": float(np.ptp(height[-500:])),
        "wing_sweep_frequency_hz": [peak_frequency(wings[:, i]) for i in (0, 3)],
        "wing_sweep_span_rad_after_first_second": [float(np.ptp(wings[500:, i])) for i in (0, 3)],
    }
result["scope"] = (
    "Five-second trajectories; FFT after first second with endpoint linear detrend, "
    "strongest 1-100 Hz bin. Same declared wing_position recipe, separate native "
    "CPU hosts and different initial heights/headings. Not a paired performance "
    "comparison or physiological wingbeat claim. The reference is not learned."
)
Path(__file__).with_name("report.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
