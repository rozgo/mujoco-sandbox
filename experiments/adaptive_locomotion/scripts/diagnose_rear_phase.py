import hashlib
import json

import mujoco
import numpy as np

from adaptive_locomotion.bodies import ROOT
from adaptive_locomotion.evaluate import make_case

r = ROOT / "outputs/locomotion/recordings/limb_visible_steps_verified"
m = json.loads((r / "capture.json").read_text())
rows = []
for case in ["lower_fl", "lower_fr", "whole_fl", "whole_fr"]:
    s = np.load(r / f"{case}.npz")
    e = next(x for x in m["cases"] if x["case"] == case)
    assert (
        hashlib.sha256((r / f"{case}.npz").read_bytes()).hexdigest()
        == e["trajectory_sha256"]
    )
    env = make_case(case, trials=1, seed=9143, timestep=0.0005)
    model = env.groups[0].model
    data = mujoco.MjData(model)
    ids = [model.geom(f"{leg}_terminal").id for leg in ["RL", "RR"]]
    h = []
    for q in s["qpos"]:
        data.qpos[:] = q
        mujoco.mj_forward(model, data)
        h.append(data.geom_xpos[ids, 2] - model.geom_size[ids, 0])
    env.close()
    h = np.array(h)
    # Both feet are physically clear by at least 1 cm, not just weakly loaded.
    clear = h > 0.01
    rising = [np.flatnonzero(clear[1:, i] & ~clear[:-1, i]) + 1 for i in range(2)]
    phases = []
    for t in rising[1]:
        j = np.searchsorted(rising[0], t, side="right") - 1
        if j >= 0 and j + 1 < len(rising[0]) and rising[0][j] >= 50:
            phases.append((t - rising[0][j]) / (rising[0][j + 1] - rising[0][j]))
    theta = 2 * np.pi * np.asarray(phases)
    mean = np.mean(np.exp(1j * theta))
    phase = float(np.angle(mean) / (2 * np.pi) % 1)
    rows.append(
        {
            "case": case,
            "rear_swing_phase_fraction": phase,
            "rear_swing_phase_concentration": float(abs(mean)),
            "phase_samples": len(phases),
            "both_rear_feet_above_1cm_fraction": float(clear[50:].all(1).mean()),
            "rear_swing_duty_rl_rr": clear[50:].mean(0).tolist(),
            "trajectory_sha256": e["trajectory_sha256"],
        }
    )
report = {
    "video": "previews/locomotion/limb_visible_steps_verified.mp4",
    "seed": 9143,
    "physics_timestep_s": 0.0005,
    "state_sample_s": 0.02,
    "window_s": [1, 12],
    "measurement": "RL-to-RR crossing of 1 cm clearance, phase relative to bracketing RL cycles. 0 or 1=synchronous, 0.5=alternating. Circular concentration near 1 means consistent timing. One fixed demo per case, not a generalization test.",
    "cases": rows,
}
(ROOT / "docs/locomotion/REAR_ALTERNATION_DIAGNOSTIC.json").write_text(
    json.dumps(report, indent=2) + "\n"
)
print(json.dumps(report, indent=2))
