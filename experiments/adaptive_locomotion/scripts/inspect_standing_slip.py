"""Measure contact-point slip in the existing terrain traces, without stepping."""

import hashlib
import json

import mujoco
import numpy as np

from adaptive_locomotion.bodies import LEGS, ROOT


def main():
    source = ROOT / "previews/locomotion/standing/failure_review.json"
    report = json.loads(source.read_text())
    rows = []
    for case in report["cases"][:8]:
        assert case["body"] == "healthy" and not case["transition"]
        for field in ("model", "trajectory"):
            assert (
                hashlib.sha256((ROOT / case[field]).read_bytes()).hexdigest()
                == case[field + "_sha256"]
            )
        model = mujoco.MjModel.from_binary_path(str(ROOT / case["model"]))
        data = mujoco.MjData(model)
        trace = np.load(ROOT / case["trajectory"], allow_pickle=False)
        terminal = {model.geom(f"{leg}_terminal").id: i for i, leg in enumerate(LEGS)}
        jac = np.zeros((3, model.nv))
        speeds, coefficients = [], []
        for k, t in enumerate(trace["time"]):
            if t < 2:
                continue
            data.qpos[:], data.qvel[:], data.ctrl[:] = (
                trace["qpos"][k],
                trace["qvel"][k],
                trace["ctrl"][k],
            )
            mujoco.mj_forward(model, data)
            for contact in data.contact:
                if contact.dist > 0:
                    continue
                pair = (int(contact.geom1), int(contact.geom2))
                foot = next((g for g in pair if g in terminal), None)
                if foot is None or trace["tip_forces"][k, terminal[foot]] <= 5:
                    continue
                other = pair[1] if pair[0] == foot else pair[0]
                if model.geom_bodyid[other] != 0:
                    continue
                mujoco.mj_jac(
                    model, data, jac, None, contact.pos, model.geom_bodyid[foot]
                )
                velocity = jac @ data.qvel
                normal = contact.frame[:3]
                tangential = velocity - normal * np.dot(normal, velocity)
                speeds.append(float(np.linalg.norm(tangential)))
                coefficients.append(float(contact.friction[0]))
        rows.append(
            {
                "surface": case["surface"],
                "trial": case["selected_trial"],
                "contact_samples": len(speeds),
                "mean_tangential_slip_mps": float(np.mean(speeds)),
                "p95_tangential_slip_mps": float(np.percentile(speeds, 95)),
                "foot_sliding_friction": [
                    float(model.geom_friction[g, 0]) for g in terminal
                ],
                "observed_contact_sliding_friction": sorted(set(coefficients)),
                "friction_cone": mujoco.mjtCone(model.opt.cone).name,
                "impratio": model.opt.impratio,
                "noslip_iterations": model.opt.noslip_iterations,
                "hold_drift_m": case["repeated_row"]["max_idle_drift_m"],
            }
        )
        trace.close()
    output = ROOT / "docs/locomotion/standing/TERRAIN_SLIP_REVIEW.json"
    output.write_text(
        json.dumps(
            {
                "source": str(source.relative_to(ROOT)),
                "source_report_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "checkpoint_sha256": report["checkpoint_sha256"],
                "method": "Replay kinematics only, no mj_step. Tangential velocity at each active foot-to-static-terrain contact point from mj_jac @ qvel; recorded terminal load >5 N; hold window 2–10 s; 50 Hz samples, each contact equally weighted.",
                "limitation": "Slip is measured, but no friction counterfactual was run. These measurements do not establish how much slip is caused by the coefficient versus contact regularization, geometry or control. Reconstructed contacts are sampled, not all physics substeps.",
                "new_training_seconds": 0,
                "new_simulation_seconds": 0,
                "cases": rows,
            },
            indent=2,
        )
        + "\n"
    )
    for row in rows:
        print(
            row["surface"],
            round(row["mean_tangential_slip_mps"] * 100, 2),
            "cm/s mean",
            row["observed_contact_sliding_friction"],
        )


if __name__ == "__main__":
    main()
