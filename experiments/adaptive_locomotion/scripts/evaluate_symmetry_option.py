"""Read-only frozen-policy reflection probe; never used by the deployed viewer."""

import hashlib
import json

import mujoco
import numpy as np
import torch

from adaptive_locomotion.bodies import ROOT
from adaptive_locomotion.evaluate import rollout
from adaptive_locomotion.rear_overlap import rear_swing_metrics
from adaptive_locomotion.symmetry import mirror_action, mirror_observation
from adaptive_locomotion.symmetry_audit import audit
from adaptive_locomotion.train import load_checkpoint


class ReflectedProbe(torch.nn.Module):
    def __init__(self, policy):
        super().__init__()
        self.policy = policy

    def forward(self, obs, context=None, history=None):
        action, value = self.policy(mirror_observation(obs))
        return mirror_action(action), value


def main():
    path = ROOT / "assets/locomotion/checkpoints/limb_visible_steps_selected_seed2.pt"
    policy, _ = load_checkpoint(path)
    assert policy.mode == "blind"
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    rows = []
    for case in ("lower_fl", "lower_fr", "whole_fl", "whole_fr"):
        for label, net in (
            ("original", policy),
            ("reflected_probe", ReflectedProbe(policy)),
        ):
            report, frames, model = rollout(
                net, case, trials=8, seed=9161, capture=True, timestep=0.0005
            )
            data = mujoco.MjData(model)
            ids = [model.geom(f"{l}_terminal").id for l in ("RL", "RR")]
            positions = []
            for frame in frames:
                data.qpos[:] = frame["qpos"]
                mujoco.mj_forward(model, data)
                p = np.zeros((4, 3))
                p[2:] = data.geom_xpos[ids]
                positions.append(p)
            radii = np.r_[0.0, 0.0, model.geom_size[ids, 0]]
            timing = rear_swing_metrics(np.asarray(positions), radii)
            row = {
                "case": case,
                "method": label,
                "trials": 8,
                "completed_with_allowed_support": report[
                    "completed_with_allowed_support"
                ],
                "first_trial_rear_timing": timing,
                "full_task_report": report,
            }
            rows.append(row)
            print(
                json.dumps({k: v for k, v in row.items() if k != "full_task_report"}),
                flush=True,
            )
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
    bank = audit(
        path, path, ROOT / "docs/locomotion/SYMMETRY_CURRENT_AUDIT.json", seed=9161
    )
    result = {
        "checkpoint": str(path.relative_to(ROOT)),
        "checkpoint_sha256": digest,
        "seed": 9161,
        "new_training_seconds": 0,
        "production_controller_changed": False,
        "physics_timestep_s": 0.0005,
        "task_trials_per_case_method": 8,
        "phase_trials_per_case_method": 1,
        "description": "Diagnostic pi_reflected(s)=mirror_action(pi(mirror_observation(s))). Weights unchanged, no training or deployment modification. Shows what reflecting existing behavior does, not what a trained symmetry loss would achieve. Physical rollouts use the normal torque/contact engine.",
        "cases": rows,
        "fixed_state_mirror_audit": "docs/locomotion/SYMMETRY_CURRENT_AUDIT.json",
        "mean_current_action_mirror_rmse": bank["mean_reference_mirror_rmse"],
    }
    (ROOT / "docs/locomotion/SYMMETRY_OPTION_EVALUATION.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
