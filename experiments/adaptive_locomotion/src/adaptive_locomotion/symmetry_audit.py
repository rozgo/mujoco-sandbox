"""Compare soft bilateral consistency on one fixed bank of reference-policy states."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .bodies import ROOT
from .evaluate import lane_command, make_case
from .limb_loss import LOSS_CASES
from .symmetry import mirror_action, mirror_observation
from .train import load_checkpoint


def audit(reference, checkpoint, output, seed=9145):
    torch.set_num_threads(1)
    original, _ = load_checkpoint(reference)
    candidate, _ = load_checkpoint(checkpoint)
    reports, all_obs = [], []
    with torch.no_grad():
        for case in LOSS_CASES:
            env = make_case(case, trials=4, seed=seed)
            bank = []
            try:
                for step in range(200):
                    lane_command(env)
                    obs = torch.tensor(env.obs())
                    action, _ = original(obs)
                    if step >= 50 and step % 5 == 0:
                        bank.append(obs)
                    env.step(action.numpy())
            finally:
                env.close()
            obs = torch.cat(bank)
            all_obs.append(obs)
            reflected = mirror_observation(obs)
            valid = reflected[:, 45:57]
            row = {"case": case, "states": len(obs)}
            for label, net in (("reference", original), ("candidate", candidate)):
                action, _ = net(obs)
                other, _ = net(reflected)
                error = ((other - mirror_action(action)) * valid).square().sum(
                    -1
                ) / valid.sum(-1)
                row[label + "_mirror_rmse"] = float(error.mean().sqrt())
            reports.append(row)
        bank = torch.cat(all_obs)
        cpu_actions, _ = candidate(bank)
        agreement = None
        if torch.backends.mps.is_available():
            other, _ = candidate.to("mps")(bank.to("mps"))
            agreement = float((cpu_actions - other.cpu()).abs().max())
    result = {
        "reference": str(Path(reference).resolve().relative_to(ROOT)),
        "checkpoint": str(Path(checkpoint).resolve().relative_to(ROOT)),
        "reference_sha256": hashlib.sha256(Path(reference).read_bytes()).hexdigest(),
        "checkpoint_sha256": hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
        "seed": seed,
        "description": "Same reference-policy states for both models; raw mean actions, existing joints only. This measures policy consistency, not paired gait equality.",
        "states_sha256": hashlib.sha256(bank.numpy().tobytes()).hexdigest(),
        "cases": reports,
        "mean_reference_mirror_rmse": float(
            np.mean([r["reference_mirror_rmse"] for r in reports])
        ),
        "mean_candidate_mirror_rmse": float(
            np.mean([r["candidate_mirror_rmse"] for r in reports])
        ),
        "cpu_mps_max_action_difference": agreement,
    }
    Path(output).write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=9145)
    print(json.dumps(audit(**vars(parser.parse_args())), indent=2))
