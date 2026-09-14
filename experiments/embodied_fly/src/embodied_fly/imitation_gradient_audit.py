"""Measure the first teacher anchor's gradient without changing any weights."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from embodied_fly.evaluate import load_actor
from embodied_fly.provenance import evidence, sha256, utc_now
from embodied_fly.train import synchronize
from embodied_fly.wing_readout_ppo import freeze_upstream, readout_action, teacher_features


def audit(args):
    if args.output.exists():
        raise FileExistsError("Preserve prior diagnostics")
    started = time.perf_counter()
    torch.set_num_threads(4)
    device = torch.device(args.device)
    actor, parent = load_actor(args.checkpoint, args.graph, device)
    actor.train()
    freeze_upstream(actor)
    rng = np.random.default_rng()
    rng.bit_generator.state = json.loads(parent["sampler_state_json"])
    rng.permutation(parent["ppo_recipe"]["horizon"] // parent["ppo_recipe"]["sequence"])
    starts = rng.integers(64, 936, size=8)
    starts[0] = 0
    obs = torch.as_tensor(
        np.load(args.dataset / "observations.npy", mmap_mode="r")[:8, :1000].copy(),
        device=device,
    )
    target = torch.as_tensor(
        np.load(args.dataset / "actions.npy", mmap_mode="r")[:8, :1000].copy(),
        device=device,
    )
    features = teacher_features(actor, obs)
    indices = torch.as_tensor(starts[None] + np.arange(64)[:, None], device=device)
    worlds = torch.arange(8, device=device)[None]
    predicted = readout_action(actor, features[worlds, indices])
    loss = (predicted[..., 14:20] - target[worlds, indices][..., 14:20]).square().mean()
    loss.backward()
    gradients = {
        n: float(p.grad.norm()) for n, p in actor.named_parameters() if p.requires_grad
    }
    teacher_norm = float(np.sqrt(sum(x * x for x in gradients.values())))
    comparisons = {}
    for run in args.run:
        record = json.loads((run / "report.json").read_text())
        if record["parent_checkpoint_sha256"] != sha256(args.checkpoint):
            raise ValueError("Recorded physical gradients must share this parent")
        if record["recipe"]["critic_warmup_rollouts"] != 0:
            raise ValueError("Expected actor updates on the first rollout")
        physical = record["physical_gradient_audit"]
        physical_norm = float(np.sqrt(sum(x * x for x in physical.values())))
        comparisons[run.name] = {
            "training_report_sha256": sha256(run / "report.json"),
            "physical_gradient_l2": physical_norm,
            "anchor_to_physical_gradient_norm_ratio": teacher_norm / physical_norm,
            "recorded_first_anchor_loss": record["progress"][0]["imitation_loss"],
        }
    synchronize(device)
    result = {
        "provenance": evidence(),
        "completed_utc": utc_now(),
        "wall_seconds": time.perf_counter() - started,
        "checkpoint_sha256": sha256(args.checkpoint),
        "observations_sha256": sha256(args.dataset / "observations.npy"),
        "actions_sha256": sha256(args.dataset / "actions.npy"),
        "first_anchor_loss": float(loss.detach()),
        "anchor_gradient_per_parameter": gradients,
        "anchor_gradient_l2": teacher_norm,
        "comparisons": comparisons,
        "scope": "First update only, before gradient clipping or Adam; no optimizer steps, training, or live controls",
        "limitation": "A norm ratio is not a gradient angle or a bound on Adam updates; later gradients may differ",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: result[k]
                for k in (
                    "wall_seconds",
                    "first_anchor_loss",
                    "anchor_gradient_l2",
                    "comparisons",
                )
            }
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--run", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    audit(parser.parse_args())
