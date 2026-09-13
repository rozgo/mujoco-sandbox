"""Explicit parameter initialization for the new shared wing position body.

This is not a compatible resume or a learned controller result. Six output rows
are reinitialized for new action units. All other existing tensors are preserved.
"""

import argparse
import json
import time
from pathlib import Path

import mujoco
import numpy as np
import torch

from embodied_fly.batch import FlyBatch
from embodied_fly.evaluate import load_actor
from embodied_fly.physical_contract import physical_contract
from embodied_fly.provenance import evidence, sha256, utc_now
from embodied_fly.wing_position import normalize_targets
from embodied_fly.wing_position import report as actuator_report


def initialize_wing_rows(actor, targets):
    if actor.wing_residual is not None or actor.observation_size != 397:
        raise ValueError("Migration requires an unextended 397-input motor actor")
    original = {k: v.detach().cpu().clone() for k, v in actor.state_dict().items()}
    with torch.no_grad():
        actor.motor_decoder[3].weight[14:20] = 0
        actor.motor_decoder[3].bias[14:20] = torch.as_tensor(
            np.arctanh(np.clip(targets, -0.98, 0.98)),
            dtype=actor.motor_decoder[3].bias.dtype,
            device=actor.motor_decoder[3].bias.device,
        )
    actor.enable_wing_residual(128)
    # Every old tensor outside the declared six output rows must remain exact.
    for name, before in original.items():
        after = actor.state_dict()[name].detach().cpu()
        if name in ("motor_decoder.3.weight", "motor_decoder.3.bias"):
            keep = np.r_[0:14, 20:78]
            assert torch.equal(before[keep], after[keep])
        else:
            assert torch.equal(before, after), name


def run(args):
    started = time.perf_counter()
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    actor, parent = load_actor(args.resume, args.graph, torch.device("cpu"))
    old = FlyBatch(3, 3, 14, preset="wing_motion")
    if not parent.get("motor_only") or parent["physical_contract"] != physical_contract(
        old.model
    ):
        raise ValueError("Migration must verify the parent's exact original physical body")
    new = FlyBatch(3, 3, 14, preset="wing_position")
    t = new.template
    target = normalize_targets(new.model, t.data.qpos[t.wing_angle_indices])
    initialize_wing_rows(actor, target)
    args.output.mkdir(parents=True, exist_ok=False)
    mujoco.mj_saveModel(new.model, str(args.output / "model.mjb"))
    migration = {
        "source": parent["physical_contract"],
        "destination": physical_contract(new.model),
        "parent_checkpoint_sha256": sha256(args.resume),
        "six_wing_rows": "zero weights; resting position bias capped at normalized +/-0.98",
        "new_wing_readout": "815 -> 128 tanh -> 6; zero final layer",
        "all_other_original_tensors_exactly_preserved": True,
        "actuators": actuator_report(),
        "training_transitions": 0,
        "policy_success": False,
        "seed": args.seed,
    }
    checkpoint = {
        k: parent[k]
        for k in (
            "graph_sha256",
            "graph_metadata_sha256",
            "observation_size",
            "sensor_extension_size",
            "action_size",
        )
    }
    checkpoint.update(
        state_dict={k: v.detach().cpu() for k, v in actor.state_dict().items()},
        motor_only=True,
        wing_residual_enabled=True,
        wing_residual_hidden=128,
        physical_contract=physical_contract(new.model),
        migration=migration,
        parent_checkpoint_sha256=sha256(args.resume),
        config={
            "internal_steps": actor.internal_steps,
            "preset": "wing_position",
            "ground_posture": True,
            "hover_reference": "state",
        },
    )
    torch.save(checkpoint, args.output / "actor.pt")
    result = {
        "provenance": evidence(),
        "completed_utc": utc_now(),
        "migration_seconds": time.perf_counter() - started,
        "checkpoint_sha256": sha256(args.output / "actor.pt"),
        "model_sha256": sha256(args.output / "model.mjb"),
        **migration,
    }
    (args.output / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--resume", type=Path, required=True)
    p.add_argument("--graph", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seed", type=int, default=95002)
    run(p.parse_args())
