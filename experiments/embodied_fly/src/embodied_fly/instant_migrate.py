"""Explicit frozen-weight transfer to per-tick wing forces, without training."""

import argparse
import json
import time
from pathlib import Path

import mujoco
import numpy as np
import torch

from embodied_fly.batch import FlyBatch
from embodied_fly.physical_contract import ARRAYS, OPTIONS, physical_contract
from embodied_fly.provenance import evidence, sha256, utc_now


def transfer(parent, old_model, new_model, parent_hash):
    source, destination = physical_contract(old_model), physical_contract(new_model)
    if (
        not parent.get("motor_only")
        or parent.get("physical_contract") != source
        or source["preset"] != "wing_position"
        or "wing_response" in source
        or destination.get("wing_response") != "instant"
    ):
        raise ValueError(
            "Transfer requires the recorded filtered position body and instant destination"
        )
    for name in ARRAYS:
        np.testing.assert_array_equal(getattr(old_model, name), getattr(new_model, name))
    for name in OPTIONS:
        np.testing.assert_array_equal(
            getattr(old_model.opt, name), getattr(new_model.opt, name)
        )
    migration = {
        "source": source,
        "destination": destination,
        "parent_checkpoint_sha256": parent_hash,
        "change": "remove 12 ms wing-activity average; use current speed and angles every physics step",
        "actor_critic_normalization_exploration_and_optimizer_tensors_unchanged": True,
        "rigid_body_joints_actuators_contacts_damping_and_clocks_unchanged": True,
        "training_transitions": 0,
        "optimizer_updates": 0,
        "policy_success": "not established; frozen-weight transfer must be evaluated",
    }
    checkpoint = dict(parent)
    checkpoint.update(
        physical_contract=destination,
        force_response_migration=migration,
        parent_checkpoint_sha256=parent_hash,
        config=dict(parent["config"], wing_response="instant"),
    )
    return checkpoint, migration


def run(args):
    started = time.perf_counter()
    run_evidence = evidence()
    parent = torch.load(args.resume, map_location="cpu", weights_only=True)
    old = FlyBatch(1, 1, 14, preset="wing_position")
    new = FlyBatch(1, 1, 14, preset="wing_position", wing_response="instant")
    checkpoint, migration = transfer(parent, old.model, new.model, sha256(args.resume))
    checkpoint["parent_source_commit"] = parent.get("source_commit")
    checkpoint["source_commit"] = run_evidence["source_commit"]
    args.output.mkdir(parents=True, exist_ok=False)
    mujoco.mj_saveModel(new.model, str(args.output / "model.mjb"))
    torch.save(checkpoint, args.output / "actor.pt")
    result = {
        "provenance": run_evidence,
        "completed_utc": utc_now(),
        "migration_seconds": time.perf_counter() - started,
        "checkpoint_sha256": sha256(args.output / "actor.pt"),
        "model_sha256": sha256(args.output / "model.mjb"),
        "force_model": new.wing_forces.report(),
        **migration,
    }
    (args.output / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())
