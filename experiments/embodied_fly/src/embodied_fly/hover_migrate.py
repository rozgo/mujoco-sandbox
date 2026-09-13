"""Explicit transfer to accepted 1 kHz plant and two horizontal feedback inputs."""

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
        parent.get("physical_contract") != source
        or not parent.get("motor_only")
        or parent.get("sensor_extension_size") != 14
        or parent.get("observation_size") != 397
        or source["preset"] != "wing_position"
        or destination.get("wing_response") != "instant"
        or destination["physics_hz"] != 1000
    ):
        raise ValueError("Requires recorded 397-input position actor and accepted 1 kHz plant")
    for name in ARRAYS:
        np.testing.assert_array_equal(getattr(old_model, name), getattr(new_model, name))
    for name in OPTIONS:
        if name != "timestep":
            np.testing.assert_array_equal(
                getattr(old_model.opt, name), getattr(new_model.opt, name)
            )
    state = dict(parent["state_dict"])
    state["sensor_extension.weight"] = torch.cat(
        [
            state["sensor_extension.weight"],
            torch.zeros(128, 2, dtype=state["sensor_extension.weight"].dtype),
        ],
        dim=1,
    )
    for name, fill in [("observation_mean", 0), ("observation_std", 1)]:
        state[name] = torch.cat([state[name], torch.full((2,), fill, dtype=state[name].dtype)])
    child = dict(
        parent,
        state_dict=state,
        observation_size=399,
        sensor_extension_size=16,
        physical_contract=destination,
        parent_checkpoint_sha256=parent_hash,
        method="frozen actor transfer for hover-first PPO",
        config=dict(parent["config"], wing_response="instant", physics_hz=1000),
    )
    # New physics, reward and critic input dimensions warrant fresh optimizers
    # and value fitting. Actor weights and old normalization stay intact.
    for key in [
        "critic_state_dict",
        "optimizer_state_dict",
        "value_optimizer_state_dict",
        "log_std",
    ]:
        child.pop(key, None)
    report = {
        "source": source,
        "destination": destination,
        "parent_checkpoint_sha256": parent_hash,
        "actor_old_weights_and_normalization_unchanged": True,
        "added_inputs": [
            "horizontal target error, anatomical x / 1 cm",
            "horizontal target error, anatomical y / 1 cm",
        ],
        "new_encoder_weights": "256 zero-initialized weights; old actor outputs initially identical for matching input prefix/state",
        "physics_changes": [
            "instantaneous wing force (no activity average)",
            "1000 Hz physics; 500 Hz controller unchanged",
        ],
        "body_arrays_actuator_limits_and_other_options_unchanged": True,
        "critic_and_optimizers": "new at next PPO stage; no stale value targets or Adam moments transferred",
        "training_transitions": 0,
        "optimizer_updates": 0,
        "policy_success": "not established; evaluate physical transfer",
    }
    child["hover_plant_migration"] = report
    return child, report


def run(args):
    started, provenance = time.perf_counter(), evidence()
    parent = torch.load(args.resume, map_location="cpu", weights_only=True)
    contract = parent["physical_contract"]
    old = FlyBatch(
        1,
        1,
        14,
        preset="wing_position",
        wing_response=contract.get("wing_response", "filtered"),
        physics_hz=contract["physics_hz"],
    )
    new = FlyBatch(1, 1, 16, preset="wing_position", wing_response="instant", physics_hz=1000)
    child, report = transfer(parent, old.model, new.model, sha256(args.resume))
    child["source_commit"] = provenance["source_commit"]
    args.output.mkdir(parents=True, exist_ok=False)
    torch.save(child, args.output / "actor.pt")
    mujoco.mj_saveModel(new.model, str(args.output / "model.mjb"))
    report.update(
        provenance=provenance,
        completed_utc=utc_now(),
        migration_seconds=time.perf_counter() - started,
        checkpoint_sha256=sha256(args.output / "actor.pt"),
    )
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ["migration_seconds", "checkpoint_sha256"]}))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--resume", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    run(p.parse_args())
