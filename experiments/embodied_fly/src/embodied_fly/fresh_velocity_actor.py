"""Initialize a velocity-commanded MaleCNS student without loading any policy."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import mujoco
import torch

from embodied_fly.brain import EmbodiedBrain, load_malecns
from embodied_fly.physical_contract import physical_contract
from embodied_fly.provenance import evidence, sha256, utc_now
from embodied_fly.velocity_motor import OBSERVATION_SIZE, SCHEMA, schema_report


def initialize(adjacency, sensory, descending, motor, seed, legacy_wing_readout=False):
    """Standard fresh layer initialization, independent of all earlier fits."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        actor = EmbodiedBrain(
            adjacency,
            sensory,
            descending,
            motor,
            OBSERVATION_SIZE,
            78,
            internal_steps=4,
            sensor_extension_size=0,
            motor_only=True,
            wing_residual_enabled=legacy_wing_readout,
            wing_residual_hidden=128 if legacy_wing_readout else 0,
            shared_decoder_hidden=0 if legacy_wing_readout else 384,
        )
        # This is a fresh parallel readout, not a zero-initialized correction
        # grafted onto an already trained motor policy.
        if legacy_wing_readout:
            actor.wing_residual.network[-1].reset_parameters()
    return actor


def parameter_digest(actor):
    digest = hashlib.sha256()
    for name, parameter in actor.named_parameters():
        digest.update(name.encode())
        digest.update(parameter.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def run(args):
    started = time.perf_counter()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    provenance = evidence()
    graph_hash, metadata_hash = (
        sha256(args.graph / "weights.npz"),
        sha256(args.graph / "brain.npz"),
    )
    adjacency, sensory, descending, motor = load_malecns(args.graph)
    actor = initialize(
        adjacency, sensory, descending, motor, args.seed, args.legacy_wing_readout
    )
    model = mujoco.MjModel.from_binary_path(str(args.model))
    contract = physical_contract(model)
    if (
        contract["physics_hz"] != 1000
        or contract["wing_response"] != "instant"
        or model.nu != actor.action_size
        or not contract.get("independent_heading_control", False)
    ):
        raise ValueError("Fresh student requires the current accepted 1 kHz flight body")
    state = actor.state_dict()
    checkpoint = {
        "state_dict": state,
        "observation_size": OBSERVATION_SIZE,
        "action_size": 78,
        "sensor_extension_size": 0,
        "observation_schema": SCHEMA,
        "motor_only": True,
        "wing_residual_enabled": actor.wing_residual is not None,
        "wing_residual_hidden": actor.wing_residual_hidden,
        "shared_decoder_hidden": actor.shared_decoder_hidden,
        "graph_sha256": graph_hash,
        "graph_metadata_sha256": metadata_hash,
        "physical_contract": contract,
        "config": {"internal_steps": 4, "initialization_seed": args.seed},
        "source_commit": provenance["source_commit"],
        "method": "Fresh random motor student; no learning yet",
        "initialization": "from_scratch",
        "parent_checkpoint_sha256": None,
        "initial_parameter_sha256": parameter_digest(actor),
        "training_updates": 0,
        "command_interface": schema_report(),
    }
    torch.save(checkpoint, args.output / "initial_actor.pt")
    assert graph_hash == sha256(args.graph / "weights.npz")
    report = {
        "provenance": provenance,
        "completed_utc": utc_now(),
        "initialization_wall_seconds": time.perf_counter() - started,
        "checkpoint_sha256": sha256(args.output / "initial_actor.pt"),
        "initial_parameter_sha256": checkpoint["initial_parameter_sha256"],
        "initialization_seed": args.seed,
        "parent_checkpoint_loaded": False,
        "optimizer_created_or_loaded": False,
        "training_updates": 0,
        "physical_steps": 0,
        "learned_normalization_loaded": False,
        "observation_mean_zero_std_one": bool(
            torch.all(actor.observation_mean == 0) and torch.all(actor.observation_std == 1)
        ),
        "cell_initialization": "Fresh neutral gain/leak/bias defaults; no learned neuron parameters inherited",
        "network_initialization": "New seeded random Linear layers; normalization affine weights one/bias zero",
        "recurrent_state": "Zero at episode start; never loaded from an older run",
        "graph_sha256": graph_hash,
        "graph_metadata_sha256": metadata_hash,
        "neurons": actor.core.neurons,
        "connections": actor.core.connections,
        "actor_parameters": sum(p.numel() for p in actor.parameters()),
        "trainable_motor_parameters": sum(
            p.numel() for p in actor.parameters() if p.requires_grad
        ),
        "physical_contract": contract,
        "physical_model_sha256": sha256(args.model),
        "command_interface": schema_report(),
        "scope": "Untrained initialization only; no learned behavior claimed",
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--graph", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seed", type=int, default=121101)
    p.add_argument(
        "--legacy-wing-readout",
        action="store_true",
        help="Reproduce historical initialization only; new policies use the full-body decoder",
    )
    run(p.parse_args())
