"""Merge legacy motor outputs into one dense full-body decoder, without fitting."""

import argparse
import json
import time
from pathlib import Path

import torch

from embodied_fly.evaluate import load_actor
from embodied_fly.full_body_decoder import consolidate_actor
from embodied_fly.provenance import evidence, sha256, utc_now


@torch.no_grad()
def run(args):
    started = time.perf_counter()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    device = torch.device(args.device)
    actor, parent = load_actor(args.checkpoint, args.graph, device)
    if actor.shared_decoder_hidden or actor.wing_residual is None:
        raise ValueError("Migration requires the preserved legacy full actor")
    old = {
        name: value.detach().cpu().clone()
        for name, value in actor.state_dict().items()
        if not name.startswith(("motor_decoder.", "wing_residual."))
    }
    # Exercise generic motor activity, including the standardized feature tails.
    generator = torch.Generator(device=device).manual_seed(391301)
    motor = torch.cat(
        [
            torch.randn(512, len(actor.motor_ids), generator=generator, device=device) * scale
            for scale in (0.01, 0.2, 1, 10)
        ],
        0,
    )
    normalized = actor.motor_decoder[0](motor)
    expected_logits = actor.motor_decoder[3](
        actor.motor_decoder[2](actor.motor_decoder[1](normalized))
    )
    expected_logits[:, 14:20] += actor.wing_residual(normalized)
    expected = expected_logits.tanh()
    consolidate_actor(actor)
    actual = actor.motor_decoder(motor)
    error = float((expected - actual).abs().max())
    if error > 3e-6:
        raise RuntimeError(f"Consolidated decoder disagreement {error}")
    for key, value in old.items():
        if not torch.equal(value, actor.state_dict()[key].cpu()):
            raise RuntimeError(f"Non-decoder state changed: {key}")
    # Decoder topology changed: old Adam moments cannot be reused. Keep all
    # original policy weights/history in the immutable parent checkpoint.
    drop = {
        "state_dict",
        "optimizer_state_dict",
        "value_optimizer_state_dict",
        "critic_state_dict",
        "sampler_state_json",
        "torch_rng_state",
        "cuda_rng_state",
        "ppo_recipe",
        "ppo_counters",
        "imitation_recipe",
    }
    checkpoint = {k: v for k, v in parent.items() if k not in drop}
    migration = {
        "kind": "algebraic_full_body_decoder_consolidation_v1",
        "parent_sha256": sha256(args.checkpoint),
        "optimizer_updates": 0,
        "training_seconds": 0,
        "motor_neurons": len(actor.motor_ids),
        "generic_input_views": 2,
        "shared_hidden_units": actor.shared_decoder_hidden,
        "outputs": actor.action_size,
        "maximum_action_disagreement": error,
        "audit_motor_samples": len(motor),
        "all_other_tensors_exactly_preserved": True,
        "output_masks": False,
        "separate_wing_branch": False,
        "decoder_parameters": sum(p.numel() for p in actor.motor_decoder.parameters()),
        "future_trainable_scope": "all motor_decoder parameters; all 78 output rows; no joint-group masks",
        "optimizer_transition": "fresh optimizer required for changed decoder topology",
    }
    checkpoint.update(
        state_dict={k: v.detach().cpu() for k, v in actor.state_dict().items()},
        wing_residual_enabled=False,
        wing_residual_hidden=0,
        shared_decoder_hidden=actor.shared_decoder_hidden,
        decoder_migration=migration,
        parent_checkpoint_sha256=sha256(args.checkpoint),
        method="Consolidated full-body decoder; preserved learned motor function; no additional training",
    )
    torch.save(checkpoint, args.output / "actor.pt")
    report = {
        "provenance": evidence(),
        "completed_utc": utc_now(),
        "wall_seconds": time.perf_counter() - started,
        "checkpoint_sha256": sha256(args.output / "actor.pt"),
        "physical_contract": parent["physical_contract"],
        "migration": migration,
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "provenance"}), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--graph", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    run(p.parse_args())
