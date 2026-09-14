"""Collect parent-reached flight states and verify complete restoration before PPO."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from embodied_fly.evaluate import load_actor
from embodied_fly.physical_contract import physical_contract
from embodied_fly.provenance import evidence, sha256, utc_now
from embodied_fly.recovery_starts import capture, restore
from embodied_fly.train import synchronize
from embodied_fly.velocity_demonstrations import environment
from embodied_fly.velocity_hover import HoverReward, start_states
from embodied_fly.velocity_motor import observation


@torch.no_grad()
def collect(args):
    started = utc_now()
    begin = time.perf_counter()
    args.output.mkdir(parents=True, exist_ok=False)
    device = torch.device(args.device)
    actor, parent = load_actor(args.checkpoint, args.graph, device)
    actor.eval()
    env = environment(10, 10)
    if physical_contract(env.model) != parent["physical_contract"]:
        raise ValueError("Recovery collection physical contract differs")
    env.reset(np.arange(10), state=start_states(args.dataset, range(10)))
    reward = HoverReward(env, 2)
    commands = np.zeros((10, 4), np.float32)
    memory = actor.initial_state(10)
    anchors = (1000, 2000, 3000)
    steps = anchors[-1] + 64
    snapshots, records = [], []
    checks = {a: [] for a in anchors}
    setup_seconds = time.perf_counter() - begin
    synchronize(device)
    begin = time.perf_counter()
    ever_failed = np.zeros(10, dtype=bool)
    for step in range(steps):
        if step in anchors:
            if np.any(ever_failed):
                raise RuntimeError(
                    "Parent failed before a recovery initialization; preserve diagnostic"
                )
            snapshots.append(capture(env, reward, memory))
            records.extend(
                {
                    "episode": i,
                    "parent_flight_seconds": step * 0.002,
                    "split": "training" if i < 8 else "validation",
                }
                for i in range(10)
            )
        obs = observation(env, commands)
        result = actor(torch.as_tensor(obs, device=device), memory)
        action = result.action.cpu().numpy()
        env.step(action)
        score, failed, _ = reward()
        ever_failed |= failed
        memory = result.state
        for anchor in anchors:
            if anchor <= step < anchor + 64:
                checks[anchor].append(
                    {
                        "observation": obs.copy(),
                        "action": action.copy(),
                        "qpos": env.fields["qpos"].copy(),
                        "reward": score.copy(),
                    }
                )
    synchronize(device)
    collection_seconds = time.perf_counter() - begin
    arrays = {k: np.concatenate([s[k] for s in snapshots], axis=0) for k in snapshots[0]}
    np.savez_compressed(args.output / "states.npz", **arrays)
    tolerances = {"observation": 2e-6, "action": 2e-6, "qpos": 1e-8, "reward": 2e-7}
    errors = {k: 0.0 for k in tolerances}
    synchronize(device)
    begin = time.perf_counter()
    for anchor, saved in zip(anchors, snapshots, strict=True):
        restore(env, reward, memory, np.arange(10), saved)
        for expected in checks[anchor]:
            obs = observation(env, commands)
            result = actor(torch.as_tensor(obs, device=device), memory)
            action = result.action.cpu().numpy()
            env.step(action)
            score, _, _ = reward()
            memory = result.state
            actual = {
                "observation": obs,
                "action": action,
                "qpos": env.fields["qpos"],
                "reward": score,
            }
            for k, previous in errors.items():
                errors[k] = max(previous, float(np.max(np.abs(actual[k] - expected[k]))))
    synchronize(device)
    audit_seconds = time.perf_counter() - begin
    passed = not bool(ever_failed.any()) and all(errors[k] <= tolerances[k] for k in errors)
    report = {
        "provenance": evidence(),
        "started_utc": started,
        "completed_utc": utc_now(),
        "parent_checkpoint_sha256": sha256(args.checkpoint),
        "physical_contract": parent["physical_contract"],
        "fixed_graph_sha256": parent["graph_sha256"],
        "reward_recipe": reward.recipe,
        "states_sha256": sha256(args.output / "states.npz"),
        "records": records,
        "training_records": 24,
        "validation_records": 6,
        "collection_worlds": 10,
        "collection_transitions": steps * 10,
        "audit_transitions": len(anchors) * 64 * 10,
        "setup_seconds": setup_seconds,
        "collection_seconds": collection_seconds,
        "restoration_audit_seconds": audit_seconds,
        "restoration_errors": errors,
        "restoration_tolerances": tolerances,
        "parent_failures": int(ever_failed.sum()),
        "restored_continuation_seconds_per_state": 0.128,
        "passed": passed,
        "scope": "Frozen parent collection and reset verification only; no optimization or new force",
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)
    if not passed:
        raise RuntimeError("Full-brain recovery restoration failed; do not train")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--graph", type=Path, required=True)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    collect(p.parse_args())
