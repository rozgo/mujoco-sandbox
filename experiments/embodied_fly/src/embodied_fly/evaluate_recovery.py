"""Evaluate actor-only correction from withheld, matched flight/brain histories."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from embodied_fly.evaluate import load_actor
from embodied_fly.provenance import evidence, sha256, utc_now
from embodied_fly.recovery_starts import RecoveryStarts, restore
from embodied_fly.train import synchronize
from embodied_fly.velocity_demonstrations import environment, post_row, pre_row
from embodied_fly.velocity_exercise import rolling_velocity
from embodied_fly.velocity_hover import failures, metrics, reward_from_recipe
from embodied_fly.velocity_motor import observation


def recovery_windows(arrays, failure_seconds, seconds):
    if failure_seconds is not None or len(arrays["time"]) < round(seconds * 500):
        return {"complete_recovery_window": False}
    velocity = rolling_velocity(arrays["measured_velocity"] * 10)
    result = {"complete_recovery_window": True}
    for label, v in [("first_second", velocity[:500]), ("last_second", velocity[-500:])]:
        result[label] = {
            "total_rms_mm_s": float(np.sqrt(np.mean(np.sum(v**2, axis=1)))),
            "horizontal_rms_mm_s": float(np.sqrt(np.mean(np.sum(v[:, :2] ** 2, axis=1)))),
            "vertical_rms_mm_s": float(np.sqrt(np.mean(v[:, 2] ** 2))),
            "mean_vertical_mm_s": float(v[:, 2].mean()),
        }
    return result


@torch.no_grad()
def evaluate(args):
    args.output.mkdir(parents=True, exist_ok=False)
    begin = time.perf_counter()
    device = torch.device(args.device)
    actor, checkpoint = load_actor(args.checkpoint, args.graph, device)
    actor.eval()
    env = environment(6, 6)
    bank_report = json.loads((args.bank / "report.json").read_text())
    reward = reward_from_recipe(env, bank_report["reward_recipe"])
    if checkpoint["physical_contract"] != bank_report["physical_contract"]:
        raise ValueError("Evaluation physical contract differs")
    bank = RecoveryStarts(
        args.bank, env, bank_report["parent_checkpoint_sha256"], actor, reward, 0
    )
    chosen = np.array([i for i, r in enumerate(bank.records) if r["episode"] >= 8])
    if len(chosen) != 6:
        raise ValueError("Expected all six withheld recovery histories")
    memory = actor.initial_state(6)
    restore(env, reward, memory, np.arange(6), {k: v[chosen] for k, v in bank.arrays.items()})
    commands = np.zeros((6, 4), np.float32)
    setup_seconds = time.perf_counter() - begin
    synchronize(device)
    begin = time.perf_counter()
    first_failure = np.full(6, np.nan)
    rows = []
    for step in range(round(args.seconds * 500)):
        # Preserve the existing PPO observation convention at the saved boundary.
        row = pre_row(env, commands, step * 0.002, 0, refresh=False)
        row["angular_velocity"] = env.velocity()[:, :3].copy()
        result = actor(torch.as_tensor(observation(env, commands), device=device), memory)
        row["action"] = result.action.cpu().numpy()
        env.step(row["action"])
        reward()
        memory = result.state
        row.update(post_row(env))
        rows.append(row)
        newly = failures(env) & np.isnan(first_failure)
        first_failure[newly] = (step + 1) * 0.002
    synchronize(device)
    capture_seconds = time.perf_counter() - begin
    cases = []
    for world, record in enumerate(chosen):
        arrays = {k: np.stack([r[k][world] for r in rows]) for k in rows[0]}
        failure = None if np.isnan(first_failure[world]) else float(first_failure[world])
        file = args.output / f"recovery_{int(record):02d}.npz"
        np.savez_compressed(file, **arrays)
        cases.append(
            {
                "record": int(record),
                **bank.records[record],
                "file": file.name,
                "sha256": sha256(file),
                **metrics(arrays, failure, args.seconds),
                **recovery_windows(arrays, failure, args.seconds),
            }
        )
    report = {
        "provenance": evidence(),
        "completed_utc": utc_now(),
        "checkpoint_sha256": sha256(args.checkpoint),
        "physical_contract": checkpoint["physical_contract"],
        "bank_sha256": bank_report["states_sha256"],
        "seconds": args.seconds,
        "cases": cases,
        "setup_seconds": setup_seconds,
        "capture_seconds": capture_seconds,
        "total_wall_seconds": time.perf_counter() - begin + setup_seconds,
        "scope": "Six withheld parent-reached body/neural histories; actor alone; no resets or assistance during flight",
        "observation_timing": "physics-step return, matching PPO and recovery-bank collection",
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--graph", type=Path, required=True)
    p.add_argument("--bank", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seconds", type=float, default=3)
    args = p.parse_args()
    if not np.isfinite(args.seconds) or args.seconds < 2:
        raise ValueError("At least two physical seconds required for non-overlapping windows")
    evaluate(args)
