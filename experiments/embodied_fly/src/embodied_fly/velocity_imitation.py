"""Timed, resumable recurrent imitation from complete velocity exercises."""

import argparse
import json
import time
from pathlib import Path

import mujoco
import numpy as np
import torch

from embodied_fly.evaluate import load_actor
from embodied_fly.physical_contract import physical_contract
from embodied_fly.ppo_timing import recurrent_forward
from embodied_fly.provenance import evidence, sha256, utc_now
from embodied_fly.train import synchronize
from embodied_fly.velocity_exercise import STAGES
from embodied_fly.velocity_motor import SCHEMA
from embodied_fly.wing_position import wing_actuators


def sample_windows(
    rng, episodes, batch, sequence, warmup, steps, cold_starts=0, stage_windows=None
):
    """Cover every stage each update; never join two physical episodes."""
    if (
        cold_starts < 0
        or batch - cold_starts < len(STAGES)
        or min(sequence, warmup, steps) <= 0
    ):
        raise ValueError("Need at least 50 samples and positive sequence lengths")
    boundaries = np.rint(np.r_[0, np.cumsum([s[1] for s in STAGES])] * 500).astype(int)
    episodes = np.asarray(list(episodes))
    regular = batch - cold_starts
    stages = np.r_[
        np.arange(len(STAGES)), rng.integers(len(STAGES), size=regular - len(STAGES))
    ]
    rng.shuffle(stages)
    if stage_windows is None:
        starts = np.array(
            [
                rng.integers(boundaries[s], min(boundaries[s + 1], steps - sequence + 1))
                for s in stages
            ]
        )
        worlds = rng.choice(episodes, size=regular)
    else:
        worlds = rng.choice(episodes, size=regular)
        starts = np.array(
            [
                rng.integers(
                    stage_windows[w, s, 0], min(stage_windows[w, s, 1], steps - sequence + 1)
                )
                for w, s in zip(worlds, stages, strict=True)
            ]
        )
    if cold_starts:
        # Whole recorded episode starts with zero neural memory. Recovery
        # episodes can start with already moving wings; no live reset is inserted.
        cold_worlds = np.resize(rng.permutation(episodes), cold_starts)
        starts = np.r_[starts, np.zeros(cold_starts, dtype=int)]
        worlds = np.r_[worlds, cold_worlds]
        stages = np.r_[stages, np.zeros(cold_starts, dtype=int)]
    indices = starts[None] + np.arange(-warmup, sequence)[:, None]
    valid = indices >= 0
    return worlds, indices.clip(0, steps - 1), valid, stages


def validate_resume_recipe(
    previous, current, allow_sampling_change=False, allow_dataset_change=False
):
    previous = dict(previous, cold_starts=previous.get("cold_starts", 0))
    if previous == current:
        return False
    allowed = set()
    if allow_sampling_change:
        allowed.add("cold_starts")
    if allow_dataset_change:
        allowed.add("dataset_report_sha256")
    if {k: v for k, v in previous.items() if k not in allowed} != {
        k: v for k, v in current.items() if k not in allowed
    }:
        raise ValueError(
            "Continuation must preserve the recipe except explicit sampling/dataset changes"
        )
    return True


def sequence_loss(actor, observations, targets, valid, warmup, wings, gradients):
    """Rebuild recurrent context with current weights, then truncate gradients."""
    state = actor.initial_state(observations.shape[1])
    context = torch.ones(observations.shape[1], dtype=torch.long, device=observations.device)
    with torch.no_grad():
        for step in range(warmup):
            result = actor(observations[step], state, activity_override=context)
            state = torch.where(valid[step][None], result.state, state)
    state = state.detach()
    errors = []
    with torch.set_grad_enabled(gradients):
        for step in range(warmup, len(observations)):
            result = recurrent_forward(actor, observations[step], state, context, 1, gradients)
            state = result.state
            errors.append((result.action - targets[step - warmup]).square())
        per_channel = torch.stack(errors).mean((0, 1))
        other = torch.ones(actor.action_size, dtype=torch.bool, device=observations.device)
        other[wings] = False
        wing = per_channel[wings].mean()
        posture = per_channel[other].mean()
        loss = wing + 0.1 * posture
    return loss, wing, posture, per_channel


def train(args):
    if args.seconds <= 0 or args.lr <= 0 or args.snapshot_seconds < 0:
        raise ValueError("Positive budget and learning rate required")
    args.output.mkdir(parents=True, exist_ok=False)
    setup_started = time.perf_counter()
    provenance = evidence()
    torch.set_num_threads(4)
    device = torch.device(args.device)
    actor, parent = load_actor(args.initial or args.resume, args.graph, device)
    if parent.get("observation_schema") != SCHEMA or not actor.motor_only:
        raise ValueError("Only the fresh 391-input velocity student is supported")
    if args.initial and (
        parent.get("training_updates") != 0 or parent.get("initialization") != "from_scratch"
    ):
        raise ValueError("Initial input must be a fresh untrained checkpoint")
    data_report = json.loads((args.dataset / "report.json").read_text())
    if not data_report["passed"] or data_report.get("probe_only", False):
        raise ValueError("Teacher dataset has not passed the accepted flight gates")
    for filename in ("observations", "actions"):
        if sha256(args.dataset / f"{filename}.npy") != data_report[f"{filename}_sha256"]:
            raise ValueError("Demonstration checksum mismatch")
    model = mujoco.MjModel.from_binary_path(str(args.dataset / "model.mjb"))
    if (
        physical_contract(model) != parent["physical_contract"]
        or parent["physical_contract"] != data_report["physical_contract"]
    ):
        raise ValueError("Teacher and student physics differ")
    observations = torch.as_tensor(np.load(args.dataset / "observations.npy"), device=device)
    targets = torch.as_tensor(np.load(args.dataset / "actions.npy"), device=device)
    if (
        observations.shape[:2] != targets.shape[:2]
        or observations.shape[2] != 391
        or targets.shape[2] != 78
    ):
        raise ValueError("Invalid demonstration layout")
    wings = torch.as_tensor(wing_actuators(model), device=device)
    train_episodes = data_report.get("train_episode_ids", list(range(8)))
    validation_episodes = data_report.get("validation_episode_ids", [8, 9])
    if set(train_episodes) & set(validation_episodes):
        raise ValueError("Training and validation episodes overlap")
    stage_windows = data_report.get("stage_windows")
    if stage_windows is not None:
        stage_windows = np.asarray(stage_windows, dtype=np.int64)
        if stage_windows.shape != (observations.shape[0], len(STAGES), 2):
            raise ValueError("Invalid per-episode stage window metadata")
    parameters = [p for p in actor.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(parameters, lr=args.lr)
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    recipe = {
        "batch": args.batch,
        "sequence": args.sequence,
        "warmup": args.warmup,
        "lr": args.lr,
        "seed": args.seed,
        "dataset_report_sha256": sha256(args.dataset / "report.json"),
        "loss": "wing MSE + 0.1 * nonwing MSE",
        "gradient_clip": 1,
        "dtype": "float32",
        "cold_starts": args.cold_starts,
    }
    sampling_changed = False
    if args.resume:
        sampling_changed = validate_resume_recipe(
            parent["imitation_recipe"],
            recipe,
            args.allow_sampling_change,
            args.allow_dataset_change,
        )
        optimizer.load_state_dict(parent["optimizer_state_dict"])
        rng.bit_generator.state = json.loads(parent["sampler_state_json"])
        torch.set_rng_state(parent["torch_rng_state"].cpu())
        if device.type == "cuda":
            torch.cuda.set_rng_state(parent["cuda_rng_state"].cpu(), device)
    initial_parameters = {k: p.detach().cpu().clone() for k, p in actor.named_parameters()}
    validation_windows = sample_windows(
        np.random.default_rng(121102),
        validation_episodes,
        args.batch,
        args.sequence,
        args.warmup,
        observations.shape[1],
        stage_windows=stage_windows,
    )

    def get_batch(windows):
        worlds, indices, valid, _ = windows
        wi = torch.as_tensor(worlds, device=device)[None]
        ix = torch.as_tensor(indices, device=device)
        return (
            observations[wi, ix],
            targets[wi, ix[args.warmup :]],
            torch.as_tensor(valid, device=device),
        )

    def validate():
        actor.eval()
        begin = time.perf_counter()
        obs, target, valid = get_batch(validation_windows)
        loss, wing, posture, channel = sequence_loss(
            actor, obs, target, valid, args.warmup, wings, False
        )
        synchronize(device)
        return {
            "loss": float(loss),
            "wing_mse": float(wing),
            "posture_mse": float(posture),
            "wing_channel_mse": channel[wings].cpu().tolist(),
            "wall_seconds": time.perf_counter() - begin,
            "full_flight_evaluation": False,
            "split": f"held-out episodes {validation_episodes}",
        }

    synchronize(device)
    setup_seconds = time.perf_counter() - setup_started
    before = validate()
    actor.train()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    progress = []
    audit = None
    stage_counts = np.zeros(len(STAGES), dtype=int)
    snapshots = []
    checkpoint_seconds = 0.0
    next_snapshot = args.snapshot_seconds or float("inf")
    parent_hash = sha256(args.initial or args.resume)

    def save_checkpoint(name, training_seconds):
        result = {
            k: v for k, v in parent.items() if k not in ("state_dict", "optimizer_state_dict")
        }
        result.update(
            state_dict={k: v.detach().cpu() for k, v in actor.state_dict().items()},
            optimizer_state_dict=optimizer.state_dict(),
            sampler_state_json=json.dumps(rng.bit_generator.state),
            torch_rng_state=torch.get_rng_state(),
            cuda_rng_state=torch.cuda.get_rng_state(device).cpu()
            if device.type == "cuda"
            else torch.empty(0),
            training_updates=parent.get("training_updates", 0) + len(progress),
            cumulative_training_seconds=parent.get("cumulative_training_seconds", 0)
            + training_seconds,
            parent_checkpoint_sha256=parent_hash,
            source_commit=provenance["source_commit"],
            imitation_recipe=recipe,
            method="Recurrent velocity PID imitation from complete physical demonstrations",
        )
        torch.save(result, args.output / name)
        return result

    started_utc = utc_now()
    synchronize(device)
    started = time.perf_counter()
    with (args.output / "progress.jsonl").open("x") as log:
        while time.perf_counter() - started - checkpoint_seconds < args.seconds:
            windows = sample_windows(
                rng,
                train_episodes,
                args.batch,
                args.sequence,
                args.warmup,
                observations.shape[1],
                args.cold_starts,
                stage_windows,
            )
            obs, target, valid = get_batch(windows)
            optimizer.zero_grad(set_to_none=True)
            loss, wing, posture, channel = sequence_loss(
                actor, obs, target, valid, args.warmup, wings, True
            )
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite imitation loss")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(parameters, 1, error_if_nonfinite=True)
            if audit is None:
                audit = {
                    name: float(p.grad.norm())
                    for name, p in actor.named_parameters()
                    if p.requires_grad and p.grad is not None
                }
                for prefix in (
                    "sensory_encoder.",
                    "motor_decoder.",
                    "wing_residual.",
                    "core.",
                ):
                    if not any(v > 0 for k, v in audit.items() if k.startswith(prefix)):
                        raise RuntimeError(f"No gradient through {prefix}")
            optimizer.step()
            synchronize(device)
            stage_counts += np.bincount(windows[3], minlength=len(STAGES))
            row = {
                "update": len(progress) + 1,
                "elapsed_seconds": time.perf_counter() - started - checkpoint_seconds,
                "loss": float(loss.detach()),
                "wing_mse": float(wing.detach()),
                "posture_mse": float(posture.detach()),
                "gradient_norm": float(norm),
                "wing_channel_mse": channel[wings].detach().cpu().tolist(),
                "stages_sampled": np.flatnonzero(stage_counts).tolist(),
            }
            progress.append(row)
            log.write(json.dumps(row) + "\n")
            log.flush()
            print(json.dumps(row), flush=True)
            if row["elapsed_seconds"] >= next_snapshot:
                checkpoint_started = time.perf_counter()
                name = f"checkpoint_{int(next_snapshot):04d}s.pt"
                saved = save_checkpoint(name, row["elapsed_seconds"])
                snapshots.append(
                    {
                        "file": name,
                        "sha256": sha256(args.output / name),
                        "training_seconds": row["elapsed_seconds"],
                        "cumulative_updates": saved["training_updates"],
                    }
                )
                checkpoint_seconds += time.perf_counter() - checkpoint_started
                next_snapshot += args.snapshot_seconds
    synchronize(device)
    training_seconds = time.perf_counter() - started - checkpoint_seconds
    after = validate()
    # All continuation state is retained; no reset to random weights next burst.
    result = save_checkpoint("actor.pt", training_seconds)
    changed = [
        k
        for k, p in actor.named_parameters()
        if not torch.equal(p.detach().cpu(), initial_parameters[k])
    ]
    assert not any(k.startswith(("utility_head.", "intention_encoder.")) for k in changed)
    assert torch.all(actor.observation_mean == 0) and torch.all(actor.observation_std == 1)
    assert sha256(args.graph / "weights.npz") == parent["graph_sha256"]
    report = {
        "provenance": provenance,
        "started_utc": started_utc,
        "completed_utc": utc_now(),
        "recipe": recipe,
        "setup_seconds": setup_seconds,
        "requested_training_seconds": args.seconds,
        "training_wall_seconds": training_seconds,
        "checkpoint_write_seconds_excluded": checkpoint_seconds,
        "snapshot_interval_seconds": args.snapshot_seconds,
        "snapshots": snapshots,
        "explicit_sampling_change": sampling_changed,
        "explicit_dataset_change": bool(
            args.resume
            and parent["imitation_recipe"]["dataset_report_sha256"]
            != recipe["dataset_report_sha256"]
        ),
        "train_episode_ids": train_episodes,
        "validation_episode_ids": validation_episodes,
        "updates_this_burst": len(progress),
        "cumulative_updates": result["training_updates"],
        "cumulative_training_seconds": result["cumulative_training_seconds"],
        "supervised_targets": len(progress) * args.batch * args.sequence,
        "warmup_observations": len(progress) * args.batch * args.warmup,
        "padded_cold_start_context_slots": len(progress) * args.cold_starts * args.warmup,
        "cold_start_sequences": len(progress) * args.cold_starts,
        "unique_training_dataset_transitions": len(train_episodes) * observations.shape[1],
        "physical_worlds_during_optimizer": 0,
        "physical_steps_during_optimizer": 0,
        "replay_sequences_in_parallel": args.batch,
        "stage_window_counts": stage_counts.tolist(),
        "validation_before": before,
        "validation_after": after,
        "gradient_audit": audit,
        "changed_parameters": changed,
        "graph_sha256": parent["graph_sha256"],
        "graph_unchanged": True,
        "physical_contract": parent["physical_contract"],
        "checkpoint_sha256": sha256(args.output / "actor.pt"),
        "parent_checkpoint_sha256": result["parent_checkpoint_sha256"],
        "optimizer_continuation_saved": True,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device)
        if device.type == "cuda"
        else 0,
        "neural_device": torch.cuda.get_device_name(device)
        if device.type == "cuda"
        else str(device),
        "actor_parameters": sum(p.numel() for p in actor.parameters()),
        "trainable_parameters": sum(p.numel() for p in parameters),
        "teacher_absent_at_deployment": True,
        "learned_flight_not_yet_evaluated": True,
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: v
                for k, v in report.items()
                if k not in ("provenance", "gradient_audit", "changed_parameters")
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--initial", type=Path)
    mode.add_argument("--resume", type=Path)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seconds", type=float, default=60)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--sequence", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=121101)
    parser.add_argument("--cold-starts", type=int, default=0)
    parser.add_argument("--allow-sampling-change", action="store_true")
    parser.add_argument("--allow-dataset-change", action="store_true")
    parser.add_argument("--snapshot-seconds", type=float, default=0)
    train(parser.parse_args())
