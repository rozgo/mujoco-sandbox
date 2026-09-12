"""Separate commands/artifacts for the shared walking and standing extension."""

import argparse
from pathlib import Path

from .bodies import ROOT

BASELINE = ROOT / "assets/locomotion/checkpoints/adaptive_dog_v1.pt"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("preview")
    p.add_argument("--aggressive", action="store_true")
    p = commands.add_parser("train")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--resume", type=Path, default=BASELINE)
    p.add_argument("--seconds", type=float, default=60)
    p.add_argument("--seed", type=int, default=12)
    p.add_argument("--num-envs", type=int, default=4096)
    p.add_argument("--profile", choices=("healthy", "mixed"), default="healthy")
    p.add_argument(
        "--surfaces", choices=("gentle", "aggressive", "all"), default="gentle"
    )
    p.add_argument("--physics-backend", choices=("mjbatch", "warp"), default="warp")
    p.add_argument("--device", default="cuda")
    p.add_argument("--learning-rate", type=float, default=0.0001)
    p.add_argument("--reference-weight", type=float, default=3)
    p.add_argument("--walking-replay-weight", type=float, default=0)
    p.add_argument("--idle-support-weight", type=float, default=2)
    p.add_argument("--idle-drift-weight", type=float, default=0)
    p.add_argument("--idle-episode-steps", type=int, default=500)
    p.add_argument("--substep-support", action="store_true")
    p.add_argument("--initial-std", type=float, default=0.20)
    p.add_argument("--max-iterations", type=int)
    p.add_argument(
        "--reason",
        default="First short standing extension with walking retained by frozen v1 targets.",
    )
    p = commands.add_parser("evaluate")
    p.add_argument("--checkpoint", type=Path, default=BASELINE)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--surfaces", default="gentle")
    p.add_argument("--bodies", default="healthy")
    p.add_argument("--trials", type=int, default=4)
    p.add_argument("--seconds", type=float, default=10)
    p.add_argument("--seed", type=int, default=9301)
    p.add_argument("--physics-backend", choices=("mjbatch", "warp"), default="mjbatch")
    p.add_argument("--no-transitions", action="store_true")
    p = commands.add_parser("record")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--physics-backend", choices=("mjbatch", "warp"), default="mjbatch")
    p.add_argument("--seed", type=int, default=9311)
    p.add_argument("--replay-from", type=Path)
    p = commands.add_parser("view")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--surface", default="gap_fr")
    p.add_argument("--body", default="healthy")
    p.add_argument("--seconds", type=float, default=0)
    p.add_argument("--transition", action="store_true")
    args = parser.parse_args()
    if args.command == "preview":
        from .standing_surfaces import preview

        preview(args.aggressive)
    elif args.command == "train":
        from .train import load_checkpoint, train

        _, parent = load_checkpoint(args.resume)
        train(
            args.output,
            seconds=args.seconds,
            resume=args.resume,
            mode="support",
            seed=args.seed,
            num_envs=args.num_envs,
            device=args.device,
            physics_backend=args.physics_backend,
            standing_profile=args.profile,
            standing_surfaces=args.surfaces,
            walking_replay_weight=args.walking_replay_weight,
            idle_support_weight=args.idle_support_weight,
            idle_drift_weight=args.idle_drift_weight,
            idle_episode_steps=args.idle_episode_steps,
            substep_support=args.substep_support,
            learning_rate=args.learning_rate,
            reference=BASELINE,
            reference_loss_weight=args.reference_weight,
            reward_profile="walk",
            stride_weight=1,
            balance_weight=5,
            body_motion_weight=1,
            damage_flight_weight=3,
            visible_step_weight=2,
            rear_overlap_weight=3,
            initial_std=args.initial_std,
            minibatch_size=3072,
            max_iterations=args.max_iterations,
            allowance=parent["cumulative_training_seconds"] + args.seconds + 1,
            extension_reason=args.reason,
        )
    elif args.command == "record":
        from .standing_record import record

        record(
            args.checkpoint,
            args.output,
            args.physics_backend,
            seed=args.seed,
            replay_from=args.replay_from,
        )
    elif args.command == "view":
        from .standing_record import view

        view(args.checkpoint, args.surface, args.body, args.seconds, args.transition)
    else:
        from .standing_evaluate import evaluate

        report = evaluate(
            args.checkpoint,
            args.output,
            args.surfaces,
            args.bodies,
            args.trials,
            args.seconds,
            args.seed,
            args.physics_backend,
            not args.no_transitions,
        )
        return 0 if report["passed"] == report["trials"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
