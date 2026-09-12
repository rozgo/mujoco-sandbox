"""Moving-support experiments, separate from the accepted static release."""

import argparse
from pathlib import Path

from .bodies import ROOT

STANDING = ROOT / "assets/locomotion/checkpoints/standing/consolidate_120s_seed12.pt"
WALKING = ROOT / "assets/locomotion/checkpoints/adaptive_dog_v1.pt"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("preview")
    p = commands.add_parser("train")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--resume", type=Path, default=STANDING)
    p.add_argument("--seconds", type=float, default=60)
    p.add_argument("--seed", type=int, default=21)
    p.add_argument("--num-envs", type=int, default=4096)
    p.add_argument("--profile", choices=("healthy", "mixed"), default="mixed")
    p.add_argument("--motion-scale", type=float, default=1)
    p.add_argument("--physics-backend", choices=("mjbatch", "warp"), default="warp")
    p.add_argument("--device", default="cuda")
    p.add_argument("--learning-rate", type=float, default=0.0001)
    p.add_argument("--standing-replay-weight", type=float, default=50)
    p.add_argument("--initial-std", type=float, default=0.12)
    p.add_argument("--max-iterations", type=int)
    p.add_argument(
        "--reason",
        default="Short moving-support extension; preserve static standing and walking in a shared policy.",
    )
    p = commands.add_parser("evaluate")
    p.add_argument("--checkpoint", type=Path, default=STANDING)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--motions", default="all")
    p.add_argument("--body", default="healthy")
    p.add_argument("--trials", type=int, default=4)
    p.add_argument("--seconds", type=float, default=10)
    p.add_argument("--seed", type=int, default=9401)
    p.add_argument("--physics-backend", choices=("mjbatch", "warp"), default="mjbatch")
    p.add_argument("--world-observations", action="store_true")
    p.add_argument("--motion-scale", type=float, default=1)
    p = commands.add_parser("view")
    p.add_argument("--theme", choices=("classic", "graphite"), default="classic")
    p.add_argument("--checkpoint", type=Path, default=STANDING)
    p.add_argument("--motion", default="combined")
    p.add_argument("--body", default="healthy")
    p.add_argument("--seconds", type=float, default=0)
    p.add_argument("--world-observations", action="store_true")
    p = commands.add_parser("record")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--baseline", type=Path, default=STANDING)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seed", type=int, default=9411)
    p.add_argument("--physics-backend", choices=("mjbatch", "warp"), default="mjbatch")
    p.add_argument("--replay-from", type=Path)
    args = parser.parse_args()
    if args.command == "preview":
        from .moving_scene import preview

        preview()
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
            standing_profile="mixed",
            moving_profile=args.profile,
            motion_scale=args.motion_scale,
            walking_replay_weight=15,
            standing_replay_weight=args.standing_replay_weight,
            standing_reference=STANDING,
            idle_support_weight=8,
            idle_drift_weight=3,
            substep_support=True,
            learning_rate=args.learning_rate,
            reference=WALKING,
            reference_loss_weight=3,
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
    elif args.command == "view":
        from .moving_record import view

        view(
            args.checkpoint,
            args.motion,
            args.body,
            args.seconds,
            not args.world_observations,
            theme=args.theme,
        )
    elif args.command == "record":
        from .moving_record import record

        record(
            args.checkpoint,
            args.baseline,
            args.output,
            args.seed,
            args.physics_backend,
            args.replay_from,
        )
    elif args.command == "evaluate":
        from .moving_evaluate import evaluate

        report = evaluate(
            args.checkpoint,
            args.output,
            args.motions,
            args.trials,
            args.seconds,
            args.seed,
            args.physics_backend,
            not args.world_observations,
            args.body,
            args.motion_scale,
        )
        return 0 if report["passed"] == report["trials"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
