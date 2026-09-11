import argparse
from pathlib import Path

from .bodies import ROOT, preview


def main():
    parser = argparse.ArgumentParser(description="Experimental adaptive Go2")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("preview")
    p.add_argument("--output", type=Path, default=ROOT / "previews/locomotion/static")
    p = sub.add_parser("train")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seconds", type=float, default=60)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num-envs", type=int, default=512)
    p.add_argument("--mode", choices=("oracle", "history", "blind"), default="oracle")
    p.add_argument("--resume", type=Path)
    p.add_argument("--bodies", default="all")
    p.add_argument("--terrain", default="flat", choices=("flat", "steps", "test_steps"))
    p.add_argument("--device", default="auto", choices=("auto", "cpu", "mps", "cuda"))
    p.add_argument("--threads", type=int, default=16)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--allowance", type=float, default=300)
    p.add_argument("--extension-reason", default="")
    p.add_argument("--reward-profile", choices=("adaptive", "walk"), default="adaptive")
    p.add_argument("--support-weight", type=float, default=2.0)
    p.add_argument("--stride-weight", type=float, default=0.0)
    p.add_argument("--balance-weight", type=float, default=0.0)
    p.add_argument("--body-motion-weight", type=float, default=0.0)
    p.add_argument("--damage-action-rate-weight", type=float, default=0.0)
    p.add_argument("--damage-angular-rate-weight", type=float, default=0.0)
    p.add_argument("--damage-joint-accel-weight", type=float, default=0.0)
    p.add_argument("--damage-flight-weight", type=float, default=0.0)
    p.add_argument("--damage-clearance-weight", type=float, default=0.0)
    p.add_argument("--front-reference-scale", type=float, default=1.0)
    p.add_argument("--symmetry-weight", type=float, default=0.0)
    p.add_argument("--retention-curriculum", action="store_true")
    p.add_argument("--reference", type=Path)
    p.add_argument("--reference-reward-weight", type=float, default=0.0)
    p.add_argument("--reference-loss-weight", type=float, default=0.0)
    p.add_argument("--damage-healthy-reward-weight", type=float, default=0.0)
    p.add_argument("--damage-healthy-loss-weight", type=float, default=0.0)
    p.add_argument(
        "--healthy-style-source",
        choices=("policy", "motion", "motion_sequence"),
        default="policy",
    )
    p.add_argument("--learning-rate", type=float, default=0.001)
    p.add_argument("--pair-level", choices=("mild", "hard"))
    p.add_argument(
        "--limb-stage", choices=("one", "lower", "whole", "front", "consolidate")
    )
    p.add_argument("--neutralize-validity", choices=("all", "hips_thighs"))
    p.add_argument("--initial-std", type=float)
    p.add_argument("--front-reference", type=Path)
    p.add_argument("--single-reference", type=Path)
    p.add_argument("--single-reference-weight", type=float, default=0.0)
    p = sub.add_parser("evaluate")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--cases", default="all")
    p.add_argument("--trials", type=int, default=16)
    p.add_argument("--seconds", type=float, default=12)
    p = sub.add_parser("record")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--cases", default="short_fl,unseen_pair,unseen_weak,short_steps")
    p.add_argument(
        "--replay",
        action="store_true",
        help="Rerender this output's existing validated trajectories",
    )
    p.add_argument("--seconds", type=float, default=12)
    p = sub.add_parser("view")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--case", default="short_fl")
    p.add_argument("--seconds", type=float, default=0)
    p.add_argument("--presentation", choices=("original", "damage"), default="original")
    p = sub.add_parser(
        "grid",
        help="Record all trained body families using one policy in a synchronized 4K grid",
    )
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "assets/locomotion/checkpoints/paired_selected_545s_seed2.pt",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=ROOT / "previews/locomotion/all_trained_cases.mp4",
    )
    p.add_argument("--seconds", type=float, default=12)
    p.add_argument("--seed", type=int, default=9137)
    p.add_argument("--family", choices=("partial", "limb_loss"), default="partial")
    p.add_argument("--presentation", choices=("original", "damage"), default="original")
    p.add_argument(
        "--label", help="Override the grid heading for clearly labeled experiments"
    )
    p.add_argument(
        "--replay-from",
        type=Path,
        help="Reuse this saved capture directory; refuse new rollouts",
    )
    p = sub.add_parser("walk", help="View the dedicated healthy-only walking policy")
    p.add_argument("--seconds", type=float, default=0)
    p.add_argument(
        "--style", choices=("balanced", "longer", "compact"), default="balanced"
    )
    p = sub.add_parser("compare")
    p.add_argument("--left", type=Path, required=True)
    p.add_argument("--right", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--case", default="short_steps")
    p.add_argument("--seconds", type=float, default=12)
    args = parser.parse_args()
    if args.command == "preview":
        print(preview(args.output))
    elif args.command == "train":
        from .train import train

        kwargs = vars(args)
        kwargs.pop("command")
        train(**kwargs)
    elif args.command == "evaluate":
        from .evaluate import evaluate

        kwargs = vars(args)
        kwargs.pop("command")
        evaluate(**kwargs)
    elif args.command == "record":
        from .record import record

        kwargs = vars(args)
        kwargs.pop("command")
        record(**kwargs)
    elif args.command == "view":
        from .record import view

        kwargs = vars(args)
        kwargs.pop("command")
        view(**kwargs)
    elif args.command == "grid":
        from .grid import grid

        kwargs = vars(args)
        kwargs.pop("command")
        grid(**kwargs)
    elif args.command == "walk":
        from .record import view

        view(
            ROOT
            / "assets/locomotion/checkpoints"
            / {
                "balanced": "healthy_balanced_405s_seed2.pt",
                "longer": "healthy_stride_300s_seed2.pt",
                "compact": "healthy_feet_210s_seed2.pt",
            }[args.style],
            case="healthy",
            seconds=args.seconds,
        )
    elif args.command == "compare":
        from .record import compare

        kwargs = vars(args)
        kwargs.pop("command")
        compare(**kwargs)


if __name__ == "__main__":
    main()
