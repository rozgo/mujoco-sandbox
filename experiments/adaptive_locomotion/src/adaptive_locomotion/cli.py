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
    p.add_argument("--seconds", type=float, default=12)
    p = sub.add_parser("view")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--case", default="short_fl")
    p.add_argument("--seconds", type=float, default=0)
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
    elif args.command == "compare":
        from .record import compare

        kwargs = vars(args)
        kwargs.pop("command")
        compare(**kwargs)


if __name__ == "__main__":
    main()
