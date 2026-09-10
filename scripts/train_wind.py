"""Matched FNO/PINO training, usable unchanged on macOS or a CUDA host."""

import argparse
import json
import shutil
from pathlib import Path

import torch

from sixlegs.wind.operator import train


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("outputs/wind/training.npz"))
    parser.add_argument("--output", type=Path, default=Path("outputs/wind/model"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=64)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--modes", type=int, default=16)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    print("torch", torch.__version__, "CUDA", torch.cuda.is_available(), flush=True)
    info = {
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "epochs_each": args.epochs,
        "width": args.width,
        "modes": args.modes,
        "neuraloperator_commit": "00b7d86f8d74ff0af55da53eb585fe26df9c71f0",
        "hermitian_symmetry": True,
        "tf32": False,
    }
    common = {
        "epochs": args.epochs,
        "backend": args.device,
        "width": args.width,
        "modes": args.modes,
    }
    train(args.data, args.output / "physics", physics_training=True, **common)
    train(args.data, args.output / "data_only", physics_training=False, **common)
    shutil.copy2(args.output / "physics/pino.pt", args.output / "pino.pt")
    shutil.copy2(args.output / "data_only/fno.pt", args.output / "fno.pt")
    (args.output / "training_setup.json").write_text(json.dumps(info, indent=2) + "\n")


if __name__ == "__main__":
    main()
