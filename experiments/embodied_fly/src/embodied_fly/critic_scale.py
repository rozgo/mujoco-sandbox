"""Replay measured-return fits with training-only input statistics; no actor."""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn

from embodied_fly.exploration_value import fit_capacity, quality, window_returns
from embodied_fly.provenance import evidence, sha256, utc_now


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    source = json.loads((args.capture / "report.json").read_text())
    for file, key in (
        ("capture.npz", "capture_sha256"),
        ("critic_anchors.npz", "critic_anchors_sha256"),
    ):
        assert sha256(args.capture / file) == source[key]
    anchors = np.load(args.capture / "critic_anchors.npz")
    capture = np.load(args.capture / "capture.npz")
    returns = window_returns(capture["reward"], math.exp(-0.002 / 5), 2000)
    valid = anchors["index"] < len(returns)
    x = torch.from_numpy(anchors["features"][valid].reshape(-1, anchors["features"].shape[-1]))
    y = torch.from_numpy(returns[anchors["index"][valid]].reshape(-1))
    train = np.tile(np.arange(64) % 16 < 12, valid.sum())
    test = (~train) & np.tile(np.arange(64) >= 16, valid.sum())
    safe_test = test & np.tile(np.arange(64) < 48, valid.sum())
    torch.set_num_threads(4)
    torch.manual_seed(120502)
    network = nn.Sequential(
        nn.Linear(x.shape[-1], 128),
        nn.Tanh(),
        nn.Linear(128, 128),
        nn.Tanh(),
        nn.Linear(128, 1),
    )
    mean = x[train].mean(0)
    scale = x[train].std(0, unbiased=False).clamp_min(0.05)
    fits = {}
    for standardize in (False, True):
        features = ((x - mean) / scale).clamp(-10, 10) if standardize else x
        for normalize_targets in (False, True):
            name = f"inputs_{'scaled' if standardize else 'original'}_targets_{'scaled' if normalize_targets else 'original'}"
            fitted, report = fit_capacity(
                network,
                features,
                y,
                train,
                test,
                steps=1000,
                seed=120503,
                normalized=normalize_targets,
            )
            with torch.no_grad():
                prediction = fitted(features).squeeze(-1)
            report["safe_noise_test"] = quality(prediction[safe_test], y[safe_test])
            fits[name] = report
    report = {
        "provenance": evidence(),
        "completed_utc": utc_now(),
        "source_report_sha256": sha256(args.capture / "report.json"),
        "scope": "Frozen actor traces; diagnostic four-second return fits only. Complete random streams held out. Same starting physical state. No actor/PPO updates. Input statistics use training streams only.",
        "input_dimension": x.shape[-1],
        "input_scale_floor": 0.05,
        "input_clip": 10,
        "fits": fits,
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
