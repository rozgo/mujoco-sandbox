"""Paired critic minibatch comparison on recorded measured future rewards."""

import argparse
import json
import math
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from embodied_fly.exploration_value import quality, window_returns
from embodied_fly.ppo import Critic
from embodied_fly.ppo_critic import fit_critic
from embodied_fly.provenance import evidence, sha256, utc_now


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    provenance = evidence()
    source = json.loads((args.capture / "report.json").read_text())
    for file, key in (
        ("capture.npz", "capture_sha256"),
        ("critic_anchors.npz", "critic_anchors_sha256"),
    ):
        assert sha256(args.capture / file) == source[key]
    anchors = np.load(args.capture / "critic_anchors.npz")
    capture = np.load(args.capture / "capture.npz")
    returns = window_returns(capture["reward"], math.exp(-0.002 / 5), 2000)
    # 120 complete anchors permit 15 equal eight-anchor minibatches per epoch.
    times = anchors["index"][:120]
    assert times[-1] < len(returns)
    torch.set_num_threads(4)
    device = torch.device(args.device)
    features = torch.as_tensor(anchors["features"][:120], device=device)
    target = torch.as_tensor(returns[times], device=device)
    brain_shape = SimpleNamespace(
        observation_size=399, descending_ids=range(features.shape[-1] - 399)
    )
    rows = []
    for cohort, end in (("safe_noise", 48), ("mixed_with_failures", 64)):
        world_ids = np.arange(16, end)
        training = world_ids[world_ids % 16 < 12]
        testing = world_ids[world_ids % 16 >= 12]
        x, y = features[:, training], target[:, training]
        xt, yt = features[:, testing], target[:, testing]
        for seed in (120701, 120702, 120703):
            for standardized in (False, True):
                for shuffled in (False, True):
                    torch.manual_seed(seed)
                    critic = Critic(brain_shape, standardized).to(device)
                    if standardized:
                        critic.network.calibrate(x[:82])
                    optimizer = torch.optim.Adam(critic.parameters(), lr=1e-4, eps=1e-5)
                    fit_start = time.perf_counter()
                    fit = fit_critic(
                        critic,
                        optimizer,
                        x,
                        y,
                        8,
                        64,
                        np.random.default_rng(seed + 100),
                        shuffle=shuffled,
                    )
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    elapsed = time.perf_counter() - fit_start
                    with torch.no_grad():
                        prediction = critic.network(xt).squeeze(-1)
                    test = quality(prediction.cpu().flatten(), yt.cpu().flatten())
                    # Remove the shared time profile to expose discrimination
                    # among worlds at the same time; this is evaluation only.
                    within = quality(
                        (prediction - prediction.mean(1, keepdim=True)).cpu().flatten(),
                        (yt - yt.mean(1, keepdim=True)).cpu().flatten(),
                    )
                    row = {
                        "cohort": cohort,
                        "seed": seed,
                        "standardized_inputs": standardized,
                        "shuffled_transitions": shuffled,
                        "fit_seconds": elapsed,
                        "training_worlds": training.tolist(),
                        "test_worlds": testing.tolist(),
                        "test": test,
                        "within_time_test": within,
                        "fit": {
                            k: v for k, v in fit.items() if k not in ("predictions", "losses")
                        },
                    }
                    rows.append(row)
                    print(json.dumps(row), flush=True)
    comparisons = []
    for cohort in ("safe_noise", "mixed_with_failures"):
        for standardized in (False, True):
            subset = [
                r
                for r in rows
                if r["cohort"] == cohort and r["standardized_inputs"] == standardized
            ]
            groups = {
                shuffle: [r for r in subset if r["shuffled_transitions"] == shuffle]
                for shuffle in (False, True)
            }
            stats = {
                str(k): {
                    metric: float(np.median([r["test"][metric] for r in v]))
                    for metric in ("rmse", "explained_variance")
                }
                for k, v in groups.items()
            }
            within = {
                str(k): float(np.median([r["within_time_test"]["rmse"] for r in v]))
                for k, v in groups.items()
            }
            a, b = stats["False"], stats["True"]
            passed = (
                b["explained_variance"] >= 0.5
                and (
                    b["explained_variance"] >= a["explained_variance"] + 0.1
                    or b["rmse"] <= 0.8 * a["rmse"]
                )
                and within["True"] <= within["False"] * 1.05
            )
            comparisons.append(
                {
                    "cohort": cohort,
                    "standardized_inputs": standardized,
                    "median_test_by_shuffled": stats,
                    "median_within_time_rmse_by_shuffled": within,
                    "declared_progress_gate": bool(passed),
                }
            )
    report = {
        "provenance": provenance,
        "completed_utc": utc_now(),
        "wall_seconds": time.perf_counter() - started,
        "scope": "Recorded 10s frozen actor captures; four-second measured discounted reward windows, no bootstrap. Train/test split by complete independent random streams, same nominal start. No new physics or actor training. First 120 anchors at 0..5.95s; standardized inputs use training anchors 0..4.05s only. Future rewards are labels only.",
        "source_capture_report_sha256": sha256(args.capture / "report.json"),
        "config": {
            "device": str(device),
            "epochs": 64,
            "sequence": 8,
            "updates_per_fit": 960,
            "learning_rate": 1e-4,
            "seeds": [120701, 120702, 120703],
            "target_scaling": False,
        },
        "comparisons": comparisons,
        "fits": rows,
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(comparisons, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    run(parser.parse_args())
