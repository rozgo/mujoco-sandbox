"""Learning curve and held-out physics-residual evidence, without policy claims."""

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def render(args):
    if args.output.exists():
        raise FileExistsError("Preserve previous result figures")
    training = json.loads((args.training / "report.json").read_text())
    evaluation = json.loads((args.evaluation / "report.json").read_text())
    previous = json.loads(args.previous.read_text())
    bg, fg, yellow, green, gray, red = (
        "#1F1F1F",
        "#E6E1DB",
        "#FFC31F",
        "#73AC87",
        "#A7A4A0",
        "#BC6657",
    )
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "text.color": fg,
            "axes.labelcolor": fg,
            "xtick.color": fg,
            "ytick.color": fg,
            "axes.edgecolor": gray,
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.6), facecolor=bg)
    for ax in axes:
        ax.set_facecolor(bg)
        ax.grid(alpha=0.15)
        ax.spines[["top", "right"]].set_visible(False)
    rows = training["stages"]["prober"]["records"]
    axes[0].plot(
        [r["training_seconds"] for r in rows],
        [r["validation"]["selection"] for r in rows],
        "o-",
        color=yellow,
        label="Integrated residual probe",
    )
    axes[0].axhline(
        rows[0]["validation"]["selection"],
        color=green,
        ls="--",
        label="Zero residual / physics prior",
    )
    axes[0].set(
        xlabel="Probe optimizer time (s)",
        ylabel="Validation velocity error (mm/s)",
        title="Does learning improve the physical prior?",
    )
    horizons = evaluation["forecasts"]["aggregate"]
    for name, color, label in (
        ("jepa", yellow, "Physics + learned residual"),
        ("analytical", green, "Analytical predictor"),
        ("constant_velocity", gray, "Constant velocity"),
    ):
        axes[1].plot(
            [float(h) * 1000 for h in horizons],
            [v[name]["velocity_mm_s"] for v in horizons.values()],
            "o-",
            color=color,
            label=label,
        )
    axes[1].plot(
        [float(h) * 1000 for h in horizons],
        [previous["forecasts"]["aggregate"][h]["jepa"]["velocity_mm_s"] for h in horizons],
        "x--",
        color=red,
        label="Previous direct-state probe",
    )
    axes[1].set(
        xlabel="Prediction horizon (ms)",
        ylabel="Held-out velocity error (mm/s)",
        title=f"{evaluation['forecasts']['windows']} held-out flight windows",
    )
    actual, prediction, analytic = [], [], []
    for case in evaluation["interventions"]["cases"]:
        with np.load(args.evaluation / f"{case['name']}.npz") as z:
            mask = z["valid"][1:15, -1] & z["valid"][0, -1]
            for key, destination in (
                ("truth", actual),
                ("predicted", prediction),
                ("analytical", analytic),
            ):
                x = z[key][:, -1, 3:6]
                destination.extend(((x[1:15] - x[0])[mask] * 10).ravel())
    axes[2].scatter(
        actual, analytic, color=green, s=16, marker="x", alpha=0.5, label="Analytical"
    )
    axes[2].scatter(actual, prediction, color=yellow, s=9, alpha=0.4, label="Learned residual")
    extent = max(np.max(np.abs(x)) for x in (actual, prediction, analytic)) * 1.05
    axes[2].plot(
        [-extent, extent], [-extent, extent], "--", color=gray, label="Perfect prediction"
    )
    axes[2].set(
        xlabel="Actual change in velocity (mm/s)",
        ylabel="Predicted change in velocity (mm/s)",
        title="504 command changes · 200 ms · XYZ",
        xlim=(-extent, extent),
        ylim=(-extent, extent),
    )
    for ax in axes:
        ax.legend(facecolor=bg, edgecolor=gray, labelcolor=fg, fontsize=8)
    elapsed = sum(s["training_seconds"] for s in training["stages"].values())
    fig.suptitle(
        "FLY WORLD MODEL / LEARNED DYNAMICS RESIDUALS", fontsize=16, fontweight="bold", y=0.98
    )
    fig.text(
        0.03,
        0.02,
        f"{elapsed:.1f} s GPU optimization · 1 kHz physical integration · 500 Hz commands · Same test cases · No policy update",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.93))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160, facecolor=bg)
    plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--training", type=Path, required=True)
    p.add_argument("--evaluation", type=Path, required=True)
    p.add_argument(
        "--previous",
        type=Path,
        default=Path("docs/embodied_fly/world_model/pilot_01/evaluation.json"),
    )
    p.add_argument("--output", type=Path, required=True)
    render(p.parse_args())
