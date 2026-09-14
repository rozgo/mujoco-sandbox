"""Static scientific comparison from saved prediction and intervention evidence."""

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def render(folder, output):
    if output.exists():
        raise FileExistsError("Preserve earlier comparison plots")
    report = json.loads((folder / "report.json").read_text())
    bg, fg = "#1F1F1F", "#E6E1DB"
    colors = {"jepa": "#FFC31F", "analytical": "#73AC87", "constant_velocity": "#A7A4A0"}
    labels = {
        "jepa": "JEPA pilot",
        "analytical": "Analytical predictor",
        "constant_velocity": "Constant velocity",
    }
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "text.color": fg,
            "axes.labelcolor": fg,
            "xtick.color": fg,
            "ytick.color": fg,
            "axes.edgecolor": "#77716B",
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4), facecolor=bg)
    for ax in axes:
        ax.set_facecolor(bg)
        ax.grid(alpha=0.15)
        ax.spines[["top", "right"]].set_visible(False)
    horizons = report["forecasts"]["aggregate"]
    for key, color in colors.items():
        axes[0].plot(
            [float(h) * 1000 for h in horizons],
            [v[key]["velocity_mm_s"] for v in horizons.values()],
            "o-",
            color=color,
            label=labels[key],
            linewidth=2,
        )
    axes[0].set(
        xlabel="Prediction horizon (ms)",
        ylabel="Velocity prediction error (mm/s)",
        title="336 held-out forecast windows",
    )
    axes[0].legend(facecolor=bg, edgecolor="#77716B", labelcolor=fg)
    actual, predicted, analytical = [], [], []
    for case in report["interventions"]["cases"]:
        with np.load(folder / f"{case['name']}.npz") as z:
            valid = z["valid"][1:15, -1] & z["valid"][0, -1]
            for key, dest in (
                ("truth", actual),
                ("predicted", predicted),
                ("analytical", analytical),
            ):
                x = z[key][:, -1, 3:6]
                dest.extend(((x[1:15] - x[0])[valid] * 10).ravel().tolist())
    actual, predicted, analytical = map(np.asarray, (actual, predicted, analytical))
    axes[1].scatter(
        actual,
        analytical,
        color=colors["analytical"],
        marker="x",
        s=18,
        alpha=0.65,
        label=labels["analytical"],
    )
    axes[1].scatter(
        actual, predicted, color=colors["jepa"], s=10, alpha=0.4, label=labels["jepa"]
    )
    extent = max(np.max(np.abs(x)) for x in (actual, predicted, analytical)) * 1.1
    axes[1].plot(
        [-extent, extent],
        [-extent, extent],
        "--",
        color="#BC6657",
        alpha=0.7,
        label="Perfect prediction",
    )
    axes[1].set(
        xlim=(-extent, extent),
        ylim=(-extent, extent),
        xlabel="Actual velocity change from correction (mm/s)",
        ylabel="Predicted velocity change (mm/s)",
        title="504 wing corrections · 200 ms · all XYZ axes",
    )
    axes[1].legend(facecolor=bg, edgecolor="#77716B", labelcolor=fg, fontsize=9)
    fig.suptitle(
        "FLY WORLD MODEL  /  FIRST PREDICTION PILOT", fontsize=16, fontweight="bold", y=0.99
    )
    fig.text(
        0.04,
        0.015,
        "Same MuJoCo plant  ·  1 kHz physics / 500 Hz commands  ·  No policy update",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.93))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160, facecolor=bg)
    plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--evaluation", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    render(args.evaluation, args.output)
