"""Render saved benchmark evidence without advancing the simulation.

Run with the existing wind extra for matplotlib; no learning stack is executed.
"""

import argparse
import hashlib
import json
from pathlib import Path
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np
from PIL import Image, ImageDraw

from sixlegs.neural_insertion.rescaling import CableConfig, load_cable
from sixlegs.neural_insertion.units import Units
from sixlegs.neural_insertion.visuals import font


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    started = time.perf_counter()
    args.output.mkdir(parents=True, exist_ok=True)
    report = json.loads((args.run / "manifest.json").read_text())
    traces = {n: np.load(args.run / (n + ".npz")) for n in
              ("bend_16", "bend_32", "bend_64", "contact_coarse", "contact_fine")}
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), layout="constrained")
    colors = ["#7994a9", "#328bae", "#108275"]
    for n, color in zip((16, 32, 64), colors):
        v = traces[f"bend_{n}"]["vertices"][-1]
        axes[0, 0].plot(v[:, 0]*1e3, -v[:, 2]*1e6, label=f"{n} segments", color=color)
    c = CableConfig(**report["cases"]["bend_64"]["config"])
    x = np.linspace(0, c.length_m, 100)
    ref = 1e-8*x*x*(3*c.length_m-x)/(6*c.ei)
    axes[0, 0].plot(x*1e3, ref*1e6, "k--", label="Small-deflection beam")
    axes[0, 0].set(title="Loaded bending: 10 nN after 2 s", xlabel="Axial position (mm)",
                   ylabel="Downward deflection (µm)")
    axes[0, 0].legend()
    reference = report["cantilever_reference_m"]
    errors = [abs(abs(traces[f"bend_{n}"]["tip"][-1, 2]) - reference)/reference*100 for n in (16, 32, 64)]
    axes[0, 1].plot([16, 32, 64], errors, "o-", color="#108275")
    axes[0, 1].axhline(3, color="#c75c21", ls="--", label="3% preliminary gate")
    axes[0, 1].set(title="Spatial refinement of bending", xlabel="Segments", ylabel="Analytical tip error (%)", xticks=[16,32,64])
    axes[0, 1].legend()
    a, b = traces["contact_coarse"], traces["contact_fine"]
    for trace, dt, color in ((a, .15625, "#328bae"), (b, .078125, "#c75c21")):
        axes[1, 0].plot(trace["time"]*1e3, (trace["tip"][:, 2]+.006)*1e6,
                        label=f"dt = {dt:g} µs", color=color)
    axes[1, 0].axhline(20, color="black", ls=":", label="Radius above support")
    axes[1, 0].set(title="Frictional drop: rebound detail", xlabel="Simulation time (ms)",
                   ylabel="Tip center above support (µm)", xlim=(45, 60), ylim=(0, 100))
    axes[1, 0].legend()
    err = np.linalg.norm(a["tip"]-b["tip"], axis=-1)*1e6
    axes[1, 1].plot(a["time"]*1e3, err, color="#c75c21")
    axes[1, 1].axhline(1, color="black", ls="--", label="1 µm acceptance limit")
    axes[1, 1].set(title=f"Timestep convergence FAIL: {err.max():.2f} µm", xlabel="Simulation time (ms)",
                   ylabel="Matched tip separation (µm)")
    axes[1, 1].legend()
    for ax in axes.flat:
        ax.grid(alpha=.15)
    fig.suptitle("40 µm thread • rescaling study • contact model not accepted", fontsize=17)
    fig.savefig(args.output / "quality_review.png", dpi=150)
    fig.savefig(args.output / "quality_review.svg")
    plt.close(fig)

    # Replay one logged state only. qpos/qvel retain the recorded engine units.
    case = report["cases"]["contact_fine"]
    model, data = load_cable(CableConfig(**case["config"]), Units(**case["units"]))
    idx = int(err.argmax())
    data.qpos[:] = b["qpos"][idx]
    data.qvel[:] = b["qvel"][idx]
    data.time = b["time"][idx]
    mujoco.mj_forward(model, data)
    with mujoco.Renderer(model, height=750, width=1600) as renderer:
        renderer.update_scene(data, camera="cable")
        raw = Image.fromarray(renderer.render())
    frame = Image.new("RGB", (1600, 900), "#101c25")
    frame.paste(raw, (0, 85))
    draw = ImageDraw.Draw(frame)
    draw.text((25,20), "FLEXIBLE THREAD / CONTACT DIAGNOSTIC", font=font(28), fill="#eff6f7")
    draw.text((25,855), f"Recorded physics at {data.time*1e3:.1f} ms | 44 mm × 40 µm | contact convergence FAILED", font=font(21), fill="#ffc580")
    frame.save(args.output / "contact_diagnostic.png")
    images = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
              for p in args.output.iterdir() if p.suffix in (".png", ".svg")}
    (args.output / "review_manifest.json").write_text(json.dumps({
        "input_manifest_sha256": hashlib.sha256((args.run / "manifest.json").read_bytes()).hexdigest(),
        "replay_timestamp_s": float(data.time), "simulation_advanced_s": 0,
        "render_and_plot_wall_s": time.perf_counter()-started, "files_sha256": images,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
