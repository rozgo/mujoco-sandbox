"""Plot the recorded thread-clock and task-fixture studies without advancing physics."""

import argparse
import hashlib
import json
from pathlib import Path
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sixlegs.neural_insertion.contact_audit import checked_trace


def series(a, b):
    return a["time"]*1000, np.maximum(np.linalg.norm(a["vertices"]-b["vertices"], axis=-1).max(axis=1)*1e6, 1e-12)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("clock", type=Path)
    p.add_argument("fixtures", type=Path)
    p.add_argument("release", type=Path)
    p.add_argument("output", type=Path)
    args = p.parse_args()
    started = time.perf_counter()
    args.output.mkdir(parents=True, exist_ok=True)
    clock = json.loads((args.clock/"manifest.json").read_text())
    fixtures = json.loads((args.fixtures/"manifest.json").read_text())
    release = json.loads((args.release/"manifest.json").read_text())
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), layout="constrained")
    fig.suptitle("Task-regime thread physics: timestep, penetration and probe fixtures", fontsize=18, weight="bold")
    colors = {5e-6: "#008c89", 1e-5: "#6a6a9f", 2e-5: "#ae5e14", 4e-5: "#a53751"}
    ax = axes[0, 0]
    for material, style in (("illustrative_100mpa", "-"), ("polyimide_2p5gpa", "--")):
        for tau in clock["time_constants_s"]:
            rows = sorted((v["dt_s"], v["difference_um"]) for v in clock["comparisons"].values()
                          if not v.get("units") and v["material"] == material and v["integrator"] == "RK4"
                          and v["time_constant_s"] == tau)
            ax.plot([r[0]*1e6 for r in rows], [r[1] for r in rows], style, marker="o", color=colors[tau],
                    label=f"τ {tau*1e6:g} µs, {material.split('_')[0]}")
    ax.axhline(1, color="#a53751", linestyle=":", label="1 µm gate")
    ax.set(xscale="log", yscale="log", xlabel="Timestep (µs), compared with its half", ylabel="Settling disagreement (µm)",
           title="Settling fixture, RK4: timestep disagreement")
    ax = axes[0, 1]
    for material, marker in (("illustrative_100mpa", "o"), ("polyimide_2p5gpa", "s")):
        taus = clock["time_constants_s"]
        settled = [max(c["settled_penetration_um"] for c in clock["cases"].values() if c["material"] == material
                       and c["time_constant_s"] == t and c["integrator"] == "RK4") for t in taus]
        peak = [max(c["peak_penetration_um"] for c in clock["cases"].values() if c["material"] == material
                    and c["time_constant_s"] == t and c["integrator"] == "RK4") for t in taus]
        ax.plot([t*1e6 for t in taus], settled, marker=marker, color="#008c89", label=f"settled, {material.split('_')[0]}")
        ax.plot([t*1e6 for t in taus], peak, marker=marker, color="#ae5e14", label=f"peak, {material.split('_')[0]}")
    ax.axhline(.2, color="#008c89", linestyle=":", label="0.2 µm settled gate")
    ax.axhline(2, color="#ae5e14", linestyle=":", label="2 µm peak gate")
    ax.set(xscale="log", yscale="log", xlabel="Contact time constant τ (µs)", ylabel="Penetration (µm)",
           title="Penetration grows with τ; τ = 20 µs keeps both gates")
    ax = axes[1, 0]
    for name, manifest, folder, dt, color in (("drag", fixtures, args.fixtures, 1e-5, "#008c89"),
                                               ("press", fixtures, args.fixtures, 1e-5, "#6a6a9f"),
                                               ("release", fixtures, args.fixtures, 1e-5, "#a53751"),
                                               ("release", release, args.release, 5e-6, "#ae5e14")):
        for material, style in (("illustrative_100mpa", "-"), ("polyimide_2p5gpa", "--")):
            a = f"{name}_{material}_tau20us_dt{dt*1e9:g}ns_mm_g_s"
            b = f"{name}_{material}_tau20us_dt{dt*1e9/2:g}ns_mm_g_s"
            t, e = series(checked_trace(folder, manifest, a), checked_trace(folder, manifest, b))
            ax.plot(t, e, style, color=color, label=f"{name} {dt*1e6:g}→{dt*1e6/2:g} µs, {material.split('_')[0]}")
    ax.axhline(1, color="#a53751", linestyle=":", label="1 µm gate")
    ax.set(yscale="log", ylim=(1e-9, 10), xlabel="Time (ms)", ylabel="Max centerline disagreement (µm)",
           title="Probe fixtures: timestep disagreement over time")
    ax = axes[1, 1]
    for material, style in (("illustrative_100mpa", "-"), ("polyimide_2p5gpa", "--")):
        tr = checked_trace(args.fixtures, fixtures, f"press_{material}_tau20us_dt10000ns_mm_g_s")
        base = tr["support_n"][(tr["time"] > .002) & (tr["time"] < .005)].mean()
        ax.plot(tr["time"]*1000, -tr["command"]*1e6, style, color="#444444", label=f"applied, {material.split('_')[0]}")
        ax.plot(tr["time"]*1000, (tr["support_n"]-base)*1e6, style, color="#008c89", alpha=.8,
                label=f"support reaction increase, {material.split('_')[0]}")
    ax.set(xlabel="Time (ms)", ylabel="Force (µN)", title="Press: support reaction follows the applied load")
    for ax in axes.flat:
        ax.grid(alpha=.16)
        ax.legend(fontsize=7)
        ax.spines[["top", "right"]].set_visible(False)
    fig.supxlabel("CPU MuJoCo 3.12.0 • DER thread, ramped contact law • numerical gates only; materials are presets, not measurements", fontsize=9)
    for suffix in ("png", "svg"):
        fig.savefig(args.output/("quality_review."+suffix), dpi=150)
    plt.close(fig)
    receipt = {"inputs": {k: hashlib.sha256((getattr(args, k)/"manifest.json").read_bytes()).hexdigest()
                          for k in ("clock", "fixtures", "release")},
               "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "plot_wall_s": time.perf_counter()-started, "additional_simulation_s": 0,
               "artifacts": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in args.output.iterdir()
                             if p.suffix in (".png", ".svg")}}
    (args.output/"review_manifest.json").write_text(json.dumps(receipt, indent=2)+"\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
