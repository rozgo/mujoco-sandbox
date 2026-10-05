"""Plot recorded DER gates; never advance simulation while preparing figures."""

import argparse
import hashlib
import json
from pathlib import Path
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--refinement", type=Path, help="Additional halved-step run")
    args = parser.parse_args()
    start = time.perf_counter()
    report = json.loads((args.run/"manifest.json").read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), layout="constrained")
    fig.suptitle("MuJoCo DER: measured numerical acceptance", fontsize=20, weight="bold")
    teal, red, gray = "#008b85", "#bd4257", "#667684"

    ax = axes[0, 0]
    names = ["bend", "twist", "mixed", "spatial_seed_7", "spatial_seed_17", "spatial_seed_29"]
    x = np.arange(len(names))
    for shift, variant, color in [(-.19, "published", gray), (.19, "direct", teal)]:
        values = [max(report["audit"][variant]["virtual_work"][n]["relative_error"]) for n in names]
        ax.bar(x+shift, np.maximum(values, 1e-12), width=.36, label=variant, color=color)
    ax.axhline(1e-4, color=red, linestyle="--", label="acceptance limit")
    ax.set_yscale("log")
    ax.set_xticks(x, ["Bend", "Twist", "Mixed", "3D / 7", "3D / 17", "3D / 29"])
    ax.set_ylabel("Force versus energy-gradient relative error")
    ax.set_title("Force consistency: corrected projection passes")
    ax.legend(fontsize=9)

    ax = axes[0, 1]
    counts = [16, 32, 64]
    measured = []
    for n in counts:
        t = np.load(args.run/f"bend_{n}.npz")
        measured.append(abs(t["tip"][-1, 2])*1e6)
    ax.plot(counts, measured, "o-", color=teal, label="DER measurement")
    reference = report["cantilever_analytical_m"]*1e6
    ax.axhline(reference, color=gray, linestyle="--", label="small-deflection reference")
    ax.set_xlabel("Segments")
    ax.set_ylabel("Tip deflection (µm)")
    ax.set_title("Cantilever: preliminary 3% gate")
    ax.legend(fontsize=9)

    ax = axes[1, 0]
    for name, color in [("relax_coarse", gray), ("relax_fine", teal)]:
        t = np.load(args.run/(name+".npz"))
        energy = t["energy_j"].sum(axis=1)
        dt = report["cases"][name]["config"]["dt_s"]*1e6
        ax.plot(t["time"]*1000, (energy-energy[0])/energy[0]*100, color=color, label=f"dt = {dt:g} µs")
    ax.axhline(.1, color=red, linestyle="--")
    ax.axhline(-.1, color=red, linestyle="--", label="±0.1% gate")
    ax.set_xlabel("Simulation time (ms)")
    ax.set_ylabel("Mechanical energy change (%)")
    ax.set_title("Free relaxation: no load, gravity or damping")
    ax.legend(fontsize=9)

    ax = axes[1, 1]
    for a, b, color in [("contact_medium", "contact_coarse", gray), ("contact_coarse", "contact_fine", teal)]:
        ta = np.load(args.run/(a+".npz"))
        tb = np.load(args.run/(b+".npz"))
        error = np.linalg.norm(ta["vertices"]-tb["vertices"], axis=-1).max(axis=1)*1e6
        dt_a = report["cases"][a]["config"]["dt_s"]*1e9
        dt_b = report["cases"][b]["config"]["dt_s"]*1e9
        ax.plot(ta["time"]*1000, error, color=color, label=f"{dt_a:g} → {dt_b:g} ns")
    if args.refinement:
        refined = json.loads((args.refinement/"manifest.json").read_text())
        ta = np.load(args.run/"contact_fine.npz")
        tb = np.load(args.refinement/"contact_fine.npz")
        np.testing.assert_allclose(ta["time"], tb["time"], rtol=0, atol=1e-9)
        error = np.linalg.norm(ta["vertices"]-tb["vertices"], axis=-1).max(axis=1)*1e6
        dt_a = report["cases"]["contact_fine"]["config"]["dt_s"]*1e9
        dt_b = refined["cases"]["contact_fine"]["config"]["dt_s"]*1e9
        ax.plot(ta["time"]*1000, error, color="#b46514", label=f"{dt_a:g} → {dt_b:g} ns")
    ax.axhline(1., color=red, linestyle="--", label="1 µm gate")
    ax.set_xlabel("Simulation time (ms)")
    ax.set_ylabel("Maximum centerline difference (µm)")
    ax.set_title("Contact/rebound: matched-time refinement")
    ax.legend(fontsize=9)
    for ax in axes.flat:
        ax.grid(alpha=.16)
        ax.spines[["top", "right"]].set_visible(False)
    note = ("40 µm diameter • illustrative material • CPU MuJoCo 3.12.0 • "
            "quasistatic twist • physical calibration and tool release unvalidated")
    fig.supxlabel(note, fontsize=10)
    for suffix in ("png", "svg"):
        fig.savefig(args.output/("quality_review."+suffix), dpi=160)
    plt.close(fig)
    receipt = {"run_manifest_sha256": hashlib.sha256((args.run/"manifest.json").read_bytes()).hexdigest(),
               "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "additional_simulation_s": 0, "plot_wall_s": time.perf_counter()-start,
               "artifacts": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                             for p in args.output.glob("quality_review.*")}}
    if args.refinement:
        receipt["refinement_manifest_sha256"] = hashlib.sha256((args.refinement/"manifest.json").read_bytes()).hexdigest()
    (args.output/"review_manifest.json").write_text(json.dumps(receipt, indent=2)+"\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
