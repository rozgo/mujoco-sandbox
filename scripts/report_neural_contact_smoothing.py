"""Plot the recorded contact-smoothing experiment without advancing physics."""

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
from sixlegs.neural_insertion.contact_smoothing import BASELINE, CANDIDATE, CONTROLS, DTS, SEEDS, case_name

UNIT_LABELS = {"mm_g_s": "mm–g–s", "mm_ug_s": "mm–µg–s", "tenth_mm_g_s": "0.1 mm–g–s"}
COMBO_LABELS = {BASELINE: "baseline: flat law + implicitfast", CONTROLS[0]: "control: flat law + RK4",
                CONTROLS[1]: "control: ramped law + implicitfast", CANDIDATE: "candidate: ramped law + RK4"}
COLORS = {BASELINE: "#a53751", CONTROLS[0]: "#ae5e14", CONTROLS[1]: "#6a6a9f", CANDIDATE: "#008c89"}


def series(a, b):
    error = np.linalg.norm(a["vertices"]-b["vertices"], axis=-1).max(axis=1)*1e6
    return a["time"]*1000, np.maximum(error, 1e-12)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--isolation", type=Path, help="Hash-checked isolation run for baseline unit pairs")
    parser.add_argument("--sensitivity", type=Path, help="Hash-checked audit sensitivity run for baseline seeds")
    args = parser.parse_args()
    started = time.perf_counter()
    args.output.mkdir(parents=True, exist_ok=True)
    report = json.loads((args.run/"manifest.json").read_text())
    if report["status"] == "running":
        raise ValueError("Run is still active")
    traces = {name: checked_trace(args.run, report, name) for name in report["cases"]}
    kinds = list(dict.fromkeys(c["kind"] for c in report["cases"].values()))
    chains = [k for k in ("chain_no_elastic", "der", "der_settle") if k in kinds]
    rows = 1+len(chains)
    fig, axes = plt.subplots(rows, 2, figsize=(15, 4.6*rows), layout="constrained", squeeze=False)
    fig.suptitle("Contact smoothing: continuous-entry impedance ramp and RK4 on the recorded fixtures", fontsize=18, weight="bold")

    # Row 0: rigid normal drop against the independent continuous reference; rigid gate summary.
    ax = axes[0, 0]
    reference = report.get("independent_reference", {})
    labels, values, colors = [], [], []
    for law_name, entry in reference.items():
        for name, c in entry["comparisons"].items():
            combo = (law_name, c["integrator"])
            labels.append(f"{law_name} / {c['integrator']} / {c['timestep_s']*1e9:g} ns")
            values.append(c["mujoco_vs_continuous_um"])
            colors.append(COLORS.get(combo, "#777777"))
    if values:
        ax.barh(range(len(values)), values, color=colors)
        ax.set_yticks(range(len(values)), labels, fontsize=8)
        ax.set(xscale="log", xlabel="MuJoCo position error versus independent continuous reference (µm)",
               title="Rigid normal drop: error against the independent reference")
        ax.axvline(1, color="#a53751", linestyle=":", label="1 µm gate")
        ax.invert_yaxis()
    ax = axes[0, 1]
    labels, values, colors = [], [], []
    for key, comp in report["comparisons"].items():
        if key.endswith("_timestep") and key.startswith("segment_"):
            case = report["cases"][comp["cases"][0]]
            law, integrator, units = case["law"]["name"], case["integrator"], case["units"]["name"]
            labels.append(f"{case['kind'].replace('segment_', '')} / {law} / {integrator} / {UNIT_LABELS[units]}")
            values.append(max(comp["difference_um"], 1e-6))
            colors.append(COLORS.get((law, integrator), "#777777"))
    ax.barh(range(len(values)), values, color=colors)
    ax.set_yticks(range(len(values)), labels, fontsize=6.5)
    ax.axvline(1, color="#a53751", linestyle=":", label="1 µm gate")
    ax.set(xscale="log", xlabel="Timestep difference 156.25 → 78.125 ns (µm)", title="Rigid fixtures: timestep gate by law and integrator")
    ax.invert_yaxis()

    for row, kind in enumerate(chains, start=1):
        ax = axes[row, 0]
        for combo in (BASELINE, *CONTROLS, CANDIDATE):
            names = [case_name(kind, *combo, level, "mm_g_s") for level, _ in DTS]
            if all(n in traces for n in names):
                t, e = series(traces[names[0]], traces[names[1]])
                ax.plot(t, e, color=COLORS[combo], label=COMBO_LABELS[combo], linewidth=1.6)
        extra = [c for c in report["cases"] if c.startswith(f"{kind}_{CANDIDATE[0]}_{CANDIDATE[1]}_mm_g_s_finer")]
        if extra:
            t, e = series(traces[case_name(kind, *CANDIDATE, "fine", "mm_g_s")], traces[extra[0]])
            ax.plot(t, e, color=COLORS[CANDIDATE], linestyle="--", label="candidate: 78.125 → 39.0625 ns", linewidth=1.4)
        ax.axhline(1, color="#a53751", linestyle=":", label="1 µm gate")
        xlabel = "Time after release (ms)" if kind == "der_settle" else "Time after pre-impact checkpoint (ms)"
        ax.set(yscale="log", ylim=(1e-12, 1e3), xlabel=xlabel, ylabel="Max centerline disagreement (µm)",
               title=f"{kind}: coarse-versus-fine timestep disagreement")
        ax = axes[row, 1]
        for alternative, color in (("mm_ug_s", "#008c89"), ("tenth_mm_g_s", "#ae5e14")):
            for level, style in (("coarse", ":"), ("fine", "-")):
                a, b = [case_name(kind, *CANDIDATE, level, u) for u in ("mm_g_s", alternative)]
                if a in traces and b in traces:
                    t, e = series(traces[a], traces[b])
                    ax.plot(t, e, color=color, linestyle=style, linewidth=1.4,
                            label=f"candidate: mm–g–s vs {UNIT_LABELS[alternative]} ({level})")
        if args.isolation and kind in ("chain_no_elastic", "der"):
            iso = json.loads((args.isolation/"manifest.json").read_text())
            base = checked_trace(args.isolation, iso, f"{kind}_mm_g_s_fine")
            for alternative, color in (("mm_ug_s", "#008c89"), ("tenth_mm_g_s", "#ae5e14")):
                t, e = series(base, checked_trace(args.isolation, iso, f"{kind}_{alternative}_fine"))
                ax.plot(t, e, color=color, linestyle="-", alpha=.35, linewidth=3,
                        label=f"baseline: mm–g–s vs {UNIT_LABELS[alternative]} (fine)")
        if kind == "der":
            fine = traces[case_name("der", *CANDIDATE, "fine", "mm_g_s")]
            for seed in SEEDS:
                name = case_name("der", *CANDIDATE, "fine", "mm_g_s", seed)
                if name in traces:
                    t, e = series(traces[name], fine)
                    ax.plot(t, e, color="#444444", linewidth=.9, alpha=.8, label=f"candidate: 1e−12 rad seed {seed}" if seed == SEEDS[0] else None)
            if args.sensitivity:
                sens = json.loads((args.sensitivity/"manifest.json").read_text())
                if args.isolation:
                    for seed in SEEDS:
                        t, e = series(checked_trace(args.sensitivity, sens, f"perturb_seed_{seed}"), base)
                        ax.plot(t, e, color="#444444", linewidth=2.2, alpha=.3, label="baseline: 1e−12 rad seeds" if seed == SEEDS[0] else None)
        ax.axhline(.01, color="#a53751", linestyle=":", label="0.01 µm unit gate")
        ax.axhline(1, color="#a53751", linestyle="--", alpha=.5, label="1 µm")
        ax.set(yscale="log", ylim=(1e-12, 1e3), xlabel=xlabel, ylabel="Max centerline disagreement (µm)",
               title=f"{kind}: unit systems" + (" and 1e−12 rad perturbations" if kind == "der" else ""))
    for ax in axes.flat:
        ax.grid(alpha=.16)
        ax.legend(fontsize=7)
        ax.spines[["top", "right"]].set_visible(False)
    fig.supxlabel("CPU MuJoCo 3.12.0 • same fixtures, masses, dimensions, friction and checkpoint • only solimp/solref and integrator vary • accuracy remains unaccepted for the robot", fontsize=9)
    for suffix in ("png", "svg"):
        fig.savefig(args.output/("quality_review."+suffix), dpi=150)
    plt.close(fig)
    receipt = {"input_manifest_sha256": hashlib.sha256((args.run/"manifest.json").read_bytes()).hexdigest(),
               "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "plot_wall_s": time.perf_counter()-started, "additional_simulation_s": 0,
               "artifacts": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in args.output.iterdir()
                             if p.name in ("quality_review.png", "quality_review.svg")}}
    (args.output/"review_manifest.json").write_text(json.dumps(receipt, indent=2)+"\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
