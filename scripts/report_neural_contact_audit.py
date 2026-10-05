"""Review recorded contact-audit results and replay frozen divergence events."""

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

from sixlegs.neural_insertion.contact_audit import evaluate_same_state, checked_trace
from sixlegs.neural_insertion.contact_diagnostics import load_fixture, UNIT_SYSTEMS


def event_replay(pair):
    manifest = json.loads((pair/"manifest.json").read_text())
    result = {}
    for unit in UNIT_SYSTEMS[1:]:
        p = pair/(unit.name+".npz")
        assert hashlib.sha256(p.read_bytes()).hexdigest() == manifest["cases"][unit.name]["trajectory_sha256"]
        with np.load(p) as saved:
            rows = []
            for side in (0, 1):
                state = {k: saved['contact_set_differs__'+k+'_'+str(side)] for k in ('qpos', 'qvel', 'plugin_state')}
                audit = evaluate_same_state('der', state)
                c, m, d, _ = load_fixture('der', 7.8125e-8)
                for k, v in state.items():
                    getattr(d, k)[:] = v
                mujoco.mj_forward(m, d)
                gaps = {}
                for geom in (17, 18):
                    zaxis = d.geom_xmat[geom].reshape(3, 3)[2, 2]
                    gap = (d.geom_xpos[geom, 2]-abs(zaxis)*m.geom_size[geom, 1]-m.geom_size[geom, 0])/1000-c.floor_z_m
                    gaps[m.geom(geom).name] = float(gap)
                rows.append({'state_from_pair_side': side, 'surface_gaps_m': gaps,
                             'same_state_unit_comparison': audit})
        result[unit.name] = rows
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("audit", "pair", "sensitivity", "baseline", "refinement", "output"):
        p.add_argument(name, type=Path)
    args = p.parse_args()
    started = time.perf_counter()
    args.output.mkdir(parents=True, exist_ok=True)
    reports = {name: json.loads((getattr(args, name)/"manifest.json").read_text())
               for name in ("audit", "pair", "sensitivity", "baseline", "refinement")}
    if any(r["status"] == "running" for r in reports.values()):
        raise ValueError("Wait for all runs to finish")
    replay = event_replay(args.pair)
    (args.output/"event_replay.json").write_text(json.dumps(replay, indent=2)+"\n")
    refpath = args.audit/"normal_reference.npz"
    assert hashlib.sha256(refpath.read_bytes()).hexdigest() == reports["audit"]["independent_reference"]["trajectory_sha256"]
    with np.load(refpath) as data:
        times, reference = data["time"], data["tighter"]
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), layout="constrained")
    fig.suptitle("Contact audit: force consistency, impact timing and sensitivity", fontsize=19, weight="bold")
    axes[0, 0].plot(times*1000, reference[:, 0]*1e6, color="black", linewidth=2, label="Independent continuous-time reference")
    sources = [("baseline", "segment_normal_mm_g_s_coarse"), ("baseline", "segment_normal_mm_g_s_fine")]
    sources += [("refinement", name) for name in reports["refinement"]["cases"]]
    for source, name in sources:
        trace = checked_trace(getattr(args, source), reports[source], name)
        gap = trace["qpos"][:, 2]/1000-(-.006+20e-6)
        dt = reports[source]["cases"][name]["config"]["dt_s"]*1e9
        axes[0, 0].plot(times*1000, gap*1e6, label=f"MuJoCo / {dt:g} ns", alpha=.8)
        axes[0, 1].plot(times*1000, abs(gap-reference[:, 0])*1e6, label=f"{dt:g} ns")
    axes[0, 0].set(xlim=(34.9, 46), ylim=(-5, 130), xlabel="Drop time (ms)", ylabel="Surface gap (µm)", title="Rigid normal drop: independent reference")
    axes[0, 1].set(xlim=(34.9, 47), xlabel="Drop time (ms)", ylabel="Position disagreement (µm)", title="Smaller timestep does not improve every impact phase")
    axes[0, 1].axhline(1, color="#a53751", linestyle=":", label="1 µm gate")
    for name, label, color in [("mm_ug_s", "Change mass units", "#008c89"), ("tenth_mm_g_s", "Change length units", "#ae5e14")]:
        with np.load(args.pair/(name+".npz")) as saved:
            trace = saved["telemetry"]
        axes[1, 0].plot(trace[:, 0]*1000, np.maximum(trace[:, 1], 1e-12), color=color, label=label)
        event = reports["pair"]["cases"][name]["events"]["contact_set_differs"]["time_s"]
        axes[1, 0].axvline(event*1000, color=color, linestyle="--", alpha=.7)
    axes[1, 0].axhline(.01, color="#a53751", linestyle=":", label="0.01 µm unit gate")
    axes[1, 0].set(yscale="log", ylim=(1e-12, 1), xlabel="Time after pre-impact checkpoint (ms)",
                   ylabel="Maximum centerline disagreement (µm)", title="DER: dashed lines mark first differing contact sets")
    baseline = checked_trace(args.baseline, reports["baseline"], "der_mm_g_s_fine")
    for name, case in reports["sensitivity"]["cases"].items():
        trace = checked_trace(args.sensitivity, reports["sensitivity"], name)
        error = np.linalg.norm(trace["vertices"]-baseline["vertices"], axis=-1).max(axis=1)*1e6
        label = (f"1e−12 rad / seed {case['seed']}" if case["seed"] is not None else "Unperturbed / tolerance = 0")
        axes[1, 1].plot(trace["time"]*1000, np.maximum(error, 1e-12), label=label)
    axes[1, 1].set(yscale="log", xlabel="Time after pre-impact checkpoint (ms)", ylabel="Maximum centerline disagreement (µm)",
                   title="DER: same units, tiny initial changes or solver tolerance")
    for ax in axes.flat:
        ax.grid(alpha=.16)
        ax.legend(fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
    fig.supxlabel("CPU MuJoCo 3.12.0 • physical model unchanged • diagnostic perturbations are labeled • accuracy remains unaccepted", fontsize=10)
    for suffix in ("png", "svg"):
        fig.savefig(args.output/("quality_review."+suffix), dpi=150)
    plt.close(fig)
    receipt = {"input_manifest_sha256": {name: hashlib.sha256((getattr(args, name)/"manifest.json").read_bytes()).hexdigest() for name in reports},
               "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "plot_and_scratch_evaluation_wall_s": time.perf_counter()-started, "additional_simulation_s": 0,
               "artifacts": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in args.output.iterdir()
                             if p.name in ("quality_review.png", "quality_review.svg", "event_replay.json")}}
    (args.output/"review_manifest.json").write_text(json.dumps(receipt, indent=2)+"\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
