"""Plot recorded contact-isolation comparisons without advancing physics."""

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
    args = parser.parse_args()
    start = time.perf_counter()
    manifest_path = args.run/"manifest.json"
    report = json.loads(manifest_path.read_text())
    if report["status"] == "running":
        raise ValueError("Run is still active")
    traces = {}
    for name, case in report["cases"].items():
        path = args.run/(name+".npz")
        assert hashlib.sha256(path.read_bytes()).hexdigest() == case["trajectory_sha256"]
        with np.load(path) as trace:
            traces[name] = {k: trace[k] for k in ("vertices", "time")}
    kinds = list(dict.fromkeys(c["kind"] for c in report["cases"].values()))
    fig, axes = plt.subplots(len(kinds), 2, figsize=(15, 3.4*len(kinds)), layout="constrained", squeeze=False)
    fig.suptitle("Contact isolation: timestep and unit agreement", fontsize=21, weight="bold")
    unit_labels = {"mm_g_s": "mm–g–s", "mm_ug_s": "mm–µg–s", "tenth_mm_g_s": "0.1 mm–g–s"}
    kind_labels = {"segment_normal": "Rigid / normal drop", "segment_oblique": "Rigid / angled + sliding",
                   "chain_no_elastic": "Chain / no elastic forces", "der": "Chain / flexible DER"}
    for row, kind in enumerate(kinds):
        for col, category in enumerate(("timestep", "units")):
            ax = axes[row, col]
            selected = [(k, v) for k, v in report["comparisons"].items() if k.startswith(kind+"_") and k.endswith(category)]
            for key, comparison in selected:
                a, b = comparison["cases"]
                ta, tb = traces[a], traces[b]
                np.testing.assert_allclose(ta["time"], tb["time"], rtol=0, atol=1e-9)
                error = np.linalg.norm(ta["vertices"]-tb["vertices"], axis=-1).max(axis=1)*1e6
                assert np.isclose(error.max(), comparison["difference_um"], rtol=1e-12, atol=1e-12)
                case = report["cases"][b]
                label = unit_labels[case["units"]["name"]]
                if category == "units":
                    label += " / "+("156 ns" if b.endswith("coarse") else "78 ns")
                label += f" / max {error.max():.3g} µm"
                ax.plot(ta["time"]*1000, np.maximum(error, 1e-10), label=label,
                        linestyle="--" if b.endswith("coarse") else "-", linewidth=1.5)
            limit = 1 if category == "timestep" else .01
            ax.axhline(limit, color="#b22d4d", linestyle=":", label=f"{limit:g} µm gate")
            ax.set_yscale("log")
            ax.set_ylim(bottom=1e-10)
            ax.set_title(kind_labels[kind]+(" / halve timestep" if col == 0 else " / change internal units"), fontsize=12)
            ax.set_xlabel("Time since fixture initialization (ms)")
            ax.set_ylabel("Maximum centerline difference (µm)")
            ax.legend(fontsize=8, loc="lower right", framealpha=.93)
            ax.grid(alpha=.18)
            ax.spines[["top", "right"]].set_visible(False)
    fig.supxlabel("40 µm diameter • CPU MuJoCo 3.12.0 • SI measurements • chains start at the same pre-impact checkpoint\n"
                  "Numerical trajectory disagreement, not measured physical error; rigid controls are diagnostic only.", fontsize=10)
    args.output.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "svg"):
        fig.savefig(args.output/("quality_review."+suffix), dpi=150)
    plt.close(fig)
    receipt = {"run_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
               "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "additional_simulation_s": 0, "plot_wall_s": time.perf_counter()-start,
               "artifacts": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                             for p in args.output.glob("quality_review.*")}}
    (args.output/"review_manifest.json").write_text(json.dumps(receipt, indent=2)+"\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
