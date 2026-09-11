"""Evaluate final matched CPU/Warp checkpoints and retain all paired results."""

import argparse
import hashlib
import json

from evaluate_clearance import evaluate
from evaluate_visible_steps import compare

from adaptive_locomotion.bodies import ROOT


def main(labels):
    docs = ROOT / "docs/locomotion/warp_training"
    for label in labels:
        directory = docs / label
        provenance = json.loads((directory / "checkpoint.json").read_text())
        checkpoint = ROOT / provenance["checkpoint"]
        assert (
            hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            == provenance["checkpoint_sha256"]
        )
        output = directory / "evaluation.json"
        if output.exists():
            saved = json.loads(output.read_text())
            assert (
                saved["sha256"] == provenance["checkpoint_sha256"]
                and saved["seed"] == 9237
            )
        else:
            evaluate(checkpoint, output, trials=8, seed=9237, timestep=0.0005)
    pairs = []
    for seed in (2, 3, 4):
        paths = [
            docs / f"matched_{backend}_seed{seed}" / "evaluation.json"
            for backend in ("mjbatch", "warp")
        ]
        if not all(p.exists() for p in paths):
            continue
        cpu, warp = [json.loads(p.read_text()) for p in paths]
        relative = compare(cpu, warp)
        pairs.append(
            {
                "seed": seed,
                "cpu_completed": sum(
                    c["completed_with_allowed_support"] for c in cpu["cases"]
                ),
                "warp_completed": sum(
                    c["completed_with_allowed_support"] for c in warp["cases"]
                ),
                "gait_comparison": relative,
                "by_case": {
                    a["case"]: {
                        "cpu": a["completed_with_allowed_support"],
                        "warp": b["completed_with_allowed_support"],
                    }
                    for a, b in zip(cpu["cases"], warp["cases"], strict=True)
                },
            }
        )
        print(
            json.dumps({k: v for k, v in pairs[-1].items() if k != "gait_comparison"}),
            flush=True,
        )
    (docs / "matched_pairs.json").write_text(
        json.dumps({"seed": 9237, "pairs": pairs}, indent=2) + "\n"
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("labels", nargs="+")
    main(p.parse_args().labels)
