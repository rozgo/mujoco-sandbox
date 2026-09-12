"""Frozen-policy backend/timestep cross-check before changing any learning settings."""

import json
import time

from adaptive_locomotion.bodies import ROOT
from adaptive_locomotion.evaluate import rollout
from adaptive_locomotion.train import load_checkpoint


def main():
    output = ROOT / "outputs/locomotion/warp_training/transfer_diagnostic.json"
    if output.exists():
        raise ValueError("Preserve the existing diagnostic")
    started = time.perf_counter()
    results = []
    for label in ("matched_mjbatch_seed4", "matched_warp_seed4"):
        checkpoint = ROOT / f"assets/locomotion/checkpoints/warp_training/{label}.pt"
        policy, _ = load_checkpoint(checkpoint)
        for backend in ("mjbatch", "warp"):
            for timestep in (0.002, 0.0005):
                for case in ("lower_fr", "whole_fr"):
                    result, _, _ = rollout(
                        policy,
                        case,
                        trials=4,
                        seed=9241,
                        timestep=timestep,
                        physics_backend=backend,
                    )
                    results.append(
                        {
                            "label": label,
                            "physics_backend": backend,
                            "physics_timestep_s": timestep,
                            "result": result,
                        }
                    )
                    print(
                        json.dumps(
                            {
                                "label": label,
                                "backend": backend,
                                "dt": timestep,
                                "case": case,
                                "valid": result["completed_with_allowed_support"],
                                "distance": result["mean_distance_m"],
                            }
                        ),
                        flush=True,
                    )
                    output.write_text(
                        json.dumps(
                            {
                                "results": results,
                                "wall_seconds": time.perf_counter() - started,
                                "new_training_seconds": 0,
                                "complete": len(results) == 16,
                            },
                            indent=2,
                        )
                        + "\n"
                    )


if __name__ == "__main__":
    main()
