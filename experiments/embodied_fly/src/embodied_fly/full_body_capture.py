"""Physical replay of the shared decoder, with observer-only neural maps."""

import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from embodied_fly.evaluate import load_actor
from embodied_fly.neural_view import NeuralProjection
from embodied_fly.provenance import evidence, sha256, utc_now
from embodied_fly.velocity_hover import evaluate_hover
from embodied_fly.world_data import (
    HISTORY,
    HORIZON,
    layout,
    physical_features,
    physical_metrics,
    window_arrays,
)
from embodied_fly.world_residual_evaluate import ResidualEvaluator


def run(args):
    started = time.perf_counter()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    device = torch.device(args.device)
    actor, checkpoint = load_actor(args.checkpoint, args.graph, device)
    if actor.wing_residual is not None or not actor.shared_decoder_hidden:
        raise ValueError("Current capture requires the shared full-body decoder")
    projection = NeuralProjection.from_graph(args.graph, device=device, size=96)
    maps, times = [], []
    parent = json.loads((args.parent_capture / "report.json").read_text())
    timing = parent.get("observation_timing")
    if timing not in ("refresh before actor", "physics-step return, matching PPO collection"):
        raise ValueError("Parent must declare its observation timing")

    def observe(state, step):
        if step % 10 == 0:
            maps.append(projection.project(state).astype(np.float16))
            times.append(step * 0.002)

    result = evaluate_hover(
        actor,
        checkpoint,
        args.dataset,
        args.output / "flight",
        device,
        refresh_observations=timing == "refresh before actor",
        state_observer=observe,
    )
    np.savez_compressed(
        args.output / "neural_maps.npz",
        activity=np.stack(maps),
        time=np.asarray(times),
        occupancy=projection.occupancy,
    )
    if parent["physical_contract"] != checkpoint["physical_contract"]:
        raise ValueError("Parent capture has different physics")
    differences = []
    for case in result["cases"]:
        original = next(c for c in parent["cases"] if c["episode"] == case["episode"])
        if sha256(args.parent_capture / original["file"]) != original["sha256"]:
            raise ValueError("Parent capture checksum differs")
        with (
            np.load(args.output / "flight" / case["file"]) as z,
            np.load(args.parent_capture / original["file"]) as p,
        ):
            length = min(len(z["time"]), len(p["time"]))
            differences.append(
                {
                    "episode": case["episode"],
                    "samples": length,
                    "max_position_discrepancy_mm": float(
                        np.abs(z["qpos"][:length, :3] - p["qpos"][:length, :3]).max() * 10
                    ),
                    "max_action_discrepancy": float(
                        np.abs(z["action"][:length] - p["action"][:length]).max()
                    ),
                    "velocity_rms_before_mm_s": original["velocity_rms_mm_s"],
                    "velocity_rms_after_mm_s": case["velocity_rms_mm_s"],
                    "survived_before": original["survived_ten_seconds"],
                    "survived_after": case["survived_ten_seconds"],
                }
            )
    # Predict consequences of the recorded command sequence for the new capture.
    # This is an observer audit, never a source of live actuator commands.
    predictor = ResidualEvaluator(
        SimpleNamespace(
            dataset=args.world_data,
            checkpoint=args.world_model,
            root=args.dataset.parent,
            device=args.device,
        )
    )
    indices = layout(predictor.model)
    case = next(c for c in result["cases"] if c["episode"] == 8)
    with np.load(args.output / "flight" / case["file"]) as z:
        qpos, qvel, act, actions = (z[k].copy() for k in ("qpos", "qvel", "act", "action"))
    features, metrics = (
        physical_features(qpos, qvel, act),
        physical_metrics(qpos, qvel, indices),
    )
    starts = np.arange(0, len(actions) - HORIZON, 10)
    batch = window_arrays(features, metrics, actions, starts)
    prediction, analytic = [], []
    for begin in range(0, len(starts), 64):
        s = slice(begin, begin + 64)
        prediction.append(
            predictor.predict(
                batch["sequence"][s, :HISTORY], batch["actions"][s], batch["initial"][s]
            )
        )
        analytic.append(
            predictor.analytic.predict(
                batch["initial"][s], batch["actions"][s], qpos[starts[s]]
            )[0]
        )
    np.savez_compressed(
        args.output / "forecast_windows.npz",
        starts=starts,
        time=starts * 0.002,
        initial=batch["initial"],
        actual=batch["target"],
        predicted=np.concatenate(prediction),
        analytical=np.concatenate(analytic),
    )
    report = {
        "provenance": evidence(),
        "completed_utc": utc_now(),
        "wall_seconds": time.perf_counter() - started,
        "checkpoint_sha256": sha256(args.checkpoint),
        "world_model_sha256": sha256(args.world_model),
        "physical_contract": checkpoint["physical_contract"],
        "migration": checkpoint["decoder_migration"],
        "flight": result,
        "parent_comparison": differences,
        "neural_view": projection.report(),
        "neural_capture_hz": 50,
        "forecast_episode": 8,
        "forecast_windows": len(starts),
        "forecast_scope": "Given recorded future actuator commands; observer prediction only; no model-guided policy update",
        "all_four_survive": all(c["survived_ten_seconds"] for c in result["cases"]),
        "neural_maps_sha256": sha256(args.output / "neural_maps.npz"),
        "forecast_windows_sha256": sha256(args.output / "forecast_windows.npz"),
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps({k: v for k, v in report.items() if k not in ("provenance", "flight")}),
        flush=True,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--graph", type=Path, required=True)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--parent-capture", type=Path, required=True)
    p.add_argument("--world-data", type=Path, required=True)
    p.add_argument("--world-model", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    run(p.parse_args())
