"""Training/validation-only physical interventions for acceleration-residual JEPA."""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from embodied_fly.physical_contract import physical_contract
from embodied_fly.provenance import evidence, sha256, utc_now
from embodied_fly.velocity_demonstrations import environment
from embodied_fly.velocity_hover import failures
from embodied_fly.world_analytic import AnalyticalFly
from embodied_fly.world_data import (
    DT,
    HISTORY,
    HORIZON,
    physical_features,
    physical_metrics,
    window_arrays,
)
from embodied_fly.world_evaluate import action_variants


def collect(args):
    started = time.perf_counter()
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((args.dataset / "manifest.json").read_text())
    env = environment(16, 16)
    if physical_contract(env.model) != manifest["physical_contract"]:
        raise ValueError("Intervention collection must use identical physics")
    analytic = AnalyticalFly(env.model)
    records, excluded = [], []
    for record in manifest["records"]:
        if record["split"] == "test":
            continue  # Never augment training with the previous held-out intervention cases.
        if record["family"] not in ("pid", "student11", "student13"):
            continue
        source = args.root / record["source"]
        if sha256(source) != record["source_sha256"]:
            raise ValueError("Source checksum differs")
        with np.load(source) as z:
            states = {
                k: z[k][: record["states"]].copy() for k in ("qpos", "qvel", "act", "ctrl")
            }
        arrays = {
            k: np.load(args.dataset / record["name"] / f"{k}.npy")
            for k in ("features", "metrics", "actions")
        }
        times = (
            (0, 0.2, 0.6, 1, 2, 2.016, 4, 6, 8, 9)
            if record["split"] == "train"
            else (0, 0.2, 0.6, 2, 2.016, 4)
        )
        for index, seconds in enumerate(times):
            start = round(seconds / DT)
            if start + HORIZON >= record["states"]:
                continue
            bias = (0.0015, 0.003, 0.006)[index % 3]
            actions, names, clipped = action_variants(
                arrays["actions"][start : start + HORIZON],
                manifest["layout"]["wing_action"],
                bias=bias,
                gain=bias * 10,
            )
            original = window_arrays(
                arrays["features"], arrays["metrics"], arrays["actions"], [start]
            )
            history = np.repeat(original["sequence"][:, :HISTORY], 16, 0)
            initial = np.repeat(original["initial"], 16, 0)
            initial_qpos = np.repeat(states["qpos"][start : start + 1], 16, 0)
            inertia = analytic.inertia(initial_qpos[:1])[0]
            env.reset(
                np.arange(16),
                state={k: np.repeat(v[start : start + 1], 16, 0) for k, v in states.items()},
            )
            features, metrics = [], []
            alive = np.ones(16, bool)
            for step in range(HORIZON):
                env.step(actions[:, step])
                features.append(
                    physical_features(
                        env.fields["qpos"], env.fields["qvel"], env.fields["act"]
                    )
                )
                metrics.append(
                    physical_metrics(
                        env.fields["qpos"], env.fields["qvel"], manifest["layout"]
                    )
                )
                alive &= ~failures(env)
            name = f"{record['name']}_{start:05d}"
            if not alive.all():
                excluded.append({"name": name, "failed": int((~alive).sum())})
                continue
            future = np.stack(features, 1)
            future = np.concatenate((future, np.ones((*future.shape[:2], 1), np.float32)), -1)
            target = np.stack(metrics, 1)
            replay = float(np.max(np.abs(target[0, :, :3] - original["target"][0, :, :3])))
            duplicate = float(np.max(np.abs(target[0] - target[15])))
            if replay > 1e-4 or duplicate > 1e-7:
                raise RuntimeError(
                    "Intervention continuation does not replay its physical source"
                )
            file = args.output / f"{name}.npz"
            np.savez_compressed(
                file,
                sequence=np.concatenate((history, future), 1),
                actions=actions,
                initial=initial,
                target=target,
                initial_qpos=initial_qpos,
                inertia=inertia,
            )
            row = {
                "name": name,
                "file": file.name,
                "sha256": sha256(file),
                "split": record["split"],
                "episode": record["episode"],
                "family": record["family"],
                "start_seconds": seconds,
                "bias": bias,
                "sweep_gain": bias * 10,
                "clipped_action_elements": clipped,
                "variants": names,
                "source_sha256": record["source_sha256"],
                "replay_position_error_cm": replay,
                "duplicate_error": duplicate,
            }
            records.append(row)
            print(
                json.dumps({"group": name, "split": row["split"], "replay_error_cm": replay}),
                flush=True,
            )
    report = {
        "schema": "fly-residual-interventions-v1",
        "provenance": evidence(),
        "dataset_sha256": sha256(args.dataset / "manifest.json"),
        "physical_contract": manifest["physical_contract"],
        "records": records,
        "excluded": excluded,
        "worlds": 16,
        "physics_threads": 16,
        "seconds_per_continuation": DT * HORIZON,
        "wall_seconds": time.perf_counter() - started,
        "completed_utc": utc_now(),
    }
    (args.output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--root", type=Path, default=Path("outputs/embodied_fly"))
    p.add_argument("--output", type=Path, required=True)
    collect(p.parse_args())
