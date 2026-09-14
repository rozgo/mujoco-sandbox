"""Causal, episode-disjoint state/action windows for the fly prediction pilot."""

import argparse
import json
import time
from pathlib import Path

import mujoco
import numpy as np

from embodied_fly.physical_contract import physical_contract
from embodied_fly.provenance import evidence, sha256, utc_now
from embodied_fly.wing_position import wing_actuators

DT = 0.002
HISTORY = 25
BLOCK = 1  # Predict every 500 Hz state; never average away wing phase.
HORIZON = 100
METRIC_SCALE = np.array(
    [0.1] * 3 + [1.0] * 3 + [0.1] * 9 + [1.0] * 3 + [0.5] * 6 + [50.0] * 6, np.float32
)


def split_for_episode(episode):
    if episode in (0, 2, 3, 4, 5):
        return "train"
    if episode in (1, 6, 7):
        return "validation"
    if episode in (8, 9):
        return "test"
    raise ValueError("Episode has no predetermined split")


def quaternion_matrix(q):
    q = np.asarray(q, dtype=np.float64)
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    w, x, y, z = np.moveaxis(q, -1, 0)
    return np.stack(
        (
            1 - 2 * (y * y + z * z),
            2 * (x * y - w * z),
            2 * (x * z + w * y),
            2 * (x * y + w * z),
            1 - 2 * (x * x + z * z),
            2 * (y * z - w * x),
            2 * (x * z - w * y),
            2 * (y * z + w * x),
            1 - 2 * (x * x + y * y),
        ),
        axis=-1,
    )


def layout(model):
    ids = wing_actuators(model)
    joints = model.actuator_trnid[ids, 0]
    return {
        "wing_action": ids.tolist(),
        "wing_qpos": model.jnt_qposadr[joints].tolist(),
        "wing_qvel": model.jnt_dofadr[joints].tolist(),
        "nq": model.nq,
        "nv": model.nv,
        "na": model.na,
        "nu": model.nu,
    }


def physical_features(qpos, qvel, act):
    # Absolute x/y omitted; height remains to distinguish proximity to the floor.
    # Quaternion sign is canonicalized independently of any future observation.
    pose = np.asarray(qpos[..., 3:]).copy()
    pose[..., :4] *= np.where(pose[..., :1] < 0, -1, 1)
    return np.concatenate((pose, qvel, act, qpos[..., 2:3]), axis=-1).astype(np.float32)


def physical_metrics(qpos, qvel, indices):
    return np.concatenate(
        (
            qpos[..., :3],
            qvel[..., :3],
            quaternion_matrix(qpos[..., 3:7]),
            qvel[..., 3:6],
            qpos[..., indices["wing_qpos"]],
            qvel[..., indices["wing_qvel"]],
        ),
        axis=-1,
    ).astype(np.float32)


def window_arrays(features, metrics, actions, starts):
    """state[t] precedes action[t]; action[t] produces state[t+1]."""
    starts = np.asarray(starts, dtype=np.int64)
    if (
        starts.ndim != 1
        or np.any(starts < 0)
        or np.any(starts + BLOCK * HORIZON >= len(features))
    ):
        raise ValueError("Window must end at an observed, valid next state")
    # Left padding repeats the first available measurement, with a presence bit.
    end = starts[:, None] + np.arange(HORIZON + 1)[None] * BLOCK
    raw_indices = starts[:, None] + np.arange(-HISTORY + 1, BLOCK * HORIZON + 1)
    indices = np.maximum(raw_indices, 0)
    sequence = np.concatenate((features[indices], (raw_indices >= 0)[..., None]), axis=-1)
    future_actions = actions[starts[:, None] + np.arange(BLOCK * HORIZON)]
    return {
        "sequence": sequence.astype(np.float32),
        "actions": future_actions.reshape(-1, HORIZON, BLOCK * actions.shape[-1]),
        "initial": metrics[starts],
        "target": metrics[end[:, 1:]],
    }


def prepare(args):
    begin = time.perf_counter()
    args.output.mkdir(parents=True, exist_ok=False)
    sources = [
        ("pid", "velocity_teacher_dataset_01"),
        ("student11", "velocity_hover_ppo_11/midpoint"),
        ("student13", "velocity_hover_ppo_13/final"),
        ("recovery13", "recovery_eval_14/parent"),
        ("failed14", "velocity_hover_ppo_14/final"),
    ]
    records, contract, indices = [], None, None
    for family, folder in sources:
        folder = args.root / folder
        report = json.loads((folder / "report.json").read_text())
        current = report["physical_contract"]
        if contract is None:
            contract = current
            model = mujoco.MjModel.from_binary_path(str(folder / "model.mjb"))
            if physical_contract(model) != contract:
                raise ValueError("Saved model fingerprint does not match data")
            indices = layout(model)
        if current != contract:
            raise ValueError("Mixed physical contracts in world-model dataset")
        for case in report["cases"]:
            file = folder / case["file"]
            episode = case.get("episode", case.get("world"))
            if episode is None:
                # Recovery report carries its original episode explicitly.
                raise ValueError(f"Missing episode lineage in {family}")
            episode = int(episode)
            split = split_for_episode(episode)
            # The failed final actor is an additional shift test, never fitted.
            if family == "failed14" and split != "test":
                continue
            digest = sha256(file)
            if digest != case["sha256"]:
                raise ValueError("Source trajectory checksum differs")
            with np.load(file) as z:
                n = len(z["action"])
                if case.get("first_failure_seconds") is not None:
                    n = min(n, round(case["first_failure_seconds"] / DT))
                if n <= BLOCK * HORIZON:
                    raise ValueError("Too few valid transitions")
                t = z["time"][:n]
                if not np.allclose(np.diff(t), DT, atol=1e-8, rtol=0):
                    raise ValueError("Source timing is not 500 Hz")
                qpos, qvel, act = (z[k][:n].copy() for k in ("qpos", "qvel", "act"))
                actions = z["action"][:n].copy()
                # Existing captures store both pre-state and post-position.
                error = float(np.max(np.abs(qpos[1:, :3] - z["post_position"][: n - 1])))
                if error > 1e-8:
                    raise ValueError("Recorded pre/action/post alignment differs")
            name = f"{family}_{file.stem}"
            destination = args.output / name
            destination.mkdir()
            for key, values in {
                "features": physical_features(qpos, qvel, act),
                "metrics": physical_metrics(qpos, qvel, indices),
                "actions": actions,
            }.items():
                if not np.isfinite(values).all():
                    raise ValueError("Nonfinite training data")
                np.save(destination / f"{key}.npy", values.astype(np.float32))
            records.append(
                {
                    "name": name,
                    "family": family,
                    "episode": episode,
                    "split": split,
                    "states": n,
                    "source": str(file.relative_to(args.root)),
                    "source_sha256": digest,
                    "source_report_sha256": sha256(folder / "report.json"),
                    "pre_post_position_error_cm": error,
                }
            )
            print(json.dumps({"prepared": name, "states": n, "split": split}), flush=True)
    manifest = {
        "schema": "fly-world-data-v1",
        "provenance": evidence(),
        "physical_contract": contract,
        "layout": indices,
        "control_dt": DT,
        "history_steps": HISTORY,
        "action_block": BLOCK,
        "prediction_steps": HORIZON,
        "records": records,
        "wall_seconds": time.perf_counter() - begin,
        "completed_utc": utc_now(),
        "future_actions_are_conditioning_not_policy_inputs": True,
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("outputs/embodied_fly"))
    p.add_argument("--output", type=Path, required=True)
    prepare(p.parse_args())
