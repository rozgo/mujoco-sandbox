"""Matched body, neural and reward histories for episode initialization only."""

import json
from pathlib import Path

import numpy as np
import torch

from embodied_fly.physical_contract import physical_contract
from embodied_fly.provenance import sha256

AUXILIARY = (
    "previous_action",
    "mean_sensors",
    "command",
    "needs",
    "requested_height_cm",
    "requested_xy_cm",
    "forbidden_peak",
    "_wing_applied",
)
DERIVED = ("sensordata", "xmat", "ximat")
WING = ("activity", "wrench", "lift")


def capture(env, reward, memory):
    """Copy a pre-action boundary, preserving the existing observation timing."""
    result = {"integration": env.batch.bind("state").copy()}
    result.update({f"aux_{k}": getattr(env, k).copy() for k in AUXILIARY})
    result.update({f"derived_{k}": env.fields[k].copy() for k in DERIVED})
    result.update({f"wing_{k}": getattr(env.wing_forces, k).copy() for k in WING})
    result.update(
        memory=memory.detach().cpu().numpy().T.copy(),
        # Oldest / next-to-replace sample first, independently of destination index.
        reward_history=np.moveaxis(np.roll(reward.history, -reward.index, axis=0), 1, 0),
        reward_total=reward.total.copy(),
        reward_count=reward.count.copy(),
        physical_age=env.ages.copy(),
    )
    return result


def restore(env, reward, memory, ids, saved, *, elapsed_reset=True):
    """Initialize selected episodes; never call this during their live control."""
    ids = np.asarray(ids, dtype=np.int64)
    if ids.ndim != 1 or len(np.unique(ids)) != len(ids) or np.any((ids < 0) | (ids >= env.n)):
        raise ValueError("Distinct valid destination worlds required")
    if not len(ids):
        return memory
    state = env.batch.bind("state")
    if saved["integration"].shape != state[ids].shape:
        raise ValueError("Full MuJoCo integration state shape differs")
    if saved["memory"].shape != (len(ids), memory.shape[0]):
        raise ValueError("Neural state does not match physical worlds")
    env.batch.reset(ids)
    state[ids] = saved["integration"]
    env.batch.forward(ids)
    # Forward rebuilds derived geometry but can change warm-start state. Restore
    # the integration snapshot again, then the historical observation caches.
    state[ids] = saved["integration"]
    for key in DERIVED:
        env.fields[key][ids] = saved[f"derived_{key}"]
    for key in AUXILIARY:
        getattr(env, key)[ids] = saved[f"aux_{key}"]
    for key in WING:
        getattr(env.wing_forces, key)[ids] = saved[f"wing_{key}"]
    env.ages[ids] = 0 if elapsed_reset else saved["physical_age"]
    reward.history[:, ids] = np.roll(
        np.moveaxis(saved["reward_history"], 0, 1), reward.index, axis=0
    )
    reward.total[ids] = saved["reward_total"]
    reward.count[ids] = saved["reward_count"]
    memory[:, ids] = torch.as_tensor(saved["memory"].T.copy(), device=memory.device)
    return memory


class RecoveryStarts:
    def __init__(self, path, env, parent_sha, actor, reward, seed):
        self.path = Path(path)
        report = json.loads((self.path / "report.json").read_text())
        if (
            not report["passed"]
            or report["parent_checkpoint_sha256"] != parent_sha
            or report["physical_contract"] != physical_contract(env.model)
            or report["reward_recipe"] != reward.recipe
        ):
            raise ValueError(
                "Recovery bank parent/physics/reward or restoration audit differs"
            )
        if sha256(self.path / "states.npz") != report["states_sha256"]:
            raise ValueError("Recovery initialization checksum differs")
        with np.load(self.path / "states.npz") as arrays:
            self.arrays = {k: arrays[k].copy() for k in arrays.files}
        self.records = report["records"]
        self.training = np.array([i for i, r in enumerate(self.records) if r["episode"] < 8])
        if not len(self.training) or env.n % 2:
            raise ValueError("Need training-only starts and an even world count")
        self.enabled = np.arange(env.n) >= env.n // 2
        self.rng = np.random.default_rng(seed)
        self.events = []
        self.active_record = np.full(env.n, -1, dtype=int)
        self.recipe = {
            "kind": "parent-generated physical/neural recovery episode starts",
            "bank_sha256": report["states_sha256"],
            "bank_report_sha256": sha256(self.path / "report.json"),
            "normal_worlds": env.n // 2,
            "recovery_worlds": env.n // 2,
            "training_records": len(self.training),
            "training_episodes": list(range(8)),
            "parent_flight_seconds": [2, 4, 6],
            "seed": seed,
            "reset_scope": "full integration, historical sensors/actions, neural memory, reward ring; elapsed episode timer restarts",
            "runtime_assistance": False,
        }
        if actor.core.neurons != self.arrays["memory"].shape[1]:
            raise ValueError("Recovery bank graph size differs")

    def reset(self, env, reward, memory, ids):
        ids = np.asarray(ids)
        targets = ids[self.enabled[ids]]
        self.active_record[ids] = -1
        if not len(targets):
            return memory
        chosen = self.rng.choice(self.training, len(targets))
        saved = {k: v[chosen] for k, v in self.arrays.items()}
        restore(env, reward, memory, targets, saved)
        self.active_record[targets] = chosen
        for world, record in zip(targets, chosen, strict=True):
            self.events.append(
                {"world": int(world), "record": int(record), **self.records[record]}
            )
        return memory
