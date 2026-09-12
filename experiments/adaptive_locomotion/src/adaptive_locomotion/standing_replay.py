"""Training-only rehearsal of accepted standing states across all original surfaces."""

import hashlib
import time
from pathlib import Path

import numpy as np
import torch

from .bodies import PRESETS
from .limb_loss import LOSS_BODIES
from .standing_env import StandingEnv
from .standing_surfaces import SURFACES


def collect(checkpoint, cache, seed=22):
    from .train import load_checkpoint

    cache = Path(cache)
    digest = hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()
    start = time.perf_counter()
    if cache.exists():
        with np.load(cache, allow_pickle=False) as data:
            if str(data["checkpoint_sha256"]) != digest or int(data["seed"]) != seed:
                raise ValueError("Standing rehearsal provenance mismatch")
            return data["obs"], data["support"], data["actions"], 0.0
    net, _ = load_checkpoint(checkpoint)
    net = net.cpu().eval()
    cases = [(PRESETS["healthy"], s) for s in SURFACES] + [
        (b, "flat") for b in LOSS_BODIES
    ]
    env = StandingEnv(len(cases) * 4, seed, cases=cases, schedule=False, threads=8)
    obs, support, actions = [], [], []
    try:
        for _ in range(500):
            o, s = env.obs(), env.support_obs()
            with torch.no_grad():
                a, _ = net(torch.as_tensor(o), support=torch.as_tensor(s))
            # Keep the original natural balances, including strict failed gates;
            # the purpose is to preserve behavior, not relabel it as success.
            keep = (env.up[:, 2] > 0.5) & np.isfinite(o).all(1)
            obs.append(o[keep].copy())
            support.append(s[keep].copy())
            actions.append(a.numpy()[keep].copy())
            env.step(a.numpy())
    finally:
        env.close()
    o, s, a = np.concatenate(obs), np.concatenate(support), np.concatenate(actions)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache, obs=o, support=s, actions=a, checkpoint_sha256=digest, seed=seed
    )
    return o, s, a, time.perf_counter() - start
