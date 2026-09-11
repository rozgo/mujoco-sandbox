"""Training-only rehearsal of the accepted nine-body walking behavior."""

import hashlib
import time
from pathlib import Path

import numpy as np
import torch

from .bodies import PRESETS
from .env import DogEnv
from .evaluate import lane_command
from .limb_loss import LOSS_BODIES
from .train import load_checkpoint


def collect(checkpoint, cache, seed=12):
    cache = Path(cache)
    digest = hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()
    start = time.perf_counter()
    if cache.exists():
        with np.load(cache) as data:
            if str(data["checkpoint_sha256"]) != digest or int(data["seed"]) != seed:
                raise ValueError("Walking replay provenance mismatch")
            return data["obs"], data["actions"], 0.0
    net, _ = load_checkpoint(checkpoint)
    net.eval()
    env = DogEnv(
        72,
        seed,
        bodies=[PRESETS["healthy"], *LOSS_BODIES],
        faults=False,
        randomize_strength=False,
        threads=8,
    )
    observations, actions = [], []
    try:
        for _ in range(500):
            lane_command(env)
            obs = env.obs()
            with torch.no_grad():
                act, _ = net(torch.as_tensor(obs))
            keep = (env.up[:, 2] > 0.8) & (env.bad_support_force < 1)
            observations.append(obs[keep].copy())
            actions.append(act.numpy()[keep].copy())
            _, done, _, _ = env.step(act.numpy())
            env.reset(np.flatnonzero(done))
    finally:
        env.close()
    obs, act = np.concatenate(observations), np.concatenate(actions)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache, obs=obs, actions=act, checkpoint_sha256=digest, seed=seed
    )
    return obs, act, time.perf_counter() - start
