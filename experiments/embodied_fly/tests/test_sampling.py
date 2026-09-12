import numpy as np
import torch

from embodied_fly.train import sample


def test_reset_supervision_keeps_episode_boundaries_and_first_action():
    episodes = [
        {
            "observation": (i * 1000 + np.arange(24))[:, None],
            "action": (i * 1000 + np.arange(24) + 0.5)[:, None],
            "activity": np.full(24, i, dtype=np.int64),
        }
        for i in range(4)
    ]
    for reset_start in (False, True):
        batch = sample(
            episodes,
            np.random.default_rng(9),
            8,
            32,
            torch.device("cpu"),
            reset_start=reset_start,
        )
        ids = batch["activity"][0]
        torch.testing.assert_close(batch["activity"], ids.expand(8, 32))
        torch.testing.assert_close(
            batch["action"] - batch["observation"], torch.full((8, 32, 1), 0.5)
        )
        torch.testing.assert_close(
            batch["observation"][1:] - batch["observation"][:-1], torch.ones((7, 32, 1))
        )
        if reset_start:
            torch.testing.assert_close(batch["observation"][0, :, 0], ids.float() * 1000)
