import numpy as np
from adaptive_locomotion.bodies import PRESETS
from adaptive_locomotion.env import DogEnv
from adaptive_locomotion.limb_loss import LOSS_BODIES


def test_smoothness_changes_reward_without_filtering_physics():
    for body in (
        PRESETS["healthy"],
        next(b for b in LOSS_BODIES if b.name == "whole_fr"),
    ):
        envs = [
            DogEnv(
                1,
                bodies=[body],
                randomize=False,
                faults=False,
                threads=1,
                damage_action_rate_weight=w,
                damage_angular_rate_weight=w,
            )
            for w in (0, 0.1)
        ]
        try:
            for step in range(20):
                action = np.full((1, 12), 0.2 * (-1) ** step)
                rewards = [env.step(action)[0] for env in envs]
                np.testing.assert_array_equal(
                    envs[0].groups[0].qpos, envs[1].groups[0].qpos
                )
                if body.name == "healthy":
                    np.testing.assert_array_equal(*rewards)
                else:
                    assert rewards[1][0] < rewards[0][0]
        finally:
            for env in envs:
                env.close()


def test_nonexistent_joints_do_not_receive_added_smoothness_cost():
    body = next(b for b in LOSS_BODIES if b.name == "whole_fr")
    envs = [
        DogEnv(
            1,
            bodies=[body],
            randomize=False,
            faults=False,
            threads=1,
            damage_action_rate_weight=w,
        )
        for w in (0, 10)
    ]
    try:
        for step in range(20):
            action = (1 - envs[0].valid) * (-1) ** step
            rewards = [env.step(action)[0] for env in envs]
            np.testing.assert_array_equal(*rewards)
    finally:
        for env in envs:
            env.close()
