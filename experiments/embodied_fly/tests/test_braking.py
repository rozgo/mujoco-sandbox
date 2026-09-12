from pathlib import Path

import numpy as np

from embodied_fly.batch import FlyBatch
from embodied_fly.braking import BrakingTeacher


def test_batched_braking_teacher_matches_single_oracle_without_state_writes():
    teacher_path = (
        Path(__file__).resolve().parents[3] / "assets/embodied_fly/teachers/walking.npz"
    )
    env = FlyBatch(2, 2)
    teacher = BrakingTeacher(env, teacher_path)
    oracle = teacher.oracle
    single = env.template
    oracle.set_reference(0, 0, 2)
    env.reset([0], yaw=[0.23])
    single.reset(yaw=0.23)
    for _ in range(20):
        before = env.fields["qpos"].copy()
        observations = teacher.observation()
        expected = np.concatenate(
            [
                np.asarray(oracle.observation(0, "receding")[key]).reshape(-1)
                for key in teacher.policy.manifest["observation_shapes"]
            ]
        ).astype(np.float32)
        np.testing.assert_allclose(observations[0], expected, atol=2e-6, rtol=2e-6)
        actions = teacher.act().numpy()
        np.testing.assert_allclose(actions[0], oracle.act(0, "receding"), atol=2e-5, rtol=2e-5)
        np.testing.assert_array_equal(env.fields["qpos"], before)
        single.step(actions[0])
        env.step(actions)
