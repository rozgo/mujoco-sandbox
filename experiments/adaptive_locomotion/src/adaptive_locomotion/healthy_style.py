"""Training-only healthy-policy targets for surviving joints after removal."""

import numpy as np


def teacher_observation(obs):
    """Query the intact teacher without its out-of-distribution validity bits.

    Missing joints are represented at nominal pose, zero speed and zero previous
    action. All other measurements are actual current feedback. Only the teacher
    gets this hypothetical intact encoding; the deployed actor sees real inputs.
    """
    result = obs.copy()
    valid = obs[:, 45:57]
    for start in (0, 12, 30):
        result[:, start : start + 12] *= valid
    result[:, 45:57] = 1
    return result


def style_mask(obs):
    valid = obs[:, 45:57]
    return valid * (valid < 1).any(1)[:, None]


def style_bonus(action, target, mask):
    error = (((np.clip(action, -3, 3) - target) * mask) ** 2).sum(1)
    count = mask.sum(1)
    return (count > 0) * np.exp(-error / np.maximum(count, 1) / 0.25)


def style_loss(mean, target, mask):
    count = mask.sum(-1)
    error = (((mean - target) * mask).square().sum(-1)) / count.clamp_min(1)
    return error.sum() / (count > 0).sum().clamp_min(1)


class HealthyMotion:
    """Nearest healthy motion per leg; phase is inferred from joint state.

    This is a small reference library, not an external clock or runtime lookup.
    Joint angles, velocities and commanded speed select a healthy sample. Every
    surviving joint is compared; the removed channel contributes no distance.
    """

    def __init__(self, observations, actions):
        self.observations = np.asarray(observations, np.float32)
        self.actions = np.asarray(actions, np.float32)
        self.features = self.feature(self.observations)
        self.scale = np.maximum(
            self.features.std(0), np.array([0.1] * 3 + [0.05] * 3 + [0.2])
        ).astype(np.float32)

    @staticmethod
    def feature(obs):
        return np.concatenate(
            (
                obs[:, :12].reshape(-1, 4, 3),
                obs[:, 12:24].reshape(-1, 4, 3),
                np.broadcast_to(obs[:, 42:43, None], (len(obs), 4, 1)),
            ),
            axis=2,
        )

    def query(self, obs):
        features = self.feature(obs)
        valid = obs[:, 45:57].reshape(-1, 4, 3)
        mask = np.concatenate((valid, valid, np.ones((len(obs), 4, 1))), axis=2)
        target = np.empty((len(obs), 4, 3), np.float32)
        errors = np.zeros((len(obs), 4), np.float32)
        for leg in range(4):
            delta = (features[:, None, leg] - self.features[None, :, leg]) / self.scale[
                leg
            ]
            distance = (delta**2 * mask[:, None, leg]).sum(2) / mask[:, leg].sum(1)[
                :, None
            ]
            nearest = distance.argmin(1)
            target[:, leg] = self.actions.reshape(-1, 4, 3)[nearest, leg]
            errors[:, leg] = distance[np.arange(len(obs)), nearest]
        return target.reshape(-1, 12), errors

    @classmethod
    def collect(cls, teacher, seed, output):
        import hashlib

        import torch

        from .evaluate import make_case

        env = make_case("healthy", trials=4, seed=seed, support_substeps=True)
        observations, actions = [], []
        device = next(teacher.parameters()).device
        try:
            for step in range(300):
                env.commands[:] = 0
                env.commands[:, 0] = [0.3, 0.45, 0.6, 0.75]
                obs = env.obs().copy()
                with torch.no_grad():
                    action, _ = teacher(torch.as_tensor(obs, device=device))
                action = action.cpu().numpy()
                if step >= 50 and step % 4 == 0:
                    observations.append(obs)
                    actions.append(action)
                _, _, fell, _ = env.step(action)
                group = env.groups[0]
                bad = group.support_peaks[:, ~group.support_allowed].max()
                if fell.any() or bad > 1:
                    raise ValueError("Healthy reference motion lost valid support")
        finally:
            env.close()
        obs, act = np.concatenate(observations), np.concatenate(actions)
        np.savez_compressed(output, observations=obs, actions=act)
        return cls(obs, act), hashlib.sha256(output.read_bytes()).hexdigest()
