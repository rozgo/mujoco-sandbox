"""Soft bilateral policy consistency, including the reflected missing joints.

Independent implementation of the mirror-loss idea described by Mittal et al.,
ICRA 2024, and RSL-RL's symmetry extension. This compares two different states;
it never ties the left and right actions within one state or at deployment.
"""

JOINT_SWAP = [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8]


def mirror_action(action):
    result = action[..., JOINT_SWAP].clone()
    result[..., ::3] *= -1
    return result


def mirror_observation(obs):
    result = obs.clone()
    for start in (0, 12, 30):
        result[..., start : start + 12] = mirror_action(obs[..., start : start + 12])
    result[..., [24, 26, 28, 43, 44]] *= -1
    result[..., 45:57] = obs[..., [45 + j for j in JOINT_SWAP]]
    # Nine range rays: three lateral positions for each forward distance.
    result[..., 57:66] = obs[..., [59, 58, 57, 62, 61, 60, 65, 64, 63]]
    return result


def mirror_loss(learner, obs, action_mean):
    reflected = mirror_observation(obs)
    predicted, _ = learner(reflected)
    target = mirror_action(action_mean.detach())
    valid = reflected[..., 45:57]
    return (
        ((predicted - target) * valid).square().sum(-1) / valid.sum(-1).clamp_min(1)
    ).mean()
