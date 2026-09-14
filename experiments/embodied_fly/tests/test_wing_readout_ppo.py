import torch
from test_brain import tiny_graph

from embodied_fly.brain import EmbodiedBrain
from embodied_fly.ppo import joint_log_probability, motor_distribution
from embodied_fly.wing_readout_ppo import (
    freeze_upstream,
    motor_features,
    readout_action,
    readout_replay,
)


def test_cached_readout_is_exact_before_and_after_update_with_resets():
    torch.manual_seed(402)
    actor = EmbodiedBrain(
        tiny_graph(),
        [0, 1],
        [2, 3],
        [4, 5],
        4,
        78,
        wing_residual_enabled=True,
        wing_residual_hidden=8,
    ).train()
    actor.set_motor_only(True)
    freeze_upstream(actor)
    initial = {n: p.clone() for n, p in actor.named_parameters()}
    observations = torch.randn(12, 3, 4)
    done = torch.zeros(12, 3, dtype=torch.bool)
    done[4, 0] = done[8, 2] = True
    active, log_std = torch.ones(78, dtype=torch.bool), torch.full((78,), -5.8)
    data = {k: [] for k in ("motor_features", "mean_action", "latent", "logp")}
    memory = actor.initial_state(3)
    with torch.no_grad():
        for t, obs in enumerate(observations):
            result = actor(obs, memory)
            distribution = motor_distribution(result, active, log_std, 0.0005)
            latent = distribution.sample()
            data["motor_features"].append(motor_features(actor, result.state))
            data["mean_action"].append(result.action)
            data["latent"].append(latent)
            data["logp"].append(
                joint_log_probability(result, distribution, latent, None, motor_only=True)
            )
            memory = actor.reset_worlds(result.state, done[t])
    data = {k: torch.stack(v) for k, v in data.items()}
    logp, actions, _ = readout_replay(actor, data, 0, 12, active, log_std)
    torch.testing.assert_close(actions, data["mean_action"])
    torch.testing.assert_close(logp, data["logp"], atol=0.003, rtol=0)
    optimizer = torch.optim.Adam(actor.wing_residual.parameters(), lr=0.01)
    optimizer.zero_grad()
    actions[..., 14:20].square().mean().backward()
    optimizer.step()
    memory = actor.initial_state(3)
    for t, obs in enumerate(observations):
        result = actor(obs, memory)
        torch.testing.assert_close(
            readout_action(actor, data["motor_features"][t]), result.action
        )
        memory = actor.reset_worlds(result.state, done[t])
    for name, parameter in actor.named_parameters():
        if not name.startswith("wing_residual."):
            torch.testing.assert_close(parameter, initial[name], rtol=0, atol=0)
    assert any(
        not torch.equal(p, initial[n])
        for n, p in actor.named_parameters()
        if n.startswith("wing_residual.")
    )
