import torch
from test_full_body_decoder import legacy_actor

from embodied_fly.full_body_decoder import consolidate_actor, motor_features
from embodied_fly.full_body_ppo import conditional_replay
from embodied_fly.ppo import joint_log_probability, motor_distribution


def test_raw_full_body_likelihood_and_gradients_match_recurrent_replay_after_updates():
    actor = legacy_actor()
    consolidate_actor(actor)
    observations = torch.randn(12, 3, 4, dtype=torch.float64)
    resets = torch.zeros(12, 3, dtype=torch.bool)
    resets[3, 0] = resets[8, 2] = True
    active = torch.ones(78, dtype=torch.bool)
    log_std = torch.full((78,), -6.5, dtype=torch.float64)
    raw, latent, means, logps = [], [], [], []
    state = actor.initial_state(3)
    with torch.no_grad():
        for t, obs in enumerate(observations):
            result = actor(obs, state)
            dist = motor_distribution(result, active, log_std, 0.0005)
            sample = dist.sample()
            raw.append(motor_features(actor, result.state))
            latent.append(sample)
            means.append(result.action)
            logps.append(joint_log_probability(result, dist, sample, None, motor_only=True))
            state = actor.reset_worlds(result.state, resets[t])
    raw, latent = torch.stack(raw), torch.stack(latent)
    lp, act = conditional_replay(actor, raw, latent, active, log_std)
    torch.testing.assert_close(act, torch.stack(means), atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(lp, torch.stack(logps), atol=1e-8, rtol=1e-12)
    optimizer = torch.optim.SGD(actor.motor_decoder.parameters(), lr=1e-7)
    (-lp.mean()).backward()
    assert (actor.motor_decoder[3].weight.grad.abs().sum(1) > 0).all()
    optimizer.step()
    optimizer.zero_grad()
    cached_lp, cached_act = conditional_replay(actor, raw, latent, active, log_std)
    (-cached_lp.mean()).backward()
    cached_gradients = [p.grad.clone() for p in actor.motor_decoder.parameters()]
    optimizer.zero_grad()
    state = actor.initial_state(3)
    losses = []
    for t, obs in enumerate(observations):
        result = actor(obs, state)
        dist = motor_distribution(result, active, log_std, 0.0005)
        lp = joint_log_probability(result, dist, latent[t], None, motor_only=True)
        torch.testing.assert_close(lp, cached_lp[t], atol=1e-8, rtol=1e-12)
        torch.testing.assert_close(result.action, cached_act[t], atol=1e-12, rtol=1e-12)
        losses.append(lp)
        state = actor.reset_worlds(result.state, resets[t])
    (-torch.stack(losses).mean()).backward()
    for p, expected in zip(actor.motor_decoder.parameters(), cached_gradients, strict=True):
        torch.testing.assert_close(p.grad, expected, atol=1e-7, rtol=1e-9)
    assert all(
        p.grad is None
        for n, p in actor.named_parameters()
        if not n.startswith("motor_decoder.")
    )
