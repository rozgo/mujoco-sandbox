import torch

from adaptive_locomotion.env import CONTEXT_DIM, HISTORY, OBS_DIM
from adaptive_locomotion.policy import Policy


def test_deployed_history_actions_ignore_private_simulator_context():
    torch.manual_seed(7)
    net = Policy("history")
    obs = torch.randn(4, OBS_DIM)
    history = torch.randn(4, HISTORY, OBS_DIM)
    with torch.no_grad():
        a, _ = net(obs, torch.zeros(4, CONTEXT_DIM), history)
        b, _ = net(obs, torch.ones(4, CONTEXT_DIM), history)
        c, _ = net(obs, None, history)
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    torch.testing.assert_close(a, c, rtol=0, atol=0)


def test_history_encoder_does_not_mix_independent_environments():
    torch.manual_seed(3)
    net = Policy("history")
    history = torch.randn(3, HISTORY, OBS_DIM)
    with torch.no_grad():
        batched = net.estimate(history)
        solo = torch.cat([net.estimate(h[None]) for h in history], 0)
    torch.testing.assert_close(batched, solo, rtol=1e-5, atol=1e-6)
