import torch

from embodied_fly.full_body_decoder import FullBodyDecoder
from embodied_fly.full_body_guidance import motion_cost, project_to_parent


def steady_path():
    path = torch.zeros(2, 100, 30, dtype=torch.float64)
    path[..., 2] = 2
    path[..., 6:15] = torch.eye(3).reshape(9)
    return path, torch.ones(2, 200, dtype=torch.float64)


def test_motion_cost_prefers_stationary_supported_flight_and_penalizes_every_axis():
    path, lift = steady_path()
    base, _ = motion_cost(path, lift)
    assert base == 0
    for axis in range(3):
        moving = path.clone()
        moving[..., 3 + axis] = 0.5
        value, _ = motion_cost(moving, lift)
        torch.testing.assert_close(value, torch.tensor(1.0, dtype=torch.float64))
    collapsed = path.clone()
    collapsed[..., 2] = 0.4
    collapsed[..., 14] = 0.5
    value, _ = motion_cost(collapsed, torch.zeros_like(lift))
    assert value > 5


def test_motion_cost_does_not_penalize_phase_shifted_wing_angles():
    path, lift = steady_path()
    t = torch.arange(100, dtype=torch.float64) * 0.002
    for phase in (0, 1.7, 3.1):
        varied = path.clone()
        varied[..., 18:24] = torch.sin(2 * torch.pi * 30 * t + phase)[None, :, None]
        value, _ = motion_cost(varied, lift)
        assert value == 0


def test_action_trust_projection_is_on_shared_weights_not_runtime_outputs():
    torch.manual_seed(12)
    decoder = FullBodyDecoder(8, 12, 78).double()
    origin = {k: v.detach().clone() for k, v in decoder.state_dict().items()}
    calibration = torch.randn(30, 8, dtype=torch.float64)
    initial = decoder(calibration).detach()
    with torch.no_grad():
        decoder[3].bias.add_(0.2)
    discrepancy, fraction = project_to_parent(decoder, origin, calibration, initial, 0.015)
    assert discrepancy <= 0.015 and 0 < fraction < 1
    assert (decoder[3].bias != origin["3.bias"]).all()
    assert decoder(calibration).shape == (30, 78)
