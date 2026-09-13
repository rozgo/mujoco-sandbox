import torch

from embodied_fly.train import weighted_motor_loss


def test_same_clock_mixed_tasks_receive_their_own_ground_and_wing_weights():
    predicted = torch.tensor([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]], requires_grad=True)
    target = torch.zeros_like(predicted)
    flight = torch.tensor([0.0, 1.0])
    result = weighted_motor_loss(predicted, target, flight, [1, 2], 2, 4)
    # Ground keeps 4x all-channel retention; flight adds 2x wing MSE.
    expected = (4 * (14 / 3) + (14 / 3) + 2 * (13 / 2)) / 2
    torch.testing.assert_close(result, torch.tensor(expected))
    result.backward()
    assert torch.isfinite(predicted.grad).all()
