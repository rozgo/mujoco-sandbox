from types import SimpleNamespace

import numpy as np
import pytest

from embodied_fly.coordinated_hover_audit import counterexamples
from embodied_fly.velocity_hover import HoverReward, reward_from_recipe, reward_rates


def test_coordinated_reward_prefers_recovery_to_failure_and_rejects_axis_tradeoffs():
    result = counterexamples()
    assert result["passed"]
    assert result["examples"]["immediate_failure"]["return"] == -2
    assert result["examples"]["immediate_failure"]["duration_seconds"] == 0.002
    assert result["examples"]["stationary"]["return"] == pytest.approx(50)


@pytest.mark.parametrize(
    "objective,scale,weight",
    [("separate", 2, 2), ("separate", 2, 3), ("vector", 2, 2), ("vector", 0.5, 2)],
)
def test_saved_reward_recipe_restores_exact_axis_scales_and_terms(objective, scale, weight):
    env = SimpleNamespace(n=2)
    source = HoverReward(env, scale, weight, objective)
    restored = reward_from_recipe(env, source.recipe)
    assert source.recipe == restored.recipe
    args = (np.array([[0.5, 1, -2], [0, 0, 0]]), np.zeros((2, 3)), np.ones(2))
    expected = reward_rates(*args, scale, weight, objective)
    actual = reward_rates(
        *args, restored.horizontal_scale, restored.vertical_weight, restored.velocity_objective
    )
    for key in expected:
        np.testing.assert_array_equal(expected[key], actual[key])
