import numpy as np
from fly_survival.biology import Needs, Resources
from fly_survival.utility import Thinker


def test_resource_is_finite_and_allocation_is_order_independent():
    a = Resources((0.2, 0.1, 0))
    demands = np.array([[0.2, 0.05, 0.4], [0.3, 0.08, 0.3]])
    x = a.consume(demands)
    b = Resources((0.2, 0.1, 0))
    np.testing.assert_allclose(x, b.consume(demands[::-1])[::-1])
    np.testing.assert_allclose(x.sum(axis=0), a.initial - a.remaining)
    assert np.all(a.remaining >= 0)


def test_needs_deplete_at_rest_and_damage_disables_living_state():
    n = Needs()
    for _ in range(100):
        n.advance(0.02, 0, 0)
    assert n.energy < 0.7 and n.hydration < 0.7
    n.advance(0.02, 0, 0, impact=0.5)
    assert not n.alive
    previous = n.values()
    assert n.advance(1, 0, 0, food=1) == 0
    assert n.values() == previous


def test_heat_accumulates_and_cools():
    n = Needs()
    for _ in range(50):
        n.advance(0.02, 0, 1)
    hot = n.heat
    for _ in range(50):
        n.advance(0.02, 0, 0)
    assert 0 < n.heat < hot


def test_all_activity_scores_apply_and_urgent_hazard_can_interrupt():
    t = Thinker()
    f = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0])
    t.tick(f, 0.02)
    assert t.scores.shape == (6,)
    f[5] = 1
    assert t.tick(f, 0.02) == 5
    assert np.all((t.scores >= 0) & (t.scores <= 1))
