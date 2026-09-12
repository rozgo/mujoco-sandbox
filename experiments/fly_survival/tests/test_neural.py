import sys

import numpy as np
import pytest
from fly_survival.paths import NEURAL_DATA, VENDOR


@pytest.mark.skipif(
    not (NEURAL_DATA / "weights.npz").exists(),
    reason="Official connectome cache is not installed",
)
def test_batched_brains_do_not_leak_stimulation_between_flies():
    import numba

    numba.set_num_threads(2)
    sys.path.insert(0, str(VENDOR))
    from fly_brain import FlyBrain

    brain = FlyBrain(data=NEURAL_DATA, batch=3, device="cpu", seed=19)
    brain.noise_hz = 0
    cells = brain.cells(["LC4", "LPLC2"], side="L")
    totals = np.zeros(3, int)
    for _ in range(20):
        fired = brain.step(inject=[(cells, np.array([0.5, 0, 0]))])
        totals += np.array([len(f) for f in fired])
    assert totals[0] > len(cells)
    assert totals[1] == totals[2] == 0
    np.testing.assert_allclose(brain.v[:, 1], brain.v[:, 2])
