"""Re-simulate two early end-to-end attempts whose full recordings were not kept, for review videos.

Both start from checkpoints saved by the original runs and replay the original
motion with the scene geometry of today (keeper pads and fences exist but sit
well clear in v0; the tube variant adds geometry only at the target).

  v0_plunge   run_v0 after transport: the original straight 0.6 s plunge to 0.5 mm
              above the tissue, keeper still open (the units bug); ends at MuJoCo's
              instability warning.
  v3a_squeeze run_v3 after seat: the keeper driven 30 um past contact in one
              0.1 s move; ends at the ejection and instability.

Usage: uv run --locked python scripts/reproduce_e2e_attempts.py {v0_plunge,v3a_squeeze}
"""

from datetime import datetime, timezone
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))

from sixlegs.neural_insertion import e2e  # noqa: E402
from sixlegs.neural_insertion.e2e_scene import KEEPER_CLOSED  # noqa: E402

OUT = ROOT/"outputs/neural_insertion/e2e"


def v0_plunge():
    c = e2e.Cycle(0, tubes=False)
    c.restore(OUT/"run_v0/checkpoints/07_transport.pkl")
    site = c.target_site()
    surface = c.tissue.surface(site)
    tip_xy = site[:2]+np.array((0, e2e.CAPTURE_OFFSET))
    q = c.joints(np.array((*tip_xy, surface+.5)), keeper=c.q_ref[4])
    return c, lambda: c.move("lower (straight plunge)", q, .60, .05)


def v3a_squeeze():
    c = e2e.Cycle(0, tubes=True)
    c.restore(OUT/"run_v3/checkpoints/04_seat.pkl")
    q = c.q_ref.copy()
    q[4] = KEEPER_CLOSED*e2e.L+.030
    return c, lambda: c.move("keeper (30 um squeeze)", q, .10, .05)


def main():
    name = sys.argv[1]
    started = datetime.now(timezone.utc).isoformat()
    cycle, action = {"v0_plunge": v0_plunge, "v3a_squeeze": v3a_squeeze}[name]()
    status = "completed"
    try:
        action()
    except (RuntimeError, FloatingPointError) as error:
        status = f"stopped: {error}"
    finally:
        report = e2e.save(cycle, OUT/f"replay_{name}", f"re-simulated: {status}", started)
    print(report["status"])


if __name__ == "__main__":
    main()
