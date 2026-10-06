"""End-to-end cycle, design v2: hook the loop, pinch, peel, align, insert fast, snap back.

Same runner machinery as the first pass (servo, measured checks, checkpoints,
loss and instability stops); scripted yardstick motions, no disturbances.
Tissue: the phantom with the needle-tissue force model (no tube).

Usage: uv run --locked python -m sixlegs.neural_insertion.modern --output DIR [--stop-after STEP]
"""

import argparse
from datetime import datetime, timezone
from pathlib import Path

import mujoco
import numpy as np

from . import e2e
from .modern_scene import LOOP_ABOVE_TIP, PINCHER_OPEN, load_modern
from .motion import JOINT_NAMES, Servo
from .tissue import Tissue, TissueParams

L = 1000.
INSERT_DEPTH = 2.0       # mm: needle tip below the surface at full insertion
PINCH = -.02             # rad: commanded past contact; the torque limit sets the pinch force


class ModernCycle(e2e.Cycle):
    def __init__(self, target=0):
        self.m, self.d, meta = load_modern()
        self.meta, self.target = meta, target
        self.qadr = np.array([self.m.jnt_qposadr[self.m.joint(n).id] for n in JOINT_NAMES])
        scratch = mujoco.MjData(self.m)
        mujoco.mj_kinematics(self.m, scratch)
        self.tip0 = scratch.site("needle_tip").xpos.copy()
        self.servo = Servo(self.m)
        self.tissue = Tissue(self.m, meta["thread_bodies"], TissueParams())
        self.trace = {k: [] for k in ("time", "qpos", "tip", "eyelet", "depth", "needle_axial", "act_force",
                                       "ncon", "phase")}
        self.events, self.q_ref, self.prohibited, self.done = [], None, [], []
        self.robot = {self.m.body(n).id for n in ("stage_x", "stage_y", "stage_z", "insertion", "retainer",
                                                    "pincher")}
        self.phantom = self.m.geom("tissue_phantom").id
        self.clamp, self.on_step = None, None
        self.ret = JOINT_NAMES.index("retainer")
        self.pincher = self.m.actuator("pincher").id

    def start(self):
        loop = self.eyelet
        q = self.joints(np.array((loop[0], loop[1], loop[2]+1.0)))
        self.d.qpos[self.qadr] = q  # initialization, before any stepping
        self.d.ctrl[:] = 0
        self.d.ctrl[self.servo.act] = q
        self.d.qpos[self.m.jnt_qposadr[self.m.joint("pincher").id]] = PINCHER_OPEN
        self.d.ctrl[self.pincher] = PINCHER_OPEN
        mujoco.mj_forward(self.m, self.d)
        self.q_ref = q
        return q

    def steps(self):
        site = self.target_site()
        surface = self.tissue.surface(site)
        loop0 = self.eyelet.copy()

        def at(xyz):
            return self.joints(np.asarray(xyz, float))

        def settle():
            self.move("settle", self.start(), .01)

        def descend():
            # Tip through the loop's centre until the ledge sits on the loop's far edge.
            self.move("descend", at((loop0[0], loop0[1], loop0[2]-LOOP_ABOVE_TIP+.005)), .10, .01)

        def pinch():
            self.d.ctrl[self.pincher] = PINCH
            self.move("pinch", self.q_ref, 0., .06)
            gap = float(np.linalg.norm(self.eyelet-self.tip))
            self.check("loop on the needle", gap < .5, {"loop_to_tip_mm": gap})

        def peel():
            # Up 1.5 mm while moving 1 mm toward the anchor, so the thread never goes taut.
            tip = self.tip_of(self.q_ref)
            self.move("peel", at((tip[0]+1.0, tip[1], tip[2]+1.5)), .10, .02, carried=True)
            lifted = self.eyelet[2]-loop0[2]
            self.check("loop lifted with the needle", lifted > 1.0, {"loop_lift_mm": float(lifted)})

        def transport():
            # Back over the implant carrier to the target at carry height; the thread folds into a slack U.
            tip = self.tip_of(self.q_ref)
            self.move("transport", at((site[0], site[1], tip[2])), .25, .03, carried=True)

        def align():
            # Down to 0.3 mm above the surface.
            self.move("align", at((site[0], site[1], surface+.3)), .10, .02, carried=True)
            lateral = float(np.linalg.norm(self.tip[:2]-site[:2]))
            self.check("aligned over the target", lateral < .02, {"tip_to_target_lateral_mm": lateral})

        def release_pincher():
            self.d.ctrl[self.pincher] = PINCHER_OPEN
            self.move("open pincher", self.q_ref, 0., .03, carried=True)

        def insert():
            # Fast insertion: 2.3 mm in 25 ms (peak about 170 mm/s).
            self.move("insert", at((site[0], site[1], surface-INSERT_DEPTH)), .025, .02, carried=True)
            depth = self.tissue.surface(self.eyelet)-self.eyelet[2]
            self.check("loop inserted", depth > 1.0, {"loop_depth_mm": float(depth),
                                                      "punctured": self.tissue.state.punctured})

        def retract():
            # Snap retraction: 3 mm in 25 ms, peak acceleration about 28,000 mm/s^2.
            self.move("retract", at((site[0], site[1], surface+1.0)), .025, .10)
            depth = self.tissue.surface(self.eyelet)-self.eyelet[2]
            gap = float(np.linalg.norm(self.eyelet-self.tip))
            lateral = float(np.linalg.norm(self.eyelet[:2]-site[:2]))
            self.check("thread left in tissue", depth > 1.0 and gap > 1.5,
                       {"loop_depth_mm": float(depth), "loop_to_tip_mm": gap, "placement_lateral_mm": lateral})

        return [settle, descend, pinch, peel, transport, align, release_pincher, insert, retract]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stop-after")
    args = parser.parse_args()
    started = datetime.now(timezone.utc).isoformat()
    cycle = ModernCycle()
    status = "completed"
    try:
        cycle.run(checkpoints=args.output/"checkpoints", stop_after=args.stop_after)
        if args.stop_after:
            status = f"stopped after {args.stop_after} (requested)"
    except (RuntimeError, FloatingPointError) as error:
        status = f"stopped: {error}"
    finally:
        report = e2e.save(cycle, args.output, status, started)
    print(report["status"], flush=True)


if __name__ == "__main__":
    main()
