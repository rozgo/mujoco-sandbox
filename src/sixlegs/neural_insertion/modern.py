"""End-to-end cycle, design v3: thread the loop, latch, peel, align, insert fast, snap back.

Same runner machinery as the first pass (servo, measured checks, checkpoints,
loss and instability stops); scripted yardstick motions, no disturbances.
The Z stage carries the tool for approach and transport with the needle held
at Q_PICK; only the insertion carriage strokes into tissue, out of the cannula,
and back into it. The latch is spring-closed: the wire's tension ramps to open
it and ramps off to let the spring swing the fork in under the loop.
Tissue: the phantom with the needle-tissue force model (no tube).

Usage: uv run --locked python -m sixlegs.neural_insertion.modern --output DIR [--stop-after STEP]
"""

import argparse
from datetime import datetime, timezone
from pathlib import Path

import mujoco
import numpy as np

from . import e2e
from .modern_scene import JOINTS, OPEN_STOP, Q_PICK, SEATED, WIRE_OPEN, load_modern
from .motion import Servo
from .tissue import Tissue, TissueParams

INSERT_DEPTH = 2.6       # mm: point below the surface at full stroke; the shoulder (and loop) about 2.0 mm deep
RETRACT_INTO = 1.0       # mm: the stroke ends this far above Q_PICK, the point 0.3 mm inside the cannula
CONTAINED = .015         # mm: loop centre to needle axis while seated and during insertion
ON_NEEDLE = .020


class ModernCycle(e2e.Cycle):
    def __init__(self, target=0, fast=False):
        self.m, self.d, meta = load_modern(fast=fast)
        if fast:
            self.servo_every = 1                                     # servo every 50 µs step
            self.record_every = round(1e-3/self.m.opt.timestep)      # replay state every 1 ms
        self.meta, self.target = meta, target
        self.qadr = np.array([self.m.jnt_qposadr[self.m.joint(n).id] for n in JOINTS])
        scratch = mujoco.MjData(self.m)
        mujoco.mj_kinematics(self.m, scratch)
        self.tip0 = scratch.site("needle_tip").xpos.copy()
        self.servo = Servo(self.m, names=JOINTS)
        self.tissue = Tissue(self.m, meta["thread_bodies"], TissueParams())
        self.trace = {k: [] for k in ("time", "qpos", "tip", "eyelet", "depth", "needle_axial", "act_force",
                                       "ncon", "phase")}
        self.events, self.q_ref, self.prohibited, self.done = [], None, [], []
        self.robot = {self.m.body(n).id for n in (*JOINTS, "cartridge", "latch")}
        self.phantom = self.m.geom("tissue_phantom").id
        self.obstacles = {self.m.geom(n).id for n in ("tissue_phantom", "backing_film", "film_support",
                                                      "implant_carrier")}
        self.clamp, self.on_step, self.aux = None, None, None
        self.wire = self.m.actuator("latch_wire").id
        self.latch = {g for g in range(self.m.ngeom) if self.m.geom(g).name.startswith("latch_")}
        self.latch_q = self.m.jnt_qposadr[self.m.joint("latch").id]
        self.ring = {g for g in range(self.m.ngeom) if self.m.geom(g).name.startswith("eyelet_")}
        self.worst = {}

    # ------------------------------------------------------------ geometry
    def joints(self, tip, stroke=0.):
        """Point at `tip` with the insertion carriage at Q_PICK+stroke; the Z stage takes up the rest."""
        q = np.zeros(len(JOINTS))
        q[0], q[1] = tip[0]-self.tip0[0], tip[1]-self.tip0[1]
        q[3] = Q_PICK+stroke
        q[2] = tip[2]-self.tip0[2]+q[3]
        return q

    def tip_of(self, q):
        return np.array((self.tip0[0]+q[0], self.tip0[1]+q[1], self.tip0[2]+q[2]-q[3]))

    def offset(self):
        """Loop centre to the needle's axis, mm."""
        return float(np.linalg.norm(self.eyelet[:2]-self.tip[:2]))

    def seat_height(self):
        return float(self.eyelet[2]-self.tip[2])

    def latch_load(self):
        """Normal force between the latch and the loop, µN."""
        f, total = np.zeros(6), 0.
        for i, c in enumerate(self.d.contact[:self.d.ncon]):
            if (c.geom1 in self.latch and c.geom2 in self.ring) or (c.geom2 in self.latch and c.geom1 in self.ring):
                mujoco.mj_contactForce(self.m, self.d, i, f)
                total += f[0]
        return total

    def wire_ramp(self, start, end, duration):
        def command(t):
            s = min(t/duration, 1.)
            self.d.ctrl[self.wire] = start+(end-start)*(10*s**3-15*s**4+6*s**5)
        return command

    def watch(self, label):
        """Observer for the worst loop-to-axis offset during a phase (every 50 steps, 250 µs)."""
        def observe(name):
            self.worst[label] = max(self.worst.get(label, 0.), self.offset())
            self.worst[label+"_low"] = min(self.worst.get(label+"_low", np.inf), self.seat_height())
        return observe

    # ------------------------------------------------------------ cycle
    def start(self):
        loop = self.eyelet
        q = self.joints(np.array((loop[0], loop[1], loop[2]+.3)))
        self.d.qpos[self.qadr] = q  # initialization, before any stepping
        self.d.ctrl[:] = 0
        self.d.ctrl[self.servo.act] = q
        self.d.qpos[self.latch_q] = OPEN_STOP
        self.d.ctrl[self.wire] = WIRE_OPEN
        mujoco.mj_forward(self.m, self.d)
        self.q_ref = q
        return q

    def steps(self):
        site = self.target_site()
        surface = self.tissue.surface(site)
        loop0 = self.eyelet.copy()

        def at(xyz, stroke=0.):
            return self.joints(np.asarray(xyz, float), stroke)

        def settle():
            self.move("settle", self.start(), .01)

        def descend():
            # Point down through the loop's centre; the shoulder presses the loop 10 µm down so it sits against it.
            self.move("descend", at((loop0[0], loop0[1], loop0[2]-SEATED-.010)), .10, .01)

        def latch():
            # Wire tension ramps off; the preloaded spring swings the fork in under the loop, onto its stop.
            self.aux = self.wire_ramp(WIRE_OPEN, 0., .03)
            self.move("latch", self.q_ref, 0., .05)
            self.aux = None
            self.check("loop latched on the needle", self.offset() < ON_NEEDLE and abs(self.seat_height()-SEATED) < .015
                       and self.d.qpos[self.latch_q] < .01,
                       {"loop_to_axis_mm": self.offset(), "loop_above_point_mm": self.seat_height(),
                        "seated_mm": SEATED, "latch_rad": float(self.d.qpos[self.latch_q])})

        def peel():
            # Up 1.5 mm while moving 1 mm toward the anchor, so the thread never goes taut.
            tip = self.tip_of(self.q_ref)
            self.move("peel", at((tip[0]+1.0, tip[1], tip[2]+1.5)), .10, .02, carried=True)
            lifted = self.eyelet[2]-loop0[2]
            self.check("loop lifted, still latched", lifted > 1.0 and self.offset() < ON_NEEDLE
                       and abs(self.seat_height()-SEATED) < .03,
                       {"loop_lift_mm": float(lifted), "loop_to_axis_mm": self.offset(),
                        "loop_above_point_mm": self.seat_height(), "latch_load_uN": self.latch_load()})

        def transport():
            # Back over the implant carrier to the target at carry height; the thread folds into a slack U.
            tip = self.tip_of(self.q_ref)
            self.move("transport", at((site[0], site[1], tip[2])), .25, .03, carried=True)

        def align():
            # Point down to 0.3 mm above the surface.
            self.move("align", at((site[0], site[1], surface+.3)), .10, .02, carried=True)
            lateral = float(np.linalg.norm(self.tip[:2]-site[:2]))
            self.check("aligned, loop still latched", lateral < .02 and abs(self.seat_height()-SEATED) < .03,
                       {"tip_to_target_lateral_mm": lateral, "loop_to_axis_mm": self.offset(),
                        "loop_above_point_mm": self.seat_height(), "latch_rad": float(self.d.qpos[self.latch_q])})

        def release_latch():
            self.aux = self.wire_ramp(0., WIRE_OPEN, .02)
            self.move("open latch", self.q_ref, 0., .03, carried=True)
            self.aux = None
            self.check("latch open, loop on the needle", self.offset() < ON_NEEDLE and self.latch_load() == 0.,
                       {"loop_to_axis_mm": self.offset(), "loop_above_point_mm": self.seat_height(),
                        "latch_rad": float(self.d.qpos[self.latch_q])})

        def insert():
            # The carriage alone strokes out of the cannula: point to 2.6 mm deep in 25 ms (peak about 210 mm/s).
            q = self.q_ref.copy()
            q[3] = q[2]-(surface-INSERT_DEPTH-self.tip0[2])
            self.on_step = self.watch("insert")
            self.move("insert", q, .025, .02, carried=True)
            self.on_step = None
            depth = self.tissue.surface(self.eyelet)-self.eyelet[2]
            worst = self.worst["insert"]
            self.check("loop inserted on the needle's axis", depth > 1.5 and worst < CONTAINED,
                       {"loop_depth_mm": float(depth), "punctured": self.tissue.state.punctured,
                        "worst_loop_to_axis_mm": worst, "lowest_loop_above_point_mm": self.worst["insert_low"],
                        "peak_needle_axial_uN": self.tissue.state.peak_axial})

        def retract():
            # Snap back into the cannula: about 3.9 mm in 30 ms, peak acceleration about 25,000 mm/s^2.
            q = self.q_ref.copy()
            q[3] = Q_PICK-RETRACT_INTO
            self.move("retract", q, .03, .10)
            depth = self.tissue.surface(self.eyelet)-self.eyelet[2]
            gap = float(np.linalg.norm(self.eyelet-self.tip))
            lateral = float(np.linalg.norm(self.eyelet[:2]-site[:2]))
            self.check("thread left in tissue", depth > 1.5 and gap > 1.5 and lateral < .025,
                       {"loop_depth_mm": float(depth), "loop_to_tip_mm": gap, "placement_lateral_mm": lateral})

        return [settle, descend, latch, peel, transport, align, release_latch, insert, retract]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stop-after")
    parser.add_argument("--fast", action="store_true", help="fast thread settings (50 µs step; modern_scene.FAST)")
    args = parser.parse_args()
    started = datetime.now(timezone.utc).isoformat()
    cycle = ModernCycle(fast=args.fast)
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
