"""Thread-tube cycle: the needle sticks to the waiting thread end, inserts it, lets go at depth, snaps back.

1. The needle strokes down from its ready position; when its point reaches the thread's end, a MuJoCo
   connect constraint bonds them (abstracted sticking) and the tube releases its hold on the thread.
2. The needle carries the thread out of the tube into the tissue (needle-tissue force model, tissue.py).
3. At depth the bond switches off (abstracted release); the needle snaps back up.
4. The tool lifts away; the tube slides off the rest of the thread, which stays in the tissue.
Scripted yardstick motions, no disturbances, fast thread settings (modern_scene.FAST).

Usage: uv run --locked python -m sixlegs.neural_insertion.tube_cycle --output DIR
"""

import argparse
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import mujoco
import numpy as np

from . import e2e
from .motion import Servo
from .tissue import Tissue, TissueParams
from .tube_scene import END_BELOW_POINT, INSERTED, Q_READY, SEGMENT, review_pose

JOINTS = ("stage_x", "stage_y", "stage_z", "insertion")
DEPTH = 2.0          # mm: thread end below the surface at full stroke
STICK = .030         # mm: needle point within this of the thread's end forms the bond
LIFT = 5.0           # mm: tool lift after the snap back


class TubeTissue(Tissue):
    """Needle forces from tissue.py; thread grip as soft MuJoCo connect constraints that slip.

    A segment below the surface near the needle's entry is pinned where it is. When its constraint force
    exceeds the grip force (f_retain), the pin moves toward the segment until the force is back at that
    limit: the same stick-slip rule as tissue.py, solved by MuJoCo instead of applied explicitly.
    """

    def __init__(self, model, names, params, anchored):
        super().__init__(model, names, params, end_site="thread_S_last", anchored=anchored, grip=False)
        self.eqs = [model.equality(f"grip_{n}").id for n in names[-anchored:]]
        self.peak_grip = 0.

    def step(self, data):
        depth = super().step(data)
        m, p, s = self.m, self.p, self.state
        rows = (data.efc_type[:data.nefc] == mujoco.mjtConstraint.mjCNSTR_EQUALITY)
        for b, eq in zip(self.bodies, self.eqs):
            point = data.xipos[b]
            near = s.entry is not None and np.linalg.norm(point[:2]-s.entry[:2]) < p.track_radius
            if self.surface(point)-point[2] <= 0 or not near:
                data.eq_active[eq] = 0
                continue
            if not data.eq_active[eq]:
                m.eq_data[eq, 0:3] = point
                m.eq_data[eq, 3:6] = data.xmat[b].reshape(3, 3).T@(point-data.xpos[b])
                data.eq_active[eq] = 1
                continue
            force = float(np.linalg.norm(data.efc_force[:data.nefc][rows & (data.efc_id[:data.nefc] == eq)]))
            self.peak_grip = max(self.peak_grip, force)
            if force > p.f_retain:  # slip: move the pin toward the segment
                anchor = m.eq_data[eq, 0:3].copy()
                m.eq_data[eq, 0:3] = point+(anchor-point)*p.f_retain/force
        return depth


class TubeCycle(e2e.Cycle):
    def __init__(self, target=0):
        self.m, self.d, meta = review_pose(target)
        self.meta, self.target = meta, target
        self.qadr = np.array([self.m.jnt_qposadr[self.m.joint(n).id] for n in JOINTS])
        self.q_start = self.d.qpos[self.qadr].copy()
        self.state0 = self.d.qpos.copy()
        scratch = mujoco.MjData(self.m)
        mujoco.mj_kinematics(self.m, scratch)
        self.tip0 = scratch.site("needle_tip").xpos.copy()
        self.servo = Servo(self.m, names=JOINTS)
        self.servo_every, self.record_every = 1, round(1e-3/self.m.opt.timestep)
        # Tissue grip per unit length: TissueParams' anchor stiffness and slip force are per 1.375 mm segment.
        base, scale = TissueParams(), SEGMENT/1.375
        params = replace(base, k_anchor=base.k_anchor*scale, f_retain=base.f_retain*scale)
        self.tissue = TubeTissue(self.m, meta["thread_bodies"], params, INSERTED)
        self.trace = {k: [] for k in ("time", "qpos", "tip", "eyelet", "depth", "needle_axial", "act_force",
                                       "ncon", "phase")}
        self.events, self.q_ref, self.prohibited, self.done = [], None, [], []
        self.robot = {self.m.body(n).id for n in (*JOINTS, "thread_tube")}
        self.phantom = self.m.geom("tissue_phantom").id
        self.obstacles = {self.phantom}
        self.clamp, self.on_step, self.aux = None, None, None
        self.hold, self.bond = self.m.equality("tube_hold").id, self.m.equality("needle_bond").id
        self.first, self.last = (self.m.body(meta["thread_bodies"][i]).id for i in (0, -1))
        self.bonded_at = None

    @property
    def eyelet(self):  # the thread's end (name kept for the shared runner and trace)
        return self.d.site("thread_S_last").xpos.copy()

    def joints(self, tip, stroke=0.):
        q = np.zeros(len(JOINTS))
        q[0], q[1] = tip[0]-self.tip0[0], tip[1]-self.tip0[1]
        q[3] = Q_READY+stroke
        q[2] = tip[2]-self.tip0[2]+q[3]
        return q

    def tip_of(self, q):
        return np.array((self.tip0[0]+q[0], self.tip0[1]+q[1], self.tip0[2]+q[2]-q[3]))

    def connect(self, eq, point):
        """Switch on a connect constraint at a world point, anchored where its two bodies are now."""
        b1, b2 = self.m.eq_obj1id[eq], self.m.eq_obj2id[eq]
        for k, b in enumerate((b1, b2)):
            self.m.eq_data[eq, 3*k:3*k+3] = self.d.xmat[b].reshape(3, 3).T@(point-self.d.xpos[b])
        self.d.eq_active[eq] = 1

    def start(self):
        self.d.qpos[:] = self.state0  # initialization of the review pose
        self.d.qvel[:] = 0
        self.d.ctrl[:] = 0
        self.d.ctrl[self.servo.act] = self.q_start
        mujoco.mj_forward(self.m, self.d)
        self.connect(self.hold, self.d.xpos[self.first].copy())
        mujoco.mj_forward(self.m, self.d)
        self.q_ref = self.q_start.copy()
        return self.q_ref

    def stick(self, t):
        """Each step while descending: bond when the needle point reaches the thread's end."""
        if self.bonded_at is None and np.linalg.norm(self.tip-self.eyelet) < STICK:
            self.connect(self.bond, self.eyelet)
            self.d.eq_active[self.hold] = 0  # the tube lets the thread go once the needle has it
            self.bonded_at = float(self.d.time)

    def steps(self):
        site = self.target_site()
        surface = self.tissue.surface(site)

        def settle():
            self.move("ready", self.start(), .01)

        def pick():
            # Needle strokes down onto the thread's end and 20 µm past it; the bond forms on reaching it.
            q = self.q_ref.copy()
            q[3] += END_BELOW_POINT+.02
            self.aux = self.stick
            self.move("needle down", q, .04, .005)
            self.aux = None
            self.check("needle stuck to the thread end", self.bonded_at is not None,
                       {"bonded_at_s": self.bonded_at, "point_to_end_mm": float(np.linalg.norm(self.tip-self.eyelet))})

        def insert():
            # Fast stroke: the thread's end to 2.0 mm below the surface in 25 ms.
            q = self.q_ref.copy()
            q[3] = q[2]-(surface-DEPTH-self.tip0[2])
            self.move("insert", q, .025, .01, carried=True)
            depth = surface-self.eyelet[2]
            self.check("thread end at depth", depth > 1.8, {"end_depth_mm": float(depth),
                                                              "punctured": self.tissue.state.punctured})

        def release():
            self.d.eq_active[self.bond] = 0  # abstracted release
            self.move("release", self.q_ref, 0., .01)

        def retract():
            q = self.q_ref.copy()
            q[3] = Q_READY
            self.move("snap back", q, .03, .05)
            depth = self.tissue.surface(self.eyelet)-self.eyelet[2]
            gap = float(np.linalg.norm(self.eyelet-self.tip))
            self.check("thread left in tissue", depth > 1.5 and gap > 1.5, {"end_depth_mm": float(depth),
                                                                            "end_to_point_mm": gap})

        def lift():
            q = self.q_ref.copy()
            q[2] += LIFT
            self.move("lift away", q, .15, .10)
            depth = self.tissue.surface(self.eyelet)-self.eyelet[2]
            lateral = float(np.linalg.norm(self.eyelet[:2]-site[:2]))
            self.check("thread stays after the tool leaves", depth > 1.5,
                       {"end_depth_mm": float(depth), "placement_lateral_mm": lateral,
                        "thread_root_z_above_surface_mm": float(self.d.xpos[self.first][2]-surface)})

        return [settle, pick, insert, release, retract, lift]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = datetime.now(timezone.utc).isoformat()
    cycle = TubeCycle()
    status = "completed"
    try:
        cycle.run(checkpoints=args.output/"checkpoints")
    except (RuntimeError, FloatingPointError) as error:
        status = f"stopped: {error}"
    finally:
        report = e2e.save(cycle, args.output, status, started)
    print(report["status"], flush=True)


if __name__ == "__main__":
    main()
