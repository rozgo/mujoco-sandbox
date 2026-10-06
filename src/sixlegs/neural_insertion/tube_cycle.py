"""Thread-tube cycle over several sites: stick to the waiting thread end, insert, let go at depth, snap back.

At each site:
1. The needle strokes down from its ready position; when its point reaches the thread's end, a MuJoCo
   connect constraint bonds them (abstracted sticking) and the tube releases its hold on the thread.
2. The needle carries the thread out of the tube into the tissue (needle-tissue force model, tissue.py).
3. At depth the bond switches off (abstracted release); the needle snaps back up.
4. The tool lifts clear of the thread it left standing.
Between sites a spare thread is reloaded into the tube (an explicit reset: it appears there, straight and at
rest), and the tool moves over the next site and descends. Placed threads stay simulated in the tissue.
Scripted yardstick motions, no disturbances, fast thread settings (modern_scene.FAST).

Usage: uv run --locked python -m sixlegs.neural_insertion.tube_cycle --output DIR [--sites 0 5 1]
"""

import argparse
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import mujoco
import numpy as np

from . import e2e
from .motion import Servo
from .tissue import Tissue, TissueParams, TissueState
from .tube_scene import END_BELOW_POINT, INSERTED, Q_READY, SEGMENT, review_pose

JOINTS = ("stage_x", "stage_y", "stage_z", "insertion")
DEPTH = 2.0          # mm: thread end below the surface at full stroke
STICK = .030         # mm: needle point within this of the thread's end forms the bond
LIFT = 8.0           # mm: tool lift after the snap back, clear of the ~6 mm of thread left standing
SITES = (0, 5, 1)


class TubeTissue(Tissue):
    """Needle forces from tissue.py; thread grip as soft MuJoCo connect constraints that slip.

    A segment of the thread being inserted, below the surface near the needle's entry, is pinned where it is.
    When its constraint force exceeds the grip force (f_retain), the pin moves toward the segment until the
    force is back at that limit: the stick-slip rule of tissue.py, solved by MuJoCo instead of applied
    explicitly. A pinned segment stays pinned while it is below the surface, including after the needle has
    moved on to another site.
    """

    def __init__(self, model, threads, params, anchored):
        super().__init__(model, threads[0], params, end_site="thread_S_last_0", anchored=anchored, grip=False)
        self.grips = [[(model.body(n).id, model.equality(f"grip_{n}").id) for n in names[-anchored:]]
                      for names in threads]
        self.current = 0
        self.peak_grip = 0.

    def new_site(self):
        self.state = TissueState()  # the needle's puncture state starts over at each site

    def step(self, data):
        depth = super().step(data)
        m, p, s = self.m, self.p, self.state
        rows = (data.efc_type[:data.nefc] == mujoco.mjtConstraint.mjCNSTR_EQUALITY)
        for k, grips in enumerate(self.grips):
            for b, eq in grips:
                point = data.xipos[b]
                if self.surface(point)-point[2] <= 0:
                    data.eq_active[eq] = 0
                    continue
                if not data.eq_active[eq]:
                    near = s.entry is not None and np.linalg.norm(point[:2]-s.entry[:2]) < p.track_radius
                    if k == self.current and near:
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
    def __init__(self, sites=SITES):
        self.m, self.d, meta = review_pose(sites[0])
        self.meta, self.sites, self.target = meta, tuple(sites), sites[0]
        self.threads = meta["threads"]
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
        self.tissue = TubeTissue(self.m, self.threads, params, INSERTED)
        self.trace = {k: [] for k in ("time", "qpos", "tip", "eyelet", "depth", "needle_axial", "act_force",
                                       "ncon", "phase")}
        self.events, self.q_ref, self.prohibited, self.done = [], None, [], []
        self.robot = {self.m.body(n).id for n in (*JOINTS, "thread_tube")}
        self.phantom = self.m.geom("tissue_phantom").id
        self.obstacles = {self.phantom}
        self.clamp, self.on_step, self.aux = None, None, None
        self.tube = self.m.body("thread_tube").id
        self.k, self.bonded_at = 0, None
        self.placed = []  # (site, thread) inserted so far

    # ------------------------------------------------------------ thread bookkeeping
    def eq(self, name, k=None):
        return self.m.equality(f"{name}_{self.k if k is None else k}").id

    def end_of(self, k):
        return self.d.site(f"thread_S_last_{k}").xpos.copy()

    @property
    def eyelet(self):  # the current thread's end (name kept for the shared runner and trace)
        return self.end_of(self.k)

    def connect(self, eq, point):
        """Switch on a connect constraint at a world point, anchored where its two bodies are now."""
        for j, b in enumerate((self.m.eq_obj1id[eq], self.m.eq_obj2id[eq])):
            self.m.eq_data[eq, 3*j:3*j+3] = self.d.xmat[b].reshape(3, 3).T@(point-self.d.xpos[b])
        self.d.eq_active[eq] = 1

    def joints(self, tip, stroke=0.):
        q = np.zeros(len(JOINTS))
        q[0], q[1] = tip[0]-self.tip0[0], tip[1]-self.tip0[1]
        q[3] = Q_READY+stroke
        q[2] = tip[2]-self.tip0[2]+q[3]
        return q

    def tip_of(self, q):
        return np.array((self.tip0[0]+q[0], self.tip0[1]+q[1], self.tip0[2]+q[2]-q[3]))

    def site(self, i):
        return self.d.site(f"target_{i}").xpos.copy()

    # ------------------------------------------------------------ cycle
    def start(self):
        self.d.qpos[:] = self.state0  # initialization of the review pose
        self.d.qvel[:] = 0
        self.d.ctrl[:] = 0
        self.d.ctrl[self.servo.act] = self.q_start
        mujoco.mj_forward(self.m, self.d)
        self.tube0 = self.d.xpos[self.tube].copy()
        self.roots0 = [self.d.qpos[self.m.jnt_qposadr[self.m.joint(f"thread_root_{k}").id]:][:7].copy()
                       for k in range(len(self.threads))]
        self.connect(self.eq("tube_hold"), self.d.xpos[self.m.body(self.threads[0][0]).id].copy())
        mujoco.mj_forward(self.m, self.d)
        self.q_ref = self.q_start.copy()
        return self.q_ref

    def reload(self, k):
        """Explicit reset: spare thread k appears in the tube, straight and at rest, where thread 0 waited."""
        free = self.m.joint(f"thread_root_{k}")
        pose = self.roots0[0].copy()
        pose[:3] += self.d.xpos[self.tube]-self.tube0
        self.d.qpos[free.qposadr[0]:free.qposadr[0]+7] = pose
        for n in self.threads[k]:
            b = self.m.body(n).id
            self.m.body_gravcomp[b] = 0.
            if self.m.body_jntnum[b] and self.m.jnt_type[self.m.body_jntadr[b]] == mujoco.mjtJoint.mjJNT_BALL:
                a = self.m.jnt_qposadr[self.m.body_jntadr[b]]
                self.d.qpos[a:a+4] = (1., 0., 0., 0.)
            self.d.qvel[self.m.body_dofadr[b]:self.m.body_dofadr[b]+self.m.body_dofnum[b]] = 0.
        mujoco.mj_forward(self.m, self.d)
        self.k, self.bonded_at = k, None
        self.tissue.current = k
        self.connect(self.eq("tube_hold"), self.d.xpos[self.m.body(self.threads[k][0]).id].copy())
        mujoco.mj_forward(self.m, self.d)

    def stick(self, t):
        """Each step while descending: bond when the needle point reaches the thread's end."""
        if self.bonded_at is None and np.linalg.norm(self.tip-self.eyelet) < STICK:
            self.connect(self.eq("needle_bond"), self.eyelet)
            self.d.eq_active[self.eq("tube_hold")] = 0  # the tube lets the thread go once the needle has it
            self.bonded_at = float(self.d.time)

    def steps(self):
        steps = []

        def named(fn, name):
            fn.__name__ = name
            steps.append(fn)

        for n, i in enumerate(self.sites):
            k = n

            def settle():
                self.move("ready", self.start(), .01)

            def reload(k=k, i=i):
                self.reload(k)
                self.move(f"reload thread {k}", self.q_ref, 0., .01)

            def move_over(i=i):
                # Lateral at the lifted height, then down to the ready pose over the next site.
                before = [self.end_of(j) for _, j in self.placed]
                tip = self.tip_of(self.q_ref)
                site = self.site(i)
                self.move(f"move to site {i}", self.joints((site[0], site[1], tip[2])), .20, .01)
                moved = max((float(np.linalg.norm(self.end_of(j)-e)) for (_, j), e in zip(self.placed, before)),
                            default=0.)
                self.check(f"placed threads undisturbed by the move to site {i}", moved < .01,
                           {"largest_end_shift_mm": moved})

            def descend(i=i):
                site = self.site(i)
                surface = self.tissue.surface(site)
                self.tissue.new_site()
                self.move(f"descend at site {i}", self.joints((site[0], site[1], surface+1.+END_BELOW_POINT)), .10, .01)

            def pick(i=i):
                q = self.q_ref.copy()
                q[3] += END_BELOW_POINT+.02
                self.aux = self.stick
                self.move(f"needle down {i}", q, .04, .005)
                self.aux = None
                self.check(f"site {i}: needle stuck to the thread end", self.bonded_at is not None,
                           {"bonded_at_s": self.bonded_at, "point_to_end_mm": float(np.linalg.norm(self.tip-self.eyelet))})

            def insert(i=i):
                surface = self.tissue.surface(self.site(i))
                q = self.q_ref.copy()
                q[3] = q[2]-(surface-DEPTH-self.tip0[2])
                self.move(f"insert {i}", q, .025, .03, carried=True)  # 30 ms at depth to settle
                depth = surface-self.eyelet[2]
                self.check(f"site {i}: thread end at depth", depth > 1.8,
                           {"end_depth_mm": float(depth), "punctured": self.tissue.state.punctured,
                            "peak_needle_axial_uN": self.tissue.state.peak_axial})

            def release(i=i):
                self.d.eq_active[self.eq("needle_bond")] = 0  # abstracted release
                self.move(f"release {i}", self.q_ref, 0., .01)

            def retract(i=i, k=k):
                q = self.q_ref.copy()
                q[3] = Q_READY
                self.move(f"snap back {i}", q, .03, .05)
                site = self.site(i)
                depth = self.tissue.surface(site)-self.eyelet[2]
                gap = float(np.linalg.norm(self.eyelet-self.tip))
                lateral = float(np.linalg.norm(self.eyelet[:2]-site[:2]))
                self.check(f"site {i}: thread left in tissue", depth > 1.5 and gap > 1.5,
                           {"end_depth_mm": float(depth), "end_to_point_mm": gap, "placement_lateral_mm": lateral})
                self.placed.append((i, k))

            def lift(i=i):
                q = self.q_ref.copy()
                q[2] += LIFT
                self.move(f"lift at site {i}", q, .15, .05)
                worst = min(self.tissue.surface(self.site(s))-self.end_of(j)[2] for s, j in self.placed)
                self.check(f"after site {i}: every placed thread still in the tissue", worst > 1.5,
                           {"shallowest_end_depth_mm": float(worst)})

            if n == 0:
                named(settle, "settle")
            else:
                named(reload, f"reload_{k}")
                named(move_over, f"move_to_{i}")
                named(descend, f"descend_{i}")
            for fn, name in ((pick, "pick"), (insert, "insert"), (release, "release"), (retract, "retract"),
                             (lift, "lift")):
                named(fn, f"{name}_{i}")
        return steps


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sites", type=int, nargs="+", default=list(SITES))
    args = parser.parse_args()
    started = datetime.now(timezone.utc).isoformat()
    cycle = TubeCycle(args.sites)
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
