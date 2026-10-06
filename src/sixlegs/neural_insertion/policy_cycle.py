"""Learned insertion policy in the full thread-tube simulation: the check on the C training environment.

At each site the trained policy (insert_core.h, PufferLib 5.0) steers the needle point from where the tool
is: stage X, Y and Z velocity commands at 50 Hz, smoothed and integrated into the servo references as in
the C environment. Its 24 observations are built here from the full simulation's own sensing
(disturbance.py): the delayed, noisy needle point and velocity, the measured target, the thread end's
offset from the needle's axis as a camera would measure it (the DER thread in its tube, not a drawn
offset), the nearest vessel, the height above the dome and the two nearest placed threads. When the policy
starts the stroke, the stage references are held and the approved cycle's programmed stroke runs
(tube_cycle.py: needle down onto the thread end, bond, insert, 30 ms at depth, release, snap back), then
the tool lifts and the next thread is reloaded. The C environment abstracts that stroke as one 50 ms
minimum-jerk motion with 10 ms at depth.

Failures stop the run with the partial record: no bond (thread end too far off the needle's axis), the
waiting thread end dragged into the tissue, the needle, tube or waiting thread touching a placed thread,
robot contact with the phantom, the 6 s approach deadline, instability.

Usage: uv run --locked python -m sixlegs.neural_insertion.policy_cycle (--weights W.bin | --scripted compensate)
           --output DIR [--sites 0 5 1] [--level 1 --seed 1001]
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np

from . import e2e
from .align_policy import Policy
from .disturbance import VEL_NOISE
from .rl_scene import vessel_polylines
from .tube_cycle import JOINTS, LIFT, SITES, TubeCycle
from .tube_scene import Q_READY

# insert_core.h / surgical_core.h, SI
VMAX = np.array((.04, .04, .02))   # m/s at |action| = 1 (X, Y, Z stage)
FILTER_S = .01                     # reference velocity smoothing
REPEAT_S = .020                    # policy period
HORIZON = 300                      # policy steps per site (6 s)
READY = 2e-3                       # hover: needle point above the surface
END_NOISE = 2e-6                   # thread end measurement noise per axis
TRIGGER = .5
VESSEL_HEIGHT, VESSEL_MARGIN = 3e-3, 3e-4


class Scripted:
    """insert_core.h si_scripted on the same observation: compensate=False aims the needle at the target
    (the approved cycle's yardstick), True aims so the thread end lands on it. Programmed control."""

    def __init__(self, compensate):
        self.compensate = compensate
        self.label = f"scripted yardstick ({'compensating' if compensate else 'aiming the needle'})"

    def act(self, obs, first):
        gain = 6.
        fine = np.abs(obs[10:13]) < 1
        e = np.where(fine, obs[10:13]*1e-4, obs[0:3]*.005).astype(float)
        if not self.compensate:
            e[:2] -= obs[13:15]*50e-6
        v = -gain*e
        lateral, height = float(np.hypot(e[0], e[1])), float(obs[17])*.01
        if lateral > 5e-4 and gain*(VESSEL_HEIGHT-height) > v[2]:
            v[2] = gain*(VESSEL_HEIGHT-height)
        action = np.zeros(4, np.float32)
        action[:3] = np.cbrt(np.clip(v, -VMAX, VMAX)/VMAX)
        speed = float(np.linalg.norm(obs[3:6]))*VMAX[0]
        action[3] = float(lateral < 5e-6 and abs(e[2]) < 2e-5 and speed < 2e-4)
        return action


class Learned:
    def __init__(self, weights, hidden=128, layers=2):
        self.policy = Policy(weights, hidden, layers, obs=24, actions=4)
        self.label = "learned policy (approach and stroke trigger) with the programmed stroke"
        self.weights, self.weights_sha256 = str(weights), self.policy.weights_sha256

    def act(self, obs, first):
        return self.policy.act(obs, first)


class PolicyCycle(TubeCycle):
    def __init__(self, agent, sites=SITES, level=0., seed=0):
        super().__init__(sites, level, seed)
        self.policy = agent
        self.vel_hist = []
        self.thread_touches = []
        self.approaches = []
        self.placed_rest = []  # placed thread ends, tissue at rest (xy, mm)
        self.thread_geoms = [{g for g in range(self.m.ngeom) if self.m.geom_bodyid[g] in
                              {self.m.body(n).id for n in names}} for names in self.threads]
        self.robot_geoms = {g for g in range(self.m.ngeom) if self.m.geom_bodyid[g] in self.robot}
        self.vessels = None

    # ------------------------------------------------------------ sensing
    def disturb(self):
        super().disturb()
        q = self.d.qvel
        dof = self.dofs
        vel = np.array((q[dof["stage_x"]], q[dof["stage_y"]], q[dof["stage_z"]]-q[dof["insertion"]]))
        self.vel_hist.append((self.d.time, vel))
        while len(self.vel_hist) > 2 and self.vel_hist[1][0] <= self.d.time-self.dist.latency-1e-12:
            self.vel_hist.pop(0)

    def measure_vel(self):
        vel = self.vel_hist[0][1].copy() if self.vel_hist else np.zeros(3)
        for t, v in self.vel_hist:
            if t <= self.d.time-self.dist.latency+1e-12:
                vel = v.copy()
        if self.dist.level > 0:
            vel += VEL_NOISE*self.dist.sensing*self.dist.rng.normal(size=3)
        return vel

    def vessel(self, x, y):
        """Offset to the nearest vessel edge's centreline point and the clearance (mm), tissue at rest."""
        best, off = 1e9, np.zeros(2)
        p = np.array((x, y))
        for a, b, r in self.vessels:
            e = b-a
            t = np.clip((p-a)@e/(e@e), 0, 1) if e@e > 0 else 0.
            v = a+t*e-p
            dist = np.linalg.norm(v)-r
            if dist < best:
                best, off = dist, v
        return off, best

    def surface_rest(self, x, y):
        o = self.tissue.offset
        self.tissue.offset = np.zeros(3)
        z = self.tissue.surface((x, y, 0.))
        self.tissue.offset = o
        return z

    def observe(self, i, tick, prev):
        """insert_core.h si_observe, from the full simulation's sensing (mm here, SI in the observation)."""
        tip = self.dist.measure_tip(self.d.time)
        vel = self.measure_vel()
        goal = self.measured_site(i)
        self.last_tip_meas = tip.copy()
        s = self.dist.sensing if self.dist.level > 0 else .5
        delta = (self.eyelet[:2]-self.tip[:2])*1e-3+END_NOISE*s*self.dist.rng.normal(size=2)  # camera, m
        hover = np.array((goal[0]*1e-3-delta[0], goal[1]*1e-3-delta[1], (goal[2]-self.site_lift[i])*1e-3+READY))
        e = tip*1e-3-hover
        obs = np.zeros(24, np.float32)
        obs[0:3] = np.clip(e/.005, -5, 5)
        obs[3:6] = np.clip(vel*1e-3/VMAX[0], -2, 2)
        obs[6:10] = prev
        obs[10:13] = np.clip(e/1e-4, -1, 1)
        obs[13:15] = np.clip(delta/50e-6, -2, 2)
        off, _ = self.vessel(tip[0], tip[1])
        obs[15:17] = np.clip(off*1e-3/.005, -1, 1)
        obs[17] = np.clip((tip[2]-self.surface_rest(tip[0], tip[1]))*1e-3/.01, -1, 2)
        near = sorted(self.placed_rest, key=lambda p: np.hypot(p[0]-tip[0], p[1]-tip[1]))[:2]
        obs[18:22] = 1.
        for k, p in enumerate(near):
            obs[18+2*k:20+2*k] = np.clip((p-tip[:2])*1e-3/.005, -1, 1)
        obs[22] = 0.
        obs[23] = 1.-tick/HORIZON
        return obs, {"hover_error_um": (e*1e6).tolist(), "delta_um": (delta*1e6).tolist()}

    def touches(self):
        """Contacts of the needle, tube or waiting thread with a placed thread."""
        placed = set().union(*(self.thread_geoms[k] for _, k in self.placed)) if self.placed else set()
        mine = self.robot_geoms | self.thread_geoms[self.k]
        for c in self.d.contact[:self.d.ncon]:
            if (c.geom1 in placed and c.geom2 in mine) or (c.geom2 in placed and c.geom1 in mine):
                return self.m.geom(c.geom1).name, self.m.geom(c.geom2).name
        return None

    # ------------------------------------------------------------ the learned approach
    def approach(self, i):
        """The policy steers the stages until it starts the stroke; the stroke itself is programmed."""
        self.current_site = i
        self.tissue.new_site()
        dt = self.m.opt.timestep
        per = round(REPEAT_S/dt)
        q_ref = self.q_ref.copy()
        v_ref = np.zeros(4)
        lo = self.m.jnt_range[[self.m.joint(n).id for n in JOINTS], 0]+.2
        hi = self.m.jnt_range[[self.m.joint(n).id for n in JOINTS], 1]-.2
        prev = np.zeros(4, np.float32)
        name = f"policy approach {i}"
        started = time.perf_counter()
        record = {"site": i, "outcome": None, "vessel_steps": 0}
        for tick in range(HORIZON):
            obs, sensed = self.observe(i, tick, prev)
            action = self.policy.act(obs, tick == 0)
            applied = np.clip(action, -1, 1)
            if applied[3] > TRIGGER:
                true_site = self.site(i)
                hover = np.array((true_site[0]-(self.eyelet[0]-self.tip[0]), true_site[1]-(self.eyelet[1]-self.tip[1]),
                                  self.tissue.surface(true_site)+READY*1e3))
                record.update(outcome="triggered", policy_steps=tick, approach_s=tick*REPEAT_S,
                              trigger_true_hover_error_um=((self.tip-hover)*1e3).tolist(),
                              trigger_end_to_target_lateral_um=float(np.linalg.norm(self.eyelet[:2]-true_site[:2])*1e3),
                              **{f"trigger_{k}": v for k, v in sensed.items()})
                break
            v_cmd = np.zeros(4)
            v_cmd[:3] = applied[:3]**3*VMAX*1e3  # mm/s
            for k in range(per):
                nxt = v_ref+(v_cmd-v_ref)*dt/FILTER_S
                q = q_ref+nxt*dt
                out = (q < lo) | (q > hi)
                q, nxt[out] = np.clip(q, lo, hi), 0.
                a_ref = (nxt-v_ref)/dt
                v_ref, q_ref = nxt, q
                q_ref[3], v_ref[3], a_ref[3] = Q_READY, 0., 0.
                self.servo.command(self.d, q_ref, v_ref, a_ref)
                self.advance(name, k)
                if k % 20 == 0:
                    touch = self.touches()
                    if touch:
                        self.thread_touches.append({"time_s": float(self.d.time), "geoms": list(touch)})
            prev = applied.astype(np.float32)
            tip = self.tip
            off, clearance = self.vessel(tip[0], tip[1])
            if tip[2]-self.surface_rest(tip[0], tip[1]) < VESSEL_HEIGHT*1e3 and clearance < VESSEL_MARGIN*1e3:
                record["vessel_steps"] += 1
            failure = None
            if self.eyelet[2] < self.tissue.surface(self.eyelet):
                failure = "waiting thread end dragged into the tissue"
            elif self.thread_touches:
                failure = f"touched a placed thread: {self.thread_touches[-1]['geoms']}"
            elif self.prohibited:
                failure = f"robot contact with the phantom: {self.prohibited[-1]['geoms']}"
            if failure:
                record.update(outcome=failure, policy_steps=tick+1)
                break
        else:
            record.update(outcome="approach deadline (6 s) without a stroke", policy_steps=HORIZON)
        self.q_ref = q_ref
        self.finish(name, started)
        record["wall_s"] = time.perf_counter()-started
        self.approaches.append(record)
        print(f"  approach {i}: {record['outcome']} after {record['policy_steps']} policy steps", flush=True)
        self.check(f"site {i}: policy started the stroke", record["outcome"] == "triggered",
                   {k: v for k, v in record.items() if k.startswith("trigger_") or k == "outcome"})

    # ------------------------------------------------------------ cycle
    def start(self):
        q = super().start()
        lines = vessel_polylines(self.m, self.d)
        self.vessels = [(np.array(a[:2]), np.array(b[:2]), v["radius_m"]) for v in lines.values()
                        for a, b in v["segments"]]
        return q

    def steps(self):
        yardstick = {fn.__name__: fn for fn in super().steps()}
        steps = []

        def named(fn, name):
            fn.__name__ = name
            steps.append(fn)

        def rise():
            q = self.q_ref.copy()
            q[2] += LIFT
            self.move("lift at start", q, .15, .02)

        named(yardstick["settle"], "settle")
        named(rise, "lift_start")
        for n, i in enumerate(self.sites):
            if n > 0:
                named(yardstick[f"reload_{n}"], f"reload_{n}")

            def approach(i=i):
                self.approach(i)
            named(approach, f"approach_{i}")
            for name in ("pick", "insert", "release", "retract"):
                named(yardstick[f"{name}_{i}"], f"{name}_{i}")

            def lift(i=i, inner=yardstick[f"lift_{i}"]):
                inner()
                k = self.placed[-1][1]
                end = self.end_of(k)-self.tissue.offset
                self.placed_rest.append(end[:2].copy())
            named(lift, f"lift_{i}")
        return steps


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    agent = parser.add_mutually_exclusive_group(required=True)
    agent.add_argument("--weights", type=Path)
    agent.add_argument("--scripted", choices=("aim", "compensate"), help="insert_core.h yardstick instead")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sites", type=int, nargs="+", default=list(SITES))
    parser.add_argument("--level", type=float, default=0.)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    started = datetime.now(timezone.utc).isoformat()
    agent = Learned(args.weights) if args.weights else Scripted(args.scripted == "compensate")
    cycle = PolicyCycle(agent, args.sites, args.level, args.seed)
    status = "completed"
    try:
        cycle.run()
    except (RuntimeError, FloatingPointError) as error:
        status = f"stopped: {error}"
    finally:
        report = e2e.save(cycle, args.output, status, started)
        report.update(controller=agent.label, weights=getattr(agent, "weights", None),
                      weights_sha256=getattr(agent, "weights_sha256", None), level=args.level,
                      seed=args.seed, sites=args.sites, approaches=cycle.approaches, results=cycle.results,
                      thread_touches=cycle.thread_touches[:20])
        (args.output/"report.json").write_text(json.dumps(report, indent=1)+"\n")
    print(report["status"], flush=True)
    for r in cycle.results:
        print(f"  site {r['site']}: placement {r['placement_lateral_um']:.1f} µm, depth {r['end_depth_mm']:.2f} mm")


if __name__ == "__main__":
    main()
