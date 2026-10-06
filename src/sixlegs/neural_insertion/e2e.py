"""First end-to-end insertion cycle: pick, carry, align, insert, release, withdraw.

Scripted yardstick control (not learned) on the end-to-end variant
(e2e_scene.py) with the full DER thread at the task-regime clock and the
provisional tissue model (tissue.py). No disturbances. Motion comes only from
mj_step under the force-limited servo; tissue forces are documented applied
forces. Each phase ends with a check on measured state; a failed check stops the
run and keeps the partial record. Engine units: mm, g, s.

Usage: uv run --locked python -m sixlegs.neural_insertion.e2e --output DIR [--target 0]
"""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import pickle
import shutil
import time

import mujoco
import numpy as np

from .e2e_scene import BED_TOP, EYELET_CENTER, KEEPER_CLOSED, KEEPER_DESIGN, TUBE, UNITS, load_e2e
from .motion import JOINT_NAMES, Servo, minimum_jerk
from .tissue import Tissue, TissueParams

L = UNITS.length
SERVO_EVERY = 4            # servo update every 20 us
RECORD_EVERY = 200         # replay state every 1 ms
DESCEND_OFFSET = .0725     # mm: needle axis +Y of the eyelet centre while passing through it
CAPTURE_OFFSET = -.040     # mm: needle axis relative to the eyelet centre with the rim in the slot
SLOT_DEPTH = .23           # mm: tip below the bed so the slot straddles the rim
INSERT_DEPTH = 2.0         # mm: tip below the tissue surface
WORK_Z = -6.
CARRY_Z = 49.5             # mm: Z stage during transport (travel limit 50); the 44 mm thread hangs clear
LAY_OFFSET = 40.           # mm: descent starts this far -X of the target and lays the thread out behind
LOST_MM = 1.0              # eyelet farther than this from the tip while carried: thread lost
LOWER_S = 1.5              # duration of the laying descent
KEEPER_OPEN = .5
CLAMP_FORCE = 100.         # uN net keeper force on the rim: the validated press-fixture load (100 uN);
                           # 5 mN made the clamped few-microgram ring numerically stiff and it blew up (run_v4)
CLAMP_SPEED = .5           # mm/s: damping caps the keeper's closing speed in force mode


class Cycle:
    def __init__(self, target=0, tubes=False):
        self.m, self.d, meta = load_e2e(tubes=tubes)
        params = TissueParams()
        if tubes:
            c = meta["tube"]["center_m"]
            params = TissueParams(tubes=True, channel=(c[0]*L, c[1]*L, TUBE["radius"]*L))
        self.meta = meta
        self.target = target
        self.qadr = np.array([self.m.jnt_qposadr[self.m.joint(n).id] for n in JOINT_NAMES])
        scratch = mujoco.MjData(self.m)
        mujoco.mj_kinematics(self.m, scratch)
        self.tip0 = scratch.site("needle_tip").xpos.copy()
        self.servo = Servo(self.m)
        self.tissue = Tissue(self.m, meta["thread_bodies"], params)
        self.trace = {k: [] for k in ("time", "qpos", "tip", "eyelet", "depth", "needle_axial", "act_force",
                                       "ncon", "phase")}
        self.events = []
        self.q_ref = None
        self.robot = {self.m.body(n).id for n in ("stage_x", "stage_y", "stage_z", "insertion", "retainer")}
        self.phantom = self.m.geom("tissue_phantom").id
        self.obstacles = {self.phantom}
        self.prohibited = []
        self.done = []
        self.clamp = None          # net clamp force (uN) while the keeper is in force mode
        self.on_step = None        # optional observer (live viewer); must not change state
        self.aux = None            # optional extra command at each servo update, aux(t in move)
        self.ret = JOINT_NAMES.index("retainer")
        self.ret_dof = self.m.jnt_dofadr[self.m.joint("retainer").id]
        self.ret_qadr = self.m.jnt_qposadr[self.m.joint("retainer").id]
        self.ret_act = self.m.actuator("retainer").id

    # ------------------------------------------------------------ geometry
    def joints(self, tip, z_stage=WORK_Z, keeper=KEEPER_OPEN):
        q = np.zeros(5)
        q[0], q[1] = tip[0]-self.tip0[0], tip[1]-self.tip0[1]
        q[2] = z_stage
        q[3] = self.tip0[2]+z_stage-tip[2]
        q[4] = keeper
        return q

    def tip_of(self, q):
        return np.array((self.tip0[0]+q[0], self.tip0[1]+q[1], self.tip0[2]+q[2]-q[3]))

    @property
    def eyelet(self):
        return self.d.site("eyelet_center").xpos.copy()

    @property
    def tip(self):
        return self.d.site("needle_tip").xpos.copy()

    def target_site(self):
        return self.d.site(f"target_{self.target}").xpos.copy()

    # ------------------------------------------------------------ stepping
    def start(self):
        C = EYELET_CENTER*L
        q = self.joints(np.array((C[0], C[1]+DESCEND_OFFSET, BED_TOP*L+1.0)))
        self.d.qpos[self.qadr] = q  # initialization, before any stepping
        self.d.ctrl[:] = q
        mujoco.mj_forward(self.m, self.d)
        self.q_ref = q
        return q

    def move(self, name, q_goal, duration, hold=0., carried=False):
        """Minimum-jerk joint move from the current reference, under the servo and tissue model."""
        q0, q1 = self.q_ref.copy(), np.asarray(q_goal, float)
        steps = round((duration+hold)/self.m.opt.timestep)
        t0 = self.d.time
        started = time.perf_counter()
        for k in range(steps):
            t = self.d.time-t0
            if k % getattr(self, "servo_every", SERVO_EVERY) == 0:
                q, v, a = minimum_jerk(q0, q1, duration, t) if t < duration else (q1, 0*q1, 0*q1)
                self.servo.command(self.d, q, v, a)
                if self.clamp is not None:
                    self.clamp_command()
                if self.aux is not None:
                    self.aux(t)
            self.tissue.step(self.d)
            before = self.d.time
            mujoco.mj_step(self.m, self.d)
            if self.d.warning[mujoco.mjtWarning.mjWARN_BADQACC].number:
                # MuJoCo resets the state after this warning; stop rather than continue from the reset.
                raise FloatingPointError(f"MuJoCo instability (bad qacc) at t={before:.5f} s in {name}")
            if self.on_step is not None and k % 50 == 0:
                self.on_step(name)
            if k % getattr(self, "record_every", RECORD_EVERY) == 0:
                self.record(name)
                if not np.isfinite(self.d.qpos).all():
                    raise FloatingPointError("nonfinite state")
                if carried and np.linalg.norm(self.eyelet-self.tip) > LOST_MM:
                    raise RuntimeError(f"thread lost from the needle at t={self.d.time:.4f} s in {name}")
        self.q_ref = q1
        self.record(name)
        warnings = [int(w.number) for w in self.d.warning]
        self.events.append({"phase": name, "end_s": float(self.d.time), "wall_s": time.perf_counter()-started,
                            "tip": self.tip.tolist(), "eyelet": self.eyelet.tolist(),
                            "warnings": warnings})
        print(f"{name:10s} t={self.d.time:.3f}s wall={time.perf_counter()-started:6.1f}s "
              f"tip={np.round(self.tip, 3)} eyelet={np.round(self.eyelet, 3)} ncon={self.d.ncon}", flush=True)

    def record(self, name):
        tr, d = self.trace, self.d
        tr["time"].append(d.time)
        tr["qpos"].append(d.qpos.copy())
        tr["tip"].append(self.tip)
        tr["eyelet"].append(self.eyelet)
        tr["depth"].append(self.tissue.surface(self.eyelet)-self.eyelet[2])
        tr["needle_axial"].append(float(d.xfrc_applied[self.tissue.needle, 2]))
        tr["act_force"].append(d.actuator_force.copy())
        tr["ncon"].append(d.ncon)
        tr["phase"].append(name)
        for c in d.contact[:d.ncon]:
            b1, b2 = self.m.geom_bodyid[c.geom1], self.m.geom_bodyid[c.geom2]
            if (c.geom1 in self.obstacles or c.geom2 in self.obstacles) and (b1 in self.robot or b2 in self.robot):
                self.prohibited.append({"time_s": float(d.time), "geoms": [self.m.geom(c.geom1).name,
                                                                         self.m.geom(c.geom2).name]})

    def clamp_command(self):
        """Force-limited keeper: cancel its own weight, press the rim with the clamp force, and damp the
        closing speed. The drive is a position actuator, F = kp*(ctrl - q) - kv*qdot, so the command
        that yields a desired force F is ctrl = q + (F + kv*qdot)/kp."""
        d, j = self.d, self.ret
        qdot = d.qvel[self.ret_dof]
        damping = CLAMP_FORCE/CLAMP_SPEED
        force = d.qfrc_bias[self.ret_dof]+self.clamp-damping*qdot
        kp, kv = self.servo.kp[j], self.servo.kv[j]
        lo, hi = self.m.actuator_ctrlrange[self.ret_act]
        d.ctrl[self.ret_act] = np.clip(d.qpos[self.ret_qadr]+(force+kv*qdot)/kp, lo, hi)

    def checkpoint(self, path):
        state = np.zeros(mujoco.mj_stateSize(self.m, mujoco.mjtState.mjSTATE_INTEGRATION))
        mujoco.mj_getState(self.m, self.d, state, mujoco.mjtState.mjSTATE_INTEGRATION)
        blob = {"state": state, "q_ref": self.q_ref, "integral": self.servo.integral, "tissue": self.tissue.state,
                "trace": self.trace, "events": self.events, "prohibited": self.prohibited, "done": self.done,
                "clamp": self.clamp}
        Path(path).write_bytes(pickle.dumps(blob))

    def restore(self, path):
        blob = pickle.loads(Path(path).read_bytes())
        mujoco.mj_setState(self.m, self.d, blob["state"], mujoco.mjtState.mjSTATE_INTEGRATION)
        mujoco.mj_forward(self.m, self.d)
        self.q_ref, self.servo.integral, self.tissue.state = blob["q_ref"], blob["integral"], blob["tissue"]
        self.trace, self.events, self.prohibited, self.done = blob["trace"], blob["events"], blob["prohibited"], blob["done"]
        self.clamp = blob.get("clamp")

    def check(self, name, ok, detail):
        self.events.append({"check": name, "passed": bool(ok), **detail})
        print(f"  check {name}: {'PASS' if ok else 'FAIL'} {detail}", flush=True)
        if not ok:
            raise RuntimeError(f"check failed: {name}")

    # ------------------------------------------------------------ the cycle
    def steps(self):
        """The cycle as named steps; each ends with its own measured check where one applies."""
        C, bed = EYELET_CENTER*L, BED_TOP*L
        site = self.target_site()
        surface = self.tissue.surface(site)
        tip_xy = site[:2]+np.array((0, CAPTURE_OFFSET))  # needle axis such that the eyelet centre is over the target
        released_xy = tip_xy+np.array((0, DESCEND_OFFSET-CAPTURE_OFFSET))

        def at(xyz, keeper=KEEPER_OPEN, z_stage=WORK_Z):
            return self.joints(np.asarray(xyz, float), z_stage=z_stage, keeper=keeper)

        def settle():
            self.move("settle", self.start(), .005)

        def descend():
            self.move("descend", at((C[0], C[1]+DESCEND_OFFSET, bed-SLOT_DEPTH)), .15, .01)

        def capture():
            self.move("capture", at((C[0], C[1]+CAPTURE_OFFSET, bed-SLOT_DEPTH)), .05, .01)

        def seat():
            self.move("seat", at((C[0], C[1]+CAPTURE_OFFSET, bed-SLOT_DEPTH+.40)), .05, .01)
            lifted = self.eyelet[2]-(bed+.020)
            self.check("eyelet lifted by the slot", lifted > .20, {"eyelet_lift_mm": float(lifted)})

        def keeper():
            q = self.q_ref.copy()
            if KEEPER_DESIGN == "latch":
                # Latch: fast to 50 µm above its closed position, then close slowly; no contact intended.
                q[4] = KEEPER_CLOSED*L-.05
                self.move("keeper", q, .10, .005, carried=True)
                q = q.copy()
                q[4] = KEEPER_CLOSED*L
                self.move("keeper", q, .10, .02, carried=True)
                travel = self.d.qpos[self.ret_qadr]
                self.check("latch closes the slot", abs(travel-q[4]) < .01,
                           {"latch_travel_error_um": float((travel-q[4])*1e3)})
                return
            # Clamp: fast to 50 µm above the rim in position mode, then force mode: the keeper closes at no
            # more than 0.5 mm/s and presses the rim onto the slot's lower lip with a limited force.
            q[4] = KEEPER_CLOSED*L-.05
            self.move("keeper", q, .10, .005, carried=True)
            self.clamp = CLAMP_FORCE
            self.move("keeper", q, 0., .20, carried=True)
            gap = min(mujoco.mj_geomDistance(self.m, self.d, self.m.geom(f"keeper_pad_{s}").id,
                                             self.m.geom(f"eyelet_{k}").id, 1., None)
                      for s in (-1, 1) for k in range(12))
            self.check("keeper clamps the rim", gap < .002, {"pad_to_rim_gap_um": float(gap*1e3)})

        def lift():
            q = self.q_ref.copy()
            q[2] = CARRY_Z
            self.move("lift", q, .60, .05, carried=True)
            gap = np.linalg.norm(self.eyelet-self.tip)
            self.check("eyelet carried", gap < .6, {"eyelet_to_tip_mm": float(gap)})

        def transport():
            # To the start of the laying descent, LAY_OFFSET -X of the target, at full carry height.
            q = self.q_ref.copy()
            q[0], q[1] = tip_xy[0]-LAY_OFFSET-self.tip0[0], tip_xy[1]-self.tip0[1]
            q[2] = CARRY_Z
            self.move("transport", q, 1.2, .10, carried=True)

        def lower():
            # Diagonal descent: the thread's free end lands first and the thread lies out behind
            # the needle instead of piling up under it (a straight plunge buckled it; run_v0).
            self.move("lower", at((*tip_xy, surface+.5), keeper=KEEPER_CLOSED*L), LOWER_S, .10, carried=True)
            gap = np.linalg.norm(self.eyelet-self.tip)
            lateral = np.linalg.norm(self.eyelet[:2]-site[:2])
            self.check("aligned over target with eyelet", gap < .6 and lateral < .1,
                       {"eyelet_to_tip_mm": float(gap), "eyelet_to_target_lateral_mm": float(lateral)})

        def unkeep():
            self.clamp = None
            self.servo.integral[self.ret] = 0.
            q = self.q_ref.copy()
            q[4] = KEEPER_OPEN
            self.move("unkeep", q, .10, .01, carried=True)

        def insert():
            self.move("insert", at((*tip_xy, surface-INSERT_DEPTH)), .60, .05, carried=True)
            depth = self.tissue.surface(self.eyelet)-self.eyelet[2]
            self.check("eyelet inserted", depth > 1.0, {"eyelet_depth_mm": float(depth),
                                                        "punctured": self.tissue.state.punctured})

        def release():
            self.move("release", at((*released_xy, surface-INSERT_DEPTH)), .20, .02)

        def withdraw():
            self.move("withdraw", at((*released_xy, surface+2.0)), .50, .10)
            depth = self.tissue.surface(self.eyelet)-self.eyelet[2]
            gap = np.linalg.norm(self.eyelet-self.tip)
            lateral = np.linalg.norm(self.eyelet[:2]-site[:2])
            self.check("thread retained after withdrawal", depth > 1.0 and gap > 2.0,
                       {"eyelet_depth_mm": float(depth), "eyelet_to_tip_mm": float(gap),
                        "placement_lateral_mm": float(lateral)})

        return [settle, descend, capture, seat, keeper, lift, transport, lower, unkeep, insert, release, withdraw]

    def run(self, checkpoints=None, stop_after=None):
        for step in self.steps():
            name = step.__name__
            if name in self.done:
                continue
            step()
            self.done.append(name)
            if checkpoints is not None:
                Path(checkpoints).mkdir(parents=True, exist_ok=True)
                self.checkpoint(Path(checkpoints)/f"{len(self.done):02d}_{name}.pkl")
            if name == stop_after:
                break


def save(cycle, output, status, started):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    trace = {k: np.array(v) for k, v in cycle.trace.items()}
    np.savez_compressed(output/"trace.npz", **trace)
    state = np.zeros(mujoco.mj_stateSize(cycle.m, mujoco.mjtState.mjSTATE_INTEGRATION))
    mujoco.mj_getState(cycle.m, cycle.d, state, mujoco.mjtState.mjSTATE_INTEGRATION)
    np.save(output/"final_state.npy", state)
    if cycle.meta.get("xml_path"):
        shutil.copy(cycle.meta["xml_path"], output/"scene.xml")  # exact replay after the builder changes
    report = {"status": status, "variant": cycle.meta["variant"], "tube": cycle.meta["tube"], "started_utc": started, "finished_utc": datetime.now(timezone.utc).isoformat(),
              "simulated_s": float(cycle.d.time), "target": cycle.target, "units": UNITS.name,
              "timestep_s": cycle.m.opt.timestep, "scene_sha256": cycle.meta["xml_sha256"],
              "missed_channel": cycle.tissue.state.missed,
              "tissue_params": asdict(cycle.tissue.p), "punctured": cycle.tissue.state.punctured,
              "peak_needle_axial_uN": cycle.tissue.state.peak_axial,
              "peak_needle_lateral_uN": cycle.tissue.state.peak_lateral,
              "prohibited_contacts": cycle.prohibited[:50], "prohibited_count": len(cycle.prohibited),
              "events": cycle.events, "controller": "scripted yardstick (minimum-jerk joint references, programmed servo)",
              "disturbances": "none"}
    (output/"report.json").write_text(json.dumps(report, indent=1)+"\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target", type=int, default=0)
    parser.add_argument("--resume", type=Path, help="checkpoint to continue from")
    parser.add_argument("--stop-after", help="step name to stop after")
    parser.add_argument("--tubes", action="store_true", help="pre-formed tissue channel variant")
    parser.add_argument("--lower-seconds", type=float, default=LOWER_S, help="duration of the laying descent")
    args = parser.parse_args()
    globals()["LOWER_S"] = args.lower_seconds
    started = datetime.now(timezone.utc).isoformat()
    cycle = Cycle(args.target, tubes=args.tubes)
    if args.resume:
        cycle.restore(args.resume)
    status = "completed"
    try:
        cycle.run(checkpoints=args.output/"checkpoints", stop_after=args.stop_after)
        if args.stop_after:
            status = f"stopped after {args.stop_after} (requested)"
    except (RuntimeError, FloatingPointError) as error:
        status = f"stopped: {error}"
    finally:
        report = save(cycle, args.output, status, started)
    print(json.dumps({k: report[k] for k in ("status", "simulated_s", "punctured", "peak_needle_axial_uN",
                                             "prohibited_count")}, indent=1))


if __name__ == "__main__":
    main()
