"""Existing FlyGym walking/reflex skill, driven by bounded descending commands."""

import mujoco as mj
import numpy as np
from flygym.compose import ActuatorType
from flygym_demo.complex_terrain import (
    HybridControllerObservation,
    HybridTurningController,
    PreprogrammedSteps,
    apply_locomotion_action,
)


class Motors:
    def __init__(self, arena, seed=42, timestep=0.001):
        self.arena, self.timestep = arena, timestep
        self.controllers = []
        self.foot_ids = []
        self.stumble_ids = []
        m = arena.sim.mj_model
        steps = PreprogrammedSteps()
        for i, fly in enumerate(arena.flies):
            ctrl = HybridTurningController(
                timestep=timestep,
                preprogrammed_steps=steps,
                output_dof_order=fly.get_actuated_jointdofs_order("position"),
            )
            ctrl.reset(seed=seed + i)
            self.controllers.append(ctrl)
            self.foot_ids.append(
                [m.body(f"{fly.name}/{leg}_tarsus5").id for leg in ctrl.legs]
            )
            self.stumble_ids.append(
                [
                    [
                        m.body(f"{fly.name}/{leg}_{link}").id
                        for link in ("tibia", "tarsus1", "tarsus2")
                    ]
                    for leg in ctrl.legs
                ]
            )
        self.foot_ids = np.array(self.foot_ids)
        self.stumble_ids = np.array(self.stumble_ids)
        self.body_force = np.zeros((m.nbody, 3))
        self.contact_force = np.zeros(6)
        self.dead_actuators = [
            np.array(
                [j for j in range(m.nu) if m.actuator(j).name.startswith(f.name + "/")]
            )
            for f in arena.flies
        ]
        self.holding = np.zeros(len(arena.flies), dtype=bool)
        self.hold_targets = [None] * len(arena.flies)
        self.position_ids = arena.sim._intern_actuatorids_by_type_by_fly[
            ActuatorType.POSITION
        ]

    def contact_forces(self):
        m, d = self.arena.sim.mj_model, self.arena.sim.mj_data
        self.body_force.fill(0)
        for k in range(d.ncon):
            c = d.contact[k]
            if c.efc_address < 0:
                continue
            mj.mj_contactForce(m, d, k, self.contact_force)
            world_force = c.frame.reshape(3, 3).T @ self.contact_force[:3]
            self.body_force[m.geom_bodyid[c.geom1]] -= world_force
            self.body_force[m.geom_bodyid[c.geom2]] += world_force
        return self.body_force

    def apply(self, commands, alive=None):
        sim = self.arena.sim
        d = sim.mj_data
        forces = self.contact_forces()
        for i, (fly, controller, bid) in enumerate(
            zip(self.arena.flies, self.controllers, self.arena.fly_body_ids)
        ):
            if alive is not None and not alive[i]:
                ids = self.dead_actuators[i]
                # Death is a permanent disabling of active torques, not body removal.
                sim.mj_model.actuator_gainprm[ids, :] = 0
                sim.mj_model.actuator_biasprm[ids, :] = 0
                d.ctrl[ids] = 0
                continue
            if np.max(np.abs(commands[i])) < 1e-5:
                ids = self.position_ids[fly.name]
                if not self.holding[i]:
                    joints = sim.mj_model.actuator_trnid[ids, 0]
                    self.hold_targets[i] = d.qpos[
                        sim.mj_model.jnt_qposadr[joints]
                    ].copy()
                self.holding[i] = True
                d.ctrl[ids] = self.hold_targets[i]
                sim.set_leg_adhesion_states(fly.name, np.ones(6))
                continue
            self.holding[i] = False
            obs = HybridControllerObservation(
                float(d.xpos[bid, 2]),
                d.xpos[self.foot_ids[i], 2],
                forces[self.stumble_ids[i]],
                d.xmat[bid].reshape(3, 3)[:, 0],
            )
            action = controller.step(np.clip(commands[i], -1.2, 1.2), obs)
            apply_locomotion_action(sim, fly.name, action)

    def step(self, commands, alive=None):
        self.apply(commands, alive)
        m, d = self.arena.sim.mj_model, self.arena.sim.mj_data
        n = round(self.timestep / m.opt.timestep)
        if not np.isclose(n * m.opt.timestep, self.timestep):
            raise ValueError(
                "Motor period must contain an integer number of physics steps"
            )
        mj.mj_step(m, d, nstep=n)


def probe(n_flies=1, seconds=1.0, command=(1.0, 1.0), seed=42):
    import json
    import time

    from PIL import Image

    from .paths import OUTPUTS, PREVIEWS
    from .scene import build, render

    start = time.perf_counter()
    arena = build(n_flies)
    motor = Motors(arena, seed=seed)
    d = arena.sim.mj_data
    qpos, xyz, commands = [], [], []
    max_force = 0
    for step in range(round(seconds / motor.timestep)):
        c = np.tile(command, (n_flies, 1))
        motor.step(c)
        if step % 20 == 0:
            if not np.isfinite(d.qpos).all() or d.warning.number.sum():
                raise RuntimeError("Numerical failure in motor probe")
            qpos.append(d.qpos.copy())
            xyz.append(d.xpos[arena.fly_body_ids].copy())
            commands.append(c)
        max_force = max(max_force, float(np.abs(d.actuator_force).max()))
    duration = time.perf_counter() - start
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    name = f"motor_{n_flies}_{command[0]}_{command[1]}"
    np.savez_compressed(OUTPUTS / f"{name}.npz", qpos=qpos, xyz=xyz, commands=commands)
    Image.fromarray(render(arena)).save(PREVIEWS / f"{name}.png")
    result = {
        "wall_seconds": duration,
        "simulation_seconds": float(d.time),
        "n_flies": n_flies,
        "final_xyz": d.xpos[arena.fly_body_ids].tolist(),
        "max_actuator_force_model_units": max_force,
        "command": command,
        "warnings": d.warning.number.tolist(),
    }
    (OUTPUTS / f"{name}.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--flies", type=int, default=1)
    p.add_argument("--seconds", type=float, default=1.0)
    p.add_argument("--left", type=float, default=1.0)
    p.add_argument("--right", type=float, default=1.0)
    a = p.parse_args()
    probe(a.flies, a.seconds, (a.left, a.right))
