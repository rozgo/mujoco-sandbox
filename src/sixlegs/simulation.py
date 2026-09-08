"""Shared physics stepping for the live viewer, tests, and saved recordings."""
import json
from pathlib import Path
import time

import mujoco
import numpy as np

from sixlegs.scene import load_scene
from sixlegs.task import TransferDemo


class Simulation:
    def __init__(self):
        self.model,self.data=load_scene()
        self.demo=TransferDemo(self.model,self.data)
        self.ticks=0
        self.maximum_torque_fraction=0.

    def step(self):
        if self.demo.done:return
        if self.ticks%5==0:self.demo.update(.01)
        mujoco.mj_step(self.model,self.data)
        self.demo.observe()
        caps=np.max(np.abs(self.model.actuator_forcerange),axis=1)
        fraction=float(np.max(np.abs(self.data.actuator_force)/caps))
        self.maximum_torque_fraction=max(self.maximum_torque_fraction,fraction)
        if fraction>1.00001:raise RuntimeError('Actuator exceeded its physical torque cap')
        self.ticks+=1

    def report(self):
        report=self.demo.verify()
        report['maximum_torque_fraction']=self.maximum_torque_fraction
        report['completed']=self.demo.done
        return report


def run_headless(output, record=True, max_seconds=180):
    output=Path(output)
    output.mkdir(parents=True,exist_ok=True)
    sim=Simulation()
    states,controls,times,phases=[],[],[],[]
    started=time.monotonic()
    try:
        while not sim.demo.done and sim.data.time<max_seconds:
            sim.step()
            if record and sim.ticks%25==0:
                states.append(sim.data.qpos.copy())
                controls.append(sim.data.ctrl.copy())
                times.append(sim.data.time)
                phases.append(sim.demo.index)
        if not sim.demo.done:raise RuntimeError('Demo exceeded its simulation time limit')
    finally:
        report=sim.report()
        report['wall_seconds']=time.monotonic()-started
        (output/'transfer_report.json').write_text(json.dumps(report,indent=2)+'\n')
        if record:
            np.savez_compressed(output/'transfer.npz',qpos=states,ctrl=controls,time=times,phase=phases)
    if not report['success']:raise RuntimeError(f'Transfer failed; see {output / "transfer_report.json"}')
    print(f"PASS: both objects released on destination in {sim.data.time:.2f} simulated seconds.",flush=True)
    return report
