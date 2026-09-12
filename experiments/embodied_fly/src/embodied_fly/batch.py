"""Native CPU worlds with exactly the single-fly actuator/observation convention.

Added sensors only expose existing physics. No change to masses, contacts or gains.
The authoritative acceptance evaluator still enumerates every physical contact.
"""

import argparse
import json
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
from mjbatch import Batch

from embodied_fly.body import CONTROL_DT, SUBSTEPS, FlyEnvironment
from embodied_fly.provenance import evidence, utc_now


class FlyBatch:
    def __init__(self, worlds, threads=0):
        self.template = FlyEnvironment()
        single = self.template
        root = single.fly.mjcf_model.root_model
        xml = ET.fromstring(root.to_xml_string())
        sensors = xml.find("sensor")
        # Both origin choices are intentional: retain inertia-frame observations,
        # but measure forward motion in anatomical thorax axes for rewards.
        for frame in ("body", "xbody"):
            for kind in ("angvel", "linvel"):
                ET.SubElement(
                    sensors,
                    "frame" + kind,
                    name=f"batch_{frame}_{kind}",
                    objtype=frame,
                    objname="walker/thorax",
                )
        for i, geom in enumerate(single.foot_geoms):
            ET.SubElement(
                sensors,
                "contact",
                name=f"batch_foot_{i}",
                geom1=single.model.geom(geom).name,
                data="found",
                num="1",
            )
        self.forbidden = [
            i
            for i in range(single.model.ngeom)
            if single.model.geom_bodyid[i] != 0
            and (single.model.geom_contype[i] or single.model.geom_conaffinity[i])
            and i not in single.support_geoms
        ]
        for i in self.forbidden:
            ET.SubElement(
                sensors,
                "contact",
                name=f"batch_forbidden_{i}",
                geom1=single.model.geom(i).name,
                body2="world",
                data="force",
                reduce="maxforce",
                num="1",
            )
        self.model = mujoco.MjSpec.from_string(
            ET.tostring(xml, encoding="unicode"), assets=root.get_assets()
        ).compile()
        self.model.actuator_forcelimited[:] = single.model.actuator_forcelimited
        self.model.actuator_forcerange[:] = single.model.actuator_forcerange
        self.n = worlds
        self.batch = Batch(self.model, worlds, num_threads=threads)
        self.fields = {
            name: self.batch.bind(name)
            for name in (
                "qpos",
                "qvel",
                "act",
                "ctrl",
                "xmat",
                "ximat",
                "sensordata",
                "warning",
            )
        }
        self.mean_sensors = self.fields["sensordata"].copy()
        self.previous_action = np.zeros((worlds, self.model.nu), np.float32)
        self.command = np.zeros((worlds, 3), np.float32)
        self.needs = np.zeros((worlds, 5), np.float32)
        self.ages = np.zeros(worlds, np.int64)
        self.forbidden_peak = np.zeros(worlds)
        self.body_weight = self.model.body_mass.sum() * 981
        self.sensor_addresses = {
            self.model.sensor(i).name: self.model.sensor_adr[i]
            for i in range(self.model.nsensor)
        }
        self.reset(np.arange(worlds))

    def reset(self, ids, yaw=None):
        ids = np.asarray(ids, dtype=np.int64)
        if not len(ids):
            return
        # Reset copies the tested initial full anatomy, including passive wings.
        self.template.reset()
        # mjbatch copies only changed field elements. Reset first so an unchanged
        # requested wing pose is not lost when mj_resetData restores qpos0.
        self.batch.reset(ids)
        self.fields["qpos"][ids] = self.template.data.qpos
        self.fields["qvel"][ids] = 0
        self.fields["act"][ids] = 0
        self.fields["ctrl"][ids] = 0
        if yaw is not None:
            self.fields["qpos"][ids, 3] = np.cos(np.asarray(yaw) / 2)
            self.fields["qpos"][ids, 6] = np.sin(np.asarray(yaw) / 2)
        self.batch.forward(ids)
        self.mean_sensors[ids] = self.fields["sensordata"][ids]
        self.previous_action[ids] = 0
        self.ages[ids] = 0
        self.forbidden_peak[ids] = 0

    def velocity(self, anatomical=True):
        frame = "xbody" if anatomical else "body"
        rotation_field = "xmat" if anatomical else "ximat"
        rotation = self.fields[rotation_field][:, self.template.thorax_id].reshape(-1, 3, 3)
        values = []
        for kind in ("angvel", "linvel"):
            adr = self.sensor_addresses[f"batch_{frame}_{kind}"]
            global_velocity = self.fields["sensordata"][:, adr : adr + 3]
            values.append(np.einsum("nji,nj->ni", rotation, global_velocity))
        return np.concatenate(values, axis=1)

    def observation(self):
        single = self.template
        ranges = self.model.jnt_range[single.joint_ids]
        q = self.fields["qpos"][:, single.qpos_indices]
        normalized_q = (
            2 * (q - ranges[:, 0]) / np.maximum(ranges[:, 1] - ranges[:, 0], 1e-4) - 1
        )
        foot_adr = [self.sensor_addresses[f"batch_foot_{i}"] for i in range(6)]
        foot_touch = (self.fields["sensordata"][:, foot_adr] > 0).astype(np.float32)
        return np.concatenate(
            (
                np.clip(normalized_q, -5, 5),
                np.clip(self.fields["qvel"][:, single.qvel_indices] / 100, -10, 10),
                self.fields["act"],
                self.velocity(False) / (20, 20, 20, 10, 10, 10),
                self.fields["xmat"][:, single.thorax_id, 6:9],
                foot_touch,
                self.previous_action,
                self.command / 10,
                self.needs,
            ),
            axis=1,
        ).astype(np.float32)

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        if action.shape != self.previous_action.shape or not np.isfinite(action).all():
            raise ValueError("Invalid batched actor output")
        self.previous_action[:] = np.clip(action, -1, 1)
        inactive = self.template.walking_inactive
        self.previous_action[:, inactive] = self.template.passive_action[inactive]
        self.fields["ctrl"][:] = self.template.low + (self.previous_action + 1) * 0.5 * (
            self.template.high - self.template.low
        )
        self.forbidden_peak[:] = 0
        addresses = [self.sensor_addresses[f"batch_forbidden_{i}"] for i in self.forbidden]
        self.mean_sensors[:] = 0
        for _ in range(SUBSTEPS):
            self.batch.step()
            self.mean_sensors += self.fields["sensordata"]
            # Sensor's force x is normal force of its maximum-norm contact.
            # This is a training proxy; native acceptance checks all contacts.
            force = np.abs(self.fields["sensordata"][:, addresses]).max(axis=1)
            np.maximum(self.forbidden_peak, force, out=self.forbidden_peak)
        self.mean_sensors /= SUBSTEPS
        if np.any(self.fields["warning"]) or not np.isfinite(self.fields["qpos"]).all():
            raise RuntimeError("MuJoCo numerical failure in batch")
        self.ages += 1
        return self.observation()


def benchmark(args):
    args.output.mkdir(parents=True, exist_ok=False)
    records = []
    for worlds in args.worlds:
        setup = time.perf_counter()
        env = FlyBatch(worlds, args.threads)
        setup = time.perf_counter() - setup
        action = np.broadcast_to(env.template.passive_action, (worlds, env.model.nu))
        # Performance probe only: passive input is not a trained skill.
        started = time.perf_counter()
        for _ in range(args.steps):
            env.step(action)
        elapsed = time.perf_counter() - started
        records.append(
            {
                "worlds": worlds,
                "threads": env.batch.num_threads,
                "setup_seconds": setup,
                "step_seconds": elapsed,
                "transitions": worlds * args.steps,
                "transitions_per_second": worlds * args.steps / elapsed,
                "aggregate_simulated_seconds": worlds * args.steps * CONTROL_DT,
                "warning_count": int(env.fields["warning"].sum()),
            }
        )
        print(json.dumps(records[-1]), flush=True)
        del env
    report = {
        "provenance": evidence(),
        "completed_utc": utc_now(),
        "measurements": records,
        "scope": "CPU physics plus observation/contact extraction; no neural inference or learning",
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worlds", nargs="+", type=int, default=[1, 8, 16, 32])
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--steps", type=int, default=100)
    benchmark(parser.parse_args())
