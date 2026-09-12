"""Local chemical/light/range sensing and an explicit camera-feature encoder."""

from dataclasses import dataclass

import mujoco as mj
import numpy as np

from .scene import LAMP_POS, RESOURCE_POS


@dataclass
class Percept:
    food: float
    water: float
    food_gradient: np.ndarray
    water_gradient: np.ndarray
    heat: float
    heat_gradient: np.ndarray
    shade: float
    shade_gradient: np.ndarray
    threat: float
    crowding: float
    avoidance: np.ndarray
    vision: np.ndarray


class Sensors:
    def __init__(self, arena, vision=True):
        self.arena = arena
        self.use_vision = vision
        self.renderer = None
        n = len(arena.flies)
        self.last_overhead = np.full(n, 30.0)
        self.last_gray = np.zeros((n, 2, 56, 64), np.float32)
        self.last_features = np.zeros((n, 2, 3), np.float32)
        self.eye_frames = [None] * n
        m = arena.sim.mj_model
        self.own_geoms = [
            np.array(
                [g for g in range(m.ngeom) if m.geom(g).name.startswith(f.name + "/")]
            )
            for f in arena.flies
        ]
        self.geom_id = np.zeros(1, dtype=np.int32)
        self.roof_ids = [m.geom(f"shelter_roof_{i}").id for i in range(2)]
        self.swat_ids = {m.geom("swatter_pad").id, m.geom("swatter_arm").id}

    def ray(self, origin, direction, exclude=-1):
        m, d = self.arena.sim.mj_model, self.arena.sim.mj_data
        dist = mj.mj_ray(
            m,
            d,
            np.asarray(origin, dtype=float),
            np.asarray(direction, dtype=float),
            None,
            1,
            exclude,
            self.geom_id,
        )
        return float(dist), int(self.geom_id[0])

    def illumination(self, xy, z=1.2):
        origin = np.array([xy[0], xy[1], z])
        lamp = np.array([*LAMP_POS, 14.2])
        delta = lamp - origin
        distance = np.linalg.norm(delta)
        hit, _gid = self.ray(origin, delta / distance)
        visible = hit < 0 or hit >= distance - 0.2
        return float(np.exp(-np.sum((xy - LAMP_POS) ** 2) / (2 * 4.2**2)) * visible)

    def odor(self, points, remaining):
        delta = np.asarray(points)[..., None, :] - RESOURCE_POS
        dist = np.linalg.norm(delta, axis=-1)
        # Volatile concentration saturates while a patch is present; it vanishes on depletion.
        return np.exp(-dist / 8.0) * (remaining > 1e-6)

    def eyes(self, i):
        if not self.use_vision:
            return self.last_features[i]
        m, d = self.arena.sim.mj_model, self.arena.sim.mj_data
        if self.renderer is None:
            self.renderer = mj.Renderer(m, height=112, width=128)
        ids = self.own_geoms[i]
        alpha = m.geom_rgba[ids, 3].copy()
        m.geom_rgba[ids, 3] = 0
        images = []
        try:
            cameras = self.arena.sim._intern_eye_camera_ids_by_fly[
                self.arena.flies[i].name
            ]
            for cid in cameras:
                self.renderer.update_scene(d, camera=cid)
                self.renderer.scene.flags[mj.mjtRndFlag.mjRND_SHADOW] = False
                images.append(self.renderer.render().copy())
        finally:
            m.geom_rgba[ids, 3] = alpha
        image = np.array(images, dtype=np.float32) / 255
        gray = image[:, ::2, ::2, :].mean(axis=-1)
        growth = np.clip(self.last_gray[i] - gray, 0, 1).mean(axis=(1, 2))
        self.last_gray[i] = gray
        green = (
            (image[..., 1] > image[..., 0] * 1.15)
            & (image[..., 1] > image[..., 2] * 1.15)
        ).mean(axis=(1, 2))
        cyan = (
            (image[..., 1] > image[..., 0] * 1.3)
            & (image[..., 2] > image[..., 0] * 1.3)
        ).mean(axis=(1, 2))
        self.last_features[i] = np.stack(
            (
                np.clip(green * 10, 0, 1),
                np.clip(cyan * 10, 0, 1),
                np.clip(growth * 8, 0, 1),
            ),
            axis=1,
        )
        self.eye_frames[i] = np.concatenate(images, axis=1)
        return self.last_features[i]

    def sample(self, i, remaining, dt=0.02, refresh_eyes=False):
        m, d = self.arena.sim.mj_model, self.arena.sim.mj_data
        bid = self.arena.fly_body_ids[i]
        pos = d.xpos[bid].copy()
        heading = d.xmat[bid].reshape(3, 3)[:2, 0]
        heading /= max(np.linalg.norm(heading), 1e-8)
        side = np.array([-heading[1], heading[0]])
        probes = pos[:2] + np.array([heading, -heading, side, -side]) * 0.45
        odor = self.odor(probes, remaining)
        food = odor[:, :2].sum(axis=1)
        water = odor[:, 2]

        def gradient(samples):
            return (samples[0] - samples[1]) * heading + (
                samples[2] - samples[3]
            ) * side

        heat_probes = np.array(
            [self.illumination(p, max(pos[2] + 0.5, 0.4)) for p in probes]
        )
        shade = []
        for p in probes:
            distance, gid = self.ray([*p, max(pos[2], 0.4)], [0, 0, 1], bid)
            shade.append(float(gid in self.roof_ids and 0 <= distance < 4))
        # A local upward range sensor gives current clearance and observed approach.
        above, gid = self.ray(pos + [0, 0, 0.4], [0, 0, 1], bid)
        distance = above if above >= 0 else 30.0
        approach = max(0, (self.last_overhead[i] - distance) / dt)
        self.last_overhead[i] = distance
        threat = (
            float(gid in self.swat_ids)
            * float(np.clip((12 - distance) / 12, 0, 1))
            * float(np.clip(approach / 12 + 0.3, 0, 1))
        )
        avoid = np.zeros(2)
        near = 0.0
        for angle in (-0.7, 0, 0.7):
            direction = np.cos(angle) * heading + np.sin(angle) * side
            origin = np.array([*(pos[:2] + heading * 1.2), max(pos[2] - 0.2, 0.4)])
            hit, hit_id = self.ray(origin, [*direction, 0], bid)
            if 0 <= hit < 2.0:
                factor = (2 - hit) / 2
                avoid -= direction * factor
                body = m.geom_bodyid[hit_id]
                near = max(near, float(m.body(body).name.startswith("fly_")) * factor)
        feature = self.eyes(i) if refresh_eyes else self.last_features[i]
        return Percept(
            float(np.clip(food.mean(), 0, 1)),
            float(water.mean()),
            gradient(food),
            gradient(water),
            self.illumination(pos[:2], max(pos[2] + 0.3, 0.4)),
            gradient(heat_probes),
            float(np.mean(shade)),
            gradient(shade),
            threat,
            near,
            avoid,
            feature.copy(),
        )

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
