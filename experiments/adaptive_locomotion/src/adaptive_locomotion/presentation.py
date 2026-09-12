"""Observer-only damage markers and contact-readable lighting.

Markers exist only in MjvScene: they add no mass, collision or sensor geometry.
Lighting follows MuJoCo's shadow-map guidance:
https://mujoco.readthedocs.io/en/stable/XMLreference.html#visual-quality
"""

import xml.etree.ElementTree as ET
from functools import lru_cache

import mujoco
import numpy as np

from .bodies import VENDOR


def theme_hooks(theme="classic"):
    if theme == "classic":
        return configure, damage_markers
    if theme == "graphite":
        from .graphite import configure as configure_graphite
        from .graphite import decorate

        return configure_graphite, decorate
    raise ValueError(f"Unknown presentation theme: {theme}")


@lru_cache
def attachment_points():
    root = ET.parse(VENDOR / "go2.xml").getroot()
    return {
        leg: np.fromstring(root.find(f".//body[@name='{leg}_hip']").get("pos"), sep=" ")
        for leg in ("FL", "FR", "RL", "RR")
    }


def configure(model):
    """Only rendering properties change; call before allocating the renderer."""
    model.vis.quality.shadowsize = 4096
    model.vis.headlight.ambient[:] = 0.20
    model.vis.headlight.diffuse[:] = 0.25
    model.vis.headlight.specular[:] = 0.10
    model.vis.map.shadowscale = 1.0
    model.geom_rgba[model.geom("floor").id] = [0.42, 0.47, 0.51, 1]
    base = model.body("base").id
    model.light_bodyid[0] = base
    model.light_mode[0] = mujoco.mjtCamLight.mjCAMLIGHT_FIXED
    model.light_type[0] = mujoco.mjtLightType.mjLIGHT_SPOT
    model.light_pos[0] = [-1.0, -1.4, 2.8]
    direction = np.array([1.0, 1.4, -2.8])
    model.light_dir[0] = direction / np.linalg.norm(direction)
    model.light_cutoff[0] = 40
    model.light_castshadow[0] = True
    model.light_ambient[0] = [0.05, 0.05, 0.05]
    model.light_diffuse[0] = [0.95, 0.92, 0.86]
    model.light_specular[0] = [0.12, 0.12, 0.12]


def camera_azimuth(body):
    # MuJoCo's azimuth convention: -50 exposes the left side, +50 the right.
    return (
        -50
        if any(leg.endswith("L") for leg in (*body.absent, *body.absent_legs))
        else 50
    )


def damage_markers(scene, model, data, body):
    """Append opaque 3D annotation spheres, anchored to actual cut locations."""
    points = [
        (data.geom_xpos[model.geom(f"{leg}_stump").id], 0.028) for leg in body.absent
    ]
    base = model.body("base").id
    rotation = data.xmat[base].reshape(3, 3)
    points += [
        (data.xpos[base] + rotation @ attachment_points()[leg], 0.040)
        for leg in body.absent_legs
    ]
    for pos, radius in points:
        if scene.ngeom >= scene.maxgeom:
            raise RuntimeError("Scene has no room for the damage marker")
        geom = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(
            geom,
            mujoco.mjtGeom.mjGEOM_SPHERE,
            np.full(3, radius),
            pos,
            np.eye(3).ravel(),
            np.array([1.0, 0.24, 0.015, 1.0], dtype=np.float32),
        )
        geom.category = mujoco.mjtCatBit.mjCAT_DECOR
        geom.emission = 0.45
        geom.specular = 0.15
        scene.ngeom += 1
    return len(points)
