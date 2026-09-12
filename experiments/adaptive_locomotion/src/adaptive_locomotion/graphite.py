"""Opt-in neutral presentation; decorations live only in the observer scene."""

from itertools import pairwise

import mujoco
import numpy as np
from PIL import Image, ImageDraw

from .presentation import configure as classic_configure
from .presentation import damage_markers
from .record import font

BG, PANEL, LINE = "#08090b", "#131518", "#30343a"
WHITE, MUTED, TEAL, ORANGE = "#edeff1", "#929aa3", "#49d6c5", "#ff6b23"


def configure(model):
    """Materials/light only. The opt-in theme never changes physical parameters."""
    classic_configure(model)
    model.vis.headlight.ambient[:] = 0.35
    model.vis.headlight.diffuse[:] = 0.25
    model.vis.headlight.specular[:] = 0.10
    model.vis.quality.offsamples = 8
    model.geom_rgba[model.geom("floor").id] = [0.13, 0.135, 0.14, 1]
    for name, rgba, specular, shininess in (
        ("gray", [0.52, 0.54, 0.56, 1], 0.45, 0.65),
        ("metal", [0.72, 0.74, 0.76, 1], 0.65, 0.75),
        ("black", [0.035, 0.039, 0.043, 1], 0.15, 0.45),
        ("white", [0.70, 0.72, 0.74, 1], 0.35, 0.6),
    ):
        idx = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MATERIAL, name)
        if idx >= 0:
            model.mat_rgba[idx] = rgba
            model.mat_specular[idx] = specular
            model.mat_shininess[idx] = shininess
    base = model.body("base").id
    for i in range(model.ngeom):
        body = int(model.geom_bodyid[i])
        while body not in (0, base):
            body = int(model.body_parentid[body])
        name = model.geom(i).name or ""
        if body == base:
            if name.endswith("_member"):
                model.geom_rgba[i] = [0.58, 0.60, 0.62, 1]
            elif name.endswith("_terminal"):
                model.geom_rgba[i] = [0.13, 0.14, 0.15, 1]
        elif name == "deck_surface":
            model.geom_rgba[i] = [0.085, 0.31, 0.32, 1]
        elif name == "deck_fixture":
            model.geom_rgba[i] = [0.055, 0.06, 0.065, 1]
        elif name != "floor" and model.geom_contype[i] == 0:
            model.geom_rgba[i] = [0.20, 0.23, 0.24, 1]
        elif name != "floor":
            # Physical terrain stays subtly teal against the neutral floor.
            model.geom_rgba[i] = [0.10, 0.29, 0.30, 1]
    model.light_bodyid[0] = base
    model.light_mode[0] = mujoco.mjtCamLight.mjCAMLIGHT_TRACK
    model.light_pos[0] = [1.2, -1.8, 3.0]
    direction = np.array([-1.2, 1.8, -3.0])
    model.light_dir[0] = direction / np.linalg.norm(direction)
    model.light_cutoff[0] = 35
    model.light_ambient[0] = [0.06, 0.06, 0.06]
    model.light_diffuse[0] = [1.0, 1.0, 1.0]
    model.light_specular[0] = [0.35, 0.35, 0.35]


def geom(scene, kind, size, position, rotation, rgba, emission=0):
    if scene.ngeom >= scene.maxgeom:
        raise RuntimeError("Observer scene decoration capacity exceeded")
    item = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        item,
        kind,
        np.asarray(size),
        np.asarray(position),
        np.asarray(rotation).ravel(),
        np.asarray(rgba, dtype=np.float32),
    )
    item.category = mujoco.mjtCatBit.mjCAT_DECOR
    item.emission, item.specular, item.shininess = emission, 0.35, 0.65
    scene.ngeom += 1


def decorate(
    scene,
    model,
    data,
    body,
    *,
    panel_color=(0.22, 0.235, 0.25, 1),
    seam_color=(0.25, 0.7, 0.65, 1),
    nose_color=(0.10, 0.11, 0.12, 1),
    marker_color=(1.0, 0.24, 0.015, 1.0),
):
    """Blank cosmetic plates cover raised/recessed branding on both body sides."""
    base = model.body("base").id
    origin, rotation = data.xpos[base], data.xmat[base].reshape(3, 3)
    for sign in (-1, 1):
        # A thin faceted skin follows the original body's side curvature,
        # rather than a rectangular plate protruding above its silhouette.
        profile = (
            (0.003, 0.098),
            (0.015, 0.098),
            (0.025, 0.0965),
            (0.035, 0.0925),
            (0.045, 0.086),
            (0.052, 0.077),
            (0.058, 0.069),
        )
        for (za, ya), (zb, yb) in pairwise(profile):
            dy, dz = sign * (yb - ya), zb - za
            theta = np.arctan2(-dy, dz)
            local = np.array(
                [
                    [1, 0, 0],
                    [0, np.cos(theta), -np.sin(theta)],
                    [0, np.sin(theta), np.cos(theta)],
                ]
            )
            geom(
                scene,
                mujoco.mjtGeom.mjGEOM_BOX,
                [0.112, 0.0015, np.hypot(dy, dz) / 2 + 0.00025],
                origin + rotation @ [0, sign * (ya + yb) / 2, (za + zb) / 2],
                rotation @ local,
                panel_color,
            )
        # Fine lower seam accent, part of the same cosmetic side plate.
        geom(
            scene,
            mujoco.mjtGeom.mjGEOM_BOX,
            [0.094, 0.0018, 0.001],
            origin + rotation @ [0, sign * 0.098, 0.0045],
            rotation,
            seam_color,
            0.12,
        )
        # Small angled nose plate covers the manufacturer's forward wordmark.
        angle = sign * -0.55
        local_rot = np.array(
            [
                [np.cos(angle), -np.sin(angle), 0],
                [np.sin(angle), np.cos(angle), 0],
                [0, 0, 1],
            ]
        )
        geom(
            scene,
            mujoco.mjtGeom.mjGEOM_BOX,
            [0.029, 0.004, 0.017],
            origin + rotation @ [0.287, sign * 0.050, 0.037],
            rotation @ local_rot,
            nose_color,
        )
    damage_markers(scene, model, data, body, marker_color)


def layout(
    main, right_top, right_bottom, *, title, subtitle, seconds, drift=None, contact=None
):
    """A restrained 1080p observation layout with accents reserved for state."""
    canvas = Image.new("RGB", (1920, 1080), BG)
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((32, 30, 38, 90), fill=TEAL)
    draw.text((58, 23), "MUJOCO / ADAPTIVE LOCOMOTION", font=font(20), fill=MUTED)
    draw.text((56, 52), title, font=font(37), fill=WHITE)
    draw.text((56, 104), subtitle, font=font(21), fill=MUTED)
    canvas.paste(main, (24, 150))
    canvas.paste(right_top, (1304, 150))
    canvas.paste(right_bottom, (1304, 586))
    for x, y, label in (
        (42, 168, "01 / OBSERVER"),
        (1322, 168, "02 / OVERHEAD"),
        (1322, 604, "03 / HEAD CAMERA"),
    ):
        draw.rounded_rectangle((x - 6, y - 5, x + 230, y + 27), radius=3, fill=PANEL)
        draw.text((x + 5, y), label, font=font(18), fill=WHITE)
    draw.line((24, 1010, 1896, 1010), fill=LINE, width=1)
    draw.text((32, 1030), f"{seconds:05.2f} s   /   1x", font=font(24), fill=WHITE)
    draw.text((280, 1032), "LEARNED BALANCE", font=font(21), fill=TEAL)
    if drift is not None:
        draw.text(
            (565, 1032),
            f"DECK DRIFT  {drift * 100:04.1f} cm",
            font=font(21),
            fill=MUTED,
        )
    if contact is not None:
        for i, (label, load) in enumerate(
            zip(("FL", "FR", "RL", "RR"), contact, strict=True)
        ):
            x = 1410 + i * 118
            draw.ellipse((x, 1035, x + 13, 1048), fill=TEAL if load > 1 else LINE)
            draw.text((x + 22, 1029), label, font=font(22), fill=WHITE)
    return canvas
