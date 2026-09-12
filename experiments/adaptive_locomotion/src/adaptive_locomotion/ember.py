"""Optional construction-machine palette; only observer rendering is changed."""

import mujoco
import numpy as np
from PIL import Image, ImageDraw

from . import graphite
from .record import font

CHARCOAL = "#1F1F1F"
STEEL = "#4B4847"
IVORY = "#E6E1DB"
YELLOW = "#FFC31F"
ORANGE = "#E88107"
GREEN = "#3F6B53"
RED = "#C64B3C"
PALETTE = {
    "charcoal": CHARCOAL,
    "steel": STEEL,
    "ivory": IVORY,
    "yellow": YELLOW,
    "orange": ORANGE,
    "green": GREEN,
    "red": RED,
}


def rgba(color):
    return [int(color[k : k + 2], 16) / 255 for k in (1, 3, 5)] + [1.0]


def configure(model):
    graphite.configure(model)
    for name, color, specular in (
        ("white", YELLOW, 0.28),
        ("gray", STEEL, 0.4),
        ("metal", IVORY, 0.6),
        ("black", CHARCOAL, 0.12),
    ):
        index = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MATERIAL, name)
        if index >= 0:
            model.mat_rgba[index] = rgba(color)
            model.mat_specular[index] = specular
    for i in range(model.ngeom):
        name = model.geom(i).name or ""
        if name == "floor":
            model.geom_rgba[i] = [0.17, 0.165, 0.16, 1]
        elif name.endswith("_member"):
            model.geom_rgba[i] = [0.62, 0.60, 0.57, 1]
        elif name.endswith("_terminal"):
            model.geom_rgba[i] = rgba(CHARCOAL)
        elif name == "deck_surface":
            model.geom_rgba[i] = rgba(STEEL)
        elif name == "deck_fixture":
            model.geom_rgba[i] = rgba(ORANGE)
        elif model.geom_bodyid[i] == 0 and name != "floor":
            model.geom_rgba[i] = rgba(STEEL)


def decorate(scene, model, data, body):
    graphite.decorate(
        scene,
        model,
        data,
        body,
        panel_color=rgba(YELLOW),
        seam_color=rgba(ORANGE),
        nose_color=rgba(CHARCOAL),
        marker_color=rgba(RED),
    )
    # Edge labels are decorations: no new contact, mass, sensing or support.
    deck = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "deck_surface")
    if deck >= 0:
        rotation = data.geom_xmat[deck].reshape(3, 3)
        center = data.geom_xpos[deck]
        sx, sy, sz = model.geom_size[deck]
        for sign in (-1, 1):
            graphite.geom(
                scene,
                mujoco.mjtGeom.mjGEOM_BOX,
                [sx - 0.015, 0.012, 0.001],
                center + rotation @ [0, sign * (sy - 0.025), sz + 0.002],
                rotation,
                rgba(YELLOW),
            )
            for x in np.arange(-sx + 0.1, sx - 0.04, 0.16):
                graphite.geom(
                    scene,
                    mujoco.mjtGeom.mjGEOM_BOX,
                    [0.035, 0.013, 0.001],
                    center + rotation @ [x, sign * (sy - 0.025), sz + 0.004],
                    rotation,
                    rgba(CHARCOAL),
                )


def layout(
    main, right_top, right_bottom, *, title, subtitle, seconds, drift=None, contact=None
):
    canvas = Image.new("RGB", (1920, 1080), CHARCOAL)
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((24, 26, 35, 118), fill=YELLOW)
    draw.text((56, 24), "FIELD ROBOTICS  /  EMBER", font=font(20), fill=ORANGE)
    draw.text((54, 51), title, font=font(37), fill=IVORY)
    draw.text((56, 104), subtitle, font=font(21), fill=IVORY)
    canvas.paste(main, (24, 150))
    canvas.paste(right_top, (1304, 150))
    canvas.paste(right_bottom, (1304, 586))
    for x, y, label in (
        (42, 168, "01 / OBSERVER"),
        (1322, 168, "02 / OVERHEAD"),
        (1322, 604, "03 / HEAD CAMERA"),
    ):
        draw.rectangle((x - 6, y - 5, x + 238, y + 29), fill=CHARCOAL)
        draw.rectangle((x - 6, y - 5, x - 2, y + 29), fill=YELLOW)
        draw.text((x + 7, y), label, font=font(18), fill=IVORY)
    draw.line((24, 1010, 1896, 1010), fill=STEEL, width=2)
    draw.text((32, 1030), f"{seconds:05.2f} s   /   1x", font=font(24), fill=IVORY)
    draw.text((280, 1032), "LEARNED BALANCE", font=font(21), fill=YELLOW)
    if drift is not None:
        draw.text(
            (565, 1032),
            f"DECK DRIFT  {drift * 100:04.1f} cm",
            font=font(21),
            fill=IVORY,
        )
    if contact is not None:
        for i, (label, load) in enumerate(
            zip(("FL", "FR", "RL", "RR"), contact, strict=True)
        ):
            x = 1410 + i * 118
            draw.ellipse((x, 1035, x + 13, 1048), fill=GREEN if load > 1 else STEEL)
            draw.text((x + 22, 1029), label, font=font(22), fill=IVORY)
    return canvas
