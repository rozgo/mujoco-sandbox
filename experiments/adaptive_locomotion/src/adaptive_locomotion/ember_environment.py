"""Industrial test-bay presentation candidate; all added geometry is observer-only."""

import mujoco
import numpy as np

from . import ember, graphite

IDENTITY = np.eye(3)
BLACK = [0.035, 0.035, 0.033, 1]
PLATE = [0.30, 0.295, 0.28, 1]
RIM = [0.16, 0.16, 0.15, 1]
SEAM = [0.07, 0.07, 0.065, 1]
METAL = [0.51, 0.50, 0.47, 1]
YELLOW = ember.rgba(ember.YELLOW)


def configure(model):
    ember.configure(model)
    model.geom_rgba[model.geom("floor").id] = [0.12, 0.12, 0.115, 1]
    base = model.body("base").id
    for i in range(model.ngeom):
        name = model.geom(i).name or ""
        body = int(model.geom_bodyid[i])
        while body not in (0, base):
            body = int(model.body_parentid[body])
        if body == 0 and model.geom_contype[i] == 0:
            # Replace old floor and deck paint, including nested environment bodies.
            model.geom_rgba[i, 3] = 0
        if name.startswith("standing_") or name == "deck_surface":
            model.geom_rgba[i] = RIM


def box(scene, size, position, color, rotation=IDENTITY, emission=0):
    graphite.geom(
        scene, mujoco.mjtGeom.mjGEOM_BOX, size, position, rotation, color, emission
    )


def hazard(scene, origin, rotation, length, width=0.028, pitch=0.115):
    """Diagonal yellow/black paint, confined to a narrow rectangular strip."""
    origin = np.asarray(origin)
    box(scene, [length / 2, width / 2, 0.00035], origin, BLACK, rotation)
    # Four aligned thin rows form each slanted stripe without overhanging edges.
    rows = 6
    for row in range(rows):
        y = -width / 2 + (row + 0.5) * width / rows
        for x in np.arange(-length / 2 - pitch, length / 2 + pitch, pitch):
            left = max(-length / 2, x + y)
            right = min(length / 2, x + y + pitch * 0.48)
            if right <= left:
                continue
            box(
                scene,
                [(right - left) / 2, width / rows / 2, 0.0002],
                origin + rotation @ [(left + right) / 2, y, 0.0006],
                YELLOW,
                rotation,
            )


def digit(scene, number, origin, scale=0.12):
    """Small floor distance stencil made from flush painted segments."""
    segments = {
        "a": (0.5, 1.8, 0.40, 0.065),
        "b": (0.95, 1.35, 0.065, 0.37),
        "c": (0.95, 0.45, 0.065, 0.37),
        "d": (0.5, 0, 0.40, 0.065),
        "e": (0.05, 0.45, 0.065, 0.37),
        "f": (0.05, 1.35, 0.065, 0.37),
        "g": (0.5, 0.9, 0.40, 0.065),
    }
    codes = (
        "abcdef",
        "bc",
        "abdeg",
        "abcdg",
        "bcfg",
        "acdfg",
        "acdefg",
        "abc",
        "abcdefg",
        "abcdfg",
    )
    for char in codes[number]:
        x, y, sx, sy = segments[char]
        box(
            scene,
            [sx * scale, sy * scale, 0.0002],
            np.asarray(origin) + [x * scale, y * scale, 0],
            METAL,
        )


def floor_details(scene, model, data):
    z = float(data.geom_xpos[model.geom("floor").id, 2])
    # Fixed world coordinates make motion and scale legible in mounted cameras.
    for x in np.arange(-4, 13, 1):
        box(scene, [0.003, 3.8, 0.0003], [x, 0, z + 0.0004], SEAM)
    for y in np.arange(-3, 4, 1):
        box(scene, [8, 0.003, 0.0003], [4, y, z + 0.0004], SEAM)
    for sign in (-1, 1):
        box(scene, [8, 0.009, 0.0003], [4, sign * 1.30, z + 0.001], YELLOW)
        box(scene, [8, 0.003, 0.0003], [4, sign * 1.36, z + 0.001], METAL)
        for x in np.arange(-3.5, 12, 0.5):
            box(scene, [0.002, 0.035, 0.0003], [x, sign * 1.34, z + 0.0015], METAL)
    for x in range(9):
        box(scene, [0.010, 0.10, 0.0003], [x, 1.18, z + 0.001], YELLOW)
        digit(scene, x, [x - 0.06, 1.45, z + 0.001])
    # Background shell is outside the demonstration envelope, not a new obstacle.
    for sign in (-1, 1):
        box(scene, [8, 0.035, 1.45], [4, sign * 3.7, z + 1.45], [0.09, 0.092, 0.087, 1])
        box(scene, [8, 0.065, 0.09], [4, sign * 3.65, z + 0.09], BLACK)
        box(scene, [8, 0.012, 0.015], [4, sign * 3.60, z + 0.25], YELLOW)
        for x in np.arange(-4, 13, 1):
            box(scene, [0.023, 0.065, 1.45], [x, sign * 3.63, z + 1.45], RIM)
        for x in (-2, 2, 6, 10):
            box(
                scene,
                [0.40, 0.01, 0.018],
                [x, sign * 3.58, z + 1.45],
                [0.68, 0.67, 0.59, 1],
                emission=0.35,
            )
    box(scene, [0.04, 3.7, 1.45], [-4, 0, z + 1.45], [0.095, 0.095, 0.09, 1])


def surface_details(scene, model, data, index):
    origin = data.geom_xpos[index]
    rotation = data.geom_xmat[index].reshape(3, 3)
    sx, sy, sz = model.geom_size[index]
    top = sz + 0.0006

    def local(size, position, color):
        box(scene, size, origin + rotation @ position, color, rotation)

    # Flush inset steel skin and seams retain the exact top-face silhouette.
    local([sx - 0.044, sy - 0.044, 0.0003], [0, 0, top], PLATE)
    for axis, extent, other in ((0, sx, sy), (1, sy, sx)):
        turn = IDENTITY if axis == 0 else np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        for sign in (-1, 1):
            p = (
                [0, sign * (sy - 0.018), top + 0.0004]
                if axis == 0
                else [sign * (sx - 0.018), 0, top + 0.0004]
            )
            hazard(scene, origin + rotation @ p, rotation @ turn, 2 * extent - 0.012)
            # Recessed side panel stays close to the original box face.
            if sz > 0.045:
                size = [extent - 0.035, 0.0006, max(0.012, sz - 0.032)]
                p = [0, sign * (other + 0.0005), 0]
                box(
                    scene,
                    size,
                    origin + rotation @ (turn @ p),
                    [0.105, 0.108, 0.102, 1],
                    rotation @ turn,
                )
    # Four corner fasteners, aligned to the actual tilted/moving surface normal.
    for x in (-sx + 0.062, sx - 0.062):
        for y in (-sy + 0.062, sy - 0.062):
            graphite.geom(
                scene,
                mujoco.mjtGeom.mjGEOM_CYLINDER,
                [0.008, 0.0008, 0],
                origin + rotation @ [x, y, top + 0.001],
                rotation,
                METAL,
            )
            local([0.004, 0.001, 0.0002], [x, y, top + 0.002], BLACK)
    # Panel divisions stay understated relative to the safety border.
    if sx > 0.5:
        local([0.0015, sy - 0.055, 0.0002], [0, 0, top + 0.0005], SEAM)
    for sign in (-1, 1):
        local(
            [sx - 0.055, 0.0012, 0.0002], [0, sign * (sy - 0.046), top + 0.0005], METAL
        )


def decorate(scene, model, data, body=None):
    if body is not None:
        graphite.decorate(
            scene,
            model,
            data,
            body,
            panel_color=ember.rgba(ember.YELLOW),
            seam_color=ember.rgba(ember.ORANGE),
            nose_color=ember.rgba(ember.CHARCOAL),
            marker_color=ember.rgba(ember.RED),
        )
    floor_details(scene, model, data)
    for i in range(model.ngeom):
        name = model.geom(i).name or ""
        if name == "deck_surface" or name.startswith(
            ("standing_slope", "standing_pad_", "standing_step_")
        ):
            surface_details(scene, model, data, i)
