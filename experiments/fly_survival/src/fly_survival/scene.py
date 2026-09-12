"""Physical arena, in FlyGym's millimeter / gram / second units."""

import math
from dataclasses import dataclass

import mujoco as mj
import numpy as np
from flygym import Simulation
from flygym.compose import FlatGroundWorld
from flygym.utils.math import Rotation3D
from flygym_demo.complex_terrain import make_locomotion_fly

COLORS = {
    "floor": (0.095, 0.105, 0.105, 1),
    "steel": (0.27, 0.29, 0.29, 1),
    "dark": (0.045, 0.052, 0.052, 1),
    "yellow": (1, 0.76, 0.06, 1),
    "food": (0.43, 0.63, 0.29, 1),
    "water": (0.25, 0.65, 0.70, 1),
    "red": (0.83, 0.22, 0.15, 1),
    "ivory": (0.80, 0.79, 0.72, 1),
}
RESOURCE_POS = np.array([[-17.0, 9.0], [13.0, 8.0], [-10.0, -11.0]])
SHELTERS = np.array([[-19.0, -3.0], [8.0, -11.0]])
LAMP_POS = np.array([13.0, 8.0])
SPAWNS = [
    (-14, -7, 0.1),
    (-6, -4, 1.0),
    (1, -4, 2.0),
    (15, -5, 2.5),
    (-8, 6, -0.3),
    (0, 9, -0.8),
    (8, 1, 1.2),
    (19, 1, 2.7),
]
TERRAIN_BIT = 1 << 8


def geom(body, name, kind, pos, size, color="steel", physical=False, **kwargs):
    # Observer paint and ribs must never change moving-body mass/inertia.
    if not physical and "mass" not in kwargs:
        kwargs["mass"] = 0
    return body.add_geom(
        name=name,
        type=getattr(mj.mjtGeom, "mjGEOM_" + kind.upper()),
        pos=pos,
        size=size,
        rgba=COLORS.get(color, color),
        contype=TERRAIN_BIT if physical else 0,
        conaffinity=255 if physical else 0,
        friction=(1.0, 0.005, 0.0001),
        **kwargs,
    )


def stripe_edge(body, name, x, y, length, along="x", z=0.035):
    for i in range(int(length / 1.3)):
        offset = -length / 2 + (i + 0.5) * 1.3
        pos = (x + offset, y, z) if along == "x" else (x, y + offset, z)
        size = (0.43, 0.20, 0.012) if along == "x" else (0.20, 0.43, 0.012)
        geom(body, f"{name}_{i}", "box", pos, size, "yellow" if i % 2 == 0 else "dark")


@dataclass
class Arena:
    sim: Simulation
    flies: list
    swat_joint: int
    swat_actuator: int
    fly_body_ids: list


def build(n_flies=8):
    if not 1 <= n_flies <= 8:
        raise ValueError("The shared arena supports one through eight flies")
    world = FlatGroundWorld(half_size=80)
    root = world.mjcf_root
    wb = root.worldbody
    world.ground_geom.material = ""
    world.ground_geom.rgba = COLORS["floor"]
    world.ground_geom.contype = TERRAIN_BIT
    world.ground_geom.conaffinity = 255
    for texture in root.textures:
        if texture.name == "skybox":
            texture.rgb1 = [0.025, 0.033, 0.035]
            texture.rgb2 = [0.025, 0.033, 0.035]
    # Every relief geom is explicitly either physical or observer decoration.
    geom(wb, "slab", "box", (0, 0, -0.82), (27, 19, 0.8), "dark")
    for axis, value, size in [
        (0, -26, (0.35, 18, 1.2)),
        (0, 26, (0.35, 18, 1.2)),
        (1, -18, (26, 0.35, 1.2)),
        (1, 18, (26, 0.35, 1.2)),
    ]:
        pos = [0, 0, 1.2]
        pos[axis] = value
        geom(wb, f"rail_{axis}_{value}", "box", pos, size, "steel", True)
    stripe_edge(wb, "north", 0, 17.3, 51)
    stripe_edge(wb, "south", 0, -17.3, 51)
    stripe_edge(wb, "west", -25.3, 0, 34, "y")
    stripe_edge(wb, "east", 25.3, 0, 34, "y")
    for x in range(-24, 25, 6):
        geom(wb, f"seam_x_{x}", "box", (x, 0, 0.012), (0.012, 17, 0.008), "steel")
    for y in range(-16, 17, 4):
        geom(wb, f"seam_y_{y}", "box", (0, y, 0.012), (25, 0.012, 0.008), "steel")
    for i, (x, y) in enumerate(RESOURCE_POS):
        col = "water" if i == 2 else "food"
        geom(
            wb,
            f"resource_base_{i}",
            "cylinder",
            (x, y, 0.10),
            (2.4, 0.10, 0),
            "steel",
            True,
        )
        geom(wb, f"resource_{i}", "cylinder", (x, y, 0.23), (2.1, 0.035, 0), col)
        for k in range(12):
            a = k * math.pi / 6
            geom(
                wb,
                f"resource_tick_{i}_{k}",
                "box",
                (x + 2.8 * math.cos(a), y + 2.8 * math.sin(a), 0.025),
                (0.18, 0.10, 0.01),
                col,
            )
    for i, (x, y) in enumerate(SHELTERS):
        geom(
            wb,
            f"shelter_roof_{i}",
            "box",
            (x, y, 3.25),
            (4.2, 3.0, 0.25),
            "steel",
            True,
        )
        geom(wb, f"shelter_top_{i}", "box", (x, y, 3.51), (3.6, 2.4, 0.025), "dark")
        for dx in (-3.8, 3.8):
            for dy in (-2.6, 2.6):
                geom(
                    wb,
                    f"shelter_post_{i}_{dx}_{dy}",
                    "box",
                    (x + dx, y + dy, 1.5),
                    (0.18, 0.18, 1.5),
                    "steel",
                    True,
                )
        stripe_edge(wb, f"shelter_edge_{i}", x, y - 2.8, 8, z=3.55)
        geom(
            wb,
            f"refuge_floor_{i}",
            "box",
            (x, y, 0.028),
            (3.6, 2.3, 0.01),
            (0.13, 0.20, 0.17, 1),
        )
    # Visible heat fixture. Heat transport is separately modeled, never implicit in lights.
    geom(wb, "lamp_mast", "box", (23, 13, 9), (0.3, 0.3, 9), "steel", True)
    geom(wb, "lamp_boom", "box", (18, 13, 17.8), (5, 0.3, 0.3), "steel")
    geom(wb, "lamp_crossbar", "box", (13, 10.5, 17.8), (0.3, 2.5, 0.3), "steel")
    geom(wb, "lamp_stem", "cylinder", (13, 8, 16.6), (0.2, 1.2, 0), "steel")
    geom(wb, "lamp_head", "cylinder", (13, 8, 15), (2.4, 0.6, 0), "dark")
    geom(
        wb,
        "lamp_emitter",
        "cylinder",
        (13, 8, 14.35),
        (2.1, 0.08, 0),
        (1, 0.25, 0.08, 1),
    )
    for k in range(48):
        a = k * 2 * math.pi / 48
        geom(
            wb,
            f"heat_boundary_{k}",
            "box",
            (13 + 5.8 * math.cos(a), 8 + 5.8 * math.sin(a), 0.032),
            (0.16, 0.08, 0.01),
            "red",
        )
    wb.add_light(
        name="heat_light",
        pos=(13, 8, 14),
        dir=(0, 0, -1),
        diffuse=(0.7, 0.16, 0.04),
        specular=(0.1, 0.02, 0),
        cutoff=27,
        castshadow=True,
    )
    # A torque-limited hinged swatter. Rest angle -0.82 lifts it clear of flies.
    geom(wb, "swatter_base", "box", (15, 19.5, 1.3), (2.2, 1.2, 1.3), "dark")
    swat = wb.add_body(name="swatter", pos=(15, 18, 1.2))
    swat.add_joint(
        name="swatter_hinge",
        type=mj.mjtJoint.mjJNT_HINGE,
        axis=(1, 0, 0),
        limited=True,
        range=(-1.25, 0.035),
        damping=80,
        armature=0.1,
    )
    geom(
        swat,
        "swatter_arm",
        "box",
        (0, -6, 0),
        (0.4, 6, 0.22),
        "steel",
        True,
        mass=0.015,
    )
    geom(
        swat,
        "swatter_pad",
        "box",
        (0, -12, 0),
        (4.7, 3.6, 0.18),
        "dark",
        True,
        mass=0.025,
    )
    for y in (-15.3, -8.7):
        stripe_edge(swat, f"swat_edge_{y}", 0, y, 9, z=0.21)
    for x in (-3, -1, 1, 3):
        geom(swat, f"swat_rib_{x}", "box", (x, -12, 0.2), (0.035, 3.2, 0.035), "steel")
    act = root.add_actuator(
        name="swatter_drive",
        target="swatter_hinge",
        trntype=mj.mjtTrn.mjTRN_JOINT,
        gaintype=mj.mjtGain.mjGAIN_FIXED,
        biastype=mj.mjtBias.mjBIAS_AFFINE,
        gainprm=[20000, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        biasprm=[0, -20000, -250, 0, 0, 0, 0, 0, 0, 0],
        ctrllimited=True,
        ctrlrange=(-1.25, 0.035),
        forcelimited=True,
        forcerange=(-12000, 12000),
    )
    flies = []
    for i in range(n_flies):
        fly = make_locomotion_fly(name=f"fly_{i:02d}", colorize=False)
        fly.add_vision()
        for segment, geoms in fly.bodyseg_to_mjcfgeom.items():
            for g in geoms:
                g.rgba = COLORS["ivory"] if segment.is_leg() else COLORS["steel"]
                if "eye" in segment.name:
                    g.rgba = COLORS["red"]
                if "wing" in segment.name:
                    g.rgba = (0.65, 0.69, 0.67, 0.45)
                g.contype = 1 << i
                g.conaffinity = TERRAIN_BIT | (255 ^ (1 << i))
        fly.add_tracking_camera(
            name="follow",
            pos_offset=(-5, -9, 5),
            rotation=Rotation3D("euler", (0.95, 0, -0.45)),
        )
        x, y, yaw = SPAWNS[i]
        world.add_fly(
            fly,
            (x, y, 1.05),
            Rotation3D("quat", (math.cos(yaw / 2), 0, 0, math.sin(yaw / 2))),
            add_ground_contact_sensors=False,
        )
        flies.append(fly)
    # add_fly inherits upstream global settings; set shared presentation after composition.
    root.visual.global_.offwidth = 1920
    root.visual.global_.offheight = 1200
    root.visual.quality.shadowsize = 4096
    root.visual.headlight.ambient = [0.25, 0.25, 0.25]
    root.visual.headlight.diffuse = [0.45, 0.45, 0.45]
    root.visual.map.znear = 0.001
    root.stat.extent = 60
    root.option.noslip_iterations = 0  # matched CPU/Warp supported solver configuration
    wb.add_light(
        name="key",
        pos=(-15, -8, 38),
        dir=(0.3, 0.2, -1),
        diffuse=(0.8, 0.82, 0.82),
        castshadow=True,
    )
    wb.add_light(
        name="fill",
        pos=(20, -10, 22),
        dir=(-0.5, 0.3, -1),
        diffuse=(0.3, 0.32, 0.34),
        castshadow=False,
    )
    sim = Simulation(world)
    model, data = sim.mj_model, sim.mj_data
    joint = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, "swatter_hinge")
    actuator = mj.mj_name2id(model, mj.mjtObj.mjOBJ_ACTUATOR, act.name)
    data.qpos[model.jnt_qposadr[joint]] = -0.82
    data.ctrl[:] = model.key_ctrl[0]
    data.ctrl[actuator] = -0.82
    mj.mj_forward(model, data)
    bids = [
        mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, f"{f.name}/c_thorax") for f in flies
    ]
    return Arena(sim, flies, joint, actuator, bids)


def render(arena, width=1500, height=1000, view="overview"):
    model, data = arena.sim.mj_model, arena.sim.mj_data
    camera = mj.MjvCamera()
    camera.type = mj.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = [0, 0, 1]
    camera.distance = 66
    camera.azimuth = 100
    camera.elevation = -57
    if view == "swatter":
        camera.lookat[:] = [14, 9, 4]
        camera.distance = 34
        camera.azimuth = 135
        camera.elevation = -30
    elif view == "refuge":
        camera.lookat[:] = [-17, -4, 1]
        camera.distance = 20
        camera.azimuth = 115
        camera.elevation = -24
    with mj.Renderer(model, height=height, width=width) as renderer:
        renderer.update_scene(data, camera=camera)
        renderer.scene.flags[mj.mjtRndFlag.mjRND_SHADOW] = True
        return renderer.render().copy()
