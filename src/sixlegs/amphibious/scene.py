"""Go2 with physical leg-mounted floats and a smooth-bank test basin."""

import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from sixlegs.scene import ROOT, add, camera, vec

VENDOR = ROOT / "assets/menagerie/unitree_go2"
LEGS = ("FL", "FR", "RL", "RR")
WATER_Z = -0.14
DT = 0.002
START_X, FINISH_X = -2.55, 3.95
FLOAT_RADIUS, FLOAT_HALF_LENGTH = 0.085, 0.10
FLOAT_MASS = 0.45


def bed_height(x):
    if x < -1.6:
        return 0.0
    if x < -0.2:
        return -0.38 * (x + 1.6) / 1.4
    if x < 1.8:
        return -0.38
    if x < 3.2:
        return -0.38 + 0.38 * (x - 1.8) / 1.4
    return 0.0


def ground_height(x, y):
    return float(bed_height(x))


def build_xml():
    root = ET.parse(VENDOR / "go2.xml").getroot()
    root.set("model", "amphibious_go2_attachment_demo")
    root.find("compiler").set("meshdir", str(VENDOR / "assets"))
    root.find("option").attrib.update(
        timestep=str(DT), integrator="implicitfast", iterations="80"
    )
    root.remove(root.find("keyframe"))
    # Preserve vendor rigid-body properties. New float child bodies contribute
    # their own mass and inertia instead of overriding the thigh inertials.
    visual = add(root, "visual")
    add(visual, "global", offwidth="1920", offheight="1080")
    add(visual, "quality", shadowsize="2048", offsamples="4")
    add(visual, "headlight", ambient=".35 .35 .35", diffuse=".6 .6 .6")
    asset = root.find("asset")
    add(
        asset,
        "texture",
        name="sky",
        type="skybox",
        builtin="gradient",
        rgb1=".21 .30 .37",
        rgb2=".08 .12 .17",
        width="512",
        height="3072",
    )
    for name, color in {
        "sand": ".55 .49 .36 1",
        "bed": ".23 .34 .33 1",
        "float": ".06 .23 .28 1",
        "strap": ".68 .78 .75 1",
        "accent": ".98 .62 .19 1",
        "water": ".12 .48 .58 .38",
    }.items():
        add(
            asset,
            "material",
            name=name,
            rgba=color,
            specular=".35",
            shininess=".3",
            reflectance=".12" if name == "water" else "0",
        )
    world = root.find("worldbody")
    base = world.find("body[@name='base']")
    base.set("pos", f"{START_X} 0 .31")
    base.find("freejoint").set("name", "floating_base")
    # Keep vendor visuals but make collision primitives hidden in the renderer.
    for geom in root.findall(".//default[@class='collision']/geom"):
        geom.set("friction", "1.0 .01 .001")
        geom.set("condim", "3")
    foot = root.find(".//default[@class='foot']/geom")
    foot.set("solimp", ".95 .99 .001")
    foot.set("solref", ".01 1")
    foot.set("margin", "0")
    for leg in LEGS:
        calf = root.find(f".//body[@name='{leg}_calf']")
        add(
            calf, "site", name=leg + "_toe", pos="-.002 0 -.213", size=".006", group="4"
        )
        thigh = root.find(f".//body[@name='{leg}_thigh']")
        sign = 1 if leg.endswith("L") else -1
        pod = add(thigh, "body", name=leg + "_float", pos=f"0 {sign * 0.14} -.07")
        add(
            pod,
            "geom",
            name=leg + "_float_shell",
            type="capsule",
            size=f"{FLOAT_RADIUS} {FLOAT_HALF_LENGTH}",
            material="float",
            mass=str(FLOAT_MASS),
            group="0",
            condim="3",
            friction=".7 .01 .001",
            margin="0",
        )
        for z in (-0.065, 0.065):
            add(
                pod,
                "geom",
                type="cylinder",
                size=".087 .013",
                pos=f"0 0 {z}",
                material="strap",
                contype="0",
                conaffinity="0",
                mass="0",
                group="0",
            )
        add(
            pod,
            "geom",
            type="sphere",
            size=".025",
            pos="0 0 .171",
            material="accent",
            contype="0",
            conaffinity="0",
            mass="0",
            group="0",
        )
        add(
            thigh,
            "geom",
            name=leg + "_float_bracket",
            type="capsule",
            fromto=f"0 0 -.07 0 {sign * 0.14} -.07",
            size=".012",
            material="strap",
            mass="0",
            contype="0",
            conaffinity="0",
            group="0",
        )
    camera(base, "front", (0.31, 0, 0.055), (3, 0, -0.12), 65)
    for side, y in (("left", 0.10), ("right", -0.10)):
        add(
            base,
            "site",
            name=side + "_thruster",
            pos=f"-.21 {y} -.06",
            size=".008",
            group="4",
        )
        add(
            base,
            "geom",
            type="cylinder",
            size=".024 .035",
            pos=f"-.21 {y} -.06",
            quat="1 0 1 0",
            material="accent",
            mass="0",
            contype="0",
            conaffinity="0",
            group="0",
        )
    # Fixed terrain includes true convex ramp geometry.
    add(
        world,
        "geom",
        name="underlay",
        type="plane",
        size="8 4 .1",
        pos="0 0 -.8",
        material="sand",
    )
    for name, x, hx, z in (
        ("left_bank", -2.65, 1.05, 0.0),
        ("pool_bed", 0.8, 1.0, -0.38),
        ("right_bank", 4.05, 0.85, 0.0),
    ):
        add(
            world,
            "geom",
            name=name,
            type="box",
            pos=f"{x} 0 {z - 0.10}",
            size=f"{hx} 1.35 .1",
            material="sand" if z == 0 else "bed",
            friction="1 .01 .001",
        )
    for name, x0, x1, z0, z1 in (
        ("entry", -1.6, -0.2, 0.0, -0.38),
        ("exit", 1.8, 3.2, -0.38, 0.0),
    ):
        verts = [
            [x, y, z]
            for x, z in ((x0, -0.8), (x1, -0.8), (x0, z0), (x1, z1))
            for y in (-1.35, 1.35)
        ]
        add(asset, "mesh", name=name + "_ramp", vertex=vec(np.array(verts).ravel()))
        add(
            world,
            "geom",
            name=name + "_ramp",
            type="mesh",
            mesh=name + "_ramp",
            material="sand",
            friction="1 .01 .001",
        )
    add(
        world,
        "geom",
        name="water_surface",
        type="box",
        size="2.05 1.345 .002",
        pos=f".8 0 {WATER_Z}",
        material="water",
        contype="0",
        conaffinity="0",
        group="1",
    )
    for y in (-1.42, 1.42):
        add(
            world,
            "geom",
            name="basin_wall_" + str(y),
            type="box",
            size="4.3 .06 .39",
            pos=f".6 {y} -.41",
            rgba=".28 .34 .35 1",
        )
    for x in (-3.35, 4.55):
        for y in (-0.6, 0.6):
            add(
                world,
                "geom",
                type="cylinder",
                size=".045 .16",
                pos=f"{x} {y} .16",
                material="accent",
            )
    add(world, "light", pos="-2 -4 7", dir=".2 .3 -1", diffuse=".95 .90 .80")
    add(
        world,
        "light",
        pos="4 3 5",
        dir="-.3 -.4 -1",
        diffuse=".55 .68 .80",
        castshadow="false",
    )
    camera(world, "overview", (6, -8, 5.5), (0.7, 0, -0.1), 46)
    camera(world, "side", (0.8, -7, 1.5), (0.8, 0, -0.08), 43)
    camera(world, "detail", (-1.25, -1.7, 1.0), (START_X, 0, 0.17), 45)
    camera(world, "overhead", (0.8, -0.001, 7), (0.8, 0, 0), 60)
    for actuator in root.find("actuator"):
        limit = 45.43 if actuator.get("class") == "knee" else 23.7
        actuator.set("forcerange", f"{-limit} {limit}")
    ET.indent(root, space="  ")
    return ET.tostring(root, encoding="unicode")


def load_scene():
    xml = build_xml()
    path = ROOT / "build/amphibious_scene.xml"
    path.parent.mkdir(exist_ok=True)
    path.write_text(xml)
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    d.qpos[7:] = np.tile([0, 0.82, -1.64], 4)
    mujoco.mj_forward(m, d)
    # Start with foot surfaces resting on the level bank.
    d.qpos[2] -= min(d.site(leg + "_toe").xpos[2] for leg in LEGS) - 0.022
    mujoco.mj_forward(m, d)
    return m, d
