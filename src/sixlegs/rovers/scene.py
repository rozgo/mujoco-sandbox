"""An inspection yard and six independently driven, suspended RC vehicles."""

import xml.etree.ElementTree as ET

import mujoco

from sixlegs.scene import ROOT, add, camera

DT = 0.004
STARTS = [(-7.5, y) for y in (-5.4, -3.4, -1.4, 0.6, 2.6, 4.6)]
COLORS = [
    "0.12 0.76 0.78 1",
    "0.98 0.57 0.17 1",
    "0.38 0.55 0.96 1",
    "0.76 0.43 0.88 1",
    "0.47 0.78 0.35 1",
    "0.96 0.36 0.43 1",
]
# This static facility map is public. Marker locations below are simulator-only.
BUILDINGS = [
    (-0.6, -2.0, 1.1, 1.05, 1.1),
    (0.7, 2.2, 1.3, 0.95, 1.3),
    (4.0, 0.0, 0.2, 2.1, 0.65),
]
MARKERS = [
    (-5.2, -5.3),
    (-2.0, -5.2),
    (3.2, -5.0),
    (7.0, -2.8),
    (6.5, 3.8),
    (1.0, 5.5),
    (-3.8, 4.8),
    (-5.8, 0.7),
]
WHEEL_RADIUS = 0.09
WHEELBASE = 0.38
TRACK = 0.48


def rover(world, actuators, index):
    name = f"r{index}"
    x, y = STARTS[index]
    body = add(world, "body", name=name, pos=f"{x} {y} .18")
    add(body, "freejoint", name=name + "_free")
    add(
        body,
        "geom",
        name=name + "_chassis",
        type="box",
        size=".255 .135 .055",
        mass="2.2",
        rgba=COLORS[index],
    )
    add(
        body,
        "geom",
        type="box",
        size=".14 .115 .037",
        pos="-.04 0 .08",
        mass=".6",
        material="dark",
    )
    add(
        body,
        "geom",
        type="box",
        size=".05 .095 .035",
        pos=".18 0 .075",
        mass=".12",
        material="dark",
    )
    add(
        body,
        "geom",
        type="capsule",
        fromto="-.18 .09 .09 -.18 .09 .43",
        size=".006",
        mass=".025",
        material="dark",
        contype="0",
        conaffinity="0",
    )
    add(
        body,
        "geom",
        type="sphere",
        pos="-.18 .09 .44",
        size=".013",
        mass=".005",
        rgba=COLORS[index],
        contype="0",
        conaffinity="0",
    )
    add(
        body, "site", name=name + "_antenna", pos="-.18 .09 .44", size=".007", group="4"
    )
    add(body, "site", name=name + "_sensor", pos=".255 0 .085", size=".005", group="4")
    camera(body, name + "_front", (0.27, 0, 0.085), (3, 0, -0.15), 72)
    for ylight in (-0.08, 0.08):
        add(
            body,
            "geom",
            type="sphere",
            pos=f".259 {ylight} .02",
            size=".017",
            rgba=".9 1 .85 1",
            mass=".005",
            contype="0",
            conaffinity="0",
        )
    for front, xwheel in ((True, 0.19), (False, -0.19)):
        for left, ywheel in ((True, 0.24), (False, -0.24)):
            suffix = ("f" if front else "b") + ("l" if left else "r")
            suspension = add(
                body,
                "body",
                name=f"{name}_{suffix}_suspension",
                pos=f"{xwheel} {ywheel} -.09",
            )
            add(
                suspension,
                "joint",
                name=f"{name}_{suffix}_spring",
                type="slide",
                axis="0 0 1",
                range="-.025 .025",
                stiffness="1700",
                damping="24",
                armature=".005",
            )
            add(
                suspension,
                "inertial",
                pos="0 0 0",
                mass=".08",
                diaginertia=".0002 .0002 .0002",
            )
            carrier = add(suspension, "body", name=f"{name}_{suffix}_carrier")
            if front:
                joint = f"{name}_{suffix}_steer"
                add(
                    carrier,
                    "joint",
                    name=joint,
                    axis="0 0 1",
                    range="-.65 .65",
                    damping=".2",
                    armature=".003",
                )
                add(
                    carrier,
                    "inertial",
                    pos="0 0 0",
                    mass=".03",
                    diaginertia=".0001 .0001 .0001",
                )
                add(
                    actuators,
                    "position",
                    name=joint,
                    joint=joint,
                    kp="35",
                    kv="2",
                    ctrlrange="-.65 .65",
                    forcerange="-2 2",
                )
            wheel = add(carrier, "body", name=f"{name}_{suffix}_wheel")
            joint = f"{name}_{suffix}_drive"
            add(
                wheel,
                "joint",
                name=joint,
                axis="0 1 0",
                damping=".015",
                armature=".001",
            )
            add(
                wheel,
                "geom",
                name=joint + "_tire",
                type="cylinder",
                size=".09 .038",
                quat="1 1 0 0",
                mass=".18",
                material="rubber",
                friction="1.1 .01 .001",
                condim="4",
            )
            add(
                wheel,
                "geom",
                type="cylinder",
                size=".045 .039",
                quat="1 1 0 0",
                mass=".025",
                material="alloy",
                contype="0",
                conaffinity="0",
            )
            add(
                wheel,
                "geom",
                type="box",
                size=".008 .04 .045",
                mass=".005",
                material="dark",
                contype="0",
                conaffinity="0",
            )
            add(
                actuators,
                "velocity",
                name=joint,
                joint=joint,
                kv=".35",
                ctrlrange="-12 12",
                forcerange="-1.2 1.2",
            )


def load_scene():
    root = ET.Element("mujoco", model="rover_comms_inspection_yard")
    add(root, "compiler", angle="radian", autolimits="true")
    add(
        root,
        "option",
        timestep=DT,
        integrator="implicitfast",
        cone="elliptic",
        iterations="50",
    )
    visual = add(root, "visual")
    add(visual, "global", offwidth="1920", offheight="1080")
    add(visual, "quality", shadowsize="2048", offsamples="4")
    add(visual, "headlight", ambient=".4 .4 .4", diffuse=".7 .7 .7")
    default = add(root, "default")
    add(default, "geom", solref=".012 1", solimp=".95 .99 .001")
    assets = add(root, "asset")
    add(
        assets,
        "texture",
        name="sky",
        type="skybox",
        builtin="gradient",
        rgb1=".19 .24 .30",
        rgb2=".06 .08 .11",
        width="512",
        height="3072",
    )
    add(
        assets,
        "texture",
        name="grid",
        type="2d",
        builtin="checker",
        rgb1=".22 .27 .29",
        rgb2=".25 .30 .32",
        width="512",
        height="512",
    )
    add(
        assets,
        "material",
        name="ground",
        texture="grid",
        texrepeat="1 1",
        texuniform="true",
    )
    for name, color in {
        "dark": ".08 .11 .14 1",
        "rubber": ".035 .045 .05 1",
        "alloy": ".62 .68 .72 1",
        "building": ".53 .60 .63 1",
        "trim": ".24 .31 .35 1",
    }.items():
        add(assets, "material", name=name, rgba=color, specular=".3", shininess=".3")
    # Convex shallow ramp: physics geometry, not a texture.
    add(
        assets,
        "mesh",
        name="ramp",
        vertex="-1 -.7 0 -1 .7 0 1 -.7 0 1 .7 0 1 -.7 .18 1 .7 .18",
    )
    world = add(root, "worldbody")
    add(world, "geom", name="floor", type="plane", size="11 9 .1", material="ground")
    add(world, "light", pos="-5 -6 12", dir=".3 .4 -1", diffuse=".9 .86 .79")
    add(
        world,
        "light",
        pos="6 7 10",
        dir="-.3 -.3 -1",
        diffuse=".55 .68 .8",
        castshadow="false",
    )
    for i, (x, y, hx, hy, height) in enumerate(BUILDINGS):
        body = add(world, "body", name=f"building_{i}", pos=f"{x} {y} 0")
        add(
            body,
            "geom",
            name=f"building_{i}",
            type="box",
            pos=f"0 0 {height / 2}",
            size=f"{hx} {hy} {height / 2}",
            material="building",
        )
        add(
            body,
            "geom",
            type="box",
            pos=f"0 0 {height + 0.02}",
            size=f"{hx + 0.04} {hy + 0.04} .025",
            material="trim",
            contype="0",
            conaffinity="0",
        )
    add(
        world,
        "geom",
        name="ramp",
        type="mesh",
        mesh="ramp",
        pos="7 6.5 .001",
        rgba=".74 .57 .29 1",
    )
    for x, y, sx, sy in (
        (0, -7.7, 9.8, 0.06),
        (0, 7.7, 9.8, 0.06),
        (-9.8, 0, 0.06, 7.7),
        (9.8, 0, 0.06, 7.7),
    ):
        add(
            world,
            "geom",
            type="box",
            pos=f"{x} {y} .12",
            size=f"{sx} {sy} .12",
            material="trim",
        )
    for i, (x, y) in enumerate(MARKERS):
        body = add(world, "body", name=f"marker_{i}", pos=f"{x} {y} 0")
        add(
            body,
            "geom",
            type="cylinder",
            size=".16 .008",
            pos="0 0 .01",
            rgba=".98 .75 .18 1",
            contype="0",
            conaffinity="0",
        )
        add(
            body,
            "geom",
            type="box",
            size=".035 .035 .14",
            pos="0 0 .14",
            rgba=".98 .75 .18 1",
            contype="0",
            conaffinity="0",
        )
        add(body, "site", name=f"marker_{i}", pos="0 0 .29", size=".015", group="4")
    actuators = add(root, "actuator")
    for i in range(6):
        rover(world, actuators, i)
    camera(world, "overview", (13, -18, 18), (0, 0, 0), 48)
    camera(world, "overhead", (0, -0.001, 24), (0, 0, 0), 48)
    camera(world, "detail", (-4.8, -8.5, 2.5), (-7.1, -4.4, 0.2), 48)
    path = ROOT / "build/rover_scene.xml"
    path.parent.mkdir(exist_ok=True)
    ET.indent(root, space="  ")
    ET.ElementTree(root).write(path, encoding="unicode")
    model = mujoco.MjModel.from_xml_path(str(path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data
