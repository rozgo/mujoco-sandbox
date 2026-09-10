"""A small delivery quadrotor, physical sling load, and open test course."""

import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

from sixlegs.scene import ROOT, add, camera, vec

DT = 0.002
DRONE_MASS = 2.0
LOAD_MASS = 0.2
CABLE = 0.75
START = np.array([-2.7, 0.0, 1.20])
FINISH = np.array([2.7, 0.0, 0.35])
ROTOR_XY = np.array([[0.24, 0.24], [-0.24, 0.24], [-0.24, -0.24], [0.24, -0.24]])
YAW_SIGNS = np.array([1.0, -1.0, 1.0, -1.0])


def build_xml():
    root = ET.Element("mujoco", model="neural_wind_delivery")
    add(root, "compiler", angle="radian", autolimits="true")
    add(
        root,
        "option",
        timestep=str(DT),
        integrator="implicitfast",
        iterations="80",
        gravity="0 0 -9.81",
    )
    visual = add(root, "visual")
    add(visual, "global", offwidth="1920", offheight="1080")
    add(visual, "quality", shadowsize="2048", offsamples="4")
    add(visual, "headlight", ambient=".4 .4 .4", diffuse=".65 .65 .65")
    default = add(root, "default")
    add(default, "geom", friction=".8 .01 .001", solref=".008 1", solimp=".95 .99 .001")
    assets = add(root, "asset")
    add(
        assets,
        "texture",
        name="sky",
        type="skybox",
        builtin="gradient",
        rgb1=".16 .24 .30",
        rgb2=".035 .065 .10",
        width="512",
        height="3072",
    )
    add(
        assets,
        "texture",
        name="grid",
        type="2d",
        builtin="checker",
        rgb1=".13 .18 .21",
        rgb2=".16 .22 .25",
        width="512",
        height="512",
    )
    add(
        assets,
        "material",
        name="floor",
        texture="grid",
        texrepeat="12 10",
        texuniform="true",
        reflectance=".08",
    )
    for name, color in {
        "shell": ".08 .26 .30 1",
        "metal": ".60 .72 .76 1",
        "orange": "1 .55 .16 1",
        "cyan": ".1 .86 .80 1",
        "white": ".84 .92 .94 1",
        "rubber": ".04 .06 .07 1",
    }.items():
        add(assets, "material", name=name, rgba=color, specular=".5", shininess=".4")
    world = add(root, "worldbody")
    add(world, "geom", name="floor", type="plane", size="6 5 .1", material="floor")
    for x, label in ((-2.7, "launch"), (2.7, "delivery")):
        add(
            world,
            "geom",
            name=label + "_platform",
            type="box",
            pos=f"{x} 0 .15",
            size=".52 .52 .15",
            material="metal",
        )
        add(
            world,
            "geom",
            type="box",
            pos=f"{x} 0 .306",
            size=".47 .47 .006",
            material="shell",
            contype="0",
            conaffinity="0",
        )
        for y in (-0.44, 0.44):
            add(
                world,
                "geom",
                type="box",
                size=".44 .012 .008",
                pos=f"{x} {y} .316",
                material="cyan",
                contype="0",
                conaffinity="0",
            )
        add(
            world,
            "geom",
            type="cylinder",
            size=".20 .007",
            pos=f"{x} 0 .32",
            material="orange",
            contype="0",
            conaffinity="0",
        )
    for x in (-0.95, 1.0):
        for y in (-0.85, 0.85):
            add(
                world,
                "geom",
                name=f"gate_{x}_{y}",
                type="capsule",
                fromto=f"{x} {y} .05 {x} {y} 2.5",
                size=".035",
                material="metal",
            )
            add(
                world,
                "geom",
                type="cylinder",
                pos=f"{x} {y} .05",
                size=".15 .05",
                material="orange",
            )
        add(
            world,
            "geom",
            name=f"gate_top_{x}",
            type="capsule",
            fromto=f"{x} -.85 2.5 {x} .85 2.5",
            size=".035",
            material="cyan",
        )
    # Flow probes are visual context, not flow-generating obstacle boundaries.
    for x in np.linspace(-3.5, 3.5, 8):
        add(
            world,
            "geom",
            type="box",
            size=".10 .10 .025",
            pos=f"{x} -2.4 .025",
            material="shell",
        )
        add(
            world,
            "geom",
            type="capsule",
            fromto=f"{x} -2.4 .05 {x} -2.4 .5",
            size=".012",
            material="metal",
        )
    drone = add(world, "body", name="drone", pos=vec(START))
    add(drone, "freejoint", name="drone_free")
    add(
        drone,
        "inertial",
        pos="0 0 0",
        mass=str(DRONE_MASS),
        diaginertia=".035 .035 .055",
    )
    add(
        drone,
        "geom",
        name="drone_shell",
        type="box",
        size=".145 .105 .055",
        material="shell",
        mass="0",
    )
    add(
        drone,
        "geom",
        type="box",
        pos="0 0 .073",
        size=".085 .065 .025",
        material="metal",
        mass="0",
    )
    for i, (x, y) in enumerate(ROTOR_XY):
        add(
            drone,
            "geom",
            type="capsule",
            fromto=f"0 0 0 {x} {y} 0",
            size=".018",
            material="metal",
            mass="0",
        )
        add(
            drone,
            "geom",
            name=f"motor_{i}",
            type="cylinder",
            size=".04 .025",
            pos=f"{x} {y} .018",
            material="orange",
            mass="0",
        )
        add(
            drone,
            "geom",
            type="cylinder",
            size=".115 .003",
            pos=f"{x} {y} .047",
            rgba=".65 .82 .86 .22",
            contype="0",
            conaffinity="0",
            mass="0",
        )
        add(
            drone,
            "geom",
            type="capsule",
            fromto=f"{x - 0.10} {y} .051 {x + 0.10} {y} .051",
            size=".008",
            material="rubber",
            mass="0",
            contype="0",
            conaffinity="0",
        )
        add(
            drone, "site", name=f"rotor_{i}", pos=f"{x} {y} .02", size=".005", group="4"
        )
    for y in (-0.15, 0.15):
        add(
            drone,
            "geom",
            type="capsule",
            fromto=f"-.12 {y} -.09 .12 {y} -.09",
            size=".012",
            material="rubber",
            mass="0",
        )
    add(drone, "site", name="hook", pos="0 0 -.05", size=".012", material="orange")
    camera(drone, "front", (0.16, 0, 0.015), (3, 0, -0.20), 70)
    camera(drone, "loadcam", (0.06, 0, -0.08), (0.01, 0.001, -1), 75)
    payload = add(world, "body", name="payload", pos=vec(START + [0, 0, -0.05 - CABLE]))
    add(payload, "freejoint", name="payload_free")
    add(
        payload,
        "geom",
        name="package",
        type="box",
        size=".13 .13 .10",
        mass=str(LOAD_MASS),
        material="orange",
    )
    for y in (-0.08, 0.08):
        add(
            payload,
            "geom",
            type="box",
            size=".132 .012 .102",
            pos=f"0 {y} 0",
            mass="0",
            material="white",
            contype="0",
            conaffinity="0",
        )
    add(payload, "site", name="load_hook", pos="0 0 .10", size=".006", group="4")
    tendon = add(root, "tendon")
    spatial = add(
        tendon,
        "spatial",
        name="sling",
        width=".003",
        rgba=".8 .9 .95 1",
        limited="true",
        range=f"0 {CABLE - 0.10}",
        solreflimit=".006 1",
        solimplimit=".99 .999 .0001",
    )
    add(spatial, "site", site="hook")
    add(spatial, "site", site="load_hook")
    actuators = add(root, "actuator")
    for i in range(4):
        add(
            actuators,
            "general",
            name=f"thrust_{i}",
            site=f"rotor_{i}",
            gear=f"0 0 1 0 0 {YAW_SIGNS[i] * 0.018}",
            ctrllimited="true",
            ctrlrange="0 10",
            forcelimited="true",
            forcerange="0 10",
        )
    add(world, "light", pos="-3 -4 7", dir=".3 .3 -1", diffuse=".9 .92 1")
    add(
        world,
        "light",
        pos="4 4 6",
        dir="-.3 -.3 -1",
        diffuse=".65 .8 1",
        castshadow="false",
    )
    camera(world, "overview", (7, -9, 6), (0.1, 0, 0.9), 46)
    camera(world, "side", (0.2, -8, 2.6), (0.2, 0, 1.2), 48)
    camera(world, "detail", (-1.2, -1.8, 2.1), START + [0, 0, -0.25], 46)
    camera(world, "overhead", (0, -0.001, 9), (0, 0, 0), 52)
    ET.indent(root, space="  ")
    return ET.tostring(root, encoding="unicode")


def load_scene():
    xml = build_xml()
    p = ROOT / "build/wind_scene.xml"
    p.parent.mkdir(exist_ok=True)
    p.write_text(xml)
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    return m, d


def preview(output=ROOT / "previews/wind"):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    m, d = load_scene()
    opt = mujoco.MjvOption()
    opt.sitegroup[4] = 0
    with mujoco.Renderer(m, 900, 1600) as r:
        for cam in ("overview", "detail", "side", "front", "loadcam"):
            r.update_scene(d, camera=cam, scene_option=opt)
            Image.fromarray(r.render()).save(output / f"{cam}.png")
    return {
        "drone_mass": m.body("drone").mass,
        "load_mass": m.body("payload").mass,
        "actuators": m.nu,
        "contacts": d.ncon,
    }
