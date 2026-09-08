"""Deterministic MJCF assembly. Vendor files stay byte-for-byte upstream."""
from copy import deepcopy
from pathlib import Path
import math
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
VENDOR = ROOT / "assets/menagerie"
SCENE = ROOT / "build/scene.xml"
START = (-2.15, -1.35, 0.78)
ARM_HOME = (0, 0.26179939, math.pi, -2.26892803, 0, 0.95993109, math.pi / 2)


def add(parent, tag, **attrs):
    return ET.SubElement(parent, tag, {k: str(v) for k, v in attrs.items()})


def vec(values):
    return " ".join(f"{v:.9g}" for v in values)


def camera(parent, name, eye, target, fovy=48):
    eye, target = np.array(eye), np.array(target)
    z = eye - target
    z = z / np.linalg.norm(z)
    x = np.cross([0, 0, 1], z)
    x = x / np.linalg.norm(x)
    y = np.cross(z, x)
    return add(parent, "camera", name=name, pos=vec(eye), xyaxes=vec([*x, *y]), fovy=fovy)


def namespaced_model(folder, filename, prefix):
    tree = ET.parse(VENDOR / folder / filename).getroot()
    # Implicit mesh names must be materialized before changing filenames.
    for mesh in tree.findall("asset/mesh"):
        mesh.set("name", mesh.get("name", Path(mesh.get("file")).stem))
        mesh.set("file", str((VENDOR / folder / "assets" / mesh.get("file")).resolve()))
    references = {"name", "class", "childclass", "mesh", "material", "joint", "joint1", "joint2",
                  "body1", "body2", "tendon", "site", "site1", "site2"}
    for element in tree.iter():
        for key in references & element.attrib.keys():
            element.set(key, prefix + element.get(key))
    return tree


def attach_arm(root, chassis, side, y):
    prefix = side + "_arm_"
    arm = namespaced_model("kinova_gen3", "gen3.xml", prefix)
    grip = namespaced_model("robotiq_2f85", "2f85.xml", side + "_grip_")
    # Scope all upstream defaults to this instance (visual/collision names overlap).
    defaults = deepcopy(arm.find("default"))
    defaults.set("class", prefix + "defaults")
    root.find("default").append(defaults)
    for default in grip.find("default"):
        root.find("default").append(deepcopy(default))
    for source in (arm, grip):
        for section in ("asset", "actuator", "contact", "tendon", "equality"):
            if source.find(section) is not None:
                for element in source.find(section):
                    root.find(section).append(deepcopy(element))
    mount = add(chassis, "body", name=side + "_mount", pos=f"0.20 {y} 0.155")
    add(mount, "geom", name=side + "_pedestal", type="cylinder", size="0.09 0.04", pos="0 0 -0.035", mass="0.45", material="metal")
    base = deepcopy(arm.find("worldbody/body"))
    base.set("childclass", prefix + "defaults")
    mount.append(base)
    # Finite software travel limits on continuous joints avoid unlimited cable winding.
    limits = (math.pi, 2.24, math.pi, 2.57, math.pi, 2.09, math.pi)
    for i, limit in enumerate(limits, 1):
        j = base.find(f".//joint[@name='{prefix}joint_{i}']")
        j.set("range", vec([-limit, limit]))
        j.set("damping", "1")
        j.set("armature", "0.02")
        actuator = root.find(f"actuator/position[@name='{prefix}joint_{i}']")
        actuator.set("ctrlrange", vec([-limit, limit]))
    wrist = base.find(f".//body[@name='{prefix}bracelet_link']")
    wrist.find("camera").set("name", side + "_wrist")
    # Kinova README specifies direct mounting: its interface includes the coupling.
    gripper_base = deepcopy(grip.find("worldbody/body/body"))
    gripper_base.set("pos", "0 0 -0.06149039")
    gripper_base.set("quat", "0 -1 1 0")
    gripper_base.set("childclass", side + "_grip_2f85")
    wrist.append(gripper_base)
    add(root.find("contact"), "exclude", body1=prefix + "bracelet_link", body2=side + "_grip_base")


def add_leg(root, chassis, name, x, y, angle):
    hip = add(chassis, "body", name=name + "_coxa", pos=f"{x} {y} -0.04", euler=f"0 0 {angle}")
    add(hip, "joint", name=name + "_yaw", axis="0 0 1", range="-0.65 0.65", damping="4", armature="0.035")
    add(hip, "geom", name=name + "_coxa_geom", type="capsule", fromto="0 0 0 0.22 0 0", size="0.058", mass="0.75", material="metal")
    femur = add(hip, "body", name=name + "_femur", pos="0.22 0 0")
    add(femur, "joint", name=name + "_hip", axis="0 1 0", range="-0.75 0.7", damping="6", armature="0.05")
    add(femur, "geom", name=name + "_femur_geom", type="capsule", fromto="0 0 0 0.36 0 0.08", size="0.048", mass="1.15", material="shell")
    add(femur, "geom", type="sphere", size="0.071", mass="0.12", material="accent", contype="0", conaffinity="0")
    tibia = add(femur, "body", name=name + "_tibia", pos="0.36 0 0.08")
    add(tibia, "joint", name=name + "_knee", axis="0 1 0", range="-0.65 0.85", damping="5", armature="0.035")
    add(tibia, "geom", name=name + "_tibia_geom", type="capsule", fromto="0 0 0 0.24 0 -0.78", size="0.036", mass="0.85", material="metal")
    add(tibia, "geom", name=name + "_foot", type="sphere", pos="0.24 0 -0.78", size="0.04", mass="0.15", material="rubber", friction="1.2 0.02 0.002", condim="4")
    add(tibia, "site", name=name + "_toe", pos="0.24 0 -0.78", size="0.008", group="4")
    for joint, kp, kv, torque in (("yaw", 400, 35, 65), ("hip", 650, 45, 120), ("knee", 600, 40, 100)):
        ranges = {"yaw": "-0.65 0.65", "hip": "-0.75 0.7", "knee": "-0.65 0.85"}
        add(root.find("actuator"), "position", name=name + "_" + joint, joint=name + "_" + joint,
            kp=kp, kv=kv, ctrlrange=ranges[joint], forcerange=f"-{torque} {torque}")


def table(world, name, x, y):
    body = add(world, "body", name=name, pos=f"{x} {y} 0")
    add(body, "geom", name=name + "_top", type="box", pos="0 0 0.80", size="0.65 0.70 0.04", material="wood")
    for dx in (-0.53, 0.53):
        for dy in (-0.58, 0.58):
            add(body, "geom", type="box", pos=f"{dx} {dy} 0.38", size="0.045 0.045 0.38", material="metal")
    # Colored edge strip distinguishes source and destination.
    add(body, "geom", type="box", pos="-0.653 0 0.80", size="0.004 0.70 0.032", material="accent" if name == "source" else "destination", contype="0", conaffinity="0")


def build_scene(path=SCENE):
    root = ET.Element("mujoco", model="sixlegs_dual_arm")
    add(root, "compiler", angle="radian", autolimits="true")
    add(root, "option", timestep="0.002", integrator="implicitfast", cone="elliptic", impratio="10", iterations="100")
    visual = add(root, "visual")
    add(visual, "global", offwidth="1920", offheight="1440")
    add(visual, "quality", shadowsize="4096", offsamples="4")
    add(visual, "headlight", ambient="0.3 0.3 0.3", diffuse="0.65 0.65 0.65", specular="0.2 0.2 0.2")
    add(visual, "map", znear="0.01", zfar="40")
    default = add(root, "default")
    add(default, "geom", friction="0.8 0.01 0.001", solref="0.01 1", solimp="0.95 0.99 0.001")
    asset = add(root, "asset")
    add(asset, "texture", name="sky", type="skybox", builtin="gradient", rgb1="0.15 0.19 0.24", rgb2="0.035 0.045 0.06", width="512", height="3072")
    add(asset, "texture", name="floor_tex", type="2d", builtin="checker", rgb1="0.20 0.24 0.28", rgb2="0.23 0.27 0.31", width="512", height="512")
    add(asset, "material", name="floor", texture="floor_tex", texrepeat="2 2", texuniform="true", reflectance="0.08")
    for name, color in {"shell":"0.73 0.80 0.84 1", "metal":"0.16 0.20 0.24 1", "accent":"0.04 0.70 0.72 1", "rubber":"0.045 0.06 0.07 1", "wood":"0.58 0.40 0.25 1", "destination":"0.48 0.78 0.35 1", "mug":"0.95 0.48 0.12 1", "block":"0.23 0.48 0.94 1"}.items():
        add(asset, "material", name=name, rgba=color, specular="0.35", shininess="0.35")
    for section in ("actuator", "contact", "tendon", "equality"):
        add(root, section)
    world = add(root, "worldbody")
    add(world, "geom", name="floor", type="plane", size="8 8 0.1", material="floor")
    add(world, "light", pos="-3 -4 7", dir="0.3 0.4 -1", diffuse="0.9 0.85 0.78", castshadow="true")
    add(world, "light", pos="3 4 6", dir="-0.3 -0.2 -1", diffuse="0.55 0.68 0.85", castshadow="false")
    camera(world, "third_person", (4.4, -5.8, 5.3), (-0.6, 0.8, 0.6), 47)
    camera(world, "robot_detail", (0.65, -4.7, 2.9), (-2.05, -1.3, 0.9), 47)
    camera(world, "overhead", (-0.3, 0.6, 10), (-0.3, 0.601, 0), 48)
    robot = add(world, "body", name="chassis", pos=vec(START))
    add(robot, "freejoint", name="floating_base")
    add(robot, "geom", name="chassis_hull", type="box", size="0.57 0.31 0.115", mass="16", material="shell")
    add(robot, "geom", name="belly", type="box", pos="-0.08 0 -0.115", size="0.39 0.25 0.055", mass="4", material="metal")
    add(robot, "geom", name="deck", type="box", pos="0 0 0.12", size="0.51 0.285 0.025", mass="1.2", material="metal")
    for y in (-0.317, 0.317):
        add(robot, "geom", type="box", pos=f"0 {y} 0.025", size="0.4 0.008 0.018", material="accent", mass="0.025", contype="0", conaffinity="0")
    head = add(robot, "body", name="head", pos="0.53 0 0.19", euler="0 0 0.4")
    add(head, "geom", type="box", size="0.055 0.115 0.055", mass="0.35", material="metal")
    for y in (-0.065, 0.065):
        add(head, "geom", type="sphere", pos=f"0.055 {y} 0", size="0.023", mass="0.015", material="accent", contype="0", conaffinity="0")
    camera(head, "head", (0.065, 0, 0), (2, 0, -0.28), 75)
    for side, sign in (("left", 1), ("right", -1)):
        for label, x, angle in (("front", .43, math.pi/4), ("middle", 0, math.pi/2), ("rear", -.43, 3*math.pi/4)):
            add_leg(root, robot, side + "_" + label, x, sign*.32, sign*angle)
        attach_arm(root, robot, side, sign*.235)
    table(world, "source", 0.6, 0)
    table(world, "destination", 0.6, 3.6)
    barrier = add(world, "body", name="barrier", pos="0 1.8 0")
    add(barrier, "geom", name="barrier_panel", type="box", pos="0 0 0.62", size="1.2 0.09 0.62", rgba="0.83 0.57 0.18 1")
    for x in (-1, 1):
        add(barrier, "geom", type="box", pos=f"{x} 0 0.045", size="0.12 0.30 0.045", material="metal")
    # Hollow mug: floor plus segmented wall and a collision-enabled loop handle.
    mug = add(world, "body", name="mug", pos="0.13 -0.23 0.84")
    add(mug, "freejoint", name="mug_free")
    add(mug, "geom", name="mug_bottom", type="cylinder", pos="0 0 0.004", size="0.032 0.004", mass="0.06", material="mug")
    for i in range(20):
        a = 2*math.pi*i/20
        add(mug, "geom", name=f"mug_wall_{i}", type="box", pos=vec([.031*math.cos(a), .031*math.sin(a), .043]), euler=f"0 0 {a}", size="0.003 0.0052 0.035", mass="0.006", material="mug")
    for i in range(12):
        a, b = 2*math.pi*i/12, 2*math.pi*(i+1)/12
        p = [.035 + .023*math.cos(a), 0, .043 + .026*math.sin(a)]
        q = [.035 + .023*math.cos(b), 0, .043 + .026*math.sin(b)]
        add(mug, "geom", name=f"mug_handle_{i}", type="capsule", fromto=vec(p+q), size="0.004", mass="0.0025", material="mug")
    add(mug, "site", name="mug_grasp", pos="0 0 0.045", size="0.005", group="4")
    block = add(world, "body", name="block", pos="0.13 0.22 0.871")
    add(block, "freejoint", name="block_free")
    add(block, "geom", name="block_geom", type="box", size="0.025 0.025 0.03", mass="0.12", material="block")
    add(block, "site", name="block_grasp", size="0.005", group="4")
    for i, (x, y, size, color) in enumerate(((.72,-.42,.045,"0.61 0.27 0.28 1"),(.84,.08,.055,"0.43 0.65 0.37 1"),(.49,.43,.035,"0.75 0.72 0.58 1"),(.92,.49,.04,"0.55 0.41 0.68 1"))):
        body = add(world, "body", name=f"clutter_{i}", pos=f"{x} {y} {0.841+size}")
        add(body, "freejoint", name=f"clutter_{i}_free")
        add(body, "geom", name=f"clutter_{i}_geom", type="box", size=f"{size} {size} {size}", mass="0.15", rgba=color)
    # Non-colliding flush placement markings.
    for y in (3.37, 3.82):
        add(world, "geom", type="box", pos=f"0.13 {y} 0.841", size="0.09 0.09 0.0008", rgba="0.48 0.78 0.35 0.6", contype="0", conaffinity="0")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(root, space="  ")
    ET.ElementTree(root).write(path, encoding="unicode")
    # Save a complete keyframe so the generated MJCF carries its review pose.
    model = mujoco.MjModel.from_xml_path(str(path))
    data = mujoco.MjData(model)
    for side in ("left", "right"):
        for i, value in enumerate(ARM_HOME, 1):
            name = f"{side}_arm_joint_{i}"
            data.joint(name).qpos[0] = value
            data.ctrl[model.actuator(name).id] = value
    add(add(root, "keyframe"), "key", name="preview", qpos=vec(data.qpos), ctrl=vec(data.ctrl))
    ET.indent(root, space="  ")
    ET.ElementTree(root).write(path, encoding="unicode")
    return path


def load_scene():
    model = mujoco.MjModel.from_xml_path(str(build_scene()))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    return model, data
