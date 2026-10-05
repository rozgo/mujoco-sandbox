"""Static v1 surgical workcell, expressed entirely in SI units."""

from pathlib import Path
import math
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
SCENE = ROOT / "build/neural_insertion/static_v1.xml"
FIELD_Y = -0.055
SURFACE_Z = 0.112
CAMERAS = ("overview", "mechanism", "surgical_field", "tool_clearance", "microscope", "thread_fixture")
JOINTS = {
    # travel [m], moving-body mass [kg], damping [N s/m], kp [N/m], kv, force [N]
    "stage_x": ((-0.175, 0.175), 2.4, 12.0, 12000, 160, 100),
    "stage_y": ((-0.075, 0.075), 1.3, 10.0, 10000, 120, 80),
    "stage_z": ((-0.008, 0.050), 0.65, 6.0, 8000, 85, 40),
    "insertion": ((0.0, 0.018), 0.025, 0.15, 1200, 4, 2),
    "retainer": ((0.0, 0.006), 0.004, 0.03, 300, 0.7, 0.3),
}


def vec(values):
    return " ".join(f"{float(v):.10g}" for v in values)


def add(parent, tag, **attrs):
    return ET.SubElement(parent, tag, {k: str(v) for k, v in attrs.items()})


def geom(parent, name, kind, size, pos=(0, 0, 0), material="shell", **kw):
    return add(parent, "geom", name=name, type=kind, size=vec(size), pos=vec(pos), material=material, **kw)


def line(parent, name, points, radius, material, **kw):
    for i, (a, b) in enumerate(zip(points, points[1:])):
        add(parent, "geom", name=f"{name}_{i}", type="capsule", fromto=vec([*a, *b]),
            size=radius, material=material, **kw)


def camera(parent, name, eye, target, fovy=42):
    eye, target = np.array(eye, dtype=float), np.array(target, dtype=float)
    z = eye - target
    z /= np.linalg.norm(z)
    up = np.array([0, 0, 1]) if abs(z[2]) < 0.999 else np.array([0, 1, 0])
    x = np.cross(up, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return add(parent, "camera", name=name, pos=vec(eye), xyaxes=vec([*x, *y]), fovy=fovy)


def ring(parent, name, center, radius, thickness, material, **kw):
    points = [(center[0] + radius * math.cos(t), center[1] + radius * math.sin(t), center[2])
              for t in np.linspace(0, 2 * math.pi, 49)]
    line(parent, name, points, thickness, material, **kw)


def surface_z(x, y):
    return SURFACE_Z - 0.003 * ((x / 0.046) ** 2 + ((y - FIELD_Y) / 0.036) ** 2)


def moving_body(parent, name, pos, axis, inertia):
    bounds, mass, damping, *_ = JOINTS[name]
    body = add(parent, "body", name=name, pos=vec(pos))
    add(body, "joint", name=name, type="slide", axis=vec(axis), range=vec(bounds),
        damping=damping, frictionloss="0.005" if name in ("insertion", "retainer") else "0.2")
    add(body, "inertial", mass=mass, pos="0 0 0", diaginertia=vec(inertia))
    return body


def build_scene(path=SCENE):
    root = ET.Element("mujoco", model="surgical_insertion_static_v1")
    add(root, "compiler", angle="radian", autolimits="true")
    add(root, "option", timestep="0.001", integrator="implicitfast", gravity="0 0 -9.81",
        cone="elliptic", iterations="80")
    add(root, "statistic", center="0 -0.01 0.24", extent="0.8")
    visual = add(root, "visual")
    add(visual, "global", offwidth="1800", offheight="1200")
    add(visual, "quality", shadowsize="4096", offsamples="4")
    add(visual, "headlight", ambient="0.3 0.3 0.3", diffuse="0.45 0.45 0.45", specular="0.15 0.15 0.15")
    add(visual, "map", znear="0.00002", zfar="8", shadowclip="1.5")
    default = add(root, "default")
    add(default, "geom", friction="0.5 0.005 0.0001", solref="0.004 1", solimp="0.95 0.99 0.0001")
    markings = add(default, "default", **{"class": "marking"})
    add(markings, "geom", contype="0", conaffinity="0", mass="0", group="2")
    asset = add(root, "asset")
    add(asset, "texture", name="sky", type="skybox", builtin="gradient", rgb1="0.12 0.17 0.21",
        rgb2="0.025 0.040 0.055", width="512", height="3072")
    colors = {
        "shell": "0.82 0.88 0.88 1", "graphite": "0.065 0.10 0.13 1",
        "rail": "0.32 0.40 0.44 1", "steel": "0.69 0.76 0.79 1",
        "teal": "0.025 0.62 0.58 1", "blueglass": "0.035 0.19 0.26 1",
        "rubber": "0.023 0.030 0.035 1", "tissue": "0.70 0.37 0.37 1",
        "vessel": "0.42 0.035 0.05 1", "vessel_tip": "0.64 0.06 0.065 1",
        "target": "0.27 0.95 0.80 1", "thread": "1.0 0.66 0.18 1",
        "floor": "0.065 0.09 0.12 1", "tray": "0.21 0.31 0.35 1",
    }
    for name, color in colors.items():
        add(asset, "material", name=name, rgba=color, specular="0.45" if name in ("steel", "rail") else "0.18",
            shininess="0.45" if name in ("steel", "rail") else "0.2")

    # Convex phantom with a shallow domed top; the same mesh is visible and colliding.
    vertices, faces = [], []
    for level in ("bottom", "edge", "mid"):
        for i in range(64):
            a = 2 * math.pi * i / 64
            r = 0.5 if level == "mid" else 1.0
            x, y = 0.046 * r * math.cos(a), 0.036 * r * math.sin(a)
            z = 0.087 if level == "bottom" else surface_z(x, y + FIELD_Y)
            vertices.append((x, y, z))
    vertices.extend([(0, 0, SURFACE_Z), (0, 0, 0.087)])
    for i in range(64):
        j = (i + 1) % 64
        faces.extend([(i, j, 64 + j), (i, 64 + j, 64 + i),
                      (64 + i, 64 + j, 128 + j), (64 + i, 128 + j, 128 + i),
                      (128 + i, 128 + j, 192), (j, i, 193)])
    add(asset, "mesh", name="phantom", vertex=vec(np.array(vertices).ravel()),
        face=" ".join(str(v) for f in faces for v in f))
    world = add(root, "worldbody")
    geom(world, "floor", "plane", (2, 2, .05), (0, 0, -.024), "floor")
    add(world, "light", pos="-0.55 -0.7 1.25", dir="0.25 0.35 -1", diffuse="0.65 0.63 0.59", castshadow="false")
    add(world, "light", pos="0.7 0.25 0.9", dir="-0.5 -0.3 -1", diffuse="0.4 0.5 0.6", castshadow="false")
    base = add(world, "body", name="base")
    geom(base, "plinth", "box", (.34, .235, .018), (0, 0, .022), "graphite")
    geom(base, "deck", "box", (.329, .224, .008), (0, 0, .048), "shell")
    geom(base, "front_trim", "box", (.295, .001, .0025), (0, -.2355, .027), "teal")
    for x in (-.29, .29):
        for y in (-.19, .19):
            geom(base, f"foot_{x}_{y}", "cylinder", (.026, .014), (x, y, -.01), "rubber")
    for x in (-.267, .267):
        geom(base, f"tower_{x}", "box", (.030, .037, .204), (x, .095, .260), "shell")
        geom(base, f"tower_inset_{x}", "box", (.018, .003, .17), (x, .055, .26), "graphite")
        geom(base, f"tower_light_{x}", "box", (.002, .001, .042), (x, .051, .355), "teal")
    geom(base, "beam", "box", (.302, .043, .037), (0, .095, .465), "shell")
    geom(base, "beam_insert", "box", (.247, .004, .027), (0, .048, .465), "graphite")
    for z in (.450, .480):
        geom(base, f"x_rail_{z}", "box", (.239, .007, .004), (0, .041, z), "steel")
    for x in np.linspace(-.22, .22, 9):
        geom(base, f"rail_bolt_{x}", "cylinder", (.003, .0015), (x, .032, .465), "steel", euler=f"{math.pi / 2} 0 0")

    xbody = moving_body(base, "stage_x", (0, .025, .465), (1, 0, 0), (.006, .008, .008))
    geom(xbody, "x_carriage", "box", (.043, .010, .046), (0, -.012, 0), "graphite")
    geom(xbody, "x_face", "box", (.038, .003, .038), (0, -.026, 0), "shell")
    geom(xbody, "y_bearing", "box", (.024, .030, .005), (0, -.022, -.046), "steel")
    ybody = moving_body(xbody, "stage_y", (0, -.060, -.070), (0, 1, 0), (.004, .0018, .003))
    geom(ybody, "y_boom", "box", (.026, .100, .017), (0, .025, 0), "graphite")
    for x in (-.017, .017):
        geom(ybody, f"y_rail_{x}", "box", (.003, .090, .003), (x, .025, .019), "steel")
    geom(ybody, "z_backbone", "box", (.034, .014, .091), (0, .025, -.075), "shell")
    for x in (-.021, .021):
        geom(ybody, f"z_rail_{x}", "box", (.004, .003, .075), (x, .008, -.075), "steel")
    zbody = moving_body(ybody, "stage_z", (0, -.020, -.140), (0, 0, 1), (.00065, .00065, .00045))
    geom(zbody, "tool_head", "box", (.029, .026, .037), (0, 0, .026), "shell")
    geom(zbody, "head_face", "box", (.021, .002, .021), (0, -.0285, .028), "graphite")
    geom(zbody, "head_led", "box", (.013, .0008, .001), (0, -.0308, .031), "teal")
    geom(zbody, "tool_socket", "cylinder", (.011, .016), (0, 0, -.024), "graphite")
    # Optical head sits beside the tool with an unobstructed oblique field of view.
    geom(zbody, "scope_bracket", "box", (.021, .012, .007), (.044, .013, .015), "graphite")
    geom(zbody, "scope_barrel", "cylinder", (.013, .032), (.048, .013, -.017), "graphite")
    geom(zbody, "scope_ring", "cylinder", (.014, .004), (.048, .013, -.049), "steel")
    geom(zbody, "scope_lens", "cylinder", (.010, .0008), (.048, .013, -.0535), "blueglass")
    camera(zbody, "microscope", (.048, .013, -.0545), (0, 0, -.143), 23)
    needle = moving_body(zbody, "insertion", (0, 0, -.047), (0, 0, -1), (2e-6, 2e-6, 3e-7))
    geom(needle, "needle_drive", "cylinder", (.006, .010), material="steel")
    geom(needle, "needle_shank", "cylinder", (.0016, .022), (0, 0, -.032), "steel")
    geom(needle, "needle_taper", "capsule", (.00035, .006), (0, 0, -.059), "steel")
    geom(needle, "needle", "capsule", (.000075, .009), (0, 0, -.074), "steel")
    add(needle, "site", name="needle_tip", pos="0 0 -0.083075", size="0.0001", group="4")
    retainer = moving_body(zbody, "retainer", (.002, 0, -.044), (0, 0, -1), (3e-7, 3e-7, 1e-8))
    geom(retainer, "retainer_drive", "box", (.002, .003, .007), (.010, 0, 0), "teal")
    line(retainer, "retainer_arm", [(0.01, 0, -.007), (.010, 0, -.049), (.0008, 0, -.075)], .00065, "steel")
    for side in (-1, 1):
        line(retainer, f"retainer_fork_{side}", [(.0008, 0, -.075), (0, side * .0004, -.079), (0, side * .0004, -.082)], .00010, "steel")

    # Supported phantom and the surrounding sterile work platform.
    stage = add(world, "body", name="specimen_support", pos=f"0 {FIELD_Y} 0")
    geom(stage, "specimen_base", "box", (.064, .054, .009), (0, 0, .065), "graphite")
    geom(stage, "specimen_tray", "box", (.057, .047, .006), (0, 0, .080), "steel")
    geom(stage, "tissue_phantom", "mesh", (), material="tissue", mesh="phantom")
    for x in (-.052, .052):
        geom(stage, f"clamp_{x}", "box", (.004, .025, .005), (x, 0, .091), "tray")
        for y in (-.018, .018):
            geom(stage, f"clamp_screw_{x}_{y}", "cylinder", (.0025, .0015), (x, y, .097), "steel")
    # Surface vessels and target rings are markings on the phantom, not extra rigid obstacles.
    surface_marks = {"class": "marking"}
    paths = [
        [(-.039, -.004), (-.03, -.002), (-.022, .003), (-.017, .01), (-.009, .014), (.0, .019), (.012, .022), (.027, .02)],
        [(-.021, .004), (-.021, -.006), (-.015, -.015), (-.013, -.026)],
        [(-.009, .014), (-.005, .006), (.004, -.002), (.018, -.006), (.039, -.009)],
        [(.004, -.002), (.006, -.013), (.013, -.02), (.021, -.023)],
        [(.018, -.006), (.026, .003), (.036, .007)],
        [(-.03, -.002), (-.03, -.012), (-.032, -.02)],
        [(.0, .019), (-.004, .026), (-.003, .033)],
    ]
    for p, points in enumerate(paths):
        smooth = []
        for a, b in zip(points, points[1:]):
            for t in np.linspace(0, 1, 8, endpoint=False):
                xx, yy = np.array(a) * (1 - t) + np.array(b) * t
                smooth.append((xx, yy, surface_z(xx, yy + FIELD_Y) + .00010))
        xx, yy = points[-1]
        smooth.append((xx, yy, surface_z(xx, yy + FIELD_Y) + .00010))
        line(stage, f"vessel_{p}", smooth, .00050 if p == 0 else .00028, "vessel" if p == 0 else "vessel_tip", **surface_marks)
    for i, (x, y) in enumerate(((0, 0), (-.011, -.005), (.019, .009), (.024, -.017), (-.029, .012), (.009, .01))):
        z = surface_z(x, y + FIELD_Y) + .00016
        ring(stage, f"target_{i}", (x, y, z), .00125, .00007, "target", **surface_marks)
        add(stage, "site", name=f"target_{i}", pos=vec((x, y, z)), size="0.00015", group="4")
    # Cassette holds presentation samples. Flexible pickup mechanics come next.
    fixture = add(world, "body", name="thread_cassette", pos=f"-0.110 {FIELD_Y} 0")
    geom(fixture, "cassette_pedestal", "box", (.027, .034, .012), (0, 0, .068), "graphite")
    geom(fixture, "cassette_base", "box", (.027, .034, .004), (0, 0, .084), "shell")
    geom(fixture, "cassette_insert", "box", (.022, .029, .003), (0, 0, .091), "blueglass")
    for y in (-.023, .023):
        geom(fixture, f"presentation_rail_{y}", "box", (.021, .002, .007), (0, y, .101), "steel")
    for i, x in enumerate((-.015, -.005, .005, .015)):
        points = [(x + .0015 * math.sin(t * math.pi * 2), -.022 + .044 * t, .10825) for t in np.linspace(0, 1, 40)]
        line(fixture, f"presented_thread_{i}", points, .000020, "thread", **surface_marks)
        ring(fixture, f"pickup_loop_{i}", (x, -.022, .10825), .0004, .000020, "thread", **surface_marks)
        geom(fixture, f"thread_tab_{i}", "box", (.002, .0025, .0002), (x, .026, .1083), "thread", **surface_marks)
    # Grid and perimeter marks make the scale of the deck legible.
    for x in np.linspace(-.2, .2, 9):
        geom(base, f"deck_tick_{x}", "box", (.0005, .006, .0001), (x, -.178, .0561), "rail", **surface_marks)
    camera(world, "overview", (.70, -.96, .70), (-.01, -.005, .245), 37)
    camera(world, "mechanism", (.32, -.44, .40), (.0, -.018, .335), 38)
    camera(world, "surgical_field", (.135, -.29, .25), (-.037, FIELD_Y, .106), 39)
    camera(world, "tool_clearance", (.036, -.092, .137), (0, FIELD_Y, .119), 37)
    camera(world, "thread_fixture", (-.071, -.107, .151), (-.11, FIELD_Y, .106), 34)
    actuators = add(root, "actuator")
    for name, (bounds, _, _, kp, kv, force) in JOINTS.items():
        add(actuators, "position", name=name, joint=name, kp=kp, kv=kv,
            ctrlrange=vec(bounds), forcerange=vec((-force, force)))
    # MuJoCo's standard adjacent-body filtering applies to the joint guides.
    # Tool, fixture, specimen and nonadjacent robot collisions remain enabled.
    keyframes = add(root, "keyframe")
    add(keyframes, "key", name="inspection", qpos="0 0 0 0 0", ctrl="0 0 0 0 0")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(root, space="  ")
    ET.ElementTree(root).write(path, encoding="unicode")
    return path


def load_scene(path=SCENE):
    model = mujoco.MjModel.from_xml_path(str(build_scene(path)))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("inspection").id)
    mujoco.mj_forward(model, data)
    return model, data
