"""Compile physical Go2 variants; vendor assets are never edited.

Calves use a declared mass-distribution proxy in every variant, including healthy.
No geometry changes or state assignment take place during a live rollout.
"""

import json
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[4]
VENDOR = ROOT / "assets/menagerie/unitree_go2"
LEGS = ("FL", "FR", "RL", "RR")
JOINTS = tuple(
    f"{leg}_{part}_joint" for leg in LEGS for part in ("hip", "thigh", "calf")
)
STAND = np.array([0.1, 0.8, -1.6, -0.1, 0.8, -1.6, 0.1, 0.8, -1.6, -0.1, 0.8, -1.6])
LIMITS = np.tile([23.7, 23.7, 45.43], 4)
DT, CONTROL_DT, ACTION_SCALE = 0.002, 0.02, 0.65


@dataclass(frozen=True)
class BodySpec:
    name: str = "healthy"
    calf: tuple[float, ...] = (1.0, 1.0, 1.0, 1.0)
    absent: tuple[str, ...] = ()
    absent_legs: tuple[str, ...] = ()

    def __post_init__(self):
        if len(self.calf) != 4 or not all(0.2 <= x <= 1 for x in self.calf):
            raise ValueError(
                "Remaining calf fractions must contain four values in [0.2, 1]."
            )
        if not set(self.absent) <= set(LEGS):
            raise ValueError("Unknown absent calf")
        if not set(self.absent_legs) < set(LEGS):
            raise ValueError("Unknown absent leg, or no remaining support limbs")
        if set(self.absent) & set(self.absent_legs):
            raise ValueError("A removal must specify either calf or whole leg")


PRESETS = {
    "healthy": BodySpec(),
    "short_fl": BodySpec("short_fl", (0.7, 1, 1, 1)),
    "short_fr": BodySpec("short_fr", (1, 0.7, 1, 1)),
    "short_rl": BodySpec("short_rl", (1, 1, 0.7, 1)),
    "short_rr": BodySpec("short_rr", (1, 1, 1, 0.7)),
    "missing_fl": BodySpec("missing_fl", absent=("FL",)),
    "mixed": BodySpec("mixed", (0.65, 1, 1, 0.8)),
}


def add(parent, tag, **attrs):
    return ET.SubElement(parent, tag, {k: str(v) for k, v in attrs.items()})


def allowed_support_names(body):
    """A present terminal surface, or the designated distal stump after removal."""
    return tuple(
        f"{leg}_{'stump' if leg in body.absent else 'terminal'}"
        for leg in LEGS
        if leg not in body.absent_legs
    )


def build_model(
    body=None,
    terrain="flat",
    timestep=DT,
    sensing=True,
    contact_profile="firm",
    support_sensing=True,
):
    body = body or BodySpec()
    root = ET.parse(VENDOR / "go2.xml").getroot()
    root.set("model", "adaptive_go2_" + body.name)
    root.find("compiler").set("meshdir", str(VENDOR / "assets"))
    root.find("option").attrib.update(
        timestep=str(timestep),
        integrator="implicitfast",
        solver="Newton",
        iterations="30",
        cone="pyramidal",
        impratio="1",
    )
    for tag in ("keyframe", "actuator", "sensor"):
        element = root.find(tag)
        if element is not None:
            root.remove(element)
    # Native position servos replace the motor shortcut, retaining torque caps.
    for default in root.findall(".//default"):
        for motor in list(default.findall("motor")):
            default.remove(motor)
    collision = root.find(".//default[@class='collision']/geom")
    collision.attrib.update(
        condim="3",
        friction=".8 .005 .0001",
        margin="0",
        solref=".008 1",
        solimp=".9 .95 .001",
    )
    visual = add(root, "visual")
    add(visual, "global", offwidth="1280", offheight="720", fovy="48")
    add(visual, "quality", shadowsize="2048", offsamples="4")
    add(visual, "headlight", ambient=".45 .45 .45", diffuse=".6 .6 .6")
    world = root.find("worldbody")
    base = world.find("body[@name='base']")
    base.set("pos", "0 0 .34")
    base.find("freejoint").set("name", "floating_base")
    add(world, "light", pos="2 -3 5", dir="-.3 .3 -1", diffuse=".8 .8 .8")
    add(
        world,
        "geom",
        name="floor",
        type="plane",
        size="12 4 .1",
        rgba=".16 .20 .24 1",
        friction=".8 .005 .0001",
        condim="3",
    )
    # Course decoration has no collision or mass; all obstacles are physical.
    for x in range(8):
        add(
            world,
            "geom",
            type="box",
            pos=f"{x} 0 .0005",
            size=".012 .9 .0005",
            rgba=".35 .46 .5 1",
            contype="0",
            conaffinity="0",
        )
    standing = terrain.startswith("stand_")
    if not standing and terrain not in ("flat", "steps", "test_steps", "heldout_steps"):
        raise ValueError(terrain)
    if terrain != "flat" and not standing:
        h = 0.04 if terrain == "steps" else 0.06
        obstacles = ((1.6, 0.32, h), (2.8, 0.5, 1.5 * h), (4.1, 0.4, h))
        if terrain == "heldout_steps":
            obstacles = ((1.4, 0.25, 0.035), (2.55, 0.35, 0.05), (3.75, 0.28, 0.045))
        for x, length, height in obstacles:
            add(
                world,
                "geom",
                name=f"step_{x}",
                type="box",
                size=f"{length} 1 {height / 2}",
                pos=f"{x} 0 {height / 2}",
                rgba=".23 .55 .56 1",
                friction=".8 .005 .0001",
                condim="3",
            )
    for i, leg in enumerate(LEGS):
        if leg in body.absent_legs:
            base.remove(root.find(f".//body[@name='{leg}_hip']"))
            continue
        calf = root.find(f".//body[@name='{leg}_calf']")
        thigh = root.find(f".//body[@name='{leg}_thigh']")
        if leg in body.absent:
            thigh.remove(calf)
            add(
                thigh,
                "geom",
                name=f"{leg}_stump",
                type="sphere",
                pos="0 0 -.213",
                size=".015",
                mass=".008",
                rgba=".95 .43 .13 1",
                condim="3",
                friction=".6 .005 .0001",
            )
            add(thigh, "site", name=f"{leg}_tip", pos="0 0 -.213", size=".004")
            continue
        fraction = body.calf[i]
        length = 0.213 * fraction
        for child in list(calf):
            if child.tag in ("geom", "inertial"):
                calf.remove(child)
        color = ".7 .75 .77 1" if fraction == 1 else ".95 .43 .13 1"
        # Fixed 70 g proximal housing plus 171.352 g removable member at full length.
        # Compiler derives the changed body's COM and positive inertia from these geoms.
        add(
            calf,
            "geom",
            name=f"{leg}_housing",
            type="sphere",
            size=".025",
            mass=".07",
            rgba=".18 .21 .24 1",
            condim="3",
            friction=".8 .005 .0001",
        )
        add(
            calf,
            "geom",
            name=f"{leg}_member",
            type="capsule",
            fromto=f"0 0 -.02 0 0 {-length}",
            size=".013",
            mass=str(0.151352 * fraction),
            rgba=color,
            condim="3",
            friction=".6 .005 .0001",
        )
        radius = 0.022 if fraction == 1 else 0.014
        add(
            calf,
            "geom",
            name=f"{leg}_terminal",
            type="sphere",
            pos=f"0 0 {-length}",
            size=str(radius),
            mass=str(0.02 * fraction),
            rgba=color,
            condim="3",
            friction=(".8 .005 .0001" if fraction == 1 else ".6 .005 .0001"),
        )
        add(calf, "site", name=f"{leg}_tip", pos=f"0 0 {-length}", size=".004")
    actuator = add(root, "actuator")
    present = {j.attrib["name"] for j in root.findall(".//joint") if "name" in j.attrib}
    for j, cap in zip(JOINTS, LIMITS, strict=True):
        if j in present:
            add(
                actuator,
                "position",
                name=j.removesuffix("_joint"),
                joint=j,
                kp="20",
                kv=".5",
                ctrllimited="false",
                forcelimited="true",
                forcerange=f"{-cap} {cap}",
            )
    sensor = add(root, "sensor")
    add(sensor, "gyro", name="gyro", site="imu")
    add(sensor, "velocimeter", name="velocity_truth", site="imu")
    # Range sensors point down and forward from a mast on the body. Native rays
    # intersect scene geometry; returned distances are delivered at 20 Hz.
    for i, (x, y) in enumerate(
        (x, y) for x in (0.25, 0.55, 0.85) for y in (-0.2, 0, 0.2)
    ):
        direction = np.array([x, y, -0.48])
        direction /= np.linalg.norm(direction)
        quat = np.zeros(4)
        mujoco.mju_quatZ2Vec(quat, direction)
        add(
            base,
            "site",
            name=f"ray_{i}",
            pos=".08 0 .18",
            size=".001",
            rgba="0 0 0 0",
            quat=" ".join(map(str, quat)),
        )
        if sensing:
            add(sensor, "rangefinder", name=f"range_{i}", site=f"ray_{i}")
    add(
        base,
        "camera",
        name="head",
        pos=".30 0 .05",
        xyaxes="0 -1 0 .2 0 .98",
        fovy="75",
    )
    add(
        world,
        "camera",
        name="overview",
        pos="2.2 -4 2.6",
        xyaxes="1 0 0 0 .52 .85",
        fovy="58",
    )
    if standing:
        from .standing_surfaces import compose

        compose(root, terrain.removeprefix("stand_"))
    if contact_profile not in ("firm", "legacy_soft"):
        raise ValueError(contact_profile)
    if contact_profile == "firm":
        # Newly composed feet otherwise inherit MuJoCo's 20 ms contact time
        # constant, permitting centimetres of penetration. Set every physical
        # surface explicitly, including the floor, obstacle and stump surfaces.
        for geom in world.iter("geom"):
            if geom.get("class") == "visual" or geom.get("contype") == "0":
                continue
            geom.attrib.update(solref=".006 1", solimp=".95 .99 .001", margin="0")
    if support_sensing:
        # Native contact-force sensors are reward/diagnostic truth only. They
        # preserve physical collisions and do not add channels to actor input.
        for element in world.iter("body"):
            for i, geom in enumerate(element.findall("geom")):
                if geom.get("class") == "visual" or geom.get("contype") == "0":
                    continue
                if not geom.get("name"):
                    geom.set("name", f"{element.get('name')}_collision_{i}")
                add(
                    sensor,
                    "contact",
                    name="support_" + geom.get("name"),
                    geom1=geom.get("name"),
                    body2="world",
                    data="found force",
                    reduce="netforce",
                    num="1",
                )
    spec = mujoco.MjSpec.from_string(ET.tostring(root, encoding="unicode"))
    model = spec.compile()
    model.vis.global_.bvactive = 0
    return model


def joint_mapping(model):
    indices, qpos, dofs, actuators = [], [], [], []
    for i, name in enumerate(JOINTS):
        j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if j >= 0:
            indices.append(i)
            qpos.append(model.jnt_qposadr[j])
            dofs.append(model.jnt_dofadr[j])
            actuators.append(model.actuator(name.removesuffix("_joint")).id)
    return tuple(np.array(x, dtype=int) for x in (indices, qpos, dofs, actuators))


def initialize(model, data):
    """Initialization only: avoid initial ground interpenetration in every body."""
    slot, qadr, _, _ = joint_mapping(model)
    data.qpos[qadr] = STAND[slot]
    mujoco.mj_forward(model, data)
    tips = [i for i in range(model.nsite) if model.site(i).name.endswith("_tip")]
    z = min(data.site_xpos[i, 2] for i in tips)
    data.qpos[2] += 0.024 - z
    mujoco.mj_forward(model, data)
    from .standing_surfaces import initialize_support, name_for

    surface = name_for(model)
    if surface is not None:
        initialize_support(model, data, surface)


def manifest(body, model):
    return {
        "body": asdict(body),
        "total_mass_kg": float(model.body_mass.sum()),
        "nq": model.nq,
        "nv": model.nv,
        "actuators": model.nu,
        "physics_timestep_s": model.opt.timestep,
        "torque_limits_nm": model.actuator_forcerange.tolist(),
        "joint_limits_rad": model.jnt_range.tolist(),
        "body_masses_kg": {
            model.body(i).name: float(model.body_mass[i]) for i in range(1, model.nbody)
        },
    }


def preview(output):
    from PIL import Image, ImageDraw

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    panels, records = [], []
    for key in ("healthy", "short_fl", "missing_fl", "mixed"):
        body = PRESETS[key]
        model = build_model(body, "steps")
        data = mujoco.MjData(model)
        initialize(model, data)
        renderer = mujoco.Renderer(model, height=480, width=640)
        camera = mujoco.MjvCamera()
        camera.lookat[:] = [0.1, 0, 0.22]
        camera.distance, camera.azimuth, camera.elevation = 1.4, 140, -22
        renderer.update_scene(data, camera=camera)
        panel = Image.fromarray(renderer.render())
        renderer.close()
        draw = ImageDraw.Draw(panel)
        draw.rectangle((0, 0, 640, 43), fill=(15, 24, 30))
        draw.text(
            (16, 14),
            f"{key.replace('_', ' ').upper()}  |  {model.body_mass.sum():.2f} kg  |  STATIC PREVIEW",
            fill="white",
        )
        panels.append(panel)
        records.append(manifest(body, model))
    board = Image.new("RGB", (1280, 960))
    for i, panel in enumerate(panels):
        board.paste(panel, ((i % 2) * 640, (i // 2) * 480))
    board.save(output / "bodies.png")
    (output / "bodies.json").write_text(json.dumps(records, indent=2) + "\n")
    model = build_model(PRESETS["short_fl"], "steps")
    data = mujoco.MjData(model)
    initialize(model, data)
    with mujoco.Renderer(model, height=720, width=1280) as renderer:
        camera = mujoco.MjvCamera()
        camera.lookat[:] = [2.2, 0, 0.1]
        camera.distance, camera.azimuth, camera.elevation = 6.0, 110, -32
        renderer.update_scene(data, camera=camera)
        Image.fromarray(renderer.render()).save(output / "course.png")
    return str(output / "bodies.png")
