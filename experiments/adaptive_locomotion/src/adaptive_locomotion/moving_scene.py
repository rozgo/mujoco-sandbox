"""Six-axis physical motion deck. Scene composition and reset only."""

import json

import mujoco

from .bodies import ROOT, add

AXES = ("x", "y", "z", "yaw", "pitch", "roll")
TOP = 0.50


def compose(root):
    from .standing_surfaces import compose as standing_compose

    standing_compose(root, "flat")
    world = root.find("worldbody")
    deck = add(world, "body", name="motion_deck", pos="0 0 .45")
    # A simplified motion table: independent slides and gimbal joints. Finite
    # mass, real reaction forces and bounded servos; no mocap or state writes.
    add(deck, "inertial", pos="0 0 0", mass="35", diaginertia="4.23 7.50 11.67")
    directions = ("1 0 0", "0 1 0", "0 0 1", "0 0 1", "0 1 0", "1 0 0")
    for i, (name, axis) in enumerate(zip(AXES, directions, strict=True)):
        travel = ".55" if i < 2 else ".20" if i == 2 else "1.4" if i == 3 else ".45"
        add(
            deck,
            "joint",
            name=f"deck_{name}",
            type="slide" if i < 3 else "hinge",
            axis=axis,
            range=f"-{travel} {travel}",
            damping="0",
            armature="0",
        )
        cap, kp, kv = (3000, 30000, 1800) if i < 3 else (1000, 6000, 350)
        add(
            root.find("actuator"),
            "position",
            name=f"deck_{name}",
            joint=f"deck_{name}",
            kp=str(kp),
            kv=str(kv),
            ctrlrange=f"-{travel} {travel}",
            forcerange=f"-{cap} {cap}",
            forcelimited="true",
        )
    add(
        deck,
        "geom",
        name="deck_surface",
        type="box",
        size=".8 .6 .05",
        rgba=".13 .43 .46 1",
        friction=".8 .005 .0001",
        condim="3",
    )
    for x in (-0.60, -0.30, 0, 0.30, 0.60):
        add(
            deck,
            "geom",
            type="box",
            size=".004 .59 .0005",
            pos=f"{x} 0 .0506",
            rgba=".46 .75 .73 1",
            contype="0",
            conaffinity="0",
            mass="0",
            group="2",
        )
    add(deck, "site", name="deck_top", pos="0 0 .05", size=".001", rgba="0 0 0 0")
    for sensor in ("framepos", "framequat", "framelinvel", "frameangvel"):
        add(
            root.find("sensor"),
            sensor,
            name=f"deck_{sensor}",
            objtype="site",
            objname="deck_top",
        )
    # The base is a visible physical fixture below the working envelope.
    add(
        world,
        "geom",
        name="deck_fixture",
        type="box",
        size=".25 .20 .06",
        pos="0 0 .06",
        rgba=".11 .14 .18 1",
        friction=".8 .005 .0001",
    )
    for y in (-1.0, 1.0):
        add(
            world,
            "geom",
            type="box",
            size="2 .012 .001",
            pos=f"0 {y} .001",
            rgba=".8 .47 .15 1",
            contype="0",
            conaffinity="0",
        )
    environment = add(world, "body", name="support_environment")
    for element in list(world):
        if element.tag == "geom" or element is deck:
            world.remove(element)
            environment.append(element)
    add(root.find("custom"), "numeric", name="moving_support", data="1")


def is_moving(model):
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "motion_deck") >= 0


def initialize(model, data):
    # Standard flat initialization is reused, translated once at reset.
    data.qpos[2] += TOP
    mujoco.mj_forward(model, data)


def preview():
    from PIL import Image, ImageDraw

    from .bodies import PRESETS, build_model, manifest
    from .bodies import initialize as reset
    from .presentation import configure
    from .record import font

    model = build_model(PRESETS["healthy"], "moving")
    data = mujoco.MjData(model)
    reset(model, data)
    configure(model)
    canvas = Image.new("RGB", (1920, 720), "#091219")
    option = mujoco.MjvOption()
    option.flags[mujoco.mjtVisFlag.mjVIS_RANGEFINDER] = False
    for i, (azimuth, elevation, distance) in enumerate(
        ((45, -23, 2.65), (90, -65, 2.5), (40, -12, 1.5))
    ):
        camera = mujoco.MjvCamera()
        camera.lookat[:] = [0, 0, 0.53]
        camera.azimuth, camera.elevation, camera.distance = azimuth, elevation, distance
        with mujoco.Renderer(model, height=620, width=640) as renderer:
            renderer.update_scene(data, camera=camera, scene_option=option)
            canvas.paste(Image.fromarray(renderer.render()), (i * 640, 60))
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (22, 17),
        "MOVING SUPPORT LAB  |  Static scene preview",
        font=font(30),
        fill="#91d8d0",
    )
    draw.text(
        (22, 679),
        "35 kg actuated deck  /  6 axes  /  free-body dog  /  original joint torque limits",
        font=font(24),
        fill="white",
    )
    output = ROOT / "previews/locomotion/moving/static.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)
    report = manifest(PRESETS["healthy"], model)
    report.pop("joint_limits_rad")
    report.pop("torque_limits_nm")
    report["joint_limits"] = [
        {
            "name": model.joint(i).name,
            "range": model.jnt_range[i].tolist(),
            "units": "m"
            if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_SLIDE
            else "rad"
            if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_HINGE
            else "free joint",
        }
        for i in range(model.njnt)
    ]
    report["actuator_force_limits"] = [
        {
            "name": model.actuator(i).name,
            "range": model.actuator_forcerange[i].tolist(),
            "units": "N"
            if model.actuator(i).name in ("deck_x", "deck_y", "deck_z")
            else "N m",
        }
        for i in range(model.nu)
    ]
    report["initial_max_penetration_m"] = (
        max(0, -float(data.contact.dist.min())) if data.ncon else 0
    )
    report["initial_qpos"] = data.qpos.tolist()
    output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    preview()
