"""Physical support surfaces and initialization-only pose placement."""

import json

import mujoco
import numpy as np

from .bodies import LEGS, ROOT, add, joint_mapping, manifest

SURFACES = (
    "flat",
    "pads",
    "slope_x",
    "slope_y",
    "gap_fl",
    "gap_fr",
    "gap_rl",
    "gap_rr",
    "pads_high",
    "pads_extreme",
    "slope_x_12",
    "slope_y_12",
    "slope_x_18",
    "slope_y_18",
    "slope_x_24",
    "slope_y_24",
    "steps_12",
    "steps_20",
    "steps_28",
)
GENTLE = SURFACES[:8]
AGGRESSIVE = SURFACES[8:]
PAD_HEIGHTS = np.array([0.065, 0.015, 0.040, 0.0])


def pad_heights(name):
    return {
        "pads": PAD_HEIGHTS,
        "pads_high": np.array([0.18, 0.02, 0.10, 0]),
        "pads_extreme": np.array([0.24, 0, 0.04, 0.16]),
    }[name]


def slope_degrees(name):
    return float(name.split("_")[2]) if len(name.split("_")) == 3 else 6.0


def surface_height(name, x, y):
    x, y = np.broadcast_arrays(np.asarray(x), np.asarray(y))
    if name == "flat":
        return np.zeros_like(x, dtype=float)
    if name.startswith("slope"):
        along_x = name.split("_")[1] == "x"
        axis, transverse = (x, y) if along_x else (y, x)
        extent, transverse_extent = (0.75, 0.6) if along_x else (0.6, 0.75)
        angle = np.deg2rad(slope_degrees(name))
        height = 0.12 - np.tan(angle) * axis
        # Shift of a tilted box's top-face center relative to world origin is zero.
        inside = (np.abs(axis / np.cos(angle)) <= extent) & (
            np.abs(transverse) <= transverse_extent
        )
        return np.where(inside, height, -0.40)
    if name.startswith("steps"):
        riser = float(name.split("_")[1]) / 100
        level = np.where(x < 0, 0, np.where(x < 0.45, 1, 2))
        return np.where((np.abs(x) <= 0.9) & (np.abs(y) <= 0.6), riser * level, -0.40)
    index = (x < 0).astype(int) * 2 + (y < 0).astype(int)
    inside = (np.abs(x) <= 0.75) & (np.abs(y) <= 0.6)
    height = (
        pad_heights(name)[index]
        if name.startswith("pads")
        else np.full_like(x, 0.12, dtype=float)
    )
    if name.startswith("gap_"):
        missing = LEGS.index(name[-2:].upper())
        height = np.where(index == missing, -0.40, height)
    return np.where(inside, height, -0.40)


def compose(root, name):
    if name not in SURFACES:
        raise ValueError(name)
    world = root.find("worldbody")
    floor = world.find("geom[@name='floor']")
    floor.set("pos", "0 0 0" if name == "flat" else "0 0 -0.40")
    floor.set("rgba", ".24 .28 .32 1")
    # Remove flat-floor lane paint, which is observer-only.
    for geom in list(world.findall("geom")):
        if geom.get("contype") == "0":
            world.remove(geom)
    custom = root.find("custom")
    if custom is None:
        custom = add(root, "custom")
    add(custom, "numeric", name="standing_surface", data=str(SURFACES.index(name)))
    if name.startswith("slope"):
        rotation = (
            np.array([0.0, np.deg2rad(slope_degrees(name)), 0.0])
            if name.split("_")[1] == "x"
            else np.array([-np.deg2rad(slope_degrees(name)), 0.0, 0.0])
        )
        normal = (
            np.array(
                [
                    np.sin(np.deg2rad(slope_degrees(name))),
                    0,
                    np.cos(np.deg2rad(slope_degrees(name))),
                ]
            )
            if name.split("_")[1] == "x"
            else np.array(
                [
                    0,
                    np.sin(np.deg2rad(slope_degrees(name))),
                    np.cos(np.deg2rad(slope_degrees(name))),
                ]
            )
        )
        center = np.array([0, 0, 0.12]) - 0.06 * normal
        add(
            world,
            "geom",
            name="standing_slope",
            type="box",
            size=".75 .6 .06",
            pos=" ".join(map(str, center)),
            euler=" ".join(map(str, rotation)),
            rgba=".17 .45 .48 1",
            friction=".8 .005 .0001",
            condim="3",
        )
    elif name.startswith("steps"):
        riser = float(name.split("_")[1]) / 100
        for i, (left, right) in enumerate(((-0.9, 0), (0, 0.45), (0.45, 0.9))):
            top = i * riser
            half = (top + 0.40) / 2
            add(
                world,
                "geom",
                name=f"standing_step_{i}",
                type="box",
                size=f"{(right - left) / 2} .6 {half}",
                pos=f"{(left + right) / 2} 0 {top - half}",
                rgba=".17 .45 .48 1",
                friction=".8 .005 .0001",
                condim="3",
            )
    elif name != "flat":
        for i, leg in enumerate(LEGS):
            if name == f"gap_{leg.lower()}":
                continue
            x, y = (0.375 if i < 2 else -0.375), (0.30 if i % 2 == 0 else -0.30)
            top = pad_heights(name)[i] if name.startswith("pads") else 0.12
            half = (top + 0.40) / 2
            add(
                world,
                "geom",
                name=f"standing_pad_{leg}",
                type="box",
                size=f".375 .30 {half}",
                pos=f"{x} {y} {top - half}",
                rgba=".17 .45 .48 1",
                friction=".8 .005 .0001",
                condim="3",
            )
    sensor, base = root.find("sensor"), world.find("body[@name='base']")
    # Put the observer camera ahead of the nose mesh, rather than inside it.
    # Cameras do not contribute mass, collision geometry or policy observations.
    base.find("camera[@name='head']").set("pos", ".40 0 .08")
    for i in range(4):
        x, y = (0.12 if i < 2 else -0.12), (0.06 if i % 2 == 0 else -0.06)
        add(
            base,
            "site",
            name=f"standing_ray_{i}",
            pos=f"{x} {y} -.065",
            quat="0 1 0 0",
            size=".001",
            rgba="0 0 0 0",
        )
        add(sensor, "rangefinder", name=f"standing_range_{i}", site=f"standing_ray_{i}")


def name_for(model):
    index = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_NUMERIC, "standing_surface")
    return (
        None
        if index < 0
        else SURFACES[int(model.numeric_data[model.numeric_adr[index]])]
    )


def initialize_support(model, data, name):
    """Set one collision-safe initial pose; never called during live stepping."""
    slot, qadr, dadr, _ = joint_mapping(model)
    geoms = []
    for leg in LEGS:
        for suffix in ("terminal", "stump"):
            gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"{leg}_{suffix}")
            if gid >= 0:
                geoms.append((leg, gid))
                break
    mujoco.mj_forward(model, data)
    targets = {leg: data.geom_xpos[gid].copy() for leg, gid in geoms}
    heights = [
        float(surface_height(name, point[0], point[1])) for point in targets.values()
    ]
    reference = float(np.median([h for h in heights if h > -0.3]))
    data.qpos[2] = reference + 0.30
    if name.startswith("gap"):
        missing = LEGS.index(name[-2:].upper())
        data.qpos[0] -= 0.035 if missing < 2 else -0.035
        data.qpos[1] -= 0.035 if missing % 2 == 0 else -0.035
    for (leg, gid), height in zip(geoms, heights, strict=True):
        targets[leg][2] = (
            (height if height > -0.3 else reference + 0.06)
            + model.geom_size[gid, 0]
            + 0.002
        )
    jac = np.zeros((3, model.nv))
    for _ in range(80):
        worst = 0.0
        for leg, gid in geoms:
            indices = np.flatnonzero(slot // 3 == LEGS.index(leg))
            mujoco.mj_forward(model, data)
            error = targets[leg] - data.geom_xpos[gid]
            worst = max(worst, float(np.linalg.norm(error)))
            mujoco.mj_jacGeom(model, data, jac, None, gid)
            j = jac[:, dadr[indices]]
            delta = j.T @ np.linalg.solve(j @ j.T + np.eye(3) * 1e-5, error)
            data.qpos[qadr[indices]] += np.clip(delta, -0.08, 0.08)
            for address in qadr[indices]:
                joint = np.flatnonzero(model.jnt_qposadr == address)[0]
                data.qpos[address] = np.clip(
                    data.qpos[address], *model.jnt_range[joint]
                )
        if worst < 0.0001:
            break
    mujoco.mj_forward(model, data)


def preview(aggressive=False):
    from PIL import Image, ImageDraw

    from .bodies import PRESETS, build_model, initialize
    from .presentation import configure
    from .record import font

    output = (
        ROOT
        / "previews/locomotion/standing"
        / ("static_aggressive.png" if aggressive else "static.png")
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    names = (
        (
            "pads_high",
            "pads_extreme",
            "slope_x_18",
            "slope_y_24",
            "steps_20",
            "steps_28",
        )
        if aggressive
        else SURFACES[:6]
    )
    canvas = Image.new("RGB", (1920, 1080), "#091219")
    records = []
    for i, name in enumerate(names):
        m = build_model(PRESETS["healthy"], f"stand_{name}")
        d = mujoco.MjData(m)
        initialize(m, d)
        configure(m)
        cam = mujoco.MjvCamera()
        cam.lookat[:] = [0, 0, 0.14]
        cam.distance, cam.azimuth, cam.elevation = (
            1.55,
            (-50 if name.endswith("fl") else 50),
            -23,
        )
        option = mujoco.MjvOption()
        option.flags[mujoco.mjtVisFlag.mjVIS_RANGEFINDER] = False
        with mujoco.Renderer(m, height=470, width=640) as renderer:
            renderer.update_scene(d, camera=cam, scene_option=option)
            canvas.paste(
                Image.fromarray(renderer.render()), ((i % 3) * 640, (i // 3) * 540 + 60)
            )
        draw = ImageDraw.Draw(canvas)
        draw.text(
            ((i % 3) * 640 + 16, (i // 3) * 540 + 12),
            name.upper().replace("_", " "),
            font=font(27),
            fill="#91d8d0",
        )
        records.append(
            {
                "surface": name,
                **manifest(PRESETS["healthy"], m),
                "initial_max_penetration_m": max(0, -float(d.contact.dist.min()))
                if d.ncon
                else 0,
                "initial_qpos": d.qpos.tolist(),
            }
        )
    canvas.save(output)
    output.with_suffix(".json").write_text(json.dumps(records, indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    import sys

    preview(aggressive="--aggressive" in sys.argv)
