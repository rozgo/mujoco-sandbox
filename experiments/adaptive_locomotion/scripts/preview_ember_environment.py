"""Review the industrial floor and every support family without running physics."""

import argparse
import hashlib
import json
from pathlib import Path

import mujoco
from adaptive_locomotion import ember, ember_environment
from adaptive_locomotion.bodies import build_model, initialize
from adaptive_locomotion.record import font
from PIL import Image, ImageDraw


def main(output):
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas = Image.new("RGB", (2400, 1860), ember.CHARCOAL)
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((30, 30, 43, 118), fill=ember.YELLOW)
    draw.text((65, 28), "EMBER / INDUSTRIAL TEST BAY", font=font(43), fill=ember.IVORY)
    draw.text(
        (67, 90),
        "STATIC ENVIRONMENT STUDY / Floor, steel surfaces and a consistent hazard-border system",
        font=font(23),
        fill=ember.IVORY,
    )
    cases = (
        (
            "flat",
            "01 / MARKED TEST FLOOR",
            "Panel seams, measured lane marks and distance stencils",
            3.5,
            -42,
            [1.1, 0.55, 0],
        ),
        (
            "moving",
            "02 / MOVING DECK",
            "Four-sided hazard border, inset plates and flush fasteners",
            2.45,
            -30,
            [0, 0, 0.32],
        ),
        (
            "stand_pads_extreme",
            "03 / MODULAR PADS",
            "Separate steel modules with readable edges and side panels",
            2.15,
            -35,
            [0, 0, 0.04],
        ),
        (
            "stand_slope_x_24",
            "04 / SLOPED SURFACE",
            "The same details follow the actual 24° support face",
            2.25,
            -27,
            [0, 0, 0.05],
        ),
        (
            "stand_steps_20",
            "05 / STEP MODULES",
            "Visible tread borders and dark riser faces",
            2.40,
            -28,
            [0.10, 0, 0.15],
        ),
        (
            "stand_gap_fr",
            "06 / MISSING SUPPORT",
            "The gap remains empty; markings make its boundary clear",
            2.20,
            -40,
            [0, 0, 0],
        ),
    )
    records = []
    for i, (terrain, title, note, distance, elevation, lookat) in enumerate(cases):
        model = build_model(terrain=terrain)
        data = mujoco.MjData(model)
        initialize(model, data)
        ember_environment.configure(model)
        # Hide the robot in this environment-only review; no stepping or deletion.
        base = model.body("base").id
        for gid in range(model.ngeom):
            bid = int(model.geom_bodyid[gid])
            while bid not in (0, base):
                bid = int(model.body_parentid[bid])
            if bid == base:
                model.geom_matid[gid] = -1
                model.geom_rgba[gid, 3] = 0
        model.vis.global_.offwidth, model.vis.global_.offheight = 1152, 430
        camera = mujoco.MjvCamera()
        camera.lookat[:] = lookat
        camera.distance, camera.elevation, camera.azimuth = distance, elevation, 50
        option = mujoco.MjvOption()
        option.flags[mujoco.mjtVisFlag.mjVIS_RANGEFINDER] = False
        option.sitegroup[:] = 0
        with mujoco.Renderer(model, width=1152, height=430) as renderer:
            renderer.update_scene(data, camera=camera, scene_option=option)
            before = renderer.scene.ngeom
            ember_environment.decorate(renderer.scene, model, data)
            picture = Image.fromarray(renderer.render())
            x, y = 32 + i % 2 * 1184, 159 + i // 2 * 550
            canvas.paste(picture, (x, y + 44))
            picture.save(output.parent / f"environment_{i:02d}.png")
            records.append(
                {"terrain": terrain, "observer_geoms": renderer.scene.ngeom - before}
            )
        draw.text((x + 4, y), title, font=font(27), fill=ember.YELLOW)
        draw.text((x + 4, y + 491), note, font=font(22), fill=ember.IVORY)
    draw.line((32, 1820, 2368, 1820), fill=ember.STEEL, width=2)
    draw.text(
        (36, 1834),
        "PREVIEW ONLY / NO PHYSICS STEPS / NO POLICY OR CONTACT CHANGES / ROBOT HIDDEN TO SHOW THE ENVIRONMENT",
        font=font(20),
        fill=ember.IVORY,
    )
    canvas.save(output)
    output.with_suffix(".json").write_text(
        json.dumps(
            {
                "static": True,
                "new_physics_steps": 0,
                "new_training_seconds": 0,
                "robot_hidden_for_preview": True,
                "theme_candidate": "ember_environment",
                "cases": records,
                "preview_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
            },
            indent=2,
        )
        + "\n"
    )
    print(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("previews/locomotion/ember/environment_v2.png"),
    )
    main(parser.parse_args().output)
