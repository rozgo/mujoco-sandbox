"""Render a static industrial-palette study with all seven user-selected colors."""

import argparse
import hashlib
import json
from pathlib import Path

import mujoco
from adaptive_locomotion import ember
from adaptive_locomotion.bodies import BodySpec, build_model, initialize
from adaptive_locomotion.record import font
from PIL import Image, ImageDraw


def preview(output):
    canvas = Image.new("RGB", (1920, 1200), ember.CHARCOAL)
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((30, 28, 40, 119), fill=ember.YELLOW)
    draw.text((64, 25), "EMBER / FIELD ROBOTICS", font=font(43), fill=ember.IVORY)
    draw.text(
        (66, 88),
        "PALETTE STUDY     /     Steel, machinery yellow, oxide red and forest green",
        font=font(23),
        fill=ember.IVORY,
    )
    cases = (
        (
            BodySpec(),
            "flat",
            50,
            "01 / MACHINERY YELLOW",
            "Painted shell / exposed steel",
        ),
        (
            BodySpec(),
            "flat",
            -90,
            "02 / CHARCOAL + STEEL",
            "Dark joints / visible ground contact",
        ),
        (
            BodySpec("lower_fr", absent=("FR",)),
            "flat",
            50,
            "03 / DAMAGE ACCENT",
            "Oxide red marks the physical cut",
        ),
        (
            BodySpec(),
            "moving",
            50,
            "04 / SUPPORT SURFACE",
            "Neutral deck / yellow edge markings",
        ),
    )
    option = mujoco.MjvOption()
    option.flags[mujoco.mjtVisFlag.mjVIS_RANGEFINDER] = False
    for i, (body, terrain, azimuth, title, note) in enumerate(cases):
        model = build_model(body, terrain=terrain)
        data = mujoco.MjData(model)
        initialize(model, data)
        ember.configure(model)
        model.vis.global_.offwidth, model.vis.global_.offheight = 912, 334
        camera = mujoco.MjvCamera()
        camera.lookat[:] = data.qpos[:3] - [0, 0, 0.05 if terrain == "flat" else 0.18]
        camera.azimuth = azimuth
        camera.elevation = -18 if terrain == "flat" else -26
        camera.distance = 1.3 if terrain == "flat" else 2.5
        with mujoco.Renderer(model, height=334, width=912) as renderer:
            renderer.update_scene(data, camera=camera, scene_option=option)
            ember.decorate(renderer.scene, model, data, body)
            x, y = 32 + i % 2 * 944, 154 + i // 2 * 420
            canvas.paste(Image.fromarray(renderer.render()), (x, y + 40))
        draw.text((x, y), title, font=font(24), fill=ember.YELLOW)
        draw.text(
            (x, y + 383), note, font=font(21), fill=ember.RED if i == 2 else ember.IVORY
        )
    roles = ("BASE", "STRUCTURE", "TYPE", "PANELS", "MECHANISMS", "CONTACT", "DAMAGE")
    for i, ((name, color), role) in enumerate(
        zip(ember.PALETTE.items(), roles, strict=True)
    ):
        x = 32 + i * 267
        draw.rectangle((x, 1035, x + 246, 1076), fill=color, outline=ember.STEEL)
        draw.text(
            (x, 1090), f"{name.upper()}  {color}", font=font(18), fill=ember.IVORY
        )
        draw.text((x, 1120), role, font=font(18), fill=ember.IVORY)
    draw.text(
        (32, 1164),
        "STATIC PREVIEW / NATIVE MUJOCO RENDER",
        font=font(18),
        fill=ember.IVORY,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)
    output.with_suffix(".json").write_text(
        json.dumps(
            {
                "theme": "ember",
                "palette": ember.PALETTE,
                "dimensions": list(canvas.size),
                "static": True,
                "new_training_seconds": 0,
                "new_simulation_steps": 0,
                "preview_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
                "scope": "Observer-only materials, lighting, overlays and decoration. Original vendor assets, policy, mass, contacts and actuators are unchanged.",
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    preview(parser.parse_args().output)
