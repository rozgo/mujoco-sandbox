"""Complete segment removals, separate from archived shortened-calf work."""

import json
from pathlib import Path

from .bodies import LEGS, PRESETS, BodySpec

LOWER_BODIES = tuple(BodySpec(f"lower_{leg.lower()}", absent=(leg,)) for leg in LEGS)
WHOLE_BODIES = tuple(
    BodySpec(f"whole_{leg.lower()}", absent_legs=(leg,)) for leg in LEGS
)
LOSS_BODIES = (*LOWER_BODIES, *WHOLE_BODIES)
LOSS_CASES = ("healthy", *(b.name for b in LOSS_BODIES))


def curriculum(stage, n):
    if n < 32 or n % 16:
        raise ValueError("Limb-loss stages require a multiple of 16, at least 32")
    if stage == "one":
        return [PRESETS["healthy"], LOWER_BODIES[1]], [n // 2, n // 2]
    if stage == "lower":
        return [PRESETS["healthy"], *LOWER_BODIES], [n // 4] + [3 * n // 16] * 4
    if stage == "whole":
        return [PRESETS["healthy"], *LOSS_BODIES], [n // 4] + [n // 16] * 4 + [
            n // 8
        ] * 4
    if stage == "front":
        small = n // 32
        healthy = n // 4
        focus = (n - healthy - 6 * small) // 2
        counts = [n - 6 * small - 2 * focus] + [
            focus if b.name in ("lower_fr", "whole_fr") else small for b in LOSS_BODIES
        ]
        return [PRESETS["healthy"], *LOSS_BODIES], counts
    if stage == "consolidate":
        healthy = n // 4
        remaining = n - healthy
        return [PRESETS["healthy"], *LOSS_BODIES], [healthy] + [
            remaining // 8 + (i < remaining % 8) for i in range(8)
        ]
    raise ValueError(stage)


def preview(output):
    import mujoco
    from PIL import Image, ImageDraw

    from .bodies import build_model, initialize, manifest
    from .record import font

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    board = Image.new("RGB", (1920, 1080), (12, 22, 29))
    records = []
    # Representative removals, both sides, all nine compiled body manifests.
    for b in (PRESETS["healthy"], *LOSS_BODIES):
        m = build_model(b)
        records.append(manifest(b, m))
    for i, b in enumerate((PRESETS["healthy"], LOWER_BODIES[1], WHOLE_BODIES[1])):
        m = build_model(b)
        d = mujoco.MjData(m)
        initialize(m, d)
        for row, azimuth in enumerate((50, 90)):
            camera = mujoco.MjvCamera()
            camera.lookat[:] = [0, 0, 0.24]
            camera.distance, camera.azimuth, camera.elevation = 1.1, azimuth, -16
            option = mujoco.MjvOption()
            option.flags[mujoco.mjtVisFlag.mjVIS_RANGEFINDER] = False
            with mujoco.Renderer(m, height=440, width=640) as renderer:
                renderer.update_scene(d, camera=camera, scene_option=option)
                board.paste(
                    Image.fromarray(renderer.render()), (i * 640, row * 540 + 72)
                )
        draw = ImageDraw.Draw(board)
        title = ("HEALTHY", "FR LOWER LEG REMOVED", "FR ENTIRE LEG REMOVED")[i]
        draw.text((i * 640 + 20, 15), title, font=font(24), fill="white")
        draw.text(
            (i * 640 + 20, 48),
            f"{m.body_mass.sum():.3f} kg / {m.nu} actuators",
            font=font(18),
            fill="#6ee0cc",
        )
        draw.text(
            (i * 640 + 20, 550),
            (
                "Intact feet",
                "Orange = exposed upper-leg stump",
                "Only three supporting legs",
            )[i],
            font=font(20),
            fill="white",
        )
    board.save(output)
    output.with_suffix(".json").write_text(json.dumps(records, indent=2) + "\n")
    return str(output)
