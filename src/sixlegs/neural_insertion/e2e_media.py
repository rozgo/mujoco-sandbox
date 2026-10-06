"""Render recorded end-to-end states (replay, not simulation) into review frames and video.

Usage: uv run --locked python -m sixlegs.neural_insertion.e2e_media RUN_DIR [--speed 1] [--fps 30]
"""

import argparse
import json
from pathlib import Path

import imageio_ffmpeg
import mujoco
import numpy as np
from PIL import Image, ImageDraw

from .e2e_scene import load_e2e
from .visuals import font

W, H = 1920, 1080
PANEL_W, PANEL_H = 940, 860


def camera(lookat, distance, azimuth, elevation):
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = lookat
    cam.distance, cam.azimuth, cam.elevation = distance, azimuth, elevation
    return cam


def card(title, lines):
    """Closing card: what this attempt was and how it ended."""
    canvas = Image.new("RGB", (W, H), "#17152b")
    draw = ImageDraw.Draw(canvas)
    draw.text((120, 300), title, font=font(46), fill="#f1ecfa")
    for k, line in enumerate(lines):
        draw.text((120, 400+k*52), line, font=font(30), fill="#d9b8ff" if k == 0 else "#b9b3d6")
    return np.asarray(canvas)


def render(run, speed=1., fps=30, output=None, title=None, outcome=(), hold_s=3.):
    run = Path(run)
    trace = dict(np.load(run/"trace.npz", allow_pickle=True))
    report = json.loads((run/"report.json").read_text())
    modern = report.get("variant") == "modern"
    if modern:
        from .modern_scene import load_modern
        m, d, _ = load_modern()
    else:
        m, d, _ = load_e2e(tubes=report.get("variant") == "tubes")
    renderer = mujoco.Renderer(m, PANEL_H, PANEL_W)
    see_through = [m.geom("tissue_phantom").id]+[g for g in range(m.ngeom) if m.geom(g).name.startswith("tube_")]
    wide_distance, wide_drop = (22., 2.) if modern else (70., 8.)
    opaque = m.geom_rgba[see_through].copy()
    times = trace["time"]
    frames_t = np.arange(times[0], times[-1], speed/fps)
    output = Path(output or run/"cycle.mp4")
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio_ffmpeg.write_frames(str(output), (W, H), fps=fps, codec="libx264", quality=8,
                                         pix_fmt_out="yuv420p", macro_block_size=2,
                                         output_params=["-movflags", "+faststart"])
    writer.send(None)
    stills = run/"frames"
    stills.mkdir(exist_ok=True)
    last_phase = None
    for n, t in enumerate(frames_t):
        i = min(int(np.searchsorted(times, t)), len(times)-1)
        d.qpos[:] = trace["qpos"][i]
        mujoco.mj_kinematics(m, d)
        tip, eyelet = trace["tip"][i], trace["eyelet"][i]
        wide = camera((tip+eyelet)/2+np.array((0, 0, -wide_drop)), wide_distance, 135, -22)
        near = camera(eyelet, 2.2, 205, -18)
        canvas = Image.new("RGB", (W, H), "#17152b")
        for k, cam in enumerate((wide, near)):
            # Detail view only: phantom and channel drawn translucent (display only) to show the inserted thread.
            m.geom_rgba[see_through] = opaque
            if k == 1:
                m.geom_rgba[see_through, 3] = .3
            renderer.update_scene(d, cam)
            canvas.paste(Image.fromarray(renderer.render()), (13+k*(PANEL_W+14), 120))
        draw = ImageDraw.Draw(canvas)
        phase = str(trace["phase"][i])
        draw.text((24, 20), title or "END-TO-END CYCLE  /  SCRIPTED YARDSTICK, NO DISTURBANCES", font=font(28),
                  fill="#f1ecfa")
        draw.text((24, 64), f"t = {trace['time'][i]:.3f} s    phase: {phase}    playback {speed:g}x    "
                  f"eyelet depth {trace['depth'][i]:+.2f} mm    needle axial {trace['needle_axial'][i]/1e3:+.2f} mN",
                  font=font(22), fill="#d9b8ff")
        draw.text((24, 1000), "Wide: tool and thread   |   Detail: eyelet and slotted needle, about 2 mm across. "
                  "Replay of recorded MuJoCo states; DER thread, provisional tissue model.", font=font(20), fill="#b9b3d6")
        frame = np.asarray(canvas)
        writer.send(np.ascontiguousarray(frame))
        if phase != last_phase:
            Image.fromarray(frame).save(stills/f"{n:04d}_{phase}.png")
            last_phase = phase
    if outcome:
        end = card(title or "End-to-end cycle", list(outcome))
        for _ in range(round(hold_s*fps)):
            writer.send(np.ascontiguousarray(end))
        Image.fromarray(end).save(stills/"9999_card.png")
    writer.close()
    return output, len(frames_t)+(round(hold_s*fps) if outcome else 0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--speed", type=float, default=1.)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--title")
    parser.add_argument("--outcome", action="append", default=[], help="closing card line (repeatable)")
    args = parser.parse_args()
    print(render(args.run, args.speed, args.fps, args.output, args.title, args.outcome))


if __name__ == "__main__":
    main()
