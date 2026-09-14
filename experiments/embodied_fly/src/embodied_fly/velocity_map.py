"""Replace a tiny overview with a fixed-scale map of the recorded physical path."""

import argparse
import json
import time
from pathlib import Path

import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw

from embodied_fly.provenance import evidence, sha256, utc_now
from embodied_fly.record import font


def record(source, video, output):
    started = time.perf_counter()
    if output.exists():
        raise FileExistsError("Preserve the previous video")
    manifest = json.loads(video.with_suffix(".json").read_text())
    assert sha256(video) == manifest["sha256"]
    assert sha256(source / "capture.npz") == manifest["capture_sha256"]
    with np.load(source / "capture.npz") as capture:
        positions = capture["qpos"][:, :2] * 10  # CGS centimeters to millimeters.
        headings = capture["heading"].copy()
        times = capture["time"].copy()
    center = (positions.min(0) + positions.max(0)) / 2
    span = np.maximum(np.ptp(positions, axis=0) + 10, 10)
    scale = float(min(424 / span[0], 224 / span[1]))
    screen_center = np.array([1344.0, 297.0])
    xy = (positions - center) * [scale, -scale] + screen_center
    panel = Image.new("RGB", (480, 330), "#1b2025")
    background = ImageDraw.Draw(panel)
    background.text((12, 9), "FLIGHT PATH / WORLD XY", font=font(18), fill="#e6e1db")
    low = center - np.array([212, 112]) / scale
    high = center + np.array([212, 112]) / scale
    for axis in range(2):
        for value in np.arange(np.ceil(low[axis] / 5) * 5, high[axis], 5):
            if axis == 0:
                x = 240 + (value - center[0]) * scale
                background.line((x, 60, x, 284), fill="#343a3d")
            else:
                y = 172 - (value - center[1]) * scale
                background.line((28, y, 452, y), fill="#343a3d")
    background.text(
        (12, 307), "5 mm grid | dot: start | arrow: fly", font=font(15), fill="#a8b0b5"
    )
    reader = imageio_ffmpeg.read_frames(str(video), pix_fmt="rgb24")
    metadata = next(reader)
    size, fps = tuple(manifest["size"]), manifest["fps"]
    assert metadata["size"] == size and metadata["fps"] == fps
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio_ffmpeg.write_frames(
        str(output),
        size,
        fps=fps,
        codec="libx264",
        quality=8,
        pix_fmt_in="rgb24",
        pix_fmt_out="yuv420p",
        macro_block_size=1,
        output_params=["-movflags", "+faststart"],
    )
    writer.send(None)
    count = 0
    try:
        for index, raw in enumerate(reader):
            step = round(index / fps / 0.002)
            assert abs(times[step] - index / fps) < 1e-9
            board = Image.fromarray(np.frombuffer(raw, np.uint8).reshape(size[1], size[0], 3))
            board.paste(panel, (1104, 125))
            draw = ImageDraw.Draw(board)
            past = list(map(tuple, xy[: step + 1 : 50]))
            if len(past) > 1:
                draw.line(past, fill="#67665f", width=2)
            recent = list(map(tuple, xy[max(0, step - 1000) : step + 1 : 10]))
            if len(recent) > 1:
                draw.line(recent, fill="#70a88a", width=3)
            x, y = xy[0]
            draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill="#e6e1db")
            at, yaw = xy[step], headings[step]
            forward = np.array([np.cos(yaw), -np.sin(yaw)])
            side = np.array([-forward[1], forward[0]])
            draw.polygon(
                [
                    tuple(at + forward * 10),
                    tuple(at - forward * 7 + side * 6),
                    tuple(at - forward * 7 - side * 6),
                ],
                fill="#ffc31f",
            )
            writer.send(np.asarray(board))
            count += 1
    finally:
        reader.close()
        writer.close()
    assert count == manifest["frames"]
    result = {
        **manifest,
        "provenance": evidence(),
        "completed_utc": utc_now(),
        "source_video_sha256": manifest["sha256"],
        "sha256": sha256(output),
        "source_render_seconds": manifest["render_seconds"],
        "render_seconds": time.perf_counter() - started,
        "render_operation": "Re-encode original frames with recorded XY path map; no new physics or temporal resampling",
        "camera": {
            **manifest["camera"],
            "overview": "Fixed-scale world XY map; actual past trajectory and current heading; marker is not body scale",
        },
        "map": {
            "grid_mm": 5,
            "pixels_per_mm": scale,
            "center_world_mm": center.tolist(),
            "recent_path_seconds": 2,
        },
    }
    output.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    record(args.source, args.video, args.output)
