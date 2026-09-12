"""Compare source and generated video at matching elapsed times, without retiming."""

import argparse
import json
import math
from pathlib import Path

import imageio_ffmpeg
from PIL import Image, ImageDraw, ImageFont, ImageOps

from .cli import ROOT, inspect_video, save, sha


def font(size):
    for name in (
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "DejaVuSans.ttf",
    ):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default(size=size)


class Frames:
    def __init__(self, path):
        self.reader = imageio_ffmpeg.read_frames(str(path), pix_fmt="rgb24")
        self.meta = next(self.reader)
        self.index = -1
        self.image = None

    def at(self, seconds):
        target = math.floor(seconds * self.meta["fps"] + 1e-7)
        while self.index < target:
            raw = next(self.reader)
            self.image = Image.frombytes("RGB", self.meta["size"], raw)
            self.index += 1
        return self.image

    def close(self):
        self.reader.close()


def compare(source, generated, output):
    if output.exists():
        raise FileExistsError("Comparison exists; choose a new output path")
    source_info = json.loads(source.with_suffix(".json").read_text())
    gen_info = json.loads(generated.with_suffix(".json").read_text())
    if sha(source) != source_info["sha256"] or sha(generated) != gen_info["sha256"]:
        raise ValueError("Video differs from saved manifest")
    if gen_info["source_sha256"] != sha(source):
        raise ValueError("Generated result belongs to a different source")
    duration = min(source_info["duration_s"], gen_info["duration_s"])
    fps, width, height = 25, 1920, 900
    count = math.floor(duration * fps + 1e-7)
    a, b = Frames(source), Frames(generated)
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio_ffmpeg.write_frames(
        str(output),
        (width, height),
        fps=fps,
        codec="libx264",
        quality=8,
        pix_fmt_out="yuv420p",
        macro_block_size=1,
        output_params=["-movflags", "+faststart"],
    )
    writer.send(None)
    model_label = gen_info["model"].split("/")[1].replace("-", " ").upper()
    try:
        for i in range(count):
            canvas = Image.new("RGB", (width, height), "#091219")
            draw = ImageDraw.Draw(canvas)
            draw.text(
                (24, 18),
                "FROM SIMULATION TO CINEMATIC PRESENTATION",
                font=font(34),
                fill="#91d8d0",
            )
            draw.text((24, 76), "ORIGINAL MUJOCO", font=font(25), fill="white")
            draw.text(
                (984, 76), f"AI RESTYLE / {model_label}", font=font(25), fill="#ffc595"
            )
            for x, frame in ((0, a.at(i / fps)), (960, b.at(i / fps))):
                panel = ImageOps.contain(frame, (960, 675), Image.Resampling.LANCZOS)
                canvas.paste(
                    panel, (x + (960 - panel.width) // 2, 120 + (675 - panel.height) // 2)
                )
            draw.text(
                (24, 817),
                f"{i / fps:.2f} s  |  Matched elapsed time  |  1x playback",
                font=font(25),
                fill="white",
            )
            draw.text(
                (24, 859),
                "Generative presentation test. "
                "Robot motion and contacts must be checked against the original.",
                font=font(23),
                fill="#b6c8ce",
            )
            writer.send(canvas.tobytes())
    finally:
        a.close()
        b.close()
        writer.close()
    info = inspect_video(output, ROOT / "build/video_styling/comparison")
    assert info["frames"] == count
    info.update(
        source=str(source.relative_to(ROOT)),
        generated=str(generated.relative_to(ROOT)),
        source_sha256=sha(source),
        generated_sha256=sha(generated),
        playback_speed=1,
        retimed=False,
        frame_selection="Nearest preceding source frame at each 25 Hz elapsed timestamp",
        visual_inspection="pending",
    )
    save(output.with_suffix(".json"), info)
    chosen = info["inspection_frames"]
    sheet = Image.new("RGB", (1280, 3 * 330), "#091219")
    for i, path in enumerate(chosen[:6]):
        frame = Image.open(ROOT / path)
        sheet.paste(frame.resize((640, 300)), ((i % 2) * 640, (i // 2) * 330))
        ImageDraw.Draw(sheet).text(
            ((i % 2) * 640 + 10, (i // 2) * 330 + 306),
            Path(path).stem,
            font=font(16),
            fill="white",
        )
    sheet.save(ROOT / "build/video_styling/contact_sheet.png")
    return info


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--generated", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    print(
        json.dumps(
            compare(a.source.resolve(), a.generated.resolve(), a.output.resolve()), indent=2
        )
    )


if __name__ == "__main__":
    main()
