"""Show every captured command case from one checkpoint, without omitting failures."""

import argparse
import json
import subprocess
import tempfile
import time
from pathlib import Path

import imageio_ffmpeg

from embodied_fly.provenance import evidence, sha256, utc_now
from embodied_fly.record import record


def montage(source, output):
    if output.exists():
        raise FileExistsError("Preserve earlier videos; choose a new output")
    evaluation = json.loads((source / "report.json").read_text())
    if not evaluation["one_checkpoint_for_all_cases"] or not evaluation["results"]:
        raise ValueError("A nonempty single-checkpoint evaluation is required")
    started = time.perf_counter()
    provenance = evidence()
    segments = []
    with tempfile.TemporaryDirectory(prefix="fly-montage-") as temporary:
        folder = Path(temporary)
        for i, case in enumerate(evaluation["results"]):
            video = folder / f"segment_{i:02d}.mp4"
            record(source, case["case"], video)
            segments.append(json.loads(video.with_suffix(".json").read_text()))
        listing = folder / "concat.txt"
        listing.write_text(
            "".join(f"file 'segment_{i:02d}.mp4'\n" for i in range(len(segments)))
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                imageio_ffmpeg.get_ffmpeg_exe(),
                "-v",
                "error",
                "-f",
                "concat",
                "-safe",
                "1",
                "-i",
                str(listing),
                "-c",
                "copy",
                "-movflags",
                "+faststart",
                str(output),
            ],
            check=True,
        )
    reader = imageio_ffmpeg.read_frames(str(output))
    metadata = next(reader)
    frames = sum(1 for _ in reader)
    expected = sum(segment["frames"] for segment in segments)
    if frames != expected or metadata["size"] != (1600, 900) or metadata["fps"] != 50:
        raise RuntimeError("Encoded montage does not match the complete case captures")
    report = {
        "provenance": provenance,
        "completed_utc": utc_now(),
        "source_evaluation_sha256": sha256(source / "report.json"),
        "checkpoint_sha256": evaluation["checkpoint_sha256"],
        "all_source_cases_included": True,
        "segments": segments,
        "frames": frames,
        "fps": 50,
        "duration_seconds": frames / 50,
        "size": metadata["size"],
        "playback_multiplier": 1,
        "fully_decoded": True,
        "render_and_encode_seconds": time.perf_counter() - started,
        "video_sha256": sha256(output),
    }
    output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k not in ("provenance", "segments")}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    montage(args.source, args.output)
