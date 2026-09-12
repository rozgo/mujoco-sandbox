"""Halve each completed chapter by trimming contiguous frames, never speeding up."""

import argparse
import json
import subprocess
import time
from pathlib import Path

import imageio_ffmpeg
import numpy as np
from adaptive_locomotion.bodies import ROOT
from PIL import Image, ImageDraw
from record_unified_showcase import EXPECTED, FPS, read, sha, write


def main(source, output):
    if output.exists():
        raise ValueError("Preserve previous cuts; select a new output filename")
    parent = read(source.with_suffix(".json"))
    assert sha(source) == parent["video_sha256"]
    assert parent["checkpoint_sha256"] == EXPECTED
    assert parent["environment"] == "industrial" and parent["playback_speed"] == 1
    assert parent["frames"] == 4350 and parent["fps"] == FPS
    started = time.perf_counter()
    edits, filters, cursor = [], [], 0
    for i, chapter in enumerate(parent["chapters"]):
        start, end = (round(chapter[k] * FPS) for k in ("start_s", "end_s"))
        original = end - start
        assert original % 2 == 0
        count = original // 2
        cases = chapter["cases"]
        offset, reason = 0, "First half retains initial support and gait adjustments"
        if cases and all(k.startswith("transition_") for k in cases):
            offset = 3 * FPS
            reason = "3–9 s retains walking, the full hold, and walking again"
        elif cases and all(k.startswith("moving_") for k in cases):
            offset = (original - count + 1) // 2
            reason = "Middle five seconds emphasize established platform motion"
        elif not cases:
            reason = "Results card shortened from six to three seconds"
        selected_start, selected_end = start + offset, start + offset + count
        assert start <= selected_start < selected_end <= end
        filters.append(
            f"[v{i}]trim=start_frame={selected_start}:end_frame={selected_end},"
            f"setpts=PTS-STARTPTS[s{i}]"
        )
        edits.append(
            {
                "title": chapter["title"],
                "cases": cases,
                "start_s": cursor / FPS,
                "end_s": (cursor + count) / FPS,
                "source_start_frame": selected_start,
                "source_end_frame_exclusive": selected_end,
                "source_start_s": selected_start / FPS,
                "source_end_s": selected_end / FPS,
                "source_chapter_offset_s": offset / FPS,
                "source_chapter_frames": original,
                "frames": count,
                "selection_reason": reason,
            }
        )
        cursor += count
    graph = (
        f"[0:v]split={len(edits)}"
        + "".join(f"[v{i}]" for i in range(len(edits)))
        + ";"
        + ";".join(filters)
        + ";"
        + "".join(f"[s{i}]" for i in range(len(edits)))
        + f"concat=n={len(edits)}:v=1:a=0[out]"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            imageio_ffmpeg.get_ffmpeg_exe(),
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(source),
            "-filter_complex",
            graph,
            "-map",
            "[out]",
            "-an",
            "-c:v",
            "libx264",
            "-crf",
            "18",
            "-preset",
            "medium",
            "-pix_fmt",
            "yuv420p",
            "-r",
            str(FPS),
            "-movflags",
            "+faststart",
            str(output),
        ],
        check=True,
    )
    encode_seconds = time.perf_counter() - started
    assert sha(source) == parent["video_sha256"]
    report = {
        "video_sha256": sha(output),
        "source_video": str(source.relative_to(ROOT)),
        "source_video_sha256": parent["video_sha256"],
        "source_manifest_sha256": sha(source.with_suffix(".json")),
        "checkpoint_sha256": EXPECTED,
        "checkpoint_count": 1,
        "theme": "ember",
        "environment": "industrial",
        "playback_speed": 1,
        "fps": FPS,
        "dimensions": [1920, 1080],
        "frames": cursor,
        "duration_s": cursor / FPS,
        "chapters": edits,
        "source_case_count": len(parent["cases"]),
        "outcomes_refer_to": "Complete original 10/12-second trials, not shortened excerpts",
        "new_training_seconds": 0,
        "new_physics_steps": 0,
        "new_mujoco_render_frames": 0,
        "edit_encode_seconds": encode_seconds,
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "visual_inspection": "pending",
    }
    write(output.with_suffix(".json"), report)
    qa_started = time.perf_counter()
    directory = ROOT / "outputs/locomotion/ember" / output.stem
    directory.mkdir(parents=True, exist_ok=True)
    samples = sorted(
        {
            index
            for c in edits
            for index in (
                round(c["start_s"] * FPS),
                (round(c["start_s"] * FPS) + round(c["end_s"] * FPS) - 1) // 2,
                round(c["end_s"] * FPS) - 1,
            )
        }
    )
    slots = {index: slot for slot, index in enumerate(samples)}
    sheets = [
        Image.new("RGB", (1920, 1140), "#1F1F1F")
        for _ in range((len(samples) + 8) // 9)
    ]
    reader = imageio_ffmpeg.read_frames(str(output), pix_fmt="rgb24")
    metadata = next(reader)
    count = 0
    hero = edits[-3]
    thumbnail = (round(hero["start_s"] * FPS) + round(hero["end_s"] * FPS) - 1) // 2
    for index, raw in enumerate(reader):
        count += 1
        assert len(raw) == 1920 * 1080 * 3
        if index not in slots:
            continue
        frame = Image.fromarray(np.frombuffer(raw, np.uint8).reshape(1080, 1920, 3))
        frame.save(directory / f"frame_{index:04d}.png")
        if index == thumbnail:
            frame.save(output.with_suffix(".png"))
        slot = slots[index]
        x, y = slot % 3 * 640, slot % 9 // 3 * 380
        sheet = sheets[slot // 9]
        sheet.paste(frame.resize((640, 360)), (x, y + 20))
        ImageDraw.Draw(sheet).text(
            (x + 8, y + 4), f"Frame {index} / video {index / FPS:.2f} s", fill="white"
        )
    for i, sheet in enumerate(sheets):
        sheet.save(directory / f"sheet_{i}.png")
    checks = {
        "complete_decode": count == cursor == 2175,
        "half_duration": cursor / FPS == parent["duration_s"] / 2 == 87,
        "each_chapter_halved": all(
            c["frames"] * 2 == c["source_chapter_frames"] for c in edits
        ),
        "all_51_conditions_retained": len([key for c in edits for key in c["cases"]])
        == 51,
        "dimensions": metadata["size"] == (1920, 1080),
        "fps": metadata["fps"] == FPS,
        "duration": abs(metadata["duration"] - 87) < 0.05,
        "one_x_contiguous_source_intervals": all(
            c["source_end_frame_exclusive"] - c["source_start_frame"] == c["frames"]
            for c in edits
        ),
        "source_preserved": sha(source) == parent["video_sha256"],
    }
    write(
        output.with_suffix(".qa.json"),
        {
            "video_sha256": report["video_sha256"],
            "checks": checks,
            "decoded_frames": count,
            "sampled_frames": samples,
            "visual_inspection": "pending",
            "qa_seconds": time.perf_counter() - qa_started,
        },
    )
    print(
        json.dumps(
            {
                "video": str(output),
                "edit_encode_seconds": encode_seconds,
                "checks": checks,
            },
            indent=2,
        )
    )
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=ROOT / "previews/locomotion/ember/adaptive_dog_complete_v2.mp4",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "previews/locomotion/ember/adaptive_dog_short_v1.mp4",
    )
    args = parser.parse_args()
    raise SystemExit(main(args.source.resolve(), args.output.resolve()))
