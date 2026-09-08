"""Views of the physical world, delivered packets and each rover's local belief."""

import bisect
import json
from pathlib import Path

import imageio_ffmpeg
import mujoco
import numpy as np
from PIL import Image, ImageDraw

from sixlegs.recording import font

from .scene import BUILDINGS, COLORS, load_scene

PALETTE = [tuple(int(float(v) * 255) for v in c.split()[:3]) for c in COLORS]


def add_links(scene, positions, events, time):
    for event in events[-120:]:
        age = time - event["time"]
        if not 0 <= age < 0.6 or scene.ngeom >= scene.maxgeom:
            continue
        a, b = positions[event["sender"]].copy(), positions[event["receiver"]].copy()
        a[2] += 0.45
        b[2] += 0.45
        color = (
            [0.1, 0.9, 0.65, 0.7]
            if event["status"] == "delivered"
            else [1.0, 0.3, 0.25, 0.4]
        )
        geom = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(
            geom,
            mujoco.mjtGeom.mjGEOM_LINE,
            np.zeros(3),
            np.zeros(3),
            np.eye(3).ravel(),
            np.array(color, dtype=np.float32),
        )
        mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_LINE, 2.0, a, b)
        scene.ngeom += 1


def preview(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    m, d = load_scene()
    opt = mujoco.MjvOption()
    opt.sitegroup[:] = 0
    with mujoco.Renderer(m, 900, 1600) as renderer:
        for camera in ("overview", "overhead", "detail", "r0_front", "r1_front"):
            renderer.update_scene(d, camera=camera, scene_option=opt)
            Image.fromarray(renderer.render()).save(output / f"{camera}.png")
    print(f"Saved scene previews to {output}")


def dashboard(renderer, small, m, d, snapshot, events, case, report, selected=1):
    t = snapshot["time"]
    frame = Image.new("RGB", (1920, 1080), "#101b24")
    draw = ImageDraw.Draw(frame)
    normal, title = font(21), font(28)
    positions = np.array([d.body(f"r{i}").xpos.copy() for i in range(6)])
    opt = mujoco.MjvOption()
    opt.sitegroup[:] = 0
    renderer.update_scene(d, camera="overview", scene_option=opt)
    add_links(renderer.scene, positions, events, t)
    frame.paste(Image.fromarray(renderer.render()), (0, 64))
    draw.text(
        (18, 12),
        f"ROVER COMMS  /  {case.upper()}  /  {t:05.1f}s",
        font=title,
        fill="#eff6f8",
    )
    draw.text(
        (850, 17),
        f"INSPECTED {len(snapshot['inspected'])}/8",
        font=title,
        fill="#70dfc3",
    )
    for index in range(6):
        small.update_scene(d, camera=f"r{index}_front", scene_option=opt)
        x = (index % 3) * 426
        y = 710 + (index // 3) * 178
        frame.paste(Image.fromarray(small.render()), (x, y))
        role = "SCOUT" if index % 2 == 0 else "INSPECTOR"
        draw.text(
            (x + 12, y - 27),
            f"R{index} FRONT / {role}",
            font=font(19),
            fill=PALETTE[index],
        )
    panel = 1280
    draw.rectangle((panel, 64, 1919, 1079), fill="#132530")
    draw.text(
        (panel + 20, 78), "RADIO LINKS / R1 LOCAL BELIEF", font=normal, fill="#bce7e9"
    )
    draw.text(
        (panel + 20, 106),
        "Rover poses: observer / markers: R1 evidence",
        font=font(17),
        fill="#88a2b1",
    )

    def point(p):
        return (panel + 28 + (p[0] + 10) / 20 * 584, 125 + (8 - p[1]) / 16 * 365)

    for x, y, hx, hy, height in BUILDINGS:
        left, bottom = point((x - hx, y - hy))
        right, top = point((x + hx, y + hy))
        draw.rectangle((left, top, right, bottom), fill="#435560")
    if case == "degraded" and 18 <= t < 76:
        x0, y0 = point((-7, 7))
        x1, y1 = point((7, -7))
        draw.ellipse((x0, y0, x1, y1), outline="#bb693d", width=2)
    for event in events[-120:]:
        if 0 <= t - event["time"] < 0.6:
            color = "#3be2ac" if event["status"] == "delivered" else "#d96b63"
            draw.line(
                (
                    point(positions[event["sender"]]),
                    point(positions[event["receiver"]]),
                ),
                fill=color,
                width=2,
            )
    chosen = snapshot["agents"][selected]
    for marker, b in chosen["belief"].items():
        x, y = point((b["x"], b["y"]))
        age = t - b["observed"]
        color = (
            "#60d8a9"
            if int(marker) in chosen["completed"]
            else ("#e2b558" if age < 15 else "#9b8062")
        )
        draw.rectangle((x - 4, y - 4, x + 4, y + 4), fill=color)
        draw.text((x + 5, y - 9), str(marker), font=font(15), fill=color)
    for i, p in enumerate(positions):
        x, y = point(p)
        color = PALETTE[i] if snapshot["agents"][i]["radio_on"] else "#68717a"
        draw.ellipse((x - 7, y - 7, x + 7, y + 7), fill=color)
        draw.text((x + 8, y - 10), f"R{i}", font=font(17), fill=color)
    draw.text(
        (panel + 20, 497),
        "Packet trails held 0.6s for visibility",
        font=font(17),
        fill="#88a2b1",
    )
    radio = snapshot["radio"]
    counts = radio["counts"]
    draw.text(
        (panel + 20, 532),
        f"DELIVERED {counts.get('delivered', 0)}   LOST {counts.get('lost', 0)}   PDR {radio['delivery_ratio']:.0%}",
        font=normal,
        fill="#e8f2f6",
    )
    for i, a in enumerate(snapshot["agents"]):
        y = 580 + i * 53
        target = "patrol" if a["target"] is None else f"inspect M{a['target']}"
        on = "ON" if a["radio_on"] else "OFF"
        draw.text(
            (panel + 20, y),
            f"R{i} {a['role']:<9}  {target:<12} RADIO {on}",
            font=font(19),
            fill=PALETTE[i],
        )
        draw.text(
            (panel + 45, y + 24),
            f"Known {len(a['belief'])}/8  Done {len(a['completed'])}/8  Queue {a['queue']}",
            font=font(17),
            fill="#a5bdcb",
        )
    recent = [e for e in report["remote_route_changes"] if e["time"] <= t]
    draw.text(
        (panel + 20, 920), "DELIVERY → DECISION EVIDENCE", font=normal, fill="#78dcce"
    )
    if recent:
        event = recent[-1]
        draw.text(
            (panel + 20, 954),
            f"R{event['rover']} changed route to M{event['marker']}",
            font=normal,
            fill="#eff6f8",
        )
        draw.text(
            (panel + 20, 985),
            f"after packet {event['packet']} from {event['source']}",
            font=font(19),
            fill="#b8ccd7",
        )
        draw.text(
            (panel + 20, 1014),
            f"Received {event['received']:.2f}s → decision {event['time']:.2f}s",
            font=font(19),
            fill="#b8ccd7",
        )
    else:
        draw.text(
            (panel + 20, 957),
            "No remote report has changed a route.",
            font=font(19),
            fill="#b8ccd7",
        )
    transport = (
        "Real Reticulum 1.5.2 over simulated LoRa"
        if report["transport"] == "reticulum-plain"
        else "Simulated LoRa packets"
    )
    draw.text(
        (18, 1058),
        f"MuJoCo wheel physics  |  {transport}  |  Local sensing + delivered evidence",
        font=font(17),
        fill="#b8ccd7",
    )
    return frame


def record(run_dir, output, speed=2.0, fps=15):
    directory = Path(run_dir)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    trajectory = np.load(directory / "trajectory.npz")
    telemetry = json.loads((directory / "telemetry.json").read_text())
    report = json.loads((directory / "report.json").read_text())
    snapshots = telemetry["snapshots"]
    st = [s["time"] for s in snapshots]
    events = telemetry["packets"]
    et = [e["time"] for e in events]
    m, d = load_scene()
    final = float(trajectory["time"][-1])
    times = np.r_[np.arange(0, final, speed / fps), [final] * (3 * fps)]
    writer = imageio_ffmpeg.write_frames(
        str(output),
        (1920, 1080),
        fps=fps,
        codec="libx264",
        quality=7,
        macro_block_size=2,
        output_params=["-movflags", "+faststart", "-threads", "4"],
    )
    writer.send(None)
    with (
        mujoco.Renderer(m, 600, 1280) as renderer,
        mujoco.Renderer(m, 150, 426) as small,
    ):
        try:
            for count, t in enumerate(times):
                i = max(0, bisect.bisect_right(trajectory["time"], t) - 1)
                d.qpos[:] = trajectory["qpos"][i]
                mujoco.mj_forward(m, d)
                si = max(0, bisect.bisect_right(st, t) - 1)
                ei = bisect.bisect_right(et, t)
                frame = dashboard(
                    renderer,
                    small,
                    m,
                    d,
                    snapshots[si],
                    events[max(0, ei - 120) : ei],
                    report["case"],
                    report,
                )
                ImageDraw.Draw(frame).text(
                    (1100, 22), f"{speed:g}x replay", font=font(19), fill="#b8ccd7"
                )
                writer.send(np.asarray(frame))
                if count == int(12 * fps / speed):
                    frame.save(output.with_suffix(".png"))
                if count % 100 == 0:
                    print(f"{count}/{len(times)} video frames", flush=True)
        finally:
            writer.close()
    print(f"Saved {output}")
