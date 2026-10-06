"""Observer video of a recorded robot run. Replays saved joint states only.

Every panel shows the same simulation timestamp. Pixels are observer output;
the controller never reads them.
"""

import hashlib
import json
from pathlib import Path
import time

import imageio_ffmpeg
import mujoco
import numpy as np
from PIL import Image, ImageDraw

from .motion import JOINT_NAMES, TARGETS
from .scene import load_scene
from .visuals import font, options

W, H = 960, 540
FPS = 30
BG, INK, MUTED, ACCENT, WARN = "#101c25", "#eef4f5", "#9fb4bf", "#43c8b5", "#e2a33b"
PHASE_TEXT = {"transit": "TRANSIT ABOVE TARGET", "descend": "DESCEND TO 1 mm HOVER", "hover": "HOVER / HOLD",
              "raise": "RAISE TO 5 mm TRANSIT HEIGHT"}


def follow_camera(tip):
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = tip+(0, 0, .004)
    cam.distance, cam.azimuth, cam.elevation = .055, 118, -18
    return cam


def caption(image, title, detail=None):
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, W, 40), fill=BG)
    draw.rectangle((14, 11, 18, 29), fill=ACCENT)
    draw.text((28, 9), title, font=font(20), fill=INK)
    if detail:
        draw.text((W-14-draw.textlength(detail, font=font(15)), 13), detail, font=font(15), fill=MUTED)
    return image


def telemetry(trace, i, reference, sites):
    panel = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(panel)
    t = trace["time"][i]
    segment, moving = reference.phase(t)
    name, target = reference.labels[segment]
    phase = "hover" if name == "descend" and not moving else name
    tip = trace["tip"][i]
    lateral = np.linalg.norm(tip[:2]-sites[target][:2])*1e6
    height = (tip[2]-sites[target][2])*1e3
    draw.text((28, 22), "SURGICAL INSERTION ROBOT  /  TARGET APPROACH", font=font(24), fill=INK)
    draw.text((28, 58), "Programmed servo  /  MuJoCo 1 ms implicitfast  /  real time 1.0x", font=font(16), fill=MUTED)
    draw.text((28, 104), f"t = {t:6.3f} s", font=font(34), fill=INK)
    draw.text((300, 112), PHASE_TEXT[phase], font=font(22), fill=ACCENT if phase == "hover" else WARN)
    draw.text((28, 160), f"TARGET {target}", font=font(22), fill=INK)
    for k in TARGETS:
        x = 180+k*44
        done = k < target or (k == target and phase == "hover")
        draw.ellipse((x, 163, x+22, 185), outline=ACCENT, width=2, fill=ACCENT if done else None)
    rows = [("Tip offset from target, lateral", f"{lateral:10.1f} µm"),
            ("Tip height above target", f"{height:10.3f} mm"),
            ("Tracking error vs reference", f"{np.linalg.norm(tip-trace['tip_ref'][i])*1e6:10.2f} µm"),
            ("Gantry speed", f"{np.linalg.norm(trace['qvel'][i][:3])*1e3:10.1f} mm/s")]
    for k, (label, value) in enumerate(rows):
        y = 214+k*40
        draw.text((28, y), label, font=font(19), fill=MUTED)
        draw.text((620, y), value, font=font(21), fill=INK)
    draw.text((28, 382), "Actuator force / limit", font=font(17), fill=MUTED)
    limits = np.array(reference.force_limits)
    for k, joint in enumerate(JOINT_NAMES):
        y = 412+k*22
        frac = min(abs(trace["force"][i][k])/limits[k], 1)
        draw.text((28, y-2), joint, font=font(15), fill=MUTED)
        draw.rectangle((160, y, 160+600, y+12), outline="#2a3d48")
        draw.rectangle((160, y, 160+int(600*frac), y+12), fill=ACCENT)
        draw.text((775, y-3), f"{100*frac:4.1f}%", font=font(15), fill=INK)
    return panel


def results_card(report):
    card = Image.new("RGB", (1920, 1080), BG)
    draw = ImageDraw.Draw(card)
    tour = report["tour"]
    draw.text((90, 80), "RESULTS  /  SIX-TARGET APPROACH TOUR", font=font(46), fill=INK)
    draw.text((90, 150), "Measured in simulation  /  programmed servo, not a learned policy  /  no thread or insertion yet",
              font=font(24), fill=MUTED)
    rows = [("Hold, 2 s", f"tip drift {report['hold']['tip_drift_m']*1e6:.3f} µm (gate < 1 µm)"),
            ("Hover error after 0.3 s, all targets",
             f"lateral ≤ {max(h['max_lateral_error_m'] for h in tour['targets'])*1e6:.2f} µm, "
             f"vertical ≤ {max(h['max_vertical_error_m'] for h in tour['targets'])*1e6:.2f} µm (gate < 10 µm)"),
            ("Tracking while moving", f"RMS {tour['tracking_rms_while_moving_m']*1e6:.2f} µm, max {tour['tracking_max_while_moving_m']*1e6:.2f} µm"),
            ("Robot–environment contacts", f"{tour['robot_environment_contacts']}"),
            ("Closest needle–phantom gap", f"{tour['min_needle_phantom_gap_m']*1e3:.3f} mm"),
            ("Peak actuator force", f"{100*max(tour['max_force_fraction'].values()):.1f}% of limit"),
            ("Gates", "PASSED" if report["passed"] else "FAILED")]
    for k, (label, value) in enumerate(rows):
        y = 270+k*92
        draw.text((90, y), label, font=font(30), fill=MUTED)
        draw.text((820, y), value, font=font(32), fill=ACCENT if k == len(rows)-1 and report["passed"] else INK)
    return card


def record(report, trace, reference, output, card_s=3.):
    started = time.perf_counter()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    video = output/"approach_tour.mp4"
    model, data = load_scene()
    qadr = [model.jnt_qposadr[model.joint(n).id] for n in JOINT_NAMES]
    sites = {k: data.site(f"target_{k}").xpos.copy() for k in TARGETS}
    times = np.arange(0, trace["time"][-1]+1e-9, 1/FPS)
    writer = imageio_ffmpeg.write_frames(str(video), (1920, 1080), fps=FPS, codec="libx264", quality=8,
                                         macro_block_size=2, output_params=["-movflags", "+faststart"])
    writer.send(None)
    stills = {}
    marks = {"opening": .0, "first_hover": None, "transit": None, "final_hover": None}
    with mujoco.Renderer(model, H, W) as renderer:
        opt = options()
        for f, t in enumerate(times):
            i = int(np.argmin(abs(trace["time"]-t)))
            data.qpos[qadr] = trace["qpos"][i]  # replay of a recorded state
            mujoco.mj_forward(model, data)
            frame = Image.new("RGB", (1920, 1080), BG)
            renderer.update_scene(data, camera="overview", scene_option=opt)
            frame.paste(caption(Image.fromarray(renderer.render()), "OVERVIEW", "fixed world camera"), (0, 0))
            renderer.update_scene(data, camera=follow_camera(data.site("needle_tip").xpos.copy()), scene_option=opt)
            frame.paste(caption(Image.fromarray(renderer.render()), "NEEDLE CLOSE-UP", "observer camera following the tip"), (W, 0))
            renderer.update_scene(data, camera="microscope", scene_option=opt)
            frame.paste(caption(Image.fromarray(renderer.render()), "MOUNTED MICROSCOPE", "tool-mounted camera, observer only"), (0, H))
            frame.paste(telemetry(trace, i, reference, sites), (W, H))
            writer.send(np.asarray(frame))
            segment, moving = reference.phase(t)
            name, target = reference.labels[segment]
            if marks["first_hover"] is None and name == "descend" and not moving:
                marks["first_hover"] = f
            if marks["transit"] is None and name == "transit" and target == 3:
                marks["transit"] = f
            if f in (0, marks["first_hover"], marks["transit"]) or f == len(times)-1:
                stills[f] = frame
        card = results_card(report)
        for _ in range(int(card_s*FPS)):
            writer.send(np.asarray(card))
    writer.close()
    for f, frame in stills.items():
        frame.save(output/f"frame_{f:04d}.png")
    card.save(output/"results_card.png")
    receipt = {"video": video.name, "video_sha256": hashlib.sha256(video.read_bytes()).hexdigest(),
               "fps": FPS, "frames": len(times)+int(card_s*FPS), "action_frames": len(times),
               "simulated_s": float(trace["time"][-1]), "playback": "1.0x real time",
               "render_wall_s": time.perf_counter()-started, "additional_simulation_s": 0,
               "stills": sorted(f"frame_{f:04d}.png" for f in stills)}
    (output/"video_manifest.json").write_text(json.dumps(receipt, indent=2)+"\n")
    return receipt
