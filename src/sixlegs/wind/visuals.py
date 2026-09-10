"""Physical paired replay, numerical wind tracers, and forecast evidence."""

import json
from itertools import pairwise
from pathlib import Path

import imageio_ffmpeg
import matplotlib
import mujoco
import numpy as np
from PIL import Image, ImageDraw

from sixlegs.recording import font

from .control import reference
from .flow import DOMAIN, FORECAST_DT
from .scene import load_scene
from .weather import Weather

AMBER = "#f4b659"
TEAL = "#65e2d2"
INK = "#10202b"
TEXT = "#e9f2f5"


def follow(d, close=False):
    cam = mujoco.MjvCamera()
    cam.lookat[:] = (
        d.body("payload").xpos if close else reference(d.time)[0] + [0, 0, -0.35]
    )
    cam.distance = 0.95 if close else 2.65
    cam.azimuth = 100 if close else 125
    cam.elevation = -12 if close else -18
    return cam


def line(scene, a, b, color=(0.3, 0.9, 0.9, 0.5), width=1.0, arrow=False):
    if scene.ngeom >= scene.maxgeom:
        return
    geom = scene.geoms[scene.ngeom]
    kind = mujoco.mjtGeom.mjGEOM_ARROW if arrow else mujoco.mjtGeom.mjGEOM_LINE
    mujoco.mjv_initGeom(
        geom,
        kind,
        np.zeros(3),
        np.zeros(3),
        np.eye(3).ravel(),
        np.array(color, dtype=np.float32),
    )
    mujoco.mjv_connector(geom, kind, width, np.asarray(a), np.asarray(b))
    scene.ngeom += 1


class Tracers:
    def __init__(self, weather):
        self.weather = weather
        self.time = 0.0
        rng = np.random.default_rng(44)
        self.initial = np.column_stack(
            (
                rng.uniform(-6, 6, 420),
                rng.uniform(-6, 6, 420),
                rng.uniform(0.8, 2.45, 420),
            )
        )
        self.points = self.initial.copy()

    def at(self, t):
        if t < self.time:
            self.time = 0.0
            self.points[:] = self.initial
        while self.time < t - 1e-9:
            dt = min(0.05, t - self.time)
            v = self.weather.at(self.time, self.points[:, :2])
            self.points[:, :2] = (
                self.points[:, :2] + dt * v + DOMAIN / 2
            ) % DOMAIN - DOMAIN / 2
            self.time += dt
        return self.points, self.weather.at(t, self.points[:, :2])


def add_flow(scene, weather, t):
    if not hasattr(weather, "tracers"):
        weather.tracers = Tracers(weather)
    p, v = weather.tracers.at(t)
    for position, wind in zip(p, v):
        if abs(position[0]) > 4.1 or abs(position[1]) > 2.2:
            continue
        speed = np.linalg.norm(wind)
        blend = min(speed / 6.0, 1.0)
        color = (0.18 + 0.70 * blend, 0.68 + 0.12 * blend, 0.95 - 0.50 * blend, 0.32)
        line(scene, position - np.r_[wind, 0.0] * 0.10, position, color, 1.3)


def add_evidence(scene, d, snapshot, color):
    base = d.body("drone").xpos.copy()
    payload = d.body("payload").xpos.copy()
    forces = np.array(snapshot["wind_force_n"])
    for p, f in ((base, forces[:4].sum(axis=0)), (payload, forces[4])):
        if np.linalg.norm(f) > 0.08:
            line(scene, p, p + 0.14 * f, (*color, 0.9), 0.012, True)
    if not snapshot["released"]:
        predicted = np.array(snapshot["predicted_load"])
        points = np.column_stack((predicted, np.full(len(predicted), payload[2])))
        for a, b in pairwise(points):
            line(scene, a, b, (*color, 0.7), 2.0)


def vorticity(field):
    n = field.shape[-1]
    kx = (2 * np.pi * np.fft.rfftfreq(n, d=DOMAIN / n))[None, :]
    ky = (2 * np.pi * np.fft.fftfreq(n, d=DOMAIN / n))[:, None]
    return np.fft.irfft2(
        1j * kx * np.fft.rfft2(field[1]) - 1j * ky * np.fft.rfft2(field[0]), s=(n, n)
    )


def heatmap(array, size=190, error=False):
    cmap = matplotlib.colormaps["magma" if error else "RdBu_r"]
    scaled = np.clip(array / 0.8, 0, 1) if error else np.clip(0.5 + array / 8, 0, 1)
    rgb = (cmap(scaled)[..., :3] * 255).astype(np.uint8)
    return Image.fromarray(rgb[::-1]).resize((size, size), Image.Resampling.BILINEAR)


def flow_panel(frame, weather, t):
    draw = ImageDraw.Draw(frame)
    draw.text((20, 690), "WIND FORECAST / ONE SECOND AHEAD", font=font(20), fill=TEAL)
    obs = min(int((t + 1e-8) / FORECAST_DT), len(weather.predictions) - 1)
    lead = 4
    future = obs * FORECAST_DT + 1.0
    truth = weather.field(future)
    prediction = weather.predictions[obs, lead]
    arrays = [
        vorticity(truth),
        vorticity(prediction),
        np.sqrt(np.mean((truth - prediction) ** 2, axis=0)),
    ]
    for i, (array, label) in enumerate(
        zip(arrays, ("REFERENCE", "PINO", "VELOCITY ERROR"))
    ):
        x = 18 + i * 210
        draw.text((x, 725), label, font=font(17), fill=TEXT)
        frame.paste(heatmap(array, error=i == 2), (x, 756))
    rmse = np.sqrt(np.mean((truth - prediction) ** 2))
    draw.text(
        (20, 967),
        f"128 × 128 field · forecast RMSE {rmse:.3f} m/s",
        font=font(19),
        fill="#b5cad5",
    )
    draw.text(
        (20, 1000),
        "Reference future shown for evaluation only",
        font=font(17),
        fill="#88a7b8",
    )


def metrics_panel(frame, snapshots, trajectories, i, reports):
    draw = ImageDraw.Draw(frame)
    left, top = 1310, 735
    width, height = 580, 200
    draw.text((1300, 690), "PAYLOAD TRACKING ERROR", font=font(21), fill=TEXT)
    last = max(i, 1)
    peak = max(
        np.max(
            np.linalg.norm(
                trajectories[k]["qpos"][:, 7:9]
                - np.array([s["reference"][:2] for s in snapshots[k]]),
                axis=1,
            )
        )
        * 100
        for k in reports
    )
    limit = max(15.0, np.ceil(peak / 5) * 5)
    draw.line(
        (left, top, left, top + height, left + width, top + height),
        fill="#516b7a",
        width=2,
    )
    for k, color in (("persistence", AMBER), ("pino", TEAL)):
        tr = trajectories[k]
        ss = snapshots[k]
        ids = np.arange(0, min(last, len(tr["time"])), 5)
        vals = np.array(
            [
                np.linalg.norm(tr["qpos"][j, 7:9] - np.array(ss[j]["reference"])[:2])
                * 100
                for j in ids
            ]
        )
        if len(ids) > 1:
            points = [
                (
                    left + float(tr["time"][j]) / 43 * width,
                    top + height - height * min(v / limit, 1.0),
                )
                for j, v in zip(ids, vals)
            ]
            draw.line(points, fill=color, width=3)
    draw.text((left - 5, top - 23), f"{limit:g} cm", font=font(16), fill="#8fabba")
    draw.text(
        (left + width - 40, top + height + 5), "43 s", font=font(16), fill="#8fabba"
    )
    for j, (k, color, label) in enumerate(
        (("persistence", AMBER, "Frozen wind"), ("pino", TEAL, "PINO forecast"))
    ):
        r = reports[k]
        draw.text(
            (1300, 974 + j * 34),
            f"{label}: {100 * r['payload_tracking_rmse_m']:.1f} cm crossing RMSE",
            font=font(20),
            fill=color,
        )


def dashboard(
    main, small, models, datas, snapshots, trajectories, index, weather, reports
):
    frame = Image.new("RGB", (1920, 1080), INK)
    draw = ImageDraw.Draw(frame)
    t = snapshots["pino"][index]["time"]
    draw.text((22, 16), "LEARNING THE WIND", font=font(31), fill=TEXT)
    draw.text(
        (1160, 22),
        f"Identical reference wind + actuator limits  |  {t:04.1f}s",
        font=font(22),
        fill="#b5cad5",
    )
    opt = mujoco.MjvOption()
    opt.sitegroup[4] = 0
    for j, (kind, color, rgb) in enumerate(
        (("persistence", AMBER, (0.96, 0.71, 0.35)), ("pino", TEAL, (0.40, 0.89, 0.82)))
    ):
        d = datas[kind]
        m = models[kind]
        ss = snapshots[kind][index]
        d.time = t
        d.qpos[:] = trajectories[kind]["qpos"][index]
        m.tendon_width[0] = 0 if ss["released"] else 0.003
        mujoco.mj_forward(m, d)
        renderer = main[kind]
        renderer.update_scene(d, camera=follow(d), scene_option=opt)
        add_flow(renderer.scene, weather, t)
        add_evidence(renderer.scene, d, ss, rgb)
        frame.paste(Image.fromarray(renderer.render()), (j * 960, 108))
        label = (
            "FROZEN-FIELD FORECAST"
            if kind == "persistence"
            else "LEARNED PINO FORECAST"
        )
        draw.text((22 + j * 960, 70), label, font=font(24), fill=color)
        status = ss["phase"]
        swing = ss["swing_deg"]
        draw.rectangle((j * 960 + 15, 599, j * 960 + 945, 640), fill=INK)
        draw.text(
            (j * 960 + 27, 607),
            f"{status}   |   Package swing {swing:.1f}°",
            font=font(23),
            fill=color,
        )
    draw.line((960, 66, 960, 657), fill="#33505c", width=2)
    draw.text(
        (22, 656),
        "Wind tracers follow the numerical field · arrows show applied forces · paths show controller predictions",
        font=font(19),
        fill="#9db9c8",
    )
    flow_panel(frame, weather, t)
    if t < 5:
        camera, camera_label = "overview", "COURSE OVERVIEW / PINO"
    elif t < 12:
        camera, camera_label = "loadcam", "DOWNWARD PAYLOAD CAMERA / PINO"
    elif t < 20:
        camera, camera_label = "front", "FORWARD ONBOARD CAMERA / PINO"
    elif t < 26:
        camera, camera_label = "overhead", "OVERHEAD CAMERA / PINO"
    else:
        camera, camera_label = follow(datas["pino"], True), "PAYLOAD CLOSE-UP / PINO"
    small.update_scene(
        datas["pino"],
        camera=camera,
        scene_option=opt,
    )
    frame.paste(Image.fromarray(small.render()), (650, 737))
    draw.text(
        (660, 690),
        camera_label,
        font=font(21),
        fill=TEAL,
    )
    if t >= 26:
        draw.text(
            (660, 1004),
            "Physical cable tension, contact and release",
            font=font(18),
            fill="#b5cad5",
        )
    metrics_panel(frame, snapshots, trajectories, index, reports)
    draw.text(
        (20, 1054),
        "MuJoCo rigid bodies + 2D background flow · 64² training data + 128² physics loss · Full flight shown at real time",
        font=font(18),
        fill="#8aa8b8",
    )
    return frame


def result_card(reports, comparison, validation):
    frame = Image.new("RGB", (1920, 1080), INK)
    d = ImageDraw.Draw(frame)
    d.text((90, 80), "PREDICT THE FIELD. TEST THE FLIGHT.", font=font(49), fill=TEXT)
    d.text(
        (92, 154),
        "Independent Navier–Stokes wind drives every MuJoCo evaluation.",
        font=font(28),
        fill="#b9d0dd",
    )
    d.text(
        (92, 207),
        "Mean payload tracking RMSE · crossing t = 6–32 s · lower is better",
        font=font(24),
        fill="#8aa8b8",
    )
    grouped = {
        k: [r for r in comparison if r["forecast"] == k]
        for k in ("persistence", "fno", "pino", "oracle")
    }
    maxerr = max(
        np.mean([r["payload_tracking_rmse_m"] for r in group])
        for group in grouped.values()
    )
    colors = [AMBER, "#91b3e7", TEAL, "#b49cd7"]
    for j, ((kind, group), color, label) in enumerate(
        zip(
            grouped.items(),
            colors,
            (
                "Frozen field",
                "FNO · data only",
                "PINO · data + physics",
                "True future · diagnostic",
            ),
        )
    ):
        error = np.mean([r["payload_tracking_rmse_m"] for r in group]) * 100
        y = 280 + j * 110
        d.text((92, y), label, font=font(29), fill=color)
        d.rounded_rectangle(
            (640, y + 2, 640 + 850 * error / (maxerr * 100), y + 40),
            radius=6,
            fill=color,
        )
        d.text((1570, y), f"{error:.2f} cm", font=font(32), fill=color)
    n = len(grouped["pino"])
    success = sum(r["success"] for r in grouped["pino"])
    reduction = 1 - np.mean(
        [r["payload_tracking_rmse_m"] for r in grouped["pino"]]
    ) / np.mean([r["payload_tracking_rmse_m"] for r in grouped["persistence"]])
    d.text(
        (92, 765),
        f"{n} held-out wind cases   |   PINO deliveries {success}/{n}   |   Tracking RMSE {reduction:.0%} lower",
        font=font(34),
        fill=TEAL,
    )
    if validation:
        rows = [
            r
            for r in validation["rows"]
            if r["model"] == "pino" and r["resolution"] == 128 and r["lead_s"] == 2
        ]
        rmse = np.mean([r["velocity_rmse_m_s"] for r in rows])
        d.text(
            (92, 843),
            f"Two-second flow forecast RMSE: {rmse:.3f} m/s on independent test trajectories.",
            font=font(27),
            fill="#b9d0dd",
        )
    d.text(
        (92, 927),
        "One-way flow coupling, approximate drag, idealized rotors; no resolved rotor wash.",
        font=font(24),
        fill="#8aa8b8",
    )
    d.text(
        (92, 970),
        "The learned forecast helps control. MuJoCo still supplies rigid-body and contact dynamics.",
        font=font(24),
        fill="#8aa8b8",
    )
    return frame


def record(run_dir, weather_dir, output, seed=300, fps=25):
    run_dir = Path(run_dir)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    kinds = ("persistence", "pino")
    models = {}
    datas = {}
    trajectories = {}
    snapshots = {}
    reports = {}
    for kind in kinds:
        folder = run_dir / f"{seed}-{kind}"
        models[kind], datas[kind] = load_scene()
        trajectories[kind] = np.load(folder / "trajectory.npz")
        snapshots[kind] = json.loads((folder / "telemetry.json").read_text())
        reports[kind] = json.loads((folder / "report.json").read_text())
        if not reports[kind]["success"]:
            raise ValueError(f"{kind} delivery did not complete")
    weather = Weather(weather_dir, "pino")
    for kind in kinds:
        if reports[kind].get("weather") != weather.provenance:
            raise ValueError(
                "Flight and weather provenance differ. Re-run wind-demo compare "
                "with the selected model checkpoints before recording."
            )
    comparison = json.loads((run_dir / "comparison.json").read_text())
    vp = Path(__file__).parents[3] / "docs/wind/operator_validation.json"
    validation = json.loads(vp.read_text()) if vp.exists() else None
    count = len(snapshots["pino"])
    frames = [0] * (4 * fps) + list(range(count)) + [count - 1] * (5 * fps)
    writer = imageio_ffmpeg.write_frames(
        str(output),
        (1920, 1080),
        fps=fps,
        codec="libx264",
        quality=8,
        macro_block_size=2,
        output_params=["-movflags", "+faststart", "-threads", "4"],
    )
    writer.send(None)
    saved = set()
    with (
        mujoco.Renderer(models["pino"], 540, 960) as learned,
        mujoco.Renderer(models["persistence"], 540, 960) as baseline,
        mujoco.Renderer(models["pino"], 252, 630) as small,
    ):
        main = {"pino": learned, "persistence": baseline}
        try:
            for n, i in enumerate(frames):
                frame = dashboard(
                    main,
                    small,
                    models,
                    datas,
                    snapshots,
                    trajectories,
                    i,
                    weather,
                    reports,
                )
                writer.send(np.asarray(frame))
                phase = snapshots["pino"][i]["phase"]
                if phase not in saved:
                    frame.save(
                        output.parent / f"frame-{phase.lower().replace(' ', '-')}.png"
                    )
                    saved.add(phase)
                if i == 500:
                    frame.save(output.with_suffix(".png"))
                if n % 100 == 0:
                    print(f"{n}/{len(frames) + 10 * fps} frames", flush=True)
            card = result_card(reports, comparison, validation)
            card.save(output.parent / "results.png")
            for _ in range(10 * fps):
                writer.send(np.asarray(card))
        finally:
            writer.close()
    info = {
        "weather": weather.provenance,
        "fps": fps,
        "frames": len(frames) + 10 * fps,
        "duration_s": (len(frames) + 10 * fps) / fps,
        "flight_speed": 1.0,
        "seed": seed,
        "resolution": [1920, 1080],
    }
    output.with_suffix(".json").write_text(json.dumps(info, indent=2) + "\n")
    return info
