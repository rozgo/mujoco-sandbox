"""Render the workcell parts gallery: one view per part, highlighted against the ghosted machine.

Uses the thread-tube workcell at a recorded moment of a nominal run (level 1, seed 1001): the thread stuck
to the needle partway into the tissue, so tool parts are shown doing their job. Writes
previews/neural_insertion/glossary_v2/*.png and glossary.json (the captions the journal renders). Replay of
a recorded state; no simulation.

Usage: uv run --locked python scripts/render_glossary.py [--run DIR] [--time S]
"""

import argparse
import json
from pathlib import Path
import sys

import mujoco
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))

from sixlegs.neural_insertion.der import plugin  # noqa: E402

OUT = ROOT/"previews/neural_insertion/glossary_v2"
RUN = ROOT/"outputs/neural_insertion/site_tube/L1_s1001"
TIME = 0.0745  # s: thread stuck to the needle, the point just through the tissue at site 0
W, H = 1200, 800
GHOST = .22

# id, title, role, specs, geom-name prefixes, camera (azimuth, elevation, distance scale or ("tip", mm))
PARTS = [
    ("workcell", "The workcell", "A gantry robot that carries a needle and a thread tube to a tissue phantom and "
     "inserts threads, one site after another.", "Deck 658 × 448 mm. Simulated in MuJoCo 3.12.", None, (135, -20, 980)),
    ("base", "Base and deck", "The rigid table everything mounts to. The disturbance model treats it as an "
     "isolation table that still vibrates a little.", "Nominal table vibration up to 1 mm/s² per component, 5–60 Hz.",
     ("plinth", "deck", "front_trim", "foot"), (135, -28, 1.2)),
    ("gantry", "Gantry frame", "Two towers and a beam carry the X rail high above the work area.",
     "Beam centre 465 mm above the deck.", ("tower", "beam", "x_rail", "rail_bolt"), (135, -18, 1.3)),
    ("stage_x", "X stage", "Carriage running along the beam: moves the whole tool left and right.",
     "Travel ±175 mm · moving mass 2.4 kg · force limit 100 N.", ("x_carriage", "x_face", "y_bearing"), (140, -20, 3.0)),
    ("stage_y", "Y stage", "Boom hanging from the X carriage: moves the tool toward and away from the viewer.",
     "Travel ±75 mm · 1.3 kg · 80 N.", ("y_boom", "y_rail", "z_backbone", "z_rail"), (140, -18, 2.6)),
    ("stage_z", "Z stage (tool head)", "Raises and lowers the tool head with the thread tube; it carries the tool "
     "between sites and lifts it clear of the threads left standing.", "Travel −8 to +50 mm · 0.65 kg · 40 N.",
     ("tool_head", "head_face", "head_led", "tool_socket"), (140, -15, 3.0)),
    ("microscope", "Microscope", "Optical head beside the needle with an oblique view of the work point. In these "
     "experiments its camera is for observers only; the controllers do not use pixels.", "Camera field of view 23°.",
     ("scope_",), (60, -10, 3.0)),
    ("carriage", "Needle carriage", "Fast vertical slide that drives the needle: it strokes the needle onto the "
     "waiting thread end and on into the tissue, then snaps back.", "Travel 0–18 mm · 25 g · 2 N · strokes at up to "
     "about 200 mm/s.", ("needle_drive", "needle_shank", "needle_taper"), (140, -12, 3.0)),
    ("needle", "Needle", "A conical point, a thin tip section and a stiffer shaft. When its point reaches the "
     "thread's end the two stick (a stand-in for a chemical bond), and the needle carries the thread into the "
     "tissue through a puncture.", "Tip section Ø 50 µm · shaft Ø 100 µm.", ("=needle", "needle_point", "needle_shaft"),
     (215, -12, ("tip", 2.4))),
    ("tube", "Thread tube", "Holds the next thread on an arm beside the needle, angled 15° from vertical, with "
     "the thread's end on the needle's path. It moves with the tool head, not with the needle's stroke.",
     "Bore Ø 200 µm · 9 mm long.", ("tube_glass", "tube_cap", "arm_reach", "arm_upright"), (150, -16, ("tip", 6.0))),
    ("thread", "Thread", "The implant: a flexible 40 µm thread, drawn in lime. It is simulated as a discrete "
     "elastic rod (a published model for MuJoCo, arXiv 2310.00911, ported with a corrected force projection): 18 "
     "segments with bending and twisting stiffness, stepped at 50 µs with added joint inertia and damping so it "
     "runs fast.", "Ø 40 µm · 8.25 mm · 18 segments · illustrative 100 MPa.", ("thread_G",), (150, -24, ("tip", 3.0))),
    ("support", "Specimen support", "Tray and clamps that hold the tissue phantom on the deck.", "Tray 114 × 94 mm.",
     ("specimen_", "clamp"), (135, -30, 1.3)),
    ("phantom", "Tissue phantom", "A domed block standing in for tissue, with a needle-tissue force model: "
     "dimpling to puncture at 2 mN, then cutting and shaft friction, and grip on the thread inside. It moves "
     "with breathing and pulse in the disturbed runs.", "92 × 72 mm, dome 3 mm high.", ("tissue_phantom",),
     (135, -30, 1.4)),
    ("targets", "Target sites", "Six marked insertion sites. A three-site run places threads at sites 0, 5 and 1.",
     "Ring markers 2.5 mm across (markings, no collision).", ("target",), (135, -40, 1.6)),
    ("vessels", "Vessels", "Blood-vessel markings on the surface. They are markings for the task to avoid, not "
     "obstacles.", "Main vessel Ø 1 mm, branches Ø 0.56 mm.", ("vessel",), (135, -40, 1.4)),
]


def part_geoms(model, prefixes):
    if prefixes is None:
        return list(range(model.ngeom))
    names = [model.geom(g).name for g in range(model.ngeom)]
    exact = {p[1:] for p in prefixes if p.startswith("=")}
    starts = tuple(p for p in prefixes if not p.startswith("="))
    return [g for g, n in enumerate(names) if n in exact or (starts and n.startswith(starts))]


def frame(model, data, geoms, azimuth, elevation, scale):
    if isinstance(scale, tuple):  # close-up centred just above the needle point, fixed distance in mm
        center, distance = data.site("needle_tip").xpos+np.array((0, 0, .4)), scale[1]
    elif len(geoms) == model.ngeom:
        center, distance = np.array((-40., -40., 180.)), scale
    else:
        lo = np.min([data.geom_xpos[g]-model.geom_rbound[g] for g in geoms], 0)
        hi = np.max([data.geom_xpos[g]+model.geom_rbound[g] for g in geoms], 0)
        center, extent = (lo+hi)/2, float(np.max(hi-lo))
        distance = extent*scale+.5
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:], cam.distance, cam.azimuth, cam.elevation = center, distance, azimuth, elevation
    return cam


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=RUN)
    parser.add_argument("--time", type=float, default=TIME)
    args = parser.parse_args()
    plugin()
    model = mujoco.MjModel.from_xml_path(str(args.run/"scene.xml"))
    data = mujoco.MjData(model)
    trace = dict(np.load(args.run/"trace.npz", allow_pickle=True))
    k = int(np.searchsorted(trace["time"], args.time))
    data.qpos[:] = trace["qpos"][k]  # replay of a recorded state
    spec = model.body("specimen_support").id
    data.mocap_pos[model.body_mocapid[spec]] = model.body_pos[spec]+trace["tissue_offset"][k]
    mujoco.mj_forward(model, data)
    OUT.mkdir(parents=True, exist_ok=True)
    base_rgba = model.geom_rgba.copy()
    renderer = mujoco.Renderer(model, H, W)
    hidden = [g for g in range(model.ngeom) if model.geom(g).name.startswith("thread_G")
              and model.geom(g).name.split("_")[-1] != "0"]  # spare threads, parked off to the side
    entries = []
    for i, (pid, title, role, specs, prefixes, (az, el, scale)) in enumerate(PARTS):
        geoms = part_geoms(model, prefixes)
        model.geom_rgba[:] = base_rgba
        model.geom_rgba[hidden, 3] = 0
        if prefixes is not None:
            mask = np.ones(model.ngeom, bool)
            mask[geoms] = False
            model.geom_rgba[mask, 3] = base_rgba[mask, 3]*GHOST
            model.geom_rgba[model.geom("floor").id, 3] = 1.  # keep the ground solid
            model.geom_rgba[hidden, 3] = 0
            if pid in ("needle", "thread", "tube"):
                model.geom_rgba[model.geom("tissue_phantom").id, 3] = .10
            if pid == "tube":  # the glass is faint on its own; draw it more opaque for this view
                model.geom_rgba[model.geom("tube_glass").id] = (.62, .78, .98, .72)
        renderer.update_scene(data, frame(model, data, geoms, az, el, scale))
        name = f"{i:02d}_{pid}.png"
        Image.fromarray(renderer.render()).save(OUT/name)
        entries.append({"id": pid, "title": title, "role": role, "specs": specs, "image": name, "geoms": len(geoms)})
        print(name, len(geoms), "geoms", flush=True)
    model.geom_rgba[:] = base_rgba
    (OUT/"glossary.json").write_text(json.dumps({
        "source": f"thread-tube workcell, recorded state at t = {args.time} s of {args.run.name} (thread stuck to "
                  "the needle, entering the tissue)", "parts": entries}, indent=1)+"\n")


if __name__ == "__main__":
    main()
