"""Render the workcell parts glossary: one view per part, highlighted against the ghosted machine.

Uses the end-to-end scene (tube variant) at a recorded state with the eyelet
clamped on the needle above the cassette, so tool parts are shown doing their
job. Writes previews/neural_insertion/glossary_v1/*.png and glossary.json (the
captions the journal renders). Replay of a recorded state; no simulation.

Usage: uv run --locked python scripts/render_glossary.py [--state CHECKPOINT]
"""

import argparse
import json
from pathlib import Path
import pickle
import sys

import mujoco
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))

from sixlegs.neural_insertion.e2e_scene import load_e2e  # noqa: E402

OUT = ROOT/"previews/neural_insertion/glossary_v1"
STATE = ROOT/"outputs/neural_insertion/e2e/run_v6/checkpoints/05_keeper.pkl"  # latch closed on a picked eyelet
W, H = 1200, 800
GHOST = .22

# id, title, role, specs, geom-name prefixes, camera (azimuth, elevation, distance scale or fixed mm)
PARTS = [
    ("workcell", "The workcell", "A five-axis gantry robot that picks a thread, carries it to a tissue phantom and inserts it.",
     "Deck 658 × 448 mm. Simulated in MuJoCo; this view is the end-to-end scene.", None, (135, -20, 980)),
    ("base", "Base and deck", "The rigid table everything mounts to. Disturbance models treat it as an isolation table that can still vibrate.",
     "Nominal table vibration 1 mm/s² per component, 5–60 Hz.", ("plinth", "deck", "front_trim", "foot_"), (135, -28, 1.2)),
    ("gantry", "Gantry frame", "Two towers and a beam carry the X rail high above the work area.",
     "Beam centre 465 mm above the deck.", ("tower_", "beam", "x_rail_", "rail_bolt_"), (135, -18, 1.3)),
    ("stage_x", "X stage", "Carriage running along the beam: moves the whole tool left and right.",
     "Travel ±175 mm · moving mass 2.4 kg · force limit 100 N.", ("x_carriage", "x_face", "y_bearing"), (140, -20, 3.0)),
    ("stage_y", "Y stage", "Boom hanging from the X carriage: moves the tool toward and away from the viewer.",
     "Travel ±75 mm · 1.3 kg · 80 N.", ("y_boom", "y_rail_", "z_backbone", "z_rail_"), (140, -18, 2.6)),
    ("stage_z", "Z stage (tool head)", "Raises and lowers the whole tool head; it lifts the 44 mm thread clear during transport.",
     "Travel −8 to +50 mm · 0.65 kg · 40 N.", ("tool_head", "head_face", "head_led", "tool_socket"), (140, -15, 3.0)),
    ("microscope", "Microscope", "Optical head beside the needle with an oblique view of the work point. In these "
     "experiments its camera is for observers only; the controllers do not use pixels.",
     "Camera field of view 23°.", ("scope_",), (60, -10, 3.0)),
    ("carriage", "Needle carriage", "Fine vertical slide that drives the needle; it supplies the last millimetres down to the "
     "tissue and the insertion stroke.", "Travel 0–18 mm · 25 g · 2 N.", ("needle_drive", "needle_shank", "needle_taper"),
     (140, -12, 3.0)),
    ("needle", "Needle with slotted tip", "A 150 µm needle. Two small lips near the tip form a side slot that captures the "
     "thread's eyelet rim: lifting carries it up, inserting pushes it down, sliding sideways releases it.",
     "Ø 150 µm · lips 40 µm wire reaching 200 µm · slot opening 140 µm.", ("=needle", "slot_lip_"), (215, -12, ("tip", 1.5))),
    ("keeper", "Keeper (latch)", "Rides on the needle carriage. During transport a single latch bar lowers just past "
     "the ends of the slot's lips, closing the slot into an eye: the eyelet can swing but cannot leave, as on a knitting "
     "machine's latch needle. It lifts away before insertion. Earlier clamp and cage designs lost the eyelet.",
     "Travel 0–6 mm · latch bar 40 µm wire, 10 µm past the lip ends · flexure guided.", ("keeper_",),
     (235, -28, ("tip", 1.3))),
    ("thread", "Thread", "The implant: a flexible 40 µm thread, carried in by the needle rather than pushed. It is "
     "simulated as a discrete elastic rod (a published model for MuJoCo, arXiv 2310.00911, ported with a corrected "
     "force projection): 32 rigid segments joined by ball joints, 99 degrees of freedom, with bending and twisting "
     "forces from the rod's elastic energy. Its light, firm contacts settle in microseconds, so the whole simulation "
     "steps at 5 µs (200 kHz, RK4, four force evaluations per step); in the full workcell that costs about 3 ms per "
     "step, roughly 600 times slower than real time on one CPU core.",
     "Ø 40 µm · 44 mm · 32 segments · 5 µs step (200 kHz) · illustrative 100 MPa. Shown up close: at machine scale "
     "it is thinner than a pixel.", ("thread_G",), (150, -30, ("tip", 4.5))),
    ("eyelet", "Eyelet", "A small rigid ring on the thread's free end, the handle the needle picks up and pushes into "
     "tissue.", "Ring radius 180 µm · wire Ø 40 µm.", ("eyelet_",), (200, -35, ("tip", 1.2))),
    ("cassette", "Thread cassette", "Presents threads for pickup. A shallow trench under each eyelet lets the needle's "
     "lower lip pass beneath the rim.", "Trench 0.2 mm wide, 0.6 mm deep.", ("cassette_", "bed_", "presentation_rail_",
                                                                                "presented_thread_", "pickup_loop_",
                                                                                "thread_tab_"), (120, -40, 1.4)),
    ("support", "Specimen support", "Tray and clamps that hold the tissue phantom still on the deck.", "Tray 114 × 94 mm.",
     ("specimen_", "clamp_"), (135, -30, 1.3)),
    ("phantom", "Tissue phantom", "A domed block standing in for tissue. Its surface is a real collision surface for the "
     "robot and the thread.", "92 × 72 mm, dome 3 mm high.", ("tissue_phantom",), (135, -30, 1.4)),
    ("targets", "Target sites", "Six marked insertion sites. Alignment learns to bring the needle 1 mm above any of "
     "them within 10 µm.", "Six sites, ring markers 2.5 mm across (markings, no collision).", ("target_",),
     (135, -40, 1.6)),
    ("vessels", "Vessels", "Blood-vessel markings on the surface. The robot must not pass low over them; they are "
     "markings, scored by the task, not obstacles.", "Main vessel Ø 1 mm, branches Ø 0.56 mm.", ("vessel_",),
     (135, -40, 1.4)),
    ("tube", "Insertion tube", "A pre-formed tube at the target with real collision: it guides the needle and holds the "
     "thread by contact. Tissue puncture is not modelled yet; inside the tube the needle meets an axial cutting and "
     "friction force.",
     "Bore radius 0.28 mm · 3 mm deep · skin around the opening.", ("tube_",), (200, -14, 1.5)),
]


def restore(model, data, path):
    blob = pickle.loads(Path(path).read_bytes())
    mujoco.mj_setState(model, data, blob["state"], mujoco.mjtState.mjSTATE_INTEGRATION)
    mujoco.mj_forward(model, data)


def part_geoms(model, prefixes):
    if prefixes is None:
        return list(range(model.ngeom))
    names = [model.geom(g).name for g in range(model.ngeom)]
    exact = {p[1:] for p in prefixes if p.startswith("=")}
    starts = tuple(p for p in prefixes if not p.startswith("="))
    return [g for g, n in enumerate(names) if n in exact or (starts and n.startswith(starts))]


def frame(model, data, geoms, azimuth, elevation, scale):
    if isinstance(scale, tuple):  # close-up centred just above the needle tip, fixed distance in mm
        center, distance = data.site("needle_tip").xpos+np.array((0, 0, .2)), scale[1]
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
    parser.add_argument("--state", type=Path, default=STATE)
    args = parser.parse_args()
    model, data, meta = load_e2e(tubes=True)
    restore(model, data, args.state)
    OUT.mkdir(parents=True, exist_ok=True)
    base_rgba = model.geom_rgba.copy()
    # Lift the tissue-coloured tube slightly so it reads against the phantom.
    renderer = mujoco.Renderer(model, H, W)
    entries = []
    for i, (pid, title, role, specs, prefixes, (az, el, scale)) in enumerate(PARTS):
        geoms = part_geoms(model, prefixes)
        model.geom_rgba[:] = base_rgba
        if prefixes is not None:
            mask = np.ones(model.ngeom, bool)
            mask[geoms] = False
            model.geom_rgba[mask, 3] = base_rgba[mask, 3]*GHOST
            model.geom_rgba[model.geom("floor").id, 3] = 1.  # keep the ground solid
            if pid == "tube":
                model.geom_rgba[model.geom("tissue_phantom").id, 3] = .06
        renderer.update_scene(data, frame(model, data, geoms, az, el, scale))
        name = f"{i:02d}_{pid}.png"
        Image.fromarray(renderer.render()).save(OUT/name)
        entries.append({"id": pid, "title": title, "role": role, "specs": specs, "image": name,
                        "geoms": len(geoms)})
        print(name, len(geoms), "geoms", flush=True)
    model.geom_rgba[:] = base_rgba
    (OUT/"glossary.json").write_text(json.dumps({
        "source": "end-to-end scene (tube variant), recorded state with the eyelet clamped above the cassette",
        "scene_sha256": meta["xml_sha256"], "parts": entries}, indent=1)+"\n")


if __name__ == "__main__":
    main()
