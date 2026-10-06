"""Physics-only copy of the workcell for parallel learning environments.

Removes only the zero-mass, non-colliding marking geoms (vessels, target rings,
cassette thread drawings) and caps the per-world arena. Target sites, collision
geometry, bodies, joints, actuators and options are unchanged; a test checks
that the servo tour produces the same joint trajectory as the full scene.
Vessel centerlines are exported separately for the learning observation.
"""

import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from .scene import FIELD_Y, ROOT, SCENE, SURFACE_Z, build_scene, surface_z

# Mirrors scene.surface_z; a test checks the two agree.
DOME_DEPTH_M, DOME_AX_M, DOME_AY_M = .003, .046, .036

RL_SCENE = ROOT/"build/neural_insertion/rl/surgical_align.xml"
ARENA = "2M"


def vessel_polylines(model, data):
    """World-frame vessel centerlines and radii from the full scene's marking capsules."""
    lines = {}
    for g in range(model.ngeom):
        name = model.geom(g).name
        if not name.startswith("vessel_"):
            continue
        vessel = int(name.split("_")[1])
        a, b = data.geom_xpos[g]-data.geom_xmat[g].reshape(3, 3)[:, 2]*model.geom_size[g, 1], \
            data.geom_xpos[g]+data.geom_xmat[g].reshape(3, 3)[:, 2]*model.geom_size[g, 1]
        lines.setdefault(vessel, {"radius_m": float(model.geom_size[g, 0]), "segments": []})["segments"].append(
            [a.tolist(), b.tolist()])
    return lines


def build_rl_scene(path=RL_SCENE):
    full = build_scene(SCENE)
    root = ET.parse(full).getroot()
    root.set("model", "surgical_align_physics")
    removed = 0
    for parent in root.iter():
        for child in list(parent):
            if child.tag == "geom" and child.get("class") == "marking":
                parent.remove(child)
                removed += 1
    size = root.find("size")
    if size is None:
        size = ET.SubElement(root, "size")
    size.set("memory", ARENA)
    model = mujoco.MjModel.from_xml_path(str(full))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    vessels = vessel_polylines(model, data)
    # Task data travels inside the model: per vessel segment (x0, y0, x1, y1, radius),
    # and the analytic dome (surface z, field y, depth, x and y semi-axes).
    # MuJoCo 3.12 aborted while parsing one 920-value numeric, so each segment is
    # stored separately as vessel_<index> = (x0, y0, x1, y1, radius).
    custom = ET.SubElement(root, "custom")
    # Round to the picometre so every platform writes byte-identical model files
    # (compiled transforms differ across math libraries by ~1e-18 m).
    segments = [tuple(round(v, 12)+0. for v in (a[0], a[1], b[0], b[1], line["radius_m"]))
                for line in vessels.values() for a, b in line["segments"]]
    for i, values in enumerate(segments):
        ET.SubElement(custom, "numeric", name=f"vessel_{i:03d}", size="5",
                      data=" ".join(format(v, ".17g") for v in values))
    ET.SubElement(custom, "numeric", name="surface_dome", size="5",
                  data=" ".join(format(v, ".17g") for v in (SURFACE_Z, FIELD_Y, DOME_DEPTH_M, DOME_AX_M, DOME_AY_M)))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(path, encoding="unicode")
    meta = {"removed_marking_geoms": removed, "arena": ARENA, "field_y_m": FIELD_Y,
            "targets_m": [data.site(f"target_{k}").xpos.tolist() for k in range(6)],
            "vessels": vessels}
    (path.parent/"surgical_align_meta.json").write_text(json.dumps(meta, indent=1)+"\n")
    return path, meta


def surface_height(x, y):
    """Analytic dome height used to build the phantom mesh (valid over the phantom)."""
    return surface_z(x, y)
