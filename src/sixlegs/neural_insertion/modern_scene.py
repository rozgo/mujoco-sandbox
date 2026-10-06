"""End-to-end scene, design v2 (modern interpretation; docs/neural_insertion/MODERN_DESIGN.md).

Built like the first-pass variant (SI workcell, then millimetre-gram engine
units, RK4 at 5 µs, task-regime contact law, still air), with a different
thread supply and tool:

- An implant carrier 3 mm from target 0 anchors the thread's root (a clamped
  discrete elastic rod, 6 segments, 8.25 mm). The thread lies on a backing film
  1 mm above the tissue running away from the target, its loop presented over
  the film's far end. The robot picks the loop and brings it back toward the
  implant, so the thread always has slack (a straight thread lifted at its far
  end goes taut; run_v5).
- The needle has a Ø 50 µm tip section that passes through the loop and a
  ledge above it that presses the loop's far edge: advancing pushes the loop
  into tissue, retracting slides out of it.
- A rotary pincher on the needle carriage swings its jaw in at loop height and
  pins the loop's neck against the needle with a torque-limited force.
- No insertion tube: the tissue is the phantom plus the needle-tissue force
  model of tissue.py (puncture, cutting, shaft friction, thread grip).
The retainer slide is kept as an empty, unused axis so the shared servo and
joint layout are unchanged.
"""

import math
from dataclasses import replace
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from .der import der_xml
from .e2e_scene import (AIR_DENSITY, AIR_VISCOSITY, DT_S, MATERIAL, TIME_CONSTANT_S, UNITS, _find, _parent_map,
                        _scale_si, _theme, _vec)
from .scene import FIELD_Y, ROOT, add, build_scene, surface_z
from .task_fixtures import fixture_config, law_for

MODERN_SCENE = ROOT/"build/neural_insertion/e2e/modern_v1.xml"

# Needle (SI, relative to the insertion body; the tip is at TIP_Z).
TIP_Z = -0.083075
TIP_R, SHAFT_R = 25e-6, 40e-6
TIP_LENGTH = .6e-3
LEDGE_Z, LEDGE_R, LEDGE_REACH = .30e-3, 12e-6, -.17e-3  # ledge height above the tip, radius, reach (toward -X,
                                                        # the loop's far edge, away from the anchor)
# Thread and loop.
SEGMENTS, SEGMENT = 6, 1.375e-3
FREE_LENGTH = SEGMENTS*SEGMENT
LOOP_R, WIRE = 100e-6, 20e-6
INSERTED_SEGMENTS = 4      # loop-side segments that may follow the needle through the puncture track
# Presentation, world SI: the anchor 3 mm -X of target 0, the thread running -X from it, film top 1 mm
# above the tissue at the target.
TARGET0 = np.array((0., FIELD_Y))
ANCHOR_X = TARGET0[0]-3e-3
LOOP_CENTER_XY = np.array((ANCHOR_X-FREE_LENGTH-LOOP_R, TARGET0[1]))
FILM_TOP = surface_z(*TARGET0)+1e-3
FILM_END_X = LOOP_CENTER_XY[0]+LOOP_R+WIRE+450e-6    # loop and neck overhang the film's far end by 0.45 mm
# Pincher (engine units, mm, relative to the insertion body), defined closed (hinge angle 0).
LOOP_ABOVE_TIP = (LEDGE_Z-LEDGE_R-WIRE)*1e3          # loop centre height above the tip when engaged
# The jaw swings in from the side (-Y), across the thread's direction, and pins the ring's near wire
# against the needle; the thread leaves the ring toward -X, out of the jaw's path.
# Closed jaw 15 µm below the ring wire's centre as it actually sits under the ledge (0.24 mm above the
# tip, measured in run_v4): the jaw rises as it closes, so it pushes the wire inward and up, locking
# the ring between jaw, needle and ledge. A jaw above the centre pressed it down and off (run_v4).
JAW_HEIGHT = .225
JAW_CLOSED = np.array((0., -(TIP_R+2*WIRE+15e-6)*1e3, TIP_Z*1e3+JAW_HEIGHT))
HINGE = np.array((0., -1.5, TIP_Z*1e3+4.0))
PINCH_FORCE_UN = 100.                                 # µN on the thread (the validated press load)
PINCHER_OPEN = .0858                                  # rad: jaw 0.32 mm out (-Y) and 0.13 mm down


def _needle(root):
    needle = _find(root, "body", "insertion")
    parents = _parent_map(root)
    old = _find(root, "geom", "needle")
    parents[old].remove(old)
    tip = TIP_Z
    add(needle, "geom", name="needle_shaft", type="capsule", size=_vec((SHAFT_R,)), conaffinity="1",
        fromto=_vec((0, 0, -.065, 0, 0, tip+TIP_LENGTH)), material="steel")
    add(needle, "geom", name="needle", type="capsule", size=_vec((TIP_R,)), conaffinity="1",
        fromto=_vec((0, 0, tip+TIP_LENGTH, 0, 0, tip+TIP_R)), material="steel")
    add(needle, "geom", name="needle_ledge", type="capsule", size=_vec((LEDGE_R,)), conaffinity="1",
        fromto=_vec((0, 0, tip+LEDGE_Z, LEDGE_REACH, 0, tip+LEDGE_Z)), material="steel")
    # Empty the retainer slide (kept so the joint layout and servo are unchanged).
    retainer = _find(root, "body", "retainer")
    for g in list(retainer.iter("geom")):
        parents[g].remove(g)


def _presentation(root):
    """Implant carrier, backing film. World SI; static."""
    world = root.find("worldbody")
    y = TARGET0[1]
    anchor_x = ANCHOR_X
    film = (FILM_END_X, anchor_x+.3e-3)
    t = 40e-6
    add(world, "geom", name="backing_film", type="box", material="tray", contype="1", conaffinity="1",
        size=_vec(((film[1]-film[0])/2, .3e-3, t/2)), pos=_vec(((film[0]+film[1])/2, y, FILM_TOP-t/2)))
    base = surface_z(film[0], y)
    add(world, "geom", name="film_support", type="box", material="graphite", contype="1", conaffinity="1",
        size=_vec(((film[1]-film[0])/2-.2e-3, .25e-3, (FILM_TOP-t-base)/2)),
        pos=_vec(((film[0]+film[1])/2, y, (FILM_TOP-t+base)/2)))
    base = surface_z(anchor_x, y)
    add(world, "geom", name="implant_carrier", type="box", material="shell", contype="1", conaffinity="1",
        size=_vec((.8e-3, 1.0e-3, (FILM_TOP+.4e-3-base)/2)),
        pos=_vec((anchor_x+.8e-3, y, (FILM_TOP+.4e-3+base)/2)))
    return np.array((anchor_x, y, FILM_TOP+WIRE))


def _thread(root, units, config, anchor):
    rod = ET.fromstring(der_xml(config, units))
    L = units.length
    root.insert(1, rod.find("extension"))
    world = root.find("worldbody")
    first = rod.find("worldbody").find("body")
    first.set("pos", _vec(anchor*L))  # clamped root: no joint on the first body
    first.set("quat", "0 0 0 1")       # rest shape along -X, away from the target
    world.append(first)
    bodies = [b for b in first.iter("body") if b.get("name", "").startswith("thread_B")]
    for i, b in enumerate(bodies):
        b.find("geom").set("conaffinity", "1" if i >= len(bodies)-INSERTED_SEGMENTS else "3")
    last = bodies[-1]
    seg = config.length_m/config.segments
    center = np.array((seg+LOOP_R, 0, 0))
    for k in range(12):
        a0, a1 = 2*math.pi*k/12, 2*math.pi*(k+1)/12
        p0 = center+LOOP_R*np.array((math.cos(a0), math.sin(a0), 0))
        p1 = center+LOOP_R*np.array((math.cos(a1), math.sin(a1), 0))
        add(last, "geom", name=f"eyelet_{k}", type="capsule", size=_vec((WIRE*L,)),
            fromto=_vec(np.concatenate((p0, p1))*L), conaffinity="1", condim="3",
            friction=_vec((config.friction, 1e-6*L, 1e-7*L)), mass="0")
    add(last, "site", name="eyelet_center", pos=_vec(center*L), size=_vec((WIRE*L,)))
    contact = root.find("contact")
    if contact is None:
        contact = add(root, "contact")
    for e in rod.find("contact"):
        contact.append(e)
    return [b.get("name") for b in bodies]


def _pincher(root):
    """Rotary pincher on the needle carriage, engine units (mm, g). Defined closed; opens at +angle."""
    needle = _find(root, "body", "insertion")
    body = add(needle, "body", name="pincher", pos=_vec(HINGE))
    add(body, "joint", name="pincher", type="hinge", axis="-1 0 0", range=_vec((-.02, .12)), damping="2e-4")
    jaw = JAW_CLOSED-HINGE
    elbow = jaw+np.array((0., -.02, .05))
    add(body, "inertial", pos=_vec(elbow/2), mass="2e-3", diaginertia=_vec((2e-3*4., 2e-3*4., 1e-5)))
    add(body, "geom", name="pincher_arm", type="capsule", size=".03", fromto=_vec((0, 0, 0, *elbow)),
        material="teal", conaffinity="1")
    add(body, "geom", name="pincher_jaw", type="capsule", size=".015", conaffinity="1", material="teal",
        fromto=_vec((jaw[0]-.06, jaw[1], jaw[2], jaw[0]+.06, jaw[1], jaw[2])))
    # Torque limit sets the pinch: the jaw's moment arm about the hinge for a sideways push is its height.
    torque = PINCH_FORCE_UN*abs(jaw[2])
    add(root.find("actuator"), "position", name="pincher", joint="pincher", kp="5e4", kv="40",
        ctrlrange=_vec((-.02, .12)), forcerange=_vec((-torque, torque)))


def build_modern_scene(path=MODERN_SCENE, units=UNITS):
    si = build_scene(Path(path).with_suffix(".si.xml"))
    root = ET.parse(si).getroot()
    root.set("model", "surgical_insertion_modern_v1")
    for e in list(root):
        if e.tag == "keyframe":
            root.remove(e)
    root.find("default").find("geom").set("conaffinity", "3")
    _find(root, "geom", "tissue_phantom").set("contype", "2")
    _find(root, "geom", "tissue_phantom").set("conaffinity", "2")
    _needle(root)
    anchor = _presentation(root)
    _scale_si(root, units)
    law = law_for(TIME_CONSTANT_S)
    config = replace(fixture_config("drag", DT_S, MATERIAL, law), length_m=FREE_LENGTH, segments=SEGMENTS)
    names = _thread(root, units, config, anchor)
    _pincher(root)
    _theme(root)
    option = root.find("option")
    for k, v in {"timestep": format(DT_S, "g"), "integrator": "RK4", "solver": "Newton", "tolerance": "1e-12",
                 "iterations": "100", "cone": "elliptic", "jacobian": "sparse",
                 "density": format(AIR_DENSITY*units.mass/units.length**3, ".6g"),
                 "viscosity": format(AIR_VISCOSITY*units.mass/units.length, ".6g")}.items():
        option.set(k, v)
    solimp, solref = _vec(law.solimp(units)), _vec(law.solref())
    for geom in root.iter("geom"):
        geom.set("solimp", solimp)
        geom.set("solref", solref)
    # The pincher rides on the needle carriage, and MuJoCo skips parent-child contacts; explicit pairs
    # let the jaw meet the needle, so it pinches the thread against it instead of passing through.
    contact = root.find("contact")
    for other in ("needle", "needle_ledge", "needle_shaft"):
        add(contact, "pair", geom1="pincher_jaw", geom2=other, condim="3", solref=solref, solimp=solimp,
            friction=_vec((.3, .3, 1e-3, 1e-4, 1e-4)))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(root, space="  ")
    ET.ElementTree(root).write(path, encoding="unicode")
    return path, {"thread_bodies": names, "config": config, "law": law, "anchor_m": anchor.tolist()}


def load_modern(path=MODERN_SCENE):
    import hashlib
    from .der import plugin
    plugin()
    p, meta = build_modern_scene(path)
    meta["xml_sha256"] = hashlib.sha256(Path(p).read_bytes()).hexdigest()
    meta["variant"] = "modern"
    meta["tube"] = None
    model = mujoco.MjModel.from_xml_path(str(p))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data, meta
