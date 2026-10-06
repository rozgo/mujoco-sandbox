"""End-to-end scene, design v3 (modern interpretation; docs/neural_insertion/MODERN_DESIGN.md).

Built like the first-pass variant (SI workcell, then millimetre-gram engine
units, RK4 at 5 µs, task-regime contact law, still air), with a different
thread supply and tool. Every moving part is a body with mass, a joint or a
weld to the part that carries it, and collision geometry:

- Implant carrier 1.5 mm ahead (+X) and 3 mm beside (-Y) target 0, anchoring
  the thread's root (a clamped discrete elastic rod, 6 segments, 8.25 mm). The
  thread lies on a backing film 1 mm above the tissue, running -X past the
  target; its loop overhangs the film's far end. The robot picks the loop and
  carries it toward the implant to the target, so the thread bows with slack
  and never folds back over itself. (With the target on the far side of the
  implant, the fold spun the loop half a turn on the needle and its neck pried
  the latch open: cycle_v4.)
- Cartridge on the tool head: a bracket from the tool socket holds a cannula
  around the needle. The needle (insertion carriage) slides inside it and alone
  strokes into tissue; the cannula stays above the surface.
- Needle: a conical point, a Ø 50 µm tip section and a Ø 100 µm shaft whose end
  is a full shoulder 0.6 mm above the point. The loop is a rigid ring fitted to
  the tip section (8-10 µm radial play) that cannot pass the shaft: the shoulder
  pushes it into tissue, and the fit keeps it on the needle's axis.
- Latch: a finger hinged on a clevis on the cannula wall (+Y, away from the
  thread's side), ending in a fork whose
  two prongs straddle the tip section just under the seated loop. A preloaded
  torsion spring holds it closed against a stop, trapping the loop between the
  fork and the shoulder without squeezing the thread; a pull wire (tendon) from
  the cartridge opens it against the spring to a second stop. (A side jaw that
  pinched the loop against the needle with 100 µN let the peel pull the loop
  down past it: friction is not a lock. cycle_v3, first run.)
- No insertion tube: the tissue is the phantom plus the needle-tissue force
  model of tissue.py (puncture, cutting, shaft friction, thread grip).
The base workcell's thread cassette and retainer slide belong to the first
design and are removed.
"""

import hashlib
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

MODERN_SCENE = ROOT/"build/neural_insertion/e2e/modern_v3.xml"
FAST_SCENE = ROOT/"build/neural_insertion/e2e/modern_v3_fast.xml"
# Fast thread: coarser settings, as game engines treat ropes. 50 µs step instead of 5 µs; contacts settle in
# 200 µs instead of 20 µs; solver tolerance 1e-8; rotational inertia (armature) of 1e-17 kg m² on each thread
# joint slows the thread's twisting (alone it would need steps near 10 µs) while adding under 3% to bending.
FAST = {"dt": 50e-6, "contact_s": 2e-4, "tolerance": "1e-8", "iterations": "30", "armature_kg_m2": 1e-17}
JOINTS = ("stage_x", "stage_y", "stage_z", "insertion")
STEEL = 7.9e-3                 # g/mm³

# Engine units below (mm, g, s; force in µN, torque in µN·mm) unless marked SI.
# Needle, relative to the insertion body; the point is the needle_tip site.
TIP = -83.075
POINT_LENGTH = .20             # conical lead-in from the point to the tip section
TIP_R = .025                   # tip section, Ø 50 µm
SEAT = .60                     # shoulder (shaft end) above the point
SHAFT_R = .050                 # shaft, Ø 100 µm
SHAFT_TOP = -65.               # joins the base scene's needle taper
# Loop: rigid ring on the thread's free end, fitted to the tip section.
LOOP_R, WIRE = .055, .020      # centreline radius, wire radius
RING_SEGMENTS = 12
SEATED = SEAT+SHAFT_R-math.sqrt((SHAFT_R+WIRE)**2-(LOOP_R*math.cos(math.pi/RING_SEGMENTS))**2)  # loop centre
                                                                     # above the point with the loop on the shoulder
# Kinematics: approach and transport on the Z stage with the needle at Q_PICK on the insertion carriage;
# the carriage strokes beyond it only to insert, and retracts into the cannula.
Q_PICK = 6.0
STROKE_MAX = 4.0
# Cartridge, in a frame welded to the Z stage at the insertion body's zero position.
CANNULA_R = .10                # Ø 200 µm tube (bore Ø 120 µm), drawn and collided as a solid rod
CANNULA_END = TIP-Q_PICK+.70   # 0.7 mm above the point at Q_PICK; covers the seated loop from above
HUB_Z = SHAFT_TOP-.35-Q_PICK-STROKE_MAX-1.5  # clear of the taper at the full stroke
STRUT_X = 8.
SOCKET_BOTTOM = 7.             # tool socket's lower face in this frame
# Latch (defined closed, angle 0, on its stop; opens outward at positive angles). It sits on the +Y side,
# opposite the implant, so the loop's neck points away from it at pickup and at the target.
SIDE = 1.
PRONG_R = .012
PRONG_X = TIP_R+PRONG_R+.004                         # prongs straddle the tip section with 4 µm clearance
LOOP_Z = TIP-Q_PICK+SEATED                           # seated loop centre at Q_PICK
PRONG_Z = LOOP_Z-WIRE-.004-PRONG_R                   # prong tops 4 µm under the seated loop's wire
PRONG_Y = (SIDE*.12, -SIDE*.06)                     # prongs run under the whole ring width when closed
HINGE = np.array((0., SIDE*(CANNULA_R+.04), CANNULA_END+.80))
ARM = HINGE[2]-PRONG_Z
SPRING_REF = -.30              # rad: spring preload against the closed stop
SPRING_K = 100.*ARM/(0-SPRING_REF)  # 100 µN at the prongs' lever arm: about 0.5 mN of downward pull on the
                                    # loop pries the latch (moment arm about 0.18 mm), far above thread loads
OPEN_STOP = .30                # rad: prong tips about 0.21 mm out, clear of the ring
TAB = np.array((0., SIDE*.12, .10))  # wire lever from the hinge (outboard and up: tension opens)
WIRE_OPEN = 3000.              # µN of wire tension: holds the finger on its open stop (2 mN only balanced
                               # the spring short of it, at 0.27 rad: cycle_v5)
WIRE_MAX = 3500.
# Presentation, world SI: the anchor 1.5 mm +X and 3 mm -Y of target 0, the thread running -X from it,
# film top 1 mm above the tissue at the anchor. The film and carrier stay clear of the vessel markings.
SEGMENTS, SEGMENT = 6, 1.375e-3
FREE_LENGTH = SEGMENTS*SEGMENT
INSERTED_SEGMENTS = 4          # loop-side segments that may follow the needle through the puncture track
TARGET0 = np.array((0., FIELD_Y))
ANCHOR_X, ANCHOR_Y = TARGET0[0]+1.5e-3, TARGET0[1]-3e-3
LOOP_CENTER_XY = np.array((ANCHOR_X-FREE_LENGTH-LOOP_R*1e-3, ANCHOR_Y))
FILM_TOP = surface_z(ANCHOR_X, ANCHOR_Y)+1e-3
FILM_END_X = LOOP_CENTER_XY[0]+(LOOP_R+WIRE)*1e-3+450e-6  # loop and neck overhang the film's far end by 0.45 mm


def _remove_first_design(root):
    """Drop the base workcell's cassette, retainer slide and their camera and actuator."""
    parents = _parent_map(root)
    for tag, name in (("body", "thread_cassette"), ("body", "retainer"), ("camera", "thread_fixture")):
        e = _find(root, tag, name)
        parents[e].remove(e)
    actuators = root.find("actuator")
    for a in list(actuators):
        if a.get("name") == "retainer":
            actuators.remove(a)


def _presentation(root):
    """Implant carrier, backing film. World SI; static."""
    world = root.find("worldbody")
    y = ANCHOR_Y
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
    return np.array((anchor_x, y, FILM_TOP+WIRE*1e-3))


def _thread(root, units, config, anchor, armature=0.):
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
        if armature and b.find("joint") is not None:
            b.find("joint").set("armature", repr(armature*units.mass*units.length**2))
    last = bodies[-1]
    seg = config.length_m*L/config.segments
    center = np.array((seg+LOOP_R, 0, 0))
    for k in range(RING_SEGMENTS):
        a0, a1 = 2*math.pi*k/RING_SEGMENTS, 2*math.pi*(k+1)/RING_SEGMENTS
        p0 = center+LOOP_R*np.array((math.cos(a0), math.sin(a0), 0))
        p1 = center+LOOP_R*np.array((math.cos(a1), math.sin(a1), 0))
        add(last, "geom", name=f"eyelet_{k}", type="capsule", size=_vec((WIRE,)),
            fromto=_vec(np.concatenate((p0, p1))), conaffinity="1", condim="3",
            friction=_vec((config.friction, 1e-6*L, 1e-7*L)), mass="0")
    add(last, "site", name="eyelet_center", pos=_vec(center), size=_vec((WIRE,)))
    # The ring's mass (same material as the thread) joins the last segment's inertial.
    inertial = last.find("inertial")
    ms, xs = float(inertial.get("mass")), float(inertial.get("pos").split()[0])
    axial, transverse = sorted(float(v) for v in inertial.get("diaginertia").split())[::2]
    mr = config.density_kg_m3*units.mass/L**3*2*math.pi*LOOP_R*math.pi*WIRE**2
    m = ms+mr
    x = (ms*xs+mr*center[0])/m
    shift = ms*(xs-x)**2+mr*(center[0]-x)**2
    inertial.attrib.pop("quat", None)
    inertial.set("mass", repr(m))
    inertial.set("pos", _vec((x, 0, 0)))
    inertial.set("diaginertia", _vec((axial+mr*LOOP_R**2/2, transverse+mr*LOOP_R**2/2+shift,
                                      transverse+mr*LOOP_R**2+shift)))
    contact = root.find("contact")
    if contact is None:
        contact = add(root, "contact")
    for e in rod.find("contact"):
        contact.append(e)
    return [b.get("name") for b in bodies], mr


def _needle(root):
    """Conical point, tip section and shouldered shaft on the insertion carriage (engine units)."""
    needle = _find(root, "body", "insertion")
    parents = _parent_map(root)
    old = _find(root, "geom", "needle")
    parents[old].remove(old)
    ring = [(TIP_R*math.cos(a), TIP_R*math.sin(a), POINT_LENGTH) for a in np.linspace(0, 2*math.pi, 16,
                                                                                         endpoint=False)]
    add(root.find("asset"), "mesh", name="needle_point", vertex=" ".join(_vec(p) for p in [(0, 0, 0), *ring]))
    add(needle, "geom", name="needle_point", type="mesh", mesh="needle_point", pos=_vec((0, 0, TIP)),
        material="steel", conaffinity="1", mass="0")
    add(needle, "geom", name="needle", type="capsule", size=_vec((TIP_R,)), conaffinity="1", material="steel",
        fromto=_vec((0, 0, TIP+POINT_LENGTH, 0, 0, TIP+SEAT+SHAFT_R)), mass="0")
    add(needle, "geom", name="needle_shaft", type="capsule", size=_vec((SHAFT_R,)), conaffinity="1",
        fromto=_vec((0, 0, SHAFT_TOP, 0, 0, TIP+SEAT+SHAFT_R)), material="steel", mass="0")


def _cartridge(root, solref, solimp):
    """Bracket, hub, cannula and clevis welded to the Z stage; the spring-closed, wire-opened latch."""
    zbody = _find(root, "body", "stage_z")
    origin = _find(root, "body", "insertion").get("pos")
    cart = add(zbody, "body", name="cartridge", pos=origin)
    tube = STEEL*math.pi*(CANNULA_R**2-.06**2)*(HUB_Z-CANNULA_END)
    add(cart, "geom", name="cannula", type="cylinder", material="steel", mass=repr(float(tube)),
        fromto=_vec((0, 0, CANNULA_END, 0, 0, HUB_Z)), size=_vec((CANNULA_R,)))
    add(cart, "geom", name="cartridge_hub", type="cylinder", material="graphite", density=repr(STEEL),
        fromto=_vec((0, 0, HUB_Z-.5, 0, 0, HUB_Z+.5)), size=".5")
    add(cart, "geom", name="cartridge_arm", type="capsule", material="graphite", density=repr(STEEL),
        fromto=_vec((0, 0, HUB_Z, STRUT_X, 0, HUB_Z)), size=".4")
    add(cart, "geom", name="cartridge_strut", type="capsule", material="graphite", density=repr(STEEL),
        fromto=_vec((STRUT_X, 0, HUB_Z, STRUT_X, 0, SOCKET_BOTTOM+1.)), size=".75")
    add(cart, "geom", name="latch_clevis", type="box", material="steel", density=repr(STEEL),
        size=_vec((.05, .03, .06)), pos=_vec((0, SIDE*(CANNULA_R+.015), HINGE[2])))
    add(cart, "site", name="latch_wire_anchor", pos=_vec((0, HINGE[1]+TAB[1], HINGE[2]+4.)), size=".01")

    finger = add(cart, "body", name="latch", pos=_vec(HINGE))
    add(finger, "joint", name="latch", type="hinge", axis=_vec((SIDE, 0, 0)), range=_vec((0, OPEN_STOP)),
        stiffness=repr(float(SPRING_K)), springref=repr(SPRING_REF), damping=".05",
        solreflimit=solref, solimplimit=solimp)  # stops as stiff as the contacts (MuJoCo's default is 20 ms)
    y0, y1 = PRONG_Y
    elbow = np.array((0., 0., PRONG_Z-HINGE[2]))
    bar = elbow+np.array((0., y0-HINGE[1], 0.))
    parts = [("latch_finger", (0, 0, 0), elbow, .02), ("latch_heel", elbow, bar, .02),
             ("latch_bar", bar+(-PRONG_X, 0, 0), bar+(PRONG_X, 0, 0), PRONG_R),
             ("latch_tab", (0, 0, 0), TAB, .02)]
    for side, x in (("a", -PRONG_X), ("b", PRONG_X)):
        parts.append((f"latch_prong_{side}", bar+(x, 0, 0), bar+(x, y1-y0, 0), PRONG_R))
    for name, a, b, r in parts:
        add(finger, "geom", name=name, type="capsule", fromto=_vec((*a, *b)), size=repr(r), material="teal",
            density=repr(STEEL), conaffinity="3")
    add(finger, "site", name="latch_wire_tab", pos=_vec(TAB), size=".01")
    tendon = root.find("tendon")
    if tendon is None:
        tendon = add(root, "tendon")
    wire = add(tendon, "spatial", name="latch_wire", width=".005", rgba="0.69 0.76 0.79 1")
    add(wire, "site", site="latch_wire_anchor")
    add(wire, "site", site="latch_wire_tab")
    # Motor on the wire: positive control is tension (a pull shortens the wire), capped at WIRE_MAX.
    add(root.find("actuator"), "motor", name="latch_wire", tendon="latch_wire", gear="-1",
        ctrlrange=_vec((0, WIRE_MAX)), ctrllimited="true")


def build_modern_scene(path=MODERN_SCENE, units=UNITS, fast=False):
    si = build_scene(Path(path).with_suffix(".si.xml"))
    root = ET.parse(si).getroot()
    root.set("model", "surgical_insertion_modern_v3")
    for e in list(root):
        if e.tag == "keyframe":
            root.remove(e)
    root.find("default").find("geom").set("conaffinity", "3")
    _find(root, "geom", "tissue_phantom").set("contype", "2")
    _find(root, "geom", "tissue_phantom").set("conaffinity", "2")
    _remove_first_design(root)
    anchor = _presentation(root)
    _scale_si(root, units)
    dt = FAST["dt"] if fast else DT_S
    law = law_for(FAST["contact_s"] if fast else TIME_CONSTANT_S)
    config = replace(fixture_config("drag", dt, MATERIAL, law), length_m=FREE_LENGTH, segments=SEGMENTS)
    names, ring_mass = _thread(root, units, config, anchor, FAST["armature_kg_m2"] if fast else 0.)
    _needle(root)
    solimp, solref = _vec(law.solimp(units)), _vec(law.solref())
    _cartridge(root, solref, solimp)
    _theme(root)
    option = root.find("option")
    for k, v in {"timestep": format(dt, "g"), "integrator": "RK4", "solver": "Newton",
                 "tolerance": FAST["tolerance"] if fast else "1e-12", "iterations": FAST["iterations"] if fast else "100",
                 "cone": "elliptic", "jacobian": "sparse",
                 "density": format(AIR_DENSITY*units.mass/units.length**3, ".6g"),
                 "viscosity": format(AIR_VISCOSITY*units.mass/units.length, ".6g")}.items():
        option.set(k, v)
    for geom in root.iter("geom"):
        geom.set("solimp", solimp)
        geom.set("solref", solref)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(root, space="  ")
    ET.ElementTree(root).write(path, encoding="unicode")
    return path, {"thread_bodies": names, "config": config, "law": law, "anchor_m": anchor.tolist(),
                  "ring_mass_g": ring_mass}


def load_modern(path=None, fast=False):
    from .der import plugin
    plugin()
    p, meta = build_modern_scene(path or (FAST_SCENE if fast else MODERN_SCENE), fast=fast)
    meta["xml_path"] = str(p)
    meta["fast"] = fast
    meta["xml_sha256"] = hashlib.sha256(Path(p).read_bytes()).hexdigest()
    meta["variant"] = "modern"
    meta["tube"] = None
    model = mujoco.MjModel.from_xml_path(str(p))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data, meta
