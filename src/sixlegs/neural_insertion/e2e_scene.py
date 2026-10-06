"""End-to-end workcell variant: the robot, a full DER thread and a pickup eyelet.

A separate variant of the static workcell (the alignment scene is unchanged),
built in SI and converted to millimetre-gram-second engine units so the 40 µm
thread compiles, as in the thread studies. Mechanism, first design, ours and
provisional:

- The thread's free end carries a rigid eyelet (180 µm ring radius, 20 µm wire).
- The needle has a side slot near its tip: two lips protruding toward -Y. The
  needle passes down through the eyelet, moves -Y so the eyelet rim enters the
  slot, and then carries the rim up or down. Moving back +Y releases it.
- The keeper (the retainer slide, here mounted on the needle carriage so it
  tracks the tip) lowers two prongs beside the slot opening during transport.
  Pads at the prong tips close over the rim and fences hang past it, caging the
  rim on the slot's lower lip (v1.2; the first pass held the eyelet by gravity
  alone and lost it, a pad clamp without fences let it slide out, and a firm
  squeeze ejected it). It retracts before insertion.
- The cassette presents the thread on a bed with a short trench under the
  eyelet, so the lower lip can pass below the rim.
- Tool tip, slot and the eyelet end of the thread do not collide with the
  phantom mesh; tissue.py applies the needle-tissue and retention forces there.
  Every other robot and thread contact with the phantom remains.

All geoms use the task-regime contact law (20 µs, impedance ramp) and the
thread clock (RK4, 5 µs), as in TASK_REGIME.md.
"""

import math
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from .der import der_xml
from .scene import FIELD_Y, ROOT, add, build_scene, surface_z
from .task_fixtures import fixture_config, law_for
from .units import MM_G

E2E_SCENE = ROOT/"build/neural_insertion/e2e/e2e_v0.xml"
TUBE_SCENE = ROOT/"build/neural_insertion/e2e/e2e_tubes_v1.xml"
UNITS = MM_G
TIME_CONSTANT_S = 2e-5
DT_S = 5e-6
MATERIAL = "illustrative_100mpa"

# Geometry, SI metres. The needle tip is at -0.083075 m in the insertion body.
TIP_Z = -0.083075
LIP_RADIUS = 20e-6
LIP_REACH = 200e-6           # lip centreline from the needle axis toward -Y
LIP_Z = (150e-6, 330e-6)     # lower and upper lip centre heights above the tip
EYELET_RADIUS = 180e-6       # ring centreline radius
EYELET_WIRE = 20e-6
EYELET_SEGMENTS = 12
KEEPER_RADIUS = 20e-6
KEEPER_X = 60e-6             # two prongs at +-X, beside the slot lips
KEEPER_Y = -190e-6
KEEPER_BOTTOM = 3.05e-3      # prong bottom above the tip at keeper travel 0
KEEPER_LENGTH = 5e-3
KEEPER_PAD = (-190e-6, -100e-6)  # pads at the prong tips run +Y over the eyelet rim (at x = +-60 µm)
CLAMP_OVERLAP = 0.           # cage, not clamp: pads stop at the rim (a 30 µm squeeze ejected it, run_v3)
KEEPER_FENCE = (-200e-6, 100e-6)  # fence y and its bottom above the tip: blocks the rim sliding out of the slot
# Keeper design. "latch" (v2, current): one bar closes the slot's open end, so the slot becomes a
# closed eye, as on a knitting machine's latch needle; the ring may swing but cannot leave. "clamp":
# pads press the rim onto the lower lip with a limited force (v1.x, kept for comparison).
KEEPER_DESIGN = "latch"
LATCH_Y = -(LIP_REACH+LIP_RADIUS+10e-6+KEEPER_RADIUS)  # 10 µm past the lip ends: less than the 40 µm rim wire
LATCH_BOTTOM = 80e-6         # latch bottom above the tip when closed: below the lower lip (bottom at 130 µm)
if KEEPER_DESIGN == "latch":
    KEEPER_CLOSED = KEEPER_BOTTOM-LATCH_BOTTOM
else:  # keeper travel that seats the pads on the rim (resting on the lower lip)
    KEEPER_CLOSED = KEEPER_BOTTOM-(LIP_Z[0]+LIP_RADIUS+2*EYELET_WIRE+KEEPER_RADIUS)+CLAMP_OVERLAP

# Presentation: thread 0 of the cassette. Cassette body at (-0.110, FIELD_Y, 0).
CASSETTE = np.array((-.110, FIELD_Y, 0.))
BED_TOP = .108
EYELET_CENTER = CASSETTE+np.array((-.015, -.019, BED_TOP+EYELET_WIRE))
TRENCH = {"x": 100e-6, "y": (-350e-6, 200e-6), "depth": 600e-6}
INSERTED_SEGMENTS = 2        # segments nearest the eyelet that may enter tissue
# Tissue channel ("tube") variant, the user's proposal: a rigid, pre-formed channel
# at the target with collision, so guidance and retention come from contact.
TUBE = {"radius": 280e-6, "offset_y": -30e-6, "depth": 3e-3, "wall": 50e-6, "segments": 16,
        "skin": 100e-6, "skin_half": 2.5e-3}


def _vec(values):
    return " ".join(format(float(v), ".12g") for v in values)


def _scale(attr, factor):
    return _vec(np.array(attr.split(), float)*factor)


def _scale_si(root, units):
    """Convert the SI workcell to engine units, element by element."""
    L, M = units.length, units.mass
    for e in root.iter():
        tag, a = e.tag, e.attrib
        if tag in ("geom", "site", "body", "camera", "light", "inertial"):
            for key in ("pos", "fromto"):
                if key in a:
                    e.set(key, _scale(a[key], L))
        if tag in ("geom", "site") and "size" in a:
            e.set("size", _scale(a["size"], L))
        if tag == "geom" and "friction" in a:
            f = np.array(a["friction"].split(), float)
            f[1:] *= L  # torsional and rolling coefficients are lengths
            e.set("friction", _vec(f))
        if tag == "inertial":
            e.set("mass", _scale(a["mass"], M))
            e.set("diaginertia", _scale(a["diaginertia"], M*L**2))
        if tag == "joint" and a.get("type") == "slide":
            e.set("range", _scale(a["range"], L))
            e.set("damping", _scale(a["damping"], M))
            e.set("frictionloss", _scale(a["frictionloss"], M*L))
        if tag == "position":
            e.set("kp", _scale(a["kp"], M))
            e.set("kv", _scale(a["kv"], M))
            e.set("ctrlrange", _scale(a["ctrlrange"], L))
            e.set("forcerange", _scale(a["forcerange"], M*L))
        if tag == "mesh":
            e.set("vertex", _scale(a["vertex"], L))
        if tag == "statistic":
            e.set("center", _scale(a["center"], L))
            e.set("extent", _scale(a["extent"], L))
        if tag == "option":
            e.set("gravity", _scale(a["gravity"], L))


def _find(root, tag, name):
    for e in root.iter(tag):
        if e.get("name") == name:
            return e
    raise KeyError(name)


def _parent_map(root):
    return {child: parent for parent in root.iter() for child in parent}


def _tool(root):
    """Slot lips on the needle; the keeper replaces the retainer fork, on the needle carriage."""
    needle = _find(root, "body", "insertion")
    for i, z in enumerate(LIP_Z):
        add(needle, "geom", name=f"slot_lip_{i}", type="capsule", size=_vec((LIP_RADIUS,)),
            fromto=_vec((0, 0, TIP_Z+z, 0, -LIP_REACH, TIP_Z+z)), material="steel", conaffinity="1")
    _find(root, "geom", "needle").set("conaffinity", "1")  # tissue forces come from tissue.py
    parents = _parent_map(root)
    old = _find(root, "body", "retainer")
    parents[old].remove(old)
    keeper = add(needle, "body", name="retainer", pos="0 0 0")
    # Flexure-guided keeper: negligible sliding friction (10 uN), so a 100 uN clamp force is well defined.
    add(keeper, "joint", name="retainer", type="slide", axis="0 0 -1", range=_vec((0, .006)),
        damping=".03", frictionloss="1e-5")
    add(keeper, "inertial", mass=".004", pos="0 0 0", diaginertia=_vec((3e-7, 3e-7, 1e-8)))
    bottom, top = TIP_Z+KEEPER_BOTTOM, TIP_Z+KEEPER_BOTTOM+KEEPER_LENGTH
    if KEEPER_DESIGN == "latch":
        add(keeper, "geom", name="keeper_latch", type="capsule", size=_vec((KEEPER_RADIUS,)),
            fromto=_vec((0, LATCH_Y, bottom, 0, LATCH_Y, top)), material="teal")
        path = [(0, LATCH_Y, top), (0, -2.5e-3, top), (0, -2.5e-3, -.012), (0, -7e-3, -.012)]
        for i, (a, b) in enumerate(zip(path, path[1:])):
            add(keeper, "geom", name=f"keeper_arm_{i}", type="capsule", size=_vec((1.2e-4 if i else 6e-5,)),
                fromto=_vec((*a, *b)), material="steel")
        add(keeper, "geom", name="keeper_drive", type="box", size=_vec((.002, .002, .004)), pos=_vec((0, -.0085, -.008)),
            material="teal")
        return
    for side in (-1, 1):
        add(keeper, "geom", name=f"keeper_prong_{side}", type="capsule", size=_vec((KEEPER_RADIUS,)),
            fromto=_vec((side*KEEPER_X, KEEPER_Y, bottom, side*KEEPER_X, KEEPER_Y, top)), material="teal")
        # Clamp pad: presses the eyelet rim onto the slot's lower lip (mechanism v1).
        add(keeper, "geom", name=f"keeper_pad_{side}", type="capsule", size=_vec((KEEPER_RADIUS,)),
            fromto=_vec((side*KEEPER_X, KEEPER_PAD[0], bottom, side*KEEPER_X, KEEPER_PAD[1], bottom)),
            material="teal")
        # Fence: hangs from the pad's outer end past the rim, so the rim cannot slide out of the slot
        # beneath the pads (v1.1; v1 lost the eyelet that way during the lift).
        drop = (LIP_Z[0]+LIP_RADIUS+2*EYELET_WIRE+KEEPER_RADIUS)-KEEPER_FENCE[1]
        add(keeper, "geom", name=f"keeper_fence_{side}", type="capsule", size=_vec((KEEPER_RADIUS,)),
            fromto=_vec((side*KEEPER_X, KEEPER_FENCE[0], bottom, side*KEEPER_X, KEEPER_FENCE[0], bottom-drop)),
            material="teal")
    path = [(0, KEEPER_Y, top), (0, -2.5e-3, top), (0, -2.5e-3, -.012), (0, -7e-3, -.012)]
    add(keeper, "geom", name="keeper_bridge", type="capsule", size=_vec((KEEPER_RADIUS*2,)),
        fromto=_vec((-KEEPER_X, KEEPER_Y, top, KEEPER_X, KEEPER_Y, top)), material="teal")
    for i, (a, b) in enumerate(zip(path, path[1:])):
        add(keeper, "geom", name=f"keeper_arm_{i}", type="capsule", size=_vec((1.2e-4 if i else 6e-5,)),
            fromto=_vec((*a, *b)), material="steel")
    add(keeper, "geom", name="keeper_drive", type="box", size=_vec((.002, .002, .004)), pos=_vec((0, -.0085, -.008)),
        material="teal")


def _cassette(root):
    """Bed with a trench under the eyelet, replacing presentation sample 0's markings."""
    fixture = _find(root, "body", "thread_cassette")
    parents = _parent_map(root)
    for name in ("presented_thread_0", "pickup_loop_0", "thread_tab_0"):
        for e in [g for g in fixture.iter("geom") if g.get("name", "").startswith(name)]:
            parents[e].remove(e)
    c = EYELET_CENTER-CASSETTE
    tx, (y0, y1), depth = TRENCH["x"], TRENCH["y"], TRENCH["depth"]
    x_lo, x_hi, y_lo, y_hi = -.021, .021, -.021, .021
    half = .0015
    z = BED_TOP-half
    blocks = {  # (x range, y range) around the trench, all with top at BED_TOP
        "bed_left": ((x_lo, c[0]-tx), (y_lo, y_hi)),
        "bed_right": ((c[0]+tx, x_hi), (y_lo, y_hi)),
        "bed_front": ((c[0]-tx, c[0]+tx), (y_lo, c[1]+y0)),
        "bed_back": ((c[0]-tx, c[0]+tx), (c[1]+y1, y_hi)),
    }
    for name, ((a, b), (p, q)) in blocks.items():
        add(fixture, "geom", name=name, type="box", size=_vec(((b-a)/2, (q-p)/2, half)),
            pos=_vec(((a+b)/2, (p+q)/2, z)), material="graphite")
    add(fixture, "geom", name="bed_trench_floor", type="box", size=_vec((tx, (y1-y0)/2, half)),
        pos=_vec((c[0], c[1]+(y0+y1)/2, BED_TOP-depth-half)), material="tray")


def _tube(root, target=0):
    """A channel at a target site: 16 wall plates, a floor and a surface skin around the opening.

    The channel collides with the needle, slot, eyelet and thread (bit 1), not
    with the phantom mesh. Elsewhere the inserted tool and thread end do not
    collide with the phantom, so the runner flags a needle tip below the surface
    outside the channel as a miss.
    """
    stage = _find(root, "body", "specimen_support")
    site = np.array(_find(root, "site", f"target_{target}").get("pos").split(), float)
    z_top = surface_z(site[0], site[1]+FIELD_Y)
    c = np.array((site[0], site[1]+TUBE["offset_y"]))
    r, wall, depth, n = TUBE["radius"], TUBE["wall"], TUBE["depth"], TUBE["segments"]
    half_width = (r+wall)*math.tan(math.pi/n)*1.02
    for k in range(n):
        a = 2*math.pi*k/n
        center = c+(r+wall/2)*np.array((math.cos(a), math.sin(a)))
        add(stage, "geom", name=f"tube_wall_{k}", type="box", size=_vec((wall/2, half_width, depth/2)),
            pos=_vec((*center, z_top-depth/2)), euler=_vec((0, 0, a)), material="tissue",
            contype="1", conaffinity="1", group="1")
    add(stage, "geom", name="tube_floor", type="box", size=_vec((r+wall, r+wall, wall/2)),
        pos=_vec((*c, z_top-depth-wall/2)), material="tissue", contype="1", conaffinity="1", group="1")
    h, t = TUBE["skin_half"], TUBE["skin"]
    for name, (x0, x1, y0, y1) in {"left": (-h, -r, -h, h), "right": (r, h, -h, h),
                                    "front": (-r, r, -h, -r), "back": (-r, r, r, h)}.items():
        add(stage, "geom", name=f"tube_skin_{name}", type="box", size=_vec(((x1-x0)/2, (y1-y0)/2, t/2)),
            pos=_vec((c[0]+(x0+x1)/2, c[1]+(y0+y1)/2, z_top-t/2)), material="tissue",
            contype="1", conaffinity="1", group="1")
    return {"center_m": [float(c[0]), float(c[1]+FIELD_Y)], "top_z_m": float(z_top)}


def _thread(root, units, config):
    """Insert the DER thread as a free body chain lying on the bed, eyelet end at -Y."""
    rod = ET.fromstring(der_xml(config, units))
    L = units.length
    root.insert(1, rod.find("extension"))
    world = root.find("worldbody")
    first = rod.find("worldbody").find("body")
    # Rest shape: straight along local +X. Root at +Y, free end (eyelet) toward -Y.
    tail = EYELET_CENTER+np.array((0, EYELET_RADIUS, 0))  # where the thread meets the ring
    root_pos = tail+np.array((0, config.length_m, 0))
    first.set("pos", _vec(root_pos*L))
    first.set("quat", _vec((math.cos(-math.pi/4), 0, 0, math.sin(-math.pi/4))))
    first.insert(0, ET.Element("freejoint", {"name": "thread_root"}))
    world.append(first)
    bodies = [b for b in first.iter("body") if b.get("name", "").startswith("thread_B")]
    for i, b in enumerate(bodies):
        g = b.find("geom")
        g.set("conaffinity", "1" if i >= len(bodies)-INSERTED_SEGMENTS else "3")
    last = bodies[-1]
    seg = config.length_m/config.segments
    center = np.array((seg+EYELET_RADIUS, 0, 0))
    for k in range(EYELET_SEGMENTS):
        a0, a1 = 2*math.pi*k/EYELET_SEGMENTS, 2*math.pi*(k+1)/EYELET_SEGMENTS
        p0 = center+EYELET_RADIUS*np.array((math.cos(a0), math.sin(a0), 0))
        p1 = center+EYELET_RADIUS*np.array((math.cos(a1), math.sin(a1), 0))
        add(last, "geom", name=f"eyelet_{k}", type="capsule", size=_vec((EYELET_WIRE*L,)),
            fromto=_vec(np.concatenate((p0, p1))*L), rgba="1 .67 .12 1", conaffinity="1", condim="3",
            friction=_vec((config.friction, 1e-6*L, 1e-7*L)), mass="0")
    add(last, "site", name="eyelet_center", pos=_vec(center*L), size=_vec((EYELET_WIRE*L,)))
    contact = root.find("contact")
    if contact is None:
        contact = add(root, "contact")
    for e in rod.find("contact"):
        contact.append(e)
    return [b.get("name") for b in bodies]


def _hex(color, alpha=1.):
    return _vec([int(color[i:i+2], 16)/255 for i in (1, 3, 5)]+[alpha])


def _theme(root):
    """Site palette (#D38AAA #AF86AC #8981AF #7378A8 #646DA0) for renders; colours only."""
    from .site_export import THEME
    for material in root.find("asset").iter("material"):
        if material.get("name") in THEME:
            material.set("rgba", _hex(THEME[material.get("name")][0]))
    for texture in root.find("asset").iter("texture"):
        if texture.get("name") == "sky":
            texture.set("rgb1", _vec([int("8981AF"[i:i+2], 16)/255 for i in (0, 2, 4)]))
            texture.set("rgb2", _vec([int("646DA0"[i:i+2], 16)/255 for i in (0, 2, 4)]))
    thread = _hex(THEME["thread"][0])
    for geom in root.iter("geom"):
        name = geom.get("name", "")
        if name.startswith("thread_G") or name.startswith("eyelet_"):
            geom.set("rgba", thread)


AIR_DENSITY = 1.204          # kg/m^3, air at 20 C
AIR_VISCOSITY = 1.81e-5      # Pa s


def build_e2e_scene(path=E2E_SCENE, units=UNITS, tubes=False, air=True):
    si = build_scene(Path(path).with_suffix(".si.xml"))
    root = ET.parse(si).getroot()
    root.set("model", "surgical_insertion_e2e_v0")
    for e in list(root):
        if e.tag == "keyframe":
            root.remove(e)
    default_geom = root.find("default").find("geom")
    default_geom.set("conaffinity", "3")
    _find(root, "geom", "tissue_phantom").set("contype", "2")
    _find(root, "geom", "tissue_phantom").set("conaffinity", "2")
    _tool(root)
    _cassette(root)
    tube = _tube(root) if tubes else None
    _scale_si(root, units)
    law = law_for(TIME_CONSTANT_S)
    config = fixture_config("drag", DT_S, MATERIAL, law)
    names = _thread(root, units, config)
    _theme(root)
    option = root.find("option")
    for k, v in {"timestep": format(DT_S, "g"), "integrator": "RK4", "solver": "Newton", "tolerance": "1e-12",
                 "iterations": "100", "cone": "elliptic", "jacobian": "sparse"}.items():
        option.set(k, v)
    if air:
        # Still air at room temperature (1.204 kg/m^3, 1.81e-5 Pa s) through MuJoCo's inertia-based fluid model.
        # Drag on a thread segment at 0.1 m/s is about a quarter of its weight; it damps whipping and ringing.
        option.set("density", format(AIR_DENSITY*units.mass/units.length**3, ".6g"))
        option.set("viscosity", format(AIR_VISCOSITY*units.mass/units.length, ".6g"))
    solimp, solref = _vec(law.solimp(units)), _vec(law.solref())
    for geom in root.iter("geom"):
        geom.set("solimp", solimp)
        geom.set("solref", solref)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(root, space="  ")
    ET.ElementTree(root).write(path, encoding="unicode")
    return path, {"thread_bodies": names, "config": config, "law": law, "tube": tube}


def load_e2e(path=None, tubes=False):
    import hashlib
    from .der import plugin
    plugin()
    p, meta = build_e2e_scene(path or (TUBE_SCENE if tubes else E2E_SCENE), tubes=tubes)
    meta["xml_sha256"] = hashlib.sha256(Path(p).read_bytes()).hexdigest()
    meta["variant"] = "tubes" if tubes else "force_model"
    model = mujoco.MjModel.from_xml_path(str(p))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data, meta


def phantom_surface_m(x, y):
    """Tissue surface height (SI) at world x, y, from the phantom's analytic dome."""
    return surface_z(x, y)
