"""Experimental published DER port. Acceptance is reported by der_benchmarks."""

import ctypes
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from .der_build import build
from .rescaling import CableConfig, cable_xml, load_cable
from .scene import add
from .units import MM_G

_library = None
_library_path = None


def plugin():
    global _library, _library_path
    if _library is None:
        _library_path = build()
        mujoco.mj_loadPluginLibrary(str(_library_path))
        _library = ctypes.CDLL(str(_library_path))
        _library.sixlegs_der_metrics.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                                               np.ctypeslib.ndpointer(dtype=np.float64, shape=(4,))]
        _library.sixlegs_der_metrics.restype = None
    return _library


def build_provenance():
    plugin()
    return json.loads((_library_path.parent / "build.json").read_text())


def numbers(values):
    return " ".join(format(float(x), ".17g") for x in values)


def der_xml(config=CableConfig(), units=MM_G, twist=0., projection="direct"):
    """Expand the original compiled chain with identical rigid/contact properties.

    One massless, fixed terminal frame provides the endpoint position to the
    upstream implementation. It adds no DOF, collision, support or physical mass.
    """
    plugin()
    if projection not in ("published", "direct"):
        raise ValueError("Unknown DER force projection")
    native, _ = load_cable(config, units)
    root = ET.fromstring(cable_xml(config, units))
    root.set("model", "published_der_"+units.name)
    root.remove(root.find("extension"))
    extension = add(root, "extension")
    instance = add(add(extension, "plugin", plugin="sixlegs.der_qst"), "instance", name="rod")
    for key, value in {"bend": config.young_pa*units.modulus,
                       "twist": config.young_pa/(2*(1+config.poisson))*units.modulus,
                       "flat": "false", "vmax": 0, "twist_displace": twist,
                       "projection": projection}.items():
        add(instance, "config", key=key, value=str(value))
    world = root.find("worldbody")
    world.remove(world.find("composite"))
    parent = world
    ids = [i for i in range(native.nbody) if native.body(i).name.startswith("thread_B")]
    for i in ids:
        body = add(parent, "body", name=native.body(i).name,
                   pos=numbers(native.body_pos[i]), quat=numbers(native.body_quat[i]))
        add(body, "inertial", pos=numbers(native.body_ipos[i]),
            quat=numbers(native.body_iquat[i]), mass=format(native.body_mass[i], ".17g"),
            diaginertia=numbers(native.body_inertia[i]))
        if native.body_jntnum[i]:
            j = native.body_jntadr[i]
            add(body, "joint", name=native.joint(j).name, type="ball",
                pos=numbers(native.jnt_pos[j]),
                damping=format(native.dof_damping[native.jnt_dofadr[j]], ".17g"))
        g = native.body_geomadr[i]
        add(body, "geom", name=native.geom(g).name, type="capsule", size=numbers(native.geom_size[g]),
            pos=numbers(native.geom_pos[g]), quat=numbers(native.geom_quat[g]),
            rgba=numbers(native.geom_rgba[g]), condim=str(native.geom_condim[g]),
            friction=numbers(native.geom_friction[g]), solref=numbers(native.geom_solref[g]),
            solimp=numbers(native.geom_solimp[g]))
        add(body, "plugin", instance="rod")
        parent = body
    endpoint = native.site("thread_S_last").id
    add(parent, "site", name="thread_S_last", pos=numbers(native.site_pos[endpoint]), size=str(config.radius_m*units.length))
    end = add(parent, "body", name="thread_endpoint", pos=numbers(native.site_pos[endpoint]))
    add(end, "plugin", instance="rod")
    contact = add(root, "contact")
    for a, b in zip(ids[:-1], ids[1:]):
        add(contact, "exclude", body1=native.body(a).name, body2=native.body(b).name)
    return ET.tostring(root, encoding="unicode")


def load_der(config=CableConfig(), units=MM_G, twist=0., projection="direct"):
    model = mujoco.MjModel.from_xml_string(der_xml(config, units, twist, projection))
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    return model, data


def elastic_metrics(model, data, units=MM_G):
    result = np.zeros(4)
    plugin().sixlegs_der_metrics(model._address, data._address, result)
    return {"bending_j": result[0]/units.torque, "twisting_j": result[1]/units.torque,
            "end_twist_rad": result[2], "twist_length_m": result[3]/units.length}


def vertices(model, data, units=MM_G):
    ids = [i for i in range(model.nbody) if model.body(i).name.startswith("thread_B")]
    return np.vstack((data.xpos[ids], data.site("thread_S_last").xpos))/units.length


def preview(output):
    from PIL import Image, ImageDraw
    from .visuals import font
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    m, d = load_der()
    with mujoco.Renderer(m, height=800, width=1600) as renderer:
        option = mujoco.MjvOption()
        renderer.update_scene(d, camera="cable", scene_option=option)
        renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
        raw = Image.fromarray(renderer.render())
    frame = Image.new("RGB", (1600, 930), "#101c25")
    frame.paste(raw, (0,80))
    draw = ImageDraw.Draw(frame)
    draw.text((25,20), "DER / MUJOCO 3.12 / DIRECT FORCE PROJECTION / STATIC BENCH", font=font(25), fill="#eff6f7")
    draw.text((25,895), "44 mm long | 40 micrometre diameter | physical mass/radius preserved | t = 0", font=font(20), fill="#6ed8c6")
    frame.save(output)
    return output
