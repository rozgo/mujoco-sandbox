"""Thread-tube design (user, October 6): static scene for design review.

A thread tube on an arm from the tool head sits beside the needle, angled in
toward it, and moves with the tool. The thread waits in the tube with its plain
end on the needle's path. On its way down the needle point meets the end and
sticks to it (an abstracted bond, standing in for a chemical attachment),
carries it into the tissue, and releases it at depth. A new thread appears in
the tube for the next site. Engine units: mm, g, s.
"""

import hashlib
import math
import os
from dataclasses import replace
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from .der import der_xml
from .e2e_scene import AIR_DENSITY, AIR_VISCOSITY, MATERIAL, UNITS, _find, _scale_si, _theme, _vec
from .modern_scene import FAST, STEEL, TIP, _needle, _remove_first_design
from .scene import FIELD_Y, ROOT, add, build_scene, surface_z
from .task_fixtures import fixture_config, law_for

TUBE_SCENE = ROOT/"build/neural_insertion/e2e/tube_v1.xml"
Q_READY = 6.0               # insertion carriage at the ready position
END_BELOW_POINT = 1.0       # thread end this far below the needle point at the ready position
TUBE_ANGLE = math.radians(15.)  # from vertical, toward +X (45 deg dragged the thread around a sharp corner
                                # at insertion speed and it wrapped the needle: tube_design/cycle_v3)
TUBE_LENGTH, TUBE_R, TUBE_WALL = 9.0, .14, .02
THREAD_R = .02
# 18 segments of 0.46 mm (user, October 6): at 6 x 1.4 mm the rigid segments could not bend at the tube's
# mouth and jammed in the 0.2 mm bore (tube_design/cycle_v2). Shorter segments wiggle faster; a larger joint
# armature keeps them stable at the 50 µs step (slower fine wiggles, same shape and sag).
SEGMENTS, SEGMENT = 18, 8.25/18
THREAD_ARMATURE = 3e-17     # kg m² per thread joint
THREAD_DAMPING = 1e-12      # N m s/rad per thread joint: internal damping, about half of critical for a
                            # segment's bending; with none the thread whipped after release (sites_check)
INSERTED = 9                # thread end segments that pass the tissue surface (tissue force model)
GRIP_TIME_S = 1e-3          # tissue grip constraint time constant
THREADS = 3                 # thread 0 in the tube, spares parked weightless off to the side
THREAD_RGBA = "0.36 1 0.08 1"  # neon lime
PARK_OFFSET = np.array((70., 0., 20.))  # mm between parked threads (world), clear of the robot and tissue
ARM_X = 8.


def _frame_origin():
    """World position of the tool frame (insertion body at zero) with all joints at zero."""
    m = mujoco.MjModel.from_xml_path(str(build_scene(TUBE_SCENE.with_name(f"tube_probe_{os.getpid()}.xml"))))
    d = mujoco.MjData(m)
    mujoco.mj_kinematics(m, d)
    return d.xpos[m.body("insertion").id]*UNITS.length


def build_tube_scene(stage_z, path=TUBE_SCENE, units=UNITS):
    """Static review pose: tool frame lowered by stage_z (mm); needle at Q_READY."""
    origin = _frame_origin()
    si = build_scene(Path(path).with_suffix(".si.xml"))
    root = ET.parse(si).getroot()
    root.set("model", "surgical_insertion_tube_v1")
    for e in list(root):
        if e.tag == "keyframe":
            root.remove(e)
    root.find("default").find("geom").set("conaffinity", "3")
    _find(root, "geom", "tissue_phantom").set("contype", "2")
    _find(root, "geom", "tissue_phantom").set("conaffinity", "2")
    _remove_first_design(root)
    # The phantom, its markings and targets move together (breathing and pulse, disturbance.py), driven as a
    # mocap body: prescribed motion, not simulated.
    _find(root, "body", "specimen_support").set("mocap", "true")
    _scale_si(root, units)
    _needle(root)
    s, c = math.sin(TUBE_ANGLE), math.cos(TUBE_ANGLE)
    up = np.array((s, 0., c))                       # along the tube, from its exit up and out
    point = np.array((0., 0., TIP-Q_READY))         # needle point, tool frame
    end = point-np.array((0., 0., END_BELOW_POINT))  # the thread's end, on the needle's axis
    # The thread rests on the bottom of the bore (as gravity leaves it), so the tube's axis sits above it.
    rest = (TUBE_R-2*TUBE_WALL)-THREAD_R
    axis_off = rest*np.array((-c, 0., s))
    exit_ = end+axis_off+((TUBE_R+.10)/s)*up        # tube rim about 90 µm clear of the needle shaft
    top = exit_+TUBE_LENGTH*up

    # Tube and arm, welded to the Z stage (they move with the tool, not with the insertion stroke).
    zbody = _find(root, "body", "stage_z")
    tool = add(zbody, "body", name="thread_tube", pos=_find(root, "body", "insertion").get("pos"))
    ring = 16
    axis_x = np.cross(up, (0., 1., 0.)); axis_x /= np.linalg.norm(axis_x)
    for k in range(ring):  # hollow tube: wall capsules around the bore
        a = 2*math.pi*k/ring
        off = (TUBE_R-TUBE_WALL)*(math.cos(a)*axis_x+math.sin(a)*np.array((0., 1., 0.)))
        add(tool, "geom", name=f"tube_wall_{k}", type="capsule", size=repr(TUBE_WALL), group="3",
            fromto=_vec((*(exit_+off), *(top+off))), rgba="0 0 0 0", density=repr(STEEL))
    add(tool, "geom", name="tube_glass", type="cylinder", size=repr(TUBE_R), fromto=_vec((*exit_, *top)),
        contype="0", conaffinity="0", rgba="0.75 0.85 0.95 0.28", mass="0")
    add(tool, "geom", name="tube_cap", type="cylinder", size=repr(TUBE_R+.03),
        fromto=_vec((*top, *(top+.3*up))), material="graphite", density=repr(STEEL))
    elbow = np.array((ARM_X, 0., top[2]+.3))
    for name, a, b, r in (("arm_upright", (ARM_X, 0., 8.), elbow, .6),
                          ("arm_reach", elbow, top+.3*up, .35)):
        add(tool, "geom", name=name, type="capsule", fromto=_vec((*a, *b)), size=repr(r), material="steel",
            density=repr(STEEL))

    # Threads, world frame at the review pose; free roots (they pay out of the tube). Thread 0 waits in the
    # tube; the spares wait off to the side, weightless (gravcomp), until the cycle reloads one into the tube.
    law = law_for(FAST["contact_s"])
    config = replace(fixture_config("drag", FAST["dt"], MATERIAL, law), length_m=SEGMENTS*SEGMENT/1e3,
                     segments=SEGMENTS)
    start = end+SEGMENTS*SEGMENT*up
    world = origin+np.array((0., 0., stage_z))
    th = math.pi/2+TUBE_ANGLE                        # rotate the rod's +x to point down the tube
    contact = root.find("contact")
    if contact is None:
        contact = add(root, "contact")
    equality = add(root, "equality")
    threads, plugin = [], None
    for k in range(THREADS):
        rod = ET.fromstring(der_xml(config, units))
        for e in rod.iter():  # unique names per thread: thread_* -> thread_*_k, plugin instance rod -> rodk
            for attr in ("name", "body1", "body2", "site", "joint", "geom"):
                v = e.get(attr)
                if v and v.startswith("thread_"):
                    e.set(attr, f"{v}_{k}")
            if e.tag in ("plugin", "instance") and "rod" in (e.get("instance"), e.get("name")):
                e.set("instance" if e.get("instance") else "name", f"rod{k}")
        if plugin is None:
            root.insert(1, rod.find("extension"))
            plugin = root.find("extension").find("plugin")
        else:
            plugin.append(rod.find("extension").find("plugin").find("instance"))
        first = rod.find("worldbody").find("body")
        first.set("pos", _vec(world+start+k*PARK_OFFSET))
        first.set("quat", _vec((math.cos(th/2), 0., math.sin(th/2), 0.)))
        first.insert(0, ET.Element("freejoint", name=f"thread_root_{k}"))
        root.find("worldbody").append(first)
        bodies = [b for b in first.iter("body") if b.get("name", "").startswith("thread_B")]
        for b in bodies:
            j = b.find("joint")
            if j is not None:
                j.set("armature", repr(THREAD_ARMATURE*units.mass*units.length**2))
                j.set("damping", repr(THREAD_DAMPING*units.mass*units.length**2))
            if k:
                b.set("gravcomp", "1")
        for b in bodies[-INSERTED:]:  # segments that may follow the needle into the tissue
            b.find("geom").set("conaffinity", "1")
        for e in rod.find("contact"):
            contact.append(e)
        names = [b.get("name") for b in bodies]
        threads.append(names)
        # The needle meets the thread's end through the bond, not through contact.
        add(contact, "exclude", body1="insertion", body2=names[-1])
        # Abstracted handling (stand-ins, switched at run time by tube_cycle.py): the tube holds the thread's
        # back end until the needle picks it; the needle's bond holds its end.
        add(equality, "connect", name=f"tube_hold_{k}", body1="thread_tube", body2=names[0], anchor="0 0 0",
            active="false", solref=_vec(law.solref()), solimp=_vec(law.solimp(units)))
        add(equality, "connect", name=f"needle_bond_{k}", body1="insertion", body2=names[-1], anchor="0 0 0",
            active="false", solref=_vec(law.solref()), solimp=_vec(law.solimp(units)))
        # Tissue grip as soft constraints (solved implicitly, stable at the 50 µs step): each segment that
        # enters the tissue is pinned where it is and slips past the grip force (tube_cycle.TubeTissue). An
        # explicit spring-damper on these 0.8 µg segments overshot and diverged (tube_design/check_v5).
        for n in names[-INSERTED:]:
            add(equality, "connect", name=f"grip_{n}", body1="specimen_support", body2=n, anchor="0 0 0", active="false",
                solref=_vec((GRIP_TIME_S, 1.)), solimp=_vec(law.solimp(units)))
    _theme(root)
    # Thread in neon lime with a slight glow: high contrast against the pink tissue, magenta vessels, grey
    # robot and purple backdrop (colours only).
    add(root.find("asset"), "material", name="thread_lime", rgba=THREAD_RGBA, emission=".45", specular=".2")
    for geom in root.iter("geom"):
        if geom.get("name", "").startswith("thread_G"):
            geom.attrib.pop("rgba", None)
            geom.set("material", "thread_lime")
    option = root.find("option")
    for k, v in {"timestep": format(FAST["dt"], "g"), "integrator": "RK4", "solver": "Newton",
                 "tolerance": FAST["tolerance"], "iterations": FAST["iterations"], "cone": "elliptic",
                 "jacobian": "sparse", "density": format(AIR_DENSITY*units.mass/units.length**3, ".6g"),
                 "viscosity": format(AIR_VISCOSITY*units.mass/units.length, ".6g")}.items():
        option.set(k, v)
    solimp, solref = _vec(law.solimp(units)), _vec(law.solref())
    for geom in root.iter("geom"):
        geom.set("solimp", solimp)
        geom.set("solref", solref)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(root, space="  ")
    ET.ElementTree(root).write(path, encoding="unicode")
    return path, threads


def review_pose(target=0):
    """Model and data at the review pose over a target: thread end 1 mm above the surface, needle point 1 mm
    above the end."""
    from .der import plugin
    plugin()
    surface = surface_z(0., FIELD_Y)*UNITS.length
    origin = _frame_origin()
    stage_z = surface+1.0-(origin[2]+TIP-Q_READY-END_BELOW_POINT)
    path, threads = build_tube_scene(stage_z, TUBE_SCENE.with_name(f"tube_v1_{os.getpid()}.xml"))  # per process
    m = mujoco.MjModel.from_xml_path(str(path))
    d = mujoco.MjData(m)
    d.qpos[m.jnt_qposadr[m.joint("stage_z").id]] = stage_z  # initialization of the review pose
    d.qpos[m.jnt_qposadr[m.joint("insertion").id]] = Q_READY
    mujoco.mj_forward(m, d)
    meta = {"variant": "tube", "tube": None, "thread_bodies": threads[0], "threads": threads, "xml_path": str(path),
            "xml_sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(), "stage_z_mm": stage_z}
    return m, d, meta


def view():
    """Hold the review pose in the native viewer (no stepping: nothing moves)."""
    import time
    import mujoco.viewer
    m, d, _ = review_pose()
    with mujoco.viewer.launch_passive(m, d, show_left_ui=False, show_right_ui=False) as viewer:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        viewer.cam.lookat[:] = d.site("thread_S_last_0").xpos+np.array((.4, 0., .6))
        viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 4.0, 120, -12
        viewer.set_texts((None, None, "THREAD-TUBE DESIGN | static review pose, nothing simulated | "
                          "needle above, thread in the tube, its end on the needle's path", None))
        while viewer.is_running():
            viewer.sync()
            time.sleep(1/30)


if __name__ == "__main__":
    import sys
    if sys.argv[1:] == ["view"]:
        from .e2e_view import relaunch_under_mjpython
        relaunch_under_mjpython("sixlegs.neural_insertion.tube_scene")
        view()
