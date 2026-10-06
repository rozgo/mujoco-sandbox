"""Export the workcell and recorded episodes for the web replay viewer.

The viewer replays recorded states; it does not simulate. Geometry is the
MuJoCo scene tessellated per (body, material), in body coordinates, so moving
bodies need only their recorded poses. Episode poses come from MuJoCo forward
kinematics of recorded joint positions. SI units, Z up.
"""

import json
from pathlib import Path
import struct

import mujoco
import numpy as np

from .motion import JOINT_NAMES
from .scene import load_scene

MOVING = ("stage_x", "stage_y", "stage_z", "insertion", "retainer")
MAX_VERTS = 60000  # raylib meshes use 16-bit indices

# Physically based materials in the site palette (#D38AAA #AF86AC #8981AF #7378A8 #646DA0).
# albedo sRGB hex, metallic, roughness, emissive, wrap (soft subsurface-like diffuse).
THEME = {
    "shell": ("#D8D3E3", 0.0, 0.46, 0.0, 0.0),
    "graphite": ("#2E3150", 0.1, 0.55, 0.0, 0.0),
    "rail": ("#B9BCC8", 1.0, 0.34, 0.0, 0.0),
    "steel": ("#DADDE6", 1.0, 0.30, 0.0, 0.0),
    "teal": ("#D38AAA", 0.0, 0.35, 1.6, 0.0),
    "blueglass": ("#1C2140", 0.0, 0.06, 0.0, 0.0),
    "rubber": ("#1A1B26", 0.0, 0.88, 0.0, 0.0),
    "tissue": ("#D9909F", 0.0, 0.36, 0.0, 0.45),
    "vessel": ("#7C1F43", 0.0, 0.30, 0.0, 0.35),
    "vessel_tip": ("#A0305A", 0.0, 0.30, 0.0, 0.35),
    "target": ("#C9C4F2", 0.0, 0.30, 2.2, 0.0),
    "thread": ("#F0C27A", 0.0, 0.40, 0.25, 0.2),
    "floor": ("#3A3557", 0.0, 0.85, 0.0, 0.0),
    "tray": ("#646DA0", 0.6, 0.38, 0.0, 0.0),
}


def srgb_to_linear(hex_color):
    c = np.array([int(hex_color[i:i+2], 16)/255 for i in (1, 3, 5)])
    return np.where(c <= .04045, c/12.92, ((c+.055)/1.055)**2.4)


def quat_matrix(q):
    m = np.zeros(9)
    mujoco.mju_quat2Mat(m, q)
    return m.reshape(3, 3)


def sphere_cap(radius, center_z, upper, rings=6, segments=20):
    """Hemisphere at z = center_z, opening toward -z (upper) or +z."""
    verts, norms, idx = [], [], []
    for i in range(rings+1):
        phi = (np.pi/2)*i/rings
        for j in range(segments+1):
            th = 2*np.pi*j/segments
            n = np.array([np.cos(phi)*np.cos(th), np.cos(phi)*np.sin(th), np.sin(phi) if upper else -np.sin(phi)])
            verts.append(n*radius+(0, 0, center_z))
            norms.append(n)
    for i in range(rings):
        for j in range(segments):
            a, b = i*(segments+1)+j, (i+1)*(segments+1)+j
            idx += [a, a+1, b, a+1, b+1, b] if upper else [a, b, a+1, a+1, b, b+1]  # outward, counter-clockwise
    return np.array(verts), np.array(norms), np.array(idx)


def tube(radius, half, segments=20, caps=False):
    verts, norms, idx = [], [], []
    for z in (-half, half):
        for j in range(segments+1):
            th = 2*np.pi*j/segments
            n = np.array([np.cos(th), np.sin(th), 0])
            verts.append(n*radius+(0, 0, z))
            norms.append(n)
    for j in range(segments):
        a, b = j, segments+1+j
        idx += [a, a+1, b, a+1, b+1, b]
    out = [(np.array(verts), np.array(norms), np.array(idx))]
    if caps:
        for z, s in ((-half, -1), (half, 1)):
            v = [(0, 0, z)]+[(radius*np.cos(2*np.pi*j/segments), radius*np.sin(2*np.pi*j/segments), z) for j in range(segments)]
            n = [(0, 0, s)]*(segments+1)
            f = []
            for j in range(segments):
                a, b = 1+j, 1+(j+1) % segments
                f += [0, a, b] if s > 0 else [0, b, a]
            out.append((np.array(v, float), np.array(n, float), np.array(f)))
    return out


def box(half):
    verts, norms, idx = [], [], []
    for axis in range(3):
        for sign in (-1, 1):
            n = np.zeros(3)
            n[axis] = sign
            u, v = [k for k in range(3) if k != axis]
            base = len(verts)
            for a, b in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
                p = np.zeros(3)
                p[axis], p[u], p[v] = sign*half[axis], a*half[u], b*half[v]
                verts.append(p)
                norms.append(n)
            quad = [base, base+1, base+2, base, base+2, base+3]
            # Keep counter-clockwise winding seen from outside.
            if np.dot(np.cross(verts[base+1]-verts[base], verts[base+2]-verts[base]), n) < 0:
                quad = [base, base+2, base+1, base, base+3, base+2]
            idx += quad
    return [(np.array(verts), np.array(norms), np.array(idx))]


def mesh_geom(model, geom, crease_deg=40):
    mid = model.geom_dataid[geom]
    v0, nv = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
    f0, nf = model.mesh_faceadr[mid], model.mesh_facenum[mid]
    verts = model.mesh_vert[v0:v0+nv].astype(float)
    faces = model.mesh_face[f0:f0+nf]
    fn = np.cross(verts[faces[:, 1]]-verts[faces[:, 0]], verts[faces[:, 2]]-verts[faces[:, 0]])
    fn /= np.linalg.norm(fn, axis=1, keepdims=True)
    # Smooth shading across facets within the crease angle; geometry stays exact.
    out_v, out_n, out_i = [], [], []
    cos_crease = np.cos(np.radians(crease_deg))
    incident = {}
    for f, tri in enumerate(faces):
        for k in tri:
            incident.setdefault(int(k), []).append(f)
    for f, tri in enumerate(faces):
        for k in tri:
            near = [g for g in incident[int(k)] if np.dot(fn[g], fn[f]) > cos_crease]
            n = fn[near].sum(axis=0)
            out_v.append(verts[k])
            out_n.append(n/np.linalg.norm(n))
            out_i.append(len(out_v)-1)
    return [(np.array(out_v), np.array(out_n), np.array(out_i))]


def tessellate(model, geom):
    kind, size = int(model.geom_type[geom]), model.geom_size[geom]
    if kind == mujoco.mjtGeom.mjGEOM_BOX:
        return box(size)
    if kind == mujoco.mjtGeom.mjGEOM_CYLINDER:
        return tube(size[0], size[1], caps=True)
    if kind == mujoco.mjtGeom.mjGEOM_CAPSULE:
        segments = 12 if size[0] < 1e-3 else 20
        return tube(size[0], size[1], segments) + [sphere_cap(size[0], size[1], True, 4, segments),
                                                    sphere_cap(size[0], -size[1], False, 4, segments)]
    if kind == mujoco.mjtGeom.mjGEOM_SPHERE:
        return [sphere_cap(size[0], 0, True), sphere_cap(size[0], 0, False)]
    if kind == mujoco.mjtGeom.mjGEOM_PLANE:
        return box(np.array([size[0], size[1], 1e-5]))
    if kind == mujoco.mjtGeom.mjGEOM_MESH:
        return mesh_geom(model, geom)
    raise ValueError(f"Unsupported geom type {kind}")


def scene_chunks(model):
    """Merged triangle meshes per (body, material) in body coordinates."""
    materials = [model.material(i).name for i in range(model.nmat)]
    groups = {}
    for g in range(model.ngeom):
        mat = int(model.geom_matid[g])
        if mat < 0:
            continue
        body = int(model.geom_bodyid[g])
        rot, pos = quat_matrix(model.geom_quat[g]), model.geom_pos[g]
        for v, n, i in tessellate(model, g):
            groups.setdefault((body, mat), []).append((v@rot.T+pos, n@rot.T, i))
    chunks = []
    for (body, mat), parts in sorted(groups.items()):
        verts, norms, idx, count = [], [], [], 0
        for v, n, i in parts:
            if count+len(v) > MAX_VERTS:
                chunks.append((body, mat, np.vstack(verts), np.vstack(norms), np.concatenate(idx)))
                verts, norms, idx, count = [], [], [], 0
            verts.append(v)
            norms.append(n)
            idx.append(i+count)
            count += len(v)
        chunks.append((body, mat, np.vstack(verts), np.vstack(norms), np.concatenate(idx)))
    return chunks, materials


def write_scene(path, model, data):
    chunks, materials = scene_chunks(model)
    with open(path, "wb") as f:
        f.write(b"NIS1")
        f.write(struct.pack("<III", model.nbody, len(materials), len(chunks)))
        for b in range(model.nbody):
            name = model.body(b).name.encode()[:31]
            f.write(struct.pack("<i32s3f4fB", b, name, *data.xpos[b], *data.xquat[b], model.body(b).name in MOVING))
        for name in materials:
            hex_color, metal, rough, emissive, wrap = THEME[name]
            f.write(struct.pack("<16s3f4f", name.encode(), *srgb_to_linear(hex_color), metal, rough, emissive, wrap))
        for body, mat, v, n, i in chunks:
            f.write(struct.pack("<iiII", body, mat, len(v), len(i)))
            f.write(v.astype("<f4").tobytes())
            f.write(n.astype("<f4").tobytes())
            f.write(i.astype("<u2").tobytes())
    return {"chunks": len(chunks), "vertices": int(sum(len(c[2]) for c in chunks)),
            "triangles": int(sum(len(c[4]) for c in chunks)//3), "materials": materials,
            "theme": {k: list(v) for k, v in THEME.items()}}


def episode_frames(model, data, qpos_rows):
    """Moving-body world positions and the needle tip for recorded joint positions."""
    qadr = [model.jnt_qposadr[model.joint(n).id] for n in JOINT_NAMES]
    bodies = [model.body(n).id for n in MOVING]
    tip = model.site("needle_tip").id
    poses, tips = [], []
    for q in qpos_rows:
        data.qpos[qadr] = q  # replay of a recorded state
        mujoco.mj_kinematics(model, data)
        poses.append(data.xpos[bodies].copy())
        tips.append(data.site_xpos[tip].copy())
    return np.array(poses), np.array(tips)


# Per-frame floats after time, moving-body positions and tip: lateral and vertical
# error (um, to the true moving goal), true goal (3), measured tip (3), target
# estimate (3), table acceleration (3), force noise, extra friction and vibration
# inertial force on each slide (3 x 5, N along the joint axis), |action|.
EXTRA = 2+3+3+3+3+3*len(JOINT_NAMES)+1


def write_replays(path, model, data, episodes):
    """episodes: dicts with seed, policy, outcome and states (align_policy.states_from_rows).

    Format NIR2. The goal moves with modeled tissue motion, so it is stored per
    frame; disturbance components are the core's recorded values at the 50 Hz
    policy rate (vibration content above 25 Hz is aliased in this record).
    """
    mujoco.mj_kinematics(model, data)
    axes = [data.xaxis[model.joint(n).id].copy() for n in JOINT_NAMES]
    with open(path, "wb") as f:
        f.write(b"NIR2")
        f.write(struct.pack("<III", len(episodes), len(MOVING), EXTRA))
        f.write(struct.pack(f"<{len(MOVING)}i", *[model.body(n).id for n in MOVING]))
        for axis in axes:
            f.write(struct.pack("<3f", *axis))
        for e in episodes:
            s = e["states"]
            poses, tips = episode_frames(model, data, s["qpos"])
            goal = s["goal"]
            lateral = np.linalg.norm(tips[:, :2]-goal[:, :2], axis=1)*1e6
            vertical = (tips[:, 2]-goal[:, 2])*1e6
            f.write(struct.pack("<IIIII", e["seed"], s["target"], e["policy"], e["outcome"], len(poses)))
            f.write(struct.pack("<2f", s["level"][0], s["latency_s"][0]))
            extra = np.column_stack([lateral, vertical, goal, s["measured_tip"], s["measured_goal"], s["base_acc"],
                                     s["force_noise"], s["friction"], s["vibration_force"], s["action_norm"]])
            frames = np.column_stack([s["time"], poses.reshape(len(poses), -1), tips, extra])
            f.write(frames.astype("<f4").tobytes())
