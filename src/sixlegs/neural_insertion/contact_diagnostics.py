"""Contact isolation: rigid segments, zero-elasticity chain, and matched DER."""

import argparse
import ctypes
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import subprocess
import time
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from .der import build_provenance, der_xml, elastic_metrics, numbers, vertices
from .rescaling import CableConfig, load_cable
from .scene import ROOT, add, camera
from .units import MM_G, MM_UG, Units

CHECKPOINT = ROOT/"assets/neural_insertion/contact_preimpact.json"
KINDS = ("segment_normal", "segment_oblique", "chain_no_elastic", "der")
# 0.1 mm units keep the 1.375 mm segment spacing exactly representable by
# MuJoCo's composite generator, unlike the previously studied cm fixture.
TENTH_MM_G = Units("tenth_mm_g_s", 10000, 1000)
UNIT_SYSTEMS = (MM_G, MM_UG, TENTH_MM_G)
_kernel = None


def kernel():
    global _kernel
    if _kernel is not None:
        return _kernel
    source = Path(__file__).with_name("contact_kernel.cc")
    key = hashlib.sha256(source.read_bytes()+mujoco.__version__.encode()+platform.platform().encode()).hexdigest()[:16]
    folder = ROOT/"build/neural_insertion/contact_kernel"/key
    folder.mkdir(parents=True, exist_ok=True)
    dest = folder/("kernel.dylib" if platform.system() == "Darwin" else "kernel.so")
    if not dest.exists():
        mjdir = Path(mujoco.__file__).parent
        libs = list(mjdir.glob("libmujoco.*.dylib" if platform.system() == "Darwin" else "libmujoco.so.*"))
        if len(libs) != 1:
            raise RuntimeError("Expected one installed MuJoCo library")
        command = ["clang++" if platform.system() == "Darwin" else "c++", "-O2", "-std=c++17", "-shared", "-fPIC",
                   "-I"+str(mjdir/"include"), str(source), str(libs[0]), "-Wl,-rpath,"+str(mjdir), "-o", str(dest)]
        subprocess.run(command, check=True)
    _kernel = ctypes.CDLL(str(dest))
    _kernel.contact_chunk.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                     np.ctypeslib.ndpointer(dtype=np.float64, shape=(15,))]
    _kernel.contact_chunk.restype = None
    return _kernel


def checkpoint():
    saved = json.loads(CHECKPOINT.read_text())
    # Rebuilds on another platform can have different binary/compiler hashes;
    # require the same implementation and patch set, not the original binary.
    current = build_provenance()
    expected = saved["plugin"]
    if any(current[k] != expected[k] for k in ("source_sha256", "upstream", "eigen_revision", "eigen_archive_sha256", "patches")):
        raise ValueError("DER plugin differs from baseline")
    state = {k: np.array(v) for k, v in saved["state"].items()}
    return saved, state


def model_xml(kind, config, units):
    if kind not in KINDS:
        raise ValueError("Unknown diagnostic fixture")
    root = ET.fromstring(der_xml(config, units))
    root.set("model", "contact_isolation_"+kind)
    if kind != "der":
        root.remove(root.find("extension"))
        for body in root.findall(".//body"):
            p = body.find("plugin")
            if p is not None:
                body.remove(p)
    if kind.startswith("segment"):
        native, _ = load_cable(config, units)
        b = native.body("thread_B_1").id
        g = native.body_geomadr[b]
        world = root.find("worldbody")
        for child in list(world):
            if child.tag != "geom" or child.get("name") != "support":
                world.remove(child)
        root.remove(root.find("contact"))
        angle = math.radians(15 if kind == "segment_oblique" else 0)
        length = config.length_m/config.segments
        center_z = config.radius_m + length*math.sin(angle)/2
        body = add(world, "body", name="segment", pos=numbers((0, 0, center_z*units.length)),
                   quat=numbers((math.cos(angle/2), 0, math.sin(angle/2), 0)))
        add(body, "freejoint", name="segment_free")
        add(body, "inertial", pos="0 0 0", mass=str(native.body_mass[b]),
            diaginertia=numbers(native.body_inertia[b]), quat=numbers(native.body_iquat[b]))
        add(body, "geom", name="segment_geom", type="capsule", size=numbers(native.geom_size[g]),
            pos=numbers(native.geom_pos[g]-native.body_ipos[b]), quat=numbers(native.geom_quat[g]),
            solref=numbers(native.geom_solref[g]), solimp=numbers(native.geom_solimp[g]),
            condim=str(native.geom_condim[g]), friction=numbers(native.geom_friction[g]), rgba="1 .67 .12 1")
        for name, x in (("end_a", -length/2), ("end_b", length/2)):
            add(body, "site", name=name, pos=numbers((x*units.length, 0, 0)), size=str(config.radius_m*units.length))
        camera(world, "cable", np.array((.003, -.011, .001))*units.length,
               np.array((.001, 0, -.003))*units.length, 48)
        root.find("statistic").set("center", numbers(np.array((0, 0, -.003))*units.length))
        root.find("statistic").set("extent", str(.012*units.length))
    return ET.tostring(root, encoding="unicode")


def load_fixture(kind, dt, units=MM_G, friction=.3):
    case, state = checkpoint()
    c = replace(CableConfig(**case["config"]), dt_s=dt, friction=friction)
    xml = model_xml(kind, c, units)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    if kind.startswith("segment"):
        if kind == "segment_oblique":
            data.qvel[0] = .05*units.length
    else:
        data.qpos[:] = state["qpos"]  # ball-joint coordinates are unit independent
        data.qvel[:] = state["qvel"]
        if kind == "der":
            data.plugin_state[:] = state["plugin_state"]
    mujoco.mj_forward(model, data)
    if data.ncon:
        raise ValueError("Diagnostic starts before contact")
    return c, model, data, xml


def centerline(kind, model, data, units):
    if kind.startswith("segment"):
        return np.array([data.site("end_a").xpos, data.site("end_b").xpos])/units.length
    return vertices(model, data, units)


def energy(kind, model, data, units):
    mv = np.zeros(model.nv)
    mujoco.mj_mulM(model, data, mv, data.qvel)
    kinetic = .5*data.qvel@mv/units.torque
    gravity = -np.sum(model.body_mass[:, None]*data.xipos*model.opt.gravity)/units.torque
    elastic = 0.
    if kind == "der":
        e = elastic_metrics(model, data, units)
        elastic = e["bending_j"]+e["twisting_j"]
    return np.array((kinetic, gravity, elastic))


def momentum(model, data, body, units):
    # Free body origin is its COM; free-joint linear velocity is in world axes.
    mass = model.body_mass[body]/units.mass
    p = mass*data.qvel[:3]/units.length
    omega = data.xmat[body].reshape(3, 3)@data.qvel[3:6]
    rotation = data.ximat[body].reshape(3, 3)
    inertia = (rotation*(model.body_inertia[body]/units.torque))@rotation.T
    h = inertia@omega+np.cross(data.xipos[body]/units.length, p)
    return p, h


def physical_signature(kind, model, data, units):
    """Compiled SI quantities: catch conversion/geometry errors before dynamics."""
    return {
        "mass_kg": model.body_mass/units.mass,
        "inertia_kg_m2": model.body_inertia/units.torque,
        "body_pos_m": model.body_pos/units.length,
        "body_ipos_m": model.body_ipos/units.length,
        "geom_pos_m": model.geom_pos/units.length,
        "geom_size_m": model.geom_size/units.length,
        "gravity_m_s2": model.opt.gravity/units.length,
        "vertices_m": centerline(kind, model, data, units),
        "solref": model.geom_solref.copy(),
        "solimp": model.geom_solimp/np.array((1, 1, units.length, 1, 1)),
        "friction": model.geom_friction/np.array((1, units.length, units.length)),
        "energy_j": energy(kind, model, data, units),
        "body_quat": model.body_quat.copy(),
        "body_iquat": model.body_iquat.copy(),
        "geom_quat": model.geom_quat.copy(),
        "armature": model.dof_armature.copy(),  # must remain zero
        "damping": model.dof_damping.copy(),  # must remain zero
    }


def unit_audit(kind, friction=.3):
    signatures = {}
    for units in UNIT_SYSTEMS:
        _, model, data, _ = load_fixture(kind, 1.5625e-7, units, friction)
        signatures[units.name] = physical_signature(kind, model, data, units)
    reference = signatures[MM_G.name]
    result = {}
    for units in UNIT_SYSTEMS[1:]:
        errors = {}
        for name, a in reference.items():
            b = signatures[units.name][name]
            scale = max(float(np.max(abs(a))), 1e-30)
            errors[name] = float(np.max(abs(a-b))/scale)
        result[units.name] = {"maximum_relative_errors": errors,
                             "passed": bool(max(errors.values()) < 1e-10)}
    return result


def trial(kind, dt, units=MM_G, friction=.3, duration=None):
    started = time.perf_counter()
    c, m, d, xml = load_fixture(kind, dt, units, friction)
    duration = (.06 if kind.startswith("segment") else .03) if duration is None else duration
    sample_dt = .0001
    n = round(duration/dt)
    every = round(sample_dt/dt)
    if n < 1 or every < 1 or not math.isclose(n*dt, duration, rel_tol=1e-10) or not math.isclose(every*dt, sample_dt, rel_tol=1e-10):
        raise ValueError("Duration and sampling must match timestep")
    support = m.geom("support").id
    body = m.body("segment").id if kind.startswith("segment") else -1
    initial_momentum = momentum(m, d, body, units) if body >= 0 else None
    summary = np.zeros(15)
    first_contact = None
    trace = {k: [] for k in ("time", "vertices", "qpos", "qvel", "plugin_state", "energy_j", "contact_work_j")}

    def sample():
        trace["time"].append(float(d.time))
        trace["vertices"].append(centerline(kind, m, d, units))
        for name in ("qpos", "qvel", "plugin_state"):
            trace[name].append(getattr(d, name).copy())
        trace["energy_j"].append(energy(kind, m, d, units))
        trace["contact_work_j"].append(summary[3]/units.torque)

    sample()
    lib = kernel()
    for start in range(0, n, every):
        chunk = np.zeros(15)
        lib.contact_chunk(m._address, d._address, min(every, n-start), support, body, chunk)
        for index in (0, 3, 4, 5, 6, 7, 8, 9, 10):
            summary[index] += chunk[index]
        for index in (1, 2, 11, 14):
            summary[index] = max(summary[index], chunk[index])
        if first_contact is None and chunk[12] >= 0:
            first_contact = float(chunk[12])
        mujoco.mj_forward(m, d)
        sample()
        if chunk[14]:
            break
    trace = {k: np.array(v) for k, v in trace.items()}
    energy_total = trace["energy_j"].sum(axis=1)
    work_residual = energy_total-energy_total[0]-trace["contact_work_j"]
    # Positive normalization: kinetic + magnitude of gravity + elastic energy at start.
    energy_scale = float(np.abs(trace["energy_j"][0]).sum())
    report = {"kind": kind, "config": asdict(c), "units": asdict(units), "duration_s": duration,
              "simulated_s": float(d.time), "completed": bool(not summary[14] and int(summary[0]) == n),
              "sample_interval_s": sample_dt, "initial_checkpoint_s": 0 if body >= 0 else .03,
              "failure": "warning_or_nonfinite" if summary[14] else None,
              "peak_penetration_um": float(summary[1]/units.length*1e6),
              "peak_normal_force_n": float(summary[2]/units.force), "first_contact_s": first_contact,
              "max_solver_iterations": int(summary[11]), "warnings": d.warning.number.tolist(),
              "contact_impulse_ns": (summary[4:7]/units.force).tolist(),
              "max_work_residual_j": float(np.max(abs(work_residual))),
              "relative_work_residual": float(np.max(abs(work_residual))/energy_scale),
              "max_energy_gain_fraction": float(max(0, np.max(energy_total-energy_total[0]))/energy_scale),
              "energy_scale_j": energy_scale, "scene_sha256": hashlib.sha256(xml.encode()).hexdigest(),
              "physical_total_mass_kg": float(m.body_mass[1:].sum()/units.mass),
              "wall_s": time.perf_counter()-started}
    if body >= 0:
        p, h = momentum(m, d, body, units)
        gravity_impulse = m.body_mass[body]/units.mass*m.opt.gravity/units.length*float(d.time)
        contact_impulse = summary[4:7]/units.force
        linear_error = p-initial_momentum[0]-gravity_impulse-contact_impulse
        angular_error = h-initial_momentum[1]-summary[7:10]/units.torque
        report["linear_impulse_residual_ns"] = linear_error.tolist()
        report["relative_linear_impulse_residual"] = float(np.linalg.norm(linear_error)/max(np.linalg.norm(gravity_impulse)+np.linalg.norm(contact_impulse), 1e-30))
        report["angular_impulse_residual_nms"] = angular_error.tolist()
    return report, trace, xml


def difference(a, b):
    if a["vertices"].shape != b["vertices"].shape or not np.allclose(a["time"], b["time"], rtol=0, atol=1e-9):
        return None
    return float(np.linalg.norm(a["vertices"]-b["vertices"], axis=-1).max()*1e6)


def run(output, kinds=KINDS, friction=.3):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    path = output/"manifest.json"
    if path.exists():
        raise FileExistsError("Preserve previous runs: use a fresh directory")
    started = time.perf_counter()
    source = Path(__file__).parent
    report = {"status": "running", "started_utc": datetime.now(timezone.utc).isoformat(),
              "mujoco": mujoco.__version__, "backend": "MuJoCo CPU", "machine": platform.machine(),
              "plugin": build_provenance(), "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in [source/"contact_diagnostics.py", source/"contact_kernel.cc", source/"der.py", source/"units.py"]},
              "checkpoint_sha256": hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest(),
              "baseline_manifest_sha256": json.loads(CHECKPOINT.read_text())["baseline_manifest_sha256"],
              "cases": {}, "comparisons": {}, "unit_audits": {}, "friction": friction,
              "gates_um": {"timestep": 1., "units": .01},
              "controls": "Segment fixtures share section/contact properties, not full-chain effective mass. Chains start from identical pre-contact qpos/qvel.",
              "timing_note": "Wall includes construction, native stepping and telemetry; not isolated compute time."}
    traces = {}
    for kind in kinds:
        report["unit_audits"][kind] = unit_audit(kind, friction)
        if not all(v["passed"] for v in report["unit_audits"][kind].values()):
            report["status"] = "compiled_unit_audit_failed"
            path.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
            raise ValueError("Unit conversion changes compiled physical parameters; see manifest")
        for units in UNIT_SYSTEMS:
            for level, dt in (("coarse", 1.5625e-7), ("fine", 7.8125e-8)):
                name = f"{kind}_{units.name}_{level}"
                result, trace, xml = trial(kind, dt, units, friction)
                traj_path = output/(name+".npz")
                np.savez_compressed(traj_path, **trace)
                (output/(name+".xml")).write_text(xml)
                result["trajectory_sha256"] = hashlib.sha256(traj_path.read_bytes()).hexdigest()
                report["cases"][name] = result
                traces[name] = trace
                path.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
                print(f"{name}: completed={result['completed']} penetration={result['peak_penetration_um']:.4g}um work_residual={result['relative_work_residual']:.3g} wall={result['wall_s']:.2f}s", flush=True)
        for units in UNIT_SYSTEMS:
            a, b = [f"{kind}_{units.name}_{level}" for level in ("coarse", "fine")]
            value = difference(traces[a], traces[b])
            report["comparisons"][f"{kind}_{units.name}_timestep"] = {"difference_um": value,
                "cases": [a, b], "gate_um": 1.,
                "passed": bool(value is not None and value < 1 and all(report["cases"][x]["completed"] and report["cases"][x]["first_contact_s"] is not None for x in (a, b)))}
        for alternative in UNIT_SYSTEMS[1:]:
            for level in ("coarse", "fine"):
                a, b = [f"{kind}_{units.name}_{level}" for units in (MM_G, alternative)]
                value = difference(traces[a], traces[b])
                report["comparisons"][f"{kind}_{alternative.name}_{level}_units"] = {"difference_um": value,
                    "cases": [a, b], "gate_um": .01,
                    "passed": bool(value is not None and value < .01 and all(report["cases"][x]["completed"] and report["cases"][x]["first_contact_s"] is not None for x in (a, b)))}
    report["success"] = all(v["passed"] for v in report["comparisons"].values())
    report["status"] = "diagnostic_comparisons_passed" if report["success"] else "diagnostic_comparisons_failed"
    report["elapsed_wall_s"] = time.perf_counter()-started
    report["simulated_s"] = sum(c["simulated_s"] for c in report["cases"].values())
    report["finished_utc"] = datetime.now(timezone.utc).isoformat()
    report["ready_for_robot_or_rl"] = False
    path.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(json.dumps(report["comparisons"], indent=2), flush=True)
    return report


def preview(output):
    from PIL import Image, ImageDraw
    from .visuals import font
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    sheet = Image.new("RGB", (1600, 1040), "#101c25")
    for i, kind in enumerate(KINDS):
        c, m, d, _ = load_fixture(kind, 1.5625e-7)
        with mujoco.Renderer(m, height=450, width=800) as r:
            r.update_scene(d, camera="cable")
            r.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
            frame = Image.fromarray(r.render())
        x, y = (i%2)*800, (i//2)*520
        sheet.paste(frame, (x, y+55))
        draw = ImageDraw.Draw(sheet)
        draw.text((x+20, y+15), kind.replace("_", " "), font=font(24), fill="#eff6f7")
    sheet.save(output/"initial.png")
    return output/"initial.png"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "preview"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--kinds", nargs="+", choices=KINDS, default=list(KINDS))
    parser.add_argument("--friction", type=float, default=.3)
    args = parser.parse_args()
    if args.command == "preview":
        print(preview(args.output))
    else:
        result = run(args.output, args.kinds, args.friction)
        raise SystemExit(0 if result["success"] else 1)


if __name__ == "__main__":
    main()
