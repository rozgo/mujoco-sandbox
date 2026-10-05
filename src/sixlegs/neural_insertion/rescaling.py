"""Microscale cable benchmarks. Every public parameter/result uses SI units."""

from dataclasses import asdict, dataclass
from pathlib import Path
import hashlib
import json
import math
import time
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from PIL import Image, ImageDraw

from .scene import ROOT, add, vec, camera
from .units import Units, SI, MM_G
from .visuals import font


@dataclass(frozen=True)
class CableConfig:
    """Illustrative circular rod, not calibrated thread material.

    Defaults are for static inspection/bending; contact evaluation explicitly
    supplies much smaller timesteps and contact time constants.
    """
    length_m: float = .044
    radius_m: float = 20e-6
    density_kg_m3: float = 1400
    young_pa: float = 1e8
    poisson: float = .45
    segments: int = 32
    dt_s: float = .0001
    relaxation_s: float = .002
    gravity_m_s2: float = 9.81
    floor_z_m: float = -.006
    contact_time_s: float = .0006
    contact_impedance: float = .9999
    integrator: str = "implicitfast"
    friction: float = .3

    def __post_init__(self):
        positive = (self.length_m, self.radius_m, self.density_kg_m3,
                    self.young_pa, self.dt_s, self.contact_time_s)
        if not all(math.isfinite(x) and x > 0 for x in positive):
            raise ValueError("Dimensions, density, modulus and time scales must be positive")
        if isinstance(self.segments, bool) or not isinstance(self.segments, int) or self.segments < 2:
            raise ValueError("segments must be an integer >= 2")
        if not -1 < self.poisson < .5:
            raise ValueError("Isotropic Poisson ratio must lie between -1 and 0.5")
        if not all(math.isfinite(x) and x >= 0 for x in (self.relaxation_s, self.friction)):
            raise ValueError("Damping and friction must be finite and nonnegative")
        if not 0 < self.contact_impedance < 1:
            raise ValueError("Contact impedance must lie between zero and one")
        if not all(math.isfinite(x) for x in (self.gravity_m_s2, self.floor_z_m)):
            raise ValueError("Gravity and support position must be finite")
        if self.integrator not in ("implicitfast", "implicit", "RK4"):
            raise ValueError("Unsupported benchmark integrator")

    @property
    def ei(self):
        return self.young_pa * math.pi * self.radius_m**4 / 4

    @property
    def mass_kg(self):
        return self.density_kg_m3 * math.pi * self.radius_m**2 * self.length_m


def cable_xml(config=CableConfig(), units=MM_G):
    c, u = config, units
    root = ET.Element("mujoco", model="microscale_cable_" + u.name)
    add(root, "compiler", angle="radian", autolimits="true")
    add(root, "option", timestep=c.dt_s, integrator=c.integrator, gravity=vec((0, 0, -c.gravity_m_s2*u.length)),
        solver="Newton", tolerance="1e-12", iterations="100", cone="elliptic", jacobian="sparse")
    add(add(root, "extension"), "plugin", plugin="mujoco.elasticity.cable")
    default = add(root, "default")
    add(default, "geom", density=c.density_kg_m3*u.density, condim="3", friction=vec((c.friction, .000001*u.length, .0000001*u.length)),
        solref=vec((c.contact_time_s, 1)), solimp=vec((c.contact_impedance, c.contact_impedance, 1e-6*u.length, .5, 2)))
    visual = add(root, "visual")
    add(visual, "global", offwidth="1600", offheight="1000")
    add(visual, "map", znear=".0001", zfar="10")
    add(visual, "headlight", ambient=".4 .4 .4", diffuse=".6 .6 .6")
    world = add(root, "worldbody")
    add(world, "geom", name="support", type="plane", pos=vec((0,0,c.floor_z_m*u.length)),
        size=vec((.08*u.length,.04*u.length,.001*u.length)), rgba=".055 .11 .15 1")
    add(world, "geom", name="clamp", type="box", size=vec((.0007*u.length,.001*u.length,.0005*u.length)),
        pos=vec((-.0007*u.length,0,0)), rgba=".45 .58 .62 1")
    comp = add(world, "composite", prefix="thread_", type="cable", curve="s", count=f"{c.segments+1} 1 1",
        size=c.length_m*u.length, initial="none")
    plugin = add(comp, "plugin", plugin="mujoco.elasticity.cable")
    add(plugin, "config", key="bend", value=c.young_pa*u.modulus)
    add(plugin, "config", key="twist", value=c.young_pa/(2*(1+c.poisson))*u.modulus)
    add(plugin, "config", key="vmax", value="0")
    ds = c.length_m/c.segments
    # Kelvin-type rotational damping: c_theta = tau * EI / ds. No added inertia.
    add(comp, "joint", kind="main", damping=c.relaxation_s*c.ei/ds*u.torque)
    # Explicit mass prevents overlapping capsule endcaps adding material mass.
    # Composite geoms carry their own defaults: set contacts explicitly here.
    add(comp, "geom", type="capsule", size=c.radius_m*u.length,
        mass=c.mass_kg/c.segments*u.mass, rgba="1 .67 .12 1", condim="3",
        friction=vec((c.friction, .000001*u.length, .0000001*u.length)),
        solref=vec((c.contact_time_s, 1)), solimp=vec((c.contact_impedance, c.contact_impedance, 1e-6*u.length, .5, 2)))
    camera(world, "cable", np.array((.5*c.length_m,-.9*c.length_m,.3*c.length_m))*u.length,
           np.array((.5*c.length_m,0,-.002))*u.length, 43)
    add(root, "statistic", center=vec((c.length_m*.5*u.length,0,0)), extent=c.length_m*1.5*u.length)
    return ET.tostring(root, encoding="unicode")


def load_cable(config=CableConfig(), units=MM_G):
    model = mujoco.MjModel.from_xml_string(cable_xml(config, units))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


def initial_preview(output=ROOT / "previews/neural_insertion/rescaling/initial.png"):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    m,d = load_cable()
    with mujoco.Renderer(m, height=800, width=1600) as r:
        r.update_scene(d, camera="cable")
        raw = Image.fromarray(r.render())
    frame = Image.new("RGB", (1600, 930), "#101c25")
    frame.paste(raw, (0,80))
    draw = ImageDraw.Draw(frame)
    draw.text((25,20), "FLEXIBLE THREAD / UNIT-SCALING BENCH", font=font(28), fill="#eff6f7")
    draw.text((25,895), "44 mm long | 40 micrometre diameter | actual geometry | initial static pose", font=font(21), fill="#6ed8c6")
    frame.save(output)
    return output


def trial(config=CableConfig(), units=MM_G, duration=1., force_n=(0,0,0), torque_nm=(0,0,0), release_at=None, sample_interval_s=.002):
    """Run one fixed case. Numerical completion is separate from quality acceptance.

    Coordinates in trajectory vertices/tip use SI; qpos/qvel use engine units.
    The final-quarter penetration window does not establish that motion settled.
    """
    if not all(math.isfinite(x) and x > 0 for x in (duration, sample_interval_s)):
        raise ValueError("Duration and sampling interval must be finite and positive")
    steps = round(duration / config.dt_s)
    if steps < 1 or not math.isclose(steps * config.dt_s, duration, rel_tol=1e-10):
        raise ValueError("Duration must be an integer number of physics steps")
    if release_at is not None and not 0 <= release_at <= duration:
        raise ValueError("Release time must lie within the trial")
    for load in (force_n, torque_nm):
        if np.shape(load) != (3,) or not np.isfinite(load).all():
            raise ValueError("Loads must be finite world-frame three-vectors")
    if config.integrator == "RK4" and (any(force_n) or any(torque_nm)):
        raise ValueError("RK4 benchmark is passive only; applied loads need substage callbacks")
    started=time.perf_counter()
    report={"units":asdict(units),"config":asdict(config),"requested_duration_s":duration,
            "force_n":list(force_n),"torque_nm":list(torque_nm),"release_at_s":release_at,"sample_interval_s":sample_interval_s,
            "scene_sha256":hashlib.sha256(cable_xml(config,units).encode()).hexdigest()}
    try:
        m,d=load_cable(config,units)
    except ValueError as exc:
        return report | {"compiled":False,"error":str(exc),"simulated_s":0,"wall_s":time.perf_counter()-started},None
    endpoint=m.site("thread_S_last").id
    body=m.body("thread_B_last").id
    ids=np.array([i for i in range(m.nbody) if m.body(i).name.startswith("thread_B")])
    times=[];ends=[];bodies=[];positions=[];velocities=[];max_pen=0.;max_pen_settled=0.;max_stretch=0.;loads=[];failed=None
    support_id=m.geom("support").id
    max_iterations=0
    f=np.asarray(force_n)*units.force;tau=np.asarray(torque_nm)*units.torque
    every=max(1,round(sample_interval_s/config.dt_s))
    for step in range(steps):
        mujoco.mj_step1(m,d)
        d.qfrc_applied[:]=0
        active=release_at is None or d.time < release_at
        if active:
            mujoco.mj_applyFT(m,d,f,tau,d.site_xpos[endpoint],body,d.qfrc_applied)
        # Contacts below correspond to this substep's position, before integration.
        for contact in d.contact:
            if contact.geom1 == support_id or contact.geom2 == support_id:
                pen=max(0.,-contact.dist/units.length)
                max_pen=max(max_pen,pen)
                if d.time > duration*.75:max_pen_settled=max(max_pen_settled,pen)
        if config.integrator == "RK4":
            mujoco.mj_step(m,d)
        else:
            mujoco.mj_step2(m,d)
        max_iterations=max(max_iterations,int(np.max(d.solver_niter)))
        if np.any(d.warning.number) or not np.isfinite(d.qpos).all() or not np.isfinite(d.qvel).all():
            failed="numerical_warning_or_nonfinite";break
        if (step+1)%every == 0 or step + 1 == steps:
            mujoco.mj_forward(m,d)
            times.append(d.time);ends.append(d.site_xpos[endpoint].copy()/units.length)
            bodies.append(np.vstack((d.xpos[ids],d.site_xpos[endpoint]))/units.length)
            positions.append(d.qpos.copy());velocities.append(d.qvel.copy())
            lengths=np.linalg.norm(np.diff(bodies[-1],axis=0),axis=1)
            max_stretch=max(max_stretch,float(np.max(np.abs(lengths-config.length_m/config.segments))))
            loads.append(active)
    mujoco.mj_forward(m,d)
    report.update(compiled=True, failure=failed, simulated_s=float(d.time),wall_s=time.perf_counter()-started,
                  min_principal_inertia_engine=float(m.body_inertia[ids].min()), total_cable_mass_kg=float(m.body_mass[ids].sum()/units.mass),
                  peak_support_penetration_um=max_pen*1e6,
                  final_quarter_support_penetration_um=max_pen_settled*1e6,
                  max_segment_length_error_m=max_stretch, final_tip_m=(d.site_xpos[endpoint]/units.length).tolist(),
                  final_tip_quat=d.xquat[body].tolist(),warnings=d.warning.number.tolist())
    report["max_solver_iterations"] = max_iterations
    report["completed"] = failed is None and math.isclose(d.time, duration, rel_tol=1e-8)
    return report, {"time":np.array(times),"tip":np.array(ends),"vertices":np.array(bodies),
                    "qpos":np.array(positions),"qvel":np.array(velocities),"load_active":np.array(loads)}
