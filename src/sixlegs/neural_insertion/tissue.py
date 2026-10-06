"""Provisional needle-tissue and thread-retention model for the end-to-end variant.

MuJoCo has no puncture model, so the phantom's interaction with the tool and
the inserted thread is an explicit, documented force model applied through
xfrc_applied and rebuilt from scratch every step. Magnitudes are illustrative
placeholders for a soft tissue until measured; they are not calibrated to any
published insertion data.

Needle (insertion body, applied at the tip):
  before puncture  axial spring k_dimple * depth, until depth reaches d_puncture
  after puncture   axial cutting force while advancing, plus Coulomb-like shaft
                   friction per inserted length opposing axial motion
  inside tissue    lateral spring and damper toward the entry point
Thread bodies that may enter tissue (the eyelet end): once below the surface
they are anchored to the tissue by a soft spring and damper with a force cap;
past the cap the anchor slips (stick-slip retention).
Engine units: mm, g, s; force unit g mm/s^2 = 1 uN.
"""

from dataclasses import dataclass, field

import mujoco
import numpy as np

from .scene import surface_z

UN = 1.0  # micronewton in engine force units


@dataclass
class TissueParams:
    k_dimple: float = 5.0e3 * UN       # uN/mm (5 N/m) before puncture
    d_puncture: float = .40            # mm of dimpling at puncture (2 mN)
    f_cut: float = 500. * UN           # uN while advancing after puncture
    f_shaft: float = 200. * UN         # uN per mm of inserted shaft, Coulomb-like
    v_smooth: float = .1               # mm/s friction smoothing velocity
    k_lateral: float = 3.0e4 * UN      # uN/mm (30 N/m) toward the entry point
    c_lateral: float = 50. * UN        # uN s/mm
    k_anchor: float = 2.0e3 * UN       # uN/mm (2 N/m) per thread body
    f_retain: float = 50. * UN         # uN per thread body before slip
    track_radius: float = 1.0          # mm from the needle entry within which bodies anchor
    tubes: bool = False                # channel variant: contacts guide and hold; only axial needle force here
    channel: tuple = (0., 0., .28)     # channel centre x, y and radius (mm) in the tubes variant


@dataclass
class TissueState:
    punctured: bool = False
    entry: np.ndarray | None = None
    anchors: dict = field(default_factory=dict)
    peak_axial: float = 0.
    peak_lateral: float = 0.
    missed: bool = False


class Tissue:
    def __init__(self, model, thread_bodies, params=TissueParams(), units_length=1000., end_site="eyelet_center",
                 anchored=3, grip=True):
        self.m, self.p, self.L = model, params, units_length
        self.needle = model.body("insertion").id
        self.tip = model.site("needle_tip").id
        self.eyelet = model.site(end_site).id
        self.last = model.body(thread_bodies[-1]).id
        self.bodies = [model.body(n).id for n in thread_bodies[-anchored:]]  # bodies that may enter tissue
        self.mass = {b: model.body_subtreemass[b] if b == self.bodies[-1] else model.body_mass[b] for b in self.bodies}
        self.state = TissueState()
        self.grip = grip  # False: the caller holds the thread (for example with constraints)
        self.offset = np.zeros(3)  # tissue displacement (breathing, pulse); the needle entry is kept in tissue frame
        self._vel = np.zeros(6)

    def surface(self, p):
        """Tissue surface height under world point p, with the tissue displaced by self.offset."""
        o = self.offset
        return surface_z((p[0]-o[0])/self.L, (p[1]-o[1])/self.L)*self.L+o[2]

    def _apply(self, data, body, point, force):
        data.xfrc_applied[body, :3] += force
        data.xfrc_applied[body, 3:] += np.cross(point-data.xipos[body], force)

    def step(self, data):
        """Rebuild the tissue forces from the current state."""
        p, s = self.p, self.state
        data.xfrc_applied[self.needle] = 0
        for b in self.bodies:
            data.xfrc_applied[b] = 0
        tip_world = data.site_xpos[self.tip]
        tip = tip_world-self.offset  # tissue frame
        depth = self.surface(tip_world)-tip_world[2]
        mujoco.mj_objectVelocity(self.m, data, mujoco.mjtObj.mjOBJ_SITE, self.tip, self._vel, 0)
        v = self._vel[3:].copy()
        axial = 0.
        if p.tubes:
            # Pre-formed channel: walls and skin are real contacts. Only cutting and
            # shaft friction act along the needle while its tip is below the surface.
            if depth > 0:
                cx, cy, radius = p.channel
                if np.hypot(tip[0]-cx, tip[1]-cy) > radius:
                    s.missed = True
                s.punctured = True
                friction = p.f_shaft*depth*np.tanh(-v[2]/p.v_smooth)
                axial = (p.f_cut if v[2] < 0 else 0.)*np.tanh(-v[2]/p.v_smooth)+friction
                self._apply(data, self.needle, tip_world, np.array((0, 0, axial)))
                s.peak_axial = max(s.peak_axial, abs(axial))
            return depth
        if depth > 0:
            if s.entry is None:
                s.entry = tip.copy()
            if not s.punctured and depth >= p.d_puncture:
                s.punctured = True
            if not s.punctured:
                axial = p.k_dimple*depth
            else:
                friction = p.f_shaft*depth*np.tanh(-v[2]/p.v_smooth)
                cutting = p.f_cut if v[2] < 0 else 0.
                axial = cutting*np.tanh(-v[2]/p.v_smooth)+friction
            lateral = np.zeros(3)
            lateral[:2] = -p.k_lateral*(tip[:2]-s.entry[:2])-p.c_lateral*v[:2]
            force = lateral+np.array((0, 0, axial))
            self._apply(data, self.needle, tip_world, force)
            s.peak_axial = max(s.peak_axial, abs(axial))
            s.peak_lateral = max(s.peak_lateral, float(np.linalg.norm(lateral)))
        elif s.entry is not None and not s.punctured:
            s.entry = None  # dimple released without puncture
        if not self.grip:
            return depth
        # Thread retention: bodies below the surface near the needle track.
        for b in self.bodies:
            point = data.site_xpos[self.eyelet] if b == self.last else data.xipos[b]
            below = self.surface(point)-point[2]
            near = s.entry is not None and np.linalg.norm(point[:2]-s.entry[:2]) < p.track_radius
            if below <= 0 or not near:
                s.anchors.pop(b, None)
                continue
            anchor = s.anchors.setdefault(b, point.copy())
            pull = p.k_anchor*(anchor-point)
            size = np.linalg.norm(pull)
            if size > p.f_retain:  # slip: drag the anchor along so the force stays at the cap
                anchor[:] = point+(anchor-point)*p.f_retain/size
                pull *= p.f_retain/size
            mujoco.mj_objectVelocity(self.m, data, mujoco.mjtObj.mjOBJ_BODY, b, self._vel, 0)
            c = 2*np.sqrt(p.k_anchor*self.mass[b])
            self._apply(data, b, point, pull-c*self._vel[3:])
        return depth
