"""Disturbance layer for the MuJoCo cycles, mirroring the alignment core (native/surgical_core.h, scale v2).

Engine units (mm, g, s; forces in µN). Illustrative magnitudes until replaced by measured hardware and
tissue data. Level 1 is the nominal quality workcell, level 2 the stress setting (twice nominal); level 0 is
undisturbed. Three groups, each scalable for ablations:

  robot    correlated force noise (0.05 s) and extra Coulomb friction on each slide; table vibration
           (three 5-60 Hz components per axis) as inertial forces through the mass each slide carries
  sensing  needle-tip measurement noise and latency; target estimate noise, per-run bias and slow drift
  tissue   breathing (0.2-0.4 Hz) and pulse (1-2 Hz) motion of the tissue

Parameters come from their own random stream, so level 0 and the physics stay reproducible.
"""

from dataclasses import dataclass

import numpy as np

FORCE_TAU = .05
FORCE_SIGMA = {"stage_x": 1e4, "stage_y": 1e4, "stage_z": 5e3, "insertion": 1e3}    # µN (0.010 ... 0.001 N)
FRICTION_MAX = {"stage_x": 2e4, "stage_y": 2e4, "stage_z": 1e4, "insertion": 1e3}   # µN
VIB_AMP, VIB_F = 1.0, (5., 60.)          # mm/s² per component and axis (0.001 m/s²)
TIP_NOISE, VEL_NOISE = .5e-3, .05        # mm, mm/s
LATENCY_MAX = 10e-3                      # s at level 1
GOAL_NOISE, GOAL_BIAS, GOAL_DRIFT = 3e-3, 1e-3, 1e-3   # mm, mm, mm/sqrt(s)
BREATH = (10e-3, 30e-3)                  # mm lateral, vertical
PULSE = (3e-3, 10e-3)


@dataclass
class Groups:
    robot: float = 1.
    sensing: float = 1.
    tissue: float = 1.


class Disturbance:
    def __init__(self, level=0., seed=0, groups=Groups()):
        self.level, self.seed = float(level), seed
        r = np.random.default_rng(seed)
        self.rng = np.random.default_rng(seed+1_000_003)  # per-step noise stream
        self.robot, self.sensing, self.tissue = (level*groups.robot, level*groups.sensing, level*groups.tissue)
        names = list(FORCE_SIGMA)
        self.force = {n: FORCE_SIGMA[n]*self.robot*r.normal() for n in names}
        self.friction = {n: FRICTION_MAX[n]*self.robot*r.uniform() for n in names}
        self.vib = [[(VIB_AMP*self.robot*r.uniform(), r.uniform(*VIB_F), r.uniform(0, 2*np.pi)) for _ in range(3)]
                    for _ in range(3)]
        self.motion = []
        for (lat, vert), (f0, f1) in ((BREATH, (.2, .4)), (PULSE, (1., 2.))):
            heading, lateral = r.uniform(0, 2*np.pi), lat*self.tissue*r.uniform()
            amp = np.array((lateral*np.cos(heading), lateral*np.sin(heading), vert*self.tissue*r.uniform()))
            self.motion.append((amp, r.uniform(f0, f1), r.uniform(0, 2*np.pi)))
        self.goal_bias = GOAL_BIAS*self.sensing*r.normal(size=3)
        self.goal_drift = np.zeros(3)
        self.latency = float(np.floor(r.uniform(0, LATENCY_MAX*1e3*self.sensing+1)))*1e-3  # s, whole ms
        self.history = []  # (time, tip) for latency

    def summary(self):
        return {"level": self.level, "seed": self.seed, "robot": self.robot, "sensing": self.sensing,
                "tissue": self.tissue, "latency_ms": self.latency*1e3,
                "breathing_amp_um": (self.motion[0][0]*1e3).round(2).tolist(), "breathing_hz": self.motion[0][1],
                "pulse_amp_um": (self.motion[1][0]*1e3).round(2).tolist(), "pulse_hz": self.motion[1][1],
                "target_bias_um": (self.goal_bias*1e3).round(2).tolist(),
                "friction_uN": {k: round(v, 1) for k, v in self.friction.items()}}

    # ------------------------------------------------------------ tissue
    def tissue_offset(self, t):
        return sum((amp*np.sin(2*np.pi*f*t+ph) for amp, f, ph in self.motion), np.zeros(3))

    # ------------------------------------------------------------ robot
    def robot_forces(self, dt, t, qvel, carried_mass, axes):
        """Generalized forces (µN) per slide: force noise, extra friction, table vibration through the mass."""
        a = np.zeros(3)
        if self.level > 0:
            for k in range(3):
                a[k] = sum(amp*np.sin(2*np.pi*f*t+ph) for amp, f, ph in self.vib[k])
        out = {}
        for n in FORCE_SIGMA:
            if self.level <= 0:
                out[n] = 0.
                continue
            sigma = FORCE_SIGMA[n]*self.robot
            self.force[n] += -self.force[n]*dt/FORCE_TAU+sigma*np.sqrt(2*dt/FORCE_TAU)*self.rng.normal()
            friction = -self.friction[n]*np.tanh(qvel[n]/1e-1)  # 0.1 mm/s smoothing (1e-4 m/s in the core)
            out[n] = self.force[n]+friction-carried_mass[n]*float(a@axes[n])
        return out

    # ------------------------------------------------------------ sensing
    def record_tip(self, t, tip):
        self.history.append((t, tip.copy()))
        while len(self.history) > 2 and self.history[1][0] <= t-self.latency-1e-12:
            self.history.pop(0)

    def measure_tip(self, t):
        """Needle tip as the sensor reports it: delayed by the latency, with noise."""
        tip = self.history[0][1].copy() if self.history else np.zeros(3)
        for tt, p in self.history:
            if tt <= t-self.latency+1e-12:
                tip = p.copy()
        if self.level > 0:
            tip += TIP_NOISE*self.sensing*self.rng.normal(size=3)
        return tip

    def measure_target(self, true_target, dt_since_last):
        """Target estimate: bias, slow drift (random walk) and per-measurement noise."""
        if self.level <= 0:
            return true_target.copy()
        self.goal_drift += GOAL_DRIFT*self.sensing*np.sqrt(max(dt_since_last, 0.))*self.rng.normal(size=3)
        return true_target+self.goal_bias+self.goal_drift+GOAL_NOISE*self.sensing*self.rng.normal(size=3)
