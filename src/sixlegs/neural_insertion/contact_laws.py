"""Named contact regularizations for the thread fixtures. SI parameters.

MuJoCo 3.12 (pinned engine_core_constraint.c) computes, per contact normal row,
  a_ref = -B*v - K*d(r)*pos,  B = 2/(d_width*tau),  K = 1/(d_width*tau)^2,
  R     = (1-d(r))/d(r) * diagA,
where d(r) ramps from d0 to d_width over `width` with a midpoint/power spline.
The baseline fixture uses d0 == d_width, so the damping term B*v is applied in
full at the first penetrating step: the force is discontinuous at entry. A
ramp with d0 << d_width makes the force continuous in state at entry. Both are
numerical regularizations, not calibrated thread/tissue laws.
"""

from dataclasses import dataclass
import math

import mujoco


@dataclass(frozen=True)
class ContactLaw:
    name: str
    d0: float = .9999
    d_width: float = .9999
    width_m: float = 1e-6
    midpoint: float = .5
    power: float = 2.
    time_constant_s: float = 1e-5
    damping_ratio: float = 1.

    def __post_init__(self):
        if not (mujoco.mjMINIMP <= self.d0 <= mujoco.mjMAXIMP and mujoco.mjMINIMP <= self.d_width <= mujoco.mjMAXIMP):
            raise ValueError("Impedances must lie within MuJoCo's clamped range")
        if not all(math.isfinite(x) and x > 0 for x in (self.width_m, self.time_constant_s, self.damping_ratio)):
            raise ValueError("Width, time constant and damping ratio must be positive")
        if not 0 < self.midpoint < 1 or self.power < 1:
            raise ValueError("Midpoint must lie in (0,1) and power must be at least 1")

    @property
    def continuous_entry(self):
        return self.d0 < self.d_width

    def impedance(self, penetration_m):
        """d(r): the pinned getimpedance spline, in SI penetration."""
        if self.d0 == self.d_width or self.width_m <= mujoco.mjMINVAL:
            return .5*(self.d0+self.d_width)
        x = abs(penetration_m)/self.width_m
        if x >= 1:
            return self.d_width
        if x <= 0:
            return self.d0
        if self.power == 1:
            y = x
        elif x <= self.midpoint:
            y = x**self.power/self.midpoint**(self.power-1)
        else:
            y = 1-(1-x)**self.power/(1-self.midpoint)**(self.power-1)
        return self.d0+y*(self.d_width-self.d0)

    def solimp(self, units):
        return (self.d0, self.d_width, self.width_m*units.length, self.midpoint, self.power)

    def solref(self):
        return (self.time_constant_s, self.damping_ratio)


# `flat` is the recorded baseline of the isolation/audit studies. The ramps
# start at MuJoCo's minimum impedance and reach the baseline impedance at
# `width`; a shorter time constant keeps peak penetration under the 2 µm gate.
LAWS = {
    "flat": ContactLaw("flat"),
    "ramp_10us": ContactLaw("ramp_10us", d0=mujoco.mjMINIMP, power=1., time_constant_s=1e-5),
    "ramp_5us": ContactLaw("ramp_5us", d0=mujoco.mjMINIMP, power=1., time_constant_s=5e-6),
    "ramp_2um_5us": ContactLaw("ramp_2um_5us", d0=mujoco.mjMINIMP, width_m=2e-6, power=1., time_constant_s=5e-6),
}
INTEGRATORS = ("implicitfast", "RK4")
