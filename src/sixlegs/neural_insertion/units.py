"""Explicit SI-to-engine conversion. Time and angles remain seconds/radians."""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class Units:
    name: str
    length: float = 1.0  # engine length units per metre
    mass: float = 1.0  # engine mass units per kilogram

    def __post_init__(self):
        if not all(math.isfinite(x) and x > 0 for x in (self.length, self.mass)):
            raise ValueError("Unit conversion factors must be finite and positive")

    @property
    def force(self):
        return self.mass * self.length

    @property
    def torque(self):
        return self.mass * self.length**2

    @property
    def density(self):
        return self.mass / self.length**3

    @property
    def modulus(self):
        return self.mass / self.length

    @property
    def bending(self):
        return self.mass * self.length**3


SI = Units("m_kg_s")
MM_KG = Units("mm_kg_s", 1000, 1)
MM_G = Units("mm_g_s", 1000, 1000)
CM_G = Units("cm_g_s", 100, 1000)
MM_MG = Units("mm_mg_s", 1000, 1e6)
MM_UG = Units("mm_ug_s", 1000, 1e9)
SYSTEMS = {u.name: u for u in (SI, MM_KG, MM_G, CM_G, MM_MG, MM_UG)}
