"""Separate, reproducible difficulty presets; the original remains the default."""

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Profile:
    name: str = "standard"
    wind_scale: float = 1.0
    cruise_seconds: float = 24.0

    @property
    def cruise_end(self):
        return 6.0 + self.cruise_seconds

    @property
    def lower_start(self):
        return self.cruise_end + 2.0

    @property
    def duration(self):
        return self.lower_start + 11.0

    def metadata(self):
        return asdict(self)


STANDARD = Profile()
AGGRESSIVE = Profile("aggressive", wind_scale=1.5, cruise_seconds=12.0)
PROFILES = {p.name: p for p in (STANDARD, AGGRESSIVE)}
