"""Named thread material presets. SI. Geometry (40 µm circular, 44 mm) is fixed.

Neither preset is a measured thread. `illustrative_100mpa` is the compliant
value used by every earlier study. `polyimide_2p5gpa` uses typical polyimide
film properties (DuPont Kapton HN data sheet: tensile modulus 2.5 GPa at 23 °C,
density 1.42 g/cm³, Poisson ratio 0.34) as a stand-in for a neural-probe
polymer until the actual thread is specified and measured.
"""

MATERIALS = {
    "illustrative_100mpa": {"young_pa": 1e8, "poisson": .45, "density_kg_m3": 1400.},
    "polyimide_2p5gpa": {"young_pa": 2.5e9, "poisson": .34, "density_kg_m3": 1420.},
}
DEFAULT_MATERIAL = "illustrative_100mpa"
