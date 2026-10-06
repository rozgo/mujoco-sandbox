"""Task-regime thread physics: materials, clock study and probe fixtures."""

import mujoco
import numpy as np
import pytest

from sixlegs.neural_insertion.contact_smoothing import load_variant
from sixlegs.neural_insertion.contact_laws import LAWS
from sixlegs.neural_insertion.materials import MATERIALS
from sixlegs.neural_insertion.task_fixtures import fixture_config, fixture_xml, law_for, trial
from sixlegs.neural_insertion.thread_clock import FRACTIONS, TIME_CONSTANTS, specs
from sixlegs.neural_insertion.units import MM_G


def test_material_presets_change_only_material_and_need_a_rest_start():
    base, _, _, _ = load_variant("der_settle", 1e-6, MM_G, LAWS["ramp_5us"], "RK4")
    stiff, m, _, _ = load_variant("der_settle", 1e-6, MM_G, LAWS["ramp_5us"], "RK4", material="polyimide_2p5gpa")
    assert stiff.young_pa == 2.5e9 and stiff.radius_m == base.radius_m and stiff.length_m == base.length_m
    assert m.body_mass[1:].sum()/MM_G.mass == pytest.approx(stiff.mass_kg, rel=1e-12)
    with pytest.raises(ValueError):
        load_variant("der", 1e-6, MM_G, LAWS["ramp_5us"], "RK4", material="polyimide_2p5gpa")


def test_clock_ladder_respects_refsafe_and_covers_both_materials():
    work = specs()
    assert {w[0] for w in work} == set(MATERIALS)
    assert min(TIME_CONSTANTS[0]/f for f in FRACTIONS) > 0 and max(FRACTIONS) >= 16
    assert all(tau/fraction <= tau/2 for _, _, tau, fraction, _ in work)  # timestep <= tau/2


def test_probe_fixtures_start_in_intended_contact_and_exclude_the_support():
    law = law_for(2e-5)
    for name, contacts in (("drag", True), ("press", True), ("release", False)):
        c = fixture_config(name, 1e-5, "polyimide_2p5gpa", law)
        m = mujoco.MjModel.from_xml_string(fixture_xml(name, c, MM_G, law, "RK4"))
        d = mujoco.MjData(m)
        mujoco.mj_forward(m, d)
        probe, support = m.geom("probe_geom").id, m.geom("support").id
        assert (d.ncon > 0) == contacts
        assert all(-k.dist < 1e-9*MM_G.length for k in d.contact[:d.ncon])  # touching, not penetrating
        assert not any({probe, support} == {k.geom1, k.geom2} for k in d.contact[:d.ncon])
        assert m.body_gravcomp[m.body("probe").id] == 1


def test_press_transmits_the_applied_load_to_the_support():
    report, trace, _ = trial("press", 1e-5, MM_G, "polyimide_2p5gpa", 2e-5)
    assert report["completed"] and report["peak_penetration_um"] < 2
    assert report["press"]["relative_balance_error"] < .01
    assert np.isfinite(trace["vertices"]).all()
